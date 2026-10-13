"""Press coverage via Google News RSS search (fallback while GDELT throttles the IP).

Each (edition, query) is searched month by month from 2000 with the after:/before: operators; a
window that hits the 100-item cap is split in half down to a single day. Gives title,
outlet and the day of publication. Resumable: finished windows are checkpointed.
"""
import html
import re
import time
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote, urlparse

from ..common import make_id, new_record, norm_title, raw_path, read_jsonl, session, write_jsonl

RSS = "https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={gl}:{lang}"
# (hl, gl, lang, query)
SEARCHES = [
    ("en-US", "US", "en", '"central bank digital currency" OR "central bank digital currencies"'),
    ("en-US", "US", "en", "CBDC"),
    ("en-US", "US", "en", '"digital euro"'),
    ("en-US", "US", "en", '"e-CNY" OR "digital yuan" OR "digital renminbi"'),
    ("en-GB", "GB", "en", '"digital pound" OR "Britcoin"'),
    ("en-IN", "IN", "en", '"digital rupee" OR "e-rupee" OR "e₹"'),
    ("en-NG", "NG", "en", 'eNaira'),
    ("en-US", "US", "en", '"e-krona" OR mBridge OR "Project Agora" OR "wholesale CBDC" OR "digital dollar CBDC"'),
    ("fr", "FR", "fr", '"euro numérique" OR "monnaie numérique de banque centrale"'),
    ("de", "DE", "de", '"digitaler Euro" OR "digitales Zentralbankgeld"'),
    ("es-419", "MX", "es-419", '"moneda digital de banco central" OR "euro digital"'),
    ("pt-BR", "BR", "pt-419", 'Drex OR "real digital" OR "moeda digital de banco central"'),
    ("it", "IT", "it", '"euro digitale"'),
    ("ru", "RU", "ru", '"цифровой рубль"'),
    ("ja", "JP", "ja", '"中央銀行デジタル通貨" OR "デジタル円"'),
    ("zh-TW", "TW", "zh-Hant", '"數位人民幣" OR "數字人民幣" OR "央行數位貨幣"'),
    # same core query seen from other English-language editions (each edition surfaces its own outlets)
    *[(f"en-{gl}", gl, "en", '"central bank digital currency" OR CBDC')
      for gl in ("GB", "IN", "AU", "CA", "SG", "ZA", "NG", "KE", "PH", "MY", "PK", "IE")],
    # older / alternative names and broader variants
    ("en-US", "US", "en", '"digital fiat currency" OR "central bank cryptocurrency" OR Fedcoin OR RSCoin '
                          'OR "sovereign digital currency" OR "national digital currency" OR "digital legal tender"'),
    ("en-US", "US", "en", '"digital dollar" "Federal Reserve"'),
    ("en-US", "US", "en", '"tokenized deposits" OR "tokenised deposits" OR "unified ledger" OR "Project Agora"'),
    ("en-US", "US", "en", '"Project Helvetia" OR "Project Dunbar" OR "Project Icebreaker" OR "Project Hamilton" '
                          'OR "Project Ubin" OR "Project Jasper" OR "Project Stella" OR "Project mBridge"'),
    ("en-US", "US", "en", '"e-HKD" OR "Sand Dollar" OR DCash OR "JAM-DEX" OR Bakong OR "e-Cedi" OR "digital shekel"'),
    ("fr", "CA", "fr", '"monnaie numérique de banque centrale" OR "dollar numérique"'),
    ("es-419", "AR", "es-419", '"moneda digital" "banco central"'),
    ("es", "ES", "es", '"euro digital" OR "moneda digital de banco central"'),
    ("pt-PT", "PT", "pt-150", '"euro digital" OR "moeda digital do banco central"'),
    ("nl", "NL", "nl", '"digitale euro"'),
    ("de", "AT", "de", '"digitaler Euro" OR "digitale Euro"'),
    ("de", "CH", "de", '"digitaler Franken" OR "digitales Zentralbankgeld" OR "Helvetia"'),
    ("sv", "SE", "sv", '"e-krona" OR "e-kronan"'),
    ("no", "NO", "no", '"digitale sentralbankpenger" OR "digital krone"'),
    ("pl", "PL", "pl", '"cyfrowe euro" OR "cyfrowy złoty" OR "cyfrowa waluta banku centralnego"'),
    ("uk", "UA", "uk", '"цифрова гривня" OR "е-гривня" OR "цифрова валюта центрального банку"'),
    ("ru", "RU", "ru", '"цифровая валюта центрального банка" OR ЦВЦБ'),
    ("tr", "TR", "tr", '"dijital Türk lirası" OR "dijital lira" OR "merkez bankası dijital para"'),
    ("ar", "AE", "ar", '"العملة الرقمية للبنك المركزي" OR "الدرهم الرقمي"'),
    ("ar", "SA", "ar", '"العملة الرقمية" "البنك المركزي"'),
    ("id", "ID", "id", '"rupiah digital"'),
    ("ko", "KR", "ko", '"중앙은행 디지털화폐" OR "CBDC"'),
    ("ja", "JP", "ja", 'CBDC'),
    ("zh-HK", "HK", "zh-Hant", '"數字港元" OR "e-HKD" OR "數碼港元"'),
    ("hi", "IN", "hi", '"डिजिटल रुपया" OR "ई-रुपी"'),
    ("vi", "VN", "vi", '"tiền kỹ thuật số" "ngân hàng trung ương"'),
    ("th", "TH", "th", '"สกุลเงินดิจิทัลของธนาคารกลาง"'),
]
CAP, MIN_DAYS = 95, 1
S = session(retries=3)
_last = [0.0]


def _fetch(hl, gl, lang, q, a, b):
    wait = 1.2 - (time.monotonic() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.monotonic()
    query = f"{q} after:{a:%Y-%m-%d} before:{b:%Y-%m-%d}"
    r = S.get(RSS.format(q=quote(query), hl=hl, gl=gl, lang=lang), timeout=60)
    r.raise_for_status()
    return re.findall(r"<item>(.*?)</item>", r.text, re.S)


def _tag(item, tag):
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", item, re.S)
    return html.unescape(m.group(1)).strip() if m else ""


def to_record(item, lang):
    outlet = _tag(item, "source")
    m = re.search(r'<source url="([^"]+)"', item)
    domain = urlparse(m.group(1)).netloc.removeprefix("www.") if m else ""
    title = _tag(item, "title")
    if outlet and title.endswith(" - " + outlet):
        title = title[: -len(outlet) - 3]
    d = parsedate_to_datetime(_tag(item, "pubDate")).date().isoformat() if _tag(item, "pubDate") else ""
    link = _tag(item, "link")
    return new_record(
        record_id=make_id("gn", f"{norm_title(title)}|{domain}"), source_db="Google News",
        source_type="press", document_type="news article", title=title, issuer=outlet or domain,
        issuer_type="news outlet", venue=domain, publication_date=d,
        date_precision="day" if d else "", year=d[:4], language=lang.split("-")[0],
        language_origin="Google News edition", url=link, native_id=_tag(item, "guid") or link,
    )


def _windows(start, end):
    """Yearly windows before 2015 (the archive is sparse there), monthly after."""
    a = start
    while a < end:
        if a.year < 2015:
            b = min(date(a.year + 1, 1, 1), end)
        else:
            b = min((a.replace(day=1) + timedelta(days=32)).replace(day=1), end)
        yield a, b
        a = b


def _days(key):
    return (date.fromisoformat(key[10:]) - date.fromisoformat(key[:10])).days


def harvest(start="2000-01-01"):
    out_p, ck_p = raw_path("gnews"), raw_path("gnews_done")
    seen = {r["record_id"] for r in read_jsonl(out_p)}
    # windows from older runs (no "n", split stopped at 7 days) may have been saturated: redo short ones
    done = {(c.get("gl", ""), c["q"], c["a"]) for c in read_jsonl(ck_p)
            if "n" in c or _days(c["a"]) > 8}
    end = date.today() + timedelta(days=1)
    errors = 0
    for hl, gl, lang, q in SEARCHES:
        n0 = len(seen)
        stack = list(_windows(date.fromisoformat(start), end))[::-1]
        while stack:
            a, b = stack.pop()
            key = a.isoformat() + b.isoformat()
            if (gl, q, key) in done or ("", q, key) in done:
                continue
            try:
                items = _fetch(hl, gl, lang, q, a, b)
                errors = 0
            except Exception as e:
                errors += 1
                print(f"  [gnews] {gl} {q[:30]} {a}: {type(e).__name__} {e}", flush=True)
                if errors >= 20:
                    print("  [gnews] too many consecutive errors, stopping (re-run to resume)", flush=True)
                    return len(seen)
                stack.append((a, b))  # retry this window after a pause
                time.sleep(min(60 * errors, 900))
                continue
            if len(items) >= CAP and (b - a).days > MIN_DAYS:
                mid = a + (b - a) / 2
                stack += [(mid, b), (a, mid)]
                continue
            new = [r for r in (to_record(x, lang) for x in items) if r["record_id"] not in seen]
            seen.update(r["record_id"] for r in new)
            write_jsonl(out_p, new, mode="a")
            write_jsonl(ck_p, [{"gl": gl, "q": q, "a": key, "n": len(items)}], mode="a")
        print(f"  [gnews] {gl} {q[:50]}: +{len(seen) - n0} (total {len(seen)})", flush=True)
    return len(seen)
