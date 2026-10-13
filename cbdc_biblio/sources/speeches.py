"""Central bankers' speeches.

* BIS bulk archive (all central banks, 1996 -> dump date, full text).
* ECB speeches CSV (full text, incl. non-English originals).
* BIS RSS feed to top up speeches published after the bulk dump.
"""
import csv
import io
import re
import sys
import zipfile

import feedparser

from ..common import (date_and_precision, detect_language, make_id, new_record, raw_path,
                      relevance, save_fulltext, session, write_jsonl)
from ..config import CACHE

S = session()
csv.field_size_limit(sys.maxsize)

BIS_ZIP = "https://www.bis.org/pages/download-central-bankers-speeches/speeches.zip"
ECB_CSV = "https://www.ecb.europa.eu/press/key/shared/data/all_ECB_speeches.csv"
BIS_RSS = "https://www.bis.org/doclist/cbspeeches.rss"

_INST = re.compile(
    r"((?:[A-Z][\w'’.\-]+ )*(?:Bank|Banque|Banco|Banca|Bundesbank|Riksbank|Nationalbank|Authority|"
    r"Reserve|Settlements|Fund|Bank Al-Maghrib)(?: (?:of|de|do|del|d'|du|for|al)(?: the)? "
    r"[A-Z][\w'’.\-]+(?: (?:and )?[A-Z][\w'’.\-]+)*)?)")
_FED = re.compile(r"Federal Reserve(?! Bank)|Board of Governors", re.I)


def issuer_from(text):
    if not text:
        return ""
    if _FED.search(text) and not re.search(r"Federal Reserve Bank of", text):
        return "Board of Governors of the Federal Reserve System"
    m = _INST.search(text)
    return m.group(1).strip() if m else ""


def _keep(rel, mentions):
    return rel == "core" or (rel in ("substantial", "peripheral") and mentions >= 2)


def _speech_record(prefix, src, native, title, subtitle, speaker, date, text, url, issuer):
    rel, mentions, terms = relevance(f"{title}\n{subtitle}", text)
    if not _keep(rel, mentions):
        return None
    rid = make_id(prefix, native)
    path, n = save_fulltext(rid, text)
    lang = detect_language(text)
    d, prec = date_and_precision(date)
    return new_record(
        record_id=rid, source_db=src, source_type="speech", document_type="speech",
        title=title, subtitle=subtitle, authors=speaker, issuer=issuer, issuer_type="central bank",
        publication_date=d, date_precision=prec, year=d[:4], language=lang,
        language_origin="detected from text", url=url, native_id=native,
        cbdc_terms=terms, cbdc_mentions=mentions, relevance=rel,
        full_text_path=path, full_text_chars=n, full_text_status="from source",
    )


def bis():
    zp = CACHE / "bis_speeches.zip"
    if not zp.exists():
        print("  [bis] downloading bulk archive (~130 MB)", flush=True)
        with S.get(BIS_ZIP, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(zp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    out = []
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as fh:
            for row in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")):
                title = re.sub(r"^[^:]{3,80}:\s+", "", row["title"]).strip()  # drop "Speaker: " prefix
                rec = _speech_record("bis", "BIS central bankers' speeches", row["url"], title,
                                     row["description"], row["author"], row["date"], row["text"],
                                     row["url"], issuer_from(row["description"]) or issuer_from(row["title"]))
                if rec:
                    rec["pdf_url"] = row["url"]
                    out.append(rec)
    out += _bis_rss(max(r.get("publication_date", "") for r in out) if out else "")
    print(f"  [bis] kept {len(out)} speeches", flush=True)
    write_jsonl(raw_path("bis_speeches"), out)
    return len(out)


def _bis_rss(after):
    """Speeches newer than the bulk dump: RSS metadata + PDF text."""
    from ..fulltext import extract_url
    feed = feedparser.parse(S.get(BIS_RSS, timeout=60).content)
    out = []
    for e in feed.entries:
        d, _ = date_and_precision(e.get("dc_date") or e.get("updated") or e.get("published"))
        if not d or d <= after:
            continue
        res = extract_url(e.link.replace(".htm", ".pdf") if e.link.endswith(".htm") else e.link)
        title = re.sub(r"^[^:]{3,80}:\s+", "", e.title)
        desc = re.sub(r"<[^>]+>", " ", e.get("description", ""))
        rec = _speech_record("bis", "BIS central bankers' speeches (RSS)", e.link, title, desc,
                             e.get("cb_person_name") or e.get("author", ""), d, res.get("text", ""),
                             e.link, issuer_from(desc))
        if rec:
            out.append(rec)
    print(f"  [bis] +{len(out)} recent speeches from RSS", flush=True)
    return out


def ecb():
    p = CACHE / "ecb_speeches.csv"
    if not p.exists():
        p.write_bytes(S.get(ECB_CSV, timeout=600).content)
    out = []
    with open(p, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="|"):
            native = f"{row['date']}|{row['title']}"
            rec = _speech_record("ecb", "ECB speeches", native, row["title"], row["subtitle"],
                                 row["speakers"], row["date"], row["contents"],
                                 "", "European Central Bank")  # the dump has no per-speech URL
            if rec:
                out.append(rec)
    print(f"  [ecb] kept {len(out)} speeches", flush=True)
    write_jsonl(raw_path("ecb_speeches"), out)
    return len(out)
