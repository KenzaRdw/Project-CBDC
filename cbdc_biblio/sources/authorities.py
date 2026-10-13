"""Official announcements, reports, consultations and press releases from authorities.

Starts from each authority's CBDC hub page, follows links one level deep (same site
+ PDFs), extracts every document and keeps those that are about CBDC. Seeds can be
added freely to SEEDS; dead seeds are reported and skipped.
"""
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urldefrag, urlparse

from bs4 import BeautifulSoup

from ..common import (date_and_precision, date_from_text, detect_language, make_id, new_record, raw_path,
                      relevance, save_fulltext, session, write_jsonl)
from ..fulltext import extract_url

# (issuer, issuer_type, country/area, hub url)
# Not crawlable on 2026-10-08 (refuse automated clients or hub moved): Council of the EU,
# Bank of England, IMF, RBA, RBNZ, ECCB, HKMA, Norges Bank, Danmarks Nationalbank, SARB,
# Bank of Korea, CBN eNaira. Their speeches are covered by the BIS archive and their papers by OpenAlex.
SEEDS = [
    ("European Central Bank", "central bank", "Euro area", "https://www.ecb.europa.eu/euro/digital_euro/html/index.en.html"),
    ("European Central Bank", "central bank", "Euro area", "https://www.ecb.europa.eu/euro/digital_euro/timeline/html/index.en.html"),
    ("European Central Bank", "central bank", "Euro area", "https://www.ecb.europa.eu/euro/digital_euro/progress/html/index.en.html"),
    ("European Central Bank", "central bank", "Euro area", "https://www.ecb.europa.eu/euro/digital_euro/features/html/index.en.html"),
    ("European Commission", "government", "EU", "https://finance.ec.europa.eu/digital-finance/digital-euro_en"),
    ("European Parliament", "legislature", "EU", "https://www.europarl.europa.eu/legislative-train/theme-an-economy-that-works-for-people/file-digital-euro"),
    ("Deutsche Bundesbank", "central bank", "Germany", "https://www.bundesbank.de/de/aufgaben/unbarer-zahlungsverkehr/digitaler-euro"),
    ("Banque de France", "central bank", "France", "https://www.banque-france.fr/fr/publications-et-statistiques/publications?search=euro%20num%C3%A9rique"),
    ("HM Treasury", "government", "United Kingdom", "https://www.gov.uk/government/consultations/the-digital-pound-a-new-form-of-money-for-households-and-businesses"),
    ("Board of Governors of the Federal Reserve System", "central bank", "United States", "https://www.federalreserve.gov/central-bank-digital-currency.htm"),
    ("Federal Reserve Bank of Boston", "central bank", "United States", "https://www.bostonfed.org/payment-services/project-hamilton.aspx"),
    ("The White House", "government", "United States", "https://www.whitehouse.gov/presidential-actions/2025/01/strengthening-american-leadership-in-digital-financial-technology/"),
    ("BIS Innovation Hub", "international organisation", "International", "https://www.bis.org/about/bisih/topics/cbdc.htm"),
    ("World Bank", "international organisation", "International", "https://www.worldbank.org/en/topic/financialsector/brief/central-bank-digital-currencies"),
    ("Bank of Canada", "central bank", "Canada", "https://www.bankofcanada.ca/research/digital-currencies-and-fintech/"),
    ("Sveriges Riksbank", "central bank", "Sweden", "https://www.riksbank.se/en-gb/payments--cash/e-krona/"),
    ("Swiss National Bank", "central bank", "Switzerland", "https://www.snb.ch/en/snb-explained/cbdc"),
    ("Swiss National Bank", "central bank", "Switzerland", "https://www.snb.ch/en/the-snb/mandates-goals/payment-transactions/projekt_helvetia"),
    ("Bank of Japan", "central bank", "Japan", "https://www.boj.or.jp/en/paym/digital/index.htm"),
    ("Monetary Authority of Singapore", "central bank", "Singapore", "https://www.mas.gov.sg/schemes-and-initiatives/project-orchid"),
    ("Reserve Bank of India", "central bank", "India", "https://www.rbi.org.in/Scripts/PublicationsView.aspx?id=21149"),
    ("Banco Central do Brasil", "central bank", "Brazil", "https://www.bcb.gov.br/en/financialstability/drex_en"),
    ("Bank of Russia", "central bank", "Russia", "https://cbr.ru/eng/analytics/d_ok/dig_ruble/"),
    ("Central Bank of The Bahamas", "central bank", "Bahamas", "https://www.sanddollar.bs/"),
    ("Bank of Jamaica", "central bank", "Jamaica", "https://boj.org.jm/core-functions/currency/cbdc/"),
    ("People's Bank of China", "central bank", "China", "https://www.pbc.gov.cn/en/3688006/index.html"),
]

DOCLIKE = re.compile(r"\.pdf($|\?)|press|publication|pub/|report|paper|speech|consult|statement|news|"
                     r"blog|interview|letter|opinion|decision|regulation|legislat|faq|digital|cbdc|"
                     r"e-?krona|e-?hkd|drex|sand-?dollar|naira|project", re.I)
SKIP = re.compile(r"\.(jpg|jpeg|png|gif|svg|zip|xlsx?|csv|mp4|mp3)($|\?)|mailto:|javascript:|"
                  r"/(careers|jobs|contact|search|login|cookies?|privacy|sitemap)\b", re.I)
GENERIC = re.compile(r"^(need help|press releases?|glossary|home|read more|download|pdf|more|"
                     r"publications?|news|overview|faqs?)\b", re.I)
S = session(retries=2)


def _links(issuer, url):
    try:
        r = S.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"  [authorities] seed failed {url}: {e}", flush=True)
        return []
    host = urlparse(r.url).netloc.split(":")[0]
    root = ".".join(host.split(".")[-2:])
    soup = BeautifulSoup(r.text, "lxml")
    out = {r.url: ""}
    for a in soup.find_all("a", href=True):
        u = urldefrag(urljoin(r.url, a["href"]))[0]
        if not u.startswith("http") or SKIP.search(u):
            continue
        same_site = urlparse(u).netloc.endswith(root)
        text = a.get_text(" ", strip=True)
        if (same_site and (DOCLIKE.search(u) or DOCLIKE.search(text))) or u.lower().endswith(".pdf"):
            out.setdefault(u, text)
    return list(out.items())[:250]


def _doc_type(url, title):
    s = f"{url} {title}".lower()
    for k, v in [("press", "press release"), ("speech", "speech"), ("interview", "interview"),
                 ("blog", "blog post"), ("consult", "consultation"), ("regulation", "legislation"),
                 ("legislat", "legislation"), ("presidential-action", "executive order"),
                 ("report", "report"), ("paper", "paper"), ("faq", "FAQ"), (".pdf", "official document")]:
        if k in s:
            return v
    return "web page"


def harvest():
    jobs = {}
    for issuer, itype, country, seed in SEEDS:
        for u, anchor in _links(issuer, seed):
            jobs.setdefault(u, (issuer, itype, country, anchor))
    print(f"  [authorities] {len(jobs)} candidate documents from {len(SEEDS)} hubs", flush=True)

    def work(item):
        u, (issuer, itype, country, anchor) = item
        res = extract_url(u)
        if res.get("status") != "ok":
            return None
        meta, text = res.get("meta") or {}, res["text"]
        title = meta.get("title") or ""
        if len(anchor) >= 20 and not GENERIC.match(anchor) and (len(title) < 25 or GENERIC.match(title)):
            title = anchor
        title = title or anchor or u.rsplit("/", 1)[-1]
        rel, n, terms = relevance(f"{title}\n{anchor}", text)
        if not (rel == "core" or (rel in ("substantial", "peripheral") and n >= 2)):
            return None
        rid = make_id("auth", u)
        path, chars = save_fulltext(rid, text)
        d, prec = date_and_precision(meta.get("date"))
        prec = f"{prec} (page metadata)" if d else ""
        # page metadata missing, implausible for a CBDC document, or EUR-Lex (whose page dates are unreliable)
        if not d or d < "2014" or "eur-lex.europa.eu" in u:
            d = date_from_text(text) or (d if "eur-lex.europa.eu" in u else "")
            prec = "day (found in document text)" if d else ""
        return new_record(
            record_id=rid, source_db="Authorities (hub crawl)", source_type="official",
            document_type=_doc_type(u, title), title=title, subtitle=meta.get("description"),
            authors=meta.get("author"), issuer=issuer, issuer_type=itype, country=country,
            publication_date=d, date_precision=prec,
            year=d[:4], language=meta.get("language") or detect_language(text),
            language_origin="page metadata" if meta.get("language") else "detected from text",
            url=u, pdf_url=u if res.get("kind") == "pdf" else "", native_id=u,
            keywords_author=meta.get("tags"), cbdc_terms=terms, cbdc_mentions=n, relevance=rel,
            full_text_path=path, full_text_chars=chars, full_text_status="crawled",
        )

    with ThreadPoolExecutor(8) as ex:
        out = [r for r in ex.map(work, jobs.items()) if r]
    write_jsonl(raw_path("authorities"), out)
    print(f"  [authorities] kept {len(out)} CBDC documents", flush=True)
    return len(out)
