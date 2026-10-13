"""Merge all raw sources -> deduplicated, enriched corpus -> CSV / Scopus-CSV / RIS."""
import csv
import html
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from .common import (auto_keywords, detect_language, join, norm_title, raw_path, read_jsonl,
                     split_sections, theme)
from .config import COLUMNS, DATA, ROOT

SOURCE_FILES = ["openalex", "semanticscholar", "arxiv", "hal", "doaj",
                "bis_speeches", "ecb_speeches", "authorities", "gdelt", "gnews"]
# metadata priority when merging duplicates (first wins for scalar fields)
PRIORITY = {"OpenAlex": 0, "Semantic Scholar": 1, "HAL": 2, "DOAJ": 3, "arXiv": 4,
            "ECB speeches": 0, "BIS central bankers' speeches": 1}
PREC_RANK = {"day": 4, "month": 2, "year": 1}


def _prec(r):
    p = (r.get("date_precision") or "").split(" ")[0]
    rank = PREC_RANK.get(p, 0)
    return rank - 1 if "Jan 1" in (r.get("date_precision") or "") else rank


def load_all():
    recs = []
    for name in SOURCE_FILES:
        rows = read_jsonl(raw_path(name))
        print(f"  {name:16s} {len(rows):7d}")
        recs += rows
    return recs


def attach_fulltext(recs):
    idx = {}
    for e in read_jsonl(raw_path("fulltext_index")):
        idx[e["record_id"]] = e
    for r in recs:
        e = idx.get(r["record_id"])
        if r.get("full_text_path"):
            r.setdefault("full_text_status", "from source")
            continue
        if not e:
            r["full_text_status"] = "not fetched"
            continue
        r["full_text_status"] = f"downloaded ({e['kind']})" if e["status"] == "ok" else e["status"]
        if e["status"] == "ok":
            r["full_text_path"], r["full_text_chars"] = e["path"], e["chars"]
        # press: the article page usually carries the real publication date / author / outlet name
        if r.get("source_type") == "press":
            if e.get("page_date_precision") == "day":
                r["publication_date"], r["date_precision"] = e["page_date"], "day (article metadata)"
                r["year"] = e["page_date"][:4]
            r["authors"] = r.get("authors") or e.get("page_author", "")
            if e.get("page_sitename"):
                r["issuer"] = e["page_sitename"]
            r["keywords_author"] = r.get("keywords_author") or e.get("page_tags", "")
            r["subtitle"] = r.get("subtitle") or e.get("page_description", "")


def _key(r):
    st = r.get("source_type")
    if st == "academic":
        if r.get("doi"):
            return "doi:" + r["doi"].lower().strip()
        return f"t:{norm_title(r.get('title'))[:120]}:{r.get('year')}"
    if st == "speech":
        return f"s:{r.get('publication_date')}:{norm_title(r.get('title'))[:50]}"
    return "u:" + (r.get("url") or r["record_id"])


def merge(group):
    group.sort(key=lambda r: PRIORITY.get(r["source_db"], 9))
    m = dict(group[0])
    for r in group[1:]:
        for k, v in r.items():
            if v not in (None, "", []) and m.get(k) in (None, "", []):
                m[k] = v
    m["source_db"] = join(r["source_db"] for r in group)
    best = max(group, key=_prec)
    m["publication_date"], m["date_precision"] = best.get("publication_date"), best.get("date_precision")
    m["abstract"] = max((r.get("abstract") or "" for r in group), key=len)
    for k in ("keywords_author", "keywords_indexed"):
        m[k] = join(x for r in group for x in (r.get(k) or "").split("; "))
    ft = max(group, key=lambda r: r.get("full_text_chars") or 0)
    for k in ("full_text_path", "full_text_chars", "full_text_status"):
        m[k] = ft.get(k)
    m["merged_ids"] = join(r["record_id"] for r in group)
    return m


def dedupe(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[_key(r)].append(r)
    out = [merge(g) if len(g) > 1 else g[0] for g in groups.values()]
    # second pass for academic items: same normalised title + year across different DOIs
    by_t = defaultdict(list)
    for r in out:
        if r.get("source_type") == "academic" and len(norm_title(r.get("title"))) > 25:
            by_t[(norm_title(r["title"]), str(r.get("year")))].append(r)
    drop = set()
    for g in by_t.values():
        if len(g) > 1:
            merged = merge(g)
            for r in g:
                drop.add(id(r))
            out.append(merged)
    return [r for r in out if id(r) not in drop]


def _read_text(r):
    p = r.get("full_text_path")
    return (ROOT / p).read_text(encoding="utf-8") if p and (ROOT / p).exists() else ""


FIRST_YEAR = 2014  # first CBDC proposals (Fedcoin 2014, Bank of England 2015): earlier "themes" were all checked false


def _year(r):
    try:
        return int(str(r.get("publication_date") or r.get("year") or "")[:4])
    except ValueError:
        return 9999


def enrich(r):
    for k in ("title", "subtitle", "venue", "issuer", "publisher", "abstract", "authors"):
        if isinstance(r.get(k), str):
            r[k] = html.unescape(r[k]).strip()
    text = _read_text(r)
    r["relevance"], r["theme_score"], r["cbdc_mentions"], r["cbdc_terms"] = theme(
        title=" ".join(str(r.get(k) or "") for k in ("title", "subtitle")),
        keywords="; ".join(str(r.get(k) or "") for k in ("keywords_author", "keywords_indexed")).strip("; "),
        abstract=r.get("abstract") or "", body=text, context=r.get("source_type") == "press")
    if r["relevance"] == "off-topic" and not text and not r.get("abstract") and r.get("source_type") == "press":
        r["relevance"] = "unverified (no text)"  # GDELT matched the article body, which we could not fetch
    if r["relevance"] in ("main theme", "secondary theme") and _year(r) < FIRST_YEAR:
        r["relevance"] = "dated before 2014"  # homonyms (CBDC projects, DCEP chutes) or misdated metadata
    lang = (r.get("language") or "").lower()
    if ";" in lang or len(lang) != 2:  # e.g. DOAJ journal languages "EN; RU": detect the article's own
        lang = detect_language(" ".join([r.get("title") or "", r.get("abstract") or "", text[:3000]])) or lang[:2]
        r["language_origin"] = "detected from text"
    r["language"] = lang
    if not r.get("language"):
        r["language"] = detect_language(text or r.get("abstract") or r.get("title"))
        r["language_origin"] = "detected from text"
    if text:
        r.update(split_sections(text))
    r["keywords_auto"] = auto_keywords(
        "\n".join([r.get("title") or "", r.get("abstract") or "", r.get("introduction") or "",
                   r.get("conclusion") or "", text[:6000] if not r.get("abstract") else ""]),
        (r.get("language") or "en")[:2])
    if not r.get("year") and r.get("publication_date"):
        r["year"] = r["publication_date"][:4]
    return r


def build(min_relevance=None):
    """Keeps every record by default; filter later on `relevance` / `theme_score`.
    Pass e.g. ("main theme", "secondary theme") to drop the rest at build time."""
    print("Loading raw sources")
    recs = load_all()
    attach_fulltext(recs)
    recs = dedupe(recs)
    print(f"After de-duplication: {len(recs)}")
    with ProcessPoolExecutor() as ex:
        recs = list(ex.map(enrich, recs, chunksize=64))
    if min_relevance:
        recs = [r for r in recs if r["relevance"] in min_relevance]
        print(f"After relevance filter: {len(recs)}")
    refs = {x["id"]: x["cite"] for x in read_jsonl(raw_path("openalex_refs"))}
    for r in recs:
        r["references"] = "; ".join(refs[i] for i in r.pop("_refs", []) if i in refs)
    recs.sort(key=lambda r: (-r["theme_score"], r.get("publication_date") or "", r.get("title") or ""))
    write_master(recs)  # single deliverable; write_scopus / write_ris stay available if a tool needs them
    for old in ("cbdc_corpus_scopus.csv", "cbdc_corpus.ris"):
        (DATA / old).unlink(missing_ok=True)
    summary(recs)


# ----------------------------------------------------------------- writers

def write_master(recs):
    p = DATA / "cbdc_corpus.csv"
    extra = sorted({k for r in recs for k in r} - set(COLUMNS) - {"_refs"})
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS + extra, extrasaction="ignore", quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in recs:
            row = dict(r)
            row["full_text"] = _read_text(r)
            w.writerow(row)
    print(f"Wrote {p.relative_to(ROOT)}")


ORG_RE = re.compile(r"\b(bank|banque|banco|banca|bundesbank|riksbank|authority|commission|ministry|treasury|"
                    r"institute|fund|council|parliament|reserve|board|office|university|association|"
                    r"group|department|agency|house|government|secretariat|committee|system)\b", re.I)


def _scopus_author(name):
    if ORG_RE.search(name) or len(name.split()) > 5:
        return name  # organisation as author: keep the full name
    bits = name.replace(",", " ").split()
    if len(bits) < 2:
        return name
    return f"{bits[-1]} {''.join(b[0] + '.' for b in bits[:-1] if b[0].isalpha())}"


SCOPUS_TYPES = {"article": "Article", "journal-article": "Article", "journalarticle": "Article",
                "preprint": "Preprint", "conference": "Conference paper", "comm": "Conference paper",
                "art": "Article", "couv": "Book chapter", "ouv": "Book", "these": "Thesis",
                "undefined": "Working paper", "academic": "Undefined", "trad": "Translation", "report": "Report", "rapport": "Report",
                "casereport": "Article", "editorial": "Editorial", "letterstoeditor": "Letter",
                "booksection": "Book chapter", "study": "Article", "metaanalysis": "Review",
                "book": "Book", "book-chapter": "Book chapter", "report": "Report",
                "speech": "Speech", "news article": "News article", "review": "Review",
                "proceedings-article": "Conference paper", "dissertation": "Thesis"}


def write_scopus(recs):
    """Scopus-like CSV: readable by bibliometrix convert2df(dbsource='scopus', format='csv') and VOSviewer."""
    p = DATA / "cbdc_corpus_scopus.csv"
    cols = ["Authors", "Author full names", "Author(s) ID", "Title", "Year", "Source title", "Volume",
            "Issue", "Art. No.", "Page start", "Page end", "Page count", "Cited by", "DOI", "Link",
            "Affiliations", "Authors with affiliations", "Abstract", "Author Keywords",
            "Index Keywords", "References", "Correspondence Address", "Publisher", "ISSN",
            "Language of Original Document", "Abbreviated Source Title", "Document Type",
            "Publication Stage", "Open Access", "Source", "EID"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in recs:
            names = [a for a in (r.get("authors") or "").split("; ") if a]
            dt = r.get("document_type") or r.get("source_type") or ""
            w.writerow({
                "Authors": "; ".join(_scopus_author(a) for a in names) or "[No author name available]",
                "Author full names": "; ".join(names),
                "Title": r.get("title"), "Year": r.get("year"),
                "Source title": r.get("venue") or r.get("issuer") or "",
                "Volume": r.get("volume"), "Issue": r.get("issue"),
                "Page start": r.get("page_start"), "Page end": r.get("page_end"),
                "Cited by": r.get("cited_by_count"), "DOI": r.get("doi"), "Link": r.get("url"),
                "Affiliations": r.get("author_affiliations") or r.get("issuer"),
                "Authors with affiliations": r.get("author_affiliations"),
                "Abstract": r.get("abstract") or (r.get("introduction") or "")[:3000],
                "Author Keywords": r.get("keywords_author") or r.get("keywords_indexed"),
                "Index Keywords": join((r.get("keywords_indexed") or "").split("; ")
                                       + (r.get("keywords_auto") or "").split("; ")),
                "References": r.get("references"), "Publisher": r.get("publisher") or r.get("issuer"),
                "Language of Original Document": r.get("language"),
                "Abbreviated Source Title": r.get("venue") or r.get("issuer"),
                "Document Type": SCOPUS_TYPES.get(dt.split(";")[0].strip().lower(), dt.title()),
                "Publication Stage": "Final", "Open Access": r.get("open_access"),
                "Source": r.get("source_db"), "EID": r["record_id"],
            })
    print(f"Wrote {p.relative_to(ROOT)}")


RIS_TYPES = {"academic": "JOUR", "speech": "GEN", "press": "NEWS", "official": "GOVDOC"}


def write_ris(recs):
    p = DATA / "cbdc_corpus.ris"
    with open(p, "w", encoding="utf-8") as f:
        for r in recs:
            ty = RIS_TYPES.get(r.get("source_type"), "GEN")
            dt = (r.get("document_type") or "").lower()
            if "preprint" in dt or "report" in dt or "working" in dt:
                ty = "RPRT"
            elif "chapter" in dt:
                ty = "CHAP"
            elif dt == "book":
                ty = "BOOK"
            lines = [("TY", ty), ("ID", r["record_id"]), ("TI", r.get("title"))]
            lines += [("AU", a) for a in (r.get("authors") or "").split("; ") if a]
            d = r.get("publication_date") or ""
            lines += [("PY", r.get("year")), ("DA", d.replace("-", "/")), ("T2", r.get("venue")),
                      ("PB", r.get("publisher") or r.get("issuer")), ("AB", r.get("abstract")),
                      ("DO", r.get("doi")), ("UR", r.get("url")), ("LA", r.get("language")),
                      ("M3", r.get("document_type")), ("CY", r.get("country")),
                      ("N1", f"Issuer: {r.get('issuer') or ''} | Source: {r.get('source_db')} | "
                             f"Relevance: {r.get('relevance')} ({r.get('theme_score')}) | Full text: {r.get('full_text_path') or 'n/a'}")]
            kws = join(((r.get("keywords_author") or "") + "; " + (r.get("keywords_indexed") or "")).split("; "))
            lines += [("KW", k) for k in kws.split("; ") if k]
            for tag, val in lines:
                if val not in (None, ""):
                    f.write(f"{tag}  - {str(val).replace(chr(10), ' ')}\n")
            f.write("ER  - \n\n")
    print(f"Wrote {p.relative_to(ROOT)}")


def summary(recs):
    print("\nBy source type:", dict(Counter(r.get("source_type") for r in recs)))
    print("By relevance:  ", dict(Counter(r.get("relevance") for r in recs)))
    print("Top languages: ", Counter(r.get("language") for r in recs).most_common(12))
    print("By year:       ", sorted(Counter(str(r.get("year")) for r in recs).items()))
    print("With full text:", sum(bool(r.get("full_text_path")) for r in recs),
          "| with abstract:", sum(bool(r.get("abstract")) for r in recs),
          "| day-precise dates:", sum((r.get("date_precision") or "").startswith("day") for r in recs))
