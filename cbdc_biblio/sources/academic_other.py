"""Other academic / open-archive sources: Semantic Scholar, arXiv, HAL, DOAJ."""
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

from ..common import (Throttle, clean, date_and_precision, join, make_id, new_record, raw_path,
                      read_jsonl, session, write_jsonl)
from ..config import QUERY_PHRASES

S = session()

# ----------------------------------------------------------------- Semantic Scholar

def semanticscholar():
    out = {r["native_id"]: r for r in read_jsonl(raw_path("semanticscholar"))}  # never lose earlier hits
    phrases = [p for p in QUERY_PHRASES if p.isascii()]
    for i in range(0, len(phrases), 15):  # long OR-queries get truncated: query in chunks
        _s2_query(" | ".join(f'"{p}"' for p in phrases[i:i + 15]), out)
    write_jsonl(raw_path("semanticscholar"), out.values())
    return len(out)


def _s2_query(q, out):
    fields = ("title,abstract,authors,year,publicationDate,venue,publicationVenue,externalIds,"
              "openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,citationCount,journal,url")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search/bulk?query={quote(q)}&fields={fields}"
    token = None
    while True:
        u = url + (f"&token={token}" if token else "")
        for attempt in range(8):
            r = S.get(u, timeout=90)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            break
        r.raise_for_status()
        d = r.json()
        for p in d.get("data") or []:
            ext = p.get("externalIds") or {}
            date, prec = date_and_precision(p.get("publicationDate") or p.get("year"))
            pv = p.get("publicationVenue") or {}
            out[p["paperId"]] = (new_record(
                record_id=make_id("s2", p["paperId"]), source_db="Semantic Scholar",
                source_type="academic", document_type=join(p.get("publicationTypes") or []),
                title=p.get("title"), authors=join(a.get("name") for a in p.get("authors") or []),
                venue=p.get("venue") or pv.get("name"), issuer=p.get("venue") or pv.get("name"),
                issuer_type=pv.get("type"), publication_date=date, date_precision=prec,
                year=p.get("year"), doi=ext.get("DOI"), url=p.get("url"),
                pdf_url=(p.get("openAccessPdf") or {}).get("url"), native_id=p["paperId"],
                abstract=p.get("abstract"),
                keywords_indexed=join([f.get("category") for f in p.get("s2FieldsOfStudy") or []]),
                cited_by_count=p.get("citationCount"),
                open_access="yes" if (p.get("openAccessPdf") or {}).get("url") else "",
                ssrn_or_arxiv=join([ext.get("ArXiv"), ext.get("SSRN")]),
            ))
        print(f"  [s2] {len(out)} / {d.get('total')}", flush=True)
        token = d.get("token")
        if not token:
            break
        time.sleep(1.5)

# ----------------------------------------------------------------- arXiv

ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def arxiv():
    phrases = [p for p in QUERY_PHRASES if p.isascii() and not p.startswith("Project ")]
    q = " OR ".join(f'all:"{p}"' if " " in p else f"all:{p}" for p in phrases)
    out, start, t = {}, 0, Throttle(3.5)
    while True:
        t.wait()
        r = S.get("https://export.arxiv.org/api/query", params={
            "search_query": q, "start": start, "max_results": 200,
            "sortBy": "submittedDate", "sortOrder": "ascending"}, timeout=90)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        entries = root.findall("a:entry", ATOM)
        for e in entries:
            aid = e.findtext("a:id", "", ATOM)
            pdf = next((l.get("href") for l in e.findall("a:link", ATOM) if l.get("title") == "pdf"), "")
            authors = e.findall("a:author", ATOM)
            affs = []
            for a in authors:
                aff = [x.text for x in a.findall("arxiv:affiliation", ATOM) if x.text]
                nm = a.findtext("a:name", "", ATOM)
                affs.append(f"{nm} ({', '.join(aff)})" if aff else nm)
            date, prec = date_and_precision(e.findtext("a:published", "", ATOM))
            cats = [c.get("term") for c in e.findall("a:category", ATOM)]
            out[aid] = new_record(
                record_id=make_id("arxiv", aid), source_db="arXiv", source_type="academic",
                document_type="preprint", title=clean(e.findtext("a:title", "", ATOM)),
                authors=join(a.findtext("a:name", "", ATOM) for a in authors),
                author_affiliations="; ".join(affs), issuer="arXiv (Cornell University)",
                issuer_type="repository", venue=e.findtext("arxiv:journal_ref", "", ATOM) or "arXiv",
                publication_date=date, date_precision=prec, year=date[:4],
                doi=e.findtext("arxiv:doi", "", ATOM) or f"10.48550/arXiv.{aid.rsplit('/', 1)[-1].split('v')[0]}",
                url=aid, pdf_url=pdf, native_id=aid,
                abstract=clean(e.findtext("a:summary", "", ATOM)),
                keywords_indexed=join(cats), open_access="green", license="arXiv",
            )
        print(f"  [arxiv] {len(out)}", flush=True)
        if len(entries) < 200:
            break
        start += 200
    write_jsonl(raw_path("arxiv"), out.values())
    return len(out)

# ----------------------------------------------------------------- HAL (French open archive)

def hal():
    q = " OR ".join(f'"{p}"' for p in QUERY_PHRASES)
    fl = ("halId_s,uri_s,title_s,subTitle_s,abstract_s,authFullName_s,structName_s,producedDate_s,"
          "publicationDate_s,language_s,keyword_s,docType_s,journalTitle_s,journalPublisher_s,"
          "doiId_s,files_s,country_s,bookTitle_s,conferenceTitle_s")
    out, start = [], 0
    while True:
        r = S.get("https://api.archives-ouvertes.fr/search/", params={
            "q": q, "fl": fl, "rows": 500, "start": start, "wt": "json"}, timeout=90)
        r.raise_for_status()
        docs = r.json()["response"]["docs"]
        for d in docs:
            date, prec = date_and_precision(d.get("publicationDate_s") or d.get("producedDate_s"))
            first = lambda k: (d.get(k) or [""])[0] if isinstance(d.get(k), list) else d.get(k, "")
            out.append(new_record(
                record_id=make_id("hal", d["halId_s"]), source_db="HAL", source_type="academic",
                document_type=d.get("docType_s"), title=first("title_s"), subtitle=first("subTitle_s"),
                authors=join(d.get("authFullName_s")), author_affiliations=join(d.get("structName_s")),
                issuer=d.get("journalPublisher_s") or d.get("journalTitle_s") or "HAL",
                issuer_type="journal" if d.get("journalTitle_s") else "repository",
                venue=d.get("journalTitle_s") or d.get("bookTitle_s") or d.get("conferenceTitle_s"),
                publisher=d.get("journalPublisher_s"), country=join(d.get("country_s")),
                publication_date=date, date_precision=prec, year=date[:4],
                language=first("language_s"), language_origin="HAL metadata",
                doi=d.get("doiId_s"), url=d.get("uri_s"), pdf_url=first("files_s"),
                native_id=d["halId_s"], abstract=first("abstract_s"),
                keywords_author=join(d.get("keyword_s")),
                open_access="green" if d.get("files_s") else "",
            ))
        print(f"  [hal] {len(out)}", flush=True)
        if len(docs) < 500:
            break
        start += 500
    write_jsonl(raw_path("hal"), out)
    return len(out)

# ----------------------------------------------------------------- DOAJ

def doaj():
    out = {}
    for p in QUERY_PHRASES:
        page = 1
        while True:
            r = S.get(f"https://doaj.org/api/search/articles/{quote(chr(34) + p + chr(34))}",
                      params={"page": page, "pageSize": 100}, timeout=60)
            if r.status_code != 200:
                break
            d = r.json()
            for a in d.get("results") or []:
                b = a.get("bibjson") or {}
                j = b.get("journal") or {}
                doi = next((i.get("id") for i in b.get("identifier") or [] if i.get("type") == "doi"), "")
                ym = "-".join(x for x in [b.get("year"), (b.get("month") or "").zfill(2) if b.get("month") else ""] if x)
                date, prec = date_and_precision(ym)
                link = next((l.get("url") for l in b.get("link") or [] if l.get("type") == "fulltext"), "")
                out[a["id"]] = new_record(
                    record_id=make_id("doaj", a["id"]), source_db="DOAJ", source_type="academic",
                    document_type="journal-article", title=b.get("title"),
                    authors=join(x.get("name") for x in b.get("author") or []),
                    author_affiliations="; ".join(
                        f"{x.get('name')} ({x.get('affiliation')})" if x.get("affiliation") else x.get("name", "")
                        for x in b.get("author") or []),
                    issuer=j.get("publisher") or j.get("title"), issuer_type="journal",
                    venue=j.get("title"), publisher=j.get("publisher"), country=j.get("country"),
                    publication_date=date, date_precision=prec, year=b.get("year"),
                    language=join(j.get("language")), language_origin="DOAJ journal language",
                    doi=doi, url=link, native_id=a["id"], abstract=b.get("abstract"),
                    keywords_author=join(b.get("keywords")), open_access="gold",
                    volume=j.get("volume"), issue=j.get("number"),
                    page_start=b.get("start_page"), page_end=b.get("end_page"),
                )
            if page * 100 >= d.get("total", 0):
                break
            page += 1
            time.sleep(0.3)
        time.sleep(0.3)
    print(f"  [doaj] {len(out)}", flush=True)
    write_jsonl(raw_path("doaj"), out.values())
    return len(out)

