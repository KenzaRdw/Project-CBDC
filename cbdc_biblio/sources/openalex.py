"""OpenAlex: journal articles, working papers, books, chapters, preprints, reports.

Every phrase in QUERY_PHRASES is searched in titles+abstracts; results are merged
on the OpenAlex id. Referenced works are resolved afterwards into citation strings
(Scopus "References" style) so co-citation analysis works in bibliometrix/VOSviewer.
"""
import os
from urllib.parse import quote

from ..common import (Throttle, date_and_precision, join, make_id, new_record, raw_path,
                      read_jsonl, session, write_jsonl)
from ..config import QUERY_PHRASES

API = "https://api.openalex.org/works"
SELECT = ("id,doi,display_name,publication_date,publication_year,language,type,primary_location,"
          "best_oa_location,open_access,authorships,keywords,topics,abstract_inverted_index,"
          "cited_by_count,referenced_works,biblio")
S = session()
T = Throttle(0.12)
# Without a key OpenAlex grants a small daily budget per IP. Free key: https://openalex.org/settings/api
KEY = os.environ.get("OPENALEX_API_KEY", "")


def _get(url):
    T.wait()
    r = S.get(url + (f"&api_key={KEY}" if KEY else ""), timeout=60)
    if r.status_code == 429:
        raise RuntimeError("OpenAlex budget exhausted (set OPENALEX_API_KEY or wait for midnight UTC)")
    r.raise_for_status()
    return r.json()


def _abstract(inv):
    if not inv:
        return ""
    pos = [(i, w) for w, idx in inv.items() for i in idx]
    return " ".join(w for _, w in sorted(pos))


def _iter_phrase(phrase):
    cursor = "*"
    while cursor:
        d = _get(f"{API}?filter=title_and_abstract.search:{quote(chr(34) + phrase + chr(34))}"
                 f"&per-page=200&cursor={cursor}&select={SELECT}")
        yield from d["results"]
        cursor = d["meta"].get("next_cursor")
        if not d["results"]:
            break


def to_record(w):
    pl = w.get("primary_location") or {}
    src = pl.get("source") or {}
    oa = w.get("best_oa_location") or {}
    auths = w.get("authorships") or []
    date, prec = date_and_precision(w.get("publication_date"))
    if date.endswith("-01-01"):
        prec = "day (Jan 1: may be year-only)"
    affs = []
    for a in auths:
        insts = [i.get("display_name") for i in a.get("institutions") or [] if i.get("display_name")]
        name = (a.get("author") or {}).get("display_name", "")
        affs.append(f"{name} ({', '.join(insts)})" if insts else name)
    countries = join(c for a in auths for c in (a.get("countries") or []))
    b = w.get("biblio") or {}
    return new_record(
        record_id=make_id("oa", w["id"]), source_db="OpenAlex", source_type="academic",
        document_type=w.get("type"), title=w.get("display_name"),
        authors=join((a.get("author") or {}).get("display_name") for a in auths),
        author_affiliations="; ".join(affs),
        issuer=src.get("host_organization_name") or src.get("display_name"),
        issuer_type=src.get("type"), venue=src.get("display_name"),
        publisher=src.get("host_organization_name"), country=countries,
        publication_date=date, date_precision=prec, year=w.get("publication_year"),
        language=w.get("language"), language_origin="OpenAlex metadata",
        doi=(w.get("doi") or "").replace("https://doi.org/", ""),
        url=w.get("doi") or pl.get("landing_page_url"),
        pdf_url=oa.get("pdf_url") or pl.get("pdf_url"), native_id=w["id"],
        abstract=_abstract(w.get("abstract_inverted_index")),
        keywords_indexed=join([k.get("display_name") for k in w.get("keywords") or []]
                              + [t.get("display_name") for t in (w.get("topics") or [])[:3]]),
        cited_by_count=w.get("cited_by_count"),
        open_access=(w.get("open_access") or {}).get("oa_status"), license=oa.get("license"),
        volume=b.get("volume"), issue=b.get("issue"), page_start=b.get("first_page"),
        page_end=b.get("last_page"), _refs=w.get("referenced_works") or [],
    )


def harvest():
    """Resumable: phrases already completed (openalex_done.jsonl) are skipped."""
    seen = {r["native_id"]: r for r in read_jsonl(raw_path("openalex"))}
    done_p = raw_path("openalex_done")
    done = {r["phrase"] for r in read_jsonl(done_p)}
    for p in QUERY_PHRASES:
        if p in done:
            continue
        n0, got = len(seen), {}
        try:
            for w in _iter_phrase(p):
                got.setdefault(w["id"], w)
        except Exception as e:
            print(f"  [openalex] '{p}' failed: {e}", flush=True)
            if "budget" in str(e):
                break
            continue
        for wid, w in got.items():
            seen.setdefault(wid, to_record(w))
        write_jsonl(raw_path("openalex"), seen.values())
        write_jsonl(done_p, [{"phrase": p}], mode="a")
        print(f"  [openalex] '{p}': +{len(seen) - n0} (total {len(seen)})", flush=True)
    resolve_references()
    return len(seen)


def _cite(w):
    auths = [(a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or []][:6]
    auth = ", ".join(_short(a) for a in auths if a)
    src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    b = w.get("biblio") or {}
    parts = [auth, w.get("display_name") or "", src]
    if b.get("volume"):
        parts.append(str(b["volume"]))
    if b.get("first_page"):
        parts.append(f"pp. {b['first_page']}-{b.get('last_page') or ''}".rstrip("-"))
    s = ", ".join(p for p in parts if p)
    return f"{s}, ({w.get('publication_year') or ''})"


def _short(name):
    """'Christine Lagarde' -> 'Lagarde C.' (Scopus style)."""
    bits = name.replace(",", " ").split()
    if len(bits) < 2:
        return name
    return f"{bits[-1]} {''.join(b[0] + '.' for b in bits[:-1])}"


def resolve_references():
    """Turn referenced OpenAlex ids into citation strings, cached in openalex_refs.jsonl."""
    recs = read_jsonl(raw_path("openalex"))
    cache_p = raw_path("openalex_refs")
    cache = {r["id"]: r["cite"] for r in read_jsonl(cache_p)}
    todo = sorted({i for r in recs for i in r.get("_refs", [])} - set(cache))
    print(f"  [openalex] resolving {len(todo)} referenced works", flush=True)
    sel = "id,display_name,publication_year,authorships,primary_location,biblio"
    for k in range(0, len(todo), 50):
        ids = "|".join(i.rsplit("/", 1)[-1] for i in todo[k:k + 50])
        try:
            d = _get(f"{API}?filter=openalex:{ids}&per-page=50&select={sel}")
            new = [{"id": w["id"], "cite": _cite(w)} for w in d["results"]]
        except Exception as e:
            print(f"  [openalex] refs batch failed: {e}", flush=True)
            if "budget" in str(e):
                return
            continue
        write_jsonl(cache_p, new, mode="a")
        if k % 5000 == 0:
            print(f"  [openalex] refs {k}/{len(todo)}", flush=True)
