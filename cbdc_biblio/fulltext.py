"""Download and extract full content (PDF or HTML) for records that lack it.

Results are cached in data/raw/fulltext_index.jsonl (record_id -> path, status, page
metadata such as the publication date / author / site name found in the page), so
the step is resumable and can be re-run at any time.
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from .common import (date_and_precision, now_iso, raw_path, read_jsonl, save_fulltext, session,
                     write_jsonl)

MAX_BYTES = 40 * 1024 * 1024
_local = threading.local()
_lock = threading.Lock()


def _sess():
    if not hasattr(_local, "s"):
        _local.s = session(retries=2, backoff=1.0)
        _local.s.headers.update({"Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
                                 "Accept-Language": "en,fr;q=0.8,*;q=0.5"})
    return _local.s


def _pdf_text(data):
    import pymupdf
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        meta = doc.metadata or {}
        text = "\n\n".join(page.get_text("text") for page in doc)
    d = meta.get("creationDate", "")  # D:20230115...
    date = f"{d[2:6]}-{d[6:8]}-{d[8:10]}" if d.startswith("D:") and len(d) >= 10 else ""
    return text, {"date": date, "author": meta.get("author", ""), "title": meta.get("title", "")}


def _html_text(html, url):
    import trafilatura
    doc = trafilatura.bare_extraction(html, url=url, with_metadata=True, include_comments=False,
                                      include_tables=True, favor_recall=True)
    if not doc:
        return "", {}
    get = (lambda k: getattr(doc, k, None)) if not isinstance(doc, dict) else doc.get
    meta = {k: get(k) or "" for k in ("date", "author", "sitename", "title", "description",
                                      "language", "categories", "tags")}
    meta = {k: "; ".join(v) if isinstance(v, list) else str(v) for k, v in meta.items()}
    return get("text") or "", meta


def extract_url(url):
    """Fetch url and return {'text', 'status', 'kind', 'meta'}."""
    if not url:
        return {"status": "no url"}
    try:
        r = _sess().get(url, timeout=(15, 60), stream=True, allow_redirects=True)
        if r.status_code >= 400:
            return {"status": f"http {r.status_code}"}
        data = b""
        for chunk in r.iter_content(1 << 16):
            data += chunk
            if len(data) > MAX_BYTES:
                return {"status": "too large"}
        ctype = r.headers.get("content-type", "").lower()
        if "pdf" in ctype or data[:5] == b"%PDF-":
            text, meta = _pdf_text(data)
            kind = "pdf"
        else:
            text, meta = _html_text(data.decode(r.encoding or "utf-8", errors="replace"), r.url)
            kind = "html"
        if len(text.strip()) < 300:
            return {"status": f"too short ({kind})", "kind": kind, "meta": meta}
        return {"text": text, "status": "ok", "kind": kind, "meta": meta, "final_url": r.url}
    except Exception as e:
        return {"status": f"error: {type(e).__name__}"}


def candidates(rec):
    urls = [rec.get("pdf_url"), rec.get("url")]
    if rec.get("doi") and not rec.get("url"):
        urls.append(f"https://doi.org/{rec['doi']}")
    return [u for u in dict.fromkeys(u for u in urls if u)]


def _offtopic_ids():
    """Records the last build scored off-topic although it had their abstract: full text would not change that."""
    import csv
    from .config import DATA
    p = DATA / "cbdc_corpus.csv"
    if not p.exists():
        return set()
    csv.field_size_limit(1 << 30)
    out = set()
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("relevance") == "off-topic" and row.get("abstract"):
                out.update((row.get("merged_ids") or row["record_id"]).split("; "))
    return out


def run(records, workers=16, sources=None, retry_failed=False, skip_offtopic=False):
    idx_p = raw_path("fulltext_index")
    done = {r["record_id"]: r for r in read_jsonl(idx_p)}
    skip = _offtopic_ids() if skip_offtopic else set()
    todo = [r for r in records
            if not r.get("full_text_path")
            and r["record_id"] not in skip
            and (sources is None or r.get("source_db") in sources)
            and (r["record_id"] not in done or (retry_failed and done[r["record_id"]]["status"] != "ok"))
            and candidates(r)]
    # spread load across hosts so a single site is not hammered
    todo.sort(key=lambda r: urlparse(candidates(r)[0]).netloc)
    todo = todo[::2] + todo[1::2]
    print(f"  [fulltext] {len(todo)} records to fetch ({len(done)} already in index)", flush=True)

    def work(rec):
        res = {}
        for u in candidates(rec):
            res = extract_url(u)
            if res.get("status") == "ok":
                break
        entry = {"record_id": rec["record_id"], "status": res.get("status", "?"),
                 "kind": res.get("kind", ""), "checked_at": now_iso()}
        if res.get("status") == "ok":
            entry["path"], entry["chars"] = save_fulltext(rec["record_id"], res["text"])
        meta = res.get("meta") or {}
        d, prec = date_and_precision(meta.get("date"))
        entry.update({"page_date": d, "page_date_precision": prec,
                      **{f"page_{k}": v for k, v in meta.items() if k != "date" and v}})
        return entry

    ok = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            e = f.result()
            ok += e["status"] == "ok"
            with _lock:
                write_jsonl(idx_p, [e], mode="a")
            if i % 250 == 0:
                print(f"  [fulltext] {i}/{len(todo)} done, {ok} ok", flush=True)
    print(f"  [fulltext] finished: {ok}/{len(todo)} extracted", flush=True)
