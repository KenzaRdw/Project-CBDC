"""Worldwide press coverage via the GDELT DOC 2.0 API (65 languages, machine-translated search).

Strategy: get the article-volume timeline, cut the period into windows that hold
fewer than 250 articles (the per-call cap), and split any window that still hits the
cap. The API allows one call every ~5 s (we use 10 s), so a full 2017->today harvest takes a while;
progress is checkpointed and the harvest resumes where it stopped.
"""
import json
import time
from datetime import datetime, timedelta, timezone

from ..common import GDELT_LANG, date_and_precision, make_id, new_record, raw_path, read_jsonl, session, write_jsonl

API = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = ('("central bank digital currency" OR "central bank digital currencies" OR CBDC OR '
         '"digital euro" OR "e-CNY" OR "digital yuan" OR "digital renminbi" OR "digital pound" OR '
         '"e-krona" OR eNaira OR "digital rupee" OR "e-rupee" OR mBridge OR "digital ruble" OR "Drex")')
FMT = "%Y%m%d%H%M%S"
S = session(retries=0)
_last = [0.0]


def _call(params):
    for attempt in range(30):
        wait = 10.0 - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
        try:
            r = S.get(API, params={"query": QUERY, "format": "json", **params}, timeout=90)
            txt = r.text.strip()
            if r.status_code == 200 and txt.startswith("{"):
                return json.loads(txt, strict=False)
            if r.status_code == 200 and not txt:
                return {}
        except Exception:
            pass
        time.sleep(min(60 * (attempt + 1), 600))  # GDELT throttles the whole IP after bursts
    raise RuntimeError("GDELT keeps refusing requests")


def _volume(start, end):
    d = _call({"mode": "timelinevolraw", "timelinesmooth": 0,
               "startdatetime": start.strftime(FMT), "enddatetime": end.strftime(FMT)})
    rows = (d.get("timeline") or [{}])[0].get("data") or []
    return [(datetime.strptime(x["date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc), x["value"]) for x in rows]


def _windows(start, end, cap=200):
    """Group timeline steps into windows each expected to hold <= cap articles."""
    out = []
    # query the timeline year by year so the resolution stays daily
    y = start
    while y < end:
        y2 = min(datetime(y.year + 1, 1, 1, tzinfo=timezone.utc), end)
        steps = _volume(y, y2)
        w0, acc = y, 0
        for i, (t, v) in enumerate(steps):
            t_next = steps[i + 1][0] if i + 1 < len(steps) else y2
            if acc + v > cap and acc:
                out.append((w0, t))
                w0, acc = t, 0
            acc += v
            if i + 1 == len(steps):
                out.append((w0, t_next))
        if not steps:
            out.append((y, y2))
        print(f"  [gdelt] planned {y.year}: {sum(v for _, v in steps)} articles", flush=True)
        y = y2
    return out


def _articles(a, b, depth=0):
    d = _call({"mode": "artlist", "maxrecords": 250, "sort": "dateasc",
               "startdatetime": a.strftime(FMT), "enddatetime": b.strftime(FMT)})
    arts = d.get("articles") or []
    if len(arts) >= 250 and (b - a) > timedelta(hours=1) and depth < 8:
        mid = a + (b - a) / 2
        return _articles(a, mid, depth + 1) + _articles(mid, b, depth + 1)
    return arts


def to_record(x):
    date, prec = date_and_precision(x.get("seendate"))
    return new_record(
        record_id=make_id("news", x["url"]), source_db="GDELT", source_type="press",
        document_type="news article", title=x.get("title"), issuer=x.get("domain"),
        issuer_type="news outlet", venue=x.get("domain"), country=x.get("sourcecountry"),
        publication_date=date, date_precision=prec + " (first seen by GDELT)", year=date[:4],
        language=GDELT_LANG.get(x.get("language"), x.get("language")),
        language_origin="GDELT metadata", url=x["url"], native_id=x["url"],
    )


def harvest(start="2017-01-01", end=None):
    start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc)
    out_p, ck_p = raw_path("gdelt"), raw_path("gdelt_checkpoint")
    seen = {r["native_id"] for r in read_jsonl(out_p)}
    ck = read_jsonl(ck_p)
    if ck:
        start = max(start, datetime.fromisoformat(ck[-1]["done_until"]))
        print(f"  [gdelt] resuming from {start:%Y-%m-%d %H:%M}", flush=True)
    wins = _windows(start, end)
    print(f"  [gdelt] {len(wins)} windows to fetch", flush=True)
    for i, (a, b) in enumerate(wins, 1):
        new = [to_record(x) for x in _articles(a, b) if x.get("url") and x["url"] not in seen]
        seen.update(r["native_id"] for r in new)
        write_jsonl(out_p, new, mode="a")
        write_jsonl(ck_p, [{"done_until": b.isoformat()}], mode="a")
        if i % 20 == 0 or i == len(wins):
            print(f"  [gdelt] window {i}/{len(wins)} ({a:%Y-%m-%d}) total {len(seen)}", flush=True)
    return len(seen)
