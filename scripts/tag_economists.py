"""Fetch the OpenAlex field and author institutions of every on-theme academic record.

Writes data/cache/openalex_fields.csv (one row per record), read by analysis.economists().
Resumable: records already in the cache are skipped.
"""
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/cache/openalex_fields.csv"
KEY = os.environ.get("OPENALEX_API_KEY", "")
RELEVANT = ["main theme", "secondary theme"]


def openalex_id(row):
    for s in (row.native_id, row.merged_ids):
        m = re.search(r"\bW\d{4,}", str(s))
        if m:
            return m.group(0)
    return None


def fetch(ids):
    url = (f"https://api.openalex.org/works?filter=openalex:{'|'.join(ids)}&per-page=50"
           "&select=id,primary_topic,topics,authorships" + (f"&api_key={KEY}" if KEY else ""))
    for attempt in range(5):
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            return r.json()["results"]
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()


def row_of(w):
    pt = w.get("primary_topic") or {}
    insts = [i for a in w.get("authorships") or [] for i in a.get("institutions") or []]
    return {"openalex_id": w["id"].rsplit("/", 1)[-1],
            "field": (pt.get("field") or {}).get("display_name", ""),
            "subfield": (pt.get("subfield") or {}).get("display_name", ""),
            "topic_fields": "; ".join(dict.fromkeys((t.get("field") or {}).get("display_name", "")
                                                    for t in w.get("topics") or [])),
            "institutions": "; ".join(dict.fromkeys(i.get("display_name", "") for i in insts)),
            "institution_types": "; ".join(dict.fromkeys(i.get("type") or "" for i in insts))}


def main():
    d = pd.read_csv(ROOT / "data/cbdc_corpus.csv", low_memory=False,
                    usecols=["record_id", "source_type", "relevance", "native_id", "merged_ids"])
    a = d[(d.source_type == "academic") & d.relevance.isin(RELEVANT)].copy()
    a["openalex_id"] = a.apply(openalex_id, axis=1)
    a = a[a.openalex_id.notna()]
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["openalex_id"])
    todo = sorted(set(a.openalex_id) - set(done.openalex_id))
    print(f"{len(a)} records with an OpenAlex id, {len(todo)} to fetch", file=sys.stderr)
    rows = []
    for k in range(0, len(todo), 50):
        rows += [row_of(w) for w in fetch(todo[k:k + 50])]
        time.sleep(0.15)
    new = pd.DataFrame(rows)
    allrows = pd.concat([done.drop(columns="record_id", errors="ignore"), new], ignore_index=True).drop_duplicates("openalex_id")
    links = a[["record_id", "openalex_id"]].merge(allrows, on="openalex_id", how="inner")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    links.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(links)} records)", file=sys.stderr)


if __name__ == "__main__":
    main()
