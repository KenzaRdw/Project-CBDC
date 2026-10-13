"""CBDC corpus pipeline.

    python run.py harvest [source ...]   # default: all sources
    python run.py fulltext [--workers N] [--sources OpenAlex GDELT ...] [--retry-failed] [--skip-offtopic]
    python run.py build

Sources: openalex semanticscholar arxiv hal doaj bis ecb authorities gdelt gnews
"""
import argparse
import time

from cbdc_biblio import build, fulltext
from cbdc_biblio.common import raw_path, read_jsonl
from cbdc_biblio.sources import academic_other, authorities, gdelt, gnews, openalex, speeches

HARVESTERS = {
    "openalex": openalex.harvest,
    "semanticscholar": academic_other.semanticscholar,
    "arxiv": academic_other.arxiv,
    "hal": academic_other.hal,
    "doaj": academic_other.doaj,
    "bis": speeches.bis,
    "ecb": speeches.ecb,
    "authorities": authorities.harvest,
    "gdelt": gdelt.harvest,
    "gnews": gnews.harvest,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("sources", nargs="*", default=list(HARVESTERS))
    f = sub.add_parser("fulltext")
    f.add_argument("--workers", type=int, default=16)
    f.add_argument("--sources", nargs="*", help="source_db values, e.g. OpenAlex GDELT")
    f.add_argument("--retry-failed", action="store_true")
    f.add_argument("--skip-offtopic", action="store_true", help="skip records already scored off-topic from their abstract")
    sub.add_parser("build")
    a = ap.parse_args()

    if a.cmd == "harvest":
        for s in a.sources:
            t = time.time()
            print(f"== {s}", flush=True)
            try:
                n = HARVESTERS[s]()
                print(f"== {s}: {n} records in {time.time() - t:.0f}s", flush=True)
            except Exception as e:
                print(f"== {s} FAILED: {type(e).__name__}: {e}", flush=True)
    elif a.cmd == "fulltext":
        recs = [r for name in build.SOURCE_FILES for r in read_jsonl(raw_path(name))]
        fulltext.run(recs, workers=a.workers, sources=set(a.sources) if a.sources else None,
                     retry_failed=a.retry_failed, skip_offtopic=a.skip_offtopic)
    else:
        build.build()


if __name__ == "__main__":
    main()
