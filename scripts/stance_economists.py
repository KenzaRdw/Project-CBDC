"""Top up the LLM stance sample so that every actor x period cell of the economists' corpus has up to 100 texts."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbdc_biblio import analysis as A  # noqa: E402
from cbdc_biblio import nlp  # noqa: E402

d = A.load(ROOT / "data/cbdc_corpus.csv")
d = d[d.relevance.isin(A.THEME)]
d = d[A.economists(d, pd.read_csv(ROOT / "data/cache/openalex_fields.csv"))]
s = nlp.stance_sample(d, ROOT, ROOT / "data/cache/stance_sample.csv")
print(s.groupby(["actor", "period"]).size())
