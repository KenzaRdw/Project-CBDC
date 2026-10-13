#!/bin/zsh
# Largest possible corpus: every source, every search phrase, as far back as each source goes.
# Everything is kept; the `relevance` column is used to reduce the set afterwards.
# Rebuilds data/cbdc_corpus.csv and the slides after each stage, so partial results are usable.
# Usage: OPENALEX_API_KEY=... nohup scripts/harvest_max.sh > logs/harvest_max.log 2>&1 &
cd "${0:A:h}/.."
mkdir -p logs
PY=.venv/bin/python
rebuild() { $PY -u run.py build && $PY scripts/make_slides.py; }

echo "== $(date) press (Google News, 2000 -> today) in parallel"
$PY -u run.py harvest gnews > logs/harvest_gnews.log 2>&1 &
GN=$!

echo "== $(date) academic sources"
$PY -u run.py harvest openalex semanticscholar arxiv hal doaj
rebuild

echo "== $(date) full text of academic records"
while pgrep -f "run.py fulltext" > /dev/null; do sleep 300; done   # let any running download finish first
$PY -u run.py fulltext --workers 12 --skip-offtopic --sources OpenAlex "Semantic Scholar" arXiv HAL DOAJ

echo "== $(date) waiting for the press harvest"
wait $GN
tail -3 logs/harvest_gnews.log
rebuild

echo "== $(date) GDELT (press, 2017 -> today, often throttled)"
$PY -u run.py harvest gdelt
$PY -u run.py fulltext --workers 16 --sources GDELT
rebuild
echo "== $(date) done"
