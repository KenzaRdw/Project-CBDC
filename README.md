# Project-CBDC
A discourse analysis of central bank digital currencies (CBDCs) – M2 course project


## CBDC corpus for bibliometric analysis

Harvests everything about **central bank digital currencies** (CBDC, digital euro, e-CNY,
digital pound, e-krona, eNaira, mBridge, …, in EN/FR/ES/PT/DE/IT/ZH/JA/KO/RU) from:

| Source | What | Full content | Date precision |
|---|---|---|---|
| OpenAlex | journal articles, working papers (IMF, BIS, NBER, ECB…), books, chapters, preprints | abstract + open-access PDF | day (Jan 1 flagged as possibly year-only) |
| Semantic Scholar | same, incl. SSRN items | abstract + OA PDF | day / year |
| arXiv | preprints | abstract + PDF | day |
| HAL | French open archive (multilingual) | abstract + PDF, author keywords | day |
| DOAJ | open-access journals | abstract + author keywords | month |
| BIS central bankers' speeches | every central bank, 1996 → today | full text | day |
| ECB speeches | incl. originals in German/French/Italian… | full text | day |
| Authorities (hub crawl) | ECB, Commission, Council, EP, BoE, HM Treasury, Fed, White House, BIS, IMF, World Bank, Riksbank, BoJ, RBI, PBoC, BCB… press releases, reports, consultations | full text | day (page metadata) when published |
| Google News | worldwide press, 55 country editions × languages, 2000 → today | headline only | day |
| GDELT | worldwide press, 2017 → today (blocked by rate limits in this run) | article text (fetched) | day |

## Getting the data

`data/` is not in Git (933 MB; GitHub refuses files over 100 MB). Download `cbdc_data.tar.xz` from the
repository's **Releases** page (release `data-v1`) and unpack it in the project folder:

```bash
tar -xf cbdc_data.tar.xz      # recreates data/cbdc_corpus.csv, data/cache/ and data/fulltext/
```

That is enough to run `scripts/make_slides.py`. Only a full re-harvest needs `data/raw/`.
To publish a new version: `tar -cf - data/cbdc_corpus.csv data/cache data/fulltext | xz -T0 > cbdc_data.tar.xz`,
then attach the file to a new release.

## Run

```bash
.venv/bin/python run.py harvest            # all sources (or: harvest openalex bis ...)
.venv/bin/python run.py fulltext           # download & extract PDFs / articles (resumable)
.venv/bin/python run.py build              # merge, de-duplicate, enrich, export
```

`scripts/harvest_max.sh` chains everything (all sources, full text, GDELT) and rebuilds after each stage.
`.venv/bin/python scripts/make_slides.py` regenerates the presentation from the CSV (it overwrites hand edits to the `.tex`).
It needs `torch sentence-transformers bertopic scikit-learn transformers statsmodels mlx-lm networkx` in the venv
(Apple Silicon for `mlx-lm`). Models are downloaded on first use; heavy results are cached in `data/cache/`
(`embeddings.npz`, `tone.csv`, `stance_sample.csv`, the last one resumable).
`.venv/bin/python scripts/tag_economists.py` fetches each paper's OpenAlex field and author institutions
(`data/cache/openalex_fields.csv`): the talk keeps only texts by economists (research in economics or finance, or with a
central bank / IMF / BIS author; central bank speeches; documents of central banks and international institutions).
`.venv/bin/python scripts/stance_economists.py` tops up the LLM stance sample for that corpus.
`.venv/bin/python scripts/validate_coders.py` (about 1.5 h) scores every candidate coder against the hand-coded samples
and writes `presentation/validation_coders.csv`, which the slides read.

**Rate limits.**
* **OpenAlex.** Without a key you get a small free daily budget per IP, which resets at 00:00 UTC.
  A free key lifts this limit: get one at <https://openalex.org/settings/api>, then
  `export OPENALEX_API_KEY=...` before running.
* **GDELT.** Allows one request every 5 s, and throttles the whole IP for hours after a burst.
  The harvester paces itself and resumes from its checkpoint.
* **Google News** (press, `gnews` source) starts refusing requests (HTTP 503) after a few thousand calls;
  re-run `harvest gnews` later, it resumes from `raw/gnews_done.jsonl`.

Every step can be re-run; GDELT and full-text downloads resume where they stopped.
To add an authority, add a line to `SEEDS` in `cbdc_biblio/sources/authorities.py`.
To add search terms, edit `TERMS` / `QUERY_PHRASES` in `cbdc_biblio/config.py`.

## Outputs (`data/`)

* **`cbdc_corpus.csv`**: the master file, one row per document, UTF-8, all fields quoted.
  It includes the full text, so open it with pandas/R, not Excel (Excel cuts cells at 32,767 characters).
* **`fulltext/<record_id>.txt`**: full content of every document.
* `raw/*.jsonl`: per-source harvest, before merging.

## Master CSV columns

| Column | Meaning |
|---|---|
| `record_id` | stable id (`oa-…`, `bis-…`, `gn-…`, …); `merged_ids` lists the duplicates merged into it |
| `source_db` | database(s) the record came from |
| `source_type` | `academic`, `speech`, `official`, `press` |
| `document_type` | article, preprint, book-chapter, speech, press release, consultation, news article… |
| `title`, `subtitle` | title; subtitle, speech description or article lead |
| `authors`, `author_affiliations` | `;`-separated; affiliations as `Name (Institution)` |
| `issuer`, `issuer_type` | publishing organisation (journal publisher, central bank, ministry, news outlet) and its kind |
| `venue`, `publisher`, `country` | journal or series, publisher, country or countries |
| `publication_date`, `date_precision`, `year` | `YYYY-MM-DD` whenever possible; precision says `day`, `month`, `year` and where the date came from |
| `language`, `language_origin` | ISO 639-1 code of the document as retrieved; whether it comes from metadata or was detected |
| `doi`, `url`, `pdf_url`, `native_id` | identifiers and links |
| `abstract` | author abstract (academic) |
| `introduction`, `conclusion` | taken from the document's own section headings, or else from its opening/closing paragraphs (see `sections_method`) |
| `section_headings` | outline of the document (`|`-separated) |
| `keywords_author` | author or publisher keywords (HAL, DOAJ, page tags) |
| `keywords_indexed` | database keywords (OpenAlex keywords/topics, arXiv categories, S2 fields) |
| `keywords_auto` | YAKE keywords extracted from title + abstract + intro + conclusion (available for every record) |
| `cbdc_terms`, `cbdc_mentions` | which CBDC terms appear in the document, and how often |
| `relevance`, `theme_score` | how central CBDC (Central Bank Digital Currency, not homonyms) is to the text, 0–100: `main theme` (≥ 50), `secondary theme` (25–49), `mentioned only`, `ambiguous terms only`, `unverified (no text)` (press found by a CBDC search, headline does not say it), `dated before 2014` (would be a theme, but predates the concept: all checked to be homonyms or misdated), `off-topic` |
| `cited_by_count`, `references` | citations; reference list in Scopus style (for co-citation analysis) |
| `open_access`, `license` | OA status |
| `full_text_chars`, `full_text_path`, `full_text_status`, `full_text` | full content and how it was obtained |

## Caveats

* **Sites that refuse automated clients** (as of 2026-10-08): Council of the EU, Bank of England, IMF, RBA, RBNZ, ECCB, HKMA, Norges Bank, Danmarks Nationalbank, SARB, Bank of Korea, eNaira.
  Their speeches still come in through the BIS archive and their papers through OpenAlex and Semantic Scholar.

* **Language.** BIS speeches are mostly English versions of speeches that may first have been given in another language; `language` is the language of the text retrieved.
* **Dates.**
  * Press dates are the article's own metadata date when the page could be fetched; otherwise they are GDELT's first-seen date, which is usually the same day.
  * OpenAlex gives `YYYY-01-01` for some works where only the year is known; these are flagged in `date_precision`.
* **Full text** covers only open-access or freely reachable documents; paywalled papers keep their abstract.
* **Filtering.** Every record is kept. The analyses use `relevance in ("main theme", "secondary theme")`.
  Validated on 138 texts read blind (`presentation/validation_relevance.csv`): precision ≈ 0.89, recall ≈ 0.89;
  `main theme` is reliable (29/30), `secondary theme` is not (11/29), so check results on `main theme` alone.
* **Press dates.** The news search favours recent articles: press counts by year reflect coverage, not only attention.
* **Final state (2026-10-11).** The press harvest and the extra full-text downloads were stopped before the end
  (Google News rate limit); both resume if re-run.

## Analyses

| Question | Method used | Alternatives tested |
|---|---|---|
| What | BERTopic (SBERT + UMAP + HDBSCAN, c-TF-IDF labels), co-word / Callon map, covariate logit on topic shares | LDA / STM |
| Who | co-citation, co-authorship, Lotka, Bradford, central-bank affiliations, log-odds with Dirichlet prior | TF-IDF |
| Why | motivation codebook (validated), covariate logit | SBERT zero-shot / classifier, NLI, local LLMs 3B / 7B |
| How | FinBERT tone of CBDC sentences; stance: levels from hand coding, changes from a local LLM (Qwen2.5 7B) on a stratified sample | NLI, FinBERT, SBERT classifier |
| When | c-TF-IDF by period, Kleinberg bursts, event study with placebo dates | Poisson z-test |

* `cbdc_biblio/analysis.py`: networks, thematic map, log-odds, codebook, event study + placebo, laws, bursts,
  central-bank authorship, covariate-adjusted effects (`adjusted`).
* `cbdc_biblio/nlp.py`: embeddings, BERTopic, c-TF-IDF, SBERT coders, FinBERT tone, LLM stance on a stratified sample.
* Robustness (slide): main theme only, English only, journal articles only, without the current year, high-recall coding.

## Presentation (`presentation/`)

* `cbdc_corpus.pdf`: the slides (groundwork of the paper, about 23 minutes), built from the CSV.
* Hand-coded validation samples (one coder, AI-assisted; a second human coder is still needed):
  * `validation_relevance.csv`: relevance filter, 140 texts stratified by label;
  * `validation_motivations_dev.csv` / `_test.csv`: motivation codebook, 100 development and 50 held-out texts;
  * `validation_tone.csv`: FinBERT tone, 90 sentences (30 per predicted label);
  * `validation_stance.csv` (+ `validation_stance_llm.csv`, the LLM's labels on the same texts): stance, 150 texts stratified by actor;
  * `validation_coders.csv`: every coder's precision / recall / κ against these samples.
* `second_coder_sheet.csv`: 200 items (60 relevance, 50 motivations, 50 stance, 40 tone) for a second, human coder;
  definitions as in the slides (relevance: CBDC is a main or secondary theme; stance: favourable / neutral / critical
  towards CBDCs; tone: of the sentence). Compare with the first coding to report inter-coder κ.
