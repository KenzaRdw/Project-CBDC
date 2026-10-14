"""Text-as-data analyses (texts with CBDC as main or secondary theme).

  embed            multilingual sentence embeddings (SBERT, Reimers & Gurevych 2019), cached
  topics           BERTopic (Grootendorst 2022): embeddings -> UMAP -> HDBSCAN, labelled by class-based TF-IDF
  period_terms     class-based TF-IDF with periods as classes: what is new in each period
  validate_motives the motivation codebook against SBERT alternatives, on the hand-coded test set
  tone             FinBERT (Araci 2019) on the sentences that mention CBDCs
"""
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from .analysis import MOTIVES, MOTIVE_RE, PERIODS, STOP, TOPIC, _motive_text
from .config import STRONG, TERM_RE

SBERT = "paraphrase-multilingual-MiniLM-L12-v2"
FINBERT = "ProsusAI/finbert"
NON_PRESS = ["academic", "speech", "official"]


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SBERT, device=_device())


def doc_text(r, root):
    """Title + abstract (academic), title + opening of the text (speeches, official), headline (press)."""
    title = r.get("title") if isinstance(r.get("title"), str) else ""
    if r["source_type"] == "press":
        return title
    if isinstance(r.get("abstract"), str) and len(r["abstract"]) > 100:
        return f"{title}. {r['abstract']}"
    return f"{title}. {_motive_text(r, root)[:1500]}"


def embed(d, root, cache):
    """One embedding per record, cached on disk by record_id."""
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        have = dict(zip(z["ids"], z["emb"]))
    todo = [r for r in d.to_dict("records") if r["record_id"] not in have]
    if todo:
        emb = _sbert().encode([doc_text(r, root) for r in todo], batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True)
        have.update(zip([r["record_id"] for r in todo], emb))
        np.savez(cache, ids=np.array(list(have)), emb=np.stack(list(have.values())))
    return np.stack([have[i] for i in d.record_id])


# ----------------------------------------------------------------- class-based TF-IDF

# non-English function words in mixed-language abstracts, and repository boilerplate ("Researcher", Zenodo)
FOREIGN_STOP = {"dan", "yang", "dalam", "untuk", "dengan", "pada", "ini", "atau", "des", "les", "der", "die", "und",
                "researcher", "creators", "description",
                "ugc", "care", "listed", "remarks", "conference", "speech", "ladies", "gentlemen"}


def _words(text):
    return [w for w in re.findall(r"[a-z][a-z\-]{2,}", text.lower())
            if w not in STOP and w not in TOPIC and w not in FOREIGN_STOP and not w.endswith("-")]


def ctfidf(class_texts, k=6, min_df=5, max_df=0.15):
    """Class-based TF-IDF (Grootendorst 2022): all documents of a class are joined into one,
        W(t, c) = tf(t, c) / |c| * log(1 + A / f(t))
    tf(t, c) count of term t in class c, |c| words in c, f(t) count of t over all classes,
    A average number of words per class. Unigrams and bigrams. Terms found in more than
    max_df of all documents (money, policy, financial...) are corpus stop words.
    """
    docs = {c: [_words(t) for t in texts] for c, texts in class_texts.items()}
    df = Counter(w for ds in docs.values() for ws in ds for w in set(ws))
    n_docs = sum(len(ds) for ds in docs.values()) or 1
    common = {w for w, v in df.items() if v / n_docs > max_df}
    counts = {}
    for c, ds in docs.items():
        cnt = Counter()
        for w in ds:
            w = [x for x in w if x not in common]
            cnt.update(w)
            cnt.update(f"{a} {b}" for a, b in zip(w, w[1:]) if a != b)
        counts[c] = cnt
    f = sum(counts.values(), Counter())
    a = np.mean([sum(v for t, v in c.items() if " " not in t) for c in counts.values()]) or 1
    out = {}
    for c, cnt in counts.items():
        size = sum(cnt.values()) or 1
        w = {t: v / size * math.log(1 + a / f[t]) for t, v in cnt.items() if f[t] >= min_df}
        picked = []
        for t in sorted(w, key=lambda t: -w[t]):  # skip a word already inside a picked bigram, and the reverse
            if any(set(t.split()) & set(p.split()) for p in picked):
                continue
            picked.append(t)
            if len(picked) == k:
                break
        out[c] = picked
    return out


# ----------------------------------------------------------------- topics (BERTopic)

def topics(d, root, cache, min_size=60, selection="eom"):
    """BERTopic on research, speeches and official documents (headlines are too short).

    Clusters are found in all languages (the embedding space is multilingual); labels use
    the English documents of each topic.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    t = d[d.source_type.isin(NON_PRESS)].reset_index(drop=True)
    emb = embed(t, root, cache)
    texts = [doc_text(r, root) for r in t.to_dict("records")]
    model = BERTopic(umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42),
                     hdbscan_model=HDBSCAN(min_cluster_size=min_size, min_samples=10, metric="euclidean",
                                           cluster_selection_method=selection, prediction_data=True),
                     calculate_probabilities=False)
    topic, _ = model.fit_transform(texts, emb)
    n_outliers = int(np.sum(np.array(topic) == -1))
    topic = model.reduce_outliers(texts, topic, strategy="embeddings", embeddings=emb)
    t["topic"] = topic
    en = t[t.language == "en"]
    labels = ctfidf({k: list(en[en.topic == k].apply(lambda r: doc_text(r, root), axis=1))
                     for k in sorted(set(topic))}, k=5)
    rows = []
    for k, g in t.groupby("topic"):
        row = {"topic": k, "terms": labels.get(k, []), "n": len(g),
               "academic": 100 * (g.source_type == "academic").mean()}
        for lbl, y0, y1 in PERIODS:
            p = t[(t.year >= y0) & (t.year <= y1)]
            row[lbl] = 100 * ((p.topic == k).sum()) / max(len(p), 1)
        rows.append(row)
    tab = pd.DataFrame(rows).sort_values("n", ascending=False)
    actors = {a: 100 * t[t.source_type.replace({"official": "speech"}) == a].topic.value_counts(normalize=True)
              for a in ("academic", "speech")}
    return {"table": tab, "n_docs": len(t), "n_outliers": n_outliers, "assign": t[["record_id", "topic"]],
            "actors": actors}


def period_terms(d, root, k=8):
    """What each period talks about that the others do not: c-TF-IDF with periods as classes."""
    e = d[(d.language == "en") & d.source_type.isin(NON_PRESS)]
    classes = {lbl: [doc_text(r, root) for r in e[(e.year >= y0) & (e.year <= y1)].to_dict("records")]
               for lbl, y0, y1 in PERIODS}
    return ctfidf(classes, k=k, min_df=10)


# ----------------------------------------------------------------- motivations, by meaning

# short definitions, written once from the codebook; the multilingual model matches them in any language
PROTOTYPES = {
    "Decline of cash": ["The use of cash is declining as people pay digitally.",
                        "Cash is disappearing and the public may lose access to central bank money."],
    "Financial inclusion": ["A CBDC could give unbanked people access to financial services.",
                            "Digital currency can improve financial inclusion for the underserved population."],
    "Monetary sovereignty": ["Private stablecoins and big tech currencies threaten monetary sovereignty.",
                             "Central banks must keep control of money against private cryptocurrencies."],
    "Payment efficiency": ["A digital currency makes payments faster, cheaper and more efficient.",
                           "CBDC can reduce transaction costs and modernise the payment system."],
    "Cross-border payments": ["CBDCs can improve cross-border payments and remittances.",
                              "International settlement between countries could be faster with digital currencies."],
    "Monetary policy": ["A CBDC affects the transmission of monetary policy and interest rates.",
                        "An interest-bearing digital currency gives the central bank a new policy tool."],
    "Financial stability": ["A CBDC could cause bank runs and disintermediation of commercial banks.",
                            "Deposits may move from banks to the central bank, threatening financial stability."],
    "Privacy and surveillance": ["A digital currency raises concerns about privacy and state surveillance.",
                                 "Users' payment data must be protected; anonymity is limited."],
    "Geopolitics and sanctions": ["CBDCs could challenge the dominance of the US dollar and help avoid sanctions.",
                                  "Digital currencies are part of geopolitical competition between great powers."],
    "Technology (DLT, programmability)": ["The digital currency is built on blockchain and distributed ledger technology.",
                                          "Smart contracts make the digital money programmable."],
}


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?。;])\s+", text) if len(s) > 25][:40]


def motive_similarity(texts, model=None):
    """For each text and motivation: highest cosine similarity between a sentence and a definition."""
    model = model or _sbert()
    names = list(PROTOTYPES)
    proto = model.encode([p for k in names for p in PROTOTYPES[k]], normalize_embeddings=True)
    owner = np.repeat(np.arange(len(names)), [len(PROTOTYPES[k]) for k in names])
    sents = [_sentences(t) or [t[:500]] for t in texts]
    flat = [s for ss in sents for s in ss]
    emb = model.encode(flat, batch_size=128, normalize_embeddings=True, show_progress_bar=len(flat) > 5000)
    sim = emb @ proto.T
    best = np.stack([sim[:, owner == k].max(axis=1) for k in range(len(names))], axis=1)
    out, i = [], 0
    for ss in sents:
        out.append(best[i:i + len(ss)].max(axis=0))
        i += len(ss)
    return pd.DataFrame(out, columns=names)


def _prf(y, p):
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    n = len(y)
    po = (y == p).mean()
    pe = (y.mean() * p.mean()) + ((1 - y.mean()) * (1 - p.mean()))
    return {"precision": tp / (tp + fp) if tp + fp else float("nan"), "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0, "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan"), "n": n}


def _codebook(texts):
    return {k: pd.Series(texts).str.contains(MOTIVE_RE[k]).values for k in MOTIVES}


def _classifiers(dev, model, C=10.0):
    """One logistic regression per motivation on the text embedding, trained on the hand-coded
    development set (class weights balanced: positives are rare)."""
    from sklearn.linear_model import LogisticRegression
    x = model.encode(list(dev.text), normalize_embeddings=True)
    return {k: LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(x, dev[f"manual: {k}"].astype(bool))
            for k in MOTIVES}


def validate_motives(dev, test):
    """Four ways of coding motivations, all scored on the 50 held-out texts (500 text x motivation pairs).

      codebook     the regular expressions (tuned on the development set)
      definitions  zero-shot: a sentence within tau of a motivation's definition (tau chosen on dev)
      classifier   logistic regression on SBERT embeddings, trained on the 100 development texts
      union        codebook OR classifier
    """
    model = _sbert()
    names = list(MOTIVES)
    y = np.concatenate([test[f"manual: {k}"].values.astype(bool) for k in names])
    flat = lambda pred: np.concatenate([np.asarray(pred[k], bool) for k in names])
    s_dev, s_test = motive_similarity(list(dev.text), model), motive_similarity(list(test.text), model)
    y_dev = np.concatenate([dev[f"manual: {k}"].values.astype(bool) for k in names])
    tau = max(np.arange(0.40, 0.80, 0.01), key=lambda t: _prf(y_dev, flat({k: s_dev[k] >= t for k in names}))["f1"])
    clf = _classifiers(dev, model)
    x_test = model.encode(list(test.text), normalize_embeddings=True)
    cb = _codebook(test.text)
    learned = {k: clf[k].predict(x_test) for k in names}
    res = {"Codebook (regex)": _prf(y, flat(cb)),
           "SBERT, definitions (zero-shot)": _prf(y, flat({k: s_test[k] >= tau for k in names})),
           "SBERT, classifier (100 labels)": _prf(y, flat(learned)),
           "Codebook OR classifier": _prf(y, flat({k: cb[k] | learned[k] for k in names}))}
    return {"tau": float(tau), "overall": pd.DataFrame(res).T}


def motivations_union(d, root, dev):
    """Robustness: motivation shares with the high-recall coding (codebook OR SBERT classifier)."""
    model = _sbert()
    clf = _classifiers(dev, model)
    rows = []
    for r in d[d.source_type.isin(NON_PRESS)].to_dict("records"):
        text = _motive_text(r, root)
        if len(text) >= 200:
            rows.append({"record_id": r["record_id"], "source_type": r["source_type"], "year": r["year"],
                         "language": r.get("language"), "text": text[:1500]})
    m = pd.DataFrame(rows)
    x = model.encode(list(m.text), batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    cb = _codebook(m.text)
    for k in MOTIVES:
        m[k] = cb[k] | clf[k].predict(x).astype(bool)
    ks = list(MOTIVES)
    return {"by_period": pd.DataFrame({lbl: m[(m.year >= y0) & (m.year <= y1)][ks].mean().mul(100)
                                       for lbl, y0, y1 in PERIODS}), "n": len(m), "table": m.drop(columns="text")}


# ----------------------------------------------------------------- tone of CBDC sentences

def _strong(s):
    return any(TERM_RE[k].search(s) for k in STRONG)


def cbdc_sentences(d, root, per_doc=30):
    """English sentences that mention a CBDC: abstracts (academic), full texts (speeches,
    official), headlines (press)."""
    out = []
    for r in d[d.language == "en"].to_dict("records"):
        st = r["source_type"]
        if st == "press":
            texts = [r.get("title") or ""]
        elif st == "academic":
            texts = _sentences(r.get("abstract") if isinstance(r.get("abstract"), str) else "")
        else:
            p = r.get("full_text_path")
            body = (root / p).read_text(encoding="utf-8")[:60000] if isinstance(p, str) and (root / p).exists() else ""
            texts = [s for s in re.split(r"(?<=[.!?])\s+", body) if 25 < len(s) < 600][:2000]
        picked = [s.strip() for s in texts if isinstance(s, str) and _strong(s)][:per_doc]
        out += [{"record_id": r["record_id"], "source_type": st, "date": r.get("publication_date"),
                 "year": r["year"], "sentence": s} for s in picked]
    return pd.DataFrame(out)


def tone(d, root, cache):
    """FinBERT probabilities per sentence; tone = p(positive) - p(negative), in [-1, 1].
    Cached: sentences are scored once."""
    if cache.exists():
        return pd.read_parquet(cache) if cache.suffix == ".parquet" else pd.read_csv(cache)
    from transformers import pipeline
    s = cbdc_sentences(d, root)
    clf = pipeline("text-classification", model=FINBERT, device=_device(), top_k=None, truncation=True, max_length=128)
    probs = clf(list(s.sentence), batch_size=64)
    for lab in ("positive", "negative", "neutral"):
        s[lab] = [next(x["score"] for x in p if x["label"] == lab) for p in probs]
    s["label"] = s[["positive", "negative", "neutral"]].idxmax(axis=1)
    s["tone"] = s.positive - s.negative
    s.to_csv(cache, index=False)
    return s


# ----------------------------------------------------------------- stance (local LLM on a stratified sample)

LLM = "mlx-community/Qwen2.5-7B-Instruct-4bit"
STANCE_SYSTEM = ("You read texts about central bank digital currencies (CBDC). What is the author's stance towards CBDCs? "
                 "Answer with one word: favourable (supports or presents them positively), critical (opposes them or "
                 "stresses their risks and problems), or neutral (descriptive or balanced).")
STANCES = ["critical", "neutral", "favourable"]


def stance_unit(r, root):
    """What the coder reads: the headline (press), title + abstract (research), or title + the first five
    sentences that name a CBDC (speeches, official documents)."""
    title = r.get("title") if isinstance(r.get("title"), str) else ""
    if r["source_type"] == "press":
        return title
    if r["source_type"] == "academic" and isinstance(r.get("abstract"), str) and len(r["abstract"]) > 100:
        return f"{title}. {r['abstract'][:1200]}"
    p = r.get("full_text_path")
    body = (root / p).read_text(encoding="utf-8")[:60000] if isinstance(p, str) and (root / p).exists() else ""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", body) if 25 < len(x) < 500 and _strong(x)]
    return f"{title}. {' '.join(sents[:5])}"[:1200]


def stance_sample(d, root, cache, per_cell=100, seed=5):
    """Stratified random sample (actor x period), coded by a local LLM; resumable (one CSV line per text)."""
    from mlx_lm import generate, load
    f = d.copy()
    f["actor"] = f.source_type.replace({"official": "speech"})
    f["period"] = None
    for lbl, y0, y1 in PERIODS:
        f.loc[(f.year >= y0) & (f.year <= y1), "period"] = lbl
    f = f.dropna(subset=["period"])
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["record_id"])
    # texts already coded were drawn at random, so they stay a random sample of any subset of the corpus;
    # each cell is topped up with a random draw from the texts not coded yet
    f["coded"] = f.record_id.isin(done.record_id)
    f = f.sample(frac=1, random_state=seed).sort_values("coded", ascending=False, kind="stable")
    sample = f.groupby(["actor", "period"], group_keys=False).head(per_cell)
    todo = sample[~sample.record_id.isin(done.record_id)]
    if len(todo):
        model, tok = load(LLM)
        with open(cache, "a", encoding="utf-8") as out:
            if not len(done):
                out.write("record_id,actor,period,stance\n")
            for r in todo.to_dict("records"):
                msgs = [{"role": "system", "content": STANCE_SYSTEM},
                        {"role": "user", "content": "Text:\n" + stance_unit(r, root) + "\n\nAnswer:"}]
                a = generate(model, tok, prompt=tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                             max_tokens=5, verbose=False).lower()
                lab = "favourable" if "favour" in a else "critical" if "critic" in a else "neutral"
                out.write(f"{r['record_id']},{r['actor']},{r['period']},{lab}\n")
                out.flush()
    coded = pd.read_csv(cache)
    return coded[coded.record_id.isin(sample.record_id)]

