"""Score every candidate coder against the hand-coded samples; writes presentation/validation_coders.csv.

Every coder is scored twice: on the whole sample ("all") and on the texts written by economists ("economists",
the corpus of the talk). Per-text predictions are saved in presentation/validation_predictions_*.csv.

    .venv/bin/python scripts/validate_coders.py        # about 1.5 hours on an M1 (the local LLMs dominate)

Motivations (multi-label, 74 held-out texts x 10 motivations, 50 by economists; tuned on the 100 development texts):
  codebook, SBERT zero-shot, SBERT classifier, NLI zero-shot, local LLMs (Qwen2.5 3B and 7B, 4-bit).
Stance (favourable / neutral / critical, 177 texts, 100 by economists; thresholds and classifiers by 5-fold cross-validation):
  FinBERT tone, SBERT classifier, NLI zero-shot, local LLM 7B.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbdc_biblio import analysis as A  # noqa: E402
from cbdc_biblio import nlp  # noqa: E402
from cbdc_biblio.analysis import MOTIVES, MOTIVE_RE  # noqa: E402

OUT = ROOT / "presentation"
NLI = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
LLMS = {"Local LLM, Qwen2.5 3B": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "Local LLM, Qwen2.5 7B": "mlx-community/Qwen2.5-7B-Instruct-4bit"}
NAMES = list(MOTIVES)
DEFS = {"Decline of cash": "cash use is falling, or access to cash / public money is at risk",
        "Financial inclusion": "reaching unbanked or underbanked people, access to financial services",
        "Monetary sovereignty": "threat from private money, stablecoins, Big Tech, crypto or foreign currencies to the "
                                "central bank's control of money",
        "Payment efficiency": "cheaper, faster, more efficient, resilient or innovative domestic payments",
        "Cross-border payments": "international payments, remittances, cross-border settlement",
        "Monetary policy": "monetary policy, interest rates, transmission, money supply",
        "Financial stability": "bank disintermediation, bank runs, deposits leaving banks, financial stability risks",
        "Privacy and surveillance": "privacy, anonymity, data protection, surveillance",
        "Geopolitics and sanctions": "geopolitics, sanctions, dollar dominance, currency internationalisation",
        "Technology (DLT, programmability)": "blockchain, distributed ledger, tokenisation, smart contracts, "
                                             "programmable money"}
HYP = {k: f"The text raises {d}." for k, d in DEFS.items()}
STANCE_HYP = ["The author is in favour of central bank digital currencies.",
              "The author is critical of central bank digital currencies."]


def nli_scores(texts, hyps, batch=16):
    """Entailment probability of each hypothesis given each text (multilingual mDeBERTa NLI)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NLI)
    mod = AutoModelForSequenceClassification.from_pretrained(NLI).to(nlp._device()).eval()
    out = np.zeros((len(texts), len(hyps)))
    for j, h in enumerate(hyps):
        for b in range(0, len(texts), batch):
            x = tok(texts[b:b + batch], [h] * len(texts[b:b + batch]), truncation="only_first", max_length=320,
                    padding=True, return_tensors="pt").to(nlp._device())
            with torch.no_grad():
                out[b:b + batch, j] = mod(**x).logits.softmax(-1)[:, 0].cpu().numpy()
    del mod
    free()
    return out


def llm(model_id, system, texts, max_tokens=25):
    from mlx_lm import generate, load
    model, tok = load(model_id)
    answers = []
    for t in texts:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": "Text:\n" + t[:1500] + "\n\nAnswer:"}]
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        answers.append(generate(model, tok, prompt=p, max_tokens=max_tokens, verbose=False))
    del model
    free()
    return answers


def free():
    """Release GPU memory between models (an 8 GB machine cannot hold two of them)."""
    import gc
    import mlx.core as mx
    gc.collect()
    mx.clear_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def economist_ids():
    d = A.load(ROOT / "data/cbdc_corpus.csv")
    d = d[d.relevance.isin(A.THEME)]
    return set(d[A.economists(d, pd.read_csv(ROOT / "data/cache/openalex_fields.csv"))].record_id)


ECON = economist_ids()


def motivations():
    dev, test = (pd.read_csv(OUT / f"validation_motivations_{s}.csv") for s in ("dev", "test"))
    y = lambda df: np.concatenate([df[f"manual: {k}"].values.astype(bool) for k in NAMES])
    flat = lambda pred, m=None: np.concatenate([np.asarray(pred[k], bool)[m] if m is not None else np.asarray(pred[k], bool)
                                                for k in NAMES])
    preds = nlp.validate_motives(dev, test)["preds"]
    sd, st = nli_scores(list(dev.text), list(HYP.values())), nli_scores(list(test.text), list(HYP.values()))
    tau = max(np.arange(0.05, 0.96, 0.05),
              key=lambda t: nlp._prf(y(dev), flat({k: sd[:, i] >= t for i, k in enumerate(NAMES)}))["f1"])
    preds["NLI zero-shot (mDeBERTa)"] = {k: st[:, i] >= tau for i, k in enumerate(NAMES)}
    system = ("You code research texts about central bank digital currencies (CBDC). List which motivations or issues "
              "the text raises as reasons to study, issue or worry about a CBDC. Motivations:\n"
              + "\n".join(f"{i + 1}. {k}: {DEFS[k]}" for i, k in enumerate(NAMES))
              + "\nAnswer only with the numbers that apply, comma-separated, or 0 if none.")
    for name, mid in LLMS.items():
        ans = llm(mid, system, list(test.text))
        nums = [{int(x) for x in re.findall(r"\d+", a) if 1 <= int(x) <= 10} for a in ans]
        preds[name] = {k: np.array([(i + 1) in n for n in nums]) for i, k in enumerate(NAMES)}
    pd.DataFrame([{"record_id": r, "coder": c, "motivation": k, "pred": bool(p[k][j])}
                  for c, p in preds.items() for k in NAMES for j, r in enumerate(test.record_id)]).to_csv(
        OUT / "validation_predictions_motivations.csv", index=False)
    yt = test.set_index("record_id")
    rows = []
    for scope, m in [("all", np.ones(len(test), bool)), ("economists", test.record_id.isin(ECON).values)]:
        ys = np.concatenate([yt[f"manual: {k}"].values.astype(bool)[m] for k in NAMES])
        for c, p in preds.items():
            rows.append({"coder": c, "scope": scope, "n_texts": int(m.sum()), **nlp._prf(ys, flat(p, m))})
    return pd.DataFrame(rows).assign(task="motivations")


def stance():
    s = pd.read_csv(OUT / "validation_stance.csv")
    y = s["manual: stance"].values
    texts = [str(t if st == "press" else f"{t}. {u}")[:1200] for t, u, st in zip(s.title, s.unit, s.source_type)]
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = {}

    def cv_threshold(score_fn, grid):
        pred = np.empty(len(y), object)
        for tr, te in skf.split(texts, y):
            t = max(grid, key=lambda t: cohen_kappa_score(y[tr], score_fn(t)[tr]))
            pred[te] = score_fn(t)[te]
        return pred

    P = nli_scores(texts, STANCE_HYP)
    rows["NLI zero-shot (mDeBERTa)"] = cv_threshold(
        lambda t: np.where((P[:, 0] > t) & (P[:, 0] >= P[:, 1]), "favourable", np.where(P[:, 1] > t, "critical", "neutral")),
        np.arange(0.1, 0.95, 0.05))
    from transformers import pipeline
    fb = pipeline("text-classification", model=nlp.FINBERT, device=nlp._device(), top_k=None, truncation=True, max_length=256)
    tone = np.array([next(a["score"] for a in p if a["label"] == "positive") - next(a["score"] for a in p if a["label"] == "negative")
                     for p in fb(texts, batch_size=16)])
    rows["FinBERT tone"] = cv_threshold(lambda t: np.where(tone > t, "favourable", np.where(tone < -t, "critical", "neutral")),
                                        np.arange(0.05, 0.95, 0.05))
    e = nlp._sbert().encode(texts, normalize_embeddings=True)
    pred = np.empty(len(y), object)
    for tr, te in skf.split(texts, y):
        pred[te] = LogisticRegression(C=10, class_weight="balanced", max_iter=3000).fit(e[tr], y[tr]).predict(e[te])
    rows["SBERT classifier"] = pred
    ans = llm(LLMS["Local LLM, Qwen2.5 7B"], nlp.STANCE_SYSTEM, texts, max_tokens=5)
    rows["Local LLM, Qwen2.5 7B"] = np.array(["favourable" if "favour" in a.lower() else "critical" if "critic" in a.lower()
                                              else "neutral" for a in ans])
    pd.DataFrame({"record_id": s.record_id, "llm": rows["Local LLM, Qwen2.5 7B"]}).to_csv(
        OUT / "validation_stance_llm.csv", index=False)
    pd.DataFrame({"record_id": s.record_id, **rows}).to_csv(OUT / "validation_predictions_stance.csv", index=False)
    out = []
    for scope, m in [("all", np.ones(len(y), bool)), ("economists", s.record_id.isin(ECON).values)]:
        for k, p in rows.items():
            out.append({"coder": k, "scope": scope, "n_texts": int(m.sum()), "accuracy": accuracy_score(y[m], p[m]),
                        "kappa": cohen_kappa_score(y[m], p[m])})
    return pd.DataFrame(out).assign(task="stance")


if __name__ == "__main__":
    out = pd.concat([motivations(), stance()])
    out.to_csv(OUT / "validation_coders.csv", index=False)
    print(out.round(2))
