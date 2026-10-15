"""Tables of the paper's appendix, computed from the dataset and the validation files.

    .venv/bin/python scripts/make_appendix.py      # writes paper/tables/*.tex, \\input by paper/cbdc_paper.tex
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbdc_biblio import analysis as A  # noqa: E402
from cbdc_biblio import nlp  # noqa: E402

P = ROOT / "presentation"
OUT = ROOT / "paper" / "tables"
LABELS = ["main theme", "secondary theme", "mentioned only", "unverified (no text)", "ambiguous terms only",
          "off-topic", "dated before 2014"]


def tex(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#").replace("$", r"\$"))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, den = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / den, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(c - h, 0), min(c + h, 1)


def num(x, f="{:.2f}"):
    return "--" if x != x else f.format(x)


def table(name, spec, head, rows, caption, label, note=""):
    body = "\n".join(rows)
    (OUT / f"{name}.tex").write_text(
        "\\begin{table}[htbp]\n\\centering\\small\n"
        f"\\caption{{{caption}}}\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
        + (f"\\par\\smallskip\\parbox{{0.92\\linewidth}}{{\\footnotesize {note}}}\n" if note else "")
        + "\\end{table}\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full = A.load(ROOT / "data/cbdc_corpus.csv")
    theme_ = full[full.relevance.isin(A.THEME)]
    fields = pd.read_csv(ROOT / "data/cache/openalex_fields.csv")
    econ = theme_[A.economists(theme_, fields)]

    # --- sources: records by database, at each stage of the filter
    src = lambda d: d.source_db.fillna("").str.split("; ").explode().replace({
        "BIS central bankers' speeches (RSS)": "BIS central bankers' speeches",
        "Authorities (hub crawl)": "Central banks and authorities (crawl)"}).value_counts()
    s_all, s_th, s_ec = src(full), src(theme_), src(econ)
    rows = [f"{tex(k)} & {s_all.get(k, 0):,} & {s_th.get(k, 0):,} & {s_ec.get(k, 0):,} \\\\" for k in s_all.index if k]
    rows.append(f"\\midrule Distinct records & {len(full):,} & {len(theme_):,} & {len(econ):,} \\\\")
    table("sources", "@{}lrrr@{}", r"\textbf{Source} & \textbf{Collected} & \textbf{CBDC a theme} & \textbf{By economists}",
          rows, "Records by source and filtering stage", "tab:sources",
          "A record found in several sources counts once per source; the last row counts distinct records.")

    # --- relevance labels and their validation
    rel = full.relevance.value_counts()
    v = pd.read_csv(P / "validation_relevance.csv")
    v = v[v["manual: CBDC is a theme"].notna()]
    rows = []
    for k in LABELS:
        if not rel.get(k, 0):
            continue
        g = v[v.relevance == k]["manual: CBDC is a theme"].astype(int)
        lo, hi = wilson(g.sum(), len(g))
        rows.append(f"{tex(k)} & {rel.get(k, 0):,} & {len(g)} & {int(g.sum())} & "
                    + (f"{lo:.2f}--{hi:.2f}" if len(g) else "--") + " \\\\")
    table("relevance", "@{}lrrrr@{}", r"\textbf{Label} & \textbf{Records} & \textbf{Read} & \textbf{Theme} & \textbf{95\% CI}",
          rows, "Relevance labels and their validation", "tab:relevance",
          f"Read: records of the stratified sample ({len(v)} texts) read blind to the label; Theme: read as having CBDC "
          "as a theme; Wilson intervals on the share.")

    # --- who counts as an economist
    ff = theme_[["record_id"]].merge(fields, on="record_id", how="left").set_index(theme_.index)
    ac = theme_.source_type == "academic"
    by_field = ff.field.eq(A.ECON_FIELD)
    by_cb = ff.institutions.fillna("").str.contains(A.CB_AFFIL) | theme_.author_affiliations.fillna("").str.contains(A.CB_AFFIL)
    by_venue = ff.field.isna() & theme_.venue.fillna("").str.contains(A.ECON_VENUE)
    off = theme_.source_type == "official"
    inst = off & theme_.issuer_type.isin(["central bank", "international organisation"])
    rows = [f"{a} & {b:,} & {c} \\\\" for a, b, c in [
        ("Research classified in Economics, Econometrics and Finance", int((ac & by_field).sum()), "kept"),
        ("Research in another field, with a central bank, IMF or BIS author", int((ac & ~by_field & by_cb).sum()), "kept"),
        ("Research without OpenAlex record, in an economics venue", int((ac & ~by_field & ~by_cb & by_venue).sum()), "kept"),
        ("Research in other fields", int((ac & ~(by_field | by_cb | by_venue)).sum()), "dropped"),
        ("Central bank speeches", int((theme_.source_type == "speech").sum()), "kept"),
        ("Documents of central banks and international institutions", int(inst.sum()), "kept"),
        ("Documents of governments and parliaments", int((off & ~inst).sum()), "dropped"),
        ("Press articles", int((theme_.source_type == "press").sum()), "dropped")]]
    rows.append(f"\\midrule Texts by economists & {len(econ):,} & \\\\")
    table("economists", "@{}p{0.62\\linewidth}rl@{}", r"\textbf{Texts with CBDC as a theme} & \textbf{Texts} & ",
          rows, "Who counts as an economist", "tab:economists",
          "Field: OpenAlex primary topic of the work; author institutions from OpenAlex and from the record's affiliations.")
    drop = ff[ac & ~(by_field | by_cb | by_venue)].field.fillna("no OpenAlex record").value_counts()
    rows = [f"{tex(k)} & {c:,} \\\\" for k, c in drop.items()]
    table("dropped_fields", "@{}lr@{}", r"\textbf{OpenAlex field of research dropped} & \textbf{Texts}", rows,
          "Research with CBDC as a theme left out by the economist filter, by field", "tab:dropped")

    # --- validation samples: size and composition
    ids = set(econ.record_id)
    samples = [("Relevance", "validation_relevance.csv"), ("Motivations, development", "validation_motivations_dev.csv"),
               ("Motivations, test (held out)", "validation_motivations_test.csv"), ("Stance", "validation_stance.csv"),
               ("Tone (sentences)", "validation_tone.csv")]
    rows = []
    for name, f in samples:
        s = pd.read_csv(P / f)
        rows.append(f"{name} & {len(s)} & {int(s.record_id.isin(ids).sum())} \\\\")
    table("samples", "@{}lrr@{}", r"\textbf{Sample} & \textbf{Units} & \textbf{From economists' texts}", rows,
          "Hand-coded validation samples", "tab:samples")

    # --- motivations: codebook by motivation, economists' held-out texts
    test = pd.read_csv(P / "validation_motivations_test.csv")
    te = test[test.record_id.isin(ids)].reset_index(drop=True)
    cb = nlp._codebook(te.text)
    rows, Y, Pp = [], [], []
    for k in A.MOTIVES:
        y = te[f"manual: {k}"].values.astype(bool)
        r = nlp._prf(y, cb[k])
        Y.append(y), Pp.append(cb[k])
        rows.append(f"{tex(k)} & {int(y.sum())} & {int(cb[k].sum())} & {num(r['precision'])} & {num(r['recall'])} & "
                    f"{num(r['kappa'])} \\\\")
    a = nlp._prf(np.concatenate(Y), np.concatenate(Pp))
    rows.append(f"\\midrule All (pooled) & {int(np.concatenate(Y).sum())} & {int(np.concatenate(Pp).sum())} & "
                f"{num(a['precision'])} & {num(a['recall'])} & {num(a['kappa'])} \\\\")
    table("codebook", "@{}lrrrrr@{}", r"\textbf{Motivation} & \textbf{Read} & \textbf{Coded} & \textbf{Precision} & "
          r"\textbf{Recall} & $\kappa$", rows, "Motivation codebook on the held-out texts written by economists",
          "tab:codebook", f"{len(te)} held-out texts by economists (of {len(test)}), never used to write the codebook. "
          "Read: texts where reading found the motivation; Coded: texts flagged by the codebook.")

    # --- every coder, both samples
    vc = pd.read_csv(P / "validation_coders.csv")
    for task, name, cols, label in [("motivations", "coders_motivations", ["precision", "recall", "f1", "kappa"], "tab:coders-mot"),
                                    ("stance", "coders_stance", ["accuracy", "kappa"], "tab:coders-stance")]:
        e, al = (vc[(vc.task == task) & (vc.scope == s_)].set_index("coder") for s_ in ("economists", "all"))
        rows = [f"{tex(c)} & " + " & ".join(num(r[x]) for x in cols) + f" & {num(al.loc[c, 'kappa'])} \\\\"
                for c, r in e.sort_values("kappa", ascending=False).iterrows()]
        hd = " & ".join({"precision": r"\textbf{P}", "recall": r"\textbf{R}", "f1": "$F_1$", "kappa": r"$\kappa$",
                         "accuracy": r"\textbf{Accuracy}"}[x] for x in cols)
        table(name, "@{}l" + "r" * (len(cols) + 1) + "@{}", rf"\textbf{{Coder}} & {hd} & $\kappa$\textbf{{, all}}", rows,
              f"Candidate coders for {'motivations' if task == 'motivations' else 'stance'}",
              label, f"Scores on the {int(e.n_texts.iloc[0])} texts written by economists; last column: $\\kappa$ on all "
                     f"{int(al.n_texts.iloc[0])} hand-coded texts. "
              + ("Thresholds and classifiers set on the 100 development texts." if task == "motivations"
                 else "Thresholds and classifiers by 5-fold cross-validation."))

    # --- tone: confusion matrix on economists' sentences
    tv = pd.read_csv(P / "validation_tone.csv")
    tv = tv[tv.record_id.isin(ids)]
    labs = ["negative", "neutral", "positive"]
    rows = [f"{l} & " + " & ".join(str(int(((tv.label == l) & (tv.manual == m)).sum())) for m in labs)
            + f" & {((tv.label == l) & (tv.manual == l)).sum() / max((tv.label == l).sum(), 1):.2f} \\\\" for l in labs]
    po = (tv.label == tv.manual).mean()
    pe = sum((tv.label == l).mean() * (tv.manual == l).mean() for l in labs)
    table("tone", "@{}lrrrr@{}", r"\textbf{FinBERT $\downarrow$ / reading $\rightarrow$} & \textbf{negative} & "
          r"\textbf{neutral} & \textbf{positive} & \textbf{Precision}", rows, "FinBERT tone against reading", "tab:tone",
          f"{len(tv)} sentences from economists' texts, 30 per FinBERT label, read blind to the label: agreement "
          f"{100 * po:.0f}\\%, $\\kappa = {(po - pe) / (1 - pe):.2f}$.")

    # --- stance by reading
    sv = pd.read_csv(P / "validation_stance.csv")
    sv = sv[sv.record_id.isin(ids)].assign(actor=lambda x: x.source_type.replace({"official": "speech"}))
    rows = []
    for a_, lbl in [("academic", "Research economists"), ("speech", "Central banks and institutions")]:
        h = sv[sv.actor == a_]["manual: stance"]
        cells = []
        for s_ in ["favourable", "neutral", "critical"]:
            lo, hi = wilson(int((h == s_).sum()), len(h))
            cells.append(f"{100 * (h == s_).mean():.0f} [{100 * lo:.0f}--{100 * hi:.0f}]")
        rows.append(f"{lbl} & {len(h)} & " + " & ".join(cells) + " \\\\")
    table("stance", "@{}lrrrr@{}", r"\textbf{Actor} & $n$ & \textbf{Favourable} & \textbf{Neutral} & \textbf{Critical}",
          rows, "Stance by reading, texts written by economists", "tab:stance", "Percentages with Wilson 95\\% intervals.")
    # --- policy events: attention (15 days after vs the 3 months before), tested against random dates
    ev, _ = A.event_study(econ)
    pl = A.placebo(econ)
    rows = [f"{pd.Timestamp(r.date):%d %b %Y} & {tex(r.event)} & {r.after} & {r.expected:.1f} & {r.ratio:.1f} & "
            f"{(pl['z'] >= r.z).mean():.2f} \\\\" for r in ev.itertuples()]
    table("events", "@{}llrrrr@{}", r"\textbf{Date} & \textbf{Event} & \textbf{Texts} & \textbf{Expected} & "
          r"\textbf{Ratio} & $p$", rows, "Policy events and the attention they received", "tab:events",
          f"Texts by economists dated to the day, in the 15 days from the event; expected from the daily rate over days "
          f"$-90$ to $-15$. $p$: share of {pl['n']} random dates with a larger $z = (N - E)/\\sqrt{{E}}$.")

    # --- every topic of the topic model
    tp = nlp.topics(econ, ROOT, ROOT / "data/cache/embeddings.npz", min_size=40, selection="leaf")
    periods = [p_ for p_, _, _ in A.PERIODS]
    rows = [f"{i} & {tex(', '.join(r['terms'][:5]))} & {r['n']:,} & " + " & ".join(f"{r[p_]:.1f}" for p_ in periods) + " \\\\"
            for i, (_, r) in enumerate(tp["table"].iterrows(), 1)]
    table("topics", "@{}rp{0.5\\linewidth}rrrr@{}", r"& \textbf{Topic (top terms)} & \textbf{Texts} & "
          + " & ".join(f"\\textbf{{{p_}}}" for p_ in periods), rows, "Topics of the economists' corpus", "tab:topics",
          f"BERTopic on {tp['n_docs']:,} texts (multilingual SBERT embeddings, UMAP, HDBSCAN with leaf selection and at "
          f"least 40 texts per topic); {100 * tp['n_outliers'] / tp['n_docs']:.0f}\\% of texts were outliers, assigned to the "
          "nearest topic. Terms: class-based TF-IDF on English texts. Period columns: \\% of the period's texts.")
    print(f"Wrote {len(list(OUT.glob('*.tex')))} tables to {OUT}")


if __name__ == "__main__":
    main()
