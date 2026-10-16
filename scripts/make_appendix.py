"""Tables (and the data of figures) of the paper, computed from the dataset and the validation files.

    .venv/bin/python scripts/make_appendix.py      # writes paper/tables/*.tex, \\input by paper/cbdc_paper.tex
"""
import math
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbdc_biblio import analysis as A  # noqa: E402
from cbdc_biblio import nlp  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from make_slides import quadrant, thematic_tikz  # noqa: E402

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

    # --- descriptive statistics of the economists' corpus (section 3.5)
    yrs = list(range(2016, int(econ.year.max()) + 1))
    kinds = [("academic", "Research", "c1"), ("speech", "Speeches", "c2"), ("official", "Documents", "c3")]
    plots = [f"\\addplot[fill={c}, draw={c}] coordinates {{"
             + " ".join(f"({y},{int(((econ.year == y) & (econ.source_type == k)).sum())})" for y in yrs)
             + f"}};\n\\addlegendentry{{{lbl}}}" for k, lbl, c in kinds]
    (OUT / "fig_years.tex").write_text(
        "\\begin{figure}[htbp]\n\\centering\n\\begin{tikzpicture}\n"
        "\\begin{axis}[ybar stacked, width=0.85\\linewidth, height=6cm, bar width=12pt, xtick=data, ymin=0,\n"
        "  xticklabel style={/pgf/number format/1000 sep=}, ylabel={Texts}, axis lines*=left, ymajorgrids,\n"
        "  legend style={at={(0.02,0.98)}, anchor=north west, draw=none, font=\\footnotesize}, tick label style={font=\\footnotesize}]\n"
        + "\n".join(plots) + "\n\\end{axis}\n\\end{tikzpicture}\n"
        f"\\caption{{Texts written by economists, by year and type ({int(econ.year.max())}: until October)}}\\label{{fig:years}}\n"
        "\\end{figure}\n", encoding="utf-8")
    dp = econ.date_precision.fillna("")
    lang = econ.language.value_counts()
    names = {"en": "English", "ru": "Russian", "de": "German", "uk": "Ukrainian", "es": "Spanish", "fr": "French",
             "pt": "Portuguese", "it": "Italian"}
    doc = econ.document_type.value_counts()
    pct = lambda m: f"{100 * m.mean():.0f}\\%"
    # two panels side by side, so the table fits in the flow of the text
    panel = lambda title, items: [(f"\\textit{{{title}}}", None)] + [(f"\\quad {a}", b) for a, b in items]
    left = (panel("Type of text", [("Research", f"{(econ.source_type == 'academic').sum():,}"),
                                   ("Central bank speeches", f"{(econ.source_type == 'speech').sum():,}"),
                                   ("Documents of institutions", f"{(econ.source_type == 'official').sum():,}"),
                                   ("Total", f"{len(econ):,}")])
            + panel("Relevance", [("CBDC the main theme", f"{(econ.relevance == 'main theme').sum():,}"),
                                  ("CBDC a secondary theme", f"{(econ.relevance == 'secondary theme').sum():,}")])
            + panel("Research, by document type", [(tex(k.capitalize().replace("-", " ")), f"{doc.get(k, 0):,}")
                                                   for k in ("article", "preprint", "book-chapter", "conference-paper", "dissertation")]))
    right = (panel(f"Language ({econ.language.nunique()} in all)", [(names.get(k, k), f"{c:,}") for k, c in lang.head(6).items()])
             + panel("Fields available", [("Exact day of publication", pct(dp.str.startswith("day") & ~dp.str.contains("Jan 1"))),
                                          ("Abstract", pct(econ.abstract.notna())), ("Introduction", pct(econ.introduction.notna())),
                                          ("Full text", pct(econ.full_text_path.notna())),
                                          ("Keywords", pct(econ.keywords_author.notna() | econ.keywords_indexed.notna())),
                                          ("Reference list (research)", pct(econ[econ.source_type == "academic"].references.notna()))]))
    cell = lambda c: "&" if c is None else (f"\\multicolumn{{2}}{{@{{}}l}}{{{c[0]}}}" if c[1] is None else f"{c[0]} & {c[1]}")
    rows = [f"{cell(l)} & {cell(r)} \\\\" for l, r in zip_longest(left, right)]
    (OUT / "descriptive.tex").write_text(
        "\\begin{table}[!htbp]\n\\centering\\small\n\\caption{The economists' corpus}\\label{tab:descriptive}\n"
        "\\begin{tabular}{@{}lr@{\\hspace{2.5em}}lr@{}}\n\\toprule\n"
        "& \\textbf{Texts} && \\textbf{Texts} \\\\\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")

    # --- what is written: thematic map, keyword shares by period, adjusted topic changes (section 5)
    themes = A.thematic_map(econ)
    tz, mx, my = thematic_tikz(themes, w=8.5, h=5.5)
    legend = "\\\\[2pt]\n".join(f"\\textbf{{{i + 1}}} ({quadrant(t, mx, my)}): {tex(', '.join(t['members'][:4]))}"
                               for i, t in enumerate(themes))
    (OUT / "fig_thematic.tex").write_text(
        "\\begin{figure}[H]\n\\centering\n\\fitfigure{" + tz + "}{%\n\\medskip\n\\parbox{0.9\\linewidth}{\\footnotesize " + legend + "}\n"
        "\\caption{Thematic map of the keywords of research by economists}\\label{fig:thematic}}\n\\end{figure}\n",
        encoding="utf-8")
    kw = A.keyword_network(econ)
    ev = kw["evolution"]
    periods = [p_ for p_, _, _ in A.PERIODS]
    shown = [("monetary policy and economic impact", "c1", "*"), ("cryptocurrency", "c5", "diamond*"),
             ("financial inclusion", "c4", "square*"), ("digital ruble", "c2", "triangle*"), ("digital euro", "c3", "otimes*")]
    lines = [f"\\addplot[{c}, thick, mark={m}] coordinates {{" + " ".join(f"({p_},{ev.loc[k, p_]:.1f})" for p_ in periods)
             + f"}};\n\\addlegendentry{{{tex(k)}}}" for k, c, m in shown if k in ev.index]
    (OUT / "fig_keywords.tex").write_text(
        "\\begin{figure}[H]\n\\centering\n\\fitfigure{\\begin{tikzpicture}\n"
        "\\begin{axis}[width=0.8\\linewidth, height=6cm, symbolic x coords={" + ",".join(periods) + "}, xtick=data,\n"
        "  ylabel={\\% of the period's papers}, ymin=0, axis lines*=left, ymajorgrids, enlarge x limits=0.15,\n"
        "  legend style={at={(0.5,-0.22)}, anchor=north, legend columns=3, draw=none, font=\\footnotesize, /tikz/every even column/.append style={column sep=8pt}},\n"
        "  legend cell align=left, tick label style={font=\\footnotesize}]\n"
        + "\n".join(lines) + "\n\\end{axis}\n\\end{tikzpicture}}{%\n"
        "\\caption{Share of research papers carrying each keyword, by period ("
        + ", ".join(f"{kw['period_n'][p_]:,} papers in {p_}" for p_ in periods) + ")}\\label{fig:keywords}}\n\\end{figure}\n",
        encoding="utf-8")
    tp = nlp.topics(econ, ROOT, ROOT / "data/cache/embeddings.npz", min_size=40, selection="leaf")
    tf = A.with_covariates(econ[econ.source_type.isin(nlp.NON_PRESS)].merge(tp["assign"], on="record_id"))
    pts, labels = [], []
    for i, (_, r) in enumerate(tp["table"].head(12).iterrows(), 1):
        tf["is_k"] = tf.topic == r["topic"]
        e_ = A.adjusted(tf, "is_k").get("period")
        if not e_:
            continue
        lab = f"{i}: {', '.join(r['terms'][:3])}"
        labels.append(tex(lab))
        pts.append(f"({e_[0]:.2f},{{{tex(lab)}}}) += ({e_[2] - e_[0]:.2f},0) -= ({e_[0] - e_[1]:.2f},0)")
    (OUT / "fig_topics.tex").write_text(
        "\\begin{figure}[H]\n\\centering\n\\fitfigure{\\begin{tikzpicture}\n"
        "\\begin{axis}[width=0.62\\linewidth, height=8cm, symbolic y coords={" + ",".join("{" + l + "}" for l in labels[::-1]) + "},\n"
        "  ytick=data, y dir=normal, xmin=-30, xmax=30, xlabel={adjusted change, points (2023--2026 vs 2014--2019)}, axis lines*=left, xmajorgrids,\n"
        "  tick label style={font=\\footnotesize}, label style={font=\\footnotesize}, enlarge y limits=0.05]\n"
        "\\addplot[c1, only marks, mark=*, error bars/.cd, x dir=both, x explicit] coordinates {" + " ".join(pts) + "};\n"
        "\\draw[c2, dashed] (axis cs:0,{" + labels[0] + "}) -- (axis cs:0,{" + labels[-1] + "});\n"
        "\\end{axis}\n\\end{tikzpicture}}{%\n"
        "\\caption{Change in the share of the twelve largest topics, adjusted for the type of author and the language, "
        "with 95\\% confidence intervals (axis cut at $\\pm 30$ points: topics almost absent in 2014--2019 have very wide intervals)}\\label{fig:topics}}\n\\end{figure}\n", encoding="utf-8")

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
