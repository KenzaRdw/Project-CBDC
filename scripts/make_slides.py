"""Generate the Beamer presentation of the corpus from data/cbdc_corpus.csv.

    .venv/bin/python scripts/make_slides.py      # writes + compiles presentation/cbdc_corpus.pdf

Every number, table and network in the deck is computed here from the dataset, so the
slides follow the data whenever the corpus is rebuilt. The LaTeX lives in TEMPLATE below;
<<NAME>> placeholders are filled from the dict built in main().
"""
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import date
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbdc_biblio import analysis as A  # noqa: E402
from cbdc_biblio import nlp  # noqa: E402
from cbdc_biblio.common import theme  # noqa: E402

OUT = ROOT / "presentation"
TYPES = [("academic", "Academic"), ("speech", "Speeches"), ("official", "Official docs"), ("press", "Press")]
RELEVANT = ["main theme", "secondary theme"]  # CBDC is a theme of the text, not a passing mention
REL_LABEL = {"main theme": "Main theme ($\\geq 50$)", "secondary theme": "Secondary theme (25--49)",
             "mentioned only": "Mentioned only ($<25$)", "unverified (no text)": "Press, text not available",
             "ambiguous terms only": "Ambiguous terms only", "off-topic": "Off-topic (homonyms, no match)",
             "dated before 2014": "Dated before 2014 (checked: false)"}
LANG = {"en": "English", "ru": "Russian", "uk": "Ukrainian", "es": "Spanish", "de": "German", "fr": "French",
        "ja": "Japanese", "pt": "Portuguese", "it": "Italian", "tr": "Turkish", "id": "Indonesian",
        "zh": "Chinese", "pl": "Polish", "ko": "Korean", "ar": "Arabic", "nl": "Dutch", "sv": "Swedish",
        "hi": "Hindi", "vi": "Vietnamese", "th": "Thai", "no": "Norwegian"}
SHORT = {"Zenodo (CERN European Organization for Nuclear Research)": "Zenodo (CERN repository)",
         "CyberLeninK (CyberLeninka)": "CyberLeninka (Russian repository)",
         "Board of Governors of the Federal Reserve System": "Federal Reserve Board",
         "Advances in finance, accounting, and economics book series": "Advances in Finance, Accounting & Economics",
         "EKONOMIKA I UPRAVLENIE PROBLEMY RESHENIYA": "Ekonomika i Upravlenie"}
# a topic is about one jurisdiction when one of its two top terms names a country or its currency
JURISDICTION = {"euro", "european", "ecb", "europe", "pound", "ruble", "russian", "russia", "china", "e-cny", "rmb", "yuan",
                "chinese", "india", "rupee", "rbi", "indian", "indonesia", "rupiah", "nigeria", "enaira", "africa",
                "african", "bahamas", "caribbean", "sweden", "e-krona", "riksbank", "brazil", "drex", "japan", "korea",
                "ukraine", "e-hryvnia", "turkey", "iran", "canada", "australia"}
COLORS = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]

# examples shown on the scoring slide: (description, kwargs for theme())
FILLER = "Inflation stayed high and banks tightened lending standards across the region. " * 40
EXAMPLES = [
    ("Paper, CBDC in title", dict(title="The digital euro and monetary sovereignty",
                                  abstract="We study the design of the digital euro and its effect on banks. "
                                           "We use a model of deposits. Results matter for policy.")),
    ("Paper, generic title, CBDC abstract", dict(
        title="Money in the digital age",
        abstract="This paper examines central bank digital currency (CBDC) design. We find that a CBDC with "
                 "remuneration crowds out deposits. Policy implications for CBDC issuance follow.")),
    ("Paper, one passing mention", dict(
        title="Bitcoin price dynamics and investor attention",
        abstract="We analyse Bitcoin returns using Google Trends data. Volatility is driven by retail attention. "
                 "Sentiment matters too. Regulatory news such as discussions of central bank digital "
                 "currencies also matters. Results are robust to alternative specifications. We use a long "
                 "sample of daily data. Implications for investors are discussed.")),
    ("Speech, one paragraph on it", dict(
        title="Monetary policy outlook",
        body=FILLER * 3 + " The digital euro project progresses. A digital euro would complement cash. " + FILLER * 3)),
    ("Speech devoted to it", dict(
        title="Remarks at a conference",
        body=(" The digital euro would be a form of central bank money. " + FILLER[:400]) * 30)),
    ("Medical case report", dict(
        title="Chronic bullous dermatosis of childhood (CBDC): a case report",
        abstract="A 5-year-old girl presented with tense blisters; CBDC was diagnosed by direct "
                 "immunofluorescence showing linear IgA deposits along the basement membrane zone.")),
]


# ----------------------------------------------------------------- LaTeX helpers

CYR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюяієїґ",
               "a b v g d e e zh z i y k l m n o p r s t u f kh ts ch sh shch  y  e yu ya i ye yi g".split(" ")))


def translit(s):
    """Cyrillic -> Latin (pdflatex cannot typeset it); other non-Latin scripts are dropped."""
    s = re.sub("[\u2010\u2011]", "-", s)
    out = "".join((CYR.get(c.lower(), c).capitalize() if c.isupper() else CYR.get(c, c)) for c in s)
    return re.sub(r"[^\x00-\u024F’–—…]", "", out).strip() or "[non-Latin]"


def tex(s, n=None):
    s = translit(str(s if s == s else ""))
    if n and len(s) > n:
        s = s[: n - 1].rstrip() + "…"
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\~{}"), ("^", r"\^{}"), ("…", r"\dots{}")]:
        s = s.replace(a, b)
    return s


def latin(s):
    return isinstance(s, str) and not re.search(r"[^\x00-ɏ’–—…]", s)


def rows(series, n, width=60):
    series = series.rename(index=SHORT)
    series = series[[latin(i) for i in series.index]].head(n)
    return "\n".join(f"{tex(k, width)} & {v:,} \\\\" for k, v in series.items())


def two_col(left_title, left, right_title, right, k=8):
    return (r"\begin{columns}[T]" + "\n"
            + "".join(r"\begin{column}{0.48\textwidth}\footnotesize\textbf{" + t + r"}\\[2pt]"
                      + r"\begin{tabular}{@{}p{0.78\linewidth}r@{}}" + rows(s, k, 46)
                      + r"\end{tabular}\end{column}" + "\n"
                      for t, s in [(left_title, left), (right_title, right)])
            + r"\end{columns}")


def network_tikz(g, w=11.0, h=5.3, maxlen=24, max_edges=160):
    """Draw a networkx graph (nodes carry occ, cluster, label) as TikZ, VOSviewer style."""
    if not len(g):
        return r"\emph{Not enough data yet.}"
    pos = A.layout(g, w, h)
    occ_max = max(d["occ"] for _, d in g.nodes(data=True))
    edges = sorted(g.edges(data=True), key=lambda e: -e[2]["strength"])[:max_edges]
    s_max = max((e[2]["strength"] for e in edges), default=1)
    out = [r"\begin{tikzpicture}"]
    for a, b, d in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        out.append(f"\\draw[gray, opacity={0.12 + 0.55 * d['strength'] / s_max:.2f}, line width=0.35pt] "
                   f"({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")
    for n, d in g.nodes(data=True):
        x, y = pos[n]
        r = 0.6 + 2.6 * math.sqrt(d["occ"] / occ_max)
        col = COLORS[d.get("cluster", 0) % len(COLORS)]
        out.append(f"\\fill[{col}, opacity=0.85] ({x:.2f},{y:.2f}) circle ({r:.2f}mm);")
        out.append(f"\\node[font=\\fontsize{{4.6}}{{5}}\\selectfont, above={r * 0.7:.2f}mm, inner sep=0.5pt] "
                   f"at ({x:.2f},{y:.2f}) {{{tex(d['label'], maxlen)}}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out)


def cluster_list(g, comms, k=4, per=4, width=95):
    lines = []
    for i, c in enumerate(comms[:k]):
        top = sorted(c, key=lambda n: -g.nodes[n]["occ"])[:per]
        lines.append(f"\\item[\\textcolor{{{COLORS[i]}}}{{$\\bullet$}}] "
                     + tex(", ".join(g.nodes[n]["label"] for n in top), width))
    return "\n".join(lines)


def heat_rows(df, cols, fmt="{:.1f}", width=40, cap=60):
    """Table rows with cells shaded by value (darker = higher)."""
    out = []
    for name, r in df.iterrows():
        cells = " & ".join(f"\\cellcolor{{c1!{min(cap, int(r[c] * 1.5))}}}{fmt.format(r[c])}" for c in cols)
        out.append(f"{tex(name, width)} & {cells} \\\\")
    return "\n".join(out)


def thematic_tikz(themes, w=6.4, h=4.6):
    """Strategic diagram: x = centrality, y = density, quadrants split at the medians."""
    if not themes:
        return r"\emph{Not enough data yet.}"
    xs, ys = [t["centrality"] for t in themes], [t["density"] for t in themes]
    mx, my = sorted(xs)[len(xs) // 2], sorted(ys)[len(ys) // 2]
    x0, x1, y0, y1 = min(xs) * 0.8, max(xs) * 1.15, min(ys) * 0.8, max(ys) * 1.15
    sx = lambda v: (v - x0) / (x1 - x0) * w
    sy = lambda v: (v - y0) / (y1 - y0) * h
    s_max = max(t["size"] for t in themes)
    out = [r"\begin{tikzpicture}[font=\scriptsize]",
           f"\\draw[gray!60, dashed] ({sx(mx):.2f},0) -- ({sx(mx):.2f},{h});",
           f"\\draw[gray!60, dashed] (0,{sy(my):.2f}) -- ({w},{sy(my):.2f});",
           f"\\draw[-{{Stealth}}, gray] (0,0) -- ({w + 0.2},0) node[below, text=gray, pos=0.5, yshift=-1mm] {{centrality $\\to$}};",
           f"\\draw[-{{Stealth}}, gray] (0,0) -- (0,{h + 0.2}) node[above, text=gray] {{density}};",
           f"\\node[gray, anchor=south east, font=\\scriptsize\\bfseries] at ({w},{h}) {{Motor}};",
           f"\\node[gray, anchor=south west, font=\\scriptsize\\bfseries] at (0.1,{h}) {{Niche}};",
           f"\\node[gray, anchor=north west, font=\\scriptsize\\bfseries] at (0.1,-0.35) {{Emerging / declining}};",
           f"\\node[gray, anchor=north east, font=\\scriptsize\\bfseries] at ({w},-0.35) {{Basic}};"]
    for i, t in enumerate(themes):
        r = 2 + 4.5 * math.sqrt(t["size"] / s_max)
        x, y = sx(t["centrality"]), sy(t["density"])
        out.append(f"\\fill[{COLORS[i % len(COLORS)]}, opacity=0.8] ({x:.2f},{y:.2f}) circle ({r:.1f}mm);")
        out.append(f"\\node[font=\\scriptsize, text=white] at ({x:.2f},{y:.2f}) {{\\textbf{{{i + 1}}}}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out), mx, my


def quadrant(t, mx, my):
    hi_c, hi_d = t["centrality"] >= mx, t["density"] >= my
    return {(True, True): "motor", (False, True): "niche", (False, False): "emerging or declining",
            (True, False): "basic"}[(hi_c, hi_d)]


def robustness(d, mo_table, tone_doc, stance_doc, variants):
    """Headline results re-estimated on each variant: {label: (record ids kept, motivation table or None)}."""
    rows = []
    for label, (ids, mtab) in variants.items():
        dv = d[d.record_id.isin(ids)]
        mt = mtab if mtab is not None else mo_table
        mt = A.with_covariates(mt[mt.record_id.isin(ids)].join(d.set_index("record_id")[["language"]], on="record_id")
                               if "language" not in mt else mt[mt.record_id.isin(ids)])
        e_inc = A.adjusted(mt, "Financial inclusion").get("period", (np.nan,) * 3)
        e_mp = A.adjusted(mt, "Monetary policy").get("period", (np.nan,) * 3)
        td = tone_doc[tone_doc.record_id.isin(ids)]
        gap = td[td.actor == "speech"].tone.mean() - td[td.actor == "press"].tone.mean()
        sd = stance_doc[stance_doc.record_id.isin(ids)]
        fav = lambda a: 100 * (sd[sd.actor == a].stance == "favourable").mean()
        ev, _ = A.event_study(dv)
        pl = A.placebo(dv)
        n_sig = int(sum((pl["z"] >= z).mean() < 0.05 for z in ev.z))
        rows.append((label, len(dv), e_inc, e_mp, gap, fav("speech") - fav("academic"), n_sig, len(ev)))
    return rows


def main():
    full = A.load(ROOT / "data/cbdc_corpus.csv")
    rel = full.relevance.value_counts()
    n_all = len(full)
    d = full[full.relevance.isin(RELEVANT)].copy()
    n = len(d)
    by_type = d.source_type.value_counts()
    types = [(k, lbl) for k, lbl in TYPES if by_type.get(k, 0)]
    today = date.today()
    periods = [p for p, _, _ in A.PERIODS]
    v = {"TIMES": f"{n / 1024:.0f}", "DATE": f"{today:%B %Y}", "N_ALL": f"{n_all:,}", "N": f"{n:,}", "N_LANG": str(d.language.nunique()),
         "Y0": str(int(d.year.min())), "Y1": str(int(d.year.max())), "N_COLS": str(len(full.columns) + 1)}

    # --- corpus description
    yrs = list(range(2016, today.year + 1))
    v["YEAR_PLOTS"] = "\n".join(
        r"\addplot coordinates {" + " ".join(f"({y},{int(d[d.source_type == k].year.value_counts().get(y, 0))})" for y in yrs)
        + "};\n" + f"\\addlegendentry{{{lbl}}}" for k, lbl in types)
    v.update({"BEFORE": str(int((d.year < yrs[0]).sum())), "YEAR0": str(yrs[0]),
              "THIS_YEAR": str(today.year), "THIS_MONTH": f"{today:%B}"})
    per_year = d.year.value_counts()
    v["GROWTH"] = f"{per_year.get(2019, 0):,} in 2019 to {per_year.get(today.year - 1, 0):,} in {today.year - 1}"
    lang = d.language[~d.language.isin(["en"]) & d.language.notna() & (d.language != "")].value_counts().head(10)[::-1]
    v["LANG_COORDS"] = " ".join(f"({c},{{{LANG.get(k, k)}}})" for k, c in lang.items())
    v["LANG_SYM"] = ",".join(f"{{{LANG.get(k, k)}}}" for k in lang.index)
    second = lang.index[-1] if len(lang) else ""
    v["LANG2"], v["LANG2_N"] = LANG.get(second, second), f"{int(lang.iloc[-1]) if len(lang) else 0:,}"
    v["LANG2_PRESS"] = f"{int(((d.language == second) & (d.source_type == 'press')).sum()):,}"
    v["EN_N"], v["EN_PCT"] = f"{(d.language == 'en').sum():,}", f"{100 * (d.language == 'en').mean():.0f}"
    src = d.source_db.str.split("; ").explode().replace({
        "BIS central bankers' speeches (RSS)": "BIS central bankers' speeches",
        "Authorities (hub crawl)": "Central banks & authorities (crawl)"}).value_counts()
    v["SRC_ROWS"] = rows(src, 10, 40)
    v["TOP_TYPE"] = {"academic": "Academic texts", "press": "Press articles", "speech": "Speeches",
                     "official": "Official documents"}[by_type.idxmax()]
    v["TYPE_ROWS"] = "\n".join(f"{lbl} & {by_type.get(k, 0):,} \\\\" for k, lbl in types)
    dp = d.date_precision.fillna("")
    pct = lambda m: f"{100 * m.sum() / n:.0f}\\%"
    v["QUALITY_ROWS"] = "\n".join(f"{a} & {b} \\\\" for a, b in [
        ("Exact day of publication", pct(dp.str.startswith("day") & ~dp.str.contains("Jan 1"))),
        ("Issuer / outlet identified", pct(d.issuer.notna() | d.venue.notna())),
        ("Abstract", pct(d.abstract.notna())), ("Full text", pct(d.full_text_path.notna())),
        ("Keywords", pct(d.keywords_author.notna() | d.keywords_indexed.notna())),
        ("Reference list", pct(d.references.notna()))])
    v["REL_ROWS"] = "\n".join(f"{REL_LABEL.get(k, k)} & {rel.get(k, 0):,} \\\\" for k in REL_LABEL if rel.get(k, 0))
    v["EXAMPLE_ROWS"] = "\n".join(f"{desc} & {s} & {lab} \\\\" for desc, kw in EXAMPLES for lab, s, _, _ in [theme(**kw)])

    acad = d[d.source_type == "academic"]
    v["VENUE_ROWS"] = rows(acad.venue.value_counts(), 7, 44)
    cited = acad.dropna(subset=["cited_by_count"]).sort_values("cited_by_count", ascending=False)
    cited = cited[~cited.title.str.lower().str.replace(r"\W", "", regex=True).str[-40:].duplicated()].head(5)
    v["CITED_ROWS"] = "\n".join(
        f"{tex(r.authors.split('; ')[0].split()[-1]) if isinstance(r.authors, str) else ''} ({int(r.year)}) & "
        f"{tex(r.title.title() if r.title.isupper() else r.title, 58)} & {int(r.cited_by_count):,} \\\\"
        for r in cited.itertuples())
    v["VOICES"] = (r"\begin{columns}[T]" + "".join(
        r"\begin{column}{0.32\textwidth}\scriptsize\textbf{" + t + r"}\\[2pt]"
        + r"\begin{tabular}{@{}p{0.8\linewidth}r@{}}" + rows(s, 8, 30) + r"\end{tabular}\end{column}"
        for t, s in [("Speeches", d[d.source_type == "speech"].issuer.value_counts()),
                     ("Official documents", d[d.source_type == "official"].issuer.value_counts()),
                     ("Press outlets", d[d.source_type == "press"].issuer.value_counts())]) + r"\end{columns}")
    top_bank = d[d.source_type == "speech"].issuer.value_counts()
    v["TOP_BANK"] = tex(top_bank.index[0]) if len(top_bank) else "--"

    # --- science maps
    cc = A.co_citation(d)
    v["COCIT_NET"] = network_tikz(cc["graph"], w=9.0, h=4.1)
    v["COCIT_CLUSTERS"] = cluster_list(cc["graph"], cc["clusters"])
    v["COCIT_STATS"] = (f"{cc['n_docs']:,} papers with reference lists, {cc['n_refs']:,} distinct references; "
                        f"{len(cc['graph'])} most co-cited shown, {len(cc['clusters'])} clusters, $Q={cc['modularity']:.2f}$.")
    v["COCIT_TOP"] = "\n".join(f"{tex(r)} & {c} \\\\" for r, c in cc["top"][:8])
    v["COCIT_CORE"] = ", ".join(tex(r) for r, _ in cc["top"][:3])

    ca = A.co_authorship(d, top=80)
    for _, dd in ca["graph"].nodes(data=True):
        dd["label"] = dd["label"].split()[-1]
    v["COAUTH_NET"] = network_tikz(ca["graph"], w=8.4, h=5.6, maxlen=22)
    v["COAUTH_TOP"] = "\n".join(f"{tex(a, 26)} & {c} \\\\" for a, c in ca["top"][:8])
    cb = A.central_bank_authors(d)
    v["CB_AUTH"] = (f"Papers with a central bank, BIS, IMF or World Bank author: \\alert{{{100 * cb['share']:.0f}\\%}} "
                    f"({100 * cb['by_period'][periods[0]]:.0f}\\% in {periods[0]}, {100 * cb['by_period'][periods[-1]]:.0f}\\% in "
                    f"{periods[-1]}), but \\alert{{{100 * cb['cite_share']:.0f}\\% of citations}}.")
    v["CB_SHARE"], v["CB_CITE"] = f"{100 * cb['share']:.0f}", f"{100 * cb['cite_share']:.0f}"
    v["COAUTH_STATS"] = (f"{ca['n_authors']:,} authors; {100 * ca['coauthored']:.0f}\\% of papers co-authored "
                         f"(mean team {ca['mean_team']:.1f}); {100 * ca['intl']:.0f}\\% international. Top countries: "
                         + ", ".join(f"{c} {k}" for c, k in ca["countries"][:5]) + ".")

    themes = A.thematic_map(d)
    v["THEMATIC"], mx, my = thematic_tikz(themes)
    v["THEME_ROWS"] = "\n".join(
        f"\\textcolor{{{COLORS[i % len(COLORS)]}}}{{\\textbf{{{i + 1}}}}} & {tex(', '.join(t['members'][:3]), 70)} & {quadrant(t, mx, my)} \\\\"
        for i, t in enumerate(themes))
    motor = [t for t in themes if quadrant(t, mx, my) == "motor"]
    v["MOTOR"] = "; ".join(tex(t["members"][0]) for t in motor) or "--"

    kw = A.keyword_network(d)
    evo = kw["evolution"].sort_values("change", ascending=False)
    periods = [p for p, _, _ in A.PERIODS]
    v["EVO_HEAD"] = (" & ".join(f"\\textbf{{{p}}}" for p in periods) + r" & \textbf{Change}\\" + "\n"
                     + r"\textit{papers in period} & " + " & ".join(f"\\textit{{{kw['period_n'][p]:,}}}" for p in periods) + " &")

    def change(x):
        col, sign = ("c4", "+") if x >= 0 else ("c2", "")
        return f"\\textcolor{{{col}}}{{${sign}{x:.1f}$}}"

    v["EVO_ROWS"] = "\n".join(
        f"{tex(w_, 40)} & " + " & ".join(f"\\cellcolor{{c1!{min(60, int(r[p]))}}}{r[p]:.1f}" for p in periods)
        + f" & {change(r['change'])} \\\\" for w_, r in evo.head(12).iterrows())
    v["EVO_UP"] = tex(evo.index[0])
    v["EVO_DOWN"] = tex(evo.head(12).index[-1])

    # --- who says what, why, when
    ds, ntok = A.distinctive(d, ROOT)
    groups = list(A.GROUPS)
    v["DIST_HEAD"] = " & ".join(f"\\textbf{{{A.GROUPS[g]}}} \\scriptsize({ntok[g]:,} words)" for g in groups)
    v["DIST_ROWS"] = "\n".join(
        " & ".join(f"{tex(ds[g][i][0])} \\hfill {ds[g][i][1]:.0f}" if i < len(ds[g]) else "" for g in groups) + r" \\"
        for i in range(8))
    v["DIST_TAKE"] = "; ".join(f"{A.GROUPS[g].lower()}: \\emph{{{', '.join(tex(w_) for w_, _ in ds[g][:3])}}}" for g in groups)

    mo = A.motivations(d, ROOT)
    mt = A.with_covariates(mo["table"].join(d.set_index("record_id")[["language"]], on="record_id"))
    adj = {k: A.adjusted(mt, k) for k in A.MOTIVES}
    tab = mo["by_period"].copy()
    tab["All"] = mo["overall"]
    tab = tab.sort_values("All", ascending=False)

    def eff(e):
        mark = "\\textbf" if e[1] > 0 or e[2] < 0 else ""
        return f"{mark}{{{e[0]:+.1f}}} \\tiny[{e[1]:+.0f}, {e[2]:+.0f}]"
    v["MOTIVE_HEAD"] = (" & ".join(f"\\textbf{{{c}}}" for c in periods)
                        + r" & \textbf{Change} & \textbf{Central banks}")
    v["MOTIVE_ROWS"] = "\n".join(
        f"{tex(k, 40)} & " + " & ".join(f"\\cellcolor{{c1!{min(60, int(r[c] * 1.5))}}}{r[c]:.1f}" for c in periods)
        + f" & {eff(adj[k]['period'])} & {eff(adj[k]['actor'])} \\\\" for k, r in tab.iterrows())
    v["MOTIVE_N"], v["MOTIVE_ANY"] = f"{mo['n']:,}", f"{100 * mo['n_any']:.0f}"
    sig = {k: e["period"] for k, e in adj.items() if e["period"][1] > 0 or e["period"][2] < 0}
    order = sorted(sig, key=lambda k: sig[k][0])
    v["MOTIVE_UP"] = ", ".join(f"{tex(k.lower())} (${sig[k][0]:+.0f}$\\,pts)" for k in order[::-1][:3] if sig[k][0] > 0)
    v["MOTIVE_DOWN"] = ", ".join(f"{tex(k.lower())} (${sig[k][0]:+.0f}$\\,pts)" for k in order[:2] if sig[k][0] < 0)
    act = {k: e["actor"] for k, e in adj.items()}
    v["MOTIVE_GAP"] = (f"central banks stress \\emph{{{tex(max(act, key=lambda k: act[k][0]))}}}, academics "
                       f"\\emph{{{tex(min(act, key=lambda k: act[k][0]))}}}")
    val_p = OUT / "validation_results.csv"  # held-out test set: see presentation/validation_motivations_test.csv
    val = pd.read_csv(val_p, index_col=0) if val_p.exists() else None
    if val is not None:
        a = val.loc["All (micro)"]
        v["VAL_LINE"] = (f"Validated on {int(len(pd.read_csv(OUT / 'validation_motivations_test.csv')))} held-out texts "
                         f"(reading-based coding, AI-assisted): precision {a.precision:.2f}, recall {a.recall:.2f}, "
                         f"$\\kappa = {a.kappa:.2f}$.")
        v["VAL_CAVEAT"] = (f"Recall $\\approx {a.recall:.2f}$: shares are \\alert{{lower bounds}}; "
                           "compare motivations and periods rather than read levels.")
        v["VAL_ROWS"] = "\n".join(
            f"{tex(k)} & {int(r.manual)} & {int(r.auto)} & {r.precision:.2f} & {r.recall:.2f} & {r.kappa:.2f} \\\\"
            for k, r in val.drop(index="All (micro)").sort_values("kappa", ascending=False).iterrows())
        v["VAL_ALL"] = (f"\\textbf{{All}} & {int(a.manual)} & {int(a.auto)} & {a.precision:.2f} & {a.recall:.2f} & "
                        f"{a.kappa:.2f} \\\\")
    else:
        v["VAL_LINE"] = "Validation pending."
        v["VAL_CAVEAT"] = v["VAL_ROWS"] = v["VAL_ALL"] = ""

    ev, weekly = A.event_study(d)
    pl = A.placebo(d)
    ev["p"] = [float((pl["z"] >= z).mean()) for z in ev.z]  # empirical p-value against placebo dates
    weekly = weekly[weekly.index >= pd.Period("2019-01-01", "W")]
    dec = lambda p: p.start_time.year + (p.start_time.dayofyear - 1) / 365.25
    pts = [f"({dec(p):.3f},{c})" for p, c in weekly.items()]
    v["WEEKLY"] = "\n  ".join(" ".join(pts[i:i + 12]) for i in range(0, len(pts), 12))  # short lines for linters
    marks = []
    for i, r in enumerate(ev.itertuples(), 1):
        t0 = pd.Timestamp(r.date)
        x = t0.year + (t0.dayofyear - 1) / 365.25
        top = "\\ymaxev" if i % 2 else "0.9*\\ymaxev"
        marks.append(f"\\draw[{'c2' if r.p < 0.05 else 'gray'}, thin] (axis cs:{x:.3f},0) -- (axis cs:{x:.3f},{top}) "
                     f"node[above, font=\\scriptsize, inner sep=1pt] {{{i}}};")
    v["EV_MARKS"] = "\n".join(marks)
    v["EV_YMAX"] = str(int(weekly.max() * 1.08) + 1)
    v["EV_ROWS"] = "\n".join(
        f"{i} & {pd.Timestamp(r.date):%b %Y} & {tex(r.event)} & "
        + (f"\\textbf{{{r.ratio:.1f}}}" if r.p < 0.05 else f"{r.ratio:.1f}") + f" & {r.z:+.1f} & {r.p:.2f} \\\\"
        for i, r in enumerate(ev.itertuples(), 1))
    sig = ev[ev.p < 0.05].sort_values("z", ascending=False)
    v["EV_NSIG"], v["EV_NTOT"] = str(len(sig)), str(len(ev))
    v["EV_TAKE"] = (f"{len(sig)} of {len(ev)} events beat 95\\% of placebo dates; strongest: "
                    + ", ".join(tex(e) for e in sig.event.head(2)) + ".")

    v["PERIOD_HEAD"] = " & ".join(f"\\textbf{{{p}}}" for p in periods)
    v["TONE_XTICKS"] = ",".join(map(str, range(2019, today.year + 1)))
    v["EV_P"] = f"{pl['share_sig'] * 100:.0f}"
    v["EV_Z95"] = f"{pl['z95']:.1f}"
    v["EV_NPL"] = str(pl["n"])

    # --- when: keyword bursts (Kleinberg), drawn as a timeline
    bu = A.bursts(d)
    q0, q1 = pd.Period("2016Q1", "Q"), pd.Timestamp.today().to_period("Q")
    span_q = (q1 - q0).n
    xq = lambda q: 6.6 * (pd.Period(q, "Q") - q0).n / span_q
    bl = [r"\begin{tikzpicture}[font=\scriptsize, y=-0.42cm]"]
    for yr in range(2016, today.year + 1):
        x = xq(f"{yr}Q1")
        bl.append(f"\\draw[gray!25] ({x:.2f},-0.6) -- ({x:.2f},{len(bu) - 0.4:.2f}); \\node[gray, above, font=\\tiny] at ({x:.2f},-0.6) {{{yr}}};")
    w_max = bu.weight.max()
    for i, r in enumerate(bu.itertuples()):
        x0, x1 = xq(r.start), xq(r.end) + 6.6 / span_q
        bl.append(f"\\node[anchor=east] at (-0.1,{i}) {{{tex(r.keyword, 30)}}};")
        bl.append(f"\\fill[c2, opacity={0.35 + 0.6 * r.weight / w_max:.2f}] ({x0:.2f},{i - 0.3}) rectangle ({x1:.2f},{i + 0.3});")
    bl.append(r"\end{tikzpicture}")
    v["BURSTS"] = "\n".join(bl)
    v["BURST_EARLY"] = ", ".join(tex(k) for k in bu[bu.start < "2020Q1"].keyword.head(3))
    v["BURST_LATE"] = ", ".join(tex(k) for k in bu[bu.start >= "2025Q1"].keyword.head(4))

    # --- paper framing: earlier reviews, found in the corpus itself
    rv = d[(d.source_type == "academic")
           & d.title.str.contains(r"bibliometr|scientometr|systematic\s+(?:literature\s+)?review|literature\s+review|"
                                  r"mapping\s+study|text\s+mining", case=False, na=False)
           & d.title.str.contains(r"CBDC|central\s+bank\s+digital|digital\s+(?:euro|yuan|rupee|currenc)", case=False, na=False)]
    rv = rv[~rv.title.str.lower().str.replace(r"\W", "", regex=True).duplicated()]
    v["N_REVIEWS"], v["REV_Y0"], v["REV_Y1"] = str(len(rv)), str(int(rv.year.min())), str(int(rv.year.max()))

    # --- validation of the relevance filter (hand-coded stratified sample)
    s = pd.read_csv(OUT / "validation_relevance.csv")
    s = s[s["manual: CBDC is a theme"].notna()]
    s["y"] = s["manual: CBDC is a theme"].astype(int)
    g = s.groupby("relevance").y.agg(["size", "sum"])
    g["N"] = rel.reindex(g.index).fillna(0)
    g["est"] = g["sum"] / g["size"] * g.N
    th = [k for k in RELEVANT if k in g.index]
    prec, rec = g.loc[th, "est"].sum() / g.loc[th, "N"].sum(), g.loc[th, "est"].sum() / g.est.sum()

    def wilson(k, n, z=1.96):
        p, den = k / n, 1 + z * z / n
        c, h = (p + z * z / (2 * n)) / den, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return c - h, c + h

    def rel_row(k):
        if k not in g.index:
            return "-- & --"
        lo, hi = wilson(g.loc[k, "sum"], g.loc[k, "size"])
        return f"{int(g.loc[k, 'sum'])}/{int(g.loc[k, 'size'])} & {max(lo, 0):.2f}--{min(hi, 1):.2f}"
    v["REL_ROWS"] = "\n".join(f"{REL_LABEL.get(k, k)} & {rel.get(k, 0):,} & {rel_row(k)} \\\\"
                              for k in REL_LABEL if rel.get(k, 0))
    v["REL_PREC"], v["REL_REC"], v["REL_N"] = f"{prec:.2f}", f"{rec:.2f}", str(int(g["size"].sum()))
    v["REL_MISSED"] = f"{g.loc['unverified (no text)', 'est']:,.0f}" if "unverified (no text)" in g.index else "?"

    # --- bibliometric laws
    L = A.laws(d)
    v["LAWS"] = (f"Growth {100 * L['cagr']:.0f}\\%/year ({L['y0']}--{L['y1']}). "
                 f"Lotka: $f(n) \\propto n^{{-{L['beta']:.1f}}}$, {100 * L['single']:.0f}\\% of authors publish once. "
                 f"Bradford: {L['bradford'][0]} core sources hold \\nicefrac{{1}}{{3}} of the papers "
                 f"(zones {L['bradford'][0]} : {L['bradford'][1]:,} : {L['bradford'][2]:,}).")
    v["CAGR"] = f"{100 * L['cagr']:.0f}"

    # --- topics (SBERT + BERTopic, c-TF-IDF labels) and what is new in each period
    tp = nlp.topics(d, ROOT, ROOT / "data/cache/embeddings.npz")
    tt = tp["table"]
    jur = lambda terms: any(w in JURISDICTION for w in terms[:2])
    shown = tt.head(12)
    tf = A.with_covariates(d[d.source_type.isin(nlp.NON_PRESS)].merge(tp["assign"], on="record_id"))
    tadj = {}
    for k in shown.topic:
        tf["is_k"] = tf.topic == k
        tadj[k] = A.adjusted(tf, "is_k")

    def teff(e):
        mark = "\\textbf" if e[1] > 0 or e[2] < 0 else ""
        return f"{mark}{{{e[0]:+.1f}}}"
    v["TOPIC_ROWS"] = "\n".join(
        f"{i} & {tex(', '.join(r['terms'][:4]), 46)} & {r['n']:,} & "
        + " & ".join(f"\\cellcolor{{c1!{min(60, int(r[p] * 3))}}}{r[p]:.1f}" for p in periods)
        + f" & {teff(tadj[r['topic']]['period'])} & {teff(tadj[r['topic']]['actor'])} \\\\"
        for i, (_, r) in enumerate(shown.iterrows(), 1))
    j_share = {p: tt[tt.terms.apply(jur)][p].sum() for p in periods}
    v["TOPIC_N"], v["TOPIC_K"] = f"{tp['n_docs']:,}", str(len(tt))
    v["TOPIC_OUT"] = f"{100 * tp['n_outliers'] / tp['n_docs']:.0f}"
    jt = tt[tt.terms.apply(jur)].copy()
    jt["delta"] = jt[periods[-1]] - jt[periods[0]]
    fall, rise_ = jt[jt.delta < -1].sort_values("delta"), jt[jt.delta > 1].sort_values("delta", ascending=False)
    lo_, hi_ = sorted([j_share[periods[0]], j_share[periods[-1]]])
    v["TOPIC_JUR"] = (f"{lo_:.0f}--{hi_:.0f}\\% of texts in every period, but the mix shifts: "
                      + ", ".join(f"\\emph{{{tex(t)}}}" for t in fall.terms.str[0].head(2)) + " fall; "
                      + ", ".join(f"\\emph{{{tex(t)}}}" for t in rise_.terms.str[0].head(3)) + " rise")
    pt = nlp.period_terms(d, ROOT)
    v["PERIOD_TERMS"] = "; ".join(f"\\textbf{{{p}}}: \\emph{{{tex(', '.join(pt[p][:6]))}}}" for p in periods)

    # --- why: the codebook against SBERT, and robustness of the trends
    dev, test = pd.read_csv(OUT / "validation_motivations_dev.csv"), pd.read_csv(OUT / "validation_motivations_test.csv")
    vc = pd.read_csv(OUT / "validation_coders.csv", index_col=0)  # written by scripts/validate_coders.py
    vmot, vst = vc[vc.task == "motivations"], vc[vc.task == "stance"]
    v["MVAL_ROWS"] = "\n".join(("\\rowcolor{c4!12}" if k.startswith("Codebook (") else "")
                               + f"{k} & {r.precision:.2f} & {r.recall:.2f} & {r.f1:.2f} & {r.kappa:.2f} \\\\"
                               for k, r in vmot.sort_values("kappa", ascending=False).iterrows())
    k_cb = vmot.loc["Codebook (regex)", "kappa"]
    k_alt = vmot.drop(index=[i for i in vmot.index if i.startswith("Codebook")]).kappa
    v["MVAL_BEST_ALT"] = f"{k_alt.max():.2f}"
    v["MVAL_CB"] = f"{k_cb:.2f}"
    v["MVAL_ALT_LIST"] = ", ".join(f"{i.split(' (')[0].replace('Local LLM, ', 'LLM ')} {r:.2f}" for i, r in k_alt.sort_values(ascending=False).items())
    v["ST_ALT_LIST"] = ", ".join(f"{i.split(' (')[0].replace('Local LLM, ', 'LLM ')} {r:.2f}"
                                 for i, r in vst.kappa.sort_values(ascending=False).items())
    mu = nlp.motivations_union(d, ROOT, dev)
    mm = A.motivations(d[d.relevance == "main theme"], ROOT)
    sign = lambda bp: np.sign(bp[periods[-1]] - bp[periods[0]])
    base = sign(mo["by_period"])
    v["MOTIVE_ROBUST"] = (f"same direction of change for {int((sign(mu['by_period']) == base).sum())} of {len(base)} "
                          f"motivations with the high-recall coding, {int((sign(mm['by_period']) == base).sum())} of "
                          f"{len(base)} on main-theme texts only.")

    # --- how: tone of the sentences that mention CBDCs (FinBERT)
    snt = nlp.tone(d, ROOT, ROOT / "data/cache/tone.csv")
    snt = snt[snt.record_id.isin(d.record_id)].copy()
    snt["actor"] = snt.source_type.replace({"official": "speech"})
    doc = snt.groupby(["record_id", "actor"]).agg(tone=("tone", "mean"), year=("year", "first")).reset_index()
    actors = [("academic", "Academics", "c1"), ("speech", "Central banks", "c2"), ("press", "Press", "c4")]
    tyrs = list(range(2019, today.year + 1))
    v["TONE_PLOTS"] = "\n".join(
        f"\\addplot[{c}, thick, mark=*, mark size=1.2pt] coordinates {{"
        + " ".join(f"({y},{doc[(doc.actor == a) & (doc.year == y)].tone.mean():.3f})" for y in tyrs
                   if ((doc.actor == a) & (doc.year == y)).sum() >= 20)
        + f"}};\n\\addlegendentry{{{lbl}}}" for a, lbl, c in actors)
    trow = []
    for a, lbl, _ in actors:
        x, ss = doc[doc.actor == a].tone, snt[snt.actor == a]
        trow.append(f"{lbl} & {len(x):,} & {len(ss):,} & {100 * (ss.label == 'positive').mean():.0f} & "
                    f"{100 * (ss.label == 'negative').mean():.0f} & ${x.mean():.2f} \\pm {1.96 * x.std() / math.sqrt(len(x)):.2f}$ \\\\")
    v["TONE_ROWS"] = "\n".join(trow)
    tv = pd.read_csv(OUT / "validation_tone.csv")
    labs = ["negative", "neutral", "positive"]
    po = (tv.label == tv.manual).mean()
    pe = sum((tv.label == l).mean() * (tv.manual == l).mean() for l in labs)
    v["TONE_VAL"] = (f"{len(tv)} sentences read (30 per predicted label): agreement {100 * po:.0f}\\%, "
                     f"$\\kappa = {(po - pe) / (1 - pe):.2f}$; positive and negative confused "
                     f"{int(((tv.label == 'positive') & (tv.manual == 'negative')).sum() + ((tv.label == 'negative') & (tv.manual == 'positive')).sum())} times.")
    v["TONE_XTAB"] = "\n".join(f"{l} & " + " & ".join(str(int(((tv.label == l) & (tv.manual == m)).sum())) for m in labs)
                               + f" & {((tv.label == l) & (tv.manual == l)).sum() / max((tv.label == l).sum(), 1):.2f} \\\\" for l in labs)
    v["N_SENT"] = f"{len(snt):,}"
    v["TONE_KAPPA"] = f"{(po - pe) / (1 - pe):.2f}"
    ac = doc[(doc.actor == "academic")]
    v["TONE_ACAD"] = (f"{ac[ac.year == tyrs[0]].tone.mean():.2f} in {tyrs[0]} to "
                      f"{ac[ac.year == today.year - 1].tone.mean():.2f} in {today.year - 1}")

    # --- how (2): stance. Levels from the hand-coded texts (50 per actor); changes over time from a local LLM
    # on a stratified sample (its bias towards "favourable" inflates levels but largely cancels in differences)
    ss = nlp.stance_sample(d, ROOT, ROOT / "data/cache/stance_sample.csv")  # cached: codes only what is missing
    sv = pd.read_csv(OUT / "validation_stance.csv").merge(pd.read_csv(OUT / "validation_stance_llm.csv"), on="record_id")
    sv["actor"] = sv.source_type.replace({"official": "speech"})
    vst_k = vst.loc["Local LLM, Qwen2.5 7B", "kappa"]
    level, trend = [], []
    for a, lbl in [("academic", "Academics"), ("speech", "Central banks"), ("press", "Press")]:
        h = sv[sv.actor == a]["manual: stance"]
        cell_ = lambda lab: (lambda lo_hi: f"{100 * (h == lab).mean():.0f} \\tiny[{100 * lo_hi[0]:.0f}--{100 * lo_hi[1]:.0f}]")(
            wilson((h == lab).sum(), len(h)))
        level.append(f"{lbl} & {len(h)} & {cell_('favourable')} & {cell_('critical')} \\\\")
        fav = [(ss[(ss.actor == a) & (ss.period == p_)].stance == "favourable") for p_ in periods]
        f0, f1 = fav[0].mean(), fav[-1].mean()
        se = math.sqrt(f0 * (1 - f0) / len(fav[0]) + f1 * (1 - f1) / len(fav[-1]))
        dlt, lo_, hi_ = 100 * (f1 - f0), 100 * (f1 - f0 - 1.96 * se), 100 * (f1 - f0 + 1.96 * se)
        mark = "\\textbf" if lo_ > 0 or hi_ < 0 else ""
        trend.append(f"{lbl} & " + " & ".join(f"{100 * x.mean():.0f}" for x in fav)
                     + f" & {mark}{{{dlt:+.0f}}} \\tiny[{lo_:+.0f}, {hi_:+.0f}] \\\\")
    v["STANCE_LEVEL"], v["STANCE_TREND"] = "\n".join(level), "\n".join(trend)
    v["STANCE_N"] = f"{len(ss):,}"
    v["STANCE_K"] = f"{vst_k:.2f}"
    v["STANCE_RAWFAV"] = f"{100 * (sv.llm == 'favourable').mean():.0f}"
    v["STANCE_TRUEFAV"] = f"{100 * (sv['manual: stance'] == 'favourable').mean():.0f}"
    hc = sv.groupby("actor")["manual: stance"].value_counts(normalize=True).unstack().fillna(0)
    v["STANCE_TAKE"] = (f"By reading: {100 * hc.loc['speech', 'favourable']:.0f}\\% of central bank texts are favourable, "
                        f"{100 * hc.loc['academic', 'favourable']:.0f}\\% of research; criticism is rare and mostly in the press "
                        f"({100 * hc.loc['press', 'critical']:.0f}\\%)")

    # --- robustness: the headline results on alternative samples
    ids_all = set(d.record_id)
    journal = set(d[(d.source_type != "academic") | (d.document_type == "article")].record_id)
    stance_doc = ss.assign(actor=ss.actor)
    rob = robustness(d, mo["table"], doc, stance_doc, {
        "Baseline": (ids_all, None),
        "Main theme only": (set(d[d.relevance == "main theme"].record_id), None),
        "English only": (set(d[d.language == "en"].record_id), None),
        "Journal articles only (research)": (journal, None),
        f"Without {today.year} (partial year)": (set(d[d.year < today.year].record_id), None),
        "High-recall motivation coding": (ids_all, mu["table"]),
    })

    def cell(e):
        return "--" if e != e or (isinstance(e, tuple) and e[0] != e[0]) else (
            f"${e[0]:+.1f}$ \\tiny[{e[1]:+.0f}, {e[2]:+.0f}]" if isinstance(e, tuple) else f"${e:+.2f}$")
    v["ROB_ROWS"] = "\n".join(
        f"{lbl} & {n:,} & {cell(inc)} & {cell(mp)} & {cell(gap)} & ${fav:+.0f}$ & {k}/{K} \\\\"
        for lbl, n, inc, mp, gap, fav, k, K in rob)

    body = TEMPLATE
    for key, val in v.items():
        body = body.replace(f"<<{key}>>", val)
    missing = re.findall(r"<<[A-Z_0-9]+>>", body)
    assert not missing, missing
    OUT.mkdir(exist_ok=True)
    body = GENERATED + body
    (OUT / "cbdc_corpus.tex").write_text(body, encoding="utf-8")
    notes = body.replace(r"\begin{document}", "\\setbeameroption{show only notes}\n\\begin{document}", 1)
    (OUT / "cbdc_corpus_notes.tex").write_text(notes, encoding="utf-8")
    build = OUT / ".build"  # compile aside, then swap the PDF in atomically: an open viewer never sees a half-written file
    for f in ("cbdc_corpus", "cbdc_corpus_notes"):
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={build}", f"{f}.tex"],
                       cwd=OUT, check=True, stdout=subprocess.DEVNULL)
        tmp = OUT / f".{f}.pdf.tmp"
        shutil.copyfile(build / f"{f}.pdf", tmp)
        os.replace(tmp, OUT / f"{f}.pdf")
    print(f"Wrote {OUT / 'cbdc_corpus.pdf'} and the speaker notes {OUT / 'cbdc_corpus_notes.pdf'}")


GENERATED = "% Final version, generated from data/cbdc_corpus.csv by scripts/make_slides.py.\n"

TEMPLATE = r"""\PassOptionsToPackage{table}{xcolor}
\documentclass[aspectratio=169,10pt]{beamer}
\usetheme[progressbar=frametitle,sectionpage=progressbar]{metropolis}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,booktabs,pgfplots,listings,textcomp,nicefrac}
\usetikzlibrary{positioning,arrows.meta}
\pgfplotsset{compat=1.18}
\definecolor{c1}{HTML}{23395B}\definecolor{c2}{HTML}{D1495B}\definecolor{c3}{HTML}{EDAE49}
\definecolor{c4}{HTML}{00798C}\definecolor{c5}{HTML}{8F6593}\definecolor{c6}{HTML}{6A994E}
\definecolor{c7}{HTML}{E07A5F}\definecolor{c8}{HTML}{3D5A80}\definecolor{codebg}{HTML}{F3F3F0}
\pgfplotscreateplotcyclelist{corpus}{{fill=c1,draw=c1},{fill=c2,draw=c2},{fill=c3,draw=c3},{fill=c4,draw=c4}}
\lstset{language=Python, basicstyle=\ttfamily\fontsize{6.6}{7.6}\selectfont, keywordstyle=\color{c1}\bfseries,
  stringstyle=\color{c4}, backgroundcolor=\color{codebg}, frame=none, showstringspaces=false,
  columns=fullflexible, keepspaces=true, xleftmargin=4pt, framexleftmargin=4pt,
  upquote=true, literate={é}{{\'e}}1 {è}{{\`e}}1 {’}{{\textquotesingle}}1}
\tikzset{box/.style={draw=c1, rounded corners=2pt, fill=white, align=center, font=\scriptsize, minimum height=7mm,
  text width=19mm}, arr/.style={-{Stealth[length=2mm]}, c1, thick}}
\setbeamertemplate{note page}{\pagecolor{white}\vspace{6mm}\hspace{6mm}\begin{minipage}{0.9\paperwidth}\large\insertnote\end{minipage}}
\newcommand{\ymaxev}{<<EV_YMAX>>}

\title{A Bibliometric Corpus on Central Bank Digital Currencies}
\subtitle{Data, methods and first results of a paper in progress}
\author{Adriel Mazere}
\institute{M2 -- Private crypto}
\date{<<DATE>>}
\begin{document}
\begin{frame}
\titlepage
\note{\textbf{Title (0:15). Total: about 23 minutes.} This talk presents the groundwork of a paper: a bibliometric and text-as-data study of everything written on central bank digital currencies.}
\end{frame}

\begin{frame}{The paper in one slide}
\textbf{Research question.} Why do researchers, central bankers and journalists write about CBDCs, and how has the agenda shifted since 2014?
\\[3pt]
\alert{What} is written? \quad \alert{Who} writes? \quad \alert{Why} (stated motivations)? \quad \alert{How} (tone)? \quad \alert{When} (around which events)?\\[6pt]
\textbf{Contribution.}
\begin{enumerate}\setlength\itemsep{2pt}
\item \alert{Data}: research, speeches, official documents and press, in <<N_LANG>> languages: <<TIMES>> times the largest earlier review.
\item \alert{Measurement}: relevance filter, motivations and tone, each checked against hand coding.
\item \alert{Dynamics}: attention tested against policy events with placebo inference.
\end{enumerate}
\vfill
\begin{center}\large <<N_ALL>> collected \quad$\to$\quad \alert{<<N>>} with CBDC as a theme \quad$\cdot$\quad <<Y0>>--<<Y1>>\end{center}
\note{\textbf{The paper in one slide (1:00).}
\begin{itemize}
\item One question: why do people write about CBDCs, and how has that changed?
\item Five sub-questions structure the talk: what, who, why, how, when.
\item Three contributions: the data, validated measures, and the timing of attention.
\item Headline: <<N_ALL>> texts collected, <<N>> keep CBDC as a theme.
\end{itemize}}
\end{frame}

\begin{frame}{Related work: <<N_REVIEWS>> reviews already exist}
\footnotesize Five recent bibliometric reviews:\\[2pt]
\begin{tabular}{@{}llrl@{}}\toprule
\textbf{Study} & \textbf{Data} & \textbf{Documents} & \textbf{Tools}\\\midrule
Kvedaravi\v{c}i\={u}t\.{e} \& \v{S}apkauskien\.{e} (2024) & WoS + Scopus & 228 & VOSviewer\\
Fahad \& Bulut (2024) & Scopus & 169 of 323 & Biblioshiny\\
Bapat \& Sharma (2025) & Scopus & 371 & bibliometrics + TCCM\\
Prodan et al.\ (2024) & Scopus + WoS & 747 of 1{,}024 & PRISMA, VOSviewer\\
Paulovici (2026) & Scopus & 892 & VOSviewer\\\midrule
\textbf{This study} & \textbf{5 databases, speeches, press\dots} & \textbf{<<N>>} & \textbf{+ text as data}\\
\bottomrule\end{tabular}\\[6pt]
\small
\begin{itemize}\setlength\itemsep{1pt}
\item \alert{Gap}: one or two databases (Scopus, WoS), mostly English journal articles: central bankers and the press are absent.
\item Maps say \emph{what} is studied, not \emph{why}; no measure is validated.
\end{itemize}
\scriptsize Found in our own corpus: <<N_REVIEWS>> titles announce a literature, systematic or bibliometric review of CBDCs (<<REV_Y0>>--<<REV_Y1>>).
\note{\textbf{Related work (0:50).}
\begin{itemize}
\item The topic is not new for bibliometrics: about <<N_REVIEWS>> reviews, most of them since 2023.
\item The bibliometric ones rely on Scopus or Web of Science: a few hundred journal articles, often English only (Prodan et al.\ drop non-English texts).
\item The closest to us is Hoang et al.\ (2023, RIBAF): text mining, but still academic only.
\item Our angle: every type of text, every language, and measures that are validated.
\end{itemize}}
\end{frame}

\begin{frame}{Outline}
\tableofcontents
\note{\textbf{Outline (0:15).} Method first (collection, scoring), then results (corpus, maps, why and when).}
\end{frame}

% =====================================================================
\section{Collecting the texts}

\begin{frame}{Pipeline}
\begin{center}
\begin{tikzpicture}[node distance=4mm and 7mm]
\node[box] (s1) {Academic APIs\\\tiny OpenAlex, Semantic Scholar,\\\tiny arXiv, HAL, DOAJ};
\node[box, below=of s1] (s2) {Speeches\\\tiny BIS (1996--), ECB};
\node[box, below=of s2] (s3) {Authorities\\\tiny central banks, EU, IMF\dots};
\node[box, below=of s3] (s4) {Press\\\tiny Google News, GDELT};
\node[box, right=of s2, yshift=-5.5mm, text width=16mm] (raw) {\textbf{harvest}\\\tiny one JSONL\\\tiny per source};
\node[box, right=of raw, text width=16mm] (ft) {\textbf{full text}\\\tiny PDF / HTML\\\tiny extraction};
\node[box, right=of ft, text width=21mm] (bd) {\textbf{build}\\\tiny de-duplicate, language,\\\tiny sections, keywords,\\\tiny relevance score};
\node[box, right=of bd, text width=15mm, fill=c1!10] (csv) {\textbf{one CSV}\\\tiny <<N_COLS>> columns};
\foreach \s in {s1,s2,s3,s4} \draw[arr] (\s.east) -- (raw.west);
\draw[arr] (raw) -- (ft); \draw[arr] (ft) -- (bd); \draw[arr] (bd) -- (csv);
\end{tikzpicture}
\end{center}
\small
\begin{itemize}
\item 115 search phrases, 12 languages: \emph{CBDC, digital euro, e-CNY, eNaira, Drex, digital fiat currency, Fedcoin, euro numérique, digitaler Euro, digital ruble\dots}
\item \alert{Recall first}: nothing is filtered at harvest time; every step is resumable.
\end{itemize}
\note{\textbf{Pipeline (0:45).}
\begin{itemize}
\item Four families of sources, each harvested into its own raw file.
\item Then full text is downloaded and parsed, then everything is merged into one CSV.
\item Search vocabulary includes older names (digital fiat currency, Fedcoin) to reach the pre-2017 literature.
\item Design choice: collect broadly, decide relevance later.
\end{itemize}}
\end{frame}

\begin{frame}[fragile]{Press: recursive bisection of time windows}
\begin{columns}[T]
\begin{column}{0.53\textwidth}
\begin{lstlisting}
stack = monthly_windows("2000-01-01", today)
while stack:
    a, b = stack.pop()
    items = fetch(query, after=a, before=b)
    if len(items) >= CAP and (b - a).days > 1:
        m = a + (b - a) / 2
        stack += [(m, b), (a, m)]
        continue
    save(items)
    checkpoint(query, a, b, len(items))
\end{lstlisting}
\end{column}
\begin{column}{0.43\textwidth}\small
A search returns at most $C=100$ items. For a window $W=[a,b)$ with $N(W)$ articles:
\begin{itemize}
\item $N(W) < 95$ or $|W| \leq 1$ day: \alert{keep} $W$;
\item otherwise \alert{split} at $m = a + \frac{b-a}{2}$.
\item Depth $\leq \lceil \log_2 31 \rceil = 5$ per month; $2L-1$ requests for $L$ windows kept.
\item 55 searches (country editions $\times$ languages).
\end{itemize}
\end{column}
\end{columns}
\note{\textbf{Press harvesting (0:45).}
\begin{itemize}
\item Problem: a news search returns at most 100 results, but a busy month has thousands.
\item Solution: if a month is full, cut it in two, and repeat until each window holds fewer than the cap, down to one day.
\item A month needs at most 5 levels of splitting; the cost is $2L-1$ requests.
\end{itemize}}
\end{frame}

% =====================================================================
\section{Scoring relevance}

\begin{frame}[fragile]{Step 1: is it really \emph{Central Bank Digital Currency}?}
\begin{columns}[T]
\begin{column}{0.5\textwidth}
\begin{lstlisting}
TERMS = {
  "CBDC": r"\bCBDCs?\b|\bwCBDCs?\b|\brCBDCs?\b",
  "central bank digital currency":
      r"central[\s-]+banks?(?:['’]s?)?[\s-]+"
      r"digital[\s-]+currenc(?:y|ies)",
  "digital euro": r"digital(?:e[rn]?)?\s+euro\b"
      r"|euro\s+numérique|euro\s+digitale?\b",
  "DCEP (e-CNY)": r"\bDC/?EP\b",
  ...
}
\end{lstlisting}
\end{column}
\begin{column}{0.47\textwidth}\small
\begin{itemize}
\item ``CBDC'' is also a \emph{skin disease}; ``DCEP'' a chemical.
\item Ambiguous terms $t\in\mathcal{A}$ count only with \alert{monetary context}:
\end{itemize}
\[
\text{count}(t) = n_t \cdot \mathbf{1}\big[t \notin \mathcal{A} \,\lor\, \text{context}\big]
\]
\begin{itemize}
\item context = \emph{currency, payment, central bank, monnaie, Geld\dots}
\item \emph{digital dollar}, \emph{Sand Dollar}: reported, never scored.
\end{itemize}
\end{column}
\end{columns}
\note{\textbf{Homonyms (0:45).}
\begin{itemize}
\item Each concept is a regular expression; the same list drives searching and scoring.
\item ``CBDC'' also means \emph{chronic bullous dermatosis of childhood}: an ambiguous term only counts if the text also talks about money or payments.
\end{itemize}}
\end{frame}

\begin{frame}{Step 2: how central is CBDC to the text?}
Each available field $f$ gets a centrality $c_f \in [0,1]$:
\begin{align*}
c_{\text{title}} &= \mathbf{1}[\text{CBDC term in title}] &
c_{\text{kw}} &= 0.4\cdot\mathbf{1}[\text{CBDC in keywords}]\\
c_{\text{abs}} &= \frac{\#\{\text{sentences mentioning CBDC}\}}{\#\{\text{sentences}\}} &
c_{\text{body}} &= \tfrac12\min\!\Big(1,\frac{\rho}{3}\Big) + \tfrac12\,\sigma
\end{align*}
$\rho$ = mentions per 1{,}000 words; $\sigma$ = share of the ten slices of the text that mention CBDC.
\[
\boxed{\text{score} = 100\Big(0.6\,\max_{f\in F} c_f + 0.4\,\frac{1}{|F|}\sum_{f\in F} c_f\Big)}
\qquad F = \text{fields available}
\]
\small One field clearly about CBDC suffices; a passing mention stays low; missing fields are not penalised.
\note{\textbf{Theme score (0:45).}
\begin{itemize}
\item Not ``does the word appear'' but ``is it the subject''.
\item Title: strongest signal. Abstract: share of sentences. Full text: density and spread across the document.
\item The max rewards one clear signal; the mean rewards consistency. Keywords weigh less: partly machine-assigned.
\end{itemize}}
\end{frame}

\begin{frame}{Step 3: labels, and how good is the filter?}
\begin{columns}[T]
\begin{column}{0.58\textwidth}\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrr@{}}\toprule
\textbf{Label (all texts kept)} & \textbf{Texts} & \textbf{Theme} & \textbf{95\% CI}\\\midrule
<<REL_ROWS>>
\bottomrule\end{tabular}\\[3pt]
Theme = share read as having CBDC as a theme, in a stratified sample of <<REL_N>> texts read blind to the label; Wilson intervals.
\end{column}
\begin{column}{0.4\textwidth}\small
Strata $h$ weighted by size $N_h$; $T$ = theme labels:
\[ \widehat{P} = \frac{\sum_{h\in T} N_h \hat p_h}{\sum_{h\in T} N_h}, \qquad \widehat{R} = \frac{\sum_{h\in T} N_h \hat p_h}{\sum_{h} N_h \hat p_h} \]
\begin{itemize}\setlength\itemsep{1pt}
\item Precision \alert{<<REL_PREC>>}, recall \alert{<<REL_REC>>}.
\item Main theme is reliable; \alert{secondary theme is noisy} $\Rightarrow$ robustness on main theme only.
\item Most missed texts: press without text ($\approx$ <<REL_MISSED>>).
\end{itemize}
\end{column}
\end{columns}
\vspace{2pt}
\small From here on: \alert{main + secondary theme} (<<N>> texts).
\note{\textbf{Validating the filter (0:50).}
\begin{itemize}
\item A paper needs to show the filter works: I read a stratified sample without seeing the label.
\item Main theme: almost always right. Secondary theme: only a minority really is about CBDCs, so every result is re-run on main theme only.
\item Recall loss comes mainly from press items whose text could not be fetched.
\end{itemize}}
\end{frame}

% =====================================================================
\section{The corpus}

\begin{frame}{Sources and completeness}
\begin{columns}[T]
\begin{column}{0.46\textwidth}\footnotesize
\begin{tabular}{@{}lr@{}}\toprule
\textbf{Database} & \textbf{Documents}\\\midrule
<<SRC_ROWS>>
\bottomrule\end{tabular}\\[2pt]
\scriptsize Merged duplicates count in each database.
\end{column}
\begin{column}{0.5\textwidth}\footnotesize
\begin{tabular}{@{}lr@{}}\toprule
\textbf{Type of text} & \textbf{Documents}\\\midrule
<<TYPE_ROWS>>
\midrule Total & <<N>>\\\bottomrule\end{tabular}\\[6pt]
\begin{tabular}{@{}lr@{}}\toprule
\textbf{Field available} & \textbf{Share}\\\midrule
<<QUALITY_ROWS>>
\bottomrule\end{tabular}
\end{column}
\end{columns}
\note{\textbf{Sources (0:35).}
\begin{itemize}
\item <<TOP_TYPE>> form the largest group; speeches and official documents are few but long and fully available.
\item Completeness: most texts are dated to the day; full text and references are the scarcest fields.
\end{itemize}}
\end{frame}

\begin{frame}{A literature that took off after 2020}
\begin{center}
\begin{tikzpicture}
\begin{axis}[ybar stacked, width=0.95\textwidth, height=0.68\textheight, bar width=14pt,
  cycle list name=corpus, xtick=data, xticklabel style={/pgf/number format/1000 sep=},
  ymin=0, legend style={at={(0.02,0.98)},anchor=north west,font=\scriptsize,draw=none},
  ylabel={Documents}, tick label style={font=\scriptsize}, axis lines*=left, ymajorgrids]
<<YEAR_PLOTS>>
\end{axis}
\end{tikzpicture}
\end{center}
\scriptsize <<BEFORE>> documents date from before <<YEAR0>>; <<THIS_YEAR>> runs to <<THIS_MONTH>>. Press counts follow the search engine's coverage, which favours recent articles: read trends on academic texts.
\note{\textbf{Growth (0:40).}
\begin{itemize}
\item From <<GROWTH>>.
\item Turning point 2020: COVID and the decline of cash, Libra (2019), the ECB report on a digital euro (Oct.\ 2020).\item Caution: the press bar for <<THIS_YEAR>> is mostly a harvesting effect (news search returns recent articles first); in the paper, press is normalised by its coverage.
\end{itemize}}
\end{frame}

\begin{frame}{Languages}
\begin{columns}[c]
\begin{column}{0.58\textwidth}
\begin{tikzpicture}
\begin{axis}[xbar, width=\textwidth, height=0.72\textheight, bar width=7pt, xmin=0,
  symbolic y coords={<<LANG_SYM>>}, ytick=data, tick label style={font=\scriptsize},
  axis lines*=left, nodes near coords, nodes near coords style={font=\tiny},
  xlabel={Documents}, xlabel style={font=\scriptsize}, enlarge x limits={upper,value=0.12}]
\addplot[fill=c1, draw=c1] coordinates {<<LANG_COORDS>>};
\end{axis}
\end{tikzpicture}
\end{column}
\begin{column}{0.4\textwidth}\small
\begin{itemize}
\item English dominates: <<EN_N>> texts (<<EN_PCT>>\%); chart shows the others.
\item Second language: \alert{<<LANG2>>} (<<LANG2_N>> texts, of which <<LANG2_PRESS>> press).
\item Speeches kept in their original language.
\end{itemize}
\end{column}
\end{columns}
\note{\textbf{Languages (0:25).} English dominates, but <<LANG2>> is a large second block: a reminder that English-only databases miss part of the debate.}
\end{frame}

\begin{frame}{Academic literature}
\begin{columns}[T]
\begin{column}{0.4\textwidth}\scriptsize\textbf{Main outlets}\\[2pt]
\begin{tabular}{@{}p{0.8\linewidth}r@{}}<<VENUE_ROWS>>\end{tabular}\\[4pt]
Working papers (SSRN, RePEc) outweigh journals.\\[4pt]
\textbf{Bibliometric laws.} <<LAWS>>
\end{column}
\begin{column}{0.58\textwidth}\scriptsize\textbf{Most cited papers} (global citations)\\[2pt]
\begin{tabular}{@{}p{0.2\linewidth}p{0.66\linewidth}r@{}}
<<CITED_ROWS>>
\end{tabular}
\end{column}
\end{columns}
\note{\textbf{Academic literature (0:45).}
\begin{itemize}
\item Much of the debate happens in working papers, often by central bank economists.
\item The most cited papers are macro-modelling papers from 2016--2020.
\item Laws: output grows <<CAGR>>\% a year; most authors write once (a Lotka exponent above 2): a young, open field around a small core of sources.
\end{itemize}}
\end{frame}

\begin{frame}{Who speaks: central banks, authorities, press}
<<VOICES>>
\note{\textbf{Voices (0:35).}
\begin{itemize}
\item <<TOP_BANK>> leads the speeches: the digital euro project makes it the most vocal institution.
\item The press is led by specialised fintech outlets before generalist media.
\end{itemize}}
\end{frame}

% =====================================================================
\section{Mapping the field}

\begin{frame}{Co-citation: the intellectual base}
\begin{columns}[T]
\begin{column}{0.7\textwidth}
<<COCIT_NET>>
\end{column}
\begin{column}{0.28\textwidth}\scriptsize
\textbf{Most cited within the corpus}\\[2pt]
\begin{tabular}{@{}lr@{}}<<COCIT_TOP>>\end{tabular}
\end{column}
\end{columns}
\vspace{2pt}
\scriptsize\textbf{Clusters} (largest members):
\begin{itemize}\setlength\itemsep{0pt}
<<COCIT_CLUSTERS>>
\end{itemize}
<<COCIT_STATS>>
\note{\textbf{Co-citation (0:50).}
\begin{itemize}
\item Method in one line: two items are linked when they appear in the same paper, weighted by association strength, grouped by the Louvain algorithm (details in backup).
\item Two references are linked when the same paper cites both: this shows what the field builds on.
\item Core: <<COCIT_CORE>>.
\item Clusters read as schools: CBDC design and bank intermediation models; central bank policy reports; adoption studies.
\end{itemize}}
\end{frame}

\begin{frame}{Co-authorship: research communities}
\begin{columns}[T]
\begin{column}{0.62\textwidth}
<<COAUTH_NET>>
\end{column}
\begin{column}{0.36\textwidth}\scriptsize
\textbf{Most productive authors}\\[2pt]
\begin{tabular}{@{}lr@{}}<<COAUTH_TOP>>\end{tabular}\\[6pt]
<<COAUTH_STATS>>\\[4pt]
<<CB_AUTH>>
\end{column}
\end{columns}
\note{\textbf{Co-authorship (0:45).}
\begin{itemize}
\item Small, separate teams: the field is fragmented, around institutions (BIS, IMF, academic groups).
\item Productivity is concentrated on a few very prolific authors.
\item Who sets the agenda: central bank and international-institution economists write few of the papers but collect a third of the citations, though their share of papers is shrinking as the field opens up.
\end{itemize}}
\end{frame}

\begin{frame}{Thematic map: which themes drive the field?}
\begin{columns}[c]
\begin{column}{0.47\textwidth}
<<THEMATIC>>
\end{column}
\begin{column}{0.51\textwidth}\scriptsize
Keyword clusters (Callon), with $e_{ij} = c_{ij}^2 / (c_i c_j)$:
\[ \text{centrality} = 10\!\!\sum_{i\in C,\,j\notin C}\!\! e_{ij}, \qquad \text{density} = \frac{100}{|C|}\!\sum_{i,j\in C}\! e_{ij} \]
\begin{tabular}{@{}lp{0.62\linewidth}l@{}}\toprule
 & \textbf{Theme} (top keywords) & \textbf{Quadrant}\\\midrule
<<THEME_ROWS>>
\bottomrule\end{tabular}
\end{column}
\end{columns}
\note{\textbf{Thematic map (0:50).}
\begin{itemize}
\item Centrality: how connected a theme is to the others. Density: how developed it is internally.
\item Motor themes (central and developed): <<MOTOR>>.
\item Niche themes are developed but isolated; basic themes are transversal but less structured.
\end{itemize}}
\end{frame}

\begin{frame}{How the themes evolve}
\scriptsize
\begin{tabular}{@{}p{0.42\textwidth}rrrr@{}}\toprule
\textbf{Keyword} (\% of the period's papers) & <<EVO_HEAD>>\\\midrule
<<EVO_ROWS>>
\bottomrule\end{tabular}\\[4pt]
Sorted by change between the last two periods. Darker = more frequent.\\[3pt]
\textbf{New in each period} (c-TF-IDF, periods as classes): <<PERIOD_TERMS>>.
\note{\textbf{Evolution (0:35).}
\begin{itemize}
\item Rising fastest: <<EVO_UP>>; falling most: <<EVO_DOWN>> (shares, while the field itself grows).
\item The literature moves from ``how would it work'' to ``who adopts it and for whom''.
\item Bottom line, TF-IDF answers ``when'': each period's vocabulary compared to the others.
\end{itemize}}
\end{frame}

% =====================================================================
\section{Text as data: what, why, how, when}

\begin{frame}{Toolbox: the method chosen for each question}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lp{0.43\textwidth}p{0.36\textwidth}@{}}\toprule
\textbf{Question} & \textbf{Used} & \textbf{Also tested, and why not}\\\midrule
What? & topics: \alert{BERTopic} (SBERT, UMAP, HDBSCAN), labels by \alert{c-TF-IDF}; co-word and Callon map; \alert{covariate logit} on topic shares & LDA / STM: bag of words, one language at a time; their covariate model is applied to BERTopic instead\\
Who? & co-citation, co-authorship, Lotka, Bradford, \alert{affiliations}; \alert{log-odds} with Dirichlet prior & plain TF-IDF: no shrinkage of rare words\\
Why? & motivations: \alert{validated codebook}, $\kappa = <<MVAL_CB>>$; covariate logit & SBERT, NLI, local LLMs: $\kappa \le <<MVAL_BEST_ALT>>$\\
How? & tone: \alert{FinBERT} on CBDC sentences, $\kappa = <<TONE_KAPPA>>$; stance: levels by reading, changes by a \alert{local LLM} on a stratified sample, $\kappa = <<STANCE_K>>$ & stance: <<ST_ALT_LIST>>\\
When? & vocabulary: \alert{c-TF-IDF} by period; \alert{Kleinberg bursts}; \alert{event study} with placebo dates & Poisson $z$-test: <<EV_P>>\% false positives\\\midrule
All & six robustness specifications; every coder checked on hand-coded texts & \\
\bottomrule\end{tabular}
\note{\textbf{Toolbox (0:45).}
\begin{itemize}
\item One row per question; the middle column is what the results use.
\item The right column matters for the paper: each alternative was run against the same hand-coded texts, and lost.
\item Remaining improvements are tuning (more labels, bigger language models), not new methods.
\end{itemize}}
\end{frame}

\begin{frame}{What: topics found by meaning (SBERT + BERTopic)}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}rp{0.4\textwidth}rrrrrr@{}}\toprule
 & \textbf{Topic} (top c-TF-IDF terms) & \textbf{Texts} & \multicolumn{3}{c}{\textbf{\% of the period's texts}} & \multicolumn{2}{c}{\textbf{Adjusted, points}}\\
 & & & <<PERIOD_HEAD>> & \textbf{change} & \textbf{CB}\\\midrule
<<TOPIC_ROWS>>
\bottomrule\end{tabular}\\[3pt]
<<TOPIC_N>> texts (research, speeches, official), multilingual SBERT; <<TOPIC_K>> topics (<<TOPIC_OUT>>\% outliers reassigned). Labels: $W_{t,c} = \frac{\text{tf}_{t,c}}{|c|}\log\!\big(1+\frac{A}{f_t}\big)$. Adjusted: logit of topic membership on period, actor, English (the covariate model of an STM); change = 2023--26 vs 2014--19; CB = central banks vs academics; \textbf{bold}: 95\% CI excludes 0.\\[2pt]
\small Topics tied to \alert{one jurisdiction}: <<TOPIC_JUR>>.
\note{\textbf{Topics (1:00).}
\begin{itemize}
\item Unlike keywords, embeddings group texts by meaning and across languages: a Russian and an English paper on the digital ruble land together.
\item Two kinds of topics: issues (privacy, cross-border, stablecoins, adoption) and jurisdictions (digital euro, ruble, e-CNY, rupee).
\item Jurisdiction topics keep a stable share, but move away from the digital euro towards the ruble, the e-CNY and the rupee: the literature follows national projects.
\item Central banks dominate one topic, the digital euro; researchers the rest.
\end{itemize}}
\end{frame}


\begin{frame}{Who says what: academics, central bankers, press}
\scriptsize
Distinctive words, \alert{log-odds ratio with informative Dirichlet prior} (Monroe, Colaresi \& Quinn, 2008), group $i$ vs.\ others $j$:
\[
\delta_w = \log\frac{y_{iw}+\alpha_w}{n_i+\alpha_0-y_{iw}-\alpha_w} - \log\frac{y_{jw}+\alpha_w}{n_j+\alpha_0-y_{jw}-\alpha_w},
\qquad z_w = \delta_w \Big/ \sqrt{\tfrac{1}{y_{iw}+\alpha_w}+\tfrac{1}{y_{jw}+\alpha_w}}
\]
\footnotesize
\begin{tabular}{@{}p{0.3\textwidth}p{0.3\textwidth}p{0.3\textwidth}@{}}\toprule
<<DIST_HEAD>>\\\midrule
<<DIST_ROWS>>
\bottomrule\end{tabular}\\[3pt]
\scriptsize English texts; $z$-scores; $\alpha_w = \alpha_0\cdot$ corpus frequency of $w$; topic words removed. Academics: title + abstract; central bankers: speeches; press: headlines.
\note{\textbf{Framing (0:40).}
\begin{itemize}
\item The prior shrinks rare words, so the list is not dominated by noise.
\item <<DIST_TAKE>>.
\end{itemize}}
\end{frame}

\begin{frame}[fragile]{Why study CBDCs? Coding stated motivations}
\begin{columns}[T]
\begin{column}{0.53\textwidth}
\begin{lstlisting}
MOTIVES = {
  "Financial inclusion":
      r"financial\s+inclusion|unbanked|underbanked"
      r"|access\s+to\s+(?:financial\s+services|finance)",
  "Monetary sovereignty":
      r"monetary\s+sovereignty|global\s+stablecoin"
      r"|\blibra\b|big\s*tech|dollari[sz]ation",
  "Decline of cash":
      r"declin\w*\s+(?:in|of)\s+(?:the\s+)?cash"
      r"|cashless|use\s+of\s+cash",
  ...
}
\end{lstlisting}
\end{column}
\begin{column}{0.44\textwidth}\small
\begin{itemize}
\item Motivations are stated in the \alert{introduction} (else abstract, else opening of the text).
\item 10 motivations, codebook built by reading introductions.
\item For text $t$, motivation $k$, group $G$:
\begin{align*} m_{k}(t) &= \mathbf{1}[\text{regex}_k \text{ matches } t]\\ \text{share}_k(G) &= \tfrac{1}{|G|}\textstyle\sum_{t\in G} m_k(t) \end{align*}
\item <<MOTIVE_N>> texts coded; <<MOTIVE_ANY>>\% state at least one motivation.
\item <<VAL_LINE>>
\end{itemize}
\end{column}
\end{columns}
\note{\textbf{Motivations: method (0:35).}
\begin{itemize}
\item The ``why'' requires reading, so the reading is done once to build the codebook, then applied to every text.
\item A text can have several motivations. Press is excluded: headlines are too short.
\item Validation: codebook tuned on 100 texts, then tested on 50 others never used for tuning. High precision, modest recall: what is found is right, but about half is missed (non-English texts, paraphrases). Details in the backup slide.
\end{itemize}}
\end{frame}

\begin{frame}{Why study CBDCs? Results}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}p{0.29\textwidth}rrrrr@{}}\toprule
 & \multicolumn{3}{c}{\textbf{\% of texts}} & \multicolumn{2}{c}{\textbf{Adjusted effect, points [95\% CI]}}\\
\textbf{Motivation} & <<MOTIVE_HEAD>>\\
 & & & & \textbf{2023--26 vs 2014--19} & \textbf{vs academics}\\\midrule
<<MOTIVE_ROWS>>
\bottomrule\end{tabular}\\[3pt]
Logit of each motivation on period, actor and English; average marginal effects, robust SEs. \textbf{Bold}: CI excludes 0.
\begin{itemize}\setlength\itemsep{0pt}
\item \alert{Rising:} <<MOTIVE_UP>>. \quad \alert{Falling:} <<MOTIVE_DOWN>>.
\item Actors differ: <<MOTIVE_GAP>>. \quad <<VAL_CAVEAT>>
\end{itemize}
\note{\textbf{Motivations: results (1:00).}
\begin{itemize}
\item The adjusted columns compare like with like: same mix of actors and languages in each period.
\item Early literature: monetary policy, how a CBDC changes rates and bank funding. Recent literature: inclusion, cross-border payments, geopolitics, privacy.
\item Central bankers and academics do not give the same reasons.
\end{itemize}}
\end{frame}

\begin{frame}{Why: is the codebook the best coder?}
\begin{columns}[T]
\begin{column}{0.58\textwidth}\scriptsize
\begin{tabular}{@{}lrrrr@{}}\toprule
\textbf{Coder} & \textbf{P} & \textbf{R} & $F_1$ & $\kappa$\\\midrule
<<MVAL_ROWS>>
\bottomrule\end{tabular}\\[3pt]
50 held-out texts $\times$ 10 motivations; thresholds and classifiers set on the 100 development texts. NLI: multilingual mDeBERTa; LLMs: Qwen2.5 Instruct, 4-bit, run locally (\texttt{scripts/validate\_coders.py}).
\end{column}
\begin{column}{0.4\textwidth}\small
\begin{itemize}\setlength\itemsep{2pt}
\item \alert{The codebook wins}: $\kappa = <<MVAL_CB>>$; best alternative <<MVAL_BEST_ALT>>.
\item Models confuse ``mentions privacy'' with ``studies CBDCs because of privacy''; LLMs over-tag.
\item Trends hold: <<MOTIVE_ROBUST>>
\end{itemize}
\end{column}
\end{columns}
\note{\textbf{Measuring motivations (0:40).}
\begin{itemize}
\item Five machine coders against the same hand-coded texts: SBERT in two ways, an NLI model, two local LLMs.
\item None beats the codebook; the codebook's weakness is recall, so shares are lower bounds.
\item A larger LLM behind an API could do better; it would be checked the same way. That is tuning, not a new method.
\end{itemize}}
\end{frame}

\begin{frame}{How: the tone of what is said about CBDCs}
\begin{columns}[T]
\begin{column}{0.46\textwidth}
\begin{tikzpicture}
\begin{axis}[width=\textwidth, height=0.62\textheight, xtick={<<TONE_XTICKS>>}, xticklabel style={/pgf/number format/1000 sep=, font=\tiny, rotate=45, anchor=north east, inner sep=1pt},
  yticklabel style={font=\scriptsize, /pgf/number format/fixed}, ylabel={net tone}, ylabel style={font=\scriptsize},
  legend style={at={(0.5,-0.18)}, anchor=north, font=\tiny, draw=none, legend columns=3}, axis lines*=left, ymajorgrids]
<<TONE_PLOTS>>
\end{axis}
\end{tikzpicture}
\end{column}
\begin{column}{0.52\textwidth}\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrr@{}}\toprule
 & \textbf{Texts} & \textbf{Sent.} & \textbf{\%+} & \textbf{\%--} & \textbf{Net tone}\\\midrule
<<TONE_ROWS>>
\bottomrule\end{tabular}\\[4pt]
FinBERT on the <<N_SENT>> English sentences that name a CBDC (central banks: speeches and official documents). Sentence tone $= P(+) - P(-)$, averaged per text, then per group ($\pm$ 95\% CI).\\[3pt]
<<TONE_VAL>>
\end{column}
\end{columns}
\note{\textbf{Tone (0:50).}
\begin{itemize}
\item Only sentences that name a CBDC are scored, so this is the tone about CBDCs, not about the economy in general.
\item Central banks are the most positive, the press the least.
\item Academic tone rises (<<TONE_ACAD>>), in line with the move from risks (stability) to benefits (inclusion).
\item Validation: FinBERT rarely confuses positive with negative, but sees tone in neutral sentences: compare groups and years, do not read levels.
\end{itemize}}
\end{frame}

\begin{frame}{How: stance towards CBDCs}
\begin{columns}[T]
\begin{column}{0.44\textwidth}\scriptsize
\textbf{Level} (read: 50 texts per actor), \% [95\% CI]\\[2pt]
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrr@{}}\toprule
 & $n$ & \textbf{Favourable} & \textbf{Critical}\\\midrule
<<STANCE_LEVEL>>
\bottomrule\end{tabular}\\[6pt]
\small <<STANCE_TAKE>>.
\end{column}
\begin{column}{0.54\textwidth}\scriptsize
\textbf{Change} (local LLM, <<STANCE_N>> texts sampled per actor $\times$ period), \% favourable\\[2pt]
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrr@{}}\toprule
 & <<PERIOD_HEAD>> & \textbf{Change, pts}\\\midrule
<<STANCE_TREND>>
\bottomrule\end{tabular}\\[4pt]
The LLM (Qwen2.5 7B, $\kappa = <<STANCE_K>>$, best of five coders) calls <<STANCE_RAWFAV>>\% of the hand-coded texts favourable against <<STANCE_TRUEFAV>>\% by reading: its levels are inflated, so only its \emph{changes} are used (assuming a stable bias). \textbf{Bold}: CI excludes 0.
\end{column}
\end{columns}
\note{\textbf{Stance (0:50).}
\begin{itemize}
\item Tone says how positive the words are; stance says whether the author is for or against CBDCs.
\item Levels come from reading: central banks are the advocates, researchers mostly neutral, criticism is rare and mostly in the press.
\item Changes come from the language model, run on a random sample: central banks have become much more favourable since 2020, as their own projects advanced.
\end{itemize}}
\end{frame}

\begin{frame}{When: which themes burst, and when}
\hspace*{3.3cm}<<BURSTS>>\\[2pt]
\scriptsize Kleinberg (2002) burst detection on keywords, by quarter: keyword $w$ in $r_t$ of $d_t$ papers; normal rate $p_0 = R/D$, burst rate $p_1 = 2p_0$; cost $\sigma_i(t) = -\ln\big[\binom{d_t}{r_t} p_i^{r_t}(1-p_i)^{d_t-r_t}\big]$, entering a burst costs $\ln T$; the cheapest state path (Viterbi) gives the bursts. Darker = stronger.
\note{\textbf{Bursts (0:45).}
\begin{itemize}
\item Where the event study tests dates I chose, burst detection lets the data say when a theme suddenly takes off.
\item Early bursts: <<BURST_EARLY>>. Latest bursts: <<BURST_LATE>>.
\item Read with the topics: the field moves from monetary policy to sovereignty, tokenisation and national projects.
\end{itemize}}
\end{frame}

\begin{frame}{When? Attention around policy events}
\begin{columns}[c]
\begin{column}{0.5\textwidth}
\begin{tikzpicture}
\begin{axis}[width=\textwidth, height=0.62\textheight, ymin=0, ymax=\ymaxev, xmin=2019, xmax=<<THIS_YEAR>>.85,
  xticklabel style={/pgf/number format/1000 sep=, font=\scriptsize}, xtick={2019,...,<<THIS_YEAR>>},
  yticklabel style={font=\scriptsize}, ylabel={texts per week}, ylabel style={font=\scriptsize},
  axis lines*=left, clip=false]
\addplot[c1, thin, const plot] coordinates {<<WEEKLY>>};
<<EV_MARKS>>
\end{axis}
\end{tikzpicture}
\end{column}
\begin{column}{0.48\textwidth}\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}rlp{0.46\linewidth}rrr@{}}\toprule
 & \textbf{Date} & \textbf{Event} & \textbf{Ratio} & $z$ & $p$\\\midrule
<<EV_ROWS>>
\bottomrule\end{tabular}\\[3pt]
Baseline $\lambda$ = daily rate over $[-90,-15]$ days; $E = 15\lambda$; ratio $= N_{[0,14]}/E$; $z = (N-E)/\sqrt{E}$.
\alert{Placebo}: on <<EV_NPL>> random dates, <<EV_P>>\% have $z > 1.96$ (overdispersion, trend), so $p$ = share of placebo dates with a larger $z$ (95th percentile: $z = <<EV_Z95>>$).
\end{column}
\end{columns}
\note{\textbf{Event study (0:50).}
\begin{itemize}
\item For each event, compare the texts in the two weeks after it with the rate of the three months before.
\item The naive Poisson test is far too lenient: a quarter of random dates would look significant. So each event is compared with the placebo dates instead.
\item Red lines: $p < 0.05$ against the placebo distribution.
\item <<EV_TAKE>>
\item Attention follows institutional milestones (central bank reports, legislation) more than market news. Sand Dollar comes 18 days after the ECB report: the two windows overlap.
\end{itemize}}
\end{frame}

\begin{frame}{Robustness}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrcccrr@{}}\toprule
 & & \multicolumn{2}{c}{\textbf{Why: change, points}} & \textbf{How: tone} & \textbf{How: stance} & \textbf{When}\\
\textbf{Specification} & \textbf{Texts} & \textbf{inclusion} & \textbf{monetary policy} & \textbf{CB -- press} & \textbf{fav.\ CB -- acad.} & \textbf{events}\\\midrule
<<ROB_ROWS>>
\bottomrule\end{tabular}\\[4pt]
Changes: 2023--26 vs 2014--19, adjusted (logit, 95\% CI). Tone: mean net tone gap. Stance: raw LLM labels, points. Events: $p < 0.05$ against placebo dates.\\[3pt]
\small Signs and orders of magnitude hold in every specification.
\note{\textbf{Robustness (0:50).}
\begin{itemize}
\item Each row re-runs the headline results on a different sample: only texts clearly about CBDCs, only English, only journal articles, without the partial year, or with the high-recall coding.
\item The direction of every result is unchanged; magnitudes move a little. This is the table that answers a referee's ``what if''.
\end{itemize}}
\end{frame}

% =====================================================================
\section{Conclusion}

\begin{frame}{Takeaways and limits}
\begin{columns}[T]
\begin{column}{0.54\textwidth}\footnotesize
\textbf{First results}
\begin{itemize}\setlength\itemsep{1pt}
\item \alert{What:} <<N>> texts; output grows <<CAGR>>\% a year; topics follow national projects.
\item \alert{Who:} a core of co-cited papers (<<COCIT_CORE>>); central bank economists write <<CB_SHARE>>\% of papers but get <<CB_CITE>>\% of citations; <<TOP_BANK>> the most vocal central bank.
\item \alert{Why:} rising <<MOTIVE_UP>>; falling <<MOTIVE_DOWN>>.
\item \alert{How:} central banks most positive in tone and stance, and increasingly so; criticism is rare, mostly in the press.
\item \alert{When:} bursts move from monetary policy to sovereignty and tokenisation; <<EV_NSIG>> of <<EV_NTOT>> milestones move attention.
\end{itemize}
\end{column}
\begin{column}{0.43\textwidth}\footnotesize
\textbf{Limits}
\begin{itemize}\setlength\itemsep{1pt}
\item Press: headlines only; coverage favours recent years; harvest cut short by rate limits.
\item ``Secondary theme'' is noisy: results re-run on main theme only.
\item Motivations: recall $\approx$ 0.5 (lower bounds). Tone: English only. Stance: LLM for changes only.
\item One coder (AI-assisted) for all validation samples.
\item Keywords partly machine-assigned; authors matched on spelling.
\end{itemize}
\end{column}
\end{columns}
\note{\textbf{Takeaways (0:50).}
\begin{itemize}
\item Recap the five answers: what, who, why, how, when.
\item Main message: CBDC research moved from a monetary-policy question to one of inclusion, cross-border payments and national projects; the tone has become more positive; attention follows central bank milestones.
\item Limits are known and measured: each one has a fix in the paper plan.
\end{itemize}}
\end{frame}

\begin{frame}{From this talk to the paper}
\begin{columns}[T]
\begin{column}{0.5\textwidth}\scriptsize
\textbf{Structure}\\[2pt]
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lll@{}}\toprule
 & \textbf{Section} & \textbf{Status}\\\midrule
1 & Introduction, research question & to write\\
2 & Related work (<<N_REVIEWS>> reviews) & material\\
3 & Data: corpus, relevance filter & validated\\
4 & Methods: maps, topics, coders, tone, stance, events & done\\
5 & Results: what, who, why, how, when & done\\
6 & Robustness (6 specifications) & done\\
7 & Discussion, limits & to write\\
\bottomrule\end{tabular}\\[6pt]
\textbf{Working title.} \emph{Why central bank digital currencies? Researchers, central bankers and the press, 2014--<<Y1>>.}
\end{column}
\begin{column}{0.47\textwidth}\scriptsize
\textbf{Before submission}
\begin{itemize}\setlength\itemsep{1pt}
\item Finish the press harvest and full-text downloads.
\item Second human coder: the 200-item sheet is ready (\texttt{second\_coder\_sheet.csv}); report inter-coder $\kappa$.
\item Release the dataset and code with a DOI.
\item Methods are complete; what is left is data collection and a second reader, not new tools.
\end{itemize}
\textbf{Candidate outlets}: \emph{Scientometrics}; \emph{Technological Forecasting and Social Change}; \emph{Research in International Business and Finance} (published the closest review, Hoang et al.\ 2023).
\end{column}
\end{columns}
\note{\textbf{Paper plan (1:00).}
\begin{itemize}
\item Data, methods, results and robustness are done; the method for each question was chosen against the alternatives.
\item What is left: finishing the press harvest, a second human reader, and writing.
\item Thank you; questions.
\end{itemize}}
\end{frame}

\appendix
\begin{frame}{Backup: validation of the motivation coding}
\footnotesize
Codebook tuned on 100 texts (development set), then evaluated on 50 \alert{held-out} texts; unit = first 1{,}500 characters of the introduction / abstract.
\[ \text{precision} = \frac{TP}{TP+FP}, \quad \text{recall} = \frac{TP}{TP+FN}, \quad \kappa = \frac{p_o - p_e}{1 - p_e} \]
\begin{tabular}{@{}lrrrrr@{}}\toprule
\textbf{Motivation} & \textbf{Read} & \textbf{Coded} & \textbf{Precision} & \textbf{Recall} & $\kappa$\\\midrule
<<VAL_ROWS>>
\midrule
<<VAL_ALL>>
\bottomrule\end{tabular}\\[3pt]
\scriptsize Read = texts where the reading found the motivation; Coded = texts flagged by the codebook. Files: \texttt{presentation/validation\_motivations\_dev.csv}, \texttt{\_test.csv}.
\note{\textbf{Backup (only if asked).} The reference coding was done by reading each text against the codebook definitions (AI-assisted, spot-checked). Payment efficiency is the weakest category: it is expressed in too many ways for keywords.}
\end{frame}
\begin{frame}{Backup: how the science maps are built}
\begin{itemize}
\item \textbf{Networks.} Nodes = references, authors or keywords; $c_{ij}$ = documents where $i$ and $j$ appear together, $c_i$ = occurrences of $i$. Edges weighted by \alert{association strength}:
\[ s_{ij} = \frac{c_{ij}}{c_i\,c_j} \]
\item \textbf{Clusters.} Louvain algorithm, maximising modularity
\[ Q = \frac{1}{2m}\sum_{i,j}\Big[s_{ij} - \frac{k_i k_j}{2m}\Big]\,\delta(g_i,g_j) \]
\item \textbf{Layout.} Force-directed: linked nodes end up close. Size $\propto \sqrt{c_i}$, colour = cluster.
\end{itemize}
\note{\textbf{Backup (only if asked).}
\begin{itemize}
\item Association strength corrects for popularity: two very frequent items co-occur often by chance.
\item Louvain groups nodes so that links are dense inside groups and sparse between them; $Q$ measures how clear the grouping is.
\end{itemize}}
\end{frame}

\begin{frame}{Backup: how the score behaves}
\footnotesize
\begin{tabular}{@{}lrl@{}}\toprule
\textbf{Example} & \textbf{Score} & \textbf{Label}\\\midrule
<<EXAMPLE_ROWS>>
\bottomrule\end{tabular}
\note{\textbf{Backup (only if asked).} A speech with one paragraph on the digital euro is ``mentioned only''; a medical case report on CBDC (the skin disease) scores zero.}
\end{frame}

\begin{frame}[fragile]{Backup: academic sources, cursor pagination under a rate limit}
\begin{columns}[T]
\begin{column}{0.55\textwidth}
\begin{lstlisting}
def iter_phrase(phrase):
    cursor = "*"
    while cursor:
        d = get(f'{API}?filter=title_and_abstract'
                f'.search:"{phrase}"&per-page=200'
                f'&cursor={cursor}')
        yield from d["results"]
        cursor = d["meta"].get("next_cursor")

class Throttle:
    def wait(self):
        dt = self.interval - (now() - self.last)
        if dt > 0: sleep(dt)
        self.last = now()
\end{lstlisting}
\end{column}
\begin{column}{0.41\textwidth}\small
\begin{itemize}
\item Each phrase searched in titles and abstracts; results merged on the source id.
\item Requests spaced by $t_{k+1}-t_k \geq \Delta$\\ ($\Delta = 0.12$\,s OpenAlex, $1.2$\,s Google News, $10$\,s GDELT).
\item On HTTP 429: back-off $w_j = \min(60j,\,900)$\,s after the $j$-th consecutive failure.
\item Reference lists resolved into citations (for co-citation).
\end{itemize}
\end{column}
\end{columns}
\note{\textbf{Backup (only if asked).}
\begin{itemize}
\item APIs return results page by page; a cursor points to the next page until none is left.
\item Free APIs limit the request rate: a throttle guarantees a minimum gap $\Delta$ between calls, and the program waits longer after each refusal.
\end{itemize}}
\end{frame}

\begin{frame}[fragile]{Backup: full text and de-duplication}
\begin{columns}[T]
\begin{column}{0.46\textwidth}\small
\textbf{Full text}
\begin{itemize}
\item PDFs (PyMuPDF) and web pages (trafilatura).
\item Split into introduction / conclusion / section headings.
\item Page metadata (date, author, outlet) fills gaps.
\end{itemize}
\end{column}
\begin{column}{0.51\textwidth}
\begin{lstlisting}
def key(r):
    if r.source_type == "academic":
        if r.doi: return "doi:" + r.doi.lower()
        return f"t:{norm(r.title)[:120]}:{r.year}"
    if r.source_type == "speech":
        return f"s:{r.date}:{norm(r.title)[:50]}"
    return "u:" + r.url
\end{lstlisting}
\small
\begin{itemize}
\item Second pass: same title + year across DOIs.
\item Merge: most precise date, longest abstract, union of keywords.
\end{itemize}
\end{column}
\end{columns}
\note{\textbf{Backup (only if asked).}
\begin{itemize}
\item The same paper appears in several databases: we match on DOI, else on normalised title and year.
\item When merging, keep the best of each copy: the most precise date, the longest abstract.
\end{itemize}}
\end{frame}

\begin{frame}{Backup: validation of the tone measure}
\footnotesize
30 sentences sampled per FinBERT label, read blind to the label.\\[4pt]
\begin{tabular}{@{}lrrrr@{}}\toprule
\textbf{FinBERT $\downarrow$ / reading $\rightarrow$} & \textbf{negative} & \textbf{neutral} & \textbf{positive} & \textbf{Precision}\\\midrule
<<TONE_XTAB>>
\bottomrule\end{tabular}\\[4pt]
<<TONE_VAL>>\\[3pt]
\scriptsize File: \texttt{presentation/validation\_tone.csv}.
\note{\textbf{Backup (only if asked).} Errors are almost all ``toned vs.\ neutral'': FinBERT was trained on financial news, where descriptive sentences about a new product read as positive.}
\end{frame}

\begin{frame}{References}
\scriptsize
\begin{itemize}\setlength\itemsep{1pt}
\item Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. arXiv:1908.10063.
\item Aria, M., \& Cuccurullo, C. (2017). bibliometrix: An R-tool for comprehensive science mapping analysis. \emph{Journal of Informetrics}, 11(4).
\item Blondel, V. D., Guillaume, J.-L., Lambiotte, R., \& Lefebvre, E. (2008). Fast unfolding of communities in large networks. \emph{J.\ Stat.\ Mech.}
\item Callon, M., Courtial, J.-P., \& Laville, F. (1991). Co-word analysis as a tool for describing the network of interactions between basic and technological research. \emph{Scientometrics}, 22(1).
\item Cobo, M. J., López-Herrera, A. G., Herrera-Viedma, E., \& Herrera, F. (2011). An approach for detecting, quantifying, and visualizing the evolution of a research field. \emph{Journal of Informetrics}, 5(1).
\item Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
\item Kleinberg, J. (2002). Bursty and hierarchical structure in streams. \emph{KDD}.
\item Monroe, B. L., Colaresi, M. P., \& Quinn, K. M. (2008). Fightin' words. \emph{Political Analysis}, 16(4).
\item Reimers, N., \& Gurevych, I. (2019). Sentence-BERT. \emph{EMNLP}.
\item Roberts, M. E., Stewart, B. M., Tingley, D., et al.\ (2014). Structural topic models for open-ended survey responses. \emph{AJPS}, 58(4).
\item van Eck, N. J., \& Waltman, L. (2010). Software survey: VOSviewer. \emph{Scientometrics}, 84(2).
\item Reviews cited: Bapat \& Sharma (2025); Fahad \& Bulut (2024); Hoang, Ngo \& Vu (2023); Kvedaravi\v{c}i\={u}t\.{e} \& \v{S}apkauskien\.{e} (2024); Paulovici (2026); Prodan et al.\ (2024): full records in the corpus.
\end{itemize}
\note{\textbf{References.}}
\end{frame}
\end{document}
"""

if __name__ == "__main__":
    main()
