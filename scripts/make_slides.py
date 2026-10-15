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
        gap = td[td.actor == "speech"].tone.mean() - td[td.actor == "academic"].tone.mean()
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
    d_theme = full[full.relevance.isin(RELEVANT)].copy()
    fields = pd.read_csv(ROOT / "data/cache/openalex_fields.csv")  # written by scripts/tag_economists.py
    d = d_theme[A.economists(d_theme, fields)].copy()  # the corpus of the talk: texts written by economists
    n = len(d)
    by_type = d.source_type.value_counts()
    types = [(k, lbl) for k, lbl in TYPES if by_type.get(k, 0)]
    today = date.today()
    periods = [p for p, _, _ in A.PERIODS]
    v = {"N_THEME": f"{len(d_theme):,}", "TIMES": f"{len(d_theme) / 1024:.0f}", "DATE": f"{today:%B %Y}", "N_ALL": f"{n_all:,}", "N": f"{n:,}", "N_LANG": str(d.language.nunique()),
         "Y0": str(int(d.year.min())), "Y1": str(int(d.year.max())), "N_COLS": str(len(full.columns) + 1)}

    # --- who counts as an economist (step 4 of the filter)
    ff = d_theme[["record_id"]].merge(fields, on="record_id", how="left").set_index(d_theme.index)
    ac = d_theme.source_type == "academic"
    by_field = ff.field.eq(A.ECON_FIELD)
    by_cb = ff.institutions.fillna("").str.contains(A.CB_AFFIL) | d_theme.author_affiliations.fillna("").str.contains(A.CB_AFFIL)
    by_venue = ff.field.isna() & d_theme.venue.fillna("").str.contains(A.ECON_VENUE)
    off = d_theme.source_type == "official"
    v["ECON_ROWS"] = "\n".join(f"{a} & {b:,} & {c} \\\\" for a, b, c in [
        ("Research, classified in economics or finance (OpenAlex)", int((ac & by_field).sum()), "kept"),
        ("Research, other field but an author at a central bank, IMF or BIS", int((ac & ~by_field & by_cb).sum()), "kept"),
        ("Research without OpenAlex record, economics venue", int((ac & ~by_field & ~by_cb & by_venue).sum()), "kept"),
        ("Research in other fields", int((ac & ~(by_field | by_cb | by_venue)).sum()), "dropped"),
        ("Central bank speeches", int((d_theme.source_type == "speech").sum()), "kept"),
        ("Documents of central banks and international institutions",
         int((off & d_theme.issuer_type.isin(["central bank", "international organisation"])).sum()), "kept"),
        ("Documents of governments and parliaments", int((off & ~d_theme.issuer_type.isin(["central bank", "international organisation"])).sum()), "dropped"),
        ("Press articles (no author)", int((d_theme.source_type == "press").sum()), "dropped")])
    drop_f = ff[ac & ~(by_field | by_cb | by_venue)].field.fillna("no OpenAlex record").value_counts()
    v["ECON_DROPPED"] = ", ".join(f"{tex(k)} {c:,}" for k, c in drop_f.head(4).items())

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
    inst = d[["record_id"]].merge(fields, on="record_id").institutions.dropna().str.split("; ").explode()
    inst_counts = inst[inst != ""].value_counts()
    v["VOICES"] = (r"\begin{columns}[T]" + "".join(
        r"\begin{column}{0.32\textwidth}\scriptsize\textbf{" + t + r"}\\[2pt]"
        + r"\begin{tabular}{@{}p{0.8\linewidth}r@{}}" + rows(s, 8, 30) + r"\end{tabular}\end{column}"
        for t, s in [("Speeches, by central bank", d[d.source_type == "speech"].issuer.value_counts()),
                     ("Documents, by institution", d[d.source_type == "official"].issuer.value_counts()),
                     ("Research, by author institution", inst_counts)]) + r"\end{columns}")
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
    # codebook on the held-out texts written by economists (the corpus of the talk); "all" = the whole test set
    test_all = pd.read_csv(OUT / "validation_motivations_test.csv")
    test_e = test_all[test_all.record_id.isin(d.record_id)].reset_index(drop=True)
    cbk = nlp._codebook(test_e.text)
    ym = {k: test_e[f"manual: {k}"].values.astype(bool) for k in A.MOTIVES}
    per = {k: {**nlp._prf(ym[k], cbk[k]), "manual": int(ym[k].sum()), "auto": int(cbk[k].sum())} for k in A.MOTIVES}
    a = {**nlp._prf(np.concatenate(list(ym.values())), np.concatenate([cbk[k] for k in A.MOTIVES])),
         "manual": sum(r["manual"] for r in per.values()), "auto": sum(r["auto"] for r in per.values())}
    num = lambda x: "--" if x != x else f"{x:.2f}"
    v["VAL_N"], v["VAL_N_ALL"] = str(len(test_e)), str(len(test_all))
    v["VAL_LINE"] = (f"Validated on the {len(test_e)} held-out texts by economists (of {len(test_all)}; reading-based "
                     f"coding, AI-assisted): precision {num(a['precision'])}, recall {num(a['recall'])}, $\\kappa = {num(a['kappa'])}$.")
    v["VAL_CAVEAT"] = (f"Recall $\\approx {num(a['recall'])}$: shares are \\alert{{lower bounds}}; "
                       "compare motivations and periods rather than read levels.")
    v["VAL_ROWS"] = "\n".join(
        f"{tex(k)} & {r['manual']} & {r['auto']} & {num(r['precision'])} & {num(r['recall'])} & {num(r['kappa'])} \\\\"
        for k, r in sorted(per.items(), key=lambda kv: -(kv[1]["kappa"] if kv[1]["kappa"] == kv[1]["kappa"] else -9)))
    v["VAL_ALL"] = (f"\\textbf{{All}} & {a['manual']} & {a['auto']} & {num(a['precision'])} & {num(a['recall'])} & "
                    f"{num(a['kappa'])} \\\\")

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
    rv = d_theme[(d_theme.source_type == "academic")
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
    tp = nlp.topics(d, ROOT, ROOT / "data/cache/embeddings.npz", min_size=40, selection="leaf")
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
    vc = pd.read_csv(OUT / "validation_coders.csv")  # written by scripts/validate_coders.py
    by_scope = lambda sc, task: vc[(vc.scope == sc) & (vc.task == task)].set_index("coder")
    vmot, vst = by_scope("economists", "motivations"), by_scope("economists", "stance")
    vmot_all, vst_all = by_scope("all", "motivations"), by_scope("all", "stance")
    v["MVAL_N"], v["MVAL_N_ALL"] = str(int(vmot.n_texts.iloc[0])), str(int(vmot_all.n_texts.iloc[0]))
    v["ST_N"], v["ST_N_ALL"] = str(int(vst.n_texts.iloc[0])), str(int(vst_all.n_texts.iloc[0]))
    v["MVAL_ROWS"] = "\n".join(("\\rowcolor{c4!12}" if k.startswith("Codebook (") else "")
                               + f"{k} & {r.precision:.2f} & {r.recall:.2f} & {r.f1:.2f} & {r.kappa:.2f} & "
                               + f"{vmot_all.loc[k, 'kappa']:.2f} \\\\"
                               for k, r in vmot.sort_values("kappa", ascending=False).iterrows())
    v["ST_ROWS"] = "\n".join(("\\rowcolor{c4!12}" if k.startswith("Local LLM") else "")
                             + f"{k} & {r.accuracy:.2f} & {r.kappa:.2f} & {vst_all.loc[k, 'kappa']:.2f} \\\\"
                             for k, r in vst.sort_values("kappa", ascending=False).iterrows())
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
    actors = [("academic", "Research economists", "c1"), ("speech", "Central banks \\& institutions", "c2")]
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
    tv_all = pd.read_csv(OUT / "validation_tone.csv")
    tv = tv_all[tv_all.record_id.isin(d.record_id)]  # sentences from economists' texts
    labs = ["negative", "neutral", "positive"]
    kap = lambda t: (lambda po, pe: (po - pe) / (1 - pe))((t.label == t.manual).mean(),
                                                         sum((t.label == l).mean() * (t.manual == l).mean() for l in labs))
    po = (tv.label == tv.manual).mean()
    pe = sum((tv.label == l).mean() * (tv.manual == l).mean() for l in labs)
    v["TONE_N"], v["TONE_KAPPA_ALL"] = str(len(tv)), f"{kap(tv_all):.2f}"
    v["TONE_VAL"] = (f"{len(tv)} sentences from economists' texts read blind (of {len(tv_all)}): agreement {100 * po:.0f}\\%, "
                     f"$\\kappa = {(po - pe) / (1 - pe):.2f}$; positive and negative confused "
                     f"{int(((tv.label == 'positive') & (tv.manual == 'negative')).sum() + ((tv.label == 'negative') & (tv.manual == 'positive')).sum())} times.")
    v["TONE_XTAB"] = "\n".join(f"{l} & " + " & ".join(str(int(((tv.label == l) & (tv.manual == m)).sum())) for m in labs)
                               + f" & {((tv.label == l) & (tv.manual == l)).sum() / max((tv.label == l).sum(), 1):.2f} \\\\" for l in labs)
    v["N_SENT"] = f"{len(snt):,}"
    v["TONE_KAPPA"] = f"{(po - pe) / (1 - pe):.2f}"
    ac = doc[(doc.actor == "academic")]
    v["TONE_ACAD"] = (f"{ac[ac.year == tyrs[0]].tone.mean():.2f} in {tyrs[0]} to "
                      f"{ac[ac.year == today.year - 1].tone.mean():.2f} in {today.year - 1}")

    # --- do events move tone? 180 days after vs before each event, against random dates
    dpr = d.date_precision.fillna("")
    days = d[dpr.str.startswith("day") & ~dpr.str.contains("Jan 1")].set_index("record_id").publication_date
    doc["day"] = pd.to_datetime(doc.record_id.map(days), errors="coerce")
    te, te95 = A.tone_events(doc)
    v["TEV_ROWS"] = "\n".join(
        f"{pd.Timestamp(r.date):%b %Y} & {tex(r.event)} & {r.before} & {r.after} & "
        + (f"\\textbf{{{r.delta:+.3f}}}" if r.p < 0.05 else f"{r.delta:+.3f}") + f" & {r.p:.2f} \\\\"
        for r in te.itertuples())
    v["TEV_NSIG"], v["TEV_95"] = str(int((te.p < 0.05).sum())), f"{te95:.3f}"
    v["TEV_MIN_P"] = f"{te.p.min():.2f}"

    # --- how (2): stance. Levels from the hand-coded texts (50 per actor); changes over time from a local LLM
    # on a stratified sample (its bias towards "favourable" inflates levels but largely cancels in differences)
    ss = nlp.stance_sample(d, ROOT, ROOT / "data/cache/stance_sample.csv")  # cached: codes only what is missing
    sv = pd.read_csv(OUT / "validation_stance.csv").merge(pd.read_csv(OUT / "validation_stance_llm.csv"), on="record_id")
    sv = sv[sv.record_id.isin(d.record_id)]
    sv["actor"] = sv.source_type.replace({"official": "speech"})
    vst_k = vst.loc["Local LLM, Qwen2.5 7B", "kappa"]
    level, trend = [], []
    for a, lbl in [("academic", "Research economists"), ("speech", "Central banks \\& institutions")]:
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
    v["STANCE_TAKE"] = (f"By reading: {100 * hc.loc['speech', 'favourable']:.0f}\\% of central bank texts are favourable; "
                        f"research economists are mostly \\alert{{neutral}} ({100 * hc.loc['academic', 'neutral']:.0f}\\%): "
                        "they analyse, central banks advocate")
    st_fav = {a: [100 * (ss[(ss.actor == a) & (ss.period == p_)].stance == "favourable").mean() for p_ in periods]
              for a in ("academic", "speech")}

    # --- the 2022 sanctions: difference-in-differences on texts about Russia (or in Russian) vs the others
    mtab = mo["table"].merge(d[["record_id", "language", "keywords_indexed"]], on="record_id")
    treated = (mtab.language == "ru") | (mtab.title.fillna("") + " " + mtab.text + " "
                                         + mtab.keywords_indexed.fillna("")).str.contains(A.RUSSIA)
    dd_ = {k: A.did(mtab, treated, k, years=(2020, today.year)) for k in A.MOTIVES}
    geo = dd_["Geopolitics and sanctions"]
    v["DID_PLOT"] = " ".join(f"({t},{g:.1f}) +- (0,{(hi - lo) / 2:.1f})" for t, g, lo, hi in geo["by_year"])
    v["DID_POOLED"] = f"${geo['pooled'][0]:+.1f}$ points [{geo['pooled'][1]:+.0f}, {geo['pooled'][2]:+.0f}]"
    v["DID_NT"], v["DID_N"] = f"{geo['n_treated']:,}", f"{geo['n']:,}"
    v["DID_ROWS"] = "\n".join(
        f"{tex(k, 34)} & " + (f"\\textbf{{{e['pooled'][0]:+.1f}}}" if e["pooled"][1] > 0 or e["pooled"][2] < 0
                              else f"{e['pooled'][0]:+.1f}") + f" \\tiny[{e['pooled'][1]:+.0f}, {e['pooled'][2]:+.0f}] \\\\"
        for k, e in sorted(dd_.items(), key=lambda kv: -kv[1]["pooled"][0]))
    v["DID_OTHERS"] = str(sum(1 for k, e in dd_.items() if k != "Geopolitics and sanctions"
                              and (e["pooled"][1] > 0 or e["pooled"][2] < 0)))

    # --- synthesis: the discussion and its environment, period by period
    def in_period(y, p_):
        lo_y, hi_y = next((a_, b_) for l_, a_, b_ in A.PERIODS if l_ == p_)
        return lo_y <= y <= hi_y
    ev_sig = set(ev[ev.p < 0.05].event)
    syn = {
        "Policy events (bold: attention reacts)": [
            "; ".join((f"\\textbf{{{tex(e)}}}" if e in ev_sig else tex(e)) for dt_, e in A.EVENTS if in_period(int(dt_[:4]), p_))
            or "--" for p_ in periods],
        "Texts per year": [f"{d[d.year.between(*next((a_, min(b_, today.year)) for l_, a_, b_ in A.PERIODS if l_ == p_))].year.value_counts().mean():,.0f}"
                           for p_ in periods],
        "Bursting keywords": [", ".join(tex(k) for k in bu[bu.start.map(lambda q: in_period(int(q[:4]), p_))].keyword.head(3)) or "--"
                              for p_ in periods],
        "Most frequent motivations": [", ".join(tex(k.split(" (")[0].lower()) for k in mo["by_period"][p_].sort_values(ascending=False).index[:3])
                                      for p_ in periods],
        "Net tone (research / central banks)": [
            " / ".join(f"{doc[(doc.actor == a) & doc.year.map(lambda y: in_period(y, p_))].tone.mean():.2f}" for a in ("academic", "speech"))
            for p_ in periods],
        "Favourable, LLM (research / central banks)": [f"{st_fav['academic'][i]:.0f}\\% / {st_fav['speech'][i]:.0f}\\%"
                                                       for i in range(len(periods))],
    }
    v["SYN_ROWS"] = "\n".join(f"{k} & " + " & ".join(vals) + " \\\\" for k, vals in syn.items())

    # --- robustness: the headline results on alternative samples
    ids_all = set(d.record_id)
    journal = set(d[(d.source_type != "academic") | (d.document_type == "article")].record_id)
    stance_doc = ss.assign(actor=ss.actor)
    rob = robustness(d, mo["table"], doc, stance_doc, {
        "Baseline: economists": (ids_all, None),
        "Main theme only": (set(d[d.relevance == "main theme"].record_id), None),
        "English only": (set(d[d.language == "en"].record_id), None),
        "Journal articles only (research)": (journal, None),
        f"Without {today.year} (partial year)": (set(d[d.year < today.year].record_id), None),
        "High-recall motivation coding": (ids_all, mu["table"]),
    })
    # same results when research from every discipline is kept (press still excluded: no author)
    wide = d_theme[d_theme.source_type != "press"]
    mo_w = A.motivations(wide, ROOT)
    snt_w = nlp.tone(wide, ROOT, ROOT / "data/cache/tone.csv")
    snt_w = snt_w[snt_w.record_id.isin(wide.record_id)].assign(actor=lambda x: x.source_type.replace({"official": "speech"}))
    doc_w = snt_w.groupby(["record_id", "actor"]).agg(tone=("tone", "mean"), year=("year", "first")).reset_index()
    ss_w = pd.read_csv(ROOT / "data/cache/stance_sample.csv")
    rob += robustness(wide, mo_w["table"], doc_w, ss_w[ss_w.record_id.isin(wide.record_id)],
                      {"All disciplines (not only economists)": (set(wide.record_id), None)})

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
    build = OUT / ".build"  # compile aside, then swap the PDF in atomically: an open viewer never sees a half-written file
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={build}", "cbdc_corpus.tex"],
                   cwd=OUT, check=True, stdout=subprocess.DEVNULL)
    tmp = OUT / ".cbdc_corpus.pdf.tmp"
    shutil.copyfile(build / "cbdc_corpus.pdf", tmp)
    os.replace(tmp, OUT / "cbdc_corpus.pdf")
    print(f"Wrote {OUT / 'cbdc_corpus.pdf'}")


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

% links between each result and the slide that explains its method
\makeatletter\newcommand{\slabel}[1]{\edef\@currentlabel{\the\c@framenumber}\label{#1}}\makeatother
\newcommand{\method}[1]{\hfill\hyperref[#1]{\tiny\textcolor{c4}{$\triangleright$ method explained: slide~\ref*{#1}}}}
\newcommand{\usedon}[1]{\hyperref[#1]{\textcolor{c4}{slide~\ref*{#1}}}}
\newcommand{\mhead}[1]{\textcolor{c1}{\textbf{#1}}\\[1pt]}

\title{How Economists Discuss Central Bank Digital Currencies}
\subtitle{Themes, tone and stance, 2014--<<Y1>>: a bibliometric and text-as-data study}
\author{Louis Drogo, Kenza Redwane and Adriel Mazere}
\institute{M2 -- Private crypto}
\date{<<DATE>>}
\begin{document}
\begin{frame}
\titlepage
\note{\textbf{Title (0:15). Total: about 25 minutes.}}
\end{frame}

\begin{frame}{The paper in one slide}
\textbf{Research question.} How has economists and institutional discourse on CBDCs evolved between 2014 and <<Y1>>, and to what extent does this evolution align with changes in the policy environment?\\[5pt]
\small
\begin{tabular}{@{}ll@{}}
\alert{Who} writes? & research economists, central banks and institutions\\
\alert{What} do they write about? & themes and topics, and how they shift\\
\alert{How} do they discuss CBDCs? & \textbf{tone and stance: the core of the paper}\\
\alert{Why}, in their own words? & stated motivations\\
\alert{When}, and with which environment? & bursts, policy events, sanctions\\
\end{tabular}\\[5pt]
\textbf{Scope.} Descriptive: we measure changes and show what moves with them; we do not claim causes.
\vfill
\begin{center}<<N_ALL>> texts collected \;$\to$\; <<N_THEME>> with CBDC as a theme \;$\to$\; \alert{<<N>> written by economists}\end{center}
\note{\textbf{The paper in one slide (1:00).} One question, five sub-questions; tone and stance are the core.}
\end{frame}

\begin{frame}{Outline}
\tableofcontents
\note{\textbf{Outline (0:10).}}
\end{frame}

% =====================================================================
\section{Introduction}

\begin{frame}{Related literature: CBDCs}
% to be completed from the reading list
\note{\textbf{Literature (0:45).}}
\end{frame}

\begin{frame}{Related literature: what the field builds on}
% to be completed: reading list + co-citation clusters (section ``Who writes'')
\note{\textbf{Literature (0:45).}}
\end{frame}

\begin{frame}{What this paper adds to <<N_REVIEWS>> earlier reviews}
\footnotesize Five recent bibliometric reviews:\\[2pt]
\begin{tabular}{@{}llrl@{}}\toprule
\textbf{Study} & \textbf{Data} & \textbf{Documents} & \textbf{Tools}\\\midrule
Kvedaravi\v{c}i\={u}t\.{e} \& \v{S}apkauskien\.{e} (2024) & WoS + Scopus & 228 & VOSviewer\\
Fahad \& Bulut (2024) & Scopus & 169 of 323 & Biblioshiny\\
Bapat \& Sharma (2025) & Scopus & 371 & bibliometrics + TCCM\\
Prodan et al.\ (2024) & Scopus + WoS & 747 of 1{,}024 & PRISMA, VOSviewer\\
Paulovici (2026) & Scopus & 892 & VOSviewer\\\midrule
\textbf{This study} & \textbf{5 databases + central banks} & \textbf{<<N>>} & \textbf{+ text as data}\\
\bottomrule\end{tabular}\\[6pt]
\small
\begin{itemize}\setlength\itemsep{1pt}
\item \alert{Gap}: one or two databases, mostly English journal articles; central banks absent; who writes is not checked.
\item Maps say \emph{what} is studied, not \emph{how} (tone, stance) or \emph{why}; no measure is validated.
\item \alert{Here}: economists only (checked text by text), all languages, tone and stance over time, every coder validated.
\end{itemize}
\scriptsize Found in our own corpus: <<N_REVIEWS>> titles announce a literature, systematic or bibliometric review of CBDCs (<<REV_Y0>>--<<REV_Y1>>).
\note{\textbf{What this paper adds (1:00).} The closest is Hoang et al.\ (2023, RIBAF): text mining, but academic only and no stance.}
\end{frame}

\begin{frame}{Approach: one tool per question}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lp{0.36\textwidth}p{0.3\textwidth}l@{}}\toprule
\textbf{Question} & \textbf{Tool} & \textbf{Why this one} & \textbf{Explained}\\\midrule
Who? & co-citation, co-authorship (association strength, Louvain); laws; log-odds with Dirichlet prior & standard science maps; log-odds shrinks rare words & \usedon{m:net}, \usedon{m:logodds}\\
What? & thematic map; \alert{BERTopic} (SBERT, UMAP, HDBSCAN); labels by \alert{c-TF-IDF} & groups texts by meaning, across languages & \usedon{m:sbert}, \usedon{m:bertopic}\\
How? & tone: \alert{FinBERT} on CBDC sentences ($\kappa = <<TONE_KAPPA>>$); stance: reading + \alert{local LLM} ($\kappa = <<STANCE_K>>$) & best of the coders tested against reading & \usedon{m:finbert}, \usedon{m:stance}\\
Why? & \alert{validated codebook} ($\kappa = <<MVAL_CB>>$) + \alert{covariate logit} & beats SBERT, NLI, LLMs ($\kappa \le <<MVAL_BEST_ALT>>$) & \usedon{m:codebook}, \usedon{m:logit}\\
When? & \alert{Kleinberg bursts}; \alert{event studies} with placebo dates; \alert{difference-in-differences} & placebo fixes false positives (<<EV_P>>\%) & \usedon{m:burst}, \usedon{m:event}, \usedon{m:did}\\\midrule
All & 7 robustness specifications; every coder checked on hand-coded texts & & \usedon{m:robust}\\
\bottomrule\end{tabular}\\[4pt]
\small Every result slide links to the slide that explains its method (appendix).
\note{\textbf{Approach (0:45).}}
\end{frame}

% =====================================================================
\section{Data}

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
\item \alert{Recall first}: collect everything, then filter in four steps: homonyms, centrality, labels, \alert{economists}.
\end{itemize}
\note{\textbf{Pipeline (0:40).}}
\end{frame}

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
\note{\textbf{Homonyms (0:30).}}
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
\note{\textbf{Theme score (0:30).}}
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
\item Secondary theme is noisy $\Rightarrow$ robustness on main theme only.
\end{itemize}
\end{column}
\end{columns}
\vspace{2pt}
\small Kept: \alert{main + secondary theme} (<<N_THEME>> texts).
\note{\textbf{Validating the filter (0:40).}}
\end{frame}

\begin{frame}{Step 4: is it written by an economist?}
\slabel{r:econ}
\footnotesize
\begin{tabular}{@{}p{0.62\textwidth}rl@{}}\toprule
\textbf{Texts with CBDC as a theme} & \textbf{Texts} & \\\midrule
<<ECON_ROWS>>
\bottomrule\end{tabular}\\[4pt]
\scriptsize Field = OpenAlex primary topic (``Economics, Econometrics and Finance''); institutions = authors' affiliations in OpenAlex. Research dropped: <<ECON_DROPPED>>.\\[4pt]
\small
\begin{itemize}\setlength\itemsep{1pt}
\item \alert{<<N>> texts by economists}: research economists, and central banks and international institutions (staff economists, institutional voice).
\item Press dropped: no author is known, and journalists are not economists.
\end{itemize}
\method{m:econ}
\note{\textbf{Economists (0:50).} This answers the course's question: the corpus is economists' writing, checked text by text.}
\end{frame}

\begin{frame}{Sources and completeness (economists' corpus)}
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
\note{\textbf{Sources (0:30).}}
\end{frame}

\begin{frame}{A literature that took off after 2020}
\slabel{r:growth}
\begin{columns}[c]
\begin{column}{0.66\textwidth}
\begin{tikzpicture}
\begin{axis}[ybar stacked, width=\textwidth, height=0.7\textheight, bar width=11pt,
  cycle list name=corpus, xtick=data, xticklabel style={/pgf/number format/1000 sep=, font=\scriptsize, rotate=45, anchor=north east},
  ymin=0, legend style={at={(0.02,0.98)},anchor=north west,font=\scriptsize,draw=none},
  ylabel={Documents}, tick label style={font=\scriptsize}, axis lines*=left, ymajorgrids]
<<YEAR_PLOTS>>
\end{axis}
\end{tikzpicture}
\end{column}
\begin{column}{0.32\textwidth}\small
\begin{itemize}
\item From <<GROWTH>>.
\item Growth <<CAGR>>\% a year.
\item Turning point 2020: Libra (2019), COVID, the ECB report (Oct.\ 2020).
\end{itemize}
\scriptsize <<BEFORE>> documents before <<YEAR0>>; <<THIS_YEAR>> runs to <<THIS_MONTH>>.
\end{column}
\end{columns}
\note{\textbf{Growth (0:30).}}
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
\item English dominates: <<EN_N>> texts (<<EN_PCT>>\%); the chart shows the others.
\item Second language: \alert{<<LANG2>>} (<<LANG2_N>> texts).
\item Everything kept in its original language: the topic model and the codebook are multilingual.
\end{itemize}
\end{column}
\end{columns}
\note{\textbf{Languages (0:20).}}
\end{frame}

% =====================================================================
\section{Who writes}

\begin{frame}{Research economists: outlets, most cited, laws}
\slabel{r:research}
\begin{columns}[T]
\begin{column}{0.4\textwidth}\scriptsize\textbf{Main outlets}\\[2pt]
\begin{tabular}{@{}p{0.8\linewidth}r@{}}<<VENUE_ROWS>>\end{tabular}\\[4pt]
\textbf{Bibliometric laws.} <<LAWS>>
\end{column}
\begin{column}{0.58\textwidth}\scriptsize\textbf{Most cited papers} (global citations)\\[2pt]
\begin{tabular}{@{}p{0.2\linewidth}p{0.66\linewidth}r@{}}
<<CITED_ROWS>>
\end{tabular}\\[6pt]
\small Working papers (SSRN, RePEc) weigh as much as journals; a young, open field: most authors write once.
\end{column}
\end{columns}
\method{m:laws}
\note{\textbf{Research economists (0:40).}}
\end{frame}

\begin{frame}{Who speaks: central banks, institutions, research centres}
<<VOICES>>
\vspace{3pt}
\small \alert{<<TOP_BANK>>} is the most vocal central bank: the digital euro makes it the hub of the debate.
\note{\textbf{Voices (0:35).} Institutions under their own names: speeches by central bank, documents by institution, research by the authors' institutions.}
\end{frame}

\begin{frame}{Co-citation: the intellectual base}
\slabel{r:cocit}
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
<<COCIT_STATS>> \method{m:net}
\note{\textbf{Co-citation (0:50).} Core: <<COCIT_CORE>>.}
\end{frame}

\begin{frame}{Co-authorship: research communities}
\slabel{r:coauth}
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
\method{m:net}
\note{\textbf{Co-authorship (0:45).}}
\end{frame}

\begin{frame}{Who says what: research economists vs central bankers}
\slabel{r:logodds}
\footnotesize
Words most over-used by each group compared with the other ($z$-scores):\\[4pt]
\begin{tabular}{@{}p{0.42\textwidth}p{0.42\textwidth}@{}}\toprule
<<DIST_HEAD>>\\\midrule
<<DIST_ROWS>>
\bottomrule\end{tabular}\\[4pt]
\scriptsize English texts; research: title + abstract; central bankers: speeches; topic words (\emph{CBDC, digital, currency\dots}) removed.\\[3pt]
\small \alert{Reading}: <<DIST_TAKE>>.
\method{m:logodds}
\note{\textbf{Framing (0:40).}}
\end{frame}

% =====================================================================
\section{What is written}

\begin{frame}{Thematic map: which themes drive the field?}
\slabel{r:thematic}
\begin{columns}[c]
\begin{column}{0.47\textwidth}
<<THEMATIC>>
\end{column}
\begin{column}{0.51\textwidth}\scriptsize
Keyword clusters placed by \alert{centrality} (links to other themes) and \alert{density} (internal cohesion).\\[3pt]
\begin{tabular}{@{}lp{0.62\linewidth}l@{}}\toprule
 & \textbf{Theme} (top keywords) & \textbf{Quadrant}\\\midrule
<<THEME_ROWS>>
\bottomrule\end{tabular}\\[3pt]
\small Motor themes: <<MOTOR>>.
\end{column}
\end{columns}
\method{m:callon}
\note{\textbf{Thematic map (0:45).}}
\end{frame}

\begin{frame}{How the themes evolve}
\slabel{r:evo}
\scriptsize
\begin{tabular}{@{}p{0.42\textwidth}rrrr@{}}\toprule
\textbf{Keyword} (\% of the period's papers) & <<EVO_HEAD>>\\\midrule
<<EVO_ROWS>>
\bottomrule\end{tabular}\\[4pt]
Sorted by change between the last two periods. Darker = more frequent.\\[3pt]
\textbf{New in each period} (c-TF-IDF, periods as classes): <<PERIOD_TERMS>>.\\[2pt]
\small Rising fastest: \alert{<<EVO_UP>>}; falling most: \alert{<<EVO_DOWN>>}.
\method{m:ctfidf}
\note{\textbf{Evolution (0:35).}}
\end{frame}

\begin{frame}{Topics found by meaning (SBERT + BERTopic)}
\slabel{r:topics}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}rp{0.4\textwidth}rrrrrr@{}}\toprule
 & \textbf{Topic} (top c-TF-IDF terms) & \textbf{Texts} & \multicolumn{3}{c}{\textbf{\% of the period's texts}} & \multicolumn{2}{c}{\textbf{Adjusted, points}}\\
 & & & <<PERIOD_HEAD>> & \textbf{change} & \textbf{CB}\\\midrule
<<TOPIC_ROWS>>
\bottomrule\end{tabular}\\[3pt]
<<TOPIC_N>> texts; <<TOPIC_K>> topics (<<TOPIC_OUT>>\% outliers reassigned). Adjusted: logit on period, actor, English; change = 2023--26 vs 2014--19; CB = central banks vs research; \textbf{bold}: 95\% CI excludes 0.\\[2pt]
\small Topics tied to \alert{one jurisdiction}: <<TOPIC_JUR>>.
\method{m:bertopic}
\note{\textbf{Topics (1:00).} Embeddings group texts by meaning across languages; the literature follows national projects.}
\end{frame}

% =====================================================================
\section{How: tone and stance}

\begin{frame}{Tone: how positive is what economists say about CBDCs?}
\slabel{r:tone}
\begin{columns}[T]
\begin{column}{0.46\textwidth}
\begin{tikzpicture}
\begin{axis}[width=\textwidth, height=0.62\textheight, xtick={<<TONE_XTICKS>>}, xticklabel style={/pgf/number format/1000 sep=, font=\tiny, rotate=45, anchor=north east, inner sep=1pt},
  yticklabel style={font=\scriptsize, /pgf/number format/fixed}, ylabel={net tone}, ylabel style={font=\scriptsize},
  legend style={at={(0.5,-0.18)}, anchor=north, font=\tiny, draw=none, legend columns=2}, axis lines*=left, ymajorgrids]
<<TONE_PLOTS>>
\end{axis}
\end{tikzpicture}
\end{column}
\begin{column}{0.52\textwidth}\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{0.3\linewidth}rrrrr@{}}\toprule
 & \textbf{Texts} & \textbf{Sent.} & \textbf{\%+} & \textbf{\%--} & \textbf{Net tone}\\\midrule
<<TONE_ROWS>>
\bottomrule\end{tabular}\\[4pt]
FinBERT on the <<N_SENT>> English sentences that name a CBDC. Sentence tone $= P(+) - P(-)$, averaged per text, then per group ($\pm$ 95\% CI). <<TONE_VAL>>\\[4pt]
\small \alert{Reading}: tone rises steadily in both groups; research economists from <<TONE_ACAD>>; central banks stay slightly more positive.
\end{column}
\end{columns}
\method{m:finbert}
\note{\textbf{Tone (1:00).} Only sentences naming a CBDC are scored: tone about CBDCs, not about the economy.}
\end{frame}

\begin{frame}{Do policy events move the tone?}
\slabel{r:tev}
\begin{columns}[T]
\begin{column}{0.6\textwidth}\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrr@{}}\toprule
\textbf{Date} & \textbf{Event} & \textbf{Before} & \textbf{After} & $\Delta$\textbf{tone} & $p$\\\midrule
<<TEV_ROWS>>
\bottomrule\end{tabular}
\end{column}
\begin{column}{0.38\textwidth}\small
For each event: mean tone of the texts in the 180 days \emph{after} minus the 180 days \emph{before}. Compared with 500 random dates: $p$ = share of random dates with a larger $|\Delta|$ (95th percentile: <<TEV_95>>).\\[6pt]
\alert{Result}: <<TEV_NSIG>> of 12 events move the tone (smallest $p$ = <<TEV_MIN_P>>).\\[4pt]
Tone \alert{drifts}, it does not \alert{jump}: no single event explains the rise.
\end{column}
\end{columns}
\method{m:tev}
\note{\textbf{Tone and events (0:50).} A negative result, and an informative one: the change is gradual.}
\end{frame}

\begin{frame}{Stance: for or against CBDCs?}
\slabel{r:stance}
\begin{columns}[T]
\begin{column}{0.44\textwidth}\scriptsize
\textbf{Level}, read by hand, \% [95\% CI]\\[2pt]
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{0.36\linewidth}rrr@{}}\toprule
 & $n$ & \textbf{Favourable} & \textbf{Critical}\\\midrule
<<STANCE_LEVEL>>
\bottomrule\end{tabular}\\[6pt]
\small <<STANCE_TAKE>>.
\end{column}
\begin{column}{0.54\textwidth}\scriptsize
\textbf{Change} (local LLM, <<STANCE_N>> texts sampled per actor $\times$ period), \% favourable\\[2pt]
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{0.27\linewidth}rrrr@{}}\toprule
 & <<PERIOD_HEAD>> & \textbf{Change, pts}\\\midrule
<<STANCE_TREND>>
\bottomrule\end{tabular}\\[4pt]
The LLM ($\kappa = <<STANCE_K>>$ on <<ST_N>> economists' texts, best of four coders, \usedon{b:stance}) calls <<STANCE_RAWFAV>>\% of the hand-coded texts favourable against <<STANCE_TRUEFAV>>\% by reading: its levels are inflated, so only its \emph{changes} are used. \textbf{Bold}: CI excludes 0.\\[4pt]
\small \alert{Reading}: both groups become more favourable; central banks much more, as their own projects advance.
\end{column}
\end{columns}
\method{m:stance}
\note{\textbf{Stance (1:00).} Tone = how positive the words are; stance = whether the author is for or against.}
\end{frame}

% =====================================================================
\section{Why: stated motivations}

\begin{frame}[fragile]{Coding stated motivations}
\slabel{r:codebook}
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
\item Motivations are stated in the \alert{introduction} (else abstract, else opening).
\item 10 motivations; codebook built by reading introductions; several per text allowed.
\item <<MOTIVE_N>> texts coded; <<MOTIVE_ANY>>\% state at least one.
\item <<VAL_LINE>>
\end{itemize}
\end{column}
\end{columns}
\method{m:codebook}
\note{\textbf{Motivations: method (0:35).}}
\end{frame}

\begin{frame}{Why: what economists say CBDCs are for}
\slabel{r:motives}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}p{0.29\textwidth}rrrrr@{}}\toprule
 & \multicolumn{3}{c}{\textbf{\% of texts}} & \multicolumn{2}{c}{\textbf{Adjusted effect, points [95\% CI]}}\\
\textbf{Motivation} & <<MOTIVE_HEAD>>\\
 & & & & \textbf{2023--26 vs 2014--19} & \textbf{vs research}\\\midrule
<<MOTIVE_ROWS>>
\bottomrule\end{tabular}\\[3pt]
Logit of each motivation on period, actor and English; average marginal effects, robust SEs. \textbf{Bold}: CI excludes 0.
\begin{itemize}\setlength\itemsep{0pt}
\item \alert{Rising:} <<MOTIVE_UP>>. \quad \alert{Falling:} <<MOTIVE_DOWN>>.
\item From a \alert{monetary-policy instrument} to a tool for \alert{inclusion, payments and geopolitics}. <<VAL_CAVEAT>>
\end{itemize}
\method{m:logit}
\note{\textbf{Motivations: results (1:00).}}
\end{frame}

\begin{frame}{Is the codebook the best coder?}
\slabel{r:coders}
\begin{columns}[T]
\begin{column}{0.58\textwidth}\scriptsize
\begin{tabular}{@{}lrrrrr@{}}\toprule
\textbf{Coder} & \textbf{P} & \textbf{R} & $F_1$ & $\kappa$ & $\kappa$\textbf{, all}\\\midrule
<<MVAL_ROWS>>
\bottomrule\end{tabular}\\[3pt]
The <<MVAL_N>> held-out texts written by economists $\times$ 10 motivations; last column: $\kappa$ on all <<MVAL_N_ALL>> held-out texts. Thresholds and classifiers set on 100 development texts.
\end{column}
\begin{column}{0.4\textwidth}\small
\begin{itemize}\setlength\itemsep{2pt}
\item \alert{The codebook wins}: $\kappa = <<MVAL_CB>>$; best alternative <<MVAL_BEST_ALT>>.
\item Models confuse ``mentions privacy'' with ``studies CBDCs because of privacy''.
\item Trends hold: <<MOTIVE_ROBUST>>
\end{itemize}
\end{column}
\end{columns}
\method{m:codebook}
\note{\textbf{Measuring motivations (0:35).}}
\end{frame}

% =====================================================================
\section{When: the environment}

\begin{frame}{Which themes burst, and when}
\slabel{r:burst}
\hspace*{3.3cm}<<BURSTS>>\\[2pt]
\small \alert{Reading}: early bursts on <<BURST_EARLY>>; the latest on <<BURST_LATE>>.
\method{m:burst}
\note{\textbf{Bursts (0:40).} The data say when a theme suddenly takes off.}
\end{frame}

\begin{frame}{Attention around policy events}
\slabel{r:event}
\begin{columns}[c]
\begin{column}{0.48\textwidth}
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
\begin{column}{0.5\textwidth}\tiny
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}rlp{0.52\linewidth}rrr@{}}\toprule
 & \textbf{Date} & \textbf{Event} & \textbf{Ratio} & $z$ & $p$\\\midrule
<<EV_ROWS>>
\bottomrule\end{tabular}\\[3pt]
\scriptsize Texts in the 15 days after the event vs.\ the rate of the 3 months before. $p$ = share of <<EV_NPL>> random dates with a larger $z$ (the naive test flags <<EV_P>>\% of them).\\[2pt]
\alert{Reading}: <<EV_TAKE>>
\end{column}
\end{columns}
\method{m:event}
\note{\textbf{Event study (0:50).} Red lines: $p < 0.05$ against placebo dates.}
\end{frame}

\begin{frame}{The 2022 sanctions: did geopolitics enter the discussion?}
\slabel{r:did}
\begin{columns}[T]
\begin{column}{0.5\textwidth}
\begin{tikzpicture}
\begin{axis}[width=\textwidth, height=0.6\textheight, xtick={2020,...,<<THIS_YEAR>>}, xticklabel style={/pgf/number format/1000 sep=, font=\scriptsize},
  yticklabel style={font=\scriptsize}, ylabel={gap, points}, ylabel style={font=\scriptsize}, axis lines*=left, ymajorgrids,
  extra x ticks={2021.5}, extra x tick labels={}, extra x tick style={grid=major, major grid style={c2, dashed}}]
\addplot[c1, thick, mark=*, mark size=1.5pt, error bars/.cd, y dir=both, y explicit] coordinates {<<DID_PLOT>>};
\draw[gray] (axis cs:2020,0) -- (axis cs:<<THIS_YEAR>>,0);
\end{axis}
\end{tikzpicture}\\
\scriptsize Gap in the share stating \emph{geopolitics and sanctions}: texts about Russia (or in Russian) minus the others, relative to 2021; 95\% CI.
\end{column}
\begin{column}{0.48\textwidth}\scriptsize
<<DID_NT>> texts about Russia among <<DID_N>> (2020--<<THIS_YEAR>>). Pooled effect after 2021, every motivation:\\[2pt]
\begin{tabular}{@{}p{0.6\linewidth}l@{}}\toprule
\textbf{Motivation} & \textbf{DiD, points}\\\midrule
<<DID_ROWS>>
\bottomrule\end{tabular}\\[4pt]
\small \alert{Reading}: geopolitics rises by <<DID_POOLED>> more in texts about Russia, and the gap widens each year. But <<DID_OTHERS>> other motivations shift too: \alert{consistent with} a sanctions effect, not proof.
\end{column}
\end{columns}
\method{m:did}
\note{\textbf{Sanctions (1:00).} Only one clean pre-year (2020): pre-trends cannot be fully tested.}
\end{frame}

% =====================================================================
\section{The discussion and its environment}

\begin{frame}{Putting it together, period by period}
\slabel{r:syn}
\scriptsize
\setlength{\tabcolsep}{4pt}
\tiny
\begin{tabular}{@{}p{0.19\textwidth}p{0.25\textwidth}p{0.25\textwidth}p{0.26\textwidth}@{}}\toprule
 & <<PERIOD_HEAD>>\\\midrule
<<SYN_ROWS>>
\bottomrule\end{tabular}\\[2pt]
\scriptsize
\begin{itemize}\setlength\itemsep{0pt}
\item \alert{2014--19}: a monetary-policy question, alongside Libra and the decline of cash; few texts, cautious tone.
\item \alert{2020--22}: central bank reports and pilots; the discussion grows and turns favourable.
\item \alert{2023--26}: national projects, sanctions, cross-border payments; the most favourable period.
\end{itemize}
\note{\textbf{Synthesis (1:00).} Each row is a result from the previous sections.}
\end{frame}

\begin{frame}{What we can and cannot say about causes}
\slabel{r:causal}
\footnotesize
\begin{tabular}{@{}p{0.25\textwidth}p{0.27\textwidth}p{0.2\textwidth}p{0.18\textwidth}@{}}\toprule
\textbf{Link} & \textbf{Test} & \textbf{Result} & \textbf{Verdict}\\\midrule
Events $\to$ attention & event study vs.\ placebo dates & <<EV_NSIG>> of <<EV_NTOT>> events & \textcolor{c6}{\textbf{timing evidence}}\\
Events $\to$ tone & 180-day windows vs.\ placebo dates & <<TEV_NSIG>> of 12 events & \textcolor{c2}{\textbf{no effect found}}\\
Sanctions $\to$ geopolitics motive & difference-in-differences & <<DID_POOLED>> & \textcolor{c3}{\textbf{consistent with}}\\
Events $\to$ stance & -- & sample too small around each date & not testable\\
Motivations $\leftrightarrow$ stance & -- & measured in the same texts & association only\\
\bottomrule\end{tabular}\\[8pt]
\small
\begin{itemize}\setlength\itemsep{2pt}
\item The paper stays \alert{descriptive}: it measures changes and shows what moves with them.
\item Explaining \emph{why} economists changed their stance would need other evidence (interviews, authors' careers): hypotheses for further work.
\end{itemize}
\note{\textbf{Causality (0:50).} Answers the teacher's question directly.}
\end{frame}

% =====================================================================
\section{Robustness}

\begin{frame}{Robustness}
\slabel{r:robust}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrcccrr@{}}\toprule
 & & \multicolumn{2}{c}{\textbf{Why: change, points}} & \textbf{How: tone} & \textbf{How: stance} & \textbf{When}\\
\textbf{Specification} & \textbf{Texts} & \textbf{inclusion} & \textbf{monetary policy} & \textbf{CB -- research} & \textbf{fav.\ CB -- research} & \textbf{events}\\\midrule
<<ROB_ROWS>>
\bottomrule\end{tabular}\\[4pt]
Changes: 2023--26 vs 2014--19, adjusted (logit, 95\% CI). Tone: mean net tone gap. Stance: raw LLM labels, points. Events: $p < 0.05$ against placebo dates.\\[3pt]
\small Signs and orders of magnitude hold in every specification, including when research from every discipline is kept.
\method{m:robust}
\note{\textbf{Robustness (0:45).}}
\end{frame}

% =====================================================================
\section{Conclusion}

\begin{frame}{Takeaways and limits}
\begin{columns}[T]
\begin{column}{0.56\textwidth}\footnotesize
\textbf{How economists' discussion of CBDCs evolved}
\begin{itemize}\setlength\itemsep{1pt}
\item \alert{Who:} <<N>> texts; central bank economists write <<CB_SHARE>>\% of the papers but get <<CB_CITE>>\% of citations.
\item \alert{What:} from monetary policy and design to inclusion and national projects (digital ruble, rupee, e-CNY).
\item \alert{How:} tone and stance become more favourable; research economists stay mostly neutral, central banks advocate.
\item \alert{Why:} rising <<MOTIVE_UP>>; falling <<MOTIVE_DOWN>>.
\item \alert{When:} <<EV_NSIG>> of <<EV_NTOT>> events move attention, <<TEV_NSIG>> of 12 move tone; geopolitics rises after the 2022 sanctions.
\end{itemize}
\end{column}
\begin{column}{0.41\textwidth}\footnotesize
\textbf{Limits}
\begin{itemize}\setlength\itemsep{1pt}
\item Economists identified by field and affiliation, not by training.
\item Motivations: recall $\approx$ 0.5 (lower bounds). Tone: English only. Stance: LLM for changes only.
\item One coder (AI-assisted) for the validation samples.
\item Descriptive: causes stay hypotheses.
\end{itemize}
\end{column}
\end{columns}
\note{\textbf{Takeaways (0:50).} Thank you; questions.}
\end{frame}

% =====================================================================
\appendix
\section*{Methods explained}

\begin{frame}{Methods explained: how to read this appendix}
\small
One slide per tool, each in four parts:
\begin{itemize}
\item \textbf{Idea}: what the tool does, in words.
\item \textbf{How}: the computation, step by step, with its formula.
\item \textbf{Why this tool}: what it solves, and what it was compared with.
\item \textbf{Used on}: links to the result slides.
\end{itemize}
\vspace{4pt}
\scriptsize
\begin{tabular}{@{}llll@{}}\toprule
\textbf{Tool} & \textbf{Slide} & \textbf{Tool} & \textbf{Slide}\\\midrule
Who is an economist & \usedon{m:econ} & Covariate-adjusted logit & \usedon{m:logit}\\
Networks: association strength, Louvain & \usedon{m:net} & Codebook and validation & \usedon{m:codebook}\\
Thematic map (Callon) & \usedon{m:callon} & Tone (FinBERT) & \usedon{m:finbert}\\
Bibliometric laws & \usedon{m:laws} & Stance (reading + LLM) & \usedon{m:stance}\\
Log-odds with Dirichlet prior & \usedon{m:logodds} & Bursts (Kleinberg) & \usedon{m:burst}\\
Sentence embeddings (SBERT) & \usedon{m:sbert} & Event study and placebo & \usedon{m:event}\\
Topic model (BERTopic) & \usedon{m:bertopic} & Tone around events & \usedon{m:tev}\\
Class-based TF-IDF & \usedon{m:ctfidf} & Difference-in-differences & \usedon{m:did}\\
 & & Robustness & \usedon{m:robust}\\
\bottomrule\end{tabular}
\end{frame}

\begin{frame}{Method: who counts as an economist}
\slabel{m:econ}
\begin{columns}[T]
\begin{column}{0.5\textwidth}\footnotesize
\mhead{Idea}
The course asks how \emph{economists} discuss CBDCs, so each text must be traced to an economist or an economic institution.\\[4pt]
\mhead{How}
\begin{itemize}\setlength\itemsep{1pt}
\item \textbf{Research}: kept if OpenAlex classifies it in \emph{Economics, Econometrics and Finance} (primary topic, from title, abstract, venue, citations), or an author works at a central bank, the IMF or the BIS; without OpenAlex record: an economics venue.
\item \textbf{Institutions}: central bank speeches; documents of central banks and international financial institutions.
\item \textbf{Dropped}: other fields (computer science, law\dots), governments and parliaments, the press.
\end{itemize}
\end{column}
\begin{column}{0.46\textwidth}\footnotesize
\mhead{Why this tool}
Authors' training is not recorded anywhere at scale; the field of the paper and the employer of the author are, and they are what defines a discipline in bibliometrics.\\[4pt]
Speeches are kept as \emph{institutional} voices: written with the bank's economics staff, even when the speaker is not an economist.\\[4pt]
Robustness: results with every discipline kept are on \usedon{r:robust}.\\[6pt]
\mhead{Used on}
\usedon{r:econ}, and every result after it.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: networks (association strength, Louvain)}
\slabel{m:net}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
Two items (references, authors, keywords) are linked when they appear in the same paper; groups of tightly linked items are schools, teams or themes.\\[4pt]
\mhead{How}
\begin{enumerate}\setlength\itemsep{1pt}
\item Count $c_{ij}$ = papers where $i$ and $j$ appear together; $c_i$ = papers with $i$.
\item Normalise by \alert{association strength} $s_{ij} = c_{ij} / (c_i c_j)$: the ratio of observed to expected co-occurrence.
\item Cluster with \alert{Louvain}: move nodes between groups while the modularity
$Q = \frac{1}{2m}\sum_{ij}\big[s_{ij} - \frac{k_ik_j}{2m}\big]\delta(g_i,g_j)$ rises.
\item Draw: force-directed layout; size $\propto\sqrt{c_i}$.
\end{enumerate}
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
Raw counts favour popular items: two classics are co-cited often by chance. Association strength removes this (van Eck \& Waltman, 2009); it is VOSviewer's default.\\[3pt]
Louvain is fast and needs no number of clusters; $Q$ tells how clear the grouping is.\\[3pt]
\textbf{Co-citation} (references cited together) shows the intellectual base; \textbf{co-authorship} shows teams.\\[6pt]
\mhead{Used on}
\usedon{r:cocit}, \usedon{r:coauth}, \usedon{r:thematic}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: thematic map (Callon)}
\slabel{m:callon}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
Place each cluster of keywords on two axes: how much it is linked to the rest of the field, and how developed it is inside.\\[4pt]
\mhead{How}
With the equivalence index $e_{ij} = c_{ij}^2 / (c_i c_j)$:
\[ \text{centrality} = 10\!\!\sum_{i\in C,\,j\notin C}\!\! e_{ij}, \qquad \text{density} = \frac{100}{|C|}\!\sum_{i,j\in C}\! e_{ij} \]
Quadrants, split at the medians:
\begin{itemize}\setlength\itemsep{0pt}
\item \textbf{motor}: central and dense; \textbf{basic}: central, loose;
\item \textbf{niche}: dense, isolated; \textbf{emerging or declining}: neither.
\end{itemize}
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
A keyword list says what is frequent; the map says what \emph{structures} the field. It is the standard tool of science mapping (Callon et al.\ 1991; Cobo et al.\ 2011; bibliometrix).\\[3pt]
Limit: keywords are partly machine-assigned, so the topic model (meaning-based) is used alongside.\\[6pt]
\mhead{Used on}
\usedon{r:thematic}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: bibliometric laws}
\slabel{m:laws}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
Three regularities describe the shape of any scientific field: how fast it grows, how output is spread among authors, and among journals.\\[4pt]
\mhead{How}
\begin{itemize}\setlength\itemsep{1pt}
\item \textbf{Growth}: compound annual rate $g = (N_{y_1}/N_{y_0})^{1/(y_1-y_0)} - 1$.
\item \textbf{Lotka}: share of authors with $n$ papers $f(n) \propto n^{-\beta}$; $\beta$ fitted on the log-log counts ($\beta \approx 2$ in mature fields).
\item \textbf{Bradford}: sort sources by output, cut in three zones holding $\nicefrac{1}{3}$ of papers each; the first zone is the core.
\end{itemize}
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
They situate the field: a high growth rate and a large $\beta$ (most authors write once) mean a young, open field around a small core of outlets.\\[3pt]
They are the descriptive baseline of every bibliometric review, so our numbers can be compared with the earlier reviews.\\[6pt]
\mhead{Used on}
\usedon{r:growth}, \usedon{r:research}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: distinctive words (log-odds with Dirichlet prior)}
\slabel{m:logodds}
\begin{columns}[T]
\begin{column}{0.54\textwidth}\footnotesize
\mhead{Idea}
Which words does one group use \emph{more} than the other, once chance is accounted for?\\[4pt]
\mhead{How}
For word $w$, group $i$ vs.\ the other group $j$, with counts $y$, totals $n$, and a prior $\alpha_w = \alpha_0 \times$ (corpus frequency of $w$):
\[\delta_w = \log\frac{y_{iw}+\alpha_w}{n_i+\alpha_0-y_{iw}-\alpha_w} - \log\frac{y_{jw}+\alpha_w}{n_j+\alpha_0-y_{jw}-\alpha_w}\]
\[ z_w = \delta_w \Big/ \sqrt{\tfrac{1}{y_{iw}+\alpha_w}+\tfrac{1}{y_{jw}+\alpha_w}} \]
Words are ranked by $z_w$.
\end{column}
\begin{column}{0.42\textwidth}\footnotesize
\mhead{Why this tool}
Plain frequencies list words everyone uses; TF-IDF or raw log-ratios are dominated by rare words (a word used twice by one group gets an infinite ratio). The prior \alert{shrinks} rare words towards zero and the $z$-score accounts for uncertainty (Monroe, Colaresi \& Quinn, 2008).\\[6pt]
\mhead{Used on}
\usedon{r:logodds}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: sentence embeddings (SBERT)}
\slabel{m:sbert}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
Turn each text into a vector of 384 numbers such that texts with similar \emph{meaning} get close vectors, whatever their words or language.\\[4pt]
\mhead{How}
\begin{itemize}\setlength\itemsep{1pt}
\item A transformer network reads title + abstract (or opening) and averages its word representations into one vector $\mathbf{e}_t$.
\item Model: \texttt{paraphrase-multilingual-MiniLM-L12-v2}, trained so that translations and paraphrases land together (50+ languages).
\item Similarity: cosine $\cos(\mathbf{e}_s,\mathbf{e}_t) = \frac{\mathbf{e}_s\cdot\mathbf{e}_t}{\|\mathbf{e}_s\|\|\mathbf{e}_t\|}$.
\end{itemize}
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
Keywords and word counts treat ``digital ruble'' and its Russian original (\emph{tsifrovoy rubl'}) as unrelated; embeddings do not. With <<EN_PCT>>\% English and many Russian texts, this is what makes a single topic model possible.\\[3pt]
Tested also as a motivation coder (similarity to prototypes): it lost to the codebook (\usedon{r:coders}).\\[6pt]
\mhead{Used on}
\usedon{r:topics}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: topic model (BERTopic)}
\slabel{m:bertopic}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
Find groups of texts about the same subject, without fixing the subjects in advance.\\[4pt]
\mhead{How}
\begin{enumerate}\setlength\itemsep{1pt}
\item \textbf{Embed} each text (SBERT).
\item \textbf{Reduce} 384 dimensions to 5 with \alert{UMAP}, which keeps neighbours together.
\item \textbf{Cluster} with \alert{HDBSCAN}: dense regions become topics (at least 40 texts; leaf selection, which favours fine topics over one large blob); texts in no dense region are outliers, then assigned to the nearest topic.
\item \textbf{Label} each topic by its c-TF-IDF terms (\usedon{m:ctfidf}).
\end{enumerate}
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
LDA and STM count words, one language at a time, and need the number of topics. BERTopic works on meaning, across languages, and finds the number itself (Grootendorst, 2022).\\[3pt]
The covariate part of STM is kept: topic shares are adjusted for period, actor and language (\usedon{m:logit}).\\[6pt]
\mhead{Used on}
\usedon{r:topics}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: class-based TF-IDF (c-TF-IDF)}
\slabel{m:ctfidf}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
Which words characterise a class of texts (a topic, or a period) compared with all the others?\\[4pt]
\mhead{How}
Merge the texts of each class $c$ into one document; for term $t$:
\[ W_{t,c} = \frac{\text{tf}_{t,c}}{|c|}\;\log\!\Big(1 + \frac{A}{f_t}\Big) \]
$\text{tf}_{t,c}$ = count of $t$ in class $c$, $|c|$ = words in $c$, $A$ = average words per class, $f_t$ = count of $t$ in all classes. Words in more than 15\% of texts and generic words are removed.
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
Ordinary TF-IDF scores words within one document; here the unit is a whole class. A frequent word in one class but rare elsewhere scores high: it is what is \emph{new} or \emph{specific}.\\[3pt]
Periods as classes answer ``when'': which vocabulary appears in each period.\\[6pt]
\mhead{Used on}
\usedon{r:topics} (topic labels), \usedon{r:evo} (periods).
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: covariate-adjusted logit}
\slabel{m:logit}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
A raw change between periods can come from a change in \emph{who} writes (more central banks, more Russian texts). Compare like with like.\\[4pt]
\mhead{How}
For each outcome $y_t \in\{0,1\}$ (a motivation, a topic):
\[ P(y_t=1) = \Lambda\big(a + b_{\text{period}} + c_{\text{actor}} + e\cdot\text{English}\big) \]
\begin{itemize}\setlength\itemsep{1pt}
\item \textbf{Average marginal effect}: predict every text in 2023--26, then in 2014--19, average the difference (points).
\item 95\% CI: robust (HC1) standard errors; coefficients drawn 400 times from their estimated distribution.
\end{itemize}
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
It is the ``prevalence'' model of a structural topic model (Roberts et al., 2014), applied to any coded outcome. Marginal effects are read in percentage points, unlike log-odds coefficients.\\[6pt]
\mhead{Used on}
\usedon{r:motives}, \usedon{r:topics}, \usedon{r:robust}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: codebook and validation}
\slabel{m:codebook}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
Write down, for each of 10 motivations, the phrases authors use to state it; flag a text when its introduction contains one.\\[4pt]
\mhead{How}
\begin{enumerate}\setlength\itemsep{1pt}
\item Read 100 introductions (\textbf{development set}); write regular expressions in 7 languages.
\item Code <<VAL_N_ALL>> \emph{other} texts by reading (\textbf{test set}), never used for tuning; scores on the <<VAL_N>> written by economists.
\item Compare: precision $=\frac{TP}{TP+FP}$, recall $=\frac{TP}{TP+FN}$, Cohen's
$\kappa = \frac{p_o-p_e}{1-p_e}$ (agreement beyond chance).
\end{enumerate}
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
Motivations need reading; a codebook makes the reading explicit, reproducible and checkable.\\[3pt]
Five machine coders were tested on the same texts (SBERT $\times$ 2, NLI mDeBERTa, Qwen2.5 3B and 7B): see \usedon{r:coders}.\\[3pt]
High precision, recall $\approx 0.5$: shares are lower bounds; changes are robust (\usedon{r:coders}).\\[6pt]
\mhead{Used on}
\usedon{r:codebook}, \usedon{r:motives}, \usedon{r:coders}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: tone (FinBERT)}
\slabel{m:finbert}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
Measure how positive or negative the sentences \emph{about CBDCs} are.\\[4pt]
\mhead{How}
\begin{enumerate}\setlength\itemsep{1pt}
\item Keep the English sentences that name a CBDC (same term list as the filter).
\item FinBERT (a BERT model trained on financial news) gives $P(+), P(0), P(-)$ for each sentence.
\item Sentence tone $= P(+) - P(-) \in[-1,1]$; averaged per text, then per group and year.
\item Validation: <<TONE_N>> sentences from economists' texts read blind; $\kappa = <<TONE_KAPPA>>$ (all <<TONE_KAPPA_ALL>>).
\end{enumerate}
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
Dictionary methods (Loughran--McDonald) miss context (``not risky''); general sentiment models misread finance (``liability'' is not negative). FinBERT is the standard for economic text (Araci, 2019).\\[3pt]
Scoring only CBDC sentences isolates the tone \emph{about CBDCs}, not about the economy.\\[3pt]
Limit: it sees tone in neutral sentences, so compare groups and years, not levels.\\[6pt]
\mhead{Used on}
\usedon{r:tone}, \usedon{r:tev}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: stance (reading + local language model)}
\slabel{m:stance}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
Tone is about words; stance is about the \emph{author's position}: favourable, neutral or critical towards issuing a CBDC.\\[4pt]
\mhead{How}
\begin{itemize}\setlength\itemsep{1pt}
\item \textbf{Levels}: texts read by hand per actor; share with a \alert{Wilson} interval $\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+z^2/n}$.
\item \textbf{Changes}: Qwen2.5 7B, run locally, labels a random sample of up to 100 texts per actor $\times$ period; change $= \hat p_{\text{late}} - \hat p_{\text{early}}$, CI $\pm 1.96\sqrt{\sum \hat p(1-\hat p)/n}$.
\end{itemize}
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
Stance needs understanding of the argument: only a language model can do it at scale. Four coders were tested (\usedon{b:stance}); the LLM is best ($\kappa = <<STANCE_K>>$) but over-calls ``favourable''.\\[3pt]
A constant bias cancels in a difference: so the LLM gives \emph{changes}, reading gives \emph{levels}.\\[6pt]
\mhead{Used on}
\usedon{r:stance}, \usedon{r:syn}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: bursts (Kleinberg)}
\slabel{m:burst}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
Let the data say when a keyword suddenly becomes frequent, instead of choosing dates.\\[4pt]
\mhead{How}
For keyword $w$, quarter $t$: $r_t$ of $d_t$ papers use it.
\begin{itemize}\setlength\itemsep{1pt}
\item Two states: normal rate $p_0 = R/D$ (overall), burst rate $p_1 = 2p_0$.
\item Cost of a quarter in state $i$: $\sigma_i(t) = -\ln\big[\binom{d_t}{r_t} p_i^{r_t}(1-p_i)^{d_t-r_t}\big]$; entering a burst costs $\ln T$.
\item The cheapest sequence of states (Viterbi algorithm) gives the bursts; weight = cost saved.
\end{itemize}
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
A simple growth rate reacts to noise; the entry cost means only sustained rises count. It is the standard ``burst detection'' of CiteSpace and science-of-science work (Kleinberg, 2002).\\[6pt]
\mhead{Used on}
\usedon{r:burst}, \usedon{r:syn}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: event study with placebo dates}
\slabel{m:event}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
Does writing on CBDCs jump right after a policy event, more than on an ordinary day?\\[4pt]
\mhead{How}
\begin{itemize}\setlength\itemsep{1pt}
\item Baseline rate $\lambda$ = texts per day over days $[-90,-15]$ before the event.
\item Expected over $[0,14]$: $E = 15\lambda$; ratio $= N/E$; $z = (N-E)/\sqrt{E}$.
\item \alert{Placebo}: the same $z$ on <<EV_NPL>> random dates; $p$ = share of random dates with a larger $z$.
\end{itemize}
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
The usual Poisson test assumes a stable rate; attention trends upward and comes in waves, so <<EV_P>>\% of random dates look ``significant''. Comparing with random dates gives the right yardstick (as in finance event studies, MacKinlay 1997).\\[3pt]
It is causal only in a narrow sense: timing of attention.\\[6pt]
\mhead{Used on}
\usedon{r:event}, \usedon{r:causal}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: do events move the tone?}
\slabel{m:tev}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
Same logic as the event study, applied to the tone of the texts rather than their number.\\[4pt]
\mhead{How}
\begin{itemize}\setlength\itemsep{1pt}
\item Texts dated to the day, with a tone score.
\item $\Delta = $ mean tone over $[t_0, t_0+180)$ minus mean tone over $[t_0-180, t_0)$.
\item Same $\Delta$ on 500 random dates; $p$ = share with $|\Delta|$ at least as large (two-sided).
\end{itemize}
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
Tone rises over time anyway; a before/after difference alone would credit every event with that trend. Random dates carry the same trend, so only a jump beyond it counts.\\[3pt]
180 days: enough texts on each side (100+).\\[6pt]
\mhead{Used on}
\usedon{r:tev}, \usedon{r:causal}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: difference-in-differences}
\slabel{m:did}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
If the 2022 sanctions pushed economists to discuss geopolitics, texts about Russia should change \emph{more} than the others after 2022.\\[4pt]
\mhead{How}
Linear probability model, robust SEs; $y$ = motivation stated, $T$ = text about Russia or in Russian:
\[ y_{it} = a_T + b_t + \sum_{k\neq 2021} \gamma_k\, T_i\,\mathbf{1}[t=k] + \varepsilon_{it} \]
$\gamma_k$ = gap in year $k$ relative to 2021. Pooled: one $\gamma$ for $t > 2021$.
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
A simple before/after mixes the sanctions with everything else that happened in 2022; the other texts serve as the control.\\[3pt]
Two checks: \textbf{pre-trends} ($\gamma_k \approx 0$ before 2022: only 2020 is usable) and \textbf{placebo outcomes} (other motivations should not move: some do). Hence ``consistent with'', not proof.\\[6pt]
\mhead{Used on}
\usedon{r:did}, \usedon{r:causal}.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Method: robustness}
\slabel{m:robust}
\begin{columns}[T]
\begin{column}{0.52\textwidth}\footnotesize
\mhead{Idea}
Re-run the headline results on samples that remove one possible source of error at a time.\\[4pt]
\mhead{How}
\begin{itemize}\setlength\itemsep{1pt}
\item \textbf{Main theme only}: the secondary theme label is noisy.
\item \textbf{English only}: language mix changes over time.
\item \textbf{Journal articles only}: working papers and chapters differ.
\item \textbf{Without the current year}: partial year.
\item \textbf{High-recall coding}: codebook + SBERT matches.
\item \textbf{All disciplines}: the economist filter itself.
\end{itemize}
\end{column}
\begin{column}{0.44\textwidth}\footnotesize
\mhead{Why this tool}
Each measure has a known weakness; a result that holds whichever way the sample is cut does not depend on it. It answers a referee's ``what if'' in one table.\\[6pt]
\mhead{Used on}
\usedon{r:robust}.
\end{column}
\end{columns}
\end{frame}

\section*{Backup}

\begin{frame}{Backup: validation of the motivation coding}
\footnotesize
Codebook tuned on 100 texts (development set), then evaluated on the <<VAL_N>> \alert{held-out} texts written by economists (of <<VAL_N_ALL>>); unit = first 1{,}500 characters of the introduction / abstract.\\[3pt]
\begin{tabular}{@{}lrrrrr@{}}\toprule
\textbf{Motivation} & \textbf{Read} & \textbf{Coded} & \textbf{Precision} & \textbf{Recall} & $\kappa$\\\midrule
<<VAL_ROWS>>
\midrule
<<VAL_ALL>>
\bottomrule\end{tabular}\\[3pt]
\scriptsize Read = texts where the reading found the motivation; Coded = texts flagged by the codebook. Files: \texttt{presentation/validation\_motivations\_dev.csv}, \texttt{\_test.csv}.
\end{frame}

\begin{frame}{Backup: validation of the stance coders}
\slabel{b:stance}
\footnotesize
Each coder against the stance read by hand (favourable / neutral / critical); thresholds and classifiers by 5-fold cross-validation.\\[4pt]
\begin{tabular}{@{}lrrr@{}}\toprule
\textbf{Coder} & \textbf{Accuracy} & $\kappa$ & $\kappa$\textbf{, all}\\\midrule
<<ST_ROWS>>
\bottomrule\end{tabular}\\[4pt]
Scores on the <<ST_N>> texts written by economists; last column: all <<ST_N_ALL>> hand-coded texts (including the press).\\[3pt]
\scriptsize Files: \texttt{presentation/validation\_stance.csv}, \texttt{validation\_predictions\_stance.csv}.
\end{frame}

\begin{frame}{Backup: validation of the tone measure}
\footnotesize
Sentences sampled per FinBERT label, read blind to the label; table: the <<TONE_N>> from economists' texts (30 per label).\\[4pt]
\begin{tabular}{@{}lrrrr@{}}\toprule
\textbf{FinBERT $\downarrow$ / reading $\rightarrow$} & \textbf{negative} & \textbf{neutral} & \textbf{positive} & \textbf{Precision}\\\midrule
<<TONE_XTAB>>
\bottomrule\end{tabular}\\[4pt]
<<TONE_VAL>>\\[3pt]
\scriptsize File: \texttt{presentation/validation\_tone.csv}.
\end{frame}

\begin{frame}{Backup: how the relevance score behaves}
\footnotesize
\begin{tabular}{@{}lrl@{}}\toprule
\textbf{Example} & \textbf{Score} & \textbf{Label}\\\midrule
<<EXAMPLE_ROWS>>
\bottomrule\end{tabular}
\end{frame}

\begin{frame}[fragile]{Backup: press, recursive bisection of time windows}
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
A news search returns at most $C=100$ items. A window holding more is cut in two until each holds fewer, down to one day: depth $\leq 5$ per month, $2L-1$ requests for $L$ windows. The press was collected, then left out of the economists' corpus.
\end{column}
\end{columns}
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
\item Requests spaced by $t_{k+1}-t_k \geq \Delta$.
\item On HTTP 429: back-off $w_j = \min(60j,\,900)$\,s.
\item Reference lists resolved into citations (for co-citation).
\end{itemize}
\end{column}
\end{columns}
\end{frame}

\begin{frame}[fragile]{Backup: full text and de-duplication}
\begin{columns}[T]
\begin{column}{0.46\textwidth}\small
\textbf{Full text}
\begin{itemize}
\item PDFs (PyMuPDF) and web pages (trafilatura).
\item Split into introduction / conclusion / section headings.
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
\small Merge: most precise date, longest abstract, union of keywords.
\end{column}
\end{columns}
\end{frame}

\begin{frame}{References}
\fontsize{6.2}{7.2}\selectfont
\begin{itemize}\setlength\itemsep{0pt}
\item Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. arXiv:1908.10063.
\item Aria, M., \& Cuccurullo, C. (2017). bibliometrix: An R-tool for comprehensive science mapping analysis. \emph{Journal of Informetrics}, 11(4).
\item Blondel, V. D., Guillaume, J.-L., Lambiotte, R., \& Lefebvre, E. (2008). Fast unfolding of communities in large networks. \emph{J.\ Stat.\ Mech.}
\item Callon, M., Courtial, J.-P., \& Laville, F. (1991). Co-word analysis as a tool for describing the network of interactions between basic and technological research. \emph{Scientometrics}, 22(1).
\item Campello, R., Moulavi, D., \& Sander, J. (2013). Density-based clustering based on hierarchical density estimates. \emph{PAKDD}.
\item Cobo, M. J., López-Herrera, A. G., Herrera-Viedma, E., \& Herrera, F. (2011). An approach for detecting, quantifying, and visualizing the evolution of a research field. \emph{Journal of Informetrics}, 5(1).
\item Cohen, J. (1960). A coefficient of agreement for nominal scales. \emph{Educational and Psychological Measurement}, 20(1).
\item Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
\item Kleinberg, J. (2002). Bursty and hierarchical structure in streams. \emph{KDD}.
\item MacKinlay, A. C. (1997). Event studies in economics and finance. \emph{Journal of Economic Literature}, 35(1).
\item McInnes, L., Healy, J., \& Melville, J. (2018). UMAP: Uniform manifold approximation and projection. arXiv:1802.03426.
\item Monroe, B. L., Colaresi, M. P., \& Quinn, K. M. (2008). Fightin' words. \emph{Political Analysis}, 16(4).
\item Reimers, N., \& Gurevych, I. (2019). Sentence-BERT. \emph{EMNLP}.
\item Roberts, M. E., Stewart, B. M., Tingley, D., et al.\ (2014). Structural topic models for open-ended survey responses. \emph{AJPS}, 58(4).
\item van Eck, N. J., \& Waltman, L. (2009). How to normalize cooccurrence data? \emph{JASIST}, 60(8).
\item van Eck, N. J., \& Waltman, L. (2010). Software survey: VOSviewer. \emph{Scientometrics}, 84(2).
\item Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. \emph{JASA}, 22(158).
\item Reviews cited: Bapat \& Sharma (2025); Fahad \& Bulut (2024); Hoang, Ngo \& Vu (2023); Kvedaravi\v{c}i\={u}t\.{e} \& \v{S}apkauskien\.{e} (2024); Paulovici (2026); Prodan et al.\ (2024): full records in the corpus.
\end{itemize}
\end{frame}
\end{document}
"""

if __name__ == "__main__":
    main()
