"""Bibliometric analyses of the corpus (texts with CBDC as main or secondary theme).

  co_citation      references cited together -> intellectual base (clusters = schools of thought)
  co_authorship    authors writing together -> research communities
  keyword_network  keywords used together -> research fronts, and their evolution over time
  distinctive      words over-used by academics / central bankers / press (log-odds, Monroe et al. 2008)
  thematic_map     Callon centrality and density of keyword clusters
  motivations      why study CBDCs: codebook of stated motivations
  event_study      attention around policy events, tested against placebo dates
  laws             growth, Lotka, Bradford

Networks follow VOSviewer's conventions: edge weight normalised by association strength
    s_ij = c_ij / (c_i * c_j)
(c_ij co-occurrences, c_i total occurrences), clusters by Louvain modularity maximisation.
"""
import math
import re
from collections import Counter
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

THEME = ("main theme", "secondary theme")
ORG = re.compile(r"\b(institute|university|bank|association|committee|group|council|ministry|inst)\b", re.I)
GENERIC_KW = re.compile(r"^(cbdcs?|central banks? digital currenc(y|ies)( \(cbdcs?\))?|digital currenc(y|ies)|"
                        r"central bank digital currency \(cbdc\))$")
STOP = set("""a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her
here hers him his how i if in into is it its itself just me more most my no nor not now of off on once only or
other our out over own same she should so some such than that the their them then there these they this those
through to too under until up very was we were what when where which while who whom why will with would you your
new one two may might must us use used using paper study also however based within well would per via towards toward
whether among across without since yet many much first second third et al thus today year years like let
think thank ladies gentlemen also said says would""".split())
# the shared topic itself: removed so the comparison shows how each group frames it
TOPIC = {"cbdc", "cbdcs", "digital", "currency", "currencies", "central", "bank", "banks"}
DISCIPLINES = {"economic", "economics", "business", "computer science", "political science", "law", "finance",
               "sociology", "mathematics", "philosophy", "engineering", "geography", "psychology", "history",
               "art", "medicine", "environmental science", "physics", "biology", "chemistry", "accounting",
               "management", "marketing", "economy", "monetary economics", "financial system", "computer security"}


# ----------------------------------------------------------------- who counts as an economist

ECON_FIELD = "Economics, Econometrics and Finance"
ECON_VENUE = re.compile(r"econom|financ|monetar|money|bank|credit|repec|nber|cepr|working\s+paper", re.I)


def economists(d, fields):
    """Texts written by economists: research classified in economics or finance by OpenAlex (or, without an
    OpenAlex record, published in an economics venue) or with an author at a central bank, the IMF or the BIS;
    central bank speeches; documents of central banks and international financial organisations. Press is excluded."""
    f = d[["record_id"]].merge(fields, on="record_id", how="left")
    f.index = d.index
    tagged = f.field.notna()
    research = (f.field.eq(ECON_FIELD) | f.institutions.fillna("").str.contains(CB_AFFIL)
                | d.author_affiliations.fillna("").str.contains(CB_AFFIL)
                | (~tagged & d.venue.fillna("").str.contains(ECON_VENUE)))
    return (((d.source_type == "academic") & research) | (d.source_type == "speech")
            | ((d.source_type == "official") & d.issuer_type.isin(["central bank", "international organisation"])))


def load(path):
    d = pd.read_csv(path, low_memory=False, usecols=lambda c: c != "full_text")
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    return d


# ----------------------------------------------------------------- generic network helpers

def _network(docs_items, top, min_edge=1, min_component=2):
    """docs_items: list of item lists (one per document). Keeps the `top` most frequent items."""
    occ = Counter(i for items in docs_items for i in set(items))
    keep = {i for i, _ in occ.most_common(top)}
    co = Counter()
    for items in docs_items:
        for a, b in combinations(sorted(set(items) & keep), 2):
            co[a, b] += 1
    g = nx.Graph()
    for i in keep:
        g.add_node(i, occ=occ[i])
    for (a, b), c in co.items():
        if c >= min_edge:
            g.add_edge(a, b, count=c, strength=c / (occ[a] * occ[b]))
    g.remove_nodes_from([n for c in list(nx.connected_components(g)) if len(c) < min_component for n in c])
    comms = nx.community.louvain_communities(g, weight="strength", seed=7) if len(g) else []
    comms = sorted(comms, key=lambda c: -sum(g.nodes[n]["occ"] for n in c))
    for k, c in enumerate(comms):
        for n in c:
            g.nodes[n]["cluster"] = k
    q = nx.community.modularity(g, comms, weight="strength") if comms else 0
    return g, comms, q


def layout(g, w=11.0, h=5.6):
    """Force-directed (Fruchterman-Reingold) positions scaled to a w x h cm box.

    A fragmented network (e.g. co-authorship) is laid out component by component, each in
    its own cell of a grid, so small teams do not drift to the corners.
    """
    if not len(g):
        return {}
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    if len(comps) == 1:
        cells = [(g, 0, 0, w, h)]
    else:
        cols = math.ceil(math.sqrt(len(comps) * w / h))
        rows = math.ceil(len(comps) / cols)
        cw, ch = w / cols, h / rows
        cells = [(g.subgraph(c), (i % cols) * cw, (rows - 1 - i // cols) * ch, cw * 0.8, ch * 0.75)
                 for i, c in enumerate(comps)]
    out = {}
    for sub, x0, y0, cw, ch in cells:
        pos = nx.spring_layout(sub, weight="strength", seed=3, k=1.6 / math.sqrt(len(sub)), iterations=300)
        if len(sub) > 5:  # pull weakly linked outliers back to the edge of the cloud
            c = np.median(np.array(list(pos.values())), axis=0)
            r = {n: np.linalg.norm(p - c) for n, p in pos.items()}
            r_max = 1.15 * np.percentile(list(r.values()), 85)
            pos = {n: c + (p - c) * min(1, r_max / r[n]) if r[n] else p for n, p in pos.items()}
        xs, ys = zip(*pos.values())
        sx = lambda v: (v - min(xs)) / ((max(xs) - min(xs)) or 1) * cw
        sy = lambda v: (v - min(ys)) / ((max(ys) - min(ys)) or 1) * ch
        out.update({n: (x0 + sx(x), y0 + sy(y)) for n, (x, y) in pos.items()})
    return out


# ----------------------------------------------------------------- 1. co-citation

def _ref_label(ref):
    """'Bordo M.D., Levin A.T., Central Bank Digital ..., (2017)' -> 'Bordo (2017)'."""
    first = ref.split(",")[0].strip()
    surname = first.split()[0] if first else "?"
    y = re.search(r"\((\d{4})\)\s*$", ref)
    return f"{surname} ({y.group(1) if y else 'n.d.'})"


def _ref_title(ref):
    """First comma-separated field of 4+ words = the title (authors come first, as 'Hall R.E.')."""
    return next((f.strip() for f in ref.split(", ") if len(f.split()) >= 4), ref)


# references whose metadata is wrong in OpenAlex (e.g. the Bitcoin white paper credited to "Squarepants S.")
CANONICAL = {"bitcoinapeertopeerelectroniccashsystem": "Nakamoto (2008)"}


def co_citation(d, top=40):
    """References are identified by normalised title, so variants of the same work are merged."""
    a = d[(d.source_type == "academic") & d.references.notna()]
    norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
    variants = {}
    refs = []
    for s in a.references:
        doc = []
        for r in s.split("; "):
            if len(r) > 15:
                k = norm(_ref_title(r.strip()))
                variants.setdefault(k, Counter())[r.strip()] += 1
                doc.append(k)
        refs.append(doc)
    rep = {k: c.most_common(1)[0][0] for k, c in variants.items()}
    g, comms, q = _network(refs, top, min_edge=2, min_component=5)
    g.remove_nodes_from([n for n in list(g) if g.degree(n) < 3])  # loosely attached references blur the map
    main = max(nx.connected_components(g), key=len) if len(g) else set()  # one intellectual base
    g = g.subgraph(main).copy()
    comms = [c & main for c in comms if c & main]
    labels = {n: CANONICAL.get(n) or _ref_label(rep[n]) for n in g}
    dup = Counter(labels.values())
    seen = Counter()
    for n in sorted(g, key=lambda n: -g.nodes[n]["occ"]):  # Auer (2020a), Auer (2020b)
        if dup[labels[n]] > 1:
            seen[labels[n]] += 1
            labels[n] = labels[n][:-1] + "abcdef"[seen[labels[n]] - 1] + ")"
    nx.set_node_attributes(g, labels, "label")
    cited = Counter(r for rs in refs for r in set(rs))
    top_cited = [(CANONICAL.get(k) or _ref_label(rep[k]), c) for k, c in cited.most_common(10)]
    return {"graph": g, "clusters": comms, "modularity": q, "n_docs": len(refs),
            "n_refs": len(cited), "top": top_cited}


# ----------------------------------------------------------------- 2. co-authorship

def _author_key(name):
    """Full name without punctuation / case: 'Peterson K. Ozili' == 'Peterson K Ozili'.
    (Surname + initial would merge distinct people, e.g. every 'Wang Y.')"""
    return " ".join(re.sub(r"[.,]", " ", name).lower().split())


def co_authorship(d, top=45):
    a = d[(d.source_type == "academic") & d.authors.notna()]
    names = Counter()
    teams = []
    for s in a.authors:
        t = [x for x in s.split("; ") if x]
        names.update((_author_key(x), x) for x in t)
        teams.append([_author_key(x) for x in t])
    teams = [[x for x in t if not ORG.search(x)] for t in teams if len(t) <= 25]  # no consortia / institutions
    display = {}
    for (k, x), c in names.most_common():
        display.setdefault(k, x)
    g, comms, q = _network(teams, top, min_edge=1, min_component=3)
    nx.set_node_attributes(g, {n: display[n] for n in g}, "label")
    prod = Counter(display[x] for t in teams for x in set(t))
    countries = a.country.dropna().str.split("; ")
    return {"graph": g, "clusters": comms, "modularity": q, "top": prod.most_common(10),
            "n_authors": len(prod), "coauthored": float(np.mean([len(t) > 1 for t in teams])),
            "mean_team": float(np.mean([len(t) for t in teams])),
            "intl": float(countries.str.len().gt(1).mean()),
            "countries": Counter(c for cs in countries for c in set(cs)).most_common(10)}


# ----------------------------------------------------------------- 3. keywords

# database keywords attached by mistake ("Libras" sign language for papers on Facebook's Libra)
MISTAGS = {"brazilian sign language (libras)"}


def _keywords(row):
    kws = []
    for col in ("keywords_author", "keywords_indexed"):
        if isinstance(row.get(col), str):
            kws += [k.strip().lower() for k in row[col].split(";")]
    out = []
    for k in kws:
        k = re.sub(r"\s*\((cbdcs?|dlt)\)$", "", k)
        if " " not in k and not k.endswith(("ss", "us", "is")):
            k = re.sub(r"(?<=[a-z]{3})s$", "", k)  # plurals of single words: stablecoins -> stablecoin
        k = re.sub(r"currencies$", "currency", k)
        if k and not GENERIC_KW.match(k) and k not in DISCIPLINES and k not in MISTAGS and len(k) < 60:
            out.append(k)
    return list(dict.fromkeys(out))


PERIODS = [("2014--2019", 0, 2019), ("2020--2022", 2020, 2022), ("2023--2026", 2023, 2100)]


def keyword_network(d, top=40):
    a = d[d.source_type == "academic"].copy()
    a["kw"] = [_keywords(r) for r in a.to_dict("records")]
    a = a[a.kw.str.len() > 0]
    g, comms, q = _network(list(a.kw), top, min_edge=2)
    nx.set_node_attributes(g, {n: n for n in g}, "label")
    # evolution: share of each period's documents carrying the keyword
    occ = Counter(k for ks in a.kw for k in ks)
    words = [k for k, _ in occ.most_common(16)]
    shares = {}
    for lbl, y0, y1 in PERIODS:
        p = a[(a.year >= y0) & (a.year <= y1)]
        shares[lbl] = {w: 100 * p.kw.apply(lambda ks: w in ks).mean() if len(p) else 0 for w in words}
    evo = pd.DataFrame(shares).loc[words]
    evo["change"] = evo[PERIODS[-1][0]] - evo[PERIODS[1][0]]
    return {"graph": g, "clusters": comms, "modularity": q, "evolution": evo,
            "n_docs": len(a), "period_n": {l: int(((a.year >= y0) & (a.year <= y1)).sum()) for l, y0, y1 in PERIODS}}


# ----------------------------------------------------------------- 4. who says what

def _text(row, root):
    if row["source_type"] == "academic":
        return f"{row.get('title') or ''} {row.get('abstract') or ''}"
    if row["source_type"] == "press":
        return str(row.get("title") or "")
    p = row.get("full_text_path")
    body = (root / p).read_text(encoding="utf-8")[:6000] if isinstance(p, str) and (root / p).exists() else ""
    return f"{row.get('title') or ''} {body}"


def _tokens(text):
    return [w for w in re.findall(r"[a-z][a-z\-]{2,}", text.lower())
            if w not in STOP and w not in TOPIC and not w.endswith("-")]


GROUPS = {"academic": "Research economists", "speech": "Central bankers"}


def distinctive(d, root, k=10, a0=500.0):
    """Monroe, Colaresi & Quinn (2008): log-odds ratio with informative Dirichlet prior.

    For word w, group i vs the other groups j:
      delta_w = log[(y_iw + a_w) / (n_i + a0 - y_iw - a_w)] - log[(y_jw + a_w) / (n_j + a0 - y_jw - a_w)]
      var     = 1/(y_iw + a_w) + 1/(y_jw + a_w),   z_w = delta_w / sqrt(var)
    with a_w = a0 * (corpus frequency of w). Shrinks rare words towards zero.
    """
    e = d[(d.language == "en") & d.source_type.isin(GROUPS)]
    counts = {g: Counter() for g in GROUPS}
    for r in e.to_dict("records"):
        counts[r["source_type"]].update(_tokens(_text(r, root)))
    tot = sum(counts.values(), Counter())
    N = sum(tot.values())
    vocab = [w for w, c in tot.items() if c >= 20]
    out = {}
    for g in GROUPS:
        yi, rest = counts[g], tot - counts[g]
        ni, nj = sum(yi.values()), sum(rest.values())
        z = {}
        for w in vocab:
            aw = a0 * tot[w] / N
            d_ = (math.log((yi[w] + aw) / (ni + a0 - yi[w] - aw))
                  - math.log((rest[w] + aw) / (nj + a0 - rest[w] - aw)))
            z[w] = d_ / math.sqrt(1 / (yi[w] + aw) + 1 / (rest[w] + aw))
        out[g] = sorted(z.items(), key=lambda x: -x[1])[:k]
    return out, {g: sum(c.values()) for g, c in counts.items()}


# ----------------------------------------------------------------- 5. thematic map (Callon)

def thematic_map(d, top=80):
    """Clusters of keywords placed by Callon centrality and density (Cobo et al., 2011).

    With the equivalence index e_ij = c_ij^2 / (c_i c_j):
      centrality(C) = 10 * sum of e_ij over pairs with i in C, j outside C   (links to other themes)
      density(C)    = 100 * sum of e_ij over pairs inside C / |C|            (internal cohesion)
    Quadrants split at the medians: motor (high, high), niche (low centrality, high density),
    emerging or declining (low, low), basic and transversal (high centrality, low density).
    """
    a = d[d.source_type == "academic"].copy()
    docs = [k for k in (_keywords(r) for r in a.to_dict("records")) if k]
    g, comms, _ = _network(docs, top, min_edge=2)
    e = {(i, j): dd["count"] ** 2 / (g.nodes[i]["occ"] * g.nodes[j]["occ"]) for i, j, dd in g.edges(data=True)}
    themes = []
    for c in comms:
        if len(c) < 3:
            continue
        inside = sum(v for (i, j), v in e.items() if i in c and j in c)
        outside = sum(v for (i, j), v in e.items() if (i in c) != (j in c))
        top_kw = sorted(c, key=lambda n: -g.nodes[n]["occ"])
        themes.append({"label": ", ".join(top_kw[:2]), "members": top_kw, "size": sum(g.nodes[n]["occ"] for n in c),
                       "centrality": 10 * outside, "density": 100 * inside / len(c)})
    return themes


# ----------------------------------------------------------------- 6. why study CBDCs? (motivations)

MOTIVES = {
    "Decline of cash": r"declin\w*\s+(?:in\s+|of\s+)?(?:the\s+)?(?:use\s+of\s+)?(?:physical\s+)?cash|cashless|cash\s+usage|"
                       r"use\s+of\s+cash|less[- ]cash|(?:rely|reliance)\s+(?:less\s+)?on\s+cash|reduc\w+\s+cash|cash\s+reliance|"
                       r"decrease\s+in\s+cash|disappear\w*\s+of\s+cash|baisse\s+de\s+l.usage\s+des\s+esp[eè]ces|bargeld|"
                       r"налич\w+\s+(?:деньг|расчет)\w*\s+сокращ|сокращени\w+\s+налич",
    "Financial inclusion": r"financial\s+inclusion|unbanked|underbanked|under-banked|access\s+to\s+(?:formal\s+)?(?:financial\s+services|finance|banking)|"
                           r"financial\s+accessib|inclusion\s+financi[eè]re|inclusi[oó]n\s+financiera|inclus[aã]o\s+financeira|"
                           r"финансов\w+\s+(?:доступност|инклюзи)|фінансов\w+\s+інклюзі|доступн\w+\s+фінансов|доступност\w+\s+финансов",
    "Monetary sovereignty": r"monetary\s+sovereignty|souverainet[ée]\s+mon[ée]taire|soberan[ií]a\s+monet[aá]ria|"
                            r"global\s+stablecoin|\blibra\b|\bdiem\b|big\s*tech|private\s+(?:digital\s+)?(?:money|currenc|crypto)|"
                            r"dollari[sz]ation|currency\s+substitution|"
                            r"(?:threat|competition|challenge)\w*\s+(?:posed\s+)?(?:by|from|of)\s+(?:private|crypto|stablecoin)|"
                            r"(?:monetary|currency)\s+(?:control|monopoly)|монетарн\w+\s+суверенитет|денежн\w+\s+суверенитет|"
                            r"частн\w+\s+(?:деньг|цифров\w+\s+валют)|альтернатив\w+\s+криптовалют",
    "Payment efficiency": r"payment\w*\s+(?:system\w*\s+)?efficien|efficien\w+\s+(?:of\s+(?:the\s+)?(?:digital\s+)?)?(?:payment|settlement|transaction)|"
                          r"cost\w*\s+of\s+(?:payment|transaction)|transaction\s+costs?|(?:faster|cheaper|instant)\s+(?:payment|transfer|settlement)|"
                          r"(?:reduce|lower)\w*\s+(?:the\s+)?(?:transaction\s+)?costs|speed\s+(?:of|up)\s+(?:transaction|settlement|payment)|"
                          r"interoperab|payment\s+innovation|moderni[sz]\w+\s+(?:the\s+)?payment|resilien\w+\s+of\s+(?:the\s+)?payment|"
                          r"efici[eê]ncia|эффективност\w+\s+(?:платеж|расчет|финансов)|издерж|собівартість\s+транзакц|прискор\w+\s+розрахун",
    "Cross-border payments": r"cross[- ]border|remittance|mbridge|international\s+(?:payment|settlement|trade\s+settlement)|"
                             r"transfrontali|трансграничн|транскордон|международн\w+\s+(?:расчет|платеж)",
    "Monetary policy": r"monetary\s+policy|policy\s+transmission|interest[- ]bearing|negative\s+(?:interest\s+)?rate|"
                       r"(?:zero|effective)\s+lower\s+bound|remunerat|money\s+supply|monetary\s+control|"
                       r"politique\s+mon[ée]taire|pol[ií]tica\s+monet[aá]ria|geldpolitik|денежно-кредитн|грошово-кредитн|"
                       r"денежн\w+\s+масс|chính\s+sách\s+tiền\s+tệ",
    "Financial stability": r"financial\s+(?:stability|instability)|stability\s+of\s+the\s+financial|bank\s+runs?\b|disintermediat|"
                           r"deposit\s+(?:outflow|substitution|flight)|deposit-taking|bank\s+(?:funding|lending)|banking\s+(?:system|sector)|"
                           r"(?:financial|bank)\w*\s+intermediation|systemic\s+(?:risk|threat)|disrupt\w*\s+(?:the\s+way\s+)?(?:traditional\s+)?banks|"
                           r"stabilit[ée]\s+financi[eè]re|estabilidade\s+financeira|intermedia[çc][aã]o\s+banc|"
                           r"финансов\w+\s+(?:стабильност|устойчивост)|фінансов\w+\s+стабільн|банковск\w+\s+систем|hệ\s+thống\s+ngân\s+hàng",
    "Privacy and surveillance": r"privacy|anonymi|surveillance|data\s+protection|confidentialit|vie\s+priv[ée]e|datenschutz|"
                                r"privacidad|privacidade|приватност|конфиденциальност|защит\w+\s+(?:информац|персональн)|"
                                r"quyền\s+riêng\s+tư|techno-authoritarian|government\s+overreach",
    "Geopolitics and sanctions": r"geopolit|sanction|de-?dollari|hegemon|reserve\s+currenc|internationali[sz]ation\s+of|"
                                 r"exorbitant\s+privilege|\bbrics\b|power\s+competition|international\s+(?:monetary|financial)\s+order|"
                                 r"international\s+(?:role|competitiveness)\s+of|санкц|геополит|интернационализац",
    "Technology (DLT, programmability)": r"blockchain|distributed\s+ledger|\bdlt\b|smart\s+contract|programmab|tokeni[sz]|"
                                         r"cryptograph|decentrali[sz]|\bledger\b|блокчейн|распределенн\w+\s+реестр|"
                                         r"прорывн\w+\s+(?:информационн|цифров)\w*\s+технолог",
}
MOTIVE_RE = {k: re.compile(v, re.I) for k, v in MOTIVES.items()}


def _motive_text(row, root):
    """Where motivations are stated: the introduction, else the abstract, else the opening of the full text."""
    for col in ("introduction", "abstract"):
        if isinstance(row.get(col), str) and len(row[col]) > 200:
            return row[col][:6000]
    p = row.get("full_text_path")
    return (root / p).read_text(encoding="utf-8")[:6000] if isinstance(p, str) and (root / p).exists() else ""


def motivations(d, root):
    """Share of texts whose opening mentions each motivation, by period and by actor.

    m_k(t) = 1[regex_k matches the opening of text t];   share_k(G) = (1/|G|) sum_{t in G} m_k(t)
    Press is left out: headlines are too short to state a motivation.
    """
    rows = []
    for r in d[d.source_type.isin(["academic", "speech", "official"])].to_dict("records"):
        text = _motive_text(r, root)
        if len(text) < 200:
            continue
        rows.append({"record_id": r["record_id"], "source_type": r["source_type"], "year": r["year"],
                     "title": r.get("title"), "text": text[:1500],
                     **{k: bool(rx.search(text)) for k, rx in MOTIVE_RE.items()}})
    m = pd.DataFrame(rows)
    ks = list(MOTIVES)
    overall = m[ks].mean().mul(100).sort_values(ascending=False)
    by_actor = m.groupby(m.source_type.replace({"official": "speech"}))[ks].mean().mul(100).T
    by_period = pd.DataFrame({lbl: m[(m.year >= y0) & (m.year <= y1)][ks].mean().mul(100)
                              for lbl, y0, y1 in PERIODS})
    return {"table": m, "overall": overall, "by_actor": by_actor, "by_period": by_period,
            "n": len(m), "n_any": float(m[ks].any(axis=1).mean())}


# ----------------------------------------------------------------- 7. event study

EVENTS = [
    ("2019-06-18", "Facebook announces Libra"),
    ("2020-10-02", "ECB digital euro report"),
    ("2020-10-20", "Bahamas launches Sand Dollar"),
    ("2021-07-14", "ECB investigation phase"),
    ("2021-10-25", "Nigeria launches eNaira"),
    ("2022-01-20", "Fed CBDC discussion paper"),
    ("2022-03-09", "US order on digital assets"),
    ("2022-11-01", "India e-rupee pilot"),
    ("2023-02-07", "UK digital pound consultation"),
    ("2023-06-28", "EU digital euro bill"),
    ("2023-10-18", "ECB preparation phase"),
    ("2025-01-23", "US order banning a US CBDC"),
]


def event_study(d, pre=(-90, -15), post=(0, 14)):
    """Abnormal attention around policy events (texts dated to the day, all types).

    Baseline daily rate  lambda = N[pre] / |pre|;  expected after the event  E = lambda * |post|;
    ratio = N[post] / E;  Poisson z = (N[post] - E) / sqrt(E).
    """
    dd = d[d.date_precision.fillna("").str.startswith("day") & ~d.date_precision.fillna("").str.contains("Jan 1")]
    days = pd.to_datetime(dd.publication_date, errors="coerce").dropna().dt.normalize()
    counts = days.value_counts()
    out = []
    for date_, label in EVENTS:
        t0 = pd.Timestamp(date_)
        n_in = lambda a, b: int(counts[(counts.index >= t0 + pd.Timedelta(days=a)) & (counts.index <= t0 + pd.Timedelta(days=b))].sum())
        base, after = n_in(*pre), n_in(*post)
        lam = base / (pre[1] - pre[0] + 1)
        exp_ = lam * (post[1] - post[0] + 1)
        out.append({"date": date_, "event": label, "after": after, "expected": exp_,
                    "ratio": after / exp_ if exp_ else float("nan"),
                    "z": (after - exp_) / math.sqrt(exp_) if exp_ else float("nan")})
    weekly = days.dt.to_period("W").value_counts().sort_index()
    return pd.DataFrame(out), weekly


def placebo(d, n=500, start="2019-06-01", end="2025-06-01", pre=(-90, -15), post=(0, 14), seed=7):
    """Same test on random dates: the share with z > 1.96 is the false-positive rate of the design
    (attention is overdispersed and trending, so it exceeds the nominal 2.5%)."""
    dd = d[d.date_precision.fillna("").str.startswith("day") & ~d.date_precision.fillna("").str.contains("Jan 1")]
    counts = pd.to_datetime(dd.publication_date, errors="coerce").dropna().dt.normalize().value_counts()
    daily = counts.reindex(pd.date_range(counts.index.min(), counts.index.max()), fill_value=0)
    cum = daily.cumsum()
    span = lambda t, a, b: cum[t + pd.Timedelta(days=b)] - cum[t + pd.Timedelta(days=a - 1)]
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end)
    z = []
    for t in days[rng.choice(len(days), n, replace=False)]:
        lam = span(t, *pre) / (pre[1] - pre[0] + 1)
        e = lam * (post[1] - post[0] + 1)
        if e > 0:
            z.append((span(t, *post) - e) / math.sqrt(e))
    z = np.array(z)
    return {"n": len(z), "share_sig": float((z > 1.96).mean()), "z95": float(np.percentile(z, 95)), "z": z}


def tone_events(doc, window=180, n=500, start="2019-06-01", end="2025-06-01", seed=7):
    """Do events move tone? doc: one row per text with `day` and `tone`.

    For event date t0:  Delta = mean tone of texts in [t0, t0 + w)  -  mean tone in [t0 - w, t0);
    p = share of random dates with |Delta| at least as large (two-sided placebo test)."""
    doc = doc.dropna(subset=["day", "tone"]).sort_values("day")

    def delta(t0):
        w = pd.Timedelta(days=window)
        b = doc[(doc.day >= t0 - w) & (doc.day < t0)].tone
        a = doc[(doc.day >= t0) & (doc.day < t0 + w)].tone
        return a.mean() - b.mean(), len(b), len(a)
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end)
    pl = np.abs(np.array([delta(t)[0] for t in days[rng.choice(len(days), n, replace=False)]]))
    out = []
    for date_, label in EVENTS:
        dl, nb, na = delta(pd.Timestamp(date_))
        out.append({"date": date_, "event": label, "before": nb, "after": na, "delta": dl,
                    "p": float((pl >= abs(dl)).mean())})
    return pd.DataFrame(out), float(np.percentile(pl, 95))


# texts about Russia: exposed to the 2022 sanctions shock
RUSSIA = re.compile(r"russia|ruble|rouble|\bcbr\b|росси|рубл|банк\w*\s+росси", re.I)


def did(frame, treated, outcome, ref=2021, years=(2018, 2026)):
    """Event-study difference-in-differences (linear probability model, robust SEs):

        y_it = a_g + b_t + sum_{k != ref} g_k * 1[treated_i] * 1[t = k] + e_it

    g_k for k < ref test parallel pre-trends; the pooled effect replaces the yearly interactions by
    1[treated] * 1[t > ref]. Returns {"by_year": [(year, g, lo, hi)], "pooled": (g, lo, hi), "n_treated", "n"}."""
    import statsmodels.api as sm
    f = frame[(frame.year >= years[0]) & (frame.year <= years[1])].copy()
    f["tr"] = treated.reindex(f.index).fillna(False).astype(float)
    y = f[outcome].astype(float).values
    yrs = sorted(int(x) for x in f.year.unique())
    base = {"const": np.ones(len(f)), "treated": f.tr.values}
    for t in yrs:
        if t != ref:
            base[f"y{t}"] = (f.year == t).astype(float).values
    X = pd.DataFrame(base)
    for t in yrs:
        if t != ref:
            X[f"tr_y{t}"] = f.tr.values * (f.year == t).values
    m = sm.OLS(y, X).fit(cov_type="HC1")
    ci = m.conf_int()
    by_year = [(t, 0.0, 0.0, 0.0) if t == ref else
               (t, 100 * m.params[f"tr_y{t}"], 100 * ci.loc[f"tr_y{t}", 0], 100 * ci.loc[f"tr_y{t}", 1]) for t in yrs]
    P = X[["const", "treated"] + [c for c in X if c.startswith("y")]].copy()
    P["post"] = f.tr.values * (f.year > ref).values
    mp = sm.OLS(y, P).fit(cov_type="HC1")
    pre = [r for r in by_year if r[0] < ref]
    return {"by_year": by_year, "pooled": (100 * mp.params["post"], 100 * mp.conf_int().loc["post", 0],
                                           100 * mp.conf_int().loc["post", 1]),
            "pre_ok": all(lo <= 0 <= hi for _, _, lo, hi in pre), "n_treated": int(f.tr.sum()), "n": len(f)}


# ----------------------------------------------------------------- 8. bibliometric laws

def laws(d, y0=2017, y1=2025):
    """Standard performance indicators of a field (as reported by bibliometrix).

    Growth   CAGR = (N_y1 / N_y0)^(1 / (y1 - y0)) - 1 on academic output
    Lotka    share of authors with n papers f(n) = C / n^beta, beta by least squares on log f(n), n = 1..10
             (Lotka's classical value: beta = 2)
    Bradford sources ranked by output, cut into three zones with one third of the papers each;
             the core is zone 1 (Bradford: zone sizes grow as 1 : k : k^2)
    """
    a = d[d.source_type == "academic"]
    per_year = a.year.value_counts()
    cagr = (per_year.get(y1, 0) / per_year.get(y0, 1)) ** (1 / (y1 - y0)) - 1
    prod = Counter(_author_key(x) for s in a.authors.dropna() for x in set(s.split("; ")) if x and not ORG.search(x))
    f = Counter(prod.values())
    n = np.arange(1, 11)
    share = np.array([f.get(k, 0) for k in n]) / len(prod)
    ok = share > 0
    beta = -np.polyfit(np.log(n[ok]), np.log(share[ok]), 1)[0]
    venues = a.venue.dropna().value_counts()
    cum = venues.cumsum() / venues.sum()
    zones = [int((cum <= 1 / 3).sum()) + 1, int((cum <= 2 / 3).sum()) + 1]
    zones = [zones[0], zones[1] - zones[0], len(venues) - zones[1]]
    return {"cagr": cagr, "y0": y0, "y1": y1, "beta": float(beta), "single": float(share[0]),
            "n_authors": len(prod), "bradford": zones, "n_venues": len(venues),
            "core": list(venues.index[:zones[0]])}


# ----------------------------------------------------------------- 9. bursts (Kleinberg)

def bursts(d, top=300, s=2.0, gamma=1.0, k=12):
    """Kleinberg (2002) burst detection, batched two-state version (as in CiteSpace), on keywords.

    Periods t = quarters; d_t papers, r_t of them carry keyword w; base rate p0 = R / D, burst rate p1 = s p0.
    Cost of state i in period t: sigma_i = -ln[ C(d_t, r_t) p_i^r_t (1 - p_i)^(d_t - r_t) ];
    entering the burst state costs gamma ln(T), leaving it is free; the cheapest state sequence (Viterbi)
    gives the burst periods; weight = sum over burst periods of (sigma_0 - sigma_1).
    """
    a = d[(d.source_type == "academic") & d.publication_date.notna()].copy()
    a["kw"] = [_keywords(r) for r in a.to_dict("records")]
    a["q"] = pd.to_datetime(a.publication_date, errors="coerce").dt.to_period("Q")
    a = a[a.q.notna() & (a.q >= pd.Period("2016Q1", "Q")) & (a.q < pd.Timestamp.today().to_period("Q"))]
    qs = pd.period_range(a.q.min(), a.q.max(), freq="Q")
    dt = a.q.value_counts().reindex(qs, fill_value=0).values
    occ = Counter(w for ks in a.kw for w in ks)
    T = len(qs)
    out = []
    for w, _ in occ.most_common(top):
        r = a[a.kw.apply(lambda ks: w in ks)].q.value_counts().reindex(qs, fill_value=0).values
        p0 = r.sum() / dt.sum()
        p1 = min(s * p0, 0.9999)

        def sigma(p):
            return np.array([-(math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
                               + x * math.log(p) + (n - x) * math.log(1 - p)) if n else 0.0 for x, n in zip(r, dt)])
        c0, c1 = sigma(p0), sigma(p1)
        up = gamma * math.log(T)
        cost, back = np.array([c0[0], c1[0] + up]), []
        for t in range(1, T):
            stay0, from1 = cost[0], cost[1]
            n0 = min(stay0, from1) + c0[t]
            n1 = min(cost[1], cost[0] + up) + c1[t]
            back.append((0 if stay0 <= from1 else 1, 1 if cost[1] <= cost[0] + up else 0))
            cost = np.array([n0, n1])
        state = [int(np.argmin(cost))]
        for b in reversed(back):
            state.append(b[state[-1]])
        state = state[::-1]
        seg = [t for t in range(T) if state[t]]
        if not seg:
            continue
        runs, cur = [], [seg[0]]
        for t in seg[1:]:
            if t == cur[-1] + 1:
                cur.append(t)
            else:
                runs.append(cur)
                cur = [t]
        runs.append(cur)
        for run in runs:
            out.append({"keyword": w, "start": str(qs[run[0]]), "end": str(qs[run[-1]]),
                        "weight": float(sum(c0[t] - c1[t] for t in run)), "n": int(occ[w])})
    b = pd.DataFrame(out)
    return b.sort_values("weight", ascending=False).drop_duplicates("keyword").head(k).sort_values("start") if len(b) else b


# ----------------------------------------------------------------- 10. who: central bank authors

CB_AFFIL = re.compile(
    r"central\s+bank|reserve\s+bank|federal\s+reserve|bundesbank|banque\s+de\s+france|banca\s+d.italia|banco\s+de\s+espa|"
    r"riksbank|norges\s+bank|national\s+bank\s+of|people.s\s+bank\s+of\s+china|banco\s+central|bank\s+for\s+international\s+settlements|"
    r"international\s+monetary\s+fund|monetary\s+authority|bank\s+of\s+(?:england|canada|japan|korea|israel|italy|finland|"
    r"thailand|russia|france|spain|greece|portugal|lithuania|estonia|latvia|ghana|jamaica|mexico)|"
    r"de\s+nederlandsche\s+bank|oesterreichische\s+nationalbank|swiss\s+national\s+bank|world\s+bank", re.I)


def central_bank_authors(d):
    """Academic papers with at least one author at a central bank, the BIS, the IMF or the World Bank."""
    a = d[(d.source_type == "academic") & d.author_affiliations.notna()].copy()
    a["cb"] = a.author_affiliations.str.contains(CB_AFFIL)
    cites = a.cited_by_count.fillna(0)
    return {"share": float(a.cb.mean()), "cite_share": float(cites[a.cb].sum() / max(cites.sum(), 1)),
            "by_period": {lbl: float(a[(a.year >= y0) & (a.year <= y1)].cb.mean()) for lbl, y0, y1 in PERIODS},
            "n": len(a)}


# ----------------------------------------------------------------- 11. covariate-adjusted changes

def adjusted(frame, outcome, covars=("actor", "english")):
    """Logit  P(y = 1) = Lambda(a + b_period + c_actor + e * english); returns average marginal effects (in points)
    of the last period vs the first and of central banks vs academics, with 95% CIs (robust SEs, coefficients
    simulated from their estimated distribution). The prevalence part of a structural topic model
    (Roberts et al. 2014), applied to any coded outcome."""
    import statsmodels.api as sm
    f = frame.dropna(subset=["period"]).copy()
    y = f[outcome].astype(int).values
    if y.min() == y.max():
        return {}
    first, last = PERIODS[0][0], PERIODS[-1][0]
    levels = {"period": [p for p, _, _ in PERIODS if p != first]}
    if "actor" in covars and f.actor.nunique() > 1:
        levels["actor"] = [a for a in ("central bank", "press") if (f.actor == a).any()]

    def design(g):
        cols = {"const": np.ones(len(g))}
        for var, lv in levels.items():
            for v_ in lv:
                cols[f"{var}={v_}"] = (g[var] == v_).astype(float).values
        if "english" in covars and f.english.nunique() > 1:
            cols["english"] = g.english.astype(float).values
        return pd.DataFrame(cols)
    try:
        m = sm.Logit(y, design(f)).fit(disp=0, cov_type="HC1", maxiter=200)
    except Exception:
        return {}
    sims = np.random.default_rng(1).multivariate_normal(m.params.values, m.cov_params().values, 400)
    sig = lambda z: 1 / (1 + np.exp(-z))

    def ame(var, a_, b_):
        x0, x1 = f.copy(), f.copy()
        x0[var], x1[var] = a_, b_
        X0, X1 = design(x0).values, design(x1).values
        point = (sig(X1 @ m.params.values) - sig(X0 @ m.params.values)).mean()
        draws = [(sig(X1 @ bb) - sig(X0 @ bb)).mean() for bb in sims]
        return 100 * float(point), 100 * float(np.percentile(draws, 2.5)), 100 * float(np.percentile(draws, 97.5))
    out = {"period": ame("period", first, last)}
    if "actor" in levels and "central bank" in levels["actor"]:
        out["actor"] = ame("actor", "academic", "central bank")
    return out


def with_covariates(frame):
    """Adds period, actor (academic / central bank / press) and english columns."""
    f = frame.copy()
    f["period"] = None
    for lbl, y0, y1 in PERIODS:
        f.loc[(f.year >= y0) & (f.year <= y1), "period"] = lbl
    f["actor"] = f.source_type.map({"academic": "academic", "speech": "central bank", "official": "central bank",
                                    "press": "press"})
    f["english"] = (f.language == "en").astype(int)
    return f
