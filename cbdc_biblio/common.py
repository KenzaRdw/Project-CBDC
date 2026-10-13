"""Shared helpers: HTTP, JSONL I/O, relevance scoring, language, keywords, sections."""
import hashlib
import json
import re
import threading
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import CONTEXT_RE, NEEDS_CONTEXT, RAW, STRICT_CONTEXT_RE, STRONG, TERM_RE, USER_AGENT

# --------------------------------------------------------------------------- HTTP

def session(retries=5, backoff=2.0):
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    r = Retry(total=retries, backoff_factor=backoff, status_forcelist=(429, 500, 502, 503, 504),
              allowed_methods=("GET", "HEAD"), respect_retry_after_header=True)
    s.mount("http://", HTTPAdapter(max_retries=r, pool_maxsize=32))
    s.mount("https://", HTTPAdapter(max_retries=r, pool_maxsize=32))
    return s


class Throttle:
    """Minimum interval between calls (thread-safe)."""

    def __init__(self, interval):
        self.interval, self.last, self.lock = interval, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            delta = self.interval - (time.monotonic() - self.last)
            if delta > 0:
                time.sleep(delta)
            self.last = time.monotonic()


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# --------------------------------------------------------------------------- I/O

def raw_path(source):
    return RAW / f"{source}.jsonl"


def write_jsonl(path, records, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_id(prefix, native):
    return f"{prefix}-{hashlib.sha1(str(native).encode()).hexdigest()[:12]}"


def new_record(**kw):
    r = {"retrieved_at": now_iso()}
    r.update({k: v for k, v in kw.items() if v not in (None, "", [], {})})
    return r

# --------------------------------------------------------------------------- text utils

def clean(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).replace("\x00", "")
    return re.sub(r"[ \t ]+", " ", s).strip()


def norm_title(t):
    t = unicodedata.normalize("NFKD", (t or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9Ѐ-ӿ一-鿿]+", "", t)


def join(values, sep="; "):
    out, seen = [], set()
    for v in values or []:
        v = clean(v)
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return sep.join(out)


def date_and_precision(value):
    """Normalise a date-ish value to (YYYY[-MM[-DD]], precision)."""
    if not value:
        return "", ""
    v = str(value).strip()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", v)
    if m:
        return m.group(0), "day"
    m = re.match(r"(\d{4})(\d{2})(\d{2})T", v)  # GDELT 20240131T120000Z
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "day"
    m = re.match(r"(\d{4})-(\d{2})$", v)
    if m:
        return v, "month"
    m = re.match(r"(\d{4})$", v)
    if m:
        return v, "year"
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%d %B %Y", "%B %d, %Y", "%a, %d %b %Y %H:%M:%S %z"):
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d"), "day"
        except ValueError:
            pass
    return "", ""

_MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
_TXT_DATES = [
    (re.compile(r"\b(\d{1,2})\.(\d{1,2})\.((?:19|20)\d{2})\b"), lambda m: (m[3], m[2], m[1])),
    (re.compile(r"\b(\d{1,2})\s+(%s)\s+((?:19|20)\d{2})\b" % "|".join(_MONTHS), re.I),
     lambda m: (m[3], _MONTHS[m[2].lower()], m[1])),
    (re.compile(r"\b(%s)\s+(\d{1,2}),\s+((?:19|20)\d{2})\b" % "|".join(_MONTHS), re.I),
     lambda m: (m[3], _MONTHS[m[1].lower()], m[2])),
]


def date_from_text(text, min_year=2014):
    """First plausible full date in the opening of a document (e.g. 'Brussels, 28.6.2023')."""
    head = (text or "")[:3000]
    hits = []
    for rx, f in _TXT_DATES:
        for m in rx.finditer(head):
            y, mo, d = (int(x) for x in f(m))
            if min_year <= y <= datetime.now().year and 1 <= mo <= 12 and 1 <= d <= 31:
                hits.append((m.start(), f"{y:04d}-{mo:02d}-{d:02d}"))
    return min(hits)[1] if hits else ""

# --------------------------------------------------------------------------- relevance

def match_terms(text):
    """Return (total mentions, {label: count}) of CBDC vocabulary in text."""
    counts = {}
    if not text:
        return 0, counts
    for label, rx in TERM_RE.items():
        n = len(rx.findall(text))
        if n:
            counts[label] = n
    return sum(counts.values()), counts


THEME_LABELS = ["main theme", "secondary theme", "mentioned only", "ambiguous terms only", "off-topic"]


def theme(title="", keywords="", abstract="", body="", context=False):
    """How central CBDCs (Central Bank Digital Currency, not homonyms) are to a document.

    Returns (label, score 0-100, total mentions, terms string). Each available field gets a
    centrality in [0, 1]:
      title     1 if a CBDC term is in the title
      keywords  0.4 if CBDC is among the author / indexed keywords (partly machine-assigned)
      abstract  share of its sentences that mention CBDCs
      body      half density (mentions per 1,000 words, saturating at 3), half spread
                (share of ten equal slices of the text that mention CBDCs)
    score = 100 * (0.6 * strongest field + 0.4 * mean of available fields): one field clearly
    about CBDCs makes it the main theme; a passing mention stays low. Missing fields (no
    abstract, no full text) are not counted, so a headline-only press item is judged on its title.
    Ambiguous terms ("CBDC", "digital pound", "DCEP"...) count only with monetary context;
    never-unambiguous ones ("digital dollar", "Sand Dollar") are reported but do not score.
    """
    head = "\n".join([title or "", keywords or "", abstract or ""])
    # context=True: the record was found by a CBDC-specific search (press headlines are too short to
    # carry monetary context of their own)
    has_ctx = context or CONTEXT_RE.search(head) or STRICT_CONTEXT_RE.search((body or "")[:200000])

    def counts(text):
        _, c = match_terms(text)
        return c if has_ctx else {k: v for k, v in c.items() if k not in NEEDS_CONTEXT}

    def n_strong(text):
        return sum(v for k, v in counts(text).items() if k in STRONG)

    allc = Counter()
    for t in (title, keywords, abstract, body):
        allc.update(counts(t))
    terms = "; ".join(f"{k} ({v})" for k, v in allc.most_common())

    c = {"title": float(n_strong(title) > 0)}
    if keywords:  # partly machine-assigned (OpenAlex, S2): supporting evidence only
        c["keywords"] = 0.4 * (n_strong(keywords) > 0)
    if abstract and len(abstract) > 100:
        sents = [x for x in re.split(r"(?<=[.!?。])\s+", abstract) if len(x) > 20] or [abstract]
        c["abstract"] = sum(1 for x in sents if n_strong(x)) / len(sents)
    words = len(body.split()) if body else 0
    if words > 300:
        density = n_strong(body) * 1000 / words
        slices = [body[i * len(body) // 10:(i + 1) * len(body) // 10] for i in range(10)]
        spread = sum(1 for x in slices if n_strong(x)) / 10
        c["body"] = 0.5 * min(1, density / 3) + 0.5 * spread
    score = round(100 * (0.6 * max(c.values()) + 0.4 * sum(c.values()) / len(c)))

    if score >= 50:
        label = "main theme"
    elif score >= 25:
        label = "secondary theme"
    elif score > 0:
        label = "mentioned only"
    elif allc:
        label = "ambiguous terms only"
    else:
        label = "off-topic"
    return label, score, sum(allc.values()), terms


def relevance(head_text, body_text=""):
    """Classify a document.

    core        - strong CBDC term in title / abstract / keywords
    substantial - >=3 strong mentions in the body
    peripheral  - 1-2 strong mentions in the body
    weak        - only ambiguous terms (e.g. "digital dollar") matched
    none        - nothing matched, or no monetary context at all
    """
    _, head = match_terms(head_text)
    _, body = match_terms(body_text)
    if not CONTEXT_RE.search(head_text or "") and not STRICT_CONTEXT_RE.search((body_text or "")[:200000]):
        head = {k: v for k, v in head.items() if k not in NEEDS_CONTEXT}
        body = {k: v for k, v in body.items() if k not in NEEDS_CONTEXT}
    allc = {k: head.get(k, 0) + body.get(k, 0) for k in set(head) | set(body)}
    strong_head = sum(v for k, v in head.items() if k in STRONG)
    strong_body = sum(v for k, v in body.items() if k in STRONG)
    if strong_head:
        rel = "core"
    elif strong_body >= 3:
        rel = "substantial"
    elif strong_body:
        rel = "peripheral"
    elif allc:
        rel = "weak"
    else:
        rel = "none"
    terms = "; ".join(f"{k} ({v})" for k, v in sorted(allc.items(), key=lambda x: -x[1]))
    return rel, sum(allc.values()), terms

# --------------------------------------------------------------------------- language

_LANG_LOCK = threading.Lock()


def detect_language(text):
    text = (text or "")[:4000]
    if len(re.sub(r"\W", "", text)) < 30:
        return ""
    from langdetect import DetectorFactory, detect
    with _LANG_LOCK:
        DetectorFactory.seed = 0
        try:
            return detect(text)
        except Exception:
            return ""


GDELT_LANG = {
    "English": "en", "French": "fr", "German": "de", "Spanish": "es", "Portuguese": "pt",
    "Italian": "it", "Chinese": "zh", "Japanese": "ja", "Korean": "ko", "Russian": "ru",
    "Arabic": "ar", "Turkish": "tr", "Dutch": "nl", "Polish": "pl", "Indonesian": "id",
    "Vietnamese": "vi", "Thai": "th", "Hindi": "hi", "Ukrainian": "uk", "Swedish": "sv",
    "Norwegian": "no", "Danish": "da", "Finnish": "fi", "Greek": "el", "Czech": "cs",
    "Romanian": "ro", "Hungarian": "hu", "Hebrew": "he", "Persian": "fa", "Urdu": "ur",
    "Bengali": "bn", "Malay": "ms", "Bulgarian": "bg", "Croatian": "hr", "Serbian": "sr",
    "Slovak": "sk", "Slovenian": "sl", "Lithuanian": "lt", "Latvian": "lv", "Estonian": "et",
    "Catalan": "ca", "Tamil": "ta", "Swahili": "sw", "Filipino": "tl",
}

# --------------------------------------------------------------------------- keywords

_YAKE_LANGS = {"en", "fr", "de", "es", "pt", "it", "nl", "ru", "pl", "tr", "sv", "da", "fi", "no",
               "ro", "cs", "el", "hu", "uk", "ar", "id", "ja", "zh"}


def auto_keywords(text, lang="en", top=12):
    text = (text or "")[:12000]
    if len(text) < 200:
        return ""
    import yake
    try:
        ex = yake.KeywordExtractor(lan=lang if lang in _YAKE_LANGS else "en", n=3, top=top, dedupLim=0.7)
        return "; ".join(k for k, _ in ex.extract_keywords(text))
    except Exception:
        return ""

# --------------------------------------------------------------------------- sections

_ENUM = r"(?:\d{1,2}(?:\.\d{1,2})*|[IVXL]{1,5}|[A-H])[.)]?\s+"
_NUM = rf"(?:#{{1,4}}\s*)?(?:{_ENUM})?"  # optional markdown hashes + optional numbering
_INTRO = r"introduc\w*|einleitung|einf[üu]hrung|introdu[cç][ãa]o|introduzione|inleiding|引言|序論|はじめに|введение|wst[ęe]p"
_CONCL = (r"conclu\w*|concluding\s+remarks?|final\s+remarks?|closing\s+remarks?|summary(?:\s+and\s+conclusions?)?|"
          r"final\s+thoughts|fazit|schlussfolgerung\w*|schlussbemerkung\w*|considera[çc][õo]es\s+finais|"
          r"conclusiones|conclusioni|consideraciones\s+finales|结论|結論|おわりに|заключение|выводы|wnioski|outlook")
_BACK = r"references|bibliography|bibliographie|literatur\w*|referencias|refer[êe]ncias|riferimenti|appendi\w+|annex\w*|acknowledg\w+|notes|endnotes|footnotes|参考文献|литература"

HEADING_RE = re.compile(rf"^\s*{_NUM}[A-ZÀ-Ý一-鿿Ѐ-Я][^\n]{{1,90}}$")
INTRO_RE = re.compile(rf"^\s*{_NUM}(?:{_INTRO})\b[\s:.]*$", re.I)
CONCL_RE = re.compile(rf"^\s*{_NUM}(?:{_CONCL})\b[^\n]{{0,40}}$", re.I)
BACK_RE = re.compile(rf"^\s*{_NUM}(?:{_BACK})\b[\s:.]*$", re.I)
FOOTNOTE_RE = re.compile(r"^\s*[-–•*]?\s*\[?\d{1,3}\]?\s{1,3}(?:See|For|Cf|Ibid|https?:|[A-Z][a-z]+,? [A-Z]\.|.*\(\d{4}\))")
NUMBERED_RE = re.compile(rf"^\s*(?:#{{1,4}}\s*)?{_ENUM}[A-ZÀ-Ý][^\n]{{2,90}}$")


def _is_heading(line):
    s = line.strip()
    if not s or len(s) > 100 or s.endswith((".", ",", ";")) and not re.match(_ENUM, s):
        return False
    if INTRO_RE.match(s) or CONCL_RE.match(s) or BACK_RE.match(s):
        return True
    if s.startswith("#"):
        return True
    if NUMBERED_RE.match(s) and len(s.split()) <= 12:
        return True
    letters = re.sub(r"[^A-Za-zÀ-ÿ]", "", s)
    return len(letters) >= 6 and letters.isupper() and len(s.split()) <= 10


def split_sections(text, max_chars=8000):
    """Heuristic extraction of introduction / conclusion / outline.

    Returns dict(introduction, conclusion, section_headings, sections_method).
    Falls back to the opening / closing paragraphs (typical for speeches and press)
    when no explicit headings are found; the method column says which was used.
    """
    out = {"introduction": "", "conclusion": "", "section_headings": "", "sections_method": ""}
    if not text or len(text) < 500:
        return out
    lines = text.splitlines()
    heads = [(i, l.strip().lstrip("#").strip()) for i, l in enumerate(lines) if _is_heading(l)]
    out["section_headings"] = " | ".join(h for _, h in heads[:60])

    def body_after(idx):
        nxt = next((i for i, _ in heads if i > idx), len(lines))
        chunk = "\n".join(lines[idx + 1:nxt]).strip()
        chunk = "\n\n".join(p for p in re.split(r"\n\s*\n", chunk) if not FOOTNOTE_RE.match(p))
        return chunk[:max_chars]

    methods = []
    intro_idx = next((i for i, h in heads if INTRO_RE.match(lines[i])), None)
    if intro_idx is not None:
        out["introduction"] = body_after(intro_idx)
        methods.append("intro:heading")
    concl_idx = next((i for i, h in reversed(heads) if CONCL_RE.match(lines[i])), None)
    if concl_idx is not None and concl_idx != intro_idx:
        out["conclusion"] = body_after(concl_idx)
        methods.append("conclusion:heading")

    # fallback: opening / closing paragraphs (excluding back matter)
    back_idx = next((i for i, h in heads if BACK_RE.match(lines[i]) and i > len(lines) * 0.5), len(lines))
    paras = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(lines[:back_idx]))
             if len(p.strip()) > 80 and not FOOTNOTE_RE.match(p)]
    if not paras:
        paras = [p for p in re.split(r"(?<=[.!?])\s+", " ".join(lines[:back_idx])) if p]
    if not out["introduction"] and paras:
        out["introduction"] = _take(paras, 1500)
        methods.append("intro:opening-paragraphs")
    if not out["conclusion"] and len(paras) > 3:
        out["conclusion"] = _take(list(reversed(paras)), 1500, reverse=True)
        methods.append("conclusion:closing-paragraphs")
    out["sections_method"] = "; ".join(methods)
    return out


def _take(paras, limit, reverse=False):
    acc, n = [], 0
    for p in paras:
        acc.append(p)
        n += len(p)
        if n >= limit:
            break
    if reverse:
        acc.reverse()
    return "\n\n".join(acc)[: limit * 3]

# --------------------------------------------------------------------------- full text store

def save_fulltext(record_id, text):
    """Write full content to data/fulltext/<id>.txt; return (relative path, chars)."""
    from .config import FULLTEXT, ROOT
    text = clean_block(text)
    if not text:
        return "", 0
    p = FULLTEXT / f"{record_id}.txt"
    p.write_text(text, encoding="utf-8")
    return str(p.relative_to(ROOT)), len(text)


def clean_block(text):
    """Normalise whitespace but keep paragraph breaks."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).replace("\x00", "").replace("\r", "")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n[ ]+", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
