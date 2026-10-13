"""Search vocabulary, paths and output schema for the CBDC corpus."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"            # one JSONL per source, as harvested
FULLTEXT = DATA / "fulltext"  # one .txt per record (full content)
CACHE = DATA / "cache"        # bulk downloads (BIS / ECB speech dumps)
for _d in (RAW, FULLTEXT, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

USER_AGENT = "CBDC-bibliometrics/1.0 (academic research; python-requests)"

# ---------------------------------------------------------------------------
# Vocabulary. Each entry: canonical label -> regex (case-insensitive).
# Used both to build source queries and to score relevance of every record.
# ---------------------------------------------------------------------------
TERMS = {
    # generic
    "CBDC": r"\bCBDCs?\b|\bwCBDCs?\b|\brCBDCs?\b|\bmCBDCs?\b",
    "central bank digital currency": r"central[\s-]+banks?(?:['’]s?)?[\s-]+digital[\s-]+currenc(?:y|ies)",
    "digital central bank money": r"digital\s+central[\s-]+bank\s+money",
    "tokenised central bank money": r"tokeni[sz]ed\s+(?:central[\s-]+bank\s+money|reserves)",
    # projects / currencies
    "digital euro": r"digital(?:e[rn]?)?\s+euro\b|euro\s+num[ée]rique|euro\s+digitale?\b|euro\s+digital\b|digitale\s+euro",
    "e-CNY / digital yuan": r"\be-?CNY\b|digital\s+(?:yuan|renminbi|RMB)\b|数字人民币|央行数字货币|法定数字货币",
    "DCEP (e-CNY)": r"\bDC/?EP\b|digital\s+currency\s+electronic\s+payment",
    "digital pound": r"digital\s+pound(?!-)|britcoin",
    "digital dollar": r"digital\s+(?:US\s+)?dollar\b",
    "e-krona": r"\be-?krona\b",
    "digital rupee": r"\be-?rupee\b|digital\s+rupee|\be₹",
    "eNaira": r"\be-?naira\b",
    "Sand Dollar": r"\bsand\s+dollar\b",
    "DCash": r"\bDCash\b",
    "JAM-DEX": r"\bJAM-?DEX\b",
    "digital ruble": r"digital\s+ro?ubles?|цифров\w+\s+рубл\w*",
    "Drex / digital real": r"\bDrex\b|real\s+digital\b|digital\s+real\b",
    "e-hryvnia": r"\be-?hryvnia\b",
    "digital yen": r"digital\s+yen\b",
    "e-HKD": r"\be-?HKD\b",
    "e-peso": r"\be-?peso\b",
    "digital fiat / central bank crypto": (
        r"digital\s+fiat\s+currenc|central[\s-]+bank[\s-]+(?:issued\s+)?(?:crypto-?currenc|digital\s+(?:money|cash)|"
        r"e-?money|electronic\s+(?:money|cash))|electronic\s+central[\s-]+bank\s+money|\bFedcoin\b|\bRSCoin\b|"
        r"\bCADcoin\b|sovereign\s+digital\s+currenc|digital\s+legal\s+tender|national\s+digital\s+currenc"),
    "tokenised deposits / unified ledger": r"tokeni[sz]ed\s+deposits?|unified\s+ledger",
    "other national CBDCs": (
        r"digital\s+(?:won|shekel|lira|rial)\b|\be-?cedi\b|\be-?dirham\b|\bBakong\b|rupiah\s+digital|"
        r"dijital\s+t[üu]rk\s+liras|cyfrow\w+\s+(?:euro|z[łl]ot)|цифров\w+\s+гривн|е-гривн|"
        r"العملة\s+الرقمية|중앙은행\s*디지털|디지털\s*화폐|デジタル円"),
    "BIS Innovation Hub projects": (
        r"\bmBridge\b|project\s+(?:Helvetia|Jura|Dunbar|Icebreaker|Aurum|Rosalind|Tourbillon|"
        r"Mariana|Hamilton|Stella|Ubin|Inthanon|Khokha|Jasper|Agor[aá]|Orchid|Polaris|Sela|Mandala|Ensemble)"
    ),
    # other languages
    "FR: MNBC": r"monnaies?\s+num[ée]riques?\s+de\s+banques?\s+centrales?|\bMNBC\b",
    "ES: MDBC": r"monedas?\s+digitales?\s+(?:de|del)\s+bancos?\s+centrale?s?|dinero\s+digital\s+de\s+banco\s+central",
    "PT: MDBC": r"moedas?\s+digita(?:l|is)\s+(?:de|do|dos)\s+bancos?\s+centra(?:l|is)",
    "DE: digitales Zentralbankgeld": r"digitale[sn]?\s+zentralbankgeld|digitale\s+zentralbankw[äa]hrung",
    "IT: valuta digitale di banca centrale": r"(?:valuta|moneta)\s+digitale\s+(?:di|della)\s+banca\s+centrale",
    "JA/KO/RU": r"中央銀行デジタル通貨|중앙은행\s*디지털\s*화폐|цифров\w+\s+валют\w+\s+центральн\w+\s+банк\w*",
}
TERM_RE = {k: re.compile(v, re.I) for k, v in TERMS.items()}

# Terms that are unambiguous on their own. "digital dollar", "e-peso", "Drex"
# etc. only count when a generic CBDC term also appears (see common.relevance).
STRONG = {
    "CBDC", "central bank digital currency", "digital central bank money", "digital euro",
    "e-CNY / digital yuan", "digital pound", "e-krona", "eNaira", "DCash", "JAM-DEX",
    "BIS Innovation Hub projects", "FR: MNBC", "ES: MDBC", "PT: MDBC",
    "DE: digitales Zentralbankgeld", "IT: valuta digitale di banca centrale", "JA/KO/RU",
    "tokenised central bank money", "digital rupee", "digital ruble", "e-hryvnia", "e-HKD",
    "digital fiat / central bank crypto", "DCEP (e-CNY)",
}

# Ambiguous terms only count when the document also has monetary context (CONTEXT_RE)
NEEDS_CONTEXT = {"CBDC", "BIS Innovation Hub projects", "digital pound", "digital dollar", "digital yen",
                 "e-peso", "Drex / digital real", "Sand Dollar", "DCash", "e-HKD", "e-krona", "digital rupee",
                 "JAM-DEX", "eNaira", "e-hryvnia", "digital ruble",
                 "digital fiat / central bank crypto", "tokenised deposits / unified ledger", "other national CBDCs",
                 "DCEP (e-CNY)"}

# Monetary context. "CBDC" is also "chronic bullous dermatosis of childhood",
# "Project Stella" a satellite, "digital Pound-Drever-Hall" a laser lock, etc.
CONTEXT_RE = re.compile(
    r"\bcurrenc|monetar|money|payment|central[\s-]+bank|banking|financ|econom|экономи|ledger|blockchain|stablecoin|crypto|fintech|"
    r"monnaie|mon[ée]taire|paiement|banque|moneda|dinero|banco|moeda|pagamento|geld|w[äa]hrung|zahlung|finanz|"
    r"valuta|moneta|banca|货币|支付|银行|通貨|決済|銀行|화폐|은행|валют|денеж|деньг|банк|платеж", re.I)

# Stricter context required when the evidence comes only from the body text.
STRICT_CONTEXT_RE = re.compile(
    r"central[\s-]+bank|digital[\s-]+currenc|monetary|payments?\b|stablecoin|cryptocurrenc|"
    r"banques?\s+centrales?|bancos?\s+centra|zentralbank|banca\s+centrale|monnaie|moneda|moeda|"
    r"中央银行|央行|货币|中央銀行|通貨|중앙은행|центральн\w+\s+банк|валют", re.I)

# Phrases sent to search APIs (each one is queried separately where needed).
QUERY_PHRASES = [
    "central bank digital currency", "central bank digital currencies", "CBDC",
    "digital euro", "e-CNY", "digital yuan", "digital renminbi", "digital pound", "e-krona",
    "digital rupee", "e-rupee", "eNaira", "DCash", "JAM-DEX", "digital ruble",
    "digital rouble", "e-hryvnia", "e-HKD", "mBridge", "Project Helvetia", "Project Jura",
    "Project Dunbar", "Project Hamilton", "Project Ubin", "Project Inthanon", "Project Jasper",
    "Project Stella", "Project Khokha", "Project Rosalind", "Project Tourbillon",
    "Project Mariana", "Project Agora", "tokenised central bank money",
    "digital central bank money", "wholesale CBDC", "retail CBDC",
    "monnaie numérique de banque centrale", "euro numérique",
    "moneda digital de banco central", "moeda digital de banco central",
    "digitaler Euro", "digitales Zentralbankgeld", "euro digitale",
    "valuta digitale di banca centrale", "数字人民币", "中央銀行デジタル通貨", "цифровой рубль",
    # earlier / alternative names (pre-2017 literature) and broader variants
    "digital fiat currency", "digital fiat currencies", "central bank cryptocurrency",
    "central bank cryptocurrencies", "central bank issued digital currency", "central bank e-money",
    "central bank electronic money", "electronic central bank money", "central bank digital money",
    "central bank digital cash", "sovereign digital currency", "national digital currency",
    "digital legal tender", "digital base money", "Fedcoin", "RSCoin", "CADcoin", "DCEP",
    "digital currency electronic payment", "tokenised deposits", "tokenized deposits",
    "tokenized central bank money", "tokenised reserves", "unified ledger", "digital dollar",
    "digital yen", "digital won", "digital real", "Drex", "e-peso", "Sand Dollar", "Bakong",
    "e-Cedi", "digital shekel", "digital lira", "digital rial", "e-dirham",
    "Project Icebreaker", "Project Aurum", "Project Polaris", "Project Sela", "Project Mandala",
    "Project Ensemble", "Project Orchid", "Project Cedar", "Project Atom", "Project Bakong",
    "Project Inthanon-LionRock", "Project Nexus", "Project Pine",
    "monnaie numérique de banque centrale de gros", "monnaie digitale de banque centrale",
    "dinero digital de banco central", "digitales Zentralbankgeld", "digitale Zentralbankwährung",
    "moneta digitale di banca centrale", "e-krona", "digitale euro", "euro digital",
    "цифровая валюта центрального банка", "цифрова гривня", "央行数字货币", "法定数字货币",
    "デジタル円", "중앙은행 디지털화폐", "rupiah digital", "dijital Türk lirası",
    "العملة الرقمية للبنك المركزي", "cyfrowe euro", "cyfrowy złoty",
]
QUERY_PHRASES = list(dict.fromkeys(QUERY_PHRASES))

# ---------------------------------------------------------------------------
# Output schema (column order of the master CSV).
# ---------------------------------------------------------------------------
COLUMNS = [
    "record_id", "source_db", "source_type", "document_type",
    "title", "subtitle", "authors", "author_affiliations",
    "issuer", "issuer_type", "venue", "publisher", "country",
    "publication_date", "date_precision", "year",
    "language", "language_origin",
    "doi", "url", "pdf_url", "native_id",
    "abstract", "introduction", "conclusion", "section_headings", "sections_method",
    "keywords_author", "keywords_indexed", "keywords_auto",
    "cbdc_terms", "cbdc_mentions", "relevance", "theme_score",
    "cited_by_count", "references", "open_access", "license",
    "full_text_chars", "full_text_path", "full_text_status", "full_text",
    "retrieved_at",
]
