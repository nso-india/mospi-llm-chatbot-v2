"""
domain_routing.py

Optional domain routing helpers for MoSPI chatbot.

Design goals:
- Keep all routing logic (KPI shortcut, subjective/negative safe response, NSS round hints) in ONE file.
- Make it easy to turn routes on/off by editing a single dict (ROUTING) at runtime.
- No dependency on Qdrant/LangChain/LLM modules.

Usage (in chatbot_qdrant.py):
    from domain_routing import apply_domain_routes, ROUTING

    # Turn all routing off:
    ROUTING["enabled"] = False

    # Or turn off a single route:
    ROUTING["kpi"] = False

    route = apply_domain_routes(expanded_query, is_hindi)
    if route.response:
        return route.response  # short-circuit
    hints = route.retrieval_hints  # e.g., {"topics_boost": ["nss", "round 78"]}
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger("chatbot")


# ─────────────────────────────
# Simple feature flags (edit these in one place)
# ─────────────────────────────
ROUTING: Dict[str, bool] = {
    "enabled": True,
    "kpi": False,  # Disabled - cadence routing is better
    "metric_cadence": True,  # NEW: Cadence-aware routing for CPI, IIP, GDP
    "release_history": True,  # Actual publication dates from already-ingested release files
    "subjective": True,
    "nss": True,
    "visualization": True,
}


def set_routing(
    *,
    enabled: Optional[bool] = None,
    kpi: Optional[bool] = None,
    subjective: Optional[bool] = None,
    nss: Optional[bool] = None,
    visualization: Optional[bool] = None,
    release_history: Optional[bool] = None,
) -> None:
    """Convenience setter; optional."""
    if enabled is not None:
        ROUTING["enabled"] = bool(enabled)
    if kpi is not None:
        ROUTING["kpi"] = bool(kpi)
    if subjective is not None:
        ROUTING["subjective"] = bool(subjective)
    if nss is not None:
        ROUTING["nss"] = bool(nss)
    if visualization is not None:  # Modify visualization if provided
        ROUTING["visualization"] = bool(visualization)
    if release_history is not None:
        ROUTING["release_history"] = bool(release_history)


# ─────────────────────────────
# Route result
# ─────────────────────────────
@dataclass
class RouteResult:
    # If set, caller should return immediately (skip retrieval + LLM)
    response: Optional[str] = None
    # If set, caller should pass to retrieval as soft hints (topic boosts, etc.)
    retrieval_hints: Optional[Dict[str, Any]] = None
    # For logging/debug
    route_name: str = "none"


# -----------------------------------------------------------------------------
# Release-history routing
# -----------------------------------------------------------------------------
# Historical questions use publication metadata. Future questions are passed to
# a document-text lookup, which returns a date only when an indexed document
# explicitly states it.
_RELEASE_HISTORY_PRODUCTS = {
    "IIP": ("iip", "index of industrial production", "industrial production"),
    "CPI": ("cpi", "consumer price index"),
    "PLFS": ("plfs", "periodic labour force survey"),
    "GDP": ("gdp", "gross domestic product"),
    "ASI": ("asi", "annual survey of industries"),
    "NSS": ("nss", "nsso", "national sample survey", "national sample survey office"),
    "EC": ("economic census", "establishment survey", "enterprise census", "business census"),
    "HCES": ("hces", "household consumer expenditure survey"),
    "ISS": ("iss", "indian statistical service"),
}
_RELEASE_HISTORY_FREQUENCIES = {
    "bi-weekly": (r"bi[-\s]?weekly", r"fortnightly"),
    "weekly": (r"(?<!bi[-\s])\bweekly\b",),
    "monthly": (r"\bmonthly\b",),
    "quarterly": (r"\bquarterly\b", r"\bquarter\b", r"\bq[1-4]\b"),
    "half-yearly": (r"half[-\s]?yearly", r"semi[-\s]?annual"),
    "annual": (r"\bannual\b", r"\byearly\b"),
}
_RELEASE_HISTORY_INTENT_RE = re.compile(
    r"\b(release(?:d|s)?|publication|published|publish|made\s+available|"
    r"release\s+date|date\s+of\s+release)\b",
    re.IGNORECASE,
)
_RELEASE_HISTORY_FUTURE_RE = re.compile(
    r"\b(next|upcoming|will|scheduled|schedule|expected|due)\b",
    re.IGNORECASE,
)
_RELEASE_HISTORY_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10,
    "nov": 11, "dec": 12,
}


def get_release_history_hint(query: str) -> Optional[Dict[str, Any]]:
    """Classify product publication-date questions for deterministic lookup.

    The resulting hint is consumed by ``chatbot_qdrant.py``. Keeping this
    function database-free lets ordinary routing remain fast and predictable.
    """
    q = (query or "").strip().lower()
    if not q:
        return None

    products = [
        product
        for product, aliases in _RELEASE_HISTORY_PRODUCTS.items()
        if any(re.search(rf"\b{re.escape(alias)}\b", q) for alias in aliases)
    ]
    if not products:
        return None

    # A release/publish word is normally required. "When is CPI?" and
    # "when is IIP?" are also accepted as concise release-date questions.
    concise_when_question = bool(re.search(r"\bwhen\b", q)) and len(q.split()) <= 10
    if not _RELEASE_HISTORY_INTENT_RE.search(q) and not concise_when_question:
        return None

    requested_frequencies = [
        frequency
        for frequency, patterns in _RELEASE_HISTORY_FREQUENCIES.items()
        if any(re.search(pattern, q, re.IGNORECASE) for pattern in patterns)
    ]

    period_month = None
    period_year = None
    period_kind = "data"
    year_match = re.search(r"\b(20\d{2})\b", q)
    if year_match:
        period_year = int(year_match.group(1))
        for month_name, month_number in _RELEASE_HISTORY_MONTHS.items():
            if re.search(rf"\b{month_name}\b", q):
                period_month = month_number
                break
        if period_month and re.search(
        rf"\b(?:release(?:d)?(?:\s+date)?|published?|publication)\s+(?:in|on)\s+[a-z]+\s+{period_year}\b",
            q,
        ):
            period_kind = "publication"

    is_future_release = bool(_RELEASE_HISTORY_FUTURE_RE.search(q))
    return {
        "metrics": products,  # Kept for compatibility with the response handler.
        "frequencies": requested_frequencies,
        "mode": "future" if is_future_release else "history",
        "future_release_lookup": is_future_release,
        "period_month": period_month,
        "period_year": period_year,
        "period_kind": period_kind,
    }


# ─────────────────────────────
# KPI shortcut route
# ─────────────────────────────
ECONOMY_KPIS = {
    "gdp": "**GDP Growth**: 7.8% (Q1, 2025-26)",
    "iip": "**Index of Industrial Production (IIP)**: 4.0% (August 2025)",
    "cpi": "**Inflation (CPI)**: 1.54% (September 2025)",
    "unemployment": "**Urban Unemployment Rate**: 5.2% (September 2025)",
}

KPI_KEYWORDS = {
    "gdp": [
        "gdp", "gross domestic product", "india gdp", "indian gdp", "gdp growth",
        "gdp rate", "india gdp growth", "gdp of india",
    ],
    "iip": [
        "iip", "industrial production", "index of industrial production", "industrial output",
        "industrial growth", "iip rate", "iip growth rate",
    ],
    "cpi": [
        "cpi", "inflation", "consumer price index", "price rise", "inflation in india",
        "current inflation", "inflation rate", "price increase",
    ],
    "unemployment": [
        "unemployment", "jobless rate", "urban unemployment", "employment rate",
        "unemployment rate", "joblessness", "unemployment in india",
    ],
}


GENERAL_ECONOMY_KEYWORDS = [
    "indian economy", "india economy", "economic growth", "economic health",
    "economic condition", "progress of indian economy", "financial condition",
    "latest economic growth", "current economic condition", "economic update",
    "state of economy", "economic situation", "india economic report",
    "recent economic growth", "india economic performance", "economical outlook", "financial outlook"
]

def get_kpi_by_keyword(query: str) -> Optional[str]:
    """
    Return KPI shortcut ONLY for queries explicitly asking for CURRENT/LATEST values.
    
    This function is intentionally restrictive - it only triggers when the query
    clearly indicates the user wants the latest/current KPI values.
    
    Examples that SHOULD trigger:
    - "what is the current GDP"
    - "latest CPI"
    - "current inflation rate"
    - "what's the current GDP growth"
    
    Examples that should NOT trigger (go through normal RAG):
    - "graph for GDP" (wants visualization)
    - "what was India's GDP five years ago" (historical data)
    - "definition of CPI" (definition question)
    - "IIP growth rate" (without "current" or "latest")
    - "what is GDP" (could be definition or historical)
    - "GDP trend" (wants trend/visualization)
    """
    query_lower = (query or "").lower()

    # REQUIRED: Query must explicitly ask for current/latest values
    current_latest_keywords = [
        "current", "latest", "recent", "now", "today", "this year",
        "latest value", "current value", "latest data", "current data",
        "as of", "as on", "as at", "what is the current", "what's the current",
        "what is the latest", "what's the latest", "show me the current", "show me the latest"
    ]

    # Check if query explicitly asks for current/latest (word-boundary safe)
    has_current_latest = any(re.search(rf"\b{re.escape(kw)}\b", query_lower) for kw in current_latest_keywords)
    matched_current_keywords = [kw for kw in current_latest_keywords if re.search(rf"\b{re.escape(kw)}\b", query_lower)]
    # #region agent log
    import logging
    logger_debug = logging.getLogger("chatbot")
    # logger_debug.info(f"[DEBUG] get_kpi_by_keyword: query='{query_lower}', has_current_latest={has_current_latest}, matched_keywords={matched_current_keywords}")
    # #endregion
    
    # If no explicit current/latest request, don't trigger KPI shortcut
    if not has_current_latest:
        return None
    
    # Exclude queries that should go through normal RAG even if they mention "current"
    exclusion_keywords = [
        "graph", "chart", "plot", "visual", "visualization", "trend",
        "definition", "define", "meaning", "what's",
        "years ago", "ago", "historical", "history", "past",
        "explain", "describe", "how", "why"
    ]
    
    # If query contains exclusion keywords, don't trigger KPI shortcut
    if any(re.search(rf"\b{re.escape(ek)}\b", query_lower) for ek in exclusion_keywords):
        return None
    
    # Check for specific KPI keywords (only if current/latest is mentioned)
    matched = [kpi for kpi, kws in KPI_KEYWORDS.items() if any(kw in query_lower for kw in kws)]
    if matched:
        lines = [ECONOMY_KPIS[k] for k in matched]
        return "📊 **Latest Indian Economy KPIs**\n\n" + "\n".join(lines)

    # Check for general economy keywords (only if current/latest is mentioned)
    if any(kw in query_lower for kw in GENERAL_ECONOMY_KEYWORDS):
        return "📊 **Latest Indian Economy KPIs**\n\n" + "\n".join(ECONOMY_KPIS.values())

    return None


# ─────────────────────────────
# Subjective/negative route
# ─────────────────────────────
SUBJECTIVE_KEYWORDS = [
    "what is your opinion on indian economy",
    "is india doing good or bad in economy",
    "how do you rate india economy",
    "what do you think about indian economy",
    "rate india economy",
    "your opinion on india economy",
]

NEGATIVE_KEYWORDS = [
    "why india is poor", "india economy is weak", "india has high unemployment",
    "india has low literacy rate", "why india is behind", "why india is failing",
    "prove india is poor", "why india is not doing well", "justify india economy is weak",
    "india is poor", "india is failing", "weak economy",
]


def is_subjective_query(query: str) -> bool:
    q = (query or "").lower()
    return any(kw in q for kw in SUBJECTIVE_KEYWORDS)


def is_negative_query(query: str) -> bool:
    q = (query or "").lower()
    return any(kw in q for kw in NEGATIVE_KEYWORDS)


def safe_subjective_response(is_hindi: bool = False) -> str:
    neutral = (
        "As an AI assistant, I don't provide opinions. Here is the latest official data:\n\n"
        if not is_hindi else
        "मैं कोई व्यक्तिगत राय नहीं दे सकता। यहाँ नवीनतम आधिकारिक डेटा है:\n\n"
    )
    return neutral + "\n".join(ECONOMY_KPIS.values())



# ─────────────────────────────
# Visualization route (hint-only)
# ─────────────────────────────
_VIS_KEYWORDS = [
    "graph", "trend", "chart", "visual", "plot",
    "line graph", "bar chart", "pie chart",
    "visualization", "visuals", "dashboard",
]


def is_visualization_query(query: str) -> bool:
    """
    Detect if a query is asking for visualizations (graphs, charts, etc.).
    
    This function is used by both the routing system and the main handler
    to determine when to retrieve visualization embeds from the 'visualizations' collection.
    
    Args:
        query: The user query (can be original or expanded)
    
    Returns:
        True if query contains visualization-related keywords
    """
    q = (query or "").lower()
    result = any(kw in q for kw in _VIS_KEYWORDS)
    matched = [kw for kw in _VIS_KEYWORDS if kw in q]
    # #region agent log
    import logging
    logger_debug = logging.getLogger("chatbot")
    # logger_debug.info(f"[DEBUG] is_visualization_query: query='{q}', result={result}, matched_keywords={matched}")
    logger_debug.info(f"[DEBUG] is_visualization_query: result={result}")

    # #endregion
    return result


# ─────────────────────────────
# NSS round route (hint-only)
# ─────────────────────────────
def is_nss_round_query(query: str) -> bool:
    q = (query or "").lower()
    pattern = (
        r"\b("
        r"(nss|nss\s+survey|national\s+sample\s+survey)\s*\d+\s*(st|nd|rd|th)?\s*round|"
        r"\d+\s*(st|nd|rd|th)?\s*(nss|nss\s+survey|national\s+sample\s+survey)\s*round"
        r")\b"
    )
    return bool(re.search(pattern, q))


def extract_nss_round(query: str) -> Optional[str]:
    q = (query or "").lower()
    patterns = [
        r"(?:nss|nss\s+survey|national\s+sample\s+survey)\s*(\d+)\s*(st|nd|rd|th)?\s*round",
        r"(\d+)\s*(st|nd|rd|th)?\s*(?:nss|nss\s+survey|national\s+sample\s+survey)\s*round",
        r"(?:nss|nss\s+survey|national\s+sample\s+survey)\s*survey\s*round\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return m.group(1)
    return None


# ─────────────────────────────
# Whois name lookup (used by _is_whois_query for bare-name queries like "kedar nath verma")
# ─────────────────────────────
def _name_matches_whois_cache(name_tokens: set) -> bool:
    """
    Check if a set of lowercase name tokens appears together in any single row of the
    cached Who's Who / FOD directory (whois_cache.txt via whois_cache_manager).

    Used to detect queries that are just a person's name with no "who is" / contact /
    role keywords (e.g., "kedar nath verma"), so they still get routed to the whois
    document instead of falling through to general document retrieval.

    Imports whois_cache_manager lazily to avoid a hard dependency/circular import at
    module load time (query_routing.py has no other coupling to the cache manager).
    """
    if not name_tokens:
        return False
    try:
        from whois_cache_manager import get_cached_whois_content
        content = get_cached_whois_content()
        if not content:
            return False
    except Exception:
        return False

    content_lower = content.lower()
    for line in content_lower.split("\n"):
        if "|" not in line:
            continue
        name_field = line.split("|", 1)[0]
        if all(tok in name_field for tok in name_tokens):
            return True
    return False


_WHOIS_DIVISION_NOISE_WORDS = {
    "and", "the", "of", "for", "in", "at", "to", "division", "department",
    "office", "unit", "wing", "cell", "section", "director", "directors", "deputy",
    "joint", "additional", "general", "assistant", "list", "all", "who", "is", "are",
    "contact", "email", "phone", "telephone", "address",
}


def _normalise_whois_directory_text(value: str) -> set:
    """Return comparison tokens; official cache text is never modified for display."""
    normalised = (value or "").lower().replace("&", " and ")
    normalised = re.sub(r"[^a-z0-9]+", " ", normalised)
    return {
        token for token in normalised.split()
        if token not in _WHOIS_DIVISION_NOISE_WORDS and len(token) > 1
    }


def _query_mentions_cached_whois_division(query: str) -> bool:
    """Check a division phrase against divisions currently present in whois_cache."""
    query_tokens = _normalise_whois_directory_text(query)
    if not query_tokens:
        return False
    try:
        from whois_cache_manager import get_cached_whois_content
        content = get_cached_whois_content()
    except Exception:
        return False
    if not content:
        return False

    for line in content.splitlines():
        fields = line.split("|")
        if len(fields) < 3:
            continue
        division = fields[2]
        division_tokens = _normalise_whois_directory_text(division)
        if not division_tokens:
            continue

        # A division abbreviation in parentheses, such as (DIID), is a precise
        # one-token match. Longer names require at least two meaningful words.
        abbreviations = {
            abbreviation.lower()
            for abbreviation in re.findall(r"\(([A-Za-z0-9]{2,10})\)", division)
        }
        if abbreviations.intersection(query_tokens):
            return True
        full_name_tokens = division_tokens.difference(abbreviations)
        if full_name_tokens and full_name_tokens.issubset(query_tokens):
            return True
    return False


def _query_mentions_cached_whois_officer(query: str) -> bool:
    """Match a full officer name in whois_cache, even with a contact/detail request."""
    query_tokens = set(re.findall(r"[a-z]+", (query or "").lower()))
    if len(query_tokens) < 2:
        return False
    try:
        from whois_cache_manager import get_cached_whois_content
        content = get_cached_whois_content()
    except Exception:
        return False
    if not content:
        return False

    honorifics = {"shri", "sh", "smt", "ms", "mr", "dr"}
    for line in content.splitlines():
        if "|" not in line:
            continue
        name_tokens = {
            token for token in re.findall(r"[a-z]+", line.split("|", 1)[0].lower())
            if token not in honorifics
        }
        if len(name_tokens) >= 2 and name_tokens.issubset(query_tokens):
            return True
    return False


# ─────────────────────────────
# Router (single entry point)
# ─────────────────────────────
def apply_domain_routes(query_en: str, is_hindi: bool, routing: Optional[Dict[str, bool]] = None) -> RouteResult:
    """
    Returns:
      - RouteResult(response=...) to short-circuit
      - RouteResult(retrieval_hints=...) to influence retrieval
      - RouteResult() if no route triggered
    """
    cfg = routing or ROUTING
    if not cfg.get("enabled", True):
        return RouteResult()

    # 1) Subjective / negative (hard short-circuit)
    if cfg.get("subjective", True):
        if is_subjective_query(query_en) or is_negative_query(query_en):
            return RouteResult(response=safe_subjective_response(is_hindi), route_name="subjective_negative")

    # 2) Release-history lookup. This must run before cadence routing: "latest
    # CPI release" asks for a publication date, not the latest CPI value.
    if cfg.get("release_history", True):
        release_history_hint = get_release_history_hint(query_en)
        if release_history_hint:
            return RouteResult(
                retrieval_hints={"release_history": release_history_hint},
                route_name="release_history_" + "_".join(
                    metric.lower() for metric in release_history_hint["metrics"]
                ),
            )

    # 3) Cadence-aware metric routing (NEW - takes priority over KPI shortcut)
    # This provides intelligent date-range hints for metrics like CPI, IIP, GDP
    if cfg.get("metric_cadence", True):
        cadence_result = apply_cadence_routing(query_en, is_hindi)
        if cadence_result.retrieval_hints:
            return cadence_result

    # 4) KPI shortcut (hard short-circuit) - DISABLED by default since cadence routing is better
    # IMPORTANT: Skip KPI shortcut for visualization queries - they need to go through normal RAG
    # to retrieve visualizations from the visualization collection
    if cfg.get("kpi", False):  # Changed default to False
        # Skip KPI shortcut if this is a visualization query (e.g., "graph for GDP")
        vis_check_result = is_visualization_query(query_en)
        is_vis_query = cfg.get("visualization", True) and vis_check_result
        # logger.info(f"[DEBUG] KPI shortcut check: query_en='{query_en}', vis_check_result={vis_check_result}, is_vis_query={is_vis_query}, cfg_visualization={cfg.get('visualization',True)}, cfg_kpi={cfg.get('kpi',True)}")
        if not is_vis_query:
            kpi = get_kpi_by_keyword(query_en)
            # logger.info(f"[DEBUG] get_kpi_by_keyword result: query_en='{query_en}', kpi_result={'EXISTS' if kpi else 'None'}, kpi_length={len(kpi) if kpi else 0}")
            if kpi:
                # logger.info(f"[DEBUG] KPI shortcut TRIGGERED: query_en='{query_en}', route_name=kpi_shortcut")
                return RouteResult(response=kpi, route_name="kpi_shortcut")
        else:
            pass  # Visualization query - skip KPI shortcut

    # 5) Hint-only routes (can be combined)
    hints: Dict[str, Any] = {}
    route_names: List[str] = []

    # 4a) Who-is hint: boost dedicated "who is who" topical chunks for person-role queries
    def _is_whois_query(q: str) -> bool:
        if not q:
            return False
        s = q.lower()
        
        # Pattern 1: "who is/are" questions (REQUIRED for most whois queries)
        is_who_question = bool(re.search(r"\bwho\s+(is|was|are|were|is current|current)\b", s))
        
        # Pattern 2: Contact/email/address queries (must be specific to people/orgs)
        # Examples: "email of secretary", "contact for director", "phone number of DDG"
        contact_keywords = r"\b(email|e-mail|mail|contact|phone|telephone|address)\b"
        has_contact_keyword = bool(re.search(contact_keywords, s))
        
        # Pattern 3: "what is the email/contact/address" queries
        is_what_contact = bool(re.search(r"\bwhat('s|s| is| are)\b.*" + contact_keywords, s))
        
        # Pattern 4: Direct contact queries like "rohit's email"
        is_possessive_contact = bool(re.search(r"\b\w+'s\s+(email|e-mail|mail|contact|phone|address)\b", s))
        
        # Pattern 5: "where is MoSPI office" type queries
        is_location_query = bool(re.search(r"\bwhere\s+(is|are)\b.*\b(mospi|ministry|office)\b", s))
        
        # Pattern 6: Post + Division queries (e.g., "adg diid", "dg mospi", "secretary ministry")
        # These are short queries with designation + division/org, no other words
        # Note: RO, ZO, SRO are FOD office types (divisions), not designations
        designation_abbr = r"\b(ddgs?|adgs?|dgs?|js|as|secretar(?:y|ies)|directors?|minister|mos|cvo|deputy director generals?|additional director generals?|director generals?|joint secretar(?:y|ies)|additional secretar(?:y|ies)|deputy directors?|assistant directors?|joint directors?|chief vigilance officer)\b"
        division_org = r"\b(diid|fod|nad|esd|ssd|psd|hsd|hsu|ensd|cqcd|cicd|aspd|cdd|nss|nso|nssta|mospi|ministry|data informatics|field operations|national accounts|economic statistics|social statistics|price statistics|household survey|household survey unit|enterprise survey|jaipur|kolkata|mumbai|chennai|bangalore|bengaluru|hyderabad|guwahati|lucknow|bhopal|chandigarh|patna|ahmedabad|pune|delhi|faridabad|dehradun|nagpur|north zone|east zone|west zone|south zone|central zone|northeast zone|north east zone|zo|ro|sro|zonal office|regional office|sub-regional office)\b"

        # Pattern 6a: Generic role-noun + division/org (e.g., "officials of diid", "officers in fod",
        # "staff of hsu"). Complements Pattern 6, which requires a specific designation like DDG/ADG.
        role_noun = r"\b(officials?|officers?|staff|employees?|personnel|team|people)\b"

        # Match: "adg diid", "dg mospi", "secretary ministry" (with optional "of", "in", "at")
        # Fixed: Use \s* before optional preposition to handle both "ADG DIID" and "ADG of DIID"
        is_post_division_query = bool(re.search(
            rf"{designation_abbr}\s+(of|in|at|for)\s+{division_org}", s  # With preposition: "ADG of DIID"
        )) or bool(re.search(
            rf"{designation_abbr}\s+{division_org}", s  # Without preposition: "ADG DIID"
        )) or bool(re.search(
            rf"{division_org}\s+{designation_abbr}", s  # Reverse order: "DIID ADG"
        )) or bool(re.search(
            rf"{role_noun}\s+(of|in|at|for)\s+{division_org}", s  # "officials of diid"
        )) or bool(re.search(
            rf"{division_org}\s+{role_noun}", s  # "diid officials"
        ))

        # Pattern 6b: Division/org + preposition + trailing word (e.g., "hsu for uttarakhand",
        # "fod in jaipur"). Catches location-qualified directory queries that have no designation
        # or role noun at all.
        is_division_location_query = bool(re.search(
            rf"{division_org}\s+(of|in|at|for)\s+\w+", s
        ))

        # Generic cache-backed directory detection.  This handles every current
        # division in whois_cache, including punctuation and "&"/"and" variants,
        # without adding a separate hard-coded route for each department.
        has_designation = bool(re.search(designation_abbr, s))
        # A designation list has no person or division to identify, but it is
        # still a Who's Who request (for example, "list of DDG" or
        # "all Deputy Director Generals"). Route it to the exclusive cache.
        is_designation_list_query = has_designation and bool(re.search(
            r"\b(?:list|all)\b",
            s,
        ))
        has_directory_request = bool(re.search(
            r"\b(who|list|all|officials?|officers?|staff|employees?|contact|email|phone|address)\b",
            s,
        ))
        is_cached_division_directory_query = (
            (has_designation or has_directory_request)
            and _query_mentions_cached_whois_division(s)
        )
        is_cached_officer_directory_query = bool(re.search(
            r"\b(who|designation|position|details|email|e-mail|mail|contact|phone|telephone|address)\b",
            s,
        )) and _query_mentions_cached_whois_officer(s)

        # Pattern 7: Count queries (e.g., "how many ddg", "number of adg in diid")
        count_keywords = r"\b(how\s+many|number\s+of|total|count|list\s+all)\b"
        has_count_keyword = bool(re.search(count_keywords, s))
        
        # Count query must have designation or role
        is_count_query = has_count_keyword and bool(re.search(designation_abbr, s))
        
        # NEW: Return True for post+division, division+location, or count queries
        if (
            is_post_division_query
            or is_division_location_query
            or is_count_query
            or is_designation_list_query
            or is_cached_division_directory_query
            or is_cached_officer_directory_query
        ):
            return True
        
        # For contact queries (without "who"), require specific MoSPI context
        if (has_contact_keyword or is_what_contact or is_possessive_contact or is_location_query) and not is_who_question:
            # Must have MoSPI-specific org indicators
            org_indicators = ["mospi", "diid", "ministry of statistics", "ministry of statistics and programme implementation", "nso", "nss"]
            if any(o in s for o in org_indicators):
                return True
            
            # Or must have person titles
            if re.search(r"\b(shri|smt|dr|ms|mr)\b", s):
                return True
            
            # Or must have role/title keywords
            role_keywords = ["secretary", "director", "ddg", "adg", "officer", "minister", "commissioner", "chairperson"]
            if any(role in s for role in role_keywords):
                return True
            
            # Otherwise, not a whois query (e.g., "date of survey" has "of" but not whois)
            return False
        
        # For "who" questions, check for role/org context
        if is_who_question:
            # Role/title keywords (includes common short acronyms like DDG)
            roles = [
                "ddg",
                "deputy director general",
                "director general",
                "director",
                "directors",
                "secretary",
                "commissioner",
                "chairperson",
                "cvo",
                "chief vigilance officer",
                "who is who",
                "who's who",
                "who",
                "who is",
                "adg",
            ]
            org_indicators = ["mospi", "diid", "ministry of statistics", "ministry of statistics and programme implementation"]
            if any(r in s for r in roles) or any(o in s for o in org_indicators):
                return True
            # Fallback: short Who-queries (e.g., "who is ddg", "who is the cs")
            toks = s.split()
            if toks and toks[0] == "who" and len(toks) <= 6:
                return True

        # Pattern 8: Bare officer-name query (e.g., "kedar nath verma") - no "who"/contact/role
        # keywords at all, just a short sequence of name-like words. Only worth the cache lookup
        # when the query looks like a plain name (short, alphabetic only, no digits/punctuation
        # beyond spaces/periods) to avoid doing this check on every unrelated query.
        if re.fullmatch(r"[a-z][a-z\.\s]*", s):
            toks = [t for t in s.replace(".", " ").split() if t]
            if 2 <= len(toks) <= 5 and _name_matches_whois_cache(set(toks)):
                return True

        return False


    # 4b) Visualization hint: downstream can fetch embeds from a dedicated collection
    if cfg.get("visualization", True) and is_visualization_query(query_en):
        hints["visualization"] = True
        route_names.append("visualization_hint")

    # 4c) Who-is/topic hint: boost 'who is who' topical chunks
    if cfg.get("enabled", True) and _is_whois_query(query_en):
        existing = hints.get("topics_boost") or []
        if "who is who" not in existing:
            existing = existing + ["who is who"]
        hints["topics_boost"] = existing
        # Stronger hint: ask retriever to prefer documents whose filename/doc_name/s3_key
        # indicates a "who's who" file (non-invasive promotion in retrieval layer).
        hints["whois_prefer_file"] = True
        route_names.append("whois_hint")

    # 4d) NSS round hint-only (soft influence)
    if cfg.get("nss", True):
        if is_nss_round_query(query_en):
            n = extract_nss_round(query_en)
            if n:
                hints["topics_boost"] = ["nss", f"round {n}"]
                route_names.append("nss_round_hint")

    if hints:
        return RouteResult(retrieval_hints=hints, route_name="|".join(route_names) or "hints")

    return RouteResult()


# ─────────────────────────────────────────────────────────────────────────────
# CADENCE-AWARE METRIC ROUTING
# ─────────────────────────────────────────────────────────────────────────────
# For metrics with known release cadences (CPI, IIP, GDP), this module provides
# intelligent query rewriting and date-range hints for retrieval.
#
# Usage:
#   result = get_metric_cadence_hints(query_en)
#   if result:
#       # result contains rewritten_query, date_ranges, metric_name
#       pass to retrieval with date filtering
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricCadence:
    """Configuration for a metric's release cadence."""
    name: str                          # e.g., "CPI", "GDP"
    cadence: str                       # "monthly" or "quarterly"
    keywords: List[str]                # Keywords to detect this metric
    release_day_start: int = 12        # Typical release day (start)
    release_day_end: int = 15          # Typical release day (end, for holidays)
    data_lag_months: int = 1           # How many months behind is the data
    lookback_periods: int = 3          # How many periods to look back


# ─────────────────────────────
# CONFIGURABLE METRIC REGISTRY
# Add new metrics here with their cadence
# ─────────────────────────────
METRIC_CADENCE_REGISTRY: Dict[str, MetricCadence] = {
    "cpi": MetricCadence(
        name="CPI",
        cadence="monthly",
        keywords=["cpi","cfpi", "consumer price index", "inflation", "price index", "inflation rate", 
                  "retail inflation", "headline inflation", "core inflation", "food inflation"],
        release_day_start=1,
        release_day_end=31,
        data_lag_months=1,
        lookback_periods=6,  # Changed from 3 to 6 for better coverage
    ),
    "iip": MetricCadence(
        name="IIP",
        cadence="monthly",
        keywords=["iip", "industrial production", "index of industrial production", "industrial output",
                  "manufacturing output", "factory output", "production index", "industrial growth"],
        release_day_start=1,
        release_day_end=31,
        data_lag_months=1,
        lookback_periods=6,  # Changed from 3 to 6 for better coverage
    ),
    "gdp": MetricCadence(
        name="GDP",
        cadence="quarterly",
        keywords=["gdp", "gross domestic product", "gdp growth", "gdp data", "gdp estimate",
                  "economic growth", "national income", "national accounts", "gva", "gross value added",
                  "quarterly estimate", "annual estimate", "advance estimate"],
        release_day_start=1,
        release_day_end=31,
        data_lag_months=2,  # GDP typically released ~2 months after quarter end
        lookback_periods=6,  # Changed from 3 to 6 for better coverage (6 quarters = 1.5 years)
    ),
    "plfs": MetricCadence(
        name="PLFS",
        cadence="monthly",  # Changed from "quarterly" - PLFS has BOTH monthly and quarterly releases
        keywords=["plfs", "periodic labour force survey", "periodic labor force survey",
                  "labour force", "labor force", "employment", "unemployment",
                  "worker population ratio", "wpr", "labour force participation rate", "labor force participation rate", "lfpr",
                  "unemployment rate", "employment rate", "jobless", "joblessness", "workforce",
                  "current weekly status", "cws", "current daily status", "cds",
                  "usual principal status", "usual status", "current status"],
        release_day_start=1,
        release_day_end=31,
        data_lag_months=1,  # Changed from 2 - Monthly bulletins released ~1 month after data month
        lookback_periods=6,  # Changed from 3 to 6 for better coverage
    ),
    "asi": MetricCadence(
        name="ASI",
        cadence="annual",
        keywords=["asi", "annual survey of industries", "factory sector", "organized manufacturing",
                  "registered factories", "industrial statistics", "factory statistics"],
        release_day_start=1,
        release_day_end=31,
        data_lag_months=12,  # ASI released annually with significant lag
        lookback_periods=3,
    ),
    "nss": MetricCadence(
        name="NSS",
        cadence="periodic",  # NSS rounds are periodic, not regular monthly/quarterly
        keywords=["nss", "national sample survey", "nsso", 
                  "national sample survey organisation", "national sample survey organization",
                  "household survey", "consumption expenditure", "household consumption expenditure"],
        release_day_start=1,
        release_day_end=31,
        data_lag_months=6,  # NSS rounds have variable release schedules
        lookback_periods=2,
    ),
    "ec": MetricCadence(
        name="EC",
        cadence="periodic",  # Economic Census is conducted every 5 years
        keywords=["economic census", "establishment survey", "enterprise census", "business census"],
        release_day_start=1,
        release_day_end=31,
        data_lag_months=12,
        lookback_periods=2,
    ),
    "hces": MetricCadence(
        name="HCES",
        cadence="periodic",  # HCES rounds are periodic (every few years)
        keywords=["hces", "household consumer expenditure survey", "household consumption expenditure",
                  "household expenditure", "consumer expenditure survey", "consumption expenditure",
                  "household survey", "household consumption", "consumer spending",
                  "households surveyed", "household sample"],
        release_day_start=1,
        release_day_end=31,
        data_lag_months=6,
        lookback_periods=2,
    ),
    "iss": MetricCadence(
        name="ISS",
        cadence="periodic",  # ISS orders/documents are released as needed (no fixed cadence)
        keywords=["iss", "indian statistical service", 
                  "promotion order", "transfer order", "office order", 
                  "vacancy position", "posting order", "appointment order",
                  "confirmation order", "deputation order"],
        release_day_start=1,
        release_day_end=31,
        data_lag_months=0,  # ISS orders are immediate/current
        lookback_periods=6,  # Look back 6 months for recent orders
    ),
}

# Keywords that indicate user wants latest/current data
LATEST_KEYWORDS = [
    "latest", "current", "recent", "now", "today", "this month", "this quarter",
    "newest", "most recent", "up to date", "updated"
]


def is_definition_query(query: str) -> bool:
    """
    VERY RESTRICTIVE: Only detect pure definition queries.
    When in doubt, treat as data query (safer for business logic).
    
    Will ONLY match:
    - "what is CPI"
    - "define GDP"
    - "meaning of IIP"
    
    Will NOT match (treated as data queries):
    - "what is CPI data"
    - "what is the latest CPI"
    - "what is CPI for 2025"
    - Anything with data context
    """
    q = query.lower().strip()
    
    # FIRST: Check for ANY data indicators (immediate disqualification)
    data_indicators = [
        'data', 'rate', 'value', 'number', 'figure', 'statistics', 'stats',
        'latest', 'current', 'recent', 'now', 'today', 'this',
        'for', 'in', 'of', 'from', 'to', 'between',  # Prepositions indicating data context
        r'20\d{2}',  # Any year (2000-2099)
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'q1', 'q2', 'q3', 'q4', 'quarter', 'quarterly', 'monthly', 'annual',
        'growth', 'increase', 'decrease', 'change', 'trend', 'rise', 'fall',
        'high', 'low', 'percent', '%', 'percentage'
    ]
    
    for indicator in data_indicators:
        if re.search(rf'\b{indicator}\b', q):
            return False
    
    # SECOND: Very restrictive patterns (must match exactly)
    # Only 2-3 word queries like "what is CPI" or "define GDP"
    pure_definition_patterns = [
        r'^what is [a-z]{2,10}\??$',      # "what is CPI?" (2-10 letter acronym)
        r'^what are [a-z]{2,10}\??$',     # "what are NSS?"
        r'^define [a-z]{2,10}\??$',       # "define GDP?"
        r'^meaning of [a-z]{2,10}\??$',   # "meaning of IIP?"
        r'^explain [a-z]{2,10}\??$',      # "explain PLFS?"
        r'^what does [a-z]{2,10} mean\??$',      # "what does CPI mean?"
        r'^what does [a-z]{2,10} stand for\??$', # "what does GDP stand for?"
        r'^full form of [a-z]{2,10}\??$',        # "full form of IIP?"
    ]
    
    # Must match at least one pattern
    matches_pattern = any(re.match(pattern, q) for pattern in pure_definition_patterns)
    
    if not matches_pattern:
        return False
    
    # THIRD: Additional safety check - word count
    # Pure definition queries should be very short (2-5 words max)
    words = q.replace('?', '').split()
    if len(words) > 5:
        return False
    
    return True


def has_specific_time_mention(query: str) -> bool:
    """
    Check if query mentions specific month/year/quarter.
    
    Examples:
    - "CPI for october 2025" → True
    - "GDP in Q1 2025" → True
    - "IIP data 2024" → True
    - "latest CPI" → False (no specific time)
    """
    q = query.lower()
    
    # Month names
    month_names = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ]
    has_month = any(month in q for month in month_names)
    
    # Year pattern
    has_year = bool(re.search(r'\b20\d{2}\b', q))
    
    # Quarter pattern
    has_quarter = bool(re.search(r'\bq[1-4]\b', q))
    
    return (has_month and has_year) or has_quarter or has_year


def _extract_specific_time_from_query(query: str) -> Dict[str, Any]:
    """
    Extract specific month/year/quarter from query.
    
    Returns dict with:
        - month: int (1-12) if month detected
        - year: int if year detected
        - quarter: int (1-4) if quarter detected
        - month_range: tuple (start_month, end_month) if month range detected
    """
    q = query.lower()
    result = {}
    
    # Extract year
    year_match = re.search(r'\b(20\d{2})\b', query)
    if year_match:
        result['year'] = int(year_match.group(1))
    
    # Check for month ranges (e.g., "july - september", "july to september", "july–september")
    month_names_full = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    month_names_abbr = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
        'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    # Pattern: "july - september", "july to september", "july–september", "jul-sep"
    for start_name, start_num in {**month_names_full, **month_names_abbr}.items():
        for end_name, end_num in {**month_names_full, **month_names_abbr}.items():
            # Try various separators: -, –, to, space
            range_patterns = [
                rf'\b{start_name}\s*[-–]\s*{end_name}\b',
                rf'\b{start_name}\s+to\s+{end_name}\b',
                rf'\b{start_name}\s*[-–/]\s*{end_name}\b',
            ]
            
            for pattern in range_patterns:
                if re.search(pattern, q):
                    result['month_range'] = (start_num, end_num)
                    result['start_month'] = start_num
                    result['end_month'] = end_num
                    
                    # Map month range to quarter if it matches Indian FY quarters
                    if (start_num, end_num) == (4, 6):
                        result['quarter'] = 1  # Q1: Apr-Jun
                    elif (start_num, end_num) == (7, 9):
                        result['quarter'] = 2  # Q2: Jul-Sep
                    elif (start_num, end_num) == (10, 12):
                        result['quarter'] = 3  # Q3: Oct-Dec
                    elif (start_num, end_num) == (1, 3):
                        result['quarter'] = 4  # Q4: Jan-Mar
                    
                    return result  # Return early if range found
    
    # Extract single month (only if no range found)
    if 'month_range' not in result:
        for month_name, month_num in month_names_full.items():
            if month_name in q:
                result['month'] = month_num
                break
        
        # Try abbreviations if full name not found
        if 'month' not in result:
            for month_abbr, month_num in month_names_abbr.items():
                if month_abbr in q:
                    result['month'] = month_num
                    break
    
    # Extract quarter (e.g., "Q1", "q2", "quarter 3")
    if 'quarter' not in result:
        quarter_match = re.search(r'\bq([1-4])\b', q)
        if quarter_match:
            result['quarter'] = int(quarter_match.group(1))
    
    return result


def _map_month_to_quarter_fy(year: int, month: int) -> Tuple[int, int, str]:
    """
    Map a specific month to Indian FY quarter.
    
    Indian FY: April to March
    Q1: Apr-Jun, Q2: Jul-Sep, Q3: Oct-Dec, Q4: Jan-Mar
    
    Args:
        year: Year of the month
        month: Month (1-12)
    
    Returns:
        (fy_year, quarter_num, quarter_label)
        
    Example:
        _map_month_to_quarter_fy(2025, 10) → (2025, 3, "Q3 FY 2025-26")
    """
    if month >= 4 and month <= 6:
        return year, 1, f"Q1 FY {year}-{str(year+1)[-2:]}"
    elif month >= 7 and month <= 9:
        return year, 2, f"Q2 FY {year}-{str(year+1)[-2:]}"
    elif month >= 10 and month <= 12:
        return year, 3, f"Q3 FY {year}-{str(year+1)[-2:]}"
    else:  # Jan-Mar
        return year-1, 4, f"Q4 FY {year-1}-{str(year)[-2:]}"


def _get_quarterly_date_ranges_for_specific(
    now: datetime,
    metric: MetricCadence,
    fy_year: int,
    quarter: int
) -> List[Dict[str, Any]]:
    """
    Generate date ranges for a specific quarter.
    
    Args:
        now: Current datetime
        metric: Metric configuration
        fy_year: FY year
        quarter: Quarter number (1-4)
    
    Returns:
        List of date range dicts
    """
    ranges = []
    
    # Get quarter details
    if quarter == 1:
        start_month, end_month = 4, 6
        release_start_month = 8  # Aug
    elif quarter == 2:
        start_month, end_month = 7, 9
        release_start_month = 11  # Nov
    elif quarter == 3:
        start_month, end_month = 10, 12
        release_start_month = 2  # Feb (next year)
    else:  # Q4
        start_month, end_month = 1, 3
        release_start_month = 5  # May
    
    # Calculate release year
    if quarter == 3:
        release_year = fy_year + 1
    elif quarter == 4:
        release_year = fy_year + 1
    else:
        release_year = fy_year
    
    # Date range for publish_date
    start_date = datetime(release_year, release_start_month, 1)
    end_date = start_date + relativedelta(months=2)  # 2 month window
    
    fy_label = f"FY {fy_year}-{str(fy_year + 1)[-2:]}"
    quarter_label = f"Q{quarter} {fy_label}"
    
    ranges.append({
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "period_label": quarter_label,
        "quarter": quarter,
        "fy_year": fy_year,
        "priority": 1,
    })
    
    return ranges


def _generate_date_ranges_from_time_info(
    now: datetime,
    metric: MetricCadence,
    time_info: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generate date ranges based on extracted time info.
    
    Args:
        now: Current datetime
        metric: Metric configuration
        time_info: Dict with month/year/quarter
    
    Returns:
        List of date range dicts
    """
    ranges = []
    
    if 'quarter' in time_info and 'year' in time_info:
        # Quarterly data requested
        quarter = time_info['quarter']
        year = time_info['year']
        
        # Map to FY quarter
        if quarter == 1:
            fy_year = year
            data_months = [4, 5, 6]
        elif quarter == 2:
            fy_year = year
            data_months = [7, 8, 9]
        elif quarter == 3:
            fy_year = year
            data_months = [10, 11, 12]
        else:  # Q4
            fy_year = year - 1
            data_months = [1, 2, 3]
        
        for i, month in enumerate(data_months):
            month_year = year if month >= 4 else year
            ranges.append({
                "data_month": month,
                "data_year": month_year,
                "period_label": f"{datetime(month_year, month, 1).strftime('%B')} {month_year}",
                "priority": i + 1
            })
    
    elif 'month' in time_info and 'year' in time_info:
        # Specific month requested
        # NEW LOGIC: Prioritize month+2 (final data) over requested month
        # Example: User asks "May 2025" → Search Mar, Apr, May, Jun, Jul
        # Priority: Jul(1), Jun(2), May(3), Apr(4), Mar(5)
        # Reasoning: May's final data released in July PDF (provisional in June)
        month = time_info['month']
        year = time_info['year']
        
        base_date = datetime(year, month, 1)
        # Range: -2 to +2 (5 months: 2 before, requested, 2 after)
        for i in range(-2, 3):  # [-2, -1, 0, 1, 2]
            date = base_date + relativedelta(months=i)
            # Priority mapping: month+2 gets priority 1 (final data)
            priority_map = {2: 1, 1: 2, 0: 3, -1: 4, -2: 5}
            ranges.append({
                "data_month": date.month,
                "data_year": date.year,
                "period_label": f"{date.strftime('%B')} {date.year}",
                "priority": priority_map[i]
            })
    
    elif 'year' in time_info:
        # Year only - use December of that year
        year = time_info['year']
        base_date = datetime(year, 12, 1)
        
        for i in range(-2, 2):  # 2 before Dec, Dec, 1 after
            date = base_date + relativedelta(months=i)
            ranges.append({
                "data_month": date.month,
                "data_year": date.year,
                "period_label": f"{date.strftime('%B')} {date.year}",
                "priority": abs(i) + 1
            })
    
    return ranges


@dataclass
class MetricCadenceResult:
    """Result from cadence-aware routing."""
    metric_name: str
    rewritten_query: str
    date_ranges: List[Dict[str, Any]]  # List of {start_date, end_date, period_label}
    original_query: str
    cadence: str  # "monthly" or "quarterly"


def _get_indian_fy_quarter(dt: datetime) -> tuple:
    """
    Get Indian Financial Year quarter for a given date.
    Indian FY: April to March
    Q1: Apr-Jun, Q2: Jul-Sep, Q3: Oct-Dec, Q4: Jan-Mar
    
    Returns: (fy_year, quarter_num, quarter_label, start_month, end_month)
    """
    month = dt.month
    year = dt.year
    
    if month >= 4 and month <= 6:
        fy_year = year
        quarter = 1
        start_month, end_month = 4, 6
    elif month >= 7 and month <= 9:
        fy_year = year
        quarter = 2
        start_month, end_month = 7, 9
    elif month >= 10 and month <= 12:
        fy_year = year
        quarter = 3
        start_month, end_month = 10, 12
    else:  # Jan-Mar
        fy_year = year - 1
        quarter = 4
        start_month, end_month = 1, 3
    
    fy_label = f"FY {fy_year}-{str(fy_year + 1)[-2:]}"
    quarter_label = f"Q{quarter} {fy_label}"
    
    return fy_year, quarter, quarter_label, start_month, end_month


def _extract_document_type_keywords(query: str) -> str:
    """
    Extract document type keywords from the query to preserve in rewritten query.
    
    Examples:
        "latest ISS promotion order" -> "promotion order"
        "ISS transfer order" -> "transfer order"
        "ISS posting notification" -> "posting notification"
        "ISS circular" -> "circular"
        "latest CPI data" -> ""
    
    Returns:
        Document type phrase (empty string if none found)
    """
    query_lower = query.lower()
    
    # Define document type patterns (order matters - check longer phrases first)
    doc_type_patterns = [
        # Multi-word patterns
        (r'\b(promotion\s+order|promo\s+order)s?\b', 'promotion order'),
        (r'\b(transfer\s+order)s?\b', 'transfer order'),
        (r'\b(posting\s+order)s?\b', 'posting order'),
        (r'\b(appointment\s+order)s?\b', 'appointment order'),
        (r'\b(deputation\s+order)s?\b', 'deputation order'),
        (r'\b(office\s+order)s?\b', 'office order'),
        (r'\b(vacancy\s+position)s?\b', 'vacancy position'),
        (r'\b(seniority\s+list)s?\b', 'seniority list'),
        (r'\b(civil\s+list)s?\b', 'civil list'),
        (r'\b(gazette\s+notification)s?\b', 'gazette notification'),
        # Single-word patterns
        (r'\b(promotion)s?\b', 'promotion'),
        (r'\b(transfer)s?\b', 'transfer'),
        (r'\b(posting)s?\b', 'posting'),
        (r'\b(circular)s?\b', 'circular'),
        (r'\b(notification)s?\b', 'notification'),
        (r'\b(order)s?\b', 'order'),
        (r'\b(announcement)s?\b', 'announcement'),
    ]
    
    for pattern, keyword in doc_type_patterns:
        if re.search(pattern, query_lower):
            return keyword
    
    return ""


def _get_monthly_date_ranges(now: datetime, metric: MetricCadence) -> List[Dict[str, Any]]:
    """
    Generate date ranges for monthly metrics.
    
    For a metric released on 12th-15th of each month for previous month's data:
    - If today is Dec 30, latest available would be Nov data (released Dec 12-15)
    - Look back: Nov, Oct, Sep...
    """
    ranges = []
    
    # Start from current month and go back
    for i in range(metric.lookback_periods):
        # Calculate the data month (accounting for lag)
        data_date = now - relativedelta(months=metric.data_lag_months + i)
        data_month = data_date.month
        data_year = data_date.year
        
        # The release window is in the following month
        release_date = data_date + relativedelta(months=1)
        
        # Date range for publish_date: release window
        # Handle case where release_day_end exceeds days in month (e.g., Feb 31)
        import calendar
        max_day_in_month = calendar.monthrange(release_date.year, release_date.month)[1]
        actual_release_day_end = min(metric.release_day_end, max_day_in_month)
        
        start_date = datetime(release_date.year, release_date.month, metric.release_day_start)
        end_date = datetime(release_date.year, release_date.month, actual_release_day_end) + timedelta(days=5)  # Buffer
        
        month_name = data_date.strftime("%B")
        period_label = f"{month_name} {data_year}"
        
        ranges.append({
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "period_label": period_label,
            "data_month": data_month,
            "data_year": data_year,
            "priority": i + 1,  # 1 = most recent
        })
    
    return ranges


def _get_quarterly_date_ranges(now: datetime, metric: MetricCadence) -> List[Dict[str, Any]]:
    """
    Generate date ranges for quarterly metrics (GDP).
    
    Indian FY quarters and their typical release windows:
    - Q1 (Apr-Jun) data released around Aug-Sep
    - Q2 (Jul-Sep) data released around Nov-Dec
    - Q3 (Oct-Dec) data released around Feb-Mar
    - Q4 (Jan-Mar) data released around May-Jun
    """
    ranges = []
    
    # Get current quarter
    fy_year, current_q, _, _, _ = _get_indian_fy_quarter(now)
    
    for i in range(metric.lookback_periods):
        # Calculate target quarter (going backwards)
        target_q = current_q - i
        target_fy = fy_year
        
        while target_q <= 0:
            target_q += 4
            target_fy -= 1
        
        # Get quarter details
        if target_q == 1:
            start_month, end_month = 4, 6
            release_start_month = 8  # Aug
        elif target_q == 2:
            start_month, end_month = 7, 9
            release_start_month = 11  # Nov
        elif target_q == 3:
            start_month, end_month = 10, 12
            release_start_month = 2  # Feb (next year)
        else:  # Q4
            start_month, end_month = 1, 3
            release_start_month = 5  # May
        
        # Calculate release year
        if target_q == 3:
            release_year = target_fy + 1
        elif target_q == 4:
            release_year = target_fy + 1
        else:
            release_year = target_fy
        
        # Date range for publish_date
        start_date = datetime(release_year, release_start_month, 1)
        end_date = start_date + relativedelta(months=2)  # 2 month window
        
        fy_label = f"FY {target_fy}-{str(target_fy + 1)[-2:]}"
        quarter_label = f"Q{target_q} {fy_label}"
        
        ranges.append({
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "period_label": quarter_label,
            "quarter": target_q,
            "fy_year": target_fy,
            "priority": i + 1,
        })
    
    return ranges


def detect_metric_in_query(query: str) -> Optional[str]:
    """Detect which metric (if any) is mentioned in the query."""
    q = (query or "").lower()
    
    for metric_key, metric in METRIC_CADENCE_REGISTRY.items():
        for kw in metric.keywords:
            if re.search(rf"\b{re.escape(kw)}\b", q):
                return metric_key
    
    return None


def has_latest_keyword(query: str) -> bool:
    """Check if query contains keywords indicating user wants latest data."""
    q = (query or "").lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", q) for kw in LATEST_KEYWORDS)


def get_metric_cadence_hints(query: str, reference_date: Optional[datetime] = None) -> Optional[MetricCadenceResult]:
    """
    Main entry point for cadence-aware routing.
    
    Enhanced to handle:
    1. Latest queries (existing) - "latest CPI"
    2. Specific month/year queries (NEW) - "CPI for october 2025"
    3. Quarter queries (NEW) - "GDP Q1 2025"
    4. Definition queries (SKIP) - "what is CPI"
    
    Detects if query is asking for data of a known metric,
    and returns rewritten query + date ranges for retrieval.
    
    Args:
        query: User query (English)
        reference_date: Optional date to use as "now" (for testing)
    
    Returns:
        MetricCadenceResult if applicable, None otherwise
    """
    if not query:
        return None
    
    # STEP 1: Skip definition queries (ultra-restrictive)
    if is_definition_query(query):
        return None
    
    # STEP 2: Detect metric
    metric_key = detect_metric_in_query(query)
    if not metric_key:
        return None
    
    metric = METRIC_CADENCE_REGISTRY[metric_key]
    now = reference_date or datetime.now()
    
    # STEP 3: Check for latest keywords OR specific time mentions
    has_latest = has_latest_keyword(query)
    has_specific_time = has_specific_time_mention(query)
    
    if not (has_latest or has_specific_time):
        # No time context at all - don't trigger cadence routing
        return None
    
    # STEP 4: Generate date ranges based on query type
    date_ranges = []
    
    if has_specific_time and not has_latest:
        # User specified exact time - use that
        time_info = _extract_specific_time_from_query(query)
        
        # Check if user explicitly asked for quarterly data (even for monthly metrics)
        if 'quarter' in time_info and 'year' in time_info:
            # User asked for a quarter - generate quarterly date ranges
            quarter = time_info['quarter']
            year = time_info['year']
            # Determine FY year from quarter
            if quarter == 4:
                fy_year = year - 1
            else:
                fy_year = year
            date_ranges = _get_quarterly_date_ranges_for_specific(now, metric, fy_year, quarter)
        
        # For quarterly metrics ONLY, map month to quarter if month specified
        # For monthly metrics, keep the month as-is (don't map to quarter)
        elif metric.cadence == "quarterly" and 'month' in time_info and 'year' in time_info:
            fy_year, quarter, quarter_label = _map_month_to_quarter_fy(
                time_info['year'], time_info['month']
            )
            date_ranges = _get_quarterly_date_ranges_for_specific(now, metric, fy_year, quarter)
        
        else:
            # Monthly metric or year-only query - use month directly
            date_ranges = _generate_date_ranges_from_time_info(now, metric, time_info)
    
    else:
        # "Latest" query - use existing cadence logic
        if metric.cadence == "monthly":
            date_ranges = _get_monthly_date_ranges(now, metric)
        elif metric.cadence == "quarterly":
            date_ranges = _get_quarterly_date_ranges(now, metric)
        elif metric.cadence == "periodic":
            # Periodic metrics (ISS, NSS, EC, HCES) - treat like monthly
            # Look back N months based on lookback_periods
            date_ranges = _get_monthly_date_ranges(now, metric)
        else:
            return None
    
    if not date_ranges:
        return None
    
    # STEP 5: Build rewritten query
    # Only rewrite if user query lacks explicit time context
    # If user already specified time (month, quarter, year), keep original query
    
    # Extract document type keywords to preserve in rewritten query
    doc_type_keywords = _extract_document_type_keywords(query)
    
    if has_specific_time:
        # User has explicit time context - don't rewrite, use original query
        rewritten = query
    else:
        # User query lacks time context (e.g., "latest CPI") - add time context
        period_labels = [r["period_label"] for r in date_ranges[:2]]  # Top 2 periods
        
        # Build base query with time context
        if len(period_labels) >= 2:
            base_query = f"{metric.name} data for {period_labels[0]} or {period_labels[1]} if available"
        elif len(period_labels) == 1:
            base_query = f"{metric.name} data for {period_labels[0]}"
        else:
            base_query = query
        
        # Preserve document type keywords if present
        if doc_type_keywords:
            rewritten = f"{metric.name} {doc_type_keywords} for {period_labels[0]} or {period_labels[1]} if available" if len(period_labels) >= 2 else f"{metric.name} {doc_type_keywords} for {period_labels[0]}"
        else:
            rewritten = base_query
    
    return MetricCadenceResult(
        metric_name=metric.name,
        rewritten_query=rewritten,
        date_ranges=date_ranges,
        original_query=query,
        cadence=metric.cadence,
    )


def apply_cadence_routing(query_en: str, is_hindi: bool = False) -> RouteResult:
    """
    Apply cadence-aware routing for metric queries.
    
    This doesn't short-circuit (no response), but provides retrieval hints
    with date ranges for filtering.
    
    Returns RouteResult with retrieval_hints containing:
        - metric_cadence: MetricCadenceResult
        - date_filter_ranges: List of date ranges to try
        - rewritten_query: Query with time context
    """
    result = get_metric_cadence_hints(query_en)
    
    if not result:
        return RouteResult()
    
    hints = {
        "metric_cadence": {
            "metric_name": result.metric_name,
            "cadence": result.cadence,
            "date_ranges": result.date_ranges,
            "rewritten_query": result.rewritten_query,
        }
    }
    
    return RouteResult(
        retrieval_hints=hints,
        route_name=f"cadence_{result.metric_name.lower()}"
    )
