from nltk.tokenize import RegexpTokenizer, word_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from country_list import countries_for_language

# ─────────────────────────────
# LangChain Libraries
# ─────────────────────────────
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

# ─────────────────────────────
# Qdran
"""
Qdrant-powered chatbot handler
Migrated from Chroma to Qdrant with full feature parity
All core logic unchanged, only vector DB layer replaced
"""

# ─────────────────────────────
# Standard Library Imports
# ─────────────────────────────
import asyncio
import re
import os
import math
import json
import logging
import time
import difflib
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Dict, List, Tuple, Set, Optional, Any
from urllib.parse import quote

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def ist_now() -> datetime:
    """Get current datetime in IST timezone"""
    return datetime.now(IST)

# ─────────────────────────────
# Third-Party Libraries
# ─────────────────────────────
import nltk
import torch
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from langdetect import detect
#from nltk.tokenize impot Imports (CHANGED)
# ─────────────────────────────
from qdrant_vector_store import (
    build_qdrant_store,
    list_doc_names_qdrant,
    delete_chunks_for_doc_qdrant,
    update_doc_metadata_qdrant,
)
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ─────────────────────────────
# Project-Specific Imports
# ─────────────────────────────
from models import Interaction
from pred_res import PREDEFINED_RESPONSES
from session_store import get_session_store, cleanup_expired_sessions_redis
from analytics import AnalyticsService

# ─────────────────────────────
# Retrieval V2 (New Simplified Retrieval)
# ─────────────────────────────
from retrieval_v2 import retrieve_priority_chunks_v2

from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file\

# nltk.download("punkt", quiet=True)

# ========== Logging Setup ==========
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/chatbot.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("chatbot")

# Configure logging
def setup_logging(log_dir="logs"):
    """Setup logging to both file and console"""
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log filename with date
    log_filename = os.path.join(log_dir, f"chatbot_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Create logger
    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler - logs everything
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler - logs INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_filename}")
    
    return logger

# Initialize logger
logger = setup_logging()


SOURCES_DEBUG_MAX_DOCS = 5
DEBUG_SOURCES = True  # keep existing debug gates in code


ENABLE_RETRIEVAL_DEBUG = os.getenv("ENABLE_RETRIEVAL_DEBUG", "1").strip().lower() in ("1", "true", "yes", "y", "on")
RETRIEVAL_DEBUG_MAX_DOCS = int(os.getenv("RETRIEVAL_DEBUG_MAX_DOCS", "20"))  # Max docs to log in detail


ENABLE_METRIC_FILTERING = os.getenv("ENABLE_METRIC_FILTERING", "0").strip().lower() in ("1", "true", "yes", "y", "on")


RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large").strip()


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mospi_collection_v2")
QDRANT_COLLECTION = "mospi_collection_bge_v3"
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")
INFO_COLLECTION_NAME = QDRANT_COLLECTION
VIS_COLLECTION_NAME = os.getenv("VIS_COLLECTION_NAME", "visualizations_qdrant_bge")

INFO_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
OLLAMA_REWRITE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_ANSWER_URL = os.getenv("OLLAMA_BASE_URL")

# vLLM configuration (OpenAI-compatible API)
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://192.168.0.96:8001/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "openai/gpt-oss-20b")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy-key")


SHORT_FORM_EXPANSION = {
    "ADB": "Asian Development Bank",
    "ADG": "Additional Director General",
    "AES-256": "Advanced Encryption Standard with a 256-bit key",
    "AI": "Artificial Intelligence",
    "API": "Application Programming Interface",
    "ASI": "Annual Survey of Industries",
    "ASPD": "Administrative Statistics and Policy Division",
    "AWS": "Amazon Web Services",
    "CDD": "Capacity Development Division",
    "CERT-In": "Computer Emergency Response Team - Indian",
    "CISO": "Chief Information Security Officer",
    "CMS": "Content Management System",
    "CPI": "Consumer Price Index",
    "CS": "Central Statistics",
    "CSP": "Cloud Service Provider",
    "CSV": "Comma Separated Values",
    "DB": "Database",
    "DBIM": "Digital Brand Identity Manual",
    "DDG": "Deputy Director General",
    "DDI": "Data Documentation Initiative",
    "DI Lab": "Data Innovation Lab",
    "DIID": "Data Information and Innovation Division",
    "DPDP Act": "Digital Personal Data Protection Act",
    "DR": "Disaster Recovery",
    "EC": "Economic Census",
    "EnSD": "Enterprise Survey Division",
    "ESD": "Economic Statistics Division",
    "ETL": "Extract Transform Load",
    "FOD": "Field Operations Division",
    "FRS": "Functional Requirements Specification",
    "GB": "Gigabyte",
    "GDP": "Gross Domestic Product",
    "GIGW": "Guidelines for Indian Government Websites",
    "GIS": "Geographical Information System",
    "GMS": "Grant Management System",
    "GSBPM": "Generic Statistical Business Process Model",
    "HCES": "Household Consumption Expenditure Survey",
    "HRMS": "Human Resource Management System",
    "HSD": "Household Survey Division",
    "HSU": "Household Survey Unit",
    "ICP": "International Comparison Program",
    "iGOT": "Integrated Government Online Training",
    "IIP": "Index of Industrial Production",
    "ILO": "International Labour Organization",
    "IMF": "International Monetary Fund",
    "IoT": "Internet of Things",
    "IPMD": "Infrastructure and Project Monitoring Division",
    "IPR": "Intellectual Property Rights",
    "ISI": "Indian Statistical Institute",
    "ISO/IEC": "International Organization for Standardization / International Electrotechnical Commission",
    "ISS": "Indian Statistical Service",
    "JSO": "Junior Statistical Officer",
    "JSON": "JavaScript Object Notation",
    "KMS": "Knowledge Management System",
    "KPI": "Key Performance Indicators",
    "LCNC": "Low Code No Code",
    "LGD": "Local Government Directory",
    "LMS": "Learning Management System",
    "MCDI": "Management Capability Development Index",
    "MeitY": "Ministry of Electronics and Information Technology",
    "MHA": "Ministry of Home Affairs",
    "ML": "Machine Learning",
    "MOOC": "Massive Open Online Courses",
    "MoSPI": "Ministry of Statistics and Programme Implementation",
    "MPLADS": "Members of Parliament Local Area Development Scheme",
    "MXDP": "Multi Experience Development Platforms",
    "NAD": "National Accounts Division",
    "NDSAP": "National Data Sharing and Accessibility Policy",
    "NISDP": "National Integrated Statistical Data Platform",
    "NLP": "Natural Language Processing",
    "NQAF": "National Quality Assurance Framework",
    "NSA": "National Academy of Statistical Administration",
    "NSC": "National Statistical Commission",
    "NSO": "National Statistical Office",
    "NSS": "National Sample Survey",
    "NSSTA": "National Statistical System Training Academy",
    "OCMS": "Online Content Management System",
    "PB": "Petabytes",
    "PDF": "Portable Document Format",
    "PDPB": "Personal Data Protection Bill",
    "PoC": "Proof of Concepts",
    "RBAC": "Role Based Access Control",
    "RFP": "Request for Proposal",
    "RTI": "Right To Information",
    "SASA": "State Agricultural Statistics Authorities",
    "SDDS": "Special Data Dissemination Standards",
    "SDMX": "Statistical Data and Metadata eXchange Standards",
    "SIEM": "Security Information and Event Management",
    "SLA": "Service Level Agreement",
    "SMS": "Short Message Service",
    "SOP": "Standard Operating Procedures",
    "SQAF": "Statistical Quality Assessment Framework",
    "SRS": "Software Requirements Specification",
    "SSB": "State Statistical Bureaus",
    "SSD": "Social Statistics Division",
    "SSO": "Senior Statistical Officer",
    "SSS": "Subordinate Statistical Service",
    "SSSP": "Support for Statistical Strengthening Programme",
    "STQC": "Standardization Testing and Quality Certification Directorate",
    "TB": "Terabyte",
    "TLS 1.2/1.3": "Transport Layer Security 1.2/1.3",
    "TPP": "Twenty Point Programme",
    "TSU": "Technical Support Unit",
    "UI/UX": "User Interface and User Experience",
    "UN": "United Nations",
    "UNCEBD": "UN Committee of Experts on Big Data and Data Science for Official Statistics",
    "UNECE": "United Nations Economic Commission in Europe",
    "VC": "Virtual Classroom",
    "XML": "Extensible Markup Language",
}

def expand_short_forms_in_query(query: str) -> str:
    """
    SMART expansion: Only expand acronyms that are NOT already expanded.
    This prevents duplicate expansion when LLM has already expanded an acronym.
    
    E.g., "What is CPI?" → "What is Consumer Price Index?"
    E.g., "What is Consumer Price Index (CPI)?" → "What is Consumer Price Index (CPI)?" (no change)
    E.g., "who is ddg" → "who is Deputy Director General"
    
    Case-insensitive matching to handle "DDG", "ddg", "Ddg", etc.
    Uses word boundary matching to avoid false positives (e.g., 'it' in 'site').
    """
    if not query:
        return query
    
    result = query
    # Sort by length (longest first) to handle multi-word abbreviations (e.g., "DI Lab", "TLS 1.2/1.3")
    for abbr in sorted(SHORT_FORM_EXPANSION.keys(), key=len, reverse=True):
        expansion = SHORT_FORM_EXPANSION[abbr]
        
        # SMART CHECK: Skip if the expansion already exists in the query (case-insensitive)
        # This prevents duplicate expansion like "GDP (GDP)" or "CPI (CPI)"
        if expansion.lower() in result.lower():
            continue
        
        # Use word boundary matching with case-insensitive flag to match any case variant
        # This handles "DDG", "ddg", "Ddg", etc.
        pattern = r'\b' + re.escape(abbr) + r'\b'
        if re.search(pattern, result, re.IGNORECASE):
            # Replace with case-insensitive matching
            result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
    
    return result

# ========== Global State ==========
memory_sessions = get_session_store()  # Redis-backed; shared across backend containers
info_embedding_model = None  # For main collection
# vis_embedding_model = None    # For visualization collection
info_vectordb = None
vis_vectordb = None
info_retriever = None
llm = None  # Main LLM for response generation (nemotron-3-nano)
llm_query_rewrite = None  # Lightweight LLM for query rewriting (llama3.2:3b)
custom_prompt = None
whoswho_names = set()
qdrant_client: QdrantClient = None
qdrant_client_vis: QdrantClient = None

# Reranker globals (preloaded at startup for lowest latency)
reranker_tokenizer = None
reranker_model = None
_reranker_device = None
_reranker_fp16 = None

from whois_cache_manager import (
    get_cached_whois_doc,
    start_whois_cache_refresh_task,
    get_cache_status as get_whois_cache_status,
    force_refresh as force_whois_refresh,
)

# Legacy globals for backward compatibility (now managed by whois_cache_manager)
CACHED_WHOIS_DOC: Optional[Document] = None  # Updated by _initialize_cached_whois


def _filter_whois_content_for_query(content: str, query: str, _allow_spell_retry: bool = True) -> str:

    if not content or not query:
        return content
    
    query_lower = query.lower()
    
    # FOD Zone mapping - maps zone keywords to their standardized names in the cache
    fod_zone_map = {
        'north zone': ['north zone', 'northern zone', 'north', 'fod north'],
        'east zone': ['east zone', 'eastern zone', 'east', 'fod east'],
        'west zone': ['west zone', 'western zone', 'west', 'fod west'],
        'south zone': ['south zone', 'southern zone', 'south', 'fod south'],
        'central zone': ['central zone', 'centre zone', 'fod central'],
        'north east zone': ['north east zone', 'northeast zone', 'north-east zone', 'northeastern zone', 'fod north east', 'fod northeast'],
        'as wing': ['as wing', 'faridabad', 'as wing (faridabad)', 'as wing faridabad'],
        'fod hqrs': ['fod hqrs', 'fod headquarters', 'fod delhi', 'fod hq', 'field operations division (fod) headquarters'],
    }
    

    fod_office_types = {
        'zo': ['zo', 'zonal office', 'zone office', 'zo('],  # ZO appears in Division column
        'ro': ['ro', 'regional office', 'ro('],  # RO appears in Division column
        'sro': ['sro', 'sub-regional office', 'sub regional office', 'sro('],  # SRO appears in Division column
    }
    
    # Detect FOD zone filter
    fod_zone_filter = None
    for zone_key, zone_variations in fod_zone_map.items():
        for variation in zone_variations:
            if variation in query_lower:
                fod_zone_filter = zone_key
                break
        if fod_zone_filter:
            break
    
    # DEBUG: Log zone detection
    if fod_zone_filter:
        logger.info(f"[WHOIS_FILTER_DEBUG] Detected fod_zone_filter: '{fod_zone_filter}'")
    
    # Detect FOD office type filter (ZO, RO, SRO - these go in Division, not Designation)
    fod_office_type_filter = None
    for office_key, office_variations in fod_office_types.items():
        for variation in office_variations:
            # Use word boundary for short acronyms like "ro", "zo"
            if len(variation) <= 3 and not variation.endswith('('):
                if re.search(rf'\b{re.escape(variation)}\b', query_lower):
                    fod_office_type_filter = office_key
                    break
            else:
                if variation in query_lower:
                    fod_office_type_filter = office_key
                    break
        if fod_office_type_filter:
            break
    
    # DEBUG: Log office type detection
    if fod_office_type_filter:
        logger.info(f"[WHOIS_FILTER_DEBUG] Detected fod_office_type_filter: '{fod_office_type_filter}'")
    
    # Detect FOD city/location queries (e.g., "RO Jaipur", "DDG ZO Jaipur")
    # Common FOD city patterns
    fod_city_pattern = r'\b(jaipur|kolkata|mumbai|chennai|bangalore|hyderabad|guwahati|lucknow|bhopal|chandigarh|patna|ahmedabad|pune|delhi|faridabad|shimla|dharamshala|mandi|hamirpur|srinagar|ajmer|baramulla|anantnag|jodhpur)\b'
    fod_city_match = re.search(fod_city_pattern, query_lower)
    fod_city_filter = fod_city_match.group(1) if fod_city_match else None
    
    # Division abbreviation mapping for better matching
    division_abbr_map = {
        'diid': 'data informatics innovation',
        'fod': 'field operations',
        'nad': 'national accounts',
        'esd': 'economic statistics',
        'ssd': 'social statistics',
        'psd': 'price statistics',
        'hsd': 'household survey',
        'ensd': 'enterprise survey',
        'cqcd': 'coordination quality control',
        'cicd': 'coordination international cooperation',
        'aspd': 'administrative statistics policy',
        'cdd': 'capacity development division',
        'nss': 'national sample survey',
        'nso': 'national statistical office',
        'mplads': 'member parliament local area development',
        'NSSTA': 'National Statistical Systems Training Academy'
    }
    
    # Expand division abbreviations in query for better matching
    query_expanded = query_lower
    for abbr, full in division_abbr_map.items():
        if re.search(rf'\b{abbr}\b', query_lower):
            query_expanded = re.sub(rf'\b{abbr}\b', full, query_expanded)
    
    lines = content.strip().split("\n")
    
    if len(lines) < 2:
        return content
    
    header = lines[0]
    data_lines = lines[1:]

    def _spell_correct_query_against_cache() -> Optional[str]:
        designation_counts = defaultdict(int)
        name_vocab = set()
        for line in data_lines:
            fields = line.split("|")
            if len(fields) < 3:
                continue
            name_vocab.update(re.findall(r"[a-z]{4,}", fields[0].lower()))
            for word in set(re.findall(r"[a-z]{4,}", fields[1].lower())):
                designation_counts[word] += 1
        designation_vocab = set(designation_counts)
        if not designation_vocab:
            return None

        def _canonical_designation_word(word: str) -> Optional[str]:
            candidates = [
                candidate for candidate in
                difflib.get_close_matches(word, designation_vocab, n=3, cutoff=0.86)
                if candidate != word
            ]
            if not candidates:
                return None
            best = candidates[0]
            if word not in designation_vocab:
                return best
            if designation_counts[best] >= 5 * max(1, designation_counts[word]):
                return best
            return None

        # Words that are ordinary query phrasing rather than directory terms.
        non_directory_words = {
            "mospi", "ministry", "statistics", "programme", "implementation",
            "please", "details", "detail", "information", "number", "current",
            "latest", "there", "their", "about", "which", "whose", "posted",
            "working", "located",
        }

        rebuilt_parts = []
        corrected_any = False
        for part in re.split(r"(\W+)", query_lower):
            if (
                len(part) >= 5
                and part.isalpha()
                and part not in non_directory_words
                and part not in name_vocab
                and not difflib.get_close_matches(part, name_vocab, n=1, cutoff=0.85)
            ):
                replacement = _canonical_designation_word(part)
                if replacement:
                    logger.info(
                        "[WHOIS_FILTER_DEBUG] Spell-corrected query word '%s' -> '%s'",
                        part,
                        replacement,
                    )
                    rebuilt_parts.append(replacement)
                    corrected_any = True
                    continue
            rebuilt_parts.append(part)

        return "".join(rebuilt_parts) if corrected_any else None

    if _allow_spell_retry:
        corrected_query = _spell_correct_query_against_cache()
        if corrected_query and corrected_query.strip() != query_lower.strip():
            retry_result = _filter_whois_content_for_query(
                content, corrected_query, _allow_spell_retry=False
            )
            if not retry_result.lstrip().lower().startswith("no "):
                logger.info(
                    "[WHOIS_FILTER_DEBUG] Spell-corrected retry succeeded: '%s' -> '%s'",
                    query,
                    corrected_query,
                )
                return retry_result

    # Detect if this is a plural query asking for "all" officers of a type
    is_plural_query = bool(re.search(r'\b(all|list|who\s+are)\b', query_lower))
    
    # Check if this is a general MoSPI office/org query (not person-specific)
    is_org_query = bool(re.search(r'\b(mospi|ministry)\b.*(address|office|email|contact|phone)', query_lower)) or \
                   bool(re.search(r'\b(address|email|contact|phone).*(mospi|ministry)\b', query_lower)) or \
                   bool(re.search(r"where\s+(is|are)\s+(mospi|ministry|the\s+office)", query_lower))
    
    # NEW: Check if this is a query for "all FOD" officers (no specific zone)
    is_all_fod_query = bool(re.search(r'\ball\s+fod\b', query_lower)) or \
                       bool(re.search(r'\bfod\s+(officers|zones|all)\b', query_lower)) or \
                       (bool(re.search(r'\bfod\b', query_lower)) and is_plural_query)
    
    # DEBUG: Log all FOD query detection
    if is_all_fod_query:
        logger.info(f"[WHOIS_FILTER_DEBUG] Detected is_all_fod_query: True")
    
    if is_org_query:
        # Return Secretary's office info as the main MoSPI contact
        for line in data_lines:
            fields = line.split("|")
            if len(fields) >= 6:
                designation = fields[1].lower()
                # Secretary is the main office contact
                if designation == "secretary" or "secretary|ministry" in line.lower():
                    name = fields[0].strip()
                    designation_clean = fields[1].strip()
                    division = fields[2].strip()
                    contact = fields[3].strip()
                    email = fields[4].strip().replace("[at]", "@").replace("[dot]", ".").replace("[dash]", "-")
                    address = fields[5].strip()
                    
                    return f"""=== MoSPI Office Information ===

Main Office Contact (Secretary's Office):
• Name: {name}
• Designation: {designation_clean}
• Division: {division}
• Contact: {contact}
• Email: {email}
• Address: {address}

Note: MoSPI headquarters is located at Khurshid Lal Bhawan, Janpath, New Delhi - 110001."""
        
        # Fallback if secretary not found
        return "MoSPI headquarters: Khurshid Lal Bhawan, Janpath, New Delhi - 110001. For specific contacts, please ask about a particular officer or division."
    
    # Designation patterns - order matters (most specific first)
    designation_patterns = [
        (
            r"^\s*(?:who\s+is\s+(?:the\s+)?)?(?:current\s+)?(?:dg|director general)\s+(?:of\s+)?central statistics\s*\??\s*$",
            r"director general\s*\(\s*central statistics\s*\)",
        ),
        (
            r"^\s*(?:who\s+is\s+(?:the\s+)?)?(?:current\s+)?(?:pps|principal private secretary)\s+(?:to\s+)?(?:the\s+)?(?:dg|director general)\s*\(?\s*central statistics\s*\)?\s*\??\s*$",
            r"\bpps\s+to\s+dg\s*\(\s*central statistics\s*\)",
        ),
        (r'\b(ddg|deputy director general)\b', r'deputy director general'),
        (r'\b(adg|additional director general)\b', r'additional director general'),
        (r'\b(pps|principal private secretary)\b', r'\b(pps|principal private secretary)\b'),
        (r'\b(pdg|principal director general)\b', r'^\s*principal director general'),

        (r'\b(dg|director general)\b', r'^\s*director general'),
        (r'\b(js|joint secretary)\b', r'joint secretary'),
        (r'\b(as|additional secretary)\b', r'additional secretary'),

        (r'\bunder\s+secretary\b', r'^\s*under secretary'),
        (r'\bdeputy\s+secretary\b', r'^\s*deputy secretary'),
        (r'\b(ps|private secretary)\b', r'private secretary'),
        (r'\bsecretary\b', r'^\s*secretary(?:\s*\([^)]*\))?\s*$'),
        (r'(?<!prime )(?<!chief )\b(minister|mos|mo s)\b', r'(minister|mos|mo s)'),
        (r'\b(cvo|chief vigilance officer)\b', r'\bcvo\b'), 
        (r'\b(deputy director|dd)\b', r'^deputy director(?:\s|\(|$)'),
        (r'\b(assistant director|ad)\b', r'^assistant director(?:\s|\(|$)'),
        (r'\b(joint director|jd)\b', r'^joint director(?:\s|\(|$)'),
        (r'\bdirectors?\b', r'^director(?:\s*\([^)]*\))?\s*$'),
    ]
    
    designation_filter = None
    for query_pattern, match_pattern in designation_patterns:
        if re.search(query_pattern, query_lower):
            designation_filter = match_pattern
            break

    def _designation_display(pattern_text: str) -> str:
        first_branch = (pattern_text or "").split("|")[0]
        # Single letters here are regex classes (\s, \b, \d, \w), not words.
        words = [
            word for word in re.findall(r"[a-z]+", first_branch.lower())
            if len(word) > 1
        ]
        if not words:
            return "matching"
        return " ".join(
            word.upper() if len(word) <= 3 else word.title() for word in words
        )
    
    # DEBUG: Log designation detection
    if designation_filter:
        logger.info(f"[WHOIS_FILTER_DEBUG] Detected designation_filter: '{designation_filter}'")
    
    # Extract name parts (remove common words)
    stop_words = {'who', 'is', 'are', 'the', 'of', 'for', 'mospi', 'ministry', 'statistics', 
                  'programme', 'implementation', 'email', 'mail', 'contact', 'phone', 
                  'address', 'office', 'what', 'where', 'tell', 'me', 'about', 'give',
                  'can', 'you', 'please', 'show', 'find', 'get', 'all', 'list'}
    words = re.findall(r'\b[a-z]+\b', query_lower)
    name_parts = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Division keywords - enhanced with more variations + FOD zones
    division_keywords = {
        "diid": ["diid", "data information and innovation division", "data information innovation", "data informatics innovation", "data informatics and innovation division"],
        "nss": ["nss", "national sample survey", "survey division"],
        "nso": ["nso", "national statistical office"],
        "nad": ["nad", "national accounts division", "national accounts"],
        "esd": ["esd", "economic statistics division", "economic statistics"],
        "ssd": ["ssd", "social statistics division", "social statistics"],
        "psd": ["psd", "price statistics division", "price statistics"],
        "hsd": ["hsd", "household survey division", "household survey"],
        "hsu": ["hsu", "household survey unit"],
        "fod": ["fod", "field operations division", "field operations"],
        "mplads": ["mplads", "member of parliament local area development scheme"],
        "nic": ["nic", "national informatics centre"],
        "cqcd": ["cqcd", "coordination and quality control division", "coordination quality control"],
        "cicd": ["cicd", "coordination and international cooperation unit", "coordination international cooperation"],
        "ensd": ["ensd", "enterprise survey division", "enterprise survey"],
        "aspd": ["aspd", "administrative statistics and policy division", "administrative statistics policy"],
        "cdd": ["cdd", "capacity development division", "capacity development"],
        # FOD zones as pseudo-divisions
        "fod_north": ["fod north zone", "north zone", "northern zone"],
        "fod_east": ["fod east zone", "east zone", "eastern zone"],
        "fod_west": ["fod west zone", "west zone", "western zone"],
        "fod_south": ["fod south zone", "south zone", "southern zone"],
        "fod_central": ["fod central zone", "central zone"],
        "fod_northeast": ["fod north east zone", "northeast zone", "north-east zone"],
        "fod_aswing": ["as wing", "faridabad"],
        "fod_hqrs": ["fod hqrs", "fod headquarters", "fod delhi"],
    }
    
    designation_vocabulary = {
        "secretary", "secretaries", "under", "principal", "private",
        "personal", "minister", "mos", "pps", "ps", "pa", "steno",
        "stenographer", "jso", "sso", "dg", "ddg", "adg", "cvo",
        "officer", "officers", "official", "officials",
    }

    division_filter = None
    # Check both original and expanded query
    for div_key, div_variations in division_keywords.items():
        for variation in div_variations:
            if variation in query_lower or variation in query_expanded:
                division_filter = div_key
                break
        if division_filter:
            break

    if not division_filter:
        division_noise_words = {
            "and", "the", "of", "for", "in", "at", "to", "division", "department",
            "office", "unit", "wing", "cell", "section", "director", "directors", "deputy",
            "joint", "additional", "general", "assistant", "list", "all", "who", "is", "are",
            "contact", "email", "phone", "telephone", "address",
        }

        def _division_comparison_tokens(value):
            normalised = (value or "").lower().replace("&", " and ")
            normalised = re.sub(r"[^a-z0-9]+", " ", normalised)
            return {
                token for token in normalised.split()
                if token not in division_noise_words and len(token) > 1
            }

        query_division_tokens = _division_comparison_tokens(query_lower)

        def _candidate_division_matches_query(candidate_text):
            candidate_tokens = _division_comparison_tokens(candidate_text)
            if not candidate_tokens:
                return False
            abbreviations = {
                abbreviation.lower()
                for abbreviation in re.findall(r"\(([A-Za-z0-9]{2,10})\)", candidate_text)
            }
            full_name_tokens = candidate_tokens.difference(abbreviations)

            matched_tokens = set()
            if abbreviations.intersection(query_division_tokens):
                matched_tokens = abbreviations.intersection(query_division_tokens)
            elif full_name_tokens and full_name_tokens.issubset(query_division_tokens):
                matched_tokens = full_name_tokens

            return bool(matched_tokens and matched_tokens.difference(designation_vocabulary))

        for line in data_lines:
            fields = line.split("|")
            if len(fields) < 3:
                continue

            candidate_texts = [
                qualifier.strip() for qualifier in re.findall(r"\(([^)]+)\)", fields[1])
            ]
            candidate_division = fields[2].strip()
            if not candidate_division.lower().startswith("office of"):
                candidate_texts.append(candidate_division)

            for candidate_text in candidate_texts:
                if _candidate_division_matches_query(candidate_text):
                    division_filter = candidate_text
                    logger.info(
                        "[WHOIS_FILTER_DEBUG] Matched cached division '%s' for query '%s'",
                        candidate_text,
                        query,
                    )
                    break
            if division_filter:
                break
    

    is_ambiguous_contact_query = (
        # Has contact keywords (email, phone, contact, address)
        bool(re.search(r'\b(email|e-mail|mail|contact|phone|address)\b', query_lower)) and
        # Has designation but NO division specified
        designation_filter is not None and
        division_filter is None and
        # No specific person name mentioned
        len(name_parts) == 0
    )
    
    # Treat ambiguous queries as plural (return all matches with note)
    if is_ambiguous_contact_query:
        is_plural_query = True
    
    # Treat ambiguous queries as plural (return all matches with note)
    if is_ambiguous_contact_query:
        is_plural_query = True
    

    def _fod_office_type_matches(office_type_filter, division_field):

        if not office_type_filter or not division_field:
            return False
        
        office_norm = office_type_filter.lower().strip()
        div_norm = division_field.lower().strip()
        
        # Check if office type appears at start of division (e.g., "ZO(CQCD)")
        if div_norm.startswith(office_norm + '('):
            return True
        
        # Check if office type appears with space (e.g., "ZO (CQCD)")
        if div_norm.startswith(office_norm + ' ('):
            return True
        
        # Check for full form
        office_full_forms = {
            'zo': 'zonal office',
            'ro': 'regional office',
            'sro': 'sub-regional office',
        }
        if office_norm in office_full_forms and office_full_forms[office_norm] in div_norm:
            return True
        
        return False
    
    def _fod_zone_matches(fod_zone_filter, fod_zone_field="", division_field="", address_field=""):
        if not fod_zone_filter:
            return False
        
        zone_norm = fod_zone_filter.lower().strip()
        
        if not fod_zone_field or fod_zone_field.strip() == "":
            # Fallback: Check if this might be a Who's Who FOD HQ entry
            # Only match if division contains "field operations" AND "hq"
            if division_field:
                div_norm = division_field.lower().strip()
                # Special case: FOD HQ entries in Who's Who section
                if "field operations" in div_norm and "hq" in div_norm:
                    # Only match if looking for "fod hqrs" zone
                    if zone_norm == "fod hqrs":
                        return True
                # Otherwise, no zone field = no match for zone queries
                return False
            return False
        

        zone_field_norm = fod_zone_field.lower().strip()
        

        zone_field_clean = zone_field_norm.lstrip('#').strip()
        if zone_norm.replace('_', ' ') in zone_field_clean:
            return True
        
        if 'fod' in zone_field_clean:
            # Extract text after "FOD"
            parts = zone_field_clean.split('fod')
            if len(parts) > 1:
                zone_part = parts[-1].strip().strip('(').strip(')').strip()
                if zone_norm.replace('_', ' ') in zone_part:
                    return True
        
        # Special zone matches
        if zone_norm == 'as wing' and 'as wing' in zone_field_clean:
            return True
        if zone_norm == 'fod hqrs' and ('hqrs' in zone_field_clean or 'headquarters' in zone_field_clean):
            return True
        
        # Method 2: Fallback to division field (for FOD rows with zone info)
        if division_field:
            div_norm = division_field.lower().strip()
            if zone_norm.replace('_', ' ') in div_norm:
                return True
            if f"fod {zone_norm.replace('_', ' ')}" in div_norm:
                return True
        
        # Method 3: Fallback to address field for city-based zones
        if address_field:
            addr_norm = address_field.lower().strip()
            if zone_norm == 'as wing' and 'faridabad' in addr_norm:
                return True
            if zone_norm == 'fod hqrs' and 'delhi' in addr_norm:
                return True
        
        return False
    
    # Helper function for city/location matching
    def _city_matches(city_filter, address_field, division_field=""):
        """
        Check if city filter matches address or division field.
        """
        if not city_filter or not address_field:
            return False
        
        city_norm = city_filter.lower().strip()
        addr_norm = address_field.lower().strip()
        div_norm = division_field.lower().strip() if division_field else ""
        
        # Direct city match in address or division
        return city_norm in addr_norm or city_norm in div_norm
    
    # Helper function for better division matching
    def _division_matches(division_filter, division_field):
        """
        Check if division filter matches division field.
        Handles abbreviations, parentheses, and variations.
        
        Examples:
        - "fod" matches "Field Operations Division (HQ)"
        - "field operations" matches "Field Operations Division (HQ)"
        - "diid" matches "Data Informatics & Innovation Division (DIID)"
        """
        if not division_filter or not division_field:
            return False
        
        # Normalize both strings
        filter_norm = division_filter.lower().strip()
        field_norm = division_field.lower().strip()
        
        # Remove parentheses content from field (e.g., "(HQ)", "(DIID)")
        field_clean = re.sub(r'\s*\([^)]*\)', '', field_norm).strip()
        
        # Method 1: Direct substring match
        if filter_norm in field_clean or filter_norm in field_norm:
            return True
        
        # Method 2: Word-based matching (all filter words present in field)
        filter_words = set(filter_norm.split())
        field_words = set(field_clean.split())
        if filter_words and filter_words.issubset(field_words):
            return True
        
        # Method 3: Abbreviation match using expanded query
        # Check if expanded form matches
        if filter_norm in division_abbr_map:
            expanded = division_abbr_map[filter_norm]
            expanded_words = set(expanded.split())
            if expanded_words and expanded_words.issubset(field_words):
                return True

        def _normalised_words(value):
            value = (value or "").lower().replace("&", " and ")
            value = re.sub(r"[^a-z0-9]+", " ", value)
            ignored = {"and", "the", "of", "for", "in", "at", "to", "division", "department", "office", "unit", "wing", "cell", "section"}
            return {word for word in value.split() if word not in ignored and len(word) > 1}

        normalised_filter_words = _normalised_words(division_filter)
        normalised_field_words = _normalised_words(division_field)
        if normalised_filter_words and normalised_filter_words.issubset(normalised_field_words):
            return True
        
        return False

    def _designation_qualifier_matches(division_filter_value, designation_field_value):

        if not division_filter_value or not designation_field_value:
            return False
        qualifiers = re.findall(r"\(([^)]+)\)", designation_field_value)
        return any(_division_matches(division_filter_value, qualifier) for qualifier in qualifiers)

    def _fuzzy_token_in_text(token: str, text: str, threshold: float = 0.84) -> bool:
        if not token or not text:
            return False
        if token in text:
            return True
        for word in re.findall(r"[a-z0-9]+", text):
            if len(word) < 3:
                continue
            if difflib.SequenceMatcher(None, token, word).ratio() >= threshold:
                return True
        return False

    def _fallback_token_search() -> List[str]:
        filler_words = {
            "posted", "posting", "working", "work", "works", "located", "based",
            "sitting", "stationed", "serving", "deployed", "available",
            "person", "people", "someone", "anybody", "anyone", "incharge",
            "charge", "there", "here", "currently", "presently",
        }
        search_tokens = [token for token in name_parts if token not in filler_words]
        if not search_tokens:
            return []
        rows = []
        for line in data_lines:
            fields = line.split("|")
            if len(fields) < 3:
                continue
            searchable = " ".join(fields[:6]).lower()
            if all(_fuzzy_token_in_text(token, searchable) for token in search_tokens):
                rows.append(line)
        return rows
    
    # Filter rows with enhanced FOD support
    matched_rows = []
    partial_name_matches = []  # Fallback pool for Priority 9 (single-word name matches)
    for line in data_lines:
        line_lower = line.lower()
        fields = line.split("|")
        if len(fields) < 2:
            continue
        
        name_field = fields[0].lower() if len(fields) > 0 else ""
        designation_field = fields[1].lower() if len(fields) > 1 else ""
        division_field = fields[2].lower() if len(fields) > 2 else ""
        address_field = fields[5].lower() if len(fields) > 5 else ""
        # NEW: Extract FOD_Zone field (7th column) if it exists
        fod_zone_field = fields[6].lower() if len(fields) > 6 else ""
        
        # DEBUG: Log first few rows being processed
        if len(matched_rows) == 0 and (fod_zone_filter or fod_office_type_filter):
            logger.info(f"[WHOIS_FILTER_DEBUG] Processing row: name='{fields[0][:30]}...', designation='{designation_field[:30]}', division='{division_field[:30]}', fod_zone_field='{fod_zone_field[:50]}'")
        
        # Priority 1: FOD zone + designation match (e.g., "DDG of FOD North Zone")
        if fod_zone_filter and designation_filter:
            if re.search(designation_filter, designation_field):
                zone_match_result = _fod_zone_matches(fod_zone_filter, fod_zone_field, division_field, address_field)
                # DEBUG: Log matching attempt
                if len(matched_rows) < 10:  # Log first 10 attempts
                    logger.info(f"[WHOIS_FILTER_DEBUG] Priority 1 check: name='{name_field[:30]}', designation match={True}, zone_match={zone_match_result}, fod_zone_field='{fod_zone_field[:70]}'")
                if zone_match_result:
                    matched_rows.append(line)
                    logger.info(f"[WHOIS_FILTER_DEBUG] MATCHED Priority 1: {fields[0][:50]}")
                    continue
                else:
                    if not fod_zone_field or fod_zone_field.strip() == "":
                        continue  
        
 
        if fod_office_type_filter and fod_zone_filter and designation_filter:
            if re.search(designation_filter, designation_field):
                if _fod_office_type_matches(fod_office_type_filter, division_field):
                    if _fod_zone_matches(fod_zone_filter, fod_zone_field, division_field, address_field):
                        matched_rows.append(line)
                        continue
        
        # Priority 3: FOD city + designation match (e.g., "DDG Jaipur", "DDG ZO Jaipur")
        if fod_city_filter and designation_filter:
            if re.search(designation_filter, designation_field):
                # Also check if office type matches if specified
                if fod_office_type_filter:
                    if _fod_office_type_matches(fod_office_type_filter, division_field):
                        if _city_matches(fod_city_filter, address_field, division_field):
                            matched_rows.append(line)
                            continue
                else:
                    if _city_matches(fod_city_filter, address_field, division_field):
                        matched_rows.append(line)
                        continue
        
        # Priority 4: FOD office type + zone (e.g., "ZO North Zone", "RO in North Zone")
        if fod_office_type_filter and fod_zone_filter and not designation_filter:
            if _fod_office_type_matches(fod_office_type_filter, division_field):
                if _fod_zone_matches(fod_zone_filter, fod_zone_field, division_field, address_field):
                    matched_rows.append(line)
                    continue
        
        # Priority 5: FOD office type + city (e.g., "RO Jaipur", "ZO Kolkata")
        if fod_office_type_filter and fod_city_filter and not designation_filter:
            if _fod_office_type_matches(fod_office_type_filter, division_field):
                if _city_matches(fod_city_filter, address_field, division_field):
                    matched_rows.append(line)
                    continue
        
        # Priority 6: FOD zone only (e.g., "contact for North Zone")
        if fod_zone_filter and not designation_filter and not fod_office_type_filter:
            if _fod_zone_matches(fod_zone_filter, fod_zone_field, division_field, address_field):
                matched_rows.append(line)
                continue
        
        # Priority 7: Exact designation match (original logic)
        if designation_filter:
            if re.search(designation_filter, designation_field):
                if fod_zone_filter and (not fod_zone_field or fod_zone_field.strip() == ""):
                    # Skip Who's Who entries when filtering by zone
                    if len(matched_rows) < 5:  # Log first few skips
                        logger.info(f"[WHOIS_FILTER_DEBUG] Priority 7 SKIPPED (zone guard): name='{name_field[:30]}', no zone field")
                    continue
                if division_filter:
                    if _division_matches(division_filter, division_field) or \
                       _designation_qualifier_matches(division_filter, designation_field):
                        matched_rows.append(line)
                        continue
                else:
                    if is_all_fod_query:
                        if fod_zone_field and fod_zone_field.strip() != "":
                            matched_rows.append(line)
                            continue
                        else:
                            # Skip Who's Who entries for "all FOD" queries
                            continue
                    
                    # No division or zone filter, include all matching designations
                    matched_rows.append(line)
                    continue
        
        if not designation_filter and division_filter:
            if _division_matches(division_filter, division_field) or _division_matches(division_filter, address_field):
                matched_rows.append(line)
                continue
        
        if not designation_filter and name_parts:
            if all(part in name_field for part in name_parts):
                matched_rows.append(line)
            elif any(part in name_field for part in name_parts):
                partial_name_matches.append(line)
    

    if not matched_rows and partial_name_matches:
        matched_rows = partial_name_matches


    if division_filter and len(matched_rows) > 1:
        division_variation_words = set()
        for variation in division_keywords.get(division_filter, []):
            division_variation_words.update(variation.split())
        structural_words = designation_vocabulary | {
            "division", "office", "unit", "wing", "cell", "section", "department",
            "director", "general", "deputy", "additional", "assistant", "joint",
        }
        extra_location_words = [
            w for w in name_parts
            if w not in division_variation_words
            and w != division_filter
            and w not in structural_words
        ]
        if extra_location_words:
            narrowed_rows = [
                row for row in matched_rows
                if any(loc_word in row.lower() for loc_word in extra_location_words)
            ]
            if narrowed_rows:
                matched_rows = narrowed_rows


    if designation_filter and '(minister|mos|mo s)' in designation_filter and len(matched_rows) > 1:

        minister_rows = []
        for row in matched_rows:
            fields = row.split("|")
            if len(fields) >= 3:
                designation = fields[1].strip()
                division = fields[2].strip()
                # Check if this is the actual minister (not staff)
                if designation.startswith("Hon'ble MOS") or division.lower() == "ministry":
                    minister_rows.append(row)
        
        # If we found the actual minister, use only that
        if minister_rows:
            matched_rows = minister_rows

    if designation_filter and len(matched_rows) > 1:
        subordinate_designation_re = re.compile(
            r"^\s*(?:sr\.?\s+|senior\s+|addl\.?\s+|additional\s+)?"
            r"(?:pps|ps|pa|jso|sso|steno|stenographer|"
            r"principal\s+private\s+secretary|senior\s+private\s+secretary|"
            r"private\s+secretary|personal\s+assistant|"
            r"junior\s+statistical\s+officer|protocol\s+officer)"
            r"[^|]*?\b(?:to|of)\b",
            re.IGNORECASE,
        )
        query_asks_for_staff = bool(re.search(
            r"\b(pps|ps|pa|jso|sso|steno|stenographer|private\s+secretary|"
            r"personal\s+assistant|protocol\s+officer|"
            r"junior\s+statistical\s+officer)\b",
            query_lower,
        ))
        if not query_asks_for_staff:
            primary_rows = [
                row for row in matched_rows
                if not subordinate_designation_re.match(row.split("|")[1])
            ]
            if primary_rows and len(primary_rows) < len(matched_rows):
                logger.info(
                    "[WHOIS_FILTER_DEBUG] Preferring %d post-holder row(s) over %d total matched row(s)",
                    len(primary_rows),
                    len(matched_rows),
                )
                matched_rows = primary_rows

    if not matched_rows:
        fallback_rows = _fallback_token_search()
        if fallback_rows:
            logger.info(
                f"[WHOIS_FILTER_DEBUG] Strict rules found nothing; fallback token search matched {len(fallback_rows)} row(s) for query='{query}'"
            )
            matched_rows = fallback_rows

    if not matched_rows:
        # Enhanced error messages for FOD queries
        if fod_zone_filter and designation_filter:
            zone_display = fod_zone_filter.replace('_', ' ').title()
            desig_display = _designation_display(designation_filter)
            return f"No {desig_display} officers found in FOD {zone_display}. Please check if the designation exists in this zone or try a different zone."
        elif fod_city_filter and designation_filter:
            desig_display = _designation_display(designation_filter)
            return f"No {desig_display} officers found in {fod_city_filter.title()}. Please check if there is a regional office in this city."
        elif fod_zone_filter:
            zone_display = fod_zone_filter.replace('_', ' ').title()
            return f"No officers found for FOD {zone_display}. The directory contains {len(data_lines)} officers total."
        # Provide more helpful error message for plural queries
        elif is_plural_query and designation_filter and division_filter:
            return f"No {_designation_display(designation_filter)} officers found in {division_filter.upper()}. The MoSPI directory contains {len(data_lines)} officers total."
        elif designation_filter:
            return f"No {_designation_display(designation_filter)} officers found. The MoSPI directory contains {len(data_lines)} officers total."
        else:
            return f"No officers found matching the query. The MoSPI directory contains {len(data_lines)} officers."
    

    result_lines = []
    
    if matched_rows:
        logger.info(f"[WHOIS_FILTER_DEBUG] Total matched_rows: {len(matched_rows)}")
        for i, row in enumerate(matched_rows[:5]):
            fields_preview = row.split("|")
            name_preview = fields_preview[0][:40] if len(fields_preview) > 0 else ""
            zone_preview = fields_preview[6][:50] if len(fields_preview) > 6 else "NO_ZONE"
            logger.info(f"[WHOIS_FILTER_DEBUG] matched_rows[{i}]: name='{name_preview}', zone='{zone_preview}'")
    
    # Add note for multiple matches if needed
    if len(matched_rows) > 1:
        if is_plural_query:
            if is_ambiguous_contact_query:
                result_lines.append("Note: Multiple officers found. Please specify division/zone for a specific contact (e.g., 'DDG of FOD North Zone').\n")
        else:
            result_lines.append("Note: Multiple officers found matching your query:\n")
    
    for row in matched_rows[:50]:
        fields = row.split("|")
        if len(fields) >= 3:
            name = fields[0].strip()
            designation = fields[1].strip()
            division = fields[2].strip()
            contact = fields[3].strip() if len(fields) > 3 else ""
            email = fields[4].strip() if len(fields) > 4 else ""
            address = fields[5].strip() if len(fields) > 5 else ""
            

            entry = (
                f"• {name}\n"
                f"  Designation : {designation}\n"
                f"  Division: {division}"
            )
            if contact and contact not in ["----", "...", "---", ""]:
                entry += f"\n  Contact: {contact}"
            if email and email not in ["----", "...", "---", ""]:
                email_clean = email.replace("[at]", "@").replace("[dot]", ".").replace("[dash]", "-")
                entry += f"\n  Email: {email_clean}"
            if address and address not in ["----", "...", "---", ""]:
                entry += f"\n  Address: {address}"
            result_lines.append(entry)
    
    if len(matched_rows) > 50:
        result_lines.append(f"\n(Showing 50 of {len(matched_rows)} matching officers)")
    elif len(matched_rows) > 1:
        result_lines.append(f"\n(Found {len(matched_rows)} officers)")
    
    return "\n\n".join(result_lines)


def _initialize_cached_whois():
    """Initialize the cached whois document at startup using the cache manager."""
    global CACHED_WHOIS_DOC
    
    # Start the cache manager (loads from file + starts background refresh)
    start_whois_cache_refresh_task()
    
    # Get the cached doc for backward compatibility
    CACHED_WHOIS_DOC = get_cached_whois_doc()
    
    if CACHED_WHOIS_DOC:
        logger.info(f" Cached whois document initialized: {len(CACHED_WHOIS_DOC.page_content)} chars")
    else:
        logger.warning(" Whois cache not available - whois queries will use fallback retrieval")


def create_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

def create_enhanced_session_context():
    """Create enhanced session context for better conversation continuity"""
    return {
        "conversation_memory": create_memory(),
        "document_context": [],  # Store recent document references
        "topic_context": [],     # Store recent topics/entities discussed
        "query_context": [],     # Store recent query-response pairs with metadata
        "last_retrieval_docs": [],  # Store last retrieved documents for follow-up
        "clarify_streak": 0,  # Consecutive turns where the bot asked for clarification
    }

def update_session_context(session_context: Dict, query: str, response: str, docs: List = None, topics: List = None):
    """Update session context with new query-response information"""
    
    # Update query context (keep last 10 exchanges)
    query_info = {
        "query": query,
        "response": response[:500],  # Truncate long responses
        "timestamp": datetime.now().isoformat(),
        "doc_count": len(docs) if docs else 0,
        "topics": topics or []
    }
    session_context["query_context"].append(query_info)
    if len(session_context["query_context"]) > 10:
        session_context["query_context"] = session_context["query_context"][-10:]
    
    # Update document context (keep last 20 unique documents)
    if docs:
        for doc in docs[:5]:  # Only store top 5 docs per query
            doc_ref = {
                "file_name": getattr(doc, 'metadata', {}).get('file_name', 'unknown'),
                "page_number": getattr(doc, 'metadata', {}).get('page_number', 1),
                "content_preview": doc.page_content[:200] if hasattr(doc, 'page_content') else "",
                "query_context": query[:100],  # What query retrieved this doc
                "timestamp": datetime.now().isoformat()
            }
            session_context["document_context"].append(doc_ref)
        
        # Keep unique documents, prioritize recent ones
        seen_docs = set()
        unique_docs = []
        for doc_ref in reversed(session_context["document_context"]):
            doc_key = f"{doc_ref['file_name']}_{doc_ref['page_number']}"
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                unique_docs.append(doc_ref)
                if len(unique_docs) >= 20:
                    break
        session_context["document_context"] = list(reversed(unique_docs))
        
        # Store last retrieval docs for immediate follow-up
        session_context["last_retrieval_docs"] = docs[:10]
    
    # Update topic context (extract and store key entities/topics)
    if topics:
        for topic in topics:
            if topic not in session_context["topic_context"]:
                session_context["topic_context"].append(topic)
        # Keep last 20 topics
        if len(session_context["topic_context"]) > 20:
            session_context["topic_context"] = session_context["topic_context"][-20:]

def extract_topics_from_query_and_docs(query: str, docs: List) -> List[str]:
    """Extract key topics/entities from query and retrieved documents"""
    topics = []
    
    # Extract from query
    query_lower = query.lower()
    
    # Common metrics and surveys
    metrics = ["cpi", "iip", "gdp", "plfs", "asi", "unemployment", "inflation", "industrial production"]
    for metric in metrics:
        if metric in query_lower:
            topics.append(metric.upper())
    
    # Time periods
    import re
    time_patterns = [
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(2020|2021|2022|2023|2024|2025)\b',
        r'\b(latest|recent|current)\b'
    ]
    for pattern in time_patterns:
        matches = re.findall(pattern, query_lower)
        topics.extend(matches)
    
    # Extract from document metadata
    if docs:
        for doc in docs[:3]:  # Check top 3 docs
            metadata = getattr(doc, 'metadata', {})
            doc_topics = metadata.get('topics', [])
            if doc_topics:
                topics.extend(doc_topics[:3])  # Add top 3 topics per doc
    
    # Remove duplicates and return
    return list(set(topics))

def build_enhanced_context_for_llm(session_context: Dict, current_query: str) -> str:
    """Build enhanced context string for LLM prompt"""
    
    context_parts = []
    
    # Recent conversation (last 6 messages for better continuity)
    memory = session_context.get("conversation_memory")
    if memory and memory.chat_memory.messages:
        hist_msgs = memory.chat_memory.messages[-6:]  # Increased from 4 to 6
        if hist_msgs:
            chat_history = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in hist_msgs])
            context_parts.append(f"=== RECENT CONVERSATION ===\n{chat_history}")
    
    # Recent topics discussed
    topics = session_context.get("topic_context", [])
    if topics:
        recent_topics = topics[-10:]  # Last 10 topics
        context_parts.append(f"=== RECENT TOPICS DISCUSSED ===\n{', '.join(recent_topics)}")
    
    # Recent document references (for follow-up questions)
    doc_context = session_context.get("document_context", [])
    if doc_context:
        recent_docs = doc_context[-5:]  # Last 5 documents
        doc_refs = []
        for doc in recent_docs:
            doc_refs.append(f"• {doc['file_name']} (p.{doc['page_number']}) - {doc['content_preview']}")
        context_parts.append(f"=== RECENTLY REFERENCED DOCUMENTS ===\n" + "\n".join(doc_refs))
    
    # Recent query patterns (for understanding user intent)
    query_context = session_context.get("query_context", [])
    if query_context and len(query_context) > 1:
        recent_queries = query_context[-3:]  # Last 3 queries
        query_patterns = []
        for q_info in recent_queries:
            query_patterns.append(f"Q: {q_info['query'][:100]}... → Topics: {', '.join(q_info['topics'][:3])}")
        context_parts.append(f"=== RECENT QUERY PATTERNS ===\n" + "\n".join(query_patterns))
    
    return "\n\n".join(context_parts) if context_parts else ""

def cleanup_expired_sessions(ttl_minutes: int = 90):
    cleanup_expired_sessions_redis(ttl_minutes)

# ========== Initialization ==========
def _build_qdrant_store(collection_name: str, embedding_model_instance):
    """Initialize Qdrant vector store with a specific embedding model"""
    if not QDRANT_URL:
        raise RuntimeError("❌ QDRANT_URL is required!")
    logger.info(f"🔗 Connecting to Qdrant at {QDRANT_URL} for collection: {collection_name}")
    return build_qdrant_store(collection_name, embedding_model_instance)
def load_whoswho_names(whoswho_chunks) -> Set[str]:
    """
    Extracts all officer names from Who's Who.json chunks for fast lookup.
    """
    names = set()
    for chunk in whoswho_chunks:
        match = re.search(r"name:\s*([^,]+),", chunk.page_content, re.IGNORECASE)
        if match:
            name = match.group(1).strip().lower()
            names.add(name)
    logger.info(f"Loaded {len(names)} officer names")
    return names
def initialize_components():

    global info_embedding_model, info_vectordb, info_retriever, vis_vectordb
    global llm_answer_client, llm_query_rewrite_client, custom_prompt, whoswho_names, qdrant_client, qdrant_client_vis

    info_embedding_model = HuggingFaceEmbeddings(model_name=INFO_EMBEDDING_MODEL)
    logger.info(f" Initialized info embedding model: {INFO_EMBEDDING_MODEL}")
    info_vectordb = _build_qdrant_store(INFO_COLLECTION_NAME, info_embedding_model)
    vis_vectordb = _build_qdrant_store(VIS_COLLECTION_NAME, info_embedding_model)

    qdrant_client = info_vectordb.client
    qdrant_client_vis = vis_vectordb.client
    info_retriever = info_vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    global llm_answer_client, llm_query_rewrite_client
    
    llm_answer_client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY
    )
    logger.info(f" Initialized main answer LLM: {VLLM_MODEL} (vLLM/OpenAI API at {VLLM_BASE_URL})")
    
    llm_query_rewrite_client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY
    )
    logger.info(f" Initialized query rewrite LLM: {VLLM_MODEL} (vLLM/OpenAI API at {VLLM_BASE_URL})")
    
    global reranker_tokenizer, reranker_model
    if reranker_tokenizer is None or reranker_model is None:
        logger.info(f"Loading BGE reranker ({RERANKER_MODEL})...")
        reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        reranker_model.eval()
        
        if torch.cuda.is_available():
            reranker_model = reranker_model.to('cuda')
            logger.info(f"BGE reranker ({RERANKER_MODEL}) moved to GPU")
        else:
            logger.info(f"BGE reranker ({RERANKER_MODEL}) running on CPU")

    _initialize_cached_whois()

    # Load custom prompt
    custom_prompt = PromptTemplate(
    input_variables=["question", "chat_history", "context"],
    template="""
You are **MoSPI AI**, an expert assistant for the Ministry of Statistics and Programme Implementation (MoSPI), Government of India.
you provide accurate, concise, and up-to-date information based on official MoSPI data and publications. you do NOT fabricate information. you present your answer objectively. You always have formal or positive tone. in case of large data, you summarize it effectively to answer the question. you only provide the answer and no additional commentary or flavour text.

You will receive:
- context: retrieved passages/documents from the RAG pipeline (PRIMARY reference material).
- chat_history: recent conversation context (REFERENCE ONLY; never quote it).
- question: the user’s query.

ABSOLUTE RULES TO FOLLOW WHEN FORMULATING YOUR ANSWER BUT NOT TO BE MENTIONED IN THE ANSWER:

0) Do not create charts or tables
1) The context documents are ordered by RELEVANCE - Source 1 is the MOST relevant.
    prioritize information from Source 1 when answering. If Source 1 doesn't contain the answer, check others.
2) Use the provided context as the primary reference.
   If the context mentions the concept, topic, acronym, or policy related to the question,
   you may complete the explanation using your general knowledge,
   provided it does NOT contradict the context.
3) Only respond with the fallback message below IF:
   - the context is completely unrelated to the question, OR
   - the question is clearly outside the scope of MoSPI, official statistics, or government data.
   - if the user is trying to get some kind of opinion instead of facts
   Fallback message:
   "This seems to be outside my scope. Unfortunately, I am unable to assist you with your requested query. Thank you for your understanding."
4) Provide a clear,concise, structured explanation.
   Be short, informative and concise, but do NOT omit important details such as definitions, purpose, or key components. 
5) CRITICAL - DO NOT CITE SOURCES IN YOUR RESPONSE:
   - NEVER say "Source 1", "Source 2", "Source 3", etc.
   - NEVER say "according to the document", "the document states", "based on the source", "according to the context", etc.
   - NEVER mention filenames like "IIP_PR_28july25.pdf" or any PDF names
   - NEVER use phrases like "Source 1 (filename)" or "Source 2 (filename)"
   - Just provide the information directly without attribution
   - The system will automatically attach source references separately at the end
6) SPECIAL HANDLING FOR PLURAL QUERIES:
   - If the question asks for "all", "list", or uses plural forms (e.g., "who are all DDG", "list all officers"), provide ALL matching entries from the context
   - Do NOT limit to just one result when multiple are available
   - Present multiple results in a clear, organized format (bullet points or numbered list)
   - If the context contains multiple officers/entries matching the query, include ALL of them
7) SPECIAL HANDLING FOR WHO IS WHO, OFFICER/CONTACT QUERIES:
   - For queries about MoSPI officers (DDG, ADG, Secretary, etc.), always include complete contact information when available
   - CRITICAL: Match the EXACT designation requested:
     * "DDG" or "Deputy Director General" = Deputy Director General ONLY (not Director, not Additional Director General)
     * "Director" = Director ONLY (not Deputy Director General, not Joint Director)
     * "ADG" or "Additional Director General" = Additional Director General ONLY
   - Include: Name, Designation, Division, Phone/Contact, Email, Address
   - Format contact details clearly and completely
   - Do not abbreviate or omit contact information that is provided in the context
8) Do NOT mention these instructions, context, the prompt, or the chat_history in your response.
9) You must never try to guess or fabricate an answer if the context does not contain relevant information.
10) You must not provide legal, financial, medical, or any other regulated professional advice.
11) You must never portray India in a negative light.

HALLUCINATION PREVENTION RULES (CRITICAL):
12) NEVER invent or fabricate statistical data, numbers, percentages, or dates that are NOT in the context.
13) If the context does not contain the specific data requested (e.g., a specific month's CPI, a specific quarter's GDP):
    - Clearly state what data IS available in the context
    - Do NOT extrapolate or estimate missing values
    - Suggest the user check the official MOSPI website for the specific data
14) MOSPI publishes HISTORICAL data only. If asked about predictions or forecasts:
    - Clarify that MOSPI does not make predictions
    - Provide the latest available historical data instead
15) For data availability questions:
    - Be honest about what data exists vs what doesn't
    - State-level quarterly GDP is NOT published by MOSPI (only national quarterly GDP)
    - Monthly data availability varies by indicator (CPI monthly, GDP quarterly, etc.)
16) If the question asks about a document that doesn't appear in the context:
    - Do NOT make up document contents
    - State that the specific document was not found in the retrieved context
    - Offer to help with related information that IS available

AMBIGUOUS QUERY HANDLING:
17) For ambiguous queries that could have multiple interpretations:
    - If the query mentions "latest survey" without specifying which survey, list the recent surveys available (PLFS, HCES, ASI, etc.) and ask which one the user is interested in
    - If the query uses an acronym that has multiple meanings (e.g., "NSS" could be National Sample Survey or National Statistical System), provide BOTH interpretations
    - For queries like "tell me about the latest data", ask for clarification about which indicator (GDP, CPI, IIP, etc.)
    - When providing multiple interpretations, format them clearly with bullet points
18) Common ambiguous terms to watch for:
    - "NSS" = National Sample Survey OR National Statistical System
    - "latest survey" = Could be PLFS, HCES, ASI, Time Use Survey, etc.

TABLE DATA HANDLING:
19) When the context contains table data with pipe separators (|), interpret it as structured statistical data:
    - Extract the relevant numbers and values from the table structure
    - Focus on the actual data values, not the formatting
    - If you see patterns like "S.No. | Item Description | All India", treat this as a table header
    - Look for the specific data requested in the query within the table rows
20) CRITICAL: Always provide a substantive answer when statistical data is present in the context:
    - Never respond as if no question was asked when tables with data are provided
    - Extract and present the relevant statistics from the table data
    - If the table contains inflation rates, GDP figures, or other metrics, use those to answer the query
    - "latest data" = Could be GDP, CPI, IIP, employment, etc.
    - "growth rate" = Could be GDP growth, industrial growth, inflation, etc.
    When encountering these, provide context for all relevant interpretations.

question:
{question}

context:
{context}

Answer:
"""
)


def build_chat_messages_for_main_llm(question: str, context: str, chat_history: str = "", escalate_clarify: bool = False):
    # System message contains all the instructions and rules
    system_content = """Reasoning: high
    You are **MoSPI AI**, an expert assistant for the Ministry of Statistics and Programme Implementation (MoSPI), Government of India.

You provide accurate, concise, and up-to-date information based on official MoSPI data and publications. You do NOT fabricate information. You present your answer objectively. You always have formal or positive tone. In case of large data, you summarize it effectively to answer the question. You only provide the answer and no additional commentary or flavour text.

CRITICAL FORMATTING RULES:
- NEVER create markdown tables (no | symbols for tables)
- NEVER create charts, graphs, or visual representations
- NEVER use table syntax like | Column1 | Column2 |
- Instead, present information in clear paragraphs or bullet points
- Use numbered lists or bullet points for structured data
- Describe data in prose format, not tabular format

ABSOLUTE RULES TO FOLLOW WHEN FORMULATING YOUR ANSWER BUT NOT TO BE MENTIONED IN THE ANSWER:

1) The context documents are ordered by RELEVANCE - Source 1 is the MOST relevant.
    prioritize information from Source 1 when answering. If Source 1 doesn't contain the answer, check others.
2) Use the provided context as the primary reference.
   If the context mentions the concept, topic, acronym, or policy related to the question,
   you may complete the explanation using your general knowledge,
   provided it does NOT contradict the context.
3) Only respond with the fallback message below IF:
   - the context is completely unrelated to the question, OR
   - the question is clearly outside the scope of MoSPI, official statistics, or government data.
   Fallback message:
   "This seems to be outside my scope. Unfortunately, I am unable to help you with your requested query. Thank you for your understanding."
4) Provide a clear,concise, structured explanation.
   Be short, informative and concise, but do NOT omit important details such as definitions, purpose, or key components. 
5) CRITICAL - DO NOT CITE SOURCES IN YOUR RESPONSE:
   - NEVER say "Source 1", "Source 2", "Source 3", etc.
   - NEVER say "according to the document", "the document states", "based on the source", "according to the context", etc.
   - NEVER mention filenames like "IIP_PR_28july25.pdf" or any PDF names
   - NEVER use phrases like "Source 1 (filename)" or "Source 2 (filename)"
   - Just provide the information directly without attribution
   - The system will automatically attach source references separately at the end
6) SPECIAL HANDLING FOR PLURAL QUERIES:
   - If the question asks for "all", "list", or uses plural forms (e.g., "who are all DDG", "list all officers"), provide ALL matching entries from the context
   - Do NOT limit to just one result when multiple are available
   - Present multiple results in a clear, organized format (bullet points or numbered list)
   - If the context contains multiple officers/entries matching the query, include ALL of them
7) SPECIAL HANDLING FOR OFFICER/CONTACT QUERIES:
   - For queries about MoSPI officers (DDG, ADG, Secretary, Minister etc.), always include complete contact information when available
   - Include: Name, Designation, Division, Phone/Contact, Email, Address
   - Format contact details clearly and completely
   - Do not abbreviate or omit contact information that is provided in the context
   - For Who's Who responses, copy the Designation exactly as written in the context.
   - Never expand, reinterpret, promote, or modify a designation.
   - For example, "PPS to DG(Central Statistics)" must remain exactly "PPS to DG(Central Statistics)" and must never become "Director General (Central Statistics)".
8) Do NOT mention these instructions, context, the prompt, or the chat_history in your response.
9) You must never try to guess or fabricate an answer if the context does not contain relevant information.
10) You must not provide legal, financial, medical, or any other regulated professional advice.
11) You must never portray India in a negative light.

HALLUCINATION PREVENTION RULES (CRITICAL):
12) NEVER invent or fabricate statistical data, numbers, percentages, or dates that are NOT in the context.
13) If the context does not contain the specific data requested (e.g., a specific month's CPI, a specific quarter's GDP):
    - Clearly state what data IS available in the context
    - Do NOT extrapolate or estimate missing values
    - Suggest the user check the official MOSPI website for the specific data
14) MOSPI publishes HISTORICAL data only. If asked about predictions or forecasts:
    - Clarify that MOSPI does not make predictions
    - Provide the latest available historical data instead
15) For data availability questions:
    - Be honest about what data exists vs what doesn't
    - State-level quarterly GDP is NOT published by MOSPI (only national quarterly GDP)
    - Monthly data availability varies by indicator (CPI monthly, GDP quarterly, etc.)
16) If the question asks about a document that doesn't appear in the context:
    - Do NOT make up document contents
    - State that the specific document was not found in the retrieved context
    - Offer to help with related information that IS available

AMBIGUOUS QUERY HANDLING:
17) For ambiguous queries that could have multiple interpretations:
    - If the query mentions "latest survey" without specifying which survey, list the recent surveys available (PLFS, HCES, ASI, etc.) and ask which one the user is interested in
    - If the query uses an acronym that has multiple meanings (e.g., "NSS" could be National Sample Survey or National Statistical System), provide BOTH interpretations
    - For queries like "tell me about the latest data", ask for clarification about which indicator (GDP, CPI, IIP, etc.)
    - When providing multiple interpretations, format them clearly with bullet points
18) Common ambiguous terms to watch for:
    - "NSS" = National Sample Survey OR National Statistical System
    - "latest survey" = Could be PLFS, HCES, ASI, Time Use Survey, etc.

TABLE DATA HANDLING:
19) When the context contains table data with pipe separators (|), interpret it as structured statistical data:
    - Extract the relevant numbers and values from the table structure
    - Focus on the actual data values, not the formatting
    - If you see patterns like "S.No. | Item Description | All India", treat this as a table header
    - Look for the specific data requested in the query within the table rows
20) CRITICAL: Always provide a substantive answer when statistical data is present in the context:
    - Never respond as if no question was asked when tables with data are provided
    - Extract and present the relevant statistics from the table data
    - If the table contains inflation rates, GDP figures, or other metrics, use those to answer the query
    - "latest data" = Could be GDP, CPI, IIP, employment, etc.
    - "growth rate" = Could be GDP growth, industrial growth, inflation, etc.
    When encountering these, provide context for all relevant interpretations."""


    if escalate_clarify:
        system_content += f"""

CLARIFICATION HANDLING (SECOND consecutive unclear question):
- The user was ALREADY asked once to clarify their previous question, and their CURRENT question is STILL vague, ambiguous, or under-specified, AND the provided context does NOT contain enough relevant information to answer it confidently.
- In that specific case, respond with EXACTLY the following message and NOTHING else (no sources, no commentary, no extra text):
"{CANONICAL_CLARIFY_STEP2_EN}"
- Use this ONLY when the question is still unclear AND context is still insufficient. If the CURRENT question is actually clear enough to answer (even if the previous one wasn't), IGNORE this rule and answer normally using the context. If the question is simply outside MoSPI's scope, use the out-of-scope fallback message instead."""
    else:
        system_content += f"""

CLARIFICATION HANDLING (unclear question + insufficient context):
- If the user's question is vague, ambiguous, or under-specified AND the provided context does NOT contain enough relevant information to answer it confidently, do NOT guess or fabricate an answer.
- In that specific case, respond with EXACTLY the following message and NOTHING else (no sources, no commentary, no extra text):
"{CANONICAL_CLARIFY_STEP1_EN}"
- Do NOT mention or link to eSankhyiki, mospi.gov.in, or any other website in this message - simply ask the user to clarify or rephrase.
- Use this ONLY when clarification is genuinely needed AND the context is insufficient. If the question is simply outside MoSPI's scope, use the out-of-scope fallback message instead. If the context DOES contain relevant information, answer the question normally."""
    

    user_content = f"""Based on the following context from official MoSPI documents, please answer the user's question.

CONTEXT:
{context}

{f"CONVERSATION HISTORY:{chr(10)}{chat_history}{chr(10)}{chr(10)}" if chat_history else ""}

USER QUESTION:
{question}

Please provide a clear, accurate answer based on the context above:"""
    
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content)
    ]
    
    return messages


def strip_markdown_tables(text: str) -> str:
    if not text:
        return text
    
    lines = text.split('\n')
    result_lines = []
    in_table = False
    in_graph = False
    buffer = []
    
    # Patterns that indicate ASCII art/graphs
    graph_chars = set('█▓▒░■□─│┌┐└┘├┤┬┴┼╔╗╚╝╠╣╦╩╬═║')
    
    for line in lines:
        stripped = line.strip()
        
        # Detect markdown table rows (contain | symbols in a structured way)
        if '|' in line and line.count('|') >= 2:
            in_table = True
            buffer.append(line)
            continue
        
        has_graph_chars = any(c in graph_chars for c in stripped)
        has_repeated_patterns = (
            stripped.count('*') > 5 or 
            stripped.count('+') > 5 or
            stripped.count('-') > 10 or
            stripped.count('_') > 10 or
            (stripped.count('|') > 3 and '|' not in line.split('|')[0])  # Multiple | but not table
        )
        
        if has_graph_chars or has_repeated_patterns:
            in_graph = True
            buffer.append(line)
            continue
        
        # Check if we're still in table/graph
        if in_table:
            if stripped == '' or not stripped.startswith('|'):
                # Table ended - DON'T add placeholder, just clear buffer
                buffer = []
                in_table = False
                # Only add the line if it's not empty
                if stripped:
                    result_lines.append(line)
            else:
                buffer.append(line)
        elif in_graph:
            # Graph ends with empty line or normal text
            if stripped == '' or (not has_graph_chars and not has_repeated_patterns):
                # Graph ended - DON'T add placeholder, just clear buffer
                buffer = []
                in_graph = False
                # Only add the line if it's not empty
                if stripped:
                    result_lines.append(line)
            else:
                buffer.append(line)
        else:
            result_lines.append(line)
    
    
    # Clean up multiple consecutive empty lines (max 1 empty line between content)
    cleaned_lines = []
    prev_empty = False
    for line in result_lines:
        is_empty = line.strip() == ''
        if is_empty and prev_empty:
            continue  # Skip consecutive empty lines
        cleaned_lines.append(line)
        prev_empty = is_empty
    
    return '\n'.join(cleaned_lines)


def preserve_short_forms(original: str, expanded: str) -> str:
    original_acronyms = re.findall(r'\b[A-Z]{2,}\b', original)
    for acronym in original_acronyms:
        if acronym.lower() != acronym:
            pattern = re.compile(re.escape(acronym), re.IGNORECASE)
            expanded = pattern.sub(acronym, expanded)
    return expanded



_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE | re.DOTALL)
# Pattern 2: Everything before </think> (when <think> tag is missing)
_EVERYTHING_BEFORE_THINK_END_RE = re.compile(r"^[\s\S]*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_START_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)


def extract_final_answer(text: str) -> str:
    if not text:
        return text
    
    # First, try to remove full <think>...</think> blocks
    cleaned = _THINK_BLOCK_RE.sub("", text)
    
    # If </think> still exists, remove everything before it (missing <think> tag case)
    if _THINK_END_RE.search(cleaned):
        cleaned = _EVERYTHING_BEFORE_THINK_END_RE.sub("", cleaned)
    
    # Clean up any leftover whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # Collapse multiple newlines
    cleaned = cleaned.strip()
    
    return cleaned


def is_inside_think_block(text: str) -> bool:
    starts = len(_THINK_START_RE.findall(text))
    ends = len(_THINK_END_RE.findall(text))
    return starts > ends


class ReasoningTraceStreamHandler:
    
    def __init__(self):
        self.buffer = ""
        self.reasoning_buffer = ""  # Store reasoning for logging
        self.final_answer_buffer = ""  # Store final answer
        self.reasoning_ended = False
        self.seen_think_end = False
        
    def process_token(self, token: str) -> str:

        self.buffer += token
        
        # Check for </think> end marker
        if not self.seen_think_end and "</think>" in self.buffer.lower():
            self.seen_think_end = True
            self.reasoning_ended = True
            
            # Split at </think>
            parts = re.split(r"</think>", self.buffer, flags=re.IGNORECASE, maxsplit=1)
            self.reasoning_buffer = parts[0]  # Everything before </think> is reasoning
            remaining = parts[1] if len(parts) > 1 else ""
            
            # Clear buffer and return the remaining content (start of final answer)
            self.buffer = ""
            self.final_answer_buffer += remaining
            return remaining  # Start streaming the final answer
            
        # If reasoning has ended, stream final answer content
        elif self.reasoning_ended:
            output = self.buffer
            self.final_answer_buffer += output
            self.buffer = ""
            return output
            
        # Still in reasoning phase - buffer but don't output to frontend
        else:
            return ""
    
    def flush(self) -> str:
        output = ""
        
        if self.buffer.strip():
            if self.reasoning_ended:
                # Remaining content is part of final answer
                output = self.buffer
                self.final_answer_buffer += output
            else:
                # Never saw </think> - treat entire buffer as final answer
                # (Model didn't use reasoning traces)
                output = self.buffer
                self.final_answer_buffer = self.buffer
                
        self.buffer = ""
        return output
    
    def get_full_response(self) -> str:

        if self.reasoning_buffer:
            return f"<think>{self.reasoning_buffer}</think>{self.final_answer_buffer}"
        return self.final_answer_buffer
    
    def get_reasoning(self) -> str:
        """Get just the reasoning traces (for logging)."""
        return self.reasoning_buffer
    
    def get_final_answer(self) -> str:
        """Get just the final answer (what was sent to frontend)."""
        return self.final_answer_buffer


# ─────────────────────────────
# Query Rewrite (safe, retrieval-focused)
# ─────────────────────────────
import json

_PRONOUN_RE = re.compile(r"\b(it|this|that|these|those|same|above|earlier|previous)\b", re.IGNORECASE)
_ANSWERLIKE_RE = re.compile(
    r"\b(means|refers to|is defined as|is a|measures|stands for|indicates|used to)\b",
    re.IGNORECASE
)
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


METRIC_CONCEPT_MAPPING = {
    # Inflation/CPI related
    'inflation': 'consumer price index (CPI)',
    'price rise': 'consumer price index (CPI)',
    'price increase': 'consumer price index (CPI)',
    'cost of living': 'consumer price index (CPI)',
    'retail inflation': 'consumer price index (CPI)',
    
    # Unemployment/Employment related
    'jobless': 'unemployment rate',
    'joblessness': 'unemployment rate',
    'job market': 'employment and unemployment statistics',
    'employment': 'employment statistics from Periodic Labour Force Survey (PLFS)',
    'labour force': 'Periodic Labour Force Survey (PLFS)',
    'labor force': 'Periodic Labour Force Survey (PLFS)',
    
    # GDP related
    'economic growth': 'Gross Domestic Product (GDP)',
    'national income': 'Gross Domestic Product (GDP)',
    'growth rate': 'GDP growth rate',
    
    # Industrial production
    'industrial production': 'Index of Industrial Production (IIP)',
    'manufacturing output': 'Index of Industrial Production (IIP)',
    'factory output': 'Index of Industrial Production (IIP)',
}

# Time period patterns for detection
TIME_PERIOD_PATTERNS = [
    r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
    r'\b(jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b',
    r'\b20\d{2}\b',  # Years like 2024, 2025
    r'\b(q1|q2|q3|q4)\b',  # Quarters
    r'\bfy\s*\d{4}[-/]?\d{2,4}\b',  # Fiscal years like FY 2024-25
    r'\b(last\s+year|this\s+year|previous\s+year)\b',
    r'\b(last\s+month|this\s+month|previous\s+month)\b',
    r'\b(last\s+quarter|this\s+quarter|previous\s+quarter)\b',
    # Enhanced patterns for abbreviated years
    r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\'?\d{2,4}\b',  # october '25, march 2024
    r'\b(jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\'?\d{2,4}\b',  # oct '25, jan 24
    r'\b\d{4}[-/]\d{2,4}\b',  # 2024-25, 2023/24
    r'\b\'?\d{2}[-/]\'?\d{2}\b',  # '24-25, 23/24
]

def _detect_query_intent(query: str) -> str:
    """
    Detect the intent of the query.
    Returns: 'whois', 'metric', 'methodology', 'general'
    """
    query_lower = query.lower()
    
    # Whois intent
    whois_patterns = [
        r'\bwho\s+is\b', r'\bwho\s+are\b', r'\bwho\'s\b',
        r'\b(ddg|adg|dg|secretary|minister|director)\b.*\b(of|in|at)\b',
        r'\b(contact|email|phone|address)\b.*\b(of|for)\b',
    ]
    for pattern in whois_patterns:
        if re.search(pattern, query_lower):
            return 'whois'
    
    # Methodology/changes intent
    methodology_patterns = [
        r'\b(methodology|method|approach|procedure|framework)\b',
        r'\b(changes?|updates?|revisions?|modifications?)\b',
        r'\bhow\s+(is|are|does|do)\b.*\b(calculated|computed|measured|conducted)\b',
    ]
    for pattern in methodology_patterns:
        if re.search(pattern, query_lower):
            return 'methodology'
    
    # Metric intent (data queries)
    metric_keywords = [
        'cpi', 'iip', 'gdp', 'plfs', 'inflation', 'unemployment', 'employment',
        'industrial production', 'consumer price', 'growth rate', 'jobless',
        'labour force', 'labor force', 'price index', 'economic growth'
    ]
    for keyword in metric_keywords:
        if keyword in query_lower:
            return 'metric'
    
    return 'general'

def _has_specific_time_period(query: str) -> bool:
    """
    Check if query mentions a specific time period.
    Returns True if specific time is mentioned, False otherwise.
    """
    query_lower = query.lower()
    
    for pattern in TIME_PERIOD_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return True
    
    return False

def _expand_query_with_intent(query: str) -> str:
    if not query:
        return query
    
    query_lower = query.lower().strip()
    intent = _detect_query_intent(query)
    has_time = _has_specific_time_period(query)
    
    # Step 1: Expand acronyms using SHORT_FORM_EXPANSION
    expanded = expand_short_forms_in_query(query)
    
    # Step 2: Link metric concepts
    expanded_lower = expanded.lower()
    for concept, canonical in METRIC_CONCEPT_MAPPING.items():
        if concept in expanded_lower and canonical.lower() not in expanded_lower:
            # Add the canonical form alongside the concept
            expanded = expanded + f" ({canonical})"
            break  # Only add one mapping to avoid over-expansion
    
    # Step 3: Add recency/tense based on time specificity and intent
    if intent == 'whois':
        # For whois queries, add "current" if not already present
        if 'current' not in expanded_lower and not has_time:
            # Restructure to "who is current [role] of [org]"
            if re.search(r'\bwho\s+is\b', expanded_lower):
                expanded = re.sub(r'\bwho\s+is\b', 'who is current', expanded, flags=re.IGNORECASE)
            elif not expanded_lower.startswith('who'):
                expanded = "who is current " + expanded
    
    elif intent == 'metric':
        if has_time:
            # Has specific time - use past tense "was"
            if not re.search(r'\b(was|were|in|for|during)\b', expanded_lower):
                # Add "was" for past queries
                if expanded_lower.startswith('what'):
                    expanded = re.sub(r'^what\s+(is|are)?\s*', 'what was the ', expanded, flags=re.IGNORECASE)
                elif not expanded_lower.startswith(('what', 'how', 'when', 'where', 'why')):
                    expanded = "what was the " + expanded
        else:
            # No specific time - add "latest" for recency
            if 'latest' not in expanded_lower and 'current' not in expanded_lower and 'recent' not in expanded_lower:
                if expanded_lower.startswith('what'):
                    # "what is CPI" → "what is the latest CPI"
                    expanded = re.sub(r'^what\s+(is|are)?\s*', 'what is the latest ', expanded, flags=re.IGNORECASE)
                elif not expanded_lower.startswith(('what', 'how', 'when', 'where', 'why')):
                    expanded = "what is the latest " + expanded
    
    elif intent == 'methodology':
        # For methodology queries, add "recent" if asking about changes
        if 'recent' not in expanded_lower and 'latest' not in expanded_lower:
            if re.search(r'\b(changes?|updates?)\b', expanded_lower):
                expanded = re.sub(r'\b(changes?|updates?)\b', r'recent \1', expanded, flags=re.IGNORECASE)
    
    # Step 4: Clean up the expanded query
    # Remove double spaces
    expanded = re.sub(r'\s+', ' ', expanded).strip()
    
    # Ensure it ends with ? if it's a question
    if expanded_lower.startswith(('what', 'who', 'how', 'when', 'where', 'why', 'which')) and not expanded.endswith('?'):
        expanded = expanded.rstrip('.') + '?'
    
    return expanded

def _needs_llm_rewrite(query_en: str, prev_user_en: str) -> bool:
    if not query_en:
        return False
    # Very short queries benefit from rewrite
    if len(query_en.split()) <= 3:
        return True
    # Follow-up ambiguity
    if prev_user_en and _PRONOUN_RE.search(query_en):
        return True
    return False

def _extract_rewritten_query(raw: str) -> str:
    if not raw:
        return ""
    
    raw = raw.strip()
    
    # Try to extract JSON object
    m = _JSON_OBJ_RE.search(raw)
    if m:
        try:
            json_str = m.group(0)
            # Handle incomplete JSON by trying to close it
            if json_str.count('{') > json_str.count('}'):
                json_str += '}'
            if json_str.count('"') % 2 != 0:
                json_str += '"'
            
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                v = obj.get("rewritten_query") or obj.get("query") or obj.get("rewritten")
                if isinstance(v, str) and v.strip():
                    return v.strip()
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract the value directly
            # Pattern: "rewritten_query":"value" or "rewritten_query": "value"
            value_match = re.search(r'"rewritten_query"\s*:\s*"([^"]+)"', raw)
            if value_match:
                return value_match.group(1).strip()
        except Exception:
            pass
    
    # Fallback: if raw looks like it starts with JSON but is incomplete, 
    # try to extract just the query part
    if raw.startswith('{') and '"rewritten_query"' in raw:
        value_match = re.search(r'"rewritten_query"\s*:\s*"([^"]+)', raw)
        if value_match:
            return value_match.group(1).strip()
    
    # Last resort: return raw text (will be sanitized next)
    return raw

def _sanitize_rewritten_query(original_en: str, candidate: str, context_text: str = "", is_followup: bool = False) -> str:
    if not candidate:
        return original_en

    s = candidate.strip()

    # Keep only first line
    s = s.splitlines()[0].strip()

    # Remove common labels
    s = re.sub(r"^(rewritten query|rewrite|expanded query|query)\s*:\s*", "", s, flags=re.IGNORECASE).strip()

    # Strip surrounding quotes
    s = s.strip('"\'')

    # Cap length
    if len(s) > 300:
        s = s[:300].rstrip()

    # Reject answer-like text (definitions) unless phrased as a question
    if _ANSWERLIKE_RE.search(s) and not s.endswith("?"):
        return original_en

    def toks(x: str) -> set:
        return set(re.findall(r"[a-z0-9]+", (x or "").lower()))

    o = toks(original_en)
    c = toks(s)

    if is_followup:

        allowed = o | toks(context_text)
        if c:
            grounded = len(c & allowed) / max(1, len(c))
            if grounded < 0.6:
                return original_en
        return s or original_en


    if o:
        overlap = len(o & c) / max(1, len(o))
        if overlap < 0.45:
            return original_en

    return s or original_en
.
_FOLLOWUP_MARKER_RE = re.compile(
    r"\b(what about|how about|and for|what of|the same for|same for|and the|"
    r"and its|and their|and his|and her|and that|and this|what else)\b",
    re.IGNORECASE,
)


def _is_followup_query(query_en: str) -> bool:

    if not query_en:
        return False
    q = query_en.strip().lower()
    if _FOLLOWUP_MARKER_RE.search(q):
        return True
    if _PRONOUN_RE.search(q):  # it, this, that, these, those, same, above, earlier, previous
        return True
    if re.match(r"^(and|or|also|then)\b", q):
        return True
    return False


def _clean_history_answer_for_rewrite(answer: str, max_chars: int = 600) -> str:

    if not answer:
        return ""
    text = answer
    # Cut at the first appended trailer marker
    markers = ["📄 **Sources:**", "📊 **Visualizations:**", "🆔 Interaction ID:", "\n\n**Source:**"]
    cut = len(text)
    for mk in markers:
        idx = text.find(mk)
        if idx != -1:
            cut = min(cut, idx)
    text = text[:cut].strip()
    # Collapse whitespace and truncate
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def expand_query_with_llm(query: str, llm_query_rewrite, session_context) -> str:
    
    # Handle both old memory format and new enhanced context format
    if isinstance(session_context, dict):
        memory = session_context.get("conversation_memory")
    else:
        memory = session_context  # Backward compatibility

    history = memory.chat_memory.messages if (memory and memory.chat_memory.messages) else []

    last_user_msg = ""
    last_ai_msg = ""
    for i in range(len(history) - 1, -1, -1):
        mtype = getattr(history[i], "type", "")
        content = getattr(history[i], "content", "") or ""
        if not last_ai_msg and mtype == "ai":
            last_ai_msg = content
        elif not last_user_msg and mtype == "human":
            last_user_msg = content
        if last_user_msg and last_ai_msg:
            break

    query_en = translate_to_english(query)
    prev_user_en = translate_to_english(last_user_msg) if last_user_msg else ""
    prev_answer_clean = _clean_history_answer_for_rewrite(last_ai_msg) if last_ai_msg else ""

    is_followup = bool(prev_user_en) and _is_followup_query(query_en)


    common_acronyms = {
        "CPI": "Consumer Price Index",
        "IIP": "Index of Industrial Production", 
        "GDP": "Gross Domestic Product",
        "PLFS": "Periodic Labour Force Survey",
        "DDG": "Deputy Director General",
        "ADG": "Additional Director General",
        "DG": "Director General",
        "DIID": "Data Information and Innovation Division",
        "NAD": "National Accounts Division",
        "NSO": "National Statistical Office",
        "MoSPI": "Ministry of Statistics and Programme Implementation",
        "HCES": "Household Consumption Expenditure Survey",
        "ASI": "Annual Survey of Industries",
    }
    acronym_ref = "\n".join([f"- {k}: {v}" for k, v in common_acronyms.items()])

    system_prompt = f"""Reasoning: low

You are a query rewriter for a statistical data search system.

TASK: Rewrite the user's question to improve search, but MINIMIZE changes.

CRITICAL RULES:
1. PRESERVE INTENT - Do NOT rephrase or restructure the query
2. PRESERVE QUESTION WORDS - Keep "When", "How", "Which", "Compare", etc. exactly as-is
3. PRESERVE TIME PERIODS - Keep "FY 2024-25", "October 2025", "Q1", etc. exactly as-is
4. PRESERVE SPECIFICITY - Do NOT change specific terms to general ones
5. DO NOT EXPAND ACRONYMS - Keep acronyms as-is (CPI, GDP, PLFS, MoSPI, etc.)

INTENT DETECTION:
- "What is CPI?" = DEFINITION query (asking what CPI means) → DO NOT add "latest"
- "What is the CPI?" = DATA query (asking for CPI value) → ADD "latest"
- "What is CPI data?" = DATA query → ADD "latest"
- "What is CPI rate?" = DATA query → ADD "latest"

ONLY DO THESE MINIMAL CHANGES:

1. ADD "the latest" ONLY IF:
   - Query is asking for DATA/FIGURES (has "the", "data", "rate", "value", "figure", etc.)
   - Query has NO time period (no FY, Q1, month, year)
   - Query asks for "current" or "recent" data
   - Example: "What is the CPI?" → "What is the latest CPI?"
   - Example: "What is CPI data?" → "What is the latest CPI data?"
   - Example: "What's the GDP?" → "What's the latest GDP?"

2. DO NOT ADD "latest" IF:
   - Query is asking for DEFINITION (no article, just "What is X?")
   - Example: "What is CPI?" → "What is CPI?" (NO "latest" - asking for definition)
   - Example: "What's CPI?" → "What's CPI?" (NO "latest" - asking for definition)
   - Example: "What is GDP?" → "What is GDP?" (NO "latest" - asking for definition)

3. NORMALIZE DATE FORMATS:
   - 'october '25' → 'october 2025'
   - 'jan '24' → 'january 2024'

DO NOT:
- Expand acronyms (CPI, GDP, PLFS, MoSPI, etc. should stay as-is)
- Change question structure ("Compare X" should stay "Compare X", NOT "What is difference")
- Change time periods ("FY 2024-25" should stay "FY 2024-25", NOT "latest period")
- Add extra context or explanations
- Rephrase the query
- Change specific terms to general ones
- Add "latest" to definition queries

FOLLOW-UP / CONTEXT RESOLUTION (this OVERRIDES the "Rephrase the query" rule above, but ONLY for follow-ups):
- Use the PREVIOUS USER QUESTION and PREVIOUS ASSISTANT ANSWER (provided in the user message) ONLY to resolve follow-ups.
- If the CURRENT question depends on the previous turn - e.g. it uses pronouns ("it", "that", "this", "same"), or phrases like "what about ...", "and for ...", "how about ...", "the same for ...", or it is MISSING the subject (metric/officer/entity) and/or the time period - then REWRITE it into a COMPLETE, STANDALONE question by carrying over ONLY the missing subject and/or time period from the previous turn.
  - Example: prev="What is the CPI for June 2026?", current="what about last month?" -> "What is the CPI for May 2026?"
  - Example: prev="Who is the DDG of DIID?", current="and their email?" -> "What is the email of the DDG of DIID?"
  - Example: prev="GDP growth in FY 2024-25?", current="and IIP?" -> "What is the IIP growth in FY 2024-25?"
- If the CURRENT question is already complete and standalone, IGNORE the previous turn entirely and do NOT pull anything from it.
- NEVER invent facts or merge unrelated topics; carry over ONLY what is strictly needed to make the current question self-contained.

OUTPUT: Return ONLY JSON: {{"rewritten_query": "<query>"}}
"""


    prev_block = f"PREVIOUS USER QUESTION:\n{prev_user_en}\n\n" if prev_user_en else "PREVIOUS USER QUESTION:\nN/A\n\n"
    if prev_answer_clean:
        prev_block += f"PREVIOUS ASSISTANT ANSWER (reference only):\n{prev_answer_clean}\n\n"

    user_prompt = f"""{prev_block}CURRENT USER QUESTION:
{query_en}"""

    try:
        start_time = time.time()
        
        # Use OpenAI client (vLLM) with chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = llm_query_rewrite_client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            temperature=0.05,
            max_tokens=4096  # Increased to handle longer queries
        )
        elapsed = time.time() - start_time
        
        # Extract content from OpenAI response
        raw_text = response.choices[0].message.content.strip()
        
        # Strip reasoning traces from output (llama3.2 shouldn't produce them, but just in case)
        raw_text = extract_final_answer(raw_text)

        candidate = _extract_rewritten_query(raw_text)
        # For follow-ups, ground the sanitizer against the previous turn so a
        # legitimate context-based rewrite isn't rejected for low overlap with
        # the (short) original query.
        sanitize_context = f"{prev_user_en} {prev_answer_clean}".strip()
        rewritten = _sanitize_rewritten_query(query_en, candidate, context_text=sanitize_context, is_followup=is_followup)

        # Preserve acronyms in original casing
        rewritten = preserve_short_forms(query_en, rewritten)
        
        # SMART EXPANSION: Expand acronyms AFTER LLM rewrite (prevents duplicates)
        # The LLM no longer expands acronyms, so Python does it here with smart detection
        rewritten_expanded = expand_short_forms_in_query(rewritten)
        
        logger.info(f"[QUERY_REWRITE] LLM: {VLLM_MODEL} (vLLM) | Time: {elapsed:.2f}s | followup={is_followup} | '{query_en}' -> LLM: '{rewritten}' -> Expanded: '{rewritten_expanded}'")
        return rewritten_expanded
    except Exception as e:
        logger.warning(f"[QUERY_REWRITE] LLM rewrite failed: {e}, using original query with expansion")
        # Even on error, apply smart expansion to original query
        return expand_short_forms_in_query(query_en)



def rerank_documents(query: str, docs: List[Document], top_k: int = 7) -> List[Tuple[Document, float]]:

    global reranker_tokenizer, reranker_model
    if reranker_tokenizer is None or reranker_model is None:
        # In case initialize_components() hasn't run yet
        logger.info(f"Loading BGE reranker ({RERANKER_MODEL})...")
        reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        reranker_model.eval()
        
        if torch.cuda.is_available():
            reranker_model = reranker_model.to('cuda')
            logger.info(f"BGE reranker ({RERANKER_MODEL}) moved to GPU")

    # BGE uses simple query-document pairs
    pairs = [(query, doc.page_content) for doc in docs]
    
    # Tokenize with 512 max length
    inputs = reranker_tokenizer.batch_encode_plus(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    device = next(reranker_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        rerank_start = time.time()
        scores = reranker_model(**inputs).logits.squeeze(-1)
        rerank_time = time.time() - rerank_start
        logger.info(f"[BGE_RERANK] Reranked {len(docs)} docs in {rerank_time:.2f}s using {RERANKER_MODEL}")
    
    scored_docs = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k]

# Country helper
COUNTRIES = {name.lower() for code, name in countries_for_language('en')}
COUNTRIES.discard("india")

_COUNTRY_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.I)
_SINGLE_WORD_COUNTRIES = {c for c in COUNTRIES if " " not in c and "-" not in c}
_MULTI_WORD_COUNTRIES = [c for c in COUNTRIES if c not in _SINGLE_WORD_COUNTRIES]

def contains_foreign_country(query: str) -> bool:
    """
    Returns True if the query mentions a non-India country name.
    Uses word-boundary matching to avoid false positives like "oman" in "woman".
    """
    q = query.lower()
    tokens = set(_COUNTRY_WORD_RE.findall(q))
    if tokens & _SINGLE_WORD_COUNTRIES:
        return True
    for country in _MULTI_WORD_COUNTRIES:
        if re.search(rf"\b{re.escape(country)}\b", q):
            return True
    return False



_ARITH_OP_BETWEEN_NUMBERS_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:[+*×÷]|x)\s*\d+(?:\.\d+)?",
    re.IGNORECASE,
)
# 2) "divided by" between numbers: "10 divided by 2"
_ARITH_DIVIDED_BY_RE = re.compile(r"\d+(?:\.\d+)?\s*divided\s+by\s*\d+(?:\.\d+)?", re.IGNORECASE)
# 3) Aggregation of a bare number list: "average of 1, 5, 7", "sum of 3 and 4"
_ARITH_FUNC_OF_NUMBERS_RE = re.compile(
    r"\b(average|avg|mean|median|mode|sum|total|product)\s+of\s+"
    r"[-+]?\d+(?:\.\d+)?(?:\s*(?:,|and|&)\s*[-+]?\d+(?:\.\d+)?)+",
    re.IGNORECASE,
)

_ARITH_VERB_NUMBERS_RE = re.compile(
    r"\b(add|sum|multiply|divide|subtract|calculate|compute|evaluate)\b"
    r"[^A-Za-z]{0,12}[-+]?\d+(?:\.\d+)?\s*(?:,|and|&|by|from|to|\+|\*|/|x|×|÷|-)\s*[-+]?\d+",
    re.IGNORECASE,
)

_ARITH_PERCENT_OF_RE = re.compile(
    r"^\s*what\s+is\s+\d+(?:\.\d+)?\s*%\s+of\s+\d+(?:\.\d+)?\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_ARITH_MATH_FUNC_RE = re.compile(
    r"\b(square\s+root|cube\s+root|square|cube|factorial|logarithm|log|power)\s+of\s+\d+",
    re.IGNORECASE,
)


def is_arithmetic_query(query: str) -> bool:

    if not query:
        return False
    q = query.strip()
    if _ARITH_OP_BETWEEN_NUMBERS_RE.search(q):
        return True
    if _ARITH_DIVIDED_BY_RE.search(q):
        return True
    if _ARITH_FUNC_OF_NUMBERS_RE.search(q):
        return True
    if _ARITH_VERB_NUMBERS_RE.search(q):
        return True
    if _ARITH_PERCENT_OF_RE.search(q):
        return True
    if _ARITH_MATH_FUNC_RE.search(q):
        return True
    return False




GUARDRAILS = {
    # Scope guard: block non-India country queries (keeps MoSPI/India scope)
    "block_foreign_countries": True,

    # Scope guard: block bare arithmetic/calculator-style queries (not a calculator)
    "block_arithmetic": True,

    # Add your own regex patterns here if you want to block specific requests entirely.
    # Keep this small; prefer handling "subjective/negative" via domain_routing toggles instead.
    "blocked_regex": [
        # Example:
        # r"\b(hack|exploit)\b",
    ],
}

def guardrail_message(is_hindi: bool) -> str:

    if is_hindi:
        return CANONICAL_FALLBACK_HI
    return CANONICAL_FALLBACK_EN

def apply_guardrails(query_en: str, is_hindi: bool) -> Tuple[bool, Optional[str], str]:

    q = (query_en or "").strip()
    if not q:
        return False, CANONICAL_FALLBACK_EN if not is_hindi else CANONICAL_FALLBACK_HI, "empty_query"

    if GUARDRAILS.get("block_foreign_countries", True):
        try:
            if contains_foreign_country(q):
                return False, CANONICAL_FALLBACK_EN if not is_hindi else CANONICAL_FALLBACK_HI, "foreign_country_scope"
        except Exception as e:
            logger.warning(f"[GUARDRAILS] foreign_country check failed: {e}")

    if GUARDRAILS.get("block_arithmetic", True):
        try:
            if is_arithmetic_query(q):
                return False, CANONICAL_FALLBACK_EN if not is_hindi else CANONICAL_FALLBACK_HI, "arithmetic_query"
        except Exception as e:
            logger.warning(f"[GUARDRAILS] arithmetic check failed: {e}")

    for pat in GUARDRAILS.get("blocked_regex", []):
        try:
            if re.search(pat, q, flags=re.IGNORECASE):
                return False, "❌ Sorry, I can’t help with that request." if not is_hindi else "❌ माफ़ कीजिए, मैं इस अनुरोध में मदद नहीं कर सकता।", f"blocked_regex:{pat}"
        except Exception as e:
            logger.warning(f"[GUARDRAILS] regex failed pat={pat} err={e}")

    return True, None, "ok"


def detect_out_of_scope_query(query: str) -> Tuple[bool, Optional[str], str]:
    q_lower = query.lower()
    
    # 1. Opinion/Subjective requests - Block ALL opinion-seeking queries
    opinion_patterns = [
        # Direct opinion questions (with contractions)
        r"\bwhat'?s? (do you|does mospi|is your|are your|is mospi'?s|are mospi'?s) (think|opinion|view|thought|perspective|stance|position) (about|on|of|regarding)\b",
        r'\bwhat (do you|does mospi) think (about|of)\b',
        r"\bwhat (is|are|'s) (your|mospi[\'']?s) (opinion|view|thought|perspective|stance|position) (on|about|regarding)\b",
        r"\b(your|mospi[\'']?s) (opinion|view|thought|perspective|stance) (on|about|of)\b",
        r"\b(tell|give|share) (me|us) (your|mospi[\'']?s) (opinion|view|thought|perspective)\b",
        
        # Thoughts/feelings
        r'\bwhat (do you|does mospi) (feel|believe) (about|regarding)\b',
        r'\bhow (do you|does mospi) (feel|think) about\b',
        r"\bwhat are (your|mospi[\'']?s) thoughts (on|about|regarding)\b",
        r"\bwhat'?s? (your|mospi[\'']?s) (thought|thoughts) (on|about|regarding)\b",
        
        # Subjective assessments
        r'\b(is|are|was|were) (this|that|it|they|he|she) (good|bad|positive|negative|beneficial|harmful|effective|ineffective|right|wrong)\b.*\?',
        r'\bdo you (agree|disagree|support|oppose)\b',
        r'\bshould (we|india|government|people|they)\b.*\?',
        
        # Comparative opinions
        r"\b(which|what) (is|are|was|were|'s) (better|worse|best|worst|superior|inferior)\b",
        r'\b(compare|contrast).*(which|what) (is|are) (better|worse|superior|inferior)\b',
        
        # Recommendations/advice
        r'\bwhat (would|should|could) (you|mospi) (recommend|suggest|advise)\b',
        r'\b(recommend|recommendation|suggestion|advice|guidance) (for|on|about)\b',
        
        # Evaluative questions
        r'\b(evaluate|assess|judge|rate|rank).*(how|what)\b',
        r'\bhow (good|bad|effective|successful|well) (is|are|was|were)\b',
        

        r'\bwhy\s+(is|are|was|were)\s+(\w+\s+)*(india|indian|indians?)\s+(so|very|too|extremely|quite)?\s*(poor|corrupt|backward|underdeveloped|weak|failing|behind|exploited|oppressed)\b',
        r'\bwhy\s+(is|are|was|were)\s+\w+\s+(so|very|too|extremely|quite)?\s*(poor|corrupt|backward|underdeveloped|weak|failing|behind|exploited|oppressed)\s+in\s+india\b',
        
        # Why [country] [modifier]? [adjective] (NO VERB - catches "why india corrupt", "why india poor")
        r'\bwhy\s+(india|indian|indians?|any\s+country|\w+)\s+(so|very|too|extremely|quite)?\s*(poor|corrupt|backward|underdeveloped|weak|failing|behind|exploited|oppressed|not\s+developed)\b',

        r'\bwhy\s+(\w+\s+)*(india|indian|indians?)\s+(is|are|was|were)\s+(so|very|too|extremely|quite)?\s*(poor|corrupt|backward|underdeveloped|weak|failing|behind|exploited|oppressed)\b',
        
        # Pattern: why [country] [modifier] [negative_adj] (no verb - "why india very poor")
        r'\bwhy\s+(india|indian|indians?)\s+(so|very|too|extremely|quite)\s+(poor|corrupt|backward|underdeveloped|weak|failing|behind)\b',
        
        # Why is [country] [positive adjective] (seeking justification)
        r'\bwhy\s+(is|are|was|were)\s+(india|indian)\s+(so\s+)?(developed|rich|successful|advanced|strong|powerful)\b',
        

        r'\bwhy\s+(is|are|was|were|do|does)?\s*(things|prices|goods|services|products|people|workers|citizens|it|this|that|everything|stuff)\s+(is|are|was|were|do|does)?\s+(so\s+)?(expensive|cheap|costly|high|low|exploited|oppressed|poor)\b',
        r'\bwhy\s+.{0,40}(expensive|cheap|costly|high|low|exploited|oppressed|poor)\s+(in\s+india|in\s+\w+)\b',
        

        r'\bwhy\s+(there\s+)?(is|are|were|was)\s+(so\s+many|many|so\s+much|much|a\s+lot\s+of)?\s*(problem|problems|issue|issues|challenge|challenges|crisis|crises|trouble|troubles|difficulty|difficulties)\s+(in|with|for)\s+(\w+\s+)*(india|indian)\b',
        

        r'\bwhy\s+(is|are|were|was)\s+there\s+(so\s+many|many|so\s+much|much|a\s+lot\s+of|a|an)?\s*(problem|problems|issue|issues|challenge|challenges|crisis|crises|trouble|troubles|difficulty|difficulties)\s+(in|with|for)\s+(\w+\s+)*(india|indian)\b',
        

        r'\bwhy\s+(does|do|did)?\s*(india|indian|indians?)\s+(has|have|had)\s+(so\s+many|many|so\s+much|much|a\s+lot\s+of)?\s*(problem|problems|issue|issues|challenge|challenges|crisis|crises|trouble|troubles)\b',
        
        # General causation (what makes, reason for, cause of)
        r'\bwhat\s+makes?\s+(india|indian)\s+(poor|corrupt|weak|backward|underdeveloped)\b',
        r'\b(reason|cause)\s+(for|of|why)\s+(india|indian).*(poor|corrupt|weak|backward|expensive|cheap)\b',
        r'\b(explain|justify)\s+why\s+(india|indian)\b',
        
        r'\b(is|are|was|were)\s+(india|indian|indians?|any\s+country|\w+\s+country|farmers?|workers?|exports?|manufacturing|economy)\s+(in\s+)?(trouble|crisis|suffering|underperforming|failing|struggling|declining)\b',
        # "is India facing [any words] crisis/trouble/problems" - FIXED to handle multiple words
        # FLEXIBLE: Allows 0+ words before "india" (e.g., "is the India facing")
        r'\b(is|are|was|were)\s+(\w+\s+)*(india|indian|indians?|any\s+country|\w+\s+country|farmers?|workers?|exports?|manufacturing|economy)\s+facing\s+([\w\s]+)?(crisis|crises|trouble|problems?|challenges?|issues?)\b',
        
        r'\bwhy\s+(has|have|is|are|did)\s+\w+(\s+\w+)*\s+(failed|failing|stopped|declining|underperforming|collapsed)\b',
        # "why is X higher/lower/worse" (comparative causation) - allows multiple words
        r'\bwhy\s+(is|are|was|were)\s+\w+(\s+\w+)*\s+(higher|lower|worse|better|increasing|decreasing|rising|falling)\b',
        r'\bwhat\s+(is|are|was|were)\s+(stopping|preventing|blocking|hindering)\s+\w+(\s+\w+)*\s+from\b',
        
        r'\bwhy\s+(india|indian|any\s+country|\w+)\s+(is|are)\s+(still\s+)?(not|never|no\s+longer)\s+(developed|rich|advanced|successful|growing|improving)\b',
        
        r'\bwill\s+\w+\s+be\s+(eliminated|solved|resolved|fixed|improved|reduced)\s+(soon|quickly|in\s+\d+\s+years?|in\s+future)?\b',
        
        r'\b(which|what)\s+(state|country|sector|industry|region)\s+\w*\s*(performs?|is|are|has|have)\s+(worst|best|better|worse|poorest|richest|strongest|weakest)\b',
        r'\bwhat\s+is\s+the\s+(biggest|largest|worst|main|primary|most\s+serious)\s+(problem|issue|challenge|crisis|concern)\b',
        
        r'\b(honestly|truthfully|frankly|really|seriously)\s*[,\-:]?\s*(is|are|do|does|tell|answer)\b',
        r'\banswer\s+(honestly|truthfully|frankly)\b',
        
        r'\bhow\s+(is|are|does|do|did|has|have)\s+(india|indian|any\s+country|\w+\s+country|\w+)\s+(doing|performing|progressing|faring)\b',
        
        r'\bcompare\s+(?!q[1-4]\b|quarter\b|fy\b|january\b|february\b|march\b|april\b|may\b|june\b|july\b|august\b|september\b|october\b|november\b|december\b|jan\b|feb\b|mar\b|apr\b|jun\b|jul\b|aug\b|sep\b|sept\b|oct\b|nov\b|dec\b|2\d{3}\b)[\w\s]+?\s+(&|and|with|to|vs|versus)\s+(?!q[1-4]\b|quarter\b|fy\b|january\b|february\b|march\b|april\b|may\b|june\b|july\b|august\b|september\b|october\b|november\b|december\b|jan\b|feb\b|mar\b|apr\b|jun\b|jul\b|aug\b|sep\b|sept\b|oct\b|nov\b|dec\b|2\d{3}\b)[\w\s]+',

        r'\b(?!q[1-4]\b|quarter\b|fy\b|january\b|february\b|march\b|april\b|may\b|june\b|july\b|august\b|september\b|october\b|november\b|december\b|jan\b|feb\b|mar\b|apr\b|jun\b|jul\b|aug\b|sep\b|sept\b|oct\b|nov\b|dec\b|2\d{3}\b)[\w\s]+\s+(vs|versus)\s+(?!q[1-4]\b|quarter\b|fy\b|january\b|february\b|march\b|april\b|may\b|june\b|july\b|august\b|september\b|october\b|november\b|december\b|jan\b|feb\b|mar\b|apr\b|jun\b|jul\b|aug\b|sep\b|sept\b|oct\b|nov\b|dec\b|2\d{3}\b)[\w\s]+',

        r'\bdifference\s+between\s+(?!q[1-4]\b|quarter\b|fy\b|january\b|february\b|march\b|april\b|may\b|june\b|july\b|august\b|september\b|october\b|november\b|december\b|jan\b|feb\b|mar\b|apr\b|jun\b|jul\b|aug\b|sep\b|sept\b|oct\b|nov\b|dec\b|2\d{3}\b)[\w\s]+\s+and\s+(?!q[1-4]\b|quarter\b|fy\b|january\b|february\b|march\b|april\b|may\b|june\b|july\b|august\b|september\b|october\b|november\b|december\b|jan\b|feb\b|mar\b|apr\b|jun\b|jul\b|aug\b|sep\b|sept\b|oct\b|nov\b|dec\b|2\d{3}\b)[\w\s]+',
        
        # X or Y (which is better implied)
        r'\b(india|indian)\s+or\s+\w+\b.*\?',
        r'\b\w+\s+or\s+(india|indian)\b.*\?',
        
        r'\b(prove|show|demonstrate|justify)\s+(that|why)\s+(india|indian)\b',
        r'\b(statistics|stats|data)\s+(prove|proves|show|shows|justify|justifies)\b.*(india|indian)',
        r'\buse\s+(statistics|stats|data)\s+to\s+(prove|show|justify)\b',
    ]
    
    for pattern in opinion_patterns:
        if re.search(pattern, q_lower):
            # Return canonical fallback for consistency
            return True, CANONICAL_FALLBACK_EN, "opinion_request"
    
    prediction_patterns = [
        r'\b(predict|prediction|forecast|forecasting|projected|projection)\b.*\b(gdp|cpi|inflation|iip|growth|unemployment|plfs)\b',
        r'\b(gdp|cpi|inflation|iip|growth|unemployment|plfs)\b.*\b(predict|prediction|forecast|forecasting|projected|projection)\b',
        r'\bwhat will\b.*\b(gdp|cpi|inflation|iip|growth|rate)\b.*\b(be|become)\b',
        r'\b(next year|next quarter|next month)\b.*\b(gdp|cpi|inflation|iip|growth)\b.*\b(be|estimate|expect)\b',
        r'\bmospi.*(predict|forecast|projection)\b',
    ]
    
    for pattern in prediction_patterns:
        if re.search(pattern, q_lower):
            # Return canonical fallback for consistency
            return True, CANONICAL_FALLBACK_EN, "prediction_request"
    
    policy_patterns = [
        r'\b(is the|are the).*(government|policy|scheme|program).*(effective|working|successful|good|bad)\b',
        r'\b(effective|effectiveness|success|failure).*(government|policy|scheme|program|initiative)\b',
        r'\bshould\s+(government|india|mospi|ministry)\b',
        r'\b(recommend|suggestion|advice)\b.*(policy|government|economic)\b',
        r'\b(evaluate|assess|judge).*(government|policy|economic)\s*(performance|effectiveness)\b',
    ]
    
    for pattern in policy_patterns:
        if re.search(pattern, q_lower):
            # Return canonical fallback for consistency
            return True, CANONICAL_FALLBACK_EN, "policy_judgment"
    
    # 4. State-level quarterly GDP requests
    state_names = r'(state|pradesh|karnataka|maharashtra|tamil\s*nadu|uttar\s*pradesh|gujarat|rajasthan|madhya\s*pradesh|bihar|west\s*bengal|andhra|telangana|kerala|odisha|assam|punjab|haryana|jharkhand|chhattisgarh|uttarakhand|himachal|goa)'
    state_quarterly_patterns = [
        # State ... quarter ... GDP
        rf'\b{state_names}\b.*\b(quarterly|q1|q2|q3|q4|quarter)\b.*\b(gdp|gva|growth)\b',
        # Quarter ... GDP ... state
        rf'\b(quarterly|q1|q2|q3|q4|quarter)\b.*\b(gdp|gva|growth)\b.*\b{state_names}\b',
        # GDP ... state ... quarter (e.g., "GDP of Uttarakhand in Q2")
        rf'\b(gdp|gva|growth)\b.*\b{state_names}\b.*\b(quarterly|q1|q2|q3|q4|quarter)\b',
        # GDP ... quarter ... state
        rf'\b(gdp|gva|growth)\b.*\b(quarterly|q1|q2|q3|q4|quarter)\b.*\b{state_names}\b',
    ]
    
    for pattern in state_quarterly_patterns:
        if re.search(pattern, q_lower):
            # Return canonical fallback for consistency
            return True, CANONICAL_FALLBACK_EN, "state_quarterly_gdp"
    

    from datetime import datetime
    current_date = datetime.now()
    current_year = current_date.year
    

    future_fy_pattern = r'\bfy\s*(\d{4})[:-]?(\d{2,4})\b'
    fy_match = re.search(future_fy_pattern, q_lower)
    if fy_match:
        fy_start = int(fy_match.group(1))
        # If FY start year is more than 1 year ahead, it's definitely future
        if fy_start > current_year + 1:
            # Return canonical fallback for consistency
            return True, CANONICAL_FALLBACK_EN, "future_fy_data"
    
    return False, None, "ok"


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"

_TRANSLATION_ERROR_MARKERS = (
    "error 500",
    "server error",
    "that's an error",
    "that’s an error",
    "please try again later",
    "translation unavailable",
)


def _clean_translation_output(text: str) -> str:
    """Normalize GPT-OSS translation output without changing its meaning."""
    cleaned = extract_final_answer((text or "").strip()).strip()
    cleaned = re.sub(r"^```(?:text|plain)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(
        r"^(?:english|hindi)\s+translation\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _translation_result_is_valid(source_text: str, translated_text: str, target_language: str) -> bool:
    """Reject empty output and provider/error text before it reaches the pipeline."""
    if not translated_text:
        return False

    translated_lower = translated_text.lower()
    if any(marker in translated_lower for marker in _TRANSLATION_ERROR_MARKERS):
        return False

    # A Hindi-to-English request that returns the original Devanagari text was
    # not translated. Do not send it into the English retrieval pipeline.
    if target_language == "English" and translated_text == source_text:
        if re.search(r"[\u0900-\u097f]", translated_text):
            return False

    return True


def _translate_with_gpt_oss(
    text: str,
    target_language: str,
    max_tokens: int,
) -> str:
    """Translate text through the existing local OpenAI-compatible vLLM client."""
    if not text:
        return text

    client = globals().get("llm_answer_client")
    if client is None:
        logger.error("[TRANSLATION] GPT-OSS client is not initialized; returning original text")
        return text

    system_prompt = (
        "You are a translation engine. "
        f"Translate the input text to {target_language}. "
        "Return only the translation. Do not explain, summarize, answer the question, "
        "or add commentary. Preserve names, numbers, dates, URLs, acronyms, and "
        "statistical terms exactly."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"TEXT TO TRANSLATE:\n{text}"},
    ]

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
        )
        raw_translation = response.choices[0].message.content or ""
        translated = _clean_translation_output(raw_translation)

        if not _translation_result_is_valid(text, translated, target_language):
            logger.warning(
                "[TRANSLATION] GPT-OSS returned invalid %s translation; returning original text",
                target_language,
            )
            return text

        logger.info(
            "[TRANSLATION] GPT-OSS %s translation completed in %.2fs",
            target_language,
            time.time() - start_time,
        )
        return translated
    except Exception as exc:
        logger.warning(
            "[TRANSLATION] GPT-OSS %s translation failed: %s; returning original text",
            target_language,
            exc,
        )
        return text


def translate_to_english(text: str) -> str:
    if detect_language(text) == "en":
        return text
    return _translate_with_gpt_oss(text, target_language="English", max_tokens=128)


def translate_to_hindi(text: str) -> str:
    if detect_language(text) == "hi":
        return text
    return _translate_with_gpt_oss(text, target_language="Hindi", max_tokens=2048)


def clean_unicode_corruption(text: str) -> str:
    if not text:
        return text
    
    # Common Unicode corruption patterns
    corruption_fixes = [
        # Bullet points
        ('âž¢', '•'),
        ('â€¢', '•'),
        
        # Quotes
        ('â€˜', '''),  # left single quote
        ('â€™', '''),  # right single quote
        ('â€œ', '"'),  # left double quote
        ('â€', '"'),   # right double quote
        
        # Dashes
        ('â€"', '–'),  # en dash
        ('â€"', '—'),  # em dash
        
        # Apostrophes
        ('â€™', "'"),  # apostrophe
        
        # Ellipsis
        ('â€¦', '…'),
        
        # Other common corruptions
        ('Â', ''),     # Remove stray Â characters
        ('Ã¢', ''),    # Remove stray Ã¢ characters
        ('Ã¯Â¿Â½', ''), # Remove replacement character sequences
        
        # Additional PDF extraction corruptions
        ('ï¿½', ''),   # Replacement character
        ('â€‹', ''),   # Zero-width space
        ('â€Œ', ''),   # Zero-width non-joiner
        ('â€', ''),    # Zero-width joiner
    ]
    
    cleaned_text = text
    for corrupted, clean in corruption_fixes:
        cleaned_text = cleaned_text.replace(corrupted, clean)
    
    # Normalize excessive whitespace but preserve document structure
    import re
    # Replace multiple consecutive spaces with single space, but keep newlines
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
    # Normalize line endings but preserve paragraph breaks
    cleaned_text = re.sub(r'\r\n', '\n', cleaned_text)
    cleaned_text = re.sub(r'\r', '\n', cleaned_text)
    
    return cleaned_text.strip()

def clean_unicode_for_preview(text: str) -> str:

    if not text:
        return text
    
    # First apply the standard Unicode corruption fixes
    cleaned_text = text
    
    # Common Unicode corruption patterns (same as main function)
    corruption_fixes = [
        # Bullet points
        ('âž¢', '•'),
        ('â€¢', '•'),
        
        # Quotes
        ('â€˜', '''),  # left single quote
        ('â€™', '''),  # right single quote
        ('â€œ', '"'),  # left double quote
        ('â€', '"'),   # right double quote
        
        # Dashes
        ('â€"', '–'),  # en dash
        ('â€"', '—'),  # em dash
        
        # Apostrophes
        ('â€™', "'"),  # apostrophe
        
        # Ellipsis
        ('â€¦', '…'),
        
        # Other common corruptions
        ('Â', ''),     # Remove stray Â characters
        ('Ã¢', ''),    # Remove stray Ã¢ characters
        ('Ã¯Â¿Â½', ''), # Remove replacement character sequences
        
        # Additional PDF extraction corruptions
        ('ï¿½', ''),   # Replacement character
        ('â€‹', ''),   # Zero-width space
        ('â€Œ', ''),   # Zero-width non-joiner
        ('â€', ''),    # Zero-width joiner
    ]
    
    for corrupted, clean in corruption_fixes:
        cleaned_text = cleaned_text.replace(corrupted, clean)
    
    # For preview contexts: normalize all whitespace to spaces for single-line display
    cleaned_text = cleaned_text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    
    # Normalize multiple spaces to single space
    import re
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()

def clean_labels(text: str) -> str:
    patterns = [
        r"\bQ:\s*",
        r"\bA:\s*",
        r"\bAnswer:\s*",
        r"\bMoSPI AI Answer:\s*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()

def is_fallback_response(text: str) -> bool:

    if not text:
        return False
    
    text_stripped = text.strip()
    

    if text_stripped == CANONICAL_FALLBACK_EN or text_stripped == CANONICAL_FALLBACK_HI:
        return True

    if CANONICAL_FALLBACK_EN in text_stripped or CANONICAL_FALLBACK_HI in text_stripped:
        return True
    
    canonical_phrases = [
        r"outside\s+(my\s+)?scope",  # "outside my scope"
        r"unable\s+to\s+(assist|help)\s+you",  # "unable to assist/help you"
        r"(cannot|can't)\s+(assist|help)\s+you",  # "cannot assist/help you"
    ]
    
    # If text is short (<200 chars) and contains canonical phrases, it's likely a paraphrased fallback
    if len(text_stripped) < 200:
        canonical_match_count = sum(1 for phrase in canonical_phrases if re.search(phrase, text_stripped, re.I))
        if canonical_match_count >= 2:
            return True
    
    # Definitive fallback patterns (reduced set - only clear "no info" indicators)
    fallback_patterns = [
        re.compile(r"\bno (relevant|available)?\s*information\b", re.I),
        re.compile(r"\bi (don['']t|do not) have (any )?(info|information|details)\b", re.I),
        re.compile(r"\b(unable|cannot|can't) (to )?(find|provide|locate|answer)\b", re.I),
        re.compile(r"\b(outside|beyond) (my )?(scope|knowledge|ability)\b", re.I),
        re.compile(r"\bsorry[, ]? i (don['']t|do not) have\b", re.I),
        re.compile(r"\bi (couldn['']t|cannot|can't) find (any )?(info|information|details)\b", re.I),
        re.compile(r"\bi couldn['']t find any information\b", re.I),
        # NEW: "documents do not contain" pattern (HYBRID approach - always treat as fallback)
        re.compile(r"\b(the\s+)?(documents?|sources?|data|information|text)\s+(provided\s+)?(do\s+not|does\s+not|doesn't|don't)\s+contain\b", re.I),
    ]
    
    documents_pattern = re.compile(r"\b(the\s+)?(documents?|sources?|data|information|text)\s+(provided\s+)?(do\s+not|does\s+not|doesn't|don't)\s+contain\b", re.I)
    if documents_pattern.search(text_stripped):
        return True
    
    # Check if any other fallback pattern matches
    pattern_matched = any(pattern.search(text_stripped) for pattern in fallback_patterns)
    
    if not pattern_matched:
        return False
    
    remaining_text = text_stripped
    for pattern in fallback_patterns:
        remaining_text = pattern.sub('', remaining_text)
    

    remaining_text = re.sub(r'[\s\.,;:!?\-]+', ' ', remaining_text).strip()
    

    if len(remaining_text) > 100:
        return False
    
    return True

def is_blocked_query(query: str) -> bool:
    query = query.lower()
    BLOCKED_PATTERNS = [
        r"\bwhy\s+is\s+india\s+poor\b",
        r"\bwhy\s+india\s+is\s+poor\b",
        r"why.*india.*poor",
        r"(statistics|stats|data).*(prove|proves|show|justify).*(india).*(poor)",
        r"\bwhy\s+is\s+there\s+high\s+unemployment\b",
        r"\bwhy\s+is\s+unemployment\s+high\b",
        r"why.*india.*unemployment",
        r"(statistics|stats|data).*(prove|proves|show|justify).*(india).*(unemployment)",
        r"\bwhy\s+india\s+has\s+low\s+literacy\s+rate\b",
        r"\bwhy\s+is\s+literacy\s+low\s+in\s+india\b",
        r"why.*literacy.*india",
        r"(statistics|stats|data).*(prove|proves|show|justify).*(india).*(literacy|low)",
        r"\bwhy\s+poverty\s+is\s+high\s+in\s+india\b",
        r"why.*poverty.*india",
        r"(statistics|stats|data).*(prove|proves|show|justify).*(poverty).*(india)",
        r"\bwhy\s+is\s+india\s+not\s+doing\s+well\b",
        r"\bwhy\s+does\s+india\s+lag\b",
        r"why.*india.*lag",
        r"why.*india.*not doing well",
        r"justify.*india",
        r"(statistics|stats|data).*(prove|proves|show|justify).*(india).*(not\s+doing\s+well|lag|failure)",
        r"*movie",
    ]
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query):
            return True
    return False

# Economy KPIs
economy_model = None  # lazy-loaded in initialize_components (if needed)
ECONOMY_KPIS = {
    "gdp": "**GDP Growth**: 7.8% (Q1, 2025-26)",
    "iip": "**Index of Industrial Production (IIP)**: 4.0% (August 2025)",
    "cpi": "**Inflation (CPI)**: 1.54% (September 2025)",
    "unemployment": "**Urban Unemployment Rate**: 5.2% (September 2025)"
}

KPI_KEYWORDS = {
    "gdp": ["gdp", "gross domestic product", "india gdp", "indian gdp", "gdp growth", "gdp rate", "india gdp growth", "gdp of india"],
    "iip": ["iip", "industrial production", "index of industrial production", "industrial output", "industrial growth", "iip rate", "iip growth rate"],
    "cpi": ["cpi", "inflation", "consumer price index", "price rise", "inflation in india", "current inflation", "inflation rate", "price increase"],
    "unemployment": ["unemployment", "jobless rate", "urban unemployment", "employment rate", "unemployment rate", "joblessness", "unemployment in india"]
}

GENERAL_ECONOMY_KEYWORDS = [
    "indian economy", "india economy", "economic growth", "economic health",
    "economic condition", "progress of indian economy", "financial condition",
    "latest economic growth", "current economic condition", "economic update",
    "state of economy", "economic situation", "india economic report",
    "recent economic growth", "india economic performance", "economical outlook", "financial outlook"
]

def get_kpi_by_keyword(query: str) -> str | None:
    query_lower = (query or "").lower()

    # Explicitness requirement: user must request current/latest data
    current_latest_keywords = [
        "current", "latest", "recent", "now", "today", "this year",
        "latest value", "current value", "latest data", "current data",
        "as of", "as on", "as at"
    ]

    has_current_latest = any(kw in query_lower for kw in current_latest_keywords)

    # Exclude queries that clearly ask for definitions, visualizations, history, or "what is"
    exclusion_keywords = [
        "graph", "chart", "plot", "visual", "visualization", "trend",
        "definition", "define", "meaning", "what is", "what's", "years ago", "ago",
        "historical", "history", "explain", "describe", "how", "why"
    ]

    if not has_current_latest:
        return None

    if any(ek in query_lower for ek in exclusion_keywords):
        return None

    matched = [kpi for kpi, kws in KPI_KEYWORDS.items() if any(kw in query_lower for kw in kws)]
    if matched:
        lines = [ECONOMY_KPIS[k] for k in matched]
        return "📊 **Latest Indian Economy KPIs**\n\n" + "\n".join(lines)

    if any(kw in query_lower for kw in GENERAL_ECONOMY_KEYWORDS):
        return "📊 **Latest Indian Economy KPIs**\n\n" + "\n".join(ECONOMY_KPIS.values())

    return None

# ========== Canonical Fallback Messages ==========
# Single source of truth for fallback responses
CANONICAL_FALLBACK_EN = "This seems to be outside my scope. Unfortunately, I am unable to assist you with your requested query. Thank you for your understanding."
CANONICAL_FALLBACK_HI = "यह मेरे दायरे से बाहर लगता है। दुर्भाग्य से, मैं आपके अनुरोधित प्रश्न में सहायता नहीं कर सकता। धन्यवाद।"

def fallback_response(is_hindi: bool = False) -> str:
    """Return the canonical fallback message for the given language."""
    return CANONICAL_FALLBACK_HI if is_hindi else CANONICAL_FALLBACK_EN


ESANKHYIKI_URL = "https://esankhyiki.mospi.gov.in/"
MOSPI_URL = "https://mospi.gov.in/"

_CLARIFY_STEP1_MARKER_EN = "could you please clarify or rephrase your question"
CANONICAL_CLARIFY_STEP1_EN = (
    "I'm not fully certain what you're looking for, and I couldn't find enough relevant "
    "information to answer confidently. Could you please clarify or rephrase your question "
    "with a bit more detail?"
)
_CLARIFY_STEP1_MARKER_HI = "कृपया अपना प्रश्न स्पष्ट करें या दोबारा पूछें"
CANONICAL_CLARIFY_STEP1_HI = (
    "मुझे पूरी तरह स्पष्ट नहीं है कि आप क्या जानना चाहते हैं, और मुझे विश्वासपूर्वक उत्तर देने के लिए "
    "पर्याप्त प्रासंगिक जानकारी नहीं मिली। कृपया अपना प्रश्न स्पष्ट करें या दोबारा पूछें, थोड़े और विवरण के साथ।"
)

CANONICAL_CLARIFY_STEP2_EN = (
    "I'm still not able to understand your question clearly enough to answer confidently. "
    f"For detailed statistics and data, please visit {ESANKHYIKI_URL} — for anything else about "
    f"MoSPI in general, please visit {MOSPI_URL}"
)
CANONICAL_CLARIFY_STEP2_HI = (
    "मुझे अब भी आपका प्रश्न पर्याप्त स्पष्ट रूप से समझ नहीं आ रहा है, जिससे मैं विश्वासपूर्वक उत्तर दे सकूं। "
    f"विस्तृत आँकड़ों और डेटा के लिए कृपया {ESANKHYIKI_URL} पर जाएँ — MoSPI से संबंधित अन्य सामान्य जानकारी के लिए "
    f"कृपया {MOSPI_URL} पर जाएँ।"
)


def clarification_response(is_hindi: bool = False, escalate: bool = False) -> str:
    """Return the canonical clarification message for the given language/step."""
    if escalate:
        return CANONICAL_CLARIFY_STEP2_HI if is_hindi else CANONICAL_CLARIFY_STEP2_EN
    return CANONICAL_CLARIFY_STEP1_HI if is_hindi else CANONICAL_CLARIFY_STEP1_EN


def is_first_clarify_response(text: str) -> bool:
    """Detect the step-1 (no redirect links) clarification response."""
    if not text:
        return False
    t = text.lower()
    return (_CLARIFY_STEP1_MARKER_EN.lower() in t) or (_CLARIFY_STEP1_MARKER_HI in text)


def is_escalated_clarify_response(text: str) -> bool:
    """Detect the step-2 (eSankhyiki + mospi.gov.in redirect) clarification response."""
    if not text:
        return False
    return "esankhyiki.mospi.gov.in" in text.lower()


def is_clarification_response(text: str) -> bool:
    """Detect whether an answer is EITHER clarification step."""
    return is_first_clarify_response(text) or is_escalated_clarify_response(text)



def _dedupe_key(doc: Document) -> Tuple[Any, Any, Any]:
    meta = getattr(doc, "metadata", {}) or {}
    md5 = meta.get("md5") or meta.get("file_url") or meta.get("text_uri") or meta.get("file_name")
    page = meta.get("page_number")
    content = (getattr(doc, "page_content", "") or "").strip()
    sig = hash(content[:4000])
    return (md5, page, sig)


def run_llm_stream(messages):
    """Stream LLM response using vLLM with OpenAI-compatible API"""
    # Convert LangChain message format to OpenAI format if needed
    openai_messages = []
    for msg in messages:
        if hasattr(msg, 'type'):  # LangChain message object
            if msg.type == 'system':
                openai_messages.append({"role": "system", "content": msg.content})
            elif msg.type == 'human':
                openai_messages.append({"role": "user", "content": msg.content})
            elif msg.type == 'ai':
                openai_messages.append({"role": "assistant", "content": msg.content})
        else:  # Already in dict format
            openai_messages.append(msg)
    
    stream = llm_answer_client.chat.completions.create(
        model=VLLM_MODEL,
        messages=openai_messages,
        temperature=0.1,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

async def stream_llm_response(messages) -> AsyncGenerator[str, None]:
    """
    Stream LLM response tokens asynchronously using ChatOllama with message roles.
    
    Args:
        messages: List of SystemMessage and HumanMessage for ChatOllama
        
    Yields:
        String tokens from the LLM response
    """
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def producer():
        try:
            for chunk in run_llm_stream(messages):
                # Extract content from AIMessageChunk
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                asyncio.run_coroutine_threadsafe(queue.put(text), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    import threading
    threading.Thread(target=producer).start()

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token



def _flatten_qdrant_metadata(meta: dict) -> dict:
    """
    Normalize/flatten Qdrant payload metadata so we can reliably extract sources.
    
    This function handles metadata from both collections:
    - Main collection (mospi_collection_v2): Standard document metadata
    - Visualization collection: Legacy metadata with embed_code, embed, etc.
    
    Common shapes:
      1) {"page_content": "...", "metadata": {...}}
      2) {"metadata": {...}}  (LangChain may already map metadata_payload_key)
      3) {"payload": {...}}   (some wrappers)
      4) {"metadata": "{...json...}"} (metadata accidentally stored as a JSON string)
    """
    if not isinstance(meta, dict):
        return {}

    # Unwrap common wrappers
    if isinstance(meta.get("payload"), dict):
        meta = meta["payload"]
    if isinstance(meta.get("_payload"), dict):
        meta = meta["_payload"]

    nested = meta.get("metadata")

    # Handle metadata stored as a JSON string
    if isinstance(nested, str):
        try:
            nested = json.loads(nested)
        except Exception:
            nested = None

    if isinstance(nested, dict):
        merged = dict(meta)
        merged.pop("metadata", None)
        merged.update(nested)
        return merged

    return dict(meta)


def filter_documents_by_metric(docs: List[Document], detected_metric: str) -> List[Document]:

    if not detected_metric or not docs:
        return docs
    
    metric_lower = detected_metric.lower()
    
    # Define metric patterns (same as in format_sources for consistency)
    metric_patterns = [metric_lower]
    if metric_lower == 'plfs':
        metric_patterns.extend(['plfs', 'labour force', 'labor force', 'employment', 'periodic labour'])
    elif metric_lower == 'cpi':
        metric_patterns.extend(['cpi', 'consumer price', 'inflation'])
    elif metric_lower == 'iip':
        metric_patterns.extend(['iip', 'industrial production', 'manufacturing'])
    elif metric_lower == 'gdp':
        metric_patterns.extend(['gdp', 'gross domestic', 'national accounts'])
    
    filtered_docs = []
    filtered_out_count = 0
    
    for doc in docs:
        meta = _flatten_qdrant_metadata(getattr(doc, "metadata", {}) or {})
        file_name = str(meta.get("file_name") or meta.get("doc_name") or "").lower()
        
        # Check if any pattern matches
        if any(pattern in file_name for pattern in metric_patterns):
            filtered_docs.append(doc)
        else:
            filtered_out_count += 1
            logger.debug(f"[METRIC_FILTER] Filtered out non-{detected_metric} doc: {file_name[:80]}")
    
    logger.info(f"[METRIC_FILTER] Filtered {len(docs)} → {len(filtered_docs)} docs for metric={detected_metric} (removed {filtered_out_count})")
    
    return filtered_docs



def _source_label_from_doc(doc: Document) -> str:
    """Human-readable source label for a Qdrant/LangChain Document.
    
    Returns a fallback label "Document" if no identifying information is found,
    rather than returning an empty string which would cause the source to be skipped.
    """
    meta = _flatten_qdrant_metadata(getattr(doc, "metadata", {}) or {})

    # Prefer title if available, then fall back to file names
    name = (
        meta.get("title")           # PRIORITIZE title first
        or meta.get("page_title")
        or meta.get("doc_name")
        or meta.get("pdf_name")
        or meta.get("file_name")
        or meta.get("filename")
        or meta.get("document_name")
        or meta.get("chart_title")
        or meta.get("embed_Code")
    )

    # Fall back to URIs/paths (often .txt chunks)
    if not name:
        uri = (
            meta.get("url")
            or meta.get("file_url")
            or meta.get("text_uri")
            or meta.get("s3_uri")
            or meta.get("s3_key")
        )
        if uri:
            name = str(uri).rstrip("/").split("/")[-1]

    # Fallback: use point id or generic "Document" label
    if not name:
        pid = meta.get("point_id") or meta.get("id")
        if pid:
            return f"Document (ID: {pid})"
        # Ultimate fallback - never return empty string
        return "Document"

    # Normalize JSON FAQs naming
    if str(name).lower().endswith(".json"):
        base = "MOSPI FAQs"
    else:
        base = os.path.splitext(str(name))[0]

    page = meta.get("page_number") or meta.get("page") or meta.get("page_index")
    if isinstance(page, int):
        return f"{base} (p. {page})"
    if isinstance(page, str) and page.isdigit():
        return f"{base} (p. {int(page)})"

    return base


def _normalize_pdf_filename(pdf_name: str) -> str:

    if not pdf_name:
        return pdf_name
    name = str(pdf_name).strip()

    # Only normalize patterns like ...2023-24...pdf (keep the 4-digit year part)
    m = re.match(r"^(.*?\d{4})(?:[-_]\d{2}.*)\.pdf$", name, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}.pdf"
    return name


def _parse_s3_like_uri(uri: str) -> tuple[str, str] | None:
    """Parse s3:/bucket/key, s3://bucket/key, or s3:bucket/key into (bucket, key)."""
    if not uri:
        return None
    u = str(uri).strip()

    if u.startswith("s3://"):
        rest = u[len("s3://"):]
    elif u.startswith("s3:/"):
        rest = u[len("s3:/"):]
    elif u.startswith("s3:"):
        rest = u[len("s3:"):].lstrip("/")
    else:
        return None

    if "/" not in rest:
        return None
    bucket, key = rest.split("/", 1)
    bucket = bucket.strip()
    key = key.strip().lstrip("/")
    if not bucket or not key:
        return None
    return bucket, key


def _public_s3_https_url(bucket: str, key: str) -> str:
    """Generate public S3 HTTPS URL with proper URL encoding for special characters.
    
    Encodes spaces as %20 and other special characters while preserving path separators.
    """
    region = (
        os.getenv("MOSPI_S3_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "ap-south-1"
    )
    # URL-encode the key, preserving forward slashes as path separators
    encoded_key = quote(key.lstrip('/'), safe='/')
    return f"https://{bucket}.s3.{region}.amazonaws.com/{encoded_key}"


def _source_url_from_doc(doc: Document) -> str | None:

    meta = _flatten_qdrant_metadata(getattr(doc, "metadata", {}) or {})

    # Optional page anchor (ONLY if the target is a PDF)
    page = meta.get("page_number") or meta.get("page") or meta.get("page_index")
    page_int = None
    if isinstance(page, int):
        page_int = page
    elif isinstance(page, str) and page.isdigit():
        page_int = int(page)

    def _is_pdf(url: str) -> bool:
        u = (url or "").lower().split("#", 1)[0].split("?", 1)[0]
        return u.endswith(".pdf")

    def _maybe_add_pdf_page(url: str) -> str:
        if page_int and _is_pdf(url) and "#page=" not in url.lower():
            return f"{url}#page={page_int}"
        return url

    # Candidate URL-ish fields (in priority order)
    candidates: List[str] = []
    for k in ("url", "file_url", "text_uri", "s3_uri"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())

    # If provided as bucket + key, synthesize an s3:// URI candidate
    bucket = meta.get("s3_bucket")
    key = meta.get("s3_key")
    if isinstance(bucket, str) and bucket.strip() and isinstance(key, str) and key.strip():
        candidates.append(f"s3://{bucket.strip()}/{key.strip().lstrip('/')}")

    # First usable candidate wins, but always normalize S3-like URLs to public HTTPS
    for raw in candidates:
        low = raw.lower()

        # If it's already a public S3 HTTPS URL, normalize it
        if low.startswith("https://") and ".s3." in low and ".amazonaws.com/" in low:
            try:
                # Example: https://bucket.s3.region.amazonaws.com/key
                parts = low.split(".s3.", 1)
                bucket = parts[0].replace("https://", "").replace("http://", "")
                rest = parts[1]
                region_and_key = rest.split(".amazonaws.com/", 1)
                if len(region_and_key) == 2:
                    region = region_and_key[0]
                    key = region_and_key[1]
                    url = _public_s3_https_url(bucket, key)
                    return _maybe_add_pdf_page(url)
            except Exception:
                pass  # fallback to returning as-is if parsing fails
            return _maybe_add_pdf_page(raw)

        # Handle legacy S3 URLs like https://s3.amazonaws.com/bucket/key
        if low.startswith("https://s3.amazonaws.com/"):
            try:
                # Remove query/hash for key extraction
                url_main = raw.split("?", 1)[0].split("#", 1)[0]
                path = url_main[len("https://s3.amazonaws.com/"):]
                # The first segment is the bucket, the rest is the key
                if "/" in path:
                    bucket, key = path.split("/", 1)
                    url = _public_s3_https_url(bucket, key)
                    # Re-append any hash (e.g., #page=18)
                    if "#" in raw:
                        url += "#" + raw.split("#", 1)[1]
                    return _maybe_add_pdf_page(url)
            except Exception:
                pass
            return _maybe_add_pdf_page(raw)

        # If it's a regular HTTP/HTTPS URL (not S3), return as-is
        if low.startswith(("http://", "https://")):
            return _maybe_add_pdf_page(raw)

        # If it's an S3-style URI, convert to public HTTPS
        parsed = _parse_s3_like_uri(raw)
        if parsed:
            b, k = parsed
            url = _public_s3_https_url(b, k)
            return _maybe_add_pdf_page(url)

    return None




DISABLE_CONTEXT_TRUNCATION = os.getenv("DISABLE_CONTEXT_TRUNCATION", "1").strip().lower() in ("1", "true", "yes", "y", "on")
CONTEXT_MAX_CHARS_PER_DOC = int(os.getenv("CONTEXT_MAX_CHARS_PER_DOC", "50000"))  # ~12,500 tokens per doc
CONTEXT_MAX_TOTAL_CHARS = int(os.getenv("CONTEXT_MAX_TOTAL_CHARS", "500000"))  # ~125,000 tokens total (leave buffer)

def build_context(
    docs: List[Document],
    truncate: Optional[bool] = None,
    max_chars_per_doc: Optional[int] = None,
    max_total_chars: Optional[int] = None,
    include_headers: bool = True,
) -> str:

    if truncate is None:
        truncate = not DISABLE_CONTEXT_TRUNCATION

    if max_chars_per_doc is None:
        max_chars_per_doc = CONTEXT_MAX_CHARS_PER_DOC
    if max_total_chars is None:
        max_total_chars = CONTEXT_MAX_TOTAL_CHARS

    # If truncation disabled, effectively remove caps
    if not truncate:
        max_chars_per_doc = 10**9
        max_total_chars = 10**9

    def _truncate(s: str, limit: int) -> str:
        s = (s or "").strip()
        if len(s) <= limit:
            return s
        head = s[: int(limit * 0.85)].rstrip()
        tail = s[- int(limit * 0.15):].lstrip()
        return f"{head}\n...\n{tail}"

    def _clean_table_formatting(content: str) -> str:

        if not content:
            return content
            
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Only process lines that contain pipes
            if '|' not in line:
                cleaned_lines.append(line)
                continue
            
            # Split by pipes and analyze the structure
            parts = [part.strip() for part in line.split('|')]
            
            # Count meaningful (non-empty) parts
            meaningful_parts = [part for part in parts if part and part.strip()]
            empty_parts = len(parts) - len(meaningful_parts)
            
            is_hierarchical = len(parts) > 0 and not parts[0].strip() and any(parts[1:])
            has_many_columns = len(meaningful_parts) > 4
            has_excessive_empty = empty_parts > 5
            

            if has_excessive_empty and not has_many_columns and not is_hierarchical:
                # This looks like a malformed line with excessive empty pipes
                if meaningful_parts:
                    # Rejoin meaningful parts with proper spacing
                    cleaned_line = ' | '.join(meaningful_parts)
                    cleaned_lines.append(cleaned_line)
                # If no meaningful content, skip the line entirely
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    blocks: List[str] = []
    total = 0

    # Context building debug logging
    if ENABLE_RETRIEVAL_DEBUG:
        logger.info(f"[CONTEXT_DEBUG] === BUILDING CONTEXT FOR LLM ===")
        logger.info(f"[CONTEXT_DEBUG] Input documents: {len(docs)}")
        logger.info(f"[CONTEXT_DEBUG] Truncation: {truncate}, max_chars_per_doc: {max_chars_per_doc}, max_total_chars: {max_total_chars}")

    for i, d in enumerate(docs, start=1):
        meta = _flatten_qdrant_metadata(getattr(d, "metadata", {}) or {})
        content = (getattr(d, "page_content", "") or meta.get("page_content") or "").strip()
        if not content:
            continue

        # Clean up malformed table formatting before processing
        original_content = content
        content = _clean_table_formatting(content)
        
        # Log table cleanup if significant changes were made
        if len(original_content) - len(content) > 100:  # Significant cleanup occurred
            original_lines = len(original_content.split('\n'))
            cleaned_lines = len(content.split('\n'))
            logger.info(f"[CONTEXT_DEBUG] Table cleanup applied to {meta.get('file_name', 'unknown')}: reduced from {len(original_content)} to {len(content)} chars, {original_lines} to {cleaned_lines} lines")

        # Log document details before truncation
        if ENABLE_RETRIEVAL_DEBUG and i <= RETRIEVAL_DEBUG_MAX_DOCS:
            file_name = meta.get("file_name") or meta.get("doc_name") or "unknown_file"
            page = meta.get("page_number")
            pub = meta.get("publish_date")
            logger.info(f"[CONTEXT_DEBUG] Doc {i}: {file_name} (p.{page})")
            logger.info(f"[CONTEXT_DEBUG]        publish_date: {pub}")
            logger.info(f"[CONTEXT_DEBUG]        content_len: {len(content)}")
            logger.info(f"[CONTEXT_DEBUG]        content_preview: {clean_unicode_for_preview(content[:200])}...")
            
            # Check for date mismatches in content
            content_lower = content.lower()
            if "november 2025" in file_name.lower() and ("2022" in content or "2023" in content or "2024" in content):
                years_found = []
                for year in ["2022", "2023", "2024"]:
                    if year in content:
                        years_found.append(year)
                logger.warning(f"[CONTEXT_DEBUG]   DATA QUALITY ISSUE: {file_name} contains old data from years: {years_found}")
                logger.warning(f"[CONTEXT_DEBUG]   This explains why LLM returns old dates despite correct filename/metadata")

        content = _truncate(content, max_chars_per_doc)

        header = ""
        if include_headers:
            file_name = meta.get("file_name") or meta.get("doc_name") or "unknown_file"
            page = meta.get("page_number")
            pub = meta.get("publish_date")
            bits = [f"Source {i}: {file_name}"]
            if page is not None:
                bits.append(f"p. {page}")
            if pub:
                bits.append(f"publish_date: {pub}")
            header = " | ".join(bits) + "\n"

        block = header + content

        if total + len(block) > max_total_chars:
            remaining = max_total_chars - total
            if ENABLE_RETRIEVAL_DEBUG:
                file_name = meta.get("file_name") or meta.get("doc_name") or "unknown_file"
                page = meta.get("page_number")
                logger.warning(f"[CONTEXT_DEBUG]  TRUNCATION: Doc {i} ({file_name} p.{page}) - remaining budget: {remaining} chars")
            if remaining <= 200:
                if ENABLE_RETRIEVAL_DEBUG:
                    logger.warning(f"[CONTEXT_DEBUG]  STOPPING: Budget exhausted, skipping remaining {len(docs) - i + 1} documents!")
                break
            block = _truncate(block, remaining)
            if ENABLE_RETRIEVAL_DEBUG:
                logger.warning(f"[CONTEXT_DEBUG]  Doc {i} truncated to {len(block)} chars (from {len(header + content)})")

        blocks.append(block)
        total += len(block)
        
        if ENABLE_RETRIEVAL_DEBUG:
            logger.info(f"[CONTEXT_DEBUG] Doc {i} added: {len(block)} chars, total now: {total}/{max_total_chars}")

        if total >= max_total_chars:
            if ENABLE_RETRIEVAL_DEBUG:
                logger.warning(f"[CONTEXT_DEBUG]  BUDGET FULL: Stopping after doc {i}, skipping {len(docs) - i} remaining docs")
            break

    final_context = "\n\n---\n\n".join(blocks).strip()
    
    if ENABLE_RETRIEVAL_DEBUG:
        logger.info(f"[CONTEXT_DEBUG] === FINAL CONTEXT FOR LLM ===")
        logger.info(f"[CONTEXT_DEBUG] Total blocks: {len(blocks)}")
        logger.info(f"[CONTEXT_DEBUG] Total characters: {len(final_context)}")
        logger.info(f"[CONTEXT_DEBUG] Context preview: {final_context[:500]}...")
        
        # Log any data quality issues found
        if "november 2025" in final_context.lower() and ("2022" in final_context or "2023" in final_context or "2024" in final_context):
            logger.warning(f"[CONTEXT_DEBUG]   CONFIRMED: Final context contains mixed date data - this is the root cause of incorrect LLM responses")

    return final_context


def format_sources(docs: List[Document], max_sources: int = 10, detected_metric: Optional[str] = None) -> str:
    """
    Returns a markdown sources block with consolidated page numbers.
    Groups documents by (title/file_name, url) and consolidates page numbers.
    
    Example output:
    - [Document Title (p. 1, 2, 5)](url)
    - [Another Document (p. 3)](url)
    
    Args:
        docs: List of documents to format as sources (already filtered by metric if applicable)
        max_sources: Maximum number of unique documents to display
        detected_metric: Kept for backward compatibility but not used (filtering now happens earlier)
    
    Note: Metric filtering now happens BEFORE this function is called (in the main handler),
          so documents passed here are already filtered to match the detected metric.
    """
    from collections import defaultdict
    
    # Group documents by (base_name, url)
    # Key: (base_name, url), Value: list of page numbers
    doc_groups = defaultdict(list)
    skipped_count = 0

    for d in docs:
        meta = _flatten_qdrant_metadata(getattr(d, "metadata", {}) or {})
        
        # Get the base name (title or file_name without extension)
        title = meta.get("title") or meta.get("page_title")
        file_name = str(meta.get("file_name") or meta.get("doc_name") or "")
        
        # Use title if available, otherwise use file_name
        if title:
            base_name = str(title).strip()
        elif file_name:
            base_name = os.path.splitext(file_name)[0]
        else:
            base_name = ""
        
        # Get URL
        url = _source_url_from_doc(d) or ""
        
        # Skip documents without URL or base_name
        if not url or not url.strip() or not base_name:
            skipped_count += 1
            logger.debug(f"[SOURCES] Skipping doc without URL or name. base_name='{base_name}', url='{url[:50] if url else ''}'")
            continue
        
        url_for_grouping = url.split("#page=")[0] if "#page=" in url.lower() else url
        
        # Get page number
        page = meta.get("page_number") or meta.get("page") or meta.get("page_index")
        page_num = None
        if isinstance(page, int):
            page_num = page
        elif isinstance(page, str) and page.isdigit():
            page_num = int(page)
        

        key = (base_name, url_for_grouping)
        
        # DEBUG: Log grouping information
        logger.info(f"[SOURCES_GROUPING] base_name='{base_name}', page={page_num}, url_original='{url[:80]}...', url_for_grouping='{url_for_grouping[:80]}...'")
        if page_num is not None:
            doc_groups[key].append(page_num)
        else:
            # Document without page number - add None to indicate it exists
            if None not in doc_groups[key]:
                doc_groups[key].append(None)

    if skipped_count > 0:
        logger.info(f"[SOURCES] Skipped {skipped_count} docs (empty label or no URL)")

    if not doc_groups:
        return ""  # Return empty string instead of "No sources found"

    # Format consolidated sources
    lines = ["📄 **Sources:**"]
    count = 0
    
    for (base_name, url_for_grouping), pages in doc_groups.items():
        if count >= max_sources:
            break
        
        # Sort and dedupe page numbers
        page_nums = sorted(set(p for p in pages if p is not None))
        
        # Format label with consolidated pages
        if page_nums:
            pages_str = ", ".join(str(p) for p in page_nums)
            label = f"{base_name} (p. {pages_str})"
        else:
            # No page numbers available
            label = base_name
        

        lines.append(f"- [{label}]({url_for_grouping})")
        count += 1

    # Return empty string if no valid sources after filtering
    if len(lines) <= 1:
        return ""

    return "\n".join(lines)





def retrieve_visualization_docs(query: str, k: int = 3) -> List[Document]:
    if not query:
        logger.warning("---------------------[VIS] Empty query provided")
        return []
    
    if not qdrant_client_vis:
        logger.error("--------------[VIS] qdrant_client_vis not initialized - visualization collection unavailable")
        return []
    
    try:
        logger.info(f"---------------[VIS] Searching visualization collection for: '{query}'")
        
        # Get query embedding using the same model as the collection
        query_vector = info_embedding_model.embed_query(query)
        
        # Direct Qdrant search on visualization collection
        search_result = qdrant_client_vis.search(
            collection_name=VIS_COLLECTION_NAME,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
            score_threshold=0.1  # Minimum similarity threshold
        )
        
        logger.info(f"---------------[VIS] Found {len(search_result)} visualization results")
        
        docs: List[Document] = []
        for i, point in enumerate(search_result):
            payload = point.payload or {}
            
            # Extract visualization metadata
            chart_title = payload.get("chart_title", "")
            chart_url = payload.get("chart_url", "")
            embed_code = payload.get("embed_code", "")
            doc_id = payload.get("doc_id", "")
            score = getattr(point, 'score', 0.0)
            
            logger.info(f"---------------[VIS] Result {i+1}: title='{chart_title[:50]}...', score={score:.4f}")
            

            docs.append(Document(
                page_content=chart_title,  # Use chart_title as searchable content
                metadata={
                    "chart_title": chart_title,
                    "chart_url": chart_url,
                    "embed_code": embed_code,
                    "doc_id": doc_id,
                    "score": score,
                    "collection_type": "visualization"
                }
            ))
        
        logger.info(f"---------------[VIS] Retrieved {len(docs)} visualization documents for query: '{query[:50]}...'")
        return docs
        
    except Exception as e:
        logger.error(f"------------------------[VIS] Retrieval failed: {e}", exc_info=True)
        return []


def extract_visualization_embeds(docs: List[Document]) -> List[str]:

    embeds: List[str] = []
    for i, doc in enumerate(docs or []):
        meta = doc.metadata or {}
        
        # Extract embed_code from metadata
        embed_code = meta.get("embed_code", "").strip()
        chart_title = meta.get("chart_title", f"visualization_{i}")
        chart_url = meta.get("chart_url", "")
        
        if embed_code:
            embeds.append(embed_code)
            logger.info(f"-------------[VIS] Extracted embed from chart: '{chart_title[:50]}...'")
        else:
            logger.debug(f"---------------[VIS] No embed_code in doc {i}. chart_title='{chart_title}', available_keys={list(meta.keys())}")

    # Remove duplicates while preserving order
    seen = set()
    unique_embeds = []
    for embed in embeds:
        if embed not in seen:
            seen.add(embed)
            unique_embeds.append(embed)

    logger.info(f"---------------[VIS] Extracted {len(unique_embeds)} unique embeds from {len(docs)} docs")
    return unique_embeds

_RELEASE_LOOKUP_PRODUCTS = {
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
_RELEASE_DEFAULT_FREQUENCIES = {
    "IIP": "monthly",
    "CPI": "monthly",
    "GDP": "quarterly",
    "PLFS": "monthly",
    "ASI": "annual",
    "NSS": "periodic",
    "EC": "periodic",
    "HCES": "periodic",
    "ISS": "periodic",
}

_RELEASE_ALLOWED_FREQUENCIES = {
    "IIP": "monthly",
    "CPI": "monthly",
    "GDP": "quarterly",
    "ASI": "annual",
    "NSS": "periodic",
    "EC": "periodic",
    "HCES": "periodic",
    "ISS": "periodic",
}
_RELEASE_FREQUENCY_PATTERNS = (
    ("bi-weekly", (r"bi[-\s]?weekly", r"fortnightly")),
    ("weekly", (r"(?<!bi[-\s])\bweekly\b",)),
    ("monthly", (r"\bmonthly\b",)),
    ("quarterly", (r"\bquarterly\b", r"\bquarter\b", r"\bq[1-4]\b")),
    ("half-yearly", (r"half[-\s]?yearly", r"semi[-\s]?annual")),
    ("annual", (r"\bannual\b", r"\byearly\b")),
)
_RELEASE_FREQUENCY_ORDER = tuple(item[0] for item in _RELEASE_FREQUENCY_PATTERNS) + ("periodic", "general")
_RELEASE_LOOKUP_MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}
_RELEASE_LOOKUP_MAX_POINTS = 3000


def _parse_release_publish_date(value: Any) -> Optional[datetime]:
    """Parse a Qdrant publish_date value without treating missing dates as current."""
    if not value:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "no_date"}:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _release_record_matches_data_period(record: Dict[str, Any], month: int, year: int) -> bool:
    """Match a requested data period against the release document's identity fields."""
    month_name = _RELEASE_LOOKUP_MONTHS[month].lower()
    period_patterns = (
        f"{month_name} {year}",
        f"{month_name}, {year}",
        f"{month_name}-{year}",
        f"{month_name}_{year}",
    )
    identity = " ".join(
        str(record.get(key) or "") for key in ("title", "file_name", "preview")
    ).lower()
    return any(pattern in identity for pattern in period_patterns)


def _record_mentions_release_product(identity_text: str, aliases: Tuple[str, ...]) -> bool:
    """Use boundaries so short aliases such as RBI/NSS do not match other words."""
    return any(re.search(rf"\b{re.escape(alias)}\b", identity_text) for alias in aliases)


def _get_release_frequency(product: str, record: Dict[str, Any]) -> str:
    """Classify a release document's cadence from its indexed identity fields."""
    allowed_frequency = _RELEASE_ALLOWED_FREQUENCIES.get(product)
    if allowed_frequency:
        return allowed_frequency

    identity = " ".join(
        str(record.get(key) or "") for key in ("title", "file_name", "preview")
    ).lower()
    for frequency, patterns in _RELEASE_FREQUENCY_PATTERNS:
        if any(re.search(pattern, identity, re.IGNORECASE) for pattern in patterns):
            # The configured PLFS product has monthly and quarterly releases;
            # do not surface an incidental annual/weekly phrase from its text.
            if product != "PLFS" or frequency in {"monthly", "quarterly"}:
                return frequency
    return _RELEASE_DEFAULT_FREQUENCIES.get(product, "general")


def _find_release_history_records(metric: str) -> List[Dict[str, Any]]:
    """Return unique, dated release documents for one product, newest first."""
    if qdrant_client is None:
        logger.error("[RELEASE_HISTORY] Qdrant client is not initialized")
        return []

    aliases = _RELEASE_LOOKUP_PRODUCTS.get(metric, ())
    if not aliases:
        return []

    conditions = []
    for alias in aliases:
        conditions.extend([
            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=alias)),
            qmodels.FieldCondition(key="title", match=qmodels.MatchText(text=alias)),
        ])

    records_by_document: Dict[str, Dict[str, Any]] = {}
    offset = None
    scanned = 0
    try:
        while scanned < _RELEASE_LOOKUP_MAX_POINTS:
            points, offset = qdrant_client.scroll(
                collection_name=INFO_COLLECTION_NAME,
                scroll_filter=qmodels.Filter(should=conditions),
                limit=min(256, _RELEASE_LOOKUP_MAX_POINTS - scanned),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            scanned += len(points)

            for point in points:
                payload = point.payload or {}
                file_name = str(payload.get("file_name") or "")
                title = str(payload.get("title") or "")
                identity_text = f"{file_name} {title}".lower()
                if not _record_mentions_release_product(identity_text, aliases):
                    continue

                published_at = _parse_release_publish_date(payload.get("publish_date"))
                if not published_at:
                    continue

                document_key = str(
                    payload.get("md5")
                    or payload.get("file_url")
                    or f"{file_name}|{payload.get('publish_date')}"
                )
                record = {
                    "metric": metric,
                    "publish_date": published_at,
                    "file_name": file_name,
                    "title": title,
                    "file_url": str(payload.get("file_url") or ""),
                    "preview": str(payload.get("page_content") or "")[:12000],
                }
                record["frequency"] = _get_release_frequency(metric, record)
                existing = records_by_document.get(document_key)
                if not existing or published_at > existing["publish_date"]:
                    records_by_document[document_key] = record
                elif record["preview"] and record["preview"] not in existing["preview"]:
                    existing["preview"] = (existing["preview"] + "\n" + record["preview"])[:50000]

            if offset is None:
                break
    except Exception as exc:
        logger.exception("[RELEASE_HISTORY] Failed to look up %s release documents: %s", metric, exc)
        return []

    records = sorted(records_by_document.values(), key=lambda record: record["publish_date"], reverse=True)
    logger.info("[RELEASE_HISTORY] %s: %d dated documents from %d points", metric, len(records), scanned)
    return records


_RELEASE_DATE_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10,
    "nov": 11, "dec": 12,
}
_RELEASE_DATE_MONTH_PATTERN = "|".join(sorted(_RELEASE_DATE_MONTHS, key=len, reverse=True))
_RELEASE_FUTURE_DATE_PATTERNS = (
    re.compile(rf"\b([0-3]?\d)(?:st|nd|rd|th)?\s+({_RELEASE_DATE_MONTH_PATTERN})\s*,?\s*(20\d{{2}})\b", re.IGNORECASE),
    re.compile(rf"\b({_RELEASE_DATE_MONTH_PATTERN})\s+([0-3]?\d)(?:st|nd|rd|th)?\s*,?\s*(20\d{{2}})\b", re.IGNORECASE),
    re.compile(r"\b([0-3]?\d)[./-]([01]?\d)[./-](20\d{2})\b"),
)


def _extract_explicit_future_release_dates(text: str, aliases: Tuple[str, ...]) -> List[datetime]:
    """Find dated product mentions in a document; never infer an omitted date."""
    if not text:
        return []

    today = ist_now().date()
    dates: List[datetime] = []
    seen = set()
    for alias in aliases:
        for alias_match in re.finditer(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
            start = max(0, alias_match.start() - 300)
            end = min(len(text), alias_match.end() + 900)
            window = text[start:end]
            for pattern_index, pattern in enumerate(_RELEASE_FUTURE_DATE_PATTERNS):
                for match in pattern.finditer(window):
                    try:
                        if pattern_index == 0:
                            day, month_name, year = match.groups()
                            candidate = datetime(int(year), _RELEASE_DATE_MONTHS[month_name.lower()], int(day))
                        elif pattern_index == 1:
                            month_name, day, year = match.groups()
                            candidate = datetime(int(year), _RELEASE_DATE_MONTHS[month_name.lower()], int(day))
                        else:
                            day, month, year = match.groups()
                            candidate = datetime(int(year), int(month), int(day))
                    except (KeyError, ValueError):
                        continue
                    if candidate.date() >= today and candidate.date() not in seen:
                        seen.add(candidate.date())
                        dates.append(candidate)
    return sorted(dates)


def _release_record_to_source_doc(record: Dict[str, Any]) -> Document:
    """Create one document-level source entry for a direct release answer."""
    return Document(
        page_content="",
        metadata={
            "title": record.get("title") or "",
            "file_name": record.get("file_name") or "",
            "file_url": record.get("file_url") or "",
        },
    )


def _answer_future_release_from_documents(hint: Dict[str, Any]) -> Tuple[str, List[str], List[Document]]:
    """Answer only from an explicit future date in an indexed document."""
    products = hint.get("metrics") or []
    requested_frequencies = hint.get("frequencies") or []
    result_lines = []
    source_names = []
    source_docs = []

    for product in products:
        records = _find_release_history_records(product)
        if requested_frequencies:
            records = [
                record for record in records
                if record.get("frequency") in requested_frequencies
            ]

        candidates = []
        aliases = _RELEASE_LOOKUP_PRODUCTS.get(product, ())
        for record in records:
            for date_value in _extract_explicit_future_release_dates(
                " ".join(str(record.get(key) or "") for key in ("title", "file_name", "preview")),
                aliases,
            ):
                candidates.append((date_value, record))

        if not candidates:
            result_lines.append(
                f"- {product}: no future release date is explicitly mentioned in the indexed {product} documents."
            )
            continue

        # The nearest explicitly stated date is the next release. A newer
        # document wins when documents state the same date.
        date_value, record = min(
            candidates,
            key=lambda item: (item[0], -item[1]["publish_date"].timestamp()),
        )
        date_text = date_value.strftime("%d %B %Y").lstrip("0")
        source_name = record["title"] or record["file_name"] or "Indexed release document"
        result_lines.append(
            f"- {product}: next release date is {date_text}."
        )
        if source_name not in source_names:
            source_names.append(source_name)
            source_docs.append(_release_record_to_source_doc(record))

    return "Next release date(s):\n\n" + "\n".join(result_lines), source_names, source_docs


def _answer_release_history(hint: Dict[str, Any]) -> Tuple[str, List[str], List[Document]]:
    """Build deterministic product release dates from already-ingested files."""
    products = hint.get("metrics") or []
    requested_frequencies = hint.get("frequencies") or []
    mode = hint.get("mode", "history")
    month = hint.get("period_month")
    year = hint.get("period_year")

    if mode == "future":
        return _answer_future_release_from_documents(hint)

    result_lines = []
    source_names = []
    source_docs = []
    period_label = f"{_RELEASE_LOOKUP_MONTHS[month]} {year}" if month and year else None
    for product in products:
        records = _find_release_history_records(product)
        if month and year:
            if hint.get("period_kind") == "publication":
                records = [
                    record for record in records
                    if record["publish_date"].year == year and record["publish_date"].month == month
                ]
            else:
                records = [
                    record for record in records
                    if _release_record_matches_data_period(record, month, year)
                ]

        if requested_frequencies:
            records = [
                record for record in records
                if record.get("frequency") in requested_frequencies
            ]

        if not records:
            detail = f" for {period_label}" if period_label else ""
            frequency_detail = f" ({', '.join(requested_frequencies)})" if requested_frequencies else ""
            result_lines.append(f"- {product}{frequency_detail}: no matching dated release document was found{detail}.")
            continue

        # With no requested cadence, one newest document is returned for every
        # cadence present in the indexed product files (PLFS/RBI can have many).
        records_by_frequency: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            records_by_frequency[record.get("frequency", "general")].append(record)
        frequency_order = requested_frequencies or [
            frequency for frequency in _RELEASE_FREQUENCY_ORDER
            if frequency in records_by_frequency
        ]
        for frequency in frequency_order:
            frequency_records = records_by_frequency.get(frequency, [])
            if not frequency_records:
                continue
            record = frequency_records[0]
            date_text = record["publish_date"].strftime("%d %B %Y").lstrip("0")
            product_label = product if frequency == "general" else f"{product} ({frequency})"
            if period_label and hint.get("period_kind") != "publication":
                result_lines.append(f"- {product_label} for {period_label}: published on {date_text}.")
            elif period_label:
                result_lines.append(f"- {product_label} release in {period_label}: published on {date_text}.")
            else:
                result_lines.append(f"- {product_label}: latest ingested release document was published on {date_text}.")

            source_name = record["title"] or record["file_name"]
            if source_name and source_name not in source_names:
                source_names.append(source_name)
                source_docs.append(_release_record_to_source_doc(record))

    heading = "Release dates from the latest ingested source files:"
    return heading + "\n\n" + "\n".join(result_lines), source_names, source_docs


async def handle_question(request):
    request_start_time = time.time()  # Track total request time
    session_id = request.session_id
    original_query = (request.query or "").strip()

    # Predefined responses (fast path)
    normalized_query = original_query.lower()
    if normalized_query in PREDEFINED_RESPONSES:
        predefined_response = PREDEFINED_RESPONSES[normalized_query]
        if session_id in memory_sessions:
            pre_created_at, session_context = memory_sessions[session_id]
            
            # Handle backward compatibility
            if isinstance(session_context, dict):
                memory = session_context["conversation_memory"]
            else:
                memory = session_context
                
            memory.chat_memory.add_user_message(original_query)
            memory.chat_memory.add_ai_message(predefined_response)

            # Not a clarify response; reset streak if session_context is the enhanced dict form
            if isinstance(session_context, dict):
                session_context["clarify_streak"] = 0

            # Persist updated conversation history back to Redis (see note in handler)
            try:
                memory_sessions[session_id] = (pre_created_at, session_context)
            except Exception as e:
                logger.error(f"[Session {session_id}] Failed to persist predefined interaction context: {e}", exc_info=True)
            
            # Save predefined response interaction to MongoDB
            try:
                total_time_ms = (time.time() - request_start_time) * 1000
                interaction = Interaction(
                    session_id=session_id,
                    timestamp=ist_now(),
                    query=original_query,
                    response=predefined_response,
                    sources=["Predefined Response"],
                    response_time_ms=total_time_ms,
                    component_timings={"total_ms": round(total_time_ms, 2)}
                )
                await interaction.insert()
                await AnalyticsService.track_user_activity(session_id)
                logger.info(f"[Session {session_id}]  Saved predefined response interaction to MongoDB")
            except Exception as e:
                logger.error(f"[Session {session_id}] ❌ Failed to save predefined interaction: {e}", exc_info=True)


        logger.info(f"[Session {session_id}] [TIMING] Predefined response | Total: {time.time() - request_start_time:.2f}s")
        return StreamingResponse(iter([predefined_response]), media_type="text/plain")

    if session_id not in memory_sessions:
        raise HTTPException(status_code=404, detail="Invalid session ID.")

    session_created_at, session_context = memory_sessions[session_id]
    
    # Handle backward compatibility - convert old memory format to new enhanced format
    if not isinstance(session_context, dict):
        old_memory = session_context
        session_context = create_enhanced_session_context()
        session_context["conversation_memory"] = old_memory
    
    memory = session_context["conversation_memory"]

    def _persist_session():
        try:
            memory_sessions[session_id] = (session_created_at, session_context)
        except Exception as e:
            logger.error(f"[Session {session_id}] Failed to persist session context: {e}", exc_info=True)

    prior_clarify_streak = int(session_context.get("clarify_streak", 0) or 0)
    escalate_clarify = prior_clarify_streak >= 1
    if escalate_clarify:
        logger.info(f"[Session {session_id}] [CLARIFY] Prior clarify_streak={prior_clarify_streak}; will escalate to redirect if still unclear")

    # Language + translation
    is_hindi = (detect_language(original_query) == "hi")
    query_en = translate_to_english(original_query)
    
    # Expand short forms (acronyms) in English query to improve retrieval
    query_en_expanded = expand_short_forms_in_query(query_en)

    logger.info(f"[Session {session_id}] 🟦 user='{original_query}' | en='{query_en}' | expanded='{query_en_expanded}'")

    # Guardrails (pre)
    allowed, msg, reason = apply_guardrails(query_en, is_hindi)
    if not allowed:
        logger.info(f"[Session {session_id}] [GUARDRAILS] blocked(pre) reason={reason}")
        memory.chat_memory.add_user_message(original_query)
        memory.chat_memory.add_ai_message(msg)
        session_context["clarify_streak"] = 0  # Not a clarify response; reset streak
        _persist_session()
        
        # Save guardrails-blocked interaction to MongoDB
        try:
            total_time_ms = (time.time() - request_start_time) * 1000
            interaction = Interaction(
                session_id=session_id,
                timestamp=ist_now(),
                query=original_query,
                response=msg,
                sources=["Guardrails"],
                response_time_ms=total_time_ms,
                component_timings={"total_ms": round(total_time_ms, 2)}
            )
            await interaction.insert()
            await AnalyticsService.track_user_activity(session_id)
            logger.info(f"[Session {session_id}]  Saved guardrails interaction to MongoDB")
        except Exception as e:
            logger.error(f"[Session {session_id}] ❌ Failed to save guardrails interaction: {e}", exc_info=True)
        
        return StreamingResponse(iter([msg]), media_type="text/plain")

    # Out-of-scope detection (hallucination prevention)
    is_out_of_scope, out_of_scope_response, oos_reason = detect_out_of_scope_query(query_en)
    if is_out_of_scope:
        logger.info(f"[Session {session_id}] [OUT_OF_SCOPE] detected reason = {oos_reason}")
        memory.chat_memory.add_user_message(original_query)
        memory.chat_memory.add_ai_message(out_of_scope_response)
        session_context["clarify_streak"] = 0  # Not a clarify response; reset streak
        _persist_session()
        
        # Save out-of-scope interaction to MongoDB
        try:
            total_time_ms = (time.time() - request_start_time) * 1000
            interaction = Interaction(
                session_id=session_id,
                timestamp=ist_now(),
                query=original_query,
                response=out_of_scope_response,
                sources=["Out-of-Scope"],
                response_time_ms=total_time_ms,
                component_timings={"total_ms": round(total_time_ms, 2)}
            )
            await interaction.insert()
            await AnalyticsService.track_user_activity(session_id)
            logger.info(f"[Session {session_id}]  Saved out-of-scope interaction to MongoDB")
        except Exception as e:
            logger.error(f"[Session {session_id}] ❌ Failed to save out-of-scope interaction: {e}", exc_info=True)
        
        return StreamingResponse(iter([out_of_scope_response]), media_type="text/plain")

    # Optional domain routing (toggleable; implemented in domain_routing.py)
    route = None
    try:
        from query_routing import apply_domain_routes  # local file next to chatbot_qdrant.py
        route = apply_domain_routes(query_en, is_hindi)
        # #region agent log
        logger.info(f"[DEBUG] apply_domain_routes result: query_en='{query_en}', has_response={bool(route and route.response)}, route_name={route.route_name if route else None}, response_preview='{route.response[:200] if route and route.response else None}'")
        # #endregion
        if route and route.response:
            logger.info(f"[Session {session_id}] [ROUTING] shortcircuit route={route.route_name}")
            memory.chat_memory.add_user_message(original_query)
            memory.chat_memory.add_ai_message(route.response)
            session_context["clarify_streak"] = 0  # Not a clarify response; reset streak
            _persist_session()
            
            # Save domain routing interaction to MongoDB
            try:
                total_time_ms = (time.time() - request_start_time) * 1000
                interaction = Interaction(
                    session_id=session_id,
                    timestamp=ist_now(),
                    query=original_query,
                    response=route.response,
                    sources=[f"Domain Routing: {route.route_name}"],
                    response_time_ms=total_time_ms,
                    component_timings={"total_ms": round(total_time_ms, 2)}
                )
                await interaction.insert()
                await AnalyticsService.track_user_activity(session_id)
                logger.info(f"[Session {session_id}]  Saved domain routing interaction to MongoDB")
            except Exception as e:
                logger.error(f"[Session {session_id}] ❌ Failed to save domain routing interaction: {e}", exc_info=True)
            
            return StreamingResponse(iter([route.response]), media_type="text/plain")
    except Exception as e:
        logger.info(f"[Session {session_id}] [ROUTING] skipped/unavailable: {e}")
        route = None

    # Deterministic release-history answers use document metadata rather than
    # passing an old release-schedule sentence through the LLM.
    release_history_hint = None
    if route and route.retrieval_hints:
        release_history_hint = route.retrieval_hints.get("release_history")
    if release_history_hint:
        release_answer, release_sources, release_source_docs = _answer_release_history(release_history_hint)
        if is_hindi:
            release_answer = translate_to_hindi(release_answer).strip()

        release_sources_block = format_sources(release_source_docs)
        if release_sources_block:
            release_answer += "\n\n" + release_sources_block
        elif release_sources:
            release_answer += "\n\n📄 **Sources:**\n" + "\n".join(
                f"- {source}" for source in release_sources
            )

        memory.chat_memory.add_user_message(original_query)
        memory.chat_memory.add_ai_message(release_answer)
        session_context["clarify_streak"] = 0
        _persist_session()

        interaction_id = None
        try:
            total_time_ms = (time.time() - request_start_time) * 1000
            interaction = Interaction(
                session_id=session_id,
                timestamp=ist_now(),
                query=original_query,
                response=release_answer,
                sources=release_sources or ["Release History Lookup"],
                response_time_ms=total_time_ms,
                component_timings={"release_history_ms": round(total_time_ms, 2)},
            )
            await interaction.insert()
            await AnalyticsService.track_user_activity(session_id)
            interaction_id = str(interaction.id)
        except Exception as e:
            logger.error(
                f"[Session {session_id}] Failed to save release-history interaction: {e}",
                exc_info=True,
            )

        logger.info(
            f"[Session {session_id}] [RELEASE_HISTORY] Served metrics="
            f"{release_history_hint.get('metrics', [])} in {(time.time() - request_start_time):.2f}s"
        )

        if interaction_id:
            release_answer += f"\n\n🆔 Interaction ID: {interaction_id}"
        return StreamingResponse(iter([release_answer]), media_type="text/plain")


    query_rewrite_start = time.time()
    expanded_query = expand_query_with_llm(query_en_expanded, llm_query_rewrite, session_context)
    query_rewrite_time = time.time() - query_rewrite_start
    logger.info(f"[Session {session_id}] [QUERY_REWRITE] '{query_en}' -> '{expanded_query}'")

    # Guardrails (post rewrite)
    allowed2, msg2, reason2 = apply_guardrails(expanded_query, is_hindi)
    if not allowed2:
        logger.info(f"[Session {session_id}] [GUARDRAILS] blocked(post-rewrite) reason={reason2}")
        memory.chat_memory.add_user_message(original_query)
        memory.chat_memory.add_ai_message(msg2)
        session_context["clarify_streak"] = 0  # Not a clarify response; reset streak
        _persist_session()
        
        # Save guardrails-blocked interaction to MongoDB (post-rewrite)
        try:
            total_time_ms = (time.time() - request_start_time) * 1000
            interaction = Interaction(
                session_id=session_id,
                timestamp=ist_now(),
                query=original_query,
                response=msg2,
                sources=["Guardrails (Post-Rewrite)"],
                response_time_ms=total_time_ms,
                component_timings={
                    "query_rewrite_ms": round(query_rewrite_time * 1000, 2),
                    "total_ms": round(total_time_ms, 2)
                }
            )
            await interaction.insert()
            await AnalyticsService.track_user_activity(session_id)
            logger.info(f"[Session {session_id}]  Saved guardrails (post-rewrite) interaction to MongoDB")
        except Exception as e:
            logger.error(f"[Session {session_id}] ❌ Failed to save guardrails (post-rewrite) interaction: {e}", exc_info=True)
        
        return StreamingResponse(iter([msg2]), media_type="text/plain")


    is_out_of_scope2, out_of_scope_response2, oos_reason2 = detect_out_of_scope_query(expanded_query)
    if is_out_of_scope2:
        logger.info(f"[Session {session_id}] [OUT_OF_SCOPE] detected (post-rewrite) reason = {oos_reason2}")
        memory.chat_memory.add_user_message(original_query)
        memory.chat_memory.add_ai_message(out_of_scope_response2)
        session_context["clarify_streak"] = 0  # Not a clarify response; reset streak
        _persist_session()
        
        # Save out-of-scope interaction to MongoDB (post-rewrite)
        try:
            total_time_ms = (time.time() - request_start_time) * 1000
            interaction = Interaction(
                session_id=session_id,
                timestamp=ist_now(),
                query=original_query,
                response=out_of_scope_response2,
                sources=["Out-of-Scope (Post-Rewrite)"],
                response_time_ms=total_time_ms,
                component_timings={
                    "query_rewrite_ms": round(query_rewrite_time * 1000, 2),
                    "total_ms": round(total_time_ms, 2)
                }
            )
            await interaction.insert()
            await AnalyticsService.track_user_activity(session_id)
            logger.info(f"[Session {session_id}]  Saved out-of-scope (post-rewrite) interaction to MongoDB")
        except Exception as e:
            logger.error(f"[Session {session_id}] ❌ Failed to save out-of-scope (post-rewrite) interaction: {e}", exc_info=True)
        
        return StreamingResponse(iter([out_of_scope_response2]), media_type="text/plain")

    # Check for visualization query (check both original and expanded query, like old pipeline)
    # Import here to avoid circular dependency
    wants_visualization = False
    try:
        from query_routing import is_visualization_query
        wants_visualization = is_visualization_query(query_en) or is_visualization_query(expanded_query)
        # #region agent log
        logger.info(f"[DEBUG] wants_visualization check: query_en='{query_en}', expanded_query='{expanded_query}', wants_visualization={wants_visualization}, query_en_check={is_visualization_query(query_en)}, expanded_query_check={is_visualization_query(expanded_query)}")
        # #endregion
        if wants_visualization:
            logger.info(f"-------------------[VIS] Detected visualization query: '{query_en}' -> '{expanded_query}'--------------")
        else:
            logger.debug(f"----------------- [VIS] Not a visualization query (checked via query_routing)---------------")
    except Exception as e:
        # Fallback: simple keyword check if import fails
        vis_keywords = ["graph", "trend", "chart", "visual", "plot", "line graph", "bar chart", "pie chart", "visualization", "visuals", "dashboard"]
        wants_visualization = any(kw in query_en.lower() for kw in vis_keywords) or any(kw in expanded_query.lower() for kw in vis_keywords)
        # #region agent log
        matched_kw = [kw for kw in vis_keywords if kw in query_en.lower() or kw in expanded_query.lower()]
        logger.info(f"[DEBUG] wants_visualization fallback check: query_en='{query_en}', expanded_query='{expanded_query}', wants_visualization={wants_visualization}, error={str(e)}, matched_keywords={matched_kw}")
        # #endregion
        if wants_visualization:
            logger.info(f"-------------------[VIS] Detected visualization query (fallback): '{query_en}' -> '{expanded_query}'--------------")
        else:
            logger.warning(f"--------------- [VIS] Import failed and fallback didn't detect visualization: {e}-------------=")


    topics_boost = None
    whois_prefer_file = False
    metric_cadence_hints = None
    if route and route.retrieval_hints:
        topics_boost = route.retrieval_hints.get("topics_boost")
        whois_prefer_file = bool(route.retrieval_hints.get("whois_prefer_file", False))
        metric_cadence_hints = route.retrieval_hints.get("metric_cadence")
        
        # Log cadence hints if present
        if metric_cadence_hints:
            logger.info(f"[CADENCE] Cadence hints received from query_routing: "
                       f"metric={metric_cadence_hints.get('metric_name')}, "
                       f"cadence={metric_cadence_hints.get('cadence')}, "
                       f"date_ranges={len(metric_cadence_hints.get('date_ranges', []))} ranges")

    # If cadence routing detected a metric query, use the rewritten query
    retrieval_query = expanded_query
    if metric_cadence_hints:
        retrieval_query = metric_cadence_hints.get("rewritten_query", expanded_query)
        logger.info(f"[CADENCE] Using rewritten query from cadence routing: '{retrieval_query}'")

    # Visualization-first approach: For visualization queries, try visualization collection first
    vis_docs: List[Document] = []
    vis_embeds: List[str] = []
    
    if wants_visualization:
        logger.info(f"------------------------- [VIS] Attempting to retrieve visualizations for query: '{expanded_query}'------------")
        try:
            vis_docs = retrieve_visualization_docs(expanded_query, k=1)  # Get top 1 result
            vis_embeds = extract_visualization_embeds(vis_docs)
            logger.info(f"------------------------[VIS] Retrieved {len(vis_docs)} visualization docs, extracted {len(vis_embeds)} embeds--------------")
            
            if vis_embeds:
                # SUCCESS: Found relevant visualization, return visualization-only response
                logger.info(f"---------------------[VIS] Found relevant visualization, returning visualization-only response--------------")
                
                # Get chart info for source attribution
                chart_title = vis_docs[0].metadata.get("chart_title", "Visualization") if vis_docs else "Visualization"
                chart_url = vis_docs[0].metadata.get("chart_url", "") if vis_docs else ""
                
                # Create visualization-only response
                if is_hindi:
                    vis_msg = f"यहाँ आपके प्रश्न से संबंधित दृश्य प्रतिनिधित्व है:\n\n📊 **{chart_title}**"
                else:
                    vis_msg = f"Here is a visual representation related to your query:\n\n📊 **{chart_title}**"
                
                # Add embed code
                output_response = vis_msg + "\n\n" + vis_embeds[0]
                
                # Add source information
                if chart_url:
                    source_info = f"\n\n**Source:** [{chart_title}]({chart_url})"
                    output_response += source_info
                
                memory.chat_memory.add_user_message(original_query)
                memory.chat_memory.add_ai_message(output_response)
                session_context["clarify_streak"] = 0  # Not a clarify response; reset streak
                _persist_session()
                return StreamingResponse(iter([output_response]), media_type="text/plain")
            else:
                # NO VISUALIZATION FOUND: Fall back to general search
                logger.info(f"------------------[VIS] No relevant visualization found, falling back to general document search------------")
                fallback_msg = (
                    "कोई संबंधित चार्ट उपलब्ध नहीं है। सामान्य खोज परिणाम प्रस्तुत कर रहे हैं:"
                    if is_hindi else
                    "No relevant chart available. Showing general search results:"
                )
                # Continue to general document search below
                vis_fallback_message = fallback_msg
                
        except Exception as e:
            logger.error(f"--------------[VIS] Failed to retrieve visualizations: {e}-------------", exc_info=True)
            # Continue to general document search on error
            vis_fallback_message = (
                "विज़ुअलाइज़ेशन खोजने में त्रुटि। सामान्य खोज परिणाम प्रस्तुत कर रहे हैं:"
                if is_hindi else
                "Error finding visualizations. Showing general search results:"
            )
    else:
        logger.debug(f"-------------------[VIS] Not a visualization query, proceeding with general document search-------------")
        vis_fallback_message = None

    # General document retrieval (only if visualization query failed or not a visualization query)
    logger.info(f"[RETRIEVAL] Proceeding with general document search")
    

    retrieval_start = time.time()
    
    # Use V2 retrieval (simplified: 5 vector + 5 metric → rerank → 10 to LLM)
    logger.info(f"[RETRIEVAL] Using V2 (simplified: 5 vector + 5 metric → rerank → 10 to LLM)")
    reranked_with_scores, _, cadence_match_info, detected_metric = retrieve_priority_chunks_v2(
        retrieval_query,
        info_vectordb,
        rerank_function=rerank_documents,  # Use original BGE reranker
        final_top_k=10,  # V2 returns 10 docs
        original_query=query_en,
        cadence_hints=metric_cadence_hints,  # NEW: Pass cadence hints from query_routing
        whois_prefer_file=whois_prefer_file,  # NEW: Pass whois routing hint
    )
    
    retrieval_time = time.time() - retrieval_start
    logger.info(f"[Session {session_id}] [TIMING] Retrieval | Time: {retrieval_time:.2f}s | Docs: {len(reranked_with_scores)}")
    
    # Log detected metric for source filtering
    if detected_metric:
        logger.info(f"[METRIC_DETECTION] Detected metric: {detected_metric} - will filter sources accordingly")
    
    # If cadence routing was used but no matching docs found, return fallback
    if metric_cadence_hints and cadence_match_info and not cadence_match_info.get("has_matches"):
        metric_name = cadence_match_info.get("metric_name", "metric")
        expected_periods = cadence_match_info.get("expected_periods", [])
        periods_str = ", ".join(expected_periods[:2]) if expected_periods else "recent periods"
        
        if is_hindi:
            fallback_msg = f"❌ माफ़ कीजिए, {metric_name} के लिए {periods_str} का डेटा अभी उपलब्ध नहीं है। कृपया बाद में पुनः प्रयास करें या MoSPI की आधिकारिक वेबसाइट देखें।"
        else:
            fallback_msg = f"❌ Sorry, the latest {metric_name} data for {periods_str} is not yet available in our database. Please check back later or visit the official MoSPI website for the most recent releases."
        
        logger.info(f"[CADENCE] No matches found for {metric_name}, returning fallback")
        memory.chat_memory.add_user_message(original_query)
        memory.chat_memory.add_ai_message(fallback_msg)
        session_context["clarify_streak"] = 0  # Not a clarify response; reset streak
        _persist_session()
        return StreamingResponse(iter([fallback_msg]), media_type="text/plain")
    
    docs = [d for (d, _s) in reranked_with_scores]
    
    # METRIC FILTERING: Filter documents BEFORE passing to LLM (configurable via ENABLE_METRIC_FILTERING)
    # This ensures LLM only sees relevant documents for the detected metric
    if ENABLE_METRIC_FILTERING and detected_metric:
        docs_before_filter = len(docs)
        docs = filter_documents_by_metric(docs, detected_metric)
        docs_after_filter = len(docs)
        
        # If all documents were filtered out, log warning and use unfiltered docs as fallback
        if docs_after_filter == 0:
            logger.warning(f"[METRIC_FILTER] All {docs_before_filter} documents filtered out for metric={detected_metric}, using unfiltered docs as fallback")
            docs = [d for (d, _s) in reranked_with_scores]
        else:
            logger.info(f"[METRIC_FILTER] Successfully filtered {docs_before_filter} → {docs_after_filter} docs for metric={detected_metric}")
    elif detected_metric and not ENABLE_METRIC_FILTERING:
        logger.debug(f"[METRIC_FILTER] Metric filtering disabled (ENABLE_METRIC_FILTERING=False), passing all {len(docs)} docs to LLM")
    
    if DEBUG_SOURCES:
        try:
            labels = [_source_label_from_doc(d) for d in docs[:SOURCES_DEBUG_MAX_DOCS]]
            logger.info(f"[SOURCES_DEBUG][Session {session_id}] docs_sample_labels={labels}")
        except Exception as e:
            logger.exception(f"[SOURCES_DEBUG][Session {session_id}] Failed to compute sample labels: {e}")


    whois_direct_response = None
    whois_source_label = "whois_cache"
    if whois_prefer_file and docs:
        # Check if first doc is whois doc (it should be the only doc for whois queries)
        first_doc = docs[0]
        metadata = first_doc.metadata or {}
        file_name = str(metadata.get("file_name", ""))

        if (
            "whois" in file_name.lower()
            or "whos_who" in file_name.lower()
            or "whois_source" in metadata
        ):
            logger.info(f"[WHOIS_FILTER] Filtering whois content for query: '{query_en}'")
            original_content = first_doc.page_content
            filtered_content = _filter_whois_content_for_query(original_content, query_en)
            
            # Replace with filtered document
            source_metadata = first_doc.metadata.copy() if first_doc.metadata else {}
            # S3-refreshed cache metadata uses whois_url/whois_source, while
            # the normal source formatter expects file_url/file_name.
            if not source_metadata.get("file_url") and metadata.get("whois_url"):
                source_metadata["file_url"] = metadata["whois_url"]
            if not source_metadata.get("file_name") and metadata.get("whois_source"):
                source_metadata["file_name"] = metadata["whois_source"]

            filtered_doc = Document(
                page_content=filtered_content,
                metadata=source_metadata,
            )
            docs = [filtered_doc]
            whois_direct_response = filtered_content
            whois_source_label = _source_label_from_doc(first_doc) or "whois_cache"
            logger.info(f"[WHOIS_FILTER] Filtered whois doc: {len(original_content)} → {len(filtered_content)} chars")
            logger.info(f"[WHOIS_FILTER] Filtered content preview: {filtered_content[:300]}...")

    if whois_direct_response is not None:
        whois_sources_block = format_sources(docs)
        if not whois_sources_block:
            whois_sources_block = f"📄 **Sources:**\n- {whois_source_label}"
        whois_final_response = whois_direct_response + "\n\n" + whois_sources_block

        memory.chat_memory.add_user_message(original_query)
        memory.chat_memory.add_ai_message(whois_final_response)
        session_context["clarify_streak"] = 0
        update_session_context(
            session_context,
            original_query,
            whois_final_response,
            docs,
            topics=["who is who"],
        )
        _persist_session()

        interaction_id = None
        try:
            total_time_ms = (time.time() - request_start_time) * 1000
            interaction = Interaction(
                session_id=session_id,
                timestamp=ist_now(),
                query=original_query,
                response=whois_final_response,
                sources=[whois_source_label],
                response_time_ms=total_time_ms,
                component_timings={"whois_cache_ms": round(total_time_ms, 2)},
            )
            await interaction.insert()
            await AnalyticsService.track_user_activity(session_id)
            interaction_id = str(interaction.id)
        except Exception as e:
            logger.error(f"[Session {session_id}] Failed to save Who's Who interaction: {e}", exc_info=True)

        logger.info(f"[Session {session_id}] [WHOIS_CACHE] Served direct cache response")

        if interaction_id:
            whois_final_response += f"\n\n🆔 Interaction ID: {interaction_id}"
        return StreamingResponse(iter([whois_final_response]), media_type="text/plain")

    # Build prompt context
    context = build_context(docs)

    # Enhanced chat history with better context (increased from 4 to 8 messages)
    hist_msgs = memory.chat_memory.messages[-8:] if memory.chat_memory.messages else []
    chat_history = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in hist_msgs])
    
    # Add enhanced session context for better continuity
    enhanced_session_context = build_enhanced_context_for_llm(session_context, expanded_query)
    if enhanced_session_context:
        chat_history = enhanced_session_context + "\n\n=== CURRENT CONVERSATION ===\n" + chat_history


    memory.chat_memory.add_user_message(original_query)

    # Prepare sources block (will only be added if answer is not a fallback)
    sources_block = "\n\n" + format_sources(docs, detected_metric=detected_metric) if docs else ""

    # If Hindi: do non-stream (so we can translate cleanly)
    if is_hindi:
        # Build messages for ChatOllama
        messages = build_chat_messages_for_main_llm(expanded_query, context, chat_history, escalate_clarify=escalate_clarify)
        
        # Add visualization fallback message if applicable
        if 'vis_fallback_message' in locals() and vis_fallback_message:
            # Prepend to user message
            messages[1].content = vis_fallback_message + "\n\n" + messages[1].content
        
        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if hasattr(msg, 'type'):
                if msg.type == 'system':
                    openai_messages.append({"role": "system", "content": msg.content})
                elif msg.type == 'human':
                    openai_messages.append({"role": "user", "content": msg.content})
                elif msg.type == 'ai':
                    openai_messages.append({"role": "assistant", "content": msg.content})
            else:
                openai_messages.append(msg)
        
        response = llm_answer_client.chat.completions.create(
            model=VLLM_MODEL,
            messages=openai_messages,
            temperature=0.1
        )
        raw_text = response.choices[0].message.content
        # Strip reasoning traces from output for Hindi (non-streaming)
        answer = extract_final_answer(raw_text)
        answer = clean_labels(answer).strip()
        # Strip markdown tables to prevent table rendering in UI
        answer = strip_markdown_tables(answer)

        is_clarify = is_clarification_response(answer)
        is_fallback = (not is_clarify) and (
            not answer
            or is_fallback_response(answer)
            or (answer.strip() == fallback_response(is_hindi).strip())
        )
        if is_clarify:
            clarify_escalated = is_escalated_clarify_response(answer) or escalate_clarify
            logger.info(f"[Session {session_id}] [CLARIFY] Detected clarification response (Hindi, escalated={clarify_escalated})")
            answer = clarification_response(is_hindi=True, escalate=clarify_escalated)
            session_context["clarify_streak"] = 0 if clarify_escalated else (prior_clarify_streak + 1)
        elif is_fallback:
            logger.info(f"[Session {session_id}] [FALLBACK] Detected fallback response (Hindi). Forcing canonical fallback text")
            answer = fallback_response(is_hindi=True)
            session_context["clarify_streak"] = 0
        else:
            answer = translate_to_hindi(answer).strip()
            session_context["clarify_streak"] = 0
        
        # Add visualization fallback message to final response if applicable
        final_answer = answer
        if 'vis_fallback_message' in locals() and vis_fallback_message:
            final_answer = vis_fallback_message + "\n\n" + answer
        
        # Note: vis_embeds should be empty for visualization fallback cases
        vis_block = ("\n\n📊 **Visualizations:**\n" + "\n".join(vis_embeds)) if vis_embeds else ""
        # Only add sources if answer is not a fallback
        sources_to_add = "" if (is_fallback or is_clarify) else sources_block
        if is_fallback or is_clarify:
            logger.info(f"[Session {session_id}] [SOURCES] Skipping sources ({'clarify' if is_clarify else 'fallback'} detected, Hindi)")
        final_text = final_answer + vis_block + sources_to_add

        memory.chat_memory.add_ai_message(final_text)
        
        # Update enhanced session context
        topics = extract_topics_from_query_and_docs(expanded_query, docs)
        update_session_context(session_context, original_query, final_text, docs, topics)
        _persist_session()
        
        return StreamingResponse(iter([final_text]), media_type="text/plain")

    # English: stream only final answer to frontend, log full response with reasoning
    async def generate_streamed_response() -> AsyncGenerator[str, None]:
        llm_start_time = time.time()
        logger.info(f"[Session {session_id}] 🚀 Starting LLM streaming with message roles")
        logger.info(f"[Session {session_id}] [LLM] Using: gpt-oss:20b (ChatOllama)")
        
        # Build messages for ChatOllama
        messages = build_chat_messages_for_main_llm(expanded_query, context, chat_history, escalate_clarify=escalate_clarify)
        
        # Add visualization fallback message if applicable
        if 'vis_fallback_message' in locals() and vis_fallback_message:
            yield vis_fallback_message + "\n\n"
            # Prepend to user message
            messages[1].content = vis_fallback_message + "\n\n" + messages[1].content
        
        # Use handler to separate reasoning from final answer
        handler = ReasoningTraceStreamHandler()

        async for token in stream_llm_response(messages):
            # Process token - only final answer content is returned
            output = handler.process_token(token)
            if output:
                yield output
        
        # Flush any remaining content
        remaining = handler.flush()
        if remaining:
            yield remaining

        llm_time = time.time() - llm_start_time

        full_response_with_reasoning = handler.get_full_response()
        final_answer_only = handler.get_final_answer()
        reasoning_traces = handler.get_reasoning()
        
        if reasoning_traces:
            logger.info(f"[Session {session_id}] [REASONING] Model reasoning ({len(reasoning_traces)} chars):\n{reasoning_traces}")
        
        # Log the FULL final answer before any processing
        logger.info(f"[Session {session_id}] [RESPONSE_RAW] Final answer BEFORE processing ({len(final_answer_only)} chars):")
        logger.info(f"{'='*80}\n{final_answer_only}\n{'='*80}")
        
        logger.info(f"[Session {session_id}] [TIMING] LLM streaming | Time: {llm_time:.2f}s")

        answer_clean = clean_labels(final_answer_only).strip()
        # Strip markdown tables to prevent table rendering in UI
        answer_clean = strip_markdown_tables(answer_clean)
        
        # Log the answer AFTER table stripping
        logger.info(f"[Session {session_id}] [RESPONSE_CLEAN] Final answer AFTER strip_markdown_tables ({len(answer_clean)} chars):")
        logger.info(f"{'='*80}\n{answer_clean}\n{'='*80}")
 
        is_clarify = is_clarification_response(answer_clean)
        is_fallback = (not is_clarify) and (
            not answer_clean
            or is_fallback_response(answer_clean)
            or (answer_clean.strip() == fallback_response(False).strip())
        )

        if is_clarify:
            clarify_escalated = is_escalated_clarify_response(answer_clean) or escalate_clarify
            logger.info(f"[Session {session_id}] [CLARIFY] Detected clarification response (escalated={clarify_escalated})")
            session_context["clarify_streak"] = 0 if clarify_escalated else (prior_clarify_streak + 1)
        # Log fallback detection for debugging and enforce canonical fallback
        if is_fallback:
            logger.info(f"[Session {session_id}] [FALLBACK] Detected fallback response. Forcing canonical fallback text")
            answer_clean = fallback_response(is_hindi=False)
            session_context["clarify_streak"] = 0
        if not is_clarify and not is_fallback:
            session_context["clarify_streak"] = 0


        tail = ""
        # Note: vis_embeds should be empty for visualization fallback cases
        if vis_embeds:
            tail += "\n\n📊 **Visualizations:**\n" + "\n".join(vis_embeds)
        if not is_fallback and not is_clarify and sources_block:
            tail += sources_block
        else:
            logger.info(f"[Session {session_id}] [SOURCES] Skipping sources ({'clarify' if is_clarify else 'fallback'} detected)")
        yield tail

        # Build final text for memory (include fallback message if present)
        final_text = answer_clean + tail
        if 'vis_fallback_message' in locals() and vis_fallback_message:
            final_text = vis_fallback_message + "\n\n" + final_text
        memory.chat_memory.add_ai_message(final_text)
        
        # Update enhanced session context
        topics = extract_topics_from_query_and_docs(expanded_query, docs)
        update_session_context(session_context, original_query, final_text, docs, topics)


        _persist_session()
        
        # Save interaction to MongoDB
        try:

            source_names = []
            if docs and not is_fallback and not is_clarify:
                for doc in docs:
                    # Try multiple metadata field names for document identification
                    doc_name = (
                        doc.metadata.get('file_name') or 
                        doc.metadata.get('doc_name') or 
                        doc.metadata.get('url', '')
                    )
                    if doc_name and doc_name not in source_names:
                        source_names.append(doc_name)
            elif is_fallback or is_clarify:
                logger.info(f"[Session {session_id}] [SOURCES] Skipping sources in MongoDB save ({'clarify' if is_clarify else 'fallback'} detected)")
            
            # Calculate total response time and component timings
            total_time_ms = (time.time() - request_start_time) * 1000
            component_timings = {
                "query_rewrite_ms": round(query_rewrite_time * 1000, 2),
                "retrieval_ms": round(retrieval_time * 1000, 2),
                "llm_ms": round(llm_time * 1000, 2),
                "total_ms": round(total_time_ms, 2)
            }
            
            # Create and save interaction with timing data
            interaction = Interaction(
                session_id=session_id,
                timestamp=ist_now(),
                query=original_query,
                response=final_text,
                sources=source_names,
                response_time_ms=total_time_ms,
                component_timings=component_timings
            )
            await interaction.insert()
            logger.info(f"[Session {session_id}]  Saved interaction to MongoDB (query: '{original_query[:50]}...', {len(source_names)} sources, {total_time_ms:.2f}ms)")
            
            # Track user activity
            await AnalyticsService.track_user_activity(session_id)
            
            # Send Interaction ID to frontend (hidden in stream)
            yield f"\n\n🆔 Interaction ID: {interaction.id}"
            
        except Exception as e:
            logger.error(f"[Session {session_id}] ❌ Failed to save interaction: {e}", exc_info=True)
            # Don't crash the chatbot if interaction saving fails

        
        # Log total request time
        total_time = time.time() - request_start_time
        logger.info(f"[Session {session_id}] [TIMING] ⏱️ TOTAL REQUEST | Query Rewrite: {query_rewrite_time:.2f}s | Retrieval: {retrieval_time:.2f}s | LLM: {llm_time:.2f}s | Total: {total_time:.2f}s")

    return StreamingResponse(generate_streamed_response(), media_type="text/plain")



def list_all_documents():
    """List all document names"""
    try:
        return list_doc_names_qdrant(qdrant_client, INFO_COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Error in list_all_documents: {e}")
        return {"error": f"Failed to list documents: {str(e)}"}

def get_chunks_for_doc(doc_name: str):
    """Get all chunks for a document"""
    try:
        results, _ = qdrant_client.scroll(
            collection_name=INFO_COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,
            limit=5000,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_name",
                        match=qmodels.MatchValue(value=doc_name)
                    )
                ]
            )
        )
        
        chunks = [
            {
                "chunk_id": point.payload.get("chunk_id"),
                "doc_id": point.payload.get("doc_id"),
                "content": point.payload.get("content"),
                "chunk_type": point.payload.get("chunk_type", "text"),
                "uploaded_at": point.payload.get("uploaded_at")
            }
            for point in results
        ]
        
        return {
            "document_name": doc_name,
            "total_chunks": len(chunks),
            "chunks": chunks
        } if chunks else {"message": f"No chunks found for document: {doc_name}"}
    except Exception as e:
        return {"error": f"Internal error: {str(e)}"}

def delete_chunks_for_doc(doc_names):
    """Delete chunks for documents"""
    try:
        delete_chunks_for_doc_qdrant(qdrant_client, INFO_COLLECTION_NAME, doc_names)
        return {"status": "deleted", "documents": doc_names}
    except Exception as e:
        return {"error": f"Failed to delete: {str(e)}"}

def list_all_urls():
    """List all URLs in collection"""
    try:
        results, _ = qdrant_client.scroll(
            collection_name=INFO_COLLECTION_NAME,
            with_payload=["url"],
            with_vectors=False,
            limit=5000,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="category",
                        match=qmodels.MatchValue(value="url")
                    )
                ]
            )
        )
        
        urls = {
            point.payload.get("url")
            for point in results
            if point.payload and point.payload.get("url")
        }
        return sorted(urls)
    except Exception as e:
        print(f"❌ Error in list_all_urls: {e}")
        return {"error": f"Failed to list URLs: {str(e)}"}

def get_chunks_for_url(url: str):
    """Get all chunks for a URL"""
    try:
        results, _ = qdrant_client.scroll(
            collection_name=INFO_COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,
            limit=10000,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="url",
                        match=qmodels.MatchValue(value=url)
                    ),
                    qmodels.FieldCondition(
                        key="category",
                        match=qmodels.MatchValue(value="url")
                    )
                ]
            )
        )
        
        chunks = [
            {
                "chunk_id": p.payload.get("chunk_id"),
                "url": p.payload.get("url"),
                "page_title": p.payload.get("page_title"),
                "content": p.payload.get("content")
            }
            for p in results
        ]
        
        return {
            "url": url,
            "total_chunks": len(chunks),
            "chunks": chunks
        } if chunks else {"message": f"No chunks found for URL: {url}"}
    except Exception as e:
        return {"error": f"Internal error: {str(e)}"}

VALID_DOMAINS = ["NSS Rounds", "GDP", "CPI", "IIP"]

def update_doc_metadata(
    doc_names: List[str],
    uploaded_at: Optional[str] = None,
    domain: Optional[str] = None
):
    """Update metadata for documents"""
    if domain and domain not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain. Choose from: {', '.join(VALID_DOMAINS)}")

    if uploaded_at:
        try:
            if re.match(r"^\d{4}-\d{2}-\d{2}$", uploaded_at):
                timestamp = datetime.fromisoformat(uploaded_at + "T00:00:00").isoformat()
            else:
                timestamp = datetime.fromisoformat(uploaded_at).isoformat()
        except ValueError:
            raise ValueError("Invalid datetime format")
    else:
        timestamp = datetime.now().isoformat()

    metadata_update = {"uploaded_at": timestamp}
    if domain:
        metadata_update["domain"] = domain

    try:
        update_doc_metadata_qdrant(
            client=qdrant_client,
            collection_name=INFO_COLLECTION_NAME,
            doc_names=doc_names,
            metadata=metadata_update
        )
        return {
            "status": "success",
            "available_domains": VALID_DOMAINS,
            "details": {name: "updated" for name in doc_names}
        }
    except Exception as e:
        return {"error": f"Failed to update metadata: {str(e)}"}
