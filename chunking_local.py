import os
import re
import json
import time
import uuid
import logging
import hashlib
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from openai import OpenAI


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---- Config ----
VLLM_BASE_URL = "http://10.75.8.1:8001/v1"
VLLM_API_KEY = "EMPTY"

# Local file paths
LOCAL_TEXT_DIR = Path(os.getenv("LOCAL_TEXT_DIR", "./md_files"))
WEB_FILE_METADATA_PATH = Path("web_file_metadata.json")
PROCESSED_PDF_TO_TEXT_PATH = Path("processed_pdf_to_text_files.jsonl")
PROCESSED_CHUNK_LOGS_PATH = Path("processed_chunk_logs")
SKIPPED_FILES_LOG_PATH = Path("skipped_files.jsonl")
MULTIPLE_URL_LOG_PATH = Path("multiple_url_files.jsonl")


QDRANT_HOST = os.getenv("QDRANT_HOST", "10.75.8.1")
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mospi_collection_bge_v3")

# Model settings
HF_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

EMBED_MIN_FREE_GIB = float(os.getenv("EMBED_MIN_FREE_GIB", "1.5"))

EMBED_TEXTS_PER_GIB = int(os.getenv("EMBED_TEXTS_PER_GIB", "16"))

# LLM settings for topic inference
LLM_MODEL_NAME = "openai/gpt-oss-20b"
DOC_TOPIC_TOP_K = 10  # Get 10 topics from first 5 pages
CHUNK_TOPIC_TOP_K = 5  # Not used anymore, kept for compatibility
LLAMA_BATCH_SIZE = int(os.getenv("LLAMA_BATCH_SIZE", "32"))
LLM_MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "16"))

# Snippet sizing
DOC_SNIPPET_CHARS = int(os.getenv("DOC_SNIPPET_CHARS", "2000"))  # Per page
DOC_SNIPPET_PAGES = 5  # Use first 5 pages for topic inference
CHUNK_SNIPPET_CHARS = int(os.getenv("CHUNK_SNIPPET_CHARS", "2000"))
BATCH_ITEM_CHARS = int(os.getenv("BATCH_ITEM_CHARS", "1200"))

# Retry sizing
DOC_SNIPPET_CHARS_RETRY = int(os.getenv("DOC_SNIPPET_CHARS_RETRY", "1000"))
CHUNK_SNIPPET_CHARS_RETRY = int(os.getenv("CHUNK_SNIPPET_CHARS_RETRY", "1000"))
BATCH_ITEM_CHARS_RETRY = int(os.getenv("BATCH_ITEM_CHARS_RETRY", "600"))

# Upsert batch size
QDRANT_UPSERT_BATCH = int(os.getenv("QDRANT_UPSERT_BATCH", "256"))

# Skip topics for faster processing
SKIP_TOPICS = os.getenv("SKIP_TOPICS", "0").strip() not in ("0", "false", "no")

CANDIDATE_TOPICS = [
    "population statistics", "labor and employment", "industry", "agriculture",
    "national accounts", "national income", "price indices", "inflation",
    "trade and commerce", "infrastructure", "health statistics", "education statistics",
    "environment statistics", "census and demography", "household surveys",
    "poverty and inequality", "economic growth", "methodology and definitions","GDP",
    "CPI", "CFPI", "NSS Round","PLFS","Unemployment Rate", "Employment Rate",
    "Population","Inflation","Food Inflation","Employment Rate","Consumer Food Price Index", "Consumer Price Index","Index of Industrial Production", "IIP", "Annual Survey of Industries","ASI"
]


# ---- Logging ----
def setup_logger(name: str) -> logging.Logger:
    """Setup logger with both console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(console_handler)
        
        # File handler - write to logs/chunking.log
        try:
            from pathlib import Path
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / "chunking.log",
                mode='a',
                encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(file_handler)
        except Exception as e:
            # If file logging fails, continue with console only
            logger.warning(f"Could not setup file logging: {e}")
    
    return logger

# Initialize module-level logger for use in utility functions
logger = setup_logger("chunking_local")

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def append_jsonl(path: Path, entry: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # If file doesn't exist, create it with proper permissions
        if not path.exists():
            path.touch(mode=0o666)
        
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except PermissionError as e:
        logger.error(f"Permission denied writing to {path}: {e}")
        logger.warning(f"Skipping log entry for {path}")
        # Don't raise - allow processing to continue

# ---- Page Splitting + Chunking ----
@dataclass
class Page:
    number: int
    text: str

PAGE_MARKER_RE = re.compile(r"==\s*page\s+(\d+)\s*==", re.IGNORECASE)
CONTINUATION_RE = re.compile(r"(contd\.?|continued|cont\.?)\s*$", re.IGNORECASE)
SNIPPET_CTX_CHARS = 200

def looks_like_table_text(page_text: str) -> bool:
    lines = [l for l in (page_text or '').splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    pipe_lines = sum(1 for l in lines if '|' in l)
    if pipe_lines >= 3:
        return True
    spaced_cols = sum(1 for l in lines if re.search(r'\S\s{2,}\S', l))
    return spaced_cols >= 6

def page_starts_as_continuation(page_text: str) -> bool:
    lines = [l.strip() for l in (page_text or '').splitlines() if l.strip()]
    if not lines:
        return False
    first = lines[0]
    if re.match(r'^(contd\.?|continued|cont\.?)(\b|\s)', first, re.IGNORECASE):
        return True
    if first and first[0] in ',.;:)]}':
        return True
    if re.match(r'^[a-z]', first):
        return True
    if re.match(r'^(and|or|but|so|because|which|that|with|to|from|for|in|on|of)\b', first, re.IGNORECASE):
        return True
    return False


def split_text_into_pages(text: str) -> List[Page]:
    matches = list(PAGE_MARKER_RE.finditer(text))
    if matches:
        pages: List[Page] = []
        for i, m in enumerate(matches):
            page_num = int(m.group(1))
            start = m.end()
            end = matches[i + 1].start() if (i + 1) < len(matches) else len(text)
            pages.append(Page(number=page_num, text=text[start:end].strip()))
        return pages

    if "\f" in text:
        raw = [p.strip() for p in text.split("\f")]
        return [Page(number=i + 1, text=p) for i, p in enumerate(raw) if p.strip()]

    pattern = re.compile(r"\n\s*Page\s+\d+(\s+of\s+\d+)?\s*\n", re.IGNORECASE)
    splits: List[str] = []
    last_idx = 0
    for match in pattern.finditer(text):
        splits.append(text[last_idx:match.start()])
        last_idx = match.end()
    splits.append(text[last_idx:])
    cleaned = [s.strip() for s in splits if s.strip()]
    if len(cleaned) > 1:
        return [Page(number=i + 1, text=p) for i, p in enumerate(cleaned)]

    return [Page(number=1, text=text.strip())]

def page_has_overflow(page_text: str) -> bool:
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    if not lines:
        return False
    last = lines[-1]
    if CONTINUATION_RE.search(last):
        return True
    if len(last) < 20:
        return False
    if last.endswith((".", "!", "?", ":", ";")):
        return False
    return True

def chunk_document_into_pages(text: str) -> List[Dict[str, Any]]:
    pages = split_text_into_pages(text)
    out: List[Dict[str, Any]] = []

    if not pages:
        return out

    has_over = [page_has_overflow(p.text) for p in pages]
    has_table = [looks_like_table_text(p.text) for p in pages]

    for idx, page in enumerate(pages):
        curr_text = page.text or ""
        if not curr_text.strip():
            continue
        curr_table = has_table[idx]

        includes_prev = False
        prev_chars = 0
        prev_piece = ""
        if idx > 0:
            continued = bool(has_over[idx - 1]) or page_starts_as_continuation(curr_text)
            if continued:
                includes_prev = True
                if curr_table or has_table[idx - 1]:
                    prev_piece = pages[idx - 1].text
                else:
                    prev_piece = (pages[idx - 1].text or "")[-SNIPPET_CTX_CHARS:]
                prev_chars = len(prev_piece)

        includes_next = False
        next_chars = 0
        next_piece = ""
        if has_over[idx] and idx < len(pages) - 1:
            includes_next = True
            if curr_table or has_table[idx + 1]:
                next_piece = pages[idx + 1].text
            else:
                next_piece = (pages[idx + 1].text or "")[:SNIPPET_CTX_CHARS]
            next_chars = len(next_piece)

        parts = []
        if prev_piece.strip():
            parts.append(prev_piece.strip())
        if curr_text.strip():
            parts.append(curr_text.strip())
        if next_piece.strip():
            parts.append(next_piece.strip())

        chunk_text = "\n\n".join(parts).strip()

        out.append({
            "page_number": page.number,
            "page_has_overflow": bool(has_over[idx]),
            "includes_prev_page": bool(includes_prev),
            "prev_page_chars": int(prev_chars),
            "next_page_chars": int(next_chars),
            "page_content": chunk_text,
        })

    return out


# ---- Topic Inference ----
def _dedupe_topics_case_insensitive(topics: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for t in topics or []:
        if not isinstance(t, str):
            continue
        t2 = t.strip()
        if not t2:
            continue
        k = t2.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t2)
    return out

def make_topic_snippet(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) <= max_chars:
        return t
    half = max_chars // 2
    return (t[:half] + "\n...\n" + t[-half:]).strip()


def make_first_n_pages_snippet(chunks: List[Dict[str, Any]], n_pages: int, chars_per_page: int) -> str:
    """
    Extract snippets from first N pages and combine them.
    
    Args:
        chunks: List of page chunks
        n_pages: Number of pages to extract from
        chars_per_page: Characters to extract per page
    
    Returns:
        Combined snippet from first N pages
    """
    snippets = []
    for i, chunk in enumerate(chunks[:n_pages]):
        page_text = chunk.get("page_content", "")
        snippet = make_topic_snippet(page_text, chars_per_page)
        if snippet:
            snippets.append(f"[Page {i+1}]\n{snippet}")
    
    return "\n\n".join(snippets)


def _extract_json_blob(text: str) -> Optional[str]:
    if not text:
        return None
    s = str(text).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"```$", "", s).strip()

    start_candidates = [s.find("{"), s.find("[")]
    start_candidates = [i for i in start_candidates if i != -1]
    if not start_candidates:
        return None
    start = min(start_candidates)

    end_candidates = [s.rfind("}"), s.rfind("]")]
    end_candidates = [i for i in end_candidates if i != -1]
    if not end_candidates:
        return None
    end = max(end_candidates)

    if end <= start:
        return None
    return s[start:end + 1]

def infer_topics_single_once(llm_client: OpenAI, text: str, top_k: int, max_chars: int) -> Tuple[List[str], Optional[str]]:
    snippet = make_topic_snippet(text, max_chars=max_chars)
    if not snippet:
        return [], None

    prompt = f"""You are a helpful assistant that summarizes pages from official government statistics reports.

Task:
- Produce up to {top_k} short topic labels (noun phrases) that best describe the main subjects.

OUTPUT RULES:
- Output ONLY a valid JSON array of strings.
- No explanation text.
- Use double quotes for strings.

Text:
\"\"\"{snippet}\"\"\"

Hints (optional):
{", ".join(CANDIDATE_TOPICS)}
"""

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = response.choices[0].message.content
    except Exception as e:
        return [], f"invoke_error: {e}"

    blob = _extract_json_blob(raw)
    if not blob:
        return [], "no_json_blob"

    try:
        arr = json.loads(blob)
        if isinstance(arr, list):
            arr = [x for x in arr if isinstance(x, str)]
            return _dedupe_topics_case_insensitive(arr)[:top_k], None
        return [], "json_not_array"
    except Exception as e:
        return [], f"json_parse_error: {e}"


def infer_topics_single_resilient(
    llm_client: OpenAI,
    text: str,
    top_k: int,
    max_chars: int,
    max_chars_retry: int,
) -> Tuple[List[str], Optional[str]]:
    topics, err = infer_topics_single_once(llm_client, text, top_k, max_chars=max_chars)
    if err is None:
        return topics, None

    topics2, err2 = infer_topics_single_once(llm_client, text, top_k, max_chars=max_chars_retry)
    if err2 is None:
        return topics2, None

    return [], f"failed_after_retry: first={err} second={err2}"

def infer_topics_batch_once(
    llm_client: OpenAI,
    texts: List[str],
    top_k: int,
    per_item_chars: int,
) -> Tuple[Optional[List[List[str]]], Optional[str]]:
    if not texts:
        return [], None

    items = [{"i": i, "text": make_topic_snippet(texts[i], per_item_chars)} for i in range(len(texts))]

    prompt = f"""You are a helpful assistant that tags multiple text chunks.

For each item, generate up to {top_k} topic labels.

OUTPUT RULES:
- Output ONLY valid JSON.
- Output MUST be a JSON object mapping each i (as a string) -> JSON array of strings.
- No extra text.

Input:
{json.dumps(items, ensure_ascii=False)}

Hints (optional):
{", ".join(CANDIDATE_TOPICS)}
"""

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = response.choices[0].message.content
    except Exception as e:
        return None, f"invoke_error: {e}"

    blob = _extract_json_blob(raw)
    if not blob:
        return None, "no_json_blob"

    try:
        obj = json.loads(blob)
        if not isinstance(obj, dict):
            return None, "json_not_object"

        out: List[List[str]] = []
        for i in range(len(texts)):
            v = obj.get(str(i), [])
            if isinstance(v, list):
                v = [x for x in v if isinstance(x, str)]
                out.append(_dedupe_topics_case_insensitive(v)[:top_k])
            else:
                out.append([])
        return out, None

    except Exception as e:
        return None, f"json_parse_error: {e}"


def infer_topics_batch_resilient(
    llm_client: OpenAI,
    texts: List[str],
    top_k: int,
    batch_size: int,
    per_item_chars: int,
    per_item_chars_retry: int,
    single_max_chars: int,
    single_max_chars_retry: int,
) -> Tuple[List[List[str]], Optional[str]]:
    if not texts:
        return [], None

    res, err = infer_topics_batch_once(llm_client, texts, top_k, per_item_chars=per_item_chars)
    if err is None and res is not None:
        return res, None

    retry_reason = err or "unknown_batch_failure"
    retry_batch_size = max(1, min(len(texts), max(1, batch_size // 2)))

    all_out: List[List[str]] = []
    any_failure = False
    failure_msgs: List[str] = []

    for i in range(0, len(texts), retry_batch_size):
        sub = texts[i:i + retry_batch_size]
        sub_res, sub_err = infer_topics_batch_once(llm_client, sub, top_k, per_item_chars=per_item_chars_retry)
        if sub_err is None and sub_res is not None:
            all_out.extend(sub_res)
        else:
            any_failure = True
            failure_msgs.append(f"sub_batch_failed({i}-{i+len(sub)-1}): {sub_err}")
            for t in sub:
                single_topics, ferr = infer_topics_single_resilient(
                    llm_client, t, top_k,
                    max_chars=single_max_chars, max_chars_retry=single_max_chars_retry
                )
                if ferr is None:
                    all_out.append(single_topics)
                else:
                    all_out.append([])
                    failure_msgs.append(f"single_failed: {ferr}")

    if not any_failure:
        return all_out, None

    return all_out, f"failed_after_retry: batch_first={retry_reason}; " + " | ".join(failure_msgs)

def merge_topics(doc_topics: List[str], chunk_topics: List[str]) -> List[str]:
    return _dedupe_topics_case_insensitive((doc_topics or []) + (chunk_topics or []))

# ---- Metadata Loading ----
def extract_filename_from_url(url: str) -> Optional[str]:
    """Extract just the filename from a URL."""
    if not url or url == "about:blank":
        return None
    try:
        # Get the last part of the path
        from urllib.parse import urlparse, unquote
        parsed = urlparse(url)
        path = unquote(parsed.path)
        filename = path.split('/')[-1]
        return filename if filename else None
    except Exception:
        return None

def parse_date_to_iso(date_str: str) -> Optional[str]:
    """Parse date string to YYYY-MM-DD format. Handles DD-MM-YYYY and YYYY-MM-DD formats."""
    if not date_str:
        return None
    try:
        parts = date_str.split('-')
        if len(parts) == 3:
            # Check if already in YYYY-MM-DD format (year is 4 digits and first)
            if len(parts[0]) == 4 and parts[0].isdigit():
                # Already in YYYY-MM-DD format, validate and return
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                if 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
                    return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
            else:
                # Try DD-MM-YYYY format
                day, month, year = parts
                d, m, y = int(day), int(month), int(year)
                if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2100:
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    except Exception:
        pass
    return None


# Month name -> number mapping for filename date extraction
_MONTH_NAME_TO_NUM = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def extract_date_from_filename(filename: str) -> Optional[str]:
    if not filename:
        return None

    # Work on the name without extension, lowercased
    name = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE).lower()

    month_alt = "|".join(sorted(_MONTH_NAME_TO_NUM.keys(), key=len, reverse=True))

    nb = r"(?<![a-z0-9])"   # left boundary
    na = r"(?![a-z0-9])"    # right boundary
    sep = r"[ _.\-]"        # allowed separators: space, underscore, dot, hyphen

    # 1) Day Month(name) Year  e.g. 12_july_2026 / 12-jul-2026 / 12.july.2026
    m = re.search(rf"{nb}(\d{{1,2}}){sep}({month_alt}){sep}((?:19|20)\d{{2}}){na}", name)
    if m:
        day = int(m.group(1)); mon = _MONTH_NAME_TO_NUM[m.group(2)]; year = int(m.group(3))
        if 1 <= day <= 31:
            return f"{year:04d}-{mon:02d}-{day:02d}"

    # 2) Month(name) Year  e.g. july_2026 / july-2026 / july.2026 (day defaults to 01)
    m = re.search(rf"{nb}({month_alt}){sep}((?:19|20)\d{{2}}){na}", name)
    if m:
        mon = _MONTH_NAME_TO_NUM[m.group(1)]; year = int(m.group(2))
        return f"{year:04d}-{mon:02d}-01"

    # 3) Numeric DD sep MM sep YYYY  e.g. 10.07.2026 / 10-07-2026 / 10_07_2026
    m = re.search(rf"{nb}(\d{{1,2}}){sep}(\d{{1,2}}){sep}((?:19|20)\d{{2}}){na}", name)
    if m:
        d = int(m.group(1)); mo = int(m.group(2)); y = int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return f"{y:04d}-{mo:02d}-{d:02d}"

    # 4) Numeric YYYY sep MM sep DD  e.g. 2026-07-10 / 2026_07_10
    m = re.search(rf"{nb}((?:19|20)\d{{2}}){sep}(\d{{1,2}}){sep}(\d{{1,2}}){na}", name)
    if m:
        y = int(m.group(1)); mo = int(m.group(2)); d = int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"

    # 5) Numeric MM sep Year  e.g. 7.2026 / 07-2026 / 7_2026 (day defaults to 01)
    m = re.search(rf"{nb}(0?[1-9]|1[0-2]){sep}((?:19|20)\d{{2}}){na}", name)
    if m:
        mon = int(m.group(1)); year = int(m.group(2))
        return f"{year:04d}-{mon:02d}-01"

    return None


_EXCLUDED_EXACT_NAMES = {
    "field operations division (fod) directory.pdf",
}
_EXCLUDED_PREFIXES = ("whois_fod_", "whois_", "tmp")


def is_excluded_from_chunking(pdf_name: str) -> bool:
    if not pdf_name:
        return True
    name = pdf_name.strip().lower()
    if name in _EXCLUDED_EXACT_NAMES:
        return True
    for prefix in _EXCLUDED_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


def verify_title_with_content(
    llm_client: OpenAI,
    filename_title: str,
    content_snippet: str,
) -> Optional[str]:
    if not content_snippet or not content_snippet.strip():
        return filename_title

    prompt = f"""You are refining the title of an official government statistics document.

You are given a draft title (derived from the file name) and the beginning of the document text.
Return the single best, clean, human-readable title for the document.

Rules:
- Prefer an accurate title supported by the document text.
- If the draft title is already accurate, return it (cleaned up).
- Keep it concise (one line). Do NOT add commentary or quotes.
- Output ONLY the title text.

Draft title: "{filename_title}"

Document text (first {len(content_snippet)} chars):
\"\"\"{content_snippet}\"\"\"
"""
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        refined = (response.choices[0].message.content or "").strip()
        # Strip wrapping quotes if the model added them
        refined = refined.strip('"').strip()
        return refined if refined else filename_title
    except Exception as e:
        logger.warning(f"Title verification via content failed: {e}")
        return filename_title


def build_title(
    llm_client: OpenAI,
    pdf_name: str,
    doc_text: str,
) -> str:

    try:
        from title_generator import generate_title_from_filename
        filename_title = generate_title_from_filename(pdf_name, llm_client=llm_client)
    except Exception as e:
        logger.warning(f"Filename-based title generation failed for {pdf_name}: {e}")
        filename_title = None

    if not filename_title:
        # Last-resort fallback: strip extension from filename
        filename_title = re.sub(r"\.pdf$", "", pdf_name, flags=re.IGNORECASE)

    # Verify/refine using the first 1000 characters of the document text
    snippet = (doc_text or "")[:1000]
    return verify_title_with_content(llm_client, filename_title, snippet)


def load_web_file_metadata(logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    # Download metadata from S3 using boto3
    try:
        logger.info("Downloading web_file_metadata.json from S3...")
        from web_scrap.s3_helper import get_s3_client, download_file_from_s3
        
        s3_client = get_s3_client()
        bucket_name = "mospi"
        s3_key = "data/mospi_web/web_file_metadata.json"
        
        logger.info(f"Downloading from s3://{bucket_name}/{s3_key}")
        success = download_file_from_s3(
            s3_client, 
            bucket_name, 
            s3_key, 
            str(WEB_FILE_METADATA_PATH)
        )
        
        if success:
            logger.info(f"✓ Downloaded metadata successfully to {WEB_FILE_METADATA_PATH}")
        else:
            logger.error(
                "!! Failed to download web_file_metadata.json from S3 (after retries). "
                "Falling back to the LOCAL copy, which may be STALE - recently ingested "
                "files may be missing metadata (null publish_date/url). "
                "Check S3 connectivity/credentials from this host."
            )
    except Exception as e:
        logger.error(f"!! Error downloading metadata from S3: {e}. Falling back to LOCAL (possibly STALE) copy.")
    
    # Load metadata file
    if not WEB_FILE_METADATA_PATH.exists():
        logger.warning(f"Metadata file not found: {WEB_FILE_METADATA_PATH}")
        return {}
    
    metadata_by_filename: Dict[str, Dict[str, Any]] = {}
    try:
        with WEB_FILE_METADATA_PATH.open("r", encoding="utf-8") as f:
            entries = json.load(f)
        
        for entry in entries:
            # Use file_name directly instead of extracting from URL
            file_name = entry.get("file_name")
            
            if file_name:
                # Parse and convert date
                raw_date = entry.get("publish_date")
                iso_date = parse_date_to_iso(raw_date) if raw_date else None
                
                metadata_by_filename[file_name] = {
                    "title": entry.get("title"),
                    "published_date": iso_date,
                    "file_url": entry.get("file_url"),
                    "file_name": file_name,
                }
        
        logger.info(f"Loaded {len(metadata_by_filename)} metadata entries from {WEB_FILE_METADATA_PATH}")
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
    
    return metadata_by_filename



def load_processed_pdf_to_text(logger: logging.Logger) -> Dict[str, Dict[str, Any]]:

    try:
        from web_scrap.s3_helper import get_s3_client, download_file_from_s3

        s3_client = get_s3_client()
        bucket_name = "mospi"
        s3_key = "data/mospi_web/processed_pdf_to_text_files.jsonl"

        logger.info(f"Downloading processed PDF log from s3://{bucket_name}/{s3_key}")
        if download_file_from_s3(s3_client, bucket_name, s3_key, str(PROCESSED_PDF_TO_TEXT_PATH)):
            logger.info(f"✓ Synced processed PDF log from S3 to {PROCESSED_PDF_TO_TEXT_PATH}")
        else:
            logger.warning(
                "Could not download processed_pdf_to_text_files.jsonl from S3 "
                "(after retries). Falling back to LOCAL copy if present."
            )
    except Exception as e:
        logger.warning(f"Error syncing processed PDF log from S3: {e}. Using LOCAL copy if present.")

    if not PROCESSED_PDF_TO_TEXT_PATH.exists():
        logger.warning(f"Processed PDF log not found: {PROCESSED_PDF_TO_TEXT_PATH}")
        return {}
    
    pdf_by_md5: Dict[str, Dict[str, Any]] = {}
    try:
        with PROCESSED_PDF_TO_TEXT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    md5 = entry.get("md5")
                    if md5:
                        pdf_by_md5[md5] = entry
                except Exception:
                    continue
        
        logger.info(f"Loaded {len(pdf_by_md5)} processed PDF entries from {PROCESSED_PDF_TO_TEXT_PATH}")
    except Exception as e:
        logger.error(f"Failed to load processed PDF log: {e}")
    
    return pdf_by_md5

def load_processed_chunks(logger: logging.Logger) -> Set[str]:
    if not PROCESSED_CHUNK_LOGS_PATH.exists():
        logger.info(f"Local processed_chunk_logs not found, trying S3...")
        try:
            from web_scrap.s3_helper import get_s3_client, download_file_from_s3
            
            s3_client = get_s3_client()
            bucket_name = "acct1004215-mospi"
            s3_key = "data/mospi_web/processed_chunk_logs"
            
            success = download_file_from_s3(
                s3_client,
                bucket_name,
                s3_key,
                str(PROCESSED_CHUNK_LOGS_PATH)
            )
            
            if success:
                logger.info(f"✓ Downloaded processed_chunk_logs from S3")
            else:
                logger.info(f"No processed_chunk_logs found in S3, starting fresh")
        except Exception as e:
            logger.warning(f"Could not download processed_chunk_logs from S3: {e}")
    
    if not PROCESSED_CHUNK_LOGS_PATH.exists():
        logger.info(f"No processed chunks log found at: {PROCESSED_CHUNK_LOGS_PATH}")
        return set()
    
    processed: Set[str] = set()
    try:
        with PROCESSED_CHUNK_LOGS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("status") == "SUCCESS":
                        md5 = entry.get("md5")
                        if md5:
                            processed.add(md5)
                except Exception:
                    continue
        
        logger.info(f"Loaded {len(processed)} processed chunk md5s from {PROCESSED_CHUNK_LOGS_PATH}")
    except Exception as e:
        logger.error(f"Failed to load processed chunks: {e}")
    
    return processed


# ---- Qdrant Helpers ----
def make_qdrant_client(logger: Optional[logging.Logger] = None):
    """Create Qdrant client with gRPC for better performance."""
    try:
        from qdrant_client import QdrantClient
    except Exception as e:
        if logger:
            logger.exception("Failed to import qdrant_client: %s", e)
        raise

    if logger:
        logger.info(f"Connecting to Qdrant via gRPC at {QDRANT_HOST}:{QDRANT_GRPC_PORT}")
    
    kwargs = {
        "host": QDRANT_HOST,
        "grpc_port": QDRANT_GRPC_PORT,
        "prefer_grpc": True,
        "https": False,  # Disable SSL/TLS for gRPC
    }
    
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    
    client = QdrantClient(**kwargs)
    
    if logger:
        logger.info(f"✓ Connected to Qdrant via gRPC successfully")
    
    return client

def ensure_qdrant_collection(logger: logging.Logger) -> None:
    """Verify Qdrant collection exists. FAILS if collection doesn't exist (production safety)."""
    try:
        logger.info("Connecting to Qdrant...")
        client = make_qdrant_client(logger=logger)
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}")
        raise

    try:
        logger.info(f"Checking if collection '{QDRANT_COLLECTION}' exists...")
        client.get_collection(collection_name=QDRANT_COLLECTION)
        logger.info(f"✓ Qdrant collection exists: {QDRANT_COLLECTION}")
    except Exception as e:
        # List available collections for debugging
        try:
            collections = client.get_collections()
            available = [c.name for c in collections.collections]
            logger.error(f"❌ Collection '{QDRANT_COLLECTION}' does NOT exist!")
            logger.error(f"Available collections: {available}")
        except Exception as list_error:
            logger.error(f"❌ Collection '{QDRANT_COLLECTION}' does NOT exist!")
            logger.error(f"Could not list available collections: {list_error}")
        
        raise RuntimeError(
            f"PRODUCTION SAFETY: Collection '{QDRANT_COLLECTION}' must exist before running chunking. "
            f"This script will NOT create collections automatically. "
            f"Please create the collection manually or contact admin."
        )

def make_point_id(md5: str, page_number: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{md5}:{page_number}"))

def get_existing_points_by_md5(md5: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Retrieve all existing points with given md5 from Qdrant."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        client = make_qdrant_client(logger=logger)
        
        # Scroll through all points with this md5
        points = []
        offset = None
        
        f = Filter(must=[FieldCondition(key="md5", match=MatchValue(value=md5))])
        
        while True:
            results, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=f,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            
            for point in results:
                points.append({
                    "id": point.id,
                    "payload": point.payload,
                })
            
            if offset is None:
                break
        
        return points
    except Exception as e:
        logger.warning(f"Failed to get existing points for md5 {md5}: {e}")
        return []


# ---- Main Processing ----
def process_file(
    pdf_entry: Dict[str, Any],
    web_metadata: Dict[str, Dict[str, Any]],
    llm_client: OpenAI,
    logger: logging.Logger,
    is_update: bool = False,
    existing_points: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Process a single text file and return chunking result."""
    
    md5 = pdf_entry.get("md5")
    pdf_name = pdf_entry.get("pdf_name")
    output_path = pdf_entry.get("output_path")
    timestamp = pdf_entry.get("timestamp")
    
    # Read the text file
    text_file = Path(output_path)
    if not text_file.exists():
        # Try to download from S3 if not found locally
        logger.info(f"Markdown file not found locally: {output_path}, trying S3...")
        try:
            from web_scrap.s3_helper import get_s3_client, download_file_from_s3
            
            s3_client = get_s3_client()
            bucket_name = "acct1004215-mospi"
            # Extract filename from output_path
            filename = Path(output_path).name
            s3_key = f"text_data_v1/{filename}"
            
            success = download_file_from_s3(
                s3_client,
                bucket_name,
                s3_key,
                str(text_file)
            )
            
            if success:
                logger.info(f"✓ Downloaded {filename} from S3")
            else:
                raise FileNotFoundError(f"Text file not found locally or in S3: {output_path}")
        except Exception as e:
            raise FileNotFoundError(f"Text file not found: {output_path}. S3 download failed: {e}")
    
    text = text_file.read_text(encoding="utf-8")
    
    # Chunk the document
    chunks = chunk_document_into_pages(text)
    
    # Get metadata - match by pdf_name to filename in web_metadata
    metadata = web_metadata.get(pdf_name)
    logger.info(f"Looking up metadata for pdf_name: {pdf_name} (found: {metadata is not None})")

    if metadata:
        # Prefer file_url over file_links
        file_url = metadata.get("file_url") or (metadata.get("file_links", [None])[0] if metadata.get("file_links") else None)
        # Prefer publish_date over published_date
        publish_date = metadata.get("publish_date") or metadata.get("published_date")
        # Title from metadata if available (may be missing for web-scraped entries)
        title = metadata.get("title")
        file_name = metadata.get("file_name", pdf_name)
    else:
        # No matching metadata - use defaults, then derive fallbacks below
        file_name = pdf_name
        file_url = None
        publish_date = None
        title = None
        logger.warning(f"No metadata found for {pdf_name}, deriving publish_date/title from filename")

        append_jsonl(SKIPPED_FILES_LOG_PATH, {
            "ts": utc_now_iso(),
            "md5": md5,
            "pdf_name": pdf_name,
            "reason": "no_matching_metadata",
            "file_links": None,
        })

    # Fallback: derive publish_date from the filename (e.g. "..._July_2026.pdf")
    if not publish_date:
        derived_date = extract_date_from_filename(pdf_name)
        if derived_date:
            publish_date = derived_date
            logger.info(f"Derived publish_date from filename for {pdf_name}: {publish_date}")
        else:
            logger.warning(f"Could not derive publish_date from filename for {pdf_name}")

    # Fallback: build a clean title (filename -> LLM -> verify against first 1000
    # chars of the document text) when metadata has no title.
    if not title:
        title = build_title(llm_client, pdf_name, text)
        logger.info(f"Derived title for {pdf_name}: {title}")

    logger.info(f"Metadata resolved - title: {title}, file_url: {file_url}, publish_date: {publish_date}")
    
    # For updates, preserve existing metadata and topics
    if is_update and existing_points:
        # Create a map of existing metadata by page_number
        existing_by_page = {}
        for point in existing_points:
            payload = point.get("payload", {})
            page_num = payload.get("page_number")
            if page_num is not None:
                existing_by_page[page_num] = payload
        

        if existing_by_page:
            first_payload = next(iter(existing_by_page.values()))
            file_name = first_payload.get("file_name") or file_name
            file_url = first_payload.get("file_url") or file_url
            publish_date = first_payload.get("publish_date") or publish_date
            title = first_payload.get("title") or title
            # Always use current timestamp for uploaded_at
            uploaded_at = utc_now_iso()
        else:
            uploaded_at = utc_now_iso()
    else:
        uploaded_at = utc_now_iso()
    
    # Topic inference - for new files or updates with empty topics
    if is_update and existing_points:
        # Check if existing topics are empty
        chunk_topics_list: List[List[str]] = []
        
        existing_by_page = {}
        for point in existing_points:
            payload = point.get("payload", {})
            page_num = payload.get("page_number")
            if page_num is not None:
                existing_by_page[page_num] = payload
        
        for c in chunks:
            page_num = c["page_number"]
            existing_payload = existing_by_page.get(page_num, {})
            existing_topics = existing_payload.get("topics", [])
            chunk_topics_list.append(existing_topics)
        
        # Check if all topics are empty
        all_topics_empty = all(len(topics) == 0 for topics in chunk_topics_list)
        
        # If topics are empty and SKIP_TOPICS=False, infer them
        if all_topics_empty and not SKIP_TOPICS:
            logger.info(f"Existing topics are empty, inferring new topics for {pdf_name}")
            # Infer topics from first 5 pages
            doc_snippet = make_first_n_pages_snippet(chunks, DOC_SNIPPET_PAGES, DOC_SNIPPET_CHARS)
            
            doc_topics, doc_err = infer_topics_single_resilient(
                llm_client,
                doc_snippet,
                DOC_TOPIC_TOP_K,
                DOC_SNIPPET_CHARS * DOC_SNIPPET_PAGES,
                DOC_SNIPPET_CHARS_RETRY * DOC_SNIPPET_PAGES,
            )
            
            if doc_err:
                logger.warning(f"Topic inference error: {doc_err}")
            
            # Apply same topics to all chunks
            chunk_topics_list = [doc_topics for _ in chunks]
        # else: preserve existing topics (even if empty when SKIP_TOPICS=True)
    else:
        # Infer topics for new files - use first 5 pages only
        doc_snippet = make_first_n_pages_snippet(chunks, DOC_SNIPPET_PAGES, DOC_SNIPPET_CHARS)
        
        if SKIP_TOPICS:
            doc_topics: List[str] = []
        else:
            # Infer document-level topics from first 5 pages
            doc_topics, doc_err = infer_topics_single_resilient(
                llm_client,
                doc_snippet,
                DOC_TOPIC_TOP_K,
                DOC_SNIPPET_CHARS * DOC_SNIPPET_PAGES,  # Max chars for combined snippet
                DOC_SNIPPET_CHARS_RETRY * DOC_SNIPPET_PAGES,  # Retry with smaller snippet
            )
            
            if doc_err:
                logger.warning(f"Topic inference error: {doc_err}")
        
        # Apply same topics to all chunks (no per-chunk inference)
        chunk_topics_list: List[List[str]] = [doc_topics for _ in chunks]
    
    # Build payloads
    payloads: List[Dict[str, Any]] = []
    for c, ch_topics in zip(chunks, chunk_topics_list):
        # Use the document topics directly (already applied to all chunks)
        topics = ch_topics
        
        payloads.append({
            "page_content": c["page_content"],
            "file_name": file_name,
            "file_url": file_url,
            "title": title,
            "publish_date": publish_date,
            "uploaded_at": uploaded_at,
            "md5": md5,
            "page_number": int(c["page_number"]),
            "page_has_overflow": bool(c.get("page_has_overflow", False)),
            "includes_prev_page": bool(c.get("includes_prev_page", False)),
            "prev_page_chars": int(c.get("prev_page_chars", 0)),
            "next_page_chars": int(c.get("next_page_chars", 0)),
            "topics": topics,
        })
    
    return {
        "md5": md5,
        "pdf_name": pdf_name,
        "file_url": file_url,
        "publish_date": publish_date,
        "chunks": payloads,
        "is_update": is_update,
    }


def _get_gpu_free_gib(logger: Optional[logging.Logger] = None) -> Optional[float]:
    """Return free GPU memory in GiB, or None if CUDA/torch is unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, _total = torch.cuda.mem_get_info()
            return free_bytes / (1024 ** 3)
    except Exception as e:
        if logger:
            logger.debug(f"Could not read GPU memory info: {e}")
    return None


def _empty_cuda_cache():
    """Release cached GPU memory back to the allocator (best effort)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def encode_texts(emb_model, texts: List[str], logger: logging.Logger):

    if not texts:
        return []

    # If GPU is heavily used (shared with vLLM), wait for a minimum amount of free
    # memory before starting, so we only attempt work that can fit.
    free_gib = _get_gpu_free_gib(logger)
    if free_gib is not None:
        waited = 0
        while free_gib < EMBED_MIN_FREE_GIB and waited < 300:
            logger.warning(
                f"Only {free_gib:.2f} GiB free on GPU (< {EMBED_MIN_FREE_GIB} GiB). "
                f"Waiting for memory to free up... ({waited}s)"
            )
            _empty_cuda_cache()
            time.sleep(15)
            waited += 15
            free_gib = _get_gpu_free_gib(logger)

    # Choose an initial batch size based on free memory
    batch = EMBED_BATCH_SIZE
    if free_gib is not None:
        mem_cap = max(1, int(free_gib * EMBED_TEXTS_PER_GIB))
        batch = max(1, min(EMBED_BATCH_SIZE, mem_cap))
        logger.info(f"GPU free ~{free_gib:.2f} GiB -> embedding batch_size={batch}")
    else:
        logger.info(f"GPU memory info unavailable -> embedding batch_size={batch}")

    all_vectors = []
    i = 0
    n = len(texts)
    while i < n:
        sub = texts[i:i + batch]
        try:
            vecs = emb_model.encode(
                sub,
                batch_size=batch,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_vectors.extend(list(vecs))
            i += len(sub)
        except Exception as e:
            # Detect CUDA OOM (torch.cuda.OutOfMemoryError subclasses RuntimeError)
            is_oom = "out of memory" in str(e).lower() or e.__class__.__name__ == "OutOfMemoryError"
            if not is_oom:
                raise
            _empty_cuda_cache()
            if batch <= 1:
                logger.error("CUDA OOM even at batch_size=1; cannot encode this text")
                raise
            batch = max(1, batch // 2)
            logger.warning(f"CUDA OOM detected. Reducing embedding batch_size to {batch} and retrying.")

    return all_vectors


def upsert_to_qdrant(
    result: Dict[str, Any],
    emb_model,
    logger: logging.Logger,
) -> None:
    """Upsert chunks to Qdrant. For updates, deletes old points first."""
    from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
    
    # Get Qdrant client
    qdrant = make_qdrant_client(logger=logger)
    
    md5 = result["md5"]
    payloads = result["chunks"]
    is_update = result.get("is_update", False)
    
    # For updates, delete all existing points with this md5 first
    if is_update:
        try:
            f = Filter(must=[FieldCondition(key="md5", match=MatchValue(value=md5))])
            qdrant.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=f,
            )
            logger.info(f"Deleted existing points for md5={md5}")
        except Exception as e:
            logger.warning(f"Failed to delete existing points for md5={md5}: {e}")
    
    # Generate embeddings (GPU-memory-aware, OOM-resilient batching)
    vectors = encode_texts(emb_model, [p["page_content"] for p in payloads], logger)
    
    # Create points
    points: List[PointStruct] = []
    for payload, vec in zip(payloads, vectors):
        pid = make_point_id(md5, payload["page_number"])
        points.append(PointStruct(id=pid, vector=vec.tolist(), payload=payload))
    
    # Upsert in batches
    for i in range(0, len(points), QDRANT_UPSERT_BATCH):
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i + QDRANT_UPSERT_BATCH])
    
    action = "Updated" if is_update else "Inserted"
    logger.info(f"{action} {len(points)} chunks for {result['pdf_name']}")

def check_md5_in_qdrant(md5_list: List[str], logger: logging.Logger) -> Set[str]:
    """Batch check which md5s exist in Qdrant."""
    if not md5_list:
        return set()
    
    existing: Set[str] = set()
    
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        
        logger.info(f"Checking {len(md5_list)} MD5s in Qdrant...")
        client = make_qdrant_client(logger=logger)
        
        # Process in batches
        batch_size = 1000
        for i in range(0, len(md5_list), batch_size):
            batch = md5_list[i:i + batch_size]
            
            try:
                f = Filter(must=[FieldCondition(key="md5", match=MatchAny(any=batch))])
                
                offset = None
                while True:
                    results, offset = client.scroll(
                        collection_name=QDRANT_COLLECTION,
                        scroll_filter=f,
                        limit=1000,
                        offset=offset,
                        with_payload=["md5"],
                        with_vectors=False,
                    )
                    
                    for point in results:
                        md5 = point.payload.get("md5")
                        if md5:
                            existing.add(md5)
                    
                    if offset is None:
                        break
                        
            except Exception as e:
                logger.warning(f"Batch check failed for batch {i}-{i+len(batch)}: {e}")
        
        logger.info(f"Qdrant check: {len(existing)}/{len(md5_list)} already exist")
    except Exception as e:
        logger.error(f"Failed to check MD5s in Qdrant: {e}")
        logger.exception("Full traceback:")
        raise

        
    except Exception as e:
        logger.warning(f"Failed to batch check Qdrant: {e}")
    
    return existing

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk text files and upload to Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal run
  python chunking_local.py
  
  # Dry run (show what would be processed)
  python chunking_local.py --dry-run
  
  # Skip topic inference for speed
  python chunking_local.py --skip-topics
  
  # Process only first N files
  python chunking_local.py --limit 10
  
  # Dry run with limit
  python chunking_local.py --dry-run --limit 5
  
Environment Variables:
  VLLM_BASE_URL          vLLM server URL (default: http://172.31.3.74:8001/v1)
  QDRANT_URL             Qdrant URL (default: http://localhost:6333)
  EMBED_BATCH_SIZE       Embedding batch size (default: 384)
  SKIP_TOPICS            Skip topic inference (default: 0)
        """
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    parser.add_argument(
        "--skip-topics",
        action="store_true",
        help="Skip topic inference for faster processing (5-10x speedup)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Process only first N files (useful for testing)"
    )
    
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Show statistics about files to process and exit"
    )
    
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Force update all files even if already processed"
    )
    
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logger = setup_logger("main")
    
    # Override SKIP_TOPICS if command line flag is set
    global SKIP_TOPICS
    if args.skip_topics:
        SKIP_TOPICS = True
        logger.info("Topic inference disabled via --skip-topics flag")
    
    # Load metadata
    web_metadata = load_web_file_metadata(logger)
    pdf_entries = load_processed_pdf_to_text(logger)
    processed_chunks = load_processed_chunks(logger)
    
    # Initialize Qdrant and check existing md5s (even in dry-run mode to show accurate counts)
    logger.info(f"Target Qdrant collection: {QDRANT_COLLECTION}")
    ensure_qdrant_collection(logger)
    
    all_md5s = list(pdf_entries.keys())
    existing_in_qdrant = check_md5_in_qdrant(all_md5s, logger)
    
    # Categorize files
    to_update = []  # Files that exist in Qdrant (need update) - only used with --force-update
    to_insert = []  # Files that don't exist in Qdrant (new)
    skipped_in_qdrant = []  # Files already in Qdrant (skipped in production)
    skipped_excluded = []  # FOD / Who's Who / temp files (handled elsewhere)
    skipped_no_url = []  # Files with no file_url in metadata (must not be chunked)
    
    for md5, entry in pdf_entries.items():
        pdf_name = entry.get("pdf_name")

        # Skip if already successfully processed in log (unless force-update)
        if md5 in processed_chunks and not args.force_update:
            continue

        # Exclude FOD / Who's Who / temp files - these are served by
        # whois_cache_manager and must not enter the main collection.
        if is_excluded_from_chunking(pdf_name):
            skipped_excluded.append(entry)
            logger.info(f"Excluding from chunking (FOD/Who's Who/temp): {pdf_name}")
            continue

        # A valid downloaded PDF must have a file_url in metadata. If the URL is
        # missing, the upstream ingestion was incorrect - do not chunk it.
        meta = web_metadata.get(pdf_name)
        meta_url = (meta.get("file_url") if meta else None)
        if not meta_url:
            skipped_no_url.append(entry)
            logger.warning(f"Skipping (no file_url in metadata): {pdf_name}")
            append_jsonl(SKIPPED_FILES_LOG_PATH, {
                "ts": utc_now_iso(),
                "md5": md5,
                "pdf_name": pdf_name,
                "reason": "missing_file_url",
            })
            continue

        # Check if already in Qdrant
        if md5 in existing_in_qdrant:
            if args.force_update:
                # Only update if explicitly requested
                to_update.append(entry)
            else:
                # In production, skip files already in Qdrant
                skipped_in_qdrant.append(entry)
        else:
            # New file, needs to be inserted
            to_insert.append(entry)
    
    # Apply limit if specified
    if args.limit:
        total_before = len(to_insert) + len(to_update)
        all_files = to_update + to_insert
        all_files = all_files[:args.limit]
        
        # Re-categorize after limit
        to_update = [f for f in all_files if f.get("md5") in existing_in_qdrant]
        to_insert = [f for f in all_files if f.get("md5") not in existing_in_qdrant]
        
        logger.info(f"Limit applied: processing {len(all_files)} of {total_before} files")
    
    logger.info(f"Found {len(to_insert)} new files to insert")
    if len(to_update) > 0:
        logger.info(f"Found {len(to_update)} files to update (--force-update enabled)")
    if len(skipped_in_qdrant) > 0:
        logger.info(f"Skipped {len(skipped_in_qdrant)} files already in Qdrant")
    if len(skipped_excluded) > 0:
        logger.info(f"Skipped {len(skipped_excluded)} FOD/Who's Who/temp files (excluded from chunking)")
    if len(skipped_no_url) > 0:
        logger.info(f"Skipped {len(skipped_no_url)} files with no file_url in metadata")
    logger.info(f"Total to process: {len(to_insert) + len(to_update)} (out of {len(pdf_entries)} total)")
    
    # Show statistics and exit if requested
    if args.stats_only:
        print("\n" + "=" * 70)
        print("PROCESSING STATISTICS")
        print("=" * 70)
        
        print(f"\nQdrant Collection: {QDRANT_COLLECTION}")
        print(f"Embedding Model: {HF_EMBEDDING_MODEL}")
        print(f"Topic Inference: {'DISABLED' if SKIP_TOPICS else 'ENABLED'}")
        
        print(f"\nFiles Breakdown:")
        print(f"  Total files in JSONL: {len(pdf_entries)}")
        print(f"  Already processed (in log): {len(processed_chunks)}")
        print(f"  Existing in Qdrant: {len(existing_in_qdrant)}")
        print(f"  Skipped (already in Qdrant): {len(skipped_in_qdrant)}")
        print(f"  Files to process: {len(to_insert) + len(to_update)}")
        print(f"    - New inserts: {len(to_insert)}")
        if len(to_update) > 0:
            print(f"    - Updates (--force-update): {len(to_update)}")
        
        # Estimate time
        total_files = len(to_insert) + len(to_update)
        if total_files > 0:
            if SKIP_TOPICS:
                min_time = total_files * 2 / 60  # 2 sec per file
                max_time = total_files * 5 / 60  # 5 sec per file
                print(f"\nEstimated time (without topics): {min_time:.1f}-{max_time:.1f} minutes")
            else:
                min_time = total_files * 10 / 60  # 10 sec per file
                max_time = total_files * 30 / 60  # 30 sec per file
                print(f"\nEstimated time (with topics): {min_time:.1f}-{max_time:.1f} minutes")
        
        print("=" * 70)
        return
    
    if not to_insert and not to_update:
        logger.info("Nothing to do.")
        return
    
    # Dry run mode: show what would be processed
    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN MODE - No actual processing will occur")
        print("=" * 70)
        
        print(f"\nQdrant Collection: {QDRANT_COLLECTION}")
        print(f"Embedding Model: {HF_EMBEDDING_MODEL}")
        print(f"Topic Inference: {'DISABLED' if SKIP_TOPICS else 'ENABLED'}")
        
        print(f"\nFiles Analysis:")
        print(f"  Total files in JSONL: {len(pdf_entries)}")
        print(f"  Already processed (in log): {len(processed_chunks)}")
        print(f"  Existing in Qdrant: {len(existing_in_qdrant)}")
        print(f"  Skipped (already in Qdrant): {len(skipped_in_qdrant)}")
        print(f"  Files to process: {len(to_insert) + len(to_update)}")
        print(f"    - New inserts: {len(to_insert)}")
        if len(to_update) > 0:
            print(f"    - Updates (--force-update): {len(to_update)}")
        
        # Estimate time
        total_files = len(to_insert) + len(to_update)
        if total_files > 0:
            if SKIP_TOPICS:
                min_time = total_files * 2 / 60
                max_time = total_files * 5 / 60
                print(f"\nEstimated time (without topics): {min_time:.1f}-{max_time:.1f} minutes")
            else:
                min_time = total_files * 10 / 60
                max_time = total_files * 30 / 60
                print(f"\nEstimated time (with topics): {min_time:.1f}-{max_time:.1f} minutes")
        
        print("\nFirst 10 files that would be processed:")
        all_files = to_update + to_insert
        for i, entry in enumerate(all_files[:10], 1):
            md5 = entry.get("md5")
            pdf_name = entry.get("pdf_name")
            is_update = md5 in existing_in_qdrant
            action = "UPDATE" if is_update else "INSERT"
            print(f"  {i}. [{action}] {pdf_name} (md5={md5[:8]}...)")
        
        if len(all_files) > 10:
            print(f"  ... and {len(all_files) - 10} more files")
        
        print("\nTo actually process, run without --dry-run flag")
        print("=" * 70)
        return
    
    # Initialize LLM client and embedding model for actual processing
    llm_client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {HF_EMBEDDING_MODEL}")
    emb_model = SentenceTransformer(HF_EMBEDDING_MODEL, device="cuda")
    
    # Process files (as-is from JSONL, no sorting)
    processed_count = 0
    updated_count = 0
    failed_count = 0
    
    # Combine lists (updates first, then inserts)
    all_to_process = to_update + to_insert
    
    for entry in all_to_process:
        md5 = entry.get("md5")
        pdf_name = entry.get("pdf_name")
        is_update = md5 in existing_in_qdrant
        
        try:
            action = "Updating" if is_update else "Processing"
            logger.info(f"{action} {pdf_name} (md5={md5})")
            
            # Get existing points if updating
            existing_points = None
            if is_update:
                existing_points = get_existing_points_by_md5(md5, logger)
                logger.info(f"Found {len(existing_points)} existing points for {pdf_name}")
            
            # Process file
            result = process_file(
                entry, 
                web_metadata, 
                llm_client, 
                logger,
                is_update=is_update,
                existing_points=existing_points
            )
            
            # Upsert to Qdrant
            upsert_to_qdrant(result, emb_model, logger)

            # Release cached GPU memory between files to limit fragmentation on
            # the shared GPU.
            _empty_cuda_cache()
            
            # Log success
            append_jsonl(PROCESSED_CHUNK_LOGS_PATH, {
                "log_type": "processed",
                "ts": utc_now_iso(),
                "status": "SUCCESS",
                "md5": md5,
                "file_name": pdf_name,
                "file_url": result.get("file_url"),
                "publish_date": result.get("publish_date"),
                "file_chunk_count": len(result["chunks"]),
                "is_update": is_update,
            })
            
            if is_update:
                updated_count += 1
            else:
                processed_count += 1
            
            total_done = processed_count + updated_count
            total_to_do = len(all_to_process)
            logger.info(f"Successfully {action.lower()} {pdf_name} ({total_done}/{total_to_do})")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_name}: {e}", exc_info=True)

            # Release any GPU memory held after a failure (e.g. OOM) before the
            # next file.
            _empty_cuda_cache()
            
            # Log failure
            append_jsonl(PROCESSED_CHUNK_LOGS_PATH, {
                "log_type": "processed",
                "ts": utc_now_iso(),
                "status": "FAILED",
                "md5": md5,
                "file_name": pdf_name,
                "error": str(e),
                "is_update": is_update,
            })
            
            failed_count += 1
    
    logger.info(f"Processing complete: {processed_count} new, {updated_count} updated, {failed_count} failed")
    
    # Upload processed_chunk_logs to S3
    if PROCESSED_CHUNK_LOGS_PATH.exists():
        logger.info("Uploading processed_chunk_logs to S3...")
        try:
            from web_scrap.s3_helper import get_s3_client, upload_file_to_s3
            
            s3_client = get_s3_client()
            bucket_name = "acct1004215-mospi"
            s3_key = "data/mospi_web/processed_chunk_logs"
            
            result = upload_file_to_s3(
                s3_client,
                str(PROCESSED_CHUNK_LOGS_PATH),
                bucket_name,
                s3_key
            )
            
            if result["success"]:
                logger.info(f"✓ Uploaded processed_chunk_logs to S3")
            else:
                logger.error(f"✗ Failed to upload processed_chunk_logs: {result['error']}")
        except Exception as e:
            logger.error(f"Error uploading processed_chunk_logs to S3: {e}")


if __name__ == "__main__":
    main()