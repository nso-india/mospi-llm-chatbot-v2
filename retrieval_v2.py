

import os
import re
import math
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Optional, Tuple, Any
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Import gRPC client function
from qdrant_vector_store import get_qdrant_client_grpc

# Import whois cache manager for whois routing
from whois_cache_manager import get_cached_whois_doc

# Get logger
logger = logging.getLogger("chatbot")

# Configuration
ENABLE_RETRIEVAL_DEBUG = os.getenv("ENABLE_RETRIEVAL_DEBUG", "1").strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_publish_date(v: Any) -> Optional[datetime]:

    if not v:
        return None
    
    try:
        s = str(v).strip()
        if not s or s == "no_date":
            return None
        
        # Try YYYY-MM-DD format
        if len(s) >= 10:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        
        # Try year only
        if len(s) == 4 and s.isdigit():
            return datetime(int(s), 1, 1)
        
        # Try ISO format
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    
    except Exception:
        return None


def _matches_month_year_in_filename(filename: str, target_year: int, target_month: int) -> bool:

    filename_lower = filename.lower()
    
    # Month name (full and abbreviated)
    month_name = datetime(target_year, target_month, 1).strftime("%B").lower()  # "october"
    month_abbr = datetime(target_year, target_month, 1).strftime("%b").lower()  # "oct"
    
    # Pattern 1: Full month name + year (most common)
    # "October_2025", "october-2025", "October 2025"
    if month_name in filename_lower and str(target_year) in filename_lower:
        return True
    
    # Pattern 2: Month abbreviation + year
    # "Oct_2025", "oct-2025", "Oct 2025"
    if month_abbr in filename_lower and str(target_year) in filename_lower:
        return True
    
    # Pattern 3: Numeric month formats (zero-padded)
    month_str = f"{target_month:02d}"  # "10" for October
    numeric_patterns = [
        f"{target_year}-{month_str}",  # "2025-10"
        f"{target_year}_{month_str}",  # "2025_10"
        f"{month_str}-{target_year}",  # "10-2025"
        f"{month_str}_{target_year}",  # "10_2025"
        f"{target_year}{month_str}",   # "202510"
    ]
    
    for pattern in numeric_patterns:
        if pattern in filename_lower:
            return True
    
    # Pattern 4: Numeric month without zero-padding (for months 1-9)
    if target_month < 10:
        month_str_no_pad = str(target_month)  # "1" for January
        no_pad_patterns = [
            f"{target_year}-{month_str_no_pad}-",  # "2025-1-"
            f"{target_year}_{month_str_no_pad}_",  # "2025_1_"
            f"-{month_str_no_pad}-{target_year}",  # "-1-2025"
            f"_{month_str_no_pad}_{target_year}",  # "_1_2025"
        ]
        
        for pattern in no_pad_patterns:
            if pattern in filename_lower:
                return True
    
    return False


def _convert_cadence_to_target_months(cadence_hints: Dict) -> List[Tuple[int, int]]:

    date_ranges = cadence_hints.get("date_ranges", [])
    target_months = []
    
    for range_info in date_ranges:
        data_year = range_info.get("data_year")
        data_month = range_info.get("data_month")
        
        if data_year and data_month:
            target_months.append((data_year, data_month))
        
        # For quarterly data, add all 3 months of the quarter
        quarter = range_info.get("quarter")
        fy_year = range_info.get("fy_year")
        if quarter and fy_year:
            quarter_months = {
                1: [4, 5, 6],
                2: [7, 8, 9],
                3: [10, 11, 12],
                4: [1, 2, 3]
            }
            months = quarter_months.get(quarter, [])
            for m in months:
                if m >= 4:
                    target_months.append((fy_year, m))
                else:
                    target_months.append((fy_year + 1, m))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_months = []
    for ym in target_months:
        if ym not in seen:
            seen.add(ym)
            unique_months.append(ym)
    
    return unique_months


def _extract_time_preference(query: str) -> Dict[str, Any]:
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    
    q = query.lower()
    now = datetime.now()
    
    # Check for "latest" keywords
    if any(kw in q for kw in ["latest", "current", "recent", "newest", "most recent"]):
        # Latest = current month + 5 months back (6 months total)
        target_months = []
        for i in range(6):  # Current + 5 back = 6 months
            date = now - relativedelta(months=i)
            target_months.append((date.year, date.month))
        
        return {
            "mode": "latest",
            "target_months": target_months
        }
    
    # Check for month + year (e.g., "october 2025", "jan 2024")
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 
        'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    for month_key, month_num in month_names.items():
        # Pattern: "october 2025" or "oct 2025"
        month_year_pattern = rf'\b{month_key}\s+(20\d{{2}})\b'
        match = re.search(month_year_pattern, q)
        if match:
            year = int(match.group(1))
            # Specified month + 2 months back + 1 month forward
            base_date = datetime(year, month_num, 1)
            target_months = []
            
            # 2 months before
            for i in range(2, 0, -1):
                date = base_date - relativedelta(months=i)
                target_months.append((date.year, date.month))
            
            # Specified month
            target_months.append((year, month_num))
            
            # 1 month after
            date = base_date + relativedelta(months=1)
            target_months.append((date.year, date.month))
            
            return {
                "mode": "month_year",
                "month": month_num,
                "year": year,
                "target_months": target_months
            }
    
    # Check for quarter patterns (Q1 2025, Q2 FY 2024-25, etc.)
    quarter_match = re.search(r'\b(q[1-4])\s*(?:fy\s*)?(\d{4})(?:-\d{2,4})?\b', q)
    if quarter_match:
        quarter = quarter_match.group(1).upper()
        year = int(quarter_match.group(2))
        
        # Map quarter to months
        quarter_months = {
            'Q1': [1, 2, 3],      # Jan, Feb, Mar
            'Q2': [4, 5, 6],      # Apr, May, Jun
            'Q3': [7, 8, 9],      # Jul, Aug, Sep
            'Q4': [10, 11, 12]    # Oct, Nov, Dec
        }
        
        months = quarter_months.get(quarter, [])
        target_months = [(year, m) for m in months]
        
        return {
            "mode": "quarter",
            "quarter": quarter,
            "year": year,
            "target_months": target_months
        }
    
    # Check for fiscal year quarter (FY 2024-25 Q1)
    fy_quarter_match = re.search(r'\bfy\s*(\d{4})-\d{2,4}\s+(q[1-4])\b', q)
    if fy_quarter_match:
        year = int(fy_quarter_match.group(1))
        quarter = fy_quarter_match.group(2).upper()
        
        quarter_months = {
            'Q1': [4, 5, 6],      # Apr, May, Jun (FY starts in April)
            'Q2': [7, 8, 9],      # Jul, Aug, Sep
            'Q3': [10, 11, 12],   # Oct, Nov, Dec
            'Q4': [1, 2, 3]       # Jan, Feb, Mar (next year)
        }
        
        months = quarter_months.get(quarter, [])
        target_months = []
        for m in months:
            if m >= 4:  # Apr-Dec of start year
                target_months.append((year, m))
            else:  # Jan-Mar of next year
                target_months.append((year + 1, m))
        
        return {
            "mode": "quarter",
            "quarter": quarter,
            "year": year,
            "target_months": target_months
        }
    
    # Check for specific year (YYYY)
    year_match = re.search(r'\b(20\d{2})\b', query)
    if year_match:
        year = int(year_match.group(1))
        
  
        target_months = []
        
        if year < now.year:
            # Year has ended - use December of that year
            base_date = datetime(year, 12, 1)
            # 2 months before December
            for i in range(2, 0, -1):
                date = base_date - relativedelta(months=i)
                target_months.append((date.year, date.month))
            # December
            target_months.append((year, 12))
            # 1 month after (January next year)
            target_months.append((year + 1, 1))
            
        elif year == now.year:
            # Current year - use current month
            # 3 months back
            for i in range(3, 0, -1):
                date = now - relativedelta(months=i)
                target_months.append((date.year, date.month))
            # Current month
            target_months.append((now.year, now.month))
            # 1 month forward (if not December)
            if now.month < 12:
                target_months.append((now.year, now.month + 1))
            else:
                target_months.append((now.year + 1, 1))
                
        else:  # year > now.year (future year)
            # Future year - use January of that year
            base_date = datetime(year, 1, 1)
            # 3 months back (from previous year)
            for i in range(3, 0, -1):
                date = base_date - relativedelta(months=i)
                target_months.append((date.year, date.month))
            # January
            target_months.append((year, 1))
            # 1 month forward
            target_months.append((year, 2))
        
        return {
            "mode": "year",
            "year": year,
            "target_months": target_months
        }
    
    # Check for year range (YYYY-YYYY or YYYY to YYYY)
    range_match = re.search(r'\b(20\d{2})\s*(?:-|to)\s*(20\d{2})\b', query)
    if range_match:
        start_year = int(range_match.group(1))
        end_year = int(range_match.group(2))
        
        # Generate all months in the range
        target_months = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                target_months.append((year, month))
        
        return {
            "mode": "range",
            "start_year": start_year,
            "end_year": end_year,
            "target_months": target_months
        }
    
    # No time period specified - use current month + 5 months back (6 months total)
    target_months = []
    for i in range(6):  # Current + 5 back = 6 months
        date = now - relativedelta(months=i)
        target_months.append((date.year, date.month))
    
    return {
        "mode": "none",
        "target_months": target_months
    }


def _boost_by_publish_date(
    docs: List[Document],
    time_pref: Dict,
    top_k: int
) -> List[Document]:
 
    scored_docs = []
    target_months = time_pref.get("target_months", [])
    
    for doc in docs:
        vector_score = doc.metadata.get("_vector_score", 0.0)
        publish_date_str = doc.metadata.get("publish_date", "")
        file_name = doc.metadata.get("file_name", "").lower()
        
        # Parse publish date
        publish_date = _parse_publish_date(publish_date_str)
        
        # Calculate time score based on target_months
        time_score = 0.0
        
        if target_months and publish_date:
            # Check if document's (year, month) is in target_months list
            doc_year_month = (publish_date.year, publish_date.month)
            
            if doc_year_month in target_months:
                # Give higher score to more recent months in the list (first = most recent)
                try:
                    position = target_months.index(doc_year_month)
                    # Score: 1.0 for first month, 0.9 for second, 0.8 for third, etc.
                    time_score = max(0.5, 1.0 - (position * 0.1))
                except ValueError:
                    time_score = 0.0
            
            # Also check filename for year/month patterns as fallback
            if time_score == 0.0:
                for target_year, target_month in target_months[:6]:  # Check top 3 months
                    month_name = datetime(target_year, target_month, 1).strftime("%B").lower()
                    if month_name in file_name and str(target_year) in file_name:
                        time_score = 0.7  # Lower score for filename match
                        break
        
        elif time_pref["mode"] == "latest" and publish_date:
            # Fallback: boost recent documents (exponential decay with 1-year half-life)
            days_old = (datetime.now() - publish_date).days
            time_score = math.exp(-days_old / 365.0)
        
        # Combined score: 70% vector similarity + 30% time relevance
        combined_score = 0.7 * vector_score + 0.3 * time_score
        
        scored_docs.append((doc, combined_score, vector_score, time_score))
    
    # Sort by combined score
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Log top docs for debugging
    if ENABLE_RETRIEVAL_DEBUG:
        logger.info(f"[TIME_BOOST] Target months: {target_months[:5] if target_months else 'none'}")
        logger.info(f"[TIME_BOOST] Top {min(5, len(scored_docs))} after boosting:")
        for i, (doc, combined, vec, time) in enumerate(scored_docs[:5]):
            meta = doc.metadata
            logger.info(f"[TIME_BOOST]   {i+1}. {meta.get('file_name', 'unknown')} "
                       f"(combined={combined:.3f}, vec={vec:.3f}, time={time:.3f}, "
                       f"date={meta.get('publish_date', 'no_date')})")
    
    return [doc for doc, _, _, _ in scored_docs[:top_k]]


def _boost_by_publish_date_with_rerank(
    docs: List[Document],
    time_pref: Dict,
    top_k: int,
    strict_year_filter: bool = False  # NEW: Strict year filtering
) -> List[Document]:
    scored_docs = []
    target_months = time_pref.get("target_months", [])
    
    # Extract target years for strict filtering
    target_years = set()
    if target_months:
        target_years = {year for year, month in target_months}
    
    for doc in docs:
        # Use rerank score instead of vector score
        rerank_score = doc.metadata.get("_rerank_score", 0.0)
        publish_date_str = doc.metadata.get("publish_date", "")
        file_name = doc.metadata.get("file_name", "").lower()
        
        # Parse publish date
        publish_date = _parse_publish_date(publish_date_str)
        
        # Strict year filter: Skip docs not matching target year OR with no_date
        if strict_year_filter and target_years:
            if not publish_date or publish_date_str == "no_date":
                # Reject documents with no_date when strict filtering is enabled
                logger.debug(f"[TIME_BOOST_RERANK] Skipping doc (no_date): {file_name[:50]}")
                continue
            elif publish_date.year not in target_years:
                logger.debug(f"[TIME_BOOST_RERANK] Skipping doc (wrong year): {file_name[:50]} "
                           f"(doc_year={publish_date.year}, target_years={target_years})")
                continue
        
        # Calculate time score based on target_months
        time_score = 0.0
        
        if target_months and publish_date:
            # Check if document's (year, month) is in target_months list
            doc_year_month = (publish_date.year, publish_date.month)
            
            if doc_year_month in target_months:
                # Give higher score to more recent months in the list (first = most recent)
                try:
                    position = target_months.index(doc_year_month)
                    # Score: 1.0 for first month, 0.9 for second, 0.8 for third, etc.
                    time_score = max(0.5, 1.0 - (position * 0.1))
                except ValueError:
                    time_score = 0.0
            
            # Also check filename for year/month patterns as fallback
            if time_score == 0.0:
                for target_year, target_month in target_months[:6]:  # Check top 3 months
                    month_name = datetime(target_year, target_month, 1).strftime("%B").lower()
                    if month_name in file_name and str(target_year) in file_name:
                        time_score = 0.7  # Lower score for filename match
                        break
        
        elif time_pref["mode"] == "latest" and publish_date:
            # Fallback: boost recent documents (exponential decay with 1-year half-life)
            days_old = (datetime.now() - publish_date).days
            time_score = math.exp(-days_old / 365.0)
        
        # Combined score: 70% rerank quality + 30% time relevance
        combined_score = 0.7 * rerank_score + 0.3 * time_score
        
        scored_docs.append((doc, combined_score, rerank_score, time_score))
    
    # Sort by combined score
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Log top docs for debugging
    if ENABLE_RETRIEVAL_DEBUG:
        logger.info(f"[TIME_BOOST_RERANK] Target months: {target_months[:5] if target_months else 'none'}")
        logger.info(f"[TIME_BOOST_RERANK] Strict year filter: {strict_year_filter}, target_years: {target_years if strict_year_filter else 'disabled'}")
        logger.info(f"[TIME_BOOST_RERANK] Top {min(5, len(scored_docs))} after boosting:")
        for i, (doc, combined, rerank, time) in enumerate(scored_docs[:5]):
            meta = doc.metadata
            logger.info(f"[TIME_BOOST_RERANK]   {i+1}. {meta.get('file_name', 'unknown')} "
                       f"(combined={combined:.3f}, rerank={rerank:.3f}, time={time:.3f}, "
                       f"date={meta.get('publish_date', 'no_date')})")
    
    return [doc for doc, _, _, _ in scored_docs[:top_k]]


def _detect_document_type_from_filename(filename: str) -> Optional[str]:
    filename_lower = filename.lower()
    
    # Define patterns in priority order (check more specific patterns first)
    patterns = [
        ('promotion_order', [r'promotion.*order', r'promo.*order']),
        ('transfer_order', [r'transfer.*order']),
        ('posting_order', [r'posting.*order']),
        ('appointment_order', [r'appointment.*order']),
        ('deputation_order', [r'deputation.*order']),
        ('office_order', [r'office.*order']),
        ('vacancy_position', [r'vacancy.*position', r'vacancy.*pos']),
        ('seniority_list', [r'seniority.*list', r'draft.*seniority']),
        ('civil_list', [r'civil.*list']),
        ('gazette_notification', [r'gazette.*notification', r'gazette.*notif']),
        ('circular', [r'circular']),
        ('notification', [r'notification', r'notif']),
        ('order', [r'\border\b']),  # Generic order (low priority)
    ]
    
    for doc_type, regex_patterns in patterns:
        for pattern in regex_patterns:
            if re.search(pattern, filename_lower):
                return doc_type
    
    return None


def _boost_by_document_type(query: str, docs: List[Document]) -> List[Document]:

    from query_routing import _extract_document_type_keywords
    
    # Extract document type keywords from query
    doc_type_keyword = _extract_document_type_keywords(query)
    
    if not doc_type_keyword:
        # No document type in query - no boosting needed
        return docs
    
    logger.info(f"[DOC_TYPE_BOOST] Detected document type keyword in query: '{doc_type_keyword}'")
    
    # Score each document based on filename match
    scored_docs = []
    for doc in docs:
        filename = doc.metadata.get("file_name", "").lower()
        
        # Detect document type from filename
        detected_type = _detect_document_type_from_filename(filename)
        
        # Calculate boost score (0.0 = no match, 1.0 = perfect match)
        boost_score = 0.0
        
        if detected_type:
            # Check if detected type matches query keyword
            if doc_type_keyword in detected_type or detected_type in doc_type_keyword:
                boost_score = 1.0  # Perfect match
            elif any(word in detected_type for word in doc_type_keyword.split()):
                boost_score = 0.5  # Partial match
        
        # Also check direct keyword presence in filename (fallback)
        if boost_score == 0.0:
            keywords = doc_type_keyword.split()
            matches = sum(1 for kw in keywords if kw in filename)
            if matches > 0:
                boost_score = min(matches * 0.3, 0.9)  # Up to 0.9 for keyword matches
        
        # Get existing rerank score (or default to 0)
        rerank_score = doc.metadata.get("_rerank_score", 0.0)
        
        combined_score = 0.6 * rerank_score + 0.4 * boost_score
        
        doc.metadata["_doc_type_boost"] = boost_score
        doc.metadata["_doc_type"] = detected_type or "unknown"
        doc.metadata["_combined_score"] = combined_score
        
        scored_docs.append((doc, combined_score))
    
    # Sort by combined score (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Log top 5 for debugging
    logger.info(f"[DOC_TYPE_BOOST] Top 5 after document type boosting:")
    for i, (doc, score) in enumerate(scored_docs[:5], 1):
        filename = doc.metadata.get("file_name", "unknown")[:80]
        doc_type = doc.metadata.get("_doc_type", "unknown")
        boost = doc.metadata.get("_doc_type_boost", 0.0)
        rerank = doc.metadata.get("_rerank_score", 0.0)
        logger.info(f"[DOC_TYPE_BOOST]   {i}. {filename} (type={doc_type}, boost={boost:.3f}, rerank={rerank:.3f}, combined={score:.3f})")
    
    return [doc for doc, _ in scored_docs]


def vector_search_simple(
    query: str,
    vectordb,
    rerank_function,
    top_k: int = 5,
    time_pref: Optional[Dict] = None
) -> List[Document]:
    try:
        # Generate query embedding
        query_vector = vectordb.embeddings.embed_query(query)
        
        # Get gRPC client for better performance
        try:
            qdrant_client = get_qdrant_client_grpc()
            logger.debug(f"[VECTOR_SEARCH_V2] Using gRPC client")
        except Exception as grpc_error:
            logger.warning(f"[VECTOR_SEARCH_V2] gRPC client failed, using HTTP: {grpc_error}")
            qdrant_client = vectordb.client
        
        collection_name = vectordb.collection_name
        
        # Determine search limit (get more if we need to rerank + boost by time)
        needs_time_boost = time_pref and time_pref.get("mode") != "none"
        search_limit = 100 if needs_time_boost else top_k
        
        # Search Qdrant using query_points API
        search_result = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=search_limit,
            with_payload=True,
        )
        
        # Convert to Documents
        docs = []
        for point in search_result.points:
            payload = point.payload or {}
            content = payload.get("page_content", "")
            metadata = {k: v for k, v in payload.items() if k != "page_content"}
            
            doc = Document(page_content=content, metadata=metadata)
            doc.metadata["_vector_score"] = point.score
            docs.append(doc)
        
        logger.info(f"[VECTOR_SEARCH_V2] Retrieved {len(docs)} candidates (limit={search_limit})")
        
        # Apply reranking + time boosting if needed
        if needs_time_boost and len(docs) > top_k:
            # STEP 1: Rerank with BGE to get top 20 quality candidates (~500ms)
            reranked_with_scores = rerank_function(query, docs, top_k=20)
            reranked_docs = [doc for doc, score in reranked_with_scores]
            
            # Store rerank scores in metadata for later use
            for doc, score in reranked_with_scores:
                doc.metadata["_rerank_score"] = score
            
            logger.info(f"[VECTOR_SEARCH_V2] After first rerank: {len(reranked_docs)} docs")
            
            # STEP 2: Time boost on reranked docs (70% rerank + 30% time)
            # Use strict year filter to reject docs from wrong years
            docs = _boost_by_publish_date_with_rerank(
                reranked_docs, 
                time_pref, 
                top_k,
                strict_year_filter=True  # Enable strict year filtering
            )
            
            # If strict filter returned too few docs, retry without strict filter
            # BUT: Only if the query doesn't have an explicit year (mode != "month_year" and mode != "year")
            # If user asked for specific year, don't fall back to old data
            should_fallback = (
                len(docs) < 5 and 
                len(reranked_docs) >= 5 and
                time_pref.get("mode") not in ["month_year", "year", "quarter"]  # Don't fallback for explicit time queries
            )
            
            if should_fallback:
                logger.warning(f"[VECTOR_SEARCH_V2] Strict year filter returned only {len(docs)} docs, "
                             f"retrying without strict filter to get more results")
                docs = _boost_by_publish_date_with_rerank(
                    reranked_docs, 
                    time_pref, 
                    top_k,
                    strict_year_filter=False  # Disable strict filtering
                )
            elif len(docs) < 5:
                logger.info(f"[VECTOR_SEARCH_V2] Strict year filter returned only {len(docs)} docs, "
                           f"but query has explicit year (mode={time_pref.get('mode')}), "
                           f"NOT falling back to old data")
            
            logger.info(f"[VECTOR_SEARCH_V2] After time boosting: {len(docs)} docs")
            
            # STEP 3: Apply document type boosting if query contains document type keywords
            docs = _boost_by_document_type(query, docs)
        else:
            docs = docs[:top_k]
        
        return docs
        
    except Exception as e:
        logger.error(f"[VECTOR_SEARCH_V2] Failed: {e}", exc_info=True)
        return []


# Match the normal service-order format, for example T-23/2026 or M-07/2026.
SERVICE_ORDER_STANDARD_PATTERN = re.compile(
    # Capture the letter prefix, order number, and four-digit year as separate groups.
    r"\b([A-Z]{1,4})\s*[-–—]?\s*(\d{1,4})\s*/\s*(20\d{2})\b",
    # Ignore letter case so that t-23/2026 and T-23/2026 produce the same key.
    re.IGNORECASE,
)

# Match compact source text such as T-232026, where the slash before the year is missing.
SERVICE_ORDER_COMPACT_PATTERN = re.compile(
    # The final six digits are split into the order number followed by a year beginning with 20.
    r"\b([A-Z]{1,4})\s*[-–—]\s*(\d{1,3})(20\d{2})\b",
    # Ignore letter case for the same reason as the standard pattern above.
    re.IGNORECASE,
)

# Match day-first dates such as 12.06.2026, 12-06-2026, and malformed 30.062026.
SERVICE_ORDER_DMY_DATE_PATTERN = re.compile(
    # Make the separator before the year optional to support dates copied from inconsistent titles.
    r"\b([0-3]?\d)[./-]([01]?\d)[./-]?(20\d{2})\b"
)

# Match ISO dates such as 2026-06-12, which are commonly stored in publish_date metadata.
SERVICE_ORDER_ISO_DATE_PATTERN = re.compile(
    # Capture year, month, and day separately so they can be normalized to one common format.
    r"\b(20\d{2})[-/.]([01]?\d)[-/.]([0-3]?\d)\b"
)

# Map user-facing order phrases to stable internal document-type names.
SERVICE_ORDER_TYPE_KEYWORDS = {
    # Office-order phrases are kept separate from transfer and promotion orders.
    "office_order": ("office order",),
    # Transfer queries may use either the full phrase or a transfer-specific title.
    "transfer_order": ("transfer order",),
    # Promotion queries include the common JTS and HAG title variants found in ISS files.
    "promotion_order": ("promotion order", "jts promotion", "hag promotion"),
    # Relieving titles have multiple spellings in the source website.
    "relieving_order": ("relieving order", "relieve order", "stand relieve order"),
    # Vacancy documents may not contain the word order, so vacancy-specific phrases are included.
    "vacancy": ("vacancy position", "tentative vacancy"),
    # The remaining entries cover the service-order categories currently present in MoSPI data.
    "posting_order": ("posting order",),
    "appointment_order": ("appointment order",),
    "confirmation_order": ("confirmation order",),
    "deputation_order": ("deputation order",),
    "encadrement_order": ("encadrement order", "encadrement"),
    # Generic wording supports queries such as "Order No. T-232026" and "SSS order dated ...".
    "generic_order": ("order no.", "order no", "order dated", "iss order", "sss order"),
}

# These filename terms create a reasonably narrow Qdrant scroll filter before exact client-side checks.
SERVICE_ORDER_FILENAME_TERMS = (
    # Most ISS and SSS service documents contain at least one of these words in their filename.
    "order",
    "promotion",
    "transfer",
    "vacancy",
    "posting",
    "appointment",
    "confirmation",
    "deputation",
    "relieve",
    "encadrement",
)

# Match words that clearly ask for the newest available service-order documents.
SERVICE_ORDER_LATEST_PATTERN = re.compile(
    # Keep this list specific to recency intent so ordinary ISS/SSS order searches are unchanged.
    r"\b(?:latest|newest|most\s+recent|recent|current|up[-\s]?to[-\s]?date|updated)\b",
    # Accept any letter case used by the frontend or user.
    re.IGNORECASE,
)


def _extract_service_order_keys(text: str) -> set:
    normalized_keys = set()
    # Convert None or another falsey input into an empty string before running regular expressions.
    searchable_text = text or ""

    # Read standard values such as T-23/2026 and T- 23/2026.
    for prefix, number, year in SERVICE_ORDER_STANDARD_PATTERN.findall(searchable_text):
        # Convert the numeric portion through int so 07 and 7 normalize to the same order key.
        normalized_keys.add(f"{prefix.upper()}:{int(number)}:{year}")

    # Read compact values such as T-232026, which means T-23/2026.
    for prefix, number, year in SERVICE_ORDER_COMPACT_PATTERN.findall(searchable_text):
        # Store compact and standard formats in exactly the same normalized representation.
        normalized_keys.add(f"{prefix.upper()}:{int(number)}:{year}")

    # Return every unique normalized key found in the supplied text.
    return normalized_keys


def _extract_service_order_dates(text: str) -> set:
    """Return normalized YYYY-MM-DD dates from title, filename, content, or metadata text."""
    # Start with an empty set so the same date found in several fields appears only once.
    normalized_dates = set()
    # Convert a missing input into an empty string before applying date expressions.
    searchable_text = text or ""

    # Read day-first source dates such as 12.06.2026 and 30.062026.
    for day, month, year in SERVICE_ORDER_DMY_DATE_PATTERN.findall(searchable_text):
        # Normalize all supported day-first formats to ISO YYYY-MM-DD.
        normalized_dates.add(f"{year}-{int(month):02d}-{int(day):02d}")

    # Read ISO metadata dates such as 2026-06-12.
    for year, month, day in SERVICE_ORDER_ISO_DATE_PATTERN.findall(searchable_text):
        # Preserve the same ISO representation used for day-first source dates.
        normalized_dates.add(f"{year}-{int(month):02d}-{int(day):02d}")

    # Return every unique normalized date found in the supplied text.
    return normalized_dates


def _detect_service_order_type(text: str) -> Optional[str]:
    """Detect the requested order category without deciding whether it belongs to ISS or SSS."""
    # Convert filename separators to spaces so transfer_order.pdf matches "transfer order".
    searchable_text = re.sub(r"[_-]+", " ", (text or "").lower())

    # Test each stable document type against its known user-facing phrases.
    for document_type, keywords in SERVICE_ORDER_TYPE_KEYWORDS.items():
        # Prefer longer phrases first so a more specific description wins when phrases overlap.
        for keyword in sorted(keywords, key=len, reverse=True):
            # Return the first stable document type whose phrase appears in the text.
            if keyword in searchable_text:
                return document_type

    # Return None when the text does not contain a recognized service-order phrase.
    return None


def _is_latest_service_order_query(query: str) -> bool:
    """Return True only when the user clearly requests recent or latest documents."""
    # Use the dedicated pattern instead of a broad substring check to avoid accidental activation.
    return bool(SERVICE_ORDER_LATEST_PATTERN.search(query or ""))


def _latest_service_order_file_limit(query: str, top_k: int) -> int:
    """Return one PDF for a singular latest request and up to top_k PDFs for a plural request."""
    # Treat explicit plural words as a request for a list of recent documents.
    requests_multiple_files = bool(re.search(r"\b(?:orders|documents|files)\b", query or "", re.IGNORECASE))
    # Preserve the caller's configured limit for plural requests, otherwise return only the newest PDF.
    return max(1, top_k) if requests_multiple_files else 1


def _detect_allowed_service_prefixes(query: str) -> set:
    """Restrict an explicit ISS/SSS query, or allow both services when the query is ambiguous."""
    # Normalize case once for full-form phrase checks.
    query_lower = (query or "").lower()
    # Detect an explicit ISS acronym or the full Indian Statistical Service name.
    explicitly_requests_iss = bool(re.search(r"\biss\b", query_lower)) or "indian statistical service" in query_lower
    # Detect an explicit SSS acronym or the full Subordinate Statistical Service name.
    explicitly_requests_sss = bool(re.search(r"\bsss\b", query_lower)) or "subordinate statistical service" in query_lower

    # Restrict the candidates to ISS when only ISS was explicitly requested.
    if explicitly_requests_iss and not explicitly_requests_sss:
        return {"iss"}
    # Restrict the candidates to SSS when only SSS was explicitly requested.
    if explicitly_requests_sss and not explicitly_requests_iss:
        return {"sss"}
    # Search both services when neither service, or both services, were mentioned.
    return {"iss", "sss"}


def _get_service_order_primary_text(doc: Document) -> str:
    """Combine fields that can contain the real order number and real order date."""
    # Read the API title because it often contains the order number even when the filename does not.
    document_title = str(doc.metadata.get("title", ""))
    # Read the unique stored filename because it identifies ISS/SSS and often contains the order date.
    document_file_name = str(doc.metadata.get("file_name", ""))
    # Read extracted PDF text because the exact order number may exist only inside the document body.
    document_content = str(doc.page_content or "")
    # Join the three primary fields without adding publish_date, which may differ from the order date.
    return " ".join([document_title, document_file_name, document_content])


def _get_service_order_title_filename_text(doc: Document) -> str:
    """Combine the two reliable identity fields used first when choosing a latest order date."""
    # Titles normally contain the official order date supplied by the MoSPI page.
    document_title = str(doc.metadata.get("title", ""))
    # Filenames commonly contain the order date even when publish_date contains an ingestion timestamp.
    document_file_name = str(doc.metadata.get("file_name", ""))
    # Keep page content out of this value because it may mention unrelated historical dates.
    return " ".join([document_title, document_file_name])


def _get_service_order_publish_date(doc: Document) -> str:
    """Return publish_date separately so it is used only as a secondary date fallback."""
    # Convert the metadata value to text so the common date extractor can process it safely.
    return str(doc.metadata.get("publish_date", ""))


def _get_latest_service_order_date(file_record: Dict[str, Any]) -> Tuple[str, str]:
    """Choose one sortable date per PDF using identity, publish date, then content fallback."""
    # Prefer dates from the title or filename because they normally describe the order itself.
    if file_record["title_filename_dates"]:
        return max(file_record["title_filename_dates"]), "title_or_filename"
    # Use Qdrant publish_date only when the document identity does not contain a usable date.
    if file_record["publish_dates"]:
        return max(file_record["publish_dates"]), "publish_date"
    # Use page-content dates last because an order body may reference several older orders or rules.
    if file_record["content_dates"]:
        return max(file_record["content_dates"]), "page_content_fallback"
    # Keep undated PDFs after every dated PDF during descending sorting.
    return "", "undated"


def _service_order_page_sort_key(doc: Document) -> Tuple[int, str]:
    """Sort selected chunks by page number while keeping malformed page metadata stable."""
    # Read the page number from Qdrant metadata and fall back to a large value when it is missing.
    raw_page_number = doc.metadata.get("page_number", 999999)
    # Convert numeric page values to integers so page 2 correctly sorts before page 10.
    try:
        page_number = int(raw_page_number)
    # Keep chunks with missing or malformed page values after normal numbered pages.
    except (TypeError, ValueError):
        page_number = 999999
    # Use the filename as a stable secondary key when multiple services or PDFs match.
    return page_number, str(doc.metadata.get("file_name", ""))


def _search_exact_service_order_documents(
    query: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    top_k: int = 5,
) -> List[Document]:
    """Search ISS/SSS files by structured identifiers or latest document type and date."""
    # Extract exact order identifiers such as T:23:2026 from the user query.
    query_order_keys = _extract_service_order_keys(query)
    # Extract an optional requested date such as 2026-06-12 from the user query.
    query_dates = _extract_service_order_dates(query)
    # Detect the requested order category for date-only queries and additional validation.
    query_document_type = _detect_service_order_type(query)
    # Detect a latest request separately so exact order-number behavior remains unchanged.
    latest_requested = _is_latest_service_order_query(query)
    # Latest mode requires a recognized type such as transfer, promotion, office, or vacancy.
    latest_type_request = latest_requested and bool(query_document_type)
    # Keep latest sorting off when the same query supplies an exact order key or explicit date.
    latest_mode = latest_type_request and not query_order_keys and not query_dates

    # Activate only for an exact key, a type plus date, or a type plus explicit latest intent.
    if not query_order_keys and not (query_document_type and query_dates) and not latest_mode:
        return []

    # Determine whether the user explicitly requested ISS, SSS, or left the service unspecified.
    allowed_prefixes = _detect_allowed_service_prefixes(query)
    # Build Qdrant filename conditions that narrow the scroll to likely service-order documents.
    filename_conditions = [
        # Match each known filename term independently and combine them with Qdrant should/OR logic.
        qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=term))
        # Create one Qdrant condition for every supported order-related filename term.
        for term in SERVICE_ORDER_FILENAME_TERMS
    ]
    # Wrap the filename conditions in one reusable Qdrant filter.
    service_order_filter = qmodels.Filter(should=filename_conditions)
    # Group all retrieved chunks by filename so every chunk from the chosen PDF can be returned together.
    documents_by_file: Dict[str, Dict[str, Any]] = {}
    # Start Qdrant scrolling from the beginning of the filtered result set.
    next_page_offset = None

    # Continue scrolling until Qdrant reports that no additional filtered records remain.
    while True:
        try:
            # Fetch one bounded page of likely order documents from the existing collection.
            points, next_page_offset = qdrant_client.scroll(
                # Search the same collection already used by the current retrieval pipeline.
                collection_name=collection_name,
                # Restrict the server-side scan to likely order-related filenames.
                scroll_filter=service_order_filter,
                # Process a moderate page size to avoid a large response in one request.
                limit=256,
                # Include payload fields because exact matching needs title, filename, content, and dates.
                with_payload=True,
                # Continue from the previous Qdrant page when an offset is available.
                offset=next_page_offset,
            )
        # Preserve the existing vector-search fallback if the specialized Qdrant scroll fails.
        except Exception as error:
            logger.error(f"[SERVICE_ORDER_SEARCH] Qdrant scroll failed: {error}", exc_info=True)
            return []

        # Convert each Qdrant point into the same LangChain Document shape used by the rest of retrieval_v2.
        for point in points:
            # Use an empty dictionary when a Qdrant record unexpectedly has no payload.
            payload = point.payload or {}
            # Read extracted PDF content from the standard payload field.
            page_content = str(payload.get("page_content", ""))
            # Keep all remaining payload fields as document metadata.
            metadata = {key: value for key, value in payload.items() if key != "page_content"}
            # Normalize the filename for prefix checks and stable grouping.
            file_name = str(metadata.get("file_name", "")).lower()

            # Skip records that are not clearly stored under an ISS or SSS filename prefix.
            if not any(file_name.startswith(prefix) for prefix in allowed_prefixes):
                continue

            # Create the standard Document object expected by reranking and source formatting.
            document = Document(page_content=page_content, metadata=metadata)
            # Create one aggregation record the first time a filename is encountered.
            file_record = documents_by_file.setdefault(
                file_name,
                {
                    # Preserve all chunks so the selected PDF can provide enough answer context.
                    "documents": [],
                    # Aggregate order keys across every chunk of the same PDF.
                    "order_keys": set(),
                    # Aggregate actual order dates from title, filename, and content.
                    "primary_dates": set(),
                    # Keep title/filename dates separate because they are the first latest-date choice.
                    "title_filename_dates": set(),
                    # Keep body dates separate because they are only a final latest-date fallback.
                    "content_dates": set(),
                    # Aggregate publish dates separately for secondary fallback use.
                    "publish_dates": set(),
                    # Aggregate recognized document types across the PDF.
                    "document_types": set(),
                    # Record the service from the filename prefix for ambiguity reporting.
                    "service": "ISS" if file_name.startswith("iss") else "SSS",
                },
            )
            # Add the current chunk to its filename group.
            file_record["documents"].append(document)
            # Build the primary searchable text without mixing in publish_date.
            primary_text = _get_service_order_primary_text(document)
            # Build the reliable identity text separately for latest-document date selection.
            title_filename_text = _get_service_order_title_filename_text(document)
            # Add every order key found in the PDF title, filename, or current chunk.
            file_record["order_keys"].update(_extract_service_order_keys(primary_text))
            # Add every real order date found in the title, filename, or current chunk.
            file_record["primary_dates"].update(_extract_service_order_dates(primary_text))
            # Store identity dates separately so page references cannot outrank the official order date.
            file_record["title_filename_dates"].update(
                _extract_service_order_dates(title_filename_text)
            )
            # Store content dates only for PDFs that have no identity date or publish date.
            file_record["content_dates"].update(
                _extract_service_order_dates(document.page_content)
            )
            # Add publish_date values only to the separate secondary-date collection.
            file_record["publish_dates"].update(
                _extract_service_order_dates(_get_service_order_publish_date(document))
            )
            # Detect the order type from the current chunk and its metadata.
            detected_document_type = _detect_service_order_type(primary_text)
            # Store the detected type only when a supported type phrase was actually found.
            if detected_document_type:
                file_record["document_types"].add(detected_document_type)

        # Stop when Qdrant returns no continuation offset for the filtered result set.
        if next_page_offset is None:
            break

    # Build the set of files whose normalized order number exactly matches the user's number.
    exact_order_files = {
        # Keep the grouped filename as the stable identity of the matching PDF.
        file_name
        # Examine every ISS/SSS order PDF collected from Qdrant.
        for file_name, file_record in documents_by_file.items()
        # Require at least one normalized order-key intersection.
        if query_order_keys.intersection(file_record["order_keys"])
    }
    # Narrow exact-order files to those whose real order date also matches the user query.
    exact_order_and_primary_date_files = {
        # Preserve the filename for later collection of all matching chunks.
        file_name
        # Only exact-order candidates need the stronger real-date comparison.
        for file_name in exact_order_files
        # Require a query date and an intersection with title/filename/content dates.
        if query_dates and query_dates.intersection(documents_by_file[file_name]["primary_dates"])
    }
    # Use publish_date only when no real order-date match is available.
    exact_order_and_publish_date_files = {
        # Preserve the filename for the secondary-date fallback result.
        file_name
        # Only exact-order candidates need the publish-date comparison.
        for file_name in exact_order_files
        # Require a query date and an intersection with stored publish_date metadata.
        if query_dates and query_dates.intersection(documents_by_file[file_name]["publish_dates"])
    }
    # Build date-and-type candidates for queries that do not contain a usable order number.
    document_type_and_primary_date_files = {
        # Preserve the filename for later chunk collection.
        file_name
        # Examine every collected service-order PDF.
        for file_name, file_record in documents_by_file.items()
        # Require the requested type and an actual order-date match.
        if query_document_type
        and query_document_type in file_record["document_types"]
        and query_dates.intersection(file_record["primary_dates"])
    }
    # Build the final publish-date fallback for date-and-type queries.
    document_type_and_publish_date_files = {
        # Preserve the filename for later chunk collection.
        file_name
        # Examine every collected service-order PDF.
        for file_name, file_record in documents_by_file.items()
        # Require the requested type and a publish-date match only as the last structured fallback.
        if query_document_type
        and query_document_type in file_record["document_types"]
        and query_dates.intersection(file_record["publish_dates"])
    }

    # Prepare a date-descending list only for latest type-based ISS/SSS requests.
    latest_file_candidates: List[Tuple[str, str, str]] = []
    # Do not let latest logic interfere with an exact order-number or explicit-date query.
    if latest_mode:
        # Examine every unique ISS/SSS PDF collected by the fully paginated Qdrant scroll.
        for file_name, file_record in documents_by_file.items():
            # Skip PDFs that do not match the requested transfer/promotion/office/etc. type.
            if query_document_type not in file_record["document_types"]:
                continue
            # Choose one effective date using title/filename, publish_date, then content priority.
            effective_date, date_source = _get_latest_service_order_date(file_record)
            # Save the effective date for metadata, logging, and stable downstream display.
            file_record["effective_date"] = effective_date
            # Save which source produced the date so operators can diagnose unexpected ordering.
            file_record["effective_date_source"] = date_source
            # Add one candidate per PDF; chunks are deliberately not sorted as separate documents.
            latest_file_candidates.append((file_name, effective_date, date_source))

        # Put dated PDFs first, sort dates newest-to-oldest, and use filename only as a stable tie-breaker.
        latest_file_candidates.sort(
            key=lambda candidate: (bool(candidate[1]), candidate[1], candidate[0]),
            reverse=True,
        )

    # Latest type queries select unique PDFs from the already date-descending candidate list.
    if latest_file_candidates:
        # Use one file for singular "latest order" and up to top_k files for plural "latest orders".
        latest_file_limit = _latest_service_order_file_limit(query, top_k)
        # Start with the requested number of unique newest PDFs.
        selected_latest_candidates = latest_file_candidates[:latest_file_limit]
        # If a singular query has an exact newest-date tie, include all tied PDFs without exceeding top_k.
        if latest_file_limit == 1 and selected_latest_candidates and selected_latest_candidates[0][1]:
            newest_date = selected_latest_candidates[0][1]
            selected_latest_candidates = [
                candidate
                for candidate in latest_file_candidates
                if candidate[1] == newest_date
            ][:max(1, top_k)]
        # Preserve this list order because it is already newest-to-oldest at the PDF level.
        selected_file_names = [candidate[0] for candidate in selected_latest_candidates]
        # Record a distinct reason so the final stage knows semantic reranking must be skipped.
        match_reason = "latest_document_type_by_effective_date"
    # Give exact order number plus actual order date the highest priority.
    elif exact_order_and_primary_date_files:
        selected_file_names = exact_order_and_primary_date_files
        match_reason = "exact_order_and_primary_date"
    # Use an exact order number plus publish date only when no real-date match exists.
    elif exact_order_and_publish_date_files:
        selected_file_names = exact_order_and_publish_date_files
        match_reason = "exact_order_and_publish_date_fallback"
    # Use the exact order number by itself when the user omitted a date or no date matched.
    elif exact_order_files:
        selected_file_names = exact_order_files
        match_reason = "exact_order"
    # Use document type plus real order date when the query contains no usable order number.
    elif document_type_and_primary_date_files:
        selected_file_names = document_type_and_primary_date_files
        match_reason = "document_type_and_primary_date"
    # Use document type plus publish date only as the final structured fallback.
    elif document_type_and_publish_date_files:
        selected_file_names = document_type_and_publish_date_files
        match_reason = "document_type_and_publish_date_fallback"
    # Return no specialized documents so the existing semantic/vector fallback remains unchanged.
    else:
        logger.info(f"[SERVICE_ORDER_SEARCH] No exact structured match found for query: {query}")
        return []

    # Determine whether the selected files span both services and therefore remain ambiguous.
    selected_services = {
        # Read the recorded service label for each selected filename.
        documents_by_file[file_name]["service"]
        # Examine every selected matching filename.
        for file_name in selected_file_names
    }
    # Cross-service latest results are expected; ambiguity applies only to exact structured matches.
    is_ambiguous = len(selected_services) > 1 and not latest_mode
    # Keep the selected chunks grouped by filename so ambiguous services both receive representation.
    selected_documents_by_file = []

    # Preserve effective-date order for latest results; use stable filename order for exact matches.
    selected_file_order = list(selected_file_names) if latest_mode else sorted(selected_file_names)
    # Collect documents from every selected PDF in the chosen file-level order.
    for file_name in selected_file_order:
        # Sort each PDF's chunks by page number before limiting the returned context.
        file_documents = sorted(
            documents_by_file[file_name]["documents"],
            key=_service_order_page_sort_key,
        )
        # Mark every returned chunk as an exact structured service-order result.
        for document in file_documents:
            # Tell the final retrieval stage that generic vector documents must not be mixed in.
            document.metadata["_exact_service_order_match"] = True
            # Preserve the structured reason to make debugging the selected result straightforward.
            document.metadata["_service_order_match_reason"] = match_reason
            # Record the identified service directly on the in-memory result metadata.
            document.metadata["_service_order_service"] = documents_by_file[file_name]["service"]
            # Record whether the same structured query matched both ISS and SSS files.
            document.metadata["_service_order_ambiguous"] = is_ambiguous
            # Mark latest results so the final retrieval stage preserves descending date order.
            document.metadata["_latest_service_order_match"] = latest_mode
            # Expose the effective order date selected for this PDF to downstream diagnostics.
            document.metadata["_service_order_effective_date"] = documents_by_file[file_name].get("effective_date", "")
            # Expose whether the date came from identity, publish metadata, content, or no date.
            document.metadata["_service_order_date_source"] = documents_by_file[file_name].get("effective_date_source", "")
        # Add this filename group to the final selection pool.
        selected_documents_by_file.append(file_documents)

    # Prepare the final structured chunk list.
    selected_documents: List[Document] = []
    # Latest mode returns the first page-ordered chunk from each unique date-sorted PDF.
    if latest_mode:
        # Keep one chunk per PDF so five result slots represent five different recent documents.
        selected_documents = [
            file_documents[0]
            for file_documents in selected_documents_by_file
            if file_documents
        ][:top_k]
    # Continue taking one chunk per selected file until top_k context slots are filled.
    while not latest_mode and len(selected_documents) < top_k and any(selected_documents_by_file):
        # Iterate across files rather than exhausting the first file before the second one.
        for file_documents in selected_documents_by_file:
            # Skip a file after all of its chunks have already been taken.
            if not file_documents:
                continue
            # Take the next page-ordered chunk from this PDF.
            selected_documents.append(file_documents.pop(0))
            # Stop immediately once the requested result limit has been reached.
            if len(selected_documents) >= top_k:
                break

    # Log the exact filenames, services, and priority rule that produced the structured result.
    logger.info(
        f"[SERVICE_ORDER_SEARCH] Matched files={selected_file_order}, "
        f"services={sorted(selected_services)}, reason={match_reason}, ambiguous={is_ambiguous}"
    )
    # Return only selected service-order chunks to the existing metric-search caller.
    return selected_documents


def detect_metric_in_query(query: str) -> Optional[str]:
    """
    Detect which metric (if any) is mentioned in the query.
    
    Returns: "SERVICE_ORDER", "CPI", "IIP", "GDP", "PLFS", "ASI", "NSS", "EC", or None
    """
    # Route exact order-number queries through the existing metric-search stage without a pre-search.
    if _extract_service_order_keys(query):
        logger.info(f"[METRIC_DETECTION] Exact service-order number detected in query: {query}")
        return "SERVICE_ORDER"

    # Also route document-type plus date queries, for example "SSS order dated 29.07.2026".
    if _detect_service_order_type(query) and _extract_service_order_dates(query):
        logger.info(f"[METRIC_DETECTION] Service-order type and date detected in query: {query}")
        return "SERVICE_ORDER"

    # Route latest transfer/promotion/office/etc. requests through the same ISS/SSS structured stage.
    if _detect_service_order_type(query) and _is_latest_service_order_query(query):
        logger.info(f"[METRIC_DETECTION] Latest service-order type detected in query: {query}")
        return "SERVICE_ORDER"

    q = query.lower()
    
    # Define comprehensive metric keywords
    # Note: Avoiding very short acronyms (2-3 letters) to prevent false matches
    # Using full forms instead for safety
    metric_keywords = {
        'CPI': [
            'cpi', 'consumer price index', 'inflation',
            'price index', 'price indices', 'retail inflation',
            'headline inflation', 'core inflation', 'food inflation',
            'fuel inflation', 'cost of living', 'price rise', 'inflationary'
        ],
        'IIP': [
            'iip', 'industrial production', 'index of industrial production',
            'manufacturing output', 'industrial output', 'factory output',
            'production index', 'industrial growth', 'manufacturing growth',
            'manufacturing sector', 'industrial sector'
        ],
        'GDP': [
            'gdp', 'gross domestic product', 'economic growth',
            'gdp growth', 'growth rate', 'national income',
            'national accounts', 'gva', 'gross value added',
            'economic output', 'quarterly estimate',
            'annual estimate', 'advance estimate', 'provisional estimate',
            'first advance estimate', 'second advance estimate'
        ],
        'PLFS': [
            'plfs', 'periodic labour force survey', 'periodic labor force survey',
            'labour force', 'labor force',
            'employment', 'unemployment',
            'wpr', 'worker population ratio',
            'lfpr', 'labour force participation rate', 'labor force participation rate',
            'unemployment rate', 'employment rate',
            'jobless', 'joblessness', 'workforce',
            'labour statistics', 'labor statistics', 'employment statistics',
            'rural employment', 'urban employment',
            'youth unemployment', 'female employment', 'male employment'
        ],
        'ASI': [
            'asi', 'annual survey of industries',
            'factory sector', 'organized manufacturing',
            'registered factories', 'industrial statistics',
            'factory statistics'
        ],
        'NSS': [
            'nss', 'national sample survey',
            'nsso', 'national sample survey organisation', 'national sample survey organization',
            'household survey', 'consumption expenditure',
            'household consumption expenditure', 'hces'
        ],
        'EC': [
            'economic census', 'establishment survey',
            'enterprise census', 'business census'
        ],
        'HCES': [
            'hces', 'household consumer expenditure survey',
            'household consumption expenditure survey',
            'household expenditure', 'consumer expenditure survey',
            'consumption expenditure', 'household survey',
            'household consumption', 'consumer spending',
            'households surveyed', 'household sample',  # Fixed: removed regex patterns
            'urban households', 'rural households',  # Added: specific patterns
            'households were surveyed', 'number of households'  # Added: query patternse .* for multi-word patterns
        ],
        'ISS': [
            'iss', 'indian statistical service',
            'promotion order', 'transfer order', 'office order',
            'vacancy position', 'posting order', 'appointment order',
            'confirmation order', 'deputation order',
            'iss order', 'iss orders', 'iss promotion', 'iss transfer',
            'iss vacancy', 'iss posting', 'iss appointment'
        ],
    }
    
    # Check for each metric
    for metric, keywords in metric_keywords.items():
        for keyword in keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', q):
                return metric
    
    return None


def _build_enhanced_metric_filter(
    detected_metric: str,
    time_pref: Optional[Dict] = None,
    include_backward_compat: bool = True
) -> qmodels.Filter:
    metric_lower = detected_metric.lower()
    metric_upper = detected_metric.upper()
    
    # Month names mapping
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    
    conditions = []
    enhanced_count = 0
    
    # PART 1: Enhanced AND combinations for recent documents (if time_pref available)
    if time_pref and time_pref.get("target_months"):
        target_months = time_pref.get("target_months", [])
        
        try:
            # Check if this is ISS metric (uses different naming convention)
            is_iss_metric = (detected_metric.upper() == "ISS")
            
            # For each target month, create specific AND combinations
            for year, month in target_months:
                month_name = month_names.get(month)
                year_str = str(year)
                year_month_prefix = f"{year}-{month:02d}"
                
                if not month_name:
                    continue
                
                if is_iss_metric:
                    possible_dates = []
                    for day in range(1, 32):  # 1 to 31
                        date_str = f"{year}-{month:02d}-{day:02d}"
                        possible_dates.append(date_str)
                    
                    # Pattern: iss in filename + publish_date is any day in this month
                    conditions.append(qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text="iss")),
                            qmodels.FieldCondition(key="publish_date", match=qmodels.MatchAny(any=possible_dates))
                        ]
                    ))
                    enhanced_count += 1
                    
                else:
                    conditions.append(qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text="latest_release")),
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text="Press_Release")),
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=metric_upper)),
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=month_name)),
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=year_str))
                        ]
                    ))
                    enhanced_count += 1
                    
                    # Combination 2: Press_Release AND metric AND month AND year
                    conditions.append(qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text="Press_Release")),
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=metric_upper)),
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=month_name)),
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=year_str))
                        ]
                    ))
                    enhanced_count += 1
                    
                    # Combination 3: metric AND month AND year
                    conditions.append(qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=metric_upper)),
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=month_name)),
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=year_str))
                        ]
                    ))
                    enhanced_count += 1
                    
                    # Combination 4: latest_release AND metric AND month AND year
                    conditions.append(qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text="latest_release")),
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=metric_upper)),
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=month_name)),
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=year_str))
                        ]
                    ))
                    enhanced_count += 1
                    
                    # Combination 5: metric in file_name AND publish_date in target month (format-agnostic)
                    # This catches documents regardless of filename format (Jan26, January 2026, etc.)
                    conditions.append(qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=metric_upper)),
                            qmodels.FieldCondition(key="publish_date", match=qmodels.MatchText(text=year_month_prefix))
                        ]
                    ))
                    enhanced_count += 1
        
        except Exception as e:
            logger.warning(f"[METRIC_SEARCH_V2] Failed to build enhanced filters: {e}, falling back to simple patterns")
    
    # PART 2: Backward compatibility patterns (simple metric name matches)
    # Only include if requested
    backward_compat_count = 0
    if include_backward_compat:
        backward_compat_patterns = {
            'cpi': ['cpi', 'CPI', 'consumer price', 'inflation'],
            'iip': ['iip', 'IIP', 'industrial production', 'manufacturing'],
            'gdp': ['gdp', 'GDP', 'gross domestic', 'national accounts', 'nad_pr', 'gva'],
            'plfs': ['plfs', 'PLFS', 'labour force', 'labor force', 'employment', 'periodic labour'],
            'asi': ['asi', 'ASI', 'annual survey of industries', 'factory sector'],
            'nss': ['nss', 'NSS', 'national sample survey', 'nsso', 'household survey'],
            'ec': ['economic census', 'establishment survey', 'enterprise census'],
            'hces': ['hces', 'HCES', 'household consumer expenditure', 'household consumption expenditure',
                     'consumer expenditure survey', 'household expenditure survey'],
            'iss': ['iss', 'ISS', 'indian statistical service', 'promotion', 'transfer', 
                    'office order', 'vacancy', 'posting', 'appointment'],
        }
        
        patterns = backward_compat_patterns.get(metric_lower, [metric_lower, metric_upper])
        
        for pattern in patterns:
            conditions.append(qmodels.FieldCondition(
                key="file_name",
                match=qmodels.MatchText(text=pattern)
            ))
            backward_compat_count += 1
    
    search_filter = qmodels.Filter(should=conditions)
    
    logger.info(f"[METRIC_SEARCH_V2] Built filter with {len(conditions)} conditions "
                f"({enhanced_count} enhanced + {backward_compat_count} backward compat)")
    
    return search_filter


def metric_search_simple(
    query: str,
    detected_metric: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    embeddings,  # NEW: Need embeddings to generate query vector
    top_k: int = 5,
    use_grpc: bool = True,
    time_pref: Optional[Dict] = None
) -> List[Document]:

    if use_grpc:
        try:
            qdrant_client = get_qdrant_client_grpc()
            logger.debug(f"[METRIC_SEARCH_V2] Using gRPC client")
        except Exception as grpc_error:
            logger.warning(f"[METRIC_SEARCH_V2] gRPC client failed, using provided client: {grpc_error}")

    # Keep exact ISS/SSS order handling inside the existing metric-search stage rather than adding a pre-search.
    if detected_metric.upper() == "SERVICE_ORDER":
        # Run the structured order-number/date matcher against the current Qdrant collection.
        service_order_documents = _search_exact_service_order_documents(
            # Pass the user's order query so identifiers and dates can be normalized.
            query=query,
            # Reuse the active HTTP or gRPC Qdrant client selected above.
            qdrant_client=qdrant_client,
            # Reuse the collection already configured for the chatbot.
            collection_name=collection_name,
            # Respect the same metric-result limit used by the existing pipeline.
            top_k=top_k,
        )
        # Return structured matches directly to retrieve_priority_chunks_v2 for exclusive handling.
        if service_order_documents:
            return service_order_documents
        # Return an empty metric result when no structured match exists so vector fallback remains unchanged.
        return []

    # Generate query vector
    try:
        query_vector = embeddings.embed_query(query)
    except Exception as e:
        logger.error(f"[METRIC_SEARCH_V2] Failed to generate query vector: {e}")
        return []
    
    # TWO-STAGE SEARCH: Prioritize enhanced filter, fallback to full filter
    # Stage 1: Try enhanced filter ONLY (if time_pref available)
    # This prioritizes recent documents with specific month/year patterns
    # EXCEPTION: ISS skips Stage 1 because MatchText on "iss" doesn't work reliably
    all_docs = []
    search_timeout = 30 if detected_metric.upper() in ['HCES', 'NSS', 'EC', 'ASI'] else 60
    
    # ISS-specific flag: Use scroll instead of query_points to prioritize date over vector similarity
    is_iss_metric = (detected_metric.upper() == "ISS")
    
    # ISS: Skip Stage 1, go directly to Stage 2 with backward compat
    # Reason: MatchText on file_name with "iss" doesn't match all ISS documents
    skip_stage_1 = is_iss_metric
    
    if time_pref and time_pref.get("target_months") and not skip_stage_1:
        try:
            # Build enhanced filter WITHOUT backward compatibility
            enhanced_filter = _build_enhanced_metric_filter(detected_metric, time_pref, include_backward_compat=False)
            
            logger.info(f"[METRIC_SEARCH_V2] Stage 1: Trying enhanced filter only (no backward compat)")
            
            # ISS ONLY: Use scroll to get ALL matching documents (prioritize date over vector similarity)
            # Other metrics: Use query_points (prioritize vector similarity)
            if is_iss_metric:
                logger.info(f"[METRIC_SEARCH_V2] ISS detected: Using scroll to prioritize publish_date over vector similarity")
                
                # Use scroll to get all documents matching the filter
                scroll_result = qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=enhanced_filter,
                    limit=500,  # Get up to 500 ISS documents
                    with_payload=True,
                )
                
                points = scroll_result[0]  # scroll returns (points, next_page_offset)
                logger.info(f"[METRIC_SEARCH_V2] ISS scroll found {len(points)} documents")
                
                # Convert to Documents
                for point in points:
                    payload = point.payload or {}
                    content = payload.get("page_content", "")
                    metadata = {k: v for k, v in payload.items() if k != "page_content"}
                    
                    # Filter out irrelevant documents
                    file_name = metadata.get("file_name", "").lower()
                    exclude_keywords = ['tender', 'enquiry', 'audit', 'security', 'manual', 'methodology']
                    
                    if any(kw in file_name for kw in exclude_keywords):
                        continue
                    
                    doc = Document(page_content=content, metadata=metadata)
                    all_docs.append(doc)
                
            else:
                # ALL OTHER METRICS: Use standard query_points with vector similarity
                search_result = qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=enhanced_filter,
                    limit=100,
                    with_payload=True,
                    timeout=search_timeout,
                )
                
                # Convert to Documents
                for point in search_result.points:
                    payload = point.payload or {}
                    content = payload.get("page_content", "")
                    metadata = {k: v for k, v in payload.items() if k != "page_content"}
                    
                    # Filter out irrelevant documents
                    file_name = metadata.get("file_name", "").lower()
                    exclude_keywords = ['tender', 'enquiry', 'audit', 'security', 'manual', 'methodology']
                    
                    if any(kw in file_name for kw in exclude_keywords):
                        continue
                    
                    doc = Document(page_content=content, metadata=metadata)
                    all_docs.append(doc)
            
            logger.info(f"[METRIC_SEARCH_V2] Stage 1: Found {len(all_docs)} docs with enhanced filter only")
            
            # If we found enough documents (threshold: 5), use them
            # This ensures we prioritize recent documents with specific patterns
            if len(all_docs) >= 5:
                logger.info(f"[METRIC_SEARCH_V2] Stage 1: SUCCESS - Using {len(all_docs)} docs from enhanced filter")
            else:
                logger.info(f"[METRIC_SEARCH_V2] Stage 1: INSUFFICIENT - Only {len(all_docs)} docs, trying Stage 2")
                all_docs = []  # Clear and try Stage 2
                
        except Exception as e:
            logger.warning(f"[METRIC_SEARCH_V2] Stage 1 failed: {e}, falling back to Stage 2")
            all_docs = []
    
    # Stage 2: Full filter (enhanced + backward compatibility)
    # This is used if:
    # - No time_pref available
    # - Stage 1 found < 5 documents
    # - Stage 1 failed
    # - ISS metric (skips Stage 1)
    if len(all_docs) == 0:
        try:
            logger.info(f"[METRIC_SEARCH_V2] Stage 2: Using full filter (enhanced + backward compat)")
            
            # Build full filter with backward compatibility
            search_filter = _build_enhanced_metric_filter(detected_metric, time_pref, include_backward_compat=True)
            
            # ISS: Use scroll (prioritize date over vector similarity)
            # Other metrics: Use query_points (prioritize vector similarity)
            if is_iss_metric:
                logger.info(f"[METRIC_SEARCH_V2] ISS Stage 2: Using scroll with backward compat patterns")
                
                scroll_result = qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=search_filter,
                    limit=500,
                    with_payload=True,
                )
                
                points = scroll_result[0]
                logger.info(f"[METRIC_SEARCH_V2] ISS Stage 2 scroll found {len(points)} documents")
                
                for point in points:
                    payload = point.payload or {}
                    content = payload.get("page_content", "")
                    metadata = {k: v for k, v in payload.items() if k != "page_content"}
                    
                    file_name = metadata.get("file_name", "").lower()
                    exclude_keywords = ['tender', 'enquiry', 'audit', 'security', 'manual', 'methodology']
                    
                    if any(kw in file_name for kw in exclude_keywords):
                        continue
                    
                    doc = Document(page_content=content, metadata=metadata)
                    all_docs.append(doc)
            else:
                # Other metrics: Use query_points with vector similarity
                search_result = qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=100,  # Get top 100 most relevant metric docs
                    with_payload=True,
                    timeout=search_timeout,
                )
                
                # Convert to Documents
                for point in search_result.points:
                    payload = point.payload or {}
                    content = payload.get("page_content", "")
                    metadata = {k: v for k, v in payload.items() if k != "page_content"}
                    
                    # Filter out irrelevant documents
                    file_name = metadata.get("file_name", "").lower()
                    exclude_keywords = ['tender', 'enquiry', 'audit', 'security', 'manual', 'methodology']
                    
                    if any(kw in file_name for kw in exclude_keywords):
                        continue
                    
                    doc = Document(page_content=content, metadata=metadata)
                    all_docs.append(doc)
            
            logger.info(f"[METRIC_SEARCH_V2] Stage 2: Found {len(all_docs)} docs with full filter")
            
        except Exception as filter_error:
            logger.error(f"[METRIC_SEARCH_V2] Stage 2 failed: {filter_error}, using simple fallback")
            # Fallback: simple filter with just metric name
            metric_upper = detected_metric.upper()
            search_filter = qmodels.Filter(should=[
                qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=metric_upper)),
                qmodels.FieldCondition(key="file_name", match=qmodels.MatchText(text=detected_metric.lower()))
            ])
            
            try:
                search_result = qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=100,
                    with_payload=True,
                    timeout=search_timeout,
                )
                
                for point in search_result.points:
                    payload = point.payload or {}
                    content = payload.get("page_content", "")
                    metadata = {k: v for k, v in payload.items() if k != "page_content"}
                    doc = Document(page_content=content, metadata=metadata)
                    all_docs.append(doc)
                    
                logger.info(f"[METRIC_SEARCH_V2] Fallback: Found {len(all_docs)} docs with simple filter")
            except Exception as e:
                logger.error(f"[METRIC_SEARCH_V2] All search stages failed: {e}")
                return []
    
    if len(all_docs) == 0:
        logger.warning(f"[METRIC_SEARCH_V2] No documents found for metric: {detected_metric}")
        return []
    
    try:        
        logger.info(f"[METRIC_SEARCH_V2] Found {len(all_docs)} {detected_metric} documents via vector search with filter")
        
        # IMPORTANT: Sort by publish_date (newest first) BEFORE filtering by target_months
        # This ensures recent documents are prioritized over old documents with similar vector scores
        def get_publish_date_sort_key(doc):
            date_str = doc.metadata.get("publish_date", "")
            if date_str and date_str != "no_date":
                return date_str
            return "0000-00-00"  # Put no_date documents at the end
        
        all_docs.sort(key=get_publish_date_sort_key, reverse=True)
        
        # DEBUG: Log first 10 documents AFTER sorting by date
        logger.info(f"[METRIC_SEARCH_V2] DEBUG: First 10 CPI documents AFTER date sorting:")
        for i, doc in enumerate(all_docs[:10]):
            fname = doc.metadata.get("file_name", "unknown")
            pdate = doc.metadata.get("publish_date", "no_date")
            logger.info(f"[METRIC_SEARCH_V2]   {i+1}. {fname[:80]} (publish_date={pdate})")
        
        # Filter by time preference using target_months (OR condition: filename OR date)
        target_months = time_pref.get("target_months", []) if time_pref else []
        
        # Check if metric has dual cadence (monthly + quarterly)
        has_dual_cadence = (detected_metric.upper() == "PLFS")
        
        if target_months:
            monthly_docs = []
            quarterly_docs = []
            
            # DEBUG: Log first 10 documents to see what we're working with
            if ENABLE_RETRIEVAL_DEBUG and len(all_docs) > 0:
                logger.info(f"[METRIC_SEARCH_V2] DEBUG: First 10 CPI documents:")
                for i, doc in enumerate(all_docs[:10]):
                    fname = doc.metadata.get("file_name", "unknown")
                    pdate = doc.metadata.get("publish_date", "no_date")
                    logger.info(f"[METRIC_SEARCH_V2]   {i+1}. {fname[:80]} (publish_date={pdate})")
            
            # STEP 1: Primary search - Look for monthly bulletins matching target_months
            for doc in all_docs:
                file_name = doc.metadata.get("file_name", "").lower()
                publish_date_str = doc.metadata.get("publish_date", "")
                publish_date = _parse_publish_date(publish_date_str)
                
                match_type = None  # "monthly", "quarterly", or None
                
                # Check publish_date against target_months
                if publish_date:
                    doc_year_month = (publish_date.year, publish_date.month)
                    if doc_year_month in target_months:
                        match_type = "date"
                        # DEBUG: Log matched document
                        if ENABLE_RETRIEVAL_DEBUG:
                            logger.info(f"[METRIC_SEARCH_V2] DEBUG: MATCHED by date: {file_name[:80]} "
                                      f"(doc_month={doc_year_month}, target_months={target_months[:3]})")
                
                # Check filename for year/month patterns (ENHANCED MATCHING)
                if not match_type:
                    for target_year, target_month in target_months:
                        if _matches_month_year_in_filename(file_name, target_year, target_month):
                            match_type = "filename"
                            break
                
                # Categorize by document type (monthly vs quarterly)
                if match_type:
                    # Detect if it's a monthly or quarterly bulletin
                    is_monthly = any(kw in file_name for kw in ['monthly', 'mb_', '_mb_', 'month'])
                    is_quarterly = any(kw in file_name for kw in ['quarterly', 'qb_', '_qb_', 'quarter'])
                    
                    if has_dual_cadence:
                        if is_monthly:
                            monthly_docs.append(doc)
                        elif is_quarterly:
                            quarterly_docs.append(doc)
                        else:
                            # Unknown type, add to monthly by default
                            monthly_docs.append(doc)
                    else:
                        monthly_docs.append(doc)
            
            # STEP 2: Secondary search - For dual cadence, also look for quarterly bulletins containing target months
            if has_dual_cadence and target_months:
                # Map target months to quarters (Indian FY)
                target_quarters = set()
                for target_year, target_month in target_months:
                    # Map month to Indian FY quarter
                    if target_month >= 4 and target_month <= 6:
                        # Q1: Apr-Jun
                        target_quarters.add((target_year, 1, "q1"))
                    elif target_month >= 7 and target_month <= 9:
                        # Q2: Jul-Sep
                        target_quarters.add((target_year, 2, "q2"))
                    elif target_month >= 10 and target_month <= 12:
                        # Q3: Oct-Dec
                        target_quarters.add((target_year, 3, "q3"))
                    else:  # Jan-Mar
                        # Q4: Jan-Mar (belongs to previous FY year)
                        target_quarters.add((target_year - 1, 4, "q4"))
                
                logger.info(f"[METRIC_SEARCH_V2] Dual cadence: Looking for quarterly bulletins for quarters: {target_quarters}")
                
                # Search for quarterly bulletins matching these quarters
                for doc in all_docs:
                    file_name = doc.metadata.get("file_name", "").lower()
                    
                    # Skip if already in quarterly_docs
                    if doc in quarterly_docs:
                        continue
                    
                    # Check if it's a quarterly bulletin
                    is_quarterly = any(kw in file_name for kw in ['quarterly', 'qb_', '_qb_', 'quarter'])
                    
                    if is_quarterly:
                        # Check if filename contains any of the target quarters
                        for fy_year, quarter_num, quarter_str in target_quarters:
                            # Patterns: "Q1", "q1", "Q1_2025", "2025_Q1", "april_june_2025", etc.
                            quarter_patterns = [
                                f"q{quarter_num}",
                                f"quarter_{quarter_num}",
                                f"{fy_year}",  # Year match
                            ]
                            
                            # Add month range patterns for each quarter
                            if quarter_num == 1:
                                quarter_patterns.extend(["april", "may", "june", "apr", "jun"])
                            elif quarter_num == 2:
                                quarter_patterns.extend(["july", "august", "september", "jul", "aug", "sep", "sept"])
                            elif quarter_num == 3:
                                quarter_patterns.extend(["october", "november", "december", "oct", "nov", "dec"])
                            else:  # Q4
                                quarter_patterns.extend(["january", "february", "march", "jan", "feb", "mar"])
                            
                            # Check if any pattern matches
                            if any(pattern in file_name for pattern in quarter_patterns):
                                # Additional check: year should match
                                if str(fy_year) in file_name or str(fy_year + 1) in file_name:
                                    quarterly_docs.append(doc)
                                    logger.info(f"[METRIC_SEARCH_V2] Found quarterly bulletin for {quarter_str.upper()} FY{fy_year}: {doc.metadata.get('file_name', 'unknown')[:80]}")
                                    break
            
            # Combine: prioritize monthly, then quarterly
            if has_dual_cadence:
                filtered_docs = monthly_docs + quarterly_docs
                logger.info(f"[METRIC_SEARCH_V2] Dual cadence: {len(monthly_docs)} monthly + {len(quarterly_docs)} quarterly docs")
            else:
                filtered_docs = monthly_docs
            
            if filtered_docs:
                logger.info(f"[METRIC_SEARCH_V2] Filtered to {len(filtered_docs)} docs matching target_months: "
                           f"{[(y, m) for y, m in target_months[:6]]}")
                all_docs = filtered_docs
            else:
                logger.warning(f"[METRIC_SEARCH_V2] No docs matched target_months {target_months[:6]}, "
                              f"using all {len(all_docs)} docs")
        
        # Log top docs (already sorted by date above)
        logger.info(f"[METRIC_SEARCH_V2] Top {min(5, len(all_docs))} {detected_metric} docs:")
        for i, doc in enumerate(all_docs[:5]):
            meta = doc.metadata
            doc_type = "monthly" if any(kw in meta.get('file_name', '').lower() for kw in ['monthly', 'mb_', '_mb_']) else "quarterly" if any(kw in meta.get('file_name', '').lower() for kw in ['quarterly', 'qb_', '_qb_']) else "unknown"
            logger.info(f"[METRIC_SEARCH_V2]   {i+1}. {meta.get('file_name', 'unknown')} "
                       f"(date={meta.get('publish_date', 'no_date')}, type={doc_type})")

        
        return all_docs[:top_k]
        
    except Exception as e:
        error_msg = str(e)
        if 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower():
            logger.warning(f"[METRIC_SEARCH_V2] Timeout for {detected_metric} after {search_timeout}s - "
                          f"metric docs might not exist or filter is too broad. Skipping metric search.")
            return []  # Return empty list, vector search will provide results
        else:
            logger.error(f"[METRIC_SEARCH_V2] Failed for {detected_metric}: {e}", exc_info=True)
            return []


def dedupe_documents_smart(docs: List[Document]) -> List[Document]:

    if not docs:
        return []
    
    # Group by (file_name, page_number)
    from collections import defaultdict
    groups = defaultdict(list)
    
    for doc in docs:
        meta = doc.metadata
        key = (
            meta.get("file_name", ""),
            meta.get("page_number", "")
        )
        groups[key].append(doc)
    
    # For each group, keep the one with most recent publish_date
    deduped = []
    
    for key, group_docs in groups.items():
        if len(group_docs) == 1:
            deduped.append(group_docs[0])
        else:
            # Sort by publish_date (newest first)
            def get_date_for_sort(doc):
                date_str = doc.metadata.get("publish_date", "")
                if date_str and date_str != "no_date":
                    return date_str
                return "0000-00-00"
            
            group_docs.sort(key=get_date_for_sort, reverse=True)
            deduped.append(group_docs[0])
            
            # Log deduplication
            if len(group_docs) > 1:
                logger.debug(f"[DEDUPE_V2] Kept most recent of {len(group_docs)} versions: "
                           f"{key[0]} p.{key[1]} (date={group_docs[0].metadata.get('publish_date', 'no_date')})")
    
    logger.info(f"[DEDUPE_V2] {len(docs)} docs → {len(deduped)} after smart deduplication")
    
    return deduped


def retrieve_priority_chunks_v2(
    query: str,
    vectordb,
    rerank_function,
    final_top_k: int = 10,
    original_query: Optional[str] = None,
    cadence_hints: Optional[Dict] = None,  # NEW: Cadence hints from query_routing
    whois_prefer_file: bool = False,  # NEW: Whois routing hint
) -> Tuple[List[Tuple[Document, float]], None, None, Optional[str]]:

    import time
    retrieval_start = time.time()
    
    logger.info(f"[RETRIEVAL_V2] Starting double-rerank retrieval for query: '{query[:100]}...'")
    
    # PRIORITY 0: Whois routing (exclusive mode - skip all other retrieval)
    if whois_prefer_file:
        logger.info(f"[RETRIEVAL_V2] Whois query detected - attempting to use cached whois document")
        cached_whois_doc = get_cached_whois_doc()
        
        if cached_whois_doc:
            # Return ONLY the whois document (exclusive mode)
            whois_result = [(cached_whois_doc, 10.0)]  # High score for whois doc
            retrieval_time = time.time() - retrieval_start
            logger.info(f"[RETRIEVAL_V2]  Whois routing: Returned cached whois doc ({len(cached_whois_doc.page_content)} chars) in {retrieval_time:.2f}s")
            logger.info(f"[RETRIEVAL_V2] ⏱️ TIMING BREAKDOWN: Whois Cache={retrieval_time:.2f}s | Total={retrieval_time:.2f}s")
            return whois_result, None, None, None
        else:

            logger.error(
                "[RETRIEVAL_V2] !! WHOIS CACHE EMPTY: cached whois doc is unavailable, "
                "so this whois query is FALLING BACK to normal Qdrant retrieval (degraded). "
                "Check whois_cache.txt in the container and the [WHOIS_CACHE] startup logs."
            )
    
    # PRIORITY 1: Use cadence hints if available (business logic from query_routing)
    if cadence_hints:
        target_months = _convert_cadence_to_target_months(cadence_hints)
        time_pref = {
            "mode": "cadence",
            "target_months": target_months,
            "source": "query_routing",
            "metric_name": cadence_hints.get("metric_name"),
            "cadence": cadence_hints.get("cadence")
        }
        logger.info(f"[RETRIEVAL_V2]  Using cadence hints from query_routing: "
                   f"metric={time_pref['metric_name']}, cadence={time_pref['cadence']}, "
                   f"target_months={target_months[:5] if len(target_months) > 5 else target_months}")
    else:
        # PRIORITY 2: Extract from query (fallback)
        time_pref = _extract_time_preference(query)
        time_pref["source"] = "query_extraction"
        logger.info(f"[RETRIEVAL_V2]  Using time extraction from query (fallback): mode={time_pref.get('mode')}, "
                   f"target_months={time_pref.get('target_months', [])[:5]}")
    

    vector_top_k = 5  # Default: 5 docs from vector search

    # Prefer the original English user query for exact service-order identifiers and dates.
    service_order_query = original_query or query
    # Detect whether the original query contains a structured service-order request.
    original_query_metric = detect_metric_in_query(service_order_query)
    # Preserve SERVICE_ORDER when query rewriting changed punctuation or spacing in the exact identifier.
    if original_query_metric == "SERVICE_ORDER":
        detected_metric_preview = original_query_metric
    # Use the existing rewritten-query metric detection for every non-service-order product.
    else:
        detected_metric_preview = detect_metric_in_query(query)
    if not detected_metric_preview:
        # No metric detected - get 10 docs from vector search instead of 5
        vector_top_k = 10
        logger.info(f"[RETRIEVAL_V2] No metric detected, increasing vector search to {vector_top_k} docs")
    
    vector_start = time.time()
    vector_docs = vector_search_simple(
        query=query,
        vectordb=vectordb,
        rerank_function=rerank_function,  # Pass rerank function for first rerank
        top_k=vector_top_k,
        time_pref=time_pref
    )
    vector_time = time.time() - vector_start
    logger.info(f"[RETRIEVAL_V2] Vector search (with rerank + time boost): {len(vector_docs)} docs in {vector_time:.2f}s")
    
    # STEP 2: Reuse the preview result so structured and standard metric routing remain consistent.
    detected_metric = detected_metric_preview
    
    # STEP 3: Metric search (5 docs) - only if metric detected
    metric_docs = []
    metric_time = 0.0
    
    if detected_metric:
        logger.info(f"[RETRIEVAL_V2] Detected metric: {detected_metric}")
        metric_start = time.time()
        # Use the original query only for SERVICE_ORDER so exact punctuation and dates are preserved.
        metric_search_query = service_order_query if detected_metric == "SERVICE_ORDER" else query
        metric_docs = metric_search_simple(
            query=metric_search_query,
            detected_metric=detected_metric,
            qdrant_client=vectordb.client,
            collection_name=vectordb.collection_name,
            embeddings=vectordb.embeddings,  # Pass embeddings for vector search
            top_k=5,
            time_pref=time_pref  # Pass time preference for filtering
        )
        metric_time = time.time() - metric_start
        logger.info(f"[RETRIEVAL_V2] Metric search: {len(metric_docs)} docs in {metric_time:.2f}s")
    else:
        logger.info(f"[RETRIEVAL_V2] No metric detected, skipping metric search")

    # Keep an exact structured ISS/SSS order result separate from generic vector-search documents.
    if detected_metric == "SERVICE_ORDER":
        # Select only documents explicitly marked by the structured service-order matcher.
        exact_service_order_docs = [
            # Preserve the original Document object for deduplication and reranking.
            document
            # Examine every document returned by the existing metric-search stage.
            for document in metric_docs
            # Require the in-memory marker added only by _search_exact_service_order_documents.
            if document.metadata.get("_exact_service_order_match") is True
        ]

        # Bypass generic document combination when a structured match was successfully found.
        if exact_service_order_docs:
            # Remove duplicate page chunks while preserving the selected PDF or PDFs.
            exact_service_order_docs = dedupe_documents_smart(exact_service_order_docs)
            # Detect latest results because their file-level effective-date order is already final.
            is_latest_service_order_result = any(
                document.metadata.get("_latest_service_order_match") is True
                for document in exact_service_order_docs
            )
            # Detect whether the same order number/date still matched both ISS and SSS.
            service_order_is_ambiguous = any(
                # Read the ambiguity marker calculated after comparing all selected filenames.
                document.metadata.get("_service_order_ambiguous") is True
                # Check every exact service-order chunk before final reranking.
                for document in exact_service_order_docs
            )
            # Warn operators when both services matched and the answer may need user clarification.
            if service_order_is_ambiguous:
                logger.warning(
                    "[SERVICE_ORDER_SEARCH] Exact query matched both ISS and SSS; "
                    "returning balanced evidence from both services for clarification."
                )
            # Latest files are already sorted by effective date, so semantic reranking would lose recency.
            if is_latest_service_order_result:
                # Return the expected (Document, score) shape while retaining newest-to-oldest order.
                exact_service_order_results = [
                    (document, 10.0 - (index * 0.01))
                    for index, document in enumerate(exact_service_order_docs[:final_top_k])
                ]
            # Exact number/date matches still benefit from page-level semantic reranking.
            else:
                # Rerank only the exact structured chunks against the user's original retrieval query.
                exact_service_order_results = rerank_function(
                    # Use the preserved original query so exact order punctuation and dates remain available.
                    service_order_query,
                    # Do not include any unrelated generic vector-search documents.
                    exact_service_order_docs,
                    # Respect the same final result limit used by normal retrieval.
                    top_k=final_top_k,
                )
            # Record the exclusive structured return for operational debugging.
            logger.info(
                f"[SERVICE_ORDER_SEARCH] Returning {len(exact_service_order_results)} "
                f"exclusive service-order results (ambiguous={service_order_is_ambiguous})"
            )
            # Return immediately so the normal combine-and-rerank block cannot introduce unrelated sources.
            return exact_service_order_results, None, None, detected_metric

        # Identify latest type-only intent without changing exact order-number/date fallback behavior.
        latest_type_only_request = (
            _is_latest_service_order_query(service_order_query)
            and bool(_detect_service_order_type(service_order_query))
            and not _extract_service_order_keys(service_order_query)
            and not _extract_service_order_dates(service_order_query)
        )
        # A latest ISS/SSS request must not fall back to unrelated general vector-search documents.
        if latest_type_only_request:
            logger.warning(
                "[SERVICE_ORDER_SEARCH] No latest ISS/SSS document-type match found; "
                "returning no documents instead of mixing generic vector results."
            )
            # Preserve the detected metric while returning an empty, correctly scoped result.
            return [], None, None, detected_metric

    # STEP 4: Combine and dedupe
    all_docs = vector_docs + metric_docs
    deduped_docs = dedupe_documents_smart(all_docs)
    logger.info(f"[RETRIEVAL_V2] Combined: {len(all_docs)} docs, deduped: {len(deduped_docs)} docs")
    
    # STEP 4: Combine and dedupe
    all_docs = vector_docs + metric_docs
    deduped_docs = dedupe_documents_smart(all_docs)
    logger.info(f"[RETRIEVAL_V2] Combined: {len(all_docs)} docs, deduped: {len(deduped_docs)} docs")
    
    # STEP 5: Final rerank of ~10 docs
    final_rerank_time = 0.0
    if deduped_docs:
        final_rerank_start = time.time()
        reranked = rerank_function(query, deduped_docs, top_k=final_top_k)
        final_rerank_time = time.time() - final_rerank_start
        logger.info(f"[RETRIEVAL_V2] Final rerank: {len(reranked)} docs in {final_rerank_time:.2f}s")
    else:
        logger.warning(f"[RETRIEVAL_V2] No documents to rerank!")
        reranked = []
    
    # Log final results
    if ENABLE_RETRIEVAL_DEBUG and reranked:
        logger.info(f"[RETRIEVAL_V2] === FINAL RESULTS (TOP {len(reranked)}) ===")
        for i, (doc, score) in enumerate(reranked):
            meta = doc.metadata
            logger.info(f"[RETRIEVAL_V2]   {i+1}. {meta.get('file_name', 'unknown')} "
                       f"(p.{meta.get('page_number', '?')}, score={score:.4f}, "
                       f"date={meta.get('publish_date', 'no_date')})")
    
    # Log timing breakdown
    total_time = time.time() - retrieval_start
    logger.info(f"[RETRIEVAL_V2] ⏱️ TIMING BREAKDOWN: "
               f"Vector={vector_time:.2f}s | Metric={metric_time:.2f}s | "
               f"Final Rerank={final_rerank_time:.2f}s | Total={total_time:.2f}s")
    
    # Return in expected format: (reranked_with_scores, vis_docs, cadence_info, detected_metric)
    return (reranked, None, None, detected_metric)

#Sachin Upadhyay