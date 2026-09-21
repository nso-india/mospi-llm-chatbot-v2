
import os
import json
import logging
import threading
import time
import hashlib
import re
import tempfile
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger("chatbot")

WHOIS_CACHE_FILE = os.path.join(os.path.dirname(__file__), "whois_cache.txt")
WHOIS_CACHE_METADATA_FILE = os.path.join(os.path.dirname(__file__), "whois_cache_meta.txt")

# Refresh interval: 1 day (daily)
WHOIS_REFRESH_INTERVAL_DAYS = 1
WHOIS_REFRESH_INTERVAL_SECONDS = WHOIS_REFRESH_INTERVAL_DAYS * 24 * 60 * 60

# S3 configuration for Who's Who markdown files
S3_BUCKET = os.getenv("S3_BUCKET", "acct1004215-mospi")
S3_WHOIS_PREFIX = "data/mospi_web/whois_who"
S3_WHOIS_METADATA_KEY = f"{S3_WHOIS_PREFIX}/whois_metadata.json"

# S3 configuration for FOD directory PDFs
S3_FOD_PREFIX = "data/mospi_web/fod_directory/pdfs"

# Devanagari Unicode range for language validation
DEVANAGARI_PATTERN = r'[\u0900-\u097F]'

# Default metadata (used if not fetched from S3)
DEFAULT_WHOIS_METADATA: Dict[str, Any] = {
    "file_name": "whois_cache.txt",
    "file_url": "https://www.mospi.gov.in/who's-Who",
    "source": "whois_cache_file",
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
_cache_lock = threading.Lock()
_cached_whois_content: Optional[str] = None
_cached_whois_metadata: Dict[str, Any] = DEFAULT_WHOIS_METADATA.copy()
_cached_whois_doc: Optional[Document] = None
_last_refresh_time: Optional[datetime] = None
_refresh_thread: Optional[threading.Thread] = None
_stop_refresh = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def _calculate_md5(content: str) -> str:
    """Calculate MD5 checksum of content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def _validate_english_only(content: str) -> bool:
    """
    Validate that content contains only English (Latin) characters.
    Returns False if Devanagari or other non-Latin scripts are found.
    """
    # Check for Devanagari characters (Hindi script)
    if re.search(DEVANAGARI_PATTERN, content):
        logger.warning("[WHOIS_CACHE] Non-English (Devanagari) characters detected")
        return False
    
    return True


def _download_fod_pdf_from_s3() -> Optional[str]:
    """
    Download latest FOD directory PDF from S3.
    Returns local temp file path or None if failed.
    """
    try:
        from web_scrap.s3_helper import get_s3_client, download_file_from_s3
        
        s3_client = get_s3_client()
        
        logger.info(f"[FOD] Listing PDFs in s3://{S3_BUCKET}/{S3_FOD_PREFIX}/")
        
        # List all PDFs in FOD directory using paginator
        paginator = s3_client.get_paginator('list_objects_v2')
        files = []
        
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_FOD_PREFIX):
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                key = obj['Key']
                # Only add PDF files (exclude directories)
                if key.lower().endswith('.pdf'):
                    files.append(key)
        
        if not files:
            logger.error(f"[FOD] No PDF files found in {S3_FOD_PREFIX}")
            return None
        
        # Sort by key (assuming newer files have later names)
        files.sort(reverse=True)
        latest_pdf = files[0]
        
        logger.info(f"[FOD] Latest PDF: {latest_pdf}")
        
        # Download to temp file
        temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        temp_pdf_path = temp_pdf.name
        temp_pdf.close()
        
        logger.info(f"[FOD] Downloading {latest_pdf}...")
        success = download_file_from_s3(
            s3_client,
            S3_BUCKET,
            latest_pdf,
            temp_pdf_path
        )
        
        if not success:
            logger.error(f"[FOD] Failed to download {latest_pdf}")
            return None
        
        logger.info(f"[FOD] Downloaded to {temp_pdf_path}")
        return temp_pdf_path
        
    except Exception as e:
        logger.error(f"[FOD] Error downloading from S3: {e}", exc_info=True)
        return None


def _convert_fod_pdf_to_markdown(pdf_path: str) -> Optional[tuple]:
    """
    Convert FOD PDF to markdown using batch_process_pdf_to_text.py (NuMarkdown).
    Also uploads the markdown to S3.
    
    Returns:
        tuple: (markdown_content: str, markdown_filename: str) or (None, None) if failed
    """
    try:
        import subprocess
        import sys
        import glob
        
        logger.info(f"[FOD] Converting PDF to markdown using NuMarkdown...")

        api_url = os.environ.get("PDF_CONVERTER_VLLM_URL") or os.getenv("PDF_CONVERTER_VLLM_URL", "http://10.75.8.1:8002/v1")
        
        # Debug logging
        logger.info(f"[FOD] Environment check:")
        logger.info(f"[FOD]   - os.environ.get('PDF_CONVERTER_VLLM_URL'): {os.environ.get('PDF_CONVERTER_VLLM_URL')}")
        logger.info(f"[FOD]   - os.getenv('PDF_CONVERTER_VLLM_URL'): {os.getenv('PDF_CONVERTER_VLLM_URL')}")
        logger.info(f"[FOD]   - Final API URL to use: {api_url}")
        
        # Create temp output directory
        temp_output_dir = tempfile.mkdtemp(prefix="fod_markdown_")

        temp_log_file = os.path.join(temp_output_dir, "fod_processed_log.jsonl")

        # Call batch_process_pdf_to_text.py
        # Use --local-mode to process single PDF
        cmd = [
            sys.executable,
            "batch_process_pdf_to_text.py",
            "--local-mode",
            "--input-dir", os.path.dirname(pdf_path),
            "--output-dir", temp_output_dir,
            "--api-url", api_url,  # Pass the correct API URL
            "--max-concurrent", "5",  # Lower concurrency for single PDF
            "--log-file", temp_log_file  # Isolate log from the shared pipeline log
        ]
        
        logger.info(f"[FOD] Running: {' '.join(cmd)}")
        logger.info(f"[FOD] Input PDF: {pdf_path}")
        logger.info(f"[FOD] Input dir: {os.path.dirname(pdf_path)}")
        logger.info(f"[FOD] Output dir: {temp_output_dir}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Log stdout and stderr regardless of exit code
        if result.stdout:
            logger.info(f"[FOD] batch_process stdout: {result.stdout[:1000]}")  # First 1000 chars
        if result.stderr:
            logger.error(f"[FOD] batch_process stderr: {result.stderr[:1000]}")
        
        if result.returncode != 0:
            logger.error(f"[FOD] PDF conversion failed with exit code {result.returncode}")
            return None, None
        
        # Find the output markdown file (should be only .md file in output dir)
        md_files = glob.glob(os.path.join(temp_output_dir, "*.md"))
        
        if not md_files:
            logger.error(f"[FOD] No markdown files found in {temp_output_dir}")
            # List directory contents to debug
            try:
                contents = os.listdir(temp_output_dir)
                logger.error(f"[FOD] Directory contents: {contents}")
                
                # Also check if processed_pdf_to_text_files.jsonl has entries
                if os.path.exists('processed_pdf_to_text_files.jsonl'):
                    with open('processed_pdf_to_text_files.jsonl', 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_entry = lines[-1]
                            logger.info(f"[FOD] Last processed entry: {last_entry[:200]}")
            except Exception as e:
                logger.error(f"[FOD] Error listing directory: {e}")
            
            return None, None
        
        # Use the first (and should be only) markdown file
        md_path = md_files[0]
        logger.info(f"[FOD] Found markdown file: {md_path}")
        
        # Read markdown content
        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        logger.info(f"[FOD] Conversion successful: {len(markdown_content)} chars")
        
        # Upload to S3
        markdown_filename = _upload_fod_markdown_to_s3(markdown_content)
        
        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_output_dir, ignore_errors=True)
        
        return markdown_content, markdown_filename
        
    except subprocess.TimeoutExpired:
        logger.error(f"[FOD] PDF conversion timed out after 10 minutes")
        return None, None
    except Exception as e:
        logger.error(f"[FOD] Error converting PDF: {e}", exc_info=True)
        return None, None


def _upload_fod_markdown_to_s3(markdown_content: str) -> Optional[str]:

    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_scrap'))
        from s3_helper import get_s3_client, upload_file_to_s3
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fod_directory_{timestamp}.md"
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.md', delete=False)
        temp_file.write(markdown_content)
        temp_file.close()
        
        # Upload to S3
        s3_client = get_s3_client()
        s3_key = f"{S3_FOD_PREFIX}/{filename}"
        
        logger.info(f"[FOD] Uploading markdown to s3://{S3_BUCKET}/{s3_key}...")
        result = upload_file_to_s3(s3_client, temp_file.name, S3_BUCKET, s3_key)
        
        # Cleanup temp file
        os.unlink(temp_file.name)
        
        if result["success"]:
            logger.info(f"[FOD] Uploaded successfully: {result['file_size_formatted']}")
            return filename
        else:
            logger.error(f"[FOD] Upload failed: {result['error']}")
            return None
            
    except Exception as e:
        logger.error(f"[FOD] Error uploading markdown to S3: {e}", exc_info=True)
        return None


def _parse_fod_markdown(markdown_content: str) -> tuple:
    try:
        lines = []
        current_zone = None
        zone_markers = {}  # zone_name -> line_number in output
        in_table = False
        header_written = False
        header_row = None
        
        for line in markdown_content.split('\n'):
            original_line = line
            line = line.strip()
            

            if line.startswith('##') and not line.startswith('###'):
                zone_name = line.replace('##', '').strip()
                current_zone = zone_name
                zone_markers[zone_name] = len(lines)
                lines.append(f"\n## {zone_name}")
                logger.info(f"[FOD] Found zone marker: {zone_name}")
                continue
            

            if '|' not in line and line:
                # Check for zone-related keywords
                zone_keywords = ['zone', 'wing', 'hqrs', 'headquarters', 'division (fod)']
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in zone_keywords):
                    # This might be a zone header without ## prefix
                    zone_name = line.strip()
                    current_zone = zone_name
                    zone_markers[zone_name] = len(lines)
                    lines.append(f"\n## {zone_name}")
                    logger.info(f"[FOD] Found zone marker (keyword match): {zone_name}")
                    continue
            
            # Skip empty lines, page markers, and separator rows initially
            if not line or line.startswith('---') or line.startswith('==Page'):
                continue
            
            # Check if this is a table row
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                
                # Remove empty first/last elements (from leading/trailing |)
                if parts and parts[0] == '':
                    parts = parts[1:]
                if parts and parts[-1] == '':
                    parts = parts[:-1]
                
                # Skip separator rows (---)
                if parts and all(p.replace('-', '').strip() == '' for p in parts):
                    in_table = True
                    continue
                
                # Detect header row (contains "Name", "Designation", etc.)
                if not header_written and len(parts) >= 5:
                    # Check if this looks like a header
                    header_keywords = ['name', 'designation', 'divion', 'division', 'tel', 'email', 'e-mail', 'address']
                    if any(keyword in '|'.join(parts).lower() for keyword in header_keywords):
                        header_row = parts
                        # Normalize header and add FOD_Zone column
                        normalized_header = _normalize_fod_header(parts)
                        # Add FOD_Zone as 7th column
                        normalized_header.append('FOD_Zone')
                        lines.append('|'.join(normalized_header))
                        header_written = True
                        in_table = True
                        logger.info(f"[FOD] Header detected: {parts}")
                        logger.info(f"[FOD] Normalized to: {normalized_header}")
                        continue
                
                # Process data rows (must have same number of columns as header)
                if in_table and header_written and header_row:
                    # Flexible column count (at least 5 columns for FOD data)
                    if len(parts) >= 5:
                        # Normalize data row to match header structure
                        normalized_row = _normalize_fod_data_row(parts, header_row)
                        # Add current zone as 7th column
                        zone_value = current_zone if current_zone else ""
                        normalized_row.append(zone_value)
                        lines.append('|'.join(normalized_row))
        
        if not lines:
            logger.error("[FOD] No table data extracted from markdown")
            return None, None
        
        content = '\n'.join(lines)
        logger.info(f"[FOD] Extracted {len([l for l in lines if l and not l.startswith('#')])} data rows with zone info")
        return content, zone_markers
        
    except Exception as e:
        logger.error(f"[FOD] Error parsing markdown: {e}", exc_info=True)
        return None, None


def _normalize_fod_header(header_parts: list) -> list:
    normalized = []
    
    for part in header_parts:
        part_lower = part.lower().strip()
        
        # Name column
        if 'name' in part_lower:
            normalized.append('Name')
        # Designation column
        elif 'designation' in part_lower:
            normalized.append('Designation')
        # Division column (fix typo)
        elif 'divion' in part_lower or 'division' in part_lower:
            normalized.append('Division')
        # Contact column
        elif 'tel' in part_lower or 'phone' in part_lower or 'contact' in part_lower or 'office no' in part_lower:
            normalized.append('Contact No.')
        # Email column
        elif 'email' in part_lower or 'e-mail' in part_lower or 'mail' in part_lower:
            normalized.append('Email ID')
        # Address column
        elif 'address' in part_lower:
            normalized.append('Address')
        else:
            # Keep unknown columns as-is but log them
            logger.warning(f"[FOD] Unknown header column: {part}")
            normalized.append(part)
    
    return normalized


def _normalize_fod_data_row(data_parts: list, header_parts: list) -> list:

    # If we have exactly 6 columns, return as-is
    if len(data_parts) == 6:
        return data_parts
    
    # If we have 5 columns, it might be missing Address - add empty
    if len(data_parts) == 5:
        return data_parts + ['']
    

    if len(data_parts) > 6:
        # Try to merge contact numbers (columns 3 and 4 typically)
        normalized = data_parts[:3]  # Name, Designation, Division
        
        # Merge contact numbers
        contact_parts = []
        for i in range(3, len(data_parts)):
            if i < len(data_parts) - 2:  # Not email or address
                contact_parts.append(data_parts[i])
            else:
                break
        normalized.append(' '.join(filter(None, contact_parts)))
        
        # Add email and address
        if len(data_parts) > len(contact_parts) + 3:
            normalized.append(data_parts[len(contact_parts) + 3])  # Email
        else:
            normalized.append('')
        
        if len(data_parts) > len(contact_parts) + 4:
            normalized.append(data_parts[len(contact_parts) + 4])  # Address
        else:
            normalized.append('')
        
        return normalized[:6]  # Ensure exactly 6 columns
    
    # If less than 5, pad with empty strings
    return data_parts + [''] * (6 - len(data_parts))


def _merge_whois_and_fod(whois_content: str, fod_content: str) -> str:
    merged_lines = []
    
    # Add Who's Who section
    merged_lines.append("## WHO'S WHO SECTION")
    merged_lines.append(whois_content.strip())
    merged_lines.append("")  # Blank line separator
    
    # Add FOD section
    merged_lines.append("## FOD DIRECTORY SECTION")
    merged_lines.append(fod_content.strip())
    
    return '\n'.join(merged_lines)


def _convert_markdown_to_pipe_delimited(markdown_content: str) -> str:
    lines = []
    in_table = False
    header_written = False
    column_map = None  # list of canonical column names aligned to source columns, once header is parsed

    canonical_columns = ['Name', 'Designation', 'Division', 'Contact No.', 'Email ID', 'Address']

    def _classify_header_cell(cell: str) -> Optional[str]:
        cell_lower = cell.lower().strip()
        if 'seniority' in cell_lower:
            return None  # dropped column
        if 'name' in cell_lower:
            return 'Name'
        if 'designation' in cell_lower:
            return 'Designation'
        if 'divion' in cell_lower or 'division' in cell_lower:
            return 'Division'
        if 'contact' in cell_lower or 'tel' in cell_lower or 'phone' in cell_lower:
            return 'Contact No.'
        if 'email' in cell_lower or 'e-mail' in cell_lower or 'mail' in cell_lower:
            return 'Email ID'
        if 'address' in cell_lower:
            return 'Address'
        return None

    for line in markdown_content.split('\n'):
        line = line.strip()

        # Skip empty lines and markdown headers
        if not line or line.startswith('#') or line.startswith('---') or line.startswith('Scraped on:'):
            continue

        # Check if this is a table row
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]

            # Remove empty first/last elements (from leading/trailing |)
            if parts and parts[0] == '':
                parts = parts[1:]
            if parts and parts[-1] == '':
                parts = parts[:-1]

            # Skip separator rows (---)
            if parts and all(p.replace('-', '').strip() == '' for p in parts):
                in_table = True
                continue

            # Process table header (accept 6 or 7 columns)
            if not header_written and len(parts) in (6, 7):
                classified = [_classify_header_cell(p) for p in parts]
                # Must recognize Name and Designation at minimum to treat as a real header
                if 'Name' in classified and 'Designation' in classified:
                    column_map = classified
                    lines.append('|'.join(canonical_columns))
                    header_written = True
                    in_table = True
                    continue

            # Process data rows using the column mapping discovered from the header
            if in_table and header_written and column_map and len(parts) == len(column_map):
                row = {}
                for canonical_name, value in zip(column_map, parts):
                    if canonical_name is None:
                        continue  # dropped column (e.g. Seniority Order)
                    row[canonical_name] = value
                # Only emit rows that have at least Name and Designation populated
                if row.get('Name') and row.get('Designation'):
                    ordered_values = [row.get(col, '') for col in canonical_columns]
                    lines.append('|'.join(ordered_values))

    return '\n'.join(lines)


def _download_whois_metadata_from_s3() -> Optional[list]:
    """Download whois_metadata.json from S3."""
    try:
        from web_scrap.s3_helper import get_s3_client, download_file_from_s3
        
        s3_client = get_s3_client()
        
        # Download to temp file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp:
            temp_path = tmp.name
        
        logger.info(f"[WHOIS_CACHE] Downloading metadata from s3://{S3_BUCKET}/{S3_WHOIS_METADATA_KEY}")
        success = download_file_from_s3(
            s3_client,
            S3_BUCKET,
            S3_WHOIS_METADATA_KEY,
            temp_path
        )
        
        if not success:
            logger.warning("[WHOIS_CACHE] Failed to download whois_metadata.json from S3")
            return None
        
        # Read metadata
        with open(temp_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        logger.info(f"[WHOIS_CACHE] Downloaded metadata with {len(metadata)} entries")
        return metadata
        
    except Exception as e:
        logger.error(f"[WHOIS_CACHE] Error downloading metadata from S3: {e}")
        return None


def _find_latest_whois_file(metadata: list) -> Optional[Dict[str, Any]]:
    """Find the latest whois markdown file from metadata."""
    if not metadata:
        return None
    
    try:
        # Sort by scraped_at timestamp (descending)
        sorted_metadata = sorted(
            metadata,
            key=lambda x: x.get('scraped_at', ''),
            reverse=True
        )
        
        latest = sorted_metadata[0]
        logger.info(f"[WHOIS_CACHE] Latest file: {latest.get('file_name')} (scraped: {latest.get('scraped_at')})")
        return latest
        
    except Exception as e:
        logger.error(f"[WHOIS_CACHE] Error finding latest file: {e}")
        return None


def _download_whois_markdown_from_s3(file_name: str) -> Optional[str]:
    """Download specific whois markdown file from S3."""
    try:
        from web_scrap.s3_helper import get_s3_client, download_file_from_s3
        
        s3_client = get_s3_client()
        s3_key = f"{S3_WHOIS_PREFIX}/{file_name}"
        
        # Download to temp file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.md') as tmp:
            temp_path = tmp.name
        
        logger.info(f"[WHOIS_CACHE] Downloading markdown from s3://{S3_BUCKET}/{s3_key}")
        success = download_file_from_s3(
            s3_client,
            S3_BUCKET,
            s3_key,
            temp_path
        )
        
        if not success:
            logger.warning(f"[WHOIS_CACHE] Failed to download {file_name} from S3")
            return None
        
        # Read content
        with open(temp_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean up temp file
        os.unlink(temp_path)
        
        logger.info(f"[WHOIS_CACHE] Downloaded markdown file: {len(content)} chars")
        return content
        
    except Exception as e:
        logger.error(f"[WHOIS_CACHE] Error downloading markdown from S3: {e}")
        return None


def _load_from_file() -> Optional[str]:
    """Load whois content from local cache file."""
    try:
        if os.path.exists(WHOIS_CACHE_FILE):
            with open(WHOIS_CACHE_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content and not content.startswith("PLACEHOLDER"):
                    logger.info(f"[WHOIS_CACHE] Loaded from file: {len(content)} chars")
                    return content
        logger.warning(f"[WHOIS_CACHE] Cache file not found or empty: {WHOIS_CACHE_FILE}")
    except Exception as e:
        logger.warning(f"[WHOIS_CACHE] Failed to load from file: {e}")
    return None


def _save_to_file(content: str, metadata: Dict[str, Any]) -> bool:
    """Save whois content to local cache file."""
    try:
        with open(WHOIS_CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Calculate MD5 checksum
        md5_checksum = _calculate_md5(content)
        
        # Save metadata with timestamp and additional info
        meta_content = f"last_updated: {datetime.now().isoformat()}\n"
        meta_content += f"integration_type: {metadata.get('integration_type', 'whois_only')}\n"
        meta_content += f"whois_source: {metadata.get('whois_source', metadata.get('source_file', 'unknown'))}\n"
        meta_content += f"whois_scraped_at: {metadata.get('whois_scraped_at', metadata.get('scraped_at', 'unknown'))}\n"
        meta_content += f"fod_source: {metadata.get('fod_source', 'unavailable')}\n"
        
        # Add FOD zone markers if available
        fod_zones = metadata.get('fod_zones', [])
        if fod_zones:
            meta_content += f"fod_zones: {', '.join(fod_zones)}\n"
        
        meta_content += f"md5_checksum: {md5_checksum}\n"
        meta_content += f"language_validation: {metadata.get('language_validation', 'unknown')}\n"
        meta_content += f"entry_count: {metadata.get('entry_count', 0)}\n"
        
        with open(WHOIS_CACHE_METADATA_FILE, "w", encoding="utf-8") as f:
            f.write(meta_content)
        
        logger.info(f"[WHOIS_CACHE] Saved to file: {len(content)} chars, MD5: {md5_checksum[:8]}...")
        return True
    except Exception as e:
        logger.error(f"[WHOIS_CACHE] Failed to save to file: {e}")
        return False


def _get_last_file_update() -> Optional[datetime]:
    """Get the last update time from metadata file."""
    try:
        if os.path.exists(WHOIS_CACHE_METADATA_FILE):
            with open(WHOIS_CACHE_METADATA_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("last_updated:"):
                        ts = line.split(":", 1)[1].strip()
                        return datetime.fromisoformat(ts)
    except Exception as e:
        logger.warning(f"[WHOIS_CACHE] Failed to read metadata: {e}")
    return None


def _fetch_from_s3() -> Optional[tuple]:
    """
    Fetch whois content from S3 and integrate with FOD directory.
    Returns (content, metadata) tuple or None if failed.
    
    This runs in background with validation checks.
    Integrates both Who's Who and FOD Directory data.
    """
    try:
        # ========== STEP 1: FETCH WHO'S WHO DATA ==========
        logger.info("[WHOIS_CACHE] Fetching Who's Who data from S3...")
        
        # Step 1.1: Download metadata
        metadata_list = _download_whois_metadata_from_s3()
        if not metadata_list:
            logger.warning("[WHOIS_CACHE] No Who's Who metadata found in S3")
            # Continue anyway, try to get FOD data
            whois_pipe_content = None
            whois_meta = {}
        else:
            # Step 1.2: Find latest file
            latest_file_meta = _find_latest_whois_file(metadata_list)
            if not latest_file_meta:
                logger.warning("[WHOIS_CACHE] Could not find latest Who's Who file in metadata")
                whois_pipe_content = None
                whois_meta = {}
            else:
                file_name = latest_file_meta.get('file_name')
                if not file_name:
                    logger.warning("[WHOIS_CACHE] Latest Who's Who file has no file_name")
                    whois_pipe_content = None
                    whois_meta = {}
                else:
                    # Step 1.3: Download markdown file
                    markdown_content = _download_whois_markdown_from_s3(file_name)
                    if not markdown_content:
                        logger.warning(f"[WHOIS_CACHE] Could not download Who's Who {file_name}")
                        whois_pipe_content = None
                        whois_meta = {}
                    else:
                        # Step 1.4: Validate English only
                        if not _validate_english_only(markdown_content):
                            logger.error(f"[WHOIS_CACHE] Language validation failed for Who's Who {file_name}")
                            whois_pipe_content = None
                            whois_meta = {}
                        else:
                            # Step 1.5: Convert to pipe-delimited format
                            logger.info("[WHOIS_CACHE] Converting Who's Who markdown to pipe-delimited format...")
                            whois_pipe_content = _convert_markdown_to_pipe_delimited(markdown_content)
                            
                            if not whois_pipe_content:
                                logger.error("[WHOIS_CACHE] Who's Who conversion resulted in empty content")
                                whois_pipe_content = None
                                whois_meta = {}
                            else:
                                whois_meta = {
                                    "source_file": file_name,
                                    "file_url": latest_file_meta.get('file_url', 'https://www.mospi.gov.in/who\'s-Who'),
                                    "scraped_at": latest_file_meta.get('scraped_at', ''),
                                }
                                logger.info(f"[WHOIS_CACHE] Who's Who data ready: {len(whois_pipe_content)} chars")
        
        # ========== STEP 2: FETCH FOD DIRECTORY DATA ==========
        logger.info("[WHOIS_CACHE] Fetching FOD directory data from S3...")
        
        temp_pdf_path = None
        fod_pipe_content = None
        fod_meta = {}
        
        try:
            # Step 2.1: Download FOD PDF
            temp_pdf_path = _download_fod_pdf_from_s3()
            
            if not temp_pdf_path:
                logger.warning("[WHOIS_CACHE] No FOD PDF found in S3, skipping FOD integration")
            else:
                # Step 2.2: Convert PDF to markdown and upload to S3
                fod_markdown, fod_markdown_filename = _convert_fod_pdf_to_markdown(temp_pdf_path)
                
                if not fod_markdown:
                    logger.warning("[WHOIS_CACHE] FOD PDF conversion failed, skipping FOD integration")
                else:
                    # Step 2.3: Validate English only
                    if not _validate_english_only(fod_markdown):
                        logger.error("[WHOIS_CACHE] Language validation failed for FOD directory")
                    else:
                        # Step 2.4: Parse FOD markdown and extract tables
                        logger.info("[WHOIS_CACHE] Parsing FOD markdown...")
                        fod_pipe_content, zone_markers = _parse_fod_markdown(fod_markdown)
                        
                        if not fod_pipe_content:
                            logger.warning("[WHOIS_CACHE] FOD parsing resulted in empty content")
                        else:
                            fod_meta = {
                                "source_file": fod_markdown_filename or os.path.basename(temp_pdf_path),
                                "zone_markers": list(zone_markers.keys()) if zone_markers else [],
                            }
                            logger.info(f"[WHOIS_CACHE] FOD data ready: {len(fod_pipe_content)} chars, {len(zone_markers)} zones")
        
        finally:
            # Cleanup temp PDF file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.unlink(temp_pdf_path)
                    logger.info(f"[WHOIS_CACHE] Cleaned up temp PDF: {temp_pdf_path}")
                except Exception:
                    pass
        
        # ========== STEP 3: MERGE WHO'S WHO + FOD ==========
        if not whois_pipe_content and not fod_pipe_content:
            logger.error("[WHOIS_CACHE] Both Who's Who and FOD data unavailable")
            return None
        
        # Merge content
        if whois_pipe_content and fod_pipe_content:
            logger.info("[WHOIS_CACHE] Merging Who's Who + FOD data...")
            final_content = _merge_whois_and_fod(whois_pipe_content, fod_pipe_content)
            entry_count = len([l for l in final_content.split('\n') if l.strip() and not l.startswith('#')]) - 2  # Subtract 2 headers
        elif whois_pipe_content:
            logger.info("[WHOIS_CACHE] Using Who's Who data only (FOD unavailable)")
            final_content = whois_pipe_content
            entry_count = len([l for l in final_content.split('\n') if l.strip()]) - 1  # Subtract header
        else:
            logger.info("[WHOIS_CACHE] Using FOD data only (Who's Who unavailable)")
            final_content = fod_pipe_content
            entry_count = len([l for l in final_content.split('\n') if l.strip() and not l.startswith('#')]) - 1  # Subtract header
        
        # ========== STEP 4: CHECK MD5 AND PREPARE METADATA ==========
        new_md5 = _calculate_md5(final_content)
        
        # Try to get current cache MD5
        current_md5 = None
        try:
            if os.path.exists(WHOIS_CACHE_FILE):
                with open(WHOIS_CACHE_FILE, 'r', encoding='utf-8') as f:
                    current_content = f.read()
                    current_md5 = _calculate_md5(current_content)
        except Exception as e:
            logger.warning(f"[WHOIS_CACHE] Could not read current cache for comparison: {e}")
        
        # Compare MD5
        if current_md5 and current_md5 == new_md5:
            logger.info(f"[WHOIS_CACHE] No changes detected (MD5: {new_md5[:8]}...)")
            return None
        
        logger.info(f"[WHOIS_CACHE] Content changed (old MD5: {current_md5[:8] if current_md5 else 'none'}..., new MD5: {new_md5[:8]}...)")
        
        # Prepare combined metadata
        metadata = {
            "whois_source": whois_meta.get('source_file', 'unavailable'),
            "whois_url": whois_meta.get('file_url', 'https://www.mospi.gov.in/who\'s-Who'),
            "whois_scraped_at": whois_meta.get('scraped_at', ''),
            "fod_source": fod_meta.get('source_file', 'unavailable'),
            "fod_zones": fod_meta.get('zone_markers', []),
            "entry_count": entry_count,
            "language_validation": "passed",
            "md5_checksum": new_md5,
            "integration_type": "whois_and_fod",
        }
        
        logger.info(f"[WHOIS_CACHE] Fetched from S3: {len(final_content)} chars, {entry_count} total entries")
        return (final_content, metadata)
        
    except Exception as e:
        logger.error(f"[WHOIS_CACHE] S3 fetch failed: {e}", exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CACHE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
def _update_cache(content: str, metadata: Dict[str, Any]):
    """Update the in-memory cache (thread-safe)."""
    global _cached_whois_content, _cached_whois_metadata, _cached_whois_doc, _last_refresh_time
    
    with _cache_lock:
        _cached_whois_content = content
        _cached_whois_metadata = metadata
        _cached_whois_doc = Document(
            page_content=content,
            metadata=metadata.copy()
        )
        _last_refresh_time = datetime.now()
        logger.info(f"[WHOIS_CACHE] Cache updated: {len(content)} chars")


def initialize_whois_cache():

    global _last_refresh_time
    
    # Try loading from file first
    content = _load_from_file()
    
    if content:
        _update_cache(content, DEFAULT_WHOIS_METADATA)
        _last_refresh_time = _get_last_file_update() or datetime.now()
        logger.info("[WHOIS_CACHE] Initialized from file")
    else:
        logger.warning("[WHOIS_CACHE] No cached content available at startup")


def refresh_whois_cache() -> bool:
    result = _fetch_from_s3()
    
    if result:
        content, metadata = result
        _update_cache(content, metadata)
        _save_to_file(content, metadata)
        return True
    
    return False


def _background_refresh_loop():
    """Background thread that periodically refreshes the cache."""
    logger.info(f"[WHOIS_CACHE] Background refresh thread started (interval: {WHOIS_REFRESH_INTERVAL_DAYS} days)")
    
    while not _stop_refresh.is_set():
        # Check if refresh is needed
        should_refresh = False
        
        with _cache_lock:
            if _last_refresh_time is None:
                should_refresh = True
            elif datetime.now() - _last_refresh_time > timedelta(days=WHOIS_REFRESH_INTERVAL_DAYS):
                should_refresh = True
        
        if should_refresh:
            logger.info("[WHOIS_CACHE] Starting scheduled refresh...")
            success = refresh_whois_cache()
            if success:
                logger.info("[WHOIS_CACHE] Scheduled refresh completed successfully")
            else:
                logger.warning("[WHOIS_CACHE] Scheduled refresh failed, will retry later")
        
        # Sleep for 1 day, checking stop flag periodically
        for _ in range(24 * 60):  # Check every minute for 24 hours
            if _stop_refresh.is_set():
                break
            time.sleep(60)
    
    logger.info("[WHOIS_CACHE] Background refresh thread stopped")


def start_whois_cache_refresh_task():

    # Initialize cache from local file only
    initialize_whois_cache()

    if not get_cached_whois_content():
        logger.warning(
            "[WHOIS_CACHE] Cache is EMPTY after loading from file. "
            "Triggering a one-time background refresh from S3 to self-heal "
            "(whois queries would otherwise fall back to Qdrant)."
        )

        def _one_time_startup_refresh():
            try:
                if refresh_whois_cache():
                    logger.info("[WHOIS_CACHE] One-time startup refresh succeeded; cache populated from S3.")
                else:
                    logger.error(
                        "[WHOIS_CACHE] One-time startup refresh did not populate the cache "
                        "(no S3 content or unchanged). Whois queries may fall back to Qdrant "
                        "until the next scheduled refresh."
                    )
            except Exception as e:
                logger.error(f"[WHOIS_CACHE] One-time startup refresh failed: {e}", exc_info=True)

        threading.Thread(target=_one_time_startup_refresh, daemon=True).start()
    else:
        logger.info(
            "[WHOIS_CACHE] Cache initialized from file. "
            "Auto-refresh is handled by the scheduled daily job (internal thread disabled)."
        )


def start_internal_background_refresh():

    global _refresh_thread

    # Initialize cache first
    initialize_whois_cache()

    # Start background thread
    _stop_refresh.clear()
    _refresh_thread = threading.Thread(target=_background_refresh_loop, daemon=True)
    _refresh_thread.start()
    logger.info("[WHOIS_CACHE] Internal background refresh thread started")


def stop_whois_cache_refresh_task():
    """Stop the background refresh task."""
    _stop_refresh.set()
    if _refresh_thread and _refresh_thread.is_alive():
        _refresh_thread.join(timeout=5)
    logger.info("[WHOIS_CACHE] Background refresh task stopped")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def get_cached_whois_doc() -> Optional[Document]:

    with _cache_lock:
        return _cached_whois_doc


def get_cached_whois_content() -> Optional[str]:

    with _cache_lock:
        return _cached_whois_content


def get_cache_status() -> Dict[str, Any]:

    with _cache_lock:
        return {
            "has_content": _cached_whois_content is not None,
            "content_length": len(_cached_whois_content) if _cached_whois_content else 0,
            "last_refresh": _last_refresh_time.isoformat() if _last_refresh_time else None,
            "refresh_interval_days": WHOIS_REFRESH_INTERVAL_DAYS,
        }


def force_refresh() -> bool:
    """Force an immediate cache refresh. Returns True if successful."""
    logger.info("[WHOIS_CACHE] Force refresh requested")
    return refresh_whois_cache()
