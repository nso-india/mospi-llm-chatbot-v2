#!/usr/bin/env python3
"""
Batch Processor V3 - Streaming Pipeline for H200 (Fixed for PyMuPDF)

Optimized for:
- Single H200 GPU with high concurrency
- Local file processing (no S3)
- Continuous streaming (no batch waiting)
- Maximum GPU utilization

Architecture:
- Maintains pool of concurrent requests (default: 20)
- As soon as a page completes, immediately sends next page
- No waiting for batch completion
- Processes one PDF at a time with optional overlap at end
- PyMuPDF operations run in main thread (no threading support)

IMPORTANT: PyMuPDF does NOT support threading. All PDF operations
run synchronously in the main event loop thread.

Usage:
    python batch_process_numarkdown_v3.py [--max-concurrent 20]
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict
import aiohttp
import fitz  # pymupdf


# Connection configuration
CONNECTION_POOL_LIMIT = 50  # Max concurrent connections
REQUEST_PACING_DELAY = 0.5  # 500ms delay between request starts (increased from 50ms)
CONNECT_TIMEOUT = 30  # 30s to establish connection
READ_TIMEOUT = 300  # 5 minutes to read response (increased from 180s)
TOTAL_TIMEOUT = 600  # 10 minutes total per request (increased from 300s)


# Setup logging to logs/ directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(funcName)s | %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'pdf_conversion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


def pdf_page_to_base64(pdf_path: Path, page_num: int, dpi: int = 300) -> Optional[tuple]:
    """
    Convert PDF page to base64-encoded image. Handles OCG errors with alpha=False.
    Optimized: Opens PDF once per call (caller should batch if needed).
    Returns tuple of (base64_string, format) or (None, None) on error.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Render page to pixmap
        # alpha=False handles "No default Layer config" OCG errors
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # For large images, use JPEG compression to reduce payload size
        if pix.width * pix.height > 1000000:  # > 1 megapixel
            img_bytes = pix.tobytes("jpeg")  # Removed jpg_quality for PyMuPDF 1.20.2 compatibility
            img_format = "jpeg"
        else:
            img_bytes = pix.tobytes("png")
            img_format = "png"
        
        doc.close()
        
        # Encode to base64
        return base64.b64encode(img_bytes).decode('utf-8'), img_format
    except Exception as e:
        logging.error(f"Error converting page {page_num} to image: {e}")
        return None, None


def pdf_page_to_base64_fast(pdf_doc: fitz.Document, page_num: int, dpi: int = 300) -> Optional[str]:
    """
    Fast version: Takes already-open PDF document.
    Use this when processing multiple pages from same PDF.
    Optimized to reduce payload size with JPEG compression for large images.
    """
    try:
        page = pdf_doc[page_num]
        
        # Render page to pixmap
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # For large images, use JPEG compression to reduce payload size
        # PNG can be 5-10MB, JPEG can be 500KB-1MB (10x smaller)
        # Using default JPEG quality (compatible with PyMuPDF 1.20.2)
        if pix.width * pix.height > 1000000:  # > 1 megapixel
            img_bytes = pix.tobytes("jpeg")  # Removed jpg_quality for PyMuPDF 1.20.2 compatibility
            img_format = "jpeg"
        else:
            img_bytes = pix.tobytes("png")
            img_format = "png"
        
        return base64.b64encode(img_bytes).decode('utf-8'), img_format
    except Exception as e:
        logging.error(f"Error converting page {page_num} to image: {e}")
        return None, None


async def process_page_streaming(
    session: aiohttp.ClientSession,
    pdf_path: Path,
    page_num: int,
    temperature: float,
    api_url: str,
    max_retries: int = 3
) -> Optional[str]:
    """Process a single PDF page with retries. Returns immediately when done."""
    for attempt in range(max_retries):
        try:
            # Convert page to base64 image
            result = pdf_page_to_base64(pdf_path, page_num)
            if result is None or result[0] is None:
                return None
            
            base64_image, img_format = result
            data_url = f"data:image/{img_format};base64,{base64_image}"
            
            # Call NuMarkdown API
            payload = {
                "model": "numind/NuMarkdown-8B-Thinking",
                "temperature": temperature,
                "max_tokens": 8192,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                                "min_pixels": 100 * 28 * 28,
                                "max_pixels": 5000 * 28 * 28,
                            },
                        ],
                    },
                ]
            }
            
            # Configure timeout
            timeout = aiohttp.ClientTimeout(
                total=TOTAL_TIMEOUT,
                connect=CONNECT_TIMEOUT,
                sock_read=READ_TIMEOUT
            )
            
            async with session.post(
                f"{api_url}/chat/completions",
                json=payload,
                timeout=timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Check if response has expected structure
                    if 'choices' not in result or not result['choices']:
                        logging.warning(f"Empty choices in response for page {page_num + 1}")
                        return ""
                    
                    content = result['choices'][0]['message']['content']
                    
                    # Handle empty or null content gracefully
                    if content is None or not content or content.strip() == "":
                        logging.info(f"Empty/null content for page {page_num + 1}, using empty string")
                        return ""
                    
                    # Extract answer from tags
                    if "<answer>" in content and "</answer>" in content:
                        answer = content.split("<answer>")[1].split("</answer>")[0].strip()
                    else:
                        answer = content
                    
                    return answer if answer else ""
                else:
                    error_body = await response.text()
                    logging.error(f"API error for page {page_num + 1} (attempt {attempt + 1}): HTTP {response.status}")
                    logging.error(f"  Response body: {error_body[:500]}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None
        
        except aiohttp.ClientOSError as e:
            # Handle specific connection errors
            if hasattr(e, 'errno') and e.errno == 32:  # Broken pipe
                logging.warning(f"Broken pipe for page {page_num + 1} (attempt {attempt + 1}), retrying...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)  # Longer wait for broken pipe
                    continue
            elif "Server disconnected" in str(e):
                logging.warning(f"Server disconnected for page {page_num + 1} (attempt {attempt + 1}), retrying...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
                    continue
            else:
                logging.warning(f"Connection error for page {page_num + 1} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
            return None
        except asyncio.TimeoutError:
            logging.warning(f"Timeout for page {page_num + 1} (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return None
        except aiohttp.ClientError as e:
            logging.warning(f"Client error for page {page_num + 1} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return None
        except Exception as e:
            logging.error(f"Error processing page {page_num + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return None
    
    return None


class PageTask:
    """Represents a page processing task."""
    def __init__(self, pdf_path: Path, pdf_name: str, page_num: int):
        self.pdf_path = pdf_path
        self.pdf_name = pdf_name
        self.page_num = page_num


def download_from_s3_for_processing(bucket_name: str, logger=logging) -> tuple:
    """
    Download ONLY NEW PDFs, metadata, and logs from S3 for processing.
    Checks processed_pdf_to_text_files.jsonl first to avoid re-downloading already processed PDFs.
    
    Args:
        bucket_name: S3 bucket name
        logger: Logger instance
        
    Returns:
        tuple: (temp_pdf_dir, pdf_count, metadata_downloaded, log_downloaded)
    """
    import tempfile
    import shutil
    import json
    from web_scrap.s3_helper import (
        get_s3_client, 
        get_latest_pdf_folder_from_s3,
        download_file_from_s3
    )
    
    try:
        # Get S3 client
        s3_client = get_s3_client()
        
        # Download existing processed_pdf_to_text_files.jsonl FIRST (to filter PDFs)
        processed_pdfs = set()
        log_downloaded = False
        try:
            logger.info("Downloading processed_pdf_to_text_files.jsonl to check which PDFs are already processed...")
            log_downloaded = download_file_from_s3(
                s3_client,
                bucket_name,
                "data/mospi_web/processed_pdf_to_text_files.jsonl",
                "processed_pdf_to_text_files.jsonl"
            )
            if log_downloaded:
                logger.info("✓ Downloaded processed_pdf_to_text_files.jsonl")
                # Read processed PDF names
                with open("processed_pdf_to_text_files.jsonl", 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                entry = json.loads(line)
                                processed_pdfs.add(entry['pdf_name'])
                            except:
                                pass
                logger.info(f"Found {len(processed_pdfs)} already processed PDFs")
        except Exception as e:
            logger.warning(f"No existing log file in S3 (will process all PDFs): {e}")
        
        # Find latest PDF folder
        logger.info("Finding latest PDF folder in S3...")
        latest_pdf_folder = get_latest_pdf_folder_from_s3(s3_client, bucket_name, "data/mospi_web")
        
        if not latest_pdf_folder:
            logger.error("No PDF folder found in S3")
            return None, 0, False, False
        
        logger.info(f"Latest PDF folder: s3://{bucket_name}/{latest_pdf_folder}")
        
        # List all PDFs in S3 folder
        logger.info("Listing PDFs in S3 folder...")
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=latest_pdf_folder)
        
        all_pdfs = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.pdf'):
                        pdf_name = key.split('/')[-1]
                        all_pdfs.append((key, pdf_name))
        
        logger.info(f"Found {len(all_pdfs)} total PDFs in S3")
        
        # Filter out already processed PDFs
        new_pdfs = [(key, name) for key, name in all_pdfs if name not in processed_pdfs]
        logger.info(f"Filtered to {len(new_pdfs)} NEW PDFs from latest folder (skipping {len(all_pdfs) - len(new_pdfs)} already processed)")
        
        # METADATA-BASED PROCESSING: Check web_file_metadata.json for any unprocessed PDFs
        # This catches:
        # 1. Manually uploaded files with old publish dates (uploaded to previous months)
        # 2. Web scraped files on month boundaries (e.g., scraped on 7/1 but saved to 6/30 folder)
        logger.info("Checking web_file_metadata.json for any unprocessed PDFs from other months...")
        metadata_key = "data/mospi_web/web_file_metadata.json"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_metadata.json', delete=False) as temp_metadata_file:
            temp_metadata_path = temp_metadata_file.name
        
        try:
            if download_file_from_s3(s3_client, bucket_name, metadata_key, temp_metadata_path):
                with open(temp_metadata_path, 'r', encoding='utf-8') as f:
                    metadata_list = json.load(f)
                
                logger.info(f"Found {len(metadata_list)} total entries in metadata")
                
                # Track PDFs we've already added from latest folder
                already_added = {name for _, name in new_pdfs}
                metadata_pdfs_added = 0
                
                # Find PDFs in metadata that haven't been processed
                for entry in metadata_list:
                    pdf_name = entry.get('file_name')
                    s3_path = entry.get('text_uri')  # This contains the full s3:// URI
                    
                    # Skip if already processed or already in our list
                    if not pdf_name:
                        continue
                    # Only PDFs go through the vision converter. Excel (and other
                    # non-PDF) entries are ingested by their own pipeline
                    # (excel_ingestion) and must NOT be fed to this converter.
                    if not pdf_name.lower().endswith('.pdf'):
                        logger.debug(f"[METADATA] Skipping non-PDF entry: {pdf_name}")
                        continue
                    if not s3_path:
                        logger.debug(f"[METADATA] Skipping {pdf_name}: no text_uri")
                        continue
                    if pdf_name in processed_pdfs:
                        logger.debug(f"[METADATA] Skipping {pdf_name}: already processed")
                        continue
                    if pdf_name in already_added:
                        logger.debug(f"[METADATA] Skipping {pdf_name}: already in list")
                        continue
                    
                    # Extract S3 key from s3:// URI (e.g., s3://bucket/key -> key)
                    if s3_path.startswith('s3://'):
                        # Remove s3://bucket_name/ prefix to get just the key
                        s3_key = s3_path.replace(f's3://{bucket_name}/', '')
                    else:
                        # If it's already just a key, use it as-is
                        s3_key = s3_path
                    
                    # Verify this PDF actually exists in S3
                    try:
                        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                        new_pdfs.append((s3_key, pdf_name))
                        already_added.add(pdf_name)
                        metadata_pdfs_added += 1
                        logger.info(f"[METADATA] Found unprocessed PDF: {pdf_name} (from {s3_key})")
                    except Exception as e:
                        logger.warning(f"[METADATA] PDF {pdf_name} not found in S3 at s3://{bucket_name}/{s3_key}: {e}")
                        continue
                
                logger.info(f"Added {metadata_pdfs_added} unprocessed PDFs from metadata")
                
                # Cleanup temp metadata file
                if os.path.exists(temp_metadata_path):
                    os.remove(temp_metadata_path)
            else:
                logger.warning("Could not download web_file_metadata.json, skipping metadata-based processing")
        except Exception as e:
            logger.error(f"Error processing metadata for unprocessed PDFs: {e}")
            # Cleanup temp metadata file on error
            if os.path.exists(temp_metadata_path):
                os.remove(temp_metadata_path)
        
        logger.info(f"Total PDFs to process: {len(new_pdfs)} (latest folder + metadata)")
        
        if len(new_pdfs) == 0:
            logger.info("No new PDFs to download")
            return None, 0, False, log_downloaded
        
        # Create temp directory for PDFs
        temp_pdf_dir = Path(tempfile.mkdtemp(prefix="mospi_pdfs_"))
        logger.info(f"Created temp directory: {temp_pdf_dir}")
        
        # Download ONLY new PDFs
        logger.info(f"Downloading {len(new_pdfs)} new PDFs from S3...")
        success_count = 0
        fail_count = 0
        
        for s3_key, pdf_name in new_pdfs:
            try:
                local_path = temp_pdf_dir / pdf_name
                logger.info(f"[S3] Downloading {pdf_name}...")
                s3_client.download_file(bucket_name, s3_key, str(local_path))
                success_count += 1
                logger.info(f"[S3] ✓ Downloaded {pdf_name}")
            except Exception as e:
                fail_count += 1
                logger.error(f"[S3] ✗ Failed to download {pdf_name}: {e}")
        
        logger.info(f"Downloaded {success_count} new PDFs, {fail_count} failed")
        
        # Download web_file_metadata.json
        metadata_downloaded = False
        try:
            logger.info("Downloading web_file_metadata.json...")
            metadata_downloaded = download_file_from_s3(
                s3_client,
                bucket_name,
                "data/mospi_web/web_file_metadata.json",
                "web_file_metadata.json"
            )
            if metadata_downloaded:
                logger.info("✓ Downloaded web_file_metadata.json")
        except Exception as e:
            logger.warning(f"Failed to download metadata: {e}")
        
        return temp_pdf_dir, success_count, metadata_downloaded, log_downloaded
        
    except Exception as e:
        logger.error(f"Error downloading from S3: {e}")
        return None, 0, False, False


def upload_to_s3_after_processing(bucket_name: str, output_dir: Path, logger=logging) -> bool:
    """
    Upload markdown files and updated log to S3 after processing.
    
    Args:
        bucket_name: S3 bucket name
        output_dir: Local directory containing markdown files
        logger: Logger instance
        
    Returns:
        bool: True if successful
    """
    from web_scrap.s3_helper import get_s3_client, upload_file_to_s3
    
    try:
        s3_client = get_s3_client()
        
        # Upload markdown files
        md_files = list(output_dir.glob("*.md"))
        logger.info(f"Uploading {len(md_files)} markdown files to S3...")
        
        success_count = 0
        fail_count = 0
        
        for md_file in md_files:
            s3_key = f"text_data_v1/{md_file.name}"
            result = upload_file_to_s3(
                s3_client,
                str(md_file),
                bucket_name,
                s3_key
            )
            
            if result["success"]:
                success_count += 1
                logger.info(f"✓ Uploaded {md_file.name}")
            else:
                fail_count += 1
                logger.error(f"✗ Failed to upload {md_file.name}: {result['error']}")
        
        logger.info(f"Uploaded {success_count} markdown files, {fail_count} failed")
        
        # Upload updated processed_pdf_to_text_files.jsonl
        log_file = Path("processed_pdf_to_text_files.jsonl")
        if log_file.exists():
            logger.info("Uploading processed_pdf_to_text_files.jsonl...")
            result = upload_file_to_s3(
                s3_client,
                str(log_file),
                bucket_name,
                "data/mospi_web/processed_pdf_to_text_files.jsonl"
            )
            
            if result["success"]:
                logger.info("✓ Uploaded processed_pdf_to_text_files.jsonl")
            else:
                logger.error(f"✗ Failed to upload log: {result['error']}")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        return False


class PDFProgress:
    """Tracks progress for a single PDF."""
    def __init__(self, pdf_path: Path, pdf_name: str, total_pages: int):
        self.pdf_path = pdf_path
        self.pdf_name = pdf_name
        self.total_pages = total_pages
        self.completed_pages: Dict[int, str] = {}  # page_num -> markdown
        self.failed_pages: Set[int] = set()
        self.start_time = time.time()
    
    def is_complete(self) -> bool:
        """Check if all pages are processed."""
        return len(self.completed_pages) + len(self.failed_pages) >= self.total_pages
    
    def get_markdown(self) -> str:
        """Get combined markdown with pages in order."""
        pages = []
        for page_num in sorted(self.completed_pages.keys()):
            content = self.completed_pages[page_num]
            pages.append(f"\n\n---\n==Page {page_num + 1}==\n---\n\n{content}")
        return "\n".join(pages)
    
    def get_duration(self) -> float:
        """Get processing duration in seconds."""
        return time.time() - self.start_time


class NuMarkdownBatchProcessorV3:
    """Streaming batch processor optimized for H200."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        api_url: str = "http://localhost:8002/v1",
        temperature: float = 0.3,
        max_concurrent: int = 5,
        dpi: int = 300,
        log_file: str = "processed_pdf_to_text_files.jsonl",
        dry_run: bool = False
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_url = api_url
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.dpi = dpi
        self.log_file = Path(log_file)
        self.dry_run = dry_run
        
        # Load processing log
        self.processed_files = self._load_processing_log()
        
        # Load web file metadata for enrichment
        self.web_metadata = self._load_web_file_metadata()
        
        # Stats
        self.stats = {
            'total_pdfs': 0,
            'completed_pdfs': 0,
            'failed_pdfs': 0,
            'skipped_pdfs': 0,
            'total_pages_processed': 0,
            'start_time': None
        }
        
        # Concurrency metrics
        self.concurrency_metrics = {
            'current_active': 0,
            'peak_concurrent': 0,
            'total_tasks_started': 0,
            'total_tasks_completed': 0,
            'image_encoding_time': 0.0,
            'api_call_time': 0.0,
            'last_update': time.time(),
            'peak_logged': False  # Track if we've logged peak
        }
        self.metrics_lock = asyncio.Lock()
    
    def _load_processing_log(self) -> Dict[str, Dict]:
        """Load processing log from JSONL file."""
        processed = {}
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            processed[entry['pdf_name']] = entry
                        except json.JSONDecodeError as e:
                            logging.warning(f"Skipping invalid JSON line in log: {e}")
            logging.info(f"Loaded {len(processed)} processed files from log")
        return processed
    
    def _load_web_file_metadata(self) -> Dict[str, Dict]:
        """Load web_file_metadata.json and index by filename."""
        metadata_path = Path("web_file_metadata.json")
        metadata_by_filename = {}
        
        if not metadata_path.exists():
            logging.warning(f"web_file_metadata.json not found at {metadata_path}")
            return {}
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            
            for entry in entries:
                filename = entry.get("file_name")
                if filename:
                    metadata_by_filename[filename] = entry
            
            logging.info(f"Loaded {len(metadata_by_filename)} metadata entries from web_file_metadata.json")
        except Exception as e:
            logging.error(f"Failed to load web_file_metadata.json: {e}")
        
        return metadata_by_filename
    
    def _append_to_log(self, entry: Dict):
        """Append entry to processing log."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def list_pdfs(self) -> List[Path]:
        """List all PDF files in input directory (recursively)."""
        # Debug: Log current working directory and input_dir details
        import os
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Input directory (raw): {self.input_dir}")
        logging.info(f"Input directory (absolute): {self.input_dir.absolute()}")
        logging.info(f"Input directory exists: {self.input_dir.exists()}")
        logging.info(f"Input directory is_dir: {self.input_dir.is_dir()}")
        
        if self.input_dir.exists() and self.input_dir.is_dir():
            # Try to list contents
            try:
                contents = list(self.input_dir.iterdir())
                logging.info(f"Directory contents count: {len(contents)}")
                logging.info(f"First 5 items: {[str(p) for p in contents[:5]]}")
            except Exception as e:
                logging.error(f"Error listing directory contents: {e}")
        
        pdfs = sorted(self.input_dir.rglob("*.pdf"))  # rglob for recursive search
        logging.info(f"Found {len(pdfs)} PDF files in {self.input_dir} (including subdirectories)")
        if pdfs:
            logging.info(f"First PDF found: {pdfs[0]}")
        return pdfs
    
    def get_pdf_page_count(self, pdf_path: Path) -> int:
        """Get number of pages in PDF."""
        try:
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            logging.error(f"Error getting page count for {pdf_path}: {e}")
            return 0
    
    async def process_pdf_streaming(
        self,
        pdf_path: Path,
        session: aiohttp.ClientSession
    ) -> Dict:
        """
        Process PDF with streaming pipeline.
        Maintains pool of concurrent requests, sends new page as soon as one completes.
        """
        pdf_name = pdf_path.name
        start_time = time.time()
        
        # Get page count
        total_pages = self.get_pdf_page_count(pdf_path)
        if total_pages == 0:
            return {
                'success': False,
                'error': 'Could not read PDF',
                'total_pages': 0,
                'successful_pages': 0,
                'failed_pages': [],
                'duration': time.time() - start_time
            }
        
        logging.info(f"Processing {pdf_name} ({total_pages} pages) with max {self.max_concurrent} concurrent requests")
        
        # Results storage
        completed_pages = {}
        failed_pages = set()
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(page_num: int):
            """Process single page with semaphore."""
            async with semaphore:
                result = await process_page_streaming(
                    session,
                    pdf_path,
                    page_num,
                    self.temperature,
                    self.api_url
                )
                return page_num, result
        
        # Create all tasks (they'll be limited by semaphore)
        tasks = [process_with_semaphore(page_num) for page_num in range(total_pages)]
        
        # Process all pages with streaming
        results = await asyncio.gather(*tasks)
        
        # Collect results
        for page_num, result in results:
            if result:
                completed_pages[page_num] = result
            else:
                failed_pages.add(page_num)
                logging.error(f"  Failed page {page_num + 1}")
        
        # Combine results in page order
        markdown_pages = []
        for page_num in sorted(completed_pages.keys()):
            content = completed_pages[page_num]
            markdown_pages.append(f"\n\n---\n==Page {page_num + 1}==\n---\n\n{content}")
        
        markdown_content = "\n".join(markdown_pages)
        duration = time.time() - start_time
        
        success = len(failed_pages) == 0
        
        return {
            'success': success,
            'total_pages': total_pages,
            'successful_pages': len(completed_pages),
            'failed_pages': list(failed_pages),
            'markdown_content': markdown_content,
            'duration': duration
        }
    
    async def process_batch(self):
        """Main processing loop with cross-PDF streaming."""
        # List PDFs to process
        all_pdfs = self.list_pdfs()
        
        # Filter out completed PDFs
        pdfs_to_process = []
        for pdf_path in all_pdfs:
            pdf_name = pdf_path.name
            if pdf_name in self.processed_files:
                entry = self.processed_files[pdf_name]
                if entry.get('status') == 'success':
                    self.stats['skipped_pdfs'] += 1
                    continue
            pdfs_to_process.append(pdf_path)
        
        self.stats['total_pdfs'] = len(pdfs_to_process)
        self.stats['start_time'] = time.time()
        
        logging.info(f"Processing {len(pdfs_to_process)} PDFs (skipped {self.stats['skipped_pdfs']} already processed)")
        
        if not pdfs_to_process:
            logging.info("No files to process. Exiting.")
            return
        
        # DRY RUN MODE
        if self.dry_run:
            self._dry_run_report(pdfs_to_process)
            return
        
        # Process with streaming pipeline
        await self._process_streaming_pipeline(pdfs_to_process)
        
        self._print_summary()
    
    async def _process_streaming_pipeline(self, pdfs_to_process: List[Path]):
        """
        Streaming pipeline that maintains max_concurrent requests across multiple PDFs.
        Uses dedicated thread pool for image encoding to avoid blocking.
        """
        # Global page queue across all PDFs
        page_queue = []
        
        # Active PDFs being processed
        active_pdfs: Dict[str, PDFProgress] = {}
        
        # Cache of open PDF documents (keep them open for faster access)
        pdf_docs: Dict[str, fitz.Document] = {}
        
        # PDF index
        pdf_index = 0
        
        # Load initial PDFs and populate queue
        max_active_pdfs = 1  # Process one PDF at a time (like V2) to avoid overwhelming vLLM
        
        def load_next_pdf():
            """Load next PDF and add its pages to queue."""
            nonlocal pdf_index
            if pdf_index >= len(pdfs_to_process):
                return False
            
            pdf_path = pdfs_to_process[pdf_index]
            pdf_name = pdf_path.name
            pdf_index += 1
            
            # Get page count
            total_pages = self.get_pdf_page_count(pdf_path)
            if total_pages == 0:
                logging.error(f"Could not read {pdf_name}, skipping")
                self.stats['failed_pdfs'] += 1
                return True
            
            # Create progress tracker
            progress = PDFProgress(pdf_path, pdf_name, total_pages)
            active_pdfs[pdf_name] = progress
            
            # Open PDF document and cache it
            try:
                pdf_docs[pdf_name] = fitz.open(pdf_path)
            except Exception as e:
                logging.error(f"Could not open {pdf_name}: {e}")
                self.stats['failed_pdfs'] += 1
                return True
            
            # Add pages to queue
            for page_num in range(total_pages):
                page_queue.append(PageTask(pdf_path, pdf_name, page_num))
            
            logging.info(f"Loaded {pdf_name} ({total_pages} pages) - Queue size: {len(page_queue)}")
            return True
        
        # Load initial batch of PDFs
        for _ in range(max_active_pdfs):
            if not load_next_pdf():
                break
        
        # Create session with connection pooling limits
        connector = aiohttp.TCPConnector(
            limit=CONNECTION_POOL_LIMIT,
            limit_per_host=CONNECTION_POOL_LIMIT,
            ttl_dns_cache=300,
            force_close=True,  # Don't reuse connections (vLLM may not handle reuse well)
            enable_cleanup_closed=True
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # Semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            # Active tasks
            active_tasks = set()
            
            async def process_page_task(task: PageTask):
                """Process a single page task with metrics."""
                # Track task start
                async with self.metrics_lock:
                    self.concurrency_metrics['current_active'] += 1
                    self.concurrency_metrics['total_tasks_started'] += 1
                    if self.concurrency_metrics['current_active'] > self.concurrency_metrics['peak_concurrent']:
                        self.concurrency_metrics['peak_concurrent'] = self.concurrency_metrics['current_active']
                        # Only log when we reach max_concurrent for the FIRST time
                        if (self.concurrency_metrics['peak_concurrent'] == self.max_concurrent and 
                            not self.concurrency_metrics['peak_logged']):
                            logging.info(f"Reached peak concurrency: {self.concurrency_metrics['peak_concurrent']}/{self.max_concurrent}")
                            self.concurrency_metrics['peak_logged'] = True
                
                try:
                    async with semaphore:
                        # Add small delay to pace requests (prevents overwhelming server)
                        await asyncio.sleep(REQUEST_PACING_DELAY)
                        
                        # Time image encoding (runs synchronously in main thread)
                        # PyMuPDF does NOT support threading, so we run it in main event loop
                        encode_start = time.time()
                        
                        # Get cached PDF document
                        pdf_doc = pdf_docs.get(task.pdf_name)
                        if not pdf_doc:
                            logging.error(f"PDF document not found in cache: {task.pdf_name}")
                            progress = active_pdfs[task.pdf_name]
                            progress.failed_pages.add(task.page_num)
                            return task
                        
                        # Run encoding synchronously in main thread (no threading with PyMuPDF)
                        result = pdf_page_to_base64_fast(
                            pdf_doc,
                            task.page_num,
                            self.dpi
                        )
                        
                        if result is None or result[0] is None:
                            progress = active_pdfs[task.pdf_name]
                            progress.failed_pages.add(task.page_num)
                            return task
                        
                        base64_image, img_format = result
                        encode_time = time.time() - encode_start
                        
                        async with self.metrics_lock:
                            self.concurrency_metrics['image_encoding_time'] += encode_time
                        
                        if not base64_image:
                            progress = active_pdfs[task.pdf_name]
                            progress.failed_pages.add(task.page_num)
                            return task
                        
                        # Time API call
                        api_start = time.time()
                        result = await self._call_numarkdown_api(
                            session,
                            base64_image,
                            img_format,
                            task.page_num
                        )
                        api_time = time.time() - api_start
                        
                        async with self.metrics_lock:
                            self.concurrency_metrics['api_call_time'] += api_time
                        
                        # Store result
                        progress = active_pdfs[task.pdf_name]
                        if result is not None:  # Accept empty string as valid
                            progress.completed_pages[task.page_num] = result
                            self.stats['total_pages_processed'] += 1
                            
                            # Only log first successful page
                            if self.stats['total_pages_processed'] == 1:
                                logging.info(f"First page completed: {task.pdf_name} page {task.page_num + 1} "
                                           f"(encode: {encode_time:.2f}s, api: {api_time:.2f}s)")
                        else:
                            progress.failed_pages.add(task.page_num)
                            logging.error(f"Failed: {task.pdf_name} page {task.page_num + 1} after {max_retries} attempts")
                        
                        # Check if PDF is complete
                        if progress.is_complete():
                            await self._save_pdf(progress)
                            
                            # Close and remove PDF document from cache
                            if task.pdf_name in pdf_docs:
                                pdf_docs[task.pdf_name].close()
                                del pdf_docs[task.pdf_name]
                            
                            del active_pdfs[task.pdf_name]
                            
                            # Load next PDF if queue is getting low
                            if len(page_queue) < self.max_concurrent * 2:
                                load_next_pdf()
                        
                        return task
                finally:
                    # Track task completion
                    async with self.metrics_lock:
                        self.concurrency_metrics['current_active'] -= 1
                        self.concurrency_metrics['total_tasks_completed'] += 1
            
            # Main processing loop
            last_metrics_print = time.time()
            first_progress_printed = False
            
            logging.info(f"Starting processing with max_concurrent={self.max_concurrent}, initial queue size={len(page_queue)}")
            
            while page_queue or active_tasks or pdf_index < len(pdfs_to_process):
                # Fill up to max_concurrent tasks
                while len(active_tasks) < self.max_concurrent and page_queue:
                    task = page_queue.pop(0)
                    active_task = asyncio.create_task(process_page_task(task))
                    active_tasks.add(active_task)
                    
                    # Log only the first task
                    if len(active_tasks) == 1:
                        logging.info(f"Started first task: {task.pdf_name} page {task.page_num + 1}")
                
                # Wait for at least one task to complete
                if active_tasks:
                    done, active_tasks = await asyncio.wait(
                        active_tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Print progress immediately after first completion, then every 5 seconds
                    now = time.time()
                    if not first_progress_printed or (now - last_metrics_print >= 5):
                        await self._print_progress_with_metrics()
                        last_metrics_print = now
                        first_progress_printed = True
                else:
                    # No active tasks and no queue, load more PDFs
                    if pdf_index < len(pdfs_to_process):
                        load_next_pdf()
                    else:
                        break
            
            # Final metrics print
            print()  # New line after progress line
            await self._print_progress_with_metrics()
            
            # Cleanup: close any remaining open PDFs
            for pdf_name, pdf_doc in pdf_docs.items():
                pdf_doc.close()
            pdf_docs.clear()
    
    async def _call_numarkdown_api(
        self,
        session: aiohttp.ClientSession,
        base64_image: str,
        img_format: str,
        page_num: int,
        max_retries: int = 3
    ) -> Optional[str]:
        """Call NuMarkdown API with retries and detailed timing."""
        for attempt in range(max_retries):
            try:
                data_url = f"data:image/{img_format};base64,{base64_image}"
                
                payload = {
                    "model": "numind/NuMarkdown-8B-Thinking",
                    "temperature": self.temperature,
                    "max_tokens": 8192,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_url},
                                    "min_pixels": 100 * 28 * 28,
                                    "max_pixels": 5000 * 28 * 28,
                                },
                            ],
                        },
                    ]
                }
                
                # Configure timeout
                timeout = aiohttp.ClientTimeout(
                    total=TOTAL_TIMEOUT,
                    connect=CONNECT_TIMEOUT,
                    sock_read=READ_TIMEOUT
                )
                
                # Time the actual HTTP request
                request_start = time.time()
                async with session.post(
                    f"{self.api_url}/chat/completions",
                    json=payload,
                    timeout=timeout
                ) as response:
                    request_time = time.time() - request_start
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Check if response has expected structure
                        if 'choices' not in result or not result['choices']:
                            logging.warning(f"Empty choices in response for page {page_num + 1}")
                            return ""  # Return empty string instead of None
                        
                        content = result['choices'][0]['message']['content']
                        
                        # Handle empty or null content gracefully
                        if content is None:
                            logging.info(f"Null content for page {page_num + 1} (likely empty page), using empty string")
                            return ""
                        
                        if not content or content.strip() == "":
                            logging.info(f"Empty content for page {page_num + 1} (likely empty page), using empty string")
                            return ""
                        
                        # Extract answer from tags
                        if "<answer>" in content and "</answer>" in content:
                            answer = content.split("<answer>")[1].split("</answer>")[0].strip()
                        else:
                            answer = content
                        
                        # Handle case where answer extraction results in empty string
                        if not answer or answer.strip() == "":
                            logging.info(f"Empty answer after extraction for page {page_num + 1}, using empty string")
                            return ""
                        
                        # Log slow API calls
                        if request_time > 20:
                            logging.warning(f"Slow API call: {request_time:.1f}s for page {page_num + 1}")
                        
                        return answer
                    else:
                        # Detailed error logging for non-200 responses
                        error_body = await response.text()
                        logging.error(f"API error for page {page_num + 1} (attempt {attempt + 1}): "
                                    f"HTTP {response.status}")
                        logging.error(f"  Response body: {error_body[:500]}")  # First 500 chars
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return None
            
            except aiohttp.ClientOSError as e:
                # Handle specific connection errors with detailed logging
                error_details = f"{type(e).__name__}: {e}"
                if hasattr(e, 'errno'):
                    error_details += f" (errno: {e.errno})"
                
                if hasattr(e, 'errno') and e.errno == 32:  # Broken pipe
                    logging.warning(f"Broken pipe for page {page_num + 1} (attempt {attempt + 1})")
                    logging.error(f"  Details: {error_details}")
                elif "Server disconnected" in str(e):
                    logging.warning(f"Server disconnected for page {page_num + 1} (attempt {attempt + 1})")
                    logging.error(f"  Details: {error_details}")
                else:
                    logging.warning(f"Connection error for page {page_num + 1} (attempt {attempt + 1})")
                    logging.error(f"  Details: {error_details}")
                
                if attempt < max_retries - 1:
                    retry_delay = 5 if hasattr(e, 'errno') and e.errno == 32 else 3
                    await asyncio.sleep(retry_delay)
                    continue
                return None
            except asyncio.TimeoutError:
                logging.warning(f"Timeout for page {page_num + 1} (attempt {attempt + 1})")
                logging.error(f"  Timeout after {TOTAL_TIMEOUT}s")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
            except aiohttp.ClientError as e:
                logging.warning(f"Client error for page {page_num + 1} (attempt {attempt + 1})")
                logging.error(f"  Details: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error for page {page_num + 1} (attempt {attempt + 1})")
                logging.error(f"  Details: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
            except Exception as e:
                logging.error(f"Unexpected error processing page {page_num + 1} (attempt {attempt + 1})")
                logging.error(f"  Type: {type(e).__name__}")
                logging.error(f"  Details: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
        
        return None
    
    async def _save_pdf(self, progress: PDFProgress):
        """Save completed PDF."""
        try:
            # Combine pages in order
            markdown_content = progress.get_markdown()
            
            # Save to file
            output_path = self.output_dir / progress.pdf_name.replace('.pdf', '.md')
            output_path.write_text(markdown_content, encoding='utf-8')
            
            # Calculate MD5
            md5_hash = hashlib.md5()
            with open(progress.pdf_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    md5_hash.update(chunk)
            md5 = md5_hash.hexdigest()
            
            # Get metadata from web_file_metadata.json
            metadata = self.web_metadata.get(progress.pdf_name, {})
            file_url = metadata.get("file_url")
            title = metadata.get("title")
            publish_date = metadata.get("publish_date")
            
            # Log success (print newline first to clear progress line)
            print()  # New line before log
            log_entry = {
                'pdf_name': progress.pdf_name,
                'output_path': str(output_path),
                'md5': md5,
                'file_url': file_url,  # NEW FIELD
                'title': title,  # NEW FIELD
                'publish_date': publish_date,  # NEW FIELD
                'total_pages': progress.total_pages,
                'successful_pages': len(progress.completed_pages),
                'failed_pages': list(progress.failed_pages),
                'processing_time_seconds': progress.get_duration(),
                'timestamp': datetime.now().isoformat(),
                'status': 'success' if len(progress.failed_pages) == 0 else 'partial'
            }
            self._append_to_log(log_entry)
            self.stats['completed_pdfs'] += 1
            
            logging.info(f"✓ Saved {progress.pdf_name} ({len(progress.completed_pages)}/{progress.total_pages} pages in {progress.get_duration():.1f}s)")
        
        except Exception as e:
            print()  # New line before error log
            logging.error(f"Error saving {progress.pdf_name}: {e}")
            self.stats['failed_pdfs'] += 1
    
    async def _print_progress_with_metrics(self):
        """Print progress update with concurrency metrics to CLI."""
        completed = self.stats['completed_pdfs'] + self.stats['failed_pdfs']
        remaining = self.stats['total_pdfs'] - completed
        
        async with self.metrics_lock:
            current_active = self.concurrency_metrics['current_active']
            peak_concurrent = self.concurrency_metrics['peak_concurrent']
            total_started = self.concurrency_metrics['total_tasks_started']
            total_completed = self.concurrency_metrics['total_tasks_completed']
            
            # Calculate average times
            avg_encode_time = 0.0
            avg_api_time = 0.0
            if total_completed > 0:
                avg_encode_time = self.concurrency_metrics['image_encoding_time'] / total_completed
                avg_api_time = self.concurrency_metrics['api_call_time'] / total_completed
        
        # Always print progress, even if no PDFs completed yet
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            
            # Calculate ETA
            eta_str = "calculating..."
            if completed > 0:
                avg_time = elapsed / completed
                eta_seconds = avg_time * remaining
                eta_minutes = eta_seconds / 60
                eta_str = f"{eta_minutes:.1f}m"
            
            print(f"\rPDFs: {completed}/{self.stats['total_pdfs']} | "
                  f"Pages: {self.stats['total_pages_processed']} | "
                  f"✓ {self.stats['completed_pdfs']} | "
                  f"✗ {self.stats['failed_pdfs']} | "
                  f"Active: {current_active}/{self.max_concurrent} | "
                  f"Peak: {peak_concurrent} | "
                  f"Encode: {avg_encode_time:.2f}s | "
                  f"API: {avg_api_time:.2f}s | "
                  f"ETA: {eta_str}", 
                  end='', flush=True)
    
    def _print_progress(self):
        """Print progress update to CLI (legacy method)."""
        completed = self.stats['completed_pdfs'] + self.stats['failed_pdfs']
        remaining = self.stats['total_pdfs'] - completed
        
        if completed > 0 and self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            avg_time = elapsed / completed
            eta_seconds = avg_time * remaining
            eta_minutes = eta_seconds / 60
            
            print(f"\rPDFs: {completed}/{self.stats['total_pdfs']} | "
                  f"Pages: {self.stats['total_pages_processed']} | "
                  f"✓ {self.stats['completed_pdfs']} | "
                  f"✗ {self.stats['failed_pdfs']} | "
                  f"ETA: {eta_minutes:.1f} min", 
                  end='', flush=True)
    
    def _print_summary(self):
        """Print processing summary with concurrency metrics."""
        elapsed = time.time() - self.stats['start_time']
        
        print("\n" + "=" * 80)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Total PDFs: {self.stats['total_pdfs']}")
        print(f"Completed: {self.stats['completed_pdfs']}")
        print(f"Failed: {self.stats['failed_pdfs']}")
        print(f"Skipped: {self.stats['skipped_pdfs']}")
        print(f"Total pages processed: {self.stats['total_pages_processed']}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        if self.stats['completed_pdfs'] > 0:
            avg_time = elapsed / self.stats['completed_pdfs']
            print(f"Average time per PDF: {avg_time:.1f} seconds")
        if self.stats['total_pages_processed'] > 0:
            avg_page_time = elapsed / self.stats['total_pages_processed']
            print(f"Average time per page: {avg_page_time:.1f} seconds")
            pages_per_hour = 3600 / avg_page_time
            print(f"Throughput: {pages_per_hour:.0f} pages/hour")
        
        print("\n" + "-" * 80)
        print("CONCURRENCY METRICS")
        print("-" * 80)
        print(f"Max concurrent setting: {self.max_concurrent}")
        print(f"DPI setting: {self.dpi}")
        print(f"Peak concurrent achieved: {self.concurrency_metrics['peak_concurrent']}")
        print(f"Total tasks started: {self.concurrency_metrics['total_tasks_started']}")
        print(f"Total tasks completed: {self.concurrency_metrics['total_tasks_completed']}")
        
        if self.concurrency_metrics['total_tasks_completed'] > 0:
            avg_encode = self.concurrency_metrics['image_encoding_time'] / self.concurrency_metrics['total_tasks_completed']
            avg_api = self.concurrency_metrics['api_call_time'] / self.concurrency_metrics['total_tasks_completed']
            total_avg = avg_encode + avg_api
            
            print(f"\nAverage time per page:")
            print(f"  Image encoding: {avg_encode:.2f}s ({avg_encode/total_avg*100:.1f}%)")
            print(f"  API call: {avg_api:.2f}s ({avg_api/total_avg*100:.1f}%)")
            print(f"  Total: {total_avg:.2f}s")
            
            # Bottleneck analysis
            print(f"\nBottleneck analysis:")
            if avg_encode > avg_api * 1.5:
                print(f"  ⚠️  Image encoding is the bottleneck ({avg_encode:.2f}s vs {avg_api:.2f}s)")
                print(f"      Consider: reducing DPI (currently 300), or using faster image encoding")
            elif avg_api > avg_encode * 1.5:
                print(f"  ⚠️  API calls are the bottleneck ({avg_api:.2f}s vs {avg_encode:.2f}s)")
                print(f"      Consider: increasing vLLM batch size, or adding more GPUs")
            else:
                print(f"  ✓ Balanced workload (encode: {avg_encode:.2f}s, API: {avg_api:.2f}s)")
            
            # Concurrency utilization
            utilization = (self.concurrency_metrics['peak_concurrent'] / self.max_concurrent) * 100
            print(f"\nConcurrency utilization: {utilization:.1f}%")
            if utilization < 80:
                print(f"  ⚠️  Low utilization - only reached {self.concurrency_metrics['peak_concurrent']}/{self.max_concurrent} concurrent tasks")
                print(f"      This suggests a bottleneck preventing full parallelization")
            else:
                print(f"  ✓ Good utilization - reached {self.concurrency_metrics['peak_concurrent']}/{self.max_concurrent} concurrent tasks")
        
        print("=" * 80)
    
    def _dry_run_report(self, pdfs_to_process: List[Path]):
        """Generate dry run report showing what will be processed."""
        print("\n" + "=" * 80)
        print("DRY RUN MODE - No files will be processed")
        print("=" * 80)
        
        # Sample first 10 PDFs to get page counts
        print(f"\nTotal PDFs found: {len(self.list_pdfs())}")
        print(f"Already processed (will skip): {self.stats['skipped_pdfs']}")
        print(f"PDFs to process: {len(pdfs_to_process)}")
        
        print("\nSample of PDFs to process (first 10):")
        print("-" * 80)
        
        total_pages = 0
        sample_size = min(10, len(pdfs_to_process))
        
        for i, pdf_path in enumerate(pdfs_to_process[:sample_size]):
            page_count = self.get_pdf_page_count(pdf_path)
            total_pages += page_count
            
            output_path = self.output_dir / pdf_path.name.replace('.pdf', '.md')
            
            print(f"{i+1:2d}. {pdf_path.name}")
            print(f"    Pages: {page_count}")
            print(f"    Input:  {pdf_path}")
            print(f"    Output: {output_path}")
            
            # Estimate time (assuming 2 seconds per page with 20 concurrent)
            est_time = (page_count / self.max_concurrent) * 2
            print(f"    Est. time: {est_time/60:.1f} min")
            print()
        
        # Estimate total
        if sample_size > 0:
            avg_pages = total_pages / sample_size
            est_total_pages = int(avg_pages * len(pdfs_to_process))
            
            # Estimate time (2 seconds per page with max_concurrent parallelism)
            est_total_time = (est_total_pages / self.max_concurrent) * 2
            
            print("=" * 80)
            print("ESTIMATES (based on sample):")
            print("=" * 80)
            print(f"Average pages per PDF: {avg_pages:.0f}")
            print(f"Estimated total pages: {est_total_pages:,}")
            print(f"Estimated total time: {est_total_time/3600:.1f} hours ({est_total_time/60:.0f} minutes)")
            print(f"Estimated throughput: {est_total_pages / (est_total_time/3600):.0f} pages/hour")
        
        print("\n" + "=" * 80)
        print("CONFIGURATION:")
        print("=" * 80)
        print(f"Input directory:  {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"API URL: {self.api_url}")
        print(f"Temperature: {self.temperature}")
        print(f"Max concurrent: {self.max_concurrent}")
        print(f"Log file: {self.log_file}")
        print("=" * 80)
        
        print("\nTo start actual processing, run without --dry-run flag")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process PDFs with NuMarkdown V3 (streaming pipeline for H200)"
    )
    parser.add_argument("--input-dir", default=None, 
                        help="Input directory with PDFs (only for local mode, ignored in S3 mode)")
    parser.add_argument("--output-dir", default="./md_files", 
                        help="Output directory for markdown files (default: ./md_files)")
    parser.add_argument("--api-url", default="http://localhost:8002/v1", 
                        help="vLLM API URL")
    parser.add_argument("--temperature", type=float, default=0.3, 
                        help="Sampling temperature")
    parser.add_argument("--max-concurrent", type=int, default=20, 
                        help="Maximum concurrent requests (default: 20)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for PDF rendering (default: 300, higher=better quality but slower)")
    parser.add_argument("--log-file", default="processed_pdf_to_text_files.jsonl", 
                        help="Processing log file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what will be processed without actually processing")
    parser.add_argument("--local-mode", action="store_true",
                        help="Use local input-dir instead of S3 (for testing only)")
    parser.add_argument("--s3-bucket", default="acct1004215-mospi",
                        help="S3 bucket name (default: acct1004215-mospi)")
    
    args = parser.parse_args()
    
    # S3 mode is DEFAULT (unless --local-mode is specified)
    use_s3_mode = not args.local_mode
    temp_pdf_dir = None
    
    if use_s3_mode:
        logging.info("=" * 80)
        logging.info("S3 MODE: Downloading from S3 (default behavior)")
        logging.info("=" * 80)
        
        temp_pdf_dir, pdf_count, metadata_ok, log_ok = download_from_s3_for_processing(
            bucket_name=args.s3_bucket,
            logger=logging
        )
        
        if temp_pdf_dir is None or pdf_count == 0:
            logging.error("Failed to download PDFs from S3. Exiting.")
            return 1
        
        # Override input_dir to use temp directory
        args.input_dir = str(temp_pdf_dir)
        logging.info(f"Using temp directory: {args.input_dir}")
        logging.info("=" * 80)
    else:
        # Local mode - require input_dir to be specified
        if not args.input_dir:
            logging.error("LOCAL MODE: --input-dir must be specified when using --local-mode")
            return 1
        logging.info("=" * 80)
        logging.info(f"LOCAL MODE: Using input directory: {args.input_dir}")
        logging.info("=" * 80)
    
    # Initialize processor
    processor = NuMarkdownBatchProcessorV3(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        api_url=args.api_url,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        dpi=args.dpi,
        log_file=args.log_file,
        dry_run=args.dry_run
    )
    
    # Run batch processing
    asyncio.run(processor.process_batch())
    
    # S3 upload phase (only in S3 mode)
    if use_s3_mode and not args.dry_run:
        logging.info("=" * 80)
        logging.info("S3 MODE: Uploading results to S3")
        logging.info("=" * 80)
        
        upload_success = upload_to_s3_after_processing(
            bucket_name=args.s3_bucket,
            output_dir=Path(args.output_dir),
            logger=logging
        )
        
        if upload_success:
            logging.info("✓ Successfully uploaded results to S3")
        else:
            logging.warning("⚠ Some uploads to S3 failed")
        
        logging.info("=" * 80)
    
    # Cleanup temp directory
    if temp_pdf_dir and temp_pdf_dir.exists():
        import shutil
        logging.info(f"Cleaning up temp directory: {temp_pdf_dir}")
        shutil.rmtree(temp_pdf_dir, ignore_errors=True)
        logging.info("✓ Temp directory cleaned up")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
