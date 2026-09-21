#!/usr/bin/env python3
"""
API Ingestion Pipeline
Main orchestrator for fetching documents from MoSPI APIs and uploading to S3
"""

import os
import sys
import json
import logging
import tempfile
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Import S3 helper from web_scrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'web_scrap'))
from s3_helper import get_s3_client, upload_file_to_s3, download_file_from_s3, get_s3_config

# Local imports
from .mospi_api_client import create_client as create_api_client
from .document_models import APIDocument, FileMetadata
from .deduplicator import create_checker, DuplicationChecker


class APIIngestionPipeline:
    """
    Main pipeline for API-based document ingestion
    Handles: Fetch → Deduplicate → Download → Upload → Metadata
    """
    
    def __init__(
        self,
        max_results_per_api: int = 20,
        target_date: Optional[datetime] = None,
        date_range_days: Optional[int] = None,
        dry_run: bool = False,
        skip_duplicates: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize API ingestion pipeline with date filtering
        
        Args:
            max_results_per_api: Maximum documents to fetch per API
            target_date: Target date for filtering (default: today)
            date_range_days: Check documents from last N days
            dry_run: If True, don't actually upload to S3
            skip_duplicates: If True, skip duplicate checking
            logger: Logger instance
        """
        self.max_results = max_results_per_api
        self.target_date = target_date or datetime.now()
        self.date_range_days = date_range_days
        self.dry_run = dry_run
        self.skip_duplicates = skip_duplicates
        self.logger = logger or logging.getLogger(__name__)
        
        # Calculate date range
        if date_range_days:
            from datetime import timedelta
            self.start_date = self.target_date - timedelta(days=date_range_days)
            date_filter_str = f"{self.start_date.strftime('%Y-%m-%d')} to {self.target_date.strftime('%Y-%m-%d')}"
        else:
            self.start_date = self.target_date
            date_filter_str = self.target_date.strftime('%Y-%m-%d')
        
        # Initialize S3 client
        self.s3_config = get_s3_config()
        self.s3_client = get_s3_client()
        self.bucket_name = self.s3_config['bucket_name']
        self.s3_prefix = self.s3_config['s3_prefix']  # e.g., "data/mospi_web"
        
        self.logger.info(f"[PIPELINE] Initialized (max_results={max_results_per_api}, dry_run={dry_run})")
        self.logger.info(f"[PIPELINE] Date filter: {date_filter_str}")
        self.logger.info(f"[PIPELINE] Skip duplicates: {skip_duplicates}")
        self.logger.info(f"[PIPELINE] S3 bucket: {self.bucket_name}, prefix: {self.s3_prefix}")
        
        # Stats
        self.stats = {
            "fetched": 0,
            "duplicates": 0,
            "new_files": 0,
            "uploaded": 0,
            "errors": 0
        }
    
    def load_metadata_from_s3(self) -> List[Dict]:
        """
        Load web_file_metadata.json from S3
        
        Returns:
            List of metadata entries
        """
        metadata_s3_key = f"{self.s3_prefix}/web_file_metadata.json"
        
        self.logger.info(f"[PIPELINE] Loading metadata from S3: {metadata_s3_key}")
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_metadata.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            if download_file_from_s3(self.s3_client, self.bucket_name, metadata_s3_key, temp_path):
                with open(temp_path, 'r', encoding='utf-8') as f:
                    metadata_list = json.load(f)
                self.logger.info(f"[PIPELINE] Loaded {len(metadata_list)} existing metadata entries")
            else:
                self.logger.warning(f"[PIPELINE] No existing metadata found, starting fresh")
                metadata_list = []
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return metadata_list
            
        except Exception as e:
            self.logger.error(f"[PIPELINE] Error loading metadata: {e}")
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return []
    
    def update_metadata_in_s3(self, metadata_list: List[Dict]) -> bool:
        """
        Update web_file_metadata.json in S3
        
        Args:
            metadata_list: Updated list of metadata entries
        
        Returns:
            True if successful
        """
        if self.dry_run:
            self.logger.info(f"[PIPELINE] [DRY-RUN] Would update metadata with {len(metadata_list)} entries")
            return True
        
        metadata_s3_key = f"{self.s3_prefix}/web_file_metadata.json"
        
        self.logger.info(f"[PIPELINE] Updating metadata in S3: {metadata_s3_key}")
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_metadata.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Write updated metadata
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, indent=2, ensure_ascii=False)
            
            # Upload to S3
            result = upload_file_to_s3(self.s3_client, temp_path, self.bucket_name, metadata_s3_key)
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            if result["success"]:
                self.logger.info(f"[PIPELINE] Metadata updated successfully")
                return True
            else:
                self.logger.error(f"[PIPELINE] Failed to update metadata: {result['error']}")
                return False
                
        except Exception as e:
            self.logger.error(f"[PIPELINE] Error updating metadata: {e}")
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return False
    
    def upload_pdf_to_s3(
        self,
        temp_file_path: str,
        filename: str,
        publish_date: str,
        subfolder: str = "pdfs"
    ) -> Optional[str]:
        """
        Upload a document to S3 with proper path structure.

        Args:
            temp_file_path: Path to temp file (already downloaded)
            filename: Filename to use in S3
            publish_date: Publish date (YYYY-MM-DD)
            subfolder: S3 subfolder under {year}/{month}/ ("pdfs" for PDFs,
                       "excel" for Excel workbooks)

        Returns:
            S3 URI if successful, None otherwise
        """
        if self.dry_run:
            self.logger.info(f"[PIPELINE] [DRY-RUN] Would upload: {filename}")
            return f"s3://{self.bucket_name}/data/mospi_web/2026/july/{subfolder}/{filename}"
        
        try:
            # Extract year/month from publish_date
            date_obj = datetime.strptime(publish_date, "%Y-%m-%d")
            year = date_obj.year
            month = date_obj.strftime("%B").lower()
        except Exception as e:
            self.logger.warning(f"[PIPELINE] Failed to parse date '{publish_date}': {e}, using current date")
            now = datetime.now()
            year = now.year
            month = now.strftime("%B").lower()
        
        # Build S3 key
        s3_key = f"{self.s3_prefix}/{year}/{month}/{subfolder}/{filename}"
        
        self.logger.info(f"[PIPELINE] Uploading to S3: {s3_key}")
        
        try:
            result = upload_file_to_s3(
                self.s3_client,
                temp_file_path,
                self.bucket_name,
                s3_key
            )
            
            if result["success"]:
                s3_uri = f"s3://{self.bucket_name}/{s3_key}"
                self.logger.info(f"[PIPELINE] Upload successful: {s3_uri}")
                return s3_uri
            else:
                self.logger.error(f"[PIPELINE] Upload failed: {result['error']}")
                return None
                
        except Exception as e:
            self.logger.error(f"[PIPELINE] Upload error: {e}")
            return None
    
    def process_api_documents(
        self,
        api_name: str = "latest_releases"
    ) -> Dict:
        """
        Process documents from a single API
        
        Args:
            api_name: Name of API to process (from API registry)
        
        Returns:
            Dictionary with processing results and stats
        """
        self.logger.info(f"[PIPELINE] ========== Processing API: {api_name} ==========")
        
        # Step 1: Fetch documents from API
        api_client = create_api_client(logger=self.logger)
        
        try:
            raw_documents = api_client.fetch_all_from_endpoint(
                api_name=api_name,
                max_results=self.max_results
            )
            
            self.stats["fetched"] = len(raw_documents)
            self.logger.info(f"[PIPELINE] Fetched {len(raw_documents)} documents from API")
            
        except Exception as e:
            self.logger.error(f"[PIPELINE] Failed to fetch from API: {e}")
            return {"success": False, "error": str(e), "stats": self.stats}
        finally:
            api_client.close()
        
        # Step 2: Parse into APIDocument objects
        documents = []
        for raw_doc in raw_documents:
            try:
                doc = APIDocument.from_api_response(raw_doc, source_api=api_name)
                if doc.has_files():
                    documents.append(doc)
            except Exception as e:
                self.logger.warning(f"[PIPELINE] Failed to parse document {raw_doc.get('id')}: {e}")
        
        self.logger.info(f"[PIPELINE] Parsed {len(documents)} documents with files")
        
        # Step 3: Load metadata and initialize deduplication checker
        metadata_cache = self.load_metadata_from_s3()
        dedup_checker = create_checker(
            metadata_cache=metadata_cache,
            s3_client=self.s3_client,
            bucket_name=self.bucket_name,
            logger=self.logger
        )
        
        # Step 4: Process each document (handle multiple files per document)
        new_metadata_entries = []
        
        for doc in documents:
            pdf_files = doc.get_pdf_files()
            
            if not pdf_files:
                self.logger.debug(f"[PIPELINE] No PDF files in document: {doc.title}")
                continue
            
            self.logger.info(f"[PIPELINE] Processing document: {doc.title} ({len(pdf_files)} PDFs)")
            
            for file_meta in pdf_files:
                try:
                    file_url = file_meta.get_full_url()
                    filename = file_meta.extract_filename_from_path()
                    
                    # Check duplication
                    is_dup, reason, temp_file = dedup_checker.check_duplication(
                        file_url=file_url,
                        filename=filename,
                        publish_date=doc.publish_date
                    )
                    
                    if is_dup:
                        self.stats["duplicates"] += 1
                        self.logger.info(f"[PIPELINE] Skipping duplicate: {filename} ({reason})")
                        continue
                    
                    # NEW FILE: Upload to S3
                    self.stats["new_files"] += 1
                    
                    if not temp_file:
                        self.logger.error(f"[PIPELINE] No temp file for new document: {filename}")
                        self.stats["errors"] += 1
                        continue
                    
                    # Compute MD5 (already downloaded)
                    file_md5 = DuplicationChecker.compute_md5(temp_file)
                    
                    # Upload to S3
                    s3_uri = self.upload_pdf_to_s3(
                        temp_file_path=temp_file,
                        filename=filename,
                        publish_date=doc.publish_date
                    )
                    
                    # Cleanup temp file
                    os.unlink(temp_file)
                    
                    if not s3_uri:
                        self.logger.error(f"[PIPELINE] S3 upload failed: {filename}")
                        self.stats["errors"] += 1
                        continue
                    
                    self.stats["uploaded"] += 1
                    
                    # Create metadata entry
                    metadata_entry = doc.to_metadata_entry(
                        file_metadata=file_meta,
                        md5_hash=file_md5,
                        s3_uri=s3_uri
                    )
                    
                    new_metadata_entries.append(metadata_entry)
                    self.logger.info(f"[PIPELINE] ✓ Processed: {filename} (MD5: {file_md5[:8]}...)")
                    
                except Exception as e:
                    self.logger.error(f"[PIPELINE] Error processing file {file_meta.filename}: {e}")
                    self.stats["errors"] += 1
                    # Cleanup temp file if exists
                    if temp_file and os.path.exists(temp_file):
                        os.unlink(temp_file)

        # Process Excel files (uploaded to S3, then ingested directly
        # via excel_ingestion 
        for doc in documents:
            excel_files = doc.get_excel_files()
            if not excel_files:
                continue
            self.logger.info(f"[PIPELINE] Processing document: {doc.title} ({len(excel_files)} Excel file(s))")
            for file_meta in excel_files:
                temp_file = None
                try:
                    file_url = file_meta.get_full_url()
                    filename = file_meta.extract_filename_from_path()

                    is_dup, reason, temp_file = dedup_checker.check_duplication(
                        file_url=file_url,
                        filename=filename,
                        publish_date=doc.publish_date
                    )
                    if is_dup:
                        self.stats["duplicates"] += 1
                        self.logger.info(f"[PIPELINE] Skipping duplicate Excel: {filename} ({reason})")
                        continue

                    self.stats["new_files"] += 1
                    if not temp_file:
                        self.logger.error(f"[PIPELINE] No temp file for new Excel: {filename}")
                        self.stats["errors"] += 1
                        continue

                    file_md5 = DuplicationChecker.compute_md5(temp_file)

                    # Upload to S3 under the excel/ subfolder
                    s3_uri = self.upload_pdf_to_s3(
                        temp_file_path=temp_file,
                        filename=filename,
                        publish_date=doc.publish_date,
                        subfolder="excel",
                    )
                    if not s3_uri:
                        self.logger.error(f"[PIPELINE] S3 upload failed (Excel): {filename}")
                        self.stats["errors"] += 1
                        continue

                    self.stats["uploaded"] += 1

                    # Record metadata (for dedup/records). batch_process skips
                    # non-PDF entries, so this will not be mis-converted.
                    metadata_entry = doc.to_metadata_entry(
                        file_metadata=file_meta,
                        md5_hash=file_md5,
                        s3_uri=s3_uri,
                    )
                    new_metadata_entries.append(metadata_entry)

                    # Ingest the Excel directly (extract -> chunk -> embed -> upsert)
                    if not self.dry_run:
                        try:
                            from excel_ingestion.excel_processor import process_excel_file
                            meta = {
                                "file_name": filename,
                                "file_url": file_url,
                                "title": doc.title,
                                "publish_date": doc.publish_date,
                                "md5": file_md5,
                            }
                            res = process_excel_file(temp_file, meta)
                            self.logger.info(f"[PIPELINE] Excel ingested: {filename} -> {res}")
                        except Exception as e:
                            self.logger.error(f"[PIPELINE] Excel ingestion failed for {filename}: {e}")
                            self.stats["errors"] += 1

                    self.logger.info(f"[PIPELINE] ✓ Processed Excel: {filename} (MD5: {file_md5[:8]}...)")

                except Exception as e:
                    self.logger.error(f"[PIPELINE] Error processing Excel {getattr(file_meta, 'filename', '?')}: {e}")
                    self.stats["errors"] += 1
                finally:
                    if temp_file and os.path.exists(temp_file):
                        os.unlink(temp_file)
        
        # Step 5: Update metadata in S3
        if new_metadata_entries:
            self.logger.info(f"[PIPELINE] Adding {len(new_metadata_entries)} new entries to metadata")
            
            # Append new entries to existing metadata
            updated_metadata = metadata_cache + new_metadata_entries
            
            if self.update_metadata_in_s3(updated_metadata):
                self.logger.info(f"[PIPELINE] Metadata updated successfully")
            else:
                self.logger.error(f"[PIPELINE] Failed to update metadata")
        else:
            self.logger.info(f"[PIPELINE] No new files to add to metadata")
        
        # Summary
        self.logger.info(f"[PIPELINE] ========== Processing Complete ==========")
        self.logger.info(f"[PIPELINE] Fetched: {self.stats['fetched']}")
        self.logger.info(f"[PIPELINE] Duplicates: {self.stats['duplicates']}")
        self.logger.info(f"[PIPELINE] New files: {self.stats['new_files']}")
        self.logger.info(f"[PIPELINE] Uploaded: {self.stats['uploaded']}")
        self.logger.info(f"[PIPELINE] Errors: {self.stats['errors']}")
        
        return {
            "success": True,
            "stats": self.stats,
            "new_metadata_entries": new_metadata_entries
        }


def run_ingestion(
    api_name: str = "latest_releases",
    max_results: int = 20,
    target_date: Optional[datetime] = None,
    date_range_days: Optional[int] = None,
    dry_run: bool = False,
    skip_duplicates: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Main entry point for API ingestion with date filtering
    
    Args:
        api_name: API to process
        max_results: Maximum documents to fetch
        target_date: Target date for filtering (default: today)
        date_range_days: Check documents from last N days (default: None = single date)
        dry_run: If True, don't upload to S3
        skip_duplicates: If True, skip duplicate checking (default: True)
        logger: Logger instance
    
    Returns:
        Results dictionary
    """
    pipeline = APIIngestionPipeline(
        max_results_per_api=max_results,
        target_date=target_date,
        date_range_days=date_range_days,
        dry_run=dry_run,
        skip_duplicates=skip_duplicates,
        logger=logger
    )
    
    return pipeline.process_api_documents(api_name=api_name)
