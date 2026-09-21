#!/usr/bin/env python3
"""
MD5-based deduplication checker
3-tier strategy: URL check → MD5 check → S3 path check
"""

import hashlib
import os
import tempfile
import requests
import logging
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path


class DuplicationChecker:
    """
    Handles 3-tier deduplication strategy for API ingestion
    """
    
    def __init__(
        self,
        metadata_cache: List[Dict],
        s3_client,
        bucket_name: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize deduplication checker
        
        Args:
            metadata_cache: List of existing metadata entries from web_file_metadata.json
            s3_client: boto3 S3 client
            bucket_name: S3 bucket name
            logger: Logger instance
        """
        self.metadata_cache = metadata_cache
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.logger = logger or logging.getLogger(__name__)
        
        # Build lookup sets for fast checking
        self.existing_urls = {entry['file_url'] for entry in metadata_cache if 'file_url' in entry}
        self.existing_md5s = {entry['md5'] for entry in metadata_cache if 'md5' in entry}
        
        self.logger.info(
            f"[DEDUP] Initialized with {len(self.existing_urls)} URLs "
            f"and {len(self.existing_md5s)} MD5 hashes"
        )
    
    @staticmethod
    def compute_md5(file_path: str) -> str:
        """
        Compute MD5 hash of a file
        
        Args:
            file_path: Path to file
        
        Returns:
            MD5 hash as hex string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def download_to_temp(self, file_url: str, timeout: int = 60) -> Optional[str]:
        """
        Download file to temporary location
        
        Args:
            file_url: URL to download
            timeout: Request timeout in seconds
        
        Returns:
            Path to temporary file, or None if download failed
        """
        try:
            self.logger.debug(f"[DEDUP] Downloading to temp: {file_url}")
            
            response = requests.get(file_url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Create temp file
            suffix = Path(file_url).suffix or '.pdf'
            temp_file = tempfile.NamedTemporaryFile(
                mode='wb',
                suffix=suffix,
                delete=False
            )
            
            # Download in chunks
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            
            temp_file.close()
            
            self.logger.debug(f"[DEDUP] Downloaded to: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            self.logger.error(f"[DEDUP] Download failed: {e}")
            return None
    
    def s3_file_exists(self, s3_key: str) -> bool:
        """
        Check if file exists in S3
        
        Args:
            s3_key: S3 key to check
        
        Returns:
            True if file exists
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except Exception:
            return False
    
    def check_duplication(
        self,
        file_url: str,
        filename: str,
        publish_date: str
    ) -> Tuple[bool, str, Optional[str]]:
        """
        3-tier duplication check
        
        Args:
            file_url: Full URL of the file
            filename: Filename to use for S3 path
            publish_date: Publish date (YYYY-MM-DD) for S3 path
        
        Returns:
            Tuple of:
                - is_duplicate (bool): True if file already exists
                - reason (str): Reason for duplication or "new"
                - temp_file_path (str or None): Path to temp file if downloaded (for reuse)
        """
        # TIER 1: Check URL in metadata (instant)
        if file_url in self.existing_urls:
            self.logger.info(f"[DEDUP] ✓ Duplicate found (URL match): {filename}")
            return (True, "URL exists in metadata", None)
        
        # TIER 2: Download to temp and compute MD5
        temp_file = self.download_to_temp(file_url)
        if not temp_file:
            # Download failed, cannot verify
            self.logger.warning(f"[DEDUP] Download failed, treating as new: {filename}")
            return (False, "Download failed, treating as new", None)
        
        try:
            file_md5 = self.compute_md5(temp_file)
            self.logger.debug(f"[DEDUP] Computed MD5: {file_md5}")
            
            if file_md5 in self.existing_md5s:
                # MD5 exists, cleanup temp and skip
                os.unlink(temp_file)
                self.logger.info(f"[DEDUP] ✓ Duplicate found (MD5 match): {filename}")
                return (True, "MD5 exists in metadata", None)
            
            # TIER 3: Check S3 path (edge case - file in S3 but not in metadata)
            from datetime import datetime
            try:
                date_obj = datetime.strptime(publish_date, "%Y-%m-%d")
                year = date_obj.year
                month = date_obj.strftime("%B").lower()
            except Exception:
                # Use current date as fallback
                now = datetime.now()
                year = now.year
                month = now.strftime("%B").lower()
            
            s3_key = f"data/mospi_web/{year}/{month}/pdfs/{filename}"
            
            if self.s3_file_exists(s3_key):
                # File exists in S3 but not in metadata (orphaned)
                self.logger.warning(
                    f"[DEDUP] ✓ File exists in S3 but missing from metadata: {filename}"
                )
                # Note: Caller should add this to metadata without reprocessing
                os.unlink(temp_file)
                return (True, "File exists in S3 (orphaned)", None)
            
            # NEW DOCUMENT: Not in URL cache, not in MD5 cache, not in S3
            self.logger.info(f"[DEDUP] ✗ New document (MD5: {file_md5}): {filename}")
            return (False, "New document", temp_file)
            
        except Exception as e:
            # Error during MD5 computation or S3 check
            self.logger.error(f"[DEDUP] Error during check: {e}")
            # Cleanup temp file
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
            return (False, f"Error during check: {e}", None)
    
    def batch_check(
        self,
        files_to_check: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, bool, str, Optional[str]]]:
        """
        Check multiple files for duplication
        
        Args:
            files_to_check: List of (file_url, filename, publish_date) tuples
        
        Returns:
            List of (file_url, is_duplicate, reason, temp_file_path) tuples
        """
        results = []
        
        for file_url, filename, publish_date in files_to_check:
            is_dup, reason, temp_path = self.check_duplication(
                file_url, filename, publish_date
            )
            results.append((file_url, is_dup, reason, temp_path))
        
        # Summary stats
        duplicates = sum(1 for _, is_dup, _, _ in results if is_dup)
        new_files = len(results) - duplicates
        
        self.logger.info(
            f"[DEDUP] Batch check complete: {new_files} new, {duplicates} duplicates"
        )
        
        return results


def create_checker(
    metadata_cache: List[Dict],
    s3_client,
    bucket_name: str,
    logger: Optional[logging.Logger] = None
) -> DuplicationChecker:
    """Create a new DuplicationChecker instance"""
    return DuplicationChecker(metadata_cache, s3_client, bucket_name, logger)
