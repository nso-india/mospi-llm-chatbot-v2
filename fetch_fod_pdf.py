#!/usr/bin/env python3
"""
FOD PDF Fetcher
Downloads latest FOD PDF from API, renames it, and uploads to S3
This integrates with existing whois_cache_manager.py workflow
"""

import os
import sys
import json
import logging
import requests
import tempfile
from datetime import datetime
from pathlib import Path

# Add web_scrap to path for S3 helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_scrap'))
from s3_helper import get_s3_client, upload_file_to_s3, get_s3_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FODPDFFetcher:
    """
    Fetches FOD PDF from API and uploads to S3
    """
    
    FOD_API_URL = "https://www.mospi.gov.in/api/documents/get-fod-document"
    STANDARD_FILENAME = "Field Operations Division (FOD) Directory.pdf"
    
    def __init__(self):
        """Initialize FOD fetcher with S3 client"""
        self.s3_config = get_s3_config()
        self.s3_client = get_s3_client()
        self.bucket_name = self.s3_config['bucket_name']
        
        logger.info(f"[FOD] Initialized - S3 bucket: {self.bucket_name}")
    
    def fetch_latest_fod_from_api(self) -> dict:
        """
        Fetch latest FOD PDF metadata from API
        
        Returns:
            Dictionary with FOD document info or None if failed
        """
        logger.info(f"[FOD] Fetching latest FOD PDF from API...")
        
        try:
            response = requests.post(
                self.FOD_API_URL,
                json={"lang": "en"},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'success':
                doc = data.get('data', {})
                logger.info(
                    f"[FOD] API returned: {doc.get('filename')} "
                    f"(ID: {doc.get('id')}, Size: {doc.get('size')} bytes)"
                )
                
                return {
                    'id': doc.get('id'),
                    'original_filename': doc.get('filename'),
                    'size': int(doc.get('size', 0)),
                    'url': doc.get('url'),
                    'created_at': doc.get('created_at'),
                    'full_url': f"{self.FOD_API_URL.split('/api')[0]}/{doc.get('url')}"
                }
            else:
                logger.error(f"[FOD] API returned error: {data.get('message')}")
                return None
                
        except Exception as e:
            logger.error(f"[FOD] Failed to fetch from API: {e}")
            return None
    
    def download_fod_pdf(self, fod_info: dict) -> str:
        """
        Download FOD PDF to temporary file
        
        Args:
            fod_info: FOD document info from API
        
        Returns:
            Path to temporary file or None if failed
        """
        logger.info(f"[FOD] Downloading PDF from: {fod_info['full_url']}")
        
        try:
            response = requests.get(
                fod_info['full_url'],
                timeout=60,
                stream=True
            )
            response.raise_for_status()
            
            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(
                mode='wb',
                suffix='.pdf',
                delete=False
            )
            
            # Download in chunks
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            
            temp_file.close()
            
            file_size = os.path.getsize(temp_file.name)
            logger.info(f"[FOD] Downloaded to temp file: {temp_file.name} ({file_size:,} bytes)")
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"[FOD] Download failed: {e}")
            return None
    
    def upload_to_s3(self, temp_file_path: str, fod_info: dict) -> dict:
        """
        Upload FOD PDF to S3 with standardized filename
        
        Args:
            temp_file_path: Path to downloaded PDF
            fod_info: FOD document info
        
        Returns:
            Dictionary with upload results
        """
        # Use standardized filename
        filename = self.STANDARD_FILENAME
        
        # S3 path: data/mospi_web/fod_directory/pdfs/ (matches web scraper)
        s3_prefix = self.s3_config['s3_prefix']  # data/mospi_web
        s3_key = f"{s3_prefix}/fod_directory/pdfs/{filename}"
        
        logger.info(f"[FOD] Uploading to S3: {s3_key}")
        
        try:
            result = upload_file_to_s3(
                self.s3_client,
                temp_file_path,
                self.bucket_name,
                s3_key
            )
            
            if result["success"]:
                s3_uri = f"s3://{self.bucket_name}/{s3_key}"
                logger.info(f"[FOD] Upload successful: {s3_uri}")
                
                return {
                    "success": True,
                    "s3_uri": s3_uri,
                    "s3_key": s3_key,
                    "filename": filename,
                    "original_filename": fod_info['original_filename'],
                    "api_id": fod_info['id'],
                    "created_at": fod_info['created_at'],
                    "size": fod_info['size']
                }
            else:
                logger.error(f"[FOD] Upload failed: {result['error']}")
                return {"success": False, "error": result['error']}
                
        except Exception as e:
            logger.error(f"[FOD] Upload error: {e}")
            return {"success": False, "error": str(e)}
    
    def update_metadata_file(self, upload_result: dict) -> bool:
        """
        Update fod_metadata.json in S3 to track FOD document
        
        Args:
            upload_result: Upload result dictionary
        
        Returns:
            True if successful
        """
        s3_prefix = self.s3_config['s3_prefix']
        metadata_s3_key = f"{s3_prefix}/fod_directory/pdfs/fod_metadata.json"
        
        logger.info(f"[FOD] Updating metadata file: {metadata_s3_key}")
        
        # Create metadata entry
        metadata_entry = {
            "filename": upload_result['filename'],
            "original_filename": upload_result['original_filename'],
            "s3_uri": upload_result['s3_uri'],
            "s3_key": upload_result['s3_key'],
            "api_id": upload_result['api_id'],
            "api_created_at": upload_result['created_at'],
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "size": upload_result['size'],
            "source": "fod_api"
        }
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
            json.dump(metadata_entry, temp_file, indent=2, ensure_ascii=False)
        
        try:
            # Upload metadata to S3
            result = upload_file_to_s3(
                self.s3_client,
                temp_path,
                self.bucket_name,
                metadata_s3_key
            )
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            if result["success"]:
                logger.info(f"[FOD] Metadata updated successfully")
                return True
            else:
                logger.error(f"[FOD] Metadata update failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"[FOD] Metadata update error: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return False
    
    def fetch_and_upload(self) -> dict:
        """
        Complete workflow: Fetch FOD from API, download, rename, upload to S3
        
        Returns:
            Dictionary with results
        """
        logger.info("=" * 80)
        logger.info("[FOD] Starting FOD PDF fetch and upload workflow")
        logger.info("=" * 80)
        
        # Step 1: Fetch FOD info from API
        fod_info = self.fetch_latest_fod_from_api()
        if not fod_info:
            return {"success": False, "error": "Failed to fetch FOD info from API"}
        
        # Step 2: Download PDF
        temp_file = self.download_fod_pdf(fod_info)
        if not temp_file:
            return {"success": False, "error": "Failed to download FOD PDF"}
        
        try:
            # Step 3: Upload to S3 with standardized filename
            upload_result = self.upload_to_s3(temp_file, fod_info)
            
            if not upload_result.get("success"):
                return upload_result
            
            # Step 4: Update metadata file
            self.update_metadata_file(upload_result)
            
            # Cleanup temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            
            logger.info("=" * 80)
            logger.info("[FOD] Workflow completed successfully")
            logger.info("=" * 80)
            logger.info(f"Standardized filename: {upload_result['filename']}")
            logger.info(f"Original filename: {upload_result['original_filename']}")
            logger.info(f"S3 URI: {upload_result['s3_uri']}")
            logger.info(f"API ID: {upload_result['api_id']}")
            logger.info(f"Created: {upload_result['created_at']}")
            logger.info("=" * 80)
            
            return upload_result
            
        except Exception as e:
            logger.error(f"[FOD] Workflow error: {e}")
            # Cleanup temp file
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
            return {"success": False, "error": str(e)}


def main():
    """Run FOD fetch and upload"""
    fetcher = FODPDFFetcher()
    result = fetcher.fetch_and_upload()
    
    if result.get("success"):
        print("\n✓ FOD PDF fetched and uploaded successfully!")
        return 0
    else:
        print(f"\n✗ FOD fetch failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
