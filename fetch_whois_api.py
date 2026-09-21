#!/usr/bin/env python3
"""
Who's Who API Fetcher
Downloads Who's Who directory from API, converts to markdown, and uploads to S3
Replaces web scraper's Who's Who functionality
"""

import os
import sys
import json
import logging
import requests
import tempfile
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add web_scrap to path for S3 helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_scrap'))
from s3_helper import get_s3_client, upload_file_to_s3, download_file_from_s3, get_s3_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WhosWhoAPIFetcher:
    """
    Fetches Who's Who from API, converts to markdown, and uploads to S3
    """
    
    WHOIS_API_URL = "https://www.mospi.gov.in/api/about-us/fetch-all-WhoIsWho-Classic"
    
    def __init__(self):
        """Initialize Who's Who fetcher with S3 client"""
        self.s3_config = get_s3_config()
        self.s3_client = get_s3_client()
        self.bucket_name = self.s3_config['bucket_name']
        self.s3_prefix = self.s3_config['s3_prefix']  # data/mospi_web
        
        logger.info(f"[WHOIS] Initialized - S3 bucket: {self.bucket_name}")
    
    def fetch_whois_from_api(self) -> Optional[List[Dict]]:
        """
        Fetch Who's Who directory from API
        
        Returns:
            List of officer records or None if failed
        """
        logger.info(f"[WHOIS] Fetching Who's Who from API...")
        
        try:
            response = requests.get(
                self.WHOIS_API_URL,
                params={"lang": "en"},  # English only
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, list):
                # Direct list response
                officers = data
            elif isinstance(data, dict) and data.get('data'):
                # Wrapped response
                officers = data['data']
            else:
                logger.error(f"[WHOIS] Unexpected API response format")
                return None
            
            logger.info(f"[WHOIS] API returned {len(officers)} officer records")
            
            # Validate English only (no Hindi characters)
            all_text = json.dumps(officers)
            has_hindi = bool(re.search(r'[\u0900-\u097F]', all_text))
            
            if has_hindi:
                logger.error("[WHOIS] ERROR: API returned Hindi content!")
                return None
            
            logger.info("[WHOIS] Content verified: English only")
            
            return officers
            
        except Exception as e:
            logger.error(f"[WHOIS] Failed to fetch from API: {e}")
            return None
    
    def group_by_designation(self, officers: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group officers by designation category (similar to web scraper sections)
        
        Args:
            officers: List of officer records
        
        Returns:
            Dictionary of {section_name: [officers]}
        """
        logger.info("[WHOIS] Grouping officers by designation...")
        
        grouped = {}
        
        for officer in officers:
            # Use designation_filter as section name (matches web scraper grouping)
            section = officer.get('designation_filter', 'Other Officers')
            
            if not section or section == 'null':
                section = 'Other Officers'
            
            if section not in grouped:
                grouped[section] = []
            
            grouped[section].append(officer)
        
        logger.info(f"[WHOIS] Grouped into {len(grouped)} sections")
        for section, officers_list in grouped.items():
            logger.info(f"  - {section}: {len(officers_list)} officers")
        
        return grouped
    
    def convert_to_markdown(self, grouped_officers: Dict[str, List[Dict]], scraped_at: str) -> str:
        """
        Convert grouped officers to markdown format (matches web scraper output)
        
        Args:
            grouped_officers: Dictionary of {section_name: [officers]}
            scraped_at: Timestamp string
        
        Returns:
            Markdown formatted content
        """
        logger.info("[WHOIS] Converting to markdown format...")
        
        lines = []
        lines.append("# Who's Who - MoSPI Directory")
        lines.append(f"\nScraped on: {scraped_at}\n")
        lines.append("---\n")
        
        # Define column order and headers (match web scraper)
        headers = [
            "Seniority Order",
            "Name",
            "Designation",
            "Division",
            "Address",
            "Contact No",
            "Email ID"
        ]
        
        field_mapping = {
            "Seniority Order": "seniority_order",
            "Name": "name",
            "Designation": "designation",
            "Division": "division",
            "Address": "address",
            "Contact No": "contact_no",
            "Email ID": "email_id"
        }
        
        for section_name, officers in grouped_officers.items():
            if not officers:
                continue
            
            lines.append(f"## {section_name}\n")
            
            # Create table header
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            
            # Add rows
            for officer in officers:
                values = []
                for header in headers:
                    field_name = field_mapping[header]
                    value = str(officer.get(field_name, ""))
                    # Escape pipe characters and remove newlines
                    value = value.replace("|", "\\|").replace("\n", " ")
                    values.append(value)
                
                lines.append("| " + " | ".join(values) + " |")
            
            lines.append("\n")
        
        content = "\n".join(lines)
        logger.info(f"[WHOIS] Markdown created: {len(content)} characters")
        
        return content
    
    def upload_to_s3(self, markdown_content: str) -> Dict:
        """
        Upload Who's Who markdown to S3 with timestamped filename
        
        Args:
            markdown_content: Markdown content string
        
        Returns:
            Dictionary with upload results
        """
        # Create filename with timestamp (matches web scraper pattern)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"whois_{timestamp}.md"
        
        # S3 path: data/mospi_web/whois_who/whois_YYYYMMDD_HHMMSS.md
        s3_key = f"{self.s3_prefix}/whois_who/{filename}"
        
        logger.info(f"[WHOIS] Uploading to S3: {s3_key}")
        
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.md', delete=False) as temp_file:
                temp_file.write(markdown_content)
                temp_path = temp_file.name
            
            # Upload to S3
            result = upload_file_to_s3(
                self.s3_client,
                temp_path,
                self.bucket_name,
                s3_key
            )
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            if result["success"]:
                s3_uri = f"s3://{self.bucket_name}/{s3_key}"
                logger.info(f"[WHOIS] Upload successful: {s3_uri}")
                
                return {
                    "success": True,
                    "s3_uri": s3_uri,
                    "s3_key": s3_key,
                    "filename": filename,
                    "size": len(markdown_content)
                }
            else:
                logger.error(f"[WHOIS] Upload failed: {result['error']}")
                return {"success": False, "error": result['error']}
                
        except Exception as e:
            logger.error(f"[WHOIS] Upload error: {e}")
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            return {"success": False, "error": str(e)}
    
    def update_metadata_file(self, upload_result: Dict, officer_count: int) -> bool:
        """
        Update whois_metadata.json in S3 to track Who's Who documents
        
        Args:
            upload_result: Upload result dictionary
            officer_count: Number of officers in the directory
        
        Returns:
            True if successful
        """
        metadata_s3_key = f"{self.s3_prefix}/whois_who/whois_metadata.json"
        
        logger.info(f"[WHOIS] Updating metadata file: {metadata_s3_key}")
        
        # Download existing metadata
        existing_metadata = []
        temp_download_path = None
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                temp_download_path = tmp.name
            
            if download_file_from_s3(self.s3_client, self.bucket_name, metadata_s3_key, temp_download_path):
                with open(temp_download_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
                logger.info(f"[WHOIS] Loaded {len(existing_metadata)} existing metadata entries")
            else:
                logger.info(f"[WHOIS] No existing metadata, creating new")
            
            # Cleanup download temp file
            if temp_download_path and os.path.exists(temp_download_path):
                os.unlink(temp_download_path)
        
        except Exception as e:
            logger.warning(f"[WHOIS] Could not load existing metadata: {e}")
            if temp_download_path and os.path.exists(temp_download_path):
                os.unlink(temp_download_path)
        
        # Create new metadata entry
        metadata_entry = {
            "file_name": upload_result['filename'],
            "s3_uri": upload_result['s3_uri'],
            "s3_key": upload_result['s3_key'],
            "file_size": upload_result['size'],
            "officer_count": officer_count,
            "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "whois_api"
        }
        
        # Append new entry
        existing_metadata.append(metadata_entry)
        
        # Upload updated metadata
        temp_upload_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                temp_upload_path = tmp.name
                json.dump(existing_metadata, tmp, indent=2, ensure_ascii=False)
            
            result = upload_file_to_s3(
                self.s3_client,
                temp_upload_path,
                self.bucket_name,
                metadata_s3_key
            )
            
            # Cleanup upload temp file
            if temp_upload_path and os.path.exists(temp_upload_path):
                os.unlink(temp_upload_path)
            
            if result["success"]:
                logger.info(f"[WHOIS] Metadata updated successfully")
                return True
            else:
                logger.error(f"[WHOIS] Metadata update failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"[WHOIS] Metadata update error: {e}")
            if temp_upload_path and os.path.exists(temp_upload_path):
                os.unlink(temp_upload_path)
            return False
    
    def fetch_and_upload(self) -> Dict:
        """
        Complete workflow: Fetch from API, convert to markdown, upload to S3
        
        Returns:
            Dictionary with results
        """
        logger.info("=" * 80)
        logger.info("[WHOIS] Starting Who's Who fetch and upload workflow")
        logger.info("=" * 80)
        
        # Step 1: Fetch from API
        officers = self.fetch_whois_from_api()
        if not officers:
            return {"success": False, "error": "Failed to fetch Who's Who from API"}
        
        officer_count = len(officers)
        
        # Step 2: Group by designation
        grouped_officers = self.group_by_designation(officers)
        
        # Step 3: Convert to markdown
        scraped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        markdown_content = self.convert_to_markdown(grouped_officers, scraped_at)
        
        # Step 4: Upload to S3
        upload_result = self.upload_to_s3(markdown_content)
        
        if not upload_result.get("success"):
            return upload_result
        
        # Step 5: Update metadata
        self.update_metadata_file(upload_result, officer_count)
        
        logger.info("=" * 80)
        logger.info("[WHOIS] Workflow completed successfully")
        logger.info("=" * 80)
        logger.info(f"Officers fetched: {officer_count}")
        logger.info(f"Sections: {len(grouped_officers)}")
        logger.info(f"Markdown size: {upload_result['size']:,} characters")
        logger.info(f"Filename: {upload_result['filename']}")
        logger.info(f"S3 URI: {upload_result['s3_uri']}")
        logger.info("=" * 80)
        
        upload_result["officer_count"] = officer_count
        upload_result["sections"] = len(grouped_officers)
        return upload_result


def main():
    """Run Who's Who fetch and upload"""
    fetcher = WhosWhoAPIFetcher()
    result = fetcher.fetch_and_upload()
    
    if result.get("success"):
        print("\n✓ Who's Who fetched and uploaded successfully!")
        print(f"  Officers: {result.get('officer_count')}")
        print(f"  Sections: {result.get('sections')}")
        print(f"  File: {result.get('filename')}")
        return 0
    else:
        print(f"\n✗ Who's Who fetch failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
