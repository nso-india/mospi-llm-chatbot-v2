#!/usr/bin/env python3
"""
Data models for API ingestion
Unified structure for handling different API response formats
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class FileMetadata:
    """File metadata from API response"""
    
    path: str  # Relative path from API (e.g., "uploads/latestReleases/...")
    filename: str  # Original filename
    filesize: int  # Size in bytes
    filemime: str  # MIME type
    
    # Computed fields
    full_url: Optional[str] = None
    md5_hash: Optional[str] = None
    
    def get_full_url(self, base_url: str = "https://www.mospi.gov.in/") -> str:
        """
        Build full download URL from relative path
        
        Args:
            base_url: Base URL to prepend
        
        Returns:
            Full URL for downloading the file
        """
        if self.full_url:
            return self.full_url
        
        # Handle absolute URLs (edge case)
        if self.path.startswith('http'):
            self.full_url = self.path
            return self.full_url
        
        # Handle relative paths
        if self.path.startswith('/'):
            self.full_url = f"{base_url.rstrip('/')}{self.path}"
        else:
            self.full_url = f"{base_url}{self.path}"
        
        return self.full_url
    
    def is_pdf(self) -> bool:
        """Check if file is a PDF"""
        return (
            self.filemime == "application/pdf" or
            self.filename.lower().endswith('.pdf') or
            self.path.lower().endswith('.pdf')
        )
    
    def is_excel(self) -> bool:
        """Check if file is Excel"""
        excel_mimes = [
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ]
        return (
            self.filemime in excel_mimes or
            self.filename.lower().endswith(('.xls', '.xlsx'))
        )
    
    def extract_filename_from_path(self) -> str:
        """
        Extract filename from path (handles both API filename and path)
        Prefers the filename from path (has timestamp+UUID) over clean filename
        
        Returns:
            Filename with timestamp+UUID for uniqueness
        """
        # Extract from path (e.g., "uploads/.../latest_release_123_uuid_file.pdf")
        path_filename = self.path.split('/')[-1]
        
        # Return path filename (guaranteed unique with timestamp+UUID)
        return path_filename


@dataclass
class APIDocument:
    """Unified document model for all API responses"""
    
    # Core fields (required)
    id: str
    title: str
    publish_date: str  # YYYY-MM-DD format
    
    # File metadata (at least one file required)
    files: List[FileMetadata] = field(default_factory=list)
    
    # Optional fields
    is_active: bool = True
    is_new: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Source tracking
    source_api: str = ""  # e.g., "latest_releases", "announcements"
    source_endpoint: str = ""
    
    @classmethod
    def from_api_response(cls, data: dict, source_api: str = "") -> "APIDocument":
        """
        Create APIDocument from raw API response
        
        Args:
            data: Raw API response dictionary
            source_api: Name of the source API
        
        Returns:
            APIDocument instance
        """
        # Extract files
        files = []
        for file_field in ['file_one', 'file_two', 'file_three']:
            file_data = data.get(file_field)
            if file_data and isinstance(file_data, dict):
                files.append(FileMetadata(
                    path=file_data.get('path', ''),
                    filename=file_data.get('filename', ''),
                    filesize=file_data.get('filesize', 0),
                    filemime=file_data.get('filemime', '')
                ))
        
        # Extract publish date (different field names per API)
        # Priority: published_year > published_date > created_at > current date
        publish_date = (
            data.get('published_year') or      # Most APIs
            data.get('published_date') or      # Chapter Data API
            data.get('created_at') or          # SSS API
            datetime.now().strftime("%Y-%m-%d")  # Fallback to current date
        )
        
        # Extract title (different field names per API)
        title = (
            data.get('title') or               # Most APIs
            data.get('chapter_title') or       # Chapter Data API
            data.get('file_title') or          # SSS API
            'Untitled Document'                # Fallback
        ).strip()
        
        # Extract ID (some APIs return None for id)
        doc_id = str(data.get('id') or data.get('chapter_id') or '')
        
        return cls(
            id=doc_id,
            title=title,
            publish_date=publish_date,
            files=files,
            is_active=data.get('is_active', True),
            is_new=data.get('is_new'),
            start_date=data.get('start_date'),
            end_date=data.get('end_date'),
            source_api=source_api
        )
    
    def get_pdf_files(self) -> List[FileMetadata]:
        """Get only PDF files from the document"""
        return [f for f in self.files if f.is_pdf()]
    
    def get_excel_files(self) -> List[FileMetadata]:
        """Get only Excel files from the document"""
        return [f for f in self.files if f.is_excel()]
    
    def has_files(self) -> bool:
        """Check if document has any files"""
        return len(self.files) > 0
    
    def extract_date_components(self) -> tuple:
        """
        Extract year and month from publish_date
        
        Returns:
            (year: int, month: str) - month is lowercase full name (e.g., "january")
        """
        try:
            date_obj = datetime.strptime(self.publish_date, "%Y-%m-%d")
            year = date_obj.year
            month = date_obj.strftime("%B").lower()  # Full month name
            return (year, month)
        except Exception:
            # Fallback to current date if parse fails
            now = datetime.now()
            return (now.year, now.strftime("%B").lower())
    
    def to_metadata_entry(self, file_metadata: FileMetadata, md5_hash: str, s3_uri: str) -> dict:
        """
        Create metadata entry for web_file_metadata.json
        
        Args:
            file_metadata: FileMetadata for specific file
            md5_hash: Computed MD5 hash of the file
            s3_uri: S3 URI where file is stored
        
        Returns:
            Metadata dictionary matching existing format
        """
        return {
            "file_url": file_metadata.get_full_url(),
            "title": self.title,
            "publish_date": self.publish_date,
            "file_name": file_metadata.extract_filename_from_path(),
            "md5": md5_hash,
            "file_size": file_metadata.filesize,
            "text_uri": s3_uri,
            "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "api_ingestion",
            "source_api": self.source_api,
            "api_id": self.id
        }
