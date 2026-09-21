#!/usr/bin/env python3
"""
MoSPI API Client
Handles API requests to MoSPI document endpoints with retry logic and rate limiting
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime


class MoSPIAPIClient:
    """
    Client for MoSPI document APIs
    Supports both GET and POST methods with pagination
    """
    
    BASE_URL = "https://www.mospi.gov.in"
    
    # API Configuration Registry
    API_ENDPOINTS = {
        "latest_releases": {
            "method": "POST",
            "endpoint": "/api/latest-release/get-web-latest-release-list",
            "supports_pagination": True,
            "default_page_size": 10
        },
        "announcements": {
            "method": "POST",
            "endpoint": "/api/announcement/get-web-announcement-list",
            "supports_pagination": True,
            "default_page_size": 10
        },
        "publications_reports": {
            "method": "POST",
            "endpoint": "/api/publications-reports/get-web-publications-report-list",
            "supports_pagination": True,
            "default_page_size": 10
        },
        "public_docs": {
            "method": "POST",
            "endpoint": "/api/public-doc/get-web-pub-doc-list",
            "supports_pagination": True,
            "default_page_size": 10
        },
        "iss": {
            "method": "POST",
            "endpoint": "/api/iss/get-web-iss-list",
            "supports_pagination": True,
            "default_page_size": 10
        },
        "sss": {
            "method": "POST",
            "endpoint": "/api/sss/get-sss-list-web",
            "supports_pagination": True,
            "default_page_size": 10
        },
        "visualizations": {
            "method": "POST",
            "endpoint": "/api/visualization/get-web-visualization-list",
            "supports_pagination": True,
            "default_page_size": 10
        },
        "whois": {
            "method": "GET",
            "endpoint": "/api/about-us/fetch-all-WhoIsWho-Classic",
            "supports_pagination": False,
            "params": {"lang": "en"}
        },
        "chapter_data": {
            "method": "POST",
            "endpoint": "/api/publications-reports/get-web-chapter-data",
            "supports_pagination": True,
            "default_page_size": 10
        }
    }
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 2,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize MoSPI API client
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            logger: Logger instance
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logger or logging.getLogger(__name__)
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MoSPI-APIClient/1.0",
            "Accept": "application/json"
        })
    
    def _make_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic
        
        Args:
            method: HTTP method (GET, POST)
            url: Full URL
            **kwargs: Additional request parameters
        
        Returns:
            Response JSON as dictionary
        
        Raises:
            requests.exceptions.RequestException: If request fails after retries
        """
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.debug(f"[API] {method} {url} (attempt {attempt}/{self.max_retries})")
                
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs
                )
                
                response.raise_for_status()
                
                json_data = response.json()
                
                # Check API-level status
                if json_data.get("status") == "success":
                    return json_data
                else:
                    error_msg = json_data.get("message", "Unknown API error")
                    self.logger.warning(f"[API] API returned error: {error_msg}")
                    raise requests.exceptions.HTTPError(f"API Error: {error_msg}")
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                self.logger.warning(f"[API] Timeout on attempt {attempt}: {e}")
                
            except requests.exceptions.HTTPError as e:
                last_exception = e
                self.logger.warning(f"[API] HTTP error on attempt {attempt}: {e}")
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                self.logger.warning(f"[API] Request error on attempt {attempt}: {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)  # Exponential backoff
        
        # All retries failed
        self.logger.error(f"[API] All {self.max_retries} attempts failed for {url}")
        raise last_exception
    
    def fetch_latest_releases(
        self,
        page: int = 1,
        page_size: int = 10,
        max_results: int = 20,
        sort_by_date: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch latest releases from MoSPI API
        
        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            max_results: Maximum total results to fetch (limits pagination)
            sort_by_date: If True, sort by published_year descending (most recent first)
        
        Returns:
            Dictionary with 'data' (list of documents) and 'pagination' info
        """
        config = self.API_ENDPOINTS["latest_releases"]
        url = f"{self.BASE_URL}{config['endpoint']}"
        
        self.logger.info(
            f"[API] Fetching latest releases "
            f"(page={page}, page_size={page_size}, max={max_results}, sort_by_date={sort_by_date})"
        )
        
        # Use the generic fetch method for consistency
        all_docs = self.fetch_all_from_endpoint(
            api_name="latest_releases",
            max_results=max_results,
            page_size=page_size,
            sort_by_date=sort_by_date
        )
        
        # Build response matching expected format
        return {
            "data": all_docs,
            "pagination": {
                "currentPage": 1,
                "totalItems": len(all_docs),
                "totalPages": 1,
                "pageSize": len(all_docs)
            },
            "fetched_count": len(all_docs),
            "total_available": len(all_docs)
        }
    
    def fetch_all_from_endpoint(
        self,
        api_name: str,
        max_results: int = 20,
        page_size: int = 10,
        sort_by_date: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch documents from any API endpoint (handles pagination automatically)
        
        Args:
            api_name: API name from API_ENDPOINTS registry
            max_results: Maximum total results to fetch
            page_size: Items per page
            sort_by_date: If True, sort by published_year descending (most recent first)
        
        Returns:
            List of document dictionaries (sorted by date if sort_by_date=True)
        """
        if api_name not in self.API_ENDPOINTS:
            raise ValueError(f"Unknown API: {api_name}. Available: {list(self.API_ENDPOINTS.keys())}")
        
        config = self.API_ENDPOINTS[api_name]
        method = config["method"]
        url = f"{self.BASE_URL}{config['endpoint']}"
        
        self.logger.info(f"[API] Fetching from '{api_name}' (max_results={max_results}, sort_by_date={sort_by_date})")
        
        # Handle GET requests (e.g., Who's Who)
        if method == "GET":
            params = config.get("params", {})
            response = self._make_request("GET", url, params=params)
            data = response.get("data", [])
            
            # Sort by date if requested
            if sort_by_date and data:
                data = self._sort_by_date(data)
            
            # Limit to max_results
            if len(data) > max_results:
                data = data[:max_results]
            
            self.logger.info(f"[API] Fetched {len(data)} documents from '{api_name}'")
            return data
        
        # Handle POST requests with pagination. API doesn't guarantee date sorting, so we need to:
        # 1. Fetch ALL available documents (or reasonable limit)
        # 2. Sort by date
        # 3. Return top N
        
        all_documents = []
        current_page = 1
        max_pages_to_fetch = 20  # Safety limit to avoid fetching too many
        
        self.logger.info(f"[API] Fetching all documents from '{api_name}' for date sorting...")
        
        while current_page <= max_pages_to_fetch:
            try:
                response = self._make_request("POST", url)
                
                data = response.get("data", [])
                pagination = response.get("pagination", {})
                
                if not data:
                    self.logger.info(f"[API] No more data from '{api_name}'")
                    break
                
                # Add all documents from this page
                all_documents.extend(data)
                
                self.logger.info(
                    f"[API] Page {current_page}: Fetched {len(data)} documents "
                    f"(total so far: {len(all_documents)})"
                )
                
                # Check if there are more pages
                total_pages = pagination.get("totalPages", 1)
                total_items = pagination.get("totalItems", len(all_documents))
                
                if current_page >= total_pages:
                    self.logger.info(f"[API] Reached last page ({total_pages})")
                    break
                
                current_page += 1
                
            except Exception as e:
                self.logger.error(f"[API] Error fetching page {current_page} from '{api_name}': {e}")
                break
        
        self.logger.info(f"[API] Total fetched from '{api_name}': {len(all_documents)} documents")
        
        # Sort by date (most recent first) if requested
        if sort_by_date and all_documents:
            self.logger.info(f"[API] Sorting {len(all_documents)} documents by date (most recent first)...")
            all_documents = self._sort_by_date(all_documents)
        
        # Return top N most recent
        result = all_documents[:max_results]
        
        if result:
            # Get date from first/last document (handle different field names)
            first_date = (
                result[0].get('published_year') or
                result[0].get('published_date') or
                result[0].get('created_at') or
                'N/A'
            )
            last_date = (
                result[-1].get('published_year') or
                result[-1].get('published_date') or
                result[-1].get('created_at') or
                'N/A'
            )
            self.logger.info(
                f"[API] Returning top {len(result)} most recent documents "
                f"(from {first_date} to {last_date})"
            )
        
        return result
    
    def _sort_by_date(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort documents by date field (most recent first)
        Handles multiple date field names: published_year, published_date, created_at
        
        Args:
            documents: List of document dictionaries
        
        Returns:
            Sorted list (descending by date)
        """
        from datetime import datetime
        
        def get_sort_key(doc):
            """Extract date for sorting, handle multiple field names and parsing errors"""
            try:
                # Try different date field names (priority order)
                date_str = (
                    doc.get('published_year') or
                    doc.get('published_date') or
                    doc.get('created_at') or
                    ''
                )
                
                if date_str:
                    # Try to parse YYYY-MM-DD format
                    return datetime.strptime(date_str, "%Y-%m-%d")
                else:
                    # No date, push to end
                    return datetime.min
            except Exception:
                # Parse error, push to end
                return datetime.min
        
        try:
            sorted_docs = sorted(documents, key=get_sort_key, reverse=True)
            self.logger.debug(f"[API] Sorted {len(sorted_docs)} documents by date")
            return sorted_docs
        except Exception as e:
            self.logger.warning(f"[API] Failed to sort by date: {e}, returning unsorted")
            return documents
    
    def close(self):
        """Close the session"""
        self.session.close()


# Convenience function
def create_client(logger: Optional[logging.Logger] = None) -> MoSPIAPIClient:
    """Create a new MoSPI API client instance"""
    return MoSPIAPIClient(logger=logger)
