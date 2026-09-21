#!/usr/bin/env python3
"""
Manual script to refresh whois_cache.txt from S3
Downloads latest FOD directory markdown file and updates the cache
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Manually refresh whois cache from S3"""
    logger.info("=" * 80)
    logger.info("MANUAL WHOIS CACHE REFRESH")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now()}")
    logger.info("")
    
    try:
        # Import whois cache manager
        from whois_cache_manager import force_refresh, get_cache_status
        
        # Show current cache status
        logger.info("Current cache status:")
        status = get_cache_status()
        logger.info(f"  Has content: {status['has_content']}")
        logger.info(f"  Content length: {status['content_length']} chars")
        logger.info(f"  Last refresh: {status['last_refresh']}")
        logger.info("")
        
        # Perform refresh
        logger.info("Starting refresh from S3...")
        logger.info("This will:")
        logger.info("  1. Download whois_metadata.json from S3")
        logger.info("  2. Find latest FOD directory markdown file")
        logger.info("  3. Download and validate the markdown file")
        logger.info("  4. Convert to pipe-delimited format")
        logger.info("  5. Update whois_cache.txt")
        logger.info("")
        
        success = force_refresh()
        
        if success:
            logger.info("=" * 80)
            logger.info("✅ REFRESH SUCCESSFUL")
            logger.info("=" * 80)
            
            # Show updated cache status
            updated_status = get_cache_status()
            logger.info(f"Updated content length: {updated_status['content_length']} chars")
            logger.info(f"Last refresh: {updated_status['last_refresh']}")
            logger.info("")
            logger.info("Cache file updated: whois_cache.txt")
            logger.info("Metadata file updated: whois_cache_meta.txt")
            logger.info("")
            logger.info("The chatbot will now use the refreshed cache for whois queries.")
            
            return 0
        else:
            logger.error("=" * 80)
            logger.error("✗ REFRESH FAILED")
            logger.error("=" * 80)
            logger.error("No changes detected or S3 fetch failed.")
            logger.error("")
            logger.error("Possible reasons:")
            logger.error("  1. No new FOD directory file in S3")
            logger.error("  2. Content unchanged (same MD5)")
            logger.error("  3. S3 connection issue")
            logger.error("  4. File validation failed (non-English content)")
            logger.error("")
            logger.error("Check the logs above for details.")
            
            return 1
            
    except Exception as e:
        logger.error("=" * 80)
        logger.error("✗ ERROR")
        logger.error("=" * 80)
        logger.error(f"Exception occurred: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
