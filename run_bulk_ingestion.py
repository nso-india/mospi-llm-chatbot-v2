#!/usr/bin/env python3
"""
Bulk API Ingestion Script
Runs ingestion for all configured APIs
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Ensure logs directory exists
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

# Setup logging
log_file = log_dir / f'bulk_ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file))
    ]
)

logger = logging.getLogger(__name__)

# Import pipeline
from api_ingestion.pipeline import run_ingestion


# Configuration
API_CONFIGS = {
    "latest_releases": {
        "enabled": True,
        "max_results": 20,
        "description": "Press releases (CPI, GDP, IIP, PLFS)"
    },
    "announcements": {
        "enabled": True,
        "max_results": 20,
        "description": "Official announcements and notices"
    },
    "publications_reports": {
        "enabled": True,
        "max_results": 20,
        "description": "Statistical publications and reports"
    },
    "public_docs": {
        "enabled": True,
        "max_results": 20,
        "description": "Public documents and circulars"
    },
    "iss": {
        "enabled": True,
        "max_results": 20,
        "description": "ISS orders and civil lists"
    },
    "sss": {
        "enabled": True,
        "max_results": 20,
        "description": "SSS vacancy positions"
    },
    "visualizations": {
        "enabled": False,  # DISABLED: Different structure, not compatible
        "max_results": 20,
        "description": "Charts, infographics, and visualizations"
    },
    "chapter_data": {
        "enabled": True,
        "max_results": 20,
        "description": "Publication chapter data"
    }
}


def run_bulk_ingestion(dry_run: bool = False):
    """
    Run ingestion for all enabled APIs
    
    Args:
        dry_run: If True, don't actually upload to S3
    """
    logger.info("=" * 80)
    logger.info("BULK API INGESTION")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("")
    
    # Filter enabled APIs
    enabled_apis = {
        name: config for name, config in API_CONFIGS.items()
        if config["enabled"]
    }
    
    logger.info(f"APIs to process: {len(enabled_apis)}")
    for api_name, config in enabled_apis.items():
        logger.info(f"  - {api_name}: {config['description']} (max {config['max_results']})")
    logger.info("")
    
    # Run ingestion for each API
    results = {}
    total_stats = {
        "fetched": 0,
        "duplicates": 0,
        "new_files": 0,
        "uploaded": 0,
        "errors": 0
    }
    
    for api_name, config in enabled_apis.items():
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing: {api_name}")
        logger.info("=" * 80)
        
        try:
            result = run_ingestion(
                api_name=api_name,
                max_results=config["max_results"],
                dry_run=dry_run,
                logger=logger
            )
            
            results[api_name] = result
            
            # Aggregate stats
            if result.get("success"):
                stats = result.get("stats", {})
                for key in total_stats.keys():
                    total_stats[key] += stats.get(key, 0)
                
                logger.info(f"✓ {api_name} completed")
            else:
                logger.error(f"✗ {api_name} failed: {result.get('error')}")
                total_stats["errors"] += 1
                
        except Exception as e:
            logger.error(f"✗ {api_name} error: {e}")
            results[api_name] = {"success": False, "error": str(e)}
            total_stats["errors"] += 1
    
    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("BULK INGESTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"APIs processed: {len(enabled_apis)}")
    logger.info("")
    logger.info("Aggregate Statistics:")
    logger.info(f"  Documents fetched: {total_stats['fetched']}")
    logger.info(f"  Duplicates skipped: {total_stats['duplicates']}")
    logger.info(f"  New files found: {total_stats['new_files']}")
    logger.info(f"  Files uploaded: {total_stats['uploaded']}")
    logger.info(f"  Errors: {total_stats['errors']}")
    logger.info("")
    
    # Per-API summary
    logger.info("Per-API Results:")
    for api_name, result in results.items():
        if result.get("success"):
            stats = result.get("stats", {})
            logger.info(
                f"  ✓ {api_name}: "
                f"{stats.get('new_files', 0)} new, "
                f"{stats.get('duplicates', 0)} duplicates"
            )
        else:
            logger.info(f"  ✗ {api_name}: {result.get('error', 'Failed')}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Next steps:")
    logger.info("1. Check S3 bucket for uploaded PDFs")
    logger.info("2. Verify web_file_metadata.json updated")
    logger.info("3. Monitor batch_process_pdf_to_text.py for processing")
    logger.info("4. Check Qdrant for new document chunks")
    logger.info("=" * 80)
    
    # Return success if no errors
    return 0 if total_stats["errors"] == 0 else 1


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run bulk API ingestion")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test without uploading to S3"
    )
    
    args = parser.parse_args()
    
    return run_bulk_ingestion(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
