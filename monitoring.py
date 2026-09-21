"""
Monitoring utilities for MoSPI Chatbot
Handles health checks and ingestion reporting
"""

import os
import json
import logging
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# ---- Configuration ----
MONITORING_EMAIL_API_URL = os.getenv(
    "MONITORING_EMAIL_API_URL", 
    "https://datainnovation.mospi.gov.in/api/sendmail"
)
MONITORING_EMAIL_AUTH_KEY = os.getenv("MONITORING_EMAIL_AUTH_KEY", "Dq2iLWcR")
MONITORING_EMAIL_FROM = os.getenv("MONITORING_EMAIL_FROM", "vishnu.mishra88@mospi.gov.in")
MONITORING_EMAIL_TO = os.getenv("MONITORING_EMAIL_TO", "sachin@senpiper.com")
# HEALTH_CHECK_QUERY = os.getenv("HEALTH_CHECK_QUERY", "what is MoSPI")
# HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "1800"))  # 30 minutes
SEND_EMPTY_REPORTS = os.getenv("SEND_EMPTY_REPORTS", "0") == "1"  # Send email even if no files processed

PROCESSED_CHUNK_LOGS_PATH = Path("processed_chunk_logs")

# IST timezone offset
IST_OFFSET = timedelta(hours=5, minutes=30)

# Rate limiting for health check emails
last_health_check_email_time: Optional[datetime] = None
HEALTH_CHECK_EMAIL_RATE_LIMIT = timedelta(hours=2)


# ---- Logging Setup ----
def setup_monitoring_logger() -> logging.Logger:
    """Setup logger for monitoring operations."""
    logger = logging.getLogger("monitoring")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(console_handler)
        
        # File handler - write to logs/monitoring.log
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / "monitoring.log",
                mode='a',
                encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")
    
    return logger


logger = setup_monitoring_logger()


# ---- Utility Functions ----
def utc_to_ist(utc_dt: datetime) -> datetime:
    """Convert UTC datetime to IST."""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(timezone(IST_OFFSET))


def get_ist_now() -> datetime:
    """Get current time in IST."""
    return datetime.now(timezone.utc).astimezone(timezone(IST_OFFSET))


def get_ist_date_string(dt: datetime) -> str:
    """Format datetime as IST date string (YYYY-MM-DD)."""
    ist_dt = utc_to_ist(dt) if dt.tzinfo else dt
    return ist_dt.strftime("%Y-%m-%d")


def format_date_for_email(date_str: str) -> str:
    """Convert YYYY-MM-DD to DDMmmYYYY format for email subject."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%d%b%Y")
    except Exception:
        return date_str


# ---- Email Functions ----
def send_email(subject: str, body: str, recipients: Optional[List[str]] = None, is_html: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Send email via MoSPI email API.
    
    Args:
        subject: Email subject
        body: Email body (HTML or plain text)
        recipients: List of recipient emails (defaults to env config)
        is_html: Whether body is HTML formatted (default: True)
    
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    if recipients is None:
        recipients = [email.strip() for email in MONITORING_EMAIL_TO.split(",")]
    
    payload = {
        "emailData": {
            "from": MONITORING_EMAIL_FROM,
            "to": recipients,
            "subject": subject,
            "body": body,
            "isHtml": is_html
        }
    }
    
    headers = {
        "auth-key": MONITORING_EMAIL_AUTH_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        logger.info(f"Sending email: {subject}")
        response = requests.post(
            MONITORING_EMAIL_API_URL,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"✓ Email sent successfully: {subject}")
            return True, None
        else:
            error_msg = f"Email API returned status {response.status_code}: {response.text}"
            logger.error(f"✗ Failed to send email: {error_msg}")
            return False, error_msg
            
    except Exception as e:
        error_msg = f"Exception sending email: {str(e)}"
        logger.error(f"✗ {error_msg}")
        return False, error_msg


# ---- Ingestion Report Functions ----
@dataclass
class IngestionFileEntry:
    """Represents a file entry from processed_chunk_logs."""
    filename: str
    status: str
    timestamp: str
    error: Optional[str] = None


def load_ingestion_logs_for_date(target_date: str) -> List[IngestionFileEntry]:
    """
    Load processed_chunk_logs and filter entries for target date (IST).
    
    Args:
        target_date: Date string in YYYY-MM-DD format (IST)
    
    Returns:
        List of IngestionFileEntry objects
    """
    if not PROCESSED_CHUNK_LOGS_PATH.exists():
        logger.warning(f"Processed chunk logs not found: {PROCESSED_CHUNK_LOGS_PATH}")
        return []
    
    entries: List[IngestionFileEntry] = []
    
    try:
        with PROCESSED_CHUNK_LOGS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    
                    # Parse timestamp and convert to IST
                    timestamp_str = entry.get("ts", "")  # Changed from "timestamp" to "ts"
                    if timestamp_str:
                        try:
                            # Parse ISO format timestamp
                            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                            entry_date = get_ist_date_string(ts)
                            
                            # Filter by target date
                            if entry_date == target_date:
                                entries.append(IngestionFileEntry(
                                    filename=entry.get("file_name", "unknown"),  # Changed from "pdf_name" to "file_name"
                                    status=entry.get("status", "UNKNOWN"),
                                    timestamp=timestamp_str,
                                    error=entry.get("error")
                                ))
                        except Exception as e:
                            logger.warning(f"Failed to parse timestamp: {timestamp_str}, error: {e}")
                            continue
                            
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(entries)} ingestion log entries for date {target_date}")
        
    except Exception as e:
        logger.error(f"Failed to load ingestion logs: {e}")
    
    return entries


def format_ingestion_report(date_str: str, entries: List[IngestionFileEntry]) -> str:
    """
    Format ingestion report as HTML email body.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        entries: List of file entries
    
    Returns:
        Formatted HTML email body
    """
    successful = [e for e in entries if e.status == "SUCCESS"]
    failed = [e for e in entries if e.status != "SUCCESS"]
    
    # Format date for display
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        display_date = dt.strftime("%d-%b-%Y")
    except Exception:
        display_date = date_str
    
    # Build table rows
    table_rows = ""
    if entries:
        for entry in entries:
            if entry.status == "SUCCESS":
                status_html = '<span style="color:#28a745;font-weight:bold">✓ SUCCESS</span>'
            else:
                status_html = f'<span style="color:#dc3545;font-weight:bold">✗ {entry.status}</span>'
            
            error_html = entry.error if entry.error else "-"
            
            # Format timestamp
            try:
                ts = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                timestamp_display = ts.strftime("%Y-%m-%d %H:%M:%S")
            except:
                timestamp_display = entry.timestamp
            
            table_rows += f"""
                <tr>
                    <td>{status_html}</td>
                    <td style="word-break:break-all">{entry.filename}</td>
                    <td style="font-size:11px">{timestamp_display}</td>
                    <td style="font-size:11px;color:#666">{error_html}</td>
                </tr>
            """
    else:
        table_rows = '<tr><td colspan="4" style="text-align:center;color:#666;padding:20px">No new files ingested.</td></tr>'
    
    # Build complete HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: #ffffff;
            padding: 20px;
            border-radius: 5px;
            max-width: 800px;
            margin: 0 auto;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            background-color: #13406c;
            color: #ffffff;
            padding: 20px;
            border-radius: 5px 5px 0 0;
            margin: -20px -20px 20px -20px;
        }}
        .header h2 {{
            margin: 0;
            font-size: 24px;
        }}
        .header p {{
            margin: 5px 0 0 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .summary {{
            background-color: #f0f8ff;
            padding: 15px;
            border-left: 4px solid #13406c;
            margin: 20px 0;
            border-radius: 3px;
        }}
        .summary h3 {{
            margin-top: 0;
            color: #13406c;
        }}
        .summary ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .summary li {{
            margin: 5px 0;
        }}
        .success {{
            color: #28a745;
            font-weight: bold;
        }}
        .failed {{
            color: #dc3545;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #13406c;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .footer {{
            color: #666;
            font-size: 12px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Data Ingestion Report</h2>
            <p>{display_date}</p>
        </div>
        
        <div class="summary">
            <h3>Summary</h3>
            <ul>
                <li>Total Files Processed: <strong>{len(entries)}</strong></li>
                <li class="success">Successful: {len(successful)}</li>
                <li class="failed">Failed: {len(failed)}</li>
            </ul>
        </div>
        
        <h3 style="color:#13406c">Details</h3>
        <table>
            <thead>
                <tr>
                    <th style="width:15%">Status</th>
                    <th style="width:35%">Filename</th>
                    <th style="width:20%">Timestamp</th>
                    <th style="width:30%">Error</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated automatically by MoSPI Chatbot Monitoring System</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html


def send_ingestion_report(target_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate and send ingestion report for specified date.
    
    Args:
        target_date: Date string in YYYY-MM-DD format (IST), defaults to today
    
    Returns:
        Dict with report details and status
    """
    if target_date is None:
        target_date = get_ist_date_string(get_ist_now())
    
    logger.info(f"Generating ingestion report for date: {target_date}")
    
    # Load entries
    entries = load_ingestion_logs_for_date(target_date)
    
    # Count stats
    successful = sum(1 for e in entries if e.status == "SUCCESS")
    failed = len(entries) - successful
    
    # Format email
    email_body = format_ingestion_report(target_date, entries)
    email_subject = f"MoSPI Website Chatbot Data Ingestion Report - {format_date_for_email(target_date)}"
    
    # Send email
    email_sent = False
    email_error = None
    
    if entries or SEND_EMPTY_REPORTS:
        email_sent, email_error = send_email(email_subject, email_body)
    else:
        logger.info("No entries to report and SEND_EMPTY_REPORTS=0, skipping email")
    
    return {
        "status": "success",
        "date": target_date,
        "total_files": len(entries),
        "successful": successful,
        "failed": failed,
        "email_sent": email_sent,
        "email_error": email_error,
        "files": [
            {
                "filename": e.filename,
                "status": e.status,
                "timestamp": e.timestamp,
                "error": e.error
            }
            for e in entries
        ]
    }



# ====================================================================================
# HEALTH CHECK FUNCTIONS DISABLED
# Functions disabled: should_send_health_check_email, mark_health_check_email_sent,
#                     send_health_check_failure_email
# ====================================================================================

