from fastapi import FastAPI, HTTPException, Depends, Security, File, UploadFile, APIRouter, Query, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from datetime import datetime, time as dt_time
import uuid
import shutil
import os
import sys
import tempfile
from urllib.parse import urlparse
from starlette.status import HTTP_403_FORBIDDEN
from dotenv import load_dotenv
import os
from fastapi.responses import PlainTextResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi import APIRouter, HTTPException
import re
from fastapi import Body
from datetime import datetime, timezone, timedelta
from chatbot_qdrant import info_vectordb
from typing import List, Optional
import logging
import asyncio

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

# Global lock to prevent concurrent chunking processes
chunking_lock = asyncio.Lock()
chunking_in_progress = False

def ist_now() -> datetime:
    """Get current datetime in IST timezone"""
    return datetime.now(IST)

def parse_iso_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO date string, handling 'Z' timezone suffix and URL-encoded '+' signs. Returns IST-aware datetime."""
    if not date_str:
        return None
    
    # Handle URL encoding: '+' signs in query parameters become spaces
    # Replace space before timezone offset with '+' (e.g., " 05:30" -> "+05:30")
    # Pattern: space followed by digits (timezone offset)
    import re
    # Replace space before timezone pattern (HH:MM or HHMM)
    normalized = re.sub(r' (\d{2}):?(\d{2})$', r'+\1:\2', date_str)
    
    # Replace 'Z' with '+00:00' for Python 3.10 compatibility
    normalized = normalized.replace('Z', '+00:00')
    
    # If no timezone info, try parsing as-is first
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        # If parsing fails, try adding IST timezone if no timezone present
        if '+' not in normalized and 'Z' not in normalized and normalized.count(':') >= 2:
            # Has time but no timezone - assume IST
            normalized = normalized + '+05:30'
            dt = datetime.fromisoformat(normalized)
        else:
            raise
    
    # If timezone-naive, assume it's IST
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    # If timezone-aware, convert to IST
    elif dt.tzinfo != IST:
        dt = dt.astimezone(IST)
    return dt

def normalize_to_ist(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime to IST-aware. Handles both naive and aware datetimes."""
    if dt is None:
        return None
    # If timezone-naive, assume it's IST (for backward compatibility with old records)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    # If timezone-aware, convert to IST
    return dt.astimezone(IST)
import json
import hashlib
import subprocess
import asyncio
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

# Load environment variables
load_dotenv()
API_KEY = (os.getenv("API_KEY") or "").strip()
API_KEY_NAME = "access_token"
print(f"[Debug API KEY NAME] api_key_header: {API_KEY_NAME}")
logger = logging.getLogger("chatbot")

# Create separate loggers for scraper and upload
scraper_logger = logging.getLogger("web_scraper")
upload_logger = logging.getLogger("manual_upload")

# Configure scraper logger
scraper_handler = logging.FileHandler("logs/web_scrape.log", mode="a")
scraper_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
scraper_logger.addHandler(scraper_handler)
scraper_logger.addHandler(logging.StreamHandler(sys.stdout))  # Also log to console
scraper_logger.setLevel(logging.INFO)

# Configure upload logger
upload_handler = logging.FileHandler("logs/manual_upload.log", mode="a")
upload_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
upload_logger.addHandler(upload_handler)
upload_logger.addHandler(logging.StreamHandler(sys.stdout))  # Also log to console
upload_logger.setLevel(logging.INFO)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# API key security (header only). Same scheme as /start_session (APIKeyHeader in /docs).
api_key_header = APIKeyHeader(
    name=API_KEY_NAME,
    auto_error=False,
    scheme_name="APIKeyHeader",
    description="API key (access_token header).",
)
print(f"[Debug API KEY HEADER] api_key_header: {api_key_header}")

async def get_api_key(access_token: str = Security(api_key_header)):
    if not access_token:
        logger.warning("API key validation failed: access_token header missing")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Missing access_token header",
        )
    if access_token == API_KEY:
        return API_KEY
    else:
        logger.warning("API key validation failed: access_token header invalid (wrong or expired key)")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Could not validate API key",
        )


from auth import auth_router, get_current_user, require_roles, security
from jose import JWTError, jwt
from fastapi.security import HTTPAuthorizationCredentials

JWT_SECRET = (os.getenv("JWT_SECRET") or "").strip()
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


async def verify_public_or_admin_jwt(
    request: Request,
    access_token: str = Security(api_key_header),
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    """Public chatbot API key OR admin JWT (for admin panel chat test page)."""
    if access_token and access_token == API_KEY:
        set_audit_actor(request, actor_type="api_key")
        return {"type": "api_key"}

    if credentials and credentials.scheme.lower() == "bearer" and JWT_SECRET:
        try:
            payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            user_id = payload.get("sub")
            if user_id:
                from models import AdminUser
                user = await AdminUser.get(user_id)
                if user and user.is_active:
                    set_audit_admin_user(request, user)
                    return {"type": "jwt", "user": user}
        except JWTError:
            pass

    set_audit_actor(request, actor_type="anonymous")
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )


def _require_admin_for_ops(request: Request) -> None:
    """After public_router auth: allow API key; JWT must be admin."""
    from models import user_has_any_role

    actor = getattr(request.state, "audit_actor", None) or {}
    if actor.get("actor_type") == "api_key":
        return
    roles = actor.get("actor_roles") or []
    if "admin" in roles:
        return
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Admin role or access_token API key required",
    )


# Public chatbot routes (API key or admin JWT) — same OpenAPI Authorize as /start_session.
# Admin routes (JWT + RBAC).
public_router = APIRouter(dependencies=[Depends(verify_public_or_admin_jwt)])
admin_router = APIRouter(dependencies=[Depends(get_current_user)])





# Import your existing modules
from init_db import initialize_database
from models import Interaction, MonthlyReport, find_interactions_safe
from evaluation import resume_evaluation_jobs
from evaluation import router as evaluation_router
from session_store import init_redis_session_store
from analytics import TimingMiddleware, AnalyticsService
from audit import AuditMiddleware, log_system_audit, set_audit_actor, set_audit_admin_user, set_audit_metadata
from chatbot_qdrant import (
    initialize_components,
    handle_question,
    get_chunks_for_doc,
    memory_sessions,
    create_memory,
    list_all_urls,
    get_chunks_for_url,
    cleanup_expired_sessions,
    list_all_documents,
    delete_chunks_for_doc,
    update_doc_metadata
)


# Define models
class QueryRequest(BaseModel):
    query: str
    session_id: str

class SessionResponse(BaseModel):
    session_id: str

# Create FastAPI app
app = FastAPI(title="MOSPI AI")
app.add_middleware(TimingMiddleware)
app.add_middleware(AuditMiddleware)

# Global exception handler to prevent stack trace leakage
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler that returns generic error messages to clients
    while logging full details server-side.
    """
    # Log full exception server-side
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    # Return generic error to client (no stack trace, no internal paths)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please contact support if the issue persists."
        }
    )

# Initialize scheduler
scheduler = AsyncIOScheduler(timezone=pytz.timezone('Asia/Kolkata'))
scraper_job = None  # Will hold the scheduled job reference


import os
from fastapi.staticfiles import StaticFiles
from config import STATIC_DIRECTORY, STATIC_ROUTE, STATIC_NAME

# Ensure static folder exists
if not os.path.exists(STATIC_DIRECTORY):
    os.makedirs(STATIC_DIRECTORY)

# Mount static files
app.mount(STATIC_ROUTE, StaticFiles(directory=STATIC_DIRECTORY), name=STATIC_NAME)

# Legacy alias kept for any remaining references during migration
router = public_router


def get_referrer_domain(request: Request) -> Optional[str]:
    """Referrer domain e.g. 'www.mospi.gov.in' from Referer header."""
    referer = request.headers.get("Referer") or request.headers.get("Referrer")
    if not referer:
        return None
    try:
        parsed = urlparse(referer)
        return parsed.netloc or None  # hostname without scheme
    except Exception:
        return None


# ============================================
# Background Pipeline Functions
# ============================================

async def run_pdf_conversion(output_dir: str = "./md_files"):
    """
    Background task to convert PDFs to text.
    Automatically uses S3 mode (default behavior) to download PDFs and upload results.
    
    Args:
        output_dir: Directory for output markdown files
    """
    try:
        print(f"[PIPELINE] Starting PDF conversion at {datetime.now()}")
        print(f"[PIPELINE] Output directory: {output_dir}")
        print(f"[PIPELINE] Mode: S3 (default - downloads from S3, uploads results)")
        print(f"[PIPELINE] Working directory: {os.getcwd()}")
        
        pdf_script = "batch_process_pdf_to_text.py"
        python_exe = sys.executable
        
        # Get vLLM API URL from environment
        api_url = os.getenv("PDF_CONVERTER_VLLM_URL", "http://localhost:8002/v1")
        print(f"[PIPELINE] vLLM API URL: {api_url}")
        
        # Build command with arguments (S3 mode is default, no flag needed)
        cmd_args = [
            python_exe,
            pdf_script,
            "--output-dir", output_dir,
            "--api-url", api_url
        ]
        
        print(f"[PIPELINE] Will download PDFs from S3 and upload results")
        
        process = await asyncio.create_subprocess_exec(
            *cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.getcwd(),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        
        # Stream output in real-time
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            try:
                decoded_line = line.decode('utf-8', errors='replace').strip()
                if decoded_line:
                    print(f"[PDF_CONVERT] {decoded_line}")
            except Exception as e:
                print(f"[PDF_CONVERT] Error decoding line: {e}")
        
        await process.wait()
        
        if process.returncode == 0:
            print(f"[PIPELINE] PDF conversion completed successfully at {datetime.now()}")
            # Automatically trigger chunking after PDF conversion
            await run_chunking()
        else:
            print(f"[PIPELINE] PDF conversion failed with return code {process.returncode}")
            
    except Exception as e:
        print(f"[PIPELINE] Error in PDF conversion: {str(e)}")


async def run_chunking():
    """Background task to chunk text and upload to Qdrant"""
    global chunking_in_progress
    
    # Check if chunking is already in progress
    if chunking_in_progress:
        print(f"[PIPELINE] Chunking already in progress, skipping duplicate request at {datetime.now()}")
        return
    
    # Acquire lock to prevent concurrent chunking
    async with chunking_lock:
        chunking_in_progress = True
        try:
            print(f"[PIPELINE] Starting chunking at {datetime.now()}")
            
            chunking_script = "chunking_local.py"
            python_exe = sys.executable
            
            process = await asyncio.create_subprocess_exec(
                python_exe,
                chunking_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=os.getcwd(),
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}
            )
            
            # Read output asynchronously to prevent pipe buffer from filling up
            async def read_output():
                """Continuously read from stdout to prevent pipe buffer blocking"""
                try:
                    if process.stdout:
                        while True:
                            line = await process.stdout.readline()
                            if not line:
                                break
                            # Decode and consume output (prevents buffer blocking)
                            try:
                                decoded = line.decode('utf-8', errors='ignore').strip()
                                # Optionally log important messages
                                # if decoded:
                                #     print(f"[CHUNKING] {decoded}")
                            except Exception:
                                pass  # Ignore decode errors
                except Exception as e:
                    print(f"[PIPELINE] Error reading chunking output: {e}")
            
            async def wait_for_process():
                """Wait for process to complete"""
                return await process.wait()
            
            # Run both tasks concurrently using gather (ensures true parallelism)
            try:
                # Set a reasonable timeout (30 minutes for chunking)
                await asyncio.wait_for(
                    asyncio.gather(
                        read_output(),
                        wait_for_process()
                    ),
                    timeout=1800  # 30 minutes
                )
            except asyncio.TimeoutError:
                print(f"[PIPELINE] Chunking timed out after 30 minutes, killing process")
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass
                raise Exception("Chunking process timed out")
            
            if process.returncode == 0:
                print(f"[PIPELINE] Chunking completed successfully at {datetime.now()}")
                print(f"[PIPELINE] Full data ingestion pipeline complete!")
                
                # Send ingestion report email
                try:
                    print(f"[PIPELINE] Sending ingestion report email...")
                    from monitoring import send_ingestion_report
                    report_result = send_ingestion_report()
                    if report_result.get("email_sent"):
                        print(f"[PIPELINE] ✓ Ingestion report email sent successfully")
                        print(f"[PIPELINE] Report: {report_result['successful']} successful, {report_result['failed']} failed")
                    else:
                        print(f"[PIPELINE] ✗ Failed to send ingestion report email: {report_result.get('email_error')}")
                except Exception as email_error:
                    print(f"[PIPELINE] Error sending ingestion report: {email_error}")
            else:
                print(f"[PIPELINE] Chunking failed with return code {process.returncode}")
                
        except Exception as e:
            print(f"[PIPELINE] Error in chunking: {str(e)}")
        finally:
            # Always release the lock
            chunking_in_progress = False


async def run_scheduled_scraper():
    """Background task to run web scraper at scheduled time"""
    await log_system_audit(
        "ingestion.scrape.scheduled",
        metadata={"trigger": "scheduler", "timezone": "Asia/Kolkata"},
    )
    try:
        # Capture the start date in IST for monitoring purposes
        from monitoring import get_ist_now, get_ist_date_string
        scraping_start_time = get_ist_now()
        scraping_date = get_ist_date_string(scraping_start_time)
        
        print(f"[SCHEDULED] Starting web scraper at {datetime.now()}")
        print(f"[SCHEDULED] Scraping date (IST): {scraping_date}")
        
        scraper_script = os.path.join("web_scrap", "scrape.py")
        python_exe = sys.executable
        
        process = await asyncio.create_subprocess_exec(
            python_exe,
            scraper_script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.getcwd(),
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "IS_SCHEDULED": "true"}
        )
        
        # Read and log output in real-time
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            line_str = line.decode('utf-8', errors='replace').strip()
            if line_str:
                print(f"[SCHEDULED_SCRAPER] {line_str}")
        
        await process.wait()
        
        if process.returncode == 0:
            print(f"[SCHEDULED] Web scraper completed successfully at {datetime.now()}")
            
            # Check if any new PDFs were downloaded
            results_file = os.path.join("web_scrap", "scheduled_scraping_results.json")
            new_pdfs_found = False
            pdfs_downloaded = 0
            
            try:
                if os.path.exists(results_file):
                    with open(results_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                        # Field is "total_pdfs" at root level, not nested under "summary"
                        pdfs_downloaded = results.get("total_pdfs", 0)
                        new_pdfs_found = pdfs_downloaded > 0
                        print(f"[SCHEDULED] Scraper results: {pdfs_downloaded} PDFs downloaded")
                else:
                    print(f"[SCHEDULED] Warning: Results file not found at {results_file}")
            except Exception as e:
                print(f"[SCHEDULED] Error reading scraper results: {e}")
            
            if new_pdfs_found:
                # Continue with normal pipeline: PDF conversion → chunking → email
                print(f"[SCHEDULED] New PDFs found, continuing pipeline...")
                await run_pdf_conversion(output_dir="./md_files")
            else:
                # No new PDFs - send monitoring email and stop pipeline
                print(f"[SCHEDULED] No new PDFs found, sending monitoring email and stopping pipeline")
                try:
                    import httpx
                    app_host = os.getenv("host", "localhost").strip('"')
                    base_url = f"http://{app_host}:8096"
                    headers = {"access_token": API_KEY}
                    
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        # Pass the scraping date as query parameter to ensure we check the correct date
                        response = await client.post(
                            f"{base_url}/monitor/ingested_files?date={scraping_date}",
                            headers=headers
                        )
                        
                        if response.status_code == 200:
                            print(f"[SCHEDULED] ✓ Monitoring email sent successfully")
                        else:
                            print(f"[SCHEDULED] ✗ Failed to send monitoring email: {response.status_code}")
                            
                except Exception as email_error:
                    print(f"[SCHEDULED] Error sending monitoring email: {email_error}")
        else:
            print(f"[SCHEDULED] Web scraper failed with return code {process.returncode}")
            
    except Exception as e:
        print(f"[SCHEDULED] Error running web scraper: {str(e)}")


async def run_scheduled_api_ingestion():
    """Background task to run API ingestion at scheduled time"""
    try:
        from api_ingestion.pipeline import run_ingestion
        
        logger.info(f"[SCHEDULED_API] Starting API ingestion at {datetime.now()}")
        
        # List of APIs to process (all except Who's Who for now)
        apis_to_process = [
            "latest_releases",
            "announcements",
            "publications_reports", 
            "public_docs",
            "iss",
            "sss",
            "visualizations",
            "chapter_data"
        ]
        
        total_stats = {
            "total_fetched": 0,
            "total_new_files": 0,
            "total_uploaded": 0,
            "total_errors": 0
        }
        
        # Process each API
        for api_name in apis_to_process:
            try:
                logger.info(f"[SCHEDULED_API] Processing {api_name}...")
                
                result = run_ingestion(
                    api_name=api_name,
                    max_results=20,
                    dry_run=False,
                    logger=logger
                )
                
                if result["success"]:
                    stats = result["stats"]
                    total_stats["total_fetched"] += stats["fetched"]
                    total_stats["total_new_files"] += stats["new_files"]
                    total_stats["total_uploaded"] += stats["uploaded"]
                    total_stats["total_errors"] += stats["errors"]
                    
                    logger.info(f"[SCHEDULED_API] {api_name}: {stats['uploaded']} files uploaded")
                else:
                    logger.error(f"[SCHEDULED_API] {api_name} failed: {result.get('error')}")
                    total_stats["total_errors"] += 1
                    
            except Exception as e:
                logger.error(f"[SCHEDULED_API] Error processing {api_name}: {e}")
                total_stats["total_errors"] += 1
        
        logger.info(f"[SCHEDULED_API] Completed at {datetime.now()}")
        logger.info(f"[SCHEDULED_API] Summary: {total_stats['total_uploaded']} files uploaded from {apis_to_process}")
        
        # Trigger PDF processing (same pipeline as the /convert_pdfs API) if new
        # files were uploaded.
        # Set ENABLE_SCHEDULED_PDF_CONVERSION=false in .env to disable this step.
        pdf_conversion_enabled = os.getenv("ENABLE_SCHEDULED_PDF_CONVERSION", "true").lower() in ("1", "true", "yes")
        if total_stats["total_uploaded"] > 0:
            if pdf_conversion_enabled:
                logger.info(f"[SCHEDULED_API] Triggering convert_pdfs pipeline (PDF -> text -> chunking)")
                await run_pdf_conversion(output_dir="./md_files")
            else:
                logger.info(f"[SCHEDULED_API] PDF conversion is DISABLED (ENABLE_SCHEDULED_PDF_CONVERSION=false), skipping")
        else:
            logger.info(f"[SCHEDULED_API] No new files, PDF conversion skipped")

        whois_refreshed = False
        fod_refreshed = False

        try:
            from fetch_whois_api import WhosWhoAPIFetcher
            logger.info(f"[SCHEDULED_API] Refreshing Who's Who directory...")
            whois_result = WhosWhoAPIFetcher().fetch_and_upload()
            if whois_result.get("success"):
                whois_refreshed = True
                logger.info(
                    f"[SCHEDULED_API] Who's Who refreshed - "
                    f"{whois_result.get('officer_count', 0)} officers, file: {whois_result.get('filename', 'N/A')}"
                )
            else:
                logger.error(f"[SCHEDULED_API] Who's Who refresh failed: {whois_result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"[SCHEDULED_API] Error refreshing Who's Who: {e}")

        try:
            from fetch_fod_pdf import FODPDFFetcher
            logger.info(f"[SCHEDULED_API] Refreshing FOD directory...")
            fod_result = FODPDFFetcher().fetch_and_upload()
            if fod_result.get("success"):
                fod_refreshed = True
                logger.info(f"[SCHEDULED_API] FOD refreshed - file: {fod_result.get('filename', 'N/A')}")
            else:
                logger.error(f"[SCHEDULED_API] FOD refresh failed: {fod_result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"[SCHEDULED_API] Error refreshing FOD directory: {e}")

        if whois_refreshed or fod_refreshed:
            try:
                from whois_cache_manager import force_refresh as force_whois_refresh
                logger.info(f"[SCHEDULED_API] Rebuilding Who's Who + FOD cache from S3...")
                # force_refresh is blocking (S3 download + PDF->markdown conversion);
                # run in a thread to avoid blocking the event loop.
                cache_updated = await asyncio.to_thread(force_whois_refresh)
                if cache_updated:
                    logger.info(f"[SCHEDULED_API] Who's Who + FOD cache updated successfully")
                else:
                    logger.info(f"[SCHEDULED_API] Who's Who + FOD cache unchanged (no new content)")
            except Exception as e:
                logger.error(f"[SCHEDULED_API] Error rebuilding Who's Who + FOD cache: {e}")
        else:
            logger.info(f"[SCHEDULED_API] Skipping cache rebuild (no directory was refreshed)")

    except Exception as e:
        logger.error(f"[SCHEDULED_API] Error running API ingestion: {str(e)}")


@router.get("/start_session", response_model=SessionResponse)
async def create_session(request: Request, device_id: Optional[str] = None):
    """Create a new chat session. Only proxy/chatbot traffic (no device_id) is tracked for analytics; our React app (sends device_id) is not counted."""
    cleanup_expired_sessions()
    session_id = str(uuid.uuid4())
    memory_sessions[session_id] = (ist_now(), create_memory())

    from session_store import set_session_track

    # Track analytics only for proxy (mospi.gov.in chatbot). Our React app sends device_id → do not track.
    if device_id:
        # Request from our React frontend — do not store in user_activity / do not count in user metrics
        set_session_track(session_id, False)
    else:
        # Request from proxy (chatbot on mospi.gov.in) — track; count users by session_id
        set_session_track(session_id, True)
        source = get_referrer_domain(request)  # e.g. www.mospi.gov.in
        await AnalyticsService.track_user_activity(session_id, device_id=None, source=source, from_proxy=True)
    return {"session_id": session_id}

@public_router.post("/ask")
async def ask_question(request: QueryRequest):
    cleanup_expired_sessions()
    return await handle_question(request)





from pydantic import BaseModel
from models import Interaction, MonthlyReport, find_interactions_safe
from typing import Literal

class FeedbackRequest(BaseModel):
    feedback: Literal["like", "dislike"]

@public_router.post("/interactions/{interaction_id}/feedback")
async def set_interaction_feedback(interaction_id: str, request: FeedbackRequest):
    interaction = await Interaction.get(interaction_id)
    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")

    interaction.feedback = request.feedback
    await interaction.save()
    return {
        "message": "Feedback recorded successfully",
        "interaction_id": interaction_id,
        "feedback": request.feedback
    }

@admin_router.get("/feedback_stats", dependencies=[Depends(require_roles("admin", "analyst"))])
async def get_feedback_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    # Get all interactions, optionally filtered by date
    all_interactions = await find_interactions_safe()
    
    if start_date or end_date:
        start = parse_iso_date(start_date)
        end = parse_iso_date(end_date)
        filtered = []
        for i in all_interactions:
            # Normalize timestamp to IST-aware for comparison
            timestamp_ist = normalize_to_ist(i.timestamp)
            if start and timestamp_ist < start:
                continue
            if end and timestamp_ist > end:
                continue
            filtered.append(i)
        all_interactions = filtered
    
    total = len(all_interactions)
    likes = len([i for i in all_interactions if i.feedback == "like"])
    dislikes = len([i for i in all_interactions if i.feedback == "dislike"])
    total_feedback = likes + dislikes
    
    return {
        "total_interactions": total,
        "likes": likes,
        "dislikes": dislikes,
        "like_percentage": round((likes / total_feedback) * 100, 2) if total_feedback > 0 else 0
    }


@admin_router.post("/upload_file", dependencies=[Depends(require_roles("admin", "operator"))])
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(...),
    pdf_url: str = Form(...),
    publish_date: str = Form(...)
):
    """
    Upload a PDF or Excel file manually to NxtGen S3 and log metadata.
    Uses web_scrap/manual_upload.py for consistent S3 handling.
    PDFs trigger PDF conversion and chunking; Excel files use direct Excel ingestion.
    
    Args:
        file: PDF or Excel upload
        title: Document title
        pdf_url: Original document URL
        publish_date: Publication date (YYYY-MM-DD format, required)
    
    Returns:
        Success message with file details
    """
    temp_dir = None
    try:
        filename = os.path.basename(file.filename or "")
        if not filename:
            raise HTTPException(status_code=400, detail="A file with a supported filename is required")

        upload_logger.info(f"[UPLOAD] Starting manual upload for file: {filename}")
        
        # Validate publish_date is provided
        if not publish_date or publish_date.strip() == "":
            upload_logger.error(f"[UPLOAD] No publish_date provided for {filename}")
            raise HTTPException(
                status_code=400, 
                detail="publish_date is required. Please provide a date in YYYY-MM-DD format."
            )
        
        # Validate publish_date format (YYYY-MM-DD)
        try:
            from datetime import datetime
            datetime.strptime(publish_date.strip(), "%Y-%m-%d")
        except ValueError:
            upload_logger.error(f"[UPLOAD] Invalid publish_date format: {publish_date}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid publish_date format: '{publish_date}'. Expected format: YYYY-MM-DD (e.g., 2026-01-12)"
            )
        
        # Validate file type (PDF + Excel formats)
        _fname_lower = filename.lower()
        is_excel_upload = _fname_lower.endswith(('.xlsx', '.xlsm', '.xls', '.xlsb'))
        if not (_fname_lower.endswith('.pdf') or is_excel_upload):
            upload_logger.error(f"[UPLOAD] Invalid file type: {filename}")
            raise HTTPException(status_code=400, detail="Only PDF and Excel (.xlsx/.xlsm/.xls/.xlsb) files are allowed")
        
        # Create temp directory for processing
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, filename)
        
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        upload_logger.info(f"[UPLOAD] File saved temporarily: {temp_file_path}")
        
        # Run the active helper with the same interpreter and project root as the API.
        # This prevents deployments from accidentally using a different Python
        # environment or working directory.
        project_root = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(project_root, "web_scrap", "manual_upload.py")
        
        cmd = [
            sys.executable,
            script_path,
            "--file", temp_file_path,
            "--title", title,
            "--url", pdf_url,
            "--date", publish_date
        ]
        
        upload_logger.info(f"[UPLOAD] Executing upload script: {script_path}")
        
        # Execute upload script and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        
        # Log script output
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    upload_logger.info(f"[SCRIPT] {line.strip()}")
        
        # Check if upload failed (script returns non-zero exit code).
        # The helper writes normal errors to stdout, so inspect both streams.
        if result.returncode != 0:
            error_msg = (result.stderr or result.stdout or "Upload script failed").strip()
            
            # Check if it's a duplicate error
            if "DUPLICATE DETECTED" in result.stdout or "already exists" in result.stdout:
                # Extract the duplicate reason from output
                for line in result.stdout.split('\n'):
                    if "DUPLICATE DETECTED" in line or "already exists" in line:
                        error_msg = line.replace("[UPLOAD] DUPLICATE DETECTED: ", "").strip()
                        break
                
                upload_logger.warning(f"[UPLOAD] Duplicate file rejected: {error_msg}")
                raise HTTPException(
                    status_code=409,  # Conflict status code for duplicates
                    detail=error_msg
                )
            else:
                upload_logger.error(f"[UPLOAD] Upload script failed: {error_msg}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Upload failed: {error_msg}"
                )
        
        upload_logger.info(f"[UPLOAD] Upload completed successfully for {filename}")

        if is_excel_upload:
            # Excel: ingest directly (extract -> chunk -> embed -> upsert).
            upload_logger.info(f"[UPLOAD] Triggering Excel ingestion for {filename}")

            async def _run_excel_ingest(_path: str, _tmp: str, _title: str, _url: str, _date: str, _fname: str):
                try:
                    from excel_ingestion.excel_processor import process_excel_file
                    meta = {"file_name": _fname, "title": _title, "file_url": _url, "publish_date": _date}
                    res = await asyncio.to_thread(process_excel_file, _path, meta, False, False)
                    upload_logger.info(f"[UPLOAD] Excel ingestion result for {_fname}: {res}")
                except Exception as e:
                    upload_logger.error(f"[UPLOAD] Excel ingestion failed for {_fname}: {e}", exc_info=True)
                finally:
                    try:
                        if _tmp and os.path.exists(_tmp):
                            shutil.rmtree(_tmp)
                    except Exception:
                        pass

            asyncio.create_task(_run_excel_ingest(temp_file_path, temp_dir, title, pdf_url, publish_date, filename))
            # Prevent the outer error handler from removing temp_dir before the task runs
            temp_dir = None
        else:
            # PDF: cleanup temp file and trigger the PDF conversion + chunking pipeline
            shutil.rmtree(temp_dir)
            temp_dir = None
            upload_logger.info(f"[UPLOAD] Triggering automated pipeline for {filename}")
            asyncio.create_task(run_pdf_conversion(
                output_dir="./md_files"
            ))
        
        # Read the last entry from manual_uploads.jsonl to get metadata
        log_file = os.path.join("web_scrap", "manual_uploads.jsonl")
        metadata_entry = None
        
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    metadata_entry = json.loads(lines[-1])
        
        set_audit_metadata(
            request,
            {
                "file_name": file.filename,
                "title": title,
                "pdf_url": pdf_url,
                "publish_date": publish_date,
            },
        )

        return {
            "message": f"File {file.filename} uploaded successfully",
            "pipeline_status": (
                "Excel ingestion (extract, chunk, embed, upsert) started in background"
                if is_excel_upload else
                "PDF conversion and chunking pipeline started in background"
            ),
            "metadata": metadata_entry or {
                "file_name": file.filename,
                "title": title,
                "pdf_url": pdf_url,
                "publish_date": publish_date
            }
        }
        
    except HTTPException:
        # Preserve intentional 4xx responses from validation and duplicate checks.
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise
    except subprocess.CalledProcessError as e:
        # Cleanup on error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        error_msg = e.stderr if e.stderr else str(e)
        upload_logger.error(f"[UPLOAD] Upload script failed: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {error_msg}"
        )
    except Exception as e:
        # Cleanup on error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        upload_logger.error(f"[UPLOAD] Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")



@admin_router.post("/start_web_scrape", dependencies=[Depends(require_roles("admin", "operator"))])
async def start_web_scrape():
    """
    Start web scraping process and return streaming logs.
    After completion, automatically triggers /convert_pdfs.
    Logs are also written to main logger (console + logs/chatbot.log)
    
    Returns:
        Streaming response with scraper logs
    """
    try:
        # Path to scraper script
        scraper_script = os.path.join("web_scrap", "scrape.py")
        
        if not os.path.exists(scraper_script):
            raise HTTPException(
                status_code=404,
                detail="Scraper script not found"
            )
        
        async def log_generator():
            """Generator that yields scraper logs in real-time"""
            try:
                # Get Python executable from virtual environment
                python_exe = sys.executable
                
                # Start scraper process with UTF-8 encoding and IS_SCHEDULED flag
                process = await asyncio.create_subprocess_exec(
                    python_exe,
                    scraper_script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=os.getcwd(),
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "IS_SCHEDULED": "false"}
                )
                
                start_msg = f"[START] Web scraping started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                scraper_logger.info(start_msg.strip())
                yield start_msg
                
                # Stream output line by line
                in_traceback = False
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    try:
                        decoded_line = line.decode('utf-8', errors='replace')
                        
                        # Detect start of Python traceback
                        if decoded_line.strip().startswith("Traceback (most recent call last):"):
                            in_traceback = True
                            # Log full traceback server-side
                            scraper_logger.error("Python traceback detected in scraper output")
                            scraper_logger.error(decoded_line.strip())
                            # Send generic error to client
                            yield "[ERROR] Script execution error detected\n"
                            continue
                        
                        # Skip traceback lines (file paths, line numbers, error details)
                        if in_traceback:
                            scraper_logger.error(decoded_line.strip())  # Log server-side
                            # Stop capturing traceback after the actual error message
                            if decoded_line.strip() and not decoded_line.startswith("  ") and not decoded_line.startswith("File "):
                                in_traceback = False
                                # Send generic error for the error type
                                yield "[ERROR] Check server logs for technical details\n"
                            continue
                        
                        # Log to scraper logger (console + web_scrape.log)
                        scraper_logger.info(decoded_line.strip())
                        # Yield safe output to client (already filtered above)
                        yield decoded_line
                    except Exception as e:
                        scraper_logger.error(f"Error processing line: {str(e)}", exc_info=True)
                        yield "[ERROR] Error processing output\n"
                
                # Wait for process to complete
                await process.wait()
                
                if process.returncode == 0:
                    success_msg = f"\n[SUCCESS] Web scraping completed successfully\n"
                    scraper_logger.info(success_msg.strip())
                    yield success_msg
                    
                    trigger_msg = f"[TRIGGER] Starting automated pipeline with S3 mode...\n"
                    scraper_logger.info(trigger_msg.strip())
                    yield trigger_msg
                    
                    # Automatically trigger PDF conversion pipeline (S3 mode is default)
                    asyncio.create_task(run_pdf_conversion(
                        output_dir="./md_files"
                    ))
                    
                    pipeline_msg = f"[PIPELINE] PDF conversion (S3 mode) → chunking pipeline started in background\n"
                    scraper_logger.info(pipeline_msg.strip())
                    yield pipeline_msg
                    
                    info_msg = f"[INFO] Check application logs for pipeline progress\n"
                    scraper_logger.info(info_msg.strip())
                    yield info_msg
                else:
                    error_msg = f"\n[ERROR] Web scraping failed with return code {process.returncode}\n"
                    scraper_logger.error(error_msg.strip())
                    yield error_msg
                    
                    info_msg = f"[INFO] Check logs above for error details\n"
                    scraper_logger.info(info_msg.strip())
                    yield info_msg
                    
            except Exception as e:
                # Log full error details server-side only
                scraper_logger.error(f"Scraper error: {str(e)}", exc_info=True)
                # Return generic error to client
                error_msg = "\n[ERROR] An error occurred during web scraping. Please check logs for details.\n"
                yield error_msg
        
        return StreamingResponse(
            log_generator(),
            media_type="text/plain"
        )
        
    except Exception as e:
        # Log full error details server-side
        logger.error(f"Failed to start web scraping: {str(e)}", exc_info=True)
        # Return generic error to client
        raise HTTPException(
            status_code=500,
            detail="Failed to start web scraping. Please contact support if the issue persists."
        )


@public_router.post("/convert_pdfs")
async def convert_pdfs(request: Request):
    """
    Convert PDFs to text format using batch_process_pdf_to_text.py.
    Automatically downloads PDFs from S3, processes them, and uploads results back to S3.
    Runs in background and returns immediately.
    After completion, automatically triggers /chunker.

    Auth (same as /start_session): Authorize → APIKeyHeader (access_token) OR HTTPBearer.

    Returns:
        Status message indicating process has started
        
    Example:
        curl -X POST "http://localhost:8000/convert_pdfs" \\
             -H "access_token: MoSPI-xxx"
    """
    _require_admin_for_ops(request)
    try:
        pdf_script = "batch_process_pdf_to_text.py"
        
        if not os.path.exists(pdf_script):
            raise HTTPException(
                status_code=404,
                detail=f"PDF processing script not found at {pdf_script}"
            )
        
        # Trigger background task (S3 mode is default, output to ./md_files)
        asyncio.create_task(run_pdf_conversion())
        
        return {
            "status": "started",
            "message": "PDF conversion process started in background with S3 mode",
            "mode": "S3 (automatic download/upload)",
            "output_dir": "./md_files",
            "s3_bucket": "acct1004215-mospi",
            "next_step": "Will automatically trigger /chunker after completion",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "note": "Check logs/pdf_conversion.log for progress. PDFs will be downloaded from latest S3 folder."
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start PDF conversion: {str(e)}"
        )

@public_router.post("/chunker")
async def chunker(request: Request):
    """
    Chunk converted text and prepare for embedding using chunking_local.py.
    Runs in background and returns immediately.
    Final step in the data ingestion pipeline.

    Auth (same as /start_session): Authorize → APIKeyHeader (access_token) OR HTTPBearer.
    
    Returns:
        Status message indicating process has started or already in progress
    """
    _require_admin_for_ops(request)
    try:
        chunking_script = "chunking_local.py"
        
        if not os.path.exists(chunking_script):
            raise HTTPException(
                status_code=404,
                detail=f"Chunking script not found at {chunking_script}"
            )
        
        # Check if chunking is already in progress
        if chunking_in_progress:
            return {
                "status": "already_running",
                "message": "Chunking process is already in progress",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "note": "Please wait for the current chunking process to complete"
            }
        
        # Trigger background task
        asyncio.create_task(run_chunking())
        
        return {
            "status": "started",
            "message": "Chunking process started in background",
            "pipeline_complete": "Will complete data ingestion pipeline",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "note": "Check application logs for progress"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start chunking: {str(e)}"
        )
@admin_router.get("/list_documents", dependencies=[Depends(require_roles("admin", "analyst", "operator"))])
async def list_documents():
    return list_all_documents()


@admin_router.get("/scheduler/status", dependencies=[Depends(require_roles("admin", "operator"))])
async def get_scheduler_status():
    """
    Get scheduler status and next run time for web scraper.
    
    Returns:
        Simplified scheduler status with minimal information
    """
    try:
        if scraper_job is None:
            return {
                "status": "not_scheduled",
                "message": "Web scraper job not scheduled"
            }
        
        next_run = scraper_job.next_run_time
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
        
        if next_run:
            # Calculate time until next run
            time_until = next_run - now
            hours_until = int(time_until.total_seconds() // 3600)
            minutes_until = int((time_until.total_seconds() % 3600) // 60)
            
            return {
                "status": "scheduled",
                "next_run": next_run.strftime("%Y-%m-%d %I:%M %p"),
                "time_until_hours": hours_until,
                "time_until_minutes": minutes_until
            }
        else:
            return {
                "status": "scheduled",
                "message": "Scheduled but next run time not available"
            }
            
    except Exception as e:
        # Log full error details server-side
        logger.error(f"Failed to get scheduler status: {str(e)}", exc_info=True)
        # Return generic error to client
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve scheduler status"
        )


@admin_router.get("/scraper/results", dependencies=[Depends(require_roles("admin", "operator"))])
async def get_scraper_results(type: str = "manual"):
    """
    Get scraping results
    
    Args:
        type: 'manual' or 'scheduled'
    
    Returns:
        Scraping results with PDF list and status
    """
    try:
        # Try to fetch from S3 first
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "web_scrap"))
            from s3_helper import get_s3_client, get_s3_config, download_latest_scraping_results_from_s3
            
            s3_client = get_s3_client()
            s3_config = get_s3_config()
            logs_prefix = f"{s3_config['s3_prefix']}/logs"
            
            is_scheduled = (type == "scheduled")
            results = download_latest_scraping_results_from_s3(
                s3_client,
                s3_config['bucket_name'],
                logs_prefix,
                is_scheduled
            )
            
            if results:
                print(f"[API] Fetched scraping results from S3 for type: {type}")
                return results
            else:
                print(f"[API] No results found in S3, falling back to local files")
        except Exception as s3_error:
            print(f"[API] S3 fetch failed: {str(s3_error)}, falling back to local files")
        
        # Fallback to local files
        script_dir = os.path.join(os.path.dirname(__file__), "web_scrap")
        
        if type == "scheduled":
            results_file = os.path.join(script_dir, "scheduled_scraping_results.json")
        else:
            results_file = os.path.join(script_dir, "scraping_results.json")
        
        if not os.path.exists(results_file):
            return {"status": "no_data", "message": "No scraping results available"}
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"[API] Fetched scraping results from local file for type: {type}")
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.get("/scraper/logs/download", dependencies=[Depends(require_roles("admin", "operator"))])
async def download_scraper_logs(type: str = "manual"):
    """Download scraper logs as JSON"""
    try:
        results = await get_scraper_results(type)
        
        # Create downloadable JSON
        from fastapi.responses import Response
        
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        
        return Response(
            content=json_str,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=scraper_logs_{type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.get("/scraper/logs/view", dependencies=[Depends(require_roles("admin", "operator"))])
async def view_scraper_logs(type: str = "manual"):
    """View scraper logs in browser"""
    try:
        results = await get_scraper_results(type)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.get("/get_chunks_by_docname", dependencies=[Depends(require_roles("admin", "operator"))])
async def get_chunks_by_docname(doc_name: str):
    return get_chunks_for_doc(doc_name)

@admin_router.delete("/delete_chunks_by_docname", dependencies=[Depends(require_roles("admin"))])
async def delete_chunks_by_docname(doc_name: str):
    doc_names = [d.strip() for d in doc_name.split(",")]
    return delete_chunks_for_doc(doc_names)

@admin_router.get("/list_url_titles", dependencies=[Depends(require_roles("admin", "analyst"))])
async def list_all_urls_api():
    return list_all_urls()

@admin_router.get("/get_chunks_by_url", dependencies=[Depends(require_roles("admin", "analyst"))])
async def get_chunks_by_url(url: str):
    return get_chunks_for_url(url)


@admin_router.get("/interactions", dependencies=[Depends(require_roles("admin", "analyst", "operator"))])
async def get_all_interactions(
    filter_type: Optional[str] = None,  # "day" or "custom"
    date: Optional[str] = None,  # For "day" filter (YYYY-MM-DD)
    start_date: Optional[str] = None,  # For "custom" filter (YYYY-MM-DD)
    end_date: Optional[str] = None  # For "custom" filter (YYYY-MM-DD)
):

    logs = await find_interactions_safe(sort="-timestamp")
    
    # Filter by date if provided
    if filter_type == "day" and date:
        # Single day filter: date 00:00:00 to date 23:59:59 IST
        try:
            # Parse date-only string (YYYY-MM-DD)
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            start = date_obj.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=IST)
            end = date_obj.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=IST)
            
            filtered = []
            for i in logs:
                timestamp_ist = normalize_to_ist(i.timestamp)
                if start <= timestamp_ist <= end:
                    filtered.append(i)
            logs = filtered
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD (e.g., 2026-07-10)")
    
    elif filter_type == "custom" and (start_date or end_date):
        # Custom date range filter: start_date 00:00:00 to end_date 23:59:59 IST
        try:
            start = None
            end = None
            
            if start_date:
                start_obj = datetime.strptime(start_date, "%Y-%m-%d")
                start = start_obj.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=IST)
            
            if end_date:
                end_obj = datetime.strptime(end_date, "%Y-%m-%d")
                end = end_obj.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=IST)
            
            filtered = []
            for i in logs:
                timestamp_ist = normalize_to_ist(i.timestamp)
                if start and timestamp_ist < start:
                    continue
                if end and timestamp_ist > end:
                    continue
                filtered.append(i)
            logs = filtered
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD (e.g., 2026-07-10)")
    
    elif start_date or end_date:
        # Legacy support: ISO datetime strings (backward compatible)
        start = parse_iso_date(start_date)
        end = parse_iso_date(end_date)
        filtered = []
        for i in logs:
            timestamp_ist = normalize_to_ist(i.timestamp)
            if start and timestamp_ist < start:
                continue
            if end and timestamp_ist > end:
                continue
            filtered.append(i)
        logs = filtered
    
    return [
        {
            "session_id": i.session_id,
            "timestamp": i.timestamp,
            "query": i.query,
            "response": i.response,
            "sources": i.sources or [],
            "feedback": i.feedback
        }
        for i in logs
    ]

@admin_router.get("/interactions/download", dependencies=[Depends(require_roles("admin", "analyst", "operator"))])
async def download_interactions_excel(
    filter_type: Optional[str] = None,
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):

    import io
    try:
        import pandas as pd
        from openpyxl import Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Required libraries (pandas, openpyxl) not installed. Please install: pip install pandas openpyxl"
        )
    
    # Fetch logs using same filtering logic as /interactions
    logs = await find_interactions_safe(sort="-timestamp")
    
    # Apply date filters (same logic as /interactions)
    if filter_type == "day" and date:
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            start = date_obj.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=IST)
            end = date_obj.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=IST)
            
            filtered = []
            for i in logs:
                timestamp_ist = normalize_to_ist(i.timestamp)
                if start <= timestamp_ist <= end:
                    filtered.append(i)
            logs = filtered
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    elif filter_type == "custom" and (start_date or end_date):
        try:
            start = None
            end = None
            
            if start_date:
                start_obj = datetime.strptime(start_date, "%Y-%m-%d")
                start = start_obj.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=IST)
            
            if end_date:
                end_obj = datetime.strptime(end_date, "%Y-%m-%d")
                end = end_obj.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=IST)
            
            filtered = []
            for i in logs:
                timestamp_ist = normalize_to_ist(i.timestamp)
                if start and timestamp_ist < start:
                    continue
                if end and timestamp_ist > end:
                    continue
                filtered.append(i)
            logs = filtered
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    elif start_date or end_date:
        # Legacy support
        start = parse_iso_date(start_date)
        end = parse_iso_date(end_date)
        filtered = []
        for i in logs:
            timestamp_ist = normalize_to_ist(i.timestamp)
            if start and timestamp_ist < start:
                continue
            if end and timestamp_ist > end:
                continue
            filtered.append(i)
        logs = filtered
    
    # Prepare data for Excel with specified columns
    data_rows = []
    for log in logs:
        # Extract sources (up to 10)
        sources = log.sources or []
        sources_dict = {f"sources[{i}]": sources[i] if i < len(sources) else "" for i in range(10)}
        
        # Extract component timings if available
        component_timings = getattr(log, 'component_timings', None) or {}
        
        row = {
            "_id": str(log.id),
            "session_id": log.session_id,
            "timestamp": log.timestamp.isoformat() if log.timestamp else "",
            "query": log.query or "",
            "response": log.response or "",
            **sources_dict,
            "feedback": log.feedback or "",
            "response_time_ms": getattr(log, 'response_time_ms', ""),
            "component_timings.total_ms": component_timings.get('total_ms', ""),
            "component_timings.query_rewrite_ms": component_timings.get('query_rewrite_ms', ""),
            "component_timings.retrieval_ms": component_timings.get('retrieval_ms', ""),
            "component_timings.llm_ms": component_timings.get('llm_ms', ""),
        }
        data_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Ensure column order
    column_order = [
        "_id", "session_id", "timestamp", "query", "response",
        "sources[0]", "sources[1]", "sources[2]", "sources[3]", "sources[4]",
        "sources[5]", "sources[6]", "sources[7]", "sources[8]", "sources[9]",
        "feedback", "response_time_ms",
        "component_timings.total_ms", "component_timings.query_rewrite_ms",
        "component_timings.retrieval_ms", "component_timings.llm_ms"
    ]
    df = df[column_order]
    
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Interactions')
    output.seek(0)
    
    # Generate filename with date info
    if filter_type == "day" and date:
        filename = f"interactions_{date}.xlsx"
    elif filter_type == "custom" and start_date and end_date:
        filename = f"interactions_{start_date}_to_{end_date}.xlsx"
    elif filter_type == "custom" and start_date:
        filename = f"interactions_from_{start_date}.xlsx"
    elif filter_type == "custom" and end_date:
        filename = f"interactions_until_{end_date}.xlsx"
    else:
        filename = f"interactions_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    # Return as downloadable file
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@admin_router.get("/session_stats", dependencies=[Depends(require_roles("admin", "analyst", "operator"))])
async def get_session_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    fallback_messages = [
        "This seems to be outside my scope. Unfortunately, I am unable to assist you with your requested query. Thank you for your understanding.",
        "यह मेरे दायरे से बाहर लगता है। दुर्भाग्य से, मैं आपके अनुरोधित प्रश्न में सहायता नहीं कर सकता। धन्यवाद।"
    ]
    interactions = await find_interactions_safe()
    
    # Filter by date if provided
    if start_date or end_date:
        start = parse_iso_date(start_date)
        end = parse_iso_date(end_date)
        filtered = []
        for i in interactions:
            # Normalize timestamp to IST-aware for comparison
            timestamp_ist = normalize_to_ist(i.timestamp)
            if start and timestamp_ist < start:
                continue
            if end and timestamp_ist > end:
                continue
            filtered.append(i)
        interactions = filtered
    sessions = {}

    for i in interactions:
        sid = i.session_id
        # Normalize timestamp to IST-aware for consistent comparison
        timestamp_ist = normalize_to_ist(i.timestamp)
        sessions.setdefault(sid, {
            "session_id": sid,
            "chat_count": 0,
            "queries": [],
            "responses": [],
            "start_time": timestamp_ist,
            "end_time": timestamp_ist,
            "fallback_count": 0,
            "likes_count": 0,
            "dislikes_count": 0
        })

        session = sessions[sid]
        session["chat_count"] += 1
        session["queries"].append(i.query)
        session["responses"].append(i.response)
        # Normalize existing timestamps for comparison
        start_time_ist = normalize_to_ist(session["start_time"])
        end_time_ist = normalize_to_ist(session["end_time"])
        session["start_time"] = min(start_time_ist, timestamp_ist)
        session["end_time"] = max(end_time_ist, timestamp_ist)
        if i.response.strip() in fallback_messages:
            session["fallback_count"] += 1
        if i.feedback == "like":
            session["likes_count"] += 1
        elif i.feedback == "dislike":
            session["dislikes_count"] += 1

    results = []
    for session in sessions.values():
        duration = (session["end_time"] - session["start_time"]).total_seconds() / 60.0
        avg_query_length = sum(len(q) for q in session["queries"]) / len(session["queries"])
        avg_response_length = sum(len(r) for r in session["responses"]) / len(session["responses"])
        results.append({
            "session_id": session["session_id"],
            "chat_count": session["chat_count"],
            "start_time": session["start_time"],
            "end_time": session["end_time"],
            "duration_minutes": round(duration, 2),
            "avg_query_length": round(avg_query_length, 2),
            "avg_response_length": round(avg_response_length, 2),
            "fallback_count": session["fallback_count"],
            "likes_count": session.get("likes_count", 0), 
            "dislikes_count": session.get("dislikes_count", 0),
        })

    return results

@admin_router.delete("/delete_session/{session_id}", dependencies=[Depends(require_roles("admin"))])
async def delete_session(request: Request, session_id: str):
    if session_id in memory_sessions:
        del memory_sessions[session_id]
    deleted = await Interaction.find(Interaction.session_id == session_id).delete()
    set_audit_metadata(
        request,
        {
            "resource_type": "session",
            "resource_id": session_id,
            "interactions_deleted": deleted.deleted_count if hasattr(deleted, "deleted_count") else deleted,
        },
    )
    return {
        "message": f"Session {session_id} deleted successfully.",
        "interactions_deleted": deleted
    }




CHATBOT_LOG_FILE = "logs/chatbot.log"

@admin_router.get("/chatbot_logs", dependencies=[Depends(require_roles("admin"))])
async def get_chatbot_logs(lines: int = 200):
    """
    Return the last N lines of chatbot logs.
    Default = 200 lines.
    """
    if not os.path.exists(CHATBOT_LOG_FILE):
        raise HTTPException(status_code=404, detail="Chatbot log file not found")

    try:
        with open(CHATBOT_LOG_FILE, "r") as f:
            content = f.readlines()

        # tail N lines
        last_lines = content[-lines:] if lines > 0 else content
        return PlainTextResponse("".join(last_lines))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading chatbot log file: {str(e)}")


@admin_router.get("/download_chatbot_logs", dependencies=[Depends(require_roles("admin"))])
async def download_chatbot_logs():
    """
    Download the full chatbot log file.
    """
    if not os.path.exists(CHATBOT_LOG_FILE):
        raise HTTPException(status_code=404, detail="Chatbot log file not found")

    return PlainTextResponse(
        open(CHATBOT_LOG_FILE, "r").read(),
        headers={
            "Content-Disposition": "attachment; filename=chatbot.log"
        },
        media_type="text/plain"
    )



class MetadataUpdateByDocRequest(BaseModel):
    doc_names: List[str]
    uploaded_at: Optional[str] = None


@admin_router.post("/update_metadata_by_doc", dependencies=[Depends(require_roles("admin"))])
async def update_metadata_by_doc(request: MetadataUpdateByDocRequest):
    """
    API endpoint → calls chatbot.py function to update metadata
    """
    try:
        return update_doc_metadata(
            doc_names=request.doc_names,
            uploaded_at=request.uploaded_at
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update metadata: {str(e)}")

from chatbot_qdrant import initialize_components, whoswho_names

@app.on_event("startup")
async def startup_event():
    if not API_KEY:
        raise RuntimeError(
            "API_KEY environment variable is not set. "
            "Set API_KEY in your environment (.env file, system env, or docker-compose)."
        )
    jwt_secret = (os.getenv("JWT_SECRET") or "").strip()
    if not jwt_secret:
        raise RuntimeError(
            "JWT_SECRET environment variable is not set. "
            "Required for admin panel authentication."
        )
    
    init_redis_session_store()
    initialize_components()
    await initialize_database()
    await resume_evaluation_jobs()

    from models import migrate_legacy_admin_user_roles
    await migrate_legacy_admin_user_roles()

    if os.getenv("AUTO_SEED_ADMIN", "").lower() in ("1", "true", "yes"):
        from scripts.seed_admin import seed_admin_if_needed
        await seed_admin_if_needed()
    
    # Start scheduler
    scheduler.start()
    
    # Schedule web scraper to run daily at 12 AM IST
    # Set ENABLE_SCHEDULED_WEB_SCRAPER=false in .env to disable the automated scraper
    global scraper_job
    if os.getenv("ENABLE_SCHEDULED_WEB_SCRAPER", "true").lower() in ("1", "true", "yes"):
        scraper_job = scheduler.add_job(
            run_scheduled_scraper,
            CronTrigger(hour=0, minute=0, timezone=pytz.timezone('Asia/Kolkata')),
            id='daily_web_scraper',
            name='Daily Web Scraper',
            replace_existing=True
        )
        print(f"✅ Scheduled web scraper job: daily at 12:00 AM IST")
    else:
        print(f"⚠️ Scheduled web scraper is DISABLED (ENABLE_SCHEDULED_WEB_SCRAPER=false)")
    
    # Schedule API ingestion to run daily at 8 PM IST
    if os.getenv("ENABLE_SCHEDULED_API_INGESTION", "true").lower() in ("1", "true", "yes"):
        api_ingestion_job = scheduler.add_job(
            run_scheduled_api_ingestion,
            CronTrigger(hour=20, minute=0, timezone=pytz.timezone('Asia/Kolkata')),
            id='daily_api_ingestion',
            name='Daily API Ingestion',
            replace_existing=True
        )
        print(f" Scheduled API ingestion job: daily at 8:00 PM IST")
    else:
        print(f"Scheduled API ingestion is DISABLED (ENABLE_SCHEDULED_API_INGESTION=false)")
    
    # Start periodic health check background task - DISABLED

    
    print(f" Application startup complete. Loaded {len(whoswho_names)} officer names.")

# ========== Analytics Endpoints ==========
# Note: Analytics endpoints are added to router BEFORE router is included

@admin_router.get("/analytics/users", dependencies=[Depends(require_roles("admin", "analyst"))])
async def get_user_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get user statistics"""
    start = parse_iso_date(start_date)
    end = parse_iso_date(end_date)
    return await AnalyticsService.get_user_statistics(start, end)


@admin_router.get("/analytics/performance", dependencies=[Depends(require_roles("admin", "analyst"))])
async def get_performance_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get performance metrics (response times, error rates, etc.)"""
    start = parse_iso_date(start_date)
    end = parse_iso_date(end_date)
    return await AnalyticsService.get_performance_metrics(start, end)


@admin_router.get("/analytics/engagement", dependencies=[Depends(require_roles("admin", "analyst"))])
async def get_engagement_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get engagement metrics (queries, sessions, top queries, etc.)"""
    start = parse_iso_date(start_date)
    end = parse_iso_date(end_date)
    return await AnalyticsService.get_engagement_metrics(start, end)


@admin_router.get("/analytics/monthly-report", dependencies=[Depends(require_roles("admin", "analyst"))])
async def get_monthly_report(month: Optional[str] = None):
    """Generate and get monthly analytics report. Without ?month=, generates current month (UTC)."""
    try:
        report = await AnalyticsService.generate_monthly_report(month)
        if not report or not report.get("summary"):
            raise HTTPException(status_code=500, detail="Failed to generate monthly report")
        return report
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate monthly report")


@admin_router.get("/analytics/monthly-reports", dependencies=[Depends(require_roles("admin", "analyst"))])
async def list_monthly_reports():
    """List all available monthly reports"""
    reports = await MonthlyReport.find_all().sort("-report_month").to_list()
    return [
        {
            "report_month": r.report_month,
            "generated_at": r.generated_at,
            "total_users": r.total_users,
            "active_users": r.active_users,
            "total_interactions": r.total_interactions,
            "avg_response_time_ms": r.avg_response_time_ms,
            "error_rate": r.error_rate,
            "fallback_rate": r.fallback_rate
        }
        for r in reports
    ]


@admin_router.get("/analytics/monthly-report/{month}", dependencies=[Depends(require_roles("admin", "analyst"))])
async def get_specific_monthly_report(month: str):
    """Get a specific monthly report by month (format: YYYY-MM)"""
    report = await MonthlyReport.find_one(MonthlyReport.report_month == month)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report for {month} not found")
    return report.report_data


# ============================================
# Monitoring APIs
# ============================================

from monitoring import (
    send_ingestion_report,
    # send_health_check_failure_email,
    get_ist_now,
    get_ist_date_string,
    # HEALTH_CHECK_QUERY,
    # HEALTH_CHECK_INTERVAL,
    setup_monitoring_logger
)

monitoring_logger = setup_monitoring_logger()


@admin_router.post("/monitor/ingested_files", dependencies=[Depends(require_roles("admin"))])
async def monitor_ingested_files(date: Optional[str] = None):
    """
    Generate and send ingestion report for specified date.
    
    Args:
        date: Optional date in YYYY-MM-DD format (IST). Defaults to today.
    
    Returns:
        Report details including file list and email status
    """
    try:
        monitoring_logger.info(f"Ingestion report requested for date: {date or 'today'}")
        result = send_ingestion_report(target_date=date)
        return result
    except Exception as e:
        monitoring_logger.error(f"Failed to generate ingestion report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate ingestion report: {str(e)}"
        )



from typing import List as TypingList
import json as json_module

# Global state for API ingestion jobs
api_ingestion_jobs = {}
api_ingestion_lock = asyncio.Lock()

class APIIngestionRequest(BaseModel):
    """Request model for API ingestion"""
    apis: Optional[TypingList[str]] = None  # List of API names, None = all APIs
    max_results_per_api: int = 20  # Top N documents to check per API
    target_date: Optional[str] = None  # Date filter (YYYY-MM-DD), None = today
    date_range_days: Optional[int] = None  # Check documents from last N days
    dry_run: bool = False  # If True, simulate without uploading
    force_refresh_whois: bool = False  # Force refresh Who's Who
    force_refresh_fod: bool = False  # Force refresh FOD directory
    skip_duplicates: bool = True  # Skip duplicate checking (faster but may re-upload)
    
    class Config:
        schema_extra = {
            "example": {
                "apis": [
                    "latest_releases",
                    "announcements",
                    "publications_reports",
                    "public_docs",
                    "iss",
                    "sss",
                    "chapter_data"
                ],
                "max_results_per_api": 20,
                "target_date": "2026-07-19",
                "date_range_days": None,
                "dry_run": False,
                "force_refresh_whois": False,
                "force_refresh_fod": False,
                "skip_duplicates": True
            }
        }

class APIIngestionStatus(BaseModel):
    """Status model for API ingestion job"""
    job_id: str
    status: str  # "running", "completed", "failed"
    started_at: str
    completed_at: Optional[str] = None
    apis_processed: int = 0
    total_apis: int = 0
    current_api: Optional[str] = None
    stats: Optional[dict] = None
    error: Optional[str] = None

@router.post("/api_ingestion/start")
async def start_api_ingestion(request: APIIngestionRequest):

    try:
        from api_ingestion.pipeline import APIIngestionPipeline
        from datetime import datetime, timedelta
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Parse target date (default to today)
        if request.target_date:
            try:
                target_dt = datetime.strptime(request.target_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid date format. Use YYYY-MM-DD (e.g., 2026-07-17)"
                )
        else:
            target_dt = ist_now()  # Today
        
        # Calculate date range
        if request.date_range_days:
            start_date = target_dt - timedelta(days=request.date_range_days)
            date_filter_str = f"{start_date.strftime('%Y-%m-%d')} to {target_dt.strftime('%Y-%m-%d')}"
        else:
            date_filter_str = target_dt.strftime('%Y-%m-%d')
        
        # Default to all document APIs if not specified
        if not request.apis:
            default_apis = [
                "latest_releases",
                "announcements", 
                "publications_reports",
                "public_docs",
                "iss",
                "sss",
                "chapter_data"
            ]
            apis_to_process = default_apis
        else:
            apis_to_process = request.apis
        
        # Add special handlers if requested
        special_tasks = []
        if request.force_refresh_whois:
            special_tasks.append("whois_who")
        if request.force_refresh_fod:
            special_tasks.append("fod_directory")
        
        total_tasks = len(apis_to_process) + len(special_tasks)
        
        # Initialize job status
        api_ingestion_jobs[job_id] = {
            "status": "running",
            "started_at": ist_now().isoformat(),
            "completed_at": None,
            "config": {
                "apis": apis_to_process,
                "special_tasks": special_tasks,
                "max_results_per_api": request.max_results_per_api,
                "target_date": target_dt.strftime('%Y-%m-%d'),
                "date_range_days": request.date_range_days,
                "date_filter": date_filter_str,
                "dry_run": request.dry_run,
                "skip_duplicates": request.skip_duplicates
            },
            "apis_processed": 0,
            "total_apis": total_tasks,
            "current_api": None,
            "stats": {
                "total_fetched": 0,
                "total_duplicates": 0,
                "total_new_files": 0,
                "total_uploaded": 0,
                "total_errors": 0
            },
            "error": None,
            "api_results": []
        }
        
        # Start background task
        asyncio.create_task(run_api_ingestion_job(
            job_id=job_id,
            apis=apis_to_process,
            special_tasks=special_tasks,
            max_results=request.max_results_per_api,
            target_date=target_dt,
            date_range_days=request.date_range_days,
            dry_run=request.dry_run,
            skip_duplicates=request.skip_duplicates
        ))
        
        logger.info(f"[API_INGESTION] Started job {job_id}")
        logger.info(f"[API_INGESTION]   APIs: {len(apis_to_process)} - {apis_to_process}")
        logger.info(f"[API_INGESTION]   Special tasks: {special_tasks}")
        logger.info(f"[API_INGESTION]   Max results: {request.max_results_per_api}")
        logger.info(f"[API_INGESTION]   Date filter: {date_filter_str}")
        logger.info(f"[API_INGESTION]   Dry run: {request.dry_run}")
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": f"API ingestion started for {total_tasks} tasks",
            "config": {
                "apis": apis_to_process,
                "special_tasks": special_tasks,
                "max_results_per_api": request.max_results_per_api,
                "date_filter": date_filter_str,
                "dry_run": request.dry_run
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API_INGESTION] Error starting ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting ingestion: {str(e)}")

@router.get("/api_ingestion/status/{job_id}")
async def get_api_ingestion_status(job_id: str):
    """
    Get status of API ingestion job
    
    Args:
        job_id: Job ID from start_api_ingestion
    
    Returns:
        Current status and stats
    """
    if job_id not in api_ingestion_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = api_ingestion_jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "progress": {
            "apis_processed": job["apis_processed"],
            "total_apis": job["total_apis"],
            "current_api": job["current_api"]
        },
        "stats": job["stats"],
        "api_results": job.get("api_results", []),
        "error": job["error"]
    }

@router.get("/api_ingestion/logs")
async def get_api_ingestion_logs(lines: int = 100):
    """
    Get recent API ingestion logs
    
    Args:
        lines: Number of recent lines to return
    
    Returns:
        Recent log lines
    """
    try:
        log_dir = Path("logs")
        
        # Find most recent API ingestion log file
        log_files = sorted(
            log_dir.glob("api_ingestion_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not log_files:
            return {"logs": [], "message": "No API ingestion logs found"}
        
        latest_log = log_files[0]
        
        # Read last N lines
        with open(latest_log, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "log_file": str(latest_log.name),
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines),
            "logs": [line.strip() for line in recent_lines]
        }
        
    except Exception as e:
        logger.error(f"[API_INGESTION] Error reading logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")

async def run_api_ingestion_job(
    job_id: str,
    apis: TypingList[str],
    special_tasks: TypingList[str],
    max_results: int,
    target_date: datetime,
    date_range_days: Optional[int],
    dry_run: bool,
    skip_duplicates: bool
):
    """
    Background task to run API ingestion for multiple APIs with date filtering
    
    Args:
        job_id: Job ID for tracking
        apis: List of API names to process
        special_tasks: List of special tasks (whois_who, fod_directory)
        max_results: Max results per API
        target_date: Target date for filtering
        date_range_days: Number of days to look back
        dry_run: If True, don't upload to S3
        skip_duplicates: If True, skip duplicate checking
    """
    try:
        from api_ingestion.pipeline import run_ingestion
        
        job = api_ingestion_jobs[job_id]
        
        # Process special tasks first (Who's Who, FOD)
        for task in special_tasks:
            try:
                job["current_api"] = task
                logger.info(f"[API_INGESTION] Job {job_id}: Processing special task: {task}")
                
                if task == "whois_who":
                    # Run Who's Who fetcher
                    from fetch_whois_api import WhosWhoAPIFetcher
                    
                    if dry_run:
                        logger.info(f"[API_INGESTION] [DRY-RUN] Would fetch Who's Who")
                        job["api_results"].append({
                            "api": task,
                            "status": "skipped",
                            "message": "Dry run mode"
                        })
                    else:
                        fetcher = WhosWhoAPIFetcher()
                        result = fetcher.fetch_and_upload()
                        
                        if result.get("success"):
                            job["api_results"].append({
                                "api": task,
                                "status": "success",
                                "stats": {
                                    "officers": result.get("officer_count", 0),
                                    "sections": result.get("sections", 0),
                                    "file": result.get("filename", "N/A")
                                }
                            })
                            logger.info(f"[API_INGESTION] Job {job_id}: Who's Who completed - {result.get('officer_count', 0)} officers")
                        else:
                            error = result.get("error", "Unknown error")
                            job["api_results"].append({
                                "api": task,
                                "status": "failed",
                                "error": error
                            })
                            logger.error(f"[API_INGESTION] Job {job_id}: Who's Who failed - {error}")
                
                elif task == "fod_directory":
                    # Run FOD fetcher
                    from fetch_fod_pdf import FODPDFFetcher
                    
                    if dry_run:
                        logger.info(f"[API_INGESTION] [DRY-RUN] Would fetch FOD directory")
                        job["api_results"].append({
                            "api": task,
                            "status": "skipped",
                            "message": "Dry run mode"
                        })
                    else:
                        fetcher = FODPDFFetcher()
                        result = fetcher.fetch_and_upload()
                        
                        if result.get("success"):
                            job["api_results"].append({
                                "api": task,
                                "status": "success",
                                "stats": {
                                    "file": result.get("filename", "N/A"),
                                    "size": result.get("file_size_formatted", "N/A")
                                }
                            })
                            logger.info(f"[API_INGESTION] Job {job_id}: FOD completed - {result.get('filename', 'N/A')}")
                        else:
                            error = result.get("error", "Unknown error")
                            job["api_results"].append({
                                "api": task,
                                "status": "failed",
                                "error": error
                            })
                            logger.error(f"[API_INGESTION] Job {job_id}: FOD failed - {error}")
                
                job["apis_processed"] += 1
                
            except Exception as e:
                logger.error(f"[API_INGESTION] Job {job_id}: Error processing {task}: {e}")
                job["api_results"].append({
                    "api": task,
                    "status": "failed",
                    "error": str(e)
                })
                job["apis_processed"] += 1
        
        # Process document APIs
        for api_name in apis:
            try:
                job["current_api"] = api_name
                
                logger.info(f"[API_INGESTION] Job {job_id}: Processing {api_name}")
                
                # Run ingestion for this API with date filtering
                result = run_ingestion(
                    api_name=api_name,
                    max_results=max_results,
                    target_date=target_date,
                    date_range_days=date_range_days,
                    dry_run=dry_run,
                    skip_duplicates=skip_duplicates,
                    logger=logger
                )
                
                if result["success"]:
                    stats = result["stats"]
                    
                    # Update job stats
                    job["stats"]["total_fetched"] += stats["fetched"]
                    job["stats"]["total_duplicates"] += stats["duplicates"]
                    job["stats"]["total_new_files"] += stats["new_files"]
                    job["stats"]["total_uploaded"] += stats["uploaded"]
                    job["stats"]["total_errors"] += stats["errors"]
                    
                    job["api_results"].append({
                        "api": api_name,
                        "status": "success",
                        "stats": stats
                    })
                    
                    logger.info(f"[API_INGESTION] Job {job_id}: {api_name} completed - {stats['uploaded']} uploaded")
                else:
                    error = result.get("error", "Unknown error")
                    job["api_results"].append({
                        "api": api_name,
                        "status": "failed",
                        "error": error
                    })
                    logger.error(f"[API_INGESTION] Job {job_id}: {api_name} failed - {error}")
                
                job["apis_processed"] += 1
                
            except Exception as e:
                logger.error(f"[API_INGESTION] Job {job_id}: Error processing {api_name}: {e}")
                job["api_results"].append({
                    "api": api_name,
                    "status": "failed",
                    "error": str(e)
                })
                job["apis_processed"] += 1
        
        # Mark job as completed
        job["status"] = "completed"
        job["completed_at"] = ist_now().isoformat()
        job["current_api"] = None
        
        logger.info(f"[API_INGESTION] Job {job_id} completed - {job['stats']['total_uploaded']} files uploaded")
        
        # Trigger PDF processing if new files were uploaded
        if not dry_run and job["stats"]["total_uploaded"] > 0:
            logger.info(f"[API_INGESTION] Job {job_id}: Triggering PDF conversion pipeline")
            asyncio.create_task(run_pdf_conversion(output_dir="./md_files"))
        
    except Exception as e:
        logger.error(f"[API_INGESTION] Job {job_id} failed: {e}")
        job["status"] = "failed"
        job["completed_at"] = ist_now().isoformat()
        job["error"] = str(e)


# Include routers AFTER all endpoints are defined
app.include_router(auth_router)
app.include_router(public_router)
app.include_router(admin_router)
app.include_router(evaluation_router)

@app.get("/{full_path:path}")
async def redirect_to_root(full_path: str):
    # Don't redirect API routes or analytics endpoints - let them return 404 if not found
    if full_path.startswith("auth/") or full_path.startswith("analytics/") or full_path.startswith("audit/") or full_path.startswith("evaluations") or full_path.startswith("start_session") or \
       full_path.startswith("ask") or full_path.startswith("interactions") or \
       full_path.startswith("static/"):
        raise HTTPException(status_code=404, detail="Not Found")
    
    return RedirectResponse(url="/", status_code=302)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)


