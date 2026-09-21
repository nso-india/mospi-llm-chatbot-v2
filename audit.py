"""
Application audit logging — MongoDB-backed, append-only trail for security and compliance.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from models import AdminRole, AuditLog, ist_now, primary_admin_role

logger = logging.getLogger("chatbot")

ActorType = Literal["panel_user", "admin", "api_key", "anonymous", "system"]

SKIP_PATH_PREFIXES = ("/static/",)
SKIP_PATHS = {"/", "/favicon.ico"}

ACTION_OVERRIDES: Dict[tuple[str, str], str] = {
    ("POST", "/auth/login"): "auth.login",
    ("POST", "/auth/logout"): "auth.logout",
    ("GET", "/auth/me"): "auth.me",
    ("GET", "/auth/users"): "auth.users.list",
    ("POST", "/auth/users"): "auth.user.create",
    ("PATCH", "/auth/users/{id}"): "auth.user.update",
    ("DELETE", "/auth/users/{id}"): "auth.user.delete",
    ("GET", "/auth/audit/logs"): "audit.logs.list",
    ("POST", "/upload_file"): "ingestion.upload",
    ("POST", "/start_web_scrape"): "ingestion.scrape.start",
    ("POST", "/convert_pdfs"): "ingestion.convert",
    ("POST", "/chunker"): "ingestion.chunk",
    ("DELETE", "/delete_chunks_by_docname"): "document.chunks.delete",
    ("POST", "/update_metadata_by_doc"): "document.metadata.update",
    ("DELETE", "/delete_session/{id}"): "session.delete",
    ("GET", "/interactions"): "interactions.list",
    ("GET", "/chatbot_logs"): "logs.chatbot.view",
    ("GET", "/download_chatbot_logs"): "logs.chatbot.download",
    ("GET", "/get_chunks_by_docname"): "document.chunks.view",
    ("GET", "/get_chunks_by_url"): "document.chunks.view_by_url",
    ("GET", "/list_documents"): "documents.list",
    ("GET", "/list_url_titles"): "documents.url_titles",
    ("POST", "/ask"): "chatbot.ask",
    ("GET", "/start_session"): "chatbot.session.start",
    ("POST", "/interactions/{id}/feedback"): "chatbot.feedback",
    ("POST", "/monitor/ingested_files"): "monitor.ingestion.report",
    ("GET", "/monitor/chatbot"): "monitor.chatbot.health",
    ("GET", "/scraper/logs/download"): "scraper.logs.download",
    ("GET", "/scraper/logs/view"): "scraper.logs.view",
    ("GET", "/scraper/results"): "scraper.results.view",
    ("GET", "/scheduler/status"): "scheduler.status",
}


def get_client_ip(request: Request) -> Optional[str]:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def normalize_path(path: str) -> str:
    parts = [p for p in path.strip("/").split("/") if p]
    normalized = []
    for part in parts:
        if re.fullmatch(r"[0-9a-f]{24}", part, re.IGNORECASE):
            normalized.append("{id}")
        elif re.fullmatch(r"[0-9a-f-]{36}", part, re.IGNORECASE):
            normalized.append("{id}")
        else:
            normalized.append(part)
    return "/" + "/".join(normalized) if normalized else "/"


def resolve_action(method: str, path: str) -> str:
    normalized = normalize_path(path)
    override = ACTION_OVERRIDES.get((method.upper(), normalized))
    if override:
        return override
    slug = normalized.strip("/").replace("/", ".") or "root"
    return f"api.{method.lower()}.{slug}"


def resolve_resource(path: str) -> tuple[Optional[str], Optional[str]]:
    normalized = normalize_path(path)
    if normalized.startswith("/auth/users/"):
        parts = path.strip("/").split("/")
        return "admin_user", parts[-1] if len(parts) >= 3 else None
    if normalized.startswith("/delete_session/"):
        return "session", path.strip("/").split("/")[-1]
    if normalized.startswith("/interactions/") and normalized.endswith("/feedback"):
        return "interaction", path.strip("/").split("/")[-2]
    if normalized.startswith("/analytics/monthly-report/"):
        return "monthly_report", path.strip("/").split("/")[-1]
    return None, None


def set_audit_actor(
    request: Request,
    *,
    actor_type: ActorType,
    actor_id: Optional[str] = None,
    actor_email: Optional[str] = None,
    actor_role: Optional[AdminRole] = None,
    actor_roles: Optional[List[AdminRole]] = None,
) -> None:
    request.state.audit_actor = {
        "actor_type": actor_type,
        "actor_id": actor_id,
        "actor_email": actor_email,
        "actor_role": actor_role,
        "actor_roles": actor_roles or ([actor_role] if actor_role else None),
    }


def set_audit_panel_user(request: Request, user) -> None:
    roles = user.roles or ["analyst"]
    set_audit_actor(
        request,
        actor_type="panel_user",
        actor_id=str(user.id),
        actor_email=user.email,
        actor_role=primary_admin_role(roles),
        actor_roles=roles,
    )


# Backwards-compatible alias
set_audit_admin_user = set_audit_panel_user


def set_audit_metadata(request: Request, metadata: Dict[str, Any]) -> None:
    existing = getattr(request.state, "audit_metadata", None) or {}
    existing.update(metadata)
    request.state.audit_metadata = existing


def mark_audit_logged(request: Request) -> None:
    request.state.audit_skip = True


async def log_audit(
    *,
    action: str,
    method: str,
    path: str,
    status_code: int,
    success: bool,
    actor_type: ActorType = "anonymous",
    actor_id: Optional[str] = None,
    actor_email: Optional[str] = None,
    actor_role: Optional[AdminRole] = None,
    actor_roles: Optional[List[AdminRole]] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> None:
    try:
        entry = AuditLog(
            actor_type=actor_type,
            actor_id=actor_id,
            actor_email=actor_email,
            actor_role=actor_role,
            actor_roles=actor_roles,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            method=method.upper(),
            path=path,
            status_code=status_code,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
            error_message=error_message,
            timestamp=ist_now(),
        )
        await entry.insert()
    except Exception as exc:
        logger.error("Failed to write audit log: %s", exc, exc_info=True)


async def log_audit_from_request(
    request: Request,
    *,
    status_code: int,
    success: bool,
    action: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    actor = getattr(request.state, "audit_actor", None) or {}
    metadata = getattr(request.state, "audit_metadata", None) or {}
    path = request.url.path
    method = request.method
    resolved_action = action or getattr(request.state, "audit_action", None) or resolve_action(method, path)
    resource_type, resource_id = resolve_resource(path)
    if metadata.get("resource_type"):
        resource_type = metadata.pop("resource_type")
    if metadata.get("resource_id"):
        resource_id = metadata.pop("resource_id")

    await log_audit(
        action=resolved_action,
        method=method,
        path=path,
        status_code=status_code,
        success=success,
        actor_type=actor.get("actor_type", "anonymous"),
        actor_id=actor.get("actor_id"),
        actor_email=actor.get("actor_email"),
        actor_role=actor.get("actor_role"),
        actor_roles=actor.get("actor_roles"),
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
        metadata=metadata,
        error_message=error_message,
    )


class AuditMiddleware(BaseHTTPMiddleware):
    """Record an audit entry for every API request (except static assets)."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in SKIP_PATHS or any(path.startswith(prefix) for prefix in SKIP_PATH_PREFIXES):
            return await call_next(request)

        status_code = 500
        error_message = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            error_message = str(exc)
            raise
        finally:
            if getattr(request.state, "audit_skip", False):
                pass
            else:
                try:
                    success = 200 <= status_code < 400
                    await log_audit_from_request(
                        request,
                        status_code=status_code,
                        success=success,
                        error_message=error_message,
                    )
                except Exception as exc:
                    logger.error("Audit middleware failed: %s", exc, exc_info=True)


async def log_system_audit(
    action: str,
    *,
    success: bool = True,
    status_code: int = 200,
    metadata: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> None:
    await log_audit(
        action=action,
        method="SYSTEM",
        path="/system",
        status_code=status_code,
        success=success,
        actor_type="system",
        metadata=metadata or {},
        error_message=error_message,
    )
