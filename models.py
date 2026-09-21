from beanie import Document, init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import EmailStr, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
from typing import Literal
import logging
import os
import uuid
from urllib.parse import quote_plus
from bson import ObjectId
from bson.errors import InvalidId

# ==== MongoDB Models ====

logger = logging.getLogger("chatbot")

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def ist_now() -> datetime:
    """Get current datetime in IST timezone"""
    return datetime.now(IST)


def _normalize_mongo_string(value) -> str:
    """Coerce legacy MongoDB values (int, Binary UUID, bytes) to str."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        if len(raw) == 16:
            try:
                return str(uuid.UUID(bytes=raw))
            except ValueError:
                pass
        return raw.hex()
    # bson.Binary and similar buffer types from PyMongo
    if hasattr(value, "__bytes__"):
        try:
            raw = bytes(value)
            if len(raw) == 16:
                try:
                    return str(uuid.UUID(bytes=raw))
                except ValueError:
                    pass
            return raw.hex()
        except TypeError:
            pass
    return str(value)

class Interaction(Document):
    session_id: str
    timestamp: datetime = Field(default_factory=ist_now)
    query: str
    response: str
    sources: Optional[List[str]] = []
    feedback: Optional[Literal["like", "dislike"]] = None
    # Performance metrics
    response_time_ms: Optional[float] = None  # Total response time in milliseconds
    component_timings: Optional[Dict[str, float]] = None  # {"llm": 1234.5, "vector_search": 234.1, "db": 12.3}

    @field_validator("session_id", mode="before")
    @classmethod
    def coerce_session_id(cls, value):
        return _normalize_mongo_string(value)

    @field_validator("query", "response", mode="before")
    @classmethod
    def coerce_text_fields(cls, value):
        return _normalize_mongo_string(value)

    class Settings:
        name = "interactions"


def _prepare_interaction_doc(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize _id for Beanie; return None if the document cannot be parsed."""
    prepared = dict(doc)
    raw_id = prepared.get("_id")
    if isinstance(raw_id, ObjectId):
        return prepared
    if isinstance(raw_id, str):
        try:
            prepared["_id"] = ObjectId(raw_id)
            return prepared
        except (InvalidId, TypeError, ValueError):
            return None
    return None


async def find_interactions_safe(
    query: Optional[Dict[str, Any]] = None,
    *,
    sort: Optional[str] = None,
) -> List["Interaction"]:
    """
    Load Interaction documents, skipping corrupt rows (e.g. non-ObjectId _id values).
    Production data may contain legacy string IDs that Beanie cannot parse.
    Compatible with Beanie 2.x (get_pymongo_collection) and older Motor APIs.
    """
    get_collection = getattr(Interaction, "get_pymongo_collection", None) or getattr(
        Interaction, "get_motor_collection", None
    )
    if get_collection is None:
        raise RuntimeError("Beanie Interaction collection accessor not available")

    collection = get_collection()
    cursor = collection.find(query or {})
    if sort:
        direction = -1 if sort.startswith("-") else 1
        cursor = cursor.sort(sort.lstrip("-"), direction)

    items: List[Interaction] = []
    skipped = 0

    async def _consume_doc(doc: Dict[str, Any]) -> None:
        nonlocal skipped
        prepared = _prepare_interaction_doc(doc)
        if prepared is None:
            skipped += 1
            logger.warning("Skipping interaction with invalid _id=%r", doc.get("_id"))
            return
        try:
            items.append(Interaction.model_validate(prepared))
        except Exception as exc:
            skipped += 1
            logger.warning(
                "Skipping unparseable interaction _id=%r: %s",
                doc.get("_id"),
                exc,
            )

    # Beanie 2 / PyMongo Async: cursor supports async iteration
    if hasattr(cursor, "__aiter__"):
        async for doc in cursor:
            await _consume_doc(doc)
    else:
        # Sync PyMongo fallback (run in thread if needed — rare in this stack)
        for doc in cursor:
            await _consume_doc(doc)

    if skipped:
        logger.warning("Skipped %s corrupt interaction document(s)", skipped)
    return items


class SessionAnalytics(Document):
    session_id: str
    chat_count: int
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    avg_query_length: float
    avg_response_length: float
    fallback_count: int

    class Settings:
        name = "session_analytics"


class RequestMetrics(Document):
    """Detailed metrics for API requests"""
    endpoint: str  # e.g., "/ask", "/start_session"
    method: str  # "GET", "POST", etc.
    session_id: Optional[str] = None
    response_time_ms: float
    status_code: int
    timestamp: datetime = Field(default_factory=ist_now)
    error: Optional[str] = None  # Error message if request failed
    component_timings: Optional[Dict[str, float]] = None

    class Settings:
        name = "request_metrics"


class UserActivity(Document):
    """Track user activity and sessions. Only proxy/chatbot sessions are stored (from_proxy=True)."""
    session_id: str
    device_id: Optional[str] = None
    source: Optional[str] = None  # Referrer domain e.g. "www.mospi.gov.in"
    from_proxy: bool = False  # True only when session was created from proxy (chatbot); excludes old/React app rows from counts
    first_seen: datetime = Field(default_factory=ist_now)
    last_seen: datetime = Field(default_factory=ist_now)
    total_interactions: int = 0
    total_sessions: int = 1
    is_active: bool = True

    class Settings:
        name = "user_activity"


class MonthlyReport(Document):
    """Monthly analytics reports"""
    report_month: str  # Format: "YYYY-MM" e.g., "2026-02"
    generated_at: datetime = Field(default_factory=ist_now)
    total_users: int
    active_users: int  # Users with at least one interaction
    total_interactions: int
    total_sessions: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate: float  # Percentage of failed requests
    fallback_rate: float  # Percentage of fallback responses
    feedback_stats: Dict[str, Any]  # likes, dislikes (int); feedback_rate, like_percentage (float)
    top_queries: List[Dict[str, Any]]  # [{"query": "...", "count": 10}]
    peak_usage_hours: List[Dict[str, int]]  # [{"hour": 14, "interactions": 50}]
    report_data: Dict[str, Any]  # Full report JSON for flexibility

    class Settings:
        name = "monthly_reports"


AdminRole = Literal["admin", "analyst", "operator"]
AuditActorType = Literal["panel_user", "admin", "api_key", "anonymous", "system"]

VALID_ADMIN_ROLES = frozenset({"admin", "analyst", "operator"})
ROLE_PRIORITY = {"admin": 0, "analyst": 1, "operator": 2}


def normalize_admin_roles(roles: Optional[List[AdminRole]]) -> List[AdminRole]:
    """Deduplicate and validate roles; default to analyst if empty."""
    if not roles:
        return ["analyst"]
    seen = set()
    normalized: List[AdminRole] = []
    for role in roles:
        if role in VALID_ADMIN_ROLES and role not in seen:
            seen.add(role)
            normalized.append(role)
    return normalized or ["analyst"]


def primary_admin_role(roles: List[AdminRole]) -> AdminRole:
    return min(roles, key=lambda role: ROLE_PRIORITY.get(role, 99))


def user_has_any_role(user: "AdminUser", *allowed: AdminRole) -> bool:
    user_roles = set(user.roles or [])
    return bool(user_roles & set(allowed))


def user_is_admin(user: "AdminUser") -> bool:
    return "admin" in (user.roles or [])


class AdminUser(Document):
    email: EmailStr
    username: str
    hashed_password: str
    full_name: Optional[str] = None
    roles: List[AdminRole] = Field(default_factory=lambda: ["analyst"])
    role: Optional[AdminRole] = Field(
        default=None,
        description="Legacy single-role field; kept for MongoDB migration only",
    )
    is_active: bool = True
    created_at: datetime = Field(default_factory=ist_now)
    last_login: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_single_role(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        roles = data.get("roles")
        legacy_role = data.get("role")
        if legacy_role and (not roles or (roles == ["analyst"] and legacy_role != "analyst")):
            data["roles"] = [legacy_role] if isinstance(legacy_role, str) else legacy_role
        return data

    @model_validator(mode="after")
    def sync_roles_from_legacy_role(self) -> "AdminUser":
        if self.role:
            if not self.roles or (self.roles == ["analyst"] and self.role != "analyst"):
                self.roles = normalize_admin_roles([self.role])
            elif self.role not in self.roles:
                self.roles = normalize_admin_roles([*self.roles, self.role])
        elif not self.roles:
            self.roles = ["analyst"]
        return self

    @field_validator("roles", mode="before")
    @classmethod
    def coerce_roles(cls, value: Any) -> List[AdminRole]:
        if value is None:
            return ["analyst"]
        if isinstance(value, str):
            return normalize_admin_roles([value])
        if isinstance(value, list):
            return normalize_admin_roles(value)
        return ["analyst"]

    class Settings:
        name = "admin_users"
        indexes = [
            [("email", 1)],
            [("username", 1)],
            [("roles", 1)],
        ]


async def migrate_legacy_admin_user_roles() -> int:
    """
    One-time style migration: copy legacy `role` -> `roles` in MongoDB and remove `role`.
    Safe to run on every startup.
    """
    get_collection = getattr(AdminUser, "get_pymongo_collection", None) or getattr(
        AdminUser, "get_motor_collection", None
    )
    if get_collection is None:
        return 0

    collection = get_collection()
    migrated = 0
    cursor = collection.find({"role": {"$exists": True}})
    if hasattr(cursor, "__aiter__"):
        async for doc in cursor:
            legacy_role = doc.get("role")
            roles = doc.get("roles")
            if not legacy_role:
                continue
            new_roles = roles
            if not roles or (roles == ["analyst"] and legacy_role != "analyst"):
                new_roles = [legacy_role] if isinstance(legacy_role, str) else roles
            elif isinstance(legacy_role, str) and legacy_role not in (roles or []):
                new_roles = normalize_admin_roles([*(roles or []), legacy_role])
            await collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"roles": new_roles}, "$unset": {"role": ""}},
            )
            migrated += 1
            logger.info(
                "Migrated admin_user %s: role=%s -> roles=%s",
                doc.get("email"),
                legacy_role,
                new_roles,
            )
    else:
        for doc in cursor:
            legacy_role = doc.get("role")
            roles = doc.get("roles")
            if not legacy_role:
                continue
            new_roles = roles
            if not roles or (roles == ["analyst"] and legacy_role != "analyst"):
                new_roles = [legacy_role] if isinstance(legacy_role, str) else roles
            elif isinstance(legacy_role, str) and legacy_role not in (roles or []):
                new_roles = normalize_admin_roles([*(roles or []), legacy_role])
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"roles": new_roles}, "$unset": {"role": ""}},
            )
            migrated += 1
            logger.info(
                "Migrated admin_user %s: role=%s -> roles=%s",
                doc.get("email"),
                legacy_role,
                new_roles,
            )
    if migrated:
        logger.info("Migrated %s admin user(s) from legacy role field", migrated)
    return migrated


class AuditLog(Document):
    actor_type: AuditActorType = "anonymous"
    actor_id: Optional[str] = None
    actor_email: Optional[str] = None
    actor_role: Optional[AdminRole] = None
    actor_roles: Optional[List[AdminRole]] = None

    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None

    method: str
    path: str
    status_code: int
    success: bool
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=ist_now)

    class Settings:
        name = "audit_logs"
        indexes = [
            [("timestamp", -1)],
            [("actor_email", 1), ("timestamp", -1)],
            [("action", 1), ("timestamp", -1)],
            [("success", 1), ("timestamp", -1)],
        ]


# ==== Initialization Function ====

host = os.getenv("MONGO_HOST")
username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
database= os.getenv("MONGO_DB_NAME")

credentials = ""
if username and password:
        credentials = f"{quote_plus(username)}:{quote_plus(password)}"

logger = logging.getLogger(__name__)

async def init_db():
    # Force debug output to stderr (will show in Docker logs)
    import sys
    print("="*80, file=sys.stderr)
    print("[DEBUG] MongoDB Connection Debug", file=sys.stderr)
    print(f"[DEBUG] MONGO_HOST = {host}", file=sys.stderr)
    print(f"[DEBUG] MONGO_USERNAME = {username}", file=sys.stderr)
    print(f"[DEBUG] MONGO_PASSWORD = {'*' * len(password) if password else 'None'}", file=sys.stderr)
    print(f"[DEBUG] MONGO_DB_NAME = {database}", file=sys.stderr)
    print(f"[DEBUG] credentials set = {bool(credentials)}", file=sys.stderr)
    
    # IMPORTANT: When connecting to mospi_dev, must authenticate against mospi database
    # The mospi user is created in the mospi database, not mospi_dev
    if database == "mospi_dev":
        uri = f"mongodb://{credentials}@{host}:27017/{database}?authSource=mospi"
        debug_uri = f"mongodb://{username}:***@{host}:27017/{database}?authSource=mospi"
    else:
        uri = f"mongodb://{credentials}@{host}:27017/{database}"
        debug_uri = f"mongodb://{username}:***@{host}:27017/{database}"
    
    print(f"[DEBUG] URI (masked) = {debug_uri}", file=sys.stderr)
    print("="*80, file=sys.stderr)
    
    try:
        client = AsyncIOMotorClient(uri)
        db = client[database]
        print("[DEBUG] AsyncIOMotorClient created, calling init_beanie()...", file=sys.stderr)

        # Imported here to avoid a module-level circular import:
        # evaluation models use AdminUser/ist_now from this module.
        from evaluation.evaluation import EvaluationJob, EvaluationResult
        
        await init_beanie(
            database=db,
            document_models=[
                Interaction,
                SessionAnalytics,
                RequestMetrics,
                UserActivity,
                MonthlyReport,
                AdminUser,
                AuditLog,
                EvaluationJob,
                EvaluationResult,
            ]
        )
        print("[DEBUG] init_beanie() SUCCESS!", file=sys.stderr)
        print("="*80, file=sys.stderr)
        
    except Exception as e:
        print("="*80, file=sys.stderr)
        print(f"[DEBUG] MongoDB init FAILED: {type(e).__name__}: {str(e)}", file=sys.stderr)
        print("="*80, file=sys.stderr)
        raise
