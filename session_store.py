
import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Any, List, Optional

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def ist_now() -> datetime:
    """Get current datetime in IST timezone"""
    return datetime.now(IST)

from dotenv import load_dotenv
load_dotenv()

from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document

# Use "chatbot" logger so Redis messages appear in the same log file as the rest of the app
logger = logging.getLogger("chatbot")

# Redis connection settings from env
REDIS_URL = os.getenv("REDIS_URL")  # e.g. redis://localhost:6379/0
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "90"))

_redis_client = None
_KEY_PREFIX = "session:"
_INDEX_KEY = "session:index"
_TRACK_KEY_PREFIX = "session:track:"  # track analytics for this session (proxy/chatbot) or not (our React app)


def _get_redis():
    """Lazy Redis connection (sync client)."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
    except ImportError:
        raise RuntimeError("redis package not installed. Add 'redis' to requirements.txt and install.")
    if REDIS_URL:
        _redis_client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
        )
        logger.info("Redis session store connected (REDIS_URL)")
        print("Redis session store connected (REDIS_URL)", flush=True)
    else:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
        )
        logger.info(f"Redis session store connected (host={REDIS_HOST}, port={REDIS_PORT}, db={REDIS_DB})")
        print(f"Redis session store connected (host={REDIS_HOST}, port={REDIS_PORT}, db={REDIS_DB})", flush=True)
    return _redis_client


def init_redis_session_store():
    """Call once at app startup to ensure Redis is reachable."""
    logger.info("Initializing Redis session store...")
    print("Initializing Redis session store...", flush=True)
    r = _get_redis()
    r.ping()
    logger.info("Redis session store initialized and ping OK.")
    print("Redis session store initialized and ping OK.", flush=True)


# ---------- Serialization ----------
def _serialize_memory(memory) -> dict:
    """ConversationBufferMemory -> JSON-serializable dict."""
    messages = getattr(memory, "chat_memory", None) and getattr(memory.chat_memory, "messages", []) or []
    out = []
    for m in messages:
        typ = getattr(m, "type", None) or ("human" if "HumanMessage" in type(m).__name__ else "ai")
        content = getattr(m, "content", "") or ""
        out.append({"type": typ, "content": content})
    return {"messages": out}


def _deserialize_memory(data: dict):
    """JSON dict -> ConversationBufferMemory with messages restored."""
    mem = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )
    for m in data.get("messages", []):
        typ = (m.get("type") or "human").lower()
        content = m.get("content") or ""
        if typ == "human":
            mem.chat_memory.add_user_message(content)
        else:
            mem.chat_memory.add_ai_message(content)
    return mem


def _serialize_doc(doc) -> dict:
    """LangChain Document (or already-serialized dict) -> dict."""
    if isinstance(doc, dict):
        return doc
    return {
        "page_content": getattr(doc, "page_content", "") or "",
        "metadata": getattr(doc, "metadata", {}) or {},
    }


def _deserialize_doc(data: dict) -> Document:
    """Dict -> LangChain Document."""
    return Document(
        page_content=data.get("page_content", ""),
        metadata=data.get("metadata", {}),
    )


def _serialize_context(context: Any) -> dict:
    """Session context (memory or enhanced dict) -> JSON-serializable dict."""
    if isinstance(context, dict):
        mem = context.get("conversation_memory")
        out = {
            "context_type": "enhanced",
            "conversation_memory": _serialize_memory(mem) if mem else {"messages": []},
            "document_context": context.get("document_context", []),
            "topic_context": context.get("topic_context", []),
            "query_context": context.get("query_context", []),
            "last_retrieval_docs": [_serialize_doc(d) for d in (context.get("last_retrieval_docs") or [])],
        }
        return out
    # Legacy: raw ConversationBufferMemory
    return {"context_type": "memory_only", "conversation_memory": _serialize_memory(context)}


def _deserialize_context(data: dict) -> Any:
    """JSON dict -> session context (memory or enhanced dict)."""
    if data.get("context_type") == "enhanced":
        mem = _deserialize_memory(data.get("conversation_memory", {}))
        docs = [_deserialize_doc(d) for d in data.get("last_retrieval_docs", [])]
        return {
            "conversation_memory": mem,
            "document_context": data.get("document_context", []),
            "topic_context": data.get("topic_context", []),
            "query_context": data.get("query_context", []),
            "last_retrieval_docs": docs,
        }
    # Legacy
    return _deserialize_memory(data.get("conversation_memory", {}))


def _session_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}{session_id}"


def _track_key(session_id: str) -> str:
    return f"{_TRACK_KEY_PREFIX}{session_id}"


def set_session_track(session_id: str, track: bool) -> None:
    """Mark whether to track this session in user_activity (True = proxy/chatbot, False = our React app)."""
    r = _get_redis()
    key = _track_key(session_id)
    ttl = SESSION_TTL_MINUTES * 60
    r.setex(key, ttl, "1" if track else "0")


def get_session_track(session_id: str) -> Optional[bool]:
    """True = track, False = do not track, None = key missing (legacy: treat as track)."""
    r = _get_redis()
    val = r.get(_track_key(session_id))
    if val is None:
        return None
    return val == "1"


def _ttl_seconds(created_at: datetime) -> int:
    """Seconds until session expires (90 min from creation)."""
    # Normalize both datetimes to IST-aware for comparison
    now_ist = ist_now()
    created_at_ist = normalize_to_ist(created_at)
    return max(0, int(SESSION_TTL_MINUTES * 60 - (now_ist - created_at_ist).total_seconds()))

def normalize_to_ist(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime to IST-aware. Handles both naive and aware datetimes."""
    if dt is None:
        return None
    # If timezone-naive, assume it's IST (for backward compatibility)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    # If timezone-aware, convert to IST
    return dt.astimezone(IST)


class RedisSessionStore:
    """
    Dict-like session store backed by Redis.
    Supports: session_id in store, store[session_id], store[session_id] = (created_at, context), del store[session_id], store.items().
    """

    def __contains__(self, session_id: str) -> bool:
        r = _get_redis()
        return r.exists(_session_key(session_id)) > 0

    def __getitem__(self, session_id: str) -> Tuple[datetime, Any]:
        r = _get_redis()
        key = _session_key(session_id)
        raw = r.get(key)
        if raw is None:
            raise KeyError(session_id)
        payload = json.loads(raw)
        created_at = datetime.fromisoformat(payload["created_at"])
        context = _deserialize_context(payload["context"])
        return (created_at, context)

    def __setitem__(self, session_id: str, value: Tuple[datetime, Any]) -> None:
        created_at, context = value
        r = _get_redis()
        key = _session_key(session_id)
        payload = {
            "created_at": created_at.isoformat(),
            "context": _serialize_context(context),
        }
        ttl = _ttl_seconds(created_at)
        r.setex(key, ttl, json.dumps(payload, default=str))
        # Index for cleanup: score = created timestamp
        r.zadd(_INDEX_KEY, {session_id: created_at.timestamp()})

    def __delitem__(self, session_id: str) -> None:
        r = _get_redis()
        key = _session_key(session_id)
        r.delete(key)
        r.zrem(_INDEX_KEY, session_id)

    def items(self):
        """Yield (session_id, (created_at, context)) for all sessions (used by cleanup)."""
        r = _get_redis()
        ids = r.zrange(_INDEX_KEY, 0, -1)
        for sid in ids:
            try:
                yield (sid, self[sid])
            except (KeyError, json.JSONDecodeError):
                continue


def cleanup_expired_sessions_redis(ttl_minutes: int = 90) -> None:
    """Remove sessions older than ttl_minutes from Redis (and index)."""
    r = _get_redis()
    cutoff = (ist_now() - timedelta(minutes=ttl_minutes)).timestamp()
    expired = r.zrangebyscore(_INDEX_KEY, "-inf", cutoff)
    for sid in expired:
        r.delete(_session_key(sid))
        r.zrem(_INDEX_KEY, sid)
        logger.info(f"Cleaned up expired session: {sid}")


def get_session_store() -> RedisSessionStore:
    """Return the shared Redis-backed session store (dict-like)."""
    return RedisSessionStore()
