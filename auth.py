"""
Admin panel authentication: JWT + MongoDB AdminUser + RBAC helpers.
"""
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel, EmailStr, Field, field_validator

from models import (
    AdminRole,
    AdminUser,
    AuditLog,
    ist_now,
    normalize_admin_roles,
    primary_admin_role,
    user_has_any_role,
    user_is_admin,
)
from audit import (
    get_client_ip,
    log_audit,
    mark_audit_logged,
    set_audit_admin_user,
    set_audit_metadata,
)

logger = logging.getLogger("chatbot")

JWT_SECRET = (os.getenv("JWT_SECRET") or "").strip()
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))

security = HTTPBearer(auto_error=False)

auth_router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: Optional[str] = None
    username: Optional[str] = None
    password: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserMeResponse(BaseModel):
    id: str
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    roles: List[AdminRole]


class UserListResponse(BaseModel):
    id: str
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    roles: List[AdminRole]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class CreateUserRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=8)
    full_name: Optional[str] = None
    roles: List[AdminRole] = Field(default_factory=lambda: ["analyst"], min_length=1)

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, value: List[AdminRole]) -> List[AdminRole]:
        return normalize_admin_roles(value)


class UpdateUserRequest(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(default=None, min_length=2, max_length=64)
    password: Optional[str] = Field(default=None, min_length=8)
    full_name: Optional[str] = None
    roles: Optional[List[AdminRole]] = None
    is_active: Optional[bool] = None

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, value: Optional[List[AdminRole]]) -> Optional[List[AdminRole]]:
        if value is None:
            return None
        return normalize_admin_roles(value)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except ValueError:
        return False


def create_access_token(user: AdminUser) -> str:
    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET environment variable is not set")
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "username": user.username,
        "roles": user.roles,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _user_to_me(user: AdminUser) -> UserMeResponse:
    return UserMeResponse(
        id=str(user.id),
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        roles=user.roles,
    )


def _user_to_list_item(user: AdminUser) -> UserListResponse:
    return UserListResponse(
        id=str(user.id),
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        roles=user.roles,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login,
    )


async def _count_active_admins() -> int:
    return await AdminUser.find(
        AdminUser.roles == "admin",
        AdminUser.is_active == True,
    ).count()


async def _would_remove_last_active_admin(user: AdminUser, new_roles: List[AdminRole]) -> bool:
    if not user_is_admin(user):
        return False
    if "admin" in new_roles:
        return False
    return await _count_active_admins() <= 1


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> AdminUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        )
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError:
        logger.warning("JWT validation failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await AdminUser.get(user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")
    set_audit_admin_user(request, user)
    return user


def require_roles(*allowed_roles: AdminRole) -> Callable:
    allowed = set(allowed_roles)

    async def _check(user: AdminUser = Depends(get_current_user)) -> AdminUser:
        if not user_has_any_role(user, *allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user

    return _check


@auth_router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request):
    identifier = body.email or body.username
    ip_address = get_client_ip(request)
    user_agent = request.headers.get("user-agent")

    if not body.email and not body.username:
        await log_audit(
            action="auth.login",
            method="POST",
            path="/auth/login",
            status_code=400,
            success=False,
            actor_type="anonymous",
            ip_address=ip_address,
            user_agent=user_agent,
            metadata={"identifier": identifier},
            error_message="email or username is required",
        )
        mark_audit_logged(request)
        raise HTTPException(status_code=400, detail="email or username is required")

    user: Optional[AdminUser] = None
    if body.email:
        user = await AdminUser.find_one(AdminUser.email == body.email.strip().lower())
    if user is None and body.username:
        user = await AdminUser.find_one(AdminUser.username == body.username.strip())

    if not user or not verify_password(body.password, user.hashed_password):
        logger.warning("Failed login attempt for identifier=%s", identifier)
        await log_audit(
            action="auth.login",
            method="POST",
            path="/auth/login",
            status_code=401,
            success=False,
            actor_type="anonymous",
            ip_address=ip_address,
            user_agent=user_agent,
            metadata={"identifier": identifier},
            error_message="Invalid credentials",
        )
        mark_audit_logged(request)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    if not user.is_active:
        await log_audit(
            action="auth.login",
            method="POST",
            path="/auth/login",
            status_code=403,
            success=False,
            actor_type="anonymous",
            actor_email=user.email,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata={"identifier": identifier},
            error_message="Account is disabled",
        )
        mark_audit_logged(request)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")

    user.last_login = ist_now()
    await user.save()

    await log_audit(
        action="auth.login",
        method="POST",
        path="/auth/login",
        status_code=200,
        success=True,
        actor_type="panel_user",
        actor_id=str(user.id),
        actor_email=user.email,
        actor_role=primary_admin_role(user.roles),
        actor_roles=user.roles,
        resource_type="admin_user",
        resource_id=str(user.id),
        ip_address=ip_address,
        user_agent=user_agent,
        metadata={"identifier": identifier},
    )
    mark_audit_logged(request)
    set_audit_admin_user(request, user)

    return TokenResponse(access_token=create_access_token(user))


@auth_router.get("/me", response_model=UserMeResponse)
async def me(user: AdminUser = Depends(get_current_user)):
    return _user_to_me(user)


@auth_router.post("/logout")
async def logout(request: Request, user: AdminUser = Depends(get_current_user)):
    set_audit_metadata(request, {"resource_type": "admin_user", "resource_id": str(user.id)})
    return {"message": "Logged out successfully"}


@auth_router.get("/users", response_model=list[UserListResponse])
async def list_users(_admin: AdminUser = Depends(require_roles("admin"))):
    users = await AdminUser.find_all().sort("-created_at").to_list()
    return [_user_to_list_item(user) for user in users]


@auth_router.post("/users", response_model=UserListResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: CreateUserRequest,
    request: Request,
    admin: AdminUser = Depends(require_roles("admin")),
):
    email = body.email.strip().lower()
    username = body.username.strip()

    if await AdminUser.find_one(AdminUser.email == email):
        raise HTTPException(status_code=400, detail="Email already exists")
    if await AdminUser.find_one(AdminUser.username == username):
        raise HTTPException(status_code=400, detail="Username already exists")

    user = AdminUser(
        email=email,
        username=username,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        roles=body.roles,
    )
    await user.insert()
    logger.info("Admin user created: email=%s roles=%s", user.email, user.roles)
    set_audit_metadata(
        request,
        {
            "resource_type": "admin_user",
            "resource_id": str(user.id),
            "created_email": user.email,
            "created_username": user.username,
            "created_roles": user.roles,
            "created_by": admin.email,
        },
    )
    return _user_to_list_item(user)


@auth_router.patch("/users/{user_id}", response_model=UserListResponse)
async def update_user(
    user_id: str,
    body: UpdateUserRequest,
    request: Request,
    admin: AdminUser = Depends(require_roles("admin")),
):
    user = await AdminUser.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    before = {
        "email": user.email,
        "username": user.username,
        "full_name": user.full_name,
        "roles": user.roles,
        "is_active": user.is_active,
    }
    if body.email is not None:
        email = body.email.strip().lower()
        existing = await AdminUser.find_one(AdminUser.email == email)
        if existing and str(existing.id) != user_id:
            raise HTTPException(status_code=400, detail="Email already exists")
        user.email = email

    if body.username is not None:
        username = body.username.strip()
        existing = await AdminUser.find_one(AdminUser.username == username)
        if existing and str(existing.id) != user_id:
            raise HTTPException(status_code=400, detail="Username already exists")
        user.username = username

    if body.password is not None:
        user.hashed_password = hash_password(body.password)

    if body.full_name is not None:
        user.full_name = body.full_name

    if body.roles is not None and set(body.roles) != set(user.roles):
        if await _would_remove_last_active_admin(user, body.roles):
            raise HTTPException(status_code=400, detail="Cannot remove admin role from the only active admin")
        user.roles = body.roles

    if body.is_active is not None and body.is_active != user.is_active:
        if str(user.id) == str(admin.id) and not body.is_active:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
        if user_is_admin(user) and not body.is_active:
            if await _count_active_admins() <= 1:
                raise HTTPException(status_code=400, detail="Cannot deactivate the only active admin")
        user.is_active = body.is_active

    await user.save()
    logger.info("Admin user updated: id=%s", user_id)
    set_audit_metadata(
        request,
        {
            "resource_type": "admin_user",
            "resource_id": user_id,
            "updated_by": admin.email,
            "before": before,
            "after": {
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "roles": user.roles,
                "is_active": user.is_active,
                "password_changed": body.password is not None,
            },
        },
    )
    return _user_to_list_item(user)


@auth_router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    admin: AdminUser = Depends(require_roles("admin")),
):
    if str(admin.id) == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    user = await AdminUser.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user_is_admin(user):
        admin_count = await AdminUser.find(AdminUser.roles == "admin").count()
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete the only admin")

    deleted_snapshot = {
        "email": user.email,
        "username": user.username,
        "roles": user.roles,
    }
    await user.delete()
    logger.info("Admin user deleted: id=%s", user_id)
    set_audit_metadata(
        request,
        {
            "resource_type": "admin_user",
            "resource_id": user_id,
            "deleted_by": admin.email,
            "deleted_user": deleted_snapshot,
        },
    )
    return {"message": "User deleted successfully"}


class AuditLogItem(BaseModel):
    id: str
    actor_type: str
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
    metadata: dict = {}
    error_message: Optional[str] = None
    timestamp: datetime


class AuditLogListResponse(BaseModel):
    items: List[AuditLogItem]
    total: int
    page: int
    page_size: int
    total_pages: int


def _audit_log_to_item(log: AuditLog) -> AuditLogItem:
    return AuditLogItem(
        id=str(log.id),
        actor_type=log.actor_type,
        actor_id=log.actor_id,
        actor_email=log.actor_email,
        actor_role=log.actor_role,
        actor_roles=log.actor_roles or [],
        action=log.action,
        resource_type=log.resource_type,
        resource_id=log.resource_id,
        method=log.method,
        path=log.path,
        status_code=log.status_code,
        success=log.success,
        ip_address=log.ip_address,
        user_agent=log.user_agent,
        metadata=log.metadata or {},
        error_message=log.error_message,
        timestamp=log.timestamp,
    )


def _parse_filter_date(date_str: str) -> datetime:
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.strptime(date_str.strip(), "%Y-%m-%d").replace(tzinfo=ist)


@auth_router.get("/audit/logs", response_model=AuditLogListResponse)
async def list_audit_logs(
    _admin: AdminUser = Depends(require_roles("admin")),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    action: Optional[str] = None,
    actor_email: Optional[str] = None,
    success: Optional[bool] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    from beanie.operators import And, GTE, LTE, RegEx

    conditions = []
    if action:
        conditions.append(RegEx(AuditLog.action, action, "i"))
    if actor_email:
        conditions.append(AuditLog.actor_email == actor_email.strip().lower())
    if success is not None:
        conditions.append(AuditLog.success == success)
    if from_date:
        conditions.append(GTE(AuditLog.timestamp, _parse_filter_date(from_date)))
    if to_date:
        end = _parse_filter_date(to_date).replace(hour=23, minute=59, second=59)
        conditions.append(LTE(AuditLog.timestamp, end))

    cursor = AuditLog.find(And(*conditions)) if conditions else AuditLog.find_all()
    total = await cursor.count()
    logs = await cursor.sort("-timestamp").skip((page - 1) * page_size).limit(page_size).to_list()
    total_pages = max(1, (total + page_size - 1) // page_size)

    return AuditLogListResponse(
        items=[_audit_log_to_item(log) for log in logs],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
