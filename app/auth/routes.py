"""
Authentication router: login, logout, and session inspection endpoints.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user
from app.auth.security import create_session_token, login_limiter, verify_credentials
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)


class LoginResponse(BaseModel):
    status: str
    username: str
    token: str
    message: str


class StatusResponse(BaseModel):
    authenticated: bool
    username: Optional[str] = None


@router.post("/login", response_model=LoginResponse, summary="Sign in with Key Vault credentials")
async def login(request: Request, response: Response, body: LoginRequest):
    """
    Authenticate against authorized Key Vault users.
    Sets secure HttpOnly cookie and returns signed session token.
    Enforces brute-force lockout protection.
    """
    client_ip = request.client.host if request.client else "127.0.0.1"

    # 1. Check if IP is currently locked out
    is_locked, remaining_seconds = login_limiter.is_locked_out(client_ip)
    if is_locked:
        logger.warning("[auth] Blocked login attempt from locked-out IP %s (%ds remaining)", client_ip, remaining_seconds)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Your IP has been temporarily locked out. Please try again in {remaining_seconds} seconds.",
        )

    # 2. Constant-time credential verification
    valid = verify_credentials(body.username, body.password)
    if not valid:
        is_now_locked, lockout_secs = login_limiter.record_failure(client_ip)
        logger.warning("[auth] Failed login attempt for user '%s' from IP %s", body.username, client_ip)
        if is_now_locked:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many failed login attempts. Your IP is now locked out for {lockout_secs // 60} minutes.",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password. Please check your credentials.",
        )

    # 3. Successful login -> reset limiter
    login_limiter.reset(client_ip)
    username_clean = body.username.strip().lower()
    token = create_session_token(username_clean)

    # 4. Set HttpOnly cookie
    cookie_name = getattr(settings, "AUTH_COOKIE_NAME", "infyone_session")
    is_prod = getattr(settings, "APP_ENV", "production").lower() == "production"
    max_age_secs = getattr(settings, "AUTH_SESSION_EXPIRY_HOURS", 24) * 3600

    response.set_cookie(
        key=cookie_name,
        value=token,
        max_age=max_age_secs,
        httponly=True,
        samesite="lax",
        secure=is_prod,
        path="/",
    )

    logger.info("[auth] User '%s' successfully signed in from IP %s", username_clean, client_ip)
    return LoginResponse(
        status="success",
        username=username_clean,
        token=token,
        message="Sign-in successful. Redirecting to InfyOne...",
    )


@router.post("/logout", summary="Sign out and clear session cookie")
async def logout(response: Response):
    """
    Clear session cookie and invalidate current client session.
    """
    cookie_name = getattr(settings, "AUTH_COOKIE_NAME", "infyone_session")
    response.delete_cookie(key=cookie_name, path="/")
    return {"status": "success", "message": "Successfully signed out."}


@router.get("/me", response_model=StatusResponse, summary="Inspect current session status")
async def get_my_status(request: Request):
    """
    Check if the current caller is authenticated.
    """
    user = get_current_user(request)
    if user:
        return StatusResponse(authenticated=True, username=user)
    return StatusResponse(authenticated=False, username=None)

