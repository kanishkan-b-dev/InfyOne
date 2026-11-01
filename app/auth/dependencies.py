"""
FastAPI Dependencies for route authentication and session verification.
"""
import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from app.auth.security import verify_session_token
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_current_user(request: Request) -> Optional[str]:
    """
    Extract and verify the current authenticated user from cookie or Authorization header.
    Returns username if valid; None otherwise.
    """
    cookie_name = getattr(settings, "AUTH_COOKIE_NAME", "infyone_session")

    # 1. Check HTTP-Only Cookie
    token = request.cookies.get(cookie_name)

    # 2. Check Authorization Header (Bearer token)
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()

    if not token:
        return None

    payload = verify_session_token(token)
    if not payload:
        return None

    return payload.get("sub")


def require_auth(request: Request) -> str:
    """
    Strict FastAPI dependency requiring an authenticated user.
    Raises 401 Unauthorized if missing or invalid.
    """
    # If authentication is explicitly disabled in config
    if not getattr(settings, "AUTH_REQUIRED", True):
        return "anonymous_user"

    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in to access InfyOne.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

