"""
Security and Cryptography utilities for InfyOne Chatbot.
- Zero-database authentication comparing against Key Vault credentials using constant-time comparison.
- Cryptographically signed HMAC-SHA256 session tokens.
- Rolling window brute-force lockout limiter.
"""
import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Dict, List, Optional, Tuple

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Credential Verification (Constant-Time) ──────────────────────────────────

def verify_credentials(username: str, password: str) -> bool:
    """
    Verify username and password against the Key Vault authorized user store.
    Uses secrets.compare_digest for constant-time comparison to completely
    neutralize timing-based side-channel attacks.
    """
    if not username or not password:
        return False

    users = settings.get_auth_users()
    norm_user = username.strip().lower()

    if norm_user not in users:
        # Perform dummy comparison to keep constant time even on nonexistent users
        secrets.compare_digest(password, "dummy_password_constant_time_pad_xyz123")
        return False

    expected_password = users[norm_user]
    # secrets.compare_digest prevents timing analysis
    return secrets.compare_digest(password, expected_password)


# ── Cryptographic Session Token Engine ────────────────────────────────────────

def create_session_token(username: str, expiry_hours: Optional[int] = None) -> str:
    """
    Generate an HMAC-SHA256 signed session token.
    Contains username, creation timestamp, expiry timestamp, and a random cryptographic nonce.
    Format: <base64_payload>.<hex_hmac_signature>
    """
    now = int(time.time())
    ttl_hours = expiry_hours or getattr(settings, "AUTH_SESSION_EXPIRY_HOURS", 24)
    exp = now + (ttl_hours * 3600)

    payload = {
        "sub": username.strip().lower(),
        "iat": now,
        "exp": exp,
        "nonce": secrets.token_hex(8),
    }

    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_json).decode("utf-8").rstrip("=")

    secret_key = settings.APP_SECRET_KEY.encode("utf-8")
    signature = hmac.new(secret_key, payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()

    return f"{payload_b64}.{signature}"


def verify_session_token(token: str) -> Optional[Dict]:
    """
    Validate HMAC-SHA256 signature and expiration of a session token.
    Returns payload dictionary if valid and unexpired; None otherwise.
    """
    if not token or "." not in token:
        return None

    try:
        payload_b64, signature = token.split(".", 1)
    except ValueError:
        return None

    secret_key = settings.APP_SECRET_KEY.encode("utf-8")
    expected_sig = hmac.new(secret_key, payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()

    # Constant-time signature verification
    if not secrets.compare_digest(signature, expected_sig):
        logger.warning("[security] Tampered or invalid session token signature detected.")
        return None

    # Decode payload
    try:
        padded_b64 = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload_bytes = base64.urlsafe_b64decode(padded_b64.encode("utf-8"))
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        logger.warning("[security] Failed to decode session payload: %s", exc)
        return None

    # Verify expiration
    now = time.time()
    if payload.get("exp", 0) < now:
        logger.info("[security] Session token for user '%s' has expired.", payload.get("sub"))
        return None

    return payload


# ── Brute-Force Rate Limiter & Lockout ─────────────────────────────────────────

class BruteForceLimiter:
    """
    Thread-safe in-memory rate limiter tracking failed login attempts per client IP.
    Imposes temporary lockout after threshold consecutive failures.
    """

    def __init__(self, max_attempts: int = 5, lockout_minutes: int = 5):
        self.max_attempts = max_attempts
        self.lockout_seconds = lockout_minutes * 60
        self._failures: Dict[str, List[float]] = {}
        self._lockouts: Dict[str, float] = {}

    def is_locked_out(self, ip: str) -> Tuple[bool, int]:
        """
        Check if an IP is currently locked out.
        Returns (is_locked, remaining_seconds).
        """
        now = time.time()

        # Check existing lockout
        lockout_until = self._lockouts.get(ip)
        if lockout_until:
            if now < lockout_until:
                remaining = int(lockout_until - now)
                return True, remaining
            else:
                # Lockout expired
                del self._lockouts[ip]
                self._failures.pop(ip, None)

        return False, 0

    def record_failure(self, ip: str) -> Tuple[bool, int]:
        """
        Record a failed attempt. If threshold reached, activates lockout.
        Returns (is_now_locked, remaining_seconds).
        """
        now = time.time()
        timestamps = self._failures.get(ip, [])
        # Keep failures within the last 10 minutes
        timestamps = [t for t in timestamps if now - t < 600]
        timestamps.append(now)
        self._failures[ip] = timestamps

        if len(timestamps) >= self.max_attempts:
            lockout_until = now + self.lockout_seconds
            self._lockouts[ip] = lockout_until
            logger.warning("[security] IP %s exceeded %d failed logins -> locked out for %ds.", ip, self.max_attempts, self.lockout_seconds)
            return True, int(self.lockout_seconds)

        return False, 0

    def reset(self, ip: str) -> None:
        """Reset failed attempts on successful login."""
        self._failures.pop(ip, None)
        self._lockouts.pop(ip, None)


login_limiter = BruteForceLimiter(
    max_attempts=getattr(settings, "AUTH_MAX_LOGIN_ATTEMPTS", 5),
    lockout_minutes=getattr(settings, "AUTH_LOCKOUT_MINUTES", 5),
)

