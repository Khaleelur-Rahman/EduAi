"""
Dashboard authentication: signed tokens for WhatsApp link and session cookies.
Uses HMAC-SHA256 with DASHBOARD_SECRET; constant-time comparison for verification.
"""
import os
import hmac
import hashlib
import base64
import time
from typing import Optional

# Token lifetime: 48 hours; session: 7 days
TOKEN_EXPIRY_SECONDS = 48 * 3600
SESSION_EXPIRY_SECONDS = 7 * 24 * 3600


def _get_secret() -> bytes:
    secret = os.getenv("DASHBOARD_SECRET") or os.getenv("SECRET_KEY")
    if not secret:
        raise ValueError("DASHBOARD_SECRET or SECRET_KEY must be set for dashboard auth")
    return secret.encode("utf-8") if isinstance(secret, str) else secret


def _sign(payload: str) -> str:
    key = _get_secret()
    sig = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _verify_signed(signed: str) -> Optional[str]:
    """Verify signature and return payload string or None."""
    if not signed or "." not in signed:
        return None
    key = _get_secret()
    parts = signed.rsplit(".", 1)
    if len(parts) != 2:
        return None
    payload, sig = parts
    expected = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    return payload


def generate_dashboard_token(user_id: int, phone_number: str) -> str:
    """Generate a short-lived signed token for dashboard link (e.g. in WhatsApp /progress)."""
    exp = int(time.time()) + TOKEN_EXPIRY_SECONDS
    payload = f"{user_id}|{phone_number}|{exp}"
    signed = _sign(payload)
    return base64.urlsafe_b64encode(signed.encode("utf-8")).decode("utf-8").rstrip("=")


def verify_dashboard_token(token: str) -> Optional[dict]:
    """
    Verify token and return {"user_id": int, "phone": str} or None.
    Uses constant-time comparison for the signature.
    """
    if not token:
        return None
    try:
        padded = token + "=" * (4 - len(token) % 4) if len(token) % 4 else token
        decoded = base64.urlsafe_b64decode(padded)
        signed = decoded.decode("utf-8")
    except Exception:
        return None
    payload = _verify_signed(signed)
    if not payload:
        return None
    parts = payload.split("|", 2)
    if len(parts) != 3:
        return None
    try:
        user_id = int(parts[0])
        phone = parts[1]
        exp = int(parts[2])
    except ValueError:
        return None
    if exp < int(time.time()):
        return None
    return {"user_id": user_id, "phone": phone}


def create_session_value(user_id: int, phone_number: str) -> str:
    """Create signed session cookie value."""
    exp = int(time.time()) + SESSION_EXPIRY_SECONDS
    payload = f"{user_id}|{phone_number}|{exp}"
    signed = _sign(payload)
    return base64.urlsafe_b64encode(signed.encode("utf-8")).decode("utf-8").rstrip("=")


def verify_session_value(session_value: str) -> Optional[dict]:
    """
    Verify session cookie and return {"user_id": int, "phone": str} or None.
    """
    if not session_value:
        return None
    try:
        padded = session_value + "=" * (4 - len(session_value) % 4) if len(session_value) % 4 else session_value
        decoded = base64.urlsafe_b64decode(padded)
        signed = decoded.decode("utf-8")
    except Exception:
        return None
    payload = _verify_signed(signed)
    if not payload:
        return None
    parts = payload.split("|", 2)
    if len(parts) != 3:
        return None
    try:
        user_id = int(parts[0])
        phone = parts[1]
        exp = int(parts[2])
    except ValueError:
        return None
    if exp < int(time.time()):
        return None
    return {"user_id": user_id, "phone": phone}
