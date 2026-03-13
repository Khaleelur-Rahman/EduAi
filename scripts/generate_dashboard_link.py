#!/usr/bin/env python3
"""
Generate a signed dashboard URL for testing. Run from project root with .env loaded.

Usage:
  export $(grep -v '^#' .env | xargs)
  python scripts/generate_dashboard_link.py +1234567890

Or with explicit secret:
  DASHBOARD_SECRET=your-secret python scripts/generate_dashboard_link.py +1234567890

The user must already exist in the DB (e.g. from using EduBot on WhatsApp).
"""

import os
import sys

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_dashboard_link.py <phone_number>")
        print("Example: python scripts/generate_dashboard_link.py +1234567890")
        sys.exit(1)

    phone = sys.argv[1].strip()
    if not phone.startswith("+"):
        phone = "+" + phone

    secret = os.getenv("DASHBOARD_SECRET") or os.getenv("SECRET_KEY")
    if not secret:
        print("Set DASHBOARD_SECRET or SECRET_KEY in .env or environment.")
        sys.exit(1)

    # Get user_id from DB
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.db import SessionLocal, get_user_by_phone

    db = SessionLocal()
    try:
        user = get_user_by_phone(db, phone)
        if not user:
            print(f"No user found for {phone}. Use EduBot on WhatsApp first to create an account.")
            sys.exit(1)
        from app.dashboard_auth import generate_dashboard_token
        token = generate_dashboard_token(user.id, user.phone_number)
    finally:
        db.close()

    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    base_url = base_url.rstrip("/")
    url = f"{base_url}/dashboard?token={token}"
    print(f"Dashboard link for {phone} (user_id={user.id}):")
    print(url)
    print("\nOpen this URL in a browser to view the dashboard (no PIN required).")


if __name__ == "__main__":
    main()
