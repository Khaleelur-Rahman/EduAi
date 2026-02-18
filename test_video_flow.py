"""
Tests for video generation: short-video-maker only (Cerebras narration + TTS + Pexels + Remotion).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_build_short_video_script():
    """Fallback script includes topic."""
    from app.video import _build_short_video_script

    script = _build_short_video_script("cells")
    assert "cells" in script.lower()
    assert "learning" in script.lower()
    print("  ✓ Fallback script built from topic")


def test_build_search_terms():
    """Search terms include topic and generic terms."""
    from app.video import _build_search_terms

    terms = _build_search_terms("photosynthesis")
    assert "photosynthesis" in terms
    assert "science" in terms
    print("  ✓ Search terms built from topic")


def test_video_service_interface():
    """VideoService has generate and respects SHORT_VIDEO_MAKER_URL."""
    from app.video import video_service, generate_lesson_video

    assert hasattr(video_service, "generate")
    assert callable(video_service.generate)
    if not video_service.enabled:
        out = video_service.generate("cells")
        assert out is None
        print("  ✓ VideoService returns None when SHORT_VIDEO_MAKER_URL not set")
    else:
        print("  ✓ VideoService enabled (SHORT_VIDEO_MAKER_URL set)")


def test_process_whatsapp_message_request_video_shape():
    """process_whatsapp_message_request_video returns dict with text and optionally video_bytes."""
    from app.handlers import process_whatsapp_message_request_video
    from app.db import SessionLocal, create_tables, get_user_by_phone, create_user, update_user

    create_tables()
    db = SessionLocal()
    try:
        phone = "+15551234567"
        r = process_whatsapp_message_request_video(db, phone, "/video")
        assert isinstance(r, dict)
        assert "text" in r
        assert "topic" in r["text"].lower() or "specify" in r["text"].lower()
        assert "video_bytes" not in r or r.get("video_bytes") is None
        print("  ✓ /video without topic returns error message")

        user = get_user_by_phone(db, phone)
        if not user:
            user = create_user(db, phone)
        update_user(db, user, is_onboarded=True, name="Test", age=10, onboarding_step="completed")

        r = process_whatsapp_message_request_video(db, phone, "/video cells")
        assert isinstance(r, dict)
        assert "text" in r
        if r.get("video_bytes"):
            assert isinstance(r["video_bytes"], bytes)
            assert r.get("video_content_type") == "video/mp4"
            print(f"  ✓ /video cells returned text + video ({len(r['video_bytes'])} bytes)")
        else:
            print("  ✓ /video cells returned text (video disabled or short-video-maker failed)")
    finally:
        db.close()


def test_video_store_and_serve():
    """Stored video is served at GET /video/{id}."""
    from fastapi.testclient import TestClient
    from app.main import app, _temp_video_store, _store_temp_video

    client = TestClient(app)
    fake_video = b"\x00\x00\x00\x00fake_mp4_content"
    vid = _store_temp_video(fake_video, "video/mp4")
    assert vid in _temp_video_store
    resp = client.get(f"/video/{vid}")
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("video/mp4")
    assert resp.content == fake_video
    print("  ✓ GET /video/{id} returns stored video")

    resp404 = client.get("/video/nonexistent-id")
    assert resp404.status_code == 404
    print("  ✓ GET /video/nonexistent returns 404")


if __name__ == "__main__":
    print("Video flow tests\n")
    test_build_short_video_script()
    test_build_search_terms()
    test_video_service_interface()
    test_process_whatsapp_message_request_video_shape()
    test_video_store_and_serve()
    print("\nDone.")
