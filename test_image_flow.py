"""
Test script for the image generation flow: prompt building, ImageService,
handler integration, and response shape.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_build_educational_prompt():
    """Educational prompt is built from topic."""
    from app.image import _build_educational_prompt

    prompt = _build_educational_prompt("cells")
    assert "cells" in prompt.lower()
    assert "educational" in prompt.lower()
    assert "illustration" in prompt.lower()
    assert "children" in prompt.lower()
    print("  ✓ Educational prompt built from topic")


def test_ensure_under_size_compresses_image():
    """_ensure_under_size produces output under 5MB and converts to JPEG."""
    from app.image import _ensure_under_size, MAX_IMAGE_SIZE_BYTES
    from PIL import Image
    import io

    # Create image (any size)
    img = Image.new("RGB", (800, 600), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()

    result, content_type = _ensure_under_size(raw, "image/png")
    assert len(result) <= MAX_IMAGE_SIZE_BYTES, f"Result {len(result)} > {MAX_IMAGE_SIZE_BYTES}"
    assert content_type == "image/jpeg"
    assert len(result) > 0
    print(f"  ✓ Image compressed: {len(raw)} -> {len(result)} bytes, content_type={content_type}")


def test_image_service_interface():
    """ImageService has generate method and returns None when HF not configured."""
    from app.image import image_service

    assert hasattr(image_service, "generate")
    assert callable(image_service.generate)
    # When use_hf is False, generate returns None
    if not image_service.use_hf:
        result = image_service.generate("cells")
        assert result is None
        print("  ✓ ImageService returns None when HF not configured")
    else:
        # When configured, we don't call (would hit API); just verify interface
        print("  ✓ ImageService has generate() (HF configured)")


def test_process_whatsapp_message_request_image_response_shape():
    """process_whatsapp_message_request_image returns dict with text and optionally image_bytes."""
    from unittest.mock import patch
    from app.handlers import process_whatsapp_message_request_image
    from app.db import SessionLocal, create_tables

    create_tables()
    db = SessionLocal()
    try:
        phone = "+15559999999"
        # Missing topic - no LLM needed
        r = process_whatsapp_message_request_image(db, phone, "/image")
        assert isinstance(r, dict)
        assert "text" in r
        assert "Please specify a topic" in r["text"]
        assert "image_bytes" not in r or r.get("image_bytes") is None
        print("  ✓ /image without topic returns error message")

        # With topic - mock process_whatsapp_message to avoid slow LLM
        with patch("app.handlers.process_whatsapp_message", return_value="Lesson text here."):
            r = process_whatsapp_message_request_image(db, phone, "/image cells")
        assert isinstance(r, dict)
        assert "text" in r
        assert r["text"] == "Lesson text here."
        if r.get("image_bytes"):
            assert isinstance(r["image_bytes"], bytes)
            assert "image_content_type" in r
            print(f"  ✓ /image cells returned text + image ({len(r['image_bytes'])} bytes)")
        else:
            print("  ⊘ /image cells returned text only (HF_TOKEN not set)")
    finally:
        db.close()


def test_image_response_routing():
    """Response with image_bytes routes to image send path."""
    def route(response):
        if isinstance(response, dict) and response.get("image_bytes"):
            return "image"
        if isinstance(response, dict) and response.get("audio_bytes"):
            return "audio"
        if isinstance(response, dict) and response.get("text"):
            return "text"
        return "none"

    assert route({"text": "Hi", "image_bytes": b"x", "image_content_type": "image/jpeg"}) == "image"
    assert route({"text": "Hi", "audio_bytes": b"x"}) == "audio"
    assert route({"text": "Hi"}) == "text"
    assert route({"text": "Hi", "image_bytes": None}) == "text"  # None -> falsy, may not match
    print("  ✓ Response routing: image_bytes -> image path")


def test_store_temp_image_and_serve():
    """Image store and serve endpoint work correctly."""
    from app.main import _temp_image_store, _store_temp_image
    from fastapi.testclient import TestClient
    from app.main import app

    # Clear store for test
    _temp_image_store.clear()
    fake_img = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Minimal PNG-like bytes
    img_id = _store_temp_image(fake_img, "image/png")
    assert img_id in _temp_image_store
    assert _temp_image_store[img_id]["bytes"] == fake_img
    assert _temp_image_store[img_id]["content_type"] == "image/png"
    print(f"  ✓ _store_temp_image stores and returns id: {img_id[:8]}...")

    client = TestClient(app)
    resp = client.get(f"/image/{img_id}")
    assert resp.status_code == 200
    assert resp.content == fake_img
    assert "image/png" in resp.headers.get("content-type", "")
    print("  ✓ GET /image/{id} returns stored image")

    # Non-existent
    resp404 = client.get("/image/nonexistent-id")
    assert resp404.status_code == 404
    print("  ✓ GET /image/nonexistent returns 404")


def run_all(skip_slow: bool = False):
    """Run all tests. Set skip_slow=True to skip tests that load full app (handlers, main)."""
    print("\n=== Image flow tests ===\n")
    outcomes = []
    tests = [
        ("Build educational prompt", test_build_educational_prompt),
        ("Compress image", test_ensure_under_size_compresses_image),
        ("ImageService interface", test_image_service_interface),
        ("Image response routing", test_image_response_routing),
    ]
    if not skip_slow:
        tests.extend([
            ("process_whatsapp_message_request_image shape", test_process_whatsapp_message_request_image_response_shape),
            ("Store and serve image", test_store_temp_image_and_serve),
        ])
    for name, fn in tests:
        try:
            fn()
            outcomes.append((name, True, None))
        except Exception as e:
            outcomes.append((name, False, str(e)))
            import traceback
            traceback.print_exc()

    print("\n--- Summary ---")
    for name, ok, err in outcomes:
        status = "PASS" if ok else "FAIL"
        extra = f" ({err})" if err else ""
        print(f"  [{status}] {name}{extra}")
    passed = sum(1 for _, ok, _ in outcomes if ok)
    print(f"\n{passed}/{len(outcomes)} passed")
    return passed == len(outcomes)


if __name__ == "__main__":
    import os
    skip_slow = os.getenv("SKIP_SLOW_IMAGE_TESTS", "").lower() in ("1", "true", "yes")
    if skip_slow:
        print("(SKIP_SLOW_IMAGE_TESTS=1: skipping handler + store/serve tests)\n")
    ok = run_all(skip_slow=skip_slow)
    sys.exit(0 if ok else 1)
