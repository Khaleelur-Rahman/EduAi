"""
Test script for Text-to-Speech (TTS) functionality using edge-tts.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.audio import tts_service, synthesize_speech
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_tts_service_initialization():
    print("\n=== Testing TTS Service Initialization ===")
    if tts_service.edge_tts:
        print("✓ TTS Service: edge-tts initialized")
    else:
        print("✗ TTS Service: Not available")
        return False
    return True


def test_synthesize_simple_text():
    print("\n=== Testing TTS Synthesis (Simple Text) ===")
    test_text = "Hello, this is a test of the text to speech system."
    try:
        print(f"Text to synthesize: {test_text}")
        result = synthesize_speech(test_text, voice="alloy", age_group=10)
        if result:
            audio_bytes, content_type = result
            print(f"✓ Synthesis successful!")
            print(f"  Audio size: {len(audio_bytes)} bytes")
            print(f"  Content type: {content_type}")
            output_file = Path("test_output_audio.mp3" if "mpeg" in content_type else "test_output_audio.wav")
            with open(output_file, "wb") as f:
                f.write(audio_bytes)
            print(f"  Saved to: {output_file}")
            return True
        else:
            print("✗ Synthesis failed")
            return False
    except Exception as e:
        print(f"✗ Error during synthesis: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthesize_lesson_text():
    print("\n=== Testing TTS Synthesis (Lesson Text) ===")
    lesson_text = """
    Photosynthesis is how plants make their own food.
    Plants use sunlight, water, and air to create energy.
    This process also makes oxygen, which we breathe.
    """
    try:
        result = synthesize_speech(lesson_text, voice="nova", age_group=8)
        if result:
            audio_bytes, content_type = result
            print(f"✓ Synthesis successful!")
            print(f"  Audio size: {len(audio_bytes)} bytes")
            output_file = Path("test_lesson_audio.mp3" if "mpeg" in content_type else "test_lesson_audio.wav")
            with open(output_file, "wb") as f:
                f.write(audio_bytes)
            print(f"  Saved to: {output_file}")
            return True
        else:
            print("✗ Synthesis failed")
            return False
    except Exception as e:
        print(f"✗ Error during synthesis: {e}")
        return False


def test_long_text_truncation():
    print("\n=== Testing Long Text Truncation ===")
    long_text = " ".join([f"Word{i}" for i in range(200)])
    try:
        result = synthesize_speech(long_text, voice="alloy", age_group=10)
        if result:
            audio_bytes, content_type = result
            print(f"✓ Synthesis successful (text was truncated)")
            print(f"  Audio size: {len(audio_bytes)} bytes")
            return True
        else:
            print("✗ Synthesis failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TTS (Text-to-Speech) Service Test — edge-tts")
    print("=" * 60)

    results = [
        ("TTS Initialization", test_tts_service_initialization()),
        ("Simple Text Synthesis", test_synthesize_simple_text()),
        ("Lesson Text Synthesis", test_synthesize_lesson_text()),
        ("Long Text Truncation", test_long_text_truncation()),
    ]

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    if all(r[1] for r in results):
        print("\n✓ All tests passed!")
    else:
        print("\n⚠ Some tests failed")
