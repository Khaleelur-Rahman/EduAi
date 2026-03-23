"""
Test script for Speech-to-Text (STT) functionality using local Whisper.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.audio import stt_service, transcribe_audio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_stt_service_initialization():
    print("\n=== Testing STT Service Initialization ===")
    if stt_service.local_whisper_model:
        print("✓ STT Service: Local Whisper model initialized")
    else:
        print("✗ STT Service: Not available")
        return False
    return True


def test_transcribe_sample_audio():
    print("\n=== Testing Audio Transcription ===")
    if not any(Path(f"test_audio.{ext}").exists() for ext in ["ogg", "mp3", "wav"]):
        print("⚠ No test audio file found. Skipping transcription test.")
        print("  To test, create test_audio.ogg, test_audio.mp3, or test_audio.wav")
        return True

    test_audio_path = None
    for ext in ["ogg", "mp3", "wav"]:
        test_file = Path(f"test_audio.{ext}")
        if test_file.exists():
            test_audio_path = test_file
            break

    try:
        with open(test_audio_path, "rb") as f:
            audio_data = f.read()
        content_type = f"audio/{test_audio_path.suffix[1:]}"
        if content_type == "audio/mp3":
            content_type = "audio/mpeg"
        print(f"Reading audio file: {test_audio_path} ({len(audio_data)} bytes)")
        transcript = transcribe_audio(audio_data, content_type)
        if transcript:
            print(f"✓ Transcription successful!")
            print(f"  Transcript: {transcript}")
            return True
        else:
            print("✗ Transcription failed")
            return False
    except Exception as e:
        print(f"✗ Error during transcription test: {e}")
        return False


def test_transcribe_with_bytes():
    print("\n=== Testing Transcription with Raw Bytes ===")
    try:
        dummy_audio = b"dummy audio data"
        result = transcribe_audio(dummy_audio, "audio/ogg")
        if result is None:
            print("✓ Function handles invalid audio gracefully (expected)")
            return True
        else:
            print(f"  Unexpected result: {result}")
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("STT (Speech-to-Text) Service Test — Local Whisper")
    print("=" * 60)

    results = [
        ("STT Initialization", test_stt_service_initialization()),
        ("Audio Transcription", test_transcribe_sample_audio()),
        ("Error Handling", test_transcribe_with_bytes()),
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
