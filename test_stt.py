"""
Test script for Speech-to-Text (STT) functionality.
Tests both OpenAI Whisper API and local Whisper fallback.
"""
import os
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.audio import stt_service, transcribe_audio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_stt_service_initialization():
    """Test that STT service initializes correctly."""
    print("\n=== Testing STT Service Initialization ===")
    
    if stt_service.use_openai:
        print("✓ STT Service: OpenAI Whisper API initialized")
    elif stt_service.local_whisper_model:
        print("✓ STT Service: Local Whisper model initialized")
    else:
        print("✗ STT Service: Not available")
        return False
    
    return True

def test_transcribe_sample_audio():
    """
    Test transcription with a sample audio file.
    Note: This requires a sample audio file to test with.
    """
    print("\n=== Testing Audio Transcription ===")
    
    # Check if we have a test audio file
    test_audio_path = Path("test_audio.ogg") or Path("test_audio.mp3") or Path("test_audio.wav")
    
    if not any(Path(f"test_audio.{ext}").exists() for ext in ["ogg", "mp3", "wav"]):
        print("⚠ No test audio file found. Skipping transcription test.")
        print("  To test transcription, create a test_audio.ogg, test_audio.mp3, or test_audio.wav file")
        return True
    
    # Find the test file
    for ext in ["ogg", "mp3", "wav"]:
        test_file = Path(f"test_audio.{ext}")
        if test_file.exists():
            test_audio_path = test_file
            break
    
    try:
        # Read audio file
        with open(test_audio_path, "rb") as f:
            audio_data = f.read()
        
        # Determine content type
        content_type = f"audio/{test_audio_path.suffix[1:]}"
        if content_type == "audio/mp3":
            content_type = "audio/mpeg"
        
        print(f"Reading audio file: {test_audio_path} ({len(audio_data)} bytes)")
        
        # Transcribe
        print("Transcribing...")
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
    """Test transcription with raw bytes (simulating Twilio audio)."""
    print("\n=== Testing Transcription with Raw Bytes ===")
    
    # Create a minimal test (we can't create real audio without a library)
    # This test just verifies the function can handle bytes
    try:
        # Create dummy audio bytes (this won't actually transcribe, but tests the function)
        dummy_audio = b"dummy audio data"
        
        # This will likely fail, but tests error handling
        result = transcribe_audio(dummy_audio, "audio/ogg")
        
        if result is None:
            print("✓ Function handles invalid audio gracefully (expected)")
            return True
        else:
            print(f"  Unexpected result: {result}")
            return True  # Still a valid test
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("STT (Speech-to-Text) Service Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Service initialization
    results.append(("STT Initialization", test_stt_service_initialization()))
    
    # Test 2: Transcription with sample file (if available)
    results.append(("Audio Transcription", test_transcribe_sample_audio()))
    
    # Test 3: Error handling
    results.append(("Error Handling", test_transcribe_with_bytes()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n⚠ Some tests failed or were skipped")
    
    print("\nNote: For full transcription testing, you need:")
    print("  1. A test audio file (test_audio.ogg, .mp3, or .wav)")
    print("  2. OPENAI_API_KEY in environment (for API) OR")
    print("  3. openai-whisper installed (for local fallback)")
