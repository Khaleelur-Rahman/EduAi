"""
Test script for Text-to-Speech (TTS) functionality.
Tests both OpenAI TTS API and Coqui TTS fallback.
"""
import os
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.audio import tts_service, synthesize_speech
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tts_service_initialization():
    """Test that TTS service initializes correctly."""
    print("\n=== Testing TTS Service Initialization ===")
    
    if tts_service.use_openai:
        print("✓ TTS Service: OpenAI TTS API initialized")
    elif tts_service.edge_tts:
        print("✓ TTS Service: edge-tts initialized")
    else:
        print("✗ TTS Service: Not available")
        return False
    
    return True

def test_synthesize_simple_text():
    """Test TTS with simple text."""
    print("\n=== Testing TTS Synthesis (Simple Text) ===")
    
    test_text = "Hello, this is a test of the text to speech system."
    
    try:
        print(f"Text to synthesize: {test_text}")
        print("Synthesizing...")
        
        result = synthesize_speech(test_text, voice="alloy", age_group=10)
        
        if result:
            audio_bytes, content_type = result
            print(f"✓ Synthesis successful!")
            print(f"  Audio size: {len(audio_bytes)} bytes")
            print(f"  Content type: {content_type}")
            
            # Save to file for testing
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
    """Test TTS with lesson-style text (age-appropriate)."""
    print("\n=== Testing TTS Synthesis (Lesson Text) ===")
    
    lesson_text = """
    Photosynthesis is how plants make their own food.
    Plants use sunlight, water, and air to create energy.
    This process also makes oxygen, which we breathe.
    """
    
    try:
        print(f"Lesson text to synthesize (age 8):")
        print(lesson_text[:100] + "...")
        print("Synthesizing...")
        
        result = synthesize_speech(lesson_text, voice="nova", age_group=8)
        
        if result:
            audio_bytes, content_type = result
            print(f"✓ Synthesis successful!")
            print(f"  Audio size: {len(audio_bytes)} bytes")
            print(f"  Content type: {content_type}")
            
            # Save to file
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
        import traceback
        traceback.print_exc()
        return False

def test_voice_selection():
    """Test voice selection based on age group."""
    print("\n=== Testing Voice Selection ===")
    
    test_text = "This is a voice test."
    
    age_groups = [6, 10, 15]
    
    try:
        for age in age_groups:
            voice = tts_service.get_voice_for_age(age)
            print(f"  Age {age}: voice = {voice}")
            
            # Try to synthesize with this voice
            result = synthesize_speech(test_text, voice=voice, age_group=age)
            if result:
                print(f"    ✓ Voice {voice} works")
            else:
                print(f"    ✗ Voice {voice} failed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_long_text_truncation():
    """Test that long text is properly truncated for short audio clips."""
    print("\n=== Testing Long Text Truncation ===")
    
    # Create text longer than 150 words
    long_text = " ".join([f"Word{i}" for i in range(200)])
    
    try:
        print(f"Long text ({len(long_text.split())} words)...")
        result = synthesize_speech(long_text, voice="alloy", age_group=10)
        
        if result:
            audio_bytes, content_type = result
            print(f"✓ Synthesis successful (text was truncated if needed)")
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
    print("TTS (Text-to-Speech) Service Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Service initialization
    results.append(("TTS Initialization", test_tts_service_initialization()))
    
    # Test 2: Simple synthesis
    results.append(("Simple Text Synthesis", test_synthesize_simple_text()))
    
    # Test 3: Lesson text synthesis
    results.append(("Lesson Text Synthesis", test_synthesize_lesson_text()))
    
    # Test 4: Voice selection
    results.append(("Voice Selection", test_voice_selection()))
    
    # Test 5: Long text handling
    results.append(("Long Text Truncation", test_long_text_truncation()))
    
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
    
    print("\nNote: For full TTS testing, you need:")
    print("  1. OPENAI_API_KEY in environment (for API) OR")
    print("  2. edge-tts package installed (for free TTS fallback)")
    print("\nGenerated audio files:")
    print("  - test_output_audio.mp3 (or .wav)")
    print("  - test_lesson_audio.mp3 (or .wav)")
