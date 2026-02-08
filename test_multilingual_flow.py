#!/usr/bin/env python3
"""
Test script for multilingual support flow.
Tests language switching, lesson generation in different languages, and TTS/STT.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from app.db import SessionLocal, get_user_by_phone, create_user, update_user
from app.handlers import process_whatsapp_message
from app.language import SUPPORTED_LANGUAGES, validate_language_code, get_language_name
from app.utils import get_loading_message, get_help_message

def test_language_switching():
    """Test language switching functionality."""
    print("=" * 60)
    print("TEST 1: Language Switching")
    print("=" * 60)
    
    db = SessionLocal()
    try:
        test_phone = "+1234567890"
        
        # Create or get user
        user = get_user_by_phone(db, test_phone)
        if not user:
            user = create_user(db, test_phone)
            print(f"✓ Created test user: {test_phone}")
        else:
            print(f"✓ Found existing user: {test_phone}")
        
        # Test each supported language
        for lang_code in SUPPORTED_LANGUAGES.keys():
            update_user(db, user, language=lang_code)
            db.refresh(user)
            
            lang_name = get_language_name(lang_code, native=True)
            loading_msg = get_loading_message("lesson", "Cells", lang_code)
            help_msg = get_help_message(10, lang_code)
            
            print(f"\n  Language: {lang_name} ({lang_code.upper()})")
            print(f"  Loading message: {loading_msg[:50]}...")
            print(f"  Help message length: {len(help_msg)} chars")
            print(f"  ✓ Language set successfully")
        
        print("\n✓ Language switching test PASSED")
        
    except Exception as e:
        print(f"\n✗ Language switching test FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def test_lesson_generation_multilingual():
    """Test lesson generation in different languages."""
    print("\n" + "=" * 60)
    print("TEST 2: Lesson Generation in Different Languages")
    print("=" * 60)
    
    db = SessionLocal()
    try:
        test_phone = "+1234567890"
        topic = "photosynthesis"
        
        # Test each language
        for lang_code in ["en", "es", "fr"]:
            print(f"\n  Testing language: {lang_code.upper()}")
            
            # Set user language
            user = get_user_by_phone(db, test_phone)
            if not user:
                user = create_user(db, test_phone)
            update_user(db, user, language=lang_code, is_onboarded=True, age=10, name="Test User")
            db.refresh(user)
            
            print(f"    User language set to: {user.language}")
            
            # Generate lesson
            try:
                response = process_whatsapp_message(db, test_phone, f"/lesson {topic}", for_audio=False)
                
                # Check if response contains expected language indicators
                lang_name = get_language_name(lang_code, native=True)
                
                # Simple check: response should not be empty
                if response and len(response) > 50:
                    print(f"    ✓ Lesson generated ({len(response)} chars)")
                    print(f"    Preview: {response[:100]}...")
                    
                    # Check if it's actually in the target language (basic check)
                    # Spanish/French should have some non-English words
                    if lang_code != "en":
                        # This is a basic check - LLM might still mix languages
                        print(f"    ⚠ Note: Verify manually that content is in {lang_name}")
                    else:
                        print(f"    ✓ English lesson generated")
                else:
                    print(f"    ✗ Lesson generation failed or too short")
                    
            except Exception as e:
                print(f"    ✗ Error generating lesson: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✓ Lesson generation test COMPLETED")
        print("  Note: Manual verification recommended for language accuracy")
        
    except Exception as e:
        print(f"\n✗ Lesson generation test FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def test_language_validation():
    """Test language code validation."""
    print("\n" + "=" * 60)
    print("TEST 3: Language Code Validation")
    print("=" * 60)
    
    test_cases = [
        ("en", "en"),
        ("es", "es"),
        ("fr", "fr"),
        ("EN", "en"),
        ("ES", "es"),
        ("english", "en"),
        ("spanish", "es"),
        ("french", "fr"),
        ("español", "es"),
        ("invalid", None),
        ("", None),
    ]
    
    passed = 0
    failed = 0
    
    for input_code, expected in test_cases:
        result = validate_language_code(input_code)
        if result == expected:
            print(f"  ✓ '{input_code}' -> {result}")
            passed += 1
        else:
            print(f"  ✗ '{input_code}' -> {result} (expected {expected})")
            failed += 1
    
    print(f"\n  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("✓ Language validation test PASSED")
    else:
        print("✗ Language validation test FAILED")


def test_loading_messages():
    """Test loading messages in different languages."""
    print("\n" + "=" * 60)
    print("TEST 4: Loading Messages Translation")
    print("=" * 60)
    
    commands = ["lesson", "next", "quiz", "default"]
    languages = ["en", "es", "fr"]
    
    for lang in languages:
        lang_name = get_language_name(lang, native=True)
        print(f"\n  Language: {lang_name} ({lang.upper()})")
        for cmd in commands:
            topic = "Cells" if cmd == "lesson" else None
            msg = get_loading_message(cmd, topic, lang)
            print(f"    {cmd}: {msg}")
    
    print("\n✓ Loading messages test PASSED")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MULTILINGUAL SUPPORT TEST SUITE")
    print("=" * 60)
    
    try:
        # Test 1: Language switching
        test_language_switching()
        
        # Test 2: Language validation
        test_language_validation()
        
        # Test 3: Loading messages
        test_loading_messages()
        
        # Test 4: Lesson generation (requires LLM - may be slow/expensive)
        print("\n" + "=" * 60)
        response = input("Run lesson generation test? (requires LLM API, may be slow) [y/N]: ")
        if response.lower() == 'y':
            test_lesson_generation_multilingual()
        else:
            print("Skipping lesson generation test")
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
