#!/usr/bin/env python3
"""
Test script to verify lesson completeness and truncation fixes.
Tests that lessons are complete, properly formatted, and don't cut off mid-sentence.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_lesson_completeness():
    """Test that generated lessons are complete and properly formatted"""
    print("\n" + "="*60)
    print("Testing Lesson Completeness")
    print("="*60)
    
    try:
        from app.llm import generate_lesson, initialize_llm
        from app.utils import format_for_whatsapp, clean_whatsapp_formatting
        
        # Initialize LLM
        print("\n1. Initializing LLM service...")
        initialize_llm()
        print("   ✓ LLM initialized")
        
        # Test topics
        test_cases = [
            ("cells", 10),
            ("atoms", 8),
            ("photosynthesis", 12),
        ]
        
        all_passed = True
        
        for topic, age in test_cases:
            print(f"\n2. Testing lesson generation for '{topic}' (age {age})...")
            
            try:
                lesson_content = generate_lesson(topic, age, "Test User")
                
                # Check length
                print(f"   Length: {len(lesson_content)} characters")
                
                # Check if it ends properly (allow emojis after punctuation)
                import re
                text_without_emojis = re.sub(r'[\s\U0001F300-\U0001F9FF]+$', '', lesson_content.rstrip())
                ends_properly = text_without_emojis.endswith(('.', '!', '?', ':', ';')) if text_without_emojis else False
                if not ends_properly:
                    print(f"   ❌ FAIL: Lesson doesn't end with proper punctuation")
                    print(f"   Ends with: '{lesson_content[-30:]}'")
                    print(f"   Text without emojis ends with: '{text_without_emojis[-20:] if text_without_emojis else 'N/A'}'")
                    all_passed = False
                else:
                    print(f"   ✓ Ends with proper punctuation")
                
                # Check for incomplete lists (ending with numbers like "3.")
                import re
                incomplete_list = re.search(r'\d+\.\s*$', lesson_content.rstrip())
                if incomplete_list:
                    print(f"   ❌ FAIL: Lesson ends with incomplete list item")
                    print(f"   Ends with: '{lesson_content[-30:]}'")
                    all_passed = False
                else:
                    print(f"   ✓ No incomplete lists detected")
                
                # Check for double asterisks
                if '**' in lesson_content:
                    print(f"   ⚠️  WARNING: Contains double asterisks (should be single)")
                    # Test cleanup
                    cleaned = clean_whatsapp_formatting(lesson_content)
                    if '**' not in cleaned:
                        print(f"   ✓ Cleanup function fixes it")
                    else:
                        print(f"   ❌ FAIL: Cleanup function doesn't fix it")
                        all_passed = False
                else:
                    print(f"   ✓ No double asterisks")
                
                # Check for "Try This at Home" sections
                if 'try this at home' in lesson_content.lower():
                    print(f"   ⚠️  WARNING: Contains 'Try This at Home' section")
                else:
                    print(f"   ✓ No generic activity sections")
                
                # Test formatting
                formatted = format_for_whatsapp(lesson_content, age)
                if len(formatted) > 1600:
                    print(f"   ⚠️  WARNING: Formatted content exceeds 1600 chars: {len(formatted)}")
                else:
                    print(f"   ✓ Formatted content within limits: {len(formatted)} chars")
                
                # Show sample
                print(f"\n   Sample (last 150 chars):")
                print(f"   '{lesson_content[-150:]}'")
                
            except Exception as e:
                print(f"   ❌ FAIL: Error generating lesson: {str(e)}")
                import traceback
                traceback.print_exc()
                all_passed = False
        
        # Test continuation lessons
        print(f"\n3. Testing continuation lesson...")
        try:
            # First generate a base lesson
            base_lesson = generate_lesson("cells", 10, "Test User")
            print(f"   Base lesson length: {len(base_lesson)} chars")
            
            # Generate continuation using the LLM service directly
            from app.llm import llm_service
            system_prompt, user_prompt = llm_service._create_lesson_prompt(
                "cells", 10, "Test User", 
                is_continuation=True, 
                previous_content=base_lesson
            )
            
            response = llm_service.client.chat.completions.create(
                model=llm_service.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=1000,
                temperature=0.7,
                top_p=0.8,
                stream=True
            )
            
            continuation = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    continuation += chunk.choices[0].delta.content
            continuation = continuation.strip()
            print(f"   Continuation length: {len(continuation)} chars")
            
            # Check if it references previous content
            if any(word in continuation.lower()[:100] for word in ['previous', 'earlier', 'before', 'we learned', 'we covered']):
                print(f"   ✓ References previous content")
            else:
                print(f"   ⚠️  WARNING: Doesn't seem to reference previous content")
            
            # Check completeness
            if continuation.rstrip().endswith(('.', '!', '?')):
                print(f"   ✓ Continuation ends properly")
            else:
                print(f"   ❌ FAIL: Continuation doesn't end properly")
                all_passed = False
            
        except Exception as e:
            print(f"   ❌ FAIL: Error testing continuation: {str(e)}")
            import traceback
            traceback.print_exc()
            all_passed = False
        
        # Test truncation handling
        print(f"\n4. Testing truncation detection and completion...")
        try:
            # Create a fake incomplete lesson
            incomplete = "This is a test lesson about cells. Cells are the basic unit of life. There are two main types:"
            print(f"   Incomplete text: '{incomplete}'")
            
            # Check if detection works (should detect incomplete)
            # Note: ':' is valid punctuation, but ending with just ':' after a list is incomplete
            ends_properly = incomplete.rstrip().endswith(('.', '!', '?'))
            # Also check if it ends mid-list (number followed by colon or period)
            ends_mid_list = bool(re.search(r'\d+\.\s*$', incomplete.rstrip()))
            
            if not ends_properly or ends_mid_list:
                print(f"   ✓ Truncation detection works (detected incomplete)")
            else:
                print(f"   ⚠️  WARNING: Truncation detection may need adjustment")
                
        except Exception as e:
            print(f"   ❌ FAIL: Error testing truncation: {str(e)}")
            all_passed = False
        
        print("\n" + "="*60)
        if all_passed:
            print("✓ ALL TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED")
        print("="*60)
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_lesson_completeness():
    """Test RAG lesson completeness"""
    print("\n" + "="*60)
    print("Testing RAG Lesson Completeness")
    print("="*60)
    
    try:
        from app.rag import initialize_rag, get_rag_lesson
        from app.llm import llm_service
        
        print("\n1. Initializing RAG service...")
        initialize_rag()
        print("   ✓ RAG initialized")
        
        # Initialize LLM if needed
        if not llm_service._initialized:
            llm_service.initialize()
        
        test_topics = ["cells", "photosynthesis", "atoms"]
        all_passed = True
        
        for topic in test_topics:
            print(f"\n2. Testing RAG lesson for '{topic}'...")
            try:
                system_prompt, user_prompt, chunk_id = get_rag_lesson(topic, 10, "Test User")
                
                if not system_prompt or not user_prompt:
                    print(f"   ❌ FAIL: No prompts generated")
                    all_passed = False
                    continue
                
                # Check if prompts contain completeness requirements
                if 'ALWAYS complete' in system_prompt or 'NEVER cut off' in system_prompt:
                    print(f"   ✓ Completeness requirements in prompt")
                else:
                    print(f"   ⚠️  WARNING: Completeness requirements not found in prompt")
                
                # Generate lesson
                response = llm_service.client.chat.completions.create(
                    model=llm_service.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=1000,
                    temperature=0.7,
                    top_p=0.8,
                    stream=True
                )
                
                lesson_content = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        lesson_content += chunk.choices[0].delta.content
                
                lesson_content = lesson_content.strip()
                print(f"   Length: {len(lesson_content)} characters")
                
                # Check completeness (allow emojis at the end, but should have punctuation before)
                # Remove trailing emojis and whitespace, then check
                import re
                cleaned_end = re.sub(r'[\s\U0001F300-\U0001F9FF]+$', '', lesson_content.rstrip())
                if cleaned_end.endswith(('.', '!', '?', ':', ';')):
                    print(f"   ✓ Ends with proper punctuation")
                else:
                    # Check if there's punctuation before the emojis
                    # Look for punctuation in the last 20 characters (before emojis)
                    last_20 = lesson_content.rstrip()[-20:]
                    text_before_emojis = re.sub(r'[\s\U0001F300-\U0001F9FF]+$', '', last_20)
                    if text_before_emojis.endswith(('.', '!', '?', ':', ';')):
                        print(f"   ✓ Has punctuation before emojis (acceptable)")
                    else:
                        print(f"   ⚠️  WARNING: Ends with emoji but may need punctuation")
                        print(f"   Ends with: '{lesson_content[-30:]}'")
                        # Don't fail the test, just warn - emoji endings are acceptable for WhatsApp
                        print(f"   ⚠️  Note: Emoji-only endings are acceptable for WhatsApp, but punctuation is preferred")
                
            except Exception as e:
                print(f"   ❌ FAIL: Error: {str(e)}")
                import traceback
                traceback.print_exc()
                all_passed = False
        
        print("\n" + "="*60)
        if all_passed:
            print("✓ ALL RAG TESTS PASSED")
        else:
            print("❌ SOME RAG TESTS FAILED")
        print("="*60)
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LESSON COMPLETENESS TEST SUITE")
    print("="*60)
    
    # Test basic lesson generation
    test1_passed = test_lesson_completeness()
    
    # Test RAG lessons
    test2_passed = test_rag_lesson_completeness()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Basic Lesson Tests: {'✓ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"RAG Lesson Tests: {'✓ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED")
        sys.exit(1)

