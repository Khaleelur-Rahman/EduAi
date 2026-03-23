"""
Test script for Cerebras integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.llm import llm_service

def test_cerebras_integration():
    print("Testing Cerebras Integration")
    print("=" * 50)
    
    try:
        print("1. Initializing Cerebras service...")
        llm_service.initialize()
        print("Cerebras service initialized successfully")
        
        print("\n2. Testing lesson generation...")
        lesson = llm_service.generate_lesson("photosynthesis", 10, "Test Student")
        print(f"Generated lesson ({len(lesson)} characters):")
        print(f"   {lesson[:100]}...")
        
        print("\n3. Testing with different age groups...")
        lesson_young = llm_service.generate_lesson("plants", 7, "Little Student")
        print(f"Generated lesson for age 7 ({len(lesson_young)} characters):")
        print(f"   {lesson_young[:100]}...")
        
        lesson_old = llm_service.generate_lesson("atoms", 12, "Big Student")
        print(f"Generated lesson for age 12 ({len(lesson_old)} characters):")
        print(f"   {lesson_old[:100]}...")
        
        print("\nCerebras integration test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cerebras_integration()
