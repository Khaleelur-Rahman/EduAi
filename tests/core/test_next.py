"""
Test script for /next functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.db import get_db, create_user, get_user_by_phone, get_current_lesson, create_tables
from app.handlers import process_whatsapp_message
from app.rag import initialize_rag
from app.llm import initialize_llm

def test_next_functionality():
    print("Testing /next Functionality")
    print("=" * 60)
    
    try:
        print("1. Creating database tables...")
        create_tables()
        print("Database tables created")
        
        print("2. Initializing services...")
        initialize_rag()
        initialize_llm()
        print("Services initialized\n")
        
        db = next(get_db())
        
        test_phone = "+1234567890"
        user = get_user_by_phone(db, test_phone)
        if not user:
            user = create_user(db, test_phone)
            user.name = "Test Student"
            user.age = 10
            user.country = "Test Country"
            user.is_onboarded = True
            user.onboarding_step = "completed"
            db.commit()
            print(f"Created test user: {user.name} (age {user.age})")
        else:
            print(f"Using existing test user: {user.name} (age {user.age})")
        
        print("\n3. Testing /next functionality...")
        
        # Step 1: Start a lesson
        print("\nStep 1: Starting lesson on 'photosynthesis'")
        response1 = process_whatsapp_message(db, test_phone, "/lesson photosynthesis")
        print(f"Response 1: {response1[:200]}...")
        
        # Check if lesson was created
        current_lesson = get_current_lesson(db, user.id)
        if current_lesson:
            print(f"Lesson created: {current_lesson.topic} - Step {current_lesson.lesson_step} - Chunk: {current_lesson.chunk_id}")
        else:
            print("No lesson created!")
            return
        
        # Step 2: Use /next command
        print("\nStep 2: Using /next command")
        response2 = process_whatsapp_message(db, test_phone, "/next")
        print(f"Response 2: {response2[:200]}...")
        
        # Check if lesson was updated
        updated_lesson = get_current_lesson(db, user.id)
        if updated_lesson:
            print(f"Lesson updated: {updated_lesson.topic} - Step {updated_lesson.lesson_step} - Chunk: {updated_lesson.chunk_id}")
            
            # Check if content actually changed
            if response1 != response2:
                print("Content changed between parts!")
            else:
                print("Content is the same - this is the issue!")
        else:
            print("No lesson found after /next!")
        
        # Step 3: Use /next again
        print("\nStep 3: Using /next command again")
        response3 = process_whatsapp_message(db, test_phone, "/next")
        print(f"Response 3: {response3[:200]}...")
        
        # Check if lesson was updated again
        final_lesson = get_current_lesson(db, user.id)
        if final_lesson:
            print(f"Lesson updated again: {final_lesson.topic} - Step {final_lesson.lesson_step} - Chunk: {final_lesson.chunk_id}")
            
            # Check if content changed
            if response2 != response3:
                print("Content changed between parts 2 and 3!")
            else:
                print("Content is the same between parts 2 and 3!")
        else:
            print("No lesson found after second /next!")
        
        print("\n4. Testing /next without active lesson...")
        if final_lesson:
            final_lesson.completed = True
            db.commit()
        
        response4 = process_whatsapp_message(db, test_phone, "/next")
        print(f"Response 4: {response4}")
        
        print("\n/next functionality test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test_next_functionality()
