#!/usr/bin/env python3
"""
Test script to debug quiz functionality after multiple lesson parts
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.db import get_db, create_user, get_user_by_phone, get_current_lesson, create_tables
from app.handlers import process_whatsapp_message
from app.rag import initialize_rag
from app.llm import initialize_llm

def test_quiz_after_multiple_parts():
    print("Testing Quiz After Multiple Lesson Parts")
    print("=" * 60)
    
    try:
        # Initialize services
        print("1. Creating database tables...")
        create_tables()
        print("Database tables created")
        
        print("2. Initializing services...")
        initialize_rag()
        initialize_llm()
        print("Services initialized\n")
        
        # Get database session
        db = next(get_db())
        
        # Create a test user
        test_phone = "+1234567890"
        user = get_user_by_phone(db, test_phone)
        if not user:
            user = create_user(db, test_phone)
            # Set up user profile
            user.name = "Test Student"
            user.age = 10
            user.country = "Test Country"
            user.is_onboarded = True
            user.onboarding_step = "completed"
            db.commit()
            print(f"Created test user: {user.name} (age {user.age})")
        else:
            print(f"Using existing test user: {user.name} (age {user.age})")
        
        print("\n3. Testing quiz after multiple lesson parts...")
        
        # Step 1: Start a lesson
        print("\n📚 Step 1: Starting lesson on 'gas'")
        response1 = process_whatsapp_message(db, test_phone, "/lesson gas")
        print(f"Response 1: {response1[:150]}...")
        
        # Check lesson details
        current_lesson = get_current_lesson(db, user.id)
        if current_lesson:
            print(f"Lesson created: {current_lesson.topic} - Step {current_lesson.lesson_step} - Chunk: {current_lesson.chunk_id}")
        else:
            print("No lesson created!")
            return
        
        # Step 2: Use /next command
        print("\nStep 2: Using /next command")
        response2 = process_whatsapp_message(db, test_phone, "/next")
        print(f"Response 2: {response2[:150]}...")
        
        # Check lesson details
        current_lesson = get_current_lesson(db, user.id)
        if current_lesson:
            print(f"Lesson updated: {current_lesson.topic} - Step {current_lesson.lesson_step} - Chunk: {current_lesson.chunk_id}")
        
        # Step 3: Use /next command again
        print("\nStep 3: Using /next command again")
        response3 = process_whatsapp_message(db, test_phone, "/next")
        print(f"Response 3: {response3[:150]}...")
        
        # Check lesson details
        current_lesson = get_current_lesson(db, user.id)
        if current_lesson:
            print(f"Lesson updated: {current_lesson.topic} - Step {current_lesson.lesson_step} - Chunk: {current_lesson.chunk_id}")
        
        # Step 4: Try to create a quiz
        print("\nStep 4: Trying to create quiz after 3 parts")
        response4 = process_whatsapp_message(db, test_phone, "/quiz")
        print(f"Response 4: {response4}")
        
        # Check if quiz was created
        from app.db import get_current_quiz
        current_quiz = get_current_quiz(db, user.id)
        if current_quiz:
            print(f"Quiz created: {current_quiz.topic} - Step {current_quiz.lesson_step} - Chunk: {current_quiz.chunk_id}")
        else:
            print("No quiz created!")
        
        print("\n4. Testing quiz after 4 parts...")
        
        # Step 5: Use /next command again
        print("\nStep 5: Using /next command again")
        response5 = process_whatsapp_message(db, test_phone, "/next")
        print(f"Response 5: {response5[:150]}...")
        
        # Check lesson details
        current_lesson = get_current_lesson(db, user.id)
        if current_lesson:
            print(f"Lesson updated: {current_lesson.topic} - Step {current_lesson.lesson_step} - Chunk: {current_lesson.chunk_id}")
        
        # Step 6: Try to create a quiz after 4 parts
        print("\nStep 6: Trying to create quiz after 4 parts")
        response6 = process_whatsapp_message(db, test_phone, "/quiz")
        print(f"Response 6: {response6}")
        
        # Check if quiz was created
        current_quiz = get_current_quiz(db, user.id)
        if current_quiz:
            print(f"Quiz created: {current_quiz.topic} - Step {current_quiz.lesson_step} - Chunk: {current_quiz.chunk_id}")
        else:
            print("No quiz created!")
        
        print("\nQuiz after multiple parts test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test_quiz_after_multiple_parts()
