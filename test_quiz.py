#!/usr/bin/env python3
"""
Test script for quiz functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.db import get_db, create_user, get_user_by_phone, get_current_lesson, get_user_quizzes, create_tables
from app.handlers import process_whatsapp_message
from app.rag import initialize_rag
from app.llm import initialize_llm

def test_quiz_functionality():
    print("🧩 Testing Quiz Functionality")
    print("=" * 60)
    
    try:
        # Initialize services
        print("1. Creating database tables...")
        create_tables()
        print("✅ Database tables created")
        
        print("2. Initializing services...")
        initialize_rag()
        initialize_llm()
        print("✅ Services initialized\n")
        
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
            print(f"✅ Created test user: {user.name} (age {user.age})")
        else:
            print(f"✅ Using existing test user: {user.name} (age {user.age})")
        
        print("\n3. Testing complete quiz flow...")
        
        # Step 1: Start a lesson
        print("\n📚 Step 1: Starting lesson on 'molecules'")
        response1 = process_whatsapp_message(db, test_phone, "/lesson molecules")
        print(f"Response: {response1[:150]}...")
        
        # Check if lesson was created
        current_lesson = get_current_lesson(db, user.id)
        if current_lesson:
            print(f"✅ Lesson created: {current_lesson.topic} - Step {current_lesson.lesson_step}")
        else:
            print("❌ No lesson created!")
            return
        
        # Step 2: Create a quiz
        print("\n🧩 Step 2: Creating quiz")
        response2 = process_whatsapp_message(db, test_phone, "/quiz")
        print(f"Response: {response2[:200]}...")
        
        # Check if quiz was created
        from app.db import get_current_quiz
        current_quiz = get_current_quiz(db, user.id)
        if current_quiz:
            print(f"✅ Quiz created: {current_quiz.topic} - Step {current_quiz.lesson_step}")
        else:
            print("❌ No quiz created!")
            return
        
        # Step 3: Submit answers
        print("\n📝 Step 3: Submitting quiz answers")
        response3 = process_whatsapp_message(db, test_phone, "1A, 2B, 3True")
        print(f"Response: {response3[:200]}...")
        
        # Check quiz completion by looking at quiz history
        quiz_history = get_user_quizzes(db, user.id, limit=1)
        if quiz_history and quiz_history[0].completed:
            latest_quiz = quiz_history[0]
            print(f"✅ Quiz completed: Score {latest_quiz.score}")
        else:
            print("❌ Quiz not completed properly!")
            if quiz_history:
                print(f"   Quiz status: completed={quiz_history[0].completed}, score={quiz_history[0].score}")
        
        print("\n4. Testing quiz without active lesson...")
        # Complete the current lesson to test error handling
        if current_lesson:
            current_lesson.completed = True
            db.commit()
        
        response4 = process_whatsapp_message(db, test_phone, "/quiz")
        print(f"Response: {response4}")
        
        print("\n5. Testing quiz answer without active quiz...")
        response5 = process_whatsapp_message(db, test_phone, "1A, 2B, 3True")
        print(f"Response: {response5}")
        
        print("\n6. Testing user quiz history...")
        quiz_history = get_user_quizzes(db, user.id, limit=5)
        print(f"📊 User has {len(quiz_history)} quiz records:")
        for i, quiz in enumerate(quiz_history, 1):
            print(f"  {i}. {quiz.topic} - Step {quiz.lesson_step} - Score: {quiz.score} - {quiz.created_at}")
        
        print("\n✅ Quiz functionality test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test_quiz_functionality()
