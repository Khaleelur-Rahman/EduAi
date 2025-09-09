import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from llm import generate_lesson, initialize_llm

def test_llm_with_different_ages():
    print("Testing LLM with different age groups and topics")
    print("=" * 60)
    
    test_cases = [
        {"age": 6, "topic": "colors", "name": "Emma"},
        {"age": 10, "topic": "multiplication", "name": "Alex"},
        {"age": 14, "topic": "photosynthesis", "name": "Sam"},
        {"age": 18, "topic": "calculus derivatives", "name": "Jordan"}
    ]
    
    try:
        print("Initializing LLM service...")
        initialize_llm()
        print("✅ LLM service initialized successfully\n")
        
        for i, test_case in enumerate(test_cases, 1):
            age = test_case["age"]
            topic = test_case["topic"]
            name = test_case["name"]
            
            print(f"Test {i}: Age {age} - Topic: '{topic}' - Student: {name}")
            print("-" * 50)
            
            try:
                lesson = generate_lesson(topic, age, name)
                print(f"✅ Generated lesson ({len(lesson)} characters):")
                print(lesson)
                print("\n" + "=" * 60 + "\n")
                
            except Exception as e:
                print(f"❌ Error generating lesson: {e}")
                print("\n" + "=" * 60 + "\n")
    
    except Exception as e:
        print(f"❌ Failed to initialize LLM service: {e}")
        print("Using fallback lessons only...")
        
        for i, test_case in enumerate(test_cases, 1):
            age = test_case["age"]
            topic = test_case["topic"]
            name = test_case["name"]
            
            print(f"Test {i} (Fallback): Age {age} - Topic: '{topic}' - Student: {name}")
            print("-" * 50)
            
            try:
                lesson = generate_lesson(topic, age, name)
                print(f"✅ Generated fallback lesson ({len(lesson)} characters):")
                print(lesson)
                print("\n" + "=" * 60 + "\n")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    test_llm_with_different_ages()
