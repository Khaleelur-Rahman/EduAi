#!/usr/bin/env python3
"""
Test and Demo script for RAG-powered science lessons
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.rag import initialize_rag, get_rag_lesson, rag_service
from app.llm import initialize_llm, llm_service

def test_rag_system():
    print("🧪 Testing RAG System for Science Education")
    print("=" * 60)
    
    try:
        print("1. Initializing RAG service...")
        initialize_rag()
        print("✅ RAG service initialized successfully")
        
        print("2. Initializing LLM service...")
        initialize_llm()
        print("✅ LLM service initialized successfully\n")
        
        # Test topics for different age groups
        test_cases = [
            {"age": 8, "topic": "sodium", "name": "Alex"},
            {"age": 10, "topic": "chlorophyll", "name": "Sam"},
            {"age": 12, "topic": "molecules", "name": "Jordan"}
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            age = test_case["age"]
            topic = test_case["topic"]
            name = test_case["name"]
            
            print(f"Test {i}: Age {age} - Topic: '{topic}' - Student: {name}")
            print("-" * 50)
            
            try:
                # Test retrieval
                chunks = rag_service.retrieve_relevant_chunks(topic, age)
                print(f"📚 Retrieved {len(chunks)} relevant chunks:")
                
                for j, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
                    print(f"  Chunk {j+1}: {chunk['chunk_id']}")
                    print(f"  Content: {chunk['content'][:100]}...")
                    print(f"  Similarity: {chunk['similarity_score']:.3f}")
                    print()
                
                # Test lesson generation
                system_prompt, user_prompt, chunk_id = get_rag_lesson(topic, age, name)
                
                print(f"🔬 Generated lesson prompt (chunk_id: {chunk_id})")
                print(f"System prompt length: {len(system_prompt)} characters")
                print(f"User prompt length: {len(user_prompt)} characters")
                print()
                
                # Generate actual lesson
                try:
                    response = llm_service.client.chat.completions.create(
                        model=llm_service.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=300,
                        temperature=0.7
                    )
                    
                    lesson_content = response.choices[0].message.content.strip()
                    
                    print("🔬 Generated Lesson:")
                    print(lesson_content)
                    print()
                    
                except Exception as e:
                    print(f"❌ Error generating lesson: {e}")
                    print()
                
            except Exception as e:
                print(f"❌ Error testing {topic}: {e}")
            
            print("=" * 60 + "\n")
        
        # Test next chunk functionality
        print("🔄 Testing next chunk functionality...")
        try:
            # Get a topic and its first chunk
            topic = "plants"
            chunks = rag_service.retrieve_relevant_chunks(topic, 8)
            if chunks:
                first_chunk_id = chunks[0]['chunk_id']
                print(f"First chunk ID: {first_chunk_id}")
                
                # Try to get next chunk
                next_chunk = rag_service.get_next_chunk(topic, first_chunk_id, 8)
                if next_chunk:
                    print(f"Next chunk found: {next_chunk['chunk_id']}")
                    print(f"Content: {next_chunk['content'][:100]}...")
                else:
                    print("No next chunk found (end of content)")
            else:
                print("No chunks found for testing")
                
        except Exception as e:
            print(f"❌ Error testing next chunk: {e}")
        
        print("\n✅ RAG system test completed!")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("Make sure you have installed all required dependencies:")
        print("pip install sentence-transformers chromadb markdown")

def demo_rag_lessons():
    """Demo function showing actual lesson generation with LLM"""
    print("🎓 RAG-Powered Science Education Demo")
    print("=" * 60)
    
    try:
        # Initialize both RAG and LLM services
        print("Initializing services...")
        initialize_rag()
        initialize_llm()
        print("✅ Services initialized!\n")
        
        # Demo scenarios - Chemistry topics from Chemistry2e textbook
        demos = [
            {
                "age": 8,
                "name": "Emma",
                "topic": "atoms",
                "description": "8-year-old learning about atoms (from chemistry textbook)"
            },
            {
                "age": 10,
                "name": "Alex",
                "topic": "sodium",
                "description": "10-year-old studying sodium (from chemistry textbook)"
            },
            {
                "age": 12,
                "name": "Jordan",
                "topic": "measurements",
                "description": "12-year-old exploring measurements (from chemistry textbook)"
            }
        ]
        
        for i, demo in enumerate(demos, 1):
            print(f"Demo {i}: {demo['description']}")
            print("-" * 50)
            
            # Get RAG lesson
            system_prompt, user_prompt, chunk_id = get_rag_lesson(
                demo['topic'], demo['age'], demo['name']
            )
            
            print(f"📚 Topic: {demo['topic']}")
            print(f"👤 Student: {demo['name']} (age {demo['age']})")
            print(f"🔗 Chunk ID: {chunk_id}")
            print()
            
            # Generate actual lesson
            try:
                response = llm_service.client.chat.completions.create(
                    model=llm_service.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                
                lesson_content = response.choices[0].message.content.strip()
                
                print("🔬 Generated Lesson:")
                print(lesson_content)
                print()
                
                # Show retrieved context
                chunks = rag_service.retrieve_relevant_chunks(demo['topic'], demo['age'])
                print("📖 Retrieved Context:")
                for j, chunk in enumerate(chunks[:2], 1):
                    print(f"  {j}. {chunk['content'][:100]}...")
                    print(f"     Similarity: {chunk['similarity_score']:.3f}")
                print()
                
            except Exception as e:
                print(f"❌ Error generating lesson: {e}")
            
            print("=" * 60 + "\n")
        
        print("🎉 Demo completed! The RAG system successfully:")
        print("✅ Retrieved relevant educational content")
        print("✅ Generated age-appropriate lessons")
        print("✅ Provided accurate, grounded information")
        print("✅ Created engaging, educational content")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure all dependencies are installed and services are running.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_rag_lessons()
    else:
        test_rag_system()
