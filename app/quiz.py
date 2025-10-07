import json
import logging
import re
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session

from .db import create_quiz_progress, get_current_quiz, update_quiz_progress, get_current_lesson
from .rag import rag_service
from .llm import llm_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuizGenerator:
    """Generates quizzes based on RAG content chunks"""
    
    def __init__(self):
        self.llm_service = llm_service
    
    def generate_quiz_from_chunk(self, topic: str, chunk_content: str, age_group: int, 
                                user_name: str = "") -> List[Dict[str, Any]]:
        """Generate quiz questions from a content chunk"""
        
        if not self.llm_service._initialized:
            self.llm_service.initialize()
        
        # Age-appropriate quiz style
        if age_group <= 8:
            style_guide = "Use very simple questions with 3 options each. Focus on basic facts and fun elements."
        elif age_group <= 10:
            style_guide = "Use clear questions with 3-4 options. Mix multiple choice and true/false questions."
        else:
            style_guide = "Use more detailed questions with 4 options. Include some challenging concepts."
        
        system_prompt = f"""You are an expert quiz creator for children aged {age_group} years old.
Create a quiz based on the provided educational content.

Instructions:
- Generate exactly 3 questions
- Use multiple choice questions (3-4 options each)
- Include 1 true/false question if appropriate
- Make questions directly related to the content provided
- Ensure answers are clear and unambiguous
- Use age-appropriate language: {style_guide}

Format your response as a JSON array with this structure:
[
  {{
    "question": "What is the main pigment in photosynthesis?",
    "type": "multiple_choice",
    "options": ["Chlorophyll", "Hemoglobin", "Keratin", "Myosin"],
    "correct_answer": "A",
    "explanation": "Chlorophyll is the green pigment that captures light energy."
  }},
  {{
    "question": "True or False: Photosynthesis produces oxygen.",
    "type": "true_false",
    "options": ["True", "False"],
    "correct_answer": "True",
    "explanation": "Photosynthesis releases oxygen as a byproduct."
  }}
]

Important: Only respond with valid JSON, no additional text."""

        user_prompt = f"""Create a quiz about {topic} based on this content:

{chunk_content}

Make sure the questions test understanding of the key concepts in this content."""

        try:
            response = self.llm_service.client.chat.completions.create(
                model=self.llm_service.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            quiz_json = response.choices[0].message.content.strip()
            
            # Clean up the response to extract JSON
            quiz_json = self._extract_json_from_response(quiz_json)
            
            # Parse and validate the quiz
            questions = json.loads(quiz_json)
            
            # Validate quiz structure
            if not isinstance(questions, list) or len(questions) != 3:
                raise ValueError("Invalid quiz format")
            
            for q in questions:
                required_fields = ["question", "type", "options", "correct_answer", "explanation"]
                if not all(field in q for field in required_fields):
                    raise ValueError("Missing required fields in question")
            
            logger.info(f"Generated quiz with {len(questions)} questions for topic: {topic}")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to generate quiz: {str(e)}")
            # Return fallback quiz
            return self._get_fallback_quiz(topic, age_group)
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response, handling common formatting issues"""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Find JSON array
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        
        if start_idx != -1 and end_idx != -1:
            return response[start_idx:end_idx]
        
        return response.strip()
    
    def _get_fallback_quiz(self, topic: str, age_group: int) -> List[Dict[str, Any]]:
        """Fallback quiz when LLM generation fails"""
        return [
            {
                "question": f"What is {topic}?",
                "type": "multiple_choice",
                "options": ["A scientific concept", "A type of animal", "A planet", "A color"],
                "correct_answer": "A",
                "explanation": f"{topic} is an important scientific concept you've been learning about."
            },
            {
                "question": f"True or False: {topic} is important to understand.",
                "type": "true_false",
                "options": ["True", "False"],
                "correct_answer": "True",
                "explanation": f"Yes, understanding {topic} helps us learn more about science."
            },
            {
                "question": f"Where can you find information about {topic}?",
                "type": "multiple_choice",
                "options": ["In books", "In your lesson", "On the internet", "All of the above"],
                "correct_answer": "D",
                "explanation": "You can learn about this topic from many different sources."
            }
        ]
    
    def format_quiz_for_whatsapp(self, questions: List[Dict[str, Any]], topic: str, lesson_step: int) -> str:
        """Format quiz questions for WhatsApp display"""
        quiz_text = f"🧩 *Quiz for {topic.title()} - Part {lesson_step}*\n\n"
        
        for i, q in enumerate(questions, 1):
            quiz_text += f"*Q{i}: {q['question']}*\n"
            
            if q['type'] == 'multiple_choice':
                for j, option in enumerate(q['options']):
                    letter = chr(65 + j)  # A, B, C, D
                    quiz_text += f"{letter}) {option}\n"
            elif q['type'] == 'true_false':
                quiz_text += "A) True\nB) False\n"
            
            quiz_text += "\n"
        
        quiz_text += "_Reply with your answers like: 1A, 2B, 3True_"
        return quiz_text
    
    def check_answers(self, questions: List[Dict[str, Any]], user_answers: str) -> Tuple[int, str]:
        """Check user answers and provide feedback"""
        try:
            # Parse user answers (format: "1A, 2B, 3True")
            answer_pairs = []
            for pair in user_answers.split(','):
                pair = pair.strip()
                if re.match(r'\d+[A-D]', pair) or re.match(r'\d+(True|False)', pair, re.IGNORECASE):
                    answer_pairs.append(pair)
            
            if len(answer_pairs) != len(questions):
                return 0, "Please provide answers for all questions in the format: 1A, 2B, 3True"
            
            correct_count = 0
            feedback = "📝 *Quiz Results:*\n\n"
            
            for i, (question, answer_pair) in enumerate(zip(questions, answer_pairs)):
                q_num = i + 1
                user_answer = answer_pair[1:].upper()  # Remove question number
                
                # Normalize answer format
                if user_answer in ['TRUE', 'FALSE']:
                    user_answer = user_answer.capitalize()
                elif user_answer in ['A', 'B', 'C', 'D']:
                    user_answer = user_answer
                else:
                    user_answer = user_answer.upper()
                
                correct_answer = question['correct_answer']
                is_correct = user_answer == correct_answer
                
                if is_correct:
                    correct_count += 1
                    feedback += f"✅ *Q{q_num} correct!* {question['explanation']}\n\n"
                else:
                    feedback += f"❌ *Q{q_num} wrong.* Correct answer: {correct_answer}. {question['explanation']}\n\n"
            
            score_text = f"🎯 *Score: {correct_count}/{len(questions)}*"
            if correct_count == len(questions):
                score_text += " 🎉 Perfect!"
            elif correct_count >= len(questions) * 0.7:
                score_text += " 👍 Great job!"
            else:
                score_text += " 💪 Keep studying!"
            
            feedback += score_text
            feedback += "\n\n_Type `/lesson <topic>` for something new or `/quiz` for a quiz related to this topic!_"
            return correct_count, feedback
            
        except Exception as e:
            logger.error(f"Error checking answers: {str(e)}")
            return 0, "Sorry, I had trouble checking your answers. Please try again!"

# Global quiz generator instance
quiz_generator = QuizGenerator()

def create_quiz_from_lesson(db: Session, user_id: int, topic: str, age_group: int, 
                           user_name: str = "") -> Tuple[str, int]:
    """Create a quiz based on the current lesson"""
    try:
        # Get current lesson
        current_lesson = get_current_lesson(db, user_id)
        if not current_lesson:
            return "You don't have any lessons in progress. Start a lesson with `/lesson <topic>` first! 📚", 0
        
        if not current_lesson.is_rag_lesson or not current_lesson.chunk_id:
            return "Quizzes are only available for science lessons. Try a science topic! 🔬", 0
        
        # Get the content chunk from RAG
        chunks = rag_service.retrieve_relevant_chunks(topic, age_group)
        if not chunks:
            return f"Sorry, I couldn't find content for {topic} to create a quiz. Try a different topic! 🔬", 0
        
        # Find the specific chunk used in the lesson
        target_chunk = None
        for chunk in chunks:
            if chunk['chunk_id'] == current_lesson.chunk_id:
                target_chunk = chunk
                break
        
        if not target_chunk:
            # Use the first chunk if specific chunk not found
            target_chunk = chunks[0]
        
        # Generate quiz questions
        questions = quiz_generator.generate_quiz_from_chunk(
            topic, target_chunk['content'], age_group, user_name
        )
        
        # Store quiz in database
        quiz = create_quiz_progress(
            db=db,
            user_id=user_id,
            lesson_id=current_lesson.id,
            topic=topic,
            lesson_step=current_lesson.lesson_step,
            chunk_id=current_lesson.chunk_id,
            questions=json.dumps(questions)
        )
        
        # Format quiz for display
        quiz_text = quiz_generator.format_quiz_for_whatsapp(questions, topic, current_lesson.lesson_step)
        
        logger.info(f"Created quiz for user {user_id} on topic {topic}")
        return quiz_text, quiz.id
        
    except Exception as e:
        logger.error(f"Failed to create quiz: {str(e)}")
        return f"Sorry, I had trouble creating a quiz. Please try again! 🔬", 0

def check_quiz_answers(db: Session, user_id: int, user_answers: str) -> str:
    """Check quiz answers and provide feedback"""
    try:
        # Get current quiz
        current_quiz = get_current_quiz(db, user_id)
        if not current_quiz:
            return "You don't have any active quiz. Start a lesson and use `/quiz` to create one! 🧩"
        
        # Parse questions from database
        questions = json.loads(current_quiz.questions)
        
        # Check answers
        correct_count, feedback = quiz_generator.check_answers(questions, user_answers)
        
        # Update quiz progress
        update_quiz_progress(
            db=db,
            quiz=current_quiz,
            user_answers=user_answers,
            score=correct_count,
            completed=True
        )
        
        logger.info(f"Quiz completed for user {user_id}, score: {correct_count}/{len(questions)}")
        return feedback
        
    except Exception as e:
        logger.error(f"Failed to check quiz answers: {str(e)}")
        return "Sorry, I had trouble checking your answers. Please try again! 🔬"
