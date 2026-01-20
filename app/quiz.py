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
    """Generates quizzes based on lesson content"""
    
    def __init__(self):
        self.llm_service = llm_service
    
    def generate_quiz_from_content(self, topic: str, lesson_content: str, age_group: int, 
                                user_name: str = "") -> List[Dict[str, Any]]:
        """Generate quiz questions from lesson content"""
        
        if not self.llm_service._initialized:
            self.llm_service.initialize()
        
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

        user_prompt = f"""Create a quiz about {topic} based on this lesson content:

{lesson_content}

Make sure the questions test understanding of the key concepts in this lesson content."""

        try:
            response = self.llm_service.client.chat.completions.create(
                model=self.llm_service.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=800,
                temperature=0.7,
                top_p=0.8,
                stream=True
            )
            
            quiz_json = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    quiz_json += chunk.choices[0].delta.content
            
            quiz_json = quiz_json.strip()
            
            quiz_json = self._extract_json_from_response(quiz_json)
            
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
            return self._get_fallback_quiz(topic, age_group)
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response, handling common formatting issues"""
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
        from .utils import clean_topic_title
        quiz_text = f"🧩 *Quiz for {clean_topic_title(topic)} - Part {lesson_step}*\n\n"
        
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
            # Parse answers and map them to question numbers
            answer_dict = {}  # Maps question number (int) to answer string
            # Use regex to find all answer patterns regardless of separator (comma, space, newline)
            # Pattern matches: 1A, 2B, 3True, 3False, 3T, 3F, etc.
            pairs = re.findall(r'\d+[A-DTtFf]|\d+(?:True|False)', user_answers, re.IGNORECASE)
            
            for pair in pairs:
                pair = pair.strip()
                # Extract question number and answer
                # Handle both full words (True/False) and abbreviations (T/F)
                match = re.match(r'(\d+)([A-D]|True|False|T|F)', pair, re.IGNORECASE)
                if match:
                    q_num = int(match.group(1))
                    answer = match.group(2).upper()
                    # Normalize T/F to True/False for consistency
                    if answer == 'T':
                        answer = 'TRUE'
                    elif answer == 'F':
                        answer = 'FALSE'
                    answer_dict[q_num] = answer
            
            if len(answer_dict) != len(questions):
                return 0, "Please provide answers for all questions in the format: 1A, 2B, 3True"
            
            logger.info(f"Parsed answers: {answer_dict} for {len(questions)} questions")
            
            correct_count = 0
            feedback = "📝 *Quiz Results:*\n\n"
            
            # Match answers to questions by question number
            for i, question in enumerate(questions):
                q_num = i + 1
                
                # Get the user's answer for this question number
                if q_num not in answer_dict:
                    feedback += f"❌ *Q{q_num}:* No answer provided.\n\n"
                    continue
                
                user_answer = answer_dict[q_num]
                
                # Log for debugging
                logger.debug(f"Q{q_num}: User answer={user_answer}, Question={question.get('question', '')[:50]}...")

                q_type = question.get('type', 'multiple_choice')
                options = [str(o) for o in question.get('options', [])]
                correct_answer_raw = str(question.get('correct_answer'))

                # Normalize True/False handling to accept either label (A/B) or text (True/False)
                if q_type == 'true_false':
                    # Map user answer to canonical 'True'/'False'
                    if user_answer in ['A', 'TRUE']:
                        user_answer_norm = 'True'
                    elif user_answer in ['B', 'FALSE']:
                        user_answer_norm = 'False'
                    else:
                        user_answer_norm = user_answer.capitalize()

                    # Determine correct answer canonical value
                    if correct_answer_raw.upper() in ['A', 'B'] and len(options) >= 2:
                        correct_value = options[0] if correct_answer_raw.upper() == 'A' else options[1]
                    else:
                        correct_value = correct_answer_raw
                    correct_value_norm = 'True' if str(correct_value).strip().lower() == 'true' else 'False'

                    is_correct = (user_answer_norm == correct_value_norm)
                    correct_display = correct_value_norm
                else:
                    # Multiple choice: compare by letter. If ground truth is text, map it to its letter.
                    if correct_answer_raw.upper() in ['A', 'B', 'C', 'D']:
                        correct_letter = correct_answer_raw.upper()
                    else:
                        # Try to find which option matches the provided text
                        correct_letter = None
                        for idx, opt in enumerate(options):
                            if str(opt).strip().lower() == correct_answer_raw.strip().lower():
                                correct_letter = chr(65 + idx)  # A/B/C/D
                                break
                        # Fallback to A if unknown to avoid crash
                        if not correct_letter:
                            correct_letter = 'A'
                    is_correct = (user_answer == correct_letter)
                    correct_display = correct_letter
                
                if is_correct:
                    correct_count += 1
                    feedback += f"✅ *Q{q_num} correct!* {question['explanation']}\n\n"
                else:
                    feedback += f"❌ *Q{q_num} wrong.* Correct answer: {correct_display}. {question['explanation']}\n\n"
            
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

quiz_generator = QuizGenerator()

def create_quiz_from_lesson(db: Session, user_id: int, topic: str, age_group: int, 
                           user_name: str = "") -> Tuple[str, int]:
    """Create a quiz based on the current lesson content"""
    try:
        current_lesson = get_current_lesson(db, user_id)
        if not current_lesson:
            return "You don't have any lessons in progress. Start a lesson with `/lesson <topic>` first! 📚", 0
        
        # Generate quiz questions from the actual lesson content sent to WhatsApp
        # This ensures questions align with what the student actually learned
        questions = quiz_generator.generate_quiz_from_content(
            topic, current_lesson.lesson_content, age_group, user_name
        )

        logger.info(f"current lesson: {current_lesson}")
        
        quiz = create_quiz_progress(
            db=db,
            user_id=user_id,
            lesson_id=current_lesson.id,
            topic=topic,
            lesson_step=current_lesson.lesson_step,
            chunk_id=current_lesson.chunk_id,  # Use the lesson's chunk_id (can be None for non-RAG lessons)
            questions=json.dumps(questions)
        )
        
        quiz_text = quiz_generator.format_quiz_for_whatsapp(questions, topic, current_lesson.lesson_step)
        
        logger.info(f"Created quiz for user {user_id} on topic {topic}")
        return quiz_text, quiz.id
        
    except Exception as e:
        logger.error(f"Failed to create quiz: {str(e)}")
        return f"Sorry, I had trouble creating a quiz. Please try again! 🔬", 0

def check_quiz_answers(db: Session, user_id: int, user_answers: str) -> str:
    """Check quiz answers and provide feedback"""
    try:
        current_quiz = get_current_quiz(db, user_id)
        if not current_quiz:
            return "You don't have any active quiz. Start a lesson and use `/quiz` to create one! 🧩"
        
        questions = json.loads(current_quiz.questions)
        
        correct_count, feedback = quiz_generator.check_answers(questions, user_answers)
        
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
