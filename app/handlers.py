import logging
from typing import Tuple, Optional, List, Dict
from sqlalchemy.orm import Session

from .db import User, Progress, get_user_by_phone, create_user, update_user, create_progress, get_current_lesson, update_progress, get_current_quiz
from .llm import generate_lesson
from .rag import get_rag_lesson, initialize_rag
from .quiz import create_quiz_from_lesson, check_quiz_answers
from .utils import (
    format_for_whatsapp, validate_age, validate_subjects, validate_country, 
    validate_learning_mode, get_help_message, parse_lesson_command, 
    get_greeting_emoji, store_subjects_as_json
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageHandler:
    
    def __init__(self):
        self.onboarding_steps = {
            'name': 'What should I call you? 😊',
            'age': 'How old are you? Enter a number between 6 and 12. (This helps me adjust lessons for you)',
            # 'country': 'Which country are you from?',
            # 'subjects': 'What subjects interest you? (e.g., math, science, history - separate with commas)',
            # 'learning_mode': 'Do you prefer learning through "text" or would you like "audio" lessons in the future?',
            # 'language': 'What language would you like to learn in? (Currently supporting English - just type "english" or "en")'
        }
        
        # RAG confidence threshold for determining if retrieved content is relevant
        self.rag_confidence_threshold = 0.6
    
    def process_message(self, db: Session, phone_number: str, message: str) -> str:
        try:
            user = get_user_by_phone(db, phone_number)
            is_new_user = False
            
            if not user:
                user = create_user(db, phone_number)
                is_new_user = True
                logger.info(f"New user created: {phone_number}")
            else:
                logger.info(f"Found existing user: {phone_number}, name={user.name}, step={user.onboarding_step}")
            
            if not user.is_onboarded:
                return self._handle_onboarding(db, user, message, is_new_user)
            else:
                return self._handle_regular_message(db, user, message)
                
        except Exception as e:
            logger.error(f"Error processing message from {phone_number}: {str(e)}")
            return "Sorry, I'm having some technical difficulties right now. Please try again in a moment! 🔧"
    
    def _handle_onboarding(self, db: Session, user: User, message: str, is_new_user: bool = False) -> str:
        current_step = user.onboarding_step
        
        if is_new_user and current_step == 'name' and not user.name:
            greeting = f"Welcome to your AI Tutor! {get_greeting_emoji(25)} \n\nI'm here to help you learn anything you're curious about through fun, personalized lessons!\n\n"
            return greeting + self.onboarding_steps['name']
        
        if current_step == 'name':
            return self._process_name_step(db, user, message)
        elif current_step == 'age':
            return self._process_age_step(db, user, message)
        # The following onboarding steps are not used in the simplified flow:
        # elif current_step == 'country':
        #     return self._process_country_step(db, user, message)
        # elif current_step == 'subjects':
        #     return self._process_subjects_step(db, user, message)
        # elif current_step == 'learning_mode':
        #     return self._process_learning_mode_step(db, user, message)
        # elif current_step == 'language':
        #     return self._process_language_step(db, user, message)
        
        # If we reach here, the user is on a legacy/unrecognized step (e.g., 'country').
        # Reset onboarding to 'name' to recover gracefully.
        try:
            update_user(db, user, onboarding_step='name')
        except Exception:
            pass
        # Return the first onboarding prompt directly
        return self.onboarding_steps['name']
        
        return "Something went wrong with onboarding. Let me help you start over! What's your name?"
    
    def _process_name_step(self, db: Session, user: User, message: str) -> str:
        name = message.strip()
        
        if len(name) < 1 or len(name) > 50:
            return "Please enter a name between 1 and 50 characters. What should I call you?"
        
        update_user(db, user, name=name, onboarding_step='age')
        emoji = get_greeting_emoji(25)
        
        return f"Nice to meet you, {name}! {emoji}\n\n{self.onboarding_steps['age']}"
    
    def _process_age_step(self, db: Session, user: User, message: str) -> str:
        age = validate_age(message)
        
        if age is None:
            return "Please enter a valid age (between 3 and 100). How old are you?"
        
        # Complete onboarding after collecting age in the simplified flow
        update_user(db, user, age=age, language='en', is_onboarded=True, onboarding_step='completed')
        emoji = get_greeting_emoji(age)
        
        welcome_msg = f"""
🎉 *Welcome to your personalized AI Tutor, {user.name}!* {emoji}

You're all set up! Here's what I know about you:
• Age: {age}

*Ready to learn? Try these commands:*
📚 `/lesson <topic>` - Start learning any topic
❓ `/help` - Get help and see all commands

*Example:* Try typing `/lesson photosynthesis`

What would you like to learn about first? 🚀
        """
        
        return format_for_whatsapp(welcome_msg, age)
    
    # The following handlers are unused in the simplified onboarding and are kept
    # commented for reference if we re-enable extended onboarding later.
    # def _process_country_step(self, db: Session, user: User, message: str) -> str:
    #     country = validate_country(message)
    #     if country is None:
    #         return "Please enter a valid country name. Which country are you from?"
    #     update_user(db, user, country=country, onboarding_step='subjects')
    #     return f"Great! Welcome from {country}! 🌍\n\n{self.onboarding_steps['subjects']}"
    
    # def _process_subjects_step(self, db: Session, user: User, message: str) -> str:
    #     subjects = validate_subjects(message)
    #     if not subjects:
    #         return "Please enter at least one subject you're interested in (e.g., math, science, history):"
    #     subjects_json = store_subjects_as_json(subjects)
    #     update_user(db, user, preferred_subjects=subjects_json, onboarding_step='learning_mode')
    #     subjects_text = ", ".join(subjects)
    #     return f"Awesome! I see you're interested in: {subjects_text} 📚\n\n{self.onboarding_steps['learning_mode']}"
    
    # def _process_learning_mode_step(self, db: Session, user: User, message: str) -> str:
    #     mode = validate_learning_mode(message)
    #     if mode is None:
    #         return 'Please choose either "text" for written lessons or "audio" for spoken lessons (audio coming soon!):'
    #     update_user(db, user, learning_mode=mode, onboarding_step='language')
    #     mode_text = "text-based" if mode == 'text' else "audio-based"
    #     return f"Perfect! I'll provide {mode_text} lessons. 📖\n\n{self.onboarding_steps['language']}"
    
    # def _process_language_step(self, db: Session, user: User, message: str) -> str:
    #     language = message.strip().lower()
    #     if language not in ['english', 'en', 'eng']:
    #         return 'Currently I only support English. Please type "english" or "en" to continue:'
    #     update_user(db, user, language='en', is_onboarded=True, onboarding_step='completed')
    #     emoji = get_greeting_emoji(user.age)
    #     welcome_msg = f"""
    # 🎉 *Welcome to your personalized AI Tutor, {user.name}!* {emoji}
    #
    # You're all set up! Here's what I know about you:
    # • Age: {user.age}
    # • Country: {user.country}
    # • Learning mode: {user.learning_mode}
    #
    # *Ready to learn? Try these commands:*
    # 📚 `/lesson <topic>` - Start learning any topic
    # ❓ `/help` - Get help and see all commands
    #
    # *Example:* Try typing `/lesson fractions` or `/lesson photosynthesis`
    #
    # What would you like to learn about first? 🚀
    #         """
    #     return format_for_whatsapp(welcome_msg, user.age)
    
    def _handle_regular_message(self, db: Session, user: User, message: str) -> str:
        message = message.strip()
        
        # Handle commands
        if message.lower().startswith('/help'):
            return self._handle_help_command(user)
        
        elif message.lower().startswith('/lesson'):
            return self._handle_lesson_command(db, user, message)
        
        elif message.lower().startswith('/next'):
            return self._handle_next_command(db, user)
        
        elif message.lower().startswith('/quiz'):
            return self._handle_quiz_command(db, user)
        
        # Handle quiz answers (check if user has an active quiz)
        elif self._is_quiz_answer(message):
            return self._handle_quiz_answer(db, user, message)
        
        # Handle general conversation
        else:
            return self._handle_general_message(db, user, message)
    
    def _handle_help_command(self, user: User) -> str:
        return get_help_message(user.age)
    
    def _handle_lesson_command(self, db: Session, user: User, message: str) -> str:
        topic = parse_lesson_command(message)
        if not topic:
            return "Please specify a topic! For example: `/lesson fractions` or `/lesson photosynthesis` 📚"
        
        # Try RAG retrieval first for any topic
        rag_success, retrieved_chunks, chunk_id = self._try_rag_retrieval(topic, user)
        logger.info(f"RAG success: {rag_success}, retrieved chunks: {retrieved_chunks}, chunk_id: {chunk_id}")
        
        if rag_success:
            return self._generate_rag_lesson(db, user, topic, retrieved_chunks, chunk_id)
        else:
            return self._generate_base_llm_lesson(db, user, topic)
    
    def _handle_next_command(self, db: Session, user: User) -> str:
        current_lesson = get_current_lesson(db, user.id)
        
        if not current_lesson:
            return "You don't have any lessons in progress. Start a new lesson with `/lesson <topic>`! 📚"
        
        try:
            if current_lesson.is_rag_lesson:
                system_prompt, user_prompt, chunk_id = get_rag_lesson(
                    current_lesson.topic, user.age, user.name, current_lesson.chunk_id
                )
                
                if chunk_id is None:
                    return f"Great job! You've completed the lesson on {current_lesson.topic}. Try a new topic with `/lesson <topic>`! 📚"
                
                from .llm import llm_service
                if not llm_service._initialized:
                    llm_service.initialize()
                
                response = llm_service.client.chat.completions.create(
                    model=llm_service.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=300,
                    temperature=0.7,
                    top_p=0.8,
                    stream=True
                )
                
                lesson_content = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        lesson_content += chunk.choices[0].delta.content
                
                lesson_content = lesson_content.strip()
                update_progress(db, current_lesson, 
                              lesson_content=lesson_content,
                              lesson_step=current_lesson.lesson_step + 1,
                              chunk_id=chunk_id)
                
                formatted_lesson = format_for_whatsapp(lesson_content, user.age)
                
                return f"📚 *{current_lesson.topic.title()} - Part {current_lesson.lesson_step}*\n\n{formatted_lesson}\n\n_Type `/next` to continue, /quiz for a quiz related to this topic or `/lesson <topic>` for something new!_"
            
            else:
                # Use fallback LLM approach for non-RAG lessons
                follow_up_topic = f"{current_lesson.topic} - Advanced Concepts"
                lesson_content = generate_lesson(follow_up_topic, user.age, user.name)
                update_progress(db, current_lesson, lesson_step=current_lesson.lesson_step + 1, lesson_content=lesson_content)
                
                formatted_lesson = format_for_whatsapp(lesson_content, user.age)
                
                return f"📚 *{current_lesson.topic.title()} - Part {current_lesson.lesson_step}*\n\n{formatted_lesson}\n\n_Type `/next` to continue, /quiz for a quiz related to this topic or `/lesson <topic>` for something new!_"
        
        except Exception as e:
            logger.error(f"Failed to generate next lesson part: {str(e)}")
            return "Sorry, I had trouble preparing the next part. Try starting a new lesson with `/lesson <topic>`! 📚"
    
    def _handle_general_message(self, db: Session, user: User, message: str) -> str:
        question_keywords = ['what is', 'how do', 'how does', 'explain', 'teach me', 'learn about']
        
        message_lower = message.lower()
        
        for keyword in question_keywords:
            if keyword in message_lower:
                topic = message_lower.replace(keyword, '').strip('?').strip()
                if len(topic) > 3:
                    return f"Great question! Let me teach you about {topic}. 📚\n\nTry: `/lesson {topic}`\n\nOr type `/help` to see all available commands!"
        
        responses = [
            f"Hi {user.name}! 👋 I'm here to help you learn. Try `/lesson <topic>` to start learning something new!",
            f"Hello! Ready to learn something interesting? Use `/lesson <topic>` or type `/help` for commands! 📚",
            f"Hey there! What would you like to learn about today? Type `/lesson <topic>` to get started! 🎓"
        ]
        
        if user.age <= 8:
            response = f"Hi {user.name}! 🌟 Want to learn something fun? Try `/lesson colors` or `/lesson animals`!"
        elif user.age <= 12:
            response = f"Hey {user.name}! 📚 Ready for a lesson? Try `/lesson fractions` or `/lesson dinosaurs`!"
        else:
            response = responses[hash(user.phone_number) % len(responses)]
        
        return format_for_whatsapp(response, user.age)
    
    def _generate_rag_lesson(self, db: Session, user: User, topic: str, retrieved_chunks: List[Dict], chunk_id: str) -> str:
        """Generate a lesson using RAG-retrieved content."""
        try:
            from .rag import rag_service
            system_prompt, user_prompt = rag_service.create_rag_lesson_prompt(
                topic, retrieved_chunks, user.age, user.name
            )
            
            from .llm import llm_service
            if not llm_service._initialized:
                llm_service.initialize()
            
            response = llm_service.client.chat.completions.create(
                model=llm_service.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=300,
                temperature=0.7,
                top_p=0.8,
                stream=True
            )
            
            lesson_content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    lesson_content += chunk.choices[0].delta.content
            
            lesson_content = lesson_content.strip()
            
            create_progress(db, user.id, topic, lesson_content, 
                          is_rag_lesson=True, chunk_id=chunk_id)
            
            formatted_lesson = format_for_whatsapp(lesson_content, user.age)
            
            logger.info(f"Generated RAG lesson for user {user.phone_number} on topic: {topic}")
            
            return f"📚 *Lesson: {topic.title()}*\n\n{formatted_lesson}\n\n_Type `/next` for more on this topic or `/lesson <new topic>` for something else!_"
        
        except Exception as e:
            logger.error(f"Failed to generate RAG lesson for topic {topic}: {str(e)}")
            # Fallback to base LLM if RAG generation fails
            return self._generate_base_llm_lesson(db, user, topic)
    
    def _generate_base_llm_lesson(self, db: Session, user: User, topic: str) -> str:
        """Generate a lesson using base LLM without RAG."""
        try:
            lesson_content = generate_lesson(topic, user.age, user.name)
            progress = create_progress(db, user.id, topic, lesson_content)
            formatted_lesson = format_for_whatsapp(lesson_content, user.age)
            
            logger.info(f"Generated base LLM lesson for user {user.phone_number} on topic: {topic}")
            
            return f"📚 *Lesson: {topic.title()}*\n\n{formatted_lesson}\n\n_Type `/next` for more on this topic or `/lesson <new topic>` for something else!_"
        
        except Exception as e:
            logger.error(f"Failed to generate lesson for topic {topic}: {str(e)}")
            return f"Sorry, I had trouble creating a lesson on {topic}. Please try a different topic or try again later! 📚"
    
    def _try_rag_retrieval(self, topic: str, user: User) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
        """
        Try to retrieve relevant chunks from RAG database for any topic.
        Returns (success, retrieved_chunks, chunk_id) where success indicates high confidence retrieval.
        """
        try:
            from .rag import initialize_rag
            initialize_rag()
            
            from .rag import rag_service
            retrieved_chunks = rag_service.retrieve_relevant_chunks(topic, limit=5)

            logger.info(f"Retrieved chunks: {retrieved_chunks}")
            
            if not retrieved_chunks:
                return False, None, None
            
            # Check retrieval confidence - if all similarity scores are below threshold, 
            # consider it low confidence
            if all(chunk['similarity_score'] < self.rag_confidence_threshold for chunk in retrieved_chunks):
                logger.info(f"Low confidence RAG retrieval for topic '{topic}' - scores: {[c['similarity_score'] for c in retrieved_chunks]}")
                return False, retrieved_chunks, None
            
            chunk_id = retrieved_chunks[0]['chunk_id']
            logger.info(f"High confidence RAG retrieval for topic '{topic}' - best score: {retrieved_chunks[0]['similarity_score']:.3f}")
            return True, retrieved_chunks, chunk_id
            
        except Exception as e:
            logger.error(f"Error during RAG retrieval for topic '{topic}': {str(e)}")
            return False, None, None
    
    
    def _handle_quiz_command(self, db: Session, user: User) -> str:
        """Handle /quiz command to create a quiz from current lesson"""
        try:
            current_lesson = get_current_lesson(db, user.id)
            if not current_lesson:
                return "You don't have any lessons in progress. Start a lesson with `/lesson <topic>` first! 📚"
            
            quiz_text, quiz_id = create_quiz_from_lesson(db, user.id, current_lesson.topic, user.age, user.name)
            
            if quiz_id == 0:
                return quiz_text
            
            logger.info(f"Created quiz for user {user.phone_number}")
            return quiz_text
            
        except Exception as e:
            logger.error(f"Failed to create quiz: {str(e)}")
            return "Sorry, I had trouble creating a quiz. Please try again! 🧩"
    
    def _is_quiz_answer(self, message: str) -> bool:
        """Check if message looks like quiz answers (e.g., '1A, 2B, 3True' or '1A 2A 3A')"""
        import re
        # Check if message contains patterns like "1A", "2B", "3True", etc.
        # Allow spaces or commas as separators
        pattern = r'\d+[A-D]|\d+(True|False)'
        matches = re.findall(pattern, message, re.IGNORECASE)
        # Consider it a quiz answer if we find at least 2 matches (likely multiple questions)
        return len(matches) >= 2
    
    def _handle_quiz_answer(self, db: Session, user: User, message: str) -> str:
        """Handle quiz answer submission"""
        try:
            # Check if user has an active quiz
            current_quiz = get_current_quiz(db, user.id)
            if not current_quiz:
                return "You don't have any active quiz. Start a lesson and use `/quiz` to create one! 🧩"
            
            feedback = check_quiz_answers(db, user.id, message)
            
            logger.info(f"Processed quiz answers for user {user.phone_number}")
            return feedback
            
        except Exception as e:
            logger.error(f"Failed to process quiz answers: {str(e)}")
            return "Sorry, I had trouble checking your answers. Please try again! 🧩"

message_handler = MessageHandler()

def process_whatsapp_message(db: Session, phone_number: str, message: str) -> str:
    return message_handler.process_message(db, phone_number, message)

if __name__ == "__main__":
    print("Message handlers module loaded successfully!")
