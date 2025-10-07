import logging
from typing import Tuple, Optional
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
            'country': 'Which country are you from?',
            'subjects': 'What subjects interest you? (e.g., math, science, history - separate with commas)',
            'learning_mode': 'Do you prefer learning through "text" or would you like "audio" lessons in the future?',
            'language': 'What language would you like to learn in? (Currently supporting English - just type "english" or "en")'
        }
        
        # Science topics that can use RAG
        self.science_topics = {
            # Biology topics (from Concepts of Biology textbook)
            'cell', 'cells', 'DNA', 'evolution', 'ecosystem', 'photosynthesis', 
            'mitosis', 'genetics', 'bacteria', 'virus', 'mammals', 'chromosome',
            'protein', 'enzyme', 'respiration', 'adaptation', 'organism', 'tissue',
            'organ', 'system', 'membrane', 'nucleus', 'mitochondria', 'chloroplast',
            'gene', 'allele', 'mutation', 'species', 'population', 'community',
            'biome', 'food chain', 'food web', 'decomposer', 'producer', 'consumer',
            
            # Chemistry topics (from Chemistry2e textbook)
            'atom', 'atoms', 'molecule', 'molecules', 'element', 'elements',
            'compound', 'compounds', 'sodium', 'chlorine', 'hydrogen', 'oxygen',
            'carbon', 'nitrogen', 'measurements', 'measurement', 'units', 'unit',
            'density', 'mass', 'volume', 'temperature', 'pressure', 'reaction',
            'chemical', 'chemistry', 'periodic table', 'periodic', 'table',
            'bond', 'bonds', 'ionic', 'covalent', 'metal', 'metals', 'nonmetal',
            'acid', 'base', 'ph', 'solution', 'solutions', 'mixture', 'mixtures',
            
            # Original topics
            'plants', 'trees', 'leaves', 'roots', 'flowers',
            'animals', 'birds', 'fish', 'reptiles', 'amphibians', 'habitats',
            'solar system', 'planets', 'sun', 'moon', 'earth', 'mars', 'jupiter', 'saturn',
            'energy', 'light', 'heat', 'sound', 'electricity', 'renewable energy',
            'weather', 'rain', 'snow', 'wind', 'clouds', 'climate',
            'water cycle', 'evaporation', 'condensation', 'precipitation'
        }
    
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
        elif current_step == 'country':
            return self._process_country_step(db, user, message)
        elif current_step == 'subjects':
            return self._process_subjects_step(db, user, message)
        elif current_step == 'learning_mode':
            return self._process_learning_mode_step(db, user, message)
        elif current_step == 'language':
            return self._process_language_step(db, user, message)
        
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
        
        update_user(db, user, age=age, onboarding_step='country')
        emoji = get_greeting_emoji(age)
        
        return f"Got it! {emoji}\n\n{self.onboarding_steps['country']}"
    
    def _process_country_step(self, db: Session, user: User, message: str) -> str:
        country = validate_country(message)
        
        if country is None:
            return "Please enter a valid country name. Which country are you from?"
        
        update_user(db, user, country=country, onboarding_step='subjects')
        
        return f"Great! Welcome from {country}! 🌍\n\n{self.onboarding_steps['subjects']}"
    
    def _process_subjects_step(self, db: Session, user: User, message: str) -> str:
        subjects = validate_subjects(message)
        
        if not subjects:
            return "Please enter at least one subject you're interested in (e.g., math, science, history):"
        
        subjects_json = store_subjects_as_json(subjects)
        update_user(db, user, preferred_subjects=subjects_json, onboarding_step='learning_mode')
        
        subjects_text = ", ".join(subjects)
        return f"Awesome! I see you're interested in: {subjects_text} 📚\n\n{self.onboarding_steps['learning_mode']}"
    
    def _process_learning_mode_step(self, db: Session, user: User, message: str) -> str:
        mode = validate_learning_mode(message)
        
        if mode is None:
            return 'Please choose either "text" for written lessons or "audio" for spoken lessons (audio coming soon!):'
        
        update_user(db, user, learning_mode=mode, onboarding_step='language')
        
        mode_text = "text-based" if mode == 'text' else "audio-based"
        return f"Perfect! I'll provide {mode_text} lessons. 📖\n\n{self.onboarding_steps['language']}"
    
    def _process_language_step(self, db: Session, user: User, message: str) -> str:
        language = message.strip().lower()
        
        if language not in ['english', 'en', 'eng']:
            return 'Currently I only support English. Please type "english" or "en" to continue:'
        
        # Complete onboarding
        update_user(db, user, language='en', is_onboarded=True, onboarding_step='completed')
        
        emoji = get_greeting_emoji(user.age)
        welcome_msg = f"""
🎉 *Welcome to your personalized AI Tutor, {user.name}!* {emoji}

You're all set up! Here's what I know about you:
• Age: {user.age}
• Country: {user.country}
• Learning mode: {user.learning_mode}

*Ready to learn? Try these commands:*
📚 `/lesson <topic>` - Start learning any topic
❓ `/help` - Get help and see all commands

*Example:* Try typing `/lesson fractions` or `/lesson photosynthesis`

What would you like to learn about first? 🚀
        """
        
        return format_for_whatsapp(welcome_msg, user.age)
    
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
        
        # Check if this is a science topic that can use RAG
        if self._is_science_topic(topic):
            return self._handle_rag_lesson_command(db, user, message)
        
        # For non-science topics, use the original LLM approach
        try:
            lesson_content = generate_lesson(topic, user.age, user.name)
            progress = create_progress(db, user.id, topic, lesson_content)
            formatted_lesson = format_for_whatsapp(lesson_content, user.age)
            
            logger.info(f"Generated lesson for user {user.phone_number} on topic: {topic}")
            
            return f"📚 *Lesson: {topic.title()}*\n\n{formatted_lesson}\n\n_Type `/next` for more on this topic or `/lesson <new topic>` for something else!_"
        
        except Exception as e:
            logger.error(f"Failed to generate lesson for topic {topic}: {str(e)}")
            return f"Sorry, I had trouble creating a lesson on {topic}. Please try a different topic or try again later! 📚"
    
    def _handle_next_command(self, db: Session, user: User) -> str:
        current_lesson = get_current_lesson(db, user.id)
        
        if not current_lesson:
            return "You don't have any lessons in progress. Start a new lesson with `/lesson <topic>`! 📚"
        
        try:
            # Check if this is a RAG lesson
            if current_lesson.is_rag_lesson:
                # Use RAG for continuing science lessons
                system_prompt, user_prompt, chunk_id = get_rag_lesson(
                    current_lesson.topic, user.age, user.name, current_lesson.chunk_id
                )
                
                if chunk_id is None:
                    return f"Great job! You've completed the lesson on {current_lesson.topic}. Try a new science topic with `/lesson <topic>`! 🔬"
                
                # Generate lesson using LLM with RAG context
                from .llm import llm_service
                if not llm_service._initialized:
                    llm_service.initialize()
                
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
                update_progress(db, current_lesson, 
                              lesson_content=lesson_content,
                              lesson_step=current_lesson.lesson_step + 1,
                              chunk_id=chunk_id)
                
                formatted_lesson = format_for_whatsapp(lesson_content, user.age)
                
                return f"🔬 *{current_lesson.topic.title()} - Part {current_lesson.lesson_step}*\n\n{formatted_lesson}\n\n_Type `/next` to continue, /quiz for a quiz related to this topic or`/lesson <topic>` for something new!_"
            
            else:
                # Use original LLM approach for non-RAG lessons
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
    
    def _is_science_topic(self, topic: str) -> bool:
        """Check if a topic is science-related and can use RAG."""
        topic_lower = topic.lower()
        
        # Check for exact matches
        if topic_lower in self.science_topics:
            return True
        
        # Check for partial matches
        for science_topic in self.science_topics:
            if science_topic in topic_lower or topic_lower in science_topic:
                return True
        
        return False
    
    def _handle_rag_lesson_command(self, db: Session, user: User, message: str) -> str:
        """Handle lesson command with RAG for science topics."""
        topic = parse_lesson_command(message)
        if not topic:
            return "Please specify a science topic! For example: `/lesson plants` or `/lesson solar system` 🌱🔬"
        
        # Check if user is in the right age range for RAG
        if user.age < 6 or user.age > 12:
            return f"I see you're {user.age} years old! My science lessons are designed for kids aged 6-12. Try asking about a different subject, or ask your parents to help you with science! 📚"
        
        try:
            # Get current lesson to check if we're continuing
            current_lesson = get_current_lesson(db, user.id)
            current_chunk_id = current_lesson.chunk_id if current_lesson and current_lesson.is_rag_lesson else None
            
            # Generate RAG lesson
            system_prompt, user_prompt, chunk_id = get_rag_lesson(topic, user.age, user.name, current_chunk_id)
            
            if chunk_id is None:
                return f"I'm sorry, I couldn't find information about {topic} in my science database. Try asking about plants, animals, the solar system, energy, or weather! 🔬"
            
            # Generate lesson using LLM with RAG context
            from .llm import llm_service
            if not llm_service._initialized:
                llm_service.initialize()
            
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
            
            # Create or update progress
            if current_lesson and current_lesson.is_rag_lesson and current_chunk_id:
                # Continuing existing RAG lesson
                update_progress(db, current_lesson, 
                              lesson_content=lesson_content,
                              lesson_step=current_lesson.lesson_step + 1,
                              chunk_id=chunk_id)
            else:
                # New RAG lesson
                create_progress(db, user.id, topic, lesson_content, 
                              is_rag_lesson=True, chunk_id=chunk_id)
            
            formatted_lesson = format_for_whatsapp(lesson_content, user.age)
            
            logger.info(f"Generated RAG lesson for user {user.phone_number} on topic: {topic}")
            
            return f"🔬 *Science Lesson: {topic.title()}*\n\n{formatted_lesson}\n\n_Type `/next` for more on this topic or `/lesson <new topic>` for something else!_"
        
        except Exception as e:
            logger.error(f"Failed to generate RAG lesson for topic {topic}: {str(e)}")
            return f"Sorry, I had trouble creating a science lesson on {topic}. Please try a different science topic or try again later! 🔬"
    
    def _handle_quiz_command(self, db: Session, user: User) -> str:
        """Handle /quiz command to create a quiz from current lesson"""
        try:
            # Get current lesson to determine topic
            current_lesson = get_current_lesson(db, user.id)
            if not current_lesson:
                return "You don't have any lessons in progress. Start a lesson with `/lesson <topic>` first! 📚"
            
            quiz_text, quiz_id = create_quiz_from_lesson(db, user.id, current_lesson.topic, user.age, user.name)
            
            if quiz_id == 0:
                return quiz_text  # Error message
            
            logger.info(f"Created quiz for user {user.phone_number}")
            return quiz_text
            
        except Exception as e:
            logger.error(f"Failed to create quiz: {str(e)}")
            return "Sorry, I had trouble creating a quiz. Please try again! 🧩"
    
    def _is_quiz_answer(self, message: str) -> bool:
        """Check if message looks like quiz answers (e.g., '1A, 2B, 3True')"""
        import re
        # Check if message contains patterns like "1A", "2B", "3True", etc.
        pattern = r'\d+[A-D]|\d+(True|False)'
        return bool(re.search(pattern, message, re.IGNORECASE))
    
    def _handle_quiz_answer(self, db: Session, user: User, message: str) -> str:
        """Handle quiz answer submission"""
        try:
            # Check if user has an active quiz
            current_quiz = get_current_quiz(db, user.id)
            if not current_quiz:
                return "You don't have any active quiz. Start a lesson and use `/quiz` to create one! 🧩"
            
            # Check answers and provide feedback
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
