import logging
from typing import Tuple, Optional
from sqlalchemy.orm import Session

from .db import User, Progress, get_user_by_phone, create_user, update_user, create_progress, get_current_lesson, update_progress
from .llm import generate_lesson
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
            'age': 'How old are you? (This helps me adjust lessons for you)',
            'country': 'Which country are you from?',
            'subjects': 'What subjects interest you? (e.g., math, science, history - separate with commas)',
            'learning_mode': 'Do you prefer learning through "text" or would you like "audio" lessons in the future?',
            'language': 'What language would you like to learn in? (Currently supporting English - just type "english" or "en")'
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
        
        # Handle general conversation
        else:
            return self._handle_general_message(db, user, message)
    
    def _handle_help_command(self, user: User) -> str:
        return get_help_message(user.age)
    
    def _handle_lesson_command(self, db: Session, user: User, message: str) -> str:
        
        topic = parse_lesson_command(message)
        if not topic:
            return "Please specify a topic! For example: `/lesson fractions` or `/lesson photosynthesis` 📚"
        
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
            follow_up_topic = f"{current_lesson.topic} - Advanced Concepts"
            lesson_content = generate_lesson(follow_up_topic, user.age, user.name)
            update_progress(db, current_lesson, lesson_step=current_lesson.lesson_step + 1, lesson_content=lesson_content)
            
            formatted_lesson = format_for_whatsapp(lesson_content, user.age)
            
            return f"📚 *{current_lesson.topic.title()} - Part {current_lesson.lesson_step}*\n\n{formatted_lesson}\n\n_Type `/next` to continue or `/lesson <topic>` for something new!_"
        
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

message_handler = MessageHandler()

def process_whatsapp_message(db: Session, phone_number: str, message: str) -> str:
    return message_handler.process_message(db, phone_number, message)

if __name__ == "__main__":
    print("Message handlers module loaded successfully!")
