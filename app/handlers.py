import logging
from typing import Tuple, Optional, List, Dict
from sqlalchemy.orm import Session

from .db import User, Progress, get_user_by_phone, create_user, update_user, create_progress, get_current_lesson, update_progress, get_current_quiz, get_completed_lessons, get_completed_quizzes, get_user_progress
from .llm import generate_lesson
from .rag import get_rag_lesson, initialize_rag
from .quiz import create_quiz_from_lesson, check_quiz_answers
from .utils import (
    format_for_whatsapp, validate_age,
    get_help_message, parse_lesson_command, 
    get_greeting_emoji, clean_topic_title, strip_think_tags,
    format_progress_review
)
from .audio import transcribe_audio, synthesize_speech, synthesize_speech_chunked, tts_service
from .image import generate_lesson_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _attach_lesson_image(result: dict, topic: str, language: str, context: str = "lesson") -> None:
    """Generate an image for the lesson/next topic and attach to result dict. Uses a safe, short topic for the image API."""
    lang = language or "en"
    # Use a clean, short topic so image API succeeds (avoid long/special strings from DB)
    image_topic = (clean_topic_title(topic).strip() if topic else "") or "lesson"
    if len(image_topic) > 40:
        image_topic = image_topic.split()[0] if image_topic.split() else "lesson"
    try:
        img_result = generate_lesson_image(image_topic, lang)
        if img_result:
            result["image_bytes"] = img_result[0]
            result["image_content_type"] = img_result[1]
            logger.info("Generated image for /%s topic '%s'", context, topic)
        else:
            logger.warning("Image generation failed for /%s topic '%s'; sending text only", context, topic)
    except Exception as e:
        logger.warning("Image generation error for /%s topic '%s': %s; sending text only", context, topic, e)


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
    
    def process_message(self, db: Session, phone_number: str, message: str, for_audio: bool = False) -> str:
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
                return self._handle_regular_message(db, user, message, for_audio=for_audio)
                
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

🎤 *Voice Messages:*
You can also send voice messages! Just say:
• "teach me about <topic>" (e.g., "teach me about cells")
• "next" to continue
• "help" for help

*Example:* Try typing `/lesson cells` or say "teach me about cells" in a voice message!

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
    # *Example:* Try typing `/lesson cells` or `/lesson photosynthesis`
    #
    # What would you like to learn about first? 🚀
    #         """
    #     return format_for_whatsapp(welcome_msg, user.age)
    
    def _handle_regular_message(self, db: Session, user: User, message: str, for_audio: bool = False):
        message = message.strip()
        message_lower = message.lower()
        
        # Handle commands (text format with slash)
        if message_lower.startswith('/help'):
            return self._handle_help_command(user)
        
        elif message_lower.startswith('/lesson'):
            return self._handle_lesson_command(db, user, message, for_audio=for_audio)
        
        elif message_lower.startswith('/next'):
            return self._handle_next_command(db, user, for_audio=for_audio)
        
        elif message_lower.startswith('/quiz'):
            return self._handle_quiz_command(db, user)
        
        elif message_lower.startswith('/progress') or message_lower.startswith('/review'):
            return self._handle_progress_review(db, user)
        
        elif message_lower.startswith('teach me about '):
            topic = message_lower.replace('teach me about', '').strip()
            if topic and len(topic) > 2:
                return self._handle_lesson_command(db, user, f"/lesson {topic}", for_audio=for_audio)
        elif message_lower.startswith('lesson '):
            return self._handle_lesson_command(db, user, message, for_audio=for_audio)
        elif message_lower.startswith('next'):
            return self._handle_next_command(db, user, for_audio=for_audio)
        elif message_lower.startswith('quiz'):
            return self._handle_quiz_command(db, user)
        elif message_lower.startswith('progress') or message_lower.startswith('review'):
            return self._handle_progress_review(db, user)
        elif message_lower.startswith('help'):
            return self._handle_help_command(user)
        
        # Handle quiz answers (check if user has an active quiz)
        elif self._is_quiz_answer(message):
            return self._handle_quiz_answer(db, user, message)
        
        else:
            return self._handle_general_message(db, user, message, for_audio=for_audio)
    
    def _handle_help_command(self, user: User) -> str:
        return get_help_message(user.age, user.language)
    
    def _handle_progress_review(self, db: Session, user: User) -> str:
        """Show completed lessons and quiz scores."""
        lessons = get_user_progress(db, user.id, limit=10)
        completed_quizzes = get_completed_quizzes(db, user.id, limit=10)
        return format_progress_review(lessons, completed_quizzes, language=user.language or "en")
    
    def _handle_lesson_command(self, db: Session, user: User, message: str, for_audio: bool = False):
        topic = parse_lesson_command(message)
        if not topic:
            error_msg = "Please specify a topic! For example: `/lesson cells` or `/lesson photosynthesis` 📚\n\n🎤 *Voice format:* Say \"teach me about cells\" or \"teach me about photosynthesis\""
            return error_msg if for_audio else {"text": error_msg}
        
        # Try RAG retrieval first for any topic
        rag_success, retrieved_chunks, chunk_id = self._try_rag_retrieval(topic, user)
        logger.info(f"RAG success: {rag_success}, retrieved chunks: {retrieved_chunks}, chunk_id: {chunk_id}")
        
        if rag_success:
            result = self._generate_rag_lesson(db, user, topic, retrieved_chunks, chunk_id, for_audio=for_audio)
        else:
            result = self._generate_base_llm_lesson(db, user, topic, for_audio=for_audio)
        
        # For text lessons (not audio), add image synchronously
        if not for_audio:
            if isinstance(result, str):
                result = {"text": result}
            # Generate image for the topic
            lang = user.language if user else "en"
            img_result = generate_lesson_image(topic, lang)
            if img_result:
                result["image_bytes"] = img_result[0]
                result["image_content_type"] = img_result[1]
                logger.info(f"Generated image for lesson topic '{topic}'")
            else:
                logger.warning(f"Image generation failed for topic '{topic}'; sending text only")
        
        return result
    
    def _handle_next_command(self, db: Session, user: User, for_audio: bool = False):
        current_lesson = get_current_lesson(db, user.id)
        
        if not current_lesson:
            error_msg = "You don't have any lessons in progress. Start a new lesson with `/lesson <topic>`! 📚"
            return error_msg if for_audio else {"text": error_msg}
        
        try:
            if current_lesson.is_rag_lesson:
                # Pass previous lesson content for conversational continuity
                system_prompt, user_prompt, chunk_id = get_rag_lesson(
                    current_lesson.topic, user.age, user.name, 
                    current_lesson.chunk_id, previous_content=current_lesson.lesson_content,
                    for_audio=for_audio, language=user.language
                )
                
                if chunk_id is None:
                    update_progress(db, current_lesson, completed=True)
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
                    max_completion_tokens=800,
                    temperature=0.7,
                    top_p=0.8,
                    stream=True
                )
                
                lesson_content = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        lesson_content += chunk.choices[0].delta.content
                
                lesson_content = lesson_content.strip()
                lesson_content = strip_think_tags(lesson_content)
                # Check if content exceeds Twilio's 1400 character limit
                if len(lesson_content) > 1400:
                    logger.warning(f"Next lesson response too long ({len(lesson_content)} chars), retrying with stricter limit")
                    # Retry with a much stricter character limit
                    # Language instruction
                    lang_instruction = ""
                    if user.language != "en":
                        from .language import get_language_name
                        lang_name = get_language_name(user.language, native=True)
                        lang_instruction = f"\n- Language: Generate the entire lesson in {lang_name} ({user.language.upper()}). All text, explanations, examples, and responses must be in {lang_name}."
                    
                    retry_system_prompt = f"""You are an expert science teacher for children aged {user.age} years old.
You are continuing a lesson on {current_lesson.topic}. The student has already learned the previous part.

Instructions:
- Topic: {current_lesson.topic} (continuation)
- Age group: {user.age} years old
- Length: Keep it VERY SHORT (under 1200 characters total - this is critical for WhatsApp delivery)
- Style: Use simple language, clear examples, and everyday situations{lang_instruction}
- Use the provided educational content as your source of information
- Make sure all facts are accurate and age-appropriate

CONTINUATION STRUCTURE:
- Start by briefly referencing what was covered in the previous part (1-2 sentences)
- Then continue with new information
- Do NOT repeat examples from the previous part
- Do NOT start with a new example - jump straight into continuing the explanation

CRITICAL FORMATTING RULES:
- Use single asterisk *text* for bold (WhatsApp format), NOT double asterisks **
- Do NOT include "Try This at Home" or similar activity sections unless they directly relate to the topic
- Focus on clear explanations and examples, not generic activities

CRITICAL: Keep the response under 1200 characters to ensure WhatsApp delivery. Be concise but complete."""
                    if for_audio:
                        retry_system_prompt += """

AUDIO/TTS MODE: Your reply will be read aloud by text-to-speech. Write for listening: use short, complete sentences; avoid markdown headers (##) and bullet lists—use flowing prose instead; do not include instructions like "Type /next"; use minimal or no emojis; write as if you are speaking to the student."""
                    
                    retry_response = llm_service.client.chat.completions.create(
                        model=llm_service.model_name,
                        messages=[
                            {"role": "system", "content": retry_system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_completion_tokens=600,  # Increased to ensure complete responses
                        temperature=0.7,
                        top_p=0.8,
                        stream=True
                    )
                    
                    lesson_content = ""
                    for chunk in retry_response:
                        if chunk.choices[0].delta.content:
                            lesson_content += chunk.choices[0].delta.content
                    
                    lesson_content = lesson_content.strip()
                    lesson_content = strip_think_tags(lesson_content)
                    logger.info(f"Next lesson retry response length: {len(lesson_content)} characters")
                
                # Check if content appears to be truncated (doesn't end with proper punctuation)
                # Allow emojis at the end, but check the text before emojis
                import re
                text_without_emojis = re.sub(r'[\s\U0001F300-\U0001F9FF]+$', '', lesson_content.rstrip())
                if lesson_content and text_without_emojis and not text_without_emojis.endswith(('.', '!', '?', ':', ';')):
                    logger.warning(f"Next lesson content appears truncated, attempting to complete: {lesson_content[-100:]}")
                    # Try to complete the truncated content
                    # Language instruction for completion
                    lang_instruction = ""
                    if user.language != "en":
                        from .language import get_language_name
                        lang_name = get_language_name(user.language, native=True)
                        lang_instruction = f" Complete the text in {lang_name} ({user.language.upper()})."
                    
                    completion_system_prompt = f"Complete the following educational text naturally. Only provide the completion to finish the thought, not the full text. Make sure it ends with proper punctuation.{lang_instruction} Do not include any thinking tags or reasoning - only provide the completion text."
                    
                    completion_response = llm_service.client.chat.completions.create(
                        model=llm_service.model_name,
                        messages=[
                            {"role": "system", "content": completion_system_prompt},
                            {"role": "user", "content": f"Complete this educational text (finish the thought naturally): {lesson_content[-300:]}"}
                        ],
                        max_completion_tokens=300,
                        temperature=0.3,
                        top_p=0.8,
                        stream=False
                    )
                    
                    if completion_response.choices[0].message.content:
                        completion = completion_response.choices[0].message.content.strip()
                        # Strip think tags from completion
                        completion = strip_think_tags(completion)
                        # Remove any duplicate text at the start
                        if lesson_content.endswith(completion[:20]):
                            lesson_content = lesson_content.rstrip()
                        else:
                            lesson_content += " " + completion
                        
                        # Ensure completion ends with punctuation (before any emojis)
                        text_without_emojis = re.sub(r'[\s\U0001F300-\U0001F9FF]+$', '', lesson_content.rstrip())
                        if text_without_emojis and not text_without_emojis.endswith(('.', '!', '?', ':', ';')):
                            # Add punctuation if missing
                            lesson_content = lesson_content.rstrip() + '.'
                        
                        logger.info(f"Successfully completed truncated next lesson content")
                
                # Final check - if still over limit, truncate at sentence boundary but ensure it ends properly
                if len(lesson_content) > 1400:
                    logger.warning(f"Next lesson response still too long ({len(lesson_content)} chars), truncating at sentence boundary")
                    sentences = lesson_content.split('. ')
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated + sentence + '. ') <= 1350:  # Leave room for proper ending
                            truncated += sentence + '. '
                        else:
                            break
                    # Ensure it ends with punctuation
                    truncated = truncated.strip()
                    if not truncated.endswith(('.', '!', '?')):
                        truncated += '.'
                    lesson_content = truncated
                    logger.info(f"Next lesson truncated to {len(lesson_content)} characters")
                update_progress(db, current_lesson, 
                              lesson_content=lesson_content,
                              lesson_step=current_lesson.lesson_step + 1,
                              chunk_id=chunk_id)
                
                formatted_lesson = format_for_whatsapp(lesson_content, user.age)
                
                result_text = f"*{clean_topic_title(current_lesson.topic)} - Part {current_lesson.lesson_step}*\n\n{formatted_lesson}\n\n_Type `/next` to continue, /quiz for a quiz related to this topic or `/lesson <topic>` for something new!_"
                
                if for_audio:
                    return result_text
                
                # For text lessons, always add image (same as /lesson)
                result = {"text": result_text}
                _attach_lesson_image(result, current_lesson.topic, user.language, "next")
                return result
            
            else:
                # Use fallback LLM approach for non-RAG lessons
                follow_up_topic = f"{current_lesson.topic} - Advanced Concepts"
                lesson_content = generate_lesson(
                    follow_up_topic, user.age, user.name, 
                    is_continuation=True, previous_content=current_lesson.lesson_content,
                    for_audio=for_audio
                )
                update_progress(db, current_lesson, lesson_step=current_lesson.lesson_step + 1, lesson_content=lesson_content)
                
                formatted_lesson = format_for_whatsapp(lesson_content, user.age)
                
                result_text = f"*{clean_topic_title(current_lesson.topic)} - Part {current_lesson.lesson_step}*\n\n{formatted_lesson}\n\n_Type `/next` to continue, /quiz for a quiz related to this topic or `/lesson <topic>` for something new!_"
                
                if for_audio:
                    return result_text
                
                # For text lessons, always add image (same as /lesson)
                result = {"text": result_text}
                _attach_lesson_image(result, current_lesson.topic, user.language, "next")
                return result
        
        except Exception as e:
            logger.error(f"Failed to generate next lesson part: {str(e)}")
            error_msg = "Sorry, I had trouble preparing the next part. Try starting a new lesson with `/lesson <topic>`! 📚"
            return error_msg if for_audio else {"text": error_msg}
    
    def _handle_general_message(self, db: Session, user: User, message: str, for_audio: bool = False) -> str:
        """Handle general messages with keyword fallback for natural speech."""
        question_keywords = [
            'teach me about', 'can you teach me about',
            'what is', 'how do', 'how does', 'explain', 
            'teach me', 'learn about', 'tell me about', 
            'i want to learn about', 'can you tell me about', 
            'i want to know about'
        ]
        
        message_lower = message.lower()
        
        for keyword in question_keywords:
            if keyword in message_lower:
                # Find the position of the keyword and extract text AFTER it
                keyword_pos = message_lower.find(keyword)
                if keyword_pos != -1:
                    topic = message_lower[keyword_pos + len(keyword):].strip()
                    topic = topic.replace('about', '').strip('?').strip('.').strip(',').strip()
                    common_prefixes = ['the', 'a', 'an', 'can you', 'please', 'could you']
                    for prefix in common_prefixes:
                        if topic.lower().startswith(prefix + ' '):
                            topic = topic[len(prefix) + 1:].strip()
                    topic = topic.strip('?').strip('.').strip(',').strip()
                    
                    if len(topic) > 3:
                        logger.info(f"Auto-detected learning intent for topic: '{topic}' (from message: '{message}')")
                        return self._handle_lesson_command(db, user, f"/lesson {topic}", for_audio=for_audio)
        
        voice_format_hint = "\n\n🎤 *Voice messages:* Say \"teach me about <topic>\" (e.g., \"teach me about cells\")"
        
        responses = [
            f"Hi {user.name}! 👋 I'm here to help you learn.{voice_format_hint}\n\nTry `/lesson <topic>` to start learning something new!",
            f"Hello! Ready to learn something interesting?{voice_format_hint}\n\nUse `/lesson <topic>` or type `/help` for commands! 📚",
            f"Hey there! What would you like to learn about today?{voice_format_hint}\n\nType `/lesson <topic>` to get started! 🎓"
        ]
        
        if user.age <= 8:
            response = f"Hi {user.name}! 🌟 Want to learn something fun?{voice_format_hint}\n\nTry `/lesson colors` or `/lesson animals`!"
        elif user.age <= 12:
            response = f"Hey {user.name}! 📚 Ready for a lesson?{voice_format_hint}\n\nTry `/lesson cells` or `/lesson dinosaurs`!"
        else:
            response = responses[hash(user.phone_number) % len(responses)]
        
        return format_for_whatsapp(response, user.age)
    
    def _generate_rag_lesson(self, db: Session, user: User, topic: str, retrieved_chunks: List[Dict], chunk_id: str, for_audio: bool = False) -> str:
        """Generate a lesson using RAG-retrieved content."""
        try:
            from .rag import rag_service
            system_prompt, user_prompt = rag_service.create_rag_lesson_prompt(
                topic, retrieved_chunks, user.age, user.name, for_audio=for_audio, language=user.language
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
                max_completion_tokens=800,
                temperature=0.7,
                top_p=0.8,
                stream=True
            )
            
            lesson_content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    lesson_content += chunk.choices[0].delta.content
            
            lesson_content = lesson_content.strip()
            lesson_content = strip_think_tags(lesson_content)
            if len(lesson_content) > 4600:
                logger.warning(f"RAG response too long ({len(lesson_content)} chars), retrying with stricter limit")
                # Retry with a much stricter character limit
                # Language instruction
                lang_instruction = ""
                if user.language != "en":
                    from .language import get_language_name
                    lang_name = get_language_name(user.language, native=True)
                    lang_instruction = f"\n- Language: Generate the entire lesson in {lang_name} ({user.language.upper()}). All text, explanations, examples, and responses must be in {lang_name}."
                
                retry_system_prompt = f"""You are an expert science teacher for children aged {user.age} years old.
Your goal is to create an engaging, accurate science lesson using the provided educational content.

Instructions:   
- Topic: {topic}
- Age group: {user.age} years old
- Length: Keep it VERY SHORT (under 1200 characters total - this is critical for WhatsApp delivery)
- Style: Use simple language, clear examples, and everyday situations{lang_instruction}
- Use the provided educational content as your source of information
- Make sure all facts are accurate and age-appropriate
- Structure: Brief introduction, key explanation and one simple example

CRITICAL FORMATTING RULES:
- Use single asterisk *text* for bold (WhatsApp format), NOT double asterisks **
- Do NOT include "Try This at Home" or similar activity sections unless they directly relate to the topic
- Focus on clear explanations and examples, not generic activities

CRITICAL: Keep the response under 1200 characters to ensure WhatsApp delivery. Be concise but complete."""
                if for_audio:
                    retry_system_prompt += """

AUDIO/TTS MODE: Your reply will be read aloud by text-to-speech. Write for listening: use short, complete sentences; avoid markdown headers (##) and bullet lists—use flowing prose instead; do not include instructions like "Type /next"; use minimal or no emojis; write as if you are speaking to the student."""
                
                retry_response = llm_service.client.chat.completions.create(
                    model=llm_service.model_name,
                    messages=[
                        {"role": "system", "content": retry_system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=400,  # Reduced token limit
                    temperature=0.7,
                    top_p=0.8,
                    stream=True
                )
                
                lesson_content = ""
                for chunk in retry_response:
                    if chunk.choices[0].delta.content:
                        lesson_content += chunk.choices[0].delta.content
                
                lesson_content = lesson_content.strip()
                lesson_content = strip_think_tags(lesson_content)
                logger.info(f"RAG retry response length: {len(lesson_content)} characters")
            
            # Check if content appears to be truncated (doesn't end with proper punctuation)
            if lesson_content and not lesson_content.rstrip().endswith(('.', '!', '?', ':', ';')):
                logger.warning(f"RAG content appears truncated, attempting to complete: {lesson_content[-100:]}")
                # Try to complete the truncated content
                # Language instruction for completion
                lang_instruction = ""
                if user.language != "en":
                    from .language import get_language_name
                    lang_name = get_language_name(user.language, native=True)
                    lang_instruction = f" Complete the text in {lang_name} ({user.language.upper()})."
                
                completion_system_prompt = f"Complete the following educational text naturally. Only provide the completion to finish the thought, not the full text. Make sure it ends with proper punctuation.{lang_instruction} Do not include any thinking tags or reasoning - only provide the completion text."
                
                completion_response = llm_service.client.chat.completions.create(
                    model=llm_service.model_name,
                    messages=[
                        {"role": "system", "content": completion_system_prompt},
                        {"role": "user", "content": f"Complete this educational text (finish the thought naturally): {lesson_content[-300:]}"}
                    ],
                    max_completion_tokens=300,
                    temperature=0.3,
                    top_p=0.8,
                    stream=False
                )
                
                if completion_response.choices[0].message.content:
                    completion = completion_response.choices[0].message.content.strip()
                    # Strip think tags from completion
                    completion = strip_think_tags(completion)
                    # Remove any duplicate text at the start
                    if lesson_content.endswith(completion[:20]):
                        lesson_content = lesson_content.rstrip()
                    else:
                        lesson_content += " " + completion
                    logger.info(f"Successfully completed truncated RAG content")
            
            # Final check - if still over limit, truncate at sentence boundary but ensure it ends properly
            if len(lesson_content) > 1400:
                logger.warning(f"RAG response still too long ({len(lesson_content)} chars), truncating at sentence boundary")
                sentences = lesson_content.split('. ')
                truncated = ""
                for sentence in sentences:
                    if len(truncated + sentence + '. ') <= 1350:  # Leave room for proper ending
                        truncated += sentence + '. '
                    else:
                        break
                # Ensure it ends with punctuation
                truncated = truncated.strip()
                if not truncated.endswith(('.', '!', '?')):
                    truncated += '.'
                lesson_content = truncated
                logger.info(f"RAG truncated to {len(lesson_content)} characters")
            
            create_progress(db, user.id, topic, lesson_content, 
                          is_rag_lesson=True, chunk_id=chunk_id)
            
            formatted_lesson = format_for_whatsapp(lesson_content, user.age)
            
            logger.info(f"Generated RAG lesson for user {user.phone_number} on topic: {topic}")
            
            result_text = f"📚 *Lesson: {clean_topic_title(topic)}*\n\n{formatted_lesson}\n\n_Type `/next` for more on this topic or `/lesson <new topic>` for something else!_"
            
            if for_audio:
                return result_text
            
            # For text lessons, add image (same helper as /next)
            result = {"text": result_text}
            _attach_lesson_image(result, topic, user.language, "lesson")
            return result
        
        except Exception as e:
            logger.error(f"Failed to generate RAG lesson for topic {topic}: {str(e)}")
            # Fallback to base LLM if RAG generation fails
            return self._generate_base_llm_lesson(db, user, topic, for_audio=for_audio)
    
    def _generate_base_llm_lesson(self, db: Session, user: User, topic: str, for_audio: bool = False) -> str:
        """Generate a lesson using base LLM without RAG."""
        try:
            lesson_content = generate_lesson(topic, user.age, user.name, for_audio=for_audio, language=user.language)
            progress = create_progress(db, user.id, topic, lesson_content)
            formatted_lesson = format_for_whatsapp(lesson_content, user.age)
            
            logger.info(f"Generated base LLM lesson for user {user.phone_number} on topic: {topic}")
            
            result_text = f"📚 *Lesson: {clean_topic_title(topic)}*\n\n{formatted_lesson}\n\n_Type `/next` for more on this topic or `/lesson <new topic>` for something else!_"
            
            if for_audio:
                return result_text
            
            # For text lessons, add image (same helper as /next)
            result = {"text": result_text}
            _attach_lesson_image(result, topic, user.language, "lesson")
            return result
        
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

            # logger.info(f"Retrieved chunks: {retrieved_chunks}")
            
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

def process_whatsapp_message(db: Session, phone_number: str, message: str, for_audio: bool = False):
    """Process a WhatsApp message. When for_audio is True, LLM/RAG prompts ask for spoken-style output.
    Returns str for audio mode, dict with {text, image_bytes?, image_content_type?} for text mode."""
    return message_handler.process_message(db, phone_number, message, for_audio=for_audio)


def process_whatsapp_message_request_audio(db: Session, phone_number: str, message: str) -> dict:
    """
    Handle /audio <topic> or /audio next: run lesson/next with for_audio=True, run TTS, return
    dict with same shape as process_whatsapp_audio: {text, audio_segments?, tts_failed?, ...}.
    Used when user sends a text message that explicitly requests audio.
    """
    msg = message.strip()
    msg_lower = msg.lower()
    if msg_lower == "/audio next" or msg_lower.startswith("/audio next"):
        response_text = process_whatsapp_message(db, phone_number, "/next", for_audio=True)
    elif msg_lower.startswith("/audio "):
        topic = msg[7:].strip()
        if not topic:
            return {"text": "Please specify a topic. Try /audio photosynthesis or /audio cells. 📚"}
        response_text = process_whatsapp_message(db, phone_number, f"/lesson {topic}", for_audio=True)
    else:
        return {"text": "Use /audio <topic> for an audio lesson or /audio next for the next part. 📚"}

    result = {"text": response_text}
    user = get_user_by_phone(db, phone_number)
    if not user:
        user = create_user(db, phone_number)
    if not user.is_onboarded:
        return result
    # Set lesson title for audio header (e.g. "📚 Lesson: Microbes")
    if msg_lower == "/audio next" or msg_lower.startswith("/audio next"):
        current_lesson = get_current_lesson(db, user.id)
        result["lesson_title"] = clean_topic_title(current_lesson.topic) if current_lesson else "Next part"
    elif msg_lower.startswith("/audio "):
        topic = msg[7:].strip()
        if topic:
            result["lesson_title"] = clean_topic_title(topic)
    # Don't synthesize error messages as audio - send as text only
    if response_text.strip().lower().startswith("sorry,") or "trouble creating" in response_text.lower() or "trouble preparing" in response_text.lower():
        result["tts_failed"] = True
        logger.info("Skipping TTS for error response; sending as text")
        return result
    try:
        voice = tts_service.get_voice_for_age(user.age if user.age else 10, user.language)
        age = user.age if user.age else 10
        segments = synthesize_speech_chunked(response_text, voice, age, language=user.language)
        if segments:
            result["audio_segments"] = segments
            if len(segments) == 1:
                result["audio_bytes"] = segments[0][0]
                result["audio_content_type"] = segments[0][1]
            logger.info(f"/audio response: {len(segments)} segment(s)")
        else:
            result["tts_failed"] = True
            logger.warning("TTS failed for /audio request; text backup will be sent")
    except Exception as e:
        result["tts_failed"] = True
        logger.error(f"TTS failed for /audio request: {e}; text backup will be sent")
    return result


async def process_whatsapp_audio(
    db: Session, 
    phone_number: str, 
    media_url: str, 
    content_type: str,
    return_audio: bool = True,
    twilio_account_sid: str = None,
    twilio_auth_token: str = None,
    twilio_client = None
) -> dict:
    """
    Process incoming audio message from WhatsApp.
    
    Args:
        db: Database session
        phone_number: User's phone number
        media_url: URL to fetch audio from Twilio
        content_type: MIME type of the audio
        return_audio: Whether to return audio response (TTS) or just text
    
    Returns:
        Dictionary with 'text' and optionally 'audio_bytes' and 'audio_content_type'
    """
    import requests
    
    try:
        logger.info(f"Processing audio message from {phone_number}")
        
        # Fetch audio file from Twilio (requires authentication)
        logger.info(f"Fetching audio from {media_url}")
        
        import os
        from twilio.rest import Client as TwilioClient
        
        audio_data = None
        
        # Method 1: Use provided Twilio client (preferred)
        if twilio_client:
            try:
                logger.info("Fetching media using Twilio client...")
                # Extract Message SID and Media SID from URL
                # URL format: .../Accounts/{AC}/Messages/{MM}/Media/{ME}...
                parts = media_url.split('/')
                message_sid = None
                media_sid = None
                
                for i, part in enumerate(parts):
                    if part == 'Messages' and i + 1 < len(parts):
                        message_sid = parts[i + 1]
                    if part == 'Media' and i + 1 < len(parts):
                        media_sid = parts[i + 1].split('/')[0]
                
                if message_sid and media_sid:
                    # Fetch media using Twilio client
                    media = twilio_client.messages(message_sid).media(media_sid).fetch()
                    # Get content URL (remove .json extension if present)
                    # media.uri might be a relative path, so we need to construct the full URL
                    if media.uri.startswith('http'):
                        content_url = media.uri.replace('.json', '')
                    else:
                        # Construct full URL from the base API URL
                        content_url = f"https://api.twilio.com{media.uri.replace('.json', '')}"
                    
                    if not twilio_account_sid:
                        twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
                    if not twilio_auth_token:
                        twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
                    audio_data = requests.get(
                        content_url,
                        auth=(twilio_account_sid, twilio_auth_token),
                        timeout=30
                    ).content
                    logger.info(f"Successfully fetched media using Twilio client: {len(audio_data)} bytes")
            except Exception as client_err:
                logger.warning(f"Failed to fetch using Twilio client: {client_err}, trying direct HTTP...")
        
        # Method 2: Direct HTTP request with credentials
        if not audio_data:
            if not twilio_account_sid:
                twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            if not twilio_auth_token:
                twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            
            if not twilio_account_sid or not twilio_auth_token:
                logger.error("Twilio credentials not found. Cannot fetch audio media.")
                return {
                    'text': "Sorry, I couldn't access your audio message. Please try sending a text message! 🎤"
                }         
            
            try:
                response = requests.get(
                    media_url, 
                    auth=(twilio_account_sid, twilio_auth_token),
                    timeout=30
                )
                response.raise_for_status()
                audio_data = response.content
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error fetching audio: {e}")
                logger.error(f"Response status: {response.status_code}")
                logger.error(f"Response text: {response.text[:200]}")
                raise
        
        logger.info(f"Audio fetched: {len(audio_data)} bytes, type: {content_type}")
        
        # Transcribe audio to text
        logger.info("Transcribing audio...")
        user = get_user_by_phone(db, phone_number)
        if not user:
            user = create_user(db, phone_number)
        user_language = user.language if user else "en"
        transcribed_text = transcribe_audio(audio_data, content_type, language=user_language)
        
        if not transcribed_text:
            logger.error("Failed to transcribe audio")
            return {
                'text': "Sorry, I couldn't understand your voice message. Could you please send it as text? 🎤"
            }
        
        logger.info(f"Transcription successful: {transcribed_text}")
        
        # Check if transcription matches expected voice format
        transcribed_lower = transcribed_text.lower().strip()
        voice_format_commands = ['teach me about ', 'lesson ', 'next', 'quiz', 'help', 'progress', 'review']
        uses_voice_format = any(transcribed_lower.startswith(cmd) for cmd in voice_format_commands)
        
        if not uses_voice_format and user.is_onboarded:
            logger.info("Voice message doesn't use recommended format 'teach me about <topic>', will try keyword fallback")
        
        # Send loading message for voice commands (after transcription, before processing)
        if twilio_client and uses_voice_format:
            try:
                import os
                twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")
                if twilio_phone:
                    from_twilio = f"whatsapp:{twilio_phone}" if not twilio_phone.startswith("whatsapp:") else twilio_phone
                    to_user = f"whatsapp:{phone_number}" if not phone_number.startswith("whatsapp:") else phone_number
                    if not to_user.startswith("whatsapp:+"):
                        to_user = to_user.replace("whatsapp:", "whatsapp:+")
                    
                    from .utils import get_loading_message
                    loading_text = None
                    if transcribed_lower.startswith("teach me about ") or transcribed_lower.startswith("lesson "):
                        topic = transcribed_text[len("teach me about " if transcribed_lower.startswith("teach me about ") else "lesson "):].strip()
                        if topic:
                            loading_text = get_loading_message("lesson", topic, user_language)
                    elif transcribed_lower.strip() == "next" or transcribed_lower.startswith("next "):
                        loading_text = get_loading_message("next", None, user_language)
                    elif transcribed_lower.startswith("quiz"):
                        loading_text = get_loading_message("quiz", None, user_language)
                    elif transcribed_lower.startswith("progress") or transcribed_lower.startswith("review"):
                        loading_text = get_loading_message("progress", None, user_language)
                    
                    if loading_text:
                        twilio_client.messages.create(
                            from_=from_twilio,
                            to=to_user,
                            body=loading_text,
                        )
                        logger.info(f"Sent loading message for voice: {loading_text}")
            except Exception as load_err:
                logger.warning(f"Failed to send loading message for voice: {load_err}")
        
        for_audio = return_audio and user.is_onboarded
        response_text = process_whatsapp_message(db, phone_number, transcribed_text, for_audio=for_audio)
        
        result = {'text': response_text}
        # Don't synthesize error messages as audio - send as text only
        if response_text.strip().lower().startswith("sorry,") or "trouble creating" in response_text.lower() or "trouble preparing" in response_text.lower():
            result["tts_failed"] = True
            logger.info("Skipping TTS for error response (voice); sending as text")
            return result
        # Set lesson title for audio header (e.g. "📚 Lesson: Microbes")
        if transcribed_lower.startswith("teach me about "):
            topic = transcribed_text[len("teach me about "):].strip()
            if topic:
                result["lesson_title"] = clean_topic_title(topic)
        elif transcribed_lower.startswith("lesson "):
            topic = transcribed_text[len("lesson "):].strip()
            if topic:
                result["lesson_title"] = clean_topic_title(topic)
        elif transcribed_lower.strip() == "next" or transcribed_lower.startswith("next "):
            current_lesson = get_current_lesson(db, user.id)
            result["lesson_title"] = clean_topic_title(current_lesson.topic) if current_lesson else "Next part"
        # Generate audio response if requested and user is onboarded (chunked so full paragraph is sent as multiple voice notes)
        # Fallback: result['text'] is always set above — when TTS fails, caller should send text instead.
        if return_audio and user.is_onboarded:
            try:
                voice = tts_service.get_voice_for_age(user.age if user.age else 10, user.language)
                age = user.age if user.age else 10
                logger.info(f"Generating chunked audio response (voice: {voice}, age: {age}, language: {user.language})...")
                segments = synthesize_speech_chunked(response_text, voice, age, language=user.language)
                if segments:
                    result['audio_segments'] = segments
                    if len(segments) == 1:
                        result['audio_bytes'] = segments[0][0]
                        result['audio_content_type'] = segments[0][1]
                    logger.info(f"Audio response: {len(segments)} segment(s)")
                else:
                    result['tts_failed'] = True
                    logger.warning("TTS failed (no segments); text backup will be sent instead")
            except Exception as e:
                result['tts_failed'] = True
                logger.error(f"Error generating audio response: {e}; text backup will be sent instead")
        
        return result
        
    except requests.RequestException as e:
        logger.error(f"Error fetching audio from Twilio: {e}")
        return {
            'text': "Sorry, I couldn't download your audio message. Please try again! 🎤"
        }
    except Exception as e:
        logger.error(f"Error processing audio message: {e}")
        return {
            'text': "Sorry, I'm having trouble processing your voice message. Please try sending a text message! 🎤"
        }

if __name__ == "__main__":
    print("Message handlers module loaded successfully!")
