import re
import json
from typing import List, Dict, Any, Optional


def format_for_whatsapp(text: str, age_group: int) -> str:
    
    formatted_text = apply_whatsapp_formatting(text)
    
    formatted_text = improve_readability(formatted_text)
    
    return formatted_text


    
def apply_whatsapp_formatting(text: str) -> str:
    text = re.sub(r'\b([A-Z]{2,})\b', r'*\1*', text)
    
    key_terms = ['definition', 'important', 'remember', 'key point', 'note']
    for term in key_terms:
        text = re.sub(f'({term})', r'*\1*', text, flags=re.IGNORECASE)
    
    text = re.sub(r'(Example:.*?)(\n|$)', r'_\1_\2', text, flags=re.IGNORECASE)
    text = re.sub(r'(Practice:.*?)(\n|$)', r'_\1_\2', text, flags=re.IGNORECASE)
    
    return text


def improve_readability(text: str) -> str:
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    
    text = re.sub(r'(👉\s*Practice:)', r'\n\1', text)
    
    text = re.sub(r'(Think of it|Imagine|Remember)', r'\n\1', text)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def validate_age(age_input: str) -> Optional[int]:
    try:
        age = int(age_input.strip())
        if 3 <= age <= 100:
            return age
        return None
    except ValueError:
        return None


def validate_subjects(subjects_input: str) -> List[str]:
    if not subjects_input:
        return []
    
    subject_mapping = {
        'math': 'Mathematics',
        'maths': 'Mathematics',
        'mathematics': 'Mathematics',
        'science': 'Science',
        'english': 'English',
        'history': 'History',
        'geography': 'Geography',
        'physics': 'Physics',
        'chemistry': 'Chemistry',
        'biology': 'Biology',
        'literature': 'Literature',
        'art': 'Art',
        'music': 'Music',
        'pe': 'Physical Education',
        'sports': 'Sports',
        'computer': 'Computer Science',
        'programming': 'Programming',
        'coding': 'Programming',
    }
    
    subjects = []
    for subject in subjects_input.split(','):
        subject = subject.strip().lower()
        if subject in subject_mapping:
            subjects.append(subject_mapping[subject])
        elif len(subject) > 2:
            subjects.append(subject.title())
    
    return subjects[:10]


def validate_country(country_input: str) -> Optional[str]:
    if not country_input or len(country_input.strip()) < 2:
        return None
    
    country = country_input.strip().title()
    
    if re.match(r'^[A-Za-z\s\'-]+$', country):
        return country
    
    return None


def validate_learning_mode(mode_input: str) -> Optional[str]:
    mode = mode_input.strip().lower()
    
    if mode in ['text', 'reading', 'written']:
        return 'text'
    elif mode in ['audio', 'voice', 'spoken', 'listening']:
        return 'audio'
    
    return None


def get_help_message(age_group: int) -> str:
    base_commands = """
🤖 *WhatsApp AI Tutor Commands*

📚 `/lesson <topic>` - Get a lesson on any topic
➡️ `/next` - Continue to next part of lesson
❓ `/help` - Show this help message

*Examples:*
• /lesson fractions
• /lesson photosynthesis  
• /lesson world war 2
"""
    
    if age_group <= 8:
        additional = """
🌟 *Tips for little learners:*
• Ask about anything you're curious about!
• Try topics like: animals, colors, shapes, numbers
• I'll make it super fun and easy! 🎉
"""
    elif age_group <= 12:
        additional = """
📖 *Study Tips:*
• Try school subjects: math, science, history
• Ask about homework topics
• Practice questions help you learn better! ✏️
"""
    elif age_group <= 16:
        additional = """
🎓 *Study Smart:*
• Get help with exam topics
• Ask for explanations of difficult concepts
• Perfect for homework and test prep 📝
"""
    else:
        additional = """
💼 *Professional Learning:*
• Explore any topic of interest
• Get clear, structured explanations
• Perfect for skill development and knowledge growth 📈
"""
    
    return format_for_whatsapp(base_commands + additional, age_group)


def parse_lesson_command(message: str) -> Optional[str]:
    match = re.match(r'/lesson\s+(.+)', message.strip(), re.IGNORECASE)
    if match:
        topic = match.group(1).strip()
        return topic
    
    return None


def get_greeting_emoji(age_group: int) -> str:
    if age_group <= 8:
        return "🌟"
    elif age_group <= 12:
        return "📚"
    elif age_group <= 16:
        return "🎓"
    else:
        return "👋"


def store_subjects_as_json(subjects: List[str]) -> str:
    return json.dumps(subjects)


if __name__ == "__main__":
    test_text = "Let me teach you about fractions. Practice: If you have 12 apples and eat 6, what fraction did you eat?"
    
    print("Testing formatting for different ages:")
    for age in [6, 10, 14, 25]:
        print(f"\nAge {age}:")
        formatted = format_for_whatsapp(test_text, age)
        print(formatted)
        print("-" * 50)
