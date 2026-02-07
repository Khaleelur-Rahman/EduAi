import re
import json
from typing import List, Dict, Any, Optional


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output (Qwen/Cerebras thinking tokens)."""
    if not text or not text.strip():
        return text
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()


def clean_whatsapp_formatting(text: str) -> str:
    """Clean up formatting issues in WhatsApp messages."""
    # Replace double asterisks with single asterisks (WhatsApp uses single * for bold)
    # Handle cases like **text** or **text* or *text**
    text = re.sub(r'\*\*([^*]+)\*\*', r'*\1*', text)  # **text** -> *text*
    text = re.sub(r'\*\*([^*]+)\*', r'*\1*', text)   # **text* -> *text*
    text = re.sub(r'\*([^*]+)\*\*', r'*\1*', text)   # *text** -> *text*
    
    # Remove standalone double asterisks
    text = re.sub(r'\*\*+', '', text)
    
    # Remove "Try This at Home" sections that are generic/unrelated
    # Match the pattern and everything until the next section (marked by _Type or double newline)
    text = re.sub(
        r'[\*\s]*Try This at Home[!*]*[\*\s]*.*?(?=\n\n|\n_|_Type|$)', 
        '', 
        text, 
        flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(
        r'[\*\s]*Try This[!*]*[\*\s]*.*?(?=\n\n|\n_|_Type|$)', 
        '', 
        text, 
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Clean up any remaining formatting artifacts
    text = re.sub(r'\*{3,}', '*', text)  # Replace 3+ asterisks with single
    text = re.sub(r'\s+\*+\s+', ' ', text)  # Remove isolated asterisks with spaces
    text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
    
    return text.strip()

def format_for_whatsapp(text: str, age_group: int) -> str:
    # Strip LLM thinking/reasoning blocks (e.g. <think>...</think>) before presenting to user
    text = strip_think_tags(text)
    # Clean up formatting issues
    text = clean_whatsapp_formatting(text)
    
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
🤖 *EduBot Commands*

📚 `/lesson <topic>` - Get a lesson on any topic. (e.g. `/lesson cells`)
➡️ `/next` - Continue to next part of lesson
🧩 `/quiz` - Take a quiz on your current lesson
❓ `/help` - Show this help message

🎤 *Voice Messages:*
For voice messages, use this format:
• Say "Teach me about <topic>" (e.g., "Teach me about cells")
• Say "Next" to continue
• Say "Quiz" for a quiz
• Say "Help" for help

"""
    
    if age_group <= 8:
        additional = """
🌟 *Tips for little learners:*
• Ask about anything you're curious about!
• Try science topics like: plants, animals, weather
• Use voice messages! Say "teach me about plants" 🎤
• I'll make it super fun and easy! 🎉
"""
    elif age_group <= 12:
        additional = """
📖 *Study Tips:*
• Try science topics: plants, solar system, energy, weather
• Ask about homework topics
• Use voice messages! Say "teach me about <topic>" 🎤
• Practice questions help you learn better! ✏️
"""
    elif age_group <= 16:
        additional = """
🎓 *Study Smart:*
• Get help with exam topics
• Ask for explanations of difficult concepts
• Use voice messages for quick questions! 🎤
• Perfect for homework and test prep 📝
"""
    else:
        additional = """
💼 *Professional Learning:*
• Explore any topic of interest
• Get clear, structured explanations
• Use voice messages for hands-free learning! 🎤
• Perfect for skill development and knowledge growth 📈
"""
    
    return format_for_whatsapp(base_commands + additional, age_group)


def parse_lesson_command(message: str) -> Optional[str]:
    """Parse lesson command from text or voice input.
    Supports both text format (/lesson <topic>) and voice format (lesson <topic>).
    """
    message = message.strip()
    
    # Try text format first: /lesson <topic>
    match = re.match(r'/lesson\s+(.+)', message, re.IGNORECASE)
    if match:
        topic = match.group(1).strip()
        return topic
    
    # Try voice-friendly format: lesson <topic> (without slash)
    match = re.match(r'^lesson\s+(.+)', message, re.IGNORECASE)
    if match:
        topic = match.group(1).strip()
        return topic
    
    return None

def clean_topic_title(topic: str) -> str:
    """Clean topic title by removing trailing punctuation and formatting properly."""
    if not topic:
        return topic
    
    topic = topic.rstrip('.,!?;:')
    topic = topic.title()
    
    return topic


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
