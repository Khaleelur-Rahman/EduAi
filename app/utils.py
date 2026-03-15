import re
import json
from typing import List, Dict, Any, Optional
from .language import SUPPORTED_LANGUAGES, get_language_name


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output (Qwen/Cerebras thinking tokens)."""
    if not text or not text.strip():
        return text
    # Remove complete <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove unclosed <think>... at start (e.g. if response was cut off)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


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
    
    # Match "Example:" or "Practice:" only when they appear as standalone labels
    # Require them to be at start of line, after newline, or after punctuation
    # This avoids matching when part of phrases like "Fun example:" or "great example:"
    text = re.sub(r'(^|\n|[.!?]\s+)(Example:.*?)(\n|$)', r'\1_\2_\3', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'(^|\n|[.!?]\s+)(Practice:.*?)(\n|$)', r'\1_\2_\3', text, flags=re.IGNORECASE | re.MULTILINE)
    
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


def get_help_message(age_group: int, language: str = "en") -> str:
    base_commands = """
🤖 *EduBot Commands*

📚 `/lesson <topic>` - Get a lesson on any topic. (e.g. `/lesson cells`)
➡️ `/next` - Continue to next part of a text or audio lesson 
🧩 `/quiz` - Take a quiz on your current lesson
🎤 `/audio <topic>` - Get an audio lesson (e.g. `/audio cells`)
📹 `/video <topic>` - Get a short educational video (e.g. `/video cells`)
🌐 `/language <code>` - Change language (e.g. `/language es` for Spanish)
📊 `/progress` - See your completed lessons and quiz scores
❓ `/help` - Show this help message

🎤 *Voice Messages:*
For voice messages, use this format:
• Say "Teach me about <topic>" (e.g., "Teach me about cells")
• Say "Next" to continue
• Say "Quiz" for a quiz
• Say "Progress" for your progress
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
    
    # Translate messages based on language
    if language != "en":
        base_commands = _translate_help_message(base_commands, language)
        additional = _translate_help_message(additional, language)
    
    return format_for_whatsapp(base_commands + additional, age_group)


def _translate_help_message(text: str, language: str) -> str:
    """Translate help message to target language."""
    translations = {
        "es": {
            "EduBot Commands": "Comandos de EduBot",
            "/lesson <topic>": "/lección <tema>",
            "Get a lesson on any topic.": "Obtén una lección sobre cualquier tema.",
            "Continue to next part of lesson": "Continuar a la siguiente parte de la lección",
            "Take a quiz on your current lesson": "Hacer un cuestionario sobre tu lección actual",
            "Get an audio lesson (e.g. `/audio cells`)": "Obtén una lección en audio (ej. `/audio células`)",
            "Change language (e.g. `/language es` for Spanish)": "Cambiar idioma (ej. `/idioma es` para español)",
            "/language <code>": "/idioma <código>",
            "See your completed lessons and quiz scores": "Ver tus lecciones completadas y puntuaciones",
            "/audio <topic>": "/audio <tema>",
            "/language <code>": "/idioma <código>",
            "Change language (e.g. `/language es` for Spanish)": "Cambiar idioma (ej. `/idioma es` para español)",
            "/progress": "/progreso",
            'Say "Progress" for your progress': 'Di "Progreso" para ver tu progreso',
            "Show this help message": "Mostrar este mensaje de ayuda",
            "Voice Messages:": "Mensajes de Voz:",
            "For voice messages, use this format:": "Para mensajes de voz, usa este formato:",
            'Say "Teach me about <topic>"': 'Di "Enséñame sobre <tema>"',
            'Say "Next" to continue': 'Di "Siguiente" para continuar',
            'Say "Quiz" for a quiz': 'Di "Cuestionario" para un cuestionario',
            'Say "Help" for help': 'Di "Ayuda" para ayuda',
            "Tips for little learners:": "Consejos para pequeños aprendices:",
            "Ask about anything you're curious about!": "¡Pregunta sobre cualquier cosa que te cause curiosidad!",
            "Try science topics like:": "Prueba temas de ciencia como:",
            "Use voice messages!": "¡Usa mensajes de voz!",
            "I'll make it super fun and easy!": "¡Lo haré súper divertido y fácil!",
            "Study Tips:": "Consejos de Estudio:",
            "Ask about homework topics": "Pregunta sobre temas de tarea",
            "Practice questions help you learn better!": "¡Las preguntas de práctica te ayudan a aprender mejor!",
            "Study Smart:": "Estudia Inteligentemente:",
            "Get help with exam topics": "Obtén ayuda con temas de examen",
            "Ask for explanations of difficult concepts": "Pide explicaciones de conceptos difíciles",
            "Perfect for homework and test prep": "Perfecto para tareas y preparación de exámenes",
            "Professional Learning:": "Aprendizaje Profesional:",
            "Explore any topic of interest": "Explora cualquier tema de interés",
            "Get clear, structured explanations": "Obtén explicaciones claras y estructuradas",
            "Use voice messages for hands-free learning!": "¡Usa mensajes de voz para aprender sin usar las manos!",
            "Perfect for skill development and knowledge growth": "Perfecto para el desarrollo de habilidades y crecimiento del conocimiento",
        },
        "fr": {
            "EduBot Commands": "Commandes EduBot",
            "/lesson <topic>": "/leçon <sujet>",
            "Get a lesson on any topic.": "Obtenez une leçon sur n'importe quel sujet.",
            "Continue to next part of lesson": "Continuer à la partie suivante de la leçon",
            "Take a quiz on your current lesson": "Faire un quiz sur votre leçon actuelle",
            "Get an audio lesson (e.g. `/audio cells`)": "Obtenez une leçon audio (ex. `/audio cellules`)",
            "Change language (e.g. `/language es` for Spanish)": "Changer la langue (ex. `/langue es` pour espagnol)",
            "/language <code>": "/langue <code>",
            "See your completed lessons and quiz scores": "Voir vos leçons terminées et scores de quiz",
            "/audio <topic>": "/audio <sujet>",
            "/language <code>": "/langue <code>",
            "Change language (e.g. `/language es` for Spanish)": "Changer la langue (ex. `/langue es` pour espagnol)",
            "/progress": "/progression",
            'Say "Progress" for your progress': 'Dites "Progression" pour voir votre progression',
            "Show this help message": "Afficher ce message d'aide",
            "Voice Messages:": "Messages Vocaux:",
            "For voice messages, use this format:": "Pour les messages vocaux, utilisez ce format:",
            'Say "Teach me about <topic>"': 'Dites "Apprends-moi sur <sujet>"',
            'Say "Next" to continue': 'Dites "Suivant" pour continuer',
            'Say "Quiz" for a quiz': 'Dites "Quiz" pour un quiz',
            'Say "Help" for help': 'Dites "Aide" pour l\'aide',
            "Tips for little learners:": "Conseils pour les petits apprenants:",
            "Ask about anything you're curious about!": "Demandez tout ce qui vous intrigue!",
            "Try science topics like:": "Essayez des sujets scientifiques comme:",
            "Use voice messages!": "Utilisez les messages vocaux!",
            "I'll make it super fun and easy!": "Je vais le rendre super amusant et facile!",
            "Study Tips:": "Conseils d'Étude:",
            "Ask about homework topics": "Demandez des sujets de devoirs",
            "Practice questions help you learn better!": "Les questions pratiques vous aident à mieux apprendre!",
            "Study Smart:": "Étudiez Intelligemment:",
            "Get help with exam topics": "Obtenez de l'aide sur les sujets d'examen",
            "Ask for explanations of difficult concepts": "Demandez des explications de concepts difficiles",
            "Perfect for homework and test prep": "Parfait pour les devoirs et la préparation aux tests",
            "Professional Learning:": "Apprentissage Professionnel:",
            "Explore any topic of interest": "Explorez n'importe quel sujet d'intérêt",
            "Get clear, structured explanations": "Obtenez des explications claires et structurées",
            "Use voice messages for hands-free learning!": "Utilisez les messages vocaux pour apprendre sans les mains!",
            "Perfect for skill development and knowledge growth": "Parfait pour le développement des compétences et la croissance des connaissances",
        },
        "ms": {
            "EduBot Commands": "Arahan EduBot",
            "/lesson <topic>": "/pelajaran <topik>",
            "Get a lesson on any topic.": "Dapatkan pelajaran tentang mana-mana topik.",
            "Continue to next part of lesson": "Teruskan ke bahagian seterusnya pelajaran",
            "Take a quiz on your current lesson": "Ambil kuiz tentang pelajaran semasa anda",
            "Get an audio lesson (e.g. `/audio cells`)": "Dapatkan pelajaran audio (cth. `/audio sel`)",
            "Change language (e.g. `/language es` for Spanish)": "Tukar bahasa (cth. `/bahasa es` untuk Sepanyol)",
            "/language <code>": "/bahasa <kod>",
            "See your completed lessons and quiz scores": "Lihat pelajaran dan skor kuiz anda",
            "/audio <topic>": "/audio <tajuk>",
            "/language <code>": "/bahasa <kod>",
            "Change language (e.g. `/language es` for Spanish)": "Tukar bahasa (cth. `/bahasa es` untuk Sepanyol)",
            "/progress": "/kemajuan",
            'Say "Progress" for your progress': 'Katakan "Kemajuan" untuk kemajuan anda',
            "Show this help message": "Tunjukkan mesej bantuan ini",
            "Voice Messages:": "Mesej Suara:",
            "For voice messages, use this format:": "Untuk mesej suara, gunakan format ini:",
            'Say "Teach me about <topic>"': 'Katakan "Ajar saya tentang <topik>"',
            'Say "Next" to continue': 'Katakan "Seterusnya" untuk meneruskan',
            'Say "Quiz" for a quiz': 'Katakan "Kuiz" untuk kuiz',
            'Say "Help" for help': 'Katakan "Bantuan" untuk bantuan',
            "Tips for little learners:": "Petua untuk pelajar kecil:",
            "Ask about anything you're curious about!": "Tanya tentang apa sahaja yang anda ingin tahu!",
            "Try science topics like:": "Cuba topik sains seperti:",
            "Use voice messages!": "Gunakan mesej suara!",
            "I'll make it super fun and easy!": "Saya akan menjadikannya sangat menyeronokkan dan mudah!",
            "Study Tips:": "Petua Belajar:",
            "Ask about homework topics": "Tanya tentang topik kerja rumah",
            "Practice questions help you learn better!": "Soalan latihan membantu anda belajar dengan lebih baik!",
            "Study Smart:": "Belajar dengan Bijak:",
            "Get help with exam topics": "Dapatkan bantuan dengan topik peperiksaan",
            "Ask for explanations of difficult concepts": "Minta penjelasan tentang konsep yang sukar",
            "Perfect for homework and test prep": "Sempurna untuk kerja rumah dan persediaan ujian",
            "Professional Learning:": "Pembelajaran Profesional:",
            "Explore any topic of interest": "Terokai mana-mana topik yang menarik minat",
            "Get clear, structured explanations": "Dapatkan penjelasan yang jelas dan terstruktur",
            "Use voice messages for hands-free learning!": "Gunakan mesej suara untuk pembelajaran tanpa tangan!",
            "Perfect for skill development and knowledge growth": "Sempurna untuk pembangunan kemahiran dan pertumbuhan pengetahuan",
        },
        "zh": {
            "EduBot Commands": "EduBot 命令",
            "/lesson <topic>": "/课程 <主题>",
            "Get a lesson on any topic.": "获取任何主题的课程。",
            "Continue to next part of lesson": "继续课程的下一个部分",
            "Take a quiz on your current lesson": "对你当前的课程进行测验",
            "Get an audio lesson (e.g. `/audio cells`)": "获取音频课程（如 `/audio 细胞`）",
            "Change language (e.g. `/language es` for Spanish)": "更改语言（例如 `/语言 es` 表示西班牙语）",
            "/language <code>": "/语言 <代码>",
            "See your completed lessons and quiz scores": "查看你完成的课程和测验成绩",
            "/audio <topic>": "/audio <主题>",
            "/language <code>": "/语言 <代码>",
            "Change language (e.g. `/language es` for Spanish)": "更改语言（例如 `/语言 es` 表示西班牙语）",
            "/progress": "/进度",
            'Say "Progress" for your progress': '说"进度"查看进度',
            "Show this help message": "显示此帮助消息",
            "Voice Messages:": "语音消息：",
            "For voice messages, use this format:": "对于语音消息，请使用此格式：",
            'Say "Teach me about <topic>"': '说"教我关于<主题>"',
            'Say "Next" to continue': '说"下一个"继续',
            'Say "Quiz" for a quiz': '说"测验"进行测验',
            'Say "Help" for help': '说"帮助"获取帮助',
            "Tips for little learners:": "给小学习者的提示：",
            "Ask about anything you're curious about!": "询问任何你好奇的事情！",
            "Try science topics like:": "尝试科学主题，如：",
            "Use voice messages!": "使用语音消息！",
            "I'll make it super fun and easy!": "我会让它超级有趣和简单！",
            "Study Tips:": "学习提示：",
            "Ask about homework topics": "询问作业主题",
            "Practice questions help you learn better!": "练习题帮助你更好地学习！",
            "Study Smart:": "聪明学习：",
            "Get help with exam topics": "获取考试主题的帮助",
            "Ask for explanations of difficult concepts": "请求解释困难的概念",
            "Perfect for homework and test prep": "非常适合作业和考试准备",
            "Professional Learning:": "专业学习：",
            "Explore any topic of interest": "探索任何感兴趣的主题",
            "Get clear, structured explanations": "获得清晰、结构化的解释",
            "Use voice messages for hands-free learning!": "使用语音消息进行免提学习！",
            "Perfect for skill development and knowledge growth": "非常适合技能发展和知识增长",
        },
        "hi": {
            "EduBot Commands": "EduBot कमांड",
            "/lesson <topic>": "/पाठ <विषय>",
            "Get a lesson on any topic.": "किसी भी विषय पर पाठ प्राप्त करें।",
            "Continue to next part of lesson": "पाठ के अगले भाग पर जारी रखें",
            "Take a quiz on your current lesson": "अपने वर्तमान पाठ पर क्विज़ लें",
            "Get an audio lesson (e.g. `/audio cells`)": "ऑडियो पाठ प्राप्त करें (जैसे `/audio कोशिकाएं`)",
            "Change language (e.g. `/language es` for Spanish)": "भाषा बदलें (उदा. `/भाषा es` स्पेनिश के लिए)",
            "/language <code>": "/भाषा <कोड>",
            "See your completed lessons and quiz scores": "अपने पूर्ण पाठ और क्विज़ स्कोर देखें",
            "/audio <topic>": "/audio <विषय>",
            "/language <code>": "/भाषा <कोड>",
            "Change language (e.g. `/language es` for Spanish)": "भाषा बदलें (उदा. `/भाषा es` स्पेनिश के लिए)",
            "/progress": "/प्रगति",
            'Say "Progress" for your progress': 'प्रगति के लिए "प्रगति" कहें',
            "Show this help message": "यह सहायता संदेश दिखाएं",
            "Voice Messages:": "आवाज़ संदेश:",
            "For voice messages, use this format:": "आवाज़ संदेश के लिए, इस प्रारूप का उपयोग करें:",
            'Say "Teach me about <topic>"': '"<विषय> के बारे में सिखाएं" कहें',
            'Say "Next" to continue': 'जारी रखने के लिए "अगला" कहें',
            'Say "Quiz" for a quiz': 'क्विज़ के लिए "क्विज़" कहें',
            'Say "Help" for help': 'सहायता के लिए "सहायता" कहें',
            "Tips for little learners:": "छोटे शिक्षार्थियों के लिए सुझाव:",
            "Ask about anything you're curious about!": "किसी भी चीज़ के बारे में पूछें जिसके बारे में आप उत्सुक हैं!",
            "Try science topics like:": "विज्ञान विषयों को आज़माएं जैसे:",
            "Use voice messages!": "आवाज़ संदेश का उपयोग करें!",
            "I'll make it super fun and easy!": "मैं इसे बहुत मजेदार और आसान बनाऊंगा!",
            "Study Tips:": "अध्ययन सुझाव:",
            "Ask about homework topics": "होमवर्क विषयों के बारे में पूछें",
            "Practice questions help you learn better!": "अभ्यास प्रश्न आपको बेहतर सीखने में मदद करते हैं!",
            "Study Smart:": "स्मार्ट अध्ययन:",
            "Get help with exam topics": "परीक्षा विषयों पर सहायता प्राप्त करें",
            "Ask for explanations of difficult concepts": "कठिन अवधारणाओं की व्याख्या मांगें",
            "Perfect for homework and test prep": "होमवर्क और परीक्षा की तैयारी के लिए परफेक्ट",
            "Professional Learning:": "पेशेवर सीखना:",
            "Explore any topic of interest": "किसी भी रुचि के विषय का अन्वेषण करें",
            "Get clear, structured explanations": "स्पष्ट, संरचित स्पष्टीकरण प्राप्त करें",
            "Use voice messages for hands-free learning!": "हैंड्स-फ्री सीखने के लिए आवाज़ संदेश का उपयोग करें!",
            "Perfect for skill development and knowledge growth": "कौशल विकास और ज्ञान वृद्धि के लिए परफेक्ट",
        },
    }
    
    if language not in translations:
        return text
    
    translated = text
    for en_text, translated_text in translations[language].items():
        translated = translated.replace(en_text, translated_text)
    
    return translated


def format_progress_review(
    lessons: list,
    quizzes: list,
    language: str = "en",
    dashboard_url: Optional[str] = None,
    unique_topics: Optional[int] = None,
    total_parts: Optional[int] = None,
) -> str:
    """Format progress review message for WhatsApp. lessons and quizzes are ORM objects."""
    import json

    t = _get_progress_translations(language)
    if not lessons and not quizzes:
        msg = f"📊 *{t['your_progress']}*\n\n{t['no_progress']}"
        if dashboard_url:
            msg += f"\n\n{t['view_dashboard']}: {dashboard_url}"
        return msg
    lines = [f"📊 *{t['your_progress']}*", ""]
    if unique_topics is not None and total_parts is not None:
        lines.append(t["topics_and_parts"].format(topics=unique_topics, parts=total_parts))
        lines.append(t["no_fixed_end"])
        lines.append("")

    # Lessons section - deduplicate by (topic, lesson_step)
    lines.append(f"📚 *{t['lessons']}:*")
    if not lessons:
        lines.append(f"• {t['no_lessons_yet']}")
    else:
        # Deduplicate lessons by (topic.lower(), lesson_step)
        seen_lessons = set()
        unique_lessons = []
        for p in lessons:
            key = (p.topic.lower(), p.lesson_step)
            if key not in seen_lessons:
                seen_lessons.add(key)
                unique_lessons.append(p)
        # Show only top 8 unique lessons
        for p in unique_lessons[:8]:
            title = clean_topic_title(p.topic)
            if getattr(p, "completed", False):
                lines.append(f"• {title} ({t['completed']})")
            else:
                # Format: "Topic - Part X" (removed /total_steps)
                lines.append(f"• {title} - Part {p.lesson_step}")

    lines.append("")
    lines.append(f"🧩 *{t['quizzes']}:*")
    if not quizzes:
        lines.append(f"• {t['no_quizzes_yet']}")
    else:
        for q in quizzes[:8]:
            title = clean_topic_title(q.topic)
            try:
                qs = json.loads(q.questions) if isinstance(q.questions, str) else q.questions
                total = len(qs)
            except (json.JSONDecodeError, TypeError):
                total = 3
            score = q.score if q.score is not None else 0
            # Add lesson_step if available: "Topic - Part X: Score/Total"
            lesson_step = getattr(q, "lesson_step", None)
            if lesson_step:
                lines.append(f"• {title} - Part {lesson_step}: {score}/{total}")
            else:
                lines.append(f"• {title}: {score}/{total}")

    lines.append("")
    lines.append(f"_{t['keep_learning']}_")
    if dashboard_url:
        lines.append("")
        lines.append(f"{t['view_dashboard']}: {dashboard_url}")
    return "\n".join(lines)


def _get_progress_translations(language: str) -> dict:
    """Get progress review translation strings."""
    translations = {
        "en": {
            "your_progress": "Your Progress",
            "lessons": "Lessons",
            "quizzes": "Quizzes",
            "completed": "completed",
            "no_lessons_yet": "No lessons yet",
            "no_quizzes_yet": "No quizzes yet",
            "no_progress": "No lessons or quizzes yet. Start with /lesson cells!",
            "keep_learning": "Keep learning with /lesson <topic> and /video <topic>!",
            "view_dashboard": "View your full dashboard",
            "topics_and_parts": "You’ve done {topics} topic(s) and {parts} lesson parts.",
            "no_fixed_end": "There’s no set endpoint—keep exploring!",
        },
        "es": {"your_progress": "Tu progreso", "lessons": "Lecciones", "quizzes": "Cuestionarios", "completed": "completado", "no_lessons_yet": "Aún no hay lecciones", "no_quizzes_yet": "Aún no hay cuestionarios", "no_progress": "Aún no hay lecciones ni cuestionarios. ¡Empieza con /lección células!", "keep_learning": "¡Sigue aprendiendo con /lección y /video <tema>!", "view_dashboard": "Ver tu panel completo", "topics_and_parts": "Has hecho {topics} tema(s) y {parts} partes de lección.", "no_fixed_end": "No hay un final fijo—¡sigue explorando!"},
        "fr": {"your_progress": "Votre progression", "lessons": "Leçons", "quizzes": "Quiz", "completed": "terminé", "no_lessons_yet": "Pas encore de leçons", "no_quizzes_yet": "Pas encore de quiz", "no_progress": "Pas encore de leçons ni de quiz. Commencez avec /leçon cellules !", "keep_learning": "Continuez avec /leçon et /video <sujet> !", "view_dashboard": "Voir votre tableau de bord", "topics_and_parts": "Vous avez fait {topics} sujet(s) et {parts} parties de leçon.", "no_fixed_end": "Il n’y a pas de fin fixe—continuez à explorer !"},
        "ms": {"your_progress": "Kemajuan anda", "lessons": "Pelajaran", "quizzes": "Kuiz", "completed": "siap", "no_lessons_yet": "Belum ada pelajaran", "no_quizzes_yet": "Belum ada kuiz", "no_progress": "Belum ada pelajaran atau kuiz. Mulakan dengan /pelajaran sel!", "keep_learning": "Terus belajar dengan /pelajaran dan /video <tajuk>!", "view_dashboard": "Lihat papan pemuka anda", "topics_and_parts": "Anda telah lakukan {topics} topik dan {parts} bahagian pelajaran.", "no_fixed_end": "Tiada penghujung tetap—terus terokai!"},
        "zh": {"your_progress": "你的进度", "lessons": "课程", "quizzes": "测验", "completed": "已完成", "no_lessons_yet": "暂无课程", "no_quizzes_yet": "暂无测验", "no_progress": "暂无课程或测验。用 /lesson 细胞 开始！", "keep_learning": "继续学习：/lesson 和 /video <主题>！", "view_dashboard": "查看完整仪表板", "topics_and_parts": "你已学了 {topics} 个主题，{parts} 课节。", "no_fixed_end": "没有固定终点，继续探索吧！"},
        "hi": {"your_progress": "आपकी प्रगति", "lessons": "पाठ", "quizzes": "क्विज़", "completed": "पूर्ण", "no_lessons_yet": "अभी तक कोई पाठ नहीं", "no_quizzes_yet": "अभी तक कोई क्विज़ नहीं", "no_progress": "अभी तक कोई पाठ या क्विज़ नहीं। /lesson कोशिकाएं से शुरू करें!", "keep_learning": "/lesson और /video <विषय> से सीखते रहें!", "view_dashboard": "अपना डैशबोर्ड देखें", "topics_and_parts": "आपने {topics} विषय और {parts} पाठ भाग किए।", "no_fixed_end": "कोई निश्चित अंत नहीं—खोजते रहें!"},
    }
    return translations.get(language, translations["en"])


def get_loading_message(command_type: str, topic: str = None, language: str = "en") -> str:
    """Get loading message in the specified language."""
    translations = {
        "en": {
            "lesson": f"⏳ Loading lesson: {topic.title()}" if topic else "⏳ LOADING LESSON...",
            "next": "⏳ Loading next part...",
            "quiz": "⏳ Loading quiz...",
            "progress": "⏳ Loading your progress...",
            "video": f"⏳ Generating video: {topic.title()}…" if topic else "⏳ Creating your video…",
            "default": "⏳ LOADING...",
        },
        "es": {
            "lesson": f"⏳ Cargando lección: {topic.title()}" if topic else "⏳ CARGANDO LECCIÓN...",
            "next": "⏳ Cargando siguiente parte...",
            "quiz": "⏳ Cargando cuestionario...",
            "progress": "⏳ Cargando tu progreso...",
            "video": f"⏳ Creando tu video sobre {topic.title()}…" if topic else "⏳ Creando tu video…",
            "default": "⏳ CARGANDO...",
        },
        "fr": {
            "lesson": f"⏳ Chargement de la leçon: {topic.title()}" if topic else "⏳ CHARGEMENT DE LA LEÇON...",
            "next": "⏳ Chargement de la partie suivante...",
            "quiz": "⏳ Chargement du quiz...",
            "progress": "⏳ Chargement de votre progression...",
            "video": f"⏳ Création de ta vidéo sur {topic.title()}…" if topic else "⏳ Création de ta vidéo…",
            "default": "⏳ CHARGEMENT...",
        },
        "ms": {
            "lesson": f"⏳ Memuatkan pelajaran: {topic.title()}" if topic else "⏳ MEMUATKAN PELAJARAN...",
            "next": "⏳ Memuatkan bahagian seterusnya...",
            "quiz": "⏳ Memuatkan kuiz...",
            "progress": "⏳ Memuatkan kemajuan anda...",
            "video": f"⏳ Mencipta video anda tentang {topic.title()}…" if topic else "⏳ Mencipta video…",
            "default": "⏳ MEMUATKAN...",
        },
        "zh": {
            "lesson": f"⏳ 加载课程: {topic.title()}" if topic else "⏳ 加载课程中...",
            "next": "⏳ 加载下一部分...",
            "quiz": "⏳ 加载测验...",
            "progress": "⏳ 加载你的进度...",
            "video": f"⏳ 正在生成关于 {topic.title()} 的视频…" if topic else "⏳ 正在生成视频…",
            "default": "⏳ 加载中...",
        },
        "hi": {
            "lesson": f"⏳ पाठ लोड हो रहा है: {topic.title()}" if topic else "⏳ पाठ लोड हो रहा है...",
            "next": "⏳ अगला भाग लोड हो रहा है...",
            "quiz": "⏳ क्विज़ लोड हो रहा है...",
            "progress": "⏳ आपकी प्रगति लोड हो रही है...",
            "video": f"⏳ {topic.title()} पर आपकी वीडियो बन रही है…" if topic else "⏳ वीडियो बन रही है…",
            "default": "⏳ लोड हो रहा है...",
        },
    }
    
    lang_dict = translations.get(language, translations["en"])
    return lang_dict.get(command_type, lang_dict["default"])


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
