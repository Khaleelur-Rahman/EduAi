"""
Language support utilities for multilingual EduBot.
Supports: English (en), Spanish (es), French (fr), Malay (ms), Chinese (zh), Hindi (hi)
"""
import re
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "native": "English"},
    "es": {"name": "Spanish", "native": "Español"},
    "fr": {"name": "French", "native": "Français"},
    "ms": {"name": "Malay", "native": "Bahasa Melayu"},
    "zh": {"name": "Chinese", "native": "中文"},
    "hi": {"name": "Hindi", "native": "हिन्दी"},
}

# Language detection keywords (common words/phrases)
LANGUAGE_KEYWORDS = {
    "es": ["hola", "gracias", "por favor", "sí", "no", "lección", "enseñame", "siguiente"],
    "fr": ["bonjour", "merci", "s'il vous plaît", "oui", "non", "leçon", "apprends-moi", "suivant"],
    "ms": ["hello", "terima kasih", "tolong", "ya", "tidak", "pelajaran", "ajar saya", "seterusnya"],
    "zh": ["你好", "谢谢", "请", "是", "不", "课程", "教我", "下一个"],
    "hi": ["नमस्ते", "धन्यवाद", "कृपया", "हाँ", "नहीं", "पाठ", "मुझे सिखाओ", "अगला"],
    "en": ["hello", "thanks", "please", "yes", "no", "lesson", "teach me", "next"],
}

# Edge TTS voice mapping by language and age
EDGE_TTS_VOICES = {
    "en": {
        "young": "en-US-AriaNeural",  # Age <= 8
        "medium": "en-US-JennyNeural",  # Age 9-12
        "mature": "en-US-GuyNeural",  # Age > 12
    },
    "es": {
        "young": "es-ES-ElviraNeural",  # Female, friendly
        "medium": "es-ES-AlvaroNeural",  # Male, clear
        "mature": "es-ES-AlvaroNeural",
    },
    "fr": {
        "young": "fr-FR-DeniseNeural",  # Female, friendly
        "medium": "fr-FR-HenriNeural",  # Male, clear
        "mature": "fr-FR-HenriNeural",
    },
    "ms": {
        "young": "ms-MY-YasminNeural",  # Female, friendly
        "medium": "ms-MY-OsmanNeural",  # Male, clear
        "mature": "ms-MY-OsmanNeural",
    },
    "zh": {
        "young": "zh-CN-XiaoxiaoNeural",  # Female, friendly (Simplified Chinese)
        "medium": "zh-CN-YunxiNeural",  # Male, clear
        "mature": "zh-CN-YunxiNeural",
    },
    "hi": {
        "young": "hi-IN-SwaraNeural",  # Female, friendly
        "medium": "hi-IN-MadhurNeural",  # Male, clear
        "mature": "hi-IN-MadhurNeural",
    },
}

# OpenAI TTS voices (same for all languages, but we'll use language parameter)
OPENAI_TTS_VOICES = {
    "en": {"young": "nova", "medium": "alloy", "mature": "onyx"},
    "es": {"young": "nova", "medium": "alloy", "mature": "onyx"},
    "fr": {"young": "nova", "medium": "alloy", "mature": "onyx"},
    "ms": {"young": "nova", "medium": "alloy", "mature": "onyx"},
    "zh": {"young": "nova", "medium": "alloy", "mature": "onyx"},
    "hi": {"young": "nova", "medium": "alloy", "mature": "onyx"},
}


def validate_language_code(lang_code: str) -> Optional[str]:
    """Validate and normalize language code. Returns normalized code or None."""
    if not lang_code:
        return None
    lang_code = lang_code.strip().lower()
    if lang_code in SUPPORTED_LANGUAGES:
        return lang_code
    # Try common variations
    lang_map = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "malay": "ms",
        "chinese": "zh",
        "mandarin": "zh",
        "hindi": "hi",
        "español": "es",
        "français": "fr",
        "francais": "fr",
        "bahasa melayu": "ms",
        "bahasa": "ms",
        "中文": "zh",
        "普通话": "zh",
        "हिन्दी": "hi",
        "हिंदी": "hi",
    }
    return lang_map.get(lang_code)


def detect_language_from_text(text: str) -> Optional[str]:
    """Detect language from text using keyword matching. Returns language code or None."""
    if not text or len(text.strip()) < 3:
        return None
    
    text_lower = text.lower()
    scores = {lang: 0 for lang in SUPPORTED_LANGUAGES.keys()}
    
    # Count keyword matches
    for lang, keywords in LANGUAGE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[lang] += 1
    
    # Return language with highest score, or None if no matches
    max_score = max(scores.values())
    if max_score == 0:
        return None
    
    detected = max(scores.items(), key=lambda x: x[1])[0]
    logger.info(f"Detected language: {detected} (score: {max_score})")
    return detected


def get_edge_voice_for_language_age(language: str, age_group: int) -> str:
    """Get Edge TTS voice for language and age group."""
    lang = language if language in EDGE_TTS_VOICES else "en"
    
    if age_group <= 8:
        age_key = "young"
    elif age_group <= 12:
        age_key = "medium"
    else:
        age_key = "mature"
    
    return EDGE_TTS_VOICES.get(lang, EDGE_TTS_VOICES["en"]).get(age_key, "en-US-JennyNeural")


def get_openai_voice_for_language_age(language: str, age_group: int) -> str:
    """Get OpenAI TTS voice for language and age group."""
    lang = language if language in OPENAI_TTS_VOICES else "en"
    
    if age_group <= 8:
        age_key = "young"
    elif age_group <= 12:
        age_key = "medium"
    else:
        age_key = "mature"
    
    return OPENAI_TTS_VOICES.get(lang, OPENAI_TTS_VOICES["en"]).get(age_key, "alloy")


def get_language_name(lang_code: str, native: bool = False) -> str:
    """Get language name. If native=True, returns native name."""
    if lang_code not in SUPPORTED_LANGUAGES:
        return "English"
    return SUPPORTED_LANGUAGES[lang_code]["native" if native else "name"]
