import os
import re
import logging
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_IMAGE_PROMPTS = 2

class LLMService:
    
    def __init__(self):
        self.model_name = os.getenv("CEREBRAS_MODEL", "llama3.1-8b")  # Cerebras Cloud production model
        self.api_key = os.getenv("CEREBRAS_API_KEY")
        self.client = None
        self.max_tokens = 1000  # Increased to ensure complete responses
        self.temperature = 0.7
        self.top_p = 0.8
        self._initialized = False
    
    def initialize(self):
        if self._initialized:
            return
        
        if not self.api_key:
            logger.error("CEREBRAS_API_KEY not found in environment variables")
            raise Exception("Cerebras API key is required. Please set CEREBRAS_API_KEY environment variable.")
        
        try:
            logger.info(f"Initializing Cerebras client for model: {self.model_name}")
            
            self.client = Cerebras(api_key=self.api_key)
            
            logger.info("Testing Cerebras connection...")
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": "Reply with only the word OK."}
                ],
                max_completion_tokens=10,
                temperature=0.1,
                top_p=0.8,
                stream=False
            )
            if not getattr(test_response, "choices", None) or len(test_response.choices) == 0:
                raise Exception("No response from Cerebras API")
            response_content = (getattr(test_response.choices[0].message, "content", None) or "").strip()
            logger.info("Cerebras connection successful: %s", response_content or "(ok)")
            self._initialized = True
                
        except Exception as e:
            logger.error(f"Failed to initialize Cerebras client: {str(e)}")
            raise Exception(f"Cerebras initialization failed: {str(e)}")
    
    def generate_lesson(self, topic: str, age_group: int, user_name: str = "", 
                       is_continuation: bool = False, previous_content: str = None,
                       for_audio: bool = False, language: str = "en") -> str:
        if not self._initialized:
            self.initialize()
        
        if not self._initialized or self.client is None:
            logger.info(f"Cerebras not available, using fallback lesson for topic: {topic}")
            return self._get_fallback_lesson(topic, age_group)
        
        system_prompt, user_prompt = self._create_lesson_prompt(
            topic, age_group, user_name, is_continuation=is_continuation, previous_content=previous_content,
            for_audio=for_audio, language=language
        )
        
        try:
            logger.info(f"Generating lesson with Cerebras for topic: {topic}, age: {age_group}")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True
            )
            
            lesson_content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    lesson_content += chunk.choices[0].delta.content
            
            lesson_content = lesson_content.strip()
            lesson_content = re.sub(r'<think>.*?</think>', '', lesson_content, flags=re.DOTALL | re.IGNORECASE).strip()
            if len(lesson_content) > 1400:
                logger.warning(f"Response too long ({len(lesson_content)} chars), retrying with stricter limit")
                # Retry with a much stricter character limit
                # Language instruction
                lang_instruction = ""
                if language != "en":
                    from .language import get_language_name
                    lang_name = get_language_name(language, native=True)
                    lang_instruction = f"\n- Language: Generate the entire lesson in {lang_name} ({language.upper()}). All text, explanations, examples, and responses must be in {lang_name}."
                
                if is_continuation and previous_content:
                    retry_system_prompt = f"""You are an expert educator and tutor.
You are continuing a lesson on {topic}. The student has already learned the previous part.

Instructions:
- Topic: {topic} (continuation)
- Age group: {age_group} years old
- Length: Keep it VERY SHORT (under 1200 characters total - this is critical for WhatsApp delivery).
- Style: Use simple language, clear examples, and everyday situations.{lang_instruction}

CONTINUATION STRUCTURE:
- Start by briefly referencing what was covered in the previous part (1-2 sentences)
- Then continue with new information
- Do NOT repeat examples from the previous part
- Do NOT start with a new example - jump straight into continuing the explanation

CRITICAL FORMATTING RULES:
- Use single asterisk *text* for bold (WhatsApp format), NOT double asterisks **
- Do NOT include "Try This at Home" or similar activity sections unless they directly relate to the topic
- Focus on clear explanations and examples, not generic activities

CRITICAL COMPLETENESS REQUIREMENTS:
- ALWAYS complete your response with proper ending punctuation (. ! ?)
- NEVER cut off mid-sentence, mid-list, or mid-thought
- If listing items, complete the entire list before ending
- Ensure the response is a complete, coherent continuation that can stand alone
- End naturally with a complete sentence

Make sure the explanation is accurate, easy to follow, and age-appropriate.

CRITICAL: Keep the response under 1200 characters to ensure WhatsApp delivery. Be concise but ALWAYS complete."""
                else:
                    retry_system_prompt = f"""You are an expert educator and tutor.
Your goal is to teach a topic clearly and concisely so that the learner fully understands it.

Instructions:
- Topic: {topic}
- Age group: {age_group} years old
- Length: Keep it VERY SHORT (under 1200 characters total - this is critical for WhatsApp delivery).
- Style: Use simple language, clear examples, and everyday situations.{lang_instruction}
- Structure: Brief introduction, key explanation, and one simple example

CRITICAL FORMATTING RULES:
- Use single asterisk *text* for bold (WhatsApp format), NOT double asterisks **
- Do NOT include "Try This at Home" or similar activity sections unless they directly relate to the topic
- Focus on clear explanations and examples, not generic activities

CRITICAL COMPLETENESS REQUIREMENTS:
- ALWAYS complete your response with proper ending punctuation (. ! ?)
- NEVER cut off mid-sentence, mid-list, or mid-thought
- If listing items, complete the entire list before ending
- Ensure the response is a complete, coherent lesson that can stand alone
- End naturally with a complete sentence

Make sure the explanation is accurate, easy to follow, and age-appropriate.

CRITICAL: Keep the response under 1200 characters to ensure WhatsApp delivery. Be concise but ALWAYS complete."""
                if for_audio:
                    retry_system_prompt += """

AUDIO/TTS MODE: Your reply will be read aloud by text-to-speech. Write for listening: use short, complete sentences; avoid markdown headers (##) and bullet lists—use flowing prose instead; do not include instructions like "Type /next"; use minimal or no emojis; write as if you are speaking to the student."""
                
                if is_continuation and previous_content:
                    retry_user_prompt = f"""Continue teaching about {topic}. 

Previous part of the lesson:
{previous_content[:500]}

Continue the lesson naturally, referencing what was just covered and building on it!"""
                else:
                    retry_user_prompt = f"Please teach me about {topic}."
                
                retry_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": retry_system_prompt},
                        {"role": "user", "content": retry_user_prompt}
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
                lesson_content = re.sub(r'<think>.*?</think>', '', lesson_content, flags=re.DOTALL | re.IGNORECASE).strip()
                logger.info(f"Retry response length: {len(lesson_content)} characters")
            
            # Check if content appears to be truncated (doesn't end with proper punctuation)
            # Allow emojis at the end, but check the text before emojis
            # Remove trailing emojis and whitespace to check actual ending
            text_without_emojis = re.sub(r'[\s\U0001F300-\U0001F9FF]+$', '', lesson_content.rstrip())
            if lesson_content and text_without_emojis and not text_without_emojis.endswith(('.', '!', '?', ':', ';')):
                logger.warning(f"Content appears truncated, attempting to complete: {lesson_content[-100:]}")
                # Try to complete the truncated content
                # Language instruction for completion
                lang_instruction = ""
                if language != "en":
                    from .language import get_language_name
                    lang_name = get_language_name(language, native=True)
                    lang_instruction = f" Complete the text in {lang_name} ({language.upper()})."
                
                completion_system_prompt = f"Complete the following educational text naturally. Only provide the completion to finish the thought, not the full text. Make sure it ends with proper punctuation.{lang_instruction} Do not include any thinking tags or reasoning - only provide the completion text."
                
                completion_response = self.client.chat.completions.create(
                    model=self.model_name,
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
                    completion = re.sub(r'<think>.*?</think>', '', completion, flags=re.DOTALL | re.IGNORECASE).strip()
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
                    
                    logger.info(f"Successfully completed truncated content")
            
            if lesson_content:
                lesson_content = lesson_content.strip()
                logger.info(f"Cerebras response received: {len(lesson_content)} characters")
                
                if len(lesson_content) < 50:
                    raise Exception("Generated content too short")
                
                # Final check - if still over limit, truncate at sentence boundary but ensure it ends properly
                if len(lesson_content) > 1400:
                    logger.warning(f"Response still too long ({len(lesson_content)} chars), truncating at sentence boundary")
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
                    logger.info(f"Truncated to {len(lesson_content)} characters")
                
                logger.info(f"Successfully generated lesson for topic: {topic}")
                return lesson_content
            else:
                raise Exception("No content received from Cerebras")
            
        except Exception as e:
            logger.error(f"Failed to generate lesson with Cerebras: {str(e)}")
            logger.info(f"Falling back to predefined lesson for topic: {topic}")
            return self._get_fallback_lesson(topic, age_group)

    def generate_video_script(
        self, topic: str, age_group: int = 10, num_images: int = 4, language: str = "en"
    ) -> dict:
        """
        Generate a narration script and matching image prompts for the hybrid video pipeline.
        Narration is in the requested language; image prompts stay in English for model compatibility.
        """
        t = topic.strip()

        narration = ""
        for attempt in range(3):
            narration = self._generate_narration_text(t, age_group, language)
            word_count = self._word_count_for_narration(narration, language)
            if narration and word_count >= 40:
                break
            logger.info("Narration attempt %d for '%s': %d words, retrying", attempt + 1, t, word_count)
        if not narration or self._word_count_for_narration(narration, language) < 40:
            logger.info("Video narration for '%s': using fallback", topic)
            return self._video_script_fallback(topic, num_images, language)

        image_prompts = self._generate_image_prompts(t, num_images)
        if len(image_prompts) < MIN_IMAGE_PROMPTS:
            logger.info("Only %d image prompts for '%s', using defaults", len(image_prompts), t)
            image_prompts = self._default_image_prompts(t, num_images)

        logger.info("Video script for '%s' (%s): narration %d chars, %d image prompts",
                     topic, language, len(narration), len(image_prompts))
        return {"narration": narration, "image_prompts": image_prompts}

    def _word_count_for_narration(self, text: str, language: str) -> int:
        """Approximate word count for timing; use char-based for languages without spaces."""
        if not text or not text.strip():
            return 0
        if language in ("zh", "ja", "ko", "hi", "th"):
            return max(1, len(text.strip()) // 3)
        return len(text.strip().split())

    def _generate_narration_text(self, topic_title: str, age_group: int, language: str = "en") -> str:
        """Generate plain-text narration via Cerebras streaming in the requested language."""
        if not (self._initialized or self._try_initialize()):
            return ""
        lang_instruction = ""
        if language and language != "en":
            from .language import get_language_name
            lang_name = get_language_name(language, native=True)
            lang_instruction = f" Write the ENTIRE narration in {lang_name} ({language.upper()}). No English."
        system = (
            "You write short voiceover scripts for educational videos. "
            "Output ONLY the narration text, no titles or labels. "
            "Rules: Exactly 4 to 6 short sentences. Include 2 to 3 concrete facts and one simple example. "
            "Length: about 20 to 30 seconds when read aloud (roughly 60-80 words). "
            "Plain prose, written to be read aloud. No bullet points, no markdown. "
            "Do not include <think>, reasoning, or any text that is not the spoken narration."
            + lang_instruction
        )
        user = (
            f'Write the voiceover for a short educational video about "{topic_title}". '
            f"Audience: {age_group} years old. "
            "Include 2-3 concrete facts and one simple example. "
            "Output only the narration, nothing else."
        )
        if language and language != "en":
            from .language import get_language_name
            lang_name = get_language_name(language, native=True)
            user += f" Write the narration entirely in {lang_name}."
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=self.max_tokens,
                temperature=0.7,
                top_p=0.8,
                stream=True,
            )
            raw = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    raw += chunk.choices[0].delta.content
            text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE).strip()
            if text and len(text) > 30:
                words = text.split()
                if len(words) > 100:
                    text = " ".join(words[:100]).rstrip()
                    if not text.endswith((".", "!", "?")):
                        text += "."
                return text
        except Exception as e:
            logger.warning("Narration generation failed for '%s': %s", topic_title, e)
        return ""

    def _generate_image_prompts(self, topic_title: str, num_images: int) -> list:
        """Generate image prompts via Cerebras streaming, one per line."""
        if not (self._initialized or self._try_initialize()):
            return []
        system = (
            "You write image generation prompts for Stable Diffusion XL. "
            f"Output exactly {num_images} prompts, one per line, numbered 1. 2. 3. 4. "
            "Each prompt MUST describe a COMPLETELY DIFFERENT scene, subject, or perspective. "
            "For example: 1=diagram/overview, 2=close-up of a key part, 3=real-world photo, 4=comparison/before-after. "
            "Each prompt: 20-40 words, concrete visual scene, style keywords like 'colorful, detailed, clean'. "
            "CRITICAL: Do NOT include any text, words, labels, letters, or writing in the image. "
            "No markdown, no explanations, just the numbered prompts."
        )
        user = (
            f'Write {num_images} DIVERSE image prompts for an educational video about "{topic_title}". '
            "Each must show a completely different scene or angle -- no two should look similar. "
            "Do not put any text, words, labels, or writing in the images."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=self.max_tokens,
                temperature=0.7,
                top_p=0.8,
                stream=True,
            )
            raw = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    raw += chunk.choices[0].delta.content
            text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE).strip()
            prompts = []
            for line in text.split("\n"):
                line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                if len(line) > 15:
                    prompts.append(line)
            return prompts[:num_images]
        except Exception as e:
            logger.warning("Image prompt generation failed for '%s': %s", topic_title, e)
        return []

    def _default_image_prompts(self, topic_title: str, num_images: int = 4) -> list:
        """Deterministic diverse image prompts when LLM fails."""
        return [
            f"Wide overview diagram of {topic_title}, colorful educational illustration, clean white background, detailed",
            f"Close-up microscopic or detailed view of {topic_title}, scientific photography style, vivid colors, sharp focus",
            f"Real-world outdoor photograph showing {topic_title} in nature, photorealistic, golden hour lighting",
            f"Animated cartoon style illustration explaining {topic_title}, friendly characters, bright pastel colors, simple shapes",
        ][:num_images]

    def _video_script_fallback(self, topic: str, num_images: int = 4, language: str = "en") -> dict:
        """Static fallback when LLM is unavailable. Narration in English."""
        t = topic.strip()
        narration = (
            f"Today we're going to learn about {t}. "
            f"This is a fascinating topic that scientists have studied for many years. "
            f"Understanding {t} helps us make sense of the world around us. "
            "There are several key ideas we need to explore to get the full picture. "
            "Each one builds on the last, so pay close attention. "
            "By the end of this video, you'll have a solid understanding of how it all works."
        )
        return {"narration": narration, "image_prompts": self._default_image_prompts(t, num_images)}

    def _try_initialize(self) -> bool:
        """Initialize Cerebras if possible; return True if client is ready."""
        try:
            self.initialize()
            return self._initialized and self.client is not None
        except Exception:
            return False

    def _create_lesson_prompt(self, topic: str, age_group: int, user_name: str = "", 
                             is_continuation: bool = False, previous_content: str = None,
                             for_audio: bool = False, language: str = "en"):
        if age_group <= 8:
            style_guide = "Use very simple words, short sentences, and examples with toys, animals, or games. Use lots of relevant emojis (e.g. 🌱🔬✨) throughout to make it fun!"
        elif age_group <= 12:
            style_guide = "Use simple language, clear examples, and everyday situations like school or home. Include several relevant emojis throughout the lesson (e.g. 🌱📚💡) to keep it engaging!"
        elif age_group <= 16:
            style_guide = "Use clear explanations with relatable examples and real-world situations. Include some relevant emojis (e.g. 🌿🔬✨) to keep it engaging."
        else:
            style_guide = "Use detailed explanations with comprehensive examples and professional contexts. You may add a few relevant emojis for tone."
        
        # Language instruction
        lang_instruction = ""
        if language != "en":
            from .language import get_language_name
            lang_name = get_language_name(language, native=True)
            lang_instruction = f"\n- Language: Generate the entire lesson in {lang_name} ({language.upper()}). All text, explanations, examples, and responses must be in {lang_name}."
        
        if is_continuation and previous_content:
            # Continuation lesson
            system_prompt = f"""You are an expert educator and tutor.
You are continuing a lesson on {topic}. The student has already learned the previous part.

Instructions:
- Topic: {topic} (continuation)
- Age group: {age_group} years old
- Length: Keep it concise and focused (under 1400 characters total).
- Style: {style_guide}.{lang_instruction}

CONTINUATION STRUCTURE:
- Start by briefly referencing what was covered in the previous part (1-2 sentences)
- Then continue with new information
- Do NOT repeat examples from the previous part
- Do NOT start with a new example - jump straight into continuing the explanation
- Make it feel like a natural conversation continuation

CRITICAL FORMATTING RULES:
- Use single asterisk *text* for bold (WhatsApp format), NOT double asterisks **
- Do NOT include "Try This at Home" or similar activity sections unless they directly relate to the topic
- Focus on clear explanations and examples, not generic activities
- Do not add unnecessary formatting or redundant bold markers
- Include a few relevant emojis to keep the tone engaging and match the first part of the lesson

Make sure the explanation is accurate, easy to follow, and age-appropriate.

IMPORTANT: 
- Make it conversational and connected to what the student just learned.
- End with a complete sentence, then optional emojis. Keep the response under 1400 characters to ensure WhatsApp delivery."""
            if for_audio:
                system_prompt += """

AUDIO/TTS MODE: Your reply will be read aloud by text-to-speech. Write for listening: use short, complete sentences; avoid markdown headers (##) and bullet lists—use flowing prose instead; do not include instructions like "Type /next"; use minimal or no emojis; write as if you are speaking to the student."""
            
            # Add language instruction to user prompt as well for reinforcement
            user_lang_note = ""
            if language != "en":
                from .language import get_language_name
                lang_name = get_language_name(language, native=True)
                user_lang_note = f"\n\nIMPORTANT: Write the entire continuation in {lang_name} ({language.upper()})."
            
            user_prompt = f"""Continue teaching about {topic}. 

Previous part of the lesson:
{previous_content[:500]}

Continue the lesson naturally, referencing what was just covered and building on it!{user_lang_note}"""
        else:
            # New lesson
            system_prompt = f"""You are an expert educator and tutor.
Your goal is to teach a topic clearly and concisely so that the learner fully understands it.

Instructions:
- Topic: {topic}
- Age group: {age_group} years old
- Length: Keep it concise and focused (under 1400 characters total).
- Style: {style_guide}.{lang_instruction}
- Structure:
   1. Brief introduction
   2. Key explanation (step by step, or definition + example)
   3. Real-life analogy or story that makes it easy to remember

CRITICAL FORMATTING RULES:
- Use single asterisk *text* for bold (WhatsApp format), NOT double asterisks **
- Do NOT include "Try This at Home" or similar activity sections unless they directly relate to the topic
- Focus on clear explanations and examples, not generic activities
- Do not add unnecessary formatting or redundant bold markers
- EMOJIS (required): Put 1–2 emojis in the very first line (e.g. right after the opening sentence). Put at least one emoji in the middle (e.g. after "How it works" or a key point) and one in the fun example. Use 4–8 emojis total, scattered throughout. Do NOT put all emojis only at the end—if the message is shortened, the end may be cut off. Examples: 🌱🔬✨📚💡🌞🌿

Make sure the explanation is accurate, easy to follow, and age-appropriate.

CRITICAL COMPLETENESS REQUIREMENTS:
- ALWAYS complete your response with proper ending punctuation (. ! ?) BEFORE any emojis
- If you use emojis, place them AFTER the final punctuation mark
- NEVER cut off mid-sentence, mid-list, or mid-thought
- If listing items, complete the entire list before ending
- Ensure the response is a complete, coherent lesson that can stand alone
- End naturally with a complete sentence followed by punctuation, then optional emojis

IMPORTANT: 
- Focus only on teaching the topic. Do not introduce yourself or respond to greetings. Start directly with the lesson content.
- Keep the response under 1400 characters to ensure WhatsApp delivery.
- ALWAYS provide a complete, finished response. Include several emojis as specified above."""
            if for_audio:
                system_prompt += """

AUDIO/TTS MODE: Your reply will be read aloud by text-to-speech. Write for listening: use short, complete sentences; avoid markdown headers (##) and bullet lists—use flowing prose instead; do not include instructions like "Type /next"; use minimal or no emojis; write as if you are speaking to the student."""
            
            # Add language instruction to user prompt as well for reinforcement
            user_lang_note = ""
            if language != "en":
                from .language import get_language_name
                lang_name = get_language_name(language, native=True)
                user_lang_note = f"\n\nIMPORTANT: Write the entire lesson in {lang_name} ({language.upper()})."
            
            greeting = ""
            user_prompt = f"""{greeting}Please teach me about {topic}.{user_lang_note}"""
        
        return system_prompt, user_prompt
    
    def _clean_response(self, response: str) -> str:
        response = response.strip()
        
        response = response.replace('<|endoftext|>', '')
        response = response.replace('<|end|>', '')
        response = response.replace('</s>', '')
        
        response = self._structure_educational_content(response)
        
        if len(response) > 1000:
            sentences = response.split('.')
            cleaned_sentences = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) > 800:
                    break
                cleaned_sentences.append(sentence)
                current_length += len(sentence)
            
            response = '. '.join(cleaned_sentences) + '.'
        
        return response
    
    def _structure_educational_content(self, content: str) -> str:
        content = content.strip()
        sentences = [s.strip() for s in content.split('.') if s.strip() and len(s.strip()) > 5]
        
        if len(sentences) < 1:
            return content
        
        good_sentences = []
        for sentence in sentences[:3]:
            clean_sentence = sentence.strip()
            if clean_sentence and len(clean_sentence) > 5:
                if not clean_sentence.endswith(('.', '!', '?')):
                    clean_sentence += '.'
                good_sentences.append(clean_sentence)
        
        if good_sentences:
            result = ' '.join(good_sentences)
            result += "\n\n👉 Practice: Can you give an example?"
            return result
        
        return content
    
    def _get_fallback_lesson(self, topic: str, age_group: int) -> str:
        fallback_lessons = {
            "fractions": f"""
Fractions are a way of showing parts of a whole! 🍕

Imagine you cut a pizza into 4 equal slices. If you eat 1 slice, that's 1/4 of the pizza. The number on top (numerator) shows how many parts you have. The number on the bottom (denominator) shows how many equal parts the whole is divided into.

Think of it like sharing chocolate with friends. If you break a bar into 8 pieces and keep 3, you have 3/8 of the bar! 🍫

👉 Practice: If you have 12 apples and eat 6, what fraction of the apples did you eat?
            """,
            "addition": f"""
Addition means putting numbers together! ➕

When we add, we combine groups of things. Like if you have 3 apples and I give you 2 more apples, you now have 5 apples total! We write this as 3 + 2 = 5.

Think of addition like collecting toys. If you have 4 toy cars and find 3 more, you now have 7 toy cars! 🚗

👉 Practice: If you have 6 stickers and get 4 more, how many stickers do you have in total?
            """,
            "photosynthesis": f"""
Photosynthesis is how plants make their own food! 🌱

Plants are like little chefs that use sunlight, water, and air to cook up their meals. They take in sunlight through their leaves, drink water through their roots, and breathe in carbon dioxide from the air.

Think of leaves as tiny solar panels that capture sunlight and turn it into energy. This process also makes oxygen - the air we breathe! That's why plants are so important for life on Earth.

👉 Practice: What three things do plants need for photosynthesis?
            """,
            "multiplication": f"""
Multiplication is like super-fast addition! ✖️

Instead of adding the same number over and over, we can multiply. If you have 3 groups of 4 apples each, that's 3 × 4 = 12 apples total. It's much faster than adding 4 + 4 + 4!

Think of multiplication tables like recipes. Just like a cookie recipe might say "makes 24 cookies," multiplication tells us how many we get when we have groups of the same size.

👉 Practice: If you have 5 bags with 6 marbles each, how many marbles do you have in total?
            """,
            "solar system": f"""
Our solar system is like a cosmic neighborhood! 🌞

The Sun is at the center, and eight planets orbit around it like runners on a track. Mercury is closest and hottest, while Neptune is farthest and coldest. Earth is in the perfect spot - not too hot, not too cold - just right for life!

Think of the solar system like a giant merry-go-round with the Sun in the middle. Each planet takes a different amount of time to complete one trip around the Sun - that's what we call a year!

👉 Practice: Which planet is closest to the Sun?
            """,
        }
        
        topic_lower = topic.lower()
        for key in fallback_lessons:
            if key in topic_lower:
                return fallback_lessons[key].strip()
        
        return f"""
I'd love to teach you about {topic}! 📚

{topic} is an interesting subject that helps us understand the world better. While I'm having trouble generating a detailed lesson right now, I encourage you to explore this topic further.

Think about how {topic} might relate to things you see every day. Learning becomes easier when we connect new ideas to things we already know!

👉 Practice: Can you think of one way {topic} might be useful in everyday life?

Try asking me again, or ask for help with a specific part of {topic} that interests you most!
        """.strip()

llm_service = LLMService()

def generate_lesson(topic: str, age_group: int, user_name: str = "", 
                   is_continuation: bool = False, previous_content: str = None,
                   for_audio: bool = False, language: str = "en") -> str:
    return llm_service.generate_lesson(topic, age_group, user_name, 
                                      is_continuation=is_continuation, 
                                      previous_content=previous_content,
                                      for_audio=for_audio, language=language)


def generate_video_script(
    topic: str, age_group: int = 10, num_images: int = 4, language: str = "en"
) -> dict:
    """Generate narration + image prompts for the hybrid video pipeline. Narration in requested language."""
    return llm_service.generate_video_script(topic, age_group, num_images, language)


def initialize_llm():
    llm_service.initialize()
