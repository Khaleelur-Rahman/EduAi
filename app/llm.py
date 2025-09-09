import os
import logging
from typing import Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    
    def __init__(self):
        self.model_name = "deepseek/deepseek-chat-v3.1"
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = None
        self.max_tokens = 300
        self.temperature = 0.7
        self._initialized = False
    
    def initialize(self):
        if self._initialized:
            return
        
        if not self.api_key:
            logger.error("OPENROUTER_API_KEY not found in environment variables")
            raise Exception("OpenRouter API key is required. Please set OPENROUTER_API_KEY environment variable.")
        
        try:
            logger.info(f"Initializing OpenRouter client for model: {self.model_name}")
            
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
            
            logger.info("Testing OpenRouter connection...")
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": "Hello, can you respond with just 'OK'?"}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            if test_response.choices[0].message.content:
                logger.info(f"OpenRouter connection successful: {test_response.choices[0].message.content.strip()}")
                self._initialized = True
            else:
                raise Exception("No response from OpenRouter API")
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {str(e)}")
            raise Exception(f"OpenRouter initialization failed: {str(e)}")
    
    def generate_lesson(self, topic: str, age_group: int, user_name: str = "") -> str:
        if not self._initialized:
            self.initialize()
        
        if not self._initialized or self.client is None:
            logger.info(f"OpenRouter not available, using fallback lesson for topic: {topic}")
            return self._get_fallback_lesson(topic, age_group)
        
        system_prompt, user_prompt = self._create_lesson_prompt(topic, age_group, user_name)
        
        try:
            logger.info(f"Generating lesson with DeepSeek for topic: {topic}, age: {age_group}")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            if response.choices and response.choices[0].message.content:
                lesson_content = response.choices[0].message.content.strip()
                logger.info(f"DeepSeek response received: {len(lesson_content)} characters")
                
                if len(lesson_content) < 50:
                    raise Exception("Generated content too short")
                
                logger.info(f"Successfully generated lesson for topic: {topic}")
                return lesson_content
            else:
                raise Exception("No content received from DeepSeek")
            
        except Exception as e:
            logger.error(f"Failed to generate lesson with DeepSeek: {str(e)}")
            logger.info(f"Falling back to predefined lesson for topic: {topic}")
            return self._get_fallback_lesson(topic, age_group)
    
    def _create_lesson_prompt(self, topic: str, age_group: int, user_name: str = ""):
        if age_group <= 8:
            style_guide = "Use very simple words, short sentences, and examples with toys, animals, or games"
        elif age_group <= 12:
            style_guide = "Use simple language, clear examples, and everyday situations like school or home"
        elif age_group <= 16:
            style_guide = "Use clear explanations with relatable examples and real-world situations"
        else:
            style_guide = "Use detailed explanations with comprehensive examples and professional contexts"
        
        system_prompt = f"""You are an expert educator and tutor.
Your goal is to teach a topic clearly and concisely so that the learner fully understands it.

Instructions:
- Topic: {topic}
- Age group: {age_group} years old
- Length: Keep it short and focused (150-200 words max).
- Style: {style_guide}.
- Structure:
   1. Brief introduction
   2. Key explanation (step by step, or definition + example)
   3. Real-life analogy or story that makes it easy to remember
   4. One simple practice question at the end

Make sure the explanation is **accurate**, **easy to follow**, and **age-appropriate**."""
        
        greeting = f"Hey {user_name}! " if user_name else ""
        user_prompt = f"""{greeting}Please teach me about {topic}."""
        
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

def generate_lesson(topic: str, age_group: int, user_name: str = "") -> str:
    return llm_service.generate_lesson(topic, age_group, user_name)

def initialize_llm():
    llm_service.initialize()
