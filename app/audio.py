import os
import re
import logging
import io
import tempfile
import asyncio
import concurrent.futures
import warnings
from typing import Optional, Tuple, List
from pathlib import Path
import requests

# Suppress Whisper FP16 warning on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead", category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class STTService:
    """
    Speech-to-Text service with OpenAI Whisper API (primary) and local Whisper fallback.
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.use_openai = self.openai_api_key is not None
        self.local_whisper_model = None
        
        if self.use_openai:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI Whisper API initialized (using API)")
            except ImportError:
                logger.warning("openai package not installed, falling back to local Whisper")
                self.use_openai = False
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}, falling back to local Whisper")
                self.use_openai = False
        
        if not self.use_openai:
            logger.info("Initializing local Whisper model (this may take a moment on first use)...")
            self._init_local_whisper()
    
    def _init_local_whisper(self):
        """Initialize local Whisper model as fallback."""
        try:
            import whisper
            # Suppress FP16 warning during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead", category=UserWarning)
                self.local_whisper_model = whisper.load_model("base")
            logger.info("Local Whisper model loaded successfully")
        except ImportError:
            logger.error("whisper package not installed. Install with: pip install openai-whisper")
            self.local_whisper_model = None
        except Exception as e:
            logger.error(f"Failed to load local Whisper model: {e}")
            self.local_whisper_model = None
    
    def transcribe(self, audio_data: bytes, content_type: str = "audio/ogg", language: Optional[str] = None) -> Optional[str]:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Audio file bytes
            content_type: MIME type of the audio (e.g., audio/ogg, audio/mpeg)
            language: Optional language code (en, es, fr) - helps accuracy, auto-detected if None
        
        Returns:
            Transcribed text or None if transcription fails
        """
        try:
            if self.use_openai and self.openai_client:
                return self._transcribe_openai(audio_data, content_type, language)
            elif self.local_whisper_model:
                return self._transcribe_local(audio_data, content_type)
            else:
                logger.error("No STT service available")
                return None
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return None
    
    def _transcribe_openai(self, audio_data: bytes, content_type: str, language: Optional[str] = None) -> Optional[str]:
        """Transcribe using OpenAI Whisper API. Language auto-detected if not provided."""
        try:
            audio_file = io.BytesIO(audio_data)
            audio_file.name = self._get_filename_from_content_type(content_type)
            
            # Whisper auto-detects language if not specified
            params = {
                "model": "whisper-1",
                "file": audio_file,
            }
            if language:
                params["language"] = language
            
            transcript = self.openai_client.audio.transcriptions.create(**params)
            
            text = transcript.text.strip()
            detected_lang = getattr(transcript, 'language', None)
            logger.info(f"OpenAI transcription successful ({detected_lang or 'auto'}): {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {e}")
            # Fallback to local if available
            if self.local_whisper_model:
                logger.info("Falling back to local Whisper...")
                return self._transcribe_local(audio_data, content_type)
            return None
    
    def _transcribe_local(self, audio_data: bytes, content_type: str) -> Optional[str]:
        """Transcribe using local Whisper model."""
        if not self.local_whisper_model:
            return None
        
        try:
            import whisper
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_suffix_from_content_type(content_type)) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            try:
                # Suppress FP16 warning specifically for this transcription
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead", category=UserWarning)
                    result = self.local_whisper_model.transcribe(tmp_file_path, language="en")
                text = result["text"].strip()
                logger.info(f"Local Whisper transcription successful: {len(text)} characters")
                return text
            finally:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
        except Exception as e:
            logger.error(f"Local Whisper transcription failed: {e}")
            return None
    
    def _get_filename_from_content_type(self, content_type: str) -> str:
        """Get appropriate filename extension from content type."""
        mapping = {
            "audio/ogg": "audio.ogg",
            "audio/mpeg": "audio.mp3",
            "audio/mp3": "audio.mp3",
            "audio/wav": "audio.wav",
            "audio/webm": "audio.webm",
            "audio/aac": "audio.aac",
            "audio/amr": "audio.amr",
            "audio/3gp": "audio.3gp",
        }
        return mapping.get(content_type.lower(), "audio.ogg")
    
    def _get_suffix_from_content_type(self, content_type: str) -> str:
        """Get file suffix from content type."""
        mapping = {
            "audio/ogg": ".ogg",
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/wav": ".wav",
            "audio/webm": ".webm",
            "audio/aac": ".aac",
            "audio/amr": ".amr",
            "audio/3gp": ".3gp",
        }
        return mapping.get(content_type.lower(), ".ogg")


class TTSService:
    """
    Text-to-Speech service with OpenAI TTS API (primary) and edge-tts fallback.
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.use_openai = self.openai_api_key is not None
        self.edge_tts = None
        
        if self.use_openai:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI TTS API initialized (using API)")
            except ImportError:
                logger.warning("openai package not installed, falling back to edge-tts")
                self.use_openai = False
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}, falling back to edge-tts")
                self.use_openai = False
        
        if not self.use_openai:
            logger.info("Initializing edge-tts (Microsoft Edge TTS)...")
            self._init_edge_tts()
    
    def _init_edge_tts(self):
        """Initialize edge-tts as fallback."""
        try:
            import edge_tts
            self.edge_tts = edge_tts
            logger.info("edge-tts initialized successfully (free, requires internet)")
        except ImportError:
            logger.warning("edge-tts package not installed. TTS service will not be available.")
            logger.info("To enable TTS, either:")
            logger.info("  1. Set OPENAI_API_KEY environment variable (for API-based TTS), or")
            logger.info("  2. Install edge-tts: pip install edge-tts (free, Python 3.12 compatible)")
            self.edge_tts = None
        except Exception as e:
            logger.error(f"Failed to initialize edge-tts: {e}")
            logger.info("To enable TTS, either:")
            logger.info("  1. Set OPENAI_API_KEY environment variable (for API-based TTS), or")
            logger.info("  2. Install edge-tts: pip install edge-tts (free, Python 3.12 compatible)")
            self.edge_tts = None
    
    def synthesize(self, text: str, voice: str = "alloy", age_group: int = 10, language: str = "en") -> Optional[Tuple[bytes, str]]:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (for OpenAI: alloy, echo, fable, onyx, nova, shimmer)
                   (for Edge TTS: will be auto-selected based on language and age)
            age_group: Age of the user (for adjusting speech characteristics)
            language: Language code (en, es, fr) - affects voice selection
        
        Returns:
            Tuple of (audio_bytes, content_type) or None if synthesis fails
        """
        try:
            text = self._text_to_audio_friendly(text)
            text = self._adjust_text_for_age(text, age_group)
            
            if self.use_openai and self.openai_client:
                return self._synthesize_openai(text, voice, age_group, language)
            elif self.edge_tts:
                return self._synthesize_edge_tts(text, voice, age_group, language)
            else:
                logger.error("No TTS service available")
                return None
        except Exception as e:
            logger.error(f"Error during TTS synthesis: {e}")
            return None
    
    def _text_to_audio_friendly(self, text: str) -> str:
        """
        Transform written/WhatsApp-formatted text so it reads naturally when spoken.
        - Removes or shortens headers and markdown that sound awkward aloud
        - Converts bullets and lists to fluent sentence structure
        - Strips UI prompts or turns them into short spoken cues
        - Removes emojis and excessive punctuation
        """
        if not text or not text.strip():
            return text
        t = text.strip()
        # Remove common UI/instruction lines that are for reading, not listening
        t = re.sub(
            r'_Type\s*`/next`[^_]*_',
            ' Say "Next" to continue.',
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r'_Type\s*`/[^`]+`[^_]*_',
            '',
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(r'_[^_]*Type\s+[^_]+_[^\n]*', '', t, flags=re.IGNORECASE)
        # Turn "Lesson: Topic" / "*Topic - Part N*" into spoken form
        t = re.sub(r'📚\s*\*Lesson:\s*([^*]+)\*', r'Lesson: \1.', t, flags=re.IGNORECASE)
        t = re.sub(r'\*([^*]+)\s*-\s*Part\s+(\d+)\*', r'Part \2. \1.', t, flags=re.IGNORECASE)
        # Strip bold/italic: *x* and _x_ -> x
        t = re.sub(r'\*([^*]+)\*', r'\1', t)
        t = re.sub(r'_([^_]+)_', r'\1', t)
        # Markdown headers: # ## ### -> treat as sentence (drop #, ensure period)
        t = re.sub(r'^#+\s*([^\n#]+)\s*$', r'\1.', t, flags=re.MULTILINE)
        t = re.sub(r'^#+\s*([^\n#]+)\s*\n', r'\1. ', t, flags=re.MULTILINE)
        # Bullets at line start (• - *) -> "Next, ..." or just the text with period
        def _bullet_to_sentence(match: re.Match) -> str:
            rest = (match.group(1) or '').strip()
            if not rest:
                return ''
            if rest.endswith('.'):
                return rest + ' '
            return rest.rstrip('.,;:') + '. '
        t = re.sub(r'^[\s]*[•\-*]\s+([^\n]+)', _bullet_to_sentence, t, flags=re.MULTILINE)
        # Numbered list lines "1. ..." -> keep as sentences
        t = re.sub(r'^[\s]*\d+\.\s+([^\n]+)(?=\n|$)', lambda m: (m.group(1).strip().rstrip('.') or '') + '. ', t, flags=re.MULTILINE)
        # Remove emojis (common ranges)
        t = re.sub(r'[\U0001F300-\U0001F9FF\U00002700-\U000027BF]', '', t)
        # Collapse multiple newlines to a single space so TTS doesn't over-pause
        t = re.sub(r'\n\s*\n+', ' ', t)
        t = re.sub(r'\n', ' ', t)
        # Normalize spaces and remove space before period
        t = re.sub(r' {2,}', ' ', t)
        t = re.sub(r'\s+\.', '.', t)
        t = t.strip()
        # Ensure last character is sentence-ending if non-empty
        if t and t[-1] not in '.!?':
            t = t.rstrip(';:,') + '.'
        return t
    
    def _adjust_text_for_age(self, text: str, age_group: int) -> str:
        """
        Adjust text for age-appropriate vocabulary and pacing.
        For younger users, we might want to simplify or slow down.
        """
        if age_group <= 8:
            pass
        return text
    
    def _synthesize_openai(self, text: str, voice: str = "alloy", age_group: int = 10, language: str = "en") -> Optional[Tuple[bytes, str]]:
        """Synthesize speech using OpenAI TTS API."""
        try:
            from .language import get_openai_voice_for_language_age
            # Use language-aware voice selection
            voice = get_openai_voice_for_language_age(language, age_group)
            
            # Limit text length to keep audio clips short (30-60 seconds)
            words = text.split()
            if len(words) > 150:
                text = " ".join(words[:150]) + "..."
                logger.info(f"Text truncated to {len(words[:150])} words for shorter audio clip")
            
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=1.0
            )
            
            audio_bytes = response.content
            logger.info(f"OpenAI TTS synthesis successful ({language}): {len(audio_bytes)} bytes")
            return (audio_bytes, "audio/mpeg")
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis failed: {e}")
            if self.edge_tts:
                logger.info("Falling back to edge-tts...")
                return self._synthesize_edge_tts(text, voice, age_group, language)
            return None
    
    def _synthesize_edge_tts(self, text: str, voice: str = "alloy", age_group: int = 10, language: str = "en") -> Optional[Tuple[bytes, str]]:
        """Synthesize speech using edge-tts (Microsoft Edge TTS)."""
        if not self.edge_tts:
            return None
        
        try:
            import asyncio
            import edge_tts
            from .language import get_edge_voice_for_language_age
            
            words = text.split()
            if len(words) > 150:
                text = " ".join(words[:150]) + "..."
                logger.info(f"Text truncated to {len(words[:150])} words for shorter audio clip")
            
            edge_voice = get_edge_voice_for_language_age(language, age_group)
            
            async def generate_audio():
                communicate = edge_tts.Communicate(text, edge_voice)
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                return audio_data
            
            # Handle async execution in FastAPI context (event loop already running)
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(generate_audio())
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    audio_bytes = future.result(timeout=60)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    audio_bytes = loop.run_until_complete(generate_audio())
                finally:
                    loop.close()
            
            if audio_bytes:
                logger.info(f"edge-tts synthesis successful: {len(audio_bytes)} bytes")
                return (audio_bytes, "audio/mpeg")  # edge-tts returns MP3
            else:
                logger.error("edge-tts returned empty audio")
                return None
                
        except Exception as e:
            logger.error(f"edge-tts synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_voice_for_age(self, age_group: int, language: str = "en") -> str:
        """Get appropriate voice for age group and language (OpenAI voices)."""
        from .language import get_openai_voice_for_language_age
        return get_openai_voice_for_language_age(language, age_group)


stt_service = STTService()
tts_service = TTSService()

def transcribe_audio(audio_data: bytes, content_type: str = "audio/ogg", language: Optional[str] = None) -> Optional[str]:
    """Convenience function to transcribe audio.
    
    Args:
        audio_data: Audio file bytes
        content_type: MIME type of the audio (e.g., audio/ogg, audio/mpeg)
        language: Optional language code (en, es, fr) - helps accuracy, auto-detected if None
    
    Returns:
        Transcribed text or None if transcription fails
    """
    return stt_service.transcribe(audio_data, content_type, language=language)

def synthesize_speech(text: str, voice: str = "alloy", age_group: int = 10) -> Optional[Tuple[bytes, str]]:
    """Convenience function to synthesize speech."""
    return tts_service.synthesize(text, voice, age_group)

def _chunk_text_for_audio(
    text: str,
    max_sentences_per_chunk: int = 4,
    max_words_per_chunk: int = 120,
) -> List[str]:
    """
    Split text into chunks that each end at a sentence boundary.
    Each chunk has at most max_sentences_per_chunk sentences and at most max_words_per_chunk words,
    so each audio segment is a complete thought and fits in one voice note.
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    # Split on sentence boundaries (. ! ?) followed by space or end
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]
    chunks = []
    current: List[str] = []
    current_words = 0
    for sent in sentences:
        n = len(sent.split())
        # Start a new chunk if adding this sentence would exceed limits
        if current and (
            len(current) >= max_sentences_per_chunk
            or current_words + n > max_words_per_chunk
        ):
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sent)
        current_words += n
    if current:
        chunks.append(" ".join(current))
    return chunks

def synthesize_speech_chunked(
    text: str, voice: str = "alloy", age_group: int = 10, max_segments: int = 2, language: str = "en"
) -> List[Tuple[bytes, str]]:
    """
    Synthesize text as at most max_segments audio segments, each ending at a sentence boundary.
    Chunks are merged so the full lesson is delivered in at most max_segments voice notes.
    Returns a list of (audio_bytes, content_type) per segment.
    Synthesizes chunks in parallel for faster processing.
    """
    import time
    start_time = time.time()
    try:
        full = tts_service._text_to_audio_friendly(text)
        full = tts_service._adjust_text_for_age(full, age_group)
        chunks = _chunk_text_for_audio(
            full, max_sentences_per_chunk=4, max_words_per_chunk=120
        )
        if not chunks:
            return []
        # Merge into at most max_segments chunks (e.g. 2: intro + rest)
        if len(chunks) > max_segments:
            mid = (len(chunks) + 1) // 2
            chunks = [" ".join(chunks[:mid]), " ".join(chunks[mid:])]
        
        # Synthesize chunks in parallel for faster processing
        import concurrent.futures
        results: List[Tuple[bytes, str]] = []
        
        def synthesize_chunk(chunk_idx: int, chunk_text: str) -> Optional[Tuple[bytes, str]]:
            """Synthesize a single chunk."""
            try:
                if tts_service.use_openai and tts_service.openai_client:
                    return tts_service._synthesize_openai(chunk_text, voice, age_group, language)
                elif tts_service.edge_tts:
                    return tts_service._synthesize_edge_tts(chunk_text, voice, age_group, language)
                else:
                    return None
            except Exception as e:
                logger.warning(f"Chunk {chunk_idx + 1} synthesis failed: {e}")
                return None
        
        # Use ThreadPoolExecutor to synthesize chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = {
                executor.submit(synthesize_chunk, i, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            # Collect results in order
            chunk_results = [None] * len(chunks)
            for future in concurrent.futures.as_completed(futures):
                chunk_idx = futures[future]
                try:
                    result = future.result()
                    chunk_results[chunk_idx] = result
                except Exception as e:
                    logger.warning(f"Chunk {chunk_idx + 1} future failed: {e}")
                    chunk_results[chunk_idx] = None
        
        # Build results list in order, skipping failed chunks
        for i, out in enumerate(chunk_results):
            if out:
                results.append(out)
                logger.info(f"Chunk {i + 1}/{len(chunks)} synthesized ({len(out[0])} bytes)")
            else:
                logger.warning(f"Chunk {i + 1} synthesis failed, skipping")
        
        elapsed = time.time() - start_time
        logger.info(f"TTS synthesis completed: {len(results)}/{len(chunks)} segments in {elapsed:.2f}s")
        return results
    except Exception as e:
        logger.error(f"Error during chunked TTS synthesis: {e}")
        return []

def text_to_audio_friendly(text: str) -> str:
    """
    Transform written/WhatsApp-formatted text so it reads naturally when spoken.
    Use this when you need audio-friendly text outside of TTS (e.g. for preview or tests).
    """
    return tts_service._text_to_audio_friendly(text)

def initialize_audio_services():
    """Initialize audio services (STT and TTS)."""
    try:
        if stt_service.use_openai:
            logger.info("STT: OpenAI Whisper API ready")
        elif stt_service.local_whisper_model:
            logger.info("STT: Local Whisper model ready")
        else:
            logger.warning("STT: No service available")
        
        if tts_service.use_openai:
            logger.info("TTS: OpenAI TTS API ready")
        elif tts_service.edge_tts:
            logger.info("TTS: edge-tts ready (free, requires internet)")
        else:
            logger.warning("TTS: No service available")
            logger.info("  To enable TTS: Set OPENAI_API_KEY or install edge-tts (pip install edge-tts)")
    except Exception as e:
        logger.error(f"Error during audio services initialization: {e}")

if __name__ == "__main__":
    print("Audio services module loaded successfully!")
    print(f"STT Service: {'OpenAI API' if stt_service.use_openai else 'Local Whisper' if stt_service.local_whisper_model else 'Not available'}")
    print(f"TTS Service: {'OpenAI API' if tts_service.use_openai else 'edge-tts' if tts_service.edge_tts else 'Not available'}")
