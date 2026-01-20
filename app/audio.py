import os
import logging
import io
import tempfile
import asyncio
import concurrent.futures
from typing import Optional, Tuple
from pathlib import Path
import requests

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
            self.local_whisper_model = whisper.load_model("base")
            logger.info("Local Whisper model loaded successfully")
        except ImportError:
            logger.error("whisper package not installed. Install with: pip install openai-whisper")
            self.local_whisper_model = None
        except Exception as e:
            logger.error(f"Failed to load local Whisper model: {e}")
            self.local_whisper_model = None
    
    def transcribe(self, audio_data: bytes, content_type: str = "audio/ogg") -> Optional[str]:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Audio file bytes
            content_type: MIME type of the audio (e.g., audio/ogg, audio/mpeg)
        
        Returns:
            Transcribed text or None if transcription fails
        """
        try:
            if self.use_openai and self.openai_client:
                return self._transcribe_openai(audio_data, content_type)
            elif self.local_whisper_model:
                return self._transcribe_local(audio_data, content_type)
            else:
                logger.error("No STT service available")
                return None
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return None
    
    def _transcribe_openai(self, audio_data: bytes, content_type: str) -> Optional[str]:
        """Transcribe using OpenAI Whisper API."""
        try:
            audio_file = io.BytesIO(audio_data)
            audio_file.name = self._get_filename_from_content_type(content_type)
            
            transcript = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
            
            text = transcript.text.strip()
            logger.info(f"OpenAI transcription successful: {len(text)} characters")
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
    
    def synthesize(self, text: str, voice: str = "alloy", age_group: int = 10) -> Optional[Tuple[bytes, str]]:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (for OpenAI: alloy, echo, fable, onyx, nova, shimmer)
                   (for Coqui: not used, model has default voice)
            age_group: Age of the user (for adjusting speech characteristics)
        
        Returns:
            Tuple of (audio_bytes, content_type) or None if synthesis fails
        """
        try:
            text = self._adjust_text_for_age(text, age_group)
            
            if self.use_openai and self.openai_client:
                return self._synthesize_openai(text, voice)
            elif self.edge_tts:
                return self._synthesize_edge_tts(text, voice, age_group)
            else:
                logger.error("No TTS service available")
                return None
        except Exception as e:
            logger.error(f"Error during TTS synthesis: {e}")
            return None
    
    def _adjust_text_for_age(self, text: str, age_group: int) -> str:
        """
        Adjust text for age-appropriate vocabulary and pacing.
        For younger users, we might want to simplify or slow down.
        """
        if age_group <= 8:
            pass
        return text
    
    def _synthesize_openai(self, text: str, voice: str = "alloy") -> Optional[Tuple[bytes, str]]:
        """Synthesize speech using OpenAI TTS API."""
        try:
            # Limit text length to keep audio clips short (30-60 seconds)
            # Average speaking rate is ~150 words/min, so ~75-150 words = 30-60 seconds
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
            logger.info(f"OpenAI TTS synthesis successful: {len(audio_bytes)} bytes")
            return (audio_bytes, "audio/mpeg")
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis failed: {e}")
            if self.edge_tts:
                logger.info("Falling back to edge-tts...")
                return self._synthesize_edge_tts(text, voice, age_group)
            return None
    
    def _synthesize_edge_tts(self, text: str, voice: str = "alloy", age_group: int = 10) -> Optional[Tuple[bytes, str]]:
        """Synthesize speech using edge-tts (Microsoft Edge TTS)."""
        if not self.edge_tts:
            return None
        
        try:
            import asyncio
            import edge_tts
            
            words = text.split()
            if len(words) > 150:
                text = " ".join(words[:150]) + "..."
                logger.info(f"Text truncated to {len(words[:150])} words for shorter audio clip")
            
            edge_voice = self._get_edge_voice_for_age(age_group)
            
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
    
    def _get_edge_voice_for_age(self, age_group: int) -> str:
        """Get appropriate edge-tts voice for age group."""
        # Microsoft Edge TTS voices (English US)
        # Female voices: AriaNeural, JennyNeural, MichelleNeural
        # Male voices: GuyNeural, RogerNeural, DavisNeural
        if age_group <= 8:
            return "en-US-AriaNeural"  # Softer, friendlier female voice
        elif age_group <= 12:
            return "en-US-JennyNeural"  # Clear, balanced female voice
        else:
            return "en-US-GuyNeural"  # More mature male voice
    
    def get_voice_for_age(self, age_group: int) -> str:
        """Get appropriate voice for age group (OpenAI voices)."""
        # OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
        if age_group <= 8:
            return "nova"  # Softer, friendlier voice
        elif age_group <= 12:
            return "alloy"  # Balanced, clear voice
        else:
            return "onyx"  # More mature voice


stt_service = STTService()
tts_service = TTSService()

def transcribe_audio(audio_data: bytes, content_type: str = "audio/ogg") -> Optional[str]:
    """Convenience function to transcribe audio."""
    return stt_service.transcribe(audio_data, content_type)

def synthesize_speech(text: str, voice: str = "alloy", age_group: int = 10) -> Optional[Tuple[bytes, str]]:
    """Convenience function to synthesize speech."""
    return tts_service.synthesize(text, voice, age_group)

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
