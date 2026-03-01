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

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead", category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class STTService:
    """Speech-to-Text service using local Whisper."""

    def __init__(self):
        self.local_whisper_model = None
        logger.info("Initializing local Whisper model (this may take a moment on first use)...")
        self._init_local_whisper()

    def _init_local_whisper(self):
        try:
            import whisper
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
        try:
            if self.local_whisper_model:
                return self._transcribe_local(audio_data, content_type)
            else:
                logger.error("No STT service available")
                return None
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return None

    def _transcribe_local(self, audio_data: bytes, content_type: str) -> Optional[str]:
        if not self.local_whisper_model:
            return None
        try:
            import whisper
            with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_suffix_from_content_type(content_type)) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            try:
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

    def _get_suffix_from_content_type(self, content_type: str) -> str:
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
    """Text-to-Speech service using edge-tts (Microsoft Edge TTS)."""

    def __init__(self):
        self.edge_tts = None
        logger.info("Initializing edge-tts (Microsoft Edge TTS)...")
        self._init_edge_tts()

    def _init_edge_tts(self):
        try:
            import edge_tts
            self.edge_tts = edge_tts
            logger.info("edge-tts initialized successfully (free, requires internet)")
        except ImportError:
            logger.warning("edge-tts package not installed. TTS will not be available.")
            logger.info("  Install edge-tts: pip install edge-tts")
            self.edge_tts = None
        except Exception as e:
            logger.error(f"Failed to initialize edge-tts: {e}")
            self.edge_tts = None

    @property
    def enabled(self) -> bool:
        return self.edge_tts is not None

    def synthesize(self, text: str, voice: str = "alloy", age_group: int = 10, language: str = "en") -> Optional[Tuple[bytes, str]]:
        try:
            text = self._text_to_audio_friendly(text)
            text = self._adjust_text_for_age(text, age_group)
            if self.edge_tts:
                return self._synthesize_edge_tts(text, voice, age_group, language)
            else:
                logger.error("No TTS service available")
                return None
        except Exception as e:
            logger.error(f"Error during TTS synthesis: {e}")
            return None

    def _text_to_audio_friendly(self, text: str) -> str:
        if not text or not text.strip():
            return text
        t = text.strip()
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
        t = re.sub(r'📚\s*\*Lesson:\s*([^*]+)\*', r'Lesson: \1.', t, flags=re.IGNORECASE)
        t = re.sub(r'\*([^*]+)\s*-\s*Part\s+(\d+)\*', r'Part \2. \1.', t, flags=re.IGNORECASE)
        t = re.sub(r'\*([^*]+)\*', r'\1', t)
        t = re.sub(r'_([^_]+)_', r'\1', t)
        t = re.sub(r'^#+\s*([^\n#]+)\s*$', r'\1.', t, flags=re.MULTILINE)
        t = re.sub(r'^#+\s*([^\n#]+)\s*\n', r'\1. ', t, flags=re.MULTILINE)

        def _bullet_to_sentence(match: re.Match) -> str:
            rest = (match.group(1) or '').strip()
            if not rest:
                return ''
            if rest.endswith('.'):
                return rest + ' '
            return rest.rstrip('.,;:') + '. '

        t = re.sub(r'^[\s]*[•\-*]\s+([^\n]+)', _bullet_to_sentence, t, flags=re.MULTILINE)
        t = re.sub(r'^[\s]*\d+\.\s+([^\n]+)(?=\n|$)', lambda m: (m.group(1).strip().rstrip('.') or '') + '. ', t, flags=re.MULTILINE)
        t = re.sub(r'[\U0001F300-\U0001F9FF\U00002700-\U000027BF]', '', t)
        t = re.sub(r'\n\s*\n+', ' ', t)
        t = re.sub(r'\n', ' ', t)
        t = re.sub(r' {2,}', ' ', t)
        t = re.sub(r'\s+\.', '.', t)
        t = t.strip()
        if t and t[-1] not in '.!?':
            t = t.rstrip(';:,') + '.'
        return t

    def _adjust_text_for_age(self, text: str, age_group: int) -> str:
        if age_group <= 8:
            pass
        return text

    def _synthesize_edge_tts(self, text: str, voice: str = "alloy", age_group: int = 10, language: str = "en") -> Optional[Tuple[bytes, str]]:
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

            try:
                loop = asyncio.get_running_loop()
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
                return (audio_bytes, "audio/mpeg")
            else:
                logger.error("edge-tts returned empty audio")
                return None

        except Exception as e:
            logger.error(f"edge-tts synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return None


stt_service = STTService()
tts_service = TTSService()


def transcribe_audio(audio_data: bytes, content_type: str = "audio/ogg", language: Optional[str] = None) -> Optional[str]:
    return stt_service.transcribe(audio_data, content_type, language=language)


def synthesize_speech(text: str, voice: str = "alloy", age_group: int = 10) -> Optional[Tuple[bytes, str]]:
    return tts_service.synthesize(text, voice, age_group)


def _chunk_text_for_audio(
    text: str,
    max_sentences_per_chunk: int = 4,
    max_words_per_chunk: int = 120,
) -> List[str]:
    if not text or not text.strip():
        return []
    text = text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]
    chunks = []
    current: List[str] = []
    current_words = 0
    for sent in sentences:
        n = len(sent.split())
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
        if len(chunks) > max_segments:
            mid = (len(chunks) + 1) // 2
            chunks = [" ".join(chunks[:mid]), " ".join(chunks[mid:])]

        results: List[Tuple[bytes, str]] = []

        def synthesize_chunk(chunk_idx: int, chunk_text: str) -> Optional[Tuple[bytes, str]]:
            try:
                if tts_service.edge_tts:
                    return tts_service._synthesize_edge_tts(chunk_text, voice, age_group, language)
                return None
            except Exception as e:
                logger.warning(f"Chunk {chunk_idx + 1} synthesis failed: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = {
                executor.submit(synthesize_chunk, i, chunk): i
                for i, chunk in enumerate(chunks)
            }
            chunk_results = [None] * len(chunks)
            for future in concurrent.futures.as_completed(futures):
                chunk_idx = futures[future]
                try:
                    result = future.result()
                    chunk_results[chunk_idx] = result
                except Exception as e:
                    logger.warning(f"Chunk {chunk_idx + 1} future failed: {e}")
                    chunk_results[chunk_idx] = None

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
    return tts_service._text_to_audio_friendly(text)


def initialize_audio_services():
    try:
        if stt_service.local_whisper_model:
            logger.info("STT: Local Whisper model ready")
        else:
            logger.warning("STT: No service available")

        if tts_service.edge_tts:
            logger.info("TTS: edge-tts ready (free, requires internet)")
        else:
            logger.warning("TTS: No service available")
            logger.info("  To enable TTS: pip install edge-tts")
    except Exception as e:
        logger.error(f"Error during audio services initialization: {e}")


if __name__ == "__main__":
    print("Audio services module loaded successfully!")
    print(f"STT Service: {'Local Whisper' if stt_service.local_whisper_model else 'Not available'}")
    print(f"TTS Service: {'edge-tts' if tts_service.edge_tts else 'Not available'}")
