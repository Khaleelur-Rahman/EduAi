"""
Video generation: hybrid pipeline (LLM + AI images + TTS + ffmpeg).
"""
import os
import re
import logging
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

MAX_VIDEO_SIZE_BYTES = 16 * 1024 * 1024  # Twilio WhatsApp limit
TARGET_VIDEO_SIZE_BYTES = int(MAX_VIDEO_SIZE_BYTES * 0.92)

VIDEO_FPS = 25
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
MIN_IMAGES_REQUIRED = 2


def _compress_video_to_fit(video_bytes: bytes, max_bytes: int = TARGET_VIDEO_SIZE_BYTES) -> Optional[bytes]:
    """Re-encode video with ffmpeg to fit under max_bytes."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fin:
        fin.write(video_bytes)
        input_path = fin.name
    try:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", input_path],
                capture_output=True, text=True, timeout=10,
            )
            duration_sec = float(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else 30.0
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            duration_sec = 30.0
        if duration_sec <= 0:
            duration_sec = 30.0
        target_bits = max_bytes * 8
        audio_bits = 128_000 * duration_sec
        video_bits = max(0, target_bits - audio_bits)
        video_k = max(200, min(4000, int(video_bits / duration_sec / 1000)))
        out_path = input_path + ".out.mp4"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path,
                 "-c:v", "libx264", "-b:v", f"{video_k}k",
                 "-maxrate", f"{min(4500, video_k + 500)}k",
                 "-bufsize", "1000k", "-c:a", "aac", "-b:a", "96k",
                 "-movflags", "+faststart", out_path],
                capture_output=True, timeout=120, check=True,
            )
            with open(out_path, "rb") as f:
                result = f.read()
            if len(result) <= max_bytes:
                return result
            return result if len(result) <= MAX_VIDEO_SIZE_BYTES else None
        except FileNotFoundError:
            return None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        finally:
            if os.path.exists(out_path):
                try:
                    os.unlink(out_path)
                except OSError:
                    pass
    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass
    return None


def _ensure_video_under_limit(video_bytes: bytes) -> Optional[bytes]:
    if len(video_bytes) <= MAX_VIDEO_SIZE_BYTES:
        return video_bytes
    compressed = _compress_video_to_fit(video_bytes)
    if compressed and len(compressed) <= MAX_VIDEO_SIZE_BYTES:
        return compressed
    return None


def _get_audio_duration(audio_path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 25.0


# ---------- Subtitle rendering via PIL ----------

_SUBTITLE_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def _get_subtitle_font(font_size: int, language: str = "en"):
    """Load a font that can render the given language (Latin, CJK, Devanagari, etc.)."""
    from PIL import ImageFont
    paths = _SUBTITLE_FONT_PATHS if language in ("zh", "ja", "ko", "hi") else _SUBTITLE_FONT_PATHS[:3]
    for path in paths:
        try:
            return ImageFont.truetype(path, font_size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _render_subtitle_image(
    text: str, width: int, height: int, out_path: str, language: str = "en"
) -> bool:
    """Render one subtitle sentence as a transparent PNG overlay."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    font_size = max(22, height // 22)
    font = _get_subtitle_font(font_size, language)
    draw = ImageDraw.Draw(img)

    max_text_w = int(width * 0.88)
    lines: List[str] = []
    if language in ("zh", "ja", "ko"):
        current = ""
        for char in text:
            test = current + char
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] > max_text_w and current:
                lines.append(current)
                current = char
            else:
                current = test
        if current:
            lines.append(current)
    else:
        words = text.split()
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] > max_text_w and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)
    if not lines:
        return False

    line_h = font_size + 8
    block_h = line_h * len(lines) + 16
    y_start = height - block_h - 40

    pad = 14
    draw.rectangle(
        [(width * 0.04, y_start - pad), (width * 0.96, y_start + block_h + pad)],
        fill=(0, 0, 0, 170),
    )

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (width - tw) // 2
        y = y_start + i * line_h
        draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0, 220))
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))

    img.save(out_path, "PNG")
    return True


# ---------- Timed sentence splitting ----------

def _time_sentences(narration: str, audio_duration: float, language: str = "en") -> List[dict]:
    """
    Split narration into sentences with proportional timing.
    For languages without spaces (zh, ja, ko, hi), uses character-based weight.
    Returns [{"text": str, "start": float, "end": float}, ...].
    """
    # Split on sentence-ending punctuation; for CJK etc. also split on。！？
    text = narration.strip()
    if not text:
        return []
    separators = r'(?<=[.!?])\s+'
    if language in ("zh", "ja", "ko"):
        separators = r'(?<=[.!?。！？])\s*'
    sentences = re.split(separators, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    if language in ("zh", "ja", "ko", "hi", "th"):
        weights = [max(1, len(s) // 3) for s in sentences]
    else:
        weights = [len(s.split()) for s in sentences]
    total = sum(weights) or 1
    result = []
    cursor = 0.0
    for sentence, w in zip(sentences, weights):
        dur = (w / total) * audio_duration
        result.append({"text": sentence, "start": cursor, "end": cursor + dur})
        cursor += dur
    return result


def _group_sentences_by_image(
    timed_sentences: List[dict], num_images: int,
) -> List[List[dict]]:
    """
    Split sentences into sequential groups, one per image.
    If there are fewer sentences than images, only use as many images as sentences.
    """
    n = len(timed_sentences)
    if n == 0 or num_images == 0:
        return []
    k = min(n, num_images)
    groups: List[List[dict]] = []
    base, extra = divmod(n, k)
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        groups.append(timed_sentences[start:start + size])
        start += size
    return groups


# ---------- Per-sentence clip creation ----------

_STATIC_VF = (
    f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
    f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},"
    f"fps={VIDEO_FPS}"
)


def _make_sentence_clip(
    image_path: str, clip_path: str, duration: float,
    subtitle_png: str,
) -> bool:
    """Create a static image clip with optional subtitle overlay."""
    if subtitle_png and os.path.exists(subtitle_png):
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loop", "1", "-i", image_path,
                    "-i", subtitle_png,
                    "-t", f"{duration:.3f}",
                    "-filter_complex",
                    f"[0:v]{_STATIC_VF}[bg];[bg][1:v]overlay=0:0",
                    "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-an", clip_path,
                ],
                capture_output=True, timeout=90, check=True,
            )
            return os.path.exists(clip_path)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("Sentence clip with subtitle failed: %s", e)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loop", "1", "-i", image_path,
                "-t", f"{duration:.3f}",
                "-vf", _STATIC_VF,
                "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                "-an", clip_path,
            ],
            capture_output=True, timeout=60, check=True,
        )
        return os.path.exists(clip_path)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("Sentence clip failed: %s", e)
        return False


# ---------- Final assembly ----------

def _concat_clips_with_audio(
    clip_paths: List[str], audio_path: str, output_path: str, audio_offset_seconds: float = 0.0
) -> bool:
    """Concatenate video clips and overlay audio track. audio_offset_seconds delays audio (e.g. for lead-in)."""
    concat_file = output_path + ".concat.txt"
    try:
        with open(concat_file, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")
        # All inputs first: concat (video), then audio (optionally delayed)
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file]
        if audio_offset_seconds and audio_offset_seconds > 0:
            cmd.extend(["-itsoffset", str(audio_offset_seconds), "-i", audio_path])
        else:
            cmd.extend(["-i", audio_path])
        # Explicit mapping: video from input 0, audio from input 1 (avoids exit 8 with two inputs)
        cmd.extend([
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", "-shortest", output_path,
        ])
        subprocess.run(cmd, capture_output=True, timeout=180, check=True)
        return os.path.exists(output_path)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("Concat + audio merge failed: %s", e)
        return False
    finally:
        if os.path.exists(concat_file):
            try:
                os.unlink(concat_file)
            except OSError:
                pass


# ---------- Main service ----------

class VideoService:
    """Video generation via hybrid pipeline (AI images + TTS + ffmpeg)."""

    def __init__(self):
        self._has_ffmpeg = shutil.which("ffmpeg") is not None

        from .image import image_service
        self._has_images = image_service.enabled

        self.enabled = self._has_ffmpeg and self._has_images

        if not self.enabled:
            logger.info("Video disabled: need image provider + ffmpeg")
        else:
            logger.info("Video backend: hybrid pipeline (AI images + TTS + ffmpeg)")

    def generate(
        self,
        topic: str,
        prompt_override: Optional[str] = None,
        language: str = "en",
        age_group: int = 10,
    ) -> Optional[Tuple[bytes, str, str]]:
        """Returns (video_bytes, content_type, narration_script) or None. Narration is used for quiz-after-video."""
        if not self.enabled:
            return None
        from .language import SUPPORTED_LANGUAGES
        lang = (language or "en").strip().lower()
        if lang not in SUPPORTED_LANGUAGES:
            lang = "en"
        return self._generate_hybrid_video(topic, lang, age_group)

    def _generate_hybrid_video(
        self, topic: str, language: str = "en", age_group: int = 10
    ) -> Optional[Tuple[bytes, str, str]]:
        """
        Pipeline: LLM script (in target language) -> AI images + TTS in parallel
        -> per-sentence static clips with subtitles -> concat + audio.
        Returns (video_bytes, content_type, narration_script) or None.
        """
        from .llm import generate_video_script
        from .image import image_service
        from .audio import tts_service

        tmpdir = tempfile.mkdtemp(prefix="eduai_video_")
        try:
            script = generate_video_script(topic, age_group=age_group, num_images=4, language=language)
            narration = script["narration"]
            image_prompts = script["image_prompts"]
            num_prompts = len(image_prompts)
            logger.info("Hybrid video for '%s' (%s): %d prompts, narration %d chars",
                        topic, language, num_prompts, len(narration))

            image_bytes_list: List[Optional[bytes]] = [None] * num_prompts
            audio_result = [None]

            def gen_image(idx: int, prompt: str):
                return idx, image_service.generate_from_prompt(prompt)

            def gen_audio():
                return tts_service.synthesize(narration, age_group=age_group, language=language)

            with ThreadPoolExecutor(max_workers=num_prompts + 1) as pool:
                futures = []
                for i, prompt in enumerate(image_prompts):
                    futures.append(pool.submit(gen_image, i, prompt))
                audio_future = pool.submit(gen_audio)

                for fut in as_completed(futures):
                    try:
                        idx, img_bytes = fut.result(timeout=60)
                        image_bytes_list[idx] = img_bytes
                    except Exception as e:
                        logger.warning("Image generation failed: %s", e)

                try:
                    audio_result[0] = audio_future.result(timeout=60)
                except Exception as e:
                    logger.error("TTS synthesis failed: %s", e)

            valid_images = [(i, b) for i, b in enumerate(image_bytes_list) if b is not None]
            if len(valid_images) < MIN_IMAGES_REQUIRED:
                logger.warning("Only %d images (need %d), aborting", len(valid_images), MIN_IMAGES_REQUIRED)
                return None
            if audio_result[0] is None:
                logger.warning("TTS failed, aborting")
                return None

            audio_bytes, audio_ct = audio_result[0]
            audio_ext = ".mp3" if "mpeg" in audio_ct else ".wav"
            audio_path = os.path.join(tmpdir, f"narration{audio_ext}")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            img_paths: List[str] = []
            for orig_idx, img_bytes in valid_images:
                p = os.path.join(tmpdir, f"img_{orig_idx}.jpg")
                with open(p, "wb") as f:
                    f.write(img_bytes)
                img_paths.append(p)

            num_images = len(img_paths)
            audio_duration = _get_audio_duration(audio_path)
            logger.info("Audio: %.1fs, %d images", audio_duration, num_images)

            timed = _time_sentences(narration, audio_duration, language)
            if not timed:
                logger.warning("No sentences parsed from narration")
                return None

            groups = _group_sentences_by_image(timed, num_images)

            clip_paths = []
            sent_counter = 0
            for img_idx, group in enumerate(groups):
                for s in group:
                    dur = max(0.5, s["end"] - s["start"])
                    sub_png = os.path.join(tmpdir, f"sub_{sent_counter}.png")
                    _render_subtitle_image(s["text"], VIDEO_WIDTH, VIDEO_HEIGHT, sub_png, language)
                    clip_path = os.path.join(tmpdir, f"clip_{sent_counter}.mp4")
                    if _make_sentence_clip(img_paths[img_idx], clip_path, dur, sub_png):
                        clip_paths.append(clip_path)
                    sent_counter += 1

            if len(clip_paths) < 2:
                logger.warning("Too few clips (%d), aborting", len(clip_paths))
                return None

            # Prepend a short lead-in with the first image so the video doesn't start with a black frame
            LEAD_IN_SECONDS = 0.35
            lead_in_path = os.path.join(tmpdir, "lead_in.mp4")
            if _make_sentence_clip(img_paths[0], lead_in_path, LEAD_IN_SECONDS, ""):
                clip_paths = [lead_in_path] + clip_paths
            else:
                LEAD_IN_SECONDS = 0.0

            output_path = os.path.join(tmpdir, "output.mp4")
            if not _concat_clips_with_audio(clip_paths, audio_path, output_path, audio_offset_seconds=LEAD_IN_SECONDS):
                logger.error("Final concat failed")
                return None

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            final = _ensure_video_under_limit(video_bytes)
            if final is None:
                logger.error("Video over Twilio limit")
                return None

            logger.info("Hybrid video for '%s': %s bytes", topic, len(final))
            return (final, "video/mp4", narration)

        except Exception as e:
            logger.error("Hybrid pipeline failed for '%s': %s", topic, e)
            return None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


video_service = VideoService()


def generate_lesson_video(
    topic: str,
    language: str = "en",
    age_group: int = 10,
    script_override: Optional[str] = None,
) -> Optional[Tuple[bytes, str, str]]:
    """Generate an educational video; returns (video_bytes, content_type, narration_script) or None."""
    return video_service.generate(
        topic, prompt_override=script_override, language=language, age_group=age_group
    )
