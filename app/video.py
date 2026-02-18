"""
Video generation: short-video-maker only (Cerebras narration + TTS + Pexels + Remotion).
"""
import os
import logging
import subprocess
import tempfile
import time
import requests
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# Twilio WhatsApp video limit: 16 MB
MAX_VIDEO_SIZE_BYTES = 16 * 1024 * 1024
TARGET_VIDEO_SIZE_BYTES = int(MAX_VIDEO_SIZE_BYTES * 0.92)  # ~15 MB


def _compress_video_to_fit(video_bytes: bytes, max_bytes: int = TARGET_VIDEO_SIZE_BYTES) -> Optional[bytes]:
    """
    Re-encode video with ffmpeg to fit under max_bytes. Keeps audio.
    Returns compressed bytes or None if ffmpeg unavailable / fails.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fin:
        fin.write(video_bytes)
        input_path = fin.name
    try:
        try:
            out = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    input_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode != 0 or not out.stdout or not out.stdout.strip():
                logger.warning("ffprobe could not get duration: %s", out.stderr or out.stdout)
                duration_sec = 30.0
            else:
                duration_sec = float(out.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
            logger.warning("ffprobe failed for compression: %s", e)
            duration_sec = 30.0
        if duration_sec <= 0:
            duration_sec = 30.0
        target_bits = max_bytes * 8
        audio_bits = 128_000 * duration_sec
        video_bits = max(0, target_bits - audio_bits)
        video_k = int(video_bits / duration_sec / 1000)
        video_k = max(200, min(4000, video_k))
        out_path = input_path + ".out.mp4"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", input_path,
                    "-c:v", "libx264", "-b:v", f"{video_k}k", "-maxrate", f"{min(4500, video_k + 500)}k",
                    "-bufsize", "1000k", "-c:a", "aac", "-b:a", "96k",
                    "-movflags", "+faststart", out_path,
                ],
                capture_output=True,
                timeout=120,
                check=True,
            )
            with open(out_path, "rb") as f:
                result = f.read()
            if len(result) <= max_bytes:
                return result
            logger.warning("Compressed video still too large: %s (target %s)", len(result), max_bytes)
            return result if len(result) <= MAX_VIDEO_SIZE_BYTES else None
        except FileNotFoundError:
            logger.warning("ffmpeg not found; cannot compress oversized video")
            return None
        except subprocess.CalledProcessError as e:
            logger.warning("ffmpeg compression failed: %s", e.stderr or e)
            return None
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg compression timed out")
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
    """If video is over Twilio limit, compress it. Return bytes to use (under limit) or None on failure."""
    if len(video_bytes) <= MAX_VIDEO_SIZE_BYTES:
        return video_bytes
    logger.info("Compressing video from %s to under %s bytes", len(video_bytes), MAX_VIDEO_SIZE_BYTES)
    compressed = _compress_video_to_fit(video_bytes)
    if compressed and len(compressed) <= MAX_VIDEO_SIZE_BYTES:
        logger.info("Compressed video to %s bytes", len(compressed))
        return compressed
    if compressed:
        logger.warning("Compressed video still over limit: %s", len(compressed))
    return None


def _build_short_video_script(topic: str) -> str:
    """Fallback narration when no LLM script is provided (one scene)."""
    t = topic.strip().title()
    return f"Today we're learning about {t}. This is an important concept to understand."


def _build_search_terms(topic: str) -> List[str]:
    """Search terms for Pexels footage in short-video-maker."""
    t = topic.strip().lower()
    return [t, "science", "nature"]


class VideoService:
    """Video generation via short-video-maker (Cerebras narration + TTS + Pexels + Remotion)."""

    def __init__(self):
        self.short_video_url = (
            os.getenv("SHORT_VIDEO_MAKER_URL") or os.getenv("short_video_maker_url") or ""
        ).strip().rstrip("/")
        self.poll_interval = int(os.getenv("SHORT_VIDEO_MAKER_POLL_INTERVAL", "10"))
        self.poll_timeout = int(os.getenv("SHORT_VIDEO_MAKER_TIMEOUT", "300"))
        self.enabled = bool(self.short_video_url)
        if not self.enabled:
            logger.info("Video disabled: set SHORT_VIDEO_MAKER_URL to enable")
        else:
            logger.info("Video backend: short-video-maker at %s", self.short_video_url)

    def generate(
        self,
        topic: str,
        prompt_override: Optional[str] = None,
        language: str = "en",
    ) -> Optional[Tuple[bytes, str]]:
        """
        Generate an educational video for the given topic.

        Returns:
            Tuple of (video_bytes, "video/mp4") or None if generation fails.
        """
        if not self.enabled:
            return None
        if language and language.lower() != "en":
            logger.info("Video is English-only; skipping for language %s", language)
            return None
        return self._generate_via_short_video_maker(topic, prompt_override)

    def _generate_via_short_video_maker(
        self, topic: str, script_override: Optional[str] = None
    ) -> Optional[Tuple[bytes, str]]:
        """Create video via short-video-maker: POST → poll status → GET binary."""
        text = script_override or _build_short_video_script(topic)
        search_terms = _build_search_terms(topic)
        payload = {
            "scenes": [{"text": text, "searchTerms": search_terms}],
            "config": {},
        }
        url = f"{self.short_video_url}/api/short-video"

        try:
            logger.info("Requesting video from short-video-maker for topic '%s'", topic)
            r = requests.post(
                url,
                json=payload,
                timeout=60,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
            video_id = data.get("videoId")
            if not video_id:
                logger.error("short-video-maker did not return videoId: %s", data)
                return None

            logger.info(
                "short-video-maker job %s: polling status every %ss (timeout %ss)",
                video_id,
                self.poll_interval,
                self.poll_timeout,
            )
            status_url = f"{self.short_video_url}/api/short-video/{video_id}/status"
            video_url = f"{self.short_video_url}/api/short-video/{video_id}"
            deadline = time.monotonic() + self.poll_timeout
            poll_count = 0

            while time.monotonic() < deadline:
                poll_count += 1
                sr = requests.get(status_url, timeout=90)
                sr.raise_for_status()
                status_data = sr.json()
                st = status_data.get("status", "").lower()
                elapsed = int(time.monotonic() - (deadline - self.poll_timeout))
                logger.info(
                    "short-video-maker status: %s (poll %s, %ss elapsed)",
                    st or "(empty)",
                    poll_count,
                    elapsed,
                )
                if st == "ready":
                    break
                if st == "failed" or "error" in st:
                    logger.warning("short-video-maker status failed: %s", status_data)
                    return None
                time.sleep(self.poll_interval)

            else:
                logger.error("short-video-maker timed out after %ss", self.poll_timeout)
                return None

            vr = requests.get(video_url, timeout=60)
            vr.raise_for_status()
            video_bytes = vr.content
            if not video_bytes:
                logger.warning("short-video-maker returned empty video")
                return None
            final_bytes = _ensure_video_under_limit(video_bytes)
            if final_bytes is None:
                logger.error(
                    "Video size %s exceeds Twilio limit %s and compression failed or unavailable",
                    len(video_bytes),
                    MAX_VIDEO_SIZE_BYTES,
                )
                return None
            logger.info("Generated video for topic '%s': %s bytes", topic, len(final_bytes))
            return (final_bytes, "video/mp4")

        except requests.exceptions.Timeout:
            logger.error("short-video-maker request timed out for topic '%s'", topic)
            return None
        except requests.exceptions.RequestException as e:
            logger.error("short-video-maker request failed for topic '%s': %s", topic, e)
            return None
        except Exception as e:
            logger.error("Video generation failed for topic '%s': %s", topic, e)
            return None


video_service = VideoService()


def generate_lesson_video(
    topic: str,
    language: str = "en",
    script_override: Optional[str] = None,
) -> Optional[Tuple[bytes, str]]:
    """
    Generate an educational video for a lesson topic (short-video-maker only).

    Args:
        topic: Lesson topic.
        language: User language (English only).
        script_override: Narration text from Cerebras (or fallback); longer script = longer video.

    Returns:
        (video_bytes, "video/mp4") or None.
    """
    return video_service.generate(
        topic,
        prompt_override=script_override,
        language=language,
    )
