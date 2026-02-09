"""
Image generation service using Hugging Face Inference API.
Generates educational illustrations for lesson topics.
"""
import os
import io
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# WhatsApp image limit: 5 MB
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
TARGET_SIZE = (768, 768)  # Keep under 5MB as JPEG
JPEG_QUALITY = 85


def _build_educational_prompt(topic: str) -> str:
    """Build an educational image prompt from lesson topic."""
    return (
        f"Educational illustration of {topic}, simple, colorful, "
        "suitable for children, diagram style, clean background, "
        "friendly and informative"
    )


def _ensure_under_size(image_bytes: bytes, content_type: str) -> Tuple[bytes, str]:
    """Resize/compress image to stay under WhatsApp 5MB limit."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL not installed, skipping image resize")
        if len(image_bytes) <= MAX_IMAGE_SIZE_BYTES:
            return (image_bytes, content_type)
        return (image_bytes[:MAX_IMAGE_SIZE_BYTES], content_type)  # Truncate as fallback

    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        result = out.getvalue()

        while len(result) > MAX_IMAGE_SIZE_BYTES and JPEG_QUALITY > 20:
            q = JPEG_QUALITY - 10
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=q, optimize=True)
            result = out.getvalue()
            if q <= 20:
                break

        logger.info(f"Image compressed: {len(image_bytes)} -> {len(result)} bytes")
        return (result, "image/jpeg")
    except Exception as e:
        logger.warning(f"Image resize failed: {e}")
        if len(image_bytes) <= MAX_IMAGE_SIZE_BYTES:
            return (image_bytes, content_type)
        return (image_bytes, content_type)


class ImageService:
    """Image generation using Hugging Face Inference API."""

    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.use_hf = bool(self.hf_token)
        self._client = None
        if self.use_hf:
            self._init_client()
        else:
            logger.info("HF_TOKEN not set; image generation disabled")

    def _init_client(self) -> None:
        try:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=self.hf_token)
            logger.info("Hugging Face InferenceClient initialized for image generation")
        except ImportError as e:
            logger.warning(f"Could not import InferenceClient: {e}")
            self._client = None
            self.use_hf = False
        except Exception as e:
            logger.warning(f"Failed to init HF InferenceClient: {e}")
            self._client = None
            self.use_hf = False

    def generate(self, topic: str, prompt_override: Optional[str] = None) -> Optional[Tuple[bytes, str]]:
        """
        Generate an educational image for the given topic.

        Args:
            topic: Lesson topic (e.g., "cells", "photosynthesis")
            prompt_override: Optional custom prompt; if None, uses default educational prompt.

        Returns:
            Tuple of (image_bytes, content_type) or None if generation fails.
        """
        if not self.use_hf or not self._client:
            return None

        prompt = prompt_override or _build_educational_prompt(topic)
        try:
            image = self._client.text_to_image(
                prompt,
                model="stabilityai/stable-diffusion-2-1",
            )
            if image is None:
                logger.warning("HF text_to_image returned None")
                return None

            # InferenceClient returns PIL.Image
            buf = io.BytesIO()
            if hasattr(image, "save"):
                image.save(buf, format="PNG")
                raw = buf.getvalue()
            else:
                raw = image if isinstance(image, bytes) else bytes(image)

            result, content_type = _ensure_under_size(raw, "image/png")
            logger.info(f"Generated image for topic '{topic}': {len(result)} bytes")
            return (result, content_type)
        except Exception as e:
            logger.error(f"Image generation failed for topic '{topic}': {e}")
            return None


image_service = ImageService()


def generate_lesson_image(topic: str, language: str = "en") -> Optional[Tuple[bytes, str]]:
    """
    Generate an educational image for a lesson topic.

    Args:
        topic: Lesson topic
        language: User language (for future localized prompts)

    Returns:
        (image_bytes, content_type) or None
    """
    return image_service.generate(topic)
