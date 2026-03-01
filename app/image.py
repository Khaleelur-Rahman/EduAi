"""
Image generation service with provider fallback chain:
  1. Cloudflare Workers AI (CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN) -- FLUX/SDXL, 10k free/day
  2. HuggingFace  (HF_TOKEN) -- SDXL via hf-inference, limited free tier
"""
import os
import io
import base64
import logging
import requests as _requests
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
TARGET_SIZE = (768, 768)
JPEG_QUALITY = 85

_NO_TEXT_SUFFIX = ", no text, no words, no letters, no writing, no labels, no captions, no watermarks"


def _build_educational_prompt(topic: str) -> str:
    return (
        f"Educational illustration of {topic}, simple, colorful, "
        "suitable for children, diagram style, clean background, "
        "friendly and informative"
    )


def _ensure_under_size(image_bytes: bytes, content_type: str) -> Tuple[bytes, str]:
    try:
        from PIL import Image
    except ImportError:
        if len(image_bytes) <= MAX_IMAGE_SIZE_BYTES:
            return (image_bytes, content_type)
        return (image_bytes[:MAX_IMAGE_SIZE_BYTES], content_type)
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        result = out.getvalue()
        q = JPEG_QUALITY
        while len(result) > MAX_IMAGE_SIZE_BYTES and q > 20:
            q -= 10
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=q, optimize=True)
            result = out.getvalue()
        return (result, "image/jpeg")
    except Exception:
        return (image_bytes, content_type) if len(image_bytes) <= MAX_IMAGE_SIZE_BYTES else (image_bytes, content_type)


def _pil_to_jpeg(pil_image) -> Optional[bytes]:
    buf = io.BytesIO()
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")
    pil_image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

class _CloudflareProvider:
    """Cloudflare Workers AI -- SDXL primary (supports negative_prompt), FLUX fallback."""

    _NEGATIVE = (
        "text, words, letters, writing, labels, captions, watermark, signature, "
        "title, subtitle, logo, font, typography, numbers, digits, symbols, "
        "blurry, low quality, deformed"
    )

    def __init__(self):
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        self.enabled = bool(self.account_id and self.api_token)
        if self.enabled:
            logger.info("Cloudflare Workers AI image provider ready")

    def generate(self, prompt: str) -> Optional[bytes]:
        if not self.enabled:
            return None
        safe = prompt + _NO_TEXT_SUFFIX
        headers = {"Authorization": f"Bearer {self.api_token}"}
        base = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run"

        # SDXL first -- supports negative_prompt to suppress text
        try:
            r = _requests.post(
                f"{base}/@cf/stabilityai/stable-diffusion-xl-base-1.0",
                json={"prompt": safe, "negative_prompt": self._NEGATIVE},
                headers=headers, timeout=60,
            )
            r.raise_for_status()
            img = self._extract(r)
            if img:
                return img
        except Exception as e:
            logger.warning("CF SDXL failed: %s", e)

        # FLUX fallback
        try:
            r = _requests.post(
                f"{base}/@cf/black-forest-labs/flux-1-schnell",
                json={"prompt": safe},
                headers=headers, timeout=60,
            )
            r.raise_for_status()
            img = self._extract(r)
            if img:
                return img
        except Exception as e:
            logger.warning("CF FLUX failed: %s", e)

        return None

    @staticmethod
    def _extract(r) -> Optional[bytes]:
        ct = r.headers.get("Content-Type", "")
        if "image" in ct:
            return r.content
        try:
            data = r.json()
            result = data.get("result", {})
            if isinstance(result, dict) and "image" in result:
                return base64.b64decode(result["image"])
        except Exception:
            pass
        return None


class _HuggingFaceProvider:
    """HuggingFace Inference API -- SDXL via hf-inference provider."""

    MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self):
        self.token = os.getenv("HF_TOKEN", "").strip()
        self.enabled = bool(self.token)
        self._client = None
        if self.enabled:
            self._init()

    def _init(self):
        try:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=self.token, provider="hf-inference")
            logger.info("HuggingFace image provider ready (hf-inference)")
        except Exception as e:
            logger.warning("HF InferenceClient init failed: %s", e)
            self.enabled = False

    def generate(self, prompt: str) -> Optional[bytes]:
        if not self.enabled or not self._client:
            return None
        try:
            image = self._client.text_to_image(
                prompt + _NO_TEXT_SUFFIX, model=self.MODEL,
            )
            if image is None:
                return None
            return _pil_to_jpeg(image)
        except Exception as e:
            logger.warning("HF image failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class ImageService:
    """Image generation with automatic provider fallback."""

    def __init__(self):
        self._cloudflare = _CloudflareProvider()
        self._hf = _HuggingFaceProvider()

        self._providers = [p for p in [self._cloudflare, self._hf] if p.enabled]
        names = [type(p).__name__.strip("_") for p in self._providers]
        if names:
            logger.info("Image providers (priority order): %s", " -> ".join(names))
        else:
            logger.warning("No image providers configured")

    @property
    def enabled(self) -> bool:
        return len(self._providers) > 0

    def generate(self, topic: str, prompt_override: Optional[str] = None) -> Optional[Tuple[bytes, str]]:
        prompt = prompt_override or _build_educational_prompt(topic)
        raw = self._generate_raw(prompt)
        if raw is None:
            return None
        return _ensure_under_size(raw, "image/jpeg")

    def generate_from_prompt(self, prompt: str) -> Optional[bytes]:
        """Generate from an exact prompt. Returns raw JPEG bytes or None."""
        return self._generate_raw(prompt)

    def _generate_raw(self, prompt: str) -> Optional[bytes]:
        for provider in self._providers:
            result = provider.generate(prompt)
            if result and len(result) > 100:
                return result
        return None


image_service = ImageService()


def generate_lesson_image(topic: str, language: str = "en") -> Optional[Tuple[bytes, str]]:
    return image_service.generate(topic)
