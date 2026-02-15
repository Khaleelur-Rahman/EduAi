"""
Image generation service: Cloudflare Workers AI (SDXL) or Hugging Face Inference API.
Generates educational illustrations for lesson topics.

Cloudflare (recommended, 10k free neurons/day): set CLOUDFLARE_ACCOUNT_ID and
CLOUDFLARE_API_TOKEN (from Cloudflare dashboard → Workers AI → API keys).
Hugging Face fallback: set HF_TOKEN. If both are set, Cloudflare is used first.
"""
import os
import io
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

CLOUDFLARE_SDXL_URL_TEMPLATE = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/stabilityai/stable-diffusion-xl-base-1.0"
)

# WhatsApp image limit: 5 MB
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
TARGET_SIZE = (768, 768)  # Keep under 5MB as JPEG
JPEG_QUALITY = 85


def _build_educational_prompt(topic: str) -> str:
    """Build an educational image prompt. """
    topic_clean = topic.lower().strip()

    # Lead and end with no-text rule; describe only visuals (no words to tempt the model)
    prompt = (
        "Absolutely no text in the image. No words, no letters, no labels, no writing. "
        "Pure visual diagram only: shapes, symbols, arrows, icons. "
        f"Clean educational illustration of {topic_clean}, "
        "simple layout, uncluttered, flat or subtle 3D, white or light background. "
    )

    for keyword, guidance in _TOPIC_KEYWORDS.items():
        if keyword in topic_clean:
            prompt = f"{prompt} {guidance}"
            break

    return prompt + " No text anywhere. Symbolic only."


_TOPIC_KEYWORDS = {
    "nitrogen": ": atmosphere, soil, plants, bacteria, arrows. Symbols only, no labels.",
    "oxygen": ": plants, lungs, atmosphere, arrows. Symbols only, no labels.",
    "carbon": ": plants, animals, atmosphere, arrows between them. Symbols only, no labels.",
    "cell": ": nucleus, mitochondria, membrane as shapes. No labels.",
    "photosynthesis": ": plant, sun, water, air flow, chloroplasts as shapes, arrows. No labels.",
    "transpiration": ": plants, sun, clouds, vapor, arrows from leaves. No labels.",
    "chlorophyll": ": chloroplast, green pigment, sunlight as shapes. No labels.",
    "atom": ": nucleus, electron shells as shapes. No labels.",
    "water": ": two small circles, one larger, bonds. No labels.",
    "plant": ": roots, stem, leaves as shapes. No labels.",
    "dna": ": double helix, spiral shape. No labels.",
    "ecosystem": ": plants, animals, sun, arrows. No labels.",
    "food chain": ": plant, herbivore, carnivore, arrows. No labels.",
    "solar system": ": sun and planets, orbits. No labels.",
    "molecule": ": circles and lines for atoms and bonds. No labels.",
}


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


# Hugging Face (fallback when Cloudflare not configured)
DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
IMAGE_MODEL_FALLBACKS = [
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5",
]

# Negative prompt for both providers (no text in image)
NEGATIVE_PROMPT = (
    "text, words, letters, labels, writing, captions, typography, "
    "words on the image, any writing, Carbon, crortme, recenim, persemte, anada, "
    "illegible, misspelled, gibberish, cluttered, busy, messy"
)


def _generate_via_cloudflare(prompt: str, negative_prompt: str) -> Optional[Tuple[bytes, str]]:
    """Generate image via Cloudflare Workers AI (SDXL). Returns (image_bytes, content_type) or None."""
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if not account_id or not token:
        return None
    url = CLOUDFLARE_SDXL_URL_TEMPLATE.format(account_id=account_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    }
    try:
        import base64
        import requests
        resp = requests.post(url, headers=headers, json=body, timeout=90)
        if resp.status_code != 200:
            logger.warning("Cloudflare Workers AI error: %s %s", resp.status_code, resp.text[:200])
            return None
        ct = resp.headers.get("Content-Type", "").lower()
        raw = resp.content
        if not raw:
            return None
        # Binary image response (common for image models)
        if "application/json" not in ct:
            content_type = "image/png" if "png" in ct else "image/jpeg"
            return (raw, content_type)
        # JSON response: may contain base64 image in result
        try:
            data = resp.json()
            result = data.get("result") or data
            if isinstance(result, dict):
                for key in ("image", "blob", "data", "response"):
                    b64 = result.get(key)
                    if isinstance(b64, str):
                        raw = base64.b64decode(b64)
                        if raw:
                            return (raw, "image/png")
            return None
        except Exception:
            logger.warning("Cloudflare returned JSON but no image in result: %s", resp.text[:200])
            return None
    except Exception as e:
        logger.warning("Cloudflare Workers AI request failed: %s", e)
        return None


class ImageService:
    """Image generation: Cloudflare Workers AI (SDXL) first, then Hugging Face if configured."""

    def __init__(self):
        self.cf_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.cf_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        self.use_cloudflare = bool(self.cf_account_id and self.cf_token)

        self.hf_token = os.getenv("HF_TOKEN")
        self.use_hf = bool(self.hf_token)
        self._client = None
        self._model = (os.getenv("EDUAI_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL).strip()

        if self.use_cloudflare:
            logger.info("Image generation: Cloudflare Workers AI (SDXL) enabled")
        if self.use_hf:
            self._init_client()
        if not self.use_cloudflare and not self.use_hf:
            logger.info("Image generation disabled (set CLOUDFLARE_ACCOUNT_ID+CLOUDFLARE_API_TOKEN or HF_TOKEN)")

    def _init_client(self) -> None:
        try:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=self.hf_token)
            logger.info("Hugging Face InferenceClient initialized for image generation (model=%s)", self._model)
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
        Tries Cloudflare Workers AI first (if configured), then Hugging Face.
        """
        if not self.use_cloudflare and not (self.use_hf and self._client):
            return None

        prompt = prompt_override or _build_educational_prompt(topic)
        logger.info(f"Generating image for topic '{topic}' with prompt: {prompt[:150]}...")

        import time
        start_time = time.time()

        # 1) Try Cloudflare Workers AI (SDXL) first
        if self.use_cloudflare:
            cf_result = _generate_via_cloudflare(prompt, NEGATIVE_PROMPT)
            if cf_result:
                raw, content_type = cf_result
                result, content_type = _ensure_under_size(raw, content_type)
                elapsed = time.time() - start_time
                logger.info(f"Generated image for topic '{topic}' via Cloudflare: {len(result)} bytes in {elapsed:.2f}s")
                return (result, content_type)
            logger.warning("Cloudflare image generation failed; trying Hugging Face if configured")

        # 2) Fall back to Hugging Face
        if not self.use_hf or not self._client:
            return None

        models_to_try = [self._model] + [m for m in IMAGE_MODEL_FALLBACKS if m != self._model]
        last_error = None
        for model in models_to_try:
            try:
                logger.info(f"Attempting image generation with model: {model}")
                try:
                    image = self._client.text_to_image(
                        prompt, model=model, negative_prompt=NEGATIVE_PROMPT
                    )
                except TypeError:
                    image = self._client.text_to_image(prompt, model=model)

                if image is None:
                    continue
                buf = io.BytesIO()
                if hasattr(image, "save"):
                    image.save(buf, format="PNG")
                    raw = buf.getvalue()
                else:
                    raw = image if isinstance(image, bytes) else bytes(image)
                result, content_type = _ensure_under_size(raw, "image/png")
                elapsed = time.time() - start_time
                logger.info(f"Generated image for topic '{topic}' using HF model '{model}': {len(result)} bytes in {elapsed:.2f}s")
                return (result, content_type)
            except Exception as e:
                last_error = e
                logger.warning(f"HF image generation failed with model '{model}': {e}")
                continue

        logger.error(f"Image generation failed for topic '{topic}'. Last error: {last_error}")
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
