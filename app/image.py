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
    """Build an educational image prompt from lesson topic.
    Focuses on visual accuracy, avoids text labels, and ensures scientific correctness.
    Uses negative prompts to prevent text generation.
    """
    # Clean topic name for better prompt
    topic_clean = topic.lower().strip()
    
    # Build a prompt that emphasizes visual accuracy and STRONGLY avoids text
    # Use explicit negative instructions multiple times to prevent text generation
    prompt = (
        f"Scientific diagram illustration of {topic_clean}, "
        "highly detailed, scientifically accurate, educational diagram, "
        "visual representation only, NO TEXT, NO WORDS, NO LETTERS, NO LABELS, "
        "completely text-free, pure visual diagram, "
        "clean white background, colorful, clear and easy to understand, "
        "professional scientific illustration style, "
        "crisp and clear, high quality, detailed"
    )
    
    # Add topic-specific guidance for common science topics
    topic_keywords = {
        "nitrogen": "nitrogen cycle diagram, atmospheric nitrogen N2, plants absorbing nitrogen, nitrogen-fixing bacteria, soil, arrows showing nitrogen flow, no text",
        "oxygen": "oxygen molecule O2, oxygen cycle, plants producing oxygen, lungs, respiration, oxygen in atmosphere, no text",
        "carbon": "carbon cycle, CO2 molecules, plants, animals, atmosphere, carbon dioxide, arrows showing carbon flow, no text",
        "cell": "cell structure diagram, organelles visible, nucleus, mitochondria, cell membrane, clear cell parts, no text labels",
        "photosynthesis": "plant diagram, sunlight rays, water H2O, carbon dioxide CO2, oxygen O2 bubbles, chloroplasts visible, no text",
        "atom": "atomic structure, central nucleus, electron orbitals, protons and neutrons, clear atomic model, no text",
        "water": "water molecule H2O structure, two hydrogen atoms, one oxygen atom, molecular bonds, no text",
        "plant": "plant anatomy diagram, roots, stem, leaves, flowers, clear plant parts, botanical illustration, no text",
        "animal": "animal anatomy, body structure, clear biological illustration, no text",
        "bacteria": "bacterial cell structure, simple prokaryotic cell, cell wall, DNA, no text",
        "virus": "viral structure, geometric capsid shape, genetic material inside, simple clear shape, no text",
        "dna": "DNA double helix structure, spiral ladder, nucleotide bases, genetic structure, no text",
        "ecosystem": "ecosystem diagram, plants, animals, environment, food chain arrows, no text",
        "food chain": "food chain illustration, producer plant, consumer animals, decomposer, arrows showing flow, no text",
        "solar system": "planets orbiting sun, clear planetary orbits, space diagram, no text",
        "molecule": "molecular structure, atoms connected by bonds, chemical compound diagram, no text",
    }
    
    # Add specific guidance if topic matches known keywords
    for keyword, guidance in topic_keywords.items():
        if keyword in topic_clean:
            prompt = f"{prompt}, {guidance}"
            break
    
    # Add strong negative reinforcement at the end
    prompt += ", absolutely no text, no words, no letters, no spelling, no labels, text-free diagram"
    
    return prompt


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
        logger.info(f"Generating image for topic '{topic}' with prompt: {prompt[:150]}...")
        
        # Use SDXL for better quality (more accurate, less text generation issues)
        models_to_try = [
            "stabilityai/stable-diffusion-xl-base-1.0",  # SDXL - better quality, more accurate
            "runwayml/stable-diffusion-v1-5",  # Fallback
        ]
        
        import time
        start_time = time.time()
        last_error = None
        
        for model in models_to_try:
            try:
                logger.info(f"Attempting image generation with model: {model}")
                
                # Try to use negative prompt if the API supports it
                # Some HF Inference API versions support negative prompts
                try:
                    # Attempt with negative prompt to prevent text generation
                    negative_prompt = "text, words, letters, labels, spelling, typography, writing, text labels, illegible text, misspelled words, blurry text"
                    image = self._client.text_to_image(
                        prompt, 
                        model=model,
                        negative_prompt=negative_prompt
                    )
                except TypeError:
                    # If negative_prompt not supported, use regular call
                    image = self._client.text_to_image(prompt, model=model)
                
                if image is None:
                    logger.warning(f"HF text_to_image returned None for model: {model}")
                    continue

                # InferenceClient returns PIL.Image
                buf = io.BytesIO()
                if hasattr(image, "save"):
                    image.save(buf, format="PNG")
                    raw = buf.getvalue()
                else:
                    raw = image if isinstance(image, bytes) else bytes(image)

                result, content_type = _ensure_under_size(raw, "image/png")
                elapsed = time.time() - start_time
                logger.info(f"Generated image for topic '{topic}' using model '{model}': {len(result)} bytes in {elapsed:.2f}s")
                return (result, content_type)
            except Exception as e:
                last_error = e
                elapsed = time.time() - start_time
                logger.warning(f"Image generation failed with model '{model}' after {elapsed:.2f}s: {e}")
                continue
        
        # All models failed
        elapsed = time.time() - start_time
        logger.error(f"Image generation failed for topic '{topic}' after {elapsed:.2f}s. Last error: {last_error}")
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
