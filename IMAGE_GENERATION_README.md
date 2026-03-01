# Image Generation (Lesson Illustrations)

Educational diagrams are generated for each lesson topic and sent with the lesson (e.g. via WhatsApp). The service uses **Cloudflare Workers AI (SDXL)** first when configured, then falls back to **Hugging Face Inference API**.

## Overview

- **Purpose:** Generate a single, text-free educational illustration per lesson topic (e.g. photosynthesis, carbon cycle, transpiration).
- **Design:** Prompts ask for **no text in the image** (shapes, symbols, arrows only). The lesson title in the message provides the topic name.
- **Output:** Images are resized/compressed to stay under WhatsApp’s 5 MB limit (target 768×768, JPEG).

## Providers

### 1. Cloudflare Workers AI (recommended)

- **Model:** `@cf/stabilityai/stable-diffusion-xl-base-1.0` (SDXL).
- **Free tier:** ~10,000 “neurons” per day (roughly 20–30 SDXL images).
- **No credit card** required for testing.

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `CLOUDFLARE_ACCOUNT_ID` | Your Cloudflare account ID (Workers AI page in dashboard). |
| `CLOUDFLARE_API_TOKEN` | API token with Workers AI permissions. |

**Creating the API token:**

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com) → **Workers & Pages** → **AI** → **Workers AI**.
2. Copy your **Account ID**.
3. Click **Create a Workers AI API Token**, then **Create API Token**.
4. Copy the token once (it’s only shown once) and set `CLOUDFLARE_API_TOKEN`.

### 2. Hugging Face Inference API (fallback)

- Used when Cloudflare is not configured or the Cloudflare request fails.
- **Models tried in order:** `FLUX.1-schnell` → `FLUX.1-dev` → `stable-diffusion-xl-base-1.0` → `stable-diffusion-v1-5` (configurable via `EDUAI_IMAGE_MODEL`).

**Environment variable:**

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face API token (with Inference API access). |

### Priority

If both Cloudflare and Hugging Face are configured, **Cloudflare is used first**. On failure, the app tries Hugging Face.

## Configuration summary

- **Cloudflare only:** Set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`.
- **Hugging Face only:** Set `HF_TOKEN`.
- **Both:** Cloudflare is primary; HF is fallback.
- **Neither:** Image generation is disabled (lessons are sent without an image).

## How it works

1. **Prompt:** For a given topic (e.g. `transpiration`), the app builds a prompt that asks for a clean, text-free diagram and appends topic-specific guidance (e.g. “plants, sun, clouds, vapor, arrows from leaves”).
2. **Negative prompt:** Same for both providers: avoid text, words, labels, illegible/garbled text, cluttered composition.
3. **Request:** Cloudflare is called via REST (`POST` with `prompt` and `negative_prompt`). If used, Hugging Face is called via `InferenceClient.text_to_image()` with the same prompt and negative prompt.
4. **Response:** Raw image bytes are resized/compressed with PIL (thumbnail 768×768, JPEG) so the result is under 5 MB, then returned as `(bytes, content_type)`.

## Code entry points

- **Module:** `app/image.py`
- **Public API:** `generate_lesson_image(topic: str, language: str = "en") -> Optional[Tuple[bytes, str]]`
- **Service:** `ImageService.generate(topic, prompt_override=None)` — used by lesson handlers when attaching an image to a lesson.

## Supported topic keywords

Topic-specific prompt hints are defined for:  
`nitrogen`, `oxygen`, `carbon`, `cell`, `photosynthesis`, `transpiration`, `chlorophyll`, `atom`, `water`, `plant`, `dna`, `ecosystem`, `food chain`, `solar system`, `molecule`.  

Other topics use the generic “clean educational diagram” prompt only.

## Optional: Hugging Face model selection

To force a specific Hugging Face model (e.g. SDXL only):

```bash
export EDUAI_IMAGE_MODEL=stabilityai/stable-diffusion-xl-base-1.0
```

Fallbacks are still used if that model fails.
