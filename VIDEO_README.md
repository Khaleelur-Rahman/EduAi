# Video Generation

Educational short videos for `/video <topic>`: **Cerebras narration + Cloudflare AI images + edge-tts + ffmpeg**. Narration, audio, and subtitles follow the user’s **language** (English, Spanish, French, Malay, Chinese, Hindi).

## How It Works

1. User sends `/video <topic>` (e.g. `/video photosynthesis`). The user’s set language is used.
2. **Cerebras LLM** generates a narration script in that language and 4 unique image prompts (in English for the image model).
3. In parallel:
   - **Cloudflare Workers AI** generates images (SDXL primary, FLUX fallback). HuggingFace Inference API is a secondary fallback.
   - **edge-tts** synthesizes the narration to audio in the same language (voice chosen by language and user age).
4. Narration is split into sentences with proportional timing (language-aware for languages without spaces, e.g. Chinese). Sentences are grouped sequentially across images.
5. **ffmpeg** creates a static clip per sentence (image + subtitle overlay in that language) and concatenates with the audio.
6. Video is compressed if over 16 MB (Twilio limit) and sent via WhatsApp.

## Setup

### Required

- **Cerebras** (narration + image prompts): set `CEREBRAS_API_KEY` in `.env`.
- **Cloudflare Workers AI** (image generation, 10k free requests/day):
  ```
  CLOUDFLARE_ACCOUNT_ID=your_account_id
  CLOUDFLARE_API_TOKEN=your_api_token
  ```
  Get these from the [Cloudflare dashboard](https://dash.cloudflare.com/) under **AI > Workers AI**.
- **ffmpeg**: required for video assembly. Install with `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux).

### Optional

- **HuggingFace** (fallback image generation): set `HF_TOKEN` in `.env`. Free tier is limited (~1000 requests/month).

## Output

- **Resolution:** 1280x720 (720p)
- **Duration:** matches narration length (~20-30s typical)
- **Format:** MP4 (H.264 + AAC)
- **Subtitles:** burned in, synced per-sentence
- **Images:** 3-4 per video, each shown for a group of sentences

## Limitations

- Languages: same as bot (en, es, fr, ms, zh, hi). Unsupported language falls back to English.
- Twilio: video must be ≤16 MB (auto-compressed if over limit).
- AI-generated images occasionally contain visual artifacts; negative prompts are used to suppress unwanted text in images.
