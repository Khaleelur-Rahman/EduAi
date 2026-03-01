# Video Generation

Educational short videos for `/video <topic>`: **Cerebras narration + Cloudflare AI images + edge-tts + ffmpeg**.

## How It Works

1. User sends `/video <topic>` (e.g. `/video photosynthesis`).
2. **Cerebras LLM** generates a narration script and 4 unique image prompts for the topic.
3. In parallel:
   - **Cloudflare Workers AI** generates images (SDXL primary, FLUX fallback). HuggingFace Inference API is a secondary fallback.
   - **edge-tts** (Microsoft Edge TTS) synthesizes the narration to audio.
4. Narration is split into sentences with proportional timing. Sentences are grouped sequentially across images (one group per image, each image used exactly once).
5. **ffmpeg** creates a static clip per sentence (image + subtitle overlay) and concatenates them with the audio track.
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

- English only.
- Twilio: video must be ≤16 MB (auto-compressed if over limit).
- AI-generated images occasionally contain visual artifacts; negative prompts are used to suppress unwanted text in images.
