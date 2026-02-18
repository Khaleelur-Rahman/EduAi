# Video Generation

Educational short videos for `/video <topic>` use **short-video-maker** only: Cerebras generates the narration script, then TTS + Pexels stock footage + Remotion produce the video.

## Flow

1. User sends `/video <topic>` (e.g. `/video cells`).
2. **Cerebras** generates a short narration (4–6 sentences, 2–3 facts + one example) via the same streaming API as `/lesson`.
3. **short-video-maker** (Docker) turns that script into a video: TTS voiceover + Pexels clips + Remotion.
4. Video is compressed if over 16 MB (Twilio limit) and sent via WhatsApp.

## Setup

1. **Cerebras** (for narration): set `CEREBRAS_API_KEY` in `.env`. Same as lessons.
2. **short-video-maker** (Docker):

   ```bash
   docker run -it --rm -p 3123:3123 \
     -e PEXELS_API_KEY=your_pexels_api_key \
     gyoridavid/short-video-maker:latest-tiny
   ```

   In `.env`:

   ```bash
   SHORT_VIDEO_MAKER_URL=http://localhost:3123
   ```

   Optional: `SHORT_VIDEO_MAKER_POLL_INTERVAL=10`, `SHORT_VIDEO_MAKER_TIMEOUT=300`.

3. **Compression (optional):** If videos exceed 16 MB, install `ffmpeg` so the app can re-encode them (e.g. `brew install ffmpeg`).

## Limitations

- **English only.**
- **Twilio:** Video must be ≤16 MB (compression runs when over limit if ffmpeg is available).
