# Audio Features

STT (Speech-to-Text) and TTS (Text-to-Speech) for voice-based interaction via WhatsApp.

## Architecture

### Components

1. **`app/audio.py`**: Core audio module
   - `STTService`: Speech-to-Text via local Whisper model
   - `TTSService`: Text-to-Speech via edge-tts (Microsoft Edge TTS, free)

2. **`app/main.py`**: Webhook handler
   - Handles Twilio media messages (audio input)
   - Sends audio/text responses

3. **`app/handlers.py`**: Message processing
   - `process_whatsapp_audio()`: Processes incoming voice notes

## Setup

### Dependencies

```bash
openai-whisper>=20231117    # Local Whisper for STT
edge-tts>=6.1.0             # Microsoft Edge TTS (free, requires internet)
```

No API keys required for audio services.

### Installation

```bash
pip install -r requirements.txt
```

## Usage

### Audio Input (Voice Notes)

1. User sends voice note via WhatsApp
2. Twilio webhook receives media message
3. Audio is transcribed to text using local Whisper
4. Transcribed text is processed through the message handler
5. Response is generated (text or audio)

### Audio Output (Spoken Lessons)

1. LLM generates lesson text
2. Text is transformed to be audio-friendly (`_text_to_audio_friendly`):
   - Markdown formatting stripped
   - Bullets/lists converted to flowing sentences
   - UI instructions replaced with spoken cues
   - Emojis removed
3. Text is chunked into segments (up to ~4 sentences / ~120 words each)
4. Each chunk is synthesized via edge-tts in parallel
5. Audio segments sent as WhatsApp voice notes

### Voice Selection

Voices are selected per-language and age group via edge-tts neural voices:
- **Age ≤ 8**: Friendly female voice (e.g. `en-US-AriaNeural`)
- **Age 9-12**: Clear female voice (e.g. `en-US-JennyNeural`)
- **Age > 12**: Male voice (e.g. `en-US-GuyNeural`)

Supported languages: English, Spanish, French, Malay, Chinese, Hindi.

## Services

### STT (Speech-to-Text)

- **Local Whisper** (base model): Free, no API key, works offline, ~1-3s per transcription

### TTS (Text-to-Speech)

- **edge-tts**: Free, high-quality neural voices, requires internet connection

## Troubleshooting

### Twilio "Media failed to download" (error 63019)

When using **ngrok free tier**, Twilio may get an HTML interstitial instead of your audio file.

**Fix:** Use Cloudflare Tunnel instead:
```bash
cloudflared tunnel --url http://localhost:8000
```
Set `BASE_URL` to the tunnel URL in `.env`.

## Limitations

- Audio clips are truncated to ~150 words max
- STT is English-only (Whisper `language="en"`)
- edge-tts requires internet connectivity
