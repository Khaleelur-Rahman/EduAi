# Audio Features Implementation

This document describes the STT (Speech-to-Text) and TTS (Text-to-Speech) features added to the EduAI project.

## Overview

The audio features enable:
1. **Audio Input (Voice Questions)**: Users can send voice notes via WhatsApp, which are transcribed to text and processed through the existing RAG pipeline.
2. **Audio Output (Spoken Lessons)**: LLM-generated lessons can be converted to audio and sent back via WhatsApp.

## Architecture

### Components

1. **`app/audio.py`**: Core audio processing module
   - `STTService`: Speech-to-Text service (OpenAI Whisper API with local Whisper fallback)
   - `TTSService`: Text-to-Speech service (OpenAI TTS API with Coqui TTS fallback)

2. **`app/main.py`**: Updated webhook handler
   - Handles Twilio media messages (audio input)
   - Processes audio through transcription pipeline
   - Sends audio/text responses

3. **`app/handlers.py`**: Message processing
   - `process_whatsapp_audio()`: Processes incoming audio messages
   - Integrates with existing message handler

## Setup

### Dependencies

The following packages are required (added to `requirements.txt`):

```bash
openai>=1.0.0              # For Whisper API and TTS API
openai-whisper>=20231117    # Local Whisper fallback (optional)
edge-tts>=6.1.0             # Microsoft Edge TTS fallback (free, Python 3.12 compatible)
```

**Note**: edge-tts is free and works with Python 3.12+, but requires an internet connection.

### Environment Variables

**Required for API-based services:**
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**Note**: If `OPENAI_API_KEY` is not set, the system will automatically fall back to local open-source models:
- Local Whisper for STT
- Coqui TTS for TTS

### Installation

```bash
pip install -r requirements.txt
```

## Usage

### Audio Input Flow

1. User sends voice note via WhatsApp
2. Twilio webhook receives media message with `MediaUrl0`
3. System fetches audio from Twilio URL
4. Audio is transcribed to text using STT service
5. Transcribed text is processed through existing message handler
6. Response is generated (text or audio)

### Audio Output Flow

1. LLM generates age-tailored lesson text (or handler returns WhatsApp-formatted response).
2. **LLM/RAG prompts for audio**: When the user sends a voice message, the handler uses `for_audio=True`. RAG and LLM lesson prompts then add an **AUDIO/TTS MODE** block instructing the model to write for listening: short sentences, flowing prose (no markdown headers or bullet lists), no "Type /next" instructions, minimal emojis, as if speaking to the student.
3. **Audio-friendly transform**: Before TTS, text is passed through `_text_to_audio_friendly()` so it reads naturally when spoken:
   - Headers like `*Lesson: Topic*` and `*Topic - Part N*` become short spoken phrases (e.g. “Lesson: Topic.”, “Part N. Topic.”)
   - Markdown bold/italic (`*text*`, `_text_`) is stripped to plain text
   - Bullets and numbered lists are turned into full sentences with periods
   - UI instructions (e.g. “_Type /next to continue_”) are replaced with a brief cue: “Say ‘Next’ to continue.”
   - Emojis are removed; extra newlines are collapsed so the model doesn’t over-pause
4. Text is converted to speech using TTS. To keep lessons detailed and student-friendly, the full paragraph is **chunked** into segments that each end at a sentence boundary (up to ~4 sentences or ~120 words per chunk). Each segment is synthesized separately.
5. Audio is sent back via WhatsApp: the first segment is sent in the webhook reply; any further segments are sent as follow-up messages via the Twilio REST API, so the user receives the full lesson as multiple voice notes that each make sense on their own.

The helper `text_to_audio_friendly(text)` in `app/audio` can be used to preview or test the audio-friendly version of any string.

### Voice Selection

The system automatically selects appropriate voices based on user age:
- **Age ≤ 8**: `nova` (softer, friendlier voice)
- **Age ≤ 12**: `alloy` (balanced, clear voice)
- **Age > 12**: `onyx` (more mature voice)

## Service Selection

The system uses a tiered approach:

### STT (Speech-to-Text)

1. **Primary**: OpenAI Whisper API
   - Fast, accurate, supports multiple languages
   - Requires `OPENAI_API_KEY`
   - Cost: $0.006 per minute

2. **Fallback**: Local Whisper model
   - Completely free, no API key needed
   - Slower but works offline
   - Requires `openai-whisper` package

### TTS (Text-to-Speech)

1. **Primary**: OpenAI TTS API
   - Fast, high quality, multiple voices
   - Requires `OPENAI_API_KEY`
   - Cost: $15 per 1M characters (tts-1) or $30 per 1M characters (tts-1-hd)

2. **Fallback**: edge-tts (Microsoft Edge TTS)
   - Completely free, no API key needed
   - Good quality neural voices
   - Requires internet connection
   - Requires `edge-tts` package
   - **Python 3.12 compatible**

## Testing

Three test files are provided:

1. **`test_stt.py`**: Tests STT functionality
   ```bash
   python test_stt.py
   ```

2. **`test_tts.py`**: Tests TTS functionality
   ```bash
   python test_tts.py
   ```

3. **`test_audio_integration.py`**: Tests full integration
   ```bash
   python test_audio_integration.py
   ```

### Test Audio File

For full testing, create a test audio file:
- `test_audio.ogg`, `test_audio.mp3`, or `test_audio.wav`

## Design Decisions

### Why OpenAI APIs?

- **Free tier available**: $5 free credits for new users
- **Fast**: API-based, no local model loading
- **High quality**: Production-ready accuracy
- **Multi-user support**: Handles concurrent requests

### Why Open Source Fallbacks?

- **Cost-free**: No API costs
- **Privacy**: Data stays local
- **Reliability**: Works even if API is down
- **Flexibility**: Can be customized

### Audio Clip Length

- **30-60 seconds**: Keeps bandwidth low
- **~150 words max**: Automatically truncated if longer
- **Age-appropriate**: Vocabulary adjusted for user age

## Current Limitations

1. **Audio Response Upload**: Currently, audio responses are generated but sent as text. Full audio upload to Twilio requires hosting the audio file at a publicly accessible URL (e.g., S3, Cloudinary).

2. **Media URL Hosting**: For production, implement:
   - Audio file storage (S3, etc.)
   - Temporary URL generation
   - Cleanup of old audio files

## Future Enhancements

1. **Audio Response Upload**: Implement proper audio file hosting and Twilio media upload
2. **Audio Caching**: Cache popular lessons as audio files
3. **Multiple Languages**: Support non-English audio input/output
4. **Voice Customization**: Allow users to select preferred voice
5. **Audio Quality Settings**: Adjust based on network conditions

## Troubleshooting

### STT Not Working

- Check `OPENAI_API_KEY` is set (for API) or `openai-whisper` is installed (for local)
- Verify audio file format is supported (OGG, MP3, WAV, etc.)
- Check audio file is not corrupted

### TTS Not Working

- Check `OPENAI_API_KEY` is set (for API) or `TTS` package is installed (for local)
- Verify text is not empty
- Check system has enough memory (for local models)

### Audio Not Sending

- Currently, audio responses are sent as text
- Full audio upload requires additional implementation (see Limitations)

## Cost Considerations

### Free Tier Usage

- **OpenAI**: $5 free credits (covers ~833 minutes of Whisper or ~333K characters of TTS)
- **Local Models**: Completely free after installation

### Production Costs

For 1000 users sending 1 voice message/day:
- **STT**: ~$0.006 per minute × 1 min × 1000 = $6/day
- **TTS**: ~$15 per 1M chars × 0.5M chars = $7.50/day
- **Total**: ~$13.50/day or ~$405/month

**Recommendation**: Use local models for development/testing, API for production scale.

## Security Notes

- Audio files are processed in memory and not permanently stored
- Temporary files are cleaned up after processing
- API keys should be stored securely in environment variables
- User audio data is not logged (only transcripts may be logged)

## Troubleshooting

### "Media failed to download" with ngrok (Twilio error 63019)

When using **ngrok free tier**, Twilio may fail to download audio from your `/audio/{id}` URLs. ngrok’s free plan can serve an HTML "Visit Site" interstitial for some requests; if Twilio gets that page instead of the audio file, it reports "media failed to download".

**Workarounds:**

1. **Use Cloudflare Tunnel (free, no interstitial)**  
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```  
   Set `BASE_URL` to the `*.trycloudflare.com` URL shown. Twilio can fetch media from this URL without the interstitial.

2. **Use ngrok paid** so the browser warning/interstitial is disabled.

3. **Deploy to a real host** (e.g. Railway, Render, Fly.io) and set `BASE_URL` to your app’s public URL.

**With cloudflared:** (1) Set `BASE_URL=https://YOUR-TUNNEL.trycloudflare.com` (no trailing slash) and **restart the app**. (2) In Twilio, set the webhook to `https://YOUR-TUNNEL.trycloudflare.com/whatsapp` so both webhook and media use the same URL. Quick tunnels get a new URL each run. Check app logs for "BASE_URL set" or "Media base URL" to confirm.

## Support

For issues or questions:
1. Check test files for examples
2. Review logs for error messages
3. Verify environment variables are set correctly
4. Ensure all dependencies are installed
