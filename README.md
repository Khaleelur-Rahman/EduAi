# EduAI — WhatsApp AI Tutor

EduAI is an AI-powered tutoring bot delivered over WhatsApp via Twilio. Students send text or voice messages to get personalised lessons, quizzes, audio explanations, and short educational videos on any topic.

## Features

- **Text lessons** — LLM-generated lessons with inline images
- **Audio lessons** — voice notes via local Whisper (STT) and Edge TTS
- **Short videos** — narrated slide videos generated on demand
- **Quizzes** — multiple-choice and true/false questions from lesson content
- **RAG** — science lessons grounded in real textbooks via ChromaDB
- **Multilingual** — English, Spanish, French, Malay, Chinese, Hindi
- **Progress dashboard** — web dashboard authenticated via WhatsApp code

## Architecture

```
Twilio WhatsApp  ──►  FastAPI (app/main.py)
                           │
                ┌──────────┴──────────┐
           handlers.py           background task
                │                     │
        ┌───────┼───────┐        REST API reply
       llm.py  rag.py  audio.py
                │
           ChromaDB
```

Heavy commands (lesson, next, quiz, video, audio) are processed in a background task and delivered via the Twilio REST API to avoid the 30-second webhook timeout.

## Quick Start

### Prerequisites

- Python 3.11+
- Twilio account with WhatsApp sandbox or approved number
- Cerebras API key

### Installation

```bash
git clone https://github.com/your-org/EduAI.git
cd EduAI
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in:

```env
CEREBRAS_API_KEY=your_key
TWILIO_ACCOUNT_SID=ACxxx
TWILIO_AUTH_TOKEN=xxx
TWILIO_PHONE_NUMBER=+1xxx
BASE_URL=https://your-public-url.example.com
DASHBOARD_SECRET=a-random-secret
DATABASE_URL=sqlite:///./whatsapp_tutor.db   # or postgres URL
```

Optional flags:

| Variable | Default | Description |
|---|---|---|
| `RENDER_FREE_TIER` | `0` | Defer LLM/RAG init and disable RAG to reduce memory |
| `DEFER_RAG_INIT` | `0` | Init RAG on first `/lesson` instead of at startup |
| `LESSON_USE_RAG` | `1` | Set to `0` to always use LLM without RAG |

### Running Locally

```bash
uvicorn app.main:app --reload --port 8000
```

Expose the server with a tunnel (Cloudflare Tunnel recommended — ngrok free tier causes Twilio media errors):

```bash
cloudflared tunnel --url http://localhost:8000
```

Set `BASE_URL` to the tunnel URL so Twilio can fetch media files.

### Deploying to Render

Use `render_start.py` as the Start Command:

```
python render_start.py
```

Set `RENDER_FREE_TIER=1` on the free tier to defer heavy initialisation.

## Bot Commands

| Command | Description |
|---|---|
| `/lesson <topic>` | Start a lesson on any topic |
| `/next` | Continue to the next part |
| `/quiz` | Take a quiz on the current lesson |
| `/audio <topic>` | Get an audio lesson |
| `/video <topic>` | Get a short educational video |
| `/language <code>` | Change language (e.g. `/language es`) |
| `/progress` | View completed lessons and quiz scores |
| `/help` | Show all commands |

Voice messages also work — say "teach me about cells" or "next".

## Project Layout

```
app/
  main.py            FastAPI app, routes, webhook handler
  handlers.py        Message processing logic
  llm.py             Cerebras LLM client and prompt engineering
  rag.py             ChromaDB retrieval and RAG prompts
  audio.py           Whisper STT and Edge TTS
  video.py           Video generation pipeline
  image.py           Lesson image generation
  quiz.py            Quiz creation and answer checking
  db.py              SQLAlchemy models and DB helpers
  utils.py           Formatting, translation, helpers
  language.py        Supported languages config
  dashboard_auth.py  Dashboard token and session auth
  templates/         Jinja2 HTML templates for dashboard

tests/
  audio/             STT, TTS, and audio flow tests
  video/             Video generation tests and sample files
  image/             Image generation tests
  llm/               LLM and Cerebras tests
  core/              RAG, quiz, multilingual, and flow tests

llm_evaluation/      LLM model evaluation scripts and results
chunk_evaluation/    RAG chunking experiment scripts and results
data/                Source PDFs for RAG ingestion
docs/                Feature documentation
scripts/             Utility scripts
```

## Documentation

Detailed feature docs live in `docs/`:

- [Audio Features](docs/AUDIO_FEATURES_README.md)
- [Video Generation](docs/VIDEO_README.md)
- [Image Generation](docs/IMAGE_GENERATION_README.md)
- [RAG Pipeline](docs/RAG_README.md)
- [Multilingual Support](docs/MULTILINGUAL_SUPPORT_SUMMARY.md)
- [LLM Evaluation](docs/EVALUATION_README.md)
