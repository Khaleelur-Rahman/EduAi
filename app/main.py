import os
import re
import json
import logging
import tempfile
import hmac
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Depends, HTTPException, Form, BackgroundTasks
from fastapi.responses import Response, FileResponse, RedirectResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient

from .db import get_db, create_tables
from .handlers import process_whatsapp_message, process_whatsapp_audio, process_whatsapp_message_request_audio, process_whatsapp_message_request_video
from .llm import initialize_llm
from .rag import initialize_rag
from .audio import initialize_audio_services

# Temporary in-memory audio storage for TTS files
# Format: {audio_id: {'bytes': bytes, 'content_type': str, 'created_at': datetime}}
_temp_audio_store: Dict[str, Dict[str, Any]] = {}

# Temporary image storage: file-based so Twilio can fetch media after process restarts (e.g. reload)
_TEMP_IMAGE_DIR: Path = Path(os.getenv("TEMP_IMAGE_DIR", tempfile.gettempdir())) / "eduai_images"
_TEMP_IMAGE_TTL_HOURS = 1

# Temporary in-memory video storage for generated videos (Twilio media_url)
_temp_video_store: Dict[str, Dict[str, Any]] = {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")


def _is_render_free_tier() -> bool:
    """True when running on Render free tier: no background task, no RAG (reduces memory)."""
    v = (os.getenv("RENDER_FREE_TIER") or "").strip().lower()
    return v in ("1", "true", "yes")


def _defer_rag_init() -> bool:
    """True when RAG should init on first /lesson instead of at startup (e.g. dev for faster startup)."""
    v = (os.getenv("DEFER_RAG_INIT") or "").strip().lower()
    return v in ("1", "true", "yes")


def _whatsapp_from(number: str) -> str:
    """Return 'whatsapp:+...' in E.164 for Twilio WhatsApp. Handles numbers with or without leading +."""
    if not number:
        return ""
    n = (number or "").strip().replace("whatsapp:", "").strip()
    if not n:
        return ""
    if not n.startswith("+"):
        n = "+" + n
    return f"whatsapp:{n}"

def _send_loading_message(phone_number: str, command_type: str, topic: str = None, language: str = "en") -> None:
    """Send an immediate loading message via REST API for better UX."""
    if not TWILIO_CLIENT or not TWILIO_PHONE_NUMBER:
        return
    
    try:
        from app.utils import get_loading_message
        from_twilio = _whatsapp_from(TWILIO_PHONE_NUMBER)
        to_user = _whatsapp_from(phone_number) or f"whatsapp:{phone_number}"
        
        loading_text = get_loading_message(command_type, topic, language)
        
        TWILIO_CLIENT.messages.create(
            from_=from_twilio,
            to=to_user,
            body=loading_text,
        )
        logger.info(f"Sent loading message ({language}): {loading_text}")
    except Exception as e:
        logger.warning(f"Failed to send loading message: {e}")

def _detect_command_and_send_loading(phone_number: str, message: str, language: str = "en") -> None:
    """Detect lesson/quiz/next/audio commands and send appropriate loading message."""
    if not message or not message.strip():
        return
    
    msg_lower = message.strip().lower()
    msg_stripped = message.strip()
    
    # Detect /language command
    if msg_lower.startswith("/language") or msg_lower.startswith("/lang"):
        # Don't send loading for language command
        return
    
    # Detect /next command FIRST (before /lesson to avoid confusion)
    if msg_lower == "/next" or msg_lower.startswith("/next "):
        _send_loading_message(phone_number, "next", None, language)
        return
    
    # Detect /audio next
    if msg_lower == "/audio next" or msg_lower.startswith("/audio next "):
        _send_loading_message(phone_number, "next", None, language)
        return
    
    # Detect /audio commands
    if msg_lower.startswith("/audio "):
        topic = msg_stripped[7:].strip()
        if topic and topic.lower() != "next":
            _send_loading_message(phone_number, "lesson", topic, language)
            return
    
    # Detect /lesson command
    if msg_lower.startswith("/lesson "):
        topic = msg_stripped[7:].strip()
        if topic:
            _send_loading_message(phone_number, "lesson", topic, language)
            return
    
    # Detect /quiz command
    if msg_lower.startswith("/quiz"):
        _send_loading_message(phone_number, "quiz", None, language)
        return
    
    # Detect /video command
    if msg_lower.startswith("/video"):
        topic = msg_stripped[6:].strip() if len(msg_stripped) > 6 else ""
        _send_loading_message(phone_number, "video", topic or None, language)
        return
    
    # Detect /progress and /review commands
    if msg_lower.startswith("/progress") or msg_lower.startswith("/review"):
        _send_loading_message(phone_number, "progress", None, language)
        return
    
    # Detect voice-friendly formats
    if msg_lower.startswith("teach me about "):
        topic = msg_stripped[len("teach me about "):].strip()
        if topic:
            _send_loading_message(phone_number, "lesson", topic, language)
            return
    
    if msg_lower.startswith("lesson "):
        topic = msg_stripped[len("lesson "):].strip()
        if topic and topic.lower() != "next":
            _send_loading_message(phone_number, "lesson", topic, language)
            return
    
    # Detect plain "next" (voice format)
    if msg_lower == "next" or (msg_lower.startswith("next ") and len(msg_lower.split()) <= 2):
        _send_loading_message(phone_number, "next")
        return
    
    if msg_lower.startswith("quiz"):
        _send_loading_message(phone_number, "quiz")
        return
    
    if msg_lower.startswith("progress") or msg_lower.startswith("review"):
        _send_loading_message(phone_number, "progress", None, language)
        return

if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    logger.warning("Twilio configuration not found. Please set environment variables.")
    TWILIO_CLIENT = None
else:
    TWILIO_CLIENT = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    logger.info("Twilio client initialized successfully")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting EduBot application...")
    base_url = os.getenv("BASE_URL")
    if base_url:
        logger.info("BASE_URL set: %s (media URLs will use this)", base_url.rstrip("/"))
    else:
        logger.warning(
            "BASE_URL not set. Set BASE_URL to your public URL (e.g. Cloudflare Tunnel https://xxx.trycloudflare.com) "
            "or Twilio will get 'media failed to download' when sending images/audio/video."
        )
    create_tables()
    logger.info("Database tables created/verified")
    if _is_render_free_tier():
        # Prod (Render free tier): defer init to first use so server binds quickly and fits 512MB; RAG is skipped entirely
        logger.info("LLM, RAG, and audio will initialize on first use (RENDER_FREE_TIER); RAG disabled")
    else:
        # Local: load LLM and audio on startup; RAG at startup unless DEFER_RAG_INIT (dev = init RAG on first /lesson)
        try:
            logger.info("Initializing LLM model...")
            initialize_llm()
            logger.info("LLM model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            logger.warning("Application will continue but lessons may use fallback content")
        if not _defer_rag_init():
            try:
                logger.info("Initializing RAG service...")
                initialize_rag()
                logger.info("RAG service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RAG: {str(e)}")
                logger.warning("Application will continue but science lessons may not be available")
        else:
            logger.info("RAG will initialize on first /lesson (DEFER_RAG_INIT)")
        try:
            logger.info("Initializing audio services (STT/TTS)...")
            initialize_audio_services()
            logger.info("Audio services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio services: {str(e)}")
            logger.warning("Application will continue but audio features may not be available")
    # Ensure temp image directory exists (file-based storage so Twilio can fetch media after reload)
    _TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Temp image dir: %s", _TEMP_IMAGE_DIR)
    yield
    # Cleanup: Clear temporary media stores on shutdown
    _temp_audio_store.clear()
    _temp_video_store.clear()
    logger.info("Shutting down EduBot application...")

app = FastAPI(
    title="EduBot",
    description="An AI-powered tutoring system via WhatsApp using Twilio",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request so we can see if Twilio webhook hits the app."""
    logger.info("Request: %s %s", request.method, request.url.path)
    response = await call_next(request)
    return response


# Dashboard: templates and session cookie
_templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
if os.path.isdir(_templates_dir):
    templates = Jinja2Templates(directory=_templates_dir)
else:
    templates = None  # templates not found (e.g. running from different cwd)
DASHBOARD_SESSION_COOKIE = "dashboard_session"
DASHBOARD_COOKIE_MAX_AGE = 7 * 24 * 3600  # 7 days


def _normalize_phone(phone: str) -> str:
    """Normalize to E.164-like (digits with leading +)."""
    if not phone:
        return ""
    p = (phone or "").strip().replace("whatsapp:", "").strip()
    if not p.startswith("+"):
        p = "+" + p
    return p


def _get_dashboard_user_from_request(request: Request) -> Dict[str, Any]:
    """Get current dashboard user from session cookie. Returns {} if not authenticated."""
    try:
        from .dashboard_auth import verify_session_value
        cookie = request.cookies.get(DASHBOARD_SESSION_COOKIE)
        if not cookie:
            return {}
        data = verify_session_value(cookie)
        return data if data else {}
    except Exception:
        return {}


def _compute_analytics(db: Session, user_id: int) -> Dict[str, Any]:
    """Compute progress analytics for dashboard charts and KPIs."""
    import json
    from collections import defaultdict
    from .db import Progress, QuizProgress, get_user_progress, get_user_quizzes

    progress_list = get_user_progress(db, user_id, limit=500)
    quizzes_list = get_user_quizzes(db, user_id, limit=500)

    total_lessons = len(progress_list)
    total_quizzes = len([q for q in quizzes_list if getattr(q, "completed", False)])

    # Per-topic: parts completed (max lesson_step per topic, total_parts = max total_steps for that topic)
    topic_parts = defaultdict(lambda: {"max_step": 0, "total_steps": 1})
    for p in progress_list:
        key = p.topic
        topic_parts[key]["max_step"] = max(topic_parts[key]["max_step"], p.lesson_step or 1)
        topic_parts[key]["total_steps"] = max(topic_parts[key]["total_steps"], p.total_steps or 1)
    lesson_parts_by_topic = [
        {"topic": t, "parts_completed": data["max_step"], "total_parts": data["total_steps"]}
        for t, data in sorted(topic_parts.items())
    ]
    unique_topics = len(topic_parts)  # Number of distinct topics the user has done at least one lesson part on

    quiz_scores_pct = []
    for q in quizzes_list:
        if not getattr(q, "completed", False):
            continue
        try:
            qs = json.loads(q.questions) if isinstance(q.questions, str) else q.questions
            total_q = len(qs) if isinstance(qs, list) else 3
        except (json.JSONDecodeError, TypeError):
            total_q = 3
        score = q.score if q.score is not None else 0
        if total_q > 0:
            quiz_scores_pct.append(100 * score / total_q)
    average_quiz_score_pct = sum(quiz_scores_pct) / len(quiz_scores_pct) if quiz_scores_pct else 0

    # Time series: by date (use created_at for lessons; for completed use updated_at)
    lessons_by_date = defaultdict(int)
    for p in progress_list:
        if getattr(p, "completed", False):
            d = (p.updated_at or p.created_at).strftime("%Y-%m-%d") if hasattr(p, "updated_at") else p.created_at.strftime("%Y-%m-%d")
        else:
            d = p.created_at.strftime("%Y-%m-%d")
        lessons_by_date[d] += 1
    quizzes_by_date = defaultdict(int)
    for q in quizzes_list:
        if getattr(q, "completed", False):
            d = (q.updated_at or q.created_at).strftime("%Y-%m-%d") if hasattr(q, "updated_at") else q.created_at.strftime("%Y-%m-%d")
            quizzes_by_date[d] += 1

    lessons_by_date_list = [{"date": k, "count": v} for k, v in sorted(lessons_by_date.items())]
    quizzes_by_date_list = [{"date": k, "count": v} for k, v in sorted(quizzes_by_date.items())]

    # By topic: lesson count and average quiz score
    lesson_count_by_topic = defaultdict(int)
    for p in progress_list:
        lesson_count_by_topic[p.topic] += 1
    topic_scores = defaultdict(list)
    for q in quizzes_list:
        if not getattr(q, "completed", False):
            continue
        try:
            qs = json.loads(q.questions) if isinstance(q.questions, str) else q.questions
            total_q = len(qs) if isinstance(qs, list) else 3
        except (json.JSONDecodeError, TypeError):
            total_q = 3
        score = q.score if q.score is not None else 0
        if total_q > 0:
            topic_scores[q.topic].append(100 * score / total_q)
    quiz_score_by_topic = [
        {"topic": t, "average_score_pct": sum(s) / len(s), "count": len(s)}
        for t, s in topic_scores.items()
    ]
    quiz_score_by_topic.sort(key=lambda x: x["average_score_pct"])

    return {
        "total_lessons": total_lessons,
        "unique_topics": unique_topics,
        "total_quizzes": total_quizzes,
        "average_quiz_score_pct": round(average_quiz_score_pct, 1),
        "lessons_by_date": lessons_by_date_list,
        "quizzes_by_date": quizzes_by_date_list,
        "lesson_parts_by_topic": lesson_parts_by_topic,
        "lesson_count_by_topic": [{"topic": t, "count": c} for t, c in lesson_count_by_topic.items()],
        "quiz_score_by_topic": quiz_score_by_topic,
    }


@app.get("/")
async def root():
    return {
        "message": "EduBot is running!",
        "status": "healthy",
        "endpoints": {
            "webhook": "/whatsapp",
            "ping": "/ping (keep-alive for cron)",
            "health": "/health",
            "audio": "/audio/{audio_id} (temporary TTS audio serving)",
            "image": "/image/{image_id} (temporary lesson image serving)",
            "video": "/video/{video_id} (temporary video serving)"
        }
    }


@app.get("/whatsapp")
@app.get("/whatsapp/")
async def whatsapp_webhook_get():
    """Allow checking that the tunnel reaches the app. Twilio must use POST."""
    return Response(
        content="EduBot webhook is live. Twilio should send POST requests here.",
        media_type="text/plain",
    )

@app.get("/audio/{audio_id}")
async def serve_audio(audio_id: str, request: Request):
    """
    Temporary endpoint to serve audio files for Twilio media messages.
    Audio files are stored in memory and expire after 1 hour.
    """
    if audio_id not in _temp_audio_store:
        raise HTTPException(status_code=404, detail="Audio file not found or expired")
    
    audio_data = _temp_audio_store[audio_id]
    
    # Check if audio has expired (1 hour TTL)
    if datetime.utcnow() - audio_data['created_at'] > timedelta(hours=1):
        # Clean up expired audio
        del _temp_audio_store[audio_id]
        raise HTTPException(status_code=404, detail="Audio file expired")
    
    content_type = audio_data.get('content_type', 'audio/mpeg')
    audio_bytes = audio_data['bytes']
    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Cache-Control": "no-cache"
        }
    )

def _image_paths(image_id: str) -> tuple[Path, Path, Path]:
    """Return (data path, content-type path, created-at path) for an image_id."""
    base = _TEMP_IMAGE_DIR / image_id
    return base.with_suffix(".dat"), base.with_suffix(".ct"), base.with_suffix(".ts")


@app.get("/image/{image_id}")
async def serve_image(image_id: str, request: Request):
    """
    Temporary endpoint to serve image files for Twilio media messages.
    Images are stored on disk so they remain available after process restarts (e.g. reload).
    """
    # Sanitize: only allow UUID-like ids (alphanumeric and hyphen)
    if not image_id.replace("-", "").isalnum() or len(image_id) > 64:
        raise HTTPException(status_code=404, detail="Image file not found or expired")
    dat_path, ct_path, ts_path = _image_paths(image_id)
    if not dat_path.exists() or not ct_path.exists() or not ts_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found or expired")
    try:
        created_ts = float(ts_path.read_text().strip())
    except (ValueError, OSError):
        raise HTTPException(status_code=404, detail="Image file not found or expired")
    if datetime.utcnow().timestamp() - created_ts > _TEMP_IMAGE_TTL_HOURS * 3600:
        for p in (dat_path, ct_path, ts_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        raise HTTPException(status_code=404, detail="Image file expired")
    content_type = ct_path.read_text().strip() or "image/jpeg"
    image_bytes = dat_path.read_bytes()
    return Response(
        content=image_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="lesson_{image_id}.jpg"',
            "Content-Length": str(len(image_bytes)),
            "Cache-Control": "no-cache",
        },
    )

@app.get("/video/{video_id}")
async def serve_video(video_id: str):
    """
    Temporary endpoint to serve video files for Twilio media messages.
    Videos are stored in memory and expire after 1 hour. Twilio requires
    Content-Type and Content-Length for media_url.
    """
    if video_id not in _temp_video_store:
        raise HTTPException(status_code=404, detail="Video file not found or expired")
    video_data = _temp_video_store[video_id]
    if datetime.utcnow() - video_data["created_at"] > timedelta(hours=1):
        del _temp_video_store[video_id]
        raise HTTPException(status_code=404, detail="Video file expired")
    content_type = video_data.get("content_type", "video/mp4")
    video_bytes = video_data["bytes"]
    return Response(
        content=video_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="lesson_{video_id}.mp4"',
            "Content-Length": str(len(video_bytes)),
            "Cache-Control": "no-cache",
        },
    )


def _get_base_url(request: Request) -> str:
    """Get the base URL of the server from the request.
    Supports ngrok, cloudflared, and direct access.
    Prefer BASE_URL so media URLs work when webhook is behind a different tunnel.
    Twilio must be able to GET this URL; use a public URL (e.g. Cloudflare Tunnel) or you get 'media failed to download'.
    """
    base_url = os.getenv("BASE_URL")
    if base_url:
        base_url = base_url.rstrip("/")
        logger.debug("Using BASE_URL for media: %s", base_url)
        return base_url
    # Fallback: construct from request (same host as webhook)
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    if not scheme:
        scheme = "https" if request.url.port == 443 else "http"
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "localhost:8000"
    base_url = f"{scheme}://{host}"
    if "localhost" in host or "127.0.0.1" in host:
        logger.warning(
            "Media base URL is %s — Twilio cannot reach this. Set BASE_URL to your public URL (e.g. Cloudflare Tunnel) to fix 'media failed to download'.",
            base_url,
        )
    else:
        logger.info("Using request host for media URL: %s", base_url)
    return base_url

def _store_temp_audio(audio_bytes: bytes, content_type: str) -> str:
    """Store audio bytes temporarily and return a unique ID."""
    audio_id = str(uuid.uuid4())
    _temp_audio_store[audio_id] = {
        'bytes': audio_bytes,
        'content_type': content_type,
        'created_at': datetime.utcnow()
    }
    
    # Clean up old audio files (older than 1 hour)
    current_time = datetime.utcnow()
    expired_ids = [
        aid for aid, data in _temp_audio_store.items()
        if current_time - data['created_at'] > timedelta(hours=1)
    ]
    for expired_id in expired_ids:
        del _temp_audio_store[expired_id]
    
    logger.info(f"Stored temporary audio file: {audio_id} ({len(audio_bytes)} bytes)")
    return audio_id

def _store_temp_image(image_bytes: bytes, content_type: str) -> str:
    """Store image bytes on disk temporarily and return a unique ID.
    File-based storage so Twilio can fetch the media URL after process restarts (e.g. reload).
    """
    image_id = str(uuid.uuid4())
    dat_path, ct_path, ts_path = _image_paths(image_id)
    try:
        dat_path.write_bytes(image_bytes)
        ct_path.write_text(content_type)
        ts_path.write_text(str(datetime.utcnow().timestamp()))
    except OSError as e:
        logger.error("Failed to write temp image %s: %s", image_id, e)
        raise
    # Clean up expired image files from disk (older than TTL)
    cutoff = datetime.utcnow().timestamp() - (_TEMP_IMAGE_TTL_HOURS * 3600)
    for p in _TEMP_IMAGE_DIR.iterdir():
        if p.suffix != ".ts":
            continue
        try:
            if p.stat().st_mtime < cutoff:
                base = p.stem
                for ext in (".dat", ".ct", ".ts"):
                    (_TEMP_IMAGE_DIR / (base + ext)).unlink(missing_ok=True)
        except OSError:
            pass
    logger.info("Stored temporary image file: %s (%s bytes)", image_id, len(image_bytes))
    return image_id


def _store_temp_video(video_bytes: bytes, content_type: str) -> str:
    """Store video bytes temporarily and return a unique ID."""
    video_id = str(uuid.uuid4())
    _temp_video_store[video_id] = {
        "bytes": video_bytes,
        "content_type": content_type,
        "created_at": datetime.utcnow(),
    }
    current_time = datetime.utcnow()
    expired_ids = [
        vid for vid, data in _temp_video_store.items()
        if current_time - data["created_at"] > timedelta(hours=1)
    ]
    for eid in expired_ids:
        del _temp_video_store[eid]
    logger.info(f"Stored temporary video file: {video_id} ({len(video_bytes)} bytes)")
    return video_id




def _send_response_via_rest(request_or_base_url, phone_number: str, response) -> None:
    """Send response via REST API (for background processing).
    request_or_base_url can be a Request object or a string base_url.
    """
    if not TWILIO_CLIENT or not TWILIO_PHONE_NUMBER:
        logger.warning("Twilio not configured, cannot send via REST")
        return
    
    from_twilio = _whatsapp_from(TWILIO_PHONE_NUMBER)
    to_user = _whatsapp_from(phone_number) or f"whatsapp:{phone_number}"
    
    # Extract base_url from request object or use string directly
    if hasattr(request_or_base_url, 'base_url'):
        base_url = request_or_base_url.base_url
    elif isinstance(request_or_base_url, str):
        base_url = request_or_base_url
    else:
        base_url = _get_base_url(request_or_base_url)
    base_url = (base_url or "").rstrip("/")
    # Twilio must be able to GET media URLs; localhost/127.0.0.1 cause "media failed to download"
    if base_url and ("localhost" in base_url or "127.0.0.1" in base_url):
        logger.warning(
            "Sending media with base_url=%s — Twilio cannot reach this. Set BASE_URL to your public URL (e.g. Cloudflare Tunnel) in .env",
            base_url,
        )

    if isinstance(response, dict):
        text = response.get("text", "")
        tts_failed = response.get("tts_failed", False)
        if tts_failed and text:
            # Send error text
            try:
                TWILIO_CLIENT.messages.create(
                    from_=from_twilio,
                    to=to_user,
                    body=text,
                )
                logger.info(f"Sent error text via REST ({len(text)} chars)")
            except Exception as e:
                logger.error(f"Failed to send error text via REST: {e}")
        elif response.get("audio_segments"):
            # Send audio segments
            segments = response["audio_segments"]
            try:
                audio_ids = []
                for (seg_bytes, seg_ct) in segments:
                    aid = _store_temp_audio(seg_bytes, seg_ct or "audio/mpeg")
                    audio_ids.append(aid)
                urls = [f"{base_url}/audio/{aid}" for aid in audio_ids]
                logger.info(f"Media base URL: {base_url} (first audio: {urls[0][:80]}...)")
                
                import time
                send_start = time.time()
                sent_count = 0
                for i in range(len(urls)):
                    try:
                        audio_start = time.time()
                        TWILIO_CLIENT.messages.create(
                            from_=from_twilio,
                            to=to_user,
                            media_url=[urls[i]],
                        )
                        sent_count += 1
                        logger.info(f"Audio segment {i + 1} sent in {time.time() - audio_start:.2f}s")
                    except Exception as rest_err:
                        logger.warning(f"Audio segment {i + 1} failed: {rest_err}")
                
                if sent_count:
                    total_send_time = time.time() - send_start
                    logger.info(f"Sent {sent_count} of {len(urls)} audio segment(s) in {total_send_time:.2f}s total")
            except Exception as e:
                logger.error(f"Error preparing audio response: {e}")
                if text:
                    try:
                        TWILIO_CLIENT.messages.create(from_=from_twilio, to=to_user, body=text)
                    except:
                        pass
        elif response.get("audio_bytes"):
            # Single audio segment
            try:
                audio_id = _store_temp_audio(
                    response["audio_bytes"],
                    response.get("audio_content_type", "audio/mpeg"),
                )
                audio_url = f"{base_url}/audio/{audio_id}"
                TWILIO_CLIENT.messages.create(
                    from_=from_twilio,
                    to=to_user,
                    media_url=[audio_url],
                )
                logger.info(f"Sent single audio via REST: {audio_url}")
            except Exception as e:
                logger.error(f"Error sending audio via REST: {e}")
                if text:
                    try:
                        TWILIO_CLIENT.messages.create(from_=from_twilio, to=to_user, body=text)
                    except:
                        pass
        elif response.get("image_bytes"):
            # Image (e.g. from lesson or /next command)
            try:
                image_id = _store_temp_image(
                    response["image_bytes"],
                    response.get("image_content_type", "image/jpeg"),
                )
                image_url = f"{base_url}/image/{image_id}"
                msg_params = {
                    "from_": from_twilio,
                    "to": to_user,
                    "media_url": [image_url],
                }
                if text:
                    msg_params["body"] = text
                TWILIO_CLIENT.messages.create(**msg_params)
                logger.info(f"Sent image via REST: {image_url}")
            except Exception as e:
                logger.error(f"Error sending image via REST: {e}")
        elif response.get("video_url"):
            try:
                video_url = response["video_url"]
                body = text.strip() if text else None
                if body:
                    TWILIO_CLIENT.messages.create(
                        from_=from_twilio,
                        to=to_user,
                        body=body,
                        media_url=[video_url],
                    )
                else:
                    TWILIO_CLIENT.messages.create(
                        from_=from_twilio,
                        to=to_user,
                        media_url=[video_url],
                    )
                logger.info(f"Sent video via REST: {video_url[:80]}...")
            except Exception as e:
                logger.error(f"Error sending video via REST: {e}")
                if text:
                    try:
                        TWILIO_CLIENT.messages.create(from_=from_twilio, to=to_user, body=text)
                    except:
                        pass
        elif response.get("video_bytes"):
            try:
                video_id = _store_temp_video(
                    response["video_bytes"],
                    response.get("video_content_type", "video/mp4"),
                )
                video_url = f"{base_url}/video/{video_id}"
                body = text.strip() if text else None
                if body:
                    TWILIO_CLIENT.messages.create(
                        from_=from_twilio,
                        to=to_user,
                        body=body,
                        media_url=[video_url],
                    )
                else:
                    TWILIO_CLIENT.messages.create(
                        from_=from_twilio,
                        to=to_user,
                        media_url=[video_url],
                    )
                logger.info(f"Sent video via REST: {video_url}")
            except Exception as e:
                logger.error(f"Error sending video via REST: {e}")
                if text:
                    try:
                        TWILIO_CLIENT.messages.create(from_=from_twilio, to=to_user, body=text)
                    except:
                        pass
        else:
            # Plain text
            if text:
                try:
                    TWILIO_CLIENT.messages.create(from_=from_twilio, to=to_user, body=text)
                    logger.info(f"Sent text response via REST ({len(text)} chars)")
                except Exception as e:
                    logger.error(f"Failed to send text via REST: {e}")
    else:
        # String response
        try:
            TWILIO_CLIENT.messages.create(from_=from_twilio, to=to_user, body=str(response))
            logger.info(f"Sent text response via REST ({len(str(response))} chars)")
        except Exception as e:
            logger.error(f"Failed to send text via REST: {e}")

def _apply_audio_or_fallback_response(request: Request, phone_number: str, response: dict, twiml_response) -> None:
    """Apply dict response (from process_whatsapp_audio, process_whatsapp_message_request_audio, or process_whatsapp_message) to twiml_response."""
    text = response.get("text", "")
    tts_failed = response.get("tts_failed", False)
    if tts_failed and text:
        if text.strip().lower().startswith("sorry,") or "trouble" in text.lower():
            twiml_response.message(text)
        else:
            twiml_response.message("Audio couldn't be generated. Here's your lesson:\n\n" + text)
        logger.info(f"TTS fallback: sent text backup ({len(text)} chars)")
    elif response.get("image_bytes"):
        # Image (e.g. from /image or /next command)
        try:
            image_id = _store_temp_image(
                response["image_bytes"],
                response.get("image_content_type", "image/jpeg"),
            )
            base_url = _get_base_url(request)
            image_url = f"{base_url}/image/{image_id}"
            logger.info(f"Media base URL: {base_url}")
            msg = twiml_response.message()
            msg.media(image_url)
            if text:
                msg.body(text)
            logger.info(f"Prepared image response: {image_url}")
        except Exception as e:
            logger.error(f"Error preparing image response: {e}")
            if text:
                twiml_response.message(text)
    elif response.get("audio_segments"):
        segments = response["audio_segments"]
        try:
            base_url = _get_base_url(request)
            audio_ids = []
            for (seg_bytes, seg_ct) in segments:
                aid = _store_temp_audio(seg_bytes, seg_ct or "audio/mpeg")
                audio_ids.append(aid)
            urls = [f"{base_url}/audio/{aid}" for aid in audio_ids]
            logger.info(f"Media base URL: {base_url} (first audio: {urls[0][:80]}...)")
            # Send all segments via REST in order (0, 1, 2, ...) so the intro (segment 0)
            # is delivered first. Using TwiML for segment 0 and REST for the rest can cause
            # the webhook reply to be delivered last, making the intro appear as the latest message.
            if TWILIO_CLIENT and TWILIO_PHONE_NUMBER and urls:
                import time
                send_start = time.time()
                from_twilio = _whatsapp_from(TWILIO_PHONE_NUMBER)
                to_user = _whatsapp_from(phone_number) or f"whatsapp:{phone_number}"
                sent_count = 0
                for i in range(len(urls)):
                    try:
                        audio_start = time.time()
                        TWILIO_CLIENT.messages.create(
                            from_=from_twilio,
                            to=to_user,
                            media_url=[urls[i]],
                        )
                        sent_count += 1
                        logger.info(f"Audio segment {i + 1} sent in {time.time() - audio_start:.2f}s")
                    except Exception as rest_err:
                        logger.warning(
                            "Audio segment %s failed (Twilio 21212): %s", i + 1, rest_err
                        )
                if sent_count:
                    total_send_time = time.time() - send_start
                    logger.info(f"Sent {sent_count} of {len(urls)} audio segment(s) in {total_send_time:.2f}s total")
                # Return empty TwiML so Twilio does not send a second reply; user gets only REST messages in order.
                return
            # Fallback if Twilio not configured: use TwiML for first segment only
            msg = twiml_response.message()
            msg.media(urls[0])
            if len(segments) > 1:
                msg.body(f"Here's your lesson in {len(segments)} parts.")
            logger.info(f"Prepared chunked audio response: {len(urls)} URL(s) (TwiML fallback)")
        except Exception as e:
            logger.error(f"Error preparing chunked audio response: {e}")
            if text:
                twiml_response.message(text)
    elif response.get("audio_bytes"):
        try:
            audio_id = _store_temp_audio(
                response["audio_bytes"],
                response.get("audio_content_type", "audio/mpeg"),
            )
            base_url = _get_base_url(request)
            audio_url = f"{base_url}/audio/{audio_id}"
            logger.info(f"Media base URL: {base_url}")
            msg = twiml_response.message()
            msg.media(audio_url)
            logger.info(f"Prepared audio response: {audio_url}")
        except Exception as e:
            logger.error(f"Error preparing audio response: {e}")
            if text:
                twiml_response.message(text)
    elif response.get("video_url"):
        try:
            msg = twiml_response.message()
            msg.media(response["video_url"])
            if text:
                msg.body(text)
            logger.info("Prepared video response (video_url)")
        except Exception as e:
            logger.error(f"Error preparing video response: {e}")
            if text:
                twiml_response.message(text)
    elif response.get("video_bytes"):
        try:
            video_id = _store_temp_video(
                response["video_bytes"],
                response.get("video_content_type", "video/mp4"),
            )
            base_url = _get_base_url(request)
            video_url = f"{base_url}/video/{video_id}"
            msg = twiml_response.message()
            msg.media(video_url)
            if text:
                msg.body(text)
            logger.info(f"Prepared video response: {video_url}")
        except Exception as e:
            logger.error(f"Error preparing video response: {e}")
            if text:
                twiml_response.message(text)
    else:
        if text:
            twiml_response.message(text)

@app.get("/ping")
async def ping():
    """Lightweight keep-alive endpoint. No DB/LLM/RAG. Use with external cron every ~15 min to prevent spin-down (e.g. Render free tier)."""
    return {"status": "ok", "ping": "pong"}


@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "database": "connected",
        "llm": "unknown",
        "rag": "unknown",
        "twilio": "unknown"
    }
    try:
        from .db import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        from .llm import llm_service
        if llm_service._initialized:
            health_status["llm"] = "initialized"
        else:
            health_status["llm"] = "not_initialized"
    except Exception as e:
        health_status["llm"] = f"error: {str(e)}"
    
    try:
        from .rag import rag_service
        if rag_service._initialized:
            health_status["rag"] = "initialized"
        else:
            health_status["rag"] = "not_initialized"
    except Exception as e:
        health_status["rag"] = f"error: {str(e)}"
    
    if TWILIO_CLIENT:
        health_status["twilio"] = "configured"
    else:
        health_status["twilio"] = "not_configured"
        health_status["status"] = "degraded"
    
    return health_status

def _process_message_in_background(
    phone_number: str,
    body_text: str,
    base_url: str,
) -> None:
    """Process message and send response via REST API in background.
    Uses its own DB session; do not pass the request-scoped session (it is closed after the response).
    """
    from .db import SessionLocal
    db = SessionLocal()
    try:
        msg_lower = body_text.strip().lower()
        if msg_lower.startswith("/audio"):
            response = process_whatsapp_message_request_audio(db, phone_number, body_text)
        elif msg_lower.startswith("/video"):
            response = process_whatsapp_message_request_video(db, phone_number, body_text)
        else:
            # process_whatsapp_message now returns dict with image_bytes for text lessons
            response = process_whatsapp_message(db, phone_number, body_text, for_audio=False)
        _send_response_via_rest(base_url, phone_number, response)
    except Exception as e:
        logger.error(f"Error in background message processing: {e}")
    finally:
        db.close()

@app.post("/whatsapp")
@app.post("/whatsapp/")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(None),
    From: str = Form(None),
    To: str = Form(None),
    NumMedia: str = Form("0"),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    logger.info("WhatsApp webhook entered: From=%s, Body=%s", From, (Body or "")[:80])
    try:
        if not From:
            logger.error("Twilio webhook missing From parameter")
            raise HTTPException(status_code=400, detail="Missing From")
        phone_number = From.replace('whatsapp:', '').strip()
        if not phone_number:
            logger.error("Invalid phone number received: %s", From)
            raise HTTPException(status_code=400, detail="Invalid phone number")
        
        num_media = int(NumMedia) if NumMedia else 0
        
        # Handle audio media messages (voice notes)
        if num_media > 0 and MediaUrl0 and MediaContentType0:
            content_type = MediaContentType0.lower()
            if content_type.startswith('audio/'):
                logger.info(f"Received audio message from {From} (type: {content_type})")
                
                response = await process_whatsapp_audio(
                    db, phone_number, MediaUrl0, content_type, return_audio=True,
                    twilio_account_sid=TWILIO_ACCOUNT_SID,
                    twilio_auth_token=TWILIO_AUTH_TOKEN,
                    twilio_client=TWILIO_CLIENT
                )
                
                if response:
                    twiml_response = MessagingResponse()
                    _apply_audio_or_fallback_response(request, phone_number, response, twiml_response)
                    return Response(
                        content=str(twiml_response),
                        media_type="application/xml"
                    )
                else:
                    twiml_response = MessagingResponse()
                    twiml_response.message("Sorry, I couldn't process your audio message. Please try sending a text message! 🎤")
                    return Response(
                        content=str(twiml_response),
                        media_type="application/xml"
                    )
        
        body_text = Body or ""
        logger.info(f"Received WhatsApp message from {From}: {body_text[:100]}...")
        
        # Get user to check language preference
        from .db import get_user_by_phone, create_user
        user = get_user_by_phone(db, phone_number)
        if not user:
            user = create_user(db, phone_number)
        user_language = user.language if user else "en"
        
        # Handle /language command immediately
        msg_lower = body_text.strip().lower()
        if msg_lower.startswith("/language") or msg_lower.startswith("/lang"):
            from .language import validate_language_code, SUPPORTED_LANGUAGES, get_language_name
            from .db import update_user
            
            parts = body_text.strip().split(None, 1)
            if len(parts) > 1:
                lang_code = validate_language_code(parts[1])
                if lang_code:
                    update_user(db, user, language=lang_code)
                    lang_name = get_language_name(lang_code, native=True)
                    response_msg = f"✅ Language changed to {lang_name} ({lang_code.upper()})"
                else:
                    supported = ", ".join([f"{code} ({info['native']})" for code, info in SUPPORTED_LANGUAGES.items()])
                    response_msg = f"❌ Invalid language. Supported: {supported}\n\nExample: /language es"
            else:
                current_lang = get_language_name(user_language, native=True)
                supported = "\n".join([f"• {code} - {info['native']}" for code, info in SUPPORTED_LANGUAGES.items()])
                response_msg = f"🌐 Current language: {current_lang} ({user_language.upper()})\n\nSupported languages:\n{supported}\n\nChange language: /language <code>\nExample: /language es"
            
            twiml_response = MessagingResponse()
            twiml_response.message(response_msg)
            return Response(content=str(twiml_response), media_type="application/xml")
        
        # Check if this is a command that needs loading message + background processing
        is_command = (
            msg_lower.startswith("/audio") or
            msg_lower.startswith("/video") or
            msg_lower.startswith("/lesson") or
            msg_lower.startswith("/next") or
            msg_lower.startswith("/quiz") or
            msg_lower.startswith("/progress") or
            msg_lower.startswith("/review") or
            msg_lower.startswith("teach me about") or
            msg_lower.startswith("lesson ") or
            (msg_lower.strip() == "next" or (msg_lower.startswith("next ") and len(msg_lower.split()) <= 2)) or
            msg_lower.startswith("quiz") or
            msg_lower.startswith("progress") or
            msg_lower.startswith("review")
        )
        
        if is_command:
            msg_lower = body_text.strip().lower()
            # Heavy commands (video, audio, lesson, next, quiz) can take minutes — must use background
            # task so the webhook returns before Render/Twilio timeout (~30s); result sent via REST.
            is_heavy_command = (
                msg_lower.startswith("/video") or
                msg_lower.startswith("/audio") or
                msg_lower.startswith("/lesson") or
                msg_lower.startswith("/next") or
                msg_lower.startswith("/quiz") or
                msg_lower.startswith("teach me about") or
                msg_lower.startswith("lesson ") or
                (msg_lower.strip() == "next" or (msg_lower.startswith("next ") and len(msg_lower.split()) <= 2)) or
                msg_lower.startswith("quiz")
            )
            if _is_render_free_tier() and not is_heavy_command:
                # Fast commands only (progress, review, help, language): process in-request
                _detect_command_and_send_loading(phone_number, body_text, user_language)
                response = process_whatsapp_message(db, phone_number, body_text)
                twiml_response = MessagingResponse()
                if isinstance(response, dict):
                    _apply_audio_or_fallback_response(request, phone_number, response, twiml_response)
                else:
                    twiml_response.message(response)
                logger.info(f"Sending response to {phone_number} (render free tier, in-request)")
                return Response(content=str(twiml_response), media_type="application/xml")
            # Heavy commands (or local/paid): loading message + background task, respond via REST
            _detect_command_and_send_loading(phone_number, body_text, user_language)
            base_url = _get_base_url(request)
            background_tasks.add_task(_process_message_in_background, phone_number, body_text, base_url)
            return Response(content=str(MessagingResponse()), media_type="application/xml")
        
        if body_text.strip().lower().startswith("join ") and TWILIO_CLIENT and TWILIO_PHONE_NUMBER:
            logger.info("Detected sandbox join message. Sending proactive welcome.")
            try:
                response = process_whatsapp_message(db, phone_number, body_text, for_audio=False)
                # Handle dict response (may contain image_bytes)
                if isinstance(response, dict):
                    text = response.get("text", "")
                    if response.get("image_bytes"):
                        # Send image with text via REST
                        _send_response_via_rest(_get_base_url(request), phone_number, response)
                    elif text:
                        TWILIO_CLIENT.messages.create(
                            body=text,
                            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
                            to=f"whatsapp:{phone_number}"
                        )
                else:
                    # String response (fallback)
                    TWILIO_CLIENT.messages.create(
                        body=str(response),
                        from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
                        to=f"whatsapp:{phone_number}"
                    )
                return Response(content=str(MessagingResponse()), media_type="application/xml")
            except Exception as send_err:
                logger.error(f"Failed to send proactive welcome: {str(send_err)}")
                pass

        # Non-command messages: process normally and return TwiML
        if body_text.strip().lower().startswith("/audio"):
            response = process_whatsapp_message_request_audio(db, phone_number, body_text)
        else:
            # process_whatsapp_message now returns dict with image_bytes for text lessons
            response = process_whatsapp_message(db, phone_number, body_text, for_audio=False)

        twiml_response = MessagingResponse()
        if isinstance(response, dict):
            _apply_audio_or_fallback_response(request, phone_number, response, twiml_response)
        else:
            twiml_response.message(response)

        logger.info(f"Sending response to {phone_number}")
        return Response(
            content=str(twiml_response),
            media_type="application/xml"
        )
    
    except Exception as e:
        logger.error(f"Error processing WhatsApp webhook: {str(e)}")
        twiml_response = MessagingResponse()
        twiml_response.message(
            "Sorry, I'm experiencing technical difficulties. Please try again in a moment! 🔧"
        )
        return Response(
            content=str(twiml_response),
            media_type="application/xml"
        )

@app.get("/users")
async def get_users(db: Session = Depends(get_db)):
    from .db import User
    users = db.query(User).all()
    return {
        "total_users": len(users),
        "users": [
            {
                "id": user.id,
                "phone_number": user.phone_number[-4:],
                "name": user.name,
                "age": user.age,
                "country": user.country,
                "is_onboarded": user.is_onboarded,
                "created_at": user.created_at
            } for user in users
        ]
    }

@app.get("/users/{phone_number}/progress")
async def get_user_progress(phone_number: str, db: Session = Depends(get_db)):
    from .db import get_user_by_phone, get_user_progress, get_user_quizzes
    
    user = get_user_by_phone(db, phone_number)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    progress = get_user_progress(db, user.id)
    quizzes = get_user_quizzes(db, user.id)
    
    return {
        "user": {
            "name": user.name,
            "age": user.age,
            "country": user.country
        },
        "progress": [
            {
                "id": p.id,
                "topic": p.topic,
                "lesson_step": p.lesson_step,
                "total_steps": p.total_steps,
                "completed": p.completed,
                "created_at": p.created_at
            } for p in progress
        ],
        "quizzes": [
            {
                "id": q.id,
                "topic": q.topic,
                "lesson_step": q.lesson_step,
                "score": q.score,
                "completed": q.completed,
                "created_at": q.created_at
            } for q in quizzes
        ]
    }


# ---------- Dashboard (session + PIN auth) ----------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_landing_or_redirect(request: Request, token: str = None, db: Session = Depends(get_db)):
    """If token in query: verify, set session cookie, redirect to view. Else if session valid: redirect. Else: show landing (phone + request code)."""
    from .dashboard_auth import verify_dashboard_token, verify_session_value, create_session_value
    from .db import get_user_by_phone

    if token:
        data = verify_dashboard_token(token)
        if data:
            user = get_user_by_phone(db, data["phone"])
            if user:
                session_val = create_session_value(user.id, user.phone_number)
                response = RedirectResponse(url="/dashboard/view", status_code=302)
                response.set_cookie(
                    key=DASHBOARD_SESSION_COOKIE,
                    value=session_val,
                    max_age=DASHBOARD_COOKIE_MAX_AGE,
                    httponly=True,
                    samesite="lax",
                )
                return response
        if templates:
            return templates.TemplateResponse(
                request=request,
                name="dashboard_landing.html",
                context={"error": "Link expired. Enter your number to receive a new code.", "phone": ""},
            )

    session_user = _get_dashboard_user_from_request(request)
    if session_user:
        return RedirectResponse(url="/dashboard/view", status_code=302)

    if not templates:
        raise HTTPException(status_code=503, detail="Dashboard templates not available")
    return templates.TemplateResponse(
        request=request,
        name="dashboard_landing.html",
        context={"error": None, "phone": ""},
    )


@app.get("/dashboard/view", response_class=HTMLResponse)
async def dashboard_view(request: Request, db: Session = Depends(get_db)):
    """Show dashboard (KPIs, charts, recent activity). Requires valid session."""
    session_user = _get_dashboard_user_from_request(request)
    if not session_user:
        return RedirectResponse(url="/dashboard", status_code=302)

    from .db import get_user_by_phone, get_user_progress, get_user_quizzes

    user = get_user_by_phone(db, session_user["phone"])
    if not user:
        return RedirectResponse(url="/dashboard", status_code=302)

    progress = get_user_progress(db, user.id, limit=15)
    quizzes_raw = get_user_quizzes(db, user.id, limit=15)
    # Build quiz list with total_questions for score display (e.g. 1/3)
    quizzes = []
    for q in quizzes_raw:
        total_q = 3
        try:
            qs = json.loads(q.questions) if isinstance(q.questions, str) else q.questions
            if isinstance(qs, list):
                total_q = len(qs)
        except (TypeError, ValueError):
            pass
        quizzes.append({
            "id": q.id,
            "topic": q.topic,
            "lesson_step": q.lesson_step,
            "score": q.score if q.score is not None else 0,
            "total_questions": total_q,
            "completed": bool(q.completed),
            "created_at": q.created_at,
        })
    analytics = _compute_analytics(db, user.id)
    analytics_json = json.dumps(analytics)

    if not templates:
        raise HTTPException(status_code=503, detail="Dashboard templates not available")
    return templates.TemplateResponse(
        request=request,
        name="dashboard_view.html",
        context={
            "user_name": user.name or "Learner",
            "progress": progress,
            "quizzes": quizzes,
            "analytics": analytics,
            "analytics_json": analytics_json,
        },
    )


@app.post("/dashboard/progress/{progress_id}/hide")
async def dashboard_hide_progress(progress_id: int, request: Request, db: Session = Depends(get_db)):
    """Hide a progress entry from the dashboard (user no longer wants to see it)."""
    session_user = _get_dashboard_user_from_request(request)
    if not session_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    from .db import get_user_by_phone, set_progress_hidden
    user = get_user_by_phone(db, session_user["phone"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        pid = int(progress_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid progress id")
    if set_progress_hidden(db, pid, user.id):
        return RedirectResponse(url="/dashboard/view", status_code=303)
    raise HTTPException(status_code=404, detail="Progress not found or not yours")


@app.post("/dashboard/request-code")
async def dashboard_request_code(request: Request, db: Session = Depends(get_db)):
    """Send a 6-digit code to the user's WhatsApp. Body: phone (form or JSON)."""
    import secrets
    import hashlib
    from datetime import datetime, timedelta
    from .db import get_user_by_phone, create_dashboard_code

    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    phone = (body.get("phone") or (await request.form()).get("phone") or "").strip()
    phone = _normalize_phone(phone)
    if not phone or len(phone) < 10:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    user = get_user_by_phone(db, phone)
    if not user:
        raise HTTPException(status_code=404, detail="User not found. Use EduBot on WhatsApp first.")

    code = "".join(secrets.choice("0123456789") for _ in range(6))
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    expires_at = datetime.utcnow() + timedelta(minutes=10)
    create_dashboard_code(db, phone, code_hash, expires_at)

    if TWILIO_CLIENT and TWILIO_PHONE_NUMBER:
        to_wa = _whatsapp_from(phone)
        from_wa = _whatsapp_from(TWILIO_PHONE_NUMBER)
        try:
            TWILIO_CLIENT.messages.create(
                from_=from_wa,
                to=to_wa,
                body=f"Your EduBot dashboard code is: {code}",
            )
        except Exception as e:
            logger.warning("Failed to send dashboard code via WhatsApp: %s", e)
            raise HTTPException(status_code=503, detail="Could not send code. Try again later.")

    return {"status": "ok", "message": "Code sent to your WhatsApp"}


@app.post("/dashboard/verify-code")
async def dashboard_verify_code(request: Request, db: Session = Depends(get_db)):
    """Verify 6-digit code and set session cookie."""
    import hashlib
    from .dashboard_auth import verify_session_value, create_session_value
    from .db import get_user_by_phone, get_dashboard_code, delete_dashboard_code

    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    form = await request.form() if not body else {}
    phone = (body.get("phone") or form.get("phone") or "").strip()
    phone = _normalize_phone(phone)
    code = (body.get("code") or form.get("code") or "").strip()

    if not phone or not code or len(code) != 6 or not code.isdigit():
        raise HTTPException(status_code=400, detail="Invalid phone or code")

    row = get_dashboard_code(db, phone)
    if not row:
        raise HTTPException(status_code=400, detail="Code expired or not found. Request a new code.")

    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(row.code_hash, code_hash):
        raise HTTPException(status_code=400, detail="Invalid code")

    delete_dashboard_code(db, phone)
    user = get_user_by_phone(db, phone)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    session_val = create_session_value(user.id, user.phone_number)
    response = JSONResponse(content={"status": "ok", "redirect": "/dashboard/view"})
    response.set_cookie(
        key=DASHBOARD_SESSION_COOKIE,
        value=session_val,
        max_age=DASHBOARD_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return response


@app.get("/me/progress")
async def me_progress(request: Request, db: Session = Depends(get_db)):
    """Progress for the current dashboard user (session-based)."""
    session_user = _get_dashboard_user_from_request(request)
    if not session_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    from .db import get_user_by_phone, get_user_progress, get_user_quizzes
    user = get_user_by_phone(db, session_user["phone"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    progress = get_user_progress(db, user.id)
    quizzes = get_user_quizzes(db, user.id)
    return {
        "user": {"name": user.name, "age": user.age, "country": user.country},
        "progress": [{"id": p.id, "topic": p.topic, "lesson_step": p.lesson_step, "total_steps": p.total_steps, "completed": p.completed, "created_at": p.created_at} for p in progress],
        "quizzes": [{"id": q.id, "topic": q.topic, "lesson_step": q.lesson_step, "score": q.score, "completed": q.completed, "created_at": q.created_at} for q in quizzes],
    }


@app.get("/users/{phone_number}/progress/analytics")
async def get_user_progress_analytics(phone_number: str, request: Request, db: Session = Depends(get_db)):
    """Analytics for a user by phone (for API). Session or token can be used to restrict to self later."""
    from .db import get_user_by_phone
    user = get_user_by_phone(db, _normalize_phone(phone_number))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return _compute_analytics(db, user.id)


@app.get("/me/progress/analytics")
async def me_progress_analytics(request: Request, db: Session = Depends(get_db)):
    """Analytics for the current dashboard user (session-based)."""
    session_user = _get_dashboard_user_from_request(request)
    if not session_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    from .db import get_user_by_phone
    user = get_user_by_phone(db, session_user["phone"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return _compute_analytics(db, user.id)


@app.get("/me/quiz/{quiz_id}")
async def me_quiz_detail(quiz_id: int, request: Request, db: Session = Depends(get_db)):
    """Return one quiz with questions, correct answers, and user's answers for revision."""
    session_user = _get_dashboard_user_from_request(request)
    if not session_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    from .db import get_user_by_phone, QuizProgress
    user = get_user_by_phone(db, session_user["phone"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    quiz = db.query(QuizProgress).filter(
        QuizProgress.id == int(quiz_id),
        QuizProgress.user_id == user.id,
    ).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    try:
        questions = json.loads(quiz.questions) if isinstance(quiz.questions, str) else quiz.questions
    except (TypeError, ValueError):
        questions = []
    # Parse user_answers string (e.g. "1A, 2B, 3True") into per-question answers
    answer_dict = {}
    if quiz.user_answers:
        pairs = re.findall(r"\d+[A-DTtFf]|\d+(?:True|False)", quiz.user_answers, re.IGNORECASE)
        for pair in pairs:
            match = re.match(r"(\d+)([A-D]|True|False|T|F)", pair.strip(), re.IGNORECASE)
            if match:
                q_num = int(match.group(1))
                ans = match.group(2)
                if ans.upper() == "T":
                    ans = "True"
                elif ans.upper() == "F":
                    ans = "False"
                answer_dict[q_num] = ans
    # Build list with user_answer and correct/incorrect per question
    result = []
    for i, q in enumerate(questions):
        q_num = i + 1
        user_ans = answer_dict.get(q_num, "")
        correct_ans = str(q.get("correct_answer", "")).strip()
        options = q.get("options") or []
        # Resolve correct display: letter -> option text
        if correct_ans.upper() in ("A", "B", "C", "D") and len(options) >= ord(correct_ans.upper()) - 64:
            correct_display = options[ord(correct_ans.upper()) - 65]
        else:
            correct_display = correct_ans
        # Resolve user answer display (letter -> option text)
        if user_ans and user_ans.upper() in ("A", "B", "C", "D") and len(options) >= ord(user_ans.upper()) - 64:
            user_display = options[ord(user_ans.upper()) - 65]
        else:
            user_display = user_ans or "(no answer)"
        # True/False: normalize A/B to True/False for comparison
        if q.get("type") == "true_false" and len(options) >= 2:
            correct_norm = "True" if str(options[0]).strip().lower() == "true" and correct_ans.upper() == "A" else "False"
            user_norm = "True" if (user_ans and (user_ans.upper() == "A" or str(user_ans).strip().lower() == "true")) else "False"
            is_correct = correct_norm == user_norm
        else:
            is_correct = user_ans and (user_ans.upper() == correct_ans.upper() or user_display == correct_display)
        result.append({
            "question": q.get("question", ""),
            "type": q.get("type", "multiple_choice"),
            "options": options,
            "correct_answer": correct_ans,
            "correct_display": correct_display,
            "explanation": q.get("explanation", ""),
            "user_answer": user_ans,
            "user_display": user_display,
            "correct": bool(is_correct),
        })
    return {
        "id": quiz.id,
        "topic": quiz.topic,
        "lesson_step": quiz.lesson_step,
        "score": quiz.score,
        "total_questions": len(questions),
        "completed": quiz.completed,
        "created_at": quiz.created_at.isoformat() if quiz.created_at else None,
        "questions": result,
    }


@app.post("/send-message")
async def send_message(
    phone_number: str = Form(...),
    message: str = Form(...),
):
    """
    Send a message to a WhatsApp number (for testing/admin purposes)
    """
    if not TWILIO_CLIENT:
        raise HTTPException(status_code=503, detail="Twilio not configured")
    
    try:
        if not phone_number.startswith('whatsapp:'):
            phone_number = f'whatsapp:{phone_number}'
        
        message = TWILIO_CLIENT.messages.create(
            body=message,
            from_=f'whatsapp:{TWILIO_PHONE_NUMBER}',
            to=phone_number
        )
        
        return {
            "status": "sent",
            "message_sid": message.sid,
            "to": phone_number,
            "body": message[:100] + "..." if len(message) > 100 else message
        }
    
    except Exception as e:
        logger.error(f"Failed to send message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

@app.post("/test-lesson")
async def test_lesson(
    topic: str = Form(...),
    age: int = Form(10),
    name: str = Form("Test User")
):
    """
    Test lesson generation endpoint (for debugging)
    """
    try:
        from .llm import generate_lesson
        from .utils import format_for_whatsapp
        
        lesson_content = generate_lesson(topic, age, name)
        formatted_lesson = format_for_whatsapp(lesson_content, age)
        
        return {
            "topic": topic,
            "age": age,
            "name": name,
            "raw_lesson": lesson_content,
            "formatted_lesson": formatted_lesson
        }
    
    except Exception as e:
        logger.error(f"Failed to generate test lesson: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate lesson: {str(e)}")

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "message": getattr(exc, "detail", "The requested resource does not exist"),
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    from fastapi.responses import JSONResponse
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Something went wrong on our end. Please try again later."
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
