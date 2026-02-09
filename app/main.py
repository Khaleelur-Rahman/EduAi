import os
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Depends, HTTPException, Form, BackgroundTasks
from fastapi.responses import Response, FileResponse
from sqlalchemy.orm import Session
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient

from .db import get_db, create_tables
from .handlers import process_whatsapp_message, process_whatsapp_audio, process_whatsapp_message_request_audio, process_whatsapp_message_request_image
from .llm import initialize_llm
from .rag import initialize_rag
from .audio import initialize_audio_services

# Temporary in-memory audio storage for TTS files
# Format: {audio_id: {'bytes': bytes, 'content_type': str, 'created_at': datetime}}
_temp_audio_store: Dict[str, Dict[str, Any]] = {}

# Temporary in-memory image storage for generated lesson images
# Format: {image_id: {'bytes': bytes, 'content_type': str, 'created_at': datetime}}
_temp_image_store: Dict[str, Dict[str, Any]] = {}


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

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

    # Detect /image command
    if msg_lower.startswith("/image "):
        topic = msg_stripped[7:].strip()
        if topic:
            _send_loading_message(phone_number, "lesson", topic, language)
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
        _send_loading_message(phone_number, "progress")
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
        logger.warning("BASE_URL not set; media URLs will use webhook request host (set BASE_URL for Twilio media to work behind tunnels)")
    create_tables()
    logger.info("Database tables created/verified")
    try:
        logger.info("Initializing LLM model... This may take a few minutes on first run.")
        initialize_llm()
        logger.info("LLM model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        logger.warning("Application will continue but lessons may use fallback content")
    
    try:
        logger.info("Initializing RAG service... This may take a few minutes on first run.")
        initialize_rag()
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {str(e)}")
        logger.warning("Application will continue but science lessons may not be available")
    
    try:
        logger.info("Initializing audio services (STT/TTS)...")
        initialize_audio_services()
        logger.info("Audio services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize audio services: {str(e)}")
        logger.warning("Application will continue but audio features may not be available")
    
    yield
    
    # Cleanup: Clear temporary media stores on shutdown
    _temp_audio_store.clear()
    _temp_image_store.clear()
    logger.info("Shutting down EduBot application...")

app = FastAPI(
    title="EduBot",
    description="An AI-powered tutoring system via WhatsApp using Twilio",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {
        "message": "EduBot is running!",
        "status": "healthy",
        "endpoints": {
            "webhook": "/whatsapp",
            "health": "/health",
            "audio": "/audio/{audio_id} (temporary TTS audio serving)"
        }
    }

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
            'Content-Disposition': f'inline; filename="lesson_{audio_id}.mp3"',
            'Content-Length': str(len(audio_bytes)),
            'Cache-Control': 'no-cache',
        }
    )


@app.get("/image/{image_id}")
async def serve_image(image_id: str, request: Request):
    """
    Temporary endpoint to serve generated images for Twilio media messages.
    Images are stored in memory and expire after 1 hour.
    """
    if image_id not in _temp_image_store:
        raise HTTPException(status_code=404, detail="Image not found or expired")
    
    image_data = _temp_image_store[image_id]
    
    if datetime.utcnow() - image_data['created_at'] > timedelta(hours=1):
        del _temp_image_store[image_id]
        raise HTTPException(status_code=404, detail="Image expired")
    
    content_type = image_data.get('content_type', 'image/jpeg')
    image_bytes = image_data['bytes']
    return Response(
        content=image_bytes,
        media_type=content_type,
        headers={
            'Content-Disposition': f'inline; filename="lesson_{image_id}.jpg"',
            'Content-Length': str(len(image_bytes)),
            'Cache-Control': 'no-cache',
        }
    )

def _get_base_url(request: Request) -> str:
    """Get the base URL of the server from the request.
    Supports ngrok, cloudflared, and direct access.
    Prefer BASE_URL so media URLs work when webhook is behind a different tunnel.
    """
    base_url = os.getenv("BASE_URL")
    if base_url:
        base_url = base_url.rstrip("/")
        logger.debug(f"Using BASE_URL for media: {base_url}")
        return base_url
    # Fallback: construct from request (same host as webhook)
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    if not scheme:
        scheme = "https" if request.url.port == 443 else "http"
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "localhost:8000"
    base_url = f"{scheme}://{host}"
    logger.info(f"Using request host for media URL: {base_url}")
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
    """Store image bytes temporarily and return a unique ID."""
    image_id = str(uuid.uuid4())
    _temp_image_store[image_id] = {
        'bytes': image_bytes,
        'content_type': content_type,
        'created_at': datetime.utcnow()
    }

    current_time = datetime.utcnow()
    expired_ids = [
        iid for iid, data in _temp_image_store.items()
        if current_time - data['created_at'] > timedelta(hours=1)
    ]
    for expired_id in expired_ids:
        del _temp_image_store[expired_id]

    logger.info(f"Stored temporary image file: {image_id} ({len(image_bytes)} bytes)")
    return image_id


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
            # Image (e.g. from /image command)
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
    """Apply dict response (from process_whatsapp_audio or process_whatsapp_message_request_audio) to twiml_response."""
    text = response.get("text", "")
    tts_failed = response.get("tts_failed", False)
    if tts_failed and text:
        if text.strip().lower().startswith("sorry,") or "trouble" in text.lower():
            twiml_response.message(text)
        else:
            twiml_response.message("Audio couldn't be generated. Here's your lesson:\n\n" + text)
        logger.info(f"TTS fallback: sent text backup ({len(text)} chars)")
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
    else:
        if text:
            twiml_response.message(text)

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
    db: Session
) -> None:
    """Process message and send response via REST API in background."""
    try:
        msg_lower = body_text.strip().lower()
        if msg_lower.startswith("/audio"):
            response = process_whatsapp_message_request_audio(db, phone_number, body_text)
        elif msg_lower.startswith("/image"):
            response = process_whatsapp_message_request_image(db, phone_number, body_text)
        else:
            response = process_whatsapp_message(db, phone_number, body_text)
        
        # Send response via REST API (pass base_url as string)
        _send_response_via_rest(base_url, phone_number, response)
    except Exception as e:
        logger.error(f"Error in background message processing: {e}")

@app.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(None),
    From: str = Form(...),
    To: str = Form(None),
    NumMedia: str = Form("0"),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    try:
        phone_number = From.replace('whatsapp:', '').strip()
        
        if not phone_number:
            logger.error("Invalid phone number received")
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
            msg_lower.startswith("/image") or
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
            # Send loading message immediately with user's language
            _detect_command_and_send_loading(phone_number, body_text, user_language)
            # Extract base URL before background task (request may not be available in background)
            base_url = _get_base_url(request)
            # Process in background and send via REST API
            background_tasks.add_task(_process_message_in_background, phone_number, body_text, base_url, db)
            # Return empty TwiML immediately so webhook responds fast
            return Response(content=str(MessagingResponse()), media_type="application/xml")
        
        if body_text.strip().lower().startswith("join ") and TWILIO_CLIENT and TWILIO_PHONE_NUMBER:
            logger.info("Detected sandbox join message. Sending proactive welcome.")
            try:
                response_text = process_whatsapp_message(db, phone_number, body_text)

                TWILIO_CLIENT.messages.create(
                    body=response_text,
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
            response = process_whatsapp_message(db, phone_number, body_text)

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
