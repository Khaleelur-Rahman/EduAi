import os
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Depends, HTTPException, Form
from fastapi.responses import Response, FileResponse
from sqlalchemy.orm import Session
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient

from .db import get_db, create_tables
from .handlers import process_whatsapp_message, process_whatsapp_audio
from .llm import initialize_llm
from .rag import initialize_rag
from .audio import initialize_audio_services

# Temporary in-memory audio storage for TTS files
# Format: {audio_id: {'bytes': bytes, 'content_type': str, 'created_at': datetime}}
_temp_audio_store: Dict[str, Dict[str, Any]] = {}


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN") 
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    logger.warning("Twilio configuration not found. Please set environment variables.")
    TWILIO_CLIENT = None
else:
    TWILIO_CLIENT = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    logger.info("Twilio client initialized successfully")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting EduBot application...")
    
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
    
    # Cleanup: Clear temporary audio store on shutdown
    _temp_audio_store.clear()
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
    
    return Response(
        content=audio_data['bytes'],
        media_type=content_type,
        headers={
            'Content-Disposition': f'inline; filename="lesson_{audio_id}.mp3"',
            'Cache-Control': 'no-cache'
        }
    )

def _get_base_url(request: Request) -> str:
    """Get the base URL of the server from the request.
    Supports ngrok, proxies, and direct access.
    """
    base_url = os.getenv("BASE_URL")
    if base_url:
        return base_url.rstrip('/')
    
    # Fallback: construct from request
    # Check for forwarded protocol (for ngrok/proxies)
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    if not scheme:
        scheme = "https" if request.url.port == 443 else "http"
    
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "localhost:8000"
    
    base_url = f"{scheme}://{host}"
    logger.info(f"Generated base URL: {base_url}")
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

@app.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(None),
    From: str = Form(...),
    To: str = Form(None),
    NumMedia: str = Form("0"),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None),
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
                    
                    text = response if isinstance(response, str) else response.get('text', '')
                    
                    if isinstance(response, dict) and 'audio_bytes' in response:
                        try:
                            audio_id = _store_temp_audio(
                                response['audio_bytes'],
                                response.get('audio_content_type', 'audio/mpeg')
                            )
                            
                            base_url = _get_base_url(request)
                            audio_url = f"{base_url}/audio/{audio_id}"
                            
                            logger.info(f"Generated audio URL: {audio_url}")
                            logger.info(f"Audio file will be accessible at: {audio_url}")
                            
                            message = twiml_response.message()
                            message.media(audio_url)
                            
                            if text:
                                logger.info(f"Sending audio response with text backup: {len(text)} characters")
                            
                            logger.info(f"Successfully prepared audio response for Twilio (audio URL: {audio_url})")
                            
                        except Exception as e:
                            logger.error(f"Error preparing audio response: {e}")
                            if text:
                                twiml_response.message(text)
                    else:
                        if text:
                            twiml_response.message(text)
                            logger.info(f"Sending text response for audio message: {len(text)} characters")
                    
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
        
        if body_text.strip().lower().startswith("join ") and TWILIO_CLIENT and TWILIO_PHONE_NUMBER:
            logger.info("Detected sandbox join message. Sending proactive welcome.")
            try:
                response_text = process_whatsapp_message(db, phone_number, body_text)

                TWILIO_CLIENT.messages.create(
                    body=response_text,
                    from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
                    to=f"whatsapp:{phone_number}"
                )
                # Return empty TwiML so Twilio can still send its sandbox confirmation
                return Response(content=str(MessagingResponse()), media_type="application/xml")
            except Exception as send_err:
                logger.error(f"Failed to send proactive welcome: {str(send_err)}")
                # Fall through to normal flow

        response_text = process_whatsapp_message(db, phone_number, body_text)

        twiml_response = MessagingResponse()
        twiml_response.message(response_text)
        
        logger.info(f"Sending response to {phone_number}: {response_text[:100]}... (Length: {len(response_text)} chars)")
        
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
    from .db import get_user_by_phone, get_user_progress
    
    user = get_user_by_phone(db, phone_number)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    progress = get_user_progress(db, user.id)
    
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
    return {
        "error": "Not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": ["/", "/health", "/whatsapp"]
    }

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return {
        "error": "Internal server error",
        "message": "Something went wrong on our end. Please try again later."
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
