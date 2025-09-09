import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Depends, HTTPException, Form
from fastapi.responses import Response
from sqlalchemy.orm import Session
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient

from .db import get_db, create_tables
from .handlers import process_whatsapp_message
from .llm import initialize_llm

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
    logger.info("Starting WhatsApp AI Tutor application...")
    
    create_tables()
    logger.info("Database tables created/verified")
    try:
        logger.info("Initializing LLM model... This may take a few minutes on first run.")
        initialize_llm()
        logger.info("LLM model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        logger.warning("Application will continue but lessons may use fallback content")
    
    yield
    
    logger.info("Shutting down WhatsApp AI Tutor application...")

app = FastAPI(
    title="WhatsApp AI Tutor",
    description="An AI-powered tutoring system via WhatsApp using Twilio",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {
        "message": "WhatsApp AI Tutor is running!",
        "status": "healthy",
        "endpoints": {
            "webhook": "/whatsapp",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "database": "connected",
        "llm": "unknown",
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
    
    if TWILIO_CLIENT:
        health_status["twilio"] = "configured"
    else:
        health_status["twilio"] = "not_configured"
        health_status["status"] = "degraded"
    
    return health_status

@app.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...),
    To: str = Form(None),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Received WhatsApp message from {From}: {Body[:100]}...")
        
        phone_number = From.replace('whatsapp:', '').strip()
        
        if not phone_number:
            logger.error("Invalid phone number received")
            raise HTTPException(status_code=400, detail="Invalid phone number")
        
        response_text = process_whatsapp_message(db, phone_number, Body)
        
        twiml_response = MessagingResponse()
        twiml_response.message(response_text)
        
        logger.info(f"Sending response to {phone_number}: {response_text[:100]}...")
        
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
