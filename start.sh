#!/bin/bash

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

if [ ! -f ".env" ]; then
    echo "Error: .env file not found"
    echo "Create .env file with:"
    echo "OPENROUTER_API_KEY=your_key"
    echo "TWILIO_ACCOUNT_SID=your_sid"
    echo "TWILIO_AUTH_TOKEN=your_token"
    echo "TWILIO_PHONE_NUMBER=your_number"
    exit 1
fi

set -a
source .env
set +a

if [ -z "$OPENROUTER_API_KEY" ] || [ -z "$TWILIO_ACCOUNT_SID" ] || [ -z "$TWILIO_AUTH_TOKEN" ] || [ -z "$TWILIO_PHONE_NUMBER" ]; then
    echo "Error: Missing required environment variables"
    exit 1
fi

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
