sequenceDiagram
    participant U as User
    participant WA as WhatsApp
    participant T as Twilio
    participant NG as Ngrok
    participant API as FastAPI App
    participant H as Message Handler
    participant DB as SQLite DB
    participant LLM as OpenRouter/DeepSeek

   
    Note over U,LLM: Regular Usage - Lesson Request
    
    U->>WA: "/lesson fractions"
    WA->>T: Message webhook
    T->>NG: POST /whatsapp
    NG->>API: Forward request
    API->>H: process_whatsapp_message()
    H->>DB: get_user_by_phone()
    DB-->>H: User found (onboarded)
    H->>H: _handle_regular_message()
    H->>H: _handle_lesson_command()
    H->>LLM: generate_lesson("fractions", age=25, name="John")
    LLM-->>H: Generated lesson content
    H->>DB: create_progress()
    DB-->>H: Progress saved
    H->>H: format_for_whatsapp()
    H-->>API: Formatted lesson
    API-->>NG: TwiML response
    NG-->>T: Forward response
    T-->>WA: Send message
    WA-->>U: "Lesson: Fractions\n\n[Lesson content]..."

    Note over U,LLM: Help Command
    
    U->>WA: "/help"
    Note over U,H: Similar flow, returns help message

    Note over U,LLM: Next Lesson
    
    U->>WA: "/next"
    H->>DB: get_current_lesson()
    H->>LLM: generate_lesson("fractions - Advanced")
    H->>DB: update_progress()
    H-->>U: "Fractions - Part 2\n\n[Advanced content]..."