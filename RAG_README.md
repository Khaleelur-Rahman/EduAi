# RAG-Powered Science Education System

## Overview

This document describes the RAG (Retrieval-Augmented Generation) implementation that extends your WhatsApp AI Tutor to provide accurate, grounded science lessons for children aged 6-12.

## Architecture

### Components

1. **Vector Database**: ChromaDB (local, free)
2. **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (free)
3. **LLM**: qwen-3-235b-a22b-instruct via Cerebras (free tier)
4. **Content Source**: Curated science educational content in `./data/science/`

### Data Flow

```
User Query → RAG Retrieval → Context + LLM → Age-Appropriate Lesson
```

## Features

### ✅ Implemented Features

- **Science Topic Detection**: Automatically identifies science topics for RAG processing
- **Age-Appropriate Content**: Lessons tailored for ages 6-12 with appropriate language complexity
- **Multi-Part Lessons**: Users can continue lessons with `/next` command
- **Progress Tracking**: Database tracks chunk IDs and lesson progress
- **Fallback System**: Graceful degradation if RAG is unavailable
- **Free Components**: Uses only free models and local storage

## Database Schema Updates

### New Fields in Progress Table

```sql
-- RAG-specific fields
chunk_id VARCHAR(100)      -- Track current chunk for continuation
is_rag_lesson BOOLEAN      -- Whether this is a RAG lesson
rag_context TEXT           -- Store retrieved context for reference
```