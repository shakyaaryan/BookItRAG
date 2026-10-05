# BookItRAG

A modular, production-ready FastAPI backend for Conversational RAG with Interview Booking capabilities.

## Features

- **Document Ingestion API**: Process PDF and TXT files, extract content, chunk text (recursive or semantic), compute embeddings, and store in Pinecone
- **Conversational RAG API**: Custom RAG engine using Google Gemini with multi-turn conversation history via Redis, scoped retrieval from Pinecone
- **Interview Booking**: Extract booking intents, store in SQL, send confirmation emails via SMTP

## Architecture

```
bookitrag/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py               # Pydantic Settings
│   ├── database.py             # SQLAlchemy setup
│   ├── models.py               # ORM models
│   ├── schemas.py              # Pydantic schemas
│   ├── api/
│   │   ├── dependencies.py     # DI (DB, Redis)
│   │   └── v1/
│   │       ├── ingestion.py    # Document ingestion
│   │       └── chat.py         # RAG & Booking
│   ├── services/
│   │   ├── document_service.py
│   │   ├── chunking_service.py
│   │   ├── vector_store.py     # Pinecone
│   │   ├── rag_service.py      # Custom RAG
│   │   ├── booking_service.py
│   │   └── memory_service.py   # Redis history
│   └── utils/
│       └── email.py            # Async SMTP
├── requirements.txt
└── .env.example
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv bookitrag_env
# Windows:
bookitrag_env\Scripts\activate
# Linux/macOS:
source bookitrag_env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

The default `.env.example` now uses **SQLite** (`sqlite:///./bookitrag.db`) for zero-config local development. The database file is created automatically on first run — no separate database server required.

> **Redis is still required** for chat history (`/api/v1/chat`). Run it locally with Docker:
> ```bash
> docker run -d -p 6379:6379 redis:7
> ```
> Or use a hosted provider (Redis Cloud, Upstash) and update `REDIS_URL` in `.env`.

4. Start the server (run from repo root so SQLite path resolves correctly):
```bash
uvicorn app.main:app --reload
```

## API Endpoints

### Document Ingestion
- `POST /api/v1/documents/ingest` - Ingest documents (multipart form)

### Conversational RAG & Booking
- `POST /api/v1/chat` - Chat with RAG and booking

## Environment Variables

All configuration is done via `.env` file. Copy `.env.example` to `.env` and fill in all values:

```bash
cp .env.example .env
```

### Required Variables

| Variable | Description | Where to Get |
|----------|-------------|--------------|
| `DATABASE_URL` | SQLite file (default) or PostgreSQL connection string | **Default:** `sqlite:///./bookitrag.db` (auto-created). For production: PostgreSQL provider (Supabase, Neon, Railway, etc.) |
| `REDIS_URL` | Redis connection URL | Your Redis provider (local via `docker run -d -p 6379:6379 redis:7`, Redis Cloud, Upstash, Railway) |
| `GOOGLE_API_KEY` | Google Gemini API key | [Google AI Studio](https://aistudio.google.com/apikey) → Create API Key |
| `PINECONE_API_KEY` | Pinecone vector DB API key | [Pinecone Console](https://app.pinecone.io/) → API Keys |
| `PINECONE_INDEX_NAME` | Pinecone index name | Create in Pinecone Console (dimension=384, metric=cosine) |
| `SMTP_SERVER` | SMTP server (e.g., smtp.gmail.com) | Your email provider |
| `SMTP_PORT` | SMTP port (587 for TLS) | Your email provider |
| `SMTP_USERNAME` | SMTP login username | Your email address |
| `SMTP_PASSWORD` | SMTP password / App Password | **Gmail: [App Password](https://myaccount.google.com/apppasswords) (enable 2FA first)** |
| `SENDER_EMAIL` | From email address | Your email address |

### Embedding Configuration (Must Match)

| Variable | Default | Note |
|----------|---------|------|
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model |
| `EMBEDDING_DIMENSION` | `384` | Must match model (384 for MiniLM-L6-v2) |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_SESSION_TTL` | `86400` | Chat session TTL in seconds (24h) |
| `LLM_MODEL_NAME` | `gemini-3.5-flash` | Gemini model variant |
| `APP_NAME` | `BookItRAG API` | Application name |
| `APP_VERSION` | `1.0.0` | Application version |
| `DEBUG` | `False` | Enable debug mode |

### Gmail App Password Setup (Required for Gmail)

1. Enable 2FA: https://myaccount.google.com/security
2. Create App Password: https://myaccount.google.com/apppasswords
3. Select "Mail" → Generate
4. Use the 16-character password in `SMTP_PASSWORD`

See `.env.example` for detailed inline comments on each variable.