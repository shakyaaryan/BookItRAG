# BookItRAG

A modular, production-ready FastAPI backend for Conversational RAG with Interview Booking capabilities.

## Features

- **Document Ingestion API**: Process PDF and TXT files, extract content, chunk text (recursive or semantic), compute embeddings, and store in Pinecone
- **Conversational RAG API**: Custom RAG engine using Google Gemini with multi-turn conversation history via Redis, scoped retrieval from Pinecone
- **Interview Booking**: Extract booking intents, store in SQL, send confirmation emails via SMTP
- **User Management**: Manage user identities for data isolation across documents, vectors, and chat sessions

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
│   │   ├── dependencies.py     # DI (DB, Redis, user)
│   │   └── v1/
│   │       ├── users.py        # User endpoints
│   │       ├── ingestion.py    # Document ingestion
│   │       └── chat.py         # RAG & Booking
│   ├── services/
│   │   ├── user_service.py
│   │   ├── document_service.py
│   │   ├── chunking_service.py
│   │   ├── vector_store.py     # Pinecone
│   │   ├── rag_service.py      # Custom RAG
│   │   ├── booking_service.py
│   │   └── memory_service.py   # Redis history
│   └── utils/
│       └── email.py            # Async SMTP
├── alembic/                    # Migrations
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

4. Run migrations:
```bash
alembic upgrade head
```

5. Start the server:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

### User Management
- `POST /api/v1/users/` - Create user
- `GET /api/v1/users/{user_id}` - Get user

### Document Ingestion
- `POST /api/v1/documents/ingest` - Ingest documents (multipart form)

### Conversational RAG & Booking
- `POST /api/v1/chat` - Chat with RAG and booking

## Environment Variables

See `.env.example` for all required variables.