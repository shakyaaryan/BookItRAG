# BookItRAG

A modular, production-ready FastAPI backend for **Conversational RAG with Interview Booking** capabilities. Built with Google Gemini, Pinecone, Redis, and SQLAlchemy.

## Features

- **Document Ingestion API**: Process PDF and TXT files, extract content, chunk text (recursive or semantic), compute embeddings via HuggingFace transformers, and store in Pinecone vector database
- **Conversational RAG API**: Agent-based RAG engine using Google Gemini with pre-fetched context, multi-turn conversation history via Redis, and intelligent tool-calling for booking workflows
- **Interview Booking System**: Two-step booking flow (pending → confirmed) with explicit user confirmation, SQL persistence, and SMTP email confirmations
- **Admin View APIs**: List ingested documents and interview bookings with pagination

## Architecture

```
bookitrag/
├── app/
│   ├── main.py                 # FastAPI entrypoint, lifespan, routers
│   ├── config.py               # Pydantic Settings (env-driven)
│   ├── database.py             # SQLAlchemy engine & session
│   ├── models.py               # ORM models (DocumentMetadata, InterviewBooking)
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── api/
│   │   ├── dependencies.py     # DI providers (DB, Redis)
│   │   └── v1/
│   │       ├── ingestion.py    # POST /documents/ingest
│   │       ├── chat.py         # POST /chat (RAG + Booking agent)
│   │       ├── documents.py    # GET /documents (list)
│   │       └── bookings.py     # GET /bookings, GET /bookings/{id}
│   ├── services/
│   │   ├── document_service.py # Load & save uploaded files
│   │   ├── chunking_service.py # Recursive / Semantic chunking
│   │   ├── vector_store.py     # Pinecone upsert & similarity search
│   │   ├── rag_service.py      # Legacy RAG (reference)
│   │   ├── booking_service.py  # Legacy booking (reference)
│   │   ├── agent_service.py    # LangGraph agent with booking tools
│   │   └── memory_service.py   # Redis chat history
│   └── utils/
│       ├── email.py            # Async + Sync SMTP email
│       └── llm_text.py         # Shared LLM text extraction
├── requirements.txt
├── .env.example
└── README.md
```

## Prerequisites

- **Python 3.14+**
- **Redis** (for chat history) — see Docker commands below
- **Pinecone Account** (vector database) — free tier available
- **Google AI Studio API Key** (Gemini) — free tier available
- **SMTP Credentials** (for booking emails) — Gmail App Password recommended

## Quick Start

### 1. Clone & Create Virtual Environment

```bash
git clone https://github.com/shakyaaryan/BookItRAG.git
cd BookItRAG

python -m venv bookitrag-env
# Windows:
bookitrag-env\Scripts\activate
# Linux/macOS:
source bookitrag-env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Redis with Docker

```bash
# Start Redis container (persistent, named 'redis')
docker run -d -p 6379:6379 --name redis redis:7

# Verify it's running
docker ps | findstr redis

# Stop Redis
docker stop redis

# Start existing container
docker start redis

# View logs
docker logs redis
```

**Alternative**: Use a hosted Redis provider (Redis Cloud, Upstash, Railway) and update `REDIS_URL` in `.env`.

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# Required
DATABASE_URL=sqlite:///./bookitrag.db          # Auto-created SQLite file
REDIS_URL=redis://localhost:6379/0             # Your Redis instance
GOOGLE_API_KEY=your_gemini_api_key             # Google AI Studio
PINECONE_API_KEY=your_pinecone_api_key         # Pinecone Console
PINECONE_INDEX_NAME=bookitrag-index            # Must exist (dim=384, cosine)

# SMTP for booking confirmations (Gmail example)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_16_char_app_password        # NOT your login password!
SENDER_EMAIL=your_email@gmail.com

# Embeddings (must match Pinecone index dimension)
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Optional
LLM_MODEL_NAME=gemini-3.8-flash                # Current model
REDIS_SESSION_TTL=86400
DEBUG=False
```

**Gmail App Password Setup** (required for Gmail SMTP):
1. Enable 2FA: https://myaccount.google.com/security
2. Create App Password: https://myaccount.google.com/apppasswords
3. Select "Mail" → Generate → Use the 16-character password in `SMTP_PASSWORD`

### 5. Start the Server

```bash
# Run from repo root so SQLite path resolves correctly
uvicorn app.main:app --reload --port 8000
```

Server runs at: **http://localhost:8000**

- Interactive API docs: **http://localhost:8000/docs**
- ReDoc: **http://localhost:8000/redoc**
- Health check: **http://localhost:8000/health**

---

## API Endpoints

### Document Ingestion

#### `POST /api/v1/documents/ingest`
Ingest PDF or TXT documents for RAG retrieval.

**Request** (multipart/form-data):
- `files` (List[UploadFile], required): PDF/TXT files
- `chunking_method` (string, optional): `recursive` (default) or `semantic`

**Response**:
```json
{
  "status": "success",
  "processed_files": [
    {
      "doc_id": "doc_abc123",
      "file_name": "Backend_Developer.pdf",
      "chunking_method": "recursive",
      "ingested_at": "2026-10-07T03:09:38.326534"
    }
  ]
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -F "files=@./10_IT_Job_Postings_PDFs/Backend_Developer.pdf" \
  -F "chunking_method=recursive"
```

---

### Chat (RAG + Booking Agent)

#### `POST /api/v1/chat`
Conversational RAG with interview booking. Single agent call with pre-fetched context.

**Request**:
```json
{
  "session_id": "unique-session-id",
  "message": "What backend developer jobs are available?"
}
```

**Response**:
```json
{
  "session_id": "unique-session-id",
  "response": "Based on the job postings, we have a Backend Developer role...",
  "booking_status": {
    "status": "none",
    "details": null,
    "missing": null,
    "prompt": null
  },
  "retrieved_sources": [
    {
      "file_name": "Backend_Developer.pdf",
      "page": 0,
      "ingested_at": "2026-10-07T03:09:38.326534"
    }
  ]
}
```

**Booking Flow** (multi-turn conversation):
1. User provides details → agent calls `create_pending_booking` → returns `booking_status: "pending_confirmation"` with details
2. User confirms ("yes", "confirm", "book it") → agent calls `confirm_booking` → sends email → returns `booking_status: "confirmed"`

**cURL Examples**:
```bash
# General question
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test123", "message": "What backend developer jobs are available?"}'

# Start booking (agent asks for missing fields)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test123", "message": "I want to book an interview for Backend Developer"}'

# Provide all details at once
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test123", "message": "Name: John Doe, Email: john@example.com, Role: Backend Developer, Date: 2026-10-15, Time: 14:00"}'

# Confirm booking
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test123", "message": "Yes, confirm the booking"}'
```

---

### Document Listing

#### `GET /api/v1/documents`
List all ingested documents with pagination.

**Query Parameters**:
- `limit` (int, 1-100, default: 50)
- `offset` (int, default: 0)

**Response**:
```json
[
  {
    "doc_id": "doc_abc123",
    "file_name": "Backend_Developer.pdf",
    "chunking_method": "recursive",
    "ingested_at": "2026-10-07T03:09:38.326534"
  }
]
```

**cURL Example**:
```bash
curl "http://localhost:8000/api/v1/documents?limit=10&offset=0"
```

---

### Booking Listing

#### `GET /api/v1/bookings`
List all interview bookings with pagination (no email filter).

**Query Parameters**:
- `limit` (int, 1-100, default: 50)
- `offset` (int, default: 0)

**Response**:
```json
[
  {
    "id": "uuid-string",
    "name": "John Doe",
    "email": "john@example.com",
    "job_role": "Backend Developer",
    "booking_date": "2026-10-15",
    "booking_time": "14:00",
    "confirmation": "confirmed",
    "created_at": "2026-10-07T03:15:22.123456"
  }
]
```

**cURL Example**:
```bash
curl "http://localhost:8000/api/v1/bookings?limit=20&offset=0"
```

#### `GET /api/v1/bookings/{booking_id}`
Get a specific booking by UUID.

**Path Parameter**: `booking_id` (UUID string)

**cURL Example**:
```bash
curl "http://localhost:8000/api/v1/bookings/123e4567-e89b-12d3-a456-426614174000"
```

---

## Environment Variables Reference

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | SQLAlchemy connection string | `sqlite:///./bookitrag.db` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `GOOGLE_API_KEY` | Google Gemini API key | From Google AI Studio |
| `PINECONE_API_KEY` | Pinecone API key | From Pinecone Console |
| `PINECONE_INDEX_NAME` | Pinecone index name (dim=384, cosine) | `bookitrag-index` |
| `SMTP_SERVER` | SMTP server hostname | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP port | `587` |
| `SMTP_USERNAME` | SMTP username (email) | `your_email@gmail.com` |
| `SMTP_PASSWORD` | SMTP password / App Password | `abcd efgh ijkl mnop` |
| `SENDER_EMAIL` | From email address | `your_email@gmail.com` |

### Embedding (Must Match Pinecone Index)

| Variable | Default | Note |
|----------|---------|------|
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model |
| `EMBEDDING_DIMENSION` | `384` | Must match index dimension |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_SESSION_TTL` | `86400` | Chat session TTL (seconds) |
| `LLM_MODEL_NAME` | `gemini-3.8-flash` | Gemini model variant |
| `APP_NAME` | `BookItRAG API` | Application name |
| `APP_VERSION` | `1.0.0` | Application version |
| `DEBUG` | `False` | Enable debug mode |

---

## Usage Examples

### Complete Workflow

```bash
# 1. Ingest job postings
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -F "files=@./10_IT_Job_Postings_PDFs/Backend_Developer.pdf" \
  -F "files=@./10_IT_Job_Postings_PDFs/Frontend_Developer.pdf" \
  -F "chunking_method=recursive"

# 2. Ask about jobs
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-001", "message": "What frontend developer roles are available?"}'

# 3. Book interview (multi-turn)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-001", "message": "I want to apply for Frontend Developer"}'

# Agent responds asking for missing fields (name, email, date, time)
# Provide all details:
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-001", "message": "Jane Smith, jane@email.com, Frontend Developer, 2026-10-20, 10:30"}'

# Agent creates pending booking, asks for confirmation
# Confirm:
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-001", "message": "Yes, please book it"}'

# 4. View all bookings
curl "http://localhost:8000/api/v1/bookings"
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `429 RESOURCE_EXHAUSTED` | Gemini free tier quota (20 req/day) | Wait for reset, upgrade plan, or use different model |
| `SMTPAuthenticationError` | Invalid Gmail credentials | Use App Password (not login password), enable 2FA |
| `Connection refused: localhost:6379` | Redis not running | `docker start redis` or check `REDIS_URL` |
| `Pinecone index not found` | Index name mismatch | Verify `PINECONE_INDEX_NAME` exists with dim=384 |
| `Empty response ("")` | LLM content extraction bug | Fixed via `extract_llm_text` utility; update if needed |
| `ModuleNotFoundError` | Dependencies not installed | `pip install -r requirements.txt` in activated venv |

### Debug Tips

- Enable `DEBUG=True` in `.env` for verbose logs
- Check server logs for email send attempts (`Sending confirmation email to...`)
- Visit `/docs` for interactive API testing
- Use `docker logs redis` to verify Redis health

---

## Development Notes

### Database
- **Default**: SQLite (`sqlite:///./bookitrag.db`) — file auto-created on startup
- **Production**: Use PostgreSQL (update `DATABASE_URL`)
- **Schema**: Created automatically via `Base.metadata.create_all()` on startup
- **Reset**: Delete `bookitrag.db` and restart server

### Models
- `DocumentMetadata`: `doc_id`, `file_name`, `chunking_method`, `ingested_at`
- `InterviewBooking`: `id` (UUID), `name`, `email`, `job_role`, `booking_date`, `booking_time`, `confirmation` (pending/confirmed), `created_at`

### Agent Architecture
- **Pre-fetch RAG**: Vector search runs before agent call (1 call for retrieval)
- **Agent**: LangGraph `create_agent` with 2 tools (`create_pending_booking`, `confirm_booking`)
- **Async execution**: Sync agent runs in threadpool via `asyncio.to_thread`
- **Email**: Sync `smtplib` in worker thread (async `aiosmtplib` for direct calls)

---

## License

MIT License — feel free to use and modify.