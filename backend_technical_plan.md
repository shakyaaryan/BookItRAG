# FastAPI Conversational RAG & Interview Booking Backend Architecture Plan

## 1. Executive Summary & Architecture Overview

This document outlines the technical design and architectural plan for a modular, production-ready backend using FastAPI. The backend provides two core API subsystems alongside a user management system:

1. **Document Ingestion API:** Processes `.pdf` and `.txt` files, extracts content via `PyPDFLoader` and `TextLoader`, chunks text using either Recursive or Semantic chunking strategies (`sentence-transformers/all-MiniLM-L6-v2`), computes embeddings, and stores vectors in Pinecone with metadata while saving metadata records in a SQL database via SQLAlchemy.
2. **Conversational RAG API:** A custom retrieval-augmented generation engine built using Google Gemini API (`ChatGoogleGenerativeAI`) without high-level chain abstractions (`RetrievalQAChain`). It manages multi-turn history using Redis, performs scoped retrieval from Pinecone using `user_id` context, extracts interview booking intents, stores bookings in SQL, and dispatches confirmation emails via SMTP.
3. **User Management API:** Manages user identities (`user_id`) to ensure strict data isolation across documents, vector namespaces/metadata, vector searches, and session chat histories.

### Virtual Environment Instruction
> **IMPORTANT:** All development, dependency management, database migrations, and application execution **MUST** take place inside the `bookitrag_env` virtual environment.
```bash
# Activation commands
# Linux/macOS:
source bookitrag_env/bin/activate

# Windows:
bookitrag_env\Scripts\activate
```

---

## 2. Directory Structure

```text
bookitrag/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application entrypoint, CORS, exception handlers
│   ├── config.py                # Centralized Pydantic Settings
│   ├── database.py              # SQLAlchemy engine, session maker, declarative base
│   ├── models.py                # DB ORM models (User, DocumentMetadata, InterviewBooking)
│   ├── schemas.py               # Pydantic request/response schemas
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py      # Dependency injections (DB session, Redis client, current user)
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── users.py         # User management endpoints
│   │   │   ├── ingestion.py     # Document Ingestion API
│   │   │   └── chat.py          # Conversational RAG & Booking API
│   ├── services/
│   │   ├── __init__.py
│   │   ├── user_service.py     # User creation & lookup logic
│   │   ├── document_service.py  # File handling (tempfile), PyPDF/Text loading, metadata clean up
│   │   ├── chunking_service.py  # Recursive & Semantic chunking (HuggingFace embeddings)
│   │   ├── vector_store.py      # Pinecone client initialization & UPSERT operations
│   │   ├── rag_service.py       # Custom Conversational RAG chain with Gemini & Pinecone
│   │   ├── booking_service.py   # Interview booking extraction, DB persist, and confirmation dispatch
│   │   └── memory_service.py    # Redis chat history manager with user isolation
│   └── utils/
│       ├── __init__.py
│       └── email.py             # Asynchronous SMTP email sender
├── alembic/                     # Database migrations
│   ├── env.py
│   └── versions/
├── .env.example                 # Environment variables template
├── requirements.txt
└── README.md
```

---

## 3. Configuration & Environment Variables (`pydantic-settings`)

All runtime settings and user-configurable credentials are strictly centralized in `app/config.py`.

```python
# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, EmailStr

class Settings(BaseSettings):
    # API Configuration
    APP_NAME: str = "BookItRAG API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database Settings
    DATABASE_URL: str = Field(
        ..., 
        description="SQLAlchemy DB Connection string (e.g. postgresql+psycopg2://user:pass@localhost:5432/bookitrag_db)"
    )
    
    # Redis Settings
    REDIS_URL: str = Field("redis://localhost:6379/0", description="Redis connection URL for chat history")
    REDIS_SESSION_TTL: int = Field(86400, description="Chat session TTL in seconds (default 24h)")

    # Google Gemini API
    GEMINI_API_KEY: str = Field(..., description="API key for Google Gemini")
    LLM_MODEL_NAME: str = Field("gemini-1.5-flash", description="Gemini model identifier")

    # Embedding Settings
    EMBEDDING_MODEL_NAME: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", 
        description="HuggingFace model for embeddings"
    )
    EMBEDDING_DIMENSION: int = Field(384, description="Vector embedding dimension size")

    # Pinecone Settings
    PINECONE_API_KEY: str = Field(..., description="Pinecone API key")
    PINECONE_INDEX_NAME: str = Field(..., description="Pinecone index name")

    # SMTP Server Settings
    SMTP_SERVER: str = Field(..., description="SMTP server address")
    SMTP_PORT: int = Field(587, description="SMTP server port")
    SMTP_USERNAME: str = Field(..., description="SMTP login username")
    SMTP_PASSWORD: str = Field(..., description="SMTP login password")
    SENDER_EMAIL: EmailStr = Field(..., description="Sender email address for confirmation emails")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
```

---

## 4. Database Schema Design (SQLAlchemy)

The system maintains user references (`user_id`) across all entities.

```python
# app/models.py
import uuid
import enum
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Enum as SQLEnum, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base

class ChunkingMethod(str, enum.Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    documents = relationship("DocumentMetadata", back_populates="user", cascade="all, delete-orphan")
    bookings = relationship("InterviewBooking", back_populates="user", cascade="all, delete-orphan")

class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    doc_id = Column(String(100), unique=True, nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    chunking_method = Column(SQLEnum(ChunkingMethod), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="documents")

class InterviewBooking(Base):
    __tablename__ = "interview_bookings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    booking_date = Column(String(50), nullable=False)
    booking_time = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="bookings")
```

---

## 5. End-to-End Component Workflows

### 5.1 Document Ingestion Pipeline

```
[ HTTP Multipart Request ]
        │
        ▼
[ Validate User ID & File Types (.pdf, .txt) ]
        │
        ▼
[ Loop Files -> Save to NamedTemporaryFile ]
        │
        ▼
[ Load Docs (PyPDFLoader / TextLoader) ]
        │
        ▼
[ Clean Metadata: retain 'source' (file_name), 'page', add 'doc_id', 'user_id' ]
        │
        ▼
[ Split Text (RecursiveCharacterTextSplitter OR SemanticChunker) ]
        │
        ▼
[ Embed Chunks using HuggingFace 'all-MiniLM-L6-v2' ]
        │
        ▼
[ Upsert to Pinecone with Metadata (doc_id, user_id, source, page, text) ]
        │
        ▼
[ Persist Metadata Record in SQLAlchemy DocumentMetadata ]
        │
        ▼
[ Unlink Temporary Files & Return Summary JSON ]
```

#### Metadata Sanitation & Vector Payload Format
* **Pinecone Metadata Schema:**
  ```json
  {
    "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "doc_id": "doc_e2b9c512",
    "source": "contract_2026.pdf",
    "page": 3,
    "text": "Chunk content paragraph goes here..."
  }
  ```

---

### 5.2 Conversational RAG & Booking Workflow

```
[ Client Query Payload (user_id, session_id, message) ]
        │
        ▼
[ Redis Memory Service: Fetch History for key `chat:{user_id}:{session_id}` ]
        │
        ▼
[ Rephrase Standalone Query via Gemini using Contextual Prompt ]
        │
        ▼
[ Embed Standalone Query via HuggingFace Embeddings ]
        │
        ▼
[ Pinecone Similarity Search (Filter: {"user_id": user_id}, top_k=4) ]
        │
        ▼
[ Concurrently Pass Query to Booking Extractor LLM (Structured Output Schema) ]
        │
        ├────────────────────────────────────────────────────────┐
        ▼                                                        ▼
[ Answer Generator LLM ]                              [ Booking Intent Check ]
  - System Prompt + Retrieved Chunks                    - Is complete booking?
  - Context + Message History                           - User confirmed?
        │                                                        │
        ▼                                                        ▼
[ Format Natural Response ]                   [ IF Confirmed: Save to DB + Trigger Email ]
        │                                                        │
        └──────────────────────────┬─────────────────────────────┘
                                   ▼
          [ Store Turn in Redis Memory & Return Response JSON ]
```

---

## 6. Detailed API Specification

### 6.1 User Management API
* **`POST /api/v1/users/`**
  * **Summary:** Create a new user profile.
  * **Request Body:**
    ```json
    {
      "email": "alex.smith@example.com",
      "full_name": "Alex Smith"
    }
    ```
  * **Response (210 Created):**
    ```json
    {
      "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "email": "alex.smith@example.com",
      "full_name": "Alex Smith",
      "created_at": "2026-10-05T06:55:00Z"
    }
    ```

* **`GET /api/v1/users/{user_id}`**
  * **Summary:** Retrieve user profile by ID.

---

### 6.2 Document Ingestion API
* **`POST /api/v1/documents/ingest`**
  * **Headers / Context:** `user_id` (Query parameter or Form field)
  * **Form Data:**
    * `chunking_method`: `recursive` (default) | `semantic`
    * `files`: List of binary files (`.pdf`, `.txt`)
  * **Response (200 OK):**
    ```json
    {
      "status": "success",
      "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "processed_files": [
        {
          "doc_id": "doc_e2b9c512",
          "file_name": "resume.pdf",
          "chunking_method": "recursive",
          "chunks_created": 12
        }
      ]
    }
    ```

---

### 6.3 Conversational RAG & Booking API
* **`POST /api/v1/chat`**
  * **Request Body:**
    ```json
    {
      "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "session_id": "sess_89123a",
      "message": "Can you book an interview for me tomorrow at 3 PM? My email is alex@example.com and name is Alex."
    }
    ```
  * **Response (200 OK):**
    ```json
    {
      "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "session_id": "sess_89123a",
      "response": "I have extracted your scheduling request for October 6th at 3:00 PM. Could you please confirm if you would like me to finalize this booking?",
      "booking_status": {
        "status": "pending_confirmation",
        "details": {
          "name": "Alex",
          "email": "alex@example.com",
          "date": "2026-10-06",
          "time": "15:00"
        }
      },
      "retrieved_sources": [
        {
          "file_name": "resume.pdf",
          "page": 1
        }
      ]
    }
    ```

---

## 7. Implementation Guidelines & Constraints

1. **Explicit Typing:** All functions must include strict Python type hints (`UUID`, `Optional[str]`, `List[Document]`, etc.).
2. **No High-Level RAG Abstractions:** `RetrievalQAChain`, `ConversationalRetrievalChain`, FAISS, and Chroma are strictly prohibited. The RAG pipeline must explicitly invoke `vector_store.similarity_search_by_vector` and format prompts directly using `ChatGoogleGenerativeAI`.
3. **User Data Isolation:**
   * **Pinecone Queries:** Must pass `filter={"user_id": {"$eq": str(user_id)}}`.
   * **Redis Keys:** Must follow `chat_history:{user_id}:{session_id}`.
   * **SQL Queries:** Must filter by `user_id`.
4. **Email Dispatch:** Email logic in `app/utils/email.py` must run asynchronously using `aiosmtplib` to avoid blocking FastAPI event loops during HTTP requests.
```

The comprehensive backend architecture plan has been updated and formatted into a markdown document. 

### Key System Highlights Included in the Blueprint:
1. **User Identity (`user_id`):** Embedded across SQL database relationships, Pinecone vector metadata filters (`filter={"user_id": ...}`), and Redis history keys (`chat_history:{user_id}:{session_id}`).
2. **User Management API:** Added explicit endpoints (`POST /api/v1/users/` and `GET /api/v1/users/{user_id}`) for managing user identities.
3. **Execution Context:** The plan mandates execution within the `bookitrag_env` virtual environment and enforces complete documentation standards.

Feel free to ask if you would like me to generate specific modules (e.g., the ingestion service, Gemini RAG, or Redis memory manager) next!