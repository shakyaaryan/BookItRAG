from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.schemas import ChatRequest, ChatResponse, BookingStatus, BookingDetails, RetrievedSource
from app.services.vector_store import get_vector_store
from app.services.memory_service import get_memory_service
from app.services.agent_service import get_agent_service


router = APIRouter(prefix="/chat", tags=["chat"])


def _prioritize_latest(docs: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    """Sort documents by ingested_at descending (latest first) and return top_k."""
    from datetime import datetime
    def get_ingested_at(doc):
        meta = doc.get("metadata", {})
        ingested_str = meta.get("ingested_at")
        if ingested_str:
            try:
                return datetime.fromisoformat(ingested_str)
            except ValueError:
                pass
        return datetime.min
    
    sorted_docs = sorted(docs, key=get_ingested_at, reverse=True)
    return sorted_docs[:top_k]


def _format_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    if not retrieved_docs:
        return "No relevant documents found."

    context_parts = []
    for doc in retrieved_docs:
        metadata = doc.metadata
        source = metadata.get("source", "unknown")
        page = metadata.get("page", 0)
        text = metadata.get("text", "")
        ingested_at = metadata.get("ingested_at", "")
        context_parts.append(f"[Source: {source}, Page: {page}, Ingested: {ingested_at}]\n{text}")

    return "\n\n---\n\n".join(context_parts)


@router.post(
    "",
    response_model=ChatResponse,
    summary="Conversational RAG with Interview Booking",
    description="""Chat with the BookItRAG agent for job-related questions and interview booking.

**Architecture:**
- Pre-fetches relevant job postings from Pinecone (top 10 → latest 4)
- Runs a LangGraph agent (Gemini) with pre-fetched context and 2 booking tools
- Maintains conversation history in Redis (per session_id)

**Agent Tools:**
1. `create_pending_booking(name, email, job_role, date, time)` — Creates a pending booking when ALL 5 fields are collected
2. `confirm_booking(email)` — Confirms pending booking, flips to "confirmed", sends SMTP email

**Booking Flow (Multi-turn):**
1. User expresses booking intent → Agent asks for missing fields (name, email, job_role, date, time)
2. User provides all details → Agent calls `create_pending_booking` → Returns `booking_status: "pending_confirmation"` with details
3. Agent asks: "Please confirm to book this interview."
4. User explicitly confirms ("yes", "confirm", "book it") → Agent calls `confirm_booking` → Sends email → Returns `booking_status: "confirmed"`

**Response Fields:**
- `response`: Agent's answer (job info or booking prompt)
- `booking_status`: 
  - `status`: "none" | "pending_confirmation" | "confirmed"
  - `details`: BookingDetails object with fields
  - `missing`: List of missing field names (if status="pending_confirmation")
  - `prompt`: Agent's follow-up question (if any)
- `retrieved_sources`: List of source documents used for RAG answer

**Date/Time Format:**
- Dates: YYYY-MM-DD (relative dates like "tomorrow", "next Friday" resolved using today's date)
- Times: HH:MM 24-hour format

**Session Management:**
- `session_id`: Client-generated unique identifier for conversation
- History stored in Redis with TTL (default 24h)
- Booking state correlated by email (latest pending booking per email)
""",
)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    memory_service = get_memory_service()
    vector_store = get_vector_store()
    agent_service = get_agent_service()

    history = memory_service.get_history(request.session_id)

    # 1. Pre-fetch RAG context
    retrieved_docs = vector_store.similarity_search(request.message, top_k=10)
    prioritized_docs = _prioritize_latest(retrieved_docs, top_k=4)
    rag_context = _format_context(prioritized_docs)

    # Format sources for response
    sources = []
    for doc in prioritized_docs:
        metadata = doc.metadata
        sources.append({
            "file_name": metadata.get("source", "unknown"),
            "page": metadata.get("page", 0),
            "ingested_at": metadata.get("ingested_at"),
        })

    # 2. Run agent with pre-fetched context
    answer, agent_sources, booking_status_dict = await agent_service.run(
        session_id=request.session_id,
        message=request.message,
        history=history,
        rag_context=rag_context,
    )

    # 3. Build booking status
    booking_status = BookingStatus(**booking_status_dict)

    # 4. Update memory
    memory_service.add_message(request.session_id, "user", request.message)
    memory_service.add_message(request.session_id, "assistant", answer)

    return ChatResponse(
        session_id=request.session_id,
        response=answer,
        booking_status=booking_status,
        retrieved_sources=[RetrievedSource(**s) for s in sources],
    )