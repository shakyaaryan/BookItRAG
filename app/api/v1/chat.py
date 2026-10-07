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


@router.post("", response_model=ChatResponse)
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