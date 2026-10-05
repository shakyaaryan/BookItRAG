from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.schemas import ChatRequest, ChatResponse, BookingStatus, BookingDetails, RetrievedSource
from app.services.rag_service import get_rag_service
from app.services.booking_service import get_booking_service
from app.services.memory_service import get_memory_service


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    rag_service = get_rag_service()
    booking_service = get_booking_service()
    memory_service = get_memory_service()

    history = memory_service.get_history(request.session_id)

    answer, retrieved_docs, sources = rag_service.generate_answer(
        query=request.message,
        session_id=request.session_id,
    )

    booking_result = await booking_service.process_booking(
        db=db,
        query=request.message,
        history=history,
    )

    if booking_result:
        booking_status = BookingStatus(**booking_result)
    else:
        booking_status = BookingStatus(status="none", details=None)

    # If booking needs more info, override the LLM answer with the deterministic prompt
    if booking_status.status == "needs_info":
        answer = booking_status.prompt or "Please provide the missing information."
        # Don't add the RAG answer to history on needs_info turns - we want the follow-up question
        memory_service.add_message(request.session_id, "user", request.message)
        memory_service.add_message(request.session_id, "assistant", answer)
    else:
        # Normal flow: save both messages
        memory_service.add_message(request.session_id, "user", request.message)
        memory_service.add_message(request.session_id, "assistant", answer)

    return ChatResponse(
        session_id=request.session_id,
        response=answer,
        booking_status=booking_status,
        retrieved_sources=[RetrievedSource(**s) for s in sources],
    )