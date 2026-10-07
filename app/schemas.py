from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field


class DocumentIngestRequest(BaseModel):
    chunking_method: str = Field(default="recursive", pattern="^(recursive|semantic)$")


class DocumentProcessed(BaseModel):
    doc_id: str
    file_name: str
    chunking_method: str
    chunks_created: int
    ingested_at: datetime


class DocumentIngestResponse(BaseModel):
    status: str
    processed_files: List[DocumentProcessed]


class ChatRequest(BaseModel):
    session_id: str
    message: str


class RetrievedSource(BaseModel):
    file_name: str
    page: int
    ingested_at: Optional[datetime] = None


class BookingDetails(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    date: Optional[str] = None
    time: Optional[str] = None
    job_role: Optional[str] = None
    confirmation: Optional[str] = None


class BookingStatus(BaseModel):
    status: str
    details: Optional[BookingDetails] = None
    missing: Optional[List[str]] = None
    prompt: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    booking_status: BookingStatus
    retrieved_sources: List[RetrievedSource]