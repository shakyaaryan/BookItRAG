from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=255)


class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentIngestRequest(BaseModel):
    user_id: UUID
    chunking_method: str = Field(default="recursive", pattern="^(recursive|semantic)$")


class DocumentProcessed(BaseModel):
    doc_id: str
    file_name: str
    chunking_method: str
    chunks_created: int


class DocumentIngestResponse(BaseModel):
    status: str
    user_id: UUID
    processed_files: List[DocumentProcessed]


class ChatRequest(BaseModel):
    user_id: UUID
    session_id: str
    message: str


class RetrievedSource(BaseModel):
    file_name: str
    page: int


class BookingDetails(BaseModel):
    name: str
    email: EmailStr
    date: str
    time: str


class BookingStatus(BaseModel):
    status: str
    details: Optional[BookingDetails] = None


class ChatResponse(BaseModel):
    user_id: UUID
    session_id: str
    response: str
    booking_status: BookingStatus
    retrieved_sources: List[RetrievedSource]