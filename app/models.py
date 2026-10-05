import uuid
import enum
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Enum as SQLEnum, Index
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


class ChunkingMethod(str, enum.Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(String(100), unique=True, nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    chunking_method = Column(SQLEnum(ChunkingMethod), nullable=False)
    ingested_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class InterviewBooking(Base):
    __tablename__ = "interview_bookings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    job_role = Column(String(255), nullable=False)
    booking_date = Column(String(50), nullable=False)
    booking_time = Column(String(50), nullable=False)
    confirmation = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)