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