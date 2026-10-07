from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from pydantic import BaseModel

from app.database import get_db
from app.models import InterviewBooking


router = APIRouter(prefix="/bookings", tags=["bookings"])


class BookingListItem(BaseModel):
    """Booking info for listing."""
    id: str
    name: str
    email: str
    job_role: str
    booking_date: str
    booking_time: str
    confirmation: str
    created_at: datetime


class BookingDetail(BookingListItem):
    """Detailed booking info."""
    pass


@router.get(
    "",
    response_model=List[BookingListItem],
    summary="List all interview bookings",
    description="""Retrieve a paginated list of all interview bookings, ordered by creation date (newest first).

**Pagination:**
- `limit`: Number of items per page (1-100, default 50)
- `offset`: Number of items to skip (default 0)

**No email filter** — returns all bookings across all candidates.

**Response:** Array of bookings with:
- `id`: UUID string
- `name`: Candidate name
- `email`: Candidate email
- `job_role`: Applied position
- `booking_date`: Interview date (YYYY-MM-DD)
- `booking_time`: Interview time (HH:MM 24-hour)
- `confirmation`: "pending" or "confirmed"
- `created_at`: ISO 8601 timestamp
""",
)
async def list_bookings(
    limit: int = Query(default=50, ge=1, le=100, description="Items per page (1-100)"),
    offset: int = Query(default=0, ge=0, description="Items to skip"),
    db: Session = Depends(get_db),
):
    """List all interview bookings with pagination."""
    bookings = db.query(InterviewBooking).order_by(
        InterviewBooking.created_at.desc()
    ).offset(offset).limit(limit).all()
    
    result = []
    for booking in bookings:
        result.append(BookingListItem(
            id=str(booking.id),
            name=booking.name,
            email=booking.email,
            job_role=booking.job_role,
            booking_date=booking.booking_date,
            booking_time=booking.booking_time,
            confirmation=booking.confirmation,
            created_at=booking.created_at,
        ))
    
    return result


@router.get(
    "/{booking_id}",
    response_model=BookingDetail,
    summary="Get booking by ID",
    description="""Retrieve a specific interview booking by its UUID.

**Path Parameter:**
- `booking_id`: UUID string (e.g., "123e4567-e89b-12d3-a456-426614174000")

**Returns:** Full booking details including all fields.

**Errors:**
- `400`: Invalid UUID format
- `404`: Booking not found
""",
)
async def get_booking(
    booking_id: str,
    db: Session = Depends(get_db),
):
    """Get a specific booking by ID."""
    from uuid import UUID
    try:
        uuid_obj = UUID(booking_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid booking ID format")
    
    booking = db.query(InterviewBooking).filter(InterviewBooking.id == uuid_obj).first()
    
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    return BookingDetail(
        id=str(booking.id),
        name=booking.name,
        email=booking.email,
        job_role=booking.job_role,
        booking_date=booking.booking_date,
        booking_time=booking.booking_time,
        confirmation=booking.confirmation,
        created_at=booking.created_at,
    )