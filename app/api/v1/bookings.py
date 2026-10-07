from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
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


@router.get("", response_model=List[BookingListItem])
async def list_bookings(
    email: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """List all interview bookings with optional email filter and pagination."""
    query = db.query(InterviewBooking)
    
    if email:
        query = query.filter(InterviewBooking.email == email)
    
    bookings = query.order_by(
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


@router.get("/{booking_id}", response_model=BookingDetail)
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