import json
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings
from app.models import InterviewBooking
from app.utils.email import send_booking_confirmation_email


class BookingService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
        )

    def extract_booking_info(self, query: str, history: list) -> Optional[Dict[str, Any]]:
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-6:]])

        prompt = f"""Extract interview booking information from the user's message. Return a JSON object with the following fields if the user is requesting to book an interview:
- name: Full name of the person
- email: Email address
- date: Date of the interview (YYYY-MM-DD format)
- time: Time of the interview (HH:MM format, 24-hour)

If the user is NOT requesting to book an interview, or if any required field is missing, return null.

Conversation History:
{history_text}

Current Message: {query}

Return ONLY the JSON object or null:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        try:
            if content.lower() == "null":
                return None
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    def is_booking_confirmed(self, query: str, history: list) -> bool:
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-6:]])

        prompt = f"""Determine if the user is confirming a previously discussed interview booking.
Return "true" if the user explicitly confirms/agrees to book the interview.
Return "false" if the user is asking a question, modifying details, or not confirming.

Conversation History:
{history_text}

Current Message: {query}

Return ONLY true or false:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().lower() == "true"

    def save_booking(
        self,
        db: Session,
        user_id: UUID,
        booking_data: Dict[str, Any],
    ) -> InterviewBooking:
        booking = InterviewBooking(
            user_id=user_id,
            name=booking_data["name"],
            email=booking_data["email"],
            booking_date=booking_data["date"],
            booking_time=booking_data["time"],
        )
        db.add(booking)
        db.commit()
        db.refresh(booking)
        return booking

    async def process_booking(
        self,
        db: Session,
        user_id: UUID,
        query: str,
        history: list,
    ) -> Optional[Dict[str, Any]]:
        booking_info = self.extract_booking_info(query, history)

        if not booking_info:
            return None

        is_confirmed = self.is_booking_confirmed(query, history)

        if not is_confirmed:
            return {
                "status": "pending_confirmation",
                "details": booking_info,
            }

        booking = self.save_booking(db, user_id, booking_info)
        await send_booking_confirmation_email(
            to_email=booking.email,
            name=booking.name,
            date=booking.booking_date,
            time=booking.booking_time,
        )

        return {
            "status": "confirmed",
            "details": booking_info,
        }


_booking_service: BookingService | None = None


def get_booking_service() -> BookingService:
    global _booking_service
    if _booking_service is None:
        _booking_service = BookingService()
    return _booking_service