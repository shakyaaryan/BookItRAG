import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy.orm import Session
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from app.config import settings
from app.models import InterviewBooking
from app.utils.email import send_booking_confirmation_email
from app.utils.llm_text import extract_llm_text


class BookingService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY.get_secret_value(),
            temperature=0.1,
        )

    def _today_str(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d")

    def extract_booking_info(self, query: str, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract booking fields. Returns partial info with missing list, or None if not a booking request."""
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-6:]])
        today = self._today_str()

        prompt = f"""Extract interview booking information from the user's message.

Today's date is {today}. Use it to resolve relative dates like "coming sunday", "tomorrow", "next Friday", "in 3 days" into YYYY-MM-DD format. Use 24-hour HH:MM format for times.

Return a JSON object with these fields (use null for missing):
- name: Full name of the person
- email: Email address
- date: Date of the interview (YYYY-MM-DD)
- time: Time of the interview (HH:MM, 24-hour)
- job_role: The job role/position the user is applying for
- missing: list of field names that are missing or could not be extracted

If the user is NOT requesting to book an interview, return null.

Conversation History:
{history_text}

Current Message: {query}

Return ONLY the JSON object or null:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = extract_llm_text(response)

        try:
            if content.lower() == "null":
                return None
            data = json.loads(content)
            # Normalize missing list
            missing = data.get("missing", [])
            if not isinstance(missing, list):
                missing = []
            # Ensure all expected fields exist
            for field in ["name", "email", "date", "time", "job_role"]:
                if field not in data:
                    data[field] = None
            data["missing"] = missing
            return data
        except json.JSONDecodeError:
            return None

    def is_booking_confirmed(self, query: str, history: List[Dict[str, Any]]) -> bool:
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-6:]])

        prompt = f"""Determine if the user is confirming a previously discussed interview booking.
Return "true" if the user explicitly confirms/agrees to book the interview.
Return "false" if the user is asking a question, modifying details, or not confirming.

Conversation History:
{history_text}

Current Message: {query}

Return ONLY true or false:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = extract_llm_text(response)
        return content.lower() == "true"

    def save_booking(
        self,
        db: Session,
        booking_data: Dict[str, Any],
        confirmation: str = "pending",
    ) -> InterviewBooking:
        booking = InterviewBooking(
            name=booking_data.get("name", ""),
            email=booking_data.get("email", ""),
            booking_date=booking_data.get("date", ""),
            booking_time=booking_data.get("time", ""),
            job_role=booking_data.get("job_role", ""),
            confirmation=confirmation,
        )
        db.add(booking)
        db.commit()
        db.refresh(booking)
        return booking

    def find_pending_booking(self, db: Session, email: str) -> Optional[InterviewBooking]:
        return db.query(InterviewBooking).filter(
            InterviewBooking.email == email,
            InterviewBooking.confirmation == "pending"
        ).order_by(InterviewBooking.created_at.desc()).first()

    async def process_booking(
        self,
        db: Session,
        query: str,
        history: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        booking_info = self.extract_booking_info(query, history)

        if not booking_info:
            return None

        missing = booking_info.get("missing", [])
        has_missing = len(missing) > 0

        # If info is incomplete, ask for the missing fields - don't save yet
        if has_missing:
            # Build a friendly prompt for the missing fields
            missing_prompts = {
                "name": "your full name",
                "email": "your email address",
                "date": "the interview date (e.g., 2026-10-12 or 'this Sunday')",
                "time": "the interview time (e.g., 14:30)",
                "job_role": "the job role you're applying for",
            }
            missing_desc = ", ".join([missing_prompts.get(f, f) for f in missing])
            return {
                "status": "needs_info",
                "details": {
                    "name": booking_info.get("name"),
                    "email": booking_info.get("email"),
                    "date": booking_info.get("date"),
                    "time": booking_info.get("time"),
                    "job_role": booking_info.get("job_role"),
                    "confirmation": "pending",
                },
                "missing": missing,
                "prompt": f"Got it. I still need {missing_desc}. Please provide those details.",
            }

        # All fields present - check if user is confirming
        is_confirmed = self.is_booking_confirmed(query, history)
        email = booking_info.get("email", "")

        pending = self.find_pending_booking(db, email)

        if is_confirmed and pending:
            pending.confirmation = "confirmed"
            db.commit()
            db.refresh(pending)
            await send_booking_confirmation_email(
                to_email=pending.email,
                name=pending.name,
                date=pending.booking_date,
                time=pending.booking_time,
            )
            return {
                "status": "confirmed",
                "details": {
                    "name": pending.name,
                    "email": pending.email,
                    "date": pending.booking_date,
                    "time": pending.booking_time,
                    "job_role": pending.job_role,
                    "confirmation": "confirmed",
                },
            }

        if is_confirmed and not pending:
            booking = self.save_booking(db, booking_info, confirmation="confirmed")
            await send_booking_confirmation_email(
                to_email=booking.email,
                name=booking.name,
                date=booking.booking_date,
                time=booking.booking_time,
            )
            return {
                "status": "confirmed",
                "details": {
                    "name": booking.name,
                    "email": booking.email,
                    "date": booking.booking_date,
                    "time": booking.booking_time,
                    "job_role": booking.job_role,
                    "confirmation": "confirmed",
                },
            }

        # Not confirmed yet, save as pending
        if pending:
            return {
                "status": "pending_confirmation",
                "details": {
                    "name": pending.name,
                    "email": pending.email,
                    "date": pending.booking_date,
                    "time": pending.booking_time,
                    "job_role": pending.job_role,
                    "confirmation": "pending",
                },
            }

        booking = self.save_booking(db, booking_info, confirmation="pending")
        return {
            "status": "pending_confirmation",
            "details": {
                "name": booking.name,
                "email": booking.email,
                "date": booking.booking_date,
                "time": booking.booking_time,
                "job_role": booking.job_role,
                "confirmation": "pending",
            },
        }


_booking_service: BookingService | None = None


def get_booking_service() -> BookingService:
    global _booking_service
    if _booking_service is None:
        _booking_service = BookingService()
    return _booking_service