import asyncio
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.orm import Session

from app.config import settings
from app.models import InterviewBooking
from app.database import SessionLocal

logger = logging.getLogger(__name__)


def _get_today_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _send_booking_email_sync(to_email: str, name: str, date: str, time: str) -> None:
    """Synchronous email sending for use in agent tools (runs in worker thread)."""
    smtp_host = settings.SMTP_SERVER
    smtp_port = settings.SMTP_PORT
    smtp_user = settings.SMTP_USERNAME
    smtp_password = settings.SMTP_PASSWORD.get_secret_value()
    from_email = settings.SENDER_EMAIL

    logger.info(f"Sending confirmation email to {to_email} for booking on {date} at {time}")

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = f"Interview Confirmation - {date} at {time}"

    body = f"""Dear {name},

Your interview has been confirmed for {date} at {time}.

Best regards,
BookItRAG Team
"""
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)

    logger.info(f"Confirmation email sent successfully to {to_email}")

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = f"Interview Confirmation - {date} at {time}"

    body = f"""Dear {name},

Your interview has been confirmed for {date} at {time}.

Best regards,
BookItRAG Team
"""
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


@tool
def create_pending_booking(
    name: str,
    email: str,
    job_role: str,
    date: str,
    time: str,
) -> str:
    """Create a pending interview booking. Call this when you have all required details.
    
    Args:
        name: Full name of the candidate
        email: Email address of the candidate
        job_role: Job role/position applying for
        date: Interview date in YYYY-MM-DD format
        time: Interview time in HH:MM 24-hour format
    
    Returns:
        Confirmation message with booking details
    """
    db: Session = SessionLocal()
    try:
        booking = InterviewBooking(
            name=name,
            email=email,
            job_role=job_role,
            booking_date=date,
            booking_time=time,
            confirmation="pending",
        )
        db.add(booking)
        db.commit()
        db.refresh(booking)
        return f"Pending booking created for {name} ({email}) on {date} at {time} for {job_role}. Please confirm to finalize."
    except Exception as e:
        db.rollback()
        return f"Error creating booking: {str(e)}"
    finally:
        db.close()


@tool
def confirm_booking(email: str) -> str:
    """Confirm a pending interview booking by email. Call this when user explicitly confirms.
    
    Args:
        email: Email address of the candidate to find and confirm their pending booking
    
    Returns:
        Confirmation message with booking details
    """
    db: Session = SessionLocal()
    try:
        pending = db.query(InterviewBooking).filter(
            InterviewBooking.email == email,
            InterviewBooking.confirmation == "pending"
        ).order_by(InterviewBooking.created_at.desc()).first()
        
        if not pending:
            logger.warning(f"No pending booking found for {email}")
            return f"No pending booking found for {email}."
        
        pending.confirmation = "confirmed"
        db.commit()
        db.refresh(pending)
        
        logger.info(f"Booking confirmed for {pending.name} ({pending.email}), sending confirmation email")
        
        _send_booking_email_sync(
            to_email=pending.email,
            name=pending.name,
            date=pending.booking_date,
            time=pending.booking_time,
        )
        
        logger.info(f"Booking confirmed and email sent successfully for {pending.email}")
        return f"Booking confirmed for {pending.name} ({pending.email}) on {pending.booking_date} at {pending.booking_time} for {pending.job_role}. Confirmation email sent."
    except Exception as e:
        db.rollback()
        logger.exception(f"Error confirming booking for {email}: {e}")
        raise
    finally:
        db.close()


def _build_system_prompt(rag_context: str, today: str) -> str:
    return f"""You are a helpful AI assistant for BookItRAG that answers questions about job postings and handles interview bookings.

Today's date is {today}. Use it to resolve relative dates like "coming sunday", "tomorrow", "next Friday", "in 3 days" into YYYY-MM-DD format. Use 24-hour HH:MM format for times.

CONTEXT FROM JOB POSTINGS:
{rag_context}

INSTRUCTIONS:
1. Answer questions about job postings using ONLY the provided context. If context doesn't contain enough information, say so. Cite sources using [Source: filename, Page: X] format.

2. For interview bookings, you have two tools:
   - create_pending_booking: Creates a pending booking when you have ALL of: name, email, job_role, date, time
   - confirm_booking: Confirms a pending booking when user explicitly says "yes", "confirm", "book it", etc.

3. BOOKING FLOW:
   - Extract details from conversation. Ask for missing fields (name, email, job_role, date, time).
   - Once you have ALL 5 fields, call create_pending_booking.
   - After creating pending, show the details and ask: "Please confirm to book this interview."
   - ONLY call confirm_booking when user explicitly confirms (yes, confirm, book it, etc.)
   - If user modifies details or says no, update and call create_pending_booking again.

4. Be concise and helpful. Always respond to the user - never return empty responses."""


def _history_to_messages(history: List[Dict[str, Any]]) -> List:
    """Convert Redis history format to LangChain messages."""
    messages = []
    for msg in history[-12:]:  # Last 6 turns = 12 messages
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


class AgentService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY.get_secret_value(),
            temperature=0.3,
        )
        self._agent = create_agent(
            model=self.llm,
            tools=[create_pending_booking, confirm_booking],
            system_prompt="",  # Will be set dynamically per request
        )

    async def run(
        self,
        session_id: str,
        message: str,
        history: List[Dict[str, Any]],
        rag_context: str,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Run the agent with pre-fetched RAG context.
        
        Returns:
            (answer_text, sources, booking_status_dict)
        """
        today = _get_today_str()
        system_prompt = _build_system_prompt(rag_context, today)
        
        # Create fresh agent with updated system prompt (create_agent doesn't support dynamic prompt)
        agent = create_agent(
            model=self.llm,
            tools=[create_pending_booking, confirm_booking],
            system_prompt=system_prompt,
        )
        
        lc_messages = _history_to_messages(history)
        lc_messages.append(HumanMessage(content=message))
        
        # Run sync agent in threadpool
        result = await asyncio.to_thread(agent.invoke, {"messages": lc_messages})
        
        # Parse result
        final_message = result["messages"][-1]
        answer = final_message.text if hasattr(final_message, 'text') else str(final_message.content)
        
        # Extract booking status from tool calls
        booking_status = {"status": "none", "details": None}
        for msg in result["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "create_pending_booking":
                        args = tc.get("args", {})
                        booking_status = {
                            "status": "pending_confirmation",
                            "details": {
                                "name": args.get("name"),
                                "email": args.get("email"),
                                "date": args.get("date"),
                                "time": args.get("time"),
                                "job_role": args.get("job_role"),
                                "confirmation": "pending",
                            }
                        }
                    elif tc["name"] == "confirm_booking":
                        booking_status = {
                            "status": "confirmed",
                            "details": {
                                "name": None,  # Will be filled from context
                                "email": tc.get("args", {}).get("email"),
                                "date": None,
                                "time": None,
                                "job_role": None,
                                "confirmation": "confirmed",
                            }
                        }
        
        # Extract sources from rag_context (passed separately)
        sources = []
        for line in rag_context.split("\n\n---\n\n"):
            if line.startswith("[Source: "):
                try:
                    source_part = line.split("]")[0] + "]"
                    source = source_part.replace("[Source: ", "").replace("]", "")
                    parts = source.split(", ")
                    file_name = parts[0] if len(parts) > 0 else "unknown"
                    page = int(parts[1].replace("Page: ", "")) if len(parts) > 1 else 0
                    ingested_at = parts[2].replace("Ingested: ", "") if len(parts) > 2 else ""
                    sources.append({
                        "file_name": file_name,
                        "page": page,
                        "ingested_at": ingested_at if ingested_at else None,
                    })
                except (IndexError, ValueError):
                    continue
        
        return answer, sources, booking_status


_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service