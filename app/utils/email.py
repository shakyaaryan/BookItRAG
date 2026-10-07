import aiosmtplib
import smtplib
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.config import settings


async def send_booking_confirmation_email(
    to_email: str,
    name: str,
    date: str,
    time: str,
) -> None:
    message = EmailMessage()
    message["From"] = settings.SENDER_EMAIL
    message["To"] = to_email
    message["Subject"] = "Interview Booking Confirmation - BookItRAG"

    message.set_content(f"""
Dear {name},

Your interview has been successfully booked!

Booking Details:
- Date: {date}
- Time: {time}

If you need to reschedule or have any questions, please contact us.

Best regards,
The BookItRAG Team
""")

    await aiosmtplib.send(
        message,
        hostname=settings.SMTP_SERVER,
        port=settings.SMTP_PORT,
        username=settings.SMTP_USERNAME,
        password=settings.SMTP_PASSWORD.get_secret_value(),
        start_tls=True,
    )


def send_booking_confirmation_email_sync(
    to_email: str,
    name: str,
    date: str,
    time: str,
) -> None:
    """Synchronous email sending for use in agent tools (runs in worker thread)."""
    smtp_host = settings.SMTP_SERVER
    smtp_port = settings.SMTP_PORT
    smtp_user = settings.SMTP_USERNAME
    smtp_password = settings.SMTP_PASSWORD.get_secret_value()
    from_email = settings.SENDER_EMAIL

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = "Interview Booking Confirmation - BookItRAG"

    body = f"""
Dear {name},

Your interview has been successfully booked!

Booking Details:
- Date: {date}
- Time: {time}

If you need to reschedule or have any questions, please contact us.

Best regards,
The BookItRAG Team
"""
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)