import aiosmtplib
from email.message import EmailMessage
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
        password=settings.SMTP_PASSWORD,
        start_tls=True,
    )