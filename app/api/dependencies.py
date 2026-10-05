from typing import Generator
from uuid import UUID
from fastapi import Depends, HTTPException, Header
from sqlalchemy.orm import Session
import redis

from app.database import SessionLocal
from app.config import settings
from app.models import User


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis_client() -> redis.Redis:
    return redis.from_url(settings.REDIS_URL, decode_responses=True)


async def get_current_user(
    user_id: UUID = Header(..., alias="X-User-ID"),
    db: Session = Depends(get_db),
) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user