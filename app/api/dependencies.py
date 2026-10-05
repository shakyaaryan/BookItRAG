from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session
import redis

from app.database import SessionLocal
from app.config import settings


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis_client() -> redis.Redis:
    return redis.from_url(settings.REDIS_URL, decode_responses=True)