import json
from typing import List, Dict, Any
from uuid import UUID
import redis
from app.config import settings


class MemoryService:
    def __init__(self):
        self.client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.ttl = settings.REDIS_SESSION_TTL

    def _get_key(self, user_id: UUID, session_id: str) -> str:
        return f"chat_history:{user_id}:{session_id}"

    def get_history(self, user_id: UUID, session_id: str) -> List[Dict[str, Any]]:
        key = self._get_key(user_id, session_id)
        data = self.client.lrange(key, 0, -1)
        return [json.loads(item) for item in data]

    def add_message(
        self,
        user_id: UUID,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        key = self._get_key(user_id, session_id)
        message = {"role": role, "content": content}
        self.client.rpush(key, json.dumps(message))
        self.client.expire(key, self.ttl)

    def clear_history(self, user_id: UUID, session_id: str) -> None:
        key = self._get_key(user_id, session_id)
        self.client.delete(key)


_memory_service: MemoryService | None = None


def get_memory_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service