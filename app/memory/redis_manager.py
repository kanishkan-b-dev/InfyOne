"""
Redis session / conversation memory manager.
Handles CRUD for chat history with TTL-based expiry.
Supports Redis Cloud and local Redis configurations from .env.
"""
import json
import logging
from typing import Optional

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_SESSION_PREFIX = "chat:session:"


class RedisManager:
    """Async Redis wrapper for session / conversation state."""

    def __init__(self) -> None:
        self._client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """Create the async Redis connection pool using .env credentials."""
        if settings.REDIS_URL:
            self._client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("Connecting to Redis via REDIS_URL...")
        else:
            self._client = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD or None,
                db=settings.REDIS_DB,
                ssl=settings.REDIS_SSL,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info(
                "Connecting to Redis at %s:%d (db=%d, ssl=%s)...",
                settings.REDIS_HOST,
                settings.REDIS_PORT,
                settings.REDIS_DB,
                settings.REDIS_SSL,
            )

        await self._client.ping()
        logger.info("Redis connected successfully.")

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.aclose()
            logger.info("Redis connection closed.")

    async def ping(self) -> bool:
        """Health check — returns True if Redis responds."""
        try:
            if self._client:
                await self._client.ping()
                return True
        except Exception:
            pass
        return False

    def _key(self, session_id: str) -> str:
        return f"{_SESSION_PREFIX}{session_id}"

    async def get_history(self, session_id: str) -> list[dict]:
        """Return the conversation history for a session."""
        try:
            if not self._client:
                return []
            raw = await self._client.get(self._key(session_id))
            if raw:
                data = json.loads(raw)
                return data.get("messages", [])
        except (RedisError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load session %s: %s", session_id, exc)
        return []

    async def append_messages(
        self, session_id: str, user_msg: str, assistant_msg: str
    ) -> None:
        """Append a user+assistant exchange to session history."""
        try:
            if not self._client:
                return
            history = await self.get_history(session_id)
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg})

            # Trim to max allowed pairs (each pair = 2 entries)
            max_entries = settings.REDIS_MAX_HISTORY * 2
            if len(history) > max_entries:
                history = history[-max_entries:]

            payload = json.dumps({"messages": history})
            await self._client.setex(
                self._key(session_id), settings.REDIS_SESSION_TTL, payload
            )
        except (RedisError, json.JSONDecodeError) as exc:
            logger.error("Failed to save session %s: %s", session_id, exc)

    async def clear_session(self, session_id: str) -> None:
        """Delete a session from Redis."""
        try:
            if self._client:
                await self._client.delete(self._key(session_id))
        except RedisError as exc:
            logger.warning("Failed to clear session %s: %s", session_id, exc)


# Singleton instance used across the application
redis_manager = RedisManager()
