"""
Query Cache for SHI.

Provides caching layer for repeated token queries.
Supports Redis for distributed caching or in-memory for development.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from typing import Any, TypeVar

import structlog

from ..core.config import settings

logger = structlog.get_logger()

T = TypeVar("T")


class QueryCache:
    """
    Cache layer for data queries.

    Uses Redis in production, in-memory dict for development.
    """

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or settings.redis_url
        self._redis: Any | None = None
        self._local_cache: dict[str, tuple[Any, datetime]] = {}
        self._use_redis = bool(self.redis_url and "redis://" in self.redis_url)

    async def _get_redis(self):
        """Lazy Redis connection."""
        if self._redis is None and self._use_redis:
            try:
                import redis.asyncio as redis

                self._redis = redis.from_url(self.redis_url)
                logger.info("redis_connected", url=self.redis_url)
            except Exception as e:
                logger.warning("redis_connection_failed", error=str(e))
                self._use_redis = False

        return self._redis

    async def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        # Try Redis first
        if self._use_redis:
            redis = await self._get_redis()
            if redis:
                try:
                    data = await redis.get(key)
                    if data:
                        return pickle.loads(data)
                except Exception as e:
                    logger.warning("redis_get_failed", key=key, error=str(e))

        # Fall back to local cache
        if key in self._local_cache:
            value, expires = self._local_cache[key]
            if datetime.now(timezone.utc) < expires:
                return value
            else:
                del self._local_cache[key]

        return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        # Try Redis
        if self._use_redis:
            redis = await self._get_redis()
            if redis:
                try:
                    await redis.setex(key, ttl, pickle.dumps(value))
                    return
                except Exception as e:
                    logger.warning("redis_set_failed", key=key, error=str(e))

        # Fall back to local cache
        expires = datetime.now(timezone.utc).replace(
            microsecond=0
        )  # Simplified expiry
        from datetime import timedelta

        expires = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        self._local_cache[key] = (value, expires)

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        if self._use_redis:
            redis = await self._get_redis()
            if redis:
                try:
                    await redis.delete(key)
                except Exception:
                    pass

        if key in self._local_cache:
            del self._local_cache[key]

    async def clear_prefix(self, prefix: str) -> None:
        """Clear all keys with given prefix."""
        if self._use_redis:
            redis = await self._get_redis()
            if redis:
                try:
                    keys = await redis.keys(f"{prefix}*")
                    if keys:
                        await redis.delete(*keys)
                except Exception:
                    pass

        # Local cache
        to_delete = [k for k in self._local_cache if k.startswith(prefix)]
        for k in to_delete:
            del self._local_cache[k]

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
