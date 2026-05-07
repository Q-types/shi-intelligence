"""
Cache Layer.

Redis-backed caching for expensive computations with
configurable TTL and invalidation strategies.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import (
    Callable,
    TypeVar,
    Awaitable,
    Any,
    Generic,
)

import structlog

logger = structlog.get_logger()

T = TypeVar("T")


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""

    # Default TTL in seconds
    default_ttl: int = 300  # 5 minutes

    # TTL by data type
    ttl_by_type: dict[str, int] = field(default_factory=lambda: {
        "holder_snapshot": 60,  # 1 minute - frequently changing
        "liquidity": 30,  # 30 seconds - real-time data
        "analysis_result": 300,  # 5 minutes - expensive computation
        "token_metadata": 3600,  # 1 hour - rarely changes
        "wallet_features": 600,  # 10 minutes
    })

    # Maximum cache size for in-memory fallback
    max_memory_items: int = 1000

    # Key prefix
    key_prefix: str = "shi:cache"

    # Enable compression for large values
    compress_threshold: int = 1024  # bytes


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


@dataclass
class CacheEntry(Generic[T]):
    """In-memory cache entry with metadata."""

    value: T
    created_at: float
    ttl: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.created_at + self.ttl


class MemoryCache:
    """
    In-memory LRU cache for fallback.

    Used when Redis is unavailable.
    """

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self._cache: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                return None

            entry.access_count += 1
            entry.last_accessed = time.time()
            self._stats.hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Set value in cache."""
        async with self._lock:
            # Evict if over capacity
            if len(self._cache) >= self.config.max_memory_items:
                await self._evict_lru()

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl or self.config.default_ttl,
            )
            self._stats.sets += 1

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.deletes += 1
                return True
            return False

    async def clear(self) -> int:
        """Clear all cache entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache = {}
            return count

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )
        del self._cache[lru_key]


class CacheClient:
    """
    Redis-backed cache client.

    Falls back to in-memory cache if Redis unavailable.
    """

    def __init__(
        self,
        config: CacheConfig | None = None,
        redis_url: str | None = None,
    ):
        self.config = config or CacheConfig()
        self.redis_url = redis_url
        self._redis = None
        self._fallback = MemoryCache(self.config)
        self._use_fallback = True
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        if self._use_fallback:
            return self._fallback.stats
        return self._stats

    async def _ensure_redis(self) -> bool:
        """Ensure Redis connection."""
        if self._redis is not None:
            return True

        if not self.redis_url:
            return False

        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            self._use_fallback = False
            logger.info("redis_cache_connected")
            return True
        except Exception as e:
            logger.warning("redis_cache_failed", error=str(e))
            self._use_fallback = True
            return False

    def _make_key(self, key: str) -> str:
        """Create full cache key."""
        return f"{self.config.key_prefix}:{key}"

    def _get_ttl(self, data_type: str | None) -> int:
        """Get TTL for data type."""
        if data_type and data_type in self.config.ttl_by_type:
            return self.config.ttl_by_type[data_type]
        return self.config.default_ttl

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if self._use_fallback or not await self._ensure_redis():
            return await self._fallback.get(key)

        try:
            full_key = self._make_key(key)
            value = await self._redis.get(full_key)

            if value is None:
                self._stats.misses += 1
                return None

            self._stats.hits += 1
            return json.loads(value)

        except Exception as e:
            self._stats.errors += 1
            logger.warning("cache_get_error", key=key, error=str(e))
            return await self._fallback.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        data_type: str | None = None,
    ) -> None:
        """Set value in cache."""
        effective_ttl = ttl or self._get_ttl(data_type)

        if self._use_fallback or not await self._ensure_redis():
            await self._fallback.set(key, value, effective_ttl)
            return

        try:
            full_key = self._make_key(key)
            serialized = json.dumps(value, default=str)
            await self._redis.setex(full_key, effective_ttl, serialized)
            self._stats.sets += 1

        except Exception as e:
            self._stats.errors += 1
            logger.warning("cache_set_error", key=key, error=str(e))
            await self._fallback.set(key, value, effective_ttl)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if self._use_fallback or not await self._ensure_redis():
            return await self._fallback.delete(key)

        try:
            full_key = self._make_key(key)
            result = await self._redis.delete(full_key)
            self._stats.deletes += 1
            return result > 0

        except Exception as e:
            self._stats.errors += 1
            logger.warning("cache_delete_error", key=key, error=str(e))
            return await self._fallback.delete(key)

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        if self._use_fallback:
            # In-memory doesn't support patterns efficiently
            return 0

        try:
            full_pattern = self._make_key(pattern)
            keys = []
            async for key in self._redis.scan_iter(full_pattern):
                keys.append(key)

            if keys:
                await self._redis.delete(*keys)
                self._stats.deletes += len(keys)

            return len(keys)

        except Exception as e:
            self._stats.errors += 1
            logger.warning("cache_invalidate_error", pattern=pattern, error=str(e))
            return 0

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()


def _make_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create cache key from function call."""
    # Create deterministic key from arguments
    key_parts = [func_name]

    for arg in args:
        key_parts.append(str(arg))

    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")

    key_string = ":".join(key_parts)

    # Hash if too long
    if len(key_string) > 200:
        return hashlib.sha256(key_string.encode()).hexdigest()

    return key_string


def cached(
    ttl: int | None = None,
    data_type: str | None = None,
    key_prefix: str | None = None,
):
    """
    Decorator for caching function results.

    Usage:
        @cached(ttl=300, data_type="analysis_result")
        async def analyze_token(mint: str):
            ...
    """
    # Shared cache client
    _cache_client: CacheClient | None = None

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            nonlocal _cache_client

            # Initialize cache client lazily
            if _cache_client is None:
                _cache_client = CacheClient()

            # Build cache key
            prefix = key_prefix or func.__name__
            cache_key = _make_cache_key(prefix, args, kwargs)

            # Try to get from cache
            cached_value = await _cache_client.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await _cache_client.set(
                cache_key,
                result,
                ttl=ttl,
                data_type=data_type,
            )

            return result

        return wrapper

    return decorator


async def cache_aside(
    cache: CacheClient,
    key: str,
    fetch_func: Callable[[], Awaitable[T]],
    ttl: int | None = None,
    data_type: str | None = None,
) -> T:
    """
    Cache-aside pattern implementation.

    1. Check cache for key
    2. If miss, call fetch_func
    3. Store result in cache
    4. Return result

    Args:
        cache: Cache client
        key: Cache key
        fetch_func: Function to fetch data on cache miss
        ttl: Optional TTL override
        data_type: Data type for TTL lookup

    Returns:
        Cached or fetched value
    """
    # Try cache first
    cached_value = await cache.get(key)
    if cached_value is not None:
        return cached_value

    # Fetch on miss
    value = await fetch_func()

    # Store in cache
    await cache.set(key, value, ttl=ttl, data_type=data_type)

    return value
