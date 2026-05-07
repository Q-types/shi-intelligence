"""
Rate Limiting System.

Implements sliding window rate limiting with Redis backend
for distributed deployment support.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable

import structlog

logger = structlog.get_logger()


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, user_id: str, limit: int, window_seconds: int, retry_after: float):
        self.user_id = user_id
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for {user_id}: {limit} requests per {window_seconds}s. "
            f"Retry after {retry_after:.1f}s"
        )


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Per-user limits
    user_requests_per_minute: int = 10
    user_requests_per_hour: int = 100

    # Global limits
    global_requests_per_minute: int = 100
    global_requests_per_hour: int = 1000

    # Window sizes in seconds
    minute_window: int = 60
    hour_window: int = 3600

    # Redis key prefix
    key_prefix: str = "shi:ratelimit"


@dataclass
class SlidingWindowState:
    """State for sliding window counter."""

    timestamps: list[float] = field(default_factory=list)
    window_seconds: int = 60
    max_requests: int = 10

    def is_allowed(self, current_time: float | None = None) -> tuple[bool, float]:
        """
        Check if request is allowed and return retry_after time if not.

        Returns:
            (allowed, retry_after_seconds)
        """
        now = current_time or time.time()
        cutoff = now - self.window_seconds

        # Remove expired timestamps
        self.timestamps = [ts for ts in self.timestamps if ts > cutoff]

        if len(self.timestamps) >= self.max_requests:
            # Calculate when oldest request will expire
            oldest = min(self.timestamps)
            retry_after = oldest + self.window_seconds - now
            return False, max(0, retry_after)

        return True, 0.0

    def record(self, current_time: float | None = None) -> None:
        """Record a new request."""
        now = current_time or time.time()
        self.timestamps.append(now)


class SlidingWindowLimiter:
    """
    In-memory sliding window rate limiter.

    For single-instance deployment or testing.
    """

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._user_minute_windows: dict[str, SlidingWindowState] = {}
        self._user_hour_windows: dict[str, SlidingWindowState] = {}
        self._global_minute = SlidingWindowState(
            window_seconds=self.config.minute_window,
            max_requests=self.config.global_requests_per_minute,
        )
        self._global_hour = SlidingWindowState(
            window_seconds=self.config.hour_window,
            max_requests=self.config.global_requests_per_hour,
        )
        self._lock = asyncio.Lock()

    def _get_user_windows(self, user_id: str) -> tuple[SlidingWindowState, SlidingWindowState]:
        """Get or create user rate limit windows."""
        if user_id not in self._user_minute_windows:
            self._user_minute_windows[user_id] = SlidingWindowState(
                window_seconds=self.config.minute_window,
                max_requests=self.config.user_requests_per_minute,
            )
        if user_id not in self._user_hour_windows:
            self._user_hour_windows[user_id] = SlidingWindowState(
                window_seconds=self.config.hour_window,
                max_requests=self.config.user_requests_per_hour,
            )
        return self._user_minute_windows[user_id], self._user_hour_windows[user_id]

    async def check_rate_limit(self, user_id: str) -> None:
        """
        Check if request is allowed for user.

        Raises:
            RateLimitExceeded: If any limit is exceeded
        """
        async with self._lock:
            now = time.time()

            # Check global limits first
            allowed, retry_after = self._global_minute.is_allowed(now)
            if not allowed:
                logger.warning(
                    "global_rate_limit_exceeded",
                    window="minute",
                    retry_after=retry_after,
                )
                raise RateLimitExceeded(
                    "global", self.config.global_requests_per_minute,
                    self.config.minute_window, retry_after
                )

            allowed, retry_after = self._global_hour.is_allowed(now)
            if not allowed:
                logger.warning(
                    "global_rate_limit_exceeded",
                    window="hour",
                    retry_after=retry_after,
                )
                raise RateLimitExceeded(
                    "global", self.config.global_requests_per_hour,
                    self.config.hour_window, retry_after
                )

            # Check user limits
            minute_window, hour_window = self._get_user_windows(user_id)

            allowed, retry_after = minute_window.is_allowed(now)
            if not allowed:
                logger.warning(
                    "user_rate_limit_exceeded",
                    user_id=user_id,
                    window="minute",
                    retry_after=retry_after,
                )
                raise RateLimitExceeded(
                    user_id, self.config.user_requests_per_minute,
                    self.config.minute_window, retry_after
                )

            allowed, retry_after = hour_window.is_allowed(now)
            if not allowed:
                logger.warning(
                    "user_rate_limit_exceeded",
                    user_id=user_id,
                    window="hour",
                    retry_after=retry_after,
                )
                raise RateLimitExceeded(
                    user_id, self.config.user_requests_per_hour,
                    self.config.hour_window, retry_after
                )

            # Record the request
            self._global_minute.record(now)
            self._global_hour.record(now)
            minute_window.record(now)
            hour_window.record(now)

    async def get_remaining(self, user_id: str) -> dict[str, int]:
        """Get remaining requests for user."""
        async with self._lock:
            minute_window, hour_window = self._get_user_windows(user_id)

            # Clean up expired timestamps
            now = time.time()
            minute_window.is_allowed(now)
            hour_window.is_allowed(now)

            return {
                "minute": max(0, minute_window.max_requests - len(minute_window.timestamps)),
                "hour": max(0, hour_window.max_requests - len(hour_window.timestamps)),
            }


class RateLimiter:
    """
    Redis-backed rate limiter for distributed deployment.

    Falls back to in-memory limiter if Redis unavailable.
    """

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        redis_url: str | None = None,
    ):
        self.config = config or RateLimitConfig()
        self.redis_url = redis_url
        self._redis = None
        self._fallback = SlidingWindowLimiter(self.config)
        self._use_fallback = True

    async def _ensure_redis(self) -> bool:
        """Ensure Redis connection is established."""
        if self._redis is not None:
            return True

        if not self.redis_url:
            return False

        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            self._use_fallback = False
            logger.info("redis_rate_limiter_connected")
            return True
        except Exception as e:
            logger.warning("redis_rate_limiter_failed", error=str(e))
            self._use_fallback = True
            return False

    async def check_rate_limit(self, user_id: str) -> None:
        """Check rate limit for user."""
        if self._use_fallback or not await self._ensure_redis():
            await self._fallback.check_rate_limit(user_id)
            return

        # Redis-based sliding window using sorted sets
        now = time.time()
        pipe = self._redis.pipeline()

        keys = [
            f"{self.config.key_prefix}:user:{user_id}:minute",
            f"{self.config.key_prefix}:user:{user_id}:hour",
            f"{self.config.key_prefix}:global:minute",
            f"{self.config.key_prefix}:global:hour",
        ]

        limits = [
            (self.config.user_requests_per_minute, self.config.minute_window),
            (self.config.user_requests_per_hour, self.config.hour_window),
            (self.config.global_requests_per_minute, self.config.minute_window),
            (self.config.global_requests_per_hour, self.config.hour_window),
        ]

        try:
            # Remove expired entries and count
            for key, (_, window) in zip(keys, limits):
                cutoff = now - window
                pipe.zremrangebyscore(key, "-inf", cutoff)
                pipe.zcard(key)

            results = await pipe.execute()

            # Check each limit
            for i, (key, (limit, window)) in enumerate(zip(keys, limits)):
                count = results[i * 2 + 1]  # Every other result is zcard
                if count >= limit:
                    # Find retry_after
                    oldest = await self._redis.zrange(key, 0, 0, withscores=True)
                    if oldest:
                        retry_after = oldest[0][1] + window - now
                    else:
                        retry_after = 1.0

                    raise RateLimitExceeded(
                        user_id if "user" in key else "global",
                        limit, window, max(0, retry_after)
                    )

            # Record the request
            pipe = self._redis.pipeline()
            for key, (_, window) in zip(keys, limits):
                pipe.zadd(key, {str(now): now})
                pipe.expire(key, window + 60)  # Extra buffer for cleanup
            await pipe.execute()

        except RateLimitExceeded:
            raise
        except Exception as e:
            logger.warning("redis_rate_limit_error", error=str(e))
            # Fall back to in-memory
            await self._fallback.check_rate_limit(user_id)

    async def get_remaining(self, user_id: str) -> dict[str, int]:
        """Get remaining requests for user."""
        if self._use_fallback or not await self._ensure_redis():
            return await self._fallback.get_remaining(user_id)

        try:
            now = time.time()
            minute_key = f"{self.config.key_prefix}:user:{user_id}:minute"
            hour_key = f"{self.config.key_prefix}:user:{user_id}:hour"

            pipe = self._redis.pipeline()
            pipe.zcount(minute_key, now - self.config.minute_window, "+inf")
            pipe.zcount(hour_key, now - self.config.hour_window, "+inf")
            results = await pipe.execute()

            return {
                "minute": max(0, self.config.user_requests_per_minute - results[0]),
                "hour": max(0, self.config.user_requests_per_hour - results[1]),
            }
        except Exception:
            return await self._fallback.get_remaining(user_id)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
