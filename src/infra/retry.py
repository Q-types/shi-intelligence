"""
Retry and Backoff System.

Implements exponential backoff with jitter for reliable
external API communication.
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from dataclasses import dataclass, field
from typing import (
    Callable,
    TypeVar,
    Awaitable,
    Type,
    Sequence,
)

import structlog

logger = structlog.get_logger()

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    # Maximum number of retry attempts
    max_retries: int = 3

    # Base delay between retries (seconds)
    base_delay: float = 1.0

    # Maximum delay between retries (seconds)
    max_delay: float = 60.0

    # Exponential backoff multiplier
    exponential_base: float = 2.0

    # Jitter range (0-1, percentage of delay)
    jitter: float = 0.1

    # Exceptions to retry on (empty = retry all)
    retryable_exceptions: tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )

    # Exceptions to never retry
    non_retryable_exceptions: tuple[Type[Exception], ...] = (
        ValueError,
        TypeError,
        KeyError,
    )


@dataclass
class RetryStats:
    """Statistics for retry operations."""

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_retries: int = 0
    total_delay_seconds: float = 0.0


class ExponentialBackoff:
    """
    Exponential backoff calculator with jitter.

    delay = min(max_delay, base_delay * (exponential_base ** attempt)) * (1 + random_jitter)
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Calculate exponential delay
        delay = self.config.base_delay * (
            self.config.exponential_base ** attempt
        )

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter
        jitter_range = delay * self.config.jitter
        jitter = random.uniform(-jitter_range, jitter_range)
        delay += jitter

        return max(0, delay)


class RetryPolicy:
    """
    Retry policy with configurable behavior.

    Supports:
    - Exponential backoff with jitter
    - Configurable retryable exceptions
    - Retry hooks for logging/metrics
    """

    def __init__(
        self,
        config: RetryConfig | None = None,
        on_retry: Callable[[int, Exception, float], Awaitable[None]] | None = None,
    ):
        self.config = config or RetryConfig()
        self.backoff = ExponentialBackoff(self.config)
        self.on_retry = on_retry
        self._stats = RetryStats()

    @property
    def stats(self) -> RetryStats:
        """Get retry statistics."""
        return self._stats

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if operation should be retried.

        Args:
            exception: The exception that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            True if should retry
        """
        # Check max retries
        if attempt >= self.config.max_retries:
            return False

        # Check non-retryable exceptions
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False

        # Check retryable exceptions (if specified)
        if self.config.retryable_exceptions:
            return isinstance(exception, self.config.retryable_exceptions)

        return True

    async def execute(
        self,
        func: Callable[[], Awaitable[T]],
        operation_name: str = "operation",
    ) -> T:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            operation_name: Name for logging

        Returns:
            Result of successful execution

        Raises:
            Last exception if all retries exhausted
        """
        last_exception: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            self._stats.total_attempts += 1

            try:
                result = await func()
                self._stats.successful_attempts += 1
                return result

            except Exception as e:
                last_exception = e
                self._stats.failed_attempts += 1

                if not self.should_retry(e, attempt):
                    logger.warning(
                        "retry_exhausted",
                        operation=operation_name,
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    raise

                # Calculate delay
                delay = self.backoff.get_delay(attempt)
                self._stats.total_retries += 1
                self._stats.total_delay_seconds += delay

                logger.info(
                    "retry_scheduled",
                    operation=operation_name,
                    attempt=attempt + 1,
                    max_attempts=self.config.max_retries + 1,
                    delay=delay,
                    error=str(e),
                )

                # Call retry hook if configured
                if self.on_retry:
                    await self.on_retry(attempt, e, delay)

                # Wait before retry
                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry logic error")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
    ),
):
    """
    Decorator for retry with exponential backoff.

    Usage:
        @retry_with_backoff(max_retries=3)
        async def fetch_data():
            ...
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            retryable_exceptions=retryable_exceptions,
        )
        policy = RetryPolicy(config)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await policy.execute(
                lambda: func(*args, **kwargs),
                operation_name=func.__name__,
            )

        return wrapper

    return decorator


@dataclass
class DeadLetterEntry:
    """Entry in the dead letter queue."""

    operation: str
    payload: dict
    error: str
    attempts: int
    created_at: float
    last_attempt_at: float


class DeadLetterQueue:
    """
    Queue for permanently failed operations.

    Stores failed operations for manual review or later processing.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queue: list[DeadLetterEntry] = []
        self._lock = asyncio.Lock()

    async def add(
        self,
        operation: str,
        payload: dict,
        error: str,
        attempts: int,
    ) -> None:
        """Add failed operation to queue."""
        async with self._lock:
            now = time.time()
            entry = DeadLetterEntry(
                operation=operation,
                payload=payload,
                error=error,
                attempts=attempts,
                created_at=now,
                last_attempt_at=now,
            )

            self._queue.append(entry)

            # Trim if over size
            if len(self._queue) > self.max_size:
                self._queue = self._queue[-self.max_size:]

            logger.warning(
                "dead_letter_added",
                operation=operation,
                attempts=attempts,
                error=error,
            )

    async def get_all(self) -> list[DeadLetterEntry]:
        """Get all entries in the queue."""
        async with self._lock:
            return list(self._queue)

    async def clear(self) -> int:
        """Clear the queue and return count of cleared entries."""
        async with self._lock:
            count = len(self._queue)
            self._queue = []
            return count

    async def pop(self, count: int = 1) -> list[DeadLetterEntry]:
        """Pop entries from the queue for reprocessing."""
        async with self._lock:
            entries = self._queue[:count]
            self._queue = self._queue[count:]
            return entries

    @property
    def size(self) -> int:
        """Current queue size."""
        return len(self._queue)
