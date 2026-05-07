"""
Circuit Breaker Pattern.

Implements circuit breaker for external API resilience with
automatic failure detection and recovery.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar, Awaitable, Generic

import structlog

logger = structlog.get_logger()

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(Exception):
    """Raised when circuit is open and requests are rejected."""

    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit '{name}' is open. Retry after {retry_after:.1f}s"
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Failure threshold to open circuit
    failure_threshold: int = 5

    # Success threshold to close circuit (in half-open state)
    success_threshold: int = 2

    # Time to wait before testing (open -> half-open)
    reset_timeout_seconds: float = 30.0

    # Time window for counting failures
    failure_window_seconds: float = 60.0

    # Timeout for individual calls
    call_timeout_seconds: float = 10.0


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    state_changes: int = 0


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker for external service calls.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Service failing, calls rejected immediately
    - HALF_OPEN: Testing if service recovered

    Transitions:
    - CLOSED -> OPEN: When failure_threshold exceeded
    - OPEN -> HALF_OPEN: After reset_timeout
    - HALF_OPEN -> CLOSED: When success_threshold reached
    - HALF_OPEN -> OPEN: On any failure
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        fallback: Callable[[], Awaitable[T]] | None = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback

        self._state = CircuitState.CLOSED
        self._failure_times: list[float] = []
        self._consecutive_successes = 0
        self._opened_at: float | None = None
        self._stats = CircuitStats()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitStats:
        """Circuit statistics."""
        return self._stats

    async def call(
        self,
        func: Callable[[], Awaitable[T]],
        *,
        fallback: Callable[[], Awaitable[T]] | None = None,
    ) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to call
            fallback: Optional fallback if circuit is open

        Returns:
            Result of func or fallback

        Raises:
            CircuitOpenError: If circuit is open and no fallback
        """
        async with self._lock:
            self._stats.total_calls += 1

            # Check if circuit should transition to half-open
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    self._stats.rejected_calls += 1
                    retry_after = self._get_retry_after()

                    # Try fallback
                    fb = fallback or self.fallback
                    if fb:
                        logger.info(
                            "circuit_fallback",
                            circuit=self.name,
                            retry_after=retry_after,
                        )
                        return await fb()

                    raise CircuitOpenError(self.name, retry_after)

        # Execute the call
        try:
            result = await asyncio.wait_for(
                func(),
                timeout=self.config.call_timeout_seconds,
            )
            await self._record_success()
            return result

        except asyncio.TimeoutError:
            await self._record_failure("timeout")
            raise

        except Exception as e:
            await self._record_failure(str(type(e).__name__))
            raise

    async def _record_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            now = time.time()
            self._stats.successful_calls += 1
            self._stats.last_success_time = now

            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    async def _record_failure(self, reason: str) -> None:
        """Record failed call."""
        async with self._lock:
            now = time.time()
            self._stats.failed_calls += 1
            self._stats.last_failure_time = now

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens circuit
                self._transition_to(CircuitState.OPEN)
                return

            if self._state == CircuitState.CLOSED:
                # Add to failure window
                self._failure_times.append(now)

                # Clean old failures
                cutoff = now - self.config.failure_window_seconds
                self._failure_times = [t for t in self._failure_times if t > cutoff]

                # Check threshold
                if len(self._failure_times) >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        "circuit_opened",
                        circuit=self.name,
                        failures=len(self._failure_times),
                        reason=reason,
                    )

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        old_state = self._state
        self._state = new_state
        self._stats.state_changes += 1

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._consecutive_successes = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._consecutive_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_times = []
            self._opened_at = None

        logger.info(
            "circuit_state_change",
            circuit=self.name,
            from_state=old_state.value,
            to_state=new_state.value,
        )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self._opened_at is None:
            return True
        elapsed = time.time() - self._opened_at
        return elapsed >= self.config.reset_timeout_seconds

    def _get_retry_after(self) -> float:
        """Get time until circuit may close."""
        if self._opened_at is None:
            return 0.0
        elapsed = time.time() - self._opened_at
        return max(0, self.config.reset_timeout_seconds - elapsed)

    async def force_open(self) -> None:
        """Manually open the circuit."""
        async with self._lock:
            self._transition_to(CircuitState.OPEN)

    async def force_close(self) -> None:
        """Manually close the circuit."""
        async with self._lock:
            self._transition_to(CircuitState.CLOSED)

    async def reset(self) -> None:
        """Reset circuit breaker state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_times = []
            self._consecutive_successes = 0
            self._opened_at = None
            self._stats = CircuitStats()


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    async def get_all_stats(self) -> dict[str, dict]:
        """Get statistics for all circuit breakers."""
        async with self._lock:
            return {
                name: {
                    "state": cb.state.value,
                    "total_calls": cb.stats.total_calls,
                    "successful_calls": cb.stats.successful_calls,
                    "failed_calls": cb.stats.failed_calls,
                    "rejected_calls": cb.stats.rejected_calls,
                    "state_changes": cb.stats.state_changes,
                }
                for name, cb in self._breakers.items()
            }


# Global registry
_registry = CircuitBreakerRegistry()


async def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get circuit breaker from global registry."""
    return await _registry.get_or_create(name, config)
