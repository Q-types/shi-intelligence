"""
Infrastructure Layer for SHI.

Implements:
- Rate limiting (sliding window, Redis-backed)
- Circuit breaker pattern for external APIs
- Retry with exponential backoff
- Redis caching layer
- Monitoring and health checks
"""

from .rate_limit import (
    RateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
    SlidingWindowLimiter,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitOpenError,
)
from .retry import (
    RetryPolicy,
    RetryConfig,
    retry_with_backoff,
    ExponentialBackoff,
)
from .cache import (
    CacheClient,
    CacheConfig,
    cached,
    cache_aside,
)
from .monitoring import (
    MetricsCollector,
    HealthChecker,
    HealthStatus,
    RequestTimer,
)

__all__ = [
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitExceeded",
    "SlidingWindowLimiter",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitOpenError",
    # Retry
    "RetryPolicy",
    "RetryConfig",
    "retry_with_backoff",
    "ExponentialBackoff",
    # Cache
    "CacheClient",
    "CacheConfig",
    "cached",
    "cache_aside",
    # Monitoring
    "MetricsCollector",
    "HealthChecker",
    "HealthStatus",
    "RequestTimer",
]
