"""
Monitoring and Health Checks.

Implements structured logging, Prometheus metrics,
health endpoints, and alerting infrastructure.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Awaitable, Any

import structlog

logger = structlog.get_logger()


class HealthState(Enum):
    """Health check states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthStatus:
    """Health status for a component."""

    name: str
    state: HealthState
    message: str | None = None
    latency_ms: float | None = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health status."""

    state: HealthState
    components: list[HealthStatus]
    checked_at: datetime
    version: str = "1.0.0"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON response."""
        return {
            "status": self.state.value,
            "version": self.version,
            "checked_at": self.checked_at.isoformat(),
            "components": [
                {
                    "name": c.name,
                    "status": c.state.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                    "metadata": c.metadata,
                }
                for c in self.components
            ],
        }


class HealthChecker:
    """
    Health check manager for all system components.

    Supports:
    - Component health checks
    - Dependency health (Redis, RPC, etc.)
    - Aggregated system health
    """

    def __init__(self):
        self._checks: dict[str, Callable[[], Awaitable[HealthStatus]]] = {}

    def register(
        self,
        name: str,
        check: Callable[[], Awaitable[HealthStatus]],
    ) -> None:
        """Register a health check."""
        self._checks[name] = check

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        self._checks.pop(name, None)

    async def check_component(self, name: str) -> HealthStatus | None:
        """Run health check for specific component."""
        check = self._checks.get(name)
        if check is None:
            return None

        try:
            start = time.time()
            status = await asyncio.wait_for(check(), timeout=5.0)
            status.latency_ms = (time.time() - start) * 1000
            return status
        except asyncio.TimeoutError:
            return HealthStatus(
                name=name,
                state=HealthState.UNHEALTHY,
                message="Health check timed out",
            )
        except Exception as e:
            return HealthStatus(
                name=name,
                state=HealthState.UNHEALTHY,
                message=f"Health check failed: {e}",
            )

    async def check_all(self) -> SystemHealth:
        """Run all health checks and return system health."""
        components = []

        for name in self._checks:
            status = await self.check_component(name)
            if status:
                components.append(status)

        # Determine overall health
        if any(c.state == HealthState.UNHEALTHY for c in components):
            overall = HealthState.UNHEALTHY
        elif any(c.state == HealthState.DEGRADED for c in components):
            overall = HealthState.DEGRADED
        else:
            overall = HealthState.HEALTHY

        return SystemHealth(
            state=overall,
            components=components,
            checked_at=datetime.now(timezone.utc),
        )


# Standard health check factories
async def redis_health_check(redis_url: str) -> HealthStatus:
    """Health check for Redis connection."""
    try:
        import redis.asyncio as redis
        client = redis.from_url(redis_url)
        start = time.time()
        await client.ping()
        latency = (time.time() - start) * 1000
        await client.close()

        return HealthStatus(
            name="redis",
            state=HealthState.HEALTHY,
            latency_ms=latency,
        )
    except Exception as e:
        return HealthStatus(
            name="redis",
            state=HealthState.UNHEALTHY,
            message=str(e),
        )


async def database_health_check(db_url: str) -> HealthStatus:
    """Health check for database connection."""
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text

        engine = create_async_engine(db_url)
        start = time.time()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        latency = (time.time() - start) * 1000
        await engine.dispose()

        return HealthStatus(
            name="database",
            state=HealthState.HEALTHY,
            latency_ms=latency,
        )
    except Exception as e:
        return HealthStatus(
            name="database",
            state=HealthState.UNHEALTHY,
            message=str(e),
        )


async def rpc_health_check(rpc_url: str) -> HealthStatus:
    """Health check for Solana RPC."""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            start = time.time()
            response = await client.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getHealth",
                },
                timeout=5.0,
            )
            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                if data.get("result") == "ok":
                    return HealthStatus(
                        name="solana_rpc",
                        state=HealthState.HEALTHY,
                        latency_ms=latency,
                    )

            return HealthStatus(
                name="solana_rpc",
                state=HealthState.DEGRADED,
                message="RPC returned non-ok status",
                latency_ms=latency,
            )
    except Exception as e:
        return HealthStatus(
            name="solana_rpc",
            state=HealthState.UNHEALTHY,
            message=str(e),
        )


@dataclass
class MetricValue:
    """A metric value with timestamp."""

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Metrics collector for Prometheus-style metrics.

    Supports:
    - Counters
    - Gauges
    - Histograms
    - Request timing
    """

    def __init__(self):
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

        # Histogram bucket boundaries
        self._histogram_buckets = [
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
        ]

    def _make_key(self, name: str, labels: dict[str, str] | None = None) -> str:
        """Create metric key with labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    async def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter."""
        async with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] = self._counters.get(key, 0) + value

    async def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge value."""
        async with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value

    async def observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Add observation to histogram."""
        async with self._lock:
            key = self._make_key(name, labels)
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)

            # Keep only last 10000 observations
            if len(self._histograms[key]) > 10000:
                self._histograms[key] = self._histograms[key][-10000:]

    async def get_metrics(self) -> dict[str, Any]:
        """Get all metrics in Prometheus-compatible format."""
        async with self._lock:
            metrics = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
            }

            # Compute histogram statistics
            for key, values in self._histograms.items():
                if values:
                    import numpy as np
                    arr = np.array(values)
                    metrics["histograms"][key] = {
                        "count": len(values),
                        "sum": float(np.sum(arr)),
                        "mean": float(np.mean(arr)),
                        "p50": float(np.percentile(arr, 50)),
                        "p90": float(np.percentile(arr, 90)),
                        "p99": float(np.percentile(arr, 99)),
                    }

            return metrics

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        for key, value in self._counters.items():
            lines.append(f"shi_{key} {value}")

        for key, value in self._gauges.items():
            lines.append(f"shi_{key} {value}")

        # Histograms require more complex formatting
        for key, values in self._histograms.items():
            if values:
                import numpy as np
                arr = np.array(values)
                lines.append(f"shi_{key}_count {len(values)}")
                lines.append(f"shi_{key}_sum {np.sum(arr)}")
                for bucket in self._histogram_buckets:
                    count = np.sum(arr <= bucket)
                    lines.append(f'shi_{key}_bucket{{le="{bucket}"}} {count}')
                lines.append(f'shi_{key}_bucket{{le="+Inf"}} {len(values)}')

        return "\n".join(lines)


class RequestTimer:
    """Context manager for timing requests."""

    def __init__(
        self,
        metrics: MetricsCollector,
        name: str,
        labels: dict[str, str] | None = None,
    ):
        self.metrics = metrics
        self.name = name
        self.labels = labels or {}
        self._start: float | None = None

    async def __aenter__(self) -> "RequestTimer":
        self._start = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start is not None:
            duration = time.time() - self._start

            # Record duration
            await self.metrics.observe(
                f"{self.name}_duration_seconds",
                duration,
                self.labels,
            )

            # Record success/failure
            status = "error" if exc_type else "success"
            await self.metrics.increment(
                f"{self.name}_total",
                labels={**self.labels, "status": status},
            )


@asynccontextmanager
async def timed_operation(
    metrics: MetricsCollector,
    name: str,
    labels: dict[str, str] | None = None,
):
    """Async context manager for timing operations."""
    timer = RequestTimer(metrics, name, labels)
    async with timer:
        yield timer


# Global metrics collector
_metrics = MetricsCollector()
_health_checker = HealthChecker()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    return _metrics


def get_health_checker() -> HealthChecker:
    """Get global health checker."""
    return _health_checker


# Convenience functions
async def record_request(
    endpoint: str,
    method: str = "GET",
    status_code: int = 200,
    duration_seconds: float = 0.0,
) -> None:
    """Record an HTTP request."""
    labels = {
        "endpoint": endpoint,
        "method": method,
        "status": str(status_code),
    }
    await _metrics.increment("http_requests_total", labels=labels)
    await _metrics.observe("http_request_duration_seconds", duration_seconds, labels=labels)


async def record_analysis(
    token: str,
    success: bool,
    duration_seconds: float,
) -> None:
    """Record a token analysis."""
    labels = {"status": "success" if success else "error"}
    await _metrics.increment("analysis_total", labels=labels)
    await _metrics.observe("analysis_duration_seconds", duration_seconds, labels=labels)
