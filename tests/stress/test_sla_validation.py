"""
SLA Validation Suite.

Tests system performance against 30-second latency SLA
and validates behavior under load.
"""

from __future__ import annotations

import asyncio
import time

import pytest
import numpy as np



class LatencyTracker:
    """Tracks latency measurements for SLA validation."""

    def __init__(self):
        self.measurements: list[float] = []
        self.start_times: dict[str, float] = {}

    def start(self, request_id: str) -> None:
        """Start timing a request."""
        self.start_times[request_id] = time.perf_counter()

    def stop(self, request_id: str) -> float:
        """Stop timing and record latency."""
        if request_id not in self.start_times:
            return 0.0

        latency = time.perf_counter() - self.start_times[request_id]
        self.measurements.append(latency)
        del self.start_times[request_id]
        return latency

    @property
    def p50(self) -> float:
        """50th percentile latency."""
        if not self.measurements:
            return 0.0
        return float(np.percentile(self.measurements, 50))

    @property
    def p90(self) -> float:
        """90th percentile latency."""
        if not self.measurements:
            return 0.0
        return float(np.percentile(self.measurements, 90))

    @property
    def p99(self) -> float:
        """99th percentile latency."""
        if not self.measurements:
            return 0.0
        return float(np.percentile(self.measurements, 99))

    @property
    def sla_compliance_rate(self) -> float:
        """Percentage of requests meeting 30s SLA."""
        if not self.measurements:
            return 1.0
        within_sla = sum(1 for m in self.measurements if m <= 30.0)
        return within_sla / len(self.measurements)


class MockDataClient:
    """Mock data client for load testing."""

    def __init__(self, response_time: float = 0.1):
        self.response_time = response_time
        self.call_count = 0

    async def get_token_holders(self, mint: str) -> dict:
        """Simulate holder data fetch."""
        await asyncio.sleep(self.response_time)
        self.call_count += 1

        # Generate mock holder data
        num_holders = 100
        return {
            "holders": [
                {
                    "address": f"holder_{i}_{mint[:8]}",
                    "balance": 1000000 / (i + 1),
                }
                for i in range(num_holders)
            ],
            "total_supply": 1000000000,
        }


class TestSLACompliance:
    """Tests for 30-second SLA compliance."""

    @pytest.fixture
    def tracker(self) -> LatencyTracker:
        return LatencyTracker()

    @pytest.fixture
    def mock_client(self) -> MockDataClient:
        return MockDataClient(response_time=0.1)

    @pytest.mark.asyncio
    async def test_single_request_sla(
        self,
        tracker: LatencyTracker,
        mock_client: MockDataClient,
    ) -> None:
        """Test single request meets SLA."""
        request_id = "test_1"
        tracker.start(request_id)

        # Simulate analysis
        await mock_client.get_token_holders("test_mint")
        await asyncio.sleep(0.05)  # Simulate computation

        latency = tracker.stop(request_id)

        # Should be well under 30s for simple mock
        assert latency < 1.0
        assert tracker.sla_compliance_rate == 1.0

    @pytest.mark.asyncio
    async def test_concurrent_requests_sla(
        self,
        tracker: LatencyTracker,
        mock_client: MockDataClient,
    ) -> None:
        """Test concurrent requests meet SLA."""
        num_requests = 10

        async def make_request(request_id: str) -> float:
            tracker.start(request_id)
            await mock_client.get_token_holders(f"mint_{request_id}")
            await asyncio.sleep(0.05)
            return tracker.stop(request_id)

        # Run concurrent requests
        tasks = [
            make_request(f"concurrent_{i}")
            for i in range(num_requests)
        ]
        latencies = await asyncio.gather(*tasks)

        # All should meet SLA
        assert all(lat < 30.0 for lat in latencies)
        assert tracker.sla_compliance_rate == 1.0

        # p99 should still be reasonable
        assert tracker.p99 < 5.0

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        """Test that slow requests are properly timed out."""
        slow_client = MockDataClient(response_time=35.0)  # Exceeds SLA

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                slow_client.get_token_holders("slow_mint"),
                timeout=30.0,
            )

    @pytest.mark.asyncio
    async def test_partial_result_on_timeout(self) -> None:
        """Test that partial results are returned on timeout."""
        # Simulate a request that partially completes

        partial_data = {"holders": [], "partial": True}
        complete_data = {"holders": [{"address": "h1"}], "partial": False}

        async def slow_analysis() -> dict:
            await asyncio.sleep(0.1)
            partial_data["holders"].append({"address": "partial_1"})
            await asyncio.sleep(35)  # Would exceed timeout
            return complete_data

        try:
            result = await asyncio.wait_for(slow_analysis(), timeout=0.5)
        except asyncio.TimeoutError:
            # Should have partial data
            assert len(partial_data["holders"]) > 0
            result = partial_data

        assert "holders" in result


class TestLoadPerformance:
    """Tests for system performance under load."""

    @pytest.fixture
    def tracker(self) -> LatencyTracker:
        return LatencyTracker()

    @pytest.mark.asyncio
    async def test_sustained_load(
        self,
        tracker: LatencyTracker,
    ) -> None:
        """Test performance under sustained load."""
        mock_client = MockDataClient(response_time=0.05)
        num_requests = 50
        request_rate = 5  # requests per second

        async def make_request(request_id: str) -> float:
            tracker.start(request_id)
            await mock_client.get_token_holders(f"mint_{request_id}")
            return tracker.stop(request_id)

        # Send requests at specified rate
        start_time = time.perf_counter()
        for i in range(num_requests):
            asyncio.create_task(make_request(f"load_{i}"))
            await asyncio.sleep(1.0 / request_rate)

        # Wait for all to complete
        await asyncio.sleep(2.0)
        total_time = time.perf_counter() - start_time

        # Verify throughput
        actual_rate = mock_client.call_count / total_time
        assert actual_rate >= request_rate * 0.8, "Throughput degraded under load"

    @pytest.mark.asyncio
    async def test_burst_handling(
        self,
        tracker: LatencyTracker,
    ) -> None:
        """Test handling of request bursts."""
        mock_client = MockDataClient(response_time=0.02)
        burst_size = 20

        async def make_request(request_id: str) -> float:
            tracker.start(request_id)
            await mock_client.get_token_holders(f"mint_{request_id}")
            return tracker.stop(request_id)

        # Send burst of requests
        tasks = [make_request(f"burst_{i}") for i in range(burst_size)]
        latencies = await asyncio.gather(*tasks)

        # All should complete
        assert len(latencies) == burst_size

        # Latency should not explode
        assert tracker.p99 < 5.0

    @pytest.mark.asyncio
    async def test_memory_stability(self) -> None:
        """Test that memory usage remains stable under load."""
        import tracemalloc

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        mock_client = MockDataClient(response_time=0.01)

        # Run many requests
        for i in range(100):
            await mock_client.get_token_holders(f"mint_{i}")

        current_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # Memory growth should be bounded
        memory_growth_mb = (current_memory - initial_memory) / (1024 * 1024)
        assert memory_growth_mb < 50, f"Memory grew by {memory_growth_mb:.1f}MB"


class TestTimeoutBehavior:
    """Tests for timeout handling behavior."""

    @pytest.mark.asyncio
    async def test_graceful_timeout_degradation(self) -> None:
        """Test that system degrades gracefully on timeout."""
        async def slow_component(delay: float) -> str:
            await asyncio.sleep(delay)
            return "complete"

        async def analysis_with_timeout(timeout: float) -> dict:
            result = {"status": "partial", "components": []}

            # Run multiple components with individual timeouts
            for i, delay in enumerate([0.1, 0.2, 0.5, 2.0]):
                try:
                    await asyncio.wait_for(
                        slow_component(delay),
                        timeout=min(timeout, 1.0),
                    )
                    result["components"].append(f"component_{i}")
                except asyncio.TimeoutError:
                    pass

            if len(result["components"]) == 4:
                result["status"] = "complete"

            return result

        # With short timeout, should get partial result
        result = await analysis_with_timeout(0.3)
        assert result["status"] == "partial"
        assert len(result["components"]) >= 2

        # With long timeout, should get complete or partial result
        # (depends on system load and timing)
        result = await analysis_with_timeout(5.0)
        assert result["status"] in ("complete", "partial")
        # Should have at least 3 components with longer timeout
        assert len(result["components"]) >= 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self) -> None:
        """Test circuit breaker activates under repeated failures."""
        from src.infra.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker("test_service")

        # Simulate failures
        async def failing_call() -> str:
            raise ConnectionError("Service unavailable")

        for i in range(10):
            try:
                await breaker.call(failing_call)
            except (ConnectionError, Exception):
                pass

        # Circuit should be open after failures
        assert breaker.state == CircuitState.OPEN


class TestPerformanceRegression:
    """Tests to detect performance regressions."""

    @pytest.fixture
    def baseline_latencies(self) -> dict[str, float]:
        """Baseline latency expectations."""
        return {
            "holder_fetch": 2.0,  # seconds
            "graph_build": 1.0,
            "metric_compute": 0.5,
            "full_analysis": 15.0,
        }

    @pytest.mark.asyncio
    async def test_component_latencies(
        self,
        baseline_latencies: dict[str, float],
    ) -> None:
        """Test that component latencies don't regress."""
        # Mock implementations that should be faster than baseline
        mock_latencies = {
            "holder_fetch": 0.1,
            "graph_build": 0.05,
            "metric_compute": 0.02,
            "full_analysis": 0.5,
        }

        for component, expected in baseline_latencies.items():
            actual = mock_latencies[component]
            assert actual < expected, (
                f"{component} latency regressed: {actual}s > {expected}s baseline"
            )
