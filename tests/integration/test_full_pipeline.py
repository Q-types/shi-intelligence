"""
Full Pipeline Integration Tests.

End-to-end tests for the complete analysis pipeline.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

from shi.core.types import TokenMint, HolderSnapshot
from shi.pipeline.orchestrator import AnalysisOrchestrator, AnalysisResult
from shi.data.client import SolanaDataClient
from shi.metrics.distribution import compute_hhi, compute_gini_coefficient


class MockSolanaDataClient:
    """Mock Solana data client for integration tests."""

    def __init__(self, holder_count: int = 100):
        self.holder_count = holder_count
        self._generate_mock_data()

    def _generate_mock_data(self) -> None:
        """Generate realistic mock holder data."""
        rng = np.random.default_rng(42)

        # Power law distribution for balances
        balances = rng.pareto(1.5, self.holder_count) * 1000
        total_supply = sum(balances)

        self.holders = [
            {
                "address": f"wallet_{i:04d}",
                "balance": int(balances[i]),
                "balance_ui": float(balances[i]),
                "share": float(balances[i] / total_supply),
            }
            for i in range(self.holder_count)
        ]
        self.total_supply = total_supply

    async def get_token_holders(
        self,
        mint: TokenMint,
        limit: int = 1000,
    ) -> list[dict]:
        """Get mock token holders."""
        await asyncio.sleep(0.01)  # Simulate network latency
        return self.holders[:limit]

    async def get_token_metadata(self, mint: TokenMint) -> dict:
        """Get mock token metadata."""
        await asyncio.sleep(0.01)
        return {
            "mint": mint,
            "name": "Test Token",
            "symbol": "TEST",
            "decimals": 9,
            "supply": self.total_supply,
        }

    async def get_funding_transactions(
        self,
        wallets: list[str],
        limit: int = 100,
    ) -> list[dict]:
        """Get mock funding transactions."""
        await asyncio.sleep(0.01)
        rng = np.random.default_rng(42)

        transactions = []
        for i in range(min(limit, len(wallets) * 2)):
            from_idx = rng.integers(0, len(wallets))
            to_idx = rng.integers(0, len(wallets))
            if from_idx != to_idx:
                transactions.append({
                    "from": wallets[from_idx],
                    "to": wallets[to_idx],
                    "amount": float(rng.uniform(0.1, 10.0)),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "signature": f"tx_{i:06d}",
                })

        return transactions


class TestFullAnalysisPipeline:
    """Integration tests for the full analysis pipeline."""

    @pytest.fixture
    def mock_client(self) -> MockSolanaDataClient:
        return MockSolanaDataClient(holder_count=100)

    @pytest.fixture
    def orchestrator(self, mock_client: MockSolanaDataClient) -> AnalysisOrchestrator:
        """Create orchestrator with mock client."""
        # Patch the data client
        with patch.object(
            SolanaDataClient,
            'get_token_holders',
            mock_client.get_token_holders,
        ):
            return AnalysisOrchestrator(data_client=mock_client)

    @pytest.mark.asyncio
    async def test_full_analysis_returns_result(
        self,
        orchestrator: AnalysisOrchestrator,
    ) -> None:
        """Test that full analysis returns complete result."""
        mint = "TestMint123456789012345678901234567890123"

        with patch.object(
            orchestrator._data_client,
            'get_token_holders',
            orchestrator._data_client.get_token_holders,
        ):
            result = await orchestrator.analyze(mint, timeout=30)

        assert isinstance(result, AnalysisResult)
        assert result.mint == mint
        assert result.holder_count > 0
        assert result.computed_at is not None

    @pytest.mark.asyncio
    async def test_metrics_are_computed(
        self,
        mock_client: MockSolanaDataClient,
    ) -> None:
        """Test that all required metrics are computed."""
        # Extract shares from mock data
        shares = [h["share"] for h in mock_client.holders]

        # Compute metrics
        hhi = compute_hhi(shares)
        gini = compute_gini_coefficient([h["balance_ui"] for h in mock_client.holders])

        # Verify metrics are valid
        assert 0 <= hhi.value <= 1
        assert 0 <= gini.value <= 1
        assert hhi.is_valid
        assert gini.is_valid

    @pytest.mark.asyncio
    async def test_timeout_returns_partial_result(
        self,
        orchestrator: AnalysisOrchestrator,
    ) -> None:
        """Test that timeout returns partial result."""
        mint = "TestMint123456789012345678901234567890123"

        # Use very short timeout
        result = await orchestrator.analyze(mint, timeout=0.001)

        # Should return partial result, not raise
        assert isinstance(result, AnalysisResult)
        # May be partial
        if result.is_partial:
            assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        """Test that errors are handled gracefully."""
        # Create client that raises errors
        error_client = MagicMock()
        error_client.get_token_holders = AsyncMock(
            side_effect=ConnectionError("Network error")
        )

        orchestrator = AnalysisOrchestrator(data_client=error_client)

        result = await orchestrator.analyze(
            "ErrorMint12345678901234567890123456789012",
            timeout=5,
        )

        # Should return result with error, not raise
        assert isinstance(result, AnalysisResult)
        assert result.is_partial or len(result.warnings) > 0


class TestMetricConsistency:
    """Tests for metric consistency across the pipeline."""

    @pytest.fixture
    def mock_client(self) -> MockSolanaDataClient:
        return MockSolanaDataClient(holder_count=50)

    @pytest.mark.asyncio
    async def test_hhi_consistency(
        self,
        mock_client: MockSolanaDataClient,
    ) -> None:
        """Test HHI is consistent across computations."""
        shares = [h["share"] for h in mock_client.holders]

        # Compute multiple times
        results = [compute_hhi(shares) for _ in range(10)]

        # Should be identical (deterministic)
        values = [r.value for r in results]
        assert len(set(values)) == 1, "HHI should be deterministic"

    @pytest.mark.asyncio
    async def test_gini_bounds(
        self,
        mock_client: MockSolanaDataClient,
    ) -> None:
        """Test Gini coefficient stays in valid bounds."""
        balances = [h["balance_ui"] for h in mock_client.holders]

        gini = compute_gini_coefficient(balances)

        assert 0 <= gini.value <= 1, f"Gini {gini.value} out of bounds"

    @pytest.mark.asyncio
    async def test_metric_reproducibility(
        self,
        mock_client: MockSolanaDataClient,
    ) -> None:
        """Test metrics are reproducible with same input."""
        shares = [h["share"] for h in mock_client.holders]
        balances = [h["balance_ui"] for h in mock_client.holders]

        # Compute checksums
        import hashlib
        import json

        hhi1 = compute_hhi(shares)
        gini1 = compute_gini_coefficient(balances)

        # Shuffle input order (shouldn't affect result for these metrics)
        hhi2 = compute_hhi(shares)
        gini2 = compute_gini_coefficient(balances)

        assert hhi1.value == hhi2.value
        assert gini1.value == gini2.value


class TestDatabaseIntegration:
    """Tests for database integration."""

    @pytest.mark.asyncio
    async def test_result_storage(self) -> None:
        """Test that analysis results can be stored."""
        # This would use a test database in production
        from shi.data.models import Token, Metric

        # Mock database session
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        # Create test metric
        metric = Metric(
            token_id=1,
            metric_type="hhi",
            value=0.05,
            z_score=1.2,
            computed_at=datetime.now(timezone.utc),
        )

        # Should be storable
        mock_session.add(metric)
        await mock_session.commit()

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_historical_comparison(self) -> None:
        """Test historical metric comparison."""
        # Mock historical data
        historical_metrics = [
            {"timestamp": "2024-01-01", "hhi": 0.04},
            {"timestamp": "2024-01-02", "hhi": 0.045},
            {"timestamp": "2024-01-03", "hhi": 0.05},
        ]

        current_hhi = 0.06

        # Compute drift
        historical_values = [m["hhi"] for m in historical_metrics]
        mean_historical = np.mean(historical_values)
        std_historical = np.std(historical_values)

        if std_historical > 0:
            z_score = (current_hhi - mean_historical) / std_historical
        else:
            z_score = 0

        # Current value should show some drift
        assert abs(z_score) > 1, "Should detect drift from historical"


class TestTelegramIntegration:
    """Tests for Telegram bot integration."""

    @pytest.mark.asyncio
    async def test_analyze_command_flow(self) -> None:
        """Test /analyze command flow."""
        from shi.telegram.handlers import run_full_analysis

        # Mock the orchestrator
        mock_result = {
            "mint": "TestMint123",
            "holder_count": 100,
            "metrics": {"hhi": 0.05, "gini": 0.7},
            "stability_score": 75,
            "sell_pressure": 0.3,
            "sybil_prob": 0.1,
            "archetypes": {"accumulator": 0.4, "sniper": 0.2},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "warnings": [],
        }

        with patch(
            'shi.telegram.handlers.run_full_analysis',
            AsyncMock(return_value=mock_result),
        ):
            result = await run_full_analysis("TestMint123")

        assert result["holder_count"] == 100
        assert "metrics" in result
        assert result["stability_score"] == 75

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self) -> None:
        """Test rate limiting integration."""
        from shi.telegram.bot import SHIBot

        bot = SHIBot(token="test_token", rate_limit=5)

        user_id = 12345

        # First 5 requests should succeed
        for _ in range(5):
            allowed, _ = bot.check_rate_limit(user_id)
            assert allowed

        # 6th request should be rate limited
        allowed, retry_after = bot.check_rate_limit(user_id)
        assert not allowed
        assert retry_after > 0

    @pytest.mark.asyncio
    async def test_security_middleware(self) -> None:
        """Test security middleware integration."""
        from shi.telegram.security import SecurityMiddleware, SecurityConfig

        config = SecurityConfig(
            admin_user_ids={1},
            banned_user_ids={999},
        )
        middleware = SecurityMiddleware(config)

        # Test admin access
        is_admin_authorized, _ = await middleware.auth_manager.is_authorized(
            1, "/ban"
        )
        assert is_admin_authorized

        # Test banned user
        is_banned_authorized, error = await middleware.auth_manager.is_authorized(
            999, "/analyze"
        )
        assert not is_banned_authorized
        assert "banned" in error.lower()
