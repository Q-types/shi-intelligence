"""
Integration tests for Price Integration Pipeline.

Tests the full integration of price data into the analysis pipeline.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.price_provider import JupiterPriceProvider, PriceData
from src.liquidity.pools import LiquidityFetcher, PoolInfo
from src.pipeline.orchestrator import AnalysisOrchestrator, AnalysisResult
from src.pipeline.features import FeatureEngineer


class TestPriceIntegration:
    """Tests for price integration in the analysis pipeline."""

    @pytest.fixture
    def mock_price_data(self) -> PriceData:
        """Create mock price data."""
        return PriceData(
            mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            price_usd=1.0,
            price_change_24h_pct=0.5,
            confidence="high",
            source="jupiter",
            fetched_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def mock_pool_info(self) -> PoolInfo:
        """Create mock pool info."""
        return PoolInfo(
            pool_address="pool123",
            dex="raydium",
            token_a_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            token_b_mint="So11111111111111111111111111111111111111112",
            token_a_reserve=1000000000,
            token_b_reserve=10000000000,
            token_a_decimals=6,
            token_b_decimals=9,
            liquidity_usd=1000000.0,
            volume_24h_usd=500000.0,
            fee_rate=0.0025,
            fetched_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_feature_engineer_with_price_data(self, mock_price_data):
        """Test FeatureEngineer computes price features correctly."""
        from src.core.types import HolderSnapshot, TokenBalance
        from src.graph import FundingGraph

        # Create mock snapshot
        now = datetime.now(timezone.utc)
        snapshot = HolderSnapshot(
            mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            timestamp=now,
            total_supply=1000000000,
            holder_count=2,
            balances=[
                TokenBalance(
                    wallet="11111111111111111111111111111111",
                    mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    balance=500000000,
                    decimals=6,
                    timestamp=now,
                ),
                TokenBalance(
                    wallet="22222222222222222222222222222222",
                    mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    balance=500000000,
                    decimals=6,
                    timestamp=now,
                ),
            ],
        )

        # Create empty funding graph
        funding_graph = FundingGraph()
        funding_graph.add_wallet("11111111111111111111111111111111")
        funding_graph.add_wallet("22222222222222222222222222222222")

        # Compute features with price data
        engineer = FeatureEngineer()
        features = engineer.compute_features(
            snapshot=snapshot,
            funding_graph=funding_graph,
            price_data=mock_price_data,
        )

        assert len(features) == 2

        # Verify price features are populated
        for feature in features:
            assert feature.current_price_usd == 1.0
            # Entry price depends on historical data (None without temporal context)
            # but current_price should always be set

    @pytest.mark.asyncio
    async def test_orchestrator_with_providers(self, mock_price_data, mock_pool_info):
        """Test orchestrator uses price and liquidity providers."""
        # Create mock providers
        mock_price_provider = AsyncMock(spec=JupiterPriceProvider)
        mock_price_provider.get_price.return_value = mock_price_data
        mock_price_provider.close.return_value = None

        mock_liquidity_fetcher = AsyncMock(spec=LiquidityFetcher)
        mock_liquidity_fetcher.get_all_pools.return_value = [mock_pool_info]
        mock_liquidity_fetcher.close.return_value = None

        # Create orchestrator with mocked providers
        orchestrator = AnalysisOrchestrator(
            price_provider=mock_price_provider,
            liquidity_fetcher=mock_liquidity_fetcher,
        )

        # Verify providers are set
        assert orchestrator.price_provider == mock_price_provider
        assert orchestrator.liquidity_fetcher == mock_liquidity_fetcher

        # Cleanup
        await orchestrator.close()

    @pytest.mark.asyncio
    async def test_analysis_result_includes_price_data(self, mock_price_data, mock_pool_info):
        """Test AnalysisResult correctly includes price and liquidity data."""
        from src.pipeline.metrics_pipeline import MetricsResult
        from src.core.types import MetricOutput

        now = datetime.now(timezone.utc)

        # Create mock metrics
        mock_metric = MetricOutput(
            metric_name="hhi",
            value=0.05,
            version="1.0.0",
            computed_at=now,
        )
        mock_metrics = MagicMock(spec=MetricsResult)
        mock_metrics.hhi = mock_metric
        mock_metrics.shannon_entropy = mock_metric
        mock_metrics.gini_coefficient = mock_metric
        mock_metrics.whale_dominance_ratio = mock_metric
        mock_metrics.churn_rate = mock_metric
        mock_metrics.coordination_score = mock_metric
        mock_metrics.funding_density = mock_metric
        mock_metrics.to_dict.return_value = {"hhi": 0.05}

        # Create AnalysisResult with price and liquidity data
        result = AnalysisResult(
            mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            holder_count=100,
            total_supply=1000000000,
            metrics=mock_metrics,
            archetypes={"sniper": 0.3, "accumulator": 0.7},
            risk_report=None,
            top_holder_sell_probs=[0.1, 0.1, 0.1],
            graph_stats={"vertices": 100, "edges": 50},
            price_data=mock_price_data,
            liquidity_usd=1000000.0,
            liquidity_pools=[mock_pool_info],
            liquidity_adjusted_pressure=0.5,
            analysis_version="1.0.0",
            computed_at=now,
            latency_ms=100,
            is_partial=False,
            warnings=[],
        )

        # Verify price and liquidity data is included
        assert result.price_data == mock_price_data
        assert result.liquidity_usd == 1000000.0
        assert len(result.liquidity_pools) == 1
        assert result.liquidity_adjusted_pressure == 0.5

        # Verify to_dict includes price and liquidity
        result_dict = result.to_dict()
        assert "price" in result_dict
        assert result_dict["price"]["price_usd"] == 1.0
        assert result_dict["price"]["confidence"] == "high"

        assert "liquidity" in result_dict
        assert result_dict["liquidity"]["total_usd"] == 1000000.0
        assert result_dict["liquidity"]["pool_count"] == 1
        assert result_dict["liquidity"]["adjusted_pressure"] == 0.5

    @pytest.mark.asyncio
    async def test_price_fetch_failure_graceful(self):
        """Test pipeline handles price fetch failure gracefully."""
        # Create mock provider that fails
        mock_price_provider = AsyncMock(spec=JupiterPriceProvider)
        mock_price_provider.get_price.side_effect = Exception("API Error")
        mock_price_provider.close.return_value = None

        orchestrator = AnalysisOrchestrator(
            price_provider=mock_price_provider,
        )

        # Fetch price should not raise, just return None
        warnings = []
        result = await orchestrator._fetch_price_data("test_mint", warnings)

        assert result is None
        assert len(warnings) == 1
        assert "Failed to fetch price" in warnings[0]

        await orchestrator.close()

    @pytest.mark.asyncio
    async def test_liquidity_fetch_failure_graceful(self):
        """Test pipeline handles liquidity fetch failure gracefully."""
        # Create mock fetcher that fails
        mock_liquidity_fetcher = AsyncMock(spec=LiquidityFetcher)
        mock_liquidity_fetcher.get_all_pools.side_effect = Exception("API Error")
        mock_liquidity_fetcher.close.return_value = None

        orchestrator = AnalysisOrchestrator(
            liquidity_fetcher=mock_liquidity_fetcher,
        )

        # Fetch liquidity should not raise, just return None
        warnings = []
        liquidity_usd, pools = await orchestrator._fetch_liquidity_data("test_mint", warnings)

        assert liquidity_usd is None
        assert pools == []
        assert len(warnings) == 1
        assert "Failed to fetch liquidity" in warnings[0]

        await orchestrator.close()


class TestPriceLiquidityAPIIntegration:
    """Tests for price and liquidity API endpoints."""

    @pytest.fixture
    def mock_price_data(self) -> PriceData:
        """Create mock price data."""
        return PriceData(
            mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            price_usd=1.0,
            price_change_24h_pct=None,
            confidence="high",
            source="jupiter",
            fetched_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def mock_pool_info(self) -> PoolInfo:
        """Create mock pool info."""
        return PoolInfo(
            pool_address="pool123",
            dex="raydium",
            token_a_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            token_b_mint="So11111111111111111111111111111111111111112",
            token_a_reserve=1000000000,
            token_b_reserve=10000000000,
            token_a_decimals=6,
            token_b_decimals=9,
            liquidity_usd=1000000.0,
            volume_24h_usd=500000.0,
            fee_rate=0.0025,
            fetched_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_price_endpoint_response_format(self, mock_price_data):
        """Test price endpoint returns correct response format."""
        from fastapi.testclient import TestClient
        from src.api.routes import app

        with patch("src.api.routes.JupiterPriceProvider") as MockProvider:
            # Setup mock
            mock_instance = AsyncMock()
            mock_instance.get_price.return_value = mock_price_data
            mock_instance.close.return_value = None
            MockProvider.return_value = mock_instance

            client = TestClient(app)
            response = client.get(
                "/api/v1/token/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v/price"
            )

            assert response.status_code == 200
            data = response.json()

            assert data["mint"] == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            assert data["price_usd"] == 1.0
            assert data["confidence"] == "high"
            assert data["source"] == "jupiter"

    @pytest.mark.asyncio
    async def test_price_endpoint_not_found(self):
        """Test price endpoint returns 404 when price not available."""
        from fastapi.testclient import TestClient
        from src.api.routes import app

        with patch("src.api.routes.JupiterPriceProvider") as MockProvider:
            # Setup mock to return None
            mock_instance = AsyncMock()
            mock_instance.get_price.return_value = None
            mock_instance.close.return_value = None
            MockProvider.return_value = mock_instance

            client = TestClient(app)
            response = client.get(
                "/api/v1/token/UnknownTokenMint1111111111111111111111/price"
            )

            assert response.status_code == 404
            assert "not available" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_liquidity_endpoint_response_format(self, mock_pool_info):
        """Test liquidity endpoint returns correct response format."""
        from fastapi.testclient import TestClient
        from src.api.routes import app

        with patch("src.api.routes.LiquidityFetcher") as MockFetcher:
            # Setup mock
            mock_instance = AsyncMock()
            mock_instance.get_all_pools.return_value = [mock_pool_info]
            mock_instance.close.return_value = None
            MockFetcher.return_value = mock_instance

            client = TestClient(app)
            response = client.get(
                "/api/v1/token/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v/liquidity"
            )

            assert response.status_code == 200
            data = response.json()

            assert data["mint"] == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            assert data["total_liquidity_usd"] == 1000000.0
            assert data["pool_count"] == 1
            assert len(data["pools"]) == 1
            assert data["deepest_pool"]["dex"] == "raydium"

    @pytest.mark.asyncio
    async def test_liquidity_endpoint_no_pools(self):
        """Test liquidity endpoint handles no pools gracefully."""
        from fastapi.testclient import TestClient
        from src.api.routes import app

        with patch("src.api.routes.LiquidityFetcher") as MockFetcher:
            # Setup mock to return empty list
            mock_instance = AsyncMock()
            mock_instance.get_all_pools.return_value = []
            mock_instance.close.return_value = None
            MockFetcher.return_value = mock_instance

            client = TestClient(app)
            response = client.get(
                "/api/v1/token/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v/liquidity"
            )

            assert response.status_code == 200
            data = response.json()

            assert data["total_liquidity_usd"] == 0.0
            assert data["pool_count"] == 0
            assert data["pools"] == []
            assert data["deepest_pool"] is None

    @pytest.mark.asyncio
    async def test_invalid_mint_address(self):
        """Test endpoints reject invalid mint addresses."""
        from fastapi.testclient import TestClient
        from src.api.routes import app

        client = TestClient(app)

        # Test price endpoint
        response = client.get("/api/v1/token/short/price")
        assert response.status_code == 400
        assert "Invalid mint address" in response.json()["detail"]

        # Test liquidity endpoint
        response = client.get("/api/v1/token/short/liquidity")
        assert response.status_code == 400
        assert "Invalid mint address" in response.json()["detail"]
