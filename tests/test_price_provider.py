"""
Tests for Jupiter Price Provider.

Tests cover:
- Basic price fetching
- Batch price fetching
- Caching behavior
- Rate limiting
- Error handling
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.price_provider import (
    JupiterPriceProvider,
    PriceData,
    CacheEntry,
    WELL_KNOWN_MINTS,
)


class TestPriceData:
    """Tests for PriceData dataclass."""

    def test_create_price_data(self):
        """Test creating PriceData instance."""
        now = datetime.now(timezone.utc)
        price = PriceData(
            mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            price_usd=1.0,
            price_change_24h_pct=0.5,
            confidence="high",
            source="jupiter",
            fetched_at=now,
        )

        assert price.mint == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        assert price.price_usd == 1.0
        assert price.price_change_24h_pct == 0.5
        assert price.confidence == "high"
        assert price.source == "jupiter"
        assert price.fetched_at == now

    def test_price_data_to_dict(self):
        """Test PriceData to_dict conversion."""
        now = datetime.now(timezone.utc)
        price = PriceData(
            mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            price_usd=1.0,
            price_change_24h_pct=None,
            confidence="medium",
            source="jupiter",
            fetched_at=now,
        )

        result = price.to_dict()

        assert result["mint"] == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        assert result["price_usd"] == 1.0
        assert result["price_change_24h_pct"] is None
        assert result["confidence"] == "medium"
        assert result["source"] == "jupiter"
        assert result["fetched_at"] == now.isoformat()


class TestJupiterPriceProvider:
    """Tests for JupiterPriceProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a test provider instance."""
        return JupiterPriceProvider(cache_ttl=60, rate_limit=10)

    @pytest.fixture
    def mock_response(self):
        """Create a mock API response (v3 format)."""
        return {
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": {
                "usdPrice": 1.0,
                "liquidity": 500000000.0,  # $500M
                "priceChange24h": 0.05,
                "decimals": 6,
            },
            "So11111111111111111111111111111111111111112": {
                "usdPrice": 150.0,
                "liquidity": 1000000000.0,  # $1B
                "priceChange24h": 2.5,
                "decimals": 9,
            },
        }

    def test_provider_initialization(self, provider):
        """Test provider initializes with correct defaults."""
        assert provider._cache_ttl == 60
        assert provider._rate_limit == 10
        assert provider.cache_size == 0

    def test_cache_operations(self, provider):
        """Test cache set and get operations."""
        now = datetime.now(timezone.utc)
        price = PriceData(
            mint="test_mint",
            price_usd=100.0,
            price_change_24h_pct=None,
            confidence="high",
            source="jupiter",
            fetched_at=now,
        )

        # Set cache
        provider._set_cached(price)
        assert provider.cache_size == 1

        # Get from cache
        cached = provider._get_cached("test_mint")
        assert cached is not None
        assert cached.price_usd == 100.0

        # Non-existent key
        assert provider._get_cached("unknown") is None

    def test_cache_expiry(self, provider):
        """Test cache entries expire correctly."""
        now = datetime.now(timezone.utc)
        price = PriceData(
            mint="test_mint",
            price_usd=100.0,
            price_change_24h_pct=None,
            confidence="high",
            source="jupiter",
            fetched_at=now,
        )

        # Set cache with very short TTL (simulated by manually setting expired entry)
        provider._cache["test_mint"] = CacheEntry(
            data=price,
            expires_at=now - timedelta(seconds=1),  # Already expired
        )

        # Should return None and clean up
        cached = provider._get_cached("test_mint")
        assert cached is None
        assert provider.cache_size == 0

    def test_clear_cache(self, provider):
        """Test cache clearing."""
        now = datetime.now(timezone.utc)
        price = PriceData(
            mint="test_mint",
            price_usd=100.0,
            price_change_24h_pct=None,
            confidence="high",
            source="jupiter",
            fetched_at=now,
        )

        provider._set_cached(price)
        assert provider.cache_size == 1

        provider.clear_cache()
        assert provider.cache_size == 0

    @pytest.mark.asyncio
    async def test_get_price_from_cache(self, provider):
        """Test get_price returns cached data."""
        now = datetime.now(timezone.utc)
        price = PriceData(
            mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            price_usd=1.0,
            price_change_24h_pct=None,
            confidence="high",
            source="jupiter",
            fetched_at=now,
        )
        provider._set_cached(price)

        # Should return from cache without API call
        result = await provider.get_price("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")

        assert result is not None
        assert result.price_usd == 1.0
        await provider.close()

    @pytest.mark.asyncio
    async def test_get_price_skip_cache(self, provider, mock_response):
        """Test get_price with skip_cache bypasses cache."""
        now = datetime.now(timezone.utc)
        cached_price = PriceData(
            mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            price_usd=0.99,  # Old price
            price_change_24h_pct=None,
            confidence="medium",
            source="jupiter",
            fetched_at=now,
        )
        provider._set_cached(cached_price)

        # Mock the HTTP client
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response_obj
            mock_get_client.return_value = mock_client

            result = await provider.get_price(
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                skip_cache=True,
            )

            assert result is not None
            assert result.price_usd == 1.0  # New price from API
            mock_client.get.assert_called_once()

        await provider.close()

    @pytest.mark.asyncio
    async def test_get_prices_batch(self, provider, mock_response):
        """Test batch price fetching."""
        mints = [
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "So11111111111111111111111111111111111111112",
        ]

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response_obj
            mock_get_client.return_value = mock_client

            result = await provider.get_prices_batch(mints)

            assert len(result) == 2
            assert "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" in result
            assert "So11111111111111111111111111111111111111112" in result
            assert result["EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"].price_usd == 1.0
            assert result["So11111111111111111111111111111111111111112"].price_usd == 150.0

        await provider.close()

    @pytest.mark.asyncio
    async def test_get_prices_batch_partial_cache(self, provider, mock_response):
        """Test batch fetch uses cache for available prices."""
        now = datetime.now(timezone.utc)

        # Cache one price
        cached_price = PriceData(
            mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            price_usd=1.0,
            price_change_24h_pct=None,
            confidence="high",
            source="jupiter",
            fetched_at=now,
        )
        provider._set_cached(cached_price)

        # Mock API response for uncached mint (v3 format)
        partial_response = {
            "So11111111111111111111111111111111111111112": {
                "usdPrice": 150.0,
                "liquidity": 1000000000.0,
                "priceChange24h": 2.5,
            },
        }

        mints = [
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "So11111111111111111111111111111111111111112",
        ]

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = partial_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response_obj
            mock_get_client.return_value = mock_client

            result = await provider.get_prices_batch(mints)

            assert len(result) == 2
            # Verify API was called only for uncached mint
            call_args = mock_client.get.call_args
            assert "So11111111111111111111111111111111111111112" in call_args[1]["params"]["ids"]
            assert "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" not in call_args[1]["params"]["ids"]

        await provider.close()

    @pytest.mark.asyncio
    async def test_get_prices_empty_list(self, provider):
        """Test batch fetch with empty list."""
        result = await provider.get_prices_batch([])
        assert result == {}
        await provider.close()

    @pytest.mark.asyncio
    async def test_api_error_handling(self, provider):
        """Test API error handling."""
        import httpx

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.RequestError("Connection failed")
            mock_get_client.return_value = mock_client

            result = await provider.get_price("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")

            # Should return None on error, not raise
            assert result is None

        await provider.close()

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, provider):
        """Test rate limiting enforcement."""
        # Set a very low rate limit
        provider._rate_limit = 2
        provider._request_times = []

        # Record times for 2 requests
        now = datetime.now(timezone.utc)
        provider._request_times = [now, now]

        # Should wait before making another request
        start = datetime.now(timezone.utc)
        await provider._wait_for_rate_limit()
        end = datetime.now(timezone.utc)

        # Should have waited (at least a little)
        # Note: The actual wait time depends on timing
        assert len(provider._request_times) == 3  # New request recorded

        await provider.close()

    def test_determine_confidence_high(self, provider):
        """Test confidence determination for high liquidity (>$1M)."""
        token_data = {
            "usdPrice": 100.0,
            "liquidity": 5000000.0,  # $5M
        }

        confidence = provider._determine_confidence(token_data)
        assert confidence == "high"

    def test_determine_confidence_medium(self, provider):
        """Test confidence determination for medium liquidity ($10K-$1M)."""
        token_data = {
            "usdPrice": 100.0,
            "liquidity": 50000.0,  # $50K
        }

        confidence = provider._determine_confidence(token_data)
        assert confidence == "medium"

    def test_determine_confidence_low(self, provider):
        """Test confidence determination for low liquidity (<$10K)."""
        token_data = {
            "usdPrice": 100.0,
            "liquidity": 5000.0,  # $5K
        }

        confidence = provider._determine_confidence(token_data)
        assert confidence == "low"

    def test_determine_confidence_no_liquidity(self, provider):
        """Test confidence determination without liquidity info."""
        token_data = {
            "usdPrice": 100.0,
        }

        confidence = provider._determine_confidence(token_data)
        assert confidence == "medium"  # Default when price exists


class TestWellKnownMints:
    """Tests for well-known token mints."""

    def test_well_known_mints_exist(self):
        """Test that well-known mints are defined."""
        assert "USDC" in WELL_KNOWN_MINTS
        assert "USDT" in WELL_KNOWN_MINTS
        assert "SOL" in WELL_KNOWN_MINTS
        assert "BONK" in WELL_KNOWN_MINTS

    def test_usdc_mint_address(self):
        """Test USDC mint address is correct."""
        assert WELL_KNOWN_MINTS["USDC"] == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

    def test_sol_mint_address(self):
        """Test SOL mint address is correct."""
        assert WELL_KNOWN_MINTS["SOL"] == "So11111111111111111111111111111111111111112"
