"""
Tests for Price Provider System (Sprint 7 Enhanced).

Tests cover:
- Basic price fetching (Jupiter V3)
- Batch price fetching
- Caching behavior
- Rate limiting
- Error handling (timeout, 429, malformed, etc.)
- Fallback provider hierarchy
- Schema drift detection
- Price confidence computation
- Entry price calculation
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.data.price_provider import (
    JupiterPriceProvider,
    BirdeyePriceProvider,
    PoolImpliedPriceProvider,
    PriceProviderChain,
    PriceData,
    PriceConfidence,
    CacheEntry,
    WELL_KNOWN_MINTS,
    _compute_confidence,
    _compute_payload_hash,
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
            confidence=PriceConfidence.HIGH,
            source="jupiter",
            fetched_at=now,
        )

        assert price.mint == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        assert price.price_usd == 1.0
        assert price.price_change_24h_pct == 0.5
        assert price.confidence == PriceConfidence.HIGH
        assert price.source == "jupiter"
        assert price.fetched_at == now

    def test_price_data_to_dict(self):
        """Test PriceData to_dict conversion."""
        now = datetime.now(timezone.utc)
        price = PriceData(
            mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            price_usd=1.0,
            price_change_24h_pct=None,
            confidence=PriceConfidence.MEDIUM,
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

    def test_confidence_uses_module_function(self, provider):
        """Test that provider uses module-level _compute_confidence."""
        # Confidence computation is done via _compute_confidence module function.
        # High liquidity -> high confidence
        confidence = _compute_confidence(5_000_000, "jupiter")
        assert confidence == PriceConfidence.HIGH

        # Medium liquidity -> medium confidence
        confidence = _compute_confidence(50_000, "jupiter")
        assert confidence == PriceConfidence.MEDIUM

        # Low liquidity -> low confidence
        confidence = _compute_confidence(5_000, "jupiter")
        assert confidence == PriceConfidence.LOW

        # No liquidity -> medium confidence (conservative default)
        confidence = _compute_confidence(None, "jupiter")
        assert confidence == PriceConfidence.MEDIUM


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


# =============================================================================
# Sprint 7 Enhanced Tests: Error Handling, Fallback, Schema Drift
# =============================================================================


class TestPriceConfidence:
    """Tests for price confidence computation."""

    def test_compute_confidence_high_liquidity(self):
        """High confidence for >$1M liquidity from primary source."""
        confidence = _compute_confidence(2_000_000, "jupiter")
        assert confidence == PriceConfidence.HIGH

    def test_compute_confidence_medium_liquidity(self):
        """Medium confidence for $10K-$1M liquidity."""
        confidence = _compute_confidence(50_000, "jupiter")
        assert confidence == PriceConfidence.MEDIUM

    def test_compute_confidence_low_liquidity(self):
        """Low confidence for <$10K liquidity."""
        confidence = _compute_confidence(5_000, "jupiter")
        assert confidence == PriceConfidence.LOW

    def test_compute_confidence_pool_implied_always_low(self):
        """Pool-implied prices are always low confidence."""
        # Even with high liquidity, pool-implied is low confidence
        confidence = _compute_confidence(10_000_000, "pool_implied")
        assert confidence == PriceConfidence.LOW

    def test_compute_confidence_none_liquidity(self):
        """Medium confidence when liquidity is unknown."""
        confidence = _compute_confidence(None, "jupiter")
        assert confidence == PriceConfidence.MEDIUM


class TestPayloadHash:
    """Tests for payload hash computation."""

    def test_payload_hash_deterministic(self):
        """Same input produces same hash."""
        data = {"mint": "abc123", "price": 1.0}
        hash1 = _compute_payload_hash(data)
        hash2 = _compute_payload_hash(data)
        assert hash1 == hash2

    def test_payload_hash_different_data(self):
        """Different input produces different hash."""
        data1 = {"mint": "abc123", "price": 1.0}
        data2 = {"mint": "abc123", "price": 2.0}
        hash1 = _compute_payload_hash(data1)
        hash2 = _compute_payload_hash(data2)
        assert hash1 != hash2

    def test_payload_hash_length(self):
        """Hash is 16 characters."""
        data = {"test": "data"}
        hash_val = _compute_payload_hash(data)
        assert len(hash_val) == 16


class TestJupiterErrorHandling:
    """Tests for Jupiter API error scenarios."""

    @pytest.fixture
    def provider(self):
        return JupiterPriceProvider(cache_ttl=60, rate_limit=10)

    @pytest.mark.asyncio
    async def test_timeout_error(self, provider):
        """Test handling of API timeout."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.TimeoutException("Request timed out")
            mock_get_client.return_value = mock_client

            result = await provider.get_price("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")

            assert result is None

        await provider.close()

    @pytest.mark.asyncio
    async def test_rate_limit_429_error(self, provider):
        """Test handling of 429 rate limit response."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await provider.get_price("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")

            # Should return empty result on 429
            assert result is None

        await provider.close()

    @pytest.mark.asyncio
    async def test_empty_data_response(self, provider):
        """Test handling of response with no data for token."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {}  # Empty response
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await provider.get_price("unknown_mint_address")

            assert result is None

        await provider.close()

    @pytest.mark.asyncio
    async def test_malformed_response(self, provider):
        """Test handling of malformed API response."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            # Response with token but missing required fields
            mock_response.json.return_value = {
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": {
                    # Missing "usdPrice" field
                    "liquidity": 1000000,
                }
            }
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await provider.get_price("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")

            # Should return None when price field is missing
            assert result is None

        await provider.close()

    @pytest.mark.asyncio
    async def test_batch_partial_success(self, provider):
        """Test batch fetch handles partial data availability."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            # Only one token has data
            mock_response.json.return_value = {
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": {
                    "usdPrice": 1.0,
                    "liquidity": 1000000,
                }
                # Second token has no data
            }
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            mints = [
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "unknown_mint_12345",
            ]
            result = await provider.get_prices_batch(mints)

            # Should have one result
            assert len(result) == 1
            assert "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" in result

        await provider.close()

    @pytest.mark.asyncio
    async def test_connection_error(self, provider):
        """Test handling of connection failure."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_get_client.return_value = mock_client

            result = await provider.get_price("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")

            assert result is None

        await provider.close()


class TestPriceProviderChain:
    """Tests for fallback provider chain."""

    @pytest.mark.asyncio
    async def test_chain_returns_first_success(self):
        """Chain returns result from first successful provider."""
        # Create mock providers
        mock_jupiter = MagicMock(spec=JupiterPriceProvider)
        mock_jupiter.priority = 1
        mock_jupiter.name = "jupiter"
        mock_jupiter.get_price = AsyncMock(return_value=PriceData(
            mint="test",
            price_usd=1.0,
            price_change_24h_pct=None,
            confidence=PriceConfidence.HIGH,
            source="jupiter",
            fetched_at=datetime.now(timezone.utc),
        ))

        mock_birdeye = MagicMock(spec=BirdeyePriceProvider)
        mock_birdeye.priority = 2
        mock_birdeye.name = "birdeye"
        mock_birdeye.get_price = AsyncMock()  # Should not be called

        chain = PriceProviderChain(providers=[mock_jupiter, mock_birdeye])

        result = await chain.get_price("test")

        assert result is not None
        assert result.source == "jupiter"
        mock_jupiter.get_price.assert_called_once()
        mock_birdeye.get_price.assert_not_called()

        await chain.close()

    @pytest.mark.asyncio
    async def test_chain_falls_through_on_failure(self):
        """Chain tries next provider when first fails."""
        mock_jupiter = MagicMock(spec=JupiterPriceProvider)
        mock_jupiter.priority = 1
        mock_jupiter.name = "jupiter"
        mock_jupiter.get_price = AsyncMock(return_value=None)  # Jupiter fails
        mock_jupiter.close = AsyncMock()

        mock_birdeye = MagicMock(spec=BirdeyePriceProvider)
        mock_birdeye.priority = 2
        mock_birdeye.name = "birdeye"
        mock_birdeye.get_price = AsyncMock(return_value=PriceData(
            mint="test",
            price_usd=1.0,
            price_change_24h_pct=None,
            confidence=PriceConfidence.MEDIUM,
            source="birdeye",
            fetched_at=datetime.now(timezone.utc),
        ))
        mock_birdeye.close = AsyncMock()

        chain = PriceProviderChain(providers=[mock_jupiter, mock_birdeye])

        result = await chain.get_price("test")

        assert result is not None
        assert result.source == "birdeye"
        mock_jupiter.get_price.assert_called_once()
        mock_birdeye.get_price.assert_called_once()

        await chain.close()

    @pytest.mark.asyncio
    async def test_chain_returns_none_when_all_fail(self):
        """Chain returns None when all providers fail."""
        mock_jupiter = MagicMock(spec=JupiterPriceProvider)
        mock_jupiter.priority = 1
        mock_jupiter.name = "jupiter"
        mock_jupiter.get_price = AsyncMock(return_value=None)
        mock_jupiter.close = AsyncMock()

        mock_birdeye = MagicMock(spec=BirdeyePriceProvider)
        mock_birdeye.priority = 2
        mock_birdeye.name = "birdeye"
        mock_birdeye.get_price = AsyncMock(return_value=None)
        mock_birdeye.close = AsyncMock()

        chain = PriceProviderChain(providers=[mock_jupiter, mock_birdeye])

        result = await chain.get_price("unknown_token")

        assert result is None

        await chain.close()

    @pytest.mark.asyncio
    async def test_chain_handles_provider_exception(self):
        """Chain continues to next provider on exception."""
        mock_jupiter = MagicMock(spec=JupiterPriceProvider)
        mock_jupiter.priority = 1
        mock_jupiter.name = "jupiter"
        mock_jupiter.get_price = AsyncMock(side_effect=Exception("API Error"))
        mock_jupiter.close = AsyncMock()

        mock_birdeye = MagicMock(spec=BirdeyePriceProvider)
        mock_birdeye.priority = 2
        mock_birdeye.name = "birdeye"
        mock_birdeye.get_price = AsyncMock(return_value=PriceData(
            mint="test",
            price_usd=1.0,
            price_change_24h_pct=None,
            confidence=PriceConfidence.MEDIUM,
            source="birdeye",
            fetched_at=datetime.now(timezone.utc),
        ))
        mock_birdeye.close = AsyncMock()

        chain = PriceProviderChain(providers=[mock_jupiter, mock_birdeye])

        result = await chain.get_price("test")

        # Should fall through to Birdeye despite Jupiter exception
        assert result is not None
        assert result.source == "birdeye"

        await chain.close()


class TestConfigDrivenEndpoints:
    """Tests for config-driven Jupiter endpoints."""

    def test_provider_uses_config_base_url(self):
        """Provider reads base URL from config."""
        with patch("src.data.price_provider.settings") as mock_settings:
            mock_settings.jupiter_price_base_url = "https://test-api.jup.ag"
            mock_settings.jupiter_price_path = "/price/v3"
            mock_settings.jupiter_api_key = None
            mock_settings.price_cache_ttl_seconds = 60

            provider = JupiterPriceProvider()

            assert provider._base_url == "https://test-api.jup.ag"
            assert provider.api_url == "https://test-api.jup.ag/price/v3"

    def test_provider_with_api_key_uses_pro_rate_limit(self):
        """Provider with API key uses higher rate limit."""
        with patch("src.data.price_provider.settings") as mock_settings:
            mock_settings.jupiter_price_base_url = "https://api.jup.ag"
            mock_settings.jupiter_price_path = "/price/v3"
            mock_settings.jupiter_api_key = "test_api_key"
            mock_settings.price_cache_ttl_seconds = 60

            provider = JupiterPriceProvider()

            # Pro rate limit is higher
            assert provider._rate_limit == JupiterPriceProvider.PRO_RATE_LIMIT
            assert provider._api_key == "test_api_key"


class TestSchemaValidation:
    """Tests for schema drift and validation."""

    @pytest.fixture
    def provider(self):
        return JupiterPriceProvider(cache_ttl=60, rate_limit=10)

    @pytest.mark.asyncio
    async def test_handles_v3_response_format(self, provider):
        """Test handling of V3 API response format."""
        v3_response = {
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": {
                "usdPrice": 1.0,  # V3 uses usdPrice
                "liquidity": 500000000,
                "priceChange24h": 0.05,
            }
        }

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = v3_response
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await provider.get_price("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")

            assert result is not None
            assert result.price_usd == 1.0

        await provider.close()

    @pytest.mark.asyncio
    async def test_handles_unexpected_field_names(self, provider):
        """Test graceful handling of schema changes (extra fields)."""
        response_with_extra_fields = {
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": {
                "usdPrice": 1.0,
                "liquidity": 500000000,
                "priceChange24h": 0.05,
                "newUnexpectedField": "some_value",  # New field
                "anotherNewField": 12345,
            }
        }

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = response_with_extra_fields
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await provider.get_price("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")

            # Should still work with extra fields
            assert result is not None
            assert result.price_usd == 1.0

        await provider.close()
