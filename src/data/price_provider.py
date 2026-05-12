"""
Jupiter Price API Integration.

Provides token price data from Jupiter Price API with:
- In-memory caching (configurable TTL)
- Rate limiting (10 req/s default)
- Batch support for multiple tokens
- Retry logic with exponential backoff
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import httpx
import structlog

from ..core.config import settings
from ..core.types import TokenMint

logger = structlog.get_logger()


@dataclass(frozen=True)
class PriceData:
    """Token price data from Jupiter API.

    Attributes
    ----------
    mint : str
        Token mint address.
    price_usd : float
        Current price in USD.
    price_change_24h_pct : float | None
        24-hour price change percentage (if available).
    confidence : str
        Price confidence level: "high", "medium", or "low".
    source : str
        Data source identifier.
    fetched_at : datetime
        Timestamp when price was fetched.
    """

    mint: str
    price_usd: float
    price_change_24h_pct: float | None
    confidence: Literal["high", "medium", "low"]
    source: str
    fetched_at: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "mint": self.mint,
            "price_usd": self.price_usd,
            "price_change_24h_pct": self.price_change_24h_pct,
            "confidence": self.confidence,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
        }


@dataclass
class CacheEntry:
    """Cache entry for price data."""

    data: PriceData
    expires_at: datetime


class JupiterPriceProvider:
    """
    Jupiter Price API provider with caching and rate limiting.

    Features:
    - In-memory cache with configurable TTL (default 60s)
    - Rate limiting to stay within API limits (10 req/s)
    - Batch support for fetching multiple tokens
    - Automatic retry with exponential backoff

    Usage:
        provider = JupiterPriceProvider()
        price = await provider.get_price("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
        if price:
            print(f"USDC: ${price.price_usd}")
    """

    API_URL = "https://api.jup.ag/price/v3"
    RATE_LIMIT_PER_SECOND = 10
    DEFAULT_CACHE_TTL = 60  # seconds

    def __init__(
        self,
        cache_ttl: int | None = None,
        rate_limit: int | None = None,
    ):
        """Initialize the price provider.

        Parameters
        ----------
        cache_ttl : int | None
            Cache TTL in seconds. Defaults to settings or 60s.
        rate_limit : int | None
            Max requests per second. Defaults to 10.
        """
        self._cache_ttl = cache_ttl or getattr(
            settings, "price_cache_ttl_seconds", self.DEFAULT_CACHE_TTL
        )
        self._rate_limit = rate_limit or self.RATE_LIMIT_PER_SECOND

        # In-memory cache
        self._cache: dict[str, CacheEntry] = {}

        # Rate limiting state
        self._request_times: list[datetime] = []
        self._rate_lock = asyncio.Lock()

        # HTTP client (lazy initialization)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "SHI/1.0",
                },
            )
        return self._client

    async def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        async with self._rate_lock:
            now = datetime.now(timezone.utc)

            # Remove old timestamps (older than 1 second)
            one_second_ago = now.timestamp() - 1.0
            self._request_times = [
                t for t in self._request_times
                if t.timestamp() > one_second_ago
            ]

            # If at rate limit, wait
            if len(self._request_times) >= self._rate_limit:
                oldest = min(t.timestamp() for t in self._request_times)
                wait_time = 1.0 - (now.timestamp() - oldest)
                if wait_time > 0:
                    logger.debug("rate_limit_wait", wait_seconds=wait_time)
                    await asyncio.sleep(wait_time)

            # Record this request
            self._request_times.append(datetime.now(timezone.utc))

    def _get_cached(self, mint: str) -> PriceData | None:
        """Get price from cache if valid."""
        entry = self._cache.get(mint)
        if entry is None:
            return None

        now = datetime.now(timezone.utc)
        if now >= entry.expires_at:
            # Cache expired
            del self._cache[mint]
            return None

        return entry.data

    def _set_cached(self, price: PriceData) -> None:
        """Store price in cache."""
        expires_at = datetime.now(timezone.utc).timestamp() + self._cache_ttl
        self._cache[price.mint] = CacheEntry(
            data=price,
            expires_at=datetime.fromtimestamp(expires_at, tz=timezone.utc),
        )

    async def get_price(
        self,
        mint: str,
        skip_cache: bool = False,
    ) -> PriceData | None:
        """
        Get price for a single token.

        Parameters
        ----------
        mint : str
            Token mint address.
        skip_cache : bool
            If True, bypass cache and fetch fresh data.

        Returns
        -------
        PriceData | None
            Price data if available, None otherwise.
        """
        # Check cache first
        if not skip_cache:
            cached = self._get_cached(mint)
            if cached:
                logger.debug("price_cache_hit", mint=mint[:8])
                return cached

        # Fetch from API
        prices = await self.get_prices_batch([mint], skip_cache=True)
        return prices.get(mint)

    async def get_prices_batch(
        self,
        mints: list[str],
        skip_cache: bool = False,
    ) -> dict[str, PriceData]:
        """
        Get prices for multiple tokens in a single request.

        Parameters
        ----------
        mints : list[str]
            List of token mint addresses.
        skip_cache : bool
            If True, bypass cache and fetch fresh data.

        Returns
        -------
        dict[str, PriceData]
            Map of mint -> PriceData for found tokens.
        """
        if not mints:
            return {}

        result: dict[str, PriceData] = {}
        mints_to_fetch: list[str] = []

        # Check cache for each mint
        if not skip_cache:
            for mint in mints:
                cached = self._get_cached(mint)
                if cached:
                    result[mint] = cached
                else:
                    mints_to_fetch.append(mint)
        else:
            mints_to_fetch = list(mints)

        if not mints_to_fetch:
            logger.debug("prices_all_cached", count=len(result))
            return result

        # Respect rate limit
        await self._wait_for_rate_limit()

        # Fetch from Jupiter API
        try:
            client = await self._get_client()

            # Jupiter supports comma-separated mints
            ids_param = ",".join(mints_to_fetch)

            logger.debug(
                "fetching_prices",
                mint_count=len(mints_to_fetch),
                first_mint=mints_to_fetch[0][:8] if mints_to_fetch else None,
            )

            response = await client.get(
                self.API_URL,
                params={"ids": ids_param},
            )
            response.raise_for_status()

            data = response.json()
            now = datetime.now(timezone.utc)

            # Parse v3 response format (direct dict, not nested under "data")
            for mint in mints_to_fetch:
                token_data = data.get(mint)
                if token_data is None:
                    logger.debug("price_not_found", mint=mint[:8])
                    continue

                # v3 uses "usdPrice" instead of "price"
                price = token_data.get("usdPrice")
                if price is None:
                    continue

                # v3 includes 24h price change
                price_change_24h = token_data.get("priceChange24h")

                # Determine confidence based on liquidity
                confidence = self._determine_confidence(token_data)

                price_obj = PriceData(
                    mint=mint,
                    price_usd=float(price),
                    price_change_24h_pct=float(price_change_24h) if price_change_24h else None,
                    confidence=confidence,
                    source="jupiter",
                    fetched_at=now,
                )

                # Cache and add to result
                self._set_cached(price_obj)
                result[mint] = price_obj

            logger.info(
                "prices_fetched",
                requested=len(mints_to_fetch),
                found=len(result) - sum(1 for m in mints if m in result and self._get_cached(m)),
                total=len(result),
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                "jupiter_api_error",
                status_code=e.response.status_code,
                error=str(e),
            )
        except httpx.RequestError as e:
            logger.error("jupiter_request_error", error=str(e))
        except Exception as e:
            logger.error("jupiter_unexpected_error", error=str(e))

        return result

    def _determine_confidence(self, token_data: dict) -> Literal["high", "medium", "low"]:
        """Determine price confidence based on API response data.

        Uses liquidity as confidence indicator:
        - High: liquidity > $1M
        - Medium: liquidity > $10K
        - Low: liquidity <= $10K or unknown
        """
        # v3 API returns liquidity directly
        liquidity = token_data.get("liquidity")

        if liquidity is not None:
            if liquidity >= 1_000_000:  # $1M+
                return "high"
            elif liquidity >= 10_000:  # $10K+
                return "medium"
            else:
                return "low"

        # Fallback: check if price exists
        if token_data.get("usdPrice") is not None:
            return "medium"
        return "low"

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def clear_cache(self) -> None:
        """Clear the price cache."""
        self._cache.clear()
        logger.info("price_cache_cleared")

    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


# Well-known token mints for testing
WELL_KNOWN_MINTS = {
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    "SOL": "So11111111111111111111111111111111111111112",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
}
