"""
Price Provider System with Fallback Hierarchy.

Provides token price data with:
- Jupiter V3 API (primary) with free/pro tier support
- Birdeye API (fallback)
- Pool-implied price from DEX reserves (last resort)
- In-memory caching with configurable TTL
- Rate limiting
- Computed confidence based on source/liquidity

Jupiter API Tiers:
- Free: https://lite-api.jup.ag (rate limited)
- Pro: https://api.jup.ag (requires API key)
"""

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Any, TYPE_CHECKING

import httpx
import structlog

from ..core.config import settings
from ..core.types import TokenMint

if TYPE_CHECKING:
    from ..liquidity.pools import LiquidityFetcher, PoolInfo

logger = structlog.get_logger()


class PriceConfidence(str, Enum):
    """Price confidence levels based on source and liquidity."""
    HIGH = "high"       # >$1M liquidity, primary source
    MEDIUM = "medium"   # >$10K liquidity, or fallback source
    LOW = "low"         # <$10K liquidity, or pool-implied
    NONE = "none"       # Price unavailable


@dataclass(frozen=True)
class PriceData:
    """Token price data from any provider.

    Attributes
    ----------
    mint : str
        Token mint address.
    price_usd : float
        Current price in USD.
    price_change_24h_pct : float | None
        24-hour price change percentage (if available).
    confidence : PriceConfidence
        Price confidence level computed from source and liquidity.
    source : str
        Data source identifier (jupiter, birdeye, pool_implied).
    fetched_at : datetime
        Timestamp when price was fetched.
    liquidity_usd : float | None
        Liquidity backing this price (for confidence calculation).
    payload_hash : str | None
        Hash of raw provider response for audit trail.
    """

    mint: str
    price_usd: float
    price_change_24h_pct: float | None
    confidence: PriceConfidence
    source: str
    fetched_at: datetime
    liquidity_usd: float | None = None
    payload_hash: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "mint": self.mint,
            "price_usd": self.price_usd,
            "price_change_24h_pct": self.price_change_24h_pct,
            "confidence": self.confidence.value,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "liquidity_usd": self.liquidity_usd,
        }


@dataclass
class CacheEntry:
    """Cache entry for price data."""

    data: PriceData
    expires_at: datetime


class PriceProvider(ABC):
    """Abstract base class for price providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @property
    @abstractmethod
    def priority(self) -> int:
        """Provider priority (lower = higher priority)."""
        ...

    @abstractmethod
    async def get_price(self, mint: str) -> PriceData | None:
        """Get price for a single token."""
        ...

    @abstractmethod
    async def get_prices_batch(self, mints: list[str]) -> dict[str, PriceData]:
        """Get prices for multiple tokens."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the provider."""
        ...


def _compute_payload_hash(data: dict) -> str:
    """Compute hash of API response for audit trail."""
    import json
    content = json.dumps(data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _compute_confidence(
    liquidity_usd: float | None,
    source: str,
) -> PriceConfidence:
    """Compute price confidence from liquidity and source.

    High: liquidity > $1M from primary source
    Medium: liquidity > $10K from any source, or fallback source
    Low: liquidity <= $10K or pool-implied
    """
    # Pool-implied is always low confidence
    if source == "pool_implied":
        return PriceConfidence.LOW

    if liquidity_usd is None:
        return PriceConfidence.MEDIUM

    if liquidity_usd >= 1_000_000:
        return PriceConfidence.HIGH
    elif liquidity_usd >= 10_000:
        return PriceConfidence.MEDIUM
    else:
        return PriceConfidence.LOW


class JupiterPriceProvider(PriceProvider):
    """
    Jupiter Price API V3 provider with free/pro tier support.

    Features:
    - Config-driven endpoints (lite-api.jup.ag or api.jup.ag)
    - API key support for pro tier
    - In-memory cache with configurable TTL
    - Rate limiting to stay within API limits
    - Automatic retry with exponential backoff
    """

    DEFAULT_CACHE_TTL = 60  # seconds
    FREE_RATE_LIMIT = 10  # requests per second (lite-api)
    PRO_RATE_LIMIT = 100  # requests per second (api.jup.ag with key)

    def __init__(
        self,
        cache_ttl: int | None = None,
        rate_limit: int | None = None,
    ):
        """Initialize the Jupiter price provider.

        Parameters
        ----------
        cache_ttl : int | None
            Cache TTL in seconds. Defaults to settings or 60s.
        rate_limit : int | None
            Max requests per second. Auto-detected based on API key.
        """
        self._base_url = settings.jupiter_price_base_url
        self._path = settings.jupiter_price_path
        self._api_key = settings.jupiter_api_key

        # Validate config
        if self._api_key and "lite-api" in self._base_url:
            logger.warning(
                "jupiter_config_mismatch",
                msg="API key provided but using lite-api URL. Switch to api.jup.ag for pro tier.",
            )

        self._cache_ttl = cache_ttl or settings.price_cache_ttl_seconds

        # Rate limit based on tier
        if rate_limit:
            self._rate_limit = rate_limit
        elif self._api_key:
            self._rate_limit = self.PRO_RATE_LIMIT
        else:
            self._rate_limit = self.FREE_RATE_LIMIT

        # In-memory cache
        self._cache: dict[str, CacheEntry] = {}

        # Rate limiting state
        self._request_times: list[datetime] = []
        self._rate_lock = asyncio.Lock()

        # HTTP client (lazy initialization)
        self._client: httpx.AsyncClient | None = None

        logger.info(
            "jupiter_provider_initialized",
            base_url=self._base_url,
            has_api_key=bool(self._api_key),
            rate_limit=self._rate_limit,
            cache_ttl=self._cache_ttl,
        )

    @property
    def name(self) -> str:
        return "jupiter"

    @property
    def priority(self) -> int:
        return 1  # Primary provider

    @property
    def api_url(self) -> str:
        """Full API URL."""
        return f"{self._base_url}{self._path}"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {
                "Accept": "application/json",
                "User-Agent": "SHI/1.0",
            }
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers=headers,
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
        """Get price for a single token."""
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
        """Get prices for multiple tokens in a single request."""
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
                provider="jupiter",
                mint_count=len(mints_to_fetch),
                first_mint=mints_to_fetch[0][:8] if mints_to_fetch else None,
            )

            response = await client.get(
                self.api_url,
                params={"ids": ids_param},
            )

            # Handle rate limiting
            if response.status_code == 429:
                logger.warning("jupiter_rate_limited", status=429)
                return result

            response.raise_for_status()

            data = response.json()
            now = datetime.now(timezone.utc)
            payload_hash = _compute_payload_hash(data)

            # Parse V3 response format (direct dict, not nested under "data")
            for mint in mints_to_fetch:
                token_data = data.get(mint)
                if token_data is None:
                    logger.debug("price_not_found", mint=mint[:8])
                    continue

                # V3 uses "usdPrice" instead of "price"
                price = token_data.get("usdPrice")
                if price is None:
                    continue

                # V3 includes 24h price change
                price_change_24h = token_data.get("priceChange24h")

                # V3 includes liquidity
                liquidity = token_data.get("liquidity")

                # Compute confidence from liquidity
                confidence = _compute_confidence(liquidity, "jupiter")

                price_obj = PriceData(
                    mint=mint,
                    price_usd=float(price),
                    price_change_24h_pct=float(price_change_24h) if price_change_24h else None,
                    confidence=confidence,
                    source="jupiter",
                    fetched_at=now,
                    liquidity_usd=float(liquidity) if liquidity else None,
                    payload_hash=payload_hash,
                )

                # Cache and add to result
                self._set_cached(price_obj)
                result[mint] = price_obj

            logger.info(
                "jupiter_prices_fetched",
                requested=len(mints_to_fetch),
                found=len([m for m in mints_to_fetch if m in result]),
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

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def clear_cache(self) -> None:
        """Clear the price cache."""
        self._cache.clear()
        logger.info("jupiter_cache_cleared")

    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class BirdeyePriceProvider(PriceProvider):
    """Birdeye API price provider (fallback)."""

    API_URL = "https://public-api.birdeye.so/defi/price"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or settings.birdeye_api_key
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "birdeye"

    @property
    def priority(self) -> int:
        return 2  # Secondary provider

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Accept": "application/json"}
            if self._api_key:
                headers["X-API-KEY"] = self._api_key
            self._client = httpx.AsyncClient(timeout=30.0, headers=headers)
        return self._client

    async def get_price(self, mint: str) -> PriceData | None:
        """Get price from Birdeye."""
        if not self._api_key:
            logger.debug("birdeye_no_api_key")
            return None

        try:
            client = await self._get_client()
            response = await client.get(
                self.API_URL,
                params={"address": mint},
            )

            if response.status_code == 429:
                logger.warning("birdeye_rate_limited")
                return None

            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                return None

            price_data = data.get("data", {})
            price = price_data.get("value")
            if price is None:
                return None

            liquidity = price_data.get("liquidity")
            confidence = _compute_confidence(liquidity, "birdeye")

            return PriceData(
                mint=mint,
                price_usd=float(price),
                price_change_24h_pct=price_data.get("priceChange24h"),
                confidence=confidence,
                source="birdeye",
                fetched_at=datetime.now(timezone.utc),
                liquidity_usd=float(liquidity) if liquidity else None,
                payload_hash=_compute_payload_hash(data),
            )

        except Exception as e:
            logger.warning("birdeye_fetch_failed", error=str(e))
            return None

    async def get_prices_batch(self, mints: list[str]) -> dict[str, PriceData]:
        """Birdeye doesn't support batch - fetch individually."""
        results = {}
        for mint in mints:
            price = await self.get_price(mint)
            if price:
                results[mint] = price
        return results

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class PoolImpliedPriceProvider(PriceProvider):
    """Derive price from DEX pool reserves (last resort fallback)."""

    # SOL mint for price derivation
    SOL_MINT = "So11111111111111111111111111111111111111112"
    USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

    def __init__(self, liquidity_fetcher: "LiquidityFetcher | None" = None):
        self._liquidity_fetcher = liquidity_fetcher
        self._sol_price: float | None = None
        self._sol_price_fetched_at: datetime | None = None

    @property
    def name(self) -> str:
        return "pool_implied"

    @property
    def priority(self) -> int:
        return 3  # Tertiary provider

    async def _get_sol_price(self, jupiter: JupiterPriceProvider) -> float | None:
        """Get SOL price for conversion."""
        # Cache SOL price for 5 minutes
        if (
            self._sol_price is not None
            and self._sol_price_fetched_at is not None
            and (datetime.now(timezone.utc) - self._sol_price_fetched_at).seconds < 300
        ):
            return self._sol_price

        sol_data = await jupiter.get_price(self.SOL_MINT)
        if sol_data:
            self._sol_price = sol_data.price_usd
            self._sol_price_fetched_at = datetime.now(timezone.utc)
            return self._sol_price

        return None

    async def get_price(
        self,
        mint: str,
        jupiter: JupiterPriceProvider | None = None,
    ) -> PriceData | None:
        """Derive price from pool reserves."""
        if not settings.enable_pool_implied_price:
            return None

        if self._liquidity_fetcher is None:
            from ..liquidity.pools import LiquidityFetcher
            self._liquidity_fetcher = LiquidityFetcher()

        try:
            pool = await self._liquidity_fetcher.get_deepest_pool(mint)
            if not pool:
                return None

            # Determine which side is the token
            if pool.token_a_mint == mint:
                token_reserve = pool.token_a_reserve_ui
                other_reserve = pool.token_b_reserve_ui
                other_mint = pool.token_b_mint
            else:
                token_reserve = pool.token_b_reserve_ui
                other_reserve = pool.token_a_reserve_ui
                other_mint = pool.token_a_mint

            if token_reserve == 0:
                return None

            # Price in terms of other token
            price_in_other = other_reserve / token_reserve

            # Convert to USD
            price_usd: float | None = None

            if other_mint == self.USDC_MINT:
                price_usd = price_in_other
            elif other_mint == self.SOL_MINT and jupiter:
                sol_price = await self._get_sol_price(jupiter)
                if sol_price:
                    price_usd = price_in_other * sol_price

            if price_usd is None:
                return None

            return PriceData(
                mint=mint,
                price_usd=price_usd,
                price_change_24h_pct=None,  # Not available from pools
                confidence=PriceConfidence.LOW,  # Pool-implied is always low
                source="pool_implied",
                fetched_at=datetime.now(timezone.utc),
                liquidity_usd=pool.liquidity_usd,
                payload_hash=None,
            )

        except Exception as e:
            logger.warning("pool_implied_price_failed", mint=mint[:8], error=str(e))
            return None

    async def get_prices_batch(self, mints: list[str]) -> dict[str, PriceData]:
        """Fetch pool-implied prices individually."""
        results = {}
        jupiter = JupiterPriceProvider()
        try:
            for mint in mints:
                price = await self.get_price(mint, jupiter=jupiter)
                if price:
                    results[mint] = price
        finally:
            await jupiter.close()
        return results

    async def close(self) -> None:
        if self._liquidity_fetcher:
            await self._liquidity_fetcher.close()


class PriceProviderChain:
    """
    Chain of price providers with fallback hierarchy.

    Order: Jupiter → Birdeye → Pool-implied
    """

    def __init__(
        self,
        providers: list[PriceProvider] | None = None,
    ):
        if providers:
            self._providers = sorted(providers, key=lambda p: p.priority)
        else:
            # Default chain
            self._providers = [
                JupiterPriceProvider(),
                BirdeyePriceProvider(),
                PoolImpliedPriceProvider(),
            ]

        logger.info(
            "price_provider_chain_initialized",
            providers=[p.name for p in self._providers],
        )

    async def get_price(self, mint: str) -> PriceData | None:
        """Get price, falling through providers until success."""
        for provider in self._providers:
            try:
                price = await provider.get_price(mint)
                if price is not None:
                    logger.debug(
                        "price_fetched_from_provider",
                        mint=mint[:8],
                        provider=provider.name,
                    )
                    return price
            except Exception as e:
                logger.warning(
                    "provider_failed",
                    provider=provider.name,
                    error=str(e),
                )
                continue

        logger.warning("all_price_providers_failed", mint=mint[:8])
        return None

    async def get_prices_batch(
        self,
        mints: list[str],
    ) -> dict[str, PriceData]:
        """Get prices with fallback for missing tokens."""
        results: dict[str, PriceData] = {}
        remaining_mints = list(mints)

        for provider in self._providers:
            if not remaining_mints:
                break

            try:
                prices = await provider.get_prices_batch(remaining_mints)
                results.update(prices)

                # Remove found mints from remaining
                remaining_mints = [m for m in remaining_mints if m not in prices]

                if prices:
                    logger.debug(
                        "batch_prices_from_provider",
                        provider=provider.name,
                        found=len(prices),
                        remaining=len(remaining_mints),
                    )

            except Exception as e:
                logger.warning(
                    "batch_provider_failed",
                    provider=provider.name,
                    error=str(e),
                )
                continue

        return results

    async def close(self) -> None:
        """Close all providers."""
        for provider in self._providers:
            await provider.close()


# Well-known token mints for testing
WELL_KNOWN_MINTS = {
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    "SOL": "So11111111111111111111111111111111111111112",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
}
