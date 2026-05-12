"""
Unified Solana Data Client.

Provides a single interface with:
- Provider failover
- Caching
- Rate limit handling
- Query budgeting
- Provenance logging
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator

import structlog

from .providers import DataProvider, HeliusProvider, RPCProvider, RateLimitError
from .cache import QueryCache
from ..core.types import (
    TokenMint,
    WalletAddress,
    TokenBalance,
    WalletMetadata,
    FundingEdge,
    HolderSnapshot,
)
from ..core.config import settings

logger = structlog.get_logger()


class SolanaDataClient:
    """
    Unified data client with failover and caching.

    Handles:
    - Primary provider (Helius) with fallback (RPC)
    - Query caching
    - Rate limit backoff
    - Budget tracking
    """

    def __init__(
        self,
        primary: DataProvider | None = None,
        fallback: DataProvider | None = None,
        cache: QueryCache | None = None,
    ):
        self.primary = primary or HeliusProvider()
        self.fallback = fallback or RPCProvider()
        self.cache = cache or QueryCache()

        self._query_budget = 10000  # Queries per hour
        self._queries_this_hour = 0
        self._hour_start = datetime.now(timezone.utc)

    async def get_token_holders(
        self,
        mint: TokenMint,
        *,
        limit: int | None = None,
        use_cache: bool = True,
    ) -> HolderSnapshot:
        """
        Fetch token holders with caching and failover.

        Args:
            mint: Token mint address
            limit: Optional limit (samples if exceeded)
            use_cache: Whether to check cache first

        Returns:
            HolderSnapshot with all holder data
        """
        cache_key = f"holders:{mint}:{limit or 'all'}"

        # Check cache
        if use_cache:
            cached = await self.cache.get(cache_key)
            if cached:
                logger.info("cache_hit", key=cache_key)
                return cached

        # Budget check
        self._check_budget()

        # Try primary provider
        try:
            snapshot = await self.primary.get_token_holders(mint, limit=limit)
            await self.cache.set(cache_key, snapshot, ttl=300)  # 5 min cache
            return snapshot

        except RateLimitError as e:
            logger.warning(
                "rate_limited_primary",
                provider=self.primary.name,
                retry_after=e.retry_after,
            )
            await asyncio.sleep(min(e.retry_after, 5))

        except Exception as e:
            logger.error(
                "primary_provider_failed",
                provider=self.primary.name,
                error=str(e),
            )

        # Fallback
        logger.info("using_fallback", provider=self.fallback.name)
        snapshot = await self.fallback.get_token_holders(mint, limit=limit)
        await self.cache.set(cache_key, snapshot, ttl=300)
        return snapshot

    async def get_wallet_metadata(
        self,
        wallet: WalletAddress,
        use_cache: bool = True,
    ) -> WalletMetadata:
        """Fetch wallet metadata with caching."""
        cache_key = f"wallet:{wallet}"

        if use_cache:
            cached = await self.cache.get(cache_key)
            if cached:
                return cached

        self._check_budget()

        try:
            metadata = await self.primary.get_wallet_metadata(wallet)
            await self.cache.set(cache_key, metadata, ttl=3600)  # 1 hour cache
            return metadata

        except Exception as e:
            logger.warning("primary_failed", error=str(e))
            return await self.fallback.get_wallet_metadata(wallet)

    async def get_funding_edges(
        self,
        wallets: list[WalletAddress],
    ) -> list[FundingEdge]:
        """Fetch funding relationships for wallets."""
        self._check_budget()

        edges = []
        try:
            async for edge in self.primary.get_funding_edges(wallets):
                edges.append(edge)
        except Exception as e:
            logger.warning("funding_edges_failed", error=str(e))
            async for edge in self.fallback.get_funding_edges(wallets):
                edges.append(edge)

        return edges

    async def get_historical_balances(
        self,
        wallet: WalletAddress,
        mint: TokenMint,
        *,
        since: datetime | None = None,
    ) -> list[TokenBalance]:
        """Fetch historical balance snapshots."""
        self._check_budget()

        try:
            return await self.primary.get_historical_balances(wallet, mint, since=since)
        except Exception as e:
            logger.warning("historical_failed", error=str(e))
            return await self.fallback.get_historical_balances(wallet, mint, since=since)

    def _check_budget(self) -> None:
        """Check and update query budget."""
        now = datetime.now(timezone.utc)

        # Reset hourly counter
        if (now - self._hour_start).total_seconds() > 3600:
            self._hour_start = now
            self._queries_this_hour = 0

        self._queries_this_hour += 1

        if self._queries_this_hour > self._query_budget:
            logger.warning(
                "query_budget_exceeded",
                used=self._queries_this_hour,
                budget=self._query_budget,
            )

    async def close(self) -> None:
        """Close all providers."""
        await self.primary.close()
        await self.fallback.close()
        await self.cache.close()
