"""
DEX Liquidity Pool Fetching.

Supports:
- Raydium AMM pools
- Orca Whirlpools
- Other Solana DEXes

Provides unified interface for liquidity depth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
import structlog

from ..core.types import TokenMint

logger = structlog.get_logger()


@dataclass
class PoolInfo:
    """Information about a liquidity pool."""

    pool_address: str
    dex: str  # raydium, orca, etc.
    token_a_mint: str
    token_b_mint: str
    token_a_reserve: int
    token_b_reserve: int
    token_a_decimals: int
    token_b_decimals: int
    liquidity_usd: float | None
    volume_24h_usd: float | None
    fee_rate: float
    fetched_at: datetime

    @property
    def token_a_reserve_ui(self) -> float:
        """Human-readable reserve for token A."""
        return self.token_a_reserve / (10 ** self.token_a_decimals)

    @property
    def token_b_reserve_ui(self) -> float:
        """Human-readable reserve for token B."""
        return self.token_b_reserve / (10 ** self.token_b_decimals)

    @property
    def price_a_in_b(self) -> float:
        """Price of token A in terms of token B."""
        if self.token_a_reserve == 0:
            return 0
        return self.token_b_reserve_ui / self.token_a_reserve_ui


class DEXProvider(ABC):
    """Abstract base class for DEX data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """DEX name."""
        ...

    @abstractmethod
    async def get_pools(self, token_mint: TokenMint) -> list[PoolInfo]:
        """Get all pools containing the token."""
        ...


class RaydiumProvider(DEXProvider):
    """Raydium AMM pool data provider."""

    API_URL = "https://api.raydium.io/v2"

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "raydium"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def get_pools(self, token_mint: TokenMint) -> list[PoolInfo]:
        """Fetch Raydium pools for a token."""
        client = await self._get_client()

        try:
            # Get AMM pools
            response = await client.get(
                f"{self.API_URL}/ammV3/ammPools",
            )
            response.raise_for_status()
            data = response.json()

            pools = []
            for pool in data.get("data", []):
                # Check if our token is in this pool
                mint_a = pool.get("mintA", {}).get("address", "")
                mint_b = pool.get("mintB", {}).get("address", "")

                if token_mint not in (mint_a, mint_b):
                    continue

                pools.append(PoolInfo(
                    pool_address=pool.get("id", ""),
                    dex=self.name,
                    token_a_mint=mint_a,
                    token_b_mint=mint_b,
                    token_a_reserve=int(pool.get("mintAmountA", 0)),
                    token_b_reserve=int(pool.get("mintAmountB", 0)),
                    token_a_decimals=pool.get("mintA", {}).get("decimals", 9),
                    token_b_decimals=pool.get("mintB", {}).get("decimals", 9),
                    liquidity_usd=pool.get("tvl"),
                    volume_24h_usd=pool.get("day", {}).get("volume"),
                    fee_rate=pool.get("feeRate", 0.0025),
                    fetched_at=datetime.now(timezone.utc),
                ))

            logger.info(
                "raydium_pools_fetched",
                token=token_mint[:8],
                pool_count=len(pools),
            )
            return pools

        except Exception as e:
            logger.error("raydium_fetch_failed", error=str(e))
            return []

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class OrcaProvider(DEXProvider):
    """Orca Whirlpool data provider."""

    API_URL = "https://api.orca.so"

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "orca"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def get_pools(self, token_mint: TokenMint) -> list[PoolInfo]:
        """Fetch Orca pools for a token."""
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.API_URL}/v1/whirlpool/list",
            )
            response.raise_for_status()
            data = response.json()

            pools = []
            for pool in data.get("whirlpools", []):
                mint_a = pool.get("tokenA", {}).get("mint", "")
                mint_b = pool.get("tokenB", {}).get("mint", "")

                if token_mint not in (mint_a, mint_b):
                    continue

                pools.append(PoolInfo(
                    pool_address=pool.get("address", ""),
                    dex=self.name,
                    token_a_mint=mint_a,
                    token_b_mint=mint_b,
                    token_a_reserve=int(pool.get("tokenA", {}).get("amount", 0)),
                    token_b_reserve=int(pool.get("tokenB", {}).get("amount", 0)),
                    token_a_decimals=pool.get("tokenA", {}).get("decimals", 9),
                    token_b_decimals=pool.get("tokenB", {}).get("decimals", 9),
                    liquidity_usd=pool.get("tvl"),
                    volume_24h_usd=pool.get("volume", {}).get("day"),
                    fee_rate=pool.get("lpFeeRate", 0.003),
                    fetched_at=datetime.now(timezone.utc),
                ))

            logger.info(
                "orca_pools_fetched",
                token=token_mint[:8],
                pool_count=len(pools),
            )
            return pools

        except Exception as e:
            logger.error("orca_fetch_failed", error=str(e))
            return []

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class LiquidityFetcher:
    """
    Unified liquidity fetcher across multiple DEXes.

    Aggregates pool data from all supported DEXes.
    """

    def __init__(self):
        self.providers: list[DEXProvider] = [
            RaydiumProvider(),
            OrcaProvider(),
        ]

    async def get_all_pools(self, token_mint: TokenMint) -> list[PoolInfo]:
        """Get pools from all DEXes."""
        all_pools = []

        for provider in self.providers:
            try:
                pools = await provider.get_pools(token_mint)
                all_pools.extend(pools)
            except Exception as e:
                logger.warning(
                    "provider_failed",
                    provider=provider.name,
                    error=str(e),
                )

        # Sort by liquidity (highest first)
        all_pools.sort(
            key=lambda p: p.liquidity_usd or 0,
            reverse=True,
        )

        return all_pools

    async def get_total_liquidity(self, token_mint: TokenMint) -> float:
        """Get total USD liquidity across all pools."""
        pools = await self.get_all_pools(token_mint)
        return sum(p.liquidity_usd or 0 for p in pools)

    async def get_deepest_pool(self, token_mint: TokenMint) -> PoolInfo | None:
        """Get the deepest liquidity pool."""
        pools = await self.get_all_pools(token_mint)
        return pools[0] if pools else None

    async def close(self) -> None:
        """Close all providers."""
        for provider in self.providers:
            await provider.close()
