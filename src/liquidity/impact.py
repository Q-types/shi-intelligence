"""
Price Impact Estimation.

Per INITIAL_PROMPT:
Price_Impact ~ Trade_Size / Pool_Liquidity
Liquidity_Adjusted_Pressure = Sell_Pressure * (1 / Liquidity_Depth_Factor)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

import structlog

from .pools import PoolInfo, LiquidityFetcher
from ..core.types import TokenMint

logger = structlog.get_logger()


@dataclass
class PriceImpactResult:
    """Result of price impact estimation."""

    trade_size_tokens: float
    trade_size_usd: float | None
    estimated_impact_pct: float
    pool_liquidity_usd: float | None
    pool_address: str
    dex: str
    computed_at: datetime


@dataclass
class LiquidityAdjustedPressure:
    """Liquidity-adjusted sell pressure result."""

    raw_sell_pressure: float
    liquidity_depth_factor: float
    adjusted_pressure: float
    total_liquidity_usd: float
    pool_count: int
    computed_at: datetime


class PriceImpactEstimator:
    """
    Estimates price impact for trades.

    Uses constant product AMM formula:
    For trade of size dx in pool with reserves (x, y):
    - dy = y * dx / (x + dx)
    - price_impact = 1 - (dy/dx) / (y/x)

    Simplified approximation per INITIAL_PROMPT:
    Price_Impact ~ Trade_Size / Pool_Liquidity
    """

    def __init__(self, fetcher: LiquidityFetcher | None = None):
        self.fetcher = fetcher or LiquidityFetcher()

    async def estimate_impact(
        self,
        token_mint: TokenMint,
        trade_size_tokens: float,
        token_decimals: int = 9,
    ) -> PriceImpactResult | None:
        """
        Estimate price impact for a token sale.

        Args:
            token_mint: Token to sell
            trade_size_tokens: Amount of tokens to sell (UI amount)
            token_decimals: Token decimals

        Returns:
            PriceImpactResult or None if no pools found
        """
        pool = await self.fetcher.get_deepest_pool(token_mint)

        if pool is None:
            logger.warning("no_pools_found", token=token_mint[:8])
            return None

        # Determine which side of the pool is our token
        if pool.token_a_mint == token_mint:
            reserve_tokens = pool.token_a_reserve_ui
            reserve_quote = pool.token_b_reserve_ui
        else:
            reserve_tokens = pool.token_b_reserve_ui
            reserve_quote = pool.token_a_reserve_ui

        if reserve_tokens == 0:
            return None

        # Constant product formula: x * y = k
        # After selling dx tokens: (x + dx) * (y - dy) = k
        # dy = y * dx / (x + dx)
        dx = trade_size_tokens
        x = reserve_tokens
        y = reserve_quote

        dy = y * dx / (x + dx)

        # Initial price: y/x
        # Effective price: dy/dx
        initial_price = y / x
        effective_price = dy / dx

        # Price impact = 1 - (effective / initial)
        impact_pct = 1 - (effective_price / initial_price)

        # Estimate USD value
        trade_usd = None
        if pool.liquidity_usd:
            # Rough estimate: token value = half of pool TVL / reserve
            token_price_usd = (pool.liquidity_usd / 2) / reserve_tokens
            trade_usd = trade_size_tokens * token_price_usd

        return PriceImpactResult(
            trade_size_tokens=trade_size_tokens,
            trade_size_usd=trade_usd,
            estimated_impact_pct=impact_pct,
            pool_liquidity_usd=pool.liquidity_usd,
            pool_address=pool.pool_address,
            dex=pool.dex,
            computed_at=datetime.now(timezone.utc),
        )

    async def estimate_batch_impact(
        self,
        token_mint: TokenMint,
        trade_sizes: list[float],
    ) -> list[PriceImpactResult | None]:
        """Estimate impact for multiple trade sizes."""
        results = []
        for size in trade_sizes:
            result = await self.estimate_impact(token_mint, size)
            results.append(result)
        return results


def compute_liquidity_depth_factor(
    total_liquidity_usd: float,
    scale_factor: float = 1_000_000,  # $1M reference
) -> float:
    """
    Compute liquidity depth factor for pressure adjustment.

    Uses log scale to handle wide range of liquidity values.

    Args:
        total_liquidity_usd: Total liquidity in USD
        scale_factor: Reference liquidity level

    Returns:
        Depth factor in (0, 1] range
        - Higher liquidity = factor closer to 1
        - Lower liquidity = factor closer to 0
    """
    if total_liquidity_usd <= 0:
        return 0.1  # Minimum factor for no liquidity

    # Log scale normalization
    # At $1M liquidity, factor = 1.0
    # At $10K liquidity, factor ≈ 0.67
    # At $1K liquidity, factor ≈ 0.5
    log_liquidity = math.log10(total_liquidity_usd + 1)
    log_scale = math.log10(scale_factor)

    factor = log_liquidity / log_scale

    # Clamp to reasonable range
    return max(0.1, min(1.0, factor))


async def compute_liquidity_adjusted_pressure(
    token_mint: TokenMint,
    raw_sell_pressure: float,
    fetcher: LiquidityFetcher | None = None,
) -> LiquidityAdjustedPressure:
    """
    Compute liquidity-adjusted sell pressure.

    Per INITIAL_PROMPT:
    Liquidity_Adjusted_Pressure = Sell_Pressure * (1 / Liquidity_Depth_Factor)

    Args:
        token_mint: Token to analyze
        raw_sell_pressure: Unadjusted sell pressure index
        fetcher: Optional liquidity fetcher

    Returns:
        LiquidityAdjustedPressure with adjustment details
    """
    fetcher = fetcher or LiquidityFetcher()

    pools = await fetcher.get_all_pools(token_mint)
    total_liquidity = sum(p.liquidity_usd or 0 for p in pools)

    depth_factor = compute_liquidity_depth_factor(total_liquidity)

    # Apply adjustment: lower liquidity = higher adjusted pressure
    adjusted = raw_sell_pressure / depth_factor

    logger.info(
        "liquidity_adjusted_pressure",
        token=token_mint[:8],
        raw_pressure=raw_sell_pressure,
        liquidity_usd=total_liquidity,
        depth_factor=depth_factor,
        adjusted_pressure=adjusted,
    )

    return LiquidityAdjustedPressure(
        raw_sell_pressure=raw_sell_pressure,
        liquidity_depth_factor=depth_factor,
        adjusted_pressure=adjusted,
        total_liquidity_usd=total_liquidity,
        pool_count=len(pools),
        computed_at=datetime.now(timezone.utc),
    )


def estimate_whale_impact(
    whale_balances: list[float],
    pool_reserve: float,
) -> list[float]:
    """
    Estimate price impact if each whale were to sell their full position.

    Args:
        whale_balances: List of whale token balances (UI amounts)
        pool_reserve: Pool's token reserve (UI amount)

    Returns:
        List of estimated impact percentages
    """
    impacts = []

    for balance in whale_balances:
        if pool_reserve == 0:
            impacts.append(1.0)  # 100% impact
            continue

        # Simplified impact: balance / (reserve + balance)
        impact = balance / (pool_reserve + balance)
        impacts.append(impact)

    return impacts
