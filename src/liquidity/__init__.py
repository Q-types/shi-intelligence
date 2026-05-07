"""
Liquidity Integration Module for SHI.

Fetches DEX liquidity data and computes:
- Pool depth
- Price impact estimates
- Liquidity-adjusted sell pressure

Per INITIAL_PROMPT:
Liquidity_Adjusted_Pressure = Sell_Pressure * (1 / Liquidity_Depth_Factor)
Price_Impact ~ Trade_Size / Pool_Liquidity
"""

from .pools import LiquidityFetcher, PoolInfo
from .impact import PriceImpactEstimator, compute_liquidity_adjusted_pressure

__all__ = [
    "LiquidityFetcher",
    "PoolInfo",
    "PriceImpactEstimator",
    "compute_liquidity_adjusted_pressure",
]
