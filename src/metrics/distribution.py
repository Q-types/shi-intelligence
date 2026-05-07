"""
Distribution Metrics - IMMUTABLE

WARNING: These formulas are frozen per PDR.
Do not modify without explicit human approval.

Formulas defined in PDR Section 4.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Sequence

from ..core.types import MetricOutput

# Version for all distribution metrics
_VERSION = "1.0.0"


def compute_hhi(shares: Sequence[float]) -> MetricOutput:
    """
    Herfindahl-Hirschman Index (HHI).

    FROZEN FORMULA (PDR 4.1):
        HHI = SUM( s_i^2 )

    Where:
        s_i = b_i / SUM(b_j)  (share of supply for wallet i)

    Args:
        shares: List of ownership shares (must sum to ~1.0)

    Returns:
        MetricOutput with HHI value in [0, 1]
        - 0 = perfectly distributed
        - 1 = single holder owns everything

    Note:
        HHI is scale-invariant. Input shares should be normalized.
    """
    if not shares:
        raise ValueError("Cannot compute HHI for empty holder set")

    # Validate shares are reasonable
    total = sum(shares)
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"Shares must sum to ~1.0, got {total}")

    hhi = sum(s**2 for s in shares)

    return MetricOutput(
        metric_name="hhi",
        value=hhi,
        version=_VERSION,
        computed_at=datetime.now(timezone.utc),
    )


def compute_shannon_entropy(shares: Sequence[float]) -> MetricOutput:
    """
    Shannon Entropy.

    FROZEN FORMULA (PDR 4.2):
        H = - SUM( s_i * log(s_i) )

    Where:
        s_i = share of supply for wallet i
        log = natural logarithm

    Args:
        shares: List of ownership shares (must sum to ~1.0)

    Returns:
        MetricOutput with entropy value
        - Higher values = more distributed
        - Lower values = more concentrated
        - Maximum = log(N) for N holders with equal shares

    Note:
        Wallets with zero balance are excluded from calculation.
    """
    if not shares:
        raise ValueError("Cannot compute entropy for empty holder set")

    # Filter out zero shares (log(0) undefined)
    nonzero_shares = [s for s in shares if s > 0]

    if not nonzero_shares:
        raise ValueError("All shares are zero")

    entropy = -sum(s * math.log(s) for s in nonzero_shares)

    return MetricOutput(
        metric_name="shannon_entropy",
        value=entropy,
        version=_VERSION,
        computed_at=datetime.now(timezone.utc),
    )


def compute_gini_coefficient(balances: Sequence[float]) -> MetricOutput:
    """
    Gini Coefficient.

    FROZEN FORMULA (PDR 4.3):
        G = ( SUM_i SUM_j |b_i - b_j| ) / ( 2 * N * SUM(b_i) )

    Where:
        N = number of wallets
        b_i = balance of wallet i

    Args:
        balances: List of wallet balances (raw amounts, not shares)

    Returns:
        MetricOutput with Gini in [0, 1]
        - 0 = perfect equality (all same balance)
        - 1 = perfect inequality (one holder has everything)
    """
    if not balances:
        raise ValueError("Cannot compute Gini for empty holder set")

    n = len(balances)
    total = sum(balances)

    if total == 0:
        raise ValueError("Total balance is zero")

    # Compute sum of absolute differences
    abs_diff_sum = sum(abs(bi - bj) for bi in balances for bj in balances)

    gini = abs_diff_sum / (2 * n * total)

    return MetricOutput(
        metric_name="gini_coefficient",
        value=gini,
        version=_VERSION,
        computed_at=datetime.now(timezone.utc),
    )


def compute_whale_dominance_ratio(
    balances: Sequence[float],
    total_supply: float,
    k: int = 10,
) -> MetricOutput:
    """
    Whale Dominance Ratio (WDR).

    FROZEN FORMULA (PDR 4.4):
        WDR = ( SUM_{i=1 to k} b_i ) / Total_Supply

    Where:
        k = number of top wallets (default: 10)
        b_i = balance of wallet i (sorted descending)

    Args:
        balances: List of wallet balances (raw amounts)
        total_supply: Total token supply
        k: Number of top holders to consider

    Returns:
        MetricOutput with WDR in [0, 1]
        - Higher = more whale dominated
    """
    if not balances:
        raise ValueError("Cannot compute WDR for empty holder set")

    if total_supply <= 0:
        raise ValueError("Total supply must be positive")

    # Sort descending and take top k
    sorted_balances = sorted(balances, reverse=True)
    top_k = sorted_balances[:k]

    wdr = sum(top_k) / total_supply

    return MetricOutput(
        metric_name="whale_dominance_ratio",
        value=wdr,
        version=_VERSION,
        computed_at=datetime.now(timezone.utc),
    )
