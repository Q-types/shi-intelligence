"""
Weighted Funding Graph Features for SHI.

Extracts features from edge weights (funding amounts):
- total_funding_received
- largest_funder_share
- funding_hhi (concentration index)
- funding_burst_score
- weighted_degree features
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import structlog

from .funding_graph import FundingGraph
from ..core.types import WalletAddress

logger = structlog.get_logger()


@dataclass
class WeightedGraphFeatures:
    """Weighted graph features for a wallet."""

    wallet: WalletAddress

    # Funding amount features
    total_funding_received: float  # Total lamports received
    total_funding_sent: float  # Total lamports sent
    largest_funder_share: float  # % of funding from largest funder (0-1)
    funding_hhi: float  # Herfindahl-Hirschman Index (0-1)

    # Weighted degree features
    weighted_in_degree: float  # Sum of incoming edge weights
    weighted_out_degree: float  # Sum of outgoing edge weights

    # Temporal features
    funding_burst_score: float  # Temporal concentration of funding events

    # Derived ratios
    funding_balance: float  # received - sent
    funding_concentration_ratio: float  # weighted_in / (weighted_in + weighted_out)


def compute_weighted_graph_features(
    graph: FundingGraph,
    wallets: list[WalletAddress],
) -> dict[WalletAddress, WeightedGraphFeatures]:
    """
    Compute weighted graph features for all wallets.

    Args:
        graph: Funding graph with edge weights
        wallets: List of wallet addresses

    Returns:
        Dict mapping wallet -> WeightedGraphFeatures
    """
    results = {}

    for wallet in wallets:
        features = _compute_wallet_weighted_features(graph, wallet)
        if features is not None:
            results[wallet] = features

    logger.info(
        "weighted_features_computed",
        wallet_count=len(results),
    )

    return results


def _compute_wallet_weighted_features(
    graph: FundingGraph,
    wallet: WalletAddress,
) -> Optional[WeightedGraphFeatures]:
    """Compute weighted features for a single wallet."""
    nx_graph = graph._graph

    if wallet not in nx_graph:
        return None

    # Get incoming edges with weights
    in_edges = list(nx_graph.in_edges(wallet, data=True))
    out_edges = list(nx_graph.out_edges(wallet, data=True))

    # Extract amounts (lamports)
    in_amounts = [edge[2].get("amount", 0) for edge in in_edges]
    out_amounts = [edge[2].get("amount", 0) for edge in out_edges]

    # Total funding
    total_received = sum(in_amounts)
    total_sent = sum(out_amounts)

    # Weighted degree (sum of edge weights)
    weighted_in = float(total_received) / 1e9 if total_received > 0 else 0.0  # Convert to SOL
    weighted_out = float(total_sent) / 1e9 if total_sent > 0 else 0.0

    # Largest funder share
    if total_received > 0 and in_amounts:
        largest_funder_share = max(in_amounts) / total_received
    else:
        largest_funder_share = 0.0

    # HHI (Herfindahl-Hirschman Index) for funding concentration
    # HHI = sum of squared market shares
    # HHI = 1 means single funder, HHI approaches 0 means many equal funders
    if total_received > 0 and in_amounts:
        shares = [amt / total_received for amt in in_amounts]
        hhi = sum(s * s for s in shares)
    else:
        hhi = 0.0

    # Funding burst score (temporal concentration)
    burst_score = _compute_funding_burst_score(in_edges)

    # Funding balance and concentration ratio
    funding_balance = weighted_in - weighted_out
    total_weighted = weighted_in + weighted_out
    concentration_ratio = weighted_in / total_weighted if total_weighted > 0 else 0.5

    return WeightedGraphFeatures(
        wallet=wallet,
        total_funding_received=float(total_received) / 1e9,  # SOL
        total_funding_sent=float(total_sent) / 1e9,
        largest_funder_share=largest_funder_share,
        funding_hhi=hhi,
        weighted_in_degree=weighted_in,
        weighted_out_degree=weighted_out,
        funding_burst_score=burst_score,
        funding_balance=funding_balance,
        funding_concentration_ratio=concentration_ratio,
    )


def _compute_funding_burst_score(
    edges: list,
) -> float:
    """
    Compute temporal burstiness of funding events.

    Burstiness B = (sigma - mu) / (sigma + mu)
    Where sigma = std of inter-event times, mu = mean

    B = 1: maximally bursty (all events at once)
    B = 0: Poisson process (random)
    B = -1: perfectly periodic

    Args:
        edges: List of (source, target, data) edge tuples

    Returns:
        Burstiness score in [-1, 1]
    """
    if len(edges) < 2:
        return 0.0

    # Extract timestamps
    timestamps = []
    for edge in edges:
        ts_str = edge[2].get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                timestamps.append(ts)
            except (ValueError, TypeError):
                continue

    if len(timestamps) < 2:
        return 0.0

    # Sort timestamps
    timestamps.sort()

    # Compute inter-event intervals
    intervals = []
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
        intervals.append(delta)

    if not intervals:
        return 0.0

    # Compute burstiness
    intervals_arr = np.array(intervals)
    mu = float(np.mean(intervals_arr))
    sigma = float(np.std(intervals_arr))

    if sigma + mu == 0:
        return 0.0

    burstiness = (sigma - mu) / (sigma + mu)

    return float(burstiness)


def enrich_wallet_features(
    base_features: dict,
    weighted_features: WeightedGraphFeatures,
) -> dict:
    """
    Enrich wallet feature dict with weighted graph features.

    Args:
        base_features: Existing feature dict
        weighted_features: Computed weighted features

    Returns:
        Updated feature dict
    """
    base_features.update({
        "total_funding_received": weighted_features.total_funding_received,
        "largest_funder_share": weighted_features.largest_funder_share,
        "funding_hhi": weighted_features.funding_hhi,
        "funding_burst_score": weighted_features.funding_burst_score,
        "weighted_in_degree": weighted_features.weighted_in_degree,
        "weighted_out_degree": weighted_features.weighted_out_degree,
    })

    return base_features
