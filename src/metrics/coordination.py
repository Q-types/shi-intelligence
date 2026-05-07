"""
Coordination Metrics - IMMUTABLE

WARNING: These formulas are frozen per PDR.
Do not modify without explicit human approval.

Formulas defined in PDR Sections 4.5-4.7.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence, Set

from ..core.types import MetricOutput, WalletAddress

# Version for coordination metrics
_VERSION = "1.0.0"


def compute_churn_rate(
    wallets_at_start: int,
    wallets_exited: int,
) -> MetricOutput:
    """
    Churn Rate.

    FROZEN FORMULA (PDR 4.5):
        Churn = Wallets_Exited_in_Window / Wallets_At_Window_Start

    Args:
        wallets_at_start: Number of holders at window start
        wallets_exited: Number of holders who exited during window

    Returns:
        MetricOutput with churn rate in [0, 1]
        - Higher = more turnover
    """
    if wallets_at_start <= 0:
        raise ValueError("Wallets at start must be positive")

    if wallets_exited < 0:
        raise ValueError("Wallets exited cannot be negative")

    if wallets_exited > wallets_at_start:
        raise ValueError("Wallets exited cannot exceed wallets at start")

    churn = wallets_exited / wallets_at_start

    return MetricOutput(
        metric_name="churn_rate",
        value=churn,
        version=_VERSION,
        computed_at=datetime.now(timezone.utc),
    )


def compute_coordination_score(
    cluster_wallets: Sequence[WalletAddress],
    wallets_sharing_top_funder: Set[WalletAddress],
) -> MetricOutput:
    """
    Coordination Score (Cluster-Level).

    FROZEN FORMULA (PDR 4.6):
        Coord(C) = Shared_Funder_Count / Size_of_Cluster

    Where:
        Shared_Funder_Count = number of wallets in cluster sharing dominant upstream funder

    Args:
        cluster_wallets: All wallets in the cluster
        wallets_sharing_top_funder: Wallets that share the dominant funder

    Returns:
        MetricOutput with coordination score in [0, 1]
        - Higher = more coordinated (same funding source)
    """
    if not cluster_wallets:
        raise ValueError("Cluster cannot be empty")

    cluster_size = len(cluster_wallets)
    cluster_set = set(cluster_wallets)

    # Count how many cluster wallets share the top funder
    shared_count = len(wallets_sharing_top_funder & cluster_set)

    coord = shared_count / cluster_size

    return MetricOutput(
        metric_name="coordination_score",
        value=coord,
        version=_VERSION,
        computed_at=datetime.now(timezone.utc),
    )


def compute_funding_density(
    num_vertices: int,
    num_edges: int,
) -> MetricOutput:
    """
    Funding Density.

    FROZEN FORMULA (PDR 4.7):
        Funding_Density = |E| / ( |V| * (|V| - 1) )

    Where:
        |E| = number of funding edges
        |V| = number of wallet vertices

    This is the density of the directed funding graph.

    Args:
        num_vertices: Number of wallet nodes
        num_edges: Number of funding edges

    Returns:
        MetricOutput with density in [0, 1]
        - Higher = more interconnected funding
    """
    if num_vertices < 2:
        raise ValueError("Need at least 2 vertices for density calculation")

    if num_edges < 0:
        raise ValueError("Number of edges cannot be negative")

    max_edges = num_vertices * (num_vertices - 1)  # Directed graph
    density = num_edges / max_edges

    return MetricOutput(
        metric_name="funding_density",
        value=density,
        version=_VERSION,
        computed_at=datetime.now(timezone.utc),
    )
