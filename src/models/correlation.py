"""
Cluster Correlation Adjustment.

Per INITIAL_PROMPT:
For cluster C:
    Cluster_P_sell = 1 - PRODUCT(1 - P_sell_i)

If cluster coordination score exceeds threshold:
- Apply correlation amplification factor
- Log coordination-adjusted pressure
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import structlog


logger = structlog.get_logger()


@dataclass
class ClusterSellProbability:
    """Sell probability for a coordinated cluster."""

    cluster_id: int
    wallet_count: int
    individual_probs: list[float]
    independent_cluster_prob: float  # Assuming independence
    correlated_cluster_prob: float  # With correlation adjustment
    coordination_score: float
    correlation_factor: float
    is_coordinated: bool


@dataclass
class CorrelationAdjustedPressure:
    """Correlation-adjusted sell pressure for a token."""

    raw_sell_pressure: float
    correlation_adjusted_pressure: float
    cluster_count: int
    coordinated_cluster_count: int
    adjustment_factor: float
    cluster_details: list[ClusterSellProbability]
    computed_at: datetime


class ClusterCorrelationAdjuster:
    """
    Adjusts sell probabilities for correlated clusters.

    Per INITIAL_PROMPT:
    - Coordinated wallets may sell together
    - Apply correlation amplification when coordination score exceeds threshold
    """

    def __init__(
        self,
        coordination_threshold: float = 0.5,
        correlation_amplification: float = 1.5,
    ):
        self.coordination_threshold = coordination_threshold
        self.correlation_amplification = correlation_amplification

    def compute_cluster_probability(
        self,
        cluster_id: int,
        individual_sell_probs: Sequence[float],
        coordination_score: float,
    ) -> ClusterSellProbability:
        """
        Compute sell probability for a cluster.

        Per INITIAL_PROMPT formula:
        Cluster_P_sell = 1 - PRODUCT(1 - P_sell_i)

        Args:
            cluster_id: Cluster identifier
            individual_sell_probs: P_sell for each wallet in cluster
            coordination_score: Cluster coordination score [0, 1]

        Returns:
            ClusterSellProbability with independent and correlated estimates
        """
        if not individual_sell_probs:
            return ClusterSellProbability(
                cluster_id=cluster_id,
                wallet_count=0,
                individual_probs=[],
                independent_cluster_prob=0.0,
                correlated_cluster_prob=0.0,
                coordination_score=coordination_score,
                correlation_factor=1.0,
                is_coordinated=False,
            )

        probs = list(individual_sell_probs)

        # Independent assumption: P(at least one sells) = 1 - P(all hold)
        survival_product = 1.0
        for p in probs:
            survival_product *= (1.0 - p)
        independent_prob = 1.0 - survival_product

        # Determine if cluster is coordinated
        is_coordinated = coordination_score >= self.coordination_threshold

        # Apply correlation adjustment for coordinated clusters
        if is_coordinated:
            # Amplify the probability based on coordination
            # Higher coordination = more likely to act together
            correlation_factor = 1.0 + (
                (coordination_score - self.coordination_threshold)
                / (1.0 - self.coordination_threshold)
                * (self.correlation_amplification - 1.0)
            )

            # Apply to probability (capped at 1.0)
            correlated_prob = min(1.0, independent_prob * correlation_factor)
        else:
            correlation_factor = 1.0
            correlated_prob = independent_prob

        return ClusterSellProbability(
            cluster_id=cluster_id,
            wallet_count=len(probs),
            individual_probs=probs,
            independent_cluster_prob=independent_prob,
            correlated_cluster_prob=correlated_prob,
            coordination_score=coordination_score,
            correlation_factor=correlation_factor,
            is_coordinated=is_coordinated,
        )

    def compute_total_adjusted_pressure(
        self,
        clusters: list[tuple[int, list[float], float]],  # (id, probs, coord_score)
        non_cluster_probs: list[float],
    ) -> CorrelationAdjustedPressure:
        """
        Compute correlation-adjusted total sell pressure.

        Args:
            clusters: List of (cluster_id, individual_probs, coordination_score)
            non_cluster_probs: Sell probs for non-clustered wallets

        Returns:
            CorrelationAdjustedPressure with full adjustment
        """
        cluster_results = []
        coordinated_count = 0

        # Process clusters
        for cluster_id, probs, coord_score in clusters:
            result = self.compute_cluster_probability(
                cluster_id, probs, coord_score
            )
            cluster_results.append(result)
            if result.is_coordinated:
                coordinated_count += 1

        # Raw pressure: sum of all individual probabilities
        raw_pressure = sum(non_cluster_probs)
        for cluster_id, probs, _ in clusters:
            raw_pressure += sum(probs)

        # Adjusted pressure: use correlated probabilities for clusters
        adjusted_pressure = sum(non_cluster_probs)
        for result in cluster_results:
            # For clusters, use the correlated probability instead of sum
            # This accounts for the fact that coordinated wallets may act together
            adjusted_pressure += result.correlated_cluster_prob

        # Compute overall adjustment factor
        adjustment_factor = (
            adjusted_pressure / raw_pressure if raw_pressure > 0 else 1.0
        )

        logger.info(
            "correlation_adjusted_pressure",
            raw=raw_pressure,
            adjusted=adjusted_pressure,
            factor=adjustment_factor,
            clusters=len(clusters),
            coordinated=coordinated_count,
        )

        return CorrelationAdjustedPressure(
            raw_sell_pressure=raw_pressure,
            correlation_adjusted_pressure=adjusted_pressure,
            cluster_count=len(clusters),
            coordinated_cluster_count=coordinated_count,
            adjustment_factor=adjustment_factor,
            cluster_details=cluster_results,
            computed_at=datetime.now(timezone.utc),
        )


def compute_pairwise_correlation(
    wallet_features: dict[str, list[float]],
) -> dict[tuple[str, str], float]:
    """
    Compute pairwise correlation between wallet behaviors.

    Uses feature similarity to estimate correlation.
    """
    wallets = list(wallet_features.keys())
    correlations = {}

    for i, w1 in enumerate(wallets):
        for w2 in wallets[i + 1:]:
            f1 = np.array(wallet_features[w1])
            f2 = np.array(wallet_features[w2])

            # Pearson correlation
            if len(f1) > 0 and np.std(f1) > 0 and np.std(f2) > 0:
                corr = np.corrcoef(f1, f2)[0, 1]
                correlations[(w1, w2)] = float(corr) if not np.isnan(corr) else 0.0
            else:
                correlations[(w1, w2)] = 0.0

    return correlations


def detect_coordinated_behavior(
    entry_times: list[datetime],
    temporal_window_hours: float = 24.0,
) -> float:
    """
    Detect temporal coordination in entry times.

    Returns synchronicity score [0, 1].
    """
    if len(entry_times) < 2:
        return 0.0

    # Convert to timestamps
    timestamps = sorted([t.timestamp() for t in entry_times])

    # Compute inter-arrival times
    intervals = []
    for i in range(1, len(timestamps)):
        intervals.append(timestamps[i] - timestamps[i - 1])

    if not intervals:
        return 0.0

    # Compute coefficient of variation
    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))

    if mean_interval == 0:
        return 1.0  # All entries at same time = perfect sync

    cv = std_interval / mean_interval

    # Lower CV = more synchronized
    # Convert to [0, 1] score where 1 = highly synchronized
    window_seconds = float(temporal_window_hours * 3600)

    if mean_interval < window_seconds:
        # Entries are within the window
        sync_score = 1.0 - min(1.0, cv)
    else:
        # Entries spread beyond window
        sync_score = max(0.0, 1.0 - (mean_interval / window_seconds))

    return sync_score
