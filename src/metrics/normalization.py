"""
Normalization Metrics - IMMUTABLE

WARNING: These formulas are frozen per PDR.
Do not modify without explicit human approval.

Formula defined in PDR Section 4.10.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from scipy import stats

from ..core.types import MetricOutput

# Version for normalization metrics
_VERSION = "1.0.0"


def compute_z_score(
    observed_value: float,
    baseline_mean: float,
    baseline_std: float,
    baseline_version: str,
) -> MetricOutput:
    """
    Z-Score Normalization.

    FROZEN FORMULA (PDR 4.10):
        Z = ( X - mu ) / sigma

    Where:
        X = observed metric value
        mu = baseline mean
        sigma = baseline standard deviation

    Args:
        observed_value: The metric value to normalize
        baseline_mean: Mean from baseline reference dataset
        baseline_std: Standard deviation from baseline
        baseline_version: Version identifier for the baseline dataset

    Returns:
        MetricOutput with z-score
        - Positive = above baseline mean
        - Negative = below baseline mean
        - |Z| > 2 = unusual (outside ~95% of baseline)
        - |Z| > 3 = very unusual (outside ~99.7% of baseline)
    """
    if baseline_std <= 0:
        raise ValueError("Baseline standard deviation must be positive")

    z = (observed_value - baseline_mean) / baseline_std

    return MetricOutput(
        metric_name="z_score",
        value=z,
        version=_VERSION,
        computed_at=datetime.now(timezone.utc),
        baseline_version=baseline_version,
    )


def compute_percentile(
    observed_value: float,
    baseline_values: Sequence[float],
    baseline_version: str,
) -> MetricOutput:
    """
    Percentile vs Baseline.

    Computes the percentile rank of the observed value within the
    baseline reference distribution.

    Args:
        observed_value: The metric value to rank
        baseline_values: Array of values from baseline dataset
        baseline_version: Version identifier for the baseline dataset

    Returns:
        MetricOutput with percentile in [0, 100]
        - 50 = median of baseline
        - 95 = higher than 95% of baseline
        - 5 = lower than 95% of baseline
    """
    if not baseline_values:
        raise ValueError("Baseline values cannot be empty")

    # Compute percentile rank
    percentile = stats.percentileofscore(baseline_values, observed_value)

    return MetricOutput(
        metric_name="percentile",
        value=percentile,
        version=_VERSION,
        computed_at=datetime.now(timezone.utc),
        baseline_version=baseline_version,
    )


class BaselineStatistics:
    """
    Container for baseline dataset statistics.

    Used for z-score normalization of all metrics.
    Must be versioned and persisted.
    """

    def __init__(
        self,
        version: str,
        hhi_mean: float,
        hhi_std: float,
        entropy_mean: float,
        entropy_std: float,
        gini_mean: float,
        gini_std: float,
        wdr_mean: float,
        wdr_std: float,
        churn_mean: float,
        churn_std: float,
        coordination_mean: float,
        coordination_std: float,
    ):
        self.version = version
        self.stats = {
            "hhi": {"mean": hhi_mean, "std": hhi_std},
            "shannon_entropy": {"mean": entropy_mean, "std": entropy_std},
            "gini_coefficient": {"mean": gini_mean, "std": gini_std},
            "whale_dominance_ratio": {"mean": wdr_mean, "std": wdr_std},
            "churn_rate": {"mean": churn_mean, "std": churn_std},
            "coordination_score": {"mean": coordination_mean, "std": coordination_std},
        }

    def normalize(self, metric_name: str, observed_value: float) -> MetricOutput:
        """Compute z-score for a metric using stored baseline."""
        if metric_name not in self.stats:
            raise ValueError(f"Unknown metric: {metric_name}")

        stat = self.stats[metric_name]
        return compute_z_score(
            observed_value=observed_value,
            baseline_mean=stat["mean"],
            baseline_std=stat["std"],
            baseline_version=self.version,
        )
