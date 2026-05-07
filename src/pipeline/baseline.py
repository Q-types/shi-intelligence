"""
Baseline Dataset Governance.

Manages reference datasets for z-score normalization per INITIAL_PROMPT:
- Versioned reference dataset
- Classes: established, high-liquidity, known rugs
- Minimum sample size per class
- Monthly recalibration
- Drift detection
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence

import numpy as np
import structlog
from scipy import stats

logger = structlog.get_logger()


class BaselineClass(Enum):
    """Reference dataset classes per INITIAL_PROMPT."""

    ESTABLISHED = "established"  # Known stable tokens
    HIGH_LIQUIDITY = "high_liquidity"  # Blue-chip Solana tokens
    KNOWN_RUG = "known_rug"  # Historical rug pulls


@dataclass
class BaselineDataPoint:
    """Single token's metrics for baseline."""

    mint: str
    baseline_class: BaselineClass
    hhi: float
    entropy: float
    gini: float
    wdr: float
    churn: float
    coordination: float
    recorded_at: datetime


@dataclass
class BaselineVersion:
    """Versioned baseline statistics."""

    version: str
    created_at: datetime
    sample_counts: dict[str, int]  # class -> count

    # Statistics per metric (aggregated across classes)
    hhi_mean: float
    hhi_std: float
    entropy_mean: float
    entropy_std: float
    gini_mean: float
    gini_std: float
    wdr_mean: float
    wdr_std: float
    churn_mean: float
    churn_std: float
    coordination_mean: float
    coordination_std: float

    # Per-class statistics for comparison
    class_statistics: dict[str, dict[str, tuple[float, float]]]  # class -> metric -> (mean, std)


class BaselineGovernance:
    """
    Manages baseline datasets and drift detection.

    Per INITIAL_PROMPT requirements:
    - Minimum sample size per class
    - Monthly recalibration
    - Drift detection
    - Version tracking
    """

    MIN_SAMPLES_PER_CLASS = 50
    DRIFT_THRESHOLD_ZSCORE = 2.0  # Alert if metric mean shifts > 2 std

    def __init__(self):
        self._current_baseline: BaselineVersion | None = None
        self._data_points: list[BaselineDataPoint] = []

    def add_data_point(self, point: BaselineDataPoint) -> None:
        """Add a token to the baseline dataset."""
        self._data_points.append(point)
        logger.debug(
            "baseline_point_added",
            mint=point.mint,
            baseline_class=point.baseline_class.value,
        )

    def compute_baseline(self, version: str) -> BaselineVersion:
        """
        Compute baseline statistics from collected data points.

        Args:
            version: Version string for this baseline

        Returns:
            BaselineVersion with computed statistics
        """
        logger.info("computing_baseline", version=version, points=len(self._data_points))

        # Check minimum samples per class
        class_counts = {}
        for bp in BaselineClass:
            count = sum(1 for p in self._data_points if p.baseline_class == bp)
            class_counts[bp.value] = count
            if count < self.MIN_SAMPLES_PER_CLASS:
                logger.warning(
                    "insufficient_samples",
                    baseline_class=bp.value,
                    count=count,
                    required=self.MIN_SAMPLES_PER_CLASS,
                )

        # Extract metrics arrays
        metrics = {
            "hhi": [p.hhi for p in self._data_points],
            "entropy": [p.entropy for p in self._data_points],
            "gini": [p.gini for p in self._data_points],
            "wdr": [p.wdr for p in self._data_points],
            "churn": [p.churn for p in self._data_points],
            "coordination": [p.coordination for p in self._data_points],
        }

        # Compute overall statistics
        def safe_stats(values: list[float]) -> tuple[float, float]:
            if not values:
                return 0.0, 1.0
            return float(np.mean(values)), float(np.std(values)) or 1.0

        overall = {name: safe_stats(vals) for name, vals in metrics.items()}

        # Compute per-class statistics
        class_stats: dict[str, dict[str, tuple[float, float]]] = {}
        for bp in BaselineClass:
            class_points = [p for p in self._data_points if p.baseline_class == bp]
            if class_points:
                class_metrics = {
                    "hhi": [p.hhi for p in class_points],
                    "entropy": [p.entropy for p in class_points],
                    "gini": [p.gini for p in class_points],
                    "wdr": [p.wdr for p in class_points],
                    "churn": [p.churn for p in class_points],
                    "coordination": [p.coordination for p in class_points],
                }
                class_stats[bp.value] = {
                    name: safe_stats(vals) for name, vals in class_metrics.items()
                }

        baseline = BaselineVersion(
            version=version,
            created_at=datetime.now(timezone.utc),
            sample_counts=class_counts,
            hhi_mean=overall["hhi"][0],
            hhi_std=overall["hhi"][1],
            entropy_mean=overall["entropy"][0],
            entropy_std=overall["entropy"][1],
            gini_mean=overall["gini"][0],
            gini_std=overall["gini"][1],
            wdr_mean=overall["wdr"][0],
            wdr_std=overall["wdr"][1],
            churn_mean=overall["churn"][0],
            churn_std=overall["churn"][1],
            coordination_mean=overall["coordination"][0],
            coordination_std=overall["coordination"][1],
            class_statistics=class_stats,
        )

        self._current_baseline = baseline
        logger.info(
            "baseline_computed",
            version=version,
            total_samples=len(self._data_points),
            class_counts=class_counts,
        )

        return baseline

    def detect_drift(
        self,
        new_data: Sequence[BaselineDataPoint],
    ) -> dict[str, float]:
        """
        Detect drift between current baseline and new data.

        Returns:
            Dict mapping metric -> z-score of drift
        """
        if not self._current_baseline:
            raise ValueError("No baseline computed yet")

        if len(new_data) < 10:
            logger.warning("insufficient_data_for_drift", count=len(new_data))
            return {}

        # Compute new statistics
        new_metrics = {
            "hhi": [p.hhi for p in new_data],
            "entropy": [p.entropy for p in new_data],
            "gini": [p.gini for p in new_data],
            "wdr": [p.wdr for p in new_data],
            "churn": [p.churn for p in new_data],
            "coordination": [p.coordination for p in new_data],
        }

        # Compare to baseline
        baseline = self._current_baseline
        baseline_stats = {
            "hhi": (baseline.hhi_mean, baseline.hhi_std),
            "entropy": (baseline.entropy_mean, baseline.entropy_std),
            "gini": (baseline.gini_mean, baseline.gini_std),
            "wdr": (baseline.wdr_mean, baseline.wdr_std),
            "churn": (baseline.churn_mean, baseline.churn_std),
            "coordination": (baseline.coordination_mean, baseline.coordination_std),
        }

        drift_scores = {}
        for metric, values in new_metrics.items():
            new_mean = np.mean(values)
            baseline_mean, baseline_std = baseline_stats[metric]

            # Z-score of the drift
            z_drift = (new_mean - baseline_mean) / baseline_std
            drift_scores[metric] = float(z_drift)

            if abs(z_drift) > self.DRIFT_THRESHOLD_ZSCORE:
                logger.warning(
                    "drift_detected",
                    metric=metric,
                    z_score=z_drift,
                    baseline_mean=baseline_mean,
                    new_mean=new_mean,
                )

        return drift_scores

    def should_recalibrate(
        self,
        last_calibration: datetime,
        drift_scores: dict[str, float] | None = None,
    ) -> bool:
        """
        Determine if baseline should be recalibrated.

        Returns True if:
        - More than 30 days since last calibration
        - Any metric has drifted significantly
        """
        now = datetime.now(timezone.utc)
        days_since = (now - last_calibration).days

        # Monthly recalibration
        if days_since >= 30:
            logger.info("recalibration_due", days_since=days_since)
            return True

        # Drift-based recalibration
        if drift_scores:
            max_drift = max(abs(z) for z in drift_scores.values())
            if max_drift > self.DRIFT_THRESHOLD_ZSCORE * 1.5:
                logger.info("recalibration_due_to_drift", max_drift=max_drift)
                return True

        return False

    def get_percentile(
        self,
        metric: str,
        value: float,
        baseline_class: BaselineClass | None = None,
    ) -> float:
        """
        Get percentile of a value within baseline distribution.

        Args:
            metric: Metric name
            value: Observed value
            baseline_class: Optional class for comparison

        Returns:
            Percentile (0-100)
        """
        if baseline_class:
            class_points = [
                p for p in self._data_points
                if p.baseline_class == baseline_class
            ]
        else:
            class_points = self._data_points

        if not class_points:
            return 50.0

        values = [getattr(p, metric) for p in class_points]
        return float(stats.percentileofscore(values, value))
