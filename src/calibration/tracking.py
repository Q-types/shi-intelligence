"""
Calibration Tracking.

Continuous tracking of model calibration metrics
for monitoring and alerting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Sequence
from collections import deque

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class PredictionRecord:
    """Record of a single prediction and outcome."""

    prediction_id: str
    predicted_probability: float
    actual_outcome: int  # 0 or 1
    timestamp: datetime
    model_version: str
    features: dict | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BrierScoreWindow:
    """Brier score for a time window."""

    start_time: datetime
    end_time: datetime
    brier_score: float
    num_predictions: int
    breakdown: dict = field(default_factory=dict)


class BrierScoreTracker:
    """
    Tracks Brier score over time with rolling windows.

    Per INITIAL_PROMPT: Brier score must be reported for validation.
    """

    def __init__(
        self,
        window_size: int = 1000,
        alert_threshold: float = 0.25,
    ):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self._predictions: deque[PredictionRecord] = deque(maxlen=window_size)
        self._daily_scores: list[BrierScoreWindow] = []

    def record(self, prediction: PredictionRecord) -> float | None:
        """
        Record a prediction and return current Brier score if changed significantly.

        Returns:
            Brier score if threshold exceeded, None otherwise
        """
        self._predictions.append(prediction)

        # Compute current Brier score
        if len(self._predictions) < 10:
            return None

        score = self._compute_brier_score()

        if score > self.alert_threshold:
            logger.warning(
                "brier_score_alert",
                score=score,
                threshold=self.alert_threshold,
                window_size=len(self._predictions),
            )
            return score

        return None

    def _compute_brier_score(self) -> float:
        """Compute Brier score for current window."""
        predictions = list(self._predictions)

        if not predictions:
            return 0.0

        squared_errors = [
            (p.predicted_probability - p.actual_outcome) ** 2
            for p in predictions
        ]

        return float(np.mean(squared_errors))

    def get_current_score(self) -> float:
        """Get current Brier score."""
        return self._compute_brier_score()

    def get_score_by_bucket(self, num_buckets: int = 10) -> dict[str, float]:
        """
        Get Brier score breakdown by prediction bucket.

        Helps identify which probability ranges are miscalibrated.
        """
        predictions = list(self._predictions)
        if not predictions:
            return {}

        buckets: dict[str, list[float]] = {
            f"{i/num_buckets:.1f}-{(i+1)/num_buckets:.1f}": []
            for i in range(num_buckets)
        }

        for p in predictions:
            bucket_idx = min(int(p.predicted_probability * num_buckets), num_buckets - 1)
            bucket_key = f"{bucket_idx/num_buckets:.1f}-{(bucket_idx+1)/num_buckets:.1f}"
            error = (p.predicted_probability - p.actual_outcome) ** 2
            buckets[bucket_key].append(error)

        return {
            k: float(np.mean(v)) if v else 0.0
            for k, v in buckets.items()
        }

    def compute_daily_summary(self) -> BrierScoreWindow | None:
        """Compute daily summary and archive."""
        predictions = list(self._predictions)
        if not predictions:
            return None

        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(days=1)

        daily_preds = [p for p in predictions if p.timestamp >= day_ago]

        if len(daily_preds) < 10:
            return None

        score = float(np.mean([
            (p.predicted_probability - p.actual_outcome) ** 2
            for p in daily_preds
        ]))

        window = BrierScoreWindow(
            start_time=day_ago,
            end_time=now,
            brier_score=score,
            num_predictions=len(daily_preds),
            breakdown=self.get_score_by_bucket(),
        )

        self._daily_scores.append(window)
        return window


class CalibrationTracker:
    """
    Tracks model calibration over time.

    Monitors:
    - Calibration slope (ideal = 1.0)
    - Calibration intercept (ideal = 0.0)
    - Prediction interval coverage
    """

    def __init__(
        self,
        target_slope_range: tuple[float, float] = (0.7, 1.3),
        target_coverage: float = 0.9,
    ):
        self.target_slope_range = target_slope_range
        self.target_coverage = target_coverage
        self._predictions: deque[PredictionRecord] = deque(maxlen=5000)

    def record(self, prediction: PredictionRecord) -> None:
        """Record a prediction."""
        self._predictions.append(prediction)

    def compute_calibration_curve(
        self,
        num_bins: int = 10,
    ) -> tuple[list[float], list[float], float, float]:
        """
        Compute calibration curve.

        Returns:
            (mean_predicted, fraction_positive, slope, intercept)
        """
        predictions = list(self._predictions)
        if len(predictions) < num_bins * 5:
            return [], [], 1.0, 0.0

        # Bin predictions
        bins: dict[int, list[PredictionRecord]] = {i: [] for i in range(num_bins)}

        for p in predictions:
            bin_idx = min(int(p.predicted_probability * num_bins), num_bins - 1)
            bins[bin_idx].append(p)

        mean_predicted = []
        fraction_positive = []

        for i in range(num_bins):
            if bins[i]:
                preds = [p.predicted_probability for p in bins[i]]
                outcomes = [p.actual_outcome for p in bins[i]]
                mean_predicted.append(float(np.mean(preds)))
                fraction_positive.append(float(np.mean(outcomes)))

        if len(mean_predicted) < 2:
            return mean_predicted, fraction_positive, 1.0, 0.0

        # Compute slope via linear regression
        coeffs = np.polyfit(mean_predicted, fraction_positive, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])

        return mean_predicted, fraction_positive, slope, intercept

    def is_well_calibrated(self) -> tuple[bool, str]:
        """
        Check if model is well calibrated.

        Returns:
            (is_calibrated, reason)
        """
        _, _, slope, intercept = self.compute_calibration_curve()

        min_slope, max_slope = self.target_slope_range

        if slope < min_slope:
            return False, f"Under-confident: slope {slope:.2f} < {min_slope}"
        if slope > max_slope:
            return False, f"Over-confident: slope {slope:.2f} > {max_slope}"
        if abs(intercept) > 0.1:
            return False, f"Biased: intercept {intercept:.2f}"

        return True, "Well calibrated"

    def compute_coverage(
        self,
        confidence_level: float = 0.9,
    ) -> float:
        """
        Compute prediction interval coverage.

        For sell probability predictions, check if outcomes
        fall within predicted confidence intervals.
        """
        predictions = list(self._predictions)
        if not predictions:
            return 1.0

        # For binary outcomes, compute empirical coverage
        # by checking if high-probability predictions match outcomes

        covered = 0
        total = 0

        for p in predictions:
            # Prediction is "confident" if far from 0.5
            if p.predicted_probability > confidence_level:
                # Predicted positive
                if p.actual_outcome == 1:
                    covered += 1
                total += 1
            elif p.predicted_probability < (1 - confidence_level):
                # Predicted negative
                if p.actual_outcome == 0:
                    covered += 1
                total += 1

        if total == 0:
            return 1.0

        return covered / total

    def get_summary(self) -> dict:
        """Get calibration summary."""
        mean_pred, frac_pos, slope, intercept = self.compute_calibration_curve()
        is_calibrated, reason = self.is_well_calibrated()
        coverage = self.compute_coverage()

        return {
            "num_predictions": len(self._predictions),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "is_well_calibrated": is_calibrated,
            "calibration_reason": reason,
            "coverage_90": coverage,
            "mean_predicted": mean_pred,
            "fraction_positive": frac_pos,
        }
