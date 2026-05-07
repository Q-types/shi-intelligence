"""
Model Comparison Utilities.

Compare model performance for A/B testing and
model selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class ComparisonResult:
    """Result of model comparison."""

    model_a_id: str
    model_b_id: str

    # Performance metrics
    model_a_brier: float
    model_b_brier: float
    brier_improvement: float  # Positive = B is better

    model_a_auc: float
    model_b_auc: float
    auc_improvement: float

    model_a_calibration_slope: float
    model_b_calibration_slope: float

    # Statistical significance
    is_significant: bool
    p_value: float

    # Sample sizes
    model_a_samples: int
    model_b_samples: int

    # Recommendation
    winner: str | None  # "A", "B", or None if no clear winner
    confidence: float  # Confidence in winner determination

    compared_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ModelComparator:
    """
    Compares model performance for selection decisions.

    Supports:
    - Brier score comparison
    - AUC comparison
    - Calibration comparison
    - Statistical significance testing
    """

    def __init__(
        self,
        min_samples: int = 100,
        significance_level: float = 0.05,
    ):
        self.min_samples = min_samples
        self.significance_level = significance_level

    def compare(
        self,
        model_a_predictions: Sequence[tuple[float, int]],  # (prob, outcome)
        model_b_predictions: Sequence[tuple[float, int]],
        model_a_id: str = "model_a",
        model_b_id: str = "model_b",
    ) -> ComparisonResult:
        """
        Compare two models' predictions.

        Args:
            model_a_predictions: List of (predicted_prob, actual_outcome) for model A
            model_b_predictions: List of (predicted_prob, actual_outcome) for model B
            model_a_id: Identifier for model A
            model_b_id: Identifier for model B

        Returns:
            ComparisonResult with detailed comparison
        """
        # Extract predictions and outcomes
        a_probs = [p[0] for p in model_a_predictions]
        a_outcomes = [p[1] for p in model_a_predictions]
        b_probs = [p[0] for p in model_b_predictions]
        b_outcomes = [p[1] for p in model_b_predictions]

        # Compute Brier scores
        a_brier = self._compute_brier(a_probs, a_outcomes)
        b_brier = self._compute_brier(b_probs, b_outcomes)
        brier_improvement = a_brier - b_brier  # Positive = B is better (lower Brier)

        # Compute AUC
        a_auc = self._compute_auc(a_probs, a_outcomes)
        b_auc = self._compute_auc(b_probs, b_outcomes)
        auc_improvement = b_auc - a_auc  # Positive = B is better (higher AUC)

        # Compute calibration slopes
        a_slope = self._compute_calibration_slope(a_probs, a_outcomes)
        b_slope = self._compute_calibration_slope(b_probs, b_outcomes)

        # Statistical significance test
        is_significant, p_value = self._test_significance(
            a_probs, a_outcomes, b_probs, b_outcomes
        )

        # Determine winner
        winner, confidence = self._determine_winner(
            brier_improvement,
            auc_improvement,
            a_slope,
            b_slope,
            is_significant,
        )

        return ComparisonResult(
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            model_a_brier=a_brier,
            model_b_brier=b_brier,
            brier_improvement=brier_improvement,
            model_a_auc=a_auc,
            model_b_auc=b_auc,
            auc_improvement=auc_improvement,
            model_a_calibration_slope=a_slope,
            model_b_calibration_slope=b_slope,
            is_significant=is_significant,
            p_value=p_value,
            model_a_samples=len(model_a_predictions),
            model_b_samples=len(model_b_predictions),
            winner=winner,
            confidence=confidence,
        )

    def _compute_brier(
        self,
        predictions: Sequence[float],
        outcomes: Sequence[int],
    ) -> float:
        """Compute Brier score."""
        if not predictions:
            return 1.0

        squared_errors = [
            (p - o) ** 2 for p, o in zip(predictions, outcomes)
        ]
        return float(np.mean(squared_errors))

    def _compute_auc(
        self,
        predictions: Sequence[float],
        outcomes: Sequence[int],
    ) -> float:
        """Compute AUC-ROC."""
        if len(set(outcomes)) < 2:
            return 0.5  # Can't compute AUC with single class

        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(outcomes, predictions))
        except Exception:
            return 0.5

    def _compute_calibration_slope(
        self,
        predictions: Sequence[float],
        outcomes: Sequence[int],
    ) -> float:
        """Compute calibration slope."""
        if len(predictions) < 10:
            return 1.0

        try:
            coeffs = np.polyfit(predictions, outcomes, 1)
            return float(coeffs[0])
        except Exception:
            return 1.0

    def _test_significance(
        self,
        a_probs: Sequence[float],
        a_outcomes: Sequence[int],
        b_probs: Sequence[float],
        b_outcomes: Sequence[int],
    ) -> tuple[bool, float]:
        """
        Test statistical significance of difference.

        Uses paired t-test on squared errors if same outcomes,
        otherwise uses independent t-test.
        """
        if len(a_probs) < self.min_samples or len(b_probs) < self.min_samples:
            return False, 1.0

        try:
            from scipy import stats

            a_errors = [(p - o) ** 2 for p, o in zip(a_probs, a_outcomes)]
            b_errors = [(p - o) ** 2 for p, o in zip(b_probs, b_outcomes)]

            # If same length and outcomes match, use paired test
            if len(a_errors) == len(b_errors) and a_outcomes == b_outcomes:
                _, p_value = stats.ttest_rel(a_errors, b_errors)
            else:
                _, p_value = stats.ttest_ind(a_errors, b_errors)

            return p_value < self.significance_level, float(p_value)

        except Exception:
            return False, 1.0

    def _determine_winner(
        self,
        brier_improvement: float,
        auc_improvement: float,
        a_slope: float,
        b_slope: float,
        is_significant: bool,
    ) -> tuple[str | None, float]:
        """
        Determine which model is better.

        Returns:
            (winner, confidence) where winner is "A", "B", or None
        """
        if not is_significant:
            return None, 0.0

        # Score each model
        a_score = 0.0
        b_score = 0.0

        # Brier score (lower is better)
        if brier_improvement > 0.01:  # B is better
            b_score += 1.0
        elif brier_improvement < -0.01:  # A is better
            a_score += 1.0

        # AUC (higher is better)
        if auc_improvement > 0.01:  # B is better
            b_score += 1.0
        elif auc_improvement < -0.01:  # A is better
            a_score += 1.0

        # Calibration (closer to 1.0 is better)
        a_cal_error = abs(a_slope - 1.0)
        b_cal_error = abs(b_slope - 1.0)
        if b_cal_error < a_cal_error - 0.05:
            b_score += 0.5
        elif a_cal_error < b_cal_error - 0.05:
            a_score += 0.5

        # Determine winner
        total_score = a_score + b_score
        if total_score == 0:
            return None, 0.0

        if a_score > b_score:
            confidence = a_score / total_score
            return "A", confidence
        elif b_score > a_score:
            confidence = b_score / total_score
            return "B", confidence
        else:
            return None, 0.5


def compare_model_versions(
    baseline_model_path: str,
    candidate_model_path: str,
    test_data: Sequence[dict],
) -> ComparisonResult:
    """
    Compare two model versions on test data.

    Args:
        baseline_model_path: Path to baseline model
        candidate_model_path: Path to candidate model
        test_data: Test data with features and outcomes

    Returns:
        ComparisonResult
    """
    # This is a placeholder for actual model loading and prediction
    # In production, would load models and run predictions

    comparator = ModelComparator()

    # Mock predictions for demonstration
    baseline_preds = [(0.5, 0) for _ in test_data]
    candidate_preds = [(0.5, 0) for _ in test_data]

    return comparator.compare(
        baseline_preds,
        candidate_preds,
        model_a_id=baseline_model_path,
        model_b_id=candidate_model_path,
    )
