"""
Walk-Forward Validation for Temporal Models.

Implements time-series cross-validation to prevent look-ahead bias
and validate temporal models on out-of-sample data.

CRITICAL: All temporal models must be validated using walk-forward,
NOT standard k-fold cross-validation (which leaks future information).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Sequence, TypeVar

import numpy as np
import numpy.typing as npt
import structlog

from .regimes import HolderRegimeDetector, HolderRegimeType

logger = structlog.get_logger()

T = TypeVar("T")


@dataclass
class ValidationWindow:
    """Single validation window for walk-forward."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int


@dataclass
class ValidationResult:
    """Results from a single validation window."""

    window_id: int
    metric_name: str
    score: float
    predictions: npt.NDArray[np.float64]
    actuals: npt.NDArray[np.float64]
    train_size: int
    test_size: int


@dataclass
class WalkForwardResults:
    """Aggregate results from walk-forward validation."""

    window_results: list[ValidationResult]
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    metric_name: str


class WalkForwardValidator:
    """
    Walk-forward time-series cross-validation.

    Implements expanding or rolling window validation where:
    - Training set always comes before test set (no future leakage)
    - Models are retrained on each window
    - Out-of-sample performance is aggregated
    """

    def __init__(
        self,
        train_window_days: int = 30,
        test_window_days: int = 7,
        step_days: int = 7,
        min_train_samples: int = 10,
        expanding_window: bool = True,
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_window_days: Size of training window
            test_window_days: Size of test window
            step_days: Step size between windows
            min_train_samples: Minimum samples required for training
            expanding_window: If True, use expanding window. If False, use rolling.
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.min_train_samples = min_train_samples
        self.expanding_window = expanding_window

    def create_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[ValidationWindow]:
        """
        Create validation windows for walk-forward.

        Args:
            start_date: Start of available data
            end_date: End of available data

        Returns:
            List of ValidationWindow objects
        """
        windows = []
        window_id = 0

        # Initial training window
        train_start = start_date
        train_end = start_date + timedelta(days=self.train_window_days)

        while train_end + timedelta(days=self.test_window_days) <= end_date:
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window_days)

            windows.append(
                ValidationWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    window_id=window_id,
                )
            )

            window_id += 1

            # Move to next window
            if self.expanding_window:
                # Expanding: keep train_start, extend train_end
                train_end = test_end
            else:
                # Rolling: move both start and end
                train_start = train_start + timedelta(days=self.step_days)
                train_end = train_end + timedelta(days=self.step_days)

        logger.info(
            "created_validation_windows",
            n_windows=len(windows),
            expanding=self.expanding_window,
        )

        return windows

    def validate_regime_detector(
        self,
        detector: HolderRegimeDetector,
        features: npt.NDArray[np.float64],
        timestamps: Sequence[datetime],
        true_regimes: Sequence[HolderRegimeType],
    ) -> WalkForwardResults:
        """
        Validate regime detector using walk-forward.

        Args:
            detector: HolderRegimeDetector to validate
            features: Feature matrix (n_samples, n_features)
            timestamps: Corresponding timestamps
            true_regimes: Ground truth regime labels

        Returns:
            WalkForwardResults with accuracy scores
        """
        if len(features) != len(timestamps):
            raise ValueError("Features and timestamps must have same length")

        if len(timestamps) != len(true_regimes):
            raise ValueError("Timestamps and true_regimes must have same length")

        # Create windows
        windows = self.create_windows(timestamps[0], timestamps[-1])

        window_results = []

        for window in windows:
            # Get train/test indices
            train_mask = (timestamps >= window.train_start) & (timestamps < window.train_end)
            test_mask = (timestamps >= window.test_start) & (timestamps < window.test_end)

            train_features = features[train_mask]
            test_features = features[test_mask]
            test_regimes = np.array([r for r, m in zip(true_regimes, test_mask) if m])

            if len(train_features) < self.min_train_samples:
                logger.warning(
                    "insufficient_training_samples",
                    window_id=window.window_id,
                    n_samples=len(train_features),
                )
                continue

            if len(test_features) == 0:
                logger.warning("empty_test_set", window_id=window.window_id)
                continue

            # Train detector on this window
            detector.fit([train_features])

            # Predict on test set
            predictions = []
            for i in range(len(test_features)):
                regime_state = detector.predict_regime(test_features[i : i + 1])
                predictions.append(regime_state.regime)

            # Compute accuracy
            correct = sum(1 for pred, true in zip(predictions, test_regimes) if pred == true)
            accuracy = correct / len(predictions) if predictions else 0.0

            window_results.append(
                ValidationResult(
                    window_id=window.window_id,
                    metric_name="regime_accuracy",
                    score=accuracy,
                    predictions=np.array([p.value for p in predictions]),
                    actuals=np.array([r.value for r in test_regimes]),
                    train_size=len(train_features),
                    test_size=len(test_features),
                )
            )

            logger.info(
                "window_validated",
                window_id=window.window_id,
                accuracy=accuracy,
                train_size=len(train_features),
                test_size=len(test_features),
            )

        if not window_results:
            raise ValueError("No validation windows produced results")

        scores = [r.score for r in window_results]

        return WalkForwardResults(
            window_results=window_results,
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            metric_name="regime_accuracy",
        )

    def validate_forecaster(
        self,
        forecast_fn: Callable[[npt.NDArray[np.float64], int], npt.NDArray[np.float64]],
        time_series: npt.NDArray[np.float64],
        timestamps: Sequence[datetime],
        horizon: int = 1,
    ) -> WalkForwardResults:
        """
        Validate a forecast function using walk-forward.

        Args:
            forecast_fn: Function(train_data, horizon) -> predictions
            time_series: Time-series data to forecast
            timestamps: Corresponding timestamps
            horizon: Forecast horizon (steps ahead)

        Returns:
            WalkForwardResults with MAPE/MAE scores
        """
        if len(time_series) != len(timestamps):
            raise ValueError("Time series and timestamps must have same length")

        windows = self.create_windows(timestamps[0], timestamps[-1])

        window_results = []

        for window in windows:
            # Get train/test indices
            train_mask = (timestamps >= window.train_start) & (timestamps < window.train_end)
            test_mask = (timestamps >= window.test_start) & (timestamps < window.test_end)

            train_data = time_series[train_mask]
            test_data = time_series[test_mask]

            if len(train_data) < self.min_train_samples:
                continue

            if len(test_data) < horizon:
                continue

            # Forecast
            predictions = forecast_fn(train_data, horizon)

            # Compute MAPE
            actuals = test_data[:horizon]
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

            window_results.append(
                ValidationResult(
                    window_id=window.window_id,
                    metric_name="mape",
                    score=mape,
                    predictions=predictions,
                    actuals=actuals,
                    train_size=len(train_data),
                    test_size=len(test_data),
                )
            )

        if not window_results:
            raise ValueError("No validation windows produced results")

        scores = [r.score for r in window_results]

        return WalkForwardResults(
            window_results=window_results,
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            metric_name="mape",
        )


def compute_regime_detection_metrics(
    predictions: Sequence[HolderRegimeType],
    actuals: Sequence[HolderRegimeType],
) -> dict[str, float]:
    """
    Compute classification metrics for regime detection.

    Args:
        predictions: Predicted regimes
        actuals: True regimes

    Returns:
        Dict with accuracy, precision, recall, f1
    """
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have same length")

    # Overall accuracy
    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    accuracy = correct / len(predictions) if predictions else 0.0

    # Per-class metrics (macro-averaged)
    unique_regimes = set(actuals)
    precisions = []
    recalls = []

    for regime in unique_regimes:
        tp = sum(1 for p, a in zip(predictions, actuals) if p == regime and a == regime)
        fp = sum(1 for p, a in zip(predictions, actuals) if p == regime and a != regime)
        fn = sum(1 for p, a in zip(predictions, actuals) if p != regime and a == regime)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    macro_precision = np.mean(precisions) if precisions else 0.0
    macro_recall = np.mean(recalls) if recalls else 0.0
    f1 = (
        2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1": float(f1),
    }
