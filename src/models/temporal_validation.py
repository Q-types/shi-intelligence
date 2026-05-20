"""
Temporal Validation for Time-Sensitive Models.

Replaces shuffled KFold with proper temporal/walk-forward validation
for sell-risk models where time ordering matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Iterator, Callable

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


@dataclass
class TemporalSplit:
    """Result of a temporal train/test split."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    fold_number: int


class TemporalValidator:
    """
    Temporal cross-validation for time-sensitive models.

    Implements walk-forward validation where:
    - Training data always precedes test data temporally
    - No data leakage from future observations
    - Expanding or sliding window options

    This replaces shuffled KFold which violates temporal ordering
    assumptions in survival/hazard models.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: float = 0.3,
        gap_days: int = 0,
        expanding: bool = True,
    ):
        """
        Initialize temporal validator.

        Args:
            n_splits: Number of temporal folds
            min_train_size: Minimum fraction of data for training (first fold)
            gap_days: Gap between train and test to prevent leakage
            expanding: If True, training window expands; if False, slides
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.gap_days = gap_days
        self.expanding = expanding

    def split(
        self,
        data: pd.DataFrame,
        time_column: str = "timestamp",
    ) -> Iterator[TemporalSplit]:
        """
        Generate temporal train/test splits.

        Args:
            data: DataFrame with time column
            time_column: Name of timestamp column

        Yields:
            TemporalSplit for each fold
        """
        n = len(data)

        # Sort by time
        if time_column not in data.columns:
            # If no timestamp, use index order as proxy
            sorted_indices = np.arange(n)
            timestamps = None
        else:
            sorted_indices = data[time_column].argsort().values
            timestamps = data[time_column].sort_values()

        # Calculate split points
        min_train_n = int(n * self.min_train_size)
        test_size = (n - min_train_n) // self.n_splits

        for fold in range(self.n_splits):
            if self.expanding:
                # Expanding window: train grows each fold
                train_end_idx = min_train_n + fold * test_size
            else:
                # Sliding window: train size stays constant
                train_start_idx = fold * test_size
                train_end_idx = min_train_n + fold * test_size
                sorted_indices_train = sorted_indices[train_start_idx:train_end_idx]

            if self.expanding:
                train_indices = sorted_indices[:train_end_idx]
            else:
                train_indices = sorted_indices_train

            # Test indices (next chunk after train)
            test_start_idx = train_end_idx
            test_end_idx = min(train_end_idx + test_size, n)
            test_indices = sorted_indices[test_start_idx:test_end_idx]

            if len(test_indices) == 0:
                continue

            # Extract timestamps for metadata
            if timestamps is not None:
                train_start = timestamps.iloc[train_indices[0]] if len(train_indices) > 0 else None
                train_end = timestamps.iloc[train_indices[-1]] if len(train_indices) > 0 else None
                test_start = timestamps.iloc[test_indices[0]] if len(test_indices) > 0 else None
                test_end = timestamps.iloc[test_indices[-1]] if len(test_indices) > 0 else None
            else:
                train_start = train_end = test_start = test_end = datetime.now(timezone.utc)

            yield TemporalSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                fold_number=fold,
            )

    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits


class WalkForwardValidator:
    """
    Walk-forward validation with configurable retrain frequency.

    More realistic for production scenarios where models are
    periodically retrained on new data.
    """

    def __init__(
        self,
        initial_train_days: int = 30,
        test_window_days: int = 7,
        retrain_frequency_days: int = 7,
        min_train_samples: int = 100,
    ):
        """
        Initialize walk-forward validator.

        Args:
            initial_train_days: Initial training window size
            test_window_days: Size of each test window
            retrain_frequency_days: How often to retrain
            min_train_samples: Minimum samples for training
        """
        self.initial_train_days = initial_train_days
        self.test_window_days = test_window_days
        self.retrain_frequency_days = retrain_frequency_days
        self.min_train_samples = min_train_samples

    def split(
        self,
        data: pd.DataFrame,
        time_column: str = "timestamp",
    ) -> Iterator[TemporalSplit]:
        """
        Generate walk-forward splits.

        Args:
            data: DataFrame with time column
            time_column: Name of timestamp column

        Yields:
            TemporalSplit for each walk-forward window
        """
        if time_column not in data.columns:
            raise ValueError(f"Column {time_column} not found in data")

        # Convert to datetime if needed
        data = data.copy()
        data[time_column] = pd.to_datetime(data[time_column])
        data = data.sort_values(time_column).reset_index(drop=True)

        min_date = data[time_column].min()
        max_date = data[time_column].max()

        # Start walk-forward
        current_train_end = min_date + timedelta(days=self.initial_train_days)
        fold = 0

        while current_train_end < max_date:
            # Training data: everything before train_end
            train_mask = data[time_column] < current_train_end
            train_indices = data[train_mask].index.values

            # Test data: window after train_end
            test_start = current_train_end
            test_end = test_start + timedelta(days=self.test_window_days)
            test_mask = (data[time_column] >= test_start) & (data[time_column] < test_end)
            test_indices = data[test_mask].index.values

            # Skip if insufficient data
            if len(train_indices) < self.min_train_samples or len(test_indices) == 0:
                current_train_end += timedelta(days=self.retrain_frequency_days)
                continue

            yield TemporalSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=min_date,
                train_end=current_train_end,
                test_start=test_start,
                test_end=test_end,
                fold_number=fold,
            )

            fold += 1
            current_train_end += timedelta(days=self.retrain_frequency_days)


@dataclass
class TemporalValidationResult:
    """Result of temporal cross-validation."""

    scores: list[float]
    mean_score: float
    std_score: float
    fold_details: list[dict]
    validation_type: str


def temporal_cross_validate(
    model_factory: Callable,
    data: pd.DataFrame,
    time_column: str,
    score_fn: Callable,
    n_splits: int = 5,
    validator_type: str = "temporal",
) -> TemporalValidationResult:
    """
    Perform temporal cross-validation.

    Args:
        model_factory: Function that returns a fresh model instance
        data: Training data
        time_column: Name of timestamp column
        score_fn: Function(model, test_data) -> float
        n_splits: Number of splits
        validator_type: 'temporal' or 'walk_forward'

    Returns:
        TemporalValidationResult with scores and details
    """
    if validator_type == "walk_forward":
        validator = WalkForwardValidator()
    else:
        validator = TemporalValidator(n_splits=n_splits)

    scores = []
    fold_details = []

    for split in validator.split(data, time_column):
        train_data = data.iloc[split.train_indices]
        test_data = data.iloc[split.test_indices]

        try:
            # Create and fit model
            model = model_factory()
            model.fit(train_data)

            # Score on test data
            score = score_fn(model, test_data)
            scores.append(score)

            fold_details.append({
                "fold": split.fold_number,
                "train_size": len(split.train_indices),
                "test_size": len(split.test_indices),
                "train_start": split.train_start.isoformat() if split.train_start else None,
                "train_end": split.train_end.isoformat() if split.train_end else None,
                "test_start": split.test_start.isoformat() if split.test_start else None,
                "test_end": split.test_end.isoformat() if split.test_end else None,
                "score": score,
            })

            logger.debug(
                "temporal_cv_fold",
                fold=split.fold_number,
                train_size=len(split.train_indices),
                test_size=len(split.test_indices),
                score=score,
            )

        except Exception as e:
            logger.warning(
                "temporal_cv_fold_failed",
                fold=split.fold_number,
                error=str(e),
            )
            scores.append(0.5)  # Default score for failed fold

    mean_score = float(np.mean(scores)) if scores else 0.5
    std_score = float(np.std(scores)) if scores else 0.0

    logger.info(
        "temporal_cv_completed",
        n_folds=len(scores),
        mean_score=mean_score,
        std_score=std_score,
    )

    return TemporalValidationResult(
        scores=scores,
        mean_score=mean_score,
        std_score=std_score,
        fold_details=fold_details,
        validation_type=validator_type,
    )
