"""
Trajectory Smoothing for Longitudinal Intelligence.

Raw derivatives are noisy. This module provides smoothing algorithms
to produce reliable velocity and acceleration estimates.

HARD RULES:
1. Expose both raw and smoothed trajectories
2. Default user-facing output uses smoothed
3. Confidence penalized for irregular cadence
4. Minimum data points required for computation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable
from enum import Enum
import math
import statistics

import structlog

logger = structlog.get_logger()


class SmoothingMethod(str, Enum):
    """Available smoothing methods."""

    NONE = "none"
    ROLLING_MEDIAN = "rolling_median"
    EWMA = "ewma"
    COMBINED = "combined"  # Median + EWMA


@dataclass
class SmoothedValue:
    """A value with both raw and smoothed versions."""

    raw: float
    smoothed: float
    method: SmoothingMethod
    confidence: float
    data_points: int


@dataclass
class SmoothedTrajectory:
    """Trajectory with both raw and smoothed metrics."""

    # Raw values
    raw_velocity: float
    raw_acceleration: float

    # Smoothed values
    smoothed_velocity: float
    smoothed_acceleration: float

    # Metadata
    smoothing_method: SmoothingMethod
    confidence: float
    data_points: int
    window_hours: float

    # Confidence penalties applied
    irregularity_penalty: float = 0.0
    outlier_count: int = 0


@dataclass
class TimeSeriesPoint:
    """Single point in a time series."""

    timestamp: datetime
    value: float
    weight: float = 1.0  # For weighted smoothing


class TrajectorySmoother:
    """
    Provides smoothing algorithms for trajectory computation.

    Methods:
    - Rolling Median: Robust to outliers
    - EWMA: Exponentially weighted moving average
    - Combined: Median first, then EWMA
    """

    # Configuration
    MIN_POINTS_FOR_VELOCITY = 3
    MIN_POINTS_FOR_ACCELERATION = 4
    MIN_POINTS_FOR_SMOOTHING = 5
    DEFAULT_WINDOW_SIZE = 3
    DEFAULT_EWMA_ALPHA = 0.3
    OUTLIER_THRESHOLD_SIGMA = 2.5
    MAX_CADENCE_IRREGULARITY = 0.5  # Max allowed CV for regular cadence

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        ewma_alpha: float = DEFAULT_EWMA_ALPHA,
        default_method: SmoothingMethod = SmoothingMethod.COMBINED,
    ):
        """
        Initialize the trajectory smoother.

        Args:
            window_size: Window size for rolling operations
            ewma_alpha: Alpha parameter for EWMA (0-1, higher = more recent weight)
            default_method: Default smoothing method
        """
        self.window_size = window_size
        self.ewma_alpha = ewma_alpha
        self.default_method = default_method

    def smooth_series(
        self,
        series: list[TimeSeriesPoint],
        method: Optional[SmoothingMethod] = None,
    ) -> list[TimeSeriesPoint]:
        """
        Apply smoothing to a time series.

        Args:
            series: Input time series
            method: Smoothing method (default: self.default_method)

        Returns:
            Smoothed time series
        """
        if len(series) < self.MIN_POINTS_FOR_SMOOTHING:
            return series  # Not enough points to smooth

        method = method or self.default_method

        if method == SmoothingMethod.NONE:
            return series
        elif method == SmoothingMethod.ROLLING_MEDIAN:
            return self._rolling_median(series)
        elif method == SmoothingMethod.EWMA:
            return self._ewma(series)
        elif method == SmoothingMethod.COMBINED:
            # First apply median, then EWMA
            median_smoothed = self._rolling_median(series)
            return self._ewma(median_smoothed)

        return series

    def _rolling_median(self, series: list[TimeSeriesPoint]) -> list[TimeSeriesPoint]:
        """Apply rolling median smoothing."""
        if len(series) < self.window_size:
            return series

        smoothed = []
        half_window = self.window_size // 2

        for i in range(len(series)):
            # Get window bounds
            start = max(0, i - half_window)
            end = min(len(series), i + half_window + 1)

            # Extract values in window
            window_values = [series[j].value for j in range(start, end)]

            # Compute median
            median_value = statistics.median(window_values)

            smoothed.append(TimeSeriesPoint(
                timestamp=series[i].timestamp,
                value=median_value,
                weight=series[i].weight,
            ))

        return smoothed

    def _ewma(self, series: list[TimeSeriesPoint]) -> list[TimeSeriesPoint]:
        """Apply exponentially weighted moving average."""
        if not series:
            return series

        smoothed = [series[0]]  # First point unchanged

        for i in range(1, len(series)):
            ewma_value = (
                self.ewma_alpha * series[i].value +
                (1 - self.ewma_alpha) * smoothed[i - 1].value
            )
            smoothed.append(TimeSeriesPoint(
                timestamp=series[i].timestamp,
                value=ewma_value,
                weight=series[i].weight,
            ))

        return smoothed

    def compute_smoothed_trajectory(
        self,
        series: list[TimeSeriesPoint],
        method: Optional[SmoothingMethod] = None,
    ) -> SmoothedTrajectory:
        """
        Compute trajectory with both raw and smoothed values.

        Args:
            series: Input time series
            method: Smoothing method

        Returns:
            SmoothedTrajectory with raw and smoothed values
        """
        method = method or self.default_method

        if len(series) < self.MIN_POINTS_FOR_VELOCITY:
            return SmoothedTrajectory(
                raw_velocity=0.0,
                raw_acceleration=0.0,
                smoothed_velocity=0.0,
                smoothed_acceleration=0.0,
                smoothing_method=method,
                confidence=0.0,
                data_points=len(series),
                window_hours=0.0,
            )

        # Compute raw derivatives
        raw_velocity, raw_acceleration = self._compute_derivatives(series)

        # Apply smoothing
        smoothed_series = self.smooth_series(series, method)
        smoothed_velocity, smoothed_acceleration = self._compute_derivatives(smoothed_series)

        # Compute confidence
        confidence, irregularity_penalty = self._compute_confidence(series)

        # Count outliers
        outlier_count = self._count_outliers(series)

        # Compute window hours
        window_hours = self._compute_window_hours(series)

        return SmoothedTrajectory(
            raw_velocity=raw_velocity,
            raw_acceleration=raw_acceleration,
            smoothed_velocity=smoothed_velocity,
            smoothed_acceleration=smoothed_acceleration,
            smoothing_method=method,
            confidence=confidence,
            data_points=len(series),
            window_hours=window_hours,
            irregularity_penalty=irregularity_penalty,
            outlier_count=outlier_count,
        )

    def _compute_derivatives(
        self,
        series: list[TimeSeriesPoint],
    ) -> tuple[float, float]:
        """
        Compute velocity (first derivative) and acceleration (second derivative).

        Uses robust linear regression.
        """
        if len(series) < 2:
            return 0.0, 0.0

        # Convert to hours from first point
        times = []
        values = []

        for i, point in enumerate(series):
            if i == 0:
                times.append(0.0)
            else:
                delta = (point.timestamp - series[0].timestamp).total_seconds() / 3600
                times.append(delta)
            values.append(point.value)

        # Velocity: linear regression slope
        velocity = self._robust_slope(times, values)

        # Acceleration: slope of slopes
        if len(series) < self.MIN_POINTS_FOR_ACCELERATION:
            acceleration = 0.0
        else:
            # Split series and compute slope difference
            mid = len(series) // 2
            first_half = series[:mid]
            second_half = series[mid:]

            v1, _ = self._compute_derivatives(first_half)
            v2, _ = self._compute_derivatives(second_half)

            time_delta = (
                second_half[0].timestamp - first_half[-1].timestamp
            ).total_seconds() / 3600

            acceleration = (v2 - v1) / time_delta if time_delta > 0 else 0.0

        return velocity, acceleration

    def _robust_slope(self, times: list[float], values: list[float]) -> float:
        """
        Compute robust slope estimate.

        Uses Theil-Sen estimator for outlier resistance.
        """
        n = len(times)

        if n < 2:
            return 0.0

        if n == 2:
            # Simple slope
            dt = times[1] - times[0]
            return (values[1] - values[0]) / dt if dt > 0 else 0.0

        # Theil-Sen: median of all pairwise slopes
        slopes = []

        for i in range(n):
            for j in range(i + 1, n):
                dt = times[j] - times[i]
                if dt > 0:
                    slope = (values[j] - values[i]) / dt
                    slopes.append(slope)

        if not slopes:
            return 0.0

        return statistics.median(slopes)

    def _compute_confidence(
        self,
        series: list[TimeSeriesPoint],
    ) -> tuple[float, float]:
        """
        Compute confidence score with irregularity penalty.

        Returns: (confidence, irregularity_penalty)
        """
        if len(series) < self.MIN_POINTS_FOR_VELOCITY:
            return 0.0, 0.0

        # Base confidence from data points
        point_factor = min(1.0, len(series) / 10.0)

        # Compute cadence irregularity
        intervals = []
        for i in range(1, len(series)):
            delta = (series[i].timestamp - series[i - 1].timestamp).total_seconds()
            intervals.append(delta)

        if not intervals:
            return point_factor, 0.0

        # Coefficient of variation for intervals
        mean_interval = statistics.mean(intervals)
        if mean_interval == 0:
            return point_factor, 0.0

        std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0.0
        cv = std_interval / mean_interval

        # Penalty for irregular cadence
        irregularity_penalty = min(0.5, cv / self.MAX_CADENCE_IRREGULARITY * 0.5)

        # Final confidence
        confidence = max(0.0, point_factor - irregularity_penalty)

        return round(confidence, 3), round(irregularity_penalty, 3)

    def _count_outliers(self, series: list[TimeSeriesPoint]) -> int:
        """Count outliers using z-score method."""
        if len(series) < 3:
            return 0

        values = [p.value for p in series]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0

        if std == 0:
            return 0

        outlier_count = 0
        for value in values:
            z_score = abs(value - mean) / std
            if z_score > self.OUTLIER_THRESHOLD_SIGMA:
                outlier_count += 1

        return outlier_count

    def _compute_window_hours(self, series: list[TimeSeriesPoint]) -> float:
        """Compute actual time window covered by series."""
        if len(series) < 2:
            return 0.0

        return (series[-1].timestamp - series[0].timestamp).total_seconds() / 3600

    def remove_outliers(
        self,
        series: list[TimeSeriesPoint],
        threshold_sigma: float = OUTLIER_THRESHOLD_SIGMA,
    ) -> list[TimeSeriesPoint]:
        """
        Remove outliers from series using z-score method.

        Preserves timestamps, replaces outlier values with interpolated values.
        """
        if len(series) < 3:
            return series

        values = [p.value for p in series]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0

        if std == 0:
            return series

        cleaned = []
        for i, point in enumerate(series):
            z_score = abs(point.value - mean) / std

            if z_score > threshold_sigma:
                # Replace with interpolated value
                if i == 0:
                    interpolated = series[1].value
                elif i == len(series) - 1:
                    interpolated = series[-2].value
                else:
                    interpolated = (series[i - 1].value + series[i + 1].value) / 2

                cleaned.append(TimeSeriesPoint(
                    timestamp=point.timestamp,
                    value=interpolated,
                    weight=0.5,  # Lower weight for interpolated
                ))
            else:
                cleaned.append(point)

        return cleaned


class SmoothedTrajectoryComputer:
    """
    Enhanced trajectory computer with smoothing.

    Wraps the base TrajectoryComputer with smoothing capabilities.
    """

    def __init__(
        self,
        smoother: Optional[TrajectorySmoother] = None,
        use_smoothed_default: bool = True,
    ):
        """
        Initialize smoothed trajectory computer.

        Args:
            smoother: TrajectorySmoother instance
            use_smoothed_default: Whether to use smoothed values by default
        """
        self.smoother = smoother or TrajectorySmoother()
        self.use_smoothed_default = use_smoothed_default

    def compute_velocity(
        self,
        series: list[TimeSeriesPoint],
        use_smoothed: Optional[bool] = None,
    ) -> SmoothedValue:
        """
        Compute velocity with smoothing.

        Args:
            series: Input time series
            use_smoothed: Override default smoothing preference

        Returns:
            SmoothedValue with raw and smoothed velocity
        """
        use_smoothed = use_smoothed if use_smoothed is not None else self.use_smoothed_default

        trajectory = self.smoother.compute_smoothed_trajectory(series)

        return SmoothedValue(
            raw=trajectory.raw_velocity,
            smoothed=trajectory.smoothed_velocity,
            method=trajectory.smoothing_method if use_smoothed else SmoothingMethod.NONE,
            confidence=trajectory.confidence,
            data_points=trajectory.data_points,
        )

    def compute_acceleration(
        self,
        series: list[TimeSeriesPoint],
        use_smoothed: Optional[bool] = None,
    ) -> SmoothedValue:
        """
        Compute acceleration with smoothing.

        Args:
            series: Input time series
            use_smoothed: Override default smoothing preference

        Returns:
            SmoothedValue with raw and smoothed acceleration
        """
        use_smoothed = use_smoothed if use_smoothed is not None else self.use_smoothed_default

        trajectory = self.smoother.compute_smoothed_trajectory(series)

        return SmoothedValue(
            raw=trajectory.raw_acceleration,
            smoothed=trajectory.smoothed_acceleration,
            method=trajectory.smoothing_method if use_smoothed else SmoothingMethod.NONE,
            confidence=trajectory.confidence,
            data_points=trajectory.data_points,
        )

    def get_user_facing_velocity(
        self,
        series: list[TimeSeriesPoint],
    ) -> float:
        """
        Get user-facing velocity value.

        Always returns smoothed value for user-facing output.
        """
        trajectory = self.smoother.compute_smoothed_trajectory(series)
        return trajectory.smoothed_velocity

    def get_user_facing_acceleration(
        self,
        series: list[TimeSeriesPoint],
    ) -> float:
        """
        Get user-facing acceleration value.

        Always returns smoothed value for user-facing output.
        """
        trajectory = self.smoother.compute_smoothed_trajectory(series)
        return trajectory.smoothed_acceleration
