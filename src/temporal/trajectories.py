"""
Metric Trajectory Tracking (Time-Series Specialist).

Implements:
- HHI(t), Gini(t), Churn(t), WhaleDominance(t) time-series
- First derivatives: dHHI/dt, dGini/dt, dChurn/dt
- Trend detection: centralizing vs decentralizing
- Velocity calculations for regime detection

CRITICAL: All metric calculations use FROZEN formulas from PDR.
This module only adds temporal tracking on top of those metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter1d
import structlog

logger = structlog.get_logger()


class TrendDirection(Enum):
    """Trend classification for metric evolution."""

    CENTRALIZING = "centralizing"  # HHI increasing, Gini increasing
    DECENTRALIZING = "decentralizing"  # HHI decreasing, Gini decreasing
    STABLE = "stable"  # No significant trend
    VOLATILE = "volatile"  # High variance, no clear direction


@dataclass
class MetricPoint:
    """Single metric observation at a point in time."""

    timestamp: datetime
    value: float
    metric_name: str


@dataclass
class MetricTrajectory:
    """Time-series trajectory for a metric with derivatives."""

    metric_name: str
    points: list[MetricPoint]

    # Computed properties
    trend: TrendDirection
    velocity: float  # dMetric/dt (per day)
    acceleration: float  # d²Metric/dt² (per day²)

    # Statistics
    mean: float
    std: float
    min_value: float
    max_value: float

    # Metadata
    window_days: int
    computed_at: datetime


class TrajectoryTracker:
    """
    Tracks metric evolution over time and computes derivatives.

    Time-Series Specialist Agent Implementation:
    - Stores metric snapshots in time-series database
    - Computes numerical derivatives using finite differences
    - Smooths derivatives using moving average for noise reduction
    - Detects trends and regime changes
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        min_points_for_derivative: int = 3,
    ):
        """
        Initialize trajectory tracker.

        Args:
            smoothing_window: Window size for moving average smoothing
            min_points_for_derivative: Minimum points required to compute derivatives
        """
        self.smoothing_window = smoothing_window
        self.min_points = min_points_for_derivative

    def compute_trajectory(
        self,
        points: Sequence[MetricPoint],
        window_days: Optional[int] = None,
    ) -> MetricTrajectory:
        """
        Compute trajectory from metric time-series.

        Args:
            points: Time-ordered metric observations
            window_days: Optional lookback window (filters old points)

        Returns:
            MetricTrajectory with derivatives and trend analysis
        """
        if not points:
            raise ValueError("Cannot compute trajectory from empty points")

        if len(points) < self.min_points:
            logger.warning(
                "insufficient_points_for_derivative",
                num_points=len(points),
                min_required=self.min_points,
            )

        # Sort by timestamp
        sorted_points = sorted(points, key=lambda p: p.timestamp)

        # Filter to window if specified
        if window_days:
            cutoff = sorted_points[-1].timestamp - timedelta(days=window_days)
            sorted_points = [p for p in sorted_points if p.timestamp >= cutoff]

        if not sorted_points:
            raise ValueError("No points remaining after window filter")

        metric_name = sorted_points[0].metric_name

        # Extract arrays
        timestamps = np.array([(p.timestamp - sorted_points[0].timestamp).total_seconds() / 86400
                               for p in sorted_points])  # Convert to days
        values = np.array([p.value for p in sorted_points])

        # Compute statistics
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        min_val = float(np.min(values))
        max_val = float(np.max(values))

        # Compute derivatives
        velocity = self._compute_velocity(timestamps, values)
        acceleration = self._compute_acceleration(timestamps, values)

        # Detect trend
        trend = self._detect_trend(values, velocity, std_val)

        return MetricTrajectory(
            metric_name=metric_name,
            points=sorted_points,
            trend=trend,
            velocity=velocity,
            acceleration=acceleration,
            mean=mean_val,
            std=std_val,
            min_value=min_val,
            max_value=max_val,
            window_days=window_days or int(timestamps[-1] - timestamps[0]) + 1,
            computed_at=datetime.now(timezone.utc),
        )

    def _compute_velocity(
        self,
        timestamps: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
    ) -> float:
        """
        Compute first derivative (velocity) using finite differences.

        Uses central differences where possible, forward/backward at boundaries.
        Applies smoothing to reduce noise.

        Args:
            timestamps: Time points (in days)
            values: Metric values

        Returns:
            Average velocity (dMetric/dt per day)
        """
        if len(values) < 2:
            return 0.0

        # Compute finite differences
        dt = np.diff(timestamps)
        dv = np.diff(values)

        # Avoid division by zero
        dt = np.where(dt == 0, 1e-6, dt)

        velocities = dv / dt

        # Smooth if enough points
        if len(velocities) >= self.smoothing_window:
            velocities = uniform_filter1d(velocities, size=self.smoothing_window)

        # Return mean velocity
        return float(np.mean(velocities))

    def _compute_acceleration(
        self,
        timestamps: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
    ) -> float:
        """
        Compute second derivative (acceleration).

        Args:
            timestamps: Time points (in days)
            values: Metric values

        Returns:
            Average acceleration (d²Metric/dt² per day²)
        """
        if len(values) < 3:
            return 0.0

        # Compute first derivative
        dt = np.diff(timestamps)
        dv = np.diff(values)
        dt = np.where(dt == 0, 1e-6, dt)
        first_deriv = dv / dt

        # Compute second derivative
        dt2 = np.diff(timestamps[1:])
        dv2 = np.diff(first_deriv)
        dt2 = np.where(dt2 == 0, 1e-6, dt2)
        second_deriv = dv2 / dt2

        return float(np.mean(second_deriv))

    def _detect_trend(
        self,
        values: npt.NDArray[np.float64],
        velocity: float,
        std: float,
    ) -> TrendDirection:
        """
        Detect trend direction from velocity and volatility.

        Args:
            values: Metric values
            velocity: Mean velocity
            std: Standard deviation

        Returns:
            TrendDirection classification
        """
        # Coefficient of variation
        cv = std / np.mean(values) if np.mean(values) != 0 else 0

        # High volatility check
        if cv > 0.3:  # 30% coefficient of variation
            return TrendDirection.VOLATILE

        # Velocity threshold (relative to std)
        velocity_threshold = std * 0.1  # 10% of std per day

        if abs(velocity) < velocity_threshold:
            return TrendDirection.STABLE
        elif velocity > 0:
            # Positive velocity = increasing concentration for HHI/Gini
            return TrendDirection.CENTRALIZING
        else:
            return TrendDirection.DECENTRALIZING

    def compute_multi_metric_trajectory(
        self,
        snapshots: Sequence[dict[str, Any]],
        metrics: list[str] = ["hhi", "gini", "churn_rate", "whale_dominance"],
        window_days: Optional[int] = 30,
    ) -> dict[str, MetricTrajectory]:
        """
        Compute trajectories for multiple metrics simultaneously.

        Args:
            snapshots: List of metric snapshots with timestamp and metric values
            metrics: List of metric names to track
            window_days: Lookback window

        Returns:
            Dict mapping metric name to MetricTrajectory
        """
        # Group points by metric
        metric_points: dict[str, list[MetricPoint]] = {m: [] for m in metrics}

        for snapshot in snapshots:
            timestamp = snapshot["timestamp"]
            for metric_name in metrics:
                if metric_name in snapshot:
                    metric_points[metric_name].append(
                        MetricPoint(
                            timestamp=timestamp,
                            value=snapshot[metric_name],
                            metric_name=metric_name,
                        )
                    )

        # Compute trajectories
        trajectories = {}
        for metric_name, points in metric_points.items():
            if points:
                try:
                    trajectories[metric_name] = self.compute_trajectory(
                        points, window_days=window_days
                    )
                except ValueError as e:
                    logger.warning(
                        "trajectory_computation_failed",
                        metric=metric_name,
                        error=str(e),
                    )

        return trajectories

    def detect_regime_signals(
        self,
        hhi_trajectory: MetricTrajectory,
        gini_trajectory: MetricTrajectory,
        churn_trajectory: Optional[MetricTrajectory] = None,
    ) -> dict[str, Any]:
        """
        Extract regime signals from metric trajectories.

        Used by HolderRegimeDetector for HMM feature engineering.

        Args:
            hhi_trajectory: HHI trajectory
            gini_trajectory: Gini trajectory
            churn_trajectory: Optional churn trajectory

        Returns:
            Dict of regime signals for HMM input
        """
        signals = {
            "dhhi_dt": hhi_trajectory.velocity,
            "dgini_dt": gini_trajectory.velocity,
            "hhi_trend": hhi_trajectory.trend.value,
            "gini_trend": gini_trajectory.trend.value,
            "concentration_acceleration": hhi_trajectory.acceleration,
        }

        if churn_trajectory:
            signals["dchurn_dt"] = churn_trajectory.velocity
            signals["churn_trend"] = churn_trajectory.trend.value

        # Coordination signal: HHI and Gini moving together
        if (
            hhi_trajectory.trend == gini_trajectory.trend
            and hhi_trajectory.trend == TrendDirection.CENTRALIZING
        ):
            signals["coordination_signal"] = 1.0
        else:
            signals["coordination_signal"] = 0.0

        return signals

    def forecast_next_point(
        self,
        trajectory: MetricTrajectory,
        horizon_days: int = 1,
    ) -> tuple[float, float]:
        """
        Simple linear extrapolation forecast.

        NOTE: This is a basic implementation. For production, use ARIMA/VAR models.

        Args:
            trajectory: Metric trajectory
            horizon_days: Forecast horizon in days

        Returns:
            (predicted_value, confidence_interval_width)
        """
        if not trajectory.points:
            raise ValueError("Cannot forecast from empty trajectory")

        # Simple linear extrapolation
        current_value = trajectory.points[-1].value
        predicted = current_value + (trajectory.velocity * horizon_days)

        # Confidence interval based on historical std
        ci_width = trajectory.std * np.sqrt(horizon_days) * 1.96  # 95% CI

        return predicted, ci_width
