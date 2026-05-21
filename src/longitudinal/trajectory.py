"""
Trajectory Features for Longitudinal Intelligence.

Time-series behavioral metrics with velocity and acceleration.

6 Core Trajectory Metrics:
1. accumulation_rate - Change in holder concentration over time
2. sell_acceleration - Second derivative of sell pressure
3. liquidity_decay_rate - Rate of liquidity drain
4. whale_dispersion_rate - Rate of whale distribution
5. coordination_persistence - How long coordination patterns last
6. holder_churn_velocity - Rate of holder turnover

HARD RULES:
- All trajectories must be computable from snapshots
- Trajectories are token-specific
- Minimum 3 data points for velocity, 4 for acceleration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Any
from enum import Enum
import math

import structlog

logger = structlog.get_logger()


class TrajectoryTrend(Enum):
    """Trajectory trend classification."""

    ACCELERATING_UP = "accelerating_up"
    ACCELERATING_DOWN = "accelerating_down"
    DECELERATING_UP = "decelerating_up"
    DECELERATING_DOWN = "decelerating_down"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class TimeSeriesPoint:
    """Single point in a time series."""

    timestamp: datetime
    value: float
    metadata: dict = field(default_factory=dict)


@dataclass
class TrajectoryMetric:
    """Computed trajectory metric with velocity and acceleration."""

    name: str
    current_value: float
    velocity: float  # First derivative (change per hour)
    acceleration: float  # Second derivative
    trend: TrajectoryTrend
    confidence: float  # 0-1, based on data quality
    data_points: int
    window_hours: float
    last_updated: datetime


@dataclass
class TokenTrajectory:
    """Complete trajectory profile for a token."""

    token_mint: str
    computed_at: datetime
    window_hours: float

    # Core metrics
    accumulation_rate: Optional[TrajectoryMetric] = None
    sell_acceleration: Optional[TrajectoryMetric] = None
    liquidity_decay_rate: Optional[TrajectoryMetric] = None
    whale_dispersion_rate: Optional[TrajectoryMetric] = None
    coordination_persistence: Optional[TrajectoryMetric] = None
    holder_churn_velocity: Optional[TrajectoryMetric] = None

    # Aggregate signals
    overall_health_trend: TrajectoryTrend = TrajectoryTrend.STABLE
    risk_trajectory: str = "neutral"  # "improving", "deteriorating", "neutral"


class TrajectoryComputer:
    """
    Computes trajectory features from snapshot time series.

    Uses numerical differentiation to compute velocity (first derivative)
    and acceleration (second derivative) of behavioral metrics.
    """

    VERSION = "1.0.0"

    # Minimum data points required
    MIN_POINTS_FOR_VELOCITY = 3
    MIN_POINTS_FOR_ACCELERATION = 4

    # Trend thresholds
    VELOCITY_THRESHOLD = 0.01  # Minimum velocity to be "moving"
    ACCELERATION_THRESHOLD = 0.001  # Minimum acceleration to be "accelerating"

    def __init__(self, session_factory=None):
        """Initialize the trajectory computer."""
        self.session_factory = session_factory

    async def compute_trajectory(
        self,
        token_mint: str,
        snapshots: list[dict],
        window_hours: float = 24.0,
    ) -> TokenTrajectory:
        """
        Compute trajectory features from snapshots.

        Args:
            token_mint: Token mint address
            snapshots: List of snapshot dicts with timestamp and metrics
            window_hours: Time window for computation

        Returns:
            TokenTrajectory with computed metrics
        """
        trajectory = TokenTrajectory(
            token_mint=token_mint,
            computed_at=datetime.now(timezone.utc),
            window_hours=window_hours,
        )

        if len(snapshots) < self.MIN_POINTS_FOR_VELOCITY:
            logger.warning(
                "insufficient_snapshots_for_trajectory",
                token_mint=token_mint[:8],
                snapshots=len(snapshots),
                required=self.MIN_POINTS_FOR_VELOCITY,
            )
            return trajectory

        # Sort snapshots by timestamp
        sorted_snapshots = sorted(
            snapshots, key=lambda s: s.get("timestamp", "")
        )

        # Compute each trajectory metric
        trajectory.accumulation_rate = self._compute_accumulation_rate(sorted_snapshots)
        trajectory.sell_acceleration = self._compute_sell_acceleration(sorted_snapshots)
        trajectory.liquidity_decay_rate = self._compute_liquidity_decay(sorted_snapshots)
        trajectory.whale_dispersion_rate = self._compute_whale_dispersion(sorted_snapshots)
        trajectory.coordination_persistence = self._compute_coordination_persistence(
            sorted_snapshots
        )
        trajectory.holder_churn_velocity = self._compute_holder_churn(sorted_snapshots)

        # Compute aggregate signals
        trajectory.overall_health_trend = self._compute_overall_trend(trajectory)
        trajectory.risk_trajectory = self._compute_risk_trajectory(trajectory)

        logger.info(
            "trajectory_computed",
            token_mint=token_mint[:8],
            snapshots=len(snapshots),
            health_trend=trajectory.overall_health_trend.value,
            risk=trajectory.risk_trajectory,
        )

        return trajectory

    def _compute_accumulation_rate(
        self, snapshots: list[dict]
    ) -> Optional[TrajectoryMetric]:
        """
        Compute accumulation rate trajectory.

        Measures change in top holder concentration over time.
        Positive = accumulation, Negative = distribution.
        """
        series = self._extract_series(
            snapshots, "top_10_concentration", default=0.0
        )

        if len(series) < self.MIN_POINTS_FOR_VELOCITY:
            return None

        velocity, acceleration = self._compute_derivatives(series)
        trend = self._classify_trend(velocity, acceleration)

        return TrajectoryMetric(
            name="accumulation_rate",
            current_value=series[-1].value if series else 0.0,
            velocity=velocity,
            acceleration=acceleration,
            trend=trend,
            confidence=self._compute_confidence(series),
            data_points=len(series),
            window_hours=self._compute_window_hours(series),
            last_updated=series[-1].timestamp if series else datetime.now(timezone.utc),
        )

    def _compute_sell_acceleration(
        self, snapshots: list[dict]
    ) -> Optional[TrajectoryMetric]:
        """
        Compute sell acceleration trajectory.

        Measures second derivative of sell volume.
        High acceleration = selling is speeding up.
        """
        series = self._extract_series(snapshots, "sell_volume", default=0.0)

        if len(series) < self.MIN_POINTS_FOR_ACCELERATION:
            return None

        velocity, acceleration = self._compute_derivatives(series)
        trend = self._classify_trend(velocity, acceleration)

        return TrajectoryMetric(
            name="sell_acceleration",
            current_value=series[-1].value if series else 0.0,
            velocity=velocity,
            acceleration=acceleration,
            trend=trend,
            confidence=self._compute_confidence(series),
            data_points=len(series),
            window_hours=self._compute_window_hours(series),
            last_updated=series[-1].timestamp if series else datetime.now(timezone.utc),
        )

    def _compute_liquidity_decay(
        self, snapshots: list[dict]
    ) -> Optional[TrajectoryMetric]:
        """
        Compute liquidity decay rate trajectory.

        Measures rate of liquidity drain.
        Negative velocity = liquidity leaving.
        """
        series = self._extract_series(snapshots, "total_liquidity_usd", default=0.0)

        if len(series) < self.MIN_POINTS_FOR_VELOCITY:
            return None

        velocity, acceleration = self._compute_derivatives(series)
        trend = self._classify_trend(velocity, acceleration)

        return TrajectoryMetric(
            name="liquidity_decay_rate",
            current_value=series[-1].value if series else 0.0,
            velocity=velocity,
            acceleration=acceleration,
            trend=trend,
            confidence=self._compute_confidence(series),
            data_points=len(series),
            window_hours=self._compute_window_hours(series),
            last_updated=series[-1].timestamp if series else datetime.now(timezone.utc),
        )

    def _compute_whale_dispersion(
        self, snapshots: list[dict]
    ) -> Optional[TrajectoryMetric]:
        """
        Compute whale dispersion rate trajectory.

        Measures rate at which large holders are distributing.
        Uses Gini coefficient: decreasing = more distribution.
        """
        series = self._extract_series(snapshots, "gini_coefficient", default=0.0)

        if len(series) < self.MIN_POINTS_FOR_VELOCITY:
            return None

        velocity, acceleration = self._compute_derivatives(series)
        # Invert: negative Gini velocity = dispersion (good)
        trend = self._classify_trend(-velocity, -acceleration)

        return TrajectoryMetric(
            name="whale_dispersion_rate",
            current_value=series[-1].value if series else 0.0,
            velocity=-velocity,  # Inverted
            acceleration=-acceleration,
            trend=trend,
            confidence=self._compute_confidence(series),
            data_points=len(series),
            window_hours=self._compute_window_hours(series),
            last_updated=series[-1].timestamp if series else datetime.now(timezone.utc),
        )

    def _compute_coordination_persistence(
        self, snapshots: list[dict]
    ) -> Optional[TrajectoryMetric]:
        """
        Compute coordination persistence trajectory.

        Measures how long coordination patterns persist.
        Uses coordination score from snapshots.
        """
        series = self._extract_series(snapshots, "coordination_score", default=0.0)

        if len(series) < self.MIN_POINTS_FOR_VELOCITY:
            return None

        velocity, acceleration = self._compute_derivatives(series)
        trend = self._classify_trend(velocity, acceleration)

        # Also compute persistence metric
        non_zero_count = sum(1 for p in series if p.value > 0.1)
        persistence_ratio = non_zero_count / len(series) if series else 0.0

        return TrajectoryMetric(
            name="coordination_persistence",
            current_value=persistence_ratio,
            velocity=velocity,
            acceleration=acceleration,
            trend=trend,
            confidence=self._compute_confidence(series),
            data_points=len(series),
            window_hours=self._compute_window_hours(series),
            last_updated=series[-1].timestamp if series else datetime.now(timezone.utc),
        )

    def _compute_holder_churn(
        self, snapshots: list[dict]
    ) -> Optional[TrajectoryMetric]:
        """
        Compute holder churn velocity trajectory.

        Measures rate of holder turnover.
        High churn = unstable holder base.
        """
        series = self._extract_series(snapshots, "holder_churn_rate", default=0.0)

        # If churn rate not available, estimate from holder count changes
        if not any(p.value > 0 for p in series):
            holder_series = self._extract_series(snapshots, "holder_count", default=0)
            if len(holder_series) >= 2:
                # Estimate churn as absolute change rate
                series = []
                for i in range(1, len(holder_series)):
                    delta = abs(holder_series[i].value - holder_series[i - 1].value)
                    avg = (holder_series[i].value + holder_series[i - 1].value) / 2
                    churn = delta / avg if avg > 0 else 0
                    series.append(
                        TimeSeriesPoint(
                            timestamp=holder_series[i].timestamp,
                            value=churn,
                        )
                    )

        if len(series) < self.MIN_POINTS_FOR_VELOCITY:
            return None

        velocity, acceleration = self._compute_derivatives(series)
        trend = self._classify_trend(velocity, acceleration)

        return TrajectoryMetric(
            name="holder_churn_velocity",
            current_value=series[-1].value if series else 0.0,
            velocity=velocity,
            acceleration=acceleration,
            trend=trend,
            confidence=self._compute_confidence(series),
            data_points=len(series),
            window_hours=self._compute_window_hours(series),
            last_updated=series[-1].timestamp if series else datetime.now(timezone.utc),
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_series(
        self,
        snapshots: list[dict],
        field: str,
        default: float = 0.0,
    ) -> list[TimeSeriesPoint]:
        """Extract time series from snapshots."""
        series = []

        for snapshot in snapshots:
            timestamp_str = snapshot.get("timestamp", "")
            if not timestamp_str:
                continue

            try:
                if isinstance(timestamp_str, datetime):
                    timestamp = timestamp_str
                else:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue

            value = snapshot.get(field, default)
            if value is None:
                value = default

            series.append(TimeSeriesPoint(timestamp=timestamp, value=float(value)))

        return series

    def _compute_derivatives(
        self, series: list[TimeSeriesPoint]
    ) -> tuple[float, float]:
        """
        Compute velocity (first derivative) and acceleration (second derivative).

        Uses central difference for interior points, forward/backward for endpoints.
        """
        if len(series) < 2:
            return 0.0, 0.0

        # Compute time deltas in hours
        times = []
        values = []

        for i, point in enumerate(series):
            if i == 0:
                times.append(0.0)
            else:
                delta = (point.timestamp - series[0].timestamp).total_seconds() / 3600
                times.append(delta)
            values.append(point.value)

        # First derivative: simple linear regression slope
        n = len(times)
        mean_t = sum(times) / n
        mean_v = sum(values) / n

        numerator = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
        denominator = sum((t - mean_t) ** 2 for t in times)

        velocity = numerator / denominator if denominator > 0 else 0.0

        # Second derivative: difference of velocities
        if len(series) < 4:
            acceleration = 0.0
        else:
            # Split into halves, compute velocity of each
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

    def _classify_trend(self, velocity: float, acceleration: float) -> TrajectoryTrend:
        """Classify trend from velocity and acceleration."""
        v_abs = abs(velocity)
        a_abs = abs(acceleration)

        # Check if effectively stable
        if v_abs < self.VELOCITY_THRESHOLD and a_abs < self.ACCELERATION_THRESHOLD:
            return TrajectoryTrend.STABLE

        # Check for volatility (high acceleration, low velocity)
        if a_abs > self.ACCELERATION_THRESHOLD * 10 and v_abs < self.VELOCITY_THRESHOLD:
            return TrajectoryTrend.VOLATILE

        # Classify direction and acceleration
        going_up = velocity > self.VELOCITY_THRESHOLD
        going_down = velocity < -self.VELOCITY_THRESHOLD
        speeding_up = acceleration > self.ACCELERATION_THRESHOLD
        slowing_down = acceleration < -self.ACCELERATION_THRESHOLD

        if going_up and speeding_up:
            return TrajectoryTrend.ACCELERATING_UP
        elif going_up and slowing_down:
            return TrajectoryTrend.DECELERATING_UP
        elif going_down and speeding_up:
            return TrajectoryTrend.DECELERATING_DOWN
        elif going_down and slowing_down:
            return TrajectoryTrend.ACCELERATING_DOWN
        else:
            return TrajectoryTrend.STABLE

    def _compute_confidence(self, series: list[TimeSeriesPoint]) -> float:
        """Compute confidence based on data quality."""
        if len(series) == 0:
            return 0.0

        # Factors: data points, regularity, completeness
        point_factor = min(1.0, len(series) / 10.0)  # Max at 10 points

        # Check regularity (consistent time intervals)
        if len(series) >= 2:
            intervals = []
            for i in range(1, len(series)):
                delta = (series[i].timestamp - series[i - 1].timestamp).total_seconds()
                intervals.append(delta)

            mean_interval = sum(intervals) / len(intervals)
            variance = sum((i - mean_interval) ** 2 for i in intervals) / len(intervals)
            std_dev = math.sqrt(variance)

            regularity_factor = 1.0 / (1.0 + std_dev / mean_interval) if mean_interval > 0 else 0.5
        else:
            regularity_factor = 0.5

        return point_factor * regularity_factor

    def _compute_window_hours(self, series: list[TimeSeriesPoint]) -> float:
        """Compute actual time window covered by series."""
        if len(series) < 2:
            return 0.0

        return (series[-1].timestamp - series[0].timestamp).total_seconds() / 3600

    def _compute_overall_trend(self, trajectory: TokenTrajectory) -> TrajectoryTrend:
        """Compute overall health trend from individual metrics."""
        trends = []

        for metric in [
            trajectory.accumulation_rate,
            trajectory.sell_acceleration,
            trajectory.liquidity_decay_rate,
            trajectory.whale_dispersion_rate,
            trajectory.holder_churn_velocity,
        ]:
            if metric is not None:
                trends.append(metric.trend)

        if not trends:
            return TrajectoryTrend.STABLE

        # Count trend types
        trend_counts = {}
        for trend in trends:
            trend_counts[trend] = trend_counts.get(trend, 0) + 1

        # Return most common
        return max(trend_counts, key=trend_counts.get)

    def _compute_risk_trajectory(self, trajectory: TokenTrajectory) -> str:
        """Compute risk trajectory classification."""
        positive_signals = 0
        negative_signals = 0

        # Accumulation: positive if distributing (negative velocity)
        if trajectory.accumulation_rate and trajectory.accumulation_rate.velocity < 0:
            positive_signals += 1
        elif trajectory.accumulation_rate and trajectory.accumulation_rate.velocity > 0.01:
            negative_signals += 1

        # Sell acceleration: negative if increasing
        if trajectory.sell_acceleration and trajectory.sell_acceleration.acceleration > 0:
            negative_signals += 1
        elif trajectory.sell_acceleration and trajectory.sell_acceleration.velocity < 0:
            positive_signals += 1

        # Liquidity: positive if increasing
        if trajectory.liquidity_decay_rate and trajectory.liquidity_decay_rate.velocity > 0:
            positive_signals += 1
        elif trajectory.liquidity_decay_rate and trajectory.liquidity_decay_rate.velocity < 0:
            negative_signals += 1

        # Whale dispersion: positive if dispersing
        if trajectory.whale_dispersion_rate and trajectory.whale_dispersion_rate.velocity > 0:
            positive_signals += 1
        elif trajectory.whale_dispersion_rate and trajectory.whale_dispersion_rate.velocity < 0:
            negative_signals += 1

        # Holder churn: negative if high
        if trajectory.holder_churn_velocity and trajectory.holder_churn_velocity.current_value > 0.2:
            negative_signals += 1
        elif trajectory.holder_churn_velocity and trajectory.holder_churn_velocity.current_value < 0.05:
            positive_signals += 1

        # Classify
        if positive_signals > negative_signals + 1:
            return "improving"
        elif negative_signals > positive_signals + 1:
            return "deteriorating"
        else:
            return "neutral"
