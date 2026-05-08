"""
Capital Flow Forecasting (Time-Series Specialist + Supervised ML).

Predicts short-term capital flow direction and liquidity pressure.

IMPORTANT: This is a basic implementation for Sprint 1.
For production, integrate ARIMA, VAR, or LSTM models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy import stats
import structlog

logger = structlog.get_logger()


@dataclass
class CapitalFlowForecast:
    """Capital flow forecast with confidence intervals."""

    predicted_net_flow: float  # Predicted net buy/sell pressure
    confidence_interval_lower: float
    confidence_interval_upper: float
    horizon_hours: int
    forecast_timestamp: datetime

    # Liquidity stress probability
    liquidity_stress_probability: float

    # Feature importance (for explainability)
    top_features: dict[str, float]


@dataclass
class FlowFeatures:
    """Engineered features for capital flow prediction."""

    # Metric velocities
    dhhi_dt: float
    dgini_dt: float
    dchurn_dt: float

    # Holder dynamics
    new_holders_rate: float
    exiting_holders_rate: float
    whale_accumulation_rate: float

    # Graph features
    coordination_score: float
    network_density_change: float

    # Temporal features
    time_of_day: int  # Hour
    day_of_week: int


class CapitalFlowForecaster:
    """
    Simple capital flow forecaster using linear regression on flow features.

    NOTE: This is a baseline implementation. For production:
    - Use VAR (Vector AutoRegression) for multivariate time-series
    - Add ARIMA for univariate components
    - Consider LSTM for non-linear patterns
    """

    def __init__(
        self,
        lookback_hours: int = 24,
        confidence_level: float = 0.95,
    ):
        """
        Initialize forecaster.

        Args:
            lookback_hours: Historical window for feature extraction
            confidence_level: Confidence level for intervals (default 95%)
        """
        self.lookback_hours = lookback_hours
        self.confidence_level = confidence_level
        self._coefficients: Optional[npt.NDArray[np.float64]] = None
        self._intercept: float = 0.0
        self._residual_std: float = 1.0

    def fit(
        self,
        features: Sequence[FlowFeatures],
        net_flows: Sequence[float],
    ) -> None:
        """
        Fit forecaster on historical data.

        Args:
            features: Historical flow features
            net_flows: Corresponding net capital flows (positive = accumulation)
        """
        if len(features) != len(net_flows):
            raise ValueError("Features and net_flows must have same length")

        # Convert to matrix
        X = self._features_to_matrix(features)
        y = np.array(net_flows)

        # Fit linear regression
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(X, y)

        self._coefficients = model.coef_
        self._intercept = model.intercept_

        # Compute residual std for confidence intervals
        predictions = model.predict(X)
        residuals = y - predictions
        self._residual_std = float(np.std(residuals))

        logger.info(
            "forecaster_fitted",
            n_samples=len(features),
            r_squared=model.score(X, y),
            residual_std=self._residual_std,
        )

    def forecast(
        self,
        current_features: FlowFeatures,
        horizon_hours: int = 24,
    ) -> CapitalFlowForecast:
        """
        Forecast capital flow for given horizon.

        Args:
            current_features: Current feature values
            horizon_hours: Forecast horizon

        Returns:
            CapitalFlowForecast with prediction and confidence intervals
        """
        if self._coefficients is None:
            raise RuntimeError("Forecaster not fitted. Call fit() first.")

        # Convert features to vector
        x = self._features_to_matrix([current_features])[0]

        # Predict
        predicted = float(np.dot(self._coefficients, x) + self._intercept)

        # Confidence interval
        # Wider interval for longer horizons
        horizon_factor = np.sqrt(horizon_hours / 24)
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        margin = z_score * self._residual_std * horizon_factor

        ci_lower = predicted - margin
        ci_upper = predicted + margin

        # Liquidity stress probability (heuristic)
        # High stress if predicted outflow is large and confidence is high
        if predicted < 0:
            stress_prob = min(1.0, abs(predicted) / (3 * self._residual_std))
        else:
            stress_prob = 0.0

        # Feature importance (absolute coefficients)
        feature_names = [
            "dhhi_dt",
            "dgini_dt",
            "dchurn_dt",
            "new_holders_rate",
            "exiting_holders_rate",
            "whale_accumulation_rate",
            "coordination_score",
            "network_density_change",
        ]

        importance = {
            name: float(abs(coef))
            for name, coef in zip(feature_names, self._coefficients[:8])
        }

        # Top 3 features
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3])

        return CapitalFlowForecast(
            predicted_net_flow=predicted,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            horizon_hours=horizon_hours,
            forecast_timestamp=datetime.now(timezone.utc),
            liquidity_stress_probability=stress_prob,
            top_features=top_features,
        )

    def _features_to_matrix(
        self,
        features: Sequence[FlowFeatures],
    ) -> npt.NDArray[np.float64]:
        """Convert FlowFeatures to matrix."""
        rows = []
        for f in features:
            row = [
                f.dhhi_dt,
                f.dgini_dt,
                f.dchurn_dt,
                f.new_holders_rate,
                f.exiting_holders_rate,
                f.whale_accumulation_rate,
                f.coordination_score,
                f.network_density_change,
                np.sin(2 * np.pi * f.time_of_day / 24),  # Cyclic encoding
                np.cos(2 * np.pi * f.time_of_day / 24),
                np.sin(2 * np.pi * f.day_of_week / 7),
                np.cos(2 * np.pi * f.day_of_week / 7),
            ]
            rows.append(row)

        return np.array(rows)


def extract_flow_features_from_snapshots(
    snapshots: Sequence[dict[str, Any]],
    lookback_hours: int = 24,
) -> Optional[FlowFeatures]:
    """
    Extract flow features from metric snapshots.

    Args:
        snapshots: Recent metric snapshots (sorted by time)
        lookback_hours: Lookback window

    Returns:
        FlowFeatures or None if insufficient data
    """
    if not snapshots or len(snapshots) < 2:
        return None

    # Filter to lookback window
    cutoff = snapshots[-1]["timestamp"] - timedelta(hours=lookback_hours)
    recent = [s for s in snapshots if s["timestamp"] >= cutoff]

    if len(recent) < 2:
        return None

    # Compute velocities
    dhhi_dt = (recent[-1]["hhi"] - recent[0]["hhi"]) / (
        (recent[-1]["timestamp"] - recent[0]["timestamp"]).total_seconds() / 86400
    )

    dgini_dt = (recent[-1]["gini"] - recent[0]["gini"]) / (
        (recent[-1]["timestamp"] - recent[0]["timestamp"]).total_seconds() / 86400
    )

    dchurn_dt = 0.0
    if "churn_rate" in recent[-1] and "churn_rate" in recent[0]:
        dchurn_dt = (recent[-1]["churn_rate"] - recent[0]["churn_rate"]) / (
            (recent[-1]["timestamp"] - recent[0]["timestamp"]).total_seconds() / 86400
        )

    # Holder dynamics (simplified - would need holder count tracking)
    holder_change = recent[-1]["holder_count"] - recent[0]["holder_count"]
    time_delta_hours = (recent[-1]["timestamp"] - recent[0]["timestamp"]).total_seconds() / 3600
    holder_change_rate = holder_change / time_delta_hours if time_delta_hours > 0 else 0.0

    new_holders_rate = max(0, holder_change_rate)
    exiting_holders_rate = max(0, -holder_change_rate)

    # Whale accumulation (change in whale dominance)
    whale_change = recent[-1]["whale_dominance"] - recent[0]["whale_dominance"]
    whale_accumulation_rate = whale_change / (time_delta_hours / 24) if time_delta_hours > 0 else 0.0

    # Graph features (would need graph metrics tracked)
    coordination_score = recent[-1].get("coordination_score", 0.0)
    network_density_change = 0.0  # Placeholder - needs graph tracking

    # Temporal features
    last_timestamp = recent[-1]["timestamp"]
    time_of_day = last_timestamp.hour
    day_of_week = last_timestamp.weekday()

    return FlowFeatures(
        dhhi_dt=dhhi_dt,
        dgini_dt=dgini_dt,
        dchurn_dt=dchurn_dt,
        new_holders_rate=new_holders_rate,
        exiting_holders_rate=exiting_holders_rate,
        whale_accumulation_rate=whale_accumulation_rate,
        coordination_score=coordination_score,
        network_density_change=network_density_change,
        time_of_day=time_of_day,
        day_of_week=day_of_week,
    )
