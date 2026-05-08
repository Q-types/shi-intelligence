"""
Market Regime Detection.

Per INITIAL_PROMPT:
Crypto markets are non-stationary. The system must:
- Detect volatility regime shifts
- Test proportional hazard stability across regimes
- Support time-sliced retraining
- Log regime classification state in outputs
- Trigger retraining if regime drift detected
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Sequence

import numpy as np
import structlog

logger = structlog.get_logger()


class MarketRegime(Enum):
    """Market volatility regime classification."""

    LOW_VOLATILITY = "low_volatility"
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    EXTREME = "extreme"


@dataclass
class RegimeState:
    """Current regime state with metadata."""

    regime: MarketRegime
    volatility_percentile: float  # 0-100
    confidence: float  # 0-1
    detected_at: datetime
    window_days: int
    trigger_retraining: bool


@dataclass
class RegimeTransition:
    """Detected regime transition."""

    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_time: datetime
    volatility_change: float


class RegimeDetector:
    """
    Detects market regime shifts based on volatility.

    Uses rolling volatility to classify regimes and detect shifts.
    """

    # Volatility percentile thresholds
    THRESHOLDS = {
        MarketRegime.LOW_VOLATILITY: (0, 25),
        MarketRegime.NORMAL: (25, 75),
        MarketRegime.HIGH_VOLATILITY: (75, 95),
        MarketRegime.EXTREME: (95, 100),
    }

    def __init__(
        self,
        lookback_days: int = 30,
        baseline_window_days: int = 180,
    ):
        self.lookback_days = lookback_days
        self.baseline_window_days = baseline_window_days
        self._historical_volatility: list[tuple[datetime, float]] = []
        self._current_regime: MarketRegime = MarketRegime.NORMAL
        self._last_transition: datetime | None = None

    def update(
        self,
        returns: Sequence[float],
        timestamp: datetime,
    ) -> RegimeState:
        """
        Update regime detection with new return data.

        Args:
            returns: Recent price returns (daily or intraday)
            timestamp: Current timestamp

        Returns:
            Current RegimeState
        """
        if len(returns) < 2:
            return RegimeState(
                regime=MarketRegime.NORMAL,
                volatility_percentile=50.0,
                confidence=0.0,
                detected_at=timestamp,
                window_days=self.lookback_days,
                trigger_retraining=False,
            )

        # Compute realized volatility
        volatility = float(np.std(returns) * np.sqrt(252))  # Annualized

        # Store historical
        self._historical_volatility.append((timestamp, volatility))

        # Keep only baseline window
        cutoff = timestamp - timedelta(days=self.baseline_window_days)
        self._historical_volatility = [
            (t, v) for t, v in self._historical_volatility if t > cutoff
        ]

        # Compute percentile
        historical_vols = [v for _, v in self._historical_volatility]
        if len(historical_vols) < 10:
            percentile = 50.0
        else:
            percentile = float(
                np.percentile(historical_vols, [0, 25, 50, 75, 100]).searchsorted(volatility)
            ) / 4 * 100

        # Classify regime
        new_regime = self._classify_regime(percentile)

        # Check for transition
        trigger_retraining = False
        if new_regime != self._current_regime:
            logger.info(
                "regime_transition",
                from_regime=self._current_regime.value,
                to_regime=new_regime.value,
                percentile=percentile,
            )

            # Trigger retraining on significant shifts
            if self._is_significant_shift(self._current_regime, new_regime):
                trigger_retraining = True

            self._last_transition = timestamp
            self._current_regime = new_regime

        # Compute confidence based on how clearly we're in this regime
        confidence = self._compute_confidence(percentile, new_regime)

        return RegimeState(
            regime=new_regime,
            volatility_percentile=percentile,
            confidence=confidence,
            detected_at=timestamp,
            window_days=self.lookback_days,
            trigger_retraining=trigger_retraining,
        )

    def _classify_regime(self, percentile: float) -> MarketRegime:
        """Classify regime based on volatility percentile."""
        for regime, (low, high) in self.THRESHOLDS.items():
            if low <= percentile < high:
                return regime
        return MarketRegime.EXTREME

    def _compute_confidence(
        self,
        percentile: float,
        regime: MarketRegime,
    ) -> float:
        """Compute confidence in regime classification."""
        low, high = self.THRESHOLDS[regime]
        range_size = high - low

        if range_size == 0:
            return 1.0

        # Distance from boundaries
        dist_from_low = percentile - low
        dist_from_high = high - percentile
        min_dist = min(dist_from_low, dist_from_high)

        # Confidence higher when further from boundaries
        return min(1.0, min_dist / (range_size / 2))

    def _is_significant_shift(
        self,
        from_regime: MarketRegime,
        to_regime: MarketRegime,
    ) -> bool:
        """Determine if regime shift warrants retraining."""
        # Define regime ordering for magnitude comparison
        ordering = [
            MarketRegime.LOW_VOLATILITY,
            MarketRegime.NORMAL,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.EXTREME,
        ]

        from_idx = ordering.index(from_regime)
        to_idx = ordering.index(to_regime)

        # Significant if jumping more than one level
        return abs(to_idx - from_idx) > 1

    def get_regime_history(
        self,
        since: datetime,
    ) -> list[tuple[datetime, MarketRegime, float]]:
        """Get regime history with timestamps and volatility."""
        return [
            (ts, self._classify_regime(self._vol_to_percentile(vol)), vol)
            for ts, vol in self._historical_volatility
            if ts >= since
        ]

    def _vol_to_percentile(self, vol: float) -> float:
        """Convert volatility to percentile based on history."""
        vols = [v for _, v in self._historical_volatility]
        if not vols:
            return 50.0
        return float(np.searchsorted(sorted(vols), vol) / len(vols) * 100)


class RegimeAwareRetrainer:
    """
    Manages model retraining based on regime shifts.

    Per INITIAL_PROMPT:
    - Support time-sliced retraining
    - Trigger retraining if regime drift detected
    """

    def __init__(
        self,
        min_days_between_retraining: int = 7,
        max_days_without_retraining: int = 30,
    ):
        self.min_days_between = min_days_between_retraining
        self.max_days_without = max_days_without_retraining
        self._last_retrain: datetime | None = None

    def should_retrain(
        self,
        regime_state: RegimeState,
        current_time: datetime,
    ) -> tuple[bool, str]:
        """
        Determine if model should be retrained.

        Returns:
            (should_retrain, reason)
        """
        # Check minimum interval
        if self._last_retrain:
            days_since = (current_time - self._last_retrain).days
            if days_since < self.min_days_between:
                return False, f"Too soon (last retrain {days_since}d ago)"
            days_since_float: float = float(days_since)
        else:
            days_since_float = float("inf")

        # Check regime-triggered retraining
        if regime_state.trigger_retraining:
            return True, f"Regime shift to {regime_state.regime.value}"

        # Check max interval
        if days_since_float > self.max_days_without:
            return True, f"Scheduled retraining ({days_since}d since last)"

        return False, "No retraining needed"

    def mark_retrained(self, timestamp: datetime) -> None:
        """Mark that retraining occurred."""
        self._last_retrain = timestamp
        logger.info("model_retrained", timestamp=timestamp.isoformat())

    def get_training_window(
        self,
        regime_state: RegimeState,
        current_time: datetime,
        default_days: int = 90,
    ) -> tuple[datetime, datetime]:
        """
        Get training data window based on regime.

        In high volatility, use shorter window for adaptation.
        In low volatility, use longer window for stability.
        """
        if regime_state.regime == MarketRegime.EXTREME:
            window_days = default_days // 2
        elif regime_state.regime == MarketRegime.HIGH_VOLATILITY:
            window_days = int(default_days * 0.75)
        elif regime_state.regime == MarketRegime.LOW_VOLATILITY:
            window_days = int(default_days * 1.5)
        else:
            window_days = default_days

        end_time = current_time
        start_time = current_time - timedelta(days=window_days)

        return start_time, end_time
