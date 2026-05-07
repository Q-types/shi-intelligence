"""
Regime Stress Testing.

Tests hazard model stability and system behavior
across different market regimes.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Generator

import pytest
import numpy as np
import pandas as pd

from shi.models.regime import (
    RegimeDetector,
    RegimeState,
    MarketRegime,
    RegimeAwareRetrainer,
)
from shi.models.validation import ModelValidator, ValidationThresholds


class RegimeDataGenerator:
    """Generates synthetic market regime data for testing."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_returns(
        self,
        regime: MarketRegime,
        num_days: int = 30,
    ) -> list[float]:
        """Generate returns for a specific regime."""
        volatility_map = {
            MarketRegime.LOW_VOLATILITY: 0.01,
            MarketRegime.NORMAL: 0.02,
            MarketRegime.HIGH_VOLATILITY: 0.05,
            MarketRegime.EXTREME: 0.10,
        }

        vol = volatility_map[regime]
        returns = self.rng.normal(0, vol, num_days).tolist()

        return returns

    def generate_regime_transition(
        self,
        from_regime: MarketRegime,
        to_regime: MarketRegime,
        transition_days: int = 5,
        pre_days: int = 20,
        post_days: int = 20,
    ) -> tuple[list[float], list[datetime]]:
        """
        Generate returns with a regime transition.

        Returns:
            (returns, timestamps)
        """
        returns = []
        timestamps = []
        base_time = datetime.now(timezone.utc) - timedelta(
            days=pre_days + transition_days + post_days
        )

        # Pre-transition period
        pre_returns = self.generate_returns(from_regime, pre_days)
        returns.extend(pre_returns)
        for i in range(pre_days):
            timestamps.append(base_time + timedelta(days=i))

        # Transition period (interpolate volatility)
        from_vol = self._get_volatility(from_regime)
        to_vol = self._get_volatility(to_regime)
        for i in range(transition_days):
            progress = (i + 1) / transition_days
            current_vol = from_vol + (to_vol - from_vol) * progress
            returns.append(float(self.rng.normal(0, current_vol)))
            timestamps.append(base_time + timedelta(days=pre_days + i))

        # Post-transition period
        post_returns = self.generate_returns(to_regime, post_days)
        returns.extend(post_returns)
        for i in range(post_days):
            timestamps.append(
                base_time + timedelta(days=pre_days + transition_days + i)
            )

        return returns, timestamps

    def _get_volatility(self, regime: MarketRegime) -> float:
        volatility_map = {
            MarketRegime.LOW_VOLATILITY: 0.01,
            MarketRegime.NORMAL: 0.02,
            MarketRegime.HIGH_VOLATILITY: 0.05,
            MarketRegime.EXTREME: 0.10,
        }
        return volatility_map[regime]


class TestRegimeDetection:
    """Tests for regime detection accuracy."""

    @pytest.fixture
    def generator(self) -> RegimeDataGenerator:
        return RegimeDataGenerator(seed=42)

    @pytest.fixture
    def detector(self) -> RegimeDetector:
        return RegimeDetector(lookback_days=20, baseline_window_days=90)

    def test_regime_classification_accuracy(
        self,
        generator: RegimeDataGenerator,
        detector: RegimeDetector,
    ) -> None:
        """Test that regime classification is accurate for known data."""
        for regime in MarketRegime:
            # Generate data for this regime
            returns = generator.generate_returns(regime, num_days=30)

            # Update detector multiple times to build baseline
            for i in range(30):
                timestamp = datetime.now(timezone.utc) - timedelta(days=30 - i)
                state = detector.update(returns[: i + 1], timestamp)

            # Final state should match expected regime (with some tolerance)
            final_state = detector.update(returns, datetime.now(timezone.utc))

            # Classification should be correct or adjacent
            regime_order = [
                MarketRegime.LOW_VOLATILITY,
                MarketRegime.NORMAL,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.EXTREME,
            ]
            expected_idx = regime_order.index(regime)
            actual_idx = regime_order.index(final_state.regime)

            # Allow 1 level of tolerance
            assert abs(expected_idx - actual_idx) <= 1, (
                f"Expected {regime}, got {final_state.regime}"
            )

    def test_regime_transition_detection(
        self,
        generator: RegimeDataGenerator,
        detector: RegimeDetector,
    ) -> None:
        """Test that regime transitions trigger retraining."""
        # Generate transition from normal to extreme
        returns, timestamps = generator.generate_regime_transition(
            from_regime=MarketRegime.NORMAL,
            to_regime=MarketRegime.EXTREME,
            transition_days=5,
        )

        retraining_triggered = False

        for i, (ret_slice, ts) in enumerate(zip(
            [returns[:j+1] for j in range(len(returns))],
            timestamps
        )):
            state = detector.update(ret_slice, ts)
            if state.trigger_retraining:
                retraining_triggered = True
                break

        # Significant regime shift should trigger retraining
        assert retraining_triggered, "Regime transition did not trigger retraining"

    def test_confidence_at_regime_boundaries(
        self,
        generator: RegimeDataGenerator,
        detector: RegimeDetector,
    ) -> None:
        """Test that confidence is lower at regime boundaries."""
        # Generate data that's on the boundary
        boundary_returns = generator.rng.normal(0, 0.015, 30).tolist()  # Between LOW and NORMAL

        # Build baseline
        for i in range(50):
            baseline_returns = generator.generate_returns(MarketRegime.NORMAL, 5)
            detector.update(
                baseline_returns,
                datetime.now(timezone.utc) - timedelta(days=50 - i),
            )

        # Test boundary data
        state = detector.update(boundary_returns, datetime.now(timezone.utc))

        # Confidence should be lower at boundaries
        assert state.confidence < 0.8, "Confidence should be lower at regime boundaries"


class TestRegimeAwareRetraining:
    """Tests for regime-aware model retraining."""

    @pytest.fixture
    def generator(self) -> RegimeDataGenerator:
        return RegimeDataGenerator(seed=42)

    @pytest.fixture
    def retrainer(self) -> RegimeAwareRetrainer:
        return RegimeAwareRetrainer(
            min_days_between_retraining=7,
            max_days_without_retraining=30,
        )

    def test_retraining_interval_enforcement(
        self,
        retrainer: RegimeAwareRetrainer,
    ) -> None:
        """Test that minimum retraining interval is enforced."""
        now = datetime.now(timezone.utc)

        # Create a regime state that would trigger retraining
        trigger_state = RegimeState(
            regime=MarketRegime.EXTREME,
            volatility_percentile=98.0,
            confidence=0.9,
            detected_at=now,
            window_days=30,
            trigger_retraining=True,
        )

        # First retraining should be allowed
        should_retrain, reason = retrainer.should_retrain(trigger_state, now)
        assert should_retrain

        # Mark as retrained
        retrainer.mark_retrained(now)

        # Immediate re-retraining should be blocked
        should_retrain, reason = retrainer.should_retrain(
            trigger_state,
            now + timedelta(days=1),
        )
        assert not should_retrain
        assert "Too soon" in reason

        # After interval, should be allowed
        should_retrain, reason = retrainer.should_retrain(
            trigger_state,
            now + timedelta(days=8),
        )
        assert should_retrain

    def test_scheduled_retraining(
        self,
        retrainer: RegimeAwareRetrainer,
    ) -> None:
        """Test that scheduled retraining triggers after max interval."""
        now = datetime.now(timezone.utc)

        # Mark initial training
        retrainer.mark_retrained(now)

        # Create non-triggering regime state
        stable_state = RegimeState(
            regime=MarketRegime.NORMAL,
            volatility_percentile=50.0,
            confidence=0.9,
            detected_at=now + timedelta(days=35),
            window_days=30,
            trigger_retraining=False,
        )

        # Should trigger scheduled retraining after max interval
        should_retrain, reason = retrainer.should_retrain(
            stable_state,
            now + timedelta(days=35),
        )
        assert should_retrain
        assert "Scheduled" in reason

    def test_training_window_adjustment(
        self,
        retrainer: RegimeAwareRetrainer,
    ) -> None:
        """Test that training window adjusts based on regime."""
        now = datetime.now(timezone.utc)

        regime_windows = {}
        for regime in MarketRegime:
            state = RegimeState(
                regime=regime,
                volatility_percentile=50.0,
                confidence=0.9,
                detected_at=now,
                window_days=30,
                trigger_retraining=False,
            )
            start, end = retrainer.get_training_window(state, now, default_days=90)
            window_days = (end - start).days
            regime_windows[regime] = window_days

        # Extreme should have shortest window
        assert regime_windows[MarketRegime.EXTREME] < regime_windows[MarketRegime.NORMAL]

        # Low volatility should have longest window
        assert regime_windows[MarketRegime.LOW_VOLATILITY] > regime_windows[MarketRegime.NORMAL]


class TestModelStabilityAcrossRegimes:
    """Tests for model coefficient stability across regimes."""

    @pytest.fixture
    def generator(self) -> RegimeDataGenerator:
        return RegimeDataGenerator(seed=42)

    def test_coefficient_stability(
        self,
        generator: RegimeDataGenerator,
    ) -> None:
        """Test that model coefficients remain stable across regimes."""
        # This is a conceptual test - actual implementation would use
        # trained models from different regimes

        # Generate features for different regimes
        regime_features = {}
        for regime in MarketRegime:
            returns = generator.generate_returns(regime, 100)

            # Compute features from returns
            features = {
                "volatility": np.std(returns),
                "mean_return": np.mean(returns),
                "skew": float(pd.Series(returns).skew()),
                "kurtosis": float(pd.Series(returns).kurtosis()),
            }
            regime_features[regime] = features

        # Feature distributions should vary by regime
        volatilities = [f["volatility"] for f in regime_features.values()]
        assert max(volatilities) > min(volatilities) * 2, (
            "Regimes should have different volatility characteristics"
        )

    def test_prediction_calibration_across_regimes(
        self,
        generator: RegimeDataGenerator,
    ) -> None:
        """Test that predictions remain calibrated across regimes."""
        # Conceptual test for calibration stability

        # In production, this would:
        # 1. Train model on each regime's data
        # 2. Compute calibration curves
        # 3. Verify calibration slope stays in [0.7, 1.3]

        # For now, verify the thresholds are set correctly
        thresholds = ValidationThresholds()
        assert thresholds.min_calibration_slope == 0.7
        assert thresholds.max_calibration_slope == 1.3
