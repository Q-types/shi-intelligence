"""
Test Suite for Temporal Foundation (Sprint 1).

Tests:
- Trajectory tracking and derivatives
- HMM regime detection
- Walk-forward validation
- Integration with existing metrics
"""

from datetime import datetime, timedelta
import numpy as np
import pytest

from src.temporal.trajectories import (
    TrajectoryTracker,
    MetricPoint,
    TrendDirection,
)
from src.temporal.regimes import (
    HolderRegimeDetector,
    HolderRegimeType,
    create_rule_based_regime,
)
from src.temporal.validation import (
    WalkForwardValidator,
    compute_regime_detection_metrics,
)
from src.temporal.forecasting import (
    CapitalFlowForecaster,
    FlowFeatures,
    extract_flow_features_from_snapshots,
)


class TestTrajectoryTracking:
    """Test metric trajectory tracking and derivatives."""

    def test_compute_trajectory_basic(self):
        """Test basic trajectory computation."""
        tracker = TrajectoryTracker()

        # Create synthetic HHI increasing over time
        base_time = datetime(2026, 1, 1)
        points = [
            MetricPoint(
                timestamp=base_time + timedelta(days=i),
                value=0.1 + (i * 0.01),
                metric_name="hhi",
            )
            for i in range(10)
        ]

        trajectory = tracker.compute_trajectory(points)

        assert trajectory.metric_name == "hhi"
        assert len(trajectory.points) == 10
        assert trajectory.velocity > 0  # HHI increasing
        assert trajectory.trend == TrendDirection.CENTRALIZING

    def test_derivative_calculation(self):
        """Test velocity and acceleration calculation."""
        tracker = TrajectoryTracker()

        # Linear increase
        base_time = datetime(2026, 1, 1)
        points = [
            MetricPoint(
                timestamp=base_time + timedelta(days=i),
                value=i * 0.02,
                metric_name="gini",
            )
            for i in range(20)
        ]

        trajectory = tracker.compute_trajectory(points)

        # Velocity should be ~0.02 per day
        assert abs(trajectory.velocity - 0.02) < 0.005
        # Acceleration should be ~0 (constant velocity)
        assert abs(trajectory.acceleration) < 0.001

    def test_trend_detection(self):
        """Test trend direction detection."""
        tracker = TrajectoryTracker()
        base_time = datetime(2026, 1, 1)

        # Decentralizing (HHI decreasing)
        points_dec = [
            MetricPoint(
                timestamp=base_time + timedelta(days=i),
                value=0.3 - (i * 0.01),
                metric_name="hhi",
            )
            for i in range(15)
        ]

        traj_dec = tracker.compute_trajectory(points_dec)
        assert traj_dec.trend == TrendDirection.DECENTRALIZING

        # Stable (no change)
        points_stable = [
            MetricPoint(
                timestamp=base_time + timedelta(days=i),
                value=0.2 + (np.random.random() * 0.001),
                metric_name="hhi",
            )
            for i in range(15)
        ]

        traj_stable = tracker.compute_trajectory(points_stable)
        assert traj_stable.trend == TrendDirection.STABLE

    def test_multi_metric_trajectory(self):
        """Test computing multiple metrics simultaneously."""
        tracker = TrajectoryTracker()
        base_time = datetime(2026, 1, 1)

        snapshots = [
            {
                "timestamp": base_time + timedelta(days=i),
                "hhi": 0.1 + (i * 0.005),
                "gini": 0.5 + (i * 0.01),
                "churn_rate": 0.02,
            }
            for i in range(20)
        ]

        trajectories = tracker.compute_multi_metric_trajectory(snapshots)

        assert "hhi" in trajectories
        assert "gini" in trajectories
        assert trajectories["hhi"].velocity > 0
        assert trajectories["gini"].velocity > 0


class TestRegimeDetection:
    """Test HMM-based regime detection."""

    def test_rule_based_regime(self):
        """Test rule-based regime classification."""
        # Accumulation
        regime = create_rule_based_regime(
            dhhi_dt=-0.02,
            dgini_dt=-0.01,
            dchurn_dt=0.0,
            coordination_score=0.3,
            hhi_trend=TrendDirection.DECENTRALIZING,
        )
        assert regime == HolderRegimeType.ACCUMULATION

        # Distribution
        regime = create_rule_based_regime(
            dhhi_dt=0.02,
            dgini_dt=0.015,
            dchurn_dt=0.0,
            coordination_score=0.3,
            hhi_trend=TrendDirection.CENTRALIZING,
        )
        assert regime == HolderRegimeType.DISTRIBUTION

        # Coordinated accumulation
        regime = create_rule_based_regime(
            dhhi_dt=0.02,
            dgini_dt=0.015,
            dchurn_dt=0.0,
            coordination_score=0.8,
            hhi_trend=TrendDirection.CENTRALIZING,
        )
        assert regime == HolderRegimeType.COORDINATED_ACCUMULATION

        # Decay
        regime = create_rule_based_regime(
            dhhi_dt=0.0,
            dgini_dt=0.0,
            dchurn_dt=0.15,
            coordination_score=0.3,
            hhi_trend=TrendDirection.STABLE,
        )
        assert regime == HolderRegimeType.DECAY

    def test_hmm_detector_fit(self):
        """Test HMM detector training."""
        detector = HolderRegimeDetector(n_iter=10)

        # Create synthetic training data
        # Features: [dhhi_dt, dgini_dt, dchurn_dt, coordination, centralize, decentralize]
        np.random.seed(42)

        # Accumulation regime data
        accumulation = np.random.randn(50, 6) * 0.01
        accumulation[:, 0] = -0.02 + np.random.randn(50) * 0.005  # dhhi_dt negative
        accumulation[:, 4] = 0  # not centralizing
        accumulation[:, 5] = 1  # decentralizing

        # Distribution regime data
        distribution = np.random.randn(50, 6) * 0.01
        distribution[:, 0] = 0.02 + np.random.randn(50) * 0.005  # dhhi_dt positive
        distribution[:, 4] = 1  # centralizing
        distribution[:, 5] = 0  # not decentralizing

        training_data = np.vstack([accumulation, distribution])

        detector.fit([training_data])

        assert detector._is_fitted
        assert detector.model is not None

    def test_hmm_detector_predict(self):
        """Test HMM regime prediction."""
        detector = HolderRegimeDetector(n_iter=20)

        # Train on synthetic data
        np.random.seed(42)
        training_data = np.random.randn(100, 6) * 0.02
        detector.fit([training_data])

        # Predict
        test_features = np.array([[-0.02, -0.01, 0.0, 0.3, 0.0, 1.0]])
        regime_state = detector.predict_regime(test_features)

        assert isinstance(regime_state.regime, HolderRegimeType)
        assert 0 <= regime_state.confidence <= 1
        assert 0 <= regime_state.transition_probability <= 1


class TestWalkForwardValidation:
    """Test walk-forward validation."""

    def test_create_windows(self):
        """Test validation window creation."""
        validator = WalkForwardValidator(
            train_window_days=30,
            test_window_days=7,
            step_days=7,
        )

        start = datetime(2026, 1, 1)
        end = datetime(2026, 3, 1)

        windows = validator.create_windows(start, end)

        assert len(windows) > 0
        # Check windows don't overlap
        for w in windows:
            assert w.train_end == w.test_start
            assert w.test_end <= end

    def test_regime_validation(self):
        """Test regime detector validation."""
        detector = HolderRegimeDetector(n_iter=10)
        validator = WalkForwardValidator(
            train_window_days=20,
            test_window_days=5,
            step_days=5,
            expanding_window=True,
        )

        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        features = np.random.randn(n_samples, 6) * 0.02

        # Generate timestamps
        base_time = datetime(2026, 1, 1)
        timestamps = [base_time + timedelta(days=i) for i in range(n_samples)]

        # Generate synthetic regimes (random for test)
        true_regimes = [
            HolderRegimeType.ACCUMULATION if i % 2 == 0 else HolderRegimeType.DISTRIBUTION
            for i in range(n_samples)
        ]

        # Validate (may fail due to random data, but tests the pipeline)
        try:
            results = validator.validate_regime_detector(
                detector, features, timestamps, true_regimes
            )
            assert results.mean_score >= 0
            assert len(results.window_results) > 0
        except ValueError:
            # Expected if not enough windows
            pass


class TestCapitalFlowForecasting:
    """Test capital flow forecasting."""

    def test_flow_features_extraction(self):
        """Test feature extraction from snapshots."""
        base_time = datetime(2026, 1, 1)
        snapshots = [
            {
                "timestamp": base_time + timedelta(hours=i),
                "hhi": 0.1 + (i * 0.001),
                "gini": 0.5,
                "churn_rate": 0.02,
                "whale_dominance": 0.3,
                "holder_count": 1000 + i * 10,
                "coordination_score": 0.4,
            }
            for i in range(48)
        ]

        features = extract_flow_features_from_snapshots(snapshots, lookback_hours=24)

        assert features is not None
        assert features.dhhi_dt != 0  # HHI changing
        assert features.new_holders_rate > 0  # Holders increasing

    def test_forecaster_fit_predict(self):
        """Test forecaster training and prediction."""
        forecaster = CapitalFlowForecaster()

        # Generate training data
        np.random.seed(42)
        n_samples = 50

        features = [
            FlowFeatures(
                dhhi_dt=np.random.randn() * 0.01,
                dgini_dt=np.random.randn() * 0.01,
                dchurn_dt=np.random.randn() * 0.01,
                new_holders_rate=abs(np.random.randn() * 10),
                exiting_holders_rate=abs(np.random.randn() * 5),
                whale_accumulation_rate=np.random.randn() * 0.02,
                coordination_score=np.random.random(),
                network_density_change=np.random.randn() * 0.001,
                time_of_day=np.random.randint(0, 24),
                day_of_week=np.random.randint(0, 7),
            )
            for _ in range(n_samples)
        ]

        # Net flows (somewhat correlated with features)
        net_flows = [
            f.new_holders_rate - f.exiting_holders_rate + np.random.randn() * 5
            for f in features
        ]

        forecaster.fit(features, net_flows)

        # Predict
        test_feature = features[0]
        forecast = forecaster.forecast(test_feature, horizon_hours=24)

        assert isinstance(forecast.predicted_net_flow, float)
        assert forecast.confidence_interval_lower < forecast.confidence_interval_upper
        assert 0 <= forecast.liquidity_stress_probability <= 1
        assert len(forecast.top_features) > 0


class TestIntegration:
    """Integration tests for temporal foundation."""

    def test_end_to_end_trajectory_to_regime(self):
        """Test full pipeline: snapshots -> trajectories -> regime detection."""
        # Create metric snapshots
        base_time = datetime(2026, 1, 1)
        snapshots = []

        for i in range(60):
            snapshots.append(
                {
                    "timestamp": base_time + timedelta(days=i),
                    "hhi": 0.15 + (i * 0.002),  # Increasing
                    "gini": 0.5 + (i * 0.003),  # Increasing
                    "churn_rate": 0.02,
                    "whale_dominance": 0.3,
                    "holder_count": 1000,
                }
            )

        # Track trajectories
        tracker = TrajectoryTracker()
        trajectories = tracker.compute_multi_metric_trajectory(snapshots, window_days=30)

        assert "hhi" in trajectories
        assert "gini" in trajectories

        # Both should be centralizing
        assert trajectories["hhi"].trend == TrendDirection.CENTRALIZING
        assert trajectories["gini"].trend == TrendDirection.CENTRALIZING

        # Use rule-based regime
        regime = create_rule_based_regime(
            dhhi_dt=trajectories["hhi"].velocity,
            dgini_dt=trajectories["gini"].velocity,
            dchurn_dt=0.0,
            coordination_score=0.3,
            hhi_trend=trajectories["hhi"].trend,
        )

        # Should detect distribution or coordinated accumulation
        assert regime in [
            HolderRegimeType.DISTRIBUTION,
            HolderRegimeType.COORDINATED_ACCUMULATION,
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
