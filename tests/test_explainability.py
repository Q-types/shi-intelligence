"""
Unit tests for explainability module (Sprint 4).

Tests SHAP explainer, narratives, forecasting, and dashboard data structures.
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

from src.explainability import (
    MockSHAPExplainer,
    create_mock_explainer,
    ExplanationType,
    NarrativeGenerator,
    RiskLevel,
    CapitalFlowForecaster,
    MockForecaster,
    create_sample_intelligence,
    TokenIntelligence,
    RiskFactor,
    ActionableInsight,
)
from src.temporal.regimes import HolderRegimeType


class TestSHAPExplainer:
    """Test SHAP explainer functionality."""

    def test_mock_explainer_initialization(self):
        """Test mock explainer can be initialized."""
        feature_names = ["hhi", "gini", "churn_rate"]
        explainer = MockSHAPExplainer(feature_names, baseline_value=0.5)

        assert explainer.feature_names == feature_names
        assert explainer.baseline_value == 0.5
        assert explainer._fitted is True

    def test_mock_explainer_explain(self):
        """Test mock explainer generates valid explanations."""
        feature_names = ["hhi", "gini", "churn_rate", "top10_pct"]
        explainer = MockSHAPExplainer(feature_names)

        # Generate features
        features = np.array([0.7, 0.6, 0.3, 0.8])

        # Get explanation
        explanation = explainer.explain(
            features,
            top_k=3,
            explanation_type=ExplanationType.RISK_SCORE,
            uncertainty=True,
        )

        # Assertions
        assert explanation.predicted_value >= 0.0
        assert explanation.predicted_value <= 1.0
        assert explanation.baseline_value == 0.5
        assert len(explanation.top_contributors) == 3
        assert len(explanation.all_contributions) == 4

        # Check uncertainty
        assert explanation.prediction_std is not None
        assert explanation.confidence_interval is not None
        ci_lower, ci_upper = explanation.confidence_interval
        assert ci_lower < explanation.predicted_value < ci_upper

    def test_feature_contribution_structure(self):
        """Test feature contribution has required fields."""
        explainer = MockSHAPExplainer(["hhi", "gini"])
        features = np.array([0.7, 0.6])

        explanation = explainer.explain(features)
        contrib = explanation.top_contributors[0]

        assert hasattr(contrib, "feature_name")
        assert hasattr(contrib, "shap_value")
        assert hasattr(contrib, "feature_value")
        assert hasattr(contrib, "baseline_value")
        assert hasattr(contrib, "contribution_pct")

        # Check contribution percentage is valid
        assert 0 <= contrib.contribution_pct <= 100

    def test_positive_and_negative_contributors(self):
        """Test separation of positive and negative contributors."""
        explainer = MockSHAPExplainer(["hhi", "gini", "churn_rate"])
        features = np.array([0.8, 0.2, 0.5])

        explanation = explainer.explain(features)

        # Check properties
        positive = explanation.positive_contributors
        negative = explanation.negative_contributors

        for contrib in positive:
            assert contrib.shap_value > 0

        for contrib in negative:
            assert contrib.shap_value < 0

    def test_total_shap_magnitude(self):
        """Test total SHAP magnitude calculation."""
        explainer = MockSHAPExplainer(["hhi", "gini"])
        features = np.array([0.7, 0.6])

        explanation = explainer.explain(features)
        total_mag = explanation.total_shap_magnitude

        # Should be sum of absolute values
        expected = sum(abs(v) for v in explanation.all_contributions.values())
        assert abs(total_mag - expected) < 1e-6

    def test_create_mock_explainer(self):
        """Test factory function for mock explainer."""
        explainer = create_mock_explainer(
            feature_names=["hhi", "gini"],
            baseline_value=0.6,
        )

        assert isinstance(explainer, MockSHAPExplainer)
        assert explainer.baseline_value == 0.6


class TestNarrativeGenerator:
    """Test narrative generation."""

    def test_initialization(self):
        """Test narrative generator can be initialized."""
        generator = NarrativeGenerator(verbose=True)
        assert generator.verbose is True

        generator_simple = NarrativeGenerator(verbose=False)
        assert generator_simple.verbose is False

    def test_risk_narrative_generation(self):
        """Test risk narrative generation from SHAP explanation."""
        explainer = MockSHAPExplainer(["hhi", "gini", "churn_rate"])
        features = np.array([0.7, 0.6, 0.3])
        explanation = explainer.explain(features, uncertainty=True)

        generator = NarrativeGenerator()
        narrative = generator.generate_risk_narrative(
            explanation,
            token_symbol="TEST",
        )

        # Check structure
        assert isinstance(narrative.risk_level, RiskLevel)
        assert narrative.summary is not None
        assert len(narrative.summary) > 0
        assert narrative.confidence is not None
        assert len(narrative.key_drivers) > 0
        assert len(narrative.actionable_insights) > 0

    def test_risk_level_classification(self):
        """Test risk level is correctly classified."""
        generator = NarrativeGenerator()

        # Test different risk scores
        test_cases = [
            (0.1, RiskLevel.VERY_LOW),
            (0.3, RiskLevel.LOW),
            (0.5, RiskLevel.MODERATE),
            (0.7, RiskLevel.HIGH),
            (0.9, RiskLevel.VERY_HIGH),
        ]

        for score, expected_level in test_cases:
            level = generator._classify_risk_level(score)
            assert level == expected_level

    def test_confidence_description(self):
        """Test confidence description based on uncertainty."""
        generator = NarrativeGenerator()

        # Mock explanation with low uncertainty
        explainer = MockSHAPExplainer(["hhi"])
        features = np.array([0.5])
        explanation = explainer.explain(features, uncertainty=True)

        # Override std for testing
        explanation.prediction_std = 0.03
        confidence = generator._describe_confidence(explanation)
        assert "high" in confidence.lower() or "very high" in confidence.lower()

        # High uncertainty
        explanation.prediction_std = 0.20
        confidence = generator._describe_confidence(explanation)
        assert "low" in confidence.lower() or "uncertainty" in confidence.lower()

    def test_regime_narrative(self):
        """Test regime change narrative generation."""
        generator = NarrativeGenerator()

        narrative = generator.generate_regime_narrative(
            from_regime=HolderRegimeType.STABLE,
            to_regime=HolderRegimeType.DISTRIBUTION,
            confidence=0.85,
        )

        assert narrative.transition_summary is not None
        assert "stable" in narrative.transition_summary.lower()
        assert "distribution" in narrative.transition_summary.lower()
        assert narrative.confidence is not None
        assert len(narrative.implications) > 0

    def test_anomaly_narrative(self):
        """Test anomaly detection narrative."""
        generator = NarrativeGenerator()

        # Create mock features
        explainer = MockSHAPExplainer(["clustering_coefficient", "betweenness_centrality"])
        features = np.array([0.9, 0.8])
        explanation = explainer.explain(features)

        narrative = generator.generate_anomaly_narrative(
            anomaly_score=-0.85,
            features=explanation.top_contributors,
            wallet_address="7xKXtg2CW87d...",
        )

        assert narrative.anomaly_type is not None
        assert len(narrative.evidence) > 0
        assert narrative.severity is not None
        assert narrative.recommended_action is not None
        assert narrative.confidence is not None

    def test_verbose_mode_includes_technical_details(self):
        """Test verbose mode includes technical details."""
        explainer = MockSHAPExplainer(["hhi", "gini"])
        features = np.array([0.7, 0.6])
        explanation = explainer.explain(features)

        # Non-verbose
        generator_simple = NarrativeGenerator(verbose=False)
        narrative_simple = generator_simple.generate_risk_narrative(explanation)
        assert narrative_simple.technical_details is None

        # Verbose
        generator_verbose = NarrativeGenerator(verbose=True)
        narrative_verbose = generator_verbose.generate_risk_narrative(explanation)
        assert narrative_verbose.technical_details is not None
        assert len(narrative_verbose.technical_details) > 0


class TestCapitalFlowForecaster:
    """Test capital flow forecasting."""

    def test_forecaster_initialization(self):
        """Test forecaster can be initialized."""
        forecaster = CapitalFlowForecaster(smoothing_alpha=0.3)
        assert forecaster.smoothing_alpha == 0.3
        assert forecaster.confidence_level == 0.95

    def test_forecast_generation(self):
        """Test forecast generation with valid data."""
        forecaster = CapitalFlowForecaster()

        # Generate synthetic historical data
        np.random.seed(42)
        historical_flows = np.cumsum(np.random.normal(0, 10, 30)) + 100
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        timestamps = [base_time + timedelta(days=i) for i in range(30)]

        # Generate forecast
        forecast = forecaster.forecast_capital_flows(
            historical_flows=historical_flows,
            timestamps=timestamps,
            horizon_days=7,
            token_mint="test_token",
        )

        # Assertions
        assert forecast.token_mint == "test_token"
        assert forecast.horizon_days == 7
        assert len(forecast.net_flow_forecast) == 7
        assert len(forecast.inflow_forecast) == 7
        assert len(forecast.outflow_forecast) == 7

        # Check forecast points structure
        point = forecast.net_flow_forecast[0]
        assert hasattr(point, "timestamp")
        assert hasattr(point, "predicted_value")
        assert hasattr(point, "confidence_lower")
        assert hasattr(point, "confidence_upper")
        assert hasattr(point, "prediction_std")

        # Confidence interval should be valid
        assert point.confidence_lower < point.predicted_value < point.confidence_upper

    def test_forecast_requires_minimum_data(self):
        """Test forecast raises error with insufficient data."""
        forecaster = CapitalFlowForecaster()

        # Only 5 days of data
        historical_flows = np.array([1, 2, 3, 4, 5])
        timestamps = [datetime.now(timezone.utc) - timedelta(days=i) for i in range(5)]

        with pytest.raises(ValueError, match="at least 7 days"):
            forecaster.forecast_capital_flows(historical_flows, timestamps)

    def test_backtest(self):
        """Test backtesting functionality."""
        forecaster = CapitalFlowForecaster()

        np.random.seed(42)
        historical_flows = np.cumsum(np.random.normal(0, 10, 30)) + 100
        timestamps = [datetime.now(timezone.utc) - timedelta(days=30-i) for i in range(30)]

        # Run backtest
        result = forecaster.backtest(
            historical_flows=historical_flows,
            timestamps=timestamps,
            test_periods=7,
            forecast_horizon=1,
        )

        # Assertions
        assert result.mape >= 0
        assert result.rmse >= 0
        assert result.mae >= 0
        assert 0 <= result.coverage <= 100

        assert len(result.period_predictions) == 7
        assert len(result.period_actuals) == 7
        assert len(result.period_errors) == 7

    def test_forecast_quality_evaluation(self):
        """Test forecast quality classification."""
        forecaster = CapitalFlowForecaster()

        assert "Excellent" in forecaster.evaluate_forecast_quality(8.0)
        assert "Good" in forecaster.evaluate_forecast_quality(15.0)
        assert "Acceptable" in forecaster.evaluate_forecast_quality(25.0)
        assert "Poor" in forecaster.evaluate_forecast_quality(40.0)
        assert "Very poor" in forecaster.evaluate_forecast_quality(60.0)

    def test_uncertainty_grows_with_horizon(self):
        """Test that uncertainty increases with forecast horizon."""
        forecaster = CapitalFlowForecaster()

        np.random.seed(42)
        historical_flows = np.cumsum(np.random.normal(0, 10, 30)) + 100
        timestamps = [datetime.now(timezone.utc) - timedelta(days=30-i) for i in range(30)]

        forecast = forecaster.forecast_capital_flows(
            historical_flows, timestamps, horizon_days=7
        )

        # Check uncertainty grows
        uncertainties = [
            p.confidence_upper - p.confidence_lower
            for p in forecast.net_flow_forecast
        ]

        # Later forecasts should have higher uncertainty
        assert uncertainties[-1] > uncertainties[0]


class TestMockForecaster:
    """Test mock forecaster for development."""

    def test_mock_forecaster_initialization(self):
        """Test mock forecaster initialization."""
        forecaster = MockForecaster(trend=0.5, volatility=1.5)
        assert forecaster.trend == 0.5
        assert forecaster.volatility == 1.5

    def test_mock_forecast_generation(self):
        """Test mock forecaster generates valid forecasts."""
        forecaster = MockForecaster()

        np.random.seed(42)
        historical_flows = np.random.normal(100, 20, 30)
        timestamps = [datetime.now(timezone.utc) - timedelta(days=30-i) for i in range(30)]

        forecast = forecaster.forecast_capital_flows(
            historical_flows, timestamps, horizon_days=7
        )

        assert len(forecast.net_flow_forecast) == 7
        assert forecast.historical_mape is not None
        assert forecast.historical_mape > 0

    def test_mock_backtest(self):
        """Test mock backtest generates valid results."""
        forecaster = MockForecaster()

        np.random.seed(42)
        historical_flows = np.random.normal(100, 20, 30)
        timestamps = [datetime.now(timezone.utc) - timedelta(days=30-i) for i in range(30)]

        result = forecaster.backtest(historical_flows, timestamps, test_periods=7)

        assert result.mape > 0
        assert len(result.period_predictions) == 7


class TestDashboardData:
    """Test dashboard data structures."""

    def test_sample_intelligence_creation(self):
        """Test sample intelligence can be created."""
        intelligence = create_sample_intelligence()

        assert isinstance(intelligence, TokenIntelligence)
        assert intelligence.token_mint is not None
        # Note: use_enum_values=True in Pydantic config serializes enum to string
        assert intelligence.risk_level in [e.value for e in RiskLevel]
        assert 0 <= intelligence.risk_score <= 1
        assert len(intelligence.risk_factors) > 0
        assert len(intelligence.actionable_insights) > 0

    def test_risk_factor_structure(self):
        """Test risk factor has required fields."""
        factor = RiskFactor(
            name="test_factor",
            description="Test description",
            value=0.75,
            contribution_pct=25.0,
            severity="high",
        )

        assert factor.name == "test_factor"
        assert factor.value == 0.75
        assert factor.contribution_pct == 25.0
        assert factor.severity == "high"

    def test_actionable_insight_structure(self):
        """Test actionable insight structure."""
        insight = ActionableInsight(
            title="Test insight",
            description="Description",
            priority="high",
            action_type="monitor",
        )

        assert insight.title == "Test insight"
        assert insight.priority == "high"
        assert insight.action_type == "monitor"

    def test_token_intelligence_validation(self):
        """Test token intelligence validates correctly."""
        intelligence = create_sample_intelligence()

        # Should serialize to dict
        data = intelligence.model_dump()
        assert isinstance(data, dict)
        assert "token_mint" in data
        assert "risk_score" in data

    def test_confidence_interval_structure(self):
        """Test confidence interval structure."""
        from src.explainability.dashboard_data import ConfidenceInterval

        ci = ConfidenceInterval(
            lower=0.3,
            upper=0.7,
            point_estimate=0.5,
        )

        assert ci.lower < ci.point_estimate < ci.upper


class TestIntegration:
    """Integration tests across multiple components."""

    def test_end_to_end_explanation_pipeline(self):
        """Test complete explanation pipeline."""
        # 1. Generate SHAP explanation
        explainer = MockSHAPExplainer(["hhi", "gini", "churn_rate"])
        features = np.array([0.8, 0.7, 0.4])
        explanation = explainer.explain(features, uncertainty=True)

        # 2. Generate narrative
        generator = NarrativeGenerator(verbose=True)
        narrative = generator.generate_risk_narrative(explanation, token_symbol="TEST")

        # 3. Verify complete pipeline
        assert narrative.summary is not None
        assert len(narrative.key_drivers) > 0
        assert len(narrative.actionable_insights) > 0
        assert narrative.technical_details is not None

    def test_forecast_to_dashboard_data(self):
        """Test forecast integration with dashboard data."""
        from src.explainability.dashboard_data import ForecastData, ConfidenceInterval, TimeSeriesPoint

        # Generate forecast
        forecaster = MockForecaster()
        np.random.seed(42)
        historical_flows = np.random.normal(100, 20, 30)
        timestamps = [datetime.now(timezone.utc) - timedelta(days=30-i) for i in range(30)]

        forecast = forecaster.forecast_capital_flows(
            historical_flows, timestamps, horizon_days=7
        )

        # Create dashboard forecast data
        forecast_data = ForecastData(
            token_mint="test_token",
            forecast_timestamp=datetime.now(timezone.utc),
            forecast_horizon_days=7,
            predicted_inflow=ConfidenceInterval(lower=50, upper=150, point_estimate=100),
            predicted_outflow=ConfidenceInterval(lower=30, upper=100, point_estimate=65),
            net_flow=ConfidenceInterval(lower=20, upper=80, point_estimate=35),
            forecast_series=[
                TimeSeriesPoint(timestamp=p.timestamp, value=p.predicted_value)
                for p in forecast.net_flow_forecast
            ],
            forecast_confidence_upper=[
                TimeSeriesPoint(timestamp=p.timestamp, value=p.confidence_upper)
                for p in forecast.net_flow_forecast
            ],
            forecast_confidence_lower=[
                TimeSeriesPoint(timestamp=p.timestamp, value=p.confidence_lower)
                for p in forecast.net_flow_forecast
            ],
            assumptions=forecast.assumptions,
            limitations=forecast.limitations,
        )

        # Verify
        assert len(forecast_data.forecast_series) == 7
        assert forecast_data.historical_accuracy is None or forecast_data.historical_accuracy > 0
