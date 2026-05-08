"""
Explainability Module for SHI.

Provides interpretable explanations for risk scores, regime changes, and forecasts.

Components:
- SHAP-based feature explanations (shap_explainer)
- Natural language narratives (narratives)
- Dashboard data structures (dashboard_data)
- Capital flow forecasting (forecasting)

Usage:
    from src.explainability import SHAPExplainer, NarrativeGenerator
    from src.explainability import TokenIntelligence, ForecastData
    from src.explainability import CapitalFlowForecaster

Example:
    # Explain a risk score
    explainer = SHAPExplainer(model, feature_names)
    explanation = explainer.explain(features)

    # Generate narrative
    generator = NarrativeGenerator()
    narrative = generator.generate_risk_narrative(explanation)

    # Forecast capital flows
    forecaster = CapitalFlowForecaster()
    forecast = forecaster.forecast_capital_flows(historical_data, timestamps)
"""

from .shap_explainer import (
    SHAPExplainer,
    SHAPExplanation,
    FeatureContribution,
    ExplanationType,
    MockSHAPExplainer,
    create_mock_explainer,
)

from .narratives import (
    NarrativeGenerator,
    RiskNarrative,
    RegimeNarrative,
    AnomalyNarrative,
    RiskLevel,
    TrendDirection,
)

from .dashboard_data import (
    TokenIntelligence,
    ForecastData,
    WalletProfile,
    DashboardResponse,
    RiskFactor,
    ActionableInsight,
    RegimeInfo,
    TrendInfo,
    TimeSeriesPoint,
    ConfidenceInterval,
    create_sample_intelligence,
)

from .forecasting import (
    CapitalFlowForecaster,
    CapitalFlowForecast,
    ForecastPoint,
    BacktestResult,
    MockForecaster,
)

__all__ = [
    # SHAP Explainer
    "SHAPExplainer",
    "SHAPExplanation",
    "FeatureContribution",
    "ExplanationType",
    "MockSHAPExplainer",
    "create_mock_explainer",
    # Narratives
    "NarrativeGenerator",
    "RiskNarrative",
    "RegimeNarrative",
    "AnomalyNarrative",
    "RiskLevel",
    "TrendDirection",
    # Dashboard Data
    "TokenIntelligence",
    "ForecastData",
    "WalletProfile",
    "DashboardResponse",
    "RiskFactor",
    "ActionableInsight",
    "RegimeInfo",
    "TrendInfo",
    "TimeSeriesPoint",
    "ConfidenceInterval",
    "create_sample_intelligence",
    # Forecasting
    "CapitalFlowForecaster",
    "CapitalFlowForecast",
    "ForecastPoint",
    "BacktestResult",
    "MockForecaster",
]
