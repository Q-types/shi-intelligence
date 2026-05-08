"""
Dashboard Data Structures for SHI Intelligence.

Defines JSON response structures for dashboard visualization including:
- Token intelligence summaries
- Regime information with confidence
- Risk factors and actionable insights
- Historical trends
- Forecast data

All structures are Pydantic models for validation and serialization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, ConfigDict, Field

from ..temporal.regimes import HolderRegimeType
from .narratives import RiskLevel, TrendDirection


class TimeSeriesPoint(BaseModel):
    """Single point in time series data."""

    timestamp: datetime
    value: float
    label: Optional[str] = None


class ConfidenceInterval(BaseModel):
    """Confidence interval for predictions."""

    lower: float = Field(..., description="Lower bound (95% CI)")
    upper: float = Field(..., description="Upper bound (95% CI)")
    point_estimate: float = Field(..., description="Point prediction")


class RiskFactor(BaseModel):
    """Individual risk factor with details."""

    name: str = Field(..., description="Risk factor name")
    description: str = Field(..., description="Human-readable description")
    value: float = Field(..., description="Current value")
    contribution_pct: float = Field(..., description="Percentage contribution to risk")
    severity: str = Field(..., description="low, moderate, high, critical")
    trend: Optional[TrendDirection] = Field(None, description="Recent trend")


class ActionableInsight(BaseModel):
    """Actionable recommendation for users."""

    title: str = Field(..., description="Short insight title")
    description: str = Field(..., description="Detailed explanation")
    priority: str = Field(..., description="low, medium, high")
    action_type: str = Field(
        ...,
        description="monitor, reduce_position, investigate, opportunity"
    )


class RegimeInfo(BaseModel):
    """Current regime information."""

    current_regime: HolderRegimeType = Field(..., description="Current holder regime")
    regime_confidence: float = Field(..., ge=0, le=1, description="Confidence in regime")
    previous_regime: Optional[HolderRegimeType] = Field(
        None, description="Previous regime if transition occurred"
    )
    time_in_regime: Optional[float] = Field(
        None, description="Days in current regime"
    )
    transition_probability: float = Field(
        ..., ge=0, le=1, description="Probability of regime change"
    )
    implications: List[str] = Field(..., description="What this regime means")


class TrendInfo(BaseModel):
    """Trend information for a metric."""

    direction: TrendDirection = Field(..., description="Trend direction")
    velocity: float = Field(..., description="Rate of change")
    acceleration: Optional[float] = Field(None, description="Change in velocity")
    description: str = Field(..., description="Human-readable trend description")


class TokenIntelligence(BaseModel):
    """
    Complete token intelligence summary for dashboard.

    This is the main response structure containing all analysis results.
    """

    # Basic metadata
    token_mint: str = Field(..., description="Token mint address")
    token_symbol: Optional[str] = Field(None, description="Token symbol")
    analysis_timestamp: datetime = Field(..., description="When analysis was performed")

    # Risk assessment
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score (0-1)")
    risk_confidence: str = Field(..., description="Confidence in risk assessment")
    risk_factors: List[RiskFactor] = Field(..., description="Top risk contributors")

    # Regime information
    current_regime: RegimeInfo = Field(..., description="Holder regime state")

    # Trends
    concentration_trend: TrendInfo = Field(..., description="Holder concentration trend")
    churn_trend: TrendInfo = Field(..., description="Holder churn trend")

    # Insights and recommendations
    actionable_insights: List[ActionableInsight] = Field(
        ..., description="Actionable recommendations"
    )

    # Summary narrative
    summary: str = Field(..., description="One-sentence overview")
    detailed_explanation: Optional[str] = Field(
        None, description="Detailed narrative explanation"
    )

    # Uncertainty
    uncertainty_note: Optional[str] = Field(
        None, description="Caveat about prediction uncertainty"
    )

    # Historical context
    historical_risk_scores: Optional[List[TimeSeriesPoint]] = Field(
        None, description="Historical risk score evolution"
    )
    historical_regimes: Optional[List[Dict[str, Any]]] = Field(
        None, description="Historical regime transitions"
    )

    # Anomaly detection
    anomaly_count: int = Field(0, description="Number of anomalous wallets detected")
    top_anomalies: Optional[List[Dict[str, Any]]] = Field(
        None, description="Top anomalous wallets"
    )

    model_config = ConfigDict(use_enum_values=True)


class ForecastData(BaseModel):
    """Capital flow forecast data."""

    token_mint: str = Field(..., description="Token mint address")
    forecast_timestamp: datetime = Field(..., description="When forecast was generated")
    forecast_horizon_days: int = Field(..., description="Forecast period in days")

    # Forecasts
    predicted_inflow: ConfidenceInterval = Field(
        ..., description="Predicted capital inflow"
    )
    predicted_outflow: ConfidenceInterval = Field(
        ..., description="Predicted capital outflow"
    )
    net_flow: ConfidenceInterval = Field(
        ..., description="Net capital flow (inflow - outflow)"
    )

    # Historical comparison
    historical_accuracy: Optional[float] = Field(
        None, description="MAPE from backtesting"
    )

    # Time series forecast
    forecast_series: List[TimeSeriesPoint] = Field(
        ..., description="Daily forecast points"
    )
    forecast_confidence_upper: List[TimeSeriesPoint] = Field(
        ..., description="Upper confidence bound time series"
    )
    forecast_confidence_lower: List[TimeSeriesPoint] = Field(
        ..., description="Lower confidence bound time series"
    )

    # Context
    assumptions: List[str] = Field(..., description="Forecast assumptions")
    limitations: List[str] = Field(..., description="Known limitations")


class WalletProfile(BaseModel):
    """Wallet profile evolution data."""

    wallet_address: str = Field(..., description="Wallet address")
    current_archetype: str = Field(..., description="Current wallet type")
    risk_score: float = Field(..., ge=0, le=1, description="Current risk score")
    anomaly_score: float = Field(..., description="Anomaly score")

    # Evolution metrics
    profile_velocity: float = Field(..., description="Rate of profile change")
    risk_trend: TrendDirection = Field(..., description="Risk score trend")
    time_in_archetype: float = Field(..., description="Days in current archetype")

    # Historical data
    archetype_transitions: List[Dict[str, Any]] = Field(
        ..., description="Historical archetype changes"
    )
    risk_score_history: List[TimeSeriesPoint] = Field(
        ..., description="Historical risk scores"
    )

    # Interpretation
    interpretation: str = Field(..., description="What this profile means")
    warnings: List[str] = Field(..., description="Warnings or concerns")


class DashboardResponse(BaseModel):
    """
    Top-level dashboard response wrapping all data.

    Use this for complete dashboard API responses.
    """

    success: bool = Field(True, description="Request success status")
    timestamp: datetime = Field(..., description="Response timestamp")
    data: TokenIntelligence = Field(..., description="Token intelligence data")
    forecast: Optional[ForecastData] = Field(None, description="Forecast data if requested")
    warnings: List[str] = Field(default_factory=list, description="System warnings")
    errors: List[str] = Field(default_factory=list, description="Non-fatal errors")

    model_config = ConfigDict(use_enum_values=True)


def create_sample_intelligence() -> TokenIntelligence:
    """
    Create sample token intelligence for testing.

    Returns
    -------
    TokenIntelligence
        Sample intelligence data
    """
    return TokenIntelligence(
        token_mint="4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        token_symbol="SAMPLE",
        analysis_timestamp=datetime.now(timezone.utc),
        risk_level=RiskLevel.MODERATE,
        risk_score=0.55,
        risk_confidence="High confidence",
        uncertainty_note=None,
        historical_risk_scores=None,
        historical_regimes=None,
        top_anomalies=None,
        risk_factors=[
            RiskFactor(
                name="holder_concentration",
                description="Top 10 holders control 65% of supply",
                value=0.65,
                contribution_pct=35.2,
                severity="high",
                trend=TrendDirection.INCREASING,
            ),
            RiskFactor(
                name="churn_rate",
                description="15% of holders churned in last 7 days",
                value=0.15,
                contribution_pct=22.1,
                severity="moderate",
                trend=TrendDirection.STABLE,
            ),
        ],
        current_regime=RegimeInfo(
            current_regime=HolderRegimeType.DISTRIBUTION,
            regime_confidence=0.82,
            previous_regime=HolderRegimeType.STABLE,
            time_in_regime=3.5,
            transition_probability=0.25,
            implications=[
                "Concentration increasing - higher whale risk",
                "Could signal accumulation by large players",
            ],
        ),
        concentration_trend=TrendInfo(
            direction=TrendDirection.INCREASING,
            velocity=0.05,
            acceleration=0.002,
            description="Concentration rising steadily",
        ),
        churn_trend=TrendInfo(
            direction=TrendDirection.STABLE,
            velocity=0.01,
            acceleration=0.0,
            description="Churn rate stable",
        ),
        actionable_insights=[
            ActionableInsight(
                title="Monitor whale wallets",
                description="Top holders increasing positions - watch for exits",
                priority="high",
                action_type="monitor",
            ),
            ActionableInsight(
                title="Consider position sizing",
                description="Moderate risk environment - use standard position limits",
                priority="medium",
                action_type="reduce_position",
            ),
        ],
        summary="SAMPLE faces moderate sell pressure risk (score: 0.55)",
        detailed_explanation="Holder concentration is the primary risk driver...",
        anomaly_count=3,
    )
