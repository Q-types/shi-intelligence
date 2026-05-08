"""Request and response schemas for the SHI API.

This module defines Pydantic models for API request validation
and response serialization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from src.bayesian import Evidence, EvidenceType, RiskEstimate


class TokenAnalysisRequest(BaseModel):
    """Request for token analysis.

    Attributes
    ----------
    include_historical : bool
        Whether to include historical data.
    include_forecast : bool
        Whether to include forecast data.
    forecast_days : int
        Number of days to forecast if included.
    include_explanations : bool
        Whether to include SHAP explanations.
    """

    include_historical: bool = Field(
        default=True,
        description="Include historical risk scores and regime data",
    )
    include_forecast: bool = Field(
        default=False,
        description="Include capital flow forecast",
    )
    forecast_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Forecast horizon in days (1-30)",
    )
    include_explanations: bool = Field(
        default=False,
        description="Include SHAP-based risk explanations",
    )


class ForecastRequest(BaseModel):
    """Request for capital flow forecast.

    Attributes
    ----------
    horizon_days : int
        Number of days to forecast.
    confidence_level : float
        Confidence level for intervals.
    include_backtest : bool
        Whether to include backtest results.
    """

    horizon_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Forecast horizon in days",
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for prediction intervals",
    )
    include_backtest: bool = Field(
        default=False,
        description="Include historical backtest metrics",
    )


class WalletProfileRequest(BaseModel):
    """Request for wallet profile.

    Attributes
    ----------
    include_history : bool
        Whether to include profile history.
    include_sequence : bool
        Whether to include action sequence analysis.
    history_days : int
        Number of days of history to include.
    """

    include_history: bool = Field(
        default=True,
        description="Include historical profile data",
    )
    include_sequence: bool = Field(
        default=False,
        description="Include action sequence pattern analysis",
    )
    history_days: int = Field(
        default=30,
        ge=1,
        le=90,
        description="Days of history to include",
    )


class EvidenceInput(BaseModel):
    """Evidence input for risk update.

    Attributes
    ----------
    evidence_type : str
        Type of evidence (from EvidenceType enum).
    value : float
        Observed value.
    strength : float
        Evidence strength (0 to 1).
    direction : float
        Evidence direction (-1 to 1, positive = risky).
    """

    evidence_type: str = Field(
        ...,
        description="Type of evidence (e.g., 'concentration_change', 'anomaly_detection')",
    )
    value: float = Field(
        ...,
        description="Observed value",
    )
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Evidence strength (0-1)",
    )
    direction: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Evidence direction (-1=safe, +1=risky)",
    )

    def to_evidence(self) -> Evidence:
        """Convert to Evidence object.

        Returns
        -------
        Evidence
            Converted evidence.

        Raises
        ------
        ValueError
            If evidence_type is not valid.
        """
        try:
            ev_type = EvidenceType(self.evidence_type)
        except ValueError:
            raise ValueError(f"Unknown evidence type: {self.evidence_type}")

        return Evidence(
            evidence_type=ev_type,
            value=self.value,
            timestamp=datetime.now(timezone.utc),
            strength=self.strength,
            direction=self.direction,
        )


class RiskUpdateRequest(BaseModel):
    """Request to update Bayesian risk beliefs.

    Attributes
    ----------
    evidences : list[EvidenceInput]
        List of evidence to incorporate.
    reset_beliefs : bool
        Whether to reset beliefs before updating.
    prior_alpha : float
        Alpha parameter if resetting.
    prior_beta : float
        Beta parameter if resetting.
    """

    evidences: list[EvidenceInput] = Field(
        ...,
        min_length=1,
        description="Evidence items to incorporate",
    )
    reset_beliefs: bool = Field(
        default=False,
        description="Reset beliefs to prior before updating",
    )
    prior_alpha: float = Field(
        default=1.0,
        gt=0,
        description="Alpha parameter for prior (if reset)",
    )
    prior_beta: float = Field(
        default=1.0,
        gt=0,
        description="Beta parameter for prior (if reset)",
    )


class RiskBeliefResponse(BaseModel):
    """Response for risk belief state.

    Attributes
    ----------
    token_mint : str
        Token mint address.
    rug_probability : RiskEstimateResponse
        Posterior rug probability estimate.
    composite_risk : RiskEstimateResponse
        Composite risk score.
    uncertainty_level : str
        Uncertainty category.
    updates_applied : int
        Number of evidence updates applied.
    timestamp : datetime
        Response timestamp.
    """

    token_mint: str
    rug_probability: dict[str, float]
    composite_risk: dict[str, float]
    uncertainty_level: str
    updates_applied: int
    timestamp: datetime


class SequenceAnalysisResponse(BaseModel):
    """Response for wallet sequence analysis.

    Attributes
    ----------
    wallet : str
        Wallet address.
    action_count : int
        Number of actions in sequence.
    dominant_actions : list[str]
        Most common actions.
    dump_likelihood : float
        Dump signature likelihood (0-1).
    signatures_found : list[dict]
        Matched dump signatures.
    cluster_label : str | None
        Behavioral cluster assignment.
    timestamp : datetime
        Analysis timestamp.
    """

    wallet: str
    action_count: int
    dominant_actions: list[str]
    dump_likelihood: float
    signatures_found: list[dict[str, Any]]
    cluster_label: str | None
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response.

    Attributes
    ----------
    status : str
        Health status: healthy, degraded, unhealthy.
    version : str
        API version.
    timestamp : datetime
        Check timestamp.
    components : dict[str, str]
        Component status map.
    """

    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Health check timestamp")
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Component health statuses",
    )


class ErrorResponse(BaseModel):
    """Error response.

    Attributes
    ----------
    error : str
        Error type.
    message : str
        Error message.
    details : dict | None
        Additional error details.
    timestamp : datetime
        Error timestamp.
    """

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        None,
        description="Additional error details",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp",
    )


class PaginationParams(BaseModel):
    """Pagination parameters.

    Attributes
    ----------
    offset : int
        Number of items to skip.
    limit : int
        Maximum items to return.
    """

    offset: int = Field(default=0, ge=0, description="Items to skip")
    limit: int = Field(default=20, ge=1, le=100, description="Max items to return")
