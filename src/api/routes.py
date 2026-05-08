"""FastAPI routes for the SHI API.

This module defines all REST API endpoints for token intelligence,
forecasting, wallet profiles, and risk belief updates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from src.explainability.dashboard_data import (
    TokenIntelligence,
    ForecastData,
    WalletProfile,
    DashboardResponse,
    create_sample_intelligence,
)
from .schemas import (
    TokenAnalysisRequest,
    ForecastRequest,
    WalletProfileRequest,
    RiskUpdateRequest,
    RiskBeliefResponse,
    SequenceAnalysisResponse,
    HealthResponse,
    ErrorResponse,
)
from .dependencies import (
    get_risk_model,
    get_settings,
    clear_risk_models,
)

logger = structlog.get_logger()

# API version
API_VERSION = "1.0.0"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Configured application instance.
    """
    application = FastAPI(
        title="SHI Intelligence API",
        description="Solana Holder Intelligence - Token risk analysis and forecasting",
        version=API_VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # Add CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add exception handlers
    @application.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="validation_error",
                message=str(exc),
            ).model_dump(mode="json"),
        )

    @application.exception_handler(Exception)
    async def general_error_handler(request, exc):
        logger.error("unhandled_exception", error=str(exc))
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                message="An unexpected error occurred",
            ).model_dump(mode="json"),
        )

    return application


# Create default app instance
app = create_app()


# Health endpoints
@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Check API health status.

    Returns overall health and component statuses.
    """
    components = {
        "api": "healthy",
        "database": "healthy",  # Would check actual DB
        "cache": "healthy",  # Would check actual cache
    }

    # Determine overall status
    if all(s == "healthy" for s in components.values()):
        status = "healthy"
    elif any(s == "unhealthy" for s in components.values()):
        status = "unhealthy"
    else:
        status = "degraded"

    return HealthResponse(
        status=status,
        version=API_VERSION,
        timestamp=datetime.now(timezone.utc),
        components=components,
    )


# Token intelligence endpoints
@app.get(
    "/api/v1/token/{mint}/intelligence",
    response_model=DashboardResponse,
    tags=["Token Intelligence"],
    summary="Get token intelligence",
)
async def get_token_intelligence(
    mint: str = Path(..., description="Token mint address"),
    include_historical: bool = Query(True, description="Include historical data"),
    include_forecast: bool = Query(False, description="Include forecast"),
    forecast_days: int = Query(7, ge=1, le=30, description="Forecast days"),
) -> DashboardResponse:
    """Get complete token intelligence analysis.

    Parameters
    ----------
    mint : str
        Token mint address (32-44 character base58 string).
    include_historical : bool
        Whether to include historical risk scores and regimes.
    include_forecast : bool
        Whether to include capital flow forecast.
    forecast_days : int
        Forecast horizon if forecast is included.

    Returns
    -------
    DashboardResponse
        Complete intelligence response.
    """
    logger.info("token_intelligence_request", mint=mint)

    # Validate mint address
    if len(mint) < 32 or len(mint) > 44:
        raise HTTPException(
            status_code=400,
            detail="Invalid mint address format",
        )

    # For now, return sample data
    # In production, this would call the orchestrator
    intelligence = create_sample_intelligence()

    # Override mint
    intelligence = TokenIntelligence(
        **{
            **intelligence.model_dump(),
            "token_mint": mint,
            "analysis_timestamp": datetime.now(timezone.utc),
        }
    )

    forecast = None
    if include_forecast:
        # Would generate actual forecast
        pass

    return DashboardResponse(
        success=True,
        timestamp=datetime.now(timezone.utc),
        data=intelligence,
        forecast=forecast,
        warnings=[],
        errors=[],
    )


@app.get(
    "/api/v1/token/{mint}/forecast",
    response_model=ForecastData | dict,
    tags=["Token Intelligence"],
    summary="Get capital flow forecast",
)
async def get_token_forecast(
    mint: str = Path(..., description="Token mint address"),
    days: int = Query(7, ge=1, le=30, description="Forecast horizon"),
) -> dict[str, Any]:
    """Get capital flow forecast for a token.

    Parameters
    ----------
    mint : str
        Token mint address.
    days : int
        Forecast horizon in days.

    Returns
    -------
    ForecastData
        Capital flow forecast.
    """
    logger.info("forecast_request", mint=mint, days=days)

    # Placeholder response
    return {
        "token_mint": mint,
        "forecast_timestamp": datetime.now(timezone.utc).isoformat(),
        "forecast_horizon_days": days,
        "message": "Forecast not available - orchestrator not connected",
        "note": "This endpoint requires the full analysis pipeline",
    }


@app.get(
    "/api/v1/token/{mint}/explain",
    tags=["Token Intelligence"],
    summary="Get risk explanation",
)
async def get_risk_explanation(
    mint: str = Path(..., description="Token mint address"),
    verbose: bool = Query(False, description="Include detailed SHAP values"),
) -> dict[str, Any]:
    """Get explainability data for token risk assessment.

    Parameters
    ----------
    mint : str
        Token mint address.
    verbose : bool
        Whether to include detailed SHAP feature contributions.

    Returns
    -------
    dict
        Risk explanation with contributing factors.
    """
    logger.info("explain_request", mint=mint, verbose=verbose)

    # Placeholder response
    return {
        "token_mint": mint,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "risk_score": 0.55,
        "top_factors": [
            {
                "feature": "holder_concentration",
                "contribution": 0.35,
                "direction": "positive",
                "explanation": "High concentration increases risk",
            },
            {
                "feature": "churn_rate",
                "contribution": 0.22,
                "direction": "positive",
                "explanation": "Elevated churn suggests instability",
            },
        ],
        "narrative": "Risk is primarily driven by holder concentration...",
    }


# Wallet endpoints
@app.get(
    "/api/v1/wallet/{address}/profile",
    response_model=WalletProfile | dict,
    tags=["Wallet Intelligence"],
    summary="Get wallet profile",
)
async def get_wallet_profile(
    address: str = Path(..., description="Wallet address"),
    include_history: bool = Query(True, description="Include profile history"),
    include_sequence: bool = Query(False, description="Include action sequence"),
) -> dict[str, Any]:
    """Get wallet profile with behavioral analysis.

    Parameters
    ----------
    address : str
        Wallet address (32-44 character base58 string).
    include_history : bool
        Whether to include historical profile data.
    include_sequence : bool
        Whether to include action sequence analysis.

    Returns
    -------
    WalletProfile
        Wallet profile data.
    """
    logger.info("wallet_profile_request", address=address)

    # Validate address
    if len(address) < 32 or len(address) > 44:
        raise HTTPException(
            status_code=400,
            detail="Invalid wallet address format",
        )

    # Placeholder response
    return {
        "wallet_address": address,
        "current_archetype": "accumulator",
        "risk_score": 0.3,
        "anomaly_score": -0.2,
        "profile_velocity": 0.05,
        "risk_trend": "stable",
        "time_in_archetype": 14.5,
        "archetype_transitions": [],
        "risk_score_history": [],
        "interpretation": "Wallet shows accumulation behavior",
        "warnings": [],
    }


@app.get(
    "/api/v1/wallet/{address}/sequence",
    response_model=SequenceAnalysisResponse,
    tags=["Wallet Intelligence"],
    summary="Analyze wallet action sequence",
)
async def analyze_wallet_sequence(
    address: str = Path(..., description="Wallet address"),
) -> SequenceAnalysisResponse:
    """Analyze wallet action sequence for behavioral patterns.

    Parameters
    ----------
    address : str
        Wallet address.

    Returns
    -------
    SequenceAnalysisResponse
        Sequence analysis results.
    """
    logger.info("sequence_analysis_request", address=address)

    # Placeholder response
    return SequenceAnalysisResponse(
        wallet=address,
        action_count=0,
        dominant_actions=["idle"],
        dump_likelihood=0.0,
        signatures_found=[],
        cluster_label=None,
        timestamp=datetime.now(timezone.utc),
    )


# Bayesian risk endpoints
@app.post(
    "/api/v1/token/{mint}/risk/update",
    response_model=RiskBeliefResponse,
    tags=["Risk Estimation"],
    summary="Update Bayesian risk beliefs",
)
async def update_risk_beliefs(
    mint: str = Path(..., description="Token mint address"),
    request: RiskUpdateRequest = Body(...),
) -> RiskBeliefResponse:
    """Update Bayesian risk beliefs with new evidence.

    Parameters
    ----------
    mint : str
        Token mint address.
    request : RiskUpdateRequest
        Evidence to incorporate.

    Returns
    -------
    RiskBeliefResponse
        Updated risk belief state.
    """
    logger.info(
        "risk_update_request",
        mint=mint,
        evidence_count=len(request.evidences),
    )

    # Get or create model
    model = get_risk_model(mint)

    # Reset if requested
    if request.reset_beliefs:
        model.reset(
            prior_alpha=request.prior_alpha,
            prior_beta=request.prior_beta,
        )

    # Convert and apply evidence
    for ev_input in request.evidences:
        try:
            evidence = ev_input.to_evidence()
            model.update(evidence)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Get estimates
    rug_estimate = model.posterior_rug_probability()
    composite = model.composite_risk_score()

    return RiskBeliefResponse(
        token_mint=mint,
        rug_probability=rug_estimate.to_dict(),
        composite_risk=composite.to_dict(),
        uncertainty_level=model.uncertainty_level(),
        updates_applied=model.state.total_updates,
        timestamp=datetime.now(timezone.utc),
    )


@app.get(
    "/api/v1/token/{mint}/risk/belief",
    response_model=RiskBeliefResponse,
    tags=["Risk Estimation"],
    summary="Get current risk beliefs",
)
async def get_risk_beliefs(
    mint: str = Path(..., description="Token mint address"),
) -> RiskBeliefResponse:
    """Get current Bayesian risk beliefs for a token.

    Parameters
    ----------
    mint : str
        Token mint address.

    Returns
    -------
    RiskBeliefResponse
        Current risk belief state.
    """
    model = get_risk_model(mint)

    rug_estimate = model.posterior_rug_probability()
    composite = model.composite_risk_score()

    return RiskBeliefResponse(
        token_mint=mint,
        rug_probability=rug_estimate.to_dict(),
        composite_risk=composite.to_dict(),
        uncertainty_level=model.uncertainty_level(),
        updates_applied=model.state.total_updates,
        timestamp=datetime.now(timezone.utc),
    )


@app.delete(
    "/api/v1/token/{mint}/risk/reset",
    tags=["Risk Estimation"],
    summary="Reset risk beliefs",
)
async def reset_risk_beliefs(
    mint: str = Path(..., description="Token mint address"),
    alpha: float = Query(1.0, gt=0, description="Prior alpha"),
    beta: float = Query(1.0, gt=0, description="Prior beta"),
) -> dict[str, str]:
    """Reset risk beliefs to prior for a token.

    Parameters
    ----------
    mint : str
        Token mint address.
    alpha : float
        Alpha parameter for new prior.
    beta : float
        Beta parameter for new prior.

    Returns
    -------
    dict
        Confirmation message.
    """
    model = get_risk_model(mint)
    model.reset(prior_alpha=alpha, prior_beta=beta)

    return {
        "status": "success",
        "message": f"Risk beliefs reset for {mint}",
        "prior": f"Beta({alpha}, {beta})",
    }


# Admin endpoints
@app.post(
    "/api/v1/admin/clear-cache",
    tags=["Admin"],
    summary="Clear risk model cache",
)
async def clear_cache() -> dict[str, str]:
    """Clear all cached risk models.

    Returns
    -------
    dict
        Confirmation message.
    """
    clear_risk_models()
    return {"status": "success", "message": "Risk model cache cleared"}
