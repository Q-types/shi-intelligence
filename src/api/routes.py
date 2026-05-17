"""FastAPI routes for the SHI API.

This module defines all REST API endpoints for token intelligence,
forecasting, wallet profiles, and risk belief updates.
Includes WebSocket endpoints for real-time alert streaming.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pathlib import Path as FilePath

from fastapi import Body, FastAPI, HTTPException, Query, Path, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
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
    PriceDataResponse,
    LiquidityDataResponse,
    PoolInfoResponse,
)
from src.data.price_provider import JupiterPriceProvider
from src.data.client import SolanaDataClient
from src.liquidity.pools import LiquidityFetcher
from src.metrics import (
    compute_hhi,
    compute_shannon_entropy,
    compute_gini_coefficient,
    compute_whale_dominance_ratio,
)
from src.clustering.archetypes import (
    WalletFeatureVector,
    cluster_wallets,
    assign_archetype,
    get_archetype_distribution,
)
from src.graph import FundingGraph, find_shared_funders
from .dependencies import (
    get_risk_model,
    get_settings,
    clear_risk_models,
)
from .websocket import ws_manager, SubscriptionType

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

    # Mount static files
    static_dir = FilePath(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        application.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

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


# Price endpoints
@app.get(
    "/api/v1/token/{mint}/price",
    response_model=PriceDataResponse,
    tags=["Price & Liquidity"],
    summary="Get token price",
)
async def get_token_price(
    mint: str = Path(..., description="Token mint address"),
    skip_cache: bool = Query(False, description="Bypass cache and fetch fresh data"),
) -> PriceDataResponse:
    """Get current price data for a token from Jupiter API.

    Parameters
    ----------
    mint : str
        Token mint address (32-44 character base58 string).
    skip_cache : bool
        If True, bypass the price cache and fetch fresh data.

    Returns
    -------
    PriceDataResponse
        Token price data including USD price, confidence, and source.

    Raises
    ------
    HTTPException
        404 if price data is not available for the token.
    """
    logger.info("price_request", mint=mint, skip_cache=skip_cache)

    # Validate mint address
    if len(mint) < 32 or len(mint) > 44:
        raise HTTPException(
            status_code=400,
            detail="Invalid mint address format",
        )

    # Fetch price
    provider = JupiterPriceProvider()
    try:
        price = await provider.get_price(mint, skip_cache=skip_cache)
        if price is None:
            raise HTTPException(
                status_code=404,
                detail=f"Price data not available for token {mint}",
            )

        return PriceDataResponse(
            mint=price.mint,
            price_usd=price.price_usd,
            price_change_24h_pct=price.price_change_24h_pct,
            confidence=price.confidence,
            source=price.source,
            fetched_at=price.fetched_at,
        )
    finally:
        await provider.close()


@app.get(
    "/api/v1/token/{mint}/liquidity",
    response_model=LiquidityDataResponse,
    tags=["Price & Liquidity"],
    summary="Get token liquidity",
)
async def get_token_liquidity(
    mint: str = Path(..., description="Token mint address"),
    include_pools: bool = Query(True, description="Include detailed pool data"),
    limit: int = Query(10, ge=1, le=50, description="Max pools to return"),
) -> LiquidityDataResponse:
    """Get liquidity pool data for a token from supported DEXes.

    Aggregates liquidity data from Raydium and Orca pools.

    Parameters
    ----------
    mint : str
        Token mint address (32-44 character base58 string).
    include_pools : bool
        Whether to include detailed pool information.
    limit : int
        Maximum number of pools to return (1-50).

    Returns
    -------
    LiquidityDataResponse
        Token liquidity data including total USD liquidity and pool details.
    """
    logger.info("liquidity_request", mint=mint, include_pools=include_pools)

    # Validate mint address
    if len(mint) < 32 or len(mint) > 44:
        raise HTTPException(
            status_code=400,
            detail="Invalid mint address format",
        )

    # Fetch liquidity
    fetcher = LiquidityFetcher()
    try:
        pools = await fetcher.get_all_pools(mint)

        # Calculate total liquidity
        total_liquidity = sum(p.liquidity_usd or 0 for p in pools)

        # Convert pools to response format
        pool_responses = []
        if include_pools:
            for pool in pools[:limit]:
                pool_responses.append(PoolInfoResponse(
                    pool_address=pool.pool_address,
                    dex=pool.dex,
                    token_a_mint=pool.token_a_mint,
                    token_b_mint=pool.token_b_mint,
                    liquidity_usd=pool.liquidity_usd,
                    volume_24h_usd=pool.volume_24h_usd,
                    fee_rate=pool.fee_rate,
                ))

        # Get deepest pool
        deepest = None
        if pools:
            deepest_pool = pools[0]  # Already sorted by liquidity
            deepest = PoolInfoResponse(
                pool_address=deepest_pool.pool_address,
                dex=deepest_pool.dex,
                token_a_mint=deepest_pool.token_a_mint,
                token_b_mint=deepest_pool.token_b_mint,
                liquidity_usd=deepest_pool.liquidity_usd,
                volume_24h_usd=deepest_pool.volume_24h_usd,
                fee_rate=deepest_pool.fee_rate,
            )

        return LiquidityDataResponse(
            mint=mint,
            total_liquidity_usd=total_liquidity,
            pool_count=len(pools),
            pools=pool_responses,
            deepest_pool=deepest,
            fetched_at=datetime.now(timezone.utc),
        )
    finally:
        await fetcher.close()


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


# WebSocket endpoints
@app.websocket("/api/v1/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alert streaming.

    Connect and subscribe to alerts:
    - Subscribe to specific token: {"action": "subscribe", "type": "token", "filter": "TOKEN_MINT"}
    - Subscribe to user alerts: {"action": "subscribe", "type": "user", "filter": "USER_ID"}
    - Unsubscribe: {"action": "unsubscribe", "type": "token", "filter": "TOKEN_MINT"}
    - Ping: {"action": "ping"}

    Received messages:
    - {"type": "alert", "data": {...}} - Alert notification
    - {"type": "heartbeat", "data": {"status": "alive"}} - Heartbeat (every 30s)
    - {"type": "subscribe_ack", "data": {...}} - Subscription confirmation
    - {"type": "error", "data": {"error": "..."}} - Error message
    """
    connection_id = await ws_manager.connect(websocket)

    try:
        while True:
            # Receive and handle messages
            message = await websocket.receive_text()
            await ws_manager.handle_message(connection_id, message)

    except WebSocketDisconnect:
        logger.info("websocket_client_disconnected", connection_id=connection_id)
    except Exception as e:
        logger.error("websocket_error", connection_id=connection_id, error=str(e))
    finally:
        await ws_manager.disconnect(connection_id)


@app.websocket("/api/v1/ws/alerts/token/{mint}")
async def websocket_token_alerts(
    websocket: WebSocket,
    mint: str,
):
    """WebSocket endpoint for a specific token's alerts.

    Automatically subscribes to the token specified in the path.

    Parameters
    ----------
    mint : str
        Token mint address to subscribe to.
    """
    connection_id = await ws_manager.connect(websocket)

    try:
        # Auto-subscribe to the token
        await ws_manager.subscribe(connection_id, SubscriptionType.TOKEN, mint)

        while True:
            message = await websocket.receive_text()
            await ws_manager.handle_message(connection_id, message)

    except WebSocketDisconnect:
        logger.info("websocket_client_disconnected", connection_id=connection_id)
    except Exception as e:
        logger.error("websocket_error", connection_id=connection_id, error=str(e))
    finally:
        await ws_manager.disconnect(connection_id)


@app.get(
    "/api/v1/ws/stats",
    tags=["WebSocket"],
    summary="Get WebSocket connection statistics",
)
async def get_websocket_stats() -> dict[str, Any]:
    """Get WebSocket connection and subscription statistics.

    Returns
    -------
    dict
        Statistics about active connections and subscriptions.
    """
    return {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "statistics": ws_manager.get_statistics(),
    }


# Dashboard endpoints
@app.get(
    "/api/v1/dashboard/analyze/{mint}",
    tags=["Dashboard"],
    summary="Full token analysis for dashboard",
)
async def dashboard_analyze(
    mint: str = Path(..., description="Token mint address"),
) -> dict[str, Any]:
    """Get complete token analysis for the web dashboard.

    This endpoint returns all data needed for the dashboard UI:
    - Token info (mint, supply, holder count)
    - Price data (from Jupiter API)
    - Distribution metrics (HHI, Gini, Entropy, Whale Dominance)
    - Risk score
    - Top holders with shares
    - Funding edges for graph visualization

    Parameters
    ----------
    mint : str
        Token mint address (32-44 character base58 string).

    Returns
    -------
    dict
        Complete analysis data for dashboard rendering.
    """
    logger.info("dashboard_analyze_request", mint=mint)

    # Validate mint address
    if len(mint) < 32 or len(mint) > 44:
        raise HTTPException(
            status_code=400,
            detail="Invalid mint address format (should be 32-44 characters)",
        )

    client = SolanaDataClient()
    price_provider = JupiterPriceProvider()

    try:
        # Fetch price data
        price_data = None
        try:
            price = await price_provider.get_price(mint)
            if price:
                price_data = {
                    "price_usd": price.price_usd,
                    "price_change_24h_pct": price.price_change_24h_pct,
                    "confidence": price.confidence,
                }
        except Exception as e:
            logger.warning("price_fetch_failed", mint=mint, error=str(e))

        # Fetch holder data
        snapshot = await client.get_token_holders(mint, limit=5000)

        # Compute metrics
        shares = snapshot.shares
        balances = [b.balance for b in snapshot.balances]

        hhi = compute_hhi(shares)
        entropy = compute_shannon_entropy(shares)
        gini = compute_gini_coefficient(balances)
        wdr = compute_whale_dominance_ratio(balances, snapshot.total_supply)

        # Compute risk score
        risk_score = (
            (1.0 if hhi.value > 0.25 else 0.5 if hhi.value > 0.1 else 0.0) +
            (1.0 if entropy.value < 2 else 0.5 if entropy.value < 4 else 0.0) +
            (1.0 if gini.value > 0.8 else 0.5 if gini.value > 0.5 else 0.0) +
            (1.0 if wdr.value > 0.5 else 0.5 if wdr.value > 0.3 else 0.0)
        ) / 4.0

        # Format holders for response
        sorted_balances = sorted(snapshot.balances, key=lambda x: x.balance, reverse=True)
        holders = []
        for bal in sorted_balances[:100]:  # Top 100 holders
            share = bal.balance / snapshot.total_supply if snapshot.total_supply > 0 else 0
            holders.append({
                "wallet": bal.wallet,
                "balance": bal.balance,
                "share": share,
            })

        # Fetch funding edges
        funding_edges = []
        funding_graph = FundingGraph()
        try:
            top_wallets = [h["wallet"] for h in holders[:50]]
            edges = await client.get_funding_edges(top_wallets)

            # Build funding graph
            for wallet in top_wallets:
                funding_graph.add_wallet(wallet)
            funding_graph.add_edges_from_list(edges)

            funding_edges = [
                {
                    "from": edge.source,
                    "to": edge.target,
                    "amount": edge.amount_lamports,
                }
                for edge in edges
            ]
        except Exception as e:
            logger.warning("funding_edges_failed", error=str(e))

        # Compute wallet archetypes
        archetype_assignments = {}
        archetype_distribution = {}
        shared_funders_map = {}

        try:
            # Find shared funders for coordination detection
            top_wallets = [h["wallet"] for h in holders[:50]]
            shared_funders = find_shared_funders(funding_graph, top_wallets)

            # Build shared funder count per wallet
            shared_funder_counts: dict[str, int] = {}
            for funder, funded_wallets in shared_funders.items():
                for wallet in funded_wallets:
                    shared_funder_counts[wallet] = shared_funder_counts.get(wallet, 0) + 1

            # Track which wallets share a funder for UI
            for funder, funded_wallets in shared_funders.items():
                if len(funded_wallets) >= 2:
                    for wallet in funded_wallets:
                        if wallet not in shared_funders_map:
                            shared_funders_map[wallet] = []
                        shared_funders_map[wallet].append({
                            "funder": funder,
                            "co_funded": [w for w in funded_wallets if w != wallet][:5],
                        })

            # Build feature vectors for classification
            features_list = []
            for i, holder in enumerate(holders[:50]):
                wallet = holder["wallet"]
                share = holder["share"]

                # Get graph features
                in_degree = funding_graph.get_in_degree(wallet)
                out_degree = funding_graph.get_out_degree(wallet)

                features_list.append(WalletFeatureVector(
                    wallet=wallet,
                    balance=float(holder["balance"]),
                    share=share,
                    rank=i + 1,
                    entry_time_relative=0.0,  # Would need historical data
                    holding_duration=0.0,
                    position_volatility=0.0,
                    delta_balance_7d=0.0,
                    delta_balance_30d=0.0,
                    trade_count=0,
                    burstiness=0.0,
                    swap_frequency=0.0,
                    lp_interaction_ratio=0.0,
                    in_degree=in_degree,
                    out_degree=out_degree,
                    eigenvector_centrality=0.0,
                    shared_funder_count=shared_funder_counts.get(wallet, 0),
                ))

            # Cluster into archetypes
            assignments = cluster_wallets(features_list)
            archetype_assignments = {
                wallet: {
                    "archetype": a.archetype.value,
                    "confidence": a.confidence,
                    "matching_features": a.matching_features,
                }
                for wallet, a in assignments.items()
            }

            # Get distribution
            archetype_distribution = get_archetype_distribution(assignments)

            logger.info(
                "archetypes_computed",
                wallet_count=len(archetype_assignments),
                distribution=archetype_distribution,
            )

        except Exception as e:
            logger.warning("archetype_computation_failed", error=str(e))

        # Add archetype to each holder
        for holder in holders:
            wallet = holder["wallet"]
            if wallet in archetype_assignments:
                holder["archetype"] = archetype_assignments[wallet]["archetype"]
                holder["archetype_confidence"] = archetype_assignments[wallet]["confidence"]
            else:
                holder["archetype"] = "unknown"
                holder["archetype_confidence"] = 0.0

            # Add shared funder info if exists
            if wallet in shared_funders_map:
                holder["shared_funders"] = shared_funders_map[wallet]

        return {
            "token_mint": mint,
            "price": price_data,
            "holder_count": snapshot.holder_count,
            "total_supply": snapshot.total_supply,
            "risk_score": risk_score,
            "metrics": {
                "hhi": hhi.value,
                "gini": gini.value,
                "entropy": entropy.value,
                "whale_dominance": wdr.value,
            },
            "archetypes": archetype_distribution,
            "holders": holders,
            "funding_edges": funding_edges,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    finally:
        await client.close()
        await price_provider.close()


# Serve dashboard HTML at root
@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the dashboard HTML page."""
    from fastapi.responses import FileResponse
    static_dir = FilePath(__file__).parent.parent.parent / "static"
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(status_code=404, detail="Dashboard not found")
