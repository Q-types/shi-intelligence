"""REST API for SHI Intelligence.

This module provides FastAPI endpoints for external integrations
and dashboard access to token intelligence data.
Includes WebSocket endpoints for real-time alert streaming.
"""

from __future__ import annotations

from .routes import app, create_app
from .schemas import (
    TokenAnalysisRequest,
    ForecastRequest,
    WalletProfileRequest,
    RiskUpdateRequest,
    HealthResponse,
    ErrorResponse,
)
from .dependencies import (
    get_orchestrator,
    get_forecaster,
    get_risk_model,
    get_settings,
)
from .websocket import (
    ws_manager,
    WebSocketConnectionManager,
    SubscriptionType,
    WebSocketMessage,
)

__all__ = [
    # App
    "app",
    "create_app",
    # Schemas
    "TokenAnalysisRequest",
    "ForecastRequest",
    "WalletProfileRequest",
    "RiskUpdateRequest",
    "HealthResponse",
    "ErrorResponse",
    # Dependencies
    "get_orchestrator",
    "get_forecaster",
    "get_risk_model",
    "get_settings",
    # WebSocket
    "ws_manager",
    "WebSocketConnectionManager",
    "SubscriptionType",
    "WebSocketMessage",
]
