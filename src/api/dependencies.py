"""Dependency injection for the SHI API.

This module provides FastAPI dependencies for injecting
services and configuration into route handlers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import AsyncGenerator

import structlog

from src.core.config import Settings, settings as app_settings
from src.bayesian import RiskBeliefModel

logger = structlog.get_logger()

# Cached instances
_risk_models: dict[str, RiskBeliefModel] = {}


@lru_cache()
def get_settings() -> Settings:
    """Get application settings.

    Returns
    -------
    Settings
        Application configuration.
    """
    return app_settings


async def get_orchestrator():
    """Get analysis orchestrator.

    Yields
    ------
    AnalysisOrchestrator | None
        Orchestrator instance or None if not available.

    Notes
    -----
    This is a placeholder that returns None. In production,
    this would instantiate or retrieve the orchestrator with
    proper database and cache connections.
    """
    # Placeholder - in production would create orchestrator
    # with database pool and cache connections
    try:
        # Return None for now - actual implementation would
        # create and return an AnalysisOrchestrator instance
        yield None
    finally:
        pass


async def get_forecaster():
    """Get capital flow forecaster.

    Yields
    ------
    CapitalFlowForecaster | None
        Forecaster instance or None if not available.
    """
    # Placeholder - in production would return forecaster
    yield None


def get_risk_model(token_mint: str) -> RiskBeliefModel:
    """Get or create risk belief model for a token.

    Parameters
    ----------
    token_mint : str
        Token mint address.

    Returns
    -------
    RiskBeliefModel
        Risk belief model for the token.
    """
    if token_mint not in _risk_models:
        # Create new model with default priors
        _risk_models[token_mint] = RiskBeliefModel(
            prior_alpha=1.0,
            prior_beta=2.0,  # Slightly skeptical prior
        )
        logger.info("risk_model_created", token_mint=token_mint)

    return _risk_models[token_mint]


def get_risk_model_dependency(token_mint: str):
    """Dependency factory for risk model.

    Parameters
    ----------
    token_mint : str
        Token mint address.

    Returns
    -------
    Callable
        Dependency that returns the risk model.
    """
    def _get_model() -> RiskBeliefModel:
        return get_risk_model(token_mint)
    return _get_model


async def get_sequence_encoder():
    """Get wallet action sequence encoder.

    Yields
    ------
    WalletActionEncoder
        Encoder instance.
    """
    from src.sequence import WalletActionEncoder

    encoder = WalletActionEncoder()
    yield encoder


async def get_signature_detector():
    """Get dump signature detector.

    Yields
    ------
    DumpSignatureDetector
        Detector instance.
    """
    from src.sequence import DumpSignatureDetector

    detector = DumpSignatureDetector()
    yield detector


def clear_risk_models() -> None:
    """Clear cached risk models.

    Used for testing and reset operations.
    """
    global _risk_models
    _risk_models.clear()
    logger.info("risk_models_cleared")
