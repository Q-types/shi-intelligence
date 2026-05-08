"""Bayesian risk estimation with uncertainty quantification.

This module provides Bayesian risk models that maintain probability
distributions over risk estimates, enabling:
- Uncertainty-aware risk assessment
- Continuous belief updating with new evidence
- Credible intervals for risk predictions
"""

from __future__ import annotations

from .priors import (
    PriorDistribution,
    BetaPrior,
    GammaPrior,
    NormalPrior,
    create_default_priors,
)
from .updater import (
    Evidence,
    EvidenceType,
    EvidenceBatch,
    BayesianUpdater,
)
from .risk_belief import (
    RiskBeliefModel,
    RiskBeliefState,
    BeliefUpdate,
    RiskEstimate,
)

__all__ = [
    # Priors
    "PriorDistribution",
    "BetaPrior",
    "GammaPrior",
    "NormalPrior",
    "create_default_priors",
    # Evidence and updating
    "Evidence",
    "EvidenceType",
    "EvidenceBatch",
    "BayesianUpdater",
    # Risk beliefs
    "RiskBeliefModel",
    "RiskBeliefState",
    "BeliefUpdate",
    "RiskEstimate",
]
