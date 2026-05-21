"""
Risk Scoring Module for SHI.

Computes token-level risk scores:
- Token Stability Score (0-100)
- Sell Pressure Index
- Sybil Probability Index

All scores are uncertainty-aware with confidence intervals.

Sprint 8 Extension:
- PnLCandidateFeatures: CANDIDATE features for PnL-based risk (requires validation)
- build_pnl_candidate_features: Helper to build with hard rule enforcement
"""

from .scoring import (
    compute_stability_score,
    compute_sell_pressure,
    compute_sybil_probability,
    generate_risk_report,
    RiskReport,
    # Sprint 8 CANDIDATE features
    PnLCandidateFeatures,
    build_pnl_candidate_features,
)

__all__ = [
    "compute_stability_score",
    "compute_sell_pressure",
    "compute_sybil_probability",
    "generate_risk_report",
    "RiskReport",
    # Sprint 8 CANDIDATE features
    "PnLCandidateFeatures",
    "build_pnl_candidate_features",
]
