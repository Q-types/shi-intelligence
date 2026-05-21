"""
Risk Scoring Functions.

Computes token-level risk indicators per PDR Section 6.
All scores are normalized against baseline datasets.

Sprint 8 Extension:
CANDIDATE features for PnL and profit extraction behaviour.
These are NOT production defaults - controlled by feature flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import structlog

from ..core.types import MetricOutput
from ..metrics.normalization import BaselineStatistics

logger = structlog.get_logger()


# ============================================================================
# Sprint 8: Candidate Feature Structures (NOT production defaults)
# ============================================================================


@dataclass(frozen=True)
class PnLCandidateFeatures:
    """
    Sprint 8 CANDIDATE features for PnL-based risk assessment.

    HARD RULES:
    1. confidence_score is NEVER zero (min 0.1)
    2. When confidence < threshold, show ranges not precise values
    3. Separate realised from unrealised
    4. Accounting method must be explicit
    """

    # Unrealised exposure
    unrealised_pnl_pct: float | None = None
    unrealised_pnl_confidence: float = 0.1  # Never zero

    # Realised behaviour
    realised_profit_rate: float | None = None
    realised_profit_confidence: float = 0.1

    # Exit efficiency
    exit_efficiency: float | None = None  # exit_price / peak_price
    exit_efficiency_confidence: float = 0.1

    # Liquidity sensitivity
    liquidity_sensitive_exit_score: float | None = None
    liquidity_exit_confidence: float = 0.1

    # Accounting method used
    accounting_method: str = "fifo"

    # Overall confidence (min of all components)
    overall_confidence: float = 0.1

    # Display flag - when False, show ranges not precise values
    display_precise: bool = False

    def to_dict(self) -> dict:
        """Export with confidence-aware display."""
        result = {
            "accounting_method": self.accounting_method,
            "overall_confidence": self.overall_confidence,
            "display_precise": self.display_precise,
        }

        # Unrealised
        if self.unrealised_pnl_pct is not None:
            if self.display_precise:
                result["unrealised_pnl_pct"] = self.unrealised_pnl_pct
            else:
                # Show range when low confidence
                result["unrealised_pnl_range"] = _confidence_range(
                    self.unrealised_pnl_pct, self.unrealised_pnl_confidence
                )
            result["unrealised_pnl_confidence"] = self.unrealised_pnl_confidence

        # Realised
        if self.realised_profit_rate is not None:
            if self.display_precise:
                result["realised_profit_rate"] = self.realised_profit_rate
            else:
                result["realised_profit_range"] = _confidence_range(
                    self.realised_profit_rate, self.realised_profit_confidence
                )
            result["realised_profit_confidence"] = self.realised_profit_confidence

        # Exit efficiency
        if self.exit_efficiency is not None:
            if self.display_precise:
                result["exit_efficiency"] = self.exit_efficiency
            else:
                result["exit_efficiency_range"] = _confidence_range(
                    self.exit_efficiency, self.exit_efficiency_confidence
                )
            result["exit_efficiency_confidence"] = self.exit_efficiency_confidence

        # Liquidity sensitivity
        if self.liquidity_sensitive_exit_score is not None:
            result["liquidity_sensitive_exit_score"] = self.liquidity_sensitive_exit_score
            result["liquidity_exit_confidence"] = self.liquidity_exit_confidence

        return result


def _confidence_range(value: float, confidence: float) -> tuple[float, float]:
    """
    Convert value to confidence-scaled range.

    Lower confidence = wider range.
    """
    # Range width inversely proportional to confidence
    width = (1 - confidence) * abs(value) * 2
    return (value - width / 2, value + width / 2)


def build_pnl_candidate_features(
    unrealised_pnl_pct: float | None = None,
    unrealised_confidence: float = 0.5,
    realised_profit_rate: float | None = None,
    realised_confidence: float = 0.5,
    exit_efficiency: float | None = None,
    exit_confidence: float = 0.5,
    liquidity_sensitive_exit_score: float | None = None,
    liquidity_confidence: float = 0.5,
    accounting_method: str = "fifo",
    min_confidence_for_display: float = 0.5,
    confidence_floor: float = 0.1,
) -> PnLCandidateFeatures:
    """
    Build PnLCandidateFeatures with hard rule enforcement.

    Sprint 8 Hard Rules:
    1. Confidence is NEVER zero (min 0.1)
    2. When confidence < threshold, display_precise=False
    3. Accounting method must be explicit

    Args:
        unrealised_pnl_pct: Unrealised PnL percentage
        unrealised_confidence: Confidence in unrealised calculation
        realised_profit_rate: Rate of realised profits
        realised_confidence: Confidence in realised calculation
        exit_efficiency: Exit price / peak price ratio
        exit_confidence: Confidence in exit efficiency
        liquidity_sensitive_exit_score: Liquidity-adjusted exit quality
        liquidity_confidence: Confidence in liquidity calculation
        accounting_method: Cost basis method used (fifo/lifo/weighted_average)
        min_confidence_for_display: Threshold for precise display
        confidence_floor: Minimum confidence (never zero)

    Returns:
        PnLCandidateFeatures with enforced hard rules
    """
    # Enforce confidence floor (Sprint 8 Hard Rule: never zero)
    unrealised_conf = max(confidence_floor, unrealised_confidence)
    realised_conf = max(confidence_floor, realised_confidence)
    exit_conf = max(confidence_floor, exit_confidence)
    liquidity_conf = max(confidence_floor, liquidity_confidence)

    # Overall confidence is minimum of all components
    confidences = [unrealised_conf, realised_conf, exit_conf, liquidity_conf]
    overall_conf = min(confidences)

    # Determine if we can display precise values
    display_precise = overall_conf >= min_confidence_for_display

    return PnLCandidateFeatures(
        unrealised_pnl_pct=unrealised_pnl_pct,
        unrealised_pnl_confidence=unrealised_conf,
        realised_profit_rate=realised_profit_rate,
        realised_profit_confidence=realised_conf,
        exit_efficiency=exit_efficiency,
        exit_efficiency_confidence=exit_conf,
        liquidity_sensitive_exit_score=liquidity_sensitive_exit_score,
        liquidity_exit_confidence=liquidity_conf,
        accounting_method=accounting_method,
        overall_confidence=overall_conf,
        display_precise=display_precise,
    )


@dataclass
class RiskReport:
    """Complete risk assessment for a token."""

    # Token identification
    mint: str
    snapshot_timestamp: datetime

    # Distribution metrics
    hhi: MetricOutput
    shannon_entropy: MetricOutput
    gini_coefficient: MetricOutput
    whale_dominance_ratio: MetricOutput

    # Coordination metrics
    churn_rate: MetricOutput
    coordination_score: MetricOutput
    funding_density: MetricOutput

    # Aggregated scores
    stability_score: float  # 0-100
    stability_confidence: tuple[float, float]  # CI
    sell_pressure_index: float
    sell_pressure_confidence: tuple[float, float]
    sybil_probability: float
    sybil_confidence: tuple[float, float]

    # Liquidity context
    liquidity_depth: float | None
    liquidity_adjusted_pressure: float | None

    # Versioning
    model_version: str
    baseline_version: str
    computed_at: datetime

    # Sprint 8 CANDIDATE features (NOT production defaults)
    # Only populated when use_pnl_features=True in config
    pnl_candidate_features: PnLCandidateFeatures | None = None

    # Disclaimer
    disclaimer: str = (
        "All outputs are observational and probabilistic. "
        "No causal inference is implied."
    )

    def to_dict(self) -> dict:
        """Export as dictionary."""
        result = {
            "mint": self.mint,
            "snapshot_timestamp": self.snapshot_timestamp.isoformat(),
            "metrics": {
                "hhi": {"value": self.hhi.value, "z_score": self.hhi.z_score},
                "entropy": {"value": self.shannon_entropy.value, "z_score": self.shannon_entropy.z_score},
                "gini": {"value": self.gini_coefficient.value, "z_score": self.gini_coefficient.z_score},
                "wdr": {"value": self.whale_dominance_ratio.value, "z_score": self.whale_dominance_ratio.z_score},
                "churn": {"value": self.churn_rate.value, "z_score": self.churn_rate.z_score},
                "coordination": {"value": self.coordination_score.value, "z_score": self.coordination_score.z_score},
            },
            "scores": {
                "stability": {
                    "value": self.stability_score,
                    "confidence_interval": self.stability_confidence,
                },
                "sell_pressure": {
                    "value": self.sell_pressure_index,
                    "confidence_interval": self.sell_pressure_confidence,
                    "liquidity_adjusted": self.liquidity_adjusted_pressure,
                },
                "sybil_probability": {
                    "value": self.sybil_probability,
                    "confidence_interval": self.sybil_confidence,
                },
            },
            "liquidity_depth": self.liquidity_depth,
            "model_version": self.model_version,
            "baseline_version": self.baseline_version,
            "computed_at": self.computed_at.isoformat(),
            "disclaimer": self.disclaimer,
        }

        # Sprint 8: Add candidate features if present
        if self.pnl_candidate_features is not None:
            result["pnl_candidate_features"] = self.pnl_candidate_features.to_dict()
            result["disclaimer"] += (
                " Sprint 8 PnL features are CANDIDATE features under validation."
            )

        return result


# Default weights for stability score (can be calibrated)
DEFAULT_STABILITY_WEIGHTS = {
    "hhi": -0.20,  # Higher HHI = less stable (negative weight)
    "gini": -0.20,  # Higher Gini = less stable
    "wdr": -0.25,  # Higher whale dominance = less stable
    "churn": -0.15,  # Higher churn = less stable
    "coordination": -0.20,  # Higher coordination = less stable (potential manipulation)
}


def compute_stability_score(
    hhi_z: float,
    gini_z: float,
    wdr_z: float,
    churn_z: float,
    coordination_z: float,
    weights: dict[str, float] | None = None,
) -> tuple[float, tuple[float, float]]:
    """
    Compute Token Stability Score (0-100).

    Per PDR Section 6.1:
    Weighted function of HHI, Gini, WDR, Churn, Coordination.

    Uses z-scores for each metric, then combines with weights.
    Score is inverted so higher = more stable.

    Args:
        hhi_z: Z-score of HHI
        gini_z: Z-score of Gini coefficient
        wdr_z: Z-score of Whale Dominance Ratio
        churn_z: Z-score of Churn Rate
        coordination_z: Z-score of Coordination Score
        weights: Optional custom weights

    Returns:
        (stability_score, (lower_ci, upper_ci))
    """
    w = weights or DEFAULT_STABILITY_WEIGHTS

    # Weighted sum of z-scores
    raw_score = (
        w["hhi"] * hhi_z
        + w["gini"] * gini_z
        + w["wdr"] * wdr_z
        + w["churn"] * churn_z
        + w["coordination"] * coordination_z
    )

    # Convert to 0-100 scale using sigmoid-like transformation
    # Raw score of 0 (baseline average) -> 50
    # More negative raw score -> higher stability
    # More positive raw score -> lower stability
    stability = 100 / (1 + np.exp(raw_score))

    # Confidence interval based on z-score uncertainty
    # Assuming ~0.5 std error on z-scores
    z_error = 0.5
    raw_lower = raw_score + z_error  # Worse case
    raw_upper = raw_score - z_error  # Better case

    ci_lower = 100 / (1 + np.exp(raw_lower))
    ci_upper = 100 / (1 + np.exp(raw_upper))

    return float(stability), (float(ci_lower), float(ci_upper))


def compute_sell_pressure(
    individual_sell_probs: Sequence[float],
    top_n: int = 10,
    liquidity_depth: float | None = None,
) -> tuple[float, float | None, tuple[float, float]]:
    """
    Compute Sell Pressure Index.

    Per PDR Section 4.9:
    Sell_Pressure = SUM_{i in Top_N} P_sell_i(T)

    Optionally adjusted for liquidity per INITIAL_PROMPT.

    Args:
        individual_sell_probs: Sell probabilities for top holders
        top_n: Number of top holders to consider
        liquidity_depth: Optional pool liquidity (for adjustment)

    Returns:
        (sell_pressure, liquidity_adjusted_pressure, (lower_ci, upper_ci))
    """
    if not individual_sell_probs:
        return 0.0, None, (0.0, 0.0)

    # Take top N
    probs = list(individual_sell_probs[:top_n])

    # Sum of probabilities
    sell_pressure = sum(probs)

    # Liquidity adjustment
    liquidity_adjusted = None
    if liquidity_depth is not None and liquidity_depth > 0:
        # Per INITIAL_PROMPT:
        # Liquidity_Adjusted_Pressure = Sell_Pressure * (1 / Liquidity_Depth_Factor)
        # Using log scale for depth factor
        depth_factor = np.log10(liquidity_depth + 1) / 6  # Normalize to ~0-1
        depth_factor = max(0.1, min(1.0, depth_factor))
        liquidity_adjusted = sell_pressure / depth_factor

    # Confidence interval (assuming ~10% uncertainty on each prob)
    prob_std = np.std(probs) if len(probs) > 1 else 0.1
    ci_half = 1.96 * prob_std * np.sqrt(len(probs))
    ci_lower = max(0, sell_pressure - ci_half)
    ci_upper = sell_pressure + ci_half

    return sell_pressure, liquidity_adjusted, (ci_lower, ci_upper)


def compute_sybil_probability(
    funding_density: float,
    shared_funder_ratio: float,
    temporal_clustering_score: float,
    coordination_score: float,
) -> tuple[float, tuple[float, float]]:
    """
    Compute Sybil Probability Index.

    Per PDR Section 6.2:
    Function of funding graph density, shared funder ratio,
    temporal clustering, coordination score.

    Args:
        funding_density: Graph density [0, 1]
        shared_funder_ratio: Proportion of wallets sharing funders [0, 1]
        temporal_clustering_score: Synchronicity of wallet creation [0, 1]
        coordination_score: Cluster-level coordination [0, 1]

    Returns:
        (sybil_probability, (lower_ci, upper_ci))
    """
    # Weighted combination
    weights = {
        "funding_density": 0.20,
        "shared_funder": 0.35,
        "temporal": 0.25,
        "coordination": 0.20,
    }

    raw_prob = (
        weights["funding_density"] * funding_density
        + weights["shared_funder"] * shared_funder_ratio
        + weights["temporal"] * temporal_clustering_score
        + weights["coordination"] * coordination_score
    )

    # Apply sigmoid to bound to [0, 1]
    # Shift so that average inputs give ~0.3 probability
    sybil_prob = 1 / (1 + np.exp(-5 * (raw_prob - 0.4)))

    # Confidence interval based on input uncertainty
    uncertainty = 0.1  # 10% uncertainty on inputs
    raw_lower = raw_prob - uncertainty
    raw_upper = raw_prob + uncertainty

    ci_lower = 1 / (1 + np.exp(-5 * (raw_lower - 0.4)))
    ci_upper = 1 / (1 + np.exp(-5 * (raw_upper - 0.4)))

    return float(sybil_prob), (float(ci_lower), float(ci_upper))


def generate_risk_report(
    mint: str,
    metrics: dict[str, MetricOutput],
    sell_probabilities: list[float],
    baseline: BaselineStatistics,
    liquidity_depth: float | None = None,
    model_version: str = "1.0.0",
    pnl_candidate_features: PnLCandidateFeatures | None = None,
) -> RiskReport:
    """
    Generate complete risk report for a token.

    Args:
        mint: Token mint address
        metrics: Dict of computed metrics
        sell_probabilities: Sell probs for top holders
        baseline: Baseline statistics for normalization
        liquidity_depth: Optional pool liquidity
        model_version: Current model version
        pnl_candidate_features: Sprint 8 CANDIDATE features (optional)

    Returns:
        Complete RiskReport
    """
    now = datetime.now(timezone.utc)

    # Normalize metrics against baseline
    hhi_norm = baseline.normalize("hhi", metrics["hhi"].value)
    gini_norm = baseline.normalize("gini_coefficient", metrics["gini"].value)
    wdr_norm = baseline.normalize("whale_dominance_ratio", metrics["wdr"].value)
    churn_norm = baseline.normalize("churn_rate", metrics["churn"].value)
    coord_norm = baseline.normalize("coordination_score", metrics["coordination"].value)

    # Compute aggregated scores
    stability, stability_ci = compute_stability_score(
        hhi_z=hhi_norm.value,
        gini_z=gini_norm.value,
        wdr_z=wdr_norm.value,
        churn_z=churn_norm.value,
        coordination_z=coord_norm.value,
    )

    sell_pressure, liq_adjusted, pressure_ci = compute_sell_pressure(
        sell_probabilities,
        liquidity_depth=liquidity_depth,
    )

    # For Sybil, we need additional inputs
    sybil_prob, sybil_ci = compute_sybil_probability(
        funding_density=metrics.get("funding_density", MetricOutput(
            metric_name="funding_density", value=0, version="1.0.0", computed_at=now
        )).value,
        shared_funder_ratio=metrics.get("shared_funder_ratio", MetricOutput(
            metric_name="shared_funder_ratio", value=0, version="1.0.0", computed_at=now
        )).value,
        temporal_clustering_score=0.0,  # Would need to compute
        coordination_score=metrics["coordination"].value,
    )

    # Update metrics with z-scores
    metrics["hhi"] = MetricOutput(
        metric_name="hhi",
        value=metrics["hhi"].value,
        z_score=hhi_norm.value,
        percentile=None,
        version=metrics["hhi"].version,
        computed_at=metrics["hhi"].computed_at,
        baseline_version=baseline.version,
    )

    return RiskReport(
        mint=mint,
        snapshot_timestamp=now,
        hhi=metrics["hhi"],
        shannon_entropy=metrics["entropy"],
        gini_coefficient=metrics["gini"],
        whale_dominance_ratio=metrics["wdr"],
        churn_rate=metrics["churn"],
        coordination_score=metrics["coordination"],
        funding_density=metrics.get("funding_density", MetricOutput(
            metric_name="funding_density", value=0, version="1.0.0", computed_at=now
        )),
        stability_score=stability,
        stability_confidence=stability_ci,
        sell_pressure_index=sell_pressure,
        sell_pressure_confidence=pressure_ci,
        sybil_probability=sybil_prob,
        sybil_confidence=sybil_ci,
        liquidity_depth=liquidity_depth,
        liquidity_adjusted_pressure=liq_adjusted,
        model_version=model_version,
        baseline_version=baseline.version,
        computed_at=now,
        pnl_candidate_features=pnl_candidate_features,
    )
