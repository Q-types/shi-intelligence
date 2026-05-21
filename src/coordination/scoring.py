"""
Multi-Evidence Coordination Scoring.

Implements the coordination score formula:

CoordinationScore =
    w1 * shared_funder_similarity
  + w2 * funding_time_similarity
  + w3 * funding_amount_similarity
  + w4 * first_buy_time_similarity
  + w5 * trade_sequence_similarity
  + w6 * exit_timing_similarity
  + w7 * cross_token_reuse

Classification requires:
- coordination_z >= 2.5
- empirical_p <= 0.01
- minimum_evidence_types >= 3
- cluster_size >= min_cluster_size

NEVER classify coordination from timing alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from enum import Enum

import structlog

from .features import CoordinationFeatures

logger = structlog.get_logger()


@dataclass
class CoordinationWeights:
    """Configurable weights for coordination score components."""

    shared_funder: float = 0.25
    funding_time: float = 0.15
    funding_amount: float = 0.10
    buy_time: float = 0.15
    trade_sequence: float = 0.10
    exit_timing: float = 0.10
    cross_token: float = 0.15

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = (
            self.shared_funder
            + self.funding_time
            + self.funding_amount
            + self.buy_time
            + self.trade_sequence
            + self.exit_timing
            + self.cross_token
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def to_dict(self) -> dict:
        return {
            "shared_funder": self.shared_funder,
            "funding_time": self.funding_time,
            "funding_amount": self.funding_amount,
            "buy_time": self.buy_time,
            "trade_sequence": self.trade_sequence,
            "exit_timing": self.exit_timing,
            "cross_token": self.cross_token,
        }


class CoordinationLevel(Enum):
    """Classification levels for coordination."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class MultiEvidenceCoordinationScore:
    """Score output from multi-evidence coordination analysis."""

    # Pair identification
    wallet1: str
    wallet2: str

    # Raw score (0-1)
    raw_score: float

    # Component contributions
    shared_funder_contribution: float
    funding_time_contribution: float
    funding_amount_contribution: float
    buy_time_contribution: float
    trade_sequence_contribution: float
    exit_timing_contribution: float
    cross_token_contribution: float

    # Evidence quality
    evidence_types_present: int
    evidence_type_flags: dict[str, bool] = field(default_factory=dict)

    # Weights used
    weights_version: str = "v1.0.0"

    def get_component_breakdown(self) -> dict[str, float]:
        """Get contribution breakdown for explanation."""
        return {
            "shared_funder": self.shared_funder_contribution,
            "funding_time": self.funding_time_contribution,
            "funding_amount": self.funding_amount_contribution,
            "buy_time": self.buy_time_contribution,
            "trade_sequence": self.trade_sequence_contribution,
            "exit_timing": self.exit_timing_contribution,
            "cross_token": self.cross_token_contribution,
        }


@dataclass
class CoordinationClassification:
    """Classification result for a wallet cluster."""

    cluster_id: str
    wallets: list[str]
    cluster_size: int

    # Statistical significance
    observed_score: float
    null_mean: float
    null_std: float
    z_score: float
    empirical_p: float

    # Classification decision
    is_coordinated: bool
    coordination_level: CoordinationLevel
    evidence_types_count: int

    # Thresholds used
    z_threshold: float
    p_threshold: float
    min_evidence_types: int
    min_cluster_size: int

    # Explanation
    classification_reason: str
    evidence_summary: dict = field(default_factory=dict)

    # Metadata
    classified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def compute_coordination_score(
    features: CoordinationFeatures,
    weights: Optional[CoordinationWeights] = None,
) -> MultiEvidenceCoordinationScore:
    """
    Compute multi-evidence coordination score for a wallet pair.

    Args:
        features: CoordinationFeatures for the pair
        weights: Optional custom weights (default: standard weights)

    Returns:
        MultiEvidenceCoordinationScore with raw score and component breakdown
    """
    if weights is None:
        weights = CoordinationWeights()

    # Extract component values
    shared_funder_val = (
        1.0 if features.funding.shared_funder_binary else features.funding.funder_overlap_jaccard
    )
    funding_time_val = features.funding.funding_time_similarity
    funding_amount_val = features.funding.funding_amount_similarity
    buy_time_val = features.trading.first_buy_time_similarity
    trade_sequence_val = max(
        features.trading.buy_sequence_similarity,
        features.trading.sell_sequence_similarity,
        features.trading.trade_cadence_similarity,
    )
    exit_timing_val = features.behavioral.exit_timing_similarity
    cross_token_val = features.cross_token.entity_reuse_score

    # Compute weighted contributions
    shared_funder_contrib = weights.shared_funder * shared_funder_val
    funding_time_contrib = weights.funding_time * funding_time_val
    funding_amount_contrib = weights.funding_amount * funding_amount_val
    buy_time_contrib = weights.buy_time * buy_time_val
    trade_sequence_contrib = weights.trade_sequence * trade_sequence_val
    exit_timing_contrib = weights.exit_timing * exit_timing_val
    cross_token_contrib = weights.cross_token * cross_token_val

    raw_score = (
        shared_funder_contrib
        + funding_time_contrib
        + funding_amount_contrib
        + buy_time_contrib
        + trade_sequence_contrib
        + exit_timing_contrib
        + cross_token_contrib
    )

    # Evidence type flags
    evidence_flags = {
        "shared_funder": features.funding.shared_funder_binary or features.funding.funder_overlap_jaccard > 0.3,
        "funding_time": features.funding.funding_time_similarity > 0.3,
        "funding_amount": features.funding.funding_amount_similarity > 0.3,
        "buy_time": features.trading.first_buy_time_similarity > 0.3,
        "trade_sequence": trade_sequence_val > 0.3,
        "exit_timing": features.behavioral.exit_timing_similarity > 0.3,
        "cross_token": features.cross_token.repeated_co_participation_count >= 2,
    }

    return MultiEvidenceCoordinationScore(
        wallet1=features.wallet1,
        wallet2=features.wallet2,
        raw_score=raw_score,
        shared_funder_contribution=shared_funder_contrib,
        funding_time_contribution=funding_time_contrib,
        funding_amount_contribution=funding_amount_contrib,
        buy_time_contribution=buy_time_contrib,
        trade_sequence_contribution=trade_sequence_contrib,
        exit_timing_contribution=exit_timing_contrib,
        cross_token_contribution=cross_token_contrib,
        evidence_types_present=sum(evidence_flags.values()),
        evidence_type_flags=evidence_flags,
    )


def classify_coordination(
    cluster_wallets: list[str],
    pairwise_scores: list[MultiEvidenceCoordinationScore],
    null_mean: float,
    null_std: float,
    empirical_p: float,
    z_threshold: float = 2.5,
    p_threshold: float = 0.01,
    min_evidence_types: int = 3,
    min_cluster_size: int = 3,
    cluster_id: Optional[str] = None,
) -> CoordinationClassification:
    """
    Classify whether a cluster shows significant coordination.

    CRITICAL RULES:
    1. coordination_z >= z_threshold (default 2.5)
    2. empirical_p <= p_threshold (default 0.01)
    3. minimum_evidence_types >= min_evidence_types (default 3)
    4. cluster_size >= min_cluster_size (default 3)

    NEVER classify coordination from timing alone.

    Args:
        cluster_wallets: List of wallet addresses in the cluster
        pairwise_scores: Scores for all pairs in the cluster
        null_mean: Mean score from null distribution
        null_std: Std dev from null distribution
        empirical_p: Empirical p-value from null model
        z_threshold: Z-score threshold for significance
        p_threshold: P-value threshold for significance
        min_evidence_types: Minimum evidence types required
        min_cluster_size: Minimum cluster size
        cluster_id: Optional cluster identifier

    Returns:
        CoordinationClassification with decision and explanation
    """
    cluster_size = len(cluster_wallets)

    # Compute cluster-level score as mean of pairwise scores
    if pairwise_scores:
        observed_score = sum(s.raw_score for s in pairwise_scores) / len(pairwise_scores)
    else:
        observed_score = 0.0

    # Compute z-score
    if null_std > 0:
        z_score = (observed_score - null_mean) / null_std
    else:
        z_score = 0.0 if observed_score <= null_mean else float("inf")

    # Count evidence types across all pairs
    evidence_type_counts = {
        "shared_funder": 0,
        "funding_time": 0,
        "funding_amount": 0,
        "buy_time": 0,
        "trade_sequence": 0,
        "exit_timing": 0,
        "cross_token": 0,
    }

    for score in pairwise_scores:
        for evidence_type, present in score.evidence_type_flags.items():
            if present:
                evidence_type_counts[evidence_type] += 1

    # Evidence types present in majority of pairs
    n_pairs = len(pairwise_scores) if pairwise_scores else 1
    evidence_types_present = sum(
        1 for count in evidence_type_counts.values() if count >= n_pairs / 2
    )

    # Check timing-only coordination (MUST REJECT)
    timing_only = (
        evidence_type_counts["buy_time"] >= n_pairs / 2
        or evidence_type_counts["funding_time"] >= n_pairs / 2
    ) and evidence_types_present <= 2

    # Classification decision
    passes_z = z_score >= z_threshold
    passes_p = empirical_p <= p_threshold
    passes_evidence = evidence_types_present >= min_evidence_types
    passes_size = cluster_size >= min_cluster_size
    not_timing_only = not timing_only

    is_coordinated = (
        passes_z
        and passes_p
        and passes_evidence
        and passes_size
        and not_timing_only
    )

    # Determine level
    if not is_coordinated:
        level = CoordinationLevel.NONE
    elif z_score >= 4.0 and evidence_types_present >= 5:
        level = CoordinationLevel.VERY_HIGH
    elif z_score >= 3.0 and evidence_types_present >= 4:
        level = CoordinationLevel.HIGH
    elif z_score >= 2.5 and evidence_types_present >= 3:
        level = CoordinationLevel.MODERATE
    else:
        level = CoordinationLevel.LOW

    # Build explanation
    reasons = []
    if not passes_z:
        reasons.append(f"z-score {z_score:.2f} < {z_threshold} threshold")
    if not passes_p:
        reasons.append(f"p-value {empirical_p:.4f} > {p_threshold} threshold")
    if not passes_evidence:
        reasons.append(f"only {evidence_types_present} evidence types < {min_evidence_types} required")
    if not passes_size:
        reasons.append(f"cluster size {cluster_size} < {min_cluster_size} minimum")
    if timing_only:
        reasons.append("REJECTED: timing-only coordination is not valid evidence")

    if is_coordinated:
        reason = f"Significant coordination: z={z_score:.2f}, p={empirical_p:.4f}, {evidence_types_present} evidence types"
    else:
        reason = "Not coordinated: " + "; ".join(reasons)

    return CoordinationClassification(
        cluster_id=cluster_id or f"cluster_{cluster_wallets[0][:8]}",
        wallets=cluster_wallets,
        cluster_size=cluster_size,
        observed_score=observed_score,
        null_mean=null_mean,
        null_std=null_std,
        z_score=z_score,
        empirical_p=empirical_p,
        is_coordinated=is_coordinated,
        coordination_level=level,
        evidence_types_count=evidence_types_present,
        z_threshold=z_threshold,
        p_threshold=p_threshold,
        min_evidence_types=min_evidence_types,
        min_cluster_size=min_cluster_size,
        classification_reason=reason,
        evidence_summary=evidence_type_counts,
    )


def aggregate_cluster_score(
    pairwise_scores: list[MultiEvidenceCoordinationScore],
) -> float:
    """Aggregate pairwise scores into a single cluster score."""
    if not pairwise_scores:
        return 0.0

    # Use mean of all pairwise scores
    return sum(s.raw_score for s in pairwise_scores) / len(pairwise_scores)


def get_dominant_evidence_types(
    pairwise_scores: list[MultiEvidenceCoordinationScore],
    threshold_ratio: float = 0.5,
) -> list[str]:
    """Get evidence types present in majority of pairs."""
    if not pairwise_scores:
        return []

    n_pairs = len(pairwise_scores)
    type_counts = {
        "shared_funder": 0,
        "funding_time": 0,
        "funding_amount": 0,
        "buy_time": 0,
        "trade_sequence": 0,
        "exit_timing": 0,
        "cross_token": 0,
    }

    for score in pairwise_scores:
        for evidence_type, present in score.evidence_type_flags.items():
            if present:
                type_counts[evidence_type] += 1

    threshold = n_pairs * threshold_ratio
    return [t for t, count in type_counts.items() if count >= threshold]
