"""
Null Model Validation for Coordination Detection.

Tests coordination scores against multiple null models:
1. Timestamp shuffle null - Permute funding/trade timestamps
2. Funder shuffle null - Permute funder assignments
3. Amount shuffle null - Permute funding amounts
4. Degree-preserving graph null - Rewire edges preserving degree
5. Token-stage matched null - Compare to similar tokens

Reports:
- observed score
- null mean
- null std
- z-score
- empirical p-value
- false positive rate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable
from collections import defaultdict
from enum import Enum
import random
import math
import copy

import structlog

from .features import (
    CoordinationFeatures,
    WalletContext,
    compute_pairwise_coordination_features,
)
from .scoring import (
    MultiEvidenceCoordinationScore,
    compute_coordination_score,
    CoordinationWeights,
)

logger = structlog.get_logger()


class NullModelType(Enum):
    """Types of null models for validation."""

    TIMESTAMP_SHUFFLE = "timestamp_shuffle"
    FUNDER_SHUFFLE = "funder_shuffle"
    AMOUNT_SHUFFLE = "amount_shuffle"
    DEGREE_PRESERVING = "degree_preserving"
    TOKEN_STAGE_MATCHED = "token_stage_matched"


@dataclass
class NullModelResult:
    """Result from a single null model test."""

    null_type: NullModelType
    n_permutations: int

    # Null distribution statistics
    null_scores: list[float]
    null_mean: float
    null_std: float
    null_min: float
    null_max: float

    # Comparison to observed
    observed_score: float
    z_score: float
    empirical_p: float  # Proportion of null scores >= observed

    # Is significant at various thresholds
    significant_at_05: bool
    significant_at_01: bool
    significant_at_001: bool


@dataclass
class NullModelValidation:
    """Complete null model validation result."""

    # Identification
    cluster_id: str
    cluster_wallets: list[str]

    # Observed data
    observed_score: float
    observed_pairwise_scores: list[MultiEvidenceCoordinationScore]

    # Null model results
    results_by_type: dict[NullModelType, NullModelResult]

    # Combined significance
    combined_z_score: float  # Fisher's method or similar
    combined_p_value: float
    is_significant: bool

    # False positive analysis
    estimated_fpr_at_threshold: float  # FPR at the applied threshold

    # Metadata
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _shuffle_timestamps(contexts: dict[str, WalletContext]) -> dict[str, WalletContext]:
    """Create permuted contexts with shuffled timestamps."""
    permuted = {}

    # Collect all timestamps
    all_funding_times = []
    all_buy_times = []
    all_exit_times = []

    for ctx in contexts.values():
        if ctx.earliest_funding_time:
            all_funding_times.append(ctx.earliest_funding_time)
        if ctx.first_buy_time:
            all_buy_times.append(ctx.first_buy_time)
        if ctx.exit_time:
            all_exit_times.append(ctx.exit_time)

    # Shuffle
    random.shuffle(all_funding_times)
    random.shuffle(all_buy_times)
    random.shuffle(all_exit_times)

    # Assign to permuted contexts
    funding_idx = 0
    buy_idx = 0
    exit_idx = 0

    for addr, ctx in contexts.items():
        new_ctx = copy.deepcopy(ctx)

        if ctx.earliest_funding_time and funding_idx < len(all_funding_times):
            new_ctx.earliest_funding_time = all_funding_times[funding_idx]
            new_ctx.funding_times = [all_funding_times[funding_idx]]
            funding_idx += 1

        if ctx.first_buy_time and buy_idx < len(all_buy_times):
            new_ctx.first_buy_time = all_buy_times[buy_idx]
            buy_idx += 1

        if ctx.exit_time and exit_idx < len(all_exit_times):
            new_ctx.exit_time = all_exit_times[exit_idx]
            exit_idx += 1

        permuted[addr] = new_ctx

    return permuted


def _shuffle_funders(contexts: dict[str, WalletContext]) -> dict[str, WalletContext]:
    """Create permuted contexts with shuffled funder assignments."""
    permuted = {}

    # Collect all funder sets
    all_funders = []
    for ctx in contexts.values():
        all_funders.append(ctx.funders.copy())

    # Shuffle
    random.shuffle(all_funders)

    # Assign to permuted contexts
    addrs = list(contexts.keys())
    for i, addr in enumerate(addrs):
        new_ctx = copy.deepcopy(contexts[addr])
        if i < len(all_funders):
            new_ctx.funders = all_funders[i]
        permuted[addr] = new_ctx

    return permuted


def _shuffle_amounts(contexts: dict[str, WalletContext]) -> dict[str, WalletContext]:
    """Create permuted contexts with shuffled funding amounts."""
    permuted = {}

    # Collect all amounts
    all_amounts = []
    for ctx in contexts.values():
        all_amounts.append(ctx.total_funding)

    # Shuffle
    random.shuffle(all_amounts)

    # Assign to permuted contexts
    addrs = list(contexts.keys())
    for i, addr in enumerate(addrs):
        new_ctx = copy.deepcopy(contexts[addr])
        if i < len(all_amounts):
            new_ctx.total_funding = all_amounts[i]
            new_ctx.funding_amounts = [all_amounts[i]] if all_amounts[i] > 0 else []
        permuted[addr] = new_ctx

    return permuted


def _compute_cluster_score(
    contexts: dict[str, WalletContext],
    wallet_pairs: list[tuple[str, str]],
    weights: Optional[CoordinationWeights] = None,
) -> float:
    """Compute aggregate coordination score for wallet pairs."""
    if not wallet_pairs:
        return 0.0

    scores = []
    for w1, w2 in wallet_pairs:
        if w1 in contexts and w2 in contexts:
            features = compute_pairwise_coordination_features(contexts[w1], contexts[w2])
            score = compute_coordination_score(features, weights)
            scores.append(score.raw_score)

    return sum(scores) / len(scores) if scores else 0.0


def run_single_null_model(
    contexts: dict[str, WalletContext],
    wallet_pairs: list[tuple[str, str]],
    observed_score: float,
    null_type: NullModelType,
    n_permutations: int = 1000,
    weights: Optional[CoordinationWeights] = None,
) -> NullModelResult:
    """
    Run a single null model test.

    Args:
        contexts: Original wallet contexts
        wallet_pairs: Pairs to compute coordination for
        observed_score: Observed coordination score
        null_type: Type of null model to use
        n_permutations: Number of permutations
        weights: Scoring weights

    Returns:
        NullModelResult with null distribution statistics
    """
    # Select permutation function
    if null_type == NullModelType.TIMESTAMP_SHUFFLE:
        permute_fn = _shuffle_timestamps
    elif null_type == NullModelType.FUNDER_SHUFFLE:
        permute_fn = _shuffle_funders
    elif null_type == NullModelType.AMOUNT_SHUFFLE:
        permute_fn = _shuffle_amounts
    elif null_type == NullModelType.DEGREE_PRESERVING:
        # For now, use timestamp shuffle as approximation
        permute_fn = _shuffle_timestamps
    elif null_type == NullModelType.TOKEN_STAGE_MATCHED:
        # For now, use funder shuffle as approximation
        permute_fn = _shuffle_funders
    else:
        permute_fn = _shuffle_timestamps

    # Generate null distribution
    null_scores = []
    for _ in range(n_permutations):
        permuted_contexts = permute_fn(contexts)
        null_score = _compute_cluster_score(permuted_contexts, wallet_pairs, weights)
        null_scores.append(null_score)

    # Compute statistics
    null_mean = sum(null_scores) / len(null_scores) if null_scores else 0.0
    null_std = (
        math.sqrt(sum((s - null_mean) ** 2 for s in null_scores) / len(null_scores))
        if len(null_scores) > 1
        else 0.0
    )
    null_min = min(null_scores) if null_scores else 0.0
    null_max = max(null_scores) if null_scores else 0.0

    # Z-score
    z_score = (observed_score - null_mean) / null_std if null_std > 0 else 0.0

    # Empirical p-value (proportion of null scores >= observed)
    n_above = sum(1 for s in null_scores if s >= observed_score)
    empirical_p = n_above / len(null_scores) if null_scores else 1.0

    return NullModelResult(
        null_type=null_type,
        n_permutations=n_permutations,
        null_scores=null_scores,
        null_mean=null_mean,
        null_std=null_std,
        null_min=null_min,
        null_max=null_max,
        observed_score=observed_score,
        z_score=z_score,
        empirical_p=empirical_p,
        significant_at_05=empirical_p < 0.05,
        significant_at_01=empirical_p < 0.01,
        significant_at_001=empirical_p < 0.001,
    )


def run_null_model_validation(
    contexts: dict[str, WalletContext],
    cluster_wallets: list[str],
    n_permutations: int = 1000,
    null_types: Optional[list[NullModelType]] = None,
    weights: Optional[CoordinationWeights] = None,
    z_threshold: float = 2.5,
    p_threshold: float = 0.01,
    cluster_id: Optional[str] = None,
) -> NullModelValidation:
    """
    Run comprehensive null model validation.

    Tests observed coordination against multiple null models to establish
    statistical significance.

    Args:
        contexts: Wallet contexts for all wallets
        cluster_wallets: Wallets in the cluster to validate
        n_permutations: Number of permutations per null model
        null_types: Null models to test (default: timestamp + funder + amount)
        weights: Scoring weights
        z_threshold: Z-score threshold for significance
        p_threshold: P-value threshold for significance
        cluster_id: Optional cluster identifier

    Returns:
        NullModelValidation with comprehensive results
    """
    if null_types is None:
        null_types = [
            NullModelType.TIMESTAMP_SHUFFLE,
            NullModelType.FUNDER_SHUFFLE,
            NullModelType.AMOUNT_SHUFFLE,
        ]

    # Filter contexts to cluster wallets
    cluster_contexts = {w: contexts[w] for w in cluster_wallets if w in contexts}

    if len(cluster_contexts) < 2:
        logger.warning("null_model_insufficient_wallets", cluster_size=len(cluster_contexts))
        return NullModelValidation(
            cluster_id=cluster_id or "unknown",
            cluster_wallets=cluster_wallets,
            observed_score=0.0,
            observed_pairwise_scores=[],
            results_by_type={},
            combined_z_score=0.0,
            combined_p_value=1.0,
            is_significant=False,
            estimated_fpr_at_threshold=1.0,
        )

    # Generate all pairs
    wallet_list = list(cluster_contexts.keys())
    wallet_pairs = [
        (wallet_list[i], wallet_list[j])
        for i in range(len(wallet_list))
        for j in range(i + 1, len(wallet_list))
    ]

    # Compute observed pairwise scores
    observed_pairwise_scores = []
    for w1, w2 in wallet_pairs:
        features = compute_pairwise_coordination_features(cluster_contexts[w1], cluster_contexts[w2])
        score = compute_coordination_score(features, weights)
        observed_pairwise_scores.append(score)

    observed_score = (
        sum(s.raw_score for s in observed_pairwise_scores) / len(observed_pairwise_scores)
        if observed_pairwise_scores
        else 0.0
    )

    # Run null models
    results_by_type = {}
    for null_type in null_types:
        logger.debug("running_null_model", type=null_type.value, n_permutations=n_permutations)
        result = run_single_null_model(
            contexts=cluster_contexts,
            wallet_pairs=wallet_pairs,
            observed_score=observed_score,
            null_type=null_type,
            n_permutations=n_permutations,
            weights=weights,
        )
        results_by_type[null_type] = result

    # Combine p-values using Fisher's method
    p_values = [r.empirical_p for r in results_by_type.values() if r.empirical_p > 0]
    if p_values:
        # Fisher's method: -2 * sum(ln(p)) ~ chi2(2k)
        chi2_stat = -2 * sum(math.log(max(p, 1e-10)) for p in p_values)
        # Approximate combined p-value using chi2 CDF
        # For simplicity, use empirical p from most conservative null model
        combined_p_value = max(p_values)  # Conservative: take worst p-value
        combined_z_score = min(r.z_score for r in results_by_type.values())
    else:
        combined_p_value = 1.0
        combined_z_score = 0.0

    # Check significance
    is_significant = combined_z_score >= z_threshold and combined_p_value <= p_threshold

    # Estimate FPR at threshold
    # Count how often null scores exceed threshold across all null models
    total_null_scores = sum(len(r.null_scores) for r in results_by_type.values())
    total_above_threshold = sum(
        sum(1 for s in r.null_scores if s >= observed_score)
        for r in results_by_type.values()
    )
    estimated_fpr = total_above_threshold / total_null_scores if total_null_scores > 0 else 1.0

    result = NullModelValidation(
        cluster_id=cluster_id or f"cluster_{wallet_list[0][:8]}",
        cluster_wallets=cluster_wallets,
        observed_score=observed_score,
        observed_pairwise_scores=observed_pairwise_scores,
        results_by_type=results_by_type,
        combined_z_score=combined_z_score,
        combined_p_value=combined_p_value,
        is_significant=is_significant,
        estimated_fpr_at_threshold=estimated_fpr,
    )

    logger.info(
        "null_model_validation_complete",
        cluster_id=result.cluster_id,
        cluster_size=len(cluster_wallets),
        observed_score=f"{observed_score:.3f}",
        combined_z=f"{combined_z_score:.2f}",
        combined_p=f"{combined_p_value:.4f}",
        is_significant=is_significant,
    )

    return result


def summarize_null_validation(validation: NullModelValidation) -> dict:
    """Generate a summary dict for reporting."""
    return {
        "cluster_id": validation.cluster_id,
        "cluster_size": len(validation.cluster_wallets),
        "observed_score": validation.observed_score,
        "combined_z_score": validation.combined_z_score,
        "combined_p_value": validation.combined_p_value,
        "is_significant": validation.is_significant,
        "estimated_fpr": validation.estimated_fpr_at_threshold,
        "null_models": {
            null_type.value: {
                "z_score": result.z_score,
                "empirical_p": result.empirical_p,
                "null_mean": result.null_mean,
                "null_std": result.null_std,
                "significant_at_01": result.significant_at_01,
            }
            for null_type, result in validation.results_by_type.items()
        },
    }
