"""
Multi-Evidence Coordination Detection Orchestrator.

Orchestrates the complete multi-evidence coordination detection pipeline:
1. Build wallet contexts from available data
2. Create candidate blocks using blocking strategies
3. Compute pairwise coordination features within blocks
4. Score coordination using weighted multi-evidence formula
5. Validate against null models
6. Classify coordination with statistical significance

CRITICAL: This replaces the failed temporal-only coordination detector.
The previous detector showed 0 significant detections under null model
validation and cannot distinguish true coordination from launch-time noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence
from collections import defaultdict

import structlog

from ..core.config import settings
from .features import (
    CoordinationFeatures,
    WalletContext,
    compute_pairwise_coordination_features,
    build_wallet_context,
)
from .blocking import (
    CandidateBlock,
    BlockingStrategy,
    BlockingResult,
    create_candidate_blocks,
)
from .scoring import (
    MultiEvidenceCoordinationScore,
    CoordinationClassification,
    CoordinationWeights,
    CoordinationLevel,
    compute_coordination_score,
    classify_coordination,
    aggregate_cluster_score,
    get_dominant_evidence_types,
)
from .null_model import (
    NullModelType,
    NullModelResult,
    NullModelValidation,
    run_null_model_validation,
    summarize_null_validation,
)

logger = structlog.get_logger()


@dataclass
class CoordinatedCluster:
    """A detected coordinated cluster with full evidence."""

    cluster_id: str
    wallets: list[str]
    size: int

    # Scores
    coordination_score: float
    z_score: float
    empirical_p: float

    # Classification
    classification: CoordinationClassification
    is_significant: bool
    coordination_level: CoordinationLevel

    # Evidence
    evidence_types: list[str]
    evidence_counts: dict[str, int]
    pairwise_scores: list[MultiEvidenceCoordinationScore]

    # Null model validation
    null_validation: Optional[NullModelValidation] = None

    # Blocking info
    source_block: Optional[CandidateBlock] = None


@dataclass
class CoordinationResult:
    """Complete result of multi-evidence coordination detection."""

    # Detection results
    coordinated_clusters: list[CoordinatedCluster]
    all_classifications: list[CoordinationClassification]

    # Statistics
    total_wallets_analyzed: int
    total_pairs_compared: int
    clusters_significant: int
    clusters_rejected: int

    # Blocking statistics
    blocking_result: Optional[BlockingResult] = None

    # Configuration used
    weights_version: str = "v1.0.0"
    z_threshold: float = 2.5
    p_threshold: float = 0.01
    min_evidence_types: int = 3
    min_cluster_size: int = 3

    # Metadata
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Warnings
    warnings: list[str] = field(default_factory=list)


class MultiEvidenceCoordinationDetector:
    """
    Main orchestrator for multi-evidence coordination detection.

    Usage:
        detector = MultiEvidenceCoordinationDetector()
        result = detector.detect(
            funding_graph=graph,
            trade_events=events,
            target_wallets=wallets,
        )

    CRITICAL: This detector requires multiple independent evidence types
    to classify coordination. It will NEVER classify from timing alone.
    """

    def __init__(
        self,
        weights: Optional[CoordinationWeights] = None,
        z_threshold: Optional[float] = None,
        p_threshold: Optional[float] = None,
        min_evidence_types: Optional[int] = None,
        min_cluster_size: Optional[int] = None,
        n_permutations: int = 100,
        blocking_strategies: Optional[list[BlockingStrategy]] = None,
    ):
        """
        Initialize the detector.

        Args:
            weights: Custom coordination scoring weights
            z_threshold: Z-score threshold for significance
            p_threshold: P-value threshold for significance
            min_evidence_types: Minimum evidence types required
            min_cluster_size: Minimum cluster size
            n_permutations: Number of null model permutations
            blocking_strategies: Strategies for candidate blocking
        """
        # Use config defaults if not specified
        self.weights = weights or CoordinationWeights(
            shared_funder=settings.coordination_weight_shared_funder,
            funding_time=settings.coordination_weight_funding_time,
            funding_amount=settings.coordination_weight_funding_amount,
            buy_time=settings.coordination_weight_buy_time,
            trade_sequence=settings.coordination_weight_trade_sequence,
            exit_timing=settings.coordination_weight_exit_timing,
            cross_token=settings.coordination_weight_cross_token,
        )

        self.z_threshold = z_threshold or settings.coordination_z_threshold
        self.p_threshold = p_threshold or settings.coordination_p_threshold
        self.min_evidence_types = min_evidence_types or settings.coordination_min_evidence_types
        self.min_cluster_size = min_cluster_size or settings.coordination_min_cluster_size
        self.n_permutations = n_permutations

        self.blocking_strategies = blocking_strategies or [
            BlockingStrategy.SAME_FUNDER,
            BlockingStrategy.FUNDING_TIME_WINDOW,
            BlockingStrategy.TOKEN_ENTRY_WINDOW,
            BlockingStrategy.CO_PARTICIPATION,
        ]

    def detect(
        self,
        funding_graph=None,
        trade_events: Optional[list] = None,
        holder_data: Optional[dict[str, dict]] = None,
        target_wallets: Optional[list[str]] = None,
        run_null_validation: bool = True,
    ) -> CoordinationResult:
        """
        Run multi-evidence coordination detection.

        Args:
            funding_graph: FundingGraph instance
            trade_events: List of TradeEvent objects
            holder_data: Dict mapping wallet -> holder info
            target_wallets: Specific wallets to analyze
            run_null_validation: Whether to run null model validation

        Returns:
            CoordinationResult with detected clusters
        """
        warnings = []

        # Check if multi-evidence coordination is enabled
        if not settings.use_multi_evidence_coordination:
            logger.warning("multi_evidence_coordination_disabled")
            return CoordinationResult(
                coordinated_clusters=[],
                all_classifications=[],
                total_wallets_analyzed=0,
                total_pairs_compared=0,
                clusters_significant=0,
                clusters_rejected=0,
                warnings=["Multi-evidence coordination detection is disabled"],
            )

        # Check if temporal-only is disabled (it should be)
        if settings.use_temporal_coordination:
            warnings.append(
                "WARNING: use_temporal_coordination=True is deprecated and should be disabled. "
                "Temporal-only coordination FAILED null model validation."
            )
            logger.warning("temporal_coordination_enabled_warning")

        # Determine target wallets
        if target_wallets is None:
            if funding_graph is not None:
                target_wallets = list(funding_graph._wallet_set)
            elif trade_events:
                target_wallets = list(set(e.wallet_address for e in trade_events))
            else:
                return CoordinationResult(
                    coordinated_clusters=[],
                    all_classifications=[],
                    total_wallets_analyzed=0,
                    total_pairs_compared=0,
                    clusters_significant=0,
                    clusters_rejected=0,
                    warnings=["No wallets to analyze"],
                )

        if len(target_wallets) < self.min_cluster_size:
            return CoordinationResult(
                coordinated_clusters=[],
                all_classifications=[],
                total_wallets_analyzed=len(target_wallets),
                total_pairs_compared=0,
                clusters_significant=0,
                clusters_rejected=0,
                warnings=[f"Insufficient wallets: {len(target_wallets)} < {self.min_cluster_size}"],
            )

        logger.info(
            "starting_multi_evidence_detection",
            total_wallets=len(target_wallets),
            strategies=len(self.blocking_strategies),
        )

        # Step 1: Build wallet contexts
        contexts = {}
        for wallet in target_wallets:
            ctx = build_wallet_context(
                address=wallet,
                funding_graph=funding_graph,
                trade_events=trade_events,
                holder_data=holder_data.get(wallet) if holder_data else None,
            )
            contexts[wallet] = ctx

        # Step 2: Create candidate blocks
        blocking_result = create_candidate_blocks(
            contexts=contexts,
            strategies=self.blocking_strategies,
            min_block_size=self.min_cluster_size,
        )

        if not blocking_result.blocks:
            return CoordinationResult(
                coordinated_clusters=[],
                all_classifications=[],
                total_wallets_analyzed=len(target_wallets),
                total_pairs_compared=0,
                clusters_significant=0,
                clusters_rejected=0,
                blocking_result=blocking_result,
                warnings=["No candidate blocks found"],
            )

        # Step 3: Process each block
        all_classifications = []
        coordinated_clusters = []
        total_pairs = 0

        for block in blocking_result.blocks:
            if len(block.wallets) < self.min_cluster_size:
                continue

            # Compute pairwise features and scores
            pairwise_scores = []
            wallet_list = block.wallets
            for i in range(len(wallet_list)):
                for j in range(i + 1, len(wallet_list)):
                    w1, w2 = wallet_list[i], wallet_list[j]
                    if w1 in contexts and w2 in contexts:
                        features = compute_pairwise_coordination_features(contexts[w1], contexts[w2])
                        score = compute_coordination_score(features, self.weights)
                        pairwise_scores.append(score)
                        total_pairs += 1

            if not pairwise_scores:
                continue

            # Compute aggregate score
            cluster_score = aggregate_cluster_score(pairwise_scores)

            # Run null model validation if requested
            if run_null_validation:
                null_validation = run_null_model_validation(
                    contexts=contexts,
                    cluster_wallets=block.wallets,
                    n_permutations=self.n_permutations,
                    weights=self.weights,
                    z_threshold=self.z_threshold,
                    p_threshold=self.p_threshold,
                    cluster_id=block.block_id,
                )
                null_mean = null_validation.results_by_type.get(
                    NullModelType.TIMESTAMP_SHUFFLE,
                    list(null_validation.results_by_type.values())[0] if null_validation.results_by_type else None
                )
                if null_mean:
                    z_score = null_mean.z_score
                    empirical_p = null_validation.combined_p_value
                else:
                    z_score = 0.0
                    empirical_p = 1.0
            else:
                null_validation = None
                z_score = 0.0
                empirical_p = 1.0

            # Classify
            classification = classify_coordination(
                cluster_wallets=block.wallets,
                pairwise_scores=pairwise_scores,
                null_mean=null_validation.results_by_type.get(
                    NullModelType.TIMESTAMP_SHUFFLE,
                    list(null_validation.results_by_type.values())[0] if null_validation and null_validation.results_by_type else None
                ).null_mean if null_validation and null_validation.results_by_type else 0.0,
                null_std=null_validation.results_by_type.get(
                    NullModelType.TIMESTAMP_SHUFFLE,
                    list(null_validation.results_by_type.values())[0] if null_validation and null_validation.results_by_type else None
                ).null_std if null_validation and null_validation.results_by_type else 1.0,
                empirical_p=empirical_p,
                z_threshold=self.z_threshold,
                p_threshold=self.p_threshold,
                min_evidence_types=self.min_evidence_types,
                min_cluster_size=self.min_cluster_size,
                cluster_id=block.block_id,
            )

            all_classifications.append(classification)

            if classification.is_coordinated:
                evidence_types = get_dominant_evidence_types(pairwise_scores)

                cluster = CoordinatedCluster(
                    cluster_id=block.block_id,
                    wallets=block.wallets,
                    size=len(block.wallets),
                    coordination_score=cluster_score,
                    z_score=classification.z_score,
                    empirical_p=classification.empirical_p,
                    classification=classification,
                    is_significant=True,
                    coordination_level=classification.coordination_level,
                    evidence_types=evidence_types,
                    evidence_counts=classification.evidence_summary,
                    pairwise_scores=pairwise_scores,
                    null_validation=null_validation,
                    source_block=block,
                )
                coordinated_clusters.append(cluster)

        # Sort clusters by z-score
        coordinated_clusters.sort(key=lambda c: c.z_score, reverse=True)

        result = CoordinationResult(
            coordinated_clusters=coordinated_clusters,
            all_classifications=all_classifications,
            total_wallets_analyzed=len(target_wallets),
            total_pairs_compared=total_pairs,
            clusters_significant=len(coordinated_clusters),
            clusters_rejected=len(all_classifications) - len(coordinated_clusters),
            blocking_result=blocking_result,
            z_threshold=self.z_threshold,
            p_threshold=self.p_threshold,
            min_evidence_types=self.min_evidence_types,
            min_cluster_size=self.min_cluster_size,
            warnings=warnings,
        )

        logger.info(
            "multi_evidence_detection_complete",
            total_wallets=len(target_wallets),
            total_pairs=total_pairs,
            clusters_significant=len(coordinated_clusters),
            clusters_rejected=result.clusters_rejected,
        )

        return result


def get_coordination_detector() -> MultiEvidenceCoordinationDetector:
    """Factory function to get configured detector instance."""
    return MultiEvidenceCoordinationDetector()
