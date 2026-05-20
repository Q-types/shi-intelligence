"""
Pipeline Comparison Framework for Clustering Validation.

Compares old rule-first clustering with new transformed pipelines
to validate improvement before deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import structlog

from ...clustering.diagnostics import HDBSCANDiagnostics, ClusterDiagnostics
from ...clustering.transformations import FeatureTransformer
from ...clustering.archetypes import Archetype, assign_archetype_multi_score, WalletFeatureVector

logger = structlog.get_logger()


class ClusteringPipeline(Enum):
    """Available clustering pipelines for comparison."""

    OLD_RULE_FIRST = "old_rule_first"  # Original rule-based archetype + basic clustering
    NEW_BEHAVIOR_ONLY = "new_behavior_only"  # Transformed behavioral features
    NEW_GRAPH_ONLY = "new_graph_only"  # Node2Vec embeddings only
    NEW_COMBINED = "new_combined"  # Behavioral + reduced Node2Vec


@dataclass
class ArchetypeDistribution:
    """Distribution of archetype assignments."""

    counts: dict[Archetype, int]
    percentages: dict[Archetype, float]
    unknown_percentage: float
    multi_archetype_percentage: float  # Wallets with secondary archetypes
    mean_confidence: float
    confidence_std: float


@dataclass
class ClusterStability:
    """Stability metrics from bootstrap validation."""

    adjusted_rand_index_mean: float
    adjusted_rand_index_std: float
    normalized_mutual_info_mean: float
    normalized_mutual_info_std: float
    cluster_persistence_rate: float  # % of points consistently clustered together
    n_bootstrap_samples: int


@dataclass
class PipelineComparisonResult:
    """Result of comparing clustering pipelines."""

    pipeline: ClusteringPipeline
    diagnostics: ClusterDiagnostics
    archetype_distribution: ArchetypeDistribution
    stability: Optional[ClusterStability]

    # Key metrics
    n_clusters: int
    noise_percentage: float
    silhouette_score: Optional[float]

    # Interpretability notes
    interpretability_notes: list[str] = field(default_factory=list)

    # Metadata
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sample_size: int = 0
    n_features: int = 0

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "pipeline": self.pipeline.value,
            "n_clusters": self.n_clusters,
            "noise_percentage": self.noise_percentage,
            "silhouette_score": self.silhouette_score,
            "archetype_distribution": {
                "counts": {k.value: v for k, v in self.archetype_distribution.counts.items()},
                "percentages": {k.value: v for k, v in self.archetype_distribution.percentages.items()},
                "unknown_percentage": self.archetype_distribution.unknown_percentage,
                "multi_archetype_percentage": self.archetype_distribution.multi_archetype_percentage,
                "mean_confidence": self.archetype_distribution.mean_confidence,
            },
            "stability": {
                "ari_mean": self.stability.adjusted_rand_index_mean,
                "ari_std": self.stability.adjusted_rand_index_std,
                "nmi_mean": self.stability.normalized_mutual_info_mean,
                "persistence_rate": self.stability.cluster_persistence_rate,
            } if self.stability else None,
            "interpretability_notes": self.interpretability_notes,
            "sample_size": self.sample_size,
            "n_features": self.n_features,
            "computed_at": self.computed_at.isoformat(),
        }


@dataclass
class FullComparisonReport:
    """Complete comparison across all pipelines."""

    results: dict[ClusteringPipeline, PipelineComparisonResult]
    best_pipeline: ClusteringPipeline
    recommendation: str
    decision_rationale: list[str]

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "results": {k.value: v.to_dict() for k, v in self.results.items()},
            "best_pipeline": self.best_pipeline.value,
            "recommendation": self.recommendation,
            "decision_rationale": self.decision_rationale,
        }


class ClusteringValidator:
    """
    Validates clustering pipeline upgrades.

    Compares:
    - Old rule-first pipeline
    - New transformed behaviour-only pipeline
    - New graph-only Node2Vec pipeline
    - New combined behaviour + graph pipeline

    Measures:
    - Cluster quality (silhouette, noise %)
    - Stability (bootstrap ARI, NMI)
    - Archetype distribution
    - Interpretability
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        n_bootstrap: int = 20,
        random_state: int = 42,
    ):
        """
        Initialize validator.

        Args:
            min_cluster_size: HDBSCAN min_cluster_size
            n_bootstrap: Number of bootstrap samples for stability
            random_state: Random seed for reproducibility
        """
        self.min_cluster_size = min_cluster_size
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def compare_pipelines(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
        wallet_vectors: list[WalletFeatureVector],
        graph_embeddings: Optional[NDArray[np.float64]] = None,
    ) -> FullComparisonReport:
        """
        Compare all clustering pipelines.

        Args:
            features: (n_samples, n_features) raw behavioral features
            feature_names: Feature names
            wallet_vectors: WalletFeatureVector for each sample
            graph_embeddings: Optional (n_samples, n_dims) Node2Vec embeddings

        Returns:
            FullComparisonReport with all comparisons and recommendation
        """
        logger.info(
            "starting_pipeline_comparison",
            n_samples=features.shape[0],
            n_features=features.shape[1],
            has_embeddings=graph_embeddings is not None,
        )

        results: dict[ClusteringPipeline, PipelineComparisonResult] = {}

        # 1. Old rule-first pipeline (baseline)
        logger.info("evaluating_pipeline", pipeline="old_rule_first")
        results[ClusteringPipeline.OLD_RULE_FIRST] = self._evaluate_old_pipeline(
            features, feature_names, wallet_vectors
        )

        # 2. New behavior-only pipeline
        logger.info("evaluating_pipeline", pipeline="new_behavior_only")
        results[ClusteringPipeline.NEW_BEHAVIOR_ONLY] = self._evaluate_behavior_pipeline(
            features, feature_names, wallet_vectors
        )

        # 3. Graph-only pipeline (if embeddings available)
        if graph_embeddings is not None:
            logger.info("evaluating_pipeline", pipeline="new_graph_only")
            results[ClusteringPipeline.NEW_GRAPH_ONLY] = self._evaluate_graph_pipeline(
                graph_embeddings, wallet_vectors
            )

            # 4. Combined pipeline
            logger.info("evaluating_pipeline", pipeline="new_combined")
            results[ClusteringPipeline.NEW_COMBINED] = self._evaluate_combined_pipeline(
                features, feature_names, graph_embeddings, wallet_vectors
            )

        # Determine best pipeline
        best_pipeline, recommendation, rationale = self._make_recommendation(results)

        logger.info(
            "pipeline_comparison_complete",
            best_pipeline=best_pipeline.value,
            recommendation=recommendation[:100],
        )

        return FullComparisonReport(
            results=results,
            best_pipeline=best_pipeline,
            recommendation=recommendation,
            decision_rationale=rationale,
        )

    def _evaluate_old_pipeline(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
        wallet_vectors: list[WalletFeatureVector],
    ) -> PipelineComparisonResult:
        """Evaluate old rule-first clustering pipeline."""
        from sklearn.preprocessing import StandardScaler

        # Old pipeline: StandardScaler + fillna(0) + HDBSCAN
        features_clean = np.nan_to_num(features, nan=0.0)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)

        # Run clustering
        diagnostics = HDBSCANDiagnostics(min_cluster_size=self.min_cluster_size)
        result = diagnostics.fit(features_scaled)

        # Compute archetype distribution (old rule-based)
        archetype_dist = self._compute_archetype_distribution(
            result.labels, wallet_vectors, use_multi_score=False
        )

        # Bootstrap stability
        stability = self._compute_stability(features_scaled)

        # Interpretability notes
        notes = self._assess_interpretability(result, archetype_dist, "old_rule_first")

        return PipelineComparisonResult(
            pipeline=ClusteringPipeline.OLD_RULE_FIRST,
            diagnostics=result,
            archetype_distribution=archetype_dist,
            stability=stability,
            n_clusters=result.n_clusters,
            noise_percentage=result.noise_percentage,
            silhouette_score=result.silhouette_score,
            interpretability_notes=notes,
            sample_size=features.shape[0],
            n_features=features.shape[1],
        )

    def _evaluate_behavior_pipeline(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
        wallet_vectors: list[WalletFeatureVector],
    ) -> PipelineComparisonResult:
        """Evaluate new transformed behavior-only pipeline."""
        # New pipeline: FeatureTransformer + missingness indicators
        transformer = FeatureTransformer()
        features_transformed, indicators = transformer.fit_transform(
            features, feature_names, impute=True
        )

        # Add missingness indicators as features
        if indicators.missing_flags:
            indicator_array = indicators.to_array()
            if indicator_array.size > 0:
                features_final = np.hstack([features_transformed, indicator_array])
            else:
                features_final = features_transformed
        else:
            features_final = features_transformed

        # Run clustering
        diagnostics = HDBSCANDiagnostics(min_cluster_size=self.min_cluster_size)
        result = diagnostics.fit(features_final)

        # Compute archetype distribution (new multi-score)
        archetype_dist = self._compute_archetype_distribution(
            result.labels, wallet_vectors, use_multi_score=True,
            probabilities=result.probabilities
        )

        # Bootstrap stability
        stability = self._compute_stability(features_final)

        # Interpretability notes
        notes = self._assess_interpretability(result, archetype_dist, "new_behavior_only")

        return PipelineComparisonResult(
            pipeline=ClusteringPipeline.NEW_BEHAVIOR_ONLY,
            diagnostics=result,
            archetype_distribution=archetype_dist,
            stability=stability,
            n_clusters=result.n_clusters,
            noise_percentage=result.noise_percentage,
            silhouette_score=result.silhouette_score,
            interpretability_notes=notes,
            sample_size=features.shape[0],
            n_features=features_final.shape[1],
        )

    def _evaluate_graph_pipeline(
        self,
        embeddings: NDArray[np.float64],
        wallet_vectors: list[WalletFeatureVector],
    ) -> PipelineComparisonResult:
        """Evaluate graph-only Node2Vec pipeline."""
        from sklearn.preprocessing import StandardScaler

        # Scale embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # Run clustering
        diagnostics = HDBSCANDiagnostics(min_cluster_size=self.min_cluster_size)
        result = diagnostics.fit(embeddings_scaled)

        # Compute archetype distribution
        archetype_dist = self._compute_archetype_distribution(
            result.labels, wallet_vectors, use_multi_score=True,
            probabilities=result.probabilities
        )

        # Bootstrap stability
        stability = self._compute_stability(embeddings_scaled)

        # Interpretability notes
        notes = self._assess_interpretability(result, archetype_dist, "new_graph_only")
        notes.append("Graph embeddings may capture structural patterns not visible in behavior")
        notes.append("Lower interpretability: embeddings are not directly explainable")

        return PipelineComparisonResult(
            pipeline=ClusteringPipeline.NEW_GRAPH_ONLY,
            diagnostics=result,
            archetype_distribution=archetype_dist,
            stability=stability,
            n_clusters=result.n_clusters,
            noise_percentage=result.noise_percentage,
            silhouette_score=result.silhouette_score,
            interpretability_notes=notes,
            sample_size=embeddings.shape[0],
            n_features=embeddings.shape[1],
        )

    def _evaluate_combined_pipeline(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
        embeddings: NDArray[np.float64],
        wallet_vectors: list[WalletFeatureVector],
        behavior_weight: float = 0.7,
        graph_weight: float = 0.3,
    ) -> PipelineComparisonResult:
        """Evaluate combined behavior + graph pipeline."""
        from sklearn.preprocessing import StandardScaler

        # Transform behavioral features
        transformer = FeatureTransformer()
        features_transformed, indicators = transformer.fit_transform(
            features, feature_names, impute=True
        )

        # Scale both
        behavior_scaler = StandardScaler()
        behavior_scaled = behavior_scaler.fit_transform(features_transformed)

        graph_scaler = StandardScaler()
        graph_scaled = graph_scaler.fit_transform(embeddings)

        # Combine with weights
        combined = np.hstack([
            behavior_scaled * behavior_weight,
            graph_scaled * graph_weight,
        ])

        # Run clustering
        diagnostics = HDBSCANDiagnostics(min_cluster_size=self.min_cluster_size)
        result = diagnostics.fit(combined)

        # Compute archetype distribution
        archetype_dist = self._compute_archetype_distribution(
            result.labels, wallet_vectors, use_multi_score=True,
            probabilities=result.probabilities
        )

        # Bootstrap stability
        stability = self._compute_stability(combined)

        # Interpretability notes
        notes = self._assess_interpretability(result, archetype_dist, "new_combined")
        notes.append(f"Combined: {behavior_weight:.0%} behavior + {graph_weight:.0%} graph")

        return PipelineComparisonResult(
            pipeline=ClusteringPipeline.NEW_COMBINED,
            diagnostics=result,
            archetype_distribution=archetype_dist,
            stability=stability,
            n_clusters=result.n_clusters,
            noise_percentage=result.noise_percentage,
            silhouette_score=result.silhouette_score,
            interpretability_notes=notes,
            sample_size=features.shape[0],
            n_features=combined.shape[1],
        )

    def _compute_archetype_distribution(
        self,
        labels: NDArray[np.int32],
        wallet_vectors: list[WalletFeatureVector],
        use_multi_score: bool = False,
        probabilities: Optional[NDArray[np.float64]] = None,
    ) -> ArchetypeDistribution:
        """Compute archetype distribution from clustering results."""
        counts: dict[Archetype, int] = {a: 0 for a in Archetype}
        confidences = []
        multi_archetype_count = 0

        for i, wv in enumerate(wallet_vectors):
            cluster_id = labels[i]
            cluster_prob = probabilities[i] if probabilities is not None else 0.5

            if use_multi_score:
                # Use new multi-score assignment
                cluster_status = "noise" if cluster_id == -1 else "core"
                assignment = assign_archetype_multi_score(
                    wv,
                    cluster_status=cluster_status,
                    cluster_confidence_adj=cluster_prob,
                )
                counts[assignment.primary_archetype] += 1
                confidences.append(assignment.adjusted_confidence)

                if len(assignment.secondary_archetypes) > 0:
                    multi_archetype_count += 1
            else:
                # Simple rule-based assignment (legacy)
                archetype = self._simple_archetype_assignment(wv)
                counts[archetype] += 1
                confidences.append(0.7)  # Fixed confidence for old method

        total = len(wallet_vectors)
        percentages = {a: c / total * 100 for a, c in counts.items()}
        unknown_pct = percentages.get(Archetype.UNKNOWN, 0.0)
        multi_pct = multi_archetype_count / total * 100 if total > 0 else 0.0

        return ArchetypeDistribution(
            counts=counts,
            percentages=percentages,
            unknown_percentage=unknown_pct,
            multi_archetype_percentage=multi_pct,
            mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
            confidence_std=float(np.std(confidences)) if confidences else 0.0,
        )

    def _simple_archetype_assignment(self, wv: WalletFeatureVector) -> Archetype:
        """Simple rule-based archetype assignment (legacy method)."""
        # Sniper: early entry + short hold + high turnover
        if wv.entry_time_relative < 0.1 and wv.holding_duration < 3 and wv.trade_count > 5:
            return Archetype.SNIPER

        # Dormant whale: large share + few trades + long hold
        if wv.share >= 0.01 and wv.trade_count <= 3 and wv.holding_duration >= 14:
            return Archetype.DORMANT_WHALE

        # Long-term accumulator: growing balance + long hold
        if wv.delta_balance_7d > 0 and wv.holding_duration >= 7:
            return Archetype.LONG_TERM_ACCUMULATOR

        # Liquidity actor: LP interactions
        if wv.lp_interaction_ratio > 0.3:
            return Archetype.LIQUIDITY_ACTOR

        # Coordinated cluster: shared funders
        if wv.shared_funder_count >= 3:
            return Archetype.COORDINATED_CLUSTER

        return Archetype.UNKNOWN

    def _compute_stability(
        self,
        features: NDArray[np.float64],
    ) -> ClusterStability:
        """Compute clustering stability via bootstrap."""
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        n_samples = features.shape[0]
        ari_scores = []
        nmi_scores = []

        # Get reference clustering
        ref_clusterer = HDBSCANDiagnostics(min_cluster_size=self.min_cluster_size)
        ref_result = ref_clusterer.fit(features)
        ref_labels = ref_result.labels

        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            indices = self._rng.choice(n_samples, size=n_samples, replace=True)
            bootstrap_features = features[indices]

            # Cluster bootstrap sample
            clusterer = HDBSCANDiagnostics(min_cluster_size=self.min_cluster_size)
            result = clusterer.fit(bootstrap_features)

            # Compare to reference (on common indices)
            common_mask = np.isin(np.arange(n_samples), indices)
            if common_mask.sum() > 0:
                ref_subset = ref_labels[common_mask]
                # Map bootstrap labels back
                bootstrap_labels = result.labels[:common_mask.sum()]

                try:
                    ari = adjusted_rand_score(ref_subset, bootstrap_labels)
                    nmi = normalized_mutual_info_score(ref_subset, bootstrap_labels)
                    ari_scores.append(ari)
                    nmi_scores.append(nmi)
                except Exception:
                    pass

        # Compute persistence rate
        persistence_rate = np.mean(ari_scores) if ari_scores else 0.0

        return ClusterStability(
            adjusted_rand_index_mean=float(np.mean(ari_scores)) if ari_scores else 0.0,
            adjusted_rand_index_std=float(np.std(ari_scores)) if ari_scores else 0.0,
            normalized_mutual_info_mean=float(np.mean(nmi_scores)) if nmi_scores else 0.0,
            normalized_mutual_info_std=float(np.std(nmi_scores)) if nmi_scores else 0.0,
            cluster_persistence_rate=persistence_rate,
            n_bootstrap_samples=self.n_bootstrap,
        )

    def _assess_interpretability(
        self,
        diagnostics: ClusterDiagnostics,
        archetype_dist: ArchetypeDistribution,
        pipeline_name: str,
    ) -> list[str]:
        """Assess interpretability of clustering results."""
        notes = []

        # Cluster count assessment
        if diagnostics.n_clusters == 0:
            notes.append("WARNING: No clusters found - all points as noise")
        elif diagnostics.n_clusters == 1:
            notes.append("WARNING: Only 1 cluster - may lack discrimination")
        elif diagnostics.n_clusters > 20:
            notes.append("WARNING: Many clusters (>20) - may be over-fragmented")

        # Noise assessment
        if diagnostics.noise_percentage > 50:
            notes.append(f"HIGH NOISE: {diagnostics.noise_percentage:.1f}% - many unclassified")
        elif diagnostics.noise_percentage > 30:
            notes.append(f"MODERATE NOISE: {diagnostics.noise_percentage:.1f}%")

        # Unknown archetype assessment
        if archetype_dist.unknown_percentage > 40:
            notes.append(f"HIGH UNKNOWN: {archetype_dist.unknown_percentage:.1f}% wallets unclassified")

        # Silhouette assessment
        if diagnostics.silhouette_score is not None:
            if diagnostics.silhouette_score < 0:
                notes.append(f"POOR SEPARATION: Negative silhouette ({diagnostics.silhouette_score:.3f})")
            elif diagnostics.silhouette_score < 0.25:
                notes.append(f"WEAK SEPARATION: Low silhouette ({diagnostics.silhouette_score:.3f})")
            elif diagnostics.silhouette_score > 0.5:
                notes.append(f"GOOD SEPARATION: High silhouette ({diagnostics.silhouette_score:.3f})")

        # Multi-archetype assessment
        if archetype_dist.multi_archetype_percentage > 30:
            notes.append(f"MULTI-ARCHETYPE: {archetype_dist.multi_archetype_percentage:.1f}% have secondary labels")

        return notes

    def _make_recommendation(
        self,
        results: dict[ClusteringPipeline, PipelineComparisonResult],
    ) -> tuple[ClusteringPipeline, str, list[str]]:
        """Make deployment recommendation based on comparison results."""
        rationale = []

        # Score each pipeline
        scores: dict[ClusteringPipeline, float] = {}

        for pipeline, result in results.items():
            score = 0.0

            # Silhouette (max 30 points)
            if result.silhouette_score is not None and result.silhouette_score > 0:
                score += min(30, result.silhouette_score * 60)

            # Low noise (max 25 points)
            noise_penalty = result.noise_percentage / 100
            score += max(0, 25 * (1 - noise_penalty))

            # Stability (max 25 points)
            if result.stability:
                score += min(25, result.stability.adjusted_rand_index_mean * 25)

            # Low UNKNOWN (max 10 points)
            unknown_penalty = result.archetype_distribution.unknown_percentage / 100
            score += max(0, 10 * (1 - unknown_penalty))

            # Confidence (max 10 points)
            score += result.archetype_distribution.mean_confidence * 10

            scores[pipeline] = score
            rationale.append(f"{pipeline.value}: score={score:.1f}")

        # Find best
        best_pipeline = max(scores, key=scores.get)
        best_score = scores[best_pipeline]

        # Compare to baseline
        baseline_score = scores.get(ClusteringPipeline.OLD_RULE_FIRST, 0)
        improvement = best_score - baseline_score

        if improvement > 5:
            recommendation = f"DEPLOY {best_pipeline.value}: Improvement of {improvement:.1f} points over baseline"
        elif improvement > 0:
            recommendation = f"CAUTIOUS DEPLOY {best_pipeline.value}: Marginal improvement of {improvement:.1f} points"
        else:
            recommendation = f"KEEP BASELINE: No significant improvement from {best_pipeline.value}"
            best_pipeline = ClusteringPipeline.OLD_RULE_FIRST

        rationale.append(f"Best: {best_pipeline.value} with score {best_score:.1f}")
        rationale.append(f"Improvement over baseline: {improvement:.1f}")

        return best_pipeline, recommendation, rationale
