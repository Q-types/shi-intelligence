"""
Cluster Semantics Validation.

Investigates whether clusters represent real behavioural structure
or geometric artifacts. Addresses key concerns:
- Silhouette vs stability contradiction
- Coordinated cluster dominance
- Feature contribution analysis
- Known-wallet validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from collections import Counter

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import structlog

from ...clustering.archetypes import (
    Archetype,
    WalletFeatureVector,
    assign_archetype_multi_score,
)

logger = structlog.get_logger()


@dataclass
class SilhouetteStabilityAnalysis:
    """Analysis of silhouette vs stability contradiction."""

    silhouette_score: float
    ari_mean: float
    nmi_mean: float

    # Diagnosis
    contradiction_detected: bool
    likely_causes: list[str]

    # Feature-level analysis
    per_feature_silhouette: dict[str, float]
    dominant_features: list[str]  # Features driving separation
    feature_variance_ratios: dict[str, float]  # Post-transform variance

    # Cluster geometry
    n_clusters: int
    cluster_sizes: list[int]
    size_imbalance_ratio: float  # max/min cluster size
    inter_cluster_distances: list[float]
    intra_cluster_distances: list[float]

    # Binary partition check
    is_effectively_binary: bool
    binary_partition_evidence: list[str]

    def to_dict(self) -> dict:
        return {
            "silhouette_score": self.silhouette_score,
            "ari_mean": self.ari_mean,
            "nmi_mean": self.nmi_mean,
            "contradiction_detected": self.contradiction_detected,
            "likely_causes": self.likely_causes,
            "dominant_features": self.dominant_features,
            "n_clusters": self.n_clusters,
            "cluster_sizes": self.cluster_sizes,
            "size_imbalance_ratio": self.size_imbalance_ratio,
            "is_effectively_binary": self.is_effectively_binary,
            "binary_partition_evidence": self.binary_partition_evidence,
        }


@dataclass
class CoordinationThresholdAnalysis:
    """Analysis of coordinated_cluster classification dominance."""

    total_wallets: int
    coordinated_count: int
    coordinated_percentage: float

    # Threshold analysis
    shared_funder_threshold: int
    wallets_meeting_threshold: int
    threshold_percentage: float

    # Distribution of shared_funder_count
    shared_funder_distribution: dict[int, int]  # count -> n_wallets
    median_shared_funders: float
    mean_shared_funders: float

    # Override analysis
    wallets_overridden_to_coordinated: int  # Would be different archetype without override
    override_sources: dict[str, int]  # original_archetype -> count

    # Recommendations
    recommended_threshold: int
    threshold_rationale: str

    def to_dict(self) -> dict:
        return {
            "total_wallets": self.total_wallets,
            "coordinated_count": self.coordinated_count,
            "coordinated_percentage": self.coordinated_percentage,
            "shared_funder_threshold": self.shared_funder_threshold,
            "wallets_meeting_threshold": self.wallets_meeting_threshold,
            "threshold_percentage": self.threshold_percentage,
            "shared_funder_distribution": self.shared_funder_distribution,
            "median_shared_funders": self.median_shared_funders,
            "mean_shared_funders": self.mean_shared_funders,
            "wallets_overridden_to_coordinated": self.wallets_overridden_to_coordinated,
            "override_sources": self.override_sources,
            "recommended_threshold": self.recommended_threshold,
            "threshold_rationale": self.threshold_rationale,
        }


@dataclass
class ClusterProfile:
    """Human-readable profile for a single cluster."""

    cluster_id: int
    size: int
    percentage: float

    # Feature statistics
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    feature_medians: dict[str, float]

    # Dominant characteristics
    dominant_features: list[str]  # Features significantly above/below mean
    characteristic_pattern: str  # Human-readable description

    # Archetype composition
    archetype_distribution: dict[str, int]
    primary_archetype: str
    archetype_purity: float  # % of dominant archetype

    # Temporal characteristics
    mean_entry_time: float
    mean_hold_duration: float
    entry_time_spread: float

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "percentage": self.percentage,
            "feature_means": self.feature_means,
            "dominant_features": self.dominant_features,
            "characteristic_pattern": self.characteristic_pattern,
            "archetype_distribution": self.archetype_distribution,
            "primary_archetype": self.primary_archetype,
            "archetype_purity": self.archetype_purity,
            "mean_entry_time": self.mean_entry_time,
            "mean_hold_duration": self.mean_hold_duration,
        }


@dataclass
class FeatureContributionAnalysis:
    """Analysis of which features actually drive clustering."""

    # Per-feature importance
    feature_silhouette_contributions: dict[str, float]
    feature_importance_permutation: dict[str, float]

    # Feature group analysis
    temporal_only_silhouette: float
    graph_only_silhouette: float
    trading_only_silhouette: float

    # Verdict
    primary_driver: str  # "temporal", "graph", "trading", "mixed"
    temporal_contribution_pct: float
    graph_contribution_pct: float

    # Strategic insight
    is_timing_engine: bool  # True if temporal dominates
    graph_adds_value: bool  # True if graph meaningfully contributes

    def to_dict(self) -> dict:
        return {
            "feature_silhouette_contributions": self.feature_silhouette_contributions,
            "temporal_only_silhouette": self.temporal_only_silhouette,
            "graph_only_silhouette": self.graph_only_silhouette,
            "trading_only_silhouette": self.trading_only_silhouette,
            "primary_driver": self.primary_driver,
            "temporal_contribution_pct": self.temporal_contribution_pct,
            "graph_contribution_pct": self.graph_contribution_pct,
            "is_timing_engine": self.is_timing_engine,
            "graph_adds_value": self.graph_adds_value,
        }


@dataclass
class ClusterSemanticsReport:
    """Complete cluster semantics validation report."""

    silhouette_stability: SilhouetteStabilityAnalysis
    coordination_threshold: CoordinationThresholdAnalysis
    cluster_profiles: list[ClusterProfile]
    feature_contribution: FeatureContributionAnalysis

    # Overall assessment
    clusters_are_semantic: bool  # True if clusters have real meaning
    primary_concerns: list[str]
    recommendations: list[str]

    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "silhouette_stability": self.silhouette_stability.to_dict(),
            "coordination_threshold": self.coordination_threshold.to_dict(),
            "cluster_profiles": [p.to_dict() for p in self.cluster_profiles],
            "feature_contribution": self.feature_contribution.to_dict(),
            "clusters_are_semantic": self.clusters_are_semantic,
            "primary_concerns": self.primary_concerns,
            "recommendations": self.recommendations,
            "computed_at": self.computed_at.isoformat(),
        }


# Feature groups for analysis
TEMPORAL_FEATURES = [
    "entry_time_relative", "holding_duration", "position_volatility"
]
GRAPH_FEATURES = [
    "in_degree", "out_degree", "eigenvector_centrality", "shared_funder_count",
    "total_funding_received", "largest_funder_share", "funding_hhi"
]
TRADING_FEATURES = [
    "trade_count", "burstiness", "swap_frequency", "lp_interaction_ratio",
    "delta_balance_7d", "delta_balance_30d"
]
DISTRIBUTION_FEATURES = ["balance", "share", "rank"]


class ClusterSemanticsAnalyzer:
    """
    Analyzes cluster semantic validity.

    Determines whether clusters represent real behavioural patterns
    or are artifacts of feature engineering/scaling.
    """

    def __init__(self, coordination_threshold: int = 2):
        """
        Initialize analyzer.

        Args:
            coordination_threshold: Current shared_funder_count threshold
        """
        self.coordination_threshold = coordination_threshold

    def analyze(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
        labels: NDArray[np.int32],
        wallet_vectors: list[WalletFeatureVector],
        silhouette_score: float,
        ari_mean: float,
        nmi_mean: float,
    ) -> ClusterSemanticsReport:
        """
        Run complete cluster semantics analysis.

        Args:
            features: (n_samples, n_features) feature array
            feature_names: Feature names
            labels: Cluster labels from HDBSCAN
            wallet_vectors: Original wallet feature vectors
            silhouette_score: Reported silhouette score
            ari_mean: Bootstrap ARI mean
            nmi_mean: Bootstrap NMI mean

        Returns:
            ClusterSemanticsReport with findings
        """
        logger.info(
            "starting_cluster_semantics_analysis",
            n_samples=len(labels),
            n_clusters=len(set(labels)) - (1 if -1 in labels else 0),
            silhouette=silhouette_score,
            ari=ari_mean,
        )

        # 1. Silhouette vs Stability contradiction
        silhouette_analysis = self._analyze_silhouette_stability(
            features, feature_names, labels, silhouette_score, ari_mean, nmi_mean
        )

        # 2. Coordination threshold analysis
        coordination_analysis = self._analyze_coordination_threshold(
            wallet_vectors, labels
        )

        # 3. Cluster profiles
        cluster_profiles = self._create_cluster_profiles(
            features, feature_names, labels, wallet_vectors
        )

        # 4. Feature contribution analysis
        feature_contribution = self._analyze_feature_contribution(
            features, feature_names, labels
        )

        # 5. Overall assessment
        concerns, recommendations, is_semantic = self._generate_assessment(
            silhouette_analysis,
            coordination_analysis,
            feature_contribution,
        )

        logger.info(
            "cluster_semantics_analysis_complete",
            clusters_are_semantic=is_semantic,
            n_concerns=len(concerns),
        )

        return ClusterSemanticsReport(
            silhouette_stability=silhouette_analysis,
            coordination_threshold=coordination_analysis,
            cluster_profiles=cluster_profiles,
            feature_contribution=feature_contribution,
            clusters_are_semantic=is_semantic,
            primary_concerns=concerns,
            recommendations=recommendations,
        )

    def _analyze_silhouette_stability(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
        labels: NDArray[np.int32],
        silhouette_score: float,
        ari_mean: float,
        nmi_mean: float,
    ) -> SilhouetteStabilityAnalysis:
        """Analyze the silhouette vs stability contradiction."""
        from sklearn.metrics import silhouette_samples

        # Detect contradiction
        # High silhouette (>0.7) with low stability (<0.1) is contradictory
        contradiction = silhouette_score > 0.7 and (ari_mean < 0.1 or nmi_mean < 0.1)

        likely_causes = []
        if contradiction:
            if silhouette_score > 0.95:
                likely_causes.append("EXTREME_SEPARATION: Silhouette >0.95 suggests artificial stretching")
            if ari_mean < 0.05:
                likely_causes.append("UNSTABLE_MEMBERSHIP: ARI <0.05 means cluster assignments change drastically under resampling")

        # Per-feature silhouette contribution
        # Compute silhouette for each feature independently
        per_feature_sil = {}
        valid_mask = labels >= 0
        if valid_mask.sum() > 10 and len(set(labels[valid_mask])) > 1:
            for i, name in enumerate(feature_names):
                try:
                    feat_1d = features[valid_mask, i:i+1]
                    if feat_1d.std() > 1e-10:
                        sil = silhouette_samples(feat_1d, labels[valid_mask])
                        per_feature_sil[name] = float(np.mean(sil))
                except Exception:
                    per_feature_sil[name] = 0.0

        # Find dominant features (top contributors)
        sorted_features = sorted(per_feature_sil.items(), key=lambda x: x[1], reverse=True)
        dominant = [f[0] for f in sorted_features[:5] if f[1] > 0.3]

        if len(dominant) <= 2 and contradiction:
            likely_causes.append(f"FEW_DOMINANT_FEATURES: Only {len(dominant)} features drive separation")

        # Feature variance ratios
        variance_ratios = {}
        for i, name in enumerate(feature_names):
            var = np.nanvar(features[:, i])
            variance_ratios[name] = float(var) if not np.isnan(var) else 0.0

        # Check for extreme variance (possible over-scaling)
        high_var_features = [n for n, v in variance_ratios.items() if v > 10]
        if high_var_features and contradiction:
            likely_causes.append(f"HIGH_VARIANCE_FEATURES: {len(high_var_features)} features have variance >10")

        # Cluster geometry
        unique_labels = [l for l in set(labels) if l >= 0]
        n_clusters = len(unique_labels)
        cluster_sizes = [int((labels == l).sum()) for l in unique_labels]

        size_imbalance = max(cluster_sizes) / min(cluster_sizes) if cluster_sizes and min(cluster_sizes) > 0 else 0

        if size_imbalance > 10 and contradiction:
            likely_causes.append(f"SIZE_IMBALANCE: Largest cluster {size_imbalance:.1f}x smallest")

        # Inter and intra cluster distances
        inter_distances = []
        intra_distances = []

        for label in unique_labels:
            mask = labels == label
            cluster_points = features[mask]
            if len(cluster_points) > 1:
                # Intra-cluster: mean pairwise distance within cluster
                centroid = cluster_points.mean(axis=0)
                dists = np.linalg.norm(cluster_points - centroid, axis=1)
                intra_distances.append(float(np.mean(dists)))

        # Inter-cluster: distance between centroids
        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroids.append(features[mask].mean(axis=0))

        for i, c1 in enumerate(centroids):
            for c2 in centroids[i+1:]:
                inter_distances.append(float(np.linalg.norm(c1 - c2)))

        # Binary partition check
        is_binary = n_clusters == 2
        binary_evidence = []

        if is_binary:
            binary_evidence.append("ONLY_2_CLUSTERS: Effectively binary partitioning")
            if cluster_sizes:
                dominant_pct = max(cluster_sizes) / sum(cluster_sizes) * 100
                if dominant_pct > 55:
                    binary_evidence.append(f"DOMINANT_CLUSTER: One cluster has {dominant_pct:.1f}%")

        if is_binary and contradiction:
            likely_causes.append("BINARY_PARTITION: Only 2 clusters may artificially inflate silhouette")

        return SilhouetteStabilityAnalysis(
            silhouette_score=silhouette_score,
            ari_mean=ari_mean,
            nmi_mean=nmi_mean,
            contradiction_detected=contradiction,
            likely_causes=likely_causes,
            per_feature_silhouette=per_feature_sil,
            dominant_features=dominant,
            feature_variance_ratios=variance_ratios,
            n_clusters=n_clusters,
            cluster_sizes=cluster_sizes,
            size_imbalance_ratio=size_imbalance,
            inter_cluster_distances=inter_distances,
            intra_cluster_distances=intra_distances,
            is_effectively_binary=is_binary,
            binary_partition_evidence=binary_evidence,
        )

    def _analyze_coordination_threshold(
        self,
        wallet_vectors: list[WalletFeatureVector],
        labels: NDArray[np.int32],
    ) -> CoordinationThresholdAnalysis:
        """Analyze coordinated_cluster classification dominance."""
        total = len(wallet_vectors)

        # Count shared_funder distribution
        shared_funder_counts = [wv.shared_funder_count for wv in wallet_vectors]
        distribution = Counter(shared_funder_counts)

        median_sf = float(np.median(shared_funder_counts))
        mean_sf = float(np.mean(shared_funder_counts))

        # How many meet current threshold
        meeting_threshold = sum(1 for c in shared_funder_counts if c >= self.coordination_threshold)
        threshold_pct = meeting_threshold / total * 100

        # Classify with and without coordination override
        coordinated_count = 0
        override_sources: dict[str, int] = {}
        overridden_count = 0

        for i, wv in enumerate(wallet_vectors):
            # Get assignment WITH override
            cluster_status = "noise" if labels[i] == -1 else "core"
            assignment = assign_archetype_multi_score(
                wv,
                cluster_status=cluster_status,
                cluster_confidence_adj=0.5,
            )

            if assignment.primary_archetype == Archetype.COORDINATED_CLUSTER:
                coordinated_count += 1

                # Check if this would have been different without coordination logic
                # by temporarily checking if shared_funder_count drove the decision
                if wv.shared_funder_count >= self.coordination_threshold:
                    # This might be an override case
                    # Get what the score-based assignment would be
                    scores = assignment.all_scores
                    if scores:
                        # Find highest non-coordinated score
                        non_coord_scores = {
                            a: s for a, s in scores.items()
                            if a != Archetype.COORDINATED_CLUSTER
                        }
                        if non_coord_scores:
                            would_be = max(non_coord_scores, key=non_coord_scores.get)
                            if non_coord_scores[would_be] > 0.3:  # Meaningful alternative
                                overridden_count += 1
                                key = would_be.value if hasattr(would_be, 'value') else str(would_be)
                                override_sources[key] = override_sources.get(key, 0) + 1

        coord_pct = coordinated_count / total * 100

        # Recommend threshold based on distribution
        # Find threshold that would classify ~15-25% as coordinated (reasonable range)
        recommended = self.coordination_threshold
        rationale = "Current threshold seems appropriate"

        if coord_pct > 40:
            # Too many coordinated - raise threshold
            for thresh in range(3, 10):
                would_meet = sum(1 for c in shared_funder_counts if c >= thresh)
                would_pct = would_meet / total * 100
                if would_pct < 30:
                    recommended = thresh
                    rationale = f"Raise to {thresh} to reduce coordinated from {coord_pct:.1f}% to ~{would_pct:.1f}%"
                    break
        elif coord_pct < 5:
            # Too few - lower threshold
            recommended = max(1, self.coordination_threshold - 1)
            rationale = f"Consider lowering threshold - only {coord_pct:.1f}% coordinated"

        return CoordinationThresholdAnalysis(
            total_wallets=total,
            coordinated_count=coordinated_count,
            coordinated_percentage=coord_pct,
            shared_funder_threshold=self.coordination_threshold,
            wallets_meeting_threshold=meeting_threshold,
            threshold_percentage=threshold_pct,
            shared_funder_distribution=dict(distribution),
            median_shared_funders=median_sf,
            mean_shared_funders=mean_sf,
            wallets_overridden_to_coordinated=overridden_count,
            override_sources=override_sources,
            recommended_threshold=recommended,
            threshold_rationale=rationale,
        )

    def _create_cluster_profiles(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
        labels: NDArray[np.int32],
        wallet_vectors: list[WalletFeatureVector],
    ) -> list[ClusterProfile]:
        """Create human-readable profiles for each cluster."""
        profiles = []
        total = len(labels)
        unique_labels = sorted(set(labels))

        # Global means for comparison
        global_means = {
            name: float(np.nanmean(features[:, i]))
            for i, name in enumerate(feature_names)
        }

        for label in unique_labels:
            if label == -1:
                continue  # Skip noise

            mask = labels == label
            cluster_features = features[mask]
            cluster_wallets = [wv for i, wv in enumerate(wallet_vectors) if mask[i]]
            size = int(mask.sum())
            pct = size / total * 100

            # Feature statistics
            means = {}
            stds = {}
            medians = {}
            for i, name in enumerate(feature_names):
                col = cluster_features[:, i]
                means[name] = float(np.nanmean(col))
                stds[name] = float(np.nanstd(col))
                medians[name] = float(np.nanmedian(col))

            # Dominant features (significantly different from global mean)
            dominant = []
            for name, mean in means.items():
                global_mean = global_means.get(name, 0)
                if global_mean != 0:
                    diff_pct = abs(mean - global_mean) / abs(global_mean) * 100
                    if diff_pct > 50:  # >50% different from global
                        direction = "high" if mean > global_mean else "low"
                        dominant.append(f"{name}:{direction}")

            # Generate characteristic pattern
            pattern = self._describe_cluster_pattern(dominant, means)

            # Archetype distribution within cluster
            archetype_dist: dict[str, int] = {}
            for wv in cluster_wallets:
                assignment = assign_archetype_multi_score(
                    wv, cluster_status="core", cluster_confidence_adj=0.5
                )
                arch = assignment.primary_archetype.value
                archetype_dist[arch] = archetype_dist.get(arch, 0) + 1

            # Primary archetype and purity
            if archetype_dist:
                primary = max(archetype_dist, key=archetype_dist.get)
                purity = archetype_dist[primary] / size * 100
            else:
                primary = "unknown"
                purity = 0.0

            # Temporal characteristics
            entry_times = [means.get("entry_time_relative", 0)]
            hold_durations = [means.get("holding_duration", 0)]

            profiles.append(ClusterProfile(
                cluster_id=label,
                size=size,
                percentage=pct,
                feature_means=means,
                feature_stds=stds,
                feature_medians=medians,
                dominant_features=dominant[:10],
                characteristic_pattern=pattern,
                archetype_distribution=archetype_dist,
                primary_archetype=primary,
                archetype_purity=purity,
                mean_entry_time=means.get("entry_time_relative", 0),
                mean_hold_duration=means.get("holding_duration", 0),
                entry_time_spread=stds.get("entry_time_relative", 0),
            ))

        return profiles

    def _describe_cluster_pattern(
        self,
        dominant_features: list[str],
        means: dict[str, float],
    ) -> str:
        """Generate human-readable cluster description."""
        patterns = []

        # Check temporal patterns
        if "holding_duration:high" in dominant_features:
            patterns.append("long-term holders")
        elif "holding_duration:low" in dominant_features:
            patterns.append("short-term traders")

        if "entry_time_relative:low" in dominant_features:
            patterns.append("early entrants")
        elif "entry_time_relative:high" in dominant_features:
            patterns.append("late entrants")

        # Check trading patterns
        if "trade_count:high" in dominant_features:
            patterns.append("high activity")
        if "burstiness:high" in dominant_features:
            patterns.append("bursty trading")

        # Check graph patterns
        if "shared_funder_count:high" in dominant_features:
            patterns.append("connected wallets")
        if "in_degree:high" in dominant_features or "out_degree:high" in dominant_features:
            patterns.append("high connectivity")

        # Check distribution patterns
        if "balance:high" in dominant_features:
            patterns.append("large holders")
        elif "balance:low" in dominant_features:
            patterns.append("small holders")

        if not patterns:
            return "Mixed characteristics - no dominant pattern"

        return ", ".join(patterns)

    def _analyze_feature_contribution(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
        labels: NDArray[np.int32],
    ) -> FeatureContributionAnalysis:
        """Analyze which features drive clustering."""
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans

        valid_mask = labels >= 0
        if valid_mask.sum() < 20 or len(set(labels[valid_mask])) < 2:
            return self._empty_feature_contribution()

        valid_features = features[valid_mask]
        valid_labels = labels[valid_mask]
        n_clusters = len(set(valid_labels))

        # Per-feature silhouette contribution
        feature_contributions = {}
        for i, name in enumerate(feature_names):
            try:
                feat_col = valid_features[:, i:i+1]
                if feat_col.std() > 1e-10:
                    sil = silhouette_score(feat_col, valid_labels)
                    feature_contributions[name] = float(sil)
                else:
                    feature_contributions[name] = 0.0
            except Exception:
                feature_contributions[name] = 0.0

        # Feature group analysis
        def get_feature_indices(group_names):
            return [i for i, n in enumerate(feature_names) if n in group_names]

        temporal_idx = get_feature_indices(TEMPORAL_FEATURES)
        graph_idx = get_feature_indices(GRAPH_FEATURES)
        trading_idx = get_feature_indices(TRADING_FEATURES)

        def compute_group_silhouette(indices):
            if not indices:
                return 0.0
            try:
                group_features = valid_features[:, indices]
                if group_features.std() < 1e-10:
                    return 0.0
                # Use same number of clusters
                kmeans = KMeans(n_clusters=min(n_clusters, len(valid_features) // 10), random_state=42, n_init=10)
                group_labels = kmeans.fit_predict(group_features)
                if len(set(group_labels)) < 2:
                    return 0.0
                return float(silhouette_score(group_features, group_labels))
            except Exception:
                return 0.0

        temporal_sil = compute_group_silhouette(temporal_idx)
        graph_sil = compute_group_silhouette(graph_idx)
        trading_sil = compute_group_silhouette(trading_idx)

        # Determine primary driver
        max_sil = max(temporal_sil, graph_sil, trading_sil)
        if max_sil < 0.1:
            primary_driver = "mixed"
        elif temporal_sil == max_sil:
            primary_driver = "temporal"
        elif graph_sil == max_sil:
            primary_driver = "graph"
        else:
            primary_driver = "trading"

        total_sil = temporal_sil + graph_sil + trading_sil
        if total_sil > 0:
            temporal_pct = temporal_sil / total_sil * 100
            graph_pct = graph_sil / total_sil * 100
        else:
            temporal_pct = 33.3
            graph_pct = 33.3

        # Strategic insights
        is_timing_engine = temporal_pct > 50
        graph_adds_value = graph_sil > 0.1 and graph_pct > 20

        # Permutation importance (simplified)
        permutation_importance = {}
        try:
            base_sil = silhouette_score(valid_features, valid_labels)
            for i, name in enumerate(feature_names):
                permuted = valid_features.copy()
                np.random.shuffle(permuted[:, i])
                try:
                    perm_sil = silhouette_score(permuted, valid_labels)
                    permutation_importance[name] = base_sil - perm_sil
                except Exception:
                    permutation_importance[name] = 0.0
        except Exception:
            permutation_importance = {n: 0.0 for n in feature_names}

        return FeatureContributionAnalysis(
            feature_silhouette_contributions=feature_contributions,
            feature_importance_permutation=permutation_importance,
            temporal_only_silhouette=temporal_sil,
            graph_only_silhouette=graph_sil,
            trading_only_silhouette=trading_sil,
            primary_driver=primary_driver,
            temporal_contribution_pct=temporal_pct,
            graph_contribution_pct=graph_pct,
            is_timing_engine=is_timing_engine,
            graph_adds_value=graph_adds_value,
        )

    def _empty_feature_contribution(self) -> FeatureContributionAnalysis:
        """Return empty feature contribution for edge cases."""
        return FeatureContributionAnalysis(
            feature_silhouette_contributions={},
            feature_importance_permutation={},
            temporal_only_silhouette=0.0,
            graph_only_silhouette=0.0,
            trading_only_silhouette=0.0,
            primary_driver="unknown",
            temporal_contribution_pct=0.0,
            graph_contribution_pct=0.0,
            is_timing_engine=False,
            graph_adds_value=False,
        )

    def _generate_assessment(
        self,
        silhouette_analysis: SilhouetteStabilityAnalysis,
        coordination_analysis: CoordinationThresholdAnalysis,
        feature_contribution: FeatureContributionAnalysis,
    ) -> tuple[list[str], list[str], bool]:
        """Generate overall assessment."""
        concerns = []
        recommendations = []

        # Silhouette vs stability
        if silhouette_analysis.contradiction_detected:
            concerns.append(
                f"CRITICAL: Silhouette ({silhouette_analysis.silhouette_score:.3f}) contradicts "
                f"stability (ARI={silhouette_analysis.ari_mean:.3f}). "
                "Clusters may be geometrically separated but not semantically stable."
            )
            recommendations.append(
                "Investigate feature transforms - high silhouette with low stability "
                "suggests artificial space stretching"
            )

        for cause in silhouette_analysis.likely_causes:
            concerns.append(cause)

        # Coordination dominance
        if coordination_analysis.coordinated_percentage > 40:
            concerns.append(
                f"WARNING: {coordination_analysis.coordinated_percentage:.1f}% classified as coordinated_cluster. "
                f"Threshold ({coordination_analysis.shared_funder_threshold}) may be too permissive."
            )
            recommendations.append(coordination_analysis.threshold_rationale)

        if coordination_analysis.wallets_overridden_to_coordinated > 0:
            override_pct = coordination_analysis.wallets_overridden_to_coordinated / coordination_analysis.total_wallets * 100
            if override_pct > 10:
                concerns.append(
                    f"WARNING: {override_pct:.1f}% of wallets overridden to coordinated from other archetypes"
                )

        # Feature contribution
        if feature_contribution.is_timing_engine and not feature_contribution.graph_adds_value:
            concerns.append(
                "INSIGHT: SHI is currently a timing engine, not a graph intelligence engine. "
                f"Temporal features contribute {feature_contribution.temporal_contribution_pct:.1f}% "
                f"vs graph {feature_contribution.graph_contribution_pct:.1f}%"
            )
            recommendations.append(
                "Graph features need strengthening: consider motifs, temporal dynamics, "
                "community evolution, funding topology"
            )

        # Overall semantic validity
        is_semantic = (
            not silhouette_analysis.contradiction_detected
            and coordination_analysis.coordinated_percentage < 40
            and (feature_contribution.graph_adds_value or feature_contribution.temporal_contribution_pct < 70)
        )

        if not is_semantic:
            recommendations.append(
                "Clusters may not represent real behavioural structure. "
                "Validate against known wallets before deploying."
            )

        return concerns, recommendations, is_semantic
