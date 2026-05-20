"""
Feature-Group Ablation Tests for SHI Clustering.

Tests importance of feature groups by measuring clustering quality
when each group is removed.

Feature Groups:
- distribution: balance, share, rank
- temporal: entry_time_relative, holding_duration, position_volatility
- flow: delta_balance_7d, delta_balance_30d
- trading: trade_count, burstiness, swap_frequency, lp_interaction_ratio
- graph: in_degree, out_degree, eigenvector_centrality, shared_funder_count, weighted features
- price_pnl: entry_price, current_price, unrealized_pnl_ratio, price changes
- liquidity: liquidity_usd features, sell_pressure_vs_liquidity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import structlog

from .diagnostics import HDBSCANDiagnostics, ClusterDiagnostics

logger = structlog.get_logger()


# Feature group definitions (maps to WalletFeatureVector attributes)
FEATURE_GROUPS = {
    "distribution": [
        "balance",
        "share",
        "rank",
    ],
    "temporal": [
        "entry_time_relative",
        "holding_duration",
        "position_volatility",
    ],
    "flow": [
        "delta_balance_7d",
        "delta_balance_30d",
    ],
    "trading": [
        "trade_count",
        "burstiness",
        "swap_frequency",
        "lp_interaction_ratio",
    ],
    "graph": [
        "in_degree",
        "out_degree",
        "eigenvector_centrality",
        "shared_funder_count",
        "total_funding_received",
        "largest_funder_share",
        "funding_hhi",
        "funding_burst_score",
        "weighted_in_degree",
        "weighted_out_degree",
    ],
    "price_pnl": [
        "entry_price_usd",
        "current_price_usd",
        "unrealized_pnl_ratio",
        "unrealized_pnl_usd",
        "price_change_1h_pct",
        "price_change_24h_pct",
        "price_change_7d_pct",
    ],
    "liquidity": [
        "liquidity_usd_current",
        "liquidity_usd_1h_avg",
        "liquidity_usd_24h_avg",
        "sell_pressure_vs_liquidity",
        "unrealized_profit_concentration",
    ],
}


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""

    excluded_group: str
    included_features: list[str]
    excluded_features: list[str]

    # Clustering metrics
    n_clusters: int
    noise_percentage: float
    silhouette_score: Optional[float]

    # Comparison to baseline
    silhouette_delta: Optional[float]
    noise_delta: float
    cluster_delta: int


@dataclass
class AblationStudyResult:
    """Result of full ablation study."""

    baseline: ClusterDiagnostics
    ablations: dict[str, AblationResult]
    feature_importance: dict[str, float]
    recommended_groups: list[str]

    def get_most_important_groups(self, n: int = 3) -> list[tuple[str, float]]:
        """Get top N most important feature groups."""
        sorted_importance = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_importance[:n]


class AblationTester:
    """
    Conducts feature-group ablation tests.

    Tests clustering quality by systematically removing each feature group
    and measuring the impact on cluster quality metrics.
    """

    def __init__(
        self,
        feature_groups: Optional[dict[str, list[str]]] = None,
        min_cluster_size: int = 5,
    ):
        """
        Initialize ablation tester.

        Args:
            feature_groups: Custom feature group definitions
            min_cluster_size: HDBSCAN min_cluster_size parameter
        """
        self.feature_groups = feature_groups or FEATURE_GROUPS
        self.min_cluster_size = min_cluster_size

    def run_ablation_study(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
    ) -> AblationStudyResult:
        """
        Run full ablation study.

        Args:
            features: (n_samples, n_features) array
            feature_names: Names corresponding to columns

        Returns:
            AblationStudyResult with baseline and all ablations
        """
        logger.info(
            "starting_ablation_study",
            n_samples=features.shape[0],
            n_features=features.shape[1],
            n_groups=len(self.feature_groups),
        )

        # Create feature name to index mapping
        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        # Run baseline clustering (all features)
        baseline = self._run_clustering(features, "baseline")
        logger.info(
            "baseline_clustering",
            n_clusters=baseline.n_clusters,
            silhouette=baseline.silhouette_score,
            noise_pct=baseline.noise_percentage,
        )

        # Run ablation for each group
        ablations: dict[str, AblationResult] = {}

        for group_name, group_features in self.feature_groups.items():
            # Find indices of features in this group
            group_indices = [
                name_to_idx[f] for f in group_features
                if f in name_to_idx
            ]

            if not group_indices:
                logger.debug(f"skipping_empty_group", group=group_name)
                continue

            # Get indices of features NOT in this group
            all_indices = set(range(features.shape[1]))
            keep_indices = sorted(all_indices - set(group_indices))

            if not keep_indices:
                logger.warning(f"ablation_would_remove_all_features", group=group_name)
                continue

            # Create ablated feature matrix
            ablated_features = features[:, keep_indices]
            ablated_names = [feature_names[i] for i in keep_indices]
            excluded_names = [feature_names[i] for i in group_indices]

            # Run clustering on ablated features
            diagnostics = self._run_clustering(ablated_features, f"ablate_{group_name}")

            # Compute deltas
            silhouette_delta = None
            if diagnostics.silhouette_score is not None and baseline.silhouette_score is not None:
                silhouette_delta = diagnostics.silhouette_score - baseline.silhouette_score

            ablations[group_name] = AblationResult(
                excluded_group=group_name,
                included_features=ablated_names,
                excluded_features=excluded_names,
                n_clusters=diagnostics.n_clusters,
                noise_percentage=diagnostics.noise_percentage,
                silhouette_score=diagnostics.silhouette_score,
                silhouette_delta=silhouette_delta,
                noise_delta=diagnostics.noise_percentage - baseline.noise_percentage,
                cluster_delta=diagnostics.n_clusters - baseline.n_clusters,
            )

            logger.debug(
                "ablation_completed",
                group=group_name,
                silhouette_delta=silhouette_delta,
                noise_delta=ablations[group_name].noise_delta,
            )

        # Compute feature importance
        # Higher importance = larger drop in silhouette when removed
        feature_importance = {}
        for group_name, result in ablations.items():
            if result.silhouette_delta is not None:
                # Negative delta means removing hurt clustering -> important
                feature_importance[group_name] = -result.silhouette_delta
            else:
                # Use noise increase as proxy
                feature_importance[group_name] = result.noise_delta

        # Recommend groups (those that hurt clustering when removed)
        recommended = [
            group for group, importance in feature_importance.items()
            if importance > 0
        ]

        logger.info(
            "ablation_study_completed",
            recommended_groups=recommended,
            most_important=max(feature_importance.items(), key=lambda x: x[1])[0]
            if feature_importance else None,
        )

        return AblationStudyResult(
            baseline=baseline,
            ablations=ablations,
            feature_importance=feature_importance,
            recommended_groups=recommended,
        )

    def _run_clustering(
        self,
        features: NDArray[np.float64],
        label: str,
    ) -> ClusterDiagnostics:
        """Run HDBSCAN clustering and return diagnostics."""
        from sklearn.preprocessing import StandardScaler

        # Handle NaN
        col_medians = np.nanmedian(features, axis=0)
        for i in range(features.shape[1]):
            mask = np.isnan(features[:, i])
            features[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Run HDBSCAN with diagnostics
        diagnostics = HDBSCANDiagnostics(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
        )

        return diagnostics.fit(features_scaled)

    def compare_feature_combinations(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
        combinations: list[list[str]],
    ) -> list[tuple[list[str], ClusterDiagnostics]]:
        """
        Compare multiple feature combinations.

        Args:
            features: Full feature matrix
            feature_names: All feature names
            combinations: List of feature group combinations to test

        Returns:
            List of (feature_groups, diagnostics) tuples
        """
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        results = []

        for combo in combinations:
            # Get indices for all features in these groups
            indices = []
            for group_name in combo:
                if group_name in self.feature_groups:
                    for f in self.feature_groups[group_name]:
                        if f in name_to_idx:
                            indices.append(name_to_idx[f])

            if not indices:
                continue

            indices = sorted(set(indices))
            subset_features = features[:, indices]

            diagnostics = self._run_clustering(subset_features, f"combo_{'_'.join(combo)}")
            results.append((combo, diagnostics))

        return results
