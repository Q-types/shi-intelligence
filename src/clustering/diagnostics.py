"""
HDBSCAN Clustering Diagnostics for SHI.

Provides comprehensive cluster analysis including:
- Cluster labels and membership probabilities
- Outlier scores
- Silhouette analysis
- Cluster persistence
- Noise percentage
- Cluster size distribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import structlog

logger = structlog.get_logger()


class ClusterStatus(Enum):
    """Status of a wallet's cluster assignment."""

    CORE = "core"  # High membership probability
    BORDER = "border"  # Medium membership probability
    NOISE = "noise"  # Assigned to noise cluster (-1)
    UNKNOWN = "unknown"  # Not yet clustered


@dataclass
class ClusterDiagnostics:
    """
    Comprehensive diagnostics for HDBSCAN clustering.

    Required outputs per SHI upgrade requirements:
    - cluster labels
    - membership probabilities
    - outlier scores
    - silhouette score where valid
    - cluster persistence if available
    - percentage noise
    - cluster size distribution
    """

    # Core outputs
    labels: NDArray[np.int32]
    probabilities: NDArray[np.float64]
    outlier_scores: NDArray[np.float64]

    # Aggregate metrics
    silhouette_score: Optional[float]
    noise_percentage: float
    n_clusters: int

    # Cluster-level stats
    cluster_sizes: dict[int, int]
    cluster_persistence: dict[int, float]

    # Metadata
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hdbscan_params: dict = field(default_factory=dict)

    @property
    def valid_cluster_count(self) -> int:
        """Number of non-noise clusters."""
        return len([c for c in self.cluster_sizes.keys() if c >= 0])

    @property
    def largest_cluster_size(self) -> int:
        """Size of the largest non-noise cluster."""
        valid_sizes = [s for c, s in self.cluster_sizes.items() if c >= 0]
        return max(valid_sizes) if valid_sizes else 0

    @property
    def median_cluster_size(self) -> float:
        """Median size of non-noise clusters."""
        valid_sizes = [s for c, s in self.cluster_sizes.items() if c >= 0]
        return float(np.median(valid_sizes)) if valid_sizes else 0.0

    def get_cluster_status(self, idx: int) -> ClusterStatus:
        """Get cluster status for a sample."""
        if self.labels[idx] == -1:
            return ClusterStatus.NOISE
        elif self.probabilities[idx] >= 0.8:
            return ClusterStatus.CORE
        elif self.probabilities[idx] >= 0.5:
            return ClusterStatus.BORDER
        else:
            return ClusterStatus.NOISE

    def to_dict(self) -> dict:
        """Export diagnostics as dictionary."""
        return {
            "n_clusters": self.n_clusters,
            "valid_cluster_count": self.valid_cluster_count,
            "noise_percentage": self.noise_percentage,
            "silhouette_score": self.silhouette_score,
            "largest_cluster_size": self.largest_cluster_size,
            "median_cluster_size": self.median_cluster_size,
            "cluster_sizes": self.cluster_sizes,
            "cluster_persistence": self.cluster_persistence,
            "hdbscan_params": self.hdbscan_params,
            "computed_at": self.computed_at.isoformat(),
        }


@dataclass
class WalletClusterInfo:
    """Cluster information for a single wallet."""

    wallet: str
    cluster_id: int
    membership_probability: float
    outlier_score: float
    cluster_status: ClusterStatus
    confidence_adjustment: float  # Applied to archetype confidence

    @property
    def is_noise(self) -> bool:
        """Check if wallet is in noise cluster."""
        return self.cluster_id == -1 or self.cluster_status == ClusterStatus.NOISE


class HDBSCANDiagnostics:
    """
    Computes comprehensive HDBSCAN diagnostics.

    Integrates with SHI's archetype assignment to provide:
    - Enhanced confidence scoring based on cluster membership
    - Noise handling with reduced confidence
    - Anomaly flagging for review
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        allow_single_cluster: bool = False,
    ):
        """
        Initialize HDBSCAN diagnostics.

        Args:
            min_cluster_size: Minimum size for a cluster (default: 5)
            min_samples: Core sample threshold (default: min_cluster_size)
            metric: Distance metric (default: euclidean)
            cluster_selection_method: 'eom' or 'leaf' (default: eom)
            allow_single_cluster: Allow all points in one cluster (default: False)
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster

        self._clusterer = None
        self._diagnostics: Optional[ClusterDiagnostics] = None

    def fit(self, features: NDArray[np.float64]) -> ClusterDiagnostics:
        """
        Fit HDBSCAN and compute diagnostics.

        Args:
            features: (n_samples, n_features) array

        Returns:
            ClusterDiagnostics with all metrics
        """
        import hdbscan
        from sklearn.metrics import silhouette_score

        logger.info(
            "fitting_hdbscan",
            n_samples=features.shape[0],
            n_features=features.shape[1],
            min_cluster_size=self.min_cluster_size,
        )

        # Initialize and fit HDBSCAN
        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
            prediction_data=True,  # Enable soft clustering
        )

        self._clusterer.fit(features)

        # Extract outputs
        labels = self._clusterer.labels_
        probabilities = self._clusterer.probabilities_
        outlier_scores = self._clusterer.outlier_scores_

        # Compute noise percentage
        n_noise = np.sum(labels == -1)
        noise_pct = n_noise / len(labels) if len(labels) > 0 else 0.0

        # Compute cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(label): int(count) for label, count in zip(unique_labels, counts)}

        # Compute cluster persistence (stability)
        cluster_persistence = {}
        if hasattr(self._clusterer, "cluster_persistence_"):
            for i, persistence in enumerate(self._clusterer.cluster_persistence_):
                if i < len(unique_labels):
                    cluster_persistence[int(unique_labels[i])] = float(persistence)

        # Compute silhouette score (only for non-noise points with multiple clusters)
        silhouette = None
        non_noise_mask = labels != -1
        n_valid_clusters = len([l for l in unique_labels if l >= 0])

        if n_valid_clusters >= 2 and np.sum(non_noise_mask) >= 2:
            try:
                silhouette = silhouette_score(
                    features[non_noise_mask],
                    labels[non_noise_mask],
                    metric=self.metric,
                )
            except Exception as e:
                logger.warning("silhouette_computation_failed", error=str(e))

        # Build diagnostics
        self._diagnostics = ClusterDiagnostics(
            labels=labels,
            probabilities=probabilities,
            outlier_scores=outlier_scores,
            silhouette_score=silhouette,
            noise_percentage=noise_pct,
            n_clusters=n_valid_clusters,
            cluster_sizes=cluster_sizes,
            cluster_persistence=cluster_persistence,
            hdbscan_params={
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "metric": self.metric,
                "cluster_selection_method": self.cluster_selection_method,
            },
        )

        logger.info(
            "hdbscan_fitted",
            n_clusters=n_valid_clusters,
            noise_pct=f"{noise_pct:.1%}",
            silhouette=f"{silhouette:.3f}" if silhouette else "N/A",
        )

        return self._diagnostics

    def get_wallet_info(
        self,
        wallet: str,
        idx: int,
    ) -> WalletClusterInfo:
        """
        Get cluster info for a specific wallet.

        Args:
            wallet: Wallet address
            idx: Index in the feature array

        Returns:
            WalletClusterInfo with cluster details
        """
        if self._diagnostics is None:
            raise ValueError("Must call fit() before get_wallet_info()")

        cluster_id = int(self._diagnostics.labels[idx])
        prob = float(self._diagnostics.probabilities[idx])
        outlier = float(self._diagnostics.outlier_scores[idx])
        status = self._diagnostics.get_cluster_status(idx)

        # Compute confidence adjustment
        # Noise points get reduced confidence
        # Low membership probability also reduces confidence
        if status == ClusterStatus.NOISE:
            confidence_adj = -0.3  # Reduce confidence by 30%
        elif status == ClusterStatus.BORDER:
            confidence_adj = -0.1  # Reduce confidence by 10%
        else:
            confidence_adj = 0.0  # Core points keep full confidence

        # High outlier score also reduces confidence
        if outlier > 0.8:
            confidence_adj -= 0.2

        return WalletClusterInfo(
            wallet=wallet,
            cluster_id=cluster_id,
            membership_probability=prob,
            outlier_score=outlier,
            cluster_status=status,
            confidence_adjustment=confidence_adj,
        )

    def get_anomaly_candidates(
        self,
        wallets: list[str],
        outlier_threshold: float = 0.8,
        probability_threshold: float = 0.3,
    ) -> list[WalletClusterInfo]:
        """
        Get wallets that should be flagged for anomaly review.

        Args:
            wallets: List of wallet addresses
            outlier_threshold: Outlier score threshold for flagging
            probability_threshold: Low probability threshold for flagging

        Returns:
            List of WalletClusterInfo for anomalous wallets
        """
        if self._diagnostics is None:
            raise ValueError("Must call fit() before get_anomaly_candidates()")

        anomalies = []
        for idx, wallet in enumerate(wallets):
            info = self.get_wallet_info(wallet, idx)

            # Flag if high outlier score or low membership probability
            if (
                info.outlier_score >= outlier_threshold
                or info.membership_probability <= probability_threshold
                or info.is_noise
            ):
                anomalies.append(info)

        logger.info(
            "anomaly_candidates_identified",
            total_wallets=len(wallets),
            anomaly_count=len(anomalies),
            noise_count=sum(1 for a in anomalies if a.is_noise),
        )

        return anomalies

    @property
    def diagnostics(self) -> Optional[ClusterDiagnostics]:
        """Get computed diagnostics."""
        return self._diagnostics


def compare_clustering_results(
    baseline: ClusterDiagnostics,
    new: ClusterDiagnostics,
) -> dict:
    """
    Compare two clustering results for validation.

    Args:
        baseline: Existing/baseline clustering diagnostics
        new: New clustering diagnostics to compare

    Returns:
        Comparison metrics for validation
    """
    comparison = {
        "n_clusters_change": new.n_clusters - baseline.n_clusters,
        "noise_pct_change": new.noise_percentage - baseline.noise_percentage,
        "silhouette_change": None,
        "median_size_change": new.median_cluster_size - baseline.median_cluster_size,
    }

    if baseline.silhouette_score is not None and new.silhouette_score is not None:
        comparison["silhouette_change"] = new.silhouette_score - baseline.silhouette_score

    # Flag if new clustering is significantly worse
    is_degraded = (
        (comparison["silhouette_change"] is not None and comparison["silhouette_change"] < -0.1)
        or comparison["noise_pct_change"] > 0.2
        or comparison["n_clusters_change"] < -2
    )

    comparison["is_degraded"] = is_degraded
    comparison["can_deploy"] = not is_degraded

    return comparison
