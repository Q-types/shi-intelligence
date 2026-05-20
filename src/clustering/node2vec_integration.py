"""
Node2Vec Integration for SHI Clustering.

Experimental integration of Node2Vec graph embeddings with behavioral clustering.

Features:
- Reduce embeddings to 4-8 dimensions before clustering
- Compare behavior-only, graph-only, and combined clustering
- Optional PCA/UMAP dimensionality reduction
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import structlog

from ..graph.funding_graph import FundingGraph
from ..graph.embeddings import GraphEmbedder, EmbeddingConfig, WalletEmbedding
from .diagnostics import HDBSCANDiagnostics, ClusterDiagnostics, compare_clustering_results

logger = structlog.get_logger()


class ClusteringMode(Enum):
    """Clustering mode for comparison."""

    BEHAVIOR_ONLY = "behavior_only"  # Only behavioral features
    GRAPH_ONLY = "graph_only"  # Only Node2Vec embeddings
    COMBINED = "combined"  # Behavioral + reduced embeddings


@dataclass
class Node2VecConfig:
    """Configuration for Node2Vec integration."""

    # Node2Vec parameters
    embedding_dimensions: int = 64
    walk_length: int = 30
    num_walks: int = 200
    p: float = 1.0
    q: float = 1.0

    # Dimensionality reduction for clustering
    reduced_dimensions: int = 6  # 4-8 recommended
    reduction_method: str = "pca"  # "pca" or "umap"

    # Combination weights
    behavior_weight: float = 0.7
    graph_weight: float = 0.3


@dataclass
class ClusteringComparison:
    """Result of comparing different clustering modes."""

    behavior_only: ClusterDiagnostics
    graph_only: Optional[ClusterDiagnostics]
    combined: Optional[ClusterDiagnostics]

    # Comparison metrics
    best_mode: ClusteringMode
    best_silhouette: Optional[float]

    # Mode-specific details
    mode_details: dict[ClusteringMode, dict]

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "best_mode": self.best_mode.value,
            "best_silhouette": self.best_silhouette,
            "behavior_only": self.behavior_only.to_dict(),
            "graph_only": self.graph_only.to_dict() if self.graph_only else None,
            "combined": self.combined.to_dict() if self.combined else None,
            "mode_details": {k.value: v for k, v in self.mode_details.items()},
        }


class Node2VecClusteringIntegration:
    """
    Integrates Node2Vec embeddings with behavioral clustering.

    Provides experimental comparison of:
    1. Behavior-only clustering (current approach)
    2. Graph-only clustering (Node2Vec embeddings)
    3. Combined clustering (behavioral + reduced embeddings)
    """

    def __init__(
        self,
        config: Optional[Node2VecConfig] = None,
        min_cluster_size: int = 5,
    ):
        """
        Initialize integration.

        Args:
            config: Node2Vec configuration
            min_cluster_size: HDBSCAN min_cluster_size
        """
        self.config = config or Node2VecConfig()
        self.min_cluster_size = min_cluster_size

        # Embedding state
        self._embedder: Optional[GraphEmbedder] = None
        self._reduced_embeddings: Optional[dict[str, NDArray[np.float64]]] = None

    def fit_embeddings(
        self,
        graph: FundingGraph,
        wallets: list[str],
    ) -> dict[str, NDArray[np.float64]]:
        """
        Fit Node2Vec embeddings and reduce dimensions.

        Args:
            graph: Funding graph
            wallets: List of wallets to embed

        Returns:
            Dict mapping wallet -> reduced embedding vector
        """
        logger.info(
            "fitting_node2vec_embeddings",
            n_wallets=len(wallets),
            embedding_dim=self.config.embedding_dimensions,
            reduced_dim=self.config.reduced_dimensions,
        )

        # Initialize embedder with config
        embedding_config = EmbeddingConfig(
            dimensions=self.config.embedding_dimensions,
            walk_length=self.config.walk_length,
            num_walks=self.config.num_walks,
            p=self.config.p,
            q=self.config.q,
        )

        self._embedder = GraphEmbedder(config=embedding_config)

        try:
            # Fit Node2Vec
            embeddings = self._embedder.fit_transform(graph)

            # Get embeddings for target wallets
            wallet_embeddings = {}
            for wallet in wallets:
                if wallet in embeddings:
                    wallet_embeddings[wallet] = embeddings[wallet].vector

            if not wallet_embeddings:
                logger.warning("no_embeddings_found_for_wallets")
                return {}

            # Reduce dimensions
            self._reduced_embeddings = self._reduce_dimensions(wallet_embeddings, wallets)

            logger.info(
                "embeddings_fitted",
                n_embedded=len(self._reduced_embeddings),
                final_dim=self.config.reduced_dimensions,
            )

            return self._reduced_embeddings

        except Exception as e:
            logger.error("embedding_fitting_failed", error=str(e))
            return {}

    def _reduce_dimensions(
        self,
        embeddings: dict[str, NDArray[np.float64]],
        wallets: list[str],
    ) -> dict[str, NDArray[np.float64]]:
        """Reduce embedding dimensions using PCA or UMAP."""
        # Build embedding matrix for wallets that have embeddings
        wallet_list = [w for w in wallets if w in embeddings]
        if not wallet_list:
            return {}

        embedding_matrix = np.vstack([embeddings[w] for w in wallet_list])

        if self.config.reduction_method == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=self.config.reduced_dimensions)
            reduced_matrix = reducer.fit_transform(embedding_matrix)

            logger.debug(
                "pca_reduction",
                explained_variance=sum(reducer.explained_variance_ratio_),
            )

        elif self.config.reduction_method == "umap":
            try:
                import umap

                reducer = umap.UMAP(
                    n_components=self.config.reduced_dimensions,
                    n_neighbors=15,
                    min_dist=0.1,
                    random_state=42,
                )
                reduced_matrix = reducer.fit_transform(embedding_matrix)
            except ImportError:
                logger.warning("umap_not_available_falling_back_to_pca")
                from sklearn.decomposition import PCA

                reducer = PCA(n_components=self.config.reduced_dimensions)
                reduced_matrix = reducer.fit_transform(embedding_matrix)
        else:
            raise ValueError(f"Unknown reduction method: {self.config.reduction_method}")

        # Map back to wallets
        return {wallet: reduced_matrix[i] for i, wallet in enumerate(wallet_list)}

    def compare_clustering_modes(
        self,
        behavioral_features: NDArray[np.float64],
        wallets: list[str],
        graph: Optional[FundingGraph] = None,
    ) -> ClusteringComparison:
        """
        Compare behavior-only, graph-only, and combined clustering.

        Args:
            behavioral_features: (n_samples, n_features) behavioral features
            wallets: List of wallet addresses (same order as features)
            graph: Optional funding graph for Node2Vec

        Returns:
            ClusteringComparison with all results
        """
        from sklearn.preprocessing import StandardScaler

        logger.info(
            "comparing_clustering_modes",
            n_wallets=len(wallets),
            n_behavioral_features=behavioral_features.shape[1],
        )

        mode_details: dict[ClusteringMode, dict] = {}

        # 1. Behavior-only clustering
        behavior_diagnostics = self._cluster_features(
            behavioral_features,
            "behavior_only",
        )
        mode_details[ClusteringMode.BEHAVIOR_ONLY] = {
            "n_features": behavioral_features.shape[1],
            "feature_type": "behavioral",
        }

        # 2. Graph-only clustering (if embeddings available)
        graph_diagnostics = None
        if graph is not None:
            if self._reduced_embeddings is None:
                self.fit_embeddings(graph, wallets)

            if self._reduced_embeddings:
                # Build embedding matrix for wallets
                graph_features = []
                valid_wallets = []
                for wallet in wallets:
                    if wallet in self._reduced_embeddings:
                        graph_features.append(self._reduced_embeddings[wallet])
                        valid_wallets.append(wallet)

                if graph_features:
                    graph_matrix = np.vstack(graph_features)
                    graph_diagnostics = self._cluster_features(
                        graph_matrix,
                        "graph_only",
                    )
                    mode_details[ClusteringMode.GRAPH_ONLY] = {
                        "n_features": self.config.reduced_dimensions,
                        "feature_type": "graph_embedding",
                        "n_wallets_with_embeddings": len(valid_wallets),
                    }

        # 3. Combined clustering (if embeddings available)
        combined_diagnostics = None
        if self._reduced_embeddings:
            combined_features = self._combine_features(
                behavioral_features,
                wallets,
            )
            if combined_features is not None:
                combined_diagnostics = self._cluster_features(
                    combined_features,
                    "combined",
                )
                mode_details[ClusteringMode.COMBINED] = {
                    "n_features": combined_features.shape[1],
                    "behavior_weight": self.config.behavior_weight,
                    "graph_weight": self.config.graph_weight,
                }

        # Determine best mode
        best_mode = ClusteringMode.BEHAVIOR_ONLY
        best_silhouette = behavior_diagnostics.silhouette_score

        if graph_diagnostics and graph_diagnostics.silhouette_score:
            if best_silhouette is None or graph_diagnostics.silhouette_score > best_silhouette:
                best_mode = ClusteringMode.GRAPH_ONLY
                best_silhouette = graph_diagnostics.silhouette_score

        if combined_diagnostics and combined_diagnostics.silhouette_score:
            if best_silhouette is None or combined_diagnostics.silhouette_score > best_silhouette:
                best_mode = ClusteringMode.COMBINED
                best_silhouette = combined_diagnostics.silhouette_score

        logger.info(
            "clustering_comparison_completed",
            best_mode=best_mode.value,
            best_silhouette=best_silhouette,
            behavior_silhouette=behavior_diagnostics.silhouette_score,
            graph_silhouette=graph_diagnostics.silhouette_score if graph_diagnostics else None,
            combined_silhouette=combined_diagnostics.silhouette_score if combined_diagnostics else None,
        )

        return ClusteringComparison(
            behavior_only=behavior_diagnostics,
            graph_only=graph_diagnostics,
            combined=combined_diagnostics,
            best_mode=best_mode,
            best_silhouette=best_silhouette,
            mode_details=mode_details,
        )

    def _combine_features(
        self,
        behavioral_features: NDArray[np.float64],
        wallets: list[str],
    ) -> Optional[NDArray[np.float64]]:
        """Combine behavioral features with reduced embeddings."""
        if self._reduced_embeddings is None:
            return None

        from sklearn.preprocessing import StandardScaler

        # Scale behavioral features
        behavior_scaler = StandardScaler()
        behavior_scaled = behavior_scaler.fit_transform(behavioral_features)

        # Build embedding matrix (fill missing with zeros)
        embedding_dim = self.config.reduced_dimensions
        embedding_matrix = np.zeros((len(wallets), embedding_dim))

        for i, wallet in enumerate(wallets):
            if wallet in self._reduced_embeddings:
                embedding_matrix[i] = self._reduced_embeddings[wallet]

        # Scale embeddings
        embedding_scaler = StandardScaler()
        embedding_scaled = embedding_scaler.fit_transform(embedding_matrix)

        # Combine with weights
        combined = np.hstack([
            behavior_scaled * self.config.behavior_weight,
            embedding_scaled * self.config.graph_weight,
        ])

        return combined

    def _cluster_features(
        self,
        features: NDArray[np.float64],
        label: str,
    ) -> ClusterDiagnostics:
        """Run HDBSCAN clustering on features."""
        from sklearn.preprocessing import StandardScaler

        # Handle NaN
        features = features.copy()
        col_medians = np.nanmedian(features, axis=0)
        for i in range(features.shape[1]):
            mask = np.isnan(features[:, i])
            features[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

        # Scale
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Cluster
        diagnostics = HDBSCANDiagnostics(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
        )

        return diagnostics.fit(features_scaled)

    def get_embedding_similarity(
        self,
        wallet1: str,
        wallet2: str,
    ) -> Optional[float]:
        """Get cosine similarity between two wallets based on embeddings."""
        if self._embedder is None:
            return None

        return self._embedder.compute_similarity(wallet1, wallet2)
