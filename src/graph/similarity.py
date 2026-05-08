"""
Wallet Similarity Detection for SHI.

Detects:
- Coordinated wallet clusters (Sybil detection)
- Similar behavior patterns
- Hidden relationships via embeddings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import structlog

from .embeddings import GraphEmbedder
from .funding_graph import FundingGraph
from ..core.types import WalletAddress

logger = structlog.get_logger()


@dataclass
class SimilarityScore:
    """Similarity score between two wallets."""

    wallet1: WalletAddress
    wallet2: WalletAddress
    embedding_similarity: float
    structural_similarity: float
    combined_similarity: float
    is_coordinated: bool


@dataclass
class CoordinatedCluster:
    """Cluster of coordinated wallets."""

    cluster_id: int
    wallets: Set[WalletAddress]
    mean_similarity: float
    sybil_probability: float
    shared_funders: Set[WalletAddress]


class WalletSimilarityDetector:
    """
    Detects similar wallets using embeddings and structural features.

    Combines:
    - Embedding-based similarity (Node2Vec)
    - Structural similarity (shared funders, funding patterns)
    - Behavioral similarity (timing, amounts)
    """

    def __init__(
        self,
        embedder: GraphEmbedder,
        graph: FundingGraph,
        embedding_weight: float = 0.7,
        structural_weight: float = 0.3,
    ):
        """
        Initialize similarity detector.

        Args:
            embedder: Fitted GraphEmbedder
            graph: Funding graph
            embedding_weight: Weight for embedding similarity (0-1)
            structural_weight: Weight for structural similarity (0-1)
        """
        self.embedder = embedder
        self.graph = graph
        self.embedding_weight = embedding_weight
        self.structural_weight = structural_weight

        # Ensure weights sum to 1
        total = embedding_weight + structural_weight
        self.embedding_weight /= total
        self.structural_weight /= total

    def compute_structural_similarity(
        self,
        wallet1: WalletAddress,
        wallet2: WalletAddress,
    ) -> float:
        """
        Compute structural similarity based on shared funders.

        Uses Jaccard similarity on ancestor sets.

        Args:
            wallet1: First wallet
            wallet2: Second wallet

        Returns:
            Jaccard similarity in [0, 1]
        """
        ancestors1 = self.graph.get_ancestors(wallet1, max_depth=2)
        ancestors2 = self.graph.get_ancestors(wallet2, max_depth=2)

        if not ancestors1 and not ancestors2:
            return 0.0

        intersection = len(ancestors1 & ancestors2)
        union = len(ancestors1 | ancestors2)

        if union == 0:
            return 0.0

        return intersection / union

    def compute_similarity(
        self,
        wallet1: WalletAddress,
        wallet2: WalletAddress,
        coordination_threshold: float = 0.75,
    ) -> Optional[SimilarityScore]:
        """
        Compute combined similarity between two wallets.

        Args:
            wallet1: First wallet
            wallet2: Second wallet
            coordination_threshold: Threshold for flagging as coordinated

        Returns:
            SimilarityScore or None if wallets not in embeddings
        """
        # Get embedding similarity
        emb_sim = self.embedder.compute_similarity(wallet1, wallet2)
        if emb_sim is None:
            return None

        # Get structural similarity
        struct_sim = self.compute_structural_similarity(wallet1, wallet2)

        # Combined similarity (weighted average)
        combined = (
            self.embedding_weight * emb_sim + self.structural_weight * struct_sim
        )

        # Flag as coordinated if above threshold
        is_coordinated = combined >= coordination_threshold

        return SimilarityScore(
            wallet1=wallet1,
            wallet2=wallet2,
            embedding_similarity=float(emb_sim),
            structural_similarity=struct_sim,
            combined_similarity=combined,
            is_coordinated=is_coordinated,
        )

    def find_coordinated_pairs(
        self,
        wallets: List[WalletAddress],
        min_similarity: float = 0.75,
        top_k: Optional[int] = None,
    ) -> List[SimilarityScore]:
        """
        Find pairs of wallets with high coordination likelihood.

        Args:
            wallets: Wallets to analyze
            min_similarity: Minimum similarity threshold
            top_k: Return only top k pairs (None = all above threshold)

        Returns:
            List of SimilarityScore objects sorted by combined_similarity
        """
        logger.info("finding_coordinated_pairs", wallet_count=len(wallets))

        coordinated_pairs = []

        # Compute pairwise similarities
        for i, w1 in enumerate(wallets):
            for w2 in wallets[i + 1 :]:
                score = self.compute_similarity(w1, w2, coordination_threshold=min_similarity)
                if score and score.is_coordinated:
                    coordinated_pairs.append(score)

        # Sort by combined similarity (descending)
        coordinated_pairs.sort(key=lambda x: x.combined_similarity, reverse=True)

        if top_k:
            coordinated_pairs = coordinated_pairs[:top_k]

        logger.info(
            "coordinated_pairs_found",
            count=len(coordinated_pairs),
            min_similarity=min_similarity,
        )

        return coordinated_pairs

    def detect_sybil_clusters(
        self,
        wallets: List[WalletAddress],
        similarity_threshold: float = 0.75,
        min_cluster_size: int = 3,
    ) -> List[CoordinatedCluster]:
        """
        Detect Sybil clusters using similarity graph.

        Builds similarity graph and finds connected components.

        Args:
            wallets: Wallets to analyze
            similarity_threshold: Minimum similarity to connect wallets
            min_cluster_size: Minimum wallets in a cluster

        Returns:
            List of CoordinatedCluster objects
        """
        import networkx as nx

        logger.info(
            "detecting_sybil_clusters",
            wallet_count=len(wallets),
            threshold=similarity_threshold,
        )

        # Build similarity graph
        sim_graph = nx.Graph()
        for w in wallets:
            sim_graph.add_node(w)

        # Add edges for similar wallets
        for i, w1 in enumerate(wallets):
            for w2 in wallets[i + 1 :]:
                score = self.compute_similarity(w1, w2)
                if score and score.combined_similarity >= similarity_threshold:
                    sim_graph.add_edge(w1, w2, weight=score.combined_similarity)

        # Find connected components
        clusters = []
        for cluster_id, component in enumerate(nx.connected_components(sim_graph)):
            if len(component) < min_cluster_size:
                continue

            # Compute cluster statistics
            component_list = list(component)
            similarities = []

            for i, w1 in enumerate(component_list):
                for w2 in component_list[i + 1 :]:
                    score = self.compute_similarity(w1, w2)
                    if score:
                        similarities.append(score.combined_similarity)

            mean_sim = np.mean(similarities) if similarities else 0.0

            # Find shared funders
            shared = self.graph.find_shared_funders(component_list, max_depth=2)

            # Estimate Sybil probability based on mean similarity and shared funders
            sybil_prob = self._estimate_sybil_probability(
                float(mean_sim), len(shared), len(component)
            )

            cluster = CoordinatedCluster(
                cluster_id=cluster_id,
                wallets=component,
                mean_similarity=float(mean_sim),
                sybil_probability=sybil_prob,
                shared_funders=set(shared.keys()),
            )

            clusters.append(cluster)

        logger.info("sybil_clusters_detected", cluster_count=len(clusters))

        return clusters

    def _estimate_sybil_probability(
        self,
        mean_similarity: float,
        shared_funder_count: int,
        cluster_size: int,
    ) -> float:
        """
        Estimate Sybil probability for a cluster.

        Heuristic based on:
        - High similarity → higher probability
        - Many shared funders → higher probability
        - Large cluster size → higher probability

        Args:
            mean_similarity: Mean similarity within cluster
            shared_funder_count: Number of shared funders
            cluster_size: Number of wallets in cluster

        Returns:
            Sybil probability in [0, 1]
        """
        # Base probability from similarity
        prob = mean_similarity

        # Boost for shared funders (normalized by cluster size)
        funder_ratio = min(shared_funder_count / cluster_size, 1.0)
        prob = prob * 0.7 + funder_ratio * 0.3

        # Boost for larger clusters
        size_boost = min(cluster_size / 10, 1.0) * 0.1
        prob = min(prob + size_boost, 1.0)

        return prob

    def compute_similarity_matrix(
        self,
        wallets: List[WalletAddress],
    ) -> Tuple[np.ndarray, List[WalletAddress]]:
        """
        Compute combined similarity matrix for wallets.

        Args:
            wallets: Wallets to compute similarities for

        Returns:
            (similarity_matrix, wallet_list) tuple
        """
        # Filter to wallets in embeddings
        valid_wallets = [w for w in wallets if self.embedder.get_embedding(w) is not None]

        if not valid_wallets:
            return np.array([]), []

        n = len(valid_wallets)
        sim_matrix = np.zeros((n, n))

        # Compute pairwise similarities
        for i, w1 in enumerate(valid_wallets):
            for j, w2 in enumerate(valid_wallets):
                if i == j:
                    sim_matrix[i, j] = 1.0
                elif i < j:
                    score = self.compute_similarity(w1, w2)
                    if score:
                        sim_matrix[i, j] = score.combined_similarity
                        sim_matrix[j, i] = score.combined_similarity

        return sim_matrix, valid_wallets

    def find_most_similar(
        self,
        wallet: WalletAddress,
        k: int = 10,
        min_similarity: float = 0.5,
    ) -> List[Tuple[WalletAddress, SimilarityScore]]:
        """
        Find k most similar wallets to a target wallet.

        Args:
            wallet: Target wallet
            k: Number of similar wallets to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (wallet, SimilarityScore) tuples sorted by similarity
        """
        if self.embedder.get_embedding(wallet) is None:
            logger.warning("wallet_not_in_embeddings", wallet=wallet)
            return []

        # Get all wallets with embeddings
        all_wallets = list(self.embedder.embeddings.keys())

        similarities = []
        for other_wallet in all_wallets:
            if other_wallet == wallet:
                continue

            score = self.compute_similarity(wallet, other_wallet)
            if score and score.combined_similarity >= min_similarity:
                similarities.append((other_wallet, score))

        # Sort by combined similarity (descending)
        similarities.sort(key=lambda x: x[1].combined_similarity, reverse=True)

        return similarities[:k]

    def analyze_cluster_behavior(
        self,
        cluster: CoordinatedCluster,
    ) -> Dict[str, Any]:
        """
        Analyze behavioral patterns of a coordinated cluster.

        Args:
            cluster: CoordinatedCluster to analyze

        Returns:
            Dict with behavioral insights
        """
        wallets = list(cluster.wallets)

        # Find dominant funder
        dominant_funder, count = self.graph.get_dominant_funder(wallets, max_depth=2)

        # Compute degree statistics
        in_degrees = [self.graph.get_in_degree(w) for w in wallets]
        out_degrees = [self.graph.get_out_degree(w) for w in wallets]

        analysis = {
            "cluster_id": cluster.cluster_id,
            "size": len(wallets),
            "mean_similarity": cluster.mean_similarity,
            "sybil_probability": cluster.sybil_probability,
            "dominant_funder": dominant_funder,
            "funder_coverage": count / len(wallets) if wallets else 0,
            "mean_in_degree": np.mean(in_degrees) if in_degrees else 0,
            "std_in_degree": np.std(in_degrees) if in_degrees else 0,
            "mean_out_degree": np.mean(out_degrees) if out_degrees else 0,
            "shared_funder_count": len(cluster.shared_funders),
        }

        return analysis
