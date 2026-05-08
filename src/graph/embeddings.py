"""
Node2Vec Graph Embeddings for SHI.

Embeds funding graph into latent space for:
- Wallet similarity detection
- Sybil cluster discovery
- Feature engineering for anomaly detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity

from .funding_graph import FundingGraph
from ..core.types import WalletAddress

logger = structlog.get_logger()


@dataclass
class EmbeddingConfig:
    """Configuration for Node2Vec embeddings."""

    dimensions: int = 64
    walk_length: int = 30
    num_walks: int = 200
    p: float = 1.0  # Return parameter
    q: float = 1.0  # In-out parameter
    window: int = 10  # Context window for Word2Vec
    min_count: int = 1
    batch_words: int = 4
    workers: int = 4


@dataclass
class WalletEmbedding:
    """Embedding vector for a wallet."""

    wallet: WalletAddress
    vector: np.ndarray
    embedding_id: str

    def __post_init__(self):
        """Ensure vector is numpy array."""
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector)


class GraphEmbedder:
    """
    Embeds funding graph using Node2Vec.

    Node2Vec learns node embeddings by optimizing likelihood of preserving
    network neighborhoods via random walks.

    Usage:
        embedder = GraphEmbedder()
        embeddings = embedder.fit_transform(funding_graph)
        sim_matrix = embedder.compute_similarity_matrix()
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedder.

        Args:
            config: Embedding configuration (uses defaults if None)
        """
        self.config = config or EmbeddingConfig()
        self.model: Optional[Node2Vec] = None
        self.embeddings: Dict[WalletAddress, np.ndarray] = {}
        self.embedding_id: Optional[str] = None

    def fit_transform(
        self,
        graph: FundingGraph,
        embedding_id: Optional[str] = None,
    ) -> Dict[WalletAddress, WalletEmbedding]:
        """
        Fit Node2Vec and generate embeddings.

        Args:
            graph: Funding graph to embed
            embedding_id: Optional identifier for this embedding version

        Returns:
            Dict mapping wallet -> WalletEmbedding
        """
        if graph.num_vertices < 2:
            logger.warning("graph_too_small_for_embedding", nodes=graph.num_vertices)
            return {}

        logger.info(
            "fitting_node2vec",
            nodes=graph.num_vertices,
            edges=graph.num_edges,
            dimensions=self.config.dimensions,
        )

        # Get underlying NetworkX graph
        nx_graph = graph._graph

        # Initialize Node2Vec
        self.model = Node2Vec(
            nx_graph,
            dimensions=self.config.dimensions,
            walk_length=self.config.walk_length,
            num_walks=self.config.num_walks,
            p=self.config.p,
            q=self.config.q,
            workers=self.config.workers,
        )

        # Fit model
        try:
            model = self.model.fit(
                window=self.config.window,
                min_count=self.config.min_count,
                batch_words=self.config.batch_words,
            )

            # Extract embeddings
            self.embedding_id = embedding_id or f"embedding_{graph._created_at.isoformat()}"
            self.embeddings = {}

            for node in nx_graph.nodes():
                if node in model.wv:
                    self.embeddings[node] = model.wv[node]

            logger.info(
                "node2vec_completed",
                embedding_count=len(self.embeddings),
                dimensions=self.config.dimensions,
            )

            # Return as WalletEmbedding objects
            return {
                wallet: WalletEmbedding(
                    wallet=wallet,
                    vector=vector,
                    embedding_id=self.embedding_id,
                )
                for wallet, vector in self.embeddings.items()
            }

        except Exception as e:
            logger.error("node2vec_failed", error=str(e), error_type=type(e).__name__)
            raise

    def get_embedding(self, wallet: WalletAddress) -> Optional[np.ndarray]:
        """
        Get embedding vector for a wallet.

        Args:
            wallet: Wallet address

        Returns:
            Embedding vector or None if not found
        """
        return self.embeddings.get(wallet)

    def compute_similarity(
        self,
        wallet1: WalletAddress,
        wallet2: WalletAddress,
    ) -> Optional[float]:
        """
        Compute cosine similarity between two wallets.

        Args:
            wallet1: First wallet
            wallet2: Second wallet

        Returns:
            Cosine similarity in [-1, 1] or None if either wallet not found
        """
        emb1 = self.get_embedding(wallet1)
        emb2 = self.get_embedding(wallet2)

        if emb1 is None or emb2 is None:
            return None

        # Cosine similarity
        similarity = cosine_similarity([emb1], [emb2])[0, 0]
        return float(similarity)

    def compute_similarity_matrix(
        self,
        wallets: Optional[List[WalletAddress]] = None,
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix.

        Args:
            wallets: Wallets to compute similarities for (uses all if None)

        Returns:
            NxN similarity matrix where entry (i,j) is similarity between wallets[i] and wallets[j]
        """
        if wallets is None:
            wallets = list(self.embeddings.keys())

        if not wallets:
            return np.array([])

        # Stack embeddings
        embedding_matrix = np.vstack([self.embeddings[w] for w in wallets if w in self.embeddings])

        if embedding_matrix.shape[0] == 0:
            return np.array([])

        # Compute pairwise cosine similarities
        sim_matrix = cosine_similarity(embedding_matrix)

        return sim_matrix

    def find_similar_wallets(
        self,
        wallet: WalletAddress,
        k: int = 10,
        min_similarity: float = 0.7,
    ) -> List[Tuple[WalletAddress, float]]:
        """
        Find most similar wallets to a target wallet.

        Args:
            wallet: Target wallet
            k: Number of similar wallets to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (wallet, similarity) tuples sorted by similarity (descending)
        """
        emb = self.get_embedding(wallet)
        if emb is None:
            logger.warning("wallet_not_in_embeddings", wallet=wallet)
            return []

        # Compute similarities to all other wallets
        similarities = []
        for other_wallet, other_emb in self.embeddings.items():
            if other_wallet == wallet:
                continue

            sim = cosine_similarity([emb], [other_emb])[0, 0]
            if sim >= min_similarity:
                similarities.append((other_wallet, float(sim)))

        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def cluster_embeddings(
        self,
        n_clusters: int = 5,
        method: str = "kmeans",
    ) -> Dict[WalletAddress, int]:
        """
        Cluster wallets based on embeddings.

        Args:
            n_clusters: Number of clusters
            method: Clustering method ('kmeans' or 'hdbscan')

        Returns:
            Dict mapping wallet -> cluster_id
        """
        if not self.embeddings:
            return {}

        wallets = list(self.embeddings.keys())
        embedding_matrix = np.vstack([self.embeddings[w] for w in wallets])

        if method == "kmeans":
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embedding_matrix)

        elif method == "hdbscan":
            import hdbscan

            clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
            labels = clusterer.fit_predict(embedding_matrix)

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Map wallets to cluster IDs
        wallet_to_cluster = {wallet: int(label) for wallet, label in zip(wallets, labels)}

        logger.info(
            "embeddings_clustered",
            method=method,
            n_clusters=len(set(labels)),
            wallet_count=len(wallets),
        )

        return wallet_to_cluster

    def save_embeddings(self, filepath: str) -> None:
        """
        Save embeddings to disk.

        Args:
            filepath: Path to save embeddings
        """
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "embeddings": self.embeddings,
                    "config": self.config,
                    "embedding_id": self.embedding_id,
                },
                f,
            )

        logger.info("embeddings_saved", filepath=filepath, count=len(self.embeddings))

    @classmethod
    def load_embeddings(cls, filepath: str) -> "GraphEmbedder":
        """
        Load embeddings from disk.

        Args:
            filepath: Path to load embeddings from

        Returns:
            GraphEmbedder with loaded embeddings
        """
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        embedder = cls(config=data["config"])
        embedder.embeddings = data["embeddings"]
        embedder.embedding_id = data["embedding_id"]

        logger.info("embeddings_loaded", filepath=filepath, count=len(embedder.embeddings))

        return embedder
