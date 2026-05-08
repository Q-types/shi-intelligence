"""Behavioral pattern detection in action sequences.

This module provides pattern detection and clustering capabilities
for identifying recurring behavioral motifs in wallet action sequences.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import numpy.typing as npt
import structlog
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from src.core.types import WalletAddress
from .encoder import ActionSequence, WalletActionEncoder, WalletActionType

logger = structlog.get_logger()


@dataclass(frozen=True)
class Motif:
    """A recurring behavioral pattern (motif) found in sequences.

    Attributes
    ----------
    pattern : tuple[WalletActionType, ...]
        The action pattern that defines this motif.
    frequency : int
        Number of times this motif appears across all wallets.
    affected_wallets : tuple[WalletAddress, ...]
        Wallets that exhibit this motif.
    confidence : float
        Confidence score for this motif (0 to 1).
    avg_position : float
        Average normalized position where this motif appears.
    """

    pattern: tuple[WalletActionType, ...]
    frequency: int
    affected_wallets: tuple[WalletAddress, ...]
    confidence: float
    avg_position: float = 0.5

    @property
    def pattern_str(self) -> str:
        """Return pattern as string."""
        return " -> ".join(a.value for a in self.pattern)

    @property
    def length(self) -> int:
        """Return pattern length."""
        return len(self.pattern)


@dataclass(frozen=True)
class BehaviorCluster:
    """A cluster of wallets with similar behavioral patterns.

    Attributes
    ----------
    cluster_id : int
        Unique identifier for this cluster.
    wallets : tuple[WalletAddress, ...]
        Wallets belonging to this cluster.
    centroid : npt.NDArray[np.float64]
        Cluster centroid embedding.
    characteristic_actions : tuple[WalletActionType, ...]
        Most common actions in this cluster.
    cohesion : float
        Intra-cluster cohesion score (0 to 1).
    label : str
        Human-readable label for this cluster.
    """

    cluster_id: int
    wallets: tuple[WalletAddress, ...]
    centroid: npt.NDArray[np.float64]
    characteristic_actions: tuple[WalletActionType, ...]
    cohesion: float
    label: str

    @property
    def size(self) -> int:
        """Return number of wallets in cluster."""
        return len(self.wallets)


@dataclass
class PatternConfig:
    """Configuration for pattern detection.

    Attributes
    ----------
    min_motif_length : int
        Minimum length for motif patterns.
    max_motif_length : int
        Maximum length for motif patterns.
    min_frequency : int
        Minimum frequency for a pattern to be considered a motif.
    min_cluster_size : int
        Minimum wallets per cluster.
    max_clusters : int
        Maximum number of clusters to form.
    similarity_threshold : float
        Threshold for pattern similarity.
    """

    min_motif_length: int = 2
    max_motif_length: int = 5
    min_frequency: int = 3
    min_cluster_size: int = 2
    max_clusters: int = 10
    similarity_threshold: float = 0.7


class SequencePatternDetector:
    """Detect behavioral patterns in action sequences.

    This class identifies recurring motifs, clusters wallets by behavior,
    and provides pattern analysis capabilities.

    Parameters
    ----------
    config : PatternConfig | None
        Detection configuration. Uses defaults if None.
    encoder : WalletActionEncoder | None
        Encoder for computing embeddings. Creates new if None.

    Examples
    --------
    >>> detector = SequencePatternDetector()
    >>> sequences = [seq1, seq2, seq3]  # ActionSequence objects
    >>> motifs = detector.find_motifs(sequences)
    >>> clusters = detector.cluster_behaviors(sequences)
    """

    # Pre-defined cluster labels based on characteristics
    CLUSTER_LABELS: dict[str, str] = {
        "high_buy": "Active Buyer",
        "high_sell": "Active Seller",
        "balanced": "Balanced Trader",
        "lp_focused": "Liquidity Provider",
        "passive": "Passive Holder",
        "churner": "High Frequency Trader",
        "dumper": "Potential Dumper",
        "accumulator": "Accumulator",
    }

    def __init__(
        self,
        config: PatternConfig | None = None,
        encoder: WalletActionEncoder | None = None,
    ) -> None:
        """Initialize pattern detector."""
        self.config = config or PatternConfig()
        self.encoder = encoder or WalletActionEncoder()
        self._fitted_clusters: list[BehaviorCluster] | None = None
        logger.info(
            "pattern_detector_initialized",
            min_motif_length=self.config.min_motif_length,
            max_motif_length=self.config.max_motif_length,
        )

    def find_motifs(
        self,
        sequences: Sequence[ActionSequence],
        top_k: int = 20,
    ) -> list[Motif]:
        """Find recurring behavioral motifs in sequences.

        Parameters
        ----------
        sequences : Sequence[ActionSequence]
            List of action sequences to analyze.
        top_k : int
            Number of top motifs to return.

        Returns
        -------
        list[Motif]
            List of discovered motifs, sorted by frequency.
        """
        if len(sequences) == 0:
            return []

        # Count all n-grams across sequences
        ngram_occurrences: dict[tuple[WalletActionType, ...], list[tuple[WalletAddress, float]]] = defaultdict(list)

        for seq in sequences:
            if len(seq.actions) < self.config.min_motif_length:
                continue

            # Extract n-grams of various lengths
            for n in range(self.config.min_motif_length, self.config.max_motif_length + 1):
                for i in range(len(seq.actions) - n + 1):
                    ngram = seq.actions[i : i + n]
                    position = (i + n / 2) / len(seq.actions)  # Normalized position
                    ngram_occurrences[ngram].append((seq.wallet, position))

        # Convert to motifs
        motifs: list[Motif] = []
        for pattern, occurrences in ngram_occurrences.items():
            if len(occurrences) < self.config.min_frequency:
                continue

            wallets = tuple(set(w for w, _ in occurrences))
            positions = [p for _, p in occurrences]

            # Confidence based on frequency and wallet diversity
            freq_score = min(len(occurrences) / 10, 1.0)
            diversity_score = len(wallets) / len(sequences)
            confidence = (freq_score + diversity_score) / 2

            motif = Motif(
                pattern=pattern,
                frequency=len(occurrences),
                affected_wallets=wallets,
                confidence=confidence,
                avg_position=float(np.mean(positions)),
            )
            motifs.append(motif)

        # Sort by frequency and return top-k
        motifs.sort(key=lambda m: (m.frequency, m.confidence), reverse=True)
        logger.info(
            "motifs_found",
            total_patterns=len(ngram_occurrences),
            qualifying_motifs=len(motifs),
            returned=min(top_k, len(motifs)),
        )

        return motifs[:top_k]

    def cluster_behaviors(
        self,
        sequences: Sequence[ActionSequence],
        method: str = "kmeans",
    ) -> list[BehaviorCluster]:
        """Cluster wallets by behavioral patterns.

        Parameters
        ----------
        sequences : Sequence[ActionSequence]
            List of action sequences to cluster.
        method : str
            Clustering method: "kmeans" or "dbscan".

        Returns
        -------
        list[BehaviorCluster]
            List of behavior clusters.
        """
        if len(sequences) < self.config.min_cluster_size:
            return []

        # Extract embeddings
        embeddings = []
        valid_sequences = []
        for seq in sequences:
            if seq.embedding is not None:
                embeddings.append(seq.embedding)
                valid_sequences.append(seq)

        if len(embeddings) < self.config.min_cluster_size:
            return []

        X = np.array(embeddings)

        # Determine optimal number of clusters
        n_clusters = self._determine_n_clusters(X, method)

        # Perform clustering
        if method == "kmeans":
            labels = self._kmeans_cluster(X, n_clusters)
        elif method == "dbscan":
            labels = self._dbscan_cluster(X)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Build cluster objects
        clusters = self._build_clusters(valid_sequences, X, labels)
        self._fitted_clusters = clusters

        logger.info(
            "clustering_complete",
            method=method,
            n_sequences=len(valid_sequences),
            n_clusters=len(clusters),
        )

        return clusters

    def _determine_n_clusters(
        self, X: npt.NDArray[np.float64], method: str
    ) -> int:
        """Determine optimal number of clusters using silhouette score.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Embedding matrix.
        method : str
            Clustering method.

        Returns
        -------
        int
            Optimal number of clusters.
        """
        if method == "dbscan":
            return 0  # DBSCAN determines clusters automatically

        if len(X) <= self.config.max_clusters:
            return max(2, len(X) // 2)

        best_score = -1
        best_k = 2

        for k in range(2, min(self.config.max_clusters + 1, len(X))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            if len(set(labels)) < 2:
                continue

            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def _kmeans_cluster(
        self, X: npt.NDArray[np.float64], n_clusters: int
    ) -> npt.NDArray[np.int32]:
        """Perform K-means clustering.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Embedding matrix.
        n_clusters : int
            Number of clusters.

        Returns
        -------
        npt.NDArray[np.int32]
            Cluster labels.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(X).astype(np.int32)

    def _dbscan_cluster(
        self, X: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int32]:
        """Perform DBSCAN clustering.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Embedding matrix.

        Returns
        -------
        npt.NDArray[np.int32]
            Cluster labels (-1 for noise).
        """
        dbscan = DBSCAN(
            eps=1 - self.config.similarity_threshold,
            min_samples=self.config.min_cluster_size,
            metric="cosine",
        )
        return dbscan.fit_predict(X).astype(np.int32)

    def _build_clusters(
        self,
        sequences: list[ActionSequence],
        X: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int32],
    ) -> list[BehaviorCluster]:
        """Build cluster objects from labels.

        Parameters
        ----------
        sequences : list[ActionSequence]
            List of sequences.
        X : npt.NDArray[np.float64]
            Embedding matrix.
        labels : npt.NDArray[np.int32]
            Cluster labels.

        Returns
        -------
        list[BehaviorCluster]
            List of behavior clusters.
        """
        clusters: list[BehaviorCluster] = []
        unique_labels = set(labels)

        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise in DBSCAN
                continue

            mask = labels == cluster_id
            cluster_sequences = [seq for seq, m in zip(sequences, mask) if m]

            if len(cluster_sequences) < self.config.min_cluster_size:
                continue

            # Compute centroid
            cluster_embeddings = X[mask]
            centroid = np.mean(cluster_embeddings, axis=0)

            # Compute cohesion (average similarity to centroid)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            cohesion = float(1 - np.mean(distances) / (np.std(distances) + 1e-10))
            cohesion = max(0, min(1, cohesion))

            # Find characteristic actions
            char_actions = self._find_characteristic_actions(cluster_sequences)

            # Assign label
            label = self._assign_cluster_label(cluster_sequences, char_actions)

            clusters.append(
                BehaviorCluster(
                    cluster_id=int(cluster_id),
                    wallets=tuple(seq.wallet for seq in cluster_sequences),
                    centroid=centroid,
                    characteristic_actions=char_actions,
                    cohesion=cohesion,
                    label=label,
                )
            )

        return sorted(clusters, key=lambda c: c.size, reverse=True)

    def _find_characteristic_actions(
        self, sequences: list[ActionSequence]
    ) -> tuple[WalletActionType, ...]:
        """Find most common actions in a group of sequences.

        Parameters
        ----------
        sequences : list[ActionSequence]
            Sequences in the cluster.

        Returns
        -------
        tuple[WalletActionType, ...]
            Top 3 most common action types.
        """
        action_counts: dict[WalletActionType, int] = defaultdict(int)

        for seq in sequences:
            for action in seq.actions:
                action_counts[action] += 1

        sorted_actions = sorted(
            action_counts.items(), key=lambda x: x[1], reverse=True
        )
        return tuple(a for a, _ in sorted_actions[:3])

    def _assign_cluster_label(
        self,
        sequences: list[ActionSequence],
        char_actions: tuple[WalletActionType, ...],
    ) -> str:
        """Assign a human-readable label to a cluster.

        Parameters
        ----------
        sequences : list[ActionSequence]
            Sequences in the cluster.
        char_actions : tuple[WalletActionType, ...]
            Characteristic actions.

        Returns
        -------
        str
            Human-readable cluster label.
        """
        # Count action frequencies across cluster
        total_actions = sum(len(seq.actions) for seq in sequences)
        if total_actions == 0:
            return "Unknown"

        action_counts: dict[WalletActionType, int] = defaultdict(int)
        for seq in sequences:
            for action in seq.actions:
                action_counts[action] += 1

        # Calculate ratios
        buy_ratio = action_counts.get(WalletActionType.SWAP_BUY, 0) / total_actions
        sell_ratio = action_counts.get(WalletActionType.SWAP_SELL, 0) / total_actions
        lp_ratio = (
            action_counts.get(WalletActionType.LP_ADD, 0)
            + action_counts.get(WalletActionType.LP_REMOVE, 0)
        ) / total_actions
        idle_ratio = action_counts.get(WalletActionType.IDLE, 0) / total_actions

        # Assign label based on dominant behavior
        if lp_ratio > 0.3:
            return self.CLUSTER_LABELS["lp_focused"]
        if idle_ratio > 0.5:
            return self.CLUSTER_LABELS["passive"]
        if sell_ratio > 0.4 and buy_ratio < 0.2:
            return self.CLUSTER_LABELS["dumper"]
        if buy_ratio > 0.4 and sell_ratio < 0.2:
            return self.CLUSTER_LABELS["accumulator"]
        if buy_ratio > 0.3 and sell_ratio > 0.3:
            return self.CLUSTER_LABELS["churner"]
        if abs(buy_ratio - sell_ratio) < 0.1:
            return self.CLUSTER_LABELS["balanced"]
        if buy_ratio > sell_ratio:
            return self.CLUSTER_LABELS["high_buy"]
        return self.CLUSTER_LABELS["high_sell"]

    def find_similar_sequences(
        self,
        query: ActionSequence,
        sequences: Sequence[ActionSequence],
        top_k: int = 10,
    ) -> list[tuple[ActionSequence, float]]:
        """Find sequences most similar to a query sequence.

        Parameters
        ----------
        query : ActionSequence
            Query sequence.
        sequences : Sequence[ActionSequence]
            Sequences to search.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[tuple[ActionSequence, float]]
            List of (sequence, similarity) tuples.
        """
        if query.embedding is None:
            raise ValueError("Query sequence must have embedding")

        similarities: list[tuple[ActionSequence, float]] = []

        for seq in sequences:
            if seq.embedding is None:
                continue
            sim = self.encoder.similarity(query, seq)
            similarities.append((seq, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_cluster_for_sequence(
        self, sequence: ActionSequence
    ) -> BehaviorCluster | None:
        """Find which cluster a sequence belongs to.

        Parameters
        ----------
        sequence : ActionSequence
            Sequence to classify.

        Returns
        -------
        BehaviorCluster | None
            Matching cluster or None if no fitted clusters.
        """
        if self._fitted_clusters is None or sequence.embedding is None:
            return None

        best_cluster = None
        best_similarity = -1.0

        for cluster in self._fitted_clusters:
            # Compute similarity to centroid
            dot = np.dot(sequence.embedding, cluster.centroid)
            norm1 = np.linalg.norm(sequence.embedding)
            norm2 = np.linalg.norm(cluster.centroid)

            if norm1 > 0 and norm2 > 0:
                sim = (dot / (norm1 * norm2) + 1) / 2
                if sim > best_similarity:
                    best_similarity = sim
                    best_cluster = cluster

        return best_cluster
