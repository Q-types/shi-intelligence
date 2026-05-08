"""Wallet action sequence encoding for behavioral analysis.

This module provides encoding of wallet actions into numerical sequences
suitable for pattern detection and similarity comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence

import numpy as np
import numpy.typing as npt
import structlog

from src.core.types import WalletAddress

logger = structlog.get_logger()


class WalletActionType(Enum):
    """Types of wallet actions that can be encoded."""

    FUNDED = "funded"
    SWAP_BUY = "swap_buy"
    SWAP_SELL = "swap_sell"
    LP_ADD = "lp_add"
    LP_REMOVE = "lp_remove"
    IDLE = "idle"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"

    @classmethod
    def from_string(cls, action: str) -> WalletActionType:
        """Convert string to action type.

        Parameters
        ----------
        action : str
            Action string (case insensitive).

        Returns
        -------
        WalletActionType
            Corresponding action type.

        Raises
        ------
        ValueError
            If action string is not recognized.
        """
        normalized = action.lower().strip()
        for action_type in cls:
            if action_type.value == normalized:
                return action_type
        raise ValueError(f"Unknown action type: {action}")


@dataclass(frozen=True)
class ActionSequence:
    """Encoded action sequence for a wallet.

    Attributes
    ----------
    wallet : WalletAddress
        Wallet address this sequence belongs to.
    actions : tuple[WalletActionType, ...]
        Sequence of action types.
    encoded : npt.NDArray[np.int32]
        Numerical encoding of actions.
    timestamps : tuple[datetime, ...] | None
        Optional timestamps for each action.
    embedding : npt.NDArray[np.float64] | None
        Optional embedding vector for similarity comparison.
    """

    wallet: WalletAddress
    actions: tuple[WalletActionType, ...]
    encoded: npt.NDArray[np.int32]
    timestamps: tuple[datetime, ...] | None = None
    embedding: npt.NDArray[np.float64] | None = None

    def __len__(self) -> int:
        """Return sequence length."""
        return len(self.actions)

    @property
    def action_names(self) -> list[str]:
        """Return list of action names."""
        return [a.value for a in self.actions]


@dataclass
class EncoderConfig:
    """Configuration for action encoder.

    Attributes
    ----------
    embedding_dim : int
        Dimension of embedding vectors.
    window_size : int
        Window size for n-gram features.
    use_position_encoding : bool
        Whether to add positional encoding.
    """

    embedding_dim: int = 32
    window_size: int = 3
    use_position_encoding: bool = True


class WalletActionEncoder:
    """Encode wallet actions as numerical sequences.

    This class converts wallet action strings into numerical sequences
    and computes embeddings for similarity comparison.

    Parameters
    ----------
    config : EncoderConfig | None
        Encoder configuration. Uses defaults if None.

    Examples
    --------
    >>> encoder = WalletActionEncoder()
    >>> actions = ["funded", "swap_buy", "idle", "swap_sell"]
    >>> sequence = encoder.encode_sequence("wallet123", actions)
    >>> sequence.encoded
    array([0, 1, 5, 2], dtype=int32)
    """

    # Mapping from action type to integer ID
    ACTION_TO_ID: dict[WalletActionType, int] = {
        action: i for i, action in enumerate(WalletActionType)
    }
    ID_TO_ACTION: dict[int, WalletActionType] = {
        i: action for action, i in ACTION_TO_ID.items()
    }
    NUM_ACTIONS: int = len(WalletActionType)

    def __init__(self, config: EncoderConfig | None = None) -> None:
        """Initialize encoder with configuration."""
        self.config = config or EncoderConfig()
        self._embedding_matrix: npt.NDArray[np.float64] | None = None
        self._initialized = False
        logger.info(
            "encoder_initialized",
            embedding_dim=self.config.embedding_dim,
            window_size=self.config.window_size,
        )

    def encode_sequence(
        self,
        wallet: WalletAddress,
        actions: Sequence[str],
        timestamps: Sequence[datetime] | None = None,
        compute_embedding: bool = True,
    ) -> ActionSequence:
        """Encode action sequence for a wallet.

        Parameters
        ----------
        wallet : WalletAddress
            Wallet address.
        actions : Sequence[str]
            List of action strings.
        timestamps : Sequence[datetime] | None
            Optional timestamps for each action.
        compute_embedding : bool
            Whether to compute embedding vector.

        Returns
        -------
        ActionSequence
            Encoded action sequence.

        Raises
        ------
        ValueError
            If timestamps length doesn't match actions length.
        """
        if timestamps is not None and len(timestamps) != len(actions):
            raise ValueError(
                f"Timestamps length ({len(timestamps)}) must match "
                f"actions length ({len(actions)})"
            )

        # Convert strings to action types
        action_types = tuple(WalletActionType.from_string(a) for a in actions)

        # Encode as integers
        encoded = np.array(
            [self.ACTION_TO_ID[a] for a in action_types], dtype=np.int32
        )

        # Compute embedding if requested
        embedding = None
        if compute_embedding and len(encoded) > 0:
            embedding = self.embed_sequence(encoded)

        return ActionSequence(
            wallet=wallet,
            actions=action_types,
            encoded=encoded,
            timestamps=tuple(timestamps) if timestamps else None,
            embedding=embedding,
        )

    def encode_batch(
        self,
        wallets: Sequence[WalletAddress],
        action_sequences: Sequence[Sequence[str]],
        compute_embedding: bool = True,
    ) -> list[ActionSequence]:
        """Encode multiple sequences in batch.

        Parameters
        ----------
        wallets : Sequence[WalletAddress]
            List of wallet addresses.
        action_sequences : Sequence[Sequence[str]]
            List of action sequences.
        compute_embedding : bool
            Whether to compute embeddings.

        Returns
        -------
        list[ActionSequence]
            List of encoded sequences.
        """
        if len(wallets) != len(action_sequences):
            raise ValueError("Wallets and action_sequences must have same length")

        return [
            self.encode_sequence(wallet, actions, compute_embedding=compute_embedding)
            for wallet, actions in zip(wallets, action_sequences)
        ]

    def embed_sequence(self, encoded: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
        """Compute embedding vector for encoded sequence.

        Uses n-gram frequency features with optional position encoding
        to create a fixed-size embedding regardless of sequence length.

        Parameters
        ----------
        encoded : npt.NDArray[np.int32]
            Encoded action sequence.

        Returns
        -------
        npt.NDArray[np.float64]
            Embedding vector of shape (embedding_dim,).
        """
        if len(encoded) == 0:
            return np.zeros(self.config.embedding_dim, dtype=np.float64)

        features: list[float] = []

        # 1. Action frequency features (normalized histogram)
        hist, _ = np.histogram(
            encoded, bins=self.NUM_ACTIONS, range=(0, self.NUM_ACTIONS)
        )
        freq = hist.astype(np.float64) / max(len(encoded), 1)
        features.extend(freq)

        # 2. N-gram features (bigrams and trigrams)
        if len(encoded) >= 2:
            bigrams = self._compute_ngram_features(encoded, n=2)
            features.extend(bigrams)
        else:
            features.extend([0.0] * (self.NUM_ACTIONS * self.NUM_ACTIONS))

        # 3. Transition features
        transition_feat = self._compute_transition_features(encoded)
        features.extend(transition_feat)

        # 4. Position-based features (where do certain actions occur?)
        if self.config.use_position_encoding:
            position_feat = self._compute_position_features(encoded)
            features.extend(position_feat)

        # 5. Summary statistics
        features.extend([
            len(encoded) / 100.0,  # Normalized length
            len(set(encoded)) / self.NUM_ACTIONS,  # Action diversity
            self._compute_entropy(encoded),  # Sequence entropy
        ])

        # Convert to array and reduce to embedding_dim if needed
        feature_array = np.array(features, dtype=np.float64)

        # Project to fixed embedding dimension
        return self._project_to_embedding(feature_array)

    def _compute_ngram_features(
        self, encoded: npt.NDArray[np.int32], n: int
    ) -> list[float]:
        """Compute n-gram frequency features.

        Parameters
        ----------
        encoded : npt.NDArray[np.int32]
            Encoded sequence.
        n : int
            N-gram size.

        Returns
        -------
        list[float]
            N-gram frequency features.
        """
        if len(encoded) < n:
            return [0.0] * (self.NUM_ACTIONS ** n)

        # Count n-grams
        ngram_counts: dict[tuple[int, ...], int] = {}
        for i in range(len(encoded) - n + 1):
            ngram = tuple(encoded[i : i + n].tolist())
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        # Flatten to feature vector (only keep most common for n>2)
        total = len(encoded) - n + 1
        if n == 2:
            # Full bigram matrix
            features = [0.0] * (self.NUM_ACTIONS * self.NUM_ACTIONS)
            for (a, b), count in ngram_counts.items():
                idx = a * self.NUM_ACTIONS + b
                features[idx] = count / total
        else:
            # For trigrams, use top-k features
            top_k = min(32, len(ngram_counts))
            sorted_ngrams = sorted(
                ngram_counts.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
            features = [count / total for _, count in sorted_ngrams]
            features.extend([0.0] * (32 - len(features)))

        return features

    def _compute_transition_features(
        self, encoded: npt.NDArray[np.int32]
    ) -> list[float]:
        """Compute transition probability features.

        Parameters
        ----------
        encoded : npt.NDArray[np.int32]
            Encoded sequence.

        Returns
        -------
        list[float]
            Transition features.
        """
        if len(encoded) < 2:
            return [0.0] * 5

        # Count specific transitions of interest
        transitions = {
            "buy_to_sell": 0,
            "sell_to_buy": 0,
            "funded_to_sell": 0,
            "any_to_idle": 0,
            "idle_to_any": 0,
        }

        buy_id = self.ACTION_TO_ID[WalletActionType.SWAP_BUY]
        sell_id = self.ACTION_TO_ID[WalletActionType.SWAP_SELL]
        funded_id = self.ACTION_TO_ID[WalletActionType.FUNDED]
        idle_id = self.ACTION_TO_ID[WalletActionType.IDLE]

        for i in range(len(encoded) - 1):
            curr, next_action = encoded[i], encoded[i + 1]

            if curr == buy_id and next_action == sell_id:
                transitions["buy_to_sell"] += 1
            elif curr == sell_id and next_action == buy_id:
                transitions["sell_to_buy"] += 1
            elif curr == funded_id and next_action == sell_id:
                transitions["funded_to_sell"] += 1
            if next_action == idle_id:
                transitions["any_to_idle"] += 1
            if curr == idle_id and next_action != idle_id:
                transitions["idle_to_any"] += 1

        total = len(encoded) - 1
        return [v / total for v in transitions.values()]

    def _compute_position_features(
        self, encoded: npt.NDArray[np.int32]
    ) -> list[float]:
        """Compute position-based features.

        Parameters
        ----------
        encoded : npt.NDArray[np.int32]
            Encoded sequence.

        Returns
        -------
        list[float]
            Position features indicating where actions occur.
        """
        n = len(encoded)
        if n == 0:
            return [0.0] * (self.NUM_ACTIONS * 3)

        features = []

        # For each action type, compute: first position, last position, mean position
        for action_id in range(self.NUM_ACTIONS):
            positions = np.where(encoded == action_id)[0]
            if len(positions) > 0:
                features.extend([
                    positions[0] / n,  # First occurrence (normalized)
                    positions[-1] / n,  # Last occurrence (normalized)
                    np.mean(positions) / n,  # Mean position
                ])
            else:
                features.extend([-1.0, -1.0, -1.0])  # Not present

        return features

    def _compute_entropy(self, encoded: npt.NDArray[np.int32]) -> float:
        """Compute Shannon entropy of action distribution.

        Parameters
        ----------
        encoded : npt.NDArray[np.int32]
            Encoded sequence.

        Returns
        -------
        float
            Normalized entropy (0 to 1).
        """
        if len(encoded) == 0:
            return 0.0

        _, counts = np.unique(encoded, return_counts=True)
        probs = counts / len(encoded)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Normalize by max entropy
        max_entropy = np.log2(self.NUM_ACTIONS)
        return float(entropy / max_entropy)

    def _project_to_embedding(
        self, features: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Project feature vector to fixed embedding dimension.

        Uses random projection for dimensionality reduction.

        Parameters
        ----------
        features : npt.NDArray[np.float64]
            Feature vector.

        Returns
        -------
        npt.NDArray[np.float64]
            Embedding of shape (embedding_dim,).
        """
        feature_dim = len(features)

        # Initialize projection matrix if needed
        if self._embedding_matrix is None or self._embedding_matrix.shape[0] != feature_dim:
            # Use random projection matrix (fixed seed for reproducibility)
            rng = np.random.default_rng(42)
            self._embedding_matrix = rng.standard_normal(
                (feature_dim, self.config.embedding_dim)
            ) / np.sqrt(feature_dim)

        # Project and normalize
        embedding = features @ self._embedding_matrix
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float64)

    def decode_sequence(
        self, encoded: npt.NDArray[np.int32]
    ) -> list[WalletActionType]:
        """Decode numerical sequence back to action types.

        Parameters
        ----------
        encoded : npt.NDArray[np.int32]
            Encoded sequence.

        Returns
        -------
        list[WalletActionType]
            List of action types.
        """
        return [self.ID_TO_ACTION[int(i)] for i in encoded]

    def similarity(
        self, seq1: ActionSequence, seq2: ActionSequence
    ) -> float:
        """Compute cosine similarity between two sequences.

        Parameters
        ----------
        seq1 : ActionSequence
            First sequence.
        seq2 : ActionSequence
            Second sequence.

        Returns
        -------
        float
            Cosine similarity (0 to 1).
        """
        if seq1.embedding is None or seq2.embedding is None:
            raise ValueError("Both sequences must have embeddings")

        dot = np.dot(seq1.embedding, seq2.embedding)
        norm1 = np.linalg.norm(seq1.embedding)
        norm2 = np.linalg.norm(seq2.embedding)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float((dot / (norm1 * norm2) + 1) / 2)  # Scale to 0-1
