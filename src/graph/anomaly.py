"""
Wallet Anomaly Detection for SHI.

Uses Isolation Forest on:
- Graph embeddings
- Structural features
- Behavioral patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .embeddings import GraphEmbedder
from .funding_graph import FundingGraph
from ..core.types import WalletAddress

logger = structlog.get_logger()


@dataclass
class AnomalyScore:
    """Anomaly score for a wallet."""

    wallet: WalletAddress
    score: float  # Anomaly score in [-1, 1], where -1 is most anomalous
    is_anomalous: bool
    confidence: float  # Confidence in [0, 1]
    feature_contributions: Dict[str, float]


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""

    contamination: float = 0.05  # Expected proportion of anomalies
    n_estimators: int = 100
    max_features: float = 1.0
    random_state: int = 42
    threshold: float = -0.5  # Score below this is considered anomalous


class WalletAnomalyDetector:
    """
    Detects anomalous wallets using Isolation Forest.

    Combines multiple feature types:
    - Graph embeddings (Node2Vec)
    - Structural features (degree, centrality, community)
    - Behavioral features (funding patterns, timing)
    """

    def __init__(
        self,
        embedder: GraphEmbedder,
        graph: FundingGraph,
        config: Optional[AnomalyConfig] = None,
    ):
        """
        Initialize anomaly detector.

        Args:
            embedder: Fitted GraphEmbedder
            graph: Funding graph
            config: Anomaly detection configuration
        """
        self.embedder = embedder
        self.graph = graph
        self.config = config or AnomalyConfig()

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.fitted: bool = False

    def extract_features(
        self,
        wallets: List[WalletAddress],
        include_embeddings: bool = True,
    ) -> Tuple[np.ndarray, List[WalletAddress], List[str]]:
        """
        Extract features for anomaly detection.

        Args:
            wallets: Wallets to extract features for
            include_embeddings: Whether to include embedding features

        Returns:
            (feature_matrix, valid_wallets, feature_names) tuple
        """
        features = []
        valid_wallets = []
        feature_names = []

        for wallet in wallets:
            wallet_features = []

            # Structural features
            in_degree = self.graph.get_in_degree(wallet)
            out_degree = self.graph.get_out_degree(wallet)
            ancestors = self.graph.get_ancestors(wallet, max_depth=3)
            funders = self.graph.get_funders(wallet)
            funded = self.graph.get_funded_by(wallet)

            structural = [
                in_degree,
                out_degree,
                len(ancestors),
                len(funders),
                len(funded),
                out_degree / (in_degree + 1),  # Funding ratio
            ]
            wallet_features.extend(structural)

            if not feature_names:
                feature_names.extend(
                    [
                        "in_degree",
                        "out_degree",
                        "ancestor_count",
                        "funder_count",
                        "funded_count",
                        "funding_ratio",
                    ]
                )

            # Embedding features (if available)
            if include_embeddings:
                embedding = self.embedder.get_embedding(wallet)
                if embedding is None:
                    continue  # Skip wallets without embeddings

                wallet_features.extend(embedding.tolist())

                if not feature_names or len(feature_names) == 6:
                    feature_names.extend([f"emb_{i}" for i in range(len(embedding))])

            features.append(wallet_features)
            valid_wallets.append(wallet)

        if not features:
            return np.array([]), [], []

        feature_matrix = np.array(features)

        return feature_matrix, valid_wallets, feature_names

    def fit(
        self,
        wallets: List[WalletAddress],
        include_embeddings: bool = True,
    ) -> None:
        """
        Fit Isolation Forest on wallet features.

        Args:
            wallets: Wallets to fit on
            include_embeddings: Whether to use embeddings
        """
        logger.info("fitting_anomaly_detector", wallet_count=len(wallets))

        # Extract features
        X, valid_wallets, feature_names = self.extract_features(wallets, include_embeddings)

        if X.shape[0] == 0:
            raise ValueError("No valid features extracted")

        self.feature_names = feature_names

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit Isolation Forest
        self.model = IsolationForest(
            contamination=self.config.contamination,
            n_estimators=self.config.n_estimators,
            max_features=self.config.max_features,
            random_state=self.config.random_state,
        )

        self.model.fit(X_scaled)
        self.fitted = True

        logger.info(
            "anomaly_detector_fitted",
            wallet_count=len(valid_wallets),
            feature_count=len(feature_names),
        )

    def predict(
        self,
        wallet: WalletAddress,
        include_embeddings: bool = True,
    ) -> Optional[AnomalyScore]:
        """
        Predict anomaly score for a wallet.

        Args:
            wallet: Wallet to score
            include_embeddings: Whether to use embeddings

        Returns:
            AnomalyScore or None if wallet cannot be scored
        """
        if not self.fitted or self.model is None or self.scaler is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        # Extract features
        X, valid_wallets, _ = self.extract_features([wallet], include_embeddings)

        if X.shape[0] == 0:
            return None

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict anomaly score
        score = self.model.score_samples(X_scaled)[0]
        label = self.model.predict(X_scaled)[0]  # -1 = anomaly, 1 = normal

        is_anomalous = score < self.config.threshold or label == -1

        # Compute confidence (distance from threshold)
        confidence = abs(score - self.config.threshold)
        confidence = min(confidence / 0.5, 1.0)  # Normalize to [0, 1]

        # Feature contributions (approximate)
        feature_contribs = self._compute_feature_contributions(X[0])

        anomaly_score = AnomalyScore(
            wallet=wallet,
            score=float(score),
            is_anomalous=is_anomalous,
            confidence=confidence,
            feature_contributions=feature_contribs,
        )

        return anomaly_score

    def predict_batch(
        self,
        wallets: List[WalletAddress],
        include_embeddings: bool = True,
    ) -> List[AnomalyScore]:
        """
        Predict anomaly scores for multiple wallets.

        Args:
            wallets: Wallets to score
            include_embeddings: Whether to use embeddings

        Returns:
            List of AnomalyScore objects
        """
        if not self.fitted or self.model is None or self.scaler is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        # Extract features
        X, valid_wallets, _ = self.extract_features(wallets, include_embeddings)

        if X.shape[0] == 0:
            return []

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict anomaly scores
        scores = self.model.score_samples(X_scaled)
        labels = self.model.predict(X_scaled)

        # Build AnomalyScore objects
        results = []
        for i, wallet in enumerate(valid_wallets):
            score = scores[i]
            label = labels[i]

            is_anomalous = score < self.config.threshold or label == -1
            confidence = min(abs(score - self.config.threshold) / 0.5, 1.0)

            feature_contribs = self._compute_feature_contributions(X[i])

            anomaly_score = AnomalyScore(
                wallet=wallet,
                score=float(score),
                is_anomalous=is_anomalous,
                confidence=confidence,
                feature_contributions=feature_contribs,
            )

            results.append(anomaly_score)

        logger.info(
            "anomaly_scores_predicted",
            wallet_count=len(results),
            anomalous_count=sum(1 for r in results if r.is_anomalous),
        )

        return results

    def _compute_feature_contributions(
        self,
        feature_vector: np.ndarray,
    ) -> Dict[str, float]:
        """
        Approximate feature contributions to anomaly score.

        Uses feature magnitude as proxy for contribution.

        Args:
            feature_vector: Feature values for a wallet

        Returns:
            Dict mapping feature name -> contribution score
        """
        if not self.feature_names:
            return {}

        # Normalize feature values
        feature_magnitudes = np.abs(feature_vector)
        total = np.sum(feature_magnitudes)

        if total == 0:
            return {name: 0.0 for name in self.feature_names}

        contributions = feature_magnitudes / total

        return {
            name: float(contrib) for name, contrib in zip(self.feature_names, contributions)
        }

    def find_anomalies(
        self,
        wallets: List[WalletAddress],
        top_k: Optional[int] = None,
        include_embeddings: bool = True,
    ) -> List[AnomalyScore]:
        """
        Find most anomalous wallets.

        Args:
            wallets: Wallets to analyze
            top_k: Return only top k anomalies (None = all anomalies)
            include_embeddings: Whether to use embeddings

        Returns:
            List of AnomalyScore objects sorted by anomaly score (most anomalous first)
        """
        scores = self.predict_batch(wallets, include_embeddings)

        # Filter to anomalies
        anomalies = [s for s in scores if s.is_anomalous]

        # Sort by score (ascending, since lower = more anomalous)
        anomalies.sort(key=lambda x: x.score)

        if top_k:
            anomalies = anomalies[:top_k]

        return anomalies

    def get_anomaly_distribution(
        self,
        wallets: List[WalletAddress],
        include_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """
        Get distribution statistics for anomaly scores.

        Args:
            wallets: Wallets to analyze
            include_embeddings: Whether to use embeddings

        Returns:
            Dict with distribution statistics
        """
        scores = self.predict_batch(wallets, include_embeddings)

        if not scores:
            return {}

        score_values = [s.score for s in scores]
        anomalous_count = sum(1 for s in scores if s.is_anomalous)

        distribution = {
            "total_wallets": len(scores),
            "anomalous_count": anomalous_count,
            "anomalous_percentage": anomalous_count / len(scores) * 100,
            "mean_score": float(np.mean(score_values)),
            "median_score": float(np.median(score_values)),
            "std_score": float(np.std(score_values)),
            "min_score": float(np.min(score_values)),
            "max_score": float(np.max(score_values)),
            "threshold": self.config.threshold,
        }

        return distribution

    def export_scores(
        self,
        scores: List[AnomalyScore],
    ) -> pd.DataFrame:
        """
        Export anomaly scores as DataFrame.

        Args:
            scores: List of AnomalyScore objects

        Returns:
            Pandas DataFrame with scores
        """
        if not scores:
            return pd.DataFrame()

        data = []
        for score in scores:
            row = {
                "wallet": score.wallet,
                "anomaly_score": score.score,
                "is_anomalous": score.is_anomalous,
                "confidence": score.confidence,
            }

            # Add top feature contributions
            sorted_features = sorted(
                score.feature_contributions.items(), key=lambda x: x[1], reverse=True
            )
            for i, (feature, contrib) in enumerate(sorted_features[:5]):
                row[f"top_feature_{i+1}"] = feature
                row[f"top_contrib_{i+1}"] = contrib

            data.append(row)

        return pd.DataFrame(data)
