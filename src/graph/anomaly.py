"""
Wallet Anomaly Detection for SHI.

Uses Isolation Forest on:
- Graph embeddings
- Structural features
- Behavioral patterns

Feature contributions computed via SHAP TreeExplainer for
interpretable anomaly explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .embeddings import GraphEmbedder
from .funding_graph import FundingGraph
from ..core.types import WalletAddress

# SHAP is optional but provides better explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

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

    # SHAP explanation settings
    use_shap: bool = True  # Use SHAP for feature contributions when available
    shap_background_samples: int = 100  # Background samples for SHAP (more = slower but more accurate)
    shap_check_additivity: bool = False  # Disable additivity check for speed


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

        # SHAP explainer for feature contributions
        self._shap_explainer: Optional[Any] = None
        self._shap_background: Optional[np.ndarray] = None
        self._shap_available: bool = SHAP_AVAILABLE and config.use_shap if config else SHAP_AVAILABLE

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

        # Initialize SHAP explainer for feature contributions
        self._init_shap_explainer(X_scaled)

        logger.info(
            "anomaly_detector_fitted",
            wallet_count=len(valid_wallets),
            feature_count=len(feature_names),
            shap_enabled=self._shap_explainer is not None,
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

        # Feature contributions via SHAP (or fallback to magnitude)
        feature_contribs = self._compute_feature_contributions(X[0], X_scaled[0])

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

            # Feature contributions via SHAP (or fallback to magnitude)
            feature_contribs = self._compute_feature_contributions(X[i], X_scaled[i])

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

    def _init_shap_explainer(self, X_scaled: np.ndarray) -> None:
        """
        Initialize SHAP explainer for feature contributions.

        Uses TreeExplainer for Isolation Forest, which provides exact
        Shapley values based on the tree structure.

        Args:
            X_scaled: Scaled training data for background samples
        """
        if not self._shap_available or self.model is None:
            return

        try:
            # Sample background data for SHAP (speeds up computation)
            n_background = min(self.config.shap_background_samples, X_scaled.shape[0])
            indices = np.random.choice(X_scaled.shape[0], n_background, replace=False)
            self._shap_background = X_scaled[indices]

            # TreeExplainer works directly with Isolation Forest
            # Note: IsolationForest uses the average path length as the anomaly score,
            # so we explain the decision function
            self._shap_explainer = shap.TreeExplainer(
                self.model,
                data=self._shap_background,
                feature_perturbation="interventional",
                model_output="raw",  # Use raw output for IsolationForest
            )

            logger.info(
                "shap_explainer_initialized",
                background_samples=n_background,
                feature_count=len(self.feature_names),
            )

        except Exception as e:
            logger.warning(
                "shap_explainer_init_failed",
                error=str(e),
                falling_back_to_magnitude=True,
            )
            self._shap_explainer = None

    def _compute_feature_contributions(
        self,
        feature_vector: np.ndarray,
        scaled_vector: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute feature contributions to anomaly score using SHAP.

        Uses SHAP TreeExplainer when available for accurate feature attribution.
        Falls back to feature magnitude when SHAP is unavailable.

        Args:
            feature_vector: Raw feature values for a wallet
            scaled_vector: Pre-scaled feature values (optional, for efficiency)

        Returns:
            Dict mapping feature name -> contribution score (positive = more anomalous)
        """
        if not self.feature_names:
            return {}

        # Try SHAP first
        if self._shap_explainer is not None and self.scaler is not None:
            try:
                # Scale the vector if not already scaled
                if scaled_vector is None:
                    scaled_vector = self.scaler.transform(feature_vector.reshape(1, -1))[0]

                # Compute SHAP values
                shap_values = self._shap_explainer.shap_values(
                    scaled_vector.reshape(1, -1),
                    check_additivity=self.config.shap_check_additivity,
                )

                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # Binary classifier returns [class_0, class_1]
                    shap_vals = shap_values[0][0] if len(shap_values) > 0 else shap_values[0]
                else:
                    shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values

                # For anomaly detection, negative SHAP values indicate
                # features pushing toward anomaly (lower isolation score)
                # We flip the sign so positive = more anomalous
                contributions = -np.array(shap_vals)

                # Normalize to sum to 1 for interpretability
                total = np.sum(np.abs(contributions))
                if total > 0:
                    contributions = contributions / total

                return {
                    name: float(contrib)
                    for name, contrib in zip(self.feature_names, contributions)
                }

            except Exception as e:
                logger.debug("shap_computation_failed", error=str(e))
                # Fall through to magnitude-based method

        # Fallback: Use feature magnitude as proxy for contribution
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

    def get_global_feature_importance(
        self,
        wallets: List[WalletAddress],
        include_embeddings: bool = True,
    ) -> Dict[str, float]:
        """
        Get global feature importance across all wallets using SHAP.

        Computes mean absolute SHAP values across all samples to identify
        which features are most important for anomaly detection overall.

        Args:
            wallets: Wallets to compute importance for
            include_embeddings: Whether to use embeddings

        Returns:
            Dict mapping feature name -> mean absolute SHAP value
        """
        if not self.fitted or self.model is None or self.scaler is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        if not self._shap_available or self._shap_explainer is None:
            logger.warning("shap_not_available_for_global_importance")
            return {}

        # Extract and scale features
        X, valid_wallets, _ = self.extract_features(wallets, include_embeddings)

        if X.shape[0] == 0:
            return {}

        X_scaled = self.scaler.transform(X)

        try:
            # Compute SHAP values for all samples
            shap_values = self._shap_explainer.shap_values(
                X_scaled,
                check_additivity=self.config.shap_check_additivity,
            )

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_matrix = shap_values[0] if len(shap_values) > 0 else np.array(shap_values)
            else:
                shap_matrix = shap_values

            # Compute mean absolute SHAP values per feature
            mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

            # Normalize to percentages
            total = np.sum(mean_abs_shap)
            if total > 0:
                mean_abs_shap = mean_abs_shap / total

            importance = {
                name: float(val)
                for name, val in zip(self.feature_names, mean_abs_shap)
            }

            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            logger.info(
                "global_feature_importance_computed",
                wallet_count=len(valid_wallets),
                top_feature=list(importance.keys())[0] if importance else None,
            )

            return importance

        except Exception as e:
            logger.error("global_feature_importance_failed", error=str(e))
            return {}

    def explain_anomaly(
        self,
        wallet: WalletAddress,
        top_k: int = 5,
        include_embeddings: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed explanation for why a wallet is anomalous.

        Returns the top contributing features with their SHAP values
        and a human-readable explanation.

        Args:
            wallet: Wallet to explain
            top_k: Number of top features to include
            include_embeddings: Whether to use embeddings

        Returns:
            Dict with explanation details or None if wallet cannot be explained
        """
        score = self.predict(wallet, include_embeddings)
        if score is None:
            return None

        # Get sorted feature contributions
        sorted_features = sorted(
            score.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_k]

        # Build explanation
        explanation = {
            "wallet": wallet,
            "anomaly_score": score.score,
            "is_anomalous": score.is_anomalous,
            "confidence": score.confidence,
            "explanation_method": "shap" if self._shap_explainer else "magnitude",
            "top_features": [
                {
                    "feature": name,
                    "contribution": contrib,
                    "direction": "anomalous" if contrib > 0 else "normal",
                }
                for name, contrib in sorted_features
            ],
        }

        # Generate human-readable summary
        if score.is_anomalous and sorted_features:
            top_feature, top_contrib = sorted_features[0]
            explanation["summary"] = (
                f"Wallet {wallet[:8]}... is {score.confidence:.0%} likely anomalous. "
                f"Primary driver: {top_feature} (contribution: {top_contrib:.3f})"
            )
        else:
            explanation["summary"] = f"Wallet {wallet[:8]}... appears normal."

        return explanation
