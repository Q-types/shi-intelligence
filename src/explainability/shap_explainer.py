"""
SHAP-based Explainability for SHI Risk Scores.

Provides interpretable explanations for all risk score predictions using SHAP values.
Uses TreeExplainer for tree-based models to identify top feature contributors.

Key Features:
- SHAP value computation for risk scores
- Feature importance ranking
- Uncertainty-aware explanations
- Model-agnostic interface
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import numpy.typing as npt
import structlog

logger = structlog.get_logger()


class ExplanationType(Enum):
    """Type of explanation to generate."""

    RISK_SCORE = "risk_score"  # Overall risk score explanation
    REGIME_CHANGE = "regime_change"  # Regime transition drivers
    ANOMALY_DETECTION = "anomaly_detection"  # Anomaly score drivers
    SELL_PRESSURE = "sell_pressure"  # Sell pressure contributors


@dataclass
class FeatureContribution:
    """Individual feature's contribution to prediction."""

    feature_name: str
    shap_value: float  # SHAP value (additive contribution)
    feature_value: float  # Actual feature value
    baseline_value: float  # Expected value (no contribution)
    contribution_pct: float  # Percentage of total SHAP magnitude


@dataclass
class SHAPExplanation:
    """Complete SHAP explanation for a prediction."""

    predicted_value: float  # Model prediction
    baseline_value: float  # Expected value (no features)
    top_contributors: List[FeatureContribution]  # Most important features
    all_contributions: Dict[str, float]  # All SHAP values

    # Uncertainty bounds
    prediction_std: Optional[float] = None  # Standard deviation of prediction
    confidence_interval: Optional[Tuple[float, float]] = None  # 95% CI

    # Context
    explanation_type: ExplanationType = ExplanationType.RISK_SCORE
    feature_names: Optional[List[str]] = None

    @property
    def total_shap_magnitude(self) -> float:
        """Total absolute SHAP value across all features."""
        return sum(abs(v) for v in self.all_contributions.values())

    @property
    def positive_contributors(self) -> List[FeatureContribution]:
        """Features increasing the prediction."""
        return [f for f in self.top_contributors if f.shap_value > 0]

    @property
    def negative_contributors(self) -> List[FeatureContribution]:
        """Features decreasing the prediction."""
        return [f for f in self.top_contributors if f.shap_value < 0]


class SHAPExplainer:
    """
    SHAP-based explainer for SHI models.

    Uses TreeExplainer for tree-based models (RandomForest, XGBoost).
    Provides feature-level explanations for predictions.

    Note: Requires 'shap' package to be installed.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_type: str = "tree",
        background_data: Optional[npt.NDArray[np.float64]] = None,
    ):
        """
        Initialize SHAP explainer.

        Parameters
        ----------
        model : Any
            Trained model (must support TreeExplainer or KernelExplainer)
        feature_names : List[str]
            Names of features in order
        model_type : str, optional
            Type of model: "tree", "linear", "kernel", by default "tree"
        background_data : Optional[npt.NDArray[np.float64]], optional
            Background dataset for kernel SHAP, by default None
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.background_data = background_data

        # Lazy-load SHAP to avoid import errors
        self._shap_module: Optional[Any] = None
        self._explainer: Optional[Any] = None
        self._fitted = False

    def _ensure_shap_available(self) -> None:
        """Ensure SHAP is available and load it."""
        if self._shap_module is not None:
            return

        try:
            import shap
            self._shap_module = shap
        except ImportError:
            raise ImportError(
                "SHAP is required for explainability. "
                "Install with: pip install shap"
            )

    def fit(self) -> None:
        """
        Initialize SHAP explainer for the model.

        Raises
        ------
        ImportError
            If SHAP is not installed
        ValueError
            If model type is unsupported
        """
        self._ensure_shap_available()
        assert self._shap_module is not None  # Type narrowing for mypy
        shap = self._shap_module

        if self.model_type == "tree":
            # TreeExplainer for tree-based models
            self._explainer = shap.TreeExplainer(self.model)
            logger.info("Initialized SHAP TreeExplainer")

        elif self.model_type == "linear":
            # LinearExplainer for linear models
            self._explainer = shap.LinearExplainer(self.model, self.background_data)
            logger.info("Initialized SHAP LinearExplainer")

        elif self.model_type == "kernel":
            # KernelExplainer for any model (slower)
            if self.background_data is None:
                raise ValueError("background_data required for KernelExplainer")
            self._explainer = shap.KernelExplainer(
                self.model.predict,
                self.background_data
            )
            logger.info("Initialized SHAP KernelExplainer")

        else:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. "
                f"Must be 'tree', 'linear', or 'kernel'"
            )

        self._fitted = True

    def explain(
        self,
        features: npt.NDArray[np.float64],
        top_k: int = 10,
        explanation_type: ExplanationType = ExplanationType.RISK_SCORE,
        uncertainty: bool = False,
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation for a single prediction.

        Parameters
        ----------
        features : npt.NDArray[np.float64]
            Feature vector (1D array)
        top_k : int, optional
            Number of top contributors to return, by default 10
        explanation_type : ExplanationType, optional
            Type of explanation, by default RISK_SCORE
        uncertainty : bool, optional
            Whether to compute uncertainty bounds, by default False

        Returns
        -------
        SHAPExplanation
            Complete explanation with top contributors
        """
        if not self._fitted:
            self.fit()

        assert self._explainer is not None  # Type narrowing for mypy

        # Ensure 2D shape for SHAP
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get SHAP values
        shap_values = self._explainer.shap_values(features)

        # Handle multi-output models (e.g., binary classification)
        if isinstance(shap_values, list):
            # For binary classification, use positive class
            shap_values = shap_values[1]

        # Extract for single instance
        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        # Get prediction
        if hasattr(self.model, "predict_proba"):
            prediction = self.model.predict_proba(features)[0, 1]
        else:
            prediction = self.model.predict(features)[0]

        # Baseline value (expected value)
        if hasattr(self._explainer, "expected_value"):
            baseline = self._explainer.expected_value
            # Handle multi-output
            if isinstance(baseline, (list, np.ndarray)):
                baseline = baseline[1] if len(baseline) > 1 else baseline[0]
        else:
            baseline = 0.0

        # Create feature contributions
        all_contributions = {
            name: float(shap_val)
            for name, shap_val in zip(self.feature_names, shap_values)
        }

        # Sort by absolute SHAP value
        sorted_features = sorted(
            all_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]

        total_magnitude = sum(abs(v) for v in all_contributions.values())

        top_contributors = [
            FeatureContribution(
                feature_name=name,
                shap_value=float(shap_val),
                feature_value=float(features[0, self.feature_names.index(name)]),
                baseline_value=float(baseline),
                contribution_pct=abs(shap_val) / total_magnitude * 100 if total_magnitude > 0 else 0.0,
            )
            for name, shap_val in sorted_features
        ]

        # Compute uncertainty if requested
        prediction_std = None
        confidence_interval = None

        if uncertainty and hasattr(self.model, "estimators_"):
            # For ensemble models, use variance across trees
            tree_predictions = np.array([
                tree.predict(features)[0]
                for tree in self.model.estimators_
            ])
            prediction_std = float(np.std(tree_predictions))

            # 95% confidence interval
            ci_lower = float(prediction - 1.96 * prediction_std)
            ci_upper = float(prediction + 1.96 * prediction_std)
            confidence_interval = (ci_lower, ci_upper)

        return SHAPExplanation(
            predicted_value=float(prediction),
            baseline_value=float(baseline),
            top_contributors=top_contributors,
            all_contributions=all_contributions,
            prediction_std=prediction_std,
            confidence_interval=confidence_interval,
            explanation_type=explanation_type,
            feature_names=self.feature_names,
        )

    def explain_batch(
        self,
        features: npt.NDArray[np.float64],
        top_k: int = 10,
        explanation_type: ExplanationType = ExplanationType.RISK_SCORE,
    ) -> List[SHAPExplanation]:
        """
        Generate SHAP explanations for multiple predictions.

        Parameters
        ----------
        features : npt.NDArray[np.float64]
            Feature matrix (2D array, n_samples x n_features)
        top_k : int, optional
            Number of top contributors per sample, by default 10
        explanation_type : ExplanationType, optional
            Type of explanation, by default RISK_SCORE

        Returns
        -------
        List[SHAPExplanation]
            Explanations for each sample
        """
        return [
            self.explain(
                features[i],
                top_k=top_k,
                explanation_type=explanation_type
            )
            for i in range(features.shape[0])
        ]

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Compute global feature importance (mean absolute SHAP value).

        Requires background_data to be set.

        Returns
        -------
        Dict[str, float]
            Feature name to importance mapping (sorted)
        """
        if not self._fitted:
            self.fit()

        assert self._explainer is not None  # Type narrowing for mypy

        if self.background_data is None:
            raise ValueError("background_data required for global feature importance")

        shap_values = self._explainer.shap_values(self.background_data)

        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance = {
            name: float(value)
            for name, value in zip(self.feature_names, mean_abs_shap)
        }

        # Sort by importance
        importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        return importance


def create_mock_explainer(
    feature_names: List[str],
    baseline_value: float = 0.5,
) -> "MockSHAPExplainer":
    """
    Create a mock SHAP explainer for testing without SHAP dependency.

    Parameters
    ----------
    feature_names : List[str]
        Feature names
    baseline_value : float, optional
        Baseline prediction value, by default 0.5

    Returns
    -------
    MockSHAPExplainer
        Mock explainer that generates synthetic SHAP values
    """
    return MockSHAPExplainer(feature_names, baseline_value)


class MockSHAPExplainer:
    """
    Mock SHAP explainer for testing and development.

    Generates synthetic SHAP values based on feature statistics.
    """

    def __init__(self, feature_names: List[str], baseline_value: float = 0.5):
        """Initialize mock explainer."""
        self.feature_names = feature_names
        self.baseline_value = baseline_value
        self._fitted = True

    def fit(self) -> None:
        """Mock fit (no-op)."""
        pass

    def explain(
        self,
        features: npt.NDArray[np.float64],
        top_k: int = 10,
        explanation_type: ExplanationType = ExplanationType.RISK_SCORE,
        uncertainty: bool = False,
    ) -> SHAPExplanation:
        """Generate mock SHAP explanation."""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Synthetic SHAP values (proportional to feature magnitude)
        shap_values = (features[0] - 0.5) * 0.2  # Scale to ±0.1 range

        # Add some noise
        np.random.seed(42)
        shap_values += np.random.normal(0, 0.01, size=len(shap_values))

        # Synthetic prediction
        prediction = self.baseline_value + np.sum(shap_values)
        prediction = np.clip(prediction, 0.0, 1.0)

        # Create contributions
        all_contributions = {
            name: float(shap_val)
            for name, shap_val in zip(self.feature_names, shap_values)
        }

        sorted_features = sorted(
            all_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]

        total_magnitude = sum(abs(v) for v in all_contributions.values())

        top_contributors = [
            FeatureContribution(
                feature_name=name,
                shap_value=float(shap_val),
                feature_value=float(features[0, self.feature_names.index(name)]),
                baseline_value=self.baseline_value,
                contribution_pct=abs(shap_val) / total_magnitude * 100 if total_magnitude > 0 else 0.0,
            )
            for name, shap_val in sorted_features
        ]

        # Mock uncertainty
        prediction_std = 0.05 if uncertainty else None
        confidence_interval = (
            (prediction - 0.1, prediction + 0.1) if uncertainty else None
        )

        return SHAPExplanation(
            predicted_value=float(prediction),
            baseline_value=self.baseline_value,
            top_contributors=top_contributors,
            all_contributions=all_contributions,
            prediction_std=prediction_std,
            confidence_interval=confidence_interval,
            explanation_type=explanation_type,
            feature_names=self.feature_names,
        )
