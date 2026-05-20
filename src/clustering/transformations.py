"""
Feature Transformations for SHI Clustering.

Implements robust transformations per data science best practices:
- log1p for positive heavy-tailed counts
- asinh for signed deltas
- RobustScaler for outlier-resistant standardization
- Explicit missingness handling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray
import structlog

logger = structlog.get_logger()


class TransformationType(Enum):
    """Available transformation types."""

    NONE = "none"
    LOG1P = "log1p"  # For positive heavy-tailed counts
    ASINH = "asinh"  # For signed deltas (like log but handles negatives)
    SQRT = "sqrt"  # For moderate positive skew
    ROBUST_SCALE = "robust_scale"  # IQR-based scaling


@dataclass
class FeatureTransformConfig:
    """Configuration for a feature's transformation."""

    name: str
    transform: TransformationType
    handle_missing: str = "indicator"  # 'indicator', 'median', 'zero', 'drop'
    clip_lower: Optional[float] = None
    clip_upper: Optional[float] = None
    winsorize_pct: Optional[float] = None  # e.g., 0.01 for 1% winsorization


# Define feature groups with appropriate transformations
FEATURE_GROUPS = {
    "distribution": [
        FeatureTransformConfig("balance", TransformationType.LOG1P),
        FeatureTransformConfig("share", TransformationType.NONE, clip_lower=0, clip_upper=1),
        FeatureTransformConfig("rank", TransformationType.NONE),
    ],
    "temporal": [
        FeatureTransformConfig("entry_time_relative", TransformationType.NONE, clip_lower=0, clip_upper=1),
        FeatureTransformConfig("holding_duration", TransformationType.LOG1P),
        FeatureTransformConfig("position_volatility", TransformationType.SQRT),
    ],
    "flow": [
        FeatureTransformConfig("delta_balance_7d", TransformationType.ASINH),
        FeatureTransformConfig("delta_balance_30d", TransformationType.ASINH),
    ],
    "trading": [
        FeatureTransformConfig("trade_count", TransformationType.LOG1P),
        FeatureTransformConfig("burstiness", TransformationType.NONE, clip_lower=-1, clip_upper=1),
        FeatureTransformConfig("swap_frequency", TransformationType.LOG1P),
        FeatureTransformConfig("lp_interaction_ratio", TransformationType.NONE, clip_lower=0, clip_upper=1),
    ],
    "graph": [
        FeatureTransformConfig("in_degree", TransformationType.LOG1P),
        FeatureTransformConfig("out_degree", TransformationType.LOG1P),
        FeatureTransformConfig("eigenvector_centrality", TransformationType.SQRT),
        FeatureTransformConfig("shared_funder_count", TransformationType.LOG1P),
        # New weighted graph features
        FeatureTransformConfig("total_funding_received", TransformationType.LOG1P),
        FeatureTransformConfig("largest_funder_share", TransformationType.NONE, clip_lower=0, clip_upper=1),
        FeatureTransformConfig("funding_hhi", TransformationType.NONE, clip_lower=0, clip_upper=1),
        FeatureTransformConfig("funding_burst_score", TransformationType.ASINH),
        FeatureTransformConfig("weighted_in_degree", TransformationType.LOG1P),
        FeatureTransformConfig("weighted_out_degree", TransformationType.LOG1P),
    ],
    "price_pnl": [
        FeatureTransformConfig("entry_price_usd", TransformationType.LOG1P),
        FeatureTransformConfig("current_price_usd", TransformationType.LOG1P),
        FeatureTransformConfig("unrealized_pnl_ratio", TransformationType.ASINH),
        FeatureTransformConfig("unrealized_pnl_usd", TransformationType.ASINH),
        FeatureTransformConfig("price_change_1h_pct", TransformationType.ASINH),
        FeatureTransformConfig("price_change_24h_pct", TransformationType.ASINH),
        FeatureTransformConfig("price_change_7d_pct", TransformationType.ASINH),
    ],
    "liquidity": [
        FeatureTransformConfig("liquidity_usd_current", TransformationType.LOG1P),
        FeatureTransformConfig("liquidity_usd_1h_avg", TransformationType.LOG1P),
        FeatureTransformConfig("liquidity_usd_24h_avg", TransformationType.LOG1P),
        FeatureTransformConfig("sell_pressure_vs_liquidity", TransformationType.ASINH),
    ],
}


@dataclass
class MissingnessIndicators:
    """Tracks which features have missing data."""

    missing_flags: dict[str, NDArray[np.bool_]] = field(default_factory=dict)

    def add_indicator(self, feature_name: str, is_missing: NDArray[np.bool_]) -> None:
        """Add a missingness indicator for a feature."""
        self.missing_flags[f"{feature_name}_missing"] = is_missing

    def get_indicator_names(self) -> list[str]:
        """Get list of all indicator column names."""
        return list(self.missing_flags.keys())

    def to_array(self) -> NDArray[np.float64]:
        """Convert all indicators to a feature array."""
        if not self.missing_flags:
            return np.array([]).reshape(0, 0)
        return np.column_stack([v.astype(np.float64) for v in self.missing_flags.values()])


class FeatureTransformer:
    """
    Applies transformations to features with explicit missingness handling.

    Key principles:
    1. Preserve NaN during feature engineering
    2. Add missingness indicator columns
    3. Use robust transformations for heavy-tailed distributions
    4. Only impute inside sklearn pipelines (fit/transform paradigm)
    """

    def __init__(
        self,
        feature_groups: Optional[dict[str, list[FeatureTransformConfig]]] = None,
    ):
        """
        Initialize transformer.

        Args:
            feature_groups: Custom feature group configurations (uses defaults if None)
        """
        self.feature_groups = feature_groups or FEATURE_GROUPS
        self._fitted_params: dict[str, dict] = {}
        self._is_fitted = False

    def fit(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
    ) -> "FeatureTransformer":
        """
        Fit transformer parameters (medians, IQRs) from training data.

        Args:
            features: (n_samples, n_features) array
            feature_names: Names corresponding to columns

        Returns:
            self
        """
        for i, name in enumerate(feature_names):
            config = self._get_config(name)
            col = features[:, i]

            # Compute robust statistics (ignoring NaN)
            valid_mask = ~np.isnan(col)
            valid_data = col[valid_mask]

            if len(valid_data) > 0:
                self._fitted_params[name] = {
                    "median": float(np.median(valid_data)),
                    "q1": float(np.percentile(valid_data, 25)),
                    "q3": float(np.percentile(valid_data, 75)),
                    "min": float(np.min(valid_data)),
                    "max": float(np.max(valid_data)),
                }
            else:
                self._fitted_params[name] = {
                    "median": 0.0,
                    "q1": 0.0,
                    "q3": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }

        self._is_fitted = True
        logger.info("transformer_fitted", n_features=len(feature_names))
        return self

    def transform(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
        impute: bool = False,
    ) -> tuple[NDArray[np.float64], MissingnessIndicators]:
        """
        Transform features with optional imputation.

        Args:
            features: (n_samples, n_features) array
            feature_names: Names corresponding to columns
            impute: Whether to impute missing values (only True inside sklearn pipelines)

        Returns:
            (transformed_features, missingness_indicators)
        """
        n_samples, n_features = features.shape
        transformed = np.zeros_like(features)
        missingness = MissingnessIndicators()

        for i, name in enumerate(feature_names):
            col = features[:, i].copy()
            config = self._get_config(name)

            # Track missingness
            is_missing = np.isnan(col)
            if is_missing.any():
                missingness.add_indicator(name, is_missing)

            # Apply clipping if configured
            if config.clip_lower is not None:
                col = np.where(~is_missing, np.maximum(col, config.clip_lower), col)
            if config.clip_upper is not None:
                col = np.where(~is_missing, np.minimum(col, config.clip_upper), col)

            # Apply winsorization if configured
            if config.winsorize_pct is not None and self._is_fitted:
                params = self._fitted_params.get(name, {})
                lower = params.get("q1", 0) - 1.5 * (params.get("q3", 0) - params.get("q1", 0))
                upper = params.get("q3", 0) + 1.5 * (params.get("q3", 0) - params.get("q1", 0))
                col = np.where(~is_missing, np.clip(col, lower, upper), col)

            # Apply transformation
            col = self._apply_transform(col, config.transform, is_missing)

            # Impute if requested (only inside sklearn pipelines)
            if impute and is_missing.any() and self._is_fitted:
                params = self._fitted_params.get(name, {})
                impute_value = params.get("median", 0.0)
                col = np.where(is_missing, impute_value, col)

            transformed[:, i] = col

        return transformed, missingness

    def fit_transform(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
        impute: bool = False,
    ) -> tuple[NDArray[np.float64], MissingnessIndicators]:
        """Fit and transform in one call."""
        self.fit(features, feature_names)
        return self.transform(features, feature_names, impute=impute)

    def _get_config(self, feature_name: str) -> FeatureTransformConfig:
        """Get transformation config for a feature."""
        for group_configs in self.feature_groups.values():
            for config in group_configs:
                if config.name == feature_name:
                    return config
        # Default config if not found
        return FeatureTransformConfig(feature_name, TransformationType.NONE)

    def _apply_transform(
        self,
        col: NDArray[np.float64],
        transform: TransformationType,
        is_missing: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Apply transformation to column, preserving NaN."""
        if transform == TransformationType.NONE:
            return col

        # Create output array preserving NaN
        result = col.copy()
        valid_mask = ~is_missing

        if transform == TransformationType.LOG1P:
            # log1p(x) = log(1 + x), only for non-negative values
            result[valid_mask] = np.log1p(np.maximum(col[valid_mask], 0))

        elif transform == TransformationType.ASINH:
            # asinh handles negative values: asinh(x) ≈ sign(x) * log(|x| + sqrt(x² + 1))
            result[valid_mask] = np.arcsinh(col[valid_mask])

        elif transform == TransformationType.SQRT:
            # sqrt only for non-negative values
            result[valid_mask] = np.sqrt(np.maximum(col[valid_mask], 0))

        elif transform == TransformationType.ROBUST_SCALE:
            # IQR-based scaling (requires fitted params)
            if self._is_fitted:
                # Would need feature name context here
                pass

        return result

    def get_feature_group(self, group_name: str) -> list[str]:
        """Get feature names in a group."""
        configs = self.feature_groups.get(group_name, [])
        return [c.name for c in configs]

    def get_all_feature_names(self) -> list[str]:
        """Get all configured feature names."""
        names = []
        for configs in self.feature_groups.values():
            names.extend(c.name for c in configs)
        return names


class RobustScaler:
    """
    Robust scaler using median and IQR instead of mean/std.

    More resistant to outliers than StandardScaler.
    Formula: (x - median) / IQR
    """

    def __init__(self, with_centering: bool = True, with_scaling: bool = True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.center_: Optional[NDArray[np.float64]] = None
        self.scale_: Optional[NDArray[np.float64]] = None

    def fit(self, X: NDArray[np.float64]) -> "RobustScaler":
        """Fit scaler to data."""
        if self.with_centering:
            self.center_ = np.nanmedian(X, axis=0)

        if self.with_scaling:
            q1 = np.nanpercentile(X, 25, axis=0)
            q3 = np.nanpercentile(X, 75, axis=0)
            iqr = q3 - q1
            # Avoid division by zero
            iqr = np.where(iqr == 0, 1.0, iqr)
            self.scale_ = iqr

        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform data."""
        X_transformed = X.copy()

        if self.with_centering and self.center_ is not None:
            X_transformed = X_transformed - self.center_

        if self.with_scaling and self.scale_ is not None:
            X_transformed = X_transformed / self.scale_

        return X_transformed

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fit and transform in one call."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reverse the transformation."""
        X_inv = X.copy()

        if self.with_scaling and self.scale_ is not None:
            X_inv = X_inv * self.scale_

        if self.with_centering and self.center_ is not None:
            X_inv = X_inv + self.center_

        return X_inv
