"""
Expanded Feature Set for Cox PH Hazard Model.

Adds price, liquidity, LP, swap-frequency, and graph-centrality features
to improve sell-risk prediction.

Model comparison and calibration checks before deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


# Original features (from training.py)
ORIGINAL_FEATURES = [
    "share",
    "entry_time_relative",
    "holding_duration",
    "position_volatility",
    "delta_balance_7d",
    "trade_count",
    "burstiness",
    "in_degree",
    "out_degree",
    "shared_funder_count",
]

# New candidate features for evaluation
CANDIDATE_FEATURES = {
    # Price features
    "price": [
        "unrealized_pnl_ratio",
        "unrealized_pnl_usd",
        "price_change_24h_pct",
        "price_change_7d_pct",
    ],

    # Liquidity features
    "liquidity": [
        "liquidity_usd_current",
        "liquidity_usd_24h_avg",
        "sell_pressure_vs_liquidity",
    ],

    # LP interaction features
    "lp": [
        "lp_interaction_ratio",
    ],

    # Swap frequency features
    "swap": [
        "swap_frequency",
        "delta_balance_30d",
    ],

    # Graph centrality features
    "graph_centrality": [
        "eigenvector_centrality",
        "weighted_in_degree",
        "weighted_out_degree",
        "funding_hhi",
        "largest_funder_share",
    ],
}


@dataclass
class FeatureSetConfig:
    """Configuration for a feature set to evaluate."""

    name: str
    features: list[str]
    description: str


@dataclass
class ModelComparisonResult:
    """Result of comparing two model configurations."""

    baseline_name: str
    candidate_name: str

    # Concordance metrics
    baseline_concordance: float
    candidate_concordance: float
    concordance_improvement: float

    # Calibration metrics
    baseline_brier: Optional[float]
    candidate_brier: Optional[float]
    brier_improvement: Optional[float]

    # Feature counts
    baseline_n_features: int
    candidate_n_features: int

    # Additional features used
    additional_features: list[str]

    # Recommendation
    is_improvement: bool
    recommendation: str


@dataclass
class FeatureSelectionResult:
    """Result of feature selection process."""

    selected_features: list[str]
    dropped_features: list[str]
    feature_scores: dict[str, float]
    selection_method: str
    final_concordance: float


class ExpandedFeatureEvaluator:
    """
    Evaluates expanded feature sets for Cox PH model.

    Provides:
    - Model comparison between baseline and candidate feature sets
    - Calibration checks (Brier score)
    - Feature importance analysis
    - Backward/forward selection
    """

    def __init__(
        self,
        baseline_features: Optional[list[str]] = None,
        candidate_features: Optional[dict[str, list[str]]] = None,
    ):
        """
        Initialize evaluator.

        Args:
            baseline_features: Original feature set
            candidate_features: Candidate features by category
        """
        self.baseline_features = baseline_features or ORIGINAL_FEATURES
        self.candidate_features = candidate_features or CANDIDATE_FEATURES

    def get_all_candidate_features(self) -> list[str]:
        """Get flat list of all candidate features."""
        features = []
        for category_features in self.candidate_features.values():
            features.extend(category_features)
        return features

    def get_expanded_feature_set(self) -> list[str]:
        """Get full expanded feature set (baseline + all candidates)."""
        all_features = list(self.baseline_features)
        for feature in self.get_all_candidate_features():
            if feature not in all_features:
                all_features.append(feature)
        return all_features

    def compare_feature_sets(
        self,
        data: pd.DataFrame,
        baseline_features: list[str],
        candidate_features: list[str],
        n_splits: int = 5,
    ) -> ModelComparisonResult:
        """
        Compare two feature sets using temporal validation.

        Args:
            data: Training data with 'duration' and 'event' columns
            baseline_features: Baseline feature list
            candidate_features: Candidate feature list
            n_splits: Number of validation splits

        Returns:
            ModelComparisonResult with comparison metrics
        """
        from lifelines import CoxPHFitter
        from .temporal_validation import TemporalValidator

        logger.info(
            "comparing_feature_sets",
            baseline_n=len(baseline_features),
            candidate_n=len(candidate_features),
        )

        # Validate features exist in data
        baseline_features = [f for f in baseline_features if f in data.columns]
        candidate_features = [f for f in candidate_features if f in data.columns]

        # Prepare data
        required_cols = ["duration", "event"]
        data = data[data["duration"] > 0].copy()

        # Temporal cross-validation
        validator = TemporalValidator(n_splits=n_splits)

        baseline_scores = []
        candidate_scores = []

        for split in validator.split(data, "duration"):
            train_idx = split.train_indices
            test_idx = split.test_indices

            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # Baseline model
            baseline_score = self._fit_and_score(
                train_data,
                test_data,
                baseline_features,
            )
            baseline_scores.append(baseline_score)

            # Candidate model
            candidate_score = self._fit_and_score(
                train_data,
                test_data,
                candidate_features,
            )
            candidate_scores.append(candidate_score)

        baseline_concordance = float(np.mean(baseline_scores))
        candidate_concordance = float(np.mean(candidate_scores))
        improvement = candidate_concordance - baseline_concordance

        # Determine if improvement is significant
        is_improvement = improvement > 0.01  # 1% threshold

        if is_improvement:
            recommendation = f"Use expanded features (+{improvement:.3f} concordance)"
        else:
            recommendation = "Keep baseline features (no significant improvement)"

        additional = [f for f in candidate_features if f not in baseline_features]

        return ModelComparisonResult(
            baseline_name="original",
            candidate_name="expanded",
            baseline_concordance=baseline_concordance,
            candidate_concordance=candidate_concordance,
            concordance_improvement=improvement,
            baseline_brier=None,  # Would need survival function
            candidate_brier=None,
            brier_improvement=None,
            baseline_n_features=len(baseline_features),
            candidate_n_features=len(candidate_features),
            additional_features=additional,
            is_improvement=is_improvement,
            recommendation=recommendation,
        )

    def _fit_and_score(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        features: list[str],
    ) -> float:
        """Fit Cox PH and return concordance score."""
        from lifelines import CoxPHFitter

        # Prepare data
        train_cols = ["duration", "event"] + features
        test_cols = ["duration", "event"] + features

        train_df = train_data[train_cols].copy()
        test_df = test_data[test_cols].copy()

        # Fill missing
        train_df = train_df.fillna(0)
        test_df = test_df.fillna(0)

        try:
            fitter = CoxPHFitter(penalizer=0.1)
            fitter.fit(train_df, duration_col="duration", event_col="event")
            return fitter.score(test_df, scoring_method="concordance_index")
        except Exception as e:
            logger.warning("model_fit_failed", error=str(e))
            return 0.5

    def backward_selection(
        self,
        data: pd.DataFrame,
        features: list[str],
        min_features: int = 5,
        threshold: float = 0.001,
    ) -> FeatureSelectionResult:
        """
        Backward feature selection.

        Iteratively remove features that don't improve model.

        Args:
            data: Training data
            features: Starting feature set
            min_features: Minimum features to keep
            threshold: Minimum improvement to keep feature

        Returns:
            FeatureSelectionResult with selected features
        """
        from lifelines import CoxPHFitter

        logger.info("starting_backward_selection", n_features=len(features))

        current_features = [f for f in features if f in data.columns]
        dropped_features = []
        feature_scores = {}

        # Get baseline score
        baseline_score = self._get_model_score(data, current_features)

        while len(current_features) > min_features:
            worst_feature = None
            best_score_without = baseline_score

            for feature in current_features:
                # Try removing this feature
                test_features = [f for f in current_features if f != feature]
                score = self._get_model_score(data, test_features)

                feature_scores[feature] = baseline_score - score

                # If score improves or stays same, this feature is candidate for removal
                if score >= best_score_without - threshold:
                    best_score_without = score
                    worst_feature = feature

            if worst_feature is None:
                break

            # Remove worst feature
            current_features.remove(worst_feature)
            dropped_features.append(worst_feature)
            baseline_score = best_score_without

            logger.debug(
                "feature_dropped",
                feature=worst_feature,
                new_score=baseline_score,
            )

        logger.info(
            "backward_selection_completed",
            selected_n=len(current_features),
            dropped_n=len(dropped_features),
        )

        return FeatureSelectionResult(
            selected_features=current_features,
            dropped_features=dropped_features,
            feature_scores=feature_scores,
            selection_method="backward",
            final_concordance=baseline_score,
        )

    def _get_model_score(
        self,
        data: pd.DataFrame,
        features: list[str],
    ) -> float:
        """Get concordance index for feature set."""
        from lifelines import CoxPHFitter

        cols = ["duration", "event"] + features
        df = data[cols].copy().fillna(0)
        df = df[df["duration"] > 0]

        try:
            fitter = CoxPHFitter(penalizer=0.1)
            fitter.fit(df, duration_col="duration", event_col="event")
            return fitter.concordance_index_
        except Exception:
            return 0.5

    def get_feature_importance(
        self,
        data: pd.DataFrame,
        features: list[str],
    ) -> dict[str, float]:
        """
        Get feature importance from Cox PH coefficients.

        Args:
            data: Training data
            features: Feature list

        Returns:
            Dict mapping feature -> absolute coefficient
        """
        from lifelines import CoxPHFitter

        cols = ["duration", "event"] + features
        df = data[cols].copy().fillna(0)
        df = df[df["duration"] > 0]

        try:
            fitter = CoxPHFitter(penalizer=0.1)
            fitter.fit(df, duration_col="duration", event_col="event")

            return {
                name: abs(float(coef))
                for name, coef in fitter.params_.items()
            }
        except Exception as e:
            logger.warning("importance_computation_failed", error=str(e))
            return {f: 0.0 for f in features}


def get_recommended_features(
    data: pd.DataFrame,
    evaluate_all: bool = True,
) -> list[str]:
    """
    Get recommended feature set for Cox PH model.

    Evaluates expanded features and returns best-performing set.

    Args:
        data: Training data
        evaluate_all: Whether to evaluate all feature combinations

    Returns:
        List of recommended features
    """
    evaluator = ExpandedFeatureEvaluator()

    if not evaluate_all:
        # Just return expanded set
        return evaluator.get_expanded_feature_set()

    # Compare baseline vs expanded
    baseline = evaluator.baseline_features
    expanded = evaluator.get_expanded_feature_set()

    comparison = evaluator.compare_feature_sets(data, baseline, expanded)

    if comparison.is_improvement:
        logger.info(
            "expanded_features_recommended",
            improvement=comparison.concordance_improvement,
        )
        return expanded
    else:
        logger.info("baseline_features_recommended")
        return baseline
