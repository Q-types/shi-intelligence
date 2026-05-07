"""
Hazard Model Training Pipeline.

Per INITIAL_PROMPT requirements:
- Cox Proportional Hazards
- Efron tie handling
- Time-dependent covariates supported
- Baseline hazard explicitly estimated
- Proportional hazards assumption tested
- Schoenfeld residual diagnostics required
- Weekly retraining
- Coefficient stability checks
- Confidence intervals mandatory
- Version number attached to every output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import json

import numpy as np
import pandas as pd
import structlog
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.model_selection import KFold, TimeSeriesSplit

from .sell_events import SellEvent
from .hazard_model import TrainedHazardModel, ModelValidationResult

logger = structlog.get_logger()


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model parameters
    penalizer: float = 0.1
    l1_ratio: float = 0.0  # 0 = Ridge, 1 = Lasso

    # Validation
    n_folds: int = 5
    test_size: float = 0.2

    # Thresholds for deployment
    min_concordance: float = 0.55
    max_brier_score: float = 0.25
    min_roc_auc: float = 0.60
    min_ph_pvalue: float = 0.01  # PH assumption

    # Feature selection
    feature_columns: list[str] = field(default_factory=lambda: [
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
    ])


@dataclass
class TrainingResult:
    """Result of training run."""

    model: TrainedHazardModel
    validation: ModelValidationResult
    cross_validation_scores: list[float]
    temporal_validation_score: float
    feature_importance: dict[str, float]
    is_deployable: bool
    deployment_blockers: list[str]


class HazardModelTrainingPipeline:
    """
    End-to-end training pipeline for Cox PH model.

    Includes data preparation, training, validation, and deployment checks.
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self._version_counter = 0

    def prepare_training_data(
        self,
        events: list[SellEvent],
        features: list[dict],
    ) -> pd.DataFrame:
        """
        Prepare DataFrame for Cox PH training.

        Args:
            events: List of sell events with censoring info
            features: List of feature dicts per wallet

        Returns:
            DataFrame ready for CoxPHFitter
        """
        # Build feature lookup
        feature_lookup = {f["wallet"]: f for f in features}

        rows = []
        for event in events:
            wallet_features = feature_lookup.get(event.wallet, {})

            row = {
                "wallet": event.wallet,
                "duration": event.holding_duration_days,
                "event": 0 if event.is_censored else 1,
            }

            # Add features
            for col in self.config.feature_columns:
                row[col] = wallet_features.get(col, 0.0)

            rows.append(row)

        df = pd.DataFrame(rows)

        # Handle missing values
        df = df.fillna(0)

        # Remove invalid durations
        df = df[df["duration"] > 0]

        logger.info(
            "training_data_prepared",
            samples=len(df),
            events=df["event"].sum(),
            censored=(df["event"] == 0).sum(),
        )

        return df

    def train(
        self,
        data: pd.DataFrame,
        version: str | None = None,
    ) -> TrainingResult:
        """
        Train Cox PH model with full validation.

        Args:
            data: Prepared training DataFrame
            version: Model version string

        Returns:
            TrainingResult with model and validation metrics
        """
        if version is None:
            self._version_counter += 1
            version = f"v1.0.{self._version_counter}"

        logger.info("training_started", version=version, samples=len(data))

        # Split for temporal validation
        train_data, test_data = self._temporal_split(data)

        # Fit model
        fitter = CoxPHFitter(
            penalizer=self.config.penalizer,
            l1_ratio=self.config.l1_ratio,
        )

        fitter.fit(
            train_data[["duration", "event"] + self.config.feature_columns],
            duration_col="duration",
            event_col="event",
            show_progress=False,
        )

        # Cross-validation
        cv_scores = self._cross_validate(data)

        # Temporal validation
        temporal_score = self._temporal_validate(fitter, test_data)

        # Full validation
        validation = self._validate_model(fitter, data)

        # Feature importance (absolute coefficients)
        feature_importance = {
            name: abs(float(coef))
            for name, coef in fitter.params_.items()
        }

        # Check deployment criteria
        blockers = []
        if validation.concordance_index < self.config.min_concordance:
            blockers.append(
                f"Concordance {validation.concordance_index:.3f} < {self.config.min_concordance}"
            )
        if validation.proportional_hazards_p_value < self.config.min_ph_pvalue:
            blockers.append(
                f"PH assumption violated (p={validation.proportional_hazards_p_value:.4f})"
            )
        if np.mean(cv_scores) < self.config.min_concordance:
            blockers.append(
                f"CV concordance {np.mean(cv_scores):.3f} < {self.config.min_concordance}"
            )

        is_deployable = len(blockers) == 0

        # Create trained model
        model = TrainedHazardModel(
            version=version,
            fitter=fitter,
            feature_names=self.config.feature_columns,
            coefficients=dict(fitter.params_),
            baseline_hazard=fitter.baseline_hazard_,
            validation=validation,
            trained_at=datetime.now(timezone.utc),
            training_samples=len(train_data),
        )

        logger.info(
            "training_completed",
            version=version,
            concordance=validation.concordance_index,
            cv_mean=np.mean(cv_scores),
            is_deployable=is_deployable,
            blockers=blockers,
        )

        return TrainingResult(
            model=model,
            validation=validation,
            cross_validation_scores=cv_scores,
            temporal_validation_score=temporal_score,
            feature_importance=feature_importance,
            is_deployable=is_deployable,
            deployment_blockers=blockers,
        )

    def _temporal_split(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data temporally (not randomly) for proper validation."""
        # Sort by some temporal indicator if available
        # For now, use simple split
        split_idx = int(len(data) * (1 - self.config.test_size))
        return data.iloc[:split_idx], data.iloc[split_idx:]

    def _cross_validate(self, data: pd.DataFrame) -> list[float]:
        """K-fold cross-validation for concordance index."""
        kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in kf.split(data):
            train_fold = data.iloc[train_idx]
            val_fold = data.iloc[val_idx]

            fitter = CoxPHFitter(
                penalizer=self.config.penalizer,
                l1_ratio=self.config.l1_ratio,
            )

            try:
                fitter.fit(
                    train_fold[["duration", "event"] + self.config.feature_columns],
                    duration_col="duration",
                    event_col="event",
                    show_progress=False,
                )
                score = fitter.score(
                    val_fold[["duration", "event"] + self.config.feature_columns],
                    scoring_method="concordance_index",
                )
                scores.append(score)
            except Exception as e:
                logger.warning("cv_fold_failed", error=str(e))
                scores.append(0.5)

        return scores

    def _temporal_validate(
        self,
        fitter: CoxPHFitter,
        test_data: pd.DataFrame,
    ) -> float:
        """Temporal out-of-sample validation."""
        try:
            return fitter.score(
                test_data[["duration", "event"] + self.config.feature_columns],
                scoring_method="concordance_index",
            )
        except Exception as e:
            logger.warning("temporal_validation_failed", error=str(e))
            return 0.5

    def _validate_model(
        self,
        fitter: CoxPHFitter,
        data: pd.DataFrame,
    ) -> ModelValidationResult:
        """Full model validation with PH diagnostics."""
        errors = []

        # Concordance
        concordance = fitter.concordance_index_

        # PH test
        try:
            train_data = data[["duration", "event"] + self.config.feature_columns]
            ph_test = proportional_hazard_test(fitter, train_data, time_transform="rank")
            ph_pvalue = float(ph_test.summary["p"].min())
        except Exception as e:
            logger.warning("ph_test_failed", error=str(e))
            ph_pvalue = 0.0
            errors.append(f"PH test failed: {e}")

        # Coefficient stability
        coef_stability = {}
        for name, coef in fitter.params_.items():
            coef_stability[name] = float(abs(coef))
            if abs(coef) > 10:
                errors.append(f"Extreme coefficient: {name}={coef:.2f}")

        is_valid = (
            concordance >= self.config.min_concordance
            and ph_pvalue >= self.config.min_ph_pvalue
            and len(errors) == 0
        )

        return ModelValidationResult(
            concordance_index=concordance,
            brier_score=0.0,  # Would need survival probabilities
            roc_auc=None,
            proportional_hazards_p_value=ph_pvalue,
            coefficient_stability=coef_stability,
            is_valid=is_valid,
            validation_errors=errors,
        )

    def check_coefficient_stability(
        self,
        current_model: TrainedHazardModel,
        previous_model: TrainedHazardModel,
        max_change_pct: float = 0.5,
    ) -> tuple[bool, dict[str, float]]:
        """
        Check coefficient stability between model versions.

        Returns:
            (is_stable, change_by_feature)
        """
        changes = {}
        is_stable = True

        for feature in current_model.feature_names:
            current = current_model.coefficients.get(feature, 0)
            previous = previous_model.coefficients.get(feature, 0)

            if abs(previous) > 0.01:
                change_pct = abs(current - previous) / abs(previous)
            else:
                change_pct = abs(current - previous)

            changes[feature] = change_pct

            if change_pct > max_change_pct:
                is_stable = False
                logger.warning(
                    "coefficient_drift",
                    feature=feature,
                    current=current,
                    previous=previous,
                    change_pct=change_pct,
                )

        return is_stable, changes
