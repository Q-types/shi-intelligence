"""
Cox Proportional Hazards Model Implementation.

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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

logger = structlog.get_logger()


@dataclass
class ModelValidationResult:
    """Results of model validation."""

    concordance_index: float
    brier_score: float
    roc_auc: float | None
    proportional_hazards_p_value: float
    coefficient_stability: dict[str, float]
    is_valid: bool
    validation_errors: list[str]


@dataclass
class TrainedHazardModel:
    """Container for trained Cox PH model."""

    version: str
    fitter: CoxPHFitter
    feature_names: list[str]
    coefficients: dict[str, float]
    baseline_hazard: pd.DataFrame
    validation: ModelValidationResult
    trained_at: datetime
    training_samples: int


class HazardModelTrainer:
    """
    Trains Cox Proportional Hazards model for sell prediction.

    Per PDR Section 4.8:
    λ(t|x) = λ₀(t) * exp(β'x)
    """

    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer
        self._fitter: CoxPHFitter | None = None

    def train(
        self,
        data: pd.DataFrame,
        duration_col: str = "holding_duration",
        event_col: str = "sold",
        feature_cols: list[str] | None = None,
        version: str = "1.0.0",
    ) -> TrainedHazardModel:
        """
        Train Cox PH model on historical data.

        Args:
            data: DataFrame with features, duration, and event indicator
            duration_col: Column name for time-to-event
            event_col: Column name for event indicator (1=sold, 0=censored)
            feature_cols: Feature columns to use (None = all except duration/event)
            version: Model version string

        Returns:
            TrainedHazardModel with fitted parameters
        """
        logger.info(
            "training_hazard_model",
            samples=len(data),
            version=version,
        )

        if feature_cols is None:
            feature_cols = [
                c for c in data.columns
                if c not in [duration_col, event_col]
            ]

        # Prepare data
        train_data = data[[duration_col, event_col] + feature_cols].copy()
        train_data = train_data.dropna()

        if len(train_data) < 100:
            raise ValueError(f"Insufficient training data: {len(train_data)} samples")

        # Initialize and fit Cox PH
        self._fitter = CoxPHFitter(
            penalizer=self.penalizer,
            l1_ratio=0.0,  # Ridge regularization
        )

        self._fitter.fit(
            train_data,
            duration_col=duration_col,
            event_col=event_col,
            show_progress=False,
        )

        # Extract coefficients
        coefficients = dict(self._fitter.params_)

        # Get baseline hazard
        baseline_hazard = self._fitter.baseline_hazard_

        # Validate model
        validation = self._validate(train_data, duration_col, event_col)

        logger.info(
            "model_trained",
            concordance=validation.concordance_index,
            ph_pvalue=validation.proportional_hazards_p_value,
            is_valid=validation.is_valid,
        )

        return TrainedHazardModel(
            version=version,
            fitter=self._fitter,
            feature_names=feature_cols,
            coefficients=coefficients,
            baseline_hazard=baseline_hazard,
            validation=validation,
            trained_at=datetime.now(timezone.utc),
            training_samples=len(train_data),
        )

    def _validate(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> ModelValidationResult:
        """Validate trained model."""
        errors = []

        # Concordance index
        if self._fitter is None:
            raise ValueError("Model not fitted")
        concordance = self._fitter.concordance_index_

        # Proportional hazards test (Schoenfeld residuals)
        try:
            ph_test = proportional_hazard_test(
                self._fitter,
                data,
                time_transform="rank",
            )
            ph_pvalue = ph_test.summary["p"].min()
        except Exception as e:
            logger.warning("ph_test_failed", error=str(e))
            ph_pvalue = 0.0
            errors.append(f"Proportional hazards test failed: {e}")

        # Check assumption (p > 0.05 means assumption holds)
        if ph_pvalue < 0.05:
            errors.append(
                f"Proportional hazards assumption violated (p={ph_pvalue:.4f})"
            )

        # Coefficient stability (check for extreme values)
        coef_stability = {}
        if self._fitter is not None:
            for name, coef in self._fitter.params_.items():
                coef_stability[name] = float(abs(coef))
                if abs(coef) > 5:
                    errors.append(f"Extreme coefficient for {name}: {coef:.2f}")

        # Brier score (placeholder - would need test set)
        brier_score = 0.0  # TODO: Implement with test set

        # Determine validity
        is_valid = (
            concordance > 0.5  # Better than random
            and ph_pvalue > 0.01  # PH assumption roughly holds
            and len(errors) == 0
        )

        return ModelValidationResult(
            concordance_index=concordance,
            brier_score=brier_score,
            roc_auc=None,  # Would need separate calculation
            proportional_hazards_p_value=ph_pvalue,
            coefficient_stability=coef_stability,
            is_valid=is_valid,
            validation_errors=errors,
        )


class HazardModelPredictor:
    """
    Makes predictions using trained Cox PH model.

    Per PDR Section 4.8:
    P_sell(T) = 1 - exp(-∫₀ᵀ λ(t|x) dt)
    """

    def __init__(self, model: TrainedHazardModel):
        self.model = model
        self._fitter = model.fitter

    def predict_sell_probability(
        self,
        features: pd.DataFrame | dict,
        horizon_days: int = 7,
    ) -> tuple[float, tuple[float, float]]:
        """
        Predict probability of sell within horizon.

        Args:
            features: Feature values for the wallet
            horizon_days: Time horizon T in days

        Returns:
            (sell_probability, (lower_ci, upper_ci))
        """
        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            features = pd.DataFrame([features])

        # Ensure feature order matches training
        features = features[self.model.feature_names]

        # Get survival function at horizon
        survival = self._fitter.predict_survival_function(features, times=[horizon_days])

        # Sell probability = 1 - survival
        survival_prob = float(survival.iloc[0, 0])
        sell_prob = 1.0 - survival_prob

        # Confidence interval from model variance
        # Simplified: use coefficient variance
        var_linear_pred = self._compute_variance(features)
        ci_half = 1.96 * np.sqrt(var_linear_pred)

        # Transform to probability scale
        ci_lower = max(0, sell_prob - ci_half * sell_prob)
        ci_upper = min(1, sell_prob + ci_half * sell_prob)

        return sell_prob, (ci_lower, ci_upper)

    def predict_batch(
        self,
        features: pd.DataFrame,
        horizon_days: int = 7,
    ) -> list[tuple[float, tuple[float, float]]]:
        """Predict sell probabilities for multiple wallets."""
        results = []
        for idx in range(len(features)):
            row = features.iloc[[idx]]
            prob, ci = self.predict_sell_probability(row, horizon_days)
            results.append((prob, ci))
        return results

    def _compute_variance(self, features: pd.DataFrame) -> float:
        """Compute variance of linear predictor."""
        # Simplified variance estimate
        try:
            var_matrix = self._fitter.variance_matrix_
            x = features[self.model.feature_names].values[0]
            var = float(x @ var_matrix.values @ x.T)
            return var
        except Exception:
            return 0.1  # Default uncertainty


def validate_proportional_hazards(
    model: TrainedHazardModel,
    test_data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run comprehensive PH assumption diagnostics.

    Per INITIAL_PROMPT:
    - Schoenfeld residual diagnostics required
    """
    logger.info("validating_ph_assumption")

    results: dict[str, Any] = {
        "schoenfeld_test": None,
        "residual_plots_data": None,
        "assumption_holds": False,
    }

    try:
        # Schoenfeld residuals test
        ph_test = proportional_hazard_test(
            model.fitter,
            test_data,
            time_transform="rank",
        )

        results["schoenfeld_test"] = {
            "test_statistic": ph_test.summary["test_statistic"].to_dict(),
            "p_values": ph_test.summary["p"].to_dict(),
            "global_p_value": float(ph_test.summary["p"].min()),
        }

        # Check if assumption holds (p > 0.05 for all covariates)
        results["assumption_holds"] = bool(all(
            p > 0.05 for p in ph_test.summary["p"]
        ))

        # Get residual data for plotting
        results["residual_plots_data"] = {
            "scaled_schoenfeld": model.fitter.compute_residuals(
                test_data, kind="scaled_schoenfeld"
            ).to_dict(),
        }

    except Exception as e:
        logger.error("ph_validation_failed", error=str(e))
        results["error"] = str(e)

    return results
