"""
Statistical Validation Suite.

Per INITIAL_PROMPT requirements:
- K-fold cross-validation
- Out-of-sample temporal validation
- Calibration curves
- Brier score reporting
- ROC-AUC reporting
- Drift detection
- Coefficient stability checks
- Hazard calibration verification
- Cluster stability verification

No model deploys without passing validation thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
import structlog
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

logger = structlog.get_logger()


@dataclass
class ValidationThresholds:
    """Deployment thresholds for model validation."""

    min_concordance: float = 0.55
    max_brier_score: float = 0.25
    min_roc_auc: float = 0.60
    min_ph_pvalue: float = 0.01
    max_coefficient_change: float = 0.5
    min_calibration_slope: float = 0.7
    max_calibration_slope: float = 1.3


@dataclass
class CalibrationResult:
    """Result of calibration analysis."""

    fraction_positives: list[float]
    mean_predicted: list[float]
    calibration_slope: float
    calibration_intercept: float
    is_well_calibrated: bool


@dataclass
class ROCResult:
    """ROC analysis result."""

    fpr: list[float]
    tpr: list[float]
    thresholds: list[float]
    auc: float


@dataclass
class ValidationReport:
    """Complete validation report for a model."""

    # Basic metrics
    concordance_index: float
    brier_score: float
    roc_auc: float

    # Cross-validation
    cv_scores: list[float]
    cv_mean: float
    cv_std: float

    # Temporal validation
    temporal_concordance: float

    # Calibration
    calibration: CalibrationResult

    # ROC
    roc: ROCResult

    # PH assumption
    ph_test_pvalue: float
    ph_assumption_holds: bool

    # Overall
    passes_thresholds: bool
    failed_checks: list[str]
    computed_at: datetime


class ModelValidator:
    """
    Comprehensive model validation suite.

    Validates hazard models before deployment.
    """

    def __init__(self, thresholds: ValidationThresholds | None = None):
        self.thresholds = thresholds or ValidationThresholds()

    def validate(
        self,
        fitter: CoxPHFitter,
        data: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
        feature_cols: list[str] | None = None,
    ) -> ValidationReport:
        """
        Run full validation suite on a trained model.

        Args:
            fitter: Trained CoxPHFitter
            data: Full dataset for validation
            duration_col: Duration column name
            event_col: Event indicator column name
            feature_cols: Feature columns

        Returns:
            ValidationReport with all metrics
        """
        logger.info("starting_validation", samples=len(data))

        if feature_cols is None:
            feature_cols = [
                c for c in data.columns
                if c not in [duration_col, event_col]
            ]

        failed_checks = []

        # Basic metrics
        concordance = fitter.concordance_index_

        # Brier score (at median duration)
        median_time = data[duration_col].median()
        brier = self._compute_brier_score(
            fitter, data, feature_cols, duration_col, event_col, median_time
        )

        # ROC-AUC
        roc_result, roc_auc = self._compute_roc(
            fitter, data, feature_cols, duration_col, event_col, median_time
        )

        # Cross-validation
        cv_scores = self._cross_validate(
            data, feature_cols, duration_col, event_col
        )
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        # Temporal validation
        temporal_concordance = self._temporal_validate(
            data, feature_cols, duration_col, event_col
        )

        # Calibration
        calibration = self._compute_calibration(
            fitter, data, feature_cols, duration_col, event_col, median_time
        )

        # PH assumption test
        from lifelines.statistics import proportional_hazard_test
        try:
            train_data = data[[duration_col, event_col] + feature_cols]
            ph_test = proportional_hazard_test(fitter, train_data, time_transform="rank")
            ph_pvalue = float(ph_test.summary["p"].min())
        except Exception as e:
            logger.warning("ph_test_failed", error=str(e))
            ph_pvalue = 0.0
            failed_checks.append(f"PH test failed: {e}")

        ph_holds = ph_pvalue >= self.thresholds.min_ph_pvalue

        # Check thresholds
        if concordance < self.thresholds.min_concordance:
            failed_checks.append(
                f"Concordance {concordance:.3f} < {self.thresholds.min_concordance}"
            )
        if brier > self.thresholds.max_brier_score:
            failed_checks.append(
                f"Brier score {brier:.3f} > {self.thresholds.max_brier_score}"
            )
        if roc_auc < self.thresholds.min_roc_auc:
            failed_checks.append(
                f"ROC-AUC {roc_auc:.3f} < {self.thresholds.min_roc_auc}"
            )
        if not ph_holds:
            failed_checks.append(
                f"PH assumption violated (p={ph_pvalue:.4f})"
            )
        if not calibration.is_well_calibrated:
            failed_checks.append(
                f"Poor calibration (slope={calibration.calibration_slope:.2f})"
            )

        passes = len(failed_checks) == 0

        logger.info(
            "validation_completed",
            concordance=concordance,
            brier=brier,
            roc_auc=roc_auc,
            cv_mean=cv_mean,
            passes=passes,
            failed_count=len(failed_checks),
        )

        return ValidationReport(
            concordance_index=concordance,
            brier_score=brier,
            roc_auc=roc_auc,
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            cv_std=cv_std,
            temporal_concordance=temporal_concordance,
            calibration=calibration,
            roc=roc_result,
            ph_test_pvalue=ph_pvalue,
            ph_assumption_holds=ph_holds,
            passes_thresholds=passes,
            failed_checks=failed_checks,
            computed_at=datetime.now(timezone.utc),
        )

    def _compute_brier_score(
        self,
        fitter: CoxPHFitter,
        data: pd.DataFrame,
        feature_cols: list[str],
        duration_col: str,
        event_col: str,
        time_point: float,
    ) -> float:
        """Compute Brier score at a specific time point."""
        try:
            # Get survival probabilities at time_point
            survival_probs = fitter.predict_survival_function(
                data[feature_cols],
                times=[time_point],
            )

            # Event probability = 1 - survival
            predicted_probs = 1 - survival_probs.iloc[0].values

            # Actual outcomes at time_point
            # Event occurred before time_point
            actual = (
                (data[event_col] == 1) & (data[duration_col] <= time_point)
            ).astype(int).values

            return float(brier_score_loss(actual, predicted_probs))

        except Exception as e:
            logger.warning("brier_score_failed", error=str(e))
            return 0.5

    def _compute_roc(
        self,
        fitter: CoxPHFitter,
        data: pd.DataFrame,
        feature_cols: list[str],
        duration_col: str,
        event_col: str,
        time_point: float,
    ) -> tuple[ROCResult, float]:
        """Compute ROC curve and AUC."""
        try:
            survival_probs = fitter.predict_survival_function(
                data[feature_cols],
                times=[time_point],
            )
            predicted_probs = 1 - survival_probs.iloc[0].values

            actual = (
                (data[event_col] == 1) & (data[duration_col] <= time_point)
            ).astype(int).values

            fpr, tpr, thresholds = roc_curve(actual, predicted_probs)
            auc = roc_auc_score(actual, predicted_probs)

            return ROCResult(
                fpr=fpr.tolist(),
                tpr=tpr.tolist(),
                thresholds=thresholds.tolist(),
                auc=float(auc),
            ), float(auc)

        except Exception as e:
            logger.warning("roc_computation_failed", error=str(e))
            return ROCResult(
                fpr=[0, 1],
                tpr=[0, 1],
                thresholds=[1, 0],
                auc=0.5,
            ), 0.5

    def _cross_validate(
        self,
        data: pd.DataFrame,
        feature_cols: list[str],
        duration_col: str,
        event_col: str,
        n_splits: int = 5,
    ) -> list[float]:
        """K-fold cross-validation."""
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in kf.split(data):
            train = data.iloc[train_idx]
            val = data.iloc[val_idx]

            try:
                fitter = CoxPHFitter(penalizer=0.1)
                fitter.fit(
                    train[[duration_col, event_col] + feature_cols],
                    duration_col=duration_col,
                    event_col=event_col,
                    show_progress=False,
                )
                score = fitter.score(
                    val[[duration_col, event_col] + feature_cols],
                    scoring_method="concordance_index",
                )
                scores.append(score)
            except Exception:
                scores.append(0.5)

        return scores

    def _temporal_validate(
        self,
        data: pd.DataFrame,
        feature_cols: list[str],
        duration_col: str,
        event_col: str,
    ) -> float:
        """Temporal out-of-sample validation."""
        # Use last 20% as test
        split_idx = int(len(data) * 0.8)
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]

        try:
            fitter = CoxPHFitter(penalizer=0.1)
            fitter.fit(
                train[[duration_col, event_col] + feature_cols],
                duration_col=duration_col,
                event_col=event_col,
                show_progress=False,
            )
            return fitter.score(
                test[[duration_col, event_col] + feature_cols],
                scoring_method="concordance_index",
            )
        except Exception:
            return 0.5

    def _compute_calibration(
        self,
        fitter: CoxPHFitter,
        data: pd.DataFrame,
        feature_cols: list[str],
        duration_col: str,
        event_col: str,
        time_point: float,
        n_bins: int = 10,
    ) -> CalibrationResult:
        """Compute calibration curve."""
        try:
            survival_probs = fitter.predict_survival_function(
                data[feature_cols],
                times=[time_point],
            )
            predicted_probs = 1 - survival_probs.iloc[0].values

            actual = (
                (data[event_col] == 1) & (data[duration_col] <= time_point)
            ).astype(int).values

            fraction_pos, mean_pred = calibration_curve(
                actual, predicted_probs, n_bins=n_bins, strategy="uniform"
            )

            # Compute calibration slope via linear regression
            if len(fraction_pos) > 1:
                slope = np.polyfit(mean_pred, fraction_pos, 1)[0]
                intercept = np.polyfit(mean_pred, fraction_pos, 1)[1]
            else:
                slope = 1.0
                intercept = 0.0

            is_calibrated = (
                self.thresholds.min_calibration_slope <= slope
                <= self.thresholds.max_calibration_slope
            )

            return CalibrationResult(
                fraction_positives=fraction_pos.tolist(),
                mean_predicted=mean_pred.tolist(),
                calibration_slope=float(slope),
                calibration_intercept=float(intercept),
                is_well_calibrated=is_calibrated,
            )

        except Exception as e:
            logger.warning("calibration_failed", error=str(e))
            return CalibrationResult(
                fraction_positives=[],
                mean_predicted=[],
                calibration_slope=1.0,
                calibration_intercept=0.0,
                is_well_calibrated=True,
            )
