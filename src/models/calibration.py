"""
Probability Calibration for Hazard Model Outputs.

The goal is not to maximize C-index.
The goal is to produce probabilities that are honest, stable, and decision-useful.

Current issues:
- C-index ~0.88 (acceptable discrimination)
- Calibration slope ~2.0 (poor calibration, ideal is 1.0)

Calibration methods implemented:
- Isotonic regression
- Platt scaling (logistic calibration)
- Beta calibration
- Regime-specific calibration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Literal
from enum import Enum

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import structlog

from ..core.config import settings

logger = structlog.get_logger()


class CalibrationMethod(Enum):
    """Available calibration methods."""

    ISOTONIC = "isotonic"
    PLATT = "platt"
    BETA = "beta"
    REGIME_SPECIFIC = "regime_specific"
    NONE = "none"


class TokenRegime(Enum):
    """Token lifecycle regimes for regime-specific calibration."""

    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    COORDINATED_ACCUMULATION = "coordinated_accumulation"
    DECAY = "decay"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class CalibrationMetrics:
    """Comprehensive calibration metrics."""

    # Discrimination (ranking ability)
    concordance_index: float

    # Calibration (probability accuracy)
    brier_score: float
    calibration_slope: float
    calibration_intercept: float
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float  # MCE

    # Summary statistics
    mean_predicted: float
    mean_observed: float
    n_samples: int
    n_events: int

    # Decile calibration curve
    decile_predicted: list[float]
    decile_observed: list[float]
    decile_counts: list[int]

    def to_dict(self) -> dict:
        return {
            "concordance_index": self.concordance_index,
            "brier_score": self.brier_score,
            "calibration_slope": self.calibration_slope,
            "calibration_intercept": self.calibration_intercept,
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "mean_predicted": self.mean_predicted,
            "mean_observed": self.mean_observed,
            "n_samples": self.n_samples,
            "n_events": self.n_events,
            "decile_predicted": self.decile_predicted,
            "decile_observed": self.decile_observed,
            "decile_counts": self.decile_counts,
        }


@dataclass
class ProbabilityBand:
    """Statistics for a probability band."""

    band_name: str
    lower_bound: float
    upper_bound: float
    wallet_count: int
    predicted_mean: float
    observed_rate: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    is_calibrated: bool  # predicted within CI of observed

    def to_dict(self) -> dict:
        return {
            "band_name": self.band_name,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "wallet_count": self.wallet_count,
            "predicted_mean": self.predicted_mean,
            "observed_rate": self.observed_rate,
            "confidence_interval": [self.confidence_interval_lower, self.confidence_interval_upper],
            "is_calibrated": self.is_calibrated,
        }


@dataclass
class RegimeCalibrationResult:
    """Calibration results for a single regime."""

    regime: str
    sample_size: int
    event_rate: float
    calibration_slope: float
    brier_score: float
    ece: float
    adequate_samples: bool

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "sample_size": self.sample_size,
            "event_rate": self.event_rate,
            "calibration_slope": self.calibration_slope,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "adequate_samples": self.adequate_samples,
        }


@dataclass
class CalibrationComparison:
    """Comparison of calibration methods."""

    method: str
    metrics_before: CalibrationMetrics
    metrics_after: CalibrationMetrics

    # Improvements
    brier_improvement: float
    ece_improvement: float
    slope_improvement: float  # Distance to 1.0
    concordance_change: float

    # Recommendation
    is_recommended: bool
    recommendation_reason: str

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "metrics_before": self.metrics_before.to_dict(),
            "metrics_after": self.metrics_after.to_dict(),
            "brier_improvement": self.brier_improvement,
            "ece_improvement": self.ece_improvement,
            "slope_improvement": self.slope_improvement,
            "concordance_change": self.concordance_change,
            "is_recommended": self.is_recommended,
            "recommendation_reason": self.recommendation_reason,
        }


class ProbabilityCalibrator:
    """
    Calibrates hazard model probability outputs.

    Supports multiple calibration methods with walk-forward validation
    to avoid data leakage.
    """

    def __init__(
        self,
        method: CalibrationMethod = CalibrationMethod.ISOTONIC,
        min_samples_per_regime: int = 50,
    ):
        """
        Initialize calibrator.

        Args:
            method: Calibration method to use
            min_samples_per_regime: Minimum samples for regime-specific calibration
        """
        self.method = method
        self.min_samples_per_regime = min_samples_per_regime
        self._calibrator = None
        self._regime_calibrators: dict[str, object] = {}
        self._is_fitted = False

    def fit(
        self,
        y_prob: NDArray[np.float64],
        y_true: NDArray[np.int32],
        regimes: Optional[NDArray] = None,
    ) -> None:
        """
        Fit calibrator on validation data.

        IMPORTANT: Only fit on validation data, not training data,
        to avoid leakage.

        Args:
            y_prob: Predicted probabilities
            y_true: True binary outcomes
            regimes: Optional regime labels for regime-specific calibration
        """
        logger.info(
            "fitting_calibrator",
            method=self.method.value,
            n_samples=len(y_prob),
            n_events=y_true.sum(),
        )

        if self.method == CalibrationMethod.ISOTONIC:
            self._fit_isotonic(y_prob, y_true)
        elif self.method == CalibrationMethod.PLATT:
            self._fit_platt(y_prob, y_true)
        elif self.method == CalibrationMethod.BETA:
            self._fit_beta(y_prob, y_true)
        elif self.method == CalibrationMethod.REGIME_SPECIFIC:
            if regimes is None:
                raise ValueError("Regimes required for regime-specific calibration")
            self._fit_regime_specific(y_prob, y_true, regimes)
        else:
            # No calibration
            pass

        self._is_fitted = True

    def calibrate(
        self,
        y_prob: NDArray[np.float64],
        regimes: Optional[NDArray] = None,
    ) -> NDArray[np.float64]:
        """
        Apply calibration to probabilities.

        Args:
            y_prob: Raw predicted probabilities
            regimes: Optional regime labels for regime-specific calibration

        Returns:
            Calibrated probabilities
        """
        if not self._is_fitted:
            logger.warning("calibrator_not_fitted_returning_raw")
            return y_prob

        if self.method == CalibrationMethod.ISOTONIC:
            return self._calibrate_isotonic(y_prob)
        elif self.method == CalibrationMethod.PLATT:
            return self._calibrate_platt(y_prob)
        elif self.method == CalibrationMethod.BETA:
            return self._calibrate_beta(y_prob)
        elif self.method == CalibrationMethod.REGIME_SPECIFIC:
            if regimes is None:
                logger.warning("regimes_required_for_regime_calibration")
                return y_prob
            return self._calibrate_regime_specific(y_prob, regimes)
        else:
            return y_prob

    def _fit_isotonic(self, y_prob: NDArray, y_true: NDArray) -> None:
        """Fit isotonic regression calibrator."""
        from sklearn.isotonic import IsotonicRegression

        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(y_prob, y_true)

    def _calibrate_isotonic(self, y_prob: NDArray) -> NDArray:
        """Apply isotonic calibration."""
        return self._calibrator.predict(y_prob)

    def _fit_platt(self, y_prob: NDArray, y_true: NDArray) -> None:
        """Fit Platt scaling (logistic) calibrator."""
        from sklearn.linear_model import LogisticRegression

        # Platt scaling: fit logistic regression on log-odds
        self._calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)

        # Reshape for sklearn
        X = y_prob.reshape(-1, 1)
        self._calibrator.fit(X, y_true)

    def _calibrate_platt(self, y_prob: NDArray) -> NDArray:
        """Apply Platt scaling."""
        X = y_prob.reshape(-1, 1)
        return self._calibrator.predict_proba(X)[:, 1]

    def _fit_beta(self, y_prob: NDArray, y_true: NDArray) -> None:
        """
        Fit beta calibration.

        Beta calibration: P_calibrated = 1 / (1 + 1/(e^c * (P/(1-P))^a + e^b))

        Uses scipy optimization to find a, b, c parameters.
        """
        from scipy.optimize import minimize

        def beta_transform(params, p):
            a, b, c = params
            # Avoid numerical issues
            p = np.clip(p, 1e-10, 1 - 1e-10)
            odds = p / (1 - p)
            return 1 / (1 + 1 / (np.exp(c) * np.power(odds, a) + np.exp(b)))

        def neg_log_likelihood(params):
            p_cal = beta_transform(params, y_prob)
            p_cal = np.clip(p_cal, 1e-10, 1 - 1e-10)
            # Binary cross-entropy
            return -np.mean(y_true * np.log(p_cal) + (1 - y_true) * np.log(1 - p_cal))

        # Initial parameters
        result = minimize(neg_log_likelihood, x0=[1.0, 0.0, 0.0], method="Nelder-Mead")
        self._beta_params = result.x

    def _calibrate_beta(self, y_prob: NDArray) -> NDArray:
        """Apply beta calibration."""
        a, b, c = self._beta_params
        p = np.clip(y_prob, 1e-10, 1 - 1e-10)
        odds = p / (1 - p)
        return 1 / (1 + 1 / (np.exp(c) * np.power(odds, a) + np.exp(b)))

    def _fit_regime_specific(
        self,
        y_prob: NDArray,
        y_true: NDArray,
        regimes: NDArray,
    ) -> None:
        """Fit separate calibrators per regime."""
        from sklearn.isotonic import IsotonicRegression

        unique_regimes = np.unique(regimes)

        for regime in unique_regimes:
            mask = regimes == regime
            n_samples = mask.sum()

            if n_samples >= self.min_samples_per_regime:
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(y_prob[mask], y_true[mask])
                self._regime_calibrators[regime] = calibrator
                logger.info(
                    "regime_calibrator_fitted",
                    regime=regime,
                    n_samples=n_samples,
                )
            else:
                logger.warning(
                    "regime_insufficient_samples",
                    regime=regime,
                    n_samples=n_samples,
                    min_required=self.min_samples_per_regime,
                )

        # Fallback calibrator for regimes without enough samples
        self._fallback_calibrator = IsotonicRegression(out_of_bounds="clip")
        self._fallback_calibrator.fit(y_prob, y_true)

    def _calibrate_regime_specific(
        self,
        y_prob: NDArray,
        regimes: NDArray,
    ) -> NDArray:
        """Apply regime-specific calibration."""
        result = np.zeros_like(y_prob)

        for regime in np.unique(regimes):
            mask = regimes == regime

            if regime in self._regime_calibrators:
                result[mask] = self._regime_calibrators[regime].predict(y_prob[mask])
            else:
                result[mask] = self._fallback_calibrator.predict(y_prob[mask])

        return result


def compute_calibration_metrics(
    y_prob: NDArray[np.float64],
    y_true: NDArray[np.int32],
    n_bins: int = 10,
) -> CalibrationMetrics:
    """
    Compute comprehensive calibration metrics.

    Args:
        y_prob: Predicted probabilities
        y_true: True binary outcomes
        n_bins: Number of bins for calibration curve

    Returns:
        CalibrationMetrics with all metrics
    """
    from sklearn.metrics import brier_score_loss
    from scipy.stats import linregress

    n_samples = len(y_prob)
    n_events = int(y_true.sum())

    # Brier score
    brier = brier_score_loss(y_true, y_prob)

    # Concordance index (simplified for binary)
    concordance = _compute_concordance(y_prob, y_true)

    # Calibration slope and intercept via linear regression
    # Regress observed on predicted
    try:
        slope, intercept, _, _, _ = linregress(y_prob, y_true)
    except Exception:
        slope, intercept = 0.0, 0.0

    # ECE and MCE
    ece, mce, decile_pred, decile_obs, decile_counts = _compute_ece_mce(
        y_prob, y_true, n_bins
    )

    return CalibrationMetrics(
        concordance_index=concordance,
        brier_score=brier,
        calibration_slope=slope,
        calibration_intercept=intercept,
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        mean_predicted=float(np.mean(y_prob)),
        mean_observed=float(np.mean(y_true)),
        n_samples=n_samples,
        n_events=n_events,
        decile_predicted=decile_pred,
        decile_observed=decile_obs,
        decile_counts=decile_counts,
    )


def _compute_concordance(y_prob: NDArray, y_true: NDArray) -> float:
    """Compute concordance index for binary outcomes."""
    try:
        from lifelines.utils import concordance_index
        # For binary outcomes, use the probability as the predicted "time"
        # Higher probability should correspond to shorter time (event more likely)
        return concordance_index(1 - y_true, y_prob, y_true)
    except Exception:
        # Simple approximation
        n_concordant = 0
        n_discordant = 0
        n_tied = 0

        events = np.where(y_true == 1)[0]
        non_events = np.where(y_true == 0)[0]

        for e in events:
            for ne in non_events:
                if y_prob[e] > y_prob[ne]:
                    n_concordant += 1
                elif y_prob[e] < y_prob[ne]:
                    n_discordant += 1
                else:
                    n_tied += 1

        total = n_concordant + n_discordant + n_tied
        if total == 0:
            return 0.5

        return (n_concordant + 0.5 * n_tied) / total


def _compute_ece_mce(
    y_prob: NDArray,
    y_true: NDArray,
    n_bins: int = 10,
) -> tuple[float, float, list[float], list[float], list[int]]:
    """
    Compute Expected and Maximum Calibration Error.

    ECE = sum(|bin_accuracy - bin_confidence| * bin_size / total)
    MCE = max(|bin_accuracy - bin_confidence|)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    mce = 0.0
    decile_pred = []
    decile_obs = []
    decile_counts = []

    total_samples = len(y_prob)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Include upper bound for last bin
        if i == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        bin_size = mask.sum()

        if bin_size > 0:
            bin_pred = y_prob[mask].mean()
            bin_obs = y_true[mask].mean()
            bin_error = abs(bin_pred - bin_obs)

            ece += bin_error * bin_size / total_samples
            mce = max(mce, bin_error)

            decile_pred.append(float(bin_pred))
            decile_obs.append(float(bin_obs))
            decile_counts.append(int(bin_size))
        else:
            decile_pred.append(0.0)
            decile_obs.append(0.0)
            decile_counts.append(0)

    return ece, mce, decile_pred, decile_obs, decile_counts


def compute_probability_bands(
    y_prob: NDArray[np.float64],
    y_true: NDArray[np.int32],
) -> list[ProbabilityBand]:
    """
    Compute statistics for probability bands.

    Bands: 0-10%, 10-25%, 25-50%, 50-75%, 75-100%
    """
    from scipy import stats

    bands = [
        ("0-10%", 0.0, 0.10),
        ("10-25%", 0.10, 0.25),
        ("25-50%", 0.25, 0.50),
        ("50-75%", 0.50, 0.75),
        ("75-100%", 0.75, 1.00),
    ]

    results = []

    for name, lower, upper in bands:
        mask = (y_prob >= lower) & (y_prob < upper if upper < 1.0 else y_prob <= upper)
        count = mask.sum()

        if count > 0:
            pred_mean = float(y_prob[mask].mean())
            obs_rate = float(y_true[mask].mean())

            # Wilson confidence interval for proportion
            ci_lower, ci_upper = _wilson_ci(obs_rate, count)

            # Check if predicted is within CI of observed
            is_calibrated = ci_lower <= pred_mean <= ci_upper

        else:
            pred_mean = 0.0
            obs_rate = 0.0
            ci_lower = 0.0
            ci_upper = 0.0
            is_calibrated = True  # Vacuously true

        results.append(ProbabilityBand(
            band_name=name,
            lower_bound=lower,
            upper_bound=upper,
            wallet_count=int(count),
            predicted_mean=pred_mean,
            observed_rate=obs_rate,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            is_calibrated=is_calibrated,
        ))

    return results


def _wilson_ci(p: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute Wilson confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0

    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)

    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    return max(0.0, centre - margin), min(1.0, centre + margin)


def compare_calibration_methods(
    y_prob: NDArray[np.float64],
    y_true: NDArray[np.int32],
    regimes: Optional[NDArray] = None,
) -> list[CalibrationComparison]:
    """
    Compare all calibration methods.

    Uses leave-one-out or walk-forward to avoid overfitting assessment.
    """
    methods = [
        CalibrationMethod.ISOTONIC,
        CalibrationMethod.PLATT,
        CalibrationMethod.BETA,
    ]

    if regimes is not None:
        methods.append(CalibrationMethod.REGIME_SPECIFIC)

    # Baseline metrics (uncalibrated)
    baseline_metrics = compute_calibration_metrics(y_prob, y_true)

    comparisons = []

    for method in methods:
        try:
            # Split data for calibration fitting vs evaluation
            # Use 50/50 split for simplicity (walk-forward is done separately)
            n = len(y_prob)
            split_idx = n // 2

            train_prob = y_prob[:split_idx]
            train_true = y_true[:split_idx]
            test_prob = y_prob[split_idx:]
            test_true = y_true[split_idx:]

            train_regimes = regimes[:split_idx] if regimes is not None else None
            test_regimes = regimes[split_idx:] if regimes is not None else None

            # Fit calibrator on first half
            calibrator = ProbabilityCalibrator(method=method)
            calibrator.fit(train_prob, train_true, train_regimes)

            # Calibrate second half
            calibrated_prob = calibrator.calibrate(test_prob, test_regimes)

            # Compute metrics on second half
            metrics_after = compute_calibration_metrics(calibrated_prob, test_true)
            metrics_before = compute_calibration_metrics(test_prob, test_true)

            # Compute improvements
            brier_imp = metrics_before.brier_score - metrics_after.brier_score
            ece_imp = metrics_before.expected_calibration_error - metrics_after.expected_calibration_error
            slope_imp = abs(metrics_before.calibration_slope - 1.0) - abs(metrics_after.calibration_slope - 1.0)
            c_change = metrics_after.concordance_index - metrics_before.concordance_index

            # Recommendation logic:
            # Deploy if improves Brier, ECE, slope without materially damaging C-index
            is_recommended = (
                brier_imp > 0 and
                ece_imp > 0 and
                slope_imp > 0 and
                c_change > -0.02  # Allow up to 2% C-index drop
            )

            if is_recommended:
                reason = f"Improves Brier by {brier_imp:.4f}, ECE by {ece_imp:.4f}, slope by {slope_imp:.3f}"
            elif c_change < -0.02:
                reason = f"Damages C-index by {-c_change:.3f}"
            else:
                reason = "Does not improve all calibration metrics"

            comparisons.append(CalibrationComparison(
                method=method.value,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                brier_improvement=brier_imp,
                ece_improvement=ece_imp,
                slope_improvement=slope_imp,
                concordance_change=c_change,
                is_recommended=is_recommended,
                recommendation_reason=reason,
            ))

        except Exception as e:
            logger.warning("calibration_method_failed", method=method.value, error=str(e))

    return comparisons
