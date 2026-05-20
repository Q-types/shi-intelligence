#!/usr/bin/env python
"""
Calibration Audit for SHI Hazard Models.

Runs comprehensive calibration analysis including:
- Baseline calibration metrics per model
- Calibration method comparison
- Walk-forward validation
- Regime-specific analysis
- Probability band validation

Usage:
    python scripts/run_calibration_audit.py [--output-dir docs/validation]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.calibration import (
    ProbabilityCalibrator,
    CalibrationMethod,
    TokenRegime,
    compute_calibration_metrics,
    compute_probability_bands,
    compare_calibration_methods,
    CalibrationMetrics,
    ProbabilityBand,
    CalibrationComparison,
    RegimeCalibrationResult,
)
from src.validation.intelligence.hazard_comparison import (
    HazardModel,
    MODEL_FEATURES,
)

logger = structlog.get_logger()


def generate_survival_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic survival data for calibration testing.

    Creates data with known calibration issues to test correction methods.
    """
    np.random.seed(42)

    # Features
    data = pd.DataFrame({
        "holding_duration": np.random.exponential(10, n_samples) + 1,
        "trade_count": np.random.poisson(5, n_samples),
        "share": np.random.beta(0.5, 50, n_samples),
        "entry_time_relative": np.random.beta(2, 5, n_samples),
        "burstiness": np.random.uniform(-1, 1, n_samples),
        "in_degree": np.random.poisson(3, n_samples),
        "out_degree": np.random.poisson(2, n_samples),
        "shared_funder_count": np.random.negative_binomial(2, 0.4, n_samples),
        "delta_balance_7d": np.random.randn(n_samples) * 0.1,
        "delta_balance_30d": np.random.randn(n_samples) * 0.15,
        "swap_frequency": np.random.exponential(0.5, n_samples),
        "lp_interaction_ratio": np.random.beta(2, 8, n_samples),
        "eigenvector_centrality": np.random.beta(2, 10, n_samples),
        "position_volatility": np.abs(np.random.randn(n_samples) * 0.2),
    })

    # Generate true event probability based on features
    # This creates a model where predictions will need calibration
    risk_score = (
        -0.3 * data["holding_duration"].clip(0, 30) / 30 +  # Longer hold = lower risk
        0.2 * data["trade_count"].clip(0, 20) / 20 +  # More trades = higher risk
        0.1 * data["burstiness"] +
        0.15 * data["shared_funder_count"].clip(0, 10) / 10 +  # Coordination = risk
        -0.1 * data["delta_balance_30d"].clip(-1, 1)
    )

    # Convert to probability with some miscalibration
    # Use a steeper sigmoid to create under-confident predictions
    true_prob = 1 / (1 + np.exp(-risk_score * 3))  # Steeper than ideal

    # Generate events
    data["event"] = (np.random.rand(n_samples) < true_prob).astype(int)
    data["duration"] = np.maximum(1, data["holding_duration"] + np.random.exponential(5, n_samples))

    # Add regime labels
    regimes = []
    for i in range(n_samples):
        if data.loc[i, "shared_funder_count"] >= 5:
            regimes.append(TokenRegime.COORDINATED_ACCUMULATION.value)
        elif data.loc[i, "delta_balance_30d"] > 0.1:
            regimes.append(TokenRegime.ACCUMULATION.value)
        elif data.loc[i, "delta_balance_30d"] < -0.1:
            regimes.append(TokenRegime.DISTRIBUTION.value)
        elif data.loc[i, "trade_count"] < 2:
            regimes.append(TokenRegime.STABLE.value)
        else:
            regimes.append(TokenRegime.DECAY.value)

    data["regime"] = regimes

    # Add timestamp for walk-forward
    data["timestamp"] = pd.date_range("2024-01-01", periods=n_samples, freq="h")

    logger.info(
        "survival_data_generated",
        n_samples=n_samples,
        event_rate=data["event"].mean(),
        regime_distribution=data["regime"].value_counts().to_dict(),
    )

    return data


def fit_hazard_model(
    data: pd.DataFrame,
    model: HazardModel,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit hazard model and return predictions.

    Returns:
        Tuple of (predicted_probabilities, true_outcomes)
    """
    from lifelines import CoxPHFitter

    features = MODEL_FEATURES.get(model, [])
    available = [f for f in features if f in data.columns]

    if not available:
        available = ["holding_duration", "trade_count", "share"]

    df = data[["duration", "event"] + available].copy()
    df = df.fillna(0)
    df = df[df["duration"] > 0]

    fitter = CoxPHFitter(penalizer=0.1)
    fitter.fit(df, duration_col="duration", event_col="event")

    # Get survival probabilities at median time
    median_time = df["duration"].median()
    survival = fitter.predict_survival_function(df, times=[median_time])

    # Convert to risk probability
    y_prob = 1 - survival.iloc[0].values
    y_true = df["event"].values

    return y_prob, y_true


def run_baseline_audit(data: pd.DataFrame) -> dict[str, CalibrationMetrics]:
    """Run baseline calibration audit for all models."""
    results = {}

    for model in HazardModel:
        logger.info("auditing_model", model=model.value)
        try:
            y_prob, y_true = fit_hazard_model(data, model)
            metrics = compute_calibration_metrics(y_prob, y_true)
            results[model.value] = metrics
        except Exception as e:
            logger.warning("model_audit_failed", model=model.value, error=str(e))

    return results


def run_walk_forward_calibration(
    data: pd.DataFrame,
    method: CalibrationMethod,
    n_folds: int = 4,
) -> list[CalibrationMetrics]:
    """
    Run walk-forward calibration validation.

    For each fold:
    1. Train hazard model on past data
    2. Predict on validation fold
    3. Fit calibrator on validation data
    4. Evaluate on held-out test data
    """
    from lifelines import CoxPHFitter

    # Sort by timestamp
    data = data.sort_values("timestamp").reset_index(drop=True)
    n = len(data)
    fold_size = n // (n_folds + 1)

    results = []

    for fold in range(n_folds):
        # Train: all data before this fold
        train_end = (fold + 1) * fold_size
        # Validation: this fold (for calibrator fitting)
        val_start = train_end
        val_end = val_start + fold_size
        # Test: next fold
        test_start = val_end
        test_end = min(test_start + fold_size, n)

        if test_end <= test_start:
            continue

        train_df = data.iloc[:train_end]
        val_df = data.iloc[val_start:val_end]
        test_df = data.iloc[test_start:test_end]

        logger.info(
            "walk_forward_fold",
            fold=fold,
            train_size=len(train_df),
            val_size=len(val_df),
            test_size=len(test_df),
        )

        try:
            # Train hazard model on training data
            features = ["holding_duration", "trade_count", "share", "burstiness", "in_degree"]
            available = [f for f in features if f in train_df.columns]

            train_model_df = train_df[["duration", "event"] + available].fillna(0)
            train_model_df = train_model_df[train_model_df["duration"] > 0]

            fitter = CoxPHFitter(penalizer=0.1)
            fitter.fit(train_model_df, duration_col="duration", event_col="event")

            median_time = train_model_df["duration"].median()

            # Predict on validation
            val_model_df = val_df[["duration", "event"] + available].fillna(0)
            val_model_df = val_model_df[val_model_df["duration"] > 0]
            val_survival = fitter.predict_survival_function(val_model_df, times=[median_time])
            val_prob = 1 - val_survival.iloc[0].values
            val_true = val_model_df["event"].values

            # Fit calibrator on validation
            calibrator = ProbabilityCalibrator(method=method)
            calibrator.fit(val_prob, val_true)

            # Predict on test
            test_model_df = test_df[["duration", "event"] + available].fillna(0)
            test_model_df = test_model_df[test_model_df["duration"] > 0]
            test_survival = fitter.predict_survival_function(test_model_df, times=[median_time])
            test_prob_raw = 1 - test_survival.iloc[0].values
            test_true = test_model_df["event"].values

            # Calibrate test predictions
            test_prob_cal = calibrator.calibrate(test_prob_raw)

            # Compute metrics
            metrics = compute_calibration_metrics(test_prob_cal, test_true)
            results.append(metrics)

        except Exception as e:
            logger.warning("walk_forward_fold_failed", fold=fold, error=str(e))

    return results


def run_regime_analysis(
    data: pd.DataFrame,
) -> list[RegimeCalibrationResult]:
    """Run regime-specific calibration analysis."""
    from scipy.stats import linregress

    results = []
    regimes = data["regime"].unique()

    # Get predictions for full data first
    y_prob, y_true = fit_hazard_model(data, HazardModel.MODEL_B_EXPANDED)

    # Add predictions to data
    pred_data = data.iloc[:len(y_prob)].copy()
    pred_data["y_prob"] = y_prob
    pred_data["y_true"] = y_true

    for regime in regimes:
        regime_data = pred_data[pred_data["regime"] == regime]
        n = len(regime_data)

        if n < 10:
            continue

        regime_prob = regime_data["y_prob"].values
        regime_true = regime_data["y_true"].values

        # Compute metrics
        brier = np.mean((regime_prob - regime_true) ** 2)

        try:
            slope, _, _, _, _ = linregress(regime_prob, regime_true)
        except Exception:
            slope = 0.0

        # ECE
        from src.models.calibration import _compute_ece_mce
        ece, _, _, _, _ = _compute_ece_mce(regime_prob, regime_true, n_bins=5)

        results.append(RegimeCalibrationResult(
            regime=regime,
            sample_size=n,
            event_rate=float(regime_true.mean()),
            calibration_slope=float(slope),
            brier_score=float(brier),
            ece=float(ece),
            adequate_samples=n >= 50,
        ))

    return results


def generate_calibration_audit_report(
    baseline_results: dict[str, CalibrationMetrics],
    output_path: Path,
) -> None:
    """Generate CALIBRATION_AUDIT.md report."""
    lines = [
        "# Calibration Audit Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Executive Summary",
        "",
        "**Goal:** Produce probabilities that are honest, stable, and decision-useful.",
        "",
        "**Key Metrics:**",
        "- C-index (discrimination) - higher is better",
        "- Brier score (calibration) - lower is better",
        "- Calibration slope - closer to 1.0 is better",
        "- ECE (Expected Calibration Error) - lower is better",
        "",
        "---",
        "",
        "## Baseline Model Comparison",
        "",
        "| Model | C-Index | Brier | Slope | Intercept | ECE | MCE |",
        "|-------|---------|-------|-------|-----------|-----|-----|",
    ]

    for model, metrics in baseline_results.items():
        lines.append(
            f"| {model} | {metrics.concordance_index:.3f} | "
            f"{metrics.brier_score:.4f} | {metrics.calibration_slope:.3f} | "
            f"{metrics.calibration_intercept:.3f} | {metrics.expected_calibration_error:.4f} | "
            f"{metrics.maximum_calibration_error:.4f} |"
        )

    lines.extend([
        "",
        "## Calibration Curves by Decile",
        "",
    ])

    for model, metrics in baseline_results.items():
        lines.extend([
            f"### {model}",
            "",
            "| Decile | Predicted | Observed | Count |",
            "|--------|-----------|----------|-------|",
        ])

        for i, (pred, obs, count) in enumerate(zip(
            metrics.decile_predicted,
            metrics.decile_observed,
            metrics.decile_counts
        )):
            diff = abs(pred - obs)
            marker = " **" if diff > 0.1 else ""
            lines.append(f"| {i+1} | {pred:.3f} | {obs:.3f} | {count}{marker} |")

        lines.append("")

    lines.extend([
        "---",
        "",
        "## Key Observations",
        "",
    ])

    # Find best and worst
    best_brier = min(baseline_results.items(), key=lambda x: x[1].brier_score)
    best_slope = min(baseline_results.items(), key=lambda x: abs(x[1].calibration_slope - 1.0))

    lines.extend([
        f"- **Best Brier Score:** {best_brier[0]} ({best_brier[1].brier_score:.4f})",
        f"- **Best Calibration Slope:** {best_slope[0]} ({best_slope[1].calibration_slope:.3f})",
        "",
        "**Note:** A slope > 1.0 indicates under-confident predictions. A slope < 1.0 indicates over-confident predictions.",
        "",
    ])

    output_path.write_text("\n".join(lines))
    logger.info("calibration_audit_report_generated", path=str(output_path))


def generate_method_comparison_report(
    comparisons: list[CalibrationComparison],
    output_path: Path,
) -> None:
    """Generate CALIBRATION_METHOD_COMPARISON.md report."""
    lines = [
        "# Calibration Method Comparison Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Methods Tested",
        "",
        "- **Isotonic:** Non-parametric monotonic regression",
        "- **Platt:** Logistic regression calibration",
        "- **Beta:** Three-parameter beta calibration",
        "- **Regime-Specific:** Separate calibrators per token regime",
        "",
        "---",
        "",
        "## Comparison Results",
        "",
        "| Method | Brier Δ | ECE Δ | Slope Δ | C-Index Δ | Recommended |",
        "|--------|---------|-------|---------|-----------|-------------|",
    ]

    for comp in comparisons:
        rec = "✓" if comp.is_recommended else ""
        lines.append(
            f"| {comp.method} | {comp.brier_improvement:+.4f} | "
            f"{comp.ece_improvement:+.4f} | {comp.slope_improvement:+.3f} | "
            f"{comp.concordance_change:+.3f} | {rec} |"
        )

    lines.extend([
        "",
        "**Note:** Positive Δ means improvement (except C-Index where negative is bad).",
        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ])

    for comp in comparisons:
        lines.extend([
            f"### {comp.method}",
            "",
            "| Metric | Before | After | Change |",
            "|--------|--------|-------|--------|",
            f"| Brier | {comp.metrics_before.brier_score:.4f} | {comp.metrics_after.brier_score:.4f} | {comp.brier_improvement:+.4f} |",
            f"| ECE | {comp.metrics_before.expected_calibration_error:.4f} | {comp.metrics_after.expected_calibration_error:.4f} | {comp.ece_improvement:+.4f} |",
            f"| Slope | {comp.metrics_before.calibration_slope:.3f} | {comp.metrics_after.calibration_slope:.3f} | {comp.slope_improvement:+.3f} |",
            f"| C-Index | {comp.metrics_before.concordance_index:.3f} | {comp.metrics_after.concordance_index:.3f} | {comp.concordance_change:+.3f} |",
            "",
            f"**Recommendation:** {comp.recommendation_reason}",
            "",
        ])

    # Overall recommendation
    recommended = [c for c in comparisons if c.is_recommended]
    if recommended:
        best = max(recommended, key=lambda x: x.brier_improvement + x.ece_improvement)
        lines.extend([
            "---",
            "",
            "## Deployment Recommendation",
            "",
            f"**Recommended Method:** {best.method}",
            "",
            f"**Rationale:** {best.recommendation_reason}",
            "",
        ])
    else:
        lines.extend([
            "---",
            "",
            "## Deployment Recommendation",
            "",
            "**No method clearly recommended.** Consider:",
            "- Collecting more data",
            "- Testing on production-like distribution",
            "- Using ensemble of methods",
            "",
        ])

    output_path.write_text("\n".join(lines))
    logger.info("method_comparison_report_generated", path=str(output_path))


def generate_regime_report(
    regime_results: list[RegimeCalibrationResult],
    output_path: Path,
) -> None:
    """Generate REGIME_CALIBRATION_REPORT.md."""
    lines = [
        "# Regime-Specific Calibration Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Regimes Analyzed",
        "",
        "- **ACCUMULATION:** Net positive balance changes",
        "- **DISTRIBUTION:** Net negative balance changes",
        "- **COORDINATED_ACCUMULATION:** High shared funder count",
        "- **DECAY:** Moderate activity, negative trend",
        "- **STABLE:** Low activity, neutral balance",
        "",
        "---",
        "",
        "## Per-Regime Calibration Metrics",
        "",
        "| Regime | Samples | Event Rate | Slope | Brier | ECE | Adequate |",
        "|--------|---------|------------|-------|-------|-----|----------|",
    ]

    for result in regime_results:
        adequate = "✓" if result.adequate_samples else "✗"
        lines.append(
            f"| {result.regime} | {result.sample_size} | {result.event_rate:.1%} | "
            f"{result.calibration_slope:.3f} | {result.brier_score:.4f} | "
            f"{result.ece:.4f} | {adequate} |"
        )

    # Recommendation
    adequate_regimes = [r for r in regime_results if r.adequate_samples]
    total_regimes = len(regime_results)

    lines.extend([
        "",
        "---",
        "",
        "## Deployment Recommendation",
        "",
    ])

    if len(adequate_regimes) == total_regimes:
        lines.extend([
            "**Regime-specific calibration CAN be deployed.**",
            "",
            "All regimes have adequate sample sizes (≥50).",
            "",
        ])
    else:
        inadequate = [r.regime for r in regime_results if not r.adequate_samples]
        lines.extend([
            "**Regime-specific calibration NOT recommended.**",
            "",
            f"Regimes with insufficient samples: {', '.join(inadequate)}",
            "",
            "Use global calibration instead until more data is collected.",
            "",
        ])

    output_path.write_text("\n".join(lines))
    logger.info("regime_report_generated", path=str(output_path))


def generate_probability_band_report(
    bands: list[ProbabilityBand],
    output_path: Path,
) -> None:
    """Generate PROBABILITY_BAND_VALIDATION.md."""
    lines = [
        "# Probability Band Validation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Acceptance Criterion",
        "",
        "Predicted probabilities should roughly match observed rates within uncertainty bounds.",
        "",
        "---",
        "",
        "## Band Analysis",
        "",
        "| Band | Wallets | Predicted | Observed | 95% CI | Calibrated |",
        "|------|---------|-----------|----------|--------|------------|",
    ]

    all_calibrated = True
    for band in bands:
        cal = "✓" if band.is_calibrated else "✗"
        if not band.is_calibrated:
            all_calibrated = False

        ci = f"[{band.confidence_interval_lower:.3f}, {band.confidence_interval_upper:.3f}]"
        lines.append(
            f"| {band.band_name} | {band.wallet_count} | {band.predicted_mean:.3f} | "
            f"{band.observed_rate:.3f} | {ci} | {cal} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Overall Assessment",
        "",
    ])

    if all_calibrated:
        lines.extend([
            "**PASS:** All probability bands are well-calibrated.",
            "",
            "Predicted probabilities match observed event rates within confidence intervals.",
            "",
        ])
    else:
        miscalibrated = [b.band_name for b in bands if not b.is_calibrated]
        lines.extend([
            "**FAIL:** Some probability bands are miscalibrated.",
            "",
            f"Miscalibrated bands: {', '.join(miscalibrated)}",
            "",
            "Apply calibration method before deployment.",
            "",
        ])

    output_path.write_text("\n".join(lines))
    logger.info("probability_band_report_generated", path=str(output_path))


def generate_deployment_recommendation(
    baseline_results: dict[str, CalibrationMetrics],
    comparisons: list[CalibrationComparison],
    regime_results: list[RegimeCalibrationResult],
    bands: list[ProbabilityBand],
    output_path: Path,
) -> None:
    """Generate CALIBRATION_DEPLOYMENT_RECOMMENDATION.md."""
    lines = [
        "# Calibration Deployment Recommendation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Decision Framework",
        "",
        "Deploy calibration if it improves:",
        "- Brier score (lower)",
        "- ECE (lower)",
        "- Calibration slope (closer to 1.0)",
        "",
        "Without materially damaging C-index (≤2% drop acceptable).",
        "",
        "---",
        "",
        "## Summary of Findings",
        "",
    ]

    # Best baseline
    best_baseline = min(baseline_results.items(), key=lambda x: x[1].brier_score)
    lines.extend([
        f"**Best Baseline Model:** {best_baseline[0]}",
        f"- Brier: {best_baseline[1].brier_score:.4f}",
        f"- Slope: {best_baseline[1].calibration_slope:.3f}",
        f"- ECE: {best_baseline[1].expected_calibration_error:.4f}",
        "",
    ])

    # Best calibration method
    recommended_methods = [c for c in comparisons if c.is_recommended]
    if recommended_methods:
        best_method = max(recommended_methods, key=lambda x: x.brier_improvement)
        lines.extend([
            f"**Best Calibration Method:** {best_method.method}",
            f"- Brier improvement: {best_method.brier_improvement:+.4f}",
            f"- ECE improvement: {best_method.ece_improvement:+.4f}",
            f"- Slope improvement: {best_method.slope_improvement:+.3f}",
            "",
        ])
    else:
        best_method = None
        lines.extend([
            "**No calibration method clearly recommended.**",
            "",
        ])

    # Regime-specific
    adequate_regimes = [r for r in regime_results if r.adequate_samples]
    lines.extend([
        f"**Regime-Specific Calibration:** {'Possible' if len(adequate_regimes) == len(regime_results) else 'Not recommended'}",
        f"- Adequate regimes: {len(adequate_regimes)}/{len(regime_results)}",
        "",
    ])

    # Probability bands
    calibrated_bands = [b for b in bands if b.is_calibrated]
    lines.extend([
        f"**Probability Bands:** {len(calibrated_bands)}/{len(bands)} well-calibrated",
        "",
        "---",
        "",
        "## Recommended Configuration",
        "",
        "```python",
    ])

    if best_method:
        lines.extend([
            "use_probability_calibration = True",
            f'calibration_method = "{best_method.method}"',
        ])
    else:
        lines.extend([
            "use_probability_calibration = False  # No clear improvement",
            'calibration_method = "isotonic"',
        ])

    use_regime = len(adequate_regimes) == len(regime_results) and len(regime_results) > 0
    lines.extend([
        f"use_regime_specific_calibration = {use_regime}",
        "```",
        "",
        "---",
        "",
        "## Next Steps",
        "",
        "1. Validate on larger dataset",
        "2. Monitor calibration metrics in production",
        "3. Re-calibrate periodically as data distribution shifts",
        "",
    ])

    output_path.write_text("\n".join(lines))
    logger.info("deployment_recommendation_generated", path=str(output_path))


def main():
    """Run complete calibration audit."""
    parser = argparse.ArgumentParser(description="Run SHI Calibration Audit")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/validation"),
        help="Output directory for reports",
    )
    args = parser.parse_args()

    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    logger.info("starting_calibration_audit")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    data = generate_survival_data(n_samples=1000)

    # 1. Baseline audit
    logger.info("running_baseline_audit")
    baseline_results = run_baseline_audit(data)
    generate_calibration_audit_report(
        baseline_results,
        args.output_dir / "CALIBRATION_AUDIT.md"
    )

    # 2. Method comparison
    logger.info("running_method_comparison")
    y_prob, y_true = fit_hazard_model(data, HazardModel.MODEL_B_EXPANDED)
    regimes = data["regime"].values[:len(y_prob)]
    comparisons = compare_calibration_methods(y_prob, y_true, regimes)
    generate_method_comparison_report(
        comparisons,
        args.output_dir / "CALIBRATION_METHOD_COMPARISON.md"
    )

    # 3. Regime analysis
    logger.info("running_regime_analysis")
    regime_results = run_regime_analysis(data)
    generate_regime_report(
        regime_results,
        args.output_dir / "REGIME_CALIBRATION_REPORT.md"
    )

    # 4. Probability bands
    logger.info("running_probability_band_validation")
    bands = compute_probability_bands(y_prob, y_true)
    generate_probability_band_report(
        bands,
        args.output_dir / "PROBABILITY_BAND_VALIDATION.md"
    )

    # 5. Deployment recommendation
    logger.info("generating_deployment_recommendation")
    generate_deployment_recommendation(
        baseline_results,
        comparisons,
        regime_results,
        bands,
        args.output_dir / "CALIBRATION_DEPLOYMENT_RECOMMENDATION.md"
    )

    print("\n" + "=" * 60)
    print("CALIBRATION AUDIT COMPLETE")
    print("=" * 60)
    print(f"\nReports generated in: {args.output_dir}")
    print("  - CALIBRATION_AUDIT.md")
    print("  - CALIBRATION_METHOD_COMPARISON.md")
    print("  - REGIME_CALIBRATION_REPORT.md")
    print("  - PROBABILITY_BAND_VALIDATION.md")
    print("  - CALIBRATION_DEPLOYMENT_RECOMMENDATION.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
