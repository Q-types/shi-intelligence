#!/usr/bin/env python3
"""
Scientific Model Training Script.

End-to-end pipeline for training rug pull detection models.

Steps:
1. Collect training data (if not exists)
2. Load and prepare dataset
3. Train ensemble with temporal CV
4. Run refutation tests
5. Validate and report metrics
6. Save model if passes thresholds
"""

import asyncio
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import structlog

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.models.scientific_training import (
    ScientificTrainingPipeline,
    ScientificTrainingConfig,
    ClassImbalanceConfig,
    TemporalCVConfig,
    RefutationConfig,
    ModelType,
)

logger = structlog.get_logger()


def load_training_data(data_dir: Path) -> pd.DataFrame:
    """Load collected training data."""
    collected_file = data_dir / "collected" / "training_dataset.csv"

    if not collected_file.exists():
        raise FileNotFoundError(
            f"Training data not found at {collected_file}. "
            "Run collect_training_data.py first."
        )

    df = pd.read_csv(collected_file)
    logger.info("data_loaded", samples=len(df), file=str(collected_file))

    return df


def print_validation_report(ensemble):
    """Print comprehensive validation report."""
    val = ensemble.validation

    print("\n" + "=" * 70)
    print("SCIENTIFIC MODEL VALIDATION REPORT")
    print("=" * 70)
    print(f"Version: {ensemble.version}")
    print(f"Trained: {ensemble.trained_at.isoformat()}")
    print(f"Samples: {ensemble.training_samples}")

    print("\n" + "-" * 70)
    print("INDIVIDUAL MODEL METRICS")
    print("-" * 70)

    for model_type, metrics in val.individual_metrics.items():
        print(f"\n{model_type.value.upper()}")
        print(f"  ROC-AUC:   {metrics.roc_auc:.4f}")
        print(f"  Brier:     {metrics.brier_score:.4f}")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall:    {metrics.recall:.4f}")
        print(f"  F1:        {metrics.f1:.4f}")

        if metrics.cv_scores:
            print(f"  CV Mean:   {metrics.cv_mean:.4f} (+/- {metrics.cv_std:.4f})")

        print(f"  Refutation: {'PASSED' if metrics.refutation_passed else 'FAILED'}")

        if metrics.refutation_details:
            for key, value in metrics.refutation_details.items():
                if not key.endswith("_error"):
                    print(f"    - {key}: {value}")

        print(f"  Confusion Matrix:")
        for row in metrics.confusion_matrix:
            print(f"    {row}")

    print("\n" + "-" * 70)
    print("ENSEMBLE METRICS")
    print("-" * 70)
    print(f"  ROC-AUC:           {val.ensemble_roc_auc:.4f}")
    print(f"  Brier Score:       {val.ensemble_brier_score:.4f}")
    print(f"  Precision:         {val.ensemble_precision:.4f}")
    print(f"  Recall:            {val.ensemble_recall:.4f}")
    print(f"  F1:                {val.ensemble_f1:.4f}")
    print(f"  Optimal Threshold: {val.optimal_threshold:.4f}")

    print("\n" + "-" * 70)
    print("DEPLOYMENT STATUS")
    print("-" * 70)

    if val.passes_thresholds:
        print("  STATUS: DEPLOYABLE")
    else:
        print("  STATUS: NOT DEPLOYABLE")
        print("\n  Failed Checks:")
        for check in val.failed_checks:
            print(f"    - {check}")

    print("\n" + "=" * 70)


def save_ensemble(ensemble, output_dir: Path) -> Path:
    """Save trained ensemble to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"ensemble_{ensemble.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    output_path = output_dir / filename

    with open(output_path, "wb") as f:
        pickle.dump(ensemble, f)

    logger.info("ensemble_saved", path=str(output_path))
    return output_path


async def main():
    """Main training pipeline."""
    print("=" * 70)
    print("SHI SCIENTIFIC MODEL TRAINING")
    print("=" * 70)

    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "training"
    model_dir = Path(__file__).parent.parent / "models" / "trained"

    # Check if we need to collect data first
    collected_file = data_dir / "collected" / "training_dataset.csv"

    if not collected_file.exists():
        print("\nTraining data not found. Running collection first...")
        print("-" * 70)

        from collect_training_data import TrainingDataCollector

        collector = TrainingDataCollector(data_dir / "collected")
        samples = await collector.collect_full_dataset(
            solrpds_limit=50,
            verified_rugs_limit=30,
            safe_tokens_limit=14,
        )

        if samples:
            collector.save_dataset(samples)
            collector.save_failed_tokens()
        else:
            print("ERROR: No data collected!")
            return

    # Load data
    print("\n" + "-" * 70)
    print("LOADING TRAINING DATA")
    print("-" * 70)

    df = load_training_data(data_dir)

    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Rug pulls:     {(df['label'] == 'rug').sum()}")
    print(f"  Safe tokens:   {(df['label'] == 'safe').sum()}")

    if len(df) < 20:
        print("\nWARNING: Small dataset! Results may be unreliable.")

    # Configure training pipeline
    config = ScientificTrainingConfig(
        feature_columns=[
            "hhi",
            "gini",
            "entropy",
            "whale_dominance_top10",
            "whale_dominance_top5",
            "top_holder_share",
            "holder_count",
        ],
        min_roc_auc=0.60,  # Slightly lower for small datasets
        max_brier_score=0.30,
        min_precision=0.40,
        min_recall=0.40,
        ensemble_weights={
            ModelType.XGBOOST: 0.6,
            ModelType.ISOLATION_FOREST: 0.4,
        },
        imbalance=ClassImbalanceConfig(
            use_smote=True,
            smote_k_neighbors=3,  # Lower for small datasets
            use_class_weights=True,
        ),
        temporal_cv=TemporalCVConfig(
            n_splits=3,  # Fewer folds for small datasets
        ),
        refutation=RefutationConfig(
            run_placebo_test=True,
            run_subset_test=True,
            subset_fraction=0.7,
            run_random_feature_test=True,
        ),
    )

    # Train ensemble
    print("\n" + "-" * 70)
    print("TRAINING ENSEMBLE")
    print("-" * 70)

    pipeline = ScientificTrainingPipeline(config)

    try:
        ensemble = pipeline.train_ensemble(
            data=df,
            label_column="label",
            version="v2.0.0",
        )

        # Print validation report
        print_validation_report(ensemble)

        # Save model
        if ensemble.is_deployable:
            model_path = save_ensemble(ensemble, model_dir)
            print(f"\nModel saved to: {model_path}")
        else:
            print("\nModel NOT saved (failed validation thresholds)")

            # Ask if user wants to save anyway
            print("\nDo you want to save the model anyway for debugging? (y/n)")
            # In script mode, save anyway
            model_path = save_ensemble(ensemble, model_dir / "debug")
            print(f"Debug model saved to: {model_path}")

        # Feature importance
        if ModelType.XGBOOST in ensemble.models:
            xgb = ensemble.models[ModelType.XGBOOST]
            importance = dict(zip(
                config.feature_columns,
                xgb.feature_importances_,
            ))

            print("\n" + "-" * 70)
            print("FEATURE IMPORTANCE (XGBoost)")
            print("-" * 70)

            for feature, score in sorted(importance.items(), key=lambda x: -x[1]):
                bar = "#" * int(score * 50)
                print(f"  {feature:25} {score:.4f} {bar}")

    except Exception as e:
        logger.error("training_failed", error=str(e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
