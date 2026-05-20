#!/usr/bin/env python
"""
SHI Intelligence Upgrade Validation Gate.

Runs comprehensive validation of the clustering intelligence upgrade
and generates deployment reports.

Usage:
    python scripts/run_validation_gate.py [--output-dir docs/validation]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.intelligence.pipeline_comparison import (
    ClusteringValidator,
    ClusteringPipeline,
)
from src.validation.intelligence.hazard_comparison import HazardModelValidator
from src.validation.intelligence.ablation_runner import AblationRunner
from src.validation.intelligence.missingness_analysis import MissingnessAnalyzer
from src.validation.intelligence.report_generator import ReportGenerator
from src.clustering.archetypes import WalletFeatureVector

logger = structlog.get_logger()


def generate_synthetic_data(n_samples: int = 500) -> tuple:
    """
    Generate synthetic data for validation.

    In production, this would load real wallet data from the database.

    Returns:
        Tuple of (features, feature_names, wallet_vectors, survival_data, embeddings)
    """
    logger.info("generating_synthetic_validation_data", n_samples=n_samples)

    np.random.seed(42)

    # Feature names matching WalletFeatureVector
    feature_names = [
        "balance", "share", "rank",
        "entry_time_relative", "holding_duration", "position_volatility",
        "delta_balance_7d", "delta_balance_30d",
        "trade_count", "burstiness", "swap_frequency", "lp_interaction_ratio",
        "in_degree", "out_degree", "eigenvector_centrality", "shared_funder_count",
        "total_funding_received", "largest_funder_share", "funding_hhi",
        "unrealized_pnl_ratio", "liquidity_usd_current", "sell_pressure_vs_liquidity",
    ]

    n_features = len(feature_names)

    # Generate features with realistic distributions
    features = np.zeros((n_samples, n_features))

    # Distribution features (log-normal)
    features[:, 0] = np.exp(np.random.randn(n_samples) * 2 + 5)  # balance
    features[:, 1] = np.random.beta(1, 100, n_samples)  # share (power law)
    features[:, 2] = np.arange(1, n_samples + 1)  # rank

    # Temporal features
    features[:, 3] = np.random.uniform(0, 1, n_samples)  # entry_time_relative
    features[:, 4] = np.exp(np.random.randn(n_samples) + 2)  # holding_duration
    features[:, 5] = np.abs(np.random.randn(n_samples) * 0.2)  # position_volatility

    # Flow features (can be negative)
    features[:, 6] = np.random.randn(n_samples) * 0.1  # delta_balance_7d
    features[:, 7] = np.random.randn(n_samples) * 0.15  # delta_balance_30d

    # Trading features
    features[:, 8] = np.random.poisson(5, n_samples)  # trade_count
    features[:, 9] = np.random.uniform(-1, 1, n_samples)  # burstiness
    features[:, 10] = np.random.exponential(0.5, n_samples)  # swap_frequency
    features[:, 11] = np.random.beta(2, 8, n_samples)  # lp_interaction_ratio

    # Graph features
    features[:, 12] = np.random.poisson(3, n_samples)  # in_degree
    features[:, 13] = np.random.poisson(2, n_samples)  # out_degree
    features[:, 14] = np.random.beta(2, 10, n_samples)  # eigenvector_centrality
    features[:, 15] = np.random.poisson(1, n_samples)  # shared_funder_count

    # Weighted graph features
    features[:, 16] = np.exp(np.random.randn(n_samples) + 1)  # total_funding_received
    features[:, 17] = np.random.beta(3, 2, n_samples)  # largest_funder_share
    features[:, 18] = np.random.beta(2, 5, n_samples)  # funding_hhi

    # Price/liquidity features
    features[:, 19] = np.random.randn(n_samples) * 0.5  # unrealized_pnl_ratio
    features[:, 20] = np.exp(np.random.randn(n_samples) * 1.5 + 10)  # liquidity_usd_current
    features[:, 21] = np.random.exponential(0.3, n_samples)  # sell_pressure_vs_liquidity

    # Add some missingness (realistic patterns)
    # Price data often missing for new tokens
    missing_price_mask = np.random.rand(n_samples) < 0.15
    features[missing_price_mask, 19:22] = np.nan

    # Graph data sometimes missing
    missing_graph_mask = np.random.rand(n_samples) < 0.05
    features[missing_graph_mask, 12:16] = np.nan

    # Create WalletFeatureVectors
    wallet_vectors = []
    for i in range(n_samples):
        wv = WalletFeatureVector(
            wallet=f"wallet_{i:04d}",
            balance=float(features[i, 0]),
            share=float(features[i, 1]),
            rank=int(features[i, 2]),
            entry_time_relative=float(features[i, 3]),
            holding_duration=float(features[i, 4]),
            position_volatility=float(features[i, 5]),
            delta_balance_7d=float(features[i, 6]),
            delta_balance_30d=float(features[i, 7]),
            trade_count=int(features[i, 8]),
            burstiness=float(features[i, 9]),
            swap_frequency=float(features[i, 10]),
            lp_interaction_ratio=float(features[i, 11]),
            in_degree=int(features[i, 12]) if not np.isnan(features[i, 12]) else 0,
            out_degree=int(features[i, 13]) if not np.isnan(features[i, 13]) else 0,
            eigenvector_centrality=float(features[i, 14]) if not np.isnan(features[i, 14]) else 0.0,
            shared_funder_count=int(features[i, 15]) if not np.isnan(features[i, 15]) else 0,
        )
        wallet_vectors.append(wv)

    # Generate survival data
    # Event probability influenced by features
    base_hazard = 0.3
    risk_score = (
        -0.5 * features[:, 4] +  # longer hold -> lower risk
        0.3 * features[:, 8] +   # more trades -> higher risk
        0.2 * np.nan_to_num(features[:, 19], 0)  # negative PnL -> higher risk
    )
    event_prob = 1 / (1 + np.exp(-risk_score))

    survival_data = pd.DataFrame({
        "duration": np.maximum(1, features[:, 4] + np.random.exponential(5, n_samples)),
        "event": (np.random.rand(n_samples) < event_prob).astype(int),
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="H"),
    })

    # Add features to survival data
    for i, name in enumerate(feature_names):
        survival_data[name] = features[:, i]

    # Generate Node2Vec-like embeddings (reduced dimensionality)
    embedding_dim = 8
    embeddings = np.random.randn(n_samples, embedding_dim)
    # Add some structure
    embeddings[:n_samples//3, :4] += 2  # Cluster 1
    embeddings[n_samples//3:2*n_samples//3, 4:] += 2  # Cluster 2

    logger.info(
        "synthetic_data_generated",
        n_samples=n_samples,
        n_features=n_features,
        n_events=survival_data["event"].sum(),
        missing_pct=np.isnan(features).mean() * 100,
    )

    return features, feature_names, wallet_vectors, survival_data, embeddings


def run_validation_gate(output_dir: Path) -> dict:
    """
    Run the complete validation gate.

    Args:
        output_dir: Directory for report output

    Returns:
        Dict with validation results summary
    """
    logger.info("starting_validation_gate", output_dir=str(output_dir))

    # Generate or load data
    features, feature_names, wallet_vectors, survival_data, embeddings = generate_synthetic_data()

    results = {}

    # 1. Clustering Baseline Comparison
    logger.info("running_clustering_comparison")
    clustering_validator = ClusteringValidator(min_cluster_size=5, n_bootstrap=10)
    clustering_comparison = clustering_validator.compare_pipelines(
        features=features,
        feature_names=feature_names,
        wallet_vectors=wallet_vectors,
        graph_embeddings=embeddings,
    )
    results["clustering"] = clustering_comparison

    # 2. Feature Ablation
    logger.info("running_ablation_study")
    ablation_runner = AblationRunner(min_cluster_size=5)
    ablation_results = ablation_runner.run_full_ablation(
        features=features,
        feature_names=feature_names,
        survival_data=survival_data,
    )
    results["ablation"] = ablation_results

    # 3. Hazard Model Comparison
    logger.info("running_hazard_comparison")
    hazard_validator = HazardModelValidator(n_temporal_splits=3)
    hazard_comparison = hazard_validator.compare_models(
        data=survival_data,
        duration_col="duration",
        event_col="event",
    )
    results["hazard"] = hazard_comparison

    # 4. Missingness Analysis
    logger.info("running_missingness_analysis")
    missingness_analyzer = MissingnessAnalyzer()

    # Prepare data for missingness analysis
    missingness_data = survival_data.copy()
    missingness_data["archetype"] = "unknown"  # Would be populated from clustering
    missingness_data["outlier_score"] = np.random.rand(len(missingness_data))
    missingness_data["is_coordinated"] = (missingness_data["shared_funder_count"] >= 3).astype(int)

    missingness_report = missingness_analyzer.analyze(
        data=missingness_data,
        event_col="event",
        archetype_col="archetype",
        anomaly_col="outlier_score",
        coordination_col="is_coordinated",
    )
    results["missingness"] = missingness_report

    # 5. Generate Reports
    logger.info("generating_reports")
    report_generator = ReportGenerator(output_dir)

    report_paths = {}
    report_paths["clustering"] = report_generator.generate_clustering_baseline_report(
        clustering_comparison
    )
    report_paths["ablation"] = report_generator.generate_ablation_report(ablation_results)
    report_paths["hazard"] = report_generator.generate_hazard_report(hazard_comparison)
    report_paths["missingness"] = report_generator.generate_missingness_report(missingness_report)
    report_paths["deployment"] = report_generator.generate_deployment_recommendation(
        clustering_comparison=clustering_comparison,
        hazard_comparison=hazard_comparison,
        ablation_results=ablation_results,
        missingness_report=missingness_report,
    )

    results["report_paths"] = {k: str(v) for k, v in report_paths.items()}

    # 6. Summary
    summary = {
        "best_clustering_pipeline": clustering_comparison.best_pipeline.value,
        "clustering_recommendation": clustering_comparison.recommendation,
        "best_hazard_model": hazard_comparison.best_model.value,
        "hazard_recommendation": hazard_comparison.recommendation,
        "essential_feature_groups": ablation_results.essential_groups,
        "harmful_feature_groups": ablation_results.harmful_groups,
        "informative_missing_features": len(missingness_report.informative_features),
        "reports_generated": list(report_paths.keys()),
    }

    logger.info("validation_gate_complete", **summary)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run SHI Intelligence Upgrade Validation Gate"
    )
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

    try:
        results = run_validation_gate(args.output_dir)
        print("\n" + "=" * 60)
        print("VALIDATION GATE COMPLETE")
        print("=" * 60)
        print(f"\nReports generated in: {args.output_dir}")
        for name, path in results.get("report_paths", {}).items():
            print(f"  - {name}: {path}")
        print("\nSummary:")
        print(f"  Best Clustering: {results.get('summary', {}).get('best_clustering_pipeline', 'N/A')}")
        print(f"  Best Hazard Model: {results.get('summary', {}).get('best_hazard_model', 'N/A')}")
        return 0
    except Exception as e:
        logger.error("validation_gate_failed", error=str(e))
        raise


if __name__ == "__main__":
    sys.exit(main())
