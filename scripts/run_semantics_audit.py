#!/usr/bin/env python
"""
Cluster Semantics Audit.

Investigates whether clustering results represent real behavioural
structure or geometric artifacts.

Usage:
    python scripts/run_semantics_audit.py [--output-dir docs/validation]
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

from src.validation.intelligence.cluster_semantics import ClusterSemanticsAnalyzer
from src.clustering.archetypes import WalletFeatureVector, Archetype
from src.clustering.diagnostics import HDBSCANDiagnostics

logger = structlog.get_logger()


def generate_realistic_test_data(n_samples: int = 500) -> tuple:
    """
    Generate test data with realistic Solana wallet distributions.

    This simulates the kinds of patterns that cause the silhouette/stability
    contradiction and coordination over-detection issues.
    """
    logger.info("generating_realistic_test_data", n_samples=n_samples)

    np.random.seed(42)

    feature_names = [
        "balance", "share", "rank",
        "entry_time_relative", "holding_duration", "position_volatility",
        "delta_balance_7d", "delta_balance_30d",
        "trade_count", "burstiness", "swap_frequency", "lp_interaction_ratio",
        "in_degree", "out_degree", "eigenvector_centrality", "shared_funder_count",
        "total_funding_received", "largest_funder_share", "funding_hhi",
    ]

    n_features = len(feature_names)
    features = np.zeros((n_samples, n_features))

    # Create realistic Solana wallet distributions
    # Key insight: On Solana, shared_funder_count >= 2 is VERY common

    # Distribution features (heavy-tailed)
    features[:, 0] = np.exp(np.random.randn(n_samples) * 2 + 5)  # balance
    features[:, 1] = np.random.beta(0.5, 50, n_samples)  # share (very skewed)
    features[:, 2] = np.arange(1, n_samples + 1)  # rank

    # Temporal features - these tend to cluster naturally
    # Group 1: Early entrants, long holders (~30%)
    # Group 2: Late entrants, short holders (~40%)
    # Group 3: Mixed (~30%)

    group1_size = int(n_samples * 0.3)
    group2_size = int(n_samples * 0.4)
    group3_size = n_samples - group1_size - group2_size

    # Entry time
    features[:group1_size, 3] = np.random.beta(2, 8, group1_size)  # Early
    features[group1_size:group1_size+group2_size, 3] = np.random.beta(8, 2, group2_size)  # Late
    features[group1_size+group2_size:, 3] = np.random.beta(2, 2, group3_size)  # Mixed

    # Holding duration (correlated with entry time)
    features[:group1_size, 4] = np.random.exponential(20, group1_size) + 10  # Long
    features[group1_size:group1_size+group2_size, 4] = np.random.exponential(2, group2_size) + 0.5  # Short
    features[group1_size+group2_size:, 4] = np.random.exponential(8, group3_size) + 2  # Medium

    # Position volatility
    features[:, 5] = np.abs(np.random.randn(n_samples) * 0.3)

    # Flow features
    features[:, 6] = np.random.randn(n_samples) * 0.1  # delta_7d
    features[:, 7] = np.random.randn(n_samples) * 0.15  # delta_30d

    # Trading features
    features[:, 8] = np.random.poisson(5, n_samples)  # trade_count
    features[:, 9] = np.random.uniform(-1, 1, n_samples)  # burstiness
    features[:, 10] = np.random.exponential(0.5, n_samples)  # swap_frequency
    features[:, 11] = np.random.beta(2, 8, n_samples)  # lp_interaction_ratio

    # Graph features - THIS IS THE KEY ISSUE
    # On Solana, shared funders are EXTREMELY common due to:
    # - Common funding sources (exchanges, bridges)
    # - Airdrop farmers using similar patterns
    # - Bot networks

    # Realistic shared_funder_count distribution for Solana
    # ~60% have >= 2 shared funders
    # ~40% have >= 3 shared funders
    # ~20% have >= 5 shared funders
    shared_funder_base = np.random.negative_binomial(2, 0.3, n_samples)
    features[:, 15] = shared_funder_base  # shared_funder_count

    # Other graph features
    features[:, 12] = np.random.poisson(3, n_samples)  # in_degree
    features[:, 13] = np.random.poisson(2, n_samples)  # out_degree
    features[:, 14] = np.random.beta(2, 10, n_samples)  # eigenvector_centrality

    # Weighted graph features
    features[:, 16] = np.exp(np.random.randn(n_samples) + 1)  # total_funding_received
    features[:, 17] = np.random.beta(3, 2, n_samples)  # largest_funder_share
    features[:, 18] = np.random.beta(2, 5, n_samples)  # funding_hhi

    # Create wallet vectors
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
            in_degree=int(features[i, 12]),
            out_degree=int(features[i, 13]),
            eigenvector_centrality=float(features[i, 14]),
            shared_funder_count=int(features[i, 15]),
        )
        wallet_vectors.append(wv)

    # Run HDBSCAN clustering
    from sklearn.preprocessing import RobustScaler
    from hdbscan import HDBSCAN

    # Apply robust scaling (this can artificially inflate silhouette)
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(np.nan_to_num(features, 0))

    clusterer = HDBSCAN(min_cluster_size=5, min_samples=3)
    labels = clusterer.fit_predict(scaled_features)

    # Compute metrics
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

    valid_mask = labels >= 0
    if valid_mask.sum() > 10 and len(set(labels[valid_mask])) > 1:
        silhouette = silhouette_score(scaled_features[valid_mask], labels[valid_mask])
    else:
        silhouette = 0.0

    # Bootstrap stability
    ari_scores = []
    nmi_scores = []
    for _ in range(10):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_labels = clusterer.fit_predict(scaled_features[idx])

        # Compare to original (on overlapping samples)
        common_valid = (labels >= 0) & (bootstrap_labels >= 0)
        if common_valid.sum() > 10:
            ari_scores.append(adjusted_rand_score(labels[common_valid], bootstrap_labels[common_valid]))
            nmi_scores.append(normalized_mutual_info_score(labels[common_valid], bootstrap_labels[common_valid]))

    ari_mean = float(np.mean(ari_scores)) if ari_scores else 0.0
    nmi_mean = float(np.mean(nmi_scores)) if nmi_scores else 0.0

    # Log distribution info
    shared_funder_ge2 = (features[:, 15] >= 2).sum()
    shared_funder_ge3 = (features[:, 15] >= 3).sum()

    logger.info(
        "test_data_generated",
        n_samples=n_samples,
        n_clusters=len(set(labels)) - (1 if -1 in labels else 0),
        noise_pct=(labels == -1).mean() * 100,
        silhouette=silhouette,
        ari_mean=ari_mean,
        nmi_mean=nmi_mean,
        shared_funder_ge2_pct=shared_funder_ge2 / n_samples * 100,
        shared_funder_ge3_pct=shared_funder_ge3 / n_samples * 100,
    )

    return features, feature_names, wallet_vectors, labels, silhouette, ari_mean, nmi_mean


def generate_report(report, output_path: Path) -> None:
    """Generate markdown report from semantics analysis."""
    lines = [
        "# Cluster Semantics Audit Report",
        "",
        f"Generated: {report.computed_at.isoformat()}",
        "",
        "## Executive Summary",
        "",
        f"**Clusters Represent Real Behaviour:** {'YES' if report.clusters_are_semantic else 'NO - INVESTIGATE'}",
        "",
        "### Primary Concerns",
        "",
    ]

    for concern in report.primary_concerns:
        lines.append(f"- {concern}")

    if not report.primary_concerns:
        lines.append("- No major concerns identified")

    lines.extend([
        "",
        "### Recommendations",
        "",
    ])

    for rec in report.recommendations:
        lines.append(f"- {rec}")

    # Silhouette vs Stability section
    ss = report.silhouette_stability
    lines.extend([
        "",
        "---",
        "",
        "## 1. Silhouette vs Stability Analysis",
        "",
        f"**Contradiction Detected:** {'YES' if ss.contradiction_detected else 'No'}",
        "",
        "| Metric | Value | Assessment |",
        "|--------|-------|------------|",
        f"| Silhouette Score | {ss.silhouette_score:.3f} | {'SUSPICIOUS' if ss.silhouette_score > 0.9 else 'Normal'} |",
        f"| ARI (Bootstrap) | {ss.ari_mean:.3f} | {'LOW' if ss.ari_mean < 0.1 else 'Acceptable'} |",
        f"| NMI (Bootstrap) | {ss.nmi_mean:.3f} | {'LOW' if ss.nmi_mean < 0.1 else 'Acceptable'} |",
        f"| Clusters | {ss.n_clusters} | {'Binary' if ss.n_clusters == 2 else 'Multiple'} |",
        f"| Size Imbalance | {ss.size_imbalance_ratio:.1f}x | {'HIGH' if ss.size_imbalance_ratio > 5 else 'Normal'} |",
        "",
    ])

    if ss.likely_causes:
        lines.extend([
            "### Likely Causes",
            "",
        ])
        for cause in ss.likely_causes:
            lines.append(f"- {cause}")
        lines.append("")

    if ss.dominant_features:
        lines.extend([
            "### Dominant Features Driving Separation",
            "",
            "| Feature | Single-Feature Silhouette |",
            "|---------|---------------------------|",
        ])
        for feat in ss.dominant_features[:10]:
            sil = ss.per_feature_silhouette.get(feat, 0)
            lines.append(f"| {feat} | {sil:.3f} |")
        lines.append("")

    # Coordination analysis
    ct = report.coordination_threshold
    lines.extend([
        "---",
        "",
        "## 2. Coordinated Cluster Dominance Analysis",
        "",
        f"**Coordinated Percentage:** {ct.coordinated_percentage:.1f}% ({ct.coordinated_count}/{ct.total_wallets})",
        "",
        f"**Current Threshold:** shared_funder_count >= {ct.shared_funder_threshold}",
        "",
        f"**Wallets Meeting Threshold:** {ct.threshold_percentage:.1f}% ({ct.wallets_meeting_threshold}/{ct.total_wallets})",
        "",
        "### Shared Funder Distribution",
        "",
        "| Count | Wallets | Cumulative % |",
        "|-------|---------|--------------|",
    ])

    sorted_dist = sorted(ct.shared_funder_distribution.items())
    cumulative = 0
    for count, n_wallets in sorted_dist[:10]:
        cumulative += n_wallets
        cum_pct = cumulative / ct.total_wallets * 100
        lines.append(f"| {count} | {n_wallets} | {cum_pct:.1f}% |")

    lines.extend([
        "",
        f"**Median Shared Funders:** {ct.median_shared_funders:.1f}",
        f"**Mean Shared Funders:** {ct.mean_shared_funders:.1f}",
        "",
    ])

    if ct.wallets_overridden_to_coordinated > 0:
        lines.extend([
            "### Override Analysis",
            "",
            f"**Wallets Overridden to Coordinated:** {ct.wallets_overridden_to_coordinated}",
            "",
            "| Original Archetype | Count |",
            "|-------------------|-------|",
        ])
        for arch, count in ct.override_sources.items():
            lines.append(f"| {arch} | {count} |")
        lines.append("")

    lines.extend([
        "### Threshold Recommendation",
        "",
        f"**Recommended Threshold:** {ct.recommended_threshold}",
        "",
        f"**Rationale:** {ct.threshold_rationale}",
        "",
    ])

    # Feature contribution
    fc = report.feature_contribution
    lines.extend([
        "---",
        "",
        "## 3. Feature Contribution Analysis",
        "",
        f"**Primary Driver:** {fc.primary_driver.upper()}",
        "",
        "| Feature Group | Silhouette | Contribution % |",
        "|---------------|------------|----------------|",
        f"| Temporal | {fc.temporal_only_silhouette:.3f} | {fc.temporal_contribution_pct:.1f}% |",
        f"| Graph | {fc.graph_only_silhouette:.3f} | {fc.graph_contribution_pct:.1f}% |",
        f"| Trading | {fc.trading_only_silhouette:.3f} | {100 - fc.temporal_contribution_pct - fc.graph_contribution_pct:.1f}% |",
        "",
        "### Strategic Assessment",
        "",
        f"**Is SHI a Timing Engine?** {'YES' if fc.is_timing_engine else 'No'}",
        "",
        f"**Does Graph Add Value?** {'YES' if fc.graph_adds_value else 'NO - needs strengthening'}",
        "",
    ])

    if fc.is_timing_engine:
        lines.extend([
            "**Implication:** SHI currently derives most predictive power from temporal wallet behaviour, ",
            "not from graph topology. The 'graph intelligence' narrative exceeds current graph contribution.",
            "",
        ])

    # Cluster profiles
    lines.extend([
        "---",
        "",
        "## 4. Cluster Profiles",
        "",
    ])

    for profile in report.cluster_profiles:
        lines.extend([
            f"### Cluster {profile.cluster_id}",
            "",
            f"**Size:** {profile.size} wallets ({profile.percentage:.1f}%)",
            "",
            f"**Pattern:** {profile.characteristic_pattern}",
            "",
            f"**Primary Archetype:** {profile.primary_archetype} ({profile.archetype_purity:.1f}% purity)",
            "",
            "**Archetype Distribution:**",
            "",
        ])
        for arch, count in profile.archetype_distribution.items():
            pct = count / profile.size * 100
            lines.append(f"- {arch}: {count} ({pct:.1f}%)")
        lines.extend([
            "",
            f"**Temporal:** Entry={profile.mean_entry_time:.2f}, Hold={profile.mean_hold_duration:.1f}",
            "",
        ])

    # Footer
    lines.extend([
        "---",
        "",
        "## Key Insight",
        "",
        "High silhouette with low bootstrap stability indicates **geometric separation without semantic stability**.",
        "",
        "The clusters may look clean in feature space but do not represent robust, reproducible behavioural patterns.",
        "",
        "Before deploying as default, validate against known wallet behaviours:",
        "- Known rug wallets",
        "- Known sniper wallets",
        "- Known LP operators",
        "- Known coordinated launches",
        "",
    ])

    output_path.write_text("\n".join(lines))
    logger.info("report_generated", path=str(output_path))


def main():
    """Run cluster semantics audit."""
    parser = argparse.ArgumentParser(description="Run Cluster Semantics Audit")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/validation"),
        help="Output directory for report",
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

    logger.info("starting_cluster_semantics_audit")

    # Generate test data
    features, feature_names, wallet_vectors, labels, silhouette, ari_mean, nmi_mean = (
        generate_realistic_test_data(n_samples=500)
    )

    # Run semantics analysis using configurable threshold
    from src.core.config import settings
    analyzer = ClusterSemanticsAnalyzer(coordination_threshold=settings.coordination_shared_funder_threshold)
    report = analyzer.analyze(
        features=features,
        feature_names=feature_names,
        labels=labels,
        wallet_vectors=wallet_vectors,
        silhouette_score=silhouette,
        ari_mean=ari_mean,
        nmi_mean=nmi_mean,
    )

    # Generate report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "CLUSTER_SEMANTICS_AUDIT.md"
    generate_report(report, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("CLUSTER SEMANTICS AUDIT COMPLETE")
    print("=" * 60)
    print(f"\nReport: {output_path}")
    print(f"\nClusters Semantic: {'YES' if report.clusters_are_semantic else 'NO - INVESTIGATE'}")
    print(f"\nPrimary Concerns ({len(report.primary_concerns)}):")
    for concern in report.primary_concerns[:5]:
        print(f"  - {concern[:80]}...")
    print(f"\nRecommendations ({len(report.recommendations)}):")
    for rec in report.recommendations[:3]:
        print(f"  - {rec[:80]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
