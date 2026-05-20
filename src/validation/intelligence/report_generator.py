"""
Report Generator for Validation Results.

Produces markdown reports from validation study results.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from .pipeline_comparison import FullComparisonReport, ClusteringPipeline
from .hazard_comparison import HazardComparisonResult, HazardModel
from .ablation_runner import AblationStudyResults
from .missingness_analysis import MissingnessReport

logger = structlog.get_logger()


class ReportGenerator:
    """Generates markdown validation reports."""

    def __init__(self, output_dir: Path):
        """
        Initialize generator.

        Args:
            output_dir: Directory for report output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_clustering_baseline_report(
        self,
        comparison: FullComparisonReport,
    ) -> Path:
        """Generate CLUSTERING_BASELINE_COMPARISON.md."""
        lines = [
            "# Clustering Baseline Comparison Report",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Executive Summary",
            "",
            f"**Best Pipeline:** {comparison.best_pipeline.value}",
            "",
            f"**Recommendation:** {comparison.recommendation}",
            "",
            "---",
            "",
            "## Pipeline Comparison",
            "",
            "| Pipeline | Clusters | Noise % | Silhouette | UNKNOWN % | Multi-Archetype % | Mean Confidence |",
            "|----------|----------|---------|------------|-----------|-------------------|-----------------|",
        ]

        for pipeline, result in comparison.results.items():
            sil_str = f"{result.silhouette_score:.3f}" if result.silhouette_score else "N/A"
            lines.append(
                f"| {pipeline.value} | {result.n_clusters} | "
                f"{result.noise_percentage:.1f}% | "
                f"{sil_str} | "
                f"{result.archetype_distribution.unknown_percentage:.1f}% | "
                f"{result.archetype_distribution.multi_archetype_percentage:.1f}% | "
                f"{result.archetype_distribution.mean_confidence:.3f} |"
            )

        lines.extend([
            "",
            "## Stability Analysis (Bootstrap)",
            "",
            "| Pipeline | ARI Mean | ARI Std | NMI Mean | Persistence Rate |",
            "|----------|----------|---------|----------|------------------|",
        ])

        for pipeline, result in comparison.results.items():
            if result.stability:
                lines.append(
                    f"| {pipeline.value} | "
                    f"{result.stability.adjusted_rand_index_mean:.3f} | "
                    f"{result.stability.adjusted_rand_index_std:.3f} | "
                    f"{result.stability.normalized_mutual_info_mean:.3f} | "
                    f"{result.stability.cluster_persistence_rate:.3f} |"
                )

        lines.extend([
            "",
            "## Archetype Distribution by Pipeline",
            "",
        ])

        for pipeline, result in comparison.results.items():
            lines.append(f"### {pipeline.value}")
            lines.append("")
            lines.append("| Archetype | Count | Percentage |")
            lines.append("|-----------|-------|------------|")
            for arch, count in result.archetype_distribution.counts.items():
                pct = result.archetype_distribution.percentages[arch]
                lines.append(f"| {arch.value} | {count} | {pct:.1f}% |")
            lines.append("")

        lines.extend([
            "## Interpretability Notes",
            "",
        ])

        for pipeline, result in comparison.results.items():
            lines.append(f"### {pipeline.value}")
            for note in result.interpretability_notes:
                lines.append(f"- {note}")
            lines.append("")

        lines.extend([
            "## Decision Rationale",
            "",
        ])
        for reason in comparison.decision_rationale:
            lines.append(f"- {reason}")

        lines.extend([
            "",
            "---",
            "",
            "## Acceptance Criteria Assessment",
            "",
            "The new default must improve interpretability, stability, or downstream predictive value.",
            "",
            f"**Assessment:** {comparison.recommendation}",
        ])

        report_path = self.output_dir / "CLUSTERING_BASELINE_COMPARISON.md"
        report_path.write_text("\n".join(lines))

        logger.info("clustering_report_generated", path=str(report_path))
        return report_path

    def generate_ablation_report(
        self,
        results: AblationStudyResults,
    ) -> Path:
        """Generate FEATURE_ABLATION_RESULTS.md."""
        lines = [
            "# Feature Ablation Results",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Executive Summary",
            "",
            f"**Essential Groups:** {', '.join(results.essential_groups) or 'None identified'}",
            "",
            f"**Redundant Groups:** {', '.join(results.redundant_groups) or 'None identified'}",
            "",
            f"**Harmful Groups:** {', '.join(results.harmful_groups) or 'None identified'}",
            "",
            f"**Recommendation:** {results.recommendation}",
            "",
            "---",
            "",
            "## Baseline Metrics",
            "",
            f"- **Silhouette Score:** {results.baseline_silhouette:.3f}" if results.baseline_silhouette else "- **Silhouette Score:** N/A",
            f"- **Noise Rate:** {results.baseline_noise_rate:.1f}%",
            f"- **Concordance Index:** {results.baseline_concordance:.3f}" if results.baseline_concordance else "- **Concordance Index:** N/A",
            "",
            "## Feature Group Impact Analysis",
            "",
            "| Group | Stability Δ | Noise Δ | Sell Pred Δ | Essential | Redundant | Harmful |",
            "|-------|-------------|---------|-------------|-----------|-----------|---------|",
        ]

        for group, impact in results.feature_group_impacts.items():
            sell_delta = f"{impact.sell_prediction_delta:.3f}" if impact.sell_prediction_delta else "N/A"
            lines.append(
                f"| {group} | {impact.cluster_stability_delta:.3f} | "
                f"{impact.noise_rate_delta:+.1f}% | {sell_delta} | "
                f"{'✓' if impact.is_essential else ''} | "
                f"{'✓' if impact.is_redundant else ''} | "
                f"{'✓' if impact.is_harmful else ''} |"
            )

        lines.extend([
            "",
            "## Detailed Group Analysis",
            "",
        ])

        for group, impact in results.feature_group_impacts.items():
            lines.append(f"### {group.title()}")
            lines.append("")
            lines.append(f"**Features:** {', '.join(impact.features)}")
            lines.append("")
            lines.append("**Impact Metrics:**")
            lines.append(f"- Cluster Stability Change: {impact.cluster_stability_delta:+.3f}")
            lines.append(f"- Noise Rate Change: {impact.noise_rate_delta:+.1f}%")
            lines.append(f"- UNKNOWN Rate Change: {impact.unknown_rate_delta:+.1f}%")
            if impact.sell_prediction_delta is not None:
                lines.append(f"- Sell Prediction (C-index) Change: {impact.sell_prediction_delta:+.3f}")
            lines.append("")
            lines.append("**Assessment:**")
            for note in impact.assessment_notes:
                lines.append(f"- {note}")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Recommendations",
            "",
            results.recommendation,
        ])

        report_path = self.output_dir / "FEATURE_ABLATION_RESULTS.md"
        report_path.write_text("\n".join(lines))

        logger.info("ablation_report_generated", path=str(report_path))
        return report_path

    def generate_hazard_report(
        self,
        comparison: HazardComparisonResult,
    ) -> Path:
        """Generate HAZARD_MODEL_COMPARISON.md."""
        lines = [
            "# Hazard Model Comparison Report",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Executive Summary",
            "",
            f"**Best Model:** {comparison.best_model.value}",
            "",
            f"**Recommendation:** {comparison.recommendation}",
            "",
            "---",
            "",
            "## Model Configurations",
            "",
            "| Model | Description | Features |",
            "|-------|-------------|----------|",
            "| Model A | Baseline | Original 10 features |",
            "| Model B | Expanded | + swap, LP, delta_30d, eigenvector |",
            "| Model C | Price/Liquidity | + unrealized_pnl, liquidity, sell_pressure |",
            "| Model D | Missingness | + missingness indicators |",
            "",
            "## Model Performance Comparison",
            "",
            "| Model | C-Index | C-Index Std | Brier | Calibration Slope | PH Assumption |",
            "|-------|---------|-------------|-------|-------------------|---------------|",
        ]

        for model, result in comparison.results.items():
            ph_status = "PASS" if result.ph_assumption.passes_assumption else "FAIL"
            lines.append(
                f"| {model.value} | {result.concordance_index:.3f} | "
                f"{result.concordance_std:.3f} | "
                f"{result.calibration.brier_score:.3f} | "
                f"{result.calibration.calibration_slope:.3f} | {ph_status} |"
            )

        lines.extend([
            "",
            "## Walk-Forward Validation",
            "",
            "| Model | WF Concordance | WF Std | Folds |",
            "|-------|----------------|--------|-------|",
        ])

        for model, result in comparison.results.items():
            lines.append(
                f"| {model.value} | {result.walk_forward_concordance:.3f} | "
                f"{result.walk_forward_std:.3f} | {result.n_validation_folds} |"
            )

        lines.extend([
            "",
            "## Calibration Analysis",
            "",
            "| Model | Brier Score | Slope | Intercept | Mean Predicted | Mean Observed |",
            "|-------|-------------|-------|-----------|----------------|---------------|",
        ])

        for model, result in comparison.results.items():
            lines.append(
                f"| {model.value} | {result.calibration.brier_score:.3f} | "
                f"{result.calibration.calibration_slope:.3f} | "
                f"{result.calibration.calibration_intercept:.3f} | "
                f"{result.calibration.mean_predicted:.3f} | "
                f"{result.calibration.mean_observed:.3f} |"
            )

        lines.extend([
            "",
            "## PH Assumption Test Results",
            "",
        ])

        for model, result in comparison.results.items():
            status = "✓ PASSES" if result.ph_assumption.passes_assumption else "✗ FAILS"
            lines.append(f"### {model.value}: {status}")
            lines.append(f"- Global p-value: {result.ph_assumption.global_pvalue:.3f}")
            if result.ph_assumption.violating_features:
                lines.append(f"- Violating features: {', '.join(result.ph_assumption.violating_features)}")
            lines.append("")

        lines.extend([
            "## Coefficient Stability (CV)",
            "",
            "Lower values indicate more stable coefficients across validation folds.",
            "",
        ])

        for model, result in comparison.results.items():
            lines.append(f"### {model.value}")
            stable = [f for f, cv in result.coefficient_stability.items() if cv < 0.5]
            unstable = [f for f, cv in result.coefficient_stability.items() if cv >= 0.5]
            if stable:
                lines.append(f"- Stable (CV < 0.5): {', '.join(stable[:5])}")
            if unstable:
                lines.append(f"- Unstable (CV ≥ 0.5): {', '.join(unstable[:5])}")
            lines.append("")

        lines.extend([
            "## Decision Rationale",
            "",
        ])
        for reason in comparison.decision_rationale:
            lines.append(f"- {reason}")

        lines.extend([
            "",
            "---",
            "",
            "## Important Notes",
            "",
            "- Models with improved C-index but degraded calibration are REJECTED",
            "- PH assumption violations may require stratified models",
            "- Walk-forward validation reflects production performance better than temporal CV",
        ])

        report_path = self.output_dir / "HAZARD_MODEL_COMPARISON.md"
        report_path.write_text("\n".join(lines))

        logger.info("hazard_report_generated", path=str(report_path))
        return report_path

    def generate_missingness_report(
        self,
        report: MissingnessReport,
    ) -> Path:
        """Generate MISSINGNESS_IMPACT_REPORT.md."""
        lines = [
            "# Missingness Impact Report",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Executive Summary",
            "",
            f"**Total Wallets:** {report.total_wallets:,}",
            "",
            f"**Wallets with Any Missing:** {report.wallets_with_any_missing:,} ({report.wallets_with_any_missing_pct:.1f}%)",
            "",
            f"**Informative Features:** {len(report.informative_features)}",
            "",
            f"**Recommendation:** {report.recommendation}",
            "",
            "---",
            "",
            "## Missingness by Category",
            "",
            "| Category | Features | Missing % | Any Missing % |",
            "|----------|----------|-----------|---------------|",
        ]

        for cat, stats in report.category_stats.items():
            lines.append(
                f"| {cat} | {len(stats.features)} | "
                f"{stats.missing_percentage:.1f}% | "
                f"{stats.any_missing_percentage:.1f}% |"
            )

        lines.extend([
            "",
            "## Informative Missingness",
            "",
            "Features where missingness is statistically associated with outcomes:",
            "",
            "| Feature | Missing % | Predicts Event | Predicts UNKNOWN | Predicts Anomaly | Predicts Coordination |",
            "|---------|-----------|----------------|------------------|------------------|----------------------|",
        ])

        for feat_name in report.informative_features:
            if feat_name in report.feature_patterns:
                p = report.feature_patterns[feat_name]
                lines.append(
                    f"| {feat_name} | {p.missing_percentage:.1f}% | "
                    f"{'✓' if p.missing_predicts_event else ''} | "
                    f"{'✓' if p.missing_predicts_unknown else ''} | "
                    f"{'✓' if p.missing_predicts_anomaly else ''} | "
                    f"{'✓' if p.missing_predicts_coordination else ''} |"
                )

        lines.extend([
            "",
            "## Event Rate Analysis",
            "",
            "Comparison of sell event rates between missing and present values:",
            "",
            "| Feature | Event Rate (Missing) | Event Rate (Present) | Rate Ratio | P-value |",
            "|---------|---------------------|----------------------|------------|---------|",
        ])

        for feat_name, p in report.feature_patterns.items():
            if p.event_rate_when_missing > 0 or p.event_rate_when_present > 0:
                pvalue = f"{p.chi_square_pvalue:.4f}" if p.chi_square_pvalue else "N/A"
                lines.append(
                    f"| {feat_name} | {p.event_rate_when_missing:.3f} | "
                    f"{p.event_rate_when_present:.3f} | "
                    f"{p.rate_ratio:.2f} | {pvalue} |"
                )

        lines.extend([
            "",
            "## Detailed Feature Patterns",
            "",
        ])

        # Only show features with significant missingness
        significant_missing = [
            (name, p) for name, p in report.feature_patterns.items()
            if p.missing_percentage > 5
        ]
        significant_missing.sort(key=lambda x: x[1].missing_percentage, reverse=True)

        for feat_name, p in significant_missing[:15]:
            lines.append(f"### {feat_name}")
            lines.append("")
            lines.append(f"- Missing: {p.missing_count:,} ({p.missing_percentage:.1f}%)")
            lines.append(f"- Predicts sell event: {'Yes' if p.missing_predicts_event else 'No'}")
            lines.append(f"- Predicts UNKNOWN: {'Yes' if p.missing_predicts_unknown else 'No'}")
            lines.append(f"- Predicts anomaly: {'Yes' if p.missing_predicts_anomaly else 'No'}")
            lines.append(f"- Event rate when missing: {p.event_rate_when_missing:.3f}")
            lines.append(f"- Event rate when present: {p.event_rate_when_present:.3f}")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Key Insight",
            "",
            "Missingness may be informative rather than merely inconvenient. Features where ",
            "missingness significantly predicts outcomes should have missingness indicators ",
            "retained as features in the model.",
            "",
        ])

        report_path = self.output_dir / "MISSINGNESS_IMPACT_REPORT.md"
        report_path.write_text("\n".join(lines))

        logger.info("missingness_report_generated", path=str(report_path))
        return report_path

    def generate_deployment_recommendation(
        self,
        clustering_comparison: Optional[FullComparisonReport],
        hazard_comparison: Optional[HazardComparisonResult],
        ablation_results: Optional[AblationStudyResults],
        missingness_report: Optional[MissingnessReport],
    ) -> Path:
        """Generate DEPLOYMENT_RECOMMENDATION.md."""
        lines = [
            "# Deployment Recommendation",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Overall Recommendation",
            "",
        ]

        # Collect recommendations
        recommendations = []

        if clustering_comparison:
            if "DEPLOY" in clustering_comparison.recommendation:
                recommendations.append(("Clustering", "DEPLOY", clustering_comparison.recommendation))
            else:
                recommendations.append(("Clustering", "KEEP BASELINE", clustering_comparison.recommendation))

        if hazard_comparison:
            if "DEPLOY" in hazard_comparison.recommendation:
                recommendations.append(("Hazard Model", "DEPLOY", hazard_comparison.recommendation))
            else:
                recommendations.append(("Hazard Model", "KEEP BASELINE", hazard_comparison.recommendation))

        # Summary table
        lines.extend([
            "| Component | Decision | Details |",
            "|-----------|----------|---------|",
        ])

        for component, decision, details in recommendations:
            lines.append(f"| {component} | {decision} | {details[:50]}... |")

        lines.extend([
            "",
            "---",
            "",
            "## Feature Flag Configuration",
            "",
            "Based on validation results, recommended feature flag settings:",
            "",
            "```python",
        ])

        # Determine settings based on results
        use_robust = True
        use_node2vec = False
        use_expanded_hazard = False
        use_missingness = True
        use_weighted_graph = True

        if clustering_comparison:
            if clustering_comparison.best_pipeline == ClusteringPipeline.NEW_COMBINED:
                use_node2vec = True
            elif clustering_comparison.best_pipeline == ClusteringPipeline.OLD_RULE_FIRST:
                use_robust = False

        if hazard_comparison:
            if hazard_comparison.best_model in [HazardModel.MODEL_C_PRICE_LIQUIDITY, HazardModel.MODEL_D_MISSINGNESS]:
                use_expanded_hazard = True

        lines.extend([
            f"USE_ROBUST_CLUSTERING = {str(use_robust).lower()}",
            f"USE_NODE2VEC_CLUSTERING = {str(use_node2vec).lower()}",
            f"USE_EXPANDED_HAZARD_FEATURES = {str(use_expanded_hazard).lower()}",
            f"USE_MISSINGNESS_INDICATORS = {str(use_missingness).lower()}",
            f"USE_WEIGHTED_GRAPH_FEATURES = {str(use_weighted_graph).lower()}",
            "```",
            "",
            "---",
            "",
            "## Expected Intelligence Gain",
            "",
        ])

        if clustering_comparison:
            baseline = clustering_comparison.results.get(ClusteringPipeline.OLD_RULE_FIRST)
            best = clustering_comparison.results.get(clustering_comparison.best_pipeline)
            if baseline and best and best.silhouette_score and baseline.silhouette_score:
                silhouette_gain = best.silhouette_score - baseline.silhouette_score
                noise_reduction = baseline.noise_percentage - best.noise_percentage
                lines.append(f"- Silhouette improvement: {silhouette_gain:+.3f}")
                lines.append(f"- Noise reduction: {noise_reduction:+.1f}%")

        if hazard_comparison:
            baseline = hazard_comparison.results.get(HazardModel.MODEL_A_BASELINE)
            best = hazard_comparison.results.get(hazard_comparison.best_model)
            if baseline and best:
                c_gain = best.concordance_index - baseline.concordance_index
                lines.append(f"- Concordance improvement: {c_gain:+.3f}")

        lines.extend([
            "",
            "## Runtime Impact",
            "",
            "| Component | Estimated Overhead |",
            "|-----------|-------------------|",
            "| Robust Transformations | +5-10% |",
            "| Node2Vec Embeddings | +50-100% (if enabled) |",
            "| Missingness Indicators | +2-5% |",
            "| Weighted Graph Features | +10-15% |",
            "",
            "## Risks",
            "",
            "1. **Model Drift**: New pipeline may behave differently on edge cases",
            "2. **Interpretation Changes**: Archetype meanings may shift slightly",
            "3. **Performance**: Node2Vec significantly increases computation time",
            "4. **Data Requirements**: Some features require additional data sources",
            "",
            "## Rollback Plan",
            "",
            "1. Feature flags allow instant rollback to baseline",
            "2. Old pipeline preserved behind `USE_ROBUST_CLUSTERING=false`",
            "3. Monitor key metrics for 7 days post-deployment",
            "4. Alert on >10% change in noise rate or UNKNOWN percentage",
            "",
            "---",
            "",
            "## Verification Checklist",
            "",
            "- [ ] All validation reports reviewed",
            "- [ ] Feature flags configured correctly",
            "- [ ] Monitoring dashboards updated",
            "- [ ] Rollback procedure tested",
            "- [ ] Team notified of changes",
            "",
        ])

        report_path = self.output_dir / "DEPLOYMENT_RECOMMENDATION.md"
        report_path.write_text("\n".join(lines))

        logger.info("deployment_recommendation_generated", path=str(report_path))
        return report_path
