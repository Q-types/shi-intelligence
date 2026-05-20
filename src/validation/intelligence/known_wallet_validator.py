"""
Known Wallet Validation Framework.

Validates archetype assignments against ground-truth labeled wallets.
This is essential for determining whether clusters represent real
behavioural patterns rather than geometric artifacts.

Usage:
    1. Create labeled wallet dataset (CSV with wallet, true_archetype)
    2. Run SHI clustering on those wallets
    3. Compare predicted vs actual archetypes
    4. Compute precision/recall/F1 per archetype
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
import pandas as pd
import structlog

from ...clustering.archetypes import Archetype, WalletFeatureVector, assign_archetype_multi_score

logger = structlog.get_logger()


@dataclass
class ArchetypeMetrics:
    """Precision/Recall/F1 for a single archetype."""

    archetype: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    support: int  # Total true instances

    def to_dict(self) -> dict:
        return {
            "archetype": self.archetype,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "support": self.support,
        }


@dataclass
class ConfusionEntry:
    """Single entry in confusion matrix."""

    true_archetype: str
    predicted_archetype: str
    count: int
    example_wallets: list[str]


@dataclass
class MisclassificationCase:
    """Detailed analysis of a misclassification."""

    wallet: str
    true_archetype: str
    predicted_archetype: str
    predicted_confidence: float
    secondary_archetypes: list[str]
    feature_summary: dict[str, float]
    likely_cause: str


@dataclass
class KnownWalletValidationReport:
    """Complete validation report against known wallets."""

    # Overall metrics
    total_wallets: int
    accuracy: float
    macro_f1: float
    weighted_f1: float

    # Per-archetype metrics
    archetype_metrics: dict[str, ArchetypeMetrics]

    # Confusion matrix
    confusion_matrix: list[ConfusionEntry]

    # Detailed misclassifications
    misclassifications: list[MisclassificationCase]
    misclassification_rate: float

    # Coverage analysis
    archetypes_with_ground_truth: list[str]
    archetypes_missing_ground_truth: list[str]

    # Recommendations
    recommendations: list[str]

    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "total_wallets": self.total_wallets,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "archetype_metrics": {k: v.to_dict() for k, v in self.archetype_metrics.items()},
            "confusion_matrix": [
                {"true": e.true_archetype, "predicted": e.predicted_archetype, "count": e.count}
                for e in self.confusion_matrix
            ],
            "misclassification_rate": self.misclassification_rate,
            "archetypes_with_ground_truth": self.archetypes_with_ground_truth,
            "archetypes_missing_ground_truth": self.archetypes_missing_ground_truth,
            "recommendations": self.recommendations,
            "computed_at": self.computed_at.isoformat(),
        }


# Example known wallet patterns for testing
# In production, these would come from labeled datasets
EXAMPLE_KNOWN_WALLETS = {
    # Sniper patterns: early entry, quick exit, high trade count
    "sniper": {
        "entry_time_relative": (0.0, 0.1),  # First 10%
        "holding_duration": (0.1, 5.0),  # Very short hold
        "trade_count": (5, 50),
    },
    # Long-term accumulator: early entry, long hold, low activity
    "long_term_accumulator": {
        "entry_time_relative": (0.0, 0.3),
        "holding_duration": (30, 365),
        "trade_count": (1, 10),
        "delta_balance_30d": (0, 1.0),
    },
    # Coordinated cluster: shared funders
    "coordinated_cluster": {
        "shared_funder_count": (5, 100),
    },
    # Liquidity actor: high LP interaction
    "liquidity_actor": {
        "lp_interaction_ratio": (0.3, 1.0),
    },
    # Dormant whale: large balance, no activity
    "dormant_whale": {
        "share": (0.01, 1.0),  # Top 1%+ holders
        "trade_count": (0, 5),
        "holding_duration": (14, 365),
    },
}


class KnownWalletValidator:
    """
    Validates archetype assignments against ground-truth labels.

    Use this to determine whether SHI's clustering produces
    meaningful behavioral classifications.
    """

    def __init__(self):
        """Initialize validator."""
        pass

    def validate(
        self,
        wallet_vectors: list[WalletFeatureVector],
        ground_truth: dict[str, str],  # wallet -> true_archetype
        predictions: Optional[dict[str, str]] = None,  # wallet -> predicted_archetype
    ) -> KnownWalletValidationReport:
        """
        Validate predictions against ground truth.

        Args:
            wallet_vectors: Feature vectors for wallets
            ground_truth: Dict mapping wallet address to true archetype
            predictions: Optional predictions (if None, will compute)

        Returns:
            KnownWalletValidationReport with full analysis
        """
        logger.info(
            "starting_known_wallet_validation",
            n_wallets=len(wallet_vectors),
            n_labeled=len(ground_truth),
        )

        # Build wallet lookup
        wallet_lookup = {wv.wallet: wv for wv in wallet_vectors}

        # Get predictions if not provided
        if predictions is None:
            predictions = {}
            for wallet, wv in wallet_lookup.items():
                if wallet in ground_truth:
                    assignment = assign_archetype_multi_score(
                        wv, cluster_status="core", cluster_confidence_adj=0.5
                    )
                    predictions[wallet] = assignment.primary_archetype.value

        # Filter to wallets with both ground truth and predictions
        common_wallets = set(ground_truth.keys()) & set(predictions.keys())

        if not common_wallets:
            logger.warning("no_common_wallets_for_validation")
            return self._empty_report()

        # Compute metrics
        y_true = [ground_truth[w] for w in common_wallets]
        y_pred = [predictions[w] for w in common_wallets]

        # Overall accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(common_wallets)

        # Per-archetype metrics
        all_archetypes = set(y_true) | set(y_pred)
        archetype_metrics = {}

        for arch in all_archetypes:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == arch and p == arch)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != arch and p == arch)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == arch and p != arch)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = sum(1 for t in y_true if t == arch)

            archetype_metrics[arch] = ArchetypeMetrics(
                archetype=arch,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=precision,
                recall=recall,
                f1_score=f1,
                support=support,
            )

        # Macro/weighted F1
        f1_scores = [m.f1_score for m in archetype_metrics.values()]
        supports = [m.support for m in archetype_metrics.values()]
        total_support = sum(supports)

        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        weighted_f1 = (
            sum(f * s for f, s in zip(f1_scores, supports)) / total_support
            if total_support > 0 else 0.0
        )

        # Confusion matrix
        confusion_counter: Counter = Counter()
        confusion_examples: dict[tuple, list] = {}

        for wallet in common_wallets:
            t, p = ground_truth[wallet], predictions[wallet]
            confusion_counter[(t, p)] += 1
            key = (t, p)
            if key not in confusion_examples:
                confusion_examples[key] = []
            if len(confusion_examples[key]) < 3:
                confusion_examples[key].append(wallet)

        confusion_matrix = [
            ConfusionEntry(
                true_archetype=t,
                predicted_archetype=p,
                count=count,
                example_wallets=confusion_examples.get((t, p), []),
            )
            for (t, p), count in confusion_counter.most_common()
        ]

        # Detailed misclassifications
        misclassifications = []
        for wallet in common_wallets:
            t, p = ground_truth[wallet], predictions[wallet]
            if t != p and wallet in wallet_lookup:
                wv = wallet_lookup[wallet]
                assignment = assign_archetype_multi_score(
                    wv, cluster_status="core", cluster_confidence_adj=0.5
                )

                likely_cause = self._diagnose_misclassification(wv, t, p)

                misclassifications.append(MisclassificationCase(
                    wallet=wallet,
                    true_archetype=t,
                    predicted_archetype=p,
                    predicted_confidence=assignment.primary_confidence,
                    secondary_archetypes=[a.value for a in assignment.secondary_archetypes],
                    feature_summary={
                        "entry_time_relative": wv.entry_time_relative,
                        "holding_duration": wv.holding_duration,
                        "trade_count": wv.trade_count,
                        "shared_funder_count": wv.shared_funder_count,
                        "lp_interaction_ratio": wv.lp_interaction_ratio,
                    },
                    likely_cause=likely_cause,
                ))

        misclassification_rate = len(misclassifications) / len(common_wallets)

        # Coverage analysis
        all_shi_archetypes = [a.value for a in Archetype]
        archetypes_with_gt = list(set(y_true))
        archetypes_missing_gt = [a for a in all_shi_archetypes if a not in archetypes_with_gt]

        # Recommendations
        recommendations = self._generate_recommendations(
            archetype_metrics, misclassifications, archetypes_missing_gt
        )

        logger.info(
            "known_wallet_validation_complete",
            accuracy=accuracy,
            macro_f1=macro_f1,
            misclassification_rate=misclassification_rate,
        )

        return KnownWalletValidationReport(
            total_wallets=len(common_wallets),
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            archetype_metrics=archetype_metrics,
            confusion_matrix=confusion_matrix,
            misclassifications=misclassifications[:20],  # Limit to top 20
            misclassification_rate=misclassification_rate,
            archetypes_with_ground_truth=archetypes_with_gt,
            archetypes_missing_ground_truth=archetypes_missing_gt,
            recommendations=recommendations,
        )

    def generate_synthetic_ground_truth(
        self,
        wallet_vectors: list[WalletFeatureVector],
        noise_rate: float = 0.1,
    ) -> dict[str, str]:
        """
        Generate synthetic ground truth based on feature patterns.

        This is for testing the validation framework.
        Real validation requires human-labeled data.

        Args:
            wallet_vectors: Feature vectors
            noise_rate: Rate of random label noise

        Returns:
            Dict mapping wallet to "true" archetype
        """
        ground_truth = {}
        np.random.seed(42)

        for wv in wallet_vectors:
            # Assign based on feature patterns
            archetype = self._infer_archetype_from_features(wv)

            # Add noise
            if np.random.rand() < noise_rate:
                all_archetypes = [a.value for a in Archetype if a != Archetype.UNKNOWN]
                archetype = np.random.choice(all_archetypes)

            ground_truth[wv.wallet] = archetype

        return ground_truth

    def _infer_archetype_from_features(self, wv: WalletFeatureVector) -> str:
        """Infer archetype from feature patterns (for synthetic data)."""
        # Check patterns in order of specificity

        # Coordinated: high shared funders
        if wv.shared_funder_count >= 5:
            return Archetype.COORDINATED_CLUSTER.value

        # Liquidity actor: high LP ratio
        if wv.lp_interaction_ratio >= 0.3:
            return Archetype.LIQUIDITY_ACTOR.value

        # Sniper: early entry, quick exit
        if wv.entry_time_relative < 0.1 and wv.holding_duration < 5:
            return Archetype.SNIPER.value

        # Dormant whale: large share, low activity
        if wv.share > 0.01 and wv.trade_count <= 5:
            return Archetype.DORMANT_WHALE.value

        # Long-term accumulator: long hold, positive delta
        if wv.holding_duration >= 30 and wv.delta_balance_30d >= 0:
            return Archetype.LONG_TERM_ACCUMULATOR.value

        return Archetype.UNKNOWN.value

    def _diagnose_misclassification(
        self,
        wv: WalletFeatureVector,
        true_arch: str,
        pred_arch: str,
    ) -> str:
        """Diagnose why a misclassification occurred."""
        causes = []

        # Coordination override
        if pred_arch == Archetype.COORDINATED_CLUSTER.value and true_arch != pred_arch:
            if wv.shared_funder_count >= 5:
                causes.append(f"Coordination override: shared_funder_count={wv.shared_funder_count}")

        # Threshold edge cases
        if true_arch == Archetype.SNIPER.value:
            if wv.entry_time_relative > 0.1:
                causes.append(f"Late entry: entry_time={wv.entry_time_relative:.2f}")
            if wv.holding_duration > 5:
                causes.append(f"Long hold: duration={wv.holding_duration:.1f}")

        if true_arch == Archetype.LONG_TERM_ACCUMULATOR.value:
            if wv.holding_duration < 30:
                causes.append(f"Short hold: duration={wv.holding_duration:.1f}")
            if wv.trade_count > 10:
                causes.append(f"High activity: trades={wv.trade_count}")

        if not causes:
            return "Feature pattern ambiguous"

        return "; ".join(causes)

    def _generate_recommendations(
        self,
        metrics: dict[str, ArchetypeMetrics],
        misclassifications: list[MisclassificationCase],
        missing_gt: list[str],
    ) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Low performing archetypes
        for arch, m in metrics.items():
            if m.f1_score < 0.5 and m.support >= 5:
                recommendations.append(
                    f"IMPROVE {arch}: F1={m.f1_score:.2f} (P={m.precision:.2f}, R={m.recall:.2f})"
                )

        # Common misclassification patterns
        cause_counts: Counter = Counter()
        for mc in misclassifications:
            cause_counts[mc.likely_cause] += 1

        for cause, count in cause_counts.most_common(3):
            if count >= 3:
                recommendations.append(f"FIX: {cause} (affects {count} wallets)")

        # Missing ground truth
        if missing_gt:
            recommendations.append(
                f"NEED LABELS: {', '.join(missing_gt[:3])} (no ground truth)"
            )

        return recommendations

    def _empty_report(self) -> KnownWalletValidationReport:
        """Return empty report when validation not possible."""
        return KnownWalletValidationReport(
            total_wallets=0,
            accuracy=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            archetype_metrics={},
            confusion_matrix=[],
            misclassifications=[],
            misclassification_rate=0.0,
            archetypes_with_ground_truth=[],
            archetypes_missing_ground_truth=[a.value for a in Archetype],
            recommendations=["No labeled data available for validation"],
        )

    def load_ground_truth_from_csv(self, path: Path) -> dict[str, str]:
        """
        Load ground truth labels from CSV.

        Expected format:
            wallet,archetype
            ABC123,sniper
            DEF456,long_term_accumulator

        Args:
            path: Path to CSV file

        Returns:
            Dict mapping wallet to archetype
        """
        df = pd.read_csv(path)

        if "wallet" not in df.columns or "archetype" not in df.columns:
            raise ValueError("CSV must have 'wallet' and 'archetype' columns")

        return dict(zip(df["wallet"], df["archetype"]))


def generate_validation_report(
    report: KnownWalletValidationReport,
    output_path: Path,
) -> None:
    """Generate markdown report from validation results."""
    lines = [
        "# Known Wallet Validation Report",
        "",
        f"Generated: {report.computed_at.isoformat()}",
        "",
        "## Executive Summary",
        "",
        f"**Total Labeled Wallets:** {report.total_wallets}",
        f"**Accuracy:** {report.accuracy:.1%}",
        f"**Macro F1:** {report.macro_f1:.3f}",
        f"**Weighted F1:** {report.weighted_f1:.3f}",
        f"**Misclassification Rate:** {report.misclassification_rate:.1%}",
        "",
    ]

    # Per-archetype metrics
    lines.extend([
        "---",
        "",
        "## Per-Archetype Performance",
        "",
        "| Archetype | Precision | Recall | F1 | Support |",
        "|-----------|-----------|--------|-----|---------|",
    ])

    for arch, m in sorted(report.archetype_metrics.items(), key=lambda x: -x[1].f1_score):
        lines.append(
            f"| {arch} | {m.precision:.2f} | {m.recall:.2f} | {m.f1_score:.2f} | {m.support} |"
        )

    # Confusion matrix
    lines.extend([
        "",
        "---",
        "",
        "## Confusion Matrix (Top Entries)",
        "",
        "| True | Predicted | Count |",
        "|------|-----------|-------|",
    ])

    for entry in report.confusion_matrix[:15]:
        marker = "" if entry.true_archetype == entry.predicted_archetype else " **"
        lines.append(f"| {entry.true_archetype} | {entry.predicted_archetype} | {entry.count}{marker} |")

    # Misclassifications
    if report.misclassifications:
        lines.extend([
            "",
            "---",
            "",
            "## Notable Misclassifications",
            "",
        ])

        for mc in report.misclassifications[:10]:
            lines.extend([
                f"### {mc.wallet[:12]}...",
                f"- **True:** {mc.true_archetype}",
                f"- **Predicted:** {mc.predicted_archetype} ({mc.predicted_confidence:.2f})",
                f"- **Cause:** {mc.likely_cause}",
                "",
            ])

    # Recommendations
    lines.extend([
        "---",
        "",
        "## Recommendations",
        "",
    ])

    for rec in report.recommendations:
        lines.append(f"- {rec}")

    # Coverage
    if report.archetypes_missing_ground_truth:
        lines.extend([
            "",
            "---",
            "",
            "## Ground Truth Coverage",
            "",
            "**Archetypes WITH labels:**",
            f"  {', '.join(report.archetypes_with_ground_truth)}",
            "",
            "**Archetypes MISSING labels:**",
            f"  {', '.join(report.archetypes_missing_ground_truth)}",
            "",
        ])

    output_path.write_text("\n".join(lines))
    logger.info("validation_report_generated", path=str(output_path))
