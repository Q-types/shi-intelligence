"""
Label Quality Metrics (Sprint 10 - Deliverable 5).

Computes quality metrics for the intelligence corpus:
- Inter-reviewer agreement
- Cohen's kappa (where dual review exists)
- Label distribution
- Disagreement rate
- Model-human agreement
- Per-domain precision and recall
- Ambiguous rate
- Needs-more-context rate

HARD RULE: Do not train supervised models until label quality metrics exist.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from .schema import (
    LabelDomain,
    LabelRepository,
    ReviewStatus,
)

logger = structlog.get_logger()


# ============================================================================
# Quality Metrics
# ============================================================================


@dataclass
class DomainMetrics:
    """Quality metrics for a single domain."""

    domain: str
    total_labels: int
    pending: int
    labelled: int
    verified: int
    disputed: int

    # Agreement metrics
    inter_reviewer_agreement: float | None  # % agreement on dual-reviewed
    cohens_kappa: float | None  # Chance-adjusted agreement

    # Model-human metrics
    model_human_agreement: float | None  # % where model == human
    model_precision: dict[str, float]  # Per-label precision
    model_recall: dict[str, float]  # Per-label recall

    # Distribution
    label_distribution: dict[str, int]

    # Quality indicators
    ambiguous_rate: float  # % labelled as ambiguous
    needs_context_rate: float  # % needing more context
    disagreement_rate: float  # % disputed

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "total_labels": self.total_labels,
            "pending": self.pending,
            "labelled": self.labelled,
            "verified": self.verified,
            "disputed": self.disputed,
            "inter_reviewer_agreement": self.inter_reviewer_agreement,
            "cohens_kappa": self.cohens_kappa,
            "model_human_agreement": self.model_human_agreement,
            "model_precision": self.model_precision,
            "model_recall": self.model_recall,
            "label_distribution": self.label_distribution,
            "ambiguous_rate": self.ambiguous_rate,
            "needs_context_rate": self.needs_context_rate,
            "disagreement_rate": self.disagreement_rate,
        }


@dataclass
class LabelQualityMetrics:
    """Overall quality metrics for the intelligence corpus."""

    computed_at: datetime
    total_labels: int
    total_reviewed: int  # labelled + verified + disputed
    total_verified: int

    # Overall agreement
    overall_inter_reviewer_agreement: float | None
    overall_cohens_kappa: float | None
    overall_model_human_agreement: float | None

    # By domain
    domain_metrics: dict[str, DomainMetrics]

    # Quality thresholds
    kappa_threshold_met: bool  # kappa >= 0.6
    agreement_threshold_met: bool  # agreement >= 0.8

    # Recommendations
    ready_for_training: bool
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "computed_at": self.computed_at.isoformat(),
            "total_labels": self.total_labels,
            "total_reviewed": self.total_reviewed,
            "total_verified": self.total_verified,
            "overall_inter_reviewer_agreement": self.overall_inter_reviewer_agreement,
            "overall_cohens_kappa": self.overall_cohens_kappa,
            "overall_model_human_agreement": self.overall_model_human_agreement,
            "domain_metrics": {
                k: v.to_dict() for k, v in self.domain_metrics.items()
            },
            "kappa_threshold_met": self.kappa_threshold_met,
            "agreement_threshold_met": self.agreement_threshold_met,
            "ready_for_training": self.ready_for_training,
            "recommendations": self.recommendations,
        }


# ============================================================================
# Cohen's Kappa Calculation
# ============================================================================


def compute_cohens_kappa(
    first_labels: list[str],
    second_labels: list[str],
) -> float | None:
    """
    Compute Cohen's kappa for inter-rater reliability.

    kappa = (p_o - p_e) / (1 - p_e)

    Where:
    - p_o = observed agreement
    - p_e = expected agreement by chance

    Returns:
        Kappa value (-1 to 1), or None if cannot compute
    """
    if len(first_labels) != len(second_labels):
        return None

    n = len(first_labels)
    if n == 0:
        return None

    # Get all unique labels
    all_labels = set(first_labels) | set(second_labels)

    # Count agreements
    agreements = sum(1 for a, b in zip(first_labels, second_labels) if a == b)
    p_o = agreements / n

    # Compute expected agreement by chance
    p_e = 0.0
    for label in all_labels:
        p_first = first_labels.count(label) / n
        p_second = second_labels.count(label) / n
        p_e += p_first * p_second

    # Compute kappa
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0

    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def compute_inter_reviewer_agreement(
    first_labels: list[str],
    second_labels: list[str],
) -> float | None:
    """
    Compute simple inter-reviewer agreement (% agreement).

    Returns:
        Agreement rate (0-1), or None if cannot compute
    """
    if len(first_labels) != len(second_labels):
        return None

    n = len(first_labels)
    if n == 0:
        return None

    agreements = sum(1 for a, b in zip(first_labels, second_labels) if a == b)
    return agreements / n


# ============================================================================
# Quality Metrics Computer
# ============================================================================


class QualityMetricsComputer:
    """
    Computes quality metrics for the intelligence corpus.

    Usage:
        computer = QualityMetricsComputer(repository)
        metrics = computer.compute()
    """

    def __init__(
        self,
        repository: LabelRepository,
        kappa_threshold: float = 0.6,
        agreement_threshold: float = 0.8,
    ):
        self._repo = repository
        self._kappa_threshold = kappa_threshold
        self._agreement_threshold = agreement_threshold

    def compute(self) -> LabelQualityMetrics:
        """Compute all quality metrics."""
        now = datetime.now(timezone.utc)

        # Get all labels
        all_labels = self._repo.get_labels(limit=100000)

        # Compute domain metrics
        domain_metrics = self._compute_domain_metrics(all_labels)

        # Compute overall metrics
        total_labels = len(all_labels)
        total_reviewed = sum(
            1 for l in all_labels
            if l.review_status in (ReviewStatus.LABELLED, ReviewStatus.VERIFIED, ReviewStatus.DISPUTED)
        )
        total_verified = sum(1 for l in all_labels if l.review_status == ReviewStatus.VERIFIED)

        # Compute overall agreement metrics
        overall_agreement, overall_kappa = self._compute_overall_agreement(all_labels)
        overall_model_human = self._compute_overall_model_human_agreement(all_labels)

        # Check thresholds
        kappa_met = overall_kappa is not None and overall_kappa >= self._kappa_threshold
        agreement_met = overall_agreement is not None and overall_agreement >= self._agreement_threshold

        # Generate recommendations
        recommendations = self._generate_recommendations(
            domain_metrics, overall_kappa, overall_agreement, total_verified
        )

        # Ready for training?
        ready = (
            kappa_met
            and agreement_met
            and total_verified >= 100  # Minimum verified samples
        )

        return LabelQualityMetrics(
            computed_at=now,
            total_labels=total_labels,
            total_reviewed=total_reviewed,
            total_verified=total_verified,
            overall_inter_reviewer_agreement=overall_agreement,
            overall_cohens_kappa=overall_kappa,
            overall_model_human_agreement=overall_model_human,
            domain_metrics=domain_metrics,
            kappa_threshold_met=kappa_met,
            agreement_threshold_met=agreement_met,
            ready_for_training=ready,
            recommendations=recommendations,
        )

    def _compute_domain_metrics(
        self,
        all_labels: list,
    ) -> dict[str, DomainMetrics]:
        """Compute metrics per domain."""
        # Group by domain
        by_domain: dict[str, list] = defaultdict(list)
        for label in all_labels:
            by_domain[label.domain.value].append(label)

        metrics = {}
        for domain, labels in by_domain.items():
            # Count by status
            pending = sum(1 for l in labels if l.review_status == ReviewStatus.PENDING)
            labelled = sum(1 for l in labels if l.review_status == ReviewStatus.LABELLED)
            verified = sum(1 for l in labels if l.review_status == ReviewStatus.VERIFIED)
            disputed = sum(1 for l in labels if l.review_status == ReviewStatus.DISPUTED)

            # Inter-reviewer agreement (from verified + disputed)
            dual_reviewed = [l for l in labels if l.second_label is not None]
            if dual_reviewed:
                first = [l.human_label for l in dual_reviewed]
                second = [l.second_label for l in dual_reviewed]
                agreement = compute_inter_reviewer_agreement(first, second)
                kappa = compute_cohens_kappa(first, second)
            else:
                agreement = None
                kappa = None

            # Model-human agreement
            reviewed = [l for l in labels if l.human_label is not None]
            if reviewed:
                model_human = sum(1 for l in reviewed if l.proposed_label == l.human_label) / len(reviewed)
            else:
                model_human = None

            # Precision and recall per label
            precision, recall = self._compute_precision_recall(labels)

            # Label distribution
            distribution: dict[str, int] = defaultdict(int)
            for l in labels:
                if l.human_label:
                    distribution[l.human_label] += 1
                else:
                    distribution[l.proposed_label] += 1

            # Ambiguous and needs-context rates
            reviewed_count = len(reviewed) if reviewed else 1
            ambiguous = sum(1 for l in reviewed if l.human_label and "ambiguous" in l.human_label.lower())
            needs_context = sum(1 for l in reviewed if l.human_label and "needs" in l.human_label.lower())

            total = len(labels) if labels else 1
            metrics[domain] = DomainMetrics(
                domain=domain,
                total_labels=len(labels),
                pending=pending,
                labelled=labelled,
                verified=verified,
                disputed=disputed,
                inter_reviewer_agreement=agreement,
                cohens_kappa=kappa,
                model_human_agreement=model_human,
                model_precision=precision,
                model_recall=recall,
                label_distribution=dict(distribution),
                ambiguous_rate=ambiguous / reviewed_count,
                needs_context_rate=needs_context / reviewed_count,
                disagreement_rate=disputed / total,
            )

        return metrics

    def _compute_precision_recall(
        self,
        labels: list,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute precision and recall for model predictions."""
        # Only use labels with human labels
        reviewed = [l for l in labels if l.human_label is not None]
        if not reviewed:
            return {}, {}

        # Build confusion matrix
        # true_positives[label] = model predicted label AND human confirmed
        # false_positives[label] = model predicted label BUT human said different
        # false_negatives[label] = model said different BUT human said label

        true_positives: dict[str, int] = defaultdict(int)
        false_positives: dict[str, int] = defaultdict(int)
        false_negatives: dict[str, int] = defaultdict(int)

        for l in reviewed:
            predicted = l.proposed_label
            actual = l.human_label

            if predicted == actual:
                true_positives[predicted] += 1
            else:
                false_positives[predicted] += 1
                false_negatives[actual] += 1

        # Compute precision and recall
        all_labels = set(true_positives.keys()) | set(false_positives.keys()) | set(false_negatives.keys())

        precision = {}
        recall = {}

        for label in all_labels:
            tp = true_positives[label]
            fp = false_positives[label]
            fn = false_negatives[label]

            if tp + fp > 0:
                precision[label] = tp / (tp + fp)
            else:
                precision[label] = 0.0

            if tp + fn > 0:
                recall[label] = tp / (tp + fn)
            else:
                recall[label] = 0.0

        return precision, recall

    def _compute_overall_agreement(
        self,
        all_labels: list,
    ) -> tuple[float | None, float | None]:
        """Compute overall inter-reviewer agreement and kappa."""
        dual_reviewed = [l for l in all_labels if l.second_label is not None]

        if not dual_reviewed:
            return None, None

        first = [l.human_label for l in dual_reviewed]
        second = [l.second_label for l in dual_reviewed]

        agreement = compute_inter_reviewer_agreement(first, second)
        kappa = compute_cohens_kappa(first, second)

        return agreement, kappa

    def _compute_overall_model_human_agreement(
        self,
        all_labels: list,
    ) -> float | None:
        """Compute overall model-human agreement."""
        reviewed = [l for l in all_labels if l.human_label is not None]

        if not reviewed:
            return None

        agreements = sum(1 for l in reviewed if l.proposed_label == l.human_label)
        return agreements / len(reviewed)

    def _generate_recommendations(
        self,
        domain_metrics: dict[str, DomainMetrics],
        overall_kappa: float | None,
        overall_agreement: float | None,
        total_verified: int,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Check verified count
        if total_verified < 100:
            recommendations.append(
                f"Need more verified labels ({total_verified}/100 minimum for training)"
            )

        # Check kappa
        if overall_kappa is None:
            recommendations.append("Need dual-reviewed samples to compute inter-rater reliability")
        elif overall_kappa < 0.4:
            recommendations.append(
                f"Low inter-rater reliability (kappa={overall_kappa:.2f}). "
                "Review labelling guidelines and train reviewers."
            )
        elif overall_kappa < 0.6:
            recommendations.append(
                f"Moderate inter-rater reliability (kappa={overall_kappa:.2f}). "
                "Target kappa >= 0.6 before training."
            )

        # Check domain-specific issues
        for domain, metrics in domain_metrics.items():
            if metrics.ambiguous_rate > 0.2:
                recommendations.append(
                    f"High ambiguous rate in {domain} ({metrics.ambiguous_rate:.1%}). "
                    "Consider improving evidence packages or refining label taxonomy."
                )

            if metrics.disagreement_rate > 0.15:
                recommendations.append(
                    f"High disagreement rate in {domain} ({metrics.disagreement_rate:.1%}). "
                    "Review disputed cases and update guidelines."
                )

            if metrics.pending > metrics.labelled + metrics.verified:
                recommendations.append(
                    f"Review backlog in {domain}: {metrics.pending} pending vs "
                    f"{metrics.labelled + metrics.verified} reviewed."
                )

        if not recommendations:
            recommendations.append("Label quality metrics look good. Ready for model training.")

        return recommendations

    def compute_domain(self, domain: LabelDomain) -> DomainMetrics | None:
        """Compute metrics for a single domain."""
        labels = self._repo.get_labels(domain=domain, limit=100000)
        if not labels:
            return None

        domain_metrics = self._compute_domain_metrics(labels)
        return domain_metrics.get(domain.value)
