"""
Review Queue (Sprint 10 - Deliverable 3).

Implements priority-based review queue for human labelling:
- Uncertainty weighting (high model uncertainty → high priority)
- Impact weighting (high-risk labels → high priority)
- Dataset balance weighting (underrepresented classes → high priority)
- Disagreement weighting (model-human disagreement → high priority)

Priority formula:
priority = uncertainty_weight + impact_weight + dataset_balance_weight + disagreement_weight
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from .schema import (
    LabelDomain,
    LabelRecord,
    LabelRepository,
    ReviewStatus,
)

logger = structlog.get_logger()


# ============================================================================
# Priority Configuration
# ============================================================================


@dataclass
class PriorityWeights:
    """Weights for priority calculation."""

    uncertainty: float = 0.3  # Weight for model uncertainty
    impact: float = 0.25  # Weight for potential impact
    dataset_balance: float = 0.25  # Weight for dataset balancing
    disagreement: float = 0.2  # Weight for model-human disagreement

    # Thresholds
    low_confidence_threshold: float = 0.5  # Below this = high uncertainty
    high_confidence_threshold: float = 0.8  # Above this = low uncertainty

    # Impact multipliers by domain
    domain_impact: dict[str, float] = field(default_factory=lambda: {
        "exit_event": 1.0,  # Affects PnL calculation
        "coordination": 1.2,  # Affects trust/risk
        "wallet_behaviour": 0.8,  # Profile accuracy
        "token_outcome": 1.1,  # Risk assessment
        "launch_trajectory": 0.9,  # Historical analysis
        "entity_resolution": 1.0,  # Entity accuracy
    })


# ============================================================================
# Review Priority
# ============================================================================


@dataclass
class ReviewPriority:
    """Priority score breakdown for a label."""

    total: float
    uncertainty_score: float
    impact_score: float
    balance_score: float
    disagreement_score: float
    factors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "uncertainty_score": round(self.uncertainty_score, 4),
            "impact_score": round(self.impact_score, 4),
            "balance_score": round(self.balance_score, 4),
            "disagreement_score": round(self.disagreement_score, 4),
            "factors": self.factors,
        }


# ============================================================================
# Review Queue Item
# ============================================================================


@dataclass
class ReviewQueueItem:
    """An item in the review queue."""

    label_id: str
    domain: LabelDomain
    object_type: str
    object_id: str
    proposed_label: str
    model_confidence: float
    priority: ReviewPriority
    created_at: datetime
    evidence_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "domain": self.domain.value,
            "object_type": self.object_type,
            "object_id": self.object_id,
            "proposed_label": self.proposed_label,
            "model_confidence": self.model_confidence,
            "priority": self.priority.to_dict(),
            "created_at": self.created_at.isoformat(),
            "evidence_summary": self.evidence_summary,
        }


# ============================================================================
# Review Queue
# ============================================================================


class ReviewQueue:
    """
    Priority-based review queue for human labelling.

    Prioritizes:
    - High uncertainty labels (low model confidence)
    - High impact labels (affects critical decisions)
    - Underrepresented classes (dataset balance)
    - Model-human disagreements (learning from errors)
    """

    def __init__(
        self,
        repository: LabelRepository,
        weights: PriorityWeights | None = None,
    ):
        self._repo = repository
        self._weights = weights or PriorityWeights()
        self._label_distribution: dict[str, dict[str, int]] | None = None
        self._distribution_updated_at: datetime | None = None

    def get_queue(
        self,
        domain: LabelDomain | None = None,
        limit: int = 50,
    ) -> list[ReviewQueueItem]:
        """
        Get prioritized review queue.

        Args:
            domain: Filter by domain (optional)
            limit: Maximum items to return

        Returns:
            List of ReviewQueueItem sorted by priority (highest first)
        """
        # Get pending labels
        pending = self._repo.get_labels(
            domain=domain,
            status=ReviewStatus.PENDING,
            limit=limit * 2,  # Get more than needed for sorting
        )

        # Update label distribution for balance calculation
        self._update_distribution()

        # Calculate priorities
        items = []
        for label in pending:
            priority = self._calculate_priority(label)
            item = ReviewQueueItem(
                label_id=label.label_id,
                domain=label.domain,
                object_type=label.object_type,
                object_id=label.object_id,
                proposed_label=label.proposed_label,
                model_confidence=label.model_confidence,
                priority=priority,
                created_at=label.created_at,
            )
            items.append(item)

        # Sort by priority (highest first)
        items.sort(key=lambda x: x.priority.total, reverse=True)

        return items[:limit]

    def get_disputed_queue(self, limit: int = 20) -> list[ReviewQueueItem]:
        """Get queue of disputed labels needing resolution."""
        disputed = self._repo.get_labels(status=ReviewStatus.DISPUTED, limit=limit)

        items = []
        for label in disputed:
            priority = self._calculate_priority(label)
            # Boost priority for disputes
            priority.total += 0.5
            priority.factors.append("disputed_needs_resolution")

            item = ReviewQueueItem(
                label_id=label.label_id,
                domain=label.domain,
                object_type=label.object_type,
                object_id=label.object_id,
                proposed_label=label.proposed_label,
                model_confidence=label.model_confidence,
                priority=priority,
                created_at=label.created_at,
            )
            items.append(item)

        items.sort(key=lambda x: x.priority.total, reverse=True)
        return items

    def get_verification_queue(self, limit: int = 50) -> list[ReviewQueueItem]:
        """Get queue of labelled items needing second review."""
        labelled = self._repo.get_labels(status=ReviewStatus.LABELLED, limit=limit * 2)

        items = []
        for label in labelled:
            priority = self._calculate_priority(label)
            # Adjust priority for verification (model-human agreement matters)
            if label.model_human_agree():
                priority.total *= 0.7  # Lower priority if model agreed
                priority.factors.append("model_agreed")
            else:
                priority.total *= 1.2  # Higher priority if model disagreed
                priority.factors.append("model_disagreed")

            item = ReviewQueueItem(
                label_id=label.label_id,
                domain=label.domain,
                object_type=label.object_type,
                object_id=label.object_id,
                proposed_label=label.proposed_label,
                model_confidence=label.model_confidence,
                priority=priority,
                created_at=label.created_at,
            )
            items.append(item)

        items.sort(key=lambda x: x.priority.total, reverse=True)
        return items[:limit]

    def _calculate_priority(self, label: LabelRecord) -> ReviewPriority:
        """Calculate priority score for a label."""
        factors = []

        # 1. Uncertainty score
        uncertainty_score = self._compute_uncertainty_score(label)
        if uncertainty_score > 0.5:
            factors.append(f"low_confidence:{label.model_confidence:.2f}")

        # 2. Impact score
        impact_score = self._compute_impact_score(label)
        if impact_score > 0.5:
            factors.append(f"high_impact:{label.domain.value}")

        # 3. Dataset balance score
        balance_score = self._compute_balance_score(label)
        if balance_score > 0.5:
            factors.append(f"rare_class:{label.proposed_label}")

        # 4. Disagreement score (for labelled items)
        disagreement_score = self._compute_disagreement_score(label)
        if disagreement_score > 0:
            factors.append("model_human_disagree")

        # Weighted sum
        total = (
            self._weights.uncertainty * uncertainty_score
            + self._weights.impact * impact_score
            + self._weights.dataset_balance * balance_score
            + self._weights.disagreement * disagreement_score
        )

        return ReviewPriority(
            total=total,
            uncertainty_score=uncertainty_score,
            impact_score=impact_score,
            balance_score=balance_score,
            disagreement_score=disagreement_score,
            factors=factors,
        )

    def _compute_uncertainty_score(self, label: LabelRecord) -> float:
        """
        Compute uncertainty score (0-1).

        Low confidence = high uncertainty = high score.
        """
        conf = label.model_confidence

        if conf <= self._weights.low_confidence_threshold:
            return 1.0
        elif conf >= self._weights.high_confidence_threshold:
            return 0.0
        else:
            # Linear interpolation
            range_size = self._weights.high_confidence_threshold - self._weights.low_confidence_threshold
            return 1.0 - (conf - self._weights.low_confidence_threshold) / range_size

    def _compute_impact_score(self, label: LabelRecord) -> float:
        """
        Compute impact score based on domain and label type.

        Some domains/labels have higher impact on downstream decisions.
        """
        base_impact = self._weights.domain_impact.get(label.domain.value, 1.0)

        # Boost for certain high-risk labels
        high_risk_labels = {
            "dex_sell",  # Affects PnL
            "true_coordinated",  # Trust/risk
            "rug_pull",  # Critical risk
            "same_entity",  # Entity claims
        }

        if label.proposed_label.lower() in high_risk_labels:
            base_impact *= 1.2

        return min(1.0, base_impact / 1.5)  # Normalize to 0-1

    def _compute_balance_score(self, label: LabelRecord) -> float:
        """
        Compute dataset balance score.

        Underrepresented classes get higher scores.
        """
        if not self._label_distribution:
            return 0.5  # Default if distribution unknown

        domain_dist = self._label_distribution.get(label.domain.value, {})
        if not domain_dist:
            return 0.5

        total = sum(domain_dist.values())
        if total == 0:
            return 0.5

        label_count = domain_dist.get(label.proposed_label, 0)
        label_ratio = label_count / total

        # Inverse ratio - rare classes get higher scores
        # If label is 5% of data, score = 0.95
        return 1.0 - label_ratio

    def _compute_disagreement_score(self, label: LabelRecord) -> float:
        """
        Compute disagreement score.

        1.0 if model and human disagree, 0.0 otherwise.
        """
        if label.human_label is None:
            return 0.0

        if label.proposed_label != label.human_label:
            return 1.0

        return 0.0

    def _update_distribution(self) -> None:
        """Update cached label distribution."""
        now = datetime.now(timezone.utc)

        # Refresh every 5 minutes
        if self._distribution_updated_at:
            age = (now - self._distribution_updated_at).total_seconds()
            if age < 300:
                return

        self._label_distribution = self._repo.get_label_counts()
        self._distribution_updated_at = now

    def get_progress(self) -> dict[str, Any]:
        """Get review progress statistics."""
        counts = self._repo.get_label_counts()

        total_pending = 0
        total_labelled = 0
        total_verified = 0
        total_disputed = 0

        for domain, statuses in counts.items():
            total_pending += statuses.get(ReviewStatus.PENDING.value, 0)
            total_labelled += statuses.get(ReviewStatus.LABELLED.value, 0)
            total_verified += statuses.get(ReviewStatus.VERIFIED.value, 0)
            total_disputed += statuses.get(ReviewStatus.DISPUTED.value, 0)

        total = total_pending + total_labelled + total_verified + total_disputed

        return {
            "total_labels": total,
            "pending": total_pending,
            "labelled": total_labelled,
            "verified": total_verified,
            "disputed": total_disputed,
            "completion_rate": (total_labelled + total_verified) / total if total > 0 else 0,
            "verification_rate": total_verified / (total_labelled + total_verified) if (total_labelled + total_verified) > 0 else 0,
            "dispute_rate": total_disputed / total if total > 0 else 0,
            "by_domain": counts,
        }


# ============================================================================
# Factory Function
# ============================================================================


def create_review_queue(
    repository: LabelRepository,
    weights: PriorityWeights | None = None,
) -> ReviewQueue:
    """Create a review queue."""
    return ReviewQueue(repository, weights)
