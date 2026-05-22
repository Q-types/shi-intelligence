"""
Active Learning Hooks (Sprint 10 - Deliverable 7).

Provides hooks for future active learning:
- Uncertainty sampling (prioritize uncertain predictions)
- Disagreement sampling (prioritize model-human disagreements)
- Rare-class sampling (prioritize underrepresented classes)
- High-impact sampling (prioritize labels affecting critical decisions)

IMPORTANT: These are hooks only. Do NOT train models until label quality
metrics exist and meet thresholds.

HARD RULE: Do not train supervised models until label quality metrics exist.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
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
# Sampling Strategy
# ============================================================================


class SamplingStrategy(str, Enum):
    """Active learning sampling strategies."""

    UNCERTAINTY = "uncertainty"  # Sample by model uncertainty
    DISAGREEMENT = "disagreement"  # Sample by model-human disagreement
    RARE_CLASS = "rare_class"  # Sample underrepresented classes
    HIGH_IMPACT = "high_impact"  # Sample high-impact labels
    RANDOM = "random"  # Random sampling (baseline)
    HYBRID = "hybrid"  # Combination of strategies


# ============================================================================
# Sampling Result
# ============================================================================


@dataclass
class SamplingResult:
    """Result of a sampling operation."""

    strategy: SamplingStrategy
    sampled_ids: list[str]
    scores: dict[str, float]  # label_id -> score
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "sampled_ids": self.sampled_ids,
            "scores": self.scores,
            "metadata": self.metadata,
        }


# ============================================================================
# Base Sampler
# ============================================================================


class BaseSampler(ABC):
    """Base class for active learning samplers."""

    def __init__(self, repository: LabelRepository):
        self._repo = repository

    @abstractmethod
    def sample(
        self,
        pool: list[LabelRecord],
        n: int,
    ) -> SamplingResult:
        """
        Sample n items from the pool.

        Args:
            pool: Pool of unlabelled items
            n: Number of items to sample

        Returns:
            SamplingResult with sampled IDs and scores
        """
        pass

    def get_pool(
        self,
        domain: LabelDomain | None = None,
        limit: int = 10000,
    ) -> list[LabelRecord]:
        """Get pool of pending labels."""
        return self._repo.get_labels(
            domain=domain,
            status=ReviewStatus.PENDING,
            limit=limit,
        )


# ============================================================================
# Uncertainty Sampler
# ============================================================================


class UncertaintySampler(BaseSampler):
    """
    Sample by model uncertainty.

    Prioritizes labels where the model has low confidence.
    This helps improve model performance on ambiguous cases.
    """

    def sample(
        self,
        pool: list[LabelRecord],
        n: int,
    ) -> SamplingResult:
        """Sample n most uncertain items."""
        # Score by uncertainty (1 - confidence)
        scores = {
            l.label_id: 1.0 - l.model_confidence
            for l in pool
        }

        # Sort by uncertainty (highest first)
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        sampled_ids = sorted_ids[:n]

        return SamplingResult(
            strategy=SamplingStrategy.UNCERTAINTY,
            sampled_ids=sampled_ids,
            scores={id: scores[id] for id in sampled_ids},
            metadata={
                "pool_size": len(pool),
                "mean_uncertainty": sum(scores.values()) / len(scores) if scores else 0,
            },
        )


# ============================================================================
# Disagreement Sampler
# ============================================================================


class DisagreementSampler(BaseSampler):
    """
    Sample by model-human disagreement patterns.

    Prioritizes labels similar to previous disagreements.
    This helps correct systematic model errors.
    """

    def __init__(self, repository: LabelRepository):
        super().__init__(repository)
        self._disagreement_patterns: dict[str, set[str]] = {}
        self._update_patterns()

    def _update_patterns(self) -> None:
        """Update disagreement patterns from labelled data."""
        labelled = self._repo.get_labels(status=ReviewStatus.LABELLED, limit=10000)
        verified = self._repo.get_labels(status=ReviewStatus.VERIFIED, limit=10000)
        disputed = self._repo.get_labels(status=ReviewStatus.DISPUTED, limit=10000)

        # Find patterns where model != human
        for label in labelled + verified + disputed:
            if label.human_label and label.proposed_label != label.human_label:
                pattern_key = f"{label.domain.value}:{label.proposed_label}"
                if pattern_key not in self._disagreement_patterns:
                    self._disagreement_patterns[pattern_key] = set()
                # Store factors that led to disagreement
                # In practice, would extract features from evidence

    def sample(
        self,
        pool: list[LabelRecord],
        n: int,
    ) -> SamplingResult:
        """Sample items similar to previous disagreements."""
        scores = {}

        for label in pool:
            pattern_key = f"{label.domain.value}:{label.proposed_label}"

            # Higher score if this pattern has seen disagreements
            if pattern_key in self._disagreement_patterns:
                base_score = 0.5 + 0.5 * min(len(self._disagreement_patterns[pattern_key]) / 10, 1.0)
            else:
                base_score = 0.1

            # Boost if confidence is moderate (more likely to be wrong)
            if 0.4 <= label.model_confidence <= 0.7:
                base_score *= 1.3

            scores[label.label_id] = base_score

        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        sampled_ids = sorted_ids[:n]

        return SamplingResult(
            strategy=SamplingStrategy.DISAGREEMENT,
            sampled_ids=sampled_ids,
            scores={id: scores[id] for id in sampled_ids},
            metadata={
                "pool_size": len(pool),
                "known_disagreement_patterns": len(self._disagreement_patterns),
            },
        )


# ============================================================================
# Rare Class Sampler
# ============================================================================


class RareClassSampler(BaseSampler):
    """
    Sample underrepresented classes.

    Prioritizes labels from classes with few examples.
    This helps balance the dataset.
    """

    def __init__(self, repository: LabelRepository):
        super().__init__(repository)
        self._class_counts: dict[str, int] = {}
        self._update_counts()

    def _update_counts(self) -> None:
        """Update class counts from labelled data."""
        labelled = self._repo.get_labels(status=ReviewStatus.LABELLED, limit=100000)
        verified = self._repo.get_labels(status=ReviewStatus.VERIFIED, limit=100000)

        self._class_counts = {}
        for label in labelled + verified:
            key = f"{label.domain.value}:{label.human_label or label.proposed_label}"
            self._class_counts[key] = self._class_counts.get(key, 0) + 1

    def sample(
        self,
        pool: list[LabelRecord],
        n: int,
    ) -> SamplingResult:
        """Sample items from rare classes."""
        if not self._class_counts:
            self._update_counts()

        total = sum(self._class_counts.values()) if self._class_counts else 1
        scores = {}

        for label in pool:
            key = f"{label.domain.value}:{label.proposed_label}"
            count = self._class_counts.get(key, 0)

            # Inverse frequency weighting
            if count == 0:
                score = 1.0  # Completely new class
            else:
                frequency = count / total
                score = 1.0 - frequency  # Rare = high score

            scores[label.label_id] = score

        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        sampled_ids = sorted_ids[:n]

        return SamplingResult(
            strategy=SamplingStrategy.RARE_CLASS,
            sampled_ids=sampled_ids,
            scores={id: scores[id] for id in sampled_ids},
            metadata={
                "pool_size": len(pool),
                "class_counts": self._class_counts,
            },
        )


# ============================================================================
# High Impact Sampler
# ============================================================================


class HighImpactSampler(BaseSampler):
    """
    Sample high-impact labels.

    Prioritizes labels that affect critical downstream decisions.
    """

    # Impact weights by domain
    DOMAIN_IMPACT = {
        "exit_event": 1.0,  # Affects PnL
        "coordination": 1.2,  # Affects trust/risk
        "wallet_behaviour": 0.8,
        "token_outcome": 1.1,
        "launch_trajectory": 0.9,
        "entity_resolution": 1.0,
    }

    # High-impact labels
    HIGH_IMPACT_LABELS = {
        "dex_sell",  # PnL calculation
        "true_coordinated",  # Trust assessment
        "rug_pull",  # Risk warning
        "same_entity",  # Entity claims
        "sniper",  # Behaviour profile
    }

    def sample(
        self,
        pool: list[LabelRecord],
        n: int,
    ) -> SamplingResult:
        """Sample high-impact items."""
        scores = {}

        for label in pool:
            # Base domain impact
            base = self.DOMAIN_IMPACT.get(label.domain.value, 1.0)

            # Boost for high-impact labels
            if label.proposed_label in self.HIGH_IMPACT_LABELS:
                base *= 1.5

            # Boost for low confidence high-impact (risky)
            if label.model_confidence < 0.6:
                base *= 1.2

            scores[label.label_id] = base

        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        sampled_ids = sorted_ids[:n]

        return SamplingResult(
            strategy=SamplingStrategy.HIGH_IMPACT,
            sampled_ids=sampled_ids,
            scores={id: scores[id] for id in sampled_ids},
            metadata={
                "pool_size": len(pool),
            },
        )


# ============================================================================
# Active Learning Hooks
# ============================================================================


class ActiveLearningHooks:
    """
    Hooks for active learning integration.

    IMPORTANT: These hooks prepare for active learning but do NOT
    train models. Model training requires passing label quality gates.
    """

    def __init__(self, repository: LabelRepository):
        self._repo = repository
        self._samplers = {
            SamplingStrategy.UNCERTAINTY: UncertaintySampler(repository),
            SamplingStrategy.DISAGREEMENT: DisagreementSampler(repository),
            SamplingStrategy.RARE_CLASS: RareClassSampler(repository),
            SamplingStrategy.HIGH_IMPACT: HighImpactSampler(repository),
        }
        self._training_allowed = False

    def sample(
        self,
        strategy: SamplingStrategy,
        n: int,
        domain: LabelDomain | None = None,
    ) -> SamplingResult:
        """
        Sample n items using the specified strategy.

        Args:
            strategy: Sampling strategy to use
            n: Number of items to sample
            domain: Filter to specific domain

        Returns:
            SamplingResult with sampled label IDs
        """
        sampler = self._samplers.get(strategy)
        if not sampler:
            raise ValueError(f"Unknown strategy: {strategy}")

        pool = sampler.get_pool(domain=domain)
        return sampler.sample(pool, n)

    def sample_hybrid(
        self,
        n: int,
        weights: dict[SamplingStrategy, float] | None = None,
        domain: LabelDomain | None = None,
    ) -> SamplingResult:
        """
        Sample using a weighted combination of strategies.

        Args:
            n: Total items to sample
            weights: Strategy weights (default: equal)
            domain: Filter to specific domain

        Returns:
            Combined SamplingResult
        """
        if weights is None:
            weights = {
                SamplingStrategy.UNCERTAINTY: 0.3,
                SamplingStrategy.DISAGREEMENT: 0.25,
                SamplingStrategy.RARE_CLASS: 0.25,
                SamplingStrategy.HIGH_IMPACT: 0.2,
            }

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Get samples from each strategy
        all_scores: dict[str, float] = {}
        for strategy, weight in weights.items():
            if strategy not in self._samplers:
                continue

            sampler = self._samplers[strategy]
            pool = sampler.get_pool(domain=domain)
            result = sampler.sample(pool, len(pool))  # Score all

            # Weighted scores
            for label_id, score in result.scores.items():
                all_scores[label_id] = all_scores.get(label_id, 0) + score * weight

        # Sort by combined score
        sorted_ids = sorted(all_scores.keys(), key=lambda x: all_scores[x], reverse=True)
        sampled_ids = sorted_ids[:n]

        return SamplingResult(
            strategy=SamplingStrategy.HYBRID,
            sampled_ids=sampled_ids,
            scores={id: all_scores[id] for id in sampled_ids},
            metadata={
                "weights": {k.value: v for k, v in weights.items()},
            },
        )

    def register_for_training(
        self,
        callback: callable,
        min_kappa: float = 0.6,
        min_verified: int = 100,
    ) -> None:
        """
        Register callback for when training is allowed.

        The callback will be invoked when label quality meets thresholds.
        Until then, no model training should occur.

        Args:
            callback: Function to call when training is allowed
            min_kappa: Minimum Cohen's kappa required
            min_verified: Minimum verified samples required
        """
        logger.info(
            "active_learning_training_registered",
            min_kappa=min_kappa,
            min_verified=min_verified,
            note="Training will only be allowed when quality thresholds are met",
        )
        # Store callback for future use
        self._training_callback = callback
        self._min_kappa = min_kappa
        self._min_verified = min_verified

    def check_training_readiness(self) -> dict[str, Any]:
        """
        Check if corpus is ready for model training.

        Returns:
            Readiness status and any blocking issues
        """
        from .quality_metrics import QualityMetricsComputer

        computer = QualityMetricsComputer(self._repo)
        metrics = computer.compute()

        return {
            "ready": metrics.ready_for_training,
            "kappa": metrics.overall_cohens_kappa,
            "kappa_threshold": self._min_kappa if hasattr(self, "_min_kappa") else 0.6,
            "verified_count": metrics.total_verified,
            "verified_threshold": self._min_verified if hasattr(self, "_min_verified") else 100,
            "recommendations": metrics.recommendations,
        }

    def get_sampling_stats(self) -> dict[str, Any]:
        """Get statistics about the sampling pool."""
        pending = self._repo.get_pending_count()
        counts = self._repo.get_label_counts()

        return {
            "pending_labels": pending,
            "by_domain": {
                domain: statuses.get(ReviewStatus.PENDING.value, 0)
                for domain, statuses in counts.items()
            },
        }
