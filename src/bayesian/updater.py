"""Bayesian belief updating with evidence.

This module provides the evidence types and updating logic
for Bayesian risk models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

import numpy as np
import structlog

from .priors import BetaPrior, PriorDistribution

logger = structlog.get_logger()


class EvidenceType(Enum):
    """Types of evidence that can update beliefs."""

    # Holder-related evidence
    CONCENTRATION_CHANGE = "concentration_change"
    LARGE_WALLET_MOVEMENT = "large_wallet_movement"
    NEW_HOLDER_PATTERN = "new_holder_pattern"
    HOLDER_DISTRIBUTION_SHIFT = "holder_distribution_shift"

    # Market/liquidity evidence
    LIQUIDITY_CHANGE = "liquidity_change"
    VOLUME_ANOMALY = "volume_anomaly"
    PRICE_VOLATILITY = "price_volatility"

    # Behavioral evidence
    REGIME_TRANSITION = "regime_transition"
    ANOMALY_DETECTION = "anomaly_detection"
    COORDINATED_ACTIVITY = "coordinated_activity"
    DUMP_SIGNATURE = "dump_signature"

    # Historical evidence
    SIMILAR_TOKEN_OUTCOME = "similar_token_outcome"
    PATTERN_MATCH = "pattern_match"

    # External signals
    SOCIAL_SENTIMENT = "social_sentiment"
    DEV_ACTIVITY = "dev_activity"


@dataclass(frozen=True)
class Evidence:
    """A piece of evidence for belief updating.

    Attributes
    ----------
    evidence_type : EvidenceType
        Type of evidence.
    value : float
        Observed value (interpretation depends on type).
    timestamp : datetime
        When evidence was observed.
    strength : float
        Strength/reliability of evidence (0 to 1).
    direction : float
        Direction of evidence (-1 = safe, +1 = risky).
    source : str
        Source identifier for the evidence.
    metadata : dict[str, Any]
        Additional context about the evidence.
    """

    evidence_type: EvidenceType
    value: float
    timestamp: datetime
    strength: float = 1.0
    direction: float = 0.0
    source: str = "system"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate evidence."""
        if not 0 <= self.strength <= 1:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")
        if not -1 <= self.direction <= 1:
            raise ValueError(f"direction must be in [-1, 1], got {self.direction}")

    @property
    def is_risky(self) -> bool:
        """Check if evidence points toward higher risk."""
        return self.direction > 0

    @property
    def effective_strength(self) -> float:
        """Compute effective strength (strength * |direction|)."""
        return self.strength * abs(self.direction)

    def decay(self, half_life_hours: float = 24.0) -> "Evidence":
        """Apply time decay to evidence strength.

        Parameters
        ----------
        half_life_hours : float
            Half-life for decay in hours.

        Returns
        -------
        Evidence
            New evidence with decayed strength.
        """
        now = datetime.now(timezone.utc)
        age_hours = (now - self.timestamp).total_seconds() / 3600

        decay_factor = 0.5 ** (age_hours / half_life_hours)

        return Evidence(
            evidence_type=self.evidence_type,
            value=self.value,
            timestamp=self.timestamp,
            strength=self.strength * decay_factor,
            direction=self.direction,
            source=self.source,
            metadata={**self.metadata, "decayed": True},
        )


@dataclass
class EvidenceBatch:
    """A batch of evidence for updating.

    Attributes
    ----------
    evidences : list[Evidence]
        List of evidence items.
    token_mint : str | None
        Associated token (if any).
    batch_timestamp : datetime
        When batch was created.
    """

    evidences: list[Evidence] = field(default_factory=list)
    token_mint: str | None = None
    batch_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add(self, evidence: Evidence) -> None:
        """Add evidence to batch."""
        self.evidences.append(evidence)

    def filter_by_type(self, *types: EvidenceType) -> "EvidenceBatch":
        """Filter to specific evidence types.

        Parameters
        ----------
        *types : EvidenceType
            Types to include.

        Returns
        -------
        EvidenceBatch
            Filtered batch.
        """
        filtered = [e for e in self.evidences if e.evidence_type in types]
        return EvidenceBatch(
            evidences=filtered,
            token_mint=self.token_mint,
            batch_timestamp=self.batch_timestamp,
        )

    @property
    def total_strength(self) -> float:
        """Sum of evidence strengths."""
        return sum(e.strength for e in self.evidences)

    @property
    def net_direction(self) -> float:
        """Weighted average direction."""
        if not self.evidences:
            return 0.0

        total_weight = sum(e.strength for e in self.evidences)
        if total_weight == 0:
            return 0.0

        weighted_dir = sum(e.direction * e.strength for e in self.evidences)
        return weighted_dir / total_weight

    def __len__(self) -> int:
        """Number of evidence items."""
        return len(self.evidences)


@dataclass
class UpdateConfig:
    """Configuration for belief updating.

    Attributes
    ----------
    min_strength_threshold : float
        Minimum strength to consider evidence.
    evidence_decay_hours : float
        Half-life for evidence decay.
    max_single_update : float
        Maximum change from single evidence.
    enable_decay : bool
        Whether to apply time decay.
    """

    min_strength_threshold: float = 0.1
    evidence_decay_hours: float = 24.0
    max_single_update: float = 2.0
    enable_decay: bool = True


class BayesianUpdater:
    """Update Bayesian beliefs with evidence.

    This class provides methods for updating prior distributions
    with observed evidence, computing posterior distributions.

    Parameters
    ----------
    config : UpdateConfig | None
        Updater configuration.

    Examples
    --------
    >>> updater = BayesianUpdater()
    >>> prior = BetaPrior(alpha=2, beta_=5)
    >>> evidence = Evidence(
    ...     evidence_type=EvidenceType.CONCENTRATION_CHANGE,
    ...     value=0.8,
    ...     timestamp=datetime.now(timezone.utc),
    ...     strength=0.7,
    ...     direction=0.5,
    ... )
    >>> posterior = updater.update_beta(prior, evidence)
    """

    # Mapping from evidence type to update weight
    EVIDENCE_WEIGHTS: dict[EvidenceType, float] = {
        EvidenceType.CONCENTRATION_CHANGE: 1.0,
        EvidenceType.LARGE_WALLET_MOVEMENT: 1.5,
        EvidenceType.NEW_HOLDER_PATTERN: 0.8,
        EvidenceType.HOLDER_DISTRIBUTION_SHIFT: 1.0,
        EvidenceType.LIQUIDITY_CHANGE: 1.2,
        EvidenceType.VOLUME_ANOMALY: 0.9,
        EvidenceType.PRICE_VOLATILITY: 0.7,
        EvidenceType.REGIME_TRANSITION: 1.5,
        EvidenceType.ANOMALY_DETECTION: 1.3,
        EvidenceType.COORDINATED_ACTIVITY: 1.8,
        EvidenceType.DUMP_SIGNATURE: 2.0,
        EvidenceType.SIMILAR_TOKEN_OUTCOME: 1.4,
        EvidenceType.PATTERN_MATCH: 1.1,
        EvidenceType.SOCIAL_SENTIMENT: 0.5,
        EvidenceType.DEV_ACTIVITY: 0.6,
    }

    def __init__(self, config: UpdateConfig | None = None) -> None:
        """Initialize updater with configuration."""
        self.config = config or UpdateConfig()
        logger.info(
            "bayesian_updater_initialized",
            min_threshold=self.config.min_strength_threshold,
            decay_hours=self.config.evidence_decay_hours,
        )

    def update_beta(
        self,
        prior: BetaPrior,
        evidence: Evidence | EvidenceBatch,
    ) -> BetaPrior:
        """Update Beta prior with evidence.

        Parameters
        ----------
        prior : BetaPrior
            Prior Beta distribution.
        evidence : Evidence | EvidenceBatch
            Evidence to incorporate.

        Returns
        -------
        BetaPrior
            Posterior Beta distribution.
        """
        if isinstance(evidence, EvidenceBatch):
            return self._update_beta_batch(prior, evidence)

        # Apply decay if enabled
        if self.config.enable_decay:
            evidence = evidence.decay(self.config.evidence_decay_hours)

        # Check threshold
        if evidence.strength < self.config.min_strength_threshold:
            logger.debug(
                "evidence_below_threshold",
                evidence_type=evidence.evidence_type.value,
                strength=evidence.strength,
            )
            return prior

        # Get weight for this evidence type
        weight = self.EVIDENCE_WEIGHTS.get(evidence.evidence_type, 1.0)

        # Compute update magnitude
        update_magnitude = min(
            evidence.effective_strength * weight,
            self.config.max_single_update,
        )

        # Update the prior
        posterior = prior.update_continuous(
            evidence_strength=update_magnitude,
            direction=evidence.direction,
        )

        logger.debug(
            "beta_updated",
            evidence_type=evidence.evidence_type.value,
            prior_mean=prior.mean,
            posterior_mean=posterior.mean,
            update_magnitude=update_magnitude,
        )

        return posterior

    def _update_beta_batch(
        self, prior: BetaPrior, batch: EvidenceBatch
    ) -> BetaPrior:
        """Update with a batch of evidence.

        Parameters
        ----------
        prior : BetaPrior
            Prior distribution.
        batch : EvidenceBatch
            Batch of evidence.

        Returns
        -------
        BetaPrior
            Posterior distribution.
        """
        current = prior

        for evidence in batch.evidences:
            current = self.update_beta(current, evidence)

        return current

    def compute_information_gain(
        self, prior: BetaPrior, evidence: Evidence
    ) -> float:
        """Compute expected information gain from evidence.

        Uses KL divergence between posterior and prior as measure
        of information gained.

        Parameters
        ----------
        prior : BetaPrior
            Prior distribution.
        evidence : Evidence
            Evidence to evaluate.

        Returns
        -------
        float
            Information gain in nats.
        """
        posterior = self.update_beta(prior, evidence)

        # Compute KL divergence via sampling approximation
        samples = prior.sample(size=1000, random_state=42)

        prior_log_p = np.log(prior.pdf(samples) + 1e-10)
        posterior_log_p = np.log(posterior.pdf(samples) + 1e-10)

        kl_div = float(np.mean(posterior_log_p - prior_log_p))

        return max(0, kl_div)  # Should be non-negative

    def compute_surprise(self, prior: BetaPrior, evidence: Evidence) -> float:
        """Compute surprise of evidence given prior.

        Surprise is -log(P(evidence | prior)), higher = more unexpected.

        Parameters
        ----------
        prior : BetaPrior
            Prior distribution.
        evidence : Evidence
            Observed evidence.

        Returns
        -------
        float
            Surprise value (in nats).
        """
        # Map direction to probability space
        prob_point = (evidence.direction + 1) / 2  # Map [-1, 1] to [0, 1]

        # Get prior probability density at this point
        density = prior.pdf(prob_point)

        # Surprise = -log(density)
        surprise = -np.log(max(density, 1e-10))

        return float(surprise)

    def suggest_next_evidence(
        self,
        prior: BetaPrior,
        available_types: Sequence[EvidenceType],
    ) -> EvidenceType:
        """Suggest most valuable evidence type to gather.

        Returns the evidence type with highest expected information gain.

        Parameters
        ----------
        prior : BetaPrior
            Current prior distribution.
        available_types : Sequence[EvidenceType]
            Available evidence types.

        Returns
        -------
        EvidenceType
            Recommended evidence type.
        """
        best_type = available_types[0]
        best_expected_gain = 0.0

        for ev_type in available_types:
            # Create hypothetical evidence with different directions
            gains = []
            for direction in [-0.5, 0.0, 0.5]:
                hypo_evidence = Evidence(
                    evidence_type=ev_type,
                    value=0.5,
                    timestamp=datetime.now(timezone.utc),
                    strength=0.7,
                    direction=direction,
                )
                gain = self.compute_information_gain(prior, hypo_evidence)
                gains.append(gain)

            expected_gain = np.mean(gains)

            if expected_gain > best_expected_gain:
                best_expected_gain = expected_gain
                best_type = ev_type

        logger.debug(
            "suggested_evidence_type",
            evidence_type=best_type.value,
            expected_gain=best_expected_gain,
        )

        return best_type


# Convenience functions for creating evidence
def create_concentration_evidence(
    old_hhi: float,
    new_hhi: float,
    timestamp: datetime | None = None,
) -> Evidence:
    """Create evidence from HHI concentration change.

    Parameters
    ----------
    old_hhi : float
        Previous HHI value.
    new_hhi : float
        New HHI value.
    timestamp : datetime | None
        Evidence timestamp.

    Returns
    -------
    Evidence
        Concentration change evidence.
    """
    change = new_hhi - old_hhi
    # Positive change (more concentrated) = higher risk
    direction = np.clip(change * 5, -1, 1)  # Scale and clip

    return Evidence(
        evidence_type=EvidenceType.CONCENTRATION_CHANGE,
        value=new_hhi,
        timestamp=timestamp or datetime.now(timezone.utc),
        strength=min(abs(change) * 10, 1.0),  # Stronger for larger changes
        direction=float(direction),
        metadata={"old_hhi": old_hhi, "new_hhi": new_hhi, "change": change},
    )


def create_anomaly_evidence(
    anomaly_score: float,
    wallet_count: int,
    timestamp: datetime | None = None,
) -> Evidence:
    """Create evidence from anomaly detection.

    Parameters
    ----------
    anomaly_score : float
        Anomaly score (0 to 1, higher = more anomalous).
    wallet_count : int
        Number of anomalous wallets.
    timestamp : datetime | None
        Evidence timestamp.

    Returns
    -------
    Evidence
        Anomaly detection evidence.
    """
    # High anomaly score = higher risk
    direction = np.clip(anomaly_score * 2 - 1, -1, 1)
    strength = min(0.5 + wallet_count * 0.1, 1.0)

    return Evidence(
        evidence_type=EvidenceType.ANOMALY_DETECTION,
        value=anomaly_score,
        timestamp=timestamp or datetime.now(timezone.utc),
        strength=strength,
        direction=float(direction),
        metadata={"anomaly_score": anomaly_score, "wallet_count": wallet_count},
    )


def create_regime_evidence(
    from_regime: str,
    to_regime: str,
    confidence: float,
    timestamp: datetime | None = None,
) -> Evidence:
    """Create evidence from regime transition.

    Parameters
    ----------
    from_regime : str
        Previous regime.
    to_regime : str
        New regime.
    confidence : float
        Transition confidence.
    timestamp : datetime | None
        Evidence timestamp.

    Returns
    -------
    Evidence
        Regime transition evidence.
    """
    # Map regime transitions to risk direction
    risk_map = {
        ("accumulating", "distributing"): 0.8,
        ("stable", "distributing"): 0.6,
        ("distributing", "accumulating"): -0.5,
        ("distributing", "stable"): -0.3,
    }

    direction = risk_map.get((from_regime, to_regime), 0.0)

    return Evidence(
        evidence_type=EvidenceType.REGIME_TRANSITION,
        value=confidence,
        timestamp=timestamp or datetime.now(timezone.utc),
        strength=confidence,
        direction=direction,
        metadata={"from_regime": from_regime, "to_regime": to_regime},
    )
