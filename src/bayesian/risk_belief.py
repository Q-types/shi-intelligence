"""Composite Bayesian risk belief model.

This module provides the main RiskBeliefModel class that combines
multiple risk factors into a unified uncertainty-aware risk assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import structlog

from .priors import BetaPrior, PriorDistribution, create_default_priors
from .updater import (
    BayesianUpdater,
    Evidence,
    EvidenceBatch,
    EvidenceType,
    UpdateConfig,
)

logger = structlog.get_logger()


@dataclass(frozen=True)
class RiskEstimate:
    """A point estimate of risk with uncertainty.

    Attributes
    ----------
    mean : float
        Point estimate (posterior mean).
    lower_ci : float
        Lower bound of credible interval.
    upper_ci : float
        Upper bound of credible interval.
    std : float
        Standard deviation of estimate.
    confidence_level : float
        Confidence level for interval (e.g., 0.95).
    """

    mean: float
    lower_ci: float
    upper_ci: float
    std: float
    confidence_level: float = 0.95

    @property
    def width(self) -> float:
        """Width of credible interval."""
        return self.upper_ci - self.lower_ci

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty (width / mean)."""
        if self.mean == 0:
            return float("inf")
        return self.width / self.mean

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "lower_ci": self.lower_ci,
            "upper_ci": self.upper_ci,
            "std": self.std,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True)
class BeliefUpdate:
    """Record of a belief update.

    Attributes
    ----------
    evidence : Evidence
        Evidence that triggered update.
    prior_mean : float
        Mean before update.
    posterior_mean : float
        Mean after update.
    information_gain : float
        Information gained from update.
    timestamp : datetime
        When update occurred.
    """

    evidence: Evidence
    prior_mean: float
    posterior_mean: float
    information_gain: float
    timestamp: datetime

    @property
    def mean_shift(self) -> float:
        """Change in mean from update."""
        return self.posterior_mean - self.prior_mean


@dataclass
class RiskBeliefState:
    """Complete state of a risk belief model.

    Attributes
    ----------
    rug_probability : BetaPrior
        Distribution over P(rug).
    concentration_risk : BetaPrior
        Distribution over concentration risk.
    liquidity_risk : BetaPrior
        Distribution over liquidity risk.
    coordination_risk : BetaPrior
        Distribution over coordination/sybil risk.
    update_history : list[BeliefUpdate]
        History of updates.
    created_at : datetime
        When state was created.
    last_updated : datetime
        When state was last updated.
    """

    rug_probability: BetaPrior
    concentration_risk: BetaPrior
    liquidity_risk: BetaPrior
    coordination_risk: BetaPrior
    update_history: list[BeliefUpdate] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_updates(self) -> int:
        """Total number of updates applied."""
        return len(self.update_history)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "rug_probability": self.rug_probability.to_dict(),
            "concentration_risk": self.concentration_risk.to_dict(),
            "liquidity_risk": self.liquidity_risk.to_dict(),
            "coordination_risk": self.coordination_risk.to_dict(),
            "total_updates": self.total_updates,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


class RiskBeliefModel:
    """Bayesian risk belief model with uncertainty quantification.

    This class maintains probability distributions over multiple risk
    factors and provides methods for updating beliefs with evidence
    and computing risk estimates with uncertainty.

    Parameters
    ----------
    prior_alpha : float
        Alpha parameter for rug probability prior.
    prior_beta : float
        Beta parameter for rug probability prior.
    updater_config : UpdateConfig | None
        Configuration for belief updater.

    Examples
    --------
    >>> model = RiskBeliefModel(prior_alpha=2, prior_beta=5)
    >>> evidence = Evidence(
    ...     evidence_type=EvidenceType.CONCENTRATION_CHANGE,
    ...     value=0.8,
    ...     timestamp=datetime.now(timezone.utc),
    ...     strength=0.7,
    ...     direction=0.5,
    ... )
    >>> model.update(evidence)
    >>> estimate = model.posterior_rug_probability()
    >>> print(f"P(rug) = {estimate.mean:.2%} [{estimate.lower_ci:.2%}, {estimate.upper_ci:.2%}]")
    """

    # Weights for combining risk factors into overall risk
    RISK_WEIGHTS: dict[str, float] = {
        "rug_probability": 0.4,
        "concentration_risk": 0.25,
        "liquidity_risk": 0.2,
        "coordination_risk": 0.15,
    }

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        updater_config: UpdateConfig | None = None,
    ) -> None:
        """Initialize risk belief model with priors."""
        self._state = RiskBeliefState(
            rug_probability=BetaPrior(alpha=prior_alpha, beta_=prior_beta),
            concentration_risk=BetaPrior.from_mean_concentration(0.5, 4),
            liquidity_risk=BetaPrior.from_mean_concentration(0.4, 4),
            coordination_risk=BetaPrior.from_mean_concentration(0.2, 5),
        )
        self._updater = BayesianUpdater(config=updater_config)

        logger.info(
            "risk_belief_model_initialized",
            prior_mean=self._state.rug_probability.mean,
            prior_concentration=self._state.rug_probability.concentration,
        )

    @property
    def state(self) -> RiskBeliefState:
        """Current belief state."""
        return self._state

    @property
    def alpha(self) -> float:
        """Current alpha parameter for rug probability."""
        return self._state.rug_probability.alpha

    @property
    def beta(self) -> float:
        """Current beta parameter for rug probability."""
        return self._state.rug_probability.beta_

    def update(self, evidence: Evidence | EvidenceBatch) -> "RiskBeliefModel":
        """Update beliefs with new evidence.

        Parameters
        ----------
        evidence : Evidence | EvidenceBatch
            Evidence to incorporate.

        Returns
        -------
        RiskBeliefModel
            Self for chaining.
        """
        if isinstance(evidence, EvidenceBatch):
            for e in evidence.evidences:
                self._update_single(e)
        else:
            self._update_single(evidence)

        return self

    def _update_single(self, evidence: Evidence) -> None:
        """Update with single evidence item.

        Parameters
        ----------
        evidence : Evidence
            Evidence to incorporate.
        """
        prior_mean = self._state.rug_probability.mean

        # Route evidence to appropriate risk factor
        if evidence.evidence_type in (
            EvidenceType.CONCENTRATION_CHANGE,
            EvidenceType.HOLDER_DISTRIBUTION_SHIFT,
        ):
            # Update concentration risk
            new_conc = self._updater.update_beta(
                self._state.concentration_risk, evidence
            )
            self._state.concentration_risk = new_conc

            # Also update rug probability with lower weight
            scaled_evidence = Evidence(
                evidence_type=evidence.evidence_type,
                value=evidence.value,
                timestamp=evidence.timestamp,
                strength=evidence.strength * 0.5,
                direction=evidence.direction,
                source=evidence.source,
                metadata=evidence.metadata,
            )
            new_rug = self._updater.update_beta(
                self._state.rug_probability, scaled_evidence
            )
            self._state.rug_probability = new_rug

        elif evidence.evidence_type in (
            EvidenceType.LIQUIDITY_CHANGE,
            EvidenceType.VOLUME_ANOMALY,
        ):
            # Update liquidity risk
            new_liq = self._updater.update_beta(
                self._state.liquidity_risk, evidence
            )
            self._state.liquidity_risk = new_liq

            # Also update rug probability
            scaled_evidence = Evidence(
                evidence_type=evidence.evidence_type,
                value=evidence.value,
                timestamp=evidence.timestamp,
                strength=evidence.strength * 0.4,
                direction=evidence.direction,
                source=evidence.source,
                metadata=evidence.metadata,
            )
            new_rug = self._updater.update_beta(
                self._state.rug_probability, scaled_evidence
            )
            self._state.rug_probability = new_rug

        elif evidence.evidence_type in (
            EvidenceType.COORDINATED_ACTIVITY,
            EvidenceType.DUMP_SIGNATURE,
        ):
            # Update coordination risk
            new_coord = self._updater.update_beta(
                self._state.coordination_risk, evidence
            )
            self._state.coordination_risk = new_coord

            # Update rug probability with high weight
            scaled_evidence = Evidence(
                evidence_type=evidence.evidence_type,
                value=evidence.value,
                timestamp=evidence.timestamp,
                strength=evidence.strength * 0.8,
                direction=evidence.direction,
                source=evidence.source,
                metadata=evidence.metadata,
            )
            new_rug = self._updater.update_beta(
                self._state.rug_probability, scaled_evidence
            )
            self._state.rug_probability = new_rug

        else:
            # General evidence - update rug probability directly
            new_rug = self._updater.update_beta(
                self._state.rug_probability, evidence
            )
            self._state.rug_probability = new_rug

        # Compute information gain
        info_gain = self._updater.compute_information_gain(
            BetaPrior(alpha=self.alpha, beta_=self.beta), evidence
        )

        # Record update
        update_record = BeliefUpdate(
            evidence=evidence,
            prior_mean=prior_mean,
            posterior_mean=self._state.rug_probability.mean,
            information_gain=info_gain,
            timestamp=datetime.now(timezone.utc),
        )
        self._state.update_history.append(update_record)
        self._state.last_updated = datetime.now(timezone.utc)

        logger.debug(
            "belief_updated",
            evidence_type=evidence.evidence_type.value,
            prior_mean=prior_mean,
            posterior_mean=self._state.rug_probability.mean,
            info_gain=info_gain,
        )

    def posterior_rug_probability(
        self, confidence: float = 0.95
    ) -> RiskEstimate:
        """Get posterior estimate of rug probability.

        Parameters
        ----------
        confidence : float
            Credible interval confidence level.

        Returns
        -------
        RiskEstimate
            Rug probability estimate with uncertainty.
        """
        dist = self._state.rug_probability
        lower, upper = dist.credible_interval(confidence)

        return RiskEstimate(
            mean=dist.mean,
            lower_ci=lower,
            upper_ci=upper,
            std=dist.std,
            confidence_level=confidence,
        )

    def credible_interval(
        self, alpha: float = 0.95
    ) -> tuple[float, float]:
        """Get credible interval for rug probability.

        Parameters
        ----------
        alpha : float
            Credible level (e.g., 0.95 for 95% CI).

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds.
        """
        return self._state.rug_probability.credible_interval(alpha)

    def information_gain(self, new_evidence: Evidence) -> float:
        """Calculate expected information gain from evidence.

        Parameters
        ----------
        new_evidence : Evidence
            Hypothetical evidence.

        Returns
        -------
        float
            Expected information gain in nats.
        """
        return self._updater.compute_information_gain(
            self._state.rug_probability, new_evidence
        )

    def composite_risk_score(self, confidence: float = 0.95) -> RiskEstimate:
        """Compute composite risk score combining all factors.

        Parameters
        ----------
        confidence : float
            Credible interval confidence level.

        Returns
        -------
        RiskEstimate
            Composite risk estimate.
        """
        # Sample from all distributions
        n_samples = 10000
        rng = np.random.default_rng(42)

        samples: dict[str, npt.NDArray[np.float64]] = {
            "rug_probability": self._state.rug_probability.sample(n_samples, 42),
            "concentration_risk": self._state.concentration_risk.sample(n_samples, 43),
            "liquidity_risk": self._state.liquidity_risk.sample(n_samples, 44),
            "coordination_risk": self._state.coordination_risk.sample(n_samples, 45),
        }

        # Weighted combination
        composite = np.zeros(n_samples)
        for name, weight in self.RISK_WEIGHTS.items():
            composite += weight * samples[name]

        # Compute statistics
        mean = float(np.mean(composite))
        std = float(np.std(composite))
        tail = (1 - confidence) / 2
        lower = float(np.percentile(composite, tail * 100))
        upper = float(np.percentile(composite, (1 - tail) * 100))

        return RiskEstimate(
            mean=mean,
            lower_ci=lower,
            upper_ci=upper,
            std=std,
            confidence_level=confidence,
        )

    def risk_decomposition(self) -> dict[str, RiskEstimate]:
        """Decompose risk into individual factors.

        Returns
        -------
        dict[str, RiskEstimate]
            Risk estimate for each factor.
        """
        return {
            "rug_probability": self.posterior_rug_probability(),
            "concentration_risk": self._make_estimate(self._state.concentration_risk),
            "liquidity_risk": self._make_estimate(self._state.liquidity_risk),
            "coordination_risk": self._make_estimate(self._state.coordination_risk),
        }

    def _make_estimate(self, dist: BetaPrior, confidence: float = 0.95) -> RiskEstimate:
        """Create risk estimate from distribution.

        Parameters
        ----------
        dist : BetaPrior
            Distribution to convert.
        confidence : float
            Credible interval level.

        Returns
        -------
        RiskEstimate
            Risk estimate.
        """
        lower, upper = dist.credible_interval(confidence)
        return RiskEstimate(
            mean=dist.mean,
            lower_ci=lower,
            upper_ci=upper,
            std=dist.std,
            confidence_level=confidence,
        )

    def uncertainty_level(self) -> str:
        """Categorize current uncertainty level.

        Returns
        -------
        str
            Uncertainty category: "low", "moderate", "high", "very_high".
        """
        estimate = self.posterior_rug_probability()

        if estimate.width < 0.15:
            return "low"
        elif estimate.width < 0.30:
            return "moderate"
        elif estimate.width < 0.50:
            return "high"
        else:
            return "very_high"

    def suggest_evidence(
        self, available_types: Sequence[EvidenceType] | None = None
    ) -> EvidenceType:
        """Suggest most valuable evidence type to gather.

        Parameters
        ----------
        available_types : Sequence[EvidenceType] | None
            Available evidence types. Defaults to all types.

        Returns
        -------
        EvidenceType
            Recommended evidence type.
        """
        if available_types is None:
            available_types = list(EvidenceType)

        return self._updater.suggest_next_evidence(
            self._state.rug_probability, available_types
        )

    def sample_risk(
        self, size: int = 1, random_state: int | None = None
    ) -> npt.NDArray[np.float64]:
        """Draw samples from posterior risk distribution.

        Parameters
        ----------
        size : int
            Number of samples.
        random_state : int | None
            Random seed.

        Returns
        -------
        npt.NDArray[np.float64]
            Risk samples.
        """
        return self._state.rug_probability.sample(size, random_state)

    def reset(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Reset model to initial priors.

        Parameters
        ----------
        prior_alpha : float
            Alpha for rug probability prior.
        prior_beta : float
            Beta for rug probability prior.
        """
        self._state = RiskBeliefState(
            rug_probability=BetaPrior(alpha=prior_alpha, beta_=prior_beta),
            concentration_risk=BetaPrior.from_mean_concentration(0.5, 4),
            liquidity_risk=BetaPrior.from_mean_concentration(0.4, 4),
            coordination_risk=BetaPrior.from_mean_concentration(0.2, 5),
        )

        logger.info(
            "risk_belief_model_reset",
            prior_mean=self._state.rug_probability.mean,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize model state.

        Returns
        -------
        dict[str, Any]
            Serialized state.
        """
        return {
            "state": self._state.to_dict(),
            "rug_estimate": self.posterior_rug_probability().to_dict(),
            "composite_risk": self.composite_risk_score().to_dict(),
            "uncertainty_level": self.uncertainty_level(),
        }

    @classmethod
    def from_historical_rate(
        cls,
        historical_rug_rate: float,
        sample_size: int,
        updater_config: UpdateConfig | None = None,
    ) -> "RiskBeliefModel":
        """Create model from historical rug rate.

        Parameters
        ----------
        historical_rug_rate : float
            Observed historical rug rate (0 to 1).
        sample_size : int
            Number of historical observations.
        updater_config : UpdateConfig | None
            Updater configuration.

        Returns
        -------
        RiskBeliefModel
            Model initialized with historical data.
        """
        # Convert to pseudo-counts
        rugs = int(historical_rug_rate * sample_size)
        non_rugs = sample_size - rugs

        return cls(
            prior_alpha=rugs + 1,
            prior_beta=non_rugs + 1,
            updater_config=updater_config,
        )
