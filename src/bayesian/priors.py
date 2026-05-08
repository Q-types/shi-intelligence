"""Prior distributions for Bayesian risk estimation.

This module provides prior distribution classes that can be used
as the initial beliefs in Bayesian risk models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats
import structlog

logger = structlog.get_logger()


class PriorDistribution(ABC):
    """Base class for prior distributions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Distribution name."""
        ...

    @property
    @abstractmethod
    def mean(self) -> float:
        """Distribution mean."""
        ...

    @property
    @abstractmethod
    def variance(self) -> float:
        """Distribution variance."""
        ...

    @property
    def std(self) -> float:
        """Distribution standard deviation."""
        return np.sqrt(self.variance)

    @abstractmethod
    def pdf(self, x: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Probability density function.

        Parameters
        ----------
        x : float | npt.NDArray[np.float64]
            Point(s) at which to evaluate PDF.

        Returns
        -------
        float | npt.NDArray[np.float64]
            PDF value(s).
        """
        ...

    @abstractmethod
    def cdf(self, x: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Cumulative distribution function.

        Parameters
        ----------
        x : float | npt.NDArray[np.float64]
            Point(s) at which to evaluate CDF.

        Returns
        -------
        float | npt.NDArray[np.float64]
            CDF value(s).
        """
        ...

    @abstractmethod
    def ppf(self, q: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Percent point function (inverse CDF).

        Parameters
        ----------
        q : float | npt.NDArray[np.float64]
            Quantile(s) to compute.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Value(s) at quantile(s).
        """
        ...

    @abstractmethod
    def sample(self, size: int = 1, random_state: int | None = None) -> npt.NDArray[np.float64]:
        """Draw random samples from distribution.

        Parameters
        ----------
        size : int
            Number of samples to draw.
        random_state : int | None
            Random seed for reproducibility.

        Returns
        -------
        npt.NDArray[np.float64]
            Array of samples.
        """
        ...

    def credible_interval(self, alpha: float = 0.95) -> tuple[float, float]:
        """Compute credible interval.

        Parameters
        ----------
        alpha : float
            Credible level (e.g., 0.95 for 95% CI).

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds of credible interval.
        """
        tail = (1 - alpha) / 2
        lower = float(self.ppf(tail))
        upper = float(self.ppf(1 - tail))
        return (lower, upper)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize distribution parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary of distribution parameters.
        """
        ...


@dataclass
class BetaPrior(PriorDistribution):
    """Beta distribution prior for probability parameters.

    The Beta distribution is the conjugate prior for Bernoulli/Binomial
    likelihoods, making it ideal for modeling P(rug pull).

    Parameters
    ----------
    alpha : float
        Shape parameter alpha (pseudo-counts of successes + 1).
    beta_ : float
        Shape parameter beta (pseudo-counts of failures + 1).

    Examples
    --------
    >>> prior = BetaPrior(alpha=2, beta_=5)  # Weakly skeptical prior
    >>> prior.mean  # E[p] = alpha / (alpha + beta)
    0.2857...
    """

    alpha: float = 1.0
    beta_: float = 1.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.beta_ <= 0:
            raise ValueError(f"beta must be positive, got {self.beta_}")
        self._dist = stats.beta(self.alpha, self.beta_)

    @property
    def name(self) -> str:
        """Distribution name."""
        return f"Beta({self.alpha:.2f}, {self.beta_:.2f})"

    @property
    def mean(self) -> float:
        """Expected value E[X] = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta_)

    @property
    def variance(self) -> float:
        """Variance Var[X] = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))."""
        ab = self.alpha + self.beta_
        return (self.alpha * self.beta_) / (ab**2 * (ab + 1))

    @property
    def mode(self) -> float | None:
        """Mode of distribution (if exists).

        Returns
        -------
        float | None
            Mode if alpha > 1 and beta > 1, else None.
        """
        if self.alpha > 1 and self.beta_ > 1:
            return (self.alpha - 1) / (self.alpha + self.beta_ - 2)
        return None

    @property
    def concentration(self) -> float:
        """Concentration parameter (alpha + beta).

        Higher concentration = more confident prior.
        """
        return self.alpha + self.beta_

    def pdf(self, x: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Beta PDF."""
        return self._dist.pdf(x)

    def cdf(self, x: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Beta CDF."""
        return self._dist.cdf(x)

    def ppf(self, q: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Beta quantile function."""
        return self._dist.ppf(q)

    def sample(self, size: int = 1, random_state: int | None = None) -> npt.NDArray[np.float64]:
        """Sample from Beta distribution."""
        rng = np.random.default_rng(random_state)
        return rng.beta(self.alpha, self.beta_, size=size)

    def update(self, successes: int, failures: int) -> "BetaPrior":
        """Return updated posterior after observing data.

        Parameters
        ----------
        successes : int
            Number of "success" observations (e.g., rugs).
        failures : int
            Number of "failure" observations (e.g., non-rugs).

        Returns
        -------
        BetaPrior
            Posterior distribution with updated parameters.
        """
        return BetaPrior(
            alpha=self.alpha + successes,
            beta_=self.beta_ + failures,
        )

    def update_continuous(
        self, evidence_strength: float, direction: float
    ) -> "BetaPrior":
        """Update with continuous evidence.

        Parameters
        ----------
        evidence_strength : float
            Strength of evidence (0 to 10, typical).
        direction : float
            Direction of evidence (-1 for safe, +1 for risky).

        Returns
        -------
        BetaPrior
            Updated distribution.
        """
        # Clamp direction
        direction = max(-1, min(1, direction))

        # Split evidence based on direction
        if direction >= 0:
            alpha_delta = evidence_strength * direction
            beta_delta = evidence_strength * (1 - direction) * 0.1
        else:
            alpha_delta = evidence_strength * (1 + direction) * 0.1
            beta_delta = evidence_strength * abs(direction)

        return BetaPrior(
            alpha=self.alpha + alpha_delta,
            beta_=self.beta_ + beta_delta,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "beta",
            "alpha": self.alpha,
            "beta": self.beta_,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_mean_concentration(
        cls, mean: float, concentration: float
    ) -> "BetaPrior":
        """Create Beta prior from mean and concentration.

        Parameters
        ----------
        mean : float
            Desired mean (between 0 and 1).
        concentration : float
            Concentration (alpha + beta). Higher = more confident.

        Returns
        -------
        BetaPrior
            Beta distribution with specified properties.
        """
        if not 0 < mean < 1:
            raise ValueError(f"Mean must be in (0, 1), got {mean}")
        if concentration <= 0:
            raise ValueError(f"Concentration must be positive, got {concentration}")

        alpha = mean * concentration
        beta = (1 - mean) * concentration
        return cls(alpha=alpha, beta_=beta)


@dataclass
class GammaPrior(PriorDistribution):
    """Gamma distribution prior for rate/scale parameters.

    Useful for modeling hazard rates or time scales.

    Parameters
    ----------
    shape : float
        Shape parameter (k or alpha).
    rate : float
        Rate parameter (1/scale, or beta).
    """

    shape: float = 1.0
    rate: float = 1.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.shape <= 0:
            raise ValueError(f"shape must be positive, got {self.shape}")
        if self.rate <= 0:
            raise ValueError(f"rate must be positive, got {self.rate}")
        self._dist = stats.gamma(a=self.shape, scale=1/self.rate)

    @property
    def name(self) -> str:
        """Distribution name."""
        return f"Gamma({self.shape:.2f}, {self.rate:.2f})"

    @property
    def mean(self) -> float:
        """Expected value E[X] = shape / rate."""
        return self.shape / self.rate

    @property
    def variance(self) -> float:
        """Variance Var[X] = shape / rate^2."""
        return self.shape / (self.rate**2)

    @property
    def scale(self) -> float:
        """Scale parameter (1/rate)."""
        return 1 / self.rate

    def pdf(self, x: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Gamma PDF."""
        return self._dist.pdf(x)

    def cdf(self, x: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Gamma CDF."""
        return self._dist.cdf(x)

    def ppf(self, q: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Gamma quantile function."""
        return self._dist.ppf(q)

    def sample(self, size: int = 1, random_state: int | None = None) -> npt.NDArray[np.float64]:
        """Sample from Gamma distribution."""
        rng = np.random.default_rng(random_state)
        return rng.gamma(self.shape, self.scale, size=size)

    def update(self, observations: npt.NDArray[np.float64]) -> "GammaPrior":
        """Update with new observations (Gamma-Exponential conjugacy).

        Parameters
        ----------
        observations : npt.NDArray[np.float64]
            Observed values (e.g., inter-event times).

        Returns
        -------
        GammaPrior
            Posterior distribution.
        """
        n = len(observations)
        total = float(np.sum(observations))
        return GammaPrior(
            shape=self.shape + n,
            rate=self.rate + total,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "gamma",
            "shape": self.shape,
            "rate": self.rate,
            "mean": self.mean,
            "std": self.std,
        }


@dataclass
class NormalPrior(PriorDistribution):
    """Normal (Gaussian) distribution prior.

    Useful for modeling unbounded continuous parameters.

    Parameters
    ----------
    mu : float
        Mean of distribution.
    sigma : float
        Standard deviation.
    """

    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        self._dist = stats.norm(loc=self.mu, scale=self.sigma)

    @property
    def name(self) -> str:
        """Distribution name."""
        return f"Normal({self.mu:.2f}, {self.sigma:.2f})"

    @property
    def mean(self) -> float:
        """Expected value E[X] = mu."""
        return self.mu

    @property
    def variance(self) -> float:
        """Variance Var[X] = sigma^2."""
        return self.sigma**2

    def pdf(self, x: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Normal PDF."""
        return self._dist.pdf(x)

    def cdf(self, x: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Normal CDF."""
        return self._dist.cdf(x)

    def ppf(self, q: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Normal quantile function."""
        return self._dist.ppf(q)

    def sample(self, size: int = 1, random_state: int | None = None) -> npt.NDArray[np.float64]:
        """Sample from Normal distribution."""
        rng = np.random.default_rng(random_state)
        return rng.normal(self.mu, self.sigma, size=size)

    def update(
        self, observations: npt.NDArray[np.float64], known_variance: float | None = None
    ) -> "NormalPrior":
        """Update with new observations (Normal-Normal conjugacy).

        Parameters
        ----------
        observations : npt.NDArray[np.float64]
            Observed values.
        known_variance : float | None
            Known population variance. If None, uses sample variance.

        Returns
        -------
        NormalPrior
            Posterior distribution for mean.
        """
        n = len(observations)
        x_bar = float(np.mean(observations))

        if known_variance is not None:
            sigma2 = known_variance
        else:
            sigma2 = float(np.var(observations, ddof=1)) if n > 1 else self.sigma**2

        # Prior precision
        tau_0 = 1 / self.sigma**2
        # Data precision (per observation)
        tau_data = n / sigma2

        # Posterior precision
        tau_post = tau_0 + tau_data
        sigma_post = np.sqrt(1 / tau_post)

        # Posterior mean
        mu_post = (tau_0 * self.mu + tau_data * x_bar) / tau_post

        return NormalPrior(mu=mu_post, sigma=sigma_post)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "normal",
            "mu": self.mu,
            "sigma": self.sigma,
            "mean": self.mean,
            "std": self.std,
        }


def create_default_priors() -> dict[str, PriorDistribution]:
    """Create default prior distributions for risk model.

    Returns
    -------
    dict[str, PriorDistribution]
        Dictionary mapping risk factor names to their priors.
    """
    return {
        # P(rug) - slightly skeptical, centered around 0.3
        "rug_probability": BetaPrior.from_mean_concentration(mean=0.3, concentration=5),
        # Holder concentration risk - higher concentration = higher risk
        "concentration_risk": BetaPrior.from_mean_concentration(mean=0.5, concentration=4),
        # Liquidity risk - starts moderate
        "liquidity_risk": BetaPrior.from_mean_concentration(mean=0.4, concentration=4),
        # Time to potential event (days) - exponential-like
        "time_to_event": GammaPrior(shape=2, rate=0.1),
        # Coordination score among holders
        "coordination_score": BetaPrior.from_mean_concentration(mean=0.2, concentration=5),
    }
