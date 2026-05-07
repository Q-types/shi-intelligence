"""
Holder Regime Detection using Hidden Markov Models.

Implements HMM-based regime detection for holder structure evolution.

Regime States:
- ACCUMULATION: HHI/Gini decreasing, new holders joining
- DISTRIBUTION: HHI/Gini increasing, wallets consolidating
- COORDINATED_ACCUMULATION: HHI/Gini increasing with coordination signals
- DECAY: High churn, holders leaving
- STABLE: Low velocity, stable structure

NOTE: This is distinct from the existing MarketRegime (volatility-based).
This is holder structure regime detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Sequence

import numpy as np
from hmmlearn import hmm
import structlog

from .trajectories import MetricTrajectory, TrendDirection

logger = structlog.get_logger()


class HolderRegimeType(Enum):
    """Holder structure regime classification."""

    ACCUMULATION = "accumulation"  # Decentralizing, new holders
    DISTRIBUTION = "distribution"  # Centralizing, consolidation
    COORDINATED_ACCUMULATION = "coordinated_accumulation"  # Centralization with coordination
    DECAY = "decay"  # High churn, holders exiting
    STABLE = "stable"  # Low change, stable


@dataclass
class RegimeState:
    """Current holder regime state."""

    regime: HolderRegimeType
    confidence: float  # 0-1
    transition_probability: float  # P(switch to different regime)
    timestamp: datetime

    # Supporting evidence
    dhhi_dt: float
    dgini_dt: float
    dchurn_dt: Optional[float]


@dataclass
class RegimeTransition:
    """Detected regime transition event."""

    from_regime: HolderRegimeType
    to_regime: HolderRegimeType
    timestamp: datetime
    confidence: float


class HolderRegimeDetector:
    """
    HMM-based holder regime detector.

    Uses Hidden Markov Model to classify holder structure evolution
    into discrete regime states based on metric derivatives and trends.
    """

    # Number of hidden states
    N_STATES = 5

    # State to regime mapping (learned from training, but initialized with priors)
    STATE_TO_REGIME = {
        0: HolderRegimeType.STABLE,
        1: HolderRegimeType.ACCUMULATION,
        2: HolderRegimeType.DISTRIBUTION,
        3: HolderRegimeType.COORDINATED_ACCUMULATION,
        4: HolderRegimeType.DECAY,
    }

    def __init__(
        self,
        n_iter: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_iter: Number of EM iterations for HMM training
            random_state: Random seed for reproducibility
        """
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self._is_fitted = False
        self._current_regime: Optional[HolderRegimeType] = None
        self._regime_history: list[tuple[datetime, HolderRegimeType]] = []

    def fit(
        self,
        training_sequences: Sequence[np.ndarray],
        lengths: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Fit HMM on training data.

        Args:
            training_sequences: Feature sequences for training
                Shape: (n_samples, n_features)
                Features: [dhhi_dt, dgini_dt, dchurn_dt, coordination_signal, ...]
            lengths: Lengths of individual sequences (for multiple tokens)
        """
        if not training_sequences:
            raise ValueError("Cannot fit HMM on empty training data")

        # Concatenate sequences
        X = np.vstack(training_sequences)

        # Initialize HMM with Gaussian emissions
        self.model = hmm.GaussianHMM(
            n_components=self.N_STATES,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        # Initialize with reasonable priors
        self._initialize_priors()

        # Fit model
        logger.info("fitting_hmm", n_samples=len(X), n_features=X.shape[1])

        if lengths:
            self.model.fit(X, lengths=lengths)
        else:
            self.model.fit(X)

        self._is_fitted = True

        logger.info(
            "hmm_fitted",
            converged=self.model.monitor_.converged,
            n_iter=self.model.monitor_.iter,
        )

    def _initialize_priors(self) -> None:
        """Initialize HMM with domain-knowledge priors."""
        if not self.model:
            return

        # Initial state distribution (start in stable)
        startprob = np.zeros(self.N_STATES)
        startprob[0] = 0.7  # Stable
        startprob[1:] = 0.075  # Others
        self.model.startprob_ = startprob

        # Transition matrix (diagonal dominance = persistence)
        transmat = np.zeros((self.N_STATES, self.N_STATES))
        for i in range(self.N_STATES):
            transmat[i, i] = 0.7  # Stay in state
            transmat[i, :] += 0.3 / (self.N_STATES - 1)  # Distribute to others
            transmat[i, i] = 0.7  # Reset diagonal

        self.model.transmat_ = transmat

    def predict_regime(
        self,
        features: np.ndarray,
    ) -> RegimeState:
        """
        Predict regime from feature vector.

        Args:
            features: Feature vector [dhhi_dt, dgini_dt, dchurn_dt, coordination, ...]
                Shape: (n_timesteps, n_features)

        Returns:
            RegimeState with classification and confidence
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Predict state sequence
        states = self.model.predict(features)

        # Get most recent state
        current_state = states[-1]
        regime = self.STATE_TO_REGIME.get(current_state, HolderRegimeType.STABLE)

        # Compute posterior probabilities for confidence
        posteriors = self.model.predict_proba(features)
        confidence = float(posteriors[-1, current_state])

        # Transition probability (1 - P(stay in current state))
        transition_prob = float(1 - self.model.transmat_[current_state, current_state])

        # Extract feature values
        last_features = features[-1, :]
        dhhi_dt = float(last_features[0]) if len(last_features) > 0 else 0.0
        dgini_dt = float(last_features[1]) if len(last_features) > 1 else 0.0
        dchurn_dt = float(last_features[2]) if len(last_features) > 2 else None

        return RegimeState(
            regime=regime,
            confidence=confidence,
            transition_probability=transition_prob,
            timestamp=datetime.utcnow(),
            dhhi_dt=dhhi_dt,
            dgini_dt=dgini_dt,
            dchurn_dt=dchurn_dt,
        )

    def detect_transitions(
        self,
        features: np.ndarray,
        timestamps: Sequence[datetime],
    ) -> list[RegimeTransition]:
        """
        Detect regime transitions in a sequence.

        Args:
            features: Feature sequence (n_timesteps, n_features)
            timestamps: Corresponding timestamps

        Returns:
            List of detected transitions
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Predict state sequence
        states = self.model.predict(features)
        posteriors = self.model.predict_proba(features)

        # Find transitions
        transitions = []
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                from_regime = self.STATE_TO_REGIME.get(states[i - 1], HolderRegimeType.STABLE)
                to_regime = self.STATE_TO_REGIME.get(states[i], HolderRegimeType.STABLE)

                transitions.append(
                    RegimeTransition(
                        from_regime=from_regime,
                        to_regime=to_regime,
                        timestamp=timestamps[i],
                        confidence=float(posteriors[i, states[i]]),
                    )
                )

        return transitions

    def extract_features_from_trajectories(
        self,
        hhi_traj: MetricTrajectory,
        gini_traj: MetricTrajectory,
        churn_traj: Optional[MetricTrajectory] = None,
        coordination_score: float = 0.0,
    ) -> np.ndarray:
        """
        Extract HMM features from metric trajectories.

        Args:
            hhi_traj: HHI trajectory
            gini_traj: Gini trajectory
            churn_traj: Optional churn trajectory
            coordination_score: Coordination score

        Returns:
            Feature vector for HMM
        """
        features = [
            hhi_traj.velocity,  # dHHI/dt
            gini_traj.velocity,  # dGini/dt
        ]

        if churn_traj:
            features.append(churn_traj.velocity)  # dChurn/dt
        else:
            features.append(0.0)

        features.append(coordination_score)

        # Add trend indicators (one-hot encoded)
        features.append(1.0 if hhi_traj.trend == TrendDirection.CENTRALIZING else 0.0)
        features.append(1.0 if hhi_traj.trend == TrendDirection.DECENTRALIZING else 0.0)

        return np.array(features)

    def update_history(
        self,
        timestamp: datetime,
        regime: HolderRegimeType,
    ) -> None:
        """Update regime history for tracking."""
        self._regime_history.append((timestamp, regime))
        self._current_regime = regime

    def get_regime_duration(self) -> Optional[float]:
        """
        Get duration of current regime in days.

        Returns:
            Days in current regime, or None if no history
        """
        if not self._regime_history or not self._current_regime:
            return None

        # Find last transition
        current = self._current_regime
        for i in range(len(self._regime_history) - 1, -1, -1):
            if self._regime_history[i][1] != current:
                # Found transition point
                transition_time = self._regime_history[i][0]
                current_time = self._regime_history[-1][0]
                return (current_time - transition_time).total_seconds() / 86400

        # No transition found, been in this regime since start
        if len(self._regime_history) > 1:
            start_time = self._regime_history[0][0]
            current_time = self._regime_history[-1][0]
            return (current_time - start_time).total_seconds() / 86400

        return None


def create_rule_based_regime(
    dhhi_dt: float,
    dgini_dt: float,
    dchurn_dt: Optional[float],
    coordination_score: float,
    hhi_trend: TrendDirection,
) -> HolderRegimeType:
    """
    Rule-based regime classification (fallback when HMM not fitted).

    Args:
        dhhi_dt: HHI velocity
        dgini_dt: Gini velocity
        dchurn_dt: Churn velocity
        coordination_score: Coordination score
        hhi_trend: HHI trend direction

    Returns:
        HolderRegimeType based on rules
    """
    # Decay: High churn
    if dchurn_dt and dchurn_dt > 0.1:
        return HolderRegimeType.DECAY

    # Coordinated accumulation: Centralizing with high coordination
    if hhi_trend == TrendDirection.CENTRALIZING and coordination_score > 0.7:
        return HolderRegimeType.COORDINATED_ACCUMULATION

    # Distribution: Strong centralization
    if hhi_trend == TrendDirection.CENTRALIZING and dhhi_dt > 0.01:
        return HolderRegimeType.DISTRIBUTION

    # Accumulation: Decentralizing
    if hhi_trend == TrendDirection.DECENTRALIZING and dhhi_dt < -0.01:
        return HolderRegimeType.ACCUMULATION

    # Default: Stable
    return HolderRegimeType.STABLE
