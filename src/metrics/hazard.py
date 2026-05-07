"""
Hazard Model Metrics - IMMUTABLE

WARNING: These formulas are frozen per PDR.
Do not modify without explicit human approval.

Formulas defined in PDR Sections 4.8-4.9.
Implementation uses Cox Proportional Hazards model.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.types import MetricOutput

# Version for hazard metrics
_VERSION = "1.0.0"


def compute_sell_probability(
    baseline_hazard_integral: float,
    feature_vector: NDArray[np.float64],
    beta_coefficients: NDArray[np.float64],
    horizon_days: int = 7,
) -> MetricOutput:
    """
    Sell Probability within Horizon T.

    FROZEN FORMULA (PDR 4.8):
        Hazard function: lambda(t | x) = lambda_0(t) * exp(beta^T x)

        Sell probability:
        P_sell(T) = 1 - exp( - Integral_0_to_T lambda(t | x) dt )

    Implementation:
        P_sell(T) = 1 - exp( -Lambda_0(T) * exp(beta^T x) )

        Where Lambda_0(T) = integral of baseline hazard from 0 to T

    Args:
        baseline_hazard_integral: Lambda_0(T) - cumulative baseline hazard at horizon T
        feature_vector: Wallet feature vector x
        beta_coefficients: Model coefficients beta (from Cox PH fit)
        horizon_days: Time horizon T in days

    Returns:
        MetricOutput with sell probability in [0, 1]

    Note:
        - Model must be fit using lifelines CoxPHFitter
        - Efron tie handling required
        - Schoenfeld residual diagnostics required before deployment
    """
    if baseline_hazard_integral < 0:
        raise ValueError("Baseline hazard integral cannot be negative")

    if len(feature_vector) != len(beta_coefficients):
        raise ValueError(
            f"Feature vector length {len(feature_vector)} != "
            f"coefficient length {len(beta_coefficients)}"
        )

    # Compute linear predictor
    linear_predictor = np.dot(beta_coefficients, feature_vector)

    # Compute hazard ratio
    hazard_ratio = math.exp(linear_predictor)

    # Compute cumulative hazard for this wallet
    cumulative_hazard = baseline_hazard_integral * hazard_ratio

    # Compute survival probability
    survival_prob = math.exp(-cumulative_hazard)

    # Sell probability = 1 - survival
    sell_prob = 1.0 - survival_prob

    # Clamp to valid range
    sell_prob = max(0.0, min(1.0, sell_prob))

    return MetricOutput(
        metric_name="sell_probability",
        value=sell_prob,
        version=_VERSION,
        computed_at=datetime.now(timezone.utc),
    )


def compute_sell_pressure_index(
    sell_probabilities: Sequence[float],
) -> MetricOutput:
    """
    Sell Pressure Index.

    FROZEN FORMULA (PDR 4.9):
        Sell_Pressure = SUM_{i in Top_N} P_sell_i(T)

    This is the aggregated predicted exit probability of top N wallets.

    Args:
        sell_probabilities: List of P_sell(T) for top N wallets

    Returns:
        MetricOutput with sell pressure index (sum of probabilities)

    Note:
        Input should be sell probabilities for TOP holders by balance,
        not all holders.
    """
    if not sell_probabilities:
        raise ValueError("Cannot compute sell pressure for empty list")

    # Validate all probabilities are in [0, 1]
    for i, p in enumerate(sell_probabilities):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability at index {i} out of range: {p}")

    sell_pressure = sum(sell_probabilities)

    return MetricOutput(
        metric_name="sell_pressure_index",
        value=sell_pressure,
        version=_VERSION,
        computed_at=datetime.now(timezone.utc),
    )


def compute_cluster_sell_probability(
    individual_sell_probs: Sequence[float],
) -> MetricOutput:
    """
    Cluster-Aware Sell Probability.

    FROZEN FORMULA (INITIAL_PROMPT - Correlation Adjustment):
        Cluster_P_sell = 1 - PRODUCT(1 - P_sell_i)

    This computes the probability that AT LEAST ONE wallet in the cluster sells.

    Args:
        individual_sell_probs: P_sell(T) for each wallet in the cluster

    Returns:
        MetricOutput with cluster sell probability in [0, 1]

    Note:
        This assumes conditional independence within the cluster.
        If coordination score exceeds threshold, apply correlation
        amplification factor separately.
    """
    if not individual_sell_probs:
        raise ValueError("Cannot compute cluster probability for empty list")

    # Validate all probabilities
    for i, p in enumerate(individual_sell_probs):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability at index {i} out of range: {p}")

    # Compute survival product
    survival_product = 1.0
    for p in individual_sell_probs:
        survival_product *= (1.0 - p)

    # At least one sells = 1 - all survive
    cluster_prob = 1.0 - survival_product

    return MetricOutput(
        metric_name="cluster_sell_probability",
        value=cluster_prob,
        version=_VERSION,
        computed_at=datetime.now(timezone.utc),
    )
