"""
Immutable Metrics Engine for SHI.

WARNING: THE METRICS IN THIS MODULE ARE FROZEN.
NO AGENT MAY MODIFY FORMULAS OR DEFINITIONS WITHOUT EXPLICIT HUMAN APPROVAL.

All metrics are defined exactly as specified in the PDR.
"""

from .distribution import (
    compute_hhi,
    compute_shannon_entropy,
    compute_gini_coefficient,
    compute_whale_dominance_ratio,
)
from .coordination import (
    compute_coordination_score,
    compute_funding_density,
    compute_churn_rate,
)
from .hazard import (
    compute_sell_probability,
    compute_sell_pressure_index,
)
from .normalization import (
    compute_z_score,
    compute_percentile,
)

__all__ = [
    "compute_hhi",
    "compute_shannon_entropy",
    "compute_gini_coefficient",
    "compute_whale_dominance_ratio",
    "compute_coordination_score",
    "compute_funding_density",
    "compute_churn_rate",
    "compute_sell_probability",
    "compute_sell_pressure_index",
    "compute_z_score",
    "compute_percentile",
]

# Metrics version - increment only with human approval
METRICS_VERSION = "1.0.0"
