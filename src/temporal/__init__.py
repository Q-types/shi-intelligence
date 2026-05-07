"""
Temporal Analysis Module (SHI v2).

Transform static snapshot analysis → dynamical intelligence system.

Modules:
- trajectories: Metric time-series tracking (HHI(t), Gini(t), derivatives)
- regimes: HMM-based holder regime detection
- forecasting: Capital flow prediction
"""

from .trajectories import TrajectoryTracker, MetricTrajectory, TrendDirection
from .regimes import HolderRegimeDetector, HolderRegimeType, RegimeTransition

__all__ = [
    "TrajectoryTracker",
    "MetricTrajectory",
    "TrendDirection",
    "HolderRegimeDetector",
    "HolderRegimeType",
    "RegimeTransition",
]
