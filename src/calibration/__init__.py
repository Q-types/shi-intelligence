"""
Model Calibration Tools.

Implements:
- Continuous Brier score tracking
- Calibration curve generation
- Prediction interval validation
- Drift detection and alerting
- Model comparison utilities
"""

from .tracking import (
    CalibrationTracker,
    BrierScoreTracker,
    PredictionRecord,
)
from .drift import (
    DriftDetector,
    DriftAlert,
    DriftSeverity,
)
from .comparison import (
    ModelComparator,
    ComparisonResult,
)

__all__ = [
    # Tracking
    "CalibrationTracker",
    "BrierScoreTracker",
    "PredictionRecord",
    # Drift
    "DriftDetector",
    "DriftAlert",
    "DriftSeverity",
    # Comparison
    "ModelComparator",
    "ComparisonResult",
]
