"""
Statistical Models for SHI.

Implements:
- Cox Proportional Hazards for sell probability
- Model training and validation
- Diagnostics (Schoenfeld residuals)
- Sell event detection
- Cluster correlation adjustment
- Regime detection
"""

from .hazard_model import (
    HazardModelTrainer,
    HazardModelPredictor,
    TrainedHazardModel,
    validate_proportional_hazards,
)
from .sell_events import (
    SellEvent,
    SellEventDetector,
    BalanceHistory,
)
from .training import (
    HazardModelTrainingPipeline,
    TrainingConfig,
    TrainingResult,
)
from .correlation import (
    ClusterCorrelationAdjuster,
    ClusterSellProbability,
    CorrelationAdjustedPressure,
)
from .validation import (
    ModelValidator,
    ValidationReport,
    ValidationThresholds,
)
from .regime import (
    RegimeDetector,
    RegimeState,
    MarketRegime,
    RegimeAwareRetrainer,
)

__all__ = [
    # Hazard model
    "HazardModelTrainer",
    "HazardModelPredictor",
    "TrainedHazardModel",
    "validate_proportional_hazards",
    # Sell events
    "SellEvent",
    "SellEventDetector",
    "BalanceHistory",
    # Training
    "HazardModelTrainingPipeline",
    "TrainingConfig",
    "TrainingResult",
    # Correlation
    "ClusterCorrelationAdjuster",
    "ClusterSellProbability",
    "CorrelationAdjustedPressure",
    # Validation
    "ModelValidator",
    "ValidationReport",
    "ValidationThresholds",
    # Regime
    "RegimeDetector",
    "RegimeState",
    "MarketRegime",
    "RegimeAwareRetrainer",
]
