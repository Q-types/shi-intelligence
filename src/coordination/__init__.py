"""
Multi-Evidence Coordination Detection.

This module implements a redesigned coordination detection system that requires
multiple independent evidence types to classify wallets as coordinated.

CRITICAL: The previous temporal-only coordination detector FAILED null model
validation (2026-05-21) with 0 significant detections. Timing alone cannot
distinguish true coordination from the natural timing compression of token launches.

This module replaces temporal-only detection with multi-evidence scoring:
- Shared funder similarity
- Funding time similarity
- Funding amount similarity
- First buy time similarity
- Trade sequence similarity
- Exit timing similarity
- Cross-token reuse

All classifications require:
1. z-score >= 2.5 (configurable)
2. p-value <= 0.01 (configurable)
3. Minimum 3 evidence types present (configurable)
4. Minimum cluster size (configurable)
"""

from .features import (
    CoordinationFeatures,
    FundingSimilarityFeatures,
    TradingSimilarityFeatures,
    BehavioralSimilarityFeatures,
    CrossTokenSimilarityFeatures,
    compute_pairwise_coordination_features,
)
from .blocking import (
    CandidateBlock,
    BlockingStrategy,
    create_candidate_blocks,
)
from .scoring import (
    MultiEvidenceCoordinationScore,
    CoordinationClassification,
    compute_coordination_score,
    classify_coordination,
)
from .null_model import (
    NullModelResult,
    NullModelValidation,
    run_null_model_validation,
)
from .orchestrator import (
    CoordinationResult,
    MultiEvidenceCoordinationDetector,
)

__all__ = [
    # Features
    "CoordinationFeatures",
    "FundingSimilarityFeatures",
    "TradingSimilarityFeatures",
    "BehavioralSimilarityFeatures",
    "CrossTokenSimilarityFeatures",
    "compute_pairwise_coordination_features",
    # Blocking
    "CandidateBlock",
    "BlockingStrategy",
    "create_candidate_blocks",
    # Scoring
    "MultiEvidenceCoordinationScore",
    "CoordinationClassification",
    "compute_coordination_score",
    "classify_coordination",
    # Null Model
    "NullModelResult",
    "NullModelValidation",
    "run_null_model_validation",
    # Orchestrator
    "CoordinationResult",
    "MultiEvidenceCoordinationDetector",
]
