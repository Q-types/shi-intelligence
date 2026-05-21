"""
Longitudinal Intelligence Infrastructure.

Provides time-evolving behavioral intelligence capabilities:
- Snapshot Engine: Configurable collection with dynamic cadence
- Event Store: Event-sourced raw event storage
- Replay Engine: Deterministic state reconstruction
- Trajectory Features: Time-series behavioral metrics
- Cross-Token Memory: Wallet behavior across launches
- Event Corrections: Immutable corrections and invalidations
- Trajectory Smoothing: Noise-resistant velocity/acceleration
- Price Snapshots: Historical price observations (Sprint 8)
- PnL Calculator: Cost basis and realised PnL (Sprint 8)
- Exit Classifier: Transfer/sell classification (Sprint 9)

HARD RULES:
1. Preserve replay reproducibility
2. Never overwrite historical snapshots
3. All derived metrics must be recomputable from raw events
4. Replay engine must support deterministic reconstruction
5. Time is a first-class dimension
6. Events are immutable - corrections are new events
7. behaviour_history_score is NOT reputation (pending validation)
8. Price is NOT ground truth - always includes confidence (Sprint 8)
9. Missing price reduces confidence, never becomes zero (Sprint 8)
10. Separate realised from unrealised PnL (Sprint 8)
11. Balance decrease alone is NOT a sell (Sprint 9)
12. Realised PnL requires sell confidence (Sprint 9)
13. LP actions must NOT be treated as sells (Sprint 9)
14. Transfers must NOT generate realised PnL unless later sale observed (Sprint 9)
"""

from .models import (
    # Enums
    EventType,
    TradeType,
    LiquidityAction,
    StateTransitionType,
    AccountingMethod,
    PriceConfidenceLevel,
    # Event Store
    RawEvent,
    TradeEventRecord,
    LiquidityEventRecord,
    FundingEventRecord,
    StateTransitionRecord,
    # Snapshot Engine
    SnapshotConfig,
    TokenLaunchState,
    GraphSnapshot,
    CoordinationSnapshot,
    VolumeSnapshot,
    # Trajectory
    TrajectorySnapshot,
    # Price Snapshots (Sprint 8)
    PriceSnapshot,
    RealisedPnLRecord,
    CostBasisLot,
    # Cross-Token Memory Models
    WalletBehaviorHistory,
    TokenParticipationHistory,
    WalletTokenPositionHistory,
    CoParticipationEdge,
    BehavioralSimilarityCache,
)
from .snapshot_engine import SnapshotEngine, SnapshotCadence, SnapshotScheduler
from .event_store import (
    EventStore,
    EventReplayer,
    TradeEvent,
    LiquidityEvent,
    FundingEvent,
    StateTransition,
    EventBatch,
    EventQuery,
)
from .replay_engine import ReplayEngine, ReplayMode as ReplayEngineMode, TokenState
from .trajectory import TrajectoryComputer, TokenTrajectory, TrajectoryTrend
from .cross_token_memory import (
    CrossTokenMemoryService,
    WalletMetrics,
    CoParticipationMetrics,
    WalletArchetype,
)
from .trajectory_smoothing import (
    TrajectorySmoother,
    SmoothedTrajectoryComputer,
    SmoothedTrajectory,
    SmoothedValue,
    SmoothingMethod,
    TimeSeriesPoint,
)
from .event_ordering import (
    EventOrderer,
    EventOrderingKey,
    ReplayMode as EventReplayMode,
    ReplayConfig,
    CorrectionEventBuilder,
)
from .price_snapshots import (
    PriceSnapshotService,
    PriceSnapshotCollector,
    PriceObservation,
    HistoricalPriceQuery,
    HistoricalPriceResult,
)
from .pnl_calculator import (
    CostBasisCalculator,
    RealisedPnLCalculator,
    ProfitExtractionAnalyzer,
    TradeRecord,
    CostBasisEstimate,
    RealisedPnLEstimate,
)
from .exit_classifier import (
    ExitEventType,
    ExitEvidence,
    ExitEventClassification,
    ExitClassifierConfig,
    ExitEventClassifier,
    SellConfidenceScorer,
    PnLReliabilityScorer,
    TransferChainResult,
    TransferChainConfig,
    TransferChainDetector,
    WalletInfoProvider,
    LPActionResult,
    LPActionDetector,
    CEXDepositResult,
    CEXDetectionConfig,
    CEXDepositDetector,
    create_exit_classifier,
    create_transfer_chain_detector,
    create_lp_action_detector,
    create_cex_deposit_detector,
)

__all__ = [
    # Enums
    "EventType",
    "TradeType",
    "LiquidityAction",
    "StateTransitionType",
    "ReplayEngineMode",
    "EventReplayMode",
    "TrajectoryTrend",
    "SmoothingMethod",
    "WalletArchetype",
    # Event Store Models
    "RawEvent",
    "TradeEventRecord",
    "LiquidityEventRecord",
    "FundingEventRecord",
    "StateTransitionRecord",
    # Event Store Dataclasses
    "TradeEvent",
    "LiquidityEvent",
    "FundingEvent",
    "StateTransition",
    "EventBatch",
    "EventQuery",
    # Snapshot Models
    "SnapshotConfig",
    "TokenLaunchState",
    "GraphSnapshot",
    "CoordinationSnapshot",
    "VolumeSnapshot",
    "TrajectorySnapshot",
    # Cross-Token Memory Models
    "WalletBehaviorHistory",
    "TokenParticipationHistory",
    "WalletTokenPositionHistory",
    "CoParticipationEdge",
    "BehavioralSimilarityCache",
    # Cross-Token Memory Service
    "CrossTokenMemoryService",
    "WalletMetrics",
    "CoParticipationMetrics",
    # Engines
    "SnapshotEngine",
    "SnapshotCadence",
    "SnapshotScheduler",
    "EventStore",
    "EventReplayer",
    "ReplayEngine",
    "TokenState",
    "TrajectoryComputer",
    "TokenTrajectory",
    # Trajectory Smoothing
    "TrajectorySmoother",
    "SmoothedTrajectoryComputer",
    "SmoothedTrajectory",
    "SmoothedValue",
    "TimeSeriesPoint",
    # Event Ordering
    "EventOrderer",
    "EventOrderingKey",
    "ReplayConfig",
    "CorrectionEventBuilder",
    # Sprint 8: Price Snapshots
    "PriceSnapshot",
    "PriceConfidenceLevel",
    "PriceSnapshotService",
    "PriceSnapshotCollector",
    "PriceObservation",
    "HistoricalPriceQuery",
    "HistoricalPriceResult",
    # Sprint 8: PnL Calculator
    "AccountingMethod",
    "RealisedPnLRecord",
    "CostBasisLot",
    "CostBasisCalculator",
    "RealisedPnLCalculator",
    "ProfitExtractionAnalyzer",
    "TradeRecord",
    "CostBasisEstimate",
    "RealisedPnLEstimate",
    # Sprint 9: Exit Classifier
    "ExitEventType",
    "ExitEvidence",
    "ExitEventClassification",
    "ExitClassifierConfig",
    "ExitEventClassifier",
    "SellConfidenceScorer",
    "PnLReliabilityScorer",
    "create_exit_classifier",
    # Sprint 9: Transfer Chain Detection
    "TransferChainResult",
    "TransferChainConfig",
    "TransferChainDetector",
    "WalletInfoProvider",
    "create_transfer_chain_detector",
    # Sprint 9: LP Action Separation
    "LPActionResult",
    "LPActionDetector",
    "create_lp_action_detector",
    # Sprint 9: CEX Deposit Detection
    "CEXDepositResult",
    "CEXDetectionConfig",
    "CEXDepositDetector",
    "create_cex_deposit_detector",
]
