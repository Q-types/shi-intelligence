"""
Longitudinal Intelligence Database Models.

Models for event-sourced storage, snapshots, and time-series analysis.

HARD RULES:
- Never overwrite historical data
- All events are immutable and append-only
- Snapshots are versioned and never modified
- Raw events enable full state reconstruction
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from enum import Enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..data.models import Base


# ============================================================================
# Event Types
# ============================================================================


class EventType(str, Enum):
    """Types of raw events stored in event store."""

    TRADE = "trade"
    LIQUIDITY = "liquidity"
    FUNDING = "funding"
    STATE_TRANSITION = "state_transition"
    # Correction event types (events remain immutable, corrections are new events)
    EVENT_CORRECTION = "event_correction"
    EVENT_INVALIDATION = "event_invalidation"
    BACKFILL_INSERT = "backfill_insert"
    PROVIDER_RECONCILIATION = "provider_reconciliation"


class TradeType(str, Enum):
    """Trade direction."""

    BUY = "buy"
    SELL = "sell"
    SWAP = "swap"


class LiquidityAction(str, Enum):
    """Liquidity pool action."""

    ADD = "add"
    REMOVE = "remove"
    CREATE_POOL = "create_pool"


class StateTransitionType(str, Enum):
    """Types of state transitions."""

    REGIME_CHANGE = "regime_change"
    ARCHETYPE_CHANGE = "archetype_change"
    COORDINATION_DETECTED = "coordination_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    WHALE_MOVEMENT = "whale_movement"


class AccountingMethod(str, Enum):
    """Cost basis accounting methods for PnL calculation."""

    FIFO = "fifo"  # First In, First Out (default)
    LIFO = "lifo"  # Last In, First Out
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted average cost


class PriceConfidenceLevel(str, Enum):
    """Price confidence levels (matches Sprint 7 PriceConfidence)."""

    HIGH = "high"  # >$1M liquidity, primary source
    MEDIUM = "medium"  # >$10K liquidity, or fallback source
    LOW = "low"  # <$10K liquidity, or pool-implied
    NONE = "none"  # Price unavailable


# ============================================================================
# Event Store Models (Immutable, Append-Only)
# ============================================================================


class RawEvent(Base):
    """
    Base event table for all raw events.

    Uses single-table inheritance for efficient querying across event types.
    All events are immutable - never update, only append.
    """

    __tablename__ = "raw_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(30), index=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), index=True)

    # Event sequence for ordering (monotonic within token)
    sequence_number: Mapped[int] = mapped_column(BigInteger, index=True)

    # Blockchain reference (for deterministic ordering)
    # Canonical ordering: slot > block_time > transaction_index > instruction_index > signature > event_type
    signature: Mapped[Optional[str]] = mapped_column(String(88), unique=True)
    slot: Mapped[Optional[int]] = mapped_column(BigInteger, index=True)
    block_time: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    transaction_index: Mapped[Optional[int]] = mapped_column(Integer)  # Index within block
    instruction_index: Mapped[Optional[int]] = mapped_column(Integer)  # Index within transaction

    # Correction/Invalidation tracking
    is_invalidated: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    invalidated_by_event_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    corrects_event_id: Mapped[Optional[int]] = mapped_column(BigInteger)  # For EVENT_CORRECTION
    data_version: Mapped[int] = mapped_column(Integer, default=1)  # For replay as-of versioning

    # Event-specific payload (denormalized for query efficiency)
    payload: Mapped[dict] = mapped_column(JSONB)

    # Deduplication hash
    event_hash: Mapped[str] = mapped_column(String(64), index=True)

    # Ingestion metadata
    ingested_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )
    source: Mapped[str] = mapped_column(String(50))  # helius, rpc, backfill

    __table_args__ = (
        Index("ix_raw_events_token_seq", "token_mint", "sequence_number"),
        Index("ix_raw_events_token_time", "token_mint", "timestamp"),
        Index("ix_raw_events_type_token", "event_type", "token_mint"),
    )

    __mapper_args__ = {
        "polymorphic_on": "event_type",
        "polymorphic_identity": "base",
    }


class TradeEventRecord(RawEvent):
    """Trade event (buy/sell/swap)."""

    __mapper_args__ = {"polymorphic_identity": EventType.TRADE.value}

    # Trade-specific fields stored in payload:
    # - wallet_address: str
    # - trade_type: TradeType
    # - amount: int (token amount)
    # - price_usd: float (optional)
    # - dex: str (raydium, orca, jupiter)
    # - pool_address: str


class LiquidityEventRecord(RawEvent):
    """Liquidity pool event."""

    __mapper_args__ = {"polymorphic_identity": EventType.LIQUIDITY.value}

    # Liquidity-specific fields stored in payload:
    # - wallet_address: str
    # - action: LiquidityAction
    # - pool_address: str
    # - dex: str
    # - token_amount: int
    # - quote_amount: int (SOL/USDC)
    # - lp_tokens: int (minted/burned)


class FundingEventRecord(RawEvent):
    """SOL funding transfer between wallets."""

    __mapper_args__ = {"polymorphic_identity": EventType.FUNDING.value}

    # Funding-specific fields stored in payload:
    # - source_address: str
    # - target_address: str
    # - amount_lamports: int


class StateTransitionRecord(RawEvent):
    """Derived state transition (from analytics)."""

    __mapper_args__ = {"polymorphic_identity": EventType.STATE_TRANSITION.value}

    # State transition fields stored in payload:
    # - transition_type: StateTransitionType
    # - wallet_address: str (optional, for wallet-level transitions)
    # - old_state: dict
    # - new_state: dict
    # - confidence: float
    # - evidence: dict


# ============================================================================
# Snapshot Configuration
# ============================================================================


class SnapshotConfig(Base):
    """Configuration for snapshot collection per token."""

    __tablename__ = "snapshot_configs"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_mint: Mapped[str] = mapped_column(String(44), unique=True, index=True)

    # Launch detection
    launch_detected_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    first_trade_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))

    # Current cadence state
    current_cadence_seconds: Mapped[int] = mapped_column(Integer, default=60)  # 1 min initially
    cadence_updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))

    # Snapshot collection flags
    collect_holders: Mapped[bool] = mapped_column(Boolean, default=True)
    collect_balances: Mapped[bool] = mapped_column(Boolean, default=True)
    collect_liquidity: Mapped[bool] = mapped_column(Boolean, default=True)
    collect_volume: Mapped[bool] = mapped_column(Boolean, default=True)
    collect_graph: Mapped[bool] = mapped_column(Boolean, default=True)
    collect_archetypes: Mapped[bool] = mapped_column(Boolean, default=True)
    collect_coordination: Mapped[bool] = mapped_column(Boolean, default=True)

    # Scheduling
    next_snapshot_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    last_snapshot_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    total_snapshots: Mapped[int] = mapped_column(Integer, default=0)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    paused_reason: Mapped[Optional[str]] = mapped_column(String(200))

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class TokenLaunchState(Base):
    """
    Track token launch lifecycle for dynamic cadence adjustment.

    Cadence schedule:
    - 0-1h: 1 minute snapshots (early launch, high activity)
    - 1-6h: 5 minute snapshots
    - 6-24h: 15 minute snapshots
    - 1-7d: 1 hour snapshots
    - 7d+: 6 hour snapshots (mature token)
    """

    __tablename__ = "token_launch_states"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_mint: Mapped[str] = mapped_column(String(44), unique=True, index=True)

    # Launch timeline
    pool_created_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    first_trade_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    peak_volume_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))

    # Lifecycle phase
    phase: Mapped[str] = mapped_column(String(30), default="early_launch")
    # Phases: early_launch, active, maturing, stable, dormant

    # Metrics at key points
    launch_liquidity_usd: Mapped[Optional[float]] = mapped_column(Float)
    peak_liquidity_usd: Mapped[Optional[float]] = mapped_column(Float)
    current_liquidity_usd: Mapped[Optional[float]] = mapped_column(Float)
    launch_holder_count: Mapped[Optional[int]] = mapped_column(Integer)
    peak_holder_count: Mapped[Optional[int]] = mapped_column(Integer)

    # Activity metrics
    hours_since_launch: Mapped[float] = mapped_column(Float, default=0.0)
    is_high_activity: Mapped[bool] = mapped_column(Boolean, default=True)

    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


# ============================================================================
# Snapshot Models (Immutable, Versioned)
# ============================================================================


class GraphSnapshot(Base):
    """
    Snapshot of funding graph structure at a point in time.

    Stores graph metrics and structure for replay.
    """

    __tablename__ = "graph_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), index=True)

    # Graph structure summary
    node_count: Mapped[int] = mapped_column(Integer)
    edge_count: Mapped[int] = mapped_column(Integer)
    component_count: Mapped[int] = mapped_column(Integer)
    largest_component_size: Mapped[int] = mapped_column(Integer)

    # Centrality metrics (aggregated)
    avg_in_degree: Mapped[float] = mapped_column(Float)
    avg_out_degree: Mapped[float] = mapped_column(Float)
    max_in_degree: Mapped[int] = mapped_column(Integer)
    max_out_degree: Mapped[int] = mapped_column(Integer)
    avg_pagerank: Mapped[float] = mapped_column(Float)
    max_pagerank: Mapped[float] = mapped_column(Float)

    # Community structure
    community_count: Mapped[int] = mapped_column(Integer, default=0)
    modularity_score: Mapped[Optional[float]] = mapped_column(Float)

    # Top nodes (JSON arrays for quick access)
    top_funders: Mapped[list] = mapped_column(JSONB, default=list)  # [{address, funded_count, total_amount}]
    top_centrality_nodes: Mapped[list] = mapped_column(JSONB, default=list)  # [{address, pagerank, betweenness}]

    # Full adjacency list for replay (compressed JSON)
    adjacency_data: Mapped[Optional[dict]] = mapped_column(JSONB)  # {edges: [[src, tgt, weight], ...]}

    # Metadata
    snapshot_version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_graph_snapshots_token_time", "token_mint", "timestamp"),
    )


class CoordinationSnapshot(Base):
    """
    Snapshot of coordination detection state.

    Stores detected coordination candidates and evidence.
    """

    __tablename__ = "coordination_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), index=True)

    # Detection summary
    candidate_clusters: Mapped[int] = mapped_column(Integer, default=0)
    significant_clusters: Mapped[int] = mapped_column(Integer, default=0)
    wallets_in_clusters: Mapped[int] = mapped_column(Integer, default=0)

    # Aggregate coordination metrics
    max_cluster_size: Mapped[int] = mapped_column(Integer, default=0)
    max_z_score: Mapped[float] = mapped_column(Float, default=0.0)
    min_p_value: Mapped[float] = mapped_column(Float, default=1.0)

    # Cluster details (JSON array)
    clusters: Mapped[list] = mapped_column(JSONB, default=list)
    # Format: [{cluster_id, wallets, z_score, p_value, evidence_types, coordination_level}]

    # Evidence types present
    evidence_breakdown: Mapped[dict] = mapped_column(JSONB, default=dict)
    # Format: {shared_funder: count, funding_time: count, ...}

    # Metadata
    detector_version: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_coordination_snapshots_token_time", "token_mint", "timestamp"),
    )


class VolumeSnapshot(Base):
    """
    Snapshot of trading volume metrics.

    Tracks buy/sell volume, unique traders, and activity metrics.
    """

    __tablename__ = "volume_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), index=True)

    # Window (e.g., last 1h, last 24h)
    window_seconds: Mapped[int] = mapped_column(Integer)  # 3600 for 1h, 86400 for 24h

    # Volume metrics
    buy_volume_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    sell_volume_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    total_volume_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    buy_volume_usd: Mapped[Optional[float]] = mapped_column(Float)
    sell_volume_usd: Mapped[Optional[float]] = mapped_column(Float)
    total_volume_usd: Mapped[Optional[float]] = mapped_column(Float)

    # Trade counts
    buy_count: Mapped[int] = mapped_column(Integer, default=0)
    sell_count: Mapped[int] = mapped_column(Integer, default=0)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)

    # Unique participants
    unique_buyers: Mapped[int] = mapped_column(Integer, default=0)
    unique_sellers: Mapped[int] = mapped_column(Integer, default=0)
    unique_traders: Mapped[int] = mapped_column(Integer, default=0)

    # Derived metrics
    buy_sell_ratio: Mapped[float] = mapped_column(Float, default=1.0)
    avg_trade_size_tokens: Mapped[float] = mapped_column(Float, default=0.0)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_volume_snapshots_token_time_window", "token_mint", "timestamp", "window_seconds"),
    )


# ============================================================================
# Price Snapshot (Sprint 8: Historical Price Data)
# ============================================================================


class PriceSnapshot(Base):
    """
    Historical price observation at a point in time.

    Stores versioned price data for cost basis calculation and PnL analysis.
    Price snapshots are IMMUTABLE - never modified, only appended.

    HARD RULES (Sprint 8):
    - Price is NOT ground truth - always includes confidence
    - Missing price reduces confidence, does not become zero
    - All price data includes provenance (source, fetched_at, payload_hash)
    - Data is versioned for reproducibility
    """

    __tablename__ = "price_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), index=True)

    # Price data
    price_usd: Mapped[float] = mapped_column(Float)
    price_change_24h_pct: Mapped[Optional[float]] = mapped_column(Float)

    # Liquidity context
    liquidity_usd: Mapped[Optional[float]] = mapped_column(Float)
    volume_24h_usd: Mapped[Optional[float]] = mapped_column(Float)

    # Confidence scoring (Sprint 7 enhanced)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5)  # 0.0-1.0, never zero
    confidence_level: Mapped[str] = mapped_column(String(10), default="medium")  # high/medium/low/none

    # Provenance tracking
    source: Mapped[str] = mapped_column(String(30))  # jupiter, birdeye, pool_implied
    payload_hash: Mapped[Optional[str]] = mapped_column(String(64))  # SHA256 of API response
    fetched_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))
    staleness_seconds: Mapped[int] = mapped_column(Integer, default=0)

    # Confidence reason (human-readable)
    confidence_reason: Mapped[Optional[str]] = mapped_column(String(500))

    # Versioning for schema evolution
    data_version: Mapped[int] = mapped_column(Integer, default=1)

    # Snapshot cadence context
    cadence_seconds: Mapped[int] = mapped_column(Integer)  # Interval at which collected
    sequence_in_token: Mapped[int] = mapped_column(BigInteger)  # Ordering within token

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_price_snapshots_token_time", "token_mint", "timestamp"),
        Index("ix_price_snapshots_token_seq", "token_mint", "sequence_in_token"),
        Index("ix_price_snapshots_confidence", "token_mint", "confidence_score"),
    )


# ============================================================================
# Realised PnL (Sprint 8: Profit Extraction Analysis)
# ============================================================================


class RealisedPnLRecord(Base):
    """
    Realised PnL record for a wallet exit event.

    Created when a wallet sells tokens. Uses configurable accounting method
    (FIFO, LIFO, weighted_average) to match sells to prior buys.

    HARD RULES (Sprint 8):
    - Accounting method must be explicit
    - Confidence propagates from price data
    - Do not show precise PnL when confidence is low
    - Separate realised from unrealised PnL
    """

    __tablename__ = "realised_pnl_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    wallet_address: Mapped[str] = mapped_column(String(44), index=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)

    # Exit event reference
    exit_timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), index=True)
    exit_event_id: Mapped[Optional[int]] = mapped_column(BigInteger)  # Link to RawEvent
    exit_signature: Mapped[Optional[str]] = mapped_column(String(88))

    # Exit details
    exit_tokens: Mapped[int] = mapped_column(BigInteger)  # Tokens sold
    exit_price_usd: Mapped[float] = mapped_column(Float)
    exit_value_usd: Mapped[float] = mapped_column(Float)  # exit_tokens * exit_price

    # Entry (cost basis) details
    entry_price_usd: Mapped[float] = mapped_column(Float)  # Computed from accounting method
    cost_basis_usd: Mapped[float] = mapped_column(Float)  # entry_price * exit_tokens
    accounting_method: Mapped[str] = mapped_column(String(20), default="fifo")

    # Realised PnL
    realised_pnl_usd: Mapped[float] = mapped_column(Float)  # exit_value - cost_basis
    realised_pnl_pct: Mapped[float] = mapped_column(Float)  # (exit_price - entry_price) / entry_price

    # Exit efficiency (how well did they time the exit?)
    # 1.0 = exited at peak, 0.0 = exited at worst point
    exit_efficiency: Mapped[Optional[float]] = mapped_column(Float)
    peak_price_usd: Mapped[Optional[float]] = mapped_column(Float)  # Highest price seen
    peak_to_exit_drawdown_pct: Mapped[Optional[float]] = mapped_column(Float)

    # Liquidity-adjusted metrics
    liquidity_at_exit_usd: Mapped[Optional[float]] = mapped_column(Float)
    position_vs_liquidity_pct: Mapped[Optional[float]] = mapped_column(Float)  # exit_value / liquidity
    liquidity_adjusted_pnl_usd: Mapped[Optional[float]] = mapped_column(Float)  # Estimated after slippage

    # Confidence tracking
    entry_price_confidence: Mapped[float] = mapped_column(Float, default=0.5)
    exit_price_confidence: Mapped[float] = mapped_column(Float, default=0.5)
    overall_confidence: Mapped[float] = mapped_column(Float, default=0.5)  # min(entry, exit)
    confidence_reason: Mapped[Optional[str]] = mapped_column(String(500))

    # Partial vs full exit
    is_partial_exit: Mapped[bool] = mapped_column(Boolean, default=False)
    remaining_position_tokens: Mapped[int] = mapped_column(BigInteger, default=0)

    # Metadata
    data_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_realised_pnl_wallet_token", "wallet_address", "token_mint"),
        Index("ix_realised_pnl_token_time", "token_mint", "exit_timestamp"),
        Index("ix_realised_pnl_confidence", "overall_confidence"),
    )


class CostBasisLot(Base):
    """
    Individual cost basis lot for FIFO/LIFO accounting.

    Each buy creates a lot. Sells consume lots based on accounting method.
    Enables precise cost basis tracking for partial position management.
    """

    __tablename__ = "cost_basis_lots"

    id: Mapped[int] = mapped_column(primary_key=True)
    wallet_address: Mapped[str] = mapped_column(String(44), index=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)

    # Acquisition details
    acquired_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), index=True)
    acquired_event_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    acquired_signature: Mapped[Optional[str]] = mapped_column(String(88))

    # Lot details
    original_tokens: Mapped[int] = mapped_column(BigInteger)  # Tokens at acquisition
    remaining_tokens: Mapped[int] = mapped_column(BigInteger)  # Tokens not yet sold
    acquisition_price_usd: Mapped[float] = mapped_column(Float)
    acquisition_cost_usd: Mapped[float] = mapped_column(Float)  # original_tokens * acquisition_price

    # Price confidence at acquisition
    price_confidence: Mapped[float] = mapped_column(Float, default=0.5)
    price_source: Mapped[str] = mapped_column(String(30))

    # Status
    is_fully_consumed: Mapped[bool] = mapped_column(Boolean, default=False)
    consumed_by_exits: Mapped[list] = mapped_column(JSONB, default=list)
    # Format: [{exit_id, tokens_consumed, exit_timestamp}]

    # Metadata
    data_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_cost_basis_wallet_token", "wallet_address", "token_mint"),
        Index("ix_cost_basis_token_time", "token_mint", "acquired_at"),
        Index("ix_cost_basis_not_consumed", "wallet_address", "token_mint", "is_fully_consumed"),
    )


# ============================================================================
# Trajectory Features
# ============================================================================


class TrajectorySnapshot(Base):
    """
    Behavioral trajectory features computed from time-series.

    Stores velocity/acceleration of key metrics for trend detection.
    """

    __tablename__ = "trajectory_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), index=True)

    # Window for trajectory computation
    window_hours: Mapped[int] = mapped_column(Integer, default=24)

    # Accumulation/Distribution rates
    accumulation_rate: Mapped[float] = mapped_column(Float, default=0.0)  # dBalance/dt for top holders
    sell_acceleration: Mapped[float] = mapped_column(Float, default=0.0)  # d²SellVolume/dt²

    # Liquidity dynamics
    liquidity_decay_rate: Mapped[float] = mapped_column(Float, default=0.0)  # dLiquidity/dt

    # Concentration dynamics
    whale_dispersion_rate: Mapped[float] = mapped_column(Float, default=0.0)  # dWhaleDominance/dt
    holder_churn_velocity: Mapped[float] = mapped_column(Float, default=0.0)  # New holders / Lost holders per time

    # Coordination dynamics
    coordination_persistence: Mapped[float] = mapped_column(Float, default=0.0)  # Duration of coordination signals

    # Regime trajectory
    regime_stability: Mapped[float] = mapped_column(Float, default=0.0)  # Time in current regime
    regime_transition_probability: Mapped[float] = mapped_column(Float, default=0.0)

    # Derived trend indicators
    trend_direction: Mapped[str] = mapped_column(String(20), default="stable")
    # Values: accumulating, distributing, stable, volatile
    trend_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # Raw trajectory data for replay
    metric_velocities: Mapped[dict] = mapped_column(JSONB, default=dict)
    # Format: {hhi_velocity, gini_velocity, churn_velocity, ...}

    # Metadata
    trajectory_version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_trajectory_snapshots_token_time", "token_mint", "timestamp"),
    )


# ============================================================================
# Cross-Token Behavior Memory
# ============================================================================


class WalletBehaviorHistory(Base):
    """
    Comprehensive behavioral history for a wallet across all tokens.

    Aggregates behavior patterns for cross-token analysis.
    """

    __tablename__ = "wallet_behavior_histories"

    id: Mapped[int] = mapped_column(primary_key=True)
    wallet_address: Mapped[str] = mapped_column(String(44), unique=True, index=True)

    # Participation stats
    total_tokens_participated: Mapped[int] = mapped_column(Integer, default=0)
    active_tokens_current: Mapped[int] = mapped_column(Integer, default=0)

    # Timing patterns
    avg_entry_timing_percentile: Mapped[float] = mapped_column(Float, default=50.0)  # 0-100
    avg_exit_timing_percentile: Mapped[float] = mapped_column(Float, default=50.0)
    avg_holding_days: Mapped[float] = mapped_column(Float, default=0.0)

    # Behavior patterns (counts)
    sniper_count: Mapped[int] = mapped_column(Integer, default=0)
    accumulator_count: Mapped[int] = mapped_column(Integer, default=0)
    quick_exit_count: Mapped[int] = mapped_column(Integer, default=0)
    lp_interaction_count: Mapped[int] = mapped_column(Integer, default=0)

    # Coordination participation
    coordination_cluster_count: Mapped[int] = mapped_column(Integer, default=0)
    same_entity_tokens: Mapped[int] = mapped_column(Integer, default=0)  # Tokens where linked to same entity

    # Performance
    profitable_exits: Mapped[int] = mapped_column(Integer, default=0)
    unprofitable_exits: Mapped[int] = mapped_column(Integer, default=0)
    avg_pnl_pct: Mapped[float] = mapped_column(Float, default=0.0)
    total_volume_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # Repeated behavior patterns (JSON arrays for token lists)
    repeated_coordination_with: Mapped[list] = mapped_column(JSONB, default=list)
    # Format: [{wallet, token_count, last_seen}]
    repeated_funder_usage: Mapped[list] = mapped_column(JSONB, default=list)
    # Format: [{funder, token_count, total_amount}]

    # Token interaction history (recent tokens)
    recent_tokens: Mapped[list] = mapped_column(JSONB, default=list)
    # Format: [{mint, timestamp, archetype, pnl_pct}]

    # Entry/Exit timing metrics
    entry_timing_variance: Mapped[float] = mapped_column(Float, default=0.0)
    early_exit_rate: Mapped[float] = mapped_column(Float, default=0.0)
    avg_hold_duration_hours: Mapped[float] = mapped_column(Float, default=0.0)
    avg_position_size_pct: Mapped[float] = mapped_column(Float, default=0.0)

    # Co-trader patterns
    common_co_traders: Mapped[list] = mapped_column(JSONB, default=list)
    # Format: [{wallet, co_participation_count, tokens}]

    # Realised PnL (where available)
    realised_pnl_estimate_usd: Mapped[Optional[float]] = mapped_column(Float)
    realised_pnl_pct: Mapped[Optional[float]] = mapped_column(Float)  # Sprint 8
    pnl_data_points: Mapped[int] = mapped_column(Integer, default=0)

    # Sprint 8: Profit Extraction Behaviour Features
    # These answer: who extracts profit early, who holds through volatility
    realised_profit_rate: Mapped[float] = mapped_column(Float, default=0.0)
    # Proportion of exits that are profitable
    early_profit_exit_rate: Mapped[float] = mapped_column(Float, default=0.0)
    # Proportion of profitable exits that happen within 4h of entry
    average_exit_efficiency: Mapped[float] = mapped_column(Float, default=0.0)
    # Mean of exit_efficiency across all exits (1.0 = always at peak)
    hold_through_drawdown_score: Mapped[float] = mapped_column(Float, default=0.0)
    # How often wallet holds through >20% drawdowns
    profit_taking_consistency: Mapped[float] = mapped_column(Float, default=0.0)
    # Consistency of profit-taking behavior (low variance = consistent)
    liquidity_sensitive_exit_score: Mapped[float] = mapped_column(Float, default=0.0)
    # How often wallet exits before liquidity deteriorates
    exit_efficiency: Mapped[float] = mapped_column(Float, default=0.0)
    # Overall exit timing quality (Sprint 8 cross-token)
    liquidity_adjusted_exit_quality: Mapped[float] = mapped_column(Float, default=0.0)
    # Exit quality accounting for slippage/liquidity

    # Behaviour History Score (NOT "reputation" - pending validated labels)
    # This is probabilistic and confidence-weighted
    behaviour_history_score: Mapped[float] = mapped_column(Float, default=50.0)
    behaviour_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    # Confidence: 0-1, based on number of data points and recency

    # Legacy derived scores (keeping for backwards compatibility)
    serial_sniper_score: Mapped[float] = mapped_column(Float, default=0.0)
    diamond_hands_score: Mapped[float] = mapped_column(Float, default=0.0)
    coordination_propensity: Mapped[float] = mapped_column(Float, default=0.0)

    # Metadata
    first_seen_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))
    last_updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
    history_version: Mapped[str] = mapped_column(String(20), default="1.0.0")

    __table_args__ = (
        Index("ix_wallet_behavior_serial_sniper", "serial_sniper_score"),
        Index("ix_wallet_behavior_coordination", "coordination_propensity"),
        Index("ix_wallet_behavior_score", "behaviour_history_score"),
    )


class TokenParticipationHistory(Base):
    """
    Per-token participation record for a wallet.

    Tracks wallet behavior within a specific token.
    """

    __tablename__ = "token_participation_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    wallet_address: Mapped[str] = mapped_column(String(44), index=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)

    # Timing
    first_seen_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))
    entry_timing_percentile: Mapped[float] = mapped_column(Float, default=50.0)
    exit_timing_percentile: Mapped[Optional[float]] = mapped_column(Float)

    # Position
    max_position_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    max_position_pct: Mapped[float] = mapped_column(Float, default=0.0)
    final_position_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    is_exited: Mapped[bool] = mapped_column(Boolean, default=False)

    # Behavior classification
    archetype_assigned: Mapped[Optional[str]] = mapped_column(String(50))
    # Values: sniper, accumulator, quick_exit, lp_provider, holder, etc.

    # Coordination
    in_coordination_cluster: Mapped[bool] = mapped_column(Boolean, default=False)
    coordination_cluster_id: Mapped[Optional[str]] = mapped_column(String(64))

    # Performance
    realised_pnl_pct: Mapped[Optional[float]] = mapped_column(Float)
    realised_pnl_usd: Mapped[Optional[float]] = mapped_column(Float)
    hold_duration_hours: Mapped[float] = mapped_column(Float, default=0.0)

    # Trade activity
    buy_count: Mapped[int] = mapped_column(Integer, default=0)
    sell_count: Mapped[int] = mapped_column(Integer, default=0)
    total_volume_tokens: Mapped[int] = mapped_column(BigInteger, default=0)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("wallet_address", "token_mint", name="uq_wallet_token"),
        Index("ix_token_participation_wallet", "wallet_address"),
        Index("ix_token_participation_token", "token_mint"),
        Index("ix_token_participation_archetype", "archetype_assigned"),
    )


class WalletTokenPositionHistory(Base):
    """
    Historical position snapshots for a wallet in a token.

    Enables reconstruction of position trajectory.
    """

    __tablename__ = "wallet_token_position_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    wallet_address: Mapped[str] = mapped_column(String(44), index=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), index=True)

    # Position state
    position_tokens: Mapped[int] = mapped_column(BigInteger)
    position_pct: Mapped[float] = mapped_column(Float)

    # Change since last snapshot
    delta_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    delta_source: Mapped[str] = mapped_column(String(20))
    # Values: buy, sell, transfer_in, transfer_out

    # Context
    event_sequence: Mapped[int] = mapped_column(BigInteger)  # Link to event store
    price_at_snapshot: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_position_history_wallet_token_time", "wallet_address", "token_mint", "timestamp"),
    )


class CoParticipationEdge(Base):
    """
    Edges between wallets that participate in the same tokens.

    Used for detecting coordination patterns across tokens.
    """

    __tablename__ = "co_participation_edges"

    id: Mapped[int] = mapped_column(primary_key=True)
    wallet_a: Mapped[str] = mapped_column(String(44), index=True)
    wallet_b: Mapped[str] = mapped_column(String(44), index=True)

    # Co-participation metrics
    shared_token_count: Mapped[int] = mapped_column(Integer, default=0)
    shared_tokens: Mapped[list] = mapped_column(JSONB, default=list)
    # Format: [{mint, wallet_a_timing, wallet_b_timing, timing_correlation}]

    # Timing correlation
    avg_entry_timing_correlation: Mapped[float] = mapped_column(Float, default=0.0)
    # -1 to 1: high positive = enter at same time
    avg_exit_timing_correlation: Mapped[float] = mapped_column(Float, default=0.0)

    # Behavior similarity
    archetype_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    # 0-1: proportion of tokens where same archetype assigned

    # Coordination evidence
    shared_coordination_clusters: Mapped[int] = mapped_column(Integer, default=0)
    # Count of tokens where both in same coordination cluster

    # Derived score
    coordination_likelihood: Mapped[float] = mapped_column(Float, default=0.0)
    # Composite score suggesting coordinated behavior

    first_seen_together: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))
    last_seen_together: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        # Ensure wallet_a < wallet_b to avoid duplicate edges
        UniqueConstraint("wallet_a", "wallet_b", name="uq_co_participation_pair"),
        Index("ix_co_participation_wallet_a", "wallet_a"),
        Index("ix_co_participation_wallet_b", "wallet_b"),
        Index("ix_co_participation_likelihood", "coordination_likelihood"),
    )


class BehavioralSimilarityCache(Base):
    """
    Cached behavioral similarity scores between wallets.

    Expensive to compute, so cached with TTL.
    """

    __tablename__ = "behavioral_similarity_cache"

    id: Mapped[int] = mapped_column(primary_key=True)
    wallet_a: Mapped[str] = mapped_column(String(44), index=True)
    wallet_b: Mapped[str] = mapped_column(String(44), index=True)

    # Similarity dimensions
    timing_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    # How similar entry/exit timing patterns are
    position_sizing_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    # How similar position sizes are
    archetype_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    # How often assigned same archetype
    pnl_correlation: Mapped[float] = mapped_column(Float, default=0.0)
    # How correlated their PnL outcomes are
    trade_pattern_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    # How similar their trading patterns (buy/sell frequency)

    # Aggregate similarity
    overall_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    # Weighted combination of dimensions

    # Computation metadata
    tokens_compared: Mapped[int] = mapped_column(Integer, default=0)
    computation_version: Mapped[str] = mapped_column(String(20), default="1.0.0")

    # Cache management
    computed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))
    is_stale: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        UniqueConstraint("wallet_a", "wallet_b", name="uq_similarity_pair"),
        Index("ix_similarity_overall", "overall_similarity"),
        Index("ix_similarity_expires", "expires_at"),
    )
