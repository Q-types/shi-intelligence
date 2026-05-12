"""
Database Models for SHI.

SQLAlchemy ORM models for persistent storage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    ARRAY,
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
)
from sqlalchemy.dialects.postgresql import TIMESTAMP

# Use JSON for SQLite compatibility, JSONB for PostgreSQL performance
# Both work with SQLAlchemy's JSON column type
from sqlalchemy import JSON as JSONB  # Alias for compatibility
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Token(Base):
    """Token metadata."""

    __tablename__ = "tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    mint: Mapped[str] = mapped_column(String(44), unique=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(100))
    symbol: Mapped[Optional[str]] = mapped_column(String(20))
    decimals: Mapped[int] = mapped_column(Integer, default=9)
    total_supply: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    snapshots: Mapped[list["HolderSnapshotRecord"]] = relationship(back_populates="token")
    metrics: Mapped[list["MetricRecord"]] = relationship(back_populates="token")


class Wallet(Base):
    """Wallet metadata."""

    __tablename__ = "wallets"

    id: Mapped[int] = mapped_column(primary_key=True)
    address: Mapped[str] = mapped_column(String(44), unique=True, index=True)
    funded_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey("wallets.id"))
    first_funded_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime)
    is_exchange: Mapped[bool] = mapped_column(Boolean, default=False)
    is_contract: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Self-referential relationship
    funded_by: Mapped[Optional["Wallet"]] = relationship(
        "Wallet", remote_side=[id], backref="funded_wallets"
    )


class FundingEdgeRecord(Base):
    """Funding graph edge."""

    __tablename__ = "funding_edges"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("wallets.id"), index=True)
    target_id: Mapped[int] = mapped_column(ForeignKey("wallets.id"), index=True)
    amount_lamports: Mapped[int] = mapped_column(BigInteger)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    signature: Mapped[str] = mapped_column(String(88))

    __table_args__ = (
        Index("ix_funding_edges_source_target", "source_id", "target_id"),
    )


class HolderSnapshotRecord(Base):
    """Point-in-time holder snapshot."""

    __tablename__ = "holder_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_id: Mapped[int] = mapped_column(ForeignKey("tokens.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    holder_count: Mapped[int] = mapped_column(Integer)
    total_supply: Mapped[int] = mapped_column(BigInteger)
    checksum: Mapped[str] = mapped_column(String(64))

    token: Mapped["Token"] = relationship(back_populates="snapshots")
    balances: Mapped[list["BalanceRecord"]] = relationship(back_populates="snapshot")


class BalanceRecord(Base):
    """Individual wallet balance in a snapshot."""

    __tablename__ = "balances"

    id: Mapped[int] = mapped_column(primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("holder_snapshots.id"), index=True)
    wallet_id: Mapped[int] = mapped_column(ForeignKey("wallets.id"), index=True)
    balance: Mapped[int] = mapped_column(BigInteger)
    share: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer)

    snapshot: Mapped["HolderSnapshotRecord"] = relationship(back_populates="balances")

    __table_args__ = (Index("ix_balances_snapshot_wallet", "snapshot_id", "wallet_id"),)


class WalletFeatures(Base):
    """Computed features for a wallet-token pair."""

    __tablename__ = "wallet_features"

    id: Mapped[int] = mapped_column(primary_key=True)
    wallet_id: Mapped[int] = mapped_column(ForeignKey("wallets.id"), index=True)
    token_id: Mapped[int] = mapped_column(ForeignKey("tokens.id"), index=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("holder_snapshots.id"))

    # Distribution features
    balance: Mapped[int] = mapped_column(BigInteger)
    share: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer)

    # Temporal features
    entry_time_relative: Mapped[float] = mapped_column(Float)  # Days since launch
    holding_duration: Mapped[float] = mapped_column(Float)  # Days
    position_volatility: Mapped[float] = mapped_column(Float)

    # Flow features
    delta_balance_7d: Mapped[float] = mapped_column(Float)
    delta_balance_30d: Mapped[float] = mapped_column(Float)

    # Trade features
    trade_count: Mapped[int] = mapped_column(Integer, default=0)
    burstiness: Mapped[float] = mapped_column(Float, default=0.0)
    swap_frequency: Mapped[float] = mapped_column(Float, default=0.0)
    lp_interaction_ratio: Mapped[float] = mapped_column(Float, default=0.0)

    # Graph features
    in_degree: Mapped[int] = mapped_column(Integer, default=0)
    out_degree: Mapped[int] = mapped_column(Integer, default=0)
    eigenvector_centrality: Mapped[float] = mapped_column(Float, default=0.0)
    shared_funder_count: Mapped[int] = mapped_column(Integer, default=0)

    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("wallet_id", "token_id", "snapshot_id", name="uq_wallet_token_snapshot"),
        Index("ix_wallet_features_token_snapshot", "token_id", "snapshot_id"),
    )


class ArchetypeAssignment(Base):
    """Behavioral archetype assignment."""

    __tablename__ = "archetype_assignments"

    id: Mapped[int] = mapped_column(primary_key=True)
    wallet_id: Mapped[int] = mapped_column(ForeignKey("wallets.id"), index=True)
    token_id: Mapped[int] = mapped_column(ForeignKey("tokens.id"), index=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("holder_snapshots.id"))

    archetype: Mapped[str] = mapped_column(String(50))  # sniper, accumulator, etc.
    confidence: Mapped[float] = mapped_column(Float)
    features_used: Mapped[dict] = mapped_column(JSON)
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer)

    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    model_version: Mapped[str] = mapped_column(String(20))


class MetricRecord(Base):
    """Computed metric value."""

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_id: Mapped[int] = mapped_column(ForeignKey("tokens.id"), index=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("holder_snapshots.id"))

    metric_name: Mapped[str] = mapped_column(String(50), index=True)
    value: Mapped[float] = mapped_column(Float)
    z_score: Mapped[Optional[float]] = mapped_column(Float)
    percentile: Mapped[Optional[float]] = mapped_column(Float)
    confidence_lower: Mapped[Optional[float]] = mapped_column(Float)
    confidence_upper: Mapped[Optional[float]] = mapped_column(Float)

    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    metric_version: Mapped[str] = mapped_column(String(20))
    baseline_version: Mapped[Optional[str]] = mapped_column(String(20))

    token: Mapped["Token"] = relationship(back_populates="metrics")

    __table_args__ = (
        Index("ix_metrics_token_name_snapshot", "token_id", "metric_name", "snapshot_id"),
    )


class BaselineDataset(Base):
    """Reference baseline dataset for normalization."""

    __tablename__ = "baseline_datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(20), unique=True)
    dataset_class: Mapped[str] = mapped_column(String(50))  # established, rug, blue_chip
    sample_count: Mapped[int] = mapped_column(Integer)

    # Statistics for each metric
    hhi_mean: Mapped[float] = mapped_column(Float)
    hhi_std: Mapped[float] = mapped_column(Float)
    entropy_mean: Mapped[float] = mapped_column(Float)
    entropy_std: Mapped[float] = mapped_column(Float)
    gini_mean: Mapped[float] = mapped_column(Float)
    gini_std: Mapped[float] = mapped_column(Float)
    wdr_mean: Mapped[float] = mapped_column(Float)
    wdr_std: Mapped[float] = mapped_column(Float)
    churn_mean: Mapped[float] = mapped_column(Float)
    churn_std: Mapped[float] = mapped_column(Float)
    coordination_mean: Mapped[float] = mapped_column(Float)
    coordination_std: Mapped[float] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class HazardModel(Base):
    """Trained hazard model parameters."""

    __tablename__ = "hazard_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(20), unique=True)

    # Model coefficients (JSON array)
    beta_coefficients: Mapped[dict] = mapped_column(JSON)
    feature_names: Mapped[list] = mapped_column(JSON)

    # Validation metrics
    concordance_index: Mapped[float] = mapped_column(Float)
    brier_score: Mapped[float] = mapped_column(Float)
    roc_auc: Mapped[float] = mapped_column(Float)

    # Training metadata
    training_samples: Mapped[int] = mapped_column(Integer)
    trained_at: Mapped[datetime] = mapped_column(DateTime)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class AuditLog(Base):
    """Audit log for all system events."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    event_type: Mapped[str] = mapped_column(String(50), index=True)
    entity_type: Mapped[Optional[str]] = mapped_column(String(50))
    entity_id: Mapped[Optional[str]] = mapped_column(String(100))
    details: Mapped[dict] = mapped_column(JSON)
    user_id: Mapped[Optional[str]] = mapped_column(String(100))
    ip_address: Mapped[Optional[str]] = mapped_column(String(50))


# ============================================================================
# Temporal Foundation Models (SHI v2)
# ============================================================================


class MetricSnapshot(Base):
    """Time-series snapshot of all core metrics for a token."""

    __tablename__ = "metric_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)

    # Core distribution metrics
    hhi: Mapped[float] = mapped_column(Float)
    gini: Mapped[float] = mapped_column(Float)
    entropy: Mapped[float] = mapped_column(Float)
    whale_dominance: Mapped[float] = mapped_column(Float)

    # Dynamic metrics
    churn_rate: Mapped[Optional[float]] = mapped_column(Float)
    coordination_score: Mapped[Optional[float]] = mapped_column(Float)

    # Holder structure
    holder_count: Mapped[int] = mapped_column(Integer)
    total_supply: Mapped[int] = mapped_column(BigInteger)

    # Metadata
    snapshot_version: Mapped[str] = mapped_column(String(20), default="2.0.0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_metric_snapshots_token_time", "token_mint", "timestamp"),)


class WalletProfile(Base):
    """Evolving wallet risk/behavior profile with history."""

    __tablename__ = "wallet_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    wallet_address: Mapped[str] = mapped_column(String(44), unique=True, index=True)

    # Current classification
    archetype: Mapped[Optional[str]] = mapped_column(String(50))
    risk_score: Mapped[Optional[float]] = mapped_column(Float)
    anomaly_score: Mapped[Optional[float]] = mapped_column(Float)

    # Temporal metadata
    first_seen: Mapped[datetime] = mapped_column(DateTime)
    last_updated: Mapped[datetime] = mapped_column(DateTime)

    # Profile evolution history
    profile_history: Mapped[list] = mapped_column(JSONB, default=list)

    # Graph embeddings (Node2Vec)
    node2vec_embedding: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Float))
    embedding_version: Mapped[Optional[str]] = mapped_column(String(20))

    # Watch list tracking
    is_watched: Mapped[bool] = mapped_column(Boolean, default=False)
    watch_added_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class HolderRegime(Base):
    """HMM-based holder regime detection."""

    __tablename__ = "holder_regimes"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)

    # Regime classification
    regime: Mapped[str] = mapped_column(String(50))
    # Values: accumulation, distribution, coordinated_accumulation, decay, stable

    # Confidence and metadata
    confidence: Mapped[float] = mapped_column(Float)
    transition_probability: Mapped[Optional[float]] = mapped_column(Float)

    # Supporting evidence (derivatives)
    dhhi_dt: Mapped[Optional[float]] = mapped_column(Float)
    dgini_dt: Mapped[Optional[float]] = mapped_column(Float)
    dchurn_dt: Mapped[Optional[float]] = mapped_column(Float)

    # Model metadata
    model_version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_holder_regimes_token_time", "token_mint", "timestamp"),)


class WalletAlert(Base):
    """Tracking notifications sent for wallet movements."""

    __tablename__ = "wallet_alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    wallet_address: Mapped[str] = mapped_column(String(44), index=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)
    alert_type: Mapped[str] = mapped_column(String(50))
    # Values: whale_movement, regime_change, anomaly_spike, concentration_increase

    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    severity: Mapped[str] = mapped_column(String(20))  # low, medium, high, critical

    # Alert payload
    details: Mapped[dict] = mapped_column(JSONB)

    # Delivery tracking
    sent_to_telegram: Mapped[bool] = mapped_column(Boolean, default=False)
    sent_to_webhook: Mapped[bool] = mapped_column(Boolean, default=False)
    telegram_message_id: Mapped[Optional[str]] = mapped_column(String(100))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AlertConfig(Base):
    """User-specific alert configuration."""

    __tablename__ = "alert_configs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(100), index=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)

    # Thresholds
    whale_movement_threshold: Mapped[float] = mapped_column(Float, default=0.05)
    concentration_increase_threshold: Mapped[float] = mapped_column(Float, default=0.02)
    anomaly_score_threshold: Mapped[float] = mapped_column(Float, default=-0.8)

    # Channels
    telegram_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    webhook_url: Mapped[Optional[str]] = mapped_column(String(500))

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (Index("ix_alert_configs_user_token", "user_id", "token_mint", unique=True),)


# ============================================================================
# Cross-Token Intelligence Models (Sprint 8+)
# ============================================================================


class EntityType:
    """Entity type constants."""

    SYBIL_CLUSTER = "sybil_cluster"
    WHALE_GROUP = "whale_group"
    EXCHANGE = "exchange"
    MARKET_MAKER = "market_maker"
    UNKNOWN = "unknown"


class DetectionMethod:
    """How entity membership was detected."""

    SHARED_FUNDER = "shared_funder"
    TEMPORAL_SYNC = "temporal_sync"
    BEHAVIOR_SIMILARITY = "behavior_similarity"
    MANUAL = "manual"


class ConfidenceLevel:
    """Reputation confidence levels."""

    LOW = "low"  # < 5 tokens analyzed
    MEDIUM = "medium"  # 5-20 tokens
    HIGH = "high"  # > 20 tokens


class WalletHistory(Base):
    """
    Track wallet behavior across multiple tokens.

    Records archetype, PnL, holding duration for each wallet-token interaction.
    Enables cross-token pattern detection (serial snipers, diamond hands, etc.)
    """

    __tablename__ = "wallet_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    wallet_address: Mapped[str] = mapped_column(String(44), index=True)
    token_mint: Mapped[str] = mapped_column(String(44), index=True)

    # Temporal data
    first_seen_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    holding_duration_days: Mapped[Optional[int]] = mapped_column(Integer)

    # Classification at time of analysis
    archetype_assigned: Mapped[Optional[str]] = mapped_column(String(50))
    archetype_confidence: Mapped[Optional[float]] = mapped_column(Float)

    # Price and PnL data
    entry_price_usd: Mapped[Optional[float]] = mapped_column(Float)
    exit_price_usd: Mapped[Optional[float]] = mapped_column(Float)
    realized_pnl_pct: Mapped[Optional[float]] = mapped_column(Float)

    # Behavior metrics
    max_balance: Mapped[Optional[int]] = mapped_column(BigInteger)
    max_share_pct: Mapped[Optional[float]] = mapped_column(Float)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)

    # Pattern flags (for quick filtering)
    was_sniper: Mapped[bool] = mapped_column(Boolean, default=False)
    was_accumulator: Mapped[bool] = mapped_column(Boolean, default=False)
    was_early_exit: Mapped[bool] = mapped_column(Boolean, default=False)
    token_rugged: Mapped[bool] = mapped_column(Boolean, default=False)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("wallet_address", "token_mint", name="uq_wallet_history_wallet_token"),
        Index("ix_wallet_history_wallet", "wallet_address"),
        Index("ix_wallet_history_token", "token_mint"),
        Index("ix_wallet_history_archetype", "archetype_assigned"),
        Index("ix_wallet_history_sniper", "was_sniper"),
    )


class Entity(Base):
    """
    Entity grouping for related wallets.

    Groups wallets that share funders, act in coordination, or exhibit
    similar behavior patterns. Used for Sybil detection and risk aggregation.
    """

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(50), index=True)  # EntityType values
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Primary detection info
    dominant_funder_address: Mapped[Optional[str]] = mapped_column(String(44))
    detection_method: Mapped[str] = mapped_column(String(50))  # DetectionMethod values

    # Aggregated stats (updated periodically)
    wallet_count: Mapped[int] = mapped_column(Integer, default=0)
    tokens_targeted: Mapped[int] = mapped_column(Integer, default=0)
    total_volume_usd: Mapped[Optional[float]] = mapped_column(Float)
    avg_coordination_score: Mapped[Optional[float]] = mapped_column(Float)

    # Risk assessment
    is_professional_sybil: Mapped[bool] = mapped_column(Boolean, default=False)
    risk_level: Mapped[Optional[str]] = mapped_column(String(20))  # low, medium, high, critical

    # Metadata
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    memberships: Mapped[list["EntityMembership"]] = relationship(
        back_populates="entity", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_entities_type", "entity_type"),
        Index("ix_entities_funder", "dominant_funder_address"),
        Index("ix_entities_professional", "is_professional_sybil"),
    )


class EntityMembership(Base):
    """
    Junction table linking wallets to entities.

    Tracks how each wallet was detected as part of an entity
    and the confidence of that membership.
    """

    __tablename__ = "entity_memberships"

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"), index=True)
    wallet_address: Mapped[str] = mapped_column(String(44), index=True)

    # Detection info
    membership_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    detected_via: Mapped[str] = mapped_column(String(50))  # DetectionMethod values

    # Supporting evidence
    shared_funder_address: Mapped[Optional[str]] = mapped_column(String(44))
    temporal_correlation: Mapped[Optional[float]] = mapped_column(Float)
    behavior_similarity: Mapped[Optional[float]] = mapped_column(Float)

    # Metadata
    added_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    # Relationships
    entity: Mapped["Entity"] = relationship(back_populates="memberships")

    __table_args__ = (
        UniqueConstraint("entity_id", "wallet_address", name="uq_entity_membership"),
        Index("ix_entity_membership_wallet", "wallet_address"),
        Index("ix_entity_membership_entity", "entity_id"),
    )


class WalletReputation(Base):
    """
    Cross-token reputation score for wallets.

    Aggregates behavior across all analyzed tokens to build a
    reputation score and detect patterns (serial sniper, diamond hands, etc.)
    """

    __tablename__ = "wallet_reputation"

    wallet_address: Mapped[str] = mapped_column(String(44), primary_key=True)

    # Core reputation score (0-100)
    reputation_score: Mapped[int] = mapped_column(Integer, default=50)
    confidence_level: Mapped[str] = mapped_column(String(10), default=ConfidenceLevel.LOW)

    # Token interaction counts
    tokens_analyzed: Mapped[int] = mapped_column(Integer, default=0)
    sniper_count: Mapped[int] = mapped_column(Integer, default=0)
    accumulator_count: Mapped[int] = mapped_column(Integer, default=0)
    rugpull_count: Mapped[int] = mapped_column(Integer, default=0)
    early_exit_count: Mapped[int] = mapped_column(Integer, default=0)

    # Aggregated metrics
    avg_holding_days: Mapped[Optional[float]] = mapped_column(Float)
    avg_pnl_pct: Mapped[Optional[float]] = mapped_column(Float)
    total_volume_usd: Mapped[Optional[float]] = mapped_column(Float)

    # Detected patterns (JSONB array of pattern objects)
    # Format: [{"type": "SERIAL_SNIPER", "confidence": 0.85, "token_count": 5}, ...]
    patterns: Mapped[list] = mapped_column(JSONB, default=list)

    # Risk flags
    is_known_bad_actor: Mapped[bool] = mapped_column(Boolean, default=False)
    is_known_good_actor: Mapped[bool] = mapped_column(Boolean, default=False)

    # Entity link (if part of a detected entity)
    entity_id: Mapped[Optional[int]] = mapped_column(ForeignKey("entities.id", ondelete="SET NULL"))

    # Metadata
    first_analyzed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    last_updated: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_wallet_reputation_score", "reputation_score"),
        Index("ix_wallet_reputation_confidence", "confidence_level"),
        Index("ix_wallet_reputation_bad_actor", "is_known_bad_actor"),
    )
