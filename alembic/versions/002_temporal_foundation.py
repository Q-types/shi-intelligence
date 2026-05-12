"""Add temporal foundation - metric snapshots and wallet profiles

Revision ID: 002
Revises: 001
Create Date: 2026-05-07
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
# sa.JSON() removed for SQLite compatibility


revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add temporal tracking tables for SHI v2."""

    # Metric snapshots - time series of all core metrics
    op.create_table(
        "metric_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("token_mint", sa.String(44), nullable=False, index=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),

        # Core distribution metrics
        sa.Column("hhi", sa.Float(), nullable=False),
        sa.Column("gini", sa.Float(), nullable=False),
        sa.Column("entropy", sa.Float(), nullable=False),
        sa.Column("whale_dominance", sa.Float(), nullable=False),

        # Dynamic metrics
        sa.Column("churn_rate", sa.Float(), nullable=True),
        sa.Column("coordination_score", sa.Float(), nullable=True),

        # Holder structure metrics
        sa.Column("holder_count", sa.Integer(), nullable=False),
        sa.Column("total_supply", sa.BigInteger(), nullable=False),

        # Metadata
        sa.Column("snapshot_version", sa.String(20), default="2.0.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Composite index for efficient time-series queries
    op.create_index(
        "ix_metric_snapshots_token_time",
        "metric_snapshots",
        ["token_mint", "timestamp"],
    )

    # Wallet profiles - evolving wallet risk/behavior profiles
    op.create_table(
        "wallet_profiles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("wallet_address", sa.String(44), nullable=False, unique=True, index=True),

        # Current classification
        sa.Column("archetype", sa.String(50), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=True),
        sa.Column("anomaly_score", sa.Float(), nullable=True),

        # Temporal metadata
        sa.Column("first_seen", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_updated", sa.DateTime(timezone=True), nullable=False),

        # Profile evolution history (sa.JSON() for flexibility)
        sa.Column("profile_history", sa.JSON(), default=list),
        # Structure: [{"timestamp": "2026-05-07T...", "archetype": "sniper", "risk_score": 0.8, ...}]

        # Graph embeddings (will be populated by graph-ml agent)
        sa.Column("node2vec_embedding", sa.ARRAY(sa.Float()), nullable=True),
        sa.Column("embedding_version", sa.String(20), nullable=True),

        # Watch list tracking
        sa.Column("is_watched", sa.Boolean(), default=False),
        sa.Column("watch_added_at", sa.DateTime(timezone=True), nullable=True),

        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Regime states - HMM-based holder regime tracking
    op.create_table(
        "holder_regimes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("token_mint", sa.String(44), nullable=False, index=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),

        # Regime classification
        sa.Column("regime", sa.String(50), nullable=False),
        # Values: accumulation, distribution, coordinated_accumulation, decay, stable

        # Confidence and metadata
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("transition_probability", sa.Float(), nullable=True),

        # Supporting evidence
        sa.Column("dhhi_dt", sa.Float(), nullable=True),  # HHI derivative
        sa.Column("dgini_dt", sa.Float(), nullable=True),  # Gini derivative
        sa.Column("dchurn_dt", sa.Float(), nullable=True),  # Churn derivative

        # Model metadata
        sa.Column("model_version", sa.String(20), default="1.0.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_index(
        "ix_holder_regimes_token_time",
        "holder_regimes",
        ["token_mint", "timestamp"],
    )

    # Wallet alerts - tracking notifications sent
    op.create_table(
        "wallet_alerts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("wallet_address", sa.String(44), nullable=False, index=True),
        sa.Column("token_mint", sa.String(44), nullable=False, index=True),
        sa.Column("alert_type", sa.String(50), nullable=False),
        # Values: whale_movement, regime_change, anomaly_spike, concentration_increase

        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("severity", sa.String(20), nullable=False),  # low, medium, high, critical

        # Alert payload
        sa.Column("details", sa.JSON(), nullable=False),
        # Structure: {"amount_usd": 1000000, "pct_of_supply": 0.05, "description": "..."}

        # Delivery tracking
        sa.Column("sent_to_telegram", sa.Boolean(), default=False),
        sa.Column("sent_to_webhook", sa.Boolean(), default=False),
        sa.Column("telegram_message_id", sa.String(100), nullable=True),

        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # User alert configurations
    op.create_table(
        "alert_configs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.String(100), nullable=False, index=True),
        sa.Column("token_mint", sa.String(44), nullable=False, index=True),

        # Thresholds
        sa.Column("whale_movement_threshold", sa.Float(), default=0.05),  # 5% of supply
        sa.Column("concentration_increase_threshold", sa.Float(), default=0.02),  # 2% HHI change
        sa.Column("anomaly_score_threshold", sa.Float(), default=-0.8),

        # Channels
        sa.Column("telegram_enabled", sa.Boolean(), default=True),
        sa.Column("webhook_url", sa.String(500), nullable=True),

        # Metadata
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    op.create_index(
        "ix_alert_configs_user_token",
        "alert_configs",
        ["user_id", "token_mint"],
        unique=True,
    )


def downgrade() -> None:
    """Remove temporal foundation tables."""
    op.drop_table("alert_configs")
    op.drop_table("wallet_alerts")
    op.drop_table("holder_regimes")
    op.drop_table("wallet_profiles")
    op.drop_table("metric_snapshots")
