"""Add monitoring tables - watchlist and profile snapshots

Revision ID: 004
Revises: 003
Create Date: 2026-05-07
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMPTZ


revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add monitoring-specific tables for SHI v2 Sprint 3."""

    # Wallet watchlist - tracks which wallets users are monitoring
    op.create_table(
        "wallet_watchlist",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.String(100), nullable=False, index=True),
        sa.Column("wallet_address", sa.String(44), nullable=False, index=True),
        sa.Column("token_mint", sa.String(44), nullable=False, index=True),

        # Configuration
        sa.Column("alert_threshold", sa.Float(), default=0.05),  # 5% of supply
        sa.Column("enabled", sa.Boolean(), default=True),

        # Tracking
        sa.Column("last_balance", sa.Float(), nullable=True),
        sa.Column("last_checked", TIMESTAMPTZ, nullable=True),

        # Metadata
        sa.Column("added_at", TIMESTAMPTZ, server_default=sa.func.now()),
        sa.Column("notes", sa.Text(), nullable=True),

        sa.Column("created_at", TIMESTAMPTZ, server_default=sa.func.now()),
    )

    # Composite index for user watchlist queries
    op.create_index(
        "ix_watchlist_user_token",
        "wallet_watchlist",
        ["user_id", "token_mint"],
    )

    # Unique constraint on user-wallet-token combination
    op.create_index(
        "ix_watchlist_unique",
        "wallet_watchlist",
        ["user_id", "wallet_address", "token_mint"],
        unique=True,
    )

    # Profile snapshots - detailed historical snapshots
    # (wallet_profiles.profile_history stores compact JSONB,
    #  this stores full detailed snapshots for analysis)
    op.create_table(
        "profile_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("wallet_address", sa.String(44), nullable=False, index=True),
        sa.Column("timestamp", TIMESTAMPTZ, nullable=False, index=True),

        # Profile data
        sa.Column("archetype", sa.String(50), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=True),
        sa.Column("anomaly_score", sa.Float(), nullable=True),

        # Features (for ML model inspection)
        sa.Column("features", JSONB, nullable=True),
        # Structure: {"in_degree": 5, "out_degree": 3, "funding_ratio": 0.6, ...}

        # Graph metrics
        sa.Column("centrality", sa.Float(), nullable=True),
        sa.Column("clustering_coefficient", sa.Float(), nullable=True),
        sa.Column("community_id", sa.Integer(), nullable=True),

        # Temporal metrics
        sa.Column("activity_score", sa.Float(), nullable=True),
        sa.Column("days_since_first_seen", sa.Integer(), nullable=True),

        # Metadata
        sa.Column("snapshot_version", sa.String(20), default="1.0.0"),
        sa.Column("created_at", TIMESTAMPTZ, server_default=sa.func.now()),
    )

    # Composite index for efficient time-series queries
    op.create_index(
        "ix_profile_snapshots_wallet_time",
        "profile_snapshots",
        ["wallet_address", "timestamp"],
    )

    # Alert delivery log - tracks delivery attempts
    op.create_table(
        "alert_delivery_log",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("alert_id", sa.Integer(), nullable=False, index=True),
        # References wallet_alerts.id

        sa.Column("delivery_method", sa.String(20), nullable=False),
        # Values: telegram, webhook, email

        sa.Column("attempted_at", TIMESTAMPTZ, nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),

        # Delivery details
        sa.Column("response_code", sa.Integer(), nullable=True),
        sa.Column("response_body", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),

        # Retry tracking
        sa.Column("retry_count", sa.Integer(), default=0),
        sa.Column("next_retry_at", TIMESTAMPTZ, nullable=True),

        sa.Column("created_at", TIMESTAMPTZ, server_default=sa.func.now()),
    )

    op.create_index(
        "ix_alert_delivery_alert_id",
        "alert_delivery_log",
        ["alert_id"],
    )

    # User notification preferences
    op.create_table(
        "user_notification_preferences",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.String(100), nullable=False, unique=True, index=True),

        # Telegram
        sa.Column("telegram_chat_id", sa.String(100), nullable=True),
        sa.Column("telegram_enabled", sa.Boolean(), default=True),

        # Webhooks
        sa.Column("webhook_url", sa.String(500), nullable=True),
        sa.Column("webhook_enabled", sa.Boolean(), default=False),

        # Email (future)
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("email_enabled", sa.Boolean(), default=False),

        # Quiet hours (UTC)
        sa.Column("quiet_hours_start", sa.Integer(), nullable=True),  # 0-23
        sa.Column("quiet_hours_end", sa.Integer(), nullable=True),  # 0-23

        # Global settings
        sa.Column("max_alerts_per_hour", sa.Integer(), default=10),
        sa.Column("batch_alerts", sa.Boolean(), default=False),

        sa.Column("created_at", TIMESTAMPTZ, server_default=sa.func.now()),
        sa.Column("updated_at", TIMESTAMPTZ, server_default=sa.func.now(), onupdate=sa.func.now()),
    )


def downgrade() -> None:
    """Remove monitoring tables."""
    op.drop_table("user_notification_preferences")
    op.drop_table("alert_delivery_log")
    op.drop_table("profile_snapshots")
    op.drop_table("wallet_watchlist")
