"""Add cross-token intelligence tables

Revision ID: 005
Revises: 004
Create Date: 2026-05-12

Sprint 8: Data Foundation for cross-token wallet analysis.
- wallet_history: Track wallet behavior per token
- entities: Group related wallets (sybil clusters, whale groups)
- entity_memberships: Wallet-to-entity mapping
- wallet_reputation: Cross-token reputation scores
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add cross-token intelligence tables."""

    # =========================================================================
    # wallet_history - Track wallet behavior across multiple tokens
    # =========================================================================
    op.create_table(
        "wallet_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("wallet_address", sa.String(44), nullable=False),
        sa.Column("token_mint", sa.String(44), nullable=False),

        # Temporal data
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("holding_duration_days", sa.Integer(), nullable=True),

        # Classification at time of analysis
        sa.Column("archetype_assigned", sa.String(50), nullable=True),
        sa.Column("archetype_confidence", sa.Float(), nullable=True),

        # Price and PnL data
        sa.Column("entry_price_usd", sa.Float(), nullable=True),
        sa.Column("exit_price_usd", sa.Float(), nullable=True),
        sa.Column("realized_pnl_pct", sa.Float(), nullable=True),

        # Behavior metrics
        sa.Column("max_balance", sa.BigInteger(), nullable=True),
        sa.Column("max_share_pct", sa.Float(), nullable=True),
        sa.Column("trade_count", sa.Integer(), default=0),

        # Pattern flags (for quick filtering)
        sa.Column("was_sniper", sa.Boolean(), default=False),
        sa.Column("was_accumulator", sa.Boolean(), default=False),
        sa.Column("was_early_exit", sa.Boolean(), default=False),
        sa.Column("token_rugged", sa.Boolean(), default=False),

        # Metadata
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),

        # Constraints
        sa.UniqueConstraint("wallet_address", "token_mint", name="uq_wallet_history_wallet_token"),
    )

    # Indexes for wallet_history
    op.create_index("ix_wallet_history_wallet", "wallet_history", ["wallet_address"])
    op.create_index("ix_wallet_history_token", "wallet_history", ["token_mint"])
    op.create_index("ix_wallet_history_archetype", "wallet_history", ["archetype_assigned"])
    op.create_index("ix_wallet_history_sniper", "wallet_history", ["was_sniper"])

    # =========================================================================
    # entities - Group related wallets
    # =========================================================================
    op.create_table(
        "entities",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("entity_type", sa.String(50), nullable=False),
        # Values: sybil_cluster, whale_group, exchange, market_maker, unknown
        sa.Column("confidence_score", sa.Float(), default=0.0),

        # Primary detection info
        sa.Column("dominant_funder_address", sa.String(44), nullable=True),
        sa.Column("detection_method", sa.String(50), nullable=False),
        # Values: shared_funder, temporal_sync, behavior_similarity, manual

        # Aggregated stats
        sa.Column("wallet_count", sa.Integer(), default=0),
        sa.Column("tokens_targeted", sa.Integer(), default=0),
        sa.Column("total_volume_usd", sa.Float(), nullable=True),
        sa.Column("avg_coordination_score", sa.Float(), nullable=True),

        # Risk assessment
        sa.Column("is_professional_sybil", sa.Boolean(), default=False),
        sa.Column("risk_level", sa.String(20), nullable=True),
        # Values: low, medium, high, critical

        # Metadata
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Indexes for entities
    op.create_index("ix_entities_type", "entities", ["entity_type"])
    op.create_index("ix_entities_funder", "entities", ["dominant_funder_address"])
    op.create_index("ix_entities_professional", "entities", ["is_professional_sybil"])

    # =========================================================================
    # entity_memberships - Wallet-to-entity mapping
    # =========================================================================
    op.create_table(
        "entity_memberships",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "entity_id",
            sa.Integer(),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("wallet_address", sa.String(44), nullable=False),

        # Detection info
        sa.Column("membership_confidence", sa.Float(), default=0.0),
        sa.Column("detected_via", sa.String(50), nullable=False),
        # Values: shared_funder, temporal_sync, behavior_similarity, manual

        # Supporting evidence
        sa.Column("shared_funder_address", sa.String(44), nullable=True),
        sa.Column("temporal_correlation", sa.Float(), nullable=True),
        sa.Column("behavior_similarity", sa.Float(), nullable=True),

        # Metadata
        sa.Column("added_at", sa.DateTime(timezone=True), server_default=sa.func.now()),

        # Constraints
        sa.UniqueConstraint("entity_id", "wallet_address", name="uq_entity_membership"),
    )

    # Indexes for entity_memberships
    op.create_index("ix_entity_membership_wallet", "entity_memberships", ["wallet_address"])
    op.create_index("ix_entity_membership_entity", "entity_memberships", ["entity_id"])

    # =========================================================================
    # wallet_reputation - Cross-token reputation scores
    # =========================================================================
    op.create_table(
        "wallet_reputation",
        sa.Column("wallet_address", sa.String(44), primary_key=True),

        # Core reputation score (0-100)
        sa.Column("reputation_score", sa.Integer(), default=50),
        sa.Column("confidence_level", sa.String(10), default="low"),
        # Values: low (<5 tokens), medium (5-20), high (>20)

        # Token interaction counts
        sa.Column("tokens_analyzed", sa.Integer(), default=0),
        sa.Column("sniper_count", sa.Integer(), default=0),
        sa.Column("accumulator_count", sa.Integer(), default=0),
        sa.Column("rugpull_count", sa.Integer(), default=0),
        sa.Column("early_exit_count", sa.Integer(), default=0),

        # Aggregated metrics
        sa.Column("avg_holding_days", sa.Float(), nullable=True),
        sa.Column("avg_pnl_pct", sa.Float(), nullable=True),
        sa.Column("total_volume_usd", sa.Float(), nullable=True),

        # Detected patterns (JSONB)
        # Format: [{"type": "SERIAL_SNIPER", "confidence": 0.85, "token_count": 5}, ...]
        sa.Column("patterns", sa.JSON(), default=list),

        # Risk flags
        sa.Column("is_known_bad_actor", sa.Boolean(), default=False),
        sa.Column("is_known_good_actor", sa.Boolean(), default=False),

        # Entity link
        sa.Column(
            "entity_id",
            sa.Integer(),
            sa.ForeignKey("entities.id", ondelete="SET NULL"),
            nullable=True,
        ),

        # Metadata
        sa.Column("first_analyzed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_updated", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Indexes for wallet_reputation
    op.create_index("ix_wallet_reputation_score", "wallet_reputation", ["reputation_score"])
    op.create_index("ix_wallet_reputation_confidence", "wallet_reputation", ["confidence_level"])
    op.create_index("ix_wallet_reputation_bad_actor", "wallet_reputation", ["is_known_bad_actor"])


def downgrade() -> None:
    """Remove cross-token intelligence tables."""
    op.drop_table("wallet_reputation")
    op.drop_table("entity_memberships")
    op.drop_table("entities")
    op.drop_table("wallet_history")
