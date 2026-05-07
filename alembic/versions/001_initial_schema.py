"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-02-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Tokens table
    op.create_table(
        "tokens",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("mint", sa.String(44), unique=True, index=True, nullable=False),
        sa.Column("name", sa.String(100)),
        sa.Column("symbol", sa.String(20)),
        sa.Column("decimals", sa.Integer(), default=9),
        sa.Column("total_supply", sa.BigInteger()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Wallets table
    op.create_table(
        "wallets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("address", sa.String(44), unique=True, index=True, nullable=False),
        sa.Column("funded_by_id", sa.Integer(), sa.ForeignKey("wallets.id")),
        sa.Column("first_funded_at", sa.DateTime()),
        sa.Column("first_seen_at", sa.DateTime(), nullable=False),
        sa.Column("is_exchange", sa.Boolean(), default=False),
        sa.Column("is_contract", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # Funding edges table
    op.create_table(
        "funding_edges",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("wallets.id"), index=True),
        sa.Column("target_id", sa.Integer(), sa.ForeignKey("wallets.id"), index=True),
        sa.Column("amount_lamports", sa.BigInteger()),
        sa.Column("timestamp", sa.DateTime(), index=True),
        sa.Column("signature", sa.String(88)),
    )
    op.create_index("ix_funding_edges_source_target", "funding_edges", ["source_id", "target_id"])

    # Holder snapshots table
    op.create_table(
        "holder_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("token_id", sa.Integer(), sa.ForeignKey("tokens.id"), index=True),
        sa.Column("timestamp", sa.DateTime(), index=True),
        sa.Column("holder_count", sa.Integer()),
        sa.Column("total_supply", sa.BigInteger()),
        sa.Column("checksum", sa.String(64)),
    )

    # Balances table
    op.create_table(
        "balances",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("snapshot_id", sa.Integer(), sa.ForeignKey("holder_snapshots.id"), index=True),
        sa.Column("wallet_id", sa.Integer(), sa.ForeignKey("wallets.id"), index=True),
        sa.Column("balance", sa.BigInteger()),
        sa.Column("share", sa.Float()),
        sa.Column("rank", sa.Integer()),
    )
    op.create_index("ix_balances_snapshot_wallet", "balances", ["snapshot_id", "wallet_id"])

    # Wallet features table
    op.create_table(
        "wallet_features",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("wallet_id", sa.Integer(), sa.ForeignKey("wallets.id"), index=True),
        sa.Column("token_id", sa.Integer(), sa.ForeignKey("tokens.id"), index=True),
        sa.Column("snapshot_id", sa.Integer(), sa.ForeignKey("holder_snapshots.id")),
        sa.Column("balance", sa.BigInteger()),
        sa.Column("share", sa.Float()),
        sa.Column("rank", sa.Integer()),
        sa.Column("entry_time_relative", sa.Float()),
        sa.Column("holding_duration", sa.Float()),
        sa.Column("position_volatility", sa.Float()),
        sa.Column("delta_balance_7d", sa.Float()),
        sa.Column("delta_balance_30d", sa.Float()),
        sa.Column("trade_count", sa.Integer(), default=0),
        sa.Column("burstiness", sa.Float(), default=0.0),
        sa.Column("swap_frequency", sa.Float(), default=0.0),
        sa.Column("lp_interaction_ratio", sa.Float(), default=0.0),
        sa.Column("in_degree", sa.Integer(), default=0),
        sa.Column("out_degree", sa.Integer(), default=0),
        sa.Column("eigenvector_centrality", sa.Float(), default=0.0),
        sa.Column("shared_funder_count", sa.Integer(), default=0),
        sa.Column("computed_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("wallet_id", "token_id", "snapshot_id", name="uq_wallet_token_snapshot"),
    )
    op.create_index("ix_wallet_features_token_snapshot", "wallet_features", ["token_id", "snapshot_id"])

    # Archetype assignments table
    op.create_table(
        "archetype_assignments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("wallet_id", sa.Integer(), sa.ForeignKey("wallets.id"), index=True),
        sa.Column("token_id", sa.Integer(), sa.ForeignKey("tokens.id"), index=True),
        sa.Column("snapshot_id", sa.Integer(), sa.ForeignKey("holder_snapshots.id")),
        sa.Column("archetype", sa.String(50)),
        sa.Column("confidence", sa.Float()),
        sa.Column("features_used", sa.JSON()),
        sa.Column("cluster_id", sa.Integer()),
        sa.Column("computed_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("model_version", sa.String(20)),
    )

    # Metrics table
    op.create_table(
        "metrics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("token_id", sa.Integer(), sa.ForeignKey("tokens.id"), index=True),
        sa.Column("snapshot_id", sa.Integer(), sa.ForeignKey("holder_snapshots.id")),
        sa.Column("metric_name", sa.String(50), index=True),
        sa.Column("value", sa.Float()),
        sa.Column("z_score", sa.Float()),
        sa.Column("percentile", sa.Float()),
        sa.Column("confidence_lower", sa.Float()),
        sa.Column("confidence_upper", sa.Float()),
        sa.Column("computed_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("metric_version", sa.String(20)),
        sa.Column("baseline_version", sa.String(20)),
    )
    op.create_index("ix_metrics_token_name_snapshot", "metrics", ["token_id", "metric_name", "snapshot_id"])

    # Baseline datasets table
    op.create_table(
        "baseline_datasets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("version", sa.String(20), unique=True),
        sa.Column("dataset_class", sa.String(50)),
        sa.Column("sample_count", sa.Integer()),
        sa.Column("hhi_mean", sa.Float()),
        sa.Column("hhi_std", sa.Float()),
        sa.Column("entropy_mean", sa.Float()),
        sa.Column("entropy_std", sa.Float()),
        sa.Column("gini_mean", sa.Float()),
        sa.Column("gini_std", sa.Float()),
        sa.Column("wdr_mean", sa.Float()),
        sa.Column("wdr_std", sa.Float()),
        sa.Column("churn_mean", sa.Float()),
        sa.Column("churn_std", sa.Float()),
        sa.Column("coordination_mean", sa.Float()),
        sa.Column("coordination_std", sa.Float()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("is_active", sa.Boolean(), default=True),
    )

    # Hazard models table
    op.create_table(
        "hazard_models",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("version", sa.String(20), unique=True),
        sa.Column("beta_coefficients", sa.JSON()),
        sa.Column("feature_names", sa.JSON()),
        sa.Column("concordance_index", sa.Float()),
        sa.Column("brier_score", sa.Float()),
        sa.Column("roc_auc", sa.Float()),
        sa.Column("training_samples", sa.Integer()),
        sa.Column("trained_at", sa.DateTime()),
        sa.Column("is_active", sa.Boolean(), default=True),
    )

    # Audit logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(), server_default=sa.func.now(), index=True),
        sa.Column("event_type", sa.String(50), index=True),
        sa.Column("entity_type", sa.String(50)),
        sa.Column("entity_id", sa.String(100)),
        sa.Column("details", sa.JSON()),
        sa.Column("user_id", sa.String(100)),
        sa.Column("ip_address", sa.String(50)),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("hazard_models")
    op.drop_table("baseline_datasets")
    op.drop_table("metrics")
    op.drop_table("archetype_assignments")
    op.drop_table("wallet_features")
    op.drop_table("balances")
    op.drop_table("holder_snapshots")
    op.drop_table("funding_edges")
    op.drop_table("wallets")
    op.drop_table("tokens")
