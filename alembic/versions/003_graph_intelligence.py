"""Graph Intelligence Tables - Sprint 2

Adds tables for:
- Graph embeddings (Node2Vec)
- Wallet similarity scores
- Network snapshots (dynamic metrics)
- Anomaly scores

Revision ID: 003_graph_intelligence
Revises: 002_temporal_foundation
Create Date: 2026-05-07

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY


# revision identifiers, used by Alembic.
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add graph intelligence tables."""

    # Table: wallet_embeddings
    # Stores Node2Vec embeddings for wallets
    op.create_table(
        "wallet_embeddings",
        sa.Column("wallet_address", sa.Text, primary_key=True),
        sa.Column("embedding_id", sa.Text, nullable=False),
        sa.Column("vector", ARRAY(sa.Float), nullable=False),
        sa.Column("dimensions", sa.Integer, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
    )

    # Indexes for embeddings
    op.create_index(
        "idx_wallet_embeddings_embedding_id",
        "wallet_embeddings",
        ["embedding_id"],
    )
    op.create_index(
        "idx_wallet_embeddings_created_at",
        "wallet_embeddings",
        ["created_at"],
    )

    # Table: wallet_similarities
    # Stores pairwise similarity scores
    op.create_table(
        "wallet_similarities",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("wallet1", sa.Text, nullable=False),
        sa.Column("wallet2", sa.Text, nullable=False),
        sa.Column("embedding_similarity", sa.Float, nullable=False),
        sa.Column("structural_similarity", sa.Float, nullable=False),
        sa.Column("combined_similarity", sa.Float, nullable=False),
        sa.Column("is_coordinated", sa.Boolean, nullable=False),
        sa.Column("computed_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )

    # Composite index for wallet pairs
    op.create_index(
        "idx_wallet_similarities_pair",
        "wallet_similarities",
        ["wallet1", "wallet2"],
        unique=True,
    )
    op.create_index(
        "idx_wallet_similarities_coordinated",
        "wallet_similarities",
        ["is_coordinated", "combined_similarity"],
    )

    # Table: coordinated_clusters
    # Stores detected Sybil clusters
    op.create_table(
        "coordinated_clusters",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("cluster_id", sa.Integer, nullable=False),
        sa.Column("token_mint", sa.Text, nullable=True),
        sa.Column("wallets", ARRAY(sa.Text), nullable=False),
        sa.Column("mean_similarity", sa.Float, nullable=False),
        sa.Column("sybil_probability", sa.Float, nullable=False),
        sa.Column("shared_funders", ARRAY(sa.Text), nullable=True),
        sa.Column("detected_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
    )

    op.create_index(
        "idx_coordinated_clusters_token",
        "coordinated_clusters",
        ["token_mint", "detected_at"],
    )
    op.create_index(
        "idx_coordinated_clusters_sybil_prob",
        "coordinated_clusters",
        ["sybil_probability"],
    )

    # Table: network_snapshots
    # Time-series of network metrics
    op.create_table(
        "network_snapshots",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("token_mint", sa.Text, nullable=True),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("num_nodes", sa.Integer, nullable=False),
        sa.Column("num_edges", sa.Integer, nullable=False),
        sa.Column("density", sa.Float, nullable=False),
        sa.Column("modularity", sa.Float, nullable=False),
        sa.Column("centralization", sa.Float, nullable=False),
        sa.Column("avg_clustering_coefficient", sa.Float, nullable=False),
        sa.Column("num_communities", sa.Integer, nullable=False),
        sa.Column("largest_component_size", sa.Integer, nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
    )

    op.create_index(
        "idx_network_snapshots_token_time",
        "network_snapshots",
        ["token_mint", "timestamp"],
    )
    op.create_index(
        "idx_network_snapshots_timestamp",
        "network_snapshots",
        ["timestamp"],
    )

    # Table: wallet_anomaly_scores
    # Anomaly scores from Isolation Forest
    op.create_table(
        "wallet_anomaly_scores",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("wallet_address", sa.Text, nullable=False),
        sa.Column("token_mint", sa.Text, nullable=True),
        sa.Column("anomaly_score", sa.Float, nullable=False),
        sa.Column("is_anomalous", sa.Boolean, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("feature_contributions", sa.JSON(), nullable=True),
        sa.Column("computed_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )

    op.create_index(
        "idx_wallet_anomaly_scores_wallet",
        "wallet_anomaly_scores",
        ["wallet_address", "computed_at"],
    )
    op.create_index(
        "idx_wallet_anomaly_scores_anomalous",
        "wallet_anomaly_scores",
        ["is_anomalous", "anomaly_score"],
    )
    op.create_index(
        "idx_wallet_anomaly_scores_token",
        "wallet_anomaly_scores",
        ["token_mint", "computed_at"],
    )

    # Table: community_events
    # Tracks community emergence/fragmentation
    op.create_table(
        "community_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("token_mint", sa.Text, nullable=True),
        sa.Column("event_type", sa.Text, nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("communities_before", sa.Integer, nullable=True),
        sa.Column("communities_after", sa.Integer, nullable=True),
        sa.Column("density_change", sa.Float, nullable=True),
        sa.Column("modularity_change", sa.Float, nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
    )

    op.create_index(
        "idx_community_events_token_time",
        "community_events",
        ["token_mint", "timestamp"],
    )
    op.create_index(
        "idx_community_events_type",
        "community_events",
        ["event_type", "timestamp"],
    )


def downgrade() -> None:
    """Drop graph intelligence tables."""

    op.drop_table("community_events")
    op.drop_table("wallet_anomaly_scores")
    op.drop_table("network_snapshots")
    op.drop_table("coordinated_clusters")
    op.drop_table("wallet_similarities")
    op.drop_table("wallet_embeddings")
