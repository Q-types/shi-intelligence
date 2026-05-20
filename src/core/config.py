"""
Configuration settings for SHI.

Environment-based configuration with validation.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # No prefix - read HELIUS_API_KEY, DATABASE_URL, etc. directly
        case_sensitive=False,
    )

    # Data Sources
    helius_api_key: str = Field(default="", description="Helius API key")
    helius_rpc_url: str = Field(
        default="https://mainnet.helius-rpc.com",
        description="Helius RPC endpoint",
    )
    solana_rpc_url: str = Field(
        default="https://api.mainnet-beta.solana.com",
        description="Fallback Solana RPC endpoint",
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://localhost/shi",
        description="PostgreSQL connection URL",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # Telegram
    telegram_bot_token: str = Field(default="", description="Telegram bot token")
    telegram_rate_limit_per_user: int = Field(
        default=10,
        description="Max requests per user per minute",
    )
    telegram_rate_limit_global: int = Field(
        default=100,
        description="Max global requests per minute",
    )
    admin_user_ids: str = Field(default="", description="Comma-separated admin user IDs")
    premium_user_ids: str = Field(default="", description="Comma-separated premium user IDs")

    # Processing
    max_holders_per_token: int = Field(
        default=50000,
        description="Max holders to process (sampling for larger sets)",
    )
    sla_timeout_seconds: int = Field(
        default=30,
        description="Target response time for typical tokens",
    )

    # Hazard Model
    sell_event_threshold_pct: float = Field(
        default=0.5,
        description="Sell event = reduction >= X% of peak balance",
    )
    sell_event_horizon_days: int = Field(
        default=7,
        description="Time horizon T for sell probability",
    )

    # Baseline
    baseline_version: str = Field(
        default="v1.0.0",
        description="Current baseline dataset version",
    )

    # Monitoring
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")

    # Price Data - Jupiter V3 API
    # Free tier: lite-api.jup.ag, Pro tier: api.jup.ag (requires API key)
    jupiter_price_base_url: str = Field(
        default="https://lite-api.jup.ag",
        description="Jupiter Price API base URL (lite-api.jup.ag=free, api.jup.ag=pro)",
    )
    jupiter_price_path: str = Field(
        default="/price/v3",
        description="Jupiter Price API path",
    )
    jupiter_api_key: str | None = Field(
        default=None,
        description="Jupiter API key for pro tier (api.jup.ag)",
    )
    price_cache_ttl_seconds: int = Field(
        default=60,
        description="Cache TTL for price data in seconds",
    )
    enable_price_features: bool = Field(
        default=True,
        description="Enable price-based features in analysis",
    )
    enable_price_persistence: bool = Field(
        default=True,
        description="Persist price snapshots for historical analysis",
    )

    # Fallback Price Providers
    birdeye_api_key: str | None = Field(
        default=None,
        description="Birdeye API key for fallback price data",
    )
    enable_pool_implied_price: bool = Field(
        default=True,
        description="Use pool reserves to derive price as fallback",
    )

    # Development
    debug: bool = Field(default=False, description="Enable debug mode")
    use_testnet: bool = Field(default=False, description="Use Solana testnet")

    # Clustering Intelligence Feature Flags
    # Controls which components of the upgraded clustering pipeline are active
    use_robust_clustering: bool = Field(
        default=True,
        description="Use robust transformations (log1p, asinh, RobustScaler) and missingness handling",
    )
    use_node2vec_clustering: bool = Field(
        default=False,
        description="Enable experimental Node2Vec graph embeddings in clustering",
    )
    use_expanded_hazard_features: bool = Field(
        default=False,
        description="Use expanded Cox PH features (price, liquidity, graph centrality)",
    )
    use_missingness_indicators: bool = Field(
        default=True,
        description="Add missingness indicator columns to features",
    )
    use_weighted_graph_features: bool = Field(
        default=True,
        description="Use weighted funding graph features (HHI, burst score, etc.)",
    )
    use_multi_score_archetypes: bool = Field(
        default=True,
        description="Use multi-score archetype assignment (soft labels)",
    )
    use_temporal_validation: bool = Field(
        default=True,
        description="Use temporal/walk-forward CV instead of shuffled KFold",
    )

    # Clustering Parameters
    hdbscan_min_cluster_size: int = Field(
        default=5,
        description="HDBSCAN minimum cluster size",
    )
    node2vec_reduced_dimensions: int = Field(
        default=6,
        description="Reduced dimensions for Node2Vec embeddings (4-8 recommended)",
    )
    node2vec_behavior_weight: float = Field(
        default=0.7,
        description="Weight for behavioral features in combined clustering",
    )
    node2vec_graph_weight: float = Field(
        default=0.3,
        description="Weight for graph embeddings in combined clustering",
    )


settings = Settings()
