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
        env_prefix="SHI_",
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
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")


settings = Settings()
