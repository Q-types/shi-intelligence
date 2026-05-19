"""Whale Tracker configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Base paths
_BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = _BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "whale_tracker.sqlite"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class TrackerConfig:
    """Configuration for the Whale Tracker dashboard."""

    # Database
    database_path: Path = DATABASE_PATH

    # Token to track (default: can be overridden)
    default_token_mint: str = ""

    # Discovery settings
    default_threshold_pct: float = 0.5  # 0.5% of supply
    min_balance_filter: float = 0.0  # No minimum by default
    exclude_known_contracts: bool = True

    # Live monitoring
    default_refresh_seconds: int = 60
    refresh_options: dict[str, int | None] = field(default_factory=lambda: {
        "30 seconds": 30,
        "1 minute": 60,
        "5 minutes": 300,
        "Manual only": None,
    })

    # Alert thresholds
    large_move_threshold: float = 1_000_000  # 1M tokens
    exit_threshold: float = 100  # Balance below this = exit

    # Cache TTL
    balance_cache_ttl: int = 300  # 5 minutes
    transaction_cache_ttl: int = 600  # 10 minutes

    # Solana RPC
    solana_rpc_url: str = field(default_factory=lambda: os.getenv(
        "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
    ))
    helius_api_key: str = field(default_factory=lambda: os.getenv("HELIUS_API_KEY", ""))
    rpc_rate_limit: float = 5.0  # requests per second

    # Telegram (optional)
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    @property
    def telegram_configured(self) -> bool:
        """Check if Telegram is configured."""
        return bool(self.telegram_bot_token and self.telegram_chat_id)


# Global config instance
config = TrackerConfig()
