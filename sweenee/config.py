"""SWEENEE Whale Dashboard Configuration.

Centralized configuration for token tracking, API settings, and paths.
"""

from __future__ import annotations

import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass, field
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).parent
WALLETS_DIR = BASE_DIR / "wallets"
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "sweenee.sqlite"

# Token configuration
SWEENEE_MINT = "FkAtYamtEMtgnsTeUhzhTCiT2Svyxw63UdUYp1T7pump"
SWEENEE_DECIMALS = 6  # Standard SPL token decimals

# API configuration (inherit from parent SHI project)
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")

# Rate limiting
RATE_LIMIT_PER_SECOND = 10
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0

# Cache configuration
BALANCE_CACHE_TTL_SECONDS = 300  # 5 minutes
TRANSACTION_CACHE_TTL_SECONDS = 300  # 5 minutes
DEFAULT_TRANSACTION_LOOKBACK_DAYS = 7

# Dashboard configuration
DEFAULT_TOP_WALLETS_DISPLAY = 20
EXPLORER_BASE_URL = "https://solscan.io"


@dataclass
class DashboardConfig:
    """Dashboard configuration container."""

    sweenee_mint: str = SWEENEE_MINT
    sweenee_decimals: int = SWEENEE_DECIMALS
    wallets_dir: Path = field(default_factory=lambda: WALLETS_DIR)
    database_path: Path = field(default_factory=lambda: DATABASE_PATH)

    # API settings
    solana_rpc_url: str = field(default_factory=lambda: SOLANA_RPC_URL)
    helius_api_key: str = field(default_factory=lambda: HELIUS_API_KEY)

    # Rate limiting
    rate_limit_per_second: int = RATE_LIMIT_PER_SECOND
    max_retries: int = MAX_RETRIES

    # Cache TTLs
    balance_cache_ttl: int = BALANCE_CACHE_TTL_SECONDS
    transaction_cache_ttl: int = TRANSACTION_CACHE_TTL_SECONDS

    # Display settings
    top_wallets_display: int = DEFAULT_TOP_WALLETS_DISPLAY
    explorer_base_url: str = EXPLORER_BASE_URL

    @classmethod
    def from_env(cls) -> "DashboardConfig":
        """Create config from environment variables."""
        return cls(
            sweenee_mint=os.getenv("SWEENEE_MINT", SWEENEE_MINT),
            solana_rpc_url=os.getenv("SOLANA_RPC_URL", SOLANA_RPC_URL),
            helius_api_key=os.getenv("HELIUS_API_KEY", ""),
            balance_cache_ttl=int(os.getenv("BALANCE_CACHE_TTL", BALANCE_CACHE_TTL_SECONDS)),
            transaction_cache_ttl=int(os.getenv("TRANSACTION_CACHE_TTL", TRANSACTION_CACHE_TTL_SECONDS)),
        )


# Global config instance
config = DashboardConfig.from_env()
