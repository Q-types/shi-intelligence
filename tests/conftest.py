"""Pytest configuration and shared fixtures."""

import pytest
from datetime import datetime, timezone

from src.core.types import TokenBalance, HolderSnapshot


@pytest.fixture
def sample_mint() -> str:
    """Sample token mint address."""
    return "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


@pytest.fixture
def sample_wallets() -> list[str]:
    """Sample wallet addresses."""
    return [
        "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1",
        "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
        "HN7cABqLq46Es1jh92dQQisAi6WMRPfhc2LYfzKjQsJE",
        "4sKwV2o7tSkGFWBnJuJFMJCZkxKz9BV3H7nxrKxAP6jA",
        "7RCz8wb6WXxUhAigok9ttgrVgDFFFbibcirECzWSBauM",
    ]


@pytest.fixture
def sample_holder_snapshot(sample_mint: str, sample_wallets: list[str]) -> HolderSnapshot:
    """Create a sample holder snapshot for testing."""
    now = datetime.now(timezone.utc)

    balances = [
        TokenBalance(
            wallet=sample_wallets[0],
            mint=sample_mint,
            balance=30_000_000_000,  # 30k tokens
            decimals=9,
            timestamp=now,
        ),
        TokenBalance(
            wallet=sample_wallets[1],
            mint=sample_mint,
            balance=20_000_000_000,  # 20k tokens
            decimals=9,
            timestamp=now,
        ),
        TokenBalance(
            wallet=sample_wallets[2],
            mint=sample_mint,
            balance=15_000_000_000,  # 15k tokens
            decimals=9,
            timestamp=now,
        ),
        TokenBalance(
            wallet=sample_wallets[3],
            mint=sample_mint,
            balance=10_000_000_000,  # 10k tokens
            decimals=9,
            timestamp=now,
        ),
        TokenBalance(
            wallet=sample_wallets[4],
            mint=sample_mint,
            balance=5_000_000_000,  # 5k tokens
            decimals=9,
            timestamp=now,
        ),
    ]

    total_supply = sum(b.balance for b in balances)

    return HolderSnapshot(
        mint=sample_mint,
        timestamp=now,
        total_supply=total_supply,
        holder_count=len(balances),
        balances=balances,
    )


@pytest.fixture
def large_holder_snapshot(sample_mint: str) -> HolderSnapshot:
    """Create a large holder snapshot for stress testing."""
    import random
    random.seed(42)  # Reproducible

    now = datetime.now(timezone.utc)

    # Generate 1000 holders with power-law distribution
    balances = []
    for i in range(1000):
        # Simplified wallet address
        wallet = f"wallet{i:04d}" + "x" * 35

        # Power-law-ish distribution
        balance = int(1_000_000_000 * (1000 / (i + 1)) ** 0.5)

        balances.append(TokenBalance(
            wallet=wallet,
            mint=sample_mint,
            balance=balance,
            decimals=9,
            timestamp=now,
        ))

    total_supply = sum(b.balance for b in balances)

    return HolderSnapshot(
        mint=sample_mint,
        timestamp=now,
        total_supply=total_supply,
        holder_count=len(balances),
        balances=balances,
    )
