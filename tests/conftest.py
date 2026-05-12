"""Pytest configuration and shared fixtures."""

import pytest
from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from src.core.types import TokenBalance, HolderSnapshot
from src.data.repositories.wallet_history import WalletBehaviorSummary


# ============================================================================
# Basic Fixtures (existing)
# ============================================================================


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


# ============================================================================
# Cross-Token Intelligence Fixtures
# ============================================================================


@pytest.fixture
def sample_token_mints() -> list[str]:
    """Sample token mint addresses for cross-token tests."""
    return [
        "TokenMint1111111111111111111111111111111111",
        "TokenMint2222222222222222222222222222222222",
        "TokenMint3333333333333333333333333333333333",
        "TokenMint4444444444444444444444444444444444",
        "TokenMint5555555555555555555555555555555555",
    ]


@pytest.fixture
def sybil_cluster_wallets() -> list[str]:
    """Wallets that form a sybil cluster (same funder)."""
    return [
        "SybilWallet111111111111111111111111111111111",
        "SybilWallet222222222222222222222222222222222",
        "SybilWallet333333333333333333333333333333333",
        "SybilWallet444444444444444444444444444444444",
        "SybilWallet555555555555555555555555555555555",
    ]


@pytest.fixture
def sybil_funder() -> str:
    """The wallet that funded the sybil cluster."""
    return "SybilFunder11111111111111111111111111111111"


@pytest.fixture
def serial_sniper_behavior() -> WalletBehaviorSummary:
    """Behavior summary for a serial sniper wallet."""
    return WalletBehaviorSummary(
        wallet_address="SerialSniper1111111111111111111111111111",
        tokens_analyzed=10,
        sniper_count=8,  # High sniper rate
        accumulator_count=0,
        rugpull_count=2,
        avg_holding_days=1.5,  # Very short holding
        avg_pnl_pct=150.0,  # High profits from sniping
    )


@pytest.fixture
def diamond_hands_behavior() -> WalletBehaviorSummary:
    """Behavior summary for a diamond hands wallet."""
    return WalletBehaviorSummary(
        wallet_address="DiamondHands11111111111111111111111111111",
        tokens_analyzed=15,
        sniper_count=0,
        accumulator_count=12,  # High accumulator rate
        rugpull_count=1,
        avg_holding_days=90.0,  # Long holding period
        avg_pnl_pct=50.0,  # Moderate profits
    )


@pytest.fixture
def rugpull_victim_behavior() -> WalletBehaviorSummary:
    """Behavior summary for a rugpull victim."""
    return WalletBehaviorSummary(
        wallet_address="RugpullVictim111111111111111111111111111",
        tokens_analyzed=8,
        sniper_count=0,
        accumulator_count=2,
        rugpull_count=5,  # Many rugs
        avg_holding_days=30.0,
        avg_pnl_pct=-75.0,  # Heavy losses
    )


@pytest.fixture
def new_wallet_behavior() -> WalletBehaviorSummary:
    """Behavior summary for a new wallet with limited history."""
    return WalletBehaviorSummary(
        wallet_address="NewWallet1111111111111111111111111111111",
        tokens_analyzed=2,  # Too few for pattern detection
        sniper_count=1,
        accumulator_count=0,
        rugpull_count=0,
        avg_holding_days=5.0,
        avg_pnl_pct=10.0,
    )


# ============================================================================
# Mock Database Session Fixtures
# ============================================================================


@pytest.fixture
def mock_async_session() -> AsyncMock:
    """Create a mock async database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def mock_scalar_result():
    """Factory for mock scalar results."""
    def _make_result(value):
        result = MagicMock()
        result.scalar_one_or_none.return_value = value
        result.scalar_one.return_value = value
        result.scalars.return_value.all.return_value = [value] if value else []
        result.fetchall.return_value = []
        result.fetchone.return_value = None
        return result
    return _make_result


# ============================================================================
# Entity and Membership Fixtures
# ============================================================================


@pytest.fixture
def sample_entity_data():
    """Sample entity creation data."""
    return {
        "entity_type": "sybil_cluster",
        "detection_method": "shared_funder",
        "dominant_funder_address": "FunderAddress111111111111111111111111111",
        "confidence_score": 0.85,
    }


@pytest.fixture
def sample_membership_data(sybil_cluster_wallets: list[str]):
    """Sample membership data for an entity."""
    return [
        {
            "wallet_address": wallet,
            "detected_via": "shared_funder",
            "membership_confidence": 0.9,
            "shared_funder_address": "FunderAddress111111111111111111111111111",
        }
        for wallet in sybil_cluster_wallets
    ]


# ============================================================================
# Reputation Fixtures
# ============================================================================


@pytest.fixture
def sample_reputation_data():
    """Sample reputation record data."""
    return {
        "wallet_address": "ReputationWallet111111111111111111111111",
        "reputation_score": 65,
        "confidence_level": "medium",
        "tokens_analyzed": 10,
        "sniper_count": 2,
        "accumulator_count": 5,
        "rugpull_count": 1,
        "avg_holding_days": 45.0,
        "avg_pnl_pct": 25.0,
        "patterns": [
            {"type": "DIAMOND_HANDS", "confidence": 0.7, "token_count": 5}
        ],
    }


# ============================================================================
# Funding Graph Fixtures
# ============================================================================


@pytest.fixture
def mock_funding_graph(sybil_cluster_wallets: list[str], sybil_funder: str):
    """Create a mock funding graph with sybil cluster."""
    from unittest.mock import MagicMock
    import networkx as nx

    graph = MagicMock()

    # Create actual NetworkX graph for testing
    G = nx.DiGraph()
    G.add_node(sybil_funder)
    for wallet in sybil_cluster_wallets:
        G.add_node(wallet)
        G.add_edge(sybil_funder, wallet, amount=1_000_000_000, timestamp=datetime.now(timezone.utc).isoformat())

    graph._graph = G
    graph._wallet_set = set(sybil_cluster_wallets)

    # Mock methods
    def find_shared_funders(target_wallets, max_depth=2):
        return {sybil_funder: set(sybil_cluster_wallets)}

    def get_dominant_funder(target_wallets, max_depth=2):
        return (sybil_funder, len(sybil_cluster_wallets))

    graph.find_shared_funders = find_shared_funders
    graph.get_dominant_funder = get_dominant_funder

    return graph


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
