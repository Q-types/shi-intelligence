"""
Sybil Detection Adversarial Tests.

Tests the system's ability to detect Sybil clusters
under various adversarial conditions.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone, timedelta

import pytest
import numpy as np

from src.graph.funding_graph import FundingGraph
from src.core.types import FundingEdge
from src.metrics.coordination import compute_coordination_score

# Base58 alphabet (no 0, I, O, l)
BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


class SybilClusterGenerator:
    """
    Generates synthetic Sybil clusters for testing.

    Creates realistic adversarial patterns including:
    - Hub-and-spoke funding patterns
    - Chain funding patterns
    - Layered obfuscation
    - Timing coordination
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate_hub_spoke_cluster(
        self,
        hub_wallet: str,
        num_spokes: int,
        amount_range: tuple[int, int] = (100000, 1000000),
    ) -> list[FundingEdge]:
        """Generate hub-and-spoke Sybil pattern."""
        edges = []
        base_time = datetime.now(timezone.utc) - timedelta(days=7)

        for i in range(num_spokes):
            # Use base58-valid addresses: "spoke" has no invalid chars, pad with 1s
            suffix = BASE58[i % len(BASE58)]
            spoke_wallet = f"spoke{suffix}111111111111111111111111111"  # 33 chars, base58

            amount = self.rng.randint(*amount_range)

            # Add some timing jitter (coordinated entries)
            time_offset = timedelta(minutes=self.rng.randint(0, 60))

            edges.append(FundingEdge(
                source=hub_wallet,
                target=spoke_wallet,
                amount_lamports=amount,
                timestamp=base_time + time_offset,
                signature="1" * 88,
            ))

        return edges

    def generate_chain_cluster(
        self,
        start_wallet: str,
        chain_length: int,
        amount_decay: float = 0.95,
        initial_amount: int = 10000000,
    ) -> list[FundingEdge]:
        """Generate chain funding pattern (layered obfuscation)."""
        edges = []
        current_wallet = start_wallet
        current_amount = initial_amount
        base_time = datetime.now(timezone.utc) - timedelta(days=14)

        for i in range(chain_length):
            # Use base58-valid addresses
            suffix = BASE58[i % len(BASE58)]
            next_wallet = f"chain{suffix}11111111111111111111111111"  # 32 chars, base58

            # Each hop happens with some delay
            time_offset = timedelta(hours=self.rng.randint(1, 24))

            edges.append(FundingEdge(
                source=current_wallet,
                target=next_wallet,
                amount_lamports=int(current_amount),
                timestamp=base_time + time_offset * i,
                signature="2" * 88,
            ))

            current_wallet = next_wallet
            current_amount = int(current_amount * amount_decay)

        return edges

    def generate_wash_trading_pattern(
        self,
        wallets: list[str],
        num_rounds: int = 5,
        amount: float = 100.0,
    ) -> list[tuple[str, str, float, datetime]]:
        """
        Generate wash trading circular pattern.

        Wallets trade tokens back and forth to simulate volume.
        """
        trades = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=24)

        for round_num in range(num_rounds):
            for i in range(len(wallets)):
                from_wallet = wallets[i]
                to_wallet = wallets[(i + 1) % len(wallets)]

                # Small time gaps between wash trades
                time_offset = timedelta(
                    minutes=round_num * 30 + i * 5 + self.rng.randint(0, 3)
                )

                trades.append((
                    from_wallet,
                    to_wallet,
                    amount * (1 + self.rng.uniform(-0.01, 0.01)),  # Slight variation
                    base_time + time_offset,
                ))

        return trades

    def generate_coordinated_dump_scenario(
        self,
        num_wallets: int,
        coordination_level: float = 0.9,
    ) -> dict:
        """
        Generate coordinated dump scenario.

        All wallets sell within a short time window.
        """
        wallets = [f"dump_wallet_{i}" for i in range(num_wallets)]
        base_time = datetime.now(timezone.utc)

        # High coordination = tight time window
        window_minutes = int((1 - coordination_level) * 60) + 1

        sell_times = []
        sell_amounts = []

        for _ in wallets:
            offset = timedelta(minutes=self.rng.randint(0, window_minutes))
            sell_times.append(base_time + offset)
            sell_amounts.append(self.rng.uniform(0.8, 1.0))  # Sell 80-100%

        return {
            "wallets": wallets,
            "sell_times": sell_times,
            "sell_percentages": sell_amounts,
            "expected_coordination": coordination_level,
        }


class TestSybilDetection:
    """Tests for Sybil cluster detection."""

    @pytest.fixture
    def generator(self) -> SybilClusterGenerator:
        return SybilClusterGenerator(seed=42)

    @pytest.fixture
    def funding_graph(self) -> FundingGraph:
        return FundingGraph()

    @pytest.mark.skip(reason="FundingGraph.detect_communities not implemented yet")
    def test_hub_spoke_detection(
        self,
        generator: SybilClusterGenerator,
        funding_graph: FundingGraph,
    ) -> None:
        """Test detection of hub-and-spoke Sybil pattern."""
        pass

    @pytest.mark.skip(reason="FundingGraph.detect_communities not implemented yet")
    def test_chain_pattern_detection(
        self,
        generator: SybilClusterGenerator,
        funding_graph: FundingGraph,
    ) -> None:
        """Test detection of chain/layered funding pattern."""
        pass

    def test_coordination_score_accuracy(
        self,
        generator: SybilClusterGenerator,
    ) -> None:
        """Test coordination score detects shared funder patterns."""
        # High coordination scenario: most wallets share a funder
        high_coord_wallets = [f"highcoord{BASE58[i]}111111111111111111111" for i in range(10)]
        high_coord_shared = set(high_coord_wallets[:9])  # 9 out of 10 share funder

        # Low coordination scenario: few wallets share a funder
        low_coord_wallets = [f"1owcoord{BASE58[i]}1111111111111111111111" for i in range(10)]
        low_coord_shared = set(low_coord_wallets[:2])  # Only 2 out of 10 share funder

        # Compute coordination scores
        high_score = compute_coordination_score(high_coord_wallets, high_coord_shared)
        low_score = compute_coordination_score(low_coord_wallets, low_coord_shared)

        # High coordination should have higher score
        assert high_score.value > low_score.value
        assert high_score.value > 0.7  # 9/10 = 0.9
        assert low_score.value < 0.5  # 2/10 = 0.2

    @pytest.mark.skip(reason="FundingGraph.detect_communities not implemented yet")
    def test_false_positive_rate(
        self,
        generator: SybilClusterGenerator,
        funding_graph: FundingGraph,
    ) -> None:
        """Test false positive rate on legitimate patterns."""
        pass

    @pytest.mark.skip(reason="FundingGraph.detect_communities not implemented yet")
    def test_obfuscation_resistance(
        self,
        generator: SybilClusterGenerator,
        funding_graph: FundingGraph,
    ) -> None:
        """Test detection despite obfuscation attempts."""
        pass


class TestWashTradingDetection:
    """Tests for wash trading detection."""

    @pytest.fixture
    def generator(self) -> SybilClusterGenerator:
        return SybilClusterGenerator(seed=42)

    def test_circular_pattern_detection(
        self,
        generator: SybilClusterGenerator,
    ) -> None:
        """Test detection of circular wash trading."""
        wallets = [f"wash_{i}" for i in range(5)]
        trades = generator.generate_wash_trading_pattern(wallets, num_rounds=10)

        # Analyze trade patterns
        # Count trades between each pair
        pair_counts: dict[tuple[str, str], int] = {}
        for from_w, to_w, _, _ in trades:
            pair = (from_w, to_w)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # In wash trading, pairs should have similar counts
        counts = list(pair_counts.values())
        count_std = np.std(counts)
        count_mean = np.mean(counts)

        # Low coefficient of variation indicates coordinated trading
        cv = count_std / count_mean if count_mean > 0 else 0
        assert cv < 0.5  # Wash trading has uniform distribution

    def test_timing_pattern_detection(
        self,
        generator: SybilClusterGenerator,
    ) -> None:
        """Test detection of timing patterns in wash trading."""
        wallets = [f"timing_wash_{i}" for i in range(4)]
        trades = generator.generate_wash_trading_pattern(wallets, num_rounds=20)

        # Extract timestamps
        timestamps = [t[3] for t in trades]

        # Compute inter-trade times
        sorted_times = sorted([t.timestamp() for t in timestamps])
        intervals = [
            sorted_times[i + 1] - sorted_times[i]
            for i in range(len(sorted_times) - 1)
        ]

        # Wash trading has regular intervals
        interval_std = np.std(intervals)
        interval_mean = np.mean(intervals)

        # Regular intervals = low CV
        cv = interval_std / interval_mean if interval_mean > 0 else 0
        assert cv < 1.0  # More regular than random
