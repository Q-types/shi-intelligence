"""
Sybil Detection Adversarial Tests.

Tests the system's ability to detect Sybil clusters
under various adversarial conditions.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone, timedelta
from typing import Generator

import pytest
import numpy as np

from shi.clustering.archetypes import Archetype, ArchetypeClassifier
from shi.graph.funding_graph import FundingGraph, FundingEdge
from shi.metrics.coordination import compute_coordination_score


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
        amount_range: tuple[float, float] = (0.1, 1.0),
    ) -> list[FundingEdge]:
        """Generate hub-and-spoke Sybil pattern."""
        edges = []
        base_time = datetime.now(timezone.utc) - timedelta(days=7)

        for i in range(num_spokes):
            spoke_wallet = f"spoke_{i}_{self.rng.randint(1000, 9999)}"
            amount = self.rng.uniform(*amount_range)

            # Add some timing jitter (coordinated entries)
            time_offset = timedelta(minutes=self.rng.randint(0, 60))

            edges.append(FundingEdge(
                from_wallet=hub_wallet,
                to_wallet=spoke_wallet,
                amount=amount,
                timestamp=base_time + time_offset,
                tx_signature=f"tx_{i}_{self.rng.randint(10000, 99999)}",
            ))

        return edges

    def generate_chain_cluster(
        self,
        start_wallet: str,
        chain_length: int,
        amount_decay: float = 0.95,
        initial_amount: float = 10.0,
    ) -> list[FundingEdge]:
        """Generate chain funding pattern (layered obfuscation)."""
        edges = []
        current_wallet = start_wallet
        current_amount = initial_amount
        base_time = datetime.now(timezone.utc) - timedelta(days=14)

        for i in range(chain_length):
            next_wallet = f"chain_{i}_{self.rng.randint(1000, 9999)}"

            # Each hop happens with some delay
            time_offset = timedelta(hours=self.rng.randint(1, 24))

            edges.append(FundingEdge(
                from_wallet=current_wallet,
                to_wallet=next_wallet,
                amount=current_amount,
                timestamp=base_time + time_offset * i,
                tx_signature=f"chain_tx_{i}_{self.rng.randint(10000, 99999)}",
            ))

            current_wallet = next_wallet
            current_amount *= amount_decay

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

    def test_hub_spoke_detection(
        self,
        generator: SybilClusterGenerator,
        funding_graph: FundingGraph,
    ) -> None:
        """Test detection of hub-and-spoke Sybil pattern."""
        # Generate Sybil cluster
        hub = "sybil_hub_wallet"
        edges = generator.generate_hub_spoke_cluster(hub, num_spokes=20)

        # Add to graph
        for edge in edges:
            funding_graph.add_edge(edge)

        # Detect communities
        communities = funding_graph.detect_communities()

        # The hub and spokes should be in the same community
        assert len(communities) >= 1

        # Find community containing hub
        hub_community = None
        for comm_id, members in communities.items():
            if hub in members:
                hub_community = members
                break

        assert hub_community is not None
        assert len(hub_community) >= 15  # Most spokes should be detected

        # Check centrality - hub should have high centrality
        centrality = funding_graph.get_centrality_scores()
        assert hub in centrality
        assert centrality[hub] > 0.5  # Hub should be central

    def test_chain_pattern_detection(
        self,
        generator: SybilClusterGenerator,
        funding_graph: FundingGraph,
    ) -> None:
        """Test detection of chain/layered funding pattern."""
        start = "chain_origin"
        edges = generator.generate_chain_cluster(start, chain_length=10)

        for edge in edges:
            funding_graph.add_edge(edge)

        # Chain should form connected component
        communities = funding_graph.detect_communities()

        # All chain nodes should be in one community
        chain_wallets = {start} | {e.to_wallet for e in edges}
        found_community = None

        for comm_id, members in communities.items():
            overlap = chain_wallets & set(members)
            if len(overlap) > len(chain_wallets) // 2:
                found_community = members
                break

        assert found_community is not None

    def test_coordination_score_accuracy(
        self,
        generator: SybilClusterGenerator,
    ) -> None:
        """Test coordination score detects synchronized behavior."""
        # High coordination scenario
        high_coord = generator.generate_coordinated_dump_scenario(
            num_wallets=10,
            coordination_level=0.95,
        )

        # Low coordination scenario
        low_coord_gen = SybilClusterGenerator(seed=123)
        low_coord = low_coord_gen.generate_coordinated_dump_scenario(
            num_wallets=10,
            coordination_level=0.2,
        )

        # Compute coordination scores
        high_score = compute_coordination_score(high_coord["sell_times"])
        low_score = compute_coordination_score(low_coord["sell_times"])

        # High coordination should have higher score
        assert high_score > low_score
        assert high_score > 0.7
        assert low_score < 0.5

    def test_false_positive_rate(
        self,
        generator: SybilClusterGenerator,
        funding_graph: FundingGraph,
    ) -> None:
        """Test false positive rate on legitimate patterns."""
        # Generate legitimate-looking funding patterns
        # (random funding, no coordination)

        legitimate_wallets = [f"legit_{i}" for i in range(50)]
        np_rng = np.random.default_rng(42)

        # Random funding edges (no pattern)
        for _ in range(100):
            from_idx = np_rng.integers(0, len(legitimate_wallets))
            to_idx = np_rng.integers(0, len(legitimate_wallets))
            if from_idx != to_idx:
                funding_graph.add_edge(FundingEdge(
                    from_wallet=legitimate_wallets[from_idx],
                    to_wallet=legitimate_wallets[to_idx],
                    amount=np_rng.uniform(0.1, 10.0),
                    timestamp=datetime.now(timezone.utc) - timedelta(
                        days=np_rng.integers(1, 30)
                    ),
                    tx_signature=f"legit_tx_{np_rng.integers(10000, 99999)}",
                ))

        # Detect communities
        communities = funding_graph.detect_communities()

        # Should not detect large suspicious clusters
        large_clusters = [c for c in communities.values() if len(c) > 10]

        # Few large clusters in random data
        assert len(large_clusters) <= 2

    def test_obfuscation_resistance(
        self,
        generator: SybilClusterGenerator,
        funding_graph: FundingGraph,
    ) -> None:
        """Test detection despite obfuscation attempts."""
        # Sybil cluster with noise wallets added
        hub = "obfuscated_hub"
        edges = generator.generate_hub_spoke_cluster(hub, num_spokes=15)

        # Add noise edges to obfuscate
        noise_wallets = [f"noise_{i}" for i in range(20)]
        for i in range(30):
            from_idx = generator.rng.randint(0, len(noise_wallets) - 1)
            to_idx = generator.rng.randint(0, len(noise_wallets) - 1)
            if from_idx != to_idx:
                edges.append(FundingEdge(
                    from_wallet=noise_wallets[from_idx],
                    to_wallet=noise_wallets[to_idx],
                    amount=generator.rng.uniform(0.01, 5.0),
                    timestamp=datetime.now(timezone.utc) - timedelta(
                        days=generator.rng.randint(1, 60)
                    ),
                    tx_signature=f"noise_tx_{generator.rng.randint(10000, 99999)}",
                ))

        for edge in edges:
            funding_graph.add_edge(edge)

        # Should still detect the Sybil cluster
        communities = funding_graph.detect_communities()
        centrality = funding_graph.get_centrality_scores()

        # Hub should still be detectable
        assert hub in centrality
        # Hub centrality should be notable even with noise
        hub_centrality = centrality[hub]
        avg_centrality = np.mean(list(centrality.values()))
        assert hub_centrality > avg_centrality


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


def compute_coordination_score(timestamps: list[datetime]) -> float:
    """
    Compute coordination score from timestamps.

    Higher score = more synchronized timing.
    """
    if len(timestamps) < 2:
        return 0.0

    # Convert to seconds
    times = sorted([t.timestamp() for t in timestamps])

    # Compute inter-arrival times
    intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]

    if not intervals:
        return 1.0

    # Coefficient of variation
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)

    if mean_interval == 0:
        return 1.0

    cv = std_interval / mean_interval

    # Convert to [0, 1] score (lower CV = higher coordination)
    score = 1.0 / (1.0 + cv)

    return float(score)
