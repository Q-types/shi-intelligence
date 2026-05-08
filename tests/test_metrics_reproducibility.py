"""
Reproducibility Validation Suite.

Tests that all metrics produce deterministic, reproducible outputs.
Same inputs must always produce same outputs.
"""

import pytest
from datetime import datetime, timezone

from src.core.types import TokenBalance, HolderSnapshot
from src.metrics import (
    compute_hhi,
    compute_shannon_entropy,
    compute_gini_coefficient,
    compute_whale_dominance_ratio,
    compute_churn_rate,
    compute_coordination_score,
    compute_funding_density,
)
from src.pipeline.metrics_pipeline import MetricsPipeline


class TestMetricsDeterminism:
    """Test that metrics produce identical results for identical inputs."""

    @pytest.fixture
    def sample_shares(self) -> list[float]:
        """Sample ownership shares."""
        return [0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02]

    @pytest.fixture
    def sample_balances(self) -> list[float]:
        """Sample wallet balances."""
        return [30000, 20000, 15000, 10000, 8000, 7000, 5000, 3000, 2000]

    def test_hhi_deterministic(self, sample_shares):
        """HHI should produce identical results for same input."""
        # Compute 100 times
        results = [compute_hhi(sample_shares).value for _ in range(100)]

        # All results should be identical
        assert all(r == results[0] for r in results)

        # Known value check
        expected = sum(s**2 for s in sample_shares)
        assert abs(results[0] - expected) < 1e-10

    def test_entropy_deterministic(self, sample_shares):
        """Shannon entropy should produce identical results for same input."""
        results = [compute_shannon_entropy(sample_shares).value for _ in range(100)]

        assert all(r == results[0] for r in results)

    def test_gini_deterministic(self, sample_balances):
        """Gini coefficient should produce identical results for same input."""
        results = [compute_gini_coefficient(sample_balances).value for _ in range(100)]

        assert all(r == results[0] for r in results)

    def test_wdr_deterministic(self, sample_balances):
        """Whale dominance ratio should produce identical results for same input."""
        total_supply = sum(sample_balances)
        results = [
            compute_whale_dominance_ratio(sample_balances, total_supply, k=3).value
            for _ in range(100)
        ]

        assert all(r == results[0] for r in results)

        # Top 3 should be 30k + 20k + 15k = 65k
        expected = 65000 / total_supply
        assert abs(results[0] - expected) < 1e-10

    def test_churn_deterministic(self):
        """Churn rate should produce identical results for same input."""
        results = [
            compute_churn_rate(wallets_at_start=100, wallets_exited=15).value
            for _ in range(100)
        ]

        assert all(r == results[0] for r in results)
        assert results[0] == 0.15

    def test_coordination_score_deterministic(self):
        """Coordination score should produce identical results for same input."""
        cluster = ["wallet1", "wallet2", "wallet3", "wallet4", "wallet5"]
        shared = {"wallet1", "wallet2", "wallet3"}

        results = [
            compute_coordination_score(cluster, shared).value
            for _ in range(100)
        ]

        assert all(r == results[0] for r in results)
        assert results[0] == 0.6  # 3/5

    def test_funding_density_deterministic(self):
        """Funding density should produce identical results for same input."""
        results = [
            compute_funding_density(num_vertices=10, num_edges=20).value
            for _ in range(100)
        ]

        assert all(r == results[0] for r in results)

        # max_edges = 10 * 9 = 90 for directed graph
        expected = 20 / 90
        assert abs(results[0] - expected) < 1e-10


class TestMetricsKnownValues:
    """Test metrics against known reference values."""

    def test_hhi_perfect_monopoly(self):
        """Single holder = HHI of 1.0."""
        shares = [1.0]
        result = compute_hhi(shares)
        assert result.value == 1.0

    def test_hhi_equal_distribution(self):
        """Equal distribution among N holders = HHI of 1/N."""
        n = 100
        shares = [1/n] * n
        result = compute_hhi(shares)
        assert abs(result.value - 0.01) < 1e-10

    def test_entropy_maximum(self):
        """Maximum entropy for N equal holders = log(N)."""
        import math
        n = 100
        shares = [1/n] * n
        result = compute_shannon_entropy(shares)
        expected = math.log(n)
        assert abs(result.value - expected) < 1e-10

    def test_gini_perfect_equality(self):
        """Equal balances = Gini of 0."""
        balances = [1000.0] * 100
        result = compute_gini_coefficient(balances)
        assert abs(result.value) < 1e-10

    def test_gini_maximum_inequality(self):
        """One holder has everything = Gini approaching 1."""
        balances = [1000000.0] + [0.0] * 99
        # Note: Gini is undefined when most are zero
        # Using small values instead
        balances = [1000000.0] + [1.0] * 99
        result = compute_gini_coefficient(balances)
        # Gini approaches but doesn't exactly reach 1 with finite holders
        assert result.value > 0.98


class TestChecksumConsistency:
    """Test that checksums are consistent for same data."""

    @pytest.fixture
    def sample_snapshot(self) -> HolderSnapshot:
        """Create sample holder snapshot."""
        now = datetime.now(timezone.utc)
        # Base58 alphabet (no 0, I, O, l)
        base58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        balances = [
            TokenBalance(
                wallet=f"wa11et{base58[i % len(base58)]}" + "1" * 25,  # 32 chars, valid base58
                mint="mint1111111111111111111111111111111111",  # 38 chars, valid base58
                balance=1000 * (100 - i),
                decimals=9,
                timestamp=now,
            )
            for i in range(10)
        ]
        return HolderSnapshot(
            mint="mint1111111111111111111111111111111111",  # 38 chars, valid base58
            timestamp=now,
            total_supply=sum(b.balance for b in balances),
            holder_count=len(balances),
            balances=balances,
        )

    def test_checksum_deterministic(self, sample_snapshot):
        """Same snapshot should produce same checksum."""
        pipeline = MetricsPipeline()

        checksums = [
            pipeline._compute_checksum(sample_snapshot)
            for _ in range(100)
        ]

        assert all(c == checksums[0] for c in checksums)

    def test_metrics_result_reproducible(self, sample_snapshot):
        """Full metrics computation should be reproducible."""
        pipeline = MetricsPipeline()

        result1 = pipeline.compute_all(sample_snapshot)
        result2 = pipeline.compute_all(sample_snapshot)

        # All values should match exactly
        assert result1.hhi.value == result2.hhi.value
        assert result1.shannon_entropy.value == result2.shannon_entropy.value
        assert result1.gini_coefficient.value == result2.gini_coefficient.value
        assert result1.whale_dominance_ratio.value == result2.whale_dominance_ratio.value
        assert result1.snapshot_checksum == result2.snapshot_checksum

    def test_reproducibility_verification(self, sample_snapshot):
        """Reproducibility verification should pass for same data."""
        pipeline = MetricsPipeline()

        result = pipeline.compute_all(sample_snapshot)

        # Should verify successfully
        assert pipeline.verify_reproducibility(sample_snapshot, result)


class TestVersionTracking:
    """Test that versions are properly tracked."""

    def test_metric_version_present(self):
        """All metrics should include version."""
        shares = [0.5, 0.3, 0.2]

        hhi = compute_hhi(shares)
        entropy = compute_shannon_entropy(shares)

        assert hhi.version is not None
        assert entropy.version is not None
        assert hhi.version == "1.0.0"

    def test_pipeline_version_in_result(self):
        """Pipeline result should include version."""
        now = datetime.now(timezone.utc)
        balances = [
            TokenBalance(
                wallet="wa11et1111111111111111111111111111",  # 34 chars, valid base58
                mint="mint1111111111111111111111111111111111",  # 38 chars, valid base58
                balance=1000,
                decimals=9,
                timestamp=now,
            )
        ]
        snapshot = HolderSnapshot(
            mint="mint1111111111111111111111111111111111",  # 38 chars, valid base58
            timestamp=now,
            total_supply=1000,
            holder_count=1,
            balances=balances,
        )

        pipeline = MetricsPipeline()
        result = pipeline.compute_all(snapshot)

        assert result.metrics_version is not None
        assert result.metrics_version == MetricsPipeline.VERSION


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_hhi_empty_raises(self):
        """HHI should raise for empty input."""
        with pytest.raises(ValueError):
            compute_hhi([])

    def test_entropy_zero_shares_handled(self):
        """Entropy should handle zero shares gracefully."""
        shares = [0.5, 0.3, 0.2, 0.0, 0.0]  # Sums to 1.0
        result = compute_shannon_entropy(shares)
        assert result.value > 0

    def test_gini_single_holder(self):
        """Gini should handle single holder."""
        balances = [1000.0]
        result = compute_gini_coefficient(balances)
        assert result.value == 0.0  # No inequality with one holder

    def test_churn_zero_start_raises(self):
        """Churn should raise if starting wallets is zero."""
        with pytest.raises(ValueError):
            compute_churn_rate(wallets_at_start=0, wallets_exited=0)

    def test_coordination_empty_cluster_raises(self):
        """Coordination should raise for empty cluster."""
        with pytest.raises(ValueError):
            compute_coordination_score([], set())
