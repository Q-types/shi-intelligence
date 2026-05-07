"""
Metrics Computation Pipeline.

Computes all locked metrics from PDR Section 4.
All computations are deterministic and versioned.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import structlog

from ..core.types import HolderSnapshot, MetricOutput
from ..metrics import (
    compute_hhi,
    compute_shannon_entropy,
    compute_gini_coefficient,
    compute_whale_dominance_ratio,
    compute_churn_rate,
    compute_coordination_score,
    compute_funding_density,
)
from ..graph import FundingGraph

logger = structlog.get_logger()


@dataclass
class MetricsResult:
    """Complete metrics computation result."""

    # Distribution metrics
    hhi: MetricOutput
    shannon_entropy: MetricOutput
    gini_coefficient: MetricOutput
    whale_dominance_ratio: MetricOutput

    # Coordination metrics
    churn_rate: MetricOutput | None
    coordination_score: MetricOutput | None
    funding_density: MetricOutput | None

    # Metadata
    snapshot_checksum: str
    computed_at: datetime
    metrics_version: str

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "distribution": {
                "hhi": self.hhi.value,
                "entropy": self.shannon_entropy.value,
                "gini": self.gini_coefficient.value,
                "wdr": self.whale_dominance_ratio.value,
            },
            "coordination": {
                "churn": self.churn_rate.value if self.churn_rate else None,
                "coordination": self.coordination_score.value if self.coordination_score else None,
                "density": self.funding_density.value if self.funding_density else None,
            },
            "metadata": {
                "checksum": self.snapshot_checksum,
                "computed_at": self.computed_at.isoformat(),
                "version": self.metrics_version,
            },
        }


class MetricsPipeline:
    """
    Pipeline for computing all locked metrics.

    All metrics are computed per PDR Section 4 definitions.
    Results are deterministic given same inputs.
    """

    VERSION = "1.0.0"

    def __init__(self):
        self._computation_count = 0

    def compute_all(
        self,
        snapshot: HolderSnapshot,
        funding_graph: FundingGraph | None = None,
        previous_snapshot: HolderSnapshot | None = None,
        cluster_wallets: list[str] | None = None,
        shared_funder_wallets: set[str] | None = None,
    ) -> MetricsResult:
        """
        Compute all metrics for a holder snapshot.

        Args:
            snapshot: Current holder snapshot
            funding_graph: Optional funding graph for coordination metrics
            previous_snapshot: Optional previous snapshot for churn calculation
            cluster_wallets: Optional cluster for coordination score
            shared_funder_wallets: Wallets sharing top funder

        Returns:
            MetricsResult with all computed metrics
        """
        logger.info(
            "computing_metrics",
            mint=snapshot.mint,
            holders=snapshot.holder_count,
            version=self.VERSION,
        )

        self._computation_count += 1

        # Compute snapshot checksum for reproducibility
        checksum = self._compute_checksum(snapshot)

        # Extract shares and balances
        shares = snapshot.shares
        balances = [b.balance for b in snapshot.balances]

        # Distribution metrics (always computed)
        hhi = compute_hhi(shares)
        entropy = compute_shannon_entropy(shares)
        gini = compute_gini_coefficient(balances)
        wdr = compute_whale_dominance_ratio(balances, snapshot.total_supply)

        # Coordination metrics (optional, require additional data)
        churn = None
        coordination = None
        density = None

        if previous_snapshot:
            churn = self._compute_churn(snapshot, previous_snapshot)

        if cluster_wallets and shared_funder_wallets:
            coordination = compute_coordination_score(
                cluster_wallets,
                shared_funder_wallets,
            )

        if funding_graph and funding_graph.num_vertices >= 2:
            density = compute_funding_density(
                funding_graph.num_vertices,
                funding_graph.num_edges,
            )

        result = MetricsResult(
            hhi=hhi,
            shannon_entropy=entropy,
            gini_coefficient=gini,
            whale_dominance_ratio=wdr,
            churn_rate=churn,
            coordination_score=coordination,
            funding_density=density,
            snapshot_checksum=checksum,
            computed_at=datetime.now(timezone.utc),
            metrics_version=self.VERSION,
        )

        logger.info(
            "metrics_computed",
            hhi=hhi.value,
            gini=gini.value,
            wdr=wdr.value,
            checksum=checksum,
        )

        return result

    def _compute_churn(
        self,
        current: HolderSnapshot,
        previous: HolderSnapshot,
    ) -> MetricOutput:
        """Compute churn rate between snapshots."""
        current_wallets = {b.wallet for b in current.balances if b.balance > 0}
        previous_wallets = {b.wallet for b in previous.balances if b.balance > 0}

        # Wallets that exited (had balance, now don't)
        exited = previous_wallets - current_wallets

        return compute_churn_rate(
            wallets_at_start=len(previous_wallets),
            wallets_exited=len(exited),
        )

    def _compute_checksum(self, snapshot: HolderSnapshot) -> str:
        """
        Compute deterministic checksum for snapshot.

        Used to verify reproducibility - same snapshot should
        always produce same checksum.
        """
        # Sort balances deterministically
        sorted_balances = sorted(
            [(b.wallet, b.balance) for b in snapshot.balances],
            key=lambda x: x[0],
        )

        # Build checksum input
        data = f"{snapshot.mint}:{snapshot.total_supply}:{snapshot.holder_count}"
        for wallet, balance in sorted_balances[:100]:  # Top 100 for efficiency
            data += f":{wallet[:8]}:{balance}"

        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def verify_reproducibility(
        self,
        snapshot: HolderSnapshot,
        expected_result: MetricsResult,
    ) -> bool:
        """
        Verify that recomputation produces same results.

        Used for reproducibility validation.
        """
        recomputed = self.compute_all(snapshot)

        # Check key values
        checks = [
            abs(recomputed.hhi.value - expected_result.hhi.value) < 1e-10,
            abs(recomputed.shannon_entropy.value - expected_result.shannon_entropy.value) < 1e-10,
            abs(recomputed.gini_coefficient.value - expected_result.gini_coefficient.value) < 1e-10,
            abs(recomputed.whale_dominance_ratio.value - expected_result.whale_dominance_ratio.value) < 1e-10,
            recomputed.snapshot_checksum == expected_result.snapshot_checksum,
        ]

        is_reproducible = all(checks)

        if not is_reproducible:
            logger.error(
                "reproducibility_failed",
                expected_checksum=expected_result.snapshot_checksum,
                actual_checksum=recomputed.snapshot_checksum,
            )

        return is_reproducible
