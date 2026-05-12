"""
Shared Funder Detection.

Identifies wallets funded by common sources to detect sybil clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

import structlog

from ..core.types import WalletAddress
from ..graph.funding_graph import FundingGraph

logger = structlog.get_logger()


@dataclass
class FunderCluster:
    """A cluster of wallets sharing a common funder."""

    funder_address: str
    wallet_addresses: list[str]
    funding_depth: int  # 1 = direct, 2+ = indirect
    total_funded_amount: int  # lamports
    avg_funded_amount: float
    funding_time_span_hours: Optional[float]
    confidence: float


@dataclass
class SharedFunderResult:
    """Result of shared funder detection."""

    clusters: list[FunderCluster]
    total_wallets_analyzed: int
    total_wallets_clustered: int
    dominant_funder: Optional[str]
    dominant_funder_count: int
    detection_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SharedFunderDetector:
    """
    Detects wallet clusters based on shared funding sources.

    Uses the funding graph to identify wallets that:
    - Were funded by the same source wallet
    - Share common ancestors in the funding tree
    - Show coordinated funding patterns

    Confidence scoring considers:
    - Number of wallets sharing the funder (more = higher confidence)
    - Funding depth (direct funding = higher confidence)
    - Amount consistency (similar amounts = higher confidence)
    - Timing patterns (rapid sequential funding = higher confidence)
    """

    def __init__(
        self,
        min_cluster_size: int = 2,
        max_depth: int = 2,
        min_confidence: float = 0.5,
    ):
        self.min_cluster_size = min_cluster_size
        self.max_depth = max_depth
        self.min_confidence = min_confidence

    def detect(
        self,
        funding_graph: FundingGraph,
        target_wallets: Optional[list[str]] = None,
    ) -> SharedFunderResult:
        """
        Detect wallet clusters with shared funders.

        Args:
            funding_graph: The funding graph to analyze
            target_wallets: Specific wallets to analyze (default: all)

        Returns:
            SharedFunderResult with detected clusters
        """
        # Use all wallets if none specified
        if target_wallets is None:
            target_wallets = list(funding_graph._wallet_set)

        if not target_wallets:
            return SharedFunderResult(
                clusters=[],
                total_wallets_analyzed=0,
                total_wallets_clustered=0,
                dominant_funder=None,
                dominant_funder_count=0,
            )

        # Find shared funders
        shared_funders = funding_graph.find_shared_funders(
            target_wallets, max_depth=self.max_depth
        )

        # Build clusters with confidence scoring
        clusters = []
        clustered_wallets = set()

        for funder_addr, funded_wallets in shared_funders.items():
            if len(funded_wallets) < self.min_cluster_size:
                continue

            wallet_list = list(funded_wallets)

            # Calculate cluster metrics
            cluster = self._build_cluster(
                funding_graph,
                funder_addr,
                wallet_list,
            )

            if cluster.confidence >= self.min_confidence:
                clusters.append(cluster)
                clustered_wallets.update(wallet_list)

        # Sort by confidence descending
        clusters.sort(key=lambda c: c.confidence, reverse=True)

        # Find dominant funder
        dominant_funder, dominant_count = funding_graph.get_dominant_funder(
            target_wallets, max_depth=self.max_depth
        )

        result = SharedFunderResult(
            clusters=clusters,
            total_wallets_analyzed=len(target_wallets),
            total_wallets_clustered=len(clustered_wallets),
            dominant_funder=dominant_funder,
            dominant_funder_count=dominant_count,
        )

        logger.info(
            "shared_funder_detection_complete",
            wallets_analyzed=result.total_wallets_analyzed,
            clusters_found=len(clusters),
            wallets_clustered=result.total_wallets_clustered,
            dominant_funder=dominant_funder[:8] if dominant_funder else None,
        )

        return result

    def _build_cluster(
        self,
        graph: FundingGraph,
        funder_addr: str,
        wallet_addresses: list[str],
    ) -> FunderCluster:
        """Build a cluster with calculated metrics and confidence."""
        # Determine funding depth for each wallet
        depths = []
        amounts = []
        timestamps = []

        for wallet in wallet_addresses:
            depth = self._get_funding_depth(graph, funder_addr, wallet)
            depths.append(depth)

            # Try to get funding amount/time from direct edge
            edge_data = self._get_edge_data(graph, funder_addr, wallet)
            if edge_data:
                if edge_data.get("amount"):
                    amounts.append(edge_data["amount"])
                if edge_data.get("timestamp"):
                    timestamps.append(edge_data["timestamp"])

        # Calculate metrics
        avg_depth = sum(depths) / len(depths) if depths else 1
        min_depth = min(depths) if depths else 1

        total_amount = sum(amounts)
        avg_amount = total_amount / len(amounts) if amounts else 0

        # Time span calculation
        time_span_hours = None
        if len(timestamps) >= 2:
            try:
                parsed_times = [
                    datetime.fromisoformat(t) if isinstance(t, str) else t
                    for t in timestamps
                ]
                time_delta = max(parsed_times) - min(parsed_times)
                time_span_hours = time_delta.total_seconds() / 3600
            except (ValueError, TypeError):
                pass

        # Calculate confidence score
        confidence = self._calculate_confidence(
            cluster_size=len(wallet_addresses),
            avg_depth=avg_depth,
            amounts=amounts,
            time_span_hours=time_span_hours,
        )

        return FunderCluster(
            funder_address=funder_addr,
            wallet_addresses=wallet_addresses,
            funding_depth=min_depth,
            total_funded_amount=total_amount,
            avg_funded_amount=avg_amount,
            funding_time_span_hours=time_span_hours,
            confidence=confidence,
        )

    def _get_funding_depth(
        self,
        graph: FundingGraph,
        funder: str,
        wallet: str,
    ) -> int:
        """Get the depth from funder to wallet (1 = direct)."""
        # Check direct funding
        if graph._graph.has_edge(funder, wallet):
            return 1

        # BFS to find shortest path depth
        visited = {funder}
        current_level = [funder]
        depth = 1

        while current_level and depth <= self.max_depth:
            next_level = []
            for node in current_level:
                for successor in graph._graph.successors(node):
                    if successor == wallet:
                        return depth + 1
                    if successor not in visited:
                        visited.add(successor)
                        next_level.append(successor)
            current_level = next_level
            depth += 1

        return self.max_depth + 1  # Not found within max depth

    def _get_edge_data(
        self,
        graph: FundingGraph,
        source: str,
        target: str,
    ) -> Optional[dict]:
        """Get edge data between two nodes."""
        if graph._graph.has_edge(source, target):
            return graph._graph.edges[source, target]
        return None

    def _calculate_confidence(
        self,
        cluster_size: int,
        avg_depth: float,
        amounts: list[int],
        time_span_hours: Optional[float],
    ) -> float:
        """
        Calculate confidence score for a cluster.

        Components:
        - Size factor: More wallets = higher confidence (log scale)
        - Depth factor: Direct funding = higher confidence
        - Amount factor: Consistent amounts = higher confidence
        - Timing factor: Rapid funding = higher confidence
        """
        # Size factor (0.2 - 0.4)
        # 2 wallets = 0.2, 10+ wallets = 0.4
        size_factor = min(0.2 + (cluster_size - 2) * 0.025, 0.4)

        # Depth factor (0.1 - 0.3)
        # Depth 1 = 0.3, Depth 2 = 0.2, Depth 3+ = 0.1
        depth_factor = max(0.4 - (avg_depth * 0.1), 0.1)

        # Amount consistency factor (0.0 - 0.2)
        amount_factor = 0.0
        if len(amounts) >= 2:
            avg_amt = sum(amounts) / len(amounts)
            if avg_amt > 0:
                # Coefficient of variation
                variance = sum((a - avg_amt) ** 2 for a in amounts) / len(amounts)
                std_dev = variance**0.5
                cv = std_dev / avg_amt

                # Lower CV = more consistent = higher factor
                # CV < 0.1 = very consistent, CV > 1.0 = very varied
                amount_factor = max(0.2 - (cv * 0.2), 0.0)

        # Timing factor (0.0 - 0.2)
        timing_factor = 0.0
        if time_span_hours is not None:
            if time_span_hours < 1:  # All within an hour
                timing_factor = 0.2
            elif time_span_hours < 24:  # Within a day
                timing_factor = 0.15
            elif time_span_hours < 168:  # Within a week
                timing_factor = 0.1
            else:
                timing_factor = 0.05

        total = size_factor + depth_factor + amount_factor + timing_factor

        # Normalize to 0-1 range (max possible = 1.1)
        return min(total / 1.1, 1.0)

    def detect_from_wallets(
        self,
        funding_graph: FundingGraph,
        wallets: Sequence[str],
        min_shared_for_cluster: int = 2,
    ) -> list[FunderCluster]:
        """
        Convenience method to detect clusters for specific wallets.

        Returns only clusters containing wallets from the input set.
        """
        result = self.detect(
            funding_graph,
            target_wallets=list(wallets),
        )
        return [
            c
            for c in result.clusters
            if any(w in wallets for w in c.wallet_addresses)
        ]
