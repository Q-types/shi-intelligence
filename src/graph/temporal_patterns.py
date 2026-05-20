"""
Temporal Coordination Detection for SHI.

Detects synchronized funding patterns that indicate sybil/coordinated behavior:
- Wallets funded within narrow time windows
- Burst funding patterns from single source
- Temporal clustering of activity
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

import numpy as np
import structlog

from .funding_graph import FundingGraph
from ..core.types import WalletAddress

logger = structlog.get_logger()


@dataclass
class TemporalCoordinationResult:
    """Result of temporal coordination analysis for a wallet."""

    wallet: WalletAddress

    # Synchronization score: 0 = no sync, 1 = perfectly synchronized with others
    temporal_sync_score: float

    # Time spread of funding events (hours)
    # Low spread + multiple funders = suspicious
    funding_time_spread_hours: float

    # Number of wallets funded within same time window as this wallet
    co_funded_wallet_count: int

    # Funding window details
    first_funding_time: datetime | None
    last_funding_time: datetime | None

    # Coordination cluster ID (wallets funded together)
    coordination_cluster_id: int | None


@dataclass
class TemporalCluster:
    """A cluster of wallets funded within a narrow time window."""

    cluster_id: int
    wallets: list[WalletAddress]
    funder: WalletAddress
    time_window_start: datetime
    time_window_end: datetime
    time_spread_seconds: float

    @property
    def size(self) -> int:
        return len(self.wallets)

    @property
    def is_suspicious(self) -> bool:
        """Cluster is suspicious if many wallets funded in short window."""
        # 3+ wallets funded within 1 hour is suspicious
        return self.size >= 3 and self.time_spread_seconds < 3600


def detect_temporal_coordination(
    graph: FundingGraph,
    wallets: list[WalletAddress],
    time_window_hours: float = 24.0,
    min_cluster_size: int = 3,
) -> dict[WalletAddress, TemporalCoordinationResult]:
    """
    Detect temporal coordination patterns in funding.

    Analyzes when wallets were funded and identifies clusters of wallets
    that received funding within narrow time windows - a strong sybil indicator.

    Args:
        graph: Funding graph with edge timestamps
        wallets: Wallets to analyze
        time_window_hours: Time window for clustering (default 24h)
        min_cluster_size: Minimum wallets for a coordination cluster

    Returns:
        Dict mapping wallet -> TemporalCoordinationResult
    """
    logger.info(
        "detecting_temporal_coordination",
        wallet_count=len(wallets),
        time_window_hours=time_window_hours,
    )

    nx_graph = graph._graph
    results: dict[WalletAddress, TemporalCoordinationResult] = {}

    # Step 1: Extract funding timestamps for each wallet
    wallet_funding_times: dict[str, list[tuple[str, datetime]]] = defaultdict(list)

    for wallet in wallets:
        if wallet not in nx_graph:
            continue

        # Get incoming edges (funding events)
        for source, _, data in nx_graph.in_edges(wallet, data=True):
            ts_str = data.get("timestamp")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    wallet_funding_times[wallet].append((source, ts))
                except (ValueError, TypeError):
                    continue

    # Step 2: Group wallets by funder and find temporal clusters
    funder_to_funded: dict[str, list[tuple[str, datetime]]] = defaultdict(list)

    for wallet, fundings in wallet_funding_times.items():
        for funder, ts in fundings:
            funder_to_funded[funder].append((wallet, ts))

    # Step 3: Detect temporal clusters per funder
    temporal_clusters: list[TemporalCluster] = []
    cluster_id = 0

    time_window = timedelta(hours=time_window_hours)

    for funder, funded_list in funder_to_funded.items():
        if len(funded_list) < min_cluster_size:
            continue

        # Sort by timestamp
        funded_list.sort(key=lambda x: x[1])

        # Sliding window clustering
        i = 0
        while i < len(funded_list):
            window_start = funded_list[i][1]
            window_end = window_start + time_window

            # Find all wallets funded within window
            cluster_wallets = []
            j = i
            while j < len(funded_list) and funded_list[j][1] <= window_end:
                cluster_wallets.append(funded_list[j][0])
                j += 1

            if len(cluster_wallets) >= min_cluster_size:
                # Found a coordination cluster
                actual_end = funded_list[j - 1][1]
                spread_seconds = (actual_end - window_start).total_seconds()

                temporal_clusters.append(
                    TemporalCluster(
                        cluster_id=cluster_id,
                        wallets=cluster_wallets,
                        funder=funder,
                        time_window_start=window_start,
                        time_window_end=actual_end,
                        time_spread_seconds=spread_seconds,
                    )
                )
                cluster_id += 1

                # Skip past this cluster
                i = j
            else:
                i += 1

    # Step 4: Compute wallet-level coordination metrics
    wallet_to_clusters: dict[str, list[TemporalCluster]] = defaultdict(list)
    for cluster in temporal_clusters:
        for wallet in cluster.wallets:
            wallet_to_clusters[wallet].append(cluster)

    for wallet in wallets:
        fundings = wallet_funding_times.get(wallet, [])

        # Compute time spread
        if fundings:
            times = [ts for _, ts in fundings]
            first_time = min(times)
            last_time = max(times)
            spread_hours = (last_time - first_time).total_seconds() / 3600
        else:
            first_time = None
            last_time = None
            spread_hours = 0.0

        # Find coordination clusters this wallet belongs to
        clusters = wallet_to_clusters.get(wallet, [])

        # Compute sync score based on cluster membership
        # Higher score = more coordinated with other wallets
        if clusters:
            # Score based on largest cluster size and how many clusters
            max_cluster_size = max(c.size for c in clusters)
            suspicious_clusters = [c for c in clusters if c.is_suspicious]

            # Sync score: combination of cluster size and suspiciousness
            base_score = min(max_cluster_size / 10.0, 1.0)  # Cap at 10 wallets
            suspicious_bonus = 0.3 * len(suspicious_clusters)
            sync_score = min(base_score + suspicious_bonus, 1.0)

            # Count co-funded wallets
            co_funded = set()
            for c in clusters:
                co_funded.update(c.wallets)
            co_funded.discard(wallet)

            cluster_id_val = clusters[0].cluster_id if clusters else None
        else:
            sync_score = 0.0
            co_funded = set()
            cluster_id_val = None

        results[wallet] = TemporalCoordinationResult(
            wallet=wallet,
            temporal_sync_score=sync_score,
            funding_time_spread_hours=spread_hours,
            co_funded_wallet_count=len(co_funded),
            first_funding_time=first_time,
            last_funding_time=last_time,
            coordination_cluster_id=cluster_id_val,
        )

    # Log summary
    suspicious_count = sum(
        1 for r in results.values() if r.temporal_sync_score >= 0.5
    )
    logger.info(
        "temporal_coordination_detected",
        total_wallets=len(results),
        suspicious_wallets=suspicious_count,
        temporal_clusters=len(temporal_clusters),
        suspicious_clusters=len([c for c in temporal_clusters if c.is_suspicious]),
    )

    return results


def find_synchronized_funding_groups(
    graph: FundingGraph,
    wallets: list[WalletAddress],
    time_threshold_seconds: float = 3600.0,  # 1 hour
    min_group_size: int = 3,
) -> list[list[WalletAddress]]:
    """
    Find groups of wallets that were funded at nearly the same time.

    This is a stricter version of temporal coordination detection,
    looking for very tight time clustering.

    Args:
        graph: Funding graph
        wallets: Wallets to analyze
        time_threshold_seconds: Maximum time difference to be considered "synchronized"
        min_group_size: Minimum wallets in a group

    Returns:
        List of wallet groups (each group is synchronized)
    """
    nx_graph = graph._graph

    # Extract funding times per funder
    funder_funding_events: dict[str, list[tuple[str, datetime]]] = defaultdict(list)

    for wallet in wallets:
        if wallet not in nx_graph:
            continue

        for source, _, data in nx_graph.in_edges(wallet, data=True):
            ts_str = data.get("timestamp")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    funder_funding_events[source].append((wallet, ts))
                except (ValueError, TypeError):
                    continue

    # Find synchronized groups
    synchronized_groups: list[list[WalletAddress]] = []

    for funder, events in funder_funding_events.items():
        if len(events) < min_group_size:
            continue

        # Sort by time
        events.sort(key=lambda x: x[1])

        # Use union-find style clustering based on time proximity
        groups: list[list[tuple[str, datetime]]] = []

        for wallet, ts in events:
            added = False
            for group in groups:
                # Check if within threshold of any wallet in group
                for _, group_ts in group:
                    if abs((ts - group_ts).total_seconds()) <= time_threshold_seconds:
                        group.append((wallet, ts))
                        added = True
                        break
                if added:
                    break

            if not added:
                groups.append([(wallet, ts)])

        # Extract groups meeting size threshold
        for group in groups:
            if len(group) >= min_group_size:
                synchronized_groups.append([w for w, _ in group])

    return synchronized_groups


def compute_funding_velocity(
    graph: FundingGraph,
    wallet: WalletAddress,
    window_hours: float = 24.0,
) -> Optional[float]:
    """
    Compute funding velocity (SOL per hour) for a wallet.

    High velocity in short window can indicate coordinated airdrop.

    Args:
        graph: Funding graph
        wallet: Wallet to analyze
        window_hours: Time window for velocity calculation

    Returns:
        Funding velocity in SOL/hour, or None if no data
    """
    nx_graph = graph._graph

    if wallet not in nx_graph:
        return None

    # Get funding events
    events: list[tuple[float, datetime]] = []
    for _, _, data in nx_graph.in_edges(wallet, data=True):
        amount = data.get("amount", 0)
        ts_str = data.get("timestamp")
        if ts_str and amount > 0:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                events.append((amount / 1e9, ts))  # Convert lamports to SOL
            except (ValueError, TypeError):
                continue

    if not events:
        return None

    # Get total and time span
    total_sol = sum(amt for amt, _ in events)
    times = [ts for _, ts in events]
    time_span_hours = (max(times) - min(times)).total_seconds() / 3600

    if time_span_hours == 0:
        # All at once - infinite velocity (return large number)
        return total_sol * 1000 if total_sol > 0 else None

    return total_sol / time_span_hours
