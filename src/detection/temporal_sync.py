"""
Temporal Synchronization Detection.

Identifies wallets that trade within narrow time windows across multiple tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence
from collections import defaultdict
import statistics

import structlog

logger = structlog.get_logger()


@dataclass
class TradeEvent:
    """A single trade event for temporal analysis."""

    wallet_address: str
    token_mint: str
    timestamp: datetime
    trade_type: str  # "buy" or "sell"


@dataclass
class TemporalCluster:
    """A cluster of temporally synchronized wallets."""

    wallet_addresses: list[str]
    coordination_score: float  # 0-1, higher = more synchronized
    tokens_coordinated: list[str]  # Tokens where coordination was detected
    avg_time_gap_seconds: float
    max_time_gap_seconds: float
    coordination_events: int  # Number of coordinated trade windows
    confidence: float


@dataclass
class TemporalSyncResult:
    """Result of temporal synchronization detection."""

    clusters: list[TemporalCluster]
    total_wallets_analyzed: int
    total_wallets_clustered: int
    total_events_analyzed: int
    detection_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class TemporalSyncDetector:
    """
    Detects wallet clusters based on temporal trading patterns.

    Identifies wallets that consistently trade within narrow time windows,
    suggesting coordinated or automated control.

    Detection approach:
    1. Group trades by token
    2. Find trades occurring within time_window_seconds
    3. Count how often wallet pairs appear together
    4. Score based on frequency and consistency

    Confidence scoring considers:
    - Number of coordinated events (more = higher confidence)
    - Consistency of time gaps (lower variance = higher confidence)
    - Number of tokens where coordination seen (more = higher confidence)
    """

    def __init__(
        self,
        time_window_seconds: int = 300,  # 5 minutes
        min_coordination_events: int = 3,
        min_tokens_shared: int = 2,
        min_confidence: float = 0.5,
    ):
        self.time_window_seconds = time_window_seconds
        self.min_coordination_events = min_coordination_events
        self.min_tokens_shared = min_tokens_shared
        self.min_confidence = min_confidence

    def detect(
        self,
        trade_events: Sequence[TradeEvent],
        target_wallets: Optional[list[str]] = None,
    ) -> TemporalSyncResult:
        """
        Detect temporally synchronized wallet clusters.

        Args:
            trade_events: List of trade events to analyze
            target_wallets: Specific wallets to analyze (default: all)

        Returns:
            TemporalSyncResult with detected clusters
        """
        if not trade_events:
            return TemporalSyncResult(
                clusters=[],
                total_wallets_analyzed=0,
                total_wallets_clustered=0,
                total_events_analyzed=0,
            )

        # Filter to target wallets if specified
        if target_wallets:
            target_set = set(target_wallets)
            trade_events = [e for e in trade_events if e.wallet_address in target_set]

        # Get all unique wallets
        all_wallets = list(set(e.wallet_address for e in trade_events))

        # Group events by token
        events_by_token: dict[str, list[TradeEvent]] = defaultdict(list)
        for event in trade_events:
            events_by_token[event.token_mint].append(event)

        # Sort each token's events by timestamp
        for token in events_by_token:
            events_by_token[token].sort(key=lambda e: e.timestamp)

        # Find coordination pairs
        pair_coordination = self._find_coordination_pairs(events_by_token)

        # Build clusters from pairs
        clusters = self._build_clusters(pair_coordination, all_wallets)

        # Filter by confidence
        clusters = [c for c in clusters if c.confidence >= self.min_confidence]

        # Sort by confidence descending
        clusters.sort(key=lambda c: c.confidence, reverse=True)

        # Count clustered wallets
        clustered = set()
        for cluster in clusters:
            clustered.update(cluster.wallet_addresses)

        result = TemporalSyncResult(
            clusters=clusters,
            total_wallets_analyzed=len(all_wallets),
            total_wallets_clustered=len(clustered),
            total_events_analyzed=len(trade_events),
        )

        logger.info(
            "temporal_sync_detection_complete",
            wallets_analyzed=result.total_wallets_analyzed,
            events_analyzed=result.total_events_analyzed,
            clusters_found=len(clusters),
            wallets_clustered=result.total_wallets_clustered,
        )

        return result

    def _find_coordination_pairs(
        self,
        events_by_token: dict[str, list[TradeEvent]],
    ) -> dict[tuple[str, str], dict]:
        """
        Find pairs of wallets that trade together within time windows.

        Returns dict mapping (wallet1, wallet2) -> coordination info
        """
        # pair -> {tokens: set, time_gaps: list, event_count: int}
        pair_data: dict[tuple[str, str], dict] = {}

        for token, events in events_by_token.items():
            # Find trades within time windows
            windows = self._find_time_windows(events)

            for window in windows:
                if len(window) < 2:
                    continue

                # Get all wallet pairs in this window
                wallets_in_window = list(set(e.wallet_address for e in window))

                for i, w1 in enumerate(wallets_in_window):
                    for w2 in wallets_in_window[i + 1 :]:
                        # Normalize pair order
                        pair = tuple(sorted([w1, w2]))

                        if pair not in pair_data:
                            pair_data[pair] = {
                                "tokens": set(),
                                "time_gaps": [],
                                "event_count": 0,
                            }

                        pair_data[pair]["tokens"].add(token)
                        pair_data[pair]["event_count"] += 1

                        # Calculate time gap between these wallets' trades
                        w1_events = [e for e in window if e.wallet_address == w1]
                        w2_events = [e for e in window if e.wallet_address == w2]

                        for e1 in w1_events:
                            for e2 in w2_events:
                                gap = abs(
                                    (e1.timestamp - e2.timestamp).total_seconds()
                                )
                                pair_data[pair]["time_gaps"].append(gap)

        return pair_data

    def _find_time_windows(
        self,
        events: list[TradeEvent],
    ) -> list[list[TradeEvent]]:
        """
        Group events into time windows.

        Events within time_window_seconds of each other form a window.
        """
        if not events:
            return []

        windows = []
        current_window = [events[0]]

        for event in events[1:]:
            last_time = current_window[-1].timestamp
            if (event.timestamp - last_time).total_seconds() <= self.time_window_seconds:
                current_window.append(event)
            else:
                if len(current_window) >= 2:
                    windows.append(current_window)
                current_window = [event]

        if len(current_window) >= 2:
            windows.append(current_window)

        return windows

    def _build_clusters(
        self,
        pair_coordination: dict[tuple[str, str], dict],
        all_wallets: list[str],
    ) -> list[TemporalCluster]:
        """Build clusters from coordination pairs using union-find."""
        # Filter pairs to those meeting thresholds
        strong_pairs = {
            pair: data
            for pair, data in pair_coordination.items()
            if (
                data["event_count"] >= self.min_coordination_events
                and len(data["tokens"]) >= self.min_tokens_shared
            )
        }

        if not strong_pairs:
            return []

        # Union-find for clustering
        parent = {w: w for w in all_wallets}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union wallets in strong pairs
        for (w1, w2) in strong_pairs:
            union(w1, w2)

        # Group wallets by their root
        cluster_wallets: dict[str, list[str]] = defaultdict(list)
        for wallet in all_wallets:
            root = find(wallet)
            cluster_wallets[root].append(wallet)

        # Build cluster objects
        clusters = []
        for root, wallets in cluster_wallets.items():
            if len(wallets) < 2:
                continue

            # Aggregate pair data for this cluster
            cluster_pairs = [
                (pair, data)
                for pair, data in strong_pairs.items()
                if pair[0] in wallets and pair[1] in wallets
            ]

            if not cluster_pairs:
                continue

            # Combine metrics
            all_tokens = set()
            all_time_gaps = []
            total_events = 0

            for pair, data in cluster_pairs:
                all_tokens.update(data["tokens"])
                all_time_gaps.extend(data["time_gaps"])
                total_events += data["event_count"]

            avg_gap = statistics.mean(all_time_gaps) if all_time_gaps else 0
            max_gap = max(all_time_gaps) if all_time_gaps else 0

            # Calculate coordination score
            coordination_score = self._calculate_coordination_score(
                cluster_size=len(wallets),
                event_count=total_events,
                token_count=len(all_tokens),
                time_gaps=all_time_gaps,
            )

            # Calculate confidence
            confidence = self._calculate_confidence(
                coordination_score=coordination_score,
                event_count=total_events,
                token_count=len(all_tokens),
            )

            clusters.append(
                TemporalCluster(
                    wallet_addresses=wallets,
                    coordination_score=coordination_score,
                    tokens_coordinated=list(all_tokens),
                    avg_time_gap_seconds=avg_gap,
                    max_time_gap_seconds=max_gap,
                    coordination_events=total_events,
                    confidence=confidence,
                )
            )

        return clusters

    def _calculate_coordination_score(
        self,
        cluster_size: int,
        event_count: int,
        token_count: int,
        time_gaps: list[float],
    ) -> float:
        """
        Calculate coordination score (0-1).

        Higher score = more synchronized behavior.
        """
        # Time gap component (0-0.4)
        # Smaller gaps = higher score
        if time_gaps:
            avg_gap = statistics.mean(time_gaps)
            # 0 seconds = 0.4, 300 seconds = 0
            time_component = max(0.4 - (avg_gap / 750), 0)
        else:
            time_component = 0

        # Event density component (0-0.3)
        # More events per wallet pair = higher score
        expected_pairs = cluster_size * (cluster_size - 1) / 2
        events_per_pair = event_count / max(expected_pairs, 1)
        # 10+ events per pair = max score
        event_component = min(events_per_pair / 10 * 0.3, 0.3)

        # Token breadth component (0-0.3)
        # More tokens = higher score
        # 5+ tokens = max score
        token_component = min(token_count / 5 * 0.3, 0.3)

        return time_component + event_component + token_component

    def _calculate_confidence(
        self,
        coordination_score: float,
        event_count: int,
        token_count: int,
    ) -> float:
        """
        Calculate confidence in the cluster detection.

        Based on amount of evidence supporting the detection.
        """
        # Base on coordination score (0-0.5)
        base = coordination_score * 0.5

        # Event evidence bonus (0-0.25)
        # 20+ events = max bonus
        event_bonus = min(event_count / 20 * 0.25, 0.25)

        # Token evidence bonus (0-0.25)
        # 5+ tokens = max bonus
        token_bonus = min(token_count / 5 * 0.25, 0.25)

        return min(base + event_bonus + token_bonus, 1.0)

    def detect_for_wallet_pair(
        self,
        trade_events: Sequence[TradeEvent],
        wallet1: str,
        wallet2: str,
    ) -> Optional[float]:
        """
        Calculate coordination score between two specific wallets.

        Returns coordination score or None if insufficient data.
        """
        # Filter to just these wallets
        relevant_events = [
            e
            for e in trade_events
            if e.wallet_address in (wallet1, wallet2)
        ]

        if len(relevant_events) < 4:  # Need at least 2 from each
            return None

        result = self.detect(relevant_events, [wallet1, wallet2])

        if result.clusters:
            return result.clusters[0].coordination_score
        return None
