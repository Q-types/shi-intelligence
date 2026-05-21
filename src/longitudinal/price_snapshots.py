"""
Historical Price Snapshot Service (Sprint 8).

Stores versioned price observations for cost basis calculation and PnL analysis.
Integrates with the snapshot engine cadence system.

HARD RULES:
1. Price snapshots are IMMUTABLE - never modified, only appended
2. Price is NOT ground truth - always includes confidence
3. Missing price reduces confidence, does not become zero
4. All price data includes provenance (source, fetched_at, payload_hash)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Sequence

import structlog

from .models import PriceSnapshot, PriceConfidenceLevel

logger = structlog.get_logger()


# ============================================================================
# Data Structures
# ============================================================================


@dataclass(frozen=True)
class PriceObservation:
    """
    Immutable price observation for snapshot storage.

    Captures all Sprint 7 PriceData fields plus snapshot context.
    """

    token_mint: str
    timestamp: datetime
    price_usd: float
    price_change_24h_pct: float | None
    liquidity_usd: float | None
    volume_24h_usd: float | None
    confidence_score: float  # 0.0-1.0, never zero (min 0.1)
    confidence_level: str  # high/medium/low/none
    source: str  # jupiter, birdeye, pool_implied
    payload_hash: str | None
    fetched_at: datetime
    staleness_seconds: int
    confidence_reason: str | None
    data_version: int = 1


@dataclass(frozen=True)
class HistoricalPriceQuery:
    """Query parameters for historical price lookup."""

    token_mint: str
    target_timestamp: datetime
    tolerance_seconds: int = 600  # ±10 minutes default
    min_confidence: float = 0.0  # Allow any confidence by default


@dataclass(frozen=True)
class HistoricalPriceResult:
    """Result of historical price lookup."""

    found: bool
    price_usd: float | None
    confidence_score: float
    confidence_reason: str
    snapshot_timestamp: datetime | None
    time_delta_seconds: int | None
    source: str | None
    data_version: int


# ============================================================================
# Price Snapshot Service
# ============================================================================


class PriceSnapshotService:
    """
    Service for storing and querying historical price snapshots.

    Integrates with the longitudinal snapshot engine cadence system.
    All snapshots are immutable - never modified, only appended.
    """

    def __init__(self, session_factory=None):
        """
        Initialize the price snapshot service.

        Args:
            session_factory: SQLAlchemy session factory (optional, for DB operations)
        """
        self._session_factory = session_factory
        self._sequence_counters: dict[str, int] = {}  # token_mint -> sequence

    async def record_price_snapshot(
        self,
        observation: PriceObservation,
        cadence_seconds: int,
    ) -> PriceSnapshot:
        """
        Record a new price snapshot.

        The snapshot is immutable once stored. This method returns the
        created snapshot record.

        Args:
            observation: Price observation to store
            cadence_seconds: Snapshot cadence interval (from snapshot engine)

        Returns:
            Created PriceSnapshot record
        """
        # Get next sequence number for this token
        sequence = self._get_next_sequence(observation.token_mint)

        # Ensure confidence is never zero (Sprint 8 hard rule)
        confidence_score = max(0.1, observation.confidence_score)

        snapshot = PriceSnapshot(
            token_mint=observation.token_mint,
            timestamp=observation.timestamp,
            price_usd=observation.price_usd,
            price_change_24h_pct=observation.price_change_24h_pct,
            liquidity_usd=observation.liquidity_usd,
            volume_24h_usd=observation.volume_24h_usd,
            confidence_score=confidence_score,
            confidence_level=observation.confidence_level,
            source=observation.source,
            payload_hash=observation.payload_hash,
            fetched_at=observation.fetched_at,
            staleness_seconds=observation.staleness_seconds,
            confidence_reason=observation.confidence_reason,
            data_version=observation.data_version,
            cadence_seconds=cadence_seconds,
            sequence_in_token=sequence,
        )

        logger.debug(
            "price_snapshot_recorded",
            token_mint=observation.token_mint[:8],
            price_usd=observation.price_usd,
            confidence=confidence_score,
            sequence=sequence,
        )

        return snapshot

    def _get_next_sequence(self, token_mint: str) -> int:
        """Get and increment sequence number for a token."""
        current = self._sequence_counters.get(token_mint, 0)
        self._sequence_counters[token_mint] = current + 1
        return current + 1

    async def get_price_at_time(
        self,
        query: HistoricalPriceQuery,
        snapshots: Sequence[PriceSnapshot],
    ) -> HistoricalPriceResult:
        """
        Get historical price closest to target timestamp.

        Uses tolerance window to find nearest snapshot.
        Returns confidence-weighted result.

        Args:
            query: Query parameters
            snapshots: Available price snapshots to search

        Returns:
            HistoricalPriceResult with price and confidence
        """
        if not snapshots:
            return HistoricalPriceResult(
                found=False,
                price_usd=None,
                confidence_score=0.0,
                confidence_reason="No price snapshots available",
                snapshot_timestamp=None,
                time_delta_seconds=None,
                source=None,
                data_version=1,
            )

        # Filter to matching token
        token_snapshots = [
            s for s in snapshots
            if s.token_mint == query.token_mint
        ]

        if not token_snapshots:
            return HistoricalPriceResult(
                found=False,
                price_usd=None,
                confidence_score=0.0,
                confidence_reason=f"No snapshots for token {query.token_mint[:8]}",
                snapshot_timestamp=None,
                time_delta_seconds=None,
                source=None,
                data_version=1,
            )

        # Find closest snapshot within tolerance
        best_snapshot = None
        best_delta = float("inf")

        for snapshot in token_snapshots:
            delta = abs((snapshot.timestamp - query.target_timestamp).total_seconds())
            if delta <= query.tolerance_seconds and delta < best_delta:
                if snapshot.confidence_score >= query.min_confidence:
                    best_snapshot = snapshot
                    best_delta = delta

        if best_snapshot is None:
            return HistoricalPriceResult(
                found=False,
                price_usd=None,
                confidence_score=0.0,
                confidence_reason=f"No snapshot within {query.tolerance_seconds}s tolerance",
                snapshot_timestamp=None,
                time_delta_seconds=None,
                source=None,
                data_version=1,
            )

        # Adjust confidence based on time delta
        time_penalty = best_delta / query.tolerance_seconds * 0.2  # Max 20% penalty
        adjusted_confidence = max(0.1, best_snapshot.confidence_score - time_penalty)

        confidence_reasons = [best_snapshot.confidence_reason or ""]
        if best_delta > 60:
            confidence_reasons.append(f"interpolated from {int(best_delta)}s away")

        return HistoricalPriceResult(
            found=True,
            price_usd=best_snapshot.price_usd,
            confidence_score=adjusted_confidence,
            confidence_reason="; ".join(filter(None, confidence_reasons)),
            snapshot_timestamp=best_snapshot.timestamp,
            time_delta_seconds=int(best_delta),
            source=best_snapshot.source,
            data_version=best_snapshot.data_version,
        )

    async def get_price_range(
        self,
        token_mint: str,
        start_time: datetime,
        end_time: datetime,
        snapshots: Sequence[PriceSnapshot],
    ) -> list[PriceSnapshot]:
        """
        Get all price snapshots in a time range.

        Args:
            token_mint: Token to query
            start_time: Range start
            end_time: Range end
            snapshots: Available snapshots to filter

        Returns:
            List of matching snapshots, sorted by timestamp
        """
        matching = [
            s for s in snapshots
            if s.token_mint == token_mint
            and start_time <= s.timestamp <= end_time
        ]

        return sorted(matching, key=lambda s: s.timestamp)

    async def get_peak_price(
        self,
        token_mint: str,
        since: datetime,
        snapshots: Sequence[PriceSnapshot],
    ) -> tuple[float | None, datetime | None]:
        """
        Get peak price since a given time.

        Used for exit efficiency calculation.

        Args:
            token_mint: Token to query
            since: Start time for peak search
            snapshots: Available snapshots

        Returns:
            Tuple of (peak_price, peak_timestamp) or (None, None)
        """
        matching = [
            s for s in snapshots
            if s.token_mint == token_mint and s.timestamp >= since
        ]

        if not matching:
            return None, None

        peak_snapshot = max(matching, key=lambda s: s.price_usd)
        return peak_snapshot.price_usd, peak_snapshot.timestamp

    def interpolate_price(
        self,
        target_timestamp: datetime,
        before: PriceSnapshot | None,
        after: PriceSnapshot | None,
    ) -> HistoricalPriceResult:
        """
        Interpolate price between two snapshots.

        Uses linear interpolation with reduced confidence.

        Args:
            target_timestamp: Time to interpolate for
            before: Snapshot before target time
            after: Snapshot after target time

        Returns:
            Interpolated price result with reduced confidence
        """
        if before is None and after is None:
            return HistoricalPriceResult(
                found=False,
                price_usd=None,
                confidence_score=0.0,
                confidence_reason="No snapshots available for interpolation",
                snapshot_timestamp=None,
                time_delta_seconds=None,
                source=None,
                data_version=1,
            )

        if before is None:
            # Only after available, use with penalty
            delta = (after.timestamp - target_timestamp).total_seconds()
            return HistoricalPriceResult(
                found=True,
                price_usd=after.price_usd,
                confidence_score=max(0.1, after.confidence_score * 0.7),
                confidence_reason=f"extrapolated from {int(delta)}s after",
                snapshot_timestamp=after.timestamp,
                time_delta_seconds=int(delta),
                source=after.source,
                data_version=after.data_version,
            )

        if after is None:
            # Only before available, use with penalty
            delta = (target_timestamp - before.timestamp).total_seconds()
            return HistoricalPriceResult(
                found=True,
                price_usd=before.price_usd,
                confidence_score=max(0.1, before.confidence_score * 0.7),
                confidence_reason=f"extrapolated from {int(delta)}s before",
                snapshot_timestamp=before.timestamp,
                time_delta_seconds=int(delta),
                source=before.source,
                data_version=before.data_version,
            )

        # Linear interpolation
        before_time = before.timestamp.timestamp()
        after_time = after.timestamp.timestamp()
        target_time = target_timestamp.timestamp()

        if after_time == before_time:
            # Same timestamp, use average
            interpolated_price = (before.price_usd + after.price_usd) / 2
            interpolated_confidence = min(before.confidence_score, after.confidence_score)
        else:
            # Linear interpolation
            ratio = (target_time - before_time) / (after_time - before_time)
            interpolated_price = before.price_usd + ratio * (after.price_usd - before.price_usd)

            # Confidence is min of both, with interpolation penalty
            base_confidence = min(before.confidence_score, after.confidence_score)
            span_seconds = after_time - before_time
            interpolation_penalty = min(0.3, span_seconds / 3600 * 0.1)  # Max 30% penalty
            interpolated_confidence = max(0.1, base_confidence - interpolation_penalty)

        return HistoricalPriceResult(
            found=True,
            price_usd=interpolated_price,
            confidence_score=interpolated_confidence,
            confidence_reason=f"interpolated between snapshots",
            snapshot_timestamp=target_timestamp,
            time_delta_seconds=0,
            source=f"{before.source}+{after.source}",
            data_version=max(before.data_version, after.data_version),
        )


# ============================================================================
# Price Snapshot Collector (for snapshot engine integration)
# ============================================================================


class PriceSnapshotCollector:
    """
    Collector for integrating price snapshots with the snapshot engine.

    Called during snapshot collection to capture current price.
    """

    def __init__(
        self,
        price_provider=None,
        snapshot_service: PriceSnapshotService | None = None,
    ):
        """
        Initialize the collector.

        Args:
            price_provider: Price provider (JupiterPriceProvider or chain)
            snapshot_service: Price snapshot service for storage
        """
        self._price_provider = price_provider
        self._snapshot_service = snapshot_service or PriceSnapshotService()

    async def collect(
        self,
        token_mint: str,
        cadence_seconds: int,
    ) -> PriceSnapshot | None:
        """
        Collect price snapshot for a token.

        Called by snapshot engine at configured cadence.

        Args:
            token_mint: Token to collect price for
            cadence_seconds: Current snapshot cadence

        Returns:
            Created PriceSnapshot or None if price unavailable
        """
        if self._price_provider is None:
            logger.warning("price_collector_no_provider")
            return None

        try:
            # Fetch current price
            price_data = await self._price_provider.get_price(token_mint)

            if price_data is None:
                logger.debug("price_not_available", token_mint=token_mint[:8])
                return None

            # Build observation from PriceData
            now = datetime.now(timezone.utc)

            observation = PriceObservation(
                token_mint=token_mint,
                timestamp=now,
                price_usd=price_data.price_usd,
                price_change_24h_pct=price_data.price_change_24h_pct,
                liquidity_usd=price_data.liquidity_usd,
                volume_24h_usd=price_data.volume_24h_usd,
                confidence_score=price_data.confidence_detail.confidence_score if price_data.confidence_detail else 0.5,
                confidence_level=price_data.confidence.value,
                source=price_data.source,
                payload_hash=price_data.payload_hash,
                fetched_at=price_data.fetched_at,
                staleness_seconds=price_data.staleness_seconds,
                confidence_reason=price_data.confidence_detail.confidence_reason if price_data.confidence_detail else None,
                data_version=price_data.data_version,
            )

            # Store snapshot
            snapshot = await self._snapshot_service.record_price_snapshot(
                observation, cadence_seconds
            )

            return snapshot

        except Exception as e:
            logger.error("price_collection_failed", token_mint=token_mint[:8], error=str(e))
            return None
