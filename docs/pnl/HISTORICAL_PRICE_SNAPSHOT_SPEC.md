# Historical Price Snapshot Specification

**Version**: 1.0.0
**Sprint**: 8 - Realised vs Unrealised Behaviour Intelligence
**Date**: 2026-05-21

## Overview

This document specifies the Historical Price Snapshot system for cost basis calculation and PnL analysis. The system stores versioned price observations that enable deterministic reconstruction of position values at any point in time.

## Hard Rules

1. **Price snapshots are IMMUTABLE** - never modified, only appended
2. **Price is NOT ground truth** - always includes confidence
3. **Missing price reduces confidence, does not become zero** - minimum 0.1
4. **All price data includes provenance** (source, fetched_at, payload_hash)

## Data Model

### PriceSnapshot Table

```sql
CREATE TABLE price_snapshots (
    id BIGSERIAL PRIMARY KEY,
    token_mint VARCHAR(44) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price_usd FLOAT NOT NULL,
    price_change_24h_pct FLOAT,
    liquidity_usd FLOAT,
    volume_24h_usd FLOAT,
    confidence_score FLOAT NOT NULL DEFAULT 0.5,  -- Never zero, min 0.1
    confidence_level VARCHAR(10) DEFAULT 'medium',
    source VARCHAR(30) NOT NULL,
    payload_hash VARCHAR(64),
    fetched_at TIMESTAMPTZ NOT NULL,
    staleness_seconds INTEGER DEFAULT 0,
    confidence_reason VARCHAR(500),
    data_version INTEGER DEFAULT 1,
    cadence_seconds INTEGER NOT NULL,
    sequence_in_token BIGINT NOT NULL
);

CREATE INDEX idx_price_snapshots_token_time ON price_snapshots(token_mint, timestamp);
CREATE INDEX idx_price_snapshots_token_seq ON price_snapshots(token_mint, sequence_in_token);
```

### PriceObservation (Input Structure)

```python
@dataclass(frozen=True)
class PriceObservation:
    token_mint: str
    timestamp: datetime
    price_usd: float
    price_change_24h_pct: float | None
    liquidity_usd: float | None
    volume_24h_usd: float | None
    confidence_score: float  # 0.0-1.0, never zero (min 0.1)
    confidence_level: str    # high/medium/low/none
    source: str              # jupiter, birdeye, pool_implied
    payload_hash: str | None
    fetched_at: datetime
    staleness_seconds: int
    confidence_reason: str | None
    data_version: int = 1
```

### HistoricalPriceQuery

```python
@dataclass(frozen=True)
class HistoricalPriceQuery:
    token_mint: str
    target_timestamp: datetime
    tolerance_seconds: int = 600  # ±10 minutes default
    min_confidence: float = 0.0   # Allow any confidence by default
```

### HistoricalPriceResult

```python
@dataclass(frozen=True)
class HistoricalPriceResult:
    found: bool
    price_usd: float | None
    confidence_score: float
    confidence_reason: str
    snapshot_timestamp: datetime | None
    time_delta_seconds: int | None
    source: str | None
    data_version: int
```

## Confidence Scoring

### Score Components

| Factor | Weight | Description |
|--------|--------|-------------|
| Source Reliability | 30% | Jupiter = 1.0, Birdeye = 0.9, Pool = 0.6 |
| Liquidity Backing | 40% | log10(liquidity_usd) / 6 |
| Volume Activity | 20% | 24h volume relative to position |
| Staleness | 10% | Penalty for stale data |

### Confidence Levels

| Level | Score Range | Meaning |
|-------|-------------|---------|
| HIGH | 0.8 - 1.0 | High liquidity, fresh data, reliable source |
| MEDIUM | 0.5 - 0.8 | Adequate liquidity, reasonable freshness |
| LOW | 0.2 - 0.5 | Low liquidity or stale data |
| NONE | 0.1 - 0.2 | Missing data, extrapolated |

### Hard Rule: Never Zero

```python
# Enforced in all confidence calculations
confidence_score = max(0.1, computed_confidence)
```

## Time Delta Handling

### Within Tolerance

When querying historical price, the system finds the nearest snapshot within the tolerance window:

```python
async def get_price_at_time(query, snapshots) -> HistoricalPriceResult:
    best_snapshot = None
    best_delta = float("inf")

    for snapshot in snapshots:
        delta = abs((snapshot.timestamp - query.target_timestamp).total_seconds())
        if delta <= query.tolerance_seconds and delta < best_delta:
            if snapshot.confidence_score >= query.min_confidence:
                best_snapshot = snapshot
                best_delta = delta

    # Apply time penalty to confidence
    time_penalty = best_delta / query.tolerance_seconds * 0.2  # Max 20%
    adjusted_confidence = max(0.1, best_snapshot.confidence_score - time_penalty)
```

### Interpolation

When exact timestamp not available but snapshots exist before and after:

```python
def interpolate_price(target, before, after) -> HistoricalPriceResult:
    # Linear interpolation
    ratio = (target - before.timestamp) / (after.timestamp - before.timestamp)
    interpolated_price = before.price_usd + ratio * (after.price_usd - before.price_usd)

    # Confidence reduced for interpolation
    base_confidence = min(before.confidence_score, after.confidence_score)
    span_seconds = (after.timestamp - before.timestamp).total_seconds()
    interpolation_penalty = min(0.3, span_seconds / 3600 * 0.1)

    return HistoricalPriceResult(
        found=True,
        price_usd=interpolated_price,
        confidence_score=max(0.1, base_confidence - interpolation_penalty),
        confidence_reason="interpolated between snapshots",
        ...
    )
```

## Snapshot Collection

### Integration with Snapshot Engine

The price snapshot collector integrates with the longitudinal snapshot engine cadence system:

```python
class PriceSnapshotCollector:
    async def collect(self, token_mint: str, cadence_seconds: int) -> PriceSnapshot | None:
        price_data = await self._price_provider.get_price(token_mint)

        observation = PriceObservation(
            token_mint=token_mint,
            timestamp=now,
            price_usd=price_data.price_usd,
            confidence_score=price_data.confidence_detail.confidence_score,
            source=price_data.source,
            ...
        )

        return await self._snapshot_service.record_price_snapshot(observation, cadence_seconds)
```

### Cadence Intervals

| Phase | Cadence | Rationale |
|-------|---------|-----------|
| Launch (0-4h) | 30s | High volatility period |
| Early (4-24h) | 5m | Settling period |
| Mature (24h+) | 15m | Stable tracking |

## Versioning

All price snapshots include `data_version` for schema evolution:

- **v1**: Initial schema with confidence scoring
- Future versions will increment for schema changes

Replay engine uses version to apply appropriate transformations.

## API Reference

### PriceSnapshotService

```python
class PriceSnapshotService:
    async def record_price_snapshot(
        observation: PriceObservation,
        cadence_seconds: int,
    ) -> PriceSnapshot

    async def get_price_at_time(
        query: HistoricalPriceQuery,
        snapshots: Sequence[PriceSnapshot],
    ) -> HistoricalPriceResult

    async def get_price_range(
        token_mint: str,
        start_time: datetime,
        end_time: datetime,
        snapshots: Sequence[PriceSnapshot],
    ) -> list[PriceSnapshot]

    async def get_peak_price(
        token_mint: str,
        since: datetime,
        snapshots: Sequence[PriceSnapshot],
    ) -> tuple[float | None, datetime | None]

    def interpolate_price(
        target_timestamp: datetime,
        before: PriceSnapshot | None,
        after: PriceSnapshot | None,
    ) -> HistoricalPriceResult
```

## Usage Examples

### Record Price Snapshot

```python
from src.longitudinal import PriceSnapshotService, PriceObservation

service = PriceSnapshotService()

observation = PriceObservation(
    token_mint="ABC123...",
    timestamp=datetime.now(timezone.utc),
    price_usd=0.00123,
    confidence_score=0.85,
    confidence_level="high",
    source="jupiter",
    fetched_at=datetime.now(timezone.utc),
    staleness_seconds=0,
    ...
)

snapshot = await service.record_price_snapshot(observation, cadence_seconds=300)
```

### Query Historical Price

```python
from src.longitudinal import HistoricalPriceQuery

query = HistoricalPriceQuery(
    token_mint="ABC123...",
    target_timestamp=trade_time,
    tolerance_seconds=600,
    min_confidence=0.3,
)

result = await service.get_price_at_time(query, available_snapshots)

if result.found:
    print(f"Price: ${result.price_usd} (confidence: {result.confidence_score})")
else:
    print(f"No price available: {result.confidence_reason}")
```

## Integration with Cost Basis

Price snapshots are used by the Cost Basis Calculator to determine entry prices:

```python
# For each buy trade, get historical price
for trade in buy_trades:
    query = HistoricalPriceQuery(
        token_mint=token_mint,
        target_timestamp=trade.timestamp,
    )
    result = await price_service.get_price_at_time(query, snapshots)

    lot = CostBasisLot(
        tokens=trade.tokens,
        price_usd=result.price_usd or trade.implied_price,
        confidence=result.confidence_score,
    )
```

## Replay Reproducibility

Price snapshots preserve deterministic replay:

1. **Immutable storage**: No modifications to historical snapshots
2. **Versioned schema**: `data_version` enables correct interpretation
3. **Complete provenance**: `payload_hash` enables verification
4. **Sequence tracking**: `sequence_in_token` ensures ordering

```python
# Replay at historical point
snapshots_at_time = await service.get_price_range(
    token_mint=mint,
    start_time=replay_start,
    end_time=replay_end,
    snapshots=all_snapshots,
)

# Reconstruct state using only snapshots available at that time
```
