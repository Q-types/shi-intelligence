# Realised PnL Implementation

**Version**: 1.0.0
**Sprint**: 8 - Realised vs Unrealised Behaviour Intelligence
**Date**: 2026-05-21

## Overview

This document describes the Realised PnL calculation system for tracking actual profit/loss when wallets exit positions. The system computes PnL using cost basis lots, exit efficiency metrics, and liquidity-adjusted estimates.

## Hard Rules

1. **Separate realised from unrealised** - distinct computation and storage
2. **Confidence propagates through exit** - entry and exit confidence both matter
3. **Low confidence shows ranges, not precise values** - protect against false precision
4. **Accounting method explicit** - recorded on every PnL record

## Core Concepts

### Realised vs Unrealised

| Metric | Definition | When Updated |
|--------|------------|--------------|
| **Unrealised PnL** | (current_price - avg_entry) × position | On price update |
| **Realised PnL** | (exit_price - entry_cost_basis) × exit_tokens | On sell event |

### Exit Efficiency

Measures how well-timed the exit was relative to peak price:

```
exit_efficiency = exit_price / peak_price
```

| Efficiency | Meaning |
|------------|---------|
| 0.9 - 1.0 | Excellent timing, near peak |
| 0.7 - 0.9 | Good timing |
| 0.5 - 0.7 | Moderate timing |
| < 0.5 | Poor timing, significant drawdown |

### Liquidity-Adjusted PnL

Accounts for potential slippage based on position size vs liquidity:

```python
position_vs_liquidity = exit_value / liquidity_at_exit_usd
estimated_slippage_pct = min(0.5, position_vs_liquidity * 0.1)
liquidity_adjusted_pnl = realised_pnl * (1 - estimated_slippage_pct)
```

## Data Structures

### RealisedPnLEstimate

```python
@dataclass(frozen=True)
class RealisedPnLEstimate:
    exit_tokens: int
    exit_price_usd: float
    exit_value_usd: float
    entry_price_usd: float
    cost_basis_usd: float
    realised_pnl_usd: float
    realised_pnl_pct: float
    accounting_method: AccountingMethod

    # Exit quality metrics
    exit_efficiency: float | None
    peak_price_usd: float | None
    peak_to_exit_drawdown_pct: float | None

    # Liquidity context
    liquidity_at_exit_usd: float | None
    liquidity_adjusted_pnl_usd: float | None

    # Confidence tracking
    entry_price_confidence: float
    exit_price_confidence: float
    overall_confidence: float
    confidence_reason: str

    # Partial exit tracking
    is_partial_exit: bool
    remaining_tokens: int
    lots_consumed: list[dict]
```

### RealisedPnLRecord (Database Model)

```python
class RealisedPnLRecord(Base):
    __tablename__ = "realised_pnl_records"

    id: int
    wallet_address: str
    token_mint: str
    exit_timestamp: datetime
    exit_tokens: int
    exit_price_usd: float
    entry_price_usd: float
    cost_basis_usd: float
    realised_pnl_usd: float
    realised_pnl_pct: float
    exit_efficiency: float | None
    peak_price_usd: float | None
    liquidity_at_exit_usd: float | None
    liquidity_adjusted_pnl_usd: float | None
    accounting_method: str
    entry_price_confidence: float
    exit_price_confidence: float
    overall_confidence: float
    confidence_reason: str
    created_at: datetime
```

## PnL Calculation

### Lot Matching

The calculator matches exit tokens to cost basis lots based on accounting method:

```python
def _match_lots_to_exit(exit_tokens, lots, method):
    if method == AccountingMethod.FIFO:
        sorted_lots = sorted(lots, key=lambda l: l["timestamp"])
    elif method == AccountingMethod.LIFO:
        sorted_lots = sorted(lots, key=lambda l: l["timestamp"], reverse=True)
    else:
        return _match_weighted_average(exit_tokens, lots)

    lots_consumed = []
    total_cost = 0.0
    total_confidence_weighted = 0.0
    tokens_matched = 0

    for lot in sorted_lots:
        if tokens_matched >= exit_tokens:
            break
        available = lot["remaining_tokens"]
        matched = min(available, exit_tokens - tokens_matched)

        lots_consumed.append({
            "lot_timestamp": lot["timestamp"],
            "tokens_consumed": matched,
            "price_usd": lot["price_usd"],
            "confidence": lot["confidence"],
        })

        total_cost += matched * lot["price_usd"]
        total_confidence_weighted += matched * lot["confidence"]
        tokens_matched += matched

    weighted_entry_price = total_cost / tokens_matched
    weighted_confidence = total_confidence_weighted / tokens_matched

    return lots_consumed, weighted_entry_price, weighted_confidence
```

### PnL Computation

```python
def compute_realised_pnl(exit_tokens, exit_price_usd, exit_price_confidence, cost_basis_lots, method):
    # Match lots to exit
    lots_consumed, entry_price, entry_confidence, remaining = _match_lots_to_exit(
        exit_tokens, cost_basis_lots, method
    )

    # Compute PnL
    exit_value = exit_tokens * exit_price_usd
    cost_basis = exit_tokens * entry_price
    realised_pnl_usd = exit_value - cost_basis
    realised_pnl_pct = (exit_price_usd - entry_price) / entry_price

    # Overall confidence = min(entry, exit)
    overall_confidence = min(entry_confidence, exit_price_confidence)

    return RealisedPnLEstimate(
        realised_pnl_usd=realised_pnl_usd,
        realised_pnl_pct=realised_pnl_pct,
        overall_confidence=overall_confidence,
        ...
    )
```

## Confidence Handling

### Confidence Sources

| Source | Typical Range | Impact |
|--------|---------------|--------|
| Entry price (from historical snapshot) | 0.5 - 0.95 | Affects cost basis accuracy |
| Exit price (current price) | 0.7 - 1.0 | Affects exit value accuracy |
| Overall | min(entry, exit) | Determines display mode |

### Display Mode

```python
if overall_confidence >= min_confidence_for_display:
    # Show precise values
    return {"realised_pnl_usd": 150.0}
else:
    # Show ranges to avoid false precision
    width = (1 - overall_confidence) * abs(value) * 2
    return {"realised_pnl_range": (value - width/2, value + width/2)}
```

### Confidence Reason Building

```python
confidence_reasons = []

if overall_confidence >= 0.8:
    confidence_reasons.append("high confidence")
elif overall_confidence >= 0.5:
    confidence_reasons.append("medium confidence")
else:
    confidence_reasons.append("LOW CONFIDENCE - treat as estimate")

confidence_reasons.append(f"{method.value} accounting")
confidence_reasons.append(f"{len(lots_consumed)} lots matched")
```

## API Reference

### RealisedPnLCalculator

```python
class RealisedPnLCalculator:
    def __init__(
        self,
        default_method: AccountingMethod = AccountingMethod.FIFO,
        min_confidence_for_precise: float = 0.6,
    )

    def compute_realised_pnl(
        self,
        exit_tokens: int,
        exit_price_usd: float,
        exit_price_confidence: float,
        cost_basis_lots: list[dict],
        method: AccountingMethod | None = None,
        peak_price_usd: float | None = None,
        liquidity_at_exit_usd: float | None = None,
    ) -> RealisedPnLEstimate
```

## Usage Examples

### Basic PnL Calculation

```python
from src.longitudinal import RealisedPnLCalculator, AccountingMethod

calculator = RealisedPnLCalculator()

cost_basis_lots = [
    {
        "timestamp": datetime(2026, 5, 1, 10, 0),
        "original_tokens": 1000,
        "remaining_tokens": 1000,
        "price_usd": 0.001,
        "confidence": 0.9,
        "source": "jupiter",
    },
]

result = calculator.compute_realised_pnl(
    exit_tokens=500,
    exit_price_usd=0.003,
    exit_price_confidence=0.85,
    cost_basis_lots=cost_basis_lots,
    method=AccountingMethod.FIFO,
    peak_price_usd=0.004,
    liquidity_at_exit_usd=50000,
)

print(f"Realised PnL: ${result.realised_pnl_usd:.2f} ({result.realised_pnl_pct:.1%})")
print(f"Exit efficiency: {result.exit_efficiency:.1%}")
print(f"Liquidity-adjusted PnL: ${result.liquidity_adjusted_pnl_usd:.2f}")
print(f"Confidence: {result.overall_confidence:.2f}")
```

### Partial Exit Handling

```python
# First exit: 500 of 1500 tokens
result1 = calculator.compute_realised_pnl(
    exit_tokens=500,
    exit_price_usd=0.003,
    ...
)
print(f"Partial exit: {result1.is_partial_exit}")  # True
print(f"Remaining: {result1.remaining_tokens}")    # 1000

# Second exit: remaining 1000 tokens
result2 = calculator.compute_realised_pnl(
    exit_tokens=1000,
    exit_price_usd=0.002,
    ...
)
```

## Integration Points

### Event Store Integration

When sell events are detected, trigger PnL calculation:

```python
async def on_sell_event(event: TradeEvent):
    # Get cost basis lots for wallet/token
    lots = await get_cost_basis_lots(event.wallet, event.token_mint)

    # Get current price and confidence
    price_result = await price_service.get_price_at_time(
        HistoricalPriceQuery(
            token_mint=event.token_mint,
            target_timestamp=event.timestamp,
        ),
        available_snapshots,
    )

    # Calculate PnL
    pnl_result = pnl_calculator.compute_realised_pnl(
        exit_tokens=abs(event.tokens),
        exit_price_usd=price_result.price_usd,
        exit_price_confidence=price_result.confidence_score,
        cost_basis_lots=lots,
    )

    # Store record
    await store_pnl_record(pnl_result)

    # Update wallet behavior metrics
    await update_profit_extraction_metrics(event.wallet, pnl_result)
```

### Cross-Token Memory Integration

PnL records update wallet behavior history:

```python
# In CrossTokenMemoryService
async def update_wallet_pnl_metrics(wallet: str, pnl_record: RealisedPnLRecord):
    history = await get_wallet_history(wallet)

    # Update aggregates
    history.total_realised_pnl += pnl_record.realised_pnl_usd
    history.realised_trades_count += 1

    if pnl_record.realised_pnl_usd > 0:
        history.profitable_exits += 1
    history.realised_profit_rate = history.profitable_exits / history.realised_trades_count

    if pnl_record.exit_efficiency is not None:
        history.average_exit_efficiency = (
            (history.average_exit_efficiency * (history.realised_trades_count - 1)
             + pnl_record.exit_efficiency) / history.realised_trades_count
        )

    await update_wallet_history(history)
```

## Behavioural Analysis

The PnL system enables answering key questions:

### Which wallets extract profit?

```sql
SELECT wallet_address, SUM(realised_pnl_usd) as total_pnl
FROM realised_pnl_records
WHERE realised_pnl_usd > 0
GROUP BY wallet_address
ORDER BY total_pnl DESC;
```

### Which wallets exit early?

```sql
SELECT wallet_address, AVG(exit_efficiency) as avg_efficiency
FROM realised_pnl_records
WHERE exit_efficiency < 0.5
GROUP BY wallet_address
HAVING COUNT(*) > 3;
```

### Which wallets time exits well?

```sql
SELECT wallet_address,
       AVG(exit_efficiency) as avg_efficiency,
       AVG(realised_pnl_pct) as avg_pnl_pct
FROM realised_pnl_records
WHERE overall_confidence > 0.7
GROUP BY wallet_address
ORDER BY avg_efficiency DESC;
```
