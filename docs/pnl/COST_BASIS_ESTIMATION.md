# Cost Basis Estimation

**Version**: 1.0.0
**Sprint**: 8 - Realised vs Unrealised Behaviour Intelligence
**Date**: 2026-05-21

## Overview

This document describes the cost basis estimation system for tracking wallet position costs. The system supports multiple accounting methods (FIFO, LIFO, Weighted Average) and maintains lot-level tracking for precise partial exit handling.

## Hard Rules

1. **Accounting method must be explicit** - never implied or defaulted silently
2. **Confidence always propagates** - entry price confidence affects all derived metrics
3. **Separate realised from unrealised** - distinct tracking and computation
4. **Price is not ground truth** - always includes confidence scoring

## Accounting Methods

### FIFO (First In, First Out)

**Default method.** Oldest lots are consumed first on sells.

```
Buy 1000 @ $0.001 (Lot A)
Buy 500 @ $0.002  (Lot B)
Sell 800

FIFO consumes: 800 from Lot A
Cost basis: 800 × $0.001 = $0.80
Remaining: Lot A (200), Lot B (500)
```

**Tax implications**: In rising markets, FIFO typically shows higher gains (lower cost basis from older, cheaper lots).

### LIFO (Last In, First Out)

Newest lots are consumed first on sells.

```
Buy 1000 @ $0.001 (Lot A)
Buy 500 @ $0.002  (Lot B)
Sell 800

LIFO consumes: 500 from Lot B, 300 from Lot A
Cost basis: (500 × $0.002) + (300 × $0.001) = $1.30
Remaining: Lot A (700)
```

**Tax implications**: In rising markets, LIFO typically shows lower gains (higher cost basis from newer, more expensive lots).

### Weighted Average

All lots contribute proportionally. Single average cost per token.

```
Buy 1000 @ $0.001 (Lot A)
Buy 500 @ $0.002  (Lot B)

Total cost: $1.00 + $1.00 = $2.00
Total tokens: 1500
Average: $2.00 / 1500 = $0.001333

Sell 800
Cost basis: 800 × $0.001333 = $1.0667
Remaining: 700 tokens @ $0.001333
```

**Simplicity**: Easier mental model, no lot tracking needed.

## Data Structures

### TradeRecord

```python
@dataclass(frozen=True)
class TradeRecord:
    timestamp: datetime
    tokens: int           # Positive for buy, negative for sell
    price_usd: float
    price_confidence: float
    price_source: str
    event_id: int | None
    signature: str | None
```

### CostBasisEstimate

```python
@dataclass(frozen=True)
class CostBasisEstimate:
    avg_entry_price_usd: float
    total_cost_basis_usd: float
    current_position_tokens: int
    current_position_value_usd: float | None
    unrealised_pnl_usd: float | None
    unrealised_pnl_pct: float | None
    accounting_method: AccountingMethod
    confidence: float
    confidence_reason: str
    lot_count: int
    data_points: int
```

### CostBasisLot (Database Model)

```python
class CostBasisLot(Base):
    __tablename__ = "cost_basis_lots"

    id: int
    wallet_address: str
    token_mint: str
    entry_timestamp: datetime
    original_tokens: int
    remaining_tokens: int
    entry_price_usd: float
    entry_price_confidence: float
    entry_event_id: int | None
    accounting_method: str
    created_at: datetime
```

## Lot Management

### Building Lots from Trades

```python
def _build_lots(trades: list[TradeRecord], method: AccountingMethod) -> list[dict]:
    lots = []

    for trade in sorted(trades, key=lambda t: t.timestamp):
        if trade.tokens > 0:
            # Buy: create new lot
            lots.append({
                "timestamp": trade.timestamp,
                "original_tokens": trade.tokens,
                "remaining_tokens": trade.tokens,
                "price_usd": trade.price_usd,
                "confidence": trade.price_confidence,
                "source": trade.price_source,
            })
        elif trade.tokens < 0:
            # Sell: consume from lots
            lots = _consume_lots(lots, abs(trade.tokens), method)

    return lots
```

### Consuming Lots

```python
def _consume_lots(lots: list[dict], tokens_to_consume: int, method: AccountingMethod):
    if method == AccountingMethod.FIFO:
        sorted_lots = sorted(lots, key=lambda l: l["timestamp"])
    elif method == AccountingMethod.LIFO:
        sorted_lots = sorted(lots, key=lambda l: l["timestamp"], reverse=True)
    else:
        return _consume_weighted_average(lots, tokens_to_consume)

    remaining = tokens_to_consume
    for lot in sorted_lots:
        if remaining <= 0:
            break
        consumed = min(lot["remaining_tokens"], remaining)
        lot["remaining_tokens"] -= consumed
        remaining -= consumed

    return lots
```

## Confidence Propagation

### Entry Confidence

Each lot carries the confidence from its entry price:

```python
lot = {
    "price_usd": 0.001,
    "confidence": 0.85,  # From historical price lookup
    ...
}
```

### Aggregate Confidence

When computing cost basis across multiple lots:

```python
# Weighted by token count
weighted_confidence = sum(
    lot["remaining_tokens"] * lot["confidence"]
    for lot in remaining_lots
) / total_remaining_tokens
```

### Unrealised PnL Confidence

Limited by both entry and current price confidence:

```python
if current_price_usd is not None:
    # Adjust confidence based on current price confidence
    weighted_confidence = min(weighted_confidence, current_price_confidence)
```

## API Reference

### CostBasisCalculator

```python
class CostBasisCalculator:
    def __init__(
        self,
        default_method: AccountingMethod = AccountingMethod.FIFO,
        min_confidence_for_precise: float = 0.6,
    )

    def compute_cost_basis(
        self,
        trades: list[TradeRecord],
        current_price_usd: float | None = None,
        current_price_confidence: float = 0.5,
        method: AccountingMethod | None = None,
    ) -> CostBasisEstimate
```

## Usage Examples

### Basic Cost Basis Calculation

```python
from src.longitudinal import CostBasisCalculator, AccountingMethod
from src.longitudinal.pnl_calculator import TradeRecord

calculator = CostBasisCalculator(default_method=AccountingMethod.FIFO)

trades = [
    TradeRecord(
        timestamp=datetime(2026, 5, 1, 10, 0),
        tokens=1000,
        price_usd=0.001,
        price_confidence=0.9,
        price_source="jupiter",
    ),
    TradeRecord(
        timestamp=datetime(2026, 5, 1, 12, 0),
        tokens=500,
        price_usd=0.002,
        price_confidence=0.85,
        price_source="jupiter",
    ),
]

result = calculator.compute_cost_basis(
    trades=trades,
    current_price_usd=0.003,
    current_price_confidence=0.8,
)

print(f"Average entry: ${result.avg_entry_price_usd:.6f}")
print(f"Total cost basis: ${result.total_cost_basis_usd:.2f}")
print(f"Position: {result.current_position_tokens} tokens")
print(f"Unrealised PnL: ${result.unrealised_pnl_usd:.2f} ({result.unrealised_pnl_pct:.1%})")
print(f"Confidence: {result.confidence:.2f}")
print(f"Method: {result.accounting_method.value}")
```

### Compare Accounting Methods

```python
for method in [AccountingMethod.FIFO, AccountingMethod.LIFO, AccountingMethod.WEIGHTED_AVERAGE]:
    result = calculator.compute_cost_basis(
        trades=trades,
        current_price_usd=0.003,
        current_price_confidence=0.8,
        method=method,
    )
    print(f"{method.value}: avg=${result.avg_entry_price_usd:.6f}")
```

## Edge Cases

### No Trades

```python
result = calculator.compute_cost_basis(trades=[], current_price_usd=0.001)
# Returns empty estimate with confidence_reason="No trades provided"
```

### Position Fully Exited

```python
trades = [
    TradeRecord(..., tokens=1000, ...),   # Buy
    TradeRecord(..., tokens=-1000, ...),  # Sell all
]
result = calculator.compute_cost_basis(trades, ...)
# Returns zero position with confidence_reason="Position fully exited"
```

### Missing Price Data

When entry price has low confidence:

```python
result = calculator.compute_cost_basis(
    trades=[TradeRecord(..., price_confidence=0.2, ...)],
    current_price_usd=0.003,
    current_price_confidence=0.9,
)
# Result confidence limited by low entry confidence
assert result.confidence <= 0.5  # Pulled down by low entry confidence
```

## Integration with PnL Calculator

Cost basis lots are passed to the Realised PnL Calculator for exit events:

```python
# Get cost basis
cost_result = cost_calculator.compute_cost_basis(trades, current_price_usd=None)

# Build lots for PnL calculator
cost_basis_lots = [
    {
        "timestamp": lot["timestamp"],
        "original_tokens": lot["original_tokens"],
        "remaining_tokens": lot["remaining_tokens"],
        "price_usd": lot["price_usd"],
        "confidence": lot["confidence"],
    }
    for lot in internal_lots
]

# Calculate realised PnL on sell
pnl_result = pnl_calculator.compute_realised_pnl(
    exit_tokens=500,
    exit_price_usd=0.005,
    exit_price_confidence=0.9,
    cost_basis_lots=cost_basis_lots,
    method=AccountingMethod.FIFO,
)
```

## Performance Considerations

### Lot Count

For wallets with many trades:
- **FIFO/LIFO**: O(n) per sell, where n = lot count
- **Weighted Average**: O(1) per sell (single average)

Recommendation: For wallets with >100 lots, consider weighted average.

### Database Storage

Lots are stored per-wallet, per-token. Consider archiving consumed lots (remaining_tokens=0) to historical tables for active traders.
