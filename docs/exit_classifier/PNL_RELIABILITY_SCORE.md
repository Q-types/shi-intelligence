# PnL Reliability Score

**Version**: 1.0.0
**Sprint**: 9 - Transfer/Sell Classification and PnL Validation
**Date**: 2026-05-21

## Overview

The PnL Reliability Score quantifies the overall reliability of a realised PnL estimate, combining multiple confidence factors to determine how the PnL should be displayed.

## Purpose

**HARD RULE**: Low reliability PnL must NOT display precise values.

When PnL reliability is low, displaying precise values (e.g., "$1,234.56") would convey false confidence. Instead, show ranges or mark as unavailable.

## Reliability Factors

The score combines 6 independent factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Sell confidence | 0.25 | How confident this is a true sell |
| Price confidence | 0.20 | min(entry_price_confidence, exit_price_confidence) |
| Liquidity confidence | 0.15 | Confidence in liquidity data for slippage estimate |
| Lot quality | 0.15 | Inverse of lot count (more lots = more uncertainty) |
| Transfer ambiguity | 0.15 | Penalty if transfers involved in cost basis |
| Event completeness | 0.10 | How complete the event data is |

## Formula

```
Reliability = Σ(factor_i × weight_i)

Where:
- sell_confidence: 0.0-1.0
- price_confidence: min(entry, exit) × 1.0
- liquidity_confidence: 0.0-1.0
- lot_quality: max(0.5, 1.0 - (lot_count - 1) × 0.1)
- transfer_ambiguity: 0.3 if ambiguous else 1.0
- event_completeness: 0.0-1.0
```

## Display Modes

| Reliability | Display Mode | Example |
|-------------|--------------|---------|
| ≥ 0.7 | `precise` | $1,234.56 |
| 0.4 - 0.7 | `range` | $1,000 - $1,500 |
| < 0.4 | `unavailable` | Cannot compute reliably |

## Implementation

```python
from src.longitudinal import PnLReliabilityScorer

scorer = PnLReliabilityScorer(
    weight_sell_confidence=0.25,
    weight_price_confidence=0.20,
    weight_liquidity_confidence=0.15,
    weight_lot_quality=0.15,
    weight_transfer_ambiguity=0.15,
    weight_event_completeness=0.10,
)

reliability, display_mode, components = scorer.compute_reliability(
    sell_confidence=0.85,
    entry_price_confidence=0.75,
    exit_price_confidence=0.90,
    liquidity_confidence=0.80,
    lot_count=2,
    has_transfer_ambiguity=False,
    event_completeness=0.95,
)

print(f"Reliability: {reliability:.2f}")
print(f"Display mode: {display_mode}")
```

## Component Scores

The scorer returns detailed component breakdown:

```python
{
    "sell_confidence": 0.85,
    "price_confidence": 0.75,  # min(entry, exit)
    "liquidity_confidence": 0.80,
    "lot_quality": 0.90,       # 1.0 - (2-1) × 0.1
    "transfer_ambiguity": 1.0, # No ambiguity
    "event_completeness": 0.95,
}
```

## Examples

### Example 1: High Reliability

```
Inputs:
- sell_confidence: 0.90
- entry_price_confidence: 0.85
- exit_price_confidence: 0.92
- liquidity_confidence: 0.80
- lot_count: 1
- has_transfer_ambiguity: False
- event_completeness: 0.95

Components:
- sell: 0.90 × 0.25 = 0.225
- price: 0.85 × 0.20 = 0.170
- liquidity: 0.80 × 0.15 = 0.120
- lot_quality: 1.0 × 0.15 = 0.150
- transfer: 1.0 × 0.15 = 0.150
- completeness: 0.95 × 0.10 = 0.095

Total: 0.91
Display: precise ($1,234.56)
```

### Example 2: Medium Reliability

```
Inputs:
- sell_confidence: 0.70
- entry_price_confidence: 0.50  # Historical price uncertain
- exit_price_confidence: 0.80
- liquidity_confidence: 0.60
- lot_count: 4
- has_transfer_ambiguity: False
- event_completeness: 0.80

Components:
- sell: 0.70 × 0.25 = 0.175
- price: 0.50 × 0.20 = 0.100
- liquidity: 0.60 × 0.15 = 0.090
- lot_quality: 0.70 × 0.15 = 0.105  # (1.0 - 3×0.1)
- transfer: 1.0 × 0.15 = 0.150
- completeness: 0.80 × 0.10 = 0.080

Total: 0.70
Display: precise (just above threshold)
```

### Example 3: Low Reliability (Transfer Ambiguity)

```
Inputs:
- sell_confidence: 0.75
- entry_price_confidence: 0.60
- exit_price_confidence: 0.70
- liquidity_confidence: 0.50
- lot_count: 3
- has_transfer_ambiguity: True  # Previous transfers in history
- event_completeness: 0.70

Components:
- sell: 0.75 × 0.25 = 0.188
- price: 0.60 × 0.20 = 0.120
- liquidity: 0.50 × 0.15 = 0.075
- lot_quality: 0.80 × 0.15 = 0.120
- transfer: 0.30 × 0.15 = 0.045  # Penalty!
- completeness: 0.70 × 0.10 = 0.070

Total: 0.62
Display: range ($1,000 - $1,500)
```

## Configuration

```python
from src.core.config import settings

# Display thresholds
settings.pnl_reliability_min_for_precise = 0.7
settings.pnl_reliability_min_for_range = 0.4
```

## Integration with PnL Output

```python
# In PnL report generation
def format_pnl(pnl_estimate, reliability_result):
    reliability, display_mode, _ = reliability_result

    if display_mode == "precise":
        return f"${pnl_estimate.realised_pnl_usd:.2f}"
    elif display_mode == "range":
        # Compute range based on confidence
        uncertainty = (1 - reliability) * abs(pnl_estimate.realised_pnl_usd)
        low = pnl_estimate.realised_pnl_usd - uncertainty
        high = pnl_estimate.realised_pnl_usd + uncertainty
        return f"${low:.0f} - ${high:.0f}"
    else:
        return "PnL unavailable (low confidence)"
```

## Why Transfer Ambiguity Matters

When a wallet has transfers in its history, cost basis becomes uncertain:

```
Scenario:
1. Wallet A buys 1000 tokens @ $0.001
2. Wallet A transfers 500 tokens to Wallet B (migration?)
3. Wallet A sells 500 tokens @ $0.01

Question: What is the cost basis for the sold 500 tokens?

If migration: Full cost basis is A's ($0.001)
If gift/sale: Uncertain

Transfer ambiguity reduces reliability because we can't be sure
of the correct cost basis calculation.
```

## Lot Quality Calculation

```python
def compute_lot_quality(lot_count):
    """
    More lots = more uncertainty.

    1 lot: 1.0 (simple case)
    2 lots: 0.9
    3 lots: 0.8
    ...
    6+ lots: 0.5 (floor)
    """
    return max(0.5, 1.0 - (lot_count - 1) * 0.1)
```

Multiple lots introduce uncertainty because:
1. Each lot has its own price confidence
2. Lot matching (FIFO/LIFO) may differ from user's intent
3. Partial sells create more complex calculations
