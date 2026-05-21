# Sell Confidence Score

**Version**: 1.0.0
**Sprint**: 9 - Transfer/Sell Classification and PnL Validation
**Date**: 2026-05-21

## Overview

The Sell Confidence Score quantifies how confident we are that a token balance decrease represents a true sale (as opposed to a transfer, LP action, or other exit type).

## Purpose

**HARD RULE**: Realised PnL requires sell confidence >= threshold.

Only exits with sufficient sell confidence should generate realised PnL records. This prevents contaminating profit/loss data with uncertain events.

## Evidence Factors

The sell confidence score is computed from 9 evidence factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Swap instruction | +0.15 | Transaction contains swap instruction |
| DEX program | +0.15 | Known DEX program ID detected |
| Quote received | +0.20-0.25 | SOL/USDC received in same transaction |
| Token decrease | +0.10 | Token balance decreased |
| Transaction route | +0.10 | Clean DEX route (not bridge) |
| CEX address | -0.20 | Destination is known CEX |
| LP token movement | -0.30 | LP tokens minted (LP add, not sell) |
| Migration signals | -0.10 | Destination shares funder |
| High fan-in | -0.10 | Destination is high fan-in address |

## Scoring Formula

```python
score = 0.0

# Positive evidence
if dex_detected:
    score += 0.15  # swap_instruction
    score += 0.15  # dex_program

if quote_received:
    score += 0.20
    if sol_received > 0.1:  # Significant SOL
        score += 0.05

if token_decreased:
    score += 0.10

if dex_detected and not bridge_detected:
    score += 0.10  # transaction_route

# Negative evidence
if destination_is_cex:
    score -= 0.20

if lp_token_minted:
    score -= 0.30

if destination_shares_funder:
    score -= 0.10

if destination_is_high_fan_in:
    score -= 0.10

# Clamp to [0, 1]
score = max(0.0, min(1.0, score))
```

## Score Interpretation

| Score | Interpretation | PnL Action |
|-------|---------------|------------|
| 0.9+ | High confidence sell | Compute precise PnL |
| 0.7-0.9 | Probable sell | Compute PnL with confidence note |
| 0.5-0.7 | Uncertain | Log as uncertain, no PnL |
| 0.3-0.5 | Probably not a sell | Log as transfer/other |
| <0.3 | Definitely not a sell | No PnL computation |

## Score Breakdown

The scorer provides a detailed breakdown:

```python
from src.longitudinal import SellConfidenceScorer

scorer = SellConfidenceScorer(min_confidence_for_pnl=0.7)

score, pnl_computable, breakdown = scorer.compute_score(classification)

# breakdown example:
{
    "swap_instruction": 0.15,
    "dex_program": 0.15,
    "quote_received": 0.25,
    "token_decrease": 0.10,
    "counterparty_type": 0.0,
    "lp_token_movement": 0.0,
    "destination_type": 0.0,
    "cex_address": 0.0,
    "transaction_route": 0.10,
}
# Total: 0.75 - PnL computable
```

## Configuration

```python
scorer = SellConfidenceScorer(
    min_confidence_for_pnl=0.7,  # Default threshold
)
```

## Examples

### Example 1: Clear DEX Sell

```
Evidence:
- Jupiter v6 program detected
- 0.5 SOL received
- No LP tokens
- Destination is pool

Score breakdown:
- swap_instruction: +0.15
- dex_program: +0.15
- quote_received: +0.25
- token_decrease: +0.10
- transaction_route: +0.10
Total: 0.75 ✓ PnL computable
```

### Example 2: Transfer to CEX

```
Evidence:
- No DEX program
- No SOL received
- Destination is Binance

Score breakdown:
- token_decrease: +0.10
- cex_address: -0.20
Total: -0.10 → 0.0 (clamped)
✗ Not a sell, no PnL
```

### Example 3: LP Add

```
Evidence:
- Raydium program detected
- LP tokens minted
- No SOL received

Score breakdown:
- token_decrease: +0.10
- lp_token_movement: -0.30
Total: -0.20 → 0.0 (clamped)
✗ LP action, not a sell
```

## Integration

```python
from src.longitudinal import (
    ExitEventClassifier,
    SellConfidenceScorer,
)

classifier = ExitEventClassifier()
scorer = SellConfidenceScorer()

# Classify the exit
classification = classifier.classify(...)

# Get detailed sell confidence
score, pnl_computable, breakdown = scorer.compute_score(classification)

if pnl_computable:
    # Safe to compute PnL
    pnl = compute_realised_pnl(...)
else:
    # Log uncertain exit
    log_exit_event(
        type=classification.exit_type,
        sell_confidence=score,
        reason="Below threshold",
    )
```
