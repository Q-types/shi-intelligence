# LP Action Separation

**Version**: 1.0.0
**Sprint**: 9 - Transfer/Sell Classification and PnL Validation
**Date**: 2026-05-21

## Overview

LP Action Separation ensures that liquidity provider actions (add liquidity, remove liquidity, stake, unstake) are NOT treated as sells for PnL computation purposes.

## Purpose

**HARD RULE**: LP actions must NOT be treated as sells.

When a user adds tokens to a liquidity pool, their token balance decreases but they have NOT sold the tokens - they've converted them to LP tokens. This is NOT a taxable/recordable sell event.

## LP Actions

| Action | Token Movement | LP Token Movement | Is Sell? |
|--------|---------------|-------------------|----------|
| Add Liquidity | Token OUT | LP Token IN | NO |
| Remove Liquidity | Token IN | LP Token OUT | NO |
| Stake LP | LP Token OUT | Receipt IN | NO |
| Unstake LP | Receipt OUT | LP Token IN | NO |

## Detection Method

### 1. LP Program Detection

Check for known LP/AMM program IDs in the transaction:

```python
LP_PROGRAMS = {
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "raydium_amm",
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK": "raydium_clmm",
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "orca_whirlpool",
    "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "orca_v2",
    "MERLuDFBMmsHnsBPZw2sDQZHvXFMwp8EdjudcU2HKky": "mercurial",
    "SSwpkEEcbUqx4vtoEByFjSkhKdCT862DNVb52nZg1UZ": "saber",
}
```

### 2. LP Token Movement Detection

Check for LP token minting or burning in the same transaction:

```python
# Add liquidity pattern:
# - Token A balance decreases
# - LP token balance increases (minted)

# Remove liquidity pattern:
# - LP token balance decreases (burned)
# - Token A balance increases
```

### 3. No Direct Quote Received

LP adds typically don't have direct quote asset (SOL) received:

```
Swap: Token → SOL (quote received)
LP Add: Token → LP Token (no SOL received)
```

## Confidence Scoring

```python
@dataclass(frozen=True)
class LPActionResult:
    is_lp_action: bool
    action_type: str | None  # "add_liquidity", "remove_liquidity"
    lp_program: str | None
    lp_token_mint: str | None
    lp_token_amount: int | None
    pool_address: str | None
    confidence: float
    evidence_factors: tuple[str, ...]
```

### Confidence Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| LP program detected | +0.30 | Known AMM/LP program in tx |
| LP token minted | +0.40 | Wallet received LP tokens |
| LP token burned | +0.35 | Wallet burned LP tokens |
| No quote received | +0.15 | No SOL/USDC received |
| Tokens to program | +0.10 | Tokens sent to program, not wallet |

## Usage

```python
from src.longitudinal import (
    LPActionDetector,
    create_lp_action_detector,
)

detector = create_lp_action_detector()

result = detector.detect_lp_action(
    evidence=exit_evidence,
    additional_token_movements=other_movements,
)

if result.is_lp_action:
    print(f"LP action: {result.action_type}")
    print(f"LP program: {result.lp_program}")
    # Do NOT compute PnL
```

## Examples

### Example 1: Add Liquidity

```
Transaction:
- Token balance: 1000 → 0 (-1000)
- LP token balance: 0 → 50 (+50)
- Program: Raydium AMM

Detection:
- lp_program: "raydium_amm" → +0.30
- lp_token_minted: True → +0.40
- no_quote_received: True → +0.15
Total confidence: 0.85

Result:
- is_lp_action: True
- action_type: "add_liquidity"
```

### Example 2: Regular Swap (NOT LP)

```
Transaction:
- Token balance: 1000 → 0 (-1000)
- SOL balance: +0.5 SOL
- Program: Jupiter v6

Detection:
- lp_program: None → 0
- lp_token_minted: False → 0
Total confidence: 0.0

Result:
- is_lp_action: False
- action_type: None
```

### Example 3: Remove Liquidity

```
Transaction:
- LP token balance: 50 → 0 (-50)
- Token A balance: 0 → 500 (+500)
- Token B balance: 0 → 0.25 SOL
- Program: Orca Whirlpool

Detection:
- lp_program: "orca_whirlpool" → +0.30
- lp_token_burned: True → +0.35
Total confidence: 0.65

Result:
- is_lp_action: True
- action_type: "remove_liquidity"
```

## Integration with Exit Classifier

The exit classifier uses LP detection in its priority pipeline:

```python
# In ExitEventClassifier._classify_exit()

# Priority 2: LP_ADD (token out + LP token minted)
if evidence.lp_token_minted and evidence.lp_program_detected:
    return ExitEventType.LP_ADD, 0.9, "LP tokens minted", factors

# Priority 3: LP_REMOVE (LP token burned)
if evidence.lp_token_burned and evidence.lp_program_detected:
    return ExitEventType.LP_REMOVE, 0.85, "LP tokens burned", factors
```

## Edge Cases

### 1. LP Token Received but No LP Program

Might be a token airdrop or OTC transfer of LP tokens. Confidence is lower.

### 2. LP Program but No LP Token Movement

Could be a swap routed through the AMM. Check for quote asset received.

### 3. Partial LP Add

User adds only part of their tokens to LP. The portion added is LP action, not sell.

## Configuration

```python
from src.core.config import settings

# Enable LP action separation
settings.use_lp_action_separation = True

# Minimum confidence for LP classification
settings.exit_min_lp_confidence = 0.6
```

## Rationale

LP actions are economically different from sells:
1. **Impermanent loss risk** - User still has exposure to token price
2. **LP rewards** - User earns trading fees
3. **Reversible** - User can remove liquidity later
4. **Tax treatment** - Many jurisdictions don't treat LP adds as taxable events

Treating LP adds as sells would create false PnL records and mislead risk analysis.
