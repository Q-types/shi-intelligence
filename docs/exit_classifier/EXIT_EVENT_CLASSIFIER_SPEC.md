# Exit Event Classifier Specification

**Version**: 1.0.0
**Sprint**: 9 - Transfer/Sell Classification and PnL Validation
**Date**: 2026-05-21

## Overview

The Exit Event Classifier distinguishes true sells from other token balance decreases to enable accurate realised PnL computation. A reduction in wallet token balance is NOT always a sell.

## Hard Rules

1. **Balance decrease alone is NOT a sell** - Must have additional evidence
2. **Realised PnL requires sell confidence** - Minimum threshold enforced
3. **LP actions must NOT be treated as sells** - LP add/remove are NOT sells
4. **Transfers must NOT generate realised PnL** - Unless later sale observed
5. **CEX deposits are uncertain exits** - Sale cannot be confirmed
6. **Low reliability PnL must NOT display precise values** - Show ranges
7. **All classifications must include confidence and evidence** - No exceptions

## Exit Event Types

The classifier identifies 10 distinct exit types:

| Exit Type | Definition | PnL Computable |
|-----------|------------|----------------|
| `DEX_SELL` | Swap on DEX with quote asset received | Yes |
| `TRANSFER_OUT` | Simple transfer to another wallet | No |
| `CEX_DEPOSIT` | Transfer to known/suspected CEX address | No |
| `LP_ADD` | Add liquidity to pool (LP tokens minted) | No |
| `LP_REMOVE` | Remove liquidity from pool | No |
| `BURN` | Token sent to burn address | No |
| `BRIDGE` | Cross-chain bridge transfer | No |
| `WALLET_MIGRATION` | Transfer to related/owned wallet | No |
| `PROGRAM_INTERACTION` | Unknown program interaction | No |
| `UNKNOWN_EXIT` | Cannot classify with confidence | No |

## Classification Algorithm

### Priority-Based Classification

```
1. BURN         - Destination is burn address (0x00...00)
2. LP_ADD       - LP token minted in same transaction
3. LP_REMOVE    - LP token burned in same transaction
4. DEX_SELL     - DEX program + quote asset received
5. BRIDGE       - Bridge program detected
6. CEX_DEPOSIT  - Known CEX address or high fan-in
7. WALLET_MIGRATION - Shared funder or rapid followup
8. PROGRAM_INTERACTION - Unknown program, no destination
9. TRANSFER_OUT - Has destination, no DEX
10. UNKNOWN_EXIT - Fallback
```

### Evidence Extraction

From each transaction, we extract:

```python
@dataclass(frozen=True)
class ExitEvidence:
    # Transaction context
    signature: str
    slot: int
    block_time: datetime | None

    # Token movement
    token_mint: str
    token_amount: int
    token_decimals: int

    # Program detection
    program_ids_detected: tuple[str, ...]
    dex_detected: str | None        # "jupiter_v6", "raydium_amm", etc.
    lp_program_detected: str | None
    bridge_detected: str | None

    # Quote asset movement
    sol_change_lamports: int
    has_quote_asset_received: bool
    quote_asset_mint: str | None
    quote_asset_amount: int | None

    # Destination analysis
    destination_address: str | None
    destination_is_known_cex: bool
    destination_cex_name: str | None
    destination_is_burn_address: bool
    destination_is_high_fan_in: bool

    # LP token movement
    lp_token_minted: bool
    lp_token_burned: bool
    lp_token_amount: int | None

    # Related wallet signals
    destination_shares_funder: bool
    destination_has_same_token: bool
    rapid_followup_detected: bool
```

## Known Program IDs

### DEX Programs

| Program ID | Name |
|------------|------|
| `JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4` | Jupiter v6 |
| `JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB` | Jupiter v4 |
| `JUP3c2Uh3WA4Ng34tw6kPd2G4C5BB21Xo36Je1s32Ph` | Jupiter v3 |
| `675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8` | Raydium AMM |
| `CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK` | Raydium CLMM |
| `whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc` | Orca Whirlpool |
| `9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP` | Orca v2 |
| `srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX` | OpenBook |
| `PhoeNiXZ8ByJGLkxNfZRnkUfjvmuYqLR89jjFHGqdXY` | Phoenix |

### LP Programs

| Program ID | Name |
|------------|------|
| `675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8` | Raydium AMM |
| `CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK` | Raydium CLMM |
| `whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc` | Orca Whirlpool |
| `MERLuDFBMmsHnsBPZw2sDQZHvXFMwp8EdjudcU2HKky` | Mercurial |
| `SSwpkEEcbUqx4vtoEByFjSkhKdCT862DNVb52nZg1UZ` | Saber |

### Bridge Programs

| Program ID | Name |
|------------|------|
| `wormDTUJ6AWPNvk59vGQbDvGJmqbDTdgWgAqcLBCgUb` | Wormhole |
| `worm2ZoG2kUd4vFXhvjh93UUH596ayRfgQ2MgjNMTth` | Wormhole v2 |
| `DeBr1pTRLNxMKVaQJR4i5sNRNNcV8K5bRPpgqjxh5gZQ` | Debridge |
| `3u8hJUVTA4jH1wYAyUur7FFZVQ8H635K3tSHHF4ssjQ5` | Allbridge |

## Usage

```python
from src.longitudinal import (
    ExitEventClassifier,
    create_exit_classifier,
)

# Create classifier
classifier = create_exit_classifier()

# Classify an exit event
result = classifier.classify(
    wallet_address="WalletAddress123",
    token_mint="TokenMint123",
    token_amount=-1_000_000_000,  # Negative = exit
    token_decimals=9,
    tx_data=transaction_data,
)

# Check result
print(f"Exit type: {result.exit_type}")
print(f"Confidence: {result.confidence}")
print(f"PnL computable: {result.pnl_computable}")
print(f"Sell confidence: {result.sell_confidence_score}")
```

## Configuration

```python
from src.longitudinal import ExitClassifierConfig

config = ExitClassifierConfig(
    min_sell_confidence_for_pnl=0.7,  # Threshold for PnL computation
    min_transfer_confidence=0.5,
    min_lp_confidence=0.6,
    min_sol_movement_for_swap=10_000_000,  # 0.01 SOL
    high_fan_in_threshold=100,
    use_cex_detection=True,
    use_migration_detection=True,
    use_lp_detection=True,
    use_bridge_detection=True,
)
```

## Output

```python
@dataclass(frozen=True)
class ExitEventClassification:
    exit_type: ExitEventType
    confidence: float  # 0.0-1.0
    evidence: ExitEvidence
    sell_confidence_score: float
    pnl_computable: bool
    downstream_address: str | None
    downstream_wallet_type: str | None  # "cex", "wallet", "pool", "burn"
    classification_reason: str
    confidence_factors: tuple[str, ...]
```

## Integration with PnL

The exit classifier gates PnL computation:

```python
# Only compute PnL for high-confidence sells
if classification.pnl_computable:
    pnl = pnl_calculator.compute_realised_pnl(...)
else:
    # Log the exit but don't compute PnL
    log_uncertain_exit(classification)
```
