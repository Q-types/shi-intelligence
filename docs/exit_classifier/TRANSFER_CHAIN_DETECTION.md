# Transfer Chain Detection

**Version**: 1.0.0
**Sprint**: 9 - Transfer/Sell Classification and PnL Validation
**Date**: 2026-05-21

## Overview

Transfer Chain Detection identifies wallet migrations and internal transfer chains where tokens move between wallets owned by the same entity. These transfers should NOT generate realised PnL.

## Purpose

**HARD RULE**: Transfers must NOT generate realised PnL unless later sale observed.

When a user moves tokens between their own wallets, this is not a taxable/recordable sell event. The Transfer Chain Detector identifies these migration patterns.

## Detection Features

### 1. Shared Funder

Wallets funded by the same source are likely owned by the same entity.

```
Funder (Wallet A)
    ├── Wallet B (funded with 0.1 SOL)
    └── Wallet C (funded with 0.1 SOL)

Transfer: B → C is likely migration
```

### 2. Rapid Movement

Token appears at destination and moves quickly (within 5 minutes).

```
Wallet A → Wallet B → Wallet C
     t=0       t=30s      t=60s

This is likely a migration chain, not independent transfers.
```

### 3. Same Token Held

Destination wallet continues to hold the token after transfer.

```
Wallet A: Sells 1000 tokens to DEX → PnL computable
Wallet A: Transfers 1000 tokens to B → B holds them → Migration
```

### 4. Same Behavior Pattern

Wallets exhibit similar trading patterns:
- Trading frequency
- Hold times
- Position sizes

### 5. No Quote Received

Source wallet receives no quote asset (SOL/USDC) for the transfer.

```
Transfer: A → B, no SOL movement → Migration signal
Swap: A → Pool, SOL received → True sell
```

## Confidence Scoring

```python
@dataclass
class TransferChainConfig:
    rapid_followup_seconds: int = 300  # 5 minutes
    chain_detection_window_hours: int = 24
    min_migration_confidence: float = 0.6

    # Weights
    shared_funder_weight: float = 0.30
    same_token_held_weight: float = 0.20
    rapid_movement_weight: float = 0.25
    no_quote_received_weight: float = 0.15
    same_behavior_weight: float = 0.10
```

## Output

```python
@dataclass(frozen=True)
class TransferChainResult:
    likely_migration: bool
    related_wallet_candidate: str | None
    migration_confidence: float  # 0.0-1.0
    chain_length: int
    evidence_factors: tuple[str, ...]
    chain_wallets: tuple[str, ...]
```

## Usage

```python
from src.longitudinal import (
    TransferChainDetector,
    create_transfer_chain_detector,
)

detector = create_transfer_chain_detector()

result = await detector.detect_migration(
    source_wallet="WalletA",
    destination_wallet="WalletB",
    token_mint="TokenMint123",
    transfer_timestamp=datetime.now(timezone.utc),
    wallet_info_provider=provider,  # Optional
)

if result.likely_migration:
    print(f"Migration detected with {result.migration_confidence:.0%} confidence")
    print(f"Chain: {' → '.join(result.chain_wallets)}")
    # Do NOT compute PnL
else:
    # May be genuine transfer to third party
    # Still uncertain - don't compute PnL
```

## Wallet Info Provider

The detector uses a `WalletInfoProvider` interface for lookups:

```python
class WalletInfoProvider:
    async def get_initial_funder(self, wallet: str) -> str | None
    async def get_token_balance(self, wallet: str, token: str, time: datetime) -> int | None
    async def get_next_token_movement(self, wallet: str, token: str, after: datetime) -> dict | None
    async def get_behavior_profile(self, wallet: str) -> dict | None
    async def get_next_transfer_destination(self, wallet: str, token: str) -> str | None
    async def get_quote_received(self, wallet: str, timestamp: datetime) -> bool
```

## Configuration

```python
from src.longitudinal import TransferChainConfig

config = TransferChainConfig(
    rapid_followup_seconds=300,      # 5 minutes
    chain_detection_window_hours=24, # Look 24h ahead
    min_migration_confidence=0.6,    # Threshold
    max_chain_depth=5,               # Max wallets in chain
)

detector = TransferChainDetector(config=config)
```

## Examples

### Example 1: Clear Migration

```
Source: Wallet A
Destination: Wallet B
Evidence:
- Shared funder: Yes (both funded by Wallet X)
- Rapid followup: Yes (B transfers to C in 30s)
- No quote received: Yes

Confidence: 0.30 + 0.25 + 0.15 = 0.70
Result: likely_migration = True
```

### Example 2: Genuine Transfer

```
Source: Wallet A
Destination: Wallet B (unknown)
Evidence:
- Shared funder: No
- Rapid followup: No
- B holds token: Yes

Confidence: 0.20
Result: likely_migration = False
```

## Chain Tracing

The detector traces full chains:

```python
chain = await detector._trace_chain(
    source="WalletA",
    destination="WalletB",
    token_mint="Token123",
    provider=provider,
)

# Result: ["WalletA", "WalletB", "WalletC", "WalletD"]
# Token moved through 4 wallets before settling
```

## Integration

```python
# In PnL computation pipeline
if classification.exit_type == ExitEventType.TRANSFER_OUT:
    migration_result = await transfer_detector.detect_migration(...)

    if migration_result.likely_migration:
        # Track as internal movement
        record_internal_transfer(
            source=source_wallet,
            destination=destination_wallet,
            related_wallets=migration_result.chain_wallets,
        )
        # NO PnL computation
    else:
        # Uncertain transfer to third party
        record_uncertain_exit(classification)
        # Still NO PnL computation (hard rule)
```
