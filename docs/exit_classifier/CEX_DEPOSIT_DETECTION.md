# CEX Deposit Detection

**Version**: 1.0.0
**Sprint**: 9 - Transfer/Sell Classification and PnL Validation
**Date**: 2026-05-21

## Overview

CEX Deposit Detection identifies transfers to centralized exchanges (Binance, Coinbase, etc.) which are uncertain exits where the eventual outcome (hold vs sell) cannot be determined.

## Purpose

**HARD RULE**: CEX deposits are uncertain exits unless sale can be inferred.

When tokens are sent to a CEX deposit address, we cannot know if the user:
- Will sell on the CEX
- Will hold on the CEX
- Will transfer to another wallet

Therefore, CEX deposits should NOT generate realised PnL.

## Detection Methods

### 1. Known CEX Addresses

Pre-registered hot wallet addresses for major exchanges:

```python
KNOWN_CEX_ADDRESSES = {
    # Binance
    "2ojv9BAiHUrvsm9gxDe7fJSzbNZSJcxZvf8dqmWGHG8S": "binance",
    "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9": "binance",
    # Coinbase
    "H8sMJSCQxfKiFTCfDR3DUMLPwcRbM61LGFJ8N4dK3WjS": "coinbase",
    # Kraken
    "GJRs4FwHtemZ5ZE9x3FNvJ8TMwitKTh21yxdRPqn7npE": "kraken",
    # OKX
    "5VCwKtCXgCJ6kit5FybXjvriW3xELsFDhYrPSqtJNmcD": "okx",
    # ... more addresses
}
```

### 2. High Fan-In Pattern

Addresses that receive tokens from many unique wallets are likely CEX deposit addresses:

```
         ┌── Wallet A
         ├── Wallet B
Address ←├── Wallet C    (100+ unique senders = high fan-in)
         ├── Wallet D
         └── ... (many more)
```

### 3. Exchange Labels

Address labels from external services (Solscan, etc.):

```python
label = "Binance Hot Wallet 3"
if "binance" in label.lower():
    return CEXDepositResult(is_cex_deposit=True, cex_name="binance")
```

## Confidence Levels

| Detection Method | Confidence |
|-----------------|------------|
| Known address | 0.95 |
| Exchange label | 0.85 |
| Very high fan-in (500+) | 0.75 |
| High fan-in (100+) | 0.60 |

## Configuration

```python
@dataclass
class CEXDetectionConfig:
    high_fan_in_threshold: int = 100
    very_high_fan_in_threshold: int = 500
    known_address_confidence: float = 0.95
    high_fan_in_confidence: float = 0.60
    very_high_fan_in_confidence: float = 0.75
    exchange_label_confidence: float = 0.85
```

## Output

```python
@dataclass(frozen=True)
class CEXDepositResult:
    is_cex_deposit: bool
    cex_name: str | None           # "binance", "coinbase", etc.
    deposit_address: str | None
    detection_method: str          # "known_address", "fan_in_pattern", "exchange_label"
    confidence: float
    evidence_factors: tuple[str, ...]
```

## Usage

```python
from src.longitudinal import (
    CEXDepositDetector,
    create_cex_deposit_detector,
)

detector = create_cex_deposit_detector()

result = detector.detect_cex_deposit(
    destination_address="2ojv9BAiHUrvsm9gxDe7fJSzbNZSJcxZvf8dqmWGHG8S",
    fan_in_count=500,  # Optional
    address_label="Binance Hot Wallet",  # Optional
)

if result.is_cex_deposit:
    print(f"CEX deposit to {result.cex_name}")
    print(f"Detection method: {result.detection_method}")
    print(f"Confidence: {result.confidence:.0%}")
```

## Adding Custom CEX Addresses

```python
# Via configuration
detector = create_cex_deposit_detector(
    additional_cex_addresses={
        "NewExchangeAddress123": "new_exchange",
    }
)

# Or dynamically
detector.add_known_address("AnotherAddress456", "another_exchange")
```

## Examples

### Example 1: Known Binance Address

```
Destination: 2ojv9BAiHUrvsm9gxDe7fJSzbNZSJcxZvf8dqmWGHG8S

Result:
- is_cex_deposit: True
- cex_name: "binance"
- detection_method: "known_address"
- confidence: 0.95
```

### Example 2: High Fan-In Address

```
Destination: UnknownAddress123
Fan-in count: 350 unique senders

Result:
- is_cex_deposit: True
- cex_name: None (unknown CEX)
- detection_method: "fan_in_pattern"
- confidence: 0.75
```

### Example 3: Regular Wallet

```
Destination: RegularWallet456
Fan-in count: 5 unique senders

Result:
- is_cex_deposit: False
- detection_method: "uncertain"
- confidence: 0.0
```

## Integration with Exit Classifier

```python
# In ExitEventClassifier._classify_exit()

# Priority 6: CEX_DEPOSIT
if evidence.destination_is_known_cex:
    return ExitEventType.CEX_DEPOSIT, 0.9, f"Known CEX: {evidence.destination_cex_name}", factors

if evidence.destination_is_high_fan_in and not evidence.dex_detected:
    return ExitEventType.CEX_DEPOSIT, 0.6, "High fan-in address (likely CEX)", factors
```

## Behavioral Implications

When a CEX deposit is detected:

1. **No immediate PnL** - Sale outcome unknown
2. **Track as potential exit** - May become sell later
3. **Monitor for patterns** - Same wallet may deposit and sell repeatedly
4. **Risk assessment** - CEX deposits often precede large sells

## Future Enhancements

1. **On-chain CEX sell detection** - Some CEXes settle on-chain
2. **Timing analysis** - Deposits followed by token price drop
3. **Volume correlation** - Deposit amount vs subsequent sell volume
4. **Cross-token patterns** - Same wallet CEX deposit behavior

## Known Limitations

1. **New CEX addresses** - Exchanges create new deposit addresses
2. **Multi-sig wallets** - Some CEX wallets are multi-sig
3. **Institutional custody** - May look like CEX but isn't
4. **User-to-user transfers** - Could look like high fan-in
