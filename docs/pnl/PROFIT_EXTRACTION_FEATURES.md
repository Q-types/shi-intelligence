# Profit Extraction Behaviour Features

**Version**: 1.0.0
**Sprint**: 8 - Realised vs Unrealised Behaviour Intelligence
**Date**: 2026-05-21

## Overview

This document describes the profit extraction behaviour features that analyze how wallets realize gains across token launches. These features answer key questions about wallet behavior patterns.

## Core Questions

1. **Which wallets extract profit consistently?** - Realised profit rate
2. **Which wallets exit early?** - Early profit exit rate
3. **Which wallets time exits well?** - Exit efficiency
4. **Which wallets hold through volatility?** - Hold through drawdown score
5. **Which wallets exit before liquidity deteriorates?** - Liquidity sensitive exit score

## Feature Definitions

### realised_profit_rate

**Definition**: Proportion of exits that are profitable (realised_pnl_usd > 0).

```python
realised_profit_rate = profitable_exits / total_exits
```

| Value | Interpretation |
|-------|----------------|
| 0.8+ | Consistently profitable exits |
| 0.5 - 0.8 | Mixed performance |
| < 0.5 | More losses than gains |

### early_profit_exit_rate

**Definition**: Proportion of profitable exits that occur early (before peak).

```python
early_exits = exits where realised_pnl > 0 AND exit_efficiency < 0.5
early_profit_exit_rate = early_exits / profitable_exits
```

| Value | Interpretation |
|-------|----------------|
| 0.8+ | Takes profits early, may miss upside |
| 0.3 - 0.8 | Balanced approach |
| < 0.3 | Tends to wait for higher prices |

### average_exit_efficiency

**Definition**: Average of exit_price / peak_price across all exits.

```python
average_exit_efficiency = mean(exit_efficiency for all exits)
```

| Value | Interpretation |
|-------|----------------|
| 0.9+ | Excellent timing, exits near peaks |
| 0.7 - 0.9 | Good timing |
| 0.5 - 0.7 | Moderate timing |
| < 0.5 | Poor timing, exits during drawdowns |

### hold_through_drawdown_score

**Definition**: Proportion of exits that occur during significant drawdowns (>20% from peak).

```python
large_drawdown_exits = exits where peak_to_exit_drawdown > 0.2
hold_through_drawdown_score = large_drawdown_exits / total_exits
```

| Value | Interpretation |
|-------|----------------|
| 0.8+ | Diamond hands, holds through volatility |
| 0.3 - 0.8 | Balanced approach |
| < 0.3 | Quick to exit on drawdowns |

### profit_taking_consistency

**Definition**: Inverse of standard deviation in realised PnL percentages.

```python
profit_taking_consistency = 1.0 / (1.0 + stdev(realised_pnl_pct))
```

| Value | Interpretation |
|-------|----------------|
| 0.8+ | Very consistent profit targets |
| 0.4 - 0.8 | Moderately consistent |
| < 0.4 | Highly variable exit points |

### liquidity_sensitive_exit_score

**Definition**: Inverse of average position-to-liquidity ratio at exit.

```python
avg_position_ratio = mean(exit_value / liquidity_at_exit)
liquidity_sensitive_exit_score = 1.0 / (1.0 + avg_position_ratio * 10)
```

| Value | Interpretation |
|-------|----------------|
| 0.8+ | Exits with minimal market impact |
| 0.4 - 0.8 | Moderate awareness |
| < 0.4 | Large positions relative to liquidity |

## Data Model

### WalletBehaviorHistory Fields

```python
class WalletBehaviorHistory:
    # Sprint 8: Profit Extraction Behaviour Features
    realised_profit_rate: float = 0.0
    early_profit_exit_rate: float = 0.0
    average_exit_efficiency: float = 0.0
    hold_through_drawdown_score: float = 0.0
    profit_taking_consistency: float = 0.0
    liquidity_sensitive_exit_score: float = 0.0
    exit_efficiency: float = 0.0
    liquidity_adjusted_exit_quality: float = 0.0
```

## Computation

### ProfitExtractionAnalyzer

```python
class ProfitExtractionAnalyzer:
    def compute_wallet_metrics(
        self,
        pnl_records: list[RealisedPnLRecord],
        position_history: list[dict],
    ) -> dict:
        # Filter to confident records
        valid_records = [r for r in pnl_records if r.overall_confidence >= 0.3]

        if not valid_records:
            return self._empty_metrics()

        # Realised profit rate
        profitable_exits = sum(1 for r in valid_records if r.realised_pnl_usd > 0)
        realised_profit_rate = profitable_exits / len(valid_records)

        # Early profit exit rate
        early_exits = sum(
            1 for r in valid_records
            if r.realised_pnl_usd > 0 and r.exit_efficiency < 0.5
        )
        early_profit_exit_rate = early_exits / max(1, profitable_exits)

        # Average exit efficiency
        efficiencies = [r.exit_efficiency for r in valid_records if r.exit_efficiency]
        average_exit_efficiency = mean(efficiencies) if efficiencies else 0.0

        # Hold through drawdown
        drawdowns = [r.peak_to_exit_drawdown_pct for r in valid_records if r.peak_to_exit_drawdown_pct]
        large_drawdown_holds = sum(1 for d in drawdowns if d > 0.2)
        hold_through_drawdown_score = large_drawdown_holds / len(drawdowns) if drawdowns else 0.0

        # Profit taking consistency
        pnl_pcts = [r.realised_pnl_pct for r in valid_records]
        profit_taking_consistency = 1.0 / (1.0 + stdev(pnl_pcts)) if len(pnl_pcts) > 1 else 0.0

        # Liquidity sensitive exit
        liquidity_ratios = [r.position_vs_liquidity for r in valid_records if r.position_vs_liquidity]
        if liquidity_ratios:
            avg_ratio = mean(liquidity_ratios)
            liquidity_sensitive_exit_score = 1.0 / (1.0 + avg_ratio * 10)
        else:
            liquidity_sensitive_exit_score = 0.5

        return {
            "realised_profit_rate": realised_profit_rate,
            "early_profit_exit_rate": early_profit_exit_rate,
            "average_exit_efficiency": average_exit_efficiency,
            "hold_through_drawdown_score": hold_through_drawdown_score,
            "profit_taking_consistency": profit_taking_consistency,
            "liquidity_sensitive_exit_score": liquidity_sensitive_exit_score,
        }
```

## Usage Examples

### Analyze Wallet Behaviour

```python
from src.longitudinal import ProfitExtractionAnalyzer

analyzer = ProfitExtractionAnalyzer()

# Get wallet's PnL records
pnl_records = await get_wallet_pnl_records(wallet_address)

# Compute metrics
metrics = analyzer.compute_wallet_metrics(
    pnl_records=pnl_records,
    position_history=position_snapshots,
)

print(f"Realised profit rate: {metrics['realised_profit_rate']:.1%}")
print(f"Average exit efficiency: {metrics['average_exit_efficiency']:.1%}")
print(f"Hold through drawdown: {metrics['hold_through_drawdown_score']:.1%}")
```

### Classify Wallet Archetype

```python
def classify_exit_behavior(metrics: dict) -> str:
    """Classify wallet based on profit extraction patterns."""

    if metrics["realised_profit_rate"] > 0.8 and metrics["early_profit_exit_rate"] > 0.6:
        return "early_profit_taker"  # Consistently takes profits early

    if metrics["hold_through_drawdown_score"] > 0.7:
        return "diamond_hands"  # Holds through volatility

    if metrics["average_exit_efficiency"] > 0.85:
        return "peak_timer"  # Exits near peaks

    if metrics["liquidity_sensitive_exit_score"] < 0.3:
        return "whale_dumper"  # Large exits relative to liquidity

    return "mixed_behavior"
```

## Integration with Risk Model

### Candidate Features (Sprint 8)

These features are CANDIDATE features, not production defaults:

```python
class PnLCandidateFeatures:
    unrealised_pnl_pct: float | None
    realised_profit_rate: float | None
    exit_efficiency: float | None
    liquidity_sensitive_exit_score: float | None
    accounting_method: str
    overall_confidence: float
    display_precise: bool
```

### Feature Flag Control

```python
# In config.py
use_pnl_features: bool = False  # CANDIDATE - requires validation
use_profit_extraction_features: bool = False  # CANDIDATE - requires validation
```

### Risk Report Integration

```python
if settings.use_pnl_features:
    pnl_features = build_pnl_candidate_features(
        unrealised_pnl_pct=cost_basis.unrealised_pnl_pct,
        realised_profit_rate=wallet_metrics["realised_profit_rate"],
        exit_efficiency=wallet_metrics["average_exit_efficiency"],
        liquidity_sensitive_exit_score=wallet_metrics["liquidity_sensitive_exit_score"],
        accounting_method=settings.pnl_default_accounting_method,
    )
    risk_report = generate_risk_report(
        ...,
        pnl_candidate_features=pnl_features,
    )
```

## Cross-Token Analysis

### Behavior Consistency Across Launches

```python
def analyze_cross_token_consistency(wallet: str) -> dict:
    """Analyze if wallet behaves consistently across token launches."""

    # Get metrics per token
    token_metrics = {}
    for token in wallet_tokens:
        pnl_records = get_pnl_records(wallet, token)
        token_metrics[token] = analyzer.compute_wallet_metrics(pnl_records, [])

    # Compute consistency
    exit_efficiencies = [m["average_exit_efficiency"] for m in token_metrics.values()]
    profit_rates = [m["realised_profit_rate"] for m in token_metrics.values()]

    return {
        "exit_efficiency_consistency": 1.0 / (1.0 + stdev(exit_efficiencies)),
        "profit_rate_consistency": 1.0 / (1.0 + stdev(profit_rates)),
        "tokens_analyzed": len(token_metrics),
    }
```

### Identifying Repeat Behavior

```python
def identify_repeat_early_exiters(min_tokens: int = 3) -> list[str]:
    """Find wallets that consistently exit early across multiple tokens."""

    repeat_exiters = []

    for wallet in all_wallets:
        tokens_exited_early = 0

        for token in wallet_tokens(wallet):
            pnl_records = get_pnl_records(wallet, token)
            metrics = analyzer.compute_wallet_metrics(pnl_records, [])

            if metrics["early_profit_exit_rate"] > 0.7:
                tokens_exited_early += 1

        if tokens_exited_early >= min_tokens:
            repeat_exiters.append(wallet)

    return repeat_exiters
```

## Validation Considerations

### Minimum Data Requirements

| Metric | Minimum Records | Rationale |
|--------|-----------------|-----------|
| realised_profit_rate | 3 | Need multiple exits for meaningful rate |
| exit_efficiency | 5 | Need sample for accurate average |
| profit_taking_consistency | 5 | Need variance in sample |
| hold_through_drawdown | 3 | Need drawdown events |

### Confidence Filtering

All metrics filter to records with `overall_confidence >= 0.3` to avoid noise from low-confidence PnL estimates.

### Known Limitations

1. **New wallets**: Insufficient data for reliable metrics
2. **Single-exit wallets**: Cannot compute consistency metrics
3. **Low liquidity tokens**: Liquidity metrics may be unreliable
4. **Transfer ambiguity**: Some exits may be transfers, not sales
