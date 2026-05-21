# Sprint 8 Completion Report: Realised vs Unrealised Behaviour Intelligence

**Version**: 1.0.0
**Date**: 2026-05-21
**Sprint**: 8 - Realised vs Unrealised Behaviour Intelligence

## Executive Summary

Sprint 8 successfully extended SHI from price-aware analysis to behaviour-aware profit extraction analysis. The implementation enables tracking of realised vs unrealised PnL, cost basis with configurable accounting methods, and profit extraction behaviour features.

## Sprint Directive Compliance

### Hard Rules Status

| Hard Rule | Status | Implementation |
|-----------|--------|----------------|
| Do not treat price as ground truth | COMPLIANT | All prices include confidence scoring |
| Always propagate price confidence | COMPLIANT | Confidence flows through all calculations |
| Do not show precise PnL when confidence low | COMPLIANT | `display_precise` flag with range fallback |
| Separate realised and unrealised PnL | COMPLIANT | Distinct models and computation paths |
| Accounting method must be explicit | COMPLIANT | `accounting_method` field on all estimates |
| Frozen PDR metrics unchanged | COMPLIANT | PnL features are additive only |
| Behaviour scores remain probabilistic | COMPLIANT | No deterministic claims |

### Core Questions Answered

| Question | Feature | Status |
|----------|---------|--------|
| Which wallets sit on large unrealised gains? | `unrealised_pnl_pct` | IMPLEMENTED |
| Which wallets actually realise gains? | `realised_profit_rate` | IMPLEMENTED |
| Which wallets repeatedly exit early? | `early_profit_exit_rate` | IMPLEMENTED |
| Which wallets hold through volatility? | `hold_through_drawdown_score` | IMPLEMENTED |
| Which wallets extract profit before liquidity deteriorates? | `liquidity_sensitive_exit_score` | IMPLEMENTED |

## Deliverables

### Code Deliverables

| File | Action | Status |
|------|--------|--------|
| `src/longitudinal/models.py` | ENHANCED | Added PriceSnapshot, RealisedPnLRecord, CostBasisLot, AccountingMethod, PriceConfidenceLevel |
| `src/longitudinal/price_snapshots.py` | CREATED | PriceSnapshotService, PriceSnapshotCollector, PriceObservation |
| `src/longitudinal/pnl_calculator.py` | CREATED | CostBasisCalculator, RealisedPnLCalculator, ProfitExtractionAnalyzer |
| `src/longitudinal/__init__.py` | ENHANCED | Updated exports for Sprint 8 modules |
| `src/risk/scoring.py` | ENHANCED | Added PnLCandidateFeatures, build_pnl_candidate_features |
| `src/risk/__init__.py` | ENHANCED | Updated exports for Sprint 8 features |
| `src/core/config.py` | ENHANCED | Added Sprint 8 feature flags |
| `tests/test_sprint8_pnl.py` | CREATED | 32 validation tests |

### Documentation Deliverables

| Document | Status |
|----------|--------|
| `docs/pnl/HISTORICAL_PRICE_SNAPSHOT_SPEC.md` | CREATED |
| `docs/pnl/COST_BASIS_ESTIMATION.md` | CREATED |
| `docs/pnl/REALISED_PNL_IMPLEMENTATION.md` | CREATED |
| `docs/pnl/PROFIT_EXTRACTION_FEATURES.md` | CREATED |
| `docs/pnl/SPRINT_8_VALIDATION_REPORT.md` | CREATED |
| `docs/pnl/SPRINT_8_COMPLETION_REPORT.md` | CREATED |

## Key Implementations

### 1. Historical Price Snapshots

Stores versioned price observations for cost basis calculation:

```python
class PriceSnapshot(Base):
    token_mint: str
    timestamp: datetime
    price_usd: float
    confidence_score: float  # Never zero, min 0.1
    source: str
    data_version: int
    cadence_seconds: int
    sequence_in_token: int
```

### 2. Cost Basis with Accounting Methods

Supports FIFO, LIFO, and Weighted Average:

```python
class CostBasisCalculator:
    def compute_cost_basis(
        trades: list[TradeRecord],
        current_price_usd: float | None,
        method: AccountingMethod,
    ) -> CostBasisEstimate
```

### 3. Realised PnL on Exits

Computes PnL with exit efficiency and liquidity adjustment:

```python
class RealisedPnLCalculator:
    def compute_realised_pnl(
        exit_tokens: int,
        exit_price_usd: float,
        cost_basis_lots: list[dict],
        peak_price_usd: float | None,
        liquidity_at_exit_usd: float | None,
    ) -> RealisedPnLEstimate
```

### 4. Profit Extraction Behaviour Features

Analyzes wallet behavior patterns:

```python
class ProfitExtractionAnalyzer:
    def compute_wallet_metrics(
        pnl_records: list[RealisedPnLRecord],
    ) -> dict  # realised_profit_rate, exit_efficiency, etc.
```

### 5. Risk Model Integration (Candidate Features)

Feature-flagged PnL features for risk scoring:

```python
class PnLCandidateFeatures:
    unrealised_pnl_pct: float | None
    realised_profit_rate: float | None
    exit_efficiency: float | None
    liquidity_sensitive_exit_score: float | None
    display_precise: bool  # False when low confidence
```

## Architecture Decisions

### 1. Price Snapshots as Immutable Records

**Decision**: Store price snapshots as append-only immutable records.

**Rationale**: Supports deterministic replay and audit trail per longitudinal infrastructure principles.

### 2. Configurable Accounting Methods

**Decision**: Support FIFO, LIFO, and Weighted Average with explicit selection.

**Rationale**: Different use cases require different methods; explicit selection avoids ambiguity.

### 3. Lot-Level Tracking

**Decision**: Maintain individual lots for FIFO/LIFO, convert to single average for Weighted Average.

**Rationale**: Enables precise partial exit handling while supporting simpler weighted average model.

### 4. Confidence Floor at 0.1

**Decision**: Enforce minimum confidence of 0.1 (never zero).

**Rationale**: Per hard rules, missing data reduces confidence but should not eliminate it entirely.

### 5. Candidate Features with Feature Flags

**Decision**: Add Sprint 8 features as CANDIDATE features behind feature flags.

**Rationale**: Enables controlled rollout and validation before production use.

## Configuration

New configuration options in `src/core/config.py`:

```python
# Sprint 8: Realised vs Unrealised Behaviour Intelligence
use_pnl_features: bool = False  # CANDIDATE - requires validation
use_profit_extraction_features: bool = False  # CANDIDATE - requires validation
pnl_default_accounting_method: str = "fifo"
pnl_min_confidence_for_display: float = 0.5
pnl_confidence_floor: float = 0.1
```

## Test Coverage

### Test Summary

| Category | Tests | Status |
|----------|-------|--------|
| Missing Price Handling | 2 | PASS |
| Stale Price Handling | 3 | PASS |
| Low Liquidity Handling | 2 | PASS |
| Partial Sells | 2 | PASS |
| Multiple Buys | 3 | PASS |
| Transfer Ambiguity | 2 | PASS |
| FIFO Accounting | 2 | PASS |
| LIFO Accounting | 2 | PASS |
| Weighted Average | 2 | PASS |
| Confidence Propagation | 4 | PASS |
| Risk Model Integration | 3 | PASS |
| Hard Rules Compliance | 5 | PASS |
| **Total** | **32** | **PASS** |

## Integration Points

### Snapshot Engine Integration

Price snapshots integrate with existing snapshot engine cadence system:

```python
price_collector = PriceSnapshotCollector(
    price_provider=jupiter_provider,
    snapshot_service=price_service,
)

snapshot = await price_collector.collect(token_mint, cadence_seconds=300)
```

### Event Store Integration

PnL calculation triggered by sell events:

```python
async def on_sell_event(event: TradeEvent):
    pnl_result = pnl_calculator.compute_realised_pnl(
        exit_tokens=abs(event.tokens),
        exit_price_usd=price_result.price_usd,
        cost_basis_lots=lots,
    )
    await store_pnl_record(pnl_result)
```

### Cross-Token Memory Integration

Wallet behavior history updated with PnL metrics:

```python
# In WalletBehaviorHistory
realised_profit_rate: float = 0.0
early_profit_exit_rate: float = 0.0
average_exit_efficiency: float = 0.0
hold_through_drawdown_score: float = 0.0
```

### Risk Scoring Integration

Candidate features available in risk reports:

```python
if settings.use_pnl_features:
    risk_report = generate_risk_report(
        ...,
        pnl_candidate_features=pnl_features,
    )
```

## Known Limitations

1. **Transfer Ambiguity**: Cannot definitively distinguish transfers from sales
2. **Real-Time Processing**: Batch processing only, no streaming support
3. **Multi-Token Portfolios**: Not yet optimized for complex portfolio scenarios
4. **Historical Data**: Requires price snapshots for accurate historical cost basis

## Future Enhancements

### Immediate (Next Sprint)

1. **Transfer Detection**: Heuristics to identify transfer vs sale
2. **Portfolio View**: Aggregate unrealised across all positions
3. **Exit Timing Alerts**: Notify on liquidity deterioration

### Medium-Term

1. **Tax Lot Optimization**: Suggest optimal lots for tax efficiency
2. **Behavioural Clustering**: Group wallets by profit extraction patterns
3. **Causal Analysis**: P(exit | liquidity drop)

### Long-Term

1. **ML Exit Prediction**: Predict likely exit timing
2. **Cross-Token Correlation**: Analyze behavior consistency across launches
3. **Market Impact Modeling**: Estimate actual slippage from historical data

## Conclusion

Sprint 8 successfully delivered:

1. **Historical Price Snapshot Infrastructure** - Immutable, versioned price storage
2. **Cost Basis with Accounting Methods** - FIFO, LIFO, Weighted Average support
3. **Realised PnL Calculation** - With exit efficiency and liquidity adjustment
4. **Profit Extraction Behaviour Features** - Answering key wallet behavior questions
5. **Risk Model Candidate Features** - Feature-flagged for controlled rollout
6. **Comprehensive Validation** - 32 tests covering all hard rules and edge cases
7. **Complete Documentation** - 6 specification documents

The implementation is ready for controlled rollout with feature flags and extends SHI's capability to distinguish between unrealised exposure and realised behaviour.

---

## Appendix: File Changes Summary

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/longitudinal/price_snapshots.py` | ~350 | Historical price snapshot service |
| `src/longitudinal/pnl_calculator.py` | ~450 | Cost basis and PnL calculators |
| `tests/test_sprint8_pnl.py` | ~600 | Validation test suite |
| `docs/pnl/HISTORICAL_PRICE_SNAPSHOT_SPEC.md` | ~300 | Price snapshot specification |
| `docs/pnl/COST_BASIS_ESTIMATION.md` | ~350 | Cost basis documentation |
| `docs/pnl/REALISED_PNL_IMPLEMENTATION.md` | ~400 | PnL implementation docs |
| `docs/pnl/PROFIT_EXTRACTION_FEATURES.md` | ~350 | Feature documentation |
| `docs/pnl/SPRINT_8_VALIDATION_REPORT.md` | ~250 | Validation report |
| `docs/pnl/SPRINT_8_COMPLETION_REPORT.md` | ~400 | This report |

### Modified Files

| File | Changes |
|------|---------|
| `src/longitudinal/models.py` | Added PriceSnapshot, RealisedPnLRecord, CostBasisLot, enums |
| `src/longitudinal/__init__.py` | Updated exports, added Sprint 8 hard rules to docstring |
| `src/risk/scoring.py` | Added PnLCandidateFeatures, build_pnl_candidate_features, updated RiskReport |
| `src/risk/__init__.py` | Updated exports for Sprint 8 features |
| `src/core/config.py` | Added Sprint 8 feature flags |
