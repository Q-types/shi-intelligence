# Sprint 8 Validation Report

**Version**: 1.0.0
**Sprint**: 8 - Realised vs Unrealised Behaviour Intelligence
**Date**: 2026-05-21

## Executive Summary

Sprint 8 validation testing covers all core PnL functionality with emphasis on hard rules compliance, accounting method correctness, and confidence propagation. All critical tests pass.

## Test Coverage

### Test Categories

| Category | Tests | Status |
|----------|-------|--------|
| Missing Price Handling | 2 | PASS |
| Stale Price Handling | 3 | PASS |
| Low Liquidity Handling | 2 | PASS |
| Partial Sells | 2 | PASS |
| Multiple Buys Before Sell | 3 | PASS |
| Transfer Ambiguity | 2 | PASS |
| FIFO Accounting | 2 | PASS |
| LIFO Accounting | 2 | PASS |
| Weighted Average Accounting | 2 | PASS |
| Confidence Propagation | 4 | PASS |
| Risk Model Integration | 3 | PASS |
| Hard Rules Compliance | 5 | PASS |
| **Total** | **32** | **PASS** |

## Hard Rules Validation

### Rule 1: Price Not Ground Truth

**Test**: `test_hard_rule_1_price_not_ground_truth`

**Verification**:
- CostBasisEstimate always includes `confidence` field
- Value is computed, never assumed to be precise

**Status**: PASS

### Rule 2: Confidence Always Propagates

**Test**: `test_hard_rule_2_confidence_propagates`

**Verification**:
- RealisedPnLEstimate includes `overall_confidence`
- Confidence flows from entry through exit

**Status**: PASS

### Rule 3: Low Confidence Shows Ranges

**Test**: `test_hard_rule_3_low_confidence_shows_ranges`

**Verification**:
- When `overall_confidence < min_threshold`, `display_precise = False`
- `to_dict()` returns `*_range` fields instead of precise values

**Status**: PASS

### Rule 4: Separate Realised from Unrealised

**Test**: `test_hard_rule_4_separate_realised_unrealised`

**Verification**:
- CostBasisEstimate has `unrealised_pnl_usd` and `unrealised_pnl_pct`
- RealisedPnLEstimate has `realised_pnl_usd` and `realised_pnl_pct`
- Distinct computation paths

**Status**: PASS

### Rule 5: Accounting Method Explicit

**Test**: `test_hard_rule_5_accounting_method_explicit`

**Verification**:
- CostBasisEstimate has `accounting_method` field
- Value is always set (FIFO/LIFO/WEIGHTED_AVERAGE)

**Status**: PASS

## Accounting Method Validation

### FIFO Correctness

**Test**: `test_fifo_uses_oldest_lots_first`

**Scenario**:
```
Buys: 1000 @ $0.001, 500 @ $0.002, 200 @ $0.003
Sell: 1200 tokens

Expected: Consumes all of lot 1 (1000), 200 from lot 2
Entry price: (1000 * 0.001 + 200 * 0.002) / 1200 = $0.001167
```

**Result**: Entry price matches expected value within tolerance.

**Status**: PASS

### LIFO Correctness

**Test**: `test_lifo_uses_newest_lots_first`

**Scenario**:
```
Buys: 1000 @ $0.001, 500 @ $0.002, 200 @ $0.003
Sell: 600 tokens

Expected: Consumes all of lot 3 (200), 400 from lot 2
Entry price: (200 * 0.003 + 400 * 0.002) / 600 = $0.002333
```

**Result**: Entry price matches expected value within tolerance.

**Status**: PASS

### Weighted Average Correctness

**Test**: `test_weighted_average_uniform_cost`

**Scenario**:
```
Buys: 1000 @ $0.001, 500 @ $0.002, 200 @ $0.003
Total cost: $2.60, Total tokens: 1700
Weighted average: $0.001529

Sell: 500 tokens
Entry price should be: $0.001529
```

**Result**: Entry price equals weighted average.

**Status**: PASS

### FIFO vs LIFO Difference

**Test**: `test_lifo_differs_from_fifo`

**Verification**: When prices differ across lots, FIFO and LIFO produce different entry prices for the same exit.

**Status**: PASS

## Confidence Propagation Validation

### Entry Confidence Affects Result

**Test**: `test_confidence_min_of_components`

**Scenario**: Two buys with different confidences (0.9 and 0.3)

**Verification**: Overall confidence <= 0.8 (pulled down by low-confidence trade)

**Status**: PASS

### Exit Confidence Affects Result

**Test**: `test_pnl_confidence_propagates`

**Scenario**: High entry confidence (0.7), low exit confidence (0.5)

**Verification**: Overall confidence = min(0.7, 0.5) = 0.5

**Status**: PASS

### Low Confidence Display

**Test**: `test_low_confidence_hides_precise_values`

**Scenario**: Build features with confidence 0.3, threshold 0.5

**Verification**:
- `display_precise = False`
- `to_dict()` returns `unrealised_pnl_range` not `unrealised_pnl_pct`

**Status**: PASS

## Edge Case Validation

### Missing Price

**Test**: `test_missing_price_reduces_confidence_not_zero`

**Verification**: When `current_price_usd = None`, unrealised PnL is None (not computed without price)

**Status**: PASS

### Stale Price

**Test**: `test_stale_price_reduces_confidence`

**Verification**: Prices outside immediate time window have reduced confidence score

**Status**: PASS

### Low Liquidity

**Test**: `test_low_liquidity_reduces_exit_quality`

**Verification**: Low liquidity exit has lower `liquidity_adjusted_pnl_usd` than high liquidity exit

**Status**: PASS

### Partial Sells

**Test**: `test_partial_sell_fifo`

**Verification**:
- `is_partial_exit = True`
- `remaining_tokens` correctly calculated

**Status**: PASS

## Integration Validation

### Risk Model Integration

**Test**: `test_candidate_features_structure`

**Verification**:
- PnLCandidateFeatures has all required fields
- `accounting_method` is explicit
- `to_dict()` serializes correctly

**Status**: PASS

### Confidence Floor Enforcement

**Test**: `test_candidate_features_confidence_floor`

**Verification**: Even with input confidence 0.0, output confidence >= 0.1

**Status**: PASS

## Performance Notes

### Cost Basis Calculation

- **Lot building**: O(n) where n = trade count
- **FIFO/LIFO lot consumption**: O(m) where m = lot count
- **Weighted average**: O(1) consumption

### PnL Calculation

- **Lot matching**: O(m) for FIFO/LIFO, O(1) for weighted average
- **Metrics computation**: O(1)

## Known Limitations

1. **Transfer ambiguity**: Cannot distinguish transfer from sale without additional context
2. **Multi-token positions**: Not yet tested with complex portfolio scenarios
3. **Real-time updates**: Batch processing only, no streaming support

## Recommendations

1. **Production deployment**: Enable `use_pnl_features` flag only after additional validation with real data
2. **Monitoring**: Track confidence distribution to identify systematic issues
3. **Alerting**: Alert when overall_confidence < 0.3 for significant positions

## Conclusion

Sprint 8 validation confirms:

1. All hard rules are enforced
2. Accounting methods produce correct results
3. Confidence propagates correctly through calculations
4. Edge cases are handled appropriately

The implementation is ready for controlled rollout with feature flags.
