# Sprint 9 Completion Report: Transfer/Sell Classification and PnL Validation

**Version**: 1.0.0
**Date**: 2026-05-21
**Sprint**: 9 - Transfer/Sell Classification and PnL Validation

## Executive Summary

Sprint 9 successfully extended SHI from basic sell detection to comprehensive exit event classification. The implementation distinguishes true sells from transfers, LP actions, CEX deposits, burns, bridges, and wallet migrations to enable accurate realised PnL computation.

## Sprint Directive Compliance

### Hard Rules Status

| Hard Rule | Status | Implementation |
|-----------|--------|----------------|
| Balance decrease alone is NOT a sell | COMPLIANT | 10-type classification with evidence |
| Realised PnL requires sell confidence | COMPLIANT | Min threshold enforced (0.7 default) |
| LP actions must NOT be treated as sells | COMPLIANT | Dedicated LP_ADD/LP_REMOVE types |
| Transfers must NOT generate realised PnL | COMPLIANT | TRANSFER_OUT has pnl_computable=False |
| CEX deposits are uncertain exits | COMPLIANT | CEX_DEPOSIT marked as uncertain |
| Low reliability PnL must NOT display precise | COMPLIANT | Display mode based on reliability score |
| All classifications must include confidence | COMPLIANT | confidence and evidence on all results |

### Core Questions Answered

| Question | Feature | Status |
|----------|---------|--------|
| Is this balance decrease a real sell? | ExitEventClassifier | IMPLEMENTED |
| How confident are we it's a sell? | SellConfidenceScorer | IMPLEMENTED |
| Is this a wallet migration? | TransferChainDetector | IMPLEMENTED |
| Is this an LP action? | LPActionDetector | IMPLEMENTED |
| Is this a CEX deposit? | CEXDepositDetector | IMPLEMENTED |
| Can we reliably compute PnL? | PnLReliabilityScorer | IMPLEMENTED |

## Deliverables

### Code Deliverables

| File | Action | Status |
|------|--------|--------|
| `src/longitudinal/exit_classifier.py` | CREATED | Exit classifier with all 10 types |
| `src/longitudinal/__init__.py` | ENHANCED | Added Sprint 9 exports |
| `src/core/config.py` | ENHANCED | Added Sprint 9 feature flags |
| `tests/test_sprint9_exit_classifier.py` | CREATED | 41 validation tests |

### Documentation Deliverables

| Document | Status |
|----------|--------|
| `docs/exit_classifier/EXIT_EVENT_CLASSIFIER_SPEC.md` | CREATED |
| `docs/exit_classifier/SELL_CONFIDENCE_SCORE.md` | CREATED |
| `docs/exit_classifier/TRANSFER_CHAIN_DETECTION.md` | CREATED |
| `docs/exit_classifier/LP_ACTION_SEPARATION.md` | CREATED |
| `docs/exit_classifier/CEX_DEPOSIT_DETECTION.md` | CREATED |
| `docs/exit_classifier/PNL_RELIABILITY_SCORE.md` | CREATED |
| `docs/exit_classifier/SPRINT_9_VALIDATION_REPORT.md` | CREATED |
| `docs/exit_classifier/SPRINT_9_COMPLETION_REPORT.md` | CREATED |

## Key Implementations

### 1. Exit Event Classifier

Classifies token balance decreases into 10 distinct types:

```python
class ExitEventType(str, Enum):
    DEX_SELL = "dex_sell"
    TRANSFER_OUT = "transfer_out"
    CEX_DEPOSIT = "cex_deposit"
    LP_ADD = "lp_add"
    LP_REMOVE = "lp_remove"
    BURN = "burn"
    BRIDGE = "bridge"
    WALLET_MIGRATION = "wallet_migration"
    PROGRAM_INTERACTION = "program_interaction"
    UNKNOWN_EXIT = "unknown_exit"
```

### 2. Sell Confidence Scorer

Computes detailed sell confidence from 9 evidence factors:

```python
class SellConfidenceScorer:
    def compute_score(
        classification: ExitEventClassification
    ) -> tuple[float, bool, dict[str, float]]
```

### 3. Transfer Chain Detector

Detects wallet migrations via 5 signals:

```python
class TransferChainDetector:
    async def detect_migration(
        source_wallet: str,
        destination_wallet: str,
        token_mint: str,
        transfer_timestamp: datetime,
    ) -> TransferChainResult
```

### 4. LP Action Detector

Separates LP actions from sells:

```python
class LPActionDetector:
    def detect_lp_action(
        evidence: ExitEvidence,
    ) -> LPActionResult
```

### 5. CEX Deposit Detector

Identifies CEX deposit addresses:

```python
class CEXDepositDetector:
    def detect_cex_deposit(
        destination_address: str,
        fan_in_count: int | None,
        address_label: str | None,
    ) -> CEXDepositResult
```

### 6. PnL Reliability Scorer

Computes PnL reliability from 6 factors:

```python
class PnLReliabilityScorer:
    def compute_reliability(
        sell_confidence: float,
        entry_price_confidence: float,
        exit_price_confidence: float,
        liquidity_confidence: float,
        lot_count: int,
        has_transfer_ambiguity: bool,
        event_completeness: float,
    ) -> tuple[float, str, dict[str, float]]
```

## Architecture Decisions

### 1. Priority-Based Classification

**Decision**: Use priority ordering for classification (burn > LP > DEX > bridge > CEX > migration > transfer > unknown).

**Rationale**: Some exit types are more definitive than others. Burns are always burns. LP with LP tokens is always LP. This reduces misclassification.

### 2. Evidence-First Design

**Decision**: Extract all evidence before classification.

**Rationale**: Enables detailed logging, debugging, and future model training. Evidence is immutable and can be replayed.

### 3. Multiple Detection Layers

**Decision**: Separate detectors for LP, CEX, migration instead of monolithic classifier.

**Rationale**: Each detection domain has unique requirements. Separation enables independent testing and enhancement.

### 4. Confidence Everywhere

**Decision**: Every classification, detection, and score includes confidence.

**Rationale**: Per hard rules, all claims must be probabilistic. Confidence enables downstream filtering and display decisions.

### 5. Feature Flag Control

**Decision**: All Sprint 9 features behind feature flags.

**Rationale**: Enables gradual rollout and A/B testing without code changes.

## Configuration

New configuration options in `src/core/config.py`:

```python
# Sprint 9: Transfer/Sell Classification
use_exit_classifier: bool = True
use_transfer_chain_detection: bool = True
use_lp_action_separation: bool = True
use_cex_deposit_detection: bool = True
exit_min_sell_confidence_for_pnl: float = 0.7
exit_min_transfer_confidence: float = 0.5
exit_min_lp_confidence: float = 0.6
cex_high_fan_in_threshold: int = 100
cex_very_high_fan_in_threshold: int = 500
transfer_chain_rapid_followup_seconds: int = 300
transfer_chain_max_depth: int = 5
pnl_reliability_min_for_precise: float = 0.7
pnl_reliability_min_for_range: float = 0.4
```

## Test Coverage

### Test Summary

| Category | Tests | Status |
|----------|-------|--------|
| Exit Event Classification | 7 | PASS |
| Sell Confidence Scoring | 3 | PASS |
| Hard Rules Compliance | 7 | PASS |
| Transfer Chain Detection | 2 | PASS |
| LP Action Detection | 2 | PASS |
| CEX Deposit Detection | 4 | PASS |
| PnL Reliability Scoring | 4 | PASS |
| Exit Type Coverage | 6 | PASS |
| Factory Functions | 4 | PASS |
| Integration Tests | 2 | PASS |
| **Total** | **41** | **PASS** |

## Integration Points

### Sprint 8 PnL Integration

Exit classifier gates PnL computation:

```python
if classification.pnl_computable:
    pnl = pnl_calculator.compute_realised_pnl(...)
else:
    log_uncertain_exit(classification)
```

### Event Store Integration

Classifications stored as events:

```python
async def on_exit_event(event: ExitEvent):
    classification = classifier.classify(...)
    await store_classification(classification)

    if classification.pnl_computable:
        await compute_and_store_pnl(...)
```

### Risk Model Integration

Exit patterns feed into risk scoring:

```python
if settings.use_exit_classifier:
    exit_patterns = analyze_exit_patterns(wallet)
    risk_score = compute_risk_with_exits(
        ...,
        exit_patterns=exit_patterns,
    )
```

## Known Limitations

1. **New DEX Programs**: Require program ID addition for detection
2. **New CEX Addresses**: Exchanges create new deposit addresses
3. **Complex LP Operations**: Multi-hop operations may not be fully detected
4. **Bridge Variants**: New protocols need program ID addition
5. **Async Provider Required**: Full transfer chain detection needs wallet info provider

## Future Enhancements

### Immediate (Next Sprint)

1. **WalletInfoProvider Implementation**: Concrete provider for RPC/database
2. **Program ID Registry**: Dynamic program ID discovery
3. **CEX Address Updates**: Automated CEX address list maintenance

### Medium-Term

1. **ML Classification**: Train model on labeled exit data
2. **Cross-Chain Detection**: Track bridged tokens across chains
3. **Pattern Learning**: Identify new exit patterns automatically

### Long-Term

1. **Real-Time Classification**: Stream processing for live exits
2. **Behavioral Prediction**: Predict likely exit type before it happens
3. **Market Impact Modeling**: Estimate price impact from exit classification

## Conclusion

Sprint 9 successfully delivered:

1. **Exit Event Classifier** - 10 distinct exit types with priority classification
2. **Sell Confidence Scorer** - 9-factor scoring for sell confidence
3. **Transfer Chain Detector** - 5-signal migration detection
4. **LP Action Detector** - LP add/remove separation
5. **CEX Deposit Detector** - 3-method CEX identification
6. **PnL Reliability Scorer** - 6-factor reliability with display modes
7. **Comprehensive Tests** - 41 validation tests
8. **Complete Documentation** - 8 specification documents

The implementation is ready for production deployment and significantly improves PnL accuracy by preventing false sells from contaminating realised PnL data.

---

## Appendix: File Changes Summary

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/longitudinal/exit_classifier.py` | ~1000 | Exit classification infrastructure |
| `tests/test_sprint9_exit_classifier.py` | ~600 | Validation test suite |
| `docs/exit_classifier/EXIT_EVENT_CLASSIFIER_SPEC.md` | ~250 | Classifier specification |
| `docs/exit_classifier/SELL_CONFIDENCE_SCORE.md` | ~200 | Sell confidence docs |
| `docs/exit_classifier/TRANSFER_CHAIN_DETECTION.md` | ~250 | Migration detection docs |
| `docs/exit_classifier/LP_ACTION_SEPARATION.md` | ~250 | LP action docs |
| `docs/exit_classifier/CEX_DEPOSIT_DETECTION.md` | ~250 | CEX detection docs |
| `docs/exit_classifier/PNL_RELIABILITY_SCORE.md` | ~250 | Reliability scoring docs |
| `docs/exit_classifier/SPRINT_9_VALIDATION_REPORT.md` | ~300 | Validation report |
| `docs/exit_classifier/SPRINT_9_COMPLETION_REPORT.md` | ~400 | This report |

### Modified Files

| File | Changes |
|------|---------|
| `src/longitudinal/__init__.py` | Added Sprint 9 exports, updated hard rules in docstring |
| `src/core/config.py` | Added Sprint 9 feature flags and configuration |
