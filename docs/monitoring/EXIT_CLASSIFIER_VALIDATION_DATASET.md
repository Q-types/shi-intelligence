# Exit Classifier Validation Dataset Specification

**Version**: 1.0.0
**Date**: 2026-05-22
**Sprint**: 9.5 - Post-Deployment Validation

## Overview

This document specifies the labelled validation dataset for measuring exit classifier accuracy. Human-verified labels are required before using classifier outputs as training data.

## Hard Rule

**Real-world validation required before using exit classes as training labels.**

Classifier outputs must not be treated as ground truth until:
1. Validation dataset is complete (160 samples minimum)
2. Accuracy metrics are computed and reviewed
3. False rates are within acceptable bounds

## Dataset Targets

| Exit Type | Target Samples | Purpose |
|-----------|----------------|---------|
| DEX sells | 50 | Measure sell detection precision |
| Transfers | 50 | Measure transfer detection precision |
| LP actions | 20 | Verify LP separation (critical hard rule) |
| CEX deposits | 20 | Verify CEX detection accuracy |
| Unknown exits | 20 | Understand unknown classification patterns |
| **Total** | **160** | Minimum for statistical significance |

## Validation Labels

Human reviewers assign one of these labels:

| Label | Description | Maps to Exit Type |
|-------|-------------|-------------------|
| `TRUE_DEX_SELL` | Confirmed swap on DEX with quote received | `dex_sell` |
| `TRUE_TRANSFER` | Simple token transfer to another wallet | `transfer_out` |
| `TRUE_LP_ADD` | Add liquidity action | `lp_add` |
| `TRUE_LP_REMOVE` | Remove liquidity action | `lp_remove` |
| `TRUE_CEX_DEPOSIT` | Transfer to CEX deposit address | `cex_deposit` |
| `TRUE_BURN` | Token sent to burn address | `burn` |
| `TRUE_BRIDGE` | Cross-chain bridge transfer | `bridge` |
| `TRUE_MIGRATION` | Internal wallet migration | `wallet_migration` |
| `TRUE_PROGRAM_INTERACTION` | Program interaction, not a trade | `program_interaction` |
| `AMBIGUOUS` | Cannot determine even with manual review | N/A |
| `NEEDS_MORE_CONTEXT` | Need additional data to determine | N/A |

## Validation Workflow

### 1. Sample Collection

Samples are collected automatically during monitored production:

```python
collector = create_sample_collector(
    dataset_builder,
    sampling_rate=0.1,  # Sample 10% of classifications
)

# During classification pipeline
collector.maybe_collect(
    signature=evidence.signature,
    token_mint=evidence.token_mint,
    wallet_address=wallet,
    classifier_exit_type=classification.exit_type.value,
    classifier_confidence=classification.confidence,
    classifier_evidence_json=evidence_json,
)
```

Stratified sampling ensures each exit type reaches its target.

### 2. Human Labelling

Pending samples are reviewed by humans:

```python
# Get samples needing review
pending = dataset_builder.get_pending_samples(
    exit_type="dex_sell",
    limit=10,
)

# Label a sample
dataset_builder.label_sample(
    sample_id="val_abc123def456",
    label=ValidationLabel.TRUE_DEX_SELL,
    reviewer_id="analyst_1",
    notes="Clear Jupiter swap, SOL received",
)
```

### 3. Verification (Optional)

For critical samples, a second reviewer can verify:

```python
dataset_builder.verify_sample(
    sample_id="val_abc123def456",
    label=ValidationLabel.TRUE_DEX_SELL,
    reviewer_id="analyst_2",
)
# Status becomes VERIFIED if labels match, DISPUTED if not
```

### 4. Accuracy Computation

Once sufficient samples are labelled:

```python
metrics = dataset_builder.compute_accuracy_metrics()
```

## Sample Record Format

```json
{
  "sample_id": "val_a1b2c3d4e5f6",
  "signature": "5xK8j...",
  "token_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
  "wallet_address": "Hj5k...",
  "classifier_exit_type": "dex_sell",
  "classifier_confidence": 0.92,
  "classifier_evidence_json": "{...}",
  "status": "labelled",
  "human_label": "true_dex_sell",
  "reviewer_id": "analyst_1",
  "review_timestamp": "2026-05-22T14:30:00Z",
  "review_notes": "Clear Jupiter swap, 1.5 SOL received",
  "second_reviewer_id": null,
  "second_review_timestamp": null,
  "second_label": null,
  "created_at": "2026-05-22T10:15:00Z"
}
```

## Accuracy Metrics

### Per-Type Precision

For each classifier exit type:

```
precision = correct_classifications / total_predictions
```

Where `correct_classifications` = predictions where `classifier_exit_type` matches `human_label`.

### Per-Type Recall

For each true label:

```
recall = correct_classifications / total_actual
```

Where `total_actual` = all samples with that human label.

### Critical Rates

| Metric | Formula | Target |
|--------|---------|--------|
| False Sell Rate | (transfers misclassified as sells) / (total sell predictions) | < 10% |
| False Transfer Rate | (sells misclassified as transfers) / (total transfer predictions) | < 10% |
| Unknown Exit Rate | (unknown exits) / (total classifications) | < 20% |
| LP Misclassification | (LP actions as sells) / (total LP actions) | 0% |

### Metrics Output Format

```json
{
  "total_labelled": 142,
  "by_classifier_type": {
    "dex_sell": {
      "total": 45,
      "correct": 41,
      "precision": 0.911
    },
    "transfer_out": {
      "total": 42,
      "correct": 38,
      "precision": 0.905
    }
  },
  "by_true_label": {
    "true_dex_sell": {
      "total": 50,
      "correctly_classified": 46,
      "recall": 0.920
    }
  },
  "false_sell_rate": 0.067,
  "false_transfer_rate": 0.048,
  "unknown_exit_analysis": {
    "total": 18,
    "breakdown": {
      "true_dex_sell": 3,
      "true_transfer": 8,
      "true_program_interaction": 5,
      "ambiguous": 2
    }
  }
}
```

## Review Interface

### Recommended Review Information

For each sample, reviewers should see:

1. **Transaction signature** (link to Solscan/Solana Explorer)
2. **Classifier evidence** (program IDs, SOL change, destination)
3. **Token movement** (amount, decimals, mint)
4. **Classifier decision** (exit type, confidence, factors)
5. **Destination analysis** (address, type if known)

### Review Guidelines

| Exit Type | Key Indicators |
|-----------|----------------|
| DEX_SELL | DEX program + quote asset received + no LP tokens |
| TRANSFER_OUT | Token program only + destination is wallet + no swap |
| LP_ADD | LP program + LP tokens minted + no quote received |
| LP_REMOVE | LP program + LP tokens burned + tokens received |
| CEX_DEPOSIT | Destination is known CEX or high fan-in |
| BURN | Destination is burn address |
| BRIDGE | Bridge program + cross-chain mechanics |
| WALLET_MIGRATION | Same funder + no quote + rapid movement |

### Handling Edge Cases

- **Multiple operations in tx**: Label the primary intent
- **Partial fills**: Label based on actual execution
- **Failed transactions**: Should not appear in dataset (filter upstream)
- **MEV bundles**: Label the user's actual intent, not sandwiching

## Progress Tracking

```python
progress = dataset_builder.get_progress()

print(f"Completion: {progress.completion_pct():.1f}%")
print(f"DEX sells: {progress.labelled.get('dex_sell', 0)}/50")
print(f"Transfers: {progress.labelled.get('transfer_out', 0)}/50")
print(f"LP actions: {progress.labelled.get('lp_add', 0) + progress.labelled.get('lp_remove', 0)}/20")
```

## Dataset Export

Export complete dataset for analysis:

```python
dataset_builder.export_dataset(
    output_path=Path("~/.shi/validation_dataset_v1.json")
)
```

Exported format includes:
- All samples with labels
- Current metrics
- Progress against targets

## Acceptance Criteria

Before using classifier outputs as training labels:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Total labelled | >= 160 | Minimum for statistical power |
| DEX_SELL precision | >= 85% | Core use case accuracy |
| False sell rate | <= 10% | Protect PnL accuracy |
| LP misclassification | 0% | Hard rule compliance |
| Unknown exit rate | <= 20% | Taxonomy coverage |

## Next Steps After Validation

1. **If criteria met**: Promote classifier to production status
2. **If false sell rate high**: Investigate misclassified samples, refine rules
3. **If unknown rate high**: Analyze unknown breakdown, expand taxonomy
4. **If LP misclassification > 0**: Critical bug - fix before any further use

## Integration with Sprint 10

Once validation is complete, the labelled dataset becomes the foundation for:

- **Sprint 10: Intelligence Corpus** - Extends to coordination clusters, wallet behaviour
- **Label Review System** - Human-in-the-loop for ongoing quality
- **Model Training** - Supervised learning on verified labels
