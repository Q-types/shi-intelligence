# Exit Classifier Production Report

**Version**: 1.0.0
**Date**: 2026-05-22
**Sprint**: 9.5 - Post-Deployment Validation
**Status**: READY FOR CONTROLLED ROLLOUT

## Executive Summary

Sprint 9.5 implements the production monitoring infrastructure required for controlled deployment of the exit classifier. This enables:

1. **Full evidence logging** for every classification
2. **Distribution monitoring** with drift detection
3. **Anomaly alerting** for misclassifications and threshold violations
4. **Validation dataset collection** for accuracy measurement

### Critical Distinction

| Status | Meaning |
|--------|---------|
| **Implementation ready for controlled production rollout** | Code is tested, monitoring in place |
| ~~Classification accuracy proven~~ | NOT YET - requires validation dataset |

## Deliverables

### Code Deliverables

| File | Lines | Purpose |
|------|-------|---------|
| `src/longitudinal/exit_classifier_monitoring.py` | ~500 | Logging, dashboard, alerts |
| `src/longitudinal/exit_validation_dataset.py` | ~450 | Labelled dataset builder |
| `src/longitudinal/__init__.py` | Enhanced | Sprint 9.5 exports |

### Documentation Deliverables

| Document | Status |
|----------|--------|
| `docs/monitoring/EXIT_CLASSIFIER_MONITORING.md` | Complete |
| `docs/monitoring/EXIT_CLASSIFIER_VALIDATION_DATASET.md` | Complete |
| `docs/monitoring/EXIT_CLASSIFIER_PRODUCTION_REPORT.md` | This document |

## Implementation Details

### Monitoring Components

```
MonitoredExitClassifier
├── ClassificationLogger (SQLite persistence)
├── DistributionDashboard (metrics computation)
└── AlertManager (anomaly detection)
```

### Alert Thresholds

| Alert | Threshold | Severity |
|-------|-----------|----------|
| UNKNOWN_EXIT_HIGH | > 20% | warning |
| DISTRIBUTION_SHIFT | > 2σ | warning |
| LP_AS_SELL | Any occurrence | critical |
| LOW_CONFIDENCE_PNL | confidence < 0.5 | warning |

### Validation Dataset Targets

| Exit Type | Target | Purpose |
|-----------|--------|---------|
| DEX sells | 50 | Precision measurement |
| Transfers | 50 | Recall measurement |
| LP actions | 20 | Hard rule verification |
| CEX deposits | 20 | Detection accuracy |
| Unknown | 20 | Taxonomy coverage |
| **Total** | **160** | Statistical significance |

## Hard Rules Compliance

| Hard Rule | Implementation |
|-----------|----------------|
| All classifications logged | ClassificationLogger persists every classification |
| Unknown exits not hidden | Logged with `unknown_reason` field |
| Classifier errors reviewable | Full evidence_json preserved |
| Low-confidence no precise PnL | `display_mode` field gates presentation |
| Real-world validation required | ValidationDatasetBuilder + metrics |

## Deployment Checklist

### Pre-Deployment

- [x] Monitoring code implemented and tested
- [x] Database schema verified
- [x] Alert handlers configured
- [x] Documentation complete

### Controlled Rollout

- [ ] Enable monitored classifier in production
- [ ] Verify classification logs are being written
- [ ] Confirm dashboard metrics are computing
- [ ] Validate alert system is triggering

### Post-Deployment (First 24h)

- [ ] Review initial distribution snapshot
- [ ] Check UNKNOWN_EXIT rate
- [ ] Investigate any critical alerts
- [ ] Begin validation sample collection

### Validation Phase (Week 1-2)

- [ ] Collect 160+ validation samples
- [ ] Complete human labelling
- [ ] Compute accuracy metrics
- [ ] Review false sell rate
- [ ] Verify LP misclassification = 0

### Production Promotion

- [ ] All acceptance criteria met
- [ ] Accuracy metrics documented
- [ ] Stakeholder sign-off
- [ ] Remove "monitored" status

## Usage

### Enable Monitoring

```python
from src.longitudinal import (
    create_exit_classifier,
    create_monitored_classifier,
    create_monitoring_config,
)

classifier = create_exit_classifier()
config = create_monitoring_config(
    unknown_threshold_pct=20.0,
)
monitored = create_monitored_classifier(classifier, config)
```

### Check Dashboard

```python
data = monitored.dashboard.get_dashboard_data()
print(f"Unknown rate: {data['summary']['unknown_exit_pct']}%")
```

### Collect Validation Samples

```python
from src.longitudinal import (
    create_validation_dataset_builder,
    create_sample_collector,
)

builder = create_validation_dataset_builder()
collector = create_sample_collector(builder, sampling_rate=0.1)

# During classification
collector.maybe_collect(
    signature=classification.evidence.signature,
    token_mint=classification.evidence.token_mint,
    wallet_address=wallet,
    classifier_exit_type=classification.exit_type.value,
    classifier_confidence=classification.confidence,
    classifier_evidence_json=evidence_json,
)
```

### Export Validation Dataset

```python
builder.export_dataset("~/.shi/validation_dataset_v1.json")
```

## Known Limitations

1. **CEX address list incomplete** - New addresses created frequently
2. **LP program ID coverage** - New AMMs may not be detected
3. **Historical context limited** - Transfer chain detection requires provider
4. **Single-reviewer labels** - Dual review recommended for disputed samples

## Metrics to Track

### Daily Metrics

- Total classifications
- Distribution by exit type
- UNKNOWN_EXIT rate
- Alert count by type
- Validation progress

### Weekly Metrics

- Classification accuracy (once dataset ready)
- False sell rate trend
- Unknown exit breakdown
- Distribution stability

## Next Sprint: Intelligence Corpus

After validation confirms classifier accuracy:

**Sprint 10: Intelligence Corpus + Label Review System**

Extends labelling to:
- Coordination clusters
- Wallet behaviour labels
- Token outcome labels

This transforms SHI from rule-based to learnable.

## Conclusion

Sprint 9.5 delivers the infrastructure required for safe production deployment of the exit classifier:

- **Complete evidence logging** for audit and debugging
- **Distribution monitoring** for operational awareness
- **Alert system** for anomaly detection
- **Validation framework** for accuracy measurement

The classifier is ready for controlled rollout with the understanding that classification accuracy remains unproven until validation is complete.

---

## Appendix: File Locations

| Path | Purpose |
|------|---------|
| `~/.shi/exit_classifier_monitoring.db` | Classification logs |
| `~/.shi/exit_validation_dataset.db` | Validation samples |
| `~/.shi/validation_dataset_v1.json` | Exported dataset |
