# Sprint 10 Completion Report

## Intelligence Corpus + Label Review System

**Sprint:** 10
**Status:** Complete
**Date:** 2024-01-15

---

## Executive Summary

Sprint 10 delivered the Intelligence Corpus system - a human-in-the-loop labelling infrastructure that serves as the data foundation for SHI behavioural intelligence. The system ensures that classifier outputs are properly validated before being used as training data or for downstream decisions.

---

## Deliverables

### Deliverable 1: Unified Label Schema
**Status:** Complete
**File:** `src/corpus/schema.py`

Implemented:
- 6 label domains (exit_event, coordination, wallet_behaviour, token_outcome, launch_trajectory, entity_resolution)
- Domain-specific label enums with valid values
- LabelRecord dataclass with full versioning
- LabelVersion for change tracking
- LabelDisagreement for dispute tracking
- LabelRepository with SQLite persistence

### Deliverable 2: Evidence Package System
**Status:** Complete
**File:** `src/corpus/evidence.py`

Implemented:
- Base EvidencePackage ABC
- 6 domain-specific evidence classes:
  - ExitEventEvidence
  - CoordinationEvidence
  - WalletBehaviourEvidence
  - TokenOutcomeEvidence
  - LaunchTrajectoryEvidence
  - EntityLinkEvidence
- EvidencePackageBuilder factory

### Deliverable 3: Review Queue
**Status:** Complete
**File:** `src/corpus/review_queue.py`

Implemented:
- Priority-based queue with configurable weights
- Priority formula: `uncertainty + impact + balance + disagreement`
- Pending queue, verification queue, disputed queue
- PriorityWeights configuration
- Automatic distribution caching

### Deliverable 4: Human Review API
**Status:** Complete
**File:** `src/corpus/api.py`

Implemented:
- GET /api/v1/review/queue
- GET /api/v1/review/item/{label_id}
- POST /api/v1/review/item/{label_id}/label
- POST /api/v1/review/item/{label_id}/verify
- POST /api/v1/review/item/{label_id}/dispute
- GET /api/v1/review/progress
- GET /api/v1/review/metrics
- GET /api/v1/review/domains

### Deliverable 5: Label Quality Metrics
**Status:** Complete
**File:** `src/corpus/quality_metrics.py`

Implemented:
- Cohen's kappa calculation
- Inter-reviewer agreement
- Model-human agreement
- Per-domain precision and recall
- Ambiguous rate, needs-context rate, disagreement rate
- Training readiness assessment
- Recommendations engine

### Deliverable 6: Corpus Export
**Status:** Complete
**File:** `src/corpus/export.py`

Implemented:
- JSONL export
- CSV export
- Parquet export (with PyArrow)
- Export configuration (filters, options)
- Export metadata generation
- Version tracking in all exports

### Deliverable 7: Active Learning Hooks
**Status:** Complete
**File:** `src/corpus/active_learning.py`

Implemented:
- Uncertainty sampling (low confidence → high priority)
- Disagreement sampling (model-human disagreement patterns)
- Rare class sampling (underrepresented classes)
- High impact sampling (critical labels)
- Hybrid sampling with weighted combination
- Training readiness checks

### Deliverable 8: Documentation
**Status:** Complete
**Location:** `docs/corpus/`

Documentation files:
- INTELLIGENCE_CORPUS_SCHEMA.md
- LABEL_DOMAINS.md
- EVIDENCE_PACKAGE_SPEC.md
- HUMAN_REVIEW_API.md
- LABEL_QUALITY_METRICS.md
- CORPUS_EXPORT_SPEC.md
- SPRINT_10_COMPLETION_REPORT.md (this file)

---

## Hard Rules Implemented

| Rule | Implementation |
|------|----------------|
| Model labels are not ground truth | All labels start as PENDING, require human review |
| Human labels must preserve reviewer and evidence | reviewer_id, evidence_json required fields |
| Every label must be versioned | version field, LabelVersion history |
| Disagreements must be preserved | LabelDisagreement tracking, DISPUTED status |
| Ambiguous is a valid label | All domain enums include ambiguous/unknown values |
| Do not train until quality metrics exist | ready_for_training flag with kappa >= 0.6 threshold |
| All exports must include model and data version | source_model_version, data_version in all exports |
| No identity claims beyond behavioural inference | Labels constrained to behavioural/entity inference |

---

## Quality Thresholds

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Cohen's Kappa | >= 0.6 | Inter-rater reliability |
| Agreement | >= 0.8 | Simple agreement rate |
| Verified Samples | >= 100 | Minimum dataset size |

---

## Review Workflow

```
                    ┌─────────────┐
                    │   PENDING   │
                    └──────┬──────┘
                           │
                    POST /label
                           │
                    ┌──────▼──────┐
                    │  LABELLED   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       POST /verify   POST /verify  POST /dispute
       (agree)        (disagree)
              │            │            │
       ┌──────▼──────┐ ┌──▼──────┐ ┌────▼─────┐
       │  VERIFIED   │ │DISPUTED │ │ DISPUTED │
       └─────────────┘ └─────────┘ └──────────┘
```

---

## API Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/v1/review/queue | GET | Get prioritized queue |
| /api/v1/review/item/{id} | GET | Get label details |
| /api/v1/review/item/{id}/label | POST | Submit human label |
| /api/v1/review/item/{id}/verify | POST | Verify/dispute label |
| /api/v1/review/item/{id}/dispute | POST | Dispute a label |
| /api/v1/review/progress | GET | Get progress stats |
| /api/v1/review/metrics | GET | Get quality metrics |
| /api/v1/review/domains | GET | Get valid labels per domain |

---

## Files Created

```
src/corpus/
├── __init__.py          # Module exports
├── schema.py            # Label schema and repository
├── evidence.py          # Evidence packages
├── review_queue.py      # Priority queue
├── quality_metrics.py   # Quality metrics computer
├── export.py            # Corpus export
├── active_learning.py   # Active learning hooks
└── api.py               # Human review API

docs/corpus/
├── INTELLIGENCE_CORPUS_SCHEMA.md
├── LABEL_DOMAINS.md
├── EVIDENCE_PACKAGE_SPEC.md
├── HUMAN_REVIEW_API.md
├── LABEL_QUALITY_METRICS.md
├── CORPUS_EXPORT_SPEC.md
└── SPRINT_10_COMPLETION_REPORT.md
```

---

## Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| structlog | Logging | Yes |
| fastapi | API framework | Yes |
| pydantic | Request/response models | Yes |
| pyarrow | Parquet export | Optional |

---

## Usage Example

```python
from corpus import (
    create_label_repository,
    create_review_queue,
    QualityMetricsComputer,
    CorpusExporter,
    ExportFormat,
    ExportConfig,
    LabelDomain,
)

# Initialize repository
repo = create_label_repository("corpus.db")

# Create review queue
queue = create_review_queue(repo)
items = queue.get_queue(limit=50)

# Check quality metrics
computer = QualityMetricsComputer(repo)
metrics = computer.compute()

if metrics.ready_for_training:
    # Export for training
    exporter = CorpusExporter(repo)
    exporter.export(
        "training_data.jsonl",
        ExportFormat.JSONL,
        ExportConfig(verified_only=True),
    )
```

---

## Next Steps

1. **Sprint 11**: Integrate corpus with exit classifier for validation
2. **Sprint 12**: Build review UI for human labellers
3. **Sprint 13**: Implement active learning loop with retrained models

---

## Notes

- The corpus system is designed as hooks/infrastructure first
- No models are trained until quality thresholds are met
- All design decisions prioritize data quality over automation speed
- Sprint 9.5's validation dataset will feed into the corpus

---

## Sign-off

Sprint 10 delivers a complete human-in-the-loop labelling infrastructure. The Intelligence Corpus is ready to receive labels from Sprint 9.5's monitored classifier deployment and begin the human review process.
