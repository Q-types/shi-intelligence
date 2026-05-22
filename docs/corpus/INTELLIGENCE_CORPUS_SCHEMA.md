# Intelligence Corpus Schema

## Overview

The Intelligence Corpus provides a unified schema for storing and reviewing behavioural intelligence labels across SHI. All classifier outputs flow through this system before being used for training or downstream decisions.

## Core Principle

**Classifier outputs are NOT ground truth.**

Every inferred label must support:
- Evidence
- Confidence
- Human review
- Disagreement handling
- Versioning
- Audit trail

## Hard Rules

1. **Model labels are not ground truth** - All model predictions require human verification before use as training data
2. **Human labels must preserve reviewer and evidence** - Full audit trail required
3. **Every label must be versioned** - No silent overwrites
4. **Disagreements must be preserved, not overwritten** - Track all disputes
5. **Ambiguous is a valid label** - Better to acknowledge uncertainty than force a wrong label
6. **Do not train supervised models until label quality metrics exist** - Cohen's kappa >= 0.6, verified >= 100
7. **All exports must include model and data version** - Reproducibility requirement
8. **No identity claims beyond behavioural/entity inference** - Privacy constraint

## Label Domains

The corpus supports 6 label domains:

| Domain | Description | Example Labels |
|--------|-------------|----------------|
| `exit_event` | How tokens left a wallet | DEX_SELL, TRANSFER_OUT, LP_ADD |
| `coordination` | Cluster coordination assessment | TRUE_COORDINATED, FALSE_POSITIVE |
| `wallet_behaviour` | Wallet classification | SNIPER, ACCUMULATOR, WHALE |
| `token_outcome` | Token lifecycle outcome | RUG_PULL, ORGANIC_FAILURE, SUCCESS |
| `launch_trajectory` | Launch pattern classification | ORGANIC, INSIDER_COORDINATED |
| `entity_resolution` | Entity linking decisions | SAME_ENTITY, DIFFERENT_ENTITY |

## Review Status Workflow

```
PENDING → LABELLED → VERIFIED
              ↓
          DISPUTED → (resolution) → VERIFIED/REJECTED
              ↓
    NEEDS_MORE_CONTEXT
```

### Status Definitions

- **PENDING**: Awaiting first human review
- **LABELLED**: First reviewer has assigned a label
- **VERIFIED**: Second reviewer confirmed the label
- **DISPUTED**: Reviewers disagree, needs resolution
- **REJECTED**: Label rejected as invalid
- **NEEDS_MORE_CONTEXT**: Insufficient evidence to label

## LabelRecord Schema

```python
@dataclass
class LabelRecord:
    # Identity
    label_id: str               # Unique identifier
    domain: LabelDomain         # Which domain
    object_type: str            # e.g., "exit", "cluster", "wallet"
    object_id: str              # ID of labelled object

    # Model prediction
    proposed_label: str         # Model's proposed label
    model_confidence: float     # Model's confidence (0-1)
    source_model: str           # Model name
    source_model_version: str   # Model version

    # Human review
    human_label: str | None     # Human-assigned label
    human_confidence: float | None  # Human confidence
    review_status: ReviewStatus # Current status
    reviewer_id: str | None     # Who reviewed
    notes: str | None           # Review notes

    # Dual review (for verification)
    second_label: str | None    # Second reviewer's label

    # Evidence
    evidence_json: str          # JSON blob of evidence package

    # Versioning
    data_version: str           # Data snapshot version
    created_at: datetime        # When created
    reviewed_at: datetime | None  # When reviewed

    # Audit
    version: int                # Label version number
```

## Version History

Every change to a label creates a version record:

```python
@dataclass
class LabelVersion:
    version_id: str
    label_id: str
    version: int
    previous_label: str | None
    new_label: str
    change_reason: str
    changed_by: str
    changed_at: datetime
```

## Disagreement Tracking

When reviewers disagree:

```python
@dataclass
class LabelDisagreement:
    disagreement_id: str
    label_id: str
    first_label: str
    first_reviewer: str
    second_label: str
    second_reviewer: str
    resolution: str | None      # How resolved
    resolved_at: datetime | None
```

## Database Schema (SQLite)

```sql
CREATE TABLE labels (
    label_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    proposed_label TEXT NOT NULL,
    model_confidence REAL NOT NULL,
    source_model TEXT NOT NULL,
    source_model_version TEXT NOT NULL,
    human_label TEXT,
    human_confidence REAL,
    review_status TEXT NOT NULL DEFAULT 'pending',
    reviewer_id TEXT,
    notes TEXT,
    second_label TEXT,
    evidence_json TEXT NOT NULL,
    data_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX idx_labels_domain ON labels(domain);
CREATE INDEX idx_labels_status ON labels(review_status);
CREATE INDEX idx_labels_object ON labels(object_type, object_id);
```

## Usage Example

```python
from corpus import (
    create_label_repository,
    LabelDomain,
    ExitEventLabel,
    ExitEventEvidence,
)

# Create repository
repo = create_label_repository("/path/to/corpus.db")

# Create evidence package
evidence = ExitEventEvidence(
    exit_id="exit_123",
    wallet="abc...",
    token_mint="xyz...",
    amount_tokens=1000.0,
    value_sol=5.5,
    counterparty="raydium_pool",
    transaction_signature="tx...",
    block_time="2024-01-15T10:30:00Z",
    program_id="675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    confidence_factors={"signature_match": 0.95},
)

# Create label record
repo.create_label(
    domain=LabelDomain.EXIT_EVENT,
    object_type="exit",
    object_id="exit_123",
    proposed_label=ExitEventLabel.DEX_SELL.value,
    model_confidence=0.85,
    source_model="exit_classifier",
    source_model_version="1.0.0",
    evidence_json=evidence.to_json(),
    data_version="2024-01-15",
)
```
