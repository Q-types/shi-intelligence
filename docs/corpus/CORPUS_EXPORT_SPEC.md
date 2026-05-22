# Corpus Export Specification

## Overview

The corpus export system provides standardized formats for exporting labelled data for training, analysis, and backup.

**HARD RULE:** All exports must include model and data version for reproducibility.

---

## Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| JSONL | `.jsonl` | Streaming, training pipelines |
| CSV | `.csv` | Analysis, spreadsheets |
| Parquet | `.parquet` | Large-scale analytics, efficient storage |

---

## Required Fields

Every export record must include:

| Field | Description |
|-------|-------------|
| `label` | Final label (human_label or proposed_label) |
| `evidence` | Evidence package (full or summary) |
| `confidence` | Confidence score |
| `source_model_version` | Model that made prediction |
| `review_status` | Current review status |
| `data_version` | Data snapshot version |

---

## Export Configuration

```python
@dataclass
class ExportConfig:
    # Filters
    domain: LabelDomain | None = None
    status: ReviewStatus | None = None
    min_confidence: float | None = None
    verified_only: bool = False

    # Content options
    include_evidence: bool = True
    include_version_history: bool = False

    # Metadata
    export_version: str = "1.0.0"
    export_description: str | None = None
```

---

## JSONL Format

### Structure

One JSON object per line:

```json
{"label_id":"lbl_001","domain":"exit_event","object_type":"exit","object_id":"exit_123","proposed_label":"dex_sell","human_label":"dex_sell","final_label":"dex_sell","model_confidence":0.85,"human_confidence":0.95,"review_status":"verified","source_model":"exit_classifier","source_model_version":"1.0.0","data_version":"2024-01-15","created_at":"2024-01-15T10:30:00Z","reviewed_at":"2024-01-15T14:00:00Z","evidence":{"exit_id":"exit_123","wallet":"abc...","amount_tokens":1000.0}}
{"label_id":"lbl_002",...}
```

### Usage

```python
from corpus import CorpusExporter, ExportFormat, ExportConfig

exporter = CorpusExporter(repository)

# Export all verified labels
metadata = exporter.export(
    output_path="exports/verified_labels.jsonl",
    format=ExportFormat.JSONL,
    config=ExportConfig(verified_only=True),
)

print(f"Exported {metadata['count']} labels")
```

---

## CSV Format

### Columns

| Column | Type | Description |
|--------|------|-------------|
| label_id | string | Unique identifier |
| domain | string | Label domain |
| object_type | string | Type of object |
| object_id | string | Object identifier |
| proposed_label | string | Model's prediction |
| human_label | string | Human-assigned label |
| final_label | string | Final label |
| model_confidence | float | Model confidence |
| human_confidence | float | Human confidence |
| review_status | string | Current status |
| source_model | string | Model name |
| source_model_version | string | Model version |
| data_version | string | Data version |
| created_at | ISO datetime | Creation time |
| reviewed_at | ISO datetime | Review time |
| evidence_json | string | JSON-encoded evidence |

### Example

```csv
label_id,domain,object_type,object_id,proposed_label,human_label,final_label,model_confidence,human_confidence,review_status,source_model,source_model_version,data_version,created_at,reviewed_at,evidence_json
lbl_001,exit_event,exit,exit_123,dex_sell,dex_sell,dex_sell,0.85,0.95,verified,exit_classifier,1.0.0,2024-01-15,2024-01-15T10:30:00Z,2024-01-15T14:00:00Z,"{""exit_id"":""exit_123""}"
```

---

## Parquet Format

### Schema

```
label_id: string (not null)
domain: string (not null)
object_type: string (not null)
object_id: string (not null)
proposed_label: string (not null)
human_label: string (nullable)
final_label: string (not null)
model_confidence: float64 (not null)
human_confidence: float64 (nullable)
review_status: string (not null)
source_model: string (not null)
source_model_version: string (not null)
data_version: string (not null)
created_at: string (ISO datetime)
reviewed_at: string (ISO datetime, nullable)
evidence_json: string (nullable, JSON-encoded)
```

### Requirements

Parquet export requires PyArrow:

```bash
pip install pyarrow
```

### Usage

```python
from corpus import CorpusExporter, ExportFormat

exporter = CorpusExporter(repository)

metadata = exporter.export(
    output_path="exports/corpus.parquet",
    format=ExportFormat.PARQUET,
)
```

---

## Export Metadata

Every export operation returns metadata:

```json
{
  "exported_at": "2024-01-15T12:00:00Z",
  "format": "jsonl",
  "output_path": "exports/verified_labels.jsonl",
  "count": 250,
  "config": {
    "domain": null,
    "status": null,
    "verified_only": true,
    "include_evidence": true
  },
  "export_version": "1.0.0"
}
```

---

## Filtering Options

### By Domain

```python
config = ExportConfig(domain=LabelDomain.EXIT_EVENT)
```

### By Status

```python
config = ExportConfig(status=ReviewStatus.VERIFIED)
```

### By Confidence

```python
config = ExportConfig(min_confidence=0.8)
```

### Verified Only

```python
config = ExportConfig(verified_only=True)
```

### Combinations

```python
config = ExportConfig(
    domain=LabelDomain.EXIT_EVENT,
    verified_only=True,
    min_confidence=0.9,
)
```

---

## Export Statistics

Before exporting, check available data:

```python
stats = exporter.get_export_stats()

print(f"Total labels: {stats['total_labels']}")
print(f"Verified: {stats['verified_labels']}")

for domain, info in stats['by_domain'].items():
    print(f"  {domain}: {info['total']} ({info['verified']} verified)")
```

---

## Training Data Preparation

### Recommended Pipeline

```python
from corpus import (
    CorpusExporter,
    ExportFormat,
    ExportConfig,
    QualityMetricsComputer,
    LabelDomain,
)

# 1. Check quality metrics first
computer = QualityMetricsComputer(repository)
metrics = computer.compute()

if not metrics.ready_for_training:
    print("Corpus not ready for training!")
    for rec in metrics.recommendations:
        print(f"  - {rec}")
    exit(1)

# 2. Export verified labels with high confidence
exporter = CorpusExporter(repository)

for domain in LabelDomain:
    config = ExportConfig(
        domain=domain,
        verified_only=True,
        min_confidence=0.8,
    )

    metadata = exporter.export(
        output_path=f"training_data/{domain.value}.jsonl",
        format=ExportFormat.JSONL,
        config=config,
    )

    print(f"{domain.value}: {metadata['count']} samples")
```

---

## Best Practices

1. **Always check quality metrics before exporting for training**
2. **Use verified_only=True for training data**
3. **Include evidence for debugging and audit**
4. **Version your exports** - Include date in filename
5. **Store export metadata** - Track what was exported when
6. **Use Parquet for large exports** - Better compression and performance
7. **Filter by domain for domain-specific models**

---

## Export Checklist

Before exporting for model training:

- [ ] Quality metrics meet thresholds (kappa >= 0.6)
- [ ] Sufficient verified samples (>= 100)
- [ ] Dispute rate acceptable (< 15%)
- [ ] Export includes model and data version
- [ ] Export metadata saved alongside data
