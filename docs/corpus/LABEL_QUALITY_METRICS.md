# Label Quality Metrics

## Overview

Label quality metrics determine whether the corpus is ready for model training. The system computes inter-rater reliability, model-human agreement, and per-domain precision/recall.

## Training Readiness Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Cohen's Kappa | >= 0.6 | Inter-rater reliability |
| Inter-reviewer Agreement | >= 0.8 | Simple agreement rate |
| Verified Samples | >= 100 | Minimum verified labels |

**HARD RULE:** Do not train supervised models until these thresholds are met.

---

## Cohen's Kappa

Cohen's kappa measures inter-rater reliability, adjusting for chance agreement.

### Formula

```
kappa = (p_o - p_e) / (1 - p_e)
```

Where:
- `p_o` = observed agreement (proportion of agreements)
- `p_e` = expected agreement by chance

### Interpretation

| Kappa Range | Interpretation |
|-------------|----------------|
| < 0.20 | Poor |
| 0.20 - 0.40 | Fair |
| 0.40 - 0.60 | Moderate |
| 0.60 - 0.80 | Substantial |
| 0.80 - 1.00 | Almost perfect |

### Usage

```python
from corpus import compute_cohens_kappa

first_labels = ["dex_sell", "transfer_out", "dex_sell", "dex_sell"]
second_labels = ["dex_sell", "dex_sell", "dex_sell", "dex_sell"]

kappa = compute_cohens_kappa(first_labels, second_labels)
# kappa = 0.62 (substantial agreement)
```

---

## Inter-Reviewer Agreement

Simple percentage of cases where both reviewers agree.

### Formula

```
agreement = agreements / total
```

### Usage

```python
from corpus import compute_inter_reviewer_agreement

agreement = compute_inter_reviewer_agreement(first_labels, second_labels)
# agreement = 0.75 (75% agree)
```

---

## Model-Human Agreement

Percentage of cases where model prediction equals human label.

### Interpretation

| Agreement | Implication |
|-----------|-------------|
| < 0.5 | Model needs significant improvement |
| 0.5 - 0.7 | Model captures some patterns |
| 0.7 - 0.85 | Model is useful with human oversight |
| > 0.85 | Model is highly reliable |

---

## Per-Domain Metrics

### DomainMetrics Structure

```python
@dataclass
class DomainMetrics:
    domain: str
    total_labels: int
    pending: int
    labelled: int
    verified: int
    disputed: int

    # Agreement
    inter_reviewer_agreement: float | None
    cohens_kappa: float | None
    model_human_agreement: float | None

    # Per-label performance
    model_precision: dict[str, float]
    model_recall: dict[str, float]

    # Distribution
    label_distribution: dict[str, int]

    # Quality indicators
    ambiguous_rate: float
    needs_context_rate: float
    disagreement_rate: float
```

### Precision and Recall

For each label class:

```
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
```

Where:
- True positive: Model predicted label AND human confirmed
- False positive: Model predicted label BUT human disagreed
- False negative: Model predicted different BUT human said this label

---

## Quality Indicators

### Ambiguous Rate

Percentage of labels marked as "ambiguous" by humans.

| Rate | Implication |
|------|-------------|
| < 5% | Clear label definitions |
| 5-10% | Some edge cases |
| 10-20% | Label taxonomy may need refinement |
| > 20% | Significant taxonomy issues |

### Needs-Context Rate

Percentage of labels marked as "needs_more_context".

| Rate | Implication |
|------|-------------|
| < 5% | Evidence packages are sufficient |
| 5-10% | Some evidence gaps |
| > 10% | Evidence packages need enrichment |

### Disagreement Rate

Percentage of labels in DISPUTED status.

| Rate | Implication |
|------|-------------|
| < 5% | Good reviewer alignment |
| 5-15% | Normal for complex domains |
| > 15% | Training or guideline issues |

---

## Quality Metrics Computer

### Usage

```python
from corpus import QualityMetricsComputer, create_label_repository

repo = create_label_repository("/path/to/corpus.db")
computer = QualityMetricsComputer(repo)

# Compute all metrics
metrics = computer.compute()

print(f"Ready for training: {metrics.ready_for_training}")
print(f"Cohen's Kappa: {metrics.overall_cohens_kappa}")
print(f"Agreement: {metrics.overall_inter_reviewer_agreement}")
print(f"Verified: {metrics.total_verified}")

# Get recommendations
for rec in metrics.recommendations:
    print(f"- {rec}")
```

### Output Structure

```python
@dataclass
class LabelQualityMetrics:
    computed_at: datetime
    total_labels: int
    total_reviewed: int
    total_verified: int

    overall_inter_reviewer_agreement: float | None
    overall_cohens_kappa: float | None
    overall_model_human_agreement: float | None

    domain_metrics: dict[str, DomainMetrics]

    kappa_threshold_met: bool
    agreement_threshold_met: bool

    ready_for_training: bool
    recommendations: list[str]
```

---

## Recommendations Engine

The system generates actionable recommendations based on metrics:

### Sample Recommendations

| Condition | Recommendation |
|-----------|----------------|
| verified < 100 | "Need more verified labels (X/100 minimum for training)" |
| kappa is None | "Need dual-reviewed samples to compute inter-rater reliability" |
| kappa < 0.4 | "Low inter-rater reliability. Review labelling guidelines." |
| kappa < 0.6 | "Moderate inter-rater reliability. Target kappa >= 0.6 before training." |
| ambiguous_rate > 0.2 | "High ambiguous rate in {domain}. Refine label taxonomy." |
| disagreement_rate > 0.15 | "High disagreement rate in {domain}. Update guidelines." |
| pending > reviewed | "Review backlog in {domain}: X pending vs Y reviewed." |
| All good | "Label quality metrics look good. Ready for model training." |

---

## Integration with Active Learning

Quality metrics inform active learning sampling:

```python
from corpus import ActiveLearningHooks

hooks = ActiveLearningHooks(repository)

# Check readiness before training
readiness = hooks.check_training_readiness()

if readiness["ready"]:
    # Safe to train
    pass
else:
    print(f"Not ready: kappa={readiness['kappa']}")
    for rec in readiness["recommendations"]:
        print(f"- {rec}")
```

---

## Monitoring Dashboard

Key metrics to monitor:

```
+---------------------------------------------------+
| CORPUS QUALITY DASHBOARD                          |
+---------------------------------------------------+
| Total Labels:  1,500  | Verified:    250 (16.7%)  |
| Pending:         800  | Disputed:     50 (3.3%)   |
+---------------------------------------------------+
| Cohen's Kappa:  0.72  | Agreement:  87.0%         |
| Model-Human:    81.0% | Ambiguous:   2.0%         |
+---------------------------------------------------+
| Training Ready: YES   | Recommendations: 0        |
+---------------------------------------------------+
```

---

## Best Practices

1. **Monitor metrics continuously** - Quality can drift as reviewers change
2. **Address recommendations promptly** - Don't let backlogs grow
3. **Calibrate reviewers** - Regular alignment sessions on edge cases
4. **Track per-domain** - Some domains are harder than others
5. **Use disputed cases for training** - They reveal edge cases
