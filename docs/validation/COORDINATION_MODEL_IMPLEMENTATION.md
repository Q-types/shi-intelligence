# Multi-Evidence Coordination Model Implementation

**Generated**: 2026-05-21T09:43:21.057100+00:00

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-EVIDENCE COORDINATION                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Funding   │  │   Trade     │  │  Holder     │            │
│  │   Graph     │  │   Events    │  │   Data      │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         │                │                │                    │
│         └────────────────┼────────────────┘                    │
│                          ▼                                     │
│              ┌──────────────────────┐                          │
│              │   Build Wallet       │                          │
│              │   Contexts           │                          │
│              └──────────┬───────────┘                          │
│                         │                                      │
│                         ▼                                      │
│              ┌──────────────────────┐                          │
│              │   Create Candidate   │                          │
│              │   Blocks (Blocking)  │                          │
│              └──────────┬───────────┘                          │
│                         │                                      │
│                         ▼                                      │
│              ┌──────────────────────┐                          │
│              │   Compute Pairwise   │                          │
│              │   Features           │                          │
│              └──────────┬───────────┘                          │
│                         │                                      │
│                         ▼                                      │
│              ┌──────────────────────┐                          │
│              │   Score Coordination │                          │
│              │   (Weighted Sum)     │                          │
│              └──────────┬───────────┘                          │
│                         │                                      │
│                         ▼                                      │
│              ┌──────────────────────┐                          │
│              │   Null Model         │                          │
│              │   Validation         │                          │
│              └──────────┬───────────┘                          │
│                         │                                      │
│                         ▼                                      │
│              ┌──────────────────────┐                          │
│              │   Classify           │                          │
│              │   Coordination       │                          │
│              └──────────────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Scoring Formula

```
CoordinationScore =
    w1 * shared_funder_similarity
  + w2 * funding_time_similarity
  + w3 * funding_amount_similarity
  + w4 * first_buy_time_similarity
  + w5 * trade_sequence_similarity
  + w6 * exit_timing_similarity
  + w7 * cross_token_reuse
```

### Default Weights

| Component | Weight | Rationale |
|-----------|--------|-----------|
| shared_funder | 0.25 | Strongest signal - explicit funding link |
| funding_time | 0.15 | Important but needs corroboration |
| funding_amount | 0.10 | Supporting evidence |
| buy_time | 0.15 | Important but needs corroboration |
| trade_sequence | 0.10 | Supporting evidence |
| exit_timing | 0.10 | Supporting evidence |
| cross_token | 0.15 | Strong signal - repeated behavior |
| **Total** | **1.00** | |

## Blocking Strategies

To avoid O(n²) pairwise comparison, we use blocking:

| Strategy | Description | Complexity Reduction |
|----------|-------------|---------------------|
| `SAME_FUNDER` | Group by shared funder | Very high |
| `FUNDING_TIME_WINDOW` | 24h funding windows | High |
| `POSITION_SIZE_BUCKET` | Log-scale size buckets | Moderate |
| `TOKEN_ENTRY_WINDOW` | 1h entry windows | High |
| `CO_PARTICIPATION` | Shared token history | Very high |

## Classification Thresholds

| Threshold | Default | Description |
|-----------|---------|-------------|
| `z_threshold` | 2.5 | Minimum z-score for significance |
| `p_threshold` | 0.01 | Maximum p-value for significance |
| `min_evidence_types` | 3 | Minimum independent evidence types |
| `min_cluster_size` | 3 | Minimum wallets in cluster |

## Configuration

All thresholds are configurable via `src/core/config.py`:

```python
# Multi-Evidence Coordination Detection
use_temporal_coordination: bool = False  # DISABLED
use_multi_evidence_coordination: bool = True

coordination_min_evidence_types: int = 3
coordination_z_threshold: float = 2.5
coordination_p_threshold: float = 0.01
coordination_min_cluster_size: int = 3

# Weights
coordination_weight_shared_funder: float = 0.25
coordination_weight_funding_time: float = 0.15
coordination_weight_funding_amount: float = 0.10
coordination_weight_buy_time: float = 0.15
coordination_weight_trade_sequence: float = 0.10
coordination_weight_exit_timing: float = 0.10
coordination_weight_cross_token: float = 0.15
```

## Module Structure

```
src/coordination/
├── __init__.py          # Public API exports
├── features.py          # Feature computation
├── blocking.py          # Candidate block construction
├── scoring.py           # Coordination scoring
├── null_model.py        # Null model validation
└── orchestrator.py      # Main detector orchestrator
```

## Usage Example

```python
from src.coordination import MultiEvidenceCoordinationDetector

detector = MultiEvidenceCoordinationDetector()
result = detector.detect(
    funding_graph=graph,
    trade_events=events,
    target_wallets=wallets,
    run_null_validation=True,
)

for cluster in result.coordinated_clusters:
    print(f"Cluster {cluster.cluster_id}: z={cluster.z_score:.2f}, p={cluster.empirical_p:.4f}")
```
