# Multi-Evidence Coordination Feature Specification

**Generated**: 2026-05-21T09:43:21.056524+00:00

## Overview

This document specifies the multi-evidence coordination features used to detect
wallet coordination in the SHI system. These features replace the failed
temporal-only coordination detector.

## Feature Categories

### 1. Funding Similarity Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `shared_funder_binary` | bool | Whether wallets share at least one funder | 0/1 |
| `shared_funder_depth` | int | Depth of shared funder (1=direct, 2+=indirect) | 0-3 |
| `funder_overlap_jaccard` | float | Jaccard similarity of funder sets | 0-1 |
| `funding_amount_similarity` | float | Log-scale similarity of funding amounts | 0-1 |
| `funding_amount_ratio` | float | Ratio of smaller to larger amount | 0-1 |
| `funding_time_similarity` | float | Exponential decay of funding time gap | 0-1 |
| `funding_time_gap_seconds` | float | Absolute time gap between fundings | 0-∞ |

### 2. Trading Similarity Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `first_buy_time_similarity` | float | Exponential decay of first buy time gap | 0-1 |
| `first_buy_time_gap_seconds` | float | Absolute time gap between first buys | 0-∞ |
| `buy_sequence_similarity` | float | Pearson correlation of buy amount sequences | 0-1 |
| `sell_sequence_similarity` | float | Pearson correlation of sell amount sequences | 0-1 |
| `trade_cadence_similarity` | float | Ratio of trade counts | 0-1 |
| `dex_route_similarity` | float | Jaccard similarity of DEX routes used | 0-1 |

### 3. Behavioral Similarity Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `holding_duration_similarity` | float | Ratio of holding durations | 0-1 |
| `position_size_similarity` | float | Log-scale similarity of position sizes | 0-1 |
| `profit_taking_similarity` | float | Similarity of profit percentages | 0-1 |
| `exit_timing_similarity` | float | Exponential decay of exit time gap | 0-1 |

### 4. Cross-Token Similarity Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `repeated_co_participation_count` | int | Number of tokens both wallets traded | 0-∞ |
| `shared_previous_tokens` | int | Tokens traded together before current | 0-∞ |
| `historical_exit_correlation` | float | Correlation of exit timing across tokens | 0-1 |
| `entity_reuse_score` | float | Evidence of entity reuse patterns | 0-1 |

## Feature Computation

### Time Similarity Formula

```
similarity = exp(-gap_seconds / scale)
scale = max_gap_seconds / 2.3
```

Where `max_gap_seconds` is the maximum gap for non-zero similarity.

### Amount Similarity Formula

```
log_ratio = log10(max(a1, a2) / min(a1, a2))
similarity = 1.0 / (1.0 + log_ratio)
```

### Sequence Similarity

Uses Pearson correlation coefficient converted to 0-1 range:

```
similarity = |correlation(seq1, seq2)|
```

## Evidence Type Detection

A feature is considered "present" (contributing evidence) when:

| Evidence Type | Threshold |
|---------------|-----------|
| `shared_funder` | binary=True OR jaccard>0.3 |
| `funding_time` | similarity>0.3 |
| `funding_amount` | similarity>0.3 |
| `buy_time` | similarity>0.3 |
| `trade_sequence` | similarity>0.3 |
| `exit_timing` | similarity>0.3 |
| `cross_token` | co_participation>=2 |

## Critical Rule

**NEVER** use timing features alone for coordination detection.
The minimum evidence types threshold (default: 3) ensures multiple
independent signals must be present.
