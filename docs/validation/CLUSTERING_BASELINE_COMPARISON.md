# Clustering Baseline Comparison Report

Generated: 2026-05-20T16:40:42.142027+00:00

## Executive Summary

**Best Pipeline:** new_behavior_only

**Recommendation:** DEPLOY new_behavior_only: Improvement of 31.6 points over baseline

---

## Pipeline Comparison

| Pipeline | Clusters | Noise % | Silhouette | UNKNOWN % | Multi-Archetype % | Mean Confidence |
|----------|----------|---------|------------|-----------|-------------------|-----------------|
| old_rule_first | 2 | 0.9% | 0.103 | 50.8% | 0.0% | 0.700 |
| new_behavior_only | 2 | 0.0% | 0.992 | 9.0% | 64.8% | 0.999 |
| new_graph_only | 2 | 0.7% | 0.343 | 9.0% | 64.8% | 0.890 |
| new_combined | 0 | 1.0% | N/A | 9.0% | 64.8% | 0.847 |

## Stability Analysis (Bootstrap)

| Pipeline | ARI Mean | ARI Std | NMI Mean | Persistence Rate |
|----------|----------|---------|----------|------------------|
| old_rule_first | -0.007 | 0.015 | 0.013 | -0.007 |
| new_behavior_only | 0.009 | 0.025 | 0.016 | 0.009 |
| new_graph_only | -0.006 | 0.025 | 0.023 | -0.006 |
| new_combined | 0.000 | 0.000 | 0.000 | 0.000 |

## Archetype Distribution by Pipeline

### old_rule_first

| Archetype | Count | Percentage |
|-----------|-------|------------|
| sniper | 2 | 0.4% |
| long_term_accumulator | 138 | 27.6% |
| coordinated_cluster | 24 | 4.8% |
| liquidity_actor | 72 | 14.4% |
| exchange_linked | 0 | 0.0% |
| dormant_whale | 10 | 2.0% |
| unknown | 254 | 50.8% |

### new_behavior_only

| Archetype | Count | Percentage |
|-----------|-------|------------|
| sniper | 54 | 10.8% |
| long_term_accumulator | 74 | 14.8% |
| coordinated_cluster | 285 | 57.0% |
| liquidity_actor | 31 | 6.2% |
| exchange_linked | 0 | 0.0% |
| dormant_whale | 11 | 2.2% |
| unknown | 45 | 9.0% |

### new_graph_only

| Archetype | Count | Percentage |
|-----------|-------|------------|
| sniper | 54 | 10.8% |
| long_term_accumulator | 74 | 14.8% |
| coordinated_cluster | 285 | 57.0% |
| liquidity_actor | 31 | 6.2% |
| exchange_linked | 0 | 0.0% |
| dormant_whale | 11 | 2.2% |
| unknown | 45 | 9.0% |

### new_combined

| Archetype | Count | Percentage |
|-----------|-------|------------|
| sniper | 54 | 10.8% |
| long_term_accumulator | 74 | 14.8% |
| coordinated_cluster | 285 | 57.0% |
| liquidity_actor | 31 | 6.2% |
| exchange_linked | 0 | 0.0% |
| dormant_whale | 11 | 2.2% |
| unknown | 45 | 9.0% |

## Interpretability Notes

### old_rule_first
- HIGH UNKNOWN: 50.8% wallets unclassified
- WEAK SEPARATION: Low silhouette (0.103)

### new_behavior_only
- GOOD SEPARATION: High silhouette (0.992)
- MULTI-ARCHETYPE: 64.8% have secondary labels

### new_graph_only
- MULTI-ARCHETYPE: 64.8% have secondary labels
- Graph embeddings may capture structural patterns not visible in behavior
- Lower interpretability: embeddings are not directly explainable

### new_combined
- WARNING: No clusters found - all points as noise
- MULTI-ARCHETYPE: 64.8% have secondary labels
- Combined: 70% behavior + 30% graph

## Decision Rationale

- old_rule_first: score=42.7
- new_behavior_only: score=74.3
- new_graph_only: score=63.3
- new_combined: score=42.3
- Best: new_behavior_only with score 74.3
- Improvement over baseline: 31.6

---

## Acceptance Criteria Assessment

The new default must improve interpretability, stability, or downstream predictive value.

**Assessment:** DEPLOY new_behavior_only: Improvement of 31.6 points over baseline