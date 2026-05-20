# Cluster Semantics Audit Report

Generated: 2026-05-20T17:29:30.478350+00:00

## Executive Summary

**Clusters Represent Real Behaviour:** YES

### Primary Concerns

- WARNING: 39.8% of wallets overridden to coordinated from other archetypes

### Recommendations


---

## 1. Silhouette vs Stability Analysis

**Contradiction Detected:** No

| Metric | Value | Assessment |
|--------|-------|------------|
| Silhouette Score | 0.078 | Normal |
| ARI (Bootstrap) | -0.040 | LOW |
| NMI (Bootstrap) | 0.026 | LOW |
| Clusters | 2 | Binary |
| Size Imbalance | 2.4x | Normal |

### Dominant Features Driving Separation

| Feature | Single-Feature Silhouette |
|---------|---------------------------|
| eigenvector_centrality | 0.520 |
| in_degree | 0.426 |

---

## 2. Coordinated Cluster Dominance Analysis

**Coordinated Percentage:** 39.8% (199/500)

**Current Threshold:** shared_funder_count >= 5

**Wallets Meeting Threshold:** 43.2% (216/500)

### Shared Funder Distribution

| Count | Wallets | Cumulative % |
|-------|---------|--------------|
| 0 | 38 | 7.6% |
| 1 | 75 | 22.6% |
| 2 | 61 | 34.8% |
| 3 | 62 | 47.2% |
| 4 | 48 | 56.8% |
| 5 | 46 | 66.0% |
| 6 | 33 | 72.6% |
| 7 | 34 | 79.4% |
| 8 | 23 | 84.0% |
| 9 | 18 | 87.6% |

**Median Shared Funders:** 4.0
**Mean Shared Funders:** 4.7

### Override Analysis

**Wallets Overridden to Coordinated:** 199

| Original Archetype | Count |
|-------------------|-------|
| long_term_accumulator | 53 |
| sniper | 85 |
| liquidity_actor | 42 |
| dormant_whale | 19 |

### Threshold Recommendation

**Recommended Threshold:** 5

**Rationale:** Current threshold seems appropriate

---

## 3. Feature Contribution Analysis

**Primary Driver:** TEMPORAL

| Feature Group | Silhouette | Contribution % |
|---------------|------------|----------------|
| Temporal | 0.731 | 43.2% |
| Graph | 0.476 | 28.2% |
| Trading | 0.484 | 28.6% |

### Strategic Assessment

**Is SHI a Timing Engine?** No

**Does Graph Add Value?** YES

---

## 4. Cluster Profiles

### Cluster 0

**Size:** 7 wallets (1.4%)

**Pattern:** short-term traders, small holders

**Primary Archetype:** sniper (57.1% purity)

**Archetype Distribution:**

- coordinated_cluster: 2 (28.6%)
- sniper: 4 (57.1%)
- dormant_whale: 1 (14.3%)

**Temporal:** Entry=0.60, Hold=4.2

### Cluster 1

**Size:** 17 wallets (3.4%)

**Pattern:** bursty trading, small holders

**Primary Archetype:** sniper (41.2% purity)

**Archetype Distribution:**

- long_term_accumulator: 5 (29.4%)
- unknown: 1 (5.9%)
- sniper: 7 (41.2%)
- coordinated_cluster: 4 (23.5%)

**Temporal:** Entry=0.58, Hold=8.0

---

## Key Insight

High silhouette with low bootstrap stability indicates **geometric separation without semantic stability**.

The clusters may look clean in feature space but do not represent robust, reproducible behavioural patterns.

Before deploying as default, validate against known wallet behaviours:
- Known rug wallets
- Known sniper wallets
- Known LP operators
- Known coordinated launches
