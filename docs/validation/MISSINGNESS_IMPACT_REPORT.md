# Missingness Impact Report

Generated: 2026-05-20T16:40:42.155164+00:00

## Executive Summary

**Total Wallets:** 500

**Wallets with Any Missing:** 89 (17.8%)

**Informative Features:** 0

**Recommendation:** No significant informative missingness detected

---

## Missingness by Category

| Category | Features | Missing % | Any Missing % |
|----------|----------|-----------|---------------|
| temporal_history | 3 | 0.0% | 0.0% |
| trade_history | 5 | 0.0% | 0.0% |
| price_data | 1 | 0.0% | 0.0% |
| graph_data | 7 | 2.2% | 3.8% |
| liquidity_data | 2 | 14.8% | 14.8% |

## Informative Missingness

Features where missingness is statistically associated with outcomes:

| Feature | Missing % | Predicts Event | Predicts UNKNOWN | Predicts Anomaly | Predicts Coordination |
|---------|-----------|----------------|------------------|------------------|----------------------|

## Event Rate Analysis

Comparison of sell event rates between missing and present values:

| Feature | Event Rate (Missing) | Event Rate (Present) | Rate Ratio | P-value |
|---------|---------------------|----------------------|------------|---------|
| in_degree | 0.158 | 0.210 | 0.75 | 0.7945 |
| out_degree | 0.158 | 0.210 | 0.75 | 0.7945 |
| eigenvector_centrality | 0.158 | 0.210 | 0.75 | 0.7945 |
| shared_funder_count | 0.158 | 0.210 | 0.75 | 0.7945 |
| liquidity_usd_current | 0.230 | 0.204 | 1.12 | 0.7310 |
| sell_pressure_vs_liquidity | 0.230 | 0.204 | 1.12 | 0.7310 |

## Detailed Feature Patterns

### liquidity_usd_current

- Missing: 74 (14.8%)
- Predicts sell event: No
- Predicts UNKNOWN: No
- Predicts anomaly: No
- Event rate when missing: 0.230
- Event rate when present: 0.204

### sell_pressure_vs_liquidity

- Missing: 74 (14.8%)
- Predicts sell event: No
- Predicts UNKNOWN: No
- Predicts anomaly: No
- Event rate when missing: 0.230
- Event rate when present: 0.204

---

## Key Insight

Missingness may be informative rather than merely inconvenient. Features where 
missingness significantly predicts outcomes should have missingness indicators 
retained as features in the model.
