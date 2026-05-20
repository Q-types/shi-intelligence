# Hazard Model Comparison Report

Generated: 2026-05-20T16:40:42.154973+00:00

## Executive Summary

**Best Model:** model_b_expanded

**Recommendation:** CAUTIOUS DEPLOY model_b_expanded: Marginal improvement +0.2

---

## Model Configurations

| Model | Description | Features |
|-------|-------------|----------|
| Model A | Baseline | Original 10 features |
| Model B | Expanded | + swap, LP, delta_30d, eigenvector |
| Model C | Price/Liquidity | + unrealized_pnl, liquidity, sell_pressure |
| Model D | Missingness | + missingness indicators |

## Model Performance Comparison

| Model | C-Index | C-Index Std | Brier | Calibration Slope | PH Assumption |
|-------|---------|-------------|-------|-------------------|---------------|
| model_a_baseline | 0.884 | 0.087 | 0.139 | 2.096 | PASS |
| model_b_expanded | 0.885 | 0.094 | 0.138 | 2.065 | PASS |
| model_c_price_liquidity | 0.882 | 0.092 | 0.138 | 2.061 | PASS |
| model_d_missingness | 0.882 | 0.092 | 0.138 | 2.061 | PASS |

## Walk-Forward Validation

| Model | WF Concordance | WF Std | Folds |
|-------|----------------|--------|-------|
| model_a_baseline | 0.884 | 0.087 | 3 |
| model_b_expanded | 0.885 | 0.094 | 3 |
| model_c_price_liquidity | 0.882 | 0.092 | 3 |
| model_d_missingness | 0.882 | 0.092 | 3 |

## Calibration Analysis

| Model | Brier Score | Slope | Intercept | Mean Predicted | Mean Observed |
|-------|-------------|-------|-----------|----------------|---------------|
| model_a_baseline | 0.139 | 2.096 | -0.257 | 0.222 | 0.208 |
| model_b_expanded | 0.138 | 2.065 | -0.250 | 0.222 | 0.208 |
| model_c_price_liquidity | 0.138 | 2.061 | -0.249 | 0.222 | 0.208 |
| model_d_missingness | 0.138 | 2.061 | -0.249 | 0.222 | 0.208 |

## PH Assumption Test Results

### model_a_baseline: ✓ PASSES
- Global p-value: 0.500

### model_b_expanded: ✓ PASSES
- Global p-value: 0.500

### model_c_price_liquidity: ✓ PASSES
- Global p-value: 0.500

### model_d_missingness: ✓ PASSES
- Global p-value: 0.500

## Coefficient Stability (CV)

Lower values indicate more stable coefficients across validation folds.

### model_a_baseline
- Stable (CV < 0.5): holding_duration, trade_count, burstiness, in_degree, out_degree
- Unstable (CV ≥ 0.5): share, entry_time_relative, position_volatility, delta_balance_7d

### model_b_expanded
- Stable (CV < 0.5): holding_duration, trade_count, burstiness, in_degree, out_degree
- Unstable (CV ≥ 0.5): share, entry_time_relative, position_volatility, delta_balance_7d, shared_funder_count

### model_c_price_liquidity
- Stable (CV < 0.5): holding_duration, trade_count, burstiness, in_degree, out_degree
- Unstable (CV ≥ 0.5): share, entry_time_relative, position_volatility, delta_balance_7d, shared_funder_count

### model_d_missingness
- Stable (CV < 0.5): holding_duration, trade_count, burstiness, in_degree, out_degree
- Unstable (CV ≥ 0.5): share, entry_time_relative, position_volatility, delta_balance_7d, shared_funder_count

## Decision Rationale

- model_a_baseline: C=0.884, Brier=0.139, PH=PASS, score=71.7
- model_b_expanded: C=0.885, Brier=0.138, PH=PASS, score=72.0
- model_c_price_liquidity: C=0.882, Brier=0.138, PH=PASS, score=71.8
- model_d_missingness: C=0.882, Brier=0.138, PH=PASS, score=71.8

---

## Important Notes

- Models with improved C-index but degraded calibration are REJECTED
- PH assumption violations may require stratified models
- Walk-forward validation reflects production performance better than temporal CV