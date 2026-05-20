# Calibration Audit Report

Generated: 2026-05-20T17:47:18.423726+00:00

## Executive Summary

**Goal:** Produce probabilities that are honest, stable, and decision-useful.

**Key Metrics:**
- C-index (discrimination) - higher is better
- Brier score (calibration) - lower is better
- Calibration slope - closer to 1.0 is better
- ECE (Expected Calibration Error) - lower is better

---

## Baseline Model Comparison

| Model | C-Index | Brier | Slope | Intercept | ECE | MCE |
|-------|---------|-------|-------|-----------|-----|-----|
| model_a_baseline | 0.408 | 0.2720 | 0.432 | 0.335 | 0.1410 | 0.3279 |
| model_b_expanded | 0.409 | 0.2724 | 0.424 | 0.337 | 0.1398 | 0.3376 |
| model_c_price_liquidity | 0.409 | 0.2724 | 0.424 | 0.337 | 0.1398 | 0.3376 |
| model_d_missingness | 0.409 | 0.2724 | 0.424 | 0.337 | 0.1398 | 0.3376 |

## Calibration Curves by Decile

### model_a_baseline

| Decile | Predicted | Observed | Count |
|--------|-----------|----------|-------|
| 1 | 0.057 | 0.385 | 117 ** |
| 2 | 0.148 | 0.383 | 120 ** |
| 3 | 0.255 | 0.438 | 144 ** |
| 4 | 0.348 | 0.500 | 168 ** |
| 5 | 0.453 | 0.484 | 190 |
| 6 | 0.546 | 0.617 | 175 |
| 7 | 0.633 | 0.590 | 78 |
| 8 | 0.724 | 0.625 | 8 |
| 9 | 0.000 | 0.000 | 0 |
| 10 | 0.000 | 0.000 | 0 |

### model_b_expanded

| Decile | Predicted | Observed | Count |
|--------|-----------|----------|-------|
| 1 | 0.056 | 0.393 | 117 ** |
| 2 | 0.149 | 0.387 | 124 ** |
| 3 | 0.255 | 0.419 | 148 ** |
| 4 | 0.347 | 0.513 | 158 ** |
| 5 | 0.454 | 0.497 | 193 |
| 6 | 0.550 | 0.586 | 162 |
| 7 | 0.639 | 0.591 | 88 |
| 8 | 0.735 | 0.900 | 10 ** |
| 9 | 0.000 | 0.000 | 0 |
| 10 | 0.000 | 0.000 | 0 |

### model_c_price_liquidity

| Decile | Predicted | Observed | Count |
|--------|-----------|----------|-------|
| 1 | 0.056 | 0.393 | 117 ** |
| 2 | 0.149 | 0.387 | 124 ** |
| 3 | 0.255 | 0.419 | 148 ** |
| 4 | 0.347 | 0.513 | 158 ** |
| 5 | 0.454 | 0.497 | 193 |
| 6 | 0.550 | 0.586 | 162 |
| 7 | 0.639 | 0.591 | 88 |
| 8 | 0.735 | 0.900 | 10 ** |
| 9 | 0.000 | 0.000 | 0 |
| 10 | 0.000 | 0.000 | 0 |

### model_d_missingness

| Decile | Predicted | Observed | Count |
|--------|-----------|----------|-------|
| 1 | 0.056 | 0.393 | 117 ** |
| 2 | 0.149 | 0.387 | 124 ** |
| 3 | 0.255 | 0.419 | 148 ** |
| 4 | 0.347 | 0.513 | 158 ** |
| 5 | 0.454 | 0.497 | 193 |
| 6 | 0.550 | 0.586 | 162 |
| 7 | 0.639 | 0.591 | 88 |
| 8 | 0.735 | 0.900 | 10 ** |
| 9 | 0.000 | 0.000 | 0 |
| 10 | 0.000 | 0.000 | 0 |

---

## Key Observations

- **Best Brier Score:** model_a_baseline (0.2720)
- **Best Calibration Slope:** model_a_baseline (0.432)

**Note:** A slope > 1.0 indicates under-confident predictions. A slope < 1.0 indicates over-confident predictions.
