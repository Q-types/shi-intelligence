# Calibration Method Comparison Report

Generated: 2026-05-20T17:47:18.529671+00:00

## Methods Tested

- **Isotonic:** Non-parametric monotonic regression
- **Platt:** Logistic regression calibration
- **Beta:** Three-parameter beta calibration
- **Regime-Specific:** Separate calibrators per token regime

---

## Comparison Results

| Method | Brier Δ | ECE Δ | Slope Δ | C-Index Δ | Recommended |
|--------|---------|-------|---------|-----------|-------------|
| isotonic | +0.0242 | +0.0903 | +0.213 | +0.005 | ✓ |
| platt | +0.0260 | +0.0912 | +0.374 | +0.000 | ✓ |
| beta | +0.0263 | +0.0961 | +0.389 | +0.000 | ✓ |
| regime_specific | +0.0198 | +0.0756 | +0.058 | +0.012 | ✓ |

**Note:** Positive Δ means improvement (except C-Index where negative is bad).

---

## Detailed Results

### isotonic

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Brier | 0.2726 | 0.2485 | +0.0242 |
| ECE | 0.1268 | 0.0365 | +0.0903 |
| Slope | 0.356 | 0.569 | +0.213 |
| C-Index | 0.423 | 0.428 | +0.005 |

**Recommendation:** Improves Brier by 0.0242, ECE by 0.0903, slope by 0.213

### platt

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Brier | 0.2726 | 0.2466 | +0.0260 |
| ECE | 0.1268 | 0.0355 | +0.0912 |
| Slope | 0.356 | 0.730 | +0.374 |
| C-Index | 0.423 | 0.423 | +0.000 |

**Recommendation:** Improves Brier by 0.0260, ECE by 0.0912, slope by 0.374

### beta

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Brier | 0.2726 | 0.2463 | +0.0263 |
| ECE | 0.1268 | 0.0306 | +0.0961 |
| Slope | 0.356 | 0.745 | +0.389 |
| C-Index | 0.423 | 0.423 | +0.000 |

**Recommendation:** Improves Brier by 0.0263, ECE by 0.0961, slope by 0.389

### regime_specific

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Brier | 0.2726 | 0.2529 | +0.0198 |
| ECE | 0.1268 | 0.0511 | +0.0756 |
| Slope | 0.356 | 0.414 | +0.058 |
| C-Index | 0.423 | 0.435 | +0.012 |

**Recommendation:** Improves Brier by 0.0198, ECE by 0.0756, slope by 0.058

---

## Deployment Recommendation

**Recommended Method:** beta

**Rationale:** Improves Brier by 0.0263, ECE by 0.0961, slope by 0.389
