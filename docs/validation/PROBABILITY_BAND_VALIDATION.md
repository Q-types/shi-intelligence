# Probability Band Validation Report

Generated: 2026-05-20T17:47:18.600866+00:00

## Acceptance Criterion

Predicted probabilities should roughly match observed rates within uncertainty bounds.

---

## Band Analysis

| Band | Wallets | Predicted | Observed | 95% CI | Calibrated |
|------|---------|-----------|----------|--------|------------|
| 0-10% | 117 | 0.056 | 0.393 | [0.309, 0.484] | ✗ |
| 10-25% | 189 | 0.176 | 0.376 | [0.310, 0.447] | ✗ |
| 25-50% | 434 | 0.381 | 0.498 | [0.451, 0.545] | ✗ |
| 50-75% | 257 | 0.585 | 0.599 | [0.538, 0.657] | ✓ |
| 75-100% | 3 | 0.771 | 0.667 | [0.208, 0.939] | ✓ |

---

## Overall Assessment

**FAIL:** Some probability bands are miscalibrated.

Miscalibrated bands: 0-10%, 10-25%, 25-50%

Apply calibration method before deployment.
