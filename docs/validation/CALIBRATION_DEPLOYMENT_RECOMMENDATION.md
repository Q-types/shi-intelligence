# Calibration Deployment Recommendation

Generated: 2026-05-20T17:47:18.601053+00:00

## Decision Framework

Deploy calibration if it improves:
- Brier score (lower)
- ECE (lower)
- Calibration slope (closer to 1.0)

Without materially damaging C-index (≤2% drop acceptable).

---

## Summary of Findings

**Best Baseline Model:** model_a_baseline
- Brier: 0.2720
- Slope: 0.432
- ECE: 0.1410

**Best Calibration Method:** beta
- Brier improvement: +0.0263
- ECE improvement: +0.0961
- Slope improvement: +0.389

**Regime-Specific Calibration:** Not recommended
- Adequate regimes: 4/5

**Probability Bands:** 2/5 well-calibrated

---

## Recommended Configuration

```python
use_probability_calibration = True
calibration_method = "beta"
use_regime_specific_calibration = False
```

---

## Next Steps

1. Validate on larger dataset
2. Monitor calibration metrics in production
3. Re-calibrate periodically as data distribution shifts
