# Regime-Specific Calibration Report

Generated: 2026-05-20T17:47:18.600043+00:00

## Regimes Analyzed

- **ACCUMULATION:** Net positive balance changes
- **DISTRIBUTION:** Net negative balance changes
- **COORDINATED_ACCUMULATION:** High shared funder count
- **DECAY:** Moderate activity, negative trend
- **STABLE:** Low activity, neutral balance

---

## Per-Regime Calibration Metrics

| Regime | Samples | Event Rate | Slope | Brier | ECE | Adequate |
|--------|---------|------------|-------|-------|-----|----------|
| distribution | 193 | 45.6% | 0.524 | 0.2505 | 0.1022 | ✓ |
| coordinated_accumulation | 225 | 53.8% | 0.423 | 0.2839 | 0.1726 | ✓ |
| decay | 367 | 49.6% | 0.365 | 0.2792 | 0.1475 | ✓ |
| accumulation | 202 | 45.5% | 0.424 | 0.2689 | 0.1393 | ✓ |
| stable | 13 | 46.2% | 0.683 | 0.2620 | 0.2660 | ✗ |

---

## Deployment Recommendation

**Regime-specific calibration NOT recommended.**

Regimes with insufficient samples: stable

Use global calibration instead until more data is collected.
