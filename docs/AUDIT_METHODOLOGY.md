# SHI Audit: Methodology Evaluation

**Date:** 2026-05-20

---

## Statistical Rigor Assessment

### Score: 9.0/10 - Excellent

SHI demonstrates **exceptional statistical rigor** for a crypto analytics system. The methodology draws from established academic fields:

- **Survival Analysis** (biostatistics)
- **Hidden Markov Models** (signal processing)
- **Graph Theory** (network science)
- **Bayesian Inference** (probabilistic reasoning)
- **Anomaly Detection** (machine learning)

---

## Core Methodologies

### 1. Cox Proportional Hazards Model

**Appropriateness:** ✅ Excellent choice for sell probability

**Rationale:**
- Handles censored data (wallets that haven't sold yet)
- Models time-to-event with covariates
- Well-established in survival analysis literature

**Implementation Quality:**
```python
# Efron tie handling (correct for discrete time)
# Ridge regularization (penalizer=0.1)
# Proper validation: concordance, PH test, Schoenfeld residuals
```

**Potential Issues:**
- ⚠️ **Proportional hazards assumption** may not hold if market conditions change
- Recommendation: Add time-varying covariates or regime-stratified models

### 2. HDBSCAN Clustering

**Appropriateness:** ✅ Excellent for wallet clustering

**Rationale:**
- Density-based (handles arbitrary cluster shapes)
- Automatic cluster count selection
- Robust to outliers

**Implementation Quality:**
- min_cluster_size=5 (reasonable for wallet groups)
- 14+ feature dimensions (comprehensive)
- StandardScaler normalization

**Potential Issues:**
- ⚠️ **Fixed archetypes** post-clustering may miss emergent patterns
- Recommendation: Periodic re-clustering with drift detection

### 3. Gaussian HMM Regime Detection

**Appropriateness:** ✅ Good for structural regime shifts

**Rationale:**
- Captures latent market states
- Probabilistic transitions
- Well-suited for time-series with regime changes

**Implementation Quality:**
- 5 states (reasonable for holder dynamics)
- Gaussian emissions (appropriate for normalized metrics)
- Viterbi decoding for state sequence

**Potential Issues:**
- ⚠️ **Unsupervised** - state labels assigned post-hoc
- ⚠️ **Fixed 5 states** may not match all tokens
- Recommendation: Consider sticky HDP-HMM for adaptive state count

### 4. Isolation Forest Anomaly Detection

**Appropriateness:** ✅ Good for sybil detection

**Rationale:**
- Unsupervised (no labeled sybil data required)
- Efficient on high-dimensional data
- Interpretable anomaly scores

**Implementation Quality:**
- 100 estimators
- 5% contamination (conservative)
- Features: structural + embeddings + behavioral

**Potential Issues:**
- ⚠️ **No ground truth** for sybil labels
- Recommendation: Build labeled validation set from known rugs

### 5. Bayesian Belief Updating

**Appropriateness:** ✅ Excellent for uncertainty quantification

**Rationale:**
- Explicit uncertainty representation
- Combines prior knowledge with evidence
- Interpretable credible intervals

**Implementation Quality:**
- Beta priors (conjugate for probability estimates)
- Evidence weighting
- Information gain tracking

**Potential Issues:**
- ⚠️ **Manual evidence injection** - not automated
- Recommendation: Auto-generate evidence from pipeline outputs

---

## Metric Reproducibility

### Frozen PDR Metrics (§4)

| Metric | Version | Formula Locked | Test Coverage |
|--------|---------|---------------|---------------|
| HHI | 1.0.0 | ✅ | Unit tests |
| Gini | 1.0.0 | ✅ | Unit tests |
| Shannon Entropy | 1.0.0 | ✅ | Unit tests |
| WDR | 1.0.0 | ✅ | Unit tests |
| Churn | 1.0.0 | ✅ | Unit tests |
| Coordination | 1.0.0 | ✅ | Unit tests |

**Assessment:** ✅ Excellent - Deterministic, versioned, tested

---

## Validation Framework

### Model Validation

| Model | Validation Method | Status |
|-------|------------------|--------|
| Cox PH | Concordance index, PH test, Schoenfeld | ✅ Implemented |
| HDBSCAN | Silhouette score, cluster stability | ⚠️ Partial |
| HMM | Log-likelihood, BIC | ⚠️ Partial |
| Isolation Forest | Contamination tuning | ⚠️ Partial |

### Backtesting

| Type | Status |
|------|--------|
| Historical rug detection | ⚠️ Not implemented |
| Sell probability calibration | ⚠️ Brier score tracking exists but needs data |
| Regime prediction accuracy | ⚠️ Not implemented |

**Recommendation:** Build labeled dataset from historical rugs for validation

---

## Calibration & Drift Detection

### Current State

- **Brier score tracking:** ✅ Infrastructure exists
- **Drift detection:** ⚠️ Partial (coefficient stability checks)
- **Auto-retraining:** ❌ Not implemented

### Recommendations

1. Implement automated Brier score monitoring
2. Add covariate shift detection (KL divergence on features)
3. Build retraining pipeline triggered by drift thresholds

---

## Adversarial Robustness

### Threat Model

| Attack | Mitigation | Status |
|--------|-----------|--------|
| **Sybil clusters** | Graph anomaly detection | ✅ |
| **Wash trading** | Coordination score | ✅ |
| **Funding obfuscation** | Ancestor traversal | ✅ |
| **Metric gaming** | Multiple correlated metrics | ✅ |
| **Model poisoning** | Baseline versioning | ⚠️ Partial |

### Stress Tests

- ✅ SLA tests (30-second response under load)
- ⚠️ Adversarial Sybil tests (partial)
- ❌ Red team exercises

---

## Methodological Gaps

### Critical

1. **No price integration** - Missing price-volume correlation
2. **No labeled validation set** - Cannot measure true accuracy

### Important

3. **Fixed archetype definitions** - May miss new patterns
4. **HMM state count fixed** - May not fit all tokens
5. **Cox PH assumption untested** - Proportional hazards may fail

### Nice-to-Have

6. **Node2Vec hyperparameters** - Not tuned per-token
7. **Baseline selection** - Single baseline version

---

## Conclusion

SHI's methodology is **academically rigorous and well-implemented**. The combination of survival analysis, graph intelligence, and Bayesian uncertainty is sophisticated and differentiated.

**Key strength:** Deterministic, versioned metrics ensure reproducibility.

**Key weakness:** Lack of labeled validation data limits accuracy measurement.

**Priority recommendation:** Build labeled dataset from known rugs/scams for proper model validation.
