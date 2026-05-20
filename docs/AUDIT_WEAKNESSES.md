# SHI Audit: Weaknesses & Risks

**Date:** 2026-05-20

---

## Weakness Inventory

### Critical (P0) - Must Fix

| ID | Weakness | Impact | Risk Level |
|----|----------|--------|------------|
| W1 | **No price integration** | Missing price-volume signals | HIGH |
| W2 | **Single-token focus** | Cannot detect cross-token patterns | HIGH |
| W3 | **No labeled validation set** | Cannot measure true accuracy | HIGH |
| W4 | **Telegram-only interface** | Limits market reach | HIGH |

### Important (P1) - Should Fix

| ID | Weakness | Impact | Risk Level |
|----|----------|--------|------------|
| W5 | LP interaction ratio hardcoded to 0.0 | Incomplete feature vector | MEDIUM |
| W6 | Fixed 6 archetypes | May miss emerging patterns | MEDIUM |
| W7 | HMM with fixed 5 states | May not fit all tokens | MEDIUM |
| W8 | Manual Bayesian evidence | No automated updates | MEDIUM |
| W9 | No model drift detection | Model degradation over time | MEDIUM |
| W10 | 30-second SLA | Feels slow for trading | MEDIUM |

### Minor (P2) - Nice to Fix

| ID | Weakness | Impact | Risk Level |
|----|----------|--------|------------|
| W11 | No web dashboard | Limited accessibility | LOW |
| W12 | Minimal API documentation | Developer friction | LOW |
| W13 | No content marketing | Poor discovery | LOW |
| W14 | Single baseline version | Limited normalization | LOW |

---

## Detailed Analysis

### W1: No Price Integration

**Current State:**
- All analysis is purely behavioral/structural
- Balance changes tracked, but not correlated with price
- Sell pressure calculated without price context

**Impact:**
- Cannot detect "accumulate before pump" patterns
- Cannot calculate unrealized PnL
- Missing price-volume divergence signals

**Risk:**
- Competitors with price data have richer insights
- Users must cross-reference with other tools

**Mitigation (Planned):**
- Sprint 7: Jupiter Price API integration
- Add `entry_price_usd`, `current_price_usd`, `unrealized_pnl_ratio` features

---

### W2: Single-Token Focus

**Current State:**
- Each analysis is independent per token
- No wallet behavior tracking across tokens
- No entity aggregation

**Impact:**
- Cannot identify professional sybil operators
- Cannot build wallet reputation scores
- Cannot detect serial ruggers

**Risk:**
- Sophisticated bad actors escape detection
- Miss patterns visible only at portfolio level

**Mitigation (Planned):**
- Sprint 8-11: Cross-token intelligence
- `wallet_history` table for per-token behavior
- `entity` table for wallet grouping
- Reputation scoring system

---

### W3: No Labeled Validation Set

**Current State:**
- Models trained on unlabeled data
- No ground truth for rug detection
- Cannot calculate precision/recall/F1

**Impact:**
- "5% false positive rate" is aspirational, not measured
- Cannot compare model versions objectively
- Cannot prove value to users

**Risk:**
- Models may be worse than claimed
- Credibility gap with sophisticated users

**Mitigation:**
- Build labeled dataset from:
  - Known rugs (RugDoc, community reports)
  - Known legitimate projects (6+ months survived)
  - Historical holder snapshots
- Calculate confusion matrix
- Publish accuracy metrics

---

### W4: Telegram-Only Interface

**Current State:**
- Core SHI only accessible via Telegram bot
- SWEENEE dashboard is token-specific
- No general web interface

**Impact:**
- Users who don't use Telegram excluded
- No SEO/discoverability
- Cannot embed in other products

**Risk:**
- Limited market reach
- Competitors with web UIs have advantage

**Mitigation:**
- Build web dashboard (Next.js + Tailwind)
- Maintain Telegram as power-user interface
- Add Discord bot for that community

---

### W5: LP Interaction Ratio Hardcoded

**Current State:**
```python
# In feature extraction
lp_interaction_ratio = 0.0  # Hardcoded
```

**Impact:**
- "Liquidity Actor" archetype classification is impaired
- Missing signal for market maker detection

**Mitigation:**
- Parse LP events from Helius transaction data
- Add/remove liquidity tracking
- Calculate ratio from actual interactions

---

### W6: Fixed Archetypes

**Current State:**
- 6 archetypes defined in PDR
- Rules are frozen
- No adaptation mechanism

**Impact:**
- New wallet patterns (e.g., "bot trader") not captured
- May need periodic manual updates

**Risk:**
- Classification becomes stale over time

**Mitigation:**
- Add archetype drift detection
- Periodic HDBSCAN refit
- Allow "UNKNOWN" archetype for edge cases

---

### W7: HMM Fixed 5 States

**Current State:**
- Gaussian HMM with exactly 5 hidden states
- States mapped post-hoc to regimes

**Impact:**
- Some tokens may have <5 distinct regimes
- Some may have >5 (e.g., launch → hype → crash → recovery → stability → decay)

**Mitigation:**
- Use BIC/AIC for state count selection
- Consider sticky HDP-HMM for adaptive states
- Or: regime-specific HMMs

---

### W8: Manual Bayesian Evidence

**Current State:**
- Bayesian beliefs updated via API call
- No automatic evidence generation
- Requires manual intervention

**Impact:**
- Beliefs become stale without updates
- Full potential of uncertainty quantification unused

**Mitigation:**
- Auto-generate evidence from:
  - Metric z-scores crossing thresholds
  - Regime transitions
  - Anomaly score spikes
  - New transaction patterns

---

### W9: No Model Drift Detection

**Current State:**
- Brier score tracking infrastructure exists
- No automated monitoring
- No retraining triggers

**Impact:**
- Models degrade silently
- Users receive stale predictions

**Mitigation:**
- Implement drift detection:
  - Feature distribution shifts (KL divergence)
  - Prediction calibration (Brier score over time)
  - Coefficient stability checks
- Auto-retraining pipeline on drift

---

### W10: 30-Second SLA

**Current State:**
- Full analysis targets <30 seconds
- For large tokens, may exceed SLA

**Impact:**
- Feels slow compared to instant price checkers
- Users may abandon before completion

**Mitigation:**
- Progressive response (stream partial results)
- Cache layer for repeated queries
- Quick mode (`/quick`) with 3 key metrics

---

## Risk Matrix

| Weakness | Likelihood | Impact | Priority |
|----------|------------|--------|----------|
| W1 (No price) | Certain | High | P0 |
| W2 (Single-token) | Certain | High | P0 |
| W3 (No validation) | Certain | High | P0 |
| W4 (Telegram-only) | Certain | High | P0 |
| W5 (LP hardcoded) | Certain | Medium | P1 |
| W6 (Fixed archetypes) | Medium | Medium | P1 |
| W7 (HMM states) | Medium | Medium | P1 |
| W8 (Manual Bayesian) | High | Medium | P1 |
| W9 (No drift) | Medium | High | P1 |
| W10 (30s SLA) | Low | Medium | P2 |

---

## Technical Debt Inventory

| Area | Debt | Effort to Fix |
|------|------|---------------|
| Feature extraction | LP ratio hardcoded | 2 days |
| Clustering | No drift detection | 1 week |
| HMM | Fixed state count | 1 week |
| Bayesian | Manual updates | 3 days |
| API | Missing docs | 2 days |
| Testing | No integration tests for full pipeline | 1 week |
| Monitoring | Brier score not tracked live | 3 days |

---

## Conclusion

SHI has **significant weaknesses** that limit its current value and market reach. However, most are **addressable with clear mitigation paths**.

**Top 3 priorities:**
1. **Price integration** - Completes the analytical picture
2. **Web dashboard** - Unlocks market reach
3. **Validation dataset** - Proves accuracy claims

The technical foundation is solid; these are feature gaps, not architectural flaws.
