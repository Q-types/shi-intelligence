# SHI Audit: Improvement Opportunities

**Date:** 2026-05-20

---

## Improvement Roadmap

### Phase 1: Complete the Core (Sprints 7-11)

These improvements are **already planned** and should be prioritized:

| Sprint | Focus | Key Deliverables |
|--------|-------|------------------|
| **7** | Price Integration | Jupiter API, unrealized PnL, price features |
| **8** | Data Foundation | wallet_history, entity tables, migrations |
| **9** | Entity Resolution | SharedFunderDetector, TemporalSyncDetector |
| **10** | Reputation System | ReputationScorer, PatternDetector, `/profile` |
| **11** | Cross-Token Intel | SybilNetworkDetector, EntityRiskAggregator |

### Phase 2: Accessibility (New)

| Item | Description | Effort | Impact |
|------|-------------|--------|--------|
| **Web Dashboard** | Next.js app with core features | 4 weeks | HIGH |
| **API Documentation** | OpenAPI spec + interactive docs | 1 week | MEDIUM |
| **Discord Bot** | Port Telegram commands to Discord | 2 weeks | MEDIUM |
| **Mobile App** | React Native for iOS/Android | 8 weeks | HIGH |

### Phase 3: Model Enhancement (New)

| Item | Description | Effort | Impact |
|------|-------------|--------|--------|
| **Adaptive Archetypes** | Periodic HDBSCAN refit + drift detection | 2 weeks | MEDIUM |
| **Flexible HMM** | BIC-based state selection | 1 week | LOW |
| **Auto-Bayesian** | Generate evidence from pipeline | 1 week | MEDIUM |
| **Validation Dataset** | Label historical rugs | 2 weeks | HIGH |

### Phase 4: Scale & Reliability (New)

| Item | Description | Effort | Impact |
|------|-------------|--------|--------|
| **Streaming Responses** | Progressive result delivery | 1 week | MEDIUM |
| **Enhanced Caching** | Multi-tier with smart invalidation | 1 week | MEDIUM |
| **Model Retraining** | Automated pipeline on drift | 2 weeks | MEDIUM |
| **Multi-region** | Deploy to EU/Asia | 2 weeks | LOW |

---

## Detailed Improvement Specs

### 1. Web Dashboard MVP

**Goal:** Provide web-based access to core SHI features

**Components:**
```
/dashboard
├── /analyze/[mint]     - Full token analysis
├── /wallet/[address]   - Wallet profile
├── /watchlist          - User's watched wallets
├── /alerts             - Alert configuration
└── /history            - Analysis history
```

**Tech Stack:**
- Next.js 14 (App Router)
- Tailwind CSS + shadcn/ui
- React Query for data fetching
- WebSocket for real-time alerts
- Supabase Auth (or similar)

**Features:**
- Token search with autocomplete
- Interactive metric charts
- Funding graph visualization (D3.js)
- Alert configuration UI
- Export to PDF/CSV

**Effort:** 4 weeks (2 devs)

---

### 2. Quick Analysis Mode

**Goal:** Provide instant risk assessment for casual users

**New Command:** `/quick <mint>`

**Response (< 10 seconds):**
```
🎯 Quick Risk: $TOKEN

Risk: 🟡 MEDIUM (Score: 65/100)
Phase: 📈 ACCUMULATION
Whales: Top 10 hold 45%

⚠️ 3 wallets share funders (sybil signal)

Full analysis: /analyze <mint>
```

**Implementation:**
- Precompute key metrics on popular tokens
- Return cached values instantly
- Async full analysis in background

**Effort:** 3 days

---

### 3. Validation Dataset

**Goal:** Enable measurement of true accuracy

**Dataset Structure:**
```
{
  "mint": "...",
  "snapshot_date": "2025-01-15",
  "label": "RUG",  // RUG | LEGITIMATE | SUSPICIOUS
  "label_date": "2025-01-20",  // When outcome was known
  "label_source": "RugDoc",  // Source of label
  "holder_snapshot": {...},  // Frozen metrics at snapshot_date
  "outcome_details": "Dev dumped 80% in 2 hours"
}
```

**Data Sources:**
- RugDoc reports
- Solana FM rug database
- Community-reported scams
- Long-lived tokens (>6 months) as negatives

**Target:** 500 labeled examples (300 rugs, 200 legitimate)

**Use:**
- Confusion matrix (precision, recall, F1)
- ROC-AUC for sell probability model
- Calibration plots for Bayesian beliefs

**Effort:** 2 weeks (manual labeling + automation)

---

### 4. LP Interaction Tracking

**Goal:** Replace hardcoded lp_interaction_ratio with real data

**Implementation:**
```python
async def compute_lp_ratio(wallet: str, token_mint: str) -> float:
    """Calculate LP interaction ratio from transaction history."""
    txs = await get_wallet_transactions(wallet, token_mint)

    lp_events = [tx for tx in txs if tx.type in ['ADD_LIQUIDITY', 'REMOVE_LIQUIDITY']]
    total_events = len(txs)

    return len(lp_events) / total_events if total_events > 0 else 0.0
```

**Data Sources:**
- Helius parsed transactions
- Known LP program IDs (Raydium, Orca, Meteora)

**Effort:** 2 days

---

### 5. Adaptive Archetype System

**Goal:** Detect when archetypes become stale and need refitting

**Approach:**
```
1. Track cluster stability over time
2. Monitor intra-cluster variance
3. Count "UNKNOWN" classifications
4. Trigger refit when thresholds exceeded
```

**Metrics:**
- Silhouette score trend
- Cluster membership churn rate
- % wallets in UNKNOWN category

**Threshold:** Refit if silhouette drops >10% or UNKNOWN >15%

**Effort:** 2 weeks

---

### 6. Auto-Bayesian Evidence

**Goal:** Automatically generate evidence for belief updates

**Evidence Sources:**

| Source | Evidence Type | Trigger |
|--------|--------------|---------|
| Metric z-score > 2 | METRIC | Any metric exceeds baseline |
| Regime transition | PATTERN | HMM state change |
| Anomaly spike | ANOMALY | Isolation Forest score < -0.5 |
| Coordination detected | COORDINATION | Shared funder count > 3 |
| Exchange-linked wallet | EXCHANGE_LINKED | Known CEX address |

**Implementation:**
```python
def generate_evidence(analysis_result: AnalysisResult) -> list[Evidence]:
    evidence = []

    for metric, value in analysis_result.metrics.items():
        if abs(value.z_score) > 2:
            evidence.append(Evidence(
                type=EvidenceType.METRIC,
                value=sigmoid(value.z_score),
                weight=0.3
            ))

    if analysis_result.regime.is_transition:
        evidence.append(Evidence(
            type=EvidenceType.PATTERN,
            value=0.7 if analysis_result.regime.is_risky else 0.3,
            weight=0.4
        ))

    return evidence
```

**Effort:** 1 week

---

### 7. Streaming Response API

**Goal:** Provide progressive results for better UX

**Implementation:**
```python
@app.post("/api/v1/analyze/{mint}/stream")
async def analyze_stream(mint: str):
    async def generate():
        # Step 1: Basic metrics (fast)
        yield json.dumps({"stage": "metrics", "data": basic_metrics})

        # Step 2: Holder list (medium)
        yield json.dumps({"stage": "holders", "data": top_holders})

        # Step 3: Risk scores (slow)
        yield json.dumps({"stage": "risk", "data": risk_scores})

        # Step 4: Predictions (slowest)
        yield json.dumps({"stage": "predictions", "data": predictions})

        # Complete
        yield json.dumps({"stage": "complete"})

    return StreamingResponse(generate(), media_type="application/x-ndjson")
```

**Frontend:**
- Show results as they arrive
- Progress indicator
- Graceful handling of incomplete data

**Effort:** 1 week

---

## Prioritized Improvement List

| Rank | Improvement | Effort | Impact | Priority |
|------|-------------|--------|--------|----------|
| 1 | Price Integration (Sprint 7) | 2 weeks | HIGH | P0 |
| 2 | Web Dashboard MVP | 4 weeks | HIGH | P0 |
| 3 | Validation Dataset | 2 weeks | HIGH | P0 |
| 4 | Quick Analysis Mode | 3 days | HIGH | P1 |
| 5 | Cross-Token Intel (Sprint 8-11) | 8 weeks | HIGH | P1 |
| 6 | API Documentation | 1 week | MEDIUM | P1 |
| 7 | LP Interaction Tracking | 2 days | MEDIUM | P1 |
| 8 | Auto-Bayesian Evidence | 1 week | MEDIUM | P2 |
| 9 | Adaptive Archetypes | 2 weeks | MEDIUM | P2 |
| 10 | Streaming Responses | 1 week | MEDIUM | P2 |
| 11 | Discord Bot | 2 weeks | MEDIUM | P2 |
| 12 | Mobile App | 8 weeks | HIGH | P3 |

---

## ROI Analysis

| Improvement | Dev Cost | User Impact | Revenue Potential |
|-------------|----------|-------------|-------------------|
| Web Dashboard | 4 weeks | 10x reach | HIGH (enables premium tier) |
| Price Integration | 2 weeks | 2x signal quality | MEDIUM |
| Validation Dataset | 2 weeks | Credibility | MEDIUM (case studies) |
| Mobile App | 8 weeks | 5x reach | HIGH (app store presence) |

**Best ROI:** Web Dashboard - unlocks market reach and premium monetization

---

## Conclusion

SHI has **clear improvement paths** with **well-understood ROI**. The planned sprints (7-11) address technical gaps; new initiatives should focus on **accessibility and validation**.

**Immediate actions:**
1. Execute Sprint 7 (price integration)
2. Start Web Dashboard MVP in parallel
3. Begin building validation dataset

The foundation is strong; these improvements will unlock the system's full potential.
