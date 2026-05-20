# SHI Audit: Capabilities Assessment

**Date:** 2026-05-20

---

## Module Inventory

SHI comprises **29 source modules** organized into functional layers:

| Layer | Modules | Purpose |
|-------|---------|---------|
| **Core** | types, config | Type system, configuration |
| **Data** | client, cache, providers | Solana RPC integration |
| **Metrics** | distribution, coordination, hazard, normalization | FROZEN metric formulas |
| **Clustering** | archetypes | HDBSCAN wallet classification |
| **Graph** | funding_graph, embeddings, anomaly, similarity, dynamics | Network analysis |
| **Temporal** | trajectories, regimes, forecasting | Time-series intelligence |
| **Models** | hazard_model, sell_events, regime, correlation, training, validation | ML models |
| **Risk** | scoring | Composite risk aggregation |
| **Bayesian** | priors, updater, risk_belief | Uncertainty quantification |
| **Liquidity** | pools, impact | DEX integration |
| **Explainability** | shap_explainer, narratives, forecasting, dashboard_data | Interpretability |
| **Monitoring** | watcher, alerts, profiles, notifications | Real-time tracking |
| **Telegram** | bot, commands/*, security, formatters, notifications | User interface |
| **API** | routes, schemas, websocket, dependencies | REST/WebSocket API |
| **Infrastructure** | cache, rate_limit, retry, circuit_breaker, monitoring | Reliability |
| **Pipeline** | metrics_pipeline, baseline | Orchestration |

---

## Core Capabilities

### 1. Distribution Metrics (FROZEN per PDR §4)

| Metric | Formula | Purpose |
|--------|---------|---------|
| **HHI** | Σ(s_i²) | Concentration index [0,1] |
| **Gini** | Σ\|b_i - b_j\| / (2N·Σb_i) | Inequality coefficient |
| **Shannon Entropy** | -Σ(s_i · ln(s_i)) | Holder diversity |
| **Whale Dominance Ratio** | Σ(top-k shares) | Top-10 concentration |
| **Churn Rate** | Exited / Starting | Holder turnover |
| **Coordination Score** | Shared_Funders / Cluster_Size | Sybil indicator |

All metrics are **version-controlled** and **baseline-normalized** (z-scores + percentiles).

### 2. Wallet Archetypes (6 Types)

| Archetype | Characteristics | Risk Signal |
|-----------|----------------|-------------|
| **Sniper** | High trade count, short holding | Dump risk |
| **Long-Term Accumulator** | Steady position, low volatility | Stability signal |
| **Coordinated Cluster** | Shared funders, synchronized trades | Sybil/manipulation |
| **Liquidity Actor** | LP interactions, DEX swaps | Market maker |
| **Exchange-Linked** | CEX patterns | Potential exit route |
| **Dormant Whale** | Inactive, large historical position | Sleeping risk |

Classification uses **HDBSCAN** with 14+ feature dimensions.

### 3. Survival Analysis (Cox PH)

**Purpose:** Predict 7-day sell probability per wallet

**Model:**
- λ(t|x) = λ₀(t) × exp(β'x)
- P_sell = 1 - exp(-Λ₀(T) × exp(β'x))

**Features:**
- Balance, volatility, entry time
- Holding duration, burstiness
- Funding graph metrics (in/out degree, centrality)

**Validation:**
- Concordance index (discrimination)
- Proportional hazards test
- Schoenfeld residuals

### 4. Regime Detection (HMM)

**5 Hidden States:**
1. ACCUMULATION - HHI/Gini decreasing
2. DISTRIBUTION - Concentration increasing
3. COORDINATED_ACCUMULATION - Centralization + coordination
4. DECAY - High churn, exits
5. STABLE - Low velocity

**Algorithm:** Gaussian HMM with Viterbi decoding
**Input:** [dHHI/dt, dGini/dt, dChurn/dt, coordination_signal]

### 5. Graph Intelligence

| Capability | Implementation | Output |
|------------|---------------|--------|
| Funding Graph | NetworkX DiGraph | V=wallets, E=SOL transfers |
| Node Embeddings | Node2Vec (128-d) | Structural position vectors |
| Anomaly Detection | Isolation Forest | Sybil/unusual wallet scores |
| Community Detection | HDBSCAN on embeddings | Wallet clusters |
| Ancestor Traversal | BFS (max_depth=3) | Upstream funder tree |

### 6. Real-Time Monitoring

| Feature | Implementation | SLA |
|---------|---------------|-----|
| Wallet Watching | Async polling (30s) | <30s alert delivery |
| Balance Change Detection | Delta threshold (5% supply) | Real-time |
| Alert Types | WHALE_MOVEMENT, REGIME_CHANGE, ANOMALY_SPIKE, CONCENTRATION_INCREASE | Per-type cooldowns |
| Delivery | Telegram push, WebSocket broadcast | <30s |

### 7. Bayesian Uncertainty

**Prior Distributions:** Beta(α, β) for each risk factor
**Evidence Types:** METRIC, PATTERN, EXCHANGE_LINKED, COORDINATION, ANOMALY
**Update:** Posterior = Prior × Likelihood^weight
**Output:** Mean, credible interval, standard deviation

---

## API Surface

### REST Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/analyze/{mint}` | Full token analysis |
| GET | `/api/v1/metrics/{mint}` | Distribution metrics |
| GET | `/api/v1/risk/{mint}` | Risk scores |
| GET | `/api/v1/wallet/{address}/{mint}` | Wallet profile |
| POST | `/api/v1/forecast/{mint}` | Capital flow forecast |
| WebSocket | `/api/v1/ws` | Real-time alerts |

### Telegram Commands (12)

`/analyze`, `/summary`, `/top_holders`, `/risk`, `/graph`, `/history`, `/watch`, `/alerts`, `/profile`, `/explain`, `/forecast`

---

## Performance Characteristics

| Metric | Target | Notes |
|--------|--------|-------|
| Response Time | <30s | For tokens <10k holders |
| Alert Latency | <30s | From event to delivery |
| Cache Freshness | <5 min | Holder snapshots |
| RPC Budget | 10k/hour | Helius rate limit |
| Max Holders | 50k | Per-token limit |

---

## Capability Gaps

1. **No price data integration** - Purely behavioral analysis
2. **Single-token focus** - No cross-portfolio intelligence
3. **LP ratio hardcoded** - Missing actual LP interaction data
4. **Fixed archetypes** - Cannot adapt to new patterns
5. **Manual Bayesian updates** - No automatic evidence injection
