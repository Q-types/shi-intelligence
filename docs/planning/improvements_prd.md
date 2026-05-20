# SPARK Mission: SHI Intelligence Enhancement

> **Mission ID**: shi-dynamical-intelligence-v2
> **Domain**: datascience
> **Priority**: high
> **Estimated Sprints**: 4

---

## Mission Objective

Transform SHI from **static snapshot analysis** to **dynamical intelligence system** that:

1. Tracks holder structure evolution over time
2. Provides actionable analytics for user decision-making
3. Profiles and classifies wallet risk/behavior types
4. Monitors and alerts on significant wallet movements
5. Explains predictions with interpretable outputs

---

## Project Context

**Working Directory**: `/Users/q/PythonScript/Python/Vibe/SHI`

**Current State**: SHI v1 provides structural snapshot analysis with:
- Distribution metrics (HHI, Gini, Entropy)
- Wallet archetypes (Snipers, Accumulators, Coordinated, etc.)
- Sell hazard modeling (Cox PH)
- Telegram bot interface

**Target State**: SHI v2 adds temporal dynamics, graph intelligence, and real-time monitoring.

---

## Required Skills

Load these specialist skills for the mission:

```yaml
data_science_core:
  - time-series-specialist      # Regime modeling, trajectory analysis
  - supervised-ml-specialist    # Risk scoring, classification
  - unsupervised-ml-specialist  # Clustering, dimensionality reduction
  - anomaly-detection-specialist # Suspicious wallet detection
  - graph-ml-specialist         # Node2Vec, network dynamics
  - survival-analysis-specialist # Cox PH, sell prediction
  - ml-explainability-specialist # SHAP, feature attribution
  - nlp-specialist              # Sentiment velocity (optional)

infrastructure:
  - supabase-backend            # Real-time subscriptions, alerts
  - postgres-wizard             # Time-series storage, analytics
  - api-design                  # REST/WebSocket endpoints
  - python-patterns             # Async processing, queues
```

---

## Agent Structure

### Team 1: Analytics Engine (Backend)

**Supervisor**: Analytics Architect
**Agents**:
- **Metrics Evolution Agent**: Implement time-series tracking of all core metrics
- **Graph Intelligence Agent**: Add Node2Vec embeddings, dynamic network analysis
- **Anomaly Detection Agent**: Wallet-level anomaly scoring with Isolation Forest
- **Explainability Agent**: SHAP integration for all predictions

### Team 2: Monitoring & Alerts (Real-time)

**Supervisor**: Real-time Systems Lead
**Agents**:
- **Wallet Watcher Agent**: Track large wallet movements in real-time
- **Alert Engine Agent**: Configurable notification system
- **Profile Tracker Agent**: Maintain evolving wallet profiles over time

### Team 3: User Intelligence (Frontend)

**Supervisor**: Product Intelligence Lead
**Agents**:
- **Dashboard Agent**: Visual analytics for decision-making
- **Risk Explainer Agent**: Human-readable risk explanations
- **Report Generator Agent**: PDF/Telegram summaries

---

## Sprint Breakdown

### Sprint 1: Temporal Foundation

**Goal**: Add time-series infrastructure and metric trajectories

**Tasks**:

1. **Schema Evolution** (database team)
   ```sql
   -- Add temporal tracking tables
   CREATE TABLE metric_snapshots (
     token_mint TEXT,
     timestamp TIMESTAMPTZ,
     hhi FLOAT,
     gini FLOAT,
     entropy FLOAT,
     whale_dominance FLOAT,
     churn_rate FLOAT,
     coordination_score FLOAT
   );

   CREATE TABLE wallet_profiles (
     wallet_address TEXT PRIMARY KEY,
     archetype TEXT,
     risk_score FLOAT,
     anomaly_score FLOAT,
     first_seen TIMESTAMPTZ,
     last_updated TIMESTAMPTZ,
     profile_history JSONB
   );
   ```

2. **Metric Trajectory Engine** (time-series-specialist)
   - Implement `HHI(t)`, `Gini(t)`, `Churn(t)` tracking
   - Calculate derivatives: `dHHI/dt`, `dGini/dt`
   - Add trend detection (centralizing vs decentralizing)
   - Walk-forward validation for all temporal models

3. **Regime State Machine** (time-series-specialist)
   ```python
   class HolderRegime(Enum):
       ACCUMULATION = "accumulation"
       DISTRIBUTION = "distribution"
       COORDINATED_ACCUMULATION = "coordinated_accumulation"
       DECAY = "decay"
       STABLE = "stable"
   ```
   - Hidden Markov Model for regime detection
   - Regime transition probabilities
   - Alert on regime changes

**Acceptance Criteria**:
- [ ] Metrics tracked hourly for all analyzed tokens
- [ ] Derivative calculations validated against manual checks
- [ ] Regime detection accuracy > 70% on labeled test set

---

### Sprint 2: Graph Intelligence

**Goal**: Add graph embeddings and dynamic network analysis

**Tasks**:

1. **Node2Vec Integration** (graph-ml-specialist)
   ```python
   from node2vec import Node2Vec

   def embed_funding_graph(G: nx.DiGraph, dimensions=64):
       """Embed wallets into latent space for similarity detection."""
       node2vec = Node2Vec(G, dimensions=dimensions, walk_length=30, num_walks=200)
       model = node2vec.fit(window=10, min_count=1)
       return {node: model.wv[node] for node in G.nodes()}
   ```

2. **Wallet Similarity Detection** (graph-ml-specialist)
   - Cosine similarity on embeddings
   - Hidden coordination cluster discovery
   - Sybil detection via structural similarity

3. **Dynamic Network Metrics** (graph-ml-specialist)
   - Track `modularity(t)`, `density(t)`, `centralization(t)`
   - Detect community emergence/fragmentation
   - Graph evolution velocity

4. **Anomaly Scoring** (anomaly-detection-specialist)
   ```python
   from sklearn.ensemble import IsolationForest

   def score_wallet_anomaly(wallet_features: np.ndarray) -> float:
       """Return anomaly score [-1, 1] where -1 is most anomalous."""
       model = IsolationForest(contamination=0.05, random_state=42)
       return model.fit_predict(wallet_features)
   ```

**Acceptance Criteria**:
- [ ] Node2Vec embeddings generated for all funding graphs
- [ ] Similarity threshold tuned on known Sybil examples
- [ ] Anomaly scoring integrated into wallet profiles

---

### Sprint 3: Real-time Monitoring

**Goal**: Large wallet tracking, notifications, profile updates

**Tasks**:

1. **Wallet Watcher Service** (supabase-backend)
   ```python
   class WalletWatcher:
       def __init__(self, config: WatcherConfig):
           self.thresholds = config.thresholds
           self.subscribers = []

       async def on_transaction(self, tx: Transaction):
           if self.is_significant(tx):
               await self.notify_subscribers(tx)

       def is_significant(self, tx: Transaction) -> bool:
           return (
               tx.amount_usd > self.thresholds.large_movement or
               tx.wallet in self.thresholds.watched_wallets or
               tx.causes_concentration_spike()
           )
   ```

2. **Alert Engine** (api-design)
   ```yaml
   alert_types:
     - whale_movement:
         trigger: "balance_change > 5% of supply"
         channels: [telegram, webhook]

     - regime_change:
         trigger: "regime != previous_regime"
         channels: [telegram]

     - anomaly_spike:
         trigger: "anomaly_score < -0.8"
         channels: [telegram, webhook]

     - concentration_increase:
         trigger: "dHHI/dt > threshold"
         channels: [telegram]
   ```

3. **Profile Evolution Tracker**
   - Update wallet profiles on each significant event
   - Track archetype transitions over time
   - Maintain risk score history

4. **Telegram Notification Commands**
   ```
   /watch <wallet> - Add wallet to watchlist
   /unwatch <wallet> - Remove from watchlist
   /alerts <token> - Configure token alerts
   /profile <wallet> - View wallet profile evolution
   ```

**Acceptance Criteria**:
- [ ] Webhook notifications delivered < 30s from on-chain event
- [ ] Profile updates persisted with full history
- [ ] Alert configuration stored per-user

---

### Sprint 4: User Intelligence & Explainability

**Goal**: Actionable analytics, risk explanations, decision support

**Tasks**:

1. **SHAP Integration** (ml-explainability-specialist)
   ```python
   import shap

   def explain_risk_score(wallet: str, model, features: pd.DataFrame) -> dict:
       """Return feature contributions to risk score."""
       explainer = shap.TreeExplainer(model)
       shap_values = explainer.shap_values(features)

       return {
           "risk_score": model.predict_proba(features)[0, 1],
           "top_contributors": get_top_contributors(shap_values, features),
           "explanation": generate_natural_language(shap_values)
       }
   ```

2. **Risk Dashboard Data** (api-design)
   ```json
   {
     "token_intelligence": {
       "current_regime": "distribution",
       "regime_confidence": 0.82,
       "trend": "centralizing",
       "trend_velocity": 0.03,
       "risk_level": "elevated",
       "risk_factors": [
         {"factor": "whale_accumulation", "contribution": 0.35},
         {"factor": "coordination_detected", "contribution": 0.28},
         {"factor": "high_churn", "contribution": 0.22}
       ]
     },
     "actionable_insights": [
       "Top 3 wallets accumulated 8% more supply in 24h",
       "2 coordinated clusters detected (high Sybil probability)",
       "Sell pressure index increased 40% - monitor closely"
     ]
   }
   ```

3. **Natural Language Explanations**
   ```python
   def explain_token_risk(analysis: TokenAnalysis) -> str:
       """Generate human-readable risk explanation."""
       return f"""
       **Risk Assessment: {analysis.risk_level.upper()}**

       This token is currently in a **{analysis.regime}** regime.

       Key concerns:
       {bullet_points(analysis.risk_factors)}

       The holder structure is **{analysis.trend}** at a rate of
       {analysis.trend_velocity:.1%} per day.

       **Recommendation**: {analysis.recommendation}
       """
   ```

4. **Capital Flow Forecasting** (time-series-specialist)
   ```python
   def forecast_capital_pressure(token: str, horizon_hours: int = 24) -> dict:
       """Predict net capital flow direction."""
       features = extract_flow_features(token)

       return {
           "predicted_net_flow": model.predict(features),
           "confidence_interval": calculate_ci(features),
           "liquidity_stress_probability": stress_model.predict_proba(features)
       }
   ```

**Acceptance Criteria**:
- [ ] SHAP explanations for all risk scores
- [ ] Natural language summaries pass human readability test
- [ ] Capital flow forecasts backtested with MAPE < 20%

---

## Quality Gates

### Per-Task Gates
- [ ] All new code has unit tests (coverage > 80%)
- [ ] No new critical lint errors
- [ ] Type hints on all public functions
- [ ] Docstrings following NumPy style

### Per-Sprint Gates
- [ ] Integration tests pass
- [ ] Performance benchmarks met (response < 30s)
- [ ] Backtesting on reference dataset passes
- [ ] Code review approved

### Mission Completion Gates
- [ ] All 4 sprints completed
- [ ] End-to-end test suite passes
- [ ] Documentation updated
- [ ] Telegram bot commands functional
- [ ] Alert system operational

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Node2Vec memory usage on large graphs | Batch processing, edge sampling |
| Real-time latency for alerts | Pre-compute watched wallet lists |
| Model drift in regime detection | Weekly recalibration, monitoring |
| SHAP computation cost | Cache explanations, async generation |
| False positive alerts | Configurable thresholds, cooldown periods |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Regime detection accuracy | > 75% | Labeled test set |
| Sybil detection recall | > 80% | Known Sybil examples |
| Alert latency | < 30s | End-to-end timing |
| User engagement | +50% | Telegram command frequency |
| Explanation quality | > 4/5 | User feedback rating |

---

## File Structure (New/Modified)

```
src/
├── temporal/
│   ├── __init__.py
│   ├── trajectories.py      # Metric time-series
│   ├── regimes.py           # HMM regime detection
│   └── forecasting.py       # Capital flow prediction
├── graph/
│   ├── embeddings.py        # Node2Vec integration
│   ├── dynamics.py          # Network evolution
│   └── similarity.py        # Wallet similarity
├── monitoring/
│   ├── watcher.py           # Wallet watcher service
│   ├── alerts.py            # Alert engine
│   └── profiles.py          # Profile tracker
├── explainability/
│   ├── shap_explainer.py    # SHAP integration
│   ├── narratives.py        # Natural language
│   └── dashboard_data.py    # API responses
└── telegram/
    ├── commands/
    │   ├── watch.py         # /watch, /unwatch
    │   ├── alerts.py        # /alerts config
    │   └── profile.py       # /profile command
    └── notifications.py     # Push notifications
```

---

## Launch Command

```bash
# Start the mission
spark-researcher mission start \
  --manifest /Users/q/PythonScript/Python/Vibe/SHI/MISSION_SHI_ENHANCEMENT.md \
  --domain datascience \
  --skills time-series-specialist,graph-ml-specialist,anomaly-detection-specialist,survival-analysis-specialist,ml-explainability-specialist \
  --parallel-agents 3

# Or via Telegram
/mission start shi-dynamical-intelligence-v2
```

---

## Notes for Agents

1. **Metrics are IMMUTABLE** - Do not modify formulas in PDR
2. **Uncertainty-aware outputs** - All predictions include confidence bounds
3. **Explainability first** - Every score must be explainable
4. **Real-time constraints** - Design for < 30s response times
5. **Incremental delivery** - Each sprint produces working functionality

---

*Mission designed for SPARK with domain-chip-datascience integration*
*Skills: 8 DS specialists + infrastructure stack*
*Target: Transform static analysis → dynamical intelligence*
