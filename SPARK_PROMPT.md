# SPARK Session Prompt for SHI Enhancement

Copy this prompt to start a SPARK mission session:

---

## Quick Start Prompt

```
I'm working on the Solana Holder Intelligence (SHI) project at /Users/q/PythonScript/Python/Vibe/SHI

Load these skills first:
- time-series-specialist
- graph-ml-specialist
- anomaly-detection-specialist
- survival-analysis-specialist
- ml-explainability-specialist
- supervised-ml-specialist
- unsupervised-ml-specialist

Read the mission manifest: /Users/q/PythonScript/Python/Vibe/SHI/MISSION_SHI_ENHANCEMENT.md
Read the enhancement ideas: /Users/q/PythonScript/Python/Vibe/SHI/ideas.md
Read the current PRD: /Users/q/PythonScript/Python/Vibe/SHI/PRD.md

Transform SHI from static snapshot analysis to dynamical intelligence:

CORE DELIVERABLES:
1. Time-series regime modeling - Track HHI(t), Gini(t) over time, detect regime changes (accumulation/distribution/decay)
2. Graph embeddings - Node2Vec for wallet similarity, Sybil detection, hidden coordination
3. Real-time monitoring - Watch large wallets, send alerts on significant movements
4. Wallet profiling - Track and update wallet risk profiles over time
5. Explainable analytics - SHAP explanations, natural language risk summaries

USER-FACING FEATURES:
- /watch <wallet> - Track specific wallets
- /alerts <token> - Configure notification thresholds
- /profile <wallet> - View wallet behavior history
- /regime <token> - Current regime state and trajectory
- Real-time Telegram notifications for whale movements

Start with Sprint 1: Temporal Foundation
- Add metric_snapshots table for time-series storage
- Implement trajectory tracking (derivatives of core metrics)
- Build regime state machine with HMM

Act as a coordinated team of data science agents. Each agent should use their specialist skill to contribute. Produce working code, not just plans.
```

---

## Alternative: Focused Single-Feature Prompts

### For Time-Series Only:
```
Load time-series-specialist skill.
Working on /Users/q/PythonScript/Python/Vibe/SHI

Add temporal evolution tracking to SHI:
1. Track HHI(t), Gini(t), Churn(t) over time
2. Calculate derivatives dHHI/dt to detect centralizing vs decentralizing trends
3. Implement regime detection: accumulation, distribution, coordinated_accumulation, decay
4. Use Hidden Markov Model for regime state transitions

The insight: "Two tokens may have identical HHI, but one is decentralizing, the other centralizing. Those are opposite risk profiles."

Start by reading src/metrics/ to understand current implementation.
```

### For Graph Intelligence Only:
```
Load graph-ml-specialist skill.
Working on /Users/q/PythonScript/Python/Vibe/SHI

Add graph embeddings to enhance Sybil detection:
1. Integrate Node2Vec for wallet embeddings (64 dimensions)
2. Detect wallet similarity via cosine distance in embedding space
3. Find hidden coordination - wallets that never interact but occupy similar structural positions
4. Track dynamic network metrics: modularity(t), density(t)

The insight: "Two wallets may never directly interact, but occupy similar structural positions. Embeddings detect this."

Start by reading src/graph/ to understand current implementation.
```

### For Real-Time Alerts Only:
```
Load supabase-backend and api-design skills.
Working on /Users/q/PythonScript/Python/Vibe/SHI

Build wallet monitoring and alert system:
1. WalletWatcher service - detect significant transactions
2. Alert engine with configurable triggers:
   - whale_movement: balance_change > 5% of supply
   - regime_change: regime != previous_regime
   - anomaly_spike: anomaly_score < -0.8
3. Telegram notifications via /watch and /alerts commands
4. Webhook support for external integrations

Profile tracking: maintain wallet history, archetype transitions, risk score evolution.

Start by reading src/telegram/ and src/monitoring/.
```

---

## Spawner UI Workflow

If using Spawner UI:

1. Open http://127.0.0.1:5173
2. Search skills: "time-series", "graph-ml", "anomaly"
3. Create agent chain:
   ```
   time-series-specialist → graph-ml-specialist → anomaly-detection-specialist → ml-explainability-specialist
   ```
4. Upload MISSION_SHI_ENHANCEMENT.md as context
5. Run orchestration

---

## Expected Agent Behavior

The agents should:

1. **Research first** - Read existing code in src/ before proposing changes
2. **Preserve immutable metrics** - PDR metrics are locked, don't modify formulas
3. **Add incrementally** - New modules in src/temporal/, src/monitoring/, etc.
4. **Test everything** - Add tests in tests/ for new functionality
5. **Explain outputs** - All predictions must include uncertainty and attribution

---

## Validation Checkpoints

After each sprint, verify:

- [ ] New tables created and migrated
- [ ] Core functions have unit tests
- [ ] Telegram commands respond correctly
- [ ] Performance < 30s for typical tokens
- [ ] SHAP explanations generated for risk scores
