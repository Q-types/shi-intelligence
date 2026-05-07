# PRD — Product Requirements Document

**Product:** Solana Holder Intelligence (SHI)

---

## 1. Objective

Deliver a Telegram-native bot that analyzes Solana token holders and returns:

- Distribution statistics
- Behavioral segmentation
- Coordination detection
- Probabilistic sell-risk modeling
- Stability risk indicators

---

## 2. Users

### Primary

- Token teams
- Risk analysts
- On-chain researchers
- Market makers

### Secondary

- Advanced retail users

---

## 3. Functional Requirements

| # | Requirement |
|---|-------------|
| 1 | Token ingestion via mint address |
| 2 | Holder retrieval |
| 3 | Funding graph construction |
| 4 | Wallet feature computation |
| 5 | Archetype clustering |
| 6 | Hazard modeling |
| 7 | Risk scoring |
| 8 | Telegram bot integration |
| 9 | JSON + human-readable outputs |
| 10 | Historical state storage for comparisons |

---

## 4. Non-Functional Requirements

- **Deterministic metric layer** — reproducible calculations
- **Probabilistic modeling layer** — uncertainty-aware outputs
- **Modular MCP-based orchestration** — agent-driven architecture
- **Scalable** — handles large holder sets
- **Response time** — under 30 seconds for typical tokens
- **Uncertainty-aware** — all predictions include confidence bounds

---

## 5. Metrics Lock

> **CRITICAL: The metrics defined in the PDR are final.**
> **No agent may alter formulas, definitions, or thresholds without explicit human approval.**

### Locked Metrics

- Herfindahl–Hirschman Index (HHI)
- Shannon Entropy (H)
- Gini Coefficient (G)
- Coordination Score
- Churn Rate
- Whale Dominance Ratio (WDR)
- Sell Hazard Model (Cox proportional hazards)

### Locked Archetypes

- Snipers
- Long-Term Accumulators
- Coordinated Cluster
- Liquidity Actors
- Exchange-Linked
- Dormant Whales

---

## 6. Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Accurate holder graph reconstruction | Validated against known wallet relationships |
| Meaningful clustering separation | Silhouette score, cluster stability |
| Stable hazard predictions | Consistency across retraining windows |
| Telegram usage adoption | Active users, command frequency |
| Differentiation accuracy | Backtesting: stable vs rug tokens |

---

## 7. Telegram Integration Specification

### Supported Commands

| Command | Description |
|---------|-------------|
| `/analyze <mint>` | Full token analysis |
| `/summary <mint>` | Quick overview |
| `/top_holders <mint>` | Top holder breakdown |
| `/risk <mint>` | Risk scores only |
| `/graph <mint>` | Funding graph visualization link |
| `/history <mint>` | Historical comparison |

### Output Format

- **Short summary** — Telegram-friendly, mobile-optimized
- **Expandable detail** — via inline buttons
- **Optional web dashboard link** — for deep dives
- **PDF export** — for reports

### Performance Target

**Latency:** Under 30 seconds for average token (< 10,000 holders)

---

## 8. Technical Architecture

### Data Sources

- Solana RPC (holder accounts, transactions)
- Token metadata APIs
- Historical balance snapshots

### Storage

- PostgreSQL (relational data, metrics history)
- Graph database (funding topology)
- Redis (caching, rate limiting)

### Processing

- Python backend (metrics, ML)
- HDBSCAN for clustering
- Lifelines for survival analysis
- NetworkX / igraph for graph analysis

### Interface

- python-telegram-bot
- Async processing queue
- Rate limiting per user

---

## 9. Phases

| Phase | Deliverable |
|-------|-------------|
| **Phase 1** | Architecture design, storage schema, ingestion prototype |
| **Phase 2** | Metrics layer implementation (per PDR), test token validation |
| **Phase 3** | Clustering + hazard model, backtesting on reference dataset |
| **Phase 4** | Telegram interface, test bot deployment |
| **Phase 5** | Iterative refinement, stress testing, baseline calibration |

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| RPC rate limits | Multiple providers, caching |
| Large holder sets causing timeouts | Chunked processing, background jobs |
| Model drift over time | Periodic recalibration, z-score normalization |
| Sybil evasion techniques | Continuous research, model updates |

---

## 11. Out of Scope

- Trading signals
- Price predictions
- Buy/sell recommendations
- Identity doxing
- Real-time alerting (v1)
