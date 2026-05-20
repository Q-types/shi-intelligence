# SHI - Solana Holder Intelligence

**Probabilistic on-chain intelligence for Solana token risk analysis**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is SHI?

SHI is a **production-grade data science system** that analyzes Solana token holder distributions to predict token stability and sell-risk. It combines:

- **Survival Analysis** (Cox Proportional Hazards) for sell probability prediction
- **Graph Intelligence** (funding networks, Node2Vec embeddings) for sybil detection
- **Hidden Markov Models** for regime detection (accumulation vs distribution phases)
- **HDBSCAN Clustering** for behavioral archetype classification
- **Bayesian Inference** for uncertainty-quantified risk scoring

Unlike simple holder trackers, SHI answers: *"What is the **probability** this token experiences significant sell pressure in the next 7 days?"*

---

## Key Features

| Feature | Description |
|---------|-------------|
| **6 Behavioral Archetypes** | Classify wallets as Sniper, Accumulator, Coordinated Cluster, Liquidity Actor, Exchange-Linked, or Dormant Whale |
| **7-Day Sell Probability** | Cox PH survival model predicts per-wallet and aggregate sell risk |
| **Regime Detection** | HMM identifies market phases: Accumulation, Distribution, Coordinated Accumulation, Decay, Stable |
| **Sybil Detection** | Graph-based coordination scoring using funding network analysis |
| **Real-Time Monitoring** | Async wallet watcher with configurable alerts (<30s latency) |
| **Explainable AI** | SHAP values explain why a token scores high-risk |
| **Calibrated Probabilities** | Isotonic regression ensures predicted probabilities match observed frequencies |
| **Multi-Interface** | Telegram bot, REST API, WebSocket subscriptions, Streamlit dashboard |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                               │
├─────────────────┬─────────────────┬─────────────────┬───────────────┤
│  Telegram Bot   │    REST API     │   WebSocket     │   Streamlit   │
│  14+ commands   │   9+ endpoints  │  Real-time      │   Dashboard   │
└────────┬────────┴────────┬────────┴────────┬────────┴───────┬───────┘
         │                 │                 │                │
         └─────────────────┼─────────────────┼────────────────┘
                           │                 │
┌──────────────────────────▼─────────────────▼────────────────────────┐
│                      ORCHESTRATION LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Pipeline   │  │   Feature   │  │   Metrics   │  │  Baseline   │ │
│  │Orchestrator │  │  Engineer   │  │  Pipeline   │  │ Calibration │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                       INTELLIGENCE LAYER                             │
├─────────────────┬─────────────────┬─────────────────┬───────────────┤
│   Clustering    │   Survival      │    Temporal     │    Graph      │
│   (HDBSCAN)     │   (Cox PH)      │    (HMM)        │  (Node2Vec)   │
│                 │                 │                 │               │
│ 6 Archetypes    │ 7-day P(sell)   │ 5 Regimes       │ Funding Net   │
│ Multi-score     │ Calibrated      │ Viterbi decode  │ Sybil detect  │
└─────────────────┴─────────────────┴─────────────────┴───────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                        METRICS LAYER (FROZEN)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │     HHI     │  │    Gini     │  │   Entropy   │  │ Coordination│ │
│  │ Σ(s_i²)     │  │ Inequality  │  │ -Σ(p·ln p)  │  │   Score     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│                        All formulas are auditable                    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                         DATA LAYER                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Solana RPC  │  │   Helius    │  │  Jupiter    │  │   Redis     │ │
│  │  (Holders)  │  │ (Enriched)  │  │  (Prices)   │  │  (Cache)    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Science Methodology

### 1. Survival Analysis (Cox Proportional Hazards)

Predicts the probability of a "sell event" (≥50% balance reduction) within T days:

```
λ(t|X) = λ₀(t) × exp(β'X)
P(sell in T days) = 1 - exp(-Λ₀(T) × exp(β'X))
```

**Features:** holding duration, balance volatility, entry timing, burstiness, graph centrality

**Validation:** C-index ~0.88, calibrated slope ~1.15 (post-calibration)

### 2. Behavioral Clustering (HDBSCAN)

Unsupervised clustering with 14+ engineered features assigns wallets to 6 archetypes:

| Archetype | Signature | Risk Signal |
|-----------|-----------|-------------|
| **SNIPER** | Early entry + short hold + high turnover | High dump risk |
| **LONG_TERM_ACCUMULATOR** | Gradual growth + low churn | Stability signal |
| **COORDINATED_CLUSTER** | Shared funders + temporal sync | Sybil/manipulation |
| **LIQUIDITY_ACTOR** | LP interactions + DEX swaps | Market maker |
| **EXCHANGE_LINKED** | CEX/bridge behavior | Exit route |
| **DORMANT_WHALE** | Large + inactive | Sleeping risk |

### 3. Regime Detection (Hidden Markov Model)

5-state Gaussian HMM captures holder behavior transitions:

```
Input: [dHHI/dt, dGini/dt, dChurn/dt, coordination_signal]
States: ACCUMULATION | DISTRIBUTION | COORDINATED_ACCUMULATION | DECAY | STABLE
```

Viterbi decoding identifies the most likely current regime.

### 4. Graph Intelligence

- **Funding Graph:** NetworkX DiGraph of SOL transfers between wallets
- **Node2Vec Embeddings:** 128-dimensional structural position vectors
- **Sybil Detection:** Shared funder analysis + Isolation Forest anomaly scoring
- **Community Detection:** HDBSCAN on graph embeddings

### 5. Probability Calibration

Raw model probabilities are calibrated using isotonic regression to ensure:
- Predicted 30% ≈ Observed 30% sell rate
- Expected Calibration Error (ECE) < 5%
- Validated via probability bands and walk-forward testing

---

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL (optional, for persistence)
- Redis (optional, for caching)
- Solana RPC access (Helius recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/shi.git
cd shi

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Run Analysis

```python
import asyncio
from src.pipeline.orchestrator import AnalysisOrchestrator

async def analyze_token():
    orchestrator = AnalysisOrchestrator()
    result = await orchestrator.analyze(
        token_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
    )
    print(f"Risk Score: {result.risk_score:.2f}")
    print(f"Regime: {result.regime}")
    print(f"Top Archetype: {result.dominant_archetype}")

asyncio.run(analyze_token())
```

### Start API Server

```bash
uvicorn src.api:app --reload --port 8000
```

### Start Telegram Bot

```bash
python -m src.telegram.bot
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analyze/{mint}` | POST | Full token analysis |
| `/api/v1/metrics/{mint}` | GET | Distribution metrics (HHI, Gini, etc.) |
| `/api/v1/regime/{mint}` | GET | Current holder regime |
| `/api/v1/forecast/{mint}` | POST | 7/30-day capital flow forecast |
| `/api/v1/explain/{id}` | GET | SHAP explanations for prediction |
| `/api/v1/price/{mint}` | GET | Jupiter price data |
| `/api/v1/liquidity/{pool}` | GET | Raydium/Orca pool info |
| `/ws/subscribe/{type}` | WS | Real-time alert streaming |

---

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/analyze <mint>` | Full token intelligence report |
| `/summary <mint>` | Quick 30-second overview |
| `/risk <mint>` | Risk scores only |
| `/top_holders <mint>` | Top holder breakdown + archetypes |
| `/watch <wallet> <token>` | Add to real-time monitoring |
| `/alerts` | Configure alert thresholds |
| `/profile <wallet> <token>` | Wallet evolution history |

---

## Project Structure

```
SHI/
├── src/
│   ├── core/           # Configuration, types
│   ├── metrics/        # FROZEN metric formulas (HHI, Gini, etc.)
│   ├── clustering/     # HDBSCAN archetypes, transformations
│   ├── models/         # Cox PH hazard, HMM regime, calibration
│   ├── graph/          # Funding networks, Node2Vec, anomaly
│   ├── temporal/       # Trajectories, regime detection
│   ├── bayesian/       # Uncertainty quantification
│   ├── risk/           # Composite risk scoring
│   ├── explainability/ # SHAP, narratives
│   ├── monitoring/     # Real-time watcher, alerts
│   ├── pipeline/       # Orchestration
│   ├── api/            # FastAPI endpoints
│   ├── telegram/       # Bot commands
│   └── data/           # RPC clients, price providers
├── sweenee/            # Streamlit whale dashboard
├── tests/              # 24+ test modules, 267+ tests
├── docs/               # Architecture, validation reports
└── scripts/            # Audit and analysis scripts
```

---

## Validation & Quality

### Model Validation

| Metric | Value | Target |
|--------|-------|--------|
| Cox PH C-Index | 0.876 | >0.80 |
| Calibration Slope | 1.15 | ~1.0 |
| Expected Calibration Error | 4.2% | <5% |
| Cluster Silhouette | 0.42 | >0.30 |
| Bootstrap Stability (ARI) | 0.31 | >0.25 |

### Test Coverage

- **267+ tests** across 24 modules
- **Coverage:** >80% for core ML modules
- **Quality gates:** 100% test pass rate, 0 critical lint errors

### Validation Reports

- [Calibration Audit](docs/validation/CALIBRATION_AUDIT.md)
- [Cluster Semantics](docs/validation/CLUSTER_SEMANTICS_AUDIT.md)
- [Feature Ablation](docs/validation/FEATURE_ABLATION_RESULTS.md)
- [Hazard Model Comparison](docs/validation/HAZARD_MODEL_COMPARISON.md)

---

## Technology Stack

| Category | Technologies |
|----------|--------------|
| **ML/Statistics** | lifelines (Cox PH), hdbscan, hmmlearn, scikit-learn, node2vec |
| **Graphs** | NetworkX, python-igraph |
| **Data** | pandas, numpy, scipy, pydantic |
| **API** | FastAPI, WebSockets |
| **Bot** | python-telegram-bot |
| **Database** | PostgreSQL (asyncpg), Redis, SQLAlchemy |
| **Blockchain** | solana-py, solders, Helius RPC |
| **Monitoring** | structlog, prometheus-client |

---

## What Makes SHI Different?

1. **Probabilistic, Not Deterministic** - All predictions include confidence intervals
2. **Frozen Metric Layer** - Core formulas are auditable and version-controlled
3. **Graph-Native Sybil Detection** - Funding network analysis catches coordinated wallets
4. **Regime Intelligence** - HMM captures holder behavior transitions over time
5. **Explainable Predictions** - SHAP values show exactly why a token scores high-risk
6. **Calibrated Probabilities** - When we say 30% risk, we mean it
7. **Production-Ready** - Circuit breakers, rate limiting, retry logic, structured logging

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please read the [architecture documentation](docs/ARCHITECTURE.md) first.

---

## Acknowledgments

Built with insights from survival analysis, network science, and probabilistic ML research. Special thanks to the Solana ecosystem for excellent RPC infrastructure.
