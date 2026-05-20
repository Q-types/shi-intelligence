# SHI Architecture

**Technical deep-dive into Solana Holder Intelligence system design**

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Metrics Layer](#metrics-layer)
5. [Intelligence Layer](#intelligence-layer)
6. [API Design](#api-design)
7. [Real-Time Monitoring](#real-time-monitoring)
8. [Infrastructure Patterns](#infrastructure-patterns)
9. [Configuration](#configuration)

---

## System Overview

SHI follows a **layered architecture** with strict separation between:

1. **Data Layer** - External data sources (Solana RPC, Helius, Jupiter)
2. **Metrics Layer** - Deterministic, auditable calculations (FROZEN)
3. **Intelligence Layer** - Probabilistic ML models (versioned)
4. **Orchestration Layer** - Pipeline coordination and feature engineering
5. **Interface Layer** - API, Telegram bot, WebSocket, Dashboard

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Frozen Metrics** | Core formulas in `src/metrics/` cannot be modified without explicit approval |
| **Probabilistic Outputs** | All predictions include uncertainty bounds (confidence intervals) |
| **Async-First** | I/O operations use asyncio for scalability |
| **Graceful Degradation** | Circuit breakers, fallbacks, retry logic |
| **Observability** | Structured logging, Prometheus metrics, health checks |

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INTERFACE LAYER                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │   Telegram    │  │   FastAPI     │  │  WebSocket    │  │  Streamlit   │  │
│  │     Bot       │  │    REST       │  │  Real-time    │  │  Dashboard   │  │
│  │ src/telegram/ │  │  src/api/     │  │src/api/ws.py  │  │  sweenee/    │  │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └──────┬───────┘  │
└──────────┼──────────────────┼──────────────────┼─────────────────┼──────────┘
           │                  │                  │                 │
           └──────────────────┼──────────────────┼─────────────────┘
                              │                  │
┌─────────────────────────────▼──────────────────▼────────────────────────────┐
│                          ORCHESTRATION LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    AnalysisOrchestrator                              │    │
│  │                    src/pipeline/orchestrator.py                      │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │    │
│  │  │   Holder    │  │   Feature   │  │   Metric    │  │  Baseline  │  │    │
│  │  │  Fetcher    │  │  Engineer   │  │  Computer   │  │ Calibrator │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                          INTELLIGENCE LAYER                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   Clustering    │  │    Hazard       │  │    Temporal     │              │
│  │   Engine        │  │    Model        │  │    Analysis     │              │
│  │                 │  │                 │  │                 │              │
│  │ • HDBSCAN       │  │ • Cox PH        │  │ • HMM Regime    │              │
│  │ • Archetypes    │  │ • Calibration   │  │ • Trajectories  │              │
│  │ • Multi-score   │  │ • Isotonic      │  │ • Forecasting   │              │
│  │                 │  │                 │  │                 │              │
│  │ src/clustering/ │  │ src/models/     │  │ src/temporal/   │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                    │                        │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌───────▼─────────┐              │
│  │     Graph       │  │    Bayesian     │  │  Explainability │              │
│  │  Intelligence   │  │     Risk        │  │                 │              │
│  │                 │  │                 │  │ • SHAP values   │              │
│  │ • Funding graph │  │ • Beta priors   │  │ • Narratives    │              │
│  │ • Node2Vec      │  │ • Posterior     │  │ • Feature imp.  │              │
│  │ • Sybil detect  │  │ • Confidence    │  │                 │              │
│  │                 │  │                 │  │                 │              │
│  │ src/graph/      │  │ src/bayesian/   │  │src/explainability/│            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                            METRICS LAYER (FROZEN)                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Distribution   │  │  Coordination   │  │     Hazard      │              │
│  │                 │  │                 │  │                 │              │
│  │ • HHI           │  │ • Shared funder │  │ • Sell events   │              │
│  │ • Gini          │  │ • Coordination  │  │ • Time horizons │              │
│  │ • Entropy       │  │   score         │  │ • Thresholds    │              │
│  │ • Whale ratio   │  │ • Sybil flags   │  │                 │              │
│  │                 │  │                 │  │                 │              │
│  │ src/metrics/    │  │ src/metrics/    │  │ src/metrics/    │              │
│  │ distribution.py │  │ coordination.py │  │ hazard.py       │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                              DATA LAYER                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   Solana RPC    │  │     Helius      │  │    Jupiter      │              │
│  │                 │  │                 │  │                 │              │
│  │ • getTokenLarge │  │ • Enhanced API  │  │ • Price feeds   │              │
│  │   Accounts      │  │ • Webhooks      │  │ • Batch queries │              │
│  │ • getSignatures │  │ • DAS API       │  │                 │              │
│  │                 │  │                 │  │                 │              │
│  │ src/data/       │  │ src/data/       │  │ src/data/       │              │
│  │ client.py       │  │ providers.py    │  │ price_provider.py│             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   PostgreSQL    │  │     Redis       │  │     SQLite      │              │
│  │                 │  │                 │  │                 │              │
│  │ • Holder data   │  │ • Cache layer   │  │ • Local cache   │              │
│  │ • Metrics hist  │  │ • Rate limits   │  │ • SWEENEE data  │              │
│  │ • Profile snaps │  │ • Pub/sub       │  │                 │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Analysis Pipeline (Happy Path)

```
1. REQUEST
   └─► Token mint address received via API/Telegram

2. HOLDER FETCH
   └─► Solana RPC: getTokenLargestAccounts + getProgramAccounts
   └─► Result: List[HolderData] with balances

3. FUNDING GRAPH CONSTRUCTION
   └─► Helius: getSignaturesForAddress (SOL transfers)
   └─► Result: NetworkX DiGraph (wallets as nodes, transfers as edges)

4. FEATURE ENGINEERING
   └─► 14+ features per wallet
   │   ├── Temporal: entry_time, holding_duration, delta_balance
   │   ├── Trading: trade_count, burstiness, balance_volatility
   │   └── Graph: in_degree, out_degree, shared_funder_count, centrality

5. METRIC COMPUTATION (FROZEN)
   └─► HHI, Gini, Entropy, Whale Dominance, Churn Rate
   └─► Z-score normalization against baseline

6. INTELLIGENCE MODELS
   ├─► HDBSCAN → Archetype assignment (6 types)
   ├─► Cox PH → 7-day sell probability
   ├─► HMM → Current regime (5 states)
   └─► Isolation Forest → Anomaly scores

7. CALIBRATION
   └─► Isotonic regression on raw probabilities
   └─► Bayesian posterior updates for confidence intervals

8. EXPLAINABILITY
   └─► SHAP values for top features
   └─► Natural language narrative generation

9. RESPONSE
   └─► AnalysisResult with metrics, archetypes, probabilities, explanations
```

### Sequence Diagram

```
Client          API           Orchestrator      Models           RPC
  │              │                 │               │              │
  │──analyze────►│                 │               │              │
  │              │──run_analysis──►│               │              │
  │              │                 │──fetch_holders───────────────►│
  │              │                 │◄──holders_data────────────────│
  │              │                 │               │              │
  │              │                 │──build_graph──────────────────►│
  │              │                 │◄──signatures─────────────────│
  │              │                 │               │              │
  │              │                 │──compute_features─►│          │
  │              │                 │◄──features────────│          │
  │              │                 │               │              │
  │              │                 │──run_models───►│              │
  │              │                 │  ├─HDBSCAN    │              │
  │              │                 │  ├─Cox PH     │              │
  │              │                 │  └─HMM        │              │
  │              │                 │◄──predictions─│              │
  │              │                 │               │              │
  │              │                 │──calibrate────►│              │
  │              │                 │◄──calibrated──│              │
  │              │◄──result────────│               │              │
  │◄──response───│                 │               │              │
```

---

## Metrics Layer

### Frozen Formulas

All metrics in `src/metrics/` are **immutable by design**. Changes require explicit human approval.

#### Herfindahl-Hirschman Index (HHI)

```python
# src/metrics/distribution.py
def compute_hhi(shares: np.ndarray) -> float:
    """
    HHI = Σ(s_i²) where s_i = balance_i / total_supply
    Range: [0, 1] where 1 = single holder
    """
    normalized = shares / shares.sum()
    return float(np.sum(normalized ** 2))
```

#### Gini Coefficient

```python
def compute_gini(balances: np.ndarray) -> float:
    """
    Gini = (Σ Σ |b_i - b_j|) / (2 * N * Σ b_i)
    Range: [0, 1] where 1 = maximum inequality
    """
    n = len(balances)
    sorted_balances = np.sort(balances)
    cumsum = np.cumsum(sorted_balances)
    return float((2 * np.sum((np.arange(1, n + 1) * sorted_balances)) -
                  (n + 1) * cumsum[-1]) / (n * cumsum[-1]))
```

#### Shannon Entropy

```python
def compute_entropy(shares: np.ndarray) -> float:
    """
    H = -Σ(p_i × ln(p_i)) where p_i = share of holder i
    Higher = more distributed
    """
    p = shares / shares.sum()
    p = p[p > 0]  # Avoid log(0)
    return float(-np.sum(p * np.log(p)))
```

### Baseline Calibration

```python
# src/metrics/normalization.py
def z_score_normalize(value: float, baseline_mean: float, baseline_std: float) -> float:
    """Convert raw metric to z-score relative to historical baseline."""
    return (value - baseline_mean) / baseline_std

def percentile_rank(value: float, baseline_values: np.ndarray) -> float:
    """Compute percentile rank against baseline distribution."""
    return float(np.mean(baseline_values <= value))
```

---

## Intelligence Layer

### Clustering Engine

```python
# src/clustering/archetypes.py

class WalletFeatureVector:
    """14+ dimensional feature vector for clustering."""
    balance: float
    share: float
    rank: int
    entry_time_relative: float      # Days since token launch
    holding_duration: float         # Days held
    position_volatility: float      # Std dev of balance
    delta_balance_7d: float         # Change in last 7 days
    delta_balance_30d: float        # Change in last 30 days
    trade_count: int                # Transaction count
    burstiness: float               # Temporal clustering of activity
    swap_frequency: float           # DEX interactions
    lp_interaction_ratio: float     # LP position management
    in_degree: int                  # Incoming SOL transfers
    out_degree: int                 # Outgoing SOL transfers
    shared_funder_count: int        # Wallets with same funder

class ArchetypeAssigner:
    """Rule-based archetype assignment post-clustering."""

    ARCHETYPES = [
        Archetype.SNIPER,
        Archetype.LONG_TERM_ACCUMULATOR,
        Archetype.COORDINATED_CLUSTER,
        Archetype.LIQUIDITY_ACTOR,
        Archetype.EXCHANGE_LINKED,
        Archetype.DORMANT_WHALE,
    ]

    def assign(self, features: WalletFeatureVector, cluster_id: int) -> Archetype:
        # Multi-score soft assignment with confidence
        scores = self._compute_archetype_scores(features)
        return self._select_archetype(scores, cluster_id)
```

### Hazard Model

```python
# src/models/hazard_model.py

class CoxPHHazardModel:
    """Cox Proportional Hazards for sell probability prediction."""

    def __init__(self):
        self.model = CoxPHFitter(penalizer=0.1)
        self.calibrator = IsotonicRegression(out_of_bounds='clip')

    def fit(self, features: pd.DataFrame, durations: np.ndarray, events: np.ndarray):
        """
        Fit Cox PH model on historical sell events.

        Args:
            features: Wallet feature matrix (N x D)
            durations: Time to event or censoring (days)
            events: 1 if sell event occurred, 0 if censored
        """
        df = features.copy()
        df['T'] = durations
        df['E'] = events
        self.model.fit(df, duration_col='T', event_col='E')

    def predict_survival(self, features: pd.DataFrame, horizon_days: int = 7) -> np.ndarray:
        """
        Predict P(sell within horizon_days).

        λ(t|X) = λ₀(t) × exp(β'X)
        S(t|X) = exp(-Λ₀(t) × exp(β'X))
        P(sell) = 1 - S(horizon)
        """
        survival_probs = self.model.predict_survival_function(features)
        return 1 - survival_probs.loc[horizon_days].values
```

### Regime Detection

```python
# src/temporal/regimes.py

class HolderRegimeDetector:
    """Hidden Markov Model for regime detection."""

    REGIMES = [
        'ACCUMULATION',           # HHI/Gini decreasing
        'DISTRIBUTION',           # Concentration increasing
        'COORDINATED_ACCUMULATION', # Centralization + coordination
        'DECAY',                  # High churn, exits
        'STABLE',                 # Low velocity equilibrium
    ]

    def __init__(self, n_states: int = 5):
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type='full',
            n_iter=100
        )

    def fit(self, observations: np.ndarray):
        """
        Fit HMM on metric time series.

        Args:
            observations: (T x 4) matrix of [dHHI, dGini, dChurn, coordination]
        """
        self.model.fit(observations)

    def predict_regime(self, observations: np.ndarray) -> str:
        """Viterbi decode to find most likely current regime."""
        state_sequence = self.model.predict(observations)
        return self.REGIMES[state_sequence[-1]]
```

---

## API Design

### FastAPI Routes

```python
# src/api/routes.py

from fastapi import APIRouter, Depends, HTTPException
from src.pipeline.orchestrator import AnalysisOrchestrator
from src.api.schemas import AnalysisRequest, AnalysisResponse

router = APIRouter(prefix="/api/v1")

@router.post("/analyze/{token_mint}", response_model=AnalysisResponse)
async def analyze_token(
    token_mint: str,
    request: AnalysisRequest,
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator)
):
    """Full token analysis with metrics, archetypes, and predictions."""
    try:
        result = await orchestrator.analyze(
            token_mint=token_mint,
            max_holders=request.max_holders,
            include_graph=request.include_graph
        )
        return AnalysisResponse.from_result(result)
    except TokenNotFoundError:
        raise HTTPException(status_code=404, detail="Token not found")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
```

### Response Schema

```python
# src/api/schemas.py

class AnalysisResponse(BaseModel):
    """Full analysis response."""
    token_mint: str
    timestamp: datetime

    # Metrics (FROZEN)
    metrics: MetricsResponse

    # Intelligence
    archetypes: ArchetypeDistribution
    regime: RegimeResponse
    risk_score: float
    sell_probability_7d: float

    # Confidence
    confidence_interval: tuple[float, float]
    calibration_quality: str

    # Explanations
    top_risk_factors: list[RiskFactor]
    narrative: str

    class Config:
        json_schema_extra = {
            "example": {
                "token_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "risk_score": 0.42,
                "regime": "ACCUMULATION",
                "sell_probability_7d": 0.23
            }
        }
```

---

## Real-Time Monitoring

### WalletWatcher Service

```python
# src/monitoring/watcher.py

class WalletWatcher:
    """Async service for real-time wallet monitoring."""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.watchlist: dict[str, WatchedWallet] = {}
        self._running = False

    async def start_monitoring(self):
        """Start background monitoring loop."""
        self._running = True
        while self._running:
            await self._check_all_wallets()
            await asyncio.sleep(self.check_interval)

    async def _check_all_wallets(self):
        """Check balance changes for all watched wallets."""
        for wallet in self.watchlist.values():
            old_balance = wallet.last_balance
            new_balance = await self._fetch_balance(wallet)

            if self._is_significant_change(old_balance, new_balance, wallet.threshold):
                await self._emit_alert(wallet, old_balance, new_balance)
```

### Alert Engine

```python
# src/monitoring/alerts.py

class AlertEngine:
    """Configurable alert generation with cooldowns."""

    ALERT_TYPES = [
        AlertType.WHALE_MOVEMENT,        # Large balance changes
        AlertType.REGIME_CHANGE,         # HMM state transitions
        AlertType.ANOMALY_SPIKE,         # Isolation Forest flags
        AlertType.CONCENTRATION_INCREASE # HHI/Gini increases
    ]

    async def process_event(self, event: MonitoringEvent) -> Alert | None:
        """Generate alert if event meets thresholds and not in cooldown."""
        if not self._check_thresholds(event):
            return None
        if self._in_cooldown(event.user_id, event.alert_type):
            return None

        alert = self._create_alert(event)
        self._set_cooldown(event.user_id, event.alert_type)
        return alert
```

---

## Infrastructure Patterns

### Circuit Breaker

```python
# src/infra/circuit_breaker.py

class CircuitBreaker:
    """Prevents cascade failures on external service outages."""

    STATES = ['CLOSED', 'OPEN', 'HALF_OPEN']

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = 'CLOSED'
        self.failures = 0

    async def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise CircuitOpenError()

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

### Rate Limiting

```python
# src/infra/rate_limit.py

class TokenBucketRateLimiter:
    """Token bucket algorithm for rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    async def acquire(self, tokens: int = 1) -> bool:
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
```

### Retry Logic

```python
# src/infra/retry.py

async def with_retry(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    exponential: bool = True
):
    """Exponential backoff retry wrapper."""
    for attempt in range(max_retries):
        try:
            return await func()
        except RetryableError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt if exponential else 1)
            await asyncio.sleep(delay)
```

---

## Configuration

### Environment Variables

```python
# src/core/config.py

class Settings(BaseSettings):
    """Application settings from environment."""

    model_config = SettingsConfigDict(env_file=".env")

    # Data Sources
    helius_api_key: str
    helius_rpc_url: str = "https://mainnet.helius-rpc.com"
    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"

    # Database
    database_url: str = "postgresql+asyncpg://localhost/shi"
    redis_url: str = "redis://localhost:6379/0"

    # Processing
    max_holders_per_token: int = 50000
    sla_timeout_seconds: int = 30

    # Hazard Model
    sell_event_threshold_pct: float = 0.5
    sell_event_horizon_days: int = 7

    # Feature Flags
    use_probability_calibration: bool = True
    calibration_method: str = "isotonic"
    use_robust_clustering: bool = True
    use_temporal_validation: bool = True
```

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `use_probability_calibration` | `True` | Apply isotonic calibration to hazard model |
| `use_robust_clustering` | `True` | Use log1p/asinh transforms and RobustScaler |
| `use_node2vec_clustering` | `False` | Enable experimental graph embeddings |
| `use_temporal_validation` | `True` | Walk-forward CV instead of random KFold |
| `use_regime_specific_calibration` | `False` | Per-regime calibrators (requires 50+ samples) |

---

## Directory Structure

```
src/
├── core/
│   ├── config.py           # Settings from environment
│   └── types.py            # WalletAddress, TokenMint, etc.
│
├── metrics/                # FROZEN - do not modify
│   ├── distribution.py     # HHI, Gini, Entropy
│   ├── coordination.py     # Shared funders, sybil flags
│   ├── hazard.py           # Sell event definitions
│   └── normalization.py    # Z-score, percentile ranking
│
├── clustering/
│   ├── archetypes.py       # 6 behavioral types
│   ├── transformations.py  # Feature engineering
│   ├── diagnostics.py      # Silhouette, stability
│   └── node2vec_integration.py  # Graph embeddings
│
├── models/
│   ├── hazard_model.py     # Cox PH survival
│   ├── calibration.py      # Isotonic, Platt, Beta
│   ├── regime.py           # Gaussian HMM
│   └── validation.py       # Schoenfeld residuals
│
├── graph/
│   ├── funding_graph.py    # NetworkX DiGraph
│   ├── embeddings.py       # Node2Vec
│   ├── anomaly.py          # Isolation Forest
│   └── dynamics.py         # Community detection
│
├── temporal/
│   ├── trajectories.py     # Balance evolution
│   ├── regimes.py          # HMM regime detection
│   └── forecasting.py      # Capital flow predictions
│
├── bayesian/
│   ├── priors.py           # Prior elicitation
│   ├── updater.py          # Posterior updating
│   └── confidence.py       # Calibrated intervals
│
├── explainability/
│   ├── shap_explainer.py   # SHAP values
│   └── narratives.py       # Natural language
│
├── monitoring/
│   ├── watcher.py          # Async wallet monitoring
│   ├── alerts.py           # Alert engine
│   └── profiles.py         # Profile evolution
│
├── pipeline/
│   ├── orchestrator.py     # Main analysis flow
│   ├── features.py         # Feature engineering
│   └── baseline.py         # Historical calibration
│
├── api/
│   ├── routes.py           # FastAPI endpoints
│   ├── schemas.py          # Pydantic models
│   ├── middleware.py       # Rate limiting, auth
│   └── websocket.py        # Real-time subscriptions
│
├── telegram/
│   ├── bot.py              # Main handler
│   ├── commands/           # Command implementations
│   └── notifications.py    # Alert delivery
│
├── data/
│   ├── client.py           # Solana RPC
│   ├── providers.py        # Multi-RPC fallback
│   └── price_provider.py   # Jupiter API
│
└── infra/
    ├── cache.py            # Redis/memory caching
    ├── rate_limit.py       # Token bucket
    ├── retry.py            # Exponential backoff
    └── circuit_breaker.py  # Failure isolation
```

---

## Related Documentation

- [README.md](../README.md) - Project overview
- [DATA_SCIENCE.md](DATA_SCIENCE.md) - Methodology deep-dive
- [Validation Reports](validation/) - Model validation results
