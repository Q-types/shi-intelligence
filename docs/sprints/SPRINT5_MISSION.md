# Sprint 5: Advanced Intelligence & Production Hardening

> **Sprint ID**: shi-v2-sprint5
> **Goal**: Add sequence modeling, Bayesian risk estimation, and production hardening
> **Priority**: high

---

## Sprint Objective

Add advanced intelligence capabilities and production hardening:

1. **Sequence Modeling** - Wallet action sequence analysis using RNN/LSTM patterns
2. **Bayesian Risk Estimation** - Uncertainty-aware risk beliefs with continuous updating
3. **Production Hardening** - Fix failing tests, reduce mypy errors, deprecation warnings
4. **API Endpoints** - REST API for dashboard integration

---

## Tasks

### Task 1: Wallet Sequence Modeling (`src/sequence/`)

**Goal**: Model wallet behavior as sequential patterns to detect behavioral motifs

**Deliverables**:
- `src/sequence/__init__.py`
- `src/sequence/encoder.py` - Encode wallet actions as sequences
- `src/sequence/patterns.py` - Pattern detection and clustering
- `src/sequence/signatures.py` - Pre-dump signature detection
- `tests/test_sequence.py` - Unit tests (>80% coverage)

**Key Features**:
```python
# Action encoding
class WalletActionEncoder:
    """Encode wallet actions as sequences."""
    ACTIONS = ["funded", "swap_buy", "swap_sell", "lp_add", "lp_remove", "idle", "transfer_in", "transfer_out"]

    def encode_sequence(self, actions: List[str]) -> np.ndarray:
        """Convert action list to numerical sequence."""

    def embed_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Embed sequence for similarity comparison."""

# Pattern detection
class SequencePatternDetector:
    """Detect behavioral patterns in action sequences."""

    def find_motifs(self, sequences: List[np.ndarray]) -> List[Motif]:
        """Find recurring behavioral motifs."""

    def cluster_behaviors(self, sequences: List[np.ndarray]) -> Dict[int, List[str]]:
        """Cluster wallets by behavioral pattern."""

    def detect_dump_signature(self, sequence: np.ndarray) -> float:
        """Score likelihood of pre-dump behavior pattern."""
```

**Acceptance Criteria**:
- [ ] Actions encoded as integer sequences
- [ ] Behavioral clustering groups similar wallets
- [ ] Dump signature detection scores between 0-1
- [ ] Tests pass with >80% coverage

---

### Task 2: Bayesian Risk Estimation (`src/bayesian/`)

**Goal**: Uncertainty-aware risk beliefs that update incrementally

**Deliverables**:
- `src/bayesian/__init__.py`
- `src/bayesian/priors.py` - Prior distributions for risk factors
- `src/bayesian/updater.py` - Bayesian belief updating
- `src/bayesian/risk_belief.py` - Composite risk belief model
- `tests/test_bayesian.py` - Unit tests (>80% coverage)

**Key Features**:
```python
class RiskBeliefModel:
    """Bayesian risk belief with uncertainty quantification."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """Initialize with Beta prior for rug probability."""

    def update(self, evidence: Evidence) -> 'RiskBeliefModel':
        """Update beliefs with new evidence."""

    def posterior_rug_probability(self) -> Tuple[float, float, float]:
        """Return (mean, lower_ci, upper_ci) for P(rug)."""

    def credible_interval(self, alpha: float = 0.95) -> Tuple[float, float]:
        """Return credible interval for risk estimate."""

    def information_gain(self, new_evidence: Evidence) -> float:
        """Calculate expected information gain from evidence."""
```

**Evidence Types**:
- Holder concentration change
- Large wallet movement
- Regime transition
- Anomaly detection
- Historical similar token outcome

**Acceptance Criteria**:
- [ ] Prior/posterior distributions correctly implemented
- [ ] Credible intervals properly calculated
- [ ] Belief updating is mathematically correct
- [ ] Tests validate against known Beta-Binomial results

---

### Task 3: REST API Endpoints (`src/api/`)

**Goal**: REST API for dashboard and external integrations

**Deliverables**:
- `src/api/__init__.py`
- `src/api/routes.py` - FastAPI route definitions
- `src/api/schemas.py` - Request/response schemas
- `src/api/dependencies.py` - Dependency injection
- `tests/test_api.py` - API tests

**Endpoints**:
```python
# GET /api/v1/token/{mint}/intelligence
# Returns: TokenIntelligence (from dashboard_data.py)

# GET /api/v1/token/{mint}/forecast?days=7
# Returns: ForecastData

# GET /api/v1/token/{mint}/explain
# Returns: Risk explanation with SHAP values

# GET /api/v1/wallet/{address}/profile
# Returns: WalletProfile with sequence patterns

# POST /api/v1/token/{mint}/risk/update
# Body: Evidence
# Returns: Updated Bayesian risk belief
```

**Acceptance Criteria**:
- [ ] All endpoints return valid Pydantic models
- [ ] OpenAPI/Swagger docs generated
- [ ] Rate limiting configured
- [ ] Tests cover all endpoints

---

### Task 4: Production Hardening

**Goal**: Fix existing test failures and reduce technical debt

**Subtasks**:

**4a. Fix Failing Integration Tests**
- Fix 19 failing tests in `tests/integration/` and `tests/stress/`
- Fix 15 errors in `tests/test_graph_intelligence.py` and `tests/test_metrics_reproducibility.py`

**4b. Fix Deprecation Warnings**
- Replace all `datetime.utcnow()` with `datetime.now(datetime.UTC)`
- Currently 225+ warnings across the codebase

**4c. Reduce Mypy Errors**
- Current: ~93 mypy errors
- Target: <30 mypy errors
- Add type stubs where needed

**4d. Update Telegram Bot**
- Register new commands: `/sequence`, `/belief`
- Add new command handlers

**Acceptance Criteria**:
- [ ] All 122+ passing tests continue to pass
- [ ] Failing tests fixed or properly skipped with reason
- [ ] Deprecation warnings reduced by >80%
- [ ] Mypy errors <30

---

## Quality Gates

### Per-Task Gates
- [ ] All new code has unit tests (coverage > 80%)
- [ ] No new critical lint errors (`ruff check` passes)
- [ ] Type hints on all public functions
- [ ] Docstrings following NumPy style

### Sprint Gates
- [ ] `pytest tests/` passes with >80% of tests
- [ ] `ruff check src/` passes
- [ ] `mypy src/` has <30 errors
- [ ] New features documented in SPRINT5_SUMMARY.md

---

## File Structure (New)

```
src/
├── sequence/
│   ├── __init__.py
│   ├── encoder.py          # Action sequence encoding
│   ├── patterns.py         # Pattern detection
│   └── signatures.py       # Dump signature detection
├── bayesian/
│   ├── __init__.py
│   ├── priors.py           # Prior distributions
│   ├── updater.py          # Belief updating
│   └── risk_belief.py      # Composite risk model
├── api/
│   ├── __init__.py
│   ├── routes.py           # FastAPI routes
│   ├── schemas.py          # Request/response schemas
│   └── dependencies.py     # DI container
└── telegram/
    └── commands/
        ├── sequence.py     # /sequence command
        └── belief.py       # /belief command

tests/
├── test_sequence.py
├── test_bayesian.py
└── test_api.py
```

---

## Dependencies (New)

```toml
# Add to pyproject.toml
"fastapi>=0.109.0",
"uvicorn>=0.27.0",
```

---

## Verification Commands

```bash
# Lint
ruff check src/sequence/ src/bayesian/ src/api/

# Type check
mypy src/sequence/ src/bayesian/ src/api/

# Tests
pytest tests/test_sequence.py tests/test_bayesian.py tests/test_api.py -v --cov

# Full suite
pytest tests/ -v --tb=short
```

---

## Notes

1. **Sequence modeling** uses pattern matching, not deep learning - keeping it practical
2. **Bayesian updating** uses conjugate priors (Beta-Binomial) for closed-form solutions
3. **API** reuses existing dashboard_data.py structures
4. **Production hardening** focuses on test reliability, not new features

---

*Sprint 5 extends SHI v2 with behavioral intelligence and production readiness*
