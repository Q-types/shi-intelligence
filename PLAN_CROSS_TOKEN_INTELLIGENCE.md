# SHI Cross-Token Wallet Intelligence

## Executive Summary

Transform SHI from single-token analysis to **cross-token wallet intelligence** with:
- Entity aggregation (grouping wallets by shared funders)
- Wallet reputation scoring based on historical patterns
- Professional sybil network detection
- Entity-level risk aggregation

**4 Sprints | 24 Tasks | ~2,500 lines of new code**

---

## Architecture Overview

```
                                    CROSS-TOKEN INTELLIGENCE LAYER
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
    │  │  Entity         │    │  Reputation     │    │  Sybil Network  │        │
    │  │  Resolver       │───▶│  Engine         │───▶│  Detector       │        │
    │  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘        │
    │           │                      │                      │                  │
    │           ▼                      ▼                      ▼                  │
    │  ┌─────────────────────────────────────────────────────────────────┐      │
    │  │                    CROSS-TOKEN DATABASE                          │      │
    │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │      │
    │  │  │wallet_history│ │   entities   │ │   wallet_reputation      │ │      │
    │  │  │              │ │              │ │                          │ │      │
    │  │  │- wallet      │ │- entity_id   │ │- wallet_address          │ │      │
    │  │  │- token_mint  │ │- type        │ │- reputation_score (0-100)│ │      │
    │  │  │- archetype   │ │- confidence  │ │- sniper_count            │ │      │
    │  │  │- entry_price │ │- dominant_   │ │- accumulator_count       │ │      │
    │  │  │- exit_price  │ │  funder      │ │- rugpull_count           │ │      │
    │  │  │- pnl_pct     │ └──────────────┘ │- patterns[]              │ │      │
    │  │  │- hold_days   │ ┌──────────────┐ │- confidence_level        │ │      │
    │  │  └──────────────┘ │entity_member │ └──────────────────────────┘ │      │
    │  │                   │- entity_id   │                              │      │
    │  │                   │- wallet_addr │                              │      │
    │  │                   │- confidence  │                              │      │
    │  │                   │- detected_via│                              │      │
    │  │                   └──────────────┘                              │      │
    │  └─────────────────────────────────────────────────────────────────┘      │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         EXISTING SHI PIPELINE                               │
    │                                                                             │
    │  Token ──▶ Holders ──▶ Funding Graph ──▶ Features ──▶ Archetypes ──▶ Risk  │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
                    ANALYSIS PIPELINE (ENHANCED)

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Token     │────▶│   Fetch     │────▶│   Build     │
│   Input     │     │   Holders   │     │   Funding   │
│             │     │             │     │   Graph     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Compute   │────▶│   Assign    │────▶│   *** NEW   │
│   Features  │     │   Archetypes│     │   Entity    │
│             │     │             │     │   Resolver  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   *** NEW   │────▶│   *** NEW   │────▶│   Generate  │
│   Update    │     │   Aggregate │     │   Risk      │
│   Reputation│     │   Entity    │     │   Report    │
│             │     │   Risks     │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                    ┌─────────────────┐
                                    │  AnalysisResult │
                                    │  + cross_token_ │
                                    │    insights     │
                                    │  + entity_risks │
                                    │  + reputations  │
                                    └─────────────────┘
```

---

## Sprint Plan

### Sprint 1: Data Foundation
**Goal:** Database schema for cross-token intelligence

| Task | Team | Priority | Description |
|------|------|----------|-------------|
| wallet_history table | Database | P1 | Track wallet behavior per token |
| entity tables | Database | P1 | Entity + membership tables |
| wallet_reputation table | Database | P2 | Reputation scores storage |
| Alembic migrations | Database | P2 | Database migration files |
| WalletHistoryRepository | Backend | P2 | Data access layer |
| EntityRepository | Backend | P2 | Entity CRUD operations |

**Deliverables:**
- `src/data/models.py` - New SQLAlchemy models
- `alembic/versions/xxx_cross_token.py` - Migration
- `src/data/repositories/wallet_history.py`
- `src/data/repositories/entity.py`

---

### Sprint 2: Entity Resolution Engine
**Goal:** Detect and group wallets with shared funders

| Task | Team | Priority | Description |
|------|------|----------|-------------|
| SharedFunderDetector | Backend | P1 | Cross-token funder analysis |
| TemporalSyncDetector | Backend | P2 | Synchronized behavior detection |
| EntityResolver | Backend | P1 | Orchestrator combining detectors |
| Pipeline integration | Backend | P2 | Wire into AnalysisOrchestrator |
| Test suite | Testing | P2 | Unit + integration tests |

**Algorithm: Shared Funder Detection**
```python
def detect_shared_funders(wallets, current_token):
    for wallet in wallets:
        # Get funding ancestors (depth=3)
        funders = graph.get_ancestors(wallet, max_depth=3)

        # Query wallet_history for other tokens with same funders
        for funder in funders:
            other_tokens = db.query("""
                SELECT DISTINCT wh.token_mint, wh.wallet_address
                FROM wallet_history wh
                JOIN funding_edges fe ON fe.target = wh.wallet_address
                WHERE fe.source = :funder
                AND wh.token_mint != :current_token
            """, funder=funder, current_token=current_token)

            if len(other_tokens) > 0:
                yield SharedFunderMatch(funder, wallet, other_tokens)
```

**Deliverables:**
- `src/entity/shared_funder_detector.py`
- `src/entity/temporal_sync_detector.py`
- `src/entity/resolver.py`
- `tests/test_entity_resolution.py`

---

### Sprint 3: Reputation Scoring System
**Goal:** Wallet reputation based on historical cross-token behavior

| Task | Team | Priority | Description |
|------|------|----------|-------------|
| ReputationScorer | Backend | P1 | Scoring algorithm |
| PatternDetector | Backend | P1 | Historical pattern detection |
| ReputationEngine | Backend | P1 | Orchestrator with caching |
| /reputation API | Backend | P2 | REST endpoint |
| /profile Telegram | Backend | P3 | Telegram command |
| Test suite | Testing | P2 | Comprehensive tests |

**Reputation Formula:**
```
REPUTATION_SCORE = 50 (base)
    + 3 * accumulator_count      # Bonus for patient holding
    - 5 * sniper_count           # Penalty for early dumping
    - 15 * rugpull_count         # Heavy penalty for rug participation
    + 0.1 * avg_holding_days     # Bonus for long-term holding

CONFIDENCE_LEVEL:
    LOW    = tokens_analyzed < 5
    MEDIUM = 5 <= tokens_analyzed <= 20
    HIGH   = tokens_analyzed > 20
```

**Detected Patterns:**
| Pattern | Criteria | Risk Implication |
|---------|----------|------------------|
| SERIAL_SNIPER | Sniper on 3+ tokens | HIGH - likely to dump |
| DIAMOND_HANDS | Accumulator on 5+ tokens, avg hold >30d | LOW - conviction holder |
| RUG_PARTICIPANT | Held token that rugged, exited late | MEDIUM - poor judgment |
| SMART_MONEY | Consistent positive PnL | LOW - informed trader |
| LIQUIDITY_PROVIDER | LP actor on multiple tokens | LOW - market maker |

**Deliverables:**
- `src/reputation/scorer.py`
- `src/reputation/pattern_detector.py`
- `src/reputation/engine.py`
- `src/api/routes.py` - /reputation endpoint
- `src/telegram/handlers.py` - /profile command

---

### Sprint 4: Cross-Token Intelligence
**Goal:** Advanced sybil detection and entity-level risk

| Task | Team | Priority | Description |
|------|------|----------|-------------|
| SybilNetworkDetector | Backend | P1 | Professional sybil detection |
| EntityRiskAggregator | Backend | P1 | Entity-level risk metrics |
| AnalysisResult enhancement | Backend | P2 | Add cross-token fields |
| /entity API | Backend | P2 | Entity endpoints |
| Telegram commands | Backend | P3 | /entity, /sybils commands |
| Integration tests | Testing | P1 | End-to-end validation |
| Documentation | Docs | P3 | Architecture + API docs |

**Professional Sybil Detection:**
```python
def detect_professional_sybils(entity):
    # Get all token interactions for entity wallets
    interactions = db.query("""
        SELECT token_mint, wallet_address, entry_time, exit_time
        FROM wallet_history
        WHERE wallet_address IN (
            SELECT wallet_address FROM entity_memberships
            WHERE entity_id = :entity_id
        )
    """, entity_id=entity.id)

    # Calculate network metrics
    tokens_targeted = len(set(i.token_mint for i in interactions))
    temporal_correlation = calculate_temporal_sync(interactions)

    # Flag as professional if coordinated across many tokens
    if tokens_targeted > 10 and temporal_correlation > 0.8:
        return ProfessionalSybilNetwork(
            entity=entity,
            tokens_targeted=tokens_targeted,
            coordination_score=temporal_correlation,
            risk_level="CRITICAL"
        )
```

**Entity Risk Aggregation:**
```python
@dataclass
class EntityRisk:
    entity_id: str
    combined_holdings_pct: float  # Sum of member holdings
    wallet_count: int
    avg_reputation: float
    coordinated_dump_probability: float  # P(all sell within 1 hour)
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
```

**Deliverables:**
- `src/entity/sybil_network_detector.py`
- `src/entity/risk_aggregator.py`
- `src/pipeline/orchestrator.py` - Enhanced AnalysisResult
- `src/api/routes.py` - /entity endpoints
- `docs/cross_token_intelligence.md`

---

## Database Schema

### New Tables

```sql
-- Track wallet behavior per token
CREATE TABLE wallet_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    wallet_address VARCHAR(44) NOT NULL,
    token_mint VARCHAR(44) NOT NULL,
    first_seen_at TIMESTAMP WITH TIME ZONE,
    last_seen_at TIMESTAMP WITH TIME ZONE,
    archetype_assigned VARCHAR(50),
    entry_price_usd DECIMAL(20, 10),
    exit_price_usd DECIMAL(20, 10),
    realized_pnl_pct DECIMAL(10, 4),
    holding_duration_days INTEGER,
    trade_count INTEGER,
    max_balance BIGINT,
    was_sniper BOOLEAN DEFAULT FALSE,
    was_accumulator BOOLEAN DEFAULT FALSE,
    was_rugpuller BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(wallet_address, token_mint)
);

CREATE INDEX idx_wallet_history_wallet ON wallet_history(wallet_address);
CREATE INDEX idx_wallet_history_token ON wallet_history(token_mint);

-- Entity groupings
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,  -- SYBIL_CLUSTER, WHALE_GROUP, EXCHANGE, UNKNOWN
    confidence_score DECIMAL(5, 4),
    dominant_funder_address VARCHAR(44),
    wallet_count INTEGER DEFAULT 0,
    tokens_targeted INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Wallet-to-entity mapping
CREATE TABLE entity_memberships (
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    wallet_address VARCHAR(44) NOT NULL,
    membership_confidence DECIMAL(5, 4),
    detected_via VARCHAR(50),  -- SHARED_FUNDER, TEMPORAL_SYNC, BEHAVIOR_SIMILARITY
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (entity_id, wallet_address)
);

CREATE INDEX idx_membership_wallet ON entity_memberships(wallet_address);

-- Wallet reputation scores
CREATE TABLE wallet_reputation (
    wallet_address VARCHAR(44) PRIMARY KEY,
    reputation_score INTEGER CHECK (reputation_score BETWEEN 0 AND 100),
    confidence_level VARCHAR(10),  -- LOW, MEDIUM, HIGH
    tokens_analyzed INTEGER DEFAULT 0,
    sniper_count INTEGER DEFAULT 0,
    accumulator_count INTEGER DEFAULT 0,
    rugpull_count INTEGER DEFAULT 0,
    avg_holding_days DECIMAL(10, 2),
    avg_pnl_pct DECIMAL(10, 4),
    patterns JSONB DEFAULT '[]',
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

---

## API Endpoints

### New Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/wallet/{address}/reputation` | Wallet reputation score |
| GET | `/api/v1/wallet/{address}/history` | Cross-token history |
| GET | `/api/v1/wallet/{address}/entity` | Entity membership |
| GET | `/api/v1/entity/{id}` | Entity details |
| GET | `/api/v1/entity/{id}/wallets` | Entity wallet list |
| GET | `/api/v1/token/{mint}/sybils` | Detected sybil networks |

### Response Examples

**GET /api/v1/wallet/{address}/reputation**
```json
{
    "wallet": "7xKX...abc",
    "reputation_score": 72,
    "confidence_level": "HIGH",
    "patterns": [
        {"type": "DIAMOND_HANDS", "confidence": 0.85, "token_count": 7}
    ],
    "history_summary": {
        "tokens_analyzed": 23,
        "avg_holding_days": 45.2,
        "total_pnl_pct": 124.5,
        "sniper_count": 2,
        "accumulator_count": 15
    },
    "last_updated": "2026-05-12T10:30:00Z"
}
```

**GET /api/v1/entity/{id}**
```json
{
    "entity_id": "ent_123...",
    "entity_type": "SYBIL_CLUSTER",
    "confidence_score": 0.92,
    "dominant_funder": "ABC...xyz",
    "wallets": [
        {"address": "111...", "confidence": 0.95},
        {"address": "222...", "confidence": 0.88},
        {"address": "333...", "confidence": 0.91}
    ],
    "network_metrics": {
        "tokens_targeted": 15,
        "total_volume_usd": 1250000,
        "coordination_score": 0.87,
        "is_professional": true
    },
    "risk_level": "CRITICAL"
}
```

---

## Telegram Commands

### New Commands

| Command | Description |
|---------|-------------|
| `/profile <wallet>` | Wallet reputation and history |
| `/entity <wallet>` | Show entity if wallet is part of one |
| `/sybils` | List sybil networks in last analyzed token |

### Example Output

```
/profile 7xKX...abc

📊 WALLET REPUTATION

Score: ████████░░ 72/100 (HIGH confidence)

📈 Patterns Detected:
  💎 DIAMOND_HANDS (85% confidence)

📜 History (23 tokens):
  • Avg Hold: 45 days
  • Total PnL: +124.5%
  • Sniper: 2x | Accumulator: 15x

⚠️ Risk Level: LOW
```

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Database bottleneck at scale | HIGH | Redis caching, batch queries, indexes |
| Real-time complexity | MEDIUM | Incremental updates, not full recalculation |
| "All tokens" scope creep | MEDIUM | Scope to analyzed tokens only |
| Entity merge conflicts | LOW | Confidence-weighted merge algorithm |
| False positive sybils | MEDIUM | High threshold (>80% correlation) |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Entity detection accuracy | >85% |
| Reputation score correlation with outcomes | >0.7 |
| Professional sybil recall | >90% |
| Analysis latency increase | <5s |
| API response time (reputation) | <200ms |

---

## File Structure

```
src/
├── entity/                    # NEW MODULE
│   ├── __init__.py
│   ├── shared_funder_detector.py
│   ├── temporal_sync_detector.py
│   ├── resolver.py
│   ├── sybil_network_detector.py
│   └── risk_aggregator.py
├── reputation/                # NEW MODULE
│   ├── __init__.py
│   ├── scorer.py
│   ├── pattern_detector.py
│   └── engine.py
├── data/
│   ├── models.py             # + new tables
│   └── repositories/
│       ├── wallet_history.py  # NEW
│       └── entity.py          # NEW
└── pipeline/
    └── orchestrator.py        # MODIFIED

tests/
├── test_entity_resolution.py  # NEW
├── test_reputation.py         # NEW
└── integration/
    └── test_cross_token.py    # NEW

docs/
└── cross_token_intelligence.md # NEW
```

---

## Next Steps

1. **Review this plan** - Validate architecture decisions
2. **Start Sprint 1** - Run: `spark-mission run` with Sprint 1 tasks
3. **Iterate** - Each sprint builds on previous

**Estimated Timeline:** 4 sprints x 2-3 days = ~10-12 days

---

*Generated by Architect MCP + Muse MCP + Mind MCP on 2026-05-12*
