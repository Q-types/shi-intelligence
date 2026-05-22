# Evidence Package Specification

## Overview

Every label in the Intelligence Corpus must include an evidence package. Evidence packages capture the data and reasoning that led to a classification, enabling human review and audit trails.

## Base Evidence Package

All evidence packages inherit from:

```python
@dataclass
class EvidencePackage(ABC):
    """Base evidence package."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        pass

    @abstractmethod
    def summary(self) -> str:
        """Human-readable summary."""
        pass

    @abstractmethod
    def confidence_factors(self) -> dict[str, float]:
        """Factors contributing to confidence."""
        pass
```

---

## Exit Event Evidence

**Domain:** `exit_event`

```python
@dataclass
class ExitEventEvidence(EvidencePackage):
    # Core identifiers
    exit_id: str              # Unique exit identifier
    wallet: str               # Source wallet
    token_mint: str           # Token being exited

    # Transaction details
    amount_tokens: float      # Amount of tokens
    value_sol: float          # Value in SOL at exit time
    counterparty: str         # Destination/counterparty
    transaction_signature: str  # On-chain signature
    block_time: str           # ISO timestamp

    # Classification evidence
    program_id: str           # Program that handled the exit
    instruction_type: str | None  # Parsed instruction type

    # Supporting evidence
    confidence_factors: dict[str, float]  # Factor → weight

    # Optional context
    pool_address: str | None  # If LP operation
    route_hops: list[str] | None  # If routed swap
```

### Confidence Factors

| Factor | Description |
|--------|-------------|
| `signature_match` | Transaction signature matches expected pattern |
| `program_confidence` | Confidence in program ID classification |
| `counterparty_known` | Counterparty is a known entity (CEX, pool, etc.) |
| `amount_consistency` | Amount consistent with typical operations |

---

## Coordination Evidence

**Domain:** `coordination`

```python
@dataclass
class CoordinationEvidence(EvidencePackage):
    # Cluster info
    cluster_id: str           # Cluster being evaluated
    cluster_size: int         # Number of wallets

    # Coordination signals
    timing_score: float       # Timing synchronization (0-1)
    amount_pattern_score: float  # Amount pattern similarity (0-1)
    token_overlap_score: float   # Common token holdings (0-1)
    counterparty_overlap_score: float  # Shared counterparties (0-1)

    # Network features
    funding_source_common: bool  # Same funding source?
    transaction_graph_density: float  # Graph connectivity

    # Examples
    sample_wallets: list[str]  # Sample wallets from cluster
    sample_transactions: list[str]  # Sample coordinated txs
```

### Confidence Factors

| Factor | Description |
|--------|-------------|
| `timing_coordination` | Transactions within tight time windows |
| `amount_pattern` | Similar amounts across wallets |
| `token_selection` | Same tokens targeted |
| `counterparty_pattern` | Same counterparties used |
| `funding_link` | Common funding source identified |

---

## Wallet Behaviour Evidence

**Domain:** `wallet_behaviour`

```python
@dataclass
class WalletBehaviourEvidence(EvidencePackage):
    # Wallet info
    wallet_address: str       # Wallet being classified
    observation_period_days: int  # Period observed

    # Activity metrics
    transaction_count: int    # Total transactions
    unique_tokens_traded: int # Distinct tokens
    avg_hold_time_hours: float  # Average holding period
    avg_position_size_sol: float  # Average position

    # Timing patterns
    reaction_time_median_ms: int  # Reaction to events
    active_hours: list[int]   # Hours typically active

    # Behaviour signals
    win_rate: float           # Profitable exits / total exits
    avg_profit_pct: float     # Average profit percentage
    max_drawdown_pct: float   # Worst loss
```

### Confidence Factors

| Factor | Description |
|--------|-------------|
| `sample_size` | Sufficient transactions observed |
| `pattern_consistency` | Behaviour consistent over time |
| `metric_confidence` | Metrics calculated reliably |

---

## Token Outcome Evidence

**Domain:** `token_outcome`

```python
@dataclass
class TokenOutcomeEvidence(EvidencePackage):
    # Token info
    token_mint: str           # Token address
    token_name: str           # Token name/symbol
    launch_date: str          # When launched

    # Lifecycle metrics
    peak_market_cap: float    # Maximum market cap
    peak_holder_count: int    # Maximum holders
    days_active: int          # Days with activity

    # Outcome signals
    liquidity_removed_pct: float  # LP removed %
    insider_extraction_pct: float # Insider sells %
    holder_loss_pct: float    # % holders at loss

    # Team signals
    team_wallet_activity: str # What team did
    social_activity_score: float  # Social presence
```

### Confidence Factors

| Factor | Description |
|--------|-------------|
| `data_completeness` | Full lifecycle data available |
| `timeline_clear` | Clear sequence of events |
| `extraction_pattern` | Clear extraction vs organic fade |

---

## Launch Trajectory Evidence

**Domain:** `launch_trajectory`

```python
@dataclass
class LaunchTrajectoryEvidence(EvidencePackage):
    # Token info
    token_mint: str           # Token address
    launch_timestamp: str     # Launch time

    # First-hour metrics
    first_buyers_count: int   # Buyers in first hour
    first_hour_volume_sol: float  # Volume in first hour
    sniper_percentage: float  # % bought in first 10 blocks

    # Distribution metrics
    top_10_concentration: float  # % held by top 10
    unique_funding_sources: int  # Distinct funding wallets

    # Pattern signals
    coordinated_entry_score: float  # Entry timing correlation
    pre_launch_activity: bool  # Suspicious pre-launch activity
```

### Confidence Factors

| Factor | Description |
|--------|-------------|
| `data_freshness` | Data captured near launch |
| `sample_coverage` | % of transactions captured |
| `pattern_strength` | Strength of identified pattern |

---

## Entity Link Evidence

**Domain:** `entity_resolution`

```python
@dataclass
class EntityLinkEvidence(EvidencePackage):
    # Wallets being linked
    wallet_a: str             # First wallet
    wallet_b: str             # Second wallet

    # Link signals
    direct_transfer: bool     # Direct transfers between them
    common_funding: bool      # Same funding source
    timing_correlation: float # Transaction timing correlation

    # Behavioural similarity
    token_overlap: float      # % common tokens
    activity_pattern_similarity: float  # Behaviour similarity

    # Network position
    shared_counterparties: list[str]  # Common counterparties
    graph_distance: int       # Hops apart in network
```

### Confidence Factors

| Factor | Description |
|--------|-------------|
| `direct_link` | Direct transfer exists |
| `funding_link` | Same funding source |
| `behavioural_match` | Similar behaviour patterns |
| `timing_correlation` | Correlated timing |

---

## Evidence Package Builder

For convenience, use the `EvidencePackageBuilder`:

```python
from corpus import EvidencePackageBuilder, LabelDomain

builder = EvidencePackageBuilder()

# Build exit event evidence
evidence = builder.build(
    domain=LabelDomain.EXIT_EVENT,
    exit_id="exit_123",
    wallet="abc...",
    token_mint="xyz...",
    amount_tokens=1000.0,
    value_sol=5.5,
    counterparty="raydium_pool",
    transaction_signature="tx...",
    block_time="2024-01-15T10:30:00Z",
    program_id="675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    confidence_factors={"signature_match": 0.95},
)
```

---

## Storage Format

Evidence packages are stored as JSON in the `evidence_json` field:

```json
{
  "exit_id": "exit_123",
  "wallet": "abc...",
  "token_mint": "xyz...",
  "amount_tokens": 1000.0,
  "value_sol": 5.5,
  "counterparty": "raydium_pool",
  "transaction_signature": "tx...",
  "block_time": "2024-01-15T10:30:00Z",
  "program_id": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
  "confidence_factors": {
    "signature_match": 0.95,
    "program_confidence": 0.88
  }
}
```

## Best Practices

1. **Include all confidence factors** - Transparency in classification reasoning
2. **Capture raw data** - Store the actual values, not just derived scores
3. **Timestamp everything** - When was this evidence collected?
4. **Reference transactions** - Include signatures for verification
5. **Keep summaries concise** - Human reviewers need quick context
