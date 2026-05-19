# SHI Whale Tracker Dashboard - Integration Plan

**Created:** 2026-05-18
**Status:** Planning Phase
**Architect Project ID:** 4fd53fed-1b79-49e4-946f-737e5777506a

---

## Executive Summary

The Whale Tracker is a modular dashboard component for SHI that enables users to monitor large token holders ("whales") in real-time. It provides two discovery modes, persistent live monitoring, and a visualization experience identical to the proven SWEENEE dashboard.

---

## Core Features

### 1. Wallet Discovery Modes

#### Mode A: Manual Input
- Users paste wallet addresses directly
- Support for bulk paste (one per line)
- Address validation with Solana address regex
- Optional labels/notes per wallet

#### Mode B: Auto-Discovery (% Supply Threshold)
- User sets threshold (e.g., "0.5% of supply")
- System queries SHI's holder analysis data
- Auto-identifies all wallets holding >= threshold
- Refreshes whale list on each analysis run
- Supports combining with manual additions

### 2. Real-Time Live Monitoring

Using Streamlit's `st.fragment` with `run_every` parameter:

```python
@st.fragment(run_every=refresh_interval if streaming else None)
def live_whale_monitor():
    """Auto-refreshing whale data fragment."""
    balances = fetch_whale_balances()
    render_whale_metrics(balances)
    render_whale_table(balances)
```

**Refresh Options:**
- 30 seconds (high frequency)
- 1 minute (default)
- 5 minutes (low frequency)
- Manual only (disabled auto-refresh)

**Session State Control:**
```python
# Toggle streaming on/off
if st.session_state.get("streaming", False):
    refresh_interval = st.session_state.get("refresh_seconds", 60)
else:
    refresh_interval = None
```

### 3. Dashboard Visualizations (SWEENEE-Identical)

| Component | Description |
|-----------|-------------|
| Hero Banner | Whale holding confidence score |
| Alert Banners | Recent large movement alerts |
| Metrics Cards | 8 key stats (total tracked, holding count, net flow, etc.) |
| Conviction Gauge | 0-100 whale conviction score |
| Buy/Sell Donut | Transaction type breakdown |
| Holdings Chart | Historical balance over time (Plotly area chart) |
| Whale Table | Sortable table with balances, shares, flows |
| Export Tools | CSV/JSON download buttons |

### 4. Alert System

- Large buy detection (configurable threshold)
- Large sell detection
- New whale appearance
- Whale exit (balance near zero)
- Tier transitions (e.g., promoted to Ultra Whale)

### 5. Whale Tier Classification

| Tier | Percentile | Description |
|------|------------|-------------|
| Ultra Whale | Top 1% | Largest holders |
| Mega Whale | Top 5% | Very large holders |
| Whale | Top 10% | Large holders |
| Large Holder | Top 25% | Above average |
| Standard | Below 75th | Normal holders |

---

## Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SHI Intelligence System                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐     ┌──────────────────────────────────┐  │
│  │   Main SHI App   │     │      Whale Tracker Dashboard     │  │
│  │   (Analysis)     │     │      (Separate Streamlit)        │  │
│  │                  │     │                                  │  │
│  │  - Holder scan   │────▶│  - Manual wallet input           │  │
│  │  - Risk scoring  │     │  - Auto-discovery (% threshold)  │  │
│  │  - Clustering    │     │  - Live monitoring (st.fragment) │  │
│  │                  │     │  - Alerts & notifications        │  │
│  └────────┬─────────┘     └──────────────┬───────────────────┘  │
│           │                              │                       │
│           ▼                              ▼                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Shared Data Layer                        ││
│  │                                                             ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ ││
│  │  │   SQLite    │  │   Solana    │  │     Telegram        │ ││
│  │  │   Cache     │  │   RPC       │  │     Webhook         │ ││
│  │  │             │  │   (Async)   │  │                     │ ││
│  │  │ - balances  │  │ - Helius    │  │ - Alert delivery    │ ││
│  │  │ - txns      │  │ - Rate ltd  │  │ - Daily summaries   │ ││
│  │  │ - alerts    │  │             │  │                     │ ││
│  │  │ - snapshots │  │             │  │                     │ ││
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
SHI/
├── src/
│   ├── whale_tracker/              # NEW: Whale Tracker module
│   │   ├── __init__.py
│   │   ├── app.py                  # Streamlit dashboard entry
│   │   ├── config.py               # Tracker-specific config
│   │   ├── discovery.py            # Auto-discovery logic
│   │   ├── classification.py       # Whale tier classification
│   │   ├── live_monitor.py         # st.fragment live updates
│   │   └── templates/
│   │       └── tracker.html        # Optional Jinja templates
│   │
│   ├── shared/                     # Shared components (from SWEENEE)
│   │   ├── cache.py                # SQLite cache with migrations
│   │   ├── solana_client.py        # Async RPC client
│   │   ├── token_balances.py       # Balance fetching
│   │   ├── transactions.py         # Transaction classification
│   │   ├── alerts.py               # Alert detection
│   │   ├── history.py              # Historical snapshots
│   │   ├── webhook.py              # Telegram integration
│   │   ├── export.py               # CSV/JSON export
│   │   └── metrics.py              # Dashboard metrics
│   │
│   └── ... (existing SHI modules)
│
├── data/
│   └── whale_tracker.sqlite        # Separate database for tracker
│
├── tests/
│   └── test_whale_tracker/
│       ├── test_discovery.py
│       ├── test_classification.py
│       └── test_live_monitor.py
│
└── docs/
    └── WHALE_TRACKER_PLAN.md       # This file
```

### Database Schema Extensions

```sql
-- New tables for Whale Tracker

-- Tracked whale wallets (separate from SHI analysis)
CREATE TABLE IF NOT EXISTS tracker_wallets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT NOT NULL UNIQUE,
    label TEXT,
    notes TEXT,
    discovery_mode TEXT NOT NULL,  -- 'manual' | 'auto'
    threshold_pct REAL,            -- % threshold if auto-discovered
    added_at TEXT NOT NULL,
    is_active INTEGER DEFAULT 1
);

-- Whale classification history
CREATE TABLE IF NOT EXISTS whale_classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_address TEXT NOT NULL,
    token_mint TEXT NOT NULL,
    tier TEXT NOT NULL,            -- 'ultra_whale', 'mega_whale', etc.
    balance_amount REAL NOT NULL,
    percentile_rank REAL NOT NULL,
    concentration_share REAL NOT NULL,
    holder_rank INTEGER NOT NULL,
    classified_at TEXT NOT NULL,
    FOREIGN KEY(wallet_address) REFERENCES tracker_wallets(address)
);

-- Tier transition alerts
CREATE TABLE IF NOT EXISTS tier_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_address TEXT NOT NULL,
    previous_tier TEXT,
    new_tier TEXT NOT NULL,
    balance_change REAL,
    transition_at TEXT NOT NULL,
    acknowledged INTEGER DEFAULT 0
);

-- Live monitoring sessions
CREATE TABLE IF NOT EXISTS monitoring_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_mint TEXT NOT NULL,
    wallet_count INTEGER NOT NULL,
    refresh_interval INTEGER NOT NULL,  -- seconds
    started_at TEXT NOT NULL,
    ended_at TEXT,
    total_refreshes INTEGER DEFAULT 0
);
```

### Key Classes

```python
# src/whale_tracker/classification.py

from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class WhaleTier(Enum):
    """Whale classification tiers based on percentile rank."""
    ULTRA_WHALE = "ultra_whale"      # Top 1%
    MEGA_WHALE = "mega_whale"        # Top 5%
    WHALE = "whale"                  # Top 10%
    LARGE_HOLDER = "large_holder"    # Top 25%
    STANDARD = "standard"            # Below 75th percentile

    @classmethod
    def from_percentile(cls, percentile: float) -> "WhaleTier":
        """Get tier from percentile rank (0-100, higher = larger holder)."""
        if percentile >= 99:
            return cls.ULTRA_WHALE
        elif percentile >= 95:
            return cls.MEGA_WHALE
        elif percentile >= 90:
            return cls.WHALE
        elif percentile >= 75:
            return cls.LARGE_HOLDER
        return cls.STANDARD


@dataclass
class WhaleProfile:
    """Complete profile of a whale wallet."""
    wallet_address: str
    label: str | None
    balance: float
    tier: WhaleTier
    percentile_rank: float          # 0-100
    concentration_share: float      # % of total supply
    holder_rank: int                # Rank among all holders
    discovery_mode: str             # 'manual' | 'auto'
    classified_at: datetime

    @property
    def tier_emoji(self) -> str:
        return {
            WhaleTier.ULTRA_WHALE: "🐋",
            WhaleTier.MEGA_WHALE: "🐳",
            WhaleTier.WHALE: "🐟",
            WhaleTier.LARGE_HOLDER: "🐠",
            WhaleTier.STANDARD: "🐡",
        }.get(self.tier, "❓")


@dataclass
class TierTransition:
    """Record of a wallet changing tiers."""
    wallet_address: str
    previous_tier: WhaleTier | None
    new_tier: WhaleTier
    balance_before: float
    balance_after: float
    transition_at: datetime

    @property
    def is_promotion(self) -> bool:
        """True if wallet moved to a higher tier."""
        tier_order = [WhaleTier.STANDARD, WhaleTier.LARGE_HOLDER,
                      WhaleTier.WHALE, WhaleTier.MEGA_WHALE, WhaleTier.ULTRA_WHALE]
        if self.previous_tier is None:
            return True
        return tier_order.index(self.new_tier) > tier_order.index(self.previous_tier)
```

```python
# src/whale_tracker/discovery.py

from dataclasses import dataclass

@dataclass
class DiscoveryConfig:
    """Configuration for whale auto-discovery."""
    threshold_pct: float            # % of supply to qualify as whale
    min_balance: float | None       # Optional absolute minimum
    exclude_known_contracts: bool   # Filter out DEX/protocol addresses
    include_dormant: bool           # Include wallets with no recent activity


class WhaleDiscovery:
    """Auto-discover whales based on supply threshold."""

    def __init__(self, config: DiscoveryConfig):
        self.config = config

    async def discover_whales(
        self,
        token_mint: str,
        holder_data: list[dict],  # From SHI holder analysis
    ) -> list[str]:
        """
        Find all wallets holding >= threshold % of supply.

        Args:
            token_mint: Token to analyze
            holder_data: Holder analysis from SHI pipeline

        Returns:
            List of wallet addresses qualifying as whales
        """
        total_supply = sum(h["balance"] for h in holder_data)
        threshold_balance = total_supply * (self.config.threshold_pct / 100)

        whales = []
        for holder in holder_data:
            if holder["balance"] >= threshold_balance:
                if self.config.min_balance and holder["balance"] < self.config.min_balance:
                    continue
                if self.config.exclude_known_contracts and holder.get("is_contract"):
                    continue
                whales.append(holder["address"])

        return whales
```

```python
# src/whale_tracker/live_monitor.py

import streamlit as st
from datetime import datetime

class LiveMonitor:
    """Manages live monitoring state and fragments."""

    REFRESH_OPTIONS = {
        "30 seconds": 30,
        "1 minute": 60,
        "5 minutes": 300,
        "Manual only": None,
    }

    @staticmethod
    def init_session_state():
        """Initialize session state for live monitoring."""
        if "streaming" not in st.session_state:
            st.session_state.streaming = False
        if "refresh_seconds" not in st.session_state:
            st.session_state.refresh_seconds = 60
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = None
        if "refresh_count" not in st.session_state:
            st.session_state.refresh_count = 0

    @staticmethod
    def render_controls():
        """Render streaming control UI in sidebar."""
        st.sidebar.subheader("Live Monitoring")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button(
                "Start" if not st.session_state.streaming else "Stop",
                type="primary" if not st.session_state.streaming else "secondary",
            ):
                st.session_state.streaming = not st.session_state.streaming

        with col2:
            selected = st.selectbox(
                "Refresh",
                options=list(LiveMonitor.REFRESH_OPTIONS.keys()),
                index=1,  # Default: 1 minute
                disabled=st.session_state.streaming,
            )
            st.session_state.refresh_seconds = LiveMonitor.REFRESH_OPTIONS[selected]

        if st.session_state.streaming:
            st.sidebar.success(f"Streaming every {st.session_state.refresh_seconds}s")
            st.sidebar.caption(f"Refreshes: {st.session_state.refresh_count}")

        return st.session_state.refresh_seconds if st.session_state.streaming else None
```

### Live Monitoring Pattern

```python
# src/whale_tracker/app.py

import streamlit as st
from datetime import datetime
from live_monitor import LiveMonitor

st.set_page_config(page_title="SHI Whale Tracker", layout="wide")

# Initialize
LiveMonitor.init_session_state()

# Sidebar controls
refresh_interval = LiveMonitor.render_controls()

# Main content with auto-refresh fragment
@st.fragment(run_every=refresh_interval)
def whale_dashboard():
    """Auto-refreshing whale monitoring dashboard."""

    # Update refresh tracking
    st.session_state.last_refresh = datetime.now()
    st.session_state.refresh_count += 1

    # Fetch fresh data
    balances = fetch_whale_balances(st.session_state.tracked_wallets)
    transactions = fetch_recent_transactions(st.session_state.tracked_wallets)

    # Compute metrics
    metrics = compute_whale_metrics(balances, transactions)

    # Render dashboard components
    render_confidence_banner(metrics)
    render_alert_banners(check_for_alerts(transactions))
    render_metrics_cards(metrics)
    render_conviction_gauge(metrics)
    render_whale_table(balances)
    render_historical_chart(st.session_state.tracked_wallets)

# Render the fragment
whale_dashboard()

# Static content (doesn't re-render on fragment refresh)
st.divider()
render_export_tools()
render_settings_panel()
```

---

## Integration with SHI

### Connecting to Holder Analysis

The Whale Tracker can pull holder data from SHI's existing analysis pipeline:

```python
# Integration point: src/pipeline/orchestrator.py

class AnalysisOrchestrator:
    async def get_holders_for_tracker(
        self,
        mint: str,
        threshold_pct: float,
    ) -> list[dict]:
        """
        Get holder data formatted for Whale Tracker.

        Called by Whale Tracker's auto-discovery mode.
        """
        # Run holder analysis if not cached
        result = await self.analyze(mint)

        # Extract holder data with balances
        holders = []
        for wallet in result.wallet_features:
            holders.append({
                "address": wallet.wallet_address,
                "balance": wallet.current_balance,
                "pct_of_supply": wallet.pct_of_supply,
                "archetype": wallet.archetype.value if wallet.archetype else None,
                "risk_score": wallet.risk_score,
            })

        return holders
```

### Shared Module Strategy

Move SWEENEE modules to `src/shared/` for reuse:

| Module | Current Location | Shared Location |
|--------|-----------------|-----------------|
| cache.py | sweenee/src/ | src/shared/cache.py |
| solana_client.py | sweenee/src/ | src/shared/solana_client.py |
| token_balances.py | sweenee/src/ | src/shared/token_balances.py |
| transactions.py | sweenee/src/ | src/shared/transactions.py |
| alerts.py | sweenee/src/ | src/shared/alerts.py |
| history.py | sweenee/src/ | src/shared/history.py |
| webhook.py | sweenee/src/ | src/shared/webhook.py |
| export.py | sweenee/src/ | src/shared/export.py |
| metrics.py | sweenee/src/ | src/shared/metrics.py |

---

## Implementation Phases

### Phase 1: Foundation (Sprint 1)
**Goal:** Set up project structure and core components

| Task | Files | Priority |
|------|-------|----------|
| Create whale_tracker module structure | `src/whale_tracker/` | P0 |
| Move SWEENEE modules to shared | `src/shared/` | P0 |
| Create tracker-specific config | `config.py` | P0 |
| Set up separate SQLite database | `data/whale_tracker.sqlite` | P0 |
| Implement WhaleProfile, WhaleTier classes | `classification.py` | P0 |

### Phase 2: Discovery & Classification (Sprint 2)
**Goal:** Build whale discovery and tier system

| Task | Files | Priority |
|------|-------|----------|
| Manual wallet input UI | `app.py` | P0 |
| Auto-discovery from SHI holder data | `discovery.py` | P0 |
| Whale tier classification | `classification.py` | P0 |
| Tier transition detection | `classification.py` | P1 |
| Integration with SHI orchestrator | `orchestrator.py` | P1 |

### Phase 3: Live Monitoring (Sprint 3)
**Goal:** Implement real-time dashboard

| Task | Files | Priority |
|------|-------|----------|
| st.fragment live monitoring | `live_monitor.py` | P0 |
| Session state management | `app.py` | P0 |
| Streaming controls UI | `app.py` | P0 |
| Refresh rate configuration | `config.py` | P1 |
| Monitoring session logging | `cache.py` | P2 |

### Phase 4: Dashboard Visualization (Sprint 4)
**Goal:** Full SWEENEE-identical dashboard

| Task | Files | Priority |
|------|-------|----------|
| Hero confidence banner | `app.py` | P0 |
| Metrics cards (8 metrics) | `app.py` | P0 |
| Conviction gauge | `app.py` | P0 |
| Whale table with tiers | `app.py` | P0 |
| Historical holdings chart | `app.py` | P1 |
| Buy/sell donut chart | `app.py` | P1 |

### Phase 5: Alerts & Integration (Sprint 5)
**Goal:** Complete alert system and polish

| Task | Files | Priority |
|------|-------|----------|
| Alert detection for whale movements | `alerts.py` | P0 |
| Tier transition alerts | `classification.py` | P1 |
| Telegram webhook integration | `webhook.py` | P1 |
| CSV/JSON export | `export.py` | P1 |
| Documentation | `README.md` | P2 |
| Test suite | `tests/` | P0 |

---

## Running the Whale Tracker

### Separate Dashboard Instance

```bash
# Start Whale Tracker (separate from main SHI)
cd /path/to/SHI
streamlit run src/whale_tracker/app.py --server.port 8503

# Main SHI dashboard (if running)
streamlit run src/api/dashboard.py --server.port 8501
```

### Environment Variables

```bash
# .env additions for Whale Tracker
WHALE_TRACKER_DB=/path/to/data/whale_tracker.sqlite
WHALE_TRACKER_REFRESH_DEFAULT=60
WHALE_TRACKER_ALERT_THRESHOLD=1000000
TELEGRAM_BOT_TOKEN=xxx  # Shared with SWEENEE
TELEGRAM_CHAT_ID=xxx    # Shared with SWEENEE
```

---

## Research: Live Dashboard Patterns

### Streamlit st.fragment (Recommended)

From [Streamlit Docs](https://docs.streamlit.io/develop/tutorials/execution-flow/start-and-stop-fragment-auto-reruns):

```python
# Key pattern: Dynamic run_every based on session state
if st.session_state.stream is True:
    run_every = st.session_state.run_every
else:
    run_every = None

@st.fragment(run_every=run_every)
def show_latest_data():
    # Only this fragment reruns, not the entire app
    data = fetch_new_data()
    st.line_chart(data)
```

**Requirements:** `streamlit>=1.37.0`

**Advantages:**
- Native Streamlit support
- Efficient (only fragment reruns)
- Session state preserves context
- User can toggle on/off

### Alternative: Background Worker + Cache

For scenarios requiring data fetching independent of UI:

```python
# Background data fetcher (runs in separate thread/process)
import threading
import time

class BackgroundFetcher:
    def __init__(self, cache, interval=60):
        self.cache = cache
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _fetch_loop(self):
        while self.running:
            try:
                data = fetch_whale_data()
                self.cache.update(data)
            except Exception as e:
                logger.error("fetch_failed", error=str(e))
            time.sleep(self.interval)
```

**Use case:** When you need data freshness even when no user is viewing the dashboard.

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Wallet tracking accuracy | 100% | All added wallets tracked |
| Auto-discovery accuracy | >95% | Whales above threshold detected |
| Dashboard load time | <2s | Initial page render |
| Live refresh latency | <5s | Time from data change to UI update |
| Alert delivery time | <30s | Telegram notification delay |
| Test coverage | >80% | pytest-cov report |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Solana RPC rate limits | Data delays | Implement caching, use Helius dedicated endpoint |
| Large wallet count performance | Slow UI | Pagination, lazy loading, optimize queries |
| st.fragment browser issues | Broken live updates | Fallback to manual refresh button |
| Database locking | Concurrent access errors | Use WAL mode, connection pooling |
| Memory leaks in long sessions | Crashes | Trim historical data, session cleanup |

---

## Sources & References

- [Streamlit st.fragment Documentation](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment)
- [Start and Stop Fragment Auto-Reruns Tutorial](https://docs.streamlit.io/develop/tutorials/execution-flow/start-and-stop-fragment-auto-reruns)
- [Streamlit Auto Refresh Guide](https://www.restack.io/docs/streamlit-knowledge-streamlit-auto-refresh-guide)
- [Background Cache Refresh Discussion](https://discuss.streamlit.io/t/background-cache-refresh-to-avoid-users-waiting/27639)
- [APScheduler Integration Discussion](https://discuss.streamlit.io/t/is-it-possible-to-include-a-kind-of-scheduler-within-streamlit/31279)

---

## Next Steps

1. **Review this plan** - Confirm architecture and phasing
2. **Create Sprint 1** - Start with foundation tasks
3. **Move shared modules** - Refactor SWEENEE for reuse
4. **Build discovery mode** - Manual + auto-discovery
5. **Implement live monitoring** - st.fragment pattern
6. **Deploy dashboard** - Separate port from main SHI

---

*Plan generated with Architect MCP, IdeaRalph MCP, and Mind MCP*
