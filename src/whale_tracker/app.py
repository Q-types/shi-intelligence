"""
SHI Whale Tracker Dashboard

Real-time whale monitoring with:
- Manual wallet input
- Auto-discovery by % supply threshold
- Live monitoring with st.fragment
- SWEENEE-identical visualizations
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Add parent paths for imports
_tracker_dir = Path(__file__).parent
_src_dir = _tracker_dir.parent
_shi_dir = _src_dir.parent
sys.path.insert(0, str(_shi_dir))
sys.path.insert(0, str(_src_dir))

import structlog

from whale_tracker.config import config, DATABASE_PATH
from whale_tracker.classification import (
    WhaleTier,
    WhaleProfile,
    classify_whales,
    detect_tier_transitions,
)
from whale_tracker.discovery import (
    WhaleDiscovery,
    DiscoveryConfig,
    parse_wallet_input,
    is_valid_solana_address,
)
from whale_tracker.live_monitor import LiveMonitor, render_streaming_indicator

# Import shared modules
from shared.cache import get_cache
from shared.solana_client import get_client
from shared.token_balances import fetch_all_balances, compute_balance_summary
from shared.transactions import fetch_all_transactions
from shared.alerts import AlertService, render_alert_banners
from shared.history import SnapshotService, render_historical_chart
from shared.metrics import compute_dashboard_metrics, compute_wallet_flows
from shared.export import export_wallets_csv, export_wallets_json

logger = structlog.get_logger()

# Page config
st.set_page_config(
    page_title="SHI Whale Tracker",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Helper Functions ---

def short_address(addr: str) -> str:
    """Truncate address for display."""
    if len(addr) <= 12:
        return addr
    return f"{addr[:4]}...{addr[-4:]}"


def format_number(n: float, decimals: int = 0) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,.{decimals}f}"


def run_async(coro):
    """Run async coroutine in Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# --- Visualization Functions ---

def render_tier_badge(tier: WhaleTier) -> str:
    """Render a tier badge as HTML."""
    return f"""
    <span style="
        background: {tier.color}20;
        color: {tier.color};
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    ">{tier.emoji} {tier.display_name}</span>
    """


def render_conviction_gauge(score: float) -> go.Figure:
    """Render a conviction score gauge (0-100)."""
    if score >= 70:
        color = "#4CAF50"  # Green
    elif score >= 40:
        color = "#FFC107"  # Yellow
    else:
        color = "#F44336"  # Red

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(244, 67, 54, 0.2)"},
                {"range": [40, 70], "color": "rgba(255, 193, 7, 0.2)"},
                {"range": [70, 100], "color": "rgba(76, 175, 80, 0.2)"},
            ],
        },
        number={"suffix": "%", "font": {"size": 40}},
    ))

    fig.update_layout(
        height=200,
        margin={"t": 20, "b": 20, "l": 20, "r": 20},
    )
    return fig


def render_tier_distribution(profiles: list[WhaleProfile]) -> go.Figure:
    """Render pie chart of whale tier distribution."""
    tier_counts = {}
    for tier in WhaleTier:
        count = len([p for p in profiles if p.tier == tier])
        if count > 0:
            tier_counts[tier] = count

    if not tier_counts:
        return None

    fig = go.Figure(data=[go.Pie(
        labels=[t.display_name for t in tier_counts.keys()],
        values=list(tier_counts.values()),
        marker_colors=[t.color for t in tier_counts.keys()],
        hole=0.4,
        textinfo="label+value",
        textposition="outside",
    )])

    fig.update_layout(
        height=300,
        margin={"t": 20, "b": 20, "l": 20, "r": 20},
        showlegend=False,
    )
    return fig


def compute_whale_conviction(profiles: list[WhaleProfile], metrics) -> float:
    """
    Compute whale conviction score (0-100).

    Based on:
    - % of whales still holding (30 points)
    - Net flow direction (25 points)
    - Buy/sell ratio (25 points)
    - Concentration (HHI) (20 points)
    """
    if not profiles:
        return 0

    score = 0

    # Holding ratio (30 points)
    holding = len([p for p in profiles if p.balance > 0])
    holding_ratio = holding / len(profiles) if profiles else 0
    score += holding_ratio * 30

    # Net flow (25 points)
    if metrics and hasattr(metrics, 'net_flow_24h'):
        if metrics.net_flow_24h > 0:
            score += 25
        elif metrics.net_flow_24h == 0:
            score += 12.5

    # Buy/sell ratio (25 points)
    if metrics and hasattr(metrics, 'buys_24h') and hasattr(metrics, 'sells_24h'):
        total_txs = metrics.buys_24h + metrics.sells_24h
        if total_txs > 0:
            buy_ratio = metrics.buys_24h / total_txs
            score += buy_ratio * 25

    # Concentration (20 points) - higher HHI = more concentrated = higher conviction
    if metrics and hasattr(metrics, 'hhi'):
        # HHI ranges from 0 to 10000, normalize to 0-1
        hhi_normalized = min(metrics.hhi / 5000, 1)  # Cap at 5000 for scoring
        score += hhi_normalized * 20

    return min(score, 100)


# --- Data Loading ---

def fetch_whale_data(wallets: list[str], token_mint: str):
    """Fetch balances and transactions for tracked wallets."""
    cache = get_cache(DATABASE_PATH)

    # Check cache first
    cached = cache.get_cached_balances(token_mint, max_age_seconds=config.balance_cache_ttl)
    if cached:
        # Filter to our wallets
        wallet_set = set(wallets)
        balances = [b for b in cached if b.address in wallet_set]
        if balances:
            transactions = cache.get_transactions(token_mint, hours=168)
            return balances, transactions

    # Fetch fresh data
    async def _fetch():
        client = get_client()
        balances = await fetch_all_balances(client, wallets, token_mint, cache)
        transactions = await fetch_all_transactions(client, wallets, token_mint, cache)
        return balances, transactions

    return run_async(_fetch())


# --- Main App ---

def main():
    """Main whale tracker dashboard."""

    # Initialize session state
    LiveMonitor.init_session_state()

    if "token_mint" not in st.session_state:
        st.session_state.token_mint = config.default_token_mint
    if "discovery_mode" not in st.session_state:
        st.session_state.discovery_mode = "manual"
    if "threshold_pct" not in st.session_state:
        st.session_state.threshold_pct = config.default_threshold_pct
    if "previous_profiles" not in st.session_state:
        st.session_state.previous_profiles = []

    # --- Sidebar ---
    with st.sidebar:
        st.title("🐋 Whale Tracker")
        st.caption("SHI Intelligence System")

        st.divider()

        # Token mint input
        st.subheader("Token")
        token_mint = st.text_input(
            "Token Mint Address",
            value=st.session_state.token_mint,
            placeholder="Enter token mint address...",
        )
        if token_mint != st.session_state.token_mint:
            st.session_state.token_mint = token_mint
            st.session_state.tracked_wallets = []
            st.session_state.whale_profiles = []

        st.divider()

        # Discovery mode
        st.subheader("Wallet Discovery")
        mode = st.radio(
            "Discovery Mode",
            options=["Manual Input", "Auto-Discovery"],
            index=0 if st.session_state.discovery_mode == "manual" else 1,
            horizontal=True,
        )
        st.session_state.discovery_mode = "manual" if mode == "Manual Input" else "auto"

        if st.session_state.discovery_mode == "manual":
            # Manual wallet input
            wallet_input = st.text_area(
                "Wallet Addresses",
                placeholder="Paste wallet addresses (one per line or comma-separated)",
                height=150,
            )
            if st.button("Add Wallets", type="primary", use_container_width=True):
                new_wallets = parse_wallet_input(wallet_input)
                if new_wallets:
                    existing = set(st.session_state.tracked_wallets)
                    added = [w for w in new_wallets if w not in existing]
                    st.session_state.tracked_wallets.extend(added)
                    st.success(f"Added {len(added)} wallets")
                    st.rerun()
                else:
                    st.warning("No valid addresses found")
        else:
            # Auto-discovery settings
            threshold = st.slider(
                "Supply Threshold (%)",
                min_value=0.1,
                max_value=5.0,
                value=st.session_state.threshold_pct,
                step=0.1,
                help="Track all wallets holding >= this % of supply",
            )
            st.session_state.threshold_pct = threshold

            if st.button("Discover Whales", type="primary", use_container_width=True):
                st.info("Auto-discovery requires holder data from SHI analysis...")
                # TODO: Connect to SHI orchestrator for holder data

        st.divider()

        # Live monitoring controls
        refresh_interval = LiveMonitor.render_controls()

        st.divider()

        # Tracked wallets summary
        wallet_count = len(st.session_state.tracked_wallets)
        st.metric("Tracked Wallets", wallet_count)

        if wallet_count > 0 and st.button("Clear All Wallets", type="secondary"):
            st.session_state.tracked_wallets = []
            st.session_state.whale_profiles = []
            st.rerun()

    # --- Main Content ---

    # Streaming indicator
    render_streaming_indicator()

    # Header
    st.title("🐋 SHI Whale Tracker")

    if not st.session_state.token_mint:
        st.warning("Please enter a token mint address in the sidebar to begin tracking.")
        return

    if not st.session_state.tracked_wallets:
        st.info("No wallets being tracked. Add wallets using the sidebar.")

        # Quick start guide
        with st.expander("Quick Start Guide"):
            st.markdown("""
            ### How to use the Whale Tracker

            1. **Enter Token Mint**: Paste the token's mint address in the sidebar
            2. **Add Wallets**: Choose a discovery mode:
               - **Manual**: Paste wallet addresses directly
               - **Auto-Discovery**: Set a % threshold to find whales automatically
            3. **Start Monitoring**: Click "Start" to enable live updates
            4. **Configure Refresh**: Choose update frequency (30s / 1m / 5m)

            ### Features
            - Real-time balance tracking
            - Whale tier classification (Ultra, Mega, Whale, Large, Standard)
            - Large movement alerts
            - Historical charts
            - Export to CSV/JSON
            """)
        return

    # Fetch data (with live refresh via fragment)
    @st.fragment(run_every=refresh_interval)
    def live_dashboard():
        """Auto-refreshing dashboard fragment."""
        LiveMonitor.record_refresh()

        with st.spinner("Fetching whale data..."):
            try:
                balances, transactions = fetch_whale_data(
                    st.session_state.tracked_wallets,
                    st.session_state.token_mint,
                )
            except Exception as e:
                st.error(f"Failed to fetch data: {e}")
                logger.error("fetch_failed", error=str(e))
                return

        if not balances:
            st.warning("No balance data available for tracked wallets.")
            return

        # Classify whales
        balance_dicts = [{"address": b.address, "balance": b.balance.ui_amount} for b in balances]
        profiles = classify_whales(balance_dicts, discovery_mode=st.session_state.discovery_mode)
        st.session_state.whale_profiles = profiles

        # Detect tier transitions
        transitions = detect_tier_transitions(st.session_state.previous_profiles, profiles)
        st.session_state.previous_profiles = profiles

        # Compute metrics
        metrics = compute_dashboard_metrics(balances, transactions)
        conviction = compute_whale_conviction(profiles, metrics)

        # Check for alerts
        cache = get_cache(DATABASE_PATH)
        alert_service = AlertService(
            large_move_threshold=config.large_move_threshold,
            exit_threshold=config.exit_threshold,
        )
        alert_service.cache = cache
        current_bals = {b.address: b.balance.ui_amount for b in balances}
        new_alerts = alert_service.check_transactions(transactions, current_bals)

        # --- Render Dashboard ---

        # Alert banners
        if new_alerts:
            st.markdown(render_alert_banners(new_alerts), unsafe_allow_html=True)

        # Tier transition banners
        for t in transitions[:3]:
            emoji = "⬆️" if t.is_promotion else "⬇️" if t.is_demotion else "🔄"
            color = "#4CAF50" if t.is_promotion else "#F44336" if t.is_demotion else "#2196F3"
            st.markdown(f"""
            <div style="background: {color}20; padding: 0.5rem 1rem; border-radius: 8px; margin: 0.25rem 0; border: 1px solid {color};">
                <span style="font-size: 1.2rem;">{emoji}</span>
                <span style="color: {color}; font-weight: 600; margin-left: 0.5rem;">{t.description}</span>
                <span style="color: #888; margin-left: 0.5rem;">({short_address(t.wallet_address)})</span>
            </div>
            """, unsafe_allow_html=True)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Tracked Whales",
                len(profiles),
                help="Total wallets being monitored",
            )
        with col2:
            holding = len([p for p in profiles if p.balance > 0])
            st.metric(
                "Currently Holding",
                f"{holding}/{len(profiles)}",
                help="Wallets with non-zero balance",
            )
        with col3:
            total = sum(p.balance for p in profiles)
            st.metric(
                "Total Tracked",
                format_number(total),
                help="Total tokens held by tracked wallets",
            )
        with col4:
            ultra_count = len([p for p in profiles if p.tier == WhaleTier.ULTRA_WHALE])
            mega_count = len([p for p in profiles if p.tier == WhaleTier.MEGA_WHALE])
            st.metric(
                "Top Whales",
                f"{ultra_count} Ultra / {mega_count} Mega",
                help="Ultra Whale (top 1%) and Mega Whale (top 5%)",
            )

        st.divider()

        # Conviction and tier distribution
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🎯 Whale Conviction")
            gauge_fig = render_conviction_gauge(conviction)
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Conviction explanation
            if conviction >= 70:
                st.success("High conviction - Whales are accumulating")
            elif conviction >= 40:
                st.warning("Medium conviction - Mixed signals")
            else:
                st.error("Low conviction - Whales may be distributing")

        with col2:
            st.subheader("📊 Tier Distribution")
            tier_fig = render_tier_distribution(profiles)
            if tier_fig:
                st.plotly_chart(tier_fig, use_container_width=True)
            else:
                st.info("No tier data available")

        st.divider()

        # Whale table
        st.subheader("🐋 Whale Profiles")

        # Tier filter
        tier_filter = st.multiselect(
            "Filter by Tier",
            options=[t.display_name for t in WhaleTier],
            default=[],
        )

        filtered_profiles = profiles
        if tier_filter:
            tier_values = [t for t in WhaleTier if t.display_name in tier_filter]
            filtered_profiles = [p for p in profiles if p.tier in tier_values]

        # Build table data
        table_data = []
        for p in filtered_profiles:
            table_data.append({
                "Rank": p.holder_rank,
                "Tier": f"{p.tier.emoji} {p.tier.display_name}",
                "Wallet": p.display_name,
                "Address": p.wallet_address,
                "Balance": format_number(p.balance),
                "% Share": f"{p.concentration_share:.2f}%",
                "Percentile": f"{p.percentile_rank:.1f}",
            })

        if table_data:
            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Address": st.column_config.TextColumn(width="medium"),
                },
            )
        else:
            st.info("No whales match the selected filters")

        st.divider()

        # Historical chart
        st.subheader("📈 Holdings Over Time")
        history_days = st.selectbox("Time Range", [7, 14, 30, 60, 90], index=2)

        snapshot_service = SnapshotService(cache)
        history = snapshot_service.get_total_history(
            st.session_state.token_mint,
            days=history_days,
        )

        if history:
            hist_chart = render_historical_chart(history, f"Total Holdings ({history_days}d)")
            if hist_chart:
                st.plotly_chart(hist_chart, use_container_width=True)
        else:
            st.info("Not enough historical data yet. Data will accumulate over time.")

        # Take daily snapshot
        snapshot_service.take_snapshot(balances, st.session_state.token_mint)

    # Run the live dashboard fragment
    live_dashboard()

    # --- Static Content (doesn't refresh) ---
    st.divider()

    # Export tools
    with st.expander("📥 Export Data"):
        col1, col2 = st.columns(2)

        profiles = st.session_state.whale_profiles

        with col1:
            if profiles:
                csv_data = export_wallets_csv([
                    {"address": p.wallet_address, "label": p.label, "balance": p.balance}
                    for p in profiles
                ])
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name=f"whale_tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

        with col2:
            if profiles:
                json_data = export_wallets_json([
                    p.to_dict() for p in profiles
                ])
                st.download_button(
                    "Download JSON",
                    data=json_data,
                    file_name=f"whale_tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )


if __name__ == "__main__":
    main()
