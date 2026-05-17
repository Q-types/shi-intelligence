"""SWEENEE Whale Wallet Dashboard - Streamlit Application.

A community-facing dashboard for tracking whale wallet activity
for the SWEENEE token on Solana.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure sweenee directory is in Python path for imports
_sweenee_dir = Path(__file__).parent.resolve()
if str(_sweenee_dir) not in sys.path:
    sys.path.insert(0, str(_sweenee_dir))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page
st.set_page_config(
    page_title="SWEENEE Whale Watch",
    page_icon="🐳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Display settings
SHOW_WALLET_LABELS = False  # Set to True to show whale names instead of addresses

# Import local modules
from config import config, SWEENEE_MINT, WALLETS_DIR, DATABASE_PATH
from src.wallet_loader import load_all_wallets, TrackedWallet
from src.solana_client import SolanaClient, get_client
from src.token_balances import fetch_all_balances, WalletBalance
from src.transactions import fetch_all_transactions, SweeneeTransaction, TransactionType
from src.cache import get_cache, SweeneeCache
from src.metrics import compute_dashboard_metrics, compute_wallet_flows, DashboardMetrics
from src.telegram_summary import generate_daily_summary, generate_weekly_summary
from src.history import SnapshotService, render_historical_chart
from src.alerts import AlertService, render_alert_banners
from src.export import (
    export_wallets_csv, export_wallets_json,
    export_transactions_csv, export_transactions_json,
    get_export_filename,
)


# --- Styling ---
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #888;
        font-style: italic;
        text-align: center;
        padding: 1rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
    .wallet-badge {
        background: #E3F2FD;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .flow-positive { color: #4CAF50; font-weight: bold; }
    .flow-negative { color: #F44336; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Helper Functions ---
def format_number(n: float, decimals: int = 0) -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(n) >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    elif abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.1f}K"
    elif decimals > 0:
        return f"{n:,.{decimals}f}"
    return f"{n:,.0f}"


def format_flow(n: float) -> str:
    """Format flow with color indicator."""
    if n > 0:
        return f"+{format_number(n)}"
    return format_number(n)


def compute_accumulation_days(transactions: list) -> int:
    """Count consecutive days of net positive flow."""
    if not transactions:
        return 0

    from collections import defaultdict
    daily_flow = defaultdict(float)

    for tx in transactions:
        if tx.block_time:
            day = tx.block_time.date()
            daily_flow[day] += tx.amount_change

    if not daily_flow:
        return 0

    today = datetime.now(timezone.utc).date()
    streak = 0

    for i in range(30):
        check_date = today - timedelta(days=i)
        if check_date in daily_flow:
            if daily_flow[check_date] > 0:
                streak += 1
            elif daily_flow[check_date] < 0:
                break

    return streak


def compute_market_phase(metrics) -> tuple[str, str, str]:
    """Determine accumulation/distribution phase."""
    net_7d = metrics.net_flow_7d
    net_24h = metrics.net_flow_24h

    if net_7d > 0 and net_24h >= 0:
        return ("Accumulation", "📈", "#4CAF50")
    elif net_7d > 0 and net_24h < 0:
        return ("Accumulating", "📊", "#8BC34A")
    elif net_7d < 0 and net_24h <= 0:
        return ("Distribution", "📉", "#F44336")
    elif net_7d < 0 and net_24h > 0:
        return ("Mixed", "📊", "#FF9800")
    else:
        return ("Neutral", "➡️", "#9E9E9E")


def render_confidence_banner(metrics, transactions: list):
    """Render the hero confidence banner."""
    all_holding = metrics.wallets_holding == metrics.total_tracked_wallets
    net_positive_7d = metrics.net_flow_7d >= 0

    recent_buys = sum(1 for tx in transactions
                     if tx.classification.value == "buy"
                     and tx.block_time
                     and (datetime.now(timezone.utc) - tx.block_time).total_seconds() < 86400)

    if all_holding and net_positive_7d:
        icon = "💎"
        color = "#4CAF50"
        message = f"All {metrics.total_tracked_wallets} tracked whales still holding strong"
        submessage = f"+{format_number(metrics.net_flow_7d)} SWEENEE net inflow this week"
    elif all_holding:
        icon = "🐳"
        color = "#2196F3"
        message = f"All {metrics.total_tracked_wallets} tracked whales still holding"
        submessage = "Positions stable"
    else:
        icon = "📊"
        color = "#9E9E9E"
        message = f"{metrics.wallets_holding} of {metrics.total_tracked_wallets} whales holding"
        submessage = "Monitor for changes"

    if recent_buys > 0:
        submessage += f" • {recent_buys} wallet{'s' if recent_buys > 1 else ''} bought today"

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}33 0%, {color}11 100%);
        border-left: 4px solid {color};
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid {color}44;
    ">
        <div style="font-size: 1.5rem; font-weight: 700; color: {color};">
            {icon} {message}
        </div>
        <div style="font-size: 1rem; color: #aaa; margin-top: 0.25rem;">
            {submessage}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_buy_alerts(transactions: list, wallets: list, threshold: float = 500_000):
    """Show prominent alerts for recent large buys."""
    now = datetime.now(timezone.utc)
    large_buys = []

    for tx in transactions:
        if (tx.classification.value == "buy" and
            tx.amount_change >= threshold and
            tx.block_time and
            (now - tx.block_time).total_seconds() < 86400):

            label = None
            for w in wallets:
                if w["address"] == tx.wallet_address:
                    label = w.get("label")
                    break

            large_buys.append({
                "wallet": get_wallet_display(tx.wallet_address, label),
                "amount": tx.amount_change,
                "time": tx.block_time,
            })

    if not large_buys:
        return

    large_buys.sort(key=lambda x: x["amount"], reverse=True)

    for buy in large_buys[:3]:
        hours_ago = (now - buy["time"]).total_seconds() / 3600
        time_str = f"{hours_ago:.0f}h ago" if hours_ago >= 1 else "Just now"

        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #4CAF5022 0%, transparent 100%);
            border-left: 3px solid #4CAF50;
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
        ">
            <span style="font-size: 1.1rem; margin-right: 0.5rem;">🐳</span>
            <span style="font-weight: 600; color: #4CAF50;">
                {buy["wallet"]} bought +{format_number(buy["amount"])} SWEENEE
            </span>
            <span style="color: #888; margin-left: 1rem; font-size: 0.85rem;">
                {time_str}
            </span>
        </div>
        """, unsafe_allow_html=True)


def generate_plain_english_summary(metrics, transactions: list) -> str:
    """Generate simple summary for non-technical users."""
    parts = []

    if metrics.wallets_holding == metrics.total_tracked_wallets:
        parts.append(f"✅ All {metrics.total_tracked_wallets} tracked whale wallets are still holding SWEENEE.")
    else:
        parts.append(f"📊 {metrics.wallets_holding} out of {metrics.total_tracked_wallets} whale wallets are holding SWEENEE.")

    if metrics.net_flow_7d > 0:
        parts.append(f"📈 This week, whales bought more than they sold (+{format_number(metrics.net_flow_7d)} net).")
    elif metrics.net_flow_7d < 0:
        parts.append(f"📉 This week, whales sold more than they bought ({format_number(metrics.net_flow_7d)} net).")
    else:
        parts.append("➡️ This week, whale buying and selling is balanced.")

    if metrics.transaction_count_24h == 0:
        parts.append("😴 No whale movements in the last 24 hours.")
    elif metrics.net_flow_24h > 0:
        parts.append(f"🟢 Last 24h: {metrics.transaction_count_24h} transactions, net buying.")
    elif metrics.net_flow_24h < 0:
        parts.append(f"🔴 Last 24h: {metrics.transaction_count_24h} transactions, net selling.")
    else:
        parts.append(f"⚪ Last 24h: {metrics.transaction_count_24h} transactions, balanced.")

    return " ".join(parts)


def compute_conviction_score(metrics, transactions: list) -> tuple[int, str]:
    """Compute whale conviction score 0-100."""
    score = 0

    # Factor 1: Holding ratio (30 points)
    holding_ratio = metrics.wallets_holding / max(metrics.total_tracked_wallets, 1)
    score += int(holding_ratio * 30)

    # Factor 2: 7d flow (25 points)
    if metrics.net_flow_7d > 0:
        score += 25
    elif metrics.net_flow_7d == 0:
        score += 12

    # Factor 3: 24h flow (20 points)
    if metrics.net_flow_24h > 0:
        score += 20
    elif metrics.net_flow_24h == 0:
        score += 10

    # Factor 4: Buy/sell ratio (25 points)
    buys = sum(1 for tx in transactions if tx.classification.value == "buy")
    sells = sum(1 for tx in transactions if tx.classification.value == "sell")
    total = buys + sells
    if total > 0:
        buy_ratio = buys / total
        score += int(buy_ratio * 25)
    else:
        score += 12

    # Description
    if score >= 80:
        desc = "Very Strong 💎"
    elif score >= 60:
        desc = "Strong 🐳"
    elif score >= 40:
        desc = "Moderate 📊"
    elif score >= 20:
        desc = "Weak ⚠️"
    else:
        desc = "Very Weak 🔴"

    return (min(score, 100), desc)


def render_conviction_gauge(score: int, desc: str):
    """Render a conviction gauge using Plotly."""
    if score >= 70:
        bar_color = "#4CAF50"
    elif score >= 40:
        bar_color = "#FF9800"
    else:
        bar_color = "#F44336"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Whale Conviction", "font": {"size": 16}},
        number={"suffix": "", "font": {"size": 36, "color": bar_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#ddd",
            "steps": [
                {"range": [0, 40], "color": "#FFEBEE"},
                {"range": [40, 70], "color": "#FFF3E0"},
                {"range": [70, 100], "color": "#E8F5E9"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=200,
        margin={"t": 40, "b": 0, "l": 30, "r": 30},
    )
    return fig


def render_buy_sell_donut(transactions: list):
    """Render buy/sell ratio donut chart."""
    buys = sum(1 for tx in transactions if tx.classification.value == "buy")
    sells = sum(1 for tx in transactions if tx.classification.value == "sell")
    transfers = len(transactions) - buys - sells

    if buys + sells == 0:
        return None

    buy_pct = (buys / (buys + sells)) * 100 if (buys + sells) > 0 else 0

    fig = go.Figure(data=[go.Pie(
        labels=["Buys", "Sells", "Transfers"],
        values=[buys, sells, transfers],
        hole=0.6,
        marker_colors=["#4CAF50", "#F44336", "#9E9E9E"],
        textinfo="label+percent",
        textposition="outside",
    )])

    fig.add_annotation(
        text=f"<b>{buy_pct:.0f}%</b><br>Buying",
        x=0.5, y=0.5,
        font_size=16,
        showarrow=False,
        font_color="#4CAF50" if buy_pct >= 50 else "#F44336",
    )

    fig.update_layout(
        height=250,
        showlegend=False,
        margin={"t": 20, "b": 20, "l": 20, "r": 20},
    )
    return fig


def render_activity_heatmap(transactions: list, wallets: list):
    """Render whale activity heatmap (wallet x day)."""
    if not transactions:
        return None

    # Get wallet display names
    wallet_labels = {w["address"]: get_wallet_display(w["address"], w.get("label")) for w in wallets}

    # Build activity matrix
    from collections import defaultdict
    activity = defaultdict(lambda: defaultdict(float))

    for tx in transactions:
        if tx.block_time:
            day = tx.block_time.strftime("%m/%d")
            wallet = wallet_labels.get(tx.wallet_address, short_address(tx.wallet_address))
            activity[wallet][day] += tx.amount_change

    if not activity:
        return None

    # Get unique days and wallets
    all_days = sorted(set(day for w in activity.values() for day in w.keys()))[-7:]  # Last 7 days
    all_wallets = list(activity.keys())[:15]  # Top 15 wallets

    # Build matrix
    z = []
    for wallet in all_wallets:
        row = []
        for day in all_days:
            val = activity[wallet].get(day, 0)
            # Normalize: positive = green (1), negative = red (-1), zero = gray (0)
            if val > 0:
                row.append(1)
            elif val < 0:
                row.append(-1)
            else:
                row.append(0)
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=all_days,
        y=all_wallets,
        colorscale=[
            [0, "#F44336"],      # Red for selling
            [0.5, "#E0E0E0"],    # Gray for no activity
            [1, "#4CAF50"],      # Green for buying
        ],
        zmid=0,
        showscale=False,
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:+}<extra></extra>",
    ))

    fig.update_layout(
        height=300,
        margin={"t": 20, "b": 40, "l": 100, "r": 20},
        xaxis_title="Date",
        yaxis_title="",
    )
    return fig


def short_address(addr: str) -> str:
    """Truncate address for display."""
    if len(addr) <= 12:
        return addr
    return f"{addr[:4]}...{addr[-4:]}"


def get_wallet_display(address: str, label: str | None = None) -> str:
    """Get wallet display name based on settings."""
    if SHOW_WALLET_LABELS and label:
        return label
    return short_address(address)


def run_async(coro):
    """Run async coroutine in Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# --- Data Loading ---
@st.cache_data(ttl=60)
def load_wallets() -> list[dict]:
    """Load tracked wallets from files."""
    wallets = load_all_wallets(WALLETS_DIR)
    return [{"address": w.address, "label": w.label, "notes": w.notes} for w in wallets]


def fetch_data(wallets: list[dict], use_cache: bool = True):
    """Fetch balances and transactions."""
    cache = get_cache(DATABASE_PATH)

    # Check cache first
    if use_cache:
        cached_balances = cache.get_cached_balances(
            SWEENEE_MINT, max_age_seconds=config.balance_cache_ttl
        )
        if cached_balances:
            # Enrich with labels
            wallet_labels = {w["address"]: w.get("label") for w in wallets}
            for bal in cached_balances:
                bal.label = wallet_labels.get(bal.address)

            cached_txs = cache.get_cached_transactions(SWEENEE_MINT, hours=168)
            return cached_balances, cached_txs

    # Fetch fresh data
    tracked = [
        TrackedWallet(address=w["address"], label=w.get("label"))
        for w in wallets
    ]

    async def fetch():
        client = get_client()
        try:
            balances = await fetch_all_balances(tracked, SWEENEE_MINT, client)
            addresses = [w["address"] for w in wallets]
            txs = await fetch_all_transactions(
                addresses, SWEENEE_MINT, limit_per_wallet=20, client=client
            )
            return balances, txs
        finally:
            await client.close()

    balances, transactions = run_async(fetch())

    # Save to cache
    cache.save_balances(balances)
    cache.save_transactions(transactions)

    return balances, transactions


# --- Main Dashboard ---
def main():
    # Header
    st.markdown('<h1 class="main-header">🐳 SWEENEE Whale Wallet Watch</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sub-header">Token: <code>{SWEENEE_MINT[:8]}...{SWEENEE_MINT[-8:]}</code></p>',
        unsafe_allow_html=True,
    )

    # Load wallets first so we can use them in banners
    wallets = load_wallets()

    if not wallets:
        st.warning(
            f"No wallet files found in {WALLETS_DIR}. "
            "Add wallet addresses to track in .txt, .csv, or .json format."
        )
        st.info(
            "**Quick start:** Create a file `wallets/whales.txt` with one wallet address per line."
        )
        return

    # Refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info(f"📊 Tracking **{len(wallets)}** whale wallets")
    with col2:
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    with col3:
        use_cache = st.checkbox("Use cache", value=True)

    # Fetch data
    with st.spinner("Fetching wallet data..."):
        try:
            balances, transactions = fetch_data(wallets, use_cache=use_cache)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return

    # Compute metrics
    metrics = compute_dashboard_metrics(balances, transactions)
    wallet_flows = compute_wallet_flows(balances, transactions)

    # Save dashboard run for historical tracking
    cache = get_cache(DATABASE_PATH)
    cache.save_dashboard_run(
        wallet_count=metrics.total_tracked_wallets,
        total_balance=metrics.total_sweenee,
        transaction_count=metrics.transaction_count_7d,
        summary={
            "net_flow_24h": metrics.net_flow_24h,
            "net_flow_7d": metrics.net_flow_7d,
            "wallets_holding": metrics.wallets_holding,
        }
    )

    # Save daily balance snapshots for historical charts
    snapshot_service = SnapshotService(SWEENEE_MINT)
    snapshot_service.take_snapshot(balances)

    # --- HERO CONFIDENCE BANNER ---
    render_confidence_banner(metrics, transactions)

    # --- BUY ALERTS ---
    render_buy_alerts(transactions, wallets, threshold=500_000)

    # Last updated
    st.caption(f"Last updated: {metrics.balance_fetched_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # --- PLAIN ENGLISH SUMMARY ---
    summary = generate_plain_english_summary(metrics, transactions)
    st.info(summary)

    st.divider()

    # --- Summary Metrics ---
    st.subheader("📊 Whale Status")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "🐳 Whales Monitored",
            metrics.total_tracked_wallets,
            f"💎 {metrics.wallets_holding} diamond hands",
        )
    with m2:
        st.metric(
            "💰 Whale Holdings",
            format_number(metrics.total_sweenee),
        )
    with m3:
        if metrics.net_flow_24h >= 0:
            label = "🟢 24h Net Buying"
            delta_color = "normal"
        else:
            label = "🔴 24h Net Selling"
            delta_color = "inverse"
        st.metric(
            label,
            format_flow(metrics.net_flow_24h),
            f"{metrics.transaction_count_24h} moves",
            delta_color=delta_color,
        )
    with m4:
        top_display = get_wallet_display(
            metrics.largest_holder_address or "",
            metrics.largest_holder_label
        )
        st.metric(
            "👑 Top Wallet",
            format_number(metrics.largest_holder_balance),
            top_display,
        )

    m5, m6, m7, m8 = st.columns(4)
    with m5:
        # Phase indicator
        phase, phase_emoji, phase_color = compute_market_phase(metrics)
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem;">
            <div style="font-size: 0.85rem; color: #666;">Whale Phase</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: {phase_color};">
                {phase_emoji} {phase}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m6:
        # Accumulation streak instead of HHI
        streak = compute_accumulation_days(transactions)
        if streak > 0:
            st.metric(
                "🔥 Buy Streak",
                f"{streak} days",
                "Net accumulation",
            )
        else:
            st.metric(
                "📊 Activity",
                "Mixed",
                "Monitor trend",
            )
    with m7:
        if metrics.net_flow_7d >= 0:
            st.metric(
                "📈 7d Net Buying",
                format_flow(metrics.net_flow_7d),
                f"{metrics.transaction_count_7d} moves",
            )
        else:
            st.metric(
                "📉 7d Net Selling",
                format_flow(metrics.net_flow_7d),
                f"{metrics.transaction_count_7d} moves",
                delta_color="inverse",
            )
    with m8:
        if metrics.largest_inflow_24h > 0:
            st.metric(
                "🐳 Biggest Buy 24h",
                f"+{format_number(metrics.largest_inflow_24h)}",
                short_address(metrics.largest_inflow_wallet or ""),
            )
        else:
            st.metric("🐳 Biggest Buy 24h", "—", "No large buys")

    st.divider()

    # --- Conviction & Summary Charts ---
    st.subheader("📊 Whale Conviction")

    conv_col1, conv_col2, conv_col3 = st.columns([1, 1, 1])

    with conv_col1:
        # Conviction Gauge
        score, desc = compute_conviction_score(metrics, transactions)
        gauge_fig = render_conviction_gauge(score, desc)
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.caption(f"Status: **{desc}**")

    with conv_col2:
        # Buy/Sell Donut
        st.markdown("**Transaction Breakdown**")
        donut_fig = render_buy_sell_donut(transactions)
        if donut_fig:
            st.plotly_chart(donut_fig, use_container_width=True)
        else:
            st.info("No buy/sell data yet")

    with conv_col3:
        # Quick Stats
        st.markdown("**Key Signals**")
        buys = sum(1 for tx in transactions if tx.classification.value == "buy")
        sells = sum(1 for tx in transactions if tx.classification.value == "sell")

        st.markdown(f"""
        <div style="background: rgba(76, 175, 80, 0.2); padding: 0.5rem; border-radius: 8px; margin: 0.25rem 0; border: 1px solid #4CAF50;">
            <span style="font-size: 1.2rem;">🟢</span>
            <span style="font-weight: 600; color: #4CAF50;">{buys} Buys</span>
        </div>
        <div style="background: rgba(244, 67, 54, 0.2); padding: 0.5rem; border-radius: 8px; margin: 0.25rem 0; border: 1px solid #F44336;">
            <span style="font-size: 1.2rem;">🔴</span>
            <span style="font-weight: 600; color: #F44336;">{sells} Sells</span>
        </div>
        <div style="background: rgba(33, 150, 243, 0.2); padding: 0.5rem; border-radius: 8px; margin: 0.25rem 0; border: 1px solid #2196F3;">
            <span style="font-size: 1.2rem;">💎</span>
            <span style="font-weight: 600; color: #2196F3;">{metrics.wallets_holding}/{metrics.total_tracked_wallets} Holding</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- Activity Heatmap ---
    st.subheader("🔥 Whale Activity Heatmap")
    st.caption("Green = buying, Red = selling, Gray = no activity")

    heatmap_fig = render_activity_heatmap(transactions, wallets)
    if heatmap_fig:
        st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.info("Not enough transaction data for heatmap")

    st.divider()

    # --- Historical Holdings Chart ---
    st.subheader("📈 Holdings Over Time")

    hist_col1, hist_col2 = st.columns([3, 1])
    with hist_col2:
        history_days = st.selectbox("Time Range", [7, 14, 30, 60, 90], index=2)
        show_individual = st.checkbox("Show by Wallet", value=False)

    # Create wallet label mapping for chart
    wallet_labels = {w.address: w.label or w.address[:8] for w in wallets}

    history_chart = render_historical_chart(
        SWEENEE_MINT,
        days=history_days,
        show_individual=show_individual,
        wallet_labels=wallet_labels if show_individual else None,
    )

    if history_chart:
        st.plotly_chart(history_chart, use_container_width=True)
    else:
        st.info("📊 Historical data will appear after a few days of tracking. Check back soon!")

    st.divider()

    # --- Balance Charts ---
    st.subheader("📊 Current Holdings")

    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("**🐳 Top Whale Holdings**")
        top_wallets = balances[:15]
        if top_wallets:
            df_top = pd.DataFrame([
                {
                    "Wallet": get_wallet_display(b.address, b.label),
                    "Balance": b.ui_amount,
                    "Address": b.address,
                }
                for b in top_wallets
            ])
            fig = px.bar(
                df_top,
                x="Balance",
                y="Wallet",
                orientation="h",
                color="Balance",
                color_continuous_scale=[[0, "#81C784"], [0.5, "#4CAF50"], [1, "#2E7D32"]],
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    with chart2:
        st.markdown("**📊 24h Net Flow by Whale**")
        flows_data = []
        for bal in balances[:15]:
            flow = wallet_flows.get(bal.address, {}).get("net_24h", 0)
            flows_data.append({
                "Wallet": get_wallet_display(bal.address, bal.label),
                "Net Flow": flow,
                "Color": "🟢 Buying" if flow > 0 else ("🔴 Selling" if flow < 0 else "⚪ Neutral"),
            })

        if flows_data:
            df_flow = pd.DataFrame(flows_data)
            fig2 = px.bar(
                df_flow,
                x="Net Flow",
                y="Wallet",
                orientation="h",
                color="Color",
                color_discrete_map={
                    "🟢 Buying": "#4CAF50",
                    "🔴 Selling": "#F44336",
                    "⚪ Neutral": "#9E9E9E"
                },
            )
            fig2.update_layout(
                height=400,
                yaxis={"categoryorder": "total ascending"},
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --- Wallet Balance Table ---
    st.subheader("🧭 Wallet Details")

    # Prepare table data
    table_data = []
    for i, bal in enumerate(balances, 1):
        flows = wallet_flows.get(bal.address, {})
        table_data.append({
            "Rank": i,
            "Wallet": get_wallet_display(bal.address, bal.label),
            "Address": bal.address,
            "Balance": bal.ui_amount,
            "Share %": bal.share_of_tracked * 100,
            "Net 24h": flows.get("net_24h", 0),
            "Net 7d": flows.get("net_7d", 0),
            "Txs 24h": flows.get("tx_count_24h", 0),
        })

    df = pd.DataFrame(table_data)

    # Format display
    st.dataframe(
        df,
        column_config={
            "Rank": st.column_config.NumberColumn("#", width="small"),
            "Wallet": st.column_config.TextColumn("Wallet", width="medium"),
            "Address": st.column_config.TextColumn("Full Address", width="large"),
            "Balance": st.column_config.NumberColumn(
                "SWEENEE",
                format="%.0f",
            ),
            "Share %": st.column_config.NumberColumn(
                "Share %",
                format="%.2f%%",
            ),
            "Net 24h": st.column_config.NumberColumn(
                "Net 24h",
                format="%+.0f",
            ),
            "Net 7d": st.column_config.NumberColumn(
                "Net 7d",
                format="%+.0f",
            ),
            "Txs 24h": st.column_config.NumberColumn("Txs 24h"),
        },
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    # --- Transaction Feed ---
    st.subheader("🔁 Whale Activity Feed")

    if transactions:
        tx_data = []
        for tx in transactions[:50]:
            # Find wallet label
            label = None
            for w in wallets:
                if w["address"] == tx.wallet_address:
                    label = w.get("label")
                    break

            # Color-coded indicators
            tx_type = tx.classification.value.upper()
            if tx_type == "BUY":
                indicator = "🟢"
                amount_str = f"+{format_number(abs(tx.amount_change))}"
            elif tx_type == "SELL":
                indicator = "🔴"
                amount_str = f"-{format_number(abs(tx.amount_change))}"
            elif tx_type == "TRANSFER_IN":
                indicator = "⬆️"
                amount_str = f"+{format_number(abs(tx.amount_change))}"
            elif tx_type == "TRANSFER_OUT":
                indicator = "⬇️"
                amount_str = f"-{format_number(abs(tx.amount_change))}"
            else:
                indicator = "⚪"
                amount_str = format_number(abs(tx.amount_change))

            tx_data.append({
                "": indicator,
                "Time": tx.block_time.strftime("%m/%d %H:%M") if tx.block_time else "—",
                "Wallet": get_wallet_display(tx.wallet_address, label),
                "Action": tx_type,
                "SWEENEE": amount_str,
                "🔗": tx.explorer_url,
            })

        df_tx = pd.DataFrame(tx_data)
        st.dataframe(
            df_tx,
            column_config={
                "": st.column_config.TextColumn("", width="small"),
                "Time": st.column_config.TextColumn("Time", width="small"),
                "Wallet": st.column_config.TextColumn("Wallet", width="medium"),
                "Action": st.column_config.TextColumn("Action", width="small"),
                "SWEENEE": st.column_config.TextColumn("SWEENEE", width="medium"),
                "🔗": st.column_config.LinkColumn("🔗", display_text="View", width="small"),
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No recent SWEENEE transactions found for tracked wallets.")

    st.divider()

    # --- Telegram Summary ---
    st.subheader("📣 Telegram Summary")

    tab1, tab2 = st.tabs(["Daily Summary", "Weekly Summary"])

    with tab1:
        daily = generate_daily_summary(metrics)
        st.code(daily, language=None)
        st.button("📋 Copy Daily Summary", key="copy_daily")

    with tab2:
        weekly = generate_weekly_summary(metrics)
        st.code(weekly, language=None)
        st.button("📋 Copy Weekly Summary", key="copy_weekly")

    # --- Sidebar: Exports & Settings ---
    with st.sidebar:
        st.header("🛠️ Tools")

        # Export Section
        st.subheader("📥 Export Data")

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            # Wallets CSV
            wallets_csv = export_wallets_csv(balances)
            st.download_button(
                "📊 Wallets CSV",
                data=wallets_csv,
                file_name=get_export_filename("wallets", "csv"),
                mime="text/csv",
                use_container_width=True,
            )
            # Transactions CSV
            tx_csv = export_transactions_csv(transactions)
            st.download_button(
                "📋 Transactions CSV",
                data=tx_csv,
                file_name=get_export_filename("transactions", "csv"),
                mime="text/csv",
                use_container_width=True,
            )

        with exp_col2:
            # Wallets JSON
            wallets_json = export_wallets_json(balances)
            st.download_button(
                "📊 Wallets JSON",
                data=wallets_json,
                file_name=get_export_filename("wallets", "json"),
                mime="application/json",
                use_container_width=True,
            )
            # Transactions JSON
            tx_json = export_transactions_json(transactions)
            st.download_button(
                "📋 Transactions JSON",
                data=tx_json,
                file_name=get_export_filename("transactions", "json"),
                mime="application/json",
                use_container_width=True,
            )

        st.divider()

        # Alerts Section
        st.subheader("🚨 Alert Settings")
        alert_threshold = st.number_input(
            "Large move threshold",
            min_value=100_000,
            max_value=10_000_000,
            value=1_000_000,
            step=100_000,
            format="%d",
            help="Minimum SWEENEE for large buy/sell alert",
        )

        # Check for alerts
        alert_service = AlertService(large_move_threshold=alert_threshold)
        current_bals = {b.address: b.ui_amount for b in balances}
        new_alerts = alert_service.check_transactions(transactions, current_bals)

        if new_alerts:
            st.success(f"Generated {len(new_alerts)} new alert(s)")

        # Show recent alerts
        recent_alerts = alert_service.get_recent_alerts(hours=24)
        if recent_alerts:
            st.markdown(f"**Recent Alerts:** {len(recent_alerts)}")
            for alert in recent_alerts[:3]:
                st.markdown(f"{alert.emoji} {alert.description[:30]}...")

        st.divider()

        # Info
        st.subheader("ℹ️ Info")
        st.caption(f"Wallets: {len(wallets)}")
        st.caption(f"Transactions: {len(transactions)}")
        st.caption(f"Token: {SWEENEE_MINT[:8]}...")

    # --- Footer ---
    st.markdown(
        '<div class="disclaimer">'
        "This dashboard is for community transparency and research only. "
        "It tracks selected configured wallets, not all SWEENEE holders. "
        "This is not financial advice."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
