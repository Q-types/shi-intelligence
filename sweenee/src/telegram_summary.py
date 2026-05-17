"""Telegram Summary Generator - Create Telegram-ready updates."""

from __future__ import annotations

from datetime import datetime, timezone

from .metrics import DashboardMetrics


def format_number(n: float, decimals: int = 0) -> str:
    """Format number with commas."""
    if decimals == 0:
        return f"{n:,.0f}"
    return f"{n:,.{decimals}f}"


def format_flow(n: float) -> str:
    """Format flow with sign."""
    if n >= 0:
        return f"+{format_number(n)}"
    return format_number(n)


def short_address(address: str | None) -> str:
    """Shorten address for display."""
    if not address:
        return "Unknown"
    return f"{address[:4]}...{address[-4:]}"


def generate_daily_summary(
    metrics: DashboardMetrics,
    dashboard_url: str | None = None,
) -> str:
    """Generate a daily Telegram summary.

    Returns a formatted string ready to paste into Telegram.
    """
    lines = [
        "🐳 SWEENEE Whale Wallet Watch",
        "",
        f"Tracked wallets: {metrics.total_tracked_wallets}",
        f"Wallets holding SWEENEE: {metrics.wallets_holding}",
        f"Total tracked SWEENEE: {format_number(metrics.total_sweenee)}",
        "",
        f"24h net flow: {format_flow(metrics.net_flow_24h)} SWEENEE",
        f"24h transactions: {metrics.transaction_count_24h}",
        "",
    ]

    # Largest holder
    if metrics.largest_holder_address:
        holder_name = metrics.largest_holder_label or short_address(
            metrics.largest_holder_address
        )
        lines.append("Largest holder:")
        lines.append(
            f"{holder_name} — {format_number(metrics.largest_holder_balance)} SWEENEE"
        )
        lines.append("")

    # Largest movements
    if metrics.largest_inflow_24h > 0:
        lines.append("Largest 24h inflow:")
        lines.append(
            f"+{format_number(metrics.largest_inflow_24h)} SWEENEE into {short_address(metrics.largest_inflow_wallet)}"
        )
        lines.append("")

    if metrics.largest_outflow_24h > 0:
        lines.append("Largest 24h outflow:")
        lines.append(
            f"-{format_number(metrics.largest_outflow_24h)} SWEENEE from {short_address(metrics.largest_outflow_wallet)}"
        )
        lines.append("")

    # Dashboard link
    if dashboard_url:
        lines.append(f"Dashboard: {dashboard_url}")
        lines.append("")

    lines.append("Not financial advice. Community transparency only.")

    return "\n".join(lines)


def generate_weekly_summary(
    metrics: DashboardMetrics,
    dashboard_url: str | None = None,
) -> str:
    """Generate a more detailed weekly summary."""
    lines = [
        "🐳 SWEENEE Whale Wallet Watch — Weekly Update",
        "",
        "📊 Overview",
        f"• Tracked wallets: {metrics.total_tracked_wallets}",
        f"• Currently holding: {metrics.wallets_holding} ({metrics.holding_ratio*100:.0f}%)",
        f"• Total tracked SWEENEE: {format_number(metrics.total_sweenee)}",
        "",
        "📈 Concentration",
        f"• Top 10 share: {metrics.top_10_share*100:.1f}%",
        f"• HHI (among tracked): {metrics.hhi:.4f}",
        "",
        "🔄 7-Day Activity",
        f"• Net flow: {format_flow(metrics.net_flow_7d)} SWEENEE",
        f"• Transactions: {metrics.transaction_count_7d}",
        "",
        "⏰ 24-Hour Snapshot",
        f"• Net flow: {format_flow(metrics.net_flow_24h)} SWEENEE",
        f"• Transactions: {metrics.transaction_count_24h}",
        "",
    ]

    # Top holder
    if metrics.largest_holder_address:
        holder_name = metrics.largest_holder_label or short_address(
            metrics.largest_holder_address
        )
        lines.append("👑 Largest Tracked Holder")
        lines.append(
            f"• {holder_name}: {format_number(metrics.largest_holder_balance)} SWEENEE"
        )
        lines.append("")

    # Dashboard link
    if dashboard_url:
        lines.append(f"🔗 Full dashboard: {dashboard_url}")
        lines.append("")

    lines.append("—")
    lines.append("This dashboard tracks selected whale wallets only.")
    lines.append("Not financial advice.")

    return "\n".join(lines)


def generate_alert_message(
    event_type: str,
    wallet: str,
    amount: float,
    classification: str,
    label: str | None = None,
) -> str:
    """Generate an alert message for significant movements."""
    wallet_display = label or short_address(wallet)

    if event_type == "large_inflow":
        emoji = "🟢"
        action = "received"
    elif event_type == "large_outflow":
        emoji = "🔴"
        action = "sent"
    else:
        emoji = "⚪"
        action = "moved"

    return (
        f"{emoji} SWEENEE Whale Alert\n"
        f"\n"
        f"{wallet_display} {action} {format_number(abs(amount))} SWEENEE\n"
        f"Type: {classification}\n"
        f"\n"
        f"Track more at the Whale Dashboard"
    )
