"""Historical Balance Tracking Service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import plotly.graph_objects as go
import structlog

from .cache import get_cache
from .token_balances import WalletBalance

logger = structlog.get_logger()


@dataclass
class BalanceChange:
    """Represents a significant balance change."""

    wallet_address: str
    date: str
    previous_balance: float
    new_balance: float
    change_pct: float

    @property
    def is_significant(self) -> bool:
        """Check if change is significant (>10%)."""
        return abs(self.change_pct) >= 10.0


class SnapshotService:
    """Service for managing historical balance snapshots."""

    def __init__(self, mint: str):
        self.mint = mint
        self.cache = get_cache()

    def take_snapshot(self, balances: list[WalletBalance]):
        """Take a daily snapshot of current balances.

        Deduplicates by wallet+date (upsert behavior).
        """
        self.cache.save_balance_snapshots_batch(balances, self.mint)
        logger.info(
            "snapshot_taken",
            wallet_count=len(balances),
            total_balance=sum(b.ui_amount for b in balances),
        )

    def get_history(self, wallet: str, days: int = 30) -> list[dict[str, Any]]:
        """Get balance history for a specific wallet."""
        return self.cache.get_wallet_history(wallet, self.mint, days)

    def get_total_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get aggregated total balance history."""
        return self.cache.get_total_history(self.mint, days)

    def get_all_wallet_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get per-wallet history for stacked charts."""
        return self.cache.get_all_wallet_history(self.mint, days)

    def detect_significant_changes(
        self, days: int = 7, threshold_pct: float = 10.0
    ) -> list[BalanceChange]:
        """Detect significant balance changes in recent history."""
        history = self.cache.get_all_wallet_history(self.mint, days)

        if not history:
            return []

        # Group by wallet
        wallet_history: dict[str, list[dict]] = {}
        for entry in history:
            addr = entry["wallet_address"]
            if addr not in wallet_history:
                wallet_history[addr] = []
            wallet_history[addr].append(entry)

        changes = []
        for wallet, entries in wallet_history.items():
            if len(entries) < 2:
                continue

            # Sort by date
            entries.sort(key=lambda x: x["snapshot_date"])

            # Compare consecutive days
            for i in range(1, len(entries)):
                prev = entries[i - 1]["ui_amount"]
                curr = entries[i]["ui_amount"]

                if prev > 0:
                    change_pct = ((curr - prev) / prev) * 100
                elif curr > 0:
                    change_pct = 100.0  # New position
                else:
                    continue

                if abs(change_pct) >= threshold_pct:
                    changes.append(
                        BalanceChange(
                            wallet_address=wallet,
                            date=entries[i]["snapshot_date"],
                            previous_balance=prev,
                            new_balance=curr,
                            change_pct=change_pct,
                        )
                    )

        return sorted(changes, key=lambda x: x.date, reverse=True)


def render_historical_chart(
    mint: str,
    days: int = 30,
    show_individual: bool = False,
    wallet_labels: dict[str, str] | None = None,
) -> go.Figure | None:
    """Render historical holdings chart.

    Args:
        mint: Token mint address
        days: Number of days of history
        show_individual: If True, show stacked area by wallet; if False, show aggregate
        wallet_labels: Optional mapping of wallet addresses to labels

    Returns:
        Plotly figure or None if no data
    """
    service = SnapshotService(mint)

    if show_individual:
        # Stacked area chart by wallet
        history = service.get_all_wallet_history(days)
        if not history:
            return None

        # Pivot data for stacked chart
        import pandas as pd

        df = pd.DataFrame(history)
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

        # Get unique wallets and dates
        wallets = df["wallet_address"].unique()
        dates = sorted(df["snapshot_date"].unique())

        fig = go.Figure()

        for wallet in wallets:
            wallet_data = df[df["wallet_address"] == wallet].set_index("snapshot_date")
            # Reindex to fill missing dates
            wallet_data = wallet_data.reindex(dates, fill_value=0)

            label = wallet_labels.get(wallet, wallet[:8] + "...") if wallet_labels else wallet[:8] + "..."

            fig.add_trace(
                go.Scatter(
                    x=wallet_data.index,
                    y=wallet_data["ui_amount"],
                    name=label,
                    mode="lines",
                    stackgroup="holdings",
                    hovertemplate=f"{label}<br>%{{x}}<br>%{{y:,.0f}} SWEENEE<extra></extra>",
                )
            )

    else:
        # Aggregate total line chart
        history = service.get_total_history(days)
        if not history:
            return None

        dates = [h["snapshot_date"] for h in history]
        totals = [h["total_balance"] for h in history]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=totals,
                mode="lines+markers",
                name="Total Holdings",
                line=dict(color="#4CAF50", width=3),
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(76, 175, 80, 0.2)",
                hovertemplate="%{x}<br>%{y:,.0f} SWEENEE<extra></extra>",
            )
        )

        # Add annotations for significant changes
        changes = service.detect_significant_changes(days, threshold_pct=10.0)
        for change in changes[:5]:  # Limit to 5 annotations
            # Find the total for that date
            total_on_date = next(
                (h["total_balance"] for h in history if h["snapshot_date"] == change.date),
                None,
            )
            if total_on_date:
                emoji = "📈" if change.change_pct > 0 else "📉"
                fig.add_annotation(
                    x=change.date,
                    y=total_on_date,
                    text=f"{emoji} {change.change_pct:+.0f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="#888",
                    font=dict(size=10),
                    bgcolor="rgba(0,0,0,0.6)",
                    bordercolor="#888",
                )

    # Style the chart
    fig.update_layout(
        title=dict(
            text="Whale Holdings Over Time",
            font=dict(size=18, color="#fff"),
        ),
        xaxis=dict(
            title="Date",
            gridcolor="rgba(128,128,128,0.2)",
            tickfont=dict(color="#aaa"),
        ),
        yaxis=dict(
            title="SWEENEE Balance",
            gridcolor="rgba(128,128,128,0.2)",
            tickfont=dict(color="#aaa"),
            tickformat=",",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fff"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#444",
            borderwidth=1,
        ),
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode="x unified",
    )

    return fig


def render_wallet_history_chart(
    wallet: str,
    mint: str,
    days: int = 30,
    label: str | None = None,
) -> go.Figure | None:
    """Render history chart for a single wallet."""
    service = SnapshotService(mint)
    history = service.get_history(wallet, days)

    if not history:
        return None

    dates = [h["snapshot_date"] for h in history]
    balances = [h["ui_amount"] for h in history]
    wallet_label = label or wallet[:8] + "..."

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=balances,
            mode="lines+markers",
            name=wallet_label,
            line=dict(color="#1E88E5", width=2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(30, 136, 229, 0.2)",
            hovertemplate="%{x}<br>%{y:,.0f} SWEENEE<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Balance History: {wallet_label}",
            font=dict(size=16, color="#fff"),
        ),
        xaxis=dict(
            title="Date",
            gridcolor="rgba(128,128,128,0.2)",
            tickfont=dict(color="#aaa"),
        ),
        yaxis=dict(
            title="SWEENEE Balance",
            gridcolor="rgba(128,128,128,0.2)",
            tickfont=dict(color="#aaa"),
            tickformat=",",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fff"),
        margin=dict(l=60, r=20, t=50, b=40),
        showlegend=False,
    )

    return fig
