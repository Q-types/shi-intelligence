"""Metrics - Aggregated dashboard metrics and calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from .token_balances import WalletBalance, BalanceSummary, compute_balance_summary
from .transactions import (
    SweeneeTransaction,
    compute_net_flow,
    count_transactions,
    find_largest_movement,
)


@dataclass
class DashboardMetrics:
    """Complete metrics for the dashboard."""

    # Balance metrics
    total_tracked_wallets: int
    wallets_holding: int
    total_sweenee: float
    largest_holder_address: str | None
    largest_holder_label: str | None
    largest_holder_balance: float

    # Concentration
    top_10_share: float
    hhi: float

    # 24h flow metrics
    net_flow_24h: float
    transaction_count_24h: int
    largest_inflow_24h: float
    largest_inflow_wallet: str | None
    largest_outflow_24h: float
    largest_outflow_wallet: str | None

    # 7d flow metrics
    net_flow_7d: float
    transaction_count_7d: int

    # Timestamps
    balance_fetched_at: datetime
    calculated_at: datetime

    @property
    def net_flow_24h_display(self) -> str:
        """Format 24h net flow with sign."""
        if self.net_flow_24h >= 0:
            return f"+{self.net_flow_24h:,.0f}"
        return f"{self.net_flow_24h:,.0f}"

    @property
    def holding_ratio(self) -> float:
        """Ratio of wallets currently holding SWEENEE."""
        if self.total_tracked_wallets == 0:
            return 0.0
        return self.wallets_holding / self.total_tracked_wallets


def compute_dashboard_metrics(
    balances: list[WalletBalance],
    transactions: list[SweeneeTransaction],
) -> DashboardMetrics:
    """Compute all dashboard metrics from balances and transactions."""

    # Balance summary
    balance_summary = compute_balance_summary(balances)

    # Largest holder
    largest = balance_summary.largest_holder
    largest_address = largest.address if largest else None
    largest_label = largest.label if largest else None
    largest_balance = largest.ui_amount if largest else 0.0

    # 24h metrics
    net_24h = compute_net_flow(transactions, hours=24)
    tx_count_24h = count_transactions(transactions, hours=24)
    largest_in_24h = find_largest_movement(transactions, direction="in", hours=24)
    largest_out_24h = find_largest_movement(transactions, direction="out", hours=24)

    # 7d metrics
    net_7d = compute_net_flow(transactions, hours=168)
    tx_count_7d = count_transactions(transactions, hours=168)

    return DashboardMetrics(
        total_tracked_wallets=balance_summary.total_tracked_wallets,
        wallets_holding=balance_summary.wallets_holding,
        total_sweenee=balance_summary.total_sweenee,
        largest_holder_address=largest_address,
        largest_holder_label=largest_label,
        largest_holder_balance=largest_balance,
        top_10_share=balance_summary.top_10_share,
        hhi=balance_summary.hhi,
        net_flow_24h=net_24h,
        transaction_count_24h=tx_count_24h,
        largest_inflow_24h=largest_in_24h.amount_change if largest_in_24h else 0.0,
        largest_inflow_wallet=largest_in_24h.wallet_address if largest_in_24h else None,
        largest_outflow_24h=abs(largest_out_24h.amount_change) if largest_out_24h else 0.0,
        largest_outflow_wallet=largest_out_24h.wallet_address if largest_out_24h else None,
        net_flow_7d=net_7d,
        transaction_count_7d=tx_count_7d,
        balance_fetched_at=balance_summary.fetched_at,
        calculated_at=datetime.now(timezone.utc),
    )


def compute_wallet_flows(
    wallets: list[WalletBalance],
    transactions: list[SweeneeTransaction],
) -> dict[str, dict[str, float]]:
    """Compute per-wallet flow metrics."""
    flows = {}

    for wallet in wallets:
        wallet_txs = [
            tx for tx in transactions if tx.wallet_address == wallet.address
        ]

        flows[wallet.address] = {
            "net_24h": compute_net_flow(wallet_txs, hours=24),
            "net_7d": compute_net_flow(wallet_txs, hours=168),
            "tx_count_24h": count_transactions(wallet_txs, hours=24),
            "tx_count_7d": count_transactions(wallet_txs, hours=168),
        }

    return flows
