"""Tests for metrics module."""

import pytest
from datetime import datetime, timezone, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solana_client import TokenBalance
from src.token_balances import WalletBalance, compute_balance_summary
from src.transactions import SweeneeTransaction, TransactionType
from src.metrics import compute_dashboard_metrics, compute_wallet_flows


def make_balance(address: str, amount: float, total: float = 1000000) -> WalletBalance:
    """Create a test WalletBalance."""
    return WalletBalance(
        address=address,
        label=None,
        balance=TokenBalance(
            wallet_address=address,
            token_mint="TEST",
            raw_amount=int(amount * 1e6),
            decimals=6,
            ui_amount=amount,
            fetched_at=datetime.now(timezone.utc),
        ),
        share_of_tracked=amount / total,
    )


def make_transaction(
    wallet: str,
    amount: float,
    hours_ago: float = 1,
    classification: TransactionType = TransactionType.UNKNOWN,
) -> SweeneeTransaction:
    """Create a test transaction."""
    return SweeneeTransaction(
        signature=f"sig_{wallet}_{amount}",
        block_time=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        wallet_address=wallet,
        token_mint="TEST",
        amount_change=amount,
        direction="in" if amount > 0 else "out",
        classification=classification,
    )


class TestBalanceSummary:
    """Tests for balance summary computation."""

    def test_total_balance(self):
        balances = [
            make_balance("A", 100),
            make_balance("B", 200),
            make_balance("C", 300),
        ]
        summary = compute_balance_summary(balances)
        assert summary.total_sweenee == 600

    def test_wallets_holding(self):
        balances = [
            make_balance("A", 100),
            make_balance("B", 0),
            make_balance("C", 300),
        ]
        summary = compute_balance_summary(balances)
        assert summary.wallets_holding == 2

    def test_largest_holder(self):
        balances = [
            make_balance("A", 300),
            make_balance("B", 100),
            make_balance("C", 200),
        ]
        # Sort by balance desc
        balances.sort(key=lambda x: x.ui_amount, reverse=True)
        summary = compute_balance_summary(balances)
        assert summary.largest_holder.address == "A"

    def test_hhi_calculation(self):
        # Equal distribution: HHI = 4 * (0.25)^2 = 0.25
        balances = [
            make_balance("A", 250, 1000),
            make_balance("B", 250, 1000),
            make_balance("C", 250, 1000),
            make_balance("D", 250, 1000),
        ]
        summary = compute_balance_summary(balances)
        assert abs(summary.hhi - 0.25) < 0.01

    def test_hhi_concentrated(self):
        # One holder has 100%: HHI = 1.0
        balances = [
            make_balance("A", 1000, 1000),
            make_balance("B", 0, 1000),
        ]
        summary = compute_balance_summary(balances)
        assert summary.hhi == 1.0

    def test_empty_balances(self):
        summary = compute_balance_summary([])
        assert summary.total_sweenee == 0
        assert summary.wallets_holding == 0
        assert summary.largest_holder is None


class TestDashboardMetrics:
    """Tests for dashboard metrics computation."""

    def test_net_flow_24h(self):
        balances = [make_balance("A", 1000)]
        transactions = [
            make_transaction("A", +500, hours_ago=1),
            make_transaction("A", -200, hours_ago=2),
        ]
        metrics = compute_dashboard_metrics(balances, transactions)
        assert metrics.net_flow_24h == 300

    def test_net_flow_excludes_old(self):
        balances = [make_balance("A", 1000)]
        transactions = [
            make_transaction("A", +500, hours_ago=1),  # Within 24h
            make_transaction("A", +1000, hours_ago=30),  # Outside 24h
        ]
        metrics = compute_dashboard_metrics(balances, transactions)
        assert metrics.net_flow_24h == 500

    def test_transaction_count(self):
        balances = [make_balance("A", 1000)]
        transactions = [
            make_transaction("A", +100, hours_ago=1),
            make_transaction("A", +200, hours_ago=2),
            make_transaction("A", -50, hours_ago=3),
        ]
        metrics = compute_dashboard_metrics(balances, transactions)
        assert metrics.transaction_count_24h == 3


class TestWalletFlows:
    """Tests for per-wallet flow computation."""

    def test_per_wallet_net_flow(self):
        balances = [
            make_balance("A", 1000),
            make_balance("B", 500),
        ]
        transactions = [
            make_transaction("A", +500, hours_ago=1),
            make_transaction("A", -200, hours_ago=2),
            make_transaction("B", +100, hours_ago=1),
        ]

        flows = compute_wallet_flows(balances, transactions)

        assert flows["A"]["net_24h"] == 300
        assert flows["B"]["net_24h"] == 100

    def test_wallet_with_no_transactions(self):
        balances = [make_balance("A", 1000)]
        transactions = []

        flows = compute_wallet_flows(balances, transactions)

        assert flows["A"]["net_24h"] == 0
        assert flows["A"]["tx_count_24h"] == 0
