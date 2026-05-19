"""Shared modules for SHI dashboards - adapted from SWEENEE."""

from .cache import SweeneeCache, get_cache
from .solana_client import SolanaClient, get_client, TokenBalance
from .token_balances import WalletBalance, BalanceSummary, fetch_all_balances, compute_balance_summary
from .transactions import SweeneeTransaction, TransactionType, fetch_all_transactions, classify_transaction
from .alerts import AlertService, WhaleAlert, AlertType, render_alert_banners
from .history import SnapshotService, BalanceChange, render_historical_chart
from .webhook import TelegramWebhook, WebhookResult, get_webhook
from .export import export_wallets_csv, export_wallets_json, export_transactions_csv, export_transactions_json
from .metrics import DashboardMetrics, compute_dashboard_metrics, compute_wallet_flows
from .wallet_loader import TrackedWallet, load_all_wallets, is_valid_solana_address

__all__ = [
    # Cache
    "SweeneeCache",
    "get_cache",
    # Solana
    "SolanaClient",
    "get_client",
    "TokenBalance",
    # Balances
    "WalletBalance",
    "BalanceSummary",
    "fetch_all_balances",
    "compute_balance_summary",
    # Transactions
    "SweeneeTransaction",
    "TransactionType",
    "fetch_all_transactions",
    "classify_transaction",
    # Alerts
    "AlertService",
    "WhaleAlert",
    "AlertType",
    "render_alert_banners",
    # History
    "SnapshotService",
    "BalanceChange",
    "render_historical_chart",
    # Webhook
    "TelegramWebhook",
    "WebhookResult",
    "get_webhook",
    # Export
    "export_wallets_csv",
    "export_wallets_json",
    "export_transactions_csv",
    "export_transactions_json",
    # Metrics
    "DashboardMetrics",
    "compute_dashboard_metrics",
    "compute_wallet_flows",
    # Wallet loader
    "TrackedWallet",
    "load_all_wallets",
    "is_valid_solana_address",
]
