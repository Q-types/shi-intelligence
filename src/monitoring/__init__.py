"""
Real-time wallet monitoring and alerting for SHI.

Provides WalletWatcher for tracking balance changes, alert engine for notifications,
and profile evolution tracking.
"""

from .watcher import WalletWatcher, WatchedWallet, BalanceChange
from .alerts import AlertEngine, Alert, AlertType, AlertSeverity
from .profiles import ProfileTracker, ProfileSnapshot, ProfileEvolution

__all__ = [
    "WalletWatcher",
    "WatchedWallet",
    "BalanceChange",
    "AlertEngine",
    "Alert",
    "AlertType",
    "AlertSeverity",
    "ProfileTracker",
    "ProfileSnapshot",
    "ProfileEvolution",
]
