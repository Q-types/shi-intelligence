"""
Real-time Wallet Watcher for SHI.

Monitors wallet balance changes and detects significant movements.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.types import WalletAddress, TokenMint

logger = structlog.get_logger()


@dataclass
class WatchedWallet:
    """Wallet under active monitoring."""

    wallet: WalletAddress
    token_mint: TokenMint
    user_id: str
    added_at: datetime
    last_balance: float
    last_checked: datetime
    alert_threshold: float = 0.05  # 5% of supply triggers alert


@dataclass
class BalanceChange:
    """Detected balance change event."""

    wallet: WalletAddress
    token_mint: TokenMint
    timestamp: datetime
    previous_balance: float
    new_balance: float
    delta: float
    delta_pct: float
    pct_of_supply: float
    is_significant: bool


class WalletWatcher:
    """
    Real-time wallet monitoring service.

    Tracks balance changes for watched wallets and emits events
    when significant movements occur (>5% of supply by default).
    """

    def __init__(
        self,
        db_session: AsyncSession,
        check_interval: int = 30,  # seconds
        significance_threshold: float = 0.05,  # 5% of supply
    ):
        """
        Initialize wallet watcher.

        Args:
            db_session: Database session for querying
            check_interval: How often to check balances (seconds)
            significance_threshold: Minimum movement % of supply to flag
        """
        self.db_session = db_session
        self.check_interval = check_interval
        self.significance_threshold = significance_threshold

        # In-memory tracking
        self._watched_wallets: Dict[str, WatchedWallet] = {}
        self._balance_cache: Dict[str, float] = {}
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None

    def _wallet_key(self, wallet: WalletAddress, token_mint: TokenMint) -> str:
        """Generate unique key for wallet-token pair."""
        return f"{wallet}:{token_mint}"

    async def add_watched_wallet(
        self,
        wallet: WalletAddress,
        token_mint: TokenMint,
        user_id: str,
        alert_threshold: float = 0.05,
    ) -> WatchedWallet:
        """
        Add a wallet to the watch list.

        Args:
            wallet: Wallet address to monitor
            token_mint: Token mint address
            user_id: User requesting the watch
            alert_threshold: Movement threshold for alerts

        Returns:
            WatchedWallet object
        """
        key = self._wallet_key(wallet, token_mint)

        # Get current balance
        current_balance = await self._fetch_current_balance(wallet, token_mint)

        watched = WatchedWallet(
            wallet=wallet,
            token_mint=token_mint,
            user_id=user_id,
            added_at=datetime.now(timezone.utc),
            last_balance=current_balance,
            last_checked=datetime.now(timezone.utc),
            alert_threshold=alert_threshold,
        )

        self._watched_wallets[key] = watched
        self._balance_cache[key] = current_balance

        logger.info(
            "wallet_added_to_watchlist",
            wallet=wallet,
            token=token_mint,
            user=user_id,
            current_balance=current_balance,
        )

        return watched

    async def remove_watched_wallet(
        self,
        wallet: WalletAddress,
        token_mint: TokenMint,
    ) -> bool:
        """
        Remove a wallet from the watch list.

        Args:
            wallet: Wallet address
            token_mint: Token mint address

        Returns:
            True if removed, False if not found
        """
        key = self._wallet_key(wallet, token_mint)

        if key in self._watched_wallets:
            del self._watched_wallets[key]
            self._balance_cache.pop(key, None)
            logger.info("wallet_removed_from_watchlist", wallet=wallet, token=token_mint)
            return True

        return False

    async def get_watched_wallets(
        self,
        user_id: Optional[str] = None,
        token_mint: Optional[TokenMint] = None,
    ) -> List[WatchedWallet]:
        """
        Get list of watched wallets with optional filtering.

        Args:
            user_id: Filter by user (optional)
            token_mint: Filter by token (optional)

        Returns:
            List of WatchedWallet objects
        """
        wallets = list(self._watched_wallets.values())

        if user_id:
            wallets = [w for w in wallets if w.user_id == user_id]

        if token_mint:
            wallets = [w for w in wallets if w.token_mint == token_mint]

        return wallets

    async def check_balance_changes(
        self,
        token_mint: TokenMint,
        total_supply: float,
    ) -> List[BalanceChange]:
        """
        Check for balance changes across all watched wallets for a token.

        Args:
            token_mint: Token to check
            total_supply: Total supply for calculating percentages

        Returns:
            List of detected BalanceChange events
        """
        changes = []

        # Filter watched wallets for this token
        watched = [w for w in self._watched_wallets.values() if w.token_mint == token_mint]

        if not watched:
            return changes

        logger.debug("checking_balance_changes", token=token_mint, wallet_count=len(watched))

        for wallet_obj in watched:
            try:
                key = self._wallet_key(wallet_obj.wallet, token_mint)
                previous_balance = self._balance_cache.get(key, wallet_obj.last_balance)

                # Fetch current balance
                current_balance = await self._fetch_current_balance(
                    wallet_obj.wallet, token_mint
                )

                # Calculate change
                delta = current_balance - previous_balance
                delta_pct = (delta / previous_balance * 100) if previous_balance > 0 else 0.0
                pct_of_supply = (abs(delta) / total_supply) if total_supply > 0 else 0.0

                # Check if significant
                is_significant = pct_of_supply >= wallet_obj.alert_threshold

                if abs(delta) > 0:
                    change = BalanceChange(
                        wallet=wallet_obj.wallet,
                        token_mint=token_mint,
                        timestamp=datetime.now(timezone.utc),
                        previous_balance=previous_balance,
                        new_balance=current_balance,
                        delta=delta,
                        delta_pct=delta_pct,
                        pct_of_supply=pct_of_supply,
                        is_significant=is_significant,
                    )

                    changes.append(change)

                    # Update cache
                    self._balance_cache[key] = current_balance
                    wallet_obj.last_balance = current_balance
                    wallet_obj.last_checked = datetime.now(timezone.utc)

                    if is_significant:
                        logger.info(
                            "significant_balance_change_detected",
                            wallet=wallet_obj.wallet,
                            delta=delta,
                            delta_pct=delta_pct,
                            pct_of_supply=pct_of_supply,
                        )

            except Exception as e:
                logger.error(
                    "error_checking_wallet_balance",
                    wallet=wallet_obj.wallet,
                    error=str(e),
                )

        return changes

    async def _fetch_current_balance(
        self,
        wallet: WalletAddress,
        token_mint: TokenMint,
    ) -> float:
        """
        Fetch current balance for a wallet.

        This is a placeholder - in production, this would query the RPC
        or read from the latest snapshot in the database.

        Args:
            wallet: Wallet address
            token_mint: Token mint

        Returns:
            Current balance as float
        """
        # In a real implementation, this would query Solana RPC or database
        # For now, return cached value or 0
        key = self._wallet_key(wallet, token_mint)
        return self._balance_cache.get(key, 0.0)

    async def start_monitoring(self) -> None:
        """Start the background monitoring loop."""
        if self._is_running:
            logger.warning("monitoring_already_running")
            return

        self._is_running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("wallet_monitoring_started", interval=self.check_interval)

    async def stop_monitoring(self) -> None:
        """Stop the background monitoring loop."""
        if not self._is_running:
            return

        self._is_running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("wallet_monitoring_stopped")

    async def _monitoring_loop(self) -> None:
        """Background loop that checks balances periodically."""
        logger.info("monitoring_loop_started")

        while self._is_running:
            try:
                # Group watched wallets by token
                tokens: Set[TokenMint] = {w.token_mint for w in self._watched_wallets.values()}

                for token_mint in tokens:
                    # In production, fetch total supply from database or RPC
                    total_supply = 1_000_000_000.0  # Placeholder

                    # Check for changes
                    changes = await self.check_balance_changes(token_mint, total_supply)

                    # Emit significant changes (would trigger alerts in production)
                    for change in changes:
                        if change.is_significant:
                            logger.info(
                                "emitting_balance_change_event",
                                wallet=change.wallet,
                                delta_pct=change.delta_pct,
                                pct_of_supply=change.pct_of_supply,
                            )

                # Sleep until next check
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                logger.info("monitoring_loop_cancelled")
                break
            except Exception as e:
                logger.error("monitoring_loop_error", error=str(e))
                await asyncio.sleep(self.check_interval)

        logger.info("monitoring_loop_stopped")

    def get_statistics(self) -> Dict:
        """
        Get monitoring statistics.

        Returns:
            Dict with statistics about watched wallets
        """
        tokens = {w.token_mint for w in self._watched_wallets.values()}
        users = {w.user_id for w in self._watched_wallets.values()}

        return {
            "total_watched_wallets": len(self._watched_wallets),
            "unique_tokens": len(tokens),
            "unique_users": len(users),
            "is_monitoring": self._is_running,
            "check_interval": self.check_interval,
            "significance_threshold": self.significance_threshold,
        }
