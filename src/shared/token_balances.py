"""Token Balance Fetching - Fetch SWEENEE balances for tracked wallets."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from .wallet_loader import TrackedWallet

from .solana_client import SolanaClient, TokenBalance, get_client

logger = structlog.get_logger()


@dataclass
class WalletBalance:
    """Wallet with balance information."""

    address: str
    label: str | None
    balance: TokenBalance
    share_of_tracked: float = 0.0

    @property
    def ui_amount(self) -> float:
        return self.balance.ui_amount

    @property
    def display_name(self) -> str:
        if self.label:
            return self.label
        return f"{self.address[:4]}...{self.address[-4:]}"


async def fetch_wallet_balance(
    wallet: "TrackedWallet",
    mint: str,
    client: SolanaClient | None = None,
) -> WalletBalance:
    """Fetch SWEENEE balance for a single wallet."""
    client = client or get_client()
    balance = await client.get_token_balance(wallet.address, mint)

    return WalletBalance(
        address=wallet.address,
        label=wallet.label,
        balance=balance,
    )


async def fetch_all_balances(
    wallets: list["TrackedWallet"],
    mint: str,
    client: SolanaClient | None = None,
) -> list[WalletBalance]:
    """Fetch SWEENEE balances for all tracked wallets.

    Also calculates share_of_tracked for each wallet.
    """
    client = client or get_client()

    logger.info("fetching_balances", wallet_count=len(wallets))

    # Fetch all balances concurrently
    tasks = [fetch_wallet_balance(w, mint, client) for w in wallets]
    results = await asyncio.gather(*tasks)

    # Calculate total and shares
    total_balance = sum(wb.ui_amount for wb in results)

    if total_balance > 0:
        for wb in results:
            wb.share_of_tracked = wb.ui_amount / total_balance

    # Sort by balance descending
    results.sort(key=lambda x: x.ui_amount, reverse=True)

    logger.info(
        "balances_fetched",
        wallet_count=len(results),
        total_balance=total_balance,
        wallets_with_balance=sum(1 for wb in results if wb.ui_amount > 0),
    )

    return results


@dataclass
class BalanceSummary:
    """Summary statistics for tracked wallet balances."""

    total_tracked_wallets: int
    wallets_holding: int
    total_sweenee: float
    largest_holder: WalletBalance | None
    top_10_total: float
    top_10_share: float
    hhi: float  # Herfindahl-Hirschman Index
    fetched_at: datetime


def compute_balance_summary(balances: list[WalletBalance]) -> BalanceSummary:
    """Compute summary statistics from wallet balances."""
    total_wallets = len(balances)
    wallets_holding = sum(1 for wb in balances if wb.ui_amount > 0)
    total_sweenee = sum(wb.ui_amount for wb in balances)

    # Largest holder
    largest = balances[0] if balances and balances[0].ui_amount > 0 else None

    # Top 10
    top_10 = balances[:10]
    top_10_total = sum(wb.ui_amount for wb in top_10)
    top_10_share = top_10_total / total_sweenee if total_sweenee > 0 else 0.0

    # HHI - concentration among tracked wallets
    hhi = 0.0
    if total_sweenee > 0:
        for wb in balances:
            share = wb.ui_amount / total_sweenee
            hhi += share ** 2

    return BalanceSummary(
        total_tracked_wallets=total_wallets,
        wallets_holding=wallets_holding,
        total_sweenee=total_sweenee,
        largest_holder=largest,
        top_10_total=top_10_total,
        top_10_share=top_10_share,
        hhi=hhi,
        fetched_at=datetime.now(timezone.utc),
    )
