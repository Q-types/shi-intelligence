"""
Wallet History Repository.

Provides data access for cross-token wallet behavior tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import structlog
from sqlalchemy import select, update, func, and_, or_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import WalletHistory, ConfidenceLevel

logger = structlog.get_logger()


@dataclass
class TokenInteraction:
    """Summary of a wallet's interaction with a token."""

    wallet_address: str
    token_mint: str
    first_seen_at: datetime
    last_seen_at: Optional[datetime]
    archetype: Optional[str]
    holding_duration_days: Optional[int]
    realized_pnl_pct: Optional[float]
    was_sniper: bool
    was_accumulator: bool
    token_rugged: bool


@dataclass
class WalletBehaviorSummary:
    """Cross-token behavior summary for a wallet."""

    wallet_address: str
    tokens_analyzed: int
    sniper_count: int
    accumulator_count: int
    rugpull_count: int
    avg_holding_days: Optional[float]
    avg_pnl_pct: Optional[float]


class WalletHistoryRepository:
    """
    Repository for wallet history operations.

    Provides methods to:
    - Record wallet-token interactions
    - Query wallet behavior across tokens
    - Find patterns (serial snipers, diamond hands, etc.)
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def record_interaction(
        self,
        wallet_address: str,
        token_mint: str,
        first_seen_at: datetime,
        archetype: Optional[str] = None,
        archetype_confidence: Optional[float] = None,
        entry_price_usd: Optional[float] = None,
        max_balance: Optional[int] = None,
        max_share_pct: Optional[float] = None,
        trade_count: int = 0,
    ) -> WalletHistory:
        """
        Record or update a wallet's interaction with a token.

        Uses upsert to handle both new and existing records.
        """
        # Determine pattern flags
        was_sniper = archetype == "sniper" if archetype else False
        was_accumulator = archetype == "long_term_accumulator" if archetype else False

        stmt = insert(WalletHistory).values(
            wallet_address=wallet_address,
            token_mint=token_mint,
            first_seen_at=first_seen_at,
            last_seen_at=datetime.now(timezone.utc),
            archetype_assigned=archetype,
            archetype_confidence=archetype_confidence,
            entry_price_usd=entry_price_usd,
            max_balance=max_balance,
            max_share_pct=max_share_pct,
            trade_count=trade_count,
            was_sniper=was_sniper,
            was_accumulator=was_accumulator,
            updated_at=datetime.now(timezone.utc),
        ).on_conflict_do_update(
            constraint="uq_wallet_history_wallet_token",
            set_={
                "last_seen_at": datetime.now(timezone.utc),
                "archetype_assigned": archetype,
                "archetype_confidence": archetype_confidence,
                "max_balance": max_balance,
                "max_share_pct": max_share_pct,
                "trade_count": trade_count,
                "was_sniper": was_sniper,
                "was_accumulator": was_accumulator,
                "updated_at": datetime.now(timezone.utc),
            },
        ).returning(WalletHistory)

        result = await self.session.execute(stmt)
        await self.session.commit()

        return result.scalar_one()

    async def get_wallet_history(
        self,
        wallet_address: str,
        limit: int = 100,
    ) -> Sequence[WalletHistory]:
        """Get all token interactions for a wallet."""
        stmt = (
            select(WalletHistory)
            .where(WalletHistory.wallet_address == wallet_address)
            .order_by(WalletHistory.first_seen_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_wallet_tokens(self, wallet_address: str) -> list[str]:
        """Get list of tokens a wallet has interacted with."""
        stmt = select(WalletHistory.token_mint).where(
            WalletHistory.wallet_address == wallet_address
        )
        result = await self.session.execute(stmt)
        return [row[0] for row in result.fetchall()]

    async def get_token_wallets(self, token_mint: str) -> list[str]:
        """Get list of wallets that have interacted with a token."""
        stmt = select(WalletHistory.wallet_address).where(
            WalletHistory.token_mint == token_mint
        )
        result = await self.session.execute(stmt)
        return [row[0] for row in result.fetchall()]

    async def get_behavior_summary(
        self,
        wallet_address: str,
    ) -> Optional[WalletBehaviorSummary]:
        """Get aggregated behavior summary for a wallet."""
        stmt = select(
            func.count(WalletHistory.id).label("tokens_analyzed"),
            func.sum(func.cast(WalletHistory.was_sniper, Integer)).label("sniper_count"),
            func.sum(func.cast(WalletHistory.was_accumulator, Integer)).label("accumulator_count"),
            func.sum(func.cast(WalletHistory.token_rugged, Integer)).label("rugpull_count"),
            func.avg(WalletHistory.holding_duration_days).label("avg_holding_days"),
            func.avg(WalletHistory.realized_pnl_pct).label("avg_pnl_pct"),
        ).where(WalletHistory.wallet_address == wallet_address)

        result = await self.session.execute(stmt)
        row = result.fetchone()

        if not row or row.tokens_analyzed == 0:
            return None

        return WalletBehaviorSummary(
            wallet_address=wallet_address,
            tokens_analyzed=row.tokens_analyzed or 0,
            sniper_count=row.sniper_count or 0,
            accumulator_count=row.accumulator_count or 0,
            rugpull_count=row.rugpull_count or 0,
            avg_holding_days=row.avg_holding_days,
            avg_pnl_pct=row.avg_pnl_pct,
        )

    async def find_serial_snipers(
        self,
        min_sniper_count: int = 3,
        limit: int = 100,
    ) -> list[tuple[str, int]]:
        """
        Find wallets that have been snipers on multiple tokens.

        Returns:
            List of (wallet_address, sniper_count) tuples
        """
        stmt = (
            select(
                WalletHistory.wallet_address,
                func.count(WalletHistory.id).label("sniper_count"),
            )
            .where(WalletHistory.was_sniper == True)
            .group_by(WalletHistory.wallet_address)
            .having(func.count(WalletHistory.id) >= min_sniper_count)
            .order_by(func.count(WalletHistory.id).desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [(row[0], row[1]) for row in result.fetchall()]

    async def find_diamond_hands(
        self,
        min_accumulator_count: int = 5,
        min_avg_holding_days: float = 30.0,
        limit: int = 100,
    ) -> list[tuple[str, int, float]]:
        """
        Find wallets with consistent long-term accumulator behavior.

        Returns:
            List of (wallet_address, accumulator_count, avg_holding_days) tuples
        """
        stmt = (
            select(
                WalletHistory.wallet_address,
                func.count(WalletHistory.id).label("acc_count"),
                func.avg(WalletHistory.holding_duration_days).label("avg_days"),
            )
            .where(WalletHistory.was_accumulator == True)
            .group_by(WalletHistory.wallet_address)
            .having(
                and_(
                    func.count(WalletHistory.id) >= min_accumulator_count,
                    func.avg(WalletHistory.holding_duration_days) >= min_avg_holding_days,
                )
            )
            .order_by(func.avg(WalletHistory.holding_duration_days).desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [(row[0], row[1], row[2]) for row in result.fetchall()]

    async def find_wallets_with_shared_tokens(
        self,
        wallet_addresses: list[str],
        min_shared_tokens: int = 2,
    ) -> dict[tuple[str, str], list[str]]:
        """
        Find token overlaps between wallets.

        Returns:
            Dict mapping (wallet1, wallet2) -> list of shared token mints
        """
        if len(wallet_addresses) < 2:
            return {}

        # Get all token interactions for these wallets
        stmt = select(
            WalletHistory.wallet_address,
            WalletHistory.token_mint,
        ).where(WalletHistory.wallet_address.in_(wallet_addresses))

        result = await self.session.execute(stmt)
        rows = result.fetchall()

        # Build wallet -> tokens mapping
        wallet_tokens: dict[str, set[str]] = {}
        for wallet, token in rows:
            if wallet not in wallet_tokens:
                wallet_tokens[wallet] = set()
            wallet_tokens[wallet].add(token)

        # Find overlaps
        overlaps: dict[tuple[str, str], list[str]] = {}
        wallets = list(wallet_tokens.keys())

        for i, w1 in enumerate(wallets):
            for w2 in wallets[i + 1:]:
                shared = wallet_tokens[w1] & wallet_tokens[w2]
                if len(shared) >= min_shared_tokens:
                    overlaps[(w1, w2)] = list(shared)

        return overlaps

    async def mark_token_rugged(self, token_mint: str) -> int:
        """
        Mark all wallet interactions with a token as rugged.

        Returns:
            Number of records updated
        """
        stmt = (
            update(WalletHistory)
            .where(WalletHistory.token_mint == token_mint)
            .values(token_rugged=True, updated_at=datetime.now(timezone.utc))
        )
        result = await self.session.execute(stmt)
        await self.session.commit()

        logger.info("marked_token_rugged", token=token_mint, affected=result.rowcount)
        return result.rowcount

    async def update_exit_data(
        self,
        wallet_address: str,
        token_mint: str,
        exit_price_usd: float,
        realized_pnl_pct: float,
        holding_duration_days: int,
        was_early_exit: bool = False,
    ) -> Optional[WalletHistory]:
        """Update exit data when a wallet sells a token."""
        stmt = (
            update(WalletHistory)
            .where(
                and_(
                    WalletHistory.wallet_address == wallet_address,
                    WalletHistory.token_mint == token_mint,
                )
            )
            .values(
                exit_price_usd=exit_price_usd,
                realized_pnl_pct=realized_pnl_pct,
                holding_duration_days=holding_duration_days,
                was_early_exit=was_early_exit,
                last_seen_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            .returning(WalletHistory)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()

        return result.scalar_one_or_none()


# Import Integer for cast
from sqlalchemy import Integer
