"""
Wallet Reputation Repository.

Provides data access for cross-token wallet reputation scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import structlog
from sqlalchemy import select, update, func, and_, or_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import WalletReputation, ConfidenceLevel

logger = structlog.get_logger()


@dataclass
class ReputationSummary:
    """Summary of a wallet's reputation."""

    wallet_address: str
    reputation_score: int
    confidence_level: str
    tokens_analyzed: int
    patterns: list[dict]
    is_known_bad_actor: bool
    is_known_good_actor: bool
    entity_id: Optional[int]


class WalletReputationRepository:
    """
    Repository for wallet reputation operations.

    Provides methods to:
    - Create and update reputation scores
    - Query wallets by reputation ranges
    - Manage detected patterns
    - Link wallets to entities
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_reputation(
        self,
        wallet_address: str,
    ) -> Optional[WalletReputation]:
        """Get reputation record for a wallet."""
        stmt = select(WalletReputation).where(
            WalletReputation.wallet_address == wallet_address
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_or_update_reputation(
        self,
        wallet_address: str,
        reputation_score: int,
        confidence_level: str,
        tokens_analyzed: int = 0,
        sniper_count: int = 0,
        accumulator_count: int = 0,
        rugpull_count: int = 0,
        early_exit_count: int = 0,
        avg_holding_days: Optional[float] = None,
        avg_pnl_pct: Optional[float] = None,
        total_volume_usd: Optional[float] = None,
        patterns: Optional[list] = None,
        entity_id: Optional[int] = None,
    ) -> WalletReputation:
        """
        Create or update a wallet's reputation.

        Uses upsert to handle both new and existing records.
        """
        values = {
            "wallet_address": wallet_address,
            "reputation_score": reputation_score,
            "confidence_level": confidence_level,
            "tokens_analyzed": tokens_analyzed,
            "sniper_count": sniper_count,
            "accumulator_count": accumulator_count,
            "rugpull_count": rugpull_count,
            "early_exit_count": early_exit_count,
            "avg_holding_days": avg_holding_days,
            "avg_pnl_pct": avg_pnl_pct,
            "total_volume_usd": total_volume_usd,
            "patterns": patterns or [],
            "entity_id": entity_id,
            "last_updated": datetime.now(timezone.utc),
        }

        stmt = insert(WalletReputation).values(**values).on_conflict_do_update(
            index_elements=["wallet_address"],
            set_={
                "reputation_score": values["reputation_score"],
                "confidence_level": values["confidence_level"],
                "tokens_analyzed": values["tokens_analyzed"],
                "sniper_count": values["sniper_count"],
                "accumulator_count": values["accumulator_count"],
                "rugpull_count": values["rugpull_count"],
                "early_exit_count": values["early_exit_count"],
                "avg_holding_days": values["avg_holding_days"],
                "avg_pnl_pct": values["avg_pnl_pct"],
                "total_volume_usd": values["total_volume_usd"],
                "patterns": values["patterns"],
                "entity_id": values["entity_id"],
                "last_updated": values["last_updated"],
            },
        ).returning(WalletReputation)

        result = await self.session.execute(stmt)
        await self.session.commit()

        return result.scalar_one()

    async def update_score(
        self,
        wallet_address: str,
        reputation_score: int,
        confidence_level: Optional[str] = None,
    ) -> Optional[WalletReputation]:
        """Update just the reputation score."""
        values = {
            "reputation_score": reputation_score,
            "last_updated": datetime.now(timezone.utc),
        }
        if confidence_level:
            values["confidence_level"] = confidence_level

        stmt = (
            update(WalletReputation)
            .where(WalletReputation.wallet_address == wallet_address)
            .values(**values)
            .returning(WalletReputation)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()

        return result.scalar_one_or_none()

    async def add_pattern(
        self,
        wallet_address: str,
        pattern_type: str,
        confidence: float,
        metadata: Optional[dict] = None,
    ) -> Optional[WalletReputation]:
        """Add a detected pattern to a wallet's reputation."""
        # Get current patterns
        rep = await self.get_reputation(wallet_address)
        if not rep:
            return None

        current_patterns = rep.patterns or []

        # Check if pattern type already exists
        existing = next(
            (p for p in current_patterns if p.get("type") == pattern_type), None
        )

        if existing:
            # Update existing pattern
            existing["confidence"] = confidence
            existing["detected_at"] = datetime.now(timezone.utc).isoformat()
            if metadata:
                existing.update(metadata)
        else:
            # Add new pattern
            new_pattern = {
                "type": pattern_type,
                "confidence": confidence,
                "detected_at": datetime.now(timezone.utc).isoformat(),
            }
            if metadata:
                new_pattern.update(metadata)
            current_patterns.append(new_pattern)

        # Update record
        stmt = (
            update(WalletReputation)
            .where(WalletReputation.wallet_address == wallet_address)
            .values(patterns=current_patterns, last_updated=datetime.now(timezone.utc))
            .returning(WalletReputation)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()

        return result.scalar_one_or_none()

    async def mark_bad_actor(
        self,
        wallet_address: str,
        reason: Optional[str] = None,
    ) -> Optional[WalletReputation]:
        """Mark a wallet as a known bad actor."""
        values = {
            "is_known_bad_actor": True,
            "is_known_good_actor": False,
            "last_updated": datetime.now(timezone.utc),
        }

        stmt = (
            update(WalletReputation)
            .where(WalletReputation.wallet_address == wallet_address)
            .values(**values)
            .returning(WalletReputation)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()

        rep = result.scalar_one_or_none()

        if rep and reason:
            await self.add_pattern(wallet_address, "BAD_ACTOR", 1.0, {"reason": reason})

        return rep

    async def mark_good_actor(
        self,
        wallet_address: str,
        reason: Optional[str] = None,
    ) -> Optional[WalletReputation]:
        """Mark a wallet as a known good actor."""
        values = {
            "is_known_good_actor": True,
            "is_known_bad_actor": False,
            "last_updated": datetime.now(timezone.utc),
        }

        stmt = (
            update(WalletReputation)
            .where(WalletReputation.wallet_address == wallet_address)
            .values(**values)
            .returning(WalletReputation)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()

        rep = result.scalar_one_or_none()

        if rep and reason:
            await self.add_pattern(wallet_address, "GOOD_ACTOR", 1.0, {"reason": reason})

        return rep

    async def link_to_entity(
        self,
        wallet_address: str,
        entity_id: int,
    ) -> Optional[WalletReputation]:
        """Link a wallet's reputation to an entity."""
        stmt = (
            update(WalletReputation)
            .where(WalletReputation.wallet_address == wallet_address)
            .values(entity_id=entity_id, last_updated=datetime.now(timezone.utc))
            .returning(WalletReputation)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()

        return result.scalar_one_or_none()

    async def find_by_score_range(
        self,
        min_score: int = 0,
        max_score: int = 100,
        limit: int = 100,
    ) -> Sequence[WalletReputation]:
        """Find wallets within a reputation score range."""
        stmt = (
            select(WalletReputation)
            .where(
                and_(
                    WalletReputation.reputation_score >= min_score,
                    WalletReputation.reputation_score <= max_score,
                )
            )
            .order_by(WalletReputation.reputation_score.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def find_high_risk(
        self,
        max_score: int = 30,
        limit: int = 100,
    ) -> Sequence[WalletReputation]:
        """Find high-risk wallets (low reputation scores)."""
        stmt = (
            select(WalletReputation)
            .where(
                or_(
                    WalletReputation.reputation_score <= max_score,
                    WalletReputation.is_known_bad_actor == True,
                )
            )
            .order_by(WalletReputation.reputation_score.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def find_by_pattern(
        self,
        pattern_type: str,
        min_confidence: float = 0.5,
        limit: int = 100,
    ) -> Sequence[WalletReputation]:
        """Find wallets with a specific detected pattern."""
        # Use JSONB containment operator
        stmt = (
            select(WalletReputation)
            .where(
                WalletReputation.patterns.contains([{"type": pattern_type}])
            )
            .limit(limit)
        )
        result = await self.session.execute(stmt)

        # Filter by confidence in Python (JSONB path queries are complex)
        all_reps = result.scalars().all()
        return [
            r
            for r in all_reps
            if any(
                p.get("type") == pattern_type and p.get("confidence", 0) >= min_confidence
                for p in (r.patterns or [])
            )
        ]

    async def find_serial_snipers(
        self,
        min_sniper_count: int = 3,
        limit: int = 100,
    ) -> Sequence[WalletReputation]:
        """Find wallets with multiple sniper classifications."""
        stmt = (
            select(WalletReputation)
            .where(WalletReputation.sniper_count >= min_sniper_count)
            .order_by(WalletReputation.sniper_count.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_summary(self, wallet_address: str) -> Optional[ReputationSummary]:
        """Get reputation summary for API responses."""
        rep = await self.get_reputation(wallet_address)
        if not rep:
            return None

        return ReputationSummary(
            wallet_address=rep.wallet_address,
            reputation_score=rep.reputation_score,
            confidence_level=rep.confidence_level,
            tokens_analyzed=rep.tokens_analyzed,
            patterns=rep.patterns or [],
            is_known_bad_actor=rep.is_known_bad_actor,
            is_known_good_actor=rep.is_known_good_actor,
            entity_id=rep.entity_id,
        )

    async def get_reputations_for_wallets(
        self,
        wallet_addresses: list[str],
    ) -> dict[str, WalletReputation]:
        """Get reputations for multiple wallets."""
        stmt = select(WalletReputation).where(
            WalletReputation.wallet_address.in_(wallet_addresses)
        )
        result = await self.session.execute(stmt)

        return {r.wallet_address: r for r in result.scalars().all()}

    async def bulk_update_scores(
        self,
        updates: list[tuple[str, int, str]],
    ) -> int:
        """
        Bulk update reputation scores.

        Args:
            updates: List of (wallet_address, score, confidence_level) tuples

        Returns:
            Number of records updated
        """
        updated_count = 0

        for wallet_address, score, confidence in updates:
            result = await self.update_score(wallet_address, score, confidence)
            if result:
                updated_count += 1

        return updated_count
