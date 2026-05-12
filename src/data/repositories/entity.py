"""
Entity Repository.

Provides data access for wallet entity grouping and management.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import structlog
from sqlalchemy import select, update, delete, func, and_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import Entity, EntityMembership, EntityType, DetectionMethod

logger = structlog.get_logger()


@dataclass
class EntitySummary:
    """Summary of an entity for API responses."""

    entity_id: int
    entity_type: str
    confidence_score: float
    wallet_count: int
    tokens_targeted: int
    is_professional_sybil: bool
    risk_level: Optional[str]
    dominant_funder: Optional[str]


@dataclass
class MembershipInfo:
    """Membership details for a wallet."""

    wallet_address: str
    entity_id: int
    membership_confidence: float
    detected_via: str
    shared_funder_address: Optional[str]


class EntityRepository:
    """
    Repository for entity operations.

    Provides methods to:
    - Create and manage entities
    - Add/remove wallet memberships
    - Find entities by various criteria
    - Merge entities
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_entity(
        self,
        entity_type: str,
        detection_method: str,
        dominant_funder_address: Optional[str] = None,
        confidence_score: float = 0.0,
    ) -> Entity:
        """Create a new entity."""
        entity = Entity(
            entity_type=entity_type,
            detection_method=detection_method,
            dominant_funder_address=dominant_funder_address,
            confidence_score=confidence_score,
            wallet_count=0,
            tokens_targeted=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)

        logger.info(
            "entity_created",
            entity_id=entity.id,
            entity_type=entity_type,
            funder=dominant_funder_address[:8] if dominant_funder_address else None,
        )
        return entity

    async def get_entity(self, entity_id: int) -> Optional[Entity]:
        """Get entity by ID with memberships loaded."""
        stmt = (
            select(Entity)
            .where(Entity.id == entity_id)
            .options(selectinload(Entity.memberships))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_entity_for_wallet(
        self,
        wallet_address: str,
    ) -> Optional[Entity]:
        """Get entity that a wallet belongs to."""
        stmt = (
            select(Entity)
            .join(EntityMembership, Entity.id == EntityMembership.entity_id)
            .where(EntityMembership.wallet_address == wallet_address)
            .options(selectinload(Entity.memberships))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def add_wallet_to_entity(
        self,
        entity_id: int,
        wallet_address: str,
        detected_via: str,
        membership_confidence: float = 0.0,
        shared_funder_address: Optional[str] = None,
        temporal_correlation: Optional[float] = None,
        behavior_similarity: Optional[float] = None,
    ) -> EntityMembership:
        """
        Add a wallet to an entity.

        Uses upsert to handle existing memberships.
        """
        stmt = insert(EntityMembership).values(
            entity_id=entity_id,
            wallet_address=wallet_address,
            detected_via=detected_via,
            membership_confidence=membership_confidence,
            shared_funder_address=shared_funder_address,
            temporal_correlation=temporal_correlation,
            behavior_similarity=behavior_similarity,
            added_at=datetime.now(timezone.utc),
        ).on_conflict_do_update(
            constraint="uq_entity_membership",
            set_={
                "membership_confidence": membership_confidence,
                "detected_via": detected_via,
                "temporal_correlation": temporal_correlation,
                "behavior_similarity": behavior_similarity,
            },
        ).returning(EntityMembership)

        result = await self.session.execute(stmt)
        membership = result.scalar_one()

        # Update entity wallet count
        await self._update_entity_stats(entity_id)
        await self.session.commit()

        return membership

    async def remove_wallet_from_entity(
        self,
        entity_id: int,
        wallet_address: str,
    ) -> bool:
        """Remove a wallet from an entity."""
        stmt = delete(EntityMembership).where(
            and_(
                EntityMembership.entity_id == entity_id,
                EntityMembership.wallet_address == wallet_address,
            )
        )
        result = await self.session.execute(stmt)

        if result.rowcount > 0:
            await self._update_entity_stats(entity_id)
            await self.session.commit()
            return True
        return False

    async def get_entity_wallets(self, entity_id: int) -> list[str]:
        """Get all wallet addresses in an entity."""
        stmt = select(EntityMembership.wallet_address).where(
            EntityMembership.entity_id == entity_id
        )
        result = await self.session.execute(stmt)
        return [row[0] for row in result.fetchall()]

    async def get_entity_memberships(
        self,
        entity_id: int,
    ) -> Sequence[EntityMembership]:
        """Get all memberships for an entity with details."""
        stmt = select(EntityMembership).where(
            EntityMembership.entity_id == entity_id
        ).order_by(EntityMembership.membership_confidence.desc())

        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def find_entities_by_funder(
        self,
        funder_address: str,
    ) -> Sequence[Entity]:
        """Find entities with a specific dominant funder."""
        stmt = (
            select(Entity)
            .where(Entity.dominant_funder_address == funder_address)
            .options(selectinload(Entity.memberships))
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def find_entities_by_type(
        self,
        entity_type: str,
        limit: int = 100,
    ) -> Sequence[Entity]:
        """Find entities of a specific type."""
        stmt = (
            select(Entity)
            .where(Entity.entity_type == entity_type)
            .order_by(Entity.wallet_count.desc())
            .limit(limit)
            .options(selectinload(Entity.memberships))
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def find_professional_sybils(
        self,
        min_wallet_count: int = 5,
        min_tokens_targeted: int = 3,
        limit: int = 100,
    ) -> Sequence[Entity]:
        """Find professional sybil networks."""
        stmt = (
            select(Entity)
            .where(
                and_(
                    Entity.is_professional_sybil == True,
                    Entity.wallet_count >= min_wallet_count,
                    Entity.tokens_targeted >= min_tokens_targeted,
                )
            )
            .order_by(Entity.tokens_targeted.desc())
            .limit(limit)
            .options(selectinload(Entity.memberships))
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def merge_entities(
        self,
        entity_ids: list[int],
        primary_entity_id: Optional[int] = None,
    ) -> Entity:
        """
        Merge multiple entities into one.

        Args:
            entity_ids: List of entity IDs to merge
            primary_entity_id: ID of entity to keep (uses largest if not specified)

        Returns:
            The merged entity
        """
        if len(entity_ids) < 2:
            raise ValueError("Need at least 2 entities to merge")

        # Get all entities
        entities = []
        for eid in entity_ids:
            entity = await self.get_entity(eid)
            if entity:
                entities.append(entity)

        if len(entities) < 2:
            raise ValueError("Could not find enough valid entities")

        # Determine primary entity (largest wallet count)
        if primary_entity_id:
            primary = next((e for e in entities if e.id == primary_entity_id), None)
            if not primary:
                primary = max(entities, key=lambda e: e.wallet_count)
        else:
            primary = max(entities, key=lambda e: e.wallet_count)

        # Move all memberships to primary entity
        other_ids = [e.id for e in entities if e.id != primary.id]

        for other_id in other_ids:
            # Update membership entity_id
            stmt = (
                update(EntityMembership)
                .where(EntityMembership.entity_id == other_id)
                .values(entity_id=primary.id)
            )
            await self.session.execute(stmt)

            # Delete the merged entity
            stmt = delete(Entity).where(Entity.id == other_id)
            await self.session.execute(stmt)

        # Update primary entity stats
        await self._update_entity_stats(primary.id)
        await self.session.commit()

        # Refresh and return
        await self.session.refresh(primary)

        logger.info(
            "entities_merged",
            primary_id=primary.id,
            merged_ids=other_ids,
            final_wallet_count=primary.wallet_count,
        )

        return primary

    async def update_entity_risk(
        self,
        entity_id: int,
        is_professional_sybil: bool,
        risk_level: str,
        avg_coordination_score: Optional[float] = None,
    ) -> Entity:
        """Update entity risk assessment."""
        stmt = (
            update(Entity)
            .where(Entity.id == entity_id)
            .values(
                is_professional_sybil=is_professional_sybil,
                risk_level=risk_level,
                avg_coordination_score=avg_coordination_score,
                updated_at=datetime.now(timezone.utc),
            )
            .returning(Entity)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()

        return result.scalar_one()

    async def update_entity_stats(
        self,
        entity_id: int,
        tokens_targeted: Optional[int] = None,
        total_volume_usd: Optional[float] = None,
    ) -> Entity:
        """Update entity aggregated statistics."""
        values = {"updated_at": datetime.now(timezone.utc)}

        if tokens_targeted is not None:
            values["tokens_targeted"] = tokens_targeted
        if total_volume_usd is not None:
            values["total_volume_usd"] = total_volume_usd

        stmt = (
            update(Entity)
            .where(Entity.id == entity_id)
            .values(**values)
            .returning(Entity)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()

        return result.scalar_one()

    async def delete_entity(self, entity_id: int) -> bool:
        """Delete an entity (memberships cascade)."""
        stmt = delete(Entity).where(Entity.id == entity_id)
        result = await self.session.execute(stmt)
        await self.session.commit()

        return result.rowcount > 0

    async def _update_entity_stats(self, entity_id: int) -> None:
        """Update entity's wallet_count based on memberships."""
        count_stmt = select(func.count(EntityMembership.id)).where(
            EntityMembership.entity_id == entity_id
        )
        result = await self.session.execute(count_stmt)
        count = result.scalar() or 0

        update_stmt = (
            update(Entity)
            .where(Entity.id == entity_id)
            .values(wallet_count=count, updated_at=datetime.now(timezone.utc))
        )
        await self.session.execute(update_stmt)

    async def get_entities_for_wallets(
        self,
        wallet_addresses: list[str],
    ) -> dict[str, Entity]:
        """
        Get entities for multiple wallets.

        Returns:
            Dict mapping wallet_address -> Entity (if member of one)
        """
        stmt = (
            select(EntityMembership.wallet_address, Entity)
            .join(Entity, Entity.id == EntityMembership.entity_id)
            .where(EntityMembership.wallet_address.in_(wallet_addresses))
        )
        result = await self.session.execute(stmt)

        return {row[0]: row[1] for row in result.fetchall()}

    async def get_summary(self, entity_id: int) -> Optional[EntitySummary]:
        """Get entity summary for API responses."""
        entity = await self.get_entity(entity_id)
        if not entity:
            return None

        return EntitySummary(
            entity_id=entity.id,
            entity_type=entity.entity_type,
            confidence_score=entity.confidence_score,
            wallet_count=entity.wallet_count,
            tokens_targeted=entity.tokens_targeted,
            is_professional_sybil=entity.is_professional_sybil,
            risk_level=entity.risk_level,
            dominant_funder=entity.dominant_funder_address,
        )
