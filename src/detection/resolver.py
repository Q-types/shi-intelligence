"""
Entity Resolver.

Orchestrates detection algorithms to create and manage wallet entities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..data.repositories import EntityRepository
from ..data.models import Entity
from ..graph.funding_graph import FundingGraph
from .shared_funder import SharedFunderDetector, FunderCluster
from .temporal_sync import TemporalSyncDetector, TemporalCluster, TradeEvent

logger = structlog.get_logger()


@dataclass
class ResolutionCandidate:
    """A candidate entity from detection."""

    wallet_addresses: list[str]
    detection_method: str  # shared_funder, temporal_sync, combined
    confidence: float
    dominant_funder: Optional[str] = None
    coordination_score: Optional[float] = None
    supporting_evidence: dict = field(default_factory=dict)


@dataclass
class ResolutionResult:
    """Result of entity resolution."""

    entities_created: int
    entities_merged: int
    wallets_assigned: int
    candidates_processed: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EntityResolver:
    """
    Orchestrates entity detection and persistence.

    Combines outputs from SharedFunderDetector and TemporalSyncDetector
    to create unified wallet entities. Handles:
    - De-duplication of overlapping clusters
    - Merging of related entities
    - Confidence aggregation from multiple signals
    - Persistence to database via EntityRepository
    """

    def __init__(
        self,
        session: AsyncSession,
        shared_funder_detector: Optional[SharedFunderDetector] = None,
        temporal_sync_detector: Optional[TemporalSyncDetector] = None,
        min_confidence_to_create: float = 0.6,
        merge_overlap_threshold: float = 0.5,
    ):
        self.session = session
        self.repository = EntityRepository(session)
        self.shared_funder_detector = shared_funder_detector or SharedFunderDetector()
        self.temporal_sync_detector = temporal_sync_detector or TemporalSyncDetector()
        self.min_confidence_to_create = min_confidence_to_create
        self.merge_overlap_threshold = merge_overlap_threshold

    async def resolve(
        self,
        funding_graph: Optional[FundingGraph] = None,
        trade_events: Optional[Sequence[TradeEvent]] = None,
        target_wallets: Optional[list[str]] = None,
    ) -> ResolutionResult:
        """
        Run full entity resolution pipeline.

        Args:
            funding_graph: Graph for shared funder detection
            trade_events: Events for temporal sync detection
            target_wallets: Specific wallets to analyze (default: all)

        Returns:
            ResolutionResult with statistics
        """
        candidates = []

        # Run shared funder detection
        if funding_graph is not None:
            funder_result = self.shared_funder_detector.detect(
                funding_graph, target_wallets
            )
            for cluster in funder_result.clusters:
                candidates.append(
                    ResolutionCandidate(
                        wallet_addresses=cluster.wallet_addresses,
                        detection_method="shared_funder",
                        confidence=cluster.confidence,
                        dominant_funder=cluster.funder_address,
                        supporting_evidence={
                            "funding_depth": cluster.funding_depth,
                            "total_funded_amount": cluster.total_funded_amount,
                            "time_span_hours": cluster.funding_time_span_hours,
                        },
                    )
                )

        # Run temporal sync detection
        if trade_events is not None:
            temporal_result = self.temporal_sync_detector.detect(
                trade_events, target_wallets
            )
            for cluster in temporal_result.clusters:
                candidates.append(
                    ResolutionCandidate(
                        wallet_addresses=cluster.wallet_addresses,
                        detection_method="temporal_sync",
                        confidence=cluster.confidence,
                        coordination_score=cluster.coordination_score,
                        supporting_evidence={
                            "tokens_coordinated": cluster.tokens_coordinated,
                            "coordination_events": cluster.coordination_events,
                            "avg_time_gap_seconds": cluster.avg_time_gap_seconds,
                        },
                    )
                )

        # Merge overlapping candidates
        merged_candidates = self._merge_overlapping_candidates(candidates)

        # Filter by confidence
        viable_candidates = [
            c for c in merged_candidates if c.confidence >= self.min_confidence_to_create
        ]

        # Create/update entities
        entities_created = 0
        entities_merged = 0
        wallets_assigned = 0

        for candidate in viable_candidates:
            result = await self._process_candidate(candidate)
            entities_created += result["created"]
            entities_merged += result["merged"]
            wallets_assigned += result["assigned"]

        logger.info(
            "entity_resolution_complete",
            candidates_total=len(candidates),
            candidates_merged=len(merged_candidates),
            candidates_viable=len(viable_candidates),
            entities_created=entities_created,
            entities_merged=entities_merged,
            wallets_assigned=wallets_assigned,
        )

        return ResolutionResult(
            entities_created=entities_created,
            entities_merged=entities_merged,
            wallets_assigned=wallets_assigned,
            candidates_processed=len(viable_candidates),
        )

    def _merge_overlapping_candidates(
        self,
        candidates: list[ResolutionCandidate],
    ) -> list[ResolutionCandidate]:
        """
        Merge candidates with significant wallet overlap.

        Uses overlap threshold to determine when to merge.
        """
        if not candidates:
            return []

        # Sort by confidence descending
        sorted_candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)

        merged = []
        used_indices = set()

        for i, candidate in enumerate(sorted_candidates):
            if i in used_indices:
                continue

            wallet_set = set(candidate.wallet_addresses)
            merge_group = [candidate]

            # Find overlapping candidates to merge
            for j, other in enumerate(sorted_candidates[i + 1 :], start=i + 1):
                if j in used_indices:
                    continue

                other_wallets = set(other.wallet_addresses)
                overlap = len(wallet_set & other_wallets)
                min_size = min(len(wallet_set), len(other_wallets))

                if min_size > 0 and overlap / min_size >= self.merge_overlap_threshold:
                    merge_group.append(other)
                    wallet_set.update(other_wallets)
                    used_indices.add(j)

            # Create merged candidate
            if len(merge_group) == 1:
                merged.append(candidate)
            else:
                merged.append(self._combine_candidates(merge_group))

            used_indices.add(i)

        return merged

    def _combine_candidates(
        self,
        candidates: list[ResolutionCandidate],
    ) -> ResolutionCandidate:
        """Combine multiple candidates into one."""
        # Combine wallet addresses
        all_wallets = set()
        for c in candidates:
            all_wallets.update(c.wallet_addresses)

        # Determine detection method
        methods = set(c.detection_method for c in candidates)
        if len(methods) > 1:
            detection_method = "combined"
        else:
            detection_method = methods.pop()

        # Average confidence (weighted by cluster size)
        total_weight = sum(len(c.wallet_addresses) for c in candidates)
        weighted_confidence = sum(
            c.confidence * len(c.wallet_addresses) for c in candidates
        ) / total_weight

        # Take dominant funder from highest confidence shared_funder candidate
        dominant_funder = None
        for c in sorted(candidates, key=lambda x: x.confidence, reverse=True):
            if c.dominant_funder:
                dominant_funder = c.dominant_funder
                break

        # Take coordination score from highest confidence temporal_sync candidate
        coordination_score = None
        for c in sorted(candidates, key=lambda x: x.confidence, reverse=True):
            if c.coordination_score:
                coordination_score = c.coordination_score
                break

        # Combine evidence
        combined_evidence = {
            "merged_from": [c.detection_method for c in candidates],
            "original_confidences": [c.confidence for c in candidates],
        }

        return ResolutionCandidate(
            wallet_addresses=list(all_wallets),
            detection_method=detection_method,
            confidence=weighted_confidence,
            dominant_funder=dominant_funder,
            coordination_score=coordination_score,
            supporting_evidence=combined_evidence,
        )

    async def _process_candidate(
        self,
        candidate: ResolutionCandidate,
    ) -> dict:
        """
        Process a candidate into entity.

        Returns dict with created/merged/assigned counts.
        """
        result = {"created": 0, "merged": 0, "assigned": 0}

        # Check if any wallets already belong to entities
        existing_entities = await self.repository.get_entities_for_wallets(
            candidate.wallet_addresses
        )

        existing_entity_ids = set(e.id for e in existing_entities.values())

        if not existing_entity_ids:
            # Create new entity
            entity = await self._create_entity_from_candidate(candidate)
            result["created"] = 1
            result["assigned"] = len(candidate.wallet_addresses)

        elif len(existing_entity_ids) == 1:
            # Add new wallets to existing entity
            entity_id = existing_entity_ids.pop()
            new_wallets = [
                w for w in candidate.wallet_addresses if w not in existing_entities
            ]

            for wallet in new_wallets:
                await self.repository.add_wallet_to_entity(
                    entity_id=entity_id,
                    wallet_address=wallet,
                    detected_via=candidate.detection_method,
                    membership_confidence=candidate.confidence,
                    shared_funder_address=candidate.dominant_funder,
                    temporal_correlation=candidate.coordination_score,
                )
                result["assigned"] += 1

        else:
            # Merge multiple entities
            entity_ids = list(existing_entity_ids)
            merged_entity = await self.repository.merge_entities(entity_ids)
            result["merged"] = len(entity_ids) - 1

            # Add any new wallets
            new_wallets = [
                w for w in candidate.wallet_addresses if w not in existing_entities
            ]

            for wallet in new_wallets:
                await self.repository.add_wallet_to_entity(
                    entity_id=merged_entity.id,
                    wallet_address=wallet,
                    detected_via=candidate.detection_method,
                    membership_confidence=candidate.confidence,
                    shared_funder_address=candidate.dominant_funder,
                    temporal_correlation=candidate.coordination_score,
                )
                result["assigned"] += 1

        return result

    async def _create_entity_from_candidate(
        self,
        candidate: ResolutionCandidate,
    ) -> Entity:
        """Create a new entity from a resolution candidate."""
        # Determine entity type based on signals
        entity_type = self._determine_entity_type(candidate)

        # Create entity
        entity = await self.repository.create_entity(
            entity_type=entity_type,
            detection_method=candidate.detection_method,
            dominant_funder_address=candidate.dominant_funder,
            confidence_score=candidate.confidence,
        )

        # Add all wallets
        for wallet in candidate.wallet_addresses:
            await self.repository.add_wallet_to_entity(
                entity_id=entity.id,
                wallet_address=wallet,
                detected_via=candidate.detection_method,
                membership_confidence=candidate.confidence,
                shared_funder_address=candidate.dominant_funder,
                temporal_correlation=candidate.coordination_score,
            )

        return entity

    def _determine_entity_type(self, candidate: ResolutionCandidate) -> str:
        """Determine entity type based on detection signals."""
        # High coordination score + shared funder = likely sybil
        if (
            candidate.coordination_score
            and candidate.coordination_score > 0.7
            and candidate.dominant_funder
        ):
            return "sybil_cluster"

        # Just shared funder = could be whale distributing
        if candidate.dominant_funder and not candidate.coordination_score:
            return "whale_group"

        # Just temporal sync = could be trading group
        if candidate.coordination_score and not candidate.dominant_funder:
            return "trading_group"

        # Combined signals with moderate confidence
        if candidate.detection_method == "combined":
            return "sybil_cluster"

        return "unknown"

    async def resolve_single_wallet(
        self,
        wallet_address: str,
        funding_graph: Optional[FundingGraph] = None,
        trade_events: Optional[Sequence[TradeEvent]] = None,
    ) -> Optional[Entity]:
        """
        Find or create entity for a single wallet.

        Checks if wallet belongs to an existing entity, if not
        attempts to detect related wallets and create entity.
        """
        # Check existing membership
        existing = await self.repository.get_entity_for_wallet(wallet_address)
        if existing:
            return existing

        # Try to detect clusters including this wallet
        target = [wallet_address]

        # Expand target to related wallets if we have a funding graph
        if funding_graph:
            # Get wallets funded by same source
            funders = funding_graph.get_funders(wallet_address)
            for funder in funders:
                siblings = funding_graph.get_funded_by(funder)
                target.extend(siblings)

            target = list(set(target))

        # Run resolution on expanded target
        await self.resolve(
            funding_graph=funding_graph,
            trade_events=trade_events,
            target_wallets=target,
        )

        # Return entity if one was created
        return await self.repository.get_entity_for_wallet(wallet_address)
