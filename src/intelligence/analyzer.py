"""
Cross-Token Intelligence Analyzer.

Main orchestrator for cross-token wallet analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..data.repositories import (
    EntityRepository,
    WalletHistoryRepository,
    WalletReputationRepository,
)
from ..data.models import Entity, WalletReputation
from ..detection import (
    EntityResolver,
    SharedFunderDetector,
    TemporalSyncDetector,
    SybilNetworkDetector,
    SybilAssessment,
)
from ..detection.temporal_sync import TradeEvent
from ..graph.funding_graph import FundingGraph
from ..reputation import ReputationScorer, PatternDetector

logger = structlog.get_logger()


@dataclass
class WalletIntelligence:
    """Intelligence summary for a single wallet."""

    wallet_address: str
    reputation_score: int
    risk_level: str
    confidence_level: str
    patterns: list[dict]
    entity_id: Optional[int]
    entity_type: Optional[str]
    is_sybil_member: bool
    tokens_analyzed: int


@dataclass
class EntityIntelligence:
    """Intelligence summary for an entity."""

    entity_id: int
    entity_type: str
    wallet_count: int
    tokens_targeted: int
    coordination_score: float
    is_professional_sybil: bool
    risk_level: str
    dominant_funder: Optional[str]
    wallet_addresses: list[str]


@dataclass
class IntelligenceReport:
    """Full cross-token intelligence report."""

    # Analysis metadata
    analysis_id: str
    token_mint: Optional[str]
    analyzed_at: datetime
    wallets_analyzed: int

    # Entity findings
    entities_found: int
    entities: list[EntityIntelligence]

    # Sybil findings
    sybil_networks_found: int
    sybil_wallets_count: int

    # Wallet summaries (high risk only)
    high_risk_wallets: list[WalletIntelligence]

    # Aggregate metrics
    avg_reputation_score: float
    risk_distribution: dict[str, int]  # risk_level -> count

    # Recommendations
    recommendations: list[str]


class CrossTokenAnalyzer:
    """
    Main orchestrator for cross-token intelligence.

    Provides unified interface for:
    - Entity detection and resolution
    - Sybil network identification
    - Wallet reputation scoring
    - Intelligence report generation

    Usage:
        analyzer = CrossTokenAnalyzer(session)

        # Full analysis for a token's holders
        report = await analyzer.analyze_token_holders(
            token_mint="...",
            wallet_addresses=[...],
            funding_graph=graph,
            trade_events=events,
        )

        # Quick lookup for single wallet
        intel = await analyzer.get_wallet_intelligence("wallet_address")
    """

    def __init__(
        self,
        session: AsyncSession,
    ):
        self.session = session

        # Repositories
        self.entity_repo = EntityRepository(session)
        self.history_repo = WalletHistoryRepository(session)
        self.reputation_repo = WalletReputationRepository(session)

        # Detection services
        self.shared_funder_detector = SharedFunderDetector()
        self.temporal_sync_detector = TemporalSyncDetector()
        self.entity_resolver = EntityResolver(
            session,
            self.shared_funder_detector,
            self.temporal_sync_detector,
        )
        self.sybil_detector = SybilNetworkDetector(session)

        # Reputation services
        self.pattern_detector = PatternDetector()
        self.reputation_scorer = ReputationScorer(session, self.pattern_detector)

    async def analyze_token_holders(
        self,
        wallet_addresses: list[str],
        funding_graph: Optional[FundingGraph] = None,
        trade_events: Optional[Sequence[TradeEvent]] = None,
        token_mint: Optional[str] = None,
    ) -> IntelligenceReport:
        """
        Run full cross-token intelligence analysis.

        Args:
            wallet_addresses: Wallets to analyze (typically token holders)
            funding_graph: Funding graph for entity detection
            trade_events: Trade events for temporal analysis
            token_mint: Optional token mint for context

        Returns:
            IntelligenceReport with full analysis results
        """
        analysis_id = f"cta-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        logger.info(
            "cross_token_analysis_started",
            analysis_id=analysis_id,
            wallet_count=len(wallet_addresses),
            has_funding_graph=funding_graph is not None,
            has_trade_events=trade_events is not None,
        )

        # Step 1: Entity Resolution
        resolution_result = await self.entity_resolver.resolve(
            funding_graph=funding_graph,
            trade_events=trade_events,
            target_wallets=wallet_addresses,
        )

        # Step 2: Get entities for these wallets
        wallet_entities = await self.entity_repo.get_entities_for_wallets(
            wallet_addresses
        )

        # Step 3: Sybil Assessment for detected entities
        entity_ids = set(e.id for e in wallet_entities.values())
        sybil_assessments: dict[int, SybilAssessment] = {}

        for entity_id in entity_ids:
            assessment = await self.sybil_detector.assess_and_update(entity_id)
            if assessment:
                sybil_assessments[entity_id] = assessment

        # Step 4: Score all wallets
        entity_mapping = {
            wallet: entity.id for wallet, entity in wallet_entities.items()
        }
        sybil_entities = {
            eid for eid, a in sybil_assessments.items() if a.is_professional_sybil
        }

        reputations = await self.reputation_scorer.score_and_persist_batch(
            wallet_addresses,
            entity_mapping=entity_mapping,
            sybil_entities=sybil_entities,
        )

        # Step 5: Build intelligence summaries
        entities = await self._build_entity_intelligence(
            entity_ids, sybil_assessments
        )

        high_risk = await self._build_high_risk_wallets(
            wallet_addresses, wallet_entities
        )

        # Step 6: Calculate aggregate metrics
        scores = [r.reputation_score for r in reputations]
        avg_score = sum(scores) / len(scores) if scores else 50

        risk_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for rep in reputations:
            risk = self._score_to_risk(rep.reputation_score)
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1

        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(
            entities, high_risk, sybil_assessments
        )

        # Count sybil wallets
        sybil_wallets = sum(
            1
            for wallet, entity in wallet_entities.items()
            if entity.id in sybil_entities
        )

        report = IntelligenceReport(
            analysis_id=analysis_id,
            token_mint=token_mint,
            analyzed_at=datetime.now(timezone.utc),
            wallets_analyzed=len(wallet_addresses),
            entities_found=len(entities),
            entities=entities,
            sybil_networks_found=len(sybil_entities),
            sybil_wallets_count=sybil_wallets,
            high_risk_wallets=high_risk,
            avg_reputation_score=avg_score,
            risk_distribution=risk_distribution,
            recommendations=recommendations,
        )

        logger.info(
            "cross_token_analysis_complete",
            analysis_id=analysis_id,
            entities_found=len(entities),
            sybil_networks=len(sybil_entities),
            avg_score=avg_score,
        )

        return report

    async def get_wallet_intelligence(
        self,
        wallet_address: str,
    ) -> Optional[WalletIntelligence]:
        """Get intelligence for a single wallet."""
        # Get reputation
        reputation = await self.reputation_repo.get_reputation(wallet_address)

        # Get entity membership
        entity = await self.entity_repo.get_entity_for_wallet(wallet_address)

        # If no reputation, score now
        if not reputation:
            entity_id = entity.id if entity else None
            is_sybil = entity.is_professional_sybil if entity else False

            await self.reputation_scorer.score_and_persist(
                wallet_address, entity_id, is_sybil
            )
            reputation = await self.reputation_repo.get_reputation(wallet_address)

        if not reputation:
            return None

        return WalletIntelligence(
            wallet_address=wallet_address,
            reputation_score=reputation.reputation_score,
            risk_level=self._score_to_risk(reputation.reputation_score),
            confidence_level=reputation.confidence_level,
            patterns=reputation.patterns or [],
            entity_id=entity.id if entity else None,
            entity_type=entity.entity_type if entity else None,
            is_sybil_member=entity.is_professional_sybil if entity else False,
            tokens_analyzed=reputation.tokens_analyzed,
        )

    async def get_entity_intelligence(
        self,
        entity_id: int,
    ) -> Optional[EntityIntelligence]:
        """Get intelligence for an entity."""
        entity = await self.entity_repo.get_entity(entity_id)
        if not entity:
            return None

        wallets = await self.entity_repo.get_entity_wallets(entity_id)

        return EntityIntelligence(
            entity_id=entity.id,
            entity_type=entity.entity_type,
            wallet_count=entity.wallet_count,
            tokens_targeted=entity.tokens_targeted,
            coordination_score=entity.avg_coordination_score or 0.0,
            is_professional_sybil=entity.is_professional_sybil,
            risk_level=entity.risk_level or "medium",
            dominant_funder=entity.dominant_funder_address,
            wallet_addresses=wallets,
        )

    async def record_wallet_interaction(
        self,
        wallet_address: str,
        token_mint: str,
        first_seen_at: datetime,
        archetype: Optional[str] = None,
        archetype_confidence: Optional[float] = None,
    ) -> None:
        """
        Record a wallet's interaction with a token.

        Call this after analyzing a token to build cross-token history.
        """
        await self.history_repo.record_interaction(
            wallet_address=wallet_address,
            token_mint=token_mint,
            first_seen_at=first_seen_at,
            archetype=archetype,
            archetype_confidence=archetype_confidence,
        )

    async def record_interactions_batch(
        self,
        interactions: list[dict],
    ) -> int:
        """
        Record multiple wallet interactions.

        Args:
            interactions: List of dicts with keys:
                - wallet_address, token_mint, first_seen_at
                - Optional: archetype, archetype_confidence

        Returns:
            Number of interactions recorded
        """
        count = 0
        for interaction in interactions:
            await self.history_repo.record_interaction(**interaction)
            count += 1

        logger.info("interactions_recorded", count=count)
        return count

    async def _build_entity_intelligence(
        self,
        entity_ids: set[int],
        sybil_assessments: dict[int, SybilAssessment],
    ) -> list[EntityIntelligence]:
        """Build EntityIntelligence objects for all entities."""
        entities = []

        for entity_id in entity_ids:
            intel = await self.get_entity_intelligence(entity_id)
            if intel:
                # Override risk level from sybil assessment if available
                if entity_id in sybil_assessments:
                    intel.risk_level = sybil_assessments[entity_id].risk_level
                entities.append(intel)

        # Sort by risk
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        entities.sort(key=lambda e: risk_order.get(e.risk_level, 4))

        return entities

    async def _build_high_risk_wallets(
        self,
        wallet_addresses: list[str],
        wallet_entities: dict[str, Entity],
    ) -> list[WalletIntelligence]:
        """Build list of high-risk wallets."""
        high_risk = []

        for wallet in wallet_addresses:
            intel = await self.get_wallet_intelligence(wallet)
            if intel and intel.risk_level in ("high", "critical"):
                high_risk.append(intel)

        # Sort by score (lowest first)
        high_risk.sort(key=lambda w: w.reputation_score)

        # Limit to top 50
        return high_risk[:50]

    def _score_to_risk(self, score: int) -> str:
        """Convert reputation score to risk level."""
        if score <= 20:
            return "critical"
        elif score <= 35:
            return "high"
        elif score <= 50:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(
        self,
        entities: list[EntityIntelligence],
        high_risk_wallets: list[WalletIntelligence],
        sybil_assessments: dict[int, SybilAssessment],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Sybil recommendations
        professional_sybils = [
            e for e in entities if e.is_professional_sybil
        ]
        if professional_sybils:
            total_sybil_wallets = sum(e.wallet_count for e in professional_sybils)
            recommendations.append(
                f"SYBIL ALERT: {len(professional_sybils)} professional sybil networks detected "
                f"comprising {total_sybil_wallets} wallets. Consider exclusion from airdrops."
            )

        # High risk wallet recommendations
        critical_count = sum(1 for w in high_risk_wallets if w.risk_level == "critical")
        if critical_count > 0:
            recommendations.append(
                f"CRITICAL RISK: {critical_count} wallets flagged as critical risk. "
                "Manual review recommended before any distributions."
            )

        # Entity concentration
        large_entities = [e for e in entities if e.wallet_count > 10]
        if large_entities:
            recommendations.append(
                f"CONCENTRATION: {len(large_entities)} entities with 10+ wallets detected. "
                "Consider per-entity caps for fair distribution."
            )

        # General guidance
        if not recommendations:
            recommendations.append(
                "No critical issues detected. Standard distribution criteria recommended."
            )

        return recommendations
