"""
Professional Sybil Network Detector.

Identifies sophisticated sybil operations that target multiple tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..data.repositories import EntityRepository, WalletHistoryRepository
from ..data.models import Entity

logger = structlog.get_logger()


@dataclass
class SybilIndicator:
    """Individual indicator of sybil behavior."""

    indicator_type: str
    severity: str  # low, medium, high, critical
    confidence: float
    evidence: dict


@dataclass
class SybilAssessment:
    """Full sybil assessment for an entity."""

    entity_id: int
    is_professional_sybil: bool
    risk_level: str  # low, medium, high, critical
    confidence: float
    indicators: list[SybilIndicator]
    wallet_count: int
    tokens_targeted: int
    coordination_score: float
    recommendation: str


class SybilNetworkDetector:
    """
    Detects professional sybil networks.

    Professional sybils are characterized by:
    - Multiple wallets (5+) in an entity
    - Targeting multiple tokens (3+)
    - High coordination scores (>0.6)
    - Similar behavioral patterns across wallets
    - Shared funding source

    Classification thresholds:
    - Suspected: 5+ wallets, 3+ tokens, coordination > 0.5
    - Confirmed: 10+ wallets, 5+ tokens, coordination > 0.7
    - Professional: 20+ wallets, 10+ tokens, coordination > 0.8
    """

    # Thresholds for professional sybil classification
    PROFESSIONAL_THRESHOLDS = {
        "suspected": {"wallets": 5, "tokens": 3, "coordination": 0.5},
        "confirmed": {"wallets": 10, "tokens": 5, "coordination": 0.7},
        "professional": {"wallets": 20, "tokens": 10, "coordination": 0.8},
    }

    def __init__(
        self,
        session: AsyncSession,
    ):
        self.session = session
        self.entity_repo = EntityRepository(session)
        self.history_repo = WalletHistoryRepository(session)

    async def assess_entity(
        self,
        entity_id: int,
    ) -> Optional[SybilAssessment]:
        """
        Assess whether an entity is a sybil network.

        Args:
            entity_id: Entity to assess

        Returns:
            SybilAssessment or None if entity not found
        """
        entity = await self.entity_repo.get_entity(entity_id)
        if not entity:
            return None

        # Get wallet addresses
        wallets = await self.entity_repo.get_entity_wallets(entity_id)

        # Calculate indicators
        indicators = []

        # 1. Shared funder indicator
        if entity.dominant_funder_address:
            indicators.append(
                SybilIndicator(
                    indicator_type="shared_funder",
                    severity="high" if len(wallets) > 10 else "medium",
                    confidence=0.8,
                    evidence={
                        "funder_address": entity.dominant_funder_address[:16] + "...",
                        "wallets_funded": len(wallets),
                    },
                )
            )

        # 2. Coordination indicator
        coordination_score = entity.avg_coordination_score or 0.0
        if coordination_score > 0.5:
            severity = "critical" if coordination_score > 0.8 else (
                "high" if coordination_score > 0.7 else "medium"
            )
            indicators.append(
                SybilIndicator(
                    indicator_type="temporal_coordination",
                    severity=severity,
                    confidence=coordination_score,
                    evidence={
                        "coordination_score": coordination_score,
                    },
                )
            )

        # 3. Token targeting indicator
        tokens_targeted = await self._count_tokens_targeted(wallets)
        if tokens_targeted >= 3:
            severity = "critical" if tokens_targeted > 10 else (
                "high" if tokens_targeted > 5 else "medium"
            )
            indicators.append(
                SybilIndicator(
                    indicator_type="multi_token_targeting",
                    severity=severity,
                    confidence=min(tokens_targeted / 10, 1.0),
                    evidence={
                        "tokens_targeted": tokens_targeted,
                    },
                )
            )

        # 4. Cluster size indicator
        if len(wallets) >= 5:
            severity = "critical" if len(wallets) > 20 else (
                "high" if len(wallets) > 10 else "medium"
            )
            indicators.append(
                SybilIndicator(
                    indicator_type="cluster_size",
                    severity=severity,
                    confidence=min(len(wallets) / 20, 1.0),
                    evidence={
                        "wallet_count": len(wallets),
                    },
                )
            )

        # 5. Behavioral similarity indicator
        behavior_similarity = await self._check_behavioral_similarity(wallets)
        if behavior_similarity > 0.6:
            indicators.append(
                SybilIndicator(
                    indicator_type="behavioral_similarity",
                    severity="high" if behavior_similarity > 0.8 else "medium",
                    confidence=behavior_similarity,
                    evidence={
                        "similarity_score": behavior_similarity,
                    },
                )
            )

        # Determine classification
        is_professional, risk_level, confidence = self._classify(
            wallet_count=len(wallets),
            tokens_targeted=tokens_targeted,
            coordination_score=coordination_score,
            indicators=indicators,
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            is_professional, risk_level, indicators
        )

        assessment = SybilAssessment(
            entity_id=entity_id,
            is_professional_sybil=is_professional,
            risk_level=risk_level,
            confidence=confidence,
            indicators=indicators,
            wallet_count=len(wallets),
            tokens_targeted=tokens_targeted,
            coordination_score=coordination_score,
            recommendation=recommendation,
        )

        logger.info(
            "sybil_assessment_complete",
            entity_id=entity_id,
            is_sybil=is_professional,
            risk_level=risk_level,
            wallet_count=len(wallets),
            tokens_targeted=tokens_targeted,
        )

        return assessment

    async def assess_and_update(
        self,
        entity_id: int,
    ) -> Optional[SybilAssessment]:
        """Assess entity and update its risk status in the database."""
        assessment = await self.assess_entity(entity_id)
        if not assessment:
            return None

        # Update entity risk status
        await self.entity_repo.update_entity_risk(
            entity_id=entity_id,
            is_professional_sybil=assessment.is_professional_sybil,
            risk_level=assessment.risk_level,
            avg_coordination_score=assessment.coordination_score,
        )

        # Update entity stats
        await self.entity_repo.update_entity_stats(
            entity_id=entity_id,
            tokens_targeted=assessment.tokens_targeted,
        )

        return assessment

    async def scan_all_entities(
        self,
        min_wallet_count: int = 3,
        limit: int = 1000,
    ) -> list[SybilAssessment]:
        """
        Scan all entities for sybil behavior.

        Returns list of assessments for entities that meet minimum criteria.
        """
        # Get candidate entities
        candidates = await self.entity_repo.find_entities_by_type(
            "sybil_cluster", limit=limit
        )

        # Also check other entity types
        for entity_type in ["whale_group", "trading_group", "unknown"]:
            others = await self.entity_repo.find_entities_by_type(
                entity_type, limit=limit // 4
            )
            candidates.extend(others)

        # Filter by wallet count
        candidates = [e for e in candidates if e.wallet_count >= min_wallet_count]

        # Assess each
        assessments = []
        for entity in candidates:
            assessment = await self.assess_and_update(entity.id)
            if assessment:
                assessments.append(assessment)

        # Sort by risk (highest first)
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        assessments.sort(key=lambda a: risk_order.get(a.risk_level, 4))

        logger.info(
            "sybil_scan_complete",
            entities_scanned=len(candidates),
            sybils_found=sum(1 for a in assessments if a.is_professional_sybil),
        )

        return assessments

    async def _count_tokens_targeted(self, wallets: list[str]) -> int:
        """Count unique tokens targeted by a set of wallets."""
        all_tokens = set()
        for wallet in wallets:
            tokens = await self.history_repo.get_wallet_tokens(wallet)
            all_tokens.update(tokens)
        return len(all_tokens)

    async def _check_behavioral_similarity(self, wallets: list[str]) -> float:
        """Check behavioral similarity across wallets."""
        if len(wallets) < 2:
            return 0.0

        # Get behavior summaries
        summaries = []
        for wallet in wallets[:20]:  # Limit to first 20 for performance
            summary = await self.history_repo.get_behavior_summary(wallet)
            if summary:
                summaries.append(summary)

        if len(summaries) < 2:
            return 0.0

        # Compare sniper/accumulator ratios
        sniper_ratios = []
        acc_ratios = []

        for s in summaries:
            if s.tokens_analyzed > 0:
                sniper_ratios.append(s.sniper_count / s.tokens_analyzed)
                acc_ratios.append(s.accumulator_count / s.tokens_analyzed)

        if not sniper_ratios:
            return 0.0

        # Calculate variance (lower = more similar)
        def variance(values):
            if len(values) < 2:
                return 0.0
            mean = sum(values) / len(values)
            return sum((v - mean) ** 2 for v in values) / len(values)

        sniper_var = variance(sniper_ratios)
        acc_var = variance(acc_ratios)

        # Convert variance to similarity (0-1, lower variance = higher similarity)
        # Max expected variance is ~0.25 (for 0-1 ratios)
        avg_var = (sniper_var + acc_var) / 2
        similarity = max(1.0 - (avg_var * 4), 0.0)

        return similarity

    def _classify(
        self,
        wallet_count: int,
        tokens_targeted: int,
        coordination_score: float,
        indicators: list[SybilIndicator],
    ) -> tuple[bool, str, float]:
        """
        Classify entity based on indicators.

        Returns (is_professional_sybil, risk_level, confidence)
        """
        # Check against thresholds
        thresholds = self.PROFESSIONAL_THRESHOLDS

        if (
            wallet_count >= thresholds["professional"]["wallets"]
            and tokens_targeted >= thresholds["professional"]["tokens"]
            and coordination_score >= thresholds["professional"]["coordination"]
        ):
            return True, "critical", 0.95

        if (
            wallet_count >= thresholds["confirmed"]["wallets"]
            and tokens_targeted >= thresholds["confirmed"]["tokens"]
            and coordination_score >= thresholds["confirmed"]["coordination"]
        ):
            return True, "high", 0.85

        if (
            wallet_count >= thresholds["suspected"]["wallets"]
            and tokens_targeted >= thresholds["suspected"]["tokens"]
            and coordination_score >= thresholds["suspected"]["coordination"]
        ):
            return True, "medium", 0.65

        # Check indicator severity
        critical_count = sum(1 for i in indicators if i.severity == "critical")
        high_count = sum(1 for i in indicators if i.severity == "high")

        if critical_count >= 2:
            return True, "high", 0.8
        if critical_count >= 1 and high_count >= 2:
            return True, "medium", 0.7
        if high_count >= 3:
            return False, "medium", 0.5

        return False, "low", 0.3

    def _generate_recommendation(
        self,
        is_professional: bool,
        risk_level: str,
        indicators: list[SybilIndicator],
    ) -> str:
        """Generate actionable recommendation based on assessment."""
        if not is_professional and risk_level == "low":
            return "No action required. Continue monitoring."

        recommendations = []

        if is_professional:
            recommendations.append("FLAG: Professional sybil network detected.")

        if risk_level == "critical":
            recommendations.append(
                "CRITICAL: Consider immediate blacklisting for airdrop eligibility."
            )
        elif risk_level == "high":
            recommendations.append(
                "HIGH RISK: Recommend manual review before any distribution."
            )
        elif risk_level == "medium":
            recommendations.append(
                "MEDIUM RISK: Apply enhanced scrutiny for rewards allocation."
            )

        # Specific indicator recommendations
        for indicator in indicators:
            if indicator.indicator_type == "multi_token_targeting" and indicator.severity in ("high", "critical"):
                recommendations.append(
                    f"Targets {indicator.evidence.get('tokens_targeted', 'many')} tokens - likely farming operation."
                )

        return " ".join(recommendations)
