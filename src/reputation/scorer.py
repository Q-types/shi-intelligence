"""
Reputation Scoring System.

Calculates cross-token reputation scores for wallets based on historical behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..data.repositories import WalletHistoryRepository, WalletReputationRepository
from ..data.models import WalletReputation
from .patterns import PatternDetector, PatternType, WalletPattern

logger = structlog.get_logger()


@dataclass
class ScoreComponents:
    """Breakdown of reputation score components."""

    base_score: int  # Starting score (50)
    sniper_penalty: int  # Negative for sniping behavior
    accumulator_bonus: int  # Positive for long-term holding
    rugpull_adjustment: int  # Based on rug exposure/survival
    pnl_adjustment: int  # Based on profitability
    pattern_adjustment: int  # Based on detected patterns
    entity_adjustment: int  # Based on entity associations
    final_score: int  # Clamped 0-100


@dataclass
class ScoringResult:
    """Result of scoring a wallet."""

    wallet_address: str
    reputation_score: int
    confidence_level: str  # low, medium, high
    components: ScoreComponents
    patterns: list[WalletPattern]
    risk_level: str  # low, medium, high, critical


class ReputationScorer:
    """
    Calculates and manages wallet reputation scores.

    Score formula:
    Base = 50
    - Sniper penalty: -5 per snipe (max -25)
    + Accumulator bonus: +3 per accumulation (max +20)
    - Rugpull penalty: -10 per rug if victim, +5 if survivor (max +/-20)
    +/- PnL adjustment: Based on avg PnL (-15 to +15)
    +/- Pattern adjustment: Based on detected patterns (-20 to +20)
    - Entity penalty: If in sybil cluster (-20)

    Final score clamped to 0-100
    """

    # Score weights
    BASE_SCORE = 50
    SNIPER_PENALTY_PER = -5
    SNIPER_PENALTY_MAX = -25
    ACCUMULATOR_BONUS_PER = 3
    ACCUMULATOR_BONUS_MAX = 20
    RUGPULL_VICTIM_PENALTY = -10
    RUGPULL_SURVIVOR_BONUS = 5
    RUGPULL_ADJUSTMENT_MAX = 20
    PNL_ADJUSTMENT_MAX = 15
    PATTERN_ADJUSTMENT_MAX = 20
    ENTITY_SYBIL_PENALTY = -20

    # Confidence thresholds
    LOW_CONFIDENCE_TOKENS = 5
    MEDIUM_CONFIDENCE_TOKENS = 20

    def __init__(
        self,
        session: AsyncSession,
        pattern_detector: Optional[PatternDetector] = None,
    ):
        self.session = session
        self.history_repo = WalletHistoryRepository(session)
        self.reputation_repo = WalletReputationRepository(session)
        self.pattern_detector = pattern_detector or PatternDetector()

    async def score_wallet(
        self,
        wallet_address: str,
        entity_id: Optional[int] = None,
        is_sybil_entity: bool = False,
    ) -> ScoringResult:
        """
        Calculate reputation score for a wallet.

        Args:
            wallet_address: The wallet to score
            entity_id: Optional entity this wallet belongs to
            is_sybil_entity: Whether the entity is a known sybil cluster

        Returns:
            ScoringResult with score breakdown
        """
        # Get behavior summary
        summary = await self.history_repo.get_behavior_summary(wallet_address)

        if not summary:
            # No history - return neutral score with low confidence
            return ScoringResult(
                wallet_address=wallet_address,
                reputation_score=self.BASE_SCORE,
                confidence_level="low",
                components=ScoreComponents(
                    base_score=self.BASE_SCORE,
                    sniper_penalty=0,
                    accumulator_bonus=0,
                    rugpull_adjustment=0,
                    pnl_adjustment=0,
                    pattern_adjustment=0,
                    entity_adjustment=0,
                    final_score=self.BASE_SCORE,
                ),
                patterns=[],
                risk_level="medium",
            )

        # Calculate score components
        components = self._calculate_components(summary, is_sybil_entity)

        # Detect patterns
        pattern_result = self.pattern_detector.detect_patterns(
            wallet_address, summary
        )

        # Apply pattern adjustment
        components.pattern_adjustment = self._pattern_adjustment(pattern_result.patterns)
        components.final_score = self._clamp_score(
            components.base_score
            + components.sniper_penalty
            + components.accumulator_bonus
            + components.rugpull_adjustment
            + components.pnl_adjustment
            + components.pattern_adjustment
            + components.entity_adjustment
        )

        # Determine confidence level
        confidence = self._determine_confidence(summary.tokens_analyzed)

        # Determine risk level
        risk = self._determine_risk(components.final_score, pattern_result.patterns)

        return ScoringResult(
            wallet_address=wallet_address,
            reputation_score=components.final_score,
            confidence_level=confidence,
            components=components,
            patterns=pattern_result.patterns,
            risk_level=risk,
        )

    async def score_and_persist(
        self,
        wallet_address: str,
        entity_id: Optional[int] = None,
        is_sybil_entity: bool = False,
    ) -> WalletReputation:
        """
        Score a wallet and persist the result.

        Returns the updated WalletReputation record.
        """
        result = await self.score_wallet(wallet_address, entity_id, is_sybil_entity)

        # Get summary for persistence
        summary = await self.history_repo.get_behavior_summary(wallet_address)

        # Convert patterns to dict
        patterns_dict = self.pattern_detector.patterns_to_dict(result.patterns)

        # Persist
        reputation = await self.reputation_repo.create_or_update_reputation(
            wallet_address=wallet_address,
            reputation_score=result.reputation_score,
            confidence_level=result.confidence_level,
            tokens_analyzed=summary.tokens_analyzed if summary else 0,
            sniper_count=summary.sniper_count if summary else 0,
            accumulator_count=summary.accumulator_count if summary else 0,
            rugpull_count=summary.rugpull_count if summary else 0,
            avg_holding_days=summary.avg_holding_days if summary else None,
            avg_pnl_pct=summary.avg_pnl_pct if summary else None,
            patterns=patterns_dict,
            entity_id=entity_id,
        )

        logger.info(
            "wallet_scored",
            wallet=wallet_address[:8],
            score=result.reputation_score,
            confidence=result.confidence_level,
            risk=result.risk_level,
            patterns=[p.pattern_type.value for p in result.patterns],
        )

        return reputation

    async def score_wallets_batch(
        self,
        wallet_addresses: list[str],
        entity_mapping: Optional[dict[str, int]] = None,
        sybil_entities: Optional[set[int]] = None,
    ) -> list[ScoringResult]:
        """
        Score multiple wallets.

        Args:
            wallet_addresses: Wallets to score
            entity_mapping: Optional wallet -> entity_id mapping
            sybil_entities: Set of entity IDs that are sybil clusters
        """
        results = []
        entity_mapping = entity_mapping or {}
        sybil_entities = sybil_entities or set()

        for wallet in wallet_addresses:
            entity_id = entity_mapping.get(wallet)
            is_sybil = entity_id in sybil_entities if entity_id else False

            result = await self.score_wallet(wallet, entity_id, is_sybil)
            results.append(result)

        return results

    async def score_and_persist_batch(
        self,
        wallet_addresses: list[str],
        entity_mapping: Optional[dict[str, int]] = None,
        sybil_entities: Optional[set[int]] = None,
    ) -> list[WalletReputation]:
        """Score and persist multiple wallets."""
        reputations = []
        entity_mapping = entity_mapping or {}
        sybil_entities = sybil_entities or set()

        for wallet in wallet_addresses:
            entity_id = entity_mapping.get(wallet)
            is_sybil = entity_id in sybil_entities if entity_id else False

            rep = await self.score_and_persist(wallet, entity_id, is_sybil)
            reputations.append(rep)

        logger.info(
            "batch_scoring_complete",
            wallets_scored=len(reputations),
        )

        return reputations

    def _calculate_components(
        self,
        summary,
        is_sybil_entity: bool,
    ) -> ScoreComponents:
        """Calculate score components from behavior summary."""
        # Sniper penalty
        sniper_penalty = max(
            summary.sniper_count * self.SNIPER_PENALTY_PER,
            self.SNIPER_PENALTY_MAX,
        )

        # Accumulator bonus
        accumulator_bonus = min(
            summary.accumulator_count * self.ACCUMULATOR_BONUS_PER,
            self.ACCUMULATOR_BONUS_MAX,
        )

        # Rugpull adjustment
        # This is simplified - would need more data to know if survivor
        rugpull_adjustment = max(
            summary.rugpull_count * self.RUGPULL_VICTIM_PENALTY,
            -self.RUGPULL_ADJUSTMENT_MAX,
        )

        # PnL adjustment
        pnl_adjustment = 0
        if summary.avg_pnl_pct is not None:
            if summary.avg_pnl_pct > 0:
                # Positive PnL = bonus (max +15)
                pnl_adjustment = min(
                    int(summary.avg_pnl_pct / 10),
                    self.PNL_ADJUSTMENT_MAX,
                )
            else:
                # Negative PnL = penalty (max -15)
                pnl_adjustment = max(
                    int(summary.avg_pnl_pct / 10),
                    -self.PNL_ADJUSTMENT_MAX,
                )

        # Entity adjustment
        entity_adjustment = self.ENTITY_SYBIL_PENALTY if is_sybil_entity else 0

        return ScoreComponents(
            base_score=self.BASE_SCORE,
            sniper_penalty=sniper_penalty,
            accumulator_bonus=accumulator_bonus,
            rugpull_adjustment=rugpull_adjustment,
            pnl_adjustment=pnl_adjustment,
            pattern_adjustment=0,  # Calculated separately
            entity_adjustment=entity_adjustment,
            final_score=0,  # Calculated after patterns
        )

    def _pattern_adjustment(self, patterns: list[WalletPattern]) -> int:
        """Calculate score adjustment from detected patterns."""
        adjustment = 0

        for pattern in patterns:
            weight = pattern.confidence

            if pattern.pattern_type == PatternType.SERIAL_SNIPER:
                adjustment -= int(15 * weight)
            elif pattern.pattern_type == PatternType.DIAMOND_HANDS:
                adjustment += int(15 * weight)
            elif pattern.pattern_type == PatternType.PAPER_HANDS:
                adjustment -= int(5 * weight)
            elif pattern.pattern_type == PatternType.RUGPULL_SURVIVOR:
                adjustment += int(10 * weight)
            elif pattern.pattern_type == PatternType.RUGPULL_VICTIM:
                adjustment -= int(10 * weight)
            elif pattern.pattern_type == PatternType.PROFIT_TAKER:
                adjustment += int(10 * weight)
            elif pattern.pattern_type == PatternType.LOSS_MAKER:
                adjustment -= int(10 * weight)

        # Clamp to max adjustment
        return max(
            min(adjustment, self.PATTERN_ADJUSTMENT_MAX),
            -self.PATTERN_ADJUSTMENT_MAX,
        )

    def _clamp_score(self, score: int) -> int:
        """Clamp score to 0-100 range."""
        return max(0, min(100, score))

    def _determine_confidence(self, tokens_analyzed: int) -> str:
        """Determine confidence level based on data quantity."""
        if tokens_analyzed < self.LOW_CONFIDENCE_TOKENS:
            return "low"
        elif tokens_analyzed < self.MEDIUM_CONFIDENCE_TOKENS:
            return "medium"
        else:
            return "high"

    def _determine_risk(
        self,
        score: int,
        patterns: list[WalletPattern],
    ) -> str:
        """Determine risk level from score and patterns."""
        # Check for high-risk patterns
        high_risk_patterns = {
            PatternType.SERIAL_SNIPER,
            PatternType.RUGPULL_VICTIM,
        }

        has_high_risk_pattern = any(
            p.pattern_type in high_risk_patterns and p.confidence > 0.7
            for p in patterns
        )

        if score <= 20 or has_high_risk_pattern:
            return "critical"
        elif score <= 35:
            return "high"
        elif score <= 50:
            return "medium"
        else:
            return "low"

    async def recalculate_all(
        self,
        batch_size: int = 100,
    ) -> int:
        """
        Recalculate scores for all wallets with history.

        Returns number of wallets scored.
        """
        # This would need a method to iterate all wallets
        # Placeholder for now
        logger.info("recalculate_all_started")
        return 0
