"""
Cross-Token Behaviour Memory Service.

Tracks wallet behavior across multiple token launches for pattern recognition.

HARD RULES:
1. behaviour_history_score is probabilistic, not definitive
2. Do not expose as "reputation" until validated with labeled data
3. Confidence must reflect data quality and quantity
4. All scores are updatable as new evidence arrives
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Any
from enum import Enum
import math

import structlog

from .models import (
    WalletBehaviorHistory,
    TokenParticipationHistory,
    WalletTokenPositionHistory,
    CoParticipationEdge,
    BehavioralSimilarityCache,
)

logger = structlog.get_logger()


class WalletArchetype(str, Enum):
    """Wallet behavior archetypes within a token."""

    SNIPER = "sniper"
    ACCUMULATOR = "accumulator"
    QUICK_EXIT = "quick_exit"
    LP_PROVIDER = "lp_provider"
    HOLDER = "holder"
    TRADER = "trader"
    WHALE = "whale"
    UNKNOWN = "unknown"


@dataclass
class WalletMetrics:
    """Computed metrics for a wallet."""

    total_launches: int = 0
    avg_entry_timing: float = 0.5  # 0=immediate, 1=late
    avg_exit_timing: float = 0.5
    entry_timing_variance: float = 0.0
    avg_hold_duration_hours: float = 0.0
    early_exit_rate: float = 0.0
    coordination_participation_rate: float = 0.0
    common_co_traders: list = field(default_factory=list)
    avg_position_size: float = 0.0
    realised_pnl_estimate: Optional[float] = None
    behaviour_history_score: float = 50.0
    behaviour_confidence: float = 0.0


@dataclass
class CoParticipationMetrics:
    """Metrics for co-participation between two wallets."""

    shared_token_count: int = 0
    avg_entry_correlation: float = 0.0
    avg_exit_correlation: float = 0.0
    archetype_similarity: float = 0.0
    shared_coordination_clusters: int = 0
    coordination_likelihood: float = 0.0


class CrossTokenMemoryService:
    """
    Service for managing cross-token wallet behavior memory.

    Provides:
    - Wallet behavior aggregation across tokens
    - Co-participation detection
    - Behavioral similarity computation
    - Behaviour history scoring (not reputation)
    """

    VERSION = "1.0.0"

    # Scoring weights
    WEIGHT_EARLY_EXIT = 0.15
    WEIGHT_COORDINATION = 0.25
    WEIGHT_TIMING_CONSISTENCY = 0.10
    WEIGHT_HOLD_DURATION = 0.15
    WEIGHT_PNL = 0.20
    WEIGHT_POSITION_SIZE = 0.15

    # Confidence thresholds
    MIN_LAUNCHES_FOR_SCORE = 3
    HIGH_CONFIDENCE_LAUNCHES = 10
    CACHE_TTL_HOURS = 24

    def __init__(self, session_factory=None):
        """Initialize the cross-token memory service."""
        self.session_factory = session_factory

    async def record_participation(
        self,
        wallet_address: str,
        token_mint: str,
        entry_time: datetime,
        entry_timing_percentile: float,
        position_tokens: int,
        position_pct: float,
        archetype: Optional[str] = None,
        in_coordination_cluster: bool = False,
        coordination_cluster_id: Optional[str] = None,
    ) -> TokenParticipationHistory:
        """
        Record wallet participation in a token.

        This is called when a wallet enters a new token.
        """
        logger.info(
            "recording_participation",
            wallet=wallet_address[:8],
            token=token_mint[:8],
            timing_pct=f"{entry_timing_percentile:.1f}",
        )

        participation = TokenParticipationHistory(
            wallet_address=wallet_address,
            token_mint=token_mint,
            first_seen_at=entry_time,
            last_seen_at=entry_time,
            entry_timing_percentile=entry_timing_percentile,
            max_position_tokens=position_tokens,
            max_position_pct=position_pct,
            final_position_tokens=position_tokens,
            archetype_assigned=archetype,
            in_coordination_cluster=in_coordination_cluster,
            coordination_cluster_id=coordination_cluster_id,
        )

        # Update aggregate wallet history
        await self._update_wallet_history(wallet_address, participation)

        return participation

    async def record_exit(
        self,
        wallet_address: str,
        token_mint: str,
        exit_time: datetime,
        exit_timing_percentile: float,
        realised_pnl_pct: Optional[float] = None,
        realised_pnl_usd: Optional[float] = None,
    ):
        """
        Record wallet exit from a token.

        Updates participation record with exit data.
        """
        logger.info(
            "recording_exit",
            wallet=wallet_address[:8],
            token=token_mint[:8],
            pnl_pct=f"{realised_pnl_pct:.1f}%" if realised_pnl_pct else "unknown",
        )

        # TODO: Update TokenParticipationHistory with exit data
        # TODO: Recalculate wallet behavior history

    async def update_position(
        self,
        wallet_address: str,
        token_mint: str,
        timestamp: datetime,
        position_tokens: int,
        position_pct: float,
        delta_tokens: int,
        delta_source: str,
        event_sequence: int,
        price_at_snapshot: Optional[float] = None,
    ) -> WalletTokenPositionHistory:
        """
        Record a position change for a wallet.

        Creates historical position snapshot for trajectory reconstruction.
        """
        position_record = WalletTokenPositionHistory(
            wallet_address=wallet_address,
            token_mint=token_mint,
            timestamp=timestamp,
            position_tokens=position_tokens,
            position_pct=position_pct,
            delta_tokens=delta_tokens,
            delta_source=delta_source,
            event_sequence=event_sequence,
            price_at_snapshot=price_at_snapshot,
        )

        return position_record

    async def compute_wallet_metrics(
        self,
        wallet_address: str,
        participations: list[TokenParticipationHistory],
    ) -> WalletMetrics:
        """
        Compute aggregate metrics for a wallet from participation history.

        This is the core computation for behaviour_history_score.
        """
        if not participations:
            return WalletMetrics()

        n = len(participations)

        # Entry timing
        entry_timings = [p.entry_timing_percentile for p in participations]
        avg_entry = sum(entry_timings) / n
        entry_variance = sum((t - avg_entry) ** 2 for t in entry_timings) / n

        # Exit timing (only for exited positions)
        exited = [p for p in participations if p.is_exited and p.exit_timing_percentile]
        avg_exit = sum(p.exit_timing_percentile for p in exited) / len(exited) if exited else 0.5

        # Hold duration
        hold_durations = [p.hold_duration_hours for p in participations if p.hold_duration_hours > 0]
        avg_hold = sum(hold_durations) / len(hold_durations) if hold_durations else 0.0

        # Early exit rate (exited within 1 hour)
        early_exits = [p for p in exited if p.hold_duration_hours < 1.0]
        early_exit_rate = len(early_exits) / len(exited) if exited else 0.0

        # Coordination participation
        coordinated = [p for p in participations if p.in_coordination_cluster]
        coordination_rate = len(coordinated) / n

        # Position size
        position_sizes = [p.max_position_pct for p in participations if p.max_position_pct > 0]
        avg_position_size = sum(position_sizes) / len(position_sizes) if position_sizes else 0.0

        # PnL
        pnl_values = [p.realised_pnl_pct for p in participations if p.realised_pnl_pct is not None]
        avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else None

        # Compute behaviour history score
        score = self._compute_behaviour_score(
            avg_entry_timing=avg_entry,
            entry_timing_variance=entry_variance,
            avg_hold_duration=avg_hold,
            early_exit_rate=early_exit_rate,
            coordination_rate=coordination_rate,
            avg_position_size=avg_position_size,
            avg_pnl=avg_pnl,
        )

        # Compute confidence
        confidence = self._compute_confidence(n, len(pnl_values))

        return WalletMetrics(
            total_launches=n,
            avg_entry_timing=avg_entry,
            avg_exit_timing=avg_exit,
            entry_timing_variance=entry_variance,
            avg_hold_duration_hours=avg_hold,
            early_exit_rate=early_exit_rate,
            coordination_participation_rate=coordination_rate,
            avg_position_size=avg_position_size,
            realised_pnl_estimate=avg_pnl,
            behaviour_history_score=score,
            behaviour_confidence=confidence,
        )

    def _compute_behaviour_score(
        self,
        avg_entry_timing: float,
        entry_timing_variance: float,
        avg_hold_duration: float,
        early_exit_rate: float,
        coordination_rate: float,
        avg_position_size: float,
        avg_pnl: Optional[float],
    ) -> float:
        """
        Compute behaviour history score.

        NOT reputation - this is a probabilistic behavioral metric.
        Range: 0-100
        50 = neutral/unknown
        Higher = more "organic" behavior patterns
        Lower = more "suspicious" patterns

        This is NOT a value judgment - it's a statistical pattern score.
        """
        score = 50.0  # Start at neutral

        # Early exit penalty (suspicious pattern)
        # High early exit rate suggests pump-and-dump participation
        early_exit_penalty = early_exit_rate * 20.0 * self.WEIGHT_EARLY_EXIT
        score -= early_exit_penalty

        # Coordination penalty (suspicious pattern)
        # High coordination rate suggests coordinated activity
        coordination_penalty = coordination_rate * 30.0 * self.WEIGHT_COORDINATION
        score -= coordination_penalty

        # Entry timing consistency (neutral to slight penalty)
        # Very consistent entry timing might suggest automation
        if entry_timing_variance < 0.05:  # Very consistent
            score -= 5.0 * self.WEIGHT_TIMING_CONSISTENCY

        # Hold duration bonus (positive pattern)
        # Longer holds suggest conviction
        if avg_hold_duration > 24.0:  # Held > 1 day
            hold_bonus = min(15.0, avg_hold_duration / 24.0 * 5.0)
            score += hold_bonus * self.WEIGHT_HOLD_DURATION

        # PnL adjustment (skill indicator)
        if avg_pnl is not None:
            if avg_pnl > 0:
                # Positive PnL - slight bonus
                pnl_bonus = min(10.0, avg_pnl / 100.0 * 20.0)
                score += pnl_bonus * self.WEIGHT_PNL
            else:
                # Negative PnL - neutral (losing money isn't suspicious)
                pass

        # Position size adjustment
        # Very large positions might be whale manipulation
        if avg_position_size > 10.0:  # > 10% average position
            size_penalty = min(10.0, (avg_position_size - 10.0) / 10.0 * 10.0)
            score -= size_penalty * self.WEIGHT_POSITION_SIZE

        # Clamp to valid range
        return max(0.0, min(100.0, score))

    def _compute_confidence(self, total_launches: int, pnl_data_points: int) -> float:
        """
        Compute confidence in behaviour score.

        Range: 0-1
        Based on data quantity and quality.
        """
        if total_launches < self.MIN_LAUNCHES_FOR_SCORE:
            return 0.0

        # Base confidence from launch count
        launch_factor = min(1.0, total_launches / self.HIGH_CONFIDENCE_LAUNCHES)

        # PnL data quality factor
        pnl_factor = min(1.0, pnl_data_points / total_launches) if total_launches > 0 else 0.0

        # Combined confidence
        confidence = launch_factor * 0.7 + pnl_factor * 0.3

        return round(confidence, 3)

    async def compute_co_participation(
        self,
        wallet_a: str,
        wallet_b: str,
        participations_a: list[TokenParticipationHistory],
        participations_b: list[TokenParticipationHistory],
    ) -> CoParticipationMetrics:
        """
        Compute co-participation metrics between two wallets.

        Used for detecting coordination across tokens.
        """
        # Find shared tokens
        tokens_a = {p.token_mint: p for p in participations_a}
        tokens_b = {p.token_mint: p for p in participations_b}
        shared_tokens = set(tokens_a.keys()) & set(tokens_b.keys())

        if not shared_tokens:
            return CoParticipationMetrics()

        # Compute correlations
        entry_correlations = []
        exit_correlations = []
        archetype_matches = 0
        coordination_clusters = 0

        for token in shared_tokens:
            pa = tokens_a[token]
            pb = tokens_b[token]

            # Entry timing correlation
            entry_diff = abs(pa.entry_timing_percentile - pb.entry_timing_percentile)
            entry_corr = 1.0 - entry_diff / 100.0  # 1 = same time, 0 = opposite
            entry_correlations.append(entry_corr)

            # Exit timing correlation
            if pa.exit_timing_percentile and pb.exit_timing_percentile:
                exit_diff = abs(pa.exit_timing_percentile - pb.exit_timing_percentile)
                exit_corr = 1.0 - exit_diff / 100.0
                exit_correlations.append(exit_corr)

            # Archetype similarity
            if pa.archetype_assigned == pb.archetype_assigned:
                archetype_matches += 1

            # Shared coordination clusters
            if (pa.in_coordination_cluster and pb.in_coordination_cluster and
                pa.coordination_cluster_id == pb.coordination_cluster_id):
                coordination_clusters += 1

        avg_entry_corr = sum(entry_correlations) / len(entry_correlations)
        avg_exit_corr = sum(exit_correlations) / len(exit_correlations) if exit_correlations else 0.0
        archetype_sim = archetype_matches / len(shared_tokens)

        # Compute coordination likelihood
        coordination_likelihood = self._compute_coordination_likelihood(
            shared_token_count=len(shared_tokens),
            avg_entry_correlation=avg_entry_corr,
            avg_exit_correlation=avg_exit_corr,
            archetype_similarity=archetype_sim,
            shared_coordination_clusters=coordination_clusters,
        )

        return CoParticipationMetrics(
            shared_token_count=len(shared_tokens),
            avg_entry_correlation=avg_entry_corr,
            avg_exit_correlation=avg_exit_corr,
            archetype_similarity=archetype_sim,
            shared_coordination_clusters=coordination_clusters,
            coordination_likelihood=coordination_likelihood,
        )

    def _compute_coordination_likelihood(
        self,
        shared_token_count: int,
        avg_entry_correlation: float,
        avg_exit_correlation: float,
        archetype_similarity: float,
        shared_coordination_clusters: int,
    ) -> float:
        """
        Compute likelihood that two wallets are coordinated.

        This is probabilistic and should not be treated as definitive.
        Range: 0-1
        """
        if shared_token_count < 2:
            return 0.0

        # Base score from shared tokens
        token_factor = min(1.0, shared_token_count / 5.0) * 0.2

        # Entry timing correlation
        entry_factor = max(0.0, (avg_entry_correlation - 0.5) * 2.0) * 0.25

        # Exit timing correlation
        exit_factor = max(0.0, (avg_exit_correlation - 0.5) * 2.0) * 0.20

        # Archetype similarity
        archetype_factor = archetype_similarity * 0.15

        # Shared coordination clusters (strong signal)
        cluster_factor = min(1.0, shared_coordination_clusters / 3.0) * 0.20

        likelihood = token_factor + entry_factor + exit_factor + archetype_factor + cluster_factor

        return round(min(1.0, likelihood), 3)

    async def compute_behavioral_similarity(
        self,
        wallet_a: str,
        wallet_b: str,
        history_a: WalletBehaviorHistory,
        history_b: WalletBehaviorHistory,
    ) -> BehavioralSimilarityCache:
        """
        Compute behavioral similarity between two wallets.

        Used for entity resolution and pattern matching.
        """
        # Timing similarity
        entry_diff = abs(history_a.avg_entry_timing_percentile - history_b.avg_entry_timing_percentile)
        exit_diff = abs(history_a.avg_exit_timing_percentile - history_b.avg_exit_timing_percentile)
        timing_sim = 1.0 - (entry_diff + exit_diff) / 200.0

        # Position sizing similarity
        size_a = history_a.avg_position_size_pct if hasattr(history_a, 'avg_position_size_pct') else 0.0
        size_b = history_b.avg_position_size_pct if hasattr(history_b, 'avg_position_size_pct') else 0.0
        max_size = max(size_a, size_b, 1.0)
        position_sim = 1.0 - abs(size_a - size_b) / max_size

        # Archetype distribution similarity (using counts)
        archetype_sim = self._compute_archetype_similarity(history_a, history_b)

        # PnL correlation (simplified)
        pnl_a = history_a.avg_pnl_pct if history_a.avg_pnl_pct else 0.0
        pnl_b = history_b.avg_pnl_pct if history_b.avg_pnl_pct else 0.0
        max_pnl = max(abs(pnl_a), abs(pnl_b), 1.0)
        pnl_corr = 1.0 - abs(pnl_a - pnl_b) / max_pnl

        # Trade pattern similarity
        trade_sim = self._compute_trade_pattern_similarity(history_a, history_b)

        # Overall similarity
        overall = (
            timing_sim * 0.25 +
            position_sim * 0.20 +
            archetype_sim * 0.25 +
            pnl_corr * 0.15 +
            trade_sim * 0.15
        )

        now = datetime.now(timezone.utc)
        return BehavioralSimilarityCache(
            wallet_a=min(wallet_a, wallet_b),  # Ensure consistent ordering
            wallet_b=max(wallet_a, wallet_b),
            timing_similarity=round(timing_sim, 3),
            position_sizing_similarity=round(position_sim, 3),
            archetype_similarity=round(archetype_sim, 3),
            pnl_correlation=round(pnl_corr, 3),
            trade_pattern_similarity=round(trade_sim, 3),
            overall_similarity=round(overall, 3),
            tokens_compared=min(history_a.total_tokens_participated, history_b.total_tokens_participated),
            computed_at=now,
            expires_at=now + timedelta(hours=self.CACHE_TTL_HOURS),
        )

    def _compute_archetype_similarity(
        self,
        history_a: WalletBehaviorHistory,
        history_b: WalletBehaviorHistory,
    ) -> float:
        """Compute similarity in archetype distribution."""
        total_a = max(1, history_a.sniper_count + history_a.accumulator_count +
                      history_a.quick_exit_count + history_a.lp_interaction_count)
        total_b = max(1, history_b.sniper_count + history_b.accumulator_count +
                      history_b.quick_exit_count + history_b.lp_interaction_count)

        # Normalize to proportions
        props_a = [
            history_a.sniper_count / total_a,
            history_a.accumulator_count / total_a,
            history_a.quick_exit_count / total_a,
            history_a.lp_interaction_count / total_a,
        ]
        props_b = [
            history_b.sniper_count / total_b,
            history_b.accumulator_count / total_b,
            history_b.quick_exit_count / total_b,
            history_b.lp_interaction_count / total_b,
        ]

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(props_a, props_b))
        norm_a = math.sqrt(sum(a ** 2 for a in props_a))
        norm_b = math.sqrt(sum(b ** 2 for b in props_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _compute_trade_pattern_similarity(
        self,
        history_a: WalletBehaviorHistory,
        history_b: WalletBehaviorHistory,
    ) -> float:
        """Compute similarity in trading patterns."""
        # Compare hold duration
        hold_a = history_a.avg_holding_days if history_a.avg_holding_days else 0.0
        hold_b = history_b.avg_holding_days if history_b.avg_holding_days else 0.0
        max_hold = max(hold_a, hold_b, 1.0)
        hold_sim = 1.0 - abs(hold_a - hold_b) / max_hold

        # Compare early exit rate
        exit_a = history_a.early_exit_rate if hasattr(history_a, 'early_exit_rate') else 0.0
        exit_b = history_b.early_exit_rate if hasattr(history_b, 'early_exit_rate') else 0.0
        exit_sim = 1.0 - abs(exit_a - exit_b)

        return (hold_sim + exit_sim) / 2.0

    async def _update_wallet_history(
        self,
        wallet_address: str,
        new_participation: TokenParticipationHistory,
    ):
        """
        Update aggregate wallet history with new participation.

        This is called after each new participation record.
        """
        # TODO: Implement database update
        # 1. Load existing WalletBehaviorHistory
        # 2. Increment counters
        # 3. Recompute averages
        # 4. Update behaviour_history_score
        # 5. Save
        logger.debug(
            "updating_wallet_history",
            wallet=wallet_address[:8],
            token=new_participation.token_mint[:8],
        )

    async def get_suspicious_wallets(
        self,
        min_launches: int = 3,
        max_behaviour_score: float = 40.0,
        min_confidence: float = 0.5,
    ) -> list[WalletBehaviorHistory]:
        """
        Find wallets with suspicious behavior patterns.

        Returns wallets with low behaviour_history_score and sufficient confidence.
        """
        # TODO: Implement database query
        logger.info(
            "querying_suspicious_wallets",
            min_launches=min_launches,
            max_score=max_behaviour_score,
            min_confidence=min_confidence,
        )
        return []

    async def get_co_traders(
        self,
        wallet_address: str,
        min_shared_tokens: int = 2,
        min_coordination_likelihood: float = 0.5,
    ) -> list[CoParticipationEdge]:
        """
        Find wallets that frequently co-trade with a given wallet.

        Returns co-participation edges sorted by coordination likelihood.
        """
        # TODO: Implement database query
        logger.info(
            "querying_co_traders",
            wallet=wallet_address[:8],
            min_shared=min_shared_tokens,
            min_likelihood=min_coordination_likelihood,
        )
        return []

    async def invalidate_similarity_cache(
        self,
        wallet_address: str,
    ):
        """
        Invalidate similarity cache entries for a wallet.

        Called when wallet behavior is updated.
        """
        # TODO: Implement cache invalidation
        logger.debug("invalidating_similarity_cache", wallet=wallet_address[:8])
