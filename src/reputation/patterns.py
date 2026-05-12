"""
Wallet Behavior Pattern Detection.

Identifies recurring behavioral patterns from cross-token wallet history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Sequence

import structlog

from ..data.repositories.wallet_history import WalletBehaviorSummary

logger = structlog.get_logger()


class PatternType(str, Enum):
    """Types of detectable wallet patterns."""

    SERIAL_SNIPER = "SERIAL_SNIPER"  # Frequently snipes new tokens
    DIAMOND_HANDS = "DIAMOND_HANDS"  # Long-term holder across tokens
    PAPER_HANDS = "PAPER_HANDS"  # Quick exits, short holding periods
    RUGPULL_SURVIVOR = "RUGPULL_SURVIVOR"  # Exited before multiple rugs
    RUGPULL_VICTIM = "RUGPULL_VICTIM"  # Caught in multiple rugs
    WHALE_ACCUMULATOR = "WHALE_ACCUMULATOR"  # Large positions, slow accumulation
    PROFIT_TAKER = "PROFIT_TAKER"  # Consistent profitable exits
    LOSS_MAKER = "LOSS_MAKER"  # Consistently loses money
    EARLY_ADOPTER = "EARLY_ADOPTER"  # First to enter new tokens
    COPY_TRADER = "COPY_TRADER"  # Follows other wallets


@dataclass
class WalletPattern:
    """A detected behavioral pattern for a wallet."""

    pattern_type: PatternType
    confidence: float  # 0-1
    token_count: int  # Tokens contributing to this pattern
    detected_at: datetime
    metadata: dict


@dataclass
class PatternDetectionResult:
    """Result of pattern detection for a wallet."""

    wallet_address: str
    patterns: list[WalletPattern]
    primary_pattern: Optional[PatternType]
    pattern_consistency: float  # How consistent is behavior across tokens


class PatternDetector:
    """
    Detects behavioral patterns from wallet history.

    Analyzes cross-token behavior to identify patterns like:
    - Serial snipers who consistently snipe new tokens
    - Diamond hands who hold through volatility
    - Rugpull survivors who exit before rugs
    """

    def __init__(
        self,
        min_tokens_for_pattern: int = 3,
        min_confidence: float = 0.5,
    ):
        self.min_tokens_for_pattern = min_tokens_for_pattern
        self.min_confidence = min_confidence

    def detect_patterns(
        self,
        wallet_address: str,
        behavior_summary: WalletBehaviorSummary,
        token_interactions: Optional[Sequence] = None,
    ) -> PatternDetectionResult:
        """
        Detect all applicable patterns for a wallet.

        Args:
            wallet_address: The wallet to analyze
            behavior_summary: Aggregated behavior metrics
            token_interactions: Optional detailed interaction data

        Returns:
            PatternDetectionResult with all detected patterns
        """
        patterns = []

        if behavior_summary.tokens_analyzed < self.min_tokens_for_pattern:
            return PatternDetectionResult(
                wallet_address=wallet_address,
                patterns=[],
                primary_pattern=None,
                pattern_consistency=0.0,
            )

        # Check for each pattern type
        pattern_checks = [
            self._check_serial_sniper,
            self._check_diamond_hands,
            self._check_paper_hands,
            self._check_rugpull_survivor,
            self._check_rugpull_victim,
            self._check_profit_taker,
            self._check_loss_maker,
        ]

        for check_func in pattern_checks:
            pattern = check_func(behavior_summary)
            if pattern and pattern.confidence >= self.min_confidence:
                patterns.append(pattern)

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        # Determine primary pattern
        primary = patterns[0].pattern_type if patterns else None

        # Calculate pattern consistency
        consistency = self._calculate_consistency(patterns, behavior_summary)

        return PatternDetectionResult(
            wallet_address=wallet_address,
            patterns=patterns,
            primary_pattern=primary,
            pattern_consistency=consistency,
        )

    def _check_serial_sniper(
        self,
        summary: WalletBehaviorSummary,
    ) -> Optional[WalletPattern]:
        """Check for serial sniper pattern."""
        if summary.sniper_count < self.min_tokens_for_pattern:
            return None

        # Calculate sniper ratio
        sniper_ratio = summary.sniper_count / summary.tokens_analyzed

        # Confidence based on ratio and count
        # High ratio + many tokens = high confidence
        ratio_component = min(sniper_ratio / 0.5, 1.0) * 0.5
        count_component = min(summary.sniper_count / 10, 1.0) * 0.5
        confidence = ratio_component + count_component

        if confidence < self.min_confidence:
            return None

        return WalletPattern(
            pattern_type=PatternType.SERIAL_SNIPER,
            confidence=confidence,
            token_count=summary.sniper_count,
            detected_at=datetime.now(timezone.utc),
            metadata={
                "sniper_ratio": sniper_ratio,
                "tokens_analyzed": summary.tokens_analyzed,
            },
        )

    def _check_diamond_hands(
        self,
        summary: WalletBehaviorSummary,
    ) -> Optional[WalletPattern]:
        """Check for diamond hands pattern."""
        if summary.accumulator_count < self.min_tokens_for_pattern:
            return None

        if summary.avg_holding_days is None or summary.avg_holding_days < 30:
            return None

        # Accumulator ratio
        acc_ratio = summary.accumulator_count / summary.tokens_analyzed

        # Confidence based on ratio, count, and holding duration
        ratio_component = min(acc_ratio / 0.5, 1.0) * 0.3
        count_component = min(summary.accumulator_count / 10, 1.0) * 0.3
        duration_component = min(summary.avg_holding_days / 90, 1.0) * 0.4
        confidence = ratio_component + count_component + duration_component

        if confidence < self.min_confidence:
            return None

        return WalletPattern(
            pattern_type=PatternType.DIAMOND_HANDS,
            confidence=confidence,
            token_count=summary.accumulator_count,
            detected_at=datetime.now(timezone.utc),
            metadata={
                "accumulator_ratio": acc_ratio,
                "avg_holding_days": summary.avg_holding_days,
            },
        )

    def _check_paper_hands(
        self,
        summary: WalletBehaviorSummary,
    ) -> Optional[WalletPattern]:
        """Check for paper hands pattern."""
        if summary.avg_holding_days is None or summary.avg_holding_days > 7:
            return None

        # Low holding duration across many tokens
        if summary.tokens_analyzed < self.min_tokens_for_pattern:
            return None

        # Very short holding = higher confidence
        duration_factor = 1.0 - (summary.avg_holding_days / 7)
        token_factor = min(summary.tokens_analyzed / 10, 1.0)
        confidence = duration_factor * 0.6 + token_factor * 0.4

        if confidence < self.min_confidence:
            return None

        return WalletPattern(
            pattern_type=PatternType.PAPER_HANDS,
            confidence=confidence,
            token_count=summary.tokens_analyzed,
            detected_at=datetime.now(timezone.utc),
            metadata={
                "avg_holding_days": summary.avg_holding_days,
            },
        )

    def _check_rugpull_survivor(
        self,
        summary: WalletBehaviorSummary,
    ) -> Optional[WalletPattern]:
        """Check for rugpull survivor pattern."""
        if summary.rugpull_count < 2:
            return None

        # This would require exit timing data vs rug timing
        # For now, use early_exit_count as proxy (if available)
        # A survivor would have high early exits on rugged tokens

        # Simplified: if they touched rugged tokens but have positive PnL
        if summary.avg_pnl_pct is not None and summary.avg_pnl_pct > 0:
            survival_rate = 1.0  # Simplified
            confidence = min(summary.rugpull_count / 5, 1.0) * 0.7 + 0.3
        else:
            return None

        if confidence < self.min_confidence:
            return None

        return WalletPattern(
            pattern_type=PatternType.RUGPULL_SURVIVOR,
            confidence=confidence,
            token_count=summary.rugpull_count,
            detected_at=datetime.now(timezone.utc),
            metadata={
                "rugpull_count": summary.rugpull_count,
                "avg_pnl_pct": summary.avg_pnl_pct,
            },
        )

    def _check_rugpull_victim(
        self,
        summary: WalletBehaviorSummary,
    ) -> Optional[WalletPattern]:
        """Check for rugpull victim pattern."""
        if summary.rugpull_count < 2:
            return None

        # Victim would have negative PnL and exposure to rugs
        if summary.avg_pnl_pct is not None and summary.avg_pnl_pct < -20:
            rug_ratio = summary.rugpull_count / summary.tokens_analyzed
            loss_factor = min(abs(summary.avg_pnl_pct) / 50, 1.0)
            confidence = rug_ratio * 0.5 + loss_factor * 0.5
        else:
            return None

        if confidence < self.min_confidence:
            return None

        return WalletPattern(
            pattern_type=PatternType.RUGPULL_VICTIM,
            confidence=confidence,
            token_count=summary.rugpull_count,
            detected_at=datetime.now(timezone.utc),
            metadata={
                "rugpull_count": summary.rugpull_count,
                "avg_pnl_pct": summary.avg_pnl_pct,
            },
        )

    def _check_profit_taker(
        self,
        summary: WalletBehaviorSummary,
    ) -> Optional[WalletPattern]:
        """Check for consistent profit taker pattern."""
        if summary.avg_pnl_pct is None or summary.avg_pnl_pct < 20:
            return None

        if summary.tokens_analyzed < self.min_tokens_for_pattern:
            return None

        # High average profit = higher confidence
        profit_factor = min(summary.avg_pnl_pct / 100, 1.0)
        token_factor = min(summary.tokens_analyzed / 10, 1.0)
        confidence = profit_factor * 0.6 + token_factor * 0.4

        if confidence < self.min_confidence:
            return None

        return WalletPattern(
            pattern_type=PatternType.PROFIT_TAKER,
            confidence=confidence,
            token_count=summary.tokens_analyzed,
            detected_at=datetime.now(timezone.utc),
            metadata={
                "avg_pnl_pct": summary.avg_pnl_pct,
            },
        )

    def _check_loss_maker(
        self,
        summary: WalletBehaviorSummary,
    ) -> Optional[WalletPattern]:
        """Check for consistent loss maker pattern."""
        if summary.avg_pnl_pct is None or summary.avg_pnl_pct > -20:
            return None

        if summary.tokens_analyzed < self.min_tokens_for_pattern:
            return None

        # High average loss = higher confidence for this pattern
        loss_factor = min(abs(summary.avg_pnl_pct) / 50, 1.0)
        token_factor = min(summary.tokens_analyzed / 10, 1.0)
        confidence = loss_factor * 0.6 + token_factor * 0.4

        if confidence < self.min_confidence:
            return None

        return WalletPattern(
            pattern_type=PatternType.LOSS_MAKER,
            confidence=confidence,
            token_count=summary.tokens_analyzed,
            detected_at=datetime.now(timezone.utc),
            metadata={
                "avg_pnl_pct": summary.avg_pnl_pct,
            },
        )

    def _calculate_consistency(
        self,
        patterns: list[WalletPattern],
        summary: WalletBehaviorSummary,
    ) -> float:
        """
        Calculate how consistent the wallet's behavior is.

        A wallet with one dominant pattern is more consistent than
        one with multiple weak patterns.
        """
        if not patterns:
            return 0.0

        if len(patterns) == 1:
            return patterns[0].confidence

        # Calculate variance in confidence scores
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
        variance = sum((p.confidence - avg_confidence) ** 2 for p in patterns) / len(
            patterns
        )

        # High variance (one dominant pattern) = high consistency
        # Low variance (many similar patterns) = low consistency
        max_confidence = patterns[0].confidence
        dominance = max_confidence - avg_confidence

        return min(dominance + 0.5, 1.0)

    def patterns_to_dict(self, patterns: list[WalletPattern]) -> list[dict]:
        """Convert patterns to dictionary format for storage."""
        return [
            {
                "type": p.pattern_type.value,
                "confidence": p.confidence,
                "token_count": p.token_count,
                "detected_at": p.detected_at.isoformat(),
                **p.metadata,
            }
            for p in patterns
        ]
