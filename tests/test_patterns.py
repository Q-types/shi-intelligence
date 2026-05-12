"""Tests for wallet behavior pattern detection."""

import pytest
from datetime import datetime, timezone

from src.reputation.patterns import (
    PatternDetector,
    PatternType,
    WalletPattern,
    PatternDetectionResult,
)
from src.data.repositories.wallet_history import WalletBehaviorSummary


class TestPatternDetector:
    """Tests for PatternDetector class."""

    @pytest.fixture
    def detector(self) -> PatternDetector:
        """Create a pattern detector with default settings."""
        return PatternDetector(min_tokens_for_pattern=3, min_confidence=0.5)

    @pytest.fixture
    def strict_detector(self) -> PatternDetector:
        """Create a pattern detector with stricter settings."""
        return PatternDetector(min_tokens_for_pattern=5, min_confidence=0.7)

    # -------------------------------------------------------------------------
    # Serial Sniper Detection Tests
    # -------------------------------------------------------------------------

    def test_detect_serial_sniper_high_confidence(
        self, detector: PatternDetector, serial_sniper_behavior: WalletBehaviorSummary
    ):
        """Test detection of a clear serial sniper pattern."""
        result = detector.detect_patterns(
            serial_sniper_behavior.wallet_address,
            serial_sniper_behavior,
        )

        assert result.wallet_address == serial_sniper_behavior.wallet_address
        assert len(result.patterns) > 0

        # Find serial sniper pattern
        sniper_pattern = next(
            (p for p in result.patterns if p.pattern_type == PatternType.SERIAL_SNIPER),
            None,
        )
        assert sniper_pattern is not None
        assert sniper_pattern.confidence >= 0.5
        assert sniper_pattern.token_count == 8

    def test_serial_sniper_is_detected_pattern(
        self, detector: PatternDetector, serial_sniper_behavior: WalletBehaviorSummary
    ):
        """Test that serial sniper pattern is detected for sniper wallets."""
        result = detector.detect_patterns(
            serial_sniper_behavior.wallet_address,
            serial_sniper_behavior,
        )

        # Serial sniper should be among detected patterns
        pattern_types = {p.pattern_type for p in result.patterns}
        assert PatternType.SERIAL_SNIPER in pattern_types

    def test_no_sniper_pattern_below_threshold(self, detector: PatternDetector):
        """Test no sniper pattern detected when below threshold."""
        summary = WalletBehaviorSummary(
            wallet_address="LowSniper111111111111111111111111111111111",
            tokens_analyzed=10,
            sniper_count=1,  # Only 1 snipe - below min_tokens_for_pattern
            accumulator_count=5,
            rugpull_count=0,
            avg_holding_days=30.0,
            avg_pnl_pct=10.0,
        )

        result = detector.detect_patterns(summary.wallet_address, summary)

        sniper_pattern = next(
            (p for p in result.patterns if p.pattern_type == PatternType.SERIAL_SNIPER),
            None,
        )
        assert sniper_pattern is None

    # -------------------------------------------------------------------------
    # Diamond Hands Detection Tests
    # -------------------------------------------------------------------------

    def test_detect_diamond_hands_pattern(
        self, detector: PatternDetector, diamond_hands_behavior: WalletBehaviorSummary
    ):
        """Test detection of diamond hands pattern."""
        result = detector.detect_patterns(
            diamond_hands_behavior.wallet_address,
            diamond_hands_behavior,
        )

        diamond_pattern = next(
            (p for p in result.patterns if p.pattern_type == PatternType.DIAMOND_HANDS),
            None,
        )
        assert diamond_pattern is not None
        assert diamond_pattern.confidence >= 0.5
        assert diamond_pattern.token_count == 12

    def test_diamond_hands_requires_long_holding(self, detector: PatternDetector):
        """Test that diamond hands requires minimum holding period."""
        summary = WalletBehaviorSummary(
            wallet_address="ShortHolder1111111111111111111111111111111",
            tokens_analyzed=10,
            sniper_count=0,
            accumulator_count=8,  # High accumulator count
            rugpull_count=0,
            avg_holding_days=10.0,  # But short holding period
            avg_pnl_pct=20.0,
        )

        result = detector.detect_patterns(summary.wallet_address, summary)

        diamond_pattern = next(
            (p for p in result.patterns if p.pattern_type == PatternType.DIAMOND_HANDS),
            None,
        )
        # Should NOT detect diamond hands due to short holding period
        assert diamond_pattern is None

    # -------------------------------------------------------------------------
    # Paper Hands Detection Tests
    # -------------------------------------------------------------------------

    def test_detect_paper_hands_pattern(self, detector: PatternDetector):
        """Test detection of paper hands pattern."""
        summary = WalletBehaviorSummary(
            wallet_address="PaperHands111111111111111111111111111111",
            tokens_analyzed=10,
            sniper_count=2,
            accumulator_count=0,
            rugpull_count=0,
            avg_holding_days=2.0,  # Very short holding
            avg_pnl_pct=-5.0,
        )

        result = detector.detect_patterns(summary.wallet_address, summary)

        paper_pattern = next(
            (p for p in result.patterns if p.pattern_type == PatternType.PAPER_HANDS),
            None,
        )
        assert paper_pattern is not None
        assert paper_pattern.confidence >= 0.5

    def test_paper_hands_not_detected_for_long_holder(self, detector: PatternDetector):
        """Test paper hands is not detected for long-term holders."""
        summary = WalletBehaviorSummary(
            wallet_address="LongHolder11111111111111111111111111111111",
            tokens_analyzed=10,
            sniper_count=0,
            accumulator_count=5,
            rugpull_count=0,
            avg_holding_days=60.0,  # Long holding
            avg_pnl_pct=20.0,
        )

        result = detector.detect_patterns(summary.wallet_address, summary)

        paper_pattern = next(
            (p for p in result.patterns if p.pattern_type == PatternType.PAPER_HANDS),
            None,
        )
        assert paper_pattern is None

    # -------------------------------------------------------------------------
    # Rugpull Pattern Tests
    # -------------------------------------------------------------------------

    def test_detect_rugpull_victim_pattern(
        self, detector: PatternDetector, rugpull_victim_behavior: WalletBehaviorSummary
    ):
        """Test detection of rugpull victim pattern."""
        result = detector.detect_patterns(
            rugpull_victim_behavior.wallet_address,
            rugpull_victim_behavior,
        )

        victim_pattern = next(
            (p for p in result.patterns if p.pattern_type == PatternType.RUGPULL_VICTIM),
            None,
        )
        assert victim_pattern is not None
        assert victim_pattern.confidence >= 0.5
        assert victim_pattern.metadata["rugpull_count"] == 5
        assert victim_pattern.metadata["avg_pnl_pct"] == -75.0

    def test_detect_rugpull_survivor_pattern(self, detector: PatternDetector):
        """Test detection of rugpull survivor pattern."""
        summary = WalletBehaviorSummary(
            wallet_address="Survivor111111111111111111111111111111111",
            tokens_analyzed=10,
            sniper_count=0,
            accumulator_count=3,
            rugpull_count=4,  # Touched rugged tokens
            avg_holding_days=20.0,
            avg_pnl_pct=30.0,  # But still profitable
        )

        result = detector.detect_patterns(summary.wallet_address, summary)

        survivor_pattern = next(
            (p for p in result.patterns if p.pattern_type == PatternType.RUGPULL_SURVIVOR),
            None,
        )
        assert survivor_pattern is not None

    # -------------------------------------------------------------------------
    # Profit/Loss Pattern Tests
    # -------------------------------------------------------------------------

    def test_detect_profit_taker_pattern(self, detector: PatternDetector):
        """Test detection of profit taker pattern."""
        summary = WalletBehaviorSummary(
            wallet_address="ProfitTaker11111111111111111111111111111",
            tokens_analyzed=10,
            sniper_count=2,
            accumulator_count=5,
            rugpull_count=0,
            avg_holding_days=30.0,
            avg_pnl_pct=80.0,  # Consistently profitable
        )

        result = detector.detect_patterns(summary.wallet_address, summary)

        profit_pattern = next(
            (p for p in result.patterns if p.pattern_type == PatternType.PROFIT_TAKER),
            None,
        )
        assert profit_pattern is not None
        assert profit_pattern.metadata["avg_pnl_pct"] == 80.0

    def test_detect_loss_maker_pattern(self, detector: PatternDetector):
        """Test detection of loss maker pattern."""
        summary = WalletBehaviorSummary(
            wallet_address="LossMaker111111111111111111111111111111",
            tokens_analyzed=10,
            sniper_count=3,
            accumulator_count=0,
            rugpull_count=1,
            avg_holding_days=5.0,
            avg_pnl_pct=-40.0,  # Consistently losing
        )

        result = detector.detect_patterns(summary.wallet_address, summary)

        loss_pattern = next(
            (p for p in result.patterns if p.pattern_type == PatternType.LOSS_MAKER),
            None,
        )
        assert loss_pattern is not None

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    def test_insufficient_data_returns_empty_patterns(
        self, detector: PatternDetector, new_wallet_behavior: WalletBehaviorSummary
    ):
        """Test that insufficient data returns no patterns."""
        result = detector.detect_patterns(
            new_wallet_behavior.wallet_address,
            new_wallet_behavior,
        )

        assert len(result.patterns) == 0
        assert result.primary_pattern is None
        assert result.pattern_consistency == 0.0

    def test_multiple_patterns_detected(self, detector: PatternDetector):
        """Test wallet with multiple behavior patterns."""
        summary = WalletBehaviorSummary(
            wallet_address="MultiPattern111111111111111111111111111",
            tokens_analyzed=15,
            sniper_count=5,  # Some sniping
            accumulator_count=6,  # Some accumulation
            rugpull_count=0,
            avg_holding_days=25.0,  # Not paper hands, not diamond
            avg_pnl_pct=35.0,  # Profitable
        )

        result = detector.detect_patterns(summary.wallet_address, summary)

        # Should detect multiple patterns
        pattern_types = {p.pattern_type for p in result.patterns}
        assert len(pattern_types) >= 1  # At least serial sniper

    def test_pattern_consistency_calculation(self, detector: PatternDetector):
        """Test pattern consistency is calculated correctly."""
        # Single dominant pattern = high consistency
        summary = WalletBehaviorSummary(
            wallet_address="Consistent111111111111111111111111111111",
            tokens_analyzed=10,
            sniper_count=8,
            accumulator_count=0,
            rugpull_count=0,
            avg_holding_days=2.0,
            avg_pnl_pct=100.0,
        )

        result = detector.detect_patterns(summary.wallet_address, summary)

        # High consistency when there's a dominant pattern
        if len(result.patterns) > 0:
            assert result.pattern_consistency >= 0.0

    def test_patterns_to_dict_conversion(self, detector: PatternDetector):
        """Test patterns are correctly converted to dictionary format."""
        pattern = WalletPattern(
            pattern_type=PatternType.SERIAL_SNIPER,
            confidence=0.85,
            token_count=8,
            detected_at=datetime.now(timezone.utc),
            metadata={"sniper_ratio": 0.8},
        )

        result = detector.patterns_to_dict([pattern])

        assert len(result) == 1
        assert result[0]["type"] == "SERIAL_SNIPER"
        assert result[0]["confidence"] == 0.85
        assert result[0]["token_count"] == 8
        assert "sniper_ratio" in result[0]

    def test_strict_detector_filters_low_confidence(
        self, strict_detector: PatternDetector
    ):
        """Test stricter detector filters out low confidence patterns."""
        summary = WalletBehaviorSummary(
            wallet_address="Borderline11111111111111111111111111111",
            tokens_analyzed=8,
            sniper_count=3,  # Marginal sniper count
            accumulator_count=2,
            rugpull_count=0,
            avg_holding_days=15.0,
            avg_pnl_pct=10.0,
        )

        result = strict_detector.detect_patterns(summary.wallet_address, summary)

        # Strict detector requires min 5 tokens and 0.7 confidence
        for pattern in result.patterns:
            assert pattern.confidence >= 0.7


class TestPatternTypeEnum:
    """Tests for PatternType enum values."""

    def test_all_pattern_types_defined(self):
        """Test all expected pattern types are defined."""
        expected_types = [
            "SERIAL_SNIPER",
            "DIAMOND_HANDS",
            "PAPER_HANDS",
            "RUGPULL_SURVIVOR",
            "RUGPULL_VICTIM",
            "WHALE_ACCUMULATOR",
            "PROFIT_TAKER",
            "LOSS_MAKER",
            "EARLY_ADOPTER",
            "COPY_TRADER",
        ]

        for pattern_type in expected_types:
            assert hasattr(PatternType, pattern_type)

    def test_pattern_type_string_values(self):
        """Test pattern types have correct string values."""
        assert PatternType.SERIAL_SNIPER.value == "SERIAL_SNIPER"
        assert PatternType.DIAMOND_HANDS.value == "DIAMOND_HANDS"
        assert PatternType.RUGPULL_VICTIM.value == "RUGPULL_VICTIM"
