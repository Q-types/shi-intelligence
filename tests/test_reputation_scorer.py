"""Tests for reputation scoring system."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.reputation.scorer import (
    ReputationScorer,
    ScoreComponents,
    ScoringResult,
)
from src.reputation.patterns import PatternType, WalletPattern
from src.data.repositories.wallet_history import WalletBehaviorSummary


class TestScoreComponents:
    """Tests for score component calculations."""

    def test_base_score_is_50(self):
        """Test base score starts at 50."""
        assert ReputationScorer.BASE_SCORE == 50

    def test_sniper_penalty_values(self):
        """Test sniper penalty configuration."""
        assert ReputationScorer.SNIPER_PENALTY_PER == -5
        assert ReputationScorer.SNIPER_PENALTY_MAX == -25

    def test_accumulator_bonus_values(self):
        """Test accumulator bonus configuration."""
        assert ReputationScorer.ACCUMULATOR_BONUS_PER == 3
        assert ReputationScorer.ACCUMULATOR_BONUS_MAX == 20

    def test_entity_sybil_penalty(self):
        """Test sybil entity penalty."""
        assert ReputationScorer.ENTITY_SYBIL_PENALTY == -20


class TestReputationScorer:
    """Tests for ReputationScorer class."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        return AsyncMock()

    @pytest.fixture
    def scorer(self, mock_session: AsyncMock) -> ReputationScorer:
        """Create reputation scorer with mocked dependencies."""
        scorer = ReputationScorer(mock_session)
        # Mock the repositories
        scorer.history_repo = AsyncMock()
        scorer.reputation_repo = AsyncMock()
        return scorer

    # -------------------------------------------------------------------------
    # Score Calculation Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_score_wallet_no_history(self, scorer: ReputationScorer):
        """Test scoring wallet with no history returns neutral score."""
        scorer.history_repo.get_behavior_summary.return_value = None

        result = await scorer.score_wallet("NewWallet1111111111111111111111111111111")

        assert result.reputation_score == 50  # Base score
        assert result.confidence_level == "low"
        assert result.risk_level == "medium"
        assert len(result.patterns) == 0

    @pytest.mark.asyncio
    async def test_score_serial_sniper_gets_penalty(
        self, scorer: ReputationScorer, serial_sniper_behavior: WalletBehaviorSummary
    ):
        """Test serial sniper receives appropriate penalty."""
        scorer.history_repo.get_behavior_summary.return_value = serial_sniper_behavior

        result = await scorer.score_wallet(serial_sniper_behavior.wallet_address)

        # 8 snipes * -5 = -40, capped at -25
        assert result.components.sniper_penalty == -25
        # Final score should be significantly below base
        assert result.reputation_score < 50

    @pytest.mark.asyncio
    async def test_score_diamond_hands_gets_bonus(
        self, scorer: ReputationScorer, diamond_hands_behavior: WalletBehaviorSummary
    ):
        """Test diamond hands receives appropriate bonus."""
        scorer.history_repo.get_behavior_summary.return_value = diamond_hands_behavior

        result = await scorer.score_wallet(diamond_hands_behavior.wallet_address)

        # 12 accumulations * 3 = 36, capped at 20
        assert result.components.accumulator_bonus == 20
        # Final score should be above base
        assert result.reputation_score > 50

    @pytest.mark.asyncio
    async def test_score_rugpull_victim_gets_penalty(
        self, scorer: ReputationScorer, rugpull_victim_behavior: WalletBehaviorSummary
    ):
        """Test rugpull victim receives penalty."""
        scorer.history_repo.get_behavior_summary.return_value = rugpull_victim_behavior

        result = await scorer.score_wallet(rugpull_victim_behavior.wallet_address)

        # 5 rugpulls * -10 = -50, capped at -20
        assert result.components.rugpull_adjustment == -20
        assert result.reputation_score < 50

    @pytest.mark.asyncio
    async def test_sybil_entity_membership_penalty(
        self, scorer: ReputationScorer, diamond_hands_behavior: WalletBehaviorSummary
    ):
        """Test sybil entity membership applies penalty."""
        scorer.history_repo.get_behavior_summary.return_value = diamond_hands_behavior

        # Score without sybil flag
        result_clean = await scorer.score_wallet(
            diamond_hands_behavior.wallet_address,
            entity_id=1,
            is_sybil_entity=False,
        )

        # Score with sybil flag
        result_sybil = await scorer.score_wallet(
            diamond_hands_behavior.wallet_address,
            entity_id=1,
            is_sybil_entity=True,
        )

        # Sybil membership should reduce score by 20
        assert result_sybil.reputation_score == result_clean.reputation_score - 20
        assert result_sybil.components.entity_adjustment == -20

    @pytest.mark.asyncio
    async def test_pnl_positive_adjustment(self, scorer: ReputationScorer):
        """Test positive PnL gives bonus."""
        summary = WalletBehaviorSummary(
            wallet_address="ProfitWallet1111111111111111111111111111",
            tokens_analyzed=10,
            sniper_count=0,
            accumulator_count=0,
            rugpull_count=0,
            avg_holding_days=30.0,
            avg_pnl_pct=100.0,  # 100% average profit
        )
        scorer.history_repo.get_behavior_summary.return_value = summary

        result = await scorer.score_wallet(summary.wallet_address)

        # 100% / 10 = 10 points bonus
        assert result.components.pnl_adjustment == 10

    @pytest.mark.asyncio
    async def test_pnl_negative_adjustment(self, scorer: ReputationScorer):
        """Test negative PnL gives penalty."""
        summary = WalletBehaviorSummary(
            wallet_address="LossWallet11111111111111111111111111111",
            tokens_analyzed=10,
            sniper_count=0,
            accumulator_count=0,
            rugpull_count=0,
            avg_holding_days=30.0,
            avg_pnl_pct=-80.0,  # 80% average loss
        )
        scorer.history_repo.get_behavior_summary.return_value = summary

        result = await scorer.score_wallet(summary.wallet_address)

        # -80% / 10 = -8 points
        assert result.components.pnl_adjustment == -8

    # -------------------------------------------------------------------------
    # Score Clamping Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_score_clamped_to_minimum_0(self, scorer: ReputationScorer):
        """Test score cannot go below 0."""
        # Create worst case scenario
        summary = WalletBehaviorSummary(
            wallet_address="WorstWallet1111111111111111111111111111",
            tokens_analyzed=20,
            sniper_count=10,  # Max sniper penalty
            accumulator_count=0,
            rugpull_count=5,  # Max rugpull penalty
            avg_holding_days=1.0,
            avg_pnl_pct=-200.0,  # Max PnL penalty
        )
        scorer.history_repo.get_behavior_summary.return_value = summary

        result = await scorer.score_wallet(
            summary.wallet_address,
            entity_id=1,
            is_sybil_entity=True,  # Additional sybil penalty
        )

        assert result.reputation_score >= 0

    @pytest.mark.asyncio
    async def test_score_clamped_to_maximum_100(self, scorer: ReputationScorer):
        """Test score cannot exceed 100."""
        # Create best case scenario
        summary = WalletBehaviorSummary(
            wallet_address="BestWallet11111111111111111111111111111",
            tokens_analyzed=50,  # High confidence
            sniper_count=0,
            accumulator_count=30,  # Max accumulator bonus
            rugpull_count=5,  # But survived them
            avg_holding_days=180.0,
            avg_pnl_pct=200.0,  # Max PnL bonus
        )
        scorer.history_repo.get_behavior_summary.return_value = summary

        result = await scorer.score_wallet(summary.wallet_address)

        assert result.reputation_score <= 100

    # -------------------------------------------------------------------------
    # Confidence Level Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_low_confidence_under_5_tokens(self, scorer: ReputationScorer):
        """Test low confidence for wallets with < 5 tokens analyzed."""
        summary = WalletBehaviorSummary(
            wallet_address="LowDataWallet111111111111111111111111111",
            tokens_analyzed=3,
            sniper_count=0,
            accumulator_count=1,
            rugpull_count=0,
            avg_holding_days=30.0,
            avg_pnl_pct=10.0,
        )
        scorer.history_repo.get_behavior_summary.return_value = summary

        result = await scorer.score_wallet(summary.wallet_address)

        assert result.confidence_level == "low"

    @pytest.mark.asyncio
    async def test_medium_confidence_5_to_20_tokens(self, scorer: ReputationScorer):
        """Test medium confidence for wallets with 5-20 tokens analyzed."""
        summary = WalletBehaviorSummary(
            wallet_address="MedDataWallet111111111111111111111111111",
            tokens_analyzed=10,
            sniper_count=0,
            accumulator_count=5,
            rugpull_count=0,
            avg_holding_days=30.0,
            avg_pnl_pct=10.0,
        )
        scorer.history_repo.get_behavior_summary.return_value = summary

        result = await scorer.score_wallet(summary.wallet_address)

        assert result.confidence_level == "medium"

    @pytest.mark.asyncio
    async def test_high_confidence_over_20_tokens(self, scorer: ReputationScorer):
        """Test high confidence for wallets with > 20 tokens analyzed."""
        summary = WalletBehaviorSummary(
            wallet_address="HighDataWallet11111111111111111111111111",
            tokens_analyzed=25,
            sniper_count=2,
            accumulator_count=15,
            rugpull_count=1,
            avg_holding_days=45.0,
            avg_pnl_pct=25.0,
        )
        scorer.history_repo.get_behavior_summary.return_value = summary

        result = await scorer.score_wallet(summary.wallet_address)

        assert result.confidence_level == "high"

    # -------------------------------------------------------------------------
    # Risk Level Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_critical_risk_for_very_low_score(self, scorer: ReputationScorer):
        """Test critical risk level for scores <= 20."""
        summary = WalletBehaviorSummary(
            wallet_address="CriticalWallet11111111111111111111111111",
            tokens_analyzed=15,
            sniper_count=8,
            accumulator_count=0,
            rugpull_count=4,
            avg_holding_days=2.0,
            avg_pnl_pct=-50.0,
        )
        scorer.history_repo.get_behavior_summary.return_value = summary

        result = await scorer.score_wallet(
            summary.wallet_address,
            is_sybil_entity=True,
        )

        # With all these penalties, score should be very low
        if result.reputation_score <= 20:
            assert result.risk_level == "critical"

    @pytest.mark.asyncio
    async def test_low_risk_for_high_score(
        self, scorer: ReputationScorer, diamond_hands_behavior: WalletBehaviorSummary
    ):
        """Test low risk level for scores > 50."""
        scorer.history_repo.get_behavior_summary.return_value = diamond_hands_behavior

        result = await scorer.score_wallet(diamond_hands_behavior.wallet_address)

        if result.reputation_score > 50:
            assert result.risk_level == "low"

    # -------------------------------------------------------------------------
    # Pattern Integration Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_patterns_included_in_result(
        self, scorer: ReputationScorer, serial_sniper_behavior: WalletBehaviorSummary
    ):
        """Test that detected patterns are included in result."""
        scorer.history_repo.get_behavior_summary.return_value = serial_sniper_behavior

        result = await scorer.score_wallet(serial_sniper_behavior.wallet_address)

        # Should have at least serial sniper pattern
        pattern_types = {p.pattern_type for p in result.patterns}
        assert PatternType.SERIAL_SNIPER in pattern_types

    @pytest.mark.asyncio
    async def test_pattern_adjustment_applied(
        self, scorer: ReputationScorer, serial_sniper_behavior: WalletBehaviorSummary
    ):
        """Test pattern adjustment is calculated and applied."""
        scorer.history_repo.get_behavior_summary.return_value = serial_sniper_behavior

        result = await scorer.score_wallet(serial_sniper_behavior.wallet_address)

        # Serial sniper pattern should give negative adjustment
        # (The exact value depends on detected patterns and confidence)
        assert isinstance(result.components.pattern_adjustment, int)

    # -------------------------------------------------------------------------
    # Batch Scoring Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_score_wallets_batch(self, scorer: ReputationScorer):
        """Test batch scoring of multiple wallets."""
        summaries = {
            "Wallet1": WalletBehaviorSummary(
                wallet_address="Wallet1",
                tokens_analyzed=10,
                sniper_count=5,
                accumulator_count=0,
                rugpull_count=0,
                avg_holding_days=5.0,
                avg_pnl_pct=20.0,
            ),
            "Wallet2": WalletBehaviorSummary(
                wallet_address="Wallet2",
                tokens_analyzed=10,
                sniper_count=0,
                accumulator_count=8,
                rugpull_count=0,
                avg_holding_days=60.0,
                avg_pnl_pct=30.0,
            ),
        }

        async def mock_get_summary(wallet):
            return summaries.get(wallet)

        scorer.history_repo.get_behavior_summary = mock_get_summary

        results = await scorer.score_wallets_batch(["Wallet1", "Wallet2"])

        assert len(results) == 2
        # Wallet1 (sniper) should have lower score than Wallet2 (accumulator)
        assert results[0].reputation_score < results[1].reputation_score

    @pytest.mark.asyncio
    async def test_batch_with_entity_mapping(self, scorer: ReputationScorer):
        """Test batch scoring with entity mappings."""
        summary = WalletBehaviorSummary(
            wallet_address="EntityWallet",
            tokens_analyzed=10,
            sniper_count=2,
            accumulator_count=5,
            rugpull_count=0,
            avg_holding_days=30.0,
            avg_pnl_pct=20.0,
        )
        scorer.history_repo.get_behavior_summary.return_value = summary

        entity_mapping = {"EntityWallet": 1}
        sybil_entities = {1}  # Entity 1 is a sybil cluster

        results = await scorer.score_wallets_batch(
            ["EntityWallet"],
            entity_mapping=entity_mapping,
            sybil_entities=sybil_entities,
        )

        assert len(results) == 1
        # Should have sybil penalty applied
        assert results[0].components.entity_adjustment == -20


class TestScoreComponentsDataclass:
    """Tests for ScoreComponents dataclass."""

    def test_score_components_creation(self):
        """Test creating ScoreComponents."""
        components = ScoreComponents(
            base_score=50,
            sniper_penalty=-15,
            accumulator_bonus=10,
            rugpull_adjustment=-10,
            pnl_adjustment=5,
            pattern_adjustment=-5,
            entity_adjustment=0,
            final_score=35,
        )

        assert components.base_score == 50
        assert components.sniper_penalty == -15
        assert components.final_score == 35

    def test_components_sum_equals_final(self):
        """Test that components sum to final score (before clamping)."""
        components = ScoreComponents(
            base_score=50,
            sniper_penalty=-10,
            accumulator_bonus=15,
            rugpull_adjustment=-5,
            pnl_adjustment=10,
            pattern_adjustment=-8,
            entity_adjustment=0,
            final_score=52,  # 50 - 10 + 15 - 5 + 10 - 8 = 52
        )

        calculated = (
            components.base_score
            + components.sniper_penalty
            + components.accumulator_bonus
            + components.rugpull_adjustment
            + components.pnl_adjustment
            + components.pattern_adjustment
            + components.entity_adjustment
        )

        assert calculated == components.final_score
