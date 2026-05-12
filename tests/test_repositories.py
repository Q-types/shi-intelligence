"""Tests for cross-token intelligence repositories."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.repositories.wallet_history import (
    WalletHistoryRepository,
    WalletBehaviorSummary,
    TokenInteraction,
)
from src.data.repositories.entity import (
    EntityRepository,
    EntitySummary,
    MembershipInfo,
)
from src.data.repositories.wallet_reputation import (
    WalletReputationRepository,
    ReputationSummary,
)
from src.data.models import (
    WalletHistory,
    Entity,
    EntityMembership,
    WalletReputation,
    EntityType,
    DetectionMethod,
    ConfidenceLevel,
)


# =============================================================================
# WalletHistoryRepository Tests
# =============================================================================


class TestWalletHistoryRepository:
    """Tests for WalletHistoryRepository."""

    @pytest.fixture
    def repo(self, mock_async_session: AsyncMock) -> WalletHistoryRepository:
        """Create repository with mock session."""
        return WalletHistoryRepository(mock_async_session)

    @pytest.mark.asyncio
    async def test_get_wallet_history(self, repo: WalletHistoryRepository):
        """Test retrieving wallet history."""
        wallet_address = "TestWallet1111111111111111111111111111111"

        # Mock the query result
        mock_history = MagicMock(spec=WalletHistory)
        mock_history.wallet_address = wallet_address
        mock_history.token_mint = "TokenMint1111111111111111111111111111111"
        mock_history.first_seen_at = datetime.now(timezone.utc)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_history]
        repo.session.execute.return_value = mock_result

        result = await repo.get_wallet_history(wallet_address)

        assert len(result) == 1
        assert result[0].wallet_address == wallet_address
        repo.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_wallet_tokens(self, repo: WalletHistoryRepository):
        """Test retrieving tokens a wallet has interacted with."""
        wallet_address = "TestWallet1111111111111111111111111111111"
        expected_tokens = [
            "Token1111111111111111111111111111111111111",
            "Token2222222222222222222222222222222222222",
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(t,) for t in expected_tokens]
        repo.session.execute.return_value = mock_result

        tokens = await repo.get_wallet_tokens(wallet_address)

        assert tokens == expected_tokens

    @pytest.mark.asyncio
    async def test_get_token_wallets(self, repo: WalletHistoryRepository):
        """Test retrieving wallets that interacted with a token."""
        token_mint = "TestToken11111111111111111111111111111111"
        expected_wallets = [
            "Wallet11111111111111111111111111111111111",
            "Wallet22222222222222222222222222222222222",
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(w,) for w in expected_wallets]
        repo.session.execute.return_value = mock_result

        wallets = await repo.get_token_wallets(token_mint)

        assert wallets == expected_wallets

    @pytest.mark.asyncio
    async def test_get_behavior_summary_no_data(self, repo: WalletHistoryRepository):
        """Test behavior summary returns None when no data exists."""
        wallet_address = "NewWallet111111111111111111111111111111"

        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.tokens_analyzed = 0
        mock_result.fetchone.return_value = mock_row
        repo.session.execute.return_value = mock_result

        summary = await repo.get_behavior_summary(wallet_address)

        assert summary is None

    @pytest.mark.asyncio
    async def test_get_behavior_summary_with_data(self, repo: WalletHistoryRepository):
        """Test behavior summary returns correct data."""
        wallet_address = "ActiveWallet11111111111111111111111111111"

        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.tokens_analyzed = 10
        mock_row.sniper_count = 3
        mock_row.accumulator_count = 5
        mock_row.rugpull_count = 1
        mock_row.avg_holding_days = 30.5
        mock_row.avg_pnl_pct = 25.0
        mock_result.fetchone.return_value = mock_row
        repo.session.execute.return_value = mock_result

        summary = await repo.get_behavior_summary(wallet_address)

        assert summary is not None
        assert summary.wallet_address == wallet_address
        assert summary.tokens_analyzed == 10
        assert summary.sniper_count == 3
        assert summary.accumulator_count == 5

    @pytest.mark.asyncio
    async def test_find_serial_snipers(self, repo: WalletHistoryRepository):
        """Test finding serial sniper wallets."""
        expected_snipers = [
            ("Sniper1" + "x" * 37, 8),
            ("Sniper2" + "x" * 37, 5),
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = expected_snipers
        repo.session.execute.return_value = mock_result

        snipers = await repo.find_serial_snipers(min_sniper_count=3)

        assert len(snipers) == 2
        assert snipers[0][1] == 8  # First sniper has 8 snipes

    @pytest.mark.asyncio
    async def test_find_diamond_hands(self, repo: WalletHistoryRepository):
        """Test finding diamond hands wallets."""
        expected_holders = [
            ("Diamond1" + "x" * 36, 10, 90.0),  # (wallet, acc_count, avg_days)
            ("Diamond2" + "x" * 36, 8, 60.0),
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = expected_holders
        repo.session.execute.return_value = mock_result

        holders = await repo.find_diamond_hands(min_accumulator_count=5, min_avg_holding_days=30.0)

        assert len(holders) == 2
        assert holders[0][2] == 90.0  # First holder has 90 avg days

    @pytest.mark.asyncio
    async def test_find_wallets_with_shared_tokens(self, repo: WalletHistoryRepository):
        """Test finding wallets with overlapping token interactions."""
        wallets = ["W1" + "x" * 40, "W2" + "x" * 40, "W3" + "x" * 40]

        # Simulate W1 and W2 sharing tokens T1 and T2
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (wallets[0], "T1" + "x" * 40),
            (wallets[0], "T2" + "x" * 40),
            (wallets[1], "T1" + "x" * 40),
            (wallets[1], "T2" + "x" * 40),
            (wallets[2], "T3" + "x" * 40),  # W3 has different token
        ]
        repo.session.execute.return_value = mock_result

        overlaps = await repo.find_wallets_with_shared_tokens(wallets, min_shared_tokens=2)

        # W1 and W2 share 2 tokens
        assert (wallets[0], wallets[1]) in overlaps or (wallets[1], wallets[0]) in overlaps

    @pytest.mark.asyncio
    async def test_mark_token_rugged(self, repo: WalletHistoryRepository):
        """Test marking a token as rugged."""
        token_mint = "RuggedToken111111111111111111111111111"

        mock_result = MagicMock()
        mock_result.rowcount = 5
        repo.session.execute.return_value = mock_result

        affected = await repo.mark_token_rugged(token_mint)

        assert affected == 5
        repo.session.commit.assert_called_once()


# =============================================================================
# EntityRepository Tests
# =============================================================================


class TestEntityRepository:
    """Tests for EntityRepository."""

    @pytest.fixture
    def repo(self, mock_async_session: AsyncMock) -> EntityRepository:
        """Create repository with mock session."""
        return EntityRepository(mock_async_session)

    @pytest.mark.asyncio
    async def test_create_entity(
        self, repo: EntityRepository, sample_entity_data: dict
    ):
        """Test creating a new entity."""
        entity = await repo.create_entity(
            entity_type=sample_entity_data["entity_type"],
            detection_method=sample_entity_data["detection_method"],
            dominant_funder_address=sample_entity_data["dominant_funder_address"],
            confidence_score=sample_entity_data["confidence_score"],
        )

        repo.session.add.assert_called_once()
        repo.session.commit.assert_called_once()
        repo.session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_entity(self, repo: EntityRepository):
        """Test retrieving an entity by ID."""
        mock_entity = MagicMock(spec=Entity)
        mock_entity.id = 1
        mock_entity.entity_type = EntityType.SYBIL_CLUSTER

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_entity
        repo.session.execute.return_value = mock_result

        entity = await repo.get_entity(1)

        assert entity is not None
        assert entity.id == 1

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, repo: EntityRepository):
        """Test retrieving non-existent entity returns None."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        repo.session.execute.return_value = mock_result

        entity = await repo.get_entity(999)

        assert entity is None

    @pytest.mark.asyncio
    async def test_get_entity_for_wallet(self, repo: EntityRepository):
        """Test retrieving entity for a specific wallet."""
        wallet_address = "MemberWallet11111111111111111111111111111"

        mock_entity = MagicMock(spec=Entity)
        mock_entity.id = 1

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_entity
        repo.session.execute.return_value = mock_result

        entity = await repo.get_entity_for_wallet(wallet_address)

        assert entity is not None

    @pytest.mark.asyncio
    async def test_get_entity_wallets(self, repo: EntityRepository):
        """Test retrieving all wallets in an entity."""
        expected_wallets = [
            "Member1" + "x" * 37,
            "Member2" + "x" * 37,
            "Member3" + "x" * 37,
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(w,) for w in expected_wallets]
        repo.session.execute.return_value = mock_result

        wallets = await repo.get_entity_wallets(entity_id=1)

        assert wallets == expected_wallets

    @pytest.mark.asyncio
    async def test_find_entities_by_type(self, repo: EntityRepository):
        """Test finding entities by type."""
        mock_entities = [MagicMock(spec=Entity) for _ in range(3)]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_entities
        repo.session.execute.return_value = mock_result

        entities = await repo.find_entities_by_type(EntityType.SYBIL_CLUSTER)

        assert len(entities) == 3

    @pytest.mark.asyncio
    async def test_find_professional_sybils(self, repo: EntityRepository):
        """Test finding professional sybil networks."""
        mock_entities = [MagicMock(spec=Entity) for _ in range(2)]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_entities
        repo.session.execute.return_value = mock_result

        entities = await repo.find_professional_sybils(
            min_wallet_count=5,
            min_tokens_targeted=3,
        )

        assert len(entities) == 2

    @pytest.mark.asyncio
    async def test_merge_entities_requires_two(self, repo: EntityRepository):
        """Test that merging requires at least 2 entities."""
        with pytest.raises(ValueError, match="Need at least 2 entities"):
            await repo.merge_entities([1])

    @pytest.mark.asyncio
    async def test_delete_entity(self, repo: EntityRepository):
        """Test deleting an entity."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        repo.session.execute.return_value = mock_result

        deleted = await repo.delete_entity(1)

        assert deleted is True
        repo.session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_entity_not_found(self, repo: EntityRepository):
        """Test deleting non-existent entity returns False."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        repo.session.execute.return_value = mock_result

        deleted = await repo.delete_entity(999)

        assert deleted is False


# =============================================================================
# WalletReputationRepository Tests
# =============================================================================


class TestWalletReputationRepository:
    """Tests for WalletReputationRepository."""

    @pytest.fixture
    def repo(self, mock_async_session: AsyncMock) -> WalletReputationRepository:
        """Create repository with mock session."""
        return WalletReputationRepository(mock_async_session)

    @pytest.mark.asyncio
    async def test_get_reputation(self, repo: WalletReputationRepository):
        """Test retrieving wallet reputation."""
        wallet_address = "RepWallet1111111111111111111111111111111"

        mock_rep = MagicMock(spec=WalletReputation)
        mock_rep.wallet_address = wallet_address
        mock_rep.reputation_score = 65

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_rep
        repo.session.execute.return_value = mock_result

        rep = await repo.get_reputation(wallet_address)

        assert rep is not None
        assert rep.reputation_score == 65

    @pytest.mark.asyncio
    async def test_get_reputation_not_found(self, repo: WalletReputationRepository):
        """Test retrieving non-existent reputation returns None."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        repo.session.execute.return_value = mock_result

        rep = await repo.get_reputation("NonExistent" + "x" * 33)

        assert rep is None

    @pytest.mark.asyncio
    async def test_find_by_score_range(self, repo: WalletReputationRepository):
        """Test finding wallets within score range."""
        mock_reps = [MagicMock(spec=WalletReputation) for _ in range(5)]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_reps
        repo.session.execute.return_value = mock_result

        reps = await repo.find_by_score_range(min_score=40, max_score=60)

        assert len(reps) == 5

    @pytest.mark.asyncio
    async def test_find_high_risk(self, repo: WalletReputationRepository):
        """Test finding high-risk wallets."""
        mock_reps = [MagicMock(spec=WalletReputation) for _ in range(3)]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_reps
        repo.session.execute.return_value = mock_result

        reps = await repo.find_high_risk(max_score=30)

        assert len(reps) == 3

    @pytest.mark.asyncio
    async def test_find_serial_snipers(self, repo: WalletReputationRepository):
        """Test finding wallets with multiple sniper classifications."""
        mock_reps = [MagicMock(spec=WalletReputation) for _ in range(2)]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_reps
        repo.session.execute.return_value = mock_result

        reps = await repo.find_serial_snipers(min_sniper_count=3)

        assert len(reps) == 2

    @pytest.mark.asyncio
    async def test_get_summary(self, repo: WalletReputationRepository):
        """Test getting reputation summary."""
        wallet_address = "SummaryWallet111111111111111111111111111"

        mock_rep = MagicMock(spec=WalletReputation)
        mock_rep.wallet_address = wallet_address
        mock_rep.reputation_score = 70
        mock_rep.confidence_level = ConfidenceLevel.MEDIUM
        mock_rep.tokens_analyzed = 15
        mock_rep.patterns = [{"type": "DIAMOND_HANDS", "confidence": 0.8}]
        mock_rep.is_known_bad_actor = False
        mock_rep.is_known_good_actor = False
        mock_rep.entity_id = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_rep
        repo.session.execute.return_value = mock_result

        summary = await repo.get_summary(wallet_address)

        assert summary is not None
        assert isinstance(summary, ReputationSummary)
        assert summary.reputation_score == 70

    @pytest.mark.asyncio
    async def test_get_reputations_for_wallets(self, repo: WalletReputationRepository):
        """Test getting reputations for multiple wallets."""
        wallets = ["W1" + "x" * 41, "W2" + "x" * 41]

        mock_rep1 = MagicMock(spec=WalletReputation)
        mock_rep1.wallet_address = wallets[0]
        mock_rep2 = MagicMock(spec=WalletReputation)
        mock_rep2.wallet_address = wallets[1]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_rep1, mock_rep2]
        repo.session.execute.return_value = mock_result

        reps = await repo.get_reputations_for_wallets(wallets)

        assert len(reps) == 2
        assert wallets[0] in reps


# =============================================================================
# Model Tests
# =============================================================================


class TestEntityTypeConstants:
    """Tests for EntityType constants."""

    def test_entity_types_defined(self):
        """Test all entity types are defined."""
        assert EntityType.SYBIL_CLUSTER == "sybil_cluster"
        assert EntityType.WHALE_GROUP == "whale_group"
        assert EntityType.EXCHANGE == "exchange"
        assert EntityType.MARKET_MAKER == "market_maker"
        assert EntityType.UNKNOWN == "unknown"


class TestDetectionMethodConstants:
    """Tests for DetectionMethod constants."""

    def test_detection_methods_defined(self):
        """Test all detection methods are defined."""
        assert DetectionMethod.SHARED_FUNDER == "shared_funder"
        assert DetectionMethod.TEMPORAL_SYNC == "temporal_sync"
        assert DetectionMethod.BEHAVIOR_SIMILARITY == "behavior_similarity"
        assert DetectionMethod.MANUAL == "manual"


class TestConfidenceLevelConstants:
    """Tests for ConfidenceLevel constants."""

    def test_confidence_levels_defined(self):
        """Test all confidence levels are defined."""
        assert ConfidenceLevel.LOW == "low"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.HIGH == "high"
