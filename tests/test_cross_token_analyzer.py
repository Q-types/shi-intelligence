"""Tests for CrossTokenAnalyzer - integration tests for the full intelligence pipeline."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.intelligence.analyzer import (
    CrossTokenAnalyzer,
    WalletIntelligence,
    EntityIntelligence,
    IntelligenceReport,
)
from src.data.models import Entity, WalletReputation, EntityType, ConfidenceLevel


class TestCrossTokenAnalyzer:
    """Tests for CrossTokenAnalyzer class."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        return AsyncMock()

    @pytest.fixture
    def analyzer(self, mock_session: AsyncMock) -> CrossTokenAnalyzer:
        """Create analyzer with mocked dependencies."""
        analyzer = CrossTokenAnalyzer(mock_session)

        # Mock all repositories
        analyzer.entity_repo = AsyncMock()
        analyzer.history_repo = AsyncMock()
        analyzer.reputation_repo = AsyncMock()

        # Mock detection services
        analyzer.shared_funder_detector = MagicMock()
        analyzer.temporal_sync_detector = MagicMock()
        analyzer.entity_resolver = AsyncMock()
        analyzer.sybil_detector = AsyncMock()

        # Mock reputation services
        analyzer.pattern_detector = MagicMock()
        analyzer.reputation_scorer = AsyncMock()

        return analyzer

    # -------------------------------------------------------------------------
    # Full Analysis Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_analyze_token_holders_basic(
        self,
        analyzer: CrossTokenAnalyzer,
        sample_wallets: list[str],
    ):
        """Test basic token holder analysis."""
        # Setup mocks
        analyzer.entity_resolver.resolve.return_value = MagicMock(
            entities_created=0,
            entities_updated=0,
        )
        analyzer.entity_repo.get_entities_for_wallets.return_value = {}
        analyzer.entity_repo.get_entity_for_wallet.return_value = None
        analyzer.sybil_detector.assess_and_update.return_value = None

        # Create mock reputations
        mock_reps = []
        mock_reps_dict = {}
        for wallet in sample_wallets:
            rep = MagicMock(spec=WalletReputation)
            rep.wallet_address = wallet
            rep.reputation_score = 60
            rep.confidence_level = "medium"
            rep.patterns = []
            rep.tokens_analyzed = 10
            mock_reps.append(rep)
            mock_reps_dict[wallet] = rep

        analyzer.reputation_scorer.score_and_persist_batch.return_value = mock_reps

        # Mock reputation_repo.get_reputation to return correct rep for each wallet
        async def mock_get_rep(wallet):
            return mock_reps_dict.get(wallet)
        analyzer.reputation_repo.get_reputation = mock_get_rep

        # Run analysis
        report = await analyzer.analyze_token_holders(
            wallet_addresses=sample_wallets,
            token_mint="TestToken11111111111111111111111111111",
        )

        # Verify result structure
        assert isinstance(report, IntelligenceReport)
        assert report.wallets_analyzed == len(sample_wallets)
        assert report.analyzed_at is not None
        assert report.analysis_id.startswith("cta-")

    @pytest.mark.asyncio
    async def test_analyze_with_funding_graph(
        self,
        analyzer: CrossTokenAnalyzer,
        sample_wallets: list[str],
        mock_funding_graph,
    ):
        """Test analysis with funding graph provided."""
        # Setup
        analyzer.entity_resolver.resolve.return_value = MagicMock()
        analyzer.entity_repo.get_entities_for_wallets.return_value = {}
        analyzer.entity_repo.get_entity_for_wallet.return_value = None

        mock_reps = []
        mock_reps_dict = {}
        for wallet in sample_wallets:
            rep = MagicMock(spec=WalletReputation)
            rep.wallet_address = wallet
            rep.reputation_score = 50
            rep.confidence_level = "medium"
            rep.patterns = []
            rep.tokens_analyzed = 10
            mock_reps.append(rep)
            mock_reps_dict[wallet] = rep

        analyzer.reputation_scorer.score_and_persist_batch.return_value = mock_reps

        async def mock_get_rep(wallet):
            return mock_reps_dict.get(wallet)
        analyzer.reputation_repo.get_reputation = mock_get_rep

        # Run with funding graph
        report = await analyzer.analyze_token_holders(
            wallet_addresses=sample_wallets,
            funding_graph=mock_funding_graph,
        )

        # Verify entity resolver was called with graph
        analyzer.entity_resolver.resolve.assert_called_once()
        call_kwargs = analyzer.entity_resolver.resolve.call_args.kwargs
        assert call_kwargs.get("funding_graph") == mock_funding_graph

    @pytest.mark.asyncio
    async def test_analyze_detects_sybil_networks(
        self,
        analyzer: CrossTokenAnalyzer,
        sybil_cluster_wallets: list[str],
    ):
        """Test that analysis detects sybil networks."""
        # Create mock entity for sybil cluster
        mock_entity = MagicMock(spec=Entity)
        mock_entity.id = 1
        mock_entity.entity_type = EntityType.SYBIL_CLUSTER
        mock_entity.is_professional_sybil = True
        mock_entity.wallet_count = 5
        mock_entity.tokens_targeted = 3
        mock_entity.avg_coordination_score = 0.9
        mock_entity.risk_level = "critical"
        mock_entity.dominant_funder_address = "Funder" + "x" * 38

        # Setup mocks
        analyzer.entity_resolver.resolve.return_value = MagicMock()
        analyzer.entity_repo.get_entities_for_wallets.return_value = {
            wallet: mock_entity for wallet in sybil_cluster_wallets
        }
        analyzer.entity_repo.get_entity_for_wallet.return_value = mock_entity

        # Mock get_entity to return proper entity
        async def mock_get_entity(entity_id):
            if entity_id == 1:
                return mock_entity
            return None
        analyzer.entity_repo.get_entity = mock_get_entity

        # Mock get_entity_wallets
        async def mock_get_entity_wallets(entity_id):
            if entity_id == 1:
                return sybil_cluster_wallets
            return []
        analyzer.entity_repo.get_entity_wallets = mock_get_entity_wallets

        mock_sybil_assessment = MagicMock()
        mock_sybil_assessment.is_professional_sybil = True
        mock_sybil_assessment.risk_level = "critical"
        analyzer.sybil_detector.assess_and_update.return_value = mock_sybil_assessment

        mock_reps = []
        mock_reps_dict = {}
        for wallet in sybil_cluster_wallets:
            rep = MagicMock(spec=WalletReputation)
            rep.wallet_address = wallet
            rep.reputation_score = 25
            rep.confidence_level = "medium"
            rep.patterns = []
            rep.tokens_analyzed = 10
            mock_reps.append(rep)
            mock_reps_dict[wallet] = rep
        analyzer.reputation_scorer.score_and_persist_batch.return_value = mock_reps

        async def mock_get_rep(wallet):
            return mock_reps_dict.get(wallet)
        analyzer.reputation_repo.get_reputation = mock_get_rep

        # Run analysis
        report = await analyzer.analyze_token_holders(
            wallet_addresses=sybil_cluster_wallets,
        )

        assert report.sybil_networks_found >= 1
        assert report.sybil_wallets_count > 0

    @pytest.mark.asyncio
    async def test_analyze_returns_high_risk_wallets(
        self,
        analyzer: CrossTokenAnalyzer,
        sample_wallets: list[str],
    ):
        """Test that analysis identifies high-risk wallets."""
        # Setup - create mix of risk levels
        analyzer.entity_resolver.resolve.return_value = MagicMock()
        analyzer.entity_repo.get_entities_for_wallets.return_value = {}

        # First 2 wallets high risk (score < 35), rest normal
        mock_reps = []
        for i, wallet in enumerate(sample_wallets):
            rep = MagicMock(spec=WalletReputation)
            rep.wallet_address = wallet
            rep.reputation_score = 20 if i < 2 else 60
            rep.patterns = []
            rep.confidence_level = "medium"
            rep.tokens_analyzed = 10
            mock_reps.append(rep)

        analyzer.reputation_scorer.score_and_persist_batch.return_value = mock_reps

        # Mock get_wallet_intelligence for high risk wallets
        async def mock_get_intel(wallet):
            rep = next((r for r in mock_reps if r.wallet_address == wallet), None)
            if not rep:
                return None
            return WalletIntelligence(
                wallet_address=wallet,
                reputation_score=rep.reputation_score,
                risk_level="high" if rep.reputation_score < 35 else "low",
                confidence_level="medium",
                patterns=[],
                entity_id=None,
                entity_type=None,
                is_sybil_member=False,
                tokens_analyzed=10,
            )

        analyzer.get_wallet_intelligence = mock_get_intel

        # Run analysis
        report = await analyzer.analyze_token_holders(
            wallet_addresses=sample_wallets,
        )

        # Risk distribution should be calculated
        assert "high" in report.risk_distribution or "critical" in report.risk_distribution

    @pytest.mark.asyncio
    async def test_analyze_generates_recommendations(
        self,
        analyzer: CrossTokenAnalyzer,
        sample_wallets: list[str],
    ):
        """Test that analysis generates recommendations."""
        # Setup
        analyzer.entity_resolver.resolve.return_value = MagicMock()
        analyzer.entity_repo.get_entities_for_wallets.return_value = {}
        analyzer.entity_repo.get_entity_for_wallet.return_value = None

        mock_reps = []
        mock_reps_dict = {}
        for wallet in sample_wallets:
            rep = MagicMock(spec=WalletReputation)
            rep.wallet_address = wallet
            rep.reputation_score = 60
            rep.confidence_level = "medium"
            rep.patterns = []
            rep.tokens_analyzed = 10
            mock_reps.append(rep)
            mock_reps_dict[wallet] = rep

        analyzer.reputation_scorer.score_and_persist_batch.return_value = mock_reps

        async def mock_get_rep(wallet):
            return mock_reps_dict.get(wallet)
        analyzer.reputation_repo.get_reputation = mock_get_rep

        # Run analysis
        report = await analyzer.analyze_token_holders(
            wallet_addresses=sample_wallets,
        )

        assert len(report.recommendations) > 0

    # -------------------------------------------------------------------------
    # Single Wallet Intelligence Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_wallet_intelligence_existing(
        self,
        analyzer: CrossTokenAnalyzer,
    ):
        """Test getting intelligence for wallet with existing reputation."""
        wallet_address = "ExistingWallet1111111111111111111111111"

        # Mock existing reputation
        mock_rep = MagicMock(spec=WalletReputation)
        mock_rep.wallet_address = wallet_address
        mock_rep.reputation_score = 75
        mock_rep.confidence_level = "high"
        mock_rep.patterns = [{"type": "DIAMOND_HANDS", "confidence": 0.8}]
        mock_rep.tokens_analyzed = 25

        analyzer.reputation_repo.get_reputation.return_value = mock_rep
        analyzer.entity_repo.get_entity_for_wallet.return_value = None

        intel = await analyzer.get_wallet_intelligence(wallet_address)

        assert intel is not None
        assert intel.wallet_address == wallet_address
        assert intel.reputation_score == 75
        assert intel.risk_level == "low"  # Score > 50
        assert intel.confidence_level == "high"

    @pytest.mark.asyncio
    async def test_get_wallet_intelligence_new_wallet(
        self,
        analyzer: CrossTokenAnalyzer,
    ):
        """Test getting intelligence for new wallet triggers scoring."""
        wallet_address = "NewWallet111111111111111111111111111111"

        # First call returns None (no existing reputation)
        # After scoring, second call returns the new reputation
        mock_new_rep = MagicMock(spec=WalletReputation)
        mock_new_rep.wallet_address = wallet_address
        mock_new_rep.reputation_score = 50
        mock_new_rep.confidence_level = "low"
        mock_new_rep.patterns = []
        mock_new_rep.tokens_analyzed = 0

        analyzer.reputation_repo.get_reputation.side_effect = [None, mock_new_rep]
        analyzer.entity_repo.get_entity_for_wallet.return_value = None

        intel = await analyzer.get_wallet_intelligence(wallet_address)

        # Should have triggered scoring
        analyzer.reputation_scorer.score_and_persist.assert_called_once()
        assert intel is not None
        assert intel.reputation_score == 50

    @pytest.mark.asyncio
    async def test_get_wallet_intelligence_with_entity(
        self,
        analyzer: CrossTokenAnalyzer,
    ):
        """Test getting intelligence for wallet that belongs to entity."""
        wallet_address = "EntityMember111111111111111111111111111"

        # Mock entity
        mock_entity = MagicMock(spec=Entity)
        mock_entity.id = 1
        mock_entity.entity_type = EntityType.SYBIL_CLUSTER
        mock_entity.is_professional_sybil = True

        # Mock reputation
        mock_rep = MagicMock(spec=WalletReputation)
        mock_rep.wallet_address = wallet_address
        mock_rep.reputation_score = 30
        mock_rep.confidence_level = "medium"
        mock_rep.patterns = []
        mock_rep.tokens_analyzed = 10

        analyzer.reputation_repo.get_reputation.return_value = mock_rep
        analyzer.entity_repo.get_entity_for_wallet.return_value = mock_entity

        intel = await analyzer.get_wallet_intelligence(wallet_address)

        assert intel is not None
        assert intel.entity_id == 1
        assert intel.entity_type == EntityType.SYBIL_CLUSTER
        assert intel.is_sybil_member is True

    # -------------------------------------------------------------------------
    # Entity Intelligence Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_entity_intelligence(
        self,
        analyzer: CrossTokenAnalyzer,
    ):
        """Test getting intelligence for an entity."""
        # Mock entity
        mock_entity = MagicMock(spec=Entity)
        mock_entity.id = 1
        mock_entity.entity_type = EntityType.SYBIL_CLUSTER
        mock_entity.wallet_count = 5
        mock_entity.tokens_targeted = 10
        mock_entity.avg_coordination_score = 0.85
        mock_entity.is_professional_sybil = True
        mock_entity.risk_level = "critical"
        mock_entity.dominant_funder_address = "Funder" + "x" * 38

        analyzer.entity_repo.get_entity.return_value = mock_entity
        analyzer.entity_repo.get_entity_wallets.return_value = [
            f"Member{i}" + "x" * 37 for i in range(5)
        ]

        intel = await analyzer.get_entity_intelligence(1)

        assert intel is not None
        assert intel.entity_id == 1
        assert intel.entity_type == EntityType.SYBIL_CLUSTER
        assert intel.wallet_count == 5
        assert intel.is_professional_sybil is True
        assert len(intel.wallet_addresses) == 5

    @pytest.mark.asyncio
    async def test_get_entity_intelligence_not_found(
        self,
        analyzer: CrossTokenAnalyzer,
    ):
        """Test getting intelligence for non-existent entity."""
        analyzer.entity_repo.get_entity.return_value = None

        intel = await analyzer.get_entity_intelligence(999)

        assert intel is None

    # -------------------------------------------------------------------------
    # Recording Interactions Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_record_wallet_interaction(
        self,
        analyzer: CrossTokenAnalyzer,
    ):
        """Test recording a wallet interaction."""
        await analyzer.record_wallet_interaction(
            wallet_address="Wallet" + "x" * 37,
            token_mint="Token" + "x" * 38,
            first_seen_at=datetime.now(timezone.utc),
            archetype="sniper",
            archetype_confidence=0.9,
        )

        analyzer.history_repo.record_interaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_interactions_batch(
        self,
        analyzer: CrossTokenAnalyzer,
    ):
        """Test recording multiple interactions."""
        interactions = [
            {
                "wallet_address": f"Wallet{i}" + "x" * 36,
                "token_mint": "Token" + "x" * 38,
                "first_seen_at": datetime.now(timezone.utc),
            }
            for i in range(5)
        ]

        count = await analyzer.record_interactions_batch(interactions)

        assert count == 5
        assert analyzer.history_repo.record_interaction.call_count == 5

    # -------------------------------------------------------------------------
    # Risk Level Tests
    # -------------------------------------------------------------------------

    def test_score_to_risk_critical(self, analyzer: CrossTokenAnalyzer):
        """Test critical risk for very low scores."""
        assert analyzer._score_to_risk(0) == "critical"
        assert analyzer._score_to_risk(20) == "critical"

    def test_score_to_risk_high(self, analyzer: CrossTokenAnalyzer):
        """Test high risk for low scores."""
        assert analyzer._score_to_risk(21) == "high"
        assert analyzer._score_to_risk(35) == "high"

    def test_score_to_risk_medium(self, analyzer: CrossTokenAnalyzer):
        """Test medium risk for moderate scores."""
        assert analyzer._score_to_risk(36) == "medium"
        assert analyzer._score_to_risk(50) == "medium"

    def test_score_to_risk_low(self, analyzer: CrossTokenAnalyzer):
        """Test low risk for good scores."""
        assert analyzer._score_to_risk(51) == "low"
        assert analyzer._score_to_risk(100) == "low"

    # -------------------------------------------------------------------------
    # Recommendation Tests
    # -------------------------------------------------------------------------

    def test_generate_recommendations_sybil_alert(
        self,
        analyzer: CrossTokenAnalyzer,
    ):
        """Test sybil alert recommendation is generated."""
        entities = [
            EntityIntelligence(
                entity_id=1,
                entity_type=EntityType.SYBIL_CLUSTER,
                wallet_count=10,
                tokens_targeted=5,
                coordination_score=0.9,
                is_professional_sybil=True,
                risk_level="critical",
                dominant_funder="Funder" + "x" * 38,
                wallet_addresses=[],
            )
        ]

        recommendations = analyzer._generate_recommendations(entities, [], {})

        assert any("SYBIL" in r for r in recommendations)

    def test_generate_recommendations_critical_wallets(
        self,
        analyzer: CrossTokenAnalyzer,
    ):
        """Test critical wallet recommendation is generated."""
        high_risk = [
            WalletIntelligence(
                wallet_address=f"Critical{i}" + "x" * 36,
                reputation_score=15,
                risk_level="critical",
                confidence_level="high",
                patterns=[],
                entity_id=None,
                entity_type=None,
                is_sybil_member=False,
                tokens_analyzed=20,
            )
            for i in range(3)
        ]

        recommendations = analyzer._generate_recommendations([], high_risk, {})

        assert any("CRITICAL" in r for r in recommendations)

    def test_generate_recommendations_concentration(
        self,
        analyzer: CrossTokenAnalyzer,
    ):
        """Test concentration warning recommendation."""
        large_entities = [
            EntityIntelligence(
                entity_id=1,
                entity_type=EntityType.WHALE_GROUP,
                wallet_count=15,  # > 10 wallets
                tokens_targeted=3,
                coordination_score=0.7,
                is_professional_sybil=False,
                risk_level="medium",
                dominant_funder=None,
                wallet_addresses=[],
            )
        ]

        recommendations = analyzer._generate_recommendations(large_entities, [], {})

        assert any("CONCENTRATION" in r for r in recommendations)

    def test_generate_recommendations_no_issues(
        self,
        analyzer: CrossTokenAnalyzer,
    ):
        """Test default recommendation when no issues found."""
        recommendations = analyzer._generate_recommendations([], [], {})

        assert len(recommendations) == 1
        assert "No critical issues" in recommendations[0]


class TestIntelligenceDataclasses:
    """Tests for intelligence dataclasses."""

    def test_wallet_intelligence_creation(self):
        """Test creating WalletIntelligence."""
        intel = WalletIntelligence(
            wallet_address="Wallet" + "x" * 37,
            reputation_score=65,
            risk_level="low",
            confidence_level="medium",
            patterns=[{"type": "PROFIT_TAKER", "confidence": 0.7}],
            entity_id=1,
            entity_type="sybil_cluster",
            is_sybil_member=True,
            tokens_analyzed=15,
        )

        assert intel.reputation_score == 65
        assert intel.is_sybil_member is True

    def test_entity_intelligence_creation(self):
        """Test creating EntityIntelligence."""
        intel = EntityIntelligence(
            entity_id=1,
            entity_type="sybil_cluster",
            wallet_count=5,
            tokens_targeted=10,
            coordination_score=0.85,
            is_professional_sybil=True,
            risk_level="critical",
            dominant_funder="Funder" + "x" * 38,
            wallet_addresses=["W1" + "x" * 41, "W2" + "x" * 41],
        )

        assert intel.wallet_count == 5
        assert intel.is_professional_sybil is True

    def test_intelligence_report_creation(self):
        """Test creating IntelligenceReport."""
        report = IntelligenceReport(
            analysis_id="cta-20260512120000",
            token_mint="Token" + "x" * 38,
            analyzed_at=datetime.now(timezone.utc),
            wallets_analyzed=100,
            entities_found=5,
            entities=[],
            sybil_networks_found=2,
            sybil_wallets_count=15,
            high_risk_wallets=[],
            avg_reputation_score=55.0,
            risk_distribution={"low": 60, "medium": 30, "high": 8, "critical": 2},
            recommendations=["Test recommendation"],
        )

        assert report.wallets_analyzed == 100
        assert report.sybil_networks_found == 2
        assert sum(report.risk_distribution.values()) == 100
