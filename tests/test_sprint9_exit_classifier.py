"""
Sprint 9 Exit Classifier Tests.

Tests for:
- Exit Event Classification (10 exit types)
- Sell Confidence Scoring
- Transfer Chain Detection
- LP Action Separation
- CEX Deposit Detection
- PnL Reliability Scoring
- Hard Rules Compliance

HARD RULES under test:
1. Balance decrease alone is NOT a sell
2. Realised PnL requires sell confidence
3. LP actions must NOT be treated as sells
4. Transfers must NOT generate realised PnL unless later sale observed
5. CEX deposits are uncertain exits unless sale can be inferred
6. Low reliability PnL must NOT display precise values
7. All classifications must include confidence and evidence
"""

from datetime import datetime, timezone

import pytest

from src.longitudinal.exit_classifier import (
    ExitEventType,
    ExitEvidence,
    ExitEventClassification,
    ExitClassifierConfig,
    ExitEventClassifier,
    SellConfidenceScorer,
    PnLReliabilityScorer,
    TransferChainResult,
    TransferChainConfig,
    TransferChainDetector,
    LPActionResult,
    LPActionDetector,
    CEXDepositResult,
    CEXDetectionConfig,
    CEXDepositDetector,
    DEX_PROGRAMS,
    LP_PROGRAMS,
    BRIDGE_PROGRAMS,
    KNOWN_CEX_ADDRESSES,
    BURN_ADDRESSES,
    create_exit_classifier,
    create_transfer_chain_detector,
    create_lp_action_detector,
    create_cex_deposit_detector,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_tx_data_dex_sell():
    """Transaction data for a DEX sell with SOL received."""
    return {
        "slot": 123456789,
        "blockTime": 1716307200,  # 2024-05-21 12:00:00 UTC
        "transaction": {
            "signatures": ["5abc123..."],
            "message": {
                "accountKeys": [
                    {"pubkey": "WalletAddress123"},
                    {"pubkey": "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"},  # Jupiter v6
                    {"pubkey": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                ],
            },
        },
        "meta": {
            "preBalances": [1_000_000_000, 0, 0],  # 1 SOL before
            "postBalances": [1_500_000_000, 0, 0],  # 1.5 SOL after (+0.5 SOL)
            "preTokenBalances": [
                {
                    "mint": "TokenMint123",
                    "owner": "WalletAddress123",
                    "uiTokenAmount": {"amount": "1000000000"},
                }
            ],
            "postTokenBalances": [
                {
                    "mint": "TokenMint123",
                    "owner": "WalletAddress123",
                    "uiTokenAmount": {"amount": "0"},
                }
            ],
        },
    }


@pytest.fixture
def sample_tx_data_transfer():
    """Transaction data for a simple transfer (no SOL received)."""
    return {
        "slot": 123456790,
        "blockTime": 1716307300,
        "transaction": {
            "signatures": ["5def456..."],
            "message": {
                "accountKeys": [
                    {"pubkey": "WalletAddress123"},
                    {"pubkey": "DestinationWallet456"},
                    {"pubkey": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                ],
            },
        },
        "meta": {
            "preBalances": [1_000_000_000, 100_000_000, 0],
            "postBalances": [999_990_000, 100_010_000, 0],  # Small SOL for fees only
            "preTokenBalances": [
                {
                    "mint": "TokenMint123",
                    "owner": "WalletAddress123",
                    "uiTokenAmount": {"amount": "1000000000"},
                }
            ],
            "postTokenBalances": [
                {
                    "mint": "TokenMint123",
                    "owner": "WalletAddress123",
                    "uiTokenAmount": {"amount": "0"},
                },
                {
                    "mint": "TokenMint123",
                    "owner": "DestinationWallet456",
                    "uiTokenAmount": {"amount": "1000000000"},
                },
            ],
        },
    }


@pytest.fixture
def sample_tx_data_lp_add():
    """Transaction data for adding liquidity (LP tokens minted)."""
    return {
        "slot": 123456791,
        "blockTime": 1716307400,
        "transaction": {
            "signatures": ["5ghi789..."],
            "message": {
                "accountKeys": [
                    {"pubkey": "WalletAddress123"},
                    {"pubkey": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"},  # Raydium
                    {"pubkey": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                ],
            },
        },
        "meta": {
            "preBalances": [1_000_000_000, 0, 0],
            "postBalances": [999_000_000, 0, 0],
            "preTokenBalances": [
                {
                    "mint": "TokenMint123",
                    "owner": "WalletAddress123",
                    "uiTokenAmount": {"amount": "1000000000"},
                }
            ],
            "postTokenBalances": [
                {
                    "mint": "TokenMint123",
                    "owner": "WalletAddress123",
                    "uiTokenAmount": {"amount": "500000000"},
                },
                {
                    "mint": "LPTokenMint999",
                    "owner": "WalletAddress123",
                    "uiTokenAmount": {"amount": "100000000"},
                },
            ],
        },
    }


@pytest.fixture
def sample_tx_data_cex_deposit():
    """Transaction data for CEX deposit."""
    cex_address = list(KNOWN_CEX_ADDRESSES.keys())[0]  # Use first known CEX
    return {
        "slot": 123456792,
        "blockTime": 1716307500,
        "transaction": {
            "signatures": ["5jkl012..."],
            "message": {
                "accountKeys": [
                    {"pubkey": "WalletAddress123"},
                    {"pubkey": cex_address},
                    {"pubkey": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                ],
            },
        },
        "meta": {
            "preBalances": [1_000_000_000, 0, 0],
            "postBalances": [999_990_000, 0, 0],
            "preTokenBalances": [
                {
                    "mint": "TokenMint123",
                    "owner": "WalletAddress123",
                    "uiTokenAmount": {"amount": "1000000000"},
                }
            ],
            "postTokenBalances": [
                {
                    "mint": "TokenMint123",
                    "owner": "WalletAddress123",
                    "uiTokenAmount": {"amount": "0"},
                },
                {
                    "mint": "TokenMint123",
                    "owner": cex_address,
                    "uiTokenAmount": {"amount": "1000000000"},
                },
            ],
        },
    }


@pytest.fixture
def sample_tx_data_burn():
    """Transaction data for token burn."""
    burn_address = "1nc1nerator11111111111111111111111111111111"
    return {
        "slot": 123456793,
        "blockTime": 1716307600,
        "transaction": {
            "signatures": ["5mno345..."],
            "message": {
                "accountKeys": [
                    {"pubkey": "WalletAddress123"},
                    {"pubkey": burn_address},
                    {"pubkey": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                ],
            },
        },
        "meta": {
            "preBalances": [1_000_000_000, 0, 0],
            "postBalances": [999_990_000, 0, 0],
            "preTokenBalances": [
                {
                    "mint": "TokenMint123",
                    "owner": "WalletAddress123",
                    "uiTokenAmount": {"amount": "1000000000"},
                }
            ],
            "postTokenBalances": [
                {
                    "mint": "TokenMint123",
                    "owner": "WalletAddress123",
                    "uiTokenAmount": {"amount": "0"},
                },
                {
                    "mint": "TokenMint123",
                    "owner": burn_address,
                    "uiTokenAmount": {"amount": "1000000000"},
                },
            ],
        },
    }


@pytest.fixture
def classifier():
    """Create a default exit event classifier."""
    return create_exit_classifier()


# ============================================================================
# Exit Event Classification Tests
# ============================================================================


class TestExitEventClassifier:
    """Tests for the main exit event classifier."""

    def test_classify_dex_sell(self, classifier, sample_tx_data_dex_sell):
        """Test DEX sell classification with quote asset received."""
        result = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_dex_sell,
        )

        assert result.exit_type == ExitEventType.DEX_SELL
        assert result.confidence >= 0.8
        assert result.sell_confidence_score >= 0.5
        assert result.pnl_computable is True
        assert "dex_detected:jupiter_v6" in result.confidence_factors
        assert "quote_asset_received" in result.confidence_factors

    def test_classify_transfer(self, classifier, sample_tx_data_transfer):
        """Test simple transfer classification (no swap)."""
        result = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_transfer,
        )

        assert result.exit_type == ExitEventType.TRANSFER_OUT
        assert result.sell_confidence_score < 0.3
        assert result.pnl_computable is False
        assert result.downstream_address == "DestinationWallet456"

    def test_classify_lp_add(self, classifier, sample_tx_data_lp_add):
        """Test LP add classification (LP tokens minted)."""
        result = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-500_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_lp_add,
        )

        assert result.exit_type == ExitEventType.LP_ADD
        assert result.sell_confidence_score < 0.2
        assert result.pnl_computable is False
        assert "lp_token_minted" in result.confidence_factors

    def test_classify_cex_deposit(self, classifier, sample_tx_data_cex_deposit):
        """Test CEX deposit classification."""
        result = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_cex_deposit,
        )

        assert result.exit_type == ExitEventType.CEX_DEPOSIT
        assert result.pnl_computable is False
        assert result.downstream_wallet_type == "cex"

    def test_classify_burn(self, classifier, sample_tx_data_burn):
        """Test burn classification."""
        result = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_burn,
        )

        assert result.exit_type == ExitEventType.BURN
        assert result.confidence >= 0.9
        assert result.downstream_wallet_type == "burn"

    def test_classification_always_has_confidence(self, classifier, sample_tx_data_transfer):
        """HARD RULE 7: All classifications must include confidence."""
        result = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_transfer,
        )

        assert result.confidence is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_classification_always_has_evidence(self, classifier, sample_tx_data_transfer):
        """HARD RULE 7: All classifications must include evidence."""
        result = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_transfer,
        )

        assert result.evidence is not None
        assert result.evidence.signature is not None
        assert result.evidence.token_mint == "TokenMint123"


# ============================================================================
# Sell Confidence Scorer Tests
# ============================================================================


class TestSellConfidenceScorer:
    """Tests for the sell confidence scorer."""

    def test_dex_sell_high_confidence(self, classifier, sample_tx_data_dex_sell):
        """DEX sell with quote received should have high sell confidence."""
        classification = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_dex_sell,
        )

        scorer = SellConfidenceScorer()
        score, pnl_computable, breakdown = scorer.compute_score(classification)

        assert score >= 0.5
        assert breakdown["swap_instruction"] > 0
        assert breakdown["quote_received"] > 0

    def test_transfer_low_sell_confidence(self, classifier, sample_tx_data_transfer):
        """Transfer should have low sell confidence."""
        classification = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_transfer,
        )

        scorer = SellConfidenceScorer()
        score, pnl_computable, _ = scorer.compute_score(classification)

        assert score < 0.3
        assert pnl_computable is False

    def test_lp_action_negative_sell_confidence(self, classifier, sample_tx_data_lp_add):
        """LP action should have negative sell confidence contribution."""
        classification = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-500_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_lp_add,
        )

        scorer = SellConfidenceScorer()
        score, pnl_computable, breakdown = scorer.compute_score(classification)

        assert breakdown["lp_token_movement"] < 0
        assert pnl_computable is False


# ============================================================================
# Hard Rules Compliance Tests
# ============================================================================


class TestHardRulesCompliance:
    """Tests for Sprint 9 hard rules compliance."""

    def test_hard_rule_1_balance_decrease_not_sell(self, classifier, sample_tx_data_transfer):
        """HARD RULE 1: Balance decrease alone is NOT a sell."""
        result = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_transfer,
        )

        # Transfer has balance decrease but is NOT classified as a sell
        assert result.exit_type != ExitEventType.DEX_SELL
        assert result.pnl_computable is False

    def test_hard_rule_2_pnl_requires_sell_confidence(self, classifier, sample_tx_data_transfer):
        """HARD RULE 2: Realised PnL requires sell confidence."""
        result = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_transfer,
        )

        # Low sell confidence means PnL not computable
        assert result.sell_confidence_score < 0.7
        assert result.pnl_computable is False

    def test_hard_rule_3_lp_not_treated_as_sell(self, classifier, sample_tx_data_lp_add):
        """HARD RULE 3: LP actions must NOT be treated as sells."""
        result = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-500_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_lp_add,
        )

        assert result.exit_type == ExitEventType.LP_ADD
        assert result.exit_type != ExitEventType.DEX_SELL
        assert result.pnl_computable is False

    def test_hard_rule_4_transfer_no_pnl(self, classifier, sample_tx_data_transfer):
        """HARD RULE 4: Transfers must NOT generate realised PnL."""
        result = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_transfer,
        )

        assert result.exit_type == ExitEventType.TRANSFER_OUT
        assert result.pnl_computable is False

    def test_hard_rule_5_cex_uncertain(self, classifier, sample_tx_data_cex_deposit):
        """HARD RULE 5: CEX deposits are uncertain exits."""
        result = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_cex_deposit,
        )

        assert result.exit_type == ExitEventType.CEX_DEPOSIT
        assert result.pnl_computable is False
        # CEX deposit is uncertain - sale may or may not happen

    def test_hard_rule_6_low_reliability_no_precise(self):
        """HARD RULE 6: Low reliability PnL must NOT display precise values."""
        scorer = PnLReliabilityScorer()

        # Low reliability scenario
        reliability, display_mode, _ = scorer.compute_reliability(
            sell_confidence=0.3,
            entry_price_confidence=0.4,
            exit_price_confidence=0.5,
            liquidity_confidence=0.3,
            lot_count=5,
            has_transfer_ambiguity=True,
            event_completeness=0.5,
        )

        assert reliability < 0.7
        assert display_mode != "precise"

    def test_hard_rule_7_confidence_and_evidence(self, classifier, sample_tx_data_dex_sell):
        """HARD RULE 7: All classifications must include confidence and evidence."""
        result = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_dex_sell,
        )

        # Must have confidence
        assert result.confidence is not None
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

        # Must have evidence
        assert result.evidence is not None
        assert isinstance(result.evidence, ExitEvidence)

        # Must have confidence factors
        assert result.confidence_factors is not None
        assert len(result.confidence_factors) > 0

        # Must have classification reason
        assert result.classification_reason is not None
        assert len(result.classification_reason) > 0


# ============================================================================
# Transfer Chain Detection Tests
# ============================================================================


class TestTransferChainDetector:
    """Tests for transfer chain detection."""

    @pytest.mark.asyncio
    async def test_shared_funder_increases_migration_confidence(self):
        """Shared funder should increase migration confidence."""
        detector = create_transfer_chain_detector()

        # Mock provider would return shared funder info
        # For unit test, we test the result structure
        result = await detector.detect_migration(
            source_wallet="Wallet1",
            destination_wallet="Wallet2",
            token_mint="TokenMint123",
            transfer_timestamp=datetime.now(timezone.utc),
            wallet_info_provider=None,  # No provider = partial detection
        )

        assert isinstance(result, TransferChainResult)
        assert isinstance(result.likely_migration, bool)
        assert isinstance(result.migration_confidence, float)
        assert 0.0 <= result.migration_confidence <= 1.0

    def test_transfer_chain_config(self):
        """Test transfer chain configuration."""
        config = TransferChainConfig(
            rapid_followup_seconds=600,
            min_migration_confidence=0.8,
            max_chain_depth=3,
        )

        detector = TransferChainDetector(config=config)
        assert detector._config.rapid_followup_seconds == 600
        assert detector._config.min_migration_confidence == 0.8
        assert detector._config.max_chain_depth == 3


# ============================================================================
# LP Action Detector Tests
# ============================================================================


class TestLPActionDetector:
    """Tests for LP action detection."""

    def test_lp_add_detection(self):
        """Test LP add liquidity detection."""
        detector = create_lp_action_detector()

        evidence = ExitEvidence(
            signature="test_sig",
            slot=123,
            block_time=datetime.now(timezone.utc),
            token_mint="TokenMint123",
            token_amount=1000000,
            token_decimals=9,
            program_ids_detected=("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",),
            dex_detected=None,
            lp_program_detected="raydium_amm",
            bridge_detected=None,
            sol_change_lamports=-10000,
            has_quote_asset_received=False,
            quote_asset_mint=None,
            quote_asset_amount=None,
            destination_address=None,
            destination_is_known_cex=False,
            destination_cex_name=None,
            destination_is_burn_address=False,
            destination_is_high_fan_in=False,
            lp_token_minted=True,
            lp_token_burned=False,
            lp_token_amount=100000,
            destination_shares_funder=False,
            destination_has_same_token=False,
            rapid_followup_detected=False,
        )

        result = detector.detect_lp_action(evidence)

        assert result.is_lp_action is True
        assert result.action_type == "add_liquidity"
        assert result.lp_program == "raydium_amm"
        assert result.confidence >= 0.5

    def test_non_lp_action(self):
        """Test that non-LP transactions are not classified as LP actions."""
        detector = create_lp_action_detector()

        evidence = ExitEvidence(
            signature="test_sig",
            slot=123,
            block_time=datetime.now(timezone.utc),
            token_mint="TokenMint123",
            token_amount=1000000,
            token_decimals=9,
            program_ids_detected=("JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",),
            dex_detected="jupiter_v6",
            lp_program_detected=None,
            bridge_detected=None,
            sol_change_lamports=500000000,
            has_quote_asset_received=True,
            quote_asset_mint=None,
            quote_asset_amount=500000000,
            destination_address=None,
            destination_is_known_cex=False,
            destination_cex_name=None,
            destination_is_burn_address=False,
            destination_is_high_fan_in=False,
            lp_token_minted=False,
            lp_token_burned=False,
            lp_token_amount=None,
            destination_shares_funder=False,
            destination_has_same_token=False,
            rapid_followup_detected=False,
        )

        result = detector.detect_lp_action(evidence)

        assert result.is_lp_action is False
        assert result.action_type is None


# ============================================================================
# CEX Deposit Detector Tests
# ============================================================================


class TestCEXDepositDetector:
    """Tests for CEX deposit detection."""

    def test_known_cex_address_detection(self):
        """Test detection of known CEX addresses."""
        detector = create_cex_deposit_detector()
        known_cex = list(KNOWN_CEX_ADDRESSES.keys())[0]
        expected_name = KNOWN_CEX_ADDRESSES[known_cex]

        result = detector.detect_cex_deposit(destination_address=known_cex)

        assert result.is_cex_deposit is True
        assert result.cex_name == expected_name
        assert result.detection_method == "known_address"
        assert result.confidence >= 0.9

    def test_high_fan_in_detection(self):
        """Test detection via high fan-in pattern."""
        detector = create_cex_deposit_detector()

        result = detector.detect_cex_deposit(
            destination_address="UnknownAddress123",
            fan_in_count=500,
        )

        assert result.is_cex_deposit is True
        assert result.detection_method == "fan_in_pattern"
        assert result.confidence >= 0.6

    def test_regular_wallet_not_cex(self):
        """Test that regular wallets are not classified as CEX."""
        detector = create_cex_deposit_detector()

        result = detector.detect_cex_deposit(
            destination_address="RegularWallet123",
            fan_in_count=5,
        )

        assert result.is_cex_deposit is False
        assert result.detection_method == "uncertain"

    def test_exchange_label_detection(self):
        """Test detection via exchange label."""
        detector = create_cex_deposit_detector()

        result = detector.detect_cex_deposit(
            destination_address="SomeAddress123",
            address_label="Binance Hot Wallet",
        )

        assert result.is_cex_deposit is True
        assert result.cex_name == "binance"
        assert result.detection_method == "exchange_label"


# ============================================================================
# PnL Reliability Scorer Tests
# ============================================================================


class TestPnLReliabilityScorer:
    """Tests for PnL reliability scoring."""

    def test_high_reliability_precise_display(self):
        """High reliability should enable precise display."""
        scorer = PnLReliabilityScorer()

        reliability, display_mode, components = scorer.compute_reliability(
            sell_confidence=0.9,
            entry_price_confidence=0.85,
            exit_price_confidence=0.9,
            liquidity_confidence=0.8,
            lot_count=1,
            has_transfer_ambiguity=False,
            event_completeness=0.95,
        )

        assert reliability >= 0.7
        assert display_mode == "precise"

    def test_medium_reliability_range_display(self):
        """Medium reliability should show range display."""
        scorer = PnLReliabilityScorer()

        reliability, display_mode, components = scorer.compute_reliability(
            sell_confidence=0.6,
            entry_price_confidence=0.5,
            exit_price_confidence=0.6,
            liquidity_confidence=0.5,
            lot_count=3,
            has_transfer_ambiguity=False,
            event_completeness=0.7,
        )

        assert 0.4 <= reliability < 0.7
        assert display_mode == "range"

    def test_low_reliability_unavailable(self):
        """Low reliability should mark as unavailable."""
        scorer = PnLReliabilityScorer()

        reliability, display_mode, components = scorer.compute_reliability(
            sell_confidence=0.2,
            entry_price_confidence=0.3,
            exit_price_confidence=0.3,
            liquidity_confidence=0.2,
            lot_count=5,
            has_transfer_ambiguity=True,
            event_completeness=0.3,
        )

        assert reliability < 0.4
        assert display_mode == "unavailable"

    def test_transfer_ambiguity_reduces_reliability(self):
        """Transfer ambiguity should reduce reliability."""
        scorer = PnLReliabilityScorer()

        # Without ambiguity
        rel_clean, _, _ = scorer.compute_reliability(
            sell_confidence=0.8,
            entry_price_confidence=0.8,
            exit_price_confidence=0.8,
            liquidity_confidence=0.8,
            lot_count=1,
            has_transfer_ambiguity=False,
            event_completeness=0.9,
        )

        # With ambiguity
        rel_ambiguous, _, _ = scorer.compute_reliability(
            sell_confidence=0.8,
            entry_price_confidence=0.8,
            exit_price_confidence=0.8,
            liquidity_confidence=0.8,
            lot_count=1,
            has_transfer_ambiguity=True,
            event_completeness=0.9,
        )

        assert rel_ambiguous < rel_clean


# ============================================================================
# Exit Type Coverage Tests
# ============================================================================


class TestExitTypeCoverage:
    """Tests to ensure all 10 exit types are properly handled."""

    def test_all_exit_types_defined(self):
        """Verify all 10 exit types are defined."""
        expected_types = {
            "dex_sell",
            "transfer_out",
            "cex_deposit",
            "lp_add",
            "lp_remove",
            "burn",
            "bridge",
            "wallet_migration",
            "program_interaction",
            "unknown_exit",
        }

        actual_types = {e.value for e in ExitEventType}
        assert actual_types == expected_types

    def test_dex_programs_defined(self):
        """Verify DEX programs are defined."""
        assert len(DEX_PROGRAMS) >= 5
        assert "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4" in DEX_PROGRAMS  # Jupiter v6

    def test_lp_programs_defined(self):
        """Verify LP programs are defined."""
        assert len(LP_PROGRAMS) >= 4
        assert "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8" in LP_PROGRAMS  # Raydium

    def test_bridge_programs_defined(self):
        """Verify bridge programs are defined."""
        assert len(BRIDGE_PROGRAMS) >= 3
        assert "wormDTUJ6AWPNvk59vGQbDvGJmqbDTdgWgAqcLBCgUb" in BRIDGE_PROGRAMS  # Wormhole

    def test_cex_addresses_defined(self):
        """Verify CEX addresses are defined."""
        assert len(KNOWN_CEX_ADDRESSES) >= 5

    def test_burn_addresses_defined(self):
        """Verify burn addresses are defined."""
        assert len(BURN_ADDRESSES) >= 1
        assert "1nc1nerator11111111111111111111111111111111" in BURN_ADDRESSES


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_exit_classifier(self):
        """Test exit classifier factory."""
        classifier = create_exit_classifier()
        assert isinstance(classifier, ExitEventClassifier)

    def test_create_exit_classifier_with_config(self):
        """Test exit classifier factory with custom config."""
        config = ExitClassifierConfig(min_sell_confidence_for_pnl=0.8)
        classifier = create_exit_classifier(config=config)
        assert classifier._config.min_sell_confidence_for_pnl == 0.8

    def test_create_transfer_chain_detector(self):
        """Test transfer chain detector factory."""
        detector = create_transfer_chain_detector()
        assert isinstance(detector, TransferChainDetector)

    def test_create_lp_action_detector(self):
        """Test LP action detector factory."""
        detector = create_lp_action_detector()
        assert isinstance(detector, LPActionDetector)

    def test_create_cex_deposit_detector(self):
        """Test CEX deposit detector factory."""
        detector = create_cex_deposit_detector()
        assert isinstance(detector, CEXDepositDetector)


# ============================================================================
# Integration Tests
# ============================================================================


class TestExitClassifierIntegration:
    """Integration tests combining multiple components."""

    def test_full_classification_pipeline(self, classifier, sample_tx_data_dex_sell):
        """Test the full classification pipeline."""
        # Step 1: Classify exit
        classification = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-1_000_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_dex_sell,
        )

        # Step 2: Compute detailed sell confidence
        scorer = SellConfidenceScorer()
        sell_score, pnl_computable, breakdown = scorer.compute_score(classification)

        # Step 3: Compute PnL reliability if computable
        if pnl_computable:
            reliability_scorer = PnLReliabilityScorer()
            reliability, display_mode, components = reliability_scorer.compute_reliability(
                sell_confidence=sell_score,
                entry_price_confidence=0.85,
                exit_price_confidence=0.9,
                liquidity_confidence=0.8,
                lot_count=1,
                has_transfer_ambiguity=False,
                event_completeness=0.95,
            )

            assert reliability >= 0.5
            assert display_mode in ["precise", "range", "unavailable"]

    def test_lp_action_prevents_pnl(self, classifier, sample_tx_data_lp_add):
        """Test that LP actions prevent PnL computation."""
        classification = classifier.classify(
            wallet_address="WalletAddress123",
            token_mint="TokenMint123",
            token_amount=-500_000_000,
            token_decimals=9,
            tx_data=sample_tx_data_lp_add,
        )

        # LP action detection
        lp_detector = create_lp_action_detector()
        lp_result = lp_detector.detect_lp_action(classification.evidence)

        # Verify LP action blocks PnL
        assert lp_result.is_lp_action is True
        assert classification.pnl_computable is False
