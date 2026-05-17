"""Tests for new SWEENEE dashboard modules.

Covers: alerts.py, history.py, webhook.py, export.py, transactions dex_source
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile

# Import modules under test
from src.alerts import AlertService, AlertType, WhaleAlert
from src.history import SnapshotService, BalanceChange, render_historical_chart
from src.webhook import TelegramWebhook, WebhookResult
from src.export import (
    export_wallets_csv, export_wallets_json,
    export_transactions_csv, export_transactions_json,
    get_export_filename,
)
from src.transactions import TransactionType, SweeneeTransaction, classify_transaction
from src.token_balances import WalletBalance
from src.solana_client import TokenBalance
from src.cache import SweeneeCache


# === Fixtures ===

@pytest.fixture
def sample_balance():
    """Create a sample WalletBalance."""
    return WalletBalance(
        address="9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
        label="TestWhale",
        balance=TokenBalance(
            wallet_address="9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
            token_mint="FkAtYamtEMtgnsTeUhzhTCiT2Svyxw63UdUYp1T7pump",
            raw_amount=1000000000,
            decimals=6,
            ui_amount=1000.0,
            fetched_at=datetime.now(timezone.utc),
        ),
        share_of_tracked=0.5,
    )


@pytest.fixture
def sample_balances():
    """Create multiple sample balances."""
    now = datetime.now(timezone.utc)
    return [
        WalletBalance(
            address=f"wallet{i}{'x' * 36}"[:44],
            label=f"Whale{i}",
            balance=TokenBalance(
                wallet_address=f"wallet{i}{'x' * 36}"[:44],
                token_mint="FkAtYamtEMtgnsTeUhzhTCiT2Svyxw63UdUYp1T7pump",
                raw_amount=i * 1000000000,
                decimals=6,
                ui_amount=float(i * 1000),
                fetched_at=now,
            ),
            share_of_tracked=i / 10,
        )
        for i in range(1, 4)
    ]


@pytest.fixture
def sample_transaction():
    """Create a sample transaction."""
    return SweeneeTransaction(
        signature="5abc123" + "x" * 80,
        block_time=datetime.now(timezone.utc),
        wallet_address="9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
        token_mint="FkAtYamtEMtgnsTeUhzhTCiT2Svyxw63UdUYp1T7pump",
        amount_change=1500000.0,
        direction="in",
        classification=TransactionType.BUY,
        counterparty=None,
        dex_source="jupiter_v6",
    )


@pytest.fixture
def sample_transactions():
    """Create multiple transactions."""
    now = datetime.now(timezone.utc)
    return [
        SweeneeTransaction(
            signature=f"sig{i}" + "x" * 80,
            block_time=now - timedelta(hours=i),
            wallet_address="9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
            token_mint="FkAtYamtEMtgnsTeUhzhTCiT2Svyxw63UdUYp1T7pump",
            amount_change=1000000.0 if i % 2 == 0 else -500000.0,
            direction="in" if i % 2 == 0 else "out",
            classification=TransactionType.BUY if i % 2 == 0 else TransactionType.SELL,
            dex_source="jupiter_v6",
        )
        for i in range(5)
    ]


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite"
        cache = SweeneeCache(db_path)
        yield cache


# === AlertService Tests ===

class TestAlertService:
    """Tests for AlertService."""

    def test_init_default_threshold(self):
        """Test default threshold is 1M."""
        service = AlertService()
        assert service.large_move_threshold == 1_000_000

    def test_init_custom_threshold(self):
        """Test custom threshold."""
        service = AlertService(large_move_threshold=500_000)
        assert service.large_move_threshold == 500_000

    def test_check_large_buy(self, sample_transaction, temp_db):
        """Test detecting large buy alert."""
        with patch('src.alerts.get_cache', return_value=temp_db):
            service = AlertService(large_move_threshold=1_000_000)
            sample_transaction.amount_change = 2_000_000
            sample_transaction.classification = TransactionType.BUY

            alerts = service.check_transactions([sample_transaction])

            assert len(alerts) == 1
            assert alerts[0].alert_type == AlertType.LARGE_BUY
            assert alerts[0].amount == 2_000_000

    def test_check_large_sell(self, sample_transaction, temp_db):
        """Test detecting large sell alert."""
        with patch('src.alerts.get_cache', return_value=temp_db):
            service = AlertService(large_move_threshold=1_000_000)
            sample_transaction.amount_change = -2_000_000
            sample_transaction.classification = TransactionType.SELL

            alerts = service.check_transactions([sample_transaction])

            assert len(alerts) == 1
            assert alerts[0].alert_type == AlertType.LARGE_SELL

    def test_no_alert_below_threshold(self, sample_transaction, temp_db):
        """Test no alert for small transactions."""
        with patch('src.alerts.get_cache', return_value=temp_db):
            service = AlertService(large_move_threshold=1_000_000)
            sample_transaction.amount_change = 500_000
            sample_transaction.classification = TransactionType.BUY

            alerts = service.check_transactions([sample_transaction])

            assert len(alerts) == 0

    def test_whale_exit_detection(self, sample_transaction, temp_db):
        """Test whale exit alert when balance goes to zero."""
        with patch('src.alerts.get_cache', return_value=temp_db):
            service = AlertService(large_move_threshold=1_000_000, exit_threshold=100)
            sample_transaction.amount_change = -2_000_000
            sample_transaction.classification = TransactionType.SELL
            current_balances = {sample_transaction.wallet_address: 50}  # Below exit threshold

            alerts = service.check_transactions([sample_transaction], current_balances)

            assert len(alerts) == 1
            assert alerts[0].alert_type == AlertType.WHALE_EXIT


class TestWhaleAlert:
    """Tests for WhaleAlert dataclass."""

    def test_emoji_large_buy(self):
        """Test emoji for large buy."""
        alert = WhaleAlert(
            id=1,
            wallet_address="test",
            alert_type=AlertType.LARGE_BUY,
            amount=1000000,
            threshold_triggered=1000000,
            tx_signature=None,
            created_at=datetime.now(timezone.utc),
        )
        assert alert.emoji == "🟢"

    def test_emoji_large_sell(self):
        """Test emoji for large sell."""
        alert = WhaleAlert(
            id=1,
            wallet_address="test",
            alert_type=AlertType.LARGE_SELL,
            amount=1000000,
            threshold_triggered=1000000,
            tx_signature=None,
            created_at=datetime.now(timezone.utc),
        )
        assert alert.emoji == "🔴"

    def test_description_buy(self):
        """Test description for buy alert."""
        alert = WhaleAlert(
            id=1,
            wallet_address="test",
            alert_type=AlertType.LARGE_BUY,
            amount=1500000,
            threshold_triggered=1000000,
            tx_signature=None,
            created_at=datetime.now(timezone.utc),
        )
        assert "bought" in alert.description.lower()
        assert "1,500,000" in alert.description


# === SnapshotService Tests ===

class TestSnapshotService:
    """Tests for SnapshotService."""

    def test_take_snapshot(self, sample_balances, temp_db):
        """Test taking a balance snapshot."""
        with patch('src.history.get_cache', return_value=temp_db):
            service = SnapshotService("test_mint")
            service.take_snapshot(sample_balances)

            # Verify snapshots were saved
            history = service.get_total_history(days=1)
            assert len(history) >= 1

    def test_get_history_empty(self, temp_db):
        """Test getting history when none exists."""
        with patch('src.history.get_cache', return_value=temp_db):
            service = SnapshotService("test_mint")
            history = service.get_history("nonexistent_wallet", days=30)
            assert history == []


class TestBalanceChange:
    """Tests for BalanceChange dataclass."""

    def test_is_significant_above_threshold(self):
        """Test significant change detection."""
        change = BalanceChange(
            wallet_address="test",
            date="2026-05-17",
            previous_balance=1000,
            new_balance=1200,
            change_pct=20.0,
        )
        assert change.is_significant is True

    def test_is_significant_below_threshold(self):
        """Test non-significant change."""
        change = BalanceChange(
            wallet_address="test",
            date="2026-05-17",
            previous_balance=1000,
            new_balance=1050,
            change_pct=5.0,
        )
        assert change.is_significant is False


# === Export Tests ===

class TestExportFunctions:
    """Tests for export functions."""

    def test_export_wallets_csv(self, sample_balances):
        """Test CSV export of wallets."""
        csv_output = export_wallets_csv(sample_balances)

        assert "address" in csv_output
        assert "label" in csv_output
        assert "balance" in csv_output
        assert "Whale1" in csv_output
        assert "Whale2" in csv_output

    def test_export_wallets_json(self, sample_balances):
        """Test JSON export of wallets."""
        json_output = export_wallets_json(sample_balances)
        data = json.loads(json_output)

        assert "exported_at" in data
        assert "wallet_count" in data
        assert data["wallet_count"] == 3
        assert "wallets" in data
        assert len(data["wallets"]) == 3

    def test_export_transactions_csv(self, sample_transactions):
        """Test CSV export of transactions."""
        csv_output = export_transactions_csv(sample_transactions)

        assert "signature" in csv_output
        assert "classification" in csv_output
        assert "dex_source" in csv_output
        assert "jupiter_v6" in csv_output

    def test_export_transactions_json(self, sample_transactions):
        """Test JSON export of transactions."""
        json_output = export_transactions_json(sample_transactions)
        data = json.loads(json_output)

        assert "exported_at" in data
        assert "transaction_count" in data
        assert "summary" in data
        assert "buy_count" in data["summary"]
        assert "sell_count" in data["summary"]

    def test_get_export_filename(self):
        """Test filename generation."""
        filename = get_export_filename("wallets", "csv")

        assert filename.startswith("sweenee_wallets_")
        assert filename.endswith(".csv")
        assert len(filename) > 20  # Includes timestamp


# === Webhook Tests ===

class TestTelegramWebhook:
    """Tests for TelegramWebhook."""

    def test_not_configured_without_credentials(self):
        """Test webhook reports not configured without credentials."""
        with patch.dict('os.environ', {}, clear=True):
            webhook = TelegramWebhook(bot_token="", chat_id="")
            assert webhook.is_configured is False

    def test_configured_with_credentials(self):
        """Test webhook reports configured with credentials."""
        webhook = TelegramWebhook(bot_token="test_token", chat_id="test_chat")
        assert webhook.is_configured is True

    def test_generate_hash_unique(self):
        """Test hash generation produces unique hashes."""
        webhook = TelegramWebhook(bot_token="test", chat_id="test")
        hash1 = webhook._generate_hash("alert", "message1")
        hash2 = webhook._generate_hash("alert", "message2")
        hash3 = webhook._generate_hash("summary", "message1")

        assert hash1 != hash2  # Different content
        assert hash1 != hash3  # Different type

    @pytest.mark.asyncio
    async def test_send_returns_error_when_not_configured(self):
        """Test send returns error when not configured."""
        webhook = TelegramWebhook(bot_token="", chat_id="")
        result = await webhook.send_message("test message")

        assert result.success is False
        assert "not configured" in result.error.lower()


class TestWebhookResult:
    """Tests for WebhookResult dataclass."""

    def test_successful_result(self):
        """Test successful result."""
        result = WebhookResult(success=True, message_hash="abc123")
        assert result.success is True
        assert result.error is None

    def test_failed_result(self):
        """Test failed result."""
        result = WebhookResult(success=False, message_hash="abc123", error="Connection failed")
        assert result.success is False
        assert result.error == "Connection failed"


# === Transaction DEX Detection Tests ===

class TestDexDetection:
    """Tests for DEX source detection in transactions."""

    def test_sweenee_transaction_has_dex_source(self):
        """Test SweeneeTransaction has dex_source field."""
        tx = SweeneeTransaction(
            signature="test",
            block_time=None,
            wallet_address="test",
            token_mint="test",
            amount_change=1000,
            direction="in",
            classification=TransactionType.BUY,
            dex_source="jupiter_v6",
        )
        assert tx.dex_source == "jupiter_v6"

    def test_classify_transaction_returns_dex_source(self):
        """Test classify_transaction returns dex_source for Jupiter swap."""
        # Create mock tx_data with Jupiter v6 program and significant SOL movement
        # The wallet is at index 1, so preBalances[1] and postBalances[1] matter
        tx_data = {
            "transaction": {
                "message": {
                    "accountKeys": [
                        {"pubkey": "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"},
                        {"pubkey": "test_wallet"},
                    ]
                }
            },
            "meta": {
                "preBalances": [1000000000, 500000000],  # Index 1 = wallet
                "postBalances": [1000000000, 400000000],  # -100M lamports = -0.1 SOL (>0.01 SOL)
            }
        }

        classification, counterparty, dex_source = classify_transaction(
            wallet="test_wallet",
            mint="test_mint",
            tx_data=tx_data,
            amount_change=1000000,  # Positive = buying
        )

        assert dex_source == "jupiter_v6"
        assert classification == TransactionType.BUY

    def test_classify_orca_detection(self):
        """Test Orca DEX detection."""
        tx_data = {
            "transaction": {
                "message": {
                    "accountKeys": [
                        {"pubkey": "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"},
                        {"pubkey": "test_wallet"},
                    ]
                }
            },
            "meta": {
                "preBalances": [1000000000, 500000000],
                "postBalances": [1000000000, 400000000],  # -100M lamports SOL change
            }
        }

        classification, _, dex_source = classify_transaction(
            wallet="test_wallet",
            mint="test_mint",
            tx_data=tx_data,
            amount_change=1000000,
        )

        assert dex_source == "orca"
        assert classification == TransactionType.BUY

    def test_classify_no_dex(self):
        """Test classification without DEX."""
        tx_data = {
            "transaction": {
                "message": {
                    "accountKeys": [
                        {"pubkey": "some_random_program"},
                    ]
                }
            },
            "meta": {}
        }

        classification, _, dex_source = classify_transaction(
            wallet="test_wallet",
            mint="test_mint",
            tx_data=tx_data,
            amount_change=1000000,
        )

        assert classification == TransactionType.TRANSFER_IN
        assert dex_source == "none"


# === Cache Migration Tests ===

class TestCacheMigration:
    """Tests for cache migration."""

    def test_migration_adds_dex_source_column(self):
        """Test that migration adds dex_source column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"

            # Create old-style table without dex_source
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            conn.execute("""
                CREATE TABLE sweenee_transactions (
                    signature TEXT PRIMARY KEY,
                    block_time TEXT,
                    wallet_address TEXT NOT NULL,
                    token_mint TEXT NOT NULL,
                    amount_change REAL NOT NULL,
                    direction TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    counterparty TEXT,
                    explorer_url TEXT,
                    raw_json TEXT,
                    cached_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()

            # Initialize cache - should run migration
            cache = SweeneeCache(db_path)

            # Verify dex_source column exists
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("PRAGMA table_info(sweenee_transactions)")
            columns = [row[1] for row in cursor.fetchall()]
            conn.close()

            assert "dex_source" in columns
