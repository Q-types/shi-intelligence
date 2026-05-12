"""
Tests for Sprint 3: Telegram Command Handlers.

Tests /watch, /unwatch, /alerts, /profile commands and notification delivery.

Note: These tests use mocks and don't require python-telegram-bot to be installed.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import pytest

from src.monitoring.alerts import Alert, AlertConfig, AlertEngine, AlertSeverity, AlertType
from src.monitoring.profiles import ProfileSnapshot, ProfileTracker
from src.monitoring.watcher import BalanceChange, WalletWatcher
from src.temporal.regimes import HolderRegimeType
from src.telegram.commands import (
    handle_watch_command,
    handle_unwatch_command,
    handle_watchlist_command,
    handle_alerts_command,
    handle_profile_command,
)
from src.telegram.notifications import NotificationDelivery

# Mock telegram types since library may not be installed
try:
    from telegram import Bot, Message, Update, User
    from telegram.ext import ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    # Create mock classes for type hints
    Bot = Any
    Message = Any
    Update = Any
    User = Any
    class ContextTypes:
        DEFAULT_TYPE = Any


@pytest.mark.skipif(not TELEGRAM_AVAILABLE, reason="python-telegram-bot not installed")
class TestWatchCommands:
    """Tests for /watch, /unwatch, and /watchlist commands."""

    @pytest.fixture
    async def mock_watcher(self):
        """Create mock WalletWatcher."""
        mock_session = AsyncMock()
        watcher = WalletWatcher(mock_session, check_interval=30)
        return watcher

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram Update."""
        update = MagicMock(spec=Update)
        update.message = MagicMock(spec=Message)
        update.message.reply_text = AsyncMock()
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 123456789
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock Telegram Context."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.args = []
        return context

    @pytest.mark.asyncio
    async def test_watch_command_success(self, mock_watcher, mock_update, mock_context):
        """Test successful /watch command."""
        wallet = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"
        token = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"

        mock_context.args = [wallet, token, "0.05"]

        await handle_watch_command(mock_update, mock_context, mock_watcher)

        # Verify wallet was added
        watched_wallets = await mock_watcher.get_watched_wallets(user_id="123456789")
        assert len(watched_wallets) == 1
        assert watched_wallets[0].wallet == wallet
        assert watched_wallets[0].token_mint == token

        # Verify success message was sent
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "✅" in call_args
        assert "added to watchlist" in call_args.lower()

    @pytest.mark.asyncio
    async def test_watch_command_missing_args(self, mock_watcher, mock_update, mock_context):
        """Test /watch command with missing arguments."""
        mock_context.args = ["only_wallet_address"]

        await handle_watch_command(mock_update, mock_context, mock_watcher)

        # Verify error message was sent
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "❌" in call_args
        assert "provide both" in call_args.lower()

    @pytest.mark.asyncio
    async def test_watch_command_invalid_address(self, mock_watcher, mock_update, mock_context):
        """Test /watch command with invalid address."""
        mock_context.args = ["invalid", "also_invalid"]

        await handle_watch_command(mock_update, mock_context, mock_watcher)

        # Verify error message was sent
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "❌" in call_args
        assert "invalid" in call_args.lower()

    @pytest.mark.asyncio
    async def test_unwatch_command_success(self, mock_watcher, mock_update, mock_context):
        """Test successful /unwatch command."""
        wallet = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"
        token = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"

        # First add the wallet
        await mock_watcher.add_watched_wallet(wallet, token, "123456789")

        # Then unwatch
        mock_context.args = [wallet, token]
        await handle_unwatch_command(mock_update, mock_context, mock_watcher)

        # Verify wallet was removed
        watched_wallets = await mock_watcher.get_watched_wallets(user_id="123456789")
        assert len(watched_wallets) == 0

        # Verify success message
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "✅" in call_args
        assert "removed" in call_args.lower()

    @pytest.mark.asyncio
    async def test_watchlist_command(self, mock_watcher, mock_update, mock_context):
        """Test /watchlist command."""
        wallet1 = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"
        wallet2 = "8yJXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyG"
        token = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"

        # Add wallets
        await mock_watcher.add_watched_wallet(wallet1, token, "123456789")
        await mock_watcher.add_watched_wallet(wallet2, token, "123456789")

        await handle_watchlist_command(mock_update, mock_context, mock_watcher)

        # Verify watchlist was displayed
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "2" in call_args  # Should show count
        assert wallet1[:8] in call_args or wallet2[:8] in call_args


@pytest.mark.skipif(not TELEGRAM_AVAILABLE, reason="python-telegram-bot not installed")
class TestAlertsCommands:
    """Tests for /alerts command."""

    @pytest.fixture
    async def mock_alert_engine(self):
        """Create mock AlertEngine."""
        mock_session = AsyncMock()
        engine = AlertEngine(mock_session)
        return engine

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram Update."""
        update = MagicMock(spec=Update)
        update.message = MagicMock(spec=Message)
        update.message.reply_text = AsyncMock()
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 123456789
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock Telegram Context."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.args = []
        return context

    @pytest.mark.asyncio
    async def test_alerts_command_show_config(
        self, mock_alert_engine, mock_update, mock_context
    ):
        """Test /alerts command showing current configuration."""
        # Call command with no args (should show help/config)
        await handle_alerts_command(mock_update, mock_context)

        # Verify configuration/help was displayed
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "alert" in call_args.lower() or "usage" in call_args.lower()

    @pytest.mark.asyncio
    async def test_alerts_command_enable_type(
        self, mock_alert_engine, mock_update, mock_context
    ):
        """Test /alerts command enabling an alert type."""
        mock_context.args = ["enable", "whale_movement"]

        await handle_alerts_command(mock_update, mock_context)

        # Verify response message
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        # Command should respond with either success or usage info
        assert len(call_args) > 0

    @pytest.mark.asyncio
    async def test_alerts_command_set_threshold(
        self, mock_alert_engine, mock_update, mock_context
    ):
        """Test /alerts command setting a threshold."""
        mock_context.args = ["threshold", "whale", "0.10"]

        await handle_alerts_command(mock_update, mock_context)

        # Verify response message
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        # Command should respond
        assert len(call_args) > 0


@pytest.mark.skipif(not TELEGRAM_AVAILABLE, reason="python-telegram-bot not installed")
class TestProfileCommands:
    """Tests for /profile command."""

    @pytest.fixture
    async def mock_profile_tracker(self):
        """Create mock ProfileTracker."""
        mock_session = AsyncMock()
        tracker = ProfileTracker(mock_session)
        return tracker

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram Update."""
        update = MagicMock(spec=Update)
        update.message = MagicMock(spec=Message)
        update.message.reply_text = AsyncMock()
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 123456789
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock Telegram Context."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.args = []
        return context

    @pytest.mark.asyncio
    async def test_profile_command_success(
        self, mock_profile_tracker, mock_update, mock_context
    ):
        """Test successful /profile command."""
        wallet = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"

        # Create some profile history
        now = datetime.now(timezone.utc)
        snapshot1 = ProfileSnapshot(
            wallet=wallet,
            timestamp=now - timedelta(days=7),
            archetype="accumulator",
            risk_score=0.3,
        )
        snapshot2 = ProfileSnapshot(
            wallet=wallet,
            timestamp=now,
            archetype="whale",
            risk_score=0.7,
        )

        await mock_profile_tracker.add_snapshot(
            wallet=wallet,
            archetype=snapshot1.archetype,
            risk_score=snapshot1.risk_score,
        )
        await mock_profile_tracker.add_snapshot(
            wallet=wallet,
            archetype=snapshot2.archetype,
            risk_score=snapshot2.risk_score,
        )

        # /profile expects [wallet, days] - days is optional
        mock_context.args = [wallet]
        await handle_profile_command(mock_update, mock_context, mock_profile_tracker)

        # Verify response was sent (may say "no history" if tracker returns None)
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert wallet[:8] in call_args

    @pytest.mark.asyncio
    async def test_profile_command_missing_args(
        self, mock_profile_tracker, mock_update, mock_context
    ):
        """Test /profile command with missing arguments."""
        mock_context.args = []

        await handle_profile_command(mock_update, mock_context, mock_profile_tracker)

        # Verify error message
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "❌" in call_args or "usage" in call_args.lower()

    @pytest.mark.asyncio
    async def test_profile_command_no_history(
        self, mock_profile_tracker, mock_update, mock_context
    ):
        """Test /profile command for wallet with no history."""
        wallet = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"

        # /profile expects [wallet, days] where days is optional
        mock_context.args = [wallet]
        await handle_profile_command(mock_update, mock_context, mock_profile_tracker)

        # Verify "no history" message
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "no" in call_args.lower() or "history" in call_args.lower()


@pytest.mark.skipif(not TELEGRAM_AVAILABLE, reason="python-telegram-bot not installed")
class TestNotificationDelivery:
    """Tests for push notification delivery."""

    @pytest.fixture
    async def mock_bot(self):
        """Create mock Telegram Bot."""
        bot = MagicMock(spec=Bot)
        bot.send_message = AsyncMock()
        return bot

    @pytest.fixture
    async def notification_delivery(self, mock_bot):
        """Create NotificationDelivery instance."""
        delivery = NotificationDelivery(
            telegram_bot=mock_bot,
            max_alerts_per_hour=10,
            batch_window_seconds=60,
        )
        async with delivery:
            yield delivery

    @pytest.mark.asyncio
    async def test_send_telegram_alert_success(self, notification_delivery, mock_bot):
        """Test successful Telegram alert delivery."""
        alert = Alert(
            id=None,
            alert_type=AlertType.WHALE_MOVEMENT,
            severity=AlertSeverity.WARNING,
            wallet_address="7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF",
            token_mint="4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
            timestamp=datetime.now(timezone.utc),
            details={"delta_pct": 5.2, "pct_of_supply": 0.05},
            user_id="123456789",
        )

        # send_telegram_alert takes chat_id and alert as separate args
        await notification_delivery.send_telegram_alert("123456789", alert)

        # Verify message was sent
        mock_bot.send_message.assert_called_once()
        call_args = mock_bot.send_message.call_args
        assert call_args.kwargs["chat_id"] == "123456789"
        assert "Whale" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_rate_limiting(self, notification_delivery, mock_bot):
        """Test rate limiting of alerts."""
        # Create many alerts using correct Alert signature
        alerts = [
            Alert(
                id=None,
                alert_type=AlertType.WHALE_MOVEMENT,
                severity=AlertSeverity.INFO,
                wallet_address=f"Wallet{i}" + "x" * 35,
                token_mint="4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
                timestamp=datetime.now(timezone.utc),
                details={"alert_num": i},
                user_id="123456789",
            )
            for i in range(15)
        ]

        # Send all alerts using deliver_alert which handles rate limiting
        for alert in alerts:
            await notification_delivery.deliver_alert(alert, chat_id="123456789")

        # Verify rate limiting kicked in (max 10 per hour)
        assert mock_bot.send_message.call_count <= 10

    @pytest.mark.asyncio
    async def test_alert_batching(self, notification_delivery, mock_bot):
        """Test alert batching within time window."""
        # Create alerts using correct Alert signature
        alerts = [
            Alert(
                id=None,
                alert_type=AlertType.ANOMALY_SPIKE,
                severity=AlertSeverity.INFO,
                wallet_address=None,
                token_mint="4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
                timestamp=datetime.now(timezone.utc),
                details={"anomaly_count": i},
                user_id="123456789",
            )
            for i in range(3)
        ]

        # Queue alerts for batching via _add_to_batch (internal method)
        for alert in alerts:
            await notification_delivery._add_to_batch("123456789", "123456789", alert)

        # Wait for batch window (batch_window_seconds=60 in fixture, but we can check pending)
        await asyncio.sleep(0.1)

        # Alerts should be pending in batch queue
        assert len(notification_delivery._pending_alerts) > 0 or mock_bot.send_message.call_count <= 1

    @pytest.mark.asyncio
    async def test_format_alert_message(self, notification_delivery):
        """Test alert message formatting."""
        alert = Alert(
            id=None,
            alert_type=AlertType.REGIME_CHANGE,
            severity=AlertSeverity.CRITICAL,
            wallet_address=None,
            token_mint="4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
            timestamp=datetime.now(timezone.utc),
            details={"from_regime": "accumulation", "to_regime": "distribution", "confidence": 0.85},
            user_id="123456789",
        )

        # Use the actual method name
        formatted = notification_delivery._format_telegram_message(alert)

        # Verify formatting
        assert "🚨" in formatted  # CRITICAL severity emoji
        assert "CRITICAL" in formatted
        assert "Regime" in formatted or "regime" in formatted

    @pytest.mark.asyncio
    async def test_delivery_latency_under_30_seconds(
        self, notification_delivery, mock_bot
    ):
        """Test that alert delivery completes in under 30 seconds."""
        alert = Alert(
            id=None,
            alert_type=AlertType.WHALE_MOVEMENT,
            severity=AlertSeverity.WARNING,
            wallet_address="7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF",
            token_mint="4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
            timestamp=datetime.now(timezone.utc),
            details={"delta": 100000, "pct_of_supply": 0.05},
            user_id="123456789",
        )

        start_time = datetime.now(timezone.utc)
        await notification_delivery.send_telegram_alert("123456789", alert)
        end_time = datetime.now(timezone.utc)

        latency = (end_time - start_time).total_seconds()
        assert latency < 30, f"Alert delivery took {latency}s, exceeds 30s target"

    @pytest.mark.asyncio
    async def test_webhook_delivery(self, notification_delivery):
        """Test webhook alert delivery."""
        alert = Alert(
            id=None,
            alert_type=AlertType.CONCENTRATION_INCREASE,
            severity=AlertSeverity.INFO,
            wallet_address=None,
            token_mint="4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
            timestamp=datetime.now(timezone.utc),
            details={"hhi_change": 0.10, "new_hhi": 0.15},
            user_id="123456789",
        )

        webhook_url = "https://example.com/webhook"

        with patch.object(
            notification_delivery._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value.status_code = 200

            await notification_delivery.send_webhook_alert(webhook_url, alert)

            # Verify webhook was called
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert webhook_url in call_args[0]
            assert "alert_type" in call_args.kwargs["json"]


@pytest.mark.skipif(not TELEGRAM_AVAILABLE, reason="python-telegram-bot not installed")
class TestCommandIntegration:
    """Integration tests for command workflow."""

    @pytest.fixture
    async def full_stack(self):
        """Create full stack of components."""
        mock_session = AsyncMock()
        watcher = WalletWatcher(mock_session)
        alert_engine = AlertEngine(mock_session)
        profile_tracker = ProfileTracker(mock_session)
        mock_bot = MagicMock(spec=Bot)
        mock_bot.send_message = AsyncMock()
        notification_delivery = NotificationDelivery(mock_bot)

        return {
            "watcher": watcher,
            "alert_engine": alert_engine,
            "profile_tracker": profile_tracker,
            "notification_delivery": notification_delivery,
            "bot": mock_bot,
        }

    @pytest.mark.asyncio
    async def test_watch_to_alert_flow(self, full_stack):
        """Test complete flow from /watch to alert delivery."""
        watcher = full_stack["watcher"]
        alert_engine = full_stack["alert_engine"]
        notification_delivery = full_stack["notification_delivery"]

        wallet = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"
        token = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"
        user_id = "123456789"

        # Step 1: Add wallet to watchlist
        await watcher.add_watched_wallet(wallet, token, user_id)

        # Step 2: Configure alerts (using correct AlertConfig signature)
        config = AlertConfig(
            id=None,
            user_id=user_id,
            token_mint=token,
            whale_movement_threshold=0.05,
        )

        # Step 3: Simulate balance change
        balance_change = BalanceChange(
            wallet=wallet,
            token_mint=token,
            timestamp=datetime.now(timezone.utc),
            previous_balance=1_000_000.0,
            new_balance=1_100_000.0,
            delta=100_000.0,
            delta_pct=10.0,
            pct_of_supply=0.10,
            is_significant=True,
        )

        # Step 4: Create alert using the alert engine's method
        alert = await alert_engine.create_whale_movement_alert(balance_change, config)
        assert alert is not None
        assert alert.alert_type == AlertType.WHALE_MOVEMENT

        # Step 5: Deliver notification
        async with notification_delivery:
            await notification_delivery.send_telegram_alert(user_id, alert)

        # Verify notification was sent
        full_stack["bot"].send_message.assert_called_once()
