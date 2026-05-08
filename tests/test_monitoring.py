"""
Tests for Sprint 3: Real-time Monitoring & Alerts.

Tests WalletWatcher, AlertEngine, ProfileTracker, and notification delivery.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.monitoring.watcher import WalletWatcher, BalanceChange
from src.monitoring.alerts import (
    AlertEngine,
    Alert,
    AlertType,
    AlertSeverity,
    AlertConfig,
)
from src.monitoring.profiles import ProfileTracker, ProfileSnapshot, ProfileEvolution
from src.temporal.regimes import HolderRegimeType


class TestWalletWatcher:
    """Tests for WalletWatcher service."""

    @pytest.fixture
    async def watcher(self):
        """Create WalletWatcher instance."""
        mock_session = AsyncMock()
        return WalletWatcher(
            db_session=mock_session,
            check_interval=30,
            significance_threshold=0.05,
        )

    @pytest.mark.asyncio
    async def test_add_watched_wallet(self, watcher):
        """Test adding a wallet to watchlist."""
        wallet = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"
        token = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"
        user_id = "user123"

        watched = await watcher.add_watched_wallet(
            wallet=wallet,
            token_mint=token,
            user_id=user_id,
            alert_threshold=0.05,
        )

        assert watched.wallet == wallet
        assert watched.token_mint == token
        assert watched.user_id == user_id
        assert watched.alert_threshold == 0.05

    @pytest.mark.asyncio
    async def test_remove_watched_wallet(self, watcher):
        """Test removing a wallet from watchlist."""
        wallet = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"
        token = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"

        # Add first
        await watcher.add_watched_wallet(
            wallet=wallet,
            token_mint=token,
            user_id="user123",
        )

        # Then remove
        removed = await watcher.remove_watched_wallet(wallet, token)
        assert removed is True

        # Try removing again (should fail)
        removed = await watcher.remove_watched_wallet(wallet, token)
        assert removed is False

    @pytest.mark.asyncio
    async def test_get_watched_wallets_by_user(self, watcher):
        """Test filtering watched wallets by user."""
        wallet1 = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"
        wallet2 = "8yJXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyG"
        token = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"

        await watcher.add_watched_wallet(wallet1, token, "user1")
        await watcher.add_watched_wallet(wallet2, token, "user2")

        user1_wallets = await watcher.get_watched_wallets(user_id="user1")
        assert len(user1_wallets) == 1
        assert user1_wallets[0].wallet == wallet1

    @pytest.mark.asyncio
    async def test_check_balance_changes(self, watcher):
        """Test detecting balance changes."""
        wallet = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"
        token = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"

        # Add wallet
        await watcher.add_watched_wallet(wallet, token, "user1")

        # Mock balance change
        watcher._balance_cache[f"{wallet}:{token}"] = 100_000.0

        # Simulate new balance
        async def mock_fetch_balance(w, t):
            return 150_000.0  # 50% increase

        watcher._fetch_current_balance = mock_fetch_balance

        changes = await watcher.check_balance_changes(
            token_mint=token,
            total_supply=1_000_000.0,
        )

        assert len(changes) == 1
        change = changes[0]
        assert change.wallet == wallet
        assert change.delta == 50_000.0
        assert change.pct_of_supply == 0.05  # 5% of supply
        assert change.is_significant is True

    @pytest.mark.asyncio
    async def test_watcher_statistics(self, watcher):
        """Test getting watcher statistics."""
        wallet1 = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"
        wallet2 = "8yJXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyG"
        token = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"

        await watcher.add_watched_wallet(wallet1, token, "user1")
        await watcher.add_watched_wallet(wallet2, token, "user2")

        stats = watcher.get_statistics()

        assert stats["total_watched_wallets"] == 2
        assert stats["unique_tokens"] == 1
        assert stats["unique_users"] == 2
        assert stats["is_monitoring"] is False
        assert stats["check_interval"] == 30


class TestAlertEngine:
    """Tests for AlertEngine."""

    @pytest.fixture
    async def engine(self):
        """Create AlertEngine instance."""
        mock_session = AsyncMock()
        return AlertEngine(db_session=mock_session, default_cooldown=60)

    @pytest.fixture
    def config(self):
        """Create default AlertConfig."""
        return AlertConfig(
            id=1,
            user_id="user123",
            token_mint="4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        )

    @pytest.mark.asyncio
    async def test_create_whale_movement_alert(self, engine, config):
        """Test creating whale movement alert."""
        balance_change = BalanceChange(
            wallet="7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF",
            token_mint=config.token_mint,
            timestamp=datetime.now(timezone.utc),
            previous_balance=100_000.0,
            new_balance=150_000.0,
            delta=50_000.0,
            delta_pct=50.0,
            pct_of_supply=0.05,  # 5%
            is_significant=True,
        )

        alert = await engine.create_whale_movement_alert(balance_change, config)

        assert alert is not None
        assert alert.alert_type == AlertType.WHALE_MOVEMENT
        assert alert.severity == AlertSeverity.HIGH  # 5% = HIGH
        assert alert.wallet_address == balance_change.wallet
        assert alert.details["pct_of_supply"] == 0.05

    @pytest.mark.asyncio
    async def test_whale_alert_threshold_filtering(self, engine, config):
        """Test that alerts below threshold are suppressed."""
        balance_change = BalanceChange(
            wallet="7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF",
            token_mint=config.token_mint,
            timestamp=datetime.now(timezone.utc),
            previous_balance=100_000.0,
            new_balance=102_000.0,
            delta=2_000.0,
            delta_pct=2.0,
            pct_of_supply=0.002,  # 0.2% - below 5% threshold
            is_significant=False,
        )

        alert = await engine.create_whale_movement_alert(balance_change, config)

        assert alert is None  # Should be suppressed

    @pytest.mark.asyncio
    async def test_regime_change_alert(self, engine, config):
        """Test creating regime change alert."""
        alert = await engine.create_regime_change_alert(
            token_mint=config.token_mint,
            from_regime=HolderRegimeType.STABLE,
            to_regime=HolderRegimeType.DECAY,
            confidence=0.85,
            config=config,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.REGIME_CHANGE
        assert alert.severity == AlertSeverity.CRITICAL  # DECAY = CRITICAL
        assert alert.details["from_regime"] == "stable"
        assert alert.details["to_regime"] == "decay"
        assert alert.details["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_anomaly_spike_alert(self, engine, config):
        """Test creating anomaly spike alert."""
        alert = await engine.create_anomaly_spike_alert(
            token_mint=config.token_mint,
            anomaly_count=12,
            threshold=-0.8,
            config=config,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.ANOMALY_SPIKE
        assert alert.severity == AlertSeverity.CRITICAL  # 12 anomalies = CRITICAL
        assert alert.details["anomaly_count"] == 12

    @pytest.mark.asyncio
    async def test_alert_cooldown(self, engine, config):
        """Test that cooldown prevents duplicate alerts."""
        # Create first alert
        alert1 = await engine.create_regime_change_alert(
            token_mint=config.token_mint,
            from_regime=HolderRegimeType.STABLE,
            to_regime=HolderRegimeType.DISTRIBUTION,
            confidence=0.8,
            config=config,
        )

        assert alert1 is not None

        # Try creating second alert immediately (should be suppressed)
        alert2 = await engine.create_regime_change_alert(
            token_mint=config.token_mint,
            from_regime=HolderRegimeType.DISTRIBUTION,
            to_regime=HolderRegimeType.DECAY,
            confidence=0.8,
            config=config,
        )

        assert alert2 is None  # Suppressed by cooldown


class TestProfileTracker:
    """Tests for ProfileTracker."""

    @pytest.fixture
    async def tracker(self):
        """Create ProfileTracker instance."""
        mock_session = AsyncMock()
        return ProfileTracker(db_session=mock_session)

    @pytest.mark.asyncio
    async def test_add_snapshot(self, tracker):
        """Test adding a profile snapshot."""
        wallet = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"

        snapshot = await tracker.add_snapshot(
            wallet=wallet,
            archetype="sniper",
            risk_score=0.85,
            anomaly_score=-0.5,
        )

        assert snapshot.wallet == wallet
        assert snapshot.archetype == "sniper"
        assert snapshot.risk_score == 0.85
        assert snapshot.anomaly_score == -0.5

    @pytest.mark.asyncio
    async def test_compute_profile_velocity(self, tracker):
        """Test computing profile velocity from snapshots."""
        wallet = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"

        # Create snapshots with varying risk scores
        snapshots = [
            ProfileSnapshot(
                wallet=wallet,
                timestamp=datetime.now(timezone.utc) - timedelta(days=i),
                archetype="sniper",
                risk_score=0.5 + (i * 0.05),  # Increasing trend
            )
            for i in range(5)
        ]

        velocity = tracker.compute_profile_velocity(snapshots, window_days=7)

        assert velocity > 0.0  # Should have non-zero velocity due to variance

    @pytest.mark.asyncio
    async def test_update_profile(self, tracker):
        """Test updating a wallet profile."""
        wallet = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"

        updated = await tracker.update_profile(
            wallet=wallet,
            archetype="long_term_accumulator",
            risk_score=0.3,
        )

        assert updated is True


class TestProfileEvolution:
    """Tests for ProfileEvolution functionality."""

    def test_get_archetype_duration(self):
        """Test calculating archetype duration."""
        wallet = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"

        now = datetime.now(timezone.utc)
        snapshots = [
            ProfileSnapshot(
                wallet=wallet,
                timestamp=now - timedelta(days=10),
                archetype="sniper",
                risk_score=0.8,
            ),
            ProfileSnapshot(
                wallet=wallet,
                timestamp=now,
                archetype="long_term_accumulator",
                risk_score=0.3,
            ),
        ]

        transitions = [
            (now - timedelta(days=5), "sniper", "long_term_accumulator"),
        ]

        evolution = ProfileEvolution(
            wallet=wallet,
            snapshots=snapshots,
            archetype_transitions=transitions,
            current_archetype="long_term_accumulator",
            current_risk_score=0.3,
            profile_velocity=0.1,
        )

        duration = evolution.get_archetype_duration("long_term_accumulator")
        assert duration > 0  # Should be around 5 days

    def test_get_risk_trend(self):
        """Test detecting risk score trend."""
        wallet = "7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF"

        # Increasing trend
        snapshots = [
            ProfileSnapshot(
                wallet=wallet,
                timestamp=datetime.now(timezone.utc) - timedelta(days=i),
                archetype="sniper",
                risk_score=0.5 + (i * 0.05),
            )
            for i in range(5)
        ]

        evolution = ProfileEvolution(
            wallet=wallet,
            snapshots=snapshots,
            archetype_transitions=[],
            current_archetype="sniper",
            current_risk_score=0.7,
            profile_velocity=0.1,
        )

        trend = evolution.get_risk_trend()
        assert trend == "increasing"


class TestNotifications:
    """Tests for notification delivery."""

    @pytest.mark.asyncio
    async def test_telegram_alert_formatting(self):
        """Test formatting alerts for Telegram."""
        # Import directly from module to avoid __init__.py telegram dependency
        from src.telegram.notifications import NotificationDelivery

        mock_bot = MagicMock()
        delivery = NotificationDelivery(
            telegram_bot=mock_bot,
            max_alerts_per_hour=10,
        )

        alert = Alert(
            id=1,
            alert_type=AlertType.WHALE_MOVEMENT,
            severity=AlertSeverity.HIGH,
            wallet_address="7xKXtg2CW87d97L3aKHrJSmfuQL6N7yHEjGe6QmPwFyF",
            token_mint="4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
            timestamp=datetime.now(),  # Use datetime.now() instead of deprecated utcnow()
            details={
                "delta": 50000,
                "delta_pct": 50.0,
                "pct_of_supply": 0.05,
            },
            user_id="user123",
        )

        message = delivery._format_telegram_message(alert)

        assert "HIGH" in message
        assert "Whale Movement" in message
        assert alert.wallet_address[:8] in message

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiting prevents spam."""
        # Import directly from module to avoid __init__.py telegram dependency
        from src.telegram.notifications import NotificationDelivery

        mock_bot = MagicMock()
        delivery = NotificationDelivery(
            telegram_bot=mock_bot,
            max_alerts_per_hour=3,  # Low limit for testing
        )

        user_id = "user123"

        # Send 3 alerts (should work)
        for i in range(3):
            delivery._record_alert_sent(user_id)

        # 4th should be rate limited
        assert delivery._is_rate_limited(user_id) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
