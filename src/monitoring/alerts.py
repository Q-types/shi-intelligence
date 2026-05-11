"""
Alert Engine for SHI.

Manages alert generation, configuration, and delivery tracking.
Supports whale_movement, regime_change, and anomaly_spike alerts.
Includes WebSocket broadcasting for real-time delivery.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Coroutine, Dict, List, Optional, Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.types import WalletAddress, TokenMint
from ..temporal.regimes import HolderRegimeType
from .watcher import BalanceChange

logger = structlog.get_logger()

# Type alias for alert broadcast callback
AlertBroadcastCallback = Callable[[Any], Coroutine[Any, Any, int]]


class AlertType(Enum):
    """Types of alerts supported by SHI."""

    WHALE_MOVEMENT = "whale_movement"
    REGIME_CHANGE = "regime_change"
    ANOMALY_SPIKE = "anomaly_spike"
    CONCENTRATION_INCREASE = "concentration_increase"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert event to be delivered."""

    id: Optional[int]
    alert_type: AlertType
    severity: AlertSeverity
    wallet_address: Optional[WalletAddress]
    token_mint: TokenMint
    timestamp: datetime
    details: Dict
    user_id: Optional[str] = None

    # Delivery tracking
    sent_to_telegram: bool = False
    sent_to_webhook: bool = False
    telegram_message_id: Optional[str] = None

    def get_message(self) -> str:
        """
        Format alert as human-readable message.

        Returns:
            Formatted alert message
        """
        if self.alert_type == AlertType.WHALE_MOVEMENT:
            direction = "bought" if self.details.get("delta", 0) > 0 else "sold"
            pct = abs(self.details.get("delta_pct", 0))
            supply_pct = self.details.get("pct_of_supply", 0) * 100

            wallet_str = f"{self.wallet_address[:8]}...{self.wallet_address[-6:]}" if self.wallet_address else "unknown"
            return (
                f"🐋 Whale Movement Alert\n"
                f"Wallet: {wallet_str}\n"
                f"Action: {direction} {pct:.2f}% of their holdings\n"
                f"Impact: {supply_pct:.2f}% of total supply\n"
                f"Token: {self.token_mint[:8]}..."
            )

        elif self.alert_type == AlertType.REGIME_CHANGE:
            from_regime = self.details.get("from_regime", "unknown")
            to_regime = self.details.get("to_regime", "unknown")
            confidence = self.details.get("confidence", 0) * 100

            return (
                f"📊 Regime Change Detected\n"
                f"From: {from_regime}\n"
                f"To: {to_regime}\n"
                f"Confidence: {confidence:.1f}%\n"
                f"Token: {self.token_mint[:8]}..."
            )

        elif self.alert_type == AlertType.ANOMALY_SPIKE:
            anomaly_count = self.details.get("anomaly_count", 0)
            threshold = self.details.get("threshold", 0)

            return (
                f"⚠️ Anomaly Spike Detected\n"
                f"Anomalous wallets: {anomaly_count}\n"
                f"Above threshold: {threshold}\n"
                f"Token: {self.token_mint[:8]}..."
            )

        elif self.alert_type == AlertType.CONCENTRATION_INCREASE:
            hhi_change = self.details.get("hhi_change", 0)
            new_hhi = self.details.get("new_hhi", 0)

            return (
                f"📈 Concentration Increase\n"
                f"HHI Change: +{hhi_change:.4f}\n"
                f"New HHI: {new_hhi:.4f}\n"
                f"Token: {self.token_mint[:8]}..."
            )

        return f"Alert: {self.alert_type.value}"


@dataclass
class AlertConfig:
    """User-specific alert configuration."""

    id: Optional[int]
    user_id: str
    token_mint: TokenMint

    # Thresholds
    whale_movement_threshold: float = 0.05  # 5% of supply
    concentration_increase_threshold: float = 0.02  # 2% HHI change
    anomaly_score_threshold: float = -0.8

    # Channels
    telegram_enabled: bool = True
    webhook_url: Optional[str] = None

    # Cooldown (prevent spam)
    cooldown_minutes: int = 60

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AlertEngine:
    """
    Alert generation and delivery engine.

    Manages alert rules, cooldown periods, and delivery tracking.
    Integrates with WalletWatcher and regime/anomaly detectors.
    Supports WebSocket broadcasting for real-time delivery.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        default_cooldown: int = 60,  # minutes
        broadcast_callback: Optional[AlertBroadcastCallback] = None,
    ):
        """
        Initialize alert engine.

        Args:
            db_session: Database session
            default_cooldown: Default cooldown period in minutes
            broadcast_callback: Optional callback for broadcasting alerts (e.g., WebSocket)
        """
        self.db_session = db_session
        self.default_cooldown = default_cooldown
        self._alert_history: Dict[str, datetime] = {}  # For cooldown tracking
        self._broadcast_callback = broadcast_callback
        self._broadcast_stats = {
            "total_broadcasts": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
        }

    def set_broadcast_callback(self, callback: AlertBroadcastCallback) -> None:
        """
        Set the broadcast callback for real-time alert delivery.

        Args:
            callback: Async function that takes an Alert and returns delivery count
        """
        self._broadcast_callback = callback
        logger.info("broadcast_callback_configured")

    async def _broadcast_alert(self, alert: Alert) -> int:
        """
        Broadcast an alert via the configured callback.

        Args:
            alert: Alert to broadcast

        Returns:
            Number of recipients that received the alert
        """
        if not self._broadcast_callback:
            return 0

        try:
            recipients = await self._broadcast_callback(alert)
            self._broadcast_stats["total_broadcasts"] += 1
            self._broadcast_stats["successful_deliveries"] += recipients

            logger.info(
                "alert_broadcast",
                alert_type=alert.alert_type.value,
                recipients=recipients,
            )

            return recipients

        except Exception as e:
            self._broadcast_stats["failed_deliveries"] += 1
            logger.error("alert_broadcast_failed", error=str(e))
            return 0

    def get_broadcast_stats(self) -> Dict[str, int]:
        """Get broadcast statistics."""
        return self._broadcast_stats.copy()

    def _get_cooldown_key(
        self,
        alert_type: AlertType,
        token_mint: TokenMint,
        wallet: Optional[WalletAddress] = None,
    ) -> str:
        """Generate unique key for cooldown tracking."""
        if wallet:
            return f"{alert_type.value}:{token_mint}:{wallet}"
        return f"{alert_type.value}:{token_mint}"

    def _is_in_cooldown(
        self,
        alert_type: AlertType,
        token_mint: TokenMint,
        wallet: Optional[WalletAddress] = None,
        cooldown_minutes: int = 60,
    ) -> bool:
        """
        Check if an alert type is in cooldown period.

        Args:
            alert_type: Type of alert
            token_mint: Token mint
            wallet: Optional wallet address
            cooldown_minutes: Cooldown period

        Returns:
            True if in cooldown, False otherwise
        """
        key = self._get_cooldown_key(alert_type, token_mint, wallet)

        if key in self._alert_history:
            last_sent = self._alert_history[key]
            elapsed = (datetime.now(timezone.utc) - last_sent).total_seconds() / 60

            if elapsed < cooldown_minutes:
                logger.debug(
                    "alert_in_cooldown",
                    alert_type=alert_type.value,
                    token=token_mint,
                    elapsed_minutes=elapsed,
                )
                return True

        return False

    def _mark_sent(
        self,
        alert_type: AlertType,
        token_mint: TokenMint,
        wallet: Optional[WalletAddress] = None,
    ) -> None:
        """Mark an alert as sent (for cooldown tracking)."""
        key = self._get_cooldown_key(alert_type, token_mint, wallet)
        self._alert_history[key] = datetime.now(timezone.utc)

    async def create_whale_movement_alert(
        self,
        balance_change: BalanceChange,
        config: AlertConfig,
    ) -> Optional[Alert]:
        """
        Create whale movement alert from balance change.

        Args:
            balance_change: Detected balance change
            config: User alert configuration

        Returns:
            Alert object or None if suppressed
        """
        # Check threshold
        if balance_change.pct_of_supply < config.whale_movement_threshold:
            return None

        # Check cooldown
        if self._is_in_cooldown(
            AlertType.WHALE_MOVEMENT,
            balance_change.token_mint,
            balance_change.wallet,
            config.cooldown_minutes,
        ):
            return None

        # Determine severity
        severity = AlertSeverity.INFO
        if balance_change.pct_of_supply >= 0.10:  # 10%
            severity = AlertSeverity.CRITICAL
        elif balance_change.pct_of_supply >= 0.05:  # 5%
            severity = AlertSeverity.HIGH
        elif balance_change.pct_of_supply >= 0.02:  # 2%
            severity = AlertSeverity.WARNING

        alert = Alert(
            id=None,
            alert_type=AlertType.WHALE_MOVEMENT,
            severity=severity,
            wallet_address=balance_change.wallet,
            token_mint=balance_change.token_mint,
            timestamp=balance_change.timestamp,
            details={
                "delta": balance_change.delta,
                "delta_pct": balance_change.delta_pct,
                "pct_of_supply": balance_change.pct_of_supply,
                "previous_balance": balance_change.previous_balance,
                "new_balance": balance_change.new_balance,
            },
            user_id=config.user_id,
        )

        self._mark_sent(AlertType.WHALE_MOVEMENT, balance_change.token_mint, balance_change.wallet)

        logger.info(
            "whale_movement_alert_created",
            wallet=balance_change.wallet,
            severity=severity.value,
            pct_of_supply=balance_change.pct_of_supply,
        )

        # Broadcast via WebSocket
        await self._broadcast_alert(alert)

        return alert

    async def create_regime_change_alert(
        self,
        token_mint: TokenMint,
        from_regime: HolderRegimeType,
        to_regime: HolderRegimeType,
        confidence: float,
        config: AlertConfig,
    ) -> Optional[Alert]:
        """
        Create regime change alert.

        Args:
            token_mint: Token mint
            from_regime: Previous regime
            to_regime: New regime
            confidence: Confidence in transition
            config: User alert configuration

        Returns:
            Alert object or None if suppressed
        """
        # Check cooldown
        if self._is_in_cooldown(
            AlertType.REGIME_CHANGE,
            token_mint,
            cooldown_minutes=config.cooldown_minutes,
        ):
            return None

        # Determine severity based on regime type
        severity = AlertSeverity.INFO
        if to_regime == HolderRegimeType.DECAY:
            severity = AlertSeverity.CRITICAL
        elif to_regime == HolderRegimeType.DISTRIBUTION:
            severity = AlertSeverity.HIGH
        elif to_regime == HolderRegimeType.COORDINATED_ACCUMULATION:
            severity = AlertSeverity.WARNING

        alert = Alert(
            id=None,
            alert_type=AlertType.REGIME_CHANGE,
            severity=severity,
            wallet_address=None,
            token_mint=token_mint,
            timestamp=datetime.now(timezone.utc),
            details={
                "from_regime": from_regime.value,
                "to_regime": to_regime.value,
                "confidence": confidence,
            },
            user_id=config.user_id,
        )

        self._mark_sent(AlertType.REGIME_CHANGE, token_mint)

        logger.info(
            "regime_change_alert_created",
            from_regime=from_regime.value,
            to_regime=to_regime.value,
            confidence=confidence,
        )

        # Broadcast via WebSocket
        await self._broadcast_alert(alert)

        return alert

    async def create_anomaly_spike_alert(
        self,
        token_mint: TokenMint,
        anomaly_count: int,
        threshold: float,
        config: AlertConfig,
    ) -> Optional[Alert]:
        """
        Create anomaly spike alert.

        Args:
            token_mint: Token mint
            anomaly_count: Number of anomalous wallets detected
            threshold: Anomaly score threshold used
            config: User alert configuration

        Returns:
            Alert object or None if suppressed
        """
        # Check cooldown
        if self._is_in_cooldown(
            AlertType.ANOMALY_SPIKE,
            token_mint,
            cooldown_minutes=config.cooldown_minutes,
        ):
            return None

        # Determine severity based on count
        severity = AlertSeverity.INFO
        if anomaly_count >= 10:
            severity = AlertSeverity.CRITICAL
        elif anomaly_count >= 5:
            severity = AlertSeverity.HIGH
        elif anomaly_count >= 3:
            severity = AlertSeverity.WARNING

        alert = Alert(
            id=None,
            alert_type=AlertType.ANOMALY_SPIKE,
            severity=severity,
            wallet_address=None,
            token_mint=token_mint,
            timestamp=datetime.now(timezone.utc),
            details={
                "anomaly_count": anomaly_count,
                "threshold": threshold,
            },
            user_id=config.user_id,
        )

        self._mark_sent(AlertType.ANOMALY_SPIKE, token_mint)

        logger.info(
            "anomaly_spike_alert_created",
            anomaly_count=anomaly_count,
            severity=severity.value,
        )

        # Broadcast via WebSocket
        await self._broadcast_alert(alert)

        return alert

    async def create_concentration_increase_alert(
        self,
        token_mint: TokenMint,
        hhi_change: float,
        new_hhi: float,
        config: AlertConfig,
    ) -> Optional[Alert]:
        """
        Create concentration increase alert.

        Args:
            token_mint: Token mint
            hhi_change: Change in HHI
            new_hhi: New HHI value
            config: User alert configuration

        Returns:
            Alert object or None if suppressed
        """
        # Check threshold
        if hhi_change < config.concentration_increase_threshold:
            return None

        # Check cooldown
        if self._is_in_cooldown(
            AlertType.CONCENTRATION_INCREASE,
            token_mint,
            cooldown_minutes=config.cooldown_minutes,
        ):
            return None

        # Determine severity
        severity = AlertSeverity.INFO
        if hhi_change >= 0.05:
            severity = AlertSeverity.CRITICAL
        elif hhi_change >= 0.03:
            severity = AlertSeverity.HIGH
        elif hhi_change >= 0.02:
            severity = AlertSeverity.WARNING

        alert = Alert(
            id=None,
            alert_type=AlertType.CONCENTRATION_INCREASE,
            severity=severity,
            wallet_address=None,
            token_mint=token_mint,
            timestamp=datetime.now(timezone.utc),
            details={
                "hhi_change": hhi_change,
                "new_hhi": new_hhi,
            },
            user_id=config.user_id,
        )

        self._mark_sent(AlertType.CONCENTRATION_INCREASE, token_mint)

        logger.info(
            "concentration_increase_alert_created",
            hhi_change=hhi_change,
            severity=severity.value,
        )

        # Broadcast via WebSocket
        await self._broadcast_alert(alert)

        return alert

    async def get_user_config(
        self,
        user_id: str,
        token_mint: TokenMint,
    ) -> AlertConfig:
        """
        Get alert configuration for a user/token pair.

        Args:
            user_id: User ID
            token_mint: Token mint

        Returns:
            AlertConfig (creates default if not exists)
        """
        # In production, this would query the database
        # For now, return default config
        return AlertConfig(
            id=None,
            user_id=user_id,
            token_mint=token_mint,
        )

    async def save_alert(self, alert: Alert) -> Alert:
        """
        Save alert to database.

        Args:
            alert: Alert to save

        Returns:
            Alert with id populated
        """
        # In production, this would insert into wallet_alerts table
        # For now, just assign a dummy ID
        if alert.id is None:
            alert.id = 1

        logger.info(
            "alert_saved",
            alert_id=alert.id,
            alert_type=alert.alert_type.value,
            severity=alert.severity.value,
        )

        return alert

    async def get_recent_alerts(
        self,
        user_id: Optional[str] = None,
        token_mint: Optional[TokenMint] = None,
        limit: int = 50,
    ) -> List[Alert]:
        """
        Get recent alerts with optional filtering.

        Args:
            user_id: Filter by user
            token_mint: Filter by token
            limit: Maximum number of alerts

        Returns:
            List of Alert objects
        """
        # In production, this would query wallet_alerts table
        # For now, return empty list
        return []
