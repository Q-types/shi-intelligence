"""
Push Notification System for SHI Telegram Bot.

Handles alert delivery via Telegram and webhooks with rate limiting and batching.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx
import structlog

from ..monitoring.alerts import Alert, AlertSeverity

# Optional telegram imports for testing without the library
try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    # Create placeholder types for type checking
    if TYPE_CHECKING:
        from telegram import Bot
        from telegram.error import TelegramError
    else:
        Bot = Any  # type: ignore[misc]
        TelegramError = Exception  # type: ignore[misc]

logger = structlog.get_logger()


class NotificationDelivery:
    """
    Handles push notification delivery with rate limiting and batching.

    Features:
    - Telegram message delivery
    - Webhook delivery
    - Alert batching (combine multiple alerts)
    - Rate limiting (prevent spam)
    - Retry logic
    """

    def __init__(
        self,
        telegram_bot: Optional[Any] = None,
        max_alerts_per_hour: int = 10,
        batch_window_seconds: int = 60,
    ):
        """
        Initialize notification delivery.

        Args:
            telegram_bot: Telegram Bot instance (or mock for testing)
            max_alerts_per_hour: Maximum alerts per user per hour
            batch_window_seconds: Window for batching alerts
        """
        self.bot = telegram_bot
        self.max_alerts_per_hour = max_alerts_per_hour
        self.batch_window_seconds = batch_window_seconds

        # Rate limiting tracking
        self._user_alert_counts: Dict[str, List[datetime]] = defaultdict(list)

        # Batching queues
        self._pending_alerts: Dict[str, List[Alert]] = defaultdict(list)
        self._batch_tasks: Dict[str, asyncio.Task] = {}

        # HTTP client for webhooks
        self._http_client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._http_client = httpx.AsyncClient(timeout=10.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._http_client:
            await self._http_client.aclose()

    def _is_rate_limited(self, user_id: str) -> bool:
        """
        Check if user is rate limited.

        Args:
            user_id: User ID to check

        Returns:
            True if rate limited, False otherwise
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=1)

        # Clean old entries
        self._user_alert_counts[user_id] = [
            ts for ts in self._user_alert_counts[user_id] if ts > cutoff
        ]

        # Check count
        return len(self._user_alert_counts[user_id]) >= self.max_alerts_per_hour

    def _record_alert_sent(self, user_id: str) -> None:
        """Record that an alert was sent to user."""
        self._user_alert_counts[user_id].append(datetime.now(timezone.utc))

    async def send_telegram_alert(
        self,
        chat_id: str,
        alert: Alert,
    ) -> bool:
        """
        Send alert via Telegram.

        Args:
            chat_id: Telegram chat ID
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        # Check if user is in quiet hours (future enhancement)
        # For now, skip quiet hours check

        # Format message
        message = self._format_telegram_message(alert)

        try:
            # Send message
            sent_message = await self.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode="HTML",
            )

            # Update alert with message ID
            alert.sent_to_telegram = True
            alert.telegram_message_id = str(sent_message.message_id)

            logger.info(
                "telegram_alert_sent",
                chat_id=chat_id,
                alert_type=alert.alert_type.value,
                message_id=sent_message.message_id,
            )

            return True

        except TelegramError as e:
            logger.error(
                "telegram_alert_failed",
                chat_id=chat_id,
                alert_type=alert.alert_type.value,
                error=str(e),
            )
            return False

    async def send_webhook_alert(
        self,
        webhook_url: str,
        alert: Alert,
    ) -> bool:
        """
        Send alert via webhook.

        Args:
            webhook_url: Webhook URL
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        if not self._http_client:
            logger.error("webhook_alert_no_client")
            return False

        # Build payload
        payload = {
            "alert_type": alert.alert_type.value,
            "severity": alert.severity.value,
            "wallet_address": alert.wallet_address,
            "token_mint": alert.token_mint,
            "timestamp": alert.timestamp.isoformat(),
            "details": alert.details,
        }

        try:
            response = await self._http_client.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                alert.sent_to_webhook = True
                logger.info(
                    "webhook_alert_sent",
                    url=webhook_url,
                    alert_type=alert.alert_type.value,
                )
                return True
            else:
                logger.error(
                    "webhook_alert_failed",
                    url=webhook_url,
                    status_code=response.status_code,
                    response=response.text,
                )
                return False

        except Exception as e:
            logger.error(
                "webhook_alert_error",
                url=webhook_url,
                error=str(e),
            )
            return False

    async def deliver_alert(
        self,
        alert: Alert,
        chat_id: Optional[str] = None,
        webhook_url: Optional[str] = None,
        batch: bool = False,
    ) -> bool:
        """
        Deliver an alert via configured channels.

        Args:
            alert: Alert to deliver
            chat_id: Telegram chat ID (if Telegram enabled)
            webhook_url: Webhook URL (if webhook enabled)
            batch: Whether to batch this alert

        Returns:
            True if delivered successfully to at least one channel
        """
        if not alert.user_id:
            logger.warning("alert_no_user_id", alert_id=alert.id)
            return False

        # Check rate limiting
        if self._is_rate_limited(alert.user_id):
            logger.warning(
                "alert_rate_limited",
                user_id=alert.user_id,
                alert_type=alert.alert_type.value,
            )
            return False

        success = False

        # Telegram delivery
        if chat_id:
            if batch:
                # Add to batch queue
                await self._add_to_batch(alert.user_id, chat_id, alert)
            else:
                # Send immediately
                if await self.send_telegram_alert(chat_id, alert):
                    self._record_alert_sent(alert.user_id)
                    success = True

        # Webhook delivery
        if webhook_url:
            if await self.send_webhook_alert(webhook_url, alert):
                success = True

        return success

    async def _add_to_batch(
        self,
        user_id: str,
        chat_id: str,
        alert: Alert,
    ) -> None:
        """
        Add alert to batch queue.

        Args:
            user_id: User ID
            chat_id: Telegram chat ID
            alert: Alert to batch
        """
        batch_key = f"{user_id}:{chat_id}"

        self._pending_alerts[batch_key].append(alert)

        # Start batch timer if not already running
        if batch_key not in self._batch_tasks:
            self._batch_tasks[batch_key] = asyncio.create_task(
                self._send_batched_alerts(user_id, chat_id, batch_key)
            )

    async def _send_batched_alerts(
        self,
        user_id: str,
        chat_id: str,
        batch_key: str,
    ) -> None:
        """
        Send batched alerts after waiting for batch window.

        Args:
            user_id: User ID
            chat_id: Telegram chat ID
            batch_key: Batch queue key
        """
        # Wait for batch window
        await asyncio.sleep(self.batch_window_seconds)

        # Get pending alerts
        alerts = self._pending_alerts.pop(batch_key, [])
        self._batch_tasks.pop(batch_key, None)

        if not alerts:
            return

        # Build combined message
        message = self._format_batched_message(alerts)

        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode="HTML",
            )

            self._record_alert_sent(user_id)

            logger.info(
                "batched_alerts_sent",
                user_id=user_id,
                chat_id=chat_id,
                alert_count=len(alerts),
            )

        except TelegramError as e:
            logger.error(
                "batched_alerts_failed",
                user_id=user_id,
                chat_id=chat_id,
                error=str(e),
            )

    def _format_telegram_message(self, alert: Alert) -> str:
        """
        Format alert as Telegram message with HTML.

        Args:
            alert: Alert to format

        Returns:
            Formatted HTML message
        """
        # Use severity emoji
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.HIGH: "🔴",
            AlertSeverity.CRITICAL: "🚨",
        }

        emoji = severity_emoji.get(alert.severity, "📢")

        # Get base message from alert
        base_message = alert.get_message()

        # Add severity and timestamp
        timestamp_str = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

        message = (
            f"{emoji} <b>{alert.severity.value.upper()}</b>\n\n"
            f"{base_message}\n\n"
            f"🕐 {timestamp_str}"
        )

        return message

    def _format_batched_message(self, alerts: List[Alert]) -> str:
        """
        Format multiple alerts as a single batched message.

        Args:
            alerts: Alerts to batch

        Returns:
            Formatted HTML message
        """
        header = f"📬 <b>Alert Batch</b> ({len(alerts)} alerts)\n\n"

        alert_lines = []
        for i, alert in enumerate(alerts, 1):
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.HIGH: "🔴",
                AlertSeverity.CRITICAL: "🚨",
            }
            emoji = severity_emoji.get(alert.severity, "📢")

            alert_lines.append(
                f"{i}. {emoji} {alert.alert_type.value}\n"
                f"   {alert.severity.value} • {alert.timestamp.strftime('%H:%M:%S')}"
            )

        return header + "\n\n".join(alert_lines)

    async def get_delivery_stats(self) -> Dict:
        """
        Get notification delivery statistics.

        Returns:
            Dict with delivery stats
        """
        total_alerts = sum(len(counts) for counts in self._user_alert_counts.values())
        pending_batches = sum(len(alerts) for alerts in self._pending_alerts.values())

        return {
            "total_alerts_sent_last_hour": total_alerts,
            "unique_users": len(self._user_alert_counts),
            "pending_batched_alerts": pending_batches,
            "active_batch_tasks": len(self._batch_tasks),
            "max_alerts_per_hour": self.max_alerts_per_hour,
            "batch_window_seconds": self.batch_window_seconds,
        }
