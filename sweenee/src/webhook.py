"""Telegram Webhook Service - Send alerts and summaries to Telegram."""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from .cache import get_cache
from .alerts import WhaleAlert, AlertType

logger = structlog.get_logger()

# Telegram Bot API
TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}"
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Exponential backoff base


@dataclass
class WebhookResult:
    """Result of a webhook send attempt."""

    success: bool
    message_hash: str
    error: str | None = None
    retry_count: int = 0


class TelegramWebhook:
    """Service for sending messages to Telegram with idempotency."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        """Initialize Telegram webhook.

        Args:
            bot_token: Telegram bot token (or TELEGRAM_BOT_TOKEN env var)
            chat_id: Target chat ID (or TELEGRAM_CHAT_ID env var)
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.cache = get_cache()
        self._client: httpx.AsyncClient | None = None

    @property
    def is_configured(self) -> bool:
        """Check if webhook is properly configured."""
        return bool(self.bot_token and self.chat_id)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _generate_hash(self, message_type: str, content: str) -> str:
        """Generate unique hash for idempotency check."""
        # Include date to allow same message on different days
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        data = f"{message_type}:{date_str}:{content}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    async def send_message(
        self,
        text: str,
        message_type: str = "general",
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> WebhookResult:
        """Send a message to Telegram with idempotency.

        Args:
            text: Message text to send
            message_type: Type identifier for idempotency
            parse_mode: Telegram parse mode (HTML, Markdown, MarkdownV2)
            disable_notification: Send silently

        Returns:
            WebhookResult with success status
        """
        if not self.is_configured:
            return WebhookResult(
                success=False,
                message_hash="",
                error="Telegram webhook not configured",
            )

        # Generate hash for idempotency
        message_hash = self._generate_hash(message_type, text[:200])

        # Check if already sent
        if self.cache.was_webhook_sent(message_hash):
            logger.debug("webhook_skipped_duplicate", hash=message_hash[:8])
            return WebhookResult(
                success=True,
                message_hash=message_hash,
                error="Already sent (idempotency)",
            )

        # Log attempt
        payload_preview = text[:100] + "..." if len(text) > 100 else text
        if not self.cache.log_webhook(message_hash, message_type, payload_preview, "pending"):
            # Already in log - another process is sending
            return WebhookResult(
                success=True,
                message_hash=message_hash,
                error="Already in progress",
            )

        # Send with retries
        result = await self._send_with_retry(text, parse_mode, disable_notification, message_hash)

        # Update status
        self.cache.update_webhook_status(
            message_hash,
            "sent" if result.success else "failed",
            increment_retry=not result.success,
        )

        return result

    async def _send_with_retry(
        self,
        text: str,
        parse_mode: str,
        disable_notification: bool,
        message_hash: str,
    ) -> WebhookResult:
        """Send message with exponential backoff retry."""
        url = f"{TELEGRAM_API_BASE.format(token=self.bot_token)}/sendMessage"

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                client = await self._get_client()
                response = await client.post(url, json=payload)
                response.raise_for_status()

                result = response.json()
                if result.get("ok"):
                    logger.info("webhook_sent", hash=message_hash[:8])
                    return WebhookResult(
                        success=True,
                        message_hash=message_hash,
                        retry_count=attempt,
                    )
                else:
                    last_error = result.get("description", "Unknown error")

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:100]}"
                logger.warning(
                    "webhook_http_error",
                    attempt=attempt + 1,
                    error=last_error,
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "webhook_error",
                    attempt=attempt + 1,
                    error=last_error,
                )

            # Exponential backoff
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                await asyncio.sleep(delay)

        return WebhookResult(
            success=False,
            message_hash=message_hash,
            error=last_error,
            retry_count=MAX_RETRIES,
        )

    async def send_alert(self, alert: WhaleAlert, wallet_label: str | None = None) -> WebhookResult:
        """Send a whale alert to Telegram."""
        wallet_display = wallet_label or alert.wallet_address[:8] + "..."

        emoji = alert.emoji
        amount_str = f"{alert.amount:,.0f}"

        if alert.alert_type == AlertType.LARGE_BUY:
            text = f"{emoji} <b>WHALE BUY ALERT</b>\n\n"
            text += f"Wallet: <code>{wallet_display}</code>\n"
            text += f"Amount: <b>+{amount_str} SWEENEE</b>"
        elif alert.alert_type == AlertType.LARGE_SELL:
            text = f"{emoji} <b>WHALE SELL ALERT</b>\n\n"
            text += f"Wallet: <code>{wallet_display}</code>\n"
            text += f"Amount: <b>-{amount_str} SWEENEE</b>"
        elif alert.alert_type == AlertType.WHALE_EXIT:
            text = f"{emoji} <b>WHALE EXIT</b>\n\n"
            text += f"Wallet: <code>{wallet_display}</code>\n"
            text += f"Sold: <b>{amount_str} SWEENEE</b>\n"
            text += "Position closed"
        else:
            text = f"{emoji} <b>WHALE ALERT</b>\n\n"
            text += f"Wallet: <code>{wallet_display}</code>\n"
            text += f"Amount: <b>{amount_str} SWEENEE</b>"

        return await self.send_message(
            text,
            message_type=f"alert_{alert.alert_type.value}",
            disable_notification=False,
        )

    async def send_daily_summary(self, summary_text: str) -> WebhookResult:
        """Send daily summary to Telegram."""
        return await self.send_message(
            summary_text,
            message_type="daily_summary",
            disable_notification=True,
        )

    def send_message_sync(self, text: str, message_type: str = "general") -> WebhookResult:
        """Synchronous wrapper for send_message."""
        return asyncio.run(self.send_message(text, message_type))

    def send_alert_sync(self, alert: WhaleAlert, wallet_label: str | None = None) -> WebhookResult:
        """Synchronous wrapper for send_alert."""
        return asyncio.run(self.send_alert(alert, wallet_label))


# Singleton instance
_webhook: TelegramWebhook | None = None


def get_webhook() -> TelegramWebhook:
    """Get or create webhook singleton."""
    global _webhook
    if _webhook is None:
        _webhook = TelegramWebhook()
    return _webhook
