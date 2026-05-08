"""
Telegram Bot Core.

Implements rate limiting, abuse detection, and command routing.
Integrates with security middleware for comprehensive protection.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

import structlog
from telegram import Update
from telegram.ext import (
    Application,
    ContextTypes,
)

from ..core.config import settings
from .security import (
    SecurityMiddleware,
    SecurityConfig,
    UserRole,
)
from ..infra.monitoring import get_metrics

logger = structlog.get_logger()


class RateLimiter:
    """Per-user rate limiting (legacy wrapper)."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[int, list[datetime]] = defaultdict(list)

    def is_allowed(self, user_id: int) -> bool:
        """Check if user is within rate limit."""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - self.window_seconds

        # Clean old requests
        self._requests[user_id] = [
            r for r in self._requests[user_id]
            if r.timestamp() > cutoff
        ]

        # Check limit
        if len(self._requests[user_id]) >= self.max_requests:
            return False

        # Record this request
        self._requests[user_id].append(now)
        return True

    def get_retry_after(self, user_id: int) -> int:
        """Get seconds until next allowed request."""
        if not self._requests[user_id]:
            return 0

        oldest = min(r.timestamp() for r in self._requests[user_id])
        return int(self.window_seconds - (datetime.now(timezone.utc).timestamp() - oldest))


class SHIBot:
    """
    Main Telegram bot for Solana Holder Intelligence.

    Features:
    - Rate limiting per user (Redis-backed for distributed deployment)
    - Abuse detection and auto-ban
    - Authorization with admin/premium roles
    - Comprehensive audit logging
    - Async processing with timeout enforcement
    """

    def __init__(
        self,
        token: str | None = None,
        rate_limit: int | None = None,
        admin_user_ids: set[int] | None = None,
        premium_user_ids: set[int] | None = None,
    ):
        self.token = token or settings.telegram_bot_token
        self.rate_limiter = RateLimiter(
            max_requests=rate_limit or settings.telegram_rate_limit_per_user
        )
        self._app: Application | None = None
        self._blacklist: set[int] = set()
        self._whitelist: set[str] = set()  # Token mints

        # Security configuration
        security_config = SecurityConfig(
            admin_user_ids=admin_user_ids or set(),
            premium_user_ids=premium_user_ids or set(),
            max_requests_per_minute=rate_limit or settings.telegram_rate_limit_per_user,
        )
        self.security = SecurityMiddleware(security_config)
        self.audit_logger = self.security.audit_logger
        self._metrics = get_metrics()

    async def start(self) -> None:
        """Start the bot."""
        if not self.token:
            raise ValueError("Telegram bot token not configured")

        logger.info("starting_telegram_bot")

        self._app = (
            Application.builder()
            .token(self.token)
            .build()
        )

        # Register handlers
        from .handlers import register_handlers
        register_handlers(self._app, self)

        # Start polling
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

        logger.info("telegram_bot_started")

    async def stop(self) -> None:
        """Stop the bot."""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("telegram_bot_stopped")

    def check_rate_limit(self, user_id: int) -> tuple[bool, int]:
        """
        Check if user is rate limited.

        Returns:
            (is_allowed, retry_after_seconds)
        """
        if user_id in self._blacklist:
            return False, 3600  # 1 hour for blacklisted

        allowed = self.rate_limiter.is_allowed(user_id)
        if not allowed:
            retry = self.rate_limiter.get_retry_after(user_id)
            logger.warning("rate_limited_user", user_id=user_id, retry_after=retry)
            return False, retry

        return True, 0

    def blacklist_user(self, user_id: int) -> None:
        """Add user to blacklist."""
        self._blacklist.add(user_id)
        logger.warning("user_blacklisted", user_id=user_id)

    def is_token_allowed(self, mint: str) -> bool:
        """Check if token is allowed (whitelist) or not blocked."""
        # If whitelist is empty, all tokens are allowed
        if not self._whitelist:
            return True
        return mint in self._whitelist

    async def get_user_stats(self, user_id: int) -> dict:
        """Get usage statistics for a user."""
        return await self.audit_logger.get_user_stats(user_id)

    async def is_admin(self, user_id: int) -> bool:
        """Check if user is an admin."""
        context = await self.security.auth_manager.get_user_context(user_id)
        return context.role == UserRole.ADMIN

    async def ban_user(self, user_id: int, reason: str = "") -> None:
        """Ban a user (admin function)."""
        await self.security.auth_manager.ban_user(user_id, reason)
        self._blacklist.add(user_id)

    async def unban_user(self, user_id: int) -> None:
        """Unban a user (admin function)."""
        await self.security.auth_manager.unban_user(user_id)
        self._blacklist.discard(user_id)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in bot handlers."""
    logger.error(
        "telegram_error",
        error=str(context.error),
        update=update.to_dict() if update else None,
    )

    if update and update.effective_message:
        await update.effective_message.reply_text(
            "An error occurred processing your request. Please try again."
        )
