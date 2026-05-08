"""
Telegram Bot Security.

Implements input validation, authorization, abuse prevention,
and audit logging for the Telegram interface.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Awaitable, Any

import structlog

logger = structlog.get_logger()


class UserRole(Enum):
    """User roles for authorization."""

    ANONYMOUS = "anonymous"
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"
    BANNED = "banned"


@dataclass
class UserContext:
    """Security context for a user."""

    user_id: int
    username: str | None
    role: UserRole
    request_count: int = 0
    last_request: datetime | None = None
    warnings: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEntry:
    """Audit log entry."""

    timestamp: datetime
    user_id: int
    username: str | None
    command: str
    args: str | None
    success: bool
    error: str | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    """Configuration for security features."""

    # User lists
    admin_user_ids: set[int] = field(default_factory=set)
    premium_user_ids: set[int] = field(default_factory=set)
    banned_user_ids: set[int] = field(default_factory=set)
    whitelisted_user_ids: set[int] | None = None  # None = no whitelist

    # Rate limiting
    max_requests_per_minute: int = 10
    max_requests_per_hour: int = 100

    # Abuse detection
    warning_threshold: int = 3  # Warnings before auto-ban
    spam_interval_seconds: float = 1.0  # Min time between requests
    max_message_length: int = 1000

    # Input validation
    allowed_commands: set[str] = field(default_factory=lambda: {
        "/start", "/help", "/analyze", "/watch", "/unwatch",
        "/status", "/metrics", "/settings",
    })

    # Admin-only commands
    admin_commands: set[str] = field(default_factory=lambda: {
        "/ban", "/unban", "/broadcast", "/stats", "/reload",
    })


class InputValidator:
    """
    Validates and sanitizes user input.

    Prevents injection attacks and malformed input.
    """

    # Solana address pattern (base58)
    SOLANA_ADDRESS_PATTERN = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")

    # Command pattern
    COMMAND_PATTERN = re.compile(r"^/[a-z_]+$")

    # Dangerous patterns to reject
    DANGEROUS_PATTERNS = [
        re.compile(r"<script", re.IGNORECASE),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"\x00"),  # Null bytes
    ]

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig()

    def validate_command(self, text: str) -> tuple[str | None, str | None]:
        """
        Validate and parse command from text.

        Returns:
            (command, args) or (None, error_message)
        """
        if not text:
            return None, "Empty message"

        # Check length
        if len(text) > self.config.max_message_length:
            return None, f"Message too long (max {self.config.max_message_length} chars)"

        # Check dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(text):
                return None, "Invalid characters in message"

        # Parse command
        parts = text.strip().split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else None

        # Validate command format
        if not self.COMMAND_PATTERN.match(command):
            return None, f"Invalid command format: {command}"

        # Check if command is allowed
        if command not in self.config.allowed_commands:
            if command not in self.config.admin_commands:
                return None, f"Unknown command: {command}"

        return command, args

    def validate_solana_address(self, address: str) -> tuple[str | None, str | None]:
        """
        Validate Solana address.

        Returns:
            (address, None) or (None, error_message)
        """
        if not address:
            return None, "Address is required"

        address = address.strip()

        if not self.SOLANA_ADDRESS_PATTERN.match(address):
            return None, "Invalid Solana address format"

        return address, None

    def sanitize_text(self, text: str) -> str:
        """Sanitize text for safe display."""
        if not text:
            return ""

        # Remove control characters
        sanitized = "".join(
            char for char in text
            if ord(char) >= 32 or char in "\n\t"
        )

        # Truncate if too long
        if len(sanitized) > self.config.max_message_length:
            sanitized = sanitized[:self.config.max_message_length] + "..."

        return sanitized


class AuthorizationManager:
    """
    Manages user authorization and roles.

    Supports:
    - Role-based access control
    - User whitelist/blacklist
    - Dynamic role updates
    """

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig()
        self._user_contexts: dict[int, UserContext] = {}
        self._lock = asyncio.Lock()

    async def get_user_context(
        self,
        user_id: int,
        username: str | None = None,
    ) -> UserContext:
        """Get or create user context."""
        async with self._lock:
            if user_id not in self._user_contexts:
                role = self._determine_role(user_id)
                self._user_contexts[user_id] = UserContext(
                    user_id=user_id,
                    username=username,
                    role=role,
                )
            else:
                # Update username if provided
                if username:
                    self._user_contexts[user_id].username = username

            return self._user_contexts[user_id]

    def _determine_role(self, user_id: int) -> UserRole:
        """Determine user role based on configuration."""
        if user_id in self.config.banned_user_ids:
            return UserRole.BANNED

        if user_id in self.config.admin_user_ids:
            return UserRole.ADMIN

        if user_id in self.config.premium_user_ids:
            return UserRole.PREMIUM

        # Check whitelist
        if self.config.whitelisted_user_ids is not None:
            if user_id not in self.config.whitelisted_user_ids:
                return UserRole.ANONYMOUS

        return UserRole.USER

    async def is_authorized(
        self,
        user_id: int,
        command: str,
        username: str | None = None,
    ) -> tuple[bool, str | None]:
        """
        Check if user is authorized for command.

        Returns:
            (authorized, error_message)
        """
        context = await self.get_user_context(user_id, username)

        # Check if banned
        if context.role == UserRole.BANNED:
            logger.warning("banned_user_attempt", user_id=user_id, command=command)
            return False, "You are banned from using this bot"

        # Check admin commands
        if command in self.config.admin_commands:
            if context.role != UserRole.ADMIN:
                logger.warning(
                    "unauthorized_admin_command",
                    user_id=user_id,
                    command=command,
                )
                return False, "This command requires admin privileges"

        # Check whitelist
        if self.config.whitelisted_user_ids is not None:
            if context.role == UserRole.ANONYMOUS:
                return False, "You are not authorized to use this bot"

        return True, None

    async def ban_user(self, user_id: int, reason: str = "") -> None:
        """Ban a user."""
        async with self._lock:
            self.config.banned_user_ids.add(user_id)
            if user_id in self._user_contexts:
                self._user_contexts[user_id].role = UserRole.BANNED

            logger.warning("user_banned", user_id=user_id, reason=reason)

    async def unban_user(self, user_id: int) -> None:
        """Unban a user."""
        async with self._lock:
            self.config.banned_user_ids.discard(user_id)
            if user_id in self._user_contexts:
                self._user_contexts[user_id].role = self._determine_role(user_id)

            logger.info("user_unbanned", user_id=user_id)


class AbuseDetector:
    """
    Detects and handles abusive behavior.

    Implements:
    - Spam detection
    - Rate limit enforcement
    - Automatic warnings and bans
    """

    def __init__(
        self,
        config: SecurityConfig | None = None,
        auth_manager: AuthorizationManager | None = None,
    ):
        self.config = config or SecurityConfig()
        self.auth_manager = auth_manager or AuthorizationManager(self.config)
        self._request_times: dict[int, list[float]] = {}
        self._lock = asyncio.Lock()

    async def check_abuse(
        self,
        user_id: int,
        username: str | None = None,
    ) -> tuple[bool, str | None]:
        """
        Check for abusive behavior.

        Returns:
            (allowed, warning_message)
        """
        async with self._lock:
            now = time.time()
            context = await self.auth_manager.get_user_context(user_id, username)

            # Initialize request tracking
            if user_id not in self._request_times:
                self._request_times[user_id] = []

            # Clean old requests
            minute_ago = now - 60
            hour_ago = now - 3600
            self._request_times[user_id] = [
                t for t in self._request_times[user_id]
                if t > hour_ago
            ]

            requests = self._request_times[user_id]

            # Check spam (too fast)
            if context.last_request:
                time_since_last = now - context.last_request.timestamp()
                if time_since_last < self.config.spam_interval_seconds:
                    context.warnings += 1
                    if context.warnings >= self.config.warning_threshold:
                        await self.auth_manager.ban_user(user_id, "Spam")
                        return False, "You have been banned for spamming"
                    return False, "Please slow down"

            # Check rate limits
            requests_last_minute = sum(1 for t in requests if t > minute_ago)
            if requests_last_minute >= self.config.max_requests_per_minute:
                return False, "Rate limit exceeded. Please wait a moment."

            requests_last_hour = len(requests)
            if requests_last_hour >= self.config.max_requests_per_hour:
                return False, "Hourly rate limit exceeded. Please try again later."

            # Record request
            self._request_times[user_id].append(now)
            context.request_count += 1
            context.last_request = datetime.now(timezone.utc)

            return True, None


class AuditLogger:
    """
    Audit logging for all bot commands.

    Maintains an audit trail for security and compliance.
    """

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._entries: list[AuditEntry] = []
        self._lock = asyncio.Lock()

    async def log(
        self,
        user_id: int,
        username: str | None,
        command: str,
        args: str | None = None,
        success: bool = True,
        error: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a command execution."""
        async with self._lock:
            entry = AuditEntry(
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                username=username,
                command=command,
                args=args,
                success=success,
                error=error,
                duration_ms=duration_ms,
                metadata=metadata or {},
            )

            self._entries.append(entry)

            # Trim if over size
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]

            # Also log to structlog
            logger.info(
                "audit_command",
                user_id=user_id,
                username=username,
                command=command,
                success=success,
                error=error,
                duration_ms=duration_ms,
            )

    async def get_entries(
        self,
        user_id: int | None = None,
        command: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit log entries."""
        async with self._lock:
            entries = self._entries

            if user_id is not None:
                entries = [e for e in entries if e.user_id == user_id]

            if command is not None:
                entries = [e for e in entries if e.command == command]

            if since is not None:
                entries = [e for e in entries if e.timestamp >= since]

            return entries[-limit:]

    async def get_user_stats(self, user_id: int) -> dict[str, Any]:
        """Get statistics for a user."""
        async with self._lock:
            user_entries = [e for e in self._entries if e.user_id == user_id]

            if not user_entries:
                return {"commands": 0}

            return {
                "commands": len(user_entries),
                "successful": sum(1 for e in user_entries if e.success),
                "failed": sum(1 for e in user_entries if not e.success),
                "first_seen": min(e.timestamp for e in user_entries).isoformat(),
                "last_seen": max(e.timestamp for e in user_entries).isoformat(),
                "commands_used": list(set(e.command for e in user_entries)),
            }


class SecurityMiddleware:
    """
    Middleware that combines all security features.

    Use this to wrap command handlers.
    """

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig()
        self.validator = InputValidator(self.config)
        self.auth_manager = AuthorizationManager(self.config)
        self.abuse_detector = AbuseDetector(self.config, self.auth_manager)
        self.audit_logger = AuditLogger()

    async def process(
        self,
        user_id: int,
        username: str | None,
        text: str,
        handler: Callable[[str, str | None], Awaitable[str]],
    ) -> str:
        """
        Process a command through security middleware.

        Args:
            user_id: Telegram user ID
            username: Telegram username
            text: Raw message text
            handler: Async handler function(command, args) -> response

        Returns:
            Response text
        """
        start_time = time.time()

        # Validate input
        command, args_or_error = self.validator.validate_command(text)
        if command is None:
            await self.audit_logger.log(
                user_id, username, text[:50], success=False, error=args_or_error
            )
            return f"Error: {args_or_error}"

        # Check authorization
        authorized, auth_error = await self.auth_manager.is_authorized(
            user_id, command, username
        )
        if not authorized:
            await self.audit_logger.log(
                user_id, username, command, args_or_error, success=False, error=auth_error
            )
            return f"Access denied: {auth_error}"

        # Check abuse
        allowed, abuse_warning = await self.abuse_detector.check_abuse(user_id, username)
        if not allowed:
            await self.audit_logger.log(
                user_id, username, command, args_or_error, success=False, error=abuse_warning
            )
            return abuse_warning or "Rate limit exceeded"

        # Execute handler
        try:
            response = await handler(command, args_or_error)
            duration_ms = (time.time() - start_time) * 1000

            await self.audit_logger.log(
                user_id, username, command, args_or_error,
                success=True, duration_ms=duration_ms
            )

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)

            await self.audit_logger.log(
                user_id, username, command, args_or_error,
                success=False, error=error_msg, duration_ms=duration_ms
            )

            logger.exception("command_handler_error", command=command, error=error_msg)
            return "An error occurred processing your request. Please try again later."
