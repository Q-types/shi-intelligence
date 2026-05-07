"""
Telegram Bot Interface for SHI.

Commands:
- /analyze <mint> - Full token analysis
- /summary <mint> - Quick overview
- /top_holders <mint> - Holder breakdown
- /risk <mint> - Risk scores only
- /graph <mint> - Funding graph visualization
- /history <mint> - Historical comparison

Latency target: <= 30 seconds
"""

from .bot import SHIBot
from .handlers import register_handlers
from .formatters import format_risk_report, format_holder_summary
from .security import (
    SecurityMiddleware,
    SecurityConfig,
    InputValidator,
    AuthorizationManager,
    AbuseDetector,
    AuditLogger,
    UserRole,
)

__all__ = [
    "SHIBot",
    "register_handlers",
    "format_risk_report",
    "format_holder_summary",
    # Security
    "SecurityMiddleware",
    "SecurityConfig",
    "InputValidator",
    "AuthorizationManager",
    "AbuseDetector",
    "AuditLogger",
    "UserRole",
]
