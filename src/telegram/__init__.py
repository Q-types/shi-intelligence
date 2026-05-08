"""
Telegram Bot Interface for SHI.

Commands:
- /analyze <mint> - Full token analysis
- /summary <mint> - Quick overview
- /top_holders <mint> - Holder breakdown
- /risk <mint> - Risk scores only
- /graph <mint> - Funding graph visualization
- /history <mint> - Historical comparison
- /watch <wallet> <token> - Add wallet to watchlist
- /unwatch <wallet> <token> - Remove wallet from watchlist
- /watchlist - View your watchlist
- /alerts [config] - Configure alert settings
- /profile <wallet> <token> - View wallet profile evolution
- /explain <token> [verbose] - Risk score explanation with SHAP breakdown
- /forecast <token> [days] - Capital flow forecast with confidence intervals

Latency target: <= 30 seconds
"""

# Check if telegram library is available
try:
    import telegram as _telegram_check  # noqa: F401
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# Only import telegram-dependent modules if library is available
if TELEGRAM_AVAILABLE:
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
    from .commands import (
        handle_watch_command,
        handle_unwatch_command,
        handle_watchlist_command,
        handle_alerts_command,
        handle_profile_command,
    )
    from .commands.explain import (
        handle_explain_command,
        handle_explain_regime_command,
    )
    from .commands.forecast import (
        handle_forecast_command,
        handle_forecast_backtest_command,
    )
    from .notifications import NotificationDelivery

    __all__ = [
        "TELEGRAM_AVAILABLE",
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
        # Sprint 3 Commands
        "handle_watch_command",
        "handle_unwatch_command",
        "handle_watchlist_command",
        "handle_alerts_command",
        "handle_profile_command",
        # Sprint 4 Commands
        "handle_explain_command",
        "handle_explain_regime_command",
        "handle_forecast_command",
        "handle_forecast_backtest_command",
        # Notifications
        "NotificationDelivery",
    ]
else:
    # Telegram not available - export minimal interface
    __all__ = ["TELEGRAM_AVAILABLE"]
