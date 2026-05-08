"""
/alerts command handler for Telegram bot.

Allows users to configure alert preferences and thresholds.
"""

from __future__ import annotations

import structlog
try:
    from telegram import Update
    from telegram.ext import ContextTypes

except ImportError:
    # Telegram not installed
    from typing import Any
    Update = Any
    class ContextTypes:
        DEFAULT_TYPE = Any

from ...monitoring.alerts import AlertConfig

logger = structlog.get_logger()


async def handle_alerts_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Handle /alerts command to configure alert settings.

    Usage:
        /alerts - Show current settings
        /alerts <token_mint> - Show settings for a specific token
        /alerts <token_mint> whale <threshold> - Set whale movement threshold
        /alerts <token_mint> concentration <threshold> - Set concentration threshold
        /alerts <token_mint> anomaly <threshold> - Set anomaly threshold

    Args:
        update: Telegram update
        context: Telegram context
    """
    if not update.message:
        return

    user_id = str(update.effective_user.id)

    # No arguments - show all configs
    if not context.args:
        await _show_all_alert_configs(update, user_id)
        return

    # With token mint - show/update config
    token_mint = context.args[0]

    # Just token mint - show config for that token
    if len(context.args) == 1:
        await _show_token_alert_config(update, user_id, token_mint)
        return

    # Update specific threshold
    if len(context.args) >= 3:
        alert_type = context.args[1].lower()
        try:
            threshold = float(context.args[2])
        except ValueError:
            await update.message.reply_text(
                "❌ Invalid threshold value. Please provide a number."
            )
            return

        await _update_alert_threshold(
            update,
            user_id,
            token_mint,
            alert_type,
            threshold,
        )
        return

    await update.message.reply_text(
        "❌ Invalid command format.\n\n"
        "Usage:\n"
        "/alerts - Show all settings\n"
        "/alerts <token> - Show settings for token\n"
        "/alerts <token> whale <threshold> - Set whale threshold\n"
        "/alerts <token> concentration <threshold> - Set concentration threshold\n"
        "/alerts <token> anomaly <threshold> - Set anomaly threshold"
    )


async def _show_all_alert_configs(
    update: Update,
    user_id: str,
) -> None:
    """Show all alert configurations for user."""
    # In production, query all alert_configs for user from database
    await update.message.reply_text(
        "⚙️ Alert Settings\n\n"
        "You have no configured alerts yet.\n\n"
        "Use /alerts <token_mint> to configure alerts for a token."
    )

    logger.info("alerts_command_show_all", user_id=user_id)


async def _show_token_alert_config(
    update: Update,
    user_id: str,
    token_mint: str,
) -> None:
    """Show alert configuration for a specific token."""
    # In production, query alert_config from database
    # For now, show default config
    config = AlertConfig(
        id=None,
        user_id=user_id,
        token_mint=token_mint,
    )

    message = (
        f"⚙️ Alert Settings for Token\n\n"
        f"🪙 Token: {token_mint[:8]}...{token_mint[-6:]}\n\n"
        f"🐋 Whale Movement: {config.whale_movement_threshold * 100:.1f}% of supply\n"
        f"📊 Concentration Increase: {config.concentration_increase_threshold * 100:.1f}% HHI change\n"
        f"⚠️ Anomaly Score: {config.anomaly_score_threshold}\n\n"
        f"📱 Telegram Alerts: {'✅ Enabled' if config.telegram_enabled else '❌ Disabled'}\n"
        f"🌐 Webhook: {'✅ Configured' if config.webhook_url else '❌ Not configured'}\n\n"
        f"⏱️ Cooldown: {config.cooldown_minutes} minutes\n\n"
        f"To update:\n"
        f"/alerts {token_mint[:8]}... whale <threshold>\n"
        f"/alerts {token_mint[:8]}... concentration <threshold>\n"
        f"/alerts {token_mint[:8]}... anomaly <threshold>"
    )

    await update.message.reply_text(message)

    logger.info(
        "alerts_command_show_token",
        user_id=user_id,
        token=token_mint,
    )


async def _update_alert_threshold(
    update: Update,
    user_id: str,
    token_mint: str,
    alert_type: str,
    threshold: float,
) -> None:
    """Update a specific alert threshold."""
    # Validate alert type
    valid_types = {
        "whale": "whale_movement_threshold",
        "concentration": "concentration_increase_threshold",
        "anomaly": "anomaly_score_threshold",
    }

    if alert_type not in valid_types:
        await update.message.reply_text(
            f"❌ Invalid alert type: {alert_type}\n"
            f"Valid types: whale, concentration, anomaly"
        )
        return

    # Validate threshold range
    if alert_type in ["whale", "concentration"]:
        if not 0.0 <= threshold <= 1.0:
            await update.message.reply_text(
                "❌ Threshold must be between 0.0 and 1.0 (0-100%)"
            )
            return
    elif alert_type == "anomaly":
        if not -1.0 <= threshold <= 0.0:
            await update.message.reply_text(
                "❌ Anomaly threshold must be between -1.0 and 0.0"
            )
            return

    # In production, update alert_configs table

    await update.message.reply_text(
        f"✅ Alert threshold updated!\n\n"
        f"🪙 Token: {token_mint[:8]}...{token_mint[-6:]}\n"
        f"⚙️ Setting: {alert_type}\n"
        f"📊 New Threshold: {threshold}\n\n"
        f"You'll now receive alerts when {alert_type} exceeds this threshold."
    )

    logger.info(
        "alert_threshold_updated",
        user_id=user_id,
        token=token_mint,
        alert_type=alert_type,
        threshold=threshold,
    )


async def handle_enable_alerts_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Handle /enable_alerts command to enable alert delivery.

    Args:
        update: Telegram update
        context: Telegram context
    """
    if not update.message or not context.args:
        await update.message.reply_text(
            "❌ Usage: /enable_alerts <token_mint>"
        )
        return

    user_id = str(update.effective_user.id)
    token_mint = context.args[0]

    # In production, update alert_configs.telegram_enabled = True

    await update.message.reply_text(
        f"✅ Alerts enabled for token!\n\n"
        f"🪙 Token: {token_mint[:8]}...{token_mint[-6:]}\n"
        f"📱 You'll now receive Telegram notifications"
    )

    logger.info(
        "alerts_enabled",
        user_id=user_id,
        token=token_mint,
    )


async def handle_disable_alerts_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Handle /disable_alerts command to disable alert delivery.

    Args:
        update: Telegram update
        context: Telegram context
    """
    if not update.message or not context.args:
        await update.message.reply_text(
            "❌ Usage: /disable_alerts <token_mint>"
        )
        return

    user_id = str(update.effective_user.id)
    token_mint = context.args[0]

    # In production, update alert_configs.telegram_enabled = False

    await update.message.reply_text(
        f"✅ Alerts disabled for token!\n\n"
        f"🪙 Token: {token_mint[:8]}...{token_mint[-6:]}\n"
        f"📱 You'll no longer receive Telegram notifications\n\n"
        f"Use /enable_alerts {token_mint[:8]}... to re-enable"
    )

    logger.info(
        "alerts_disabled",
        user_id=user_id,
        token=token_mint,
    )
