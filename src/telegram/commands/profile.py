"""
/profile command handler for Telegram bot.

Shows wallet profile evolution and risk analysis.
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

from ...monitoring.profiles import ProfileTracker

logger = structlog.get_logger()


async def handle_profile_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    profile_tracker: ProfileTracker,
) -> None:
    """
    Handle /profile command to show wallet profile evolution.

    Usage:
        /profile <wallet_address> [days]

    Args:
        update: Telegram update
        context: Telegram context
        profile_tracker: ProfileTracker instance
    """
    if not update.message or not context.args:
        await update.message.reply_text(
            "❌ Usage: /profile <wallet_address> [days]\n\n"
            "Example: /profile 7xKXtg2CW87d... 30\n"
            "Days is optional (default: 30)"
        )
        return

    wallet_address = context.args[0]
    lookback_days = int(context.args[1]) if len(context.args) > 1 else 30

    # Validate address
    try:
        if not (32 <= len(wallet_address) <= 44):
            raise ValueError("Invalid address format")
    except Exception as e:
        await update.message.reply_text(
            f"❌ Invalid wallet address: {str(e)}\n"
            "Please provide a valid Solana address (32-44 characters)."
        )
        return

    user_id = str(update.effective_user.id)

    try:
        # Get profile evolution
        evolution = await profile_tracker.get_profile_evolution(
            wallet=wallet_address,
            lookback_days=lookback_days,
        )

        if not evolution:
            await update.message.reply_text(
                f"❌ No profile history found for wallet.\n\n"
                f"👛 Wallet: {wallet_address[:8]}...{wallet_address[-6:]}\n\n"
                f"This wallet may not have been analyzed yet."
            )
            return

        # Build profile message
        message = _format_profile_evolution(evolution, lookback_days)

        await update.message.reply_text(message)

        logger.info(
            "profile_command_success",
            user_id=user_id,
            wallet=wallet_address,
            lookback_days=lookback_days,
        )

    except Exception as e:
        logger.error(
            "profile_command_error",
            user_id=user_id,
            wallet=wallet_address,
            error=str(e),
        )
        await update.message.reply_text(
            f"❌ Error retrieving profile: {str(e)}"
        )


def _format_profile_evolution(evolution, lookback_days: int) -> str:
    """Format ProfileEvolution as a readable message."""
    wallet_short = f"{evolution.wallet[:8]}...{evolution.wallet[-6:]}"

    # Header
    message = [
        "📊 Wallet Profile Analysis\n",
        f"👛 Wallet: {wallet_short}",
        f"📅 Period: Last {lookback_days} days\n",
    ]

    # Current profile
    message.append("🔍 Current Profile:")
    message.append(f"   Archetype: {evolution.current_archetype}")
    message.append(f"   Risk Score: {evolution.current_risk_score:.2f}")

    # Profile velocity
    velocity_emoji = "🔥" if evolution.profile_velocity > 0.2 else "📈" if evolution.profile_velocity > 0.1 else "📊"
    message.append(f"   Profile Velocity: {velocity_emoji} {evolution.profile_velocity:.3f}")

    # Risk trend
    risk_trend = evolution.get_risk_trend()
    trend_emoji = "📈" if risk_trend == "increasing" else "📉" if risk_trend == "decreasing" else "➡️"
    message.append(f"   Risk Trend: {trend_emoji} {risk_trend}\n")

    # Archetype history
    if evolution.archetype_transitions:
        message.append("🔄 Archetype Transitions:")
        for timestamp, from_arch, to_arch in evolution.archetype_transitions[-5:]:
            date_str = timestamp.strftime("%Y-%m-%d")
            message.append(f"   {date_str}: {from_arch} → {to_arch}")

        # Duration in current archetype
        duration = evolution.get_archetype_duration(evolution.current_archetype)
        message.append(f"\n   ⏱️ Time in current: {duration:.1f} days")
    else:
        message.append("🔄 No archetype transitions recorded")

    message.append("")

    # Snapshot count
    message.append(f"📸 Profile Snapshots: {len(evolution.snapshots)}")

    # Recent snapshots
    if evolution.snapshots:
        message.append("\n📈 Recent Risk Scores:")
        for snapshot in evolution.snapshots[-5:]:
            date_str = snapshot.timestamp.strftime("%Y-%m-%d")
            message.append(f"   {date_str}: {snapshot.risk_score:.2f}")

    # Interpretation
    message.append("\n💡 Interpretation:")
    if evolution.profile_velocity > 0.2:
        message.append("   ⚠️ High volatility - profile changing rapidly")
    elif evolution.profile_velocity > 0.1:
        message.append("   📊 Moderate activity - some profile changes")
    else:
        message.append("   ✅ Stable - consistent behavior pattern")

    if evolution.current_risk_score > 0.8:
        message.append("   🚨 High risk - exercise caution")
    elif evolution.current_risk_score > 0.5:
        message.append("   ⚠️ Moderate risk - monitor closely")
    else:
        message.append("   ✅ Low risk - standard behavior")

    return "\n".join(message)


async def handle_profile_stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    profile_tracker: ProfileTracker,
) -> None:
    """
    Handle /profile_stats command to show global profile statistics.

    Args:
        update: Telegram update
        context: Telegram context
        profile_tracker: ProfileTracker instance
    """
    if not update.message:
        return

    user_id = str(update.effective_user.id)

    try:
        # Get archetype distribution
        archetype_dist = await profile_tracker.get_archetype_distribution()

        # Get risk score statistics
        risk_stats = await profile_tracker.get_risk_score_stats()

        # Build message
        message = ["📊 Global Profile Statistics\n"]

        if archetype_dist:
            message.append("🏷️ Archetype Distribution:")
            total = sum(archetype_dist.values())
            for archetype, count in sorted(
                archetype_dist.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                pct = (count / total * 100) if total > 0 else 0
                message.append(f"   {archetype}: {count} ({pct:.1f}%)")
        else:
            message.append("🏷️ No archetype data available")

        message.append("")

        if risk_stats:
            message.append("📈 Risk Score Statistics:")
            message.append(f"   Mean: {risk_stats.get('mean', 0):.2f}")
            message.append(f"   Median: {risk_stats.get('median', 0):.2f}")
            message.append(f"   Std Dev: {risk_stats.get('std', 0):.2f}")
            message.append(f"   Range: {risk_stats.get('min', 0):.2f} - {risk_stats.get('max', 1):.2f}")
        else:
            message.append("📈 No risk score data available")

        await update.message.reply_text("\n".join(message))

        logger.info("profile_stats_command_success", user_id=user_id)

    except Exception as e:
        logger.error(
            "profile_stats_command_error",
            user_id=user_id,
            error=str(e),
        )
        await update.message.reply_text(
            f"❌ Error retrieving profile statistics: {str(e)}"
        )
