"""
/watch and /unwatch command handlers for Telegram bot.

Allows users to add/remove wallets from their watchlist.
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

from ...monitoring.watcher import WalletWatcher

logger = structlog.get_logger()


async def handle_watch_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    watcher: WalletWatcher,
) -> None:
    """
    Handle /watch command to add a wallet to the watchlist.

    Usage:
        /watch <wallet_address> <token_mint> [threshold]

    Args:
        update: Telegram update
        context: Telegram context
        watcher: WalletWatcher instance
    """
    if not update.message or not context.args:
        await update.message.reply_text(
            "❌ Usage: /watch <wallet_address> <token_mint> [threshold]\n\n"
            "Example: /watch 7xKXtg2CW87d... 4k3Dyjzvzp8e... 0.05\n"
            "Threshold is optional (default: 0.05 = 5% of supply)"
        )
        return

    # Parse arguments
    if len(context.args) < 2:
        await update.message.reply_text(
            "❌ Please provide both wallet address and token mint.\n"
            "Usage: /watch <wallet_address> <token_mint> [threshold]"
        )
        return

    wallet_address = context.args[0]
    token_mint = context.args[1]
    threshold = float(context.args[2]) if len(context.args) > 2 else 0.05

    # Validate addresses
    try:
        # Basic validation (in production, use proper validators)
        if not (32 <= len(wallet_address) <= 44) or not (32 <= len(token_mint) <= 44):
            raise ValueError("Invalid address format")
    except Exception as e:
        await update.message.reply_text(
            f"❌ Invalid address format: {str(e)}\n"
            "Please provide valid Solana addresses (32-44 characters)."
        )
        return

    # Get user ID
    user_id = str(update.effective_user.id)

    try:
        # Add to watchlist
        watched = await watcher.add_watched_wallet(
            wallet=wallet_address,
            token_mint=token_mint,
            user_id=user_id,
            alert_threshold=threshold,
        )

        await update.message.reply_text(
            f"✅ Wallet added to watchlist!\n\n"
            f"👛 Wallet: {wallet_address[:8]}...{wallet_address[-6:]}\n"
            f"🪙 Token: {token_mint[:8]}...{token_mint[-6:]}\n"
            f"📊 Alert Threshold: {threshold * 100:.1f}% of supply\n"
            f"💰 Current Balance: {watched.last_balance:,.0f}\n\n"
            f"You'll receive alerts when movements exceed the threshold."
        )

        logger.info(
            "watch_command_success",
            user_id=user_id,
            wallet=wallet_address,
            token=token_mint,
        )

    except Exception as e:
        logger.error(
            "watch_command_error",
            user_id=user_id,
            wallet=wallet_address,
            error=str(e),
        )
        await update.message.reply_text(
            f"❌ Error adding wallet to watchlist: {str(e)}"
        )


async def handle_unwatch_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    watcher: WalletWatcher,
) -> None:
    """
    Handle /unwatch command to remove a wallet from the watchlist.

    Usage:
        /unwatch <wallet_address> <token_mint>

    Args:
        update: Telegram update
        context: Telegram context
        watcher: WalletWatcher instance
    """
    if not update.message or not context.args:
        await update.message.reply_text(
            "❌ Usage: /unwatch <wallet_address> <token_mint>\n\n"
            "Example: /unwatch 7xKXtg2CW87d... 4k3Dyjzvzp8e..."
        )
        return

    # Parse arguments
    if len(context.args) < 2:
        await update.message.reply_text(
            "❌ Please provide both wallet address and token mint.\n"
            "Usage: /unwatch <wallet_address> <token_mint>"
        )
        return

    wallet_address = context.args[0]
    token_mint = context.args[1]

    user_id = str(update.effective_user.id)

    try:
        # Remove from watchlist
        removed = await watcher.remove_watched_wallet(
            wallet=wallet_address,
            token_mint=token_mint,
        )

        if removed:
            await update.message.reply_text(
                f"✅ Wallet removed from watchlist!\n\n"
                f"👛 Wallet: {wallet_address[:8]}...{wallet_address[-6:]}\n"
                f"🪙 Token: {token_mint[:8]}...{token_mint[-6:]}"
            )

            logger.info(
                "unwatch_command_success",
                user_id=user_id,
                wallet=wallet_address,
                token=token_mint,
            )
        else:
            await update.message.reply_text(
                "❌ Wallet not found in your watchlist.\n\n"
                "Use /watchlist to see your monitored wallets."
            )

    except Exception as e:
        logger.error(
            "unwatch_command_error",
            user_id=user_id,
            wallet=wallet_address,
            error=str(e),
        )
        await update.message.reply_text(
            f"❌ Error removing wallet from watchlist: {str(e)}"
        )


async def handle_watchlist_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    watcher: WalletWatcher,
) -> None:
    """
    Handle /watchlist command to show user's watched wallets.

    Args:
        update: Telegram update
        context: Telegram context
        watcher: WalletWatcher instance
    """
    if not update.message:
        return

    user_id = str(update.effective_user.id)

    try:
        # Get user's watched wallets
        watched_wallets = await watcher.get_watched_wallets(user_id=user_id)

        if not watched_wallets:
            await update.message.reply_text(
                "📋 Your watchlist is empty.\n\n"
                "Use /watch <wallet> <token> to add wallets to monitor."
            )
            return

        # Build message
        message_lines = ["📋 Your Watchlist\n"]

        for i, wallet_obj in enumerate(watched_wallets, 1):
            message_lines.append(
                f"{i}. 👛 {wallet_obj.wallet[:8]}...{wallet_obj.wallet[-6:]}\n"
                f"   🪙 Token: {wallet_obj.token_mint[:8]}...{wallet_obj.token_mint[-6:]}\n"
                f"   📊 Threshold: {wallet_obj.alert_threshold * 100:.1f}%\n"
                f"   💰 Balance: {wallet_obj.last_balance:,.0f}\n"
                f"   🕐 Added: {wallet_obj.added_at.strftime('%Y-%m-%d %H:%M')}\n"
            )

        message = "\n".join(message_lines)

        # Add footer
        message += (
            f"\n\nTotal: {len(watched_wallets)} wallet(s)\n"
            f"Use /unwatch <wallet> <token> to remove"
        )

        await update.message.reply_text(message)

        logger.info(
            "watchlist_command_success",
            user_id=user_id,
            wallet_count=len(watched_wallets),
        )

    except Exception as e:
        logger.error(
            "watchlist_command_error",
            user_id=user_id,
            error=str(e),
        )
        await update.message.reply_text(
            f"❌ Error retrieving watchlist: {str(e)}"
        )
