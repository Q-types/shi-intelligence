"""
/sequence command handler for Telegram bot.

Analyzes wallet action sequences and detects behavioral patterns.
"""

from __future__ import annotations

from datetime import datetime, timezone

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


from ...sequence import (
    WalletActionEncoder,
    SequencePatternDetector,
    DumpSignatureDetector,
    WalletActionType,
    ActionSequence,
)

logger = structlog.get_logger()


async def handle_sequence_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Handle /sequence command to analyze wallet action sequences.

    Usage:
        /sequence <wallet_address>

    Args:
        update: Telegram update
        context: Telegram context
    """
    if not update.message or not context.args:
        await update.message.reply_text(
            "/sequence <wallet_address>\n\n"
            "Analyzes wallet's historical action patterns.\n\n"
            "Example: /sequence 7xKXtg2CW87d..."
        )
        return

    wallet_address = context.args[0]

    # Validate address
    try:
        if not (32 <= len(wallet_address) <= 44):
            raise ValueError("Invalid address format")
    except Exception as e:
        await update.message.reply_text(
            f"Invalid wallet address: {str(e)}\n"
            "Please provide a valid Solana address (32-44 characters)."
        )
        return

    user_id = str(update.effective_user.id)

    try:
        # Initialize analyzers
        encoder = WalletActionEncoder()
        pattern_detector = SequencePatternDetector()
        dump_detector = DumpSignatureDetector()

        # Mock some actions for demo (in production, fetch from chain)
        mock_action_strings = _generate_mock_action_strings()

        # Encode sequence
        action_sequence = encoder.encode_sequence(
            wallet=wallet_address,
            actions=mock_action_strings,
        )

        # Detect patterns (requires multiple sequences for motif detection)
        motifs = pattern_detector.find_motifs([action_sequence], top_k=5)

        # Check for dump signatures
        signature_matches = dump_detector.detect(action_sequence)

        # Build response
        response_lines = [
            "Wallet Sequence Analysis",
            f"Wallet: {wallet_address[:8]}...{wallet_address[-6:]}",
            "",
            f"Actions Analyzed: {len(action_sequence)}",
            f"Action Types: {', '.join(action_sequence.action_names)}",
            "",
            "Behavioral Patterns:",
        ]

        if motifs:
            for i, motif in enumerate(motifs[:3], 1):
                response_lines.append(
                    f"  {i}. {motif.pattern_str} "
                    f"(freq: {motif.frequency}, confidence: {motif.confidence:.0%})"
                )
        else:
            response_lines.append("  No significant patterns detected")

        response_lines.extend([
            "",
            "Dump Signatures:",
        ])

        if signature_matches:
            for match in signature_matches[:3]:
                response_lines.append(
                    f"  - {match.signature.signature_type.value}: "
                    f"risk {match.risk_score:.0%}, confidence {match.confidence:.0%}"
                )
        else:
            response_lines.append("  No dump signatures detected")

        response_lines.extend([
            "",
            f"Analysis Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        ])

        await update.message.reply_text("\n".join(response_lines))

        logger.info(
            "sequence_command_completed",
            user_id=user_id,
            wallet=wallet_address[:8],
            action_count=len(action_sequence),
        )

    except Exception as e:
        logger.error(
            "sequence_command_error",
            user_id=user_id,
            wallet=wallet_address[:8],
            error=str(e),
        )
        await update.message.reply_text(
            f"Error analyzing sequence: {str(e)}\n\n"
            "Please try again later."
        )


def _generate_mock_action_strings() -> list[str]:
    """Generate mock action strings for demo purposes.

    In production, these would be fetched from on-chain data.
    Uses WalletActionType values.
    """
    return [
        "funded",
        "swap_buy",
        "idle",
        "swap_buy",
        "idle",
        "swap_sell",
    ]
