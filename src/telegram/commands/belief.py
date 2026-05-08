"""
/belief command handler for Telegram bot.

Shows Bayesian risk beliefs and uncertainty quantification.
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


from ...bayesian import (
    RiskBeliefModel,
    Evidence,
    EvidenceType,
    BetaPrior,
    create_default_priors,
)

logger = structlog.get_logger()


async def handle_belief_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Handle /belief command to show Bayesian risk beliefs.

    Usage:
        /belief <token_mint>

    Args:
        update: Telegram update
        context: Telegram context
    """
    if not update.message or not context.args:
        await update.message.reply_text(
            "/belief <token_mint>\n\n"
            "Shows Bayesian risk beliefs with uncertainty.\n\n"
            "Example: /belief 7xKXtg2CW87d..."
        )
        return

    token_mint = context.args[0]

    # Validate address
    try:
        if not (32 <= len(token_mint) <= 44):
            raise ValueError("Invalid token mint format")
    except Exception as e:
        await update.message.reply_text(
            f"Invalid token mint: {str(e)}\n"
            "Please provide a valid Solana address (32-44 characters)."
        )
        return

    user_id = str(update.effective_user.id)

    try:
        # Initialize risk belief model
        priors = create_default_priors()
        model = RiskBeliefModel(token_mint=token_mint, priors=priors)

        # Get current estimate
        estimate = model.get_risk_estimate()

        # Calculate credible interval
        ci_lower, ci_upper = model.dump_prior.credible_interval(0.95)

        # Build response
        response_lines = [
            f"Bayesian Risk Beliefs",
            f"Token: {token_mint[:8]}...{token_mint[-6:]}",
            "",
            f"Risk Score: {estimate.risk_score:.2f}",
            f"Confidence: {estimate.confidence:.0%}",
            "",
            "Dump Risk Distribution:",
            f"  Mean: {estimate.dump_probability:.2%}",
            f"  95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]",
            f"  Uncertainty: {(ci_upper - ci_lower):.2%}",
            "",
            "Prior Information:",
            f"  Evidence Count: {len(model.evidence_history)}",
            f"  Prior Strength: {priors['dump_risk'].alpha + priors['dump_risk'].beta_:.0f}",
            "",
            "Risk Components:",
            f"  Concentration Risk: {estimate.concentration_risk:.2f}",
            f"  Volatility Risk: {estimate.volatility_risk:.2f}",
            f"  Coordination Risk: {estimate.coordination_risk:.2f}",
            "",
            f"Analysis Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        ]

        await update.message.reply_text("\n".join(response_lines))

        logger.info(
            "belief_command_completed",
            user_id=user_id,
            token=token_mint[:8],
            risk_score=estimate.risk_score,
        )

    except Exception as e:
        logger.error(
            "belief_command_error",
            user_id=user_id,
            token=token_mint[:8],
            error=str(e),
        )
        await update.message.reply_text(
            f"Error getting risk beliefs: {str(e)}\n\n"
            "Please try again later."
        )
