"""
/explain command handler for Telegram bot.

Provides risk score explanations with SHAP breakdowns and actionable insights.
"""

from __future__ import annotations

from typing import Optional
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

from ...explainability import (
    NarrativeGenerator,
    MockSHAPExplainer,
    ExplanationType,
)

logger = structlog.get_logger()


async def handle_explain_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    # In production, would inject actual services
    narrative_generator: Optional[NarrativeGenerator] = None,
) -> None:
    """
    Handle /explain command to provide risk score breakdown.

    Usage:
        /explain <token_mint>
        /explain <token_mint> verbose

    Parameters
    ----------
    update : Update
        Telegram update
    context : ContextTypes.DEFAULT_TYPE
        Telegram context
    narrative_generator : Optional[NarrativeGenerator], optional
        Narrative generator service, by default None
    """
    if not update.message:
        return

    if not context.args:
        await update.message.reply_text(
            "❌ Usage: /explain <token_mint> [verbose]\n\n"
            "Example: /explain 4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R\n"
            "Add 'verbose' for technical details."
        )
        return

    # Parse arguments
    token_mint = context.args[0]
    verbose = len(context.args) > 1 and context.args[1].lower() == "verbose"

    # Validate token mint
    try:
        if not (32 <= len(token_mint) <= 44):
            raise ValueError("Invalid token mint format")
    except Exception as e:
        await update.message.reply_text(
            f"❌ Invalid token mint: {str(e)}\n"
            "Please provide a valid Solana token mint address (32-44 characters)."
        )
        return

    user_id = str(update.effective_user.id)

    # Send "analyzing" message
    processing_msg = await update.message.reply_text(
        "🔍 Analyzing token risk factors...\n"
        "This may take a few moments."
    )

    try:
        # Initialize services (mock for now)
        if narrative_generator is None:
            narrative_generator = NarrativeGenerator(verbose=verbose)

        # Get SHAP explanation (mock)
        # In production, would fetch from risk model
        explainer = MockSHAPExplainer(
            feature_names=[
                "hhi", "gini", "top10_pct", "churn_rate",
                "betweenness_centrality", "anomaly_score"
            ],
            baseline_value=0.5,
        )

        # Generate synthetic features for demo
        import numpy as np
        np.random.seed(hash(token_mint) % 2**32)
        features = np.random.uniform(0.3, 0.8, 6)

        explanation = explainer.explain(
            features,
            explanation_type=ExplanationType.RISK_SCORE,
            uncertainty=True,
        )

        # Generate narrative
        narrative = narrative_generator.generate_risk_narrative(
            explanation,
            token_symbol=f"{token_mint[:6]}...",
        )

        # Format response
        response = _format_risk_explanation(narrative, token_mint, verbose)

        # Delete processing message and send result
        await processing_msg.delete()
        await update.message.reply_text(response, parse_mode="HTML")

        logger.info(
            "Risk explanation generated",
            user_id=user_id,
            token_mint=token_mint,
            risk_level=narrative.risk_level.value,
        )

    except Exception as e:
        logger.error(
            "Failed to generate explanation",
            error=str(e),
            user_id=user_id,
            token_mint=token_mint,
        )
        await processing_msg.delete()
        await update.message.reply_text(
            f"❌ Failed to generate explanation: {str(e)}\n\n"
            "Please try again later or contact support."
        )


def _format_risk_explanation(
    narrative,
    token_mint: str,
    verbose: bool,
) -> str:
    """
    Format risk narrative for Telegram.

    Parameters
    ----------
    narrative : RiskNarrative
        Generated narrative
    token_mint : str
        Token mint address
    verbose : bool
        Whether to include technical details

    Returns
    -------
    str
        Formatted HTML message
    """
    # Risk level emoji
    level_emoji = {
        "very_low": "🟢",
        "low": "🟡",
        "moderate": "🟠",
        "high": "🔴",
        "very_high": "⚠️",
    }

    emoji = level_emoji.get(narrative.risk_level.value, "⚪")

    # Build message
    msg = f"<b>{emoji} Risk Analysis Report</b>\n\n"
    msg += f"<b>Token:</b> <code>{token_mint[:8]}...{token_mint[-8:]}</code>\n\n"

    # Summary
    msg += f"<b>Summary:</b>\n{narrative.summary}\n\n"

    # Confidence
    msg += f"<b>Confidence:</b> {narrative.confidence}\n\n"

    # Key drivers
    if narrative.key_drivers:
        msg += "<b>🔍 Key Risk Drivers:</b>\n"
        for i, driver in enumerate(narrative.key_drivers[:5], 1):
            msg += f"{i}. {driver}\n"
        msg += "\n"

    # Actionable insights
    if narrative.actionable_insights:
        msg += "<b>💡 Actionable Insights:</b>\n"
        for insight in narrative.actionable_insights:
            msg += f"• {insight}\n"
        msg += "\n"

    # Uncertainty note
    if narrative.uncertainty_note:
        msg += f"<b>⚠️ Uncertainty:</b>\n{narrative.uncertainty_note}\n\n"

    # Technical details (verbose mode)
    if verbose and narrative.technical_details:
        msg += "<b>🔧 Technical Details:</b>\n"
        msg += f"<pre>{narrative.technical_details}</pre>\n"

    # Footer
    msg += (
        "───────────────────\n"
        "Use <code>/forecast</code> to see future predictions\n"
        "Use <code>/watch</code> to monitor this token"
    )

    return msg


async def handle_explain_regime_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    narrative_generator: Optional[NarrativeGenerator] = None,
) -> None:
    """
    Handle /explain_regime command for regime transition explanations.

    Usage:
        /explain_regime <token_mint>

    Parameters
    ----------
    update : Update
        Telegram update
    context : ContextTypes.DEFAULT_TYPE
        Telegram context
    narrative_generator : Optional[NarrativeGenerator], optional
        Narrative generator, by default None
    """
    if not update.message:
        return

    if not context.args:
        await update.message.reply_text(
            "❌ Usage: /explain_regime <token_mint>\n\n"
            "Example: /explain_regime 4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"
        )
        return

    token_mint = context.args[0]

    try:
        if not (32 <= len(token_mint) <= 44):
            raise ValueError("Invalid token mint format")
    except Exception as e:
        await update.message.reply_text(f"❌ Invalid token mint: {str(e)}")
        return

    processing_msg = await update.message.reply_text("🔍 Analyzing regime state...")

    try:
        # Initialize generator
        if narrative_generator is None:
            narrative_generator = NarrativeGenerator()

        # Mock regime transition
        from ...temporal.regimes import HolderRegimeType
        regime_narrative = narrative_generator.generate_regime_narrative(
            from_regime=HolderRegimeType.STABLE,
            to_regime=HolderRegimeType.DISTRIBUTION,
            confidence=0.85,
        )

        # Format response
        msg = "<b>📊 Regime Analysis</b>\n\n"
        msg += f"<b>Token:</b> <code>{token_mint[:8]}...{token_mint[-8:]}</code>\n\n"
        msg += f"<b>Transition:</b>\n{regime_narrative.transition_summary}\n\n"
        msg += f"<b>Why:</b>\n{regime_narrative.reason}\n\n"
        msg += f"<b>Confidence:</b> {regime_narrative.confidence}\n\n"

        if regime_narrative.implications:
            msg += "<b>📌 Implications:</b>\n"
            for implication in regime_narrative.implications:
                msg += f"• {implication}\n"

        await processing_msg.delete()
        await update.message.reply_text(msg, parse_mode="HTML")

        logger.info(
            "Regime explanation generated",
            user_id=str(update.effective_user.id),
            token_mint=token_mint,
        )

    except Exception as e:
        logger.error("Failed to generate regime explanation", error=str(e))
        await processing_msg.delete()
        await update.message.reply_text(
            f"❌ Failed to generate explanation: {str(e)}"
        )
