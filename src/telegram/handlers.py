"""
Telegram Command Handlers.

Implements all bot commands per PRD Section 7.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

from .bot import SHIBot
from .formatters import (
    format_risk_report,
    format_holder_summary,
    format_quick_summary,
    format_error,
)
from ..core.config import settings
from ..validation import validate_token_mint
from ..pipeline.orchestrator import AnalysisOrchestrator
from ..data.client import SolanaDataClient

logger = structlog.get_logger()

# Global orchestrator instance
_orchestrator: AnalysisOrchestrator | None = None


def get_orchestrator() -> AnalysisOrchestrator:
    """Get or create the analysis orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AnalysisOrchestrator(
            data_client=SolanaDataClient(),
        )
    return _orchestrator

# Timeout for analysis (30 seconds per SLA)
ANALYSIS_TIMEOUT = settings.sla_timeout_seconds


def register_handlers(app: Application, bot: SHIBot) -> None:
    """Register all command handlers."""

    async def rate_limit_check(update: Update) -> bool:
        """Check rate limit and send message if limited."""
        user_id = update.effective_user.id
        allowed, retry_after = bot.check_rate_limit(user_id)
        if not allowed:
            await update.message.reply_text(
                f"Rate limit exceeded. Please try again in {retry_after} seconds."
            )
            return False
        return True

    async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        await update.message.reply_text(
            "Welcome to Solana Holder Intelligence (SHI)\n\n"
            "I analyze token holder distribution and provide probabilistic risk intelligence.\n\n"
            "Commands:\n"
            "/analyze <mint> - Full analysis\n"
            "/summary <mint> - Quick overview\n"
            "/top_holders <mint> - Holder breakdown\n"
            "/risk <mint> - Risk scores\n"
            "/help - More info\n\n"
            "Note: All outputs are probabilistic. No trading signals provided."
        )

    async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        await update.message.reply_text(
            "Solana Holder Intelligence\n\n"
            "Commands:\n"
            "/analyze <mint> - Full token analysis including distribution, "
            "archetypes, risk scores, and sell pressure\n"
            "/summary <mint> - Quick 3-line overview\n"
            "/top_holders <mint> - Top 10 holder breakdown\n"
            "/risk <mint> - Stability and risk scores only\n"
            "/graph <mint> - Funding graph visualization (link)\n"
            "/history <mint> - Historical comparison\n\n"
            "Metrics:\n"
            "- HHI (concentration)\n"
            "- Gini (inequality)\n"
            "- Whale Dominance\n"
            "- Coordination Score\n"
            "- Sell Pressure Index\n\n"
            "Disclaimer: All outputs are observational and probabilistic. "
            "No causal inference is implied. Not trading advice."
        )

    async def analyze_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /analyze <mint> command."""
        if not await rate_limit_check(update):
            return

        # Parse mint address
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "Usage: /analyze <token_mint_address>\n"
                "Example: /analyze EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            )
            return

        mint = context.args[0]

        # Validate mint
        validation = validate_token_mint(mint)
        if not validation.is_valid:
            await update.message.reply_text(
                f"Invalid token mint address: {validation.errors[0].message}"
            )
            return

        # Check if token is allowed
        if not bot.is_token_allowed(mint):
            await update.message.reply_text("This token is not available for analysis.")
            return

        # Send "analyzing" message
        status_msg = await update.message.reply_text(
            f"Analyzing token {mint[:8]}...{mint[-4:]}\n"
            "This may take up to 30 seconds."
        )

        try:
            # Run analysis with timeout
            report = await asyncio.wait_for(
                run_full_analysis(mint),
                timeout=ANALYSIS_TIMEOUT,
            )

            # Format and send report
            formatted = format_risk_report(report)

            # Delete status message
            await status_msg.delete()

            # Send report with inline buttons for more details
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("Top Holders", callback_data=f"holders:{mint}"),
                    InlineKeyboardButton("Graph", callback_data=f"graph:{mint}"),
                ],
                [
                    InlineKeyboardButton("History", callback_data=f"history:{mint}"),
                ],
            ])

            await update.message.reply_text(
                formatted,
                reply_markup=keyboard,
                parse_mode="Markdown",
            )

            logger.info(
                "analysis_completed",
                mint=mint,
                user_id=update.effective_user.id,
            )

        except asyncio.TimeoutError:
            await status_msg.edit_text(
                f"Analysis timed out after {ANALYSIS_TIMEOUT}s.\n"
                "This token may have too many holders. Try /summary for a quick overview."
            )
            logger.warning("analysis_timeout", mint=mint)

        except Exception as e:
            await status_msg.edit_text(format_error(str(e)))
            logger.error("analysis_error", mint=mint, error=str(e))

    async def summary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /summary <mint> command."""
        if not await rate_limit_check(update):
            return

        if not context.args:
            await update.message.reply_text("Usage: /summary <token_mint_address>")
            return

        mint = context.args[0]

        # Validate
        validation = validate_token_mint(mint)
        if not validation.is_valid:
            await update.message.reply_text(f"Invalid mint: {validation.errors[0].message}")
            return

        try:
            summary = await asyncio.wait_for(
                run_quick_summary(mint),
                timeout=10,  # Quick summary should be fast
            )
            await update.message.reply_text(
                format_quick_summary(summary),
                parse_mode="Markdown",
            )

        except asyncio.TimeoutError:
            await update.message.reply_text("Summary timed out. Please try again.")
        except Exception as e:
            await update.message.reply_text(format_error(str(e)))

    async def top_holders_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /top_holders <mint> command."""
        if not await rate_limit_check(update):
            return

        if not context.args:
            await update.message.reply_text("Usage: /top_holders <token_mint_address>")
            return

        mint = context.args[0]

        validation = validate_token_mint(mint)
        if not validation.is_valid:
            await update.message.reply_text(f"Invalid mint: {validation.errors[0].message}")
            return

        try:
            holders = await asyncio.wait_for(
                run_holder_analysis(mint),
                timeout=15,
            )
            await update.message.reply_text(
                format_holder_summary(holders),
                parse_mode="Markdown",
            )

        except asyncio.TimeoutError:
            await update.message.reply_text("Request timed out.")
        except Exception as e:
            await update.message.reply_text(format_error(str(e)))

    async def risk_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /risk <mint> command."""
        if not await rate_limit_check(update):
            return

        if not context.args:
            await update.message.reply_text("Usage: /risk <token_mint_address>")
            return

        mint = context.args[0]

        validation = validate_token_mint(mint)
        if not validation.is_valid:
            await update.message.reply_text(f"Invalid mint: {validation.errors[0].message}")
            return

        try:
            risk = await asyncio.wait_for(
                run_risk_analysis(mint),
                timeout=20,
            )

            msg = (
                f"Risk Analysis: `{mint[:8]}...`\n\n"
                f"Stability Score: {risk['stability']:.0f}/100\n"
                f"Sell Pressure: {risk['sell_pressure']:.2f}\n"
                f"Sybil Probability: {risk['sybil_prob']:.1%}\n\n"
                f"_Computed at {risk['timestamp']}_"
            )

            await update.message.reply_text(msg, parse_mode="Markdown")

        except asyncio.TimeoutError:
            await update.message.reply_text("Request timed out.")
        except Exception as e:
            await update.message.reply_text(format_error(str(e)))

    async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline button callbacks."""
        query = update.callback_query
        await query.answer()

        data = query.data
        action, mint = data.split(":", 1)

        if action == "holders":
            holders = await run_holder_analysis(mint)
            await query.message.reply_text(
                format_holder_summary(holders),
                parse_mode="Markdown",
            )
        elif action == "graph":
            await query.message.reply_text(
                f"Funding graph visualization for `{mint[:8]}...` coming soon.\n"
                "This feature is under development.",
                parse_mode="Markdown",
            )
        elif action == "history":
            await query.message.reply_text(
                f"Historical comparison for `{mint[:8]}...` coming soon.\n"
                "This feature is under development.",
                parse_mode="Markdown",
            )

    # Register all handlers
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("analyze", analyze_handler))
    app.add_handler(CommandHandler("summary", summary_handler))
    app.add_handler(CommandHandler("top_holders", top_holders_handler))
    app.add_handler(CommandHandler("risk", risk_handler))
    app.add_handler(CallbackQueryHandler(callback_handler))

    logger.info("telegram_handlers_registered")


# Analysis functions using the orchestrator
async def run_full_analysis(mint: str) -> dict:
    """Run full token analysis using orchestrator."""
    orchestrator = get_orchestrator()
    result = await orchestrator.analyze(mint)

    if result.is_partial or result.metrics is None:
        # Return minimal data for partial results
        return {
            "mint": mint,
            "holder_count": result.holder_count,
            "metrics": {"hhi": 0, "gini": 0, "wdr": 0, "churn": 0},
            "stability_score": 0,
            "sell_pressure": 0,
            "sybil_prob": 0,
            "archetypes": {},
            "timestamp": result.computed_at.isoformat(),
            "warnings": result.warnings,
        }

    return {
        "mint": mint,
        "holder_count": result.holder_count,
        "metrics": {
            "hhi": result.metrics.hhi.value,
            "gini": result.metrics.gini_coefficient.value,
            "wdr": result.metrics.whale_dominance_ratio.value,
            "churn": result.metrics.churn_rate.value if result.metrics.churn_rate else 0,
        },
        "stability_score": result.risk_report.stability_score if result.risk_report else 0,
        "sell_pressure": result.risk_report.sell_pressure_index if result.risk_report else 0,
        "sybil_prob": result.risk_report.sybil_probability if result.risk_report else 0,
        "archetypes": result.archetypes,
        "timestamp": result.computed_at.isoformat(),
        "warnings": result.warnings,
    }


async def run_quick_summary(mint: str) -> dict:
    """Run quick summary analysis."""
    orchestrator = get_orchestrator()

    try:
        result = await orchestrator.analyze(mint, timeout=10)

        stability = result.risk_report.stability_score if result.risk_report else 50

        if stability >= 70:
            risk_level = "Low"
        elif stability >= 40:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "mint": mint,
            "holder_count": result.holder_count,
            "stability_score": int(stability),
            "risk_level": risk_level,
        }
    except Exception:
        return {
            "mint": mint,
            "holder_count": 0,
            "stability_score": 50,
            "risk_level": "Unknown",
        }


async def run_holder_analysis(mint: str) -> dict:
    """Analyze top holders."""
    orchestrator = get_orchestrator()
    result = await orchestrator.analyze(mint, timeout=15)

    top_holders = []
    if result.metrics:
        # Get archetype assignments for top holders
        # This is a simplified version
        for i, (archetype, proportion) in enumerate(
            sorted(result.archetypes.items(), key=lambda x: x[1], reverse=True)[:5]
        ):
            top_holders.append({
                "rank": i + 1,
                "share": proportion,
                "archetype": archetype,
            })

    return {
        "mint": mint,
        "top_holders": top_holders or [
            {"rank": 1, "share": 0.0, "archetype": "unknown"}
        ],
    }


async def run_risk_analysis(mint: str) -> dict:
    """Run risk-only analysis."""
    orchestrator = get_orchestrator()
    result = await orchestrator.analyze(mint, timeout=20)

    if result.risk_report:
        return {
            "stability": result.risk_report.stability_score,
            "sell_pressure": result.risk_report.sell_pressure_index,
            "sybil_prob": result.risk_report.sybil_probability,
            "timestamp": result.computed_at.strftime("%Y-%m-%d %H:%M UTC"),
        }

    return {
        "stability": 50.0,
        "sell_pressure": 0.0,
        "sybil_prob": 0.0,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
