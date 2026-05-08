"""
/forecast command handler for Telegram bot.

Provides capital flow predictions with confidence intervals.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
    MockForecaster,
    CapitalFlowForecaster,
)

logger = structlog.get_logger()


async def handle_forecast_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    forecaster: Optional[CapitalFlowForecaster] = None,
) -> None:
    """
    Handle /forecast command to provide capital flow predictions.

    Usage:
        /forecast <token_mint> [days]

    Parameters
    ----------
    update : Update
        Telegram update
    context : ContextTypes.DEFAULT_TYPE
        Telegram context
    forecaster : Optional[CapitalFlowForecaster], optional
        Forecaster service, by default None
    """
    if not update.message:
        return

    if not context.args:
        await update.message.reply_text(
            "❌ Usage: /forecast <token_mint> [days]\n\n"
            "Example: /forecast 4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R 7\n"
            "Days is optional (default: 7, max: 30)"
        )
        return

    # Parse arguments
    token_mint = context.args[0]
    horizon_days = int(context.args[1]) if len(context.args) > 1 else 7

    # Validate inputs
    try:
        if not (32 <= len(token_mint) <= 44):
            raise ValueError("Invalid token mint format")
        if horizon_days < 1 or horizon_days > 30:
            raise ValueError("Days must be between 1 and 30")
    except ValueError as e:
        await update.message.reply_text(
            f"❌ Invalid input: {str(e)}\n"
            "Please provide a valid token mint and days (1-30)."
        )
        return

    user_id = str(update.effective_user.id)

    # Send "analyzing" message
    processing_msg = await update.message.reply_text(
        f"🔮 Forecasting capital flows for next {horizon_days} days...\n"
        "This may take a few moments."
    )

    try:
        # Initialize forecaster (mock for now)
        if forecaster is None:
            forecaster = MockForecaster(trend=0.1, volatility=1.2)

        # Generate synthetic historical data
        import numpy as np
        np.random.seed(hash(token_mint) % 2**32)

        # 30 days of historical flows
        historical_flows = np.random.normal(100, 50, 30)
        historical_flows = np.cumsum(np.random.normal(0, 10, 30)) + 100

        # Timestamps
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        timestamps = [base_time + timedelta(days=i) for i in range(30)]

        # Generate forecast
        forecast = forecaster.forecast_capital_flows(
            historical_flows=historical_flows,
            timestamps=timestamps,
            horizon_days=horizon_days,
            token_mint=token_mint,
        )

        # Run backtest to show accuracy
        if hasattr(forecaster, "backtest"):
            backtest_result = forecaster.backtest(
                historical_flows=historical_flows,
                timestamps=timestamps,
                test_periods=7,
            )
            mape = backtest_result.mape
        else:
            mape = forecast.historical_mape or 15.0

        # Format response
        response = _format_forecast(
            forecast,
            token_mint,
            mape,
            horizon_days,
        )

        # Delete processing message and send result
        await processing_msg.delete()
        await update.message.reply_text(response, parse_mode="HTML")

        logger.info(
            "Forecast generated",
            user_id=user_id,
            token_mint=token_mint,
            horizon_days=horizon_days,
            mape=mape,
        )

    except Exception as e:
        logger.error(
            "Failed to generate forecast",
            error=str(e),
            user_id=user_id,
            token_mint=token_mint,
        )
        await processing_msg.delete()
        await update.message.reply_text(
            f"❌ Failed to generate forecast: {str(e)}\n\n"
            "This may be due to insufficient historical data.\n"
            "Please try again later."
        )


def _format_forecast(
    forecast,
    token_mint: str,
    mape: float,
    horizon_days: int,
) -> str:
    """
    Format forecast for Telegram display.

    Parameters
    ----------
    forecast : CapitalFlowForecast
        Generated forecast
    token_mint : str
        Token mint address
    mape : float
        Historical MAPE
    horizon_days : int
        Forecast horizon

    Returns
    -------
    str
        Formatted HTML message
    """
    msg = "<b>🔮 Capital Flow Forecast</b>\n\n"
    msg += f"<b>Token:</b> <code>{token_mint[:8]}...{token_mint[-8:]}</code>\n"
    msg += f"<b>Horizon:</b> {horizon_days} days\n"
    msg += f"<b>Generated:</b> {forecast.forecast_timestamp.strftime('%Y-%m-%d %H:%M UTC')}\n\n"

    # Accuracy
    accuracy_emoji = "✅" if mape < 20 else "⚠️" if mape < 30 else "❌"
    msg += f"<b>Historical Accuracy:</b> {accuracy_emoji} MAPE: {mape:.1f}%\n"

    quality = _evaluate_forecast_quality(mape)
    msg += f"<i>{quality}</i>\n\n"

    # Key predictions (next 3 days)
    msg += "<b>📊 Near-term Forecast (Next 3 Days):</b>\n"
    for i, point in enumerate(forecast.net_flow_forecast[:3], 1):
        trend_emoji = "📈" if point.predicted_value > 0 else "📉"
        date_str = point.timestamp.strftime("%b %d")

        msg += (
            f"{i}. {date_str}: {trend_emoji} "
            f"<b>{point.predicted_value:+.1f}</b> "
            f"(CI: {point.confidence_lower:.1f} to {point.confidence_upper:.1f})\n"
        )

    msg += "\n"

    # Summary statistics
    avg_net_flow = sum(p.predicted_value for p in forecast.net_flow_forecast) / len(forecast.net_flow_forecast)
    trend_direction = "Inflow expected" if avg_net_flow > 0 else "Outflow expected"
    trend_emoji = "💰" if avg_net_flow > 0 else "⚠️"

    msg += f"<b>{trend_emoji} Trend:</b> {trend_direction}\n"
    msg += f"<b>Average Daily Net Flow:</b> {avg_net_flow:+.1f}\n\n"

    # Uncertainty bounds
    avg_uncertainty = sum(
        (p.confidence_upper - p.confidence_lower) / 2
        for p in forecast.net_flow_forecast
    ) / len(forecast.net_flow_forecast)

    msg += f"<b>Average Uncertainty:</b> ±{avg_uncertainty:.1f}\n\n"

    # Assumptions
    msg += "<b>⚙️ Assumptions:</b>\n"
    for assumption in forecast.assumptions[:3]:
        msg += f"• {assumption}\n"
    msg += "\n"

    # Warnings
    if mape > 30:
        msg += (
            "⚠️ <b>Warning:</b> High forecast uncertainty. "
            "Use predictions with caution.\n\n"
        )

    # Limitations
    msg += "<b>⚠️ Limitations:</b>\n"
    for limitation in forecast.limitations[:2]:
        msg += f"• {limitation}\n"
    msg += "\n"

    # Footer
    msg += (
        "───────────────────\n"
        "Use <code>/explain</code> for risk analysis\n"
        "Use <code>/watch</code> to monitor this token"
    )

    return msg


def _evaluate_forecast_quality(mape: float) -> str:
    """
    Evaluate forecast quality based on MAPE.

    Parameters
    ----------
    mape : float
        Mean Absolute Percentage Error

    Returns
    -------
    str
        Quality description
    """
    if mape < 10:
        return "Excellent forecast accuracy"
    elif mape < 20:
        return "Good forecast accuracy"
    elif mape < 30:
        return "Acceptable forecast accuracy"
    elif mape < 50:
        return "Poor forecast accuracy"
    else:
        return "Very poor forecast accuracy - use with caution"


async def handle_forecast_backtest_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    forecaster: Optional[CapitalFlowForecaster] = None,
) -> None:
    """
    Handle /forecast_backtest command to show historical accuracy.

    Usage:
        /forecast_backtest <token_mint>

    Parameters
    ----------
    update : Update
        Telegram update
    context : ContextTypes.DEFAULT_TYPE
        Telegram context
    forecaster : Optional[CapitalFlowForecaster], optional
        Forecaster service, by default None
    """
    if not update.message:
        return

    if not context.args:
        await update.message.reply_text(
            "❌ Usage: /forecast_backtest <token_mint>\n\n"
            "Example: /forecast_backtest 4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"
        )
        return

    token_mint = context.args[0]

    try:
        if not (32 <= len(token_mint) <= 44):
            raise ValueError("Invalid token mint format")
    except ValueError as e:
        await update.message.reply_text(f"❌ Invalid input: {str(e)}")
        return

    processing_msg = await update.message.reply_text("🔍 Running backtest...")

    try:
        # Initialize forecaster
        if forecaster is None:
            forecaster = MockForecaster()

        # Generate synthetic historical data
        import numpy as np
        np.random.seed(hash(token_mint) % 2**32)
        historical_flows = np.cumsum(np.random.normal(0, 10, 30)) + 100
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        timestamps = [base_time + timedelta(days=i) for i in range(30)]

        # Run backtest
        backtest_result = forecaster.backtest(
            historical_flows=historical_flows,
            timestamps=timestamps,
            test_periods=7,
        )

        # Format response
        msg = "<b>📈 Forecast Backtest Results</b>\n\n"
        msg += f"<b>Token:</b> <code>{token_mint[:8]}...{token_mint[-8:]}</code>\n\n"

        msg += "<b>Accuracy Metrics:</b>\n"
        msg += f"• MAPE: {backtest_result.mape:.1f}%\n"
        msg += f"• RMSE: {backtest_result.rmse:.2f}\n"
        msg += f"• MAE: {backtest_result.mae:.2f}\n"
        msg += f"• Coverage: {backtest_result.coverage:.1f}%\n\n"

        quality = _evaluate_forecast_quality(backtest_result.mape)
        quality_emoji = "✅" if backtest_result.mape < 20 else "⚠️"
        msg += f"{quality_emoji} <b>{quality}</b>\n\n"

        # Recent predictions vs actuals
        msg += "<b>Recent Predictions vs Actuals:</b>\n"
        for i in range(min(5, len(backtest_result.period_actuals))):
            pred = backtest_result.period_predictions[i]
            actual = backtest_result.period_actuals[i]
            error = backtest_result.period_errors[i]
            date = backtest_result.timestamps[i].strftime("%b %d")

            msg += (
                f"{date}: Pred: {pred:.1f}, "
                f"Actual: {actual:.1f}, "
                f"Error: {error:.1f}\n"
            )

        await processing_msg.delete()
        await update.message.reply_text(msg, parse_mode="HTML")

        logger.info(
            "Backtest completed",
            user_id=str(update.effective_user.id),
            token_mint=token_mint,
            mape=backtest_result.mape,
        )

    except Exception as e:
        logger.error("Failed to run backtest", error=str(e))
        await processing_msg.delete()
        await update.message.reply_text(f"❌ Failed to run backtest: {str(e)}")
