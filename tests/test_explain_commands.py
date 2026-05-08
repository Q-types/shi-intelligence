"""
Unit tests for /explain and /forecast Telegram commands (Sprint 4).

Tests command handlers for explainability features.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta

# Mock telegram before importing commands
with patch.dict("sys.modules", {"telegram": MagicMock(), "telegram.ext": MagicMock()}):
    from src.telegram.commands.explain import (
        handle_explain_command,
        handle_explain_regime_command,
        _format_risk_explanation,
    )
    from src.telegram.commands.forecast import (
        handle_forecast_command,
        handle_forecast_backtest_command,
        _format_forecast,
        _evaluate_forecast_quality,
    )

from src.explainability import (
    NarrativeGenerator,
    MockForecaster,
)


class TestExplainCommand:
    """Test /explain command handler."""

    @pytest.mark.asyncio
    async def test_explain_without_args(self):
        """Test /explain without arguments shows usage."""
        update = MagicMock()
        context = MagicMock()
        update.message = AsyncMock()
        context.args = []

        await handle_explain_command(update, context)

        # Should reply with usage message
        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args[0][0]
        assert "Usage:" in call_args
        assert "/explain" in call_args

    @pytest.mark.asyncio
    async def test_explain_with_invalid_token(self):
        """Test /explain with invalid token mint."""
        update = MagicMock()
        context = MagicMock()
        update.message = AsyncMock()
        context.args = ["invalid"]  # Too short

        await handle_explain_command(update, context)

        # Should reply with error
        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args[0][0]
        assert "Invalid" in call_args or "❌" in call_args

    @pytest.mark.asyncio
    async def test_explain_with_valid_token(self):
        """Test /explain with valid token generates explanation."""
        update = MagicMock()
        context = MagicMock()
        update.message = AsyncMock()
        update.effective_user.id = 12345
        context.args = ["4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"]

        # Mock processing message
        processing_msg = AsyncMock()
        update.message.reply_text.return_value = processing_msg

        # Mock narrative generator
        generator = NarrativeGenerator()

        await handle_explain_command(update, context, narrative_generator=generator)

        # Should delete processing message
        processing_msg.delete.assert_called_once()

        # Should send result (called twice: processing + result)
        assert update.message.reply_text.call_count == 2

        # Check final message contains expected content
        final_call = update.message.reply_text.call_args_list[1]
        final_message = final_call[0][0]
        assert "Risk Analysis Report" in final_message or "Risk" in final_message

    @pytest.mark.asyncio
    async def test_explain_verbose_mode(self):
        """Test /explain with verbose flag."""
        update = MagicMock()
        context = MagicMock()
        update.message = AsyncMock()
        update.effective_user.id = 12345
        context.args = ["4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R", "verbose"]

        processing_msg = AsyncMock()
        update.message.reply_text.return_value = processing_msg

        generator = NarrativeGenerator(verbose=True)

        await handle_explain_command(update, context, narrative_generator=generator)

        # Should complete successfully
        processing_msg.delete.assert_called_once()
        assert update.message.reply_text.call_count == 2

    @pytest.mark.asyncio
    async def test_explain_regime_command(self):
        """Test /explain_regime command."""
        update = MagicMock()
        context = MagicMock()
        update.message = AsyncMock()
        update.effective_user.id = 12345
        context.args = ["4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"]

        processing_msg = AsyncMock()
        update.message.reply_text.return_value = processing_msg

        generator = NarrativeGenerator()

        await handle_explain_regime_command(update, context, narrative_generator=generator)

        # Should complete successfully
        processing_msg.delete.assert_called_once()
        assert update.message.reply_text.call_count == 2

        # Check message contains regime info
        final_message = update.message.reply_text.call_args_list[1][0][0]
        assert "Regime" in final_message or "regime" in final_message


class TestFormatRiskExplanation:
    """Test risk explanation formatting."""

    def test_format_basic_narrative(self):
        """Test formatting basic risk narrative."""
        # Create mock narrative
        narrative = MagicMock()
        narrative.risk_level.value = "moderate"
        narrative.summary = "Token shows moderate risk"
        narrative.confidence = "High confidence"
        narrative.key_drivers = ["Driver 1", "Driver 2"]
        narrative.actionable_insights = ["Insight 1"]
        narrative.uncertainty_note = None
        narrative.technical_details = None

        token_mint = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"

        result = _format_risk_explanation(narrative, token_mint, verbose=False)

        assert isinstance(result, str)
        assert "Risk Analysis Report" in result
        assert token_mint[:8] in result
        assert "moderate" in result.lower() or "🟠" in result
        assert "Driver 1" in result

    def test_format_with_technical_details(self):
        """Test formatting with verbose technical details."""
        narrative = MagicMock()
        narrative.risk_level.value = "high"
        narrative.summary = "High risk detected"
        narrative.confidence = "High confidence"
        narrative.key_drivers = []
        narrative.actionable_insights = []
        narrative.uncertainty_note = None
        narrative.technical_details = "Baseline: 0.5\nPrediction: 0.75"

        result = _format_risk_explanation(narrative, "token123", verbose=True)

        assert "Technical Details" in result
        assert "Baseline" in result


class TestForecastCommand:
    """Test /forecast command handler."""

    @pytest.mark.asyncio
    async def test_forecast_without_args(self):
        """Test /forecast without arguments shows usage."""
        update = MagicMock()
        context = MagicMock()
        update.message = AsyncMock()
        context.args = []

        await handle_forecast_command(update, context)

        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args[0][0]
        assert "Usage:" in call_args
        assert "/forecast" in call_args

    @pytest.mark.asyncio
    async def test_forecast_with_invalid_days(self):
        """Test /forecast with invalid days parameter."""
        update = MagicMock()
        context = MagicMock()
        update.message = AsyncMock()
        context.args = ["4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R", "50"]  # Too many days

        await handle_forecast_command(update, context)

        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args[0][0]
        assert "Invalid" in call_args or "❌" in call_args

    @pytest.mark.asyncio
    async def test_forecast_with_valid_token(self):
        """Test /forecast generates forecast successfully."""
        update = MagicMock()
        context = MagicMock()
        update.message = AsyncMock()
        update.effective_user.id = 12345
        context.args = ["4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R", "7"]

        processing_msg = AsyncMock()
        update.message.reply_text.return_value = processing_msg

        forecaster = MockForecaster()

        await handle_forecast_command(update, context, forecaster=forecaster)

        # Should delete processing message
        processing_msg.delete.assert_called_once()

        # Should send result
        assert update.message.reply_text.call_count == 2

        final_message = update.message.reply_text.call_args_list[1][0][0]
        assert "Capital Flow Forecast" in final_message or "Forecast" in final_message

    @pytest.mark.asyncio
    async def test_forecast_default_horizon(self):
        """Test /forecast uses default 7-day horizon."""
        update = MagicMock()
        context = MagicMock()
        update.message = AsyncMock()
        update.effective_user.id = 12345
        context.args = ["4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"]  # No days specified

        processing_msg = AsyncMock()
        update.message.reply_text.return_value = processing_msg

        forecaster = MockForecaster()

        await handle_forecast_command(update, context, forecaster=forecaster)

        # Should complete successfully with default 7 days
        processing_msg.delete.assert_called_once()
        assert update.message.reply_text.call_count == 2

    @pytest.mark.asyncio
    async def test_forecast_backtest_command(self):
        """Test /forecast_backtest command."""
        update = MagicMock()
        context = MagicMock()
        update.message = AsyncMock()
        update.effective_user.id = 12345
        context.args = ["4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"]

        processing_msg = AsyncMock()
        update.message.reply_text.return_value = processing_msg

        forecaster = MockForecaster()

        await handle_forecast_backtest_command(update, context, forecaster=forecaster)

        # Should complete successfully
        processing_msg.delete.assert_called_once()
        assert update.message.reply_text.call_count == 2

        final_message = update.message.reply_text.call_args_list[1][0][0]
        assert "Backtest" in final_message or "MAPE" in final_message


class TestFormatForecast:
    """Test forecast formatting."""

    def test_format_basic_forecast(self):
        """Test formatting basic forecast."""
        # Create mock forecast
        from src.explainability.forecasting import CapitalFlowForecast, ForecastPoint

        forecast = CapitalFlowForecast(
            token_mint="test_token",
            forecast_timestamp=datetime.now(timezone.utc),
            horizon_days=7,
            net_flow_forecast=[
                ForecastPoint(
                    timestamp=datetime.now(timezone.utc) + timedelta(days=i),
                    predicted_value=100.0 + i * 10,
                    confidence_lower=80.0 + i * 10,
                    confidence_upper=120.0 + i * 10,
                    prediction_std=10.0,
                )
                for i in range(7)
            ],
            inflow_forecast=[],
            outflow_forecast=[],
        )

        result = _format_forecast(forecast, "token123", mape=15.0, horizon_days=7)

        assert isinstance(result, str)
        assert "Capital Flow Forecast" in result
        assert "token123" in result
        assert "MAPE: 15.0%" in result
        assert "Next 3 Days" in result

    def test_format_forecast_with_high_mape(self):
        """Test formatting includes warning for high MAPE."""
        from src.explainability.forecasting import CapitalFlowForecast, ForecastPoint

        forecast = CapitalFlowForecast(
            token_mint="test_token",
            forecast_timestamp=datetime.now(timezone.utc),
            horizon_days=7,
            net_flow_forecast=[
                ForecastPoint(
                    timestamp=datetime.now(timezone.utc) + timedelta(days=1),
                    predicted_value=100.0,
                    confidence_lower=80.0,
                    confidence_upper=120.0,
                    prediction_std=10.0,
                )
            ],
            inflow_forecast=[],
            outflow_forecast=[],
        )

        result = _format_forecast(forecast, "token123", mape=35.0, horizon_days=7)

        assert "Warning" in result or "⚠️" in result
        assert "uncertainty" in result.lower() or "caution" in result.lower()


class TestForecastQualityEvaluation:
    """Test forecast quality evaluation."""

    def test_evaluate_excellent_forecast(self):
        """Test excellent forecast quality."""
        result = _evaluate_forecast_quality(8.0)
        assert "Excellent" in result

    def test_evaluate_good_forecast(self):
        """Test good forecast quality."""
        result = _evaluate_forecast_quality(15.0)
        assert "Good" in result

    def test_evaluate_acceptable_forecast(self):
        """Test acceptable forecast quality."""
        result = _evaluate_forecast_quality(25.0)
        assert "Acceptable" in result

    def test_evaluate_poor_forecast(self):
        """Test poor forecast quality."""
        result = _evaluate_forecast_quality(40.0)
        assert "Poor" in result

    def test_evaluate_very_poor_forecast(self):
        """Test very poor forecast quality."""
        result = _evaluate_forecast_quality(60.0)
        assert "Very poor" in result or "caution" in result


class TestCommandIntegration:
    """Integration tests for commands."""

    @pytest.mark.asyncio
    async def test_explain_and_forecast_workflow(self):
        """Test typical user workflow: explain then forecast."""
        token_mint = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"

        # 1. User requests explanation
        update1 = MagicMock()
        context1 = MagicMock()
        update1.message = AsyncMock()
        update1.effective_user.id = 12345
        context1.args = [token_mint]

        processing_msg1 = AsyncMock()
        update1.message.reply_text.return_value = processing_msg1

        generator = NarrativeGenerator()
        await handle_explain_command(update1, context1, narrative_generator=generator)

        # Should complete successfully
        assert processing_msg1.delete.call_count == 1

        # 2. User requests forecast
        update2 = MagicMock()
        context2 = MagicMock()
        update2.message = AsyncMock()
        update2.effective_user.id = 12345
        context2.args = [token_mint, "7"]

        processing_msg2 = AsyncMock()
        update2.message.reply_text.return_value = processing_msg2

        forecaster = MockForecaster()
        await handle_forecast_command(update2, context2, forecaster=forecaster)

        # Should complete successfully
        assert processing_msg2.delete.call_count == 1

    @pytest.mark.asyncio
    async def test_error_handling_in_explain(self):
        """Test error handling when explanation fails."""
        update = MagicMock()
        context = MagicMock()
        update.message = AsyncMock()
        update.effective_user.id = 12345
        context.args = ["4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"]

        processing_msg = AsyncMock()
        update.message.reply_text.return_value = processing_msg

        # Mock generator that raises error
        bad_generator = MagicMock()
        bad_generator.generate_risk_narrative.side_effect = Exception("Test error")

        await handle_explain_command(update, context, narrative_generator=bad_generator)

        # Should delete processing message
        processing_msg.delete.assert_called_once()

        # Should send error message
        assert update.message.reply_text.call_count == 2
        error_message = update.message.reply_text.call_args_list[1][0][0]
        assert "Failed" in error_message or "❌" in error_message
