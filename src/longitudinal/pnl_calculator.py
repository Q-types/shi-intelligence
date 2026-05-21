"""
Cost Basis and Realised PnL Calculator (Sprint 8).

Implements cost basis estimation with configurable accounting methods
(FIFO, LIFO, weighted_average) and realised PnL calculation on sell events.

HARD RULES:
1. Do not treat price as ground truth
2. Always propagate price confidence
3. Do not show precise PnL when confidence is low
4. Separate realised and unrealised PnL
5. Accounting method must be explicit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Protocol

import structlog

from .models import AccountingMethod, CostBasisLot, RealisedPnLRecord

logger = structlog.get_logger()


# ============================================================================
# Data Structures
# ============================================================================


@dataclass(frozen=True)
class TradeRecord:
    """Individual trade for cost basis calculation."""

    timestamp: datetime
    tokens: int  # Positive for buy, negative for sell
    price_usd: float
    price_confidence: float
    price_source: str
    event_id: Optional[int] = None
    signature: Optional[str] = None


@dataclass(frozen=True)
class CostBasisEstimate:
    """
    Cost basis estimation result.

    Includes confidence weighting per Sprint 8 hard rules.
    """

    avg_entry_price_usd: float
    total_cost_basis_usd: float
    current_position_tokens: int
    current_position_value_usd: float | None  # Requires current price
    unrealised_pnl_usd: float | None
    unrealised_pnl_pct: float | None
    accounting_method: AccountingMethod
    confidence: float  # 0.0-1.0, weighted from source prices
    confidence_reason: str
    lot_count: int  # Number of cost basis lots
    data_points: int  # Number of trades used


@dataclass(frozen=True)
class RealisedPnLEstimate:
    """
    Realised PnL estimation for a sell event.

    Includes liquidity-adjusted metrics per Sprint 8.
    """

    exit_tokens: int
    exit_price_usd: float
    exit_value_usd: float
    entry_price_usd: float  # From cost basis lots
    cost_basis_usd: float
    realised_pnl_usd: float
    realised_pnl_pct: float
    accounting_method: AccountingMethod
    exit_efficiency: float | None  # 0.0-1.0, how well timed
    peak_price_usd: float | None
    peak_to_exit_drawdown_pct: float | None
    liquidity_at_exit_usd: float | None
    liquidity_adjusted_pnl_usd: float | None
    entry_price_confidence: float
    exit_price_confidence: float
    overall_confidence: float
    confidence_reason: str
    is_partial_exit: bool
    remaining_tokens: int
    lots_consumed: list[dict]  # Details of lots used


# ============================================================================
# Cost Basis Calculator
# ============================================================================


class CostBasisCalculator:
    """
    Calculates cost basis with configurable accounting methods.

    Supports:
    - FIFO (First In, First Out) - Default
    - LIFO (Last In, First Out)
    - Weighted Average

    Maintains lot-level tracking for precise partial exit handling.
    """

    def __init__(
        self,
        default_method: AccountingMethod = AccountingMethod.FIFO,
        min_confidence_for_precise: float = 0.6,
    ):
        self._default_method = default_method
        self._min_confidence_for_precise = min_confidence_for_precise

    def compute_cost_basis(
        self,
        trades: list[TradeRecord],
        current_price_usd: float | None = None,
        current_price_confidence: float = 0.5,
        method: AccountingMethod | None = None,
    ) -> CostBasisEstimate:
        """
        Compute cost basis from trade history.

        Args:
            trades: List of trades (positive tokens = buy, negative = sell)
            current_price_usd: Current price for unrealised PnL (optional)
            current_price_confidence: Confidence in current price
            method: Accounting method (defaults to FIFO)

        Returns:
            CostBasisEstimate with confidence-weighted metrics
        """
        method = method or self._default_method

        if not trades:
            return CostBasisEstimate(
                avg_entry_price_usd=0.0,
                total_cost_basis_usd=0.0,
                current_position_tokens=0,
                current_position_value_usd=None,
                unrealised_pnl_usd=None,
                unrealised_pnl_pct=None,
                accounting_method=method,
                confidence=0.0,
                confidence_reason="No trades provided",
                lot_count=0,
                data_points=0,
            )

        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        # Build lots from buys, consume from sells
        lots = self._build_lots(sorted_trades, method)

        # Compute aggregate metrics
        remaining_lots = [lot for lot in lots if lot["remaining_tokens"] > 0]

        if not remaining_lots:
            return CostBasisEstimate(
                avg_entry_price_usd=0.0,
                total_cost_basis_usd=0.0,
                current_position_tokens=0,
                current_position_value_usd=None,
                unrealised_pnl_usd=None,
                unrealised_pnl_pct=None,
                accounting_method=method,
                confidence=0.0,
                confidence_reason="Position fully exited",
                lot_count=0,
                data_points=len(trades),
            )

        # Compute weighted average entry price and confidence
        total_remaining_tokens = sum(lot["remaining_tokens"] for lot in remaining_lots)
        total_cost_basis = sum(
            lot["remaining_tokens"] * lot["price_usd"] for lot in remaining_lots
        )
        weighted_confidence = sum(
            lot["remaining_tokens"] * lot["confidence"] for lot in remaining_lots
        ) / total_remaining_tokens

        avg_entry_price = total_cost_basis / total_remaining_tokens if total_remaining_tokens > 0 else 0.0

        # Compute unrealised PnL if current price provided
        unrealised_pnl_usd = None
        unrealised_pnl_pct = None
        current_value = None

        if current_price_usd is not None and total_remaining_tokens > 0:
            current_value = total_remaining_tokens * current_price_usd
            unrealised_pnl_usd = current_value - total_cost_basis
            unrealised_pnl_pct = (current_price_usd - avg_entry_price) / avg_entry_price if avg_entry_price > 0 else 0.0

            # Adjust confidence based on current price confidence
            weighted_confidence = min(weighted_confidence, current_price_confidence)

        # Build confidence reason
        confidence_reasons = []
        if weighted_confidence >= 0.8:
            confidence_reasons.append("high price confidence")
        elif weighted_confidence >= 0.5:
            confidence_reasons.append("medium price confidence")
        else:
            confidence_reasons.append("low price confidence")

        confidence_reasons.append(f"{len(remaining_lots)} lots")
        confidence_reasons.append(f"{method.value} accounting")

        return CostBasisEstimate(
            avg_entry_price_usd=avg_entry_price,
            total_cost_basis_usd=total_cost_basis,
            current_position_tokens=total_remaining_tokens,
            current_position_value_usd=current_value,
            unrealised_pnl_usd=unrealised_pnl_usd,
            unrealised_pnl_pct=unrealised_pnl_pct,
            accounting_method=method,
            confidence=weighted_confidence,
            confidence_reason="; ".join(confidence_reasons),
            lot_count=len(remaining_lots),
            data_points=len(trades),
        )

    def _build_lots(
        self,
        trades: list[TradeRecord],
        method: AccountingMethod,
    ) -> list[dict]:
        """Build and process cost basis lots from trades."""
        lots: list[dict] = []

        for trade in trades:
            if trade.tokens > 0:
                # Buy: create new lot
                lots.append({
                    "timestamp": trade.timestamp,
                    "original_tokens": trade.tokens,
                    "remaining_tokens": trade.tokens,
                    "price_usd": trade.price_usd,
                    "confidence": trade.price_confidence,
                    "source": trade.price_source,
                    "event_id": trade.event_id,
                })
            elif trade.tokens < 0:
                # Sell: consume from lots based on method
                tokens_to_consume = abs(trade.tokens)
                lots = self._consume_lots(lots, tokens_to_consume, method)

        return lots

    def _consume_lots(
        self,
        lots: list[dict],
        tokens_to_consume: int,
        method: AccountingMethod,
    ) -> list[dict]:
        """Consume tokens from lots based on accounting method."""
        if method == AccountingMethod.FIFO:
            # Consume oldest lots first
            sorted_lots = sorted(lots, key=lambda l: l["timestamp"])
        elif method == AccountingMethod.LIFO:
            # Consume newest lots first
            sorted_lots = sorted(lots, key=lambda l: l["timestamp"], reverse=True)
        else:
            # Weighted average: consume proportionally from all lots
            return self._consume_weighted_average(lots, tokens_to_consume)

        remaining = tokens_to_consume
        for lot in sorted_lots:
            if remaining <= 0:
                break
            if lot["remaining_tokens"] <= 0:
                continue

            consumed = min(lot["remaining_tokens"], remaining)
            lot["remaining_tokens"] -= consumed
            remaining -= consumed

        return lots

    def _consume_weighted_average(
        self,
        lots: list[dict],
        tokens_to_consume: int,
    ) -> list[dict]:
        """Consume tokens proportionally from all lots (weighted average)."""
        total_remaining = sum(lot["remaining_tokens"] for lot in lots)
        if total_remaining <= 0:
            return lots

        # Consume proportionally
        for lot in lots:
            if lot["remaining_tokens"] > 0:
                proportion = lot["remaining_tokens"] / total_remaining
                lot_consume = int(tokens_to_consume * proportion)
                lot["remaining_tokens"] = max(0, lot["remaining_tokens"] - lot_consume)

        return lots


# ============================================================================
# Realised PnL Calculator
# ============================================================================


class RealisedPnLCalculator:
    """
    Calculates realised PnL on sell events.

    Uses cost basis lots to determine entry price for each exit.
    Computes exit efficiency and liquidity-adjusted metrics.
    """

    def __init__(
        self,
        default_method: AccountingMethod = AccountingMethod.FIFO,
        min_confidence_for_precise: float = 0.6,
    ):
        self._default_method = default_method
        self._min_confidence_for_precise = min_confidence_for_precise

    def compute_realised_pnl(
        self,
        exit_tokens: int,
        exit_price_usd: float,
        exit_price_confidence: float,
        cost_basis_lots: list[dict],
        method: AccountingMethod | None = None,
        peak_price_usd: float | None = None,
        liquidity_at_exit_usd: float | None = None,
    ) -> RealisedPnLEstimate:
        """
        Compute realised PnL for a sell event.

        Args:
            exit_tokens: Number of tokens being sold
            exit_price_usd: Price at exit
            exit_price_confidence: Confidence in exit price
            cost_basis_lots: Available lots for cost basis matching
            method: Accounting method (defaults to FIFO)
            peak_price_usd: Highest price seen (for exit efficiency)
            liquidity_at_exit_usd: Liquidity at exit time (for slippage estimate)

        Returns:
            RealisedPnLEstimate with confidence-weighted metrics
        """
        method = method or self._default_method

        if not cost_basis_lots or exit_tokens <= 0:
            return RealisedPnLEstimate(
                exit_tokens=exit_tokens,
                exit_price_usd=exit_price_usd,
                exit_value_usd=exit_tokens * exit_price_usd,
                entry_price_usd=0.0,
                cost_basis_usd=0.0,
                realised_pnl_usd=0.0,
                realised_pnl_pct=0.0,
                accounting_method=method,
                exit_efficiency=None,
                peak_price_usd=peak_price_usd,
                peak_to_exit_drawdown_pct=None,
                liquidity_at_exit_usd=liquidity_at_exit_usd,
                liquidity_adjusted_pnl_usd=None,
                entry_price_confidence=0.0,
                exit_price_confidence=exit_price_confidence,
                overall_confidence=0.0,
                confidence_reason="No cost basis lots available",
                is_partial_exit=False,
                remaining_tokens=0,
                lots_consumed=[],
            )

        # Match exit tokens to lots based on method
        lots_consumed, entry_price, entry_confidence, remaining = self._match_lots_to_exit(
            exit_tokens, cost_basis_lots, method
        )

        # Compute PnL
        exit_value = exit_tokens * exit_price_usd
        cost_basis = exit_tokens * entry_price
        realised_pnl_usd = exit_value - cost_basis
        realised_pnl_pct = (exit_price_usd - entry_price) / entry_price if entry_price > 0 else 0.0

        # Compute exit efficiency
        exit_efficiency = None
        peak_to_exit_drawdown = None
        if peak_price_usd is not None and peak_price_usd > 0:
            exit_efficiency = exit_price_usd / peak_price_usd
            peak_to_exit_drawdown = (peak_price_usd - exit_price_usd) / peak_price_usd

        # Compute liquidity-adjusted PnL
        liquidity_adjusted_pnl = None
        position_vs_liquidity = None
        if liquidity_at_exit_usd is not None and liquidity_at_exit_usd > 0:
            position_vs_liquidity = exit_value / liquidity_at_exit_usd
            # Simple slippage estimate: 1% per 10% of liquidity
            estimated_slippage_pct = min(0.5, position_vs_liquidity * 0.1)
            liquidity_adjusted_pnl = realised_pnl_usd * (1 - estimated_slippage_pct)

        # Compute overall confidence
        overall_confidence = min(entry_confidence, exit_price_confidence)

        # Build confidence reason
        confidence_reasons = []
        if overall_confidence >= 0.8:
            confidence_reasons.append("high confidence")
        elif overall_confidence >= 0.5:
            confidence_reasons.append("medium confidence")
        else:
            confidence_reasons.append("LOW CONFIDENCE - treat as estimate")

        confidence_reasons.append(f"{method.value} accounting")
        confidence_reasons.append(f"{len(lots_consumed)} lots matched")

        return RealisedPnLEstimate(
            exit_tokens=exit_tokens,
            exit_price_usd=exit_price_usd,
            exit_value_usd=exit_value,
            entry_price_usd=entry_price,
            cost_basis_usd=cost_basis,
            realised_pnl_usd=realised_pnl_usd,
            realised_pnl_pct=realised_pnl_pct,
            accounting_method=method,
            exit_efficiency=exit_efficiency,
            peak_price_usd=peak_price_usd,
            peak_to_exit_drawdown_pct=peak_to_exit_drawdown,
            liquidity_at_exit_usd=liquidity_at_exit_usd,
            liquidity_adjusted_pnl_usd=liquidity_adjusted_pnl,
            entry_price_confidence=entry_confidence,
            exit_price_confidence=exit_price_confidence,
            overall_confidence=overall_confidence,
            confidence_reason="; ".join(confidence_reasons),
            is_partial_exit=remaining > 0,
            remaining_tokens=remaining,
            lots_consumed=lots_consumed,
        )

    def _match_lots_to_exit(
        self,
        exit_tokens: int,
        lots: list[dict],
        method: AccountingMethod,
    ) -> tuple[list[dict], float, float, int]:
        """
        Match exit tokens to cost basis lots.

        Returns:
            (lots_consumed, weighted_entry_price, weighted_confidence, remaining_position)
        """
        if method == AccountingMethod.FIFO:
            sorted_lots = sorted(lots, key=lambda l: l["timestamp"])
        elif method == AccountingMethod.LIFO:
            sorted_lots = sorted(lots, key=lambda l: l["timestamp"], reverse=True)
        else:
            # Weighted average: use all lots proportionally
            return self._match_weighted_average(exit_tokens, lots)

        lots_consumed = []
        total_cost = 0.0
        total_confidence_weighted = 0.0
        tokens_matched = 0
        remaining_to_match = exit_tokens

        for lot in sorted_lots:
            if remaining_to_match <= 0:
                break
            if lot.get("remaining_tokens", lot.get("original_tokens", 0)) <= 0:
                continue

            available = lot.get("remaining_tokens", lot.get("original_tokens", 0))
            matched = min(available, remaining_to_match)

            lots_consumed.append({
                "lot_timestamp": lot["timestamp"].isoformat() if isinstance(lot["timestamp"], datetime) else lot["timestamp"],
                "tokens_consumed": matched,
                "price_usd": lot["price_usd"],
                "confidence": lot.get("confidence", 0.5),
            })

            total_cost += matched * lot["price_usd"]
            total_confidence_weighted += matched * lot.get("confidence", 0.5)
            tokens_matched += matched
            remaining_to_match -= matched

        # Compute weighted averages
        weighted_entry_price = total_cost / tokens_matched if tokens_matched > 0 else 0.0
        weighted_confidence = total_confidence_weighted / tokens_matched if tokens_matched > 0 else 0.0

        # Compute remaining position
        total_remaining = sum(
            lot.get("remaining_tokens", lot.get("original_tokens", 0))
            for lot in lots
        ) - exit_tokens

        return lots_consumed, weighted_entry_price, weighted_confidence, max(0, total_remaining)

    def _match_weighted_average(
        self,
        exit_tokens: int,
        lots: list[dict],
    ) -> tuple[list[dict], float, float, int]:
        """Match using weighted average (all lots contribute proportionally)."""
        total_tokens = sum(
            lot.get("remaining_tokens", lot.get("original_tokens", 0))
            for lot in lots
        )

        if total_tokens <= 0:
            return [], 0.0, 0.0, 0

        # Compute weighted average price and confidence across all lots
        total_cost = sum(
            lot.get("remaining_tokens", lot.get("original_tokens", 0)) * lot["price_usd"]
            for lot in lots
        )
        total_confidence_weighted = sum(
            lot.get("remaining_tokens", lot.get("original_tokens", 0)) * lot.get("confidence", 0.5)
            for lot in lots
        )

        avg_price = total_cost / total_tokens
        avg_confidence = total_confidence_weighted / total_tokens

        lots_consumed = [{
            "lot_timestamp": "weighted_average",
            "tokens_consumed": exit_tokens,
            "price_usd": avg_price,
            "confidence": avg_confidence,
        }]

        remaining = total_tokens - exit_tokens

        return lots_consumed, avg_price, avg_confidence, max(0, remaining)


# ============================================================================
# Profit Extraction Analyzer (Sprint 8)
# ============================================================================


class ProfitExtractionAnalyzer:
    """
    Analyzes profit extraction behavior patterns across wallets.

    Computes features that answer:
    - Which wallets extract profit early?
    - Which wallets hold through volatility?
    - Which wallets exit before liquidity deteriorates?
    """

    def compute_wallet_metrics(
        self,
        pnl_records: list[RealisedPnLRecord],
        position_history: list[dict],
    ) -> dict:
        """
        Compute profit extraction behavior metrics for a wallet.

        Args:
            pnl_records: All realised PnL records for the wallet
            position_history: Position snapshots for drawdown analysis

        Returns:
            Dict of profit extraction features
        """
        if not pnl_records:
            return self._empty_metrics()

        # Filter to exits with reasonable confidence
        valid_records = [r for r in pnl_records if r.overall_confidence >= 0.3]

        if not valid_records:
            return self._empty_metrics()

        # Realised profit rate: proportion of profitable exits
        profitable_exits = sum(1 for r in valid_records if r.realised_pnl_usd > 0)
        realised_profit_rate = profitable_exits / len(valid_records)

        # Early profit exit rate: proportion of profitable exits within 4h
        # (Would need entry timestamps - using exit_efficiency as proxy)
        early_exits = sum(
            1 for r in valid_records
            if r.realised_pnl_usd > 0 and r.exit_efficiency is not None and r.exit_efficiency < 0.5
        )
        early_profit_exit_rate = early_exits / max(1, profitable_exits)

        # Average exit efficiency
        efficiencies = [r.exit_efficiency for r in valid_records if r.exit_efficiency is not None]
        average_exit_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0.0

        # Hold through drawdown score
        # High if wallet doesn't exit during >20% drawdowns
        drawdowns = [r.peak_to_exit_drawdown_pct for r in valid_records if r.peak_to_exit_drawdown_pct is not None]
        large_drawdown_holds = sum(1 for d in drawdowns if d > 0.2)
        hold_through_drawdown_score = large_drawdown_holds / len(drawdowns) if drawdowns else 0.0

        # Profit taking consistency
        pnl_pcts = [r.realised_pnl_pct for r in valid_records]
        if len(pnl_pcts) > 1:
            import statistics
            profit_taking_consistency = 1.0 / (1.0 + statistics.stdev(pnl_pcts))
        else:
            profit_taking_consistency = 0.0

        # Liquidity sensitive exit score
        # High if wallet exits when position/liquidity ratio is low
        liquidity_ratios = [
            r.position_vs_liquidity_pct for r in valid_records
            if r.position_vs_liquidity_pct is not None and r.position_vs_liquidity_pct > 0
        ]
        if liquidity_ratios:
            avg_ratio = sum(liquidity_ratios) / len(liquidity_ratios)
            liquidity_sensitive_exit_score = 1.0 / (1.0 + avg_ratio * 10)  # Lower ratio = higher score
        else:
            liquidity_sensitive_exit_score = 0.5  # Neutral if no data

        return {
            "realised_profit_rate": realised_profit_rate,
            "early_profit_exit_rate": early_profit_exit_rate,
            "average_exit_efficiency": average_exit_efficiency,
            "hold_through_drawdown_score": hold_through_drawdown_score,
            "profit_taking_consistency": profit_taking_consistency,
            "liquidity_sensitive_exit_score": liquidity_sensitive_exit_score,
            "exit_efficiency": average_exit_efficiency,
            "pnl_data_points": len(valid_records),
        }

    def _empty_metrics(self) -> dict:
        """Return empty metrics when no data available."""
        return {
            "realised_profit_rate": 0.0,
            "early_profit_exit_rate": 0.0,
            "average_exit_efficiency": 0.0,
            "hold_through_drawdown_score": 0.0,
            "profit_taking_consistency": 0.0,
            "liquidity_sensitive_exit_score": 0.0,
            "exit_efficiency": 0.0,
            "pnl_data_points": 0,
        }
