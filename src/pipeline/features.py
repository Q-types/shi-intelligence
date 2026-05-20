"""
Feature Engineering Pipeline.

Computes all wallet-level features from raw data per PDR Section 3.2.
Sprint 7 additions: Price-derived intelligence features.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

import numpy as np
import structlog

from ..core.types import HolderSnapshot
from ..graph import FundingGraph, compute_graph_features
from ..clustering.archetypes import WalletFeatureVector

if TYPE_CHECKING:
    from ..data.price_provider import PriceData
    from ..data.repositories.price_snapshot import PriceHistory, LiquidityHistory
    from ..data.entry_price import WalletPnLData

logger = structlog.get_logger()


@dataclass
class TemporalContext:
    """Temporal context for feature computation."""

    token_launch_time: datetime
    current_time: datetime
    historical_balances: dict[str, list[tuple[datetime, int]]]  # wallet -> [(time, balance)]


@dataclass
class PriceContext:
    """Price context for price-derived feature computation (Sprint 7)."""

    price_data: "PriceData | None" = None
    price_history: "PriceHistory | None" = None
    liquidity_history: "LiquidityHistory | None" = None
    wallet_pnl_data: dict[str, "WalletPnLData"] | None = None  # wallet -> PnL data
    holder_count_7d_ago: int | None = None  # For holder growth calculation


class FeatureEngineer:
    """
    Computes all wallet-level features per PDR Section 3.2.

    Features:
    - Distribution: balance, share, rank
    - Temporal: entry time, holding duration, burstiness
    - Flow: delta balance (7d, 30d)
    - Graph: in/out degree, centrality, shared funders
    """

    def __init__(self):
        self._version = "1.0.0"

    def compute_features(
        self,
        snapshot: HolderSnapshot,
        funding_graph: FundingGraph,
        temporal_ctx: TemporalContext | None = None,
        trade_history: dict[str, list[datetime]] | None = None,
        price_data: "PriceData | None" = None,
        price_ctx: PriceContext | None = None,
    ) -> list[WalletFeatureVector]:
        """
        Compute features for all wallets in snapshot.

        Args:
            snapshot: Current holder snapshot
            funding_graph: Funding relationship graph
            temporal_ctx: Optional temporal context for time-based features
            trade_history: Optional trade timestamps per wallet
            price_data: Optional current price data for price-based features
            price_ctx: Optional price context for price-derived intelligence (Sprint 7)

        Returns:
            List of WalletFeatureVector for each holder
        """
        logger.info(
            "computing_features",
            holder_count=snapshot.holder_count,
            version=self._version,
            has_price_ctx=price_ctx is not None,
        )

        # Sort balances by amount (descending) for ranking
        sorted_balances = sorted(
            snapshot.balances,
            key=lambda b: b.balance,
            reverse=True,
        )

        # Compute graph features for all wallets
        wallet_addresses = [b.wallet for b in snapshot.balances]
        graph_features = compute_graph_features(funding_graph, wallet_addresses)

        # Compute aggregate price intelligence features
        price_intel = self._compute_price_intelligence(
            snapshot, sorted_balances, price_ctx
        )

        # Build feature vectors
        features = []
        now = datetime.now(timezone.utc)
        launch_time = temporal_ctx.token_launch_time if temporal_ctx else now - timedelta(days=30)

        for rank, balance in enumerate(sorted_balances, start=1):
            wallet = balance.wallet

            # Distribution features
            share = balance.balance / snapshot.total_supply if snapshot.total_supply > 0 else 0

            # Temporal features
            entry_time_relative = self._compute_entry_time(
                wallet, temporal_ctx, launch_time, now
            )
            holding_duration = self._compute_holding_duration(
                wallet, temporal_ctx, now
            )
            position_volatility = self._compute_position_volatility(
                wallet, temporal_ctx
            )

            # Flow features
            delta_7d, delta_30d = self._compute_delta_balances(
                wallet, balance.balance, temporal_ctx
            )

            # Trade features
            trade_count, burstiness, swap_freq = self._compute_trade_features(
                wallet, trade_history
            )

            # Graph features
            gf = graph_features.get(wallet)
            in_degree = gf.in_degree if gf else 0
            out_degree = gf.out_degree if gf else 0
            eigenvector_centrality = gf.eigenvector_centrality if gf else 0.0
            shared_funder_count = gf.shared_funder_count if gf else 0

            # Compute price features
            entry_price, current_price, pnl_ratio, pnl_usd = self._compute_wallet_price_features(
                wallet, temporal_ctx, price_data, price_ctx
            )

            # Compute liquidity features
            liq_current, liq_1h, liq_24h, liq_confidence = self._compute_liquidity_features(
                price_ctx
            )

            features.append(WalletFeatureVector(
                wallet=wallet,
                balance=float(balance.balance),
                share=share,
                rank=rank,
                entry_time_relative=entry_time_relative,
                holding_duration=holding_duration,
                position_volatility=position_volatility,
                delta_balance_7d=delta_7d,
                delta_balance_30d=delta_30d,
                trade_count=trade_count,
                burstiness=burstiness,
                swap_frequency=swap_freq,
                lp_interaction_ratio=0.0,  # Would need LP data
                in_degree=in_degree,
                out_degree=out_degree,
                eigenvector_centrality=eigenvector_centrality,
                shared_funder_count=shared_funder_count,
                # Price features
                entry_price_usd=entry_price,
                current_price_usd=current_price,
                unrealized_pnl_ratio=pnl_ratio,
                unrealized_pnl_usd=pnl_usd,
                # Price-derived intelligence (Sprint 7)
                price_change_1h_pct=price_intel.get("price_change_1h_pct"),
                price_change_24h_pct=price_intel.get("price_change_24h_pct"),
                price_change_7d_pct=price_intel.get("price_change_7d_pct"),
                holder_growth_vs_price_change=price_intel.get("holder_growth_vs_price_change"),
                whale_accumulation_vs_price_change=price_intel.get("whale_accumulation_vs_price_change"),
                sell_pressure_vs_liquidity=price_intel.get("sell_pressure_vs_liquidity"),
                unrealized_profit_concentration=price_intel.get("unrealized_profit_concentration"),
                # Liquidity smoothing
                liquidity_usd_current=liq_current,
                liquidity_usd_1h_avg=liq_1h,
                liquidity_usd_24h_avg=liq_24h,
                liquidity_depth_confidence=liq_confidence,
            ))

        logger.info("features_computed", count=len(features))
        return features

    def _compute_entry_time(
        self,
        wallet: str,
        ctx: TemporalContext | None,
        launch_time: datetime,
        now: datetime,
    ) -> float:
        """Compute entry time relative to token launch (in days)."""
        if not ctx or wallet not in ctx.historical_balances:
            return 0.0

        history = ctx.historical_balances[wallet]
        if not history:
            return 0.0

        # Find first non-zero balance
        for ts, bal in sorted(history, key=lambda x: x[0]):
            if bal > 0:
                entry = ts
                break
        else:
            return 0.0

        total_days = (now - launch_time).total_seconds() / 86400
        entry_days = (entry - launch_time).total_seconds() / 86400

        if total_days <= 0:
            return 0.0

        return entry_days / total_days  # Normalized to [0, 1]

    def _compute_holding_duration(
        self,
        wallet: str,
        ctx: TemporalContext | None,
        now: datetime,
    ) -> float:
        """Compute how long wallet has held tokens (in days)."""
        if not ctx or wallet not in ctx.historical_balances:
            return 0.0

        history = ctx.historical_balances[wallet]
        if not history:
            return 0.0

        # Find first non-zero balance
        for ts, bal in sorted(history, key=lambda x: x[0]):
            if bal > 0:
                return (now - ts).total_seconds() / 86400

        return 0.0

    def _compute_position_volatility(
        self,
        wallet: str,
        ctx: TemporalContext | None,
    ) -> float:
        """Compute volatility of position (std of balance changes)."""
        if not ctx or wallet not in ctx.historical_balances:
            return 0.0

        history = ctx.historical_balances[wallet]
        if len(history) < 2:
            return 0.0

        balances = [bal for _, bal in sorted(history, key=lambda x: x[0])]
        if max(balances) == 0:
            return 0.0

        # Normalized standard deviation
        normalized = [b / max(balances) for b in balances]
        return float(np.std(normalized))

    def _compute_delta_balances(
        self,
        wallet: str,
        current_balance: int,
        ctx: TemporalContext | None,
    ) -> tuple[float, float]:
        """Compute balance changes over 7d and 30d."""
        if not ctx or wallet not in ctx.historical_balances:
            return 0.0, 0.0

        history = ctx.historical_balances[wallet]
        if not history:
            return 0.0, 0.0

        now = ctx.current_time
        sorted_history = sorted(history, key=lambda x: x[0])

        # Find balance 7 days ago
        target_7d = now - timedelta(days=7)
        balance_7d = self._find_balance_at(sorted_history, target_7d)

        # Find balance 30 days ago
        target_30d = now - timedelta(days=30)
        balance_30d = self._find_balance_at(sorted_history, target_30d)

        # Compute deltas (normalized)
        if balance_7d > 0:
            delta_7d = (current_balance - balance_7d) / balance_7d
        else:
            delta_7d = 1.0 if current_balance > 0 else 0.0

        if balance_30d > 0:
            delta_30d = (current_balance - balance_30d) / balance_30d
        else:
            delta_30d = 1.0 if current_balance > 0 else 0.0

        return delta_7d, delta_30d

    def _find_balance_at(
        self,
        history: list[tuple[datetime, int]],
        target: datetime,
    ) -> int:
        """Find balance at or before target time."""
        result = 0
        for ts, bal in history:
            if ts <= target:
                result = bal
            else:
                break
        return result

    def _compute_trade_features(
        self,
        wallet: str,
        trade_history: dict[str, list[datetime]] | None,
    ) -> tuple[int, float, float]:
        """
        Compute trade-related features.

        Returns:
            (trade_count, burstiness, swap_frequency)
        """
        if not trade_history or wallet not in trade_history:
            return 0, 0.0, 0.0

        trades = sorted(trade_history[wallet])
        trade_count = len(trades)

        if trade_count < 2:
            return trade_count, 0.0, 0.0

        # Compute inter-trade intervals
        intervals = []
        for i in range(1, len(trades)):
            delta = (trades[i] - trades[i - 1]).total_seconds()
            intervals.append(delta)

        intervals_arr = np.array(intervals)
        mu = float(np.mean(intervals_arr))
        sigma = float(np.std(intervals_arr))

        # Burstiness per PDR Section 3.2:
        # B = (sigma - mu) / (sigma + mu)
        if sigma + mu > 0:
            burstiness = (sigma - mu) / (sigma + mu)
        else:
            burstiness = 0.0

        # Swap frequency (trades per day)
        if len(trades) >= 2:
            total_time = (trades[-1] - trades[0]).total_seconds()
            if total_time > 0:
                swap_frequency = trade_count / (total_time / 86400)
            else:
                swap_frequency = 0.0
        else:
            swap_frequency = 0.0

        return trade_count, float(burstiness), float(swap_frequency)

    def _compute_wallet_price_features(
        self,
        wallet: str,
        ctx: TemporalContext | None,
        price_data: "PriceData | None",
        price_ctx: PriceContext | None,
    ) -> tuple[float | None, float | None, float, float | None]:
        """
        Compute price-based features for a wallet.

        Returns:
            (entry_price_usd, current_price_usd, unrealized_pnl_ratio, unrealized_pnl_usd)
        """
        if price_data is None and (price_ctx is None or price_ctx.price_data is None):
            return None, None, 0.0, None

        # Use price from context if available, else from direct param
        actual_price_data = price_ctx.price_data if price_ctx and price_ctx.price_data else price_data
        if actual_price_data is None:
            return None, None, 0.0, None

        current_price = actual_price_data.price_usd

        # Get entry price from wallet PnL data if available
        entry_price: float | None = None
        pnl_usd: float | None = None

        if price_ctx and price_ctx.wallet_pnl_data and wallet in price_ctx.wallet_pnl_data:
            pnl_data = price_ctx.wallet_pnl_data[wallet]
            entry_price = pnl_data.entry_price_usd
            pnl_usd = pnl_data.unrealized_pnl_usd
            pnl_ratio = pnl_data.unrealized_pnl_ratio or 0.0
            return entry_price, current_price, pnl_ratio, pnl_usd

        # Fallback: estimate entry price from historical balance data
        if ctx and wallet in ctx.historical_balances:
            history = ctx.historical_balances[wallet]
            if history:
                # In a full implementation, we'd look up historical prices
                # For now, entry price remains None (unavailable)
                pass

        # Calculate unrealized PnL ratio
        pnl_ratio = 0.0
        if entry_price is not None and entry_price > 0:
            pnl_ratio = (current_price - entry_price) / entry_price

        return entry_price, current_price, pnl_ratio, pnl_usd

    def _compute_price_intelligence(
        self,
        snapshot: HolderSnapshot,
        sorted_balances: list,
        price_ctx: PriceContext | None,
    ) -> dict:
        """
        Compute aggregate price-derived intelligence features (Sprint 7).

        Returns dict with:
        - price_change_1h_pct, price_change_24h_pct, price_change_7d_pct
        - holder_growth_vs_price_change
        - whale_accumulation_vs_price_change
        - sell_pressure_vs_liquidity
        - unrealized_profit_concentration
        """
        result: dict = {}

        if price_ctx is None:
            return result

        # Price change features from history
        if price_ctx.price_history:
            ph = price_ctx.price_history
            result["price_change_1h_pct"] = ph.price_change_1h_pct
            result["price_change_24h_pct"] = ph.price_change_24h_pct
            result["price_change_7d_pct"] = ph.price_change_7d_pct

        # Holder growth vs price change
        # Positive = holder growth outpacing price (bullish accumulation)
        # Negative = price growth outpacing holders (possibly speculative)
        if (
            price_ctx.holder_count_7d_ago is not None
            and price_ctx.price_history
            and price_ctx.price_history.price_change_7d_pct is not None
            and snapshot.holder_count > 0
        ):
            holder_growth_pct = (
                (snapshot.holder_count - price_ctx.holder_count_7d_ago)
                / price_ctx.holder_count_7d_ago * 100
                if price_ctx.holder_count_7d_ago > 0
                else 0
            )
            result["holder_growth_vs_price_change"] = (
                holder_growth_pct - price_ctx.price_history.price_change_7d_pct
            )

        # Whale accumulation vs price change
        # Sum of whale balance changes vs price change
        # Positive = whales buying despite price falling (strong conviction)
        # Negative = whales selling while price rising (distribution)
        if sorted_balances and price_ctx.price_history:
            top_10_balances = sorted_balances[:10]
            whale_delta_sum = 0.0
            for bal in top_10_balances:
                # Get wallet delta from feature (if available)
                # For now, use 0 as placeholder - would need historical data
                pass

            # Placeholder - needs actual whale balance deltas
            result["whale_accumulation_vs_price_change"] = None

        # Sell pressure vs liquidity
        # Higher = more sell pressure relative to available liquidity (risk)
        if price_ctx.liquidity_history and price_ctx.liquidity_history.current:
            liq = price_ctx.liquidity_history.current
            if liq > 0:
                # Compute estimated sell pressure (placeholder - needs hazard model output)
                # For now, use a placeholder calculation
                result["sell_pressure_vs_liquidity"] = None

        # Unrealized profit concentration
        # % of total unrealized profit held by top 10
        if price_ctx.wallet_pnl_data and sorted_balances:
            top_10_profit = 0.0
            total_profit = 0.0

            for i, bal in enumerate(sorted_balances):
                wallet = bal.wallet
                if wallet in price_ctx.wallet_pnl_data:
                    pnl_usd = price_ctx.wallet_pnl_data[wallet].unrealized_pnl_usd
                    if pnl_usd is not None and pnl_usd > 0:
                        total_profit += pnl_usd
                        if i < 10:
                            top_10_profit += pnl_usd

            if total_profit > 0:
                result["unrealized_profit_concentration"] = top_10_profit / total_profit

        return result

    def _compute_liquidity_features(
        self,
        price_ctx: PriceContext | None,
    ) -> tuple[float | None, float | None, float | None, str | None]:
        """
        Compute liquidity smoothing features (Sprint 7).

        Returns:
            (liquidity_usd_current, liquidity_usd_1h_avg, liquidity_usd_24h_avg, confidence)
        """
        if price_ctx is None or price_ctx.liquidity_history is None:
            return None, None, None, None

        lh = price_ctx.liquidity_history
        return lh.current, lh.avg_1h, lh.avg_24h, lh.confidence

    def _compute_price_features(
        self,
        wallet: str,
        ctx: TemporalContext | None,
        price_data: "PriceData | None",
    ) -> tuple[float | None, float | None, float]:
        """
        DEPRECATED: Use _compute_wallet_price_features instead.

        Compute price-based features for a wallet.

        Returns:
            (entry_price_usd, current_price_usd, unrealized_pnl_ratio)
        """
        if price_data is None:
            return None, None, 0.0

        current_price = price_data.price_usd

        # If we have historical balance data, estimate entry price
        # For now, we use current price as we'd need historical price data
        # to calculate actual entry price
        entry_price: float | None = None

        if ctx and wallet in ctx.historical_balances:
            history = ctx.historical_balances[wallet]
            if history:
                # In a full implementation, we'd look up historical prices
                # at the entry time. For now, use current price as baseline.
                entry_price = current_price

        # Calculate unrealized PnL ratio
        pnl_ratio = 0.0
        if entry_price is not None and entry_price > 0:
            pnl_ratio = (current_price - entry_price) / entry_price

        return entry_price, current_price, pnl_ratio
