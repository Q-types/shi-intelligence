"""
Feature Engineering Pipeline.

Computes all wallet-level features from raw data per PDR Section 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np
import structlog

from ..core.types import HolderSnapshot
from ..graph import FundingGraph, compute_graph_features
from ..clustering.archetypes import WalletFeatureVector

logger = structlog.get_logger()


@dataclass
class TemporalContext:
    """Temporal context for feature computation."""

    token_launch_time: datetime
    current_time: datetime
    historical_balances: dict[str, list[tuple[datetime, int]]]  # wallet -> [(time, balance)]


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
    ) -> list[WalletFeatureVector]:
        """
        Compute features for all wallets in snapshot.

        Args:
            snapshot: Current holder snapshot
            funding_graph: Funding relationship graph
            temporal_ctx: Optional temporal context for time-based features
            trade_history: Optional trade timestamps per wallet

        Returns:
            List of WalletFeatureVector for each holder
        """
        logger.info(
            "computing_features",
            holder_count=snapshot.holder_count,
            version=self._version,
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
