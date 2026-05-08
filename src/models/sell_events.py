"""
Sell Event Detection.

Per INITIAL_PROMPT:
A sell event is defined as:
- Reduction of >= X% of wallet token balance
- Measured relative to rolling peak balance
- Occurring within time horizon T

Default parameters:
- X = 50%
- T = 7 days
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

from ..core.config import settings

logger = structlog.get_logger()


@dataclass
class SellEvent:
    """Detected sell event for survival analysis."""

    wallet: str
    token_mint: str
    event_time: datetime
    peak_balance: int
    balance_at_event: int
    reduction_pct: float
    is_censored: bool  # True if no sell event observed (still holding)
    observation_end: datetime
    holding_duration_days: float


@dataclass
class BalanceHistory:
    """Historical balance data for a wallet."""

    wallet: str
    token_mint: str
    snapshots: list[tuple[datetime, int]]  # (timestamp, balance)

    @property
    def sorted_snapshots(self) -> list[tuple[datetime, int]]:
        """Get chronologically sorted snapshots."""
        return sorted(self.snapshots, key=lambda x: x[0])


class SellEventDetector:
    """
    Detects sell events from balance history.

    Per INITIAL_PROMPT definition:
    - Sell = reduction >= threshold% of rolling peak
    - Within time horizon T
    """

    def __init__(
        self,
        threshold_pct: float | None = None,
        horizon_days: int | None = None,
    ):
        self.threshold_pct = threshold_pct or settings.sell_event_threshold_pct
        self.horizon_days = horizon_days or settings.sell_event_horizon_days
        self._version = "1.0.0"

    def detect_events(
        self,
        history: BalanceHistory,
        observation_end: datetime | None = None,
    ) -> SellEvent:
        """
        Detect sell event from balance history.

        Args:
            history: Wallet balance history
            observation_end: End of observation period (default: now)

        Returns:
            SellEvent with event or censoring info
        """
        observation_end = observation_end or datetime.now(timezone.utc)
        snapshots = history.sorted_snapshots

        if not snapshots:
            raise ValueError("Empty balance history")

        # Track rolling peak
        rolling_peak = 0
        entry_time = None

        for ts, balance in snapshots:
            # Record first non-zero balance as entry
            if entry_time is None and balance > 0:
                entry_time = ts

            # Update rolling peak
            if balance > rolling_peak:
                rolling_peak = balance

            # Check for sell event
            if rolling_peak > 0:
                reduction = (rolling_peak - balance) / rolling_peak

                if reduction >= self.threshold_pct:
                    # Sell event detected
                    holding_duration = (ts - entry_time).total_seconds() / 86400 if entry_time else 0

                    logger.debug(
                        "sell_event_detected",
                        wallet=history.wallet[:8],
                        reduction_pct=reduction,
                        peak=rolling_peak,
                        balance=balance,
                    )

                    return SellEvent(
                        wallet=history.wallet,
                        token_mint=history.token_mint,
                        event_time=ts,
                        peak_balance=rolling_peak,
                        balance_at_event=balance,
                        reduction_pct=reduction,
                        is_censored=False,
                        observation_end=observation_end,
                        holding_duration_days=holding_duration,
                    )

        # No sell event - censored observation
        last_ts, last_balance = snapshots[-1]
        holding_duration = (last_ts - entry_time).total_seconds() / 86400 if entry_time else 0

        return SellEvent(
            wallet=history.wallet,
            token_mint=history.token_mint,
            event_time=observation_end,  # Censored at observation end
            peak_balance=rolling_peak,
            balance_at_event=last_balance,
            reduction_pct=0.0,
            is_censored=True,
            observation_end=observation_end,
            holding_duration_days=holding_duration,
        )

    def detect_events_batch(
        self,
        histories: list[BalanceHistory],
        observation_end: datetime | None = None,
    ) -> list[SellEvent]:
        """Detect sell events for multiple wallets."""
        events = []
        for history in histories:
            try:
                event = self.detect_events(history, observation_end)
                events.append(event)
            except Exception as e:
                logger.warning(
                    "sell_event_detection_failed",
                    wallet=history.wallet[:8],
                    error=str(e),
                )
        return events

    def prepare_survival_data(
        self,
        events: list[SellEvent],
    ) -> tuple[list[float], list[bool], list[dict]]:
        """
        Prepare data for survival analysis.

        Returns:
            (durations, event_observed, feature_dicts)
        """
        durations = []
        event_observed = []
        features = []

        for event in events:
            durations.append(event.holding_duration_days)
            event_observed.append(not event.is_censored)
            features.append({
                "wallet": event.wallet,
                "peak_balance": event.peak_balance,
                "reduction_pct": event.reduction_pct,
            })

        return durations, event_observed, features


class RollingPeakTracker:
    """
    Tracks rolling peak balances for real-time sell detection.

    Used for incremental updates without reprocessing full history.
    """

    def __init__(self):
        self._peaks: dict[str, tuple[int, datetime]] = {}  # wallet -> (peak, peak_time)
        self._entries: dict[str, datetime] = {}  # wallet -> entry_time

    def update(
        self,
        wallet: str,
        balance: int,
        timestamp: datetime,
    ) -> tuple[int, float]:
        """
        Update peak and return current reduction from peak.

        Returns:
            (current_peak, reduction_pct)
        """
        # Track entry
        if wallet not in self._entries and balance > 0:
            self._entries[wallet] = timestamp

        # Update peak
        current_peak, _ = self._peaks.get(wallet, (0, timestamp))

        if balance > current_peak:
            self._peaks[wallet] = (balance, timestamp)
            current_peak = balance

        # Compute reduction
        if current_peak > 0:
            reduction = (current_peak - balance) / current_peak
        else:
            reduction = 0.0

        return current_peak, reduction

    def check_sell_event(
        self,
        wallet: str,
        balance: int,
        timestamp: datetime,
        threshold: float = 0.5,
    ) -> bool:
        """Check if current balance constitutes a sell event."""
        _, reduction = self.update(wallet, balance, timestamp)
        return reduction >= threshold

    def get_holding_duration(self, wallet: str, current_time: datetime) -> float:
        """Get holding duration in days."""
        entry = self._entries.get(wallet)
        if entry is None:
            return 0.0
        return (current_time - entry).total_seconds() / 86400
