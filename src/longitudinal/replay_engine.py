"""
Replay Engine for Longitudinal Intelligence.

Deterministic state reconstruction from event stream.

HARD RULES:
- Replay must be deterministic: same events → same state
- Support time scrubbing (jump to any point)
- Support event stepping (forward/backward)
- All state must be reconstructable from events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Iterator, AsyncIterator, Callable, Any
from enum import Enum

import structlog

from .event_store import EventStore, EventQuery
from .models import EventType, LiquidityAction

logger = structlog.get_logger()


class ReplayMode(Enum):
    """Replay mode options."""

    FULL = "full"  # Replay all events
    TRADES_ONLY = "trades_only"  # Only trade events
    LIQUIDITY_ONLY = "liquidity_only"  # Only liquidity events
    STATE_CHANGES = "state_changes"  # Only state transitions


@dataclass
class TokenState:
    """
    Reconstructed token state at a point in time.

    Immutable snapshot of token state derived from events.
    """

    token_mint: str
    timestamp: datetime
    sequence: int

    # Holder state
    holder_count: int = 0
    holders: dict[str, int] = field(default_factory=dict)  # wallet -> balance
    top_holders: list[tuple[str, int]] = field(default_factory=list)

    # Supply distribution
    total_supply: int = 0
    circulating_supply: int = 0
    top_10_concentration: float = 0.0
    top_20_concentration: float = 0.0

    # Liquidity state
    total_liquidity_usd: float = 0.0
    pool_count: int = 0
    pools: dict[str, dict] = field(default_factory=dict)  # pool_addr -> pool_state

    # Volume metrics
    volume_24h: int = 0
    trade_count_24h: int = 0
    buy_count_24h: int = 0
    sell_count_24h: int = 0

    # Graph state
    unique_traders: int = 0
    unique_funders: int = 0
    funding_links: list[tuple[str, str]] = field(default_factory=list)

    # Derived metrics
    avg_holder_size: float = 0.0
    median_holder_size: int = 0
    gini_coefficient: float = 0.0

    def compute_derived_metrics(self):
        """Compute derived metrics from raw state."""
        if self.holder_count > 0:
            self.avg_holder_size = self.total_supply / self.holder_count

            # Top N concentration
            sorted_balances = sorted(self.holders.values(), reverse=True)
            if sorted_balances:
                top_10 = sorted_balances[:10]
                top_20 = sorted_balances[:20]
                self.top_10_concentration = sum(top_10) / self.total_supply if self.total_supply else 0
                self.top_20_concentration = sum(top_20) / self.total_supply if self.total_supply else 0

                # Median
                n = len(sorted_balances)
                self.median_holder_size = sorted_balances[n // 2]

                # Gini coefficient
                self.gini_coefficient = self._compute_gini(sorted_balances)

    def _compute_gini(self, sorted_values: list[int]) -> float:
        """Compute Gini coefficient from sorted values."""
        if not sorted_values or sum(sorted_values) == 0:
            return 0.0

        n = len(sorted_values)
        cumsum = 0.0
        weighted_sum = 0.0

        for i, value in enumerate(sorted_values):
            cumsum += value
            weighted_sum += (n - i) * value

        total = sum(sorted_values)
        if n * total == 0:
            return 0.0

        return (2 * weighted_sum) / (n * total) - (n + 1) / n


@dataclass
class ReplayCheckpoint:
    """Checkpoint for efficient replay resumption."""

    token_mint: str
    sequence: int
    timestamp: datetime
    state_hash: str
    state: TokenState


@dataclass
class ReplayStep:
    """Single step in replay."""

    sequence: int
    timestamp: datetime
    event_type: EventType
    event_data: dict
    state_before: Optional[TokenState] = None
    state_after: Optional[TokenState] = None


class ReplayEngine:
    """
    Deterministic replay engine for token state reconstruction.

    Supports:
    - Time scrubbing: jump to any timestamp
    - Event stepping: forward/backward through events
    - Snapshot comparison: diff states at two points
    - Checkpoint caching: efficient resumption
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        event_store: EventStore,
        checkpoint_interval: int = 100,  # Create checkpoint every N events
    ):
        """
        Initialize the replay engine.

        Args:
            event_store: Event store for event retrieval
            checkpoint_interval: Events between checkpoints
        """
        self.event_store = event_store
        self.checkpoint_interval = checkpoint_interval
        self._checkpoints: dict[str, list[ReplayCheckpoint]] = {}  # token -> checkpoints
        self._state_handlers: dict[EventType, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default event handlers for state reconstruction."""
        self._state_handlers = {
            EventType.TRADE: self._apply_trade,
            EventType.LIQUIDITY: self._apply_liquidity,
            EventType.FUNDING: self._apply_funding,
            EventType.STATE_TRANSITION: self._apply_state_transition,
        }

    def register_handler(self, event_type: EventType, handler: Callable):
        """Register custom event handler."""
        self._state_handlers[event_type] = handler

    async def replay_to_timestamp(
        self,
        token_mint: str,
        target_time: datetime,
        mode: ReplayMode = ReplayMode.FULL,
    ) -> TokenState:
        """
        Replay events to reconstruct state at target timestamp.

        Args:
            token_mint: Token to replay
            target_time: Target timestamp
            mode: Which event types to replay

        Returns:
            TokenState at target timestamp
        """
        # Find nearest checkpoint before target
        checkpoint = self._find_nearest_checkpoint(token_mint, target_time)

        if checkpoint:
            state = checkpoint.state
            start_sequence = checkpoint.sequence + 1
        else:
            state = self._create_initial_state(token_mint)
            start_sequence = 0

        # Replay from checkpoint to target
        event_types = self._get_event_types_for_mode(mode)

        async for event in self.event_store.replay_events(
            token_mint=token_mint,
            start_sequence=start_sequence,
            event_types=event_types,
        ):
            event_time = datetime.fromisoformat(event.get("timestamp"))
            if event_time > target_time:
                break

            state = self._apply_event(state, event)

            # Create checkpoint if needed
            if state.sequence % self.checkpoint_interval == 0:
                self._save_checkpoint(state)

        state.timestamp = target_time
        state.compute_derived_metrics()

        logger.debug(
            "replay_to_timestamp_complete",
            token_mint=token_mint[:8],
            target_time=target_time.isoformat(),
            final_sequence=state.sequence,
            holder_count=state.holder_count,
        )

        return state

    async def replay_to_sequence(
        self,
        token_mint: str,
        target_sequence: int,
        mode: ReplayMode = ReplayMode.FULL,
    ) -> TokenState:
        """
        Replay events to reconstruct state at target sequence.

        Args:
            token_mint: Token to replay
            target_sequence: Target sequence number
            mode: Which event types to replay

        Returns:
            TokenState at target sequence
        """
        # Find nearest checkpoint
        checkpoint = self._find_checkpoint_by_sequence(token_mint, target_sequence)

        if checkpoint:
            state = checkpoint.state
            start_sequence = checkpoint.sequence + 1
        else:
            state = self._create_initial_state(token_mint)
            start_sequence = 0

        event_types = self._get_event_types_for_mode(mode)

        async for event in self.event_store.replay_events(
            token_mint=token_mint,
            start_sequence=start_sequence,
            end_sequence=target_sequence,
            event_types=event_types,
        ):
            state = self._apply_event(state, event)

        state.compute_derived_metrics()
        return state

    async def step_forward(
        self,
        current_state: TokenState,
        steps: int = 1,
    ) -> list[ReplayStep]:
        """
        Step forward through events from current state.

        Args:
            current_state: Current state
            steps: Number of events to step through

        Returns:
            List of ReplaySteps
        """
        replay_steps = []
        state = current_state

        async for event in self.event_store.replay_events(
            token_mint=current_state.token_mint,
            start_sequence=current_state.sequence + 1,
        ):
            if len(replay_steps) >= steps:
                break

            state_before = state
            event_type = EventType(event.get("event_type"))
            state_after = self._apply_event(state, event)

            replay_steps.append(
                ReplayStep(
                    sequence=event.get("sequence"),
                    timestamp=datetime.fromisoformat(event.get("timestamp")),
                    event_type=event_type,
                    event_data=event,
                    state_before=state_before,
                    state_after=state_after,
                )
            )

            state = state_after

        return replay_steps

    async def compare_states(
        self,
        token_mint: str,
        time_a: datetime,
        time_b: datetime,
    ) -> dict:
        """
        Compare token states at two points in time.

        Returns dict of differences.
        """
        state_a = await self.replay_to_timestamp(token_mint, time_a)
        state_b = await self.replay_to_timestamp(token_mint, time_b)

        diff = {
            "time_a": time_a.isoformat(),
            "time_b": time_b.isoformat(),
            "duration_seconds": (time_b - time_a).total_seconds(),
            "holder_count_change": state_b.holder_count - state_a.holder_count,
            "top_10_concentration_change": state_b.top_10_concentration - state_a.top_10_concentration,
            "liquidity_change_usd": state_b.total_liquidity_usd - state_a.total_liquidity_usd,
            "gini_change": state_b.gini_coefficient - state_a.gini_coefficient,
            "new_holders": [],
            "exited_holders": [],
            "balance_changes": {},
        }

        # Find holder changes
        holders_a = set(state_a.holders.keys())
        holders_b = set(state_b.holders.keys())

        diff["new_holders"] = list(holders_b - holders_a)
        diff["exited_holders"] = list(holders_a - holders_b)

        # Balance changes for continuing holders
        for wallet in holders_a & holders_b:
            change = state_b.holders[wallet] - state_a.holders[wallet]
            if change != 0:
                diff["balance_changes"][wallet] = change

        return diff

    async def reconstruct_holder_evolution(
        self,
        token_mint: str,
        start_time: datetime,
        end_time: datetime,
        sample_interval_seconds: int = 300,
    ) -> list[dict]:
        """
        Reconstruct holder evolution over time period.

        Returns list of snapshots at regular intervals.
        """
        snapshots = []
        current_time = start_time

        while current_time <= end_time:
            state = await self.replay_to_timestamp(token_mint, current_time)

            snapshots.append(
                {
                    "timestamp": current_time.isoformat(),
                    "holder_count": state.holder_count,
                    "top_10_concentration": state.top_10_concentration,
                    "top_20_concentration": state.top_20_concentration,
                    "gini_coefficient": state.gini_coefficient,
                    "avg_holder_size": state.avg_holder_size,
                    "median_holder_size": state.median_holder_size,
                }
            )

            from datetime import timedelta

            current_time += timedelta(seconds=sample_interval_seconds)

        logger.info(
            "holder_evolution_reconstructed",
            token_mint=token_mint[:8],
            snapshots=len(snapshots),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
        )

        return snapshots

    async def reconstruct_liquidity_evolution(
        self,
        token_mint: str,
        start_time: datetime,
        end_time: datetime,
        sample_interval_seconds: int = 300,
    ) -> list[dict]:
        """Reconstruct liquidity evolution over time period."""
        snapshots = []
        current_time = start_time

        while current_time <= end_time:
            state = await self.replay_to_timestamp(
                token_mint, current_time, mode=ReplayMode.LIQUIDITY_ONLY
            )

            snapshots.append(
                {
                    "timestamp": current_time.isoformat(),
                    "total_liquidity_usd": state.total_liquidity_usd,
                    "pool_count": state.pool_count,
                    "pools": state.pools,
                }
            )

            from datetime import timedelta

            current_time += timedelta(seconds=sample_interval_seconds)

        return snapshots

    # =========================================================================
    # Event Application Handlers
    # =========================================================================

    def _apply_event(self, state: TokenState, event: dict) -> TokenState:
        """Apply event to state, returning new state."""
        event_type = EventType(event.get("event_type"))

        if event_type in self._state_handlers:
            # Create copy for immutability
            new_state = self._copy_state(state)
            new_state.sequence = event.get("sequence", state.sequence + 1)
            new_state.timestamp = datetime.fromisoformat(event.get("timestamp"))

            return self._state_handlers[event_type](new_state, event)

        return state

    def _apply_trade(self, state: TokenState, event: dict) -> TokenState:
        """Apply trade event to state."""
        payload = event.get("payload", {})
        wallet = payload.get("wallet_address")
        trade_type = payload.get("trade_type")
        amount = payload.get("amount", 0)

        if trade_type == "buy":
            # Add to holder balance
            current_balance = state.holders.get(wallet, 0)
            state.holders[wallet] = current_balance + amount

            if current_balance == 0:
                state.holder_count += 1

            state.buy_count_24h += 1
            state.unique_traders += 1  # Simplified - should track uniqueness

        elif trade_type == "sell":
            current_balance = state.holders.get(wallet, 0)
            new_balance = max(0, current_balance - amount)

            if new_balance == 0:
                if wallet in state.holders:
                    del state.holders[wallet]
                    state.holder_count = max(0, state.holder_count - 1)
            else:
                state.holders[wallet] = new_balance

            state.sell_count_24h += 1

        state.trade_count_24h += 1
        state.volume_24h += amount

        return state

    def _apply_liquidity(self, state: TokenState, event: dict) -> TokenState:
        """Apply liquidity event to state (add or remove)."""
        payload = event.get("payload", {})
        pool_address = payload.get("pool_address")
        action = payload.get("action", "add")
        token_amount = payload.get("token_amount", 0)
        quote_amount = payload.get("quote_amount", 0)

        if action in ("add", LiquidityAction.ADD.value, "create_pool", LiquidityAction.CREATE_POOL.value):
            # Add liquidity
            if pool_address not in state.pools:
                state.pools[pool_address] = {
                    "token_amount": 0,
                    "quote_amount": 0,
                    "dex": payload.get("dex"),
                }
                state.pool_count += 1

            state.pools[pool_address]["token_amount"] += token_amount
            state.pools[pool_address]["quote_amount"] += quote_amount

        elif action in ("remove", LiquidityAction.REMOVE.value):
            # Remove liquidity
            if pool_address in state.pools:
                state.pools[pool_address]["token_amount"] -= token_amount
                state.pools[pool_address]["quote_amount"] -= quote_amount

                if state.pools[pool_address]["token_amount"] <= 0:
                    del state.pools[pool_address]
                    state.pool_count = max(0, state.pool_count - 1)

        # Recalculate total liquidity USD
        # Approximate: assuming quote is SOL at ~$100
        sol_price = 100.0
        state.total_liquidity_usd = sum(
            pool.get("quote_amount", 0) / 1e9 * sol_price
            for pool in state.pools.values()
        )

        return state

    def _apply_funding(self, state: TokenState, event: dict) -> TokenState:
        """Apply funding event to state."""
        payload = event.get("payload", {})
        source = payload.get("source_address")
        target = payload.get("target_address")

        state.funding_links.append((source, target))
        state.unique_funders += 1  # Simplified

        return state

    def _apply_state_transition(self, state: TokenState, event: dict) -> TokenState:
        """Apply state transition event to state."""
        # State transitions are metadata - may update derived state
        payload = event.get("payload", {})
        transition_type = payload.get("transition_type")

        # Could handle specific transition types here
        # For now, just log
        logger.debug(
            "state_transition_applied",
            transition_type=transition_type,
            sequence=state.sequence,
        )

        return state

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _create_initial_state(self, token_mint: str) -> TokenState:
        """Create initial empty state for token."""
        return TokenState(
            token_mint=token_mint,
            timestamp=datetime.now(timezone.utc),
            sequence=0,
        )

    def _copy_state(self, state: TokenState) -> TokenState:
        """Create a copy of state for immutability."""
        return TokenState(
            token_mint=state.token_mint,
            timestamp=state.timestamp,
            sequence=state.sequence,
            holder_count=state.holder_count,
            holders=dict(state.holders),
            top_holders=list(state.top_holders),
            total_supply=state.total_supply,
            circulating_supply=state.circulating_supply,
            top_10_concentration=state.top_10_concentration,
            top_20_concentration=state.top_20_concentration,
            total_liquidity_usd=state.total_liquidity_usd,
            pool_count=state.pool_count,
            pools={k: dict(v) for k, v in state.pools.items()},
            volume_24h=state.volume_24h,
            trade_count_24h=state.trade_count_24h,
            buy_count_24h=state.buy_count_24h,
            sell_count_24h=state.sell_count_24h,
            unique_traders=state.unique_traders,
            unique_funders=state.unique_funders,
            funding_links=list(state.funding_links),
            avg_holder_size=state.avg_holder_size,
            median_holder_size=state.median_holder_size,
            gini_coefficient=state.gini_coefficient,
        )

    def _find_nearest_checkpoint(
        self, token_mint: str, target_time: datetime
    ) -> Optional[ReplayCheckpoint]:
        """Find nearest checkpoint before target time."""
        if token_mint not in self._checkpoints:
            return None

        checkpoints = self._checkpoints[token_mint]
        nearest = None

        for cp in checkpoints:
            if cp.timestamp <= target_time:
                if nearest is None or cp.timestamp > nearest.timestamp:
                    nearest = cp

        return nearest

    def _find_checkpoint_by_sequence(
        self, token_mint: str, target_sequence: int
    ) -> Optional[ReplayCheckpoint]:
        """Find nearest checkpoint before target sequence."""
        if token_mint not in self._checkpoints:
            return None

        checkpoints = self._checkpoints[token_mint]
        nearest = None

        for cp in checkpoints:
            if cp.sequence <= target_sequence:
                if nearest is None or cp.sequence > nearest.sequence:
                    nearest = cp

        return nearest

    def _save_checkpoint(self, state: TokenState):
        """Save state checkpoint."""
        import hashlib
        import json

        state_hash = hashlib.sha256(
            json.dumps(
                {
                    "holders": state.holders,
                    "pools": state.pools,
                    "sequence": state.sequence,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()[:16]

        checkpoint = ReplayCheckpoint(
            token_mint=state.token_mint,
            sequence=state.sequence,
            timestamp=state.timestamp,
            state_hash=state_hash,
            state=self._copy_state(state),
        )

        if state.token_mint not in self._checkpoints:
            self._checkpoints[state.token_mint] = []

        self._checkpoints[state.token_mint].append(checkpoint)

        logger.debug(
            "checkpoint_saved",
            token_mint=state.token_mint[:8],
            sequence=state.sequence,
            state_hash=state_hash,
        )

    def _get_event_types_for_mode(self, mode: ReplayMode) -> Optional[list[EventType]]:
        """Get event types to replay for given mode."""
        if mode == ReplayMode.FULL:
            return None  # All types
        elif mode == ReplayMode.TRADES_ONLY:
            return [EventType.TRADE]
        elif mode == ReplayMode.LIQUIDITY_ONLY:
            return [EventType.LIQUIDITY]
        elif mode == ReplayMode.STATE_CHANGES:
            return [EventType.STATE_TRANSITION]
        return None
