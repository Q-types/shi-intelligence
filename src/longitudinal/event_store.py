"""
Event Store for Longitudinal Intelligence.

Event-sourced storage for raw events: trades, liquidity, funding, state transitions.

HARD RULES:
- Events are immutable and append-only
- Never update or delete events
- All derived metrics must be recomputable from events
- Events have monotonic sequence numbers per token
- Deduplication via event hash
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Iterator, AsyncIterator, Callable, Any
import hashlib
import json

import structlog

from .models import (
    EventType,
    TradeType,
    LiquidityAction,
    StateTransitionType,
)

logger = structlog.get_logger()


@dataclass
class TradeEvent:
    """Trade event data structure."""

    token_mint: str
    wallet_address: str
    trade_type: TradeType
    amount: int  # Token amount
    timestamp: datetime
    signature: str
    slot: int
    price_usd: Optional[float] = None
    dex: Optional[str] = None
    pool_address: Optional[str] = None


@dataclass
class LiquidityEvent:
    """Liquidity event data structure."""

    token_mint: str
    wallet_address: str
    action: LiquidityAction
    pool_address: str
    timestamp: datetime
    signature: str
    slot: int
    token_amount: int
    quote_amount: int  # SOL/USDC in lamports
    lp_tokens: int = 0
    dex: Optional[str] = None


@dataclass
class FundingEvent:
    """Funding transfer event data structure."""

    source_address: str
    target_address: str
    amount_lamports: int
    timestamp: datetime
    signature: str
    slot: int
    token_mint: Optional[str] = None  # Optional context token


@dataclass
class StateTransition:
    """State transition event data structure."""

    token_mint: str
    transition_type: StateTransitionType
    timestamp: datetime
    wallet_address: Optional[str] = None
    old_state: dict = field(default_factory=dict)
    new_state: dict = field(default_factory=dict)
    confidence: float = 1.0
    evidence: dict = field(default_factory=dict)


@dataclass
class EventBatch:
    """Batch of events for bulk insertion."""

    events: list[Any]
    source: str
    batch_id: str


@dataclass
class EventQuery:
    """Query parameters for event retrieval."""

    token_mint: Optional[str] = None
    wallet_address: Optional[str] = None
    event_types: Optional[list[EventType]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    start_sequence: Optional[int] = None
    end_sequence: Optional[int] = None
    limit: int = 1000
    offset: int = 0


@dataclass
class EventReplayState:
    """State for event replay cursor."""

    token_mint: str
    last_sequence: int
    last_timestamp: datetime
    events_processed: int


class EventStore:
    """
    Append-only event store for raw events.

    Supports:
    - Deduplication via event hash
    - Monotonic sequence numbers per token
    - Time-range queries
    - Event replay
    """

    VERSION = "1.0.0"

    def __init__(self, session_factory=None):
        """
        Initialize the event store.

        Args:
            session_factory: SQLAlchemy async session factory
        """
        self.session_factory = session_factory
        self._sequence_cache: dict[str, int] = {}  # token_mint -> last_sequence

    def compute_event_hash(self, event: Any) -> str:
        """
        Compute unique hash for event deduplication.

        Hash is based on signature + token_mint + event_type.
        """
        if hasattr(event, "signature"):
            key = f"{event.signature}:{getattr(event, 'token_mint', '')}:{type(event).__name__}"
        else:
            # For state transitions without signatures
            key = json.dumps(
                {
                    "type": type(event).__name__,
                    "token_mint": getattr(event, "token_mint", ""),
                    "timestamp": str(event.timestamp),
                    "data": str(getattr(event, "new_state", {})),
                },
                sort_keys=True,
            )

        return hashlib.sha256(key.encode()).hexdigest()[:64]

    async def get_next_sequence(self, token_mint: str) -> int:
        """
        Get next sequence number for a token.

        Sequence numbers are monotonically increasing per token.
        """
        if token_mint in self._sequence_cache:
            self._sequence_cache[token_mint] += 1
            return self._sequence_cache[token_mint]

        # TODO: Query database for last sequence
        # For now, initialize at 0
        self._sequence_cache[token_mint] = 1
        return 1

    async def append_trade(self, event: TradeEvent, source: str = "rpc") -> Optional[int]:
        """
        Append a trade event to the store.

        Returns event ID if successful, None if duplicate.
        """
        event_hash = self.compute_event_hash(event)

        # Check for duplicate
        if await self._event_exists(event_hash):
            logger.debug("duplicate_trade_event", signature=event.signature[:16])
            return None

        sequence = await self.get_next_sequence(event.token_mint)

        payload = {
            "wallet_address": event.wallet_address,
            "trade_type": event.trade_type.value,
            "amount": event.amount,
            "price_usd": event.price_usd,
            "dex": event.dex,
            "pool_address": event.pool_address,
        }

        # TODO: Insert into database
        logger.debug(
            "trade_event_appended",
            token_mint=event.token_mint[:8],
            sequence=sequence,
            trade_type=event.trade_type.value,
        )

        return sequence

    async def append_liquidity(
        self, event: LiquidityEvent, source: str = "rpc"
    ) -> Optional[int]:
        """Append a liquidity event to the store."""
        event_hash = self.compute_event_hash(event)

        if await self._event_exists(event_hash):
            return None

        sequence = await self.get_next_sequence(event.token_mint)

        payload = {
            "wallet_address": event.wallet_address,
            "action": event.action.value,
            "pool_address": event.pool_address,
            "token_amount": event.token_amount,
            "quote_amount": event.quote_amount,
            "lp_tokens": event.lp_tokens,
            "dex": event.dex,
        }

        logger.debug(
            "liquidity_event_appended",
            token_mint=event.token_mint[:8],
            sequence=sequence,
            action=event.action.value,
        )

        return sequence

    async def append_funding(
        self, event: FundingEvent, source: str = "rpc"
    ) -> Optional[int]:
        """Append a funding event to the store."""
        event_hash = self.compute_event_hash(event)

        if await self._event_exists(event_hash):
            return None

        token_mint = event.token_mint or "global"
        sequence = await self.get_next_sequence(token_mint)

        payload = {
            "source_address": event.source_address,
            "target_address": event.target_address,
            "amount_lamports": event.amount_lamports,
        }

        logger.debug(
            "funding_event_appended",
            source=event.source_address[:8],
            target=event.target_address[:8],
            sequence=sequence,
        )

        return sequence

    async def append_state_transition(
        self, event: StateTransition, source: str = "analytics"
    ) -> Optional[int]:
        """Append a state transition event to the store."""
        event_hash = self.compute_event_hash(event)

        if await self._event_exists(event_hash):
            return None

        sequence = await self.get_next_sequence(event.token_mint)

        payload = {
            "transition_type": event.transition_type.value,
            "wallet_address": event.wallet_address,
            "old_state": event.old_state,
            "new_state": event.new_state,
            "confidence": event.confidence,
            "evidence": event.evidence,
        }

        logger.debug(
            "state_transition_appended",
            token_mint=event.token_mint[:8],
            transition_type=event.transition_type.value,
            sequence=sequence,
        )

        return sequence

    async def append_batch(self, batch: EventBatch) -> int:
        """
        Append a batch of events atomically.

        Returns number of events successfully appended.
        """
        appended = 0

        for event in batch.events:
            if isinstance(event, TradeEvent):
                result = await self.append_trade(event, batch.source)
            elif isinstance(event, LiquidityEvent):
                result = await self.append_liquidity(event, batch.source)
            elif isinstance(event, FundingEvent):
                result = await self.append_funding(event, batch.source)
            elif isinstance(event, StateTransition):
                result = await self.append_state_transition(event, batch.source)
            else:
                logger.warning("unknown_event_type", event_type=type(event).__name__)
                continue

            if result is not None:
                appended += 1

        logger.info(
            "event_batch_appended",
            batch_id=batch.batch_id,
            total=len(batch.events),
            appended=appended,
        )

        return appended

    async def query_events(self, query: EventQuery) -> list[dict]:
        """
        Query events matching criteria.

        Returns list of event dicts.
        """
        # TODO: Implement actual database query
        logger.debug("querying_events", token_mint=query.token_mint[:8] if query.token_mint else None)
        return []

    async def replay_events(
        self,
        token_mint: str,
        start_sequence: int = 0,
        end_sequence: Optional[int] = None,
        event_types: Optional[list[EventType]] = None,
    ) -> AsyncIterator[dict]:
        """
        Replay events in sequence order for deterministic reconstruction.

        Yields events one at a time in sequence order.
        """
        query = EventQuery(
            token_mint=token_mint,
            event_types=event_types,
            start_sequence=start_sequence,
            end_sequence=end_sequence,
            limit=1000,
        )

        offset = 0
        while True:
            query.offset = offset
            events = await self.query_events(query)

            if not events:
                break

            for event in events:
                yield event

            if len(events) < query.limit:
                break

            offset += len(events)

    async def get_event_count(
        self,
        token_mint: str,
        event_types: Optional[list[EventType]] = None,
    ) -> int:
        """Get total event count for a token."""
        # TODO: Implement actual count query
        return 0

    async def get_latest_sequence(self, token_mint: str) -> int:
        """Get the latest sequence number for a token."""
        if token_mint in self._sequence_cache:
            return self._sequence_cache[token_mint]

        # TODO: Query database
        return 0

    async def _event_exists(self, event_hash: str) -> bool:
        """Check if event with hash already exists."""
        # TODO: Implement actual duplicate check
        return False


class EventReplayer:
    """
    Utility for replaying events to reconstruct state.

    Supports deterministic reconstruction of token state at any point in time.
    """

    def __init__(self, event_store: EventStore):
        self.store = event_store
        self._state_handlers: dict[EventType, Callable] = {}

    def register_handler(self, event_type: EventType, handler: Callable):
        """Register a handler for processing events during replay."""
        self._state_handlers[event_type] = handler

    async def replay_to_sequence(
        self,
        token_mint: str,
        target_sequence: int,
        initial_state: Optional[dict] = None,
    ) -> dict:
        """
        Replay events up to target sequence and return final state.

        Args:
            token_mint: Token to replay
            target_sequence: Sequence number to replay up to
            initial_state: Starting state (default: empty)

        Returns:
            Final state after replay
        """
        state = initial_state or {}

        async for event in self.store.replay_events(
            token_mint=token_mint,
            end_sequence=target_sequence,
        ):
            event_type = EventType(event.get("event_type"))
            if event_type in self._state_handlers:
                state = self._state_handlers[event_type](state, event)

        return state

    async def replay_to_timestamp(
        self,
        token_mint: str,
        target_time: datetime,
        initial_state: Optional[dict] = None,
    ) -> dict:
        """
        Replay events up to target timestamp and return final state.

        Args:
            token_mint: Token to replay
            target_time: Timestamp to replay up to
            initial_state: Starting state (default: empty)

        Returns:
            Final state after replay
        """
        state = initial_state or {}

        async for event in self.store.replay_events(token_mint=token_mint):
            event_time = datetime.fromisoformat(event.get("timestamp"))
            if event_time > target_time:
                break

            event_type = EventType(event.get("event_type"))
            if event_type in self._state_handlers:
                state = self._state_handlers[event_type](state, event)

        return state
