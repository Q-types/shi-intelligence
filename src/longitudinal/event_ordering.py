"""
Deterministic Event Ordering for Longitudinal Intelligence.

Defines canonical ordering for events to ensure replay determinism.

CANONICAL ORDERING (priority order):
1. slot - Solana slot number
2. block_time - Block timestamp
3. transaction_index - Index within block
4. instruction_index - Index within transaction
5. signature - Transaction signature (tiebreaker)
6. event_type - Event type ordering fallback

HARD RULES:
1. Replay must produce identical state regardless of insertion order
2. Ordering must be deterministic and reproducible
3. Events with same ordering key are ordered by event_type
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any
from enum import Enum
import hashlib

import structlog

from .models import EventType

logger = structlog.get_logger()


# Event type ordering for deterministic tiebreaking
EVENT_TYPE_ORDER = {
    EventType.FUNDING: 1,  # Funding happens before trades
    EventType.LIQUIDITY: 2,  # Liquidity setup before trades
    EventType.TRADE: 3,  # Trades
    EventType.STATE_TRANSITION: 4,  # Derived after facts
    EventType.EVENT_CORRECTION: 5,  # Corrections
    EventType.EVENT_INVALIDATION: 6,  # Invalidations
    EventType.BACKFILL_INSERT: 7,  # Backfills
    EventType.PROVIDER_RECONCILIATION: 8,  # Reconciliations
}


@dataclass
class EventOrderingKey:
    """
    Canonical ordering key for an event.

    Events are ordered by this key for deterministic replay.
    """

    slot: int
    block_time: datetime
    transaction_index: int
    instruction_index: int
    signature: str
    event_type: EventType
    data_version: int = 1

    def __lt__(self, other: "EventOrderingKey") -> bool:
        """Compare ordering keys for sorting."""
        # 1. Slot
        if self.slot != other.slot:
            return self.slot < other.slot

        # 2. Block time (if slots equal)
        if self.block_time != other.block_time:
            return self.block_time < other.block_time

        # 3. Transaction index
        if self.transaction_index != other.transaction_index:
            return self.transaction_index < other.transaction_index

        # 4. Instruction index
        if self.instruction_index != other.instruction_index:
            return self.instruction_index < other.instruction_index

        # 5. Signature (deterministic string comparison)
        if self.signature != other.signature:
            return self.signature < other.signature

        # 6. Event type ordering
        type_order_self = EVENT_TYPE_ORDER.get(self.event_type, 99)
        type_order_other = EVENT_TYPE_ORDER.get(other.event_type, 99)
        if type_order_self != type_order_other:
            return type_order_self < type_order_other

        # 7. Data version (for corrections)
        return self.data_version < other.data_version

    def __eq__(self, other: "EventOrderingKey") -> bool:
        return (
            self.slot == other.slot and
            self.block_time == other.block_time and
            self.transaction_index == other.transaction_index and
            self.instruction_index == other.instruction_index and
            self.signature == other.signature and
            self.event_type == other.event_type and
            self.data_version == other.data_version
        )

    def __hash__(self) -> int:
        return hash((
            self.slot,
            self.block_time,
            self.transaction_index,
            self.instruction_index,
            self.signature,
            self.event_type,
            self.data_version,
        ))

    def to_sort_key(self) -> tuple:
        """Convert to sortable tuple."""
        return (
            self.slot,
            self.block_time,
            self.transaction_index,
            self.instruction_index,
            self.signature,
            EVENT_TYPE_ORDER.get(self.event_type, 99),
            self.data_version,
        )


class ReplayMode(str, Enum):
    """Replay mode options."""

    NORMAL = "normal"  # Standard replay, includes corrections
    RAW = "raw"  # Replay without applying corrections
    AS_OF_VERSION = "as_of_version"  # Replay as of specific data version
    EXCLUDE_INVALIDATED = "exclude_invalidated"  # Exclude invalidated events


@dataclass
class ReplayConfig:
    """Configuration for event replay."""

    mode: ReplayMode = ReplayMode.NORMAL
    as_of_version: Optional[int] = None  # For AS_OF_VERSION mode
    include_corrections: bool = True
    skip_invalidated: bool = True
    event_types: Optional[list[EventType]] = None


class EventOrderer:
    """
    Ensures deterministic event ordering for replay.

    Handles:
    - Canonical ordering by slot/time/index/signature
    - Event corrections and invalidations
    - Data version handling
    """

    def __init__(self):
        self._ordering_cache: dict[str, EventOrderingKey] = {}

    def get_ordering_key(self, event: dict) -> EventOrderingKey:
        """
        Extract ordering key from event dict.

        Args:
            event: Event dictionary with ordering fields

        Returns:
            EventOrderingKey for sorting
        """
        return EventOrderingKey(
            slot=event.get("slot", 0),
            block_time=event.get("block_time") or event.get("timestamp"),
            transaction_index=event.get("transaction_index", 0),
            instruction_index=event.get("instruction_index", 0),
            signature=event.get("signature", ""),
            event_type=EventType(event.get("event_type")),
            data_version=event.get("data_version", 1),
        )

    def sort_events(
        self,
        events: list[dict],
        config: Optional[ReplayConfig] = None,
    ) -> list[dict]:
        """
        Sort events in canonical order.

        Args:
            events: List of event dicts
            config: Replay configuration

        Returns:
            Sorted list of events
        """
        config = config or ReplayConfig()

        # Filter based on config
        filtered = self._filter_events(events, config)

        # Sort by ordering key
        sorted_events = sorted(
            filtered,
            key=lambda e: self.get_ordering_key(e).to_sort_key()
        )

        return sorted_events

    def _filter_events(
        self,
        events: list[dict],
        config: ReplayConfig,
    ) -> list[dict]:
        """Filter events based on replay config."""
        filtered = events

        # Skip invalidated events
        if config.skip_invalidated:
            filtered = [e for e in filtered if not e.get("is_invalidated", False)]

        # Filter by data version
        if config.mode == ReplayMode.AS_OF_VERSION and config.as_of_version is not None:
            filtered = [
                e for e in filtered
                if e.get("data_version", 1) <= config.as_of_version
            ]

        # Filter by event types
        if config.event_types:
            type_values = [t.value for t in config.event_types]
            filtered = [e for e in filtered if e.get("event_type") in type_values]

        # Handle corrections
        if config.mode == ReplayMode.RAW:
            # Exclude correction events
            filtered = [
                e for e in filtered
                if e.get("event_type") not in [
                    EventType.EVENT_CORRECTION.value,
                    EventType.EVENT_INVALIDATION.value,
                ]
            ]
        elif config.include_corrections:
            # Include correction events (they apply to earlier events)
            pass  # No filtering needed

        return filtered

    def apply_corrections(
        self,
        events: list[dict],
    ) -> list[dict]:
        """
        Apply correction events to produce corrected event stream.

        Corrections don't modify original events - they supersede them.

        Args:
            events: Sorted list of events including corrections

        Returns:
            Event stream with corrections applied
        """
        # Build correction map: corrected_event_id -> correction_event
        correction_map: dict[int, dict] = {}
        invalidation_set: set[int] = set()

        for event in events:
            event_type = event.get("event_type")

            if event_type == EventType.EVENT_CORRECTION.value:
                corrects_id = event.get("corrects_event_id")
                if corrects_id:
                    correction_map[corrects_id] = event

            elif event_type == EventType.EVENT_INVALIDATION.value:
                invalidated_id = event.get("payload", {}).get("invalidated_event_id")
                if invalidated_id:
                    invalidation_set.add(invalidated_id)

        # Build corrected stream
        corrected = []

        for event in events:
            event_id = event.get("id")
            event_type = event.get("event_type")

            # Skip correction/invalidation meta-events
            if event_type in [
                EventType.EVENT_CORRECTION.value,
                EventType.EVENT_INVALIDATION.value,
            ]:
                continue

            # Skip invalidated events
            if event_id in invalidation_set:
                logger.debug("skipping_invalidated_event", event_id=event_id)
                continue

            # Apply correction if exists
            if event_id in correction_map:
                correction = correction_map[event_id]
                # Use corrected payload
                corrected_event = event.copy()
                corrected_event["payload"] = correction.get("payload", {}).get(
                    "corrected_payload", event.get("payload")
                )
                corrected_event["_corrected"] = True
                corrected_event["_correction_id"] = correction.get("id")
                corrected.append(corrected_event)
            else:
                corrected.append(event)

        return corrected

    def verify_ordering_determinism(
        self,
        events_a: list[dict],
        events_b: list[dict],
    ) -> bool:
        """
        Verify that two event lists produce the same ordering.

        Used for testing determinism.

        Args:
            events_a: First event list
            events_b: Second event list (potentially different insertion order)

        Returns:
            True if orderings match
        """
        sorted_a = self.sort_events(events_a)
        sorted_b = self.sort_events(events_b)

        if len(sorted_a) != len(sorted_b):
            return False

        for a, b in zip(sorted_a, sorted_b):
            key_a = self.get_ordering_key(a)
            key_b = self.get_ordering_key(b)

            if key_a != key_b:
                logger.warning(
                    "ordering_mismatch",
                    key_a=key_a,
                    key_b=key_b,
                )
                return False

        return True

    def compute_ordering_hash(self, events: list[dict]) -> str:
        """
        Compute deterministic hash of event ordering.

        Used for verification across replays.
        """
        sorted_events = self.sort_events(events)

        # Build hash input from ordering keys
        keys = []
        for event in sorted_events:
            key = self.get_ordering_key(event)
            keys.append(f"{key.slot}:{key.signature}:{key.event_type.value}")

        hash_input = "|".join(keys)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


class CorrectionEventBuilder:
    """
    Builder for correction events.

    Ensures corrections are properly formatted.
    """

    @staticmethod
    def create_correction(
        original_event_id: int,
        corrected_payload: dict,
        reason: str,
        source: str = "manual",
    ) -> dict:
        """
        Create a correction event.

        Args:
            original_event_id: ID of event being corrected
            corrected_payload: New payload to replace original
            reason: Reason for correction
            source: Source of correction

        Returns:
            Correction event dict
        """
        from datetime import datetime, timezone

        return {
            "event_type": EventType.EVENT_CORRECTION.value,
            "corrects_event_id": original_event_id,
            "timestamp": datetime.now(timezone.utc),
            "payload": {
                "corrected_payload": corrected_payload,
                "reason": reason,
                "source": source,
            },
            "source": source,
        }

    @staticmethod
    def create_invalidation(
        original_event_id: int,
        reason: str,
        source: str = "manual",
    ) -> dict:
        """
        Create an invalidation event.

        Args:
            original_event_id: ID of event being invalidated
            reason: Reason for invalidation
            source: Source of invalidation

        Returns:
            Invalidation event dict
        """
        from datetime import datetime, timezone

        return {
            "event_type": EventType.EVENT_INVALIDATION.value,
            "timestamp": datetime.now(timezone.utc),
            "payload": {
                "invalidated_event_id": original_event_id,
                "reason": reason,
                "source": source,
            },
            "source": source,
        }

    @staticmethod
    def create_backfill_insert(
        original_event: dict,
        backfill_source: str,
        backfill_reason: str,
    ) -> dict:
        """
        Create a backfill insert event.

        Used when inserting historical data that was missed.

        Args:
            original_event: The event being backfilled
            backfill_source: Source of backfill data
            backfill_reason: Why backfill was needed

        Returns:
            Backfill insert event dict
        """
        from datetime import datetime, timezone

        backfill_event = original_event.copy()
        backfill_event["event_type"] = EventType.BACKFILL_INSERT.value
        backfill_event["source"] = f"backfill:{backfill_source}"
        backfill_event["payload"] = {
            **original_event.get("payload", {}),
            "_backfill_metadata": {
                "original_event_type": original_event.get("event_type"),
                "backfill_source": backfill_source,
                "backfill_reason": backfill_reason,
                "backfill_time": datetime.now(timezone.utc).isoformat(),
            },
        }

        return backfill_event

    @staticmethod
    def create_provider_reconciliation(
        token_mint: str,
        reconciliation_result: dict,
        provider: str,
    ) -> dict:
        """
        Create a provider reconciliation event.

        Records when data from different providers is reconciled.

        Args:
            token_mint: Token being reconciled
            reconciliation_result: Result of reconciliation
            provider: Provider that triggered reconciliation

        Returns:
            Reconciliation event dict
        """
        from datetime import datetime, timezone

        return {
            "event_type": EventType.PROVIDER_RECONCILIATION.value,
            "token_mint": token_mint,
            "timestamp": datetime.now(timezone.utc),
            "payload": {
                "reconciliation_result": reconciliation_result,
                "provider": provider,
                "reconciled_at": datetime.now(timezone.utc).isoformat(),
            },
            "source": f"reconciliation:{provider}",
        }
