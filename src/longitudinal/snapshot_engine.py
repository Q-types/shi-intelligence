"""
Snapshot Engine for Longitudinal Intelligence.

Configurable snapshot collection with dynamic cadence based on token lifecycle.

Cadence Schedule:
- 0-1h from launch: 1 minute (early launch, high activity)
- 1-6h: 5 minutes
- 6-24h: 15 minutes
- 1-7d: 1 hour
- 7d+: 6 hours (mature token)

HARD RULES:
- Never overwrite historical snapshots
- Snapshots are versioned and immutable
- All snapshot types are configurable per token
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable, Any
from enum import Enum
import hashlib
import json

import structlog

logger = structlog.get_logger()


class SnapshotCadence(Enum):
    """Snapshot cadence levels."""

    EARLY_LAUNCH = 60  # 1 minute
    ACTIVE = 300  # 5 minutes
    MATURING = 900  # 15 minutes
    STABLE = 3600  # 1 hour
    DORMANT = 21600  # 6 hours


@dataclass
class CadenceThresholds:
    """Time thresholds for cadence transitions."""

    early_to_active_hours: float = 1.0
    active_to_maturing_hours: float = 6.0
    maturing_to_stable_hours: float = 24.0
    stable_to_dormant_days: float = 7.0


@dataclass
class SnapshotType:
    """Configuration for a snapshot type."""

    name: str
    collector: Callable[..., Any]  # Function to collect snapshot data
    enabled: bool = True
    priority: int = 1  # Lower = higher priority


@dataclass
class SnapshotResult:
    """Result of a snapshot collection."""

    token_mint: str
    timestamp: datetime
    snapshot_types: list[str]
    duration_ms: float
    success: bool
    errors: list[str] = field(default_factory=list)
    snapshot_ids: dict[str, int] = field(default_factory=dict)


class SnapshotEngine:
    """
    Engine for collecting and managing snapshots.

    Supports dynamic cadence based on token lifecycle and configurable
    snapshot types.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        session_factory=None,
        cadence_thresholds: Optional[CadenceThresholds] = None,
    ):
        """
        Initialize the snapshot engine.

        Args:
            session_factory: SQLAlchemy async session factory
            cadence_thresholds: Custom cadence thresholds
        """
        self.session_factory = session_factory
        self.thresholds = cadence_thresholds or CadenceThresholds()
        self._snapshot_collectors: dict[str, SnapshotType] = {}
        self._register_default_collectors()

    def _register_default_collectors(self):
        """Register default snapshot collectors."""
        # These will be wired to actual implementations
        self._snapshot_collectors = {
            "holders": SnapshotType(
                name="holders",
                collector=self._collect_holders,
                priority=1,
            ),
            "balances": SnapshotType(
                name="balances",
                collector=self._collect_balances,
                priority=1,
            ),
            "liquidity": SnapshotType(
                name="liquidity",
                collector=self._collect_liquidity,
                priority=2,
            ),
            "volume": SnapshotType(
                name="volume",
                collector=self._collect_volume,
                priority=2,
            ),
            "graph": SnapshotType(
                name="graph",
                collector=self._collect_graph,
                priority=3,
            ),
            "archetypes": SnapshotType(
                name="archetypes",
                collector=self._collect_archetypes,
                priority=3,
            ),
            "coordination": SnapshotType(
                name="coordination",
                collector=self._collect_coordination,
                priority=4,
            ),
            "trajectory": SnapshotType(
                name="trajectory",
                collector=self._collect_trajectory,
                priority=4,
            ),
        }

    def register_collector(self, snapshot_type: SnapshotType):
        """Register a custom snapshot collector."""
        self._snapshot_collectors[snapshot_type.name] = snapshot_type

    def determine_cadence(
        self,
        launch_time: datetime,
        current_time: Optional[datetime] = None,
    ) -> SnapshotCadence:
        """
        Determine appropriate snapshot cadence based on token age.

        Args:
            launch_time: When the token launched (first trade)
            current_time: Current time (default: now)

        Returns:
            Appropriate SnapshotCadence
        """
        current_time = current_time or datetime.now(timezone.utc)
        age = current_time - launch_time
        hours = age.total_seconds() / 3600

        if hours < self.thresholds.early_to_active_hours:
            return SnapshotCadence.EARLY_LAUNCH
        elif hours < self.thresholds.active_to_maturing_hours:
            return SnapshotCadence.ACTIVE
        elif hours < self.thresholds.maturing_to_stable_hours:
            return SnapshotCadence.MATURING
        elif hours < self.thresholds.stable_to_dormant_days * 24:
            return SnapshotCadence.STABLE
        else:
            return SnapshotCadence.DORMANT

    def get_next_snapshot_time(
        self,
        last_snapshot: datetime,
        cadence: SnapshotCadence,
    ) -> datetime:
        """Calculate when the next snapshot should be taken."""
        return last_snapshot + timedelta(seconds=cadence.value)

    async def collect_snapshot(
        self,
        token_mint: str,
        snapshot_types: Optional[list[str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> SnapshotResult:
        """
        Collect snapshots for a token.

        Args:
            token_mint: Token mint address
            snapshot_types: Types to collect (default: all enabled)
            timestamp: Snapshot timestamp (default: now)

        Returns:
            SnapshotResult with collection status
        """
        import time

        start_time = time.monotonic()
        timestamp = timestamp or datetime.now(timezone.utc)
        errors = []
        snapshot_ids = {}

        # Determine which types to collect
        if snapshot_types is None:
            types_to_collect = [
                name
                for name, st in self._snapshot_collectors.items()
                if st.enabled
            ]
        else:
            types_to_collect = [
                t for t in snapshot_types if t in self._snapshot_collectors
            ]

        # Sort by priority
        types_to_collect.sort(
            key=lambda t: self._snapshot_collectors[t].priority
        )

        # Collect each snapshot type
        collected_types = []
        for snapshot_type in types_to_collect:
            collector = self._snapshot_collectors[snapshot_type]
            try:
                snapshot_id = await collector.collector(
                    token_mint=token_mint,
                    timestamp=timestamp,
                )
                if snapshot_id:
                    snapshot_ids[snapshot_type] = snapshot_id
                    collected_types.append(snapshot_type)
            except Exception as e:
                errors.append(f"{snapshot_type}: {str(e)}")
                logger.error(
                    "snapshot_collection_error",
                    token_mint=token_mint[:8],
                    snapshot_type=snapshot_type,
                    error=str(e),
                )

        duration_ms = (time.monotonic() - start_time) * 1000

        result = SnapshotResult(
            token_mint=token_mint,
            timestamp=timestamp,
            snapshot_types=collected_types,
            duration_ms=duration_ms,
            success=len(errors) == 0,
            errors=errors,
            snapshot_ids=snapshot_ids,
        )

        logger.info(
            "snapshot_collection_complete",
            token_mint=token_mint[:8],
            types_collected=len(collected_types),
            duration_ms=f"{duration_ms:.1f}",
            success=result.success,
        )

        return result

    def compute_snapshot_checksum(self, data: dict) -> str:
        """Compute deterministic checksum for snapshot data."""
        # Sort keys for determinism
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    # =========================================================================
    # Snapshot Collectors (Stubs - to be implemented with actual data sources)
    # =========================================================================

    async def _collect_holders(
        self,
        token_mint: str,
        timestamp: datetime,
    ) -> Optional[int]:
        """Collect holder snapshot."""
        # TODO: Implement with actual holder data source
        logger.debug("collecting_holders", token_mint=token_mint[:8])
        return None

    async def _collect_balances(
        self,
        token_mint: str,
        timestamp: datetime,
    ) -> Optional[int]:
        """Collect balance snapshot."""
        logger.debug("collecting_balances", token_mint=token_mint[:8])
        return None

    async def _collect_liquidity(
        self,
        token_mint: str,
        timestamp: datetime,
    ) -> Optional[int]:
        """Collect liquidity snapshot."""
        logger.debug("collecting_liquidity", token_mint=token_mint[:8])
        return None

    async def _collect_volume(
        self,
        token_mint: str,
        timestamp: datetime,
    ) -> Optional[int]:
        """Collect volume snapshot."""
        logger.debug("collecting_volume", token_mint=token_mint[:8])
        return None

    async def _collect_graph(
        self,
        token_mint: str,
        timestamp: datetime,
    ) -> Optional[int]:
        """Collect graph snapshot."""
        logger.debug("collecting_graph", token_mint=token_mint[:8])
        return None

    async def _collect_archetypes(
        self,
        token_mint: str,
        timestamp: datetime,
    ) -> Optional[int]:
        """Collect archetype snapshot."""
        logger.debug("collecting_archetypes", token_mint=token_mint[:8])
        return None

    async def _collect_coordination(
        self,
        token_mint: str,
        timestamp: datetime,
    ) -> Optional[int]:
        """Collect coordination snapshot."""
        logger.debug("collecting_coordination", token_mint=token_mint[:8])
        return None

    async def _collect_trajectory(
        self,
        token_mint: str,
        timestamp: datetime,
    ) -> Optional[int]:
        """Collect trajectory snapshot."""
        logger.debug("collecting_trajectory", token_mint=token_mint[:8])
        return None


class SnapshotScheduler:
    """
    Scheduler for automated snapshot collection.

    Manages scheduling and execution of snapshots based on token lifecycle.
    """

    def __init__(
        self,
        engine: SnapshotEngine,
        check_interval_seconds: int = 10,
    ):
        self.engine = engine
        self.check_interval = check_interval_seconds
        self._active_tokens: dict[str, datetime] = {}  # mint -> launch_time
        self._running = False

    def register_token(self, token_mint: str, launch_time: datetime):
        """Register a token for snapshot scheduling."""
        self._active_tokens[token_mint] = launch_time
        logger.info(
            "token_registered_for_snapshots",
            token_mint=token_mint[:8],
            launch_time=launch_time.isoformat(),
        )

    def unregister_token(self, token_mint: str):
        """Unregister a token from snapshot scheduling."""
        if token_mint in self._active_tokens:
            del self._active_tokens[token_mint]

    async def run(self):
        """Run the snapshot scheduler loop."""
        import asyncio

        self._running = True
        logger.info("snapshot_scheduler_started")

        while self._running:
            now = datetime.now(timezone.utc)

            for token_mint, launch_time in list(self._active_tokens.items()):
                cadence = self.engine.determine_cadence(launch_time, now)

                # Check if snapshot is due
                # TODO: Implement proper scheduling with last_snapshot tracking

            await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        logger.info("snapshot_scheduler_stopped")
