"""
Wallet Profile Evolution Tracking for SHI.

Tracks how wallet profiles (archetype, risk score) change over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.types import WalletAddress

logger = structlog.get_logger()


@dataclass
class ProfileSnapshot:
    """Snapshot of a wallet's profile at a point in time."""

    wallet: WalletAddress
    timestamp: datetime
    archetype: str
    risk_score: float
    anomaly_score: Optional[float] = None
    features: Optional[Dict] = None


@dataclass
class ProfileEvolution:
    """Evolution of a wallet's profile over time."""

    wallet: WalletAddress
    snapshots: List[ProfileSnapshot]
    archetype_transitions: List[tuple[datetime, str, str]]  # (timestamp, from, to)
    current_archetype: str
    current_risk_score: float
    profile_velocity: float  # How fast the profile is changing

    def get_archetype_duration(self, archetype: str) -> float:
        """
        Get how long the wallet has been in a specific archetype.

        Args:
            archetype: Archetype to check

        Returns:
            Duration in days
        """
        if self.current_archetype != archetype:
            return 0.0

        # Find last transition to this archetype
        for i in range(len(self.archetype_transitions) - 1, -1, -1):
            _, from_arch, to_arch = self.archetype_transitions[i]
            if to_arch == archetype:
                transition_time = self.archetype_transitions[i][0]
                current_time = self.snapshots[-1].timestamp
                return (current_time - transition_time).total_seconds() / 86400

        # No transition found, been in this archetype since first snapshot
        if self.snapshots:
            first_time = self.snapshots[0].timestamp
            current_time = self.snapshots[-1].timestamp
            return (current_time - first_time).total_seconds() / 86400

        return 0.0

    def get_risk_trend(self) -> str:
        """
        Get risk score trend (increasing, decreasing, stable).

        Returns:
            Trend direction as string
        """
        if len(self.snapshots) < 2:
            return "stable"

        recent_scores = [s.risk_score for s in self.snapshots[-5:]]

        if len(recent_scores) >= 2:
            # Simple linear trend
            slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"

        return "stable"


class ProfileTracker:
    """
    Tracks wallet profile evolution over time.

    Stores snapshots of wallet profiles and computes metrics
    like archetype transitions and profile velocity.
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize profile tracker.

        Args:
            db_session: Database session for querying/storing
        """
        self.db_session = db_session

    async def add_snapshot(
        self,
        wallet: WalletAddress,
        archetype: str,
        risk_score: float,
        anomaly_score: Optional[float] = None,
        features: Optional[Dict] = None,
    ) -> ProfileSnapshot:
        """
        Add a new profile snapshot for a wallet.

        Args:
            wallet: Wallet address
            archetype: Current archetype classification
            risk_score: Current risk score
            anomaly_score: Optional anomaly score
            features: Optional feature dict

        Returns:
            ProfileSnapshot object
        """
        snapshot = ProfileSnapshot(
            wallet=wallet,
            timestamp=datetime.now(timezone.utc),
            archetype=archetype,
            risk_score=risk_score,
            anomaly_score=anomaly_score,
            features=features,
        )

        # In production, this would update the wallet_profiles table
        # and append to profile_history JSONB column

        logger.info(
            "profile_snapshot_added",
            wallet=wallet,
            archetype=archetype,
            risk_score=risk_score,
        )

        return snapshot

    async def get_profile_evolution(
        self,
        wallet: WalletAddress,
        lookback_days: int = 30,
    ) -> Optional[ProfileEvolution]:
        """
        Get profile evolution for a wallet.

        Args:
            wallet: Wallet address
            lookback_days: How far back to look

        Returns:
            ProfileEvolution object or None if no history
        """
        # In production, this would query wallet_profiles table
        # and parse profile_history JSONB

        # For now, return None (no history)
        logger.debug("profile_evolution_requested", wallet=wallet, lookback_days=lookback_days)
        return None

    async def detect_archetype_transition(
        self,
        wallet: WalletAddress,
        new_archetype: str,
    ) -> Optional[tuple[str, str]]:
        """
        Detect if an archetype transition occurred.

        Args:
            wallet: Wallet address
            new_archetype: New archetype classification

        Returns:
            (from_archetype, to_archetype) tuple or None if no change
        """
        # In production, this would query the current archetype from database
        # For now, assume no previous archetype exists
        return None

    def compute_profile_velocity(
        self,
        snapshots: List[ProfileSnapshot],
        window_days: int = 7,
    ) -> float:
        """
        Compute profile velocity (how fast profile is changing).

        Uses risk score variance over a time window.

        Args:
            snapshots: List of profile snapshots
            window_days: Time window for velocity calculation

        Returns:
            Velocity score (higher = faster change)
        """
        if len(snapshots) < 2:
            return 0.0

        # Filter to recent window
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        recent = [s for s in snapshots if s.timestamp >= cutoff]

        if len(recent) < 2:
            return 0.0

        # Compute risk score variance
        risk_scores = [s.risk_score for s in recent]
        velocity = float(np.std(risk_scores))

        return velocity

    async def get_wallets_with_high_velocity(
        self,
        threshold: float = 0.1,
        limit: int = 50,
    ) -> List[tuple[WalletAddress, float]]:
        """
        Find wallets with high profile velocity (rapidly changing).

        Args:
            threshold: Minimum velocity threshold
            limit: Maximum number of wallets

        Returns:
            List of (wallet, velocity) tuples
        """
        # In production, this would:
        # 1. Query wallet_profiles
        # 2. Parse profile_history JSONB
        # 3. Compute velocity for each
        # 4. Filter and sort

        logger.debug(
            "high_velocity_wallets_requested",
            threshold=threshold,
            limit=limit,
        )

        return []

    async def update_profile(
        self,
        wallet: WalletAddress,
        archetype: Optional[str] = None,
        risk_score: Optional[float] = None,
        anomaly_score: Optional[float] = None,
    ) -> bool:
        """
        Update a wallet's current profile.

        Args:
            wallet: Wallet address
            archetype: New archetype (optional)
            risk_score: New risk score (optional)
            anomaly_score: New anomaly score (optional)

        Returns:
            True if updated successfully
        """
        # In production, this would:
        # 1. Query current profile from wallet_profiles
        # 2. Detect archetype transition if archetype changed
        # 3. Update wallet_profiles table
        # 4. Append snapshot to profile_history JSONB

        updates = {}
        if archetype:
            updates["archetype"] = archetype
        if risk_score is not None:
            updates["risk_score"] = risk_score
        if anomaly_score is not None:
            updates["anomaly_score"] = anomaly_score

        if updates:
            logger.info(
                "profile_updated",
                wallet=wallet,
                updates=updates,
            )
            return True

        return False

    async def get_archetype_distribution(
        self,
    ) -> Dict[str, int]:
        """
        Get distribution of archetypes across all wallets.

        Returns:
            Dict mapping archetype -> count
        """
        # In production, this would query wallet_profiles and aggregate
        logger.debug("archetype_distribution_requested")
        return {}

    async def get_risk_score_stats(
        self,
    ) -> Dict[str, float]:
        """
        Get risk score statistics across all wallets.

        Returns:
            Dict with mean, median, std, etc.
        """
        # In production, this would query wallet_profiles and compute stats
        logger.debug("risk_score_stats_requested")
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 1.0,
        }
