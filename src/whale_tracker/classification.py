"""Whale classification system - tier-based whale identification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class WhaleTier(Enum):
    """Whale classification tiers based on percentile rank."""

    ULTRA_WHALE = "ultra_whale"      # Top 1%
    MEGA_WHALE = "mega_whale"        # Top 5%
    WHALE = "whale"                  # Top 10%
    LARGE_HOLDER = "large_holder"    # Top 25%
    STANDARD = "standard"            # Below 75th percentile

    @classmethod
    def from_percentile(cls, percentile: float) -> "WhaleTier":
        """
        Get tier from percentile rank.

        Args:
            percentile: 0-100, where 100 = largest holder

        Returns:
            WhaleTier based on percentile thresholds
        """
        if percentile >= 99:
            return cls.ULTRA_WHALE
        elif percentile >= 95:
            return cls.MEGA_WHALE
        elif percentile >= 90:
            return cls.WHALE
        elif percentile >= 75:
            return cls.LARGE_HOLDER
        return cls.STANDARD

    @property
    def emoji(self) -> str:
        """Get emoji for tier."""
        return {
            WhaleTier.ULTRA_WHALE: "🐋",
            WhaleTier.MEGA_WHALE: "🐳",
            WhaleTier.WHALE: "🐟",
            WhaleTier.LARGE_HOLDER: "🐠",
            WhaleTier.STANDARD: "🐡",
        }.get(self, "❓")

    @property
    def color(self) -> str:
        """Get color for tier (hex)."""
        return {
            WhaleTier.ULTRA_WHALE: "#FFD700",  # Gold
            WhaleTier.MEGA_WHALE: "#C0C0C0",   # Silver
            WhaleTier.WHALE: "#CD7F32",        # Bronze
            WhaleTier.LARGE_HOLDER: "#4CAF50", # Green
            WhaleTier.STANDARD: "#9E9E9E",     # Gray
        }.get(self, "#757575")

    @property
    def display_name(self) -> str:
        """Get human-readable name."""
        return {
            WhaleTier.ULTRA_WHALE: "Ultra Whale",
            WhaleTier.MEGA_WHALE: "Mega Whale",
            WhaleTier.WHALE: "Whale",
            WhaleTier.LARGE_HOLDER: "Large Holder",
            WhaleTier.STANDARD: "Standard",
        }.get(self, "Unknown")


@dataclass
class WhaleProfile:
    """Complete profile of a whale wallet."""

    wallet_address: str
    label: str | None
    balance: float
    tier: WhaleTier
    percentile_rank: float          # 0-100
    concentration_share: float      # % of total supply (0-100)
    holder_rank: int                # Rank among all holders (1 = largest)
    discovery_mode: str             # 'manual' | 'auto'
    classified_at: datetime

    @property
    def tier_emoji(self) -> str:
        """Get emoji for this whale's tier."""
        return self.tier.emoji

    @property
    def display_name(self) -> str:
        """Get display name (label or short address)."""
        if self.label:
            return self.label
        return f"{self.wallet_address[:4]}...{self.wallet_address[-4:]}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "wallet_address": self.wallet_address,
            "label": self.label,
            "balance": self.balance,
            "tier": self.tier.value,
            "percentile_rank": self.percentile_rank,
            "concentration_share": self.concentration_share,
            "holder_rank": self.holder_rank,
            "discovery_mode": self.discovery_mode,
            "classified_at": self.classified_at.isoformat(),
        }


@dataclass
class TierTransition:
    """Record of a wallet changing tiers."""

    wallet_address: str
    previous_tier: WhaleTier | None
    new_tier: WhaleTier
    balance_before: float
    balance_after: float
    transition_at: datetime

    @property
    def is_promotion(self) -> bool:
        """True if wallet moved to a higher tier."""
        tier_order = [
            WhaleTier.STANDARD,
            WhaleTier.LARGE_HOLDER,
            WhaleTier.WHALE,
            WhaleTier.MEGA_WHALE,
            WhaleTier.ULTRA_WHALE,
        ]
        if self.previous_tier is None:
            return True
        return tier_order.index(self.new_tier) > tier_order.index(self.previous_tier)

    @property
    def is_demotion(self) -> bool:
        """True if wallet moved to a lower tier."""
        if self.previous_tier is None:
            return False
        return not self.is_promotion and self.new_tier != self.previous_tier

    @property
    def description(self) -> str:
        """Get human-readable description of transition."""
        if self.previous_tier is None:
            return f"New {self.new_tier.display_name} detected"
        if self.is_promotion:
            return f"Promoted: {self.previous_tier.display_name} → {self.new_tier.display_name}"
        if self.is_demotion:
            return f"Demoted: {self.previous_tier.display_name} → {self.new_tier.display_name}"
        return f"Tier unchanged: {self.new_tier.display_name}"


def classify_whales(
    balances: list[dict[str, Any]],
    labels: dict[str, str] | None = None,
    discovery_mode: str = "manual",
) -> list[WhaleProfile]:
    """
    Classify wallets into whale tiers based on their balance.

    Args:
        balances: List of dicts with 'address' and 'balance' keys
        labels: Optional mapping of addresses to labels
        discovery_mode: 'manual' or 'auto'

    Returns:
        List of WhaleProfile objects sorted by balance (descending)
    """
    if not balances:
        return []

    labels = labels or {}
    now = datetime.now(timezone.utc)

    # Sort by balance descending
    sorted_balances = sorted(balances, key=lambda x: x.get("balance", 0), reverse=True)

    # Calculate total for concentration share
    total_balance = sum(b.get("balance", 0) for b in sorted_balances)
    if total_balance == 0:
        return []

    # Calculate percentile ranks
    n = len(sorted_balances)
    profiles = []

    for rank, wallet in enumerate(sorted_balances, start=1):
        address = wallet.get("address", wallet.get("wallet_address", ""))
        balance = wallet.get("balance", wallet.get("ui_amount", 0))

        # Percentile: 100 = largest, 0 = smallest
        # rank 1 out of 100 = 99th percentile
        percentile = 100 * (n - rank) / n if n > 1 else 100

        tier = WhaleTier.from_percentile(percentile)
        concentration = (balance / total_balance) * 100

        profile = WhaleProfile(
            wallet_address=address,
            label=labels.get(address),
            balance=balance,
            tier=tier,
            percentile_rank=percentile,
            concentration_share=concentration,
            holder_rank=rank,
            discovery_mode=discovery_mode,
            classified_at=now,
        )
        profiles.append(profile)

    logger.info(
        "whales_classified",
        total=len(profiles),
        ultra=len([p for p in profiles if p.tier == WhaleTier.ULTRA_WHALE]),
        mega=len([p for p in profiles if p.tier == WhaleTier.MEGA_WHALE]),
        whale=len([p for p in profiles if p.tier == WhaleTier.WHALE]),
    )

    return profiles


def detect_tier_transitions(
    previous_profiles: list[WhaleProfile],
    current_profiles: list[WhaleProfile],
) -> list[TierTransition]:
    """
    Detect wallets that changed tiers between two classification runs.

    Args:
        previous_profiles: Previous classification results
        current_profiles: Current classification results

    Returns:
        List of TierTransition objects for changed wallets
    """
    previous_map = {p.wallet_address: p for p in previous_profiles}
    current_map = {p.wallet_address: p for p in current_profiles}
    now = datetime.now(timezone.utc)

    transitions = []

    # Check for tier changes in current profiles
    for address, current in current_map.items():
        previous = previous_map.get(address)

        if previous is None:
            # New wallet
            transitions.append(TierTransition(
                wallet_address=address,
                previous_tier=None,
                new_tier=current.tier,
                balance_before=0,
                balance_after=current.balance,
                transition_at=now,
            ))
        elif previous.tier != current.tier:
            # Tier changed
            transitions.append(TierTransition(
                wallet_address=address,
                previous_tier=previous.tier,
                new_tier=current.tier,
                balance_before=previous.balance,
                balance_after=current.balance,
                transition_at=now,
            ))

    # Check for wallets that disappeared (exited)
    for address, previous in previous_map.items():
        if address not in current_map:
            transitions.append(TierTransition(
                wallet_address=address,
                previous_tier=previous.tier,
                new_tier=WhaleTier.STANDARD,  # Exited
                balance_before=previous.balance,
                balance_after=0,
                transition_at=now,
            ))

    if transitions:
        logger.info(
            "tier_transitions_detected",
            count=len(transitions),
            promotions=len([t for t in transitions if t.is_promotion]),
            demotions=len([t for t in transitions if t.is_demotion]),
        )

    return transitions
