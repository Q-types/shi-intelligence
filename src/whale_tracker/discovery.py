"""Whale discovery - auto-identify whales based on supply threshold."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger()

# Solana address validation regex
SOLANA_ADDRESS_REGEX = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


def is_valid_solana_address(address: str) -> bool:
    """Validate a Solana address format."""
    if not address or not isinstance(address, str):
        return False
    return bool(SOLANA_ADDRESS_REGEX.match(address))


@dataclass
class DiscoveryConfig:
    """Configuration for whale auto-discovery."""

    threshold_pct: float = 0.5          # % of supply to qualify as whale
    min_balance: float | None = None    # Optional absolute minimum
    exclude_known_contracts: bool = True  # Filter out DEX/protocol addresses
    include_dormant: bool = True        # Include wallets with no recent activity

    # Known contract addresses to exclude (DEXs, protocols, etc.)
    known_contracts: set[str] = field(default_factory=lambda: {
        # Raydium
        "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1",
        # Orca
        "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",
        # Jupiter
        "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
        # Meteora
        "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo",
    })


@dataclass
class DiscoveredWallet:
    """A wallet discovered via auto-discovery."""

    address: str
    balance: float
    pct_of_supply: float
    discovered_at: datetime
    source: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "address": self.address,
            "balance": self.balance,
            "pct_of_supply": self.pct_of_supply,
            "discovered_at": self.discovered_at.isoformat(),
            "source": self.source,
        }


class WhaleDiscovery:
    """Auto-discover whales based on supply threshold."""

    def __init__(self, config: DiscoveryConfig | None = None):
        """
        Initialize whale discovery.

        Args:
            config: Discovery configuration (uses defaults if not provided)
        """
        self.config = config or DiscoveryConfig()

    def discover_from_holders(
        self,
        holder_data: list[dict[str, Any]],
        total_supply: float | None = None,
    ) -> list[DiscoveredWallet]:
        """
        Find all wallets holding >= threshold % of supply.

        Args:
            holder_data: List of holder dicts with 'address' and 'balance' keys
            total_supply: Optional total supply (calculated from holders if not provided)

        Returns:
            List of DiscoveredWallet objects
        """
        if not holder_data:
            return []

        now = datetime.now(timezone.utc)

        # Calculate total supply if not provided
        if total_supply is None:
            total_supply = sum(h.get("balance", 0) for h in holder_data)

        if total_supply <= 0:
            logger.warning("discover_failed", reason="total_supply_zero")
            return []

        # Calculate threshold balance
        threshold_balance = total_supply * (self.config.threshold_pct / 100)

        logger.info(
            "discovery_starting",
            threshold_pct=self.config.threshold_pct,
            threshold_balance=threshold_balance,
            total_holders=len(holder_data),
        )

        whales = []
        for holder in holder_data:
            address = holder.get("address", holder.get("wallet_address", ""))
            balance = holder.get("balance", holder.get("ui_amount", 0))

            # Validate address
            if not is_valid_solana_address(address):
                continue

            # Check threshold
            if balance < threshold_balance:
                continue

            # Check minimum balance
            if self.config.min_balance and balance < self.config.min_balance:
                continue

            # Exclude known contracts
            if self.config.exclude_known_contracts:
                if address in self.config.known_contracts:
                    logger.debug("excluding_contract", address=address[:8])
                    continue

            pct_of_supply = (balance / total_supply) * 100

            whales.append(DiscoveredWallet(
                address=address,
                balance=balance,
                pct_of_supply=pct_of_supply,
                discovered_at=now,
                source="auto",
            ))

        # Sort by balance descending
        whales.sort(key=lambda w: w.balance, reverse=True)

        logger.info(
            "discovery_complete",
            whales_found=len(whales),
            threshold_pct=self.config.threshold_pct,
        )

        return whales

    def discover_from_balances(
        self,
        balances: list[Any],  # List of WalletBalance or similar
    ) -> list[DiscoveredWallet]:
        """
        Find whales from a list of balance objects.

        Convenience method that converts balance objects to holder dicts.

        Args:
            balances: List of balance objects with address and ui_amount/balance

        Returns:
            List of DiscoveredWallet objects
        """
        holder_data = []
        for bal in balances:
            # Handle both dict and object
            if hasattr(bal, "address"):
                address = bal.address
                balance = getattr(bal, "ui_amount", getattr(bal, "balance", 0))
            elif isinstance(bal, dict):
                address = bal.get("address", bal.get("wallet_address", ""))
                balance = bal.get("ui_amount", bal.get("balance", 0))
            else:
                continue

            # Handle nested balance object
            if hasattr(balance, "ui_amount"):
                balance = balance.ui_amount

            holder_data.append({"address": address, "balance": balance})

        return self.discover_from_holders(holder_data)


def parse_wallet_input(input_text: str) -> list[str]:
    """
    Parse user-provided wallet addresses.

    Supports:
    - One address per line
    - Comma-separated addresses
    - Mixed whitespace

    Args:
        input_text: Raw user input

    Returns:
        List of valid Solana addresses
    """
    if not input_text:
        return []

    # Split by newlines and commas
    parts = re.split(r"[,\n]+", input_text)

    # Clean and validate each address
    addresses = []
    for part in parts:
        address = part.strip()
        if is_valid_solana_address(address):
            if address not in addresses:  # Deduplicate
                addresses.append(address)
        elif address:  # Non-empty but invalid
            logger.warning("invalid_address_skipped", address=address[:20])

    logger.info("wallet_input_parsed", valid=len(addresses))
    return addresses
