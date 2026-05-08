"""
Core type definitions for SHI.

All types are immutable and validated.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, NewType

from pydantic import BaseModel, Field
from pydantic.functional_validators import AfterValidator

# Base58 alphabet for Solana addresses
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def validate_base58(value: str) -> str:
    """Validate base58 encoding."""
    if not all(c in BASE58_ALPHABET for c in value):
        raise ValueError(f"Invalid base58 character in: {value}")
    return value


def validate_solana_address(value: str) -> str:
    """Validate Solana address format (32-44 chars, base58)."""
    if not 32 <= len(value) <= 44:
        raise ValueError(f"Address must be 32-44 characters, got {len(value)}")
    return validate_base58(value)


def validate_signature(value: str) -> str:
    """Validate Solana signature format (87-88 chars, base58)."""
    if not 87 <= len(value) <= 88:
        raise ValueError(f"Signature must be 87-88 characters, got {len(value)}")
    return validate_base58(value)


# Type aliases with validation
WalletAddress = Annotated[str, AfterValidator(validate_solana_address)]
TokenMint = Annotated[str, AfterValidator(validate_solana_address)]
Signature = Annotated[str, AfterValidator(validate_signature)]
Lamports = NewType("Lamports", int)


class TokenBalance(BaseModel):
    """Token balance for a wallet."""

    model_config = {"frozen": True}

    wallet: WalletAddress
    mint: TokenMint
    balance: Annotated[int, Field(ge=0)]
    decimals: Annotated[int, Field(ge=0, le=18)]
    timestamp: datetime

    @property
    def ui_amount(self) -> float:
        """Human-readable balance."""
        return self.balance / (10**self.decimals)


class WalletMetadata(BaseModel):
    """Metadata for a wallet address."""

    model_config = {"frozen": True}

    address: WalletAddress
    funded_by: WalletAddress | None = None
    first_funded_at: datetime | None = None
    first_seen_at: datetime


class FundingEdge(BaseModel):
    """Edge in the funding graph."""

    model_config = {"frozen": True}

    source: WalletAddress
    target: WalletAddress
    amount_lamports: Lamports
    timestamp: datetime
    signature: Signature


class HolderSnapshot(BaseModel):
    """Snapshot of all holders for a token at a point in time."""

    model_config = {"frozen": True}

    mint: TokenMint
    timestamp: datetime
    total_supply: Annotated[int, Field(gt=0)]
    holder_count: Annotated[int, Field(ge=0)]
    balances: list[TokenBalance]

    @property
    def shares(self) -> list[float]:
        """Compute share of supply for each holder."""
        return [b.balance / self.total_supply for b in self.balances]


class MetricOutput(BaseModel):
    """Standard output format for all metrics."""

    model_config = {"frozen": True}

    metric_name: str
    value: float
    z_score: float | None = None
    percentile: float | None = None
    confidence_interval: tuple[float, float] | None = None
    version: str
    computed_at: datetime
    baseline_version: str | None = None


class ArchetypeLabel(BaseModel):
    """Behavioral archetype assignment for a wallet."""

    model_config = {"frozen": True}

    wallet: WalletAddress
    archetype: str  # One of: sniper, long_term_accumulator, coordinated_cluster, liquidity_actor, exchange_linked, dormant_whale
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    features_used: list[str]
