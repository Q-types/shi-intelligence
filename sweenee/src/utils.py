"""Utility functions for SWEENEE Dashboard."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


# Solana address validation
SOLANA_ADDRESS_PATTERN = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


def is_valid_solana_address(address: str) -> bool:
    """Check if string is a valid Solana address (base58, 32-44 chars)."""
    if not address or not isinstance(address, str):
        return False
    return bool(SOLANA_ADDRESS_PATTERN.match(address.strip()))


def short_address(address: str, chars: int = 4) -> str:
    """Truncate address for display: ABCD...WXYZ."""
    if not address:
        return ""
    if len(address) <= chars * 2 + 3:
        return address
    return f"{address[:chars]}...{address[-chars:]}"


def format_number(n: float, decimals: int = 0) -> str:
    """Format number with commas."""
    if decimals == 0:
        return f"{n:,.0f}"
    return f"{n:,.{decimals}f}"


def format_large_number(n: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(n) >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    elif abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"


def format_percentage(n: float, decimals: int = 2) -> str:
    """Format as percentage."""
    return f"{n * 100:.{decimals}f}%"


def format_timestamp(dt: datetime | None) -> str:
    """Format datetime for display."""
    if not dt:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_relative_time(dt: datetime | None) -> str:
    """Format datetime as relative time (e.g., '2 hours ago')."""
    if not dt:
        return "—"

    now = datetime.now(timezone.utc)
    diff = now - dt

    seconds = diff.total_seconds()
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        mins = int(seconds / 60)
        return f"{mins} min{'s' if mins != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"


def solscan_tx_url(signature: str) -> str:
    """Generate Solscan transaction URL."""
    return f"https://solscan.io/tx/{signature}"


def solscan_wallet_url(address: str) -> str:
    """Generate Solscan wallet URL."""
    return f"https://solscan.io/account/{address}"


def solscan_token_url(mint: str) -> str:
    """Generate Solscan token URL."""
    return f"https://solscan.io/token/{mint}"
