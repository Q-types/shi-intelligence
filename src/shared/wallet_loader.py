"""Wallet Loader - Load tracked wallet addresses from various file formats.

Supports:
- Plain text (.txt) with one wallet address per line
- CSV (.csv) with wallet/address columns
- JSON (.json) with array or object structures
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import structlog

logger = structlog.get_logger()

# Solana address validation regex (base58, 32-44 characters)
SOLANA_ADDRESS_PATTERN = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


@dataclass
class TrackedWallet:
    """A tracked whale wallet with optional metadata."""

    address: str
    label: str | None = None
    notes: str | None = None
    source_file: str | None = None

    def __post_init__(self):
        """Validate address format."""
        if not is_valid_solana_address(self.address):
            raise ValueError(f"Invalid Solana address: {self.address}")

    @property
    def short_address(self) -> str:
        """Return truncated address for display."""
        if len(self.address) <= 12:
            return self.address
        return f"{self.address[:4]}...{self.address[-4:]}"

    def display_name(self) -> str:
        """Return label if available, otherwise short address."""
        return self.label or self.short_address


def is_valid_solana_address(address: str) -> bool:
    """Validate Solana address format (base58, 32-44 chars)."""
    if not address or not isinstance(address, str):
        return False
    return bool(SOLANA_ADDRESS_PATTERN.match(address.strip()))


def load_wallets_from_txt(filepath: Path) -> Iterator[TrackedWallet]:
    """Load wallets from plain text file (one address per line)."""
    logger.debug("loading_txt", path=str(filepath))

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Handle "label: address" format
            if ":" in line and not line.startswith("0x"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    label, address = parts[0].strip(), parts[1].strip()
                    if is_valid_solana_address(address):
                        yield TrackedWallet(
                            address=address,
                            label=label,
                            source_file=str(filepath),
                        )
                        continue

            # Plain address
            if is_valid_solana_address(line):
                yield TrackedWallet(address=line, source_file=str(filepath))
            else:
                logger.warning(
                    "invalid_address_skipped",
                    path=str(filepath),
                    line=line_num,
                    value=line[:20],
                )


def load_wallets_from_csv(filepath: Path) -> Iterator[TrackedWallet]:
    """Load wallets from CSV file.

    Looks for columns: wallet, address, wallet_address, or first column.
    Optional columns: label, name, notes.
    """
    logger.debug("loading_csv", path=str(filepath))

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Normalize column names to lowercase
        if reader.fieldnames:
            fieldnames_lower = {name.lower(): name for name in reader.fieldnames}
        else:
            fieldnames_lower = {}

        # Find address column
        address_col = None
        for candidate in ["wallet", "address", "wallet_address", "walletaddress"]:
            if candidate in fieldnames_lower:
                address_col = fieldnames_lower[candidate]
                break

        # Fall back to first column
        if address_col is None and reader.fieldnames:
            address_col = reader.fieldnames[0]

        # Find optional columns
        label_col = fieldnames_lower.get("label") or fieldnames_lower.get("name")
        notes_col = fieldnames_lower.get("notes") or fieldnames_lower.get("note")

        for row_num, row in enumerate(reader, 2):
            address = row.get(address_col, "").strip() if address_col else ""

            if not address:
                continue

            if is_valid_solana_address(address):
                yield TrackedWallet(
                    address=address,
                    label=row.get(label_col, "").strip() if label_col else None,
                    notes=row.get(notes_col, "").strip() if notes_col else None,
                    source_file=str(filepath),
                )
            else:
                logger.warning(
                    "invalid_address_skipped",
                    path=str(filepath),
                    row=row_num,
                    value=address[:20],
                )


def load_wallets_from_json(filepath: Path) -> Iterator[TrackedWallet]:
    """Load wallets from JSON file.

    Supports:
    - Array of strings: ["address1", "address2"]
    - Array of objects: [{"address": "...", "label": "..."}]
    - Object with wallets key: {"wallets": [...]}
    """
    logger.debug("loading_json", path=str(filepath))

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle object with wallets key
    if isinstance(data, dict):
        for key in ["wallets", "addresses", "tracked"]:
            if key in data:
                data = data[key]
                break
        else:
            # If no known key, try to extract addresses from values
            if all(is_valid_solana_address(str(v)) for v in data.values() if isinstance(v, str)):
                for label, address in data.items():
                    if isinstance(address, str) and is_valid_solana_address(address):
                        yield TrackedWallet(
                            address=address,
                            label=label,
                            source_file=str(filepath),
                        )
                return

    # Handle array
    if isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, str):
                # Plain address string
                if is_valid_solana_address(item):
                    yield TrackedWallet(address=item, source_file=str(filepath))
                else:
                    logger.warning(
                        "invalid_address_skipped",
                        path=str(filepath),
                        index=idx,
                        value=item[:20],
                    )

            elif isinstance(item, dict):
                # Object with address field
                address = None
                for key in ["address", "wallet", "wallet_address", "walletAddress"]:
                    if key in item:
                        address = str(item[key]).strip()
                        break

                if address and is_valid_solana_address(address):
                    yield TrackedWallet(
                        address=address,
                        label=item.get("label") or item.get("name"),
                        notes=item.get("notes") or item.get("note"),
                        source_file=str(filepath),
                    )
                elif address:
                    logger.warning(
                        "invalid_address_skipped",
                        path=str(filepath),
                        index=idx,
                        value=address[:20],
                    )


def load_wallets_from_file(filepath: Path) -> list[TrackedWallet]:
    """Load wallets from a single file, auto-detecting format."""
    suffix = filepath.suffix.lower()

    if suffix == ".txt":
        return list(load_wallets_from_txt(filepath))
    elif suffix == ".csv":
        return list(load_wallets_from_csv(filepath))
    elif suffix == ".json":
        return list(load_wallets_from_json(filepath))
    else:
        # Try to detect format from content
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                first_char = f.read(1)
                if first_char in "[{":
                    return list(load_wallets_from_json(filepath))
                else:
                    return list(load_wallets_from_txt(filepath))
        except Exception as e:
            logger.error("file_load_failed", path=str(filepath), error=str(e))
            return []


def load_all_wallets(wallets_dir: Path) -> list[TrackedWallet]:
    """Load all wallets from all files in directory.

    Deduplicates by address, keeping the first occurrence with most metadata.
    """
    if not wallets_dir.exists():
        logger.warning("wallets_dir_not_found", path=str(wallets_dir))
        return []

    all_wallets: dict[str, TrackedWallet] = {}

    # Process all files in directory
    for filepath in sorted(wallets_dir.iterdir()):
        if filepath.is_file() and not filepath.name.startswith("."):
            try:
                wallets = load_wallets_from_file(filepath)
                logger.info(
                    "wallets_loaded",
                    path=str(filepath),
                    count=len(wallets),
                )

                for wallet in wallets:
                    if wallet.address not in all_wallets:
                        all_wallets[wallet.address] = wallet
                    else:
                        # Keep version with more metadata
                        existing = all_wallets[wallet.address]
                        if wallet.label and not existing.label:
                            all_wallets[wallet.address] = wallet

            except Exception as e:
                logger.error("file_load_failed", path=str(filepath), error=str(e))

    result = list(all_wallets.values())
    logger.info("total_wallets_loaded", count=len(result), unique=len(all_wallets))
    return result
