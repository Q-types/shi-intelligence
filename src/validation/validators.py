"""
Data Validators for SHI.

Validates all ingested data:
- Wallet addresses (base58, length)
- Transaction signatures
- Balance values (non-negative)
- Timestamps (not in future)

All validation failures are logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()

# Base58 alphabet for Solana
BASE58_ALPHABET = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")


class ValidationSeverity(Enum):
    """Severity of validation issues."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationError:
    """Validation error details."""

    field: str
    value: Any
    message: str
    severity: ValidationSeverity


@dataclass
class ValidationResult:
    """Result of validation check."""

    is_valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationError]

    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(is_valid=True, errors=[], warnings=[])

    @classmethod
    def failure(cls, error: ValidationError) -> "ValidationResult":
        return cls(is_valid=False, errors=[error], warnings=[])

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge two validation results."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


def validate_base58(value: str, field: str) -> ValidationResult:
    """Validate base58 encoding."""
    if not all(c in BASE58_ALPHABET for c in value):
        invalid_chars = [c for c in value if c not in BASE58_ALPHABET]
        return ValidationResult.failure(
            ValidationError(
                field=field,
                value=value,
                message=f"Invalid base58 characters: {invalid_chars}",
                severity=ValidationSeverity.ERROR,
            )
        )
    return ValidationResult.success()


def validate_wallet_address(address: str) -> ValidationResult:
    """
    Validate Solana wallet address.

    Requirements:
    - Base58 encoded
    - 32-44 characters
    """
    field = "wallet_address"

    # Length check
    if not 32 <= len(address) <= 44:
        return ValidationResult.failure(
            ValidationError(
                field=field,
                value=address,
                message=f"Address must be 32-44 chars, got {len(address)}",
                severity=ValidationSeverity.ERROR,
            )
        )

    # Base58 check
    result = validate_base58(address, field)
    if not result.is_valid:
        logger.warning(
            "invalid_wallet_address",
            address=address[:8] + "...",
            errors=[e.message for e in result.errors],
        )

    return result


def validate_token_mint(mint: str) -> ValidationResult:
    """
    Validate Solana token mint address.

    Same format as wallet address.
    """
    field = "token_mint"

    if not 32 <= len(mint) <= 44:
        return ValidationResult.failure(
            ValidationError(
                field=field,
                value=mint,
                message=f"Mint must be 32-44 chars, got {len(mint)}",
                severity=ValidationSeverity.ERROR,
            )
        )

    result = validate_base58(mint, field)
    if not result.is_valid:
        logger.warning(
            "invalid_token_mint",
            mint=mint[:8] + "...",
            errors=[e.message for e in result.errors],
        )

    return result


def validate_signature(signature: str) -> ValidationResult:
    """
    Validate Solana transaction signature.

    Requirements:
    - Base58 encoded
    - 87-88 characters
    """
    field = "signature"

    if not 87 <= len(signature) <= 88:
        return ValidationResult.failure(
            ValidationError(
                field=field,
                value=signature,
                message=f"Signature must be 87-88 chars, got {len(signature)}",
                severity=ValidationSeverity.ERROR,
            )
        )

    return validate_base58(signature, field)


def validate_balance(balance: int, field: str = "balance") -> ValidationResult:
    """
    Validate token balance.

    Requirements:
    - Non-negative integer
    - Within reasonable bounds
    """
    if not isinstance(balance, int):
        return ValidationResult.failure(
            ValidationError(
                field=field,
                value=balance,
                message=f"Balance must be integer, got {type(balance).__name__}",
                severity=ValidationSeverity.ERROR,
            )
        )

    if balance < 0:
        logger.error("negative_balance_detected", balance=balance)
        return ValidationResult.failure(
            ValidationError(
                field=field,
                value=balance,
                message="Balance cannot be negative",
                severity=ValidationSeverity.CRITICAL,
            )
        )

    # Sanity check - warn on extremely large values
    MAX_REASONABLE = 10**18  # 1 billion tokens with 9 decimals
    if balance > MAX_REASONABLE:
        result = ValidationResult.success()
        result.warnings.append(
            ValidationError(
                field=field,
                value=balance,
                message=f"Unusually large balance: {balance}",
                severity=ValidationSeverity.WARNING,
            )
        )
        return result

    return ValidationResult.success()


def validate_timestamp(
    timestamp: datetime,
    field: str = "timestamp",
    allow_future: bool = False,
) -> ValidationResult:
    """
    Validate timestamp.

    Requirements:
    - Not in the future (unless allowed)
    - Not too far in the past (Solana launched 2020)
    """
    now = datetime.now(timezone.utc)
    solana_launch = datetime(2020, 3, 16, tzinfo=timezone.utc)

    # Ensure timezone aware
    if timestamp.tzinfo is None:
        result = ValidationResult.success()
        result.warnings.append(
            ValidationError(
                field=field,
                value=timestamp,
                message="Timestamp is not timezone-aware, assuming UTC",
                severity=ValidationSeverity.WARNING,
            )
        )
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    # Future check
    if not allow_future and timestamp > now:
        return ValidationResult.failure(
            ValidationError(
                field=field,
                value=timestamp.isoformat(),
                message="Timestamp is in the future",
                severity=ValidationSeverity.ERROR,
            )
        )

    # Too old check
    if timestamp < solana_launch:
        return ValidationResult.failure(
            ValidationError(
                field=field,
                value=timestamp.isoformat(),
                message="Timestamp predates Solana launch",
                severity=ValidationSeverity.ERROR,
            )
        )

    return ValidationResult.success()


def validate_holder_snapshot(
    mint: str,
    balances: list[tuple[str, int]],
    total_supply: int,
) -> ValidationResult:
    """
    Validate entire holder snapshot.

    Checks:
    - Mint is valid
    - All wallet addresses are valid
    - All balances are valid
    - Balances sum to <= total supply
    - No duplicate wallets
    """
    result = ValidationResult.success()

    # Validate mint
    result = result.merge(validate_token_mint(mint))

    # Validate each holder
    seen_wallets = set()
    balance_sum = 0

    for wallet, balance in balances:
        # Wallet address
        result = result.merge(validate_wallet_address(wallet))

        # Duplicate check
        if wallet in seen_wallets:
            result.warnings.append(
                ValidationError(
                    field="wallet",
                    value=wallet,
                    message="Duplicate wallet in snapshot",
                    severity=ValidationSeverity.WARNING,
                )
            )
        seen_wallets.add(wallet)

        # Balance
        result = result.merge(validate_balance(balance))
        balance_sum += balance

    # Sum check
    if balance_sum > total_supply:
        result.errors.append(
            ValidationError(
                field="balance_sum",
                value=balance_sum,
                message=f"Balances sum ({balance_sum}) exceeds total supply ({total_supply})",
                severity=ValidationSeverity.ERROR,
            )
        )
        result.is_valid = False

    if not result.is_valid:
        logger.error(
            "snapshot_validation_failed",
            mint=mint,
            error_count=len(result.errors),
            warning_count=len(result.warnings),
        )

    return result
