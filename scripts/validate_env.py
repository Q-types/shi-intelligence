#!/usr/bin/env python3
"""
Environment Validation Script.

Checks that all required environment variables are set
and validates their format before starting the application.
"""

import os
import sys
import re
from pathlib import Path


def load_env_file(path: Path) -> dict[str, str]:
    """Load .env file into dictionary."""
    env_vars = {}
    if not path.exists():
        return env_vars

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()

    return env_vars


def validate_telegram_token(token: str) -> tuple[bool, str]:
    """Validate Telegram bot token format."""
    if not token:
        return False, "TELEGRAM_BOT_TOKEN is not set"

    # Format: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz
    pattern = r"^\d+:[A-Za-z0-9_-]+$"
    if not re.match(pattern, token):
        return False, "TELEGRAM_BOT_TOKEN format invalid (expected: 123456:ABC...)"

    return True, "OK"


def validate_database_url(url: str) -> tuple[bool, str]:
    """Validate database URL format."""
    if not url:
        return False, "DATABASE_URL is not set"

    if not url.startswith(("postgresql://", "postgres://")):
        return False, "DATABASE_URL must start with postgresql:// or postgres://"

    return True, "OK"


def validate_solana_rpc(url: str) -> tuple[bool, str]:
    """Validate Solana RPC URL."""
    if not url:
        return False, "SOLANA_RPC_URL is not set"

    if not url.startswith(("http://", "https://")):
        return False, "SOLANA_RPC_URL must be a valid HTTP(S) URL"

    return True, "OK"


def validate_redis_url(url: str) -> tuple[bool, str]:
    """Validate Redis URL format."""
    if not url:
        return True, "Not set (will use in-memory fallback)"

    if not url.startswith(("redis://", "rediss://")):
        return False, "REDIS_URL must start with redis:// or rediss://"

    return True, "OK"


def validate_user_ids(ids: str, name: str) -> tuple[bool, str]:
    """Validate comma-separated user IDs."""
    if not ids:
        return True, "Not set (no special users)"

    for user_id in ids.split(","):
        user_id = user_id.strip()
        if user_id and not user_id.isdigit():
            return False, f"{name} must be comma-separated integers"

    return True, "OK"


def main() -> int:
    """Run all validations."""
    print("=" * 50)
    print("SHI Environment Validation")
    print("=" * 50)
    print()

    # Load .env file
    env_path = Path(__file__).parent.parent / ".env"
    file_vars = load_env_file(env_path)

    # Merge with actual environment (env takes precedence)
    all_vars = {**file_vars, **os.environ}

    validations = [
        ("TELEGRAM_BOT_TOKEN", validate_telegram_token),
        ("DATABASE_URL", validate_database_url),
        ("SOLANA_RPC_URL", validate_solana_rpc),
        ("REDIS_URL", validate_redis_url),
    ]

    errors = []
    warnings = []

    for var_name, validator in validations:
        value = all_vars.get(var_name, "")
        is_valid, message = validator(value)

        if is_valid:
            if "Not set" in message:
                print(f"  [WARN] {var_name}: {message}")
                warnings.append(var_name)
            else:
                print(f"  [OK]   {var_name}: {message}")
        else:
            print(f"  [FAIL] {var_name}: {message}")
            errors.append((var_name, message))

    # Validate user ID lists
    for var_name in ["ADMIN_USER_IDS", "PREMIUM_USER_IDS"]:
        value = all_vars.get(var_name, "")
        is_valid, message = validate_user_ids(value, var_name)
        if is_valid:
            if "Not set" in message:
                print(f"  [WARN] {var_name}: {message}")
            else:
                print(f"  [OK]   {var_name}: {message}")
        else:
            print(f"  [FAIL] {var_name}: {message}")
            errors.append((var_name, message))

    print()
    print("=" * 50)

    if errors:
        print(f"FAILED: {len(errors)} error(s) found")
        print()
        print("Required actions:")
        for var_name, message in errors:
            print(f"  - Set {var_name}: {message}")
        return 1
    elif warnings:
        print(f"PASSED with {len(warnings)} warning(s)")
        print("The application will start but some features may be limited.")
        return 0
    else:
        print("PASSED: All validations successful")
        return 0


if __name__ == "__main__":
    sys.exit(main())
