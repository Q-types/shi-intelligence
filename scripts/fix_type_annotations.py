#!/usr/bin/env python3
"""Auto-fix common type annotation errors."""

import re
from pathlib import Path

def fix_file(filepath: Path) -> bool:
    """Fix type annotations in a single file."""
    content = filepath.read_text()
    original = content

    # Fix: dict -> dict[str, Any]
    content = re.sub(
        r'(\b(?:def|async def)\s+\w+\([^)]*)\bdict\b(?!\[)',
        r'\1dict[str, Any]',
        content
    )

    # Fix: -> dict: to -> dict[str, Any]:
    content = re.sub(
        r'(\s+)->\s*dict\s*:',
        r'\1-> dict[str, Any]:',
        content
    )

    # Fix: list -> list[Any]
    content = re.sub(
        r'(\b(?:def|async def)\s+\w+\([^)]*)\blist\b(?!\[)',
        r'\1list[Any]',
        content
    )

    # Add Any import if needed and not present
    if content != original and 'from typing import' in content:
        if 'Any' not in content.split('from typing import')[1].split('\n')[0]:
            # Add Any to existing typing import
            content = re.sub(
                r'(from typing import [^)\n]+)',
                lambda m: m.group(1) if ', Any' in m.group(1) or ' Any' in m.group(1) else m.group(1).rstrip(')') + ', Any)',
                content
            )

    if content != original:
        filepath.write_text(content)
        return True
    return False

def main() -> None:
    """Run fixes on all Python files."""
    src_dir = Path("src")
    fixed_count = 0

    for py_file in src_dir.rglob("*.py"):
        if fix_file(py_file):
            fixed_count += 1
            print(f"Fixed: {py_file}")

    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()
