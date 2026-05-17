"""Tests for wallet loader module."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wallet_loader import (
    TrackedWallet,
    is_valid_solana_address,
    load_wallets_from_txt,
    load_wallets_from_csv,
    load_wallets_from_json,
    load_all_wallets,
)


# Valid test addresses (base58 format)
VALID_ADDRESSES = [
    "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
    "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
]


class TestAddressValidation:
    """Tests for Solana address validation."""

    def test_valid_addresses(self):
        for addr in VALID_ADDRESSES:
            assert is_valid_solana_address(addr), f"Should be valid: {addr}"

    def test_invalid_addresses(self):
        invalid = [
            "",
            "abc",
            "0x1234567890abcdef",  # Ethereum format
            "too_short",
            "this_contains_invalid_chars_like_0OIl",
        ]
        for addr in invalid:
            assert not is_valid_solana_address(addr), f"Should be invalid: {addr}"

    def test_none_address(self):
        assert not is_valid_solana_address(None)

    def test_non_string(self):
        assert not is_valid_solana_address(12345)


class TestTrackedWallet:
    """Tests for TrackedWallet dataclass."""

    def test_create_valid_wallet(self):
        wallet = TrackedWallet(address=VALID_ADDRESSES[0])
        assert wallet.address == VALID_ADDRESSES[0]
        assert wallet.label is None

    def test_create_wallet_with_label(self):
        wallet = TrackedWallet(address=VALID_ADDRESSES[0], label="Whale 1")
        assert wallet.label == "Whale 1"

    def test_short_address(self):
        wallet = TrackedWallet(address=VALID_ADDRESSES[0])
        short = wallet.short_address
        assert short.startswith(VALID_ADDRESSES[0][:4])
        assert short.endswith(VALID_ADDRESSES[0][-4:])
        assert "..." in short

    def test_display_name_with_label(self):
        wallet = TrackedWallet(address=VALID_ADDRESSES[0], label="Whale 1")
        assert wallet.display_name() == "Whale 1"

    def test_display_name_without_label(self):
        wallet = TrackedWallet(address=VALID_ADDRESSES[0])
        assert "..." in wallet.display_name()

    def test_invalid_address_raises(self):
        with pytest.raises(ValueError):
            TrackedWallet(address="invalid")


class TestLoadFromTxt:
    """Tests for loading wallets from text files."""

    def test_load_simple_txt(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wallets.txt"
            path.write_text("\n".join(VALID_ADDRESSES))

            wallets = list(load_wallets_from_txt(path))
            assert len(wallets) == 3

    def test_load_txt_with_labels(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wallets.txt"
            content = f"Whale 1: {VALID_ADDRESSES[0]}\nWhale 2: {VALID_ADDRESSES[1]}"
            path.write_text(content)

            wallets = list(load_wallets_from_txt(path))
            assert len(wallets) == 2
            assert wallets[0].label == "Whale 1"

    def test_load_txt_skips_comments(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wallets.txt"
            content = f"# Comment\n{VALID_ADDRESSES[0]}\n# Another comment"
            path.write_text(content)

            wallets = list(load_wallets_from_txt(path))
            assert len(wallets) == 1

    def test_load_txt_skips_empty_lines(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wallets.txt"
            content = f"\n{VALID_ADDRESSES[0]}\n\n{VALID_ADDRESSES[1]}\n\n"
            path.write_text(content)

            wallets = list(load_wallets_from_txt(path))
            assert len(wallets) == 2


class TestLoadFromCsv:
    """Tests for loading wallets from CSV files."""

    def test_load_csv_with_address_column(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wallets.csv"
            content = f"address\n{VALID_ADDRESSES[0]}\n{VALID_ADDRESSES[1]}"
            path.write_text(content)

            wallets = list(load_wallets_from_csv(path))
            assert len(wallets) == 2

    def test_load_csv_with_wallet_column(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wallets.csv"
            content = f"wallet\n{VALID_ADDRESSES[0]}"
            path.write_text(content)

            wallets = list(load_wallets_from_csv(path))
            assert len(wallets) == 1

    def test_load_csv_with_label(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wallets.csv"
            content = f"label,address\nWhale 1,{VALID_ADDRESSES[0]}"
            path.write_text(content)

            wallets = list(load_wallets_from_csv(path))
            assert len(wallets) == 1
            assert wallets[0].label == "Whale 1"


class TestLoadFromJson:
    """Tests for loading wallets from JSON files."""

    def test_load_json_array_of_strings(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wallets.json"
            path.write_text(json.dumps(VALID_ADDRESSES))

            wallets = list(load_wallets_from_json(path))
            assert len(wallets) == 3

    def test_load_json_array_of_objects(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wallets.json"
            data = [
                {"address": VALID_ADDRESSES[0], "label": "Whale 1"},
                {"address": VALID_ADDRESSES[1], "label": "Whale 2"},
            ]
            path.write_text(json.dumps(data))

            wallets = list(load_wallets_from_json(path))
            assert len(wallets) == 2
            assert wallets[0].label == "Whale 1"

    def test_load_json_with_wallets_key(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wallets.json"
            data = {"wallets": VALID_ADDRESSES}
            path.write_text(json.dumps(data))

            wallets = list(load_wallets_from_json(path))
            assert len(wallets) == 3


class TestLoadAllWallets:
    """Tests for loading wallets from directory."""

    def test_load_from_multiple_files(self):
        with TemporaryDirectory() as tmpdir:
            wallets_dir = Path(tmpdir)

            # Create txt file
            txt_path = wallets_dir / "list1.txt"
            txt_path.write_text(VALID_ADDRESSES[0])

            # Create json file
            json_path = wallets_dir / "list2.json"
            json_path.write_text(json.dumps([VALID_ADDRESSES[1]]))

            wallets = load_all_wallets(wallets_dir)
            assert len(wallets) == 2

    def test_deduplication(self):
        with TemporaryDirectory() as tmpdir:
            wallets_dir = Path(tmpdir)

            # Same address in two files
            txt1 = wallets_dir / "list1.txt"
            txt1.write_text(VALID_ADDRESSES[0])

            txt2 = wallets_dir / "list2.txt"
            txt2.write_text(VALID_ADDRESSES[0])

            wallets = load_all_wallets(wallets_dir)
            assert len(wallets) == 1

    def test_empty_directory(self):
        with TemporaryDirectory() as tmpdir:
            wallets = load_all_wallets(Path(tmpdir))
            assert len(wallets) == 0

    def test_nonexistent_directory(self):
        wallets = load_all_wallets(Path("/nonexistent/path"))
        assert len(wallets) == 0
