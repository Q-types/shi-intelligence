"""SQLite Cache Layer - Persist balances and transactions locally."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Generator

import structlog

from .solana_client import TokenBalance
from .token_balances import WalletBalance
from .transactions import SweeneeTransaction, TransactionType

logger = structlog.get_logger()

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "sweenee.sqlite"

# Schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS tracked_wallets (
    address TEXT PRIMARY KEY,
    label TEXT,
    notes TEXT,
    source_file TEXT,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wallet_balances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_address TEXT NOT NULL,
    token_mint TEXT NOT NULL,
    raw_amount INTEGER NOT NULL,
    decimals INTEGER NOT NULL,
    ui_amount REAL NOT NULL,
    fetched_at TEXT NOT NULL,
    UNIQUE(wallet_address, token_mint, fetched_at)
);

CREATE TABLE IF NOT EXISTS sweenee_transactions (
    signature TEXT PRIMARY KEY,
    block_time TEXT,
    wallet_address TEXT NOT NULL,
    token_mint TEXT NOT NULL,
    amount_change REAL NOT NULL,
    direction TEXT NOT NULL,
    classification TEXT NOT NULL,
    counterparty TEXT,
    explorer_url TEXT,
    raw_json TEXT,
    cached_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dashboard_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    wallet_count INTEGER,
    total_balance REAL,
    transaction_count INTEGER,
    summary_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_balances_wallet ON wallet_balances(wallet_address);
CREATE INDEX IF NOT EXISTS idx_balances_fetched ON wallet_balances(fetched_at);
CREATE INDEX IF NOT EXISTS idx_txs_wallet ON sweenee_transactions(wallet_address);
CREATE INDEX IF NOT EXISTS idx_txs_time ON sweenee_transactions(block_time);
"""


class SweeneeCache:
    """SQLite cache for SWEENEE dashboard data."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()
        logger.debug("cache_initialized", path=str(self.db_path))

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # --- Wallet Methods ---

    def save_wallets(self, wallets: list[dict[str, Any]]):
        """Save tracked wallets to cache."""
        with self._connect() as conn:
            for w in wallets:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tracked_wallets
                    (address, label, notes, source_file)
                    VALUES (?, ?, ?, ?)
                    """,
                    (w["address"], w.get("label"), w.get("notes"), w.get("source_file")),
                )
            conn.commit()

    def get_wallets(self) -> list[dict[str, Any]]:
        """Get all tracked wallets from cache."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM tracked_wallets").fetchall()
            return [dict(row) for row in rows]

    # --- Balance Methods ---

    def save_balances(self, balances: list[WalletBalance]):
        """Save wallet balances to cache."""
        with self._connect() as conn:
            for bal in balances:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO wallet_balances
                    (wallet_address, token_mint, raw_amount, decimals, ui_amount, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bal.address,
                        bal.balance.token_mint,
                        bal.balance.raw_amount,
                        bal.balance.decimals,
                        bal.balance.ui_amount,
                        bal.balance.fetched_at.isoformat(),
                    ),
                )
            conn.commit()
        logger.debug("balances_cached", count=len(balances))

    def get_cached_balances(
        self, mint: str, max_age_seconds: int = 300
    ) -> list[WalletBalance] | None:
        """Get cached balances if fresh enough."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)

        with self._connect() as conn:
            # Check if we have fresh data
            latest = conn.execute(
                """
                SELECT MAX(fetched_at) as latest FROM wallet_balances
                WHERE token_mint = ?
                """,
                (mint,),
            ).fetchone()

            if not latest or not latest["latest"]:
                return None

            latest_time = datetime.fromisoformat(latest["latest"])
            if latest_time < cutoff:
                return None

            # Get balances from latest fetch
            rows = conn.execute(
                """
                SELECT * FROM wallet_balances
                WHERE token_mint = ? AND fetched_at = ?
                """,
                (mint, latest["latest"]),
            ).fetchall()

            balances = []
            total = sum(row["ui_amount"] for row in rows)

            for row in rows:
                balance = TokenBalance(
                    wallet_address=row["wallet_address"],
                    token_mint=row["token_mint"],
                    raw_amount=row["raw_amount"],
                    decimals=row["decimals"],
                    ui_amount=row["ui_amount"],
                    fetched_at=datetime.fromisoformat(row["fetched_at"]),
                )
                wb = WalletBalance(
                    address=row["wallet_address"],
                    label=None,  # Will be filled from tracked_wallets
                    balance=balance,
                    share_of_tracked=row["ui_amount"] / total if total > 0 else 0,
                )
                balances.append(wb)

            # Sort by balance
            balances.sort(key=lambda x: x.ui_amount, reverse=True)

            logger.debug("balances_from_cache", count=len(balances))
            return balances

    # --- Transaction Methods ---

    def save_transactions(self, transactions: list[SweeneeTransaction]):
        """Save transactions to cache."""
        with self._connect() as conn:
            for tx in transactions:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO sweenee_transactions
                    (signature, block_time, wallet_address, token_mint, amount_change,
                     direction, classification, counterparty, explorer_url, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tx.signature,
                        tx.block_time.isoformat() if tx.block_time else None,
                        tx.wallet_address,
                        tx.token_mint,
                        tx.amount_change,
                        tx.direction,
                        tx.classification.value,
                        tx.counterparty,
                        tx.explorer_url,
                        json.dumps(tx.raw) if tx.raw else None,
                    ),
                )
            conn.commit()
        logger.debug("transactions_cached", count=len(transactions))

    def get_cached_transactions(
        self, mint: str, hours: int = 168, limit: int = 500
    ) -> list[SweeneeTransaction]:
        """Get cached transactions."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sweenee_transactions
                WHERE token_mint = ? AND (block_time IS NULL OR block_time >= ?)
                ORDER BY block_time DESC
                LIMIT ?
                """,
                (mint, cutoff.isoformat(), limit),
            ).fetchall()

            transactions = []
            for row in rows:
                tx = SweeneeTransaction(
                    signature=row["signature"],
                    block_time=(
                        datetime.fromisoformat(row["block_time"])
                        if row["block_time"]
                        else None
                    ),
                    wallet_address=row["wallet_address"],
                    token_mint=row["token_mint"],
                    amount_change=row["amount_change"],
                    direction=row["direction"],
                    classification=TransactionType(row["classification"]),
                    counterparty=row["counterparty"],
                    explorer_url=row["explorer_url"],
                    raw=json.loads(row["raw_json"]) if row["raw_json"] else None,
                )
                transactions.append(tx)

            logger.debug("transactions_from_cache", count=len(transactions))
            return transactions

    # --- Dashboard Run Methods ---

    def save_dashboard_run(
        self,
        wallet_count: int,
        total_balance: float,
        transaction_count: int,
        summary: dict[str, Any] | None = None,
    ):
        """Record a dashboard run."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO dashboard_runs
                (run_at, wallet_count, total_balance, transaction_count, summary_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    wallet_count,
                    total_balance,
                    transaction_count,
                    json.dumps(summary) if summary else None,
                ),
            )
            conn.commit()

    def get_historical_totals(self, days: int = 30) -> list[dict[str, Any]]:
        """Get historical total balance data for charting."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_at, total_balance, wallet_count
                FROM dashboard_runs
                WHERE run_at >= ?
                ORDER BY run_at ASC
                """,
                (cutoff.isoformat(),),
            ).fetchall()

            return [dict(row) for row in rows]


# Global cache instance
_cache: SweeneeCache | None = None


def get_cache(db_path: Path | str | None = None) -> SweeneeCache:
    """Get or create the cache singleton."""
    global _cache
    if _cache is None:
        _cache = SweeneeCache(db_path)
    return _cache
