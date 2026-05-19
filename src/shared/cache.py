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
    dex_source TEXT DEFAULT 'unknown',
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

CREATE TABLE IF NOT EXISTS balance_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_address TEXT NOT NULL,
    token_mint TEXT NOT NULL,
    ui_amount REAL NOT NULL,
    snapshot_date TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(wallet_address, token_mint, snapshot_date)
);

CREATE TABLE IF NOT EXISTS whale_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_address TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    amount REAL NOT NULL,
    threshold_triggered REAL,
    tx_signature TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    acknowledged INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS webhook_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_hash TEXT UNIQUE NOT NULL,
    message_type TEXT NOT NULL,
    payload_preview TEXT,
    sent_at TEXT DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL,
    retry_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_balances_wallet ON wallet_balances(wallet_address);
CREATE INDEX IF NOT EXISTS idx_balances_fetched ON wallet_balances(fetched_at);
CREATE INDEX IF NOT EXISTS idx_txs_wallet ON sweenee_transactions(wallet_address);
CREATE INDEX IF NOT EXISTS idx_txs_time ON sweenee_transactions(block_time);
CREATE INDEX IF NOT EXISTS idx_snapshots_wallet_date ON balance_snapshots(wallet_address, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_snapshots_date ON balance_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON whale_alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_ack ON whale_alerts(acknowledged);
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
            # Run migrations for existing databases
            self._run_migrations(conn)
        logger.debug("cache_initialized", path=str(self.db_path))

    def _run_migrations(self, conn: sqlite3.Connection):
        """Run schema migrations for existing databases."""
        # Check if dex_source column exists in sweenee_transactions
        cursor = conn.execute("PRAGMA table_info(sweenee_transactions)")
        columns = [row[1] for row in cursor.fetchall()]

        if "dex_source" not in columns:
            conn.execute("ALTER TABLE sweenee_transactions ADD COLUMN dex_source TEXT DEFAULT 'unknown'")
            conn.commit()
            logger.info("migration_applied", migration="add_dex_source_column")

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

            # Get balances from latest fetch batch (within 60 seconds of latest)
            # This handles async fetches where each wallet has slightly different timestamp
            batch_cutoff = (latest_time - timedelta(seconds=60)).isoformat()
            rows = conn.execute(
                """
                SELECT * FROM wallet_balances
                WHERE token_mint = ? AND fetched_at >= ?
                """,
                (mint, batch_cutoff),
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
                     direction, classification, counterparty, dex_source, explorer_url, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        tx.dex_source,
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
                    dex_source=row["dex_source"] if "dex_source" in row.keys() else "unknown",
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

    # --- Balance Snapshot Methods ---

    def save_balance_snapshot(
        self, wallet_address: str, token_mint: str, ui_amount: float, snapshot_date: str | None = None
    ):
        """Save a daily balance snapshot (upsert by wallet+date)."""
        if snapshot_date is None:
            snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO balance_snapshots (wallet_address, token_mint, ui_amount, snapshot_date)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(wallet_address, token_mint, snapshot_date)
                DO UPDATE SET ui_amount = excluded.ui_amount, created_at = CURRENT_TIMESTAMP
                """,
                (wallet_address, token_mint, ui_amount, snapshot_date),
            )
            conn.commit()

    def save_balance_snapshots_batch(self, balances: list[WalletBalance], mint: str):
        """Save balance snapshots for all wallets (batch operation)."""
        snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._connect() as conn:
            for bal in balances:
                conn.execute(
                    """
                    INSERT INTO balance_snapshots (wallet_address, token_mint, ui_amount, snapshot_date)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(wallet_address, token_mint, snapshot_date)
                    DO UPDATE SET ui_amount = excluded.ui_amount, created_at = CURRENT_TIMESTAMP
                    """,
                    (bal.address, mint, bal.ui_amount, snapshot_date),
                )
            conn.commit()
        logger.debug("snapshots_saved", count=len(balances), date=snapshot_date)

    def get_wallet_history(
        self, wallet_address: str, mint: str, days: int = 30
    ) -> list[dict[str, Any]]:
        """Get historical balances for a single wallet."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT snapshot_date, ui_amount
                FROM balance_snapshots
                WHERE wallet_address = ? AND token_mint = ? AND snapshot_date >= ?
                ORDER BY snapshot_date ASC
                """,
                (wallet_address, mint, cutoff),
            ).fetchall()

            return [dict(row) for row in rows]

    def get_all_wallet_history(self, mint: str, days: int = 30) -> list[dict[str, Any]]:
        """Get historical balances for all wallets (for stacked area chart)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT wallet_address, snapshot_date, ui_amount
                FROM balance_snapshots
                WHERE token_mint = ? AND snapshot_date >= ?
                ORDER BY snapshot_date ASC, wallet_address ASC
                """,
                (mint, cutoff),
            ).fetchall()

            return [dict(row) for row in rows]

    def get_total_history(self, mint: str, days: int = 30) -> list[dict[str, Any]]:
        """Get aggregated total balance history by date."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT snapshot_date, SUM(ui_amount) as total_balance, COUNT(*) as wallet_count
                FROM balance_snapshots
                WHERE token_mint = ? AND snapshot_date >= ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date ASC
                """,
                (mint, cutoff),
            ).fetchall()

            return [dict(row) for row in rows]

    # --- Alert Methods ---

    def save_alert(
        self,
        wallet_address: str,
        alert_type: str,
        amount: float,
        threshold_triggered: float | None = None,
        tx_signature: str | None = None,
    ) -> int:
        """Save a whale alert and return its ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO whale_alerts (wallet_address, alert_type, amount, threshold_triggered, tx_signature)
                VALUES (?, ?, ?, ?, ?)
                """,
                (wallet_address, alert_type, amount, threshold_triggered, tx_signature),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_recent_alerts(self, hours: int = 24, include_acknowledged: bool = False) -> list[dict[str, Any]]:
        """Get recent alerts."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        with self._connect() as conn:
            if include_acknowledged:
                rows = conn.execute(
                    """
                    SELECT * FROM whale_alerts
                    WHERE created_at >= ?
                    ORDER BY created_at DESC
                    """,
                    (cutoff,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM whale_alerts
                    WHERE created_at >= ? AND acknowledged = 0
                    ORDER BY created_at DESC
                    """,
                    (cutoff,),
                ).fetchall()

            return [dict(row) for row in rows]

    def acknowledge_alert(self, alert_id: int):
        """Mark an alert as acknowledged."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE whale_alerts SET acknowledged = 1 WHERE id = ?",
                (alert_id,),
            )
            conn.commit()

    # --- Webhook Log Methods ---

    def log_webhook(
        self, message_hash: str, message_type: str, payload_preview: str, status: str
    ) -> bool:
        """Log a webhook send attempt. Returns False if already sent (idempotency check)."""
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO webhook_log (message_hash, message_type, payload_preview, status)
                    VALUES (?, ?, ?, ?)
                    """,
                    (message_hash, message_type, payload_preview, status),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Already exists - idempotency check
                return False

    def update_webhook_status(self, message_hash: str, status: str, increment_retry: bool = False):
        """Update webhook status (for retries)."""
        with self._connect() as conn:
            if increment_retry:
                conn.execute(
                    """
                    UPDATE webhook_log
                    SET status = ?, retry_count = retry_count + 1
                    WHERE message_hash = ?
                    """,
                    (status, message_hash),
                )
            else:
                conn.execute(
                    "UPDATE webhook_log SET status = ? WHERE message_hash = ?",
                    (status, message_hash),
                )
            conn.commit()

    def was_webhook_sent(self, message_hash: str) -> bool:
        """Check if a webhook was already sent (idempotency)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM webhook_log WHERE message_hash = ? AND status = 'sent'",
                (message_hash,),
            ).fetchone()
            return row is not None


# Global cache instance
_cache: SweeneeCache | None = None


def get_cache(db_path: Path | str | None = None) -> SweeneeCache:
    """Get or create the cache singleton."""
    global _cache
    if _cache is None:
        _cache = SweeneeCache(db_path)
    return _cache
