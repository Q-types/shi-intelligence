"""Transaction Fetching and Classification - Parse SWEENEE movements."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

import structlog

from .solana_client import SolanaClient, TransactionSignature, get_client

logger = structlog.get_logger()


class TransactionType(str, Enum):
    """Classification of SWEENEE token movements."""

    BUY = "buy"
    SELL = "sell"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    UNKNOWN = "unknown"


@dataclass
class SweeneeTransaction:
    """A SWEENEE token transaction involving a tracked wallet."""

    signature: str
    block_time: datetime | None
    wallet_address: str
    token_mint: str
    amount_change: float  # Positive = in, negative = out
    direction: str  # "in", "out", "neutral"
    classification: TransactionType
    counterparty: str | None = None
    dex_source: str = "unknown"  # jupiter_v6, jupiter_v4, raydium, orca, unknown
    explorer_url: str = ""
    raw: dict | None = field(default=None, repr=False)

    def __post_init__(self):
        if not self.explorer_url:
            self.explorer_url = f"https://solscan.io/tx/{self.signature}"

    @property
    def is_inflow(self) -> bool:
        return self.amount_change > 0

    @property
    def is_outflow(self) -> bool:
        return self.amount_change < 0

    @property
    def abs_amount(self) -> float:
        return abs(self.amount_change)


def classify_transaction(
    wallet: str,
    mint: str,
    tx_data: dict[str, Any],
    amount_change: float,
) -> tuple[TransactionType, str | None, str]:
    """Classify a transaction based on available data.

    Classification hierarchy:
    1. DEX swap with SWEENEE in + SOL/USDC out = buy
    2. DEX swap with SWEENEE out + SOL/USDC in = sell
    3. SWEENEE moves in without swap = transfer_in
    4. SWEENEE moves out without swap = transfer_out
    5. Otherwise = unknown

    Returns:
        Tuple of (TransactionType, counterparty, dex_source)
    """
    counterparty = None
    dex_source = "unknown"

    # Check if transaction is a DEX swap
    is_swap = False
    has_sol_movement = False
    has_stable_movement = False

    # Known DEX program IDs with names
    dex_programs = {
        "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "jupiter_v6",
        "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB": "jupiter_v4",
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "raydium",
        "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "orca",
        "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "orca_v2",
    }

    try:
        # Check account keys for DEX programs
        account_keys = (
            tx_data.get("transaction", {})
            .get("message", {})
            .get("accountKeys", [])
        )
        program_ids = [
            key.get("pubkey") if isinstance(key, dict) else key
            for key in account_keys
        ]

        for prog_id, prog_name in dex_programs.items():
            if prog_id in program_ids:
                is_swap = True
                dex_source = prog_name
                break

        # Check for SOL/stablecoin movements in pre/post balances
        meta = tx_data.get("meta", {})
        pre_balances = meta.get("preBalances", [])
        post_balances = meta.get("postBalances", [])

        if pre_balances and post_balances:
            # Find wallet index
            wallet_idx = None
            for idx, key in enumerate(account_keys):
                key_str = key.get("pubkey") if isinstance(key, dict) else key
                if key_str == wallet:
                    wallet_idx = idx
                    break

            if wallet_idx is not None and wallet_idx < len(pre_balances):
                sol_change = post_balances[wallet_idx] - pre_balances[wallet_idx]
                # Significant SOL movement (> 0.01 SOL = 10M lamports)
                if abs(sol_change) > 10_000_000:
                    has_sol_movement = True

        # Try to find counterparty from token transfers
        pre_token_balances = meta.get("preTokenBalances", [])
        post_token_balances = meta.get("postTokenBalances", [])

        # Look for other accounts with opposite SWEENEE movements
        for post_bal in post_token_balances:
            if post_bal.get("mint") == mint:
                owner = post_bal.get("owner", "")
                if owner and owner != wallet:
                    counterparty = owner
                    break

    except Exception as e:
        logger.debug("classification_parse_error", error=str(e))

    # Apply classification logic
    if amount_change > 0:
        if is_swap and has_sol_movement:
            return TransactionType.BUY, counterparty, dex_source
        else:
            return TransactionType.TRANSFER_IN, counterparty, "none"
    elif amount_change < 0:
        if is_swap and has_sol_movement:
            return TransactionType.SELL, counterparty, dex_source
        else:
            return TransactionType.TRANSFER_OUT, counterparty, "none"
    else:
        return TransactionType.UNKNOWN, counterparty, "unknown"


async def parse_transaction_for_sweenee(
    wallet: str,
    mint: str,
    signature: str,
    block_time: datetime | None,
    client: SolanaClient,
) -> SweeneeTransaction | None:
    """Parse a transaction for SWEENEE token movements."""
    tx_data = await client.get_transaction(signature)
    if not tx_data:
        return None

    try:
        meta = tx_data.get("meta", {})
        if meta.get("err"):
            # Failed transaction
            return None

        pre_token_balances = meta.get("preTokenBalances", [])
        post_token_balances = meta.get("postTokenBalances", [])

        # Find SWEENEE balance changes for this wallet
        pre_amount = 0
        post_amount = 0
        decimals = 6

        for bal in pre_token_balances:
            if bal.get("mint") == mint and bal.get("owner") == wallet:
                pre_amount = int(bal.get("uiTokenAmount", {}).get("amount", 0))
                decimals = bal.get("uiTokenAmount", {}).get("decimals", 6)

        for bal in post_token_balances:
            if bal.get("mint") == mint and bal.get("owner") == wallet:
                post_amount = int(bal.get("uiTokenAmount", {}).get("amount", 0))
                decimals = bal.get("uiTokenAmount", {}).get("decimals", 6)

        amount_change_raw = post_amount - pre_amount
        if amount_change_raw == 0:
            # No SWEENEE movement for this wallet
            return None

        amount_change = amount_change_raw / (10**decimals)

        # Determine direction
        if amount_change > 0:
            direction = "in"
        elif amount_change < 0:
            direction = "out"
        else:
            direction = "neutral"

        # Classify
        classification, counterparty, dex_source = classify_transaction(
            wallet, mint, tx_data, amount_change
        )

        return SweeneeTransaction(
            signature=signature,
            block_time=block_time,
            wallet_address=wallet,
            token_mint=mint,
            amount_change=amount_change,
            direction=direction,
            classification=classification,
            counterparty=counterparty,
            dex_source=dex_source,
            raw=tx_data,
        )

    except Exception as e:
        logger.warning(
            "transaction_parse_failed",
            signature=signature[:16],
            error=str(e),
        )
        return None


async def fetch_wallet_transactions(
    wallet: str,
    mint: str,
    limit: int = 50,
    client: SolanaClient | None = None,
) -> list[SweeneeTransaction]:
    """Fetch recent SWEENEE transactions for a wallet."""
    client = client or get_client()

    signatures = await client.get_signatures_for_address(wallet, limit=limit)

    transactions = []
    for sig in signatures:
        tx = await parse_transaction_for_sweenee(
            wallet, mint, sig.signature, sig.block_time, client
        )
        if tx:
            transactions.append(tx)

    logger.debug(
        "wallet_transactions_fetched",
        wallet=wallet[:8],
        total_sigs=len(signatures),
        sweenee_txs=len(transactions),
    )

    return transactions


async def fetch_all_transactions(
    wallets: list[str],
    mint: str,
    limit_per_wallet: int = 20,
    client: SolanaClient | None = None,
) -> list[SweeneeTransaction]:
    """Fetch SWEENEE transactions for all tracked wallets."""
    client = client or get_client()

    logger.info(
        "fetching_transactions",
        wallet_count=len(wallets),
        limit_per_wallet=limit_per_wallet,
    )

    tasks = [
        fetch_wallet_transactions(w, mint, limit_per_wallet, client) for w in wallets
    ]
    results = await asyncio.gather(*tasks)

    # Flatten and deduplicate by signature
    all_txs: dict[str, SweeneeTransaction] = {}
    for wallet_txs in results:
        for tx in wallet_txs:
            if tx.signature not in all_txs:
                all_txs[tx.signature] = tx

    # Sort by time (newest first)
    sorted_txs = sorted(
        all_txs.values(),
        key=lambda x: x.block_time or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    logger.info(
        "transactions_fetched",
        total=len(sorted_txs),
        unique=len(all_txs),
    )

    return sorted_txs


def compute_net_flow(
    transactions: list[SweeneeTransaction],
    wallet: str | None = None,
    hours: int = 24,
) -> float:
    """Compute net SWEENEE flow over a time window.

    Args:
        transactions: List of transactions to analyze
        wallet: Optional filter for specific wallet
        hours: Lookback window in hours

    Returns:
        Net flow (positive = inflow, negative = outflow)
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    net = 0.0
    for tx in transactions:
        if wallet and tx.wallet_address != wallet:
            continue
        if tx.block_time and tx.block_time < cutoff:
            continue
        net += tx.amount_change

    return net


def count_transactions(
    transactions: list[SweeneeTransaction],
    wallet: str | None = None,
    hours: int = 24,
) -> int:
    """Count transactions in a time window."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    count = 0
    for tx in transactions:
        if wallet and tx.wallet_address != wallet:
            continue
        if tx.block_time and tx.block_time < cutoff:
            continue
        count += 1

    return count


def find_largest_movement(
    transactions: list[SweeneeTransaction],
    direction: str = "in",
    hours: int = 24,
) -> SweeneeTransaction | None:
    """Find the largest inflow or outflow in a time window."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    candidates = []
    for tx in transactions:
        if tx.block_time and tx.block_time < cutoff:
            continue
        if direction == "in" and tx.amount_change > 0:
            candidates.append(tx)
        elif direction == "out" and tx.amount_change < 0:
            candidates.append(tx)

    if not candidates:
        return None

    if direction == "in":
        return max(candidates, key=lambda x: x.amount_change)
    else:
        return min(candidates, key=lambda x: x.amount_change)
