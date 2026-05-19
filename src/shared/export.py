"""Export Service - Export wallet and transaction data to CSV/JSON."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from .token_balances import WalletBalance
from .transactions import SweeneeTransaction


def export_wallets_csv(balances: list[WalletBalance]) -> str:
    """Export wallet balances to CSV format.

    Args:
        balances: List of wallet balances

    Returns:
        CSV string
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "address",
        "label",
        "balance",
        "share_of_tracked",
        "fetched_at",
    ])

    # Data
    for bal in balances:
        writer.writerow([
            bal.address,
            bal.label or "",
            bal.ui_amount,
            f"{bal.share_of_tracked * 100:.2f}%",
            bal.balance.fetched_at.isoformat() if bal.balance else "",
        ])

    return output.getvalue()


def export_wallets_json(balances: list[WalletBalance]) -> str:
    """Export wallet balances to JSON format.

    Args:
        balances: List of wallet balances

    Returns:
        JSON string
    """
    data = []
    for bal in balances:
        data.append({
            "address": bal.address,
            "label": bal.label,
            "balance": bal.ui_amount,
            "share_of_tracked": bal.share_of_tracked,
            "fetched_at": bal.balance.fetched_at.isoformat() if bal.balance else None,
        })

    return json.dumps({
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "wallet_count": len(data),
        "total_balance": sum(b["balance"] for b in data),
        "wallets": data,
    }, indent=2)


def export_transactions_csv(transactions: list[SweeneeTransaction]) -> str:
    """Export transactions to CSV format.

    Args:
        transactions: List of transactions

    Returns:
        CSV string
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "signature",
        "block_time",
        "wallet_address",
        "amount_change",
        "direction",
        "classification",
        "dex_source",
        "counterparty",
        "explorer_url",
    ])

    # Data
    for tx in transactions:
        writer.writerow([
            tx.signature,
            tx.block_time.isoformat() if tx.block_time else "",
            tx.wallet_address,
            tx.amount_change,
            tx.direction,
            tx.classification.value,
            getattr(tx, "dex_source", "unknown"),
            tx.counterparty or "",
            tx.explorer_url,
        ])

    return output.getvalue()


def export_transactions_json(transactions: list[SweeneeTransaction]) -> str:
    """Export transactions to JSON format.

    Args:
        transactions: List of transactions

    Returns:
        JSON string
    """
    data = []
    for tx in transactions:
        data.append({
            "signature": tx.signature,
            "block_time": tx.block_time.isoformat() if tx.block_time else None,
            "wallet_address": tx.wallet_address,
            "amount_change": tx.amount_change,
            "direction": tx.direction,
            "classification": tx.classification.value,
            "dex_source": getattr(tx, "dex_source", "unknown"),
            "counterparty": tx.counterparty,
            "explorer_url": tx.explorer_url,
        })

    # Calculate summary stats
    buys = [t for t in data if t["classification"] == "buy"]
    sells = [t for t in data if t["classification"] == "sell"]

    return json.dumps({
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "transaction_count": len(data),
        "summary": {
            "buy_count": len(buys),
            "sell_count": len(sells),
            "total_bought": sum(abs(t["amount_change"]) for t in buys),
            "total_sold": sum(abs(t["amount_change"]) for t in sells),
        },
        "transactions": data,
    }, indent=2)


def get_export_filename(export_type: str, format: str) -> str:
    """Generate export filename with timestamp.

    Args:
        export_type: Type of export (wallets, transactions)
        format: File format (csv, json)

    Returns:
        Filename string
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"sweenee_{export_type}_{timestamp}.{format}"
