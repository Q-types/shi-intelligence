"""Alert Service - Detect and track significant whale movements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from .cache import get_cache
from .transactions import SweeneeTransaction, TransactionType

logger = structlog.get_logger()


class AlertType(Enum):
    """Types of whale alerts."""

    LARGE_BUY = "large_buy"
    LARGE_SELL = "large_sell"
    NEW_HOLDER = "new_holder"
    WHALE_EXIT = "whale_exit"


@dataclass
class WhaleAlert:
    """A whale movement alert."""

    id: int | None
    wallet_address: str
    alert_type: AlertType
    amount: float
    threshold_triggered: float | None
    tx_signature: str | None
    created_at: datetime
    acknowledged: bool = False

    @property
    def emoji(self) -> str:
        """Get emoji for alert type."""
        return {
            AlertType.LARGE_BUY: "🟢",
            AlertType.LARGE_SELL: "🔴",
            AlertType.NEW_HOLDER: "🆕",
            AlertType.WHALE_EXIT: "🚪",
        }.get(self.alert_type, "⚠️")

    @property
    def description(self) -> str:
        """Get human-readable description."""
        amount_str = f"{self.amount:,.0f}"
        if self.alert_type == AlertType.LARGE_BUY:
            return f"Whale bought {amount_str} SWEENEE"
        elif self.alert_type == AlertType.LARGE_SELL:
            return f"Whale sold {amount_str} SWEENEE"
        elif self.alert_type == AlertType.NEW_HOLDER:
            return f"New whale appeared with {amount_str} SWEENEE"
        elif self.alert_type == AlertType.WHALE_EXIT:
            return f"Whale exited position ({amount_str} sold)"
        return f"Alert: {amount_str} SWEENEE"


class AlertService:
    """Service for detecting and managing whale alerts."""

    def __init__(
        self,
        large_move_threshold: float = 1_000_000,
        exit_threshold: float = 100,  # Balance below this = exit
    ):
        """Initialize alert service.

        Args:
            large_move_threshold: Minimum amount for large buy/sell alerts
            exit_threshold: Balance below this triggers exit alert
        """
        self.large_move_threshold = large_move_threshold
        self.exit_threshold = exit_threshold
        self.cache = get_cache()

    def check_transactions(
        self,
        transactions: list[SweeneeTransaction],
        current_balances: dict[str, float] | None = None,
    ) -> list[WhaleAlert]:
        """Check transactions for alertable movements.

        Args:
            transactions: Recent transactions to check
            current_balances: Optional current balances for exit detection

        Returns:
            List of new alerts generated
        """
        alerts = []

        for tx in transactions:
            alert = self._check_transaction(tx, current_balances)
            if alert:
                # Save to database
                alert_id = self.cache.save_alert(
                    wallet_address=alert.wallet_address,
                    alert_type=alert.alert_type.value,
                    amount=alert.amount,
                    threshold_triggered=alert.threshold_triggered,
                    tx_signature=alert.tx_signature,
                )
                alert.id = alert_id
                alerts.append(alert)
                logger.info(
                    "alert_generated",
                    type=alert.alert_type.value,
                    wallet=alert.wallet_address[:8],
                    amount=alert.amount,
                )

        return alerts

    def _check_transaction(
        self,
        tx: SweeneeTransaction,
        current_balances: dict[str, float] | None = None,
    ) -> WhaleAlert | None:
        """Check a single transaction for alert conditions."""
        amount = abs(tx.amount_change)

        # Large buy detection
        if tx.classification == TransactionType.BUY and amount >= self.large_move_threshold:
            return WhaleAlert(
                id=None,
                wallet_address=tx.wallet_address,
                alert_type=AlertType.LARGE_BUY,
                amount=amount,
                threshold_triggered=self.large_move_threshold,
                tx_signature=tx.signature,
                created_at=tx.block_time or datetime.now(timezone.utc),
            )

        # Large sell detection
        if tx.classification == TransactionType.SELL and amount >= self.large_move_threshold:
            # Check if this is an exit (balance near zero)
            if current_balances:
                current_bal = current_balances.get(tx.wallet_address, 0)
                if current_bal < self.exit_threshold:
                    return WhaleAlert(
                        id=None,
                        wallet_address=tx.wallet_address,
                        alert_type=AlertType.WHALE_EXIT,
                        amount=amount,
                        threshold_triggered=self.exit_threshold,
                        tx_signature=tx.signature,
                        created_at=tx.block_time or datetime.now(timezone.utc),
                    )

            return WhaleAlert(
                id=None,
                wallet_address=tx.wallet_address,
                alert_type=AlertType.LARGE_SELL,
                amount=amount,
                threshold_triggered=self.large_move_threshold,
                tx_signature=tx.signature,
                created_at=tx.block_time or datetime.now(timezone.utc),
            )

        return None

    def get_recent_alerts(
        self, hours: int = 24, include_acknowledged: bool = False
    ) -> list[WhaleAlert]:
        """Get recent alerts from database."""
        rows = self.cache.get_recent_alerts(hours, include_acknowledged)
        return [self._row_to_alert(row) for row in rows]

    def get_unacknowledged_alerts(self) -> list[WhaleAlert]:
        """Get all unacknowledged alerts."""
        return self.get_recent_alerts(hours=168, include_acknowledged=False)

    def acknowledge_alert(self, alert_id: int):
        """Mark an alert as acknowledged."""
        self.cache.acknowledge_alert(alert_id)

    def acknowledge_all(self):
        """Acknowledge all pending alerts."""
        alerts = self.get_unacknowledged_alerts()
        for alert in alerts:
            if alert.id:
                self.acknowledge_alert(alert.id)

    def _row_to_alert(self, row: dict[str, Any]) -> WhaleAlert:
        """Convert database row to WhaleAlert."""
        return WhaleAlert(
            id=row["id"],
            wallet_address=row["wallet_address"],
            alert_type=AlertType(row["alert_type"]),
            amount=row["amount"],
            threshold_triggered=row["threshold_triggered"],
            tx_signature=row["tx_signature"],
            created_at=datetime.fromisoformat(row["created_at"]),
            acknowledged=bool(row["acknowledged"]),
        )


def render_alert_banners(alerts: list[WhaleAlert], wallet_labels: dict[str, str] | None = None) -> str:
    """Render alerts as HTML banners for Streamlit.

    Args:
        alerts: List of alerts to render
        wallet_labels: Optional mapping of addresses to labels

    Returns:
        HTML string for st.markdown
    """
    if not alerts:
        return ""

    html_parts = []

    for alert in alerts[:5]:  # Show max 5 alerts
        wallet_display = wallet_labels.get(alert.wallet_address, alert.wallet_address[:8] + "...") if wallet_labels else alert.wallet_address[:8] + "..."

        if alert.alert_type in (AlertType.LARGE_BUY, AlertType.NEW_HOLDER):
            bg_color = "rgba(76, 175, 80, 0.2)"
            border_color = "#4CAF50"
            text_color = "#4CAF50"
        else:
            bg_color = "rgba(244, 67, 54, 0.2)"
            border_color = "#F44336"
            text_color = "#F44336"

        html_parts.append(f"""
        <div style="background: {bg_color}; padding: 0.75rem 1rem; border-radius: 8px; margin: 0.5rem 0; border: 1px solid {border_color};">
            <span style="font-size: 1.3rem;">{alert.emoji}</span>
            <span style="font-weight: 600; color: {text_color}; margin-left: 0.5rem;">{alert.description}</span>
            <span style="color: #aaa; margin-left: 0.5rem;">({wallet_display})</span>
        </div>
        """)

    return "\n".join(html_parts)
