"""
Exit Classifier Monitoring (Sprint 9.5).

Provides production monitoring for exit classification:
- Full evidence logging for every classification
- Classification distribution metrics
- Anomaly detection and alerting
- Validation dataset collection

HARD RULES:
1. All classifications must be logged with full evidence
2. Unknown exits must be logged, not hidden
3. Classifier errors must be reviewable
4. Low-confidence exits cannot produce precise PnL
5. Real-world validation required before using exit classes as training labels
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import structlog

from .exit_classifier import (
    ExitEventClassification,
    ExitEventType,
    ExitEvidence,
)

logger = structlog.get_logger()


# ============================================================================
# Classification Log Record
# ============================================================================


@dataclass(frozen=True)
class ClassificationLogRecord:
    """Complete log record for an exit classification."""

    # Timestamp
    logged_at: datetime

    # Transaction identity
    signature: str
    slot: int
    token_mint: str
    wallet_address: str

    # Classification result
    exit_type: str
    confidence: float
    sell_confidence: float
    pnl_computable: bool

    # Evidence summary
    program_ids: tuple[str, ...]
    dex_detected: str | None
    lp_program_detected: str | None
    bridge_detected: str | None
    destination_address: str | None
    destination_type: str | None  # cex, wallet, pool, burn, unknown

    # Display mode for PnL
    display_mode: str  # precise, range, unavailable

    # Unknown handling
    unknown_reason: str | None  # If UNKNOWN_EXIT, why

    # Full evidence JSON for review
    evidence_json: str
    confidence_factors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "logged_at": self.logged_at.isoformat(),
            "signature": self.signature,
            "slot": self.slot,
            "token_mint": self.token_mint,
            "wallet_address": self.wallet_address,
            "exit_type": self.exit_type,
            "confidence": self.confidence,
            "sell_confidence": self.sell_confidence,
            "pnl_computable": self.pnl_computable,
            "program_ids": list(self.program_ids),
            "dex_detected": self.dex_detected,
            "lp_program_detected": self.lp_program_detected,
            "bridge_detected": self.bridge_detected,
            "destination_address": self.destination_address,
            "destination_type": self.destination_type,
            "display_mode": self.display_mode,
            "unknown_reason": self.unknown_reason,
            "evidence_json": self.evidence_json,
            "confidence_factors": list(self.confidence_factors),
        }


# ============================================================================
# Classification Distribution Metrics
# ============================================================================


@dataclass
class ClassificationDistribution:
    """Distribution metrics for exit classifications."""

    # Counts by type
    counts: dict[str, int] = field(default_factory=dict)
    total: int = 0

    # Percentages
    percentages: dict[str, float] = field(default_factory=dict)

    # Confidence by type
    mean_confidence: dict[str, float] = field(default_factory=dict)
    median_confidence: dict[str, float] = field(default_factory=dict)

    # Time window
    window_start: datetime | None = None
    window_end: datetime | None = None

    def compute_percentages(self) -> None:
        """Compute percentages from counts."""
        if self.total == 0:
            return
        self.percentages = {
            exit_type: (count / self.total) * 100
            for exit_type, count in self.counts.items()
        }


@dataclass
class DistributionSnapshot:
    """Point-in-time snapshot of distribution for drift detection."""

    timestamp: datetime
    distribution: ClassificationDistribution
    dex_sell_pct: float
    transfer_out_pct: float
    unknown_exit_pct: float
    lp_action_pct: float  # LP_ADD + LP_REMOVE
    cex_deposit_pct: float


# ============================================================================
# Alert Types
# ============================================================================


class AlertType(str, Enum):
    """Types of monitoring alerts."""

    UNKNOWN_EXIT_HIGH = "unknown_exit_high"  # UNKNOWN_EXIT > threshold
    DISTRIBUTION_SHIFT = "distribution_shift"  # >2σ shift in any class
    LP_AS_SELL = "lp_as_sell"  # LP action classified as DEX_SELL
    LOW_CONFIDENCE_PNL = "low_confidence_pnl"  # Low confidence produced PnL
    HIGH_UNKNOWN_RATE = "high_unknown_rate"  # Rolling unknown rate high
    CLASSIFIER_ERROR = "classifier_error"  # Exception during classification


@dataclass
class MonitoringAlert:
    """Alert raised by monitoring system."""

    alert_type: AlertType
    severity: str  # info, warning, critical
    timestamp: datetime
    message: str
    details: dict[str, Any]
    signature: str | None = None  # Related transaction if applicable

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/notification."""
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "details": self.details,
            "signature": self.signature,
        }


# ============================================================================
# Monitoring Configuration
# ============================================================================


@dataclass
class MonitoringConfig:
    """Configuration for exit classifier monitoring."""

    # Storage
    db_path: Path = Path("~/.shi/exit_classifier_monitoring.db").expanduser()
    log_retention_days: int = 90

    # Alert thresholds
    unknown_exit_threshold_pct: float = 20.0  # Alert if >20%
    distribution_shift_sigma: float = 2.0  # Alert if >2σ shift
    min_samples_for_drift: int = 100  # Minimum samples before drift detection
    low_confidence_pnl_threshold: float = 0.5  # Alert if PnL with confidence < this

    # Distribution snapshot interval
    snapshot_interval_minutes: int = 60

    # Alert handlers
    alert_webhook_url: str | None = None
    alert_log_file: Path | None = None


# ============================================================================
# Classification Logger
# ============================================================================


class ClassificationLogger:
    """
    Logs every exit classification with full evidence.

    HARD RULE: All classifications must be logged with full evidence.
    """

    def __init__(
        self,
        config: MonitoringConfig | None = None,
    ):
        self._config = config or MonitoringConfig()
        self._db: sqlite3.Connection | None = None
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Ensure database exists with correct schema."""
        self._config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._config.db_path))
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS classification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT NOT NULL,
                signature TEXT NOT NULL,
                slot INTEGER NOT NULL,
                token_mint TEXT NOT NULL,
                wallet_address TEXT NOT NULL,
                exit_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                sell_confidence REAL NOT NULL,
                pnl_computable INTEGER NOT NULL,
                program_ids TEXT NOT NULL,
                dex_detected TEXT,
                lp_program_detected TEXT,
                bridge_detected TEXT,
                destination_address TEXT,
                destination_type TEXT,
                display_mode TEXT NOT NULL,
                unknown_reason TEXT,
                evidence_json TEXT NOT NULL,
                confidence_factors TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_classification_logs_signature
            ON classification_logs(signature)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_classification_logs_exit_type
            ON classification_logs(exit_type)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_classification_logs_logged_at
            ON classification_logs(logged_at)
        """)
        self._db.commit()

    def log_classification(
        self,
        classification: ExitEventClassification,
        wallet_address: str,
        pnl_reliability_score: float | None = None,
    ) -> ClassificationLogRecord:
        """
        Log a classification with full evidence.

        Args:
            classification: The exit classification
            wallet_address: Source wallet address
            pnl_reliability_score: Optional reliability score for display mode

        Returns:
            The logged record
        """
        evidence = classification.evidence

        # Determine display mode
        if pnl_reliability_score is not None:
            if pnl_reliability_score >= 0.7:
                display_mode = "precise"
            elif pnl_reliability_score >= 0.4:
                display_mode = "range"
            else:
                display_mode = "unavailable"
        else:
            display_mode = "precise" if classification.pnl_computable else "unavailable"

        # Determine unknown reason
        unknown_reason = None
        if classification.exit_type == ExitEventType.UNKNOWN_EXIT:
            unknown_reason = classification.classification_reason

        # Serialize evidence
        evidence_dict = {
            "signature": evidence.signature,
            "slot": evidence.slot,
            "block_time": evidence.block_time.isoformat() if evidence.block_time else None,
            "token_mint": evidence.token_mint,
            "token_amount": evidence.token_amount,
            "token_decimals": evidence.token_decimals,
            "program_ids_detected": list(evidence.program_ids_detected),
            "dex_detected": evidence.dex_detected,
            "lp_program_detected": evidence.lp_program_detected,
            "bridge_detected": evidence.bridge_detected,
            "sol_change_lamports": evidence.sol_change_lamports,
            "has_quote_asset_received": evidence.has_quote_asset_received,
            "destination_address": evidence.destination_address,
            "destination_is_known_cex": evidence.destination_is_known_cex,
            "destination_cex_name": evidence.destination_cex_name,
            "destination_is_burn_address": evidence.destination_is_burn_address,
            "destination_is_high_fan_in": evidence.destination_is_high_fan_in,
            "lp_token_minted": evidence.lp_token_minted,
            "lp_token_burned": evidence.lp_token_burned,
            "destination_shares_funder": evidence.destination_shares_funder,
            "rapid_followup_detected": evidence.rapid_followup_detected,
        }

        record = ClassificationLogRecord(
            logged_at=datetime.now(timezone.utc),
            signature=evidence.signature,
            slot=evidence.slot,
            token_mint=evidence.token_mint,
            wallet_address=wallet_address,
            exit_type=classification.exit_type.value,
            confidence=classification.confidence,
            sell_confidence=classification.sell_confidence_score,
            pnl_computable=classification.pnl_computable,
            program_ids=evidence.program_ids_detected,
            dex_detected=evidence.dex_detected,
            lp_program_detected=evidence.lp_program_detected,
            bridge_detected=evidence.bridge_detected,
            destination_address=evidence.destination_address,
            destination_type=classification.downstream_wallet_type,
            display_mode=display_mode,
            unknown_reason=unknown_reason,
            evidence_json=json.dumps(evidence_dict),
            confidence_factors=classification.confidence_factors,
        )

        # Store in database
        if self._db:
            self._db.execute(
                """
                INSERT INTO classification_logs (
                    logged_at, signature, slot, token_mint, wallet_address,
                    exit_type, confidence, sell_confidence, pnl_computable,
                    program_ids, dex_detected, lp_program_detected, bridge_detected,
                    destination_address, destination_type, display_mode,
                    unknown_reason, evidence_json, confidence_factors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.logged_at.isoformat(),
                    record.signature,
                    record.slot,
                    record.token_mint,
                    record.wallet_address,
                    record.exit_type,
                    record.confidence,
                    record.sell_confidence,
                    int(record.pnl_computable),
                    json.dumps(list(record.program_ids)),
                    record.dex_detected,
                    record.lp_program_detected,
                    record.bridge_detected,
                    record.destination_address,
                    record.destination_type,
                    record.display_mode,
                    record.unknown_reason,
                    record.evidence_json,
                    json.dumps(list(record.confidence_factors)),
                ),
            )
            self._db.commit()

        logger.info(
            "exit_classification_logged",
            signature=record.signature[:16] + "...",
            exit_type=record.exit_type,
            confidence=round(record.confidence, 3),
            pnl_computable=record.pnl_computable,
        )

        return record

    def get_logs(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        exit_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get classification logs with optional filters."""
        if not self._db:
            return []

        query = "SELECT * FROM classification_logs WHERE 1=1"
        params: list[Any] = []

        if since:
            query += " AND logged_at >= ?"
            params.append(since.isoformat())
        if until:
            query += " AND logged_at <= ?"
            params.append(until.isoformat())
        if exit_type:
            query += " AND exit_type = ?"
            params.append(exit_type)

        query += " ORDER BY logged_at DESC LIMIT ?"
        params.append(limit)

        cursor = self._db.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.close()


# ============================================================================
# Distribution Dashboard
# ============================================================================


class DistributionDashboard:
    """
    Builds classification distribution dashboard metrics.

    Tracks:
    - % DEX_SELL
    - % TRANSFER_OUT
    - % LP_ADD / LP_REMOVE
    - % CEX_DEPOSIT
    - % UNKNOWN_EXIT
    - Median confidence by class
    """

    def __init__(
        self,
        logger: ClassificationLogger,
        config: MonitoringConfig | None = None,
    ):
        self._logger = logger
        self._config = config or MonitoringConfig()
        self._snapshots: list[DistributionSnapshot] = []

    def compute_distribution(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> ClassificationDistribution:
        """
        Compute classification distribution for a time window.

        Args:
            since: Start of window (default: 24h ago)
            until: End of window (default: now)

        Returns:
            ClassificationDistribution with counts and percentages
        """
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(hours=24)
        if until is None:
            until = datetime.now(timezone.utc)

        logs = self._logger.get_logs(since=since, until=until, limit=100000)

        # Initialize counts
        counts: dict[str, int] = defaultdict(int)
        confidences: dict[str, list[float]] = defaultdict(list)

        for log in logs:
            exit_type = log["exit_type"]
            counts[exit_type] += 1
            confidences[exit_type].append(log["confidence"])

        total = sum(counts.values())

        # Compute mean and median confidence
        mean_conf = {
            et: statistics.mean(confs) if confs else 0.0
            for et, confs in confidences.items()
        }
        median_conf = {
            et: statistics.median(confs) if confs else 0.0
            for et, confs in confidences.items()
        }

        distribution = ClassificationDistribution(
            counts=dict(counts),
            total=total,
            mean_confidence=mean_conf,
            median_confidence=median_conf,
            window_start=since,
            window_end=until,
        )
        distribution.compute_percentages()

        return distribution

    def take_snapshot(self) -> DistributionSnapshot:
        """Take a point-in-time snapshot for drift detection."""
        distribution = self.compute_distribution()

        snapshot = DistributionSnapshot(
            timestamp=datetime.now(timezone.utc),
            distribution=distribution,
            dex_sell_pct=distribution.percentages.get("dex_sell", 0.0),
            transfer_out_pct=distribution.percentages.get("transfer_out", 0.0),
            unknown_exit_pct=distribution.percentages.get("unknown_exit", 0.0),
            lp_action_pct=(
                distribution.percentages.get("lp_add", 0.0)
                + distribution.percentages.get("lp_remove", 0.0)
            ),
            cex_deposit_pct=distribution.percentages.get("cex_deposit", 0.0),
        )

        self._snapshots.append(snapshot)

        # Keep only last 1000 snapshots
        if len(self._snapshots) > 1000:
            self._snapshots = self._snapshots[-1000:]

        return snapshot

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get formatted data for dashboard display."""
        distribution = self.compute_distribution()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_classifications": distribution.total,
            "window": {
                "start": distribution.window_start.isoformat() if distribution.window_start else None,
                "end": distribution.window_end.isoformat() if distribution.window_end else None,
            },
            "by_type": {
                exit_type: {
                    "count": distribution.counts.get(exit_type, 0),
                    "percentage": round(distribution.percentages.get(exit_type, 0.0), 2),
                    "median_confidence": round(distribution.median_confidence.get(exit_type, 0.0), 3),
                }
                for exit_type in ExitEventType
            },
            "summary": {
                "dex_sell_pct": round(distribution.percentages.get("dex_sell", 0.0), 2),
                "transfer_out_pct": round(distribution.percentages.get("transfer_out", 0.0), 2),
                "unknown_exit_pct": round(distribution.percentages.get("unknown_exit", 0.0), 2),
                "lp_action_pct": round(
                    distribution.percentages.get("lp_add", 0.0)
                    + distribution.percentages.get("lp_remove", 0.0),
                    2,
                ),
                "cex_deposit_pct": round(distribution.percentages.get("cex_deposit", 0.0), 2),
            },
        }


# ============================================================================
# Alert Manager
# ============================================================================


class AlertManager:
    """
    Monitors for anomalies and raises alerts.

    Alerts on:
    - UNKNOWN_EXIT > 20%
    - DEX_SELL proportion shifts > 2σ
    - LP actions classified as sell
    - Low-confidence events produce PnL
    """

    def __init__(
        self,
        dashboard: DistributionDashboard,
        config: MonitoringConfig | None = None,
        alert_handlers: list[Callable[[MonitoringAlert], None]] | None = None,
    ):
        self._dashboard = dashboard
        self._config = config or MonitoringConfig()
        self._handlers = alert_handlers or []
        self._alerts: list[MonitoringAlert] = []
        self._baseline_distributions: list[DistributionSnapshot] = []

    def check_classification(
        self,
        classification: ExitEventClassification,
        pnl_was_computed: bool = False,
    ) -> list[MonitoringAlert]:
        """
        Check a single classification for alert conditions.

        Args:
            classification: The classification to check
            pnl_was_computed: Whether PnL was actually computed for this

        Returns:
            List of alerts raised (if any)
        """
        alerts = []

        # Check 1: LP classified as sell (hard rule violation)
        evidence = classification.evidence
        if classification.exit_type == ExitEventType.DEX_SELL:
            if evidence.lp_token_minted or evidence.lp_token_burned:
                alert = MonitoringAlert(
                    alert_type=AlertType.LP_AS_SELL,
                    severity="critical",
                    timestamp=datetime.now(timezone.utc),
                    message="LP action incorrectly classified as DEX_SELL",
                    details={
                        "exit_type": classification.exit_type.value,
                        "lp_token_minted": evidence.lp_token_minted,
                        "lp_token_burned": evidence.lp_token_burned,
                        "confidence": classification.confidence,
                    },
                    signature=evidence.signature,
                )
                alerts.append(alert)

        # Check 2: Low confidence PnL
        if pnl_was_computed and classification.sell_confidence_score < self._config.low_confidence_pnl_threshold:
            alert = MonitoringAlert(
                alert_type=AlertType.LOW_CONFIDENCE_PNL,
                severity="warning",
                timestamp=datetime.now(timezone.utc),
                message=f"PnL computed with low confidence ({classification.sell_confidence_score:.2f})",
                details={
                    "sell_confidence": classification.sell_confidence_score,
                    "threshold": self._config.low_confidence_pnl_threshold,
                    "exit_type": classification.exit_type.value,
                },
                signature=evidence.signature,
            )
            alerts.append(alert)

        # Dispatch alerts
        for alert in alerts:
            self._dispatch_alert(alert)

        return alerts

    def check_distribution(self) -> list[MonitoringAlert]:
        """
        Check current distribution for alert conditions.

        Returns:
            List of alerts raised (if any)
        """
        alerts = []
        snapshot = self._dashboard.take_snapshot()

        # Check 1: Unknown exit rate too high
        if snapshot.unknown_exit_pct > self._config.unknown_exit_threshold_pct:
            alert = MonitoringAlert(
                alert_type=AlertType.UNKNOWN_EXIT_HIGH,
                severity="warning",
                timestamp=datetime.now(timezone.utc),
                message=f"UNKNOWN_EXIT rate is {snapshot.unknown_exit_pct:.1f}% (threshold: {self._config.unknown_exit_threshold_pct}%)",
                details={
                    "unknown_exit_pct": snapshot.unknown_exit_pct,
                    "threshold_pct": self._config.unknown_exit_threshold_pct,
                    "total_classifications": snapshot.distribution.total,
                },
            )
            alerts.append(alert)

        # Check 2: Distribution shift
        if len(self._baseline_distributions) >= self._config.min_samples_for_drift:
            shift_alerts = self._check_distribution_shift(snapshot)
            alerts.extend(shift_alerts)
        else:
            # Build baseline
            self._baseline_distributions.append(snapshot)

        # Dispatch alerts
        for alert in alerts:
            self._dispatch_alert(alert)

        return alerts

    def _check_distribution_shift(
        self,
        current: DistributionSnapshot,
    ) -> list[MonitoringAlert]:
        """Check for significant distribution shifts."""
        alerts = []

        # Compute baseline statistics
        dex_sell_baseline = [s.dex_sell_pct for s in self._baseline_distributions]
        if len(dex_sell_baseline) < 2:
            return alerts

        mean_dex = statistics.mean(dex_sell_baseline)
        stdev_dex = statistics.stdev(dex_sell_baseline) if len(dex_sell_baseline) > 1 else 0

        # Check DEX_SELL shift
        if stdev_dex > 0:
            z_score = (current.dex_sell_pct - mean_dex) / stdev_dex
            if abs(z_score) > self._config.distribution_shift_sigma:
                alert = MonitoringAlert(
                    alert_type=AlertType.DISTRIBUTION_SHIFT,
                    severity="warning",
                    timestamp=datetime.now(timezone.utc),
                    message=f"DEX_SELL distribution shifted {z_score:.1f}σ from baseline",
                    details={
                        "current_pct": current.dex_sell_pct,
                        "baseline_mean": mean_dex,
                        "baseline_stdev": stdev_dex,
                        "z_score": z_score,
                        "threshold_sigma": self._config.distribution_shift_sigma,
                    },
                )
                alerts.append(alert)

        return alerts

    def _dispatch_alert(self, alert: MonitoringAlert) -> None:
        """Dispatch alert to all handlers."""
        self._alerts.append(alert)

        logger.warning(
            "monitoring_alert",
            alert_type=alert.alert_type.value,
            severity=alert.severity,
            message=alert.message,
        )

        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error("alert_handler_error", error=str(e))

    def get_recent_alerts(
        self,
        since: datetime | None = None,
        severity: str | None = None,
        limit: int = 100,
    ) -> list[MonitoringAlert]:
        """Get recent alerts with optional filters."""
        alerts = self._alerts

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]


# ============================================================================
# Monitored Classifier Wrapper
# ============================================================================


class MonitoredExitClassifier:
    """
    Wrapper that adds monitoring to the exit classifier.

    Automatically logs all classifications and checks for alerts.
    """

    def __init__(
        self,
        classifier: "ExitEventClassifier",  # noqa: F821
        config: MonitoringConfig | None = None,
    ):
        from .exit_classifier import ExitEventClassifier

        self._classifier: ExitEventClassifier = classifier
        self._config = config or MonitoringConfig()
        self._logger = ClassificationLogger(self._config)
        self._dashboard = DistributionDashboard(self._logger, self._config)
        self._alert_manager = AlertManager(self._dashboard, self._config)
        self._classification_count = 0

    def classify(
        self,
        wallet_address: str,
        token_mint: str,
        token_amount: int,
        token_decimals: int,
        tx_data: dict[str, Any],
        related_wallet_info: dict[str, Any] | None = None,
    ) -> ExitEventClassification:
        """
        Classify with full monitoring.

        Args:
            wallet_address: Source wallet address
            token_mint: Token mint being exited
            token_amount: Raw token amount
            token_decimals: Token decimals
            tx_data: Full transaction data
            related_wallet_info: Optional related wallet info

        Returns:
            Classification result (also logged and checked for alerts)
        """
        # Run classification
        classification = self._classifier.classify(
            wallet_address=wallet_address,
            token_mint=token_mint,
            token_amount=token_amount,
            token_decimals=token_decimals,
            tx_data=tx_data,
            related_wallet_info=related_wallet_info,
        )

        # Log the classification
        self._logger.log_classification(classification, wallet_address)

        # Check for per-classification alerts
        self._alert_manager.check_classification(
            classification,
            pnl_was_computed=classification.pnl_computable,
        )

        # Periodic distribution check
        self._classification_count += 1
        if self._classification_count % 100 == 0:
            self._alert_manager.check_distribution()

        return classification

    @property
    def dashboard(self) -> DistributionDashboard:
        """Get the distribution dashboard."""
        return self._dashboard

    @property
    def alert_manager(self) -> AlertManager:
        """Get the alert manager."""
        return self._alert_manager

    @property
    def logger(self) -> ClassificationLogger:
        """Get the classification logger."""
        return self._logger


# ============================================================================
# Factory Functions
# ============================================================================


def create_monitored_classifier(
    classifier: "ExitEventClassifier",  # noqa: F821
    config: MonitoringConfig | None = None,
) -> MonitoredExitClassifier:
    """Create a monitored exit classifier."""
    return MonitoredExitClassifier(classifier, config)


def create_monitoring_config(
    db_path: str | Path | None = None,
    unknown_threshold_pct: float = 20.0,
    distribution_shift_sigma: float = 2.0,
) -> MonitoringConfig:
    """Create a monitoring configuration."""
    config = MonitoringConfig(
        unknown_exit_threshold_pct=unknown_threshold_pct,
        distribution_shift_sigma=distribution_shift_sigma,
    )
    if db_path:
        config.db_path = Path(db_path).expanduser()
    return config
