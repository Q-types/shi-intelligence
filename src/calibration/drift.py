"""
Drift Detection.

Detects model and data drift for triggering retraining.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Sequence
from collections import deque

import numpy as np
import structlog

logger = structlog.get_logger()


class DriftSeverity(Enum):
    """Severity levels for drift alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Alert for detected drift."""

    alert_id: str
    drift_type: str  # "prediction", "feature", "coefficient"
    severity: DriftSeverity
    metric_name: str
    baseline_value: float
    current_value: float
    z_score: float
    detected_at: datetime
    recommendation: str
    metadata: dict = field(default_factory=dict)


@dataclass
class DriftWindow:
    """Statistics for a drift detection window."""

    start_time: datetime
    end_time: datetime
    mean: float
    std: float
    num_samples: int


class DriftDetector:
    """
    Detects drift in model predictions and features.

    Uses statistical tests to identify when model
    behavior has shifted from baseline.
    """

    # Z-score thresholds for severity levels
    SEVERITY_THRESHOLDS = {
        DriftSeverity.LOW: 1.5,
        DriftSeverity.MEDIUM: 2.0,
        DriftSeverity.HIGH: 2.5,
        DriftSeverity.CRITICAL: 3.0,
    }

    def __init__(
        self,
        baseline_window_days: int = 30,
        detection_window_days: int = 1,
        min_samples: int = 100,
    ):
        self.baseline_window_days = baseline_window_days
        self.detection_window_days = detection_window_days
        self.min_samples = min_samples

        self._metric_history: dict[str, deque[tuple[datetime, float]]] = {}
        self._baselines: dict[str, DriftWindow] = {}
        self._alerts: list[DriftAlert] = []
        self._alert_counter = 0

    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: datetime | None = None,
    ) -> DriftAlert | None:
        """
        Record a metric value and check for drift.

        Returns:
            DriftAlert if drift detected, None otherwise
        """
        timestamp = timestamp or datetime.now(timezone.utc)

        # Initialize history if needed
        if metric_name not in self._metric_history:
            self._metric_history[metric_name] = deque(maxlen=10000)

        self._metric_history[metric_name].append((timestamp, value))

        # Update baseline if needed
        self._update_baseline(metric_name)

        # Check for drift
        return self._check_drift(metric_name, value, timestamp)

    def _update_baseline(self, metric_name: str) -> None:
        """Update baseline statistics for metric."""
        history = self._metric_history[metric_name]

        if len(history) < self.min_samples:
            return

        now = datetime.now(timezone.utc)
        baseline_cutoff = now - timedelta(days=self.baseline_window_days)
        detection_cutoff = now - timedelta(days=self.detection_window_days)

        # Use data older than detection window for baseline
        baseline_values = [
            v for t, v in history
            if baseline_cutoff <= t < detection_cutoff
        ]

        if len(baseline_values) < self.min_samples // 2:
            return

        self._baselines[metric_name] = DriftWindow(
            start_time=baseline_cutoff,
            end_time=detection_cutoff,
            mean=float(np.mean(baseline_values)),
            std=float(np.std(baseline_values)),
            num_samples=len(baseline_values),
        )

    def _check_drift(
        self,
        metric_name: str,
        value: float,
        timestamp: datetime,
    ) -> DriftAlert | None:
        """Check if current value indicates drift."""
        baseline = self._baselines.get(metric_name)

        if baseline is None or baseline.std == 0:
            return None

        # Compute z-score
        z_score = abs(value - baseline.mean) / baseline.std

        # Determine severity
        severity = None
        for sev in reversed(list(DriftSeverity)):
            if z_score >= self.SEVERITY_THRESHOLDS[sev]:
                severity = sev
                break

        if severity is None:
            return None

        # Create alert
        self._alert_counter += 1
        alert = DriftAlert(
            alert_id=f"drift_{self._alert_counter:06d}",
            drift_type="metric",
            severity=severity,
            metric_name=metric_name,
            baseline_value=baseline.mean,
            current_value=value,
            z_score=z_score,
            detected_at=timestamp,
            recommendation=self._get_recommendation(severity),
            metadata={
                "baseline_std": baseline.std,
                "baseline_samples": baseline.num_samples,
            },
        )

        self._alerts.append(alert)

        logger.warning(
            "drift_detected",
            metric=metric_name,
            severity=severity.value,
            z_score=z_score,
            baseline=baseline.mean,
            current=value,
        )

        return alert

    def _get_recommendation(self, severity: DriftSeverity) -> str:
        """Get recommendation based on severity."""
        recommendations = {
            DriftSeverity.LOW: "Monitor closely. Consider investigation if persists.",
            DriftSeverity.MEDIUM: "Investigate cause. May need model refresh.",
            DriftSeverity.HIGH: "Immediate investigation required. Schedule retraining.",
            DriftSeverity.CRITICAL: "Critical drift. Halt predictions and retrain immediately.",
        }
        return recommendations[severity]

    def detect_coefficient_drift(
        self,
        baseline_coefficients: dict[str, float],
        current_coefficients: dict[str, float],
        threshold: float = 0.5,
    ) -> list[DriftAlert]:
        """
        Detect drift in model coefficients.

        Per INITIAL_PROMPT: Coefficient stability checks required.
        """
        alerts = []

        for name, baseline_val in baseline_coefficients.items():
            if name not in current_coefficients:
                continue

            current_val = current_coefficients[name]

            # Compute relative change
            if abs(baseline_val) > 1e-6:
                relative_change = abs(current_val - baseline_val) / abs(baseline_val)
            else:
                relative_change = abs(current_val - baseline_val)

            if relative_change > threshold:
                severity = DriftSeverity.MEDIUM
                if relative_change > threshold * 2:
                    severity = DriftSeverity.HIGH
                if relative_change > threshold * 3:
                    severity = DriftSeverity.CRITICAL

                self._alert_counter += 1
                alert = DriftAlert(
                    alert_id=f"coef_drift_{self._alert_counter:06d}",
                    drift_type="coefficient",
                    severity=severity,
                    metric_name=f"coefficient_{name}",
                    baseline_value=baseline_val,
                    current_value=current_val,
                    z_score=relative_change / threshold,
                    detected_at=datetime.now(timezone.utc),
                    recommendation="Coefficient has drifted significantly. Retrain model.",
                )

                alerts.append(alert)
                self._alerts.append(alert)

        return alerts

    def detect_prediction_drift(
        self,
        baseline_predictions: Sequence[float],
        current_predictions: Sequence[float],
    ) -> DriftAlert | None:
        """
        Detect drift in prediction distribution.

        Uses Kolmogorov-Smirnov test for distribution comparison.
        """
        if len(baseline_predictions) < 30 or len(current_predictions) < 30:
            return None

        from scipy import stats

        # KS test
        ks_stat, p_value = stats.ks_2samp(
            baseline_predictions,
            current_predictions,
        )

        if p_value < 0.01:
            severity = DriftSeverity.CRITICAL
        elif p_value < 0.05:
            severity = DriftSeverity.HIGH
        elif p_value < 0.1:
            severity = DriftSeverity.MEDIUM
        else:
            return None

        self._alert_counter += 1
        alert = DriftAlert(
            alert_id=f"pred_drift_{self._alert_counter:06d}",
            drift_type="prediction",
            severity=severity,
            metric_name="prediction_distribution",
            baseline_value=float(np.mean(baseline_predictions)),
            current_value=float(np.mean(current_predictions)),
            z_score=ks_stat * 10,  # Scale for visibility
            detected_at=datetime.now(timezone.utc),
            recommendation="Prediction distribution has shifted. Investigate data or retrain.",
            metadata={
                "ks_statistic": ks_stat,
                "p_value": p_value,
            },
        )

        self._alerts.append(alert)
        return alert

    def get_recent_alerts(
        self,
        hours: int = 24,
        min_severity: DriftSeverity = DriftSeverity.LOW,
    ) -> list[DriftAlert]:
        """Get recent drift alerts."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        severity_order = list(DriftSeverity)
        min_idx = severity_order.index(min_severity)

        return [
            a for a in self._alerts
            if a.detected_at >= cutoff
            and severity_order.index(a.severity) >= min_idx
        ]

    def get_summary(self) -> dict:
        """Get drift detection summary."""
        recent_alerts = self.get_recent_alerts(hours=24)

        return {
            "total_alerts": len(self._alerts),
            "alerts_24h": len(recent_alerts),
            "critical_alerts": sum(
                1 for a in recent_alerts if a.severity == DriftSeverity.CRITICAL
            ),
            "metrics_tracked": len(self._metric_history),
            "baselines_computed": len(self._baselines),
        }
