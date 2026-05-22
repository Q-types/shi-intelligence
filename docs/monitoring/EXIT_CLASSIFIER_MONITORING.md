# Exit Classifier Monitoring Specification

**Version**: 1.0.0
**Date**: 2026-05-22
**Sprint**: 9.5 - Post-Deployment Validation

## Overview

This document specifies the production monitoring infrastructure for the exit classifier. The classifier must be deployed in **monitored mode** before using its outputs as ground truth for model training.

## Hard Rules

1. **All classifications must be logged** - No silent classification
2. **Unknown exits must be logged, not hidden** - Visibility into unclassified events
3. **Classifier errors must be reviewable** - Full evidence preserved
4. **Low-confidence exits cannot produce precise PnL** - Reliability gating
5. **Real-world validation required before training use** - No premature trust

## Architecture

```
┌─────────────────┐
│  Exit Event     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ MonitoredExit   │────▶│ Classification  │
│ Classifier      │     │ Logger          │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │ SQLite DB       │
         │              │ (Full Evidence) │
         │              └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Alert Manager   │◀────│ Distribution    │
│                 │     │ Dashboard       │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Alert Handlers  │
│ (Webhook, Log)  │
└─────────────────┘
```

## Classification Log Record

Every classification is logged with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `logged_at` | datetime | Timestamp of logging |
| `signature` | string | Transaction signature |
| `slot` | int | Solana slot number |
| `token_mint` | string | Token being exited |
| `wallet_address` | string | Source wallet |
| `exit_type` | string | Classifier's determination |
| `confidence` | float | Classification confidence (0-1) |
| `sell_confidence` | float | Specific sell confidence (0-1) |
| `pnl_computable` | bool | Whether PnL can be computed |
| `program_ids` | list[str] | Programs in transaction |
| `dex_detected` | string | DEX program if detected |
| `lp_program_detected` | string | LP program if detected |
| `bridge_detected` | string | Bridge program if detected |
| `destination_address` | string | Token destination |
| `destination_type` | string | cex/wallet/pool/burn/unknown |
| `display_mode` | string | precise/range/unavailable |
| `unknown_reason` | string | Why UNKNOWN_EXIT (if applicable) |
| `evidence_json` | string | Full serialized evidence |
| `confidence_factors` | list[str] | Factors contributing to confidence |

## Distribution Dashboard

### Metrics Tracked

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| DEX_SELL % | Percentage classified as DEX sells | Shift > 2σ |
| TRANSFER_OUT % | Percentage classified as transfers | Shift > 2σ |
| LP_ADD % | Percentage classified as LP add | - |
| LP_REMOVE % | Percentage classified as LP remove | - |
| CEX_DEPOSIT % | Percentage classified as CEX deposit | - |
| UNKNOWN_EXIT % | Percentage unknown | > 20% |
| Median Confidence | Per-class median confidence | - |

### Snapshot Schedule

Distribution snapshots taken every **60 minutes** for drift detection.

### Dashboard Data Format

```json
{
  "timestamp": "2026-05-22T10:00:00Z",
  "total_classifications": 1523,
  "window": {
    "start": "2026-05-21T10:00:00Z",
    "end": "2026-05-22T10:00:00Z"
  },
  "by_type": {
    "dex_sell": {
      "count": 892,
      "percentage": 58.57,
      "median_confidence": 0.92
    },
    "transfer_out": {
      "count": 312,
      "percentage": 20.49,
      "median_confidence": 0.71
    }
  },
  "summary": {
    "dex_sell_pct": 58.57,
    "transfer_out_pct": 20.49,
    "unknown_exit_pct": 3.21,
    "lp_action_pct": 8.14,
    "cex_deposit_pct": 5.32
  }
}
```

## Alert System

### Alert Types

| Alert Type | Severity | Trigger Condition |
|------------|----------|-------------------|
| `UNKNOWN_EXIT_HIGH` | warning | UNKNOWN_EXIT > 20% of classifications |
| `DISTRIBUTION_SHIFT` | warning | Any class shifts > 2σ from baseline |
| `LP_AS_SELL` | critical | LP action classified as DEX_SELL |
| `LOW_CONFIDENCE_PNL` | warning | PnL computed with confidence < 0.5 |
| `HIGH_UNKNOWN_RATE` | warning | Rolling unknown rate exceeds threshold |
| `CLASSIFIER_ERROR` | critical | Exception during classification |

### Alert Record Format

```json
{
  "alert_type": "unknown_exit_high",
  "severity": "warning",
  "timestamp": "2026-05-22T10:15:00Z",
  "message": "UNKNOWN_EXIT rate is 24.3% (threshold: 20.0%)",
  "details": {
    "unknown_exit_pct": 24.3,
    "threshold_pct": 20.0,
    "total_classifications": 500
  },
  "signature": null
}
```

### Alert Handlers

Alerts can be dispatched to:

1. **Structured logging** (always enabled)
2. **Webhook URL** (optional, for Slack/Discord integration)
3. **Alert log file** (optional, for offline analysis)

## Configuration

```python
@dataclass
class MonitoringConfig:
    # Storage
    db_path: Path = Path("~/.shi/exit_classifier_monitoring.db")
    log_retention_days: int = 90

    # Alert thresholds
    unknown_exit_threshold_pct: float = 20.0
    distribution_shift_sigma: float = 2.0
    min_samples_for_drift: int = 100
    low_confidence_pnl_threshold: float = 0.5

    # Distribution snapshot interval
    snapshot_interval_minutes: int = 60

    # Alert handlers
    alert_webhook_url: str | None = None
    alert_log_file: Path | None = None
```

## Usage

### Basic Monitoring

```python
from src.longitudinal import (
    create_exit_classifier,
    create_monitored_classifier,
    create_monitoring_config,
)

# Create base classifier
classifier = create_exit_classifier()

# Wrap with monitoring
config = create_monitoring_config(
    unknown_threshold_pct=20.0,
    distribution_shift_sigma=2.0,
)
monitored = create_monitored_classifier(classifier, config)

# Use normally - all classifications are logged
classification = monitored.classify(
    wallet_address=wallet,
    token_mint=mint,
    token_amount=amount,
    token_decimals=decimals,
    tx_data=tx_data,
)

# Access dashboard
dashboard_data = monitored.dashboard.get_dashboard_data()

# Check alerts
alerts = monitored.alert_manager.get_recent_alerts(
    since=datetime.now() - timedelta(hours=1),
    severity="critical",
)
```

### Custom Alert Handler

```python
def slack_alert_handler(alert: MonitoringAlert):
    if alert.severity == "critical":
        send_slack_message(
            channel="#shi-alerts",
            text=f":rotating_light: {alert.message}",
        )

alert_manager = AlertManager(
    dashboard=dashboard,
    config=config,
    alert_handlers=[slack_alert_handler],
)
```

## Drift Detection

Distribution drift is detected using z-score analysis:

```
z = (current_pct - baseline_mean) / baseline_stdev
```

If `|z| > 2.0` (configurable), a DISTRIBUTION_SHIFT alert is raised.

### Baseline Building

- Minimum 100 samples before drift detection begins
- Baseline updated with rolling window of 1000 snapshots
- Initial deployment period used to establish baseline

## Integration Points

### With Event Pipeline

```python
async def process_exit_event(event: ExitEvent):
    classification = monitored_classifier.classify(...)

    # Classification is automatically logged
    # Alerts are automatically checked

    if classification.pnl_computable:
        pnl = pnl_calculator.compute(...)
```

### With Validation Dataset

```python
from src.longitudinal import (
    create_validation_dataset_builder,
    create_sample_collector,
)

dataset_builder = create_validation_dataset_builder()
collector = create_sample_collector(dataset_builder, sampling_rate=0.1)

# Collect samples during classification
collector.maybe_collect(
    signature=evidence.signature,
    token_mint=evidence.token_mint,
    wallet_address=wallet,
    classifier_exit_type=classification.exit_type.value,
    classifier_confidence=classification.confidence,
    classifier_evidence_json=evidence_json,
)
```

## Data Retention

- Classification logs retained for **90 days** (configurable)
- Distribution snapshots retained for **1000 snapshots** (~42 days at hourly)
- Alerts retained in memory (export to file for persistence)

## Next Steps

1. Deploy monitored classifier in production
2. Establish baseline distribution (first 1000+ classifications)
3. Monitor alert rate and tune thresholds
4. Build validation dataset (see EXIT_CLASSIFIER_VALIDATION_DATASET.md)
5. Compute accuracy metrics before using for training
