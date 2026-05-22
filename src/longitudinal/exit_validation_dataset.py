"""
Exit Classifier Validation Dataset Builder (Sprint 9.5).

Builds labelled validation datasets for measuring classifier accuracy:
- Precision and recall by exit type
- False sell rate
- False transfer rate
- Unknown exit analysis

HARD RULE: Real-world validation required before using exit classes as training labels.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from .exit_classifier import ExitEventType

logger = structlog.get_logger()


# ============================================================================
# Validation Labels
# ============================================================================


class ValidationLabel(str, Enum):
    """Human-verified labels for validation."""

    TRUE_DEX_SELL = "true_dex_sell"
    TRUE_TRANSFER = "true_transfer"
    TRUE_LP_ADD = "true_lp_add"
    TRUE_LP_REMOVE = "true_lp_remove"
    TRUE_CEX_DEPOSIT = "true_cex_deposit"
    TRUE_BURN = "true_burn"
    TRUE_BRIDGE = "true_bridge"
    TRUE_MIGRATION = "true_migration"
    TRUE_PROGRAM_INTERACTION = "true_program_interaction"
    AMBIGUOUS = "ambiguous"  # Cannot determine even with manual review
    NEEDS_MORE_CONTEXT = "needs_more_context"


class ValidationStatus(str, Enum):
    """Status of a validation sample."""

    PENDING = "pending"  # Awaiting human review
    LABELLED = "labelled"  # Human label assigned
    VERIFIED = "verified"  # Label verified by second reviewer
    DISPUTED = "disputed"  # Disagreement between reviewers


# ============================================================================
# Validation Sample
# ============================================================================


@dataclass
class ValidationSample:
    """A sample in the validation dataset."""

    # Identity
    sample_id: str
    signature: str
    token_mint: str
    wallet_address: str

    # Classifier output
    classifier_exit_type: str
    classifier_confidence: float
    classifier_evidence_json: str

    # Human validation
    status: ValidationStatus = ValidationStatus.PENDING
    human_label: ValidationLabel | None = None
    reviewer_id: str | None = None
    review_timestamp: datetime | None = None
    review_notes: str | None = None

    # Second review (for verified status)
    second_reviewer_id: str | None = None
    second_review_timestamp: datetime | None = None
    second_label: ValidationLabel | None = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_correct(self) -> bool | None:
        """Check if classifier was correct (None if not labelled)."""
        if self.human_label is None:
            return None
        if self.human_label in (ValidationLabel.AMBIGUOUS, ValidationLabel.NEEDS_MORE_CONTEXT):
            return None

        # Map human labels to exit types
        label_to_exit_type = {
            ValidationLabel.TRUE_DEX_SELL: ExitEventType.DEX_SELL.value,
            ValidationLabel.TRUE_TRANSFER: ExitEventType.TRANSFER_OUT.value,
            ValidationLabel.TRUE_LP_ADD: ExitEventType.LP_ADD.value,
            ValidationLabel.TRUE_LP_REMOVE: ExitEventType.LP_REMOVE.value,
            ValidationLabel.TRUE_CEX_DEPOSIT: ExitEventType.CEX_DEPOSIT.value,
            ValidationLabel.TRUE_BURN: ExitEventType.BURN.value,
            ValidationLabel.TRUE_BRIDGE: ExitEventType.BRIDGE.value,
            ValidationLabel.TRUE_MIGRATION: ExitEventType.WALLET_MIGRATION.value,
            ValidationLabel.TRUE_PROGRAM_INTERACTION: ExitEventType.PROGRAM_INTERACTION.value,
        }

        expected_exit_type = label_to_exit_type.get(self.human_label)
        return self.classifier_exit_type == expected_exit_type

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "sample_id": self.sample_id,
            "signature": self.signature,
            "token_mint": self.token_mint,
            "wallet_address": self.wallet_address,
            "classifier_exit_type": self.classifier_exit_type,
            "classifier_confidence": self.classifier_confidence,
            "classifier_evidence_json": self.classifier_evidence_json,
            "status": self.status.value,
            "human_label": self.human_label.value if self.human_label else None,
            "reviewer_id": self.reviewer_id,
            "review_timestamp": self.review_timestamp.isoformat() if self.review_timestamp else None,
            "review_notes": self.review_notes,
            "second_reviewer_id": self.second_reviewer_id,
            "second_review_timestamp": self.second_review_timestamp.isoformat() if self.second_review_timestamp else None,
            "second_label": self.second_label.value if self.second_label else None,
            "created_at": self.created_at.isoformat(),
        }


# ============================================================================
# Dataset Targets
# ============================================================================


@dataclass
class DatasetTargets:
    """Target sample counts by exit type."""

    dex_sells: int = 50
    transfers: int = 50
    lp_actions: int = 20  # LP_ADD + LP_REMOVE combined
    cex_deposits: int = 20
    unknown_exits: int = 20

    def total(self) -> int:
        """Total target samples."""
        return self.dex_sells + self.transfers + self.lp_actions + self.cex_deposits + self.unknown_exits


@dataclass
class DatasetProgress:
    """Progress towards dataset targets."""

    targets: DatasetTargets
    collected: dict[str, int]
    labelled: dict[str, int]
    verified: dict[str, int]

    def completion_pct(self) -> float:
        """Overall completion percentage."""
        total_labelled = sum(self.labelled.values())
        return (total_labelled / self.targets.total()) * 100 if self.targets.total() > 0 else 0

    def needs_more(self, exit_type: str) -> int:
        """How many more samples needed for this exit type."""
        target_map = {
            "dex_sell": self.targets.dex_sells,
            "transfer_out": self.targets.transfers,
            "lp_add": self.targets.lp_actions // 2,
            "lp_remove": self.targets.lp_actions // 2,
            "cex_deposit": self.targets.cex_deposits,
            "unknown_exit": self.targets.unknown_exits,
        }
        target = target_map.get(exit_type, 0)
        collected = self.collected.get(exit_type, 0)
        return max(0, target - collected)


# ============================================================================
# Validation Dataset Builder
# ============================================================================


class ValidationDatasetBuilder:
    """
    Builds labelled validation datasets from classification logs.

    Target dataset:
    - 50 DEX sells
    - 50 transfers
    - 20 LP actions
    - 20 CEX deposits
    - 20 unknown exits

    Total: 160 labelled samples
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        targets: DatasetTargets | None = None,
    ):
        self._db_path = Path(db_path or "~/.shi/exit_validation_dataset.db").expanduser()
        self._targets = targets or DatasetTargets()
        self._db: sqlite3.Connection | None = None
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Ensure database exists with correct schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._db_path))
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS validation_samples (
                sample_id TEXT PRIMARY KEY,
                signature TEXT NOT NULL,
                token_mint TEXT NOT NULL,
                wallet_address TEXT NOT NULL,
                classifier_exit_type TEXT NOT NULL,
                classifier_confidence REAL NOT NULL,
                classifier_evidence_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                human_label TEXT,
                reviewer_id TEXT,
                review_timestamp TEXT,
                review_notes TEXT,
                second_reviewer_id TEXT,
                second_review_timestamp TEXT,
                second_label TEXT,
                created_at TEXT NOT NULL
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_validation_samples_exit_type
            ON validation_samples(classifier_exit_type)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_validation_samples_status
            ON validation_samples(status)
        """)
        self._db.commit()

    def add_sample(
        self,
        signature: str,
        token_mint: str,
        wallet_address: str,
        classifier_exit_type: str,
        classifier_confidence: float,
        classifier_evidence_json: str,
    ) -> ValidationSample | None:
        """
        Add a sample to the validation dataset.

        Returns None if dataset target for this type is already met.
        """
        # Check if we need more of this type
        progress = self.get_progress()
        if progress.needs_more(classifier_exit_type) <= 0:
            logger.debug(
                "validation_sample_skipped",
                exit_type=classifier_exit_type,
                reason="target_met",
            )
            return None

        # Check for duplicate
        if self._sample_exists(signature):
            return None

        # Create sample
        import uuid
        sample_id = f"val_{uuid.uuid4().hex[:12]}"

        sample = ValidationSample(
            sample_id=sample_id,
            signature=signature,
            token_mint=token_mint,
            wallet_address=wallet_address,
            classifier_exit_type=classifier_exit_type,
            classifier_confidence=classifier_confidence,
            classifier_evidence_json=classifier_evidence_json,
        )

        # Store in database
        if self._db:
            self._db.execute(
                """
                INSERT INTO validation_samples (
                    sample_id, signature, token_mint, wallet_address,
                    classifier_exit_type, classifier_confidence, classifier_evidence_json,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample.sample_id,
                    sample.signature,
                    sample.token_mint,
                    sample.wallet_address,
                    sample.classifier_exit_type,
                    sample.classifier_confidence,
                    sample.classifier_evidence_json,
                    sample.status.value,
                    sample.created_at.isoformat(),
                ),
            )
            self._db.commit()

        logger.info(
            "validation_sample_added",
            sample_id=sample_id,
            exit_type=classifier_exit_type,
        )

        return sample

    def label_sample(
        self,
        sample_id: str,
        label: ValidationLabel,
        reviewer_id: str,
        notes: str | None = None,
    ) -> bool:
        """
        Add human label to a sample.

        Returns True if successful.
        """
        if not self._db:
            return False

        now = datetime.now(timezone.utc)

        self._db.execute(
            """
            UPDATE validation_samples
            SET human_label = ?,
                reviewer_id = ?,
                review_timestamp = ?,
                review_notes = ?,
                status = ?
            WHERE sample_id = ?
            """,
            (
                label.value,
                reviewer_id,
                now.isoformat(),
                notes,
                ValidationStatus.LABELLED.value,
                sample_id,
            ),
        )
        self._db.commit()

        logger.info(
            "validation_sample_labelled",
            sample_id=sample_id,
            label=label.value,
            reviewer=reviewer_id,
        )

        return True

    def verify_sample(
        self,
        sample_id: str,
        label: ValidationLabel,
        reviewer_id: str,
    ) -> bool:
        """
        Add second verification to a sample.

        Returns True if successful.
        """
        if not self._db:
            return False

        # Get current sample
        sample = self.get_sample(sample_id)
        if not sample or sample.status != ValidationStatus.LABELLED:
            return False

        now = datetime.now(timezone.utc)

        # Check if labels match
        if sample.human_label == label:
            new_status = ValidationStatus.VERIFIED
        else:
            new_status = ValidationStatus.DISPUTED

        self._db.execute(
            """
            UPDATE validation_samples
            SET second_reviewer_id = ?,
                second_review_timestamp = ?,
                second_label = ?,
                status = ?
            WHERE sample_id = ?
            """,
            (
                reviewer_id,
                now.isoformat(),
                label.value,
                new_status.value,
                sample_id,
            ),
        )
        self._db.commit()

        logger.info(
            "validation_sample_verified",
            sample_id=sample_id,
            status=new_status.value,
        )

        return True

    def get_sample(self, sample_id: str) -> ValidationSample | None:
        """Get a sample by ID."""
        if not self._db:
            return None

        cursor = self._db.execute(
            "SELECT * FROM validation_samples WHERE sample_id = ?",
            (sample_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        columns = [desc[0] for desc in cursor.description]
        data = dict(zip(columns, row))
        return self._row_to_sample(data)

    def get_pending_samples(
        self,
        exit_type: str | None = None,
        limit: int = 10,
    ) -> list[ValidationSample]:
        """Get samples awaiting human review."""
        if not self._db:
            return []

        query = "SELECT * FROM validation_samples WHERE status = ?"
        params: list[Any] = [ValidationStatus.PENDING.value]

        if exit_type:
            query += " AND classifier_exit_type = ?"
            params.append(exit_type)

        query += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)

        cursor = self._db.execute(query, params)
        columns = [desc[0] for desc in cursor.description]

        samples = []
        for row in cursor.fetchall():
            data = dict(zip(columns, row))
            sample = self._row_to_sample(data)
            if sample:
                samples.append(sample)

        return samples

    def get_progress(self) -> DatasetProgress:
        """Get current progress towards dataset targets."""
        if not self._db:
            return DatasetProgress(
                targets=self._targets,
                collected={},
                labelled={},
                verified={},
            )

        # Count by exit type and status
        cursor = self._db.execute("""
            SELECT classifier_exit_type, status, COUNT(*) as count
            FROM validation_samples
            GROUP BY classifier_exit_type, status
        """)

        collected: dict[str, int] = {}
        labelled: dict[str, int] = {}
        verified: dict[str, int] = {}

        for row in cursor.fetchall():
            exit_type, status, count = row
            collected[exit_type] = collected.get(exit_type, 0) + count

            if status in (ValidationStatus.LABELLED.value, ValidationStatus.VERIFIED.value):
                labelled[exit_type] = labelled.get(exit_type, 0) + count

            if status == ValidationStatus.VERIFIED.value:
                verified[exit_type] = verified.get(exit_type, 0) + count

        return DatasetProgress(
            targets=self._targets,
            collected=collected,
            labelled=labelled,
            verified=verified,
        )

    def compute_accuracy_metrics(self) -> dict[str, Any]:
        """
        Compute accuracy metrics from labelled samples.

        Returns metrics including:
        - Precision by exit type
        - Recall by exit type
        - False sell rate
        - False transfer rate
        - Unknown exit analysis
        """
        if not self._db:
            return {}

        # Get all labelled samples
        cursor = self._db.execute("""
            SELECT classifier_exit_type, human_label
            FROM validation_samples
            WHERE status IN ('labelled', 'verified')
            AND human_label NOT IN ('ambiguous', 'needs_more_context')
        """)

        # Build confusion data
        predictions: dict[str, list[str]] = {}  # classifier -> list of true labels
        actuals: dict[str, list[str]] = {}  # true label -> list of classifier predictions

        for row in cursor.fetchall():
            classifier_type, human_label = row

            if classifier_type not in predictions:
                predictions[classifier_type] = []
            predictions[classifier_type].append(human_label)

            if human_label not in actuals:
                actuals[human_label] = []
            actuals[human_label].append(classifier_type)

        # Compute metrics
        metrics: dict[str, Any] = {
            "total_labelled": sum(len(preds) for preds in predictions.values()),
            "by_classifier_type": {},
            "by_true_label": {},
        }

        # Map labels to exit types for comparison
        label_to_type = {
            "true_dex_sell": "dex_sell",
            "true_transfer": "transfer_out",
            "true_lp_add": "lp_add",
            "true_lp_remove": "lp_remove",
            "true_cex_deposit": "cex_deposit",
            "true_burn": "burn",
            "true_bridge": "bridge",
            "true_migration": "wallet_migration",
            "true_program_interaction": "program_interaction",
        }

        # Precision by classifier type
        for classifier_type, true_labels in predictions.items():
            expected_label = f"true_{classifier_type}" if not classifier_type.startswith("true_") else classifier_type
            correct = sum(1 for label in true_labels if label_to_type.get(label) == classifier_type)
            total = len(true_labels)
            precision = correct / total if total > 0 else 0

            metrics["by_classifier_type"][classifier_type] = {
                "total": total,
                "correct": correct,
                "precision": round(precision, 3),
            }

        # Recall by true label
        for true_label, classifier_types in actuals.items():
            expected_type = label_to_type.get(true_label)
            if not expected_type:
                continue

            correct = sum(1 for ct in classifier_types if ct == expected_type)
            total = len(classifier_types)
            recall = correct / total if total > 0 else 0

            metrics["by_true_label"][true_label] = {
                "total": total,
                "correctly_classified": correct,
                "recall": round(recall, 3),
            }

        # False sell rate: transfers classified as sells
        false_sells = sum(
            1 for pred, labels in predictions.items()
            if pred == "dex_sell"
            for label in labels
            if label in ("true_transfer", "true_lp_add", "true_lp_remove", "true_cex_deposit")
        )
        total_sell_predictions = len(predictions.get("dex_sell", []))
        metrics["false_sell_rate"] = round(
            false_sells / total_sell_predictions if total_sell_predictions > 0 else 0,
            3,
        )

        # False transfer rate: sells classified as transfers
        false_transfers = sum(
            1 for pred, labels in predictions.items()
            if pred == "transfer_out"
            for label in labels
            if label == "true_dex_sell"
        )
        total_transfer_predictions = len(predictions.get("transfer_out", []))
        metrics["false_transfer_rate"] = round(
            false_transfers / total_transfer_predictions if total_transfer_predictions > 0 else 0,
            3,
        )

        # Unknown exit analysis
        unknown_predictions = predictions.get("unknown_exit", [])
        if unknown_predictions:
            unknown_breakdown = {}
            for label in unknown_predictions:
                unknown_breakdown[label] = unknown_breakdown.get(label, 0) + 1
            metrics["unknown_exit_analysis"] = {
                "total": len(unknown_predictions),
                "breakdown": unknown_breakdown,
            }

        return metrics

    def export_dataset(self, output_path: Path | str) -> None:
        """Export the validation dataset to JSON."""
        if not self._db:
            return

        cursor = self._db.execute("SELECT * FROM validation_samples")
        columns = [desc[0] for desc in cursor.description]

        samples = []
        for row in cursor.fetchall():
            samples.append(dict(zip(columns, row)))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                {
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "targets": {
                        "dex_sells": self._targets.dex_sells,
                        "transfers": self._targets.transfers,
                        "lp_actions": self._targets.lp_actions,
                        "cex_deposits": self._targets.cex_deposits,
                        "unknown_exits": self._targets.unknown_exits,
                    },
                    "progress": {
                        "collected": self.get_progress().collected,
                        "labelled": self.get_progress().labelled,
                        "verified": self.get_progress().verified,
                    },
                    "metrics": self.compute_accuracy_metrics(),
                    "samples": samples,
                },
                f,
                indent=2,
            )

        logger.info("validation_dataset_exported", path=str(output_path))

    def _sample_exists(self, signature: str) -> bool:
        """Check if a sample with this signature already exists."""
        if not self._db:
            return False

        cursor = self._db.execute(
            "SELECT 1 FROM validation_samples WHERE signature = ?",
            (signature,),
        )
        return cursor.fetchone() is not None

    def _row_to_sample(self, data: dict[str, Any]) -> ValidationSample | None:
        """Convert database row to ValidationSample."""
        try:
            return ValidationSample(
                sample_id=data["sample_id"],
                signature=data["signature"],
                token_mint=data["token_mint"],
                wallet_address=data["wallet_address"],
                classifier_exit_type=data["classifier_exit_type"],
                classifier_confidence=data["classifier_confidence"],
                classifier_evidence_json=data["classifier_evidence_json"],
                status=ValidationStatus(data["status"]),
                human_label=ValidationLabel(data["human_label"]) if data.get("human_label") else None,
                reviewer_id=data.get("reviewer_id"),
                review_timestamp=datetime.fromisoformat(data["review_timestamp"]) if data.get("review_timestamp") else None,
                review_notes=data.get("review_notes"),
                second_reviewer_id=data.get("second_reviewer_id"),
                second_review_timestamp=datetime.fromisoformat(data["second_review_timestamp"]) if data.get("second_review_timestamp") else None,
                second_label=ValidationLabel(data["second_label"]) if data.get("second_label") else None,
                created_at=datetime.fromisoformat(data["created_at"]),
            )
        except Exception as e:
            logger.error("sample_parse_error", error=str(e))
            return None

    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.close()


# ============================================================================
# Sample Collector (Auto-collection from monitoring)
# ============================================================================


class ValidationSampleCollector:
    """
    Automatically collects validation samples from classification logs.

    Maintains stratified sampling to meet dataset targets.
    """

    def __init__(
        self,
        dataset_builder: ValidationDatasetBuilder,
        sampling_rate: float = 0.1,  # Sample 10% of classifications
    ):
        self._builder = dataset_builder
        self._sampling_rate = sampling_rate
        self._sample_count = 0

    def maybe_collect(
        self,
        signature: str,
        token_mint: str,
        wallet_address: str,
        classifier_exit_type: str,
        classifier_confidence: float,
        classifier_evidence_json: str,
    ) -> bool:
        """
        Maybe collect this classification as a validation sample.

        Uses stratified sampling based on dataset needs.
        Returns True if collected.
        """
        import random

        # Check if we need more of this type
        progress = self._builder.get_progress()
        needed = progress.needs_more(classifier_exit_type)

        if needed <= 0:
            return False

        # Adjust sampling rate based on need
        # If we need many more, sample more aggressively
        adjusted_rate = min(1.0, self._sampling_rate * (1 + needed / 10))

        if random.random() > adjusted_rate:
            return False

        # Try to add sample
        sample = self._builder.add_sample(
            signature=signature,
            token_mint=token_mint,
            wallet_address=wallet_address,
            classifier_exit_type=classifier_exit_type,
            classifier_confidence=classifier_confidence,
            classifier_evidence_json=classifier_evidence_json,
        )

        if sample:
            self._sample_count += 1
            return True

        return False


# ============================================================================
# Factory Functions
# ============================================================================


def create_validation_dataset_builder(
    db_path: Path | str | None = None,
    targets: DatasetTargets | None = None,
) -> ValidationDatasetBuilder:
    """Create a validation dataset builder."""
    return ValidationDatasetBuilder(db_path=db_path, targets=targets)


def create_sample_collector(
    dataset_builder: ValidationDatasetBuilder,
    sampling_rate: float = 0.1,
) -> ValidationSampleCollector:
    """Create a validation sample collector."""
    return ValidationSampleCollector(dataset_builder, sampling_rate)
