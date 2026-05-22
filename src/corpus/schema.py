"""
Unified Label Schema (Sprint 10 - Deliverable 1).

Provides the core schema for the intelligence corpus:
- Label domains (exit events, coordination, wallet behaviour, etc.)
- Review statuses (pending, labelled, verified, disputed, etc.)
- Label records with versioning and audit trail
- Disagreement tracking

HARD RULES:
1. Model labels are not ground truth
2. Human labels must preserve reviewer and evidence
3. Every label must be versioned
4. Disagreements must be preserved, not overwritten
5. Ambiguous is a valid label
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


# ============================================================================
# Label Domains
# ============================================================================


class LabelDomain(str, Enum):
    """Domains for labelling in the intelligence corpus."""

    EXIT_EVENT = "exit_event"
    COORDINATION = "coordination"
    WALLET_BEHAVIOUR = "wallet_behaviour"
    TOKEN_OUTCOME = "token_outcome"
    LAUNCH_TRAJECTORY = "launch_trajectory"
    ENTITY_RESOLUTION = "entity_resolution"


# ============================================================================
# Review Status
# ============================================================================


class ReviewStatus(str, Enum):
    """Status of a label in the review workflow."""

    PENDING = "pending"  # Awaiting human review
    LABELLED = "labelled"  # First human label assigned
    VERIFIED = "verified"  # Second reviewer confirmed
    DISPUTED = "disputed"  # Reviewers disagree
    REJECTED = "rejected"  # Label rejected as invalid
    NEEDS_MORE_CONTEXT = "needs_more_context"  # Insufficient evidence


# ============================================================================
# Domain-Specific Labels
# ============================================================================


class ExitEventLabel(str, Enum):
    """Labels for exit event classification."""

    DEX_SELL = "dex_sell"
    TRANSFER_OUT = "transfer_out"
    LP_ADD = "lp_add"
    LP_REMOVE = "lp_remove"
    CEX_DEPOSIT = "cex_deposit"
    BURN = "burn"
    BRIDGE = "bridge"
    WALLET_MIGRATION = "wallet_migration"
    PROGRAM_INTERACTION = "program_interaction"
    UNKNOWN_EXIT = "unknown_exit"


class CoordinationLabel(str, Enum):
    """Labels for coordination cluster classification."""

    TRUE_COORDINATED = "true_coordinated"
    LIKELY_COORDINATED = "likely_coordinated"
    NOT_COORDINATED = "not_coordinated"
    AMBIGUOUS = "ambiguous"
    NEEDS_MORE_CONTEXT = "needs_more_context"


class WalletBehaviourLabel(str, Enum):
    """Labels for wallet behaviour profiles."""

    SNIPER = "sniper"
    LONG_TERM_ACCUMULATOR = "long_term_accumulator"
    LIQUIDITY_ACTOR = "liquidity_actor"
    DORMANT_WHALE = "dormant_whale"
    EXCHANGE_LINKED = "exchange_linked"
    COORDINATION_PARTICIPANT = "coordination_participant"
    UNKNOWN = "unknown"


class TokenOutcomeLabel(str, Enum):
    """Labels for token outcomes."""

    HEALTHY_SURVIVOR = "healthy_survivor"
    RAPID_COLLAPSE = "rapid_collapse"
    RUG_PULL = "rug_pull"
    LIQUIDITY_DRAIN = "liquidity_drain"
    WASH_VOLUME = "wash_volume"
    ORGANIC_VOLATILE = "organic_volatile"
    UNKNOWN_OUTCOME = "unknown_outcome"


class LaunchTrajectoryLabel(str, Enum):
    """Labels for launch trajectories."""

    ORGANIC_GROWTH = "organic_growth"
    COORDINATED_LAUNCH = "coordinated_launch"
    BOT_SNIPED = "bot_sniped"
    INSIDER_DISTRIBUTION = "insider_distribution"
    LIQUIDITY_COLLAPSE = "liquidity_collapse"
    SLOW_DECAY = "slow_decay"
    UNKNOWN_TRAJECTORY = "unknown_trajectory"


class EntityResolutionLabel(str, Enum):
    """Labels for entity resolution links."""

    SAME_ENTITY = "same_entity"
    LIKELY_SAME_ENTITY = "likely_same_entity"
    NOT_SAME_ENTITY = "not_same_entity"
    AMBIGUOUS = "ambiguous"


# ============================================================================
# Label Record
# ============================================================================


@dataclass
class LabelRecord:
    """
    A label in the intelligence corpus.

    Supports:
    - Multiple domains
    - Versioning
    - Evidence preservation
    - Human review workflow
    - Disagreement tracking
    """

    # Identity
    label_id: str
    domain: LabelDomain
    object_type: str  # "transaction", "wallet", "token", "cluster", "entity_pair"
    object_id: str  # The ID of the object being labelled

    # Labels
    proposed_label: str  # Model's proposed label
    human_label: str | None  # Human-assigned label (if reviewed)

    # Confidence
    model_confidence: float  # Model's confidence (0-1)
    human_confidence: float | None  # Human's confidence (if provided)

    # Evidence
    evidence_json: str  # Serialized evidence package

    # Review
    review_status: ReviewStatus
    reviewer_id: str | None
    reviewed_at: datetime | None
    review_notes: str | None

    # Second review
    second_reviewer_id: str | None
    second_reviewed_at: datetime | None
    second_label: str | None

    # Versioning
    version: int
    source_model: str  # Model that proposed the label
    source_model_version: str
    data_version: str  # Version of input data

    # Timestamps
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/export."""
        return {
            "label_id": self.label_id,
            "domain": self.domain.value,
            "object_type": self.object_type,
            "object_id": self.object_id,
            "proposed_label": self.proposed_label,
            "human_label": self.human_label,
            "model_confidence": self.model_confidence,
            "human_confidence": self.human_confidence,
            "evidence_json": self.evidence_json,
            "review_status": self.review_status.value,
            "reviewer_id": self.reviewer_id,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "review_notes": self.review_notes,
            "second_reviewer_id": self.second_reviewer_id,
            "second_reviewed_at": self.second_reviewed_at.isoformat() if self.second_reviewed_at else None,
            "second_label": self.second_label,
            "version": self.version,
            "source_model": self.source_model,
            "source_model_version": self.source_model_version,
            "data_version": self.data_version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def is_disputed(self) -> bool:
        """Check if label is disputed (reviewers disagree)."""
        return self.review_status == ReviewStatus.DISPUTED

    def is_verified(self) -> bool:
        """Check if label is verified (second reviewer confirmed)."""
        return self.review_status == ReviewStatus.VERIFIED

    def has_human_label(self) -> bool:
        """Check if a human label has been assigned."""
        return self.human_label is not None

    def model_human_agree(self) -> bool | None:
        """Check if model and human agree (None if no human label)."""
        if self.human_label is None:
            return None
        return self.proposed_label == self.human_label


@dataclass
class LabelVersion:
    """Version history entry for a label."""

    version_id: str
    label_id: str
    version_number: int
    field_changed: str
    old_value: str | None
    new_value: str
    changed_by: str
    changed_at: datetime
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "label_id": self.label_id,
            "version_number": self.version_number,
            "field_changed": self.field_changed,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "changed_by": self.changed_by,
            "changed_at": self.changed_at.isoformat(),
            "reason": self.reason,
        }


@dataclass
class LabelDisagreement:
    """Record of disagreement between reviewers."""

    disagreement_id: str
    label_id: str
    first_reviewer_id: str
    first_label: str
    first_confidence: float | None
    second_reviewer_id: str
    second_label: str
    second_confidence: float | None
    resolution: str | None  # Final resolved label
    resolved_by: str | None
    resolved_at: datetime | None
    resolution_notes: str | None
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "disagreement_id": self.disagreement_id,
            "label_id": self.label_id,
            "first_reviewer_id": self.first_reviewer_id,
            "first_label": self.first_label,
            "first_confidence": self.first_confidence,
            "second_reviewer_id": self.second_reviewer_id,
            "second_label": self.second_label,
            "second_confidence": self.second_confidence,
            "resolution": self.resolution,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
            "created_at": self.created_at.isoformat(),
        }


# ============================================================================
# Label Repository
# ============================================================================


class LabelRepository:
    """
    Repository for storing and retrieving labels.

    Provides:
    - CRUD operations for labels
    - Version history tracking
    - Disagreement recording
    - Query by domain, status, etc.
    """

    def __init__(self, db_path: Path | str):
        self._db_path = Path(db_path).expanduser()
        self._db: sqlite3.Connection | None = None
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Ensure database exists with correct schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._db_path))

        # Labels table
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                label_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                object_type TEXT NOT NULL,
                object_id TEXT NOT NULL,
                proposed_label TEXT NOT NULL,
                human_label TEXT,
                model_confidence REAL NOT NULL,
                human_confidence REAL,
                evidence_json TEXT NOT NULL,
                review_status TEXT NOT NULL,
                reviewer_id TEXT,
                reviewed_at TEXT,
                review_notes TEXT,
                second_reviewer_id TEXT,
                second_reviewed_at TEXT,
                second_label TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                source_model TEXT NOT NULL,
                source_model_version TEXT NOT NULL,
                data_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Version history table
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS label_versions (
                version_id TEXT PRIMARY KEY,
                label_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                field_changed TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT NOT NULL,
                changed_by TEXT NOT NULL,
                changed_at TEXT NOT NULL,
                reason TEXT,
                FOREIGN KEY (label_id) REFERENCES labels(label_id)
            )
        """)

        # Disagreements table
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS label_disagreements (
                disagreement_id TEXT PRIMARY KEY,
                label_id TEXT NOT NULL,
                first_reviewer_id TEXT NOT NULL,
                first_label TEXT NOT NULL,
                first_confidence REAL,
                second_reviewer_id TEXT NOT NULL,
                second_label TEXT NOT NULL,
                second_confidence REAL,
                resolution TEXT,
                resolved_by TEXT,
                resolved_at TEXT,
                resolution_notes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (label_id) REFERENCES labels(label_id)
            )
        """)

        # Indexes
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_labels_domain ON labels(domain)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_labels_status ON labels(review_status)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_labels_object ON labels(object_type, object_id)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_versions_label ON label_versions(label_id)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_disagreements_label ON label_disagreements(label_id)")

        self._db.commit()

    def create_label(
        self,
        domain: LabelDomain,
        object_type: str,
        object_id: str,
        proposed_label: str,
        model_confidence: float,
        evidence_json: str,
        source_model: str,
        source_model_version: str,
        data_version: str,
    ) -> LabelRecord:
        """Create a new label in PENDING status."""
        now = datetime.now(timezone.utc)
        label_id = f"lbl_{uuid.uuid4().hex[:16]}"

        record = LabelRecord(
            label_id=label_id,
            domain=domain,
            object_type=object_type,
            object_id=object_id,
            proposed_label=proposed_label,
            human_label=None,
            model_confidence=model_confidence,
            human_confidence=None,
            evidence_json=evidence_json,
            review_status=ReviewStatus.PENDING,
            reviewer_id=None,
            reviewed_at=None,
            review_notes=None,
            second_reviewer_id=None,
            second_reviewed_at=None,
            second_label=None,
            version=1,
            source_model=source_model,
            source_model_version=source_model_version,
            data_version=data_version,
            created_at=now,
            updated_at=now,
        )

        if self._db:
            self._db.execute(
                """
                INSERT INTO labels (
                    label_id, domain, object_type, object_id,
                    proposed_label, human_label, model_confidence, human_confidence,
                    evidence_json, review_status, reviewer_id, reviewed_at, review_notes,
                    second_reviewer_id, second_reviewed_at, second_label,
                    version, source_model, source_model_version, data_version,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.label_id, record.domain.value, record.object_type, record.object_id,
                    record.proposed_label, record.human_label, record.model_confidence, record.human_confidence,
                    record.evidence_json, record.review_status.value, record.reviewer_id,
                    record.reviewed_at.isoformat() if record.reviewed_at else None, record.review_notes,
                    record.second_reviewer_id,
                    record.second_reviewed_at.isoformat() if record.second_reviewed_at else None,
                    record.second_label, record.version, record.source_model, record.source_model_version,
                    record.data_version, record.created_at.isoformat(), record.updated_at.isoformat(),
                ),
            )
            self._db.commit()

        logger.info(
            "label_created",
            label_id=label_id,
            domain=domain.value,
            object_id=object_id[:16] + "..." if len(object_id) > 16 else object_id,
        )

        return record

    def add_human_label(
        self,
        label_id: str,
        human_label: str,
        reviewer_id: str,
        confidence: float | None = None,
        notes: str | None = None,
    ) -> bool:
        """Add first human label to a pending label."""
        record = self.get_label(label_id)
        if not record:
            return False

        if record.review_status != ReviewStatus.PENDING:
            logger.warning("label_already_reviewed", label_id=label_id, status=record.review_status.value)
            return False

        now = datetime.now(timezone.utc)
        new_version = record.version + 1

        if self._db:
            # Update label
            self._db.execute(
                """
                UPDATE labels SET
                    human_label = ?,
                    human_confidence = ?,
                    review_status = ?,
                    reviewer_id = ?,
                    reviewed_at = ?,
                    review_notes = ?,
                    version = ?,
                    updated_at = ?
                WHERE label_id = ?
                """,
                (
                    human_label, confidence, ReviewStatus.LABELLED.value,
                    reviewer_id, now.isoformat(), notes, new_version, now.isoformat(),
                    label_id,
                ),
            )

            # Record version history
            self._record_version(
                label_id=label_id,
                version_number=new_version,
                field_changed="human_label",
                old_value=None,
                new_value=human_label,
                changed_by=reviewer_id,
                reason="First human label",
            )

            self._db.commit()

        logger.info(
            "human_label_added",
            label_id=label_id,
            human_label=human_label,
            reviewer=reviewer_id,
        )

        return True

    def verify_label(
        self,
        label_id: str,
        second_label: str,
        reviewer_id: str,
        confidence: float | None = None,
    ) -> bool:
        """Add second review to verify or dispute a label."""
        record = self.get_label(label_id)
        if not record:
            return False

        if record.review_status != ReviewStatus.LABELLED:
            logger.warning("label_not_ready_for_verification", label_id=label_id, status=record.review_status.value)
            return False

        if record.reviewer_id == reviewer_id:
            logger.warning("same_reviewer", label_id=label_id)
            return False

        now = datetime.now(timezone.utc)
        new_version = record.version + 1

        # Determine if verified or disputed
        if second_label == record.human_label:
            new_status = ReviewStatus.VERIFIED
        else:
            new_status = ReviewStatus.DISPUTED
            # Record disagreement
            self._record_disagreement(
                label_id=label_id,
                first_reviewer_id=record.reviewer_id,
                first_label=record.human_label,
                first_confidence=record.human_confidence,
                second_reviewer_id=reviewer_id,
                second_label=second_label,
                second_confidence=confidence,
            )

        if self._db:
            self._db.execute(
                """
                UPDATE labels SET
                    review_status = ?,
                    second_reviewer_id = ?,
                    second_reviewed_at = ?,
                    second_label = ?,
                    version = ?,
                    updated_at = ?
                WHERE label_id = ?
                """,
                (
                    new_status.value, reviewer_id, now.isoformat(),
                    second_label, new_version, now.isoformat(),
                    label_id,
                ),
            )

            self._record_version(
                label_id=label_id,
                version_number=new_version,
                field_changed="review_status",
                old_value=ReviewStatus.LABELLED.value,
                new_value=new_status.value,
                changed_by=reviewer_id,
                reason=f"Second review: {second_label}",
            )

            self._db.commit()

        logger.info(
            "label_verified",
            label_id=label_id,
            status=new_status.value,
            second_reviewer=reviewer_id,
        )

        return True

    def resolve_dispute(
        self,
        label_id: str,
        resolution: str,
        resolver_id: str,
        notes: str | None = None,
    ) -> bool:
        """Resolve a disputed label."""
        record = self.get_label(label_id)
        if not record:
            return False

        if record.review_status != ReviewStatus.DISPUTED:
            return False

        now = datetime.now(timezone.utc)
        new_version = record.version + 1

        if self._db:
            # Update label with resolution
            self._db.execute(
                """
                UPDATE labels SET
                    human_label = ?,
                    review_status = ?,
                    review_notes = ?,
                    version = ?,
                    updated_at = ?
                WHERE label_id = ?
                """,
                (
                    resolution, ReviewStatus.VERIFIED.value, notes,
                    new_version, now.isoformat(), label_id,
                ),
            )

            # Update disagreement record
            self._db.execute(
                """
                UPDATE label_disagreements SET
                    resolution = ?,
                    resolved_by = ?,
                    resolved_at = ?,
                    resolution_notes = ?
                WHERE label_id = ?
                """,
                (resolution, resolver_id, now.isoformat(), notes, label_id),
            )

            self._record_version(
                label_id=label_id,
                version_number=new_version,
                field_changed="human_label",
                old_value=record.human_label,
                new_value=resolution,
                changed_by=resolver_id,
                reason=f"Dispute resolution: {notes}",
            )

            self._db.commit()

        return True

    def get_label(self, label_id: str) -> LabelRecord | None:
        """Get a label by ID."""
        if not self._db:
            return None

        cursor = self._db.execute("SELECT * FROM labels WHERE label_id = ?", (label_id,))
        row = cursor.fetchone()
        if not row:
            return None

        columns = [desc[0] for desc in cursor.description]
        data = dict(zip(columns, row))
        return self._row_to_record(data)

    def get_labels(
        self,
        domain: LabelDomain | None = None,
        status: ReviewStatus | None = None,
        object_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LabelRecord]:
        """Query labels with filters."""
        if not self._db:
            return []

        query = "SELECT * FROM labels WHERE 1=1"
        params: list[Any] = []

        if domain:
            query += " AND domain = ?"
            params.append(domain.value)
        if status:
            query += " AND review_status = ?"
            params.append(status.value)
        if object_type:
            query += " AND object_type = ?"
            params.append(object_type)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self._db.execute(query, params)
        columns = [desc[0] for desc in cursor.description]

        records = []
        for row in cursor.fetchall():
            data = dict(zip(columns, row))
            record = self._row_to_record(data)
            if record:
                records.append(record)

        return records

    def get_pending_count(self, domain: LabelDomain | None = None) -> int:
        """Get count of pending labels."""
        if not self._db:
            return 0

        if domain:
            cursor = self._db.execute(
                "SELECT COUNT(*) FROM labels WHERE review_status = ? AND domain = ?",
                (ReviewStatus.PENDING.value, domain.value),
            )
        else:
            cursor = self._db.execute(
                "SELECT COUNT(*) FROM labels WHERE review_status = ?",
                (ReviewStatus.PENDING.value,),
            )

        return cursor.fetchone()[0]

    def get_label_counts(self) -> dict[str, dict[str, int]]:
        """Get counts by domain and status."""
        if not self._db:
            return {}

        cursor = self._db.execute("""
            SELECT domain, review_status, COUNT(*) as count
            FROM labels
            GROUP BY domain, review_status
        """)

        counts: dict[str, dict[str, int]] = {}
        for row in cursor.fetchall():
            domain, status, count = row
            if domain not in counts:
                counts[domain] = {}
            counts[domain][status] = count

        return counts

    def get_disagreements(
        self,
        unresolved_only: bool = True,
        limit: int = 100,
    ) -> list[LabelDisagreement]:
        """Get label disagreements."""
        if not self._db:
            return []

        query = "SELECT * FROM label_disagreements"
        if unresolved_only:
            query += " WHERE resolution IS NULL"
        query += " ORDER BY created_at DESC LIMIT ?"

        cursor = self._db.execute(query, (limit,))
        columns = [desc[0] for desc in cursor.description]

        disagreements = []
        for row in cursor.fetchall():
            data = dict(zip(columns, row))
            disagreement = LabelDisagreement(
                disagreement_id=data["disagreement_id"],
                label_id=data["label_id"],
                first_reviewer_id=data["first_reviewer_id"],
                first_label=data["first_label"],
                first_confidence=data["first_confidence"],
                second_reviewer_id=data["second_reviewer_id"],
                second_label=data["second_label"],
                second_confidence=data["second_confidence"],
                resolution=data["resolution"],
                resolved_by=data["resolved_by"],
                resolved_at=datetime.fromisoformat(data["resolved_at"]) if data["resolved_at"] else None,
                resolution_notes=data["resolution_notes"],
                created_at=datetime.fromisoformat(data["created_at"]),
            )
            disagreements.append(disagreement)

        return disagreements

    def _record_version(
        self,
        label_id: str,
        version_number: int,
        field_changed: str,
        old_value: str | None,
        new_value: str,
        changed_by: str,
        reason: str | None = None,
    ) -> None:
        """Record a version history entry."""
        if not self._db:
            return

        version_id = f"ver_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        self._db.execute(
            """
            INSERT INTO label_versions (
                version_id, label_id, version_number, field_changed,
                old_value, new_value, changed_by, changed_at, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id, label_id, version_number, field_changed,
                old_value, new_value, changed_by, now.isoformat(), reason,
            ),
        )

    def _record_disagreement(
        self,
        label_id: str,
        first_reviewer_id: str,
        first_label: str,
        first_confidence: float | None,
        second_reviewer_id: str,
        second_label: str,
        second_confidence: float | None,
    ) -> None:
        """Record a disagreement between reviewers."""
        if not self._db:
            return

        disagreement_id = f"dis_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        self._db.execute(
            """
            INSERT INTO label_disagreements (
                disagreement_id, label_id,
                first_reviewer_id, first_label, first_confidence,
                second_reviewer_id, second_label, second_confidence,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                disagreement_id, label_id,
                first_reviewer_id, first_label, first_confidence,
                second_reviewer_id, second_label, second_confidence,
                now.isoformat(),
            ),
        )

    def _row_to_record(self, data: dict[str, Any]) -> LabelRecord | None:
        """Convert database row to LabelRecord."""
        try:
            return LabelRecord(
                label_id=data["label_id"],
                domain=LabelDomain(data["domain"]),
                object_type=data["object_type"],
                object_id=data["object_id"],
                proposed_label=data["proposed_label"],
                human_label=data["human_label"],
                model_confidence=data["model_confidence"],
                human_confidence=data["human_confidence"],
                evidence_json=data["evidence_json"],
                review_status=ReviewStatus(data["review_status"]),
                reviewer_id=data["reviewer_id"],
                reviewed_at=datetime.fromisoformat(data["reviewed_at"]) if data["reviewed_at"] else None,
                review_notes=data["review_notes"],
                second_reviewer_id=data["second_reviewer_id"],
                second_reviewed_at=datetime.fromisoformat(data["second_reviewed_at"]) if data["second_reviewed_at"] else None,
                second_label=data["second_label"],
                version=data["version"],
                source_model=data["source_model"],
                source_model_version=data["source_model_version"],
                data_version=data["data_version"],
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
            )
        except Exception as e:
            logger.error("label_parse_error", error=str(e))
            return None

    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.close()


# ============================================================================
# Factory Functions
# ============================================================================


def create_label_repository(
    db_path: Path | str | None = None,
) -> LabelRepository:
    """Create a label repository."""
    if db_path is None:
        db_path = Path("~/.shi/intelligence_corpus.db")
    return LabelRepository(db_path)
