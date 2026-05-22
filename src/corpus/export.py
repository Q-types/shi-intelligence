"""
Corpus Export (Sprint 10 - Deliverable 6).

Supports export formats:
- JSONL (JSON Lines)
- CSV
- Parquet

Each export must include:
- label
- evidence
- confidence
- source_model_version
- review_status
- data_version

HARD RULE: All exports must include model and data version.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import structlog

from .schema import (
    LabelDomain,
    LabelRecord,
    LabelRepository,
    ReviewStatus,
)

logger = structlog.get_logger()


# ============================================================================
# Export Format
# ============================================================================


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"


# ============================================================================
# Export Configuration
# ============================================================================


@dataclass
class ExportConfig:
    """Configuration for corpus export."""

    # Filters
    domain: LabelDomain | None = None
    status: ReviewStatus | None = None
    min_confidence: float | None = None
    verified_only: bool = False

    # Content options
    include_evidence: bool = True
    include_version_history: bool = False

    # Metadata
    export_version: str = "1.0.0"
    export_description: str | None = None


# ============================================================================
# Corpus Exporter
# ============================================================================


class CorpusExporter:
    """
    Exports intelligence corpus to various formats.

    All exports include model and data version for reproducibility.
    """

    def __init__(self, repository: LabelRepository):
        self._repo = repository

    def export(
        self,
        output_path: Path | str,
        format: ExportFormat,
        config: ExportConfig | None = None,
    ) -> dict[str, Any]:
        """
        Export corpus to file.

        Args:
            output_path: Path to output file
            format: Export format (jsonl, csv, parquet)
            config: Export configuration

        Returns:
            Export metadata (count, path, etc.)
        """
        config = config or ExportConfig()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get labels
        labels = self._get_labels(config)

        # Export based on format
        if format == ExportFormat.JSONL:
            count = self._export_jsonl(labels, output_path, config)
        elif format == ExportFormat.CSV:
            count = self._export_csv(labels, output_path, config)
        elif format == ExportFormat.PARQUET:
            count = self._export_parquet(labels, output_path, config)
        else:
            raise ValueError(f"Unsupported format: {format}")

        metadata = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "format": format.value,
            "output_path": str(output_path),
            "count": count,
            "config": {
                "domain": config.domain.value if config.domain else None,
                "status": config.status.value if config.status else None,
                "verified_only": config.verified_only,
                "include_evidence": config.include_evidence,
            },
            "export_version": config.export_version,
        }

        logger.info(
            "corpus_exported",
            format=format.value,
            count=count,
            path=str(output_path),
        )

        return metadata

    def _get_labels(self, config: ExportConfig) -> list[LabelRecord]:
        """Get labels based on config filters."""
        labels = self._repo.get_labels(
            domain=config.domain,
            status=config.status,
            limit=1000000,
        )

        # Apply additional filters
        if config.verified_only:
            labels = [l for l in labels if l.review_status == ReviewStatus.VERIFIED]

        if config.min_confidence is not None:
            labels = [l for l in labels if l.model_confidence >= config.min_confidence]

        return labels

    def _export_jsonl(
        self,
        labels: list[LabelRecord],
        output_path: Path,
        config: ExportConfig,
    ) -> int:
        """Export to JSON Lines format."""
        with open(output_path, "w") as f:
            for label in labels:
                record = self._label_to_export_record(label, config)
                f.write(json.dumps(record, default=str) + "\n")

        return len(labels)

    def _export_csv(
        self,
        labels: list[LabelRecord],
        output_path: Path,
        config: ExportConfig,
    ) -> int:
        """Export to CSV format."""
        if not labels:
            return 0

        # Define columns
        columns = [
            "label_id",
            "domain",
            "object_type",
            "object_id",
            "proposed_label",
            "human_label",
            "final_label",
            "model_confidence",
            "human_confidence",
            "review_status",
            "source_model",
            "source_model_version",
            "data_version",
            "created_at",
            "reviewed_at",
        ]

        if config.include_evidence:
            columns.append("evidence_json")

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for label in labels:
                row = {
                    "label_id": label.label_id,
                    "domain": label.domain.value,
                    "object_type": label.object_type,
                    "object_id": label.object_id,
                    "proposed_label": label.proposed_label,
                    "human_label": label.human_label,
                    "final_label": label.human_label or label.proposed_label,
                    "model_confidence": label.model_confidence,
                    "human_confidence": label.human_confidence,
                    "review_status": label.review_status.value,
                    "source_model": label.source_model,
                    "source_model_version": label.source_model_version,
                    "data_version": label.data_version,
                    "created_at": label.created_at.isoformat(),
                    "reviewed_at": label.reviewed_at.isoformat() if label.reviewed_at else None,
                }

                if config.include_evidence:
                    row["evidence_json"] = label.evidence_json

                writer.writerow(row)

        return len(labels)

    def _export_parquet(
        self,
        labels: list[LabelRecord],
        output_path: Path,
        config: ExportConfig,
    ) -> int:
        """Export to Parquet format."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.error("pyarrow_not_installed", message="Install pyarrow for Parquet export")
            raise ImportError("pyarrow required for Parquet export: pip install pyarrow")

        if not labels:
            return 0

        # Build data arrays
        data = {
            "label_id": [l.label_id for l in labels],
            "domain": [l.domain.value for l in labels],
            "object_type": [l.object_type for l in labels],
            "object_id": [l.object_id for l in labels],
            "proposed_label": [l.proposed_label for l in labels],
            "human_label": [l.human_label for l in labels],
            "final_label": [l.human_label or l.proposed_label for l in labels],
            "model_confidence": [l.model_confidence for l in labels],
            "human_confidence": [l.human_confidence for l in labels],
            "review_status": [l.review_status.value for l in labels],
            "source_model": [l.source_model for l in labels],
            "source_model_version": [l.source_model_version for l in labels],
            "data_version": [l.data_version for l in labels],
            "created_at": [l.created_at.isoformat() for l in labels],
            "reviewed_at": [l.reviewed_at.isoformat() if l.reviewed_at else None for l in labels],
        }

        if config.include_evidence:
            data["evidence_json"] = [l.evidence_json for l in labels]

        # Create table and write
        table = pa.table(data)
        pq.write_table(table, output_path)

        return len(labels)

    def _label_to_export_record(
        self,
        label: LabelRecord,
        config: ExportConfig,
    ) -> dict[str, Any]:
        """Convert label to export record."""
        record = {
            "label_id": label.label_id,
            "domain": label.domain.value,
            "object_type": label.object_type,
            "object_id": label.object_id,
            "proposed_label": label.proposed_label,
            "human_label": label.human_label,
            "final_label": label.human_label or label.proposed_label,
            "model_confidence": label.model_confidence,
            "human_confidence": label.human_confidence,
            "review_status": label.review_status.value,
            "source_model": label.source_model,
            "source_model_version": label.source_model_version,
            "data_version": label.data_version,
            "created_at": label.created_at.isoformat(),
            "reviewed_at": label.reviewed_at.isoformat() if label.reviewed_at else None,
        }

        if config.include_evidence:
            record["evidence"] = json.loads(label.evidence_json)

        return record

    def get_export_stats(self) -> dict[str, Any]:
        """Get statistics about exportable data."""
        counts = self._repo.get_label_counts()

        total = 0
        verified = 0
        by_domain = {}

        for domain, statuses in counts.items():
            domain_total = sum(statuses.values())
            domain_verified = statuses.get(ReviewStatus.VERIFIED.value, 0)

            total += domain_total
            verified += domain_verified
            by_domain[domain] = {
                "total": domain_total,
                "verified": domain_verified,
                "exportable": domain_total,
            }

        return {
            "total_labels": total,
            "verified_labels": verified,
            "by_domain": by_domain,
        }


# ============================================================================
# Factory Function
# ============================================================================


def create_corpus_exporter(repository: LabelRepository) -> CorpusExporter:
    """Create a corpus exporter."""
    return CorpusExporter(repository)
