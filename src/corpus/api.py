"""
Human Review API (Sprint 10 - Deliverable 4).

Provides REST API endpoints for human-in-the-loop labelling:
- GET /api/v1/review/queue - Get prioritized review queue
- GET /api/v1/review/item/{label_id} - Get label details with evidence
- POST /api/v1/review/item/{label_id}/label - Submit human label
- POST /api/v1/review/item/{label_id}/verify - Verify a labelled item
- POST /api/v1/review/item/{label_id}/dispute - Dispute a label
- GET /api/v1/review/progress - Get review progress stats
- GET /api/v1/review/metrics - Get quality metrics

HARD RULE: Model labels are NOT ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .evidence import EvidencePackage
from .quality_metrics import QualityMetricsComputer
from .review_queue import ReviewQueue, create_review_queue
from .schema import (
    LabelDomain,
    LabelRecord,
    LabelRepository,
    ReviewStatus,
)

logger = structlog.get_logger()

# ============================================================================
# API Models
# ============================================================================


class QueueFilter(str, Enum):
    """Review queue filter types."""

    PENDING = "pending"
    LABELLED = "labelled"
    DISPUTED = "disputed"
    VERIFICATION = "verification"


class LabelRequest(BaseModel):
    """Request to label an item."""

    human_label: str = Field(..., description="Human-assigned label")
    human_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in label (0-1)"
    )
    reviewer_id: str = Field(..., description="ID of the reviewer")
    notes: str | None = Field(None, description="Optional reviewer notes")


class VerifyRequest(BaseModel):
    """Request to verify a labelled item."""

    verified: bool = Field(..., description="Whether to verify the label")
    second_label: str | None = Field(None, description="Second reviewer's label if different")
    reviewer_id: str = Field(..., description="ID of the verifying reviewer")
    notes: str | None = Field(None, description="Optional verification notes")


class DisputeRequest(BaseModel):
    """Request to dispute a label."""

    disputed_label: str = Field(..., description="The disputed label value")
    reason: str = Field(..., description="Reason for dispute")
    reviewer_id: str = Field(..., description="ID of the disputing reviewer")


class QueueItemResponse(BaseModel):
    """Response for a queue item."""

    label_id: str
    domain: str
    object_type: str
    object_id: str
    proposed_label: str
    model_confidence: float
    priority_score: float
    priority_factors: list[str]
    created_at: str
    evidence_summary: str | None = None


class LabelDetailResponse(BaseModel):
    """Detailed response for a label."""

    label_id: str
    domain: str
    object_type: str
    object_id: str
    proposed_label: str
    human_label: str | None
    final_label: str | None
    model_confidence: float
    human_confidence: float | None
    review_status: str
    source_model: str
    source_model_version: str
    data_version: str
    evidence: dict[str, Any]
    created_at: str
    reviewed_at: str | None
    reviewer_id: str | None
    notes: str | None
    version_history: list[dict[str, Any]]


class ProgressResponse(BaseModel):
    """Response for review progress."""

    total_labels: int
    pending: int
    labelled: int
    verified: int
    disputed: int
    completion_rate: float
    verification_rate: float
    dispute_rate: float
    by_domain: dict[str, dict[str, int]]


class MetricsResponse(BaseModel):
    """Response for quality metrics."""

    computed_at: str
    total_labels: int
    total_reviewed: int
    total_verified: int
    overall_inter_reviewer_agreement: float | None
    overall_cohens_kappa: float | None
    overall_model_human_agreement: float | None
    kappa_threshold_met: bool
    agreement_threshold_met: bool
    ready_for_training: bool
    recommendations: list[str]
    domain_metrics: dict[str, dict[str, Any]]


class ActionResponse(BaseModel):
    """Response for action endpoints."""

    success: bool
    label_id: str
    new_status: str
    message: str


# ============================================================================
# API Router Factory
# ============================================================================


def create_review_api(repository: LabelRepository) -> APIRouter:
    """
    Create the review API router.

    Args:
        repository: Label repository instance

    Returns:
        FastAPI router with review endpoints
    """
    router = APIRouter(prefix="/api/v1/review", tags=["review"])
    queue = create_review_queue(repository)
    metrics_computer = QualityMetricsComputer(repository)

    @router.get("/queue", response_model=list[QueueItemResponse])
    async def get_queue(
        filter: QueueFilter = Query(QueueFilter.PENDING, description="Queue filter type"),
        domain: str | None = Query(None, description="Filter by domain"),
        limit: int = Query(50, ge=1, le=200, description="Maximum items to return"),
    ) -> list[QueueItemResponse]:
        """
        Get prioritized review queue.

        Returns items sorted by priority (highest first).
        """
        domain_enum = LabelDomain(domain) if domain else None

        if filter == QueueFilter.PENDING:
            items = queue.get_queue(domain=domain_enum, limit=limit)
        elif filter == QueueFilter.LABELLED:
            items = queue.get_verification_queue(limit=limit)
        elif filter == QueueFilter.DISPUTED:
            items = queue.get_disputed_queue(limit=limit)
        elif filter == QueueFilter.VERIFICATION:
            items = queue.get_verification_queue(limit=limit)
        else:
            items = queue.get_queue(domain=domain_enum, limit=limit)

        return [
            QueueItemResponse(
                label_id=item.label_id,
                domain=item.domain.value,
                object_type=item.object_type,
                object_id=item.object_id,
                proposed_label=item.proposed_label,
                model_confidence=item.model_confidence,
                priority_score=item.priority.total,
                priority_factors=item.priority.factors,
                created_at=item.created_at.isoformat(),
                evidence_summary=item.evidence_summary,
            )
            for item in items
        ]

    @router.get("/item/{label_id}", response_model=LabelDetailResponse)
    async def get_label_detail(label_id: str) -> LabelDetailResponse:
        """
        Get detailed information about a label including evidence.
        """
        label = repository.get_label(label_id)
        if not label:
            raise HTTPException(status_code=404, detail=f"Label {label_id} not found")

        # Get version history
        history = repository.get_version_history(label_id)

        # Parse evidence
        try:
            evidence = json.loads(label.evidence_json)
        except (json.JSONDecodeError, TypeError):
            evidence = {}

        return LabelDetailResponse(
            label_id=label.label_id,
            domain=label.domain.value,
            object_type=label.object_type,
            object_id=label.object_id,
            proposed_label=label.proposed_label,
            human_label=label.human_label,
            final_label=label.human_label or label.proposed_label,
            model_confidence=label.model_confidence,
            human_confidence=label.human_confidence,
            review_status=label.review_status.value,
            source_model=label.source_model,
            source_model_version=label.source_model_version,
            data_version=label.data_version,
            evidence=evidence,
            created_at=label.created_at.isoformat(),
            reviewed_at=label.reviewed_at.isoformat() if label.reviewed_at else None,
            reviewer_id=label.reviewer_id,
            notes=label.notes,
            version_history=history,
        )

    @router.post("/item/{label_id}/label", response_model=ActionResponse)
    async def submit_label(label_id: str, request: LabelRequest) -> ActionResponse:
        """
        Submit a human label for an item.

        Transitions status from PENDING to LABELLED.
        """
        label = repository.get_label(label_id)
        if not label:
            raise HTTPException(status_code=404, detail=f"Label {label_id} not found")

        if label.review_status != ReviewStatus.PENDING:
            raise HTTPException(
                status_code=400,
                detail=f"Label {label_id} is not pending (status: {label.review_status.value})",
            )

        # Update label
        success = repository.update_label(
            label_id=label_id,
            human_label=request.human_label,
            human_confidence=request.human_confidence,
            review_status=ReviewStatus.LABELLED,
            reviewer_id=request.reviewer_id,
            notes=request.notes,
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update label")

        logger.info(
            "label_submitted",
            label_id=label_id,
            human_label=request.human_label,
            reviewer_id=request.reviewer_id,
        )

        return ActionResponse(
            success=True,
            label_id=label_id,
            new_status=ReviewStatus.LABELLED.value,
            message=f"Label submitted: {request.human_label}",
        )

    @router.post("/item/{label_id}/verify", response_model=ActionResponse)
    async def verify_label(label_id: str, request: VerifyRequest) -> ActionResponse:
        """
        Verify or dispute a labelled item (second review).

        If verified=True, transitions to VERIFIED.
        If verified=False with different second_label, transitions to DISPUTED.
        """
        label = repository.get_label(label_id)
        if not label:
            raise HTTPException(status_code=404, detail=f"Label {label_id} not found")

        if label.review_status != ReviewStatus.LABELLED:
            raise HTTPException(
                status_code=400,
                detail=f"Label {label_id} is not labelled (status: {label.review_status.value})",
            )

        if request.verified:
            # Mark as verified
            new_status = ReviewStatus.VERIFIED
            message = "Label verified"
        else:
            # Disagreement - check if second label differs
            if request.second_label and request.second_label != label.human_label:
                new_status = ReviewStatus.DISPUTED
                message = f"Label disputed: {label.human_label} vs {request.second_label}"
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Must provide different second_label when not verifying",
                )

        success = repository.update_label(
            label_id=label_id,
            review_status=new_status,
            second_label=request.second_label,
            notes=request.notes,
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update label")

        logger.info(
            "label_verified" if request.verified else "label_disputed",
            label_id=label_id,
            new_status=new_status.value,
            reviewer_id=request.reviewer_id,
        )

        return ActionResponse(
            success=True,
            label_id=label_id,
            new_status=new_status.value,
            message=message,
        )

    @router.post("/item/{label_id}/dispute", response_model=ActionResponse)
    async def dispute_label(label_id: str, request: DisputeRequest) -> ActionResponse:
        """
        Dispute an existing label.

        Can be used on LABELLED or VERIFIED items.
        """
        label = repository.get_label(label_id)
        if not label:
            raise HTTPException(status_code=404, detail=f"Label {label_id} not found")

        if label.review_status not in (ReviewStatus.LABELLED, ReviewStatus.VERIFIED):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot dispute label with status: {label.review_status.value}",
            )

        success = repository.update_label(
            label_id=label_id,
            review_status=ReviewStatus.DISPUTED,
            second_label=request.disputed_label,
            notes=f"Dispute by {request.reviewer_id}: {request.reason}",
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update label")

        logger.info(
            "label_disputed",
            label_id=label_id,
            disputed_by=request.reviewer_id,
            reason=request.reason,
        )

        return ActionResponse(
            success=True,
            label_id=label_id,
            new_status=ReviewStatus.DISPUTED.value,
            message=f"Label disputed: {request.reason}",
        )

    @router.get("/progress", response_model=ProgressResponse)
    async def get_progress() -> ProgressResponse:
        """
        Get review progress statistics.
        """
        progress = queue.get_progress()

        return ProgressResponse(
            total_labels=progress["total_labels"],
            pending=progress["pending"],
            labelled=progress["labelled"],
            verified=progress["verified"],
            disputed=progress["disputed"],
            completion_rate=progress["completion_rate"],
            verification_rate=progress["verification_rate"],
            dispute_rate=progress["dispute_rate"],
            by_domain=progress["by_domain"],
        )

    @router.get("/metrics", response_model=MetricsResponse)
    async def get_metrics() -> MetricsResponse:
        """
        Get label quality metrics.

        Includes Cohen's kappa, inter-reviewer agreement, and recommendations.
        """
        metrics = metrics_computer.compute()

        return MetricsResponse(
            computed_at=metrics.computed_at.isoformat(),
            total_labels=metrics.total_labels,
            total_reviewed=metrics.total_reviewed,
            total_verified=metrics.total_verified,
            overall_inter_reviewer_agreement=metrics.overall_inter_reviewer_agreement,
            overall_cohens_kappa=metrics.overall_cohens_kappa,
            overall_model_human_agreement=metrics.overall_model_human_agreement,
            kappa_threshold_met=metrics.kappa_threshold_met,
            agreement_threshold_met=metrics.agreement_threshold_met,
            ready_for_training=metrics.ready_for_training,
            recommendations=metrics.recommendations,
            domain_metrics={
                k: v.to_dict() for k, v in metrics.domain_metrics.items()
            },
        )

    @router.get("/domains")
    async def get_domains() -> dict[str, list[str]]:
        """
        Get available domains and their valid labels.
        """
        from .schema import (
            CoordinationLabel,
            EntityResolutionLabel,
            ExitEventLabel,
            LaunchTrajectoryLabel,
            TokenOutcomeLabel,
            WalletBehaviourLabel,
        )

        return {
            LabelDomain.EXIT_EVENT.value: [l.value for l in ExitEventLabel],
            LabelDomain.COORDINATION.value: [l.value for l in CoordinationLabel],
            LabelDomain.WALLET_BEHAVIOUR.value: [l.value for l in WalletBehaviourLabel],
            LabelDomain.TOKEN_OUTCOME.value: [l.value for l in TokenOutcomeLabel],
            LabelDomain.LAUNCH_TRAJECTORY.value: [l.value for l in LaunchTrajectoryLabel],
            LabelDomain.ENTITY_RESOLUTION.value: [l.value for l in EntityResolutionLabel],
        }

    return router


# ============================================================================
# Standalone App Factory
# ============================================================================


def create_review_app(repository: LabelRepository) -> "FastAPI":
    """
    Create a standalone FastAPI app for the review API.

    Args:
        repository: Label repository instance

    Returns:
        FastAPI application
    """
    from fastapi import FastAPI

    app = FastAPI(
        title="Intelligence Corpus Review API",
        description="Human-in-the-loop labelling system for Solana Holder Intelligence",
        version="1.0.0",
    )

    router = create_review_api(repository)
    app.include_router(router)

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    return app
