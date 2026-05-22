"""
Intelligence Corpus - Human-in-the-Loop Labelling System (Sprint 10).

Provides the data-foundry layer for SHI behavioural intelligence:
- Unified label schema across 6 domains
- Evidence packages for each label type
- Human review workflow with versioning
- Quality metrics for label reliability
- Corpus export for training data

CORE PRINCIPLE: Classifier outputs are NOT ground truth.

Every inferred label must support:
- Evidence
- Confidence
- Human review
- Disagreement handling
- Versioning
- Audit trail

HARD RULES:
1. Model labels are not ground truth
2. Human labels must preserve reviewer and evidence
3. Every label must be versioned
4. Disagreements must be preserved, not overwritten
5. Ambiguous is a valid label
6. Do not train supervised models until label quality metrics exist
7. All exports must include model and data version
8. No identity claims beyond behavioural/entity inference
"""

from .schema import (
    # Enums
    LabelDomain,
    ReviewStatus,
    # Exit Event Labels
    ExitEventLabel,
    # Coordination Labels
    CoordinationLabel,
    # Wallet Behaviour Labels
    WalletBehaviourLabel,
    # Token Outcome Labels
    TokenOutcomeLabel,
    # Launch Trajectory Labels
    LaunchTrajectoryLabel,
    # Entity Resolution Labels
    EntityResolutionLabel,
    # Core Schema
    LabelRecord,
    LabelVersion,
    LabelDisagreement,
    # Repository
    LabelRepository,
    create_label_repository,
)
from .evidence import (
    # Base
    EvidencePackage,
    # Domain-specific evidence
    ExitEventEvidence,
    CoordinationEvidence,
    WalletBehaviourEvidence,
    TokenOutcomeEvidence,
    LaunchTrajectoryEvidence,
    EntityLinkEvidence,
    # Builder
    EvidencePackageBuilder,
)
from .review_queue import (
    ReviewPriority,
    ReviewQueueItem,
    ReviewQueue,
    PriorityWeights,
    create_review_queue,
)
from .quality_metrics import (
    LabelQualityMetrics,
    DomainMetrics,
    compute_cohens_kappa,
    compute_inter_reviewer_agreement,
    QualityMetricsComputer,
)
from .export import (
    ExportFormat,
    ExportConfig,
    CorpusExporter,
    create_corpus_exporter,
)
from .active_learning import (
    SamplingStrategy,
    ActiveLearningHooks,
    UncertaintySampler,
    DisagreementSampler,
    RareClassSampler,
    HighImpactSampler,
)
from .api import (
    QueueFilter,
    create_review_api,
    create_review_app,
)

__all__ = [
    # Enums
    "LabelDomain",
    "ReviewStatus",
    "ExitEventLabel",
    "CoordinationLabel",
    "WalletBehaviourLabel",
    "TokenOutcomeLabel",
    "LaunchTrajectoryLabel",
    "EntityResolutionLabel",
    # Core Schema
    "LabelRecord",
    "LabelVersion",
    "LabelDisagreement",
    "LabelRepository",
    "create_label_repository",
    # Evidence
    "EvidencePackage",
    "ExitEventEvidence",
    "CoordinationEvidence",
    "WalletBehaviourEvidence",
    "TokenOutcomeEvidence",
    "LaunchTrajectoryEvidence",
    "EntityLinkEvidence",
    "EvidencePackageBuilder",
    # Review Queue
    "ReviewPriority",
    "ReviewQueueItem",
    "ReviewQueue",
    "PriorityWeights",
    "create_review_queue",
    # Quality Metrics
    "LabelQualityMetrics",
    "DomainMetrics",
    "compute_cohens_kappa",
    "compute_inter_reviewer_agreement",
    "QualityMetricsComputer",
    # Export
    "ExportFormat",
    "ExportConfig",
    "CorpusExporter",
    "create_corpus_exporter",
    # Active Learning
    "SamplingStrategy",
    "ActiveLearningHooks",
    "UncertaintySampler",
    "DisagreementSampler",
    "RareClassSampler",
    "HighImpactSampler",
    # API
    "QueueFilter",
    "create_review_api",
    "create_review_app",
]
