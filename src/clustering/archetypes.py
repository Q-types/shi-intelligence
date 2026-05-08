"""
Behavioral Archetype Definitions and Assignment.

WARNING: Archetype definitions are FIXED per PDR Section 5.
These are behavioral classifications only - not identity claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
import structlog

logger = structlog.get_logger()


class Archetype(Enum):
    """
    Fixed behavioral archetypes per PDR Section 5.

    DO NOT ADD OR REMOVE archetypes without human approval.
    """

    SNIPER = "sniper"
    LONG_TERM_ACCUMULATOR = "long_term_accumulator"
    COORDINATED_CLUSTER = "coordinated_cluster"
    LIQUIDITY_ACTOR = "liquidity_actor"
    EXCHANGE_LINKED = "exchange_linked"
    DORMANT_WHALE = "dormant_whale"
    UNKNOWN = "unknown"  # For wallets that don't fit any archetype


@dataclass(frozen=True)
class ArchetypeDefinition:
    """
    Definition of an archetype with characteristic features.

    These definitions are FROZEN per PDR.
    """

    archetype: Archetype
    description: str
    required_features: tuple[str, ...]
    feature_thresholds: dict[str, tuple[str, float]]  # feature -> (operator, threshold)


# FROZEN ARCHETYPE DEFINITIONS (PDR Section 5)
ARCHETYPES: dict[Archetype, ArchetypeDefinition] = {
    Archetype.SNIPER: ArchetypeDefinition(
        archetype=Archetype.SNIPER,
        description="Early entry + short holding time + high turnover",
        required_features=("entry_time_relative", "holding_duration", "trade_count"),
        feature_thresholds={
            "entry_time_relative": ("<=", 0.1),  # Top 10% earliest
            "holding_duration": ("<=", 7.0),  # Less than 7 days
            "trade_count": (">=", 5),  # High activity
        },
    ),
    Archetype.LONG_TERM_ACCUMULATOR: ArchetypeDefinition(
        archetype=Archetype.LONG_TERM_ACCUMULATOR,
        description="Gradual position growth + low churn",
        required_features=("holding_duration", "delta_balance_30d", "trade_count"),
        feature_thresholds={
            "holding_duration": (">=", 30.0),  # At least 30 days
            "delta_balance_30d": (">=", 0.0),  # Net positive or neutral
            "trade_count": ("<=", 10),  # Low activity
        },
    ),
    Archetype.COORDINATED_CLUSTER: ArchetypeDefinition(
        archetype=Archetype.COORDINATED_CLUSTER,
        description="Shared funders + temporal synchronization",
        required_features=("shared_funder_count", "entry_time_relative"),
        feature_thresholds={
            "shared_funder_count": (">=", 1),  # At least one shared funder
            # Temporal sync checked via clustering
        },
    ),
    Archetype.LIQUIDITY_ACTOR: ArchetypeDefinition(
        archetype=Archetype.LIQUIDITY_ACTOR,
        description="Frequent LP interactions",
        required_features=("lp_interaction_ratio",),
        feature_thresholds={
            "lp_interaction_ratio": (">=", 0.3),  # >30% of txs are LP related
        },
    ),
    Archetype.EXCHANGE_LINKED: ArchetypeDefinition(
        archetype=Archetype.EXCHANGE_LINKED,
        description="High fan-out + known CEX link signatures",
        required_features=("out_degree", "is_exchange"),
        feature_thresholds={
            "out_degree": (">=", 50),  # High fan-out
            # Or: is_exchange flag
        },
    ),
    Archetype.DORMANT_WHALE: ArchetypeDefinition(
        archetype=Archetype.DORMANT_WHALE,
        description="Large share + low transaction activity",
        required_features=("share", "trade_count", "holding_duration"),
        feature_thresholds={
            "share": (">=", 0.01),  # Top 1% by balance
            "trade_count": ("<=", 3),  # Very few trades
            "holding_duration": (">=", 14.0),  # Held for 2+ weeks
        },
    ),
}


@dataclass
class WalletFeatureVector:
    """Feature vector for archetype classification."""

    wallet: str

    # Distribution features
    balance: float
    share: float
    rank: int

    # Temporal features
    entry_time_relative: float  # Days since token launch
    holding_duration: float  # Days held
    position_volatility: float

    # Flow features
    delta_balance_7d: float
    delta_balance_30d: float

    # Trade features
    trade_count: int
    burstiness: float
    swap_frequency: float
    lp_interaction_ratio: float

    # Graph features
    in_degree: int
    out_degree: int
    eigenvector_centrality: float
    shared_funder_count: int

    # Flags
    is_exchange: bool = False

    def to_array(self) -> NDArray[np.float64]:
        """Convert to numpy array for clustering."""
        return np.array([
            self.share,
            self.entry_time_relative,
            self.holding_duration,
            self.position_volatility,
            self.delta_balance_7d,
            self.delta_balance_30d,
            self.trade_count,
            self.burstiness,
            self.swap_frequency,
            self.lp_interaction_ratio,
            self.in_degree,
            self.out_degree,
            self.eigenvector_centrality,
            self.shared_funder_count,
        ], dtype=np.float64)


@dataclass
class ArchetypeAssignment:
    """Result of archetype classification."""

    wallet: str
    archetype: Archetype
    confidence: float
    matching_features: list[str]
    feature_values: dict[str, float]


def _check_threshold(value: float, operator: str, threshold: float) -> bool:
    """Check if value meets threshold condition."""
    if operator == ">=":
        return value >= threshold
    elif operator == "<=":
        return value <= threshold
    elif operator == ">":
        return value > threshold
    elif operator == "<":
        return value < threshold
    elif operator == "==":
        return value == threshold
    else:
        raise ValueError(f"Unknown operator: {operator}")


def assign_archetype(features: WalletFeatureVector) -> ArchetypeAssignment:
    """
    Assign a single wallet to its most likely archetype.

    Uses rule-based classification per PDR definitions.
    Returns the archetype with highest confidence.
    """
    best_match: ArchetypeAssignment | None = None
    best_score = 0.0

    for archetype, definition in ARCHETYPES.items():
        if archetype == Archetype.UNKNOWN:
            continue

        matching = []
        total_checks = 0

        for feature_name, (operator, threshold) in definition.feature_thresholds.items():
            total_checks += 1

            # Get feature value
            if hasattr(features, feature_name):
                value = getattr(features, feature_name)
                if _check_threshold(float(value), operator, threshold):
                    matching.append(feature_name)

        # Calculate confidence
        if total_checks > 0:
            confidence = len(matching) / total_checks
        else:
            confidence = 0.0

        # Track best match
        if confidence > best_score:
            best_score = confidence
            best_match = ArchetypeAssignment(
                wallet=features.wallet,
                archetype=archetype,
                confidence=confidence,
                matching_features=matching,
                feature_values={
                    f: float(getattr(features, f, 0.0))
                    for f in definition.required_features
                    if hasattr(features, f)
                },
            )

    # If no good match, assign UNKNOWN
    if best_match is None or best_match.confidence < 0.5:
        return ArchetypeAssignment(
            wallet=features.wallet,
            archetype=Archetype.UNKNOWN,
            confidence=0.0,
            matching_features=[],
            feature_values={},
        )

    return best_match


def cluster_wallets(
    features_list: list[WalletFeatureVector],
    min_cluster_size: int = 5,
) -> dict[str, ArchetypeAssignment]:
    """
    Cluster wallets and assign archetypes.

    Uses HDBSCAN for density-based clustering, then assigns
    archetypes based on cluster characteristics.

    Args:
        features_list: List of wallet feature vectors
        min_cluster_size: Minimum cluster size for HDBSCAN

    Returns:
        Dict mapping wallet -> ArchetypeAssignment
    """
    if not features_list:
        return {}

    logger.info("clustering_wallets", count=len(features_list))

    # First, do rule-based assignment for each wallet
    assignments = {}
    for features in features_list:
        assignments[features.wallet] = assign_archetype(features)

    # For coordinated cluster detection, we need additional analysis
    # Find wallets with shared funders
    shared_funder_wallets = [
        f for f in features_list if f.shared_funder_count > 0
    ]

    if len(shared_funder_wallets) >= min_cluster_size:
        try:
            import hdbscan

            # Cluster by funding pattern similarity
            feature_matrix = np.array([f.to_array() for f in shared_funder_wallets])

            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)

            # HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric="euclidean",
            )
            cluster_labels = clusterer.fit_predict(feature_matrix_scaled)

            # Mark coordinated clusters
            for i, features in enumerate(shared_funder_wallets):
                if cluster_labels[i] >= 0:  # Not noise
                    # Override with coordinated cluster if confidence is high
                    if features.shared_funder_count >= 2:
                        assignments[features.wallet] = ArchetypeAssignment(
                            wallet=features.wallet,
                            archetype=Archetype.COORDINATED_CLUSTER,
                            confidence=0.8,
                            matching_features=["shared_funder_count", "cluster_membership"],
                            feature_values={
                                "shared_funder_count": features.shared_funder_count,
                                "cluster_id": int(cluster_labels[i]),
                            },
                        )

            logger.info(
                "coordinated_clusters_detected",
                cluster_count=len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                noise_count=sum(1 for label in cluster_labels if label == -1),
            )

        except ImportError:
            logger.warning("hdbscan_not_available", msg="Falling back to rule-based only")
        except Exception as e:
            logger.error("clustering_failed", error=str(e))

    return assignments


def get_archetype_distribution(
    assignments: dict[str, ArchetypeAssignment],
) -> dict[str, float]:
    """
    Get proportion of wallets in each archetype.

    Returns:
        Dict mapping archetype name -> proportion
    """
    if not assignments:
        return {}

    counts: dict[str, int] = {}
    for assignment in assignments.values():
        name = assignment.archetype.value
        counts[name] = counts.get(name, 0) + 1

    total = len(assignments)
    return {name: count / total for name, count in counts.items()}
