"""
Behavioral Archetype Definitions and Assignment.

WARNING: Archetype definitions are FIXED per PDR Section 5.
These are behavioral classifications only - not identity claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray
import structlog

from ..core.config import settings

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
            # IMPORTANT: shared_funder_count >= 1 was too permissive on Solana
            # 77% of wallets had >= 2 shared funders, 44% had >= 5
            # Use 5 as minimum for meaningful coordination signal
            "shared_funder_count": (">=", 5),  # Significant coordination threshold
            # Temporal sync still checked via clustering
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

    # Advanced centrality metrics (new)
    pagerank: float | None = None  # PageRank importance score
    betweenness_centrality: float | None = None  # Bridge/hub detection

    # Weighted graph features
    total_funding_received: float | None = None  # Total SOL received as funding
    largest_funder_share: float | None = None  # % of funding from largest funder
    funding_hhi: float | None = None  # Herfindahl-Hirschman Index for funding concentration
    funding_burst_score: float | None = None  # Temporal burstiness of funding
    weighted_in_degree: float | None = None  # Sum of incoming edge weights
    weighted_out_degree: float | None = None  # Sum of outgoing edge weights

    # Temporal coordination features (new)
    temporal_sync_score: float | None = None  # Synchronized funding detection
    funding_time_spread_hours: float | None = None  # Time spread of funding events

    # Flags
    is_exchange: bool = False

    # Price features (optional, set when price data available)
    entry_price_usd: float | None = None
    current_price_usd: float | None = None
    unrealized_pnl_ratio: float = 0.0
    unrealized_pnl_usd: float | None = None

    # Price-derived intelligence features (Sprint 7)
    price_change_1h_pct: float | None = None
    price_change_24h_pct: float | None = None
    price_change_7d_pct: float | None = None

    # Price-holder divergence signals
    # Positive = whale accumulating while price falling (bullish divergence)
    # Negative = whale distributing while price rising (bearish divergence)
    holder_growth_vs_price_change: float | None = None
    whale_accumulation_vs_price_change: float | None = None

    # Liquidity-adjusted risk signals
    sell_pressure_vs_liquidity: float | None = None  # sell_pressure / liquidity_usd
    unrealized_profit_concentration: float | None = None  # % of unrealized profit in top 10

    # Liquidity smoothing
    liquidity_usd_current: float | None = None
    liquidity_usd_1h_avg: float | None = None
    liquidity_usd_24h_avg: float | None = None
    liquidity_depth_confidence: str | None = None  # high, medium, low

    def to_array(self) -> NDArray[np.float64]:
        """
        Convert to numpy array for clustering.

        Includes weighted graph features for better coordination detection.
        None values are replaced with 0.0 for clustering stability.
        """
        # Helper to safely convert None to 0.0
        def safe(val: float | None) -> float:
            return 0.0 if val is None else float(val)

        return np.array([
            # Distribution (1 feature)
            self.share,
            # Temporal (4 features)
            self.entry_time_relative,
            self.holding_duration,
            self.position_volatility,
            safe(self.funding_time_spread_hours),
            # Flow (2 features)
            self.delta_balance_7d,
            self.delta_balance_30d,
            # Trade (4 features)
            self.trade_count,
            self.burstiness,
            self.swap_frequency,
            self.lp_interaction_ratio,
            # Basic graph (4 features)
            self.in_degree,
            self.out_degree,
            self.eigenvector_centrality,
            self.shared_funder_count,
            # Advanced centrality (2 features)
            safe(self.pagerank),
            safe(self.betweenness_centrality),
            # Weighted graph (6 features) - CRITICAL for coordination detection
            safe(self.total_funding_received),
            safe(self.largest_funder_share),
            safe(self.funding_hhi),
            safe(self.funding_burst_score),
            safe(self.weighted_in_degree),
            safe(self.weighted_out_degree),
            # Temporal coordination (1 feature)
            safe(self.temporal_sync_score),
        ], dtype=np.float64)


@dataclass
class ArchetypeAssignment:
    """Result of archetype classification."""

    wallet: str
    archetype: Archetype
    confidence: float
    matching_features: list[str]
    feature_values: dict[str, float]


@dataclass
class MultiScoreAssignment:
    """
    Multi-score archetype assignment (upgraded from hard labels).

    Keeps fixed PDR archetypes but computes scores for ALL archetypes,
    enabling secondary labels and soft assignments.
    """

    wallet: str

    # Primary assignment (highest scoring)
    primary_archetype: Archetype
    primary_confidence: float

    # All archetype scores (sorted by confidence)
    all_scores: dict[Archetype, float]

    # Secondary labels (archetypes with confidence >= threshold)
    secondary_archetypes: list[Archetype]

    # Cluster-derived adjustments
    cluster_status: str = "unknown"  # core, border, noise, unknown
    cluster_confidence_adjustment: float = 0.0

    # Feature match details
    feature_matches: dict[Archetype, list[str]] = field(default_factory=dict)  # archetype -> matching features

    @property
    def adjusted_confidence(self) -> float:
        """Primary confidence adjusted for cluster status."""
        return max(0.0, min(1.0, self.primary_confidence + self.cluster_confidence_adjustment))

    @property
    def is_noise(self) -> bool:
        """Check if wallet is in noise cluster."""
        return self.cluster_status == "noise"

    @property
    def has_secondary(self) -> bool:
        """Check if wallet has secondary archetype labels."""
        return len(self.secondary_archetypes) > 0

    def get_top_n_archetypes(self, n: int = 3) -> list[tuple[Archetype, float]]:
        """Get top N archetypes by score."""
        sorted_scores = sorted(
            self.all_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_scores[:n]

    def to_legacy_assignment(self) -> ArchetypeAssignment:
        """Convert to legacy single-label ArchetypeAssignment for backward compatibility."""
        return ArchetypeAssignment(
            wallet=self.wallet,
            archetype=self.primary_archetype,
            confidence=self.adjusted_confidence,
            matching_features=self.feature_matches.get(self.primary_archetype, []),
            feature_values={},  # Would need to add feature values
        )


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


def assign_archetype_multi_score(
    features: WalletFeatureVector,
    secondary_threshold: float = 0.4,
    cluster_status: str = "unknown",
    cluster_confidence_adj: float = 0.0,
) -> MultiScoreAssignment:
    """
    Assign archetype with multi-score output.

    Computes scores for ALL archetypes, enabling:
    - Secondary labels for wallets matching multiple archetypes
    - Soft assignments based on score distribution
    - Cluster-aware confidence adjustment

    Args:
        features: Wallet feature vector
        secondary_threshold: Minimum confidence for secondary labels
        cluster_status: HDBSCAN cluster status (core, border, noise, unknown)
        cluster_confidence_adj: Confidence adjustment from cluster diagnostics

    Returns:
        MultiScoreAssignment with all archetype scores
    """
    all_scores: dict[Archetype, float] = {}
    feature_matches: dict[Archetype, list[str]] = {}

    for archetype, definition in ARCHETYPES.items():
        if archetype == Archetype.UNKNOWN:
            continue

        matching = []
        total_checks = 0

        for feature_name, (operator, threshold) in definition.feature_thresholds.items():
            total_checks += 1

            if hasattr(features, feature_name):
                value = getattr(features, feature_name)
                # Handle None values
                if value is None:
                    continue
                if _check_threshold(float(value), operator, threshold):
                    matching.append(feature_name)

        # Calculate confidence as proportion of matched thresholds
        confidence = len(matching) / total_checks if total_checks > 0 else 0.0

        all_scores[archetype] = confidence
        feature_matches[archetype] = matching

    # Add UNKNOWN with score 0
    all_scores[Archetype.UNKNOWN] = 0.0
    feature_matches[Archetype.UNKNOWN] = []

    # Determine primary archetype (highest score)
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    primary_archetype = sorted_scores[0][0]
    primary_confidence = sorted_scores[0][1]

    # If best score is too low, assign UNKNOWN
    if primary_confidence < 0.5:
        primary_archetype = Archetype.UNKNOWN
        primary_confidence = 0.0

    # Find secondary archetypes (above threshold, excluding primary)
    secondary_archetypes = [
        arch for arch, score in sorted_scores
        if score >= secondary_threshold
        and arch != primary_archetype
        and arch != Archetype.UNKNOWN
    ]

    return MultiScoreAssignment(
        wallet=features.wallet,
        primary_archetype=primary_archetype,
        primary_confidence=primary_confidence,
        all_scores=all_scores,
        secondary_archetypes=secondary_archetypes,
        cluster_status=cluster_status,
        cluster_confidence_adjustment=cluster_confidence_adj,
        feature_matches=feature_matches,
    )


def cluster_wallets_with_diagnostics(
    features_list: list[WalletFeatureVector],
    min_cluster_size: int = 5,
    return_diagnostics: bool = True,
) -> tuple[dict[str, MultiScoreAssignment], dict | None]:
    """
    Cluster wallets with full diagnostics and multi-score assignments.

    Enhanced version of cluster_wallets that:
    - Returns multi-score assignments instead of hard labels
    - Includes HDBSCAN diagnostics
    - Properly handles noise points

    Args:
        features_list: List of wallet feature vectors
        min_cluster_size: Minimum cluster size for HDBSCAN
        return_diagnostics: Whether to return clustering diagnostics

    Returns:
        (assignments_dict, diagnostics_dict or None)
    """
    if not features_list:
        return {}, None

    logger.info("clustering_wallets_with_diagnostics", count=len(features_list))

    # Import diagnostics module
    from .diagnostics import HDBSCANDiagnostics, ClusterStatus

    # Build feature matrix
    feature_matrix = np.array([f.to_array() for f in features_list])

    # Handle NaN values for clustering
    # Replace NaN with column medians for clustering only
    col_medians = np.nanmedian(feature_matrix, axis=0)
    for i in range(feature_matrix.shape[1]):
        mask = np.isnan(feature_matrix[:, i])
        feature_matrix[mask, i] = col_medians[i]

    # Run HDBSCAN with diagnostics
    hdbscan_diag = HDBSCANDiagnostics(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        diagnostics = hdbscan_diag.fit(feature_matrix_scaled)
    except Exception as e:
        logger.error("hdbscan_fit_failed", error=str(e))
        diagnostics = None

    # Assign archetypes with cluster info
    assignments: dict[str, MultiScoreAssignment] = {}

    for i, features in enumerate(features_list):
        # Get cluster status and confidence adjustment
        if diagnostics is not None:
            wallet_info = hdbscan_diag.get_wallet_info(features.wallet, i)
            cluster_status = wallet_info.cluster_status.value
            confidence_adj = wallet_info.confidence_adjustment
        else:
            cluster_status = "unknown"
            confidence_adj = 0.0

        # Assign with multi-score
        assignment = assign_archetype_multi_score(
            features,
            cluster_status=cluster_status,
            cluster_confidence_adj=confidence_adj,
        )

        # Override to COORDINATED_CLUSTER if significant shared funders detected in cluster
        # IMPORTANT: Threshold is configurable - default 2 is too permissive on Solana
        # Semantic audit found 77% of wallets have shared_funder_count >= 2
        # Recommended: 5-7 for meaningful coordination detection
        coord_threshold = settings.coordination_shared_funder_threshold
        coord_confidence = settings.coordination_confidence_threshold

        if (
            features.shared_funder_count >= coord_threshold
            and diagnostics is not None
            and diagnostics.labels[i] >= 0  # Not noise
        ):
            # Update scores to reflect coordinated behavior
            assignment.all_scores[Archetype.COORDINATED_CLUSTER] = max(
                assignment.all_scores.get(Archetype.COORDINATED_CLUSTER, 0),
                coord_confidence,
            )
            if assignment.primary_confidence < coord_confidence:
                assignment = MultiScoreAssignment(
                    wallet=features.wallet,
                    primary_archetype=Archetype.COORDINATED_CLUSTER,
                    primary_confidence=coord_confidence,
                    all_scores=assignment.all_scores,
                    secondary_archetypes=[assignment.primary_archetype]
                    if assignment.primary_archetype != Archetype.UNKNOWN
                    else [],
                    cluster_status=cluster_status,
                    cluster_confidence_adjustment=confidence_adj,
                    feature_matches=assignment.feature_matches,
                )

        assignments[features.wallet] = assignment

    # Prepare diagnostics dict
    diag_dict = diagnostics.to_dict() if diagnostics else None

    logger.info(
        "clustering_completed",
        total_wallets=len(assignments),
        noise_wallets=sum(1 for a in assignments.values() if a.is_noise),
        with_secondary=sum(1 for a in assignments.values() if a.has_secondary),
    )

    return assignments, diag_dict


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
            # Use configurable threshold (default 2 is too permissive on Solana)
            coord_threshold = settings.coordination_shared_funder_threshold
            coord_confidence = settings.coordination_confidence_threshold

            for i, features in enumerate(shared_funder_wallets):
                if cluster_labels[i] >= 0:  # Not noise
                    # Override with coordinated cluster if shared funders exceed threshold
                    if features.shared_funder_count >= coord_threshold:
                        assignments[features.wallet] = ArchetypeAssignment(
                            wallet=features.wallet,
                            archetype=Archetype.COORDINATED_CLUSTER,
                            confidence=coord_confidence,
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
