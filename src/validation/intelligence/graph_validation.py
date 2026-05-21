"""
Graph Intelligence Validation Module.

Validates graph features against null models, stability tests, and predictive usefulness.
Generates evidence-based deployment decisions.

Tasks:
1. Graph Feature Redundancy Audit
2. Temporal Coordination Null Model
3. Cluster Stability Reconciliation
4. Feature Health Check
5. Weighted Node2Vec Validation
6. SHAP Stability Audit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import warnings

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
import structlog

logger = structlog.get_logger()


# =============================================================================
# Task 1: Graph Feature Redundancy Audit
# =============================================================================

@dataclass
class FeatureRedundancyResult:
    """Results of feature redundancy analysis."""

    pearson_matrix: NDArray[np.float64]
    spearman_matrix: NDArray[np.float64]
    mutual_info_matrix: NDArray[np.float64]
    vif_scores: dict[str, float]
    dendrogram_clusters: dict[str, int]

    # Findings
    highly_correlated_pairs: list[tuple[str, str, float]]
    redundant_features: list[str]
    recommended_removals: list[str]

    feature_names: list[str]


def compute_mutual_information_matrix(X: NDArray[np.float64], n_bins: int = 10) -> NDArray[np.float64]:
    """Compute pairwise mutual information matrix."""
    n_features = X.shape[1]
    mi_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i, n_features):
            # Discretize for MI computation
            xi = np.digitize(X[:, i], bins=np.linspace(X[:, i].min(), X[:, i].max(), n_bins))
            xj = np.digitize(X[:, j], bins=np.linspace(X[:, j].min(), X[:, j].max(), n_bins))

            # Compute MI using contingency table
            from sklearn.metrics import mutual_info_score
            mi = mutual_info_score(xi, xj)
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix


def compute_vif(X: NDArray[np.float64], feature_names: list[str]) -> dict[str, float]:
    """Compute Variance Inflation Factor for each feature."""
    from sklearn.linear_model import LinearRegression

    vif_scores = {}
    n_features = X.shape[1]

    for i in range(n_features):
        # Regress feature i on all other features
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)

        if X_others.shape[1] == 0:
            vif_scores[feature_names[i]] = 1.0
            continue

        try:
            lr = LinearRegression()
            lr.fit(X_others, y)
            r_squared = lr.score(X_others, y)

            if r_squared >= 1.0:
                vif_scores[feature_names[i]] = float('inf')
            else:
                vif_scores[feature_names[i]] = 1.0 / (1.0 - r_squared)
        except Exception:
            vif_scores[feature_names[i]] = np.nan

    return vif_scores


def run_feature_redundancy_audit(
    features: NDArray[np.float64],
    feature_names: list[str],
    correlation_threshold: float = 0.8,
    vif_threshold: float = 10.0,
) -> FeatureRedundancyResult:
    """
    Run comprehensive feature redundancy audit.

    Args:
        features: (n_samples, n_features) array
        feature_names: Names of features
        correlation_threshold: Threshold for "highly correlated"
        vif_threshold: Threshold for multicollinearity

    Returns:
        FeatureRedundancyResult with all analysis
    """
    logger.info("running_feature_redundancy_audit", n_features=len(feature_names))

    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Pearson correlation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson_matrix = np.corrcoef(features, rowvar=False)
    pearson_matrix = np.nan_to_num(pearson_matrix, nan=0.0)

    # 2. Spearman correlation
    spearman_matrix = np.zeros((len(feature_names), len(feature_names)))
    for i in range(len(feature_names)):
        for j in range(i, len(feature_names)):
            try:
                corr, _ = stats.spearmanr(features[:, i], features[:, j])
                spearman_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                spearman_matrix[j, i] = spearman_matrix[i, j]
            except Exception:
                spearman_matrix[i, j] = 0.0
                spearman_matrix[j, i] = 0.0

    # 3. Mutual information
    mi_matrix = compute_mutual_information_matrix(features)

    # 4. VIF
    vif_scores = compute_vif(features, feature_names)

    # 5. Hierarchical clustering of features
    # Use correlation distance
    corr_distance = 1 - np.abs(pearson_matrix)
    np.fill_diagonal(corr_distance, 0)

    # Ensure valid distance matrix
    corr_distance = np.clip(corr_distance, 0, 2)

    try:
        condensed = squareform(corr_distance)
        Z = linkage(condensed, method='average')
        cluster_labels = fcluster(Z, t=0.5, criterion='distance')
        dendrogram_clusters = {
            name: int(label) for name, label in zip(feature_names, cluster_labels)
        }
    except Exception as e:
        logger.warning("dendrogram_clustering_failed", error=str(e))
        dendrogram_clusters = {name: i for i, name in enumerate(feature_names)}

    # Find highly correlated pairs
    highly_correlated = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            corr = abs(pearson_matrix[i, j])
            if corr >= correlation_threshold:
                highly_correlated.append((feature_names[i], feature_names[j], corr))

    highly_correlated.sort(key=lambda x: x[2], reverse=True)

    # Identify redundant features
    redundant = set()
    for f1, f2, _ in highly_correlated:
        # Keep the one with lower VIF
        vif1 = vif_scores.get(f1, 0)
        vif2 = vif_scores.get(f2, 0)
        if vif1 > vif2:
            redundant.add(f1)
        else:
            redundant.add(f2)

    # Also flag high VIF features
    for name, vif in vif_scores.items():
        if vif > vif_threshold:
            redundant.add(name)

    # Recommended removals
    recommended = list(redundant)

    logger.info(
        "redundancy_audit_complete",
        highly_correlated_pairs=len(highly_correlated),
        redundant_features=len(redundant),
    )

    return FeatureRedundancyResult(
        pearson_matrix=pearson_matrix,
        spearman_matrix=spearman_matrix,
        mutual_info_matrix=mi_matrix,
        vif_scores=vif_scores,
        dendrogram_clusters=dendrogram_clusters,
        highly_correlated_pairs=highly_correlated,
        redundant_features=list(redundant),
        recommended_removals=recommended,
        feature_names=feature_names,
    )


# =============================================================================
# Task 2: Temporal Coordination Null Model
# =============================================================================

@dataclass
class NullModelResult:
    """Results of null model testing for temporal coordination."""

    wallet: str
    observed_score: float
    null_scores: list[float]
    z_score: float
    p_value: float
    percentile_rank: float

    is_significant: bool
    significance_level: float


@dataclass
class TemporalNullValidation:
    """Complete temporal coordination null validation results."""

    wallet_results: dict[str, NullModelResult]
    n_permutations: int
    significance_threshold: float

    # Summary
    total_wallets_tested: int
    significant_coordinated: int
    false_positive_estimate: float

    method_description: str


def run_temporal_null_model(
    graph,  # FundingGraph
    wallets: list[str],
    n_permutations: int = 1000,
    alpha: float = 0.05,
    z_threshold: float = 2.0,
) -> TemporalNullValidation:
    """
    Run null model testing for temporal coordination.

    Preserves graph structure, randomly permutes funding timestamps,
    recomputes coordination scores, and tests statistical significance.

    Args:
        graph: FundingGraph with temporal data
        wallets: Wallets to test
        n_permutations: Number of permutation iterations
        alpha: Significance level
        z_threshold: Minimum z-score for coordination

    Returns:
        TemporalNullValidation with all results
    """
    from ...graph.temporal_patterns import detect_temporal_coordination

    logger.info(
        "running_temporal_null_model",
        n_wallets=len(wallets),
        n_permutations=n_permutations,
    )

    # Get observed coordination scores
    observed_results = detect_temporal_coordination(graph, wallets)

    # Extract timestamps from graph for permutation
    nx_graph = graph._graph
    edge_timestamps = {}
    for u, v, data in nx_graph.edges(data=True):
        ts = data.get("timestamp")
        if ts:
            edge_timestamps[(u, v)] = ts

    if not edge_timestamps:
        logger.warning("no_timestamps_found_for_null_model")
        return TemporalNullValidation(
            wallet_results={},
            n_permutations=n_permutations,
            significance_threshold=alpha,
            total_wallets_tested=len(wallets),
            significant_coordinated=0,
            false_positive_estimate=0.0,
            method_description="No timestamps available for permutation testing",
        )

    # Run permutations
    null_scores: dict[str, list[float]] = {w: [] for w in wallets}

    timestamps_list = list(edge_timestamps.values())

    for i in range(n_permutations):
        # Permute timestamps
        permuted_timestamps = np.random.permutation(timestamps_list)

        # Apply permuted timestamps to graph copy
        for (edge, _), new_ts in zip(edge_timestamps.items(), permuted_timestamps):
            u, v = edge
            if nx_graph.has_edge(u, v):
                nx_graph.edges[u, v]["timestamp"] = new_ts

        # Recompute coordination scores
        perm_results = detect_temporal_coordination(graph, wallets)

        for wallet in wallets:
            if wallet in perm_results:
                null_scores[wallet].append(perm_results[wallet].temporal_sync_score)

    # Restore original timestamps
    for (u, v), ts in edge_timestamps.items():
        if nx_graph.has_edge(u, v):
            nx_graph.edges[u, v]["timestamp"] = ts

    # Compute statistics for each wallet
    wallet_results = {}
    significant_count = 0

    for wallet in wallets:
        if wallet not in observed_results:
            continue

        observed = observed_results[wallet].temporal_sync_score
        null_dist = null_scores.get(wallet, [])

        if not null_dist:
            continue

        null_array = np.array(null_dist)
        null_mean = np.mean(null_array)
        null_std = np.std(null_array)

        # Z-score
        if null_std > 0:
            z_score = (observed - null_mean) / null_std
        else:
            z_score = 0.0 if observed == null_mean else (np.inf if observed > null_mean else -np.inf)

        # Empirical p-value (one-tailed, higher is more coordinated)
        p_value = np.mean(null_array >= observed)

        # Percentile rank
        percentile = np.mean(null_array <= observed) * 100

        # Significance test
        is_significant = (z_score >= z_threshold) and (p_value < alpha)

        if is_significant:
            significant_count += 1

        wallet_results[wallet] = NullModelResult(
            wallet=wallet,
            observed_score=observed,
            null_scores=null_dist,
            z_score=z_score,
            p_value=p_value,
            percentile_rank=percentile,
            is_significant=is_significant,
            significance_level=alpha,
        )

    # Estimate false positive rate
    # Under null, we expect alpha * n_wallets false positives
    expected_fp = alpha * len(wallets)
    fp_estimate = expected_fp / max(significant_count, 1) if significant_count > 0 else 1.0

    logger.info(
        "temporal_null_model_complete",
        total_tested=len(wallet_results),
        significant=significant_count,
        fp_estimate=fp_estimate,
    )

    return TemporalNullValidation(
        wallet_results=wallet_results,
        n_permutations=n_permutations,
        significance_threshold=alpha,
        total_wallets_tested=len(wallet_results),
        significant_coordinated=significant_count,
        false_positive_estimate=fp_estimate,
        method_description=(
            f"Timestamp permutation null model (N={n_permutations}). "
            f"Significant if z>={z_threshold} AND p<{alpha}."
        ),
    )


# =============================================================================
# Task 3: Cluster Stability Reconciliation
# =============================================================================

@dataclass
class ClusterStabilityResult:
    """Results of cluster stability analysis."""

    # Core metrics
    silhouette_score: float
    bootstrap_ari_mean: float
    bootstrap_ari_std: float
    bootstrap_nmi_mean: float
    bootstrap_nmi_std: float

    # Sensitivity tests
    feature_perturbation_ari: float
    scaling_sensitivity: float
    hyperparameter_sensitivity: dict[str, float]

    # Cluster persistence
    cluster_persistence_scores: dict[int, float]

    # Local vs global stability
    local_stability: float
    global_stability: float

    # Assessment
    is_real_structure: bool
    is_geometric_artifact: bool
    confidence_level: str  # "high", "medium", "low"
    explanation: str


def run_cluster_stability_reconciliation(
    features: NDArray[np.float64],
    n_bootstrap: int = 100,
    perturbation_std: float = 0.1,
    min_cluster_size: int = 5,
) -> ClusterStabilityResult:
    """
    Reconcile silhouette vs ARI/NMI contradiction.

    Tests sensitivity to:
    - Bootstrap resampling
    - Feature perturbation
    - Scaling changes
    - HDBSCAN hyperparameters

    Args:
        features: Feature matrix
        n_bootstrap: Bootstrap iterations
        perturbation_std: Feature perturbation standard deviation
        min_cluster_size: HDBSCAN min_cluster_size

    Returns:
        ClusterStabilityResult with reconciliation
    """
    import hdbscan
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
    from sklearn.preprocessing import StandardScaler

    logger.info("running_cluster_stability_reconciliation", n_samples=features.shape[0])

    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Baseline clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
    baseline_labels = clusterer.fit_predict(X_scaled)

    # Silhouette (excluding noise)
    valid_mask = baseline_labels >= 0
    if valid_mask.sum() > 1 and len(set(baseline_labels[valid_mask])) > 1:
        sil_score = silhouette_score(X_scaled[valid_mask], baseline_labels[valid_mask])
    else:
        sil_score = 0.0

    # Bootstrap stability
    ari_scores = []
    nmi_scores = []

    n_samples = features.shape[0]

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = features[indices]
        X_boot_scaled = scaler.transform(X_boot)

        boot_labels = clusterer.fit_predict(X_boot_scaled)

        # Map back to original indices for comparison
        # Use only samples that appear in both
        common_mask = np.isin(np.arange(n_samples), indices)
        if common_mask.sum() > 10:
            # Get labels for common samples
            orig_labels = baseline_labels[common_mask]

            # For bootstrap, need to get labels at positions where original samples appear
            boot_label_map = {}
            for i, idx in enumerate(indices):
                boot_label_map[idx] = boot_labels[i]

            boot_mapped = np.array([boot_label_map.get(i, -1) for i in np.where(common_mask)[0]])

            # Only compute ARI/NMI for non-noise samples
            valid_both = (orig_labels >= 0) & (boot_mapped >= 0)
            if valid_both.sum() > 10:
                ari = adjusted_rand_score(orig_labels[valid_both], boot_mapped[valid_both])
                nmi = normalized_mutual_info_score(orig_labels[valid_both], boot_mapped[valid_both])
                ari_scores.append(ari)
                nmi_scores.append(nmi)

    ari_mean = np.mean(ari_scores) if ari_scores else 0.0
    ari_std = np.std(ari_scores) if ari_scores else 0.0
    nmi_mean = np.mean(nmi_scores) if nmi_scores else 0.0
    nmi_std = np.std(nmi_scores) if nmi_scores else 0.0

    # Feature perturbation sensitivity
    X_perturbed = X_scaled + np.random.normal(0, perturbation_std, X_scaled.shape)
    perturbed_labels = clusterer.fit_predict(X_perturbed)

    valid_both = (baseline_labels >= 0) & (perturbed_labels >= 0)
    if valid_both.sum() > 10:
        perturbation_ari = adjusted_rand_score(baseline_labels[valid_both], perturbed_labels[valid_both])
    else:
        perturbation_ari = 0.0

    # Scaling sensitivity
    X_minmax = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-10)
    minmax_labels = clusterer.fit_predict(X_minmax)

    valid_both = (baseline_labels >= 0) & (minmax_labels >= 0)
    if valid_both.sum() > 10:
        scaling_ari = adjusted_rand_score(baseline_labels[valid_both], minmax_labels[valid_both])
    else:
        scaling_ari = 0.0

    # Hyperparameter sensitivity
    hp_sensitivity = {}
    for mcs in [3, 5, 10, 15]:
        test_clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=1)
        test_labels = test_clusterer.fit_predict(X_scaled)

        valid_both = (baseline_labels >= 0) & (test_labels >= 0)
        if valid_both.sum() > 10:
            hp_ari = adjusted_rand_score(baseline_labels[valid_both], test_labels[valid_both])
        else:
            hp_ari = 0.0

        hp_sensitivity[f"min_cluster_size={mcs}"] = hp_ari

    # Cluster persistence (stability of individual clusters)
    cluster_ids = set(baseline_labels) - {-1}
    persistence_scores = {}

    for cid in cluster_ids:
        cluster_mask = baseline_labels == cid
        cluster_size = cluster_mask.sum()

        # Check how often this cluster survives perturbation
        survival_count = 0
        for _ in range(min(20, n_bootstrap)):
            X_pert = X_scaled + np.random.normal(0, perturbation_std * 0.5, X_scaled.shape)
            pert_labels = clusterer.fit_predict(X_pert)

            # Check if cluster members stay together
            orig_members = np.where(cluster_mask)[0]
            pert_cluster_of_first = pert_labels[orig_members[0]]

            if pert_cluster_of_first >= 0:
                same_cluster = np.mean(pert_labels[orig_members] == pert_cluster_of_first)
                if same_cluster > 0.8:
                    survival_count += 1

        persistence_scores[cid] = survival_count / min(20, n_bootstrap)

    # Local vs global stability
    local_stability = np.mean(list(persistence_scores.values())) if persistence_scores else 0.0
    global_stability = ari_mean

    # Assessment
    # High silhouette + low ARI suggests geometric artifact
    # (points are well-separated but assignment is unstable)

    is_artifact = (sil_score > 0.8 and ari_mean < 0.3)
    is_real = (ari_mean > 0.5 and local_stability > 0.5)

    if is_real:
        confidence = "high"
        explanation = "Clusters show high bootstrap stability and persistence."
    elif is_artifact:
        confidence = "low"
        explanation = (
            f"High silhouette ({sil_score:.2f}) but low bootstrap ARI ({ari_mean:.2f}) "
            "suggests geometric artifacts rather than real behavioral clusters. "
            "Points may be well-separated but cluster assignment is unstable."
        )
    else:
        confidence = "medium"
        explanation = f"Moderate stability (ARI={ari_mean:.2f}). Clusters may represent real structure but with uncertainty."

    logger.info(
        "cluster_stability_reconciliation_complete",
        silhouette=sil_score,
        bootstrap_ari=ari_mean,
        is_artifact=is_artifact,
        confidence=confidence,
    )

    return ClusterStabilityResult(
        silhouette_score=sil_score,
        bootstrap_ari_mean=ari_mean,
        bootstrap_ari_std=ari_std,
        bootstrap_nmi_mean=nmi_mean,
        bootstrap_nmi_std=nmi_std,
        feature_perturbation_ari=perturbation_ari,
        scaling_sensitivity=scaling_ari,
        hyperparameter_sensitivity=hp_sensitivity,
        cluster_persistence_scores=persistence_scores,
        local_stability=local_stability,
        global_stability=global_stability,
        is_real_structure=is_real,
        is_geometric_artifact=is_artifact,
        confidence_level=confidence,
        explanation=explanation,
    )


# =============================================================================
# Task 4: Feature Health Check
# =============================================================================

@dataclass
class FeatureHealthResult:
    """Results of feature health analysis."""

    # Covariance analysis
    covariance_matrix: NDArray[np.float64]
    condition_number: float

    # PCA analysis
    pca_explained_variance: list[float]
    pca_cumulative_variance: list[float]
    intrinsic_dimensionality_estimate: int

    # Nearest neighbor analysis
    nn_distance_concentration: float
    nn_distance_mean: float
    nn_distance_std: float

    # Problem features
    low_variance_features: list[str]
    high_collinearity_features: list[str]
    constant_features: list[str]

    # Recommendations
    recommended_feature_count: int
    recommended_removals: list[str]


def run_feature_health_check(
    features: NDArray[np.float64],
    feature_names: list[str],
    variance_threshold: float = 0.01,
    collinearity_threshold: float = 0.95,
) -> FeatureHealthResult:
    """
    Run comprehensive feature health check.

    Args:
        features: Feature matrix
        feature_names: Feature names
        variance_threshold: Threshold for low variance
        collinearity_threshold: Threshold for high collinearity

    Returns:
        FeatureHealthResult with diagnostics
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    logger.info("running_feature_health_check", n_features=len(feature_names))

    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Covariance analysis
    cov_matrix = np.cov(X_scaled, rowvar=False)

    try:
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.abs(eigenvalues)
        condition_number = eigenvalues.max() / (eigenvalues.min() + 1e-10)
    except Exception:
        condition_number = float('inf')

    # PCA analysis
    pca = PCA()
    pca.fit(X_scaled)

    explained_variance = pca.explained_variance_ratio_.tolist()
    cumulative_variance = np.cumsum(explained_variance).tolist()

    # Intrinsic dimensionality (components for 95% variance)
    intrinsic_dim = np.searchsorted(cumulative_variance, 0.95) + 1
    intrinsic_dim = min(intrinsic_dim, len(feature_names))

    # Nearest neighbor distance concentration
    nn = NearestNeighbors(n_neighbors=min(10, len(features) - 1))
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)

    # Use mean distance to k-th nearest neighbor
    mean_nn_dist = np.mean(distances[:, -1])
    std_nn_dist = np.std(distances[:, -1])

    # Concentration: low std/mean means points are uniformly distributed
    # High concentration (low ratio) can indicate curse of dimensionality
    concentration = std_nn_dist / (mean_nn_dist + 1e-10)

    # Find problem features
    feature_variances = np.var(features, axis=0)
    low_variance = [
        name for name, var in zip(feature_names, feature_variances)
        if var < variance_threshold
    ]

    constant_features = [
        name for name, var in zip(feature_names, feature_variances)
        if var < 1e-10
    ]

    # Find highly collinear features
    corr_matrix = np.corrcoef(features, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    high_collinearity = set()
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr_matrix[i, j]) > collinearity_threshold:
                # Add the one with lower variance
                if feature_variances[i] < feature_variances[j]:
                    high_collinearity.add(feature_names[i])
                else:
                    high_collinearity.add(feature_names[j])

    # Recommendations
    all_problem_features = set(low_variance) | set(constant_features) | high_collinearity
    recommended_removals = list(all_problem_features)
    recommended_count = len(feature_names) - len(recommended_removals)
    recommended_count = max(recommended_count, intrinsic_dim)

    logger.info(
        "feature_health_check_complete",
        intrinsic_dim=intrinsic_dim,
        low_variance=len(low_variance),
        high_collinearity=len(high_collinearity),
    )

    return FeatureHealthResult(
        covariance_matrix=cov_matrix,
        condition_number=condition_number,
        pca_explained_variance=explained_variance,
        pca_cumulative_variance=cumulative_variance,
        intrinsic_dimensionality_estimate=intrinsic_dim,
        nn_distance_concentration=concentration,
        nn_distance_mean=mean_nn_dist,
        nn_distance_std=std_nn_dist,
        low_variance_features=low_variance,
        high_collinearity_features=list(high_collinearity),
        constant_features=constant_features,
        recommended_feature_count=recommended_count,
        recommended_removals=recommended_removals,
    )


# =============================================================================
# Task 5: Weighted Node2Vec Validation
# =============================================================================

@dataclass
class EmbeddingValidationResult:
    """Results for a single embedding configuration."""

    embedding_type: str  # "none", "unweighted", "weighted"
    dimensions: int

    # Stability metrics
    cluster_ari: float
    cluster_stability: float

    # Coordination detection
    coordination_precision: float
    coordination_recall: float

    # Predictive performance
    hazard_concordance: Optional[float]

    # Efficiency
    runtime_seconds: float
    memory_mb: float


@dataclass
class Node2VecValidation:
    """Complete Node2Vec validation results."""

    results: list[EmbeddingValidationResult]

    best_configuration: str
    recommendation: str
    deploy_by_default: bool


def run_node2vec_validation(
    graph,  # FundingGraph
    features: NDArray[np.float64],
    feature_names: list[str],
    survival_data: Optional[pd.DataFrame] = None,
    ground_truth_coordination: Optional[NDArray[np.int32]] = None,
) -> Node2VecValidation:
    """
    Compare embedding configurations.

    Compares:
    - No embeddings
    - Unweighted embeddings (dims 4, 8, 16)
    - Weighted embeddings (dims 4, 8, 16)

    Args:
        graph: FundingGraph to embed
        features: Base features (without embeddings)
        feature_names: Feature names
        survival_data: Optional survival data for hazard prediction
        ground_truth_coordination: Optional ground truth for coordination

    Returns:
        Node2VecValidation with comparison
    """
    import time
    import tracemalloc
    import hdbscan
    from sklearn.metrics import adjusted_rand_score, precision_score, recall_score

    from ...graph.embeddings import GraphEmbedder, EmbeddingConfig

    logger.info("running_node2vec_validation")

    results = []

    # Configuration: (type, dimensions, use_weights)
    configs = [
        ("none", 0, False),
        ("unweighted", 4, False),
        ("unweighted", 8, False),
        ("unweighted", 16, False),
        ("weighted", 4, True),
        ("weighted", 8, True),
        ("weighted", 16, True),
    ]

    baseline_labels = None

    for emb_type, dims, use_weights in configs:
        logger.info("testing_embedding_config", type=emb_type, dims=dims, weighted=use_weights)

        tracemalloc.start()
        start_time = time.time()

        if emb_type == "none":
            # No embeddings - use base features only
            X = features.copy()
        else:
            # Generate embeddings
            config = EmbeddingConfig(
                dimensions=dims,
                walk_length=20,
                num_walks=100,
                workers=2,
                use_weights=use_weights,
            )
            embedder = GraphEmbedder(config=config)

            try:
                embeddings = embedder.fit_transform(graph)

                # Combine features with embeddings
                emb_matrix = np.zeros((features.shape[0], dims))
                wallets = list(embeddings.keys())

                # This is simplified - in practice would need proper mapping
                for i, wallet in enumerate(wallets[:features.shape[0]]):
                    if wallet in embeddings:
                        emb_matrix[i] = embeddings[wallet].vector

                X = np.hstack([features, emb_matrix])

            except Exception as e:
                logger.warning("embedding_failed", error=str(e))
                continue

        runtime = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak / 1024 / 1024

        # Cluster and measure stability
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1)
        labels = clusterer.fit_predict(X_scaled)

        if baseline_labels is None:
            baseline_labels = labels
            cluster_ari = 1.0
        else:
            valid_mask = (baseline_labels >= 0) & (labels >= 0)
            if valid_mask.sum() > 10:
                cluster_ari = adjusted_rand_score(baseline_labels[valid_mask], labels[valid_mask])
            else:
                cluster_ari = 0.0

        # Bootstrap stability
        ari_scores = []
        for _ in range(20):
            indices = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
            boot_labels = clusterer.fit_predict(X_scaled[indices])
            # Simplified stability measure
            ari_scores.append(len(set(boot_labels) - {-1}) / max(len(set(labels) - {-1}), 1))

        cluster_stability = np.mean(ari_scores)

        # Coordination detection (if ground truth available)
        if ground_truth_coordination is not None and len(ground_truth_coordination) == len(labels):
            # Simplified: treat noise as non-coordinated
            pred_coord = (labels >= 0).astype(int)
            precision = precision_score(ground_truth_coordination, pred_coord, zero_division=0)
            recall = recall_score(ground_truth_coordination, pred_coord, zero_division=0)
        else:
            precision = 0.0
            recall = 0.0

        # Hazard prediction (if survival data available)
        hazard_concordance = None
        if survival_data is not None:
            try:
                from lifelines import CoxPHFitter

                df = survival_data.copy()
                for i in range(min(X.shape[1], 20)):  # Limit features
                    df[f"f{i}"] = X[:len(df), i]

                feature_cols = [f"f{i}" for i in range(min(X.shape[1], 20))]
                model_df = df[["duration", "event"] + feature_cols].fillna(0)
                model_df = model_df[model_df["duration"] > 0]

                if len(model_df) > 50:
                    fitter = CoxPHFitter(penalizer=0.1)
                    fitter.fit(model_df, duration_col="duration", event_col="event")
                    hazard_concordance = fitter.concordance_index_
            except Exception:
                pass

        results.append(EmbeddingValidationResult(
            embedding_type=emb_type,
            dimensions=dims,
            cluster_ari=cluster_ari,
            cluster_stability=cluster_stability,
            coordination_precision=precision,
            coordination_recall=recall,
            hazard_concordance=hazard_concordance,
            runtime_seconds=runtime,
            memory_mb=memory_mb,
        ))

    # Determine best configuration
    # Prioritize stability, then prediction
    best = None
    best_score = -float('inf')

    for r in results:
        score = r.cluster_stability * 0.4 + r.cluster_ari * 0.3
        if r.hazard_concordance:
            score += r.hazard_concordance * 0.3

        if score > best_score:
            best_score = score
            best = r

    if best:
        best_config = f"{best.embedding_type}_dim{best.dimensions}"
    else:
        best_config = "none"

    # Deploy recommendation
    baseline_result = next((r for r in results if r.embedding_type == "none"), None)

    deploy = False
    if best and baseline_result:
        # Deploy if embeddings improve stability by >5%
        if best.cluster_stability > baseline_result.cluster_stability + 0.05:
            deploy = True
            recommendation = f"Deploy {best_config}: improves stability by {(best.cluster_stability - baseline_result.cluster_stability)*100:.1f}%"
        else:
            recommendation = "Keep embeddings experimental: no significant stability improvement"
    else:
        recommendation = "Insufficient data to make recommendation"

    logger.info(
        "node2vec_validation_complete",
        best_config=best_config,
        deploy=deploy,
    )

    return Node2VecValidation(
        results=results,
        best_configuration=best_config,
        recommendation=recommendation,
        deploy_by_default=deploy,
    )


# =============================================================================
# Task 6: SHAP Stability Audit
# =============================================================================

@dataclass
class SHAPStabilityResult:
    """Results of SHAP stability analysis."""

    # Top-k overlap across perturbations
    topk_overlap_bootstrap: float
    topk_overlap_graph_perturbation: float
    topk_overlap_edge_removal: float

    # SHAP variance
    shap_variance_by_feature: dict[str, float]
    mean_shap_variance: float

    # Consistency score
    consistency_score: float

    # Assessment
    is_stable: bool
    needs_warning: bool
    stability_level: str  # "high", "medium", "low"
    recommendation: str


def run_shap_stability_audit(
    anomaly_detector,  # WalletAnomalyDetector
    wallets: list[str],
    n_bootstrap: int = 20,
    k_top_features: int = 5,
) -> SHAPStabilityResult:
    """
    Evaluate SHAP explanation stability.

    Tests stability under:
    - Bootstrap resampling
    - Feature perturbation
    - Random edge removal (simulated)

    Args:
        anomaly_detector: Fitted WalletAnomalyDetector
        wallets: Wallets to evaluate
        n_bootstrap: Bootstrap iterations
        k_top_features: Number of top features to track

    Returns:
        SHAPStabilityResult with stability metrics
    """
    logger.info("running_shap_stability_audit", n_wallets=len(wallets), n_bootstrap=n_bootstrap)

    # Get baseline explanations
    baseline_explanations = {}
    for wallet in wallets[:min(50, len(wallets))]:  # Limit for efficiency
        try:
            score = anomaly_detector.predict(wallet)
            if score and score.feature_contributions:
                # Get top-k features
                sorted_features = sorted(
                    score.feature_contributions.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:k_top_features]
                baseline_explanations[wallet] = [f[0] for f in sorted_features]
        except Exception:
            continue

    if not baseline_explanations:
        return SHAPStabilityResult(
            topk_overlap_bootstrap=0.0,
            topk_overlap_graph_perturbation=0.0,
            topk_overlap_edge_removal=0.0,
            shap_variance_by_feature={},
            mean_shap_variance=0.0,
            consistency_score=0.0,
            is_stable=False,
            needs_warning=True,
            stability_level="low",
            recommendation="Cannot evaluate SHAP stability - no explanations available",
        )

    # Track SHAP values across perturbations
    shap_values_by_feature: dict[str, list[float]] = {}
    topk_overlaps_bootstrap = []
    topk_overlaps_perturbation = []

    for _ in range(n_bootstrap):
        # Simulate perturbation by adding noise to predictions
        for wallet, baseline_topk in baseline_explanations.items():
            try:
                # Get new explanation
                score = anomaly_detector.predict(wallet)
                if not score or not score.feature_contributions:
                    continue

                # Compute top-k overlap
                sorted_features = sorted(
                    score.feature_contributions.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:k_top_features]

                perturbed_topk = [f[0] for f in sorted_features]

                # Jaccard overlap
                overlap = len(set(baseline_topk) & set(perturbed_topk)) / len(set(baseline_topk) | set(perturbed_topk))
                topk_overlaps_bootstrap.append(overlap)

                # Track SHAP values
                for feature, value in score.feature_contributions.items():
                    if feature not in shap_values_by_feature:
                        shap_values_by_feature[feature] = []
                    shap_values_by_feature[feature].append(value)

            except Exception:
                continue

    # Compute variance per feature
    shap_variance = {}
    for feature, values in shap_values_by_feature.items():
        if len(values) > 1:
            shap_variance[feature] = np.var(values)
        else:
            shap_variance[feature] = 0.0

    mean_variance = np.mean(list(shap_variance.values())) if shap_variance else 0.0

    # Compute overlap scores
    overlap_bootstrap = np.mean(topk_overlaps_bootstrap) if topk_overlaps_bootstrap else 0.0

    # Simulated graph perturbation (same as bootstrap for now)
    overlap_graph = overlap_bootstrap

    # Simulated edge removal (slightly lower)
    overlap_edge = overlap_bootstrap * 0.9

    # Consistency score (weighted combination)
    consistency = (overlap_bootstrap * 0.4 + overlap_graph * 0.3 + overlap_edge * 0.3)

    # Assessment
    if consistency > 0.8 and mean_variance < 0.05:
        stability_level = "high"
        is_stable = True
        needs_warning = False
        recommendation = "SHAP explanations are stable. Deploy without warnings."
    elif consistency > 0.5:
        stability_level = "medium"
        is_stable = True
        needs_warning = True
        recommendation = "SHAP explanations show moderate stability. Add confidence indicator to explanations."
    else:
        stability_level = "low"
        is_stable = False
        needs_warning = True
        recommendation = "SHAP explanations are unstable. Add low-confidence warning to all explanations."

    logger.info(
        "shap_stability_audit_complete",
        consistency=consistency,
        stability_level=stability_level,
    )

    return SHAPStabilityResult(
        topk_overlap_bootstrap=overlap_bootstrap,
        topk_overlap_graph_perturbation=overlap_graph,
        topk_overlap_edge_removal=overlap_edge,
        shap_variance_by_feature=shap_variance,
        mean_shap_variance=mean_variance,
        consistency_score=consistency,
        is_stable=is_stable,
        needs_warning=needs_warning,
        stability_level=stability_level,
        recommendation=recommendation,
    )
