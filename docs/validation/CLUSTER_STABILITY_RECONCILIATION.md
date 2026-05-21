# Cluster Stability Reconciliation

**Generated**: 2026-05-21T08:24:34.322529+00:00

## The Contradiction

- **Silhouette Score**: 0.026
- **Bootstrap ARI Mean**: 0.002 ± 0.083
- **Bootstrap NMI Mean**: 0.076 ± 0.142

## Sensitivity Analysis

### Feature Perturbation
- ARI after adding Gaussian noise: 0.000

### Scaling Sensitivity
- ARI with MinMax scaling: -0.045

### Hyperparameter Sensitivity

| min_cluster_size | ARI vs Baseline |
|-----------------|-----------------|
| min_cluster_size=3 | 0.000 |
| min_cluster_size=5 | 1.000 |
| min_cluster_size=10 | 0.000 |
| min_cluster_size=15 | 0.000 |

## Cluster Persistence

| Cluster ID | Persistence Score |
|------------|-------------------|
| 0 | 0.45 |
| 1 | 0.75 |
| 2 | 0.60 |

## Local vs Global Stability

- **Local Stability** (individual cluster persistence): 0.600
- **Global Stability** (overall assignment consistency): 0.002

## Assessment

**Confidence Level**: MEDIUM

**Is Real Structure**: No
**Is Geometric Artifact**: No

### Explanation

Moderate stability (ARI=0.00). Clusters may represent real structure but with uncertainty.

## Recommendation

Clusters show moderate stability. Use with caution and consider:
- Adding confidence intervals to cluster assignments
- Using soft clustering probabilities