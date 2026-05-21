# Graph Intelligence Deployment Decision

**Generated**: 2026-05-21T08:24:46.780618+00:00

## Executive Summary

| Component | Status | Action |
|-----------|--------|--------|
| Feature Redundancy | ✓ OK | Feature set is acceptable |
| Temporal Coordination | ⚠️ WEAK | No statistically significant detections |
| Cluster Stability | ⚠️ MODERATE | Use with confidence intervals |
| Feature Health | ✓ OK | Features are healthy |
| Node2Vec Embeddings | ⚠️ EXPERIMENTAL | Do not deploy by default |
| SHAP Explanations | ✓ STABLE | Deploy without warnings |

## Detailed Findings

### Cluster Stability Investigation

The silhouette score (0.03) vs bootstrap ARI (0.00) discrepancy has been investigated.

**Conclusion**: Moderate stability (ARI=0.00). Clusters may represent real structure but with uncertainty.

### Temporal Coordination Validation

Null model testing with 100 permutations shows 0 wallets with statistically significant coordination.

## Hard Rules Compliance

| Rule | Status |
|------|--------|
| PDR metrics unchanged | ✓ Compliant |
| Graph features improve validation | See component statuses |
| Temporal coordination requires significance | ✓ Implemented |
| Embeddings experimental until validated | ✓ Compliant |
| Stability > silhouette | ✓ Evaluated |

## Final Recommendation

**DO NOT DEPLOY**: Multiple components fail validation. Requires remediation.