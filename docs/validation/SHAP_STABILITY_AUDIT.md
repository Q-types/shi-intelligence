# SHAP Stability Audit

**Generated**: 2026-05-21T08:24:46.780249+00:00

## Stability Metrics

- **Top-K Overlap (Bootstrap)**: 1.000
- **Top-K Overlap (Graph Perturbation)**: 1.000
- **Top-K Overlap (Edge Removal)**: 0.900
- **Mean SHAP Variance**: 0.0000
- **Overall Consistency Score**: 0.970

## SHAP Variance by Feature

| Feature | Variance |
|---------|----------|
| in_degree | 0.0000 |
| ancestor_count | 0.0000 |
| funder_count | 0.0000 |
| emb_2 | 0.0000 |
| emb_0 | 0.0000 |
| emb_3 | 0.0000 |
| emb_1 | 0.0000 |
| emb_4 | 0.0000 |
| emb_6 | 0.0000 |
| emb_5 | 0.0000 |
| emb_7 | 0.0000 |
| out_degree | 0.0000 |
| funded_count | 0.0000 |
| funding_ratio | 0.0000 |

## Assessment

**Stability Level**: HIGH
**Is Stable**: Yes
**Needs Warning**: No

## Recommendation

SHAP explanations are stable. Deploy without warnings.
