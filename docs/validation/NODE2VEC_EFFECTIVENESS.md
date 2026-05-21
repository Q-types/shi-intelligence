# Node2Vec Effectiveness Validation

**Generated**: 2026-05-21T08:24:44.558455+00:00

## Configurations Tested

| Type | Dimensions | Cluster ARI | Stability | Precision | Recall | Concordance | Runtime (s) | Memory (MB) |
|------|------------|-------------|-----------|-----------|--------|-------------|-------------|-------------|
| none | 0 | 1.000 | 1.433 | 0.029 | 0.057 | 0.616 | 0.00 | 0.0 |
| unweighted | 4 | 0.000 | 1.925 | 0.101 | 0.486 | 0.616 | 2.91 | 1.4 |
| unweighted | 8 | 0.000 | 1.950 | 0.085 | 0.400 | 0.616 | 1.31 | 1.2 |
| unweighted | 16 | 0.247 | 1.283 | 0.000 | 0.000 | 0.616 | 1.29 | 1.2 |
| weighted | 4 | 0.000 | 2.125 | 0.112 | 0.571 | 0.616 | 1.33 | 1.3 |
| weighted | 8 | 0.000 | 1.800 | 0.147 | 0.800 | 0.616 | 1.32 | 1.3 |
| weighted | 16 | 0.247 | 1.550 | 0.000 | 0.000 | 0.616 | 1.33 | 1.3 |

## Best Configuration

**none_dim0**

## Recommendation

Keep embeddings experimental: no significant stability improvement

## Deployment Decision

**Deploy by Default**: No

### Reason

Embeddings do not provide sufficient stability improvement over baseline features.
Keep embeddings as experimental/optional feature.