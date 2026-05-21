# Graph Feature Redundancy Audit

**Generated**: 2026-05-21T08:24:33.682191+00:00

## Summary

- **Total Features Analyzed**: 24
- **Highly Correlated Pairs**: 3
- **Redundant Features**: 2
- **Recommended Removals**: 2

## Highly Correlated Feature Pairs

| Feature 1 | Feature 2 | Pearson Correlation |
|-----------|-----------|---------------------|
| shared_funder_count | temporal_sync_score | 0.848 |
| funding_hhi | temporal_sync_score | 0.845 |
| shared_funder_count | funding_hhi | 0.820 |

## Variance Inflation Factors (VIF)

| Feature | VIF | Assessment |
|---------|-----|------------|
| temporal_sync_score | 5.45 | Moderate |
| shared_funder_count | 4.43 | OK |
| funding_hhi | 4.42 | OK |
| funding_time_spread_hours | 1.75 | OK |
| position_volatility | 1.22 | OK |
| pagerank | 1.21 | OK |
| entry_time_relative | 1.20 | OK |
| in_degree | 1.19 | OK |
| largest_funder_share | 1.15 | OK |
| holding_duration | 1.15 | OK |
| share | 1.15 | OK |
| delta_balance_7d | 1.14 | OK |
| out_degree | 1.14 | OK |
| weighted_out_degree | 1.14 | OK |
| total_funding_received | 1.13 | OK |
| eigenvector_centrality | 1.12 | OK |
| trade_count | 1.12 | OK |
| burstiness | 1.12 | OK |
| delta_balance_30d | 1.12 | OK |
| funding_burst_score | 1.11 | OK |

## Feature Clustering (Dendrogram)

Features grouped by structural similarity:

- **Cluster 0**: share
- **Cluster 1**: entry_time_relative
- **Cluster 2**: holding_duration
- **Cluster 3**: position_volatility
- **Cluster 4**: funding_time_spread_hours
- **Cluster 5**: delta_balance_7d
- **Cluster 6**: delta_balance_30d
- **Cluster 7**: trade_count
- **Cluster 8**: burstiness
- **Cluster 9**: swap_frequency
- **Cluster 10**: lp_interaction_ratio
- **Cluster 11**: in_degree
- **Cluster 12**: out_degree
- **Cluster 13**: eigenvector_centrality
- **Cluster 14**: shared_funder_count
- **Cluster 15**: pagerank
- **Cluster 16**: betweenness_centrality
- **Cluster 17**: total_funding_received
- **Cluster 18**: largest_funder_share
- **Cluster 19**: funding_hhi
- **Cluster 20**: funding_burst_score
- **Cluster 21**: weighted_in_degree
- **Cluster 22**: weighted_out_degree
- **Cluster 23**: temporal_sync_score

## Redundant Features

Features recommended for removal due to high correlation or multicollinearity:

- `temporal_sync_score`
- `shared_funder_count`

## Recommendation

Consider removing 2 features to reduce redundancy:
- `temporal_sync_score`
- `shared_funder_count`