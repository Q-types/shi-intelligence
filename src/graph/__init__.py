"""
Graph Analysis Module for SHI.

Provides funding graph construction and analysis:
- Directed funding graph G=(V,E)
- Community detection
- Centrality metrics (eigenvector, PageRank, betweenness)
- Shared funder detection
- Node2Vec embeddings
- Wallet similarity detection
- Dynamic network metrics
- Anomaly detection
- Temporal coordination patterns
- Weighted graph features
"""

from .funding_graph import FundingGraph
from .analysis import (
    WalletGraphFeatures,
    compute_graph_features,
    detect_communities,
    find_shared_funders,
    compute_funding_entropy,
    detect_funding_clusters,
)
from .weighted_features import (
    WeightedGraphFeatures,
    compute_weighted_graph_features,
    enrich_wallet_features,
)
from .temporal_patterns import (
    TemporalCoordinationResult,
    TemporalCluster,
    detect_temporal_coordination,
    find_synchronized_funding_groups,
    compute_funding_velocity,
)
from .embeddings import (
    GraphEmbedder,
    EmbeddingConfig,
    WalletEmbedding,
)
from .similarity import (
    WalletSimilarityDetector,
    SimilarityScore,
    CoordinatedCluster,
)
from .dynamics import (
    DynamicNetworkAnalyzer,
    NetworkSnapshot,
    NetworkDynamics,
)
from .anomaly import (
    WalletAnomalyDetector,
    AnomalyScore,
    AnomalyConfig,
)

__all__ = [
    # Funding graph
    "FundingGraph",
    # Graph features (including PageRank, betweenness)
    "WalletGraphFeatures",
    "compute_graph_features",
    "detect_communities",
    "find_shared_funders",
    "compute_funding_entropy",
    "detect_funding_clusters",
    # Weighted graph features
    "WeightedGraphFeatures",
    "compute_weighted_graph_features",
    "enrich_wallet_features",
    # Temporal coordination
    "TemporalCoordinationResult",
    "TemporalCluster",
    "detect_temporal_coordination",
    "find_synchronized_funding_groups",
    "compute_funding_velocity",
    # Embeddings
    "GraphEmbedder",
    "EmbeddingConfig",
    "WalletEmbedding",
    # Similarity
    "WalletSimilarityDetector",
    "SimilarityScore",
    "CoordinatedCluster",
    # Dynamics
    "DynamicNetworkAnalyzer",
    "NetworkSnapshot",
    "NetworkDynamics",
    # Anomaly
    "WalletAnomalyDetector",
    "AnomalyScore",
    "AnomalyConfig",
]
