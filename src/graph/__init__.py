"""
Graph Analysis Module for SHI.

Provides funding graph construction and analysis:
- Directed funding graph G=(V,E)
- Community detection
- Centrality metrics
- Shared funder detection
- Node2Vec embeddings
- Wallet similarity detection
- Dynamic network metrics
- Anomaly detection
"""

from .funding_graph import FundingGraph
from .analysis import (
    compute_graph_features,
    detect_communities,
    find_shared_funders,
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
    "compute_graph_features",
    "detect_communities",
    "find_shared_funders",
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
