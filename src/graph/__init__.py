"""
Graph Analysis Module for SHI.

Provides funding graph construction and analysis:
- Directed funding graph G=(V,E)
- Community detection
- Centrality metrics
- Shared funder detection
"""

from .funding_graph import FundingGraph
from .analysis import (
    compute_graph_features,
    detect_communities,
    find_shared_funders,
)

__all__ = [
    "FundingGraph",
    "compute_graph_features",
    "detect_communities",
    "find_shared_funders",
]
