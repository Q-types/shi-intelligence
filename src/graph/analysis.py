"""
Graph Analysis Functions.

Computes graph-based features for wallets:
- Centrality measures
- Community detection
- Shared funder analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import networkx as nx
import structlog

from .funding_graph import FundingGraph
from ..core.types import WalletAddress

logger = structlog.get_logger()


@dataclass
class WalletGraphFeatures:
    """Graph-derived features for a wallet."""

    wallet: WalletAddress
    in_degree: int
    out_degree: int
    eigenvector_centrality: float
    community_id: int | None
    shared_funder_count: int
    funding_tree_size: int


def compute_graph_features(
    graph: FundingGraph,
    wallets: list[WalletAddress],
) -> dict[WalletAddress, WalletGraphFeatures]:
    """
    Compute all graph features for a list of wallets.

    Args:
        graph: The funding graph
        wallets: Wallets to compute features for

    Returns:
        Dict mapping wallet -> WalletGraphFeatures
    """
    logger.info("computing_graph_features", wallet_count=len(wallets))

    # Compute centrality once for all nodes
    centrality = graph.compute_eigenvector_centrality()

    # Find shared funders
    shared_funders = graph.find_shared_funders(wallets)

    # Count shared funders per wallet
    wallet_shared_count: dict[str, int] = {w: 0 for w in wallets}
    for funder, funded_wallets in shared_funders.items():
        for w in funded_wallets:
            if w in wallet_shared_count:
                wallet_shared_count[w] += 1

    # Detect communities
    communities = detect_communities(graph)
    wallet_to_community: dict[str, int] = {}
    for comm_id, members in enumerate(communities):
        for w in members:
            wallet_to_community[w] = comm_id

    # Build features
    features = {}
    for wallet in wallets:
        ancestors = graph.get_ancestors(wallet, max_depth=3)
        features[wallet] = WalletGraphFeatures(
            wallet=wallet,
            in_degree=graph.get_in_degree(wallet),
            out_degree=graph.get_out_degree(wallet),
            eigenvector_centrality=centrality.get(wallet, 0.0),
            community_id=wallet_to_community.get(wallet),
            shared_funder_count=wallet_shared_count.get(wallet, 0),
            funding_tree_size=len(ancestors),
        )

    return features


def detect_communities(
    graph: FundingGraph,
    resolution: float = 1.0,
) -> list[Set[WalletAddress]]:
    """
    Detect communities in the funding graph.

    Uses Louvain algorithm on the undirected version.

    Args:
        graph: The funding graph
        resolution: Resolution parameter (higher = more communities)

    Returns:
        List of community sets
    """
    if graph.num_vertices < 2:
        return []

    # Get underlying NetworkX graph and convert to undirected
    undirected = graph._graph.to_undirected()

    try:
        # Use Louvain community detection
        from networkx.algorithms.community import louvain_communities

        communities = louvain_communities(
            undirected,
            resolution=resolution,
            seed=42,  # Reproducibility
        )
        return [set(c) for c in communities]

    except Exception as e:
        logger.warning("community_detection_failed", error=str(e))
        # Fall back to connected components
        return [set(c) for c in nx.connected_components(undirected)]


def find_shared_funders(
    graph: FundingGraph,
    wallets: list[WalletAddress],
    min_shared: int = 2,
    max_depth: int = 2,
) -> dict[WalletAddress, Set[WalletAddress]]:
    """
    Find wallets that funded multiple target wallets.

    Args:
        graph: The funding graph
        wallets: Target wallets to analyze
        min_shared: Minimum wallets a funder must have funded
        max_depth: How far up the funding tree to look

    Returns:
        Dict mapping funder -> set of wallets it funded
    """
    shared = graph.find_shared_funders(wallets, max_depth)

    # Filter by minimum
    return {
        funder: funded
        for funder, funded in shared.items()
        if len(funded) >= min_shared
    }


def compute_funding_entropy(graph: FundingGraph) -> float:
    """
    Compute entropy of the funding distribution.

    Higher entropy = more distributed funding sources.
    Lower entropy = concentrated funding (potential Sybil indicator).
    """
    import math

    if graph.num_vertices == 0:
        return 0.0

    # Count how many wallets each funder funded
    funder_counts: dict[str, int] = {}
    for node in graph._wallet_set:
        funders = graph.get_funders(node)
        for f in funders:
            funder_counts[f] = funder_counts.get(f, 0) + 1

    if not funder_counts:
        return 0.0

    total = sum(funder_counts.values())
    probs = [c / total for c in funder_counts.values()]

    # Shannon entropy
    entropy = -sum(p * math.log(p) for p in probs if p > 0)

    return entropy


def detect_funding_clusters(
    graph: FundingGraph,
    wallets: list[WalletAddress],
    similarity_threshold: float = 0.5,
) -> list[Set[WalletAddress]]:
    """
    Detect clusters of wallets with similar funding patterns.

    Wallets are similar if they share funders.

    Args:
        graph: The funding graph
        wallets: Wallets to cluster
        similarity_threshold: Jaccard similarity threshold

    Returns:
        List of cluster sets
    """
    if len(wallets) < 2:
        return [set(wallets)]

    # Get ancestor sets for each wallet
    ancestor_sets: dict[str, Set[str]] = {}
    for w in wallets:
        ancestor_sets[w] = graph.get_ancestors(w, max_depth=2)

    # Build similarity graph
    sim_graph = nx.Graph()
    for w in wallets:
        sim_graph.add_node(w)

    # Add edges for similar wallets
    wallet_list = list(wallets)
    for i, w1 in enumerate(wallet_list):
        for w2 in wallet_list[i + 1 :]:
            a1, a2 = ancestor_sets[w1], ancestor_sets[w2]
            if a1 and a2:
                # Jaccard similarity
                intersection = len(a1 & a2)
                union = len(a1 | a2)
                similarity = intersection / union if union > 0 else 0

                if similarity >= similarity_threshold:
                    sim_graph.add_edge(w1, w2, weight=similarity)

    # Find connected components in similarity graph
    clusters = [set(c) for c in nx.connected_components(sim_graph)]

    logger.info(
        "funding_clusters_detected",
        wallet_count=len(wallets),
        cluster_count=len(clusters),
    )

    return clusters
