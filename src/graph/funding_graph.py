"""
Funding Graph Construction.

Builds directed graph G=(V,E) where:
- V = wallet addresses
- E = funding transfers
"""

from __future__ import annotations

import pickle
from datetime import datetime
from typing import Iterator, Set

import networkx as nx
import structlog

from ..core.types import WalletAddress, FundingEdge

logger = structlog.get_logger()


class FundingGraph:
    """
    Directed funding graph for wallet relationship analysis.

    Edges represent SOL transfers that funded wallet creation.
    """

    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._wallet_set: Set[str] = set()
        self._created_at = datetime.utcnow()

    @property
    def num_vertices(self) -> int:
        """Number of wallet nodes."""
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        """Number of funding edges."""
        return self._graph.number_of_edges()

    def add_wallet(self, wallet: WalletAddress) -> None:
        """Add a wallet node to the graph."""
        if wallet not in self._wallet_set:
            self._graph.add_node(wallet)
            self._wallet_set.add(wallet)

    def add_funding_edge(self, edge: FundingEdge) -> None:
        """
        Add a funding relationship.

        Args:
            edge: FundingEdge with source (funder) and target (funded)
        """
        # Ensure nodes exist
        self.add_wallet(edge.source)
        self.add_wallet(edge.target)

        # Add edge with metadata
        self._graph.add_edge(
            edge.source,
            edge.target,
            amount=edge.amount_lamports,
            timestamp=edge.timestamp.isoformat(),
            signature=edge.signature,
        )

    def add_edges_from_list(self, edges: list[FundingEdge]) -> None:
        """Bulk add funding edges."""
        for edge in edges:
            self.add_funding_edge(edge)

        logger.info(
            "edges_added",
            count=len(edges),
            total_nodes=self.num_vertices,
            total_edges=self.num_edges,
        )

    def get_in_degree(self, wallet: WalletAddress) -> int:
        """Get number of incoming funding edges."""
        if wallet not in self._wallet_set:
            return 0
        return self._graph.in_degree(wallet)

    def get_out_degree(self, wallet: WalletAddress) -> int:
        """Get number of outgoing funding edges (wallets funded by this one)."""
        if wallet not in self._wallet_set:
            return 0
        return self._graph.out_degree(wallet)

    def get_funders(self, wallet: WalletAddress) -> list[WalletAddress]:
        """Get all wallets that funded this wallet."""
        if wallet not in self._wallet_set:
            return []
        return list(self._graph.predecessors(wallet))

    def get_funded_by(self, wallet: WalletAddress) -> list[WalletAddress]:
        """Get all wallets funded by this wallet."""
        if wallet not in self._wallet_set:
            return []
        return list(self._graph.successors(wallet))

    def get_ancestors(
        self,
        wallet: WalletAddress,
        max_depth: int = 3,
    ) -> Set[WalletAddress]:
        """
        Get all upstream funders up to max_depth.

        This finds the "funding tree" above a wallet.
        """
        if wallet not in self._wallet_set:
            return set()

        ancestors = set()
        current_level = {wallet}

        for _ in range(max_depth):
            next_level = set()
            for w in current_level:
                funders = self.get_funders(w)
                for f in funders:
                    if f not in ancestors and f != wallet:
                        ancestors.add(f)
                        next_level.add(f)
            current_level = next_level
            if not current_level:
                break

        return ancestors

    def find_shared_funders(
        self,
        wallets: list[WalletAddress],
        max_depth: int = 2,
    ) -> dict[WalletAddress, Set[WalletAddress]]:
        """
        Find funders shared by multiple wallets.

        Returns:
            Dict mapping funder -> set of wallets it funded
        """
        funder_to_funded: dict[WalletAddress, Set[WalletAddress]] = {}

        for wallet in wallets:
            ancestors = self.get_ancestors(wallet, max_depth)
            for funder in ancestors:
                if funder not in funder_to_funded:
                    funder_to_funded[funder] = set()
                funder_to_funded[funder].add(wallet)

        # Filter to funders that funded multiple wallets
        shared = {
            funder: funded
            for funder, funded in funder_to_funded.items()
            if len(funded) > 1
        }

        return shared

    def get_dominant_funder(
        self,
        wallets: list[WalletAddress],
        max_depth: int = 2,
    ) -> tuple[WalletAddress | None, int]:
        """
        Find the funder that funded the most wallets.

        Returns:
            (funder_address, count) or (None, 0) if no shared funders
        """
        shared = self.find_shared_funders(wallets, max_depth)

        if not shared:
            return None, 0

        dominant = max(shared.items(), key=lambda x: len(x[1]))
        return dominant[0], len(dominant[1])

    def compute_eigenvector_centrality(
        self,
        max_iter: int = 100,
    ) -> dict[WalletAddress, float]:
        """
        Compute eigenvector centrality for all nodes.

        Higher values = more "important" in the funding network.
        """
        if self.num_vertices == 0:
            return {}

        try:
            return nx.eigenvector_centrality(
                self._graph,
                max_iter=max_iter,
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("eigenvector_centrality_failed_to_converge")
            # Fall back to degree centrality
            return nx.in_degree_centrality(self._graph)

    def get_connected_components(self) -> list[Set[WalletAddress]]:
        """Get weakly connected components."""
        # Convert to undirected for component analysis
        undirected = self._graph.to_undirected()
        return [set(c) for c in nx.connected_components(undirected)]

    def get_subgraph(self, wallets: Set[WalletAddress]) -> "FundingGraph":
        """Extract subgraph containing only specified wallets."""
        subgraph = FundingGraph()
        subgraph._graph = self._graph.subgraph(wallets).copy()
        subgraph._wallet_set = wallets & self._wallet_set
        return subgraph

    def serialize(self) -> bytes:
        """Serialize graph for caching."""
        return pickle.dumps(self._graph)

    @classmethod
    def deserialize(cls, data: bytes) -> "FundingGraph":
        """Deserialize cached graph."""
        graph = cls()
        graph._graph = pickle.loads(data)
        graph._wallet_set = set(graph._graph.nodes())
        return graph

    def to_dict(self) -> dict:
        """Export graph as dictionary for JSON serialization."""
        return {
            "nodes": list(self._wallet_set),
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "amount": d.get("amount", 0),
                    "timestamp": d.get("timestamp"),
                }
                for u, v, d in self._graph.edges(data=True)
            ],
            "created_at": self._created_at.isoformat(),
        }
