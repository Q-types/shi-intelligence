"""
Dynamic Network Metrics for SHI.

Tracks network evolution over time:
- Modularity(t), Density(t), Centralization(t)
- Community emergence/fragmentation
- Graph evolution velocity
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import structlog

from .funding_graph import FundingGraph

logger = structlog.get_logger()


@dataclass
class NetworkSnapshot:
    """Snapshot of network metrics at a point in time."""

    timestamp: datetime
    num_nodes: int
    num_edges: int
    density: float
    modularity: float
    centralization: float
    avg_clustering_coefficient: float
    num_communities: int
    largest_component_size: int


@dataclass
class NetworkDynamics:
    """Time-series of network metrics showing evolution."""

    snapshots: List[NetworkSnapshot]
    velocity_density: float  # dDensity/dt
    velocity_modularity: float  # dModularity/dt
    velocity_centralization: float  # dCentralization/dt
    is_fragmenting: bool
    is_consolidating: bool


class DynamicNetworkAnalyzer:
    """
    Analyzes network evolution over time.

    Tracks structural changes in the funding graph:
    - Community dynamics (emergence, merging, splitting)
    - Centralization trends
    - Network cohesion
    """

    def __init__(self):
        """Initialize analyzer."""
        self.snapshots: List[NetworkSnapshot] = []

    def compute_snapshot(
        self,
        graph: FundingGraph,
        timestamp: Optional[datetime] = None,
    ) -> NetworkSnapshot:
        """
        Compute network metrics snapshot.

        Args:
            graph: Funding graph
            timestamp: Timestamp for snapshot (default: now)

        Returns:
            NetworkSnapshot with current metrics
        """
        timestamp = timestamp or datetime.now()

        if graph.num_vertices == 0:
            return NetworkSnapshot(
                timestamp=timestamp,
                num_nodes=0,
                num_edges=0,
                density=0.0,
                modularity=0.0,
                centralization=0.0,
                avg_clustering_coefficient=0.0,
                num_communities=0,
                largest_component_size=0,
            )

        nx_graph = graph._graph
        undirected = nx_graph.to_undirected()

        # Basic metrics
        num_nodes = graph.num_vertices
        num_edges = graph.num_edges
        density = nx.density(undirected)

        # Clustering coefficient
        avg_clustering = nx.average_clustering(undirected)

        # Modularity (requires community detection)
        communities = self._detect_communities(undirected)
        num_communities = len(communities)

        modularity = 0.0
        if num_communities > 1:
            try:
                modularity = nx.community.modularity(undirected, communities)
            except Exception as e:
                logger.warning("modularity_computation_failed", error=str(e))

        # Centralization (based on degree centrality)
        centralization = self._compute_centralization(undirected)

        # Largest component size
        components = list(nx.connected_components(undirected))
        largest_component_size = len(max(components, key=len)) if components else 0

        snapshot = NetworkSnapshot(
            timestamp=timestamp,
            num_nodes=num_nodes,
            num_edges=num_edges,
            density=density,
            modularity=modularity,
            centralization=centralization,
            avg_clustering_coefficient=avg_clustering,
            num_communities=num_communities,
            largest_component_size=largest_component_size,
        )

        self.snapshots.append(snapshot)

        logger.info(
            "network_snapshot",
            nodes=num_nodes,
            edges=num_edges,
            density=f"{density:.3f}",
            modularity=f"{modularity:.3f}",
            communities=num_communities,
        )

        return snapshot

    def _detect_communities(self, graph: nx.Graph) -> List[set]:
        """
        Detect communities in graph.

        Args:
            graph: NetworkX graph

        Returns:
            List of community sets
        """
        if graph.number_of_nodes() < 2:
            return []

        try:
            from networkx.algorithms.community import louvain_communities

            communities = louvain_communities(graph, seed=42)
            return [set(c) for c in communities]
        except Exception as e:
            logger.warning("community_detection_failed", error=str(e))
            # Fall back to connected components
            return [set(c) for c in nx.connected_components(graph)]

    def _compute_centralization(self, graph: nx.Graph) -> float:
        """
        Compute network centralization.

        Measures how much the network centers around a few key nodes.

        Args:
            graph: NetworkX graph

        Returns:
            Centralization score in [0, 1]
        """
        if graph.number_of_nodes() <= 1:
            return 0.0

        # Degree centrality
        centralities = nx.degree_centrality(graph)
        values = list(centralities.values())

        if not values:
            return 0.0

        max_centrality = max(values)
        n = len(values)

        # Freeman's centralization formula
        numerator = sum(max_centrality - c for c in values)
        denominator = (n - 1) * (n - 2) if n > 2 else 1

        centralization = numerator / denominator if denominator > 0 else 0.0

        return min(centralization, 1.0)

    def compute_dynamics(
        self,
        window_size: int = 10,
    ) -> Optional[NetworkDynamics]:
        """
        Compute network dynamics from snapshots.

        Args:
            window_size: Number of recent snapshots to analyze

        Returns:
            NetworkDynamics object or None if insufficient data
        """
        if len(self.snapshots) < 2:
            logger.warning("insufficient_snapshots_for_dynamics", count=len(self.snapshots))
            return None

        # Use most recent snapshots
        recent_snapshots = self.snapshots[-window_size:]

        if len(recent_snapshots) < 2:
            return None

        # Compute velocities (rate of change per day)
        density_values = [s.density for s in recent_snapshots]
        modularity_values = [s.modularity for s in recent_snapshots]
        centralization_values = [s.centralization for s in recent_snapshots]

        # Time differences in days
        time_diffs = []
        for i in range(1, len(recent_snapshots)):
            dt = (recent_snapshots[i].timestamp - recent_snapshots[i - 1].timestamp).total_seconds()
            time_diffs.append(dt / 86400)  # Convert to days

        if not time_diffs:
            return None

        # Compute velocities using finite differences
        velocity_density = self._compute_velocity(density_values, time_diffs)
        velocity_modularity = self._compute_velocity(modularity_values, time_diffs)
        velocity_centralization = self._compute_velocity(centralization_values, time_diffs)

        # Determine trends
        is_fragmenting = velocity_modularity > 0.01 and velocity_density < -0.001
        is_consolidating = velocity_centralization > 0.01 and velocity_density > 0.001

        dynamics = NetworkDynamics(
            snapshots=recent_snapshots,
            velocity_density=velocity_density,
            velocity_modularity=velocity_modularity,
            velocity_centralization=velocity_centralization,
            is_fragmenting=is_fragmenting,
            is_consolidating=is_consolidating,
        )

        logger.info(
            "network_dynamics_computed",
            v_density=f"{velocity_density:.6f}",
            v_modularity=f"{velocity_modularity:.6f}",
            v_centralization=f"{velocity_centralization:.6f}",
            fragmenting=is_fragmenting,
            consolidating=is_consolidating,
        )

        return dynamics

    def _compute_velocity(
        self,
        values: List[float],
        time_diffs: List[float],
    ) -> float:
        """
        Compute velocity (rate of change) using finite differences.

        Args:
            values: Metric values
            time_diffs: Time differences between values (in days)

        Returns:
            Average velocity (per day)
        """
        if len(values) < 2 or len(time_diffs) == 0:
            return 0.0

        # Compute differences
        diffs = []
        for i in range(1, len(values)):
            if time_diffs[i - 1] > 0:
                diff = (values[i] - values[i - 1]) / time_diffs[i - 1]
                diffs.append(diff)

        if not diffs:
            return 0.0

        # Return mean velocity
        return np.mean(diffs)

    def detect_community_events(
        self,
        min_snapshots: int = 3,
    ) -> List[Dict[str, any]]:
        """
        Detect community emergence and fragmentation events.

        Args:
            min_snapshots: Minimum snapshots needed

        Returns:
            List of event dictionaries
        """
        if len(self.snapshots) < min_snapshots:
            return []

        events = []

        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i - 1]
            curr = self.snapshots[i]

            # Community emergence (increase in communities)
            if curr.num_communities > prev.num_communities:
                events.append(
                    {
                        "type": "community_emergence",
                        "timestamp": curr.timestamp,
                        "communities_before": prev.num_communities,
                        "communities_after": curr.num_communities,
                        "new_communities": curr.num_communities - prev.num_communities,
                    }
                )

            # Community consolidation (decrease in communities)
            elif curr.num_communities < prev.num_communities:
                events.append(
                    {
                        "type": "community_consolidation",
                        "timestamp": curr.timestamp,
                        "communities_before": prev.num_communities,
                        "communities_after": curr.num_communities,
                        "merged_communities": prev.num_communities - curr.num_communities,
                    }
                )

            # Network fragmentation (significant density drop + modularity increase)
            density_drop = prev.density - curr.density
            modularity_increase = curr.modularity - prev.modularity

            if density_drop > 0.05 and modularity_increase > 0.1:
                events.append(
                    {
                        "type": "network_fragmentation",
                        "timestamp": curr.timestamp,
                        "density_change": -density_drop,
                        "modularity_change": modularity_increase,
                    }
                )

        logger.info("community_events_detected", count=len(events))

        return events

    def get_network_health_score(self) -> Optional[float]:
        """
        Compute network health score.

        Higher score = healthier, more cohesive network.

        Returns:
            Health score in [0, 1] or None if no snapshots
        """
        if not self.snapshots:
            return None

        latest = self.snapshots[-1]

        # Health factors:
        # - Higher density is better (more connections)
        # - Lower modularity is better (less fragmented)
        # - Moderate centralization is best (not too centralized, not too decentralized)
        # - Higher clustering is better (local cohesion)

        density_score = min(latest.density / 0.5, 1.0)  # Normalize to [0, 1]
        modularity_penalty = latest.modularity  # Lower is better
        centralization_score = 1.0 - abs(latest.centralization - 0.5) * 2  # Penalize extremes
        clustering_score = latest.avg_clustering_coefficient

        # Weighted average
        health = (
            0.3 * density_score
            + 0.2 * (1.0 - modularity_penalty)
            + 0.3 * centralization_score
            + 0.2 * clustering_score
        )

        return min(max(health, 0.0), 1.0)

    def export_time_series(self) -> Dict[str, List[any]]:
        """
        Export snapshots as time series data.

        Returns:
            Dict with metric name -> list of values
        """
        if not self.snapshots:
            return {}

        return {
            "timestamps": [s.timestamp.isoformat() for s in self.snapshots],
            "num_nodes": [s.num_nodes for s in self.snapshots],
            "num_edges": [s.num_edges for s in self.snapshots],
            "density": [s.density for s in self.snapshots],
            "modularity": [s.modularity for s in self.snapshots],
            "centralization": [s.centralization for s in self.snapshots],
            "avg_clustering": [s.avg_clustering_coefficient for s in self.snapshots],
            "num_communities": [s.num_communities for s in self.snapshots],
        }
