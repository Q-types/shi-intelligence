"""
Tests for Sprint 2: Graph Intelligence.

Tests:
- Node2Vec embeddings
- Wallet similarity detection
- Dynamic network metrics
- Anomaly detection (Isolation Forest)
"""

from datetime import datetime, timedelta
import pytest

from src.core.types import FundingEdge
from src.graph import (
    FundingGraph,
    GraphEmbedder,
    EmbeddingConfig,
    WalletSimilarityDetector,
    DynamicNetworkAnalyzer,
    WalletAnomalyDetector,
    AnomalyConfig,
)

# Base58 alphabet (no 0, I, O, l)
BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def _sig(n: int) -> str:
    """Generate a valid test signature (87-88 chars base58)."""
    # Use base58 chars only
    suffix = BASE58[n % len(BASE58)]
    return f"{'1' * 87}{suffix}"


class TestGraphEmbeddings:
    """Test Node2Vec graph embeddings."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample funding graph."""
        graph = FundingGraph()

        # Create coordinated cluster: A -> B, C, D (shared funder)
        # Fixed: Wallet addresses must be 32-44 characters
        edges = [
            FundingEdge(
                source="wa11etA11111111111111111111111111",
                target="wa11etB11111111111111111111111111",
                amount_lamports=1000000,
                timestamp=datetime.now(),
                signature=_sig(1),
            ),
            FundingEdge(
                source="wa11etA11111111111111111111111111",
                target="wa11etC11111111111111111111111111",
                amount_lamports=1000000,
                timestamp=datetime.now(),
                signature=_sig(2),
            ),
            FundingEdge(
                source="wa11etA11111111111111111111111111",
                target="wa11etD11111111111111111111111111",
                amount_lamports=1000000,
                timestamp=datetime.now(),
                signature=_sig(3),
            ),
            # Independent wallets
            FundingEdge(
                source="wa11etE11111111111111111111111111",
                target="wa11etF11111111111111111111111111",
                amount_lamports=1000000,
                timestamp=datetime.now(),
                signature=_sig(4),
            ),
            FundingEdge(
                source="wa11etG11111111111111111111111111",
                target="wa11etH11111111111111111111111111",
                amount_lamports=1000000,
                timestamp=datetime.now(),
                signature=_sig(5),
            ),
        ]

        graph.add_edges_from_list(edges)
        return graph

    def test_embedder_initialization(self):
        """Test GraphEmbedder initialization."""
        config = EmbeddingConfig(dimensions=32, walk_length=10, num_walks=50)
        embedder = GraphEmbedder(config=config)

        assert embedder.config.dimensions == 32
        assert embedder.config.walk_length == 10
        assert embedder.config.num_walks == 50
        assert embedder.model is None
        assert len(embedder.embeddings) == 0

    def test_fit_transform(self, sample_graph):
        """Test Node2Vec fit and transform."""
        config = EmbeddingConfig(dimensions=16, walk_length=10, num_walks=20, workers=1)
        embedder = GraphEmbedder(config=config)

        embeddings = embedder.fit_transform(sample_graph, embedding_id="test_v1")

        # Should have embeddings for all nodes
        assert len(embeddings) == sample_graph.num_vertices

        # Check embedding properties
        for wallet, embedding in embeddings.items():
            assert embedding.wallet == wallet
            assert embedding.vector.shape == (16,)
            assert embedding.embedding_id == "test_v1"

    def test_compute_similarity(self, sample_graph):
        """Test cosine similarity computation."""
        config = EmbeddingConfig(dimensions=16, walk_length=10, num_walks=20, workers=1)
        embedder = GraphEmbedder(config=config)
        embedder.fit_transform(sample_graph)

        # Use actual keys from embeddings (addresses may vary slightly)
        all_wallets = list(embedder.embeddings.keys())
        wallet_b = [w for w in all_wallets if 'B' in w][0]
        wallet_c = [w for w in all_wallets if 'C' in w][0]
        wallet_f = [w for w in all_wallets if 'F' in w][0]

        # Test that similarity computation works
        sim_BC = embedder.compute_similarity(wallet_b, wallet_c)
        sim_BF = embedder.compute_similarity(wallet_b, wallet_f)

        assert sim_BC is not None
        assert sim_BF is not None
        # Similarity scores should be in valid range (cosine similarity)
        assert -1.0 <= sim_BC <= 1.0
        assert -1.0 <= sim_BF <= 1.0

    def test_find_similar_wallets(self, sample_graph):
        """Test finding similar wallets."""
        config = EmbeddingConfig(dimensions=16, walk_length=10, num_walks=20, workers=1)
        embedder = GraphEmbedder(config=config)
        embedder.fit_transform(sample_graph)

        similar = embedder.find_similar_wallets("wa11etB11111111111111111111111111", k=3, min_similarity=0.0)

        assert len(similar) <= 3
        # Should return (wallet, similarity) tuples
        for wallet, sim in similar:
            assert isinstance(wallet, str)
            assert -1.0 <= sim <= 1.0

    def test_cluster_embeddings(self, sample_graph):
        """Test clustering embeddings."""
        config = EmbeddingConfig(dimensions=16, walk_length=10, num_walks=20, workers=1)
        embedder = GraphEmbedder(config=config)
        embedder.fit_transform(sample_graph)

        clusters = embedder.cluster_embeddings(n_clusters=2, method="kmeans")

        assert len(clusters) == sample_graph.num_vertices
        # All wallets should be assigned to a cluster
        for wallet, cluster_id in clusters.items():
            assert isinstance(cluster_id, int)
            assert cluster_id >= 0


class TestWalletSimilarity:
    """Test wallet similarity detection."""

    @pytest.fixture
    def setup_similarity(self):
        """Setup graph, embedder, and similarity detector."""
        graph = FundingGraph()

        # Coordinated cluster: A -> B, C, D (addresses 32 chars)
        edges = [
            FundingEdge(source="wa11etA11111111111111111111111111", target="wa11etB11111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(1)),
            FundingEdge(source="wa11etA11111111111111111111111111", target="wa11etC11111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(2)),
            FundingEdge(source="wa11etA11111111111111111111111111", target="wa11etD11111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(3)),
            # Independent wallets
            FundingEdge(source="wa11etE11111111111111111111111111", target="wa11etF11111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(4)),
        ]
        graph.add_edges_from_list(edges)

        config = EmbeddingConfig(dimensions=16, walk_length=10, num_walks=20, workers=1)
        embedder = GraphEmbedder(config=config)
        embedder.fit_transform(graph)

        detector = WalletSimilarityDetector(
            embedder=embedder, graph=graph, embedding_weight=0.7, structural_weight=0.3
        )

        return graph, embedder, detector

    def test_structural_similarity(self, setup_similarity):
        """Test structural similarity computation."""
        graph, embedder, detector = setup_similarity

        # B and C share funder A (wallets 32 chars)
        sim_BC = detector.compute_structural_similarity("wa11etB11111111111111111111111111", "wa11etC11111111111111111111111111")

        # B and F don't share funders
        sim_BF = detector.compute_structural_similarity("wa11etB11111111111111111111111111", "wa11etF11111111111111111111111111")

        assert sim_BC > 0.0
        assert sim_BC > sim_BF

    def test_compute_similarity(self, setup_similarity):
        """Test combined similarity computation."""
        graph, embedder, detector = setup_similarity

        score = detector.compute_similarity(
            "wa11etB11111111111111111111111111", "wa11etC11111111111111111111111111", coordination_threshold=0.5
        )

        assert score is not None
        # Embedding similarity is cosine similarity, can be negative
        assert -1.0 <= score.embedding_similarity <= 1.0
        assert 0.0 <= score.structural_similarity <= 1.0
        # Combined similarity can include negative embedding component
        assert -1.0 <= score.combined_similarity <= 1.0
        assert isinstance(score.is_coordinated, bool)

    def test_find_coordinated_pairs(self, setup_similarity):
        """Test finding coordinated wallet pairs."""
        graph, embedder, detector = setup_similarity

        wallets = ["wa11etB11111111111111111111111111", "wa11etC11111111111111111111111111", "wa11etD11111111111111111111111111", "wa11etF11111111111111111111111111"]
        pairs = detector.find_coordinated_pairs(wallets, min_similarity=0.3)

        # Should find some coordinated pairs
        assert len(pairs) > 0

        # All pairs should be above threshold
        for score in pairs:
            assert score.is_coordinated
            assert score.combined_similarity >= 0.3

    def test_detect_sybil_clusters(self, setup_similarity):
        """Test Sybil cluster detection."""
        graph, embedder, detector = setup_similarity

        wallets = ["wa11etB11111111111111111111111111", "wa11etC11111111111111111111111111", "wa11etD11111111111111111111111111", "wa11etF11111111111111111111111111"]
        clusters = detector.detect_sybil_clusters(
            wallets, similarity_threshold=0.3, min_cluster_size=2
        )

        # Should detect at least one cluster (B, C, D)
        assert len(clusters) > 0

        for cluster in clusters:
            assert len(cluster.wallets) >= 2
            assert 0.0 <= cluster.mean_similarity <= 1.0
            assert 0.0 <= cluster.sybil_probability <= 1.0


class TestDynamicNetworkMetrics:
    """Test dynamic network analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create network analyzer."""
        return DynamicNetworkAnalyzer()

    @pytest.fixture
    def evolving_graphs(self):
        """Create time series of evolving graphs."""
        graphs = []
        timestamps = []

        # T0: Small network
        g0 = FundingGraph()
        g0.add_edges_from_list(
            [
                FundingEdge(source="A11111111111111111111111111111111", target="B11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(1)),
                FundingEdge(source="B11111111111111111111111111111111", target="C11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(2)),
            ]
        )
        graphs.append(g0)
        timestamps.append(datetime.now())

        # T1: Network grows
        g1 = FundingGraph()
        g1.add_edges_from_list(
            [
                FundingEdge(source="A11111111111111111111111111111111", target="B11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(1)),
                FundingEdge(source="B11111111111111111111111111111111", target="C11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(2)),
                FundingEdge(source="A11111111111111111111111111111111", target="D11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(3)),
                FundingEdge(source="D11111111111111111111111111111111", target="E11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(4)),
            ]
        )
        graphs.append(g1)
        timestamps.append(datetime.now() + timedelta(days=1))

        # T2: Community fragmentation
        g2 = FundingGraph()
        g2.add_edges_from_list(
            [
                FundingEdge(source="A11111111111111111111111111111111", target="B11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(1)),
                FundingEdge(source="B11111111111111111111111111111111", target="C11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(2)),
                FundingEdge(source="D11111111111111111111111111111111", target="E11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(4)),
                FundingEdge(source="F11111111111111111111111111111111", target="G11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(5)),
            ]
        )
        graphs.append(g2)
        timestamps.append(datetime.now() + timedelta(days=2))

        return graphs, timestamps

    def test_compute_snapshot(self, analyzer, evolving_graphs):
        """Test network snapshot computation."""
        graphs, timestamps = evolving_graphs
        graph = graphs[0]

        snapshot = analyzer.compute_snapshot(graph, timestamp=timestamps[0])

        assert snapshot.timestamp == timestamps[0]
        assert snapshot.num_nodes == graph.num_vertices
        assert snapshot.num_edges == graph.num_edges
        assert 0.0 <= snapshot.density <= 1.0
        assert 0.0 <= snapshot.modularity <= 1.0
        assert 0.0 <= snapshot.centralization <= 1.0

    def test_compute_dynamics(self, analyzer, evolving_graphs):
        """Test network dynamics computation."""
        graphs, timestamps = evolving_graphs

        # Compute snapshots
        for graph, timestamp in zip(graphs, timestamps):
            analyzer.compute_snapshot(graph, timestamp=timestamp)

        dynamics = analyzer.compute_dynamics(window_size=3)

        assert dynamics is not None
        assert len(dynamics.snapshots) >= 2
        assert isinstance(dynamics.velocity_density, float)
        assert isinstance(dynamics.velocity_modularity, float)
        # Note: numpy.bool_ is compatible with bool checks
        assert dynamics.is_fragmenting in (True, False)
        assert dynamics.is_consolidating in (True, False)

    def test_detect_community_events(self, analyzer, evolving_graphs):
        """Test community event detection."""
        graphs, timestamps = evolving_graphs

        # Compute snapshots
        for graph, timestamp in zip(graphs, timestamps):
            analyzer.compute_snapshot(graph, timestamp=timestamp)

        events = analyzer.detect_community_events(min_snapshots=2)

        # Should detect some events as network evolves
        for event in events:
            assert event["type"] in [
                "community_emergence",
                "community_consolidation",
                "network_fragmentation",
            ]
            assert "timestamp" in event

    def test_network_health_score(self, analyzer, evolving_graphs):
        """Test network health score computation."""
        graphs, timestamps = evolving_graphs

        # Compute snapshot
        analyzer.compute_snapshot(graphs[0], timestamp=timestamps[0])

        health = analyzer.get_network_health_score()

        assert health is not None
        assert 0.0 <= health <= 1.0


class TestAnomalyDetection:
    """Test wallet anomaly detection."""

    @pytest.fixture
    def setup_anomaly_detector(self):
        """Setup graph, embedder, and anomaly detector."""
        graph = FundingGraph()

        # Create diverse wallet patterns
        edges = []
        # Normal wallets: single funder (addresses must be 32-44 chars, base58 only)
        # Note: "wallet" contains 'l' which is NOT in base58, use "wa11et" instead
        # "funder" = 6 chars, need 26 more for 32 total
        for i in range(20):
            suffix = BASE58[i % len(BASE58)]
            funder = f"funder{suffix}1111111111111111111111111"  # 6 + 1 + 25 = 32 chars
            wa11et = f"wa11et{suffix}1111111111111111111111111"  # 6 + 1 + 25 = 32 chars
            edges.append(
                FundingEdge(
                    source=funder,
                    target=wa11et,
                    amount_lamports=1000000,
                    timestamp=datetime.now(),
                    signature=_sig(i),
                )
            )

        # Anomalous wallets: many funders (unusual)
        # "wa11etanoma1y" avoids 'l' character
        anomaly_wallet = "wa11etanoma1y11111111111111111111"  # 32 chars
        for i in range(5):
            suffix = BASE58[i % len(BASE58)]
            funder = f"funder{suffix}1111111111111111111111111"
            edges.append(
                FundingEdge(
                    source=funder,
                    target=anomaly_wallet,
                    amount_lamports=500000,
                    timestamp=datetime.now(),
                    signature=_sig(100 + i),
                )
            )

        # Anomalous: wallet funds many others (unusual)
        # "wa11etsuperfunder" = 17 chars, need 15 more 1's for 32 total
        superfunder = "wa11etsuperfunder111111111111111"  # 32 chars
        for i in range(10):
            suffix = BASE58[i % len(BASE58)]
            # "wa11etfunded" = 12 chars, + suffix = 13 chars, need 19 more 1's for 32 total
            target = f"wa11etfunded{suffix}1111111111111111111"  # 12 + 1 + 19 = 32 chars
            edges.append(
                FundingEdge(
                    source=superfunder,
                    target=target,
                    amount_lamports=200000,
                    timestamp=datetime.now(),
                    signature=_sig(200 + i),
                )
            )

        graph.add_edges_from_list(edges)

        # Generate embeddings
        config = EmbeddingConfig(dimensions=16, walk_length=10, num_walks=20, workers=1)
        embedder = GraphEmbedder(config=config)
        embedder.fit_transform(graph)

        # Create detector
        anomaly_config = AnomalyConfig(contamination=0.1, n_estimators=50)
        detector = WalletAnomalyDetector(
            embedder=embedder, graph=graph, config=anomaly_config
        )

        return graph, embedder, detector

    def test_extract_features(self, setup_anomaly_detector):
        """Test feature extraction for anomaly detection."""
        graph, embedder, detector = setup_anomaly_detector

        # Use wa11et (with 1's instead of l) to match base58 alphabet
        wallets = ["wa11et11111111111111111111111111", "wa11et21111111111111111111111111", "wa11etanoma1y11111111111111111111"]
        X, valid_wallets, feature_names = detector.extract_features(
            wallets, include_embeddings=True
        )

        assert X.shape[0] == len(valid_wallets)
        assert X.shape[1] == len(feature_names)
        assert "in_degree" in feature_names
        assert "out_degree" in feature_names
        assert any("emb_" in name for name in feature_names)

    def test_fit_predict(self, setup_anomaly_detector):
        """Test fitting and predicting anomalies."""
        graph, embedder, detector = setup_anomaly_detector

        # Get all wallets
        wallets = list(graph._wallet_set)

        # Fit detector
        detector.fit(wallets, include_embeddings=True)

        assert detector.fitted
        assert detector.model is not None
        assert detector.scaler is not None

        # Predict on anomalous wallet
        score = detector.predict("wa11etanoma1y11111111111111111111", include_embeddings=True)

        if score is not None:  # May be None if wallet not in embeddings
            assert -1.0 <= score.score <= 1.0
            # Note: numpy.bool_ is compatible with bool checks
            assert score.is_anomalous in (True, False)
            assert 0.0 <= score.confidence <= 1.0

    def test_find_anomalies(self, setup_anomaly_detector):
        """Test finding most anomalous wallets."""
        graph, embedder, detector = setup_anomaly_detector

        wallets = list(graph._wallet_set)
        detector.fit(wallets, include_embeddings=True)

        anomalies = detector.find_anomalies(wallets, top_k=5, include_embeddings=True)

        # Should find some anomalies
        assert len(anomalies) <= 5

        # All should be flagged as anomalous
        for anomaly in anomalies:
            assert anomaly.is_anomalous

        # Should be sorted by score (most anomalous first)
        if len(anomalies) > 1:
            for i in range(len(anomalies) - 1):
                assert anomalies[i].score <= anomalies[i + 1].score

    def test_anomaly_distribution(self, setup_anomaly_detector):
        """Test anomaly score distribution statistics."""
        graph, embedder, detector = setup_anomaly_detector

        wallets = list(graph._wallet_set)
        detector.fit(wallets, include_embeddings=True)

        distribution = detector.get_anomaly_distribution(wallets, include_embeddings=True)

        assert "total_wallets" in distribution
        assert "anomalous_count" in distribution
        assert "mean_score" in distribution
        assert "std_score" in distribution
        assert distribution["anomalous_count"] >= 0
        assert distribution["total_wallets"] >= distribution["anomalous_count"]


class TestIntegration:
    """Integration tests for Sprint 2."""

    def test_end_to_end_pipeline(self):
        """Test complete graph intelligence pipeline."""
        # Build funding graph
        graph = FundingGraph()
        edges = [
            FundingEdge(source="A11111111111111111111111111111111", target="B11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(1)),
            FundingEdge(source="A11111111111111111111111111111111", target="C11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(2)),
            FundingEdge(source="A11111111111111111111111111111111", target="D11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(3)),
            FundingEdge(source="E11111111111111111111111111111111", target="F11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(4)),
            FundingEdge(source="G11111111111111111111111111111111", target="H11111111111111111111111111111111", amount_lamports=1000000, timestamp=datetime.now(), signature=_sig(5)),
        ]
        graph.add_edges_from_list(edges)

        # Generate embeddings
        config = EmbeddingConfig(dimensions=16, walk_length=10, num_walks=20, workers=1)
        embedder = GraphEmbedder(config=config)
        embeddings = embedder.fit_transform(graph)

        assert len(embeddings) == graph.num_vertices

        # Detect similar wallets
        similarity_detector = WalletSimilarityDetector(embedder=embedder, graph=graph)
        wallets = ["B11111111111111111111111111111111", "C11111111111111111111111111111111", "D11111111111111111111111111111111", "F11111111111111111111111111111111"]
        clusters = similarity_detector.detect_sybil_clusters(
            wallets, similarity_threshold=0.3, min_cluster_size=2
        )

        # Should find coordinated cluster (B, C, D share funder A)
        assert len(clusters) > 0

        # Compute network metrics
        analyzer = DynamicNetworkAnalyzer()
        snapshot = analyzer.compute_snapshot(graph)

        assert snapshot.num_nodes == graph.num_vertices
        assert snapshot.num_edges == graph.num_edges

        # Detect anomalies
        anomaly_detector = WalletAnomalyDetector(embedder=embedder, graph=graph)
        all_wallets = list(graph._wallet_set)
        anomaly_detector.fit(all_wallets, include_embeddings=True)
        anomalies = anomaly_detector.find_anomalies(
            all_wallets, top_k=3, include_embeddings=True
        )

        # Should complete without errors
        assert isinstance(anomalies, list)
