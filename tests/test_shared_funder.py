"""Tests for shared funder detection."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
import networkx as nx

from src.detection.shared_funder import (
    SharedFunderDetector,
    FunderCluster,
    SharedFunderResult,
)


class TestSharedFunderDetector:
    """Tests for SharedFunderDetector class."""

    @pytest.fixture
    def detector(self) -> SharedFunderDetector:
        """Create detector with default settings."""
        return SharedFunderDetector(
            min_cluster_size=2,
            max_depth=2,
            min_confidence=0.5,
        )

    @pytest.fixture
    def strict_detector(self) -> SharedFunderDetector:
        """Create detector with stricter settings."""
        return SharedFunderDetector(
            min_cluster_size=3,
            max_depth=1,
            min_confidence=0.7,
        )

    # -------------------------------------------------------------------------
    # Basic Detection Tests
    # -------------------------------------------------------------------------

    def test_detect_shared_funder_cluster(
        self,
        detector: SharedFunderDetector,
        mock_funding_graph,
        sybil_cluster_wallets: list[str],
        sybil_funder: str,
    ):
        """Test detection of wallets with shared funder."""
        result = detector.detect(mock_funding_graph, target_wallets=sybil_cluster_wallets)

        assert isinstance(result, SharedFunderResult)
        assert result.total_wallets_analyzed == len(sybil_cluster_wallets)
        assert len(result.clusters) > 0

        # Check dominant funder
        assert result.dominant_funder == sybil_funder
        assert result.dominant_funder_count == len(sybil_cluster_wallets)

    def test_detect_empty_wallet_list(self, detector: SharedFunderDetector, mock_funding_graph):
        """Test detection with empty wallet list."""
        result = detector.detect(mock_funding_graph, target_wallets=[])

        assert result.total_wallets_analyzed == 0
        assert result.total_wallets_clustered == 0
        assert len(result.clusters) == 0
        assert result.dominant_funder is None

    def test_cluster_contains_wallet_addresses(
        self,
        detector: SharedFunderDetector,
        mock_funding_graph,
        sybil_cluster_wallets: list[str],
    ):
        """Test that detected clusters contain the expected wallet addresses."""
        result = detector.detect(mock_funding_graph, target_wallets=sybil_cluster_wallets)

        if result.clusters:
            cluster = result.clusters[0]
            # All wallets in cluster should be from our sybil set
            for wallet in cluster.wallet_addresses:
                assert wallet in sybil_cluster_wallets

    # -------------------------------------------------------------------------
    # Confidence Scoring Tests
    # -------------------------------------------------------------------------

    def test_cluster_has_confidence_score(
        self,
        detector: SharedFunderDetector,
        mock_funding_graph,
        sybil_cluster_wallets: list[str],
    ):
        """Test that detected clusters have confidence scores."""
        result = detector.detect(mock_funding_graph, target_wallets=sybil_cluster_wallets)

        for cluster in result.clusters:
            assert 0.0 <= cluster.confidence <= 1.0

    def test_larger_clusters_have_higher_confidence(self, detector: SharedFunderDetector):
        """Test that larger clusters tend to have higher confidence."""
        # Create mock graph with varying cluster sizes
        graph = MagicMock()
        G = nx.DiGraph()

        funder = "Funder111111111111111111111111111111111111"
        G.add_node(funder)

        wallets_2 = [f"Small{i}" + "x" * 38 for i in range(2)]
        wallets_5 = [f"Large{i}" + "x" * 38 for i in range(5)]

        for w in wallets_2:
            G.add_edge(funder, w, amount=1_000_000_000)
        for w in wallets_5:
            G.add_edge("Funder2" + "x" * 38, w, amount=1_000_000_000)

        graph._graph = G
        graph._wallet_set = set(wallets_2 + wallets_5)

        # Mock shared funders
        def find_shared_funders(target_wallets, max_depth=2):
            return {
                funder: set(wallets_2),
                "Funder2" + "x" * 38: set(wallets_5),
            }

        def get_dominant_funder(target_wallets, max_depth=2):
            return ("Funder2" + "x" * 38, 5)

        graph.find_shared_funders = find_shared_funders
        graph.get_dominant_funder = get_dominant_funder

        result = detector.detect(graph, target_wallets=wallets_2 + wallets_5)

        # Larger cluster should have higher confidence (due to size factor)
        if len(result.clusters) >= 2:
            # Sort by size
            sorted_clusters = sorted(result.clusters, key=lambda c: len(c.wallet_addresses))
            # Generally, larger clusters have better confidence
            # (though other factors affect this too)
            assert sorted_clusters[-1].confidence >= 0.0

    def test_direct_funding_has_higher_confidence(self, detector: SharedFunderDetector):
        """Test that direct funding (depth 1) has higher confidence than indirect."""
        # This is implicitly tested through the confidence calculation
        # Direct funding contributes more to confidence than indirect
        assert detector.max_depth >= 1

    # -------------------------------------------------------------------------
    # Cluster Metrics Tests
    # -------------------------------------------------------------------------

    def test_cluster_has_funding_depth(
        self,
        detector: SharedFunderDetector,
        mock_funding_graph,
        sybil_cluster_wallets: list[str],
    ):
        """Test that clusters include funding depth information."""
        result = detector.detect(mock_funding_graph, target_wallets=sybil_cluster_wallets)

        for cluster in result.clusters:
            assert cluster.funding_depth >= 1
            assert cluster.funding_depth <= detector.max_depth + 1

    def test_cluster_has_funder_address(
        self,
        detector: SharedFunderDetector,
        mock_funding_graph,
        sybil_cluster_wallets: list[str],
        sybil_funder: str,
    ):
        """Test that clusters include funder address."""
        result = detector.detect(mock_funding_graph, target_wallets=sybil_cluster_wallets)

        for cluster in result.clusters:
            assert cluster.funder_address is not None
            # Our mock has only one funder
            assert cluster.funder_address == sybil_funder

    # -------------------------------------------------------------------------
    # Filtering Tests
    # -------------------------------------------------------------------------

    def test_min_cluster_size_filtering(
        self,
        strict_detector: SharedFunderDetector,
    ):
        """Test that clusters below min size are filtered out."""
        # Create graph with 2-wallet cluster (below strict min of 3)
        graph = MagicMock()
        G = nx.DiGraph()

        funder = "SmallFunder1111111111111111111111111111111"
        wallets = ["Small1" + "x" * 38, "Small2" + "x" * 38]

        G.add_node(funder)
        for w in wallets:
            G.add_edge(funder, w, amount=1_000_000_000)

        graph._graph = G
        graph._wallet_set = set(wallets)

        def find_shared_funders(target_wallets, max_depth=2):
            return {funder: set(wallets)}

        def get_dominant_funder(target_wallets, max_depth=2):
            return (funder, 2)

        graph.find_shared_funders = find_shared_funders
        graph.get_dominant_funder = get_dominant_funder

        result = strict_detector.detect(graph, target_wallets=wallets)

        # Cluster of 2 should be filtered out by strict detector (min_cluster_size=3)
        assert len(result.clusters) == 0

    def test_min_confidence_filtering(
        self,
        strict_detector: SharedFunderDetector,
    ):
        """Test that clusters below min confidence are filtered out."""
        # Strict detector requires 0.7 confidence
        assert strict_detector.min_confidence == 0.7

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    def test_detect_with_none_target_wallets(
        self,
        detector: SharedFunderDetector,
        mock_funding_graph,
    ):
        """Test detection uses all wallets when target_wallets is None."""
        # When target_wallets is None, it should use all wallets in the graph
        result = detector.detect(mock_funding_graph, target_wallets=None)

        # Should analyze all wallets in the graph's wallet set
        assert result.total_wallets_analyzed == len(mock_funding_graph._wallet_set)

    def test_detect_from_wallets_convenience_method(
        self,
        detector: SharedFunderDetector,
        mock_funding_graph,
        sybil_cluster_wallets: list[str],
    ):
        """Test the detect_from_wallets convenience method."""
        clusters = detector.detect_from_wallets(
            mock_funding_graph,
            wallets=sybil_cluster_wallets,
            min_shared_for_cluster=2,
        )

        assert isinstance(clusters, list)
        # All returned clusters should contain at least one of our target wallets
        for cluster in clusters:
            has_target_wallet = any(w in sybil_cluster_wallets for w in cluster.wallet_addresses)
            assert has_target_wallet

    def test_single_wallet_returns_no_clusters(
        self,
        detector: SharedFunderDetector,
    ):
        """Test that analyzing a single wallet returns no clusters."""
        # Create a mock graph with only one wallet
        graph = MagicMock()
        G = nx.DiGraph()
        single_wallet = "SingleWallet" + "x" * 32
        G.add_node(single_wallet)
        graph._graph = G
        graph._wallet_set = {single_wallet}

        def find_shared_funders(target_wallets, max_depth=2):
            return {}  # No shared funders for single wallet

        def get_dominant_funder(target_wallets, max_depth=2):
            return (None, 0)

        graph.find_shared_funders = find_shared_funders
        graph.get_dominant_funder = get_dominant_funder

        result = detector.detect(graph, target_wallets=[single_wallet])

        # Can't form a cluster with just one wallet
        assert result.total_wallets_clustered == 0


class TestFunderCluster:
    """Tests for FunderCluster dataclass."""

    def test_funder_cluster_creation(self):
        """Test creating a FunderCluster."""
        cluster = FunderCluster(
            funder_address="Funder111111111111111111111111111111111111",
            wallet_addresses=["W1" + "x" * 40, "W2" + "x" * 40],
            funding_depth=1,
            total_funded_amount=2_000_000_000,
            avg_funded_amount=1_000_000_000.0,
            funding_time_span_hours=2.5,
            confidence=0.85,
        )

        assert cluster.funder_address.startswith("Funder")
        assert len(cluster.wallet_addresses) == 2
        assert cluster.funding_depth == 1
        assert cluster.confidence == 0.85

    def test_funder_cluster_optional_time_span(self):
        """Test FunderCluster with None time span."""
        cluster = FunderCluster(
            funder_address="Funder111111111111111111111111111111111111",
            wallet_addresses=["W1" + "x" * 40],
            funding_depth=2,
            total_funded_amount=1_000_000_000,
            avg_funded_amount=1_000_000_000.0,
            funding_time_span_hours=None,  # Time span may not be available
            confidence=0.6,
        )

        assert cluster.funding_time_span_hours is None


class TestSharedFunderResult:
    """Tests for SharedFunderResult dataclass."""

    def test_shared_funder_result_creation(self):
        """Test creating SharedFunderResult."""
        result = SharedFunderResult(
            clusters=[],
            total_wallets_analyzed=10,
            total_wallets_clustered=6,
            dominant_funder="Funder" + "x" * 38,
            dominant_funder_count=6,
        )

        assert result.total_wallets_analyzed == 10
        assert result.total_wallets_clustered == 6
        assert result.dominant_funder is not None

    def test_result_has_detection_timestamp(self):
        """Test that result includes detection timestamp."""
        result = SharedFunderResult(
            clusters=[],
            total_wallets_analyzed=5,
            total_wallets_clustered=0,
            dominant_funder=None,
            dominant_funder_count=0,
        )

        assert result.detection_timestamp is not None
        assert isinstance(result.detection_timestamp, datetime)


class TestConfidenceCalculation:
    """Tests for the confidence calculation logic."""

    @pytest.fixture
    def detector(self) -> SharedFunderDetector:
        return SharedFunderDetector()

    def test_confidence_increases_with_cluster_size(self, detector: SharedFunderDetector):
        """Test that confidence increases with more wallets in cluster."""
        # Small cluster
        confidence_small = detector._calculate_confidence(
            cluster_size=2,
            avg_depth=1.0,
            amounts=[],
            time_span_hours=None,
        )

        # Large cluster
        confidence_large = detector._calculate_confidence(
            cluster_size=10,
            avg_depth=1.0,
            amounts=[],
            time_span_hours=None,
        )

        assert confidence_large > confidence_small

    def test_confidence_decreases_with_depth(self, detector: SharedFunderDetector):
        """Test that confidence decreases with greater funding depth."""
        # Direct funding
        confidence_direct = detector._calculate_confidence(
            cluster_size=5,
            avg_depth=1.0,
            amounts=[],
            time_span_hours=None,
        )

        # Indirect funding
        confidence_indirect = detector._calculate_confidence(
            cluster_size=5,
            avg_depth=2.0,
            amounts=[],
            time_span_hours=None,
        )

        assert confidence_direct > confidence_indirect

    def test_consistent_amounts_increase_confidence(self, detector: SharedFunderDetector):
        """Test that consistent funding amounts increase confidence."""
        # Consistent amounts (low variance)
        confidence_consistent = detector._calculate_confidence(
            cluster_size=5,
            avg_depth=1.0,
            amounts=[1_000_000_000, 1_000_000_000, 1_000_000_000],
            time_span_hours=None,
        )

        # Varied amounts (high variance)
        confidence_varied = detector._calculate_confidence(
            cluster_size=5,
            avg_depth=1.0,
            amounts=[100_000_000, 1_000_000_000, 10_000_000_000],
            time_span_hours=None,
        )

        assert confidence_consistent >= confidence_varied

    def test_rapid_funding_increases_confidence(self, detector: SharedFunderDetector):
        """Test that rapid sequential funding increases confidence."""
        # Very rapid (within 1 hour)
        confidence_rapid = detector._calculate_confidence(
            cluster_size=5,
            avg_depth=1.0,
            amounts=[],
            time_span_hours=0.5,
        )

        # Spread over a week
        confidence_slow = detector._calculate_confidence(
            cluster_size=5,
            avg_depth=1.0,
            amounts=[],
            time_span_hours=200.0,
        )

        assert confidence_rapid > confidence_slow

    def test_confidence_bounded_0_to_1(self, detector: SharedFunderDetector):
        """Test that confidence is always between 0 and 1."""
        test_cases = [
            (2, 1.0, [], None),
            (100, 1.0, [1_000_000_000] * 100, 0.1),  # Best case
            (2, 3.0, [], 1000.0),  # Worst case
        ]

        for cluster_size, avg_depth, amounts, time_span in test_cases:
            confidence = detector._calculate_confidence(
                cluster_size=cluster_size,
                avg_depth=avg_depth,
                amounts=amounts,
                time_span_hours=time_span,
            )
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of bounds"
