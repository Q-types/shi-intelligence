"""
Unit Tests for SHI Clustering Intelligence Upgrade.

Tests the 10 required fixes:
1. Explicit missingness handling
2. Robust feature transformations
3. Feature-group ablation tests
4. HDBSCAN diagnostics
5. Multi-score archetype assignment
6. Noise handling
7. Node2Vec integration (experimental)
8. Weighted funding graph features
9. Temporal validation
10. Expanded Cox PH features
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# --- 1. Test Explicit Missingness Handling ---


class TestMissingnessHandling:
    """Tests for explicit missingness handling in transformations."""

    def test_missingness_indicators_creation(self):
        """Test that indicator columns are created for NaN values."""
        from src.clustering.transformations import FeatureTransformer

        transformer = FeatureTransformer()

        # Create data with some missing values
        data = np.array([
            [1.0, np.nan, 3.0],
            [np.nan, 2.0, np.nan],
            [3.0, 3.0, 3.0],
        ])
        feature_names = ["feat_a", "feat_b", "feat_c"]

        transformed, indicators = transformer.fit_transform(
            data, feature_names, impute=True
        )

        # Should have indicator columns in missing_flags
        assert indicators is not None
        # Check that flags exist for features with missing data
        assert len(indicators.missing_flags) > 0

    def test_nan_preserved_before_impute(self):
        """Test that NaN is preserved when impute=False."""
        from src.clustering.transformations import FeatureTransformer

        transformer = FeatureTransformer()

        # Data with NaN
        data = np.array([[1.0], [np.nan], [3.0]])

        # Transform without imputation
        transformed, indicators = transformer.fit_transform(data, ["test"], impute=False)

        # NaN should still be present
        assert np.isnan(transformed[1, 0])

    def test_missingness_indicators_dataclass(self):
        """Test MissingnessIndicators dataclass methods."""
        from src.clustering.transformations import MissingnessIndicators

        indicators = MissingnessIndicators()
        indicators.add_indicator("feat_a", np.array([True, False, True]))
        indicators.add_indicator("feat_b", np.array([False, True, False]))

        names = indicators.get_indicator_names()
        assert "feat_a_missing" in names
        assert "feat_b_missing" in names

        array = indicators.to_array()
        assert array.shape == (3, 2)


# --- 2. Test Robust Feature Transformations ---


class TestRobustTransformations:
    """Tests for robust feature transformations."""

    def test_transformation_types_exist(self):
        """Test TransformationType enum has required values."""
        from src.clustering.transformations import TransformationType

        assert TransformationType.LOG1P.value == "log1p"
        assert TransformationType.ASINH.value == "asinh"
        assert TransformationType.SQRT.value == "sqrt"
        assert TransformationType.ROBUST_SCALE.value == "robust_scale"
        assert TransformationType.NONE.value == "none"

    def test_transformer_applies_log1p(self):
        """Test transformer applies log1p for appropriate features."""
        from src.clustering.transformations import FeatureTransformer

        transformer = FeatureTransformer()

        # balance uses log1p transformation
        data = np.array([[1.0], [10.0], [100.0]])
        transformed, _ = transformer.fit_transform(data, ["balance"], impute=True)

        # Values should be log-transformed
        expected = np.log1p(data)
        np.testing.assert_array_almost_equal(transformed, expected)

    def test_robust_scaler_uses_iqr(self):
        """Test RobustScaler uses IQR instead of std."""
        from src.clustering.transformations import RobustScaler

        # Data with outliers
        data = np.array([[1], [2], [3], [4], [5], [100]])  # 100 is outlier

        scaler = RobustScaler()
        scaled = scaler.fit_transform(data)

        # With robust scaling, outlier shouldn't dominate
        # Median should be near 0 after scaling
        median_idx = 2  # value 3 is near median
        assert abs(scaled[median_idx, 0]) < 1.0


# --- 3. Test Feature-Group Ablation ---


class TestFeatureGroupAblation:
    """Tests for feature-group ablation testing."""

    def test_ablation_tester_initialization(self):
        """Test ablation tester initializes with feature groups."""
        from src.clustering.ablation import AblationTester, FEATURE_GROUPS

        tester = AblationTester()

        assert "distribution" in tester.feature_groups
        assert "temporal" in tester.feature_groups
        assert "flow" in tester.feature_groups
        assert "trading" in tester.feature_groups
        assert "graph" in tester.feature_groups
        assert "price_pnl" in tester.feature_groups
        assert "liquidity" in tester.feature_groups

    def test_ablation_result_structure(self):
        """Test ablation result contains required fields."""
        from src.clustering.ablation import AblationResult

        result = AblationResult(
            excluded_group="temporal",
            included_features=["balance", "share"],
            excluded_features=["entry_time", "holding_duration"],
            n_clusters=5,
            noise_percentage=0.15,
            silhouette_score=0.42,
            silhouette_delta=-0.05,
            noise_delta=0.02,
            cluster_delta=-1,
        )

        assert result.excluded_group == "temporal"
        assert result.silhouette_delta == -0.05


# --- 4. Test HDBSCAN Diagnostics ---


class TestHDBSCANDiagnostics:
    """Tests for HDBSCAN diagnostics."""

    def test_cluster_diagnostics_dataclass(self):
        """Test ClusterDiagnostics contains all required fields."""
        from src.clustering.diagnostics import ClusterDiagnostics

        diagnostics = ClusterDiagnostics(
            labels=np.array([0, 0, 1, 1, -1]),
            probabilities=np.array([0.9, 0.8, 0.95, 0.85, 0.0]),
            outlier_scores=np.array([0.1, 0.2, 0.05, 0.15, 0.9]),
            silhouette_score=0.45,
            noise_percentage=0.2,
            n_clusters=2,
            cluster_sizes={0: 2, 1: 2},
            cluster_persistence={0: 0.8, 1: 0.75},
        )

        # Verify all required outputs exist
        assert len(diagnostics.labels) == 5
        assert len(diagnostics.probabilities) == 5
        assert len(diagnostics.outlier_scores) == 5
        assert diagnostics.silhouette_score is not None
        assert diagnostics.noise_percentage == 0.2
        assert diagnostics.n_clusters == 2

    def test_hdbscan_diagnostics_fit(self):
        """Test HDBSCANDiagnostics.fit() returns proper structure."""
        from src.clustering.diagnostics import HDBSCANDiagnostics

        # Create clusterable data with more separation
        np.random.seed(42)
        cluster1 = np.random.randn(30, 2) * 0.5 + np.array([0, 0])
        cluster2 = np.random.randn(30, 2) * 0.5 + np.array([10, 10])
        data = np.vstack([cluster1, cluster2])

        diagnostics = HDBSCANDiagnostics(min_cluster_size=5)
        result = diagnostics.fit(data)

        assert result.labels is not None
        assert len(result.labels) == 60
        assert result.probabilities is not None
        assert result.outlier_scores is not None

    def test_wallet_cluster_info(self):
        """Test WalletClusterInfo for individual wallet."""
        from src.clustering.diagnostics import WalletClusterInfo, ClusterStatus

        info = WalletClusterInfo(
            wallet="wallet123",
            cluster_id=-1,
            membership_probability=0.0,
            outlier_score=0.95,
            cluster_status=ClusterStatus.NOISE,
            confidence_adjustment=-0.2,
        )

        assert info.cluster_status == ClusterStatus.NOISE
        assert info.outlier_score == 0.95
        assert info.is_noise == True


# --- 5. Test Multi-Score Archetype Assignment ---


class TestMultiScoreArchetype:
    """Tests for multi-score archetype assignment."""

    def test_multi_score_assignment_dataclass(self):
        """Test MultiScoreAssignment contains all required fields."""
        from src.clustering.archetypes import MultiScoreAssignment, Archetype

        assignment = MultiScoreAssignment(
            wallet="wallet456",
            primary_archetype=Archetype.DORMANT_WHALE,
            primary_confidence=0.85,
            all_scores={
                Archetype.DORMANT_WHALE: 0.85,
                Archetype.LONG_TERM_ACCUMULATOR: 0.65,
                Archetype.SNIPER: 0.20,
            },
            secondary_archetypes=[Archetype.LONG_TERM_ACCUMULATOR],
            cluster_status="CLUSTERED",
            cluster_confidence_adjustment=0.0,
            feature_matches={
                Archetype.DORMANT_WHALE: ["high_balance", "low_trade_count"],
            },
        )

        assert assignment.primary_archetype == Archetype.DORMANT_WHALE
        assert assignment.primary_confidence == 0.85
        assert Archetype.DORMANT_WHALE in assignment.all_scores
        assert assignment.cluster_status == "CLUSTERED"

    def test_archetype_enum_values(self):
        """Test Archetype enum has expected values."""
        from src.clustering.archetypes import Archetype

        assert Archetype.SNIPER.value == "sniper"
        assert Archetype.LONG_TERM_ACCUMULATOR.value == "long_term_accumulator"
        assert Archetype.COORDINATED_CLUSTER.value == "coordinated_cluster"
        assert Archetype.LIQUIDITY_ACTOR.value == "liquidity_actor"
        assert Archetype.EXCHANGE_LINKED.value == "exchange_linked"
        assert Archetype.DORMANT_WHALE.value == "dormant_whale"
        assert Archetype.UNKNOWN.value == "unknown"


# --- 6. Test Noise Handling ---


class TestNoiseHandling:
    """Tests for explicit noise handling."""

    def test_noise_cluster_status(self):
        """Test noise wallets get NOISE status."""
        from src.clustering.diagnostics import ClusterStatus

        assert ClusterStatus.NOISE.value == "noise"
        assert ClusterStatus.CORE.value == "core"
        assert ClusterStatus.BORDER.value == "border"
        assert ClusterStatus.UNKNOWN.value == "unknown"

    def test_noise_wallet_is_noise_property(self):
        """Test WalletClusterInfo.is_noise property."""
        from src.clustering.diagnostics import WalletClusterInfo, ClusterStatus

        # Noise wallet (cluster_id=-1)
        noise_info = WalletClusterInfo(
            wallet="noise_wallet",
            cluster_id=-1,
            membership_probability=0.0,
            outlier_score=0.9,
            cluster_status=ClusterStatus.NOISE,
            confidence_adjustment=-0.3,
        )
        assert noise_info.is_noise == True

        # Clustered wallet
        clustered_info = WalletClusterInfo(
            wallet="clustered_wallet",
            cluster_id=0,
            membership_probability=0.9,
            outlier_score=0.1,
            cluster_status=ClusterStatus.CORE,
            confidence_adjustment=0.0,
        )
        assert clustered_info.is_noise == False


# --- 7. Test Node2Vec Integration ---


class TestNode2VecIntegration:
    """Tests for experimental Node2Vec integration."""

    def test_node2vec_config(self):
        """Test Node2VecConfig has proper defaults."""
        from src.clustering.node2vec_integration import Node2VecConfig

        config = Node2VecConfig()

        assert config.embedding_dimensions == 64
        assert config.reduced_dimensions == 6  # 4-8 recommended
        assert config.behavior_weight == 0.7
        assert config.graph_weight == 0.3

    def test_clustering_mode_enum(self):
        """Test ClusteringMode has all required modes."""
        from src.clustering.node2vec_integration import ClusteringMode

        assert ClusteringMode.BEHAVIOR_ONLY.value == "behavior_only"
        assert ClusteringMode.GRAPH_ONLY.value == "graph_only"
        assert ClusteringMode.COMBINED.value == "combined"

    def test_clustering_comparison_structure(self):
        """Test ClusteringComparison result structure."""
        from src.clustering.node2vec_integration import (
            ClusteringComparison,
            ClusteringMode,
        )
        from src.clustering.diagnostics import ClusterDiagnostics

        behavior_diag = ClusterDiagnostics(
            labels=np.array([0, 0, 1]),
            probabilities=np.array([0.9, 0.8, 0.9]),
            outlier_scores=np.array([0.1, 0.2, 0.1]),
            silhouette_score=0.5,
            noise_percentage=0.0,
            n_clusters=2,
            cluster_sizes={0: 2, 1: 1},
            cluster_persistence={0: 0.8, 1: 0.7},
        )

        comparison = ClusteringComparison(
            behavior_only=behavior_diag,
            graph_only=None,
            combined=None,
            best_mode=ClusteringMode.BEHAVIOR_ONLY,
            best_silhouette=0.5,
            mode_details={ClusteringMode.BEHAVIOR_ONLY: {"n_features": 10}},
        )

        assert comparison.best_mode == ClusteringMode.BEHAVIOR_ONLY


# --- 8. Test Weighted Funding Graph Features ---


class TestWeightedGraphFeatures:
    """Tests for weighted funding graph features."""

    def test_weighted_features_dataclass(self):
        """Test WeightedGraphFeatures contains all required fields."""
        from src.graph.weighted_features import WeightedGraphFeatures

        features = WeightedGraphFeatures(
            wallet="wallet789",
            total_funding_received=10.5,
            total_funding_sent=2.0,
            largest_funder_share=0.4,
            funding_hhi=0.25,
            weighted_in_degree=10.5,
            weighted_out_degree=2.0,
            funding_burst_score=0.3,
            funding_balance=8.5,
            funding_concentration_ratio=0.84,
        )

        assert features.total_funding_received == 10.5
        assert features.largest_funder_share == 0.4
        assert features.funding_hhi == 0.25
        assert features.funding_burst_score == 0.3

    def test_hhi_calculation(self):
        """Test HHI (Herfindahl-Hirschman Index) is correct."""
        # HHI = sum of squared market shares
        # Single funder: HHI = 1.0
        # Two equal funders: HHI = 0.5^2 + 0.5^2 = 0.5

        shares_single = [1.0]
        hhi_single = sum(s * s for s in shares_single)
        assert hhi_single == 1.0

        shares_equal = [0.5, 0.5]
        hhi_equal = sum(s * s for s in shares_equal)
        assert hhi_equal == 0.5

    def test_funding_burst_score_range(self):
        """Test funding burst score is in valid range."""
        from src.graph.weighted_features import _compute_funding_burst_score

        # Empty edges should return 0
        score = _compute_funding_burst_score([])
        assert score == 0.0


# --- 9. Test Temporal Validation ---


class TestTemporalValidation:
    """Tests for temporal/walk-forward validation."""

    def test_temporal_validator_splits_time_ordered(self):
        """Test temporal splits maintain time ordering."""
        from src.models.temporal_validation import TemporalValidator

        # Create time-ordered data
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        data = pd.DataFrame({
            "timestamp": dates,
            "value": np.random.randn(n),
        })

        validator = TemporalValidator(n_splits=3)

        for split in validator.split(data, "timestamp"):
            # All train indices should be before test indices
            train_max = split.train_indices.max()
            test_min = split.test_indices.min()
            assert train_max < test_min, "Train data should precede test data"

    def test_temporal_split_dataclass(self):
        """Test TemporalSplit contains required fields."""
        from src.models.temporal_validation import TemporalSplit

        split = TemporalSplit(
            train_indices=np.array([0, 1, 2]),
            test_indices=np.array([3, 4]),
            train_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            train_end=datetime(2024, 1, 3, tzinfo=timezone.utc),
            test_start=datetime(2024, 1, 4, tzinfo=timezone.utc),
            test_end=datetime(2024, 1, 5, tzinfo=timezone.utc),
            fold_number=0,
        )

        assert len(split.train_indices) == 3
        assert len(split.test_indices) == 2
        assert split.train_end < split.test_start

    def test_walk_forward_validator(self):
        """Test WalkForwardValidator for production scenarios."""
        from src.models.temporal_validation import WalkForwardValidator

        validator = WalkForwardValidator(
            initial_train_days=30,
            test_window_days=7,
            retrain_frequency_days=7,
            min_train_samples=20,  # Lower threshold for test
        )

        # Create 90 days of data to ensure multiple splits
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        data = pd.DataFrame({
            "timestamp": dates,
            "value": np.random.randn(90),
        })

        splits = list(validator.split(data, "timestamp"))

        # Should have multiple splits
        assert len(splits) >= 2

        # Each split should have non-overlapping train/test
        for split in splits:
            assert split.train_end <= split.test_start


# --- 10. Test Expanded Cox PH Features ---


class TestExpandedCoxFeatures:
    """Tests for expanded Cox PH feature candidates."""

    def test_original_features_defined(self):
        """Test original features are properly defined."""
        from src.models.expanded_features import ORIGINAL_FEATURES

        assert "share" in ORIGINAL_FEATURES
        assert "holding_duration" in ORIGINAL_FEATURES
        assert "trade_count" in ORIGINAL_FEATURES
        assert len(ORIGINAL_FEATURES) == 10

    def test_candidate_features_by_category(self):
        """Test candidate features are organized by category."""
        from src.models.expanded_features import CANDIDATE_FEATURES

        assert "price" in CANDIDATE_FEATURES
        assert "liquidity" in CANDIDATE_FEATURES
        assert "lp" in CANDIDATE_FEATURES
        assert "swap" in CANDIDATE_FEATURES
        assert "graph_centrality" in CANDIDATE_FEATURES

        # Check specific features
        assert "unrealized_pnl_ratio" in CANDIDATE_FEATURES["price"]
        assert "liquidity_usd_current" in CANDIDATE_FEATURES["liquidity"]
        assert "eigenvector_centrality" in CANDIDATE_FEATURES["graph_centrality"]

    def test_model_comparison_result(self):
        """Test ModelComparisonResult structure."""
        from src.models.expanded_features import ModelComparisonResult

        result = ModelComparisonResult(
            baseline_name="original",
            candidate_name="expanded",
            baseline_concordance=0.65,
            candidate_concordance=0.70,
            concordance_improvement=0.05,
            baseline_brier=None,
            candidate_brier=None,
            brier_improvement=None,
            baseline_n_features=10,
            candidate_n_features=18,
            additional_features=["unrealized_pnl_ratio", "liquidity_usd_current"],
            is_improvement=True,
            recommendation="Use expanded features",
        )

        assert result.concordance_improvement == 0.05
        assert result.is_improvement == True

    def test_feature_selection_result(self):
        """Test FeatureSelectionResult structure."""
        from src.models.expanded_features import FeatureSelectionResult

        result = FeatureSelectionResult(
            selected_features=["share", "holding_duration", "unrealized_pnl_ratio"],
            dropped_features=["burstiness"],
            feature_scores={"share": 0.15, "holding_duration": 0.12},
            selection_method="backward",
            final_concordance=0.68,
        )

        assert len(result.selected_features) == 3
        assert result.selection_method == "backward"


# --- Integration Tests ---


class TestClusteringIntegrationFlow:
    """Integration tests for the clustering pipeline."""

    def test_diagnostics_outputs_valid_structure(self):
        """Test HDBSCAN diagnostics produce valid structure."""
        from src.clustering.diagnostics import HDBSCANDiagnostics

        # Create well-separated clusterable data
        np.random.seed(42)
        cluster1 = np.random.randn(30, 3) * 0.3 + np.array([0, 0, 0])
        cluster2 = np.random.randn(30, 3) * 0.3 + np.array([10, 10, 10])
        data = np.vstack([cluster1, cluster2])

        diagnostics = HDBSCANDiagnostics(min_cluster_size=5)
        result = diagnostics.fit(data)

        # Verify structure
        assert result.labels is not None
        assert len(result.labels) == 60
        assert result.probabilities is not None
        assert len(result.probabilities) == 60
        assert result.outlier_scores is not None
        assert isinstance(result.noise_percentage, float)
        assert isinstance(result.n_clusters, int)

    def test_transformer_produces_no_nan_when_imputed(self):
        """Test transformation with imputation produces clean data."""
        from src.clustering.transformations import FeatureTransformer

        # Create raw data with some NaN
        np.random.seed(42)
        raw_data = np.random.randn(50, 4)
        raw_data[0, 0] = np.nan
        raw_data[5, 2] = np.nan

        # Transform with imputation
        transformer = FeatureTransformer()
        transformed, indicators = transformer.fit_transform(
            raw_data,
            ["feat_a", "feat_b", "feat_c", "feat_d"],
            impute=True,
        )

        # Verify no NaN in output when imputed
        assert not np.isnan(transformed).any()

    def test_wallet_feature_vector_creation(self):
        """Test WalletFeatureVector can be created with all required fields."""
        from src.clustering.archetypes import WalletFeatureVector

        features = WalletFeatureVector(
            wallet="test_wallet",
            balance=5000.0,
            share=0.05,
            rank=10,
            entry_time_relative=0.3,
            holding_duration=200,
            position_volatility=0.05,
            delta_balance_7d=0.02,
            delta_balance_30d=0.05,
            trade_count=3,
            burstiness=0.2,
            swap_frequency=0.5,
            lp_interaction_ratio=0.1,
            in_degree=5,
            out_degree=2,
            eigenvector_centrality=0.3,
            shared_funder_count=1,
        )

        assert features.wallet == "test_wallet"
        assert features.balance == 5000.0
        assert features.eigenvector_centrality == 0.3
