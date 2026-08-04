"""
Unit tests for scoring_gene_scorer.py

This test suite provides comprehensive coverage for all classes and methods
in the scoring_gene_scorer module, including:
- EnhancedScoringConfig validation
- All concrete scoring strategies
- PathwayAggregator functionality
- EnhancedGeneScorer with advanced features
- Legacy compatibility

Tests include positive cases, error conditions, edge cases, and integration tests.
"""

import os
import sys
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# Import the module under test
# Note: These imports may need to be adjusted based on your actual module structure
try:
    from metaintegrator.scoring_gene_scorer import (
        EnhancedScoringConfig,
        ScoringStrategy,
        ZScoreStrategy,
        CombinedStrategy,
        EffectSizeStrategy,
        FDRInverseStrategy,
        FisherMinStrategy,
        PathwayAggregator,
        EnhancedGeneScorer,
        GeneScorer,
    )

    # Mock the core dependencies since they're imported from scoring_core
    with patch.dict(
        "sys.modules",
        {
            "scoring_core": Mock(),
        },
    ):
        # Mock the imported enums and classes
        ScoringMethod = Mock()
        ScoringMethod.Z_SCORE = "z_score"
        ScoringMethod.COMBINED = "combined"
        ScoringMethod.EFFECT_SIZE = "effect_size"
        ScoringMethod.FDR_INVERSE = "fdr_inverse"
        ScoringMethod.FISHER_MIN = "fisher_min"

        AggregationMethod = Mock()
        AggregationMethod.MEAN = "mean"
        AggregationMethod.MEDIAN = "median"
        AggregationMethod.MAX = "max"
        AggregationMethod.SUM = "sum"
        AggregationMethod.WEIGHTED_MEAN = "weighted_mean"
        AggregationMethod.GSEA_LIKE = "gsea_like"

        ScoringConfig = Mock()
        PathwayConfig = Mock()
        MetaObjectValidator = Mock()
        DataFrameUtils = Mock()
        StatisticalUtils = Mock()
        LoggerMixin = Mock()
        ValidationError = Exception
        ScoringCalculationError = Exception
        ConfigurationError = Exception

except ImportError:
    # If the module can't be imported, create mock classes for testing structure
    pytest.skip("Module dependencies not available", allow_module_level=True)


class TestEnhancedScoringConfig:
    """Test suite for EnhancedScoringConfig class."""

    def test_enhanced_config_default_values(self):
        """Test default values of enhanced configuration."""
        config = EnhancedScoringConfig()

        # Test enhanced defaults
        assert config.adaptive_normalization is True
        assert config.normalization_per_study is False
        assert config.outlier_handling == "robust"
        assert config.enable_bootstrap is False
        assert config.bootstrap_iterations == 1000
        assert config.bootstrap_confidence == 0.95
        assert config.enable_cross_validation is False
        assert config.cv_folds == 5
        assert config.cv_scoring_metric == "stability"
        assert config.parallel_processing is False
        assert config.chunk_size == 1000
        assert config.cache_intermediate is True

    def test_enhanced_config_custom_values(self):
        """Test enhanced configuration with custom values."""
        config = EnhancedScoringConfig(
            adaptive_normalization=False,
            normalization_per_study=True,
            outlier_handling="clip",
            enable_bootstrap=True,
            bootstrap_iterations=500,
            bootstrap_confidence=0.90,
            enable_cross_validation=True,
            cv_folds=3,
            cv_scoring_metric="correlation",
            parallel_processing=True,
            chunk_size=500,
            cache_intermediate=False,
        )

        assert config.adaptive_normalization is False
        assert config.normalization_per_study is True
        assert config.outlier_handling == "clip"
        assert config.enable_bootstrap is True
        assert config.bootstrap_iterations == 500
        assert config.bootstrap_confidence == 0.90
        assert config.enable_cross_validation is True
        assert config.cv_folds == 3
        assert config.cv_scoring_metric == "correlation"
        assert config.parallel_processing is True
        assert config.chunk_size == 500
        assert config.cache_intermediate is False

    def test_enhanced_config_invalid_outlier_handling(self):
        """Test validation of outlier handling parameter."""
        with pytest.raises(ValueError, match="outlier_handling must be one of"):
            EnhancedScoringConfig(outlier_handling="invalid")

    def test_enhanced_config_invalid_bootstrap_confidence(self):
        """Test validation of bootstrap confidence parameter."""
        with pytest.raises(ValueError, match="bootstrap_confidence must be between 0 and 1"):
            EnhancedScoringConfig(bootstrap_confidence=1.5)

        with pytest.raises(ValueError, match="bootstrap_confidence must be between 0 and 1"):
            EnhancedScoringConfig(bootstrap_confidence=0.0)

    def test_enhanced_config_invalid_cv_folds(self):
        """Test validation of cross-validation folds parameter."""
        with pytest.raises(ValueError, match="cv_folds must be at least 2"):
            EnhancedScoringConfig(cv_folds=1)

    def test_enhanced_config_valid_outlier_handling_values(self):
        """Test all valid outlier handling values."""
        for method in ["clip", "remove", "robust", "none"]:
            config = EnhancedScoringConfig(outlier_handling=method)
            assert config.outlier_handling == method

    def test_enhanced_config_boundary_values(self):
        """Test boundary values for configuration parameters."""
        # Test minimum valid values
        config = EnhancedScoringConfig(
            bootstrap_confidence=0.01, cv_folds=2, bootstrap_iterations=1, chunk_size=1
        )
        assert config.bootstrap_confidence == 0.01
        assert config.cv_folds == 2
        assert config.bootstrap_iterations == 1
        assert config.chunk_size == 1

        # Test maximum valid values
        config = EnhancedScoringConfig(
            bootstrap_confidence=0.99,
            cv_folds=100,
            bootstrap_iterations=10000,
            chunk_size=100000,
        )
        assert config.bootstrap_confidence == 0.99
        assert config.cv_folds == 100
        assert config.bootstrap_iterations == 10000
        assert config.chunk_size == 100000


class TestScoringStrategy:
    """Test suite for ScoringStrategy abstract base class."""

    def test_scoring_strategy_is_abstract(self):
        """Test that ScoringStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ScoringStrategy()

    def test_concrete_strategy_implements_interface(self):
        """Test that concrete strategies implement the required interface."""
        strategy = ZScoreStrategy()

        # Should have required methods
        assert hasattr(strategy, "compute_scores")
        assert hasattr(strategy, "get_strategy_name")
        assert callable(strategy.compute_scores)
        assert callable(strategy.get_strategy_name)


class TestZScoreStrategy:
    """Test suite for ZScoreStrategy class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.strategy = ZScoreStrategy()
        self.config = EnhancedScoringConfig()

    def test_get_strategy_name(self):
        """Test strategy name retrieval."""
        assert self.strategy.get_strategy_name() == "z_score"

    def test_compute_scores_valid_data(self):
        """Test z-score computation with valid data."""
        data = pd.DataFrame(
            {"effect_size": [1.0, 2.0, 3.0, 4.0, 5.0]},
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
        )

        scores = self.strategy.compute_scores(data, self.config)

        # Check that we get z-scores
        assert len(scores) == 5
        assert isinstance(scores, pd.Series)
        assert abs(scores.mean()) < 1e-10  # Should be approximately zero
        assert abs(scores.std() - 1.0) < 1e-10  # Should have unit variance

    def test_compute_scores_missing_column(self):
        """Test computation with missing effect_size column."""
        data = pd.DataFrame({"other_column": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="Z-score strategy requires 'effect_size' column"):
            self.strategy.compute_scores(data, self.config)

    def test_compute_scores_with_missing_values(self):
        """Test computation with missing values in effect_size."""
        data = pd.DataFrame(
            {"effect_size": [1.0, np.nan, 3.0, 4.0, np.nan]},
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
        )

        # Test median imputation
        config = EnhancedScoringConfig(handle_missing="median")
        scores = self.strategy.compute_scores(data, config)
        assert not scores.isna().any()
        assert len(scores) == 5

        # Test zero imputation
        config = EnhancedScoringConfig(handle_missing="zero")
        scores = self.strategy.compute_scores(data, config)
        assert not scores.isna().any()
        assert len(scores) == 5

        # Test drop missing
        config = EnhancedScoringConfig(handle_missing="drop")
        scores = self.strategy.compute_scores(data, config)
        assert not scores.isna().any()
        assert len(scores) == 3  # Only non-missing values

    def test_compute_scores_single_value(self):
        """Test computation with single value (no variance)."""
        data = pd.DataFrame({"effect_size": [5.0]}, index=["GENE1"])

        scores = self.strategy.compute_scores(data, self.config)
        assert len(scores) == 1
        assert scores.iloc[0] == 0  # Single value should give z-score of 0

    def test_compute_scores_zero_variance(self):
        """Test computation with zero variance data."""
        data = pd.DataFrame(
            {"effect_size": [2.0, 2.0, 2.0, 2.0]},
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        scores = self.strategy.compute_scores(data, self.config)
        assert len(scores) == 4
        # All scores should be 0 when there's no variance
        assert all(abs(score) < 1e-10 for score in scores)


class TestCombinedStrategy:
    """Test suite for CombinedStrategy class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.strategy = CombinedStrategy()
        self.config = EnhancedScoringConfig(
            weights={
                "effect_size": 0.4,
                "effect_fdr": 0.3,
                "fisher_score": 0.2,
                "heterogeneity": 0.1,
            }
        )

    def test_get_strategy_name(self):
        """Test strategy name retrieval."""
        assert self.strategy.get_strategy_name() == "combined"

    def test_compute_scores_all_components(self):
        """Test combined scoring with all components available."""
        data = pd.DataFrame(
            {
                "effect_size": [1.0, 2.0, 3.0],
                "effect_fdr": [0.05, 0.01, 0.1],
                "fisher_score": [5.0, 8.0, 3.0],
                "heterogeneity": [0.2, 0.1, 0.3],
            },
            index=["GENE1", "GENE2", "GENE3"],
        )

        scores = self.strategy.compute_scores(data, self.config)

        assert len(scores) == 3
        assert isinstance(scores, pd.Series)
        assert not scores.isna().any()

    def test_compute_scores_partial_components(self):
        """Test combined scoring with only some components available."""
        data = pd.DataFrame(
            {"effect_size": [1.0, 2.0, 3.0], "effect_fdr": [0.05, 0.01, 0.1]},
            index=["GENE1", "GENE2", "GENE3"],
        )

        scores = self.strategy.compute_scores(data, self.config)

        assert len(scores) == 3
        assert isinstance(scores, pd.Series)
        assert not scores.isna().any()

    def test_compute_scores_no_components(self):
        """Test combined scoring with no recognized components."""
        data = pd.DataFrame(
            {"unrecognized_column": [1.0, 2.0, 3.0]}, index=["GENE1", "GENE2", "GENE3"]
        )

        scores = self.strategy.compute_scores(data, self.config)

        # Should return zeros when no components are available
        assert len(scores) == 3
        assert all(score == 0.0 for score in scores)

    def test_normalize_component_min_max(self):
        """Test component normalization with min-max method."""
        component = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        config = EnhancedScoringConfig(normalization_method="min_max")

        normalized = self.strategy._normalize_component(component, config)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert len(normalized) == 5

    def test_normalize_component_z_score(self):
        """Test component normalization with z-score method."""
        component = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        config = EnhancedScoringConfig(normalization_method="z_score")

        normalized = self.strategy._normalize_component(component, config)

        assert abs(normalized.mean()) < 1e-10
        assert abs(normalized.std() - 1.0) < 1e-10

    def test_normalize_component_robust(self):
        """Test component normalization with robust method."""
        component = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])  # Include outlier
        config = EnhancedScoringConfig(normalization_method="robust")

        normalized = self.strategy._normalize_component(component, config)

        # Median should be around 0 after robust scaling
        assert abs(normalized.median()) < 1e-10
        assert len(normalized) == 5

    def test_handle_outliers_clip(self):
        """Test outlier handling with clipping method."""
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier

        clipped = self.strategy._handle_outliers(data, "clip")

        # Outlier should be clipped
        assert clipped.max() < 100
        assert len(clipped) == 6

    def test_handle_outliers_remove(self):
        """Test outlier handling with removal method."""
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier

        removed = self.strategy._handle_outliers(data, "remove")

        # Outlier should be removed
        assert len(removed) < 6
        assert 100 not in removed.values

    def test_handle_outliers_robust(self):
        """Test outlier handling with robust method."""
        data = pd.Series([1, 2, 3, 4, 5, 100])

        # Robust method should return data unchanged
        result = self.strategy._handle_outliers(data, "robust")
        pd.testing.assert_series_equal(result, data)

    def test_normalize_component_with_missing_values(self):
        """Test component normalization with missing values."""
        component = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        config = EnhancedScoringConfig(handle_missing="median", normalization_method="min_max")

        normalized = self.strategy._normalize_component(component, config)

        assert not normalized.isna().any()
        assert len(normalized) == 5

    def test_normalize_component_no_variance(self):
        """Test component normalization with no variance."""
        component = pd.Series([3.0, 3.0, 3.0, 3.0])
        config = EnhancedScoringConfig(normalization_method="z_score")

        normalized = self.strategy._normalize_component(component, config)

        # Should return original data when no variance
        pd.testing.assert_series_equal(normalized, component)


class TestEffectSizeStrategy:
    """Test suite for EffectSizeStrategy class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.strategy = EffectSizeStrategy()
        self.config = EnhancedScoringConfig()

    def test_get_strategy_name(self):
        """Test strategy name retrieval."""
        assert self.strategy.get_strategy_name() == "effect_size"

    def test_compute_scores_valid_data(self):
        """Test effect size scoring with valid data."""
        data = pd.DataFrame(
            {"effect_size": [1.0, -2.0, 3.0, -4.0, 5.0]},
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
        )

        scores = self.strategy.compute_scores(data, self.config)

        assert len(scores) == 5
        assert isinstance(scores, pd.Series)
        # Should return absolute values
        expected = pd.Series(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
        )
        pd.testing.assert_series_equal(scores, expected)

    def test_compute_scores_missing_column(self):
        """Test computation with missing effect_size column."""
        data = pd.DataFrame({"other_column": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="Effect size strategy requires 'effect_size' column"):
            self.strategy.compute_scores(data, self.config)

    def test_compute_scores_with_missing_values(self):
        """Test computation preserves missing values."""
        data = pd.DataFrame(
            {"effect_size": [1.0, np.nan, -3.0, 4.0]},
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        scores = self.strategy.compute_scores(data, self.config)

        assert len(scores) == 4
        assert scores.iloc[0] == 1.0
        assert np.isnan(scores.iloc[1])
        assert scores.iloc[2] == 3.0
        assert scores.iloc[3] == 4.0

    def test_compute_scores_zero_values(self):
        """Test computation with zero effect sizes."""
        data = pd.DataFrame({"effect_size": [0.0, 0.0, 0.0]}, index=["GENE1", "GENE2", "GENE3"])

        scores = self.strategy.compute_scores(data, self.config)

        assert len(scores) == 3
        assert all(score == 0.0 for score in scores)


class TestFDRInverseStrategy:
    """Test suite for FDRInverseStrategy class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.strategy = FDRInverseStrategy()
        self.config = EnhancedScoringConfig()

    def test_get_strategy_name(self):
        """Test strategy name retrieval."""
        assert self.strategy.get_strategy_name() == "fdr_inverse"

    def test_compute_scores_effect_fdr(self):
        """Test FDR inverse scoring with effect_fdr column."""
        data = pd.DataFrame(
            {"effect_fdr": [0.05, 0.01, 0.001, 0.1]},
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        scores = self.strategy.compute_scores(data, self.config)

        assert len(scores) == 4
        assert isinstance(scores, pd.Series)
        # Lower FDR should give higher score
        assert scores["GENE3"] > scores["GENE2"] > scores["GENE1"] > scores["GENE4"]

    def test_compute_scores_fdr_column(self):
        """Test FDR inverse scoring with fdr column."""
        data = pd.DataFrame({"fdr": [0.05, 0.01, 0.001]}, index=["GENE1", "GENE2", "GENE3"])

        scores = self.strategy.compute_scores(data, self.config)

        assert len(scores) == 3
        assert scores["GENE3"] > scores["GENE2"] > scores["GENE1"]

    def test_compute_scores_adj_pvalue(self):
        """Test FDR inverse scoring with adj_pvalue column."""
        data = pd.DataFrame({"adj_pvalue": [0.05, 0.01, 0.001]}, index=["GENE1", "GENE2", "GENE3"])

        scores = self.strategy.compute_scores(data, self.config)

        assert len(scores) == 3
        assert scores["GENE3"] > scores["GENE2"] > scores["GENE1"]

    def test_compute_scores_multiple_fdr_columns(self):
        """Test FDR inverse scoring when multiple FDR columns exist."""
        data = pd.DataFrame(
            {
                "effect_fdr": [0.05, 0.01],
                "fdr": [0.1, 0.2],  # Should be ignored in favor of effect_fdr
                "adj_pvalue": [0.15, 0.25],  # Should be ignored
            },
            index=["GENE1", "GENE2"],
        )

        scores = self.strategy.compute_scores(data, self.config)

        assert len(scores) == 2
        # Should use effect_fdr, so GENE2 should have higher score
        assert scores["GENE2"] > scores["GENE1"]

    def test_compute_scores_no_fdr_column(self):
        """Test FDR inverse scoring with no FDR columns."""
        data = pd.DataFrame({"other_column": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="FDR inverse strategy requires FDR column"):
            self.strategy.compute_scores(data, self.config)

    def test_compute_scores_extreme_values(self):
        """Test FDR inverse scoring with extreme FDR values."""
        data = pd.DataFrame(
            {"effect_fdr": [1e-20, 1.0, 0.5]},  # Very small, maximum, and medium values
            index=["GENE1", "GENE2", "GENE3"],
        )

        scores = self.strategy.compute_scores(data, self.config)

        assert len(scores) == 3
        assert scores["GENE1"] > scores["GENE3"] > scores["GENE2"]
        assert np.isfinite(scores).all()  # Should not have infinite values


class TestFisherMinStrategy:
    """Test suite for FisherMinStrategy class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.strategy = FisherMinStrategy()
        self.config = EnhancedScoringConfig()

    def test_get_strategy_name(self):
        """Test strategy name retrieval."""
        assert self.strategy.get_strategy_name() == "fisher_min"

    def test_compute_scores_valid_data(self):
        """Test Fisher min scoring with valid data."""
        data = pd.DataFrame(
            {"fisher_score": [1.0, 2.0, 3.0, 4.0]},
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        scores = self.strategy.compute_scores(data, self.config)

        assert len(scores) == 4
        assert isinstance(scores, pd.Series)
        # Should return the fisher_score column as-is
        expected = pd.Series([1.0, 2.0, 3.0, 4.0], index=["GENE1", "GENE2", "GENE3", "GENE4"])
        pd.testing.assert_series_equal(scores, expected)

    def test_compute_scores_missing_column(self):
        """Test computation with missing fisher_score column."""
        data = pd.DataFrame({"other_column": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="Fisher min strategy requires 'fisher_score' column"):
            self.strategy.compute_scores(data, self.config)

    def test_compute_scores_with_missing_values(self):
        """Test computation preserves missing values."""
        data = pd.DataFrame(
            {"fisher_score": [1.0, np.nan, 3.0, 4.0]},
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        scores = self.strategy.compute_scores(data, self.config)

        assert len(scores) == 4
        assert scores.iloc[0] == 1.0
        assert np.isnan(scores.iloc[1])
        assert scores.iloc[2] == 3.0
        assert scores.iloc[3] == 4.0

    def test_compute_scores_negative_values(self):
        """Test computation with negative Fisher scores."""
        data = pd.DataFrame(
            {"fisher_score": [-1.0, -2.0, 3.0, 4.0]},
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        scores = self.strategy.compute_scores(data, self.config)

        assert len(scores) == 4
        # Should preserve negative values
        assert scores["GENE1"] == -1.0
        assert scores["GENE2"] == -2.0


class TestPathwayAggregator:
    """Test suite for PathwayAggregator class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.aggregator = PathwayAggregator()
        self.gene_scores = pd.Series(
            {"GENE1": 1.0, "GENE2": 2.0, "GENE3": 3.0, "GENE4": 4.0, "GENE5": 5.0}
        )

    def test_aggregate_pathway_scores_mean(self):
        """Test pathway aggregation using mean method."""
        pathway_genes = ["GENE1", "GENE2", "GENE3"]

        score = self.aggregator.aggregate_pathway_scores(
            self.gene_scores, pathway_genes, AggregationMethod.MEAN
        )

        assert score == 2.0  # (1+2+3)/3

    def test_aggregate_pathway_scores_median(self):
        """Test pathway aggregation using median method."""
        pathway_genes = ["GENE1", "GENE2", "GENE3"]

        score = self.aggregator.aggregate_pathway_scores(
            self.gene_scores, pathway_genes, AggregationMethod.MEDIAN
        )

        assert score == 2.0  # median of [1,2,3]

    def test_aggregate_pathway_scores_max(self):
        """Test pathway aggregation using max method."""
        pathway_genes = ["GENE1", "GENE2", "GENE3"]

        score = self.aggregator.aggregate_pathway_scores(
            self.gene_scores, pathway_genes, AggregationMethod.MAX
        )

        assert score == 3.0  # max absolute value

    def test_aggregate_pathway_scores_sum(self):
        """Test pathway aggregation using sum method."""
        pathway_genes = ["GENE1", "GENE2", "GENE3"]

        score = self.aggregator.aggregate_pathway_scores(
            self.gene_scores, pathway_genes, AggregationMethod.SUM
        )

        assert score == 6.0  # 1+2+3

    def test_aggregate_pathway_scores_weighted_mean(self):
        """Test pathway aggregation using weighted mean method."""
        pathway_genes = ["GENE1", "GENE2", "GENE3"]
        weights = {"GENE1": 1.0, "GENE2": 2.0, "GENE3": 1.0}

        score = self.aggregator.aggregate_pathway_scores(
            self.gene_scores, pathway_genes, AggregationMethod.WEIGHTED_MEAN, weights
        )

        # (1*1 + 2*2 + 3*1) / (1+2+1) = 8/4 = 2.0
        assert score == 2.0

    def test_aggregate_pathway_scores_weighted_mean_no_weights(self):
        """Test pathway aggregation using weighted mean with no weights provided."""
        pathway_genes = ["GENE1", "GENE2", "GENE3"]

        score = self.aggregator.aggregate_pathway_scores(
            self.gene_scores, pathway_genes, AggregationMethod.WEIGHTED_MEAN
        )

        # Should fall back to regular mean
        assert score == 2.0

    @patch.object(PathwayAggregator, "_gsea_like_score")
    def test_aggregate_pathway_scores_gsea_like(self, mock_gsea):
        """Test pathway aggregation using GSEA-like method."""
        mock_gsea.return_value = 0.5
        pathway_genes = ["GENE1", "GENE2", "GENE3"]

        score = self.aggregator.aggregate_pathway_scores(
            self.gene_scores, pathway_genes, AggregationMethod.GSEA_LIKE
        )

        assert score == 0.5
        mock_gsea.assert_called_once()

    def test_aggregate_pathway_scores_no_genes(self):
        """Test pathway aggregation with empty gene list."""
        pathway_genes = []

        score = self.aggregator.aggregate_pathway_scores(
            self.gene_scores, pathway_genes, AggregationMethod.MEAN
        )

        assert score == 0.0

    def test_aggregate_pathway_scores_no_overlap(self):
        """Test pathway aggregation with no overlapping genes."""
        pathway_genes = ["GENE_X", "GENE_Y", "GENE_Z"]

        score = self.aggregator.aggregate_pathway_scores(
            self.gene_scores, pathway_genes, AggregationMethod.MEAN
        )

        assert score == 0.0

    def test_aggregate_pathway_scores_partial_overlap(self):
        """Test pathway aggregation with partial gene overlap."""
        pathway_genes = ["GENE1", "GENE2", "GENE_X", "GENE_Y"]

        score = self.aggregator.aggregate_pathway_scores(
            self.gene_scores, pathway_genes, AggregationMethod.MEAN
        )

        # Should only use GENE1 and GENE2: (1+2)/2 = 1.5
        assert score == 1.5

    def test_aggregate_pathway_scores_unknown_method(self):
        """Test pathway aggregation with unknown method."""
        pathway_genes = ["GENE1", "GENE2", "GENE3"]

        score = self.aggregator.aggregate_pathway_scores(
            self.gene_scores, pathway_genes, "unknown_method"
        )

        # Should return 0.0 for unknown methods
        assert score == 0.0

    def test_gsea_like_score(self):
        """Test GSEA-like enrichment score calculation."""
        pathway_scores = pd.Series({"GENE1": 1.0, "GENE3": 3.0})
        all_scores = self.gene_scores

        enrichment_score = self.aggregator._gsea_like_score(pathway_scores, all_scores)

        assert isinstance(enrichment_score, float)
        assert np.isfinite(enrichment_score)

    def test_gsea_like_score_empty_pathway(self):
        """Test GSEA-like score with empty pathway."""
        pathway_scores = pd.Series(dtype=float)
        all_scores = self.gene_scores

        enrichment_score = self.aggregator._gsea_like_score(pathway_scores, all_scores)

        assert enrichment_score == 0.0

    def test_gsea_like_score_no_overlap(self):
        """Test GSEA-like score with no overlap between pathway and all scores."""
        pathway_scores = pd.Series({"GENE_X": 1.0, "GENE_Y": 2.0})
        all_scores = self.gene_scores

        enrichment_score = self.aggregator._gsea_like_score(pathway_scores, all_scores)

        assert enrichment_score == 0.0


class TestEnhancedGeneScorer:
    """Test suite for EnhancedGeneScorer class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = EnhancedScoringConfig()
        self.scorer = EnhancedGeneScorer(self.config)

        # Mock meta object
        self.meta_object = {
            "original_result": pd.DataFrame(
                {
                    "effect_size": [1.0, 2.0, 3.0, 4.0],
                    "effect_fdr": [0.05, 0.01, 0.001, 0.1],
                    "fisher_score": [5.0, 8.0, 12.0, 3.0],
                },
                index=["GENE1", "GENE2", "GENE3", "GENE4"],
            )
        }

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        scorer = EnhancedGeneScorer()

        assert isinstance(scorer.config, EnhancedScoringConfig)
        assert hasattr(scorer, "strategies")
        assert hasattr(scorer, "pathway_aggregator")
        assert hasattr(scorer, "_score_cache")

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = EnhancedScoringConfig(enable_bootstrap=True)
        scorer = EnhancedGeneScorer(custom_config)

        assert scorer.config == custom_config
        assert scorer.config.enable_bootstrap is True

    @patch("metaintegrator.scoring_gene_scorer.MetaObjectValidator")
    def test_score_genes_success(self, mock_validator):
        """Test successful gene scoring."""
        mock_validator.validate_and_extract.return_value = None

        scores = self.scorer.score_genes(self.meta_object, ScoringMethod.EFFECT_SIZE)

        assert isinstance(scores, pd.Series)
        assert len(scores) == 4
        assert not scores.isna().any()
        mock_validator.validate_and_extract.assert_called_once_with(self.meta_object)

    @patch("metaintegrator.scoring_gene_scorer.MetaObjectValidator")
    def test_score_genes_validation_error(self, mock_validator):
        """Test gene scoring with validation error."""
        mock_validator.validate_and_extract.side_effect = ValidationError("Invalid meta object")

        with pytest.raises(ScoringCalculationError):
            self.scorer.score_genes(self.meta_object)

    def test_score_genes_invalid_method(self):
        """Test gene scoring with invalid method."""
        # The ValueError exception (Unknown scoring method: invalid_method) thrown by
        # scoring_gene_scoring.score_genes () was caught and rethrown as ScoringCalculationError
        with pytest.raises(ScoringCalculationError, match="Unknown scoring method"):
            self.scorer.score_genes(self.meta_object, "invalid_method")

    @patch("metaintegrator.scoring_gene_scorer.MetaObjectValidator")
    def test_score_genes_with_caching(self, mock_validator):
        """Test gene scoring with caching enabled."""
        mock_validator.validate.return_value = None
        config = EnhancedScoringConfig(cache_intermediate=True)
        scorer = EnhancedGeneScorer(config)

        # First call
        scores1 = scorer.score_genes(self.meta_object, ScoringMethod.EFFECT_SIZE)

        # Second call should use cache
        scores2 = scorer.score_genes(self.meta_object, ScoringMethod.EFFECT_SIZE)

        pd.testing.assert_series_equal(scores1, scores2)
        assert len(scorer._score_cache) > 0

    @patch("metaintegrator.scoring_gene_scorer.MetaObjectValidator")
    def test_score_genes_with_bootstrap(self, mock_validator):
        """Test gene scoring with bootstrap confidence intervals."""
        mock_validator.validate.return_value = None
        config = EnhancedScoringConfig(
            enable_bootstrap=True, bootstrap_iterations=10  # Small number for testing
        )
        scorer = EnhancedGeneScorer(config)

        scores = scorer.score_genes(self.meta_object, ScoringMethod.EFFECT_SIZE)

        assert isinstance(scores, pd.Series)
        confidence_intervals = scorer.get_last_confidence_intervals()
        assert confidence_intervals is not None
        assert len(confidence_intervals) > 0

    @patch("metaintegrator.scoring_gene_scorer.MetaObjectValidator")
    @patch("sklearn.model_selection.KFold")
    def test_score_genes_with_cross_validation(self, mock_kfold, mock_validator):
        """Test gene scoring with cross-validation reliability assessment."""
        mock_validator.validate.return_value = None

        # Mock KFold to return some splits
        mock_kfold.return_value.split.return_value = [
            (np.array([0, 1]), np.array([2, 3])),
            (np.array([2, 3]), np.array([0, 1])),
        ]

        config = EnhancedScoringConfig(enable_cross_validation=True, cv_folds=2)
        scorer = EnhancedGeneScorer(config)

        scores = scorer.score_genes(self.meta_object, ScoringMethod.EFFECT_SIZE)

        assert isinstance(scores, pd.Series)
        reliability_score = scorer.get_last_reliability_score()
        assert reliability_score is not None
        assert isinstance(reliability_score, float)

    def test_score_pathways(self):
        """Test pathway scoring functionality."""
        gene_scores = pd.Series({"GENE1": 1.0, "GENE2": 2.0, "GENE3": 3.0, "GENE4": 4.0})

        # Mock pathway config
        pathway_config = Mock()
        pathway_config.pathways = {
            "PATHWAY1": ["GENE1", "GENE2"],
            "PATHWAY2": ["GENE3", "GENE4"],
            "PATHWAY3": ["GENE1", "GENE3", "GENE5"],  # GENE5 not in scores
        }
        pathway_config.gene_weights = None

        pathway_scores = self.scorer.score_pathways(
            gene_scores, pathway_config, AggregationMethod.MEAN
        )

        assert isinstance(pathway_scores, dict)
        assert "PATHWAY1" in pathway_scores
        assert "PATHWAY2" in pathway_scores
        assert "PATHWAY3" in pathway_scores

        # PATHWAY1: (1+2)/2 = 1.5
        assert pathway_scores["PATHWAY1"] == 1.5
        # PATHWAY2: (3+4)/2 = 3.5
        assert pathway_scores["PATHWAY2"] == 3.5

    def test_score_pathways_insufficient_genes(self):
        """Test pathway scoring with insufficient genes."""
        gene_scores = pd.Series({"GENE1": 1.0, "GENE2": 2.0})

        # Configure to require at least 3 genes
        config = EnhancedScoringConfig(min_valid_genes=3)
        scorer = EnhancedGeneScorer(config)

        pathway_config = Mock()
        pathway_config.pathways = {
            "PATHWAY1": ["GENE1", "GENE2"]  # Only 2 genes, less than minimum
        }
        pathway_config.gene_weights = None

        pathway_scores = scorer.score_pathways(gene_scores, pathway_config, AggregationMethod.MEAN)

        assert pathway_scores["PATHWAY1"] == 0.0

    def test_score_pathways_empty_pathway(self):
        """Test pathway scoring with empty pathway."""
        gene_scores = pd.Series({"GENE1": 1.0, "GENE2": 2.0})

        pathway_config = Mock()
        pathway_config.pathways = {"EMPTY_PATHWAY": []}
        pathway_config.gene_weights = None

        pathway_scores = self.scorer.score_pathways(
            gene_scores, pathway_config, AggregationMethod.MEAN
        )

        assert pathway_scores["EMPTY_PATHWAY"] == 0.0

    def test_extract_gene_data_original_result(self):
        """Test extraction of gene data from original_result."""
        meta_object = {"original_result": pd.DataFrame({"col1": [1, 2, 3]})}

        data = self.scorer._extract_gene_data(meta_object)

        assert isinstance(data, pd.DataFrame)
        assert "col1" in data.columns

    def test_extract_gene_data_filter_res(self):
        """Test extraction of gene data from filterRes."""
        meta_object = {"filterRes": pd.DataFrame({"col1": [1, 2, 3]})}

        data = self.scorer._extract_gene_data(meta_object)

        assert isinstance(data, pd.DataFrame)
        assert "col1" in data.columns

    def test_extract_gene_data_missing_keys(self):
        """Test extraction of gene data with missing keys."""
        meta_object = {"other_key": pd.DataFrame({"col1": [1, 2, 3]})}

        with pytest.raises(ValueError, match="Meta object must contain"):
            self.scorer._extract_gene_data(meta_object)

    @patch("metaintegrator.scoring_gene_scorer.DataFrameUtils")
    def test_preprocess_data_per_study_normalization(self, mock_df_utils):
        """Test data preprocessing with per-study normalization."""
        data = pd.DataFrame(
            {
                "study": ["study1", "study1", "study2", "study2"],
                "effect_size": [1.0, 2.0, 3.0, 4.0],
                "other_col": ["a", "b", "c", "d"],
            }
        )

        mock_df_utils.get_numeric_columns.return_value = ["effect_size"]

        config = EnhancedScoringConfig(normalization_per_study=True)
        scorer = EnhancedGeneScorer(config)

        processed_data = scorer._preprocess_data(data)

        assert isinstance(processed_data, pd.DataFrame)
        assert "study" in processed_data.columns
        assert "effect_size" in processed_data.columns

    def test_preprocess_data_drop_missing(self):
        """Test data preprocessing with missing value dropping."""
        data = pd.DataFrame(
            {"effect_size": [1.0, np.nan, 3.0, 4.0], "other_col": [1, 2, 3, np.nan]}
        )

        config = EnhancedScoringConfig(handle_missing="drop")
        scorer = EnhancedGeneScorer(config)

        processed_data = scorer._preprocess_data(data)

        # Should drop rows with any NaN values
        assert len(processed_data) < len(data)

    @patch("metaintegrator.scoring_gene_scorer.MinMaxScaler")
    def test_apply_normalization_min_max(self, mock_scaler):
        """Test normalization application with min-max method."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0])
        mock_scaler_instance = Mock()
        mock_scaler.return_value = mock_scaler_instance
        mock_scaler_instance.fit_transform.return_value = np.array(
            [[0], [0.33], [0.67], [1]]
        ).flatten()

        result = self.scorer._apply_normalization(data, "min_max")

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        mock_scaler.assert_called_once()

    def test_postprocess_scores(self):
        """Test score post-processing."""
        scores = pd.Series([1.0, np.inf, 3.0, np.nan, -np.inf])
        data = pd.DataFrame()  # Not used in current implementation

        processed = self.scorer._postprocess_scores(scores, data)

        # Should handle infinite and NaN values
        assert not np.isinf(processed).any()
        assert len(processed) >= 0  # Depending on handle_missing setting

    def test_compute_bootstrap_confidence(self):
        """Test bootstrap confidence interval computation."""
        data = pd.DataFrame(
            {"effect_size": [1.0, 2.0, 3.0, 4.0, 5.0]},
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
        )

        original_scores = pd.Series(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
        )

        config = EnhancedScoringConfig(bootstrap_iterations=10, bootstrap_confidence=0.95)
        scorer = EnhancedGeneScorer(config)

        confidence_intervals = scorer._compute_bootstrap_confidence(
            data, ScoringMethod.EFFECT_SIZE, original_scores
        )

        assert isinstance(confidence_intervals, dict)
        assert len(confidence_intervals) == 5
        for gene, (lower, upper) in confidence_intervals.items():
            assert gene in original_scores.index
            # Check that we have valid intervals (could be NaN if bootstrap failed)
            if not (np.isnan(lower) or np.isnan(upper)):
                assert lower <= upper

    @patch("sklearn.model_selection.KFold")
    def test_assess_scoring_reliability(self, mock_kfold):
        """Test scoring reliability assessment."""
        data = pd.DataFrame({"effect_size": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

        # Mock KFold splits
        mock_kfold.return_value.split.return_value = [
            (np.array([0, 1, 2]), np.array([3, 4, 5])),
            (np.array([3, 4, 5]), np.array([0, 1, 2])),
        ]

        config = EnhancedScoringConfig(cv_folds=2)
        scorer = EnhancedGeneScorer(config)

        reliability = scorer._assess_scoring_reliability(data, ScoringMethod.EFFECT_SIZE)

        assert isinstance(reliability, float)
        assert 0.0 <= reliability <= 1.0 or reliability == 0.0  # Could be 0 if correlation failed

    def test_generate_cache_key(self):
        """Test cache key generation."""
        data = pd.DataFrame({"col1": [1, 2, 3]})
        config = EnhancedScoringConfig()

        cache_key1 = self.scorer._generate_cache_key(data, ScoringMethod.EFFECT_SIZE, config)
        cache_key2 = self.scorer._generate_cache_key(data, ScoringMethod.EFFECT_SIZE, config)

        # Same inputs should generate same key
        assert cache_key1 == cache_key2

        # Different method should generate different key
        cache_key3 = self.scorer._generate_cache_key(data, ScoringMethod.Z_SCORE, config)
        assert cache_key1 != cache_key3

    def test_get_last_confidence_intervals(self):
        """Test retrieval of last confidence intervals."""
        # Initially should be None
        assert self.scorer.get_last_confidence_intervals() is None

        # Set some intervals
        intervals = {"GENE1": (0.5, 1.5), "GENE2": (1.0, 2.0)}
        self.scorer._last_confidence_intervals = intervals

        assert self.scorer.get_last_confidence_intervals() == intervals

    def test_get_last_reliability_score(self):
        """Test retrieval of last reliability score."""
        # Initially should be None
        assert self.scorer.get_last_reliability_score() is None

        # Set a reliability score
        self.scorer._last_reliability_score = 0.85

        assert self.scorer.get_last_reliability_score() == 0.85

    def test_clear_cache(self):
        """Test cache clearing."""
        # Add something to cache
        self.scorer._score_cache["test_key"] = "test_value"
        assert len(self.scorer._score_cache) > 0

        # Clear cache
        self.scorer.clear_cache()

        assert len(self.scorer._score_cache) == 0


class TestGeneScorer:
    """Test suite for legacy GeneScorer class."""

    def test_init_with_legacy_config(self):
        """Test initialization with legacy ScoringConfig."""
        legacy_config = Mock()
        legacy_config.weights = {"effect_size": 1.0}
        legacy_config.normalization_method = "min_max"
        legacy_config.handle_missing = "zero"
        legacy_config.min_valid_genes = 5

        scorer = GeneScorer(legacy_config)

        assert isinstance(scorer.config, EnhancedScoringConfig)
        # Should have converted legacy config fields
        assert scorer.config.weights == legacy_config.weights

    def test_init_with_enhanced_config(self):
        """Test initialization with EnhancedScoringConfig."""
        enhanced_config = EnhancedScoringConfig()
        scorer = GeneScorer(enhanced_config)

        assert scorer.config == enhanced_config

    def test_init_no_config(self):
        """Test initialization without config."""
        scorer = GeneScorer()

        assert isinstance(scorer.config, EnhancedScoringConfig)

    @patch("metaintegrator.scoring_gene_scorer.MetaObjectValidator")
    def test_score_method_legacy_compatibility(self, mock_validator):
        """Test legacy score method for backward compatibility."""
        mock_validator.validate.return_value = None

        meta_object = {"original_result": pd.DataFrame({"effect_size": [1.0, 2.0, 3.0]})}

        scorer = GeneScorer()
        scores = scorer.score(meta_object, ScoringMethod.EFFECT_SIZE)

        assert isinstance(scores, pd.Series)
        assert len(scores) == 3


class TestIntegration:
    """Integration tests for the entire module."""

    @patch("metaintegrator.scoring_gene_scorer.MetaObjectValidator")
    def test_end_to_end_gene_scoring_workflow(self, mock_validator):
        """Test complete gene scoring workflow."""
        mock_validator.validate.return_value = None

        # Create comprehensive test data
        meta_object = {
            "original_result": pd.DataFrame(
                {
                    "effect_size": [1.5, -2.0, 3.2, -0.8, 2.1],
                    "effect_fdr": [0.05, 0.01, 0.001, 0.1, 0.02],
                    "fisher_score": [5.2, 8.1, 12.3, 2.8, 6.7],
                    "heterogeneity": [0.2, 0.1, 0.05, 0.3, 0.15],
                },
                index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
            )
        }

        # Test different scoring strategies
        config = EnhancedScoringConfig(
            weights={
                "effect_size": 0.4,
                "effect_fdr": 0.3,
                "fisher_score": 0.2,
                "heterogeneity": 0.1,
            }
        )

        scorer = EnhancedGeneScorer(config)

        # Test each scoring method
        for method in [
            ScoringMethod.Z_SCORE,
            ScoringMethod.COMBINED,
            ScoringMethod.EFFECT_SIZE,
            ScoringMethod.FDR_INVERSE,
            ScoringMethod.FISHER_MIN,
        ]:
            try:
                scores = scorer.score_genes(meta_object, method)
                assert isinstance(scores, pd.Series)
                assert len(scores) == 5
                assert scores.index.equals(meta_object["original_result"].index)
            except Exception as e:
                pytest.fail(f"Scoring with method {method} failed: {e}")

    @patch("metaintegrator.scoring_gene_scorer.MetaObjectValidator")
    def test_end_to_end_pathway_scoring_workflow(self, mock_validator):
        """Test complete pathway scoring workflow."""
        mock_validator.validate.return_value = None

        meta_object = {
            "original_result": pd.DataFrame(
                {
                    "effect_size": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "effect_fdr": [0.05, 0.01, 0.001, 0.1, 0.02],
                },
                index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
            )
        }

        scorer = EnhancedGeneScorer()

        # Get gene scores
        gene_scores = scorer.score_genes(meta_object, ScoringMethod.EFFECT_SIZE)

        # Create pathway configuration
        pathway_config = Mock()
        pathway_config.pathways = {
            "PATHWAY_A": ["GENE1", "GENE2", "GENE3"],
            "PATHWAY_B": ["GENE3", "GENE4", "GENE5"],
            "PATHWAY_C": ["GENE1", "GENE5"],
        }
        pathway_config.gene_weights = {
            "GENE1": 1.5,
            "GENE2": 1.0,
            "GENE3": 2.0,
            "GENE4": 1.0,
            "GENE5": 1.2,
        }

        # Test different aggregation methods
        for agg_method in [
            AggregationMethod.MEAN,
            AggregationMethod.MEDIAN,
            AggregationMethod.MAX,
            AggregationMethod.SUM,
            AggregationMethod.WEIGHTED_MEAN,
        ]:
            try:
                pathway_scores = scorer.score_pathways(gene_scores, pathway_config, agg_method)
                assert isinstance(pathway_scores, dict)
                assert len(pathway_scores) == 3
                assert all(
                    pathway in pathway_scores for pathway in ["PATHWAY_A", "PATHWAY_B", "PATHWAY_C"]
                )
            except Exception as e:
                pytest.fail(f"Pathway scoring with method {agg_method} failed: {e}")

    @patch("metaintegrator.scoring_gene_scorer.MetaObjectValidator")
    def test_advanced_features_workflow(self, mock_validator):
        """Test workflow with advanced features enabled."""
        mock_validator.validate.return_value = None

        meta_object = {
            "original_result": pd.DataFrame(
                {
                    "effect_size": np.random.normal(0, 1, 20),
                    "effect_fdr": np.random.uniform(0.001, 0.1, 20),
                },
                index=[f"GENE{i}" for i in range(1, 21)],
            )
        }

        # Enable all advanced features with small values for testing
        config = EnhancedScoringConfig(
            enable_bootstrap=True,
            bootstrap_iterations=5,  # Small for testing
            enable_cross_validation=True,
            cv_folds=3,
            cache_intermediate=True,
            adaptive_normalization=True,
            outlier_handling="robust",
        )

        scorer = EnhancedGeneScorer(config)

        with patch("sklearn.model_selection.KFold") as mock_kfold:
            # Mock cross-validation splits
            mock_kfold.return_value.split.return_value = [
                (np.arange(0, 13), np.arange(13, 20)),
                (np.arange(7, 20), np.arange(0, 7)),
                (
                    np.concatenate([np.arange(0, 7), np.arange(13, 20)]),
                    np.arange(7, 13),
                ),
            ]

            # Score genes with advanced features
            scores = scorer.score_genes(meta_object, ScoringMethod.COMBINED)

            assert isinstance(scores, pd.Series)
            assert len(scores) == 20

            # Check that advanced features ran
            confidence_intervals = scorer.get_last_confidence_intervals()
            reliability_score = scorer.get_last_reliability_score()

            assert confidence_intervals is not None
            assert reliability_score is not None
            assert isinstance(reliability_score, float)

            # Check cache usage
            assert len(scorer._score_cache) > 0

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        scorer = EnhancedGeneScorer()

        # Test with invalid meta object
        invalid_meta = {"invalid_key": "invalid_value"}

        with pytest.raises(ScoringCalculationError):
            scorer.score_genes(invalid_meta)

        # Test with empty data
        empty_meta = {"original_result": pd.DataFrame()}

        with pytest.raises(ScoringCalculationError):
            scorer.score_genes(empty_meta, ScoringMethod.Z_SCORE)

    def test_configuration_validation_integration(self):
        """Test configuration validation in integrated scenarios."""
        # Test invalid configuration should fail early
        with pytest.raises(ValueError):
            EnhancedScoringConfig(outlier_handling="invalid", bootstrap_confidence=1.5, cv_folds=1)

        # Test that valid edge case configurations work
        edge_config = EnhancedScoringConfig(
            bootstrap_confidence=0.01, cv_folds=2, bootstrap_iterations=1, chunk_size=1
        )

        scorer = EnhancedGeneScorer(edge_config)
        assert scorer.config == edge_config


# Pytest configuration and fixtures
@pytest.fixture
def sample_gene_data():
    """Fixture providing sample gene expression data."""
    return pd.DataFrame(
        {
            "effect_size": [1.5, -2.0, 3.2, -0.8, 2.1, 0.0],
            "effect_fdr": [0.05, 0.01, 0.001, 0.1, 0.02, 0.5],
            "fisher_score": [5.2, 8.1, 12.3, 2.8, 6.7, 1.0],
            "heterogeneity": [0.2, 0.1, 0.05, 0.3, 0.15, 0.4],
        },
        index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6"],
    )


@pytest.fixture
def sample_pathway_config():
    """Fixture providing sample pathway configuration."""
    config = Mock()
    config.pathways = {
        "PATHWAY1": ["GENE1", "GENE2", "GENE3"],
        "PATHWAY2": ["GENE3", "GENE4", "GENE5"],
        "PATHWAY3": ["GENE1", "GENE6"],
    }
    config.gene_weights = {
        "GENE1": 1.5,
        "GENE2": 1.0,
        "GENE3": 2.0,
        "GENE4": 1.2,
        "GENE5": 0.8,
        "GENE6": 1.1,
    }
    return config


@pytest.fixture
def enhanced_config():
    """Fixture providing enhanced scoring configuration."""
    return EnhancedScoringConfig(
        weights={
            "effect_size": 0.4,
            "effect_fdr": 0.3,
            "fisher_score": 0.2,
            "heterogeneity": 0.1,
        },
        normalization_method="min_max",
        handle_missing="median",
        min_valid_genes=2,
        adaptive_normalization=True,
        outlier_handling="robust",
        cache_intermediate=True,
    )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
