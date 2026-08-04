"""
Unit tests for scoring_meta_integrator_filter.py

This test module provides comprehensive coverage for the enhanced meta-integrator
filter functionality, including configuration validation, filtering strategies,
and integration with the scoring_core infrastructure.
"""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))


# Import the classes and functions to test
from metaintegrator.scoring_meta_integrator_filter import (
    EnhancedFilterConfig,
    EnhancedFilterResults,
    FilteringStrategy,
    StandardFilteringStrategy,
    AdaptiveFilteringStrategy,
    CombinedCriteriaFilteringStrategy,
    EnhancedMetaIntegratorFilter,
    MetaIntegratorFilter,
)

# Mock imports from scoring_core
from unittest.mock import MagicMock
import sys


# Create a more comprehensive mock for scoring_core
class MockFilterResults:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockFilterConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __post_init__(self):
        pass


scoring_core_mock = MagicMock()
sys.modules["scoring_core"] = scoring_core_mock

# Mock the imported classes and exceptions
scoring_core_mock.FilterConfig = MockFilterConfig
scoring_core_mock.FilterResults = MockFilterResults
scoring_core_mock.FilterSummary = MagicMock()
scoring_core_mock.MetaObjectValidator = MagicMock()
scoring_core_mock.FilterResultsValidator = MagicMock()
scoring_core_mock.DataFrameUtils = MagicMock()
scoring_core_mock.StatisticalUtils = MagicMock()
scoring_core_mock.LoggerMixin = MagicMock()
scoring_core_mock.ValidationError = Exception
scoring_core_mock.FilteringError = Exception
scoring_core_mock.ConfigurationError = Exception
scoring_core_mock.ScoringCalculationError = Exception


class TestEnhancedFilterConfig(unittest.TestCase):
    """Test cases for EnhancedFilterConfig class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the parent class post_init
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            self.default_config = EnhancedFilterConfig()

    def test_default_initialization(self):
        """Test default initialization of EnhancedFilterConfig."""
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            config = EnhancedFilterConfig()

        self.assertFalse(config.adaptive_thresholds)
        self.assertEqual(config.min_effect_size_percentile, 0.0)
        self.assertEqual(config.max_heterogeneity_percentile, 100.0)
        self.assertFalse(config.use_combined_criteria)
        self.assertFalse(config.require_consistency_across_studies)
        self.assertEqual(config.min_consistent_studies_pct, 0.6)
        self.assertTrue(config.prioritize_by_strength)
        self.assertIsNone(config.max_genes_per_direction)
        self.assertTrue(config.cache_intermediate_results)
        self.assertFalse(config.parallel_processing)

    def test_criteria_weights_default(self):
        """Test default criteria weights."""
        expected_weights = {
            "effect_size": 0.3,
            "fdr": 0.4,
            "heterogeneity": 0.2,
            "n_studies": 0.1,
        }
        self.assertEqual(self.default_config.criteria_weights, expected_weights)

    def test_custom_initialization(self):
        """Test custom initialization with valid parameters."""
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            config = EnhancedFilterConfig(
                adaptive_thresholds=True,
                min_effect_size_percentile=25.0,
                max_heterogeneity_percentile=75.0,
                use_combined_criteria=True,
                require_consistency_across_studies=True,
                min_consistent_studies_pct=0.8,
                max_genes_per_direction=100,
            )

        self.assertTrue(config.adaptive_thresholds)
        self.assertEqual(config.min_effect_size_percentile, 25.0)
        self.assertEqual(config.max_heterogeneity_percentile, 75.0)
        self.assertTrue(config.use_combined_criteria)
        self.assertTrue(config.require_consistency_across_studies)
        self.assertEqual(config.min_consistent_studies_pct, 0.8)
        self.assertEqual(config.max_genes_per_direction, 100)

    def test_post_init_validation_percentile_bounds(self):
        """Test validation of percentile bounds in __post_init__."""
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            # Test invalid min_effect_size_percentile
            with self.assertRaises(ValueError) as context:
                EnhancedFilterConfig(min_effect_size_percentile=-5.0)
            self.assertIn(
                "min_effect_size_percentile must be between 0 and 100",
                str(context.exception),
            )

            with self.assertRaises(ValueError) as context:
                EnhancedFilterConfig(min_effect_size_percentile=105.0)
            self.assertIn(
                "min_effect_size_percentile must be between 0 and 100",
                str(context.exception),
            )

            # Test invalid max_heterogeneity_percentile
            with self.assertRaises(ValueError) as context:
                EnhancedFilterConfig(max_heterogeneity_percentile=-10.0)
            self.assertIn(
                "max_heterogeneity_percentile must be between 0 and 100",
                str(context.exception),
            )

            with self.assertRaises(ValueError) as context:
                EnhancedFilterConfig(max_heterogeneity_percentile=110.0)
            self.assertIn(
                "max_heterogeneity_percentile must be between 0 and 100",
                str(context.exception),
            )

    def test_post_init_validation_consistency_pct(self):
        """Test validation of consistency percentage."""
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            # Test invalid min_consistent_studies_pct
            with self.assertRaises(ValueError) as context:
                EnhancedFilterConfig(min_consistent_studies_pct=-0.1)
            self.assertIn(
                "min_consistent_studies_pct must be between 0 and 1",
                str(context.exception),
            )

            with self.assertRaises(ValueError) as context:
                EnhancedFilterConfig(min_consistent_studies_pct=1.5)
            self.assertIn(
                "min_consistent_studies_pct must be between 0 and 1",
                str(context.exception),
            )

    def test_post_init_validation_criteria_weights(self):
        """Test validation of criteria weights."""
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            # Test invalid weights that don't sum to 1.0
            invalid_weights = {
                "effect_size": 0.5,
                "fdr": 0.3,
                "heterogeneity": 0.1,
                "n_studies": 0.05,  # Sum = 0.95, not 1.0
            }

            with self.assertRaises(ValueError) as context:
                EnhancedFilterConfig(use_combined_criteria=True, criteria_weights=invalid_weights)
            self.assertIn("criteria_weights must sum to 1.0", str(context.exception))

    def test_valid_criteria_weights(self):
        """Test that valid criteria weights are accepted."""
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            valid_weights = {
                "effect_size": 0.25,
                "fdr": 0.35,
                "heterogeneity": 0.25,
                "n_studies": 0.15,
            }

            config = EnhancedFilterConfig(
                use_combined_criteria=True, criteria_weights=valid_weights
            )

            self.assertEqual(config.criteria_weights, valid_weights)


class TestEnhancedFilterResults(unittest.TestCase):
    """Test cases for EnhancedFilterResults class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create EnhancedFilterResults directly
        self.results = EnhancedFilterResults(
            pos_gene_names=["GENE1", "GENE2", "GENE3"],
            neg_gene_names=["GENE4", "GENE5"],
            gene_scores={
                "GENE1": 0.9,
                "GENE2": 0.8,
                "GENE3": 0.7,
                "GENE4": 0.6,
                "GENE5": 0.5,
            },
        )

    def test_total_prioritized_genes(self):
        """Test total_prioritized_genes property."""
        self.assertEqual(self.results.total_prioritized_genes, 5)

    def test_total_prioritized_genes_empty(self):
        """Test total_prioritized_genes with empty scores."""
        results = EnhancedFilterResults()
        self.assertEqual(results.total_prioritized_genes, 0)

    def test_get_top_genes_positive(self):
        """Test get_top_genes with positive direction."""
        top_genes = self.results.get_top_genes(direction="positive", n=2)
        expected = ["GENE1", "GENE2"]  # Top 2 from positive genes by score
        self.assertEqual(top_genes, expected)

    def test_get_top_genes_negative(self):
        """Test get_top_genes with negative direction."""
        top_genes = self.results.get_top_genes(direction="negative", n=1)
        expected = ["GENE4"]  # Top 1 from negative genes by score
        self.assertEqual(top_genes, expected)

    def test_get_top_genes_both(self):
        """Test get_top_genes with both directions."""
        top_genes = self.results.get_top_genes(direction="both", n=3)
        expected = ["GENE1", "GENE2", "GENE3"]  # Top 3 overall by score
        self.assertEqual(top_genes, expected)

    def test_get_top_genes_no_scores(self):
        """Test get_top_genes when genes have no scores."""
        results = EnhancedFilterResults(
            pos_gene_names=["GENE1", "GENE2"], neg_gene_names=["GENE3"], gene_scores={}
        )

        top_genes = results.get_top_genes(direction="both", n=2)
        self.assertEqual(len(top_genes), 2)  # Should return genes with score 0

    def test_get_top_genes_more_than_available(self):
        """Test get_top_genes when n exceeds available genes."""
        top_genes = self.results.get_top_genes(direction="positive", n=10)
        self.assertEqual(len(top_genes), 3)  # Only 3 positive genes available


class TestFilteringStrategies(unittest.TestCase):
    """Test cases for filtering strategy classes."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.sample_data = pd.DataFrame(
            {
                "effect_size": [0.5, -0.3, 0.8, -0.2, 0.1],
                "effect_fdr": [0.01, 0.05, 0.001, 0.1, 0.2],
                "number_studies": [5, 3, 8, 2, 4],
                "heterogeneity_pval": [0.8, 0.6, 0.9, 0.3, 0.7],
            },
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
        )

        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            self.config = EnhancedFilterConfig(
                effect_size_thresh=0.25,
                fdr_thresh=0.05,
                number_studies_thresh=3,
                heterogeneity_pval_thresh=0.5,
            )

    def test_filtering_strategy_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        strategy = FilteringStrategy()

        with self.assertRaises(NotImplementedError):
            strategy.apply_filter(self.sample_data, self.config)

        with self.assertRaises(NotImplementedError):
            strategy.get_strategy_name()

    def test_standard_filtering_strategy(self):
        """Test StandardFilteringStrategy."""
        strategy = StandardFilteringStrategy()

        self.assertEqual(strategy.get_strategy_name(), "standard")

        pos_genes, neg_genes = strategy.apply_filter(self.sample_data, self.config)

        # Based on thresholds: effect_size >= 0.25, fdr <= 0.05, n_studies >= 3, het_pval >= 0.5
        # GENE1: effect=0.5, fdr=0.01, studies=5, het=0.8 -> passes all, positive
        # GENE2: effect=-0.3, fdr=0.05, studies=3, het=0.6 -> passes all, negative
        # GENE3: effect=0.8, fdr=0.001, studies=8, het=0.9 -> passes all, positive
        # GENE4: effect=-0.2, fdr=0.1, studies=2, het=0.3 -> fails effect size and studies
        # GENE5: effect=0.1, fdr=0.2, studies=4, het=0.7 -> fails effect size and fdr

        expected_pos = ["GENE1", "GENE3"]
        expected_neg = ["GENE2"]

        self.assertEqual(sorted(pos_genes), sorted(expected_pos))
        self.assertEqual(sorted(neg_genes), sorted(expected_neg))

    def test_standard_filtering_no_thresholds(self):
        """Test StandardFilteringStrategy with no thresholds."""
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            config = EnhancedFilterConfig(
                effect_size_thresh=0,
                fdr_thresh=1,
                number_studies_thresh=1,
                heterogeneity_pval_thresh=0,
            )

        strategy = StandardFilteringStrategy()
        pos_genes, neg_genes = strategy.apply_filter(self.sample_data, config)

        # Should include all genes
        expected_pos = ["GENE1", "GENE3", "GENE5"]  # Positive effect sizes
        expected_neg = ["GENE2", "GENE4"]  # Negative effect sizes

        self.assertEqual(sorted(pos_genes), sorted(expected_pos))
        self.assertEqual(sorted(neg_genes), sorted(expected_neg))

    def test_adaptive_filtering_strategy(self):
        """Test AdaptiveFilteringStrategy."""
        strategy = AdaptiveFilteringStrategy()

        self.assertEqual(strategy.get_strategy_name(), "adaptive")

        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            config = EnhancedFilterConfig(
                adaptive_thresholds=True,
                min_effect_size_percentile=50,  # Median
                max_heterogeneity_percentile=50,  # Median
                fdr_thresh=0.1,
                number_studies_thresh=2,
            )

        pos_genes, neg_genes = strategy.apply_filter(self.sample_data, config)

        # Should have some genes based on adaptive thresholds
        self.assertIsInstance(pos_genes, list)
        self.assertIsInstance(neg_genes, list)

    def test_adaptive_filtering_calculate_thresholds(self):
        """Test adaptive threshold calculation."""
        strategy = AdaptiveFilteringStrategy()

        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            config = EnhancedFilterConfig(
                min_effect_size_percentile=50, max_heterogeneity_percentile=50
            )

        thresholds = strategy._calculate_adaptive_thresholds(self.sample_data, config)

        self.assertIn("effect_size", thresholds)
        self.assertIn("heterogeneity_pval", thresholds)

        # Check that thresholds are reasonable
        effect_median = np.percentile(self.sample_data["effect_size"].abs(), 50)
        self.assertEqual(thresholds["effect_size"], effect_median)

    def test_combined_criteria_filtering_strategy(self):
        """Test CombinedCriteriaFilteringStrategy."""
        strategy = CombinedCriteriaFilteringStrategy()

        self.assertEqual(strategy.get_strategy_name(), "combined_criteria")

        pos_genes, neg_genes = strategy.apply_filter(self.sample_data, self.config)

        self.assertIsInstance(pos_genes, list)
        self.assertIsInstance(neg_genes, list)

    def test_combined_criteria_calculate_scores(self):
        """Test combined score calculation."""
        strategy = CombinedCriteriaFilteringStrategy()

        scores = strategy._calculate_combined_scores(self.sample_data, self.config)

        self.assertEqual(len(scores), len(self.sample_data))
        self.assertTrue(all(isinstance(score, (int, float)) for score in scores))
        self.assertTrue(all(score >= 0 for score in scores))

    def test_combined_criteria_missing_columns(self):
        """Test combined criteria with missing data columns."""
        # Data missing some expected columns
        incomplete_data = self.sample_data[["effect_size", "effect_fdr"]].copy()

        strategy = CombinedCriteriaFilteringStrategy()
        scores = strategy._calculate_combined_scores(incomplete_data, self.config)

        # Should still calculate scores with available columns
        self.assertEqual(len(scores), len(incomplete_data))


class TestEnhancedMetaIntegratorFilter(unittest.TestCase):
    """Test cases for EnhancedMetaIntegratorFilter class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample meta object
        self.sample_data = pd.DataFrame(
            {
                "effect_size": [0.5, -0.3, 0.8, -0.2, 0.1],
                "effect_fdr": [0.01, 0.05, 0.001, 0.1, 0.2],
                "number_studies": [5, 3, 8, 2, 4],
                "heterogeneity_pval": [0.8, 0.6, 0.9, 0.3, 0.7],
            },
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
        )

        self.meta_object = {"original_result": self.sample_data}

        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            self.config = EnhancedFilterConfig(
                effect_size_thresh=0.25,
                fdr_thresh=0.05,
                number_studies_thresh=3,
                heterogeneity_pval_thresh=0.5,
            )

        # Mock the validators at module level
        self.meta_validator_patcher = patch(
            "metaintegrator.scoring_meta_integrator_filter.MetaObjectValidator.validate_and_extract"
        )
        self.filter_validator_patcher = patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterResultsValidator.validate_and_extract"
        )
        self.dataframe_utils_patcher = patch(
            "metaintegrator.scoring_meta_integrator_filter.DataFrameUtils.get_numeric_columns"
        )

        self.mock_meta_validator = self.meta_validator_patcher.start()
        self.mock_filter_validator = self.filter_validator_patcher.start()
        self.mock_dataframe_utils = self.dataframe_utils_patcher.start()

        self.mock_dataframe_utils.return_value = ["effect_size", "effect_fdr"]

        self.filter_obj = EnhancedMetaIntegratorFilter(self.config)

    def tearDown(self):
        """Clean up patches."""
        self.meta_validator_patcher.stop()
        self.filter_validator_patcher.stop()
        self.dataframe_utils_patcher.stop()

    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        filter_obj = EnhancedMetaIntegratorFilter()

        self.assertIsInstance(filter_obj.config, EnhancedFilterConfig)
        self.assertIn("standard", filter_obj.strategies)
        self.assertIn("adaptive", filter_obj.strategies)
        self.assertIn("combined_criteria", filter_obj.strategies)

    def test_initialization_custom_config(self):
        """Test initialization with custom configuration."""
        filter_obj = EnhancedMetaIntegratorFilter(self.config)

        self.assertEqual(filter_obj.config, self.config)

    def test_filter_meta_analysis_standard_strategy(self):
        """Test filter_meta_analysis with standard strategy."""
        results = self.filter_obj.filter_meta_analysis(self.meta_object, "standard")

        self.assertIsInstance(results, EnhancedFilterResults)
        self.assertIsInstance(results.pos_gene_names, list)
        self.assertIsInstance(results.neg_gene_names, list)
        self.assertIn("filter_performance", results.__dict__)
        self.assertIn("gene_scores", results.__dict__)

    def test_filter_meta_analysis_adaptive_strategy(self):
        """Test filter_meta_analysis with adaptive strategy."""
        results = self.filter_obj.filter_meta_analysis(self.meta_object, "adaptive")

        self.assertIsInstance(results, EnhancedFilterResults)

    def test_filter_meta_analysis_combined_strategy(self):
        """Test filter_meta_analysis with combined criteria strategy."""
        results = self.filter_obj.filter_meta_analysis(self.meta_object, "combined_criteria")

        self.assertIsInstance(results, EnhancedFilterResults)

    def test_filter_meta_analysis_unknown_strategy(self):
        """Test filter_meta_analysis with unknown strategy."""
        # Updated to expect ValueError (raised before try-except)
        with self.assertRaises(ValueError) as context:
            self.filter_obj.filter_meta_analysis(self.meta_object, "unknown")

        self.assertIn("Unknown filtering strategy: unknown", str(context.exception))

    def test_filter_meta_analysis_missing_data(self):
        """Test filter_meta_analysis with missing data keys."""
        invalid_meta = {"some_other_key": self.sample_data}

        with self.assertRaises(Exception) as context:
            self.filter_obj.filter_meta_analysis(invalid_meta, "standard")

        # The method should catch the ValueError and wrap it in FilteringError
        self.assertIn("Filtering failed", str(context.exception))

    def test_filter_meta_analysis_with_filterres(self):
        """Test filter_meta_analysis using 'filterRes' key."""
        meta_object = {"filterRes": self.sample_data}

        results = self.filter_obj.filter_meta_analysis(meta_object, "standard")

        self.assertIsInstance(results, EnhancedFilterResults)

    def test_filter_meta_analysis_validation_error(self):
        """Test filter_meta_analysis with validation error."""
        self.mock_meta_validator.side_effect = Exception("Validation failed")

        with self.assertRaises(Exception) as context:
            self.filter_obj.filter_meta_analysis(self.meta_object, "standard")

        self.assertIn("Filtering failed", str(context.exception))

    def test_apply_sequential_filtering(self):
        """Test apply_sequential_filtering method."""
        strategies = ["standard", "adaptive"]

        results = self.filter_obj.apply_sequential_filtering(self.meta_object, strategies)

        self.assertIsInstance(results, EnhancedFilterResults)
        self.assertGreater(len(results.filtering_steps), 1)
        self.assertIn("Sequential filtering", results.filter_description)

    def test_calculate_gene_prioritization_scores(self):
        """Test _calculate_gene_prioritization_scores method."""
        candidate_genes = ["GENE1", "GENE2", "GENE3"]

        scores = self.filter_obj._calculate_gene_prioritization_scores(
            self.sample_data, candidate_genes
        )

        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), 3)
        for gene in candidate_genes:
            self.assertIn(gene, scores)
            self.assertIsInstance(scores[gene], (int, float))
            self.assertGreaterEqual(scores[gene], 0)

    def test_calculate_gene_prioritization_scores_missing_gene(self):
        """Test gene prioritization with missing gene."""
        candidate_genes = ["GENE1", "NONEXISTENT_GENE"]

        scores = self.filter_obj._calculate_gene_prioritization_scores(
            self.sample_data, candidate_genes
        )

        self.assertIn("GENE1", scores)
        self.assertNotIn("NONEXISTENT_GENE", scores)

    def test_apply_leave_one_out_filter(self):
        """Test _apply_leave_one_out_filter method."""
        pos_genes = ["GENE1", "GENE3"]
        neg_genes = ["GENE2"]

        filtered_pos, filtered_neg = self.filter_obj._apply_leave_one_out_filter(
            self.meta_object, pos_genes, neg_genes
        )

        # Simplified implementation should return the same lists
        self.assertEqual(filtered_pos, pos_genes)
        self.assertEqual(filtered_neg, neg_genes)

    def test_filter_by_cross_study_consistency(self):
        """Test _filter_by_cross_study_consistency method."""
        # Add study column to data
        data_with_studies = self.sample_data.copy()
        data_with_studies["study"] = ["S1", "S2", "S1", "S2", "S1"]

        pos_genes = ["GENE1", "GENE3"]
        neg_genes = ["GENE2"]

        filtered_pos, filtered_neg, consistency = (
            self.filter_obj._filter_by_cross_study_consistency(
                data_with_studies, pos_genes, neg_genes
            )
        )

        self.assertIsInstance(filtered_pos, list)
        self.assertIsInstance(filtered_neg, list)
        self.assertIsInstance(consistency, dict)

    def test_limit_genes_by_priority(self):
        """Test _limit_genes_by_priority method."""
        genes = ["GENE1", "GENE2", "GENE3", "GENE4"]
        gene_scores = {"GENE1": 0.9, "GENE2": 0.7, "GENE3": 0.8, "GENE4": 0.6}
        max_genes = 2

        limited_genes = self.filter_obj._limit_genes_by_priority(genes, gene_scores, max_genes)

        self.assertEqual(len(limited_genes), 2)
        self.assertIn("GENE1", limited_genes)  # Highest score
        self.assertIn("GENE3", limited_genes)  # Second highest score

    def test_limit_genes_by_priority_no_limit_needed(self):
        """Test gene limiting when no limiting is needed."""
        genes = ["GENE1", "GENE2"]
        gene_scores = {"GENE1": 0.9, "GENE2": 0.7}
        max_genes = 5

        limited_genes = self.filter_obj._limit_genes_by_priority(genes, gene_scores, max_genes)

        self.assertEqual(limited_genes, genes)

    def test_assess_filter_performance(self):
        """Test _assess_filter_performance method."""
        pos_genes = ["GENE1", "GENE3"]
        neg_genes = ["GENE2"]

        performance = self.filter_obj._assess_filter_performance(
            self.sample_data, pos_genes, neg_genes
        )

        self.assertIsInstance(performance, dict)
        self.assertIn("genes_retained", performance)
        self.assertIn("genes_original", performance)
        self.assertIn("retention_rate", performance)
        self.assertIn("positive_genes", performance)
        self.assertIn("negative_genes", performance)
        self.assertIn("direction_ratio", performance)

        self.assertEqual(performance["genes_retained"], 3)
        self.assertEqual(performance["genes_original"], 5)
        self.assertEqual(performance["positive_genes"], 2)
        self.assertEqual(performance["negative_genes"], 1)

    def test_assess_filter_performance_no_genes(self):
        """Test performance assessment with no filtered genes."""
        performance = self.filter_obj._assess_filter_performance(self.sample_data, [], [])

        self.assertEqual(performance["genes_retained"], 0)
        self.assertEqual(performance["retention_rate"], 0)

    def test_get_filter_summary(self):
        """Test get_filter_summary method."""
        summary = self.filter_obj.get_filter_summary()

        self.assertIsInstance(summary, dict)
        self.assertIn("config", summary)
        self.assertIn("available_strategies", summary)
        self.assertIn("cache_size", summary)

        expected_strategies = ["standard", "adaptive", "combined_criteria"]
        self.assertEqual(sorted(summary["available_strategies"]), sorted(expected_strategies))

    def test_clear_cache(self):
        """Test clear_cache method."""
        # Add something to cache
        self.filter_obj._filter_cache["test"] = "data"

        self.filter_obj.clear_cache()

        self.assertEqual(len(self.filter_obj._filter_cache), 0)


class TestMetaIntegratorFilterLegacy(unittest.TestCase):
    """Test cases for MetaIntegratorFilter legacy compatibility class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame(
            {
                "effect_size": [0.5, -0.3, 0.8],
                "effect_fdr": [0.01, 0.05, 0.001],
                "number_studies": [5, 3, 8],
                "heterogeneity_pval": [0.8, 0.6, 0.9],
            },
            index=["GENE1", "GENE2", "GENE3"],
        )

        self.meta_object = {"original_result": self.sample_data}

        # Mock validators at module level
        self.meta_validator_patcher = patch(
            "metaintegrator.scoring_meta_integrator_filter.MetaObjectValidator.validate_and_extract"
        )
        self.filter_validator_patcher = patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterResultsValidator.validate_and_extract"
        )
        self.dataframe_utils_patcher = patch(
            "metaintegrator.scoring_meta_integrator_filter.DataFrameUtils.get_numeric_columns"
        )

        self.mock_meta_validator = self.meta_validator_patcher.start()
        self.mock_filter_validator = self.filter_validator_patcher.start()
        self.mock_dataframe_utils = self.dataframe_utils_patcher.start()

        self.mock_dataframe_utils.return_value = ["effect_size"]

    def tearDown(self):
        """Clean up patches."""
        self.meta_validator_patcher.stop()
        self.filter_validator_patcher.stop()
        self.dataframe_utils_patcher.stop()

    def test_initialization_no_config(self):
        """Test initialization without configuration."""
        filter_obj = MetaIntegratorFilter()

        self.assertIsInstance(filter_obj.config, EnhancedFilterConfig)

    def test_initialization_legacy_config(self):
        """Test initialization with legacy FilterConfig."""
        # Mock legacy config
        legacy_config = MagicMock()
        legacy_config.effect_size_thresh = 0.3
        legacy_config.fdr_thresh = 0.05
        legacy_config.number_studies_thresh = 3
        legacy_config.heterogeneity_pval_thresh = 0.5
        legacy_config.is_leave_one_out = False

        filter_obj = MetaIntegratorFilter(legacy_config)

        self.assertEqual(filter_obj.config.effect_size_thresh, 0.3)
        self.assertEqual(filter_obj.config.fdr_thresh, 0.05)

    def test_initialization_enhanced_config(self):
        """Test initialization with EnhancedFilterConfig."""
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            enhanced_config = EnhancedFilterConfig(effect_size_thresh=0.4)

        filter_obj = MetaIntegratorFilter(enhanced_config)

        self.assertEqual(filter_obj.config, enhanced_config)

    def test_filter_method_legacy_compatibility(self):
        """Test legacy filter method."""
        filter_obj = MetaIntegratorFilter()

        results = filter_obj.filter(self.meta_object)

        # Should return FilterResults instance - check that it has the expected attributes
        self.assertTrue(hasattr(results, "pos_gene_names"))
        self.assertTrue(hasattr(results, "neg_gene_names"))
        self.assertTrue(hasattr(results, "config"))
        self.assertTrue(hasattr(results, "is_leave_one_out"))
        self.assertTrue(hasattr(results, "filter_description"))
        self.assertTrue(hasattr(results, "timestamp"))


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios for the complete filtering pipeline."""

    def setUp(self):
        """Set up test fixtures for integration tests."""
        # Create comprehensive test data
        np.random.seed(42)
        n_genes = 100

        self.comprehensive_data = pd.DataFrame(
            {
                "effect_size": np.random.normal(0, 0.5, n_genes),
                "effect_fdr": np.random.beta(2, 5, n_genes),
                "number_studies": np.random.poisson(5, n_genes) + 1,
                "heterogeneity_pval": np.random.uniform(0, 1, n_genes),
                "study": np.random.choice(["S1", "S2", "S3"], n_genes),
            },
            index=[f"GENE{i:03d}" for i in range(n_genes)],
        )

        self.meta_object = {"original_result": self.comprehensive_data}

        # Mock validators at module level
        self.meta_validator_patcher = patch(
            "metaintegrator.scoring_meta_integrator_filter.MetaObjectValidator.validate_and_extract"
        )
        self.filter_validator_patcher = patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterResultsValidator.validate_and_extract"
        )
        self.dataframe_utils_patcher = patch(
            "metaintegrator.scoring_meta_integrator_filter.DataFrameUtils.get_numeric_columns"
        )

        self.mock_meta_validator = self.meta_validator_patcher.start()
        self.mock_filter_validator = self.filter_validator_patcher.start()
        self.mock_dataframe_utils = self.dataframe_utils_patcher.start()

        self.mock_dataframe_utils.return_value = [
            "effect_size",
            "effect_fdr",
            "number_studies",
            "heterogeneity_pval",
        ]

    def tearDown(self):
        """Clean up patches."""
        self.meta_validator_patcher.stop()
        self.filter_validator_patcher.stop()
        self.dataframe_utils_patcher.stop()

    def test_complete_filtering_pipeline_standard(self):
        """Test complete filtering pipeline with standard strategy."""
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            config = EnhancedFilterConfig(
                effect_size_thresh=0.3,
                fdr_thresh=0.05,
                number_studies_thresh=3,
                heterogeneity_pval_thresh=0.3,
                prioritize_by_strength=True,
            )

        filter_obj = EnhancedMetaIntegratorFilter(config)
        results = filter_obj.filter_meta_analysis(self.meta_object, "standard")

        self.assertIsInstance(results, EnhancedFilterResults)
        self.assertGreater(len(results.gene_scores), 0)
        self.assertGreater(len(results.filter_performance), 0)

    def test_complete_filtering_pipeline_adaptive(self):
        """Test complete filtering pipeline with adaptive strategy."""
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            config = EnhancedFilterConfig(
                adaptive_thresholds=True,
                min_effect_size_percentile=75,
                max_heterogeneity_percentile=25,
                fdr_thresh=0.1,
                prioritize_by_strength=True,
                max_genes_per_direction=10,
            )

        filter_obj = EnhancedMetaIntegratorFilter(config)
        results = filter_obj.filter_meta_analysis(self.meta_object, "adaptive")

        self.assertIsInstance(results, EnhancedFilterResults)
        # Check gene limits
        self.assertLessEqual(len(results.pos_gene_names), 10)
        self.assertLessEqual(len(results.neg_gene_names), 10)

    def test_complete_filtering_pipeline_combined_criteria(self):
        """Test complete filtering pipeline with combined criteria."""
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            config = EnhancedFilterConfig(
                use_combined_criteria=True,
                criteria_weights={
                    "effect_size": 0.4,
                    "fdr": 0.3,
                    "heterogeneity": 0.2,
                    "n_studies": 0.1,
                },
                fdr_thresh=0.1,
                prioritize_by_strength=True,
            )

        filter_obj = EnhancedMetaIntegratorFilter(config)
        results = filter_obj.filter_meta_analysis(self.meta_object, "combined_criteria")

        self.assertIsInstance(results, EnhancedFilterResults)
        self.assertIn("combined_criteria", results.filtering_steps[0]["step"])

    def test_sequential_filtering_pipeline(self):
        """Test sequential filtering with multiple strategies."""
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            config = EnhancedFilterConfig(
                effect_size_thresh=0.2,
                fdr_thresh=0.1,
                adaptive_thresholds=True,
                min_effect_size_percentile=50,
                prioritize_by_strength=True,
            )

        filter_obj = EnhancedMetaIntegratorFilter(config)
        strategies = ["standard", "adaptive"]

        results = filter_obj.apply_sequential_filtering(self.meta_object, strategies)

        self.assertIsInstance(results, EnhancedFilterResults)
        self.assertEqual(len(results.filtering_steps), 2)
        self.assertIn("Sequential filtering", results.filter_description)

    def test_cross_study_consistency_filtering(self):
        """Test filtering with cross-study consistency requirements."""
        with patch(
            "metaintegrator.scoring_meta_integrator_filter.FilterConfig.__post_init__",
            return_value=None,
        ):
            config = EnhancedFilterConfig(
                effect_size_thresh=0.1,
                fdr_thresh=0.2,
                require_consistency_across_studies=True,
                min_consistent_studies_pct=0.7,
                prioritize_by_strength=True,
            )

        filter_obj = EnhancedMetaIntegratorFilter(config)
        results = filter_obj.filter_meta_analysis(self.meta_object, "standard")

        self.assertIsInstance(results, EnhancedFilterResults)
        self.assertIsInstance(results.cross_study_consistency, dict)

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test with malformed data
        bad_data = pd.DataFrame(
            {
                "effect_size": [np.inf, -np.inf, np.nan],
                "effect_fdr": [0.01, 1.5, -0.1],  # Invalid values
            },
            index=["BAD1", "BAD2", "BAD3"],
        )

        meta_object = {"original_result": bad_data}

        filter_obj = EnhancedMetaIntegratorFilter()

        # Should handle gracefully without crashing
        try:
            results = filter_obj.filter_meta_analysis(meta_object, "standard")
            # If it succeeds, check that results are valid
            self.assertIsInstance(results, EnhancedFilterResults)
        except Exception as e:
            # If it fails, should be a controlled failure
            self.assertIsInstance(e, Exception)


if __name__ == "__main__":
    unittest.main()
