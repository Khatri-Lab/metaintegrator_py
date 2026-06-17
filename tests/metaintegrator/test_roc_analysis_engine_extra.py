"""
Additional unit tests for roc_analysis_engine module.

This file provides comprehensive tests for components missing from the original
test suite, including KesterBuntinxEngine, UnifiedROCAnalyzer, and advanced
meta-analysis functionality.

Test Coverage Added:
- KesterBuntinxEngine: Meta-analytic ROC methodology
- UnifiedROCAnalyzer: Main public interface
- Advanced pooled analysis methods
- Plotting functionality (mocked)
- Bootstrap statistical methods
- Summary curve generation
- Error handling edge cases
- Integration workflows

Author: MetaIntegrator Team
"""

import pytest
import numpy as np
import pandas as pd
import warnings
import logging
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt

# Add the src directory to the path to import the module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.roc_analysis_engine import (
    ROCAnalysisEngine,
    SignatureScoringEngine,
    KesterBuntinxEngine,
    UnifiedROCAnalyzer,
    MetaAnalysisUtilities,
    analyze_roc_curve,
    pooled_roc_plot,
    summary_roc_plot,
    summary_roc_calc,
)


# ------------------------------------------------------------------
# Additional Test Fixtures
# ------------------------------------------------------------------
@pytest.fixture
def kb_engine():
    """Create KesterBuntinxEngine instance for testing."""
    return KesterBuntinxEngine()


@pytest.fixture
def unified_analyzer():
    """Create UnifiedROCAnalyzer instance for testing."""
    return UnifiedROCAnalyzer()


@pytest.fixture
def mock_meta_object():
    """Create mock meta object with multiple studies."""
    return {
        "originalData": {
            "study1": {
                "expr": pd.DataFrame(
                    {
                        "sample1": [2.0, 4.0, 1.0, 8.0, 3.0],
                        "sample2": [3.0, 2.0, 2.0, 6.0, 4.0],
                        "sample3": [1.5, 5.0, 1.5, 7.0, 2.5],
                        "sample4": [2.5, 3.0, 1.8, 9.0, 3.5],
                    },
                    index=["gene1", "gene2", "gene3", "gene4", "gene5"],
                ),
                "keys": ["gene1", "gene2", "gene3", "gene4", "gene5"],
                "class": [0, 0, 1, 1],
                "formattedName": "Study 1",
            },
            "study2": {
                "expr": pd.DataFrame(
                    {
                        "sample1": [1.8, 3.5, 1.2, 7.5, 2.8],
                        "sample2": [3.2, 2.1, 2.3, 5.8, 4.1],
                        "sample3": [1.3, 4.8, 1.7, 6.8, 2.3],
                        "sample4": [2.7, 2.9, 1.9, 8.8, 3.3],
                    },
                    index=["gene1", "gene2", "gene3", "gene4", "gene5"],
                ),
                "keys": ["gene1", "gene2", "gene3", "gene4", "gene5"],
                "class": [0, 0, 1, 1],
                "formattedName": "Study 2",
            },
        }
    }


@pytest.fixture
def mock_filter_object():
    """Create mock filter object."""
    return {"posGeneNames": ["gene1", "gene2"], "negGeneNames": ["gene3", "gene5"]}


@pytest.fixture
def mock_study_data():
    """Create mock study data for Kester & Buntinx analysis."""
    np.random.seed(42)
    return [
        {
            "name": "Study A",
            "y_true": np.array([0, 0, 0, 1, 1, 1]),
            "y_scores": np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.9]),
        },
        {
            "name": "Study B",
            "y_true": np.array([0, 0, 1, 1, 1]),
            "y_scores": np.array([0.2, 0.35, 0.55, 0.75, 0.85]),
        },
        {
            "name": "Study C",
            "y_true": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            "y_scores": np.array([0.15, 0.25, 0.45, 0.5, 0.65, 0.7, 0.85, 0.95]),
        },
    ]


# ------------------------------------------------------------------
# KesterBuntinxEngine Tests
# ------------------------------------------------------------------
class TestKesterBuntinxEngine:
    """Test suite for KesterBuntinxEngine class."""

    def test_initialization(self, kb_engine):
        """Test KesterBuntinxEngine initialization."""
        assert kb_engine is not None
        assert isinstance(kb_engine, KesterBuntinxEngine)
        assert hasattr(kb_engine, "roc_engine")
        assert isinstance(kb_engine.roc_engine, ROCAnalysisEngine)

    def test_calculate_meta_statistics_basic(self, kb_engine, mock_study_data):
        """Test basic meta-analysis statistics calculation."""
        result = kb_engine.calculate_meta_statistics(
            mock_study_data, boot_reps=100, min_points=3  # Reduced for faster testing
        )

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mock_study_data)

        expected_columns = [
            "study",
            "N",
            "tstar_alpha",
            "SE_alpha",
            "tstar_beta",
            "SE_beta",
            "AUC",
            "AUC_CI_lower",
            "AUC_CI_upper",
        ]
        assert all(col in result.columns for col in expected_columns)

        # Check data validity
        assert all(result["N"] > 0)
        assert all((result["AUC"] >= 0) & (result["AUC"] <= 1))
        assert all(result["AUC_CI_lower"] <= result["AUC_CI_upper"])

    def test_calculate_meta_statistics_perfect_auc(self, kb_engine):
        """Test meta-analysis with perfect AUC case."""
        perfect_study = [
            {
                "name": "Perfect Study",
                "y_true": np.array([0, 0, 0, 1, 1, 1]),
                "y_scores": np.array(
                    [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
                ),  # Perfect separation
            }
        ]

        result = kb_engine.calculate_meta_statistics(perfect_study, boot_reps=10)

        assert len(result) == 1
        assert np.isclose(result.iloc[0]["AUC"], 1.0)
        assert not np.isnan(result.iloc[0]["tstar_alpha"])
        assert result.iloc[0]["tstar_beta"] == 0.0

    def test_calculate_meta_statistics_zero_auc(self, kb_engine):
        """Test meta-analysis with zero AUC case."""
        zero_study = [
            {
                "name": "Zero AUC Study",
                "y_true": np.array([0, 0, 0, 1, 1, 1]),
                "y_scores": np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]),  # Inverted scores
            }
        ]

        result = kb_engine.calculate_meta_statistics(zero_study, boot_reps=10)

        assert len(result) == 1
        assert np.isclose(result.iloc[0]["AUC"], 0.0, atol=0.1)
        assert not np.isnan(result.iloc[0]["tstar_alpha"])
        assert result.iloc[0]["tstar_beta"] == 0.0

    def test_fit_kb_model_basic(self, kb_engine):
        """Test basic Kester & Buntinx model fitting."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.9])

        # Get ROC curve for test
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_scores)

        try:
            alpha, beta, se_alpha, se_beta = kb_engine._fit_kb_model(
                y_true, y_scores, fpr, tpr, boot_reps=50, min_points=3
            )

            # Check that parameters are finite
            assert np.isfinite(alpha)
            assert np.isfinite(beta)
            assert np.isfinite(se_alpha) and se_alpha >= 0
            assert np.isfinite(se_beta) and se_beta >= 0

        except ValueError:
            # If insufficient points, this is expected
            pass

    def test_calculate_kb_weights(self, kb_engine):
        """Test calculation of Kester & Buntinx weights."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.9])

        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_scores)

        weights = kb_engine._calculate_kb_weights(y_true, y_scores, fpr, tpr)

        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(fpr)
        assert all(weights >= 0)  # Weights should be non-negative

    def test_bootstrap_kb_regression(self, kb_engine):
        """Test bootstrap regression for K&B parameters."""
        # Create simple regression data
        S = np.array([1.0, 2.0, 3.0, 4.0])
        D = np.array([0.5, 1.0, 1.5, 2.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])

        alpha, beta, se_alpha, se_beta = kb_engine._bootstrap_kb_regression(
            S, D, weights, boot_reps=50
        )

        # Check that results are reasonable
        assert np.isfinite(alpha)
        assert np.isfinite(beta)
        assert se_alpha >= 0
        assert se_beta >= 0

    def test_weighted_least_squares(self, kb_engine):
        """Test weighted least squares regression."""
        X = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 3.0, 5.0, 7.0])  # y = 2x - 1
        weights = np.array([1.0, 1.0, 1.0, 1.0])

        alpha, beta = kb_engine._weighted_least_squares(X, y, weights)

        # Should recover approximate linear relationship
        assert np.isclose(beta, 2.0, atol=0.1)
        assert np.isclose(alpha, -1.0, atol=0.1)

    def test_generate_summary_roc(self, kb_engine):
        """Test generation of summary ROC curve."""
        alpha = 2.0
        beta = 0.5

        summary_roc = kb_engine.generate_summary_roc(alpha, beta, n_points=11)

        assert isinstance(summary_roc, pd.DataFrame)
        assert "fpr" in summary_roc.columns
        assert "tpr" in summary_roc.columns
        assert len(summary_roc) == 11

        # Check monotonicity and bounds
        assert summary_roc["fpr"].is_monotonic_increasing
        assert all((summary_roc["fpr"] >= 0) & (summary_roc["fpr"] <= 1))
        assert all((summary_roc["tpr"] >= 0) & (summary_roc["tpr"] <= 1))

    def test_generate_summary_roc_extreme_beta(self, kb_engine):
        """Test summary ROC generation with extreme beta values."""
        # Test beta clipping
        alpha = 2.0
        beta_extreme = 0.98  # Should be clipped to 0.95

        summary_roc = kb_engine.generate_summary_roc(alpha, beta_extreme)

        assert isinstance(summary_roc, pd.DataFrame)
        assert len(summary_roc) == 100  # Default n_points
        assert all((summary_roc["fpr"] >= 0) & (summary_roc["fpr"] <= 1))
        assert all((summary_roc["tpr"] >= 0) & (summary_roc["tpr"] <= 1))

    def test_meta_statistics_error_handling(self, kb_engine):
        """Test error handling in meta-statistics calculation."""
        # Invalid study data
        invalid_study = [
            {
                "name": "Invalid Study",
                "y_true": np.array([0, 0, 0]),  # No positive class
                "y_scores": np.array([0.1, 0.2, 0.3]),
            }
        ]

        result = kb_engine.calculate_meta_statistics(invalid_study, boot_reps=10)

        # Should handle gracefully
        assert len(result) == 1

        assert np.isnan(result.iloc[0]["tstar_alpha"])


# ------------------------------------------------------------------
# UnifiedROCAnalyzer Tests
# ------------------------------------------------------------------
class TestUnifiedROCAnalyzer:
    """Test suite for UnifiedROCAnalyzer class."""

    def test_initialization(self, unified_analyzer):
        """Test UnifiedROCAnalyzer initialization."""
        assert unified_analyzer is not None
        assert isinstance(unified_analyzer, UnifiedROCAnalyzer)
        assert hasattr(unified_analyzer, "roc_engine")
        assert hasattr(unified_analyzer, "scoring_engine")
        assert hasattr(unified_analyzer, "kb_engine")

    def test_analyze_single_roc(self, unified_analyzer):
        """Test single ROC analysis."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])

        result = unified_analyzer.analyze_single_roc(y_true, y_scores, title="Test ROC")

        assert isinstance(result, dict)
        assert "auc" in result
        assert 0 <= result["auc"] <= 1

    def test_pooled_roc_analysis_simple(
        self, unified_analyzer, mock_meta_object, mock_filter_object
    ):
        """Test simple pooled ROC analysis."""
        result = unified_analyzer.pooled_roc_analysis(
            mock_meta_object,
            mock_filter_object,
            method="simple",
            points=50,
            weighting=True,
        )

        assert isinstance(result, dict)
        assert result["method"] == "simple"
        assert "individual_studies" in result
        assert "pooled_curve" in result
        assert "confidence_bands" in result
        assert "parameters" in result

        # Check pooled curve structure
        pooled = result["pooled_curve"]
        assert "fpr" in pooled
        assert "tpr" in pooled
        assert "auc" in pooled
        assert len(pooled["fpr"]) == 50  # Should match points parameter

    def test_pooled_roc_analysis_kester_buntinx(
        self, unified_analyzer, mock_meta_object, mock_filter_object
    ):
        """Test Kester & Buntinx pooled ROC analysis."""
        result = unified_analyzer.pooled_roc_analysis(
            mock_meta_object,
            mock_filter_object,
            method="kester_buntinx",
            boot_reps=50,  # Reduced for testing
        )

        assert isinstance(result, dict)
        assert result["method"] == "kester_buntinx"
        assert "meta_statistics" in result
        assert "summary_curve" in result
        assert "summary_statistics" in result

        # Check meta statistics
        meta_stats = result["meta_statistics"]
        assert isinstance(meta_stats, pd.DataFrame)
        assert len(meta_stats) > 0

    def test_pooled_roc_analysis_invalid_method(
        self, unified_analyzer, mock_meta_object, mock_filter_object
    ):
        """Test invalid method error handling."""
        with pytest.raises(ValueError, match="Unknown pooled analysis method"):
            unified_analyzer.pooled_roc_analysis(
                mock_meta_object, mock_filter_object, method="invalid_method"
            )

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_roc_curves_simple(
        self,
        mock_savefig,
        mock_tight,
        mock_subplots,
        unified_analyzer,
        mock_meta_object,
        mock_filter_object,
    ):
        """Test ROC curve plotting for simple method."""
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Get analysis results
        analysis_results = unified_analyzer.pooled_roc_analysis(
            mock_meta_object, mock_filter_object, method="simple", points=10
        )

        # Test plotting
        fig = unified_analyzer.plot_roc_curves(
            analysis_results, figsize=(8, 6), save_path="/tmp/test.png"
        )

        # Verify matplotlib calls
        mock_subplots.assert_called_once_with(figsize=(8, 6))
        mock_tight.assert_called_once()
        mock_savefig.assert_called_once_with(
            "/tmp/test.png", dpi=300, bbox_inches="tight"
        )

    @patch("matplotlib.pyplot.subplots")
    def test_plot_roc_curves_kester_buntinx(
        self, mock_subplots, unified_analyzer, mock_meta_object, mock_filter_object
    ):
        """Test ROC curve plotting for Kester & Buntinx method."""
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Get analysis results
        analysis_results = unified_analyzer.pooled_roc_analysis(
            mock_meta_object, mock_filter_object, method="kester_buntinx", boot_reps=10
        )

        # Test plotting
        fig = unified_analyzer.plot_roc_curves(analysis_results)

        # Verify plot was created
        mock_subplots.assert_called_once()
        assert fig == mock_fig

    def test_plot_roc_curves_invalid_method(self, unified_analyzer):
        """Test plotting with invalid method."""
        invalid_results = {"method": "invalid_method"}

        with pytest.raises(ValueError, match="Unknown plotting method"):
            unified_analyzer.plot_roc_curves(invalid_results)

    @patch.object(UnifiedROCAnalyzer, "pooled_roc_analysis")
    @patch.object(UnifiedROCAnalyzer, "plot_roc_curves")
    def test_pooled_roc_plot_convenience(
        self,
        mock_plot,
        mock_analysis,
        unified_analyzer,
        mock_meta_object,
        mock_filter_object,
    ):
        """Test pooled ROC plot convenience method."""
        # Setup mocks
        mock_analysis.return_value = {"method": "simple", "test": "result"}
        mock_plot.return_value = Mock()

        # Test convenience method
        fig = unified_analyzer.pooled_roc_plot(
            mock_meta_object, mock_filter_object, smoothed=False, figsize=(10, 8)
        )

        # Verify calls
        mock_analysis.assert_called_once_with(
            mock_meta_object, mock_filter_object, "simple", figsize=(10, 8)
        )
        mock_plot.assert_called_once()

    @patch.object(UnifiedROCAnalyzer, "pooled_roc_plot")
    def test_summary_roc_plot_convenience(
        self, mock_pooled_plot, unified_analyzer, mock_meta_object, mock_filter_object
    ):
        """Test summary ROC plot convenience method."""
        mock_pooled_plot.return_value = Mock()

        fig = unified_analyzer.summary_roc_plot(mock_meta_object, mock_filter_object)

        mock_pooled_plot.assert_called_once_with(
            mock_meta_object, mock_filter_object, smoothed=True
        )

    def test_summary_roc_calc_convenience(
        self, unified_analyzer, mock_meta_object, mock_filter_object
    ):
        """Test summary ROC calculation convenience method."""
        result = unified_analyzer.summary_roc_calc(
            mock_meta_object, mock_filter_object, boot_reps=10
        )

        assert isinstance(result, pd.DataFrame)
        assert "AUC" in result.columns
        assert "AUCupper" in result.columns
        assert "AUClower" in result.columns

    def test_simple_pooled_analysis_no_weighting(
        self, unified_analyzer, mock_meta_object, mock_filter_object
    ):
        """Test simple pooled analysis without weighting."""
        result = unified_analyzer._simple_pooled_analysis(
            mock_meta_object, mock_filter_object, points=20, weighting=False
        )

        assert result["parameters"]["weighting"] is False
        assert "pooled_curve" in result

    def test_simple_pooled_analysis_constant_scores(self, unified_analyzer):
        """Test pooled analysis with constant signature scores."""
        # Create meta object with constant expression
        constant_meta = {
            "originalData": {
                "study1": {
                    "expr": pd.DataFrame(
                        {
                            "sample1": [5.0, 5.0, 5.0],
                            "sample2": [5.0, 5.0, 5.0],
                        },
                        index=["gene1", "gene2", "gene3"],
                    ),
                    "keys": ["gene1", "gene2", "gene3"],
                    "class": [0, 1],
                    "formattedName": "Constant Study",
                }
            }
        }

        filter_obj = {"posGeneNames": ["gene1"], "negGeneNames": ["gene2"]}

        # Should handle constant scores gracefully
        result = unified_analyzer._simple_pooled_analysis(constant_meta, filter_obj)

        assert "individual_studies" in result
        assert len(result["individual_studies"]) == 1

    def test_error_handling_empty_meta_object(
        self, unified_analyzer, mock_filter_object
    ):
        """Test error handling with empty meta object."""
        empty_meta = {"originalData": {}}

        with pytest.raises(ValueError, match="No valid studies"):
            unified_analyzer.pooled_roc_analysis(empty_meta, mock_filter_object)

    def test_error_handling_invalid_class_data(
        self, unified_analyzer, mock_filter_object
    ):
        """Test error handling with invalid class data."""
        invalid_meta = {
            "originalData": {
                "study1": {
                    "expr": pd.DataFrame(
                        {
                            "sample1": [2.0, 4.0],
                            "sample2": [3.0, 2.0],
                        },
                        index=["gene1", "gene2"],
                    ),
                    "keys": ["gene1", "gene2"],
                    "class": [0, 0],  # No positive class
                    "formattedName": "Invalid Study",
                }
            }
        }

        with pytest.raises(ValueError, match="No valid studies"):
            unified_analyzer.pooled_roc_analysis(invalid_meta, mock_filter_object)


# ------------------------------------------------------------------
# Advanced MetaAnalysisUtilities Tests
# ------------------------------------------------------------------
class TestAdvancedMetaAnalysisUtilities:
    """Additional tests for MetaAnalysisUtilities edge cases."""

    def test_interpolate_roc_curves_empty_input(self):
        """Test interpolation with empty input."""
        common_fpr, interpolated_tprs = MetaAnalysisUtilities.interpolate_roc_curves([])

        assert len(common_fpr) == 1000  # Default n_points
        assert len(interpolated_tprs) == 0

    def test_interpolate_roc_curves_single_point(self):
        """Test interpolation with single-point curves."""
        fpr_tpr_pairs = [(np.array([0.5]), np.array([0.7]))]

        common_fpr, interpolated_tprs = MetaAnalysisUtilities.interpolate_roc_curves(
            fpr_tpr_pairs, n_points=5
        )

        assert len(interpolated_tprs) == 1
        assert len(interpolated_tprs[0]) == 5

    def test_calculate_confidence_bands_single_curve(self):
        """Test confidence bands with single curve."""
        tpr_curves = [np.array([0.0, 0.5, 1.0])]
        sample_sizes = [100]

        lower, upper = MetaAnalysisUtilities.calculate_confidence_bands(
            tpr_curves, sample_sizes, method="std"
        )

        # With single curve, bands should equal the curve
        np.testing.assert_array_equal(lower, tpr_curves[0])
        np.testing.assert_array_equal(upper, tpr_curves[0])

    def test_calculate_pooled_auc_edge_weights(self):
        """Test pooled AUC with edge case weights."""
        aucs = [0.8, 0.9]
        sample_sizes = [0, 100]  # Zero weight for first study

        pooled_auc = MetaAnalysisUtilities.calculate_pooled_auc(
            aucs, sample_sizes, method="weighted"
        )

        # Should be dominated by second study
        assert pooled_auc == 0.9

    def test_calculate_pooled_tpr_mismatched_lengths(self):
        """Test pooled TPR with mismatched curve lengths."""
        tpr1 = np.array([0.0, 0.5, 1.0])
        tpr2 = np.array([0.0, 0.3, 0.7, 1.0])  # Different length

        with pytest.raises((ValueError, IndexError)):
            MetaAnalysisUtilities.calculate_pooled_tpr([tpr1, tpr2], [100, 100])


# ------------------------------------------------------------------
# Integration and Workflow Tests
# ------------------------------------------------------------------
class TestAdvancedIntegration:
    """Advanced integration tests for complex workflows."""

    def test_end_to_end_simple_pooled_workflow(
        self, unified_analyzer, mock_meta_object, mock_filter_object
    ):
        """Test complete end-to-end simple pooled workflow."""
        # Perform analysis
        results = unified_analyzer.pooled_roc_analysis(
            mock_meta_object,
            mock_filter_object,
            method="simple",
            points=10,
            weighting=True,
        )

        # Validate complete workflow
        assert "individual_studies" in results
        assert "pooled_curve" in results
        assert "confidence_bands" in results

        # Check individual studies
        studies = results["individual_studies"]
        assert len(studies) == 2  # Two studies in mock data

        for study in studies:
            assert "name" in study
            assert "auc" in study
            assert "fpr" in study
            assert "tpr" in study

        # Check pooled results
        pooled = results["pooled_curve"]
        assert len(pooled["fpr"]) == len(pooled["tpr"])
        assert 0 <= pooled["auc"] <= 1

    def test_end_to_end_kester_buntinx_workflow(
        self, unified_analyzer, mock_meta_object, mock_filter_object
    ):
        """Test complete end-to-end Kester & Buntinx workflow."""
        # Perform analysis
        results = unified_analyzer.pooled_roc_analysis(
            mock_meta_object, mock_filter_object, method="kester_buntinx", boot_reps=10
        )

        # Validate complete workflow
        assert "meta_statistics" in results
        assert "summary_curve" in results
        assert "summary_statistics" in results

        # Check meta statistics
        meta_stats = results["meta_statistics"]
        assert isinstance(meta_stats, pd.DataFrame)
        assert len(meta_stats) == 2

        # Check summary curve
        summary_curve = results["summary_curve"]
        assert "fpr" in summary_curve.columns
        assert "tpr" in summary_curve.columns

    @patch("matplotlib.pyplot.subplots")
    def test_complete_plotting_workflow(
        self, mock_subplots, unified_analyzer, mock_meta_object, mock_filter_object
    ):
        """Test complete plotting workflow."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Test simple method plotting
        fig1 = unified_analyzer.pooled_roc_plot(
            mock_meta_object, mock_filter_object, smoothed=False
        )
        assert fig1 == mock_fig

        # Test Kester & Buntinx plotting
        fig2 = unified_analyzer.pooled_roc_plot(
            mock_meta_object, mock_filter_object, smoothed=True, boot_reps=10
        )
        assert fig2 == mock_fig

    def test_statistical_consistency_workflow(self, unified_analyzer):
        """Test statistical consistency across workflow."""
        # Create known data with predictable AUC
        meta_obj = {
            "originalData": {
                "test_study": {
                    "expr": pd.DataFrame(
                        {
                            "sample1": [10.0, 1.0, 5.0],  # High pos, low neg genes
                            "sample2": [12.0, 1.5, 4.5],
                            "sample3": [2.0, 8.0, 5.0],  # Low pos, high neg genes
                            "sample4": [1.8, 9.0, 4.8],
                        },
                        index=["pos_gene", "neg_gene", "neutral"],
                    ),
                    "keys": ["pos_gene", "neg_gene", "neutral"],
                    "class": [1, 1, 0, 0],  # First two are positive class
                    "formattedName": "Test Study",
                }
            }
        }

        filter_obj = {"posGeneNames": ["pos_gene"], "negGeneNames": ["neg_gene"]}

        # Run analysis
        results = unified_analyzer.pooled_roc_analysis(
            meta_obj, filter_obj, method="simple"
        )

        # Should have good separation
        auc = results["pooled_curve"]["auc"]
        assert auc > 0.7  # Should be better than random


# ------------------------------------------------------------------
# Convenience Function Tests
# ------------------------------------------------------------------
class TestAdvancedConvenienceFunctions:
    """Tests for top-level convenience functions."""

    def test_analyze_roc_curve_convenience(self):
        """Test analyze_roc_curve convenience function."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])

        result = analyze_roc_curve(y_true, y_scores, title="Test")

        assert isinstance(result, dict)
        assert "auc" in result
        assert 0 <= result["auc"] <= 1

    @patch.object(UnifiedROCAnalyzer, "pooled_roc_plot")
    def test_pooled_roc_plot_convenience(
        self, mock_method, mock_meta_object, mock_filter_object
    ):
        """Test pooled_roc_plot convenience function."""
        mock_method.return_value = Mock()

        fig = pooled_roc_plot(mock_meta_object, mock_filter_object, smoothed=False)

        mock_method.assert_called_once_with(
            mock_meta_object, mock_filter_object, smoothed=False
        )

    @patch.object(UnifiedROCAnalyzer, "summary_roc_plot")
    def test_summary_roc_plot_convenience(
        self, mock_method, mock_meta_object, mock_filter_object
    ):
        """Test summary_roc_plot convenience function."""
        mock_method.return_value = Mock()

        fig = summary_roc_plot(mock_meta_object, mock_filter_object)

        mock_method.assert_called_once_with(mock_meta_object, mock_filter_object)

    @patch.object(UnifiedROCAnalyzer, "summary_roc_calc")
    def test_summary_roc_calc_convenience(
        self, mock_method, mock_meta_object, mock_filter_object
    ):
        """Test summary_roc_calc convenience function."""
        mock_method.return_value = pd.DataFrame(
            {"AUC": [0.8], "AUCupper": [0.9], "AUClower": [0.7]}
        )

        result = summary_roc_calc(mock_meta_object, mock_filter_object)

        assert isinstance(result, pd.DataFrame)
        mock_method.assert_called_once_with(mock_meta_object, mock_filter_object)


# ------------------------------------------------------------------
# Error Handling and Edge Cases
# ------------------------------------------------------------------
class TestAdvancedErrorHandling:
    """Advanced error handling and edge case tests."""

    def test_kb_engine_insufficient_data(self, kb_engine):
        """Test K&B engine with insufficient data."""
        minimal_study = [
            {
                "name": "Minimal Study",
                "y_true": np.array([0, 1]),  # Only 2 samples
                "y_scores": np.array([0.3, 0.7]),
            }
        ]

        # Should handle gracefully
        result = kb_engine.calculate_meta_statistics(minimal_study, boot_reps=10)
        assert len(result) == 1

    def test_unified_analyzer_malformed_data(self, unified_analyzer):
        """Test unified analyzer with malformed data."""
        malformed_meta = {
            "originalData": {
                "bad_study": {
                    "expr": "not_a_dataframe",  # Invalid type
                    "keys": [],
                    "class": [],
                    "formattedName": "Bad Study",
                }
            }
        }

        filter_obj = {"posGeneNames": ["gene1"], "negGeneNames": ["gene2"]}

        # Should fail gracefully
        with pytest.raises((ValueError, AttributeError)):
            unified_analyzer.pooled_roc_analysis(malformed_meta, filter_obj)

    def test_memory_efficiency_large_bootstrap(self, kb_engine):
        """Test memory efficiency with large bootstrap iterations."""
        # Create small study but large bootstrap
        study_data = [
            {
                "name": "Large Bootstrap Test",
                "y_true": np.array([0, 0, 1, 1]),
                "y_scores": np.array([0.1, 0.4, 0.6, 0.9]),
            }
        ]

        # Should complete without memory issues
        result = kb_engine.calculate_meta_statistics(
            study_data, boot_reps=100, min_points=2  # Reasonable size for testing
        )

        assert len(result) == 1
        assert not np.isnan(result.iloc[0]["AUC"])

    def test_numerical_stability_extreme_values(self, unified_analyzer):
        """Test numerical stability with extreme values."""
        extreme_meta = {
            "originalData": {
                "extreme_study": {
                    "expr": pd.DataFrame(
                        {
                            "sample1": [1e10, 1e-10, 1e5],
                            "sample2": [1e10, 1e-10, 1e5],
                            "sample3": [1e-10, 1e10, 1e5],
                            "sample4": [1e-10, 1e10, 1e5],
                        },
                        index=["gene1", "gene2", "gene3"],
                    ),
                    "keys": ["gene1", "gene2", "gene3"],
                    "class": [0, 0, 1, 1],
                    "formattedName": "Extreme Study",
                }
            }
        }

        filter_obj = {"posGeneNames": ["gene1"], "negGeneNames": ["gene2"]}

        # Should handle extreme values without overflow/underflow
        result = unified_analyzer.pooled_roc_analysis(extreme_meta, filter_obj)

        assert "pooled_curve" in result
        assert np.isfinite(result["pooled_curve"]["auc"])


# ------------------------------------------------------------------
# Performance and Scalability Tests
# ------------------------------------------------------------------
class TestAdvancedPerformance:
    """Advanced performance and scalability tests."""

    def test_large_study_count_scalability(self, unified_analyzer):
        """Test scalability with many studies."""
        # Create meta object with many small studies
        many_studies = {}
        for i in range(20):  # 20 studies
            many_studies[f"study_{i}"] = {
                "expr": pd.DataFrame(
                    {
                        f"sample_{j}": [
                            np.random.rand(),
                            np.random.rand(),
                            np.random.rand(),
                        ]
                        for j in range(4)
                    },
                    index=["gene1", "gene2", "gene3"],
                ),
                "keys": ["gene1", "gene2", "gene3"],
                "class": [0, 0, 1, 1],
                "formattedName": f"Study {i}",
            }

        meta_obj = {"originalData": many_studies}
        filter_obj = {"posGeneNames": ["gene1"], "negGeneNames": ["gene2"]}

        # Should complete efficiently
        result = unified_analyzer.pooled_roc_analysis(
            meta_obj, filter_obj, method="simple"
        )

        assert len(result["individual_studies"]) == 20
        assert "pooled_curve" in result

    def test_wide_expression_matrix_performance(self, unified_analyzer):
        """Test performance with wide expression matrices."""
        np.random.seed(42)

        # Create study with many samples
        n_samples = 1000
        wide_meta = {
            "originalData": {
                "wide_study": {
                    "expr": pd.DataFrame(
                        {
                            f"sample_{i}": [
                                np.random.rand(),
                                np.random.rand(),
                                np.random.rand(),
                            ]
                            for i in range(n_samples)
                        },
                        index=["gene1", "gene2", "gene3"],
                    ),
                    "keys": ["gene1", "gene2", "gene3"],
                    "class": [0] * (n_samples // 2) + [1] * (n_samples // 2),
                    "formattedName": "Wide Study",
                }
            }
        }

        filter_obj = {"posGeneNames": ["gene1"], "negGeneNames": ["gene2"]}

        # Should handle large sample size efficiently
        result = unified_analyzer.pooled_roc_analysis(
            wide_meta, filter_obj, method="simple"
        )

        assert "pooled_curve" in result
        assert len(result["individual_studies"]) == 1


# ------------------------------------------------------------------
# Run Tests
# ------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
