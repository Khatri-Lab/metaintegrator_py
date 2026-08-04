"""
Unit tests for roc_prc_analysis module.
Tests ROC/PRC curve analysis, gene search optimization, and related functionality.
"""

import os
import sys
import warnings
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.prc_analysis import (
    PRCAnalyzer,
    plot_roc_curve,
    plot_prc_curve,
    calculate_auc_ci,
    calculate_auprc_ci,
)

from metaintegrator.roc_analysis_engine import ROCAnalysisEngine


class TestROCAnalyzer:
    """Test suite for ROCAnalyzer class."""

    def setup_method(self):
        """Setup test data for each test method."""
        np.random.seed(42)
        self.analyzer = ROCAnalysisEngine()

        # Perfect separation case
        self.y_true_perfect = np.array([0, 0, 0, 1, 1, 1])
        self.y_scores_perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        # Random case
        self.y_true_random = np.array([0, 1, 0, 1, 0, 1])
        self.y_scores_random = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        # Single class case
        self.y_true_single = np.array([1, 1, 1, 1])
        self.y_scores_single = np.array([0.1, 0.3, 0.5, 0.7])

    def test_calculate_roc_perfect_separation(self):
        """Test ROC calculation with perfect separation."""
        result = self.analyzer.calculate_roc(self.y_true_perfect, self.y_scores_perfect)

        assert result["auc"] == 1.0
        assert len(result["fpr"]) > 0
        assert len(result["tpr"]) > 0
        assert len(result["thresholds"]) > 0
        assert isinstance(result["roc_curve"], pd.DataFrame)
        assert not np.isnan(result["auc_pvalue"])
        assert len(result["auc_ci"]) == 2

    def test_calculate_roc_random_classification(self):
        """Test ROC calculation with random classification."""
        result = self.analyzer.calculate_roc(self.y_true_random, self.y_scores_random)

        # AUC should be around 0.5 for random classification
        assert 0.2 <= result["auc"] <= 0.8
        assert not np.isnan(result["auc"])
        assert not np.isnan(result["auc_pvalue"])

    def test_calculate_roc_single_class(self):
        """Test ROC calculation with single class."""
        result = self.analyzer.calculate_roc(self.y_true_single, self.y_scores_single)

        assert np.isnan(result["auc"])
        assert len(result["fpr"]) == 0
        assert len(result["tpr"]) == 0
        assert np.isnan(result["auc_ci"][0])
        assert np.isnan(result["auc_ci"][1])

    def test_calculate_auc_ci_valid_input(self):
        """Test AUC confidence interval calculation."""
        auc_value = 0.75

        lower, upper = self.analyzer._calculate_auc_ci(self.y_true_perfect, auc_value)
        assert 0 <= lower <= auc_value <= upper <= 1
        assert lower < upper

    def test_calculate_auc_ci_edge_cases(self):
        """Test AUC CI with edge cases."""
        # No positive cases
        y_true_no_pos = np.array([0, 0, 0])
        y_scores_no_pos = np.array([0.1, 0.2, 0.3])

        lower, upper = self.analyzer._calculate_auc_ci(y_true_no_pos, 0.5)
        assert lower == upper == 0.5

    def test_calculate_auc_pvalue(self):
        """Test AUC p-value calculation."""

        p_value = self.analyzer._calculate_auc_pvalue(self.y_true_perfect, 1.0)

        assert 0 <= p_value <= 1
        assert not np.isnan(p_value)


class TestPRCAnalyzer:
    """Test suite for PRCAnalyzer class."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.analyzer = PRCAnalyzer()

        # Imbalanced dataset (more realistic for PRC)
        self.y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1])  # 25% positive
        self.y_scores = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.4, 0.8, 0.9])

    def test_calculate_prc_basic(self):
        """Test basic PRC calculation."""
        result = self.analyzer.calculate_prc(self.y_true, self.y_scores)

        assert "precision" in result
        assert "recall" in result
        assert "auprc" in result
        assert "baseline" in result
        assert "auprc_ci" in result

        assert 0 <= result["auprc"] <= 1
        assert result["baseline"] == 0.25  # 2/8 positive samples
        assert len(result["auprc_ci"]) == 2

    def test_calculate_auprc_ci_binomial(self):
        """Test AUPRC confidence interval with binomial method."""
        auprc_value = 0.6
        lower, upper = self.analyzer._calculate_auprc_ci(
            self.y_true, self.y_scores, auprc_value, method="binomial"
        )

        assert 0 <= lower <= auprc_value <= upper <= 1

    def test_calculate_auprc_ci_no_positives(self):
        """Test AUPRC CI when no positive samples."""
        y_true_no_pos = np.array([0, 0, 0])
        y_scores_no_pos = np.array([0.1, 0.2, 0.3])

        lower, upper = self.analyzer._calculate_auprc_ci(y_true_no_pos, y_scores_no_pos, 0.5)
        assert lower == upper == 0.5


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.y_true = np.array([0, 0, 0, 1, 1, 1])
        self.y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    def test_calculate_auc_ci_convenience(self):
        """Test AUC confidence interval convenience function."""
        lower, upper = calculate_auc_ci(self.y_true, self.y_scores)

        assert 0 <= lower <= upper <= 1

    def test_calculate_auprc_ci_convenience(self):
        """Test AUPRC confidence interval convenience function."""
        lower, upper = calculate_auprc_ci(self.y_true, self.y_scores)

        assert 0 <= lower <= upper <= 1

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_roc_curve_convenience(self, mock_show, mock_savefig):
        """Test ROC curve plotting convenience function."""
        fig = plot_roc_curve(self.y_true, self.y_scores, save_path="test.png")

        assert isinstance(fig, plt.Figure)
        mock_savefig.assert_called_once()
        plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_prc_curve_convenience(self, mock_show, mock_savefig):
        """Test PRC curve plotting convenience function."""
        fig = plot_prc_curve(self.y_true, self.y_scores, save_path="test.png")

        assert isinstance(fig, plt.Figure)
        mock_savefig.assert_called_once()
        plt.close(fig)


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def setup_method(self):
        """Setup test data."""
        self.analyzer = ROCAnalysisEngine()

    def test_roc_with_empty_arrays(self):
        """Test ROC calculation with empty arrays."""
        with pytest.raises(ValueError):
            self.analyzer.calculate_roc(np.array([]), np.array([]))

    def test_roc_with_mismatched_lengths(self):
        """Test ROC calculation with mismatched array lengths."""
        with pytest.raises(ValueError):
            self.analyzer.calculate_roc(np.array([0, 1]), np.array([0.5]))

    def test_roc_with_nan_values(self):
        """Test ROC calculation with NaN values."""
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.1, np.nan, 0.3, 0.8])

        # Should handle NaN gracefully or raise appropriate error
        try:
            result = self.analyzer.calculate_roc(y_true, y_scores)
            assert np.isnan(result["auc"]) or result["auc"] >= 0
        except ValueError:
            pass  # Acceptable to raise ValueError for NaN inputs


class TestPerformanceAndRobustness:
    """Test suite for performance and robustness."""

    def test_large_dataset_roc(self):
        """Test ROC calculation with large dataset."""
        np.random.seed(42)
        n_samples = 10000

        y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        y_scores = np.random.beta(2, 5, size=n_samples)

        analyzer = ROCAnalysisEngine()
        result = analyzer.calculate_roc(y_true, y_scores)

        assert 0 <= result["auc"] <= 1
        assert len(result["fpr"]) > 0
        assert len(result["tpr"]) > 0
        assert not np.isnan(result["auc_pvalue"])

    def test_extreme_separation_roc(self):
        """Test ROC with extreme separation."""
        analyzer = ROCAnalysisEngine()

        # Perfect separation
        y_true = np.array([0] * 100 + [1] * 100)
        y_scores = np.array([0.1] * 100 + [0.9] * 100)

        result = analyzer.calculate_roc(y_true, y_scores)
        assert result["auc"] == 1.0

        # Complete overlap (worst case)
        y_scores_overlap = np.array([0.5] * 200)
        result_overlap = analyzer.calculate_roc(y_true, y_scores_overlap)
        assert result_overlap["auc"] == 0.5


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
