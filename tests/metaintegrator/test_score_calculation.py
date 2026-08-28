"""
Unit tests for score_calculation.py

Tests for the ScoreCalculator class and calculateScore function.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

print(f"\n::: project_root: {project_root}")

from metaintegrator.score_calculation import ScoreCalculator, calculateScore


class TestScoreCalculator:
    """Test suite for ScoreCalculator class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        # Create expression data (genes x samples)
        np.random.seed(42)
        n_genes, n_samples = 20, 30

        expr_data = pd.DataFrame(
            np.random.rand(n_genes, n_samples) * 10,
            index=[f"probe_{i}" for i in range(n_genes)],
            columns=[f"sample_{i}" for i in range(n_samples)],
        )

        # Add some differential expression
        expr_data.iloc[0:5, 15:30] += 3  # Upregulated in second half

        keys = [f"GENE{i//2}" for i in range(n_genes)]  # 2 probes per gene

        dataset = {"expr": expr_data, "keys": keys}

        return dataset

    @pytest.fixture
    def sample_filter(self):
        """Create a sample filter object."""
        return {
            "posGeneNames": ["GENE0", "GENE1", "GENE2"],
            "negGeneNames": ["GENE8", "GENE9"],
        }

    def test_calculate_score_basic(self, sample_filter, sample_dataset):
        """Test basic score calculation."""
        scores = ScoreCalculator.calculate_score(
            sample_filter, sample_dataset, suppress_messages=True
        )

        # Check output shape
        assert len(scores) == sample_dataset["expr"].shape[1]

        # Check scores are standardized (approximately mean=0, std=1)
        assert abs(np.mean(scores)) < 1e-10
        assert abs(np.std(scores, ddof=1) - 1.0) < 1e-10

    def test_calculate_score_pos_only(self, sample_dataset):
        """Test with only positive genes."""
        filter_obj = {"posGeneNames": ["GENE0", "GENE1"]}

        scores = ScoreCalculator.calculate_score(filter_obj, sample_dataset, suppress_messages=True)

        assert len(scores) == sample_dataset["expr"].shape[1]
        assert abs(np.mean(scores)) < 1e-10

    def test_calculate_score_neg_only(self, sample_dataset):
        """Test with only negative genes."""
        filter_obj = {"negGeneNames": ["GENE8", "GENE9"]}

        scores = ScoreCalculator.calculate_score(filter_obj, sample_dataset, suppress_messages=True)

        assert len(scores) == sample_dataset["expr"].shape[1]
        assert abs(np.mean(scores)) < 1e-10

    def test_calculate_score_no_matching_genes(self, sample_dataset):
        """Test with genes not in dataset."""
        filter_obj = {
            "posGeneNames": ["NONEXISTENT1"],
            "negGeneNames": ["NONEXISTENT2"],
        }

        scores = ScoreCalculator.calculate_score(filter_obj, sample_dataset, suppress_messages=True)

        # Should return zeros when no genes match
        assert np.all(scores == 0)

    def test_calculate_score_negative_expression(self):
        """Test with negative expression values."""
        # Create dataset with negative values
        expr_data = pd.DataFrame(
            np.random.randn(10, 20),  # Can have negative values
            columns=[f"S{i}" for i in range(20)],
        )

        dataset = {"expr": expr_data, "keys": [f"GENE{i}" for i in range(10)]}

        filter_obj = {"posGeneNames": ["GENE0", "GENE1"]}

        # Should handle negative values by shifting
        scores = ScoreCalculator.calculate_score(filter_obj, dataset, suppress_messages=True)

        assert len(scores) == 20
        assert abs(np.mean(scores)) < 1e-10

    def test_geometric_mean(self):
        """Test geometric mean calculation."""
        # Test simple case
        x = pd.Series([1, 2, 4, 8])
        result = ScoreCalculator._geometric_mean(x)
        expected = (1 * 2 * 4 * 8) ** (1 / 4)
        assert np.isclose(result, expected)

        # Test with single value
        x = pd.Series([5.0])
        result = ScoreCalculator._geometric_mean(x)
        assert np.isclose(result, 5.0)

        # Test with NaN values (should be ignored)
        x = pd.Series([2, np.nan, 8])
        result = ScoreCalculator._geometric_mean(x)
        expected = (2 * 8) ** (1 / 2)
        assert np.isclose(result, expected)

    def test_geometric_mean_negative_values(self):
        """Test that geometric mean raises error for negative values."""
        x = pd.Series([1, -2, 3])
        with pytest.raises(ValueError, match="negative value"):
            ScoreCalculator._geometric_mean(x)

    def test_standardize(self):
        """Test standardization function."""
        # Test simple array
        x = np.array([1, 2, 3, 4, 5])
        result = ScoreCalculator._standardize(x)

        # Check mean is 0 and std is 1
        assert np.isclose(np.mean(result), 0, atol=1e-10)
        assert np.isclose(np.std(result, ddof=1), 1, atol=1e-10)

        # Test with constant array (std=0)
        x = np.array([5, 5, 5, 5])
        result = ScoreCalculator._standardize(x)
        assert np.all(result == 0)

    def test_validate_filter_object(self):
        """Test filter object validation."""
        # Valid filter
        valid_filter = {"posGeneNames": ["G1"]}
        ScoreCalculator._validate_filter_object(valid_filter)  # Should not raise

        # Invalid: not a dict
        with pytest.raises(TypeError):
            ScoreCalculator._validate_filter_object(["G1"])

        # Invalid: missing both keys
        with pytest.raises(ValueError):
            ScoreCalculator._validate_filter_object({})

    def test_validate_dataset_object(self):
        """Test dataset object validation."""
        # Valid dataset
        valid_dataset = {
            "expr": pd.DataFrame(np.random.rand(5, 10)),
            "keys": ["G1"] * 5,
        }
        ScoreCalculator._validate_dataset_object(valid_dataset)  # Should not raise

        # Invalid: not a dict
        with pytest.raises(TypeError):
            ScoreCalculator._validate_dataset_object([])

        # Invalid: missing expr
        with pytest.raises(ValueError):
            ScoreCalculator._validate_dataset_object({"keys": ["G1"]})

        # Invalid: expr not DataFrame
        with pytest.raises(TypeError):
            ScoreCalculator._validate_dataset_object(
                {"expr": [[1, 2], [3, 4]], "keys": ["G1", "G2"]}
            )

    def test_convenience_function(self, sample_filter, sample_dataset):
        """Test the convenience calculateScore function."""
        scores = calculateScore(sample_filter, sample_dataset, suppress_messages=True)

        assert len(scores) == sample_dataset["expr"].shape[1]
        assert abs(np.mean(scores)) < 1e-10


class TestScoreCalculatorIntegration:
    """Integration tests for score calculation."""

    def test_full_workflow(self):
        """Test complete score calculation workflow."""
        # Create realistic dataset
        np.random.seed(123)
        n_samples = 40

        # Expression data for 30 genes
        expr_data = pd.DataFrame(
            np.random.gamma(2, 2, (30, n_samples)),
            index=[f"probe_{i}" for i in range(30)],
            columns=[f"S{i:03d}" for i in range(n_samples)],
        )

        # Add differential expression for cases (second 20 samples)
        for i in range(10):  # First 10 genes upregulated
            expr_data.iloc[i, 20:] *= 2

        for i in range(20, 25):  # Some genes downregulated
            expr_data.iloc[i, 20:] *= 0.5

        keys = [f"GENE{i//3:03d}" for i in range(30)]  # 3 probes per gene

        dataset = {"expr": expr_data, "keys": keys}

        filter_obj = {
            "posGeneNames": [f"GENE{i:03d}" for i in range(5)],  # Upregulated genes
            "negGeneNames": [f"GENE{i:03d}" for i in range(6, 10)],  # Some downregulated
        }

        scores = ScoreCalculator.calculate_score(filter_obj, dataset, suppress_messages=False)

        # Check scores differentiate between controls and cases
        control_scores = scores[:20]
        case_scores = scores[20:]

        # Cases should have higher mean score (due to upregulated pos genes)
        assert np.mean(case_scores) > np.mean(control_scores)

        # Check standardization
        assert abs(np.mean(scores)) < 1e-10
        assert abs(np.std(scores, ddof=1) - 1.0) < 1e-10

    def test_score_with_missing_data(self):
        """Test score calculation with missing expression data."""
        np.random.seed(42)

        # Create dataset with some NaN values
        expr_data = pd.DataFrame(np.random.rand(15, 25), columns=[f"S{i}" for i in range(25)])

        # Add some NaN values
        expr_data.iloc[0, 5:8] = np.nan
        expr_data.iloc[3, 10:12] = np.nan

        dataset = {"expr": expr_data, "keys": [f"GENE{i}" for i in range(15)]}

        filter_obj = {
            "posGeneNames": ["GENE0", "GENE1", "GENE2"],
            "negGeneNames": ["GENE10", "GENE11"],
        }

        # Should handle NaN values
        scores = ScoreCalculator.calculate_score(filter_obj, dataset, suppress_messages=True)

        assert len(scores) == 25
        # Scores should be valid (no NaN in output)
        assert not np.any(np.isnan(scores))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
