"""
Unit tests for roc_prc_analysis module.
Tests ROC/PRC curve analysis, gene search optimization, and related functionality.
"""

import os
import sys
import warnings
from unittest import result
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.gene_search_optimizer import (
    GeneSearchOptimizer,
    backward_search,
    forward_search,
)


class TestGeneSearchOptimizer:
    """Test suite for GeneSearchOptimizer class."""

    def setup_method(self):
        """Setup test data for gene search optimization."""
        np.random.seed(42)
        self.optimizer = GeneSearchOptimizer()

        # Create realistic test data with gene expression
        self.meta_object = {
            "originalData": {
                "discovery1": {
                    "formattedName": "Discovery Dataset 1",
                    "expr": pd.DataFrame(
                        np.random.randn(10, 20) + 2,  # 10 probes, 20 samples
                        index=[f"probe_{i}" for i in range(10)],
                        columns=[f"sample_{i}" for i in range(20)],
                    ),
                    "keys": [
                        "GENE1",
                        "GENE2",
                        "GENE3",
                        "GENE4",
                        "GENE5",
                        "GENE1",
                        "GENE2",
                        "GENE6",
                        "GENE7",
                        "GENE8",
                    ],  # Some genes have multiple probes
                    "class": np.array([0] * 10 + [1] * 10),  # Binary classification
                },
                "discovery2": {
                    "formattedName": "Discovery Dataset 2",
                    "expr": pd.DataFrame(
                        np.random.randn(10, 16) + 2,  # 10 probes, 16 samples
                        index=[f"probe_{i}" for i in range(10)],
                        columns=[f"sample_{i}" for i in range(16)],
                    ),
                    "keys": [
                        "GENE1",
                        "GENE2",
                        "GENE3",
                        "GENE4",
                        "GENE5",
                        "GENE6",
                        "GENE7",
                        "GENE8",
                        "GENE1",
                        "GENE3",
                    ],  # Different probe mapping
                    "class": np.array([0] * 8 + [1] * 8),
                },
            }
        }

        self.filter_object = {
            "posGeneNames": ["GENE1", "GENE2", "GENE3", "GENE4"],
            "negGeneNames": ["GENE5", "GENE6"],
            "filterDescription": "Test filter",
        }

    def test_validate_meta_object_valid(self):
        """Test validation of valid meta object."""
        # Should not raise any exception
        self.optimizer._validate_meta_object(self.meta_object)

    def test_validate_meta_object_invalid(self):
        """Test validation of invalid meta objects."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            self.optimizer._validate_meta_object("not_dict")

        with pytest.raises(ValueError, match="must contain 'originalData'"):
            self.optimizer._validate_meta_object({"wrong_key": {}})

        with pytest.raises(ValueError, match="must contain at least one dataset"):
            self.optimizer._validate_meta_object({"originalData": {}})

    def test_validate_filter_object_valid(self):
        """Test validation of valid filter object."""
        # Should not raise any exception
        self.optimizer._validate_filter_object(self.filter_object)

    def test_validate_filter_object_invalid(self):
        """Test validation of invalid filter objects."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            self.optimizer._validate_filter_object("not_dict")

        with pytest.raises(ValueError, match="must contain 'posGeneNames'"):
            self.optimizer._validate_filter_object({"negGeneNames": []})

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_convert_discovery_list_to_genes(self, mock_get_sample_level):
        """Test conversion of discovery datasets to gene-level data."""
        pos_genes = ["GENE1", "GENE2"]
        neg_genes = ["GENE3", "GENE4"]

        result = self.optimizer._convert_discovery_list_to_genes(
            self.meta_object, pos_genes, neg_genes
        )

        assert len(result) == 2  # Two discovery datasets
        assert all("genes" in dataset for dataset in result)
        assert all("class" in dataset for dataset in result)

    def test_geometric_mean(self):
        """Test geometric mean calculation."""
        # Test with positive values
        data = np.array([[1, 4], [2, 8]])
        result = self.optimizer._geometric_mean(data, axis=0)
        expected = np.array([np.sqrt(2), np.sqrt(32)])  # sqrt(1*2), sqrt(4*8)
        np.testing.assert_array_almost_equal(result, expected)

    def test_geometric_mean_with_zeros(self):
        """Test geometric mean with zero values."""
        data = np.array([[0, 4], [2, 0]])
        result = self.optimizer._geometric_mean(data, axis=0)

        # Should handle zeros by adding epsilon
        assert len(result) == 2
        assert all(result > 0)  # All values should be positive

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_score_genes_matrix(self, mock_get_sample_level):
        """Test gene scoring from expression matrix."""
        # Create test gene matrix
        gene_matrix = pd.DataFrame(
            {"sample1": [2.0, 4.0, 1.0, 8.0], "sample2": [3.0, 2.0, 2.0, 6.0]},
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        pos_genes = ["GENE1", "GENE2"]
        neg_genes = ["GENE3"]

        scores = self.optimizer._score_genes_matrix(gene_matrix, pos_genes, neg_genes)

        assert len(scores) == 2  # Two samples
        assert isinstance(scores, np.ndarray)
        # Scores should be standardized (mean=0, std=1)
        # Note: We use ddof=1 (sample std) to match R's scale() function.
        # With ddof=0 (population std), a 2-element standardized vector has
        # std ≈ 0.707 rather than 1.0. This is intentional R-parity behavior.
        assert abs(np.mean(scores)) < 1e-10
        assert abs(np.std(scores, ddof=1) - 1) < 1e-10

    def test_score_genes_matrix_no_genes(self):
        """Test gene scoring with no genes present."""
        gene_matrix = pd.DataFrame(
            {"sample1": [1.0, 2.0], "sample2": [1.5, 2.5]}, index=["GENE1", "GENE2"]
        )

        scores = self.optimizer._score_genes_matrix(gene_matrix, [], [])

        assert len(scores) == 2
        assert all(score == 0 for score in scores)

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_get_weighted_aucs_genes_list(self, mock_get_sample_level):
        """Test weighted AUC calculation across gene lists."""
        # Convert to discovery genes format
        discovery_genes = self.optimizer._convert_discovery_list_to_genes(
            self.meta_object, ["GENE1", "GENE2"], ["GENE3", "GENE4"]
        )

        weights = np.array([20, 16])  # Sample sizes

        weighted_auc = self.optimizer._get_weighted_aucs_genes_list(
            discovery_genes, ["GENE1"], ["GENE2"], weights
        )

        assert isinstance(weighted_auc, float)
        assert weighted_auc >= 0

    def test_get_weighted_aucs_genes_list_no_genes(self):
        """Test weighted AUC with no genes."""
        discovery_genes = [
            {"genes": pd.DataFrame(), "class": np.array([0, 1, 0, 1])},
            {"genes": pd.DataFrame(), "class": np.array([0, 0, 1, 1])},
        ]
        weights = np.array([4, 4])

        result = self.optimizer._get_weighted_aucs_genes_list(
            discovery_genes, [], [], weights
        )

        assert result == 0.5 * np.sum(weights)  # Random performance

    def test_match_all_genes_valid(self):
        """Test gene matching validation with valid inputs."""
        discovery_genes = [
            {
                "genes": pd.DataFrame(index=["GENE1", "GENE2", "GENE3"]),
                "class": np.array([0, 1]),
            }
        ]

        # Should not raise any exception
        self.optimizer._match_all_genes(
            discovery_genes, ["GENE1", "GENE2"], ["GENE3"], ["GENE1"], []
        )

    def test_match_all_genes_invalid_yes_pos(self):
        """Test gene matching with invalid yes_pos."""
        discovery_genes = [
            {"genes": pd.DataFrame(index=["GENE1", "GENE2"]), "class": np.array([0, 1])}
        ]

        with pytest.raises(ValueError, match="yes_pos contains genes not in pos_genes"):
            self.optimizer._match_all_genes(
                discovery_genes, ["GENE1"], [], ["GENE2"], []
            )

    def test_match_all_genes_missing_genes_warning(self):
        """Test gene matching with missing genes issues warning."""
        discovery_genes = [
            {
                "genes": pd.DataFrame(index=["GENE1"]),  # Missing GENE2
                "class": np.array([0, 1]),
            }
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.optimizer._match_all_genes(
                discovery_genes, ["GENE1", "GENE2"], [], [], []
            )
            assert len(w) == 1
            assert "not found in discovery datasets" in str(w[0].message)

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_backward_search_integration(self, mock_get_sample_level):
        """Test backward search integration (mocked)."""
        # Mock the gene data conversion
        mock_gene_data = pd.DataFrame(
            np.random.randn(6, 20) + 3,
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6"],
            columns=[f"sample_{i}" for i in range(20)],
        )
        mock_get_sample_level.return_value = mock_gene_data

        result = self.optimizer.backward_search(
            self.meta_object, self.filter_object, back_thresh=-0.1
        )

        assert isinstance(result, dict)
        assert "posGeneNames" in result
        assert "negGeneNames" in result
        assert "filterDescription" in result
        assert "Backward search" in result["filterDescription"]

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_forward_search_integration(self, mock_get_sample_level):
        """Test forward search integration (mocked)."""
        # Mock the gene data conversion
        mock_gene_data = pd.DataFrame(
            np.random.randn(6, 20) + 3,
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6"],
            columns=[f"sample_{i}" for i in range(20)],
        )
        mock_get_sample_level.return_value = mock_gene_data

        result = self.optimizer.forward_search(
            self.meta_object,
            self.filter_object,
            yes_pos=["GENE1"],
            yes_neg=[],
            forward_thresh=0.1,
        )

        assert isinstance(result, dict)
        assert "posGeneNames" in result
        assert "negGeneNames" in result
        assert "filterDescription" in result
        assert "Forward search" in result["filterDescription"]

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_backward_search_inner(self, mock_get_sample_level):
        """Test backward search inner loop."""
        discovery_genes = self.optimizer._convert_discovery_list_to_genes(
            self.meta_object, ["GENE1", "GENE2"], ["GENE3", "GENE4"]
        )

        weights = np.array([20, 16])
        current_auc = 15.0

        result = self.optimizer._backward_search_inner(
            discovery_genes, ["GENE1", "GENE2"], ["GENE3"], current_auc, weights
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 2 pos + 1 neg gene
        assert all(
            col in result.columns for col in ["gene", "original", "search", "diff"]
        )

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_forward_search_inner(self, mock_get_sample_level):
        """Test forward search inner loop."""
        discovery_genes = self.optimizer._convert_discovery_list_to_genes(
            self.meta_object, ["GENE1", "GENE2", "GENE3", "GENE4"], []
        )

        weights = np.array([20, 16])
        current_auc = 10.0

        result = self.optimizer._forward_search_inner(
            discovery_genes, ["GENE3", "GENE4"], [], ["GENE1"], [], current_auc, weights
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 candidate genes
        assert all(
            col in result.columns for col in ["gene", "original", "search", "diff"]
        )


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.y_true = np.array([0, 0, 0, 1, 1, 1])
        self.y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_backward_search_convenience(self, mock_get_sample_level):
        """Test backward search convenience function."""
        # Mock data
        mock_gene_data = pd.DataFrame(np.random.randn(4, 10) + 3)
        mock_get_sample_level.return_value = mock_gene_data

        meta_object = {
            "originalData": {
                "study1": {
                    "expr": pd.DataFrame(np.random.randn(5, 10)),
                    "keys": ["G1", "G2", "G3", "G4", "G5"],
                    "class": np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
                }
            }
        }

        filter_object = {
            "posGeneNames": ["G1", "G2"],
            "negGeneNames": ["G3", "G4"],
            "filterDescription": "Test",
        }

        result = backward_search(meta_object, filter_object, back_thresh=0.0)

        assert isinstance(result, dict)
        assert "posGeneNames" in result
        assert "negGeneNames" in result

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_forward_search_convenience(self, mock_get_sample_level):
        """Test forward search convenience function."""
        # Mock data
        mock_gene_data = pd.DataFrame(np.random.randn(4, 10) + 3)
        mock_get_sample_level.return_value = mock_gene_data

        meta_object = {
            "originalData": {
                "study1": {
                    "expr": pd.DataFrame(np.random.randn(5, 10)),
                    "keys": ["G1", "G2", "G3", "G4", "G5"],
                    "class": np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
                }
            }
        }

        filter_object = {
            "posGeneNames": ["G1", "G2"],
            "negGeneNames": ["G3", "G4"],
            "filterDescription": "Test",
        }

        result = forward_search(
            meta_object, filter_object, yes_pos=["G1"], yes_neg=[], forward_thresh=0.0
        )

        assert isinstance(result, dict)
        assert "posGeneNames" in result
        assert "negGeneNames" in result


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def setup_method(self):
        """Setup test data."""
        pass

    def test_gene_search_with_no_discovery_datasets(self):
        """Test gene search with empty meta object."""
        optimizer = GeneSearchOptimizer()
        meta_object = {"originalData": {}}
        filter_object = {"posGeneNames": [], "negGeneNames": []}

        with pytest.raises(ValueError, match="must contain at least one dataset"):
            optimizer.backward_search(meta_object, filter_object)

    def test_gene_search_with_invalid_genes(self):
        """Test gene search with genes not in datasets."""
        optimizer = GeneSearchOptimizer()

        meta_object = {
            "originalData": {
                "study1": {
                    "expr": pd.DataFrame(
                        {"sample1": [1, 2]}, index=["probe1", "probe2"]
                    ),
                    "keys": ["GENE1", "GENE2"],
                    "class": np.array([0, 1]),
                }
            }
        }

        filter_object = {
            "posGeneNames": ["MISSING_GENE"],
            "negGeneNames": [],
            "filterDescription": "Test",
        }

        # Should handle missing genes gracefully with warning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # This might work but issue warnings, or raise exceptions
            # depending on implementation details
            try:
                result = optimizer.backward_search(meta_object, filter_object)
                assert isinstance(result, dict)
            except ImportError:
                # Expected if getSampleLevelGeneData not available
                pass


class TestPerformanceAndRobustness:
    """Test suite for performance and robustness."""

    def test_memory_efficiency_gene_search(self):
        """Test memory efficiency of gene search algorithms."""
        optimizer = GeneSearchOptimizer()

        # Create large mock gene matrix
        large_gene_matrix = pd.DataFrame(
            np.random.randn(1000, 100) + 3,  # 1000 genes, 100 samples
            index=[f"GENE_{i}" for i in range(1000)],
            columns=[f"sample_{i}" for i in range(100)],
        )

        pos_genes = [f"GENE_{i}" for i in range(50)]
        neg_genes = [f"GENE_{i}" for i in range(50, 100)]

        # This should not cause memory issues
        scores = optimizer._score_genes_matrix(large_gene_matrix, pos_genes, neg_genes)

        assert len(scores) == 100  # Number of samples
        assert isinstance(scores, np.ndarray)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
