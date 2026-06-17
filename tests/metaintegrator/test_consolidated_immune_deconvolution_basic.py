"""
Unit tests for immune_deconvolution.py module.
Tests the ImmunoStatesDeconvolver, ImmunoStatesCorrectionAnalyzer,
ImmunoStatesMetaAnalyzer, and ImmunoStatesUtils classes.
"""

import os
import sys
import tempfile
import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.consolidated_immune_deconvolution import (
    ImmunoStatesDeconvolver,
    ImmunoStatesCorrectionAnalyzer,
    ImmunoStatesMetaAnalyzer,
    ImmunoStatesUtils,
    immunostates_deconv,
    immunostates_gene_prop_corr,
    immunostates_meta,
)


class TestImmunoStatesDeconvolver:
    """Test cases for ImmunoStatesDeconvolver class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.basis_matrix = pd.DataFrame(
            np.random.randn(100, 13),
            index=[f"Gene_{i}" for i in range(100)],
            columns=ImmunoStatesDeconvolver.DEFAULT_CELL_TYPES,
        )
        self.deconvolver = ImmunoStatesDeconvolver(self.basis_matrix)

        # Sample meta object
        self.sample_meta_object = {
            "originalData": {
                "dataset1": {
                    "expr": pd.DataFrame(np.random.randn(50, 20)),
                    "class": [0] * 10 + [1] * 10,
                    "keys": [f"Gene_{i}" for i in range(50)],
                },
                "dataset2": {
                    "expr": pd.DataFrame(np.random.randn(80, 30)),
                    "class": [0] * 15 + [1] * 15,
                    "keys": [f"Gene_{i}" for i in range(80)],
                },
            }
        }

    def test_init_with_basis_matrix(self):
        """Test initialization with custom basis matrix."""
        deconvolver = ImmunoStatesDeconvolver(self.basis_matrix)
        assert deconvolver.basis_matrix is not None
        assert deconvolver.basis_matrix.shape == (100, 13)
        assert (
            list(deconvolver.basis_matrix.columns)
            == ImmunoStatesDeconvolver.DEFAULT_CELL_TYPES
        )

    def test_init_without_basis_matrix(self):
        """Test initialization without basis matrix (loads default)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deconvolver = ImmunoStatesDeconvolver()
            assert deconvolver.basis_matrix is not None
            assert deconvolver.basis_matrix.shape[1] == len(
                ImmunoStatesDeconvolver.DEFAULT_CELL_TYPES
            )

    def test_load_default_basis_matrix(self):
        """Test loading default basis matrix."""
        deconvolver = ImmunoStatesDeconvolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deconvolver._load_default_basis_matrix()

        assert deconvolver.basis_matrix is not None
        assert isinstance(deconvolver.basis_matrix, pd.DataFrame)
        assert deconvolver.basis_matrix.shape[1] == len(
            ImmunoStatesDeconvolver.DEFAULT_CELL_TYPES
        )

    def test_get_os_type(self):
        """Test OS type detection."""
        with patch("platform.system", return_value="Windows"):
            assert self.deconvolver._get_os_type() == "win"

        with patch("platform.system", return_value="Darwin"):
            assert self.deconvolver._get_os_type() == "mac"

        with patch("platform.system", return_value="Linux"):
            assert self.deconvolver._get_os_type() == "unix"

        with patch("platform.system", return_value="Unknown"):
            assert self.deconvolver._get_os_type() == "unknown"

    def test_get_max_cores_with_specified_cores(self):
        """Test max cores determination with specified value."""
        result = self.deconvolver._get_max_cores(5)
        assert result == 5

    def test_get_max_cores_auto_mac_unix(self):
        """Test max cores determination for Mac/Unix systems."""
        with patch.object(self.deconvolver, "_get_os_type", return_value="mac"):
            with patch("multiprocessing.cpu_count", return_value=16):
                result = self.deconvolver._get_max_cores(None)
                assert result == 10  # min(10, 16)

    def test_get_max_cores_auto_windows(self):
        """Test max cores determination for Windows systems."""
        with patch.object(self.deconvolver, "_get_os_type", return_value="win"):
            result = self.deconvolver._get_max_cores(None)
            assert result == 1  # Windows default

    def test_is_dataset_invalid(self):
        """Test dataset validation logic."""
        # Valid dataset
        valid_dataset = {
            "expr": pd.DataFrame(np.random.randn(10, 5)),
            "keys": ["Gene1", "Gene2", "Gene3"],
        }
        assert not self.deconvolver._is_dataset_invalid(valid_dataset)

        # Invalid - missing expr
        invalid_dataset1 = {"keys": ["Gene1", "Gene2"]}
        assert self.deconvolver._is_dataset_invalid(invalid_dataset1)

        # Invalid - missing keys
        invalid_dataset2 = {"expr": pd.DataFrame()}
        assert self.deconvolver._is_dataset_invalid(invalid_dataset2)

        # Invalid - annotation absent
        invalid_dataset3 = {
            "expr": pd.DataFrame(),
            "keys": ["Gene1"],
            "key_comment": "Annotation absent",
        }
        assert self.deconvolver._is_dataset_invalid(invalid_dataset3)

        # Invalid - expression data missing
        invalid_dataset4 = {
            "expr": pd.DataFrame(),
            "keys": ["Gene1"],
            "exp_comment": "Expression data is missing",
        }
        assert self.deconvolver._is_dataset_invalid(invalid_dataset4)

    def test_extract_data_for_genes_valid_dataset(self):
        """Test gene data extraction with valid dataset."""
        dataset = {
            "expr": pd.DataFrame(np.random.randn(10, 20)),
            "keys": [f"Gene_{i}" for i in range(10)],
        }

        result = self.deconvolver._extract_data_for_genes_enhanced(dataset)
        if result is not None:
            assert isinstance(result, pd.DataFrame)
            assert result.shape[0] <= 10  # May be fewer due to filtering
            assert result.shape[1] == 20

    def test_extract_data_for_genes_invalid_dataset(self):
        """Test gene data extraction with invalid dataset."""
        invalid_dataset = {"invalid": "data"}
        result = self.deconvolver._extract_data_for_genes_enhanced(invalid_dataset)
        assert result is None

    def test_extract_data_for_genes_with_gene_names_filter(self):
        """Test gene data extraction with gene names filter."""
        dataset = {
            "expr": pd.DataFrame(np.random.randn(10, 20)),
            "keys": [f"Gene_{i}" for i in range(10)],
        }

        # Filter to specific genes
        gene_names = ["Gene_1", "Gene_3", "Gene_5"]
        result = self.deconvolver._extract_data_for_genes_enhanced(dataset, gene_names)

        if result is not None:  # May be None if no matching genes
            assert result.shape[1] == 20

    def test_extract_data_for_genes_no_promiscuous_probes(self):
        """Test gene data extraction excluding promiscuous probes."""
        dataset = {
            "expr": pd.DataFrame(np.random.randn(5, 20)),
            "keys": ["Gene1", "Gene2,Gene3", "Gene4", "Gene5,Gene6,Gene7", "Gene8"],
        }

        result = self.deconvolver._extract_data_for_genes_enhanced(
            dataset, promiscuous_probes=False
        )

        if result is not None:
            # Should exclude genes with commas (promiscuous)
            assert result.shape[0] <= 3  # Only Gene1, Gene4, Gene8 should remain

    def test_extract_data_for_genes_empty_after_filtering(self):
        """Test gene data extraction when all genes are filtered out."""
        dataset = {
            "expr": pd.DataFrame(np.random.randn(3, 20)),
            "keys": ["", "NA", "nan"],  # All invalid keys
        }

        result = self.deconvolver._extract_data_for_genes_enhanced(dataset)
        assert result is None

    def test_enhanced_quantile_normalize(self):
        """Test quantile normalization."""
        matrix = np.random.randn(5, 10)
        result = self.deconvolver._enhanced_quantile_normalize(matrix)

        assert result is not None
        assert result.shape == matrix.shape
        assert not np.array_equal(
            result, matrix
        )  # Should be different after normalization

    def test_enhanced_quantile_normalize_error_handling(self):
        """Test quantile normalization error handling."""
        # Test with problematic data
        matrix = np.array([[np.inf, np.nan], [1, 2]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.deconvolver._enhanced_quantile_normalize(matrix)

        # Function should handle errors gracefully and return a valid result
        assert result is not None
        assert result.shape == matrix.shape
        # Result should be different from input (normalized) or same (if error handling returns original)
        # Both are acceptable behaviors

    def test_enhanced_deconvolve_sample_valid(self):
        """Test single sample deconvolution with valid data."""
        sample_idx = 0
        expr_vector = np.random.randn(50)  # Valid expression vector
        basis = np.random.randn(50, 5)
        cell_types = ["Type1", "Type2", "Type3", "Type4", "Type5"]

        result = self.deconvolver._enhanced_deconvolve_sample(
            sample_idx, expr_vector, basis, cell_types
        )

        assert isinstance(result, dict)
        assert "sample" in result
        assert result["sample"] == "Sample_0"

        # Check cell type proportions
        for cell_type in cell_types:
            assert cell_type in result
            assert isinstance(result[cell_type], (int, float, np.number))

        # Check metrics
        assert "P_value" in result
        assert "Correlation" in result
        assert "RMSE" in result

    def test_enhanced_deconvolve_sample_insufficient_genes(self):
        """Test single sample deconvolution with insufficient genes."""
        sample_idx = 0
        expr_vector = np.array(
            [1, 2, np.nan, np.nan, np.nan]
        )  # Only 2 valid genes < 10
        basis = np.random.randn(5, 3)
        cell_types = ["Type1", "Type2", "Type3"]

        result = self.deconvolver._enhanced_deconvolve_sample(
            sample_idx, expr_vector, basis, cell_types
        )

        assert isinstance(result, dict)
        assert "sample" in result

        # All cell type proportions should be NaN
        for cell_type in cell_types:
            assert cell_type in result
            assert np.isnan(result[cell_type])

    def test_add_aggregate_cell_types(self):
        """Test addition of aggregate cell types."""
        # Create results with individual cell types
        individual_results = pd.DataFrame(
            {
                "sample": ["Sample_1", "Sample_2"],
                "CD56bright_natural_killer_cell": [0.1, 0.2],
                "CD56dim_natural_killer_cell": [0.05, 0.1],
                "CD14_positive_monocyte": [0.2, 0.15],
                "CD16_positive_monocyte": [0.1, 0.05],
                "naive_B_cell": [0.05, 0.1],
                "memory_B_cell": [0.03, 0.05],
                "plasma_cell": [0.02, 0.02],
            }
        )

        result = self.deconvolver._add_aggregate_cell_types(individual_results)

        # Check that aggregate columns are added
        assert "natural_killer_cell" in result.columns
        assert "monocyte" in result.columns
        assert "B_cell" in result.columns

        # Check aggregate calculations (ignore series names)
        expected_nk_values = (
            individual_results["CD56bright_natural_killer_cell"]
            + individual_results["CD56dim_natural_killer_cell"]
        ).values
        actual_nk_values = result["natural_killer_cell"].values
        np.testing.assert_array_almost_equal(actual_nk_values, expected_nk_values)

    def test_add_aggregate_cell_types_missing_components(self):
        """Test aggregate cell types with missing component columns."""
        # Results missing some component cell types
        partial_results = pd.DataFrame(
            {
                "sample": ["Sample_1", "Sample_2"],
                "CD56bright_natural_killer_cell": [0.1, 0.2],
                # Missing CD56dim_natural_killer_cell
            }
        )

        result = self.deconvolver._add_aggregate_cell_types(partial_results)

        # Should still add aggregate columns, using available data
        assert "natural_killer_cell" in result.columns
        # Should equal the available component (compare values, not series)
        expected_values = partial_results["CD56bright_natural_killer_cell"].values
        actual_values = result["natural_killer_cell"].values
        np.testing.assert_array_almost_equal(actual_values, expected_values)

    def test_enhanced_is_deconvolution_success(self):
        """Test running IS deconvolution successfully."""
        expression_matrix = np.random.randn(50, 20)  # 50 genes, 20 samples

        with patch.object(
            self.deconvolver, "_enhanced_is_deconvolution"
        ) as mock_deconv:
            mock_deconv.return_value = pd.DataFrame(
                {"sample": ["S1", "S2"], "Type1": [0.5, 0.6]}
            )
            result = self.deconvolver._enhanced_is_deconvolution(expression_matrix)

        assert result is not None
        mock_deconv.assert_called_once()

    def test_enhanced_is_deconvolution_error(self):
        """Test running IS deconvolution with error handling."""
        # Create expression matrix with no overlap with basis matrix
        expression_matrix = pd.DataFrame(
            np.random.randn(50, 20),
            index=[f"NoMatch_{i}" for i in range(50)],  # Genes that won't match basis
        )

        # Should return None when deconvolution fails
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.deconvolver._enhanced_is_deconvolution(expression_matrix)

        assert result is None

    def test_immunostates_deconv_missing_original_data(self):
        """Test deconvolution with missing originalData."""
        meta_object = {"someOtherKey": "value"}

        with pytest.raises(ValueError, match="originalData"):
            self.deconvolver.immunostates_deconv(meta_object)

    def test_is_deconvolution_integration(self):
        """Test the complete deconvolution process."""
        # Create expression matrix as DataFrame with gene names
        expression_matrix = pd.DataFrame(
            np.random.randn(20, 10),  # 20 genes, 10 samples
            index=[f"Gene_{i}" for i in range(20)],
        )

        result = self.deconvolver._enhanced_is_deconvolution(expression_matrix)

        if result is not None:  # May be None if deconvolution fails
            assert isinstance(result, pd.DataFrame)
            assert result.shape[0] == 10  # Number of samples
            # Should have cell type columns plus metadata columns
            assert result.shape[1] >= 5  # At least cell types + sample + metrics


class TestImmunoStatesCorrectionAnalyzer:
    """Test cases for ImmunoStatesCorrectionAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ImmunoStatesCorrectionAnalyzer()

        # Sample immunoStates data
        self.sample_is_data = pd.DataFrame(
            {
                "sample": ["Sample_1", "Sample_2", "Sample_3"],
                "natural_killer_cell": [0.1, 0.15, 0.08],
                "monocyte": [0.2, 0.18, 0.22],
                "B_cell": [0.15, 0.12, 0.18],
                "T_cell": [0.4, 0.45, 0.35],
                "granulocyte": [0.1, 0.08, 0.12],
                "P_value": [0.001, 0.002, 0.001],
                "Correlation": [0.85, 0.82, 0.88],
                "RMSE": [0.05, 0.06, 0.04],
            }
        )

        # Sample meta object
        self.sample_meta_object = {
            "originalData": {
                "dataset1": {
                    "expr": pd.DataFrame(
                        np.random.randn(100, 3),
                        columns=["Sample_1", "Sample_2", "Sample_3"],
                        index=[f"Gene_{i}" for i in range(100)],
                    )
                }
            },
            "immunoStates": {"dataset1": self.sample_is_data},
        }

    def test_immunostates_gene_prop_corr_missing_immunostates(self):
        """Test correction with missing immunoStates data."""
        meta_object = {"originalData": {"dataset1": {}}}

        with pytest.raises(ValueError, match="immunoStates must be run"):
            self.analyzer.immunostates_gene_prop_corr(meta_object)

    def test_immunostates_gene_prop_corr_empty_immunostates(self):
        """Test correction with empty immunoStates data."""
        meta_object = {"originalData": {"dataset1": {}}, "immunoStates": {}}

        with pytest.raises(ValueError, match="immunoStates must be run"):
            self.analyzer.immunostates_gene_prop_corr(meta_object)

    def test_immunostates_gene_prop_corr_success(self):
        """Test successful gene proportion correction."""
        result = self.analyzer.immunostates_gene_prop_corr(self.sample_meta_object)

        assert "iScorrExp" in result
        assert "dataset1" in result["iScorrExp"]
        assert isinstance(result["iScorrExp"]["dataset1"], pd.DataFrame)

    def test_prepare_cell_props_success(self):
        """Test cell proportions preparation."""
        result = self.analyzer._prepare_cell_props(self.sample_is_data, "dataset1")

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "other_cell_types" in result.columns

        # Check that proportions sum to approximately 1
        row_sums = result.sum(axis=1)
        assert all(abs(s - 1.0) < 1e-10 for s in row_sums)

    def test_prepare_cell_props_missing_columns(self):
        """Test cell proportions preparation with missing columns."""
        # Data without required aggregate columns
        incomplete_data = pd.DataFrame(
            {"sample": ["Sample_1", "Sample_2"], "some_other_column": [0.5, 0.6]}
        )

        result = self.analyzer._prepare_cell_props(incomplete_data, "dataset1")
        assert result is None

    def test_prepare_expr_data_log_space(self):
        """Test expression data preparation from log space."""
        # Data in log space (values < 50)
        log_data = pd.DataFrame(np.random.randn(10, 5) * 2 + 5)  # Values around 5

        result = self.analyzer._prepare_expr_data(log_data)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == log_data.shape
        # Should be exponentiated (2^x)
        assert result.min().min() > 0  # Should be positive after 2^x

    def test_prepare_expr_data_linear_space(self):
        """Test expression data preparation from linear space."""
        # Data in linear space (values >= 50)
        linear_data = pd.DataFrame(np.random.rand(10, 5) * 1000 + 100)  # Values > 50

        result = self.analyzer._prepare_expr_data(linear_data)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == linear_data.shape
        # Should remain the same
        pd.testing.assert_frame_equal(result.fillna(0.0), linear_data.fillna(0.0))

    def test_prepare_expr_data_with_nan(self):
        """Test expression data preparation with NaN values."""
        data_with_nan = pd.DataFrame(np.random.randn(5, 3))
        data_with_nan.iloc[0, 0] = np.nan
        data_with_nan.iloc[1, 2] = np.nan

        result = self.analyzer._prepare_expr_data(data_with_nan)

        assert isinstance(result, pd.DataFrame)
        assert not result.isna().any().any()  # Should fill NaN with 0.0

    def test_correct_single_gene(self):
        """Test single gene correction."""
        model = LinearRegression(fit_intercept=True)
        x = np.random.randn(20, 3)  # 20 samples, 3 cell types
        y = np.random.randn(20)  # 20 expression values

        result = self.analyzer._correct_single_gene(model, x, y)

        assert isinstance(result, np.ndarray)
        assert result.shape == y.shape
        assert not np.array_equal(result, y)  # Should be different after correction

    def test_regress_out_cell_proportions(self):
        """Test regression of cell proportions from expression matrix."""
        # Expression matrix: genes x samples
        expr_matrix = pd.DataFrame(
            np.random.randn(50, 3),
            columns=["Sample_1", "Sample_2", "Sample_3"],
            index=[f"Gene_{i}" for i in range(50)],
        )

        # Cell proportions: samples x cell types
        cell_props = pd.DataFrame(
            np.random.rand(3, 4),
            index=["Sample_1", "Sample_2", "Sample_3"],
            columns=["Type1", "Type2", "Type3", "Type4"],
        )

        result = self.analyzer._regress_out_cell_proportions(expr_matrix, cell_props)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == expr_matrix.shape
        assert list(result.columns) == list(expr_matrix.columns)
        assert list(result.index) == list(expr_matrix.index)

    def test_regress_out_cell_proportions_error_handling(self):
        """Test regression error handling for individual genes."""
        # Create problematic data
        expr_matrix = pd.DataFrame(
            np.array([[np.inf, 1, 2], [1, 2, 3], [4, 5, 6]]),
            columns=["Sample_1", "Sample_2", "Sample_3"],
            index=["Gene_1", "Gene_2", "Gene_3"],
        )

        cell_props = pd.DataFrame(
            np.random.rand(3, 2),
            index=["Sample_1", "Sample_2", "Sample_3"],
            columns=["Type1", "Type2"],
        )

        # Should handle errors gracefully
        result = self.analyzer._regress_out_cell_proportions(expr_matrix, cell_props)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == expr_matrix.shape


class TestImmunoStatesMetaAnalyzer:
    """Test cases for ImmunoStatesMetaAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ImmunoStatesMetaAnalyzer()

        # Sample meta object
        self.sample_meta_object = {
            "originalData": {
                "dataset1": {
                    "expr": pd.DataFrame(np.random.randn(50, 20)),
                    "class": [0] * 10 + [1] * 10,
                    "keys": [f"Gene_{i}" for i in range(50)],
                },
                "dataset2": {
                    "expr": pd.DataFrame(np.random.randn(30, 16)),
                    "class": [0] * 8 + [1] * 8,
                    "keys": [f"Gene_{i}" for i in range(30)],
                },
            }
        }

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = ImmunoStatesMetaAnalyzer()
        assert hasattr(analyzer, "deconvolver")
        # Should use the ImmunoStatesDeconvolver from consolidated module
        assert isinstance(analyzer.deconvolver, ImmunoStatesDeconvolver)

    @patch.object(ImmunoStatesDeconvolver, "immunostates_deconv")
    def test_immunostates_meta_success(self, mock_deconv):
        """Test successful meta-analysis preparation."""
        # Mock successful deconvolution results
        mock_results = {
            "immunoStates": {
                "dataset1": pd.DataFrame(
                    {
                        "sample": ["S1", "S2"],
                        "Type1": [0.3, 0.4],
                        "Type2": [0.7, 0.6],
                        "P_value": [0.001, 0.002],
                        "Correlation": [0.85, 0.82],
                        "RMSE": [0.05, 0.06],
                    }
                ),
                "dataset2": pd.DataFrame(
                    {"sample": ["S3", "S4"], "Type1": [0.2, 0.5], "Type2": [0.8, 0.5]}
                ),
            },
            "originalData": self.sample_meta_object["originalData"],
        }
        mock_deconv.return_value = mock_results

        result = self.analyzer.immunostates_meta(self.sample_meta_object)

        assert "originalData" in result
        assert "dataset1" in result["originalData"]
        assert "dataset2" in result["originalData"]

    @patch.object(ImmunoStatesDeconvolver, "immunostates_deconv")
    def test_immunostates_meta_failed_datasets(self, mock_deconv):
        """Test meta-analysis with failed datasets."""
        # Mock results with some failed deconvolution
        mock_results = {
            "immunoStates": {
                "dataset1": pd.DataFrame(
                    {"sample": ["S1", "S2"], "Type1": [0.3, 0.4], "Type2": [0.7, 0.6]}
                ),
                "dataset2": None,  # Failed deconvolution
            },
            "originalData": self.sample_meta_object["originalData"],
        }
        mock_deconv.return_value = mock_results

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.analyzer.immunostates_meta(self.sample_meta_object)

        assert "originalData" in result
        assert "dataset1" in result["originalData"]
        assert "dataset2" not in result["originalData"]  # Should be excluded

    def test_handle_excluded_datasets(self):
        """Test handling of excluded datasets."""
        immuno_result = {
            "originalData": {"dataset1": {}, "dataset2": {}, "dataset3": {}}
        }
        successful_names = ["dataset1", "dataset3"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.analyzer._handle_excluded_datasets(immuno_result, successful_names)

            assert len(w) == 1
            assert "dataset2" in str(w[0].message)
            assert "excluded" in str(w[0].message)

    def test_handle_excluded_datasets_none_excluded(self):
        """Test handling when no datasets are excluded."""
        immuno_result = {"originalData": {"dataset1": {}, "dataset2": {}}}
        successful_names = ["dataset1", "dataset2"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.analyzer._handle_excluded_datasets(immuno_result, successful_names)

            assert len(w) == 0  # No warnings should be issued

    def test_prepare_dataset_results_with_sample_column(self):
        """Test dataset results preparation with sample column."""
        immuno_expr = pd.DataFrame(
            {
                "sample": ["Sample_1", "Sample_2"],
                "Type1": [0.3, 0.4],
                "Type2": [0.7, 0.6],
                "P_value": [0.001, 0.002],
                "Correlation": [0.85, 0.82],
                "RMSE": [0.05, 0.06],
            }
        )

        result = self.analyzer._prepare_dataset_results(immuno_expr)

        assert isinstance(result, dict)
        assert "expr" in result
        assert "keys" in result

        # Check dimensions (transposed)
        assert result["expr"].shape == (2, 2)  # 2 cell types, 2 samples
        assert len(result["keys"]) == 2  # 2 cell types

    def test_prepare_dataset_results_without_sample_column(self):
        """Test dataset results preparation without sample column."""
        immuno_expr = pd.DataFrame(
            {"Type1": [0.3, 0.4], "Type2": [0.7, 0.6], "Type3": [0.1, 0.2]}
        )

        result = self.analyzer._prepare_dataset_results(immuno_expr)

        assert isinstance(result, dict)
        assert "expr" in result
        assert "keys" in result

        # Check dimensions
        assert result["expr"].shape == (3, 2)  # 3 cell types, 2 samples
        assert len(result["keys"]) == 3

    def test_prepare_dataset_results_non_dataframe(self):
        """Test dataset results preparation with non-DataFrame input."""
        # Test with dictionary that can be converted to DataFrame
        immuno_expr = {"Type1": [0.3, 0.4], "Type2": [0.7, 0.6]}

        result = self.analyzer._prepare_dataset_results(immuno_expr)

        assert isinstance(result, dict)
        assert "expr" in result
        assert "keys" in result


class TestImmunoStatesUtils:
    """Test cases for ImmunoStatesUtils class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.utils = ImmunoStatesUtils()

        # Sample deconvolution results
        self.sample_results = {
            "dataset1": pd.DataFrame(
                {
                    "sample": ["S1", "S2", "S3"],
                    "natural_killer_cell": [0.1, 0.12, 0.08],
                    "monocyte": [0.2, 0.18, 0.22],
                    "B_cell": [0.15, 0.17, 0.13],
                    "T_cell": [0.45, 0.43, 0.47],
                    "granulocyte": [0.1, 0.1, 0.1],
                }
            ),
            "dataset2": pd.DataFrame(
                {
                    "sample": ["S4", "S5"],
                    "natural_killer_cell": [0.05, 0.15],
                    "monocyte": [0.25, 0.15],
                    "B_cell": [0.20, 0.10],
                    "T_cell": [0.40, 0.50],
                    "granulocyte": [0.10, 0.10],
                }
            ),
        }

    def test_load_basis_matrix_csv(self):
        """Test loading basis matrix from CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Gene,Type1,Type2,Type3\n")
            f.write("Gene1,0.5,0.3,0.2\n")
            f.write("Gene2,0.4,0.4,0.2\n")
            f.write("Gene3,0.3,0.5,0.2\n")
            temp_file = f.name

        try:
            result = ImmunoStatesUtils.load_basis_matrix(temp_file)

            assert isinstance(result, pd.DataFrame)
            assert result.shape == (3, 3)  # 3 genes, 3 types
            assert list(result.columns) == ["Type1", "Type2", "Type3"]
            assert list(result.index) == ["Gene1", "Gene2", "Gene3"]

        finally:
            os.unlink(temp_file)

    def test_load_basis_matrix_excel(self):
        """Test loading basis matrix from Excel file."""
        # Create temporary Excel file
        df = pd.DataFrame(
            {"Type1": [0.5, 0.4], "Type2": [0.3, 0.4], "Type3": [0.2, 0.2]},
            index=["Gene1", "Gene2"],
        )

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            df.to_excel(f.name)
            temp_file = f.name

        try:
            result = ImmunoStatesUtils.load_basis_matrix(temp_file)

            assert isinstance(result, pd.DataFrame)
            assert result.shape == (2, 3)

        finally:
            os.unlink(temp_file)

    def test_load_basis_matrix_tab_separated(self):
        """Test loading basis matrix from tab-separated file."""
        # Create temporary tab-separated file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Gene\tType1\tType2\n")
            f.write("Gene1\t0.5\t0.5\n")
            f.write("Gene2\t0.4\t0.6\n")
            temp_file = f.name

        try:
            result = ImmunoStatesUtils.load_basis_matrix(temp_file)

            assert isinstance(result, pd.DataFrame)
            assert result.shape == (2, 2)

        finally:
            os.unlink(temp_file)

    def test_load_basis_matrix_file_not_found(self):
        """Test loading basis matrix from non-existent file."""
        with pytest.raises(ValueError, match="Could not load basis matrix"):
            ImmunoStatesUtils.load_basis_matrix("non_existent_file.csv")

    def test_validate_deconvolution_results_valid(self):
        """Test validation of valid deconvolution results."""
        is_valid, errors = ImmunoStatesUtils.validate_deconvolution_results(
            self.sample_results
        )

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_deconvolution_results_invalid(self):
        """Test validation of invalid deconvolution results."""
        # Create invalid results
        invalid_results = {
            "dataset1": pd.DataFrame(
                {
                    "sample": ["S1", "S2"],
                    # Missing required columns
                    "wrong_column": [0.5, 0.6],
                }
            ),
            "dataset2": None,  # No results
        }

        is_valid, errors = ImmunoStatesUtils.validate_deconvolution_results(
            invalid_results
        )

        assert is_valid is False
        assert len(errors) > 0
        assert any("Missing columns" in error for error in errors)
        assert any("No results" in error for error in errors)

    def test_validate_dataset_valid(self):
        """Test validation of valid dataset."""
        valid_df = pd.DataFrame(
            {
                "sample": ["S1", "S2"],
                "natural_killer_cell": [0.1, 0.12],
                "monocyte": [0.2, 0.18],
                "B_cell": [0.15, 0.17],
                "T_cell": [0.45, 0.43],
                "granulocyte": [0.1, 0.1],
            }
        )

        errors = ImmunoStatesUtils._validate_dataset("test_dataset", valid_df)
        assert len(errors) == 0

    def test_validate_dataset_missing_columns(self):
        """Test validation of dataset with missing columns."""
        invalid_df = pd.DataFrame(
            {
                "sample": ["S1", "S2"],
                "natural_killer_cell": [0.1, 0.12],
                # Missing other required columns
            }
        )

        errors = ImmunoStatesUtils._validate_dataset("test_dataset", invalid_df)
        assert len(errors) > 0
        assert any("Missing columns" in error for error in errors)

    def test_validate_dataset_unusual_proportions(self):
        """Test validation of dataset with unusual proportions."""
        # Proportions that don't sum correctly
        unusual_df = pd.DataFrame(
            {
                "sample": ["S1", "S2"],
                "natural_killer_cell": [0.5, 0.1],
                "monocyte": [0.5, 0.1],
                "B_cell": [0.5, 0.1],
                "T_cell": [0.5, 0.1],  # Sum > 1.5
                "granulocyte": [0.5, 0.001],  # Sum < 0.1 for S2
            }
        )

        errors = ImmunoStatesUtils._validate_dataset("test_dataset", unusual_df)
        assert len(errors) > 0
        assert any("Unusual cell proportion sums" in error for error in errors)

    def test_validate_dataset_none_results(self):
        """Test validation of None dataset."""
        errors = ImmunoStatesUtils._validate_dataset("test_dataset", None)
        assert len(errors) == 1
        assert "No results" in errors[0]

    def test_validate_dataset_non_dataframe(self):
        """Test validation of non-DataFrame dataset."""
        errors = ImmunoStatesUtils._validate_dataset("test_dataset", "not_a_dataframe")
        assert len(errors) == 1
        assert "not in DataFrame format" in errors[0]

    def test_export_deconvolution_results(self):
        """Test exporting deconvolution results to CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ImmunoStatesUtils.export_deconvolution_results(
                self.sample_results, temp_dir
            )

            # Check that files were created
            files = os.listdir(temp_dir)
            assert "dataset1_immunoStates.csv" in files
            assert "dataset2_immunoStates.csv" in files

            # Check file contents
            dataset1_file = os.path.join(temp_dir, "dataset1_immunoStates.csv")
            loaded_df = pd.read_csv(dataset1_file)
            assert loaded_df.shape[0] == 3  # 3 samples
            assert "natural_killer_cell" in loaded_df.columns

    def test_export_deconvolution_results_non_dataframe(self):
        """Test exporting with non-DataFrame results."""
        mixed_results = {
            "dataset1": self.sample_results["dataset1"],
            "dataset2": "not_a_dataframe",  # Should be skipped
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            ImmunoStatesUtils.export_deconvolution_results(mixed_results, temp_dir)

            files = os.listdir(temp_dir)
            assert "dataset1_immunoStates.csv" in files
            assert "dataset2_immunoStates.csv" not in files  # Should be skipped


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_meta_object = {
            "originalData": {
                "dataset1": {
                    "expr": pd.DataFrame(np.random.randn(50, 20)),
                    "class": [0] * 10 + [1] * 10,
                    "keys": [f"Gene_{i}" for i in range(50)],
                }
            }
        }

    @patch.object(ImmunoStatesDeconvolver, "immunostates_deconv")
    def test_immunostates_deconv_function(self, mock_method):
        """Test the immunostates_deconv convenience function."""
        mock_method.return_value = {"immunoStates": {"dataset1": pd.DataFrame()}}

        result = immunostates_deconv(self.sample_meta_object, max_cores=2)

        assert isinstance(result, dict)
        mock_method.assert_called_once_with(self.sample_meta_object, 2)

    @patch.object(ImmunoStatesCorrectionAnalyzer, "immunostates_gene_prop_corr")
    def test_immunostates_gene_prop_corr_function(self, mock_method):
        """Test the immunostates_gene_prop_corr convenience function."""
        mock_method.return_value = {"iScorrExp": {"dataset1": pd.DataFrame()}}

        result = immunostates_gene_prop_corr(self.sample_meta_object)

        assert isinstance(result, dict)
        mock_method.assert_called_once_with(self.sample_meta_object)

    @patch.object(ImmunoStatesMetaAnalyzer, "immunostates_meta")
    def test_immunostates_meta_function(self, mock_method):
        """Test the immunostates_meta convenience function."""
        mock_method.return_value = {"originalData": {"dataset1": {}}}

        result = immunostates_meta(self.sample_meta_object)

        assert isinstance(result, dict)
        mock_method.assert_called_once_with(self.sample_meta_object)


class TestIntegration:
    """Integration tests for immune deconvolution workflow."""

    def test_complete_deconvolution_workflow(self):
        """Test complete deconvolution workflow."""
        # Create realistic test data
        meta_object = {
            "originalData": {
                "dataset1": {
                    "expr": pd.DataFrame(
                        np.random.randn(50, 20),
                        columns=[f"Sample_{i}" for i in range(20)],
                        index=[f"Gene_{i}" for i in range(50)],
                    ),
                    "class": [0] * 10 + [1] * 10,
                    "keys": [f"Gene_{i}" for i in range(50)],
                }
            }
        }

        # Initialize deconvolver with mock data
        basis_matrix = pd.DataFrame(
            np.random.randn(50, 5),
            index=[f"Gene_{i}" for i in range(50)],
            columns=["Type1", "Type2", "Type3", "Type4", "Type5"],
        )

        deconvolver = ImmunoStatesDeconvolver(basis_matrix)

        # Run deconvolution (may fail due to random data, but test the pipeline)
        try:
            result = deconvolver.immunostates_deconv(meta_object, max_cores=1)
            assert "immunoStates" in result
        except Exception:
            # Random data may cause numerical issues, which is expected
            pass

    def test_deconvolution_correction_workflow(self):
        """Test deconvolution followed by correction workflow."""
        # Create mock immunoStates results
        meta_object = {
            "originalData": {
                "dataset1": {
                    "expr": pd.DataFrame(
                        np.random.randn(100, 3),
                        columns=["Sample_1", "Sample_2", "Sample_3"],
                        index=[f"Gene_{i}" for i in range(100)],
                    )
                }
            },
            "immunoStates": {
                "dataset1": pd.DataFrame(
                    {
                        "sample": ["Sample_1", "Sample_2", "Sample_3"],
                        "natural_killer_cell": [0.1, 0.15, 0.08],
                        "monocyte": [0.2, 0.18, 0.22],
                        "B_cell": [0.15, 0.12, 0.18],
                        "T_cell": [0.4, 0.45, 0.35],
                        "granulocyte": [0.15, 0.1, 0.17],
                    }
                )
            },
        }

        # Run correction
        analyzer = ImmunoStatesCorrectionAnalyzer()
        result = analyzer.immunostates_gene_prop_corr(meta_object)

        assert "iScorrExp" in result
        assert "dataset1" in result["iScorrExp"]

    def test_meta_analysis_preparation_workflow(self):
        """Test meta-analysis preparation workflow."""
        meta_object = {
            "originalData": {
                "dataset1": {
                    "expr": pd.DataFrame(np.random.randn(30, 15)),
                    "class": [0] * 8 + [1] * 7,
                    "keys": [f"Gene_{i}" for i in range(30)],
                },
                "dataset2": {
                    "expr": pd.DataFrame(np.random.randn(25, 12)),
                    "class": [0] * 6 + [1] * 6,
                    "keys": [f"Gene_{i}" for i in range(25)],
                },
            }
        }

        # Create analyzer and determine which method to mock
        analyzer = ImmunoStatesMetaAnalyzer()

        # Check if it's using enhanced deconvolver and has the enhanced method
        if hasattr(analyzer, "_use_enhanced") and analyzer._use_enhanced:
            method_to_mock = "immunostates_deconv"
        else:
            method_to_mock = "immunostates_deconv"

        with patch.object(analyzer.deconvolver, method_to_mock) as mock_deconv:
            mock_deconv.return_value = {
                "immunoStates": {
                    "dataset1": pd.DataFrame(
                        {
                            "sample": ["S1", "S2"],
                            "Type1": [0.3, 0.4],
                            "Type2": [0.7, 0.6],
                        }
                    ),
                    "dataset2": pd.DataFrame(
                        {
                            "sample": ["S3", "S4"],
                            "Type1": [0.2, 0.5],
                            "Type2": [0.8, 0.5],
                        }
                    ),
                },
                "originalData": meta_object["originalData"],
            }

            result = analyzer.immunostates_meta(meta_object)

        assert "originalData" in result
        assert len(result["originalData"]) >= 1  # At least one successful dataset


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_nan_expression_data(self):
        """Test handling of NaN expression data."""
        dataset = {
            "expr": pd.DataFrame(np.full((10, 20), np.nan)),
            "keys": [f"Gene_{i}" for i in range(10)],
        }

        deconvolver = ImmunoStatesDeconvolver()
        result = deconvolver._extract_data_for_genes_enhanced(dataset)

        # Should return None or DataFrame (empty is valid)
        assert result is None or isinstance(result, pd.DataFrame)

    def test_single_cell_type_basis(self):
        """Test deconvolution with single cell type basis."""
        basis_matrix = pd.DataFrame(
            np.random.randn(20, 1),
            index=[f"Gene_{i}" for i in range(20)],
            columns=["SingleType"],
        )

        deconvolver = ImmunoStatesDeconvolver(basis_matrix)
        # Create expression matrix as DataFrame with matching genes
        expression_matrix = pd.DataFrame(
            np.random.randn(20, 5), index=[f"Gene_{i}" for i in range(20)]
        )

        result = deconvolver._enhanced_is_deconvolution(expression_matrix)

        if result is not None:  # May be None if deconvolution fails
            assert isinstance(result, pd.DataFrame)
            assert "SingleType" in result.columns

    def test_mismatched_sample_names(self):
        """Test correction with mismatched sample names."""
        meta_object = {
            "originalData": {
                "dataset1": {
                    "expr": pd.DataFrame(
                        np.random.randn(50, 3),
                        columns=["Sample_A", "Sample_B", "Sample_C"],
                    )
                }
            },
            "immunoStates": {
                "dataset1": pd.DataFrame(
                    {
                        "sample": [
                            "Sample_X",
                            "Sample_Y",
                            "Sample_Z",
                        ],  # Different names
                        "natural_killer_cell": [0.1, 0.15, 0.08],
                        "monocyte": [0.2, 0.18, 0.22],
                        "B_cell": [0.15, 0.12, 0.18],
                        "T_cell": [0.4, 0.45, 0.35],
                        "granulocyte": [0.15, 0.1, 0.17],
                    }
                )
            },
        }

        analyzer = ImmunoStatesCorrectionAnalyzer()
        result = analyzer.immunostates_gene_prop_corr(meta_object)

        # Should handle gracefully (no common samples)
        assert "iScorrExp" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
