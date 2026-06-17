"""
Comprehensive test suite for subset_original_data function.

This tests the Python port of R's subsetOriginalData() function,
ensuring parity with the original R implementation.

FIXED: test_mixed_case_sample_names() now correctly tests case sensitivity
"""

import os

# Import the function to test
import sys

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# Import the classes to test
from metaintegrator.data_processing import DataProcessor, subsetOriginalData

# ==================================================================
# TEST FIXTURES
# ==================================================================


@pytest.fixture
def basic_dataset():
    """Create a basic test dataset."""
    n_genes = 100
    n_samples = 10

    # Create expression matrix (genes x samples)
    expr_data = np.random.randn(n_genes, n_samples)
    sample_names = [f"Sample_{i+1}" for i in range(n_samples)]
    gene_names = [f"Gene_{i+1}" for i in range(n_genes)]

    expr_df = pd.DataFrame(expr_data, index=gene_names, columns=sample_names)

    # Create class vector (alternating 0s and 1s)
    class_vector = np.array([i % 2 for i in range(n_samples)])

    # Create gene keys
    keys = gene_names.copy()

    return {"expr": expr_df, "class": class_vector, "keys": keys}


@pytest.fixture
def dataset_with_pheno():
    """Create dataset with phenotype data."""
    n_genes = 50
    n_samples = 8

    sample_names = [f"Sample_{i+1}" for i in range(n_samples)]
    gene_names = [f"Gene_{i+1}" for i in range(n_genes)]

    expr_df = pd.DataFrame(
        np.random.randn(n_genes, n_samples), index=gene_names, columns=sample_names
    )

    class_vector = np.array([0, 0, 1, 1, 0, 1, 1, 0])

    # Create phenotype DataFrame
    pheno_df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
            "treatment": ["A", "B", "A", "B", "A", "B", "A", "B"],
        },
        index=sample_names,
    )

    return {
        "expr": expr_df,
        "class": class_vector,
        "keys": gene_names,
        "pheno": pheno_df,
    }


# ==================================================================
# BASIC FUNCTIONALITY TESTS
# ==================================================================


class TestBasicSubsetting:
    """Test basic subsetting functionality."""

    def test_subset_by_sample_names(self, basic_dataset):
        """Test subsetting using sample names."""
        keep_samples = ["Sample_1", "Sample_3", "Sample_5"]
        result = DataProcessor.subset_original_data(basic_dataset, keep_samples)

        # Check dimensions
        assert result["expr"].shape[1] == 3
        assert len(result["class"]) == 3
        assert len(result["keys"]) == 100  # genes should be unchanged

        # Check correct samples selected
        assert list(result["expr"].columns) == keep_samples

        # Check class vector subset correctly
        original_classes = basic_dataset["class"][[0, 2, 4]]
        np.testing.assert_array_equal(result["class"], original_classes)

    def test_subset_by_indices(self, basic_dataset):
        """Test subsetting using integer indices."""
        keep_indices = [0, 2, 4, 6]
        result = DataProcessor.subset_original_data(basic_dataset, keep_indices)

        # Check dimensions
        assert result["expr"].shape[1] == 4
        assert len(result["class"]) == 4

        # Check correct samples selected
        expected_names = ["Sample_1", "Sample_3", "Sample_5", "Sample_7"]
        assert list(result["expr"].columns) == expected_names

    def test_subset_by_boolean_array(self, basic_dataset):
        """Test subsetting using boolean mask."""
        keep_mask = np.array(
            [True, False, True, False, True, False, False, False, False, False]
        )
        result = DataProcessor.subset_original_data(basic_dataset, keep_mask)

        # Check dimensions
        assert result["expr"].shape[1] == 3
        assert len(result["class"]) == 3

        # Check correct samples selected
        expected_names = ["Sample_1", "Sample_3", "Sample_5"]
        assert list(result["expr"].columns) == expected_names

    def test_subset_preserves_gene_dimension(self, basic_dataset):
        """Verify that gene dimension is preserved."""
        keep_samples = ["Sample_1", "Sample_5"]
        result = DataProcessor.subset_original_data(basic_dataset, keep_samples)

        # Genes should be unchanged
        assert result["expr"].shape[0] == 100
        assert len(result["keys"]) == 100
        assert list(result["expr"].index) == list(basic_dataset["expr"].index)


# ==================================================================
# ERROR HANDLING TESTS
# ==================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_samples_warning(self, basic_dataset):
        """Test warning for non-existent samples."""
        keep_samples = ["Sample_1", "NonExistent", "Sample_3"]

        with pytest.warns(UserWarning, match="not present"):
            result = DataProcessor.subset_original_data(basic_dataset, keep_samples)

        # Should only keep the 2 valid samples
        assert result["expr"].shape[1] == 2
        assert list(result["expr"].columns) == ["Sample_1", "Sample_3"]

    def test_no_valid_samples_error(self, basic_dataset):
        """Test error when no valid samples specified."""
        keep_samples = ["NonExistent1", "NonExistent2"]

        with pytest.raises(ValueError, match="No specified sample names are present"):
            DataProcessor.subset_original_data(basic_dataset, keep_samples)

    def test_empty_keep_samples_error(self, basic_dataset):
        """Test error for empty keep_samples list."""
        with pytest.raises(ValueError, match="keep_samples cannot be empty"):
            DataProcessor.subset_original_data(basic_dataset, [])

    def test_invalid_indices_error(self, basic_dataset):
        """Test error for out-of-range indices."""
        keep_indices = [0, 2, 15]  # 15 is out of range (only 10 samples)

        with pytest.raises(ValueError, match="Invalid sample indices"):
            DataProcessor.subset_original_data(basic_dataset, keep_indices)

    def test_wrong_boolean_length_error(self, basic_dataset):
        """Test error for boolean array with wrong length."""
        wrong_mask = np.array([True, False, True])  # Only 3 elements, should be 10

        with pytest.raises(ValueError, match="Boolean mask length"):
            DataProcessor.subset_original_data(basic_dataset, wrong_mask)

    def test_missing_required_keys_error(self):
        """Test error when dataset missing required keys."""
        incomplete_dataset = {"expr": pd.DataFrame()}

        with pytest.raises(KeyError, match="missing required keys"):
            DataProcessor.subset_original_data(incomplete_dataset, ["Sample_1"])

    def test_invalid_keep_samples_type_error(self, basic_dataset):
        """Test error for invalid keep_samples type."""
        with pytest.raises(TypeError, match="keep_samples must be"):
            DataProcessor.subset_original_data(basic_dataset, "invalid_type")


# ==================================================================
# DATA INTEGRITY TESTS
# ==================================================================


class TestDataIntegrity:
    """Test that expr, class, and pheno stay aligned after subsetting."""

    def test_expr_class_alignment(self, basic_dataset):
        """Verify expression and class data stay aligned."""
        keep_samples = ["Sample_2", "Sample_5", "Sample_8"]
        keep_indices = [1, 4, 7]

        result = DataProcessor.subset_original_data(basic_dataset, keep_samples)

        # Check that class values match original
        original_classes = basic_dataset["class"][keep_indices]
        np.testing.assert_array_equal(result["class"], original_classes)

        # Check expression values match
        for i, sample_name in enumerate(keep_samples):
            original_col = basic_dataset["expr"][sample_name]
            result_col = result["expr"].iloc[:, i]
            np.testing.assert_array_equal(result_col.values, original_col.values)

    def test_pheno_subsetting(self, dataset_with_pheno):
        """Test that phenotype data is subsetted correctly."""
        keep_samples = ["Sample_2", "Sample_5", "Sample_7"]
        result = DataProcessor.subset_original_data(dataset_with_pheno, keep_samples)

        # Check pheno data was subsetted
        assert "pheno" in result
        assert result["pheno"].shape[0] == 3
        assert list(result["pheno"].index) == keep_samples

        # Verify pheno data matches original
        for sample in keep_samples:
            original_pheno = dataset_with_pheno["pheno"].loc[sample]
            result_pheno = result["pheno"].loc[sample]
            pd.testing.assert_series_equal(result_pheno, original_pheno)

    def test_metadata_preservation(self, basic_dataset):
        """Test that other metadata fields are preserved."""
        # Add some metadata
        basic_dataset["study_name"] = "Test Study"
        basic_dataset["platform"] = "GPL570"
        basic_dataset["formattedName"] = "TestStudy_v1"

        keep_samples = ["Sample_1", "Sample_5"]
        result = DataProcessor.subset_original_data(basic_dataset, keep_samples)

        # Check metadata preserved
        assert result["study_name"] == "Test Study"
        assert result["platform"] == "GPL570"
        assert result["formattedName"] == "TestStudy_v1"
        assert result["keys"] == basic_dataset["keys"]


# ==================================================================
# SPECIAL CASES TESTS
# ==================================================================


class TestSpecialCases:
    """Test special cases and edge scenarios."""

    def test_subset_single_sample(self, basic_dataset):
        """Test subsetting to a single sample."""
        keep_samples = ["Sample_5"]
        result = DataProcessor.subset_original_data(basic_dataset, keep_samples)

        assert result["expr"].shape[1] == 1
        assert len(result["class"]) == 1
        assert result["expr"].columns[0] == "Sample_5"

    def test_subset_all_samples(self, basic_dataset):
        """Test subsetting to all samples (identity operation)."""
        all_samples = list(basic_dataset["expr"].columns)
        result = DataProcessor.subset_original_data(basic_dataset, all_samples)

        # Should be equivalent to original
        assert result["expr"].shape == basic_dataset["expr"].shape
        pd.testing.assert_frame_equal(result["expr"], basic_dataset["expr"])
        np.testing.assert_array_equal(result["class"], basic_dataset["class"])

    def test_numpy_array_expression_matrix(self):
        """Test with numpy array instead of DataFrame."""
        dataset = {
            "expr": np.random.randn(50, 10),  # numpy array, not DataFrame
            "class": np.array([0, 1] * 5),
            "keys": [f"Gene_{i}" for i in range(50)],
        }

        keep_indices = [0, 2, 4]
        result = DataProcessor.subset_original_data(dataset, keep_indices)

        # Should work and result in DataFrame
        assert isinstance(result["expr"], pd.DataFrame)
        assert result["expr"].shape[1] == 3

    def test_subset_with_duplicate_names(self, basic_dataset):
        """Test behavior when duplicate sample names requested."""
        keep_samples = ["Sample_1", "Sample_3", "Sample_1"]  # Sample_1 twice

        # Should only keep unique samples
        result = DataProcessor.subset_original_data(basic_dataset, keep_samples)

        # Check dimensions - duplicates should result in repeated samples
        assert result["expr"].shape[1] == 3

    def test_mixed_case_sample_names(self):
        """Test that sample names are case-sensitive (FIXED VERSION)."""
        dataset = {
            "expr": pd.DataFrame(
                np.random.randn(10, 5),
                columns=["sample_1", "SAMPLE_2", "Sample_3", "sample_4", "SAMPLE_5"],
            ),
            "class": np.array([0, 1, 0, 1, 0]),
            "keys": [f"Gene_{i}" for i in range(10)],
        }

        # Test 1: Exact case match - should work
        keep_samples = ["sample_1", "Sample_3"]
        result = DataProcessor.subset_original_data(dataset, keep_samples)
        assert result["expr"].shape[1] == 2
        assert list(result["expr"].columns) == ["sample_1", "Sample_3"]

        # Test 2: Complete case mismatch - should raise ValueError
        # Dataset has: ["sample_1", "SAMPLE_2", "Sample_3", "sample_4", "SAMPLE_5"]
        # Requesting: ["Sample_1", "sample_3"] - neither matches!
        keep_samples_wrong = ["Sample_1", "sample_3"]  # Wrong cases for both
        with pytest.raises(ValueError, match="No specified sample names are present"):
            DataProcessor.subset_original_data(dataset, keep_samples_wrong)

        # Test 3: Partial case mismatch - should warn and return only valid samples
        keep_samples_partial = ["sample_1", "Sample_1", "Sample_3"]  # 2 valid, 1 wrong
        with pytest.warns(UserWarning, match="not present"):
            result_partial = DataProcessor.subset_original_data(
                dataset, keep_samples_partial
            )
            # Should only keep the 2 valid samples (sample_1 and Sample_3)
            assert result_partial["expr"].shape[1] == 2
            assert list(result_partial["expr"].columns) == ["sample_1", "Sample_3"]


# ==================================================================
# R INTERFACE COMPATIBILITY TESTS
# ==================================================================


class TestRCompatibility:
    """Test compatibility with R's subsetOriginalData interface."""

    def test_r_function_name(self, basic_dataset):
        """Test that R-style function name works."""
        keepMe = ["Sample_1", "Sample_3", "Sample_5"]
        result = subsetOriginalData(basic_dataset, keepMe)

        assert result["expr"].shape[1] == 3
        assert list(result["expr"].columns) == keepMe

    def test_r_binary_vector_style(self, basic_dataset):
        """Test R-style binary vector subsetting."""
        # R would use c(TRUE, FALSE, TRUE, ...)
        keepMe = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
        result = subsetOriginalData(basic_dataset, keepMe)

        assert result["expr"].shape[1] == 3

    def test_r_named_sample_subset(self, basic_dataset):
        """Test R-style named sample subset (exact R example from docs)."""
        # From R docs: keepMe= c("Sample 1", "Sample 13", "Sample 43")
        # Adapting to our test data
        keepMe = ["Sample_1", "Sample_3", "Sample_5"]
        result = subsetOriginalData(basic_dataset, keepMe)

        assert isinstance(result, dict)
        assert "expr" in result
        assert "class" in result
        assert result["expr"].shape[1] == len(keepMe)


# ==================================================================
# PERFORMANCE TESTS
# ==================================================================


class TestPerformance:
    """Test performance with larger datasets."""

    def test_large_dataset_subsetting(self):
        """Test subsetting large dataset."""
        n_genes = 20000  # Typical microarray
        n_samples = 100

        large_dataset = {
            "expr": pd.DataFrame(
                np.random.randn(n_genes, n_samples),
                columns=[f"Sample_{i}" for i in range(n_samples)],
            ),
            "class": np.random.randint(0, 2, n_samples),
            "keys": [f"Gene_{i}" for i in range(n_genes)],
        }

        # Subset to 20 samples
        keep_samples = [f"Sample_{i}" for i in range(0, 100, 5)]  # Every 5th sample

        import time

        start = time.time()
        result = DataProcessor.subset_original_data(large_dataset, keep_samples)
        elapsed = time.time() - start

        # Should complete quickly (< 1 second for this size)
        assert elapsed < 1.0
        assert result["expr"].shape == (n_genes, 20)
        assert len(result["class"]) == 20


# ==================================================================
# INTEGRATION TESTS
# ==================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_remove_outlier_samples(self, dataset_with_pheno):
        """Test removing outlier samples workflow."""
        # Simulate identifying outliers (e.g., samples 2 and 6)
        all_samples = list(dataset_with_pheno["expr"].columns)
        outliers = ["Sample_2", "Sample_6"]
        keep_samples = [s for s in all_samples if s not in outliers]

        result = DataProcessor.subset_original_data(dataset_with_pheno, keep_samples)

        assert result["expr"].shape[1] == 6  # 8 - 2 outliers
        assert "Sample_2" not in result["expr"].columns
        assert "Sample_6" not in result["expr"].columns

    def test_stratified_sampling(self, dataset_with_pheno):
        """Test stratified sampling by class."""
        # Get 2 samples from each class
        class_0_samples = [
            dataset_with_pheno["expr"].columns[i]
            for i, c in enumerate(dataset_with_pheno["class"])
            if c == 0
        ]
        class_1_samples = [
            dataset_with_pheno["expr"].columns[i]
            for i, c in enumerate(dataset_with_pheno["class"])
            if c == 1
        ]

        # Take 2 from each
        stratified_samples = class_0_samples[:2] + class_1_samples[:2]

        result = DataProcessor.subset_original_data(
            dataset_with_pheno, stratified_samples
        )

        # Should have 2 cases and 2 controls
        assert result["expr"].shape[1] == 4
        assert np.sum(result["class"] == 0) == 2
        assert np.sum(result["class"] == 1) == 2

    def test_batch_correction_subset(self, dataset_with_pheno):
        """Test subsetting for batch correction."""
        # Subset to only batch A samples (based on pheno)
        batch_a_samples = dataset_with_pheno["pheno"][
            dataset_with_pheno["pheno"]["treatment"] == "A"
        ].index.tolist()

        result = DataProcessor.subset_original_data(dataset_with_pheno, batch_a_samples)

        assert result["expr"].shape[1] == 4  # 4 'A' treatment samples
        assert all(result["pheno"]["treatment"] == "A")


# ==================================================================
# MAIN TEST RUNNER
# ==================================================================

if __name__ == "__main__":
    """Run all tests with verbose output."""
    pytest.main([__file__, "-v", "--tb=short", "-s"])
