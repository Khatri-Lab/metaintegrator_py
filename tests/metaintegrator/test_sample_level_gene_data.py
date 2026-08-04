"""
Unit tests for sample_level_gene_data module.
Tests the Python implementation against expected R behavior.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.sample_level_gene_data import (
    SampleLevelGeneDataProcessor,
    getSampleLevelGeneData,
    extractDataFromGEM,
    replaceValues,
    replaceNaNs,
    replaceNAs,
)


class TestSampleLevelGeneDataProcessor:
    """Test suite for SampleLevelGeneDataProcessor class."""

    def setup_method(self):
        """Setup test data for each test method."""
        # Create sample expression data (6 probes x 4 samples)
        self.expr_data = pd.DataFrame(
            {
                "Sample1": [5.2, 3.1, 7.8, 0.5, np.nan, 2.3],
                "Sample2": [4.8, 2.9, 8.1, 0.3, 6.2, 1.8],
                "Sample3": [5.5, 3.3, 7.5, 0.7, 5.8, 2.1],
                "Sample4": [4.9, 3.0, 8.0, 0.0, 6.1, np.nan],
            },
            index=[
                "PROBE001",
                "PROBE002",
                "PROBE003",
                "PROBE004",
                "PROBE005",
                "PROBE006",
            ],
        )

        # Keys mapping probes to genes (some probes map to multiple genes)
        self.keys = ["GENE1", "GENE2", "GENE1,GENE3", "GENE2", "GENE3", "GENE4"]

        # Create dataset object
        self.dataset_object = {"expr": self.expr_data, "keys": self.keys}

        # Target genes for testing
        self.target_genes = ["GENE1", "GENE2", "GENE3"]

    def test_validate_dataset_object_valid(self):
        """Test validation of valid dataset object."""
        # Should not raise any exception
        SampleLevelGeneDataProcessor._validate_dataset_object(self.dataset_object)

    def test_validate_dataset_object_missing_keys(self):
        """Test validation fails with missing keys."""
        invalid_dataset = {"expr": self.expr_data}
        with pytest.raises(ValueError, match="missing required keys"):
            SampleLevelGeneDataProcessor._validate_dataset_object(invalid_dataset)

    def test_validate_dataset_object_wrong_type(self):
        """Test validation fails with wrong data types."""
        invalid_dataset = {"expr": "not_a_dataframe", "keys": self.keys}
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            SampleLevelGeneDataProcessor._validate_dataset_object(invalid_dataset)

    def test_validate_dataset_object_mismatched_lengths(self):
        """Test validation fails with mismatched expr/keys lengths."""
        invalid_dataset = {
            "expr": self.expr_data,
            "keys": ["GENE1", "GENE2"],  # Too few keys
        }
        with pytest.raises(ValueError, match="Length of keys must match"):
            SampleLevelGeneDataProcessor._validate_dataset_object(invalid_dataset)

    def test_validate_gene_names_valid(self):
        """Test validation of valid gene names."""
        # Should not raise any exception
        SampleLevelGeneDataProcessor._validate_gene_names(self.target_genes)
        SampleLevelGeneDataProcessor._validate_gene_names(np.array(self.target_genes))
        SampleLevelGeneDataProcessor._validate_gene_names(pd.Series(self.target_genes))

    def test_validate_gene_names_empty(self):
        """Test validation fails with empty gene names."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SampleLevelGeneDataProcessor._validate_gene_names([])

    def test_validate_gene_names_wrong_type(self):
        """Test validation fails with wrong type."""
        with pytest.raises(TypeError, match="must be a list, numpy array"):
            SampleLevelGeneDataProcessor._validate_gene_names("not_a_list")

    def test_expand_df_sample_level_basic(self):
        """Test basic DataFrame expansion functionality."""
        simple_df = pd.DataFrame(
            {"Sample1": [1.0, 2.0], "Sample2": [1.5, 2.5]}, index=["PROBE1", "PROBE2"]
        )

        simple_keys = ["GENE1", "GENE2,GENE3"]

        result = SampleLevelGeneDataProcessor._expand_df_sample_level(simple_df, simple_keys)

        # Should have 3 rows (1 for GENE1, 2 for GENE2,GENE3)
        assert len(result) == 3
        assert "keys" in result.columns
        assert result["keys"].tolist() == ["GENE1", "GENE2", "GENE3"]

        # Check that values are correctly duplicated
        assert result.loc[result["keys"] == "GENE1", "Sample1"].iloc[0] == 1.0
        assert result.loc[result["keys"] == "GENE2", "Sample1"].iloc[0] == 2.0
        assert result.loc[result["keys"] == "GENE3", "Sample1"].iloc[0] == 2.0

    def test_expand_df_sample_level_with_spaces(self):
        """Test DataFrame expansion with spaces in gene names."""
        simple_df = pd.DataFrame({"Sample1": [1.0], "Sample2": [1.5]}, index=["PROBE1"])

        simple_keys = ["GENE1, GENE2 , GENE3"]

        result = SampleLevelGeneDataProcessor._expand_df_sample_level(simple_df, simple_keys)

        # Should strip spaces from gene names
        expected_genes = ["GENE1", "GENE2", "GENE3"]
        assert result["keys"].tolist() == expected_genes

    def test_summarize_split_single_probe(self):
        """Test summarization with single probe per gene."""
        single_probe_data = pd.DataFrame({"Sample1": [5.0], "Sample2": [6.0], "keys": ["GENE1"]})

        result = SampleLevelGeneDataProcessor._summarize_split(single_probe_data)

        expected = pd.Series([5.0, 6.0], index=["Sample1", "Sample2"])
        pd.testing.assert_series_equal(result, expected)

    def test_summarize_split_multiple_probes(self):
        """Test summarization with multiple probes per gene."""
        multi_probe_data = pd.DataFrame(
            {"Sample1": [4.0, 6.0], "Sample2": [5.0, 7.0], "keys": ["GENE1", "GENE1"]}
        )

        result = SampleLevelGeneDataProcessor._summarize_split(multi_probe_data)

        # Should return means: (4+6)/2=5, (5+7)/2=6
        expected = pd.Series([5.0, 6.0], index=["Sample1", "Sample2"])
        pd.testing.assert_series_equal(result, expected)

    def test_summarize_split_empty_data(self):
        """Test summarization with empty data."""
        empty_data = pd.DataFrame()
        result = SampleLevelGeneDataProcessor._summarize_split(empty_data)

        # Should return empty Series
        assert len(result) == 0

    def test_replace_values_basic(self):
        """Test basic value replacement functionality."""
        test_df = pd.DataFrame({"col1": [0, 1, -1, 2], "col2": [0.5, 0, -0.5, 3]})

        result = SampleLevelGeneDataProcessor._replace_values(
            test_df, threshold_value=0, replace_value=99
        )

        expected = pd.DataFrame(
            {
                "col1": [99, 1, 99, 2],  # 0 and -1 replaced
                "col2": [0.5, 99, 99, 3],  # 0 and -0.5 replaced
            }
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_replace_nans(self):
        """Test NaN replacement."""
        test_df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": [np.nan, 2, 3]})

        result = SampleLevelGeneDataProcessor._replace_nans(test_df, replace_value=99)

        expected = pd.DataFrame({"col1": [1.0, 99.0, 3.0], "col2": [99.0, 2.0, 3.0]})

        pd.testing.assert_frame_equal(result, expected)

    def test_replace_nas(self):
        """Test NA replacement (same as NaN in pandas)."""
        test_df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": [np.nan, 2, 3]})

        result = SampleLevelGeneDataProcessor._replace_nas(test_df, replace_value=88)

        expected = pd.DataFrame({"col1": [1.0, 88.0, 3.0], "col2": [88.0, 2.0, 3.0]})

        pd.testing.assert_frame_equal(result, expected)

    def test_make_valid_names(self):
        """Test name validation functionality."""
        input_names = ["valid_name", "123starts_with_number", "has.dots.and-dashes", ""]

        result = SampleLevelGeneDataProcessor._make_valid_names(input_names)

        expected = ["valid_name", "X123starts_with_number", "has_dots_and_dashes", "X"]
        assert result == expected

    def test_extract_data_from_gem_basic(self):
        """Test basic data extraction from GEM."""
        result = SampleLevelGeneDataProcessor._extract_data_from_gem(
            self.dataset_object, ["GENE1", "GENE2"]
        )

        # Should have 2 genes as rows
        assert len(result) == 2
        assert "GENE1" in result.index
        assert "GENE2" in result.index

        # Should have 4 samples as columns
        assert len(result.columns) == 4

        # GENE1 should be mean of PROBE001 and PROBE003 (which maps to GENE1,GENE3)
        # GENE2 should be mean of PROBE002 and PROBE004
        gene1_sample1 = result.loc["GENE1", "Sample1"]
        expected_gene1_sample1 = (5.2 + 7.8) / 2  # Mean of PROBE001 and PROBE003
        assert abs(gene1_sample1 - expected_gene1_sample1) < 1e-10

    def test_extract_data_from_gem_missing_genes(self):
        """Test data extraction with genes not found in dataset."""
        result = SampleLevelGeneDataProcessor._extract_data_from_gem(
            self.dataset_object, ["GENE1", "MISSING_GENE"]
        )

        # Should still have both genes, with MISSING_GENE as NaN
        assert len(result) == 2
        assert "GENE1" in result.index
        assert "MISSING_GENE" in result.index
        assert pd.isna(result.loc["MISSING_GENE"]).all()

    def test_extract_data_from_gem_no_matching_genes(self):
        """Test data extraction when no genes match."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = SampleLevelGeneDataProcessor._extract_data_from_gem(
                self.dataset_object, ["MISSING1", "MISSING2"]
            )

            # Should issue a warning
            assert len(w) == 1
            assert "No matching genes found" in str(w[0].message)

        # Should return DataFrame with requested genes as NaN
        assert len(result) == 2
        assert all(result.index.isin(["MISSING1", "MISSING2"]))

    def test_get_sample_level_gene_data_full_pipeline(self):
        """Test the complete gene data extraction pipeline."""
        result = SampleLevelGeneDataProcessor.get_sample_level_gene_data(
            self.dataset_object, self.target_genes
        )

        # Should have 3 genes and 4 samples
        assert result.shape == (3, 4)
        assert all(gene in result.index for gene in self.target_genes)

        # Values should be cleaned (no values <= 0, no NaNs)
        assert (result > 0).all().all()  # All values > 0
        assert not result.isna().any().any()  # No NaNs

    def test_get_sample_level_gene_data_with_duplicates(self):
        """Test gene data extraction with duplicate gene names."""
        duplicate_genes = ["GENE1", "GENE2", "GENE1"]  # GENE1 appears twice

        result = SampleLevelGeneDataProcessor.get_sample_level_gene_data(
            self.dataset_object, duplicate_genes
        )

        # Should maintain the duplicate in results
        assert len(result) == 3
        assert result.index.tolist() == ["GENE1", "GENE2", "GENE1"]


class TestConvenienceFunctions:
    """Test suite for convenience functions that match R interface."""

    def setup_method(self):
        """Setup test data."""
        self.expr_data = pd.DataFrame(
            {"Sample1": [1.0, 2.0, 3.0], "Sample2": [1.5, 2.5, 3.5]},
            index=["PROBE1", "PROBE2", "PROBE3"],
        )

        self.keys = ["GENE1", "GENE2", "GENE1"]
        self.dataset_object = {"expr": self.expr_data, "keys": self.keys}

    def test_getSampleLevelGeneData_convenience(self):
        """Test the convenience function that matches R interface."""
        result = getSampleLevelGeneData(self.dataset_object, ["GENE1", "GENE2"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "GENE1" in result.index
        assert "GENE2" in result.index

    def test_extractDataFromGEM_convenience(self):
        """Test the convenience function for data extraction."""
        result = extractDataFromGEM(self.dataset_object, ["GENE1", "GENE2"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_replaceValues_convenience(self):
        """Test the convenience function for value replacement."""
        test_df = pd.DataFrame({"col1": [0, 1, 2], "col2": [-1, 0, 1]})

        result = replaceValues(test_df, threshold_value=0, replace_value=99)

        expected = pd.DataFrame({"col1": [99, 1, 2], "col2": [99, 99, 1]})

        pd.testing.assert_frame_equal(result, expected)

    def test_replaceNaNs_convenience(self):
        """Test the convenience function for NaN replacement."""
        test_df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": [np.nan, 2, 3]})

        result = replaceNaNs(test_df, replace_value=99)

        expected = pd.DataFrame({"col1": [1.0, 99.0, 3.0], "col2": [99.0, 2.0, 3.0]})

        pd.testing.assert_frame_equal(result, expected)

    def test_replaceNAs_convenience(self):
        """Test the convenience function for NA replacement."""
        test_df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": [np.nan, 2, 3]})

        result = replaceNAs(test_df, replace_value=88)

        expected = pd.DataFrame({"col1": [1.0, 88.0, 3.0], "col2": [88.0, 2.0, 3.0]})

        pd.testing.assert_frame_equal(result, expected)


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_empty_expression_matrix(self):
        """Test with empty expression matrix."""
        empty_dataset = {"expr": pd.DataFrame(), "keys": []}

        with pytest.raises(ValueError, match="Expression matrix cannot be empty"):
            getSampleLevelGeneData(empty_dataset, ["GENE1"])

    def test_single_sample_dataset(self):
        """Test with single sample dataset."""
        single_sample = {
            "expr": pd.DataFrame({"Sample1": [1.0, 2.0]}, index=["PROBE1", "PROBE2"]),
            "keys": ["GENE1", "GENE2"],
        }

        result = getSampleLevelGeneData(single_sample, ["GENE1", "GENE2"])

        assert result.shape == (2, 1)
        assert result.columns[0] == "Sample1"

    def test_single_probe_per_gene(self):
        """Test with single probe mapping to each gene."""
        single_probe_dataset = {
            "expr": pd.DataFrame(
                {"Sample1": [1.0, 2.0, 3.0], "Sample2": [1.5, 2.5, 3.5]},
                index=["PROBE1", "PROBE2", "PROBE3"],
            ),
            "keys": ["GENE1", "GENE2", "GENE3"],
        }

        result = getSampleLevelGeneData(single_probe_dataset, ["GENE1", "GENE2", "GENE3"])

        # Should match input exactly (no averaging needed)
        assert result.shape == (3, 2)
        assert result.loc["GENE1", "Sample1"] == 1.0
        assert result.loc["GENE2", "Sample1"] == 2.0
        assert result.loc["GENE3", "Sample1"] == 3.0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
