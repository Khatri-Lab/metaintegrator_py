"""
Comprehensive unit tests for coconut_integration.py
Tests all classes, methods, and functions with edge cases and error conditions.
"""

import os
import sys
import tempfile
import warnings
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.coconut_integration import (
    COCONUTDataProcessor,
    COCONUTIntegrator,
    MetaIntegratorToCOCONUT,
    SampleLevelGeneExtractor,
    coconut_meta_integrator,
    gene_level_meta,
    get_sample_level_gene_data,
    run_coconut_analysis,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_expression_data():
    """Create sample expression data as DataFrame."""
    data = {
        "Sample_1": [10.5, 8.2, 15.3, 6.1],
        "Sample_2": [12.1, 7.8, 14.9, 5.9],
        "Sample_3": [11.3, 8.5, 16.1, 6.3],
        "Sample_4": [9.8, 7.2, 13.7, 5.7],
    }
    return pd.DataFrame(data, index=["probe1", "probe2", "probe3", "probe4"])


@pytest.fixture
def sample_keys():
    """Create sample probe-to-gene mapping."""
    return ["GENE1", "GENE2", "GENE1,GENE3", "GENE2"]


@pytest.fixture
def sample_class_data():
    """Create sample class data."""
    return [0, 0, 1, 1]


@pytest.fixture
def sample_dataset_object(sample_expression_data, sample_keys, sample_class_data):
    """Create a complete sample dataset object."""
    return {
        "expr": sample_expression_data,
        "keys": sample_keys,
        "class": sample_class_data,
    }


@pytest.fixture
def sample_pheno_data():
    """Create sample phenotype data."""
    return pd.DataFrame(
        {"age": [25, 30, 35, 40], "sex": ["M", "F", "M", "F"]},
        index=["Sample_1", "Sample_2", "Sample_3", "Sample_4"],
    )


@pytest.fixture
def sample_meta_object(sample_dataset_object):
    """Create a complete meta object with multiple datasets."""
    dataset2 = sample_dataset_object.copy()
    dataset2["expr"] = sample_dataset_object["expr"] * 1.2  # Slightly different data

    return {"originalData": {"dataset1": sample_dataset_object, "dataset2": dataset2}}


@pytest.fixture
def invalid_meta_object():
    """Create an invalid meta object for error testing."""
    return {"someOtherKey": {"dataset1": {"expr": [], "keys": [], "class": []}}}


@pytest.fixture
def incomplete_dataset_object():
    """Create incomplete dataset object for error testing."""
    return {
        "expr": pd.DataFrame([[1, 2], [3, 4]]),
        # Missing 'keys' and 'class'
    }


# ============================================================================
# COCONUTIntegrator Tests
# ============================================================================


class TestCOCONUTIntegrator:
    """Test cases for COCONUTIntegrator class."""

    def test_init(self):
        """Test COCONUTIntegrator initialization."""
        integrator = COCONUTIntegrator()
        assert integrator is not None

    def test_coconut_meta_integrator_success(self, sample_meta_object):
        """Test coconut_meta_integrator with local coconut package."""
        integrator = COCONUTIntegrator()

        # Since coconut is a local package, we can call it directly
        # The function should convert the data and call coconut.coconut()
        result = integrator.coconut_meta_integrator(sample_meta_object)

        # Verify the result has the expected COCONUT output structure
        assert isinstance(result, dict)
        assert "COCONUTList" in result
        assert "rawDiseaseList" in result
        assert "controlList" in result

    def test_coconut_meta_integrator_names_blocks_after_datasets(self, sample_meta_object):
        """Every COCONUT block carries the name of its dataset (the key in originalData)."""
        result = COCONUTIntegrator().coconut_meta_integrator(sample_meta_object)

        assert [block["name"] for block in result["controlList"]["GSEs"]] == [
            "dataset1",
            "dataset2",
        ]
        assert [block["name"] for block in result["COCONUTList"]] == ["dataset1", "dataset2"]

    def test_gene_level_meta_success(self, sample_meta_object):
        """Test successful gene_level_meta conversion."""
        integrator = COCONUTIntegrator()
        result = integrator.gene_level_meta(sample_meta_object)

        assert isinstance(result, dict)
        assert len(result) == 2  # Two datasets

        for dataset_name in ["dataset1", "dataset2"]:
            assert dataset_name in result
            dataset_result = result[dataset_name]
            assert "pheno" in dataset_result
            assert "genes" in dataset_result
            assert isinstance(dataset_result["pheno"], pd.DataFrame)
            assert isinstance(dataset_result["genes"], pd.DataFrame)
            assert "class" in dataset_result["pheno"].columns

    def test_gene_level_meta_missing_original_data(self, invalid_meta_object):
        """Test gene_level_meta with missing originalData."""
        integrator = COCONUTIntegrator()

        with pytest.raises(ValueError, match="Meta object must contain 'originalData'"):
            integrator.gene_level_meta(invalid_meta_object)

    def test_validate_dataset_object_success(self, sample_dataset_object):
        """Test successful dataset object validation."""
        integrator = COCONUTIntegrator()
        # Should not raise any exception
        integrator._validate_dataset_object(sample_dataset_object)

    def test_validate_dataset_object_missing_keys(self, incomplete_dataset_object):
        """Test dataset object validation with missing keys."""
        integrator = COCONUTIntegrator()

        # The code checks keys in order: expr, class, keys
        # incomplete_dataset_object has 'expr' but missing 'class' and 'keys'
        # So it should fail on 'class' first
        with pytest.raises(ValueError, match="Dataset object missing required key: class"):
            integrator._validate_dataset_object(incomplete_dataset_object)

    def test_create_pheno_dataframe_with_existing_pheno(
        self, sample_dataset_object, sample_pheno_data
    ):
        """Test _create_pheno_dataframe with existing phenotype data."""
        dataset_with_pheno = sample_dataset_object.copy()
        dataset_with_pheno["pheno"] = sample_pheno_data

        integrator = COCONUTIntegrator()
        result = integrator._create_pheno_dataframe(dataset_with_pheno)

        assert isinstance(result, pd.DataFrame)
        assert "class" in result.columns
        assert "age" in result.columns
        assert "sex" in result.columns
        assert len(result) == len(sample_pheno_data)

    def test_create_pheno_dataframe_without_pheno(self, sample_dataset_object):
        """Test _create_pheno_dataframe without existing phenotype data."""
        integrator = COCONUTIntegrator()
        result = integrator._create_pheno_dataframe(sample_dataset_object)

        assert isinstance(result, pd.DataFrame)
        assert "class" in result.columns
        assert len(result) == 4  # Number of samples in expression data

    def test_create_pheno_dataframe_mismatched_dimensions(
        self, sample_dataset_object, sample_pheno_data
    ):
        """Test _create_pheno_dataframe with mismatched dimensions."""
        # Create pheno data with wrong number of samples
        wrong_pheno = sample_pheno_data.iloc[:2]  # Only 2 samples instead of 4
        dataset_with_wrong_pheno = sample_dataset_object.copy()
        dataset_with_wrong_pheno["pheno"] = wrong_pheno

        integrator = COCONUTIntegrator()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = integrator._create_pheno_dataframe(dataset_with_wrong_pheno)

            assert len(w) == 1
            assert "Phenotype data length" in str(w[0].message)
            assert len(result) == 4  # Should create new pheno with correct dimensions

    def test_make_valid_names(self):
        """Test _make_valid_names method."""
        integrator = COCONUTIntegrator()

        test_names = [
            "valid_name",
            "123invalid",
            "invalid-name",
            "invalid name",
            "",
            "valid.name",
        ]
        result = integrator._make_valid_names(test_names)

        assert result[0] == "valid_name"
        assert result[1] == "X123invalid"  # Starts with number
        assert result[2] == "invalid_name"  # Dash replaced with underscore
        assert result[3] == "invalid_name"  # Space replaced with underscore
        assert result[4] == "X"  # Empty string
        assert result[5] == "valid.name"  # Dots are valid

    def test_get_unique_genes(self):
        """Test _get_unique_genes method."""
        integrator = COCONUTIntegrator()

        keys = ["GENE1", "GENE2", "GENE1,GENE3", "GENE2", "", "nan", "NA", None]
        result = integrator._get_unique_genes(keys)

        expected = ["GENE1", "GENE2", "GENE3"]
        assert sorted(result) == expected

    def test_get_unique_genes_empty_input(self):
        """Test _get_unique_genes with empty input."""
        integrator = COCONUTIntegrator()
        result = integrator._get_unique_genes([])
        assert result == []


# ============================================================================
# SampleLevelGeneExtractor Tests
# ============================================================================


class TestSampleLevelGeneExtractor:
    """Test cases for SampleLevelGeneExtractor class."""

    def test_init(self):
        """Test SampleLevelGeneExtractor initialization."""
        extractor = SampleLevelGeneExtractor()
        assert extractor is not None
        assert extractor.DEFAULT_REPLACEMENT_VALUE == 1.0
        assert extractor.DEFAULT_THRESHOLD == 0.0

    def test_get_sample_level_gene_data_success(self, sample_dataset_object):
        """Test successful get_sample_level_gene_data."""
        extractor = SampleLevelGeneExtractor()
        gene_names = ["GENE1", "GENE2", "GENE3"]

        result = extractor.get_sample_level_gene_data(sample_dataset_object, gene_names)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(gene_names)
        assert list(result.index) == gene_names
        assert len(result.columns) == 4  # Number of samples

    def test_prepare_expression_data_dataframe(self, sample_expression_data):
        """Test _prepare_expression_data with DataFrame input."""
        extractor = SampleLevelGeneExtractor()
        result = extractor._prepare_expression_data(sample_expression_data)

        assert isinstance(result, pd.DataFrame)
        assert result.equals(sample_expression_data)

    def test_prepare_expression_data_numpy_array(self):
        """Test _prepare_expression_data with numpy array input."""
        extractor = SampleLevelGeneExtractor()
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = extractor._prepare_expression_data(arr)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 3)

    def test_prepare_expression_data_invalid(self):
        """Test _prepare_expression_data with invalid input."""
        extractor = SampleLevelGeneExtractor()

        with pytest.raises(ValueError, match="Expression data must be a 2D array or DataFrame"):
            extractor._prepare_expression_data("invalid_input")

    def test_extract_data_from_gem_success(self, sample_dataset_object):
        """Test successful _extract_data_from_gem."""
        extractor = SampleLevelGeneExtractor()
        gene_names = ["GENE1", "GENE2"]

        result = extractor._extract_data_from_gem(sample_dataset_object, gene_names)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(gene_names)
        assert set(result.index) == set(gene_names)

    def test_expand_df_sample_level_success(self, sample_expression_data, sample_keys):
        """Test successful _expand_df_sample_level."""
        extractor = SampleLevelGeneExtractor()

        result = extractor._expand_df_sample_level(sample_expression_data, sample_keys)

        assert isinstance(result, pd.DataFrame)
        assert "keys" in result.columns
        assert len(result) > len(sample_expression_data)  # Should expand due to multi-gene keys

    def test_expand_df_sample_level_mismatched_length(self, sample_expression_data):
        """Test _expand_df_sample_level with mismatched key length."""
        extractor = SampleLevelGeneExtractor()
        keys = ["GENE1", "GENE2"]  # Only 2 keys for 4 rows

        with pytest.raises(ValueError, match="Keys length must match dataframe length"):
            extractor._expand_df_sample_level(sample_expression_data, keys)

    def test_expand_df_sample_level_invalid_keys(self, sample_expression_data):
        """Test _expand_df_sample_level with invalid keys."""
        extractor = SampleLevelGeneExtractor()
        keys = ["", "nan", "NA", None]  # All invalid keys

        result = extractor._expand_df_sample_level(sample_expression_data, keys)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0  # Should be empty after filtering invalid keys

    def test_replace_values(self):
        """Test _replace_values method."""
        extractor = SampleLevelGeneExtractor()
        df = pd.DataFrame([[0.5, -1, 2], [0, 3, -0.5]])

        result = extractor._replace_values(df, threshold=0, replace_value=1)

        # Values <= 0 should be replaced with 1
        assert result.loc[0, 0] == 0.5  # 0.5 > 0, should remain unchanged
        assert result.loc[0, 1] == 1  # -1 <= 0, should be replaced
        assert result.loc[0, 2] == 2  # 2 > 0, should remain unchanged
        assert result.loc[1, 0] == 1  # 0 <= 0, should be replaced
        assert result.loc[1, 1] == 3  # 3 > 0, should remain unchanged
        assert result.loc[1, 2] == 1  # -0.5 <= 0, should be replaced

    def test_make_valid_column_names(self):
        """Test _make_valid_column_names method."""
        extractor = SampleLevelGeneExtractor()

        names = ["valid_name", "123invalid", ".456invalid", "invalid-name", "", "."]
        result = extractor._make_valid_column_names(names)

        assert result[0] == "valid_name"
        assert result[1] == "X123invalid"  # Starts with number
        assert result[2] == "X.456invalid"  # Starts with dot and number
        assert result[3] == "invalid.name"  # Dash replaced with dot
        assert result[4] == "X"  # Empty string
        assert result[5] == "X"  # Just a dot

    def test_replace_nans(self):
        """Test _replace_nans method."""
        extractor = SampleLevelGeneExtractor()
        df = pd.DataFrame([[1, np.nan, 3], [np.nan, 5, 6]])

        result = extractor._replace_nans(df, replace_value=999)

        assert not result.isna().any().any()  # No NaNs should remain
        assert result.loc[0, 1] == 999  # NaN replaced with 999
        assert result.loc[1, 0] == 999  # NaN replaced with 999

    def test_replace_nas(self):
        """Test _replace_nas method."""
        extractor = SampleLevelGeneExtractor()
        df = pd.DataFrame([[1, np.nan, 3], [np.nan, 5, 6]])

        result = extractor._replace_nas(df, replace_value=888)

        assert not result.isna().any().any()  # No NaNs should remain
        assert result.loc[0, 1] == 888  # NaN replaced with 888
        assert result.loc[1, 0] == 888  # NaN replaced with 888


# ============================================================================
# COCONUTDataProcessor Tests
# ============================================================================


class TestCOCONUTDataProcessor:
    """Test cases for COCONUTDataProcessor class."""

    def test_prepare_coconut_input_without_filter(self, sample_meta_object):
        """Test prepare_coconut_input without filter results."""
        result = COCONUTDataProcessor.prepare_coconut_input(sample_meta_object)

        assert isinstance(result, dict)
        assert len(result) == 2  # Two datasets
        for dataset_name in ["dataset1", "dataset2"]:
            assert dataset_name in result
            assert "pheno" in result[dataset_name]
            assert "genes" in result[dataset_name]

    def test_prepare_coconut_input_with_filter(self, sample_meta_object):
        """Test prepare_coconut_input with filter results."""
        filter_results = {"posGeneNames": ["GENE1", "GENE2"], "negGeneNames": ["GENE3"]}

        result = COCONUTDataProcessor.prepare_coconut_input(sample_meta_object, filter_results)

        assert isinstance(result, dict)
        assert len(result) == 2  # Two datasets
        # Genes should be filtered to only include specified genes
        for dataset_name in ["dataset1", "dataset2"]:
            genes_df = result[dataset_name]["genes"]
            assert set(genes_df.index).issubset({"GENE1", "GENE2", "GENE3"})

    def test_prepare_coconut_input_empty_filter(self, sample_meta_object):
        """Test prepare_coconut_input with empty filter results."""
        filter_results = {}  # No genes specified

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = COCONUTDataProcessor.prepare_coconut_input(sample_meta_object, filter_results)

            # Empty dict means target_genes will be empty, which triggers warning
            assert len(w) == 1
            assert "No genes found in filter results" in str(w[0].message)
            assert isinstance(result, dict)

    def test_validate_coconut_format_success(self):
        """Test successful validate_coconut_format."""
        valid_data = {
            "dataset1": {
                "pheno": pd.DataFrame({"class": [0, 1, 0, 1]}, index=["S1", "S2", "S3", "S4"]),
                "genes": pd.DataFrame([[1, 2, 3, 4]], columns=["S1", "S2", "S3", "S4"]),
            }
        }

        is_valid, errors = COCONUTDataProcessor.validate_coconut_format(valid_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_coconut_format_invalid_structure(self):
        """Test validate_coconut_format with invalid structure."""
        invalid_data = "not_a_dict"

        is_valid, errors = COCONUTDataProcessor.validate_coconut_format(invalid_data)

        assert is_valid is False
        assert len(errors) > 0
        assert "COCONUT data must be a dictionary" in errors[0]

    def test_validate_coconut_format_missing_keys(self):
        """Test validate_coconut_format with missing required keys."""
        invalid_data = {
            "dataset1": {
                "pheno": pd.DataFrame({"class": [0, 1]})
                # Missing 'genes' key
            }
        }

        is_valid, errors = COCONUTDataProcessor.validate_coconut_format(invalid_data)

        assert is_valid is False
        assert any("missing 'genes' key" in error for error in errors)

    def test_validate_coconut_format_missing_class_column(self):
        """Test validate_coconut_format with missing class column."""
        invalid_data = {
            "dataset1": {
                "pheno": pd.DataFrame({"not_class": [0, 1]}),  # Missing 'class' column
                "genes": pd.DataFrame([[1, 2]]),
            }
        }

        is_valid, errors = COCONUTDataProcessor.validate_coconut_format(invalid_data)

        assert is_valid is False
        assert any("pheno missing 'class' column" in error for error in errors)

    def test_export_for_coconut(self, tmp_path):
        """Test export_for_coconut functionality."""
        coconut_data = {
            "dataset1": {
                "pheno": pd.DataFrame({"class": [0, 1], "age": [25, 30]}),
                "genes": pd.DataFrame([[1, 2], [3, 4]], index=["GENE1", "GENE2"]),
            }
        }

        output_dir = str(tmp_path / "test_output")
        COCONUTDataProcessor.export_for_coconut(coconut_data, output_dir)

        # Check if files were created
        assert os.path.exists(os.path.join(output_dir, "dataset1_pheno.csv"))
        assert os.path.exists(os.path.join(output_dir, "dataset1_genes.csv"))

        # Check file contents
        pheno_df = pd.read_csv(os.path.join(output_dir, "dataset1_pheno.csv"), index_col=0)
        genes_df = pd.read_csv(os.path.join(output_dir, "dataset1_genes.csv"), index_col=0)

        assert "class" in pheno_df.columns
        assert "age" in pheno_df.columns
        assert len(genes_df) == 2  # Two genes

    def test_check_required_packages_success(self):
        """Test check_required_packages when coconut is available."""
        # coconut should be available as a local package
        all_available, missing = COCONUTDataProcessor.check_required_packages()

        # If coconut is properly set up, there should be no missing packages
        assert all_available is True
        assert len(missing) == 0

    def test_check_required_packages_import_error(self):
        """Test check_required_packages when coconut cannot be imported."""
        # Mock an import failure for the coconut package
        with patch("builtins.__import__", side_effect=ImportError("Test error")):
            all_available, missing = COCONUTDataProcessor.check_required_packages()

            assert all_available is False
            assert len(missing) == 1
            assert "coconut" in missing[0]


# ============================================================================
# MetaIntegratorToCOCONUT Tests
# ============================================================================


class TestMetaIntegratorToCOCONUT:
    """Test cases for MetaIntegratorToCOCONUT class."""

    def test_init(self):
        """Test MetaIntegratorToCOCONUT initialization."""
        converter = MetaIntegratorToCOCONUT()
        assert converter.integrator is not None
        assert converter.extractor is not None

    def test_convert_full_pipeline_success(self, sample_meta_object):
        """Test successful convert_full_pipeline."""
        converter = MetaIntegratorToCOCONUT()

        result = converter.convert_full_pipeline(
            sample_meta_object, validate_input=True, validate_output=True
        )

        assert isinstance(result, dict)
        assert len(result) == 2  # Two datasets
        for dataset_name in ["dataset1", "dataset2"]:
            assert dataset_name in result
            assert "pheno" in result[dataset_name]
            assert "genes" in result[dataset_name]

    def test_convert_full_pipeline_invalid_input(self, invalid_meta_object):
        """Test convert_full_pipeline with invalid input."""
        converter = MetaIntegratorToCOCONUT()

        with pytest.raises(ValueError, match="Meta object must contain 'originalData'"):
            converter.convert_full_pipeline(invalid_meta_object, validate_input=True)

    def test_convert_full_pipeline_no_validation(self, sample_meta_object):
        """Test convert_full_pipeline without validation."""
        converter = MetaIntegratorToCOCONUT()

        result = converter.convert_full_pipeline(
            sample_meta_object, validate_input=False, validate_output=False
        )

        assert isinstance(result, dict)

    def test_validate_meta_object_success(self, sample_meta_object):
        """Test successful _validate_meta_object."""
        converter = MetaIntegratorToCOCONUT()
        # Should not raise any exception
        converter._validate_meta_object(sample_meta_object)

    def test_validate_meta_object_missing_original_data(self):
        """Test _validate_meta_object with missing originalData."""
        converter = MetaIntegratorToCOCONUT()
        invalid_object = {"some_other_key": {}}

        with pytest.raises(ValueError, match="Meta object must contain 'originalData'"):
            converter._validate_meta_object(invalid_object)

    def test_validate_meta_object_empty_original_data(self):
        """Test _validate_meta_object with empty originalData."""
        converter = MetaIntegratorToCOCONUT()
        invalid_object = {"originalData": {}}

        with pytest.raises(ValueError, match="originalData cannot be empty"):
            converter._validate_meta_object(invalid_object)

    def test_validate_meta_object_missing_dataset_keys(self, incomplete_dataset_object):
        """Test _validate_meta_object with incomplete dataset."""
        converter = MetaIntegratorToCOCONUT()
        invalid_object = {"originalData": {"dataset1": incomplete_dataset_object}}

        with pytest.raises(ValueError, match="missing required key"):
            converter._validate_meta_object(invalid_object)

    @patch.object(COCONUTIntegrator, "coconut_meta_integrator")
    def test_run_with_coconut(self, mock_method, sample_meta_object):
        """Test run_with_coconut method."""
        mock_result = Mock()
        mock_method.return_value = mock_result

        converter = MetaIntegratorToCOCONUT()
        result = converter.run_with_coconut(sample_meta_object, test_param="test_value")

        mock_method.assert_called_once_with(sample_meta_object, test_param="test_value")
        assert result == mock_result


# ============================================================================
# Convenience Functions Tests
# ============================================================================


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    @patch.object(COCONUTIntegrator, "coconut_meta_integrator")
    def test_coconut_meta_integrator_function(self, mock_method, sample_meta_object):
        """Test coconut_meta_integrator convenience function."""
        mock_result = Mock()
        mock_method.return_value = mock_result

        result = coconut_meta_integrator(sample_meta_object, test_param="test_value")

        mock_method.assert_called_once_with(sample_meta_object, test_param="test_value")
        assert result == mock_result

    @patch.object(COCONUTIntegrator, "gene_level_meta")
    def test_gene_level_meta_function(self, mock_method, sample_meta_object):
        """Test gene_level_meta convenience function."""
        mock_result = Mock()
        mock_method.return_value = mock_result

        result = gene_level_meta(sample_meta_object)

        mock_method.assert_called_once_with(sample_meta_object)
        assert result == mock_result

    @patch.object(SampleLevelGeneExtractor, "get_sample_level_gene_data")
    def test_get_sample_level_gene_data_function(self, mock_method, sample_dataset_object):
        """Test get_sample_level_gene_data convenience function."""
        mock_result = Mock()
        mock_method.return_value = mock_result
        gene_names = ["GENE1", "GENE2"]

        result = get_sample_level_gene_data(sample_dataset_object, gene_names)

        mock_method.assert_called_once_with(sample_dataset_object, gene_names)
        assert result == mock_result

    @patch.object(MetaIntegratorToCOCONUT, "run_with_coconut")
    def test_run_coconut_analysis_function(self, mock_method, sample_meta_object):
        """Test run_coconut_analysis convenience function."""
        mock_result = Mock()
        mock_method.return_value = mock_result

        result = run_coconut_analysis(sample_meta_object, test_param="test_value")

        mock_method.assert_called_once_with(sample_meta_object, test_param="test_value")
        assert result == mock_result


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests that test the complete workflow."""

    def test_full_coconut_workflow(self, sample_meta_object):
        """Test the complete COCONUT workflow from meta object to final result."""
        # Test the full pipeline
        converter = MetaIntegratorToCOCONUT()

        # Convert to COCONUT format
        coconut_data = converter.convert_full_pipeline(sample_meta_object)

        # Validate the format
        is_valid, errors = COCONUTDataProcessor.validate_coconut_format(coconut_data)
        assert is_valid, f"Validation errors: {errors}"

        # Test export functionality
        with tempfile.TemporaryDirectory() as tmp_dir:
            COCONUTDataProcessor.export_for_coconut(coconut_data, tmp_dir)

            # Check files exist
            for dataset_name in ["dataset1", "dataset2"]:
                pheno_file = os.path.join(tmp_dir, f"{dataset_name}_pheno.csv")
                genes_file = os.path.join(tmp_dir, f"{dataset_name}_genes.csv")
                assert os.path.exists(pheno_file)
                assert os.path.exists(genes_file)

    def test_gene_extraction_workflow(self, sample_dataset_object):
        """Test the gene extraction workflow."""
        extractor = SampleLevelGeneExtractor()
        gene_names = ["GENE1", "GENE2", "GENE3"]

        # Extract gene data
        result = extractor.get_sample_level_gene_data(sample_dataset_object, gene_names)

        # Validate result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(gene_names)
        assert set(result.index) == set(gene_names)

        # Check for no NaN values (should be replaced)
        assert not result.isna().any().any()

        # Check for no values <= 0 (should be replaced with 1.0)
        assert (result > 0).all().all()

    def test_error_handling_workflow(self):
        """Test error handling in the complete workflow."""
        # Test with completely invalid input
        converter = MetaIntegratorToCOCONUT()

        with pytest.raises(ValueError):
            converter.convert_full_pipeline({})  # Empty dict

        with pytest.raises(ValueError):
            converter.convert_full_pipeline({"wrong_key": {}})  # Wrong structure


# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_gene_list(self, sample_dataset_object):
        """Test with empty gene list."""
        extractor = SampleLevelGeneExtractor()

        result = extractor.get_sample_level_gene_data(sample_dataset_object, [])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        # Empty gene list should still have column structure
        assert len(result.columns) == 4  # Number of samples

    def test_nonexistent_genes(self, sample_dataset_object):
        """Test with genes that don't exist in the data."""
        extractor = SampleLevelGeneExtractor()
        nonexistent_genes = ["NONEXISTENT1", "NONEXISTENT2"]

        result = extractor.get_sample_level_gene_data(sample_dataset_object, nonexistent_genes)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(nonexistent_genes)
        assert list(result.index) == nonexistent_genes
        assert len(result.columns) == 4  # Number of samples
        # Since nonexistent genes will have NaN values that get replaced with DEFAULT_REPLACEMENT_VALUE (1.0)
        assert (result == 1.0).all().all()

    def test_expression_data_with_all_zeros(self):
        """Test with expression data containing all zeros."""
        dataset_object = {
            "expr": pd.DataFrame([[0, 0, 0], [0, 0, 0]], columns=["S1", "S2", "S3"]),
            "keys": ["GENE1", "GENE2"],
            "class": [0, 1, 0],
        }

        extractor = SampleLevelGeneExtractor()
        result = extractor.get_sample_level_gene_data(dataset_object, ["GENE1", "GENE2"])

        # All zeros should be replaced with 1.0 (default replacement value)
        assert (result == 1.0).all().all()

    def test_expression_data_with_negative_values(self):
        """Test with expression data containing negative values."""
        dataset_object = {
            "expr": pd.DataFrame([[-1, -2, 3], [4, -5, 6]], columns=["S1", "S2", "S3"]),
            "keys": ["GENE1", "GENE2"],
            "class": [0, 1, 0],
        }

        extractor = SampleLevelGeneExtractor()
        result = extractor.get_sample_level_gene_data(dataset_object, ["GENE1", "GENE2"])

        # Negative values should be replaced with 1.0
        assert result.loc["GENE1", "S1"] == 1.0  # -1 replaced
        assert result.loc["GENE1", "S2"] == 1.0  # -2 replaced
        assert result.loc["GENE2", "S2"] == 1.0  # -5 replaced
        # Positive values should remain unchanged
        assert result.loc["GENE1", "S3"] == 3
        assert result.loc["GENE2", "S3"] == 6

    def test_single_sample_dataset(self):
        """Test with a dataset containing only one sample."""
        dataset_object = {
            "expr": pd.DataFrame([[10.5], [8.2]], columns=["Sample_1"], index=["probe1", "probe2"]),
            "keys": ["GENE1", "GENE2"],
            "class": [0],
        }

        extractor = SampleLevelGeneExtractor()
        result = extractor.get_sample_level_gene_data(dataset_object, ["GENE1", "GENE2"])

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 1)  # 2 genes, 1 sample
        assert list(result.columns) == ["Sample_1"]

    def test_unicode_gene_names(self):
        """Test with Unicode gene names."""
        dataset_object = {
            "expr": pd.DataFrame([[1, 2], [3, 4]], columns=["S1", "S2"]),
            "keys": ["GENE_α", "GENE_β"],
            "class": [0, 1],
        }

        extractor = SampleLevelGeneExtractor()
        result = extractor.get_sample_level_gene_data(dataset_object, ["GENE_α", "GENE_β"])

        assert isinstance(result, pd.DataFrame)
        assert "GENE_α" in result.index
        assert "GENE_β" in result.index


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
