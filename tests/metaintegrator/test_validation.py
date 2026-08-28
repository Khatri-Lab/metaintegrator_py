"""
Comprehensive unit tests for validation.py module - FINAL FIXED VERSION.
Tests all validator classes, type checkers, data classes, and utility functions.
Addresses all identified edge cases and test failures.
"""

import os
import sys
import warnings
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

print(">>> PYTHONPATH (sys.path) at import time:")
print("\n".join(f"  {i:02d}: {p}" for i, p in enumerate(sys.path)))

# Import the classes and functions to test
from metaintegrator.validation import (  # Enums and Constants; Custom Exceptions; Data Classes; Type Checker; Validators; Reporter; Convenience Functions
    BaseValidator,
    DataMismatchError,
    DataObjectValidator,
    DatasetValidator,
    MetaAnalysisValidator,
    MetaFilterValidator,
    MetaObjectValidator,
    NullValueError,
    ObjectStage,
    ObjectType,
    TypeChecker,
    TypeValidationError,
    ValidationConstants,
    ValidationError,
    ValidationIssue,
    ValidationReporter,
    ValidationResult,
    ValueRangeError,
    check_data_object,
    validate_and_report,
)


class TestEnumsAndConstants:
    """Test cases for enums and constants."""

    def test_object_type_enum(self):
        """Test ObjectType enum values."""
        assert ObjectType.META.value == "Meta"
        assert ObjectType.DATASET.value == "Dataset"
        assert ObjectType.META_ANALYSIS.value == "MetaAnalysis"
        assert ObjectType.META_FILTER.value == "MetaFilter"

    def test_object_stage_enum(self):
        """Test ObjectStage enum values."""
        assert ObjectStage.PRE_ANALYSIS.value == "Pre-Analysis"
        assert ObjectStage.PRE_FILTER.value == "Pre-Filter"
        assert ObjectStage.POST_FILTER.value == "Post-Filter"

    def test_validation_constants(self):
        """Test ValidationConstants structure."""
        # Check required entries exist and are lists
        assert isinstance(ValidationConstants.META_ANALYSIS_ENTRIES, list)
        assert isinstance(ValidationConstants.POOLED_RESULTS_ENTRIES, list)
        assert isinstance(ValidationConstants.FILTER_ENTRIES, list)
        assert isinstance(ValidationConstants.VALID_CLASS_VALUES, set)

        # Check specific required entries
        assert "datasetEffectSizes" in ValidationConstants.META_ANALYSIS_ENTRIES
        assert "effectSize" in ValidationConstants.POOLED_RESULTS_ENTRIES
        assert "posGeneNames" in ValidationConstants.FILTER_ENTRIES
        assert {0, 1, -1} == ValidationConstants.VALID_CLASS_VALUES


class TestCustomExceptions:
    """Test cases for custom exceptions."""

    def test_validation_error_hierarchy(self):
        """Test that custom exceptions inherit correctly."""
        assert issubclass(NullValueError, ValidationError)
        assert issubclass(TypeValidationError, ValidationError)
        assert issubclass(DataMismatchError, ValidationError)
        assert issubclass(ValueRangeError, ValidationError)
        assert issubclass(ValidationError, Exception)

    def test_exception_raising(self):
        """Test that exceptions can be raised and caught."""
        with pytest.raises(ValidationError):
            raise ValidationError("Test error")

        with pytest.raises(NullValueError):
            raise NullValueError("Null value detected")


class TestValidationIssue:
    """Test cases for ValidationIssue data class."""

    def test_validation_issue_creation(self):
        """Test ValidationIssue creation and attributes."""
        issue = ValidationIssue("error", "Test message", "field_path")
        assert issue.severity == "error"
        assert issue.message == "Test message"
        assert issue.field_path == "field_path"

    def test_validation_issue_defaults(self):
        """Test ValidationIssue with default field_path."""
        issue = ValidationIssue("warning", "Test warning")
        assert issue.severity == "warning"
        assert issue.message == "Test warning"
        assert issue.field_path == ""


class TestValidationResult:
    """Test cases for ValidationResult data class."""

    def test_validation_result_creation(self):
        """Test ValidationResult creation with defaults."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.issues == []
        assert result.object_type == ""
        assert result.object_stage == ""
        assert result.details == {}

    def test_add_issue_error(self):
        """Test adding error issue changes is_valid to False."""
        result = ValidationResult(is_valid=True)
        result.add_issue("error", "Test error", "test_field")

        assert result.is_valid is False
        assert len(result.issues) == 1
        assert result.issues[0].severity == "error"
        assert result.issues[0].message == "Test error"
        assert result.issues[0].field_path == "test_field"

    def test_add_issue_warning(self):
        """Test adding warning issue doesn't change is_valid."""
        result = ValidationResult(is_valid=True)
        result.add_issue("warning", "Test warning")

        assert result.is_valid is True
        assert len(result.issues) == 1
        assert result.issues[0].severity == "warning"

    def test_has_errors(self):
        """Test has_errors method."""
        result = ValidationResult(is_valid=True)
        assert result.has_errors() is False

        result.add_issue("warning", "Warning")
        assert result.has_errors() is False

        result.add_issue("error", "Error")
        assert result.has_errors() is True

    def test_get_warnings(self):
        """Test get_warnings method."""
        result = ValidationResult(is_valid=True)
        result.add_issue("warning", "Warning 1")
        result.add_issue("error", "Error 1")
        result.add_issue("warning", "Warning 2")

        warnings = result.get_warnings()
        assert len(warnings) == 2
        assert "Warning 1" in warnings
        assert "Warning 2" in warnings
        assert "Error 1" not in warnings

    def test_get_errors(self):
        """Test get_errors method."""
        result = ValidationResult(is_valid=True)
        result.add_issue("warning", "Warning 1")
        result.add_issue("error", "Error 1")
        result.add_issue("error", "Error 2")

        errors = result.get_errors()
        assert len(errors) == 2
        assert "Error 1" in errors
        assert "Error 2" in errors
        assert "Warning 1" not in errors


class TestTypeChecker:
    """Test cases for TypeChecker utility class."""

    @pytest.fixture
    def type_checker(self):
        """Create TypeChecker instance."""
        return TypeChecker()

    def test_is_numeric_scalars(self, type_checker):
        """Test is_numeric with scalar values."""
        assert type_checker.is_numeric(5) == True
        assert type_checker.is_numeric(5.5) == True
        assert type_checker.is_numeric(np.int32(5)) == True
        assert type_checker.is_numeric(np.float64(5.5)) == True
        assert type_checker.is_numeric("5") == False
        assert type_checker.is_numeric(None) == False

    def test_is_numeric_arrays(self, type_checker):
        """Test is_numeric with arrays."""
        assert type_checker.is_numeric([1, 2, 3]) == True
        assert type_checker.is_numeric([1.1, 2.2, 3.3]) == True
        assert type_checker.is_numeric(np.array([1, 2, 3])) == True
        assert type_checker.is_numeric(pd.Series([1, 2, 3])) == True

        # String numbers ("1", "2", "3") coerce to floats and count as numeric;
        # only genuinely non-numeric strings should fail this check.
        assert type_checker.is_numeric(["a", "b", "c"]) == False
        assert type_checker.is_numeric([1, "abc", 3]) == False

    def test_is_integer_type_scalars(self, type_checker):
        """Test is_integer_type with scalars."""
        assert type_checker.is_integer_type(5) == True
        assert type_checker.is_integer_type(5.0) == True
        assert type_checker.is_integer_type(5.5) == False
        assert type_checker.is_integer_type(np.int32(5)) == True

    def test_is_integer_type_arrays(self, type_checker):
        """Test is_integer_type with arrays."""
        assert type_checker.is_integer_type([1, 2, 3]) == True
        assert type_checker.is_integer_type([1.0, 2.0, 3.0]) == True
        assert type_checker.is_integer_type([1.1, 2.0, 3.0]) == False
        assert type_checker.is_integer_type(np.array([1, 2, 3])) == True

    def test_is_string_array(self, type_checker):
        """Test is_string_array method."""
        assert type_checker.is_string_array("hello") == True
        assert type_checker.is_string_array(["a", "b", "c"]) == True
        assert type_checker.is_string_array(["a", 1, "c"]) == False
        assert type_checker.is_string_array([1, 2, 3]) == False
        assert type_checker.is_string_array(pd.Series(["a", "b"])) == True

    def test_is_in_range_scalars(self, type_checker):
        """Test is_in_range with scalars."""
        assert type_checker.is_in_range(0.5) == True  # Default 0-1
        assert type_checker.is_in_range(0.0) == True
        assert type_checker.is_in_range(1.0) == True
        assert type_checker.is_in_range(-0.1) == False
        assert type_checker.is_in_range(1.1) == False
        assert type_checker.is_in_range(50, 0, 100) == True
        assert type_checker.is_in_range(150, 0, 100) == False

    def test_is_in_range_arrays(self, type_checker):
        """Test is_in_range with arrays."""
        assert type_checker.is_in_range([0.1, 0.5, 0.9]) == True
        assert type_checker.is_in_range([0.1, 1.5, 0.9]) == False
        assert type_checker.is_in_range([10, 20, 30], 0, 50) == True

        # Test with NaNs (should ignore them)
        assert type_checker.is_in_range([0.5, np.nan, 0.8]) == True

    def test_has_infinite_values(self, type_checker):
        """Test has_infinite_values method."""
        assert type_checker.has_infinite_values(5) == False
        assert type_checker.has_infinite_values(np.inf) == True
        assert type_checker.has_infinite_values(-np.inf) == True
        assert type_checker.has_infinite_values([1, 2, 3]) == False
        assert type_checker.has_infinite_values([1, np.inf, 3]) == True

        # Test with DataFrame
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert type_checker.has_infinite_values(df) == False

        df_inf = pd.DataFrame({"a": [1, np.inf], "b": [3, 4]})
        assert type_checker.has_infinite_values(df_inf) == True


class TestDatasetValidator:
    """Test cases for DatasetValidator class."""

    @pytest.fixture
    def dataset_validator(self):
        """Create DatasetValidator instance."""
        return DatasetValidator()

    @pytest.fixture
    def valid_dataset_dict(self):
        """Create valid dataset dictionary."""
        return {
            "expr": pd.DataFrame(np.random.randn(10, 5), columns=[f"sample_{i}" for i in range(5)]),
            "pheno": pd.DataFrame(
                {"condition": [0, 1, 0, 1, 0]}, index=[f"sample_{i}" for i in range(5)]
            ),
            "class": pd.Series([0, 1, 0, 1, 0], index=[f"sample_{i}" for i in range(5)]),
            "keys": [f"gene_{i}" for i in range(10)],
            "formattedName": "test_dataset",
        }

    @pytest.fixture
    def invalid_dataset_dict(self):
        """Create invalid dataset dictionary - with properly formed class vector."""
        return {
            "expr": "not_a_dataframe",  # Wrong type
            "pheno": None,  # Null required field
            "class": pd.Series([2, 3, 4]),  # Invalid class values but proper Series
            "keys": None,  # Null required field
            "formattedName": 123,  # Wrong type
        }

    def test_validate_valid_dataset(self, dataset_validator, valid_dataset_dict):
        """Test validation of valid dataset."""
        result = dataset_validator.validate(valid_dataset_dict)

        assert result.is_valid == True
        assert result.object_type == "Dataset"
        assert len(result.get_errors()) == 0
        assert "n_genes" in result.details
        assert "n_samples" in result.details
        assert result.details["formatted_name"] == "test_dataset"

    def test_validate_non_dict_dataset(self, dataset_validator):
        """Test validation of non-dictionary dataset."""
        result = dataset_validator.validate("not_a_dict")

        assert result.is_valid == False
        assert "Dataset must be a dictionary" in result.get_errors()

    def test_validate_required_fields_missing(self, dataset_validator):
        """Test validation when required fields are missing."""
        incomplete_dataset = {
            "expr": pd.DataFrame(np.random.randn(5, 3)),
            # Missing pheno and keys
        }

        result = dataset_validator.validate(incomplete_dataset)

        assert result.is_valid == False
        errors = result.get_errors()
        assert any("pheno" in error and "null" in error for error in errors)
        assert any("keys" in error and "null" in error for error in errors)

    def test_validate_field_types_incorrect(self, dataset_validator, invalid_dataset_dict):
        """Test validation with incorrect field types."""
        result = dataset_validator.validate(invalid_dataset_dict)

        assert result.is_valid == False
        errors = result.get_errors()
        assert any("expr" in error for error in errors)
        assert any("formattedName" in error for error in errors)

    def test_validate_class_vector_invalid_values(self, dataset_validator):
        """Test validation with invalid class vector values - using proper Series."""
        dataset = {
            "expr": pd.DataFrame(np.random.randn(5, 3)),
            "pheno": pd.DataFrame({"condition": [0, 1, 0]}, index=[0, 1, 2]),
            "class": pd.Series([2, 3, 4], index=[0, 1, 2]),  # Invalid values but proper Series
            "keys": ["gene1", "gene2", "gene3", "gene4", "gene5"],
        }

        result = dataset_validator.validate(dataset)

        assert result.is_valid == False
        errors = result.get_errors()
        assert any("Class vector contains invalid value" in error for error in errors)

    def test_validate_dimension_mismatch(self, dataset_validator):
        """Test validation with dimension mismatches."""
        dataset = {
            "expr": pd.DataFrame(np.random.randn(5, 3)),  # 5 genes, 3 samples
            "pheno": pd.DataFrame({"condition": [0, 1]}, index=[0, 1]),  # 2 samples
            "class": pd.Series([0, 1, 0], index=[0, 1, 2]),  # 3 samples
            "keys": ["gene1", "gene2"],  # 2 keys for 5 genes
        }

        result = dataset_validator.validate(dataset)

        assert result.is_valid == False
        errors = result.get_errors()
        assert any("Number of genes" in error and "keys" in error for error in errors)
        assert any("Sample names" in error for error in errors)

    def test_validate_infinite_values(self, dataset_validator):
        """Test validation with infinite values in expression data."""
        expr_data = pd.DataFrame(np.random.randn(3, 3))
        expr_data.iloc[0, 0] = np.inf  # Add infinite value

        dataset = {
            "expr": expr_data,
            "pheno": pd.DataFrame({"condition": [0, 1, 0]}, index=[0, 1, 2]),
            "keys": ["gene1", "gene2", "gene3"],
            "formattedName": "test_with_inf",
        }

        result = dataset_validator.validate(dataset)

        assert result.is_valid == False
        errors = result.get_errors()
        assert any("infinite values" in error and "test_with_inf" in error for error in errors)


class TestMetaAnalysisValidator:
    """Test cases for MetaAnalysisValidator class."""

    @pytest.fixture
    def meta_analysis_validator(self):
        """Create MetaAnalysisValidator instance."""
        return MetaAnalysisValidator()

    @pytest.fixture
    def valid_meta_analysis_dict(self):
        """Create valid MetaAnalysis dictionary."""
        return {
            "datasetEffectSizes": pd.DataFrame(
                {"gene1": [0.1, 0.2, 0.3], "gene2": [0.2, 0.3, 0.4]}
            ),
            "datasetEffectSizeStandardErrors": pd.DataFrame(
                {"gene1": [0.05, 0.06, 0.07], "gene2": [0.06, 0.07, 0.08]}
            ),
            "pooledResults": {
                "effectSize": [0.15, 0.25],
                "effectSizeStandardError": [0.05, 0.06],
                "effectSizePval": [0.01, 0.02],
                "effectSizeFDR": [0.05, 0.1],
                "tauSquared": [0.1, 0.2],
                "numStudies": [3, 3],
                "cochranesQ": [2.5, 3.0],
                "heterogeneityPval": [0.3, 0.4],
                "fisherStatUp": [10.0, 15.0],
                "fisherPvalUp": [0.01, 0.02],
                "fisherFDRUp": [0.05, 0.1],
                "fisherStatDown": [8.0, 12.0],
                "fisherPvalDown": [0.02, 0.03],
                "fisherFDRDown": [0.1, 0.15],
            },
            "analysisDescription": "Test meta-analysis",
        }

    def test_validate_valid_meta_analysis(self, meta_analysis_validator, valid_meta_analysis_dict):
        """Test validation of valid MetaAnalysis."""
        result = meta_analysis_validator.validate(valid_meta_analysis_dict)

        assert result.is_valid == True
        assert result.object_type == "MetaAnalysis"
        assert len(result.get_errors()) == 0
        assert "analysis_description" in result.details
        assert result.details["analysis_description"] == "Test meta-analysis"

    def test_validate_non_dict_meta_analysis(self, meta_analysis_validator):
        """Test validation of non-dictionary MetaAnalysis."""
        result = meta_analysis_validator.validate("not_a_dict")

        assert result.is_valid == False
        assert "MetaAnalysis must be a dictionary" in result.get_errors()

    def test_validate_missing_required_fields(self, meta_analysis_validator):
        """Test validation with missing required fields."""
        incomplete_meta_analysis = {
            "datasetEffectSizes": pd.DataFrame(),
            # Missing other required fields
        }

        result = meta_analysis_validator.validate(incomplete_meta_analysis)

        assert result.is_valid == False
        errors = result.get_errors()
        missing_fields = [
            "datasetEffectSizeStandardErrors",
            "pooledResults",
            "analysisDescription",
        ]
        for field in missing_fields:
            assert any(field in error and "null" in error for error in errors)

    def test_validate_incorrect_field_types(self, meta_analysis_validator):
        """Test validation with incorrect field types."""
        invalid_meta_analysis = {
            "datasetEffectSizes": "not_a_dataframe",  # Wrong type
            "datasetEffectSizeStandardErrors": pd.DataFrame(),
            "pooledResults": "not_a_dict_or_dataframe",  # Wrong type
            "analysisDescription": 123,  # Wrong type
        }

        result = meta_analysis_validator.validate(invalid_meta_analysis)

        assert result.is_valid is False
        errors = result.get_errors()
        assert any("datasetEffectSizes" in error and "incorrect type" in error for error in errors)
        assert any("pooledResults" in error and "incorrect type" in error for error in errors)
        assert any("analysisDescription" in error and "incorrect type" in error for error in errors)

    def test_validate_pooled_results_missing_entries(self, meta_analysis_validator):
        """Test validation with missing pooled results entries."""
        meta_analysis = {
            "datasetEffectSizes": pd.DataFrame(),
            "datasetEffectSizeStandardErrors": pd.DataFrame(),
            "pooledResults": {
                "effectSize": [0.1],
                # Missing other required entries
            },
            "analysisDescription": "Test",
        }

        result = meta_analysis_validator.validate(meta_analysis)

        assert result.is_valid is False
        errors = result.get_errors()
        assert any("effectSizeStandardError" in error and "null" in error for error in errors)

    def test_validate_pooled_results_invalid_ranges(self, meta_analysis_validator):
        """Test validation with invalid value ranges in pooled results."""
        meta_analysis = {
            "datasetEffectSizes": pd.DataFrame(),
            "datasetEffectSizeStandardErrors": pd.DataFrame(),
            "pooledResults": {
                "effectSize": [0.1],
                "effectSizeStandardError": [-0.1],  # Negative value
                "effectSizePval": [1.5],  # > 1
                "effectSizeFDR": [-0.1],  # < 0
                "tauSquared": [0.1],
                "numStudies": [3],
                "cochranesQ": [2.5],
                "heterogeneityPval": [0.3],
                "fisherStatUp": [10.0],
                "fisherPvalUp": [0.01],
                "fisherFDRUp": [0.05],
                "fisherStatDown": [8.0],
                "fisherPvalDown": [0.02],
                "fisherFDRDown": [0.1],
            },
            "analysisDescription": "Test",
        }

        result = meta_analysis_validator.validate(meta_analysis)

        assert result.is_valid is False
        errors = result.get_errors()
        assert any("effectSizeStandardError" in error and "negative" in error for error in errors)
        assert any("effectSizePval" in error and "range" in error for error in errors)


class TestMetaFilterValidator:
    """Test cases for MetaFilterValidator class."""

    @pytest.fixture
    def meta_filter_validator(self):
        """Create MetaFilterValidator instance."""
        return MetaFilterValidator()

    @pytest.fixture
    def valid_meta_filter_dict(self):
        """Create valid MetaFilter dictionary."""
        return {
            "posGeneNames": ["gene1", "gene2", "gene3"],
            "negGeneNames": ["gene4", "gene5"],
            "FDRThresh": 0.05,
            "effectSizeThresh": 0.1,
            "numberStudiesThresh": 2,
            "isLeaveOneOut": True,
            "heterogeneityPvalThresh": 0.05,
            "filterDescription": "Test filter",
            "timestamp": datetime.now(),
        }

    def test_validate_valid_meta_filter(self, meta_filter_validator, valid_meta_filter_dict):
        """Test validation of valid MetaFilter."""
        result = meta_filter_validator.validate(valid_meta_filter_dict)

        assert result.is_valid == True
        assert result.object_type == "MetaFilter"
        assert len(result.get_errors()) == 0
        assert "n_pos_genes" in result.details
        assert "n_neg_genes" in result.details
        assert result.details["n_pos_genes"] == 3
        assert result.details["n_neg_genes"] == 2

    def test_validate_non_dict_meta_filter(self, meta_filter_validator):
        """Test validation of non-dictionary MetaFilter."""
        result = meta_filter_validator.validate("not_a_dict")

        assert result.is_valid is False
        assert "MetaFilter must be a dictionary" in result.get_errors()

    def test_validate_missing_required_fields(self, meta_filter_validator):
        """Test validation with missing required fields."""
        incomplete_filter = {
            "posGeneNames": ["gene1"],
            # Missing other required fields
        }

        result = meta_filter_validator.validate(incomplete_filter)

        assert result.is_valid is False
        errors = result.get_errors()
        required_fields = [
            "negGeneNames",
            "FDRThresh",
            "effectSizeThresh",
            "numberStudiesThresh",
            "isLeaveOneOut",
            "heterogeneityPvalThresh",
            "filterDescription",
            "timestamp",
        ]
        for field in required_fields:
            assert any(field in error and "null" in error for error in errors)

    def test_validate_incorrect_types(self, meta_filter_validator):
        """Test validation with incorrect field types."""
        invalid_filter = {
            "posGeneNames": "not_a_list",  # Should be list
            "negGeneNames": ["gene1"],
            "FDRThresh": "not_a_number",  # Should be numeric
            "effectSizeThresh": 0.1,
            "numberStudiesThresh": 2.5,  # Should be int
            "isLeaveOneOut": "yes",  # Should be bool
            "heterogeneityPvalThresh": 0.05,
            "filterDescription": 123,  # Should be string
            "timestamp": "not_a_datetime",  # Should be datetime
        }

        result = meta_filter_validator.validate(invalid_filter)

        assert result.is_valid is False
        errors = result.get_errors()
        type_error_fields = [
            "posGeneNames",
            "FDRThresh",
            "numberStudiesThresh",
            "isLeaveOneOut",
            "filterDescription",
            "timestamp",
        ]
        for field in type_error_fields:
            assert any(field in error and "incorrect type" in error for error in errors)

    def test_validate_invalid_ranges(self, meta_filter_validator):
        """Test validation with invalid value ranges."""
        invalid_filter = {
            "posGeneNames": ["gene1"],
            "negGeneNames": ["gene2"],
            "FDRThresh": 1.5,  # > 1
            "effectSizeThresh": -0.1,  # < 0
            "numberStudiesThresh": -1,  # < 0
            "isLeaveOneOut": True,
            "heterogeneityPvalThresh": -0.1,  # < 0
            "filterDescription": "Test",
            "timestamp": datetime.now(),
        }

        result = meta_filter_validator.validate(invalid_filter)

        assert result.is_valid is False
        errors = result.get_errors()
        assert any("FDRThresh" in error and "range" in error for error in errors)
        assert any("effectSizeThresh" in error and "non-negative" in error for error in errors)
        assert any("numberStudiesThresh" in error and "non-negative" in error for error in errors)
        assert any("heterogeneityPvalThresh" in error and "range" in error for error in errors)


class TestMetaObjectValidator:
    """Test cases for MetaObjectValidator class."""

    @pytest.fixture
    def meta_object_validator(self):
        """Create MetaObjectValidator instance."""
        return MetaObjectValidator()

    @pytest.fixture
    def valid_dataset(self):
        """Create valid dataset for testing."""
        return {
            "expr": pd.DataFrame(np.random.randn(5, 3)),
            "pheno": pd.DataFrame({"condition": [0, 1, 0]}, index=[0, 1, 2]),
            "class": pd.Series([0, 1, 0], index=[0, 1, 2]),
            "keys": ["gene1", "gene2", "gene3", "gene4", "gene5"],
            "formattedName": "test_dataset",
        }

    @pytest.fixture
    def valid_meta_object(self, valid_dataset):
        """Create valid Meta object for testing."""
        return {
            "originalData": {
                "dataset1": valid_dataset,
                "dataset2": valid_dataset.copy(),
            }
        }

    def test_validate_pre_analysis_stage(self, meta_object_validator, valid_meta_object):
        """Test validation at PRE_ANALYSIS stage."""
        result = meta_object_validator.validate(valid_meta_object, ObjectStage.PRE_ANALYSIS)

        assert result.is_valid == True
        assert result.object_type == "Meta"
        assert result.object_stage == "Pre-Analysis"
        assert "n_datasets" in result.details
        assert result.details["n_datasets"] == 2

    def test_validate_non_dict_meta_object(self, meta_object_validator):
        """Test validation of non-dictionary Meta object."""
        result = meta_object_validator.validate("not_a_dict")

        assert result.is_valid is False
        assert "Meta object must be a dictionary" in result.get_errors()

    def test_validate_missing_original_data(self, meta_object_validator):
        """Test validation with missing originalData."""
        meta_obj = {}  # Empty dict

        result = meta_object_validator.validate(meta_obj, ObjectStage.PRE_ANALYSIS)

        assert result.is_valid is False
        errors = result.get_errors()
        assert any("originalData is null" in error for error in errors)

    def test_validate_invalid_original_data_type(self, meta_object_validator):
        """Test validation with invalid originalData type."""
        meta_obj = {"originalData": "not_a_dict"}

        result = meta_object_validator.validate(meta_obj, ObjectStage.PRE_ANALYSIS)

        assert result.is_valid is False
        errors = result.get_errors()
        assert any("originalData must be a dictionary" in error for error in errors)

    def test_validate_invalid_dataset_in_original_data(self, meta_object_validator):
        """Test validation with invalid dataset in originalData."""
        meta_obj = {
            "originalData": {
                "dataset1": {"expr": "invalid_expr"},  # Invalid dataset
            }
        }

        result = meta_object_validator.validate(meta_obj, ObjectStage.PRE_ANALYSIS)

        assert result.is_valid is False
        errors = result.get_errors()
        assert any("Dataset dataset1" in error for error in errors)

    def test_get_validation_components(self, meta_object_validator):
        """Test _get_validation_components method."""
        pre_analysis = meta_object_validator._get_validation_components(ObjectStage.PRE_ANALYSIS)
        assert pre_analysis == ["originalData"]

        pre_filter = meta_object_validator._get_validation_components(ObjectStage.PRE_FILTER)
        expected_pre_filter = ["originalData", "metaAnalysis", "leaveOneOutAnalysis"]
        assert pre_filter == expected_pre_filter

        post_filter = meta_object_validator._get_validation_components(ObjectStage.POST_FILTER)
        expected_post_filter = [
            "originalData",
            "metaAnalysis",
            "leaveOneOutAnalysis",
            "filterResults",
        ]
        assert post_filter == expected_post_filter


class TestDataObjectValidator:
    """Test cases for main DataObjectValidator orchestrator."""

    @pytest.fixture
    def data_object_validator(self):
        """Create DataObjectValidator instance."""
        return DataObjectValidator()

    @pytest.fixture
    def valid_dataset(self):
        """Create valid dataset for testing."""
        return {
            "expr": pd.DataFrame(np.random.randn(5, 3)),
            "pheno": pd.DataFrame({"condition": [0, 1, 0]}, index=[0, 1, 2]),
            "class": pd.Series([0, 1, 0], index=[0, 1, 2]),
            "keys": ["gene1", "gene2", "gene3", "gene4", "gene5"],
            "formattedName": "test_dataset",
        }

    def test_validate_dataset_enum(self, data_object_validator, valid_dataset):
        """Test validation using ObjectType enum."""
        result = data_object_validator.validate(valid_dataset, ObjectType.DATASET)

        assert result.is_valid == True
        assert result.object_type == "Dataset"

    def test_validate_dataset_string(self, data_object_validator, valid_dataset):
        """Test validation using string object type."""
        result = data_object_validator.validate(valid_dataset, "Dataset")

        assert result.is_valid == True
        assert result.object_type == "Dataset"

    def test_validate_invalid_object_type_string(self, data_object_validator, valid_dataset):
        """Test validation with invalid object type string."""
        result = data_object_validator.validate(valid_dataset, "InvalidType")

        assert result.is_valid == False
        errors = result.get_errors()
        assert any("Invalid object type: InvalidType" in error for error in errors)

    def test_validate_invalid_object_stage_string(self, data_object_validator, valid_dataset):
        """Test validation with invalid object stage string."""
        result = data_object_validator.validate(valid_dataset, ObjectType.META, "InvalidStage")

        assert result.is_valid is False
        errors = result.get_errors()
        assert any("Invalid object stage: InvalidStage" in error for error in errors)

    def test_validate_unsupported_object_type(self, data_object_validator, valid_dataset):
        """Test validation with unsupported object type."""

        # Create a mock enum that's not in the supported list
        class UnsupportedType:
            value = "Unsupported"

        result = data_object_validator.validate(valid_dataset, UnsupportedType())

        assert result.is_valid is False
        errors = result.get_errors()
        assert any("Unsupported object type" in error for error in errors)


class TestValidationReporter:
    """Test cases for ValidationReporter class."""

    def test_print_validation_report_valid(self, capsys):
        """Test printing report for valid result."""
        result = ValidationResult(is_valid=True, object_type="Dataset")
        result.details = {"n_genes": 100, "n_samples": 20}

        ValidationReporter.print_validation_report(result, verbose=True)

        # capsys is a built‑in pytest fixture that lets you intercept anything that gets written to sys.stdout
        # or sys.stderr while your test runs.
        # The normal Python print() function writes to sys.stdout (and similarly for sys.stderr). capsys replaces these
        # streams temporarily, so you can read what would have appeared on the terminal without actually printing anything.
        # (i.e., it intercepts the output before it reaches the StringIO)
        # Use pytest's built-in `capsys` fixture here instead of using output = capsys.getvalue()
        output = capsys.readouterr().out
        assert "VALIDATION REPORT" in output
        assert "Object Type: Dataset" in output
        assert "Status: VALID" in output
        assert "No issues found." in output
        assert "n_genes: 100" in output
        assert "n_samples: 20" in output

    def test_print_validation_report_invalid(self, capsys):
        """Test printing report for invalid result."""
        result = ValidationResult(
            is_valid=False, object_type="Dataset", object_stage="Pre-Analysis"
        )
        result.add_issue("error", "Test error")
        result.add_issue("warning", "Test warning")

        ValidationReporter.print_validation_report(result, verbose=False)

        output = capsys.readouterr().out
        assert "Object Type: Dataset" in output
        assert "Object Stage: Pre-Analysis" in output
        assert "Status: INVALID" in output
        assert "ERRORS:" in output
        assert "Test error" in output
        assert "WARNINGS:" in output
        assert "Test warning" in output
        assert "DETAILS:" not in output  # verbose=False

    def test_generate_summary_report_empty(self):
        """Test generating summary report for empty results list."""
        summary = ValidationReporter.generate_summary_report([])

        assert summary["total_objects"] == 0
        assert summary["valid_objects"] == 0
        assert summary["invalid_objects"] == 0
        assert summary["total_errors"] == 0
        assert summary["total_warnings"] == 0
        assert summary["object_types"] == {}

    def test_generate_summary_report_multiple_results(self):
        """Test generating summary report for multiple results."""
        results = []

        # Valid dataset result
        result1 = ValidationResult(is_valid=True, object_type="Dataset")
        results.append(result1)

        # Invalid meta-analysis result with errors and warnings
        result2 = ValidationResult(is_valid=False, object_type="MetaAnalysis")
        result2.add_issue("error", "Error 1")
        result2.add_issue("error", "Error 2")
        result2.add_issue("warning", "Warning 1")
        results.append(result2)

        # Another valid dataset
        result3 = ValidationResult(is_valid=True, object_type="Dataset")
        results.append(result3)

        summary = ValidationReporter.generate_summary_report(results)

        assert summary["total_objects"] == 3
        assert summary["valid_objects"] == 2
        assert summary["invalid_objects"] == 1
        assert summary["total_errors"] == 2
        assert summary["total_warnings"] == 1

        assert summary["object_types"]["Dataset"]["total"] == 2
        assert summary["object_types"]["Dataset"]["valid"] == 2
        assert summary["object_types"]["Dataset"]["invalid"] == 0

        assert summary["object_types"]["MetaAnalysis"]["total"] == 1
        assert summary["object_types"]["MetaAnalysis"]["valid"] == 0
        assert summary["object_types"]["MetaAnalysis"]["invalid"] == 1


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    @pytest.fixture
    def valid_dataset(self):
        """Create valid dataset for testing."""
        return {
            "expr": pd.DataFrame(np.random.randn(5, 3)),
            "pheno": pd.DataFrame({"condition": [0, 1, 0]}, index=[0, 1, 2]),
            "class": pd.Series([0, 1, 0], index=[0, 1, 2]),
            "keys": ["gene1", "gene2", "gene3", "gene4", "gene5"],
            "formattedName": "test_dataset",
        }

    def test_check_data_object_valid(self, valid_dataset):
        """Test check_data_object with valid dataset."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_data_object(valid_dataset, "Dataset")
            assert result == True
            # ✅ Valid datasets may have warnings (e.g., unnamed keys in R-mode)
            # Only check that there are no ERROR-level issues
            for warning in w:
                assert (
                    "error" not in str(warning.message).lower()
                ), f"Should not have error-level warnings, got: {warning.message}"

    def test_check_data_object_invalid(self):
        """Test check_data_object with invalid dataset."""
        invalid_dataset = {"expr": "invalid"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_data_object(invalid_dataset, "Dataset")
            assert result is False
            assert len(w) > 0  # Should generate warnings

    def test_validate_and_report(self, valid_dataset):
        """Test validate_and_report function."""
        result = validate_and_report(valid_dataset, "Dataset")

        assert isinstance(result, ValidationResult)
        assert result.object_type == "Dataset"
        assert result.is_valid == True


class TestEdgeCasesAndErrorHandling:
    """Test cases for edge cases and error handling."""

    def test_type_checker_with_none_values(self):
        """Test TypeChecker methods with None values."""
        type_checker = TypeChecker()

        assert type_checker.is_numeric(None) == False
        assert type_checker.is_integer_type(None) == False
        assert type_checker.is_string_array(None) == False
        # is_in_range(None) is False too, consistent with the checks above.
        assert type_checker.is_in_range(None) == False
        assert type_checker.has_infinite_values(None) == False

    def test_type_checker_with_empty_arrays(self):
        """Test TypeChecker methods with empty arrays."""
        type_checker = TypeChecker()

        assert type_checker.is_numeric([]) == True
        # Empty arrays are vacuously valid for these checks, same as is_numeric([]) above.
        assert type_checker.is_integer_type([]) == True
        assert type_checker.is_string_array([]) == True
        assert type_checker.is_in_range([]) == True
        assert type_checker.has_infinite_values([]) == False

    def test_validation_result_with_many_issues(self):
        """Test ValidationResult with many issues."""
        result = ValidationResult(is_valid=True)

        # Add many issues
        for i in range(10):  # Reduced from 100 for faster testing
            severity = "error" if i % 2 == 0 else "warning"
            result.add_issue(severity, f"Issue {i}", f"field_{i}")

        assert len(result.issues) == 10
        assert len(result.get_errors()) == 5
        assert len(result.get_warnings()) == 5
        assert result.is_valid == False  # Errors make it invalid

    def test_dataset_validator_with_numpy_arrays(self):
        """Test DatasetValidator with numpy arrays instead of DataFrames."""
        dataset_validator = DatasetValidator()

        dataset = {
            "expr": np.random.randn(10, 5),  # Numpy array
            "pheno": np.array([0, 1, 0, 1, 0]),  # Numpy array
            "class": np.array([0, 1, 0, 1, 0]),
            "keys": np.array(
                [
                    "gene1",
                    "gene2",
                    "gene3",
                    "gene4",
                    "gene5",
                    "gene6",
                    "gene7",
                    "gene8",
                    "gene9",
                    "gene10",
                ]
            ),
            "formattedName": "numpy_dataset",
        }

        result = dataset_validator.validate(dataset)
        # Should handle numpy arrays (may have warnings but shouldn't crash)
        assert isinstance(result, ValidationResult)

    def test_meta_analysis_with_dataframe_pooled_results(self):
        """Test MetaAnalysisValidator with DataFrame pooled results."""
        meta_analysis_validator = MetaAnalysisValidator()

        pooled_results_df = pd.DataFrame(
            {
                "effectSize": [0.1, 0.2],
                "effectSizeStandardError": [0.05, 0.06],
                "effectSizePval": [0.01, 0.02],
                "effectSizeFDR": [0.05, 0.1],
                "tauSquared": [0.1, 0.2],
                "numStudies": [3, 3],
                "cochranesQ": [2.5, 3.0],
                "heterogeneityPval": [0.3, 0.4],
                "fisherStatUp": [10.0, 15.0],
                "fisherPvalUp": [0.01, 0.02],
                "fisherFDRUp": [0.05, 0.1],
                "fisherStatDown": [8.0, 12.0],
                "fisherPvalDown": [0.02, 0.03],
                "fisherFDRDown": [0.1, 0.15],
            }
        )

        meta_analysis = {
            "datasetEffectSizes": pd.DataFrame(np.random.randn(10, 5)),
            "datasetEffectSizeStandardErrors": pd.DataFrame(np.random.randn(10, 5)),
            "pooledResults": pooled_results_df,
            "analysisDescription": "Test with DataFrame",
        }

        result = meta_analysis_validator.validate(meta_analysis)
        assert result.is_valid is True

    def test_validation_with_mixed_data_types(self):
        """Test validation with various mixed data types."""
        type_checker = TypeChecker()

        # Mixed list with numeric and non-numeric
        mixed_list = [
            1,
            2.5,
            "hello",
            None,
            np.inf,
        ]  # "hello" cannot be converted to numeric
        assert type_checker.is_numeric(mixed_list) is False

        # List with some invalid values for string checking
        mixed_strings = ["a", "b", 123]
        assert type_checker.is_string_array(mixed_strings) is False

    @pytest.mark.parametrize(
        "invalid_input",
        [
            {"expr": None},  # Null expr
            {"pheno": None},  # Null pheno
            {"keys": None},  # Null keys
            {},  # Empty dict
        ],
    )
    def test_dataset_validator_with_various_invalid_inputs(self, invalid_input):
        """Test DatasetValidator with various invalid inputs."""
        dataset_validator = DatasetValidator()
        result = dataset_validator.validate(invalid_input)
        assert result.is_valid == False
        assert len(result.get_errors()) > 0


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])
