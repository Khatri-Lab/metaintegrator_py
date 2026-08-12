"""
Refactored validation functions for MetaIntegrator Python port.
This module contains functions to check for errors in objects used for analysis.

Key improvements:
- Composition over inheritance
- Better separation of concerns
- Structured error handling with custom exceptions
- More comprehensive type hints
- Reduced code duplication
- Better testability

Author: Refactored from original R translation by Erika Bongen
"""

import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Union, Any, Protocol, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Enums and Constants
# ------------------------------------------------------------------
class ObjectType(Enum):
    """Valid object types for validation.

    The enum members represent the distinct categories of objects that can be
    validated within the meta‑integrator framework.
    """

    META = "Meta"
    DATASET = "Dataset"
    META_ANALYSIS = "MetaAnalysis"
    META_FILTER = "MetaFilter"


class ObjectStage(Enum):
    """Valid object stages for Meta objects.

    Each member indicates a particular lifecycle stage that a Meta object may
    occupy during processing.
    """

    PRE_ANALYSIS = "Pre-Analysis"
    PRE_FILTER = "Pre-Filter"
    POST_FILTER = "Post-Filter"


class ValidationConstants:
    """Constants used throughout validation.

    Attributes:
        META_ANALYSIS_ENTRIES (list[str]): Keys required in a MetaAnalysis
            dictionary.
        POOLED_RESULTS_ENTRIES (list[str]): Keys required in a pooledResults
            dictionary.
        FILTER_ENTRIES (list[str]): Keys required in a MetaFilter dictionary.
        VALID_CLASS_VALUES (set[int]): Integer values that are considered
            valid for a class field.
    """

    # MetaAnalysis required entries
    META_ANALYSIS_ENTRIES = [
        "datasetEffectSizes",
        "datasetEffectSizeStandardErrors",
        "pooledResults",
        "analysisDescription",
    ]

    # PooledResults required entries
    POOLED_RESULTS_ENTRIES = [
        "effectSize",
        "effectSizeStandardError",
        "effectSizePval",
        "effectSizeFDR",
        "tauSquared",
        "numStudies",
        "cochranesQ",
        "heterogeneityPval",
        "fisherStatUp",
        "fisherPvalUp",
        "fisherFDRUp",
        "fisherStatDown",
        "fisherPvalDown",
        "fisherFDRDown",
    ]

    # MetaFilter required entries
    FILTER_ENTRIES = [
        "posGeneNames",
        "negGeneNames",
        "FDRThresh",
        "effectSizeThresh",
        "numberStudiesThresh",
        "isLeaveOneOut",
        "heterogeneityPvalThresh",
        "filterDescription",
        "timestamp",
    ]

    # Valid class values
    VALID_CLASS_VALUES = {0, 1, -1}


# ------------------------------------------------------------------
# Custom Exceptions
# ------------------------------------------------------------------
class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class NullValueError(ValidationError):
    """Raised when required values are null."""

    pass


class TypeValidationError(ValidationError):
    """Raised when objects are of incorrect type."""

    pass


class DataMismatchError(ValidationError):
    """Raised when data dimensions or names don't match."""

    pass


class ValueRangeError(ValidationError):
    """Raised when values are outside expected range."""

    pass


# ------------------------------------------------------------------
# Data Classes for Validation Results
# ------------------------------------------------------------------
@dataclass
class ValidationIssue:
    """Represents a single validation issue.

    Attributes:
        severity (str): The level of the issue, e.g. 'error', 'warning', or 'info'.
        message (str): Human‑readable description of the issue.
        field_path (str): Optional path to the field that caused the issue.
    """

    severity: str  # 'error', 'warning', 'info'
    message: str
    field_path: str = ""


@dataclass
class ValidationResult:
    """Complete validation result with all issues.

    Attributes:
        is_valid (bool): True if no errors were found; otherwise False.
        issues (List[ValidationIssue]): Collection of all validation issues.
        object_type (str): Type of the object that was validated.
        object_stage (str): Stage of the object during validation.
        details (Dict[str, Any]): Additional metadata about the validation run.
    """

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    object_type: str = ""
    object_stage: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, severity: str, message: str, field_path: str = "") -> None:
        """Add a validation issue.

        Args:
            severity (str): Severity level of the issue.
            message (str): Description of the issue.
            field_path (str): Optional path to the problematic field.
        """
        self.issues.append(ValidationIssue(severity, message, field_path))
        if severity == "error":
            self.is_valid = False

    def has_errors(self) -> bool:
        """Check if result has any errors.

        Returns:
            bool: True if any issue has severity 'error', otherwise False.
        """
        return any(issue.severity == "error" for issue in self.issues)

    def get_warnings(self) -> List[str]:
        """Get list of warning messages.

        Returns:
            List[str]: Messages of all issues with severity 'warning'.
        """
        return [issue.message for issue in self.issues if issue.severity == "warning"]

    def get_errors(self) -> List[str]:
        """Get list of error messages.

        Returns:
            List[str]: Messages of all issues with severity 'error'.
        """
        return [issue.message for issue in self.issues if issue.severity == "error"]


# ------------------------------------------------------------------
# Type Checking Utilities
# ------------------------------------------------------------------
class TypeChecker:
    """Utility class for checking object types and ranges."""

    @staticmethod
    def is_numeric(obj: Any) -> bool:
        """Check if object is numeric.

        Args:
            obj (Any): Object to test.

        Returns:
            bool: True if obj is a scalar numeric type or a collection that can be
            converted to a numeric array; otherwise False.
        """
        if isinstance(obj, (int, float, np.integer, np.floating)):
            return True
        if isinstance(obj, (list, np.ndarray, pd.Series)):
            try:
                np.array(obj, dtype=float)
                return True
            except (ValueError, TypeError):
                return False
        return False

    @staticmethod
    def is_integer_type(obj: Any) -> bool:
        """Check if numeric object contains only integers.

        Args:
            obj (Any): Object to test.

        Returns:
            bool: True if obj is an integer scalar or a collection that
            contains only integer values; otherwise False.
        """
        if isinstance(obj, (int, np.integer)):
            return True
        if isinstance(obj, (list, np.ndarray, pd.Series)):
            try:
                arr = np.array(obj, dtype=float)
                return np.all(arr == np.floor(arr))
            except (ValueError, TypeError):
                return False
        if isinstance(obj, float):
            return obj == int(obj)
        return False

    @staticmethod
    def is_string_array(obj: Any) -> bool:
        """Check if object is a string or array of strings.

        Args:
            obj (Any): Object to test.

        Returns:
            bool: True if obj is a string scalar or a collection that
            contains only string elements; otherwise False.
        """
        if isinstance(obj, str):
            return True
        if isinstance(obj, (list, np.ndarray, pd.Series)):
            try:
                return all(isinstance(x, str) for x in obj)
            except Exception:
                return False
        return False

    @staticmethod
    def is_in_range(obj: Any, min_val: float = 0, max_val: float = 1) -> bool:
        """Check if numeric object is in specified range.

        Args:
            obj (Any): Numeric scalar or collection to test.
            min_val (float): Minimum allowable value (inclusive). Defaults to 0.
            max_val (float): Maximum allowable value (inclusive). Defaults to 1.

        Returns:
            bool: True if every numeric element in obj lies within
            [min_val, max_val] (ignoring NaNs); otherwise False.
        """
        try:
            if isinstance(obj, (int, float)):
                return min_val <= obj <= max_val
            arr = np.array(obj, dtype=float)
            # Empty arrays satisfy vacuous truth (all zero elements are in range)
            if len(arr) == 0:
                return True
            valid_arr = arr[~np.isnan(arr)]
            # If filtering removed everything (all NaNs), return False
            if len(valid_arr) == 0:
                return False
            # Convert numpy bool to Python bool to avoid np.True_/np.False_
            result = np.all(valid_arr >= min_val) and np.all(valid_arr <= max_val)
            return bool(result)
        except Exception:
            return False

    @staticmethod
    def has_infinite_values(obj: Any) -> bool:
        """Check if numeric object contains infinite values.

        Args:
            obj (Any): Object to test.

        Returns:
            bool: True if obj contains any infinite numeric values;
            otherwise False.
        """
        try:
            if isinstance(obj, (int, float)):
                return np.isinf(obj)
            elif isinstance(obj, pd.DataFrame):
                return np.isinf(obj.values).any()
            elif isinstance(obj, np.ndarray):
                return np.isinf(obj).any()
            else:
                return np.isinf(np.array(obj)).any()
        except Exception:
            return False


# ------------------------------------------------------------------
# Validator Protocol
# ------------------------------------------------------------------
class Validator(Protocol):
    """Protocol for all validators.

    Any class that implements this protocol must provide a validate method
    that accepts an arbitrary object and returns a ValidationResult.
    """

    def validate(self, obj: Any) -> ValidationResult:
        """Validate an object and return results.

        Args:
            obj (Any): The object that needs to be validated.

        Returns:
            ValidationResult: Aggregated information about the validation
            outcome, including any issues detected.
        """
        ...


# ------------------------------------------------------------------
# Base Validator Class
# ------------------------------------------------------------------
class BaseValidator(ABC):
    """Abstract base class for all validators.

    Subclasses should provide an implementation of the validate method
    and may use the helper methods defined here.
    """

    def __init__(self) -> None:
        """Create a new BaseValidator instance."""
        self.type_checker = TypeChecker()

    @abstractmethod
    def validate(self, obj: Any) -> ValidationResult:
        """Validate an object and return results.

        Args:
            obj (Any): Object to validate.

        Returns:
            ValidationResult: Validation outcome including any issues.
        """
        pass

    def _create_result(self, object_type: str, object_stage: str = "") -> ValidationResult:
        """Create a new validation result.

        Args:
            object_type (str): Type of the object being validated.
            object_stage (str): Stage of the object during validation. Defaults to "".

        Returns:
            ValidationResult: A fresh ValidationResult instance marked as
            initially valid.
        """
        return ValidationResult(
            is_valid=True,
            object_type=object_type,
            object_stage=object_stage,
        )


# ------------------------------------------------------------------
# Dataset Validator
# ------------------------------------------------------------------
class DatasetValidator(BaseValidator):
    """Validator for dataset objects.

    This validator checks that a dataset dictionary contains all
    required fields, that those fields are of the expected types,
    and that the contained data are internally consistent and
    of acceptable quality.
    """

    def validate(self, dataset: Dict[str, Any]) -> ValidationResult:
        """Validate a dataset object.

        Args:
            dataset (Dict[str, Any]): Dictionary containing dataset
            information.

        Returns:
            ValidationResult: Aggregated validation results, including
            any detected issues and dataset metadata.
        """
        result = self._create_result(ObjectType.DATASET.value)

        if not isinstance(dataset, dict):
            result.add_issue("error", "Dataset must be a dictionary")
            return result

        # Check required fields
        self._validate_required_fields(dataset, result)

        # Check field types
        self._validate_field_types(dataset, result)

        # Check data consistency
        self._validate_data_consistency(dataset, result)

        # Check for data quality issues
        self._validate_data_quality(dataset, result)

        # Add dataset details
        result.details = self._get_dataset_details(dataset)

        return result

    def _validate_required_fields(self, dataset: Dict[str, Any], result: ValidationResult) -> None:
        """Check that required fields are present and not null.

        Args:
            dataset (Dict[str, Any]): Dataset dictionary to check.
            result (ValidationResult): Result object to record any
            issues found.

        Raises:
            None. Issues are added to result instead of raising
            exceptions.
        """
        required_fields = ["expr", "pheno", "keys"]

        for field in required_fields:
            if dataset.get(field) is None:
                result.add_issue("error", f"Dataset ${field} is null", field)

        if dataset.get("formattedName") is None:
            result.add_issue("warning", "Dataset $formattedName is null", "formattedName")

    def _validate_field_types(self, dataset: Dict[str, Any], result: ValidationResult) -> None:
        """Validate types of dataset fields.

        Args:
            dataset (Dict[str, Any]): Dataset dictionary containing the raw data.
            result (ValidationResult): Result object where any detected issues are
            recorded.

        This method checks that each of the primary fields in the dataset
        (pheno, expr, class, keys, formattedName) is of the correct
        Python/NumPy/Pandas type and that numeric fields actually contain
        numeric values.  It does not raise exceptions – all problems are
        added to result instead.
        """
        # Validate pheno
        pheno_data = dataset.get("pheno")
        if pheno_data is not None and not isinstance(pheno_data, pd.DataFrame):
            result.add_issue("error", "pheno must be a DataFrame", "pheno")

        # Validate expr
        expr_data = dataset.get("expr")
        if expr_data is not None:
            if isinstance(expr_data, pd.DataFrame):
                # pd.api.types.is_numeric_dtype, not np.issubdtype: a
                # DataFrame column's dtype can be a pandas extension dtype
                # (e.g. the modern default string dtype), which
                # np.issubdtype can't interpret and raises TypeError on
                # instead of returning False.
                if not pd.api.types.is_numeric_dtype(expr_data.dtypes.iloc[0]):
                    result.add_issue("error", "expr must contain numeric data", "expr")
            elif isinstance(expr_data, np.ndarray):
                if not np.issubdtype(expr_data.dtype, np.number):
                    result.add_issue("error", "expr must contain numeric data", "expr")
            else:
                result.add_issue("error", "expr must be DataFrame or numpy array", "expr")

        # Validate class vector
        self._validate_class_vector(dataset, result)

        # Validate keys
        keys = dataset.get("keys")
        if keys is not None and not isinstance(keys, (list, np.ndarray, dict)):
            result.add_issue("error", "keys must be a list, array, or dict", "keys")

        # Validate formattedName
        formatted_name = dataset.get("formattedName")
        if formatted_name is not None and not isinstance(formatted_name, str):
            result.add_issue("error", "formattedName must be a string", "formattedName")

    def _validate_class_vector(self, dataset: Dict[str, Any], result: ValidationResult) -> None:
        """Validate the class vector.

        Args:
            dataset (Dict[str, Any]): Dataset dictionary that may contain a
            class vector.
            result (ValidationResult): Result object where any detected issues
            are recorded.

        The class vector is optional.  When present it must be numeric and
        contain only the values 0, 1, or -1.  The method accepts the vector in
        a variety of Python or pandas containers (dict, list, Series, or a
        scalar) and normalises it to a list of floats for validation.
        """
        class_vector = dataset.get("class")

        if class_vector is None:
            return  # Class vector is optional

        # Convert to list for checking
        # Bug fix:
        # The bug was that numpy arrays (np.ndarray) needs to be included in the isinstance check.
        # When class_vector is a numpy array, it fell through to the else clause which wrapped
        # the entire array in a list: [array([0, 0, 1, 1, ...])]
        # This caused float(v) to try converting the whole array → TypeError
        if isinstance(class_vector, dict):
            values = list(class_vector.values())
        elif isinstance(class_vector, (pd.Series, list, np.ndarray)):  # ✅ FIXED: Added np.ndarray
            values = list(class_vector)
        else:
            values = [class_vector]

        # Check if all values are numeric
        try:
            numeric_values = [float(v) for v in values]
        except (ValueError, TypeError):
            result.add_issue("error", "Class vector must contain numeric values", "class")
            return

        # Check if all values are valid (0, 1, or -1)
        for val in numeric_values:
            if val not in ValidationConstants.VALID_CLASS_VALUES:
                result.add_issue(
                    "error",
                    f"Class vector contains invalid value: {val}. Only 0, 1, and -1 are allowed.",
                    "class",
                )
                break

    def _validate_data_consistency(self, dataset: Dict[str, Any], result: ValidationResult) -> None:
        """Check that dimensions and names match across data structures.

        FIXED VERSION - Handles unnamed class vectors and invalid expr types safely.

        Args:
            dataset (Dict[str, Any]): Dictionary that holds the dataset.
            result (ValidationResult): Result object where any detected
            inconsistencies are added.

        This method performs three consistency checks:

        1.  It verifies that the number of rows (genes) in the expr field
            matches the number of entries in keys. The expr field may be a
            pandas DataFrame or a NumPy array; keys may be a dict, a list,
            or a NumPy array.  When the counts differ an error is added to
            the result with field name 'expr/keys'.

        2.  It ensures that the sample identifiers in expr match the
            sample identifiers in pheno. For a DataFrame expr the sample
            names are its column names; for a NumPy array they are inferred
            from the array shape or from a flat vector. For a DataFrame
            pheno the sample names are its index; otherwise a numeric range
            is used.  Mismatches produce an error under field name
            'expr/pheno'.

        3.  When a class vector is provided it must be named and the set
            of its sample identifiers must coincide with the sample names
            in expr.  If the class vector is a plain dict or a pandas Series
            with an index, the sample names are extracted accordingly.  A
            missing or unnamed class vector triggers a warning; a mismatch
            with expr sample names results in an error under field name
            'class/expr'.
        """

        expr_data = dataset.get("expr")
        keys = dataset.get("keys")
        pheno_data = dataset.get("pheno")
        class_vector = dataset.get("class")

        # Check expr and keys consistency
        if expr_data is not None and keys is not None:
            if isinstance(expr_data, pd.DataFrame):
                n_expr_rows = len(expr_data)
            elif isinstance(expr_data, np.ndarray):
                n_expr_rows = len(expr_data)
            else:
                # Invalid type - skip this check, type validation will catch it
                n_expr_rows = 0

            if isinstance(keys, dict):
                n_keys = len(keys)
                keys_named = True
            elif isinstance(keys, (list, np.ndarray)):
                n_keys = len(keys)
                keys_named = False
                # STRICT MODE: Keys must be named to match R behavior
                # R requires names(keys) to match rownames(expr)
                result.add_issue(
                    "warning",
                    "Keys vector is not named. In R, keys must have names matching gene/probe IDs.",
                    "keys",
                )
            else:
                n_keys = 1
                keys_named = False

            if n_expr_rows > 0 and n_expr_rows != n_keys:
                result.add_issue(
                    "error",
                    f"Number of genes in expr ({n_expr_rows}) does not match "
                    f"number of keys ({n_keys})",
                    "expr/keys",
                )

        # Check expr and pheno sample consistency
        if expr_data is not None and pheno_data is not None:
            if isinstance(expr_data, pd.DataFrame):
                expr_samples = expr_data.columns.tolist()
            elif isinstance(expr_data, np.ndarray):
                # Safe handling for numpy arrays
                expr_samples = list(
                    range(expr_data.shape[1] if len(expr_data.shape) > 1 else len(expr_data))
                )
            else:
                # For invalid types (strings, etc.), skip this check
                # The type validation will catch it
                return

            if isinstance(pheno_data, pd.DataFrame):
                pheno_samples = pheno_data.index.tolist()
            else:
                pheno_samples = list(range(len(pheno_data)))

            if expr_samples != pheno_samples:
                result.add_issue(
                    "error", "Sample names in expr and pheno do not match", "expr/pheno"
                )

        # Check class vector consistency
        if class_vector is not None and expr_data is not None:
            # Try to get named samples from class vector
            if isinstance(class_vector, dict):
                class_samples = list(class_vector.keys())
                has_names = True
            elif hasattr(class_vector, "index") and hasattr(class_vector.index, "tolist"):
                class_samples = class_vector.index.tolist()
                has_names = True
            else:
                # STRICT MODE: Class vector must be named to match R behavior
                # R requires names(class) to match colnames(expr)
                has_names = False

                # Add warning that class vector is unnamed (R would reject this)
                result.add_issue(
                    "warning",
                    "Class vector is not named. In R, class vector must have names matching sample IDs.",
                    "class",
                )

                # Get number of samples from expr
                if isinstance(expr_data, pd.DataFrame):
                    n_expr_samples = len(expr_data.columns)
                elif isinstance(expr_data, np.ndarray):
                    n_expr_samples = (
                        expr_data.shape[1] if len(expr_data.shape) > 1 else len(expr_data)
                    )
                else:
                    # Invalid expr type - skip check
                    return

                # Check length match for unnamed class vector
                try:
                    class_len = len(class_vector)
                except TypeError:
                    # Scalar class value
                    class_len = 1

                if class_len != n_expr_samples:
                    result.add_issue(
                        "error",
                        f"Class vector length ({class_len}) does not match "
                        f"number of samples ({n_expr_samples})",
                        "class/expr",
                    )

            # If class vector has names, check they match expr samples
            if has_names and isinstance(expr_data, pd.DataFrame):
                expr_samples = expr_data.columns.tolist()
                if class_samples != expr_samples:
                    result.add_issue(
                        "error",
                        "Sample names in class do not match sample names in expr",
                        "class/expr",
                    )

    def _validate_data_quality(self, dataset: Dict[str, Any], result: ValidationResult) -> None:
        """Check for data quality issues.

        Args:
            dataset (Dict[str, Any]): Dictionary containing the dataset.
            result (ValidationResult): Result object where issues will be recorded.

        This method performs two quality checks. First it scans the expression
        matrix for infinite values using the type checker. When an infinite
        value is found an error is added with a message that includes the
        dataset’s formatted name. Second it examines the keys list or
        array to ensure that at least one non‑missing identifier is
        present. If all keys are None, 'na', 'nan', or empty strings a
        warning is added, again referencing the formatted name to help
        locate the problem in probe mapping files. No exceptions are
        raised – all problems are appended to result instead.
        """
        # Check for infinite values in expression data
        expr_data = dataset.get("expr")
        if expr_data is not None and self.type_checker.has_infinite_values(expr_data):
            formatted_name = dataset.get("formattedName", "Unknown")
            result.add_issue(
                "error",
                f"Expression data contains infinite values in {formatted_name}",
                "expr",
            )

        # Check for all-NA keys
        keys = dataset.get("keys")
        if keys is not None:
            if isinstance(keys, (list, np.ndarray, pd.Series)):
                valid_keys = [
                    k for k in keys if k is not None and str(k).lower() not in ["na", "nan", ""]
                ]
                if len(valid_keys) == 0:
                    formatted_name = dataset.get("formattedName", "Unknown")
                    result.add_issue(
                        "warning",
                        f"All keys are NA for {formatted_name}. "
                        "Check probe mappings to gene names.",
                        "keys",
                    )

    def _get_dataset_details(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Extract details about the dataset for reporting.

        FIXED VERSION - Safe handling of invalid expr types.

        Args:
            dataset (Dict[str, Any]): Dictionary that holds the dataset.

        Returns:
            Dict[str, Any]: Dictionary containing summary statistics that can
            be used for reporting and validation.  The returned dictionary
            includes the following keys:

            * n_genes (int): Number of genes in the expression matrix.
            * n_samples (int): Number of samples in the expression matrix.
            * n_cases (int): Number of cases (value 1) in the class vector.
            * n_controls (int): Number of controls (value 0) in the class vector.
            * formatted_name (str): Dataset formatted name, defaulting to
            'Unknown' if not provided.

        The method first inspects the expression data. If the data are a
        pandas DataFrame the number of genes is the number of rows and the
        number of samples is the number of columns. If the data are a
        NumPy array the dimensions are taken from the array shape; for a
        one‑dimensional array the number of samples is treated as one.
        It then looks at the class vector, accepting dictionaries,
        iterable objects or scalars, and counts the occurrences of 1 and 0
        to populate n_cases and n_controls. Finally it stores the
        formatted name of the dataset for easy reference in reports.  The
        returned dictionary is designed to be serialisable and suitable
        for logging or user display.
        """
        details = {}

        # Expression data details - with safe handling
        expr_data = dataset.get("expr")
        if expr_data is not None:
            try:
                if isinstance(expr_data, pd.DataFrame):
                    details["n_genes"] = len(expr_data)
                    details["n_samples"] = len(expr_data.columns)
                elif isinstance(expr_data, np.ndarray):
                    details["n_genes"] = expr_data.shape[0]
                    details["n_samples"] = expr_data.shape[1] if len(expr_data.shape) > 1 else 1
                # Skip details for invalid types (string, etc.)
            except (AttributeError, IndexError, TypeError):
                # If we can't get shape info, skip details
                pass

        # Class distribution
        class_vector = dataset.get("class")
        if class_vector is not None:
            try:
                if isinstance(class_vector, dict):
                    values = list(class_vector.values())
                elif hasattr(class_vector, "__iter__") and not isinstance(class_vector, str):
                    values = list(class_vector)
                else:
                    values = [class_vector]

                details["n_cases"] = sum(1 for v in values if v == 1)
                details["n_controls"] = sum(1 for v in values if v == 0)
            except (TypeError, ValueError):
                # If we can't process class vector, skip details
                pass

        details["formatted_name"] = dataset.get("formattedName", "Unknown")

        return details

    @staticmethod
    def check_class_vector(dataset: Dict) -> bool:
        """
        Check if dataset has proper class vector structure.

        FIXED: Now properly handles None class vectors.

        Args:
            dataset: Dictionary containing 'expr', 'class', and 'keys'

        Returns:
            True if class vector is valid (at least 2 cases and 2 controls), False otherwise
        """
        # Check required keys exist
        if "class" not in dataset or "expr" not in dataset:
            return False

        # CRITICAL FIX: Check if class vector is None BEFORE trying to use it
        class_vector_raw = dataset["class"]
        if class_vector_raw is None:
            return False  # Cannot validate a None class vector

        expr_matrix = dataset["expr"]

        # Now safe to convert to numpy array
        try:
            class_vector = np.array(class_vector_raw)
        except (ValueError, TypeError) as e:
            # Handle cases where conversion fails
            return False

        # Check if we got a valid array (not 0-dimensional)
        if class_vector.ndim == 0:
            # This happens with np.array(None) or np.array(scalar)
            return False

        # Check if class vector length matches number of samples
        try:
            n_samples = (
                expr_matrix.shape[1] if hasattr(expr_matrix, "shape") else len(expr_matrix.columns)
            )
        except (AttributeError, IndexError):
            return False

        if len(class_vector) != n_samples:
            return False

        # Check if there are at least 2 cases and 2 controls
        n_cases = np.sum(class_vector == 1)
        n_controls = np.sum(class_vector == 0)

        return bool(n_cases >= 2 and n_controls >= 2)

    @staticmethod
    def check_original_data_names(meta_object: Dict) -> bool:
        """
        Check if original data names are valid Python identifiers.

        Args:
            meta_object: Meta-analysis object containing originalData

        Returns:
            True if all names are valid, False otherwise
        """
        if "originalData" not in meta_object:
            return True

        original_names = list(meta_object["originalData"].keys())

        valid_names = [DatasetValidator.make_valid_name(name) for name in original_names]

        return original_names == valid_names

    @staticmethod
    def make_valid_name(name: str) -> str:
        """
        Convert a string to a valid Python identifier.

        Args:
            name: Original name string

        Returns:
            Valid Python identifier
        """
        # Replace invalid characters with underscores
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

        # Ensure it doesn't start with a number
        if name and name[0].isdigit():
            name = f"_{name}"

        # Ensure it's not empty
        if not name:
            name = "_unnamed"

        return name

    @staticmethod
    def convert_original_data_names(meta_object: Dict) -> Dict:
        """
        Convert original data names to valid identifiers and preserve original names.

        Args:
            meta_object: Meta-analysis object

        Returns:
            Modified meta-analysis object with converted names
        """
        if "originalData" not in meta_object:
            return meta_object

        if DatasetValidator.check_original_data_names(meta_object):
            return meta_object

        original_data = meta_object["originalData"]
        old_names = list(original_data.keys())
        new_names = [DatasetValidator.make_valid_name(name) for name in old_names]

        # Create new dictionary with converted names
        new_original_data = {}

        for old_name, new_name in zip(old_names, new_names):
            dataset = original_data[old_name].copy()

            # Store the original formatted name if not already present
            if "formattedName" not in dataset or not dataset["formattedName"]:
                dataset["formattedName"] = old_name

            new_original_data[new_name] = dataset

        meta_object["originalData"] = new_original_data
        return meta_object

    @staticmethod
    def validate_study_structure(study: Dict) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of study structure.

        Args:
            study: Study dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required keys
        required_keys = ["expr", "class", "keys"]
        for key in required_keys:
            if key not in study:
                errors.append(f"Missing required key: {key}")

        if errors:
            return False, errors

        # Validate expression matrix
        expr = study["expr"]
        if not isinstance(expr, (pd.DataFrame, np.ndarray)):
            errors.append("Expression data must be a DataFrame or numpy array")
        else:
            if isinstance(expr, np.ndarray) and expr.ndim != 2:
                errors.append("Expression array must be 2-dimensional")
            elif isinstance(expr, pd.DataFrame) and len(expr.shape) != 2:
                errors.append("Expression DataFrame must be 2-dimensional")

        # Validate class vector
        class_vector = study["class"]
        if not isinstance(class_vector, (list, np.ndarray, pd.Series)):
            errors.append("Class vector must be a list, array, or Series")
        else:
            unique_classes = np.unique(class_vector)
            if not np.array_equal(sorted(unique_classes), [0, 1]):
                errors.append("Class vector must contain only 0 (controls) and 1 (cases)")

        # Validate keys
        keys = study["keys"]
        if not isinstance(keys, (list, np.ndarray)):
            errors.append("Keys must be a list or array")

        # Check dimension consistency
        if not errors:  # Only check if basic structure is valid
            try:
                n_genes, n_samples = expr.shape if isinstance(expr, np.ndarray) else expr.shape

                if len(class_vector) != n_samples:
                    errors.append(
                        f"Class vector length ({len(class_vector)}) doesn't match "
                        f"number of samples ({n_samples})"
                    )

                if len(keys) != n_genes:
                    errors.append(
                        f"Keys length ({len(keys)}) doesn't match " f"number of genes ({n_genes})"
                    )

            except Exception as e:
                errors.append(f"Error checking dimensions: {str(e)}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_meta_object(
        meta_object: Dict, check_type: str = "Meta", stage: str = "Pre-Analysis"
    ) -> Tuple[bool, List[str]]:
        """
        Validate complete meta-analysis object.

        Args:
            meta_object: Meta-analysis object to validate
            check_type: Type of check ('Meta', 'Validation', etc.)
            stage: Analysis stage ('Pre-Analysis', 'Post-Analysis', etc.)

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Basic structure checks
        if not isinstance(meta_object, dict):
            errors.append("Meta object must be a dictionary")
            return False, errors

        # Check for originalData
        if "originalData" not in meta_object:
            errors.append("Meta object must contain 'originalData' key")
            return False, errors

        original_data = meta_object["originalData"]
        if not isinstance(original_data, dict):
            errors.append("originalData must be a dictionary")
            return False, errors

        if len(original_data) == 0:
            errors.append("originalData cannot be empty")
            return False, errors

        # Validate each dataset
        for dataset_name, dataset in original_data.items():
            is_valid, dataset_errors = DatasetValidator.validate_study_structure(dataset)
            if not is_valid:
                for error in dataset_errors:
                    errors.append(f"Dataset '{dataset_name}': {error}")

        # Stage-specific validations
        if stage == "Pre-Analysis":
            # Check class vectors for all datasets
            for dataset_name, dataset in original_data.items():
                if not DatasetValidator.check_class_vector(dataset):
                    errors.append(
                        f"Dataset '{dataset_name}': Insufficient cases/controls "
                        "(need at least 2 of each)"
                    )

        elif stage == "Post-Analysis":
            # Check for analysis results
            required_results = ["metaAnalysis"]
            for result in required_results:
                if result not in meta_object:
                    errors.append(f"Missing required analysis result: {result}")

        return len(errors) == 0, errors


# ------------------------------------------------------------------
# Meta Analysis Validator
# ------------------------------------------------------------------
class MetaAnalysisValidator(BaseValidator):
    """Validator for MetaAnalysis objects."""

    def validate(self, meta_analysis: Dict[str, Any]) -> ValidationResult:
        """Validate a MetaAnalysis object.

        Args:
            meta_analysis (Dict[str, Any]): Dictionary that holds the MetaAnalysis
            data.
            result (ValidationResult): Object where any found issues are added.

        Returns:
            ValidationResult: Result object that contains a status,
            a list of issues and any summary details.

        The method first creates a base result with the
        META_ANALYSIS type. It then checks that meta_analysis is a
        dictionary; when it is not an error is added and the
        incomplete result is returned. If the dictionary passes the
        check the following validations are performed in order:

        * The required field names listed in
        ValidationConstants.META_ANALYSIS_ENTRIES are verified to be
        present; missing fields trigger an error.
        * The field types are validated against a mapping of expected
        types. Incorrect types produce an error.
        * The pooled results section is examined by
        _validate_pooled_results.
        Finally the method populates the details section with
        information from _get_meta_analysis_details. The returned
        ValidationResult can be used by callers to determine
        success or to report issues to users.

        The method never raises an exception; every validation
        problem is recorded in result through add_issue calls.
        """
        result = self._create_result(ObjectType.META_ANALYSIS.value)

        if not isinstance(meta_analysis, dict):
            result.add_issue("error", "MetaAnalysis must be a dictionary")
            return result

        # Check required fields
        self._validate_required_fields(meta_analysis, result)

        # Check field types
        self._validate_field_types(meta_analysis, result)

        # Check pooled results
        self._validate_pooled_results(meta_analysis, result)

        # Add details
        result.details = self._get_meta_analysis_details(meta_analysis)

        return result

    def _validate_required_fields(
        self, meta_analysis: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Check required fields are present.

        Args:
            meta_analysis (Dict[str, Any]): Dictionary that holds the
            MetaAnalysis data.
            result (ValidationResult): Result object where missing field
            errors are recorded.

        For each entry specified in
        ValidationConstants.META_ANALYSIS_ENTRIES the method checks
        whether the entry exists in meta_analysis. When an entry is
        missing or its value is None an error is appended to result
        with a message that identifies the entry. No other
        modifications are performed; the method terminates after
        completing all checks.
        """
        for entry in ValidationConstants.META_ANALYSIS_ENTRIES:
            if meta_analysis.get(entry) is None:
                result.add_issue("error", f"{entry} is null", entry)

    def _validate_field_types(
        self, meta_analysis: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate field types.

        Args:
            meta_analysis (Dict[str, Any]): Dictionary that holds the
            MetaAnalysis data.
            result (ValidationResult): Object where type errors are
            recorded.

        The method builds a mapping of field names to expected
        type tuples. Each field is retrieved from meta_analysis; when
        the value exists and is not an instance of the expected type
        an error is appended to result. The fields checked are
        datasetEffectSizes, datasetEffectSizeStandardErrors,
        pooledResults and analysisDescription.  This ensures that
        downstream consumers of the MetaAnalysis object receive data
        in the correct format.  No exceptions are raised – problems
        are appended to result instead.
        """
        # Map expected types
        type_map = {
            "datasetEffectSizes": (pd.DataFrame, np.ndarray),
            "datasetEffectSizeStandardErrors": (pd.DataFrame, np.ndarray),
            "pooledResults": (pd.DataFrame, dict),
            "analysisDescription": str,
        }

        for field, expected_type in type_map.items():
            value = meta_analysis.get(field)
            if value is not None and not isinstance(value, expected_type):
                result.add_issue("error", f"{field} has incorrect type", field)

    def _validate_pooled_results(
        self, meta_analysis: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate pooled results structure and values.

        Args:
            meta_analysis (Dict[str, Any]): Dictionary that holds the
            MetaAnalysis data.
            result (ValidationResult): Result object where issues are
            recorded.

        The method first retrieves the pooledResults field from the
        meta_analysis dictionary.  When the field is missing or
        None the method exits without adding an issue.  If the field
        exists the method verifies that every required entry listed in
        ValidationConstants.POOLED_RESULTS_ENTRIES is present.  For a
        dictionary pooledResults the entry must have a non‑null value;
        for a pandas DataFrame the entry must appear in the column
        list.  Missing or null entries generate an error recorded
        under a field name that prefixes pooledResults and the
        missing entry.  After the structural checks the method
        delegates range validation to _validate_pooled_results_ranges.
        No exception is raised – all problems are appended to result
        via add_issue.
        """
        pooled_results = meta_analysis.get("pooledResults")
        if pooled_results is None:
            return

        # Check required entries
        for entry in ValidationConstants.POOLED_RESULTS_ENTRIES:
            if isinstance(pooled_results, dict):
                if pooled_results.get(entry) is None:
                    result.add_issue(
                        "error",
                        f"pooledResults.{entry} is null",
                        f"pooledResults.{entry}",
                    )
            elif isinstance(pooled_results, pd.DataFrame):
                if entry not in pooled_results.columns:
                    result.add_issue(
                        "error",
                        f"pooledResults.{entry} is missing",
                        f"pooledResults.{entry}",
                    )

        # Validate value ranges
        self._validate_pooled_results_ranges(pooled_results, result)

    def _validate_pooled_results_ranges(
        self,
        pooled_results: Union[Dict[str, Any], pd.DataFrame],
        result: ValidationResult,
    ) -> None:
        """Validate ranges of pooled results values.

        Args:
            pooled_results (Union[Dict[str, Any], pd.DataFrame]): Object that
            stores the pooled results; it may be a plain dictionary or a
            pandas DataFrame.
            result (ValidationResult): Object where range violations are
            added.

        The method defines a set of constraints for each numeric
        field that should appear in the pooled results.  Each field is
        inspected in the supplied pooled_results object.  For a
        dictionary the value is retrieved with get; for a DataFrame
        the column values are taken if the column exists.  When a
        value is found the method checks two types of constraints:
        * 'non-negative' ensures that all numeric values are greater
        than or equal to zero.
        * 'range' ensures that the value lies within a specified
        interval.  When a value violates its constraint an error is
        added to result with a message that identifies the offending
        field.  The method performs no other action and never raises
        an exception.
        """
        range_constraints = {
            "effectSizeStandardError": ("non-negative", None),
            "effectSizePval": ("range", (0, 1)),
            "effectSizeFDR": ("range", (0, 1)),
            "tauSquared": ("non-negative", None),
            "numStudies": ("non-negative", None),
            "cochranesQ": ("non-negative", None),
            "heterogeneityPval": ("range", (0, 1)),
            "fisherStatUp": ("non-negative", None),
            "fisherPvalUp": ("range", (0, 1)),
            "fisherFDRUp": ("range", (0, 1)),
            "fisherStatDown": ("non-negative", None),
            "fisherPvalDown": ("range", (0, 1)),
            "fisherFDRDown": ("range", (0, 1)),
        }

        for field, (constraint_type, constraint_value) in range_constraints.items():
            if isinstance(pooled_results, dict):
                value = pooled_results.get(field)
            elif isinstance(pooled_results, pd.DataFrame):
                value = pooled_results[field].values if field in pooled_results.columns else None
            else:
                continue

            if value is not None:
                if constraint_type == "non-negative" and not self._is_non_negative(value):
                    result.add_issue(
                        "error",
                        f"pooledResults.{field} contains negative values",
                        f"pooledResults.{field}",
                    )
                elif constraint_type == "range" and not self.type_checker.is_in_range(
                    value, constraint_value[0], constraint_value[1]
                ):
                    result.add_issue(
                        "error",
                        f"pooledResults.{field} contains values outside range "
                        f"[{constraint_value[0]}, {constraint_value[1]}]",
                        f"pooledResults.{field}",
                    )

    def _is_non_negative(self, obj: Any) -> bool:
        """Check if values are non-negative.

        Args:
            obj (Any): Scalar or array‑like object whose numeric
            components should be inspected.

        Returns:
            bool: True if all numeric components of obj are greater
            than or equal to zero and False otherwise.  The method
            attempts to convert the input into a NumPy array and
            ignores any NaN values.  Any failure during conversion
            or comparison results in False, but no exception is
            propagated to the caller.
        """
        try:
            if isinstance(obj, (int, float)):
                return obj >= 0
            arr = np.array(obj, dtype=float)
            valid_arr = arr[~np.isnan(arr)]
            return np.all(valid_arr >= 0)
        except:
            return False

    def _get_meta_analysis_details(self, meta_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract details about the meta analysis.

        Args:
            meta_analysis (Dict[str, Any]): Dictionary that holds the
            MetaAnalysis data.

        Returns:
            Dict[str, Any]: Summary dictionary that contains the
            number of genes and the analysis description.  If pooledResults
            is a pandas DataFrame the number of genes is taken from its
            length.  If pooledResults is a dictionary the method looks at
            the first entry; if that entry has a length the number of genes
            is taken from that length.  The analysis description is
            retrieved from the analysisDescription field, defaulting to
            'Unknown' when the field is missing.

        The method starts by creating an empty details dictionary.
        It then checks whether the pooledResults field is present.
        For DataFrame pooledResults the details dictionary is updated
        with a key n_genes whose value is the number of rows in the
        DataFrame.  For a dictionary pooledResults the method obtains
        the first key, and if the corresponding value supports
        len() the n_genes entry is set to that length.  Finally the
        analysis description is added under the key analysis_description.
        No exceptions are raised; when an unexpected structure is
        encountered the method simply omits the corresponding entry.
        """
        details = {}

        pooled_results = meta_analysis.get("pooledResults")
        if pooled_results is not None:
            if isinstance(pooled_results, pd.DataFrame):
                details["n_genes"] = len(pooled_results)
            elif isinstance(pooled_results, dict):
                first_key = next(iter(pooled_results), None)
                if first_key and hasattr(pooled_results[first_key], "__len__"):
                    details["n_genes"] = len(pooled_results[first_key])

        details["analysis_description"] = meta_analysis.get("analysisDescription", "Unknown")

        return details


# ------------------------------------------------------------------
# Meta Filter Validator
# ------------------------------------------------------------------
class MetaFilterValidator(BaseValidator):
    """Validator for MetaFilter objects."""

    def validate(self, meta_filter: Dict[str, Any]) -> ValidationResult:
        """Validate a MetaFilter object.

        Args:
            meta_filter (Dict[str, Any]): Dictionary that holds the MetaFilter data.
            result (ValidationResult): Result object where any validation problems are added.

        Returns:
            ValidationResult: Result object that contains the validation status,
            any detected issues and a details section.

        The method creates a base result for the META_FILTER type. It checks that
        meta_filter is a dictionary; when it is not an error is added and the
        incomplete result is returned. If the dictionary passes the type
        check the following validations are performed in order:

        * The required field names listed in ValidationConstants.FILTER_ENTRIES
        are verified to be present; missing or null entries trigger an error.
        * Field types and numeric limits are validated against a
        predefined list of expectations; violations produce an error.
        * The details section of the result is populated with
        information extracted by _get_meta_filter_details.

        No exception is raised – every problem is recorded in result via
        add_issue calls.
        """
        result = self._create_result(ObjectType.META_FILTER.value)

        if not isinstance(meta_filter, dict):
            result.add_issue("error", "MetaFilter must be a dictionary")
            return result

        # Check required fields
        self._validate_required_fields(meta_filter, result)

        # Check field types and ranges
        self._validate_field_types(meta_filter, result)

        # Add details
        result.details = self._get_meta_filter_details(meta_filter)

        return result

    def _validate_required_fields(
        self, meta_filter: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Check required fields are present.

        Args:
            meta_filter (Dict[str, Any]): Dictionary that holds the MetaFilter data.
            result (ValidationResult): Object where missing field errors are recorded.

        For each entry defined in ValidationConstants.FILTER_ENTRIES the
        method verifies that the entry exists in meta_filter and that
        its value is not None. When an entry is null an error is
        appended to result with a message that names the offending
        field. The method performs no other actions and ends after
        all entries have been examined.
        """
        for entry in ValidationConstants.FILTER_ENTRIES:
            if meta_filter.get(entry) is None:
                result.add_issue("error", f"{entry} is null", entry)

    def _validate_field_types(self, meta_filter: Dict[str, Any], result: ValidationResult) -> None:
        """Validate field types and ranges.

        Args:
            meta_filter (Dict[str, Any]): Dictionary that holds the MetaFilter data.
            result (ValidationResult): Object where type and range violations are recorded.

        The method iterates over a list of validation tuples. Each tuple
        contains a field name, an expected type, and an optional
        range constraint. For every field the value is retrieved from
        meta_filter. When the value exists and is not an instance of
        the expected type an error is added to result and the field
        is skipped.  If a range constraint is supplied the method
        checks the following:

        * 'non-negative' ensures the numeric value is greater than
        or equal to zero.
        * 'range' ensures the numeric value lies within the
        inclusive interval defined by constraint_value.

        Violations generate an error that identifies the field.
        The method never raises an exception; all problems are
        appended to result via add_issue.
        """
        validations = [
            ("posGeneNames", list, None),
            ("negGeneNames", list, None),
            ("FDRThresh", (int, float), ("range", (0, 1))),
            ("effectSizeThresh", (int, float), ("non-negative", None)),
            ("numberStudiesThresh", int, ("non-negative", None)),
            ("isLeaveOneOut", bool, None),
            ("heterogeneityPvalThresh", (int, float), ("range", (0, 1))),
            ("filterDescription", str, None),
            ("timestamp", (datetime, pd.Timestamp), None),
        ]

        for field, expected_type, range_constraint in validations:
            value = meta_filter.get(field)
            if value is not None:
                if not isinstance(value, expected_type):
                    result.add_issue("error", f"{field} has incorrect type", field)
                    continue

                # Check range constraints
                if range_constraint:
                    constraint_type, constraint_value = range_constraint
                    if constraint_type == "non-negative" and value < 0:
                        result.add_issue("error", f"{field} must be non-negative", field)
                    elif constraint_type == "range" and not (
                        constraint_value[0] <= value <= constraint_value[1]
                    ):
                        result.add_issue(
                            "error",
                            f"{field} must be in range [{constraint_value[0]}, {constraint_value[1]}]",
                            field,
                        )

    def _get_meta_filter_details(self, meta_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Extract details about the filter.

        Args:
            meta_filter (Dict[str, Any]): Dictionary that holds the MetaFilter data.

        Returns:
            Dict[str, Any]: Summary dictionary that includes the number of
            positively and negatively selected genes, the FDR threshold
            and the effect‑size threshold. When a threshold value is
            missing the string 'Unknown' is used instead.
        """
        return {
            "n_pos_genes": len(meta_filter.get("posGeneNames", [])),
            "n_neg_genes": len(meta_filter.get("negGeneNames", [])),
            "fdr_threshold": meta_filter.get("FDRThresh", "Unknown"),
            "effect_size_threshold": meta_filter.get("effectSizeThresh", "Unknown"),
        }


# ------------------------------------------------------------------
# Meta Object Validator
# ------------------------------------------------------------------
class MetaObjectValidator(BaseValidator):
    """Validator for Meta objects (main containers)."""

    def __init__(self) -> None:
        super().__init__()
        self.dataset_validator = DatasetValidator()
        self.meta_analysis_validator = MetaAnalysisValidator()
        self.meta_filter_validator = MetaFilterValidator()

    def validate(
        self,
        meta_obj: Dict[str, Any],
        stage: ObjectStage = ObjectStage.PRE_ANALYSIS,
    ) -> ValidationResult:
        """Validate a Meta object for the given stage.

        Args:
            meta_obj (Dict[str, Any]): Dictionary that holds the Meta object.
            stage (ObjectStage): Validation stage, one of PRE_ANALYSIS,
            PRE_FILTER, POST_FILTER.  The default is PRE_ANALYSIS.

        Returns:
            ValidationResult: Result object containing the validation
            status, a list of issues, and a details dictionary.

        The method creates a result for the META type with the stage
        identifier.  It first checks that meta_obj is a dictionary; if
        not an error is recorded and the incomplete result is returned.
        Then it retrieves the list of components that must be
        validated for the supplied stage by calling
        _get_validation_components.  Each component is validated in
        turn:

        * originalData – validated by _validate_original_data.
        * metaAnalysis – validated by _validate_meta_analysis_component.
        * leaveOneOutAnalysis – validated by
          _validate_leave_one_out_analysis.
        * filterResults – validated by _validate_filter_results.

        After all components are processed the method populates the
        result.details field with a summary of the Meta object for
        the current stage by invoking _get_meta_details.  No
        exception is raised – every problem is appended to result via
        add_issue.
        """
        result = self._create_result(ObjectType.META.value, stage.value)

        if not isinstance(meta_obj, dict):
            result.add_issue("error", "Meta object must be a dictionary")
            return result

        # Determine what to validate based on stage
        validation_components = self._get_validation_components(stage)

        # Validate each component
        for component in validation_components:
            if component == "originalData":
                self._validate_original_data(meta_obj, result)
            elif component == "metaAnalysis":
                self._validate_meta_analysis_component(meta_obj, result)
            elif component == "leaveOneOutAnalysis":
                self._validate_leave_one_out_analysis(meta_obj, result)
            elif component == "filterResults":
                self._validate_filter_results(meta_obj, result)

        # Add details
        result.details = self._get_meta_details(meta_obj, stage)

        return result

    def _get_validation_components(self, stage: ObjectStage) -> List[str]:
        """Get list of components to validate based on stage.

        Args:
            stage (ObjectStage): Validation stage, one of
            PRE_ANALYSIS, PRE_FILTER, POST_FILTER.

        Returns:
            List[str]: Ordered list of component names that must be
            validated for the supplied stage.  If the stage is unknown
            an empty list is returned.
        """
        if stage == ObjectStage.PRE_ANALYSIS:
            return ["originalData"]
        elif stage == ObjectStage.PRE_FILTER:
            return ["originalData", "metaAnalysis", "leaveOneOutAnalysis"]
        elif stage == ObjectStage.POST_FILTER:
            return [
                "originalData",
                "metaAnalysis",
                "leaveOneOutAnalysis",
                "filterResults",
            ]
        else:
            return []

    def _validate_original_data(self, meta_obj: Dict[str, Any], result: ValidationResult) -> None:
        """Validate originalData component.

        Args:
            meta_obj (Dict[str, Any]): Dictionary that holds the Meta
            object.
            result (ValidationResult): Result object where issues are
            appended.

        The method retrieves the originalData field from meta_obj.
        When the field is missing a null, an error is recorded under
        the field path originalData and the method exits.  If the
        field exists but is not a dictionary an error is recorded and
        the method returns.  When the field is a dictionary the
        method iterates over each dataset entry.  For every dataset
        the DatasetValidator is invoked; any issues discovered are
        added to the overall result with a path that prefixes the
        dataset name and the component field.  No exception is
        raised – all problems are recorded via add_issue.
        """
        original_data = meta_obj.get("originalData")
        if original_data is None:
            result.add_issue("error", "originalData is null", "originalData")
            return

        if not isinstance(original_data, dict):
            result.add_issue("error", "originalData must be a dictionary", "originalData")
            return

        for dataset_name, dataset_obj in original_data.items():
            dataset_result = self.dataset_validator.validate(dataset_obj)
            for issue in dataset_result.issues:
                result.add_issue(
                    issue.severity,
                    f"Dataset {dataset_name}: {issue.message}",
                    f"originalData.{dataset_name}.{issue.field_path}",
                )

    def _validate_meta_analysis_component(
        self, meta_obj: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate metaAnalysis component.

        Args:
            meta_obj (Dict[str, Any]): Dictionary that holds the Meta object.
            result (ValidationResult): Result object where issues are added.

        The method retrieves the metaAnalysis field from meta_obj.  When the
        field is missing an error is recorded under the field path
        metaAnalysis and the method exits.  If the field exists it is
        passed to MetaAnalysisValidator.  All issues returned by that
        validator are appended to the overall result with a message
        prefixing 'MetaAnalysis' and a field path that begins with
        metaAnalysis.  No exception is raised – every problem is
        appended via add_issue.
        """
        meta_analysis = meta_obj.get("metaAnalysis")
        if meta_analysis is None:
            result.add_issue("error", "metaAnalysis is null", "metaAnalysis")
            return

        analysis_result = self.meta_analysis_validator.validate(meta_analysis)

        # Add any issues
        for issue in analysis_result.issues:
            result.add_issue(
                issue.severity,
                f"MetaAnalysis: {issue.message}",
                f"metaAnalysis.{issue.field_path}",
            )

    def _validate_leave_one_out_analysis(
        self, meta_obj: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate leaveOneOutAnalysis component.

        Args:
            meta_obj (Dict[str, Any]): Dictionary that holds the Meta object.
            result (ValidationResult): Result object where issues are added.

        The method attempts to retrieve the leaveOneOutAnalysis field
        from meta_obj.  When the field is missing the component is
        considered optional and the method returns without adding any
        issue.  If the field is present the value is normalised to a
        list: dictionaries are converted to a list of their values,
        non‑list scalars are wrapped in a list, and existing lists are
        used unchanged.  Each element of the resulting list is
        validated with MetaAnalysisValidator.  Issues reported by that
        validator are appended to the overall result with a message
        prefixing 'LeaveOneOut[i]' where i is the zero‑based index
        and a field path that starts with leaveOneOutAnalysis[i].  No
        exception is raised – all problems are recorded via
        add_issue.
        """
        loo_analysis = meta_obj.get("leaveOneOutAnalysis")
        if loo_analysis is None:
            return  # Optional component

        # Convert to list if needed
        if isinstance(loo_analysis, dict):
            loo_list = list(loo_analysis.values())
        elif not isinstance(loo_analysis, list):
            loo_list = [loo_analysis]
        else:
            loo_list = loo_analysis

        # Validate each analysis
        for i, analysis in enumerate(loo_list):
            analysis_result = self.meta_analysis_validator.validate(analysis)

            # Add any issues
            for issue in analysis_result.issues:
                result.add_issue(
                    issue.severity,
                    f"LeaveOneOut[{i}]: {issue.message}",
                    f"leaveOneOutAnalysis[{i}].{issue.field_path}",
                )

    def _validate_filter_results(self, meta_obj: Dict[str, Any], result: ValidationResult) -> None:
        """Validate filterResults component.

        Args:
            meta_obj (Dict[str, Any]): Dictionary that holds the Meta object.
            result (ValidationResult): Result object where issues are added.

        The method retrieves the filterResults field from meta_obj.  When the
        field is missing an error is recorded under the field path
        filterResults and the method exits.  If the field is present the
        value is normalised to a list: dictionaries are converted to a
        list of their values, non‑list scalars are wrapped in a list,
        and existing lists are used unchanged.  Each element of the
        resulting list is validated with MetaFilterValidator.  Issues
        reported by that validator are appended to the overall result
        with a message prefixing 'Filter[i]' where i is the zero‑based
        index and a field path that starts with filterResults[i].  No
        exception is raised – all problems are recorded via
        add_issue.
        """
        filter_results = meta_obj.get("filterResults")
        if filter_results is None:
            result.add_issue("error", "filterResults is null", "filterResults")
            return

        # Convert to list if needed
        if isinstance(filter_results, dict):
            filter_list = list(filter_results.values())
        elif not isinstance(filter_results, list):
            filter_list = [filter_results]
        else:
            filter_list = filter_results

        # Validate each filter
        for i, filter_obj in enumerate(filter_list):
            filter_result = self.meta_filter_validator.validate(filter_obj)

            # Add any issues
            for issue in filter_result.issues:
                result.add_issue(
                    issue.severity,
                    f"Filter[{i}]: {issue.message}",
                    f"filterResults[{i}].{issue.field_path}",
                )

    def _get_meta_details(self, meta_obj: Dict[str, Any], stage: ObjectStage) -> Dict[str, Any]:
        """Extract details about the meta object.

        Args:
            meta_obj (Dict[str, Any]): Dictionary that holds the Meta object.
            stage (ObjectStage): Validation stage, one of PRE_ANALYSIS,
            PRE_FILTER, POST_FILTER.

        Returns:
            Dict[str, Any]: Summary dictionary that contains a stage
            identifier and counts or flags for the components that
            exist in meta_obj for the requested stage.

        The returned dictionary always contains the numeric value of
        stage.  If originalData is present the number of datasets and
        their names are recorded.  For stages PRE_FILTER and POST_FILTER
        a boolean flag indicating the presence of a metaAnalysis
        component is added.  For POST_FILTER the presence of
        filterResults is recorded and, if available, the number of
        filter results is included.  No exception is raised – the
        method simply builds and returns the summary dictionary.
        """
        details = {"stage": stage.value}

        original_data = meta_obj.get("originalData")
        if original_data and isinstance(original_data, dict):
            details["n_datasets"] = len(original_data)
            details["dataset_names"] = list(original_data.keys())

        if stage in [ObjectStage.PRE_FILTER, ObjectStage.POST_FILTER]:
            details["has_meta_analysis"] = "metaAnalysis" in meta_obj

        if stage == ObjectStage.POST_FILTER:
            filter_results = meta_obj.get("filterResults")
            details["has_filter_results"] = filter_results is not None
            if filter_results:
                details["n_filter_results"] = len(filter_results)

        return details


# ------------------------------------------------------------------
# Main Validation Orchestrator
# ------------------------------------------------------------------
class DataObjectValidator:
    """Main validation orchestrator using composition."""

    def __init__(self) -> None:
        self.dataset_validator = DatasetValidator()
        self.meta_analysis_validator = MetaAnalysisValidator()
        self.meta_filter_validator = MetaFilterValidator()
        self.meta_object_validator = MetaObjectValidator()

    def validate(
        self,
        obj: Any,
        object_type: Union[str, ObjectType],
        object_stage: Union[str, ObjectStage] = ObjectStage.PRE_ANALYSIS,
    ) -> ValidationResult:
        """Main validation entry point.

        Args:
            obj: Object to validate.
            object_type: Type of object (Meta, Dataset, MetaAnalysis, MetaFilter).
            object_stage: Stage for Meta objects, default PRE_ANALYSIS.

        Returns:
            ValidationResult with all validation information.

        The method first normalises the type arguments when strings are
        supplied.  If object_type is a string it is converted to an
        ObjectType enumeration; a ValueError during this conversion
        results in an invalid ValidationResult with a single error
        issue.  The same conversion logic applies to object_stage,
        which is normalised to an ObjectStage enumeration.  After
        normalisation the method routes the validation request to the
        appropriate validator component: DatasetValidator for a
        Dataset, MetaAnalysisValidator for a MetaAnalysis, MetaFilterValidator
        for a MetaFilter, and MetaObjectValidator for a Meta.  If the
        supplied object_type does not match one of the supported
        enumerations an invalid ValidationResult is returned with an
        error issue.  No exception is raised – every failure is
        reported by adding an issue to a ValidationResult instance.
        """
        # Convert string enums
        if isinstance(object_type, str):
            try:
                object_type = ObjectType(object_type)
            except ValueError:
                result = ValidationResult(is_valid=False)
                result.add_issue("error", f"Invalid object type: {object_type}")
                return result

        if isinstance(object_stage, str):
            try:
                object_stage = ObjectStage(object_stage)
            except ValueError:
                result = ValidationResult(is_valid=False)
                result.add_issue("error", f"Invalid object stage: {object_stage}")
                return result

        # Route to appropriate validator
        if object_type == ObjectType.DATASET:
            return self.dataset_validator.validate(obj)
        elif object_type == ObjectType.META_ANALYSIS:
            return self.meta_analysis_validator.validate(obj)
        elif object_type == ObjectType.META_FILTER:
            return self.meta_filter_validator.validate(obj)
        elif object_type == ObjectType.META:
            return self.meta_object_validator.validate(obj, object_stage)
        else:
            result = ValidationResult(is_valid=False)
            result.add_issue("error", f"Unsupported object type: {object_type}")
            return result


# ------------------------------------------------------------------
# Validation Report Generator
# ------------------------------------------------------------------
class ValidationReporter:
    """Generates formatted validation reports."""

    @staticmethod
    def print_validation_report(result: ValidationResult, verbose: bool = True) -> None:
        """Print a formatted validation report.

        Args:
            result (ValidationResult): ValidationResult object to report.
            verbose (bool): Flag to include details section, default True.

        The method prints a structured report to standard output.
        It starts with a header, then shows the object type and, if
        present, the object stage.  The status line displays VALID
        when result.is_valid is True and INVALID otherwise.  Following
        that, it lists any errors and warnings by retrieving them
        through get_errors and get_warnings.  If neither errors nor
        warnings are present a message indicating no issues is
        printed.  When verbose is True and result.details contains
        data, a details section is printed showing each key and
        value from the details dictionary.  The report is framed by
        lines of equal signs for visual separation.  No exception is
        raised; all output is directed to the console.
        """
        print("=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        print(f"Object Type: {result.object_type}")
        if result.object_stage:
            print(f"Object Stage: {result.object_stage}")
        print(f"Status: {'VALID' if result.is_valid else 'INVALID'}")
        print("-" * 60)

        # Print errors
        errors = result.get_errors()
        if errors:
            print("ERRORS:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")

        # Print warnings
        warnings = result.get_warnings()
        if warnings:
            print("WARNINGS:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")

        if not errors and not warnings:
            print("No issues found.")

        # Print details if verbose
        if verbose and result.details:
            print("-" * 60)
            print("DETAILS:")
            for key, value in result.details.items():
                print(f"  {key}: {value}")

        print("=" * 60)

    @staticmethod
    def generate_summary_report(results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate a summary report for multiple validation results.

        Args:
            results (List[ValidationResult]): List of ValidationResult
            objects to summarize.

        Returns:
            Dict[str, Any]: Dictionary containing aggregated statistics.
            Keys include total_objects, valid_objects, invalid_objects,
            total_errors, total_warnings, and object_types.  The
            object_types entry maps each object type to its own
            statistics of total, valid, and invalid counts.

        The method first builds a summary dictionary with overall
        counts of objects, valid and invalid objects, and total
        errors and warnings across all results.  It then iterates
        through each ValidationResult to count occurrences by
        object_type, updating totals and validity counts for each
        type.  The resulting dictionary is returned.  No exception
        is raised – all computation occurs directly on the supplied
        list of results.
        """
        summary = {
            "total_objects": len(results),
            "valid_objects": sum(1 for r in results if r.is_valid),
            "invalid_objects": sum(1 for r in results if not r.is_valid),
            "total_errors": sum(len(r.get_errors()) for r in results),
            "total_warnings": sum(len(r.get_warnings()) for r in results),
            "object_types": {},
        }

        # Count by object type
        for result in results:
            obj_type = result.object_type
            if obj_type not in summary["object_types"]:
                summary["object_types"][obj_type] = {
                    "total": 0,
                    "valid": 0,
                    "invalid": 0,
                }

            summary["object_types"][obj_type]["total"] += 1
            if result.is_valid:
                summary["object_types"][obj_type]["valid"] += 1
            else:
                summary["object_types"][obj_type]["invalid"] += 1

        return summary


# ------------------------------------------------------------------
# Convenience Functions for Backward Compatibility
# ------------------------------------------------------------------
def check_data_object(obj: Any, object_type: str, object_stage: str = "Pre-Analysis") -> bool:
    """
    Backward compatibility function.

    Args:
        obj: Object to validate
        object_type: Type of object
        object_stage: Stage for Meta objects

    Returns:
        True if valid, False otherwise
    """
    validator = DataObjectValidator()
    result = validator.validate(obj, object_type, object_stage)

    # Print warnings for backward compatibility
    for issue in result.issues:
        if issue.severity in ["warning", "error"]:
            warnings.warn(issue.message)

    return result.is_valid


def validate_and_report(
    obj: Any, object_type: str, object_stage: str = "Pre-Analysis"
) -> ValidationResult:
    """
    Validate an object and return detailed results.

    Args:
        obj: Object to validate
        object_type: Type of object
        object_stage: Stage for Meta objects

    Returns:
        ValidationResult with all details
    """
    validator = DataObjectValidator()
    return validator.validate(obj, object_type, object_stage)
