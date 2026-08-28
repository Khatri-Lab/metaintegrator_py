"""
Tests for metaintegrator.labeling — class label assignment.

Covers:
- Basic two-class labeling (disease vs control)
- Case-insensitive substring matching
- Explicit control_terms
- Samples that match neither (NaN exclusion)
- Error paths: missing pheno, missing column, zero matches
- Logging output
- Mutation: the function modifies and returns the same dict
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.labeling import assign_class_labels


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture
def sle_dataset():
    """Realistic SLE phenotype dataset."""
    pheno = pd.DataFrame(
        {
            "disease state:ch1": [
                "Systemic Lupus Erythematosus (SLE)",
                "Systemic Lupus Erythematosus (SLE)",
                "Normal",
                "Normal",
                "Systemic Lupus Erythematosus (SLE)",
                "Normal",
            ],
            "age": [25, 30, 35, 40, 45, 50],
        }
    )
    return {"pheno": pheno, "expr": pd.DataFrame(np.random.randn(10, 6))}


@pytest.fixture
def multi_disease_dataset():
    """Dataset with multiple disease types and ambiguous samples."""
    pheno = pd.DataFrame(
        {
            "diagnosis": [
                "SLE",
                "Rheumatoid Arthritis",
                "Healthy Control",
                "SLE",
                "Lupus nephritis",
                "Other autoimmune",
                "Healthy Control",
            ],
        }
    )
    return {"pheno": pheno}


# ------------------------------------------------------------------
# Basic functionality
# ------------------------------------------------------------------
class TestBasicLabeling:
    """Tests for assign_class_labels()'s default two-class disease/control split."""

    def test_simple_two_class(self, sle_dataset):
        """Disease vs control labeling with default control_terms=None."""
        result = assign_class_labels(
            sle_dataset,
            column_name="disease state:ch1",
            disease_terms=["SLE"],
        )
        expected = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(result["class"], expected)

    def test_returns_same_dict(self, sle_dataset):
        """assign_class_labels mutates and returns the input dict."""
        result = assign_class_labels(
            sle_dataset,
            column_name="disease state:ch1",
            disease_terms=["SLE"],
        )
        assert result is sle_dataset

    def test_case_insensitive_matching(self, sle_dataset):
        """Matching is case-insensitive."""
        result = assign_class_labels(
            sle_dataset,
            column_name="disease state:ch1",
            disease_terms=["sle"],  # lowercase
        )
        assert result["class"].sum() == 3.0

    def test_substring_matching(self, sle_dataset):
        """Matching uses substring search, not exact match."""
        result = assign_class_labels(
            sle_dataset,
            column_name="disease state:ch1",
            disease_terms=["Lupus"],  # substring of full disease name
        )
        assert result["class"].sum() == 3.0

    def test_multiple_disease_terms(self, multi_disease_dataset):
        """Multiple disease terms match as OR."""
        result = assign_class_labels(
            multi_disease_dataset,
            column_name="diagnosis",
            disease_terms=["SLE", "Lupus"],
        )
        # SLE, SLE, Lupus nephritis = 3 cases
        assert result["class"][0] == 1.0
        assert result["class"][3] == 1.0
        assert result["class"][4] == 1.0
        assert np.nansum(result["class"] == 0.0) == 4  # all others are control


# ------------------------------------------------------------------
# Explicit control terms
# ------------------------------------------------------------------
class TestExplicitControlTerms:
    """Tests for assign_class_labels() when control_terms is explicitly given."""

    def test_explicit_control_terms(self, multi_disease_dataset):
        """With control_terms, only matching samples become controls."""
        result = assign_class_labels(
            multi_disease_dataset,
            column_name="diagnosis",
            disease_terms=["SLE", "Lupus"],
            control_terms=["Healthy"],
        )
        # Cases: SLE(0), SLE(3), Lupus nephritis(4) = 1.0
        # Controls: Healthy Control(2), Healthy Control(6) = 0.0
        # Others: RA(1), Other autoimmune(5) = NaN
        assert result["class"][0] == 1.0
        assert result["class"][2] == 0.0
        assert np.isnan(result["class"][1])
        assert np.isnan(result["class"][5])

    def test_nan_for_unmatched_with_control_terms(self, multi_disease_dataset):
        """Unmatched samples are NaN when control_terms is specified."""
        result = assign_class_labels(
            multi_disease_dataset,
            column_name="diagnosis",
            disease_terms=["SLE"],
            control_terms=["Healthy"],
        )
        nan_count = np.isnan(result["class"]).sum()
        assert nan_count == 3  # RA, Lupus nephritis, Other autoimmune


# ------------------------------------------------------------------
# Edge cases and error paths
# ------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases and error paths for assign_class_labels()."""

    def test_whitespace_stripping(self):
        """Leading/trailing whitespace in pheno values is stripped."""
        ds = {"pheno": pd.DataFrame({"dx": ["  SLE  ", " Normal "]})}
        result = assign_class_labels(ds, column_name="dx", disease_terms=["SLE"])
        np.testing.assert_array_equal(result["class"], [1.0, 0.0])

    def test_missing_pheno_raises(self):
        """Dataset without 'pheno' raises ValueError."""
        with pytest.raises(ValueError, match="must contain a 'pheno'"):
            assign_class_labels({}, column_name="dx", disease_terms=["SLE"])

    def test_missing_column_raises(self):
        """Column not found in pheno raises ValueError listing available columns."""
        ds = {"pheno": pd.DataFrame({"age": [25]})}
        with pytest.raises(ValueError, match="Column 'dx' not found"):
            assign_class_labels(ds, column_name="dx", disease_terms=["SLE"])

    def test_no_disease_matches_raises(self):
        """No disease matches raises ValueError."""
        ds = {"pheno": pd.DataFrame({"dx": ["Normal", "Normal"]})}
        with pytest.raises(ValueError, match="no samples matched disease_terms"):
            assign_class_labels(ds, column_name="dx", disease_terms=["SLE"])

    def test_no_control_matches_raises(self):
        """All samples matching disease with explicit empty controls raises."""
        ds = {"pheno": pd.DataFrame({"dx": ["SLE", "Lupus"]})}
        with pytest.raises(ValueError, match="no control samples found"):
            assign_class_labels(
                ds,
                column_name="dx",
                disease_terms=["SLE", "Lupus"],
                control_terms=["Healthy"],
            )

    def test_numeric_values_coerced_to_string(self):
        """Non-string pheno values are coerced via str() before matching."""
        ds = {"pheno": pd.DataFrame({"code": [1, 0, 1, 0]})}
        result = assign_class_labels(ds, column_name="code", disease_terms=["1"])
        np.testing.assert_array_equal(result["class"], [1.0, 0.0, 1.0, 0.0])

    def test_single_sample_disease(self):
        """Works with only one disease sample (edge minimum)."""
        ds = {"pheno": pd.DataFrame({"dx": ["SLE", "Normal", "Normal"]})}
        result = assign_class_labels(ds, column_name="dx", disease_terms=["SLE"])
        assert result["class"].sum() == 1.0


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
class TestLogging:
    """Tests for assign_class_labels()'s INFO-level logging."""

    def test_logs_class_counts(self, sle_dataset, caplog):
        """INFO-level log includes case and control counts."""
        with caplog.at_level(logging.INFO, logger="metaintegrator.labeling"):
            assign_class_labels(
                sle_dataset,
                column_name="disease state:ch1",
                disease_terms=["SLE"],
                dataset_name="GSE11909",
            )
        assert any("3 cases" in msg and "3 controls" in msg for msg in caplog.messages)

    def test_dataset_name_in_log(self, sle_dataset, caplog):
        """dataset_name appears in log messages."""
        with caplog.at_level(logging.INFO, logger="metaintegrator.labeling"):
            assign_class_labels(
                sle_dataset,
                column_name="disease state:ch1",
                disease_terms=["SLE"],
                dataset_name="GSE11909_GPL96",
            )
        assert any("GSE11909_GPL96" in msg for msg in caplog.messages)

    def test_default_dataset_name(self, sle_dataset, caplog):
        """Without dataset_name, 'dataset' is used as label."""
        with caplog.at_level(logging.INFO, logger="metaintegrator.labeling"):
            assign_class_labels(
                sle_dataset,
                column_name="disease state:ch1",
                disease_terms=["SLE"],
            )
        assert any("dataset:" in msg for msg in caplog.messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
