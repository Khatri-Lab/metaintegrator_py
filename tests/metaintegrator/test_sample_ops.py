"""
Tests for metaintegrator.sample_ops — the canonical location for
sample subsetting after the data_processing split.

The comprehensive subsetting logic tests live in
``test_subset_original_data.py`` (which uses the backward-compat import
path ``metaintegrator.data_processing.DataProcessor``).  This file
covers:

- Canonical import path works
- pandas Series class vector handling
- list class vector handling
- Pheno index mismatch fallback (iloc path)
- Non-DataFrame pheno warning
- numpy integer array as keep_samples
- Class vector length mismatch
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.sample_ops import subset_original_data, subsetOriginalData


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture
def standard_dataset():
    """Standard 50-gene × 6-sample dataset."""
    n_genes, n_samples = 50, 6
    sample_names = [f"S{i}" for i in range(n_samples)]
    expr = pd.DataFrame(
        np.random.randn(n_genes, n_samples),
        index=[f"Gene_{i}" for i in range(n_genes)],
        columns=sample_names,
    )
    return {
        "expr": expr,
        "class": np.array([0, 0, 0, 1, 1, 1]),
        "keys": [f"Gene_{i}" for i in range(n_genes)],
    }


# ------------------------------------------------------------------
# Canonical import path
# ------------------------------------------------------------------
class TestCanonicalImport:

    def test_subset_original_data_importable(self):
        """subset_original_data is importable from metaintegrator.sample_ops."""
        assert callable(subset_original_data)

    def test_subsetOriginalData_importable(self):
        """R-compat wrapper is importable from metaintegrator.sample_ops."""
        assert callable(subsetOriginalData)

    def test_basic_subset_via_canonical_path(self, standard_dataset):
        """Basic subsetting works via the canonical import."""
        result = subset_original_data(standard_dataset, ["S0", "S3"])
        assert result["expr"].shape[1] == 2
        assert list(result["expr"].columns) == ["S0", "S3"]


# ------------------------------------------------------------------
# Class vector type handling
# ------------------------------------------------------------------
class TestClassVectorTypes:

    def test_series_class_vector(self, standard_dataset):
        """pd.Series class vector is converted to numpy array."""
        standard_dataset["class"] = pd.Series([0, 0, 0, 1, 1, 1])
        result = subset_original_data(standard_dataset, ["S0", "S5"])
        np.testing.assert_array_equal(result["class"], [0, 1])

    def test_list_class_vector(self, standard_dataset):
        """list class vector is converted to numpy array."""
        standard_dataset["class"] = [0, 0, 0, 1, 1, 1]
        result = subset_original_data(standard_dataset, ["S0", "S5"])
        np.testing.assert_array_equal(result["class"], [0, 1])

    def test_class_vector_length_mismatch_raises(self, standard_dataset):
        """Class vector length mismatch with expr columns raises ValueError."""
        standard_dataset["class"] = np.array([0, 1])  # only 2, need 6
        with pytest.raises(ValueError, match="Class vector length"):
            subset_original_data(standard_dataset, ["S0"])


# ------------------------------------------------------------------
# Pheno edge cases
# ------------------------------------------------------------------
class TestPhenoHandling:

    def test_pheno_index_mismatch_falls_back_to_iloc(self, standard_dataset):
        """When pheno index doesn't match sample names, iloc is used."""
        standard_dataset["pheno"] = pd.DataFrame(
            {"age": [20, 25, 30, 35, 40, 45]},
            index=["X0", "X1", "X2", "X3", "X4", "X5"],  # doesn't match S0..S5
        )
        result = subset_original_data(standard_dataset, [0, 3])
        assert result["pheno"].shape[0] == 2
        # Values should come from positions 0 and 3
        assert result["pheno"]["age"].tolist() == [20, 35]

    def test_non_dataframe_pheno_warns(self, standard_dataset):
        """Non-DataFrame pheno triggers a warning and is kept as-is."""
        standard_dataset["pheno"] = {"raw": "metadata"}
        with pytest.warns(UserWarning, match="not a DataFrame"):
            result = subset_original_data(standard_dataset, ["S0"])
        assert result["pheno"] == {"raw": "metadata"}


# ------------------------------------------------------------------
# numpy array keep_samples
# ------------------------------------------------------------------
class TestNumpyArrayKeepSamples:

    def test_numpy_integer_array(self, standard_dataset):
        """numpy array of integers is treated as indices."""
        indices = np.array([1, 3, 5])
        result = subset_original_data(standard_dataset, indices)
        assert result["expr"].shape[1] == 3
        assert list(result["expr"].columns) == ["S1", "S3", "S5"]

    def test_numpy_boolean_all_false_raises(self, standard_dataset):
        """All-false boolean mask produces empty result — valid but 0 samples."""
        mask = np.array([False] * 6)
        result = subset_original_data(standard_dataset, mask)
        assert result["expr"].shape[1] == 0

    def test_numpy_boolean_single_true(self, standard_dataset):
        """Single True in boolean mask selects exactly one sample."""
        mask = np.array([False, False, True, False, False, False])
        result = subset_original_data(standard_dataset, mask)
        assert list(result["expr"].columns) == ["S2"]


# ------------------------------------------------------------------
# Type error paths
# ------------------------------------------------------------------
class TestTypeErrors:

    def test_non_dict_dataset_raises(self):
        """Non-dict dataset raises TypeError."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            subset_original_data("not_a_dict", ["S0"])

    def test_non_dataframe_non_array_expr_raises(self):
        """expr that is neither DataFrame nor ndarray raises TypeError."""
        ds = {"expr": [[1, 2], [3, 4]], "class": [0, 1]}
        with pytest.raises(TypeError, match="expr must be"):
            subset_original_data(ds, [0])

    def test_mixed_types_in_keep_samples_raises(self, standard_dataset):
        """Mixed types in keep_samples list raises TypeError."""
        with pytest.raises(TypeError, match="all strings.*or all integers"):
            subset_original_data(standard_dataset, ["S0", 1])


# ------------------------------------------------------------------
# Data independence (deep copy)
# ------------------------------------------------------------------
class TestDataIndependence:

    def test_subset_expr_is_independent_copy(self, standard_dataset):
        """Modifying subset expr doesn't affect original."""
        result = subset_original_data(standard_dataset, ["S0", "S1"])
        result["expr"].iloc[0, 0] = 999.0
        assert standard_dataset["expr"].iloc[0, 0] != 999.0

    def test_subset_class_is_independent_copy(self, standard_dataset):
        """Modifying subset class vector doesn't affect original."""
        result = subset_original_data(standard_dataset, ["S0", "S1"])
        result["class"][0] = 99
        assert standard_dataset["class"][0] != 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
