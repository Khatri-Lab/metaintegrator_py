"""
Tests for the data_processing module split — backward compatibility.

After splitting data_processing.py into data_processing.py,
sample_ops.py, and labeling.py, all original import paths must
continue to work.  This file verifies that contract.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))


class TestBackwardCompatImports:
    """All legacy import paths still resolve."""

    def test_data_processor_from_data_processing(self):
        """DataProcessor is still importable from metaintegrator.data_processing."""
        from metaintegrator.data_processing import DataProcessor

        assert hasattr(DataProcessor, "create_combined_annotation_table")

    def test_subset_from_data_processing(self):
        """The R-compat subsetOriginalData alias is still importable from data_processing."""
        from metaintegrator.data_processing import subsetOriginalData

        assert callable(subsetOriginalData)

    def test_subset_original_data_from_data_processing(self):
        """subset_original_data is still importable from metaintegrator.data_processing."""
        from metaintegrator.data_processing import subset_original_data

        assert callable(subset_original_data)

    def test_assign_class_labels_from_data_processing(self):
        """assign_class_labels is still importable from metaintegrator.data_processing."""
        from metaintegrator.data_processing import assign_class_labels

        assert callable(assign_class_labels)

    def test_all_exports_listed(self):
        """data_processing.__all__ still lists every legacy public name."""
        from metaintegrator import data_processing

        expected = {
            "DataProcessor",
            "subset_original_data",
            "subsetOriginalData",
            "assign_class_labels",
        }
        assert expected <= set(data_processing.__all__)


class TestCanonicalImports:
    """New canonical import paths work."""

    def test_sample_ops_subset_original_data(self):
        """subset_original_data is importable from its new canonical module, sample_ops."""
        from metaintegrator.sample_ops import subset_original_data

        assert callable(subset_original_data)

    def test_sample_ops_subsetOriginalData(self):
        """subsetOriginalData is importable from its new canonical module, sample_ops."""
        from metaintegrator.sample_ops import subsetOriginalData

        assert callable(subsetOriginalData)

    def test_labeling_assign_class_labels(self):
        """assign_class_labels is importable from its new canonical module, labeling."""
        from metaintegrator.labeling import assign_class_labels

        assert callable(assign_class_labels)

    def test_sample_ops_has_all(self):
        """sample_ops.__all__ lists both subset_original_data and its R-compat alias."""
        from metaintegrator import sample_ops

        assert "subset_original_data" in sample_ops.__all__
        assert "subsetOriginalData" in sample_ops.__all__

    def test_labeling_has_all(self):
        """labeling.__all__ lists assign_class_labels."""
        from metaintegrator import labeling

        assert "assign_class_labels" in labeling.__all__


class TestPackageLevelImports:
    """Top-level package exports all names."""

    def test_package_data_processor(self):
        """DataProcessor is importable directly from the top-level metaintegrator package."""
        from metaintegrator import DataProcessor

        assert hasattr(DataProcessor, "create_combined_annotation_table")

    def test_package_subset_original_data(self):
        """subset_original_data is importable directly from the top-level package."""
        from metaintegrator import subset_original_data

        assert callable(subset_original_data)

    def test_package_subsetOriginalData(self):
        """subsetOriginalData is importable directly from the top-level package."""
        from metaintegrator import subsetOriginalData

        assert callable(subsetOriginalData)

    def test_package_assign_class_labels(self):
        """assign_class_labels is importable directly from the top-level package."""
        from metaintegrator import assign_class_labels

        assert callable(assign_class_labels)


class TestDataProcessorDelegate:
    """DataProcessor.subset_original_data still works as a static method."""

    def test_static_method_delegates(self):
        """DataProcessor.subset_original_data() still subsets expr/class correctly."""
        from metaintegrator.data_processing import DataProcessor

        ds = {
            "expr": pd.DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"]),
            "class": np.array([0, 1, 0]),
        }
        result = DataProcessor.subset_original_data(ds, ["A", "C"])
        assert list(result["expr"].columns) == ["A", "C"]
        np.testing.assert_array_equal(result["class"], [0, 0])

    def test_same_function_object(self):
        """DataProcessor.subset_original_data IS the sample_ops function."""
        from metaintegrator.data_processing import DataProcessor
        from metaintegrator.sample_ops import subset_original_data

        assert DataProcessor.subset_original_data is subset_original_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
