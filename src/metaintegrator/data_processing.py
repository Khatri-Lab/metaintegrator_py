"""
Data validation and processing for meta-analysis.

This module contains the :class:`DataProcessor` class for creating
combined annotation tables.  Sample-subsetting operations now live in
:mod:`sample_ops` and class-label assignment in :mod:`labeling`; both
are re-exported here so that existing ``from .data_processing import …``
lines continue to work.

Classes:
    DataProcessor: Creates combined annotation tables for meta-analysis.

Re-exported for backward compatibility:
    subset_original_data  (from .sample_ops)
    subsetOriginalData    (from .sample_ops)
    assign_class_labels   (from .labeling)
"""

from typing import Dict, List

import pandas as pd

from .labeling import assign_class_labels

# ---- backward-compatible re-exports --------------------------------
from .sample_ops import subset_original_data, subsetOriginalData

__all__ = [
    "DataProcessor",
    "subset_original_data",
    "subsetOriginalData",
    "assign_class_labels",
]


class DataProcessor:
    """Creates combined annotation tables for meta-analysis."""

    @staticmethod
    def create_combined_annotation_table(original_data: List[Dict]) -> pd.DataFrame:
        """
        Create combined annotation table from multiple datasets.

        Args:
            original_data: List of dataset dictionaries

        Returns:
            Combined annotation table
        """
        if not original_data:
            return pd.DataFrame()

        # Import here to avoid circular dependency
        from .core_analysis import MetaAnalysisCore

        ann_db = MetaAnalysisCore.create_annotation_table(original_data[0])

        # Add annotations from remaining datasets
        for i in range(1, len(original_data)):
            temp_ann_table = MetaAnalysisCore.create_annotation_table(original_data[i])

            # Find common probes and exclude them from new dataset
            common_probes = temp_ann_table["probeid"].isin(ann_db["probeid"])
            if common_probes.any():
                print(f"Found common probes in dataset {i}")
                temp_ann_table = temp_ann_table[~common_probes]

            ann_db = pd.concat([ann_db, temp_ann_table], ignore_index=True)

        # Set probeid as index and return symbol column
        ann_db = ann_db.set_index("probeid")
        return ann_db

    # Delegate to standalone function for backward compatibility
    subset_original_data = staticmethod(subset_original_data)
