"""
Sample subsetting operations for meta-analysis datasets.

This module provides functions for selecting and filtering samples
within dataset objects, preserving alignment of expression matrices,
class vectors, and phenotype data.

Functions:
    subset_original_data: Subset samples from a dataset object.
    subsetOriginalData: R-compatible convenience wrapper.
"""

import warnings
from typing import Dict, List, Union

import numpy as np
import pandas as pd

__all__ = [
    "subset_original_data",
    "subsetOriginalData",
]


def subset_original_data(
    dataset_object: Dict,
    keep_samples: Union[List[str], List[int], np.ndarray],
) -> Dict:
    """
    Subset samples from a dataset object.

    Python port of R's subsetOriginalData() function. Subsets all relevant
    components within the dataset object to include only the desired samples.

    Args:
        dataset_object: Dataset object containing at least 'expr' and 'class'.
                       May also contain 'pheno' and other metadata.
        keep_samples: Samples to keep, specified as either:
                     - List of sample names (strings) matching column names in expr
                     - List of sample indices (integers, 0-based)
                     - Boolean numpy array (True = keep, False = drop)

    Returns:
        Subsetted dataset object with the same structure as input

    Raises:
        ValueError: If no valid samples are found or if input is malformed
        TypeError: If keep_samples is not a supported type
        KeyError: If required keys are missing from dataset_object
    """
    # Validate input dataset structure
    if not isinstance(dataset_object, dict):
        raise TypeError("dataset_object must be a dictionary")

    required_keys = ["expr", "class"]
    missing_keys = [k for k in required_keys if k not in dataset_object]
    if missing_keys:
        raise KeyError(f"dataset_object missing required keys: {missing_keys}")

    # Get expression matrix and validate
    expr_matrix = dataset_object["expr"]
    if not isinstance(expr_matrix, pd.DataFrame):
        # Try to convert to DataFrame if it's a numpy array
        if isinstance(expr_matrix, np.ndarray):
            expr_matrix = pd.DataFrame(expr_matrix)
        else:
            raise TypeError("expr must be a pandas DataFrame or numpy array")

    # Get class vector
    class_vector = dataset_object["class"]
    if isinstance(class_vector, pd.Series):
        class_vector = class_vector.values
    elif isinstance(class_vector, list):
        class_vector = np.array(class_vector)

    n_samples = expr_matrix.shape[1]

    # Validate class vector length
    if len(class_vector) != n_samples:
        raise ValueError(
            f"Class vector length ({len(class_vector)}) doesn't match "
            f"number of samples ({n_samples})"
        )

    # Process keep_samples based on type
    if isinstance(keep_samples, np.ndarray):
        if keep_samples.dtype == bool:
            # Boolean array
            if len(keep_samples) != n_samples:
                raise ValueError(
                    f"Boolean mask length ({len(keep_samples)}) doesn't match "
                    f"number of samples ({n_samples})"
                )
            sample_indices = np.where(keep_samples)[0]
            sample_names = expr_matrix.columns[sample_indices].tolist()
        else:
            # Numeric array - treat as indices
            keep_samples = keep_samples.tolist()
            sample_indices = keep_samples
            sample_names = expr_matrix.columns[sample_indices].tolist()

    elif isinstance(keep_samples, list):
        if len(keep_samples) == 0:
            raise ValueError("keep_samples cannot be empty")

        # Check if list contains strings (sample names) or integers (indices)
        if all(isinstance(x, str) for x in keep_samples):
            # List of sample names
            sample_names = keep_samples
            available_samples = set(expr_matrix.columns)

            # Check which samples are present
            valid_samples = []
            missing_samples = []

            for sample_name in sample_names:
                if sample_name in available_samples:
                    valid_samples.append(sample_name)
                else:
                    missing_samples.append(sample_name)

            # Warn about missing samples
            if missing_samples:
                warnings.warn(
                    f"The following samples were specified but not present: "
                    f"{', '.join(missing_samples)}"
                )

            # Check if any valid samples remain
            if not valid_samples:
                raise ValueError("No specified sample names are present in the dataset object.")

            sample_names = valid_samples
            # Get indices for valid samples
            sample_indices = [expr_matrix.columns.get_loc(name) for name in sample_names]

        elif all(isinstance(x, (int, np.integer)) for x in keep_samples):
            # List of sample indices
            sample_indices = keep_samples

            # Validate indices
            invalid_indices = [idx for idx in sample_indices if idx < 0 or idx >= n_samples]
            if invalid_indices:
                raise ValueError(
                    f"Invalid sample indices (must be 0-{n_samples - 1}): {invalid_indices}"
                )

            sample_names = expr_matrix.columns[sample_indices].tolist()
        else:
            raise TypeError(
                "keep_samples list must contain all strings (sample names) "
                "or all integers (sample indices)"
            )
    else:
        raise TypeError("keep_samples must be a list, numpy array, or pandas Series")

    # Create subsetted dataset object
    subsetted_dataset = {}

    # Subset expression matrix (by columns)
    subsetted_dataset["expr"] = expr_matrix.iloc[:, sample_indices].copy()

    # Subset class vector
    subsetted_dataset["class"] = class_vector[sample_indices].copy()

    # Subset pheno data if present
    if "pheno" in dataset_object:
        pheno_data = dataset_object["pheno"]
        if isinstance(pheno_data, pd.DataFrame):
            # Subset by sample names if pheno is a DataFrame
            try:
                subsetted_dataset["pheno"] = pheno_data.loc[sample_names, :].copy()
            except KeyError:
                # If sample names don't match pheno index, try subsetting by position
                subsetted_dataset["pheno"] = pheno_data.iloc[sample_indices, :].copy()
        else:
            # If pheno is not a DataFrame, just keep reference
            warnings.warn("pheno data is not a DataFrame, cannot subset - keeping reference")
            subsetted_dataset["pheno"] = pheno_data

    # Copy all other fields from original dataset
    for key, value in dataset_object.items():
        if key not in ["expr", "class", "pheno"]:
            subsetted_dataset[key] = value

    return subsetted_dataset


# Convenience function for R interface compatibility
def subsetOriginalData(dataset_object: Dict, keepMe: Union[List, np.ndarray]) -> Dict:
    """
    R-compatible convenience function for subsetting datasets.

    Subset samples for a particular dataset. Subsets all relevant slots within
    the Dataset object to include only the desired samples.

    Args:
        dataset_object: the Dataset object to subset
        keepMe: either a binary vector for whether each sample should be
               in the subset or a list of names of samples to be in the subset

    Returns:
        Dataset object that has been subsetted to the desired samples

    Examples:
        >>> subsetted = subsetOriginalData(
        ...     dataset,
        ...     keepMe=['Sample 1', 'Sample 13', 'Sample 43']
        ... )
    """
    return subset_original_data(dataset_object, keepMe)
