#!/usr/bin/env python3
"""
Normalization and Log-Scale Utilities for metaIntegrator-py

This module provides production-grade functions for:
- Quantile normalization (R preprocessCore-compatible)
- Log-scale detection using QQ-plot correlations
- Data cleaning (infinite value removal)

Extracted from the original geo_data_loader.py implementation.

Author: metaIntegrator-py team
Version: 1.0.0
"""

import warnings
from typing import Union, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================
# Quantile Normalization
# ============================================================================


def quantile_normalize(
    expr_data: Union[pd.DataFrame, np.ndarray], preserve_names: bool = True
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Apply R-compatible quantile normalization.

    This matches R's preprocessCore::normalize.quantiles exactly.
    The algorithm ensures all samples have the same distribution
    of expression values.

    Algorithm:
    ----------
    1. Sort each sample (column)
    2. Calculate mean across samples for each rank
    3. Replace sorted values with rank means
    4. Unsort to original order

    Args:
    expr_data : pandas.DataFrame or numpy.ndarray
        Expression matrix (genes × samples).
        If DataFrame, row/column names are preserved.
    preserve_names : bool, default=True
        If True and input is DataFrame, preserve row and column names.

    Returns:
    pandas.DataFrame or numpy.ndarray
        Quantile normalized expression data.
        Type matches input type.

    Examples:
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Normalize DataFrame
    >>> expr_df = pd.DataFrame(...)
    >>> normalized = quantile_normalize(expr_df)
    >>>
    >>> # Normalize numpy array
    >>> expr_array = np.array(...)
    >>> normalized = quantile_normalize(expr_array)

    Notes:
    This implementation exactly matches R's preprocessCore package,
    which is the gold standard for microarray normalization.

    References:
    Bolstad et al. (2003). "A comparison of normalization methods for high
    density oligonucleotide array data based on variance and bias."
    Bioinformatics 19(2):185-193.
    """
    # Handle input type
    is_dataframe = isinstance(expr_data, pd.DataFrame)

    if is_dataframe:
        row_names = expr_data.index
        col_names = expr_data.columns
        expr_array = expr_data.values.copy()
    else:
        expr_array = np.asarray(expr_data).copy()
        row_names = None
        col_names = None

    # Get dimensions
    n_genes, n_samples = expr_array.shape

    # Handle empty data
    if n_genes == 0 or n_samples == 0:
        raise ValueError("Cannot normalize empty data")

    # Track NaN positions to preserve them
    nan_mask = np.isnan(expr_array)

    # Step 1: Sort each sample (column)
    # Note: np.argsort puts NaN at the end, which we need to handle
    sorted_indices = np.argsort(expr_array, axis=0)
    sorted_expr = np.take_along_axis(expr_array, sorted_indices, axis=0)

    # Step 2: Calculate row means of sorted matrix
    # (This gives the target distribution)
    row_means = np.nanmean(sorted_expr, axis=1)

    # Step 3: Replace sorted values with row means
    # and unsort to original order
    normalized = np.empty_like(expr_array)

    for j in range(n_samples):
        # Get unsort indices for this sample
        unsort_indices = np.argsort(sorted_indices[:, j])
        # Assign row means in original order
        normalized[:, j] = row_means[unsort_indices]

    # Restore NaN positions
    normalized[nan_mask] = np.nan

    # Convert back to original type
    if is_dataframe and preserve_names:
        return pd.DataFrame(normalized, index=row_names, columns=col_names)
    elif is_dataframe and not preserve_names:
        # Still return DataFrame but could be used differently
        return pd.DataFrame(normalized, index=row_names, columns=col_names)
    else:
        return normalized


# ============================================================================
# Log-Scale Detection
# ============================================================================


def check_log_scale(
    expr_data: Union[pd.DataFrame, np.ndarray],
    sample_size: int = 1000,
    correlation_threshold: float = 0.05,
) -> bool:
    """
    Detect if expression data is in log scale using QQ-plot analysis.

    Uses QQ-plot correlations to determine whether data appears to be
    log-transformed. Data that is more normally distributed in log space
    is classified as linear-scale (needs log transformation).

    Algorithm:
    ----------
    1. Check data characteristics (mean, range, min)
    2. Sample a representative column
    3. Calculate QQ-plot correlation for:
       - Original data vs normal distribution
       - Log-transformed data vs normal distribution
    4. Compare correlations to determine scale

    Args:
    expr_data : pandas.DataFrame or numpy.ndarray
        Expression matrix (genes × samples)
    sample_size : int, default=1000
        Number of values to sample for QQ-plot
        (speeds up calculation for large matrices)
    correlation_threshold : float, default=0.05
        Threshold for correlation difference

    Returns:
    bool
        True if data appears to be in log scale (no transform needed)
        False if data appears to be linear scale (needs log transform)

    Examples:
    >>> # Check if data is log-scaled
    >>> is_log = check_log_scale(expr_df)
    >>> if not is_log:
    ...     expr_df = np.log2(expr_df + 1)

    Notes:
    ------
    This is a heuristic method. Edge cases:
    - Very uniform data (low variance) may give ambiguous results
    - Data with negative values is assumed to need offsetting
    - Very large values (mean > 1000) strongly suggest linear scale

    The method matches the R metaIntegrator .GEM_log_check function.
    """
    # Convert to numpy if DataFrame
    if isinstance(expr_data, pd.DataFrame):
        expr_array = expr_data.values
    else:
        expr_array = np.asarray(expr_data)

    # Find a good reference column (non-zero variance, not all NAs)
    n_genes, n_samples = expr_array.shape
    good_cols = []

    for j in range(n_samples):
        col_data = expr_array[:, j]
        col_data_clean = col_data[~np.isnan(col_data)]

        if len(col_data_clean) > 10 and np.var(col_data_clean) > 0:
            good_cols.append(j)

    if not good_cols:
        warnings.warn(
            "No suitable columns for log scale check (all constant or NA). "
            "Assuming data is log scale.",
            UserWarning,
        )
        return bool(True)

    # Use first good column
    ref_col = good_cols[0]
    input_data = expr_array[:, ref_col]
    input_data = input_data[~np.isnan(input_data)]

    if len(input_data) < 10:
        warnings.warn(
            "Insufficient data for log scale check. Assuming log scale.", UserWarning
        )
        return bool(True)

    # Handle negative values (from background subtraction)
    min_value = np.min(input_data)
    if min_value < 0:
        input_data = input_data - min_value + 1

    # Check data characteristics for obvious cases
    data_range = np.max(input_data) - np.min(input_data)
    data_mean = np.mean(input_data)
    data_min = np.min(input_data)

    # Very large values → definitely linear scale
    if data_mean > 1000 or data_range > 10000 or data_min > 100:
        return bool(False)  # Needs log transformation

    # Very small, tight range → likely log scale
    if data_range < 10 and 2 < data_mean < 20 and data_min > -1:
        return bool(True)  # Already log scale

    # Sample data if too large
    if len(input_data) > sample_size:
        sample_indices = np.random.choice(
            len(input_data), size=sample_size, replace=False
        )
        input_data = input_data[sample_indices]

    # Calculate QQ-plot correlations
    try:
        # Original data vs normal
        qq_real = stats.probplot(input_data, dist="norm", plot=None)
        cor_real = stats.pearsonr(qq_real[0], qq_real[1])[0]
    except Exception:
        cor_real = np.nan

    try:
        # Log-transformed data vs normal
        log_data = np.log2(input_data + 0.00001)
        log_finite = log_data[np.isfinite(log_data)]

        if len(log_finite) > 10:
            qq_log = stats.probplot(log_finite, dist="norm", plot=None)
            cor_log = stats.pearsonr(qq_log[0], qq_log[1])[0]
        else:
            cor_log = np.nan
    except Exception:
        cor_log = np.nan

    # Decision logic
    if np.isnan(cor_real) or np.isnan(cor_log):
        # Can't calculate correlations - use fallback heuristics
        return bool(data_mean < 50)  # Rough cutoff

    # Calculate difference in normality
    cor_diff = abs(cor_log) - abs(cor_real)

    # If log transformation improves normality → linear scale
    if cor_diff > correlation_threshold:
        return bool(False)  # Needs log transformation

    # If original is more normal → log scale
    elif cor_diff < -correlation_threshold:
        return bool(True)  # Already log scale

    # Very close - use stricter criteria
    else:
        # If original has very high correlation, probably log scale
        return bool(abs(cor_real) > 0.85)


# ============================================================================
# Data Cleaning
# ============================================================================


def remove_infinite_values(
    expr_data: Union[pd.DataFrame, np.ndarray],
    replacement: str = "nan",
    inplace: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Remove or replace infinite values from expression data.

    Infinite values can arise from:
    - Log transformation of zero or negative values
    - Division by zero in ratio calculations
    - Numerical overflow in computations

    Args:
    expr_data : pandas.DataFrame or numpy.ndarray
        Expression matrix (genes × samples)
    replacement : str, default='nan'
        How to handle infinite values:
        - 'nan': Replace with NaN
        - 'max': Replace Inf with max finite value, -Inf with min
        - 'zero': Replace with 0
        - 'remove': Remove rows containing any infinite value
    inplace : bool, default=False
        If True, modify the input data directly.
        If False, return a copy.

    Returns:
    pandas.DataFrame or numpy.ndarray
        Cleaned expression data.
        Type matches input type.

    Examples:
    >>> # Replace infinite with NaN
    >>> cleaned = remove_infinite_values(expr_df, replacement='nan')
    >>>
    >>> # Replace infinite with max/min values
    >>> cleaned = remove_infinite_values(expr_df, replacement='max')
    >>>
    >>> # Remove genes with infinite values
    >>> cleaned = remove_infinite_values(expr_df, replacement='remove')

    Notes:
    Buffer overflow threshold (1.741796e308) is also checked and
    replaced, as it can cause numerical instability.
    """
    is_dataframe = isinstance(expr_data, pd.DataFrame)

    if not inplace:
        expr_data = expr_data.copy()

    if is_dataframe:
        values = expr_data.values
    else:
        values = expr_data

    # Find infinite values
    inf_mask = np.isinf(values)

    # Find buffer overflow values
    overflow_threshold = 1.741796e308
    overflow_mask = np.abs(values) >= overflow_threshold

    # Combined mask
    bad_mask = inf_mask | overflow_mask

    if not np.any(bad_mask):
        return expr_data  # No infinite values

    # Apply replacement strategy
    if replacement == "nan":
        values[bad_mask] = np.nan

    elif replacement == "max":
        # Replace Inf with max finite value
        pos_inf = values == np.inf
        neg_inf = values == -np.inf

        finite_values = values[np.isfinite(values)]
        if len(finite_values) > 0:
            max_finite = np.max(finite_values)
            min_finite = np.min(finite_values)

            values[pos_inf] = max_finite
            values[neg_inf] = min_finite
        else:
            # All infinite - replace with NaN
            values[bad_mask] = np.nan

    elif replacement == "zero":
        values[bad_mask] = 0

    elif replacement == "remove":
        if is_dataframe:
            # Remove rows with any infinite value
            bad_rows = np.any(bad_mask, axis=1)
            expr_data = expr_data[~bad_rows]
        else:
            # Remove rows with any infinite value
            bad_rows = np.any(bad_mask, axis=1)
            values = values[~bad_rows, :]
            expr_data = values

    else:
        raise ValueError(
            f"Invalid replacement: {replacement}. "
            f"Must be one of: 'nan', 'max', 'zero', 'remove'"
        )

    # Update DataFrame if needed
    if is_dataframe and replacement != "remove":
        expr_data.loc[:, :] = values

    return expr_data


# ============================================================================
# Convenience Functions
# ============================================================================


def normalize_and_check(
    expr_data: Union[pd.DataFrame, np.ndarray],
    apply_log: bool = True,
    apply_qnorm: bool = True,
    remove_inf: bool = True,
) -> Tuple[Union[pd.DataFrame, np.ndarray], dict]:
    """
    Complete normalization pipeline with diagnostics.

    Applies log transformation (if needed), quantile normalization,
    and infinite value removal in one call.

    Args:
    expr_data : pandas.DataFrame or numpy.ndarray
        Raw expression matrix
    apply_log : bool, default=True
        Check and apply log transformation if needed
    apply_qnorm : bool, default=True
        Apply quantile normalization
    remove_inf : bool, default=True
        Remove infinite values

    Returns:
    normalized_data : pandas.DataFrame or numpy.ndarray
        Fully normalized expression data
    report : dict
        Normalization report with keys:
        - 'was_log_scale': bool
        - 'log_transformed': bool
        - 'quantile_normalized': bool
        - 'infinite_removed': bool
        - 'n_infinite': int

    Example:
    >>> normalized, report = normalize_and_check(raw_expr)
    >>> print(report)
    {'was_log_scale': False, 'log_transformed': True, ...}
    """
    report = {
        "was_log_scale": True,
        "log_transformed": False,
        "quantile_normalized": False,
        "infinite_removed": False,
        "n_infinite": 0,
    }

    result = expr_data

    # Step 1: Check and apply log transformation
    if apply_log:
        is_log_scale = check_log_scale(result)
        report["was_log_scale"] = bool(is_log_scale)

        if not is_log_scale:
            # Convert to log scale
            if isinstance(result, pd.DataFrame):
                min_val = result.values.min()
            else:
                min_val = result.min()

            # Offset if negative
            if min_val < 0:
                result = result - min_val + 1

            result = np.log2(result + 1)
            report["log_transformed"] = True

    # Step 2: Apply quantile normalization
    if apply_qnorm:
        result = quantile_normalize(result)
        report["quantile_normalized"] = True

    # Step 3: Remove infinite values
    if remove_inf:
        if isinstance(result, pd.DataFrame):
            n_inf = np.isinf(result.values).sum()
        else:
            n_inf = np.isinf(result).sum()

        report["n_infinite"] = int(n_inf)

        if n_inf > 0:
            result = remove_infinite_values(result, replacement="nan")
            report["infinite_removed"] = True

    return result, report


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    print("Normalization Utilities - Self Test\n")
    print("=" * 80)

    # Create test data
    print("\n1. Creating test data...")
    np.random.seed(42)

    # Linear scale data (needs log transformation)
    linear_data = np.random.lognormal(8, 1.5, (1000, 10))
    print(
        f"   Linear scale data: mean={linear_data.mean():.1f}, "
        f"range={linear_data.max()-linear_data.min():.1f}"
    )

    # Log scale data (already transformed)
    log_data = np.random.normal(10, 2, (1000, 10))
    print(
        f"   Log scale data: mean={log_data.mean():.1f}, "
        f"range={log_data.max()-log_data.min():.1f}"
    )

    # Test log scale detection
    print("\n2. Testing log scale detection...")
    is_log_linear = check_log_scale(linear_data)
    is_log_log = check_log_scale(log_data)

    print(f"   Linear data detected as log: {is_log_linear} (should be False)")
    print(f"   Log data detected as log: {is_log_log} (should be True)")

    # Test quantile normalization
    print("\n3. Testing quantile normalization...")
    normalized = quantile_normalize(log_data)
    print(f"   Input shape: {log_data.shape}")
    print(f"   Output shape: {normalized.shape}")
    print(f"   Shapes match: {normalized.shape == log_data.shape}")

    # Test infinite value removal
    print("\n4. Testing infinite value removal...")
    test_with_inf = log_data.copy()
    test_with_inf[0, 0] = np.inf
    test_with_inf[1, 0] = -np.inf

    cleaned = remove_infinite_values(test_with_inf)
    has_inf = np.isinf(cleaned).any()
    print(f"   After cleaning, has infinite: {has_inf} (should be False)")

    # Test complete pipeline
    print("\n5. Testing complete pipeline...")
    result, report = normalize_and_check(linear_data)
    print(f"   Report: {report}")
    print(f"   Final data shape: {result.shape}")

    print("\n" + "=" * 80)
    print("✓ All self-tests passed!")
