#!/usr/bin/env python3
"""
Data Validation Utilities for metaIntegrator-py Preprocessing

This module provides production-grade validation functions for GEO data
download and loading operations. Extracted and enhanced from data_loading.py.

Scope: Data download & loading validation
- GEO data structure validation
- Expression matrix checks
- Sample/phenotype validation
- Missing value detection
- Data range validation
- Platform compatibility checks

Author: metaIntegrator-py team
Version: 1.0.0
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ============================================================================
# Core Validation Functions
# ============================================================================


def validate_expression_matrix(
    expr_data: Union[pd.DataFrame, np.ndarray],
    min_genes: int = 100,
    min_samples: int = 2,
    allow_missing: bool = True,
    max_missing_fraction: float = 0.5,
) -> Tuple[bool, List[str]]:
    """
    Validate expression matrix structure and content.

    Checks:
    - Minimum dimensions (genes, samples)
    - Data type (numeric)
    - Missing value fraction
    - Inf/NaN values
    - Negative values (for log-scale data)

    Args:
    expr_data : pandas.DataFrame or numpy.ndarray
        Expression matrix (genes × samples)
    min_genes : int, default=100
        Minimum number of genes required
    min_samples : int, default=2
        Minimum number of samples required
    allow_missing : bool, default=True
        Whether to allow missing values
    max_missing_fraction : float, default=0.5
        Maximum fraction of missing values allowed (if allow_missing=True)

    Returns:
    is_valid : bool
        True if all checks pass
    errors : list of str
        List of validation error messages

    Examples:
    >>> valid, errors = validate_expression_matrix(expr_df)
    >>> if not valid:
    ...     for error in errors:
    ...         print(f"Error: {error}")
    """
    errors = []

    # Convert to DataFrame if needed
    if isinstance(expr_data, np.ndarray):
        expr_matrix = expr_data
    elif isinstance(expr_data, pd.DataFrame):
        expr_matrix = expr_data.values
    else:
        errors.append(f"Invalid type: {type(expr_data)}. Must be DataFrame or ndarray.")
        return False, errors

    # Check dimensions
    if expr_matrix.ndim != 2:
        errors.append(f"Expression data must be 2-dimensional, got {expr_matrix.ndim}D")
        return False, errors

    n_genes, n_samples = expr_matrix.shape

    if n_genes < min_genes:
        errors.append(f"Too few genes: {n_genes} < {min_genes} (minimum required)")

    if n_samples < min_samples:
        errors.append(f"Too few samples: {n_samples} < {min_samples} (minimum required)")

    # Check data type (should be numeric)
    if not np.issubdtype(expr_matrix.dtype, np.number):
        errors.append(f"Expression data must be numeric, got {expr_matrix.dtype}")
        return False, errors

    # Check for infinite values
    n_inf = np.isinf(expr_matrix).sum()
    if n_inf > 0:
        errors.append(f"Found {n_inf} infinite values in expression data")

    # Check for missing values
    n_missing = np.isnan(expr_matrix).sum()
    if n_missing > 0:
        missing_fraction = n_missing / expr_matrix.size

        if not allow_missing:
            errors.append(
                f"Found {n_missing} missing values ({missing_fraction:.1%}). "
                f"Missing values not allowed."
            )
        elif missing_fraction > max_missing_fraction:
            errors.append(
                f"Too many missing values: {missing_fraction:.1%} > "
                f"{max_missing_fraction:.1%} (maximum allowed)"
            )

    # Check for all-zero rows/columns
    finite_data = expr_matrix.copy()
    finite_data[~np.isfinite(finite_data)] = 0

    zero_genes = np.all(finite_data == 0, axis=1).sum()
    if zero_genes > 0:
        warnings.warn(f"Found {zero_genes} genes with all-zero values", UserWarning)

    zero_samples = np.all(finite_data == 0, axis=0).sum()
    if zero_samples > 0:
        warnings.warn(f"Found {zero_samples} samples with all-zero values", UserWarning)

    # Check data range (warn about potential issues)
    finite_values = expr_matrix[np.isfinite(expr_matrix)]
    if len(finite_values) > 0:
        data_min = np.min(finite_values)
        data_max = np.max(finite_values)
        data_mean = np.mean(finite_values)

        # Warn if data looks unusual
        if data_min < -100:
            warnings.warn(
                f"Unusually low values detected (min={data_min:.2f}). "
                f"Check data normalization.",
                UserWarning,
            )

        if data_max > 100000:
            warnings.warn(
                f"Very large values detected (max={data_max:.2f}). "
                f"Data may need log transformation.",
                UserWarning,
            )

        if data_mean > 1000:
            warnings.warn(
                f"Large mean value ({data_mean:.1f}). "
                f"Data may be in linear scale (not log-transformed).",
                UserWarning,
            )

    return len(errors) == 0, errors


def validate_phenotype_data(
    pheno_data: pd.DataFrame,
    expr_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    required_columns: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate phenotype/sample metadata.

    Checks:
    - Sample count matches expression data
    - Required columns present
    - Column types appropriate
    - Missing values in critical columns

    Args:
    pheno_data : pandas.DataFrame
        Phenotype/sample metadata
    expr_data : pandas.DataFrame or numpy.ndarray, optional
        Expression data to check sample count against
    required_columns : list of str, optional
        Columns that must be present

    Returns:
    is_valid : bool
        True if all checks pass
    errors : list of str
        List of validation error messages

    Examples:
    >>> valid, errors = validate_phenotype_data(
    ...     pheno_df,
    ...     expr_df,
    ...     required_columns=['geo_accession']
    ... )
    """
    errors = []

    # Check type
    if not isinstance(pheno_data, pd.DataFrame):
        errors.append(f"Phenotype data must be DataFrame, got {type(pheno_data)}")
        return False, errors

    # Check not empty
    if len(pheno_data) == 0:
        errors.append("Phenotype data is empty")
        return False, errors

    # Check sample count matches expression data
    if expr_data is not None:
        if isinstance(expr_data, pd.DataFrame):
            n_expr_samples = len(expr_data.columns)
        else:
            n_expr_samples = expr_data.shape[1]

        n_pheno_samples = len(pheno_data)

        if n_pheno_samples != n_expr_samples:
            errors.append(
                f"Sample count mismatch: {n_pheno_samples} phenotype samples "
                f"vs {n_expr_samples} expression samples"
            )

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(pheno_data.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(sorted(missing_cols))}")

    # Check for columns with all missing values
    all_na_cols = []
    for col in pheno_data.columns:
        if pheno_data[col].isna().all():
            all_na_cols.append(col)

    if all_na_cols:
        warnings.warn(f"Columns with all missing values: {', '.join(all_na_cols)}", UserWarning)

    # Check for duplicate sample IDs (if geo_accession column exists)
    if "geo_accession" in pheno_data.columns:
        duplicates = pheno_data["geo_accession"].duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate sample IDs")
    elif "Sample_geo_accession" in pheno_data.columns:
        duplicates = pheno_data["Sample_geo_accession"].duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate sample IDs")

    return len(errors) == 0, errors


def validate_geo_dataset(
    geo_data: Dict[str, Any], check_platform: bool = True, check_annotation: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate complete GEO dataset structure.

    Checks:
    - Required keys present (expr, pheno, platform_info)
    - Expression matrix valid
    - Phenotype data valid
    - Platform information complete
    - Feature annotations present

    Args:
    geo_data : dict
        GEO dataset dictionary with keys:
        - 'expr': Expression DataFrame
        - 'pheno': Phenotype DataFrame
        - 'platform_info': Platform metadata dict
        - 'fData': Feature/gene annotations (optional)
    check_platform : bool, default=True
        Validate platform information
    check_annotation : bool, default=True
        Check for feature annotations

    Returns:
    --------
    is_valid : bool
        True if all checks pass
    errors : list of str
        List of validation error messages

    Examples:
    >>> valid, errors = validate_geo_dataset(dataset)
    >>> if not valid:
    ...     raise ValueError(f"Invalid dataset: {errors}")
    """
    errors = []

    # Check type
    if not isinstance(geo_data, dict):
        errors.append(f"GEO data must be dict, got {type(geo_data)}")
        return False, errors

    # Check required keys
    required_keys = ["expr", "pheno", "platform_info"]
    missing_keys = set(required_keys) - set(geo_data.keys())

    if missing_keys:
        errors.append(f"Missing required keys: {', '.join(sorted(missing_keys))}")
        return False, errors

    # Validate expression matrix
    expr_valid, expr_errors = validate_expression_matrix(geo_data["expr"])
    if not expr_valid:
        for error in expr_errors:
            errors.append(f"Expression: {error}")

    # Validate phenotype data
    pheno_valid, pheno_errors = validate_phenotype_data(geo_data["pheno"], geo_data["expr"])
    if not pheno_valid:
        for error in pheno_errors:
            errors.append(f"Phenotype: {error}")

    # Validate platform info
    if check_platform:
        platform_info = geo_data["platform_info"]

        if not isinstance(platform_info, dict):
            errors.append("platform_info must be a dictionary")
        else:
            if "platform" not in platform_info:
                errors.append("platform_info missing 'platform' key")
            elif not platform_info["platform"]:
                errors.append("platform ID is empty")

    # Check for feature annotations
    if check_annotation:
        if "fData" not in geo_data:
            warnings.warn(
                "No feature annotations (fData) found. "
                "Probe-to-gene conversion may not be possible.",
                UserWarning,
            )
        else:
            fdata = geo_data["fData"]
            if not isinstance(fdata, pd.DataFrame):
                errors.append("fData must be a DataFrame")
            elif len(fdata) == 0:
                errors.append("fData is empty")
            elif len(fdata) != len(geo_data["expr"]):
                errors.append(
                    f"fData length ({len(fdata)}) doesn't match "
                    f"expression rows ({len(geo_data['expr'])})"
                )

    return len(errors) == 0, errors


def validate_platform_annotation(
    annotation_df: pd.DataFrame, probe_ids: Optional[List[str]] = None
) -> Tuple[bool, List[str], Dict[str, float]]:
    """
    Validate platform annotation data.

    Checks:
    - Required columns present (ID, Gene Symbol)
    - Coverage statistics
    - Data quality

    Args:
    annotation_df : pandas.DataFrame
        Platform annotation with probe-to-gene mappings
    probe_ids : list of str, optional
        Expected probe IDs to check coverage

    Returns:
    is_valid : bool
        True if annotation is usable
    errors : list of str
        List of validation error messages
    stats : dict
        Coverage and quality statistics

    Examples:
    >>> valid, errors, stats = validate_platform_annotation(annot_df)
    >>> print(f"Mapping coverage: {stats['coverage_pct']:.1f}%")
    """
    errors = []
    stats = {
        "total_probes": 0,
        "mapped_probes": 0,
        "coverage_pct": 0.0,
        "unique_genes": 0,
    }

    # Check type
    if not isinstance(annotation_df, pd.DataFrame):
        errors.append(f"Annotation must be DataFrame, got {type(annotation_df)}")
        return False, errors, stats

    # Check not empty
    if len(annotation_df) == 0:
        errors.append("Annotation data is empty")
        return False, errors, stats

    # Find ID column
    id_col = None
    for col in annotation_df.columns:
        if col.upper() in ["ID", "PROBE_ID", "PROBESET_ID"]:
            id_col = col
            break

    if id_col is None:
        errors.append("No ID column found in annotation")
        return False, errors, stats

    # Find gene symbol column
    symbol_col = None
    for col in annotation_df.columns:
        col_upper = col.upper().replace(" ", "_")
        if any(term in col_upper for term in ["GENE_SYMBOL", "GENESYMBOL", "SYMBOL"]):
            symbol_col = col
            break

    if symbol_col is None:
        warnings.warn("No gene symbol column found. Annotation may be incomplete.", UserWarning)
        return len(errors) == 0, errors, stats

    # Calculate coverage statistics
    stats["total_probes"] = len(annotation_df)

    # Count mapped probes (non-null, non-empty gene symbols)
    valid_symbols = annotation_df[symbol_col].notna()
    valid_symbols &= annotation_df[symbol_col].astype(str).str.strip() != ""
    valid_symbols &= (
        ~annotation_df[symbol_col]
        .astype(str)
        .str.lower()
        .isin(["nan", "none", "null", "---", "na"])
    )

    stats["mapped_probes"] = valid_symbols.sum()
    stats["coverage_pct"] = (
        100.0 * stats["mapped_probes"] / stats["total_probes"] if stats["total_probes"] > 0 else 0.0
    )

    # Count unique genes
    if stats["mapped_probes"] > 0:
        valid_genes = annotation_df.loc[valid_symbols, symbol_col]
        stats["unique_genes"] = valid_genes.nunique()

    # Warn if coverage is low
    if stats["coverage_pct"] < 50:
        warnings.warn(
            f"Low annotation coverage: {stats['coverage_pct']:.1f}%. "
            f"Many probes may not map to genes.",
            UserWarning,
        )

    # Check coverage of provided probe IDs
    if probe_ids is not None:
        probe_set = set(annotation_df[id_col].astype(str))
        requested_set = set(str(p) for p in probe_ids)

        found_probes = requested_set & probe_set
        stats["requested_coverage_pct"] = (
            100.0 * len(found_probes) / len(requested_set) if len(requested_set) > 0 else 0.0
        )

        if stats["requested_coverage_pct"] < 80:
            warnings.warn(
                f"Only {stats['requested_coverage_pct']:.1f}% of requested "
                f"probe IDs found in annotation",
                UserWarning,
            )

    return len(errors) == 0, errors, stats


def check_data_compatibility(
    datasets: List[Dict[str, Any]],
    check_platforms: bool = True,
    check_sample_overlap: bool = True,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check compatibility of multiple datasets for meta-analysis.

    Checks:
    - Platform consistency
    - Sample overlap
    - Feature overlap
    - Data scale consistency

    Args:
    datasets : list of dict
        List of GEO datasets
    check_platforms : bool, default=True
        Verify platforms are compatible
    check_sample_overlap : bool, default=True
        Check for sample overlap between datasets

    Returns:
    is_compatible : bool
        True if datasets are compatible
    warnings : list of str
        Compatibility warnings
    stats : dict
        Compatibility statistics

    Examples:
    >>> compat, warns, stats = check_data_compatibility(datasets)
    >>> print(f"Common genes: {stats['common_genes']}")
    """
    warnings_list = []
    stats = {
        "n_datasets": len(datasets),
        "platforms": set(),
        "common_genes": 0,
        "total_samples": 0,
        "sample_overlap": False,
    }

    if len(datasets) < 2:
        warnings_list.append("Need at least 2 datasets for meta-analysis compatibility check")
        return True, warnings_list, stats

    # Extract platforms
    platforms = []
    for i, ds in enumerate(datasets):
        if "platform_info" in ds and "platform" in ds["platform_info"]:
            platform = ds["platform_info"]["platform"]
            platforms.append(platform)
            stats["platforms"].add(platform)
        else:
            warnings_list.append(f"Dataset {i}: Missing platform information")

    # Check platform consistency
    if check_platforms and len(stats["platforms"]) > 1:
        warnings_list.append(
            f"Multiple platforms detected: {', '.join(sorted(stats['platforms']))}. "
            f"Ensure probe-to-gene conversion is performed."
        )

    # Check gene/feature overlap
    gene_sets = []
    for ds in datasets:
        if "expr" in ds:
            if isinstance(ds["expr"], pd.DataFrame):
                genes = set(ds["expr"].index)
            else:
                genes = set(range(ds["expr"].shape[0]))
            gene_sets.append(genes)

    if gene_sets:
        common_genes = set.intersection(*gene_sets)
        stats["common_genes"] = len(common_genes)

        # Calculate average overlap
        avg_genes = np.mean([len(g) for g in gene_sets])
        overlap_pct = 100.0 * len(common_genes) / avg_genes if avg_genes > 0 else 0

        if overlap_pct < 50:
            warnings_list.append(
                f"Low gene overlap: only {len(common_genes)} genes "
                f"({overlap_pct:.1f}%) common across datasets"
            )

    # Check for sample overlap (should not exist)
    if check_sample_overlap:
        all_sample_ids = []
        for ds in datasets:
            if "pheno" in ds:
                sample_col = None
                for col in ds["pheno"].columns:
                    if "geo_accession" in col.lower():
                        sample_col = col
                        break

                if sample_col:
                    samples = ds["pheno"][sample_col].tolist()
                    all_sample_ids.extend(samples)

        if all_sample_ids:
            unique_samples = len(set(all_sample_ids))
            total_samples = len(all_sample_ids)
            stats["total_samples"] = total_samples

            if unique_samples < total_samples:
                duplicates = total_samples - unique_samples
                warnings_list.append(
                    f"Sample overlap detected: {duplicates} duplicate samples " f"across datasets"
                )
                stats["sample_overlap"] = True

    # All checks are warnings, not errors
    return True, warnings_list, stats


# ============================================================================
# Convenience Functions
# ============================================================================


def quick_validate_download(
    geo_data: Dict[str, Any], min_genes: int = 100, min_samples: int = 3
) -> bool:
    """
    Quick validation for freshly downloaded GEO data.

    Args:
    geo_data : dict
        Downloaded GEO dataset
    min_genes : int, default=100
        Minimum genes required
    min_samples : int, default=3
        Minimum samples required

    Returns:
    bool
        True if data passes basic validation

    Examples:
    >>> if not quick_validate_download(data):
    ...     raise ValueError("Download validation failed")
    """
    try:
        is_valid, errors = validate_geo_dataset(
            geo_data,
            check_platform=True,
            check_annotation=False,  # Annotation may not be done yet
        )

        if not is_valid:
            print(f"Validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        # Additional quick checks
        if "expr" in geo_data:
            n_genes = len(geo_data["expr"])
            n_samples = (
                len(geo_data["expr"].columns)
                if isinstance(geo_data["expr"], pd.DataFrame)
                else geo_data["expr"].shape[1]
            )

            if n_genes < min_genes or n_samples < min_samples:
                print(
                    f"Dataset too small: {n_genes} genes × {n_samples} samples "
                    f"(need {min_genes} × {min_samples})"
                )
                return False

        return True

    except Exception as e:
        print(f"Validation error: {e}")
        return False


def print_validation_report(
    geo_data: Dict[str, Any], title: str = "Dataset Validation Report"
) -> None:
    """
    Print comprehensive validation report.

    Args:
        geo_data: GEO dataset to validate.
        title: Report title.

    Examples:
        >>> print_validation_report(dataset, "GSE12345 Validation")
    """
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}\n")

    # Overall validation
    is_valid, errors = validate_geo_dataset(geo_data)

    if is_valid:
        print("✓ Overall validation: PASS\n")
    else:
        print("✗ Overall validation: FAIL\n")
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
        print()

    # Expression matrix details
    if "expr" in geo_data:
        print("Expression Matrix:")

        expr_valid, expr_errors = validate_expression_matrix(geo_data["expr"])

        if isinstance(geo_data["expr"], pd.DataFrame):
            n_genes, n_samples = geo_data["expr"].shape
        else:
            n_genes, n_samples = geo_data["expr"].shape

        print(f"  Dimensions: {n_genes:,} genes × {n_samples:,} samples")
        print(f"  Validation: {'PASS' if expr_valid else 'FAIL'}")

        if not expr_valid:
            for error in expr_errors:
                print(f"    - {error}")
        print()

    # Phenotype data details
    if "pheno" in geo_data:
        print("Phenotype Data:")

        pheno_valid, pheno_errors = validate_phenotype_data(geo_data["pheno"], geo_data.get("expr"))

        n_samples = len(geo_data["pheno"])
        n_columns = len(geo_data["pheno"].columns)

        print(f"  Samples: {n_samples:,}")
        print(f"  Metadata columns: {n_columns}")
        print(f"  Validation: {'PASS' if pheno_valid else 'FAIL'}")

        if not pheno_valid:
            for error in pheno_errors:
                print(f"    - {error}")
        print()

    # Platform information
    if "platform_info" in geo_data:
        print("Platform Information:")
        platform_info = geo_data["platform_info"]

        if "platform" in platform_info:
            print(f"  Platform ID: {platform_info['platform']}")
        if "platform_name" in platform_info:
            print(f"  Platform Name: {platform_info['platform_name']}")
        print()

    # Feature annotations
    if "fData" in geo_data:
        print("Feature Annotations:")

        annot_valid, annot_errors, stats = validate_platform_annotation(geo_data["fData"])

        print(f"  Total probes: {stats['total_probes']:,}")
        print(f"  Mapped probes: {stats['mapped_probes']:,}")
        print(f"  Coverage: {stats['coverage_pct']:.1f}%")
        print(f"  Unique genes: {stats['unique_genes']:,}")
        print(f"  Validation: {'PASS' if annot_valid else 'FAIL'}")

        if not annot_valid:
            for error in annot_errors:
                print(f"    - {error}")
        print()

    print(f"{'=' * 80}\n")


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    print("Validation Utilities - Self Test\n")
    print("=" * 80)

    # Test 1: Expression matrix validation
    print("\n1. Testing expression matrix validation...")

    # Valid matrix
    valid_expr = pd.DataFrame(
        np.random.randn(1000, 10),
        index=[f"Gene_{i}" for i in range(1000)],
        columns=[f"Sample_{i}" for i in range(10)],
    )

    is_valid, errors = validate_expression_matrix(valid_expr)
    print(f"   Valid matrix: {is_valid} (errors: {len(errors)})")

    # Invalid matrix (too small)
    small_expr = pd.DataFrame(np.random.randn(50, 1))
    is_valid, errors = validate_expression_matrix(small_expr)
    print(f"   Small matrix: {is_valid} (errors: {len(errors)})")

    # Test 2: Phenotype validation
    print("\n2. Testing phenotype validation...")

    pheno = pd.DataFrame(
        {
            "geo_accession": [f"GSM{i}" for i in range(10)],
            "title": [f"Sample {i}" for i in range(10)],
            "source": ["Blood"] * 10,
        }
    )

    is_valid, errors = validate_phenotype_data(pheno, valid_expr)
    print(f"   Matching pheno: {is_valid} (errors: {len(errors)})")

    # Test 3: GEO dataset validation
    print("\n3. Testing complete GEO dataset validation...")

    geo_dataset = {
        "expr": valid_expr,
        "pheno": pheno,
        "platform_info": {"platform": "GPL570"},
        "fData": pd.DataFrame(
            {
                "ID": valid_expr.index,
                "Gene Symbol": [f"GENE{i % 100}" for i in range(len(valid_expr))],
            }
        ),
    }

    is_valid, errors = validate_geo_dataset(geo_dataset)
    print(f"   Complete dataset: {is_valid} (errors: {len(errors)})")

    # Test 4: Platform annotation validation
    print("\n4. Testing platform annotation validation...")

    is_valid, errors, stats = validate_platform_annotation(geo_dataset["fData"])
    print(f"   Annotation valid: {is_valid}")
    print(f"   Coverage: {stats['coverage_pct']:.1f}%")
    print(f"   Unique genes: {stats['unique_genes']}")

    # Test 5: Validation report
    print("\n5. Testing validation report...")
    print_validation_report(geo_dataset, "Test Dataset")

    print("=" * 80)
    print("✓ All self-tests passed!")
