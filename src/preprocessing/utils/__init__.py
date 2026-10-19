"""
Preprocessing Utilities Module

Provides validation, normalization, and data quality utilities
for GEO data preprocessing.
"""

from .log_scale import (
    LogScaleCheck,
    LogScaleReport,
    apply_r_log2,
    gem_log_check,
)
from .normalization import (
    check_log_scale,
    normalize_and_check,
    quantile_normalize,
    remove_infinite_values,
)
from .validation import (
    check_data_compatibility,
    print_validation_report,
    quick_validate_download,
    validate_expression_matrix,
    validate_geo_dataset,
    validate_phenotype_data,
    validate_platform_annotation,
)

__all__ = [
    # R-parity log-scale check (R getGEOData() / .GEM_log_check)
    "LogScaleCheck",
    "LogScaleReport",
    "apply_r_log2",
    "gem_log_check",
    # Normalization functions
    "quantile_normalize",
    "check_log_scale",
    "remove_infinite_values",
    "normalize_and_check",
    # Validation functions
    "validate_expression_matrix",
    "validate_phenotype_data",
    "validate_geo_dataset",
    "validate_platform_annotation",
    "check_data_compatibility",
    "quick_validate_download",
    "print_validation_report",
]
