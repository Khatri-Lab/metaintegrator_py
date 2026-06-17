"""
MetaIntegrator Python Package
Enhanced with comprehensive GEO data acquisition and processing capabilities.

This package provides production-grade implementations of meta-analysis tools
for gene expression data, including:
- GEO data download and processing
- Gene annotation and symbol mapping
- Meta-analysis statistical methods
- Visualization and reporting tools

Python port of R MetaIntegrator package
Original R package by Francesco Vallania, Andrew Tam, Ravi Shankar, Aditya M. Rao
"""

# Core modules
from .core_analysis import *

# Analysis and scoring modules
from .analysis import *
from .scoring_core import *
from .scoring_meta_integrator_orchestrator import *

# Visualization modules
try:
    from visualization.enhanced_plotting import MetaAnalysisPlotter
except ImportError:
    pass

# Data validation (comprehensive framework)
from .validation import (
    DatasetValidator,
    MetaAnalysisValidator,
    MetaObjectValidator,
    DataObjectValidator,
    ValidationResult,
    check_data_object,
    validate_and_report,
)

# Data processing (transformation operations)
from .data_processing import DataProcessor
from .sample_ops import subset_original_data, subsetOriginalData
from .labeling import assign_class_labels

# Biological annotation
from .biological_annotation import *

# Specialized analysis modules
from .power_calculation import *
from .prc_analysis import *

# Integration modules
from .coconut_integration import *

__version__ = "0.0.1"
__author__ = "Khatri Lab - Python port of R MetaIntegrator"

# Main functions for easy access
__all__ = [
    # GEO Data Acquisition (NEW - Core functionality)
    "get_geo_data",
    "GEODataLoader",
    "check_geo_log_scale",
    # Gene Annotation (NEW - Core functionality)
    "GeneAnnotationEngine",
    "GEOPlatformAnnotator",
    "annotate_geo_dataset",
    "validate_geo_gene_symbols",
    # Core meta-analysis functions
    "run_meta_analysis",
    "MetaAnalysisCore",
    "EffectSizeResult",
    "InverseVarianceEngine",
    "FixedEffectsEngine",
    "DatasetValidator",
    "DataProcessor",
    # Validation framework
    "DatasetValidator",
    "MetaAnalysisValidator",
    "MetaObjectValidator",
    "DataObjectValidator",
    "ValidationResult",
    "check_data_object",
    "validate_and_report",
    # Data processing
    "DataProcessor",
    "subset_original_data",
    "subsetOriginalData",
    "assign_class_labels",
    # Data loading and validation
    "assess_data_quality",
    # Gene symbol correction
    "GeneSymbolCorrector",
    "correct_gene_symbols",
    # Analysis and scoring
    "calculate_score",
    "filter_genes",
    "FilterResult",
    "MetaAnalyzer",
    "MetaIntegratorOrchestrator",
    # Visualization
    "create_forest_plot",
    "create_violin_plot",
    "create_heatmap_plot",
    # Validation
    "validate_meta_object",
    "check_data_object",
    # Biological annotation
    "BiologicalAnnotator",
    "enrich_pathways",
    # Specialized analyses
    "ImmuneDeconvolution",
    "PowerCalculator",
    "ROCPRCAnalyzer",
]
