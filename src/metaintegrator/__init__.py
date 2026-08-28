"""
MetaIntegrator Python Package

This package provides production-grade implementations of the core
meta-analysis, filtering, scoring, power, and ROC/PRC statistical tools
for gene expression data. It is one of several top-level packages that
make up metaIntegrator-py:

- ``preprocessing`` — GEO data download/loading and gene annotation
  (``DataLoaderFactory``, ``GeneAnnotator``, ``EnsemblGeneValidator``, ...)
- ``metaintegrator`` (this package) — meta-analysis engine, gene filtering,
  scoring, power calculation, ROC/PRC analysis, COCONUT integration
- ``visualization`` — plotting (forest, volcano, heatmap, violin, ROC/PRC)
- ``coconut`` — the COCONUT batch-normalization algorithm itself

Python port of R MetaIntegrator package.
Original R package by Winston A. Haynes , Francesco Vallania , Aurelie Tomczak , Timothy Sweeney , Erika Bongen ,
Aditya M. Rao , Purvesh Khatri (Khatri Lab, Stanford University).
"""

# Analysis and scoring modules
from .analysis import *

# Core modules
from .core_analysis import *
from .scoring_core import *
from .scoring_meta_integrator_orchestrator import *

# Visualization modules (cross-package — resolved via src/ on sys.path)
# NOTE: visualization.enhanced_plotting's plotter class was renamed from
# MetaAnalysisPlotter to EnhancedMetaAnalysisPlotter to stop colliding with
# the unrelated visualization.plotting.MetaAnalysisPlotter class — see
# visualization/__init__.py's module docstring.
try:
    from visualization.enhanced_plotting import EnhancedMetaAnalysisPlotter

    _HAS_ENHANCED_PLOTTER = True
except ImportError:
    _HAS_ENHANCED_PLOTTER = False

# Biological annotation
from .biological_annotation import *

# Integration modules
from .coconut_integration import *

# Data processing (transformation operations)
from .data_processing import DataProcessor
from .labeling import assign_class_labels

# Specialized analysis modules
from .power_calculation import *
from .prc_analysis import *
from .sample_ops import subset_original_data, subsetOriginalData

# Data validation (comprehensive framework)
from .validation import (
    DataObjectValidator,
    DatasetValidator,
    MetaAnalysisValidator,
    MetaObjectValidator,
    ValidationResult,
    check_data_object,
    validate_and_report,
)

__version__ = "0.1.0"
__author__ = "Khatri Lab - Python port of R MetaIntegrator"

# Main functions for easy access.
#
# Every name below is verified to be bound in this package's namespace by
# the imports above. GEO data loading and gene annotation are NOT part of
# this list because they live in the separate ``preprocessing`` package
# (see ``preprocessing.loader_factory.DataLoaderFactory`` and
# ``preprocessing.gene_annotator.GeneAnnotator``); plotting utilities beyond
# ``EnhancedMetaAnalysisPlotter`` live in the separate ``visualization``
# package (see ``visualization.enhanced_plotting``).
__all__ = [
    # Core meta-analysis engine (core_analysis.py)
    "MetaAnalysisCore",
    "FisherCombination",
    # Meta-analysis orchestration (analysis.py)
    "MetaAnalyzer",
    "MetaAnalysisFactory",
    "MetaAnalysisEngine",
    "InverseVarianceEngine",
    "RandomEffectsEngine",
    "FixedEffectsEngine",
    "EffectSizeResult",
    "FilterResult",
    "run_meta_analysis",
    "filter_genes",
    # Validation framework (validation.py)
    "DatasetValidator",
    "MetaAnalysisValidator",
    "MetaObjectValidator",
    "DataObjectValidator",
    "ValidationResult",
    "check_data_object",
    "validate_and_report",
    # Data processing / labeling (data_processing.py, sample_ops.py, labeling.py)
    "DataProcessor",
    "subset_original_data",
    "subsetOriginalData",
    "assign_class_labels",
    # Biological annotation (biological_annotation.py)
    "BiologicalAnnotator",
    "impute_sex",
    # Statistical power analysis (power_calculation.py)
    "MetaAnalysisPowerCalculator",
    "PowerAnalysisResult",
    "PowerAnalysisPlotter",
    "PowerAnalysisSummary",
    "calc_meta_power",
    "print_power_summary",
    # ROC/PRC analysis (prc_analysis.py)
    "PRCAnalyzer",
    "plot_roc_curve",
    "plot_prc_curve",
    "multiple_roc_plot",
    "multiple_prc_plot",
    "calculate_auc_ci",
    "calculate_auprc_ci",
    # COCONUT batch-correction integration (coconut_integration.py)
    "COCONUTIntegrator",
    "COCONUTDataProcessor",
    "MetaIntegratorToCOCONUT",
    "SampleLevelGeneExtractor",
    "run_coconut_analysis",
    "coconut_meta_integrator",
    "gene_level_meta",
    "get_sample_level_gene_data",
    # Gene-signature scoring orchestration (scoring_meta_integrator_orchestrator.py)
    "MetaIntegratorOrchestrator",
    "EnhancedMetaIntegratorOrchestrator",
    "OrchestratorConfig",
    "OrchestrationResults",
    # Scoring subsystem building blocks (scoring_core.py)
    "FilterConfig",
    "ScoringConfig",
    "PathwayConfig",
    "SummaryConfig",
    "FilterResults",
    "ScoringResults",
    "create_default_filter_config",
    "create_strict_filter_config",
    "create_default_scoring_config",
    "create_default_pathway_config",
    "create_default_summary_config",
]

# Only advertise the plotter if the cross-package import above succeeded,
# so `from metaintegrator import *` never references an unbound name.
if _HAS_ENHANCED_PLOTTER:
    __all__.append("EnhancedMetaAnalysisPlotter")
