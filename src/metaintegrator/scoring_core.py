"""
Consolidated core module for scoring functionality.
This module provides common infrastructure, base classes, and utilities
shared across all scoring modules to eliminate duplication and standardize patterns.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ==================================================================
# EXCEPTION HIERARCHY
# ==================================================================


class ScoringError(Exception):
    """Base exception for all scoring operations."""

    pass


class ValidationError(ScoringError):
    """Raised when data validation fails."""

    pass


class ConfigurationError(ScoringError):
    """Raised when configuration validation fails."""

    pass


class FilteringError(ScoringError):
    """Raised when filtering operations fail."""

    pass


class ScoringCalculationError(ScoringError):
    """Raised when score calculations fail."""

    pass


# ==================================================================
# ENUMERATIONS
# ==================================================================


class ExportFormat(Enum):
    """Supported export formats."""

    CSV = "csv"
    TSV = "tsv"
    EXCEL = "xlsx"
    JSON = "json"
    PICKLE = "pkl"
    PARQUET = "parquet"


class StatisticType(Enum):
    """Types of statistical summaries."""

    BASIC = "basic"
    DETAILED = "detailed"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"


class ScoringMethod(Enum):
    """Available scoring methods."""

    Z_SCORE = "z_score"
    COMBINED = "combined"
    EFFECT_SIZE = "effect_size"
    FDR_INVERSE = "fdr_inverse"
    FISHER_MIN = "fisher_min"


class AggregationMethod(Enum):
    """Pathway aggregation methods."""

    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    SUM = "sum"
    WEIGHTED_MEAN = "weighted_mean"
    GSEA_LIKE = "gsea_like"


# ==================================================================
# BASE CONFIGURATION SYSTEM
# ==================================================================


# ------------------------------------------------------------------
# BaseConfig Class
# ------------------------------------------------------------------
@dataclass
class BaseConfig:
    """Common base for all configuration classes with standardized validation."""

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Override in subclasses for specific validation.

        Raises:
            ConfigurationError: If validation fails
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """Create configuration from dictionary.

        Args:
            data: A mapping of field names to values.
        """
        return cls(**data)


# ------------------------------------------------------------------
# FilterConfig Class
# ------------------------------------------------------------------
@dataclass
class FilterConfig(BaseConfig):
    """Configuration for filtering operations.

    Attributes:
        effect_size_thresh: Minimum effect size threshold.
        fdr_thresh: False discovery rate threshold, must be between 0 and 1.
        number_studies_thresh: Minimum number of studies required, must be at least 1.
        heterogeneity_pval_thresh: Heterogeneity p‑value threshold, must be between 0 and 1.
        is_leave_one_out: Whether to apply leave‑one‑out filtering.
    """

    effect_size_thresh: float = 0.0
    fdr_thresh: float = 0.05
    number_studies_thresh: int = 1
    heterogeneity_pval_thresh: float = 0.0
    is_leave_one_out: bool = True

    def validate(self) -> None:
        """Validate filter configuration parameters.

        Raises:
            ConfigurationError: If any parameter is outside its allowed range.
        """
        if not 0 <= self.fdr_thresh <= 1:
            raise ConfigurationError("FDR threshold must be between 0 and 1")
        if self.number_studies_thresh < 1:
            raise ConfigurationError("Number of studies threshold must be at least 1")
        if not 0 <= self.heterogeneity_pval_thresh <= 1:
            raise ConfigurationError("Heterogeneity p-value threshold must be between 0 and 1")


# ------------------------------------------------------------------
# ScoringConfig Class
# ------------------------------------------------------------------
@dataclass
class ScoringConfig(BaseConfig):
    """Configuration for scoring operations.

    Attributes:
        weights: Mapping of scoring components to their relative weights. The
        default weights are 0.3 for effect_size, 0.3 for effect_fdr,
        0.2 for fisher_score and 0.2 for heterogeneity.
        normalization_method: Method used to normalize scores. Valid
        options are 'min_max', 'z_score', and 'robust'.
        handle_missing: Strategy for dealing with missing values. Valid
        options are 'zero', 'median', and 'drop'.
        min_valid_genes: Minimum number of valid genes required for a
        score to be considered. Must be at least 1.
    """

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "effect_size": 0.3,
            "effect_fdr": 0.3,
            "fisher_score": 0.2,
            "heterogeneity": 0.2,
        }
    )
    normalization_method: str = "min_max"  # "min_max", "z_score", "robust"
    handle_missing: str = "zero"  # "zero", "median", "drop"
    min_valid_genes: int = 1

    def validate(self) -> None:
        """Validate scoring configuration.

        Raises:
            ConfigurationError: If any validation rule is violated.
        """
        # Validate weights
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            warnings.warn(f"Weights sum to {total:.3f}, not 1.0")

        if not all(0 <= w <= 1 for w in self.weights.values()):
            raise ConfigurationError("All weights must be between 0 and 1")

        # Validate other parameters
        if self.normalization_method not in ["min_max", "z_score", "robust"]:
            raise ConfigurationError("Invalid normalization method")

        if self.handle_missing not in ["zero", "median", "drop"]:
            raise ConfigurationError("Invalid missing value handling method")

        if self.min_valid_genes < 1:
            raise ConfigurationError("min_valid_genes must be at least 1")


# ------------------------------------------------------------------
# PathwayConfig Class
# ------------------------------------------------------------------
@dataclass
class PathwayConfig(BaseConfig):
    """Configuration for pathway‑level analysis.

    Attributes:
        aggregation_method: Method used to aggregate pathway scores.
        min_genes_per_pathway: Minimum number of genes required in a pathway.
        max_genes_per_pathway: Maximum number of genes allowed in a pathway.
        weight_by_significance: Whether to weight genes by significance.
        significance_column: Column name used for significance weighting.
        direction_sensitive: Whether pathway direction matters.
    """

    aggregation_method: AggregationMethod = AggregationMethod.MEAN
    min_genes_per_pathway: int = 3
    max_genes_per_pathway: int = 500
    weight_by_significance: bool = False
    significance_column: str = "effectSizeFDR"
    direction_sensitive: bool = True

    def validate(self) -> None:
        """Validate pathway configuration.

        Raises:
            ConfigurationError: If any validation rule is violated.
        """
        if self.min_genes_per_pathway < 1:
            raise ConfigurationError("min_genes_per_pathway must be at least 1")
        if self.max_genes_per_pathway < self.min_genes_per_pathway:
            raise ConfigurationError("max_genes_per_pathway must be >= min_genes_per_pathway")


# ------------------------------------------------------------------
# SummaryConfig Class
# ------------------------------------------------------------------
@dataclass
class SummaryConfig(BaseConfig):
    """Configuration for summary generation.

    Attributes:
        include_distributions: Whether to include distribution plots.
        include_correlations: Whether to include correlation analysis.
        significance_thresholds: List of significance cutoffs to use.
        effect_size_thresholds: List of effect‑size cutoffs to use.
        heterogeneity_threshold: Threshold for heterogeneity to report.
        min_studies_threshold: Minimum number of studies required to show in the summary.
    """

    include_distributions: bool = True
    include_correlations: bool = False
    significance_thresholds: List[float] = field(default_factory=lambda: [0.05, 0.01, 0.001])
    effect_size_thresholds: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.8])
    heterogeneity_threshold: float = 0.05
    min_studies_threshold: int = 2

    def validate(self) -> None:
        """Validate summary configuration.

        Raises:
            ConfigurationError: If thresholds are out of bounds.
        """
        if not all(0 <= t <= 1 for t in self.significance_thresholds):
            raise ConfigurationError


# ==================================================================
# COMMON DATA STRUCTURES
# ==================================================================


# ------------------------------------------------------------------
# OverlapAnalysis Class
# ------------------------------------------------------------------
@dataclass
class OverlapAnalysis:
    """Unified overlap analysis results between gene sets.

    This class merges the duplicate OverlapAnalysis implementations
    found in the filter_manager and filter_results_summarizer modules.
    It stores a variety of overlap information that can be queried
    through helper methods.
    """

    filter_names: List[str]
    positive_overlaps: Dict[str, Dict[str, int]]
    negative_overlaps: Dict[str, Dict[str, int]]
    jaccard_similarities: Dict[str, Dict[str, float]]
    unique_genes: Dict[str, Dict[str, Set[str]]]
    overlap_matrix: Optional[pd.DataFrame] = None
    jaccard_matrix: Optional[pd.DataFrame] = None
    overlap_statistics: Optional[Dict[str, Any]] = None
    set_statistics: Optional[Dict[str, Dict[str, Any]]] = None

    def get_overlap_matrix(self, gene_type: str = "positive") -> pd.DataFrame:
        """Return overlap matrix as DataFrame.

        Args:
            gene_type: Either 'positive' or 'negative' to select which
            overlap matrix to return. Defaults to 'positive'.

        Returns:
            pd.DataFrame: The requested overlap matrix. If the matrix
            has not been pre‑computed, it is generated from the
            stored overlap dictionaries.
        """
        if self.overlap_matrix is not None and gene_type == "positive":
            return self.overlap_matrix

        overlaps = self.positive_overlaps if gene_type == "positive" else self.negative_overlaps
        return pd.DataFrame(overlaps).fillna(0)

    def get_jaccard_matrix(self) -> pd.DataFrame:
        """Return Jaccard similarity matrix as DataFrame.

        Returns:
            pd.DataFrame: The Jaccard similarity matrix. If the matrix
            has not been pre‑computed, it is constructed from the
            stored similarity dictionaries.
        """
        if self.jaccard_matrix is not None:
            return self.jaccard_matrix
        return pd.DataFrame(self.jaccard_similarities).fillna(0)

    def get_most_similar_pairs(self, n: int = 5) -> List[Tuple[str, str, float]]:
        """Get most similar gene set pairs by Jaccard similarity.

        Args:
            n: Number of top pairs to return. Defaults to 5.

        Returns:
            List[Tuple[str, str, float]]: Each tuple contains the names
            of two gene sets and their Jaccard similarity score.
        """
        jaccard_matrix = self.get_jaccard_matrix()

        # Set diagonal to 0 to avoid self‑similarity
        jaccard_no_diag = jaccard_matrix.copy()
        np.fill_diagonal(jaccard_no_diag.values, 0)

        similarities = []
        for i in range(len(jaccard_no_diag)):
            for j in range(i + 1, len(jaccard_no_diag)):
                similarities.append(
                    (
                        jaccard_no_diag.index[i],
                        jaccard_no_diag.columns[j],
                        jaccard_no_diag.iloc[i, j],
                    )
                )

        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:n]


# ------------------------------------------------------------------
# FilterSummary Class
# ------------------------------------------------------------------
@dataclass
class FilterSummary:
    """Summary information for a single filter.

    Attributes:
        name: Identifier for the filter.
        n_positive_genes: Number of genes classified as positive.
        n_negative_genes: Number of genes classified as negative.
        total_genes: Total number of genes considered.
        parameters: Dictionary containing filter parameters.
        timestamp: Optional datetime when the filter was applied.
        description: Human readable description of the filter.
        positive_genes: List of positive gene names. Not shown in repr.
        negative_genes: List of negative gene names. Not shown in repr.

    Properties:
        gene_count_ratio: Ratio of positive to negative gene counts.
    """

    name: str
    n_positive_genes: int
    n_negative_genes: int
    total_genes: int
    parameters: Dict[str, Any]
    timestamp: Optional[datetime]
    description: str
    positive_genes: List[str] = field(repr=False)
    negative_genes: List[str] = field(repr=False)

    @property
    def gene_count_ratio(self) -> float:
        """Return ratio of positive to negative genes.

        Returns:
            float: Ratio of positive to negative genes. If there are
            no negative genes, returns infinity when positive genes
            exist, otherwise returns zero.
        """
        if self.n_negative_genes == 0:
            return float("inf") if self.n_positive_genes > 0 else 0
        return self.n_positive_genes / self.n_negative_genes


# ------------------------------------------------------------------
# StatisticalSummary Class
# ------------------------------------------------------------------
@dataclass
class StatisticalSummary:
    """Container for statistical summary results.

    Attributes:
        basic_stats: Summary statistics such as mean, median, etc.
        distribution_stats: Optional statistics describing the data
        distribution, e.g., histogram bins or density estimates.
        correlation_stats: Optional correlation matrix or related
        metrics.
        threshold_counts: Optional counts of items exceeding various
        thresholds.
        metadata: Dictionary holding additional information such
        as the date of computation or parameter settings.
    """

    basic_stats: Dict[str, Any]
    distribution_stats: Optional[Dict[str, Any]] = None
    correlation_stats: Optional[Dict[str, Any]] = None
    threshold_counts: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistical summary to dictionary format.

        Returns:
            dict: A mapping containing all stored statistics and
            metadata, ready for serialization or further processing.
        """
        return {
            "basic_stats": self.basic_stats,
            "distribution_stats": self.distribution_stats,
            "correlation_stats": self.correlation_stats,
            "threshold_counts": self.threshold_counts,
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------
# FilterResults Class
# ------------------------------------------------------------------
@dataclass
class FilterResults:
    """Container for filtering operation results.

    This class stores the output of gene filtering operations,
    including the lists of positive and negative genes,
    configuration used, and metadata about the filtering process.

    Attributes:
        pos_gene_names: List of gene identifiers classified as positive.
        neg_gene_names: List of gene identifiers classified as negative.
        config: Configuration object used for filtering.
        is_leave_one_out: Whether leave-one-out analysis was applied.
        filter_description: Human-readable description of the filter.
        timestamp: When the filtering was performed.
    """

    pos_gene_names: List[str] = field(default_factory=list)
    neg_gene_names: List[str] = field(default_factory=list)
    config: Optional[FilterConfig] = None
    is_leave_one_out: bool = False
    filter_description: str = "Unknown filter"
    timestamp: Optional[datetime] = None

    @property
    def total_genes(self) -> int:
        """Total number of filtered genes.

        Returns:
            int: Sum of positive and negative gene counts.
        """
        return len(self.pos_gene_names) + len(self.neg_gene_names)

    @property
    def positive_count(self) -> int:
        """Number of positive genes.

        Returns:
            int: Count of genes in pos_gene_names.
        """
        return len(self.pos_gene_names)

    @property
    def negative_count(self) -> int:
        """Number of negative genes.

        Returns:
            int: Count of genes in neg_gene_names.
        """
        return len(self.neg_gene_names)

    @property
    def direction_ratio(self) -> float:
        """Ratio of positive to negative genes.

        Returns:
            float: Ratio of positive to negative gene counts.
            Returns infinity if no negative genes exist but positive genes do,
            and 0 if no genes exist in either direction.
        """
        if self.negative_count == 0:
            return float("inf") if self.positive_count > 0 else 0.0
        return self.positive_count / self.negative_count

    def get_all_genes(self) -> List[str]:
        """Get all filtered genes (positive and negative).

        Returns:
            List[str]: Combined list of all gene identifiers.
        """
        return self.pos_gene_names + self.neg_gene_names

    def get_genes_by_direction(self, direction: str) -> List[str]:
        """Get genes for a specific direction.

        Args:
            direction: Either 'positive', 'negative', or 'both'.

        Returns:
            List[str]: Gene identifiers for the requested direction.

        Raises:
            ValueError: If direction is not recognized.
        """
        if direction == "positive":
            return self.pos_gene_names.copy()
        elif direction == "negative":
            return self.neg_gene_names.copy()
        elif direction == "both":
            return self.get_all_genes()
        else:
            raise ValueError(
                f"Unknown direction: {direction}. Use 'positive', 'negative', or 'both'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert filter results to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the filter results.
        """
        return {
            "pos_gene_names": self.pos_gene_names,
            "neg_gene_names": self.neg_gene_names,
            "config": self.config.to_dict() if self.config else None,
            "is_leave_one_out": self.is_leave_one_out,
            "filter_description": self.filter_description,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "total_genes": self.total_genes,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "direction_ratio": self.direction_ratio,
        }


# ------------------------------------------------------------------
# ScoringResults Class
# ------------------------------------------------------------------
@dataclass
class ScoringResults:
    """Container for gene scoring results.

    Attributes:
        scores: A pandas Series mapping gene identifiers to their
        computed scores.
        method: Name of the scoring method used.
        config: Optional ScoringConfig instance used to generate the
        scores.
        metadata: Dictionary for any additional information that may
        be useful for downstream analysis.
    """

    scores: pd.Series
    method: str
    config: Optional[ScoringConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_ranked_genes(self, ascending: bool = False) -> pd.DataFrame:
        """Get genes ranked by score.

        Args:
            ascending: If True, lower scores receive higher rank.
            Default is False to rank high scores first.

        Returns:
            pd.DataFrame: DataFrame indexed by gene identifiers with
            columns for score and rank.
        """
        ranked = pd.DataFrame({"gene": self.scores.index, "score": self.scores.values})
        ranked = ranked.sort_values("score", ascending=ascending)
        ranked["rank"] = range(1, len(ranked) + 1)
        return ranked.set_index("gene")

    def get_top_genes(self, n: int = 100, ascending: bool = False) -> List[str]:
        """Get top N genes by score.

        Args:
            n: Number of top genes to return. Default is 100.
            ascending: Same meaning as in get_ranked_genes.

        Returns:
            List[str]: List of gene identifiers ordered by score.
        """
        ranked = self.get_ranked_genes(ascending=ascending)
        return ranked.head(n).index.tolist()

    def get_percentile_threshold(self, percentile: float) -> float:
        """Get score threshold for given percentile.

        Args:
            percentile: Desired percentile (between 0 and 100).

        Returns:
            float: Score value at the specified percentile, ignoring
            missing values.
        """
        return np.percentile(self.scores.dropna(), percentile)


# ==================================================================
# UNIFIED VALIDATION SYSTEM
# ==================================================================


# ------------------------------------------------------------------
# BaseValidator Class
# ------------------------------------------------------------------
class BaseValidator(ABC):
    """Unified validation interface for all data types.

    This abstract base class defines a contract for validator
    implementations. Every concrete validator must provide a
    validate_and_extract method that checks the integrity of the
    input dictionary and returns the data as a pandas DataFrame.
    """

    @staticmethod
    @abstractmethod
    def validate_and_extract(data: Dict) -> pd.DataFrame:
        """Validate and extract data as DataFrame.

        Args:
            data: Dictionary containing raw input data to be validated.

        Returns:
            pd.DataFrame: Normalized representation of the input data.

        Raises:
            ValidationError: If the input data fails validation checks.
        """
        pass


# ------------------------------------------------------------------
# MetaAnalysisValidator Class
# ------------------------------------------------------------------
class MetaAnalysisValidator(BaseValidator):
    """Validator for meta‑analysis data with comprehensive checks.

    Attributes:
        REQUIRED_COLUMNS: List of column names that must be present in the
        pooled results.
        OPTIONAL_COLUMNS: List of column names that may be present. Missing
        optional columns will trigger a warning but not an error.
    """

    REQUIRED_COLUMNS = ["effectSize", "effectSizeStandardError"]
    OPTIONAL_COLUMNS = [
        "effectSizeFDR",
        "fisherFDRUp",
        "fisherFDRDown",
        "heterogeneityPval",
        "numStudies",
    ]

    @staticmethod
    def validate_and_extract(meta_analysis: Dict) -> pd.DataFrame:
        """Validate and extract meta‑analysis results.

        Args:
            meta_analysis: Dictionary containing meta‑analysis information.
            The dictionary must contain a key 'pooledResults' whose value
            is either a pandas DataFrame or a dictionary convertible to
            a DataFrame.

        Returns:
            pd.DataFrame: Normalized pooled results.

        Raises:
            ValidationError: If the input is not a dictionary,
            if 'pooledResults' is missing, if required columns are
            absent, or if the resulting DataFrame is empty.
        """
        if not isinstance(meta_analysis, dict):
            raise ValidationError("Meta-analysis must be a dictionary")

        if "pooledResults" not in meta_analysis:
            raise ValidationError("Meta-analysis must contain 'pooledResults'")

        results = meta_analysis["pooledResults"]
        if isinstance(results, dict):
            results = pd.DataFrame(results)

        # Check required columns
        missing_required = [
            col for col in MetaAnalysisValidator.REQUIRED_COLUMNS if col not in results.columns
        ]
        if missing_required:
            raise ValidationError(f"Missing required columns: {missing_required}")

        if len(results) == 0:
            raise ValidationError("Meta-analysis results are empty")

        # Warn about missing optional columns
        missing_optional = [
            col for col in MetaAnalysisValidator.OPTIONAL_COLUMNS if col not in results.columns
        ]
        if missing_optional:
            warnings.warn(f"Missing optional columns: {missing_optional}")

        return results

    @staticmethod
    def validate_dataframe(results: pd.DataFrame) -> None:
        """Validate a DataFrame containing meta-analysis results.

        This is an alias method for compatibility with test code.

        Args:
            results: DataFrame containing meta-analysis results to validate.

        Raises:
            ValidationError: If the DataFrame fails validation checks.
        """
        if not isinstance(results, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")

        # Check required columns
        missing_required = [
            col for col in MetaAnalysisValidator.REQUIRED_COLUMNS if col not in results.columns
        ]
        if missing_required:
            raise ValidationError(f"Missing required columns: {missing_required}")

        if len(results) == 0:
            raise ValidationError("Meta-analysis results are empty")

        # Warn about missing optional columns
        missing_optional = [
            col for col in MetaAnalysisValidator.OPTIONAL_COLUMNS if col not in results.columns
        ]
        if missing_optional:
            warnings.warn(f"Missing optional columns: {missing_optional}")


# ------------------------------------------------------------------
# FilterResultsValidator Class
# ------------------------------------------------------------------
class FilterResultsValidator(BaseValidator):
    """Validator for filter results.

    Attributes:
        REQUIRED_FIELDS: Names of fields that must be present in a
        filter result dictionary. Missing fields trigger a
        ValidationError.
        OPTIONAL_FIELDS: Mapping from optional field names to
        their default values. These fields are ignored if absent,
        but their defaults may be used by downstream code.
    """

    REQUIRED_FIELDS = ["posGeneNames", "negGeneNames"]
    OPTIONAL_FIELDS = {
        "FDRThresh": "Unknown",
        "effectSizeThresh": "Unknown",
        "numberStudiesThresh": "Unknown",
        "heterogeneityPvalThresh": "Unknown",
        "isLeaveOneOut": False,
        "timestamp": None,
        "filterDescription": "Unknown",
    }

    @staticmethod
    def validate_and_extract(filter_results: Dict) -> Dict:
        """Validate filter results structure.

        Args:
            filter_results: Dictionary containing the output of a
            filtering step. The dictionary must include all names in
            REQUIRED_FIELDS.

        Returns:
            dict: The original dictionary unchanged when validation
            succeeds.

        Raises:
            ValidationError: If filter_results is not a dictionary
            or if one or more required fields are missing.
        """
        if not isinstance(filter_results, dict):
            raise ValidationError("Filter results must be a dictionary")

        missing_required = [
            field for field in FilterResultsValidator.REQUIRED_FIELDS if field not in filter_results
        ]
        if missing_required:
            raise ValidationError(f"Missing required fields: {missing_required}")

        return filter_results


# ------------------------------------------------------------------
# MetaObjectValidator Class
# ------------------------------------------------------------------
class MetaObjectValidator(BaseValidator):
    """Validator for complete meta‑integrator objects.

    A meta‑integrator object is expected to encapsulate a meta‑analysis
    result under the key ``metaAnalysis`` and, within that
    dictionary, contain a ``pooledResults`` entry.  The validator
    simply verifies the presence of these keys; no conversion to a
    DataFrame is performed.

    Attributes:
        REQUIRED_FIELDS: Top‑level keys that must be present.
        REQUIRED_META_FIELDS: Keys that must be present inside the
        nested ``metaAnalysis`` dictionary.
    """

    REQUIRED_FIELDS = ["metaAnalysis"]
    REQUIRED_META_FIELDS = ["pooledResults"]

    @staticmethod
    def validate_and_extract(meta_object: Dict) -> Dict:
        """Validate meta‑integrator object structure.

        Args:
            meta_object: Dictionary representing the complete
            meta‑integrator object.  The dictionary must contain
            ``metaAnalysis`` and that nested dictionary must contain
            ``pooledResults``.

        Returns:
            dict: The original dictionary when validation succeeds.

        Raises:
            ValidationError: If the input is not a dictionary,
            or if any required top‑level or nested key is missing.
        """
        if not isinstance(meta_object, dict):
            raise ValidationError("Meta object must be a dictionary")

        for field in MetaObjectValidator.REQUIRED_FIELDS:
            if field not in meta_object:
                raise ValidationError(f"Meta object must contain '{field}' field")

        meta_analysis = meta_object["metaAnalysis"]
        if not isinstance(meta_analysis, dict):
            raise ValidationError("Meta-analysis must be a dictionary")

        for field in MetaObjectValidator.REQUIRED_META_FIELDS:
            if field not in meta_analysis:
                raise ValidationError(f"Meta-analysis must contain '{field}' field")

        return meta_object


# ==================================================================
# COMMON UTILITIES
# ==================================================================


# ------------------------------------------------------------------
# LoggerMixin Class
# ------------------------------------------------------------------
class LoggerMixin:
    """Mixin to provide consistent logging across all classes."""

    def __init__(self, logger: Optional[logging.Logger] = None, **kwargs):
        """
        Initialize the LoggerMixin.

        Args:
            logger: Optional logger instance. If not provided, a logger for the module is used.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        self.logger = logger or logging.getLogger(self.__class__.__module__)
        super().__init__(**kwargs)


# ------------------------------------------------------------------
# DataFrameUtils Class
# ------------------------------------------------------------------
class DataFrameUtils:
    """Utilities for DataFrame operations."""

    @staticmethod
    def ensure_dataframe(data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Convert data to DataFrame if needed.

        Args:
            data: Either a dictionary or a DataFrame.
        Returns:
            A DataFrame representation of the input.
        """
        if isinstance(data, dict):
            return pd.DataFrame(data)
        return data

    @staticmethod
    def safe_column_access(df: pd.DataFrame, column: str, default_value: Any = None) -> pd.Series:
        """
        Safely access a DataFrame column with a default value.

        Args:
            df: The DataFrame to query.
            column: The name of the column to access.
            default_value: Value to use if the column is missing.
        Returns:
            The requested column as a Series, or a Series of the default value.
        """
        if column in df.columns:
            return df[column]
        return pd.Series(default_value, index=df.index)

    @staticmethod
    def validate_columns(df: pd.DataFrame, required: List[str], optional: List[str] = None) -> None:
        """
        Validate that a DataFrame contains required columns and warn about missing optional ones.

        Args:
            df: The DataFrame to validate.
            required: A list of column names that must be present.
            optional: A list of column names that are optional.
        Raises:
            ValidationError: If any required column is missing.
        """
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValidationError(f"Missing required columns: {missing}")

        if optional:
            missing_opt = [col for col in optional if col not in df.columns]
            if missing_opt:
                warnings.warn(f"Missing optional columns: {missing_opt}")

    @staticmethod
    def get_numeric_columns(df: pd.DataFrame) -> List[str]:
        """
        Get list of numeric column names from DataFrame.

        Args:
            df: The DataFrame to analyze.

        Returns:
            List[str]: Names of columns with numeric dtypes.
        """
        return df.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def analyze_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing data patterns in a DataFrame.

        Args:
            df: DataFrame to analyze for missing data patterns.

        Returns:
            Dict[str, Any]: Dictionary containing missing data analysis for each column.
        """
        missing_analysis = {}

        for col in df.columns:
            total_values = len(df)
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / total_values) * 100 if total_values > 0 else 0

            missing_analysis[col] = {
                "missing_count": int(missing_count),
                "total_values": total_values,
                "missing_percentage": float(missing_percentage),
                "non_missing_count": total_values - missing_count,
            }

        # Overall summary
        total_missing = sum(info["missing_count"] for info in missing_analysis.values())
        total_values = sum(info["total_values"] for info in missing_analysis.values())

        missing_analysis["_summary"] = {
            "total_missing_values": total_missing,
            "total_possible_values": total_values,
            "overall_missing_percentage": (
                (total_missing / total_values * 100) if total_values > 0 else 0
            ),
            "columns_with_missing_data": sum(
                1
                for info in missing_analysis.values()
                if isinstance(info, dict) and info.get("missing_count", 0) > 0
            ),
            "columns_analyzed": len(df.columns),
        }

        return missing_analysis


# ------------------------------------------------------------------
# NormalizationUtils Class
# ------------------------------------------------------------------
class NormalizationUtils:
    """Utilities for data normalization."""

    @staticmethod
    def min_max_normalize(series: pd.Series) -> pd.Series:
        """
        Min‑max normalization to [0, 1].

        Args:
            series: The numeric series to normalize.
        Returns:
            A Series scaled to the [0, 1] interval.
        """
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)

    @staticmethod
    def z_score_normalize(series: pd.Series) -> pd.Series:
        """
        Z‑score normalization.

        Args:
            series: The numeric series to normalize.
        Returns:
            A Series with mean 0 and standard deviation 1.
        """
        return (series - series.mean()) / series.std()

    @staticmethod
    def robust_normalize(series: pd.Series) -> pd.Series:
        """
        Robust normalization using median and MAD.

        Args:
            series: The numeric series to normalize.
        Returns:
            A Series scaled by the median absolute deviation.
        """
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            return pd.Series(0, index=series.index)
        return (series - median) / mad

    @staticmethod
    def normalize_series(series: pd.Series, method: str = "min_max") -> pd.Series:
        """
        Normalize series using the specified method.

        Args:
            series: The numeric series to normalize.
            method: One of "min_max", "z_score", or "robust".
        Returns:
            A normalized Series.
        Raises:
            ValueError: If an unknown normalization method is supplied.
        """
        if method == "min_max":
            return NormalizationUtils.min_max_normalize(series)
        elif method == "z_score":
            return NormalizationUtils.z_score_normalize(series)
        elif method == "robust":
            return NormalizationUtils.robust_normalize(series)
        else:
            raise ValueError(f"Unknown normalization method: {method}")


# ------------------------------------------------------------------
# StatisticalUtils Class
# ------------------------------------------------------------------
class StatisticalUtils:
    """Utilities for statistical operations.

    This class provides a collection of static methods that perform
    common statistical calculations such as detecting outliers,
    computing Jaccard similarity, and generating overlap matrices.
    """

    @staticmethod
    def count_outliers(series: pd.Series, method: str = "iqr") -> int:
        """Count outliers in a pandas Series using the specified method.

        Args:
            series: A pandas Series containing numeric data.
            method: The outlier detection method to use. Valid options are
                'iqr' for interquartile range and 'zscore' for z‑score.
                The default is 'iqr'.

        Returns:
            int: The number of outliers detected.

        Raises:
            ValueError: If an unsupported method is provided.
        """
        if method == "iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return ((series < lower_bound) | (series > upper_bound)).sum()
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(series))
            return (z_scores > 2).sum()
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    @staticmethod
    def calculate_jaccard_similarity(set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity between two sets.

        Args:
            set1: The first set.
            set2: The second set.

        Returns:
            float: The Jaccard similarity, defined as the size of the
            intersection divided by the size of the union. If the union
            is empty, 0.0 is returned.
        """
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def calculate_overlap_matrix(gene_sets: Dict[str, Set[str]]) -> pd.DataFrame:
        """Calculate pairwise overlap matrix for a dictionary of gene sets.

        Args:
            gene_sets: A dictionary where keys are identifiers and values
                are sets of gene names.

        Returns:
            pd.DataFrame: A square matrix whose entry (i, j) is the count
            of genes shared between the ith and jth gene sets. The
            resulting DataFrame is indexed and columned by the dictionary
            keys.
        """
        names = list(gene_sets.keys())
        n = len(names)
        overlap_matrix = np.zeros((n, n), dtype=int)

        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                overlap_count = len(gene_sets[name1] & gene_sets[name2])
                overlap_matrix[i, j] = overlap_count

        return pd.DataFrame(overlap_matrix, index=names, columns=names)


# ==================================================================
# FACTORY FUNCTIONS
# ==================================================================
def create_default_filter_config() -> FilterConfig:
    """
    Create a default filter configuration.

    Returns:
        A FilterConfig instance with default settings.
    """
    return FilterConfig()


def create_strict_filter_config() -> FilterConfig:
    """
    Create a strict filter configuration with tighter thresholds.

    Returns:
        A FilterConfig instance configured for strict filtering.
    """
    return FilterConfig(
        effect_size_thresh=0.5,
        fdr_thresh=0.01,
        number_studies_thresh=3,
        heterogeneity_pval_thresh=0.05,
        is_leave_one_out=True,
    )


def create_default_scoring_config() -> ScoringConfig:
    """
    Create a default scoring configuration.

    Returns:
        A ScoringConfig instance with default parameters.
    """
    return ScoringConfig()


def create_default_pathway_config() -> PathwayConfig:
    """
    Create a default pathway configuration.

    Returns:
        A PathwayConfig instance with default settings.
    """
    return PathwayConfig()


def create_default_summary_config() -> SummaryConfig:
    """
    Create a default summary configuration.

    Returns:
        A SummaryConfig instance with default parameters.
    """
    return SummaryConfig()
