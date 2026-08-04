"""
Comprehensive unit tests for scoring_core.py module.
Tests cover all classes, methods, and functionality with edge cases.
"""

import logging
import os
import sys
from datetime import datetime

import pandas as pd
import pytest

# Add the src directory to the path to import scoring_core to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# Import all classes and functions from scoring_core
from metaintegrator.scoring_core import (
    # Exceptions
    ScoringError,
    ValidationError,
    ConfigurationError,
    FilteringError,
    ScoringCalculationError,
    # Enumerations
    ExportFormat,
    StatisticType,
    ScoringMethod,
    AggregationMethod,
    # Configuration classes
    BaseConfig,
    FilterConfig,
    ScoringConfig,
    PathwayConfig,
    SummaryConfig,
    # Data structures
    OverlapAnalysis,
    FilterSummary,
    StatisticalSummary,
    ScoringResults,
    # Validators
    BaseValidator,
    MetaAnalysisValidator,
    FilterResultsValidator,
    MetaObjectValidator,
    # Utilities
    LoggerMixin,
    DataFrameUtils,
    NormalizationUtils,
    StatisticalUtils,
    # Factory functions
    create_default_filter_config,
    create_strict_filter_config,
    create_default_scoring_config,
    create_default_pathway_config,
    create_default_summary_config,
)

# ==================================================================
# TEST EXCEPTION HIERARCHY
# ==================================================================


class TestExceptions:
    """Test all custom exceptions."""

    def test_scoring_error(self):
        """Test base ScoringError exception."""
        error = ScoringError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_validation_error(self):
        """Test ValidationError inheritance."""
        error = ValidationError("Validation failed")
        assert isinstance(error, ScoringError)
        assert str(error) == "Validation failed"

    def test_configuration_error(self):
        """Test ConfigurationError inheritance."""
        error = ConfigurationError("Config invalid")
        assert isinstance(error, ScoringError)
        assert str(error) == "Config invalid"

    def test_filtering_error(self):
        """Test FilteringError inheritance."""
        error = FilteringError("Filter failed")
        assert isinstance(error, ScoringError)
        assert str(error) == "Filter failed"

    def test_scoring_calculation_error(self):
        """Test ScoringCalculationError inheritance."""
        error = ScoringCalculationError("Calculation failed")
        assert isinstance(error, ScoringError)
        assert str(error) == "Calculation failed"


# ==================================================================
# TEST ENUMERATIONS
# ==================================================================


class TestEnumerations:
    """Test all enumeration classes."""

    def test_export_format_enum(self):
        """Test ExportFormat enumeration."""
        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.TSV.value == "tsv"
        assert ExportFormat.EXCEL.value == "xlsx"
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.PICKLE.value == "pkl"
        assert ExportFormat.PARQUET.value == "parquet"

    def test_statistic_type_enum(self):
        """Test StatisticType enumeration."""
        assert StatisticType.BASIC.value == "basic"
        assert StatisticType.DETAILED.value == "detailed"
        assert StatisticType.DISTRIBUTION.value == "distribution"
        assert StatisticType.CORRELATION.value == "correlation"

    def test_scoring_method_enum(self):
        """Test ScoringMethod enumeration."""
        assert ScoringMethod.Z_SCORE.value == "z_score"
        assert ScoringMethod.COMBINED.value == "combined"
        assert ScoringMethod.EFFECT_SIZE.value == "effect_size"
        assert ScoringMethod.FDR_INVERSE.value == "fdr_inverse"
        assert ScoringMethod.FISHER_MIN.value == "fisher_min"

    def test_aggregation_method_enum(self):
        """Test AggregationMethod enumeration."""
        assert AggregationMethod.MEAN.value == "mean"
        assert AggregationMethod.MEDIAN.value == "median"
        assert AggregationMethod.MAX.value == "max"
        assert AggregationMethod.SUM.value == "sum"
        assert AggregationMethod.WEIGHTED_MEAN.value == "weighted_mean"
        assert AggregationMethod.GSEA_LIKE.value == "gsea_like"


# ==================================================================
# TEST CONFIGURATION CLASSES
# ==================================================================


class TestBaseConfig:
    """Test BaseConfig class."""

    def test_base_config_creation(self):
        """Test BaseConfig creation and validation."""
        config = BaseConfig()
        # Should not raise exception
        config.validate()

    def test_to_dict_method(self):
        """Test conversion to dictionary."""
        config = BaseConfig()
        result = config.to_dict()
        assert isinstance(result, dict)

    def test_from_dict_method(self):
        """Test creation from dictionary."""
        data = {}
        config = BaseConfig.from_dict(data)
        assert isinstance(config, BaseConfig)


class TestFilterConfig:
    """Test FilterConfig class."""

    def test_default_filter_config(self):
        """Test default FilterConfig values."""
        config = FilterConfig()
        assert config.effect_size_thresh == 0.0
        assert config.fdr_thresh == 0.05
        assert config.number_studies_thresh == 1
        assert config.heterogeneity_pval_thresh == 0.0
        assert config.is_leave_one_out == True

    def test_valid_filter_config(self):
        """Test valid FilterConfig creation."""
        config = FilterConfig(
            effect_size_thresh=0.5,
            fdr_thresh=0.01,
            number_studies_thresh=3,
            heterogeneity_pval_thresh=0.05,
            is_leave_one_out=False,
        )
        # Should not raise exception
        config.validate()

    def test_invalid_fdr_thresh(self):
        """Test invalid FDR threshold."""
        with pytest.raises(ConfigurationError, match="FDR threshold must be between 0 and 1"):
            FilterConfig(fdr_thresh=1.5)

    def test_negative_fdr_thresh(self):
        """Test negative FDR threshold."""
        with pytest.raises(ConfigurationError, match="FDR threshold must be between 0 and 1"):
            FilterConfig(fdr_thresh=-0.1)

    def test_invalid_number_studies_thresh(self):
        """Test invalid number of studies threshold."""
        with pytest.raises(
            ConfigurationError, match="Number of studies threshold must be at least 1"
        ):
            FilterConfig(number_studies_thresh=0)

    def test_invalid_heterogeneity_pval_thresh(self):
        """Test invalid heterogeneity p-value threshold."""
        with pytest.raises(
            ConfigurationError,
            match="Heterogeneity p-value threshold must be between 0 and 1",
        ):
            FilterConfig(heterogeneity_pval_thresh=1.2)


class TestScoringConfig:
    """Test ScoringConfig class."""

    def test_default_scoring_config(self):
        """Test default ScoringConfig values."""
        config = ScoringConfig()
        expected_weights = {
            "effect_size": 0.3,
            "effect_fdr": 0.3,
            "fisher_score": 0.2,
            "heterogeneity": 0.2,
        }
        assert config.weights == expected_weights
        assert config.normalization_method == "min_max"
        assert config.handle_missing == "zero"
        assert config.min_valid_genes == 1

    def test_valid_scoring_config(self):
        """Test valid ScoringConfig creation."""
        weights = {"effect_size": 0.4, "effect_fdr": 0.6}
        config = ScoringConfig(
            weights=weights,
            normalization_method="z_score",
            handle_missing="median",
            min_valid_genes=5,
        )
        # Should not raise exception
        config.validate()

    def test_weights_sum_warning(self):
        """Test warning when weights don't sum to 1.0."""
        weights = {"effect_size": 0.5, "effect_fdr": 0.3}  # Sum = 0.8
        with pytest.warns(UserWarning, match="Weights sum to"):
            ScoringConfig(weights=weights)

    def test_invalid_weight_values(self):
        """Test invalid weight values."""
        weights = {"effect_size": 1.5, "effect_fdr": -0.2}
        with pytest.raises(ConfigurationError, match="All weights must be between 0 and 1"):
            ScoringConfig(weights=weights)

    def test_invalid_normalization_method(self):
        """Test invalid normalization method."""
        with pytest.raises(ConfigurationError, match="Invalid normalization method"):
            ScoringConfig(normalization_method="invalid")

    def test_invalid_handle_missing(self):
        """Test invalid missing value handling method."""
        with pytest.raises(ConfigurationError, match="Invalid missing value handling method"):
            ScoringConfig(handle_missing="invalid")

    def test_invalid_min_valid_genes(self):
        """Test invalid min_valid_genes."""
        with pytest.raises(ConfigurationError, match="min_valid_genes must be at least 1"):
            ScoringConfig(min_valid_genes=0)


class TestPathwayConfig:
    """Test PathwayConfig class."""

    def test_default_pathway_config(self):
        """Test default PathwayConfig values."""
        config = PathwayConfig()
        assert config.aggregation_method == AggregationMethod.MEAN
        assert config.min_genes_per_pathway == 3
        assert config.max_genes_per_pathway == 500
        assert config.weight_by_significance == False
        assert config.significance_column == "effectSizeFDR"
        assert config.direction_sensitive == True

    def test_valid_pathway_config(self):
        """Test valid PathwayConfig creation."""
        config = PathwayConfig(
            aggregation_method=AggregationMethod.MEDIAN,
            min_genes_per_pathway=5,
            max_genes_per_pathway=100,
            weight_by_significance=True,
            significance_column="pvalue",
            direction_sensitive=False,
        )
        # Should not raise exception
        config.validate()

    def test_invalid_min_genes_per_pathway(self):
        """Test invalid min_genes_per_pathway."""
        with pytest.raises(ConfigurationError, match="min_genes_per_pathway must be at least 1"):
            PathwayConfig(min_genes_per_pathway=0)

    def test_invalid_max_genes_per_pathway(self):
        """Test invalid max_genes_per_pathway."""
        with pytest.raises(
            ConfigurationError,
            match="max_genes_per_pathway must be >= min_genes_per_pathway",
        ):
            PathwayConfig(min_genes_per_pathway=10, max_genes_per_pathway=5)


class TestSummaryConfig:
    """Test SummaryConfig class."""

    def test_default_summary_config(self):
        """Test default SummaryConfig values."""
        config = SummaryConfig()
        assert config.include_distributions == True
        assert config.include_correlations == False
        assert config.significance_thresholds == [0.05, 0.01, 0.001]
        assert config.effect_size_thresholds == [0.2, 0.5, 0.8]
        assert config.heterogeneity_threshold == 0.05
        assert config.min_studies_threshold == 2

    def test_valid_summary_config(self):
        """Test valid SummaryConfig creation."""
        config = SummaryConfig(
            include_distributions=False,
            include_correlations=True,
            significance_thresholds=[0.01, 0.001],
            effect_size_thresholds=[0.3, 0.7],
            heterogeneity_threshold=0.01,
            min_studies_threshold=3,
        )
        # Should not raise exception
        config.validate()

    def test_invalid_significance_thresholds(self):
        """Test invalid significance thresholds."""
        with pytest.raises(ConfigurationError):
            SummaryConfig(significance_thresholds=[0.05, 1.5, -0.1])


# ==================================================================
# TEST DATA STRUCTURES
# ==================================================================


class TestOverlapAnalysis:
    """Test OverlapAnalysis class."""

    def create_sample_overlap_analysis(self):
        """Create sample OverlapAnalysis for testing."""
        return OverlapAnalysis(
            filter_names=["filter1", "filter2"],
            positive_overlaps={"filter1": {"filter2": 10}, "filter2": {"filter1": 10}},
            negative_overlaps={"filter1": {"filter2": 5}, "filter2": {"filter1": 5}},
            jaccard_similarities={
                "filter1": {"filter2": 0.7},
                "filter2": {"filter1": 0.7},
            },
            unique_genes={
                "filter1": {"filter2": {"gene1", "gene2"}},
                "filter2": {"filter1": {"gene3", "gene4"}},
            },
        )

    def test_overlap_analysis_creation(self):
        """Test OverlapAnalysis creation."""
        overlap = self.create_sample_overlap_analysis()
        assert overlap.filter_names == ["filter1", "filter2"]
        assert overlap.positive_overlaps["filter1"]["filter2"] == 10

    def test_get_overlap_matrix_positive(self):
        """Test getting positive overlap matrix."""
        overlap = self.create_sample_overlap_analysis()
        matrix = overlap.get_overlap_matrix("positive")
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.loc["filter1", "filter2"] == 10.0

    def test_get_overlap_matrix_negative(self):
        """Test getting negative overlap matrix."""
        overlap = self.create_sample_overlap_analysis()
        matrix = overlap.get_overlap_matrix("negative")
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.loc["filter1", "filter2"] == 5.0

    def test_get_jaccard_matrix(self):
        """Test getting Jaccard similarity matrix."""
        overlap = self.create_sample_overlap_analysis()
        matrix = overlap.get_jaccard_matrix()
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.loc["filter1", "filter2"] == 0.7

    def test_get_most_similar_pairs(self):
        """Test getting most similar pairs."""
        overlap = self.create_sample_overlap_analysis()
        pairs = overlap.get_most_similar_pairs(n=1)
        assert len(pairs) == 1
        assert len(pairs[0]) == 3  # (name1, name2, similarity)


class TestFilterSummary:
    """Test FilterSummary class."""

    def create_sample_filter_summary(self):
        """Create sample FilterSummary for testing."""
        return FilterSummary(
            name="test_filter",
            n_positive_genes=100,
            n_negative_genes=50,
            total_genes=150,
            parameters={"fdr_thresh": 0.05},
            timestamp=datetime.now(),
            description="Test filter",
            positive_genes=["gene1", "gene2"],
            negative_genes=["gene3"],
        )

    def test_filter_summary_creation(self):
        """Test FilterSummary creation."""
        summary = self.create_sample_filter_summary()
        assert summary.name == "test_filter"
        assert summary.n_positive_genes == 100
        assert summary.n_negative_genes == 50

    def test_gene_count_ratio_normal(self):
        """Test gene count ratio calculation."""
        summary = self.create_sample_filter_summary()
        assert summary.gene_count_ratio == 2.0  # 100/50

    def test_gene_count_ratio_no_negative(self):
        """Test gene count ratio with no negative genes."""
        summary = FilterSummary(
            name="test",
            n_positive_genes=100,
            n_negative_genes=0,
            total_genes=100,
            parameters={},
            timestamp=None,
            description="Test",
            positive_genes=[],
            negative_genes=[],
        )
        assert summary.gene_count_ratio == float("inf")

    def test_gene_count_ratio_no_genes(self):
        """Test gene count ratio with no genes."""
        summary = FilterSummary(
            name="test",
            n_positive_genes=0,
            n_negative_genes=0,
            total_genes=0,
            parameters={},
            timestamp=None,
            description="Test",
            positive_genes=[],
            negative_genes=[],
        )
        assert summary.gene_count_ratio == 0


class TestStatisticalSummary:
    """Test StatisticalSummary class."""

    def test_statistical_summary_creation(self):
        """Test StatisticalSummary creation."""
        summary = StatisticalSummary(
            basic_stats={"mean": 5.0, "std": 2.0},
            distribution_stats={"bins": [0, 1, 2]},
            correlation_stats={"correlation": 0.8},
            threshold_counts={"high": 10},
            metadata={"date": "2024-01-01"},
        )
        assert summary.basic_stats["mean"] == 5.0
        assert summary.distribution_stats["bins"] == [0, 1, 2]

    def test_to_dict_method(self):
        """Test conversion to dictionary."""
        summary = StatisticalSummary(basic_stats={"mean": 5.0}, metadata={"test": True})
        result = summary.to_dict()
        assert isinstance(result, dict)
        assert result["basic_stats"]["mean"] == 5.0
        assert result["metadata"]["test"] == True


class TestScoringResults:
    """Test ScoringResults class."""

    def create_sample_scoring_results(self):
        """Create sample ScoringResults for testing."""
        scores = pd.Series([1.0, 2.0, 3.0, 0.5], index=["gene1", "gene2", "gene3", "gene4"])
        return ScoringResults(scores=scores, method="test_method", metadata={"param": "value"})

    def test_scoring_results_creation(self):
        """Test ScoringResults creation."""
        results = self.create_sample_scoring_results()
        assert results.method == "test_method"
        assert len(results.scores) == 4
        assert results.metadata["param"] == "value"

    def test_get_ranked_genes(self):
        """Test getting ranked genes."""
        results = self.create_sample_scoring_results()
        ranked = results.get_ranked_genes()
        assert isinstance(ranked, pd.DataFrame)
        assert "score" in ranked.columns
        assert "rank" in ranked.columns
        assert ranked.loc["gene3", "rank"] == 1  # Highest score

    def test_get_ranked_genes_ascending(self):
        """Test getting ranked genes in ascending order."""
        results = self.create_sample_scoring_results()
        ranked = results.get_ranked_genes(ascending=True)
        assert ranked.loc["gene4", "rank"] == 1  # Lowest score

    def test_get_top_genes(self):
        """Test getting top genes."""
        results = self.create_sample_scoring_results()
        top_genes = results.get_top_genes(n=2)
        assert len(top_genes) == 2
        assert top_genes[0] == "gene3"  # Highest score
        assert top_genes[1] == "gene2"  # Second highest

    def test_get_percentile_threshold(self):
        """Test getting percentile threshold."""
        results = self.create_sample_scoring_results()
        threshold = results.get_percentile_threshold(50)
        assert threshold == 1.5  # Median of [0.5, 1.0, 2.0, 3.0]


# ==================================================================
# TEST VALIDATORS
# ==================================================================


class TestMetaAnalysisValidator:
    """Test MetaAnalysisValidator class."""

    def test_valid_meta_analysis(self):
        """Test validation of valid meta-analysis data."""
        data = {
            "pooledResults": pd.DataFrame(
                {
                    "effectSize": [1.0, 2.0],
                    "effectSizeStandardError": [0.1, 0.2],
                    "effectSizeFDR": [0.05, 0.01],
                }
            )
        }
        result = MetaAnalysisValidator.validate_and_extract(data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_meta_analysis_dict_pooled_results(self):
        """Test meta-analysis with dict pooled results."""
        data = {
            "pooledResults": {
                "effectSize": [1.0, 2.0],
                "effectSizeStandardError": [0.1, 0.2],
            }
        }
        result = MetaAnalysisValidator.validate_and_extract(data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_invalid_meta_analysis_type(self):
        """Test validation with invalid data type."""
        with pytest.raises(ValidationError, match="Meta-analysis must be a dictionary"):
            MetaAnalysisValidator.validate_and_extract("invalid")

    def test_missing_pooled_results(self):
        """Test validation with missing pooledResults."""
        data = {"otherField": "value"}
        with pytest.raises(ValidationError, match="Meta-analysis must contain 'pooledResults'"):
            MetaAnalysisValidator.validate_and_extract(data)

    def test_missing_required_columns(self):
        """Test validation with missing required columns."""
        data = {
            "pooledResults": pd.DataFrame(
                {
                    "effectSize": [1.0, 2.0]
                    # Missing effectSizeStandardError
                }
            )
        }
        with pytest.raises(ValidationError, match="Missing required columns"):
            MetaAnalysisValidator.validate_and_extract(data)

    def test_empty_results(self):
        """Test validation with empty results."""
        data = {"pooledResults": pd.DataFrame(columns=["effectSize", "effectSizeStandardError"])}
        with pytest.raises(ValidationError, match="Meta-analysis results are empty"):
            MetaAnalysisValidator.validate_and_extract(data)

    def test_missing_optional_columns_warning(self):
        """Test warning for missing optional columns."""
        data = {
            "pooledResults": pd.DataFrame(
                {
                    "effectSize": [1.0, 2.0],
                    "effectSizeStandardError": [0.1, 0.2],
                    # Missing optional columns
                }
            )
        }
        with pytest.warns(UserWarning, match="Missing optional columns"):
            MetaAnalysisValidator.validate_and_extract(data)


class TestFilterResultsValidator:
    """Test FilterResultsValidator class."""

    def test_valid_filter_results(self):
        """Test validation of valid filter results."""
        data = {
            "posGeneNames": ["gene1", "gene2"],
            "negGeneNames": ["gene3", "gene4"],
            "FDRThresh": 0.05,
        }
        result = FilterResultsValidator.validate_and_extract(data)
        assert result == data

    def test_invalid_filter_results_type(self):
        """Test validation with invalid data type."""
        with pytest.raises(ValidationError, match="Filter results must be a dictionary"):
            FilterResultsValidator.validate_and_extract("invalid")

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        data = {"posGeneNames": ["gene1"]}  # Missing negGeneNames
        with pytest.raises(ValidationError, match="Missing required fields"):
            FilterResultsValidator.validate_and_extract(data)


class TestMetaObjectValidator:
    """Test MetaObjectValidator class."""

    def test_valid_meta_object(self):
        """Test validation of valid meta-integrator object."""
        data = {"metaAnalysis": {"pooledResults": pd.DataFrame({"col": [1, 2]})}}
        result = MetaObjectValidator.validate_and_extract(data)
        assert result == data

    def test_invalid_meta_object_type(self):
        """Test validation with invalid data type."""
        with pytest.raises(ValidationError, match="Meta object must be a dictionary"):
            MetaObjectValidator.validate_and_extract("invalid")

    def test_missing_meta_analysis(self):
        """Test validation with missing metaAnalysis."""
        data = {"otherField": "value"}
        with pytest.raises(ValidationError, match="Meta object must contain 'metaAnalysis' field"):
            MetaObjectValidator.validate_and_extract(data)

    def test_invalid_meta_analysis_type(self):
        """Test validation with invalid metaAnalysis type."""
        data = {"metaAnalysis": "invalid"}
        with pytest.raises(ValidationError, match="Meta-analysis must be a dictionary"):
            MetaObjectValidator.validate_and_extract(data)

    def test_missing_pooled_results(self):
        """Test validation with missing pooledResults."""
        data = {"metaAnalysis": {"otherField": "value"}}
        with pytest.raises(
            ValidationError, match="Meta-analysis must contain 'pooledResults' field"
        ):
            MetaObjectValidator.validate_and_extract(data)


# ==================================================================
# TEST UTILITIES
# ==================================================================


class TestLoggerMixin:
    """Test LoggerMixin class."""

    class TestClass(LoggerMixin):
        """Test class that uses LoggerMixin."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    def test_logger_mixin_default_logger(self):
        """Test LoggerMixin with default logger."""
        obj = self.TestClass()
        assert hasattr(obj, "logger")
        assert isinstance(obj.logger, logging.Logger)

    def test_logger_mixin_custom_logger(self):
        """Test LoggerMixin with custom logger."""
        custom_logger = logging.getLogger("custom")
        obj = self.TestClass(logger=custom_logger)
        assert obj.logger == custom_logger


class TestDataFrameUtils:
    """Test DataFrameUtils class."""

    def test_ensure_dataframe_from_dict(self):
        """Test converting dict to DataFrame."""
        data = {"col1": [1, 2], "col2": [3, 4]}
        result = DataFrameUtils.ensure_dataframe(data)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1", "col2"]

    def test_ensure_dataframe_from_dataframe(self):
        """Test that DataFrame remains unchanged."""
        data = pd.DataFrame({"col": [1, 2, 3]})
        result = DataFrameUtils.ensure_dataframe(data)
        assert result is data

    def test_safe_column_access_existing(self):
        """Test safe access to existing column."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        result = DataFrameUtils.safe_column_access(df, "col1")
        assert isinstance(result, pd.Series)
        assert list(result) == [1, 2, 3]

    def test_safe_column_access_missing(self):
        """Test safe access to missing column."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        result = DataFrameUtils.safe_column_access(df, "missing", default_value=0)
        assert isinstance(result, pd.Series)
        assert list(result) == [0, 0, 0]

    def test_validate_columns_valid(self):
        """Test column validation with all required columns present."""
        df = pd.DataFrame({"req1": [1], "req2": [2], "opt1": [3]})
        # Should not raise exception
        DataFrameUtils.validate_columns(df, required=["req1", "req2"], optional=["opt1"])

    def test_validate_columns_missing_required(self):
        """Test column validation with missing required columns."""
        df = pd.DataFrame({"req1": [1]})
        with pytest.raises(ValidationError, match="Missing required columns"):
            DataFrameUtils.validate_columns(df, required=["req1", "req2"])

    def test_validate_columns_missing_optional_warning(self):
        """Test warning for missing optional columns."""
        df = pd.DataFrame({"req1": [1]})
        with pytest.warns(UserWarning, match="Missing optional columns"):
            DataFrameUtils.validate_columns(df, required=["req1"], optional=["opt1"])


class TestNormalizationUtils:
    """Test NormalizationUtils class."""

    def test_min_max_normalize(self):
        """Test min-max normalization."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = NormalizationUtils.min_max_normalize(series)
        assert result.min() == 0.0
        assert result.max() == 1.0
        assert result.iloc[2] == 0.5  # Middle value

    def test_min_max_normalize_constant(self):
        """Test min-max normalization with constant values."""
        series = pd.Series([5, 5, 5, 5])
        result = NormalizationUtils.min_max_normalize(series)
        assert all(result == 0.5)

    def test_z_score_normalize(self):
        """Test z-score normalization."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = NormalizationUtils.z_score_normalize(series)
        assert abs(result.mean()) < 1e-10  # Mean should be ~0
        assert abs(result.std() - 1.0) < 1e-10  # Std should be ~1

    def test_robust_normalize(self):
        """Test robust normalization."""
        series = pd.Series([1, 2, 3, 4, 100])  # With outlier
        result = NormalizationUtils.robust_normalize(series)
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)

    def test_robust_normalize_zero_mad(self):
        """Test robust normalization with zero MAD."""
        series = pd.Series([5, 5, 5, 5])
        result = NormalizationUtils.robust_normalize(series)
        assert all(result == 0)

    def test_normalize_series_min_max(self):
        """Test normalize_series with min_max method."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = NormalizationUtils.normalize_series(series, "min_max")
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_normalize_series_z_score(self):
        """Test normalize_series with z_score method."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = NormalizationUtils.normalize_series(series, "z_score")
        assert abs(result.mean()) < 1e-10

    def test_normalize_series_robust(self):
        """Test normalize_series with robust method."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = NormalizationUtils.normalize_series(series, "robust")
        assert isinstance(result, pd.Series)

    def test_normalize_series_invalid_method(self):
        """Test normalize_series with invalid method."""
        series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown normalization method"):
            NormalizationUtils.normalize_series(series, "invalid")


class TestStatisticalUtils:
    """Test StatisticalUtils class."""

    def test_count_outliers_iqr(self):
        """Test outlier counting using IQR method."""
        # Create data with clear outliers
        series = pd.Series([1, 2, 3, 4, 5, 100, 200])
        outliers = StatisticalUtils.count_outliers(series, "iqr")
        assert outliers >= 1  # Should detect the extreme values

    def test_count_outliers_zscore(self):
        """Test outlier counting using z-score method."""
        series = pd.Series([1, 2, 3, 4, 5, 100])
        outliers = StatisticalUtils.count_outliers(series, "zscore")
        assert outliers >= 1  # Should detect the extreme value

    def test_count_outliers_invalid_method(self):
        """Test outlier counting with invalid method."""
        series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown outlier detection method"):
            StatisticalUtils.count_outliers(series, "invalid")

    def test_calculate_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        set1 = {"a", "b", "c"}
        set2 = {"b", "c", "d"}
        similarity = StatisticalUtils.calculate_jaccard_similarity(set1, set2)
        expected = 2 / 4  # Intersection: {b, c}, Union: {a, b, c, d}
        assert similarity == expected

    def test_calculate_jaccard_similarity_no_overlap(self):
        """Test Jaccard similarity with no overlap."""
        set1 = {"a", "b"}
        set2 = {"c", "d"}
        similarity = StatisticalUtils.calculate_jaccard_similarity(set1, set2)
        assert similarity == 0.0

    def test_calculate_jaccard_similarity_identical(self):
        """Test Jaccard similarity with identical sets."""
        set1 = {"a", "b", "c"}
        set2 = {"a", "b", "c"}
        similarity = StatisticalUtils.calculate_jaccard_similarity(set1, set2)
        assert similarity == 1.0

    def test_calculate_jaccard_similarity_empty_sets(self):
        """Test Jaccard similarity with empty sets."""
        set1 = set()
        set2 = set()
        similarity = StatisticalUtils.calculate_jaccard_similarity(set1, set2)
        assert similarity == 0.0

    def test_calculate_overlap_matrix(self):
        """Test overlap matrix calculation."""
        gene_sets = {
            "set1": {"a", "b", "c"},
            "set2": {"b", "c", "d"},
            "set3": {"a", "d", "e"},
        }
        matrix = StatisticalUtils.calculate_overlap_matrix(gene_sets)
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.loc["set1", "set2"] == 2  # {b, c}
        assert matrix.loc["set1", "set1"] == 3  # Self overlap
        assert matrix.loc["set2", "set3"] == 1  # {d}


# ==================================================================
# TEST FACTORY FUNCTIONS
# ==================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_default_filter_config(self):
        """Test default filter config creation."""
        config = create_default_filter_config()
        assert isinstance(config, FilterConfig)
        assert config.fdr_thresh == 0.05

    def test_create_strict_filter_config(self):
        """Test strict filter config creation."""
        config = create_strict_filter_config()
        assert isinstance(config, FilterConfig)
        assert config.effect_size_thresh == 0.5
        assert config.fdr_thresh == 0.01
        assert config.number_studies_thresh == 3

    def test_create_default_scoring_config(self):
        """Test default scoring config creation."""
        config = create_default_scoring_config()
        assert isinstance(config, ScoringConfig)
        assert config.normalization_method == "min_max"

    def test_create_default_pathway_config(self):
        """Test default pathway config creation."""
        config = create_default_pathway_config()
        assert isinstance(config, PathwayConfig)
        assert config.aggregation_method == AggregationMethod.MEAN

    def test_create_default_summary_config(self):
        """Test default summary config creation."""
        config = create_default_summary_config()
        assert isinstance(config, SummaryConfig)
        assert config.include_distributions == True


# ==================================================================
# INTEGRATION TESTS
# ==================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_config_serialization_roundtrip(self):
        """Test config to dict and back conversion."""
        original_config = FilterConfig(
            effect_size_thresh=0.5, fdr_thresh=0.01, number_studies_thresh=3
        )

        # Convert to dict and back
        config_dict = original_config.to_dict()
        reconstructed_config = FilterConfig.from_dict(config_dict)

        assert reconstructed_config.effect_size_thresh == original_config.effect_size_thresh
        assert reconstructed_config.fdr_thresh == original_config.fdr_thresh
        assert reconstructed_config.number_studies_thresh == original_config.number_studies_thresh

    def test_scoring_results_with_normalization(self):
        """Test ScoringResults with normalization utilities."""
        scores = pd.Series([1, 2, 3, 4, 5], index=[f"gene{i}" for i in range(5)])
        results = ScoringResults(scores=scores, method="test")

        # Normalize scores
        normalized = NormalizationUtils.min_max_normalize(results.scores)

        # Create new results with normalized scores
        normalized_results = ScoringResults(
            scores=normalized,
            method="test_normalized",
            metadata={"original_method": "test"},
        )

        assert normalized_results.scores.min() == 0.0
        assert normalized_results.scores.max() == 1.0
        assert normalized_results.metadata["original_method"] == "test"

    def test_validator_with_dataframe_utils(self):
        """Test validator integration with DataFrame utilities."""
        # Create test data
        pooled_data = {
            "effectSize": [1.0, 2.0, 3.0],
            "effectSizeStandardError": [0.1, 0.2, 0.3],
        }

        meta_analysis = {"pooledResults": pooled_data}

        # Validate and extract
        df = MetaAnalysisValidator.validate_and_extract(meta_analysis)

        # Use DataFrame utilities
        effect_size_col = DataFrameUtils.safe_column_access(df, "effectSize")
        assert len(effect_size_col) == 3
        assert effect_size_col.iloc[0] == 1.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
