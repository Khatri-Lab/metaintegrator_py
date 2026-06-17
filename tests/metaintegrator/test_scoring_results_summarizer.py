"""
Comprehensive unit tests for scoring_results_summarizer.py module.

Tests cover:
- EnhancedSummaryConfig validation and functionality
- EnhancedResultSummarizer core analysis methods
- ResultSummarizer legacy compatibility
- Error handling and edge cases
- Integration scenarios with mock dependencies

To run these tests:
    pytest tests/metaintegrator/test_scoring_results_summarizer.py -v
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.scoring_results_summarizer import (
    EnhancedSummaryConfig,
    EnhancedResultSummarizer,
    ResultSummarizer,
)
from metaintegrator.scoring_core import (
    SummaryConfig,
    ExportFormat,
    ValidationError,
    ScoringCalculationError,
)

from metaintegrator.scoring_export_engine import ExportConfig

# ==================================================================
# FIXTURES
# ==================================================================


@pytest.fixture
def sample_results_df():
    """Create a comprehensive sample DataFrame for testing."""
    np.random.seed(42)  # For reproducible tests

    n_genes = 100
    gene_names = [f"GENE_{i:03d}" for i in range(n_genes)]

    return pd.DataFrame(
        {
            "effect_size": np.random.normal(0, 1.5, n_genes),
            "p_value": np.random.beta(1, 9, n_genes),  # Skewed toward small p-values
            "fdr": np.random.beta(1.5, 8.5, n_genes),  # Similar but slightly larger
            "log_fold_change": np.random.normal(0, 2, n_genes),
            "beta": np.random.normal(0, 1, n_genes),
            "pathway": np.random.choice(
                ["pathway_A", "pathway_B", "pathway_C"], n_genes
            ),
            "study": np.random.choice(
                ["study_1", "study_2", "study_3", "study_4"], n_genes
            ),
            "expression_level": np.random.lognormal(0, 1, n_genes),
            "confidence_interval_low": np.random.normal(-1, 0.5, n_genes),
            "confidence_interval_high": np.random.normal(1, 0.5, n_genes),
        },
        index=gene_names,
    )


@pytest.fixture
def sample_results_with_missing_data():
    """Create sample data with strategic missing values."""
    df = pd.DataFrame(
        {
            "effect_size": [1.2, np.nan, 0.8, -0.5, np.nan],
            "p_value": [0.01, 0.05, np.nan, 0.001, 0.3],
            "fdr": [0.02, np.nan, 0.1, 0.005, np.nan],
            "pathway": ["A", "B", None, "A", "C"],
            "study": ["study1", "study1", "study2", None, "study2"],
        },
        index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
    )
    return df


@pytest.fixture
def sample_results_with_outliers():
    """Create sample data with known outliers."""
    np.random.seed(123)
    normal_data = np.random.normal(0, 1, 20)
    # Add some extreme outliers
    outlier_data = np.concatenate([normal_data, [10, -8, 15, -12]])

    return pd.DataFrame(
        {
            "effect_size": outlier_data,
            "p_value": np.random.beta(2, 8, 24),
            "pathway": ["pathway_A"] * 24,
        },
        index=[f"GENE_{i:03d}" for i in range(24)],
    )


@pytest.fixture
def mock_statistical_analyzer():
    """Mock StatisticalAnalyzer for testing."""
    mock = Mock()
    mock.analyze.return_value = {
        "basic_stats": {"mean_effect_size": 0.12, "std_effect_size": 1.45},
        "distributions": {"effect_size": "normal"},
        "correlations": {"effect_size_p_value": -0.23},
    }
    return mock


@pytest.fixture
def mock_distribution_analyzer():
    """Mock DistributionAnalyzer for testing."""
    mock = Mock()
    mock.analyze.return_value = {"distribution_type": "normal"}
    return mock


@pytest.fixture
def mock_export_engine():
    """Mock ExportEngine for testing."""
    mock = Mock()
    mock.export_analysis_results.return_value = {"csv": "/path/to/results.csv"}
    mock.export_statistical_summary.return_value = {"json": "/path/to/summary.json"}
    mock._export_json.return_value = "/path/to/comprehensive.json"
    return mock


# ==================================================================
# ENHANCED SUMMARY CONFIG TESTS
# ==================================================================


class TestEnhancedSummaryConfig:
    """Test suite for EnhancedSummaryConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = EnhancedSummaryConfig()

        # Check inherited defaults
        assert config.include_distributions == True
        assert (
            config.include_correlations == False
        )  # Default is False in SummaryConfig base class
        assert config.significance_thresholds == [0.05, 0.01, 0.001]

        # Check enhanced defaults
        assert config.include_trend_analysis == False
        assert config.include_quality_assessment == True
        assert config.include_pathway_analysis == False
        assert config.include_study_level_summary == False
        assert config.generate_visualizations == True
        assert config.visualization_format == "png"
        assert config.plot_style == "seaborn"
        assert config.auto_export == True
        assert config.export_formats == [ExportFormat.CSV, ExportFormat.JSON]
        assert config.parallel_processing == False
        assert config.max_workers == 4
        assert config.memory_efficient == True

    def test_custom_initialization(self):
        """Test custom configuration values."""
        config = EnhancedSummaryConfig(
            include_trend_analysis=True,
            include_pathway_analysis=True,
            visualization_format="svg",
            plot_style="matplotlib",
            max_workers=8,
            export_formats=[ExportFormat.CSV, ExportFormat.JSON, ExportFormat.EXCEL],
        )

        assert config.include_trend_analysis == True
        assert config.include_pathway_analysis == True
        assert config.visualization_format == "svg"
        assert config.plot_style == "matplotlib"
        assert config.max_workers == 8
        assert len(config.export_formats) == 3

    def test_invalid_visualization_format(self):
        """Test validation of visualization_format parameter."""
        with pytest.raises(
            ValueError, match="visualization_format must be one of: png, svg, pdf"
        ):
            EnhancedSummaryConfig(visualization_format="invalid")

    def test_invalid_plot_style(self):
        """Test validation of plot_style parameter."""
        with pytest.raises(
            ValueError, match="plot_style must be one of: seaborn, matplotlib, plotly"
        ):
            EnhancedSummaryConfig(plot_style="invalid")

    def test_invalid_max_workers(self):
        """Test validation of max_workers parameter."""
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            EnhancedSummaryConfig(max_workers=0)

        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            EnhancedSummaryConfig(max_workers=-1)

    def test_valid_edge_cases(self):
        """Test valid edge case values."""
        # Minimum valid max_workers
        config = EnhancedSummaryConfig(max_workers=1)
        assert config.max_workers == 1

        # All valid formats
        for fmt in ["png", "svg", "pdf"]:
            config = EnhancedSummaryConfig(visualization_format=fmt)
            assert config.visualization_format == fmt

        # All valid plot styles
        for style in ["seaborn", "matplotlib", "plotly"]:
            config = EnhancedSummaryConfig(plot_style=style)
            assert config.plot_style == style


# ==================================================================
# ENHANCED RESULT SUMMARIZER TESTS
# ==================================================================


class TestEnhancedResultSummarizer:
    """Test suite for EnhancedResultSummarizer class."""

    def test_initialization_default(self):
        """Test default initialization."""
        summarizer = EnhancedResultSummarizer()

        assert isinstance(summarizer.config, EnhancedSummaryConfig)
        assert summarizer.export_config is not None
        assert summarizer.statistical_analyzer is not None
        assert summarizer.distribution_analyzer is not None
        assert summarizer.export_engine is not None
        assert isinstance(summarizer._analysis_cache, dict)
        assert len(summarizer._analysis_cache) == 0

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = EnhancedSummaryConfig(include_trend_analysis=True)
        summarizer = EnhancedResultSummarizer(config=config)

        assert summarizer.config.include_trend_analysis == True

    @patch("metaintegrator.scoring_results_summarizer.StatisticalAnalyzer")
    @patch("metaintegrator.scoring_results_summarizer.DistributionAnalyzer")
    @patch("metaintegrator.scoring_results_summarizer.ExportEngine")
    def test_initialization_with_mocks(
        self, mock_export_engine, mock_dist_analyzer, mock_stat_analyzer
    ):
        """Test initialization with mocked dependencies."""
        summarizer = EnhancedResultSummarizer()

        mock_stat_analyzer.assert_called_once()
        mock_dist_analyzer.assert_called_once()
        mock_export_engine.assert_called_once()

    def test_get_analysis_cache(self):
        """Test analysis cache retrieval."""
        summarizer = EnhancedResultSummarizer()
        summarizer._analysis_cache = {"test_key": "test_value"}

        cache = summarizer.get_analysis_cache()
        assert cache == {"test_key": "test_value"}

        # Ensure it's a copy
        cache["new_key"] = "new_value"
        assert "new_key" not in summarizer._analysis_cache

    def test_clear_analysis_cache(self):
        """Test analysis cache clearing."""
        summarizer = EnhancedResultSummarizer()
        summarizer._analysis_cache = {"test_key": "test_value"}

        summarizer.clear_analysis_cache()
        assert len(summarizer._analysis_cache) == 0


class TestSummarizeResults:
    """Test the main summarize_results method."""

    def get_enhanced_result_summarizer_obj_with_output(self):
        export_config = ExportConfig(
            base_output_dir=Path("/tmp/unittest_metaintegrator")
        )

        summarizer = EnhancedResultSummarizer(export_config=export_config)
        return summarizer

    @patch(
        "metaintegrator.scoring_results_summarizer.MetaAnalysisValidator.validate_dataframe"
    )
    def test_summarize_results_basic(self, mock_validate, sample_results_df):
        """Test basic summarize_results functionality."""
        # Create mocks
        mock_statistical_analyzer = Mock()
        mock_statistical_analyzer.analyze.return_value = {
            "basic_stats": {"mean_effect_size": 0.12, "std_effect_size": 1.45},
            "distributions": {"effect_size": "normal"},
            "correlations": {"effect_size_p_value": -0.23},
        }

        mock_export_engine = Mock()
        mock_export_engine.export_analysis_results.return_value = {
            "csv": "/path/to/results.csv"
        }
        mock_export_engine.export_statistical_summary.return_value = {
            "json": "/path/to/summary.json"
        }
        mock_export_engine._export_json.return_value = "/path/to/comprehensive.json"

        summarizer = self.get_enhanced_result_summarizer_obj_with_output()

        # Replace instance attributes with mocks
        summarizer.statistical_analyzer = mock_statistical_analyzer
        summarizer.export_engine = mock_export_engine

        result = summarizer.summarize_results(sample_results_df, "test_analysis")

        # Check structure
        assert isinstance(result, dict)
        assert "analysis_info" in result
        assert "statistical_summary" in result
        assert "gene_summary" in result
        assert "quality_assessment" in result

        # Check analysis_info
        info = result["analysis_info"]
        assert info["name"] == "test_analysis"
        assert isinstance(info["timestamp"], datetime)
        assert info["total_genes"] == len(sample_results_df)
        assert info["total_columns"] == len(sample_results_df.columns)

        # Validate was called
        mock_validate.assert_called_once_with(sample_results_df)

    def test_summarize_results_with_metadata(self, sample_results_df):
        """Test summarize_results with metadata."""
        with patch(
            "metaintegrator.scoring_results_summarizer.MetaAnalysisValidator.validate_dataframe"
        ):
            summarizer = self.get_enhanced_result_summarizer_obj_with_output()

            metadata = {"experiment_id": "EXP123", "researcher": "Dr. Smith"}
            result = summarizer.summarize_results(
                sample_results_df, "test_analysis", metadata=metadata
            )

            assert result["analysis_info"]["metadata"] == metadata

    def test_summarize_results_with_optional_analyses(self, sample_results_df):
        """Test summarize_results with optional analysis components enabled."""
        config = EnhancedSummaryConfig(
            include_trend_analysis=True,
            include_pathway_analysis=True,
            include_study_level_summary=True,
        )

        with patch(
            "metaintegrator.scoring_results_summarizer.MetaAnalysisValidator.validate_dataframe"
        ):
            export_config = ExportConfig(
                base_output_dir=Path("/tmp/unittest_metaintegrator")
            )
            summarizer = EnhancedResultSummarizer(
                config=config, export_config=export_config
            )

            result = summarizer.summarize_results(sample_results_df, "test_analysis")

            assert result["trend_analysis"] is not None
            assert result["pathway_analysis"] is not None
            assert result["study_analysis"] is not None

    @patch(
        "metaintegrator.scoring_results_summarizer.MetaAnalysisValidator.validate_dataframe"
    )
    def test_summarize_results_validation_error(self, mock_validate, sample_results_df):
        """Test summarize_results with validation error."""
        mock_validate.side_effect = ValidationError("Invalid DataFrame")

        summarizer = self.get_enhanced_result_summarizer_obj_with_output()

        with pytest.raises(ScoringCalculationError, match="Summarization failed"):
            summarizer.summarize_results(sample_results_df, "test_analysis")


class TestAnalyzeGeneLevel:
    """Test the _analyze_gene_level method."""

    def test_analyze_gene_level_basic(self, sample_results_df):
        """Test basic gene-level analysis."""
        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_gene_level(sample_results_df)

        assert "total_genes" in result
        assert "gene_statistics" in result
        assert "significant_genes" in result
        assert "ranking_analysis" in result

        assert result["total_genes"] == len(sample_results_df)

    def test_analyze_gene_level_statistics(self, sample_results_df):
        """Test gene statistics calculation."""
        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_gene_level(sample_results_df)

        stats = result["gene_statistics"]

        # Check that numeric columns are analyzed
        assert "effect_size" in stats
        assert "p_value" in stats

        # Check statistics structure
        effect_stats = stats["effect_size"]
        assert "mean" in effect_stats
        assert "median" in effect_stats
        assert "std" in effect_stats
        assert "min" in effect_stats
        assert "max" in effect_stats
        assert "non_zero_count" in effect_stats
        assert "positive_count" in effect_stats
        assert "negative_count" in effect_stats

    def test_analyze_gene_level_significant_genes(self):
        """Test significant gene identification."""
        # Create data with known significant genes
        df = pd.DataFrame(
            {
                "fdr": [0.001, 0.01, 0.05, 0.1, 0.3],
                "effect_size": [2.0, 1.5, 1.0, 0.5, 0.1],
            },
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
        )

        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_gene_level(df)

        sig_genes = result["significant_genes"]

        # Check threshold-based significance
        assert "threshold_0.05" in sig_genes
        assert (
            sig_genes["threshold_0.05"]["count"] == 2
        )  # GENE1, GENE2 (GENE3 has fdr=0.05, not < 0.05)
        assert set(sig_genes["threshold_0.05"]["genes"]) == {"GENE1", "GENE2"}

    def test_analyze_gene_level_empty_data(self):
        """Test gene analysis with empty DataFrame."""
        df = pd.DataFrame()
        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_gene_level(df)

        assert result["total_genes"] == 0
        assert len(result["gene_statistics"]) == 0

    def test_analyze_gene_level_with_missing_data(
        self, sample_results_with_missing_data
    ):
        """Test gene analysis with missing data."""
        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_gene_level(sample_results_with_missing_data)

        # Should handle missing data gracefully
        assert result["total_genes"] == 5
        assert "effect_size" in result["gene_statistics"]


class TestAssessDataQuality:
    """Test the _assess_data_quality method."""

    def test_assess_data_quality_basic(self, sample_results_df):
        """Test basic quality assessment."""
        summarizer = EnhancedResultSummarizer()
        result = summarizer._assess_data_quality(sample_results_df)

        assert "overall_score" in result
        assert "missing_data" in result
        assert "outlier_analysis" in result
        assert "data_consistency" in result
        assert "recommendations" in result

        assert 0 <= result["overall_score"] <= 100

    def test_assess_data_quality_with_missing_data(
        self, sample_results_with_missing_data
    ):
        """Test quality assessment with missing data."""
        summarizer = EnhancedResultSummarizer()
        result = summarizer._assess_data_quality(sample_results_with_missing_data)

        # Should detect missing data issues
        assert result["missing_data"] is not None
        assert result["overall_score"] < 100  # Should be penalized for missing data

    def test_assess_data_quality_with_outliers(self, sample_results_with_outliers):
        """Test quality assessment with outliers."""
        summarizer = EnhancedResultSummarizer()
        result = summarizer._assess_data_quality(sample_results_with_outliers)

        # Should detect outliers
        outlier_analysis = result["outlier_analysis"]
        assert "effect_size" in outlier_analysis
        assert outlier_analysis["effect_size"]["outlier_count"] > 0

    def test_assess_data_quality_with_duplicates(self):
        """Test quality assessment with duplicate indices."""
        df = pd.DataFrame(
            {"effect_size": [1, 2, 3], "p_value": [0.01, 0.02, 0.03]},
            index=["GENE1", "GENE1", "GENE2"],
        )  # Duplicate GENE1

        summarizer = EnhancedResultSummarizer()
        result = summarizer._assess_data_quality(df)

        issues = result["data_consistency"]["issues"]
        assert any("Duplicate gene identifiers" in issue for issue in issues)

    def test_assess_data_quality_with_infinite_values(self):
        """Test quality assessment with infinite values."""
        df = pd.DataFrame(
            {
                "effect_size": [1, np.inf, 3, -np.inf],
                "p_value": [0.01, 0.02, 0.03, 0.04],
            },
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        summarizer = EnhancedResultSummarizer()
        result = summarizer._assess_data_quality(df)

        issues = result["data_consistency"]["issues"]
        assert any("Infinite values" in issue for issue in issues)

    def test_assess_data_quality_with_extreme_values(self):
        """Test quality assessment with extreme values."""
        df = pd.DataFrame(
            {
                "effect_size": [1, 2e6, 3, -3e6],  # Extreme values
                "p_value": [0.01, 0.02, 0.03, 0.04],
            },
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        summarizer = EnhancedResultSummarizer()
        result = summarizer._assess_data_quality(df)

        issues = result["data_consistency"]["issues"]
        assert any("Extreme values" in issue for issue in issues)


class TestAnalyzeTrends:
    """Test the _analyze_trends method."""

    def test_analyze_trends_basic(self, sample_results_df):
        """Test basic trend analysis."""
        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_trends(sample_results_df)

        assert "temporal_trends" in result
        assert "magnitude_trends" in result
        assert "correlation_trends" in result
        assert "pattern_detection" in result

    def test_analyze_trends_magnitude_distributions(self):
        """Test magnitude trend detection."""
        # Create data with known patterns
        df = pd.DataFrame(
            {
                "normal_data": np.random.normal(0, 1, 50),
                "skewed_data": np.random.exponential(1, 50),
            }
        )

        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_trends(df)

        magnitude_trends = result["magnitude_trends"]
        assert "normal_data" in magnitude_trends
        assert "skewed_data" in magnitude_trends

        # Check trend components
        for col_trends in magnitude_trends.values():
            assert "distribution_type" in col_trends
            assert "trend_direction" in col_trends
            assert "volatility" in col_trends

    def test_analyze_trends_correlations(self):
        """Test correlation trend analysis."""
        # Create data with known correlations
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = 0.8 * x + 0.6 * np.random.normal(0, 1, 50)  # Positive correlation
        z = -0.7 * x + 0.5 * np.random.normal(0, 1, 50)  # Negative correlation

        df = pd.DataFrame({"x": x, "y": y, "z": z})

        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_trends(df)

        corr_trends = result["correlation_trends"]
        assert "strongest_positive" in corr_trends
        assert "strongest_negative" in corr_trends
        assert "average_correlation" in corr_trends

    def test_analyze_trends_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        df = pd.DataFrame({"col1": [1, 2, 3]})  # Only 3 points

        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_trends(df)

        # Should handle gracefully
        assert isinstance(result, dict)


class TestAnalyzePathwayLevel:
    """Test the _analyze_pathway_level method."""

    def test_analyze_pathway_level_basic(self):
        """Test basic pathway-level analysis."""
        df = pd.DataFrame(
            {
                "effect_size": [1.0, 1.5, 0.5, -0.3, -0.8],
                "p_value": [0.01, 0.001, 0.05, 0.1, 0.2],
                "pathway": ["A", "A", "B", "B", "C"],
            },
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
        )

        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_pathway_level(df)

        assert "pathway_summary" in result
        assert "pathway_comparisons" in result
        assert "pathway_enrichment" in result

    def test_analyze_pathway_level_summaries(self):
        """Test pathway summary calculations."""
        df = pd.DataFrame(
            {
                "effect_size": [1.0, 1.5, 0.5, -0.3],
                "p_value": [0.01, 0.001, 0.05, 0.1],
                "pathway": ["A", "A", "B", "B"],
            },
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_pathway_level(df)

        pathway_summary = result["pathway_summary"]

        # Check pathway A summary
        assert "A" in pathway_summary
        pathway_a_stats = pathway_summary["A"]
        assert "effect_size" in pathway_a_stats
        assert pathway_a_stats["effect_size"]["gene_count"] == 2
        assert pathway_a_stats["effect_size"]["mean"] == 1.25  # (1.0 + 1.5) / 2

    def test_analyze_pathway_level_comparisons(self):
        """Test pathway comparisons."""
        df = pd.DataFrame(
            {"effect_size": [1.0, 1.5, 0.5, -0.3], "pathway": ["A", "A", "B", "B"]},
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_pathway_level(df)

        pathway_comparisons = result["pathway_comparisons"]
        assert "comparison_performed" in pathway_comparisons
        assert pathway_comparisons["comparison_performed"] == True


class TestAnalyzeStudyLevel:
    """Test the _analyze_study_level method."""

    def test_analyze_study_level_basic(self):
        """Test basic study-level analysis."""
        df = pd.DataFrame(
            {
                "effect_size": [1.0, 1.5, 0.5, -0.3],
                "p_value": [0.01, 0.001, 0.05, 0.1],
                "study": ["study1", "study1", "study2", "study2"],
            },
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_study_level(df)

        assert "study_summary" in result
        assert "cross_study_analysis" in result
        assert "study_quality" in result

    def test_analyze_study_level_summaries(self):
        """Test study summary calculations."""
        df = pd.DataFrame(
            {
                "effect_size": [1.0, 1.5, 0.5, -0.3],
                "p_value": [0.01, np.nan, 0.05, 0.1],  # Missing value in study1
                "study": ["study1", "study1", "study2", "study2"],
            },
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_study_level(df)

        study_summary = result["study_summary"]

        # Check study1 summary
        assert "study1" in study_summary
        study1_stats = study_summary["study1"]
        assert "effect_size" in study1_stats
        assert study1_stats["effect_size"]["gene_count"] == 2
        assert "missing_percentage" in study1_stats["p_value"]

    def test_analyze_study_level_cross_study_consistency(self):
        """Test cross-study consistency analysis."""
        df = pd.DataFrame(
            {
                "effect_size": [1.0, 1.5, 0.5, -0.3],
                "study": ["study1", "study1", "study2", "study2"],
            },
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        summarizer = EnhancedResultSummarizer()
        result = summarizer._analyze_study_level(df)

        cross_study = result["cross_study_analysis"]
        assert "consistency_score" in cross_study


class TestHelperMethods:
    """Test helper methods."""

    def test_detect_distribution_pattern(self):
        """Test distribution pattern detection."""
        summarizer = EnhancedResultSummarizer()

        # Normal-like data
        normal_data = np.random.normal(0, 1, 1000)
        pattern = summarizer._detect_distribution_pattern(normal_data)
        assert pattern in ["normal-like", "irregular"]  # May vary due to randomness

        # Right-skewed data
        skewed_data = np.random.exponential(1, 1000)
        pattern = summarizer._detect_distribution_pattern(skewed_data)
        assert pattern in ["right-skewed", "heavy-tailed", "irregular"]

    def test_detect_trend_direction(self):
        """Test trend direction detection."""
        summarizer = EnhancedResultSummarizer()

        # Increasing trend
        increasing_data = np.arange(100) + np.random.normal(0, 5, 100)
        trend = summarizer._detect_trend_direction(increasing_data)
        assert trend in ["increasing", "no_trend"]  # May vary due to noise

        # Decreasing trend
        decreasing_data = -np.arange(100) + np.random.normal(0, 5, 100)
        trend = summarizer._detect_trend_direction(decreasing_data)
        assert trend in ["decreasing", "no_trend"]

    def test_find_strongest_correlation(self):
        """Test finding strongest correlation."""
        # Create correlation matrix
        corr_data = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [2, 4, 6, 8, 10],  # Perfect positive correlation with A
                "C": [-1, -2, -3, -4, -5],  # Perfect negative correlation with A
                "D": [3, 1, 4, 2, 5],  # Weak correlation
            }
        )
        corr_matrix = corr_data.corr()

        summarizer = EnhancedResultSummarizer()

        # Test strongest positive
        pos_result = summarizer._find_strongest_correlation(corr_matrix, positive=True)
        assert pos_result["value"] > 0.9  # Should be close to 1
        assert set(pos_result["variables"]) == {"A", "B"}

        # Test strongest negative
        neg_result = summarizer._find_strongest_correlation(corr_matrix, positive=False)
        assert neg_result["value"] < -0.9  # Should be close to -1
        assert set(neg_result["variables"]) == {"A", "C"}

    def test_find_strongest_correlation_empty_matrix(self):
        """Test finding correlation in empty matrix."""
        empty_corr = pd.DataFrame()
        summarizer = EnhancedResultSummarizer()

        result = summarizer._find_strongest_correlation(empty_corr)
        assert np.isnan(result["value"])
        assert result["variables"] == []

    def test_generate_quality_recommendations(self):
        """Test quality recommendation generation."""
        summarizer = EnhancedResultSummarizer()

        # High missing data scenario
        quality_assessment = {
            "missing_data": {
                "col1": {"missing_percentage": 15},
                "col2": {"missing_percentage": 20},
            },
            "outlier_analysis": {"col1": {"outlier_percentage": 8}},
            "data_consistency": {"issues": ["Duplicate identifiers"]},
        }

        recommendations = summarizer._generate_quality_recommendations(
            quality_assessment
        )

        assert len(recommendations) == 3
        assert any("imputation" in rec.lower() for rec in recommendations)
        assert any("outliers" in rec.lower() for rec in recommendations)
        assert any("consistency" in rec.lower() for rec in recommendations)

    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        summarizer = EnhancedResultSummarizer()

        # Perfect quality
        perfect_quality = {
            "missing_data": {},
            "outlier_analysis": {},
            "data_consistency": {"issues": []},
        }
        score = summarizer._calculate_quality_score(perfect_quality)
        assert score == 100.0

        # Poor quality
        poor_quality = {
            "missing_data": {
                "col1": {"missing_percentage": 50},
                "col2": {"missing_percentage": 60},
            },
            "outlier_analysis": {"col1": {"outlier_percentage": 30}},
            "data_consistency": {"issues": ["Issue 1", "Issue 2", "Issue 3"]},
        }
        score = summarizer._calculate_quality_score(poor_quality)
        assert 0 <= score <= 100
        assert score < 50  # Should be significantly penalized


class TestVisualizationAndExport:
    """Test visualization and export functionality."""

    def test_generate_visualizations(self, sample_results_df):
        """Test visualization generation."""
        summarizer = EnhancedResultSummarizer()
        summary = {"test": "data"}

        result = summarizer._generate_visualizations(sample_results_df, "test", summary)

        assert "generated" in result
        assert "plot_count" in result
        assert "format" in result

    def test_auto_export(self, sample_results_df):
        """Test auto-export functionality."""
        mock_export_engine = Mock()
        mock_export_engine.export_analysis_results.return_value = {
            "csv": "/path/to/results.csv"
        }
        mock_export_engine.export_statistical_summary.return_value = {
            "json": "/path/to/summary.json"
        }
        mock_export_engine._export_json.return_value = "/path/to/comprehensive.json"

        summarizer = EnhancedResultSummarizer()
        summarizer.export_engine = mock_export_engine

        summary = {"statistical_summary": {"test": "data"}}
        result = summarizer._auto_export(sample_results_df, summary, "test_analysis")

        assert "main_results" in result
        assert "statistical_summary" in result
        assert "comprehensive_summary" in result

    def test_auto_export_error_handling(self, sample_results_df):
        """Test auto-export error handling."""
        mock_export_engine = Mock()
        mock_export_engine.export_analysis_results.side_effect = Exception(
            "Export failed"
        )

        summarizer = EnhancedResultSummarizer()
        summarizer.export_engine = mock_export_engine

        summary = {}
        result = summarizer._auto_export(sample_results_df, summary, "test_analysis")

        assert "error" in result
        assert "Export failed" in result["error"]


# ==================================================================
# LEGACY COMPATIBILITY TESTS
# ==================================================================


class TestResultSummarizer:
    """Test suite for ResultSummarizer legacy compatibility class."""

    def test_initialization_with_legacy_config(self):
        """Test initialization with legacy SummaryConfig."""
        legacy_config = SummaryConfig(
            include_distributions=False,
            include_correlations=True,
            significance_thresholds=[0.01],
        )

        summarizer = ResultSummarizer(config=legacy_config)

        # Should convert to enhanced config
        assert isinstance(summarizer.config, EnhancedSummaryConfig)
        assert summarizer.config.include_distributions == False
        assert summarizer.config.include_correlations == True
        assert summarizer.config.significance_thresholds == [0.01]

    def test_initialization_with_enhanced_config(self):
        """Test initialization with EnhancedSummaryConfig."""
        enhanced_config = EnhancedSummaryConfig(include_trend_analysis=True)

        summarizer = ResultSummarizer(config=enhanced_config)

        assert isinstance(summarizer.config, EnhancedSummaryConfig)
        assert summarizer.config.include_trend_analysis == True

    def test_initialization_without_config(self):
        """Test initialization without config."""
        summarizer = ResultSummarizer()

        assert isinstance(summarizer.config, EnhancedSummaryConfig)

    @patch.object(EnhancedResultSummarizer, "summarize_results")
    def test_summarize_method_compatibility(self, mock_summarize, sample_results_df):
        """Test legacy summarize method."""
        # Mock the comprehensive summary
        mock_comprehensive = {
            "statistical_summary": {"mean_effect_size": 0.12},
            "other_data": "ignored",
        }
        mock_summarize.return_value = mock_comprehensive

        summarizer = ResultSummarizer()
        result = summarizer.summarize(sample_results_df)

        # Should return only statistical summary
        assert result == {"mean_effect_size": 0.12}
        mock_summarize.assert_called_once_with(sample_results_df)

    @patch.object(EnhancedResultSummarizer, "summarize_results")
    def test_summarize_method_with_no_statistical_summary(
        self, mock_summarize, sample_results_df
    ):
        """Test legacy summarize method when no statistical summary exists."""
        mock_comprehensive = {"other_data": "test"}
        mock_summarize.return_value = mock_comprehensive

        summarizer = ResultSummarizer()
        result = summarizer.summarize(sample_results_df)

        # Should return None when no statistical summary
        assert result is None


# ==================================================================
# INTEGRATION TESTS
# ==================================================================


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_workflow_minimal_config(self, sample_results_df):
        """Test complete workflow with minimal configuration."""
        config = EnhancedSummaryConfig(
            auto_export=False,  # Disable export for testing
            generate_visualizations=False,  # Disable visualizations for testing
        )

        with patch(
            "metaintegrator.scoring_results_summarizer.MetaAnalysisValidator.validate_dataframe"
        ):
            summarizer = EnhancedResultSummarizer(config=config)
            result = summarizer.summarize_results(sample_results_df, "integration_test")

            # Check all major components are present
            assert "analysis_info" in result
            assert "statistical_summary" in result
            assert "gene_summary" in result
            assert "quality_assessment" in result

            # Check that optional components are disabled
            assert result["trend_analysis"] is None
            assert result["pathway_analysis"] is None
            assert result["study_analysis"] is None

    def test_full_workflow_comprehensive_config(self, sample_results_df):
        """Test complete workflow with comprehensive configuration."""
        config = EnhancedSummaryConfig(
            include_trend_analysis=True,
            include_pathway_analysis=True,
            include_study_level_summary=True,
            auto_export=False,  # Disable export for testing
            generate_visualizations=False,  # Disable visualizations for testing
        )

        with patch(
            "metaintegrator.scoring_results_summarizer.MetaAnalysisValidator.validate_dataframe"
        ):
            summarizer = EnhancedResultSummarizer(config=config)
            result = summarizer.summarize_results(
                sample_results_df, "comprehensive_test"
            )

            # Check all components are present and non-None
            assert result["trend_analysis"] is not None
            assert result["pathway_analysis"] is not None
            assert result["study_analysis"] is not None

    def test_error_propagation(self):
        """Test that errors are properly propagated through the system."""
        with patch(
            "metaintegrator.scoring_results_summarizer.MetaAnalysisValidator.validate_dataframe"
        ) as mock_validate:
            mock_validate.side_effect = ValidationError("Test validation error")

            summarizer = EnhancedResultSummarizer()

            with pytest.raises(ScoringCalculationError) as exc_info:
                summarizer.summarize_results(pd.DataFrame(), "error_test")

            assert "Summarization failed" in str(exc_info.value)
            assert "Test validation error" in str(exc_info.value)

    def test_memory_management(self, sample_results_df):
        """Test memory management with analysis cache."""
        config = EnhancedSummaryConfig(auto_export=False, generate_visualizations=False)

        with patch(
            "metaintegrator.scoring_results_summarizer.MetaAnalysisValidator.validate_dataframe"
        ):
            summarizer = EnhancedResultSummarizer(config=config)

            # Perform analysis
            summarizer.summarize_results(sample_results_df, "memory_test")

            # Cache should be accessible
            cache = summarizer.get_analysis_cache()
            assert isinstance(cache, dict)

            # Clear cache
            summarizer.clear_analysis_cache()
            assert len(summarizer.get_analysis_cache()) == 0


# ==================================================================
# PERFORMANCE AND EDGE CASE TESTS
# ==================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def get_enhanced_result_summarizer_obj_with_output(self):
        export_config = ExportConfig(
            base_output_dir=Path("/tmp/unittest_metaintegrator")
        )

        summarizer = EnhancedResultSummarizer(export_config=export_config)
        return summarizer

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()

        with patch(
            "metaintegrator.scoring_results_summarizer.MetaAnalysisValidator.validate_dataframe"
        ):
            summarizer = self.get_enhanced_result_summarizer_obj_with_output()
            result = summarizer.summarize_results(empty_df, "empty_test")

            assert result["analysis_info"]["total_genes"] == 0

    def test_single_gene_dataframe(self):
        """Test handling of single-gene DataFrame."""
        single_gene_df = pd.DataFrame(
            {"effect_size": [1.5], "p_value": [0.01]}, index=["SINGLE_GENE"]
        )

        with patch(
            "metaintegrator.scoring_results_summarizer.MetaAnalysisValidator.validate_dataframe"
        ):
            summarizer = self.get_enhanced_result_summarizer_obj_with_output()

            result = summarizer.summarize_results(single_gene_df, "single_gene_test")

            assert result["analysis_info"]["total_genes"] == 1
            # Check that effect_size statistics exist and have expected values for single gene
            assert "effect_size" in result["gene_summary"]["gene_statistics"]
            effect_stats = result["gene_summary"]["gene_statistics"]["effect_size"]
            assert effect_stats["mean"] == 1.5  # Single value should equal mean
            assert effect_stats["positive_count"] == 1

    def test_all_missing_data(self):
        """Test handling of DataFrame with all missing data."""
        all_missing_df = pd.DataFrame(
            {
                "effect_size": [np.nan, np.nan, np.nan],
                "p_value": [np.nan, np.nan, np.nan],
            },
            index=["GENE1", "GENE2", "GENE3"],
        )

        with patch(
            "metaintegrator.scoring_results_summarizer.MetaAnalysisValidator.validate_dataframe"
        ):
            summarizer = self.get_enhanced_result_summarizer_obj_with_output()

            result = summarizer.summarize_results(all_missing_df, "all_missing_test")

            # Should handle gracefully without errors
            assert result["analysis_info"]["total_genes"] == 3

    def test_non_numeric_data_only(self):
        """Test handling of DataFrame with no numeric columns."""
        non_numeric_df = pd.DataFrame(
            {
                "gene_name": ["GENE1", "GENE2", "GENE3"],
                "pathway": ["A", "B", "C"],
                "description": ["desc1", "desc2", "desc3"],
            }
        )

        with patch(
            "metaintegrator.scoring_results_summarizer.MetaAnalysisValidator.validate_dataframe"
        ):
            summarizer = self.get_enhanced_result_summarizer_obj_with_output()

            result = summarizer.summarize_results(non_numeric_df, "non_numeric_test")

            # Should handle gracefully
            assert isinstance(result, dict)

    def test_very_large_effect_sizes(self):
        """Test handling of very large effect sizes."""
        large_effects_df = pd.DataFrame(
            {
                "effect_size": [1e10, -1e10, 1e15, -1e15],
                "p_value": [0.01, 0.02, 0.03, 0.04],
            },
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        with patch(
            "metaintegrator.scoring_results_summarizer.MetaAnalysisValidator.validate_dataframe"
        ):
            summarizer = self.get_enhanced_result_summarizer_obj_with_output()

            result = summarizer.summarize_results(
                large_effects_df, "large_effects_test"
            )

            # Should detect extreme values in quality assessment
            issues = result["quality_assessment"]["data_consistency"]["issues"]
            assert any("Extreme values" in issue for issue in issues)


if __name__ == "__main__":
    pytest.main([__file__])
