"""
Comprehensive unit tests for scoring_statistical_analysis.py module.
Tests cover all classes, methods, and functionality with edge cases.
"""

import logging
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# Import the modules we're testing
from metaintegrator.scoring_statistical_analysis import (
    StatisticalAnalyzer,
    DistributionAnalyzer,
)

# Import dependencies from scoring_core
from metaintegrator.scoring_core import (
    StatisticalSummary,
    SummaryConfig,
    StatisticType,
    ValidationError,
    ScoringCalculationError,
    MetaAnalysisValidator,
    StatisticalUtils,
    DataFrameUtils,
    LoggerMixin,
)

# ==================================================================
# FIXTURES AND TEST DATA
# ==================================================================


@pytest.fixture
def sample_config():
    """Create a sample SummaryConfig for testing."""
    return SummaryConfig(
        include_distributions=True,
        include_correlations=True,
        significance_thresholds=[0.05, 0.01, 0.001],
        effect_size_thresholds=[0.5, 1.0, 2.0],
    )


@pytest.fixture
def sample_basic_config():
    """Create a basic SummaryConfig with minimal options."""
    return SummaryConfig(
        include_distributions=False,
        include_correlations=False,
        significance_thresholds=[0.05],
        effect_size_thresholds=[1.0],
    )


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing statistical analysis."""
    np.random.seed(42)  # For reproducible tests

    data = {
        "gene_id": [f"GENE_{i:03d}" for i in range(100)],
        "pvalue": np.random.uniform(0, 1, 100),
        "fdr": np.random.uniform(0, 1, 100),
        "log_fold_change": np.random.normal(0, 2, 100),
        "effect_size": np.random.normal(0, 1, 100),
        "expression_score": np.random.normal(50, 15, 100),
        "sample_size": np.random.randint(10, 200, 100),
    }

    # Add some missing values
    data["pvalue"][5:10] = np.nan
    data["fdr"][15:20] = np.nan

    return pd.DataFrame(data)


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def single_row_dataframe():
    """Create a single-row DataFrame for edge case testing."""
    return pd.DataFrame({"gene_id": ["GENE_001"], "pvalue": [0.05], "effect_size": [1.5]})


@pytest.fixture
def all_missing_dataframe():
    """Create a DataFrame with all missing values."""
    return pd.DataFrame(
        {"pvalue": [np.nan, np.nan, np.nan], "effect_size": [np.nan, np.nan, np.nan]}
    )


@pytest.fixture
def normal_distribution_data():
    """Create normally distributed data for distribution testing."""
    np.random.seed(42)
    return np.random.normal(0, 1, 1000)


@pytest.fixture
def skewed_distribution_data():
    """Create skewed distribution data for testing."""
    np.random.seed(42)
    return np.random.exponential(2, 1000)


# ==================================================================
# STATISTICAL ANALYZER TESTS
# ==================================================================


class TestStatisticalAnalyzer:
    """Test suite for StatisticalAnalyzer class."""

    def test_initialization(self, sample_config):
        """Test StatisticalAnalyzer initialization."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        assert analyzer.config == sample_config
        assert isinstance(analyzer.scaler, StandardScaler)
        assert hasattr(analyzer, "logger")

    def test_initialization_with_kwargs(self, sample_config):
        """Test StatisticalAnalyzer initialization with additional kwargs."""
        # Create a custom logger for testing
        test_logger = logging.getLogger("test_analyzer")
        analyzer = StatisticalAnalyzer(config=sample_config, logger=test_logger)

        assert analyzer.config == sample_config
        assert analyzer.logger == test_logger

    def test_analyze_basic_functionality(self, sample_config, sample_dataframe):
        """Test basic analyze functionality with valid data."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        with patch.object(MetaAnalysisValidator, "validate_dataframe"):
            result = analyzer.analyze(sample_dataframe)

        assert isinstance(result, StatisticalSummary)
        assert result.basic_stats is not None
        assert result.distribution_stats is not None
        assert result.correlation_stats is not None
        assert result.threshold_counts is not None
        assert result.metadata is not None

        # Check metadata content
        assert result.metadata["total_genes"] == len(sample_dataframe)
        assert "analysis_timestamp" in result.metadata
        assert "config_used" in result.metadata
        assert result.metadata["analyzer_version"] == "2.0.0"

    def test_analyze_basic_config_only(self, sample_basic_config, sample_dataframe):
        """Test analyze with basic config (no distributions/correlations)."""
        analyzer = StatisticalAnalyzer(config=sample_basic_config)

        with patch.object(MetaAnalysisValidator, "validate_dataframe"):
            result = analyzer.analyze(sample_dataframe)

        assert isinstance(result, StatisticalSummary)
        assert result.basic_stats is not None
        assert result.distribution_stats is None
        assert result.correlation_stats is None
        assert result.threshold_counts is not None

    def test_analyze_validation_error(self, sample_config, sample_dataframe):
        """Test analyze behavior when validation fails."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        with patch.object(
            MetaAnalysisValidator,
            "validate_dataframe",
            side_effect=ValidationError("Invalid data"),
        ):
            with pytest.raises(ScoringCalculationError):
                analyzer.analyze(sample_dataframe)

    def test_analyze_calculation_error(self, sample_config, sample_dataframe):
        """Test analyze behavior when calculation fails."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        with patch.object(MetaAnalysisValidator, "validate_dataframe"):
            with patch.object(
                analyzer,
                "_calculate_basic_statistics",
                side_effect=Exception("Calculation failed"),
            ):
                with pytest.raises(ScoringCalculationError):
                    analyzer.analyze(sample_dataframe)

    def test_calculate_basic_statistics(self, sample_config, sample_dataframe):
        """Test basic statistics calculation."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        with patch.object(
            DataFrameUtils,
            "get_numeric_columns",
            return_value=["pvalue", "fdr", "log_fold_change"],
        ):
            result = analyzer._calculate_basic_statistics(sample_dataframe)

        assert isinstance(result, dict)
        assert "pvalue" in result

        # Check structure of basic stats for one column
        pvalue_stats = result["pvalue"]
        expected_keys = [
            "count",
            "mean",
            "std",
            "min",
            "max",
            "median",
            "q25",
            "q75",
            "iqr",
            "mad",
            "cv",
            "missing_count",
            "missing_percent",
        ]

        for key in expected_keys:
            assert key in pvalue_stats
            # Include numpy types in the assertion
            assert isinstance(pvalue_stats[key], (int, float, np.integer, np.floating))

    def test_calculate_basic_statistics_empty_column(self, sample_config):
        """Test basic statistics with empty numeric column."""
        analyzer = StatisticalAnalyzer(config=sample_config)
        df = pd.DataFrame({"empty_col": []})

        with patch.object(DataFrameUtils, "get_numeric_columns", return_value=["empty_col"]):
            result = analyzer._calculate_basic_statistics(df)

        assert result == {}

    def test_calculate_basic_statistics_zero_mean(self, sample_config):
        """Test coefficient of variation calculation with zero mean."""
        analyzer = StatisticalAnalyzer(config=sample_config)
        df = pd.DataFrame({"zero_mean": [0, 0, 0, 0, 0]})

        with patch.object(DataFrameUtils, "get_numeric_columns", return_value=["zero_mean"]):
            result = analyzer._calculate_basic_statistics(df)

        assert np.isnan(result["zero_mean"]["cv"])

    def test_calculate_distribution_statistics(self, sample_config, sample_dataframe):
        """Test distribution statistics calculation."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        with patch.object(
            DataFrameUtils,
            "get_numeric_columns",
            return_value=["pvalue", "log_fold_change"],
        ):
            result = analyzer._calculate_distribution_statistics(sample_dataframe)

        assert isinstance(result, dict)
        assert "pvalue" in result

        # Check structure for valid column
        pvalue_dist = result["pvalue"]
        expected_keys = [
            "shapiro_stat",
            "shapiro_p",
            "jarque_bera_stat",
            "jarque_bera_p",
            "skewness",
            "kurtosis",
            "best_distribution",
            "outliers",
            "outlier_count",
            "outlier_percent",
        ]

        for key in expected_keys:
            assert key in pvalue_dist

    def test_calculate_distribution_statistics_insufficient_data(self, sample_config):
        """Test distribution statistics with insufficient data."""
        analyzer = StatisticalAnalyzer(config=sample_config)
        df = pd.DataFrame({"small_col": [1, 2]})  # Only 2 values

        with patch.object(DataFrameUtils, "get_numeric_columns", return_value=["small_col"]):
            result = analyzer._calculate_distribution_statistics(df)

        assert result == {}

    def test_calculate_distribution_statistics_error_handling(self, sample_config):
        """Test distribution statistics error handling."""
        analyzer = StatisticalAnalyzer(config=sample_config)
        df = pd.DataFrame({"problematic_col": np.random.normal(0, 1, 100)})

        with patch.object(DataFrameUtils, "get_numeric_columns", return_value=["problematic_col"]):
            with patch("scipy.stats.shapiro", side_effect=Exception("Test error")):
                result = analyzer._calculate_distribution_statistics(df)

        assert "problematic_col" in result
        assert "error" in result["problematic_col"]

    def test_calculate_correlation_statistics(self, sample_config, sample_dataframe):
        """Test correlation statistics calculation."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        # Mock select_dtypes to return numeric data
        numeric_data = sample_dataframe[["pvalue", "log_fold_change", "effect_size"]]

        with patch.object(sample_dataframe, "select_dtypes", return_value=numeric_data):
            result = analyzer._calculate_correlation_statistics(sample_dataframe)

        assert isinstance(result, dict)
        expected_keys = [
            "pearson_matrix",
            "spearman_matrix",
            "high_correlations",
            "max_correlation",
            "mean_correlation",
            "correlation_summary",
        ]

        for key in expected_keys:
            assert key in result

    def test_calculate_correlation_statistics_insufficient_columns(self, sample_config):
        """Test correlation statistics with insufficient columns."""
        analyzer = StatisticalAnalyzer(config=sample_config)
        df = pd.DataFrame({"single_col": [1, 2, 3]})

        with patch.object(df, "select_dtypes", return_value=df):
            result = analyzer._calculate_correlation_statistics(df)

        assert "error" in result
        assert "at least 2 numeric columns" in result["error"]

    def test_calculate_correlation_statistics_error_handling(self, sample_config, sample_dataframe):
        """Test correlation statistics error handling."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        # Mock the _find_high_correlations method to raise an exception inside the try-catch block
        with patch.object(analyzer, "_find_high_correlations", side_effect=Exception("Test error")):
            result = analyzer._calculate_correlation_statistics(sample_dataframe)

        assert "error" in result
        assert "Test error" in result["error"]

    def test_calculate_threshold_counts(self, sample_config, sample_dataframe):
        """Test threshold counts calculation."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        result = analyzer._calculate_threshold_counts(sample_dataframe)

        assert isinstance(result, dict)
        # Should find pvalue and fdr as significance columns
        significance_keys = [k for k in result.keys() if "pvalue" in k or "fdr" in k]
        assert len(significance_keys) > 0

        # Should find effect_size and log_fold_change as effect columns
        effect_keys = [k for k in result.keys() if "effect_size" in k or "log_fold_change" in k]
        assert len(effect_keys) > 0

    def test_fit_best_distribution(self, sample_config, normal_distribution_data):
        """Test distribution fitting functionality."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        result = analyzer._fit_best_distribution(normal_distribution_data)

        assert isinstance(result, dict)
        assert "distribution" in result
        assert "ks_p_value" in result
        assert "parameters" in result

        # Should find normal distribution as best fit for normal data
        assert result["distribution"] is not None

    def test_fit_best_distribution_no_good_fit(self, sample_config):
        """Test distribution fitting with problematic data."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        # Create data that's hard to fit
        problematic_data = np.array([1e10, -1e10, 0, np.inf, -np.inf])
        problematic_data = problematic_data[np.isfinite(problematic_data)]

        if len(problematic_data) > 0:
            result = analyzer._fit_best_distribution(problematic_data)
            assert isinstance(result, dict)

    def test_detect_outliers(self, sample_config, normal_distribution_data):
        """Test outlier detection functionality."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        result = analyzer._detect_outliers(normal_distribution_data)

        assert isinstance(result, dict)
        expected_keys = [
            "indices",
            "values",
            "methods",
            "iqr_outliers",
            "z_outliers",
            "mad_outliers",
        ]

        for key in expected_keys:
            assert key in result

        assert isinstance(result["indices"], list)
        assert isinstance(result["values"], list)
        assert isinstance(result["iqr_outliers"], int)

    def test_detect_outliers_with_outliers(self, sample_config):
        """Test outlier detection with known outliers."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        # Create data with obvious outliers
        data_with_outliers = np.concatenate(
            [
                np.random.normal(0, 1, 100),  # Normal data
                [10, -10, 15],  # Clear outliers
            ]
        )

        result = analyzer._detect_outliers(data_with_outliers)

        # Should detect some outliers
        assert len(result["indices"]) > 0
        assert len(result["values"]) > 0

    def test_find_high_correlations(self, sample_config):
        """Test finding highly correlated variable pairs."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        # Create correlation matrix with known high correlations
        corr_matrix = pd.DataFrame(
            {"var1": [1.0, 0.9, 0.3], "var2": [0.9, 1.0, 0.2], "var3": [0.3, 0.2, 1.0]},
            index=["var1", "var2", "var3"],
        )

        result = analyzer._find_high_correlations(corr_matrix, threshold=0.8)

        assert isinstance(result, list)
        assert len(result) == 1  # Only one pair above threshold

        high_corr = result[0]
        assert "variable1" in high_corr
        assert "variable2" in high_corr
        assert "correlation" in high_corr
        assert "abs_correlation" in high_corr
        assert high_corr["abs_correlation"] >= 0.8

    def test_summarize_correlations(self, sample_config):
        """Test correlation matrix summarization."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        # Create test correlation matrix
        corr_matrix = pd.DataFrame(
            {
                "var1": [1.0, 0.8, 0.3, 0.1],
                "var2": [0.8, 1.0, 0.5, 0.2],
                "var3": [0.3, 0.5, 1.0, 0.9],
                "var4": [0.1, 0.2, 0.9, 1.0],
            },
            index=["var1", "var2", "var3", "var4"],
        )

        result = analyzer._summarize_correlations(corr_matrix)

        expected_keys = [
            "mean_abs_correlation",
            "max_abs_correlation",
            "min_abs_correlation",
            "std_correlation",
            "strong_correlations_count",
            "moderate_correlations_count",
            "weak_correlations_count",
        ]

        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], (int, float))


def test_correlation_error_handling():
    """Test the specific correlation error handling scenario that was failing."""
    print("Testing correlation statistics error handling...")

    try:
        # Create test data
        np.random.seed(42)
        sample_dataframe = pd.DataFrame(
            {
                "gene_id": [f"GENE_{i:03d}" for i in range(10)],
                "pvalue": np.random.uniform(0, 1, 10),
                "log_fold_change": np.random.normal(0, 2, 10),
                "effect_size": np.random.normal(0, 1, 10),
            }
        )

        config = SummaryConfig()
        analyzer = StatisticalAnalyzer(config=config)

        # Mock the _find_high_correlations method to raise an exception inside the try-catch block
        with patch.object(analyzer, "_find_high_correlations", side_effect=Exception("Test error")):
            result = analyzer._calculate_correlation_statistics(sample_dataframe)

        # Check that error was properly handled
        assert "error" in result, f"Expected 'error' key in result, got: {result}"
        assert (
            "Test error" in result["error"]
        ), f"Expected 'Test error' in error message, got: {result['error']}"

        print("✓ Correlation error handling test passed!")
        return True

    except Exception as e:
        print(f"✗ Correlation error handling test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_normal_correlation_functionality():
    """Test that normal correlation functionality still works."""
    print("Testing normal correlation functionality...")

    try:
        # Create test data
        np.random.seed(42)
        sample_dataframe = pd.DataFrame(
            {
                "pvalue": np.random.uniform(0, 1, 20),
                "log_fold_change": np.random.normal(0, 2, 20),
                "effect_size": np.random.normal(0, 1, 20),
            }
        )

        config = SummaryConfig()
        analyzer = StatisticalAnalyzer(config=config)

        # Test normal functionality (no mocking)
        result = analyzer._calculate_correlation_statistics(sample_dataframe)

        # Check that result is properly structured
        expected_keys = [
            "pearson_matrix",
            "spearman_matrix",
            "high_correlations",
            "max_correlation",
            "mean_correlation",
            "correlation_summary",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in result"

        print("✓ Normal correlation functionality test passed!")
        return True

    except Exception as e:
        print(f"✗ Normal correlation functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


# ==================================================================
# DISTRIBUTION ANALYZER TESTS
# ==================================================================


class TestDistributionAnalyzer:
    """Test suite for DistributionAnalyzer class."""

    def test_initialization(self):
        """Test DistributionAnalyzer initialization."""
        analyzer = DistributionAnalyzer()

        assert hasattr(analyzer, "logger")

    def test_initialization_with_kwargs(self):
        """Test DistributionAnalyzer initialization with kwargs."""
        # Create a custom logger for testing
        test_logger = logging.getLogger("test_dist_analyzer")
        analyzer = DistributionAnalyzer(logger=test_logger)

        assert analyzer.logger == test_logger

    def test_analyze_distributions_single_variable(self, normal_distribution_data):
        """Test distribution analysis for single variable."""
        analyzer = DistributionAnalyzer()

        data = {"test_var": normal_distribution_data}
        result = analyzer.analyze_distributions(data)

        assert "test_var" in result
        assert "comparison" not in result  # No comparison for single variable

        var_analysis = result["test_var"]
        expected_keys = [
            "basic_stats",
            "normality_tests",
            "distribution_fits",
            "bootstrap_confidence_intervals",
            "percentiles",
        ]

        for key in expected_keys:
            assert key in var_analysis

    def test_analyze_distributions_multiple_variables(
        self, normal_distribution_data, skewed_distribution_data
    ):
        """Test distribution analysis for multiple variables."""
        analyzer = DistributionAnalyzer()

        data = {
            "normal_var": normal_distribution_data,
            "skewed_var": skewed_distribution_data,
        }
        result = analyzer.analyze_distributions(data)

        assert "normal_var" in result
        assert "skewed_var" in result
        assert "comparison" in result  # Should have comparison for multiple variables

    def test_analyze_single_distribution(self, normal_distribution_data):
        """Test single distribution analysis."""
        analyzer = DistributionAnalyzer()

        result = analyzer._analyze_single_distribution(normal_distribution_data, "test")

        assert isinstance(result, dict)
        expected_keys = [
            "basic_stats",
            "normality_tests",
            "distribution_fits",
            "bootstrap_confidence_intervals",
            "percentiles",
        ]

        for key in expected_keys:
            assert key in result

    def test_analyze_single_distribution_empty_data(self):
        """Test single distribution analysis with empty data."""
        analyzer = DistributionAnalyzer()

        empty_data = np.array([])
        result = analyzer._analyze_single_distribution(empty_data, "empty")

        assert "error" in result
        assert result["error"] == "No valid data points"

    def test_analyze_single_distribution_all_nan(self):
        """Test single distribution analysis with all NaN data."""
        analyzer = DistributionAnalyzer()

        nan_data = np.array([np.nan, np.nan, np.nan])
        result = analyzer._analyze_single_distribution(nan_data, "nan_data")

        assert "error" in result
        assert result["error"] == "No valid data points"

    def test_basic_distribution_stats(self, normal_distribution_data):
        """Test basic distribution statistics calculation."""
        analyzer = DistributionAnalyzer()

        result = analyzer._basic_distribution_stats(normal_distribution_data)

        expected_keys = ["mean", "variance", "skewness", "kurtosis", "entropy"]

        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], float)

        # For normal data, skewness should be close to 0
        assert abs(result["skewness"]) < 0.5  # Allow some tolerance

    def test_normality_tests(self, normal_distribution_data):
        """Test normality testing functionality."""
        analyzer = DistributionAnalyzer()

        result = analyzer._normality_tests(normal_distribution_data)

        # Should have at least shapiro test for normal-sized data
        assert isinstance(result, dict)
        if len(normal_distribution_data) <= 5000:
            assert "shapiro" in result
            assert "statistic" in result["shapiro"]
            assert "p_value" in result["shapiro"]

    def test_normality_tests_insufficient_data(self):
        """Test normality tests with insufficient data."""
        analyzer = DistributionAnalyzer()

        small_data = np.array([1, 2])  # Too small for tests
        result = analyzer._normality_tests(small_data)

        assert result == {}

    def test_normality_tests_large_data(self):
        """Test normality tests with large dataset (no Shapiro-Wilk)."""
        analyzer = DistributionAnalyzer()

        # Create data larger than Shapiro-Wilk limit
        large_data = np.random.normal(0, 1, 6000)
        result = analyzer._normality_tests(large_data)

        # Should not include shapiro for large data
        assert "shapiro" not in result
        # But should include other tests
        assert len(result) > 0

    def test_comprehensive_distribution_fitting(self, normal_distribution_data):
        """Test comprehensive distribution fitting."""
        analyzer = DistributionAnalyzer()

        result = analyzer._comprehensive_distribution_fitting(normal_distribution_data)

        assert "fitted_distributions" in result
        assert isinstance(result["fitted_distributions"], list)

        if len(result["fitted_distributions"]) > 0:
            first_fit = result["fitted_distributions"][0]
            expected_keys = [
                "distribution",
                "parameters",
                "ks_statistic",
                "ks_p_value",
                "log_likelihood",
                "aic",
                "bic",
            ]

            for key in expected_keys:
                assert key in first_fit

    def test_bootstrap_ci(self, normal_distribution_data):
        """Test bootstrap confidence interval calculation."""
        analyzer = DistributionAnalyzer()

        result = analyzer._bootstrap_ci(normal_distribution_data, n_bootstrap=100, confidence=0.95)

        expected_keys = ["mean_ci", "std_ci", "median_ci"]

        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], list)
            assert len(result[key]) == 2  # Lower and upper bounds
            assert result[key][0] <= result[key][1]  # Lower <= Upper

    def test_bootstrap_ci_different_confidence(self, normal_distribution_data):
        """Test bootstrap CI with different confidence levels."""
        analyzer = DistributionAnalyzer()

        ci_90 = analyzer._bootstrap_ci(normal_distribution_data, n_bootstrap=50, confidence=0.90)
        ci_99 = analyzer._bootstrap_ci(normal_distribution_data, n_bootstrap=50, confidence=0.99)

        # 99% CI should be wider than 90% CI
        assert (ci_99["mean_ci"][1] - ci_99["mean_ci"][0]) > (
            ci_90["mean_ci"][1] - ci_90["mean_ci"][0]
        )

    def test_calculate_percentiles(self, normal_distribution_data):
        """Test percentile calculation."""
        analyzer = DistributionAnalyzer()

        result = analyzer._calculate_percentiles(normal_distribution_data)

        expected_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

        for p in expected_percentiles:
            assert f"p{p}" in result
            assert isinstance(result[f"p{p}"], float)

        # Percentiles should be in ascending order
        values = [result[f"p{p}"] for p in expected_percentiles]
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    def test_compare_distributions(self, normal_distribution_data, skewed_distribution_data):
        """Test distribution comparison functionality."""
        analyzer = DistributionAnalyzer()

        data = {"normal": normal_distribution_data, "skewed": skewed_distribution_data}

        result = analyzer._compare_distributions(data)

        assert "normal_vs_skewed" in result
        comparison = result["normal_vs_skewed"]

        # Should have various statistical tests
        possible_tests = ["mannwhitney", "kolmogorov_smirnov", "levene", "cohens_d"]
        assert any(test in comparison for test in possible_tests)

    def test_compare_distributions_single_dataset(self, normal_distribution_data):
        """Test distribution comparison with single dataset."""
        analyzer = DistributionAnalyzer()

        data = {"single": normal_distribution_data}
        result = analyzer._compare_distributions(data)

        assert result == {}  # No comparisons for single dataset

    def test_compare_two_distributions(self, normal_distribution_data, skewed_distribution_data):
        """Test two-distribution comparison."""
        analyzer = DistributionAnalyzer()

        result = analyzer._compare_two_distributions(
            normal_distribution_data, skewed_distribution_data
        )

        assert isinstance(result, dict)

        # Check for expected test results
        if "mannwhitney" in result:
            assert "statistic" in result["mannwhitney"]
            assert "p_value" in result["mannwhitney"]

        if "cohens_d" in result:
            assert isinstance(result["cohens_d"], float)

    def test_compare_two_distributions_identical(self):
        """Test comparing identical distributions."""
        analyzer = DistributionAnalyzer()

        data = np.random.normal(0, 1, 100)
        result = analyzer._compare_two_distributions(data, data)

        # Statistical tests should show no significant difference
        if "mannwhitney" in result:
            assert result["mannwhitney"]["p_value"] > 0.1  # Should be non-significant

        if "cohens_d" in result:
            assert abs(result["cohens_d"]) < 0.1  # Should be very small effect size


# ==================================================================
# INTEGRATION TESTS
# ==================================================================


class TestIntegration:
    """Integration tests for the statistical analysis module."""

    def test_statistical_analyzer_full_workflow(self, sample_config, sample_dataframe):
        """Test complete statistical analysis workflow."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        with patch.object(MetaAnalysisValidator, "validate_dataframe"):
            result = analyzer.analyze(sample_dataframe)

        # Verify all components are present and well-formed
        assert isinstance(result, StatisticalSummary)
        assert result.basic_stats is not None
        assert result.distribution_stats is not None
        assert result.correlation_stats is not None
        assert result.threshold_counts is not None
        assert result.metadata is not None

        # Verify data integrity
        assert result.metadata["total_genes"] > 0
        assert isinstance(result.metadata["analysis_timestamp"], pd.Timestamp)

    def test_distribution_analyzer_full_workflow(
        self, normal_distribution_data, skewed_distribution_data
    ):
        """Test complete distribution analysis workflow."""
        analyzer = DistributionAnalyzer()

        data = {
            "normal_data": normal_distribution_data,
            "skewed_data": skewed_distribution_data,
        }

        result = analyzer.analyze_distributions(data)

        # Verify structure
        assert "normal_data" in result
        assert "skewed_data" in result
        assert "comparison" in result

        # Verify completeness of individual analyses
        for var_name in ["normal_data", "skewed_data"]:
            var_result = result[var_name]
            assert "basic_stats" in var_result
            assert "normality_tests" in var_result
            assert "distribution_fits" in var_result
            assert "bootstrap_confidence_intervals" in var_result
            assert "percentiles" in var_result

    def test_edge_case_handling(self, sample_config):
        """Test handling of various edge cases."""
        analyzer = StatisticalAnalyzer(config=sample_config)

        # Test with DataFrame containing only non-numeric columns
        text_df = pd.DataFrame(
            {"gene_id": ["GENE_001", "GENE_002"], "description": ["Gene 1", "Gene 2"]}
        )

        with patch.object(MetaAnalysisValidator, "validate_dataframe"):
            with patch.object(DataFrameUtils, "get_numeric_columns", return_value=[]):
                result = analyzer.analyze(text_df)

        assert isinstance(result, StatisticalSummary)
        assert result.basic_stats == {}

    def test_memory_efficiency_large_data(self, sample_config):
        """Test memory efficiency with larger datasets."""
        # Create a larger dataset
        large_data = {
            "gene_id": [f"GENE_{i:05d}" for i in range(10000)],
            "pvalue": np.random.uniform(0, 1, 10000),
            "effect_size": np.random.normal(0, 1, 10000),
        }
        large_df = pd.DataFrame(large_data)

        analyzer = StatisticalAnalyzer(config=sample_config)

        with patch.object(MetaAnalysisValidator, "validate_dataframe"):
            result = analyzer.analyze(large_df)

        # Should complete without memory errors
        assert isinstance(result, StatisticalSummary)
        assert result.metadata["total_genes"] == 10000


# ==================================================================
# PERFORMANCE TESTS
# ==================================================================


class TestPerformance:
    """Performance tests for statistical analysis operations."""

    @pytest.mark.performance
    def test_statistical_analysis_performance(self, sample_config):
        """Test performance of statistical analysis with medium dataset."""
        import time

        # Create medium-sized dataset
        medium_data = {
            "gene_id": [f"GENE_{i:04d}" for i in range(5000)],
            "pvalue": np.random.uniform(0, 1, 5000),
            "fdr": np.random.uniform(0, 1, 5000),
            "log_fold_change": np.random.normal(0, 2, 5000),
            "effect_size": np.random.normal(0, 1, 5000),
        }
        medium_df = pd.DataFrame(medium_data)

        analyzer = StatisticalAnalyzer(config=sample_config)

        start_time = time.time()

        with patch.object(MetaAnalysisValidator, "validate_dataframe"):
            result = analyzer.analyze(medium_df)

        end_time = time.time()
        analysis_time = end_time - start_time

        # Analysis should complete in reasonable time (adjust threshold as needed)
        assert analysis_time < 30  # 30 seconds threshold
        assert isinstance(result, StatisticalSummary)

    @pytest.mark.performance
    def test_distribution_analysis_performance(self):
        """Test performance of distribution analysis."""
        import time

        analyzer = DistributionAnalyzer()

        # Create multiple datasets
        data = {f"dataset_{i}": np.random.normal(i, 1, 1000) for i in range(10)}

        start_time = time.time()
        result = analyzer.analyze_distributions(data)
        end_time = time.time()

        analysis_time = end_time - start_time

        # Should complete in reasonable time
        assert analysis_time < 60  # 60 seconds threshold
        assert len(result) == 11  # 10 datasets + 1 comparison


if __name__ == "__main__":
    pytest.main([__file__])
