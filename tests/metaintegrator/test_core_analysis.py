"""
Comprehensive pytest unit tests for metaintegrator.core_analysis module.

This test suite provides extensive coverage of MetaAnalysisCore and FisherCombination classes
using pytest features like fixtures, parametrization, and markers.
"""

import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.core_analysis import MetaAnalysisCore, FisherCombination

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_expression_data():
    """Sample expression data for testing."""
    return pd.DataFrame(
        {
            "sample1": [1.0, 2.0, 3.0, 4.0],
            "sample2": [1.5, 2.5, 3.5, 4.5],
            "sample3": [2.0, 3.0, 4.0, 5.0],
            "sample4": [2.5, 3.5, 4.5, 5.5],
        },
        index=["gene1", "gene2", "gene3", "gene4"],
    )


@pytest.fixture
def sample_study(sample_expression_data):
    """Sample study dictionary for testing."""
    return {
        "expr": sample_expression_data,
        "class": np.array([0, 0, 1, 1]),  # Controls and cases
        "keys": ["symbol1", "symbol2", "symbol3", "symbol4"],
    }


@pytest.fixture
def sample_gem_data(sample_expression_data):
    """Sample GEM data for testing."""
    return {
        "expr": sample_expression_data,
        "keys": ["symbol1", "symbol2", "symbol3", "symbol4"],
    }


@pytest.fixture
def differential_expression_data():
    """Expression data with known differential expression for realistic testing."""
    np.random.seed(42)  # For reproducible results

    n_genes = 6
    n_samples_per_group = 4

    # Create base expression data
    expr_data = pd.DataFrame(
        np.random.randn(n_genes, n_samples_per_group * 2),
        index=[f"gene_{i}" for i in range(n_genes)],
        columns=[f"sample_{i}" for i in range(n_samples_per_group * 2)],
    )

    # Add differential expression to first 3 genes
    expr_data.iloc[:3, n_samples_per_group:] += 2.0  # Cases higher

    return {
        "expr": expr_data,
        "class": np.array([0] * n_samples_per_group + [1] * n_samples_per_group),
        "keys": [f"symbol_{i}" for i in range(n_genes)],
    }


@pytest.fixture
def sample_significance_data():
    """Sample significance data for Fisher combination testing."""
    return [
        pd.DataFrame(
            {
                "P.up": [0.01, 0.05, 0.1],
                "P.down": [0.99, 0.95, 0.9],
                "keys": ["A", "B", "C"],
            }
        ),
        pd.DataFrame(
            {
                "P.up": [0.02, 0.06, 0.15],
                "P.down": [0.98, 0.94, 0.85],
                "keys": ["A", "B", "D"],
            }
        ),
    ]


@pytest.fixture
def effect_sizes_data():
    """Sample effect sizes data for testing within-study summarization."""
    return pd.DataFrame(
        {
            "keys": ["A", "A", "B", "C", "C", "C"],
            "g": [1.0, 1.5, 2.0, 0.5, 0.8, 1.2],
            "se_g": [0.2, 0.3, 0.4, 0.1, 0.15, 0.25],
        }
    )


# =============================================================================
# Test MetaAnalysisCore Class
# =============================================================================


class TestMetaAnalysisCore:
    """Test suite for MetaAnalysisCore class."""

    # -------------------------------------------------------------------------
    # Annotation and Data Processing Tests
    # -------------------------------------------------------------------------

    def test_create_annotation_table_success(self, sample_gem_data):
        """Test successful annotation table creation."""
        result = MetaAnalysisCore.create_annotation_table(sample_gem_data)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["probeid", "symbol"]
        assert len(result) == 4
        assert result["probeid"].tolist() == ["gene1", "gene2", "gene3", "gene4"]
        assert result["symbol"].tolist() == ["symbol1", "symbol2", "symbol3", "symbol4"]

    def test_create_annotation_table_missing_expr_key(self, sample_gem_data):
        """Test annotation table creation with missing 'expr' key."""
        invalid_data = {
            key: val for key, val in sample_gem_data.items() if key != "expr"
        }

        with pytest.raises(KeyError):
            MetaAnalysisCore.create_annotation_table(invalid_data)

    def test_create_annotation_table_missing_keys_key(self, sample_gem_data):
        """Test annotation table creation with missing 'keys' key."""
        invalid_data = {
            key: val for key, val in sample_gem_data.items() if key != "keys"
        }

        with pytest.raises(KeyError):
            MetaAnalysisCore.create_annotation_table(invalid_data)

    # -------------------------------------------------------------------------
    # Study Validation Tests
    # -------------------------------------------------------------------------

    def test_check_metagem_study_valid(self, sample_study):
        """Test validation of valid study."""
        # Should not raise any exception
        MetaAnalysisCore.check_metagem_study(sample_study)

    @pytest.mark.parametrize("missing_key", ["expr", "class", "keys"])
    def test_check_metagem_study_missing_keys(self, sample_study, missing_key):
        """Test validation with missing required keys."""
        invalid_study = {
            key: val for key, val in sample_study.items() if key != missing_key
        }

        with pytest.raises(ValueError, match="Missing required keys"):
            MetaAnalysisCore.check_metagem_study(invalid_study)

    @pytest.mark.parametrize(
        "invalid_classes",
        [
            np.array([1, 2, 3, 4]),  # Not 0/1
            np.array([0, 1, 2]),  # Contains other values
            np.array([0.5, 1.5]),  # Non-integer
        ],
    )
    def test_check_metagem_study_invalid_class_coding(
        self, sample_study, invalid_classes
    ):
        """Test validation with invalid class coding."""
        invalid_study = sample_study.copy()
        invalid_study["class"] = invalid_classes

        with pytest.raises(ValueError, match="must be coded as 0 or 1"):
            MetaAnalysisCore.check_metagem_study(invalid_study)

    @pytest.mark.parametrize(
        "invalid_keys",
        [
            "not_a_list",  # String instead of list
            42,  # Integer instead of list
            {"a": "dict"},  # Dictionary instead of list
        ],
    )
    def test_check_metagem_study_invalid_keys_type(self, sample_study, invalid_keys):
        """Test validation with invalid keys type."""
        invalid_study = sample_study.copy()
        invalid_study["keys"] = invalid_keys

        with pytest.raises(ValueError, match="Keys must be stored as a list or array"):
            MetaAnalysisCore.check_metagem_study(invalid_study)

    # -------------------------------------------------------------------------
    # Data Cleaning Tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "input_array,expected",
        [
            (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),  # Clean array
            (np.array([1.0, np.nan, 3.0]), np.array([1.0, 3.0])),  # With NaN
            (np.array([1.0, np.inf, 3.0]), np.array([1.0, 3.0])),  # With +Inf
            (np.array([1.0, -np.inf, 3.0]), np.array([1.0, 3.0])),  # With -Inf
            (np.array([np.nan, np.inf, -np.inf]), np.array([])),  # All invalid
        ],
    )
    def test_clean_na(self, input_array, expected):
        """Test NaN and infinite value removal."""
        result = MetaAnalysisCore.clean_na(input_array)
        np.testing.assert_array_equal(result, expected)

    # -------------------------------------------------------------------------
    # DataFrame Expansion Tests
    # -------------------------------------------------------------------------

    def test_expand_dataframe_basic(self):
        """Test basic DataFrame expansion with comma-separated keys."""
        df = pd.DataFrame({"keys": ["A,B", "C", "D,E,F"], "value": [1, 2, 3]})

        result = MetaAnalysisCore.expand_dataframe(df)

        assert len(result) == 6  # 2 + 1 + 3
        assert sorted(result["keys"].tolist()) == ["A", "B", "C", "D", "E", "F"]

        # Check values are preserved
        assert result[result["keys"] == "A"]["value"].iloc[0] == 1
        assert result[result["keys"] == "C"]["value"].iloc[0] == 2
        assert result[result["keys"] == "F"]["value"].iloc[0] == 3

    @pytest.mark.parametrize(
        "separator,keys,expected_count",
        [(",", "A,B,C", 3), ("|", "A|B", 2), (";", "A;B;C;D", 4), (":", "single", 1)],
    )
    def test_expand_dataframe_separators(self, separator, keys, expected_count):
        """Test DataFrame expansion with different separators."""
        df = pd.DataFrame({"keys": [keys], "value": [1]})
        result = MetaAnalysisCore.expand_dataframe(df, keys_sep=separator)

        assert len(result) == expected_count

    def test_expand_dataframe_whitespace_handling(self):
        """Test expansion handles whitespace in keys correctly."""
        df = pd.DataFrame({"keys": ["A, B ", " C,D "], "value": [1, 2]})

        result = MetaAnalysisCore.expand_dataframe(df)

        # Keys should be stripped of whitespace
        expected_keys = ["A", "B", "C", "D"]
        assert sorted(result["keys"].tolist()) == sorted(expected_keys)

    def test_expand_dataframe_custom_key_column(self):
        """Test expansion with custom key column name."""
        df = pd.DataFrame({"symbols": ["A,B", "C"], "value": [1, 2]})

        result = MetaAnalysisCore.expand_dataframe(df, key_name="symbols")

        assert len(result) == 3
        assert "symbols" in result.columns

    # -------------------------------------------------------------------------
    # Effect Size Calculation Tests
    # -------------------------------------------------------------------------

    def test_get_effect_size_valid_data(self):
        """Test effect size calculation with valid data."""
        expression = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        groups = np.array([0, 0, 0, 1, 1, 1])

        result = MetaAnalysisCore.get_effect_size(expression, groups)

        # Check all required keys are present
        expected_keys = [
            "n1",
            "m1",
            "sd1",
            "n2",
            "m2",
            "sd2",
            "diff",
            "pooled_sd",
            "g",
            "se_g",
        ]
        assert all(key in result for key in expected_keys)

        # Check basic calculations
        assert result["n1"] == 3  # Cases
        assert result["n2"] == 3  # Controls
        assert result["m1"] == 5.0  # Mean of cases
        assert result["m2"] == 2.0  # Mean of controls
        assert result["diff"] == 3.0  # Difference
        assert result["g"] > 0  # Effect size should be positive

    def test_get_effect_size_mismatched_lengths(self):
        """Test effect size calculation with mismatched array lengths."""
        expression = np.array([1.0, 2.0, 3.0])
        groups = np.array([0, 0, 1, 1])  # Different length

        with pytest.raises(ValueError, match="must have same length"):
            MetaAnalysisCore.get_effect_size(expression, groups)

    @pytest.mark.parametrize(
        "n_cases,n_controls",
        [
            (1, 1),  # One sample each
            (0, 2),  # No cases
            (2, 0),  # No controls
        ],
    )
    def test_get_effect_size_insufficient_samples(self, n_cases, n_controls):
        """Test effect size calculation with insufficient samples."""
        total_samples = n_cases + n_controls
        expression = np.arange(total_samples, dtype=float)
        groups = np.array([1] * n_cases + [0] * n_controls)

        result = MetaAnalysisCore.get_effect_size(expression, groups)

        # All values should be NaN
        for key in [
            "n1",
            "m1",
            "sd1",
            "n2",
            "m2",
            "sd2",
            "diff",
            "pooled_sd",
            "g",
            "se_g",
        ]:
            assert np.isnan(result[key])

    def test_get_effect_size_with_nan_values(self):
        """Test effect size calculation with NaN values."""
        expression = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        groups = np.array([0, 0, 0, 1, 1, 1])

        result = MetaAnalysisCore.get_effect_size(expression, groups)

        # Should work by excluding NaN values
        assert result["n1"] == 2  # Two valid cases
        assert result["n2"] == 2  # Two valid controls
        assert not np.isnan(result["g"])

    def test_calculate_effect_sizes(self, sample_study):
        """Test calculation of effect sizes for entire study."""
        result = MetaAnalysisCore.calculate_effect_sizes(sample_study)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # Four genes

        # Check required columns
        expected_columns = [
            "n1",
            "m1",
            "sd1",
            "n2",
            "m2",
            "sd2",
            "diff",
            "pooled_sd",
            "g",
            "se_g",
            "keys",
        ]
        assert all(col in result.columns for col in expected_columns)

        # Check keys are preserved
        expected_keys = ["symbol1", "symbol2", "symbol3", "symbol4"]
        assert result["keys"].tolist() == expected_keys

    def test_calculate_effect_sizes_invalid_study(self, sample_expression_data):
        """Test effect size calculation with invalid study."""
        invalid_study = {"expr": sample_expression_data}

        with pytest.raises(ValueError):
            MetaAnalysisCore.calculate_effect_sizes(invalid_study)

    # -------------------------------------------------------------------------
    # Effect Summarization Tests
    # -------------------------------------------------------------------------

    def test_summarize_effects_within_study_fixed_iv(self, effect_sizes_data):
        """Test within-study effect summarization using fixed effects."""
        result = MetaAnalysisCore.summarize_effects_within_study(
            effect_sizes_data, method="fixed_iv"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Three unique keys: A, B, C
        assert sorted(result.index) == ["A", "B", "C"]

        # Check that singletons are preserved
        assert np.isclose(result.loc["B"]["g"], 2.0)

        # Check that multiples were pooled (A and C should have different values)
        assert result.loc["A"]["g"] != 1.0  # Should be weighted average
        assert result.loc["C"]["g"] != 0.5  # Should be weighted average

    def test_summarize_effects_within_study_extreme(self, effect_sizes_data):
        """Test within-study effect summarization using extreme method."""
        result = MetaAnalysisCore.summarize_effects_within_study(
            effect_sizes_data, method="extreme"
        )

        assert len(result) == 3

        # For 'A': g=[1.0, 1.5], se_g=[0.2, 0.3] -> z-scores=[5.0, 5.0]
        # Should select one of them (both have same |z|)
        assert result.loc["A"]["g"] in [1.0, 1.5]

        # For 'C': g=[0.5, 0.8, 1.2], se_g=[0.1, 0.15, 0.25]
        # z-scores=[5.0, 5.33, 4.8] -> should select 0.8 (highest |z|)
        assert np.isclose(result.loc["C"]["g"], 0.8)

    @pytest.mark.parametrize("missing_col", ["g", "se_g", "keys"])
    def test_summarize_effects_missing_columns(self, effect_sizes_data, missing_col):
        """Test summarization with missing required columns."""
        incomplete_data = effect_sizes_data.drop(columns=[missing_col])

        with pytest.raises(ValueError, match="Missing required columns"):
            MetaAnalysisCore.summarize_effects_within_study(incomplete_data)

    def test_summarize_effects_single_row(self):
        """Test summarization with single row."""
        single_row = pd.DataFrame({"keys": ["A"], "g": [1.5], "se_g": [0.2]})

        result = MetaAnalysisCore.summarize_effects_within_study(single_row)

        assert len(result) == 1
        assert result.loc["A"]["g"] == 1.5
        assert result.loc["A"]["se_g"] == 0.2

    # -------------------------------------------------------------------------
    # Inverse Variance Pooling Tests
    # -------------------------------------------------------------------------

    def test_pool_inverse_variance_fixed_effects(self):
        """Test inverse variance pooling with fixed effects."""
        effect_sizes = pd.DataFrame(
            {"study1": [1.0, 2.0], "study2": [1.5, 2.5], "study3": [2.0, 3.0]},
            index=["gene1", "gene2"],
        )

        standard_errors = pd.DataFrame(
            {"study1": [0.5, 0.8], "study2": [0.6, 0.7], "study3": [0.4, 0.9]},
            index=["gene1", "gene2"],
        )

        result = MetaAnalysisCore.pool_inverse_variance(
            effect_sizes, standard_errors, method="fixed"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

        expected_columns = [
            "n_studies",
            "summary",
            "se_summary",
            "tau2",
            "p_value",
            "Q",
            "df",
            "pval_het",
        ]
        assert all(col in result.columns for col in expected_columns)

        # Check studies counted correctly
        assert result.loc["gene1"]["n_studies"] == 3
        assert result.loc["gene2"]["n_studies"] == 3

        # For fixed effects, tau2 should be 0
        assert result.loc["gene1"]["tau2"] == 0
        assert result.loc["gene2"]["tau2"] == 0

    def test_pool_inverse_variance_random_effects(self):
        """Test inverse variance pooling with random effects."""
        # Create data with high heterogeneity
        effect_sizes = pd.DataFrame(
            {
                "study1": [0.0, 1.0],
                "study2": [2.0, 2.0],
                "study3": [4.0, 3.0],  # High heterogeneity
            },
            index=["gene1", "gene2"],
        )

        standard_errors = pd.DataFrame(
            {"study1": [0.3, 0.3], "study2": [0.3, 0.3], "study3": [0.3, 0.3]},
            index=["gene1", "gene2"],
        )

        result = MetaAnalysisCore.pool_inverse_variance(
            effect_sizes, standard_errors, method="random"
        )

        # With high heterogeneity, tau2 should be > 0 for gene1
        assert result.loc["gene1"]["tau2"] > 0

    def test_pool_inverse_variance_with_missing_data(self):
        """Test pooling with missing data."""
        effect_sizes = pd.DataFrame(
            {"study1": [1.0, np.nan], "study2": [np.nan, 2.5], "study3": [2.0, 3.0]},
            index=["gene1", "gene2"],
        )

        standard_errors = pd.DataFrame(
            {"study1": [0.5, np.nan], "study2": [np.nan, 0.7], "study3": [0.4, 0.9]},
            index=["gene1", "gene2"],
        )

        result = MetaAnalysisCore.pool_inverse_variance(effect_sizes, standard_errors)

        # Should handle missing data correctly
        assert result.loc["gene1"]["n_studies"] == 2  # Only 2 valid studies
        assert result.loc["gene2"]["n_studies"] == 2

    def test_pool_inverse_variance_mismatched_indices(self):
        """Test pooling with mismatched indices."""
        effect_sizes = pd.DataFrame({"study1": [1.0, 2.0]}, index=["gene1", "gene2"])
        standard_errors = pd.DataFrame({"study1": [0.5, 0.8]}, index=["gene1", "gene3"])

        with pytest.raises(ValueError, match="must have same row indices"):
            MetaAnalysisCore.pool_inverse_variance(effect_sizes, standard_errors)

    def test_pool_inverse_variance_no_valid_data(self):
        """Test pooling with no valid data."""
        effect_sizes = pd.DataFrame(
            {"study1": [np.nan, np.nan]}, index=["gene1", "gene2"]
        )

        standard_errors = pd.DataFrame(
            {"study1": [np.nan, np.nan]}, index=["gene1", "gene2"]
        )

        result = MetaAnalysisCore.pool_inverse_variance(effect_sizes, standard_errors)

        # Should return appropriate values for no data
        assert result.loc["gene1"]["n_studies"] == 0
        assert np.isnan(result.loc["gene1"]["summary"])

    def test_pool_inverse_variance_single_study(self):
        """Test pooling with single study."""
        effect_sizes = pd.DataFrame({"study1": [1.5, 2.5]}, index=["gene1", "gene2"])
        standard_errors = pd.DataFrame({"study1": [0.5, 0.8]}, index=["gene1", "gene2"])

        result = MetaAnalysisCore.pool_inverse_variance(effect_sizes, standard_errors)

        # With single study, summary should equal original values
        assert result.loc["gene1"]["n_studies"] == 1
        assert result.loc["gene1"]["summary"] == 1.5
        assert result.loc["gene1"]["se_summary"] == 0.5

        # Heterogeneity tests should be NaN for single study
        assert np.isnan(result.loc["gene1"]["Q"])
        assert np.isnan(result.loc["gene1"]["pval_het"])

    # -------------------------------------------------------------------------
    # T-test P-value Tests
    # -------------------------------------------------------------------------

    def test_get_ttest_pvalues(self, sample_study):
        """Test t-test p-value calculation."""
        result = MetaAnalysisCore.get_ttest_pvalues(
            sample_study["expr"], sample_study["class"]
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # Four genes

        expected_columns = ["P.both", "P.down", "P.up"]
        assert all(col in result.columns for col in expected_columns)

        # Check that p-values are valid (between 0 and 1)
        for col in expected_columns:
            valid_pvals = result[col].dropna()
            assert all(0 <= p <= 1 for p in valid_pvals)

    def test_get_ttest_pvalues_insufficient_samples(self):
        """Test t-test with insufficient samples."""
        expr_data = pd.DataFrame({"sample1": [1.0], "sample2": [2.0]}, index=["gene1"])

        class_vector = np.array([0, 1])  # Only one sample per group

        result = MetaAnalysisCore.get_ttest_pvalues(expr_data, class_vector)

        # Should return NaN for insufficient samples
        assert np.isnan(result.loc["gene1"]["P.both"])
        assert np.isnan(result.loc["gene1"]["P.down"])
        assert np.isnan(result.loc["gene1"]["P.up"])

    def test_get_ttest_pvalues_directional(self, differential_expression_data):
        """Test that directional p-values make sense."""
        result = MetaAnalysisCore.get_ttest_pvalues(
            differential_expression_data["expr"], differential_expression_data["class"]
        )

        # For genes with higher expression in cases (first 3 genes),
        # P_up should be smaller than P_down
        for i in range(3):
            gene_name = f"gene_{i}"
            if gene_name in result.index:
                p_up = result.loc[gene_name]["P.up"]
                p_down = result.loc[gene_name]["P.down"]
                if not (np.isnan(p_up) or np.isnan(p_down)):
                    assert (
                        p_up < p_down
                    ), f"Gene {gene_name}: P_up should be < P_down for upregulated genes"


# =============================================================================
# Test FisherCombination Class
# =============================================================================


class TestFisherCombination:
    """Test suite for FisherCombination class."""

    # -------------------------------------------------------------------------
    # Study P-value Calculation Tests
    # -------------------------------------------------------------------------

    def test_calculate_study_pvalues(self, sample_study):
        """Test study p-value calculation."""
        result = FisherCombination.calculate_study_pvalues(sample_study)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # Four genes

        expected_columns = ["P.up", "P.down", "keys"]
        assert all(col in result.columns for col in expected_columns)

        # Check keys are preserved
        expected_keys = ["symbol1", "symbol2", "symbol3", "symbol4"]
        assert result["keys"].tolist() == expected_keys

        # Check that p-values are valid
        for col in ["P.up", "P.down"]:
            valid_pvals = result[col].dropna()
            assert all(0 <= p <= 1 for p in valid_pvals)

    def test_calculate_study_pvalues_invalid_study(self, sample_expression_data):
        """Test p-value calculation with invalid study."""
        invalid_study = {"expr": sample_expression_data}

        with pytest.raises(ValueError):
            FisherCombination.calculate_study_pvalues(invalid_study)

    # -------------------------------------------------------------------------
    # Significance Summarization Tests
    # -------------------------------------------------------------------------

    def test_summarize_significance_within_study(self):
        """Test significance summarization within study."""
        sigs = pd.DataFrame(
            {
                "P.up": [0.01, 0.05, 0.1, 0.02],
                "P.down": [0.99, 0.95, 0.9, 0.98],
                "keys": ["A", "A", "B", "C"],
            }
        )

        result = FisherCombination.summarize_significance_within_study(sigs)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Three unique keys

        # Check that minimum values were selected for 'A'
        assert result.loc["A"]["P.up"] == 0.01  # min(0.01, 0.05)
        assert result.loc["A"]["P.down"] == 0.95  # min(0.99, 0.95)

        # Check singletons are preserved
        assert result.loc["B"]["P.up"] == 0.1
        assert result.loc["C"]["P.up"] == 0.02

    # -------------------------------------------------------------------------
    # Fisher Method Application Tests
    # -------------------------------------------------------------------------

    def test_apply_fisher_vectorized(self):
        """Test vectorized Fisher method application."""
        p_matrix = pd.DataFrame(
            {
                "study1": [0.01, 0.05, np.nan],
                "study2": [0.02, np.nan, 0.1],
                "study3": [np.nan, 0.06, 0.2],
            },
            index=["gene1", "gene2", "gene3"],
        )

        result = FisherCombination._apply_fisher_vectorized(p_matrix, "test")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

        expected_columns = ["F.stat.test", "F.pval.test"]
        assert all(col in result.columns for col in expected_columns)

        # Check Fisher statistic calculation for gene1
        # -2*log(0.01*0.02) = -2*log(0.0002) ≈ 17.73
        expected_stat = -2 * np.log(0.01 * 0.02)
        assert np.isclose(result.loc["gene1"]["F.stat.test"], expected_stat, rtol=1e-10)

        # P-value should be significant for gene1 (df = 4)
        assert result.loc["gene1"]["F.pval.test"] < 0.01

    def test_apply_fisher_vectorized_no_valid_data(self):
        """Test Fisher method with no valid data."""
        p_matrix = pd.DataFrame(
            {"study1": [np.nan, np.nan], "study2": [np.nan, np.nan]},
            index=["gene1", "gene2"],
        )

        result = FisherCombination._apply_fisher_vectorized(p_matrix, "test")

        # Should return NaN for both genes
        assert np.isnan(result.loc["gene1"]["F.stat.test"])
        assert np.isnan(result.loc["gene1"]["F.pval.test"])
        assert np.isnan(result.loc["gene2"]["F.stat.test"])
        assert np.isnan(result.loc["gene2"]["F.pval.test"])

    @pytest.mark.parametrize(
        "p_values,expected_significant",
        [
            ([0.001, 0.002], True),  # Very significant
            ([0.3, 0.4], False),  # Not significant
            ([0.01, 0.05], True),  # Moderately significant
        ],
    )
    def test_apply_fisher_vectorized_significance_levels(
        self, p_values, expected_significant
    ):
        """Test Fisher method with different significance levels."""
        p_matrix = pd.DataFrame(
            {"study1": [p_values[0]], "study2": [p_values[1]]}, index=["gene1"]
        )

        result = FisherCombination._apply_fisher_vectorized(p_matrix, "test")

        combined_p = result.loc["gene1"]["F.pval.test"]
        if expected_significant:
            assert combined_p < 0.05
        else:
            assert combined_p >= 0.05

    # -------------------------------------------------------------------------
    # Multi-merge Tests
    # -------------------------------------------------------------------------

    def test_multimerge_basic(self):
        """Test basic dataframe merging."""
        df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["gene1", "gene2"])
        df2 = pd.DataFrame({"C": [5, 6], "D": [7, 8]}, index=["gene1", "gene3"])

        result = FisherCombination.multimerge([df1, df2])

        assert isinstance(result, pd.DataFrame)

        # Check column names have prefixes
        expected_columns = ["study_0_A", "study_0_B", "study_1_C", "study_1_D"]
        assert all(col in result.columns for col in expected_columns)

        # Check index union
        expected_index = ["gene1", "gene2", "gene3"]
        assert sorted(result.index.tolist()) == sorted(expected_index)

        # Check values are preserved where they exist
        assert result.loc["gene1"]["study_0_A"] == 1
        assert result.loc["gene2"]["study_0_B"] == 4

        # Check NaN for missing combinations
        assert np.isnan(result.loc["gene3"]["study_0_A"])
        assert np.isnan(result.loc["gene2"]["study_1_C"])

    def test_multimerge_empty_list(self):
        """Test multimerge with empty list."""
        result = FisherCombination.multimerge([])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_multimerge_single_dataframe(self):
        """Test multimerge with single dataframe."""
        df = pd.DataFrame({"A": [1, 2]}, index=["gene1", "gene2"])

        result = FisherCombination.multimerge([df])

        # Should have prefix
        assert "study_0_A" in result.columns
        assert len(result.columns) == 1
        assert len(result) == 2

    @pytest.mark.parametrize("n_dataframes", [2, 3, 5])
    def test_multimerge_multiple_dataframes(self, n_dataframes):
        """Test multimerge with multiple dataframes."""
        dfs = []
        for i in range(n_dataframes):
            df = pd.DataFrame(
                {f"col_{i}": [i * 10 + j for j in range(3)]},
                index=[f"gene_{j}" for j in range(3)],
            )
            dfs.append(df)

        result = FisherCombination.multimerge(dfs)

        # Should have correct number of columns
        assert len(result.columns) == n_dataframes

        # Check prefixes are correct
        for i in range(n_dataframes):
            assert f"study_{i}_col_{i}" in result.columns

    # -------------------------------------------------------------------------
    # Sum of Logs Tests
    # -------------------------------------------------------------------------

    def test_sum_of_logs_basic(self, sample_significance_data):
        """Test basic sum of logs Fisher combination."""
        result = FisherCombination.sum_of_logs(sample_significance_data)

        assert isinstance(result, pd.DataFrame)

        expected_columns = ["F.stat.up", "F.pval.up", "F.stat.down", "F.pval.down"]
        assert all(col in result.columns for col in expected_columns)

        # Should have results for genes in both studies (A, B) and singletons (C, D)
        assert len(result) >= 2  # At least A and B should be present

    def test_sum_of_logs_with_invalid_data(self):
        """Test sum of logs with invalid data that gets filtered."""
        sigs_with_invalid = pd.DataFrame(
            {
                "P.up": [0.01, np.nan, 0.05, 0.1],
                "P.down": [0.99, 0.95, np.nan, 0.9],
                "keys": ["A", "B", "NA", "C"],
            }
        )

        result = FisherCombination.sum_of_logs([sigs_with_invalid])

        # Should filter out invalid entries
        valid_genes = result.index[result["F.stat.up"].notna()].tolist()

        # Should include A and C (valid entries)
        assert "A" in valid_genes
        assert "C" in valid_genes

    def test_sum_of_logs_empty_input(self):
        """Test sum of logs with empty input."""
        result = FisherCombination.sum_of_logs([])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.slow
    def test_sum_of_logs_processing_pipeline(self, sample_significance_data):
        """Test the complete processing pipeline in sum_of_logs."""
        with patch.object(
            MetaAnalysisCore, "expand_dataframe"
        ) as mock_expand, patch.object(
            FisherCombination, "summarize_significance_within_study"
        ) as mock_summarize:
            # Set up mocks to return valid data with DOT notation
            mock_expand.side_effect = lambda x: x  # Identity function
            mock_summarize.side_effect = lambda x: x[["P.up", "P.down"]]  # Use dots!

            result = FisherCombination.sum_of_logs(sample_significance_data)

            # Verify the pipeline was called
            assert mock_expand.call_count == len(sample_significance_data)
            assert mock_summarize.call_count == len(sample_significance_data)

            # Check structure of result
            assert isinstance(result, pd.DataFrame)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for MetaAnalysisCore and FisherCombination."""

    @pytest.mark.integration
    def test_full_pipeline_effect_sizes(self, differential_expression_data):
        """Test the full pipeline for effect size calculation."""
        # Calculate effect sizes
        effect_sizes = MetaAnalysisCore.calculate_effect_sizes(
            differential_expression_data
        )

        # Should have results for all genes
        assert len(effect_sizes) == len(differential_expression_data["keys"])

        # First few genes should have larger effect sizes (we added differential expression)
        first_three_effects = effect_sizes.iloc[:3]["g"].abs().values
        last_three_effects = effect_sizes.iloc[-3:]["g"].abs().values

        # Average absolute effect size should be larger for first three genes
        assert np.mean(first_three_effects) > np.mean(last_three_effects)

    @pytest.mark.integration
    def test_full_pipeline_fisher_combination(self, differential_expression_data):
        """Test the full pipeline for Fisher's combination."""

        """
        The test fixture (differential_expression_data) creates 6 genes:
        • Genes 0-2: Have differential expression (should have p < 0.05) 
        • Genes 3-5: NO differential expression (will have p ≈ 0.3-0.5) 
        """
        # Calculate p-values
        pvalues = FisherCombination.calculate_study_pvalues(
            differential_expression_data
        )

        # Should have results for all genes
        assert len(pvalues) == len(differential_expression_data["keys"])

        # Test Fisher combination with multiple "studies" (duplicate the same study)
        sig_matrices = [pvalues, pvalues]  # Simulate two identical studies

        fisher_result = FisherCombination.sum_of_logs(sig_matrices)
        # fisher_result = FisherCombination.sum_of_logs_with_R_bugfix(sig_matrices)

        # Should have results
        assert len(fisher_result) > 0

        """
        When combining two identical p-values using Fisher's method:
        • For p < 0.37: Combined p-value becomes MORE significant (smaller) ✓ 
        • For p > 0.37: Combined p-value becomes LESS significant (larger) ✗ 
        • For p ≈ 0.29-0.30: Near the crossover point, minimal or negative improvement 
        """
        # Check that combined p-values are more significant than individual ones
        for gene in fisher_result.index:
            gene_key = f'symbol_{gene.split("_")[1]}' if "gene_" in gene else gene

            if gene_key in pvalues["keys"].values:
                mask = pvalues["keys"] == gene_key
                if mask.any():
                    original_pval = pvalues[mask]["P.up"].iloc[0]  # Use dot notation!
                    combined_pval = fisher_result.loc[gene]["F.pval.up"]

                    # Only check genes where original p-value suggests some signal
                    # Fisher's method combining identical p-values has a crossover around p=0.37
                    # Use conservative threshold to avoid false failures
                    if (
                        not np.isnan(combined_pval)
                        and not np.isnan(original_pval)
                        and original_pval < 0.25
                    ):
                        assert combined_pval <= original_pval

                    """
                    - Chi-squared statistic: -2 * sum(log(p))
                    - Degrees of freedom: 2 * n_studies`

                    For two identical studies with p-value p:
                    - Chi-squared stat = -2 * 2 * log(p) = -4 * log(p)`
                    - df = `2 * 2 = 4`
                    - Combined p-value = P(χ² > -4*log(p), df=4)

                    Fisher's statistic: X² = -2 × Σ ln(p_i)
                    For two identical p-values: X² = -4 × ln(p)
                    Combined p-value: P(χ²(4) > X²)
                    Example calculation:
                        • Original p = 0.294 
                        • X² = -4 × ln(0.294) ≈ 4.95 
                        • Combined p = P(χ²(4) > 4.95) ≈ 0.298 (WORSE!) 
                    """

    @pytest.mark.integration
    def test_effect_sizes_to_meta_analysis_pipeline(self):
        """Test pipeline from effect sizes to meta-analysis results."""
        # Create multiple "studies" with effect sizes
        np.random.seed(123)

        effect_sizes_df1 = pd.DataFrame(
            {"study1": [1.0, 0.5, 0.2], "study2": [0.8, 0.6, 0.1]},
            index=["gene1", "gene2", "gene3"],
        )

        se_df1 = pd.DataFrame(
            {"study1": [0.3, 0.2, 0.4], "study2": [0.25, 0.3, 0.35]},
            index=["gene1", "gene2", "gene3"],
        )

        # Pool using meta-analysis
        pooled_results = MetaAnalysisCore.pool_inverse_variance(
            effect_sizes_df1, se_df1, method="random"
        )

        assert len(pooled_results) == 3
        assert all(pooled_results["n_studies"] == 2)

        # Check that pooled estimates are reasonable
        for gene in pooled_results.index:
            assert not np.isnan(pooled_results.loc[gene]["summary"])
            assert pooled_results.loc[gene]["se_summary"] > 0

    @pytest.mark.integration
    def test_cross_class_data_flow(self, sample_study):
        """Test data flow between MetaAnalysisCore and FisherCombination."""
        # Start with study data
        assert sample_study is not None

        # Use MetaAnalysisCore to validate
        MetaAnalysisCore.check_metagem_study(sample_study)

        # Calculate effect sizes
        effect_sizes = MetaAnalysisCore.calculate_effect_sizes(sample_study)

        # Calculate p-values using FisherCombination
        pvalues = FisherCombination.calculate_study_pvalues(sample_study)

        # Both should have same number of genes
        assert len(effect_sizes) == len(pvalues)

        # Keys should match
        assert effect_sizes["keys"].tolist() == pvalues["keys"].tolist()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_realistic_meta_analysis_scenario(self):
        """Test a realistic meta-analysis scenario with multiple studies."""
        np.random.seed(456)

        # Create multiple studies with different sample sizes and effect sizes
        studies = []
        for i in range(3):
            n_samples = 8 + i * 2  # Different sample sizes: 8, 10, 12

            expr_data = pd.DataFrame(
                np.random.randn(5, n_samples),  # 5 genes, varying samples
                index=[f"gene_{j}" for j in range(5)],
                columns=[f"study{i}_sample_{k}" for k in range(n_samples)],
            )

            # Add differential expression to some genes
            if i == 0:
                expr_data.iloc[:2, n_samples // 2 :] += 1.5  # Strong effect in study 0
            elif i == 1:
                expr_data.iloc[
                    :2, n_samples // 2 :
                ] += 0.8  # Moderate effect in study 1
            # Study 2 has no differential expression (null study)

            study = {
                "expr": expr_data,
                "class": np.array(
                    [0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2)
                ),
                "keys": [f"symbol_{j}" for j in range(5)],
            }

            studies.append(study)

        # Calculate effect sizes for each study
        all_effect_sizes = []
        for study in studies:
            es = MetaAnalysisCore.calculate_effect_sizes(study)
            all_effect_sizes.append(es)

        # Calculate p-values for each study
        all_pvalues = []
        for study in studies:
            pvals = FisherCombination.calculate_study_pvalues(study)
            all_pvalues.append(pvals)

        # Test Fisher combination
        fisher_result = FisherCombination.sum_of_logs(all_pvalues)

        # Should have results for all genes
        assert len(fisher_result) >= 3  # Should have results for most genes

        # Genes with consistent effects across studies should have smaller p-values
        # (This is a general expectation, though specific values depend on simulation)
        assert any(fisher_result["F.pval.up"] < 0.05) or any(
            fisher_result["F.pval.down"] < 0.05
        )


# =============================================================================
# Pytest Configuration and Marks
# =============================================================================


@pytest.mark.unit
class TestQuickSanityChecks:
    """Quick sanity checks for basic functionality."""

    def test_imports_work(self):
        """Test that imports work correctly."""
        from metaintegrator.core_analysis import MetaAnalysisCore, FisherCombination

        assert MetaAnalysisCore is not None
        assert FisherCombination is not None

    def test_required_keys_constant(self):
        """Test that required keys constant is correct."""
        expected_keys = ["expr", "class", "keys"]
        assert MetaAnalysisCore.REQUIRED_STUDY_KEYS == expected_keys

    @pytest.mark.parametrize(
        "method_name",
        [
            "create_annotation_table",
            "check_metagem_study",
            "clean_na",
            "expand_dataframe",
            "get_effect_size",
            "calculate_effect_sizes",
            "summarize_effects_within_study",
            "pool_inverse_variance",
            "get_ttest_pvalues",
        ],
    )
    def test_metaanalysiscore_methods_exist(self, method_name):
        """Test that all expected MetaAnalysisCore methods exist."""
        assert hasattr(MetaAnalysisCore, method_name)
        assert callable(getattr(MetaAnalysisCore, method_name))

    @pytest.mark.parametrize(
        "method_name",
        [
            "calculate_study_pvalues",
            "summarize_significance_within_study",
            "_apply_fisher_vectorized",
            "sum_of_logs",
            "multimerge",
        ],
    )
    def test_fishercombination_methods_exist(self, method_name):
        """Test that all expected FisherCombination methods exist."""
        assert hasattr(FisherCombination, method_name)
        assert callable(getattr(FisherCombination, method_name))


if __name__ == "__main__":
    pytest.main([__file__])
