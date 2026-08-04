"""
Comprehensive unit tests for missing MetaIntegrator functions.
This module provides thorough testing of the newly implemented functions.
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

from metaintegrator.enhanced_core_analysis import (
    EnhancedMetaAnalysis,
    EnhancedFisherCombination,
)
from metaintegrator.core_analysis import MetaAnalysisCore, FisherCombination

from tests.metaintegrator.test_data_generators import (
    TestDataGenerator,
)


class TestCalculateTtestQvalues:
    """Test suite for calculate_ttest_qvalues function."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests

        # Standard test study
        self.standard_study = {
            "expr": pd.DataFrame(
                np.random.randn(50, 20),
                index=[f"Gene_{i}" for i in range(50)],
                columns=[f"Sample_{i}" for i in range(20)],
            ),
            "class": np.array([0] * 10 + [1] * 10),
            "keys": [f"GENE_{i}" for i in range(50)],
        }

        # Study with missing values
        expr_with_na = self.standard_study["expr"].copy()
        expr_with_na.iloc[0:5, 0:3] = np.nan
        self.study_with_na = {
            "expr": expr_with_na,
            "class": self.standard_study["class"],
            "keys": self.standard_study["keys"],
        }

        # Small study (edge case)
        self.small_study = {
            "expr": pd.DataFrame(
                np.random.randn(3, 4),
                index=["Gene_A", "Gene_B", "Gene_C"],
                columns=["S1", "S2", "S3", "S4"],
            ),
            "class": np.array([0, 0, 1, 1]),
            "keys": ["GENE_A", "GENE_B", "GENE_C"],
        }

    def test_standard_qvalue_calculation(self):
        """Test Q-value calculation with standard data."""
        result = EnhancedMetaAnalysis.calculate_ttest_qvalues(self.standard_study)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["Q_up", "Q_down", "keys"]
        assert len(result) == 50
        assert result.index.equals(self.standard_study["expr"].index)

        # Check Q-values are valid probabilities
        assert (result["Q_up"].dropna() >= 0).all()
        assert (result["Q_up"].dropna() <= 1).all()
        assert (result["Q_down"].dropna() >= 0).all()
        assert (result["Q_down"].dropna() <= 1).all()

        # Check keys are preserved
        pd.testing.assert_series_equal(
            result["keys"],
            pd.Series(self.standard_study["keys"], index=result.index),
            check_names=False,
        )

    def test_qvalue_vs_pvalue_relationship(self):
        """Test that Q-values are appropriately adjusted from P-values."""
        result = EnhancedMetaAnalysis.calculate_ttest_qvalues(self.standard_study)

        # Get original p-values for comparison
        pvals = MetaAnalysisCore.get_ttest_pvalues(
            self.standard_study["expr"], self.standard_study["class"]
        )

        # Q-values should generally be >= corresponding P-values due to FDR correction
        # (allowing for numerical precision and the few cases where they might be equal)
        valid_up = ~(pvals["P.up"].isna() | result["Q_up"].isna())
        valid_down = ~(pvals["P.down"].isna() | result["Q_down"].isna())

        if valid_up.any():
            assert (result.loc[valid_up, "Q_up"] >= pvals.loc[valid_up, "P.up"] - 1e-10).all()

        if valid_down.any():
            assert (
                result.loc[valid_down, "Q_down"] >= pvals.loc[valid_down, "P.down"] - 1e-10
            ).all()

    def test_missing_value_handling(self):
        """Test proper handling of missing values in expression data."""
        result = EnhancedMetaAnalysis.calculate_ttest_qvalues(self.study_with_na)

        # Check that function completes without error
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.study_with_na["keys"])

        # Check that some Q-values might be NaN where p-values are NaN
        # but function should still run
        assert not result["Q_up"].isna().all()  # Not all should be NaN
        assert not result["Q_down"].isna().all()

    def test_small_sample_handling(self):
        """Test behavior with small sample sizes."""
        result = EnhancedMetaAnalysis.calculate_ttest_qvalues(self.small_study)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["Q_up", "Q_down", "keys"]

    def test_identical_groups_edge_case(self):
        """Test behavior when groups have identical values."""
        # Create study where all cases = 1, all controls = 0
        identical_study = {
            "expr": pd.DataFrame(
                np.vstack([np.ones(10), np.zeros(10)]).T,  # All 1s  # All 0s
                columns=["Gene1", "Gene2"],
            ).T,
            "class": np.array([0] * 5 + [1] * 5),
            "keys": ["GENE1", "GENE2"],
        }

        result = EnhancedMetaAnalysis.calculate_ttest_qvalues(identical_study)

        # Should handle extreme cases without crashing
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_invalid_study_structure(self):
        """Test error handling for invalid study structures."""
        # Missing required keys
        with pytest.raises(ValueError):
            invalid_study = {"expr": pd.DataFrame(), "class": []}
            EnhancedMetaAnalysis.calculate_ttest_qvalues(invalid_study)

        # Invalid class vector
        with pytest.raises(ValueError):
            invalid_study = {
                "expr": self.standard_study["expr"],
                "class": [0, 1, 2],  # Invalid class values
                "keys": self.standard_study["keys"],
            }
            EnhancedMetaAnalysis.calculate_ttest_qvalues(invalid_study)

    def test_numerical_precision(self):
        """Test numerical precision of Q-value calculations."""
        # Create study with known statistical properties
        np.random.seed(123)
        n_genes, n_samples = 100, 20

        # Generate data with some true effects
        true_effects = np.random.normal(0, 0.5, n_genes)
        expr_data = np.random.randn(n_genes, n_samples)

        # Add effects to case samples
        case_indices = slice(10, 20)
        expr_data[:, case_indices] += true_effects.reshape(-1, 1)

        precision_study = {
            "expr": pd.DataFrame(expr_data),
            "class": np.array([0] * 10 + [1] * 10),
            "keys": [f"GENE_{i}" for i in range(n_genes)],
        }

        result = EnhancedMetaAnalysis.calculate_ttest_qvalues(precision_study)

        # Check that Q-values are reasonable given the input
        assert result["Q_up"].dropna().min() >= 0
        assert result["Q_up"].dropna().max() <= 1
        assert result["Q_down"].dropna().min() >= 0
        assert result["Q_down"].dropna().max() <= 1

    @patch("metaintegrator.core_analysis.MetaAnalysisCore.get_ttest_pvalues")
    def test_error_propagation(self, mock_pvals):
        """Test that errors from underlying functions are properly handled."""
        mock_pvals.side_effect = ValueError("Mock error")

        with pytest.raises(ValueError, match="Failed to calculate t-test p-values"):
            EnhancedMetaAnalysis.calculate_ttest_qvalues(self.standard_study)


class TestCombineSignificances:
    """Test suite for combine_significances function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample significance matrices
        self.study1_sigs = pd.DataFrame(
            {
                "P.up": [0.01, 0.05, 0.001, 0.1],
                "P.down": [0.9, 0.8, 0.99, 0.7],
                "keys": ["GENE1", "GENE2", "GENE3", "GENE4"],
            }
        )

        self.study2_sigs = pd.DataFrame(
            {
                "P.up": [0.02, 0.03, 0.002, 0.08],  # GENE6 now has valid P.up
                "P.down": [0.95, 0.85, 0.98, np.nan],  # GENE6 has NaN for P.down
                "keys": ["GENE1", "GENE2", "GENE5", "GENE6"],
            }
        )

        self.study3_sigs = pd.DataFrame(
            {
                "Q_up": [0.015, 0.04, 0.001],
                "Q_down": [0.92, 0.82, 0.97],
                "keys": ["GENE1", "GENE3", "GENE7"],
            }
        )

        # Study with multi-gene keys
        self.study_multi_keys = pd.DataFrame(
            {
                "P.up": [0.01, 0.05],
                "P.down": [0.9, 0.8],
                "keys": ["GENE1,GENE2", "GENE3"],
            }
        )

        # Study with invalid entries
        self.study_with_invalid = pd.DataFrame(
            {
                "P.up": [0.01, np.nan, 0.001, 0.1],
                "P.down": [0.9, np.nan, 0.99, 0.7],
                "keys": ["GENE1", "NA", "", "GENE4"],
            }
        )

    def test_basic_combination(self):
        """Test basic functionality with valid input."""
        result = EnhancedMetaAnalysis.combine_significances([self.study1_sigs, self.study2_sigs])

        # Check output structure
        assert isinstance(result, dict)
        assert "sigs_up" in result
        assert "sigs_down" in result
        assert "study_info" in result

        # Check that both up and down DataFrames exist
        assert isinstance(result["sigs_up"], pd.DataFrame)
        assert isinstance(result["sigs_down"], pd.DataFrame)
        assert isinstance(result["study_info"], pd.DataFrame)

        # Check that genes from both studies are included
        all_genes = set(result["sigs_up"].index) | set(result["sigs_down"].index)
        expected_genes = {"GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6"}
        assert expected_genes.issubset(all_genes)

    def test_column_naming_convention(self):
        """Test that output columns follow expected naming convention."""
        result = EnhancedMetaAnalysis.combine_significances([self.study1_sigs, self.study2_sigs])

        # Check column naming for up regulation
        up_cols = result["sigs_up"].columns.tolist()
        assert any("study_0" in col and "up" in col for col in up_cols)
        assert any("study_1" in col and "up" in col for col in up_cols)

        # Check column naming for down regulation
        down_cols = result["sigs_down"].columns.tolist()
        assert any("study_0" in col and "down" in col for col in down_cols)
        assert any("study_1" in col and "down" in col for col in down_cols)

    def test_multi_key_expansion(self):
        """Test proper handling of multi-gene keys."""
        result = EnhancedMetaAnalysis.combine_significances([self.study_multi_keys])

        # Check that multi-key entry was expanded
        genes = set(result["sigs_up"].index) | set(result["sigs_down"].index)
        assert "GENE1" in genes
        assert "GENE2" in genes  # Should be expanded from "GENE1,GENE2"
        assert "GENE3" in genes

    def test_invalid_entry_filtering(self):
        """Test that invalid entries are properly filtered out."""
        result = EnhancedMetaAnalysis.combine_significances([self.study_with_invalid])

        # Check study metadata
        study_info = result["study_info"]
        assert len(study_info) == 1
        assert study_info.iloc[0]["original_rows"] == 4
        assert study_info.iloc[0]["valid_rows"] < 4  # Some should be filtered

        # Check that valid genes remain
        genes = set(result["sigs_up"].index) | set(result["sigs_down"].index)
        assert "GENE1" in genes
        assert "GENE4" in genes

    def test_empty_input_handling(self):
        """Test error handling for empty input."""
        with pytest.raises(ValueError, match="Input list cannot be empty"):
            EnhancedMetaAnalysis.combine_significances([])

    def test_invalid_input_type(self):
        """Test error handling for invalid input types."""
        with pytest.raises(TypeError, match="Input must be a list"):
            EnhancedMetaAnalysis.combine_significances("not a list")

        with pytest.raises(TypeError, match="Element 0 is not a pandas DataFrame"):
            EnhancedMetaAnalysis.combine_significances(["not a dataframe"])

    def test_missing_keys_column(self):
        """Test error handling when 'keys' column is missing."""
        invalid_df = pd.DataFrame(
            {
                "P.up": [0.01, 0.05],
                "P.down": [0.9, 0.8],
                # Missing 'keys' column
            }
        )

        with pytest.raises(ValueError, match="DataFrame 0 missing required 'keys' column"):
            EnhancedMetaAnalysis.combine_significances([invalid_df])

    def test_different_significance_column_names(self):
        """Test handling of different significance column naming schemes."""
        # Test with Q-values instead of P-values
        result = EnhancedMetaAnalysis.combine_significances([self.study3_sigs])

        assert isinstance(result, dict)
        assert "sigs_up" in result
        assert "sigs_down" in result

    def test_study_metadata_accuracy(self):
        """Test accuracy of study processing metadata."""
        result = EnhancedMetaAnalysis.combine_significances(
            [self.study1_sigs, self.study_with_invalid]
        )

        metadata = result["study_info"]

        # Check study 0 (clean data)
        study0_meta = metadata[metadata["study_index"] == 0].iloc[0]
        assert study0_meta["original_rows"] == 4
        assert study0_meta["valid_rows"] == 4

        # Check study 1 (data with invalid entries)
        study1_meta = metadata[metadata["study_index"] == 1].iloc[0]
        assert study1_meta["original_rows"] == 4
        assert study1_meta["valid_rows"] == 2  # Only GENE1 and GENE4 are valid

    def test_no_valid_rows_error(self):
        """Test error handling when a study has no valid rows."""
        all_invalid = pd.DataFrame(
            {"P.up": [np.nan, np.nan], "P.down": [np.nan, np.nan], "keys": ["NA", ""]}
        )

        with pytest.raises(ValueError, match="Study 0: No valid rows after filtering"):
            EnhancedMetaAnalysis.combine_significances([all_invalid])

    def test_validation_disable(self):
        """Test that validation can be disabled."""
        # This should not raise an error even with invalid input when validation is off
        result = EnhancedMetaAnalysis.combine_significances(
            [self.study1_sigs], validate_input=False
        )
        assert isinstance(result, dict)

    def test_gene_overlap_handling(self):
        """Test proper handling of genes present in multiple studies."""
        result = EnhancedMetaAnalysis.combine_significances(
            [self.study1_sigs, self.study2_sigs, self.study3_sigs]
        )

        # GENE1 is in all three studies, should have values from all
        sigs_up = result["sigs_up"]
        gene1_row = sigs_up.loc["GENE1"]

        # Count non-NaN values for GENE1 (should be 3)
        non_nan_count = gene1_row.notna().sum()
        assert non_nan_count == 3

    def test_numerical_precision_preservation(self):
        """Test that numerical precision is preserved through the combination process."""
        # Create study with high precision values
        precise_study = pd.DataFrame(
            {
                "P.up": [1e-10, 1e-8, 1e-6],
                "P.down": [0.999999999, 0.99999999, 0.9999999],
                "keys": ["GENE1", "GENE2", "GENE3"],
            }
        )

        result = EnhancedMetaAnalysis.combine_significances([precise_study])

        # Check that small p-values are preserved
        sigs_up = result["sigs_up"]
        assert any(sigs_up.min() < 1e-9)


class TestEnhancedTtestPvalues:
    """Test suite for enhanced_ttest_pvalues function."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.standard_study = {
            "expr": pd.DataFrame(np.random.randn(50, 20)),
            "class": np.array([0] * 10 + [1] * 10),
            "keys": [f"GENE_{i}" for i in range(50)],
        }

    def test_pvalues_only(self):
        """Test function with Q-values disabled."""
        result = EnhancedMetaAnalysis.enhanced_ttest_pvalues(
            self.standard_study, include_qvalues=False
        )

        expected_cols = ["P.both", "P.down", "P.up", "keys"]
        assert list(result.columns) == expected_cols

    def test_pvalues_with_qvalues(self):
        """Test function with Q-values enabled."""
        result = EnhancedMetaAnalysis.enhanced_ttest_pvalues(
            self.standard_study, include_qvalues=True
        )

        expected_cols = ["P.both", "P.down", "P.up", "keys", "Q_up", "Q_down"]
        assert all(col in result.columns for col in expected_cols)

    def test_qvalue_consistency(self):
        """Test that Q-values are consistent with separate calculation."""
        result_integrated = EnhancedMetaAnalysis.enhanced_ttest_pvalues(
            self.standard_study, include_qvalues=True
        )

        result_separate = EnhancedMetaAnalysis.calculate_ttest_qvalues(self.standard_study)

        # Q-values should be identical
        pd.testing.assert_series_equal(
            result_integrated["Q_up"], result_separate["Q_up"], check_names=False
        )
        pd.testing.assert_series_equal(
            result_integrated["Q_down"], result_separate["Q_down"], check_names=False
        )


class TestEnhancedFisherCombination:
    """Test suite for enhanced Fisher combination methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sig_matrices = [
            pd.DataFrame(
                {
                    "P.up": [0.01, 0.05, 0.001],
                    "P.down": [0.9, 0.8, 0.99],
                    "keys": ["GENE1", "GENE2", "GENE3"],
                }
            ),
            pd.DataFrame(
                {
                    "P.up": [0.02, 0.03, 0.002],
                    "P.down": [0.95, 0.85, 0.98],
                    "keys": ["GENE1", "GENE2", "GENE4"],
                }
            ),
        ]

    def test_enhanced_fisher_combination(self):
        """Test enhanced Fisher combination functionality."""
        result = EnhancedFisherCombination.sum_of_logs_with_qvalues(
            self.sig_matrices, use_qvalues=False
        )

        # Check output structure
        assert isinstance(result, pd.DataFrame)

        # Check that Fisher statistics are computed
        fisher_cols = [col for col in result.columns if "F.stat" in col or "F.pval" in col]
        assert len(fisher_cols) >= 4  # At least up and down stats and pvals

        # Check that study counts are included
        assert "n_studies_up" in result.columns
        assert "n_studies_down" in result.columns


class TestIntegrationAndRegression:
    """Integration tests and regression tests for all functions."""

    def setup_method(self):
        """Set up comprehensive test data."""
        np.random.seed(42)

        # Multi-study setup for integration testing
        self.multi_study_data = []
        for i in range(3):
            study = {
                "expr": pd.DataFrame(
                    np.random.randn(30, 16),
                    index=[f"Gene_{j}" for j in range(30)],
                    columns=[f"S{i}_{j}" for j in range(16)],
                ),
                "class": np.array([0] * 8 + [1] * 8),
                "keys": [f"GENE_{j}" for j in range(30)],
            }
            self.multi_study_data.append(study)

    def test_complete_workflow_integration(self):
        """Test integration of all enhanced functions in a complete workflow."""
        # Step 1: Calculate Q-values for each study
        qvalue_results = []
        for study in self.multi_study_data:
            qvals = EnhancedMetaAnalysis.calculate_ttest_qvalues(study)
            qvalue_results.append(qvals)

        # Step 2: Combine significance matrices
        combined = EnhancedMetaAnalysis.combine_significances(qvalue_results)

        # Step 3: Apply Fisher combination
        fisher_result = FisherCombination.sum_of_logs([qvalue_results[0]])

        # Verify complete pipeline works
        assert len(qvalue_results) == 3
        assert isinstance(combined, dict)
        assert isinstance(fisher_result, pd.DataFrame)

    def test_memory_efficiency(self):
        """Test memory efficiency with larger datasets."""
        # Create larger test data
        large_study = {
            "expr": pd.DataFrame(np.random.randn(1000, 50)),
            "class": np.array([0] * 25 + [1] * 25),
            "keys": [f"GENE_{i}" for i in range(1000)],
        }

        # Test that large data can be processed
        result = EnhancedMetaAnalysis.calculate_ttest_qvalues(large_study)
        assert len(result) == 1000
        assert not result.empty

    def test_consistency_across_runs(self):
        """Test that results are consistent across multiple runs."""
        # Fix random seed for reproducibility
        np.random.seed(123)

        study = {
            "expr": pd.DataFrame(np.random.randn(20, 10)),
            "class": np.array([0] * 5 + [1] * 5),
            "keys": [f"GENE_{i}" for i in range(20)],
        }

        # Run multiple times
        result1 = EnhancedMetaAnalysis.calculate_ttest_qvalues(study)
        result2 = EnhancedMetaAnalysis.calculate_ttest_qvalues(study)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


# Performance benchmarking tests
class TestPerformanceBenchmarks:
    """Performance benchmarks for the implemented functions."""

    def test_qvalue_calculation_performance(self, benchmark=None):
        """Benchmark Q-value calculation performance."""
        # Create moderately sized dataset
        study = {
            "expr": pd.DataFrame(np.random.randn(500, 30)),
            "class": np.array([0] * 15 + [1] * 15),
            "keys": [f"GENE_{i}" for i in range(500)],
        }

        def run_qvalue_calc():
            return EnhancedMetaAnalysis.calculate_ttest_qvalues(study)

        # If pytest-benchmark is available, use it
        if benchmark is not None:
            result = benchmark(run_qvalue_calc)
        else:
            # Basic timing test
            import time

            start = time.time()
            result = run_qvalue_calc()
            end = time.time()

            # Should complete in reasonable time (< 5 seconds for this size)
            assert (end - start) < 5.0

        assert len(result) == 500

    def test_combine_significances_scalability(self):
        """Test scalability of combine_significances with multiple studies."""
        # Create multiple studies
        studies = []
        for i in range(5):
            study_sigs = pd.DataFrame(
                {
                    "P.up": np.random.uniform(0, 1, 100),
                    "P.down": np.random.uniform(0, 1, 100),
                    "keys": [f"GENE_{j}" for j in range(100)],
                }
            )
            studies.append(study_sigs)

        import time

        start = time.time()
        result = EnhancedMetaAnalysis.combine_significances(studies)
        end = time.time()

        # Should scale reasonably with number of studies
        assert (end - start) < 10.0  # Generous upper bound
        assert isinstance(result, dict)


class TestRegressionValidation:
    """Regression tests to ensure consistency with reference implementations."""

    def test_effect_size_calculation_precision(self, standard_study):
        """Test effect size calculations against known statistical properties."""
        from metaintegrator.core_analysis import MetaAnalysisCore

        # Calculate effect sizes
        effect_sizes = MetaAnalysisCore.calculate_effect_sizes(standard_study)

        # Basic sanity checks
        assert not effect_sizes["g"].isna().all()
        assert not effect_sizes["se_g"].isna().all()

        # Check that standard errors are positive
        valid_se = effect_sizes["se_g"].dropna()
        assert (valid_se > 0).all()

        # Check that effect sizes are within reasonable bounds
        valid_g = effect_sizes["g"].dropna()
        assert (np.abs(valid_g) < 10).all()  # Should not be extreme

    def test_qvalue_fdr_properties(self, standard_study):
        """Test that Q-values satisfy FDR properties."""
        from metaintegrator.enhanced_core_analysis import EnhancedMetaAnalysis

        result = EnhancedMetaAnalysis.calculate_ttest_qvalues(standard_study)

        # Q-values should be between 0 and 1
        valid_q_up = result["Q_up"].dropna()
        valid_q_down = result["Q_down"].dropna()

        assert (valid_q_up >= 0).all() and (valid_q_up <= 1).all()
        assert (valid_q_down >= 0).all() and (valid_q_down <= 1).all()

        # Q-values should generally be >= P-values
        from metaintegrator.core_analysis import MetaAnalysisCore

        pvals = MetaAnalysisCore.get_ttest_pvalues(standard_study["expr"], standard_study["class"])

        # Allow small numerical differences
        tolerance = 1e-10
        valid_mask = ~(pvals["P.up"].isna() | result["Q_up"].isna())
        if valid_mask.any():
            p_up_valid = pvals.loc[valid_mask, "P.up"]
            q_up_valid = result.loc[valid_mask, "Q_up"]
            assert (q_up_valid >= p_up_valid - tolerance).all()

    def test_fisher_combination_mathematical_properties(self, significance_matrices):
        """Test mathematical properties of Fisher's combination method."""
        from metaintegrator.enhanced_core_analysis import EnhancedMetaAnalysis
        from metaintegrator.core_analysis import FisherCombination

        # Combine significances
        combined = EnhancedMetaAnalysis.combine_significances(significance_matrices)

        # Apply Fisher's method
        fisher_result = FisherCombination._apply_fisher_vectorized(combined["sigs_up"], "up")

        # Fisher statistics should be non-negative
        valid_stats = fisher_result["F.stat.up"].dropna()
        assert (valid_stats >= 0).all()

        # P-values should be between 0 and 1
        valid_pvals = fisher_result["F.pval.up"].dropna()
        assert (valid_pvals >= 0).all() and (valid_pvals <= 1).all()


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness of implementations."""

    def test_single_gene_studies(self):
        """Test behavior with single-gene studies."""
        from metaintegrator.enhanced_core_analysis import EnhancedMetaAnalysis

        single_gene_study = TestDataGenerator.create_synthetic_study(
            n_genes=1, n_samples=10, seed=789
        )

        # Should handle single gene without error
        result = EnhancedMetaAnalysis.calculate_ttest_qvalues(single_gene_study)
        assert len(result) == 1
        assert not result.empty

    def test_extreme_effect_sizes(self):
        """Test behavior with extreme effect sizes."""
        from metaintegrator.enhanced_core_analysis import EnhancedMetaAnalysis

        extreme_study = TestDataGenerator.create_synthetic_study(
            n_genes=20, n_samples=10, effect_size=10.0, seed=999
        )

        # Should handle extreme values without numerical instability
        result = EnhancedMetaAnalysis.calculate_ttest_qvalues(extreme_study)

        # All Q-values should still be valid
        assert (result["Q_up"].dropna() >= 0).all()
        assert (result["Q_up"].dropna() <= 1).all()

    def test_all_missing_values_study(self):
        """Test behavior when a study has all missing values for some genes."""
        from metaintegrator.enhanced_core_analysis import EnhancedMetaAnalysis

        study = TestDataGenerator.create_synthetic_study(n_genes=10, n_samples=8, seed=111)

        # Set first 3 genes to all missing
        study["expr"].iloc[0:3, :] = np.nan

        result = EnhancedMetaAnalysis.calculate_ttest_qvalues(study)

        # Should have NaN for genes with all missing values
        assert result.loc[study["expr"].index[0], "Q_up"] is np.nan or pd.isna(
            result.loc[study["expr"].index[0], "Q_up"]
        )

    def test_identical_samples_robustness(self):
        """Test robustness when samples have identical values."""
        from metaintegrator.enhanced_core_analysis import EnhancedMetaAnalysis

        # Create study with some genes having identical values
        study = {
            "expr": pd.DataFrame(
                np.array(
                    [
                        [1, 1, 1, 1, 2, 2, 2, 2],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ]
                ).T,
                columns=["Gene_identical", "Gene_zero", "Gene_varying"],
            ).T,
            "class": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            "keys": ["GENE1", "GENE2", "GENE3"],
        }

        # Should handle without crashing
        result = EnhancedMetaAnalysis.calculate_ttest_qvalues(study)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_very_small_pvalues(self):
        """Test handling of very small p-values."""
        from metaintegrator.enhanced_core_analysis import EnhancedMetaAnalysis

        # Create significance matrix with very small p-values
        small_pval_matrix = pd.DataFrame(
            {
                "P.up": [1e-50, 1e-100, 1e-300, 0.05],
                "P.down": [0.9, 0.8, 0.7, 0.6],
                "keys": ["GENE1", "GENE2", "GENE3", "GENE4"],
            }
        )

        # Should handle extreme values
        result = EnhancedMetaAnalysis.combine_significances([small_pval_matrix])
        assert isinstance(result, dict)

        # Values should remain finite
        sigs_up = result["sigs_up"]
        assert np.isfinite(sigs_up.values).sum() > 0

    def test_precision_preservation(self, standard_study):
        """Test that numerical precision is preserved through calculations."""
        from metaintegrator.enhanced_core_analysis import EnhancedMetaAnalysis

        # Run calculation multiple times
        result1 = EnhancedMetaAnalysis.calculate_ttest_qvalues(standard_study)
        result2 = EnhancedMetaAnalysis.calculate_ttest_qvalues(standard_study)

        # Results should be identical to machine precision
        pd.testing.assert_frame_equal(result1, result2)


class TestPerformanceAndScalability:
    """Performance and scalability tests."""

    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        from metaintegrator.enhanced_core_analysis import EnhancedMetaAnalysis

        large_study = TestDataGenerator.create_synthetic_study(n_genes=1000, n_samples=50, seed=888)

        import time

        start_time = time.time()
        result = EnhancedMetaAnalysis.calculate_ttest_qvalues(large_study)
        end_time = time.time()

        # Should complete within reasonable time
        assert (end_time - start_time) < 30.0  # 30 seconds max
        assert len(result) == 1000

    def test_memory_usage_scalability(self):
        """Test memory usage with increasing dataset sizes."""
        from metaintegrator.enhanced_core_analysis import EnhancedMetaAnalysis

        sizes = [100, 500, 1000]
        for n_genes in sizes:
            study = TestDataGenerator.create_synthetic_study(
                n_genes=n_genes, n_samples=20, seed=777
            )

            # Should not crash with memory issues
            result = EnhancedMetaAnalysis.calculate_ttest_qvalues(study)
            assert len(result) == n_genes


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
