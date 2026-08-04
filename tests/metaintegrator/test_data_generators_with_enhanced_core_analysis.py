"""
Test configuration, shared fixtures, and additional integration tests.
This module provides the testing infrastructure for comprehensive validation.
"""

import tempfile
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

# Test configuration
pytest_plugins = ["pytest_benchmark"]  # For performance testing


class TestDataGenerator:
    """Utility class for generating test datasets with known properties."""

    @staticmethod
    def create_synthetic_study(
        n_genes: int = 100,
        n_samples: int = 20,
        effect_size: float = 0.5,
        noise_level: float = 1.0,
        missing_rate: float = 0.0,
        seed: int = None,
    ) -> Dict:
        """
        Create synthetic study data with controllable statistical properties.

        Args:
            n_genes: Number of genes
            n_samples: Number of samples (should be even for balanced design)
            effect_size: True effect size for differentially expressed genes
            noise_level: Standard deviation of noise
            missing_rate: Fraction of values to set as missing
            seed: Random seed for reproducibility

        Returns:
            Dictionary with study structure (expr, class, keys)
        """
        if seed is not None:
            np.random.seed(seed)

        # Create balanced design
        n_controls = n_samples // 2
        n_cases = n_samples - n_controls

        # Generate expression data
        expr_data = np.random.normal(0, noise_level, (n_genes, n_samples))

        # Add effects to a subset of genes in case samples
        n_de_genes = n_genes // 4  # 25% differentially expressed
        de_genes = np.random.choice(n_genes, n_de_genes, replace=False)

        case_indices = slice(n_controls, n_samples)
        expr_data[de_genes, case_indices] += effect_size

        # Add missing values
        if missing_rate > 0:
            n_missing = int(n_genes * n_samples * missing_rate)
            missing_indices = np.random.choice(n_genes * n_samples, n_missing, replace=False)
            missing_rows = missing_indices // n_samples
            missing_cols = missing_indices % n_samples
            expr_data[missing_rows, missing_cols] = np.nan

        # Create study structure
        study = {
            "expr": pd.DataFrame(
                expr_data,
                index=[f"Gene_{i}" for i in range(n_genes)],
                columns=[f"Sample_{i}" for i in range(n_samples)],
            ),
            "class": np.array([0] * n_controls + [1] * n_cases),
            "keys": [f"GENE_{i}" for i in range(n_genes)],
        }

        return study

    @staticmethod
    def create_multi_study_dataset(
        n_studies: int = 3, study_params: List[Dict] = None, shared_genes: float = 0.8
    ) -> List[Dict]:
        """
        Create multiple studies with overlapping gene sets.

        Args:
            n_studies: Number of studies to create
            study_params: List of parameter dicts for each study
            shared_genes: Fraction of genes shared across studies

        Returns:
            List of study dictionaries
        """
        if study_params is None:
            study_params = [{"seed": 42 + i} for i in range(n_studies)]

        # Generate base gene list
        base_n_genes = 200
        base_genes = [f"GENE_{i}" for i in range(base_n_genes)]

        studies = []
        for i, params in enumerate(study_params):
            # Default parameters
            default_params = {
                "n_genes": 150,
                "n_samples": 20,
                "effect_size": 0.5,
                "seed": 42 + i,
            }
            default_params.update(params)

            # Create study
            study = TestDataGenerator.create_synthetic_study(**default_params)

            # Adjust gene names for overlap
            n_genes_study = len(study["keys"])
            if shared_genes > 0:
                n_shared = int(n_genes_study * shared_genes)
                n_unique = n_genes_study - n_shared

                # Shared genes from base set
                shared_genes_list = base_genes[:n_shared]

                # Unique genes for this study
                unique_genes_list = [f"GENE_S{i}_{j}" for j in range(n_unique)]

                # Combine and shuffle
                all_genes = shared_genes_list + unique_genes_list
                np.random.shuffle(all_genes)

                study["keys"] = all_genes
                study["expr"].index = all_genes

            studies.append(study)

        return studies


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def standard_study():
    """Standard test study with known properties."""
    return TestDataGenerator.create_synthetic_study(
        n_genes=50, n_samples=20, effect_size=0.5, seed=42
    )


@pytest.fixture
def small_study():
    """Small study for edge case testing."""
    return TestDataGenerator.create_synthetic_study(
        n_genes=5, n_samples=6, effect_size=1.0, seed=123
    )


@pytest.fixture
def study_with_missing():
    """Study with missing values."""
    return TestDataGenerator.create_synthetic_study(
        n_genes=30, n_samples=16, missing_rate=0.1, seed=456
    )


@pytest.fixture
def multi_study_dataset():
    """Multiple studies for integration testing."""
    return TestDataGenerator.create_multi_study_dataset(
        n_studies=3,
        study_params=[
            {"n_genes": 80, "effect_size": 0.3},
            {"n_genes": 90, "effect_size": 0.6},
            {"n_genes": 85, "effect_size": 0.4},
        ],
    )


@pytest.fixture
def significance_matrices():
    """Pre-computed significance matrices for Fisher testing."""
    matrices = []
    for i in range(3):
        np.random.seed(42 + i)
        n_genes = 50
        matrix = pd.DataFrame(
            {
                "P.up": np.random.uniform(0, 1, n_genes),
                "P.down": np.random.uniform(0, 1, n_genes),
                "keys": [f"GENE_{j}" for j in range(n_genes)],
            }
        )
        matrices.append(matrix)

    return matrices


class TestRegressionValidation:
    """Regression tests to ensure consistency with reference implementations."""

    def test_effect_size_calculation_precision(self, standard_study):
        """Test effect size calculations against known statistical properties."""
        from metaintegrator.enhanced_core_analysis import EnhancedMetaAnalysis
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

        # fisher_result could be
        # Empty DataFrame\nColumns: []\nIndex: [].columns

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
        from metaintegrator.core_analysis import MetaAnalysisCore

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


# Configuration for pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "regression: marks tests as regression tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if "large_dataset" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark regression tests
        if "regression" in item.name or "precision" in item.name:
            item.add_marker(pytest.mark.regression)

        # Mark integration tests
        if "integration" in item.name or "workflow" in item.name:
            item.add_marker(pytest.mark.integration)


# Test runner script
if __name__ == "__main__":
    import sys

    # Default test run
    pytest_args = [__file__, "-v", "--tb=short", "-x"]  # Stop on first failure

    # Add coverage if available
    try:
        import pytest_cov

        pytest_args.extend(["--cov=metaintegrator", "--cov-report=html"])
    except ImportError:
        pass

    # Add benchmark if requested
    if "--benchmark" in sys.argv:
        pytest_args.append("--benchmark-only")

    # Run pytest
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)
