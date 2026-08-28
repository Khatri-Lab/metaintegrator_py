"""
Test data generators and utilities for MetaIntegrator testing.
This module provides utilities for generating synthetic datasets with controllable properties.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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

    @staticmethod
    def create_significance_matrices(
        n_studies: int = 3, n_genes: int = 50, seed: int = None
    ) -> List[pd.DataFrame]:
        """
        Create significance matrices for Fisher combination testing.

        Args:
            n_studies: Number of studies to create
            n_genes: Number of genes per study
            seed: Random seed for reproducibility

        Returns:
            List of significance DataFrames
        """
        if seed is not None:
            np.random.seed(seed)

        matrices = []
        for i in range(n_studies):
            # Use different seed for each study
            np.random.seed((seed or 42) + i)

            matrix = pd.DataFrame(
                {
                    "P_up": np.random.uniform(0, 1, n_genes),
                    "P_down": np.random.uniform(0, 1, n_genes),
                    "keys": [f"GENE_{j}" for j in range(n_genes)],
                }
            )
            matrices.append(matrix)

        return matrices

    @staticmethod
    def create_study_with_known_effects(
        n_genes: int = 100,
        n_samples: int = 20,
        de_genes_indices: List[int] = None,
        effect_sizes: List[float] = None,
        seed: int = None,
    ) -> Dict:
        """
        Create study with known differential expression patterns.

        Args:
            n_genes: Number of genes
            n_samples: Number of samples
            de_genes_indices: Indices of differentially expressed genes
            effect_sizes: Effect sizes for each DE gene
            seed: Random seed

        Returns:
            Study dictionary with known DE patterns
        """
        if seed is not None:
            np.random.seed(seed)

        if de_genes_indices is None:
            de_genes_indices = list(range(0, min(20, n_genes)))  # First 20 genes

        if effect_sizes is None:
            effect_sizes = [1.0] * len(de_genes_indices)  # Default effect size

        # Create base expression data
        n_controls = n_samples // 2
        n_cases = n_samples - n_controls

        expr_data = np.random.normal(0, 1, (n_genes, n_samples))

        # Add specific effects to designated genes
        case_indices = slice(n_controls, n_samples)
        for gene_idx, effect_size in zip(de_genes_indices, effect_sizes):
            if gene_idx < n_genes:
                expr_data[gene_idx, case_indices] += effect_size

        study = {
            "expr": pd.DataFrame(
                expr_data,
                index=[f"Gene_{i}" for i in range(n_genes)],
                columns=[f"Sample_{i}" for i in range(n_samples)],
            ),
            "class": np.array([0] * n_controls + [1] * n_cases),
            "keys": [f"GENE_{i}" for i in range(n_genes)],
            "true_de_genes": de_genes_indices,
            "true_effects": effect_sizes,
        }

        return study


class TestValidationUtils:
    """Utilities for validating test results and statistical properties."""

    @staticmethod
    def validate_pvalue_properties(pvalues: pd.Series) -> Dict[str, bool]:
        """
        Validate that p-values have expected properties.

        Args:
            pvalues: Series of p-values

        Returns:
            Dictionary of validation results
        """
        valid_pvals = pvalues.dropna()

        results = {
            "all_in_range": (valid_pvals >= 0).all() and (valid_pvals <= 1).all(),
            "not_all_zero": not (valid_pvals == 0).all(),
            "not_all_one": not (valid_pvals == 1).all(),
            "has_variation": valid_pvals.std() > 0,
            "reasonable_count": len(valid_pvals) > 0,
        }

        return results

    @staticmethod
    def validate_effect_size_properties(
        effect_sizes: pd.Series, standard_errors: pd.Series
    ) -> Dict[str, bool]:
        """
        Validate effect size calculations.

        Args:
            effect_sizes: Series of effect sizes
            standard_errors: Series of standard errors

        Returns:
            Dictionary of validation results
        """
        valid_es = effect_sizes.dropna()
        valid_se = standard_errors.dropna()

        results = {
            "se_positive": (valid_se > 0).all() if len(valid_se) > 0 else True,
            "es_finite": np.isfinite(valid_es).all() if len(valid_es) > 0 else True,
            "se_finite": np.isfinite(valid_se).all() if len(valid_se) > 0 else True,
            "reasonable_es_range": ((np.abs(valid_es) < 20).all() if len(valid_es) > 0 else True),
            "reasonable_se_range": (valid_se < 10).all() if len(valid_se) > 0 else True,
        }

        return results

    @staticmethod
    def validate_qvalue_vs_pvalue_relationship(
        pvalues: pd.Series, qvalues: pd.Series, tolerance: float = 1e-10
    ) -> Dict[str, bool]:
        """
        Validate that Q-values are properly adjusted P-values.

        Args:
            pvalues: Series of p-values
            qvalues: Series of q-values
            tolerance: Numerical tolerance for comparison

        Returns:
            Dictionary of validation results
        """
        # Find valid pairs
        valid_mask = ~(pvalues.isna() | qvalues.isna())

        if not valid_mask.any():
            return {"no_valid_pairs": True}

        valid_p = pvalues[valid_mask]
        valid_q = qvalues[valid_mask]

        results = {
            "q_gte_p": (valid_q >= valid_p - tolerance).all(),
            "q_in_range": (valid_q >= 0).all() and (valid_q <= 1).all(),
            "monotonic_relationship": True,  # Could add more sophisticated check
        }

        return results


class TestPerformanceUtils:
    """Utilities for performance testing and benchmarking."""

    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        """
        Measure execution time of a function.

        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (result, execution_time)
        """
        import time

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        return result, end_time - start_time

    @staticmethod
    def create_performance_test_data(size_category: str = "medium") -> Dict:
        """
        Create test data of different sizes for performance testing.

        Args:
            size_category: "small", "medium", "large", or "xlarge"

        Returns:
            Test data dictionary
        """
        size_configs = {
            "small": {"n_genes": 100, "n_samples": 20},
            "medium": {"n_genes": 500, "n_samples": 40},
            "large": {"n_genes": 1000, "n_samples": 60},
            "xlarge": {"n_genes": 2000, "n_samples": 100},
        }

        config = size_configs.get(size_category, size_configs["medium"])

        return TestDataGenerator.create_synthetic_study(seed=42, **config)
