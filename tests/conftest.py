"""
Shared pytest configuration and fixtures for all test modules.

This conftest.py provides fixtures that are accessible to all test modules
in the tests/ directory and its subdirectories.

Python port of R MetaIntegrator package
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from tests.metaintegrator.test_data_generators import (
    TestDataGenerator,
    TestPerformanceUtils,
    TestValidationUtils,
)


@pytest.fixture
def standard_study():
    """Standard synthetic study for meta-analysis testing."""
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
    """Study with missing values for robustness testing."""
    return TestDataGenerator.create_synthetic_study(
        n_genes=30, n_samples=16, missing_rate=0.1, seed=456
    )


@pytest.fixture
def multi_study_dataset():
    """Multiple studies for integration testing."""
    return TestDataGenerator.create_multi_study_dataset(
        n_studies=3,
        study_params=[
            {"n_genes": 80, "effect_size": 0.3, "seed": 42},
            {"n_genes": 90, "effect_size": 0.6, "seed": 43},
            {"n_genes": 85, "effect_size": 0.4, "seed": 44},
        ],
    )


@pytest.fixture
def significance_matrices():
    """Pre-computed significance matrices for Fisher testing."""
    return TestDataGenerator.create_significance_matrices(n_studies=3, n_genes=50, seed=42)


@pytest.fixture
def study_with_known_effects():
    """Study with known differential expression patterns."""
    return TestDataGenerator.create_study_with_known_effects(
        n_genes=100,
        n_samples=20,
        de_genes_indices=[0, 1, 2, 3, 4],  # First 5 genes are DE
        effect_sizes=[1.0, 0.8, 1.2, 0.6, 1.5],
        seed=789,
    )


@pytest.fixture
def performance_test_data():
    """Medium-sized dataset for performance testing."""
    return TestPerformanceUtils.create_performance_test_data("medium")


@pytest.fixture
def large_performance_data():
    """Large dataset for performance benchmarking."""
    return TestPerformanceUtils.create_performance_test_data("large")


def assert_valid_effect_sizes(effect_sizes_df):
    """Assert that effect sizes DataFrame has valid structure and values."""
    required_cols = ["g", "se_g", "keys"]
    assert all(col in effect_sizes_df.columns for col in required_cols)

    # Validate effect size properties
    validation_results = TestValidationUtils.validate_effect_size_properties(
        effect_sizes_df["g"], effect_sizes_df["se_g"]
    )

    for property_name, is_valid in validation_results.items():
        assert is_valid, f"Effect size validation failed: {property_name}"


def assert_valid_pvalues(pvalues_df, expected_cols=None):
    """Assert that p-values DataFrame has valid structure and values."""
    if expected_cols is None:
        expected_cols = ["P_up", "P_down", "keys"]

    assert all(col in pvalues_df.columns for col in expected_cols)

    # Validate p-value properties for each p-value column
    for col in expected_cols:
        if col.startswith("P_") or col.startswith("Q_"):
            validation_results = TestValidationUtils.validate_pvalue_properties(pvalues_df[col])

            for property_name, is_valid in validation_results.items():
                assert is_valid, f"P-value validation failed for {col}: {property_name}"


def assert_valid_qvalue_correction(pvalues, qvalues):
    """Assert that Q-values are properly FDR-corrected P-values."""
    validation_results = TestValidationUtils.validate_qvalue_vs_pvalue_relationship(
        pvalues, qvalues
    )

    for property_name, is_valid in validation_results.items():
        assert is_valid, f"Q-value correction validation failed: {property_name}"


def assert_valid_fisher_combination(fisher_results_df):
    """Assert that Fisher combination results are valid."""
    # Check required columns
    required_patterns = ["F.stat", "F.pval"]

    for pattern in required_patterns:
        matching_cols = [col for col in fisher_results_df.columns if pattern in col]
        assert len(matching_cols) > 0, f"Missing columns with pattern: {pattern}"

    # Check statistical validity
    for col in fisher_results_df.columns:
        if "F.stat" in col:
            valid_stats = fisher_results_df[col].dropna()
            assert (valid_stats >= 0).all(), f"Fisher statistics must be non-negative: {col}"

        elif "F.pval" in col or "F.Qval" in col:
            valid_pvals = fisher_results_df[col].dropna()
            assert (valid_pvals >= 0).all() and (
                valid_pvals <= 1
            ).all(), f"P-values must be between 0 and 1: {col}"


def create_meta_object_from_studies(studies_list, study_names=None):
    """Create a meta-analysis object from a list of studies."""
    if study_names is None:
        study_names = [f"Study_{i}" for i in range(len(studies_list))]

    meta_object = {
        "originalData": {},
        "metaAnalysis": None,
        "leaveOneOutAnalysis": None,
    }

    for study, name in zip(studies_list, study_names):
        # Convert study format to expected meta-analysis format
        formatted_study = {
            "expr": study["expr"],
            "class": study["class"],
            "keys": study["keys"],
        }

        # Add optional fields if present
        for optional_field in ["name", "platform", "formattedName"]:
            if optional_field in study:
                formatted_study[optional_field] = study[optional_field]

        meta_object["originalData"][name] = formatted_study

    return meta_object


def generate_test_effect_sizes():
    """Generate test effect sizes for parameterized tests."""
    return [0.1, 0.5, 1.0, 1.5, 2.0]


def generate_test_sample_sizes():
    """Generate test sample sizes for parameterized tests."""
    return [6, 10, 20, 50, 100]


def generate_test_gene_counts():
    """Generate test gene counts for parameterized tests."""
    return [10, 50, 100, 500, 1000]


def generate_missing_data_rates():
    """Generate missing data rates for robustness testing."""
    return [0.0, 0.05, 0.1, 0.2, 0.3]


def generate_significance_thresholds():
    """Generate significance thresholds for testing."""
    return [0.05, 0.01, 0.001, 1e-4, 1e-6]


@pytest.fixture(scope="session")
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="geo_test_cache_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def clean_cache_dir(temp_cache_dir, request):
    """Provide a clean cache directory for each test."""
    import uuid

    # Use UUID to create unique directory (avoids filesystem-unsafe characters)
    test_id = str(uuid.uuid4()).replace("-", "")[:16]
    test_name = getattr(request.node, "name", "unknown_test")
    # Sanitize test name for filesystem safety
    safe_test_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in test_name)[:50]
    test_cache = temp_cache_dir / f"test_{safe_test_name}_{test_id}"
    test_cache.mkdir(exist_ok=True)
    yield test_cache
    # Cleanup after test
    if test_cache.exists():
        shutil.rmtree(test_cache, ignore_errors=True)


@pytest.fixture
def gene_annotator():
    """Create a GeneAnnotator instance for testing.

    The real annotator (preprocessing.gene_annotator.GeneAnnotator) is
    stateless and operates directly on an in-memory GPL annotation
    DataFrame -- there is no persistent gene-annotation database to
    populate. Optional Ensembl validation is available via
    preprocessing.gene_validation.EnsemblGeneValidator, which loads its
    reference set from bundled package data (no network/DB required).
    """
    from preprocessing.gene_annotator import GeneAnnotator

    return GeneAnnotator()


@pytest.fixture
def mock_expression_data():
    """Create mock expression data for testing."""
    n_genes, n_samples = 100, 20

    # Create realistic log-normal expression data
    expr_data = np.random.lognormal(mean=8, sigma=1.5, size=(n_genes, n_samples))

    # Add some known genes
    gene_names = ["GAPDH", "ACTB", "SOD1", "PTEN", "TP53"] + [
        f"GENE_{i}" for i in range(5, n_genes)
    ]
    sample_names = [f"GSM{i}" for i in range(n_samples)]

    expr_df = pd.DataFrame(expr_data, index=gene_names, columns=sample_names)

    return expr_df


@pytest.fixture
def mock_phenotype_data():
    """Create mock phenotype data for testing."""
    n_samples = 20
    sample_names = [f"GSM{i}" for i in range(n_samples)]

    pheno_df = pd.DataFrame(
        {
            "title": [f"Sample_{i}" for i in range(n_samples)],
            "source_name_ch1": ["tissue"] * n_samples,
            "characteristics_ch1": [
                f'disease state: {"disease" if i < 10 else "control"}' for i in range(n_samples)
            ],
            "platform_id": ["GPL570"] * n_samples,
        },
        index=sample_names,
    )

    return pheno_df


@pytest.fixture
def mock_feature_data():
    """Create mock feature data for testing."""
    n_genes = 100
    gene_names = ["GAPDH", "ACTB", "SOD1", "PTEN", "TP53"] + [
        f"GENE_{i}" for i in range(5, n_genes)
    ]
    probe_ids = [f"PROBE_{i}" for i in range(n_genes)]

    fdata_df = pd.DataFrame(
        {
            "ID": probe_ids,
            "Gene Symbol": gene_names,
            "Gene Title": [f"Gene title for {gene}" for gene in gene_names],
            "ENTREZ_GENE_ID": [str(960 + i) for i in range(n_genes)],
        },
        index=probe_ids,
    )

    return fdata_df


@pytest.fixture
def mock_geo_series_matrix():
    """Create mock GEO series matrix data structure."""
    return {
        "GPL570": {
            "expr": pd.DataFrame(
                np.random.lognormal(mean=8, sigma=1.5, size=(100, 20)),
                index=[f"PROBE_{i}" for i in range(100)],
                columns=[f"GSM{i}" for i in range(20)],
            ),
            "pheno": pd.DataFrame(
                {
                    "title": [f"Sample_{i}" for i in range(20)],
                    "source_name_ch1": ["tissue"] * 20,
                    "characteristics_ch1": [
                        f'disease state: {"disease" if i < 10 else "control"}' for i in range(20)
                    ],
                },
                index=[f"GSM{i}" for i in range(20)],
            ),
            "fData": pd.DataFrame(
                {
                    "ID": [f"PROBE_{i}" for i in range(100)],
                    "Gene Symbol": ["GAPDH"] * 5 + [f"GENE_{i}" for i in range(5, 100)],
                },
                index=[f"PROBE_{i}" for i in range(100)],
            ),
            "platform_info": {"platform": "GPL570"},
        }
    }


@pytest.fixture
def mock_gem_object(mock_expression_data, mock_phenotype_data):
    """Create a mock GEM object for testing."""
    return {
        "name": "GSE12345",
        "expr": mock_expression_data,
        "pheno": mock_phenotype_data,
        "class": np.array([0] * 10 + [1] * 10),  # 10 controls, 10 cases
        "keys": mock_expression_data.index.tolist(),
        "platform": "GPL570",
        "load_comment": "Test data",
        "exp_comment": "Data in log scale",
        "key_comment": "Gene symbols",
        "formattedName": "Test_Study",
    }


@pytest.fixture
def geo_loader(clean_cache_dir):
    """Create a real DataLoaderFactory-backed GEO loader for testing.

    Replaces the old fixture that instantiated a nonexistent
    ``GEODataLoader`` class. The real entry point is
    ``preprocessing.loader_factory.DataLoaderFactory``.
    """
    from preprocessing.loader_factory import DataLoaderFactory

    return DataLoaderFactory.create_geo_loader(cache_dir=str(clean_cache_dir))


@pytest.fixture
def ensembl_gene_validator():
    """Create a real EnsemblGeneValidator for testing.

    Loads its reference set from bundled package data -- no network
    connection, database, or mock is required.
    """
    from preprocessing.gene_validation import EnsemblGeneValidator

    return EnsemblGeneValidator()


@pytest.fixture
def mock_requests_response():
    """Mock HTTP response for GEO data downloads."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None

    # Mock series matrix file content
    mock_content = """!Series_title	Test Series
!Series_geo_accession	GSE12345
!series_matrix_table_begin
"ID_REF"	"GSM1"	"GSM2"	"GSM3"	"GSM4"
"PROBE_1"	8.5	8.2	8.7	8.1
"PROBE_2"	7.8	7.9	7.6	7.7
"PROBE_3"	9.1	9.3	9.0	9.2
!series_matrix_table_end
"""

    mock_response.text = mock_content
    mock_response.content = mock_content.encode("utf-8")

    return mock_response


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Suppress warnings during testing
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Set random seed for reproducible tests
    np.random.seed(42)

    yield

    # Cleanup after test
    # Any global cleanup can go here


@pytest.fixture
def large_expression_dataset():
    """Create a large expression dataset for performance testing."""
    n_genes, n_samples = 50000, 500

    # Create large but sparse data for memory efficiency
    expr_data = np.random.lognormal(mean=8, sigma=1.5, size=(n_genes, n_samples))

    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    sample_names = [f"GSM{i}" for i in range(n_samples)]

    expr_df = pd.DataFrame(expr_data, index=gene_names, columns=sample_names)

    return expr_df


@pytest.fixture
def invalid_expression_data():
    """Create invalid expression data for testing error handling."""
    return pd.DataFrame(
        {
            "sample1": ["invalid", "data", np.inf, -np.inf, np.nan],
            "sample2": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=["gene1", "gene2", "gene3", "gene4", "gene5"],
    )


@pytest.fixture
def mixed_identifier_types():
    """Create mixed gene identifier types for testing annotation."""
    return {
        "gene_symbols": ["GAPDH", "SOD1", "PTEN", "TP53", "ACTB"],
        "ensembl_ids": ["ENSG00000111640", "ENSG00000142168", "ENSG00000171862"],
        "entrez_ids": ["960", "6647", "5728", "7157"],
        "refseq_ids": ["NM_002046", "NM_000454", "NM_000314"],
        "genbank_ids": ["M33197", "M27891", "U92436"],
        "invalid_ids": ["", None, "INVALID_123", "???"],
    }


# Helper functions for parameterized tests
def generate_identifier_type_combinations():
    """Generate combinations of identifier types for testing."""
    types = ["gene_symbol", "ensembl_gene", "entrez", "refseq_mrna"]
    combinations = []

    for source in types:
        for target in types:
            if source != target:
                combinations.append((source, target))

    return combinations


def generate_test_gse_ids():
    """Generate test GSE IDs for parameterized tests."""
    return ["GSE12345", "GSE67890", "GSE11111", "GSE22222"]


def generate_test_platforms():
    """Generate test platform IDs for parameterized tests."""
    return ["GPL570", "GPL96", "GPL6244", "GPL10558"]


def assert_valid_meta_object(meta_object):
    """Assert that a meta-object has valid structure."""
    assert isinstance(meta_object, dict)
    assert "originalData" in meta_object
    assert isinstance(meta_object["originalData"], dict)
    assert len(meta_object["originalData"]) > 0

    for dataset_name, dataset in meta_object["originalData"].items():
        assert isinstance(dataset, dict)
        assert "expr" in dataset
        assert "class" in dataset
        assert "keys" in dataset
        assert isinstance(dataset["expr"], pd.DataFrame)
        assert len(dataset["keys"]) == dataset["expr"].shape[0]


def assert_valid_gem_object(gem):
    """Assert that a GEM object has valid structure."""
    required_keys = ["name", "expr", "class", "keys"]
    for key in required_keys:
        assert key in gem, f"Missing required key: {key}"

    assert isinstance(gem["expr"], pd.DataFrame)
    assert len(gem["keys"]) == gem["expr"].shape[0]
    assert len(gem["class"]) in [gem["expr"].shape[1], len(gem["expr"].columns)]
