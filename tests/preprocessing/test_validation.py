"""
Production-Grade Unit Tests for validation.py

Comprehensive test suite for data validation utilities.

Author: metaIntegrator-py team
Version: 1.0.0
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from preprocessing.utils.validation import (
    validate_expression_matrix,
    validate_phenotype_data,
    validate_geo_dataset,
    validate_platform_annotation,
    check_data_compatibility,
    quick_validate_download,
    print_validation_report,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def valid_expression_matrix():
    """Create valid expression matrix."""
    return pd.DataFrame(
        np.random.randn(1000, 20),
        index=[f"Gene_{i}" for i in range(1000)],
        columns=[f"Sample_{i}" for i in range(20)],
    )


@pytest.fixture
def valid_phenotype_data():
    """Create valid phenotype data."""
    return pd.DataFrame(
        {
            "geo_accession": [f"GSM{i}" for i in range(20)],
            "Sample_title": [f"Sample {i}" for i in range(20)],
            "Sample_source": ["Blood"] * 20,
        }
    )


@pytest.fixture
def valid_geo_dataset(valid_expression_matrix, valid_phenotype_data):
    """Create valid GEO dataset."""
    return {
        "expr": valid_expression_matrix,
        "pheno": valid_phenotype_data,
        "platform_info": {
            "platform": "GPL570",
            "platform_name": "Affymetrix Human Genome U133 Plus 2.0 Array",
        },
        "fData": pd.DataFrame(
            {
                "ID": valid_expression_matrix.index,
                "Gene Symbol": [
                    f"GENE{i % 100}" for i in range(len(valid_expression_matrix))
                ],
            }
        ),
    }


@pytest.fixture
def valid_annotation():
    """Create valid platform annotation."""
    return pd.DataFrame(
        {
            "ID": [f"PROBE_{i}" for i in range(1000)],
            "Gene Symbol": [
                f"GENE{i % 100}" if i % 10 != 0 else None for i in range(1000)
            ],
        }
    )


# ============================================================================
# Expression Matrix Validation Tests
# ============================================================================


class TestValidateExpressionMatrix:
    """Test validate_expression_matrix function."""

    def test_valid_matrix(self, valid_expression_matrix):
        """Test validation of valid matrix."""
        is_valid, errors = validate_expression_matrix(valid_expression_matrix)

        assert is_valid == True
        assert len(errors) == 0

    def test_matrix_too_small_genes(self):
        """Test matrix with too few genes."""
        small_matrix = pd.DataFrame(np.random.randn(50, 10))

        is_valid, errors = validate_expression_matrix(small_matrix, min_genes=100)

        assert is_valid == False
        assert any("Too few genes" in err for err in errors)

    def test_matrix_too_small_samples(self):
        """Test matrix with too few samples."""
        small_matrix = pd.DataFrame(np.random.randn(100, 1))

        is_valid, errors = validate_expression_matrix(small_matrix, min_samples=2)

        assert is_valid == False
        assert any("Too few samples" in err for err in errors)

    def test_matrix_with_inf(self):
        """Test matrix with infinite values."""
        matrix_with_inf = pd.DataFrame(np.random.randn(100, 5))
        matrix_with_inf.iloc[10, 1] = np.inf
        matrix_with_inf.iloc[20, 2] = -np.inf

        is_valid, errors = validate_expression_matrix(matrix_with_inf)

        assert is_valid == False
        assert any("infinite" in err.lower() for err in errors)

    def test_matrix_with_missing_allowed(self):
        """Test matrix with missing values (allowed)."""
        matrix_with_na = pd.DataFrame(np.random.randn(100, 5))
        matrix_with_na.iloc[10:20, 1] = np.nan

        is_valid, errors = validate_expression_matrix(
            matrix_with_na, allow_missing=True, max_missing_fraction=0.5
        )

        assert is_valid == True

    def test_matrix_with_too_many_missing(self):
        """Test matrix with too many missing values."""
        matrix_with_na = pd.DataFrame(np.random.randn(100, 5))
        matrix_with_na.iloc[:, 0] = np.nan  # Entire column missing

        is_valid, errors = validate_expression_matrix(
            matrix_with_na, allow_missing=True, max_missing_fraction=0.1
        )

        assert is_valid == False
        assert any("Too many missing" in err for err in errors)

    def test_matrix_numpy_array(self):
        """Test with numpy array input."""
        numpy_matrix = np.random.randn(100, 5)

        is_valid, errors = validate_expression_matrix(numpy_matrix)

        assert is_valid == True
        assert len(errors) == 0

    def test_matrix_wrong_dimensions(self):
        """Test with wrong dimensions."""
        wrong_dim = np.random.randn(100)  # 1D array

        is_valid, errors = validate_expression_matrix(wrong_dim)

        assert is_valid == False
        assert any("2-dimensional" in err for err in errors)


# ============================================================================
# Phenotype Data Validation Tests
# ============================================================================


class TestValidatePhenotypeData:
    """Test validate_phenotype_data function."""

    def test_valid_phenotype(self, valid_phenotype_data, valid_expression_matrix):
        """Test validation of valid phenotype data."""
        is_valid, errors = validate_phenotype_data(
            valid_phenotype_data, valid_expression_matrix
        )

        assert is_valid == True
        assert len(errors) == 0

    def test_sample_count_mismatch(self, valid_phenotype_data):
        """Test phenotype with mismatched sample count."""
        expr = pd.DataFrame(np.random.randn(100, 10))  # 10 samples

        is_valid, errors = validate_phenotype_data(
            valid_phenotype_data, expr  # 20 samples
        )

        assert is_valid == False
        assert any("mismatch" in err.lower() for err in errors)

    def test_missing_required_columns(self, valid_phenotype_data):
        """Test phenotype missing required columns."""
        is_valid, errors = validate_phenotype_data(
            valid_phenotype_data,
            required_columns=["geo_accession", "tissue", "disease"],
        )

        assert is_valid == False
        assert any("Missing required columns" in err for err in errors)

    def test_duplicate_sample_ids(self):
        """Test phenotype with duplicate sample IDs."""
        pheno_with_dupes = pd.DataFrame(
            {
                "geo_accession": ["GSM1", "GSM2", "GSM1", "GSM3"],
                "title": ["S1", "S2", "S3", "S4"],
            }
        )

        is_valid, errors = validate_phenotype_data(pheno_with_dupes)

        assert is_valid == False
        assert any("duplicate" in err.lower() for err in errors)


# ============================================================================
# GEO Dataset Validation Tests
# ============================================================================


class TestValidateGEODataset:
    """Test validate_geo_dataset function."""

    def test_valid_dataset(self, valid_geo_dataset):
        """Test validation of valid GEO dataset."""
        is_valid, errors = validate_geo_dataset(valid_geo_dataset)

        assert is_valid == True
        assert len(errors) == 0

    def test_missing_required_keys(self):
        """Test dataset missing required keys."""
        incomplete_dataset = {"expr": pd.DataFrame(np.random.randn(100, 5))}

        is_valid, errors = validate_geo_dataset(incomplete_dataset)

        assert is_valid == False
        assert any("Missing required keys" in err for err in errors)

    def test_invalid_expression_propagates(self):
        """Test that expression errors propagate."""
        dataset = {
            "expr": pd.DataFrame(np.random.randn(50, 2)),  # Too small
            "pheno": pd.DataFrame({"sample": ["S1", "S2"]}),
            "platform_info": {"platform": "GPL570"},
        }

        is_valid, errors = validate_geo_dataset(dataset)

        assert is_valid == False
        assert any("Expression:" in err for err in errors)

    def test_fdata_length_mismatch(self, valid_geo_dataset):
        """Test fData length mismatch."""
        dataset = valid_geo_dataset.copy()
        dataset["fData"] = pd.DataFrame(
            {"ID": [f"PROBE_{i}" for i in range(500)]}  # Wrong length
        )

        is_valid, errors = validate_geo_dataset(dataset)

        assert is_valid == False
        assert any("doesn't match" in err for err in errors)


# ============================================================================
# Platform Annotation Validation Tests
# ============================================================================


class TestValidatePlatformAnnotation:
    """Test validate_platform_annotation function."""

    def test_valid_annotation(self, valid_annotation):
        """Test validation of valid annotation."""
        is_valid, errors, stats = validate_platform_annotation(valid_annotation)

        assert is_valid == True
        assert stats["total_probes"] == 1000
        assert stats["coverage_pct"] > 0

    def test_annotation_coverage_stats(self, valid_annotation):
        """Test coverage statistics calculation."""
        is_valid, errors, stats = validate_platform_annotation(valid_annotation)

        # Should have ~90% coverage (10% None)
        assert 85 < stats["coverage_pct"] < 95
        assert stats["mapped_probes"] > 800
        assert stats["unique_genes"] > 0

    def test_annotation_empty(self):
        """Test empty annotation."""
        empty_annot = pd.DataFrame()

        is_valid, errors, stats = validate_platform_annotation(empty_annot)

        assert is_valid == False
        assert any("empty" in err.lower() for err in errors)

    def test_annotation_missing_id_column(self):
        """Test annotation missing ID column."""
        bad_annot = pd.DataFrame(
            {"Symbol": ["GENE1", "GENE2"], "Name": ["Gene 1", "Gene 2"]}
        )

        is_valid, errors, stats = validate_platform_annotation(bad_annot)

        assert is_valid == False
        assert any("No ID column" in err for err in errors)


# ============================================================================
# Data Compatibility Tests
# ============================================================================


class TestCheckDataCompatibility:
    """Test check_data_compatibility function."""

    def test_compatible_datasets(self, valid_geo_dataset):
        """Test compatible datasets."""
        datasets = [valid_geo_dataset, valid_geo_dataset]

        is_compatible, warnings, stats = check_data_compatibility(datasets)

        assert is_compatible == True
        assert stats["n_datasets"] == 2
        assert stats["common_genes"] > 0

    def test_different_platforms_warning(self):
        """Test warning for different platforms."""
        dataset1 = {
            "expr": pd.DataFrame(np.random.randn(100, 5)),
            "pheno": pd.DataFrame(),
            "platform_info": {"platform": "GPL570"},
        }
        dataset2 = {
            "expr": pd.DataFrame(np.random.randn(100, 5)),
            "pheno": pd.DataFrame(),
            "platform_info": {"platform": "GPL96"},
        }

        is_compatible, warnings_list, stats = check_data_compatibility(
            [dataset1, dataset2], check_platforms=True
        )

        assert is_compatible == True  # Compatible but with warnings
        assert len(warnings_list) > 0
        assert any("Multiple platforms" in w for w in warnings_list)

    def test_low_gene_overlap_warning(self):
        """Test warning for low gene overlap."""
        dataset1 = {
            "expr": pd.DataFrame(
                np.random.randn(100, 5), index=[f"Gene_{i}" for i in range(100)]
            ),
            "pheno": pd.DataFrame(),
            "platform_info": {},
        }
        dataset2 = {
            "expr": pd.DataFrame(
                np.random.randn(100, 5), index=[f"OtherGene_{i}" for i in range(100)]
            ),
            "pheno": pd.DataFrame(),
            "platform_info": {},
        }

        is_compatible, warnings_list, stats = check_data_compatibility(
            [dataset1, dataset2]
        )

        assert stats["common_genes"] == 0
        assert any("Low gene overlap" in w for w in warnings_list)


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_quick_validate_download(self, valid_geo_dataset):
        """Test quick_validate_download."""
        result = quick_validate_download(valid_geo_dataset)

        assert result == True

    def test_quick_validate_too_small(self):
        """Test quick validation fails for small dataset."""
        small_dataset = {
            "expr": pd.DataFrame(np.random.randn(50, 2)),
            "pheno": pd.DataFrame({"sample": ["S1", "S2"]}),
            "platform_info": {"platform": "GPL570"},
        }

        result = quick_validate_download(small_dataset, min_genes=100, min_samples=3)

        assert result == False

    def test_print_validation_report(self, valid_geo_dataset, capsys):
        """Test print_validation_report."""
        print_validation_report(valid_geo_dataset, "Test Report")

        captured = capsys.readouterr()

        assert "Test Report" in captured.out
        assert "Expression Matrix" in captured.out
        assert "Phenotype Data" in captured.out
        assert "PASS" in captured.out


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_validate_none_input(self):
        """Test validation with None input."""
        is_valid, errors = validate_expression_matrix(None)

        assert is_valid == False
        assert len(errors) > 0

    def test_validate_empty_dataset(self):
        """Test validation of empty dataset."""
        empty_dataset = {}

        is_valid, errors = validate_geo_dataset(empty_dataset)

        assert is_valid == False

    def test_validate_wrong_type(self):
        """Test validation with wrong type."""
        is_valid, errors = validate_expression_matrix("not a matrix")

        assert is_valid == False
        assert any("Invalid type" in err for err in errors)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
