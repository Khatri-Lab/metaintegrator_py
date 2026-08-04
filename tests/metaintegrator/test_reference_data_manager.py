"""
Unit tests for ReferenceDataManager.

Tests cover:
- Loading from various sources (custom path, bundled, cache, mock)
- Validation of matrix structure
- Caching behavior
- Error handling
- File format support (CSV, TSV, Excel)
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.reference_data_manager import (
    ReferenceDataManager,
    load_immunostates_reference,
)

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_immunostates_matrix():
    """Create a mock immunoStates matrix for testing."""
    rng = np.random.RandomState(42)
    n_genes = 100
    cell_types = [
        "CD56bright_natural_killer_cell",
        "CD56dim_natural_killer_cell",
        "CD14_positive_monocyte",
        "CD4_positive_alpha_beta_T_cell",
        "CD8_positive_alpha_beta_T_cell",
    ]

    gene_names = [f"GENE{i}" for i in range(n_genes)]
    data = rng.randn(n_genes, len(cell_types))

    return pd.DataFrame(data, index=gene_names, columns=cell_types)


@pytest.fixture
def mock_csv_file(temp_dir, mock_immunostates_matrix):
    """Create a mock CSV file."""
    csv_path = temp_dir / "test_matrix.csv"
    mock_immunostates_matrix.to_csv(csv_path)
    return csv_path


@pytest.fixture
def mock_tsv_file(temp_dir, mock_immunostates_matrix):
    """Create a mock TSV file."""
    tsv_path = temp_dir / "test_matrix.tsv"
    mock_immunostates_matrix.to_csv(tsv_path, sep="\t")
    return tsv_path


# ===================================================================
# Test: Initialization
# ===================================================================


def test_manager_init_default():
    """Test default initialization."""
    manager = ReferenceDataManager()

    assert manager.cache_dir is not None
    assert manager.validate_on_load is True
    assert manager.allow_mock_data is True
    assert len(manager._loaded_matrices) == 0


def test_manager_init_custom_cache_dir(temp_dir):
    """Test initialization with custom cache directory."""
    custom_cache = temp_dir / "custom_cache"
    manager = ReferenceDataManager(cache_dir=custom_cache)

    assert manager.cache_dir == custom_cache
    assert custom_cache.exists()  # Should be created


def test_manager_init_no_mock_data():
    """Test initialization with mock data disabled."""
    manager = ReferenceDataManager(allow_mock_data=False)
    assert manager.allow_mock_data is False


# ===================================================================
# Test: Loading from File
# ===================================================================


def test_load_from_csv(mock_csv_file):
    """Test loading matrix from CSV file."""
    manager = ReferenceDataManager()
    matrix = manager._load_matrix_from_file(mock_csv_file)

    assert isinstance(matrix, pd.DataFrame)
    assert len(matrix) == 100
    assert len(matrix.columns) == 5
    assert all(isinstance(idx, str) for idx in matrix.index)


def test_load_from_tsv(mock_tsv_file):
    """Test loading matrix from TSV file."""
    manager = ReferenceDataManager()
    matrix = manager._load_matrix_from_file(mock_tsv_file)

    assert isinstance(matrix, pd.DataFrame)
    assert len(matrix) == 100


def test_load_nonexistent_file():
    """Test loading from non-existent file raises error."""
    manager = ReferenceDataManager()

    with pytest.raises(FileNotFoundError):
        manager._load_matrix_from_file("/nonexistent/path.csv")


# ===================================================================
# Test: Validation
# ===================================================================


def test_validate_valid_matrix(mock_immunostates_matrix):
    """Test validation passes for valid matrix."""
    manager = ReferenceDataManager()

    # Should not raise any errors
    manager._validate_immunostates_matrix(mock_immunostates_matrix)


def test_validate_empty_matrix():
    """Test validation fails for empty matrix."""
    manager = ReferenceDataManager()
    empty_matrix = pd.DataFrame()

    with pytest.raises(ValueError, match="Matrix is empty"):
        manager._validate_immunostates_matrix(empty_matrix)


def test_validate_non_dataframe():
    """Test validation fails for non-DataFrame."""
    manager = ReferenceDataManager()

    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        manager._validate_immunostates_matrix(np.array([[1, 2], [3, 4]]))


def test_validate_non_string_index():
    """Test validation fails for non-string gene names."""
    manager = ReferenceDataManager()
    matrix = pd.DataFrame(
        np.random.randn(10, 5),
        index=range(10),  # Numeric index instead of gene names
        columns=[f"CT{i}" for i in range(5)],
    )

    with pytest.raises(ValueError, match="must contain gene names"):
        manager._validate_immunostates_matrix(matrix)


def test_validate_non_numeric_values():
    """Test validation fails for non-numeric values."""
    manager = ReferenceDataManager()
    matrix = pd.DataFrame(
        [["a", "b"], ["c", "d"]], index=["Gene1", "Gene2"], columns=["CT1", "CT2"]
    )

    with pytest.raises(ValueError, match="must be numeric"):
        manager._validate_immunostates_matrix(matrix)


def test_validate_warns_few_genes():
    """Test validation warns about few genes."""
    manager = ReferenceDataManager()
    small_matrix = pd.DataFrame(
        np.random.randn(10, 5),  # Only 10 genes
        index=[f"Gene{i}" for i in range(10)],
        columns=[f"CT{i}" for i in range(5)],
    )

    with pytest.warns(UserWarning, match="only 10 genes"):
        manager._validate_immunostates_matrix(small_matrix)


def test_validate_warns_nan_values():
    """Test validation warns about NaN values."""
    manager = ReferenceDataManager()
    matrix_with_nan = pd.DataFrame(
        [[1.0, 2.0], [np.nan, 4.0]], index=["Gene1", "Gene2"], columns=["CT1", "CT2"]
    )

    with pytest.warns(UserWarning, match="NaN values"):
        manager._validate_immunostates_matrix(matrix_with_nan)


# ===================================================================
# Test: Mock Data Generation
# ===================================================================


def test_generate_mock_matrix():
    """Test mock matrix generation."""
    manager = ReferenceDataManager()
    mock_matrix = manager._generate_mock_immunostates_matrix()

    assert isinstance(mock_matrix, pd.DataFrame)
    assert mock_matrix.shape[0] == 317  # Expected number of genes
    assert mock_matrix.shape[1] == 13  # Subset of cell types
    assert all(isinstance(idx, str) for idx in mock_matrix.index)
    assert "CD3D" in mock_matrix.index  # Should include real gene names


def test_mock_matrix_reproducible():
    """Test that mock matrix generation is reproducible."""
    manager1 = ReferenceDataManager()
    manager2 = ReferenceDataManager()

    mock1 = manager1._generate_mock_immunostates_matrix()
    mock2 = manager2._generate_mock_immunostates_matrix()

    # Should be identical due to fixed random seed
    pd.testing.assert_frame_equal(mock1, mock2)


# ===================================================================
# Test: Loading ImmunoStates Matrix
# ===================================================================


def test_load_immunostates_custom_path(mock_csv_file):
    """Test loading with custom path."""
    manager = ReferenceDataManager(allow_mock_data=False)
    matrix = manager.load_immunostates_matrix(custom_path=str(mock_csv_file))

    assert isinstance(matrix, pd.DataFrame)
    assert len(matrix) == 100


def test_load_immunostates_bundled(temp_dir, mock_immunostates_matrix):
    """Test loading from bundled data location."""
    # Create mock bundled file
    bundled_path = temp_dir / "data" / "immunoStates_reference_matrix.csv"
    bundled_path.parent.mkdir(parents=True, exist_ok=True)
    mock_immunostates_matrix.to_csv(bundled_path)

    # Temporarily override PACKAGE_DATA_DIR
    original_dir = ReferenceDataManager.PACKAGE_DATA_DIR
    ReferenceDataManager.PACKAGE_DATA_DIR = temp_dir / "data"

    try:
        manager = ReferenceDataManager(allow_mock_data=False)
        matrix = manager.load_immunostates_matrix()

        assert isinstance(matrix, pd.DataFrame)
        assert len(matrix) == 100
    finally:
        ReferenceDataManager.PACKAGE_DATA_DIR = original_dir


def test_load_immunostates_cached(temp_dir, mock_immunostates_matrix):
    """Test loading from cache."""
    cache_file = temp_dir / "immunoStates_reference_matrix.csv"
    mock_immunostates_matrix.to_csv(cache_file)

    manager = ReferenceDataManager(cache_dir=temp_dir, allow_mock_data=False)
    matrix = manager.load_immunostates_matrix()

    assert isinstance(matrix, pd.DataFrame)
    assert len(matrix) == 100


def test_load_immunostates_mock_data_fallback():
    """Test fallback to mock data when file not found."""
    manager = ReferenceDataManager(allow_mock_data=True)

    with pytest.warns(UserWarning, match="MOCK immunoStates matrix"):
        matrix = manager.load_immunostates_matrix()

    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape[0] == 317  # Mock data size


def test_load_immunostates_no_mock_raises_error():
    """Test that error is raised when file not found and mock data disabled."""
    manager = ReferenceDataManager(allow_mock_data=False)

    with pytest.raises(FileNotFoundError):
        manager.load_immunostates_matrix()


# ===================================================================
# Test: Caching
# ===================================================================


def test_memory_caching(mock_csv_file):
    """Test in-memory caching works."""
    manager = ReferenceDataManager()

    # Load first time
    matrix1 = manager.load_immunostates_matrix(custom_path=str(mock_csv_file))

    # Load second time (should use cache)
    matrix2 = manager.load_immunostates_matrix(custom_path=str(mock_csv_file))

    # Should be equal but different objects (copy)
    pd.testing.assert_frame_equal(matrix1, matrix2)
    assert matrix1 is not matrix2  # Different objects (copy)


def test_cache_clearing():
    """Test cache clearing."""
    manager = ReferenceDataManager()

    # Load mock data (will be cached)
    with pytest.warns():
        manager.load_immunostates_matrix()

    assert len(manager._loaded_matrices) > 0

    # Clear cache
    manager.clear_cache()
    assert len(manager._loaded_matrices) == 0


def test_cache_disabled():
    """Test loading without using cache."""
    manager = ReferenceDataManager()

    # Load first time
    with pytest.warns():
        matrix1 = manager.load_immunostates_matrix(use_cache=True)

    # Modify cached version
    manager._loaded_matrices["immunostates"].iloc[0, 0] = 99999

    # Load with cache disabled - should get fresh copy, not the modified cache
    with pytest.warns():
        matrix2 = manager.load_immunostates_matrix(use_cache=False)

    # Should not have the modified cached value (99999)
    # Note: mock data uses fixed seed, so fresh generation gives same values as matrix1
    assert matrix2.iloc[0, 0] != 99999
    assert matrix2.iloc[0, 0] == matrix1.iloc[0, 0]  # Both should be original mock value


# ===================================================================
# Test: Metadata and Information
# ===================================================================


def test_get_available_matrices():
    """Test getting available matrices info."""
    manager = ReferenceDataManager()
    matrices = manager.get_available_matrices()

    assert "immunostates" in matrices
    assert "description" in matrices["immunostates"]
    assert "expected_genes" in matrices["immunostates"]


def test_get_matrix_info():
    """Test getting specific matrix info."""
    manager = ReferenceDataManager()
    info = manager.get_matrix_info("immunostates")

    assert info["filename"] == "immunoStates_reference_matrix.csv"
    assert info["expected_genes"] == 317


def test_get_matrix_info_unknown():
    """Test getting info for unknown matrix raises error."""
    manager = ReferenceDataManager()

    with pytest.raises(ValueError, match="Unknown matrix"):
        manager.get_matrix_info("nonexistent_matrix")


# ===================================================================
# Test: Convenience Function
# ===================================================================


def test_convenience_function_load_immunostates(mock_csv_file):
    """Test convenience function."""
    matrix = load_immunostates_reference(custom_path=str(mock_csv_file), validate=True)

    assert isinstance(matrix, pd.DataFrame)
    assert len(matrix) == 100


def test_convenience_function_validation_disabled(mock_csv_file):
    """Test convenience function with validation disabled."""
    matrix = load_immunostates_reference(custom_path=str(mock_csv_file), validate=False)

    assert isinstance(matrix, pd.DataFrame)


# ===================================================================
# Test: Integration
# ===================================================================


def test_full_workflow(temp_dir, mock_immunostates_matrix):
    """Test complete workflow from save to load."""
    # Save matrix to file
    csv_path = temp_dir / "immunoStates_reference_matrix.csv"
    mock_immunostates_matrix.to_csv(csv_path)

    # Load it back
    manager = ReferenceDataManager(cache_dir=temp_dir, allow_mock_data=False)
    loaded_matrix = manager.load_immunostates_matrix()

    # Should match original (allowing for floating point differences)
    pd.testing.assert_frame_equal(
        mock_immunostates_matrix,
        loaded_matrix,
        check_dtype=False,  # Allow dtype differences from CSV round-trip
    )


def test_error_handling_invalid_csv(temp_dir):
    """Test error handling for invalid CSV file."""
    # Create a binary file with .csv extension that pandas cannot parse
    invalid_csv = temp_dir / "invalid.csv"
    with open(invalid_csv, "wb") as f:
        # Write binary data that will cause decoding errors
        f.write(b"\x00\x01\x02\x03\xff\xfe\xfd\xfc")
        f.write(b"\x80\x81\x82\x83\x84\x85\x86\x87")

    manager = ReferenceDataManager()

    # Should raise error when trying to load
    with pytest.raises(Exception):  # Can be UnicodeDecodeError, ParserError, etc.
        manager._load_matrix_from_file(invalid_csv)


# ===================================================================
# Run tests
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
