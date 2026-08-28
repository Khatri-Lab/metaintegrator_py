"""
Reference Data Manager for ImmunoStates and other deconvolution algorithms.

This module provides production-grade loading, validation, and caching of
reference matrices used for immune cell deconvolution.

Key Features:
- Multiple data sources: bundled files, URLs, local paths
- Automatic validation of matrix structure
- Caching for performance
- Clear error messages
- Unit-testable design

Example:
    >>> manager = ReferenceDataManager()
    >>> basis_matrix = manager.load_immunostates_matrix()
    >>> print(f"Loaded {basis_matrix.shape[0]} genes × {basis_matrix.shape[1]} cell types")
"""

import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


class ReferenceDataManager:
    """
    Manages loading and validation of reference matrices for deconvolution.

    Supports multiple data sources with automatic fallback:
    1. User-provided path
    2. Bundled data file (package installation)
    3. Download from URL (with caching)
    4. Mock data (development/testing only)

    Attributes:
        cache_dir: Directory for cached downloads
        validate_on_load: Whether to validate matrices after loading
    """

    # Default locations for bundled data
    PACKAGE_DATA_DIR = Path(__file__).parent / "data"
    CACHE_DIR = Path.home() / ".metaintegrator" / "reference_data"

    # Known reference matrices and their metadata
    REFERENCE_MATRICES = {
        "immunostates": {
            "filename": "immunoStates_reference_matrix.csv",
            "url": "https://github.com/Franenco/ImmunoStates/raw/master/data/immunoStatesMatrix.rda",  # R data - needs conversion
            "description": "ImmunoStates reference matrix (24 cell types)",
            "expected_genes": 317,  # Approximate - actual matrix has ~317 genes
            "expected_cell_types": 24,
            "cell_type_names": [
                "CD56bright_natural_killer_cell",
                "CD56dim_natural_killer_cell",
                "CD14_positive_monocyte",
                "CD16_positive_monocyte",
                "naive_B_cell",
                "memory_B_cell",
                "plasma_cell",
                "CD8_positive_alpha_beta_T_cell",
                "CD4_positive_alpha_beta_T_cell",
                "gamma_delta_T_cell",
                "neutrophil",
                "eosinophil",
                "basophil",
                # Additional cell types in full immunoStates
                "myeloid_dendritic_cell",
                "plasmacytoid_dendritic_cell",
                "CD4_Tcm",
                "CD4_Tem",
                "CD8_Tcm",
                "CD8_Tem",
                "Treg",
                "Th1",
                "Th2",
                "Th17",
                "Tfh",
            ],
        }
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        validate_on_load: bool = True,
        allow_mock_data: bool = True,
    ) -> None:
        """Initialize the reference data manager.

        Args:
            cache_dir: Directory for caching downloaded files (default: ~/.metaintegrator/reference_data)
            validate_on_load: Validate matrices after loading (recommended: True)
            allow_mock_data: Allow fallback to mock data if real data unavailable (default: True for dev)
        """
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.validate_on_load = validate_on_load
        self.allow_mock_data = allow_mock_data

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache for loaded matrices (in-memory)
        self._loaded_matrices: Dict[str, pd.DataFrame] = {}

    def load_immunostates_matrix(
        self, custom_path: Optional[str] = None, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load the ImmunoStates reference matrix.

        Loading priority:
        1. Custom path (if provided)
        2. Previously loaded (if use_cache=True)
        3. Bundled package data
        4. Downloaded from URL (cached)
        5. Mock data (if allow_mock_data=True)

        Args:
            custom_path: Path to custom matrix file (optional)
            use_cache: Use in-memory cache if available

        Returns:
            DataFrame with genes as rows, cell types as columns

        Raises:
            FileNotFoundError: If matrix not found and mock data not allowed
            ValueError: If matrix validation fails

        Example:
            >>> manager = ReferenceDataManager()
            >>> matrix = manager.load_immunostates_matrix()
            >>> print(matrix.shape)
            (317, 24)  # Approximate dimensions
        """
        cache_key = "immunostates"

        # Check in-memory cache first
        if use_cache and cache_key in self._loaded_matrices:
            return self._loaded_matrices[cache_key].copy()

        # Try loading from various sources
        matrix = None
        source = None

        # 1. Try custom path
        if custom_path:
            try:
                matrix = self._load_matrix_from_file(custom_path)
                source = f"custom path: {custom_path}"
            except Exception as e:
                warnings.warn(f"Failed to load from custom path {custom_path}: {e}")

        # 2. Try bundled data
        if matrix is None:
            bundled_path = (
                self.PACKAGE_DATA_DIR / self.REFERENCE_MATRICES["immunostates"]["filename"]
            )
            if bundled_path.exists():
                try:
                    matrix = self._load_matrix_from_file(bundled_path)
                    source = f"bundled data: {bundled_path}"
                except Exception as e:
                    warnings.warn(f"Failed to load bundled data: {e}")

        # 3. Try cached download
        if matrix is None:
            cached_path = self.cache_dir / self.REFERENCE_MATRICES["immunostates"]["filename"]
            if cached_path.exists():
                try:
                    matrix = self._load_matrix_from_file(cached_path)
                    source = f"cached download: {cached_path}"
                except Exception as e:
                    warnings.warn(f"Failed to load from cache: {e}")

        # 4. Try downloading
        if matrix is None:
            try:
                matrix = self._download_immunostates_matrix()
                source = "downloaded from URL"
            except Exception as e:
                warnings.warn(f"Failed to download matrix: {e}")

        # 5. Fall back to mock data
        if matrix is None:
            if self.allow_mock_data:
                warnings.warn(
                    "\n" + "=" * 70 + "\n"
                    "WARNING: Using MOCK immunoStates matrix\n"
                    "=" * 70 + "\n"
                    "Could not load actual immunoStates reference matrix.\n"
                    "Using randomly generated mock data for development/testing.\n\n"
                    "For production use, please:\n"
                    "1. Download the matrix from: https://github.com/Franenco/ImmunoStates\n"
                    "2. Convert from R format (.rda) to CSV\n"
                    "3. Place in: " + str(self.PACKAGE_DATA_DIR) + "\n"
                    "4. Or provide custom path via: load_immunostates_matrix(custom_path='...')\n"
                    "=" * 70
                )
                matrix = self._generate_mock_immunostates_matrix()
                source = "mock data (DEVELOPMENT ONLY)"
            else:
                raise FileNotFoundError(
                    "ImmunoStates reference matrix not found. "
                    "Please provide the matrix file or enable mock data for development."
                )

        # Validate the loaded matrix
        if self.validate_on_load:
            self._validate_immunostates_matrix(matrix)

        # Cache in memory
        self._loaded_matrices[cache_key] = matrix.copy()

        print(f"Loaded ImmunoStates matrix from: {source}")
        print(f"  Dimensions: {matrix.shape[0]} genes × {matrix.shape[1]} cell types")

        return matrix.copy()

    def _load_matrix_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Load matrix from file (supports CSV, TSV, Excel).

        Args:
            file_path: Path to the matrix file

        Returns:
            DataFrame with genes as index, cell types as columns
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect file format and load
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path, index_col=0)
        elif file_path.suffix.lower() in [".tsv", ".txt"]:
            df = pd.read_csv(file_path, sep="\t", index_col=0)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, index_col=0)
        else:
            # Try CSV as default
            df = pd.read_csv(file_path, index_col=0)

        return df

    def _download_immunostates_matrix(self) -> pd.DataFrame:
        """
        Download ImmunoStates matrix from URL and cache it.

        Note: The original is in R .rda format. This method assumes
        a pre-converted CSV is available. For actual use, you'll need
        to convert the R data to CSV first.

        Returns:
            Downloaded and cached matrix
        """
        url = self.REFERENCE_MATRICES["immunostates"]["url"]
        filename = self.REFERENCE_MATRICES["immunostates"]["filename"]
        cache_path = self.cache_dir / filename

        # For R .rda files, we'd need rpy2 or pre-conversion
        # This is a placeholder that would need actual implementation
        raise NotImplementedError(
            "Automatic download of R .rda format not implemented. "
            "Please convert the matrix to CSV manually and place in: "
            f"{self.PACKAGE_DATA_DIR}"
        )

    def _validate_immunostates_matrix(self, matrix: pd.DataFrame) -> None:
        """
        Validate that the loaded matrix has the expected structure.

        Args:
            matrix: Matrix to validate

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(matrix, pd.DataFrame):
            raise ValueError("Matrix must be a pandas DataFrame")

        if matrix.empty:
            raise ValueError("Matrix is empty")

        # Check that index contains gene names (strings)
        if not all(isinstance(idx, str) for idx in matrix.index):
            raise ValueError("Matrix index must contain gene names (strings)")

        # Check for reasonable dimensions
        expected = self.REFERENCE_MATRICES["immunostates"]
        n_genes, n_cell_types = matrix.shape

        if n_genes < 50:
            warnings.warn(
                f"Matrix has only {n_genes} genes. Expected ~{expected['expected_genes']}. "
                "This may not be the full ImmunoStates reference."
            )

        if n_cell_types < 10:
            warnings.warn(
                f"Matrix has only {n_cell_types} cell types. Expected ~{expected['expected_cell_types']}."
            )

        # Check for NaN values
        if matrix.isna().any().any():
            nan_count = matrix.isna().sum().sum()
            warnings.warn(f"Matrix contains {nan_count} NaN values")

        # Check that values are numeric. pd.api.types.is_numeric_dtype (not
        # np.issubdtype) is required here: matrix.dtypes can contain pandas
        # extension dtypes (e.g. the modern default string dtype for
        # object/string columns), and np.issubdtype raises TypeError
        # ("Cannot interpret '<StringDtype...>' as a data type") rather
        # than returning False for those — is_numeric_dtype is pandas' own
        # dtype-checking utility and handles both numpy and extension
        # dtypes correctly.
        if not matrix.dtypes.apply(pd.api.types.is_numeric_dtype).all():
            raise ValueError("All matrix values must be numeric")

    def _generate_mock_immunostates_matrix(self) -> pd.DataFrame:
        """
        Generate mock ImmunoStates matrix for development/testing.

        WARNING: This is NOT real data and should only be used for:
        - Development and testing
        - Code structure validation
        - Unit tests

        DO NOT use for actual biological analysis!

        Returns:
            Mock matrix with realistic structure but random values
        """
        # Use a fixed seed for reproducibility in tests
        rng = np.random.RandomState(42)

        n_genes = 317  # Approximate size of actual immunoStates
        cell_types = self.REFERENCE_MATRICES["immunostates"]["cell_type_names"][:13]  # Use subset

        # Generate gene names (use some real immune-related genes)
        real_immune_genes = [
            "CD3D",
            "CD3E",
            "CD3G",
            "CD4",
            "CD8A",
            "CD8B",
            "CD19",
            "CD79A",
            "CD79B",
            "MS4A1",
            "CD27",
            "CD38",
            "FCGR3A",
            "FCGR3B",
            "NCAM1",
            "KLRD1",
            "KLRB1",
            "CD14",
            "ITGAM",
            "ITGAX",
            "CD68",
            "CD163",
            "MPO",
            "ELANE",
            "AZU1",
            "CTSG",
            "IL2RA",
            "FOXP3",
            "IL17A",
            "IFNG",
            "IL4",
            "IL13",
        ]
        gene_names = real_immune_genes + [
            f"Gene_{i:03d}" for i in range(len(real_immune_genes), n_genes)
        ]

        # Generate matrix with some structure
        # Cell-type specific genes should have higher expression in their cell type
        matrix_data = rng.randn(n_genes, len(cell_types))

        # Add cell-type specificity for first 100 genes
        for i in range(min(100, n_genes)):
            cell_type_idx = i % len(cell_types)
            matrix_data[i, cell_type_idx] += rng.uniform(2, 4)  # Boost specific cell type

        matrix = pd.DataFrame(matrix_data, index=gene_names[:n_genes], columns=cell_types)

        return matrix

    def get_available_matrices(self) -> Dict[str, Dict]:
        """
        Get information about available reference matrices.

        Returns:
            Dictionary of matrix names and their metadata
        """
        return self.REFERENCE_MATRICES.copy()

    def clear_cache(self) -> None:
        """Clear in-memory cache of loaded matrices."""
        self._loaded_matrices.clear()

    def get_matrix_info(self, matrix_name: str = "immunostates") -> Dict:
        """
        Get metadata about a specific matrix.

        Args:
            matrix_name: Name of the matrix

        Returns:
            Dictionary with matrix metadata
        """
        if matrix_name not in self.REFERENCE_MATRICES:
            raise ValueError(f"Unknown matrix: {matrix_name}")

        return self.REFERENCE_MATRICES[matrix_name].copy()


# Convenience function for easy access
def load_immunostates_reference(
    custom_path: Optional[str] = None, validate: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load ImmunoStates reference matrix.

    Args:
        custom_path: Optional path to custom matrix file
        validate: Whether to validate the matrix after loading

    Returns:
        ImmunoStates reference matrix

    Example:
        >>> matrix = load_immunostates_reference()
        >>> print(matrix.shape)
        (317, 24)
    """
    manager = ReferenceDataManager(validate_on_load=validate)
    return manager.load_immunostates_matrix(custom_path=custom_path)


__all__ = [
    "ReferenceDataManager",
    "load_immunostates_reference",
]
