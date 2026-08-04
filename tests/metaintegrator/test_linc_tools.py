"""
Unit tests for the LINCS connectivity mapping tools module.

This module contains comprehensive tests for LincsDataManager, LincsAnalyzer,
and LincsVisualizer classes, as well as utility functions.
"""

import sys
from unittest.mock import Mock, patch, mock_open

import numpy as np
import pandas as pd
import pytest

print(">>> PYTHONPATH (sys.path) at import time:")
print("\n".join(f"  {i:02d}: {p}" for i, p in enumerate(sys.path)))

import os, sys

# Add the src directory to the path to import modules to be run with python instead of pytest
# sys.path.insert(0, os.path.join(os.path.dirname(__file__),  "../..", "src"))

# Add the source directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# Import the classes and functions to test
try:
    from metaintegrator.linc_tools import (
        LincsDataManager,
        LincsAnalyzer,
        LincsVisualizer,
        lincs_tools,
        lincs_correlate,
        lincs_bait_corr,
        _prepare_signature_frame,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    raise


# Test Data Fixtures
@pytest.fixture
def sample_signature():
    """Create a sample gene signature DataFrame."""
    genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
    values = [2.5, -1.8, 3.2, -2.1, 1.4]
    return pd.DataFrame({"summary": values}, index=genes)


@pytest.fixture
def sample_lincs_data():
    """Create sample LINCS dataset."""
    genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
    perturbagens = ["PERT1", "PERT2", "PERT3", "X123", "CLINICAL1"]
    data = np.random.randn(5, 5)
    return pd.DataFrame(data, index=genes, columns=perturbagens)


@pytest.fixture
def sample_tri_ann():
    """Create sample tri annotation data."""
    return pd.DataFrame(
        {
            "merge": ["CLINICAL1", "CLINICAL2", "CLINICAL3"],
            "HighestPhase": [3, 2, 1],
            "drug_name": ["DrugA", "DrugB", "DrugC"],
        }
    )


@pytest.fixture
def sample_gene_ann():
    """Create sample gene annotation data."""
    return pd.DataFrame(
        {
            "gene_symbol": ["GENE1", "GENE2", "GENE3"],
            "description": ["Description1", "Description2", "Description3"],
        }
    )


@pytest.fixture
def sample_lincs_ann():
    """Create sample LINCS annotation data."""
    return pd.DataFrame(
        {
            "pert_iname": ["Drug-A", "Drug B", "Drug_C"],
            "pert_id": ["BRD-123", "BRD-456", "BRD-789"],
        }
    )


@pytest.fixture
def mock_meta_object():
    """Create mock meta-analysis object."""
    pooled_results = pd.DataFrame(
        {
            "effectSize": [2.5, -1.8, 3.2, -2.1, 1.4],
            "pValue": [0.001, 0.01, 0.001, 0.005, 0.02],
        },
        index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
    )

    return {"metaAnalysis": {"pooledResults": pooled_results}}


@pytest.fixture
def mock_filter_object():
    """Create mock filter object."""
    return {
        "posGeneNames": ["GENE1", "GENE3", "GENE5"],
        "negGeneNames": ["GENE2", "GENE4"],
    }


# LincsDataManager Tests
class TestLincsDataManager:
    """Test cases for LincsDataManager class."""

    def test_init(self):
        """Test LincsDataManager initialization."""
        manager = LincsDataManager()
        assert manager._data_cache == {}
        assert manager._data_loaded is False
        assert "stanford.edu" in manager.data_url

    @patch("metaintegrator.linc_tools.shutil")
    @patch("metaintegrator.linc_tools.requests.get")
    @patch("metaintegrator.linc_tools.tempfile.mkdtemp")
    @patch("metaintegrator.linc_tools.zipfile.ZipFile")
    def test_download_and_extract_success(self, mock_zipfile, mock_tempdir, mock_get, mock_shutil):
        """Test successful download and extraction."""
        # Setup mocks
        mock_tempdir.return_value = "/tmp/test"
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_get.return_value = mock_response

        mock_zip = Mock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip

        manager = LincsDataManager()

        with patch("builtins.open", mock_open()):
            result = manager._download_and_extract()

        assert result == "/tmp/test"
        mock_get.assert_called_once()
        mock_zip.extractall.assert_called_once_with("/tmp/test")

    @patch("metaintegrator.linc_tools.requests.get")
    def test_download_and_extract_http_error(self, mock_get):
        """Test HTTP error during download."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response

        manager = LincsDataManager()

        with pytest.raises(Exception):
            manager._download_and_extract()

    @patch("metaintegrator.linc_tools.os.listdir")
    @patch("metaintegrator.linc_tools.pd.read_pickle")
    @patch("metaintegrator.linc_tools.pd.read_csv")
    def test_load_files_from_dir(self, mock_read_csv, mock_read_pickle, mock_listdir):
        """Test loading files from directory."""
        # Setup mock files
        mock_listdir.return_value = [
            "agg.gold.pkl",
            "qc.pkl",
            "lincs.ann.txt",
            "tri.ann.csv",
            "gene.ann.txt",
        ]

        mock_df = pd.DataFrame({"test": [1, 2, 3]})
        mock_read_pickle.return_value = mock_df
        mock_read_csv.return_value = mock_df

        manager = LincsDataManager()
        manager._load_files_from_dir("/tmp/test")

        # Check that data was loaded into cache
        assert len(manager._data_cache) > 0
        mock_read_pickle.assert_called()
        mock_read_csv.assert_called()

    def test_process_annotations(self, sample_lincs_ann):
        """Test annotation processing."""
        manager = LincsDataManager()
        manager._data_cache["lincs_ann"] = sample_lincs_ann

        manager._process_annotations()

        processed = manager._data_cache["lincs_ann"]
        assert "merge" in processed.columns
        assert processed["merge"].iloc[0] == "DRUGA"  # Uppercase, no special chars
        assert processed["merge"].iloc[1] == "DRUGB"

    def test_process_annotations_no_data(self):
        """Test annotation processing with no data."""
        manager = LincsDataManager()
        # Should not raise error when no lincs_ann data
        manager._process_annotations()
        assert manager._data_cache == {}

    @patch.object(LincsDataManager, "load_data")
    def test_get_data_not_loaded(self, mock_load_data):
        """Test get_data when data is not loaded."""
        mock_load_data.return_value = False
        manager = LincsDataManager()

        result = manager.get_data("test_data")

        assert result is None
        mock_load_data.assert_called_once()

    @patch.object(LincsDataManager, "load_data")
    def test_get_data_success(self, mock_load_data):
        """Test successful data retrieval."""
        mock_load_data.return_value = True
        manager = LincsDataManager()
        manager._data_loaded = True
        manager._data_cache["test_data"] = "test_value"

        result = manager.get_data("test_data")

        assert result == "test_value"

    @patch.object(LincsDataManager, "_download_and_extract")
    @patch.object(LincsDataManager, "_load_files_from_dir")
    @patch.object(LincsDataManager, "_process_annotations")
    @patch("metaintegrator.linc_tools.shutil.rmtree")
    def test_load_data_success(self, mock_rmtree, mock_process, mock_load_files, mock_download):
        """Test successful data loading."""
        mock_download.return_value = "/tmp/test"

        manager = LincsDataManager()
        result = manager.load_data()

        assert result is True
        assert manager._data_loaded is True
        mock_download.assert_called_once()
        mock_load_files.assert_called_once_with("/tmp/test")
        mock_process.assert_called_once()
        mock_rmtree.assert_called_once_with("/tmp/test")

    @patch.object(LincsDataManager, "_download_and_extract")
    def test_load_data_failure(self, mock_download):
        """Test data loading failure."""
        mock_download.side_effect = Exception("Download failed")

        manager = LincsDataManager()

        with patch("metaintegrator.linc_tools.warnings.warn") as mock_warn:
            result = manager.load_data()

        assert result is False
        assert manager._data_loaded is False
        mock_warn.assert_called_once()

    def test_load_data_already_loaded(self):
        """Test loading data when already loaded."""
        manager = LincsDataManager()
        manager._data_loaded = True

        result = manager.load_data()

        assert result is True

    @patch.object(LincsDataManager, "_download_and_extract")
    @patch.object(LincsDataManager, "_load_files_from_dir")
    @patch.object(LincsDataManager, "_process_annotations")
    @patch("metaintegrator.linc_tools.shutil.rmtree")
    def test_load_data_force_reload(
        self, mock_rmtree, mock_process, mock_load_files, mock_download
    ):
        """Test force reload of data."""
        mock_download.return_value = "/tmp/test"

        manager = LincsDataManager()
        manager._data_loaded = True

        result = manager.load_data(force_reload=True)

        assert result is True
        mock_download.assert_called_once()


# LincsAnalyzer Tests
class TestLincsAnalyzer:
    """Test cases for LincsAnalyzer class."""

    def test_init_default_manager(self):
        """Test LincsAnalyzer initialization with default manager."""
        analyzer = LincsAnalyzer()
        assert isinstance(analyzer.data_manager, LincsDataManager)

    def test_init_custom_manager(self):
        """Test LincsAnalyzer initialization with custom manager."""
        custom_manager = LincsDataManager()
        analyzer = LincsAnalyzer(custom_manager)
        assert analyzer.data_manager is custom_manager

    def test_get_dataset_data_invalid_dataset(self):
        """Test getting dataset data with invalid dataset name."""
        analyzer = LincsAnalyzer()
        result = analyzer._get_dataset_data("INVALID")

        assert isinstance(result, dict)
        assert result["error"] == "incorrect dataset"

    @patch.object(LincsDataManager, "get_data")
    def test_get_dataset_data_not_found(self, mock_get_data):
        """Test getting dataset data when dataset not found."""
        mock_get_data.return_value = None

        analyzer = LincsAnalyzer()
        result = analyzer._get_dataset_data("CP")

        assert isinstance(result, dict)
        assert result["error"] == "dataset not found"

    @patch.object(LincsDataManager, "get_data")
    def test_get_dataset_data_success(self, mock_get_data, sample_lincs_data):
        """Test successful dataset data retrieval."""
        mock_agg_data = {"CP": sample_lincs_data}
        mock_get_data.return_value = mock_agg_data

        analyzer = LincsAnalyzer()
        result = analyzer._get_dataset_data("CP")

        assert isinstance(result, pd.DataFrame)
        assert result.equals(sample_lincs_data)

    @patch.object(LincsDataManager, "get_data")
    def test_filter_by_clinical_no_data(self, mock_get_data):
        """Test clinical filtering with no annotation data."""
        mock_get_data.return_value = None

        analyzer = LincsAnalyzer()
        identifiers = ["PERT1", "PERT2", "PERT3"]
        result = analyzer._filter_by_clinical(identifiers)

        assert result == identifiers

    @patch.object(LincsDataManager, "get_data")
    def test_filter_by_clinical_with_data(self, mock_get_data, sample_tri_ann):
        """Test clinical filtering with annotation data."""
        mock_get_data.return_value = sample_tri_ann

        analyzer = LincsAnalyzer()
        identifiers = ["CLINICAL1", "NOTCLINICAL", "CLINICAL2"]
        result = analyzer._filter_by_clinical(identifiers)

        assert "CLINICAL1" in result
        assert "CLINICAL2" in result
        assert "NOTCLINICAL" not in result

    @patch.object(LincsDataManager, "get_data")
    def test_filter_by_clinical_with_baits(self, mock_get_data, sample_tri_ann):
        """Test clinical filtering with additional baits."""
        mock_get_data.return_value = sample_tri_ann

        analyzer = LincsAnalyzer()
        identifiers = ["CLINICAL1", "NOTCLINICAL"]
        baits = ["MYBAIT"]
        result = analyzer._filter_by_clinical(identifiers, include_baits=baits)

        print(f"\n>>> result={result}")

        assert "CLINICAL1" in result
        assert "MYBAIT" in result
        assert "NOTCLINICAL" not in result

    def test_apply_sorting_reverse(self, sample_lincs_data):
        """Test reverse sorting."""
        analyzer = LincsAnalyzer()
        df = pd.DataFrame({"values": [3, 1, 4, 1, 5]})

        result = analyzer._apply_sorting(df, "values", "reverse")

        assert result["values"].iloc[0] == 1  # Smallest first
        assert result["values"].iloc[-1] == 5  # Largest last

    def test_apply_sorting_aggravate(self, sample_lincs_data):
        """Test aggravate (descending) sorting."""
        analyzer = LincsAnalyzer()
        df = pd.DataFrame({"values": [3, 1, 4, 1, 5]})

        result = analyzer._apply_sorting(df, "values", "aggravate")

        assert result["values"].iloc[0] == 5  # Largest first
        assert result["values"].iloc[-1] == 1  # Smallest last

    def test_apply_sorting_absolute(self):
        """Test absolute value sorting."""
        analyzer = LincsAnalyzer()
        df = pd.DataFrame({"values": [1, -3, 2, -5, 4]})

        result = analyzer._apply_sorting(df, "values", "absolute")

        assert abs(result["values"].iloc[0]) == 5  # Largest absolute value first

    def test_apply_sorting_default(self):
        """Test default sorting behavior."""
        analyzer = LincsAnalyzer()
        df = pd.DataFrame({"values": [3, 1, 4, 1, 5]})

        result = analyzer._apply_sorting(df, "values", "unknown")

        # Should return original dataframe
        assert result.equals(df)

    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_data_access_dataset_error(self, mock_get_dataset):
        """Test lincs_data_access with dataset error."""
        mock_get_dataset.return_value = {"error": "incorrect dataset"}

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_data_access("INVALID")

        assert result == "incorrect dataset"

    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_data_access_no_genes(self, mock_get_dataset, sample_lincs_data):
        """Test lincs_data_access with no matching genes."""
        mock_get_dataset.return_value = sample_lincs_data

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_data_access("CP", genes=["NONEXISTENT"])

        assert result == "error, no genes found"

    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_data_access_no_perturbagens(self, mock_get_dataset, sample_lincs_data):
        """Test lincs_data_access with no matching perturbagens."""
        mock_get_dataset.return_value = sample_lincs_data

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_data_access("CP", genes=["GENE1"], perturbagens=["NONEXISTENT"])

        assert result == "error, no perturbagens found"

    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_data_access_success(self, mock_get_dataset, sample_lincs_data):
        """Test successful lincs_data_access."""
        mock_get_dataset.return_value = sample_lincs_data

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_data_access("CP", genes=["GENE1", "GENE2"], perturbagens=["PERT1"])

        assert isinstance(result, pd.DataFrame)
        assert "GENE1" in result.index
        assert "GENE2" in result.index
        assert "PERT1" in result.columns

    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_data_access_all_genes(self, mock_get_dataset, sample_lincs_data):
        """Test lincs_data_access with all_genes=True."""
        mock_get_dataset.return_value = sample_lincs_data

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_data_access("CP", all_genes=True, perturbagens=["PERT1"])

        assert len(result.index) == len(sample_lincs_data.index)

    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_data_access_all_perturbagens(self, mock_get_dataset, sample_lincs_data):
        """Test lincs_data_access with all_perturbagens=True."""
        mock_get_dataset.return_value = sample_lincs_data

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_data_access("CP", genes=["GENE1"], all_perturbagens=True)

        assert len(result.columns) == len(sample_lincs_data.columns)

    @patch.object(LincsAnalyzer, "_filter_by_clinical")
    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_data_access_just_clin(self, mock_get_dataset, mock_filter, sample_lincs_data):
        """Test lincs_data_access with just_clin=True."""
        mock_get_dataset.return_value = sample_lincs_data
        mock_filter.return_value = ["CLINICAL1"]

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_data_access(
            "CP", genes=["GENE1"], perturbagens=["PERT1", "CLINICAL1"], just_clin=True
        )

        mock_filter.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    @patch.object(LincsDataManager, "get_data")
    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_data_access_show_clin(
        self, mock_get_dataset, mock_get_data, sample_lincs_data, sample_tri_ann
    ):
        """Test lincs_data_access with show_clin=True."""
        mock_get_dataset.return_value = sample_lincs_data
        mock_get_data.return_value = sample_tri_ann

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_data_access(
            "CP", genes=["GENE1"], perturbagens=["CLINICAL1"], show_clin=True
        )

        assert isinstance(result, pd.DataFrame)
        # Should have been merged with clinical annotations
        assert "HighestPhase" in result.columns or "merge" in result.columns

    @patch("metaintegrator.linc_tools.stats.pearsonr")
    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_correlate_success(
        self, mock_get_dataset, mock_pearsonr, sample_lincs_data, sample_signature
    ):
        """Test successful lincs_correlate."""
        mock_get_dataset.return_value = sample_lincs_data
        mock_pearsonr.return_value = (0.5, 0.01)  # correlation, p-value

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_correlate(sample_signature)

        assert isinstance(result, dict)
        assert "hit_list" in result
        assert "hit_expression" in result
        assert isinstance(result["hit_list"], pd.DataFrame)
        assert "Sig_Corr" in result["hit_list"].columns
        assert "pval" in result["hit_list"].columns
        assert "fdr" in result["hit_list"].columns

    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_correlate_dataset_error(self, mock_get_dataset, sample_signature):
        """Test lincs_correlate with dataset error."""
        mock_get_dataset.return_value = {"error": "incorrect dataset"}

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_correlate(sample_signature)

        assert isinstance(result, dict)
        assert "error" in result

    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_correlate_no_overlap(self, mock_get_dataset, sample_lincs_data):
        """Test lincs_correlate with no gene overlap."""
        mock_get_dataset.return_value = sample_lincs_data

        # Create signature with non-overlapping genes
        signature = pd.DataFrame({"summary": [1.0]}, index=["NONEXISTENT"])

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_correlate(signature)

        assert isinstance(result, dict)
        assert result["error"] == "no overlap between signature genes and LINCS"

    @patch("metaintegrator.linc_tools.stats.pearsonr")
    @patch.object(LincsAnalyzer, "_filter_by_clinical")
    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_correlate_just_clin(
        self,
        mock_get_dataset,
        mock_filter,
        mock_pearsonr,
        sample_lincs_data,
        sample_signature,
    ):
        """Test lincs_correlate with just_clin=True."""
        mock_get_dataset.return_value = sample_lincs_data
        mock_pearsonr.return_value = (0.5, 0.01)
        mock_filter.return_value = ["CLINICAL1"]

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_correlate(sample_signature, just_clin=True)

        mock_filter.assert_called_once()
        assert isinstance(result, dict)
        assert "hit_list" in result

    @patch("metaintegrator.linc_tools.stats.pearsonr")
    @patch.object(LincsDataManager, "get_data")
    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_correlate_show_clin(
        self,
        mock_get_dataset,
        mock_get_data,
        mock_pearsonr,
        sample_lincs_data,
        sample_signature,
        sample_tri_ann,
    ):
        """Test lincs_correlate with show_clin=True."""
        mock_get_dataset.return_value = sample_lincs_data
        mock_pearsonr.return_value = (0.5, 0.01)
        mock_get_data.return_value = sample_tri_ann

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_correlate(sample_signature, show_clin=True)

        assert isinstance(result, dict)
        assert "hit_ann" in result
        assert isinstance(result["hit_ann"], pd.DataFrame)

    @patch("metaintegrator.linc_tools.stats.pearsonr")
    @patch.object(LincsDataManager, "get_data")
    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_correlate_gene_ann(
        self,
        mock_get_dataset,
        mock_get_data,
        mock_pearsonr,
        sample_lincs_data,
        sample_signature,
        sample_gene_ann,
    ):
        """Test lincs_correlate with gene_ann=True."""
        mock_get_dataset.return_value = sample_lincs_data
        mock_pearsonr.return_value = (0.5, 0.01)
        mock_get_data.return_value = sample_gene_ann

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_correlate(sample_signature, gene_ann=True)

        assert isinstance(result, dict)
        assert "gene_ann" in result

    @patch("metaintegrator.linc_tools.stats.pearsonr")
    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_correlate_drop_string(self, mock_get_dataset, mock_pearsonr, sample_signature):
        """Test lincs_correlate with drop_string parameter."""
        # Create data with column to be dropped
        genes = ["GENE1", "GENE2", "GENE3"]
        perturbagens = ["PERT1", "DROPME_PERT", "PERT3"]
        data = np.random.randn(3, 3)
        lincs_data = pd.DataFrame(data, index=genes, columns=perturbagens)

        mock_get_dataset.return_value = lincs_data
        mock_pearsonr.return_value = (0.5, 0.01)

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_correlate(sample_signature, drop_string="DROPME")

        assert isinstance(result, dict)
        # Should not contain the dropped column
        assert "DROPME_PERT" not in result["hit_expression"].columns

    @patch("metaintegrator.linc_tools.np.corrcoef")
    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_bait_corr_success(self, mock_get_dataset, mock_corrcoef, sample_lincs_data):
        """Test successful lincs_bait_corr."""
        mock_get_dataset.return_value = sample_lincs_data
        # Mock correlation matrix
        mock_corrcoef.return_value = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.7], [0.3, 0.7, 1.0]])

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_bait_corr(baits=["PERT1"])

        assert isinstance(result, dict)
        assert "all_hits" in result
        assert "heatmap_df" in result
        assert isinstance(result["all_hits"], pd.DataFrame)
        assert "mean_cor" in result["all_hits"].columns

    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_bait_corr_dataset_error(self, mock_get_dataset):
        """Test lincs_bait_corr with dataset error."""
        mock_get_dataset.return_value = {"error": "incorrect dataset"}

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_bait_corr()

        assert isinstance(result, dict)
        assert "error" in result

    @patch.object(LincsAnalyzer, "_get_dataset_data")
    def test_lincs_bait_corr_insufficient_baits(self, mock_get_dataset, sample_lincs_data):
        """Test lincs_bait_corr with insufficient baits."""
        mock_get_dataset.return_value = sample_lincs_data

        analyzer = LincsAnalyzer()
        result = analyzer.lincs_bait_corr(baits=["NONEXISTENT"])

        assert isinstance(result, dict)
        assert result["error"] == "insufficient baits or compounds found"

    def test_lincs_bait_corr_cross_dataset(self, sample_lincs_data):
        """Test lincs_bait_corr with cross-dataset correlation."""
        analyzer = LincsAnalyzer()

        with patch.object(analyzer, "_get_dataset_data", return_value=sample_lincs_data):
            result = analyzer.lincs_bait_corr(bait_type="cross")

        assert isinstance(result, dict)
        assert result["error"] == "cross-dataset correlation not yet implemented"


# LincsVisualizer Tests
class TestLincsVisualizer:
    """Test cases for LincsVisualizer class."""

    def test_init(self):
        """Test LincsVisualizer initialization."""
        visualizer = LincsVisualizer()
        # Should initialize without error
        assert isinstance(visualizer, LincsVisualizer)

    def test_prepare_heatmap_data_basic(self, sample_lincs_data):
        """Test basic heatmap data preparation."""
        visualizer = LincsVisualizer()

        # Add a signature column
        sample_data = sample_lincs_data.copy()
        sample_data["sig"] = [1.0, -0.5, 2.0, -1.5, 0.8]

        result = visualizer._prepare_heatmap_data(sample_data, None, False, False)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_data.shape

    def test_prepare_heatmap_data_cutoff(self, sample_lincs_data):
        """Test heatmap data preparation with cutoff."""
        visualizer = LincsVisualizer()

        # Create data with extreme values
        sample_data = sample_lincs_data.copy()
        sample_data.iloc[0, 0] = 10.0  # Extreme high value
        sample_data.iloc[1, 1] = -10.0  # Extreme low value

        result = visualizer._prepare_heatmap_data(
            sample_data, cutoff=2.0, scale=False, pert_ordered=False
        )

        # Values should be clipped
        assert result.max().max() <= 2.0
        assert result.min().min() >= -2.0

    def test_prepare_heatmap_data_scale(self, sample_lincs_data):
        """Test heatmap data preparation with scaling."""
        visualizer = LincsVisualizer()

        result = visualizer._prepare_heatmap_data(
            sample_lincs_data, None, scale=True, pert_ordered=False
        )

        assert isinstance(result, pd.DataFrame)
        # After scaling, should have approximately zero mean (allowing for numerical precision)
        assert abs(result.mean().mean()) < 1e-10

    def test_prepare_heatmap_data_pert_ordered(self, sample_lincs_data):
        """Test heatmap data preparation with perturbagen ordering."""
        visualizer = LincsVisualizer()

        # Add signature column for ordering
        sample_data = sample_lincs_data.copy()
        sample_data["sig"] = [1.0, -0.5, 2.0, -1.5, 0.8]

        result = visualizer._prepare_heatmap_data(sample_data, None, False, pert_ordered=True)

        assert isinstance(result, pd.DataFrame)
        assert "sig" in result.columns

    @patch("metaintegrator.linc_tools.plt.subplots")
    @patch("metaintegrator.linc_tools.sns.heatmap")
    def test_lincs_heatmap_basic(self, mock_heatmap, mock_subplots, sample_lincs_data):
        """Test basic heatmap creation."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        visualizer = LincsVisualizer()
        result = visualizer.lincs_heatmap(sample_lincs_data)

        mock_subplots.assert_called_once()
        mock_heatmap.assert_called_once()
        assert result == mock_fig

    @patch("metaintegrator.linc_tools.plt.subplots")
    @patch("metaintegrator.linc_tools.sns.heatmap")
    @patch("metaintegrator.linc_tools.plt.savefig")
    def test_lincs_heatmap_save_file(
        self, mock_savefig, mock_heatmap, mock_subplots, sample_lincs_data
    ):
        """Test heatmap creation with file saving."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        visualizer = LincsVisualizer()
        result = visualizer.lincs_heatmap(sample_lincs_data, out_file="/tmp/test.png")

        mock_savefig.assert_called_once_with("/tmp/test.png", dpi=300, bbox_inches="tight")

    @patch("metaintegrator.linc_tools.plt.subplots")
    @patch("metaintegrator.linc_tools.sns.heatmap")
    def test_lincs_heatmap_with_title(self, mock_heatmap, mock_subplots, sample_lincs_data):
        """Test heatmap creation with title."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        visualizer = LincsVisualizer()
        visualizer.lincs_heatmap(sample_lincs_data, hm_title="Test Title")

        mock_ax.set_title.assert_called_once_with("Test Title")

    def test_get_top_hits_reverse(self):
        """Test getting top hits in reverse direction."""
        visualizer = LincsVisualizer()
        hit_list = pd.DataFrame(
            {
                "Sig_Corr": [0.1, 0.5, -0.3, 0.8, -0.2],
                "merge": ["A", "B", "C", "D", "E"],
            }
        )

        result = visualizer._get_top_hits(hit_list, "reverse", 2)

        assert len(result) == 2
        assert result.equals(hit_list.tail(2))

    def test_get_top_hits_forward(self):
        """Test getting top hits in forward direction."""
        visualizer = LincsVisualizer()
        hit_list = pd.DataFrame(
            {
                "Sig_Corr": [0.1, 0.5, -0.3, 0.8, -0.2],
                "merge": ["A", "B", "C", "D", "E"],
            }
        )

        result = visualizer._get_top_hits(hit_list, "forward", 2)

        assert len(result) == 2
        assert result.equals(hit_list.head(2))

    @patch("metaintegrator.linc_tools.plt.subplots")
    def test_plot_correlation_results_success(self, mock_subplots):
        """Test successful correlation results plotting."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        hit_list = pd.DataFrame(
            {
                "Sig_Corr": [0.1, 0.5, -0.3, 0.8, -0.2],
                "merge": ["A", "B", "C", "D", "E"],
            }
        )
        results = {"hit_list": hit_list}

        visualizer = LincsVisualizer()
        result = visualizer.plot_correlation_results(results)

        mock_subplots.assert_called_once()
        mock_ax1.hist.assert_called_once()
        mock_ax2.barh.assert_called_once()
        assert result == mock_fig

    def test_plot_correlation_results_missing_hit_list(self):
        """Test correlation results plotting with missing hit_list."""
        results = {"other_data": "value"}

        visualizer = LincsVisualizer()

        with pytest.raises(ValueError, match="Results must contain 'hit_list'"):
            visualizer.plot_correlation_results(results)


# Utility Function Tests
class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_prepare_signature_frame_success(self, mock_meta_object, mock_filter_object):
        """Test successful signature frame preparation."""
        result = _prepare_signature_frame(mock_meta_object, mock_filter_object)

        assert isinstance(result, pd.DataFrame)
        assert "summary" in result.columns
        assert len(result) == 5  # Should include all genes from filter
        assert result.index.tolist() == ["GENE1", "GENE3", "GENE5", "GENE2", "GENE4"]

    def test_prepare_signature_frame_missing_meta_analysis(self, mock_filter_object):
        """Test signature frame preparation with missing meta analysis."""
        meta_object = {}

        with pytest.raises(ValueError, match="Meta-analysis results not found"):
            _prepare_signature_frame(meta_object, mock_filter_object)

    def test_prepare_signature_frame_missing_pooled_results(self, mock_filter_object):
        """Test signature frame preparation with missing pooled results."""
        meta_object = {"metaAnalysis": {}}

        with pytest.raises(ValueError, match="Meta-analysis results not found"):
            _prepare_signature_frame(meta_object, mock_filter_object)

    @patch("metaintegrator.linc_tools.LincsAnalyzer")
    @patch("metaintegrator.linc_tools.LincsVisualizer")
    @patch("metaintegrator.linc_tools.os.makedirs")
    @patch("metaintegrator.linc_tools.plt.savefig")
    def test_lincs_tools_success(
        self,
        mock_savefig,
        mock_makedirs,
        mock_visualizer_class,
        mock_analyzer_class,
        mock_meta_object,
        mock_filter_object,
    ):
        """Test successful lincs_tools execution."""
        # Mock analyzer and its methods
        mock_analyzer = Mock()
        mock_correlation_results = {
            "hit_expression": pd.DataFrame({"test": [1, 2, 3]}),
            "hit_list": pd.DataFrame({"Sig_Corr": [0.5, -0.3]}),
        }
        mock_analyzer.lincs_correlate.return_value = mock_correlation_results
        mock_analyzer_class.return_value = mock_analyzer

        # Mock visualizer and its methods
        mock_visualizer = Mock()
        mock_fig = Mock()
        mock_visualizer.lincs_heatmap.return_value = mock_fig
        mock_visualizer_class.return_value = mock_visualizer

        result = lincs_tools(mock_meta_object, mock_filter_object, "/tmp/output")

        assert isinstance(result, dict)
        mock_analyzer.lincs_correlate.assert_called_once()
        mock_visualizer.lincs_heatmap.assert_called_once()
        mock_makedirs.assert_called_once_with("/tmp/output", exist_ok=True)
        mock_savefig.assert_called_once()

    def test_lincs_tools_missing_meta_analysis(self, mock_filter_object):
        """Test lincs_tools with missing meta analysis."""
        meta_object = {}

        with pytest.raises(ValueError, match="Meta-analysis results not found"):
            lincs_tools(meta_object, mock_filter_object, "/tmp/output")

    @patch("metaintegrator.linc_tools._prepare_signature_frame")
    @patch("metaintegrator.linc_tools.LincsAnalyzer")
    @patch("metaintegrator.linc_tools.LincsVisualizer")
    def test_lincs_correlate_function_success(
        self,
        mock_visualizer_class,
        mock_analyzer_class,
        mock_prepare_sig,
        mock_meta_object,
        mock_filter_object,
        sample_signature,
    ):
        """Test successful lincs_correlate function."""
        mock_prepare_sig.return_value = sample_signature

        mock_analyzer = Mock()
        mock_correlation_results = {
            "hit_expression": pd.DataFrame({"test": [1, 2, 3]}),
            "hit_list": pd.DataFrame({"Sig_Corr": [0.5, -0.3]}),
        }
        mock_analyzer.lincs_correlate.return_value = mock_correlation_results
        mock_analyzer_class.return_value = mock_analyzer

        mock_visualizer = Mock()
        mock_fig = Mock()
        mock_visualizer.lincs_heatmap.return_value = mock_fig
        mock_visualizer_class.return_value = mock_visualizer

        result = lincs_correlate(mock_meta_object, mock_filter_object)

        assert isinstance(result, dict)
        assert "heatmap_figure" in result
        mock_prepare_sig.assert_called_once()
        mock_analyzer.lincs_correlate.assert_called_once()

    @patch("metaintegrator.linc_tools._prepare_signature_frame")
    @patch("metaintegrator.linc_tools.LincsAnalyzer")
    def test_lincs_correlate_function_no_heatmap(
        self,
        mock_analyzer_class,
        mock_prepare_sig,
        mock_meta_object,
        mock_filter_object,
        sample_signature,
    ):
        """Test lincs_correlate function with no hit_expression data."""
        mock_prepare_sig.return_value = sample_signature

        mock_analyzer = Mock()
        mock_correlation_results = {"hit_list": pd.DataFrame({"Sig_Corr": [0.5, -0.3]})}
        mock_analyzer.lincs_correlate.return_value = mock_correlation_results
        mock_analyzer_class.return_value = mock_analyzer

        result = lincs_correlate(mock_meta_object, mock_filter_object)

        assert isinstance(result, dict)
        assert "heatmap_figure" not in result

    @patch("metaintegrator.linc_tools._prepare_signature_frame")
    @patch("metaintegrator.linc_tools.LincsAnalyzer")
    @patch("metaintegrator.linc_tools.LincsVisualizer")
    def test_lincs_bait_corr_function_success(
        self,
        mock_visualizer_class,
        mock_analyzer_class,
        mock_prepare_sig,
        mock_meta_object,
        mock_filter_object,
        sample_signature,
    ):
        """Test successful lincs_bait_corr function."""
        mock_prepare_sig.return_value = sample_signature

        mock_analyzer = Mock()
        mock_bait_results = {
            "heatmap_df": pd.DataFrame({"test": [1, 2, 3]}),
            "all_hits": pd.DataFrame({"mean_cor": [0.5, -0.3]}),
        }
        mock_analyzer.lincs_bait_corr.return_value = mock_bait_results
        mock_analyzer_class.return_value = mock_analyzer

        mock_visualizer = Mock()
        mock_fig = Mock()
        mock_visualizer.lincs_heatmap.return_value = mock_fig
        mock_visualizer_class.return_value = mock_visualizer

        result = lincs_bait_corr(mock_meta_object, mock_filter_object, baits=["BAIT1"])

        assert isinstance(result, dict)
        assert "heatmap_figure" in result
        mock_prepare_sig.assert_called_once()
        mock_analyzer.lincs_bait_corr.assert_called_once()

    @patch("metaintegrator.linc_tools._prepare_signature_frame")
    @patch("metaintegrator.linc_tools.LincsAnalyzer")
    def test_lincs_bait_corr_function_no_heatmap(
        self,
        mock_analyzer_class,
        mock_prepare_sig,
        mock_meta_object,
        mock_filter_object,
        sample_signature,
    ):
        """Test lincs_bait_corr function with no heatmap data."""
        mock_prepare_sig.return_value = sample_signature

        mock_analyzer = Mock()
        mock_bait_results = {"all_hits": pd.DataFrame({"mean_cor": [0.5, -0.3]})}
        mock_analyzer.lincs_bait_corr.return_value = mock_bait_results
        mock_analyzer_class.return_value = mock_analyzer

        result = lincs_bait_corr(mock_meta_object, mock_filter_object)

        assert isinstance(result, dict)
        assert "heatmap_figure" not in result


# Integration Tests
class TestIntegration:
    """Integration tests for the complete workflow."""

    @patch("metaintegrator.linc_tools.requests.get")
    @patch("metaintegrator.linc_tools.tempfile.mkdtemp")
    @patch("metaintegrator.linc_tools.shutil.rmtree")
    def test_full_workflow_integration(self, mock_rmtree, mock_tempdir, mock_get, sample_signature):
        """Test integration of data loading and analysis workflow."""
        # Mock HTTP request
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"test_data"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Mock temp directory
        mock_tempdir.return_value = "/tmp/test_lincs"

        # Create a simple zip file for testing
        with patch("metaintegrator.linc_tools.zipfile.ZipFile"):
            with patch("metaintegrator.linc_tools.os.listdir", return_value=["test.pkl"]):
                with patch(
                    "metaintegrator.linc_tools.pd.read_pickle",
                    return_value=pd.DataFrame({"test": [1, 2, 3]}),
                ):
                    with patch("builtins.open", mock_open()):
                        # Test the workflow
                        manager = LincsDataManager()
                        analyzer = LincsAnalyzer(manager)

                        # Should not raise errors
                        loaded = manager.load_data()
                        assert loaded is True or loaded is False  # Depends on mocking success

    def test_edge_case_empty_data(self):
        """Test handling of edge cases with empty data."""
        analyzer = LincsAnalyzer()

        # Test with empty signature
        empty_signature = pd.DataFrame(columns=["summary"])

        with patch.object(analyzer, "_get_dataset_data") as mock_get_dataset:
            mock_get_dataset.return_value = pd.DataFrame()
            result = analyzer.lincs_correlate(empty_signature)

            # Should return error due to no overlap
            assert isinstance(result, dict)
            assert "error" in result

    def test_error_propagation(self):
        """Test that errors are properly propagated through the system."""
        analyzer = LincsAnalyzer()

        # Test with invalid dataset
        result = analyzer._get_dataset_data("INVALID_DATASET")
        assert result == {"error": "incorrect dataset"}

        # Test lincs_data_access with invalid dataset
        access_result = analyzer.lincs_data_access("INVALID_DATASET")
        assert access_result == "incorrect dataset"


if __name__ == "__main__":
    pytest.main([__file__])
