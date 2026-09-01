#!/usr/bin/env python3
"""
Comprehensive unit tests for coconut_runner.py module.

This test suite provides extensive coverage of the COCONUT data loading,
processing, and output generation functionality.

Test Coverage:
    - Data loading from CSV files
    - File discovery and grouping
    - Dataset name extraction
    - Sample alignment validation
    - Output generation
    - Error handling and edge cases
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from coconut.coconut_runner import (
    _extract_dataset_name,
    _group_files_by_dataset,
    _load_single_dataset,
    _print_dataset_summary,
    _write_output_to_csv,
    debug_mode,
    debug_msg,
    load_and_run_coconut_example,
    load_gse_data_from_csv_directory,
    main,
    runner,
)

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def temp_csv_directory():
    """Create a temporary directory for test CSV files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_genes_df():
    """Create a sample genes DataFrame."""
    np.random.seed(42)
    gene_names = [f"Gene_{i:04d}" for i in range(50)]
    sample_names = [f"Sample_{i:03d}" for i in range(20)]
    data = np.random.randn(50, 20)
    return pd.DataFrame(data, index=gene_names, columns=sample_names)


@pytest.fixture
def sample_pheno_df():
    """Create a sample phenotype DataFrame."""
    sample_names = [f"Sample_{i:03d}" for i in range(20)]
    data = {
        "Healthy0.Sepsis1": [0] * 10 + [1] * 10,
        "Platform_id": ["Affy"] * 10 + ["Illumina"] * 10,
        "age": np.random.randint(20, 80, 20),
        "sex": np.random.choice(["M", "F"], 20),
    }
    return pd.DataFrame(data, index=sample_names)


@pytest.fixture
def valid_csv_files(temp_csv_directory, sample_genes_df, sample_pheno_df):
    """Create valid CSV files in the temporary directory."""
    # Dataset 1
    genes_file1 = os.path.join(temp_csv_directory, "GSEs.test_GSE1_genes.csv")
    pheno_file1 = os.path.join(temp_csv_directory, "GSEs.test_GSE1_pheno.csv")
    sample_genes_df.to_csv(genes_file1)
    sample_pheno_df.to_csv(pheno_file1)

    # Dataset 2
    genes_file2 = os.path.join(temp_csv_directory, "GSEs.test_GSE2_genes.csv")
    pheno_file2 = os.path.join(temp_csv_directory, "GSEs.test_GSE2_pheno.csv")
    sample_genes_df.to_csv(genes_file2)
    sample_pheno_df.to_csv(pheno_file2)

    return [genes_file1, pheno_file1, genes_file2, pheno_file2]


@pytest.fixture
def gse_data_list(sample_genes_df, sample_pheno_df):
    """Create a list of GSE data dictionaries for testing."""
    return [
        {"genes": sample_genes_df.copy(), "pheno": sample_pheno_df.copy()},
        {"genes": sample_genes_df.copy(), "pheno": sample_pheno_df.copy()},
    ]


# ==============================================================================
# Test Cases for debug_msg
# ==============================================================================


class TestDebugMsg:
    """Tests for the debug_msg helper function."""

    def test_debug_msg_when_enabled(self, capsys):
        """Test that debug_msg prints when debug_mode is True."""
        import coconut.coconut_runner as runner_module

        original_debug_mode = runner_module.debug_mode
        try:
            runner_module.debug_mode = True
            debug_msg("Test debug message")
            captured = capsys.readouterr()
            assert "Test debug message" in captured.err
        finally:
            runner_module.debug_mode = original_debug_mode

    def test_debug_msg_when_disabled(self, capsys):
        """Test that debug_msg doesn't print when debug_mode is False."""
        import coconut.coconut_runner as runner_module

        original_debug_mode = runner_module.debug_mode
        try:
            runner_module.debug_mode = False
            debug_msg("Test debug message")
            captured = capsys.readouterr()
            assert captured.err == ""
        finally:
            runner_module.debug_mode = original_debug_mode


# ==============================================================================
# Test Cases for _extract_dataset_name
# ==============================================================================


class TestExtractDatasetName:
    """Tests for the _extract_dataset_name function."""

    def test_extract_basic_name(self):
        """Test extraction of basic dataset name."""
        filename = "GSEs.test_GSE12345_genes.csv"
        result = _extract_dataset_name(filename, "GSEs.test_", "_genes.csv", "_pheno.csv")
        assert result == "GSE12345"

    def test_extract_with_pheno_suffix(self):
        """Test extraction when filename has pheno suffix."""
        filename = "GSEs.test_GSE54321_pheno.csv"
        result = _extract_dataset_name(filename, "GSEs.test_", "_genes.csv", "_pheno.csv")
        assert result == "GSE54321"

    def test_extract_without_prefix(self):
        """Test extraction when filename doesn't have prefix."""
        filename = "GSE12345_genes.csv"
        result = _extract_dataset_name(filename, "GSEs.test_", "_genes.csv", "_pheno.csv")
        assert result == "GSE12345"

    def test_extract_with_empty_prefix(self):
        """Test extraction with empty prefix."""
        filename = "Dataset1_genes.csv"
        result = _extract_dataset_name(filename, "", "_genes.csv", "_pheno.csv")
        assert result == "Dataset1"

    def test_extract_complex_name(self):
        """Test extraction with complex dataset name."""
        filename = "GSEs.test_GSE12345_batch2_genes.csv"
        result = _extract_dataset_name(filename, "GSEs.test_", "_genes.csv", "_pheno.csv")
        assert result == "GSE12345_batch2"


# ==============================================================================
# Test Cases for _group_files_by_dataset
# ==============================================================================


class TestGroupFilesByDataset:
    """Tests for the _group_files_by_dataset function."""

    def test_group_valid_files(self, temp_csv_directory):
        """Test grouping of valid gene and pheno files."""
        # Create test files
        genes_file = Path(temp_csv_directory) / "GSEs.test_DS1_genes.csv"
        pheno_file = Path(temp_csv_directory) / "GSEs.test_DS1_pheno.csv"
        genes_file.touch()
        pheno_file.touch()

        csv_files = [genes_file, pheno_file]
        result = _group_files_by_dataset(csv_files, "_genes.csv", "_pheno.csv", "GSEs.test_")

        assert len(result) == 1
        assert "DS1" in result
        assert "genes" in result["DS1"]
        assert "pheno" in result["DS1"]

    def test_group_multiple_datasets(self, temp_csv_directory):
        """Test grouping multiple complete datasets."""
        files = []
        for i in range(3):
            genes_file = Path(temp_csv_directory) / f"GSEs.test_DS{i}_genes.csv"
            pheno_file = Path(temp_csv_directory) / f"GSEs.test_DS{i}_pheno.csv"
            genes_file.touch()
            pheno_file.touch()
            files.extend([genes_file, pheno_file])

        result = _group_files_by_dataset(files, "_genes.csv", "_pheno.csv", "GSEs.test_")

        assert len(result) == 3
        for i in range(3):
            assert f"DS{i}" in result

    def test_group_incomplete_dataset(self, temp_csv_directory):
        """Test handling of incomplete datasets (missing pheno or genes file)."""
        # Create only genes file (missing pheno)
        genes_only = Path(temp_csv_directory) / "GSEs.test_Incomplete_genes.csv"
        genes_only.touch()

        csv_files = [genes_only]

        # No pheno file means this dataset never becomes "complete".
        with pytest.raises(ValueError, match="No complete datasets found"):
            _group_files_by_dataset(csv_files, "_genes.csv", "_pheno.csv", "GSEs.test_")

    def test_group_no_matching_files(self, temp_csv_directory):
        """Test when no files match the patterns."""
        # Create files with wrong patterns
        wrong_file = Path(temp_csv_directory) / "random_data.csv"
        wrong_file.touch()

        with pytest.raises(ValueError, match="No complete datasets found"):
            _group_files_by_dataset([wrong_file], "_genes.csv", "_pheno.csv", "GSEs.test_")

    def test_group_skip_unrecognized_files(self, temp_csv_directory):
        """Test that unrecognized files are skipped."""
        # Create valid and invalid files
        genes_file = Path(temp_csv_directory) / "GSEs.test_DS1_genes.csv"
        pheno_file = Path(temp_csv_directory) / "GSEs.test_DS1_pheno.csv"
        random_file = Path(temp_csv_directory) / "readme.txt"

        genes_file.touch()
        pheno_file.touch()
        random_file.touch()

        csv_files = [genes_file, pheno_file, random_file]
        result = _group_files_by_dataset(csv_files, "_genes.csv", "_pheno.csv", "GSEs.test_")

        assert len(result) == 1
        assert "DS1" in result


# ==============================================================================
# Test Cases for _load_single_dataset
# ==============================================================================


class TestLoadSingleDataset:
    """Tests for the _load_single_dataset function."""

    def test_load_valid_dataset(self, temp_csv_directory, sample_genes_df, sample_pheno_df):
        """Test loading a valid dataset with matching samples."""
        genes_file = Path(temp_csv_directory) / "genes.csv"
        pheno_file = Path(temp_csv_directory) / "pheno.csv"

        sample_genes_df.to_csv(genes_file)
        sample_pheno_df.to_csv(pheno_file)

        file_info = {"genes": genes_file, "pheno": pheno_file}
        result = _load_single_dataset("TestDataset", file_info)

        assert "genes" in result
        assert "pheno" in result
        assert isinstance(result["genes"], pd.DataFrame)
        assert isinstance(result["pheno"], pd.DataFrame)
        assert result["genes"].shape[1] == result["pheno"].shape[0]

    def test_load_dataset_with_sample_mismatch(self, temp_csv_directory, capsys):
        """Test loading dataset with mismatched samples between genes and pheno."""
        # Create genes with one set of samples
        genes_samples = [f"Sample_{i}" for i in range(10)]
        genes_df = pd.DataFrame(
            np.random.randn(50, 10),
            index=[f"Gene_{i}" for i in range(50)],
            columns=genes_samples,
        )

        # Create pheno with different samples
        pheno_samples = [f"Sample_{i}" for i in range(5, 15)]  # 5 common, 5 different
        pheno_df = pd.DataFrame({"attr1": range(10)}, index=pheno_samples)

        genes_file = Path(temp_csv_directory) / "genes.csv"
        pheno_file = Path(temp_csv_directory) / "pheno.csv"
        genes_df.to_csv(genes_file)
        pheno_df.to_csv(pheno_file)

        file_info = {"genes": genes_file, "pheno": pheno_file}
        result = _load_single_dataset("MismatchDataset", file_info)

        # Should only include common samples
        captured = capsys.readouterr()
        assert "Sample mismatch" in captured.out
        assert result["genes"].shape[1] == result["pheno"].shape[0]

    def test_load_dataset_no_common_samples(self, temp_csv_directory):
        """Test error when no common samples exist."""
        genes_samples = [f"Sample_A_{i}" for i in range(10)]
        genes_df = pd.DataFrame(
            np.random.randn(50, 10),
            index=[f"Gene_{i}" for i in range(50)],
            columns=genes_samples,
        )

        pheno_samples = [f"Sample_B_{i}" for i in range(10)]  # Completely different
        pheno_df = pd.DataFrame({"attr1": range(10)}, index=pheno_samples)

        genes_file = Path(temp_csv_directory) / "genes.csv"
        pheno_file = Path(temp_csv_directory) / "pheno.csv"
        genes_df.to_csv(genes_file)
        pheno_df.to_csv(pheno_file)

        file_info = {"genes": genes_file, "pheno": pheno_file}

        with pytest.raises(ValueError, match="No common samples found"):
            _load_single_dataset("NoCommonDataset", file_info)

    def test_load_dataset_with_whitespace_samples(self, temp_csv_directory):
        """Test handling of sample names with whitespace."""
        genes_samples = [f" Sample_{i} " for i in range(10)]  # With spaces
        genes_df = pd.DataFrame(
            np.random.randn(50, 10),
            index=[f"Gene_{i}" for i in range(50)],
            columns=genes_samples,
        )

        pheno_samples = [f"Sample_{i}" for i in range(10)]  # Without spaces
        pheno_df = pd.DataFrame({"attr1": range(10)}, index=pheno_samples)

        genes_file = Path(temp_csv_directory) / "genes.csv"
        pheno_file = Path(temp_csv_directory) / "pheno.csv"
        genes_df.to_csv(genes_file)
        pheno_df.to_csv(pheno_file)

        file_info = {"genes": genes_file, "pheno": pheno_file}
        result = _load_single_dataset("WhitespaceDataset", file_info)

        # Should match after stripping whitespace
        assert result["genes"].shape[1] == 10
        assert result["pheno"].shape[0] == 10


# ==============================================================================
# Test Cases for _print_dataset_summary
# ==============================================================================


class TestPrintDatasetSummary:
    """Tests for the _print_dataset_summary function."""

    def test_print_summary_single_dataset(self, gse_data_list, capsys):
        """Test printing summary for a single dataset."""
        _print_dataset_summary([gse_data_list[0]])

        captured = capsys.readouterr()
        assert "DATASET SUMMARY" in captured.out
        assert "Genes:" in captured.out
        assert "Pheno:" in captured.out
        assert "Overall:" in captured.out

    def test_print_summary_multiple_datasets(self, gse_data_list, capsys):
        """Test printing summary for multiple datasets."""
        _print_dataset_summary(gse_data_list)

        captured = capsys.readouterr()
        assert "[1] Dataset 1" in captured.out
        assert "[2] Dataset 2" in captured.out
        assert "Total unique genes:" in captured.out
        assert "Total samples:" in captured.out

    def test_print_summary_with_key_columns(self, capsys):
        """Test that key columns are identified and displayed."""
        genes_df = pd.DataFrame(np.random.randn(10, 5))
        pheno_df = pd.DataFrame(
            {
                "Healthy0_Sepsis1": [0, 1, 0, 1, 0],
                "platform": ["Affy"] * 5,
                "age": [25, 30, 35, 40, 45],
            }
        )

        gse_data = [{"genes": genes_df, "pheno": pheno_df}]
        _print_dataset_summary(gse_data)

        captured = capsys.readouterr()
        assert "Key columns:" in captured.out

    def test_print_summary_empty_list(self, capsys):
        """Test printing summary with empty dataset list."""
        _print_dataset_summary([])

        captured = capsys.readouterr()
        assert "DATASET SUMMARY" in captured.out
        assert "Total unique genes: 0" in captured.out


# ==============================================================================
# Test Cases for load_gse_data_from_csv_directory
# ==============================================================================


class TestLoadGseDataFromCsvDirectory:
    """Tests for the load_gse_data_from_csv_directory function."""

    def test_load_valid_directory(self, valid_csv_files, temp_csv_directory):
        """Test loading data from a valid directory."""
        result = load_gse_data_from_csv_directory(temp_csv_directory)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all("genes" in gse for gse in result)
        assert all("pheno" in gse for gse in result)

    def test_load_nonexistent_directory(self):
        """Test error when directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            load_gse_data_from_csv_directory("/nonexistent/directory")

    def test_load_empty_directory(self, temp_csv_directory):
        """Test error when directory has no CSV files."""
        with pytest.raises(FileNotFoundError, match="No CSV files found"):
            load_gse_data_from_csv_directory(temp_csv_directory)

    def test_load_with_custom_patterns(self, temp_csv_directory, sample_genes_df, sample_pheno_df):
        """Test loading with custom file patterns."""
        # Create files with custom patterns
        genes_file = Path(temp_csv_directory) / "dataset1_expression.csv"
        pheno_file = Path(temp_csv_directory) / "dataset1_metadata.csv"
        sample_genes_df.to_csv(genes_file)
        sample_pheno_df.to_csv(pheno_file)

        result = load_gse_data_from_csv_directory(
            temp_csv_directory,
            genes_pattern="_expression.csv",
            pheno_pattern="_metadata.csv",
            object_prefix="",
        )

        assert len(result) == 1
        assert result[0]["genes"].shape[0] > 0

    def test_load_with_incomplete_datasets(self, temp_csv_directory, sample_genes_df):
        """Test that incomplete datasets are skipped."""
        # Create only genes files (no pheno)
        genes_file1 = Path(temp_csv_directory) / "GSEs.test_DS1_genes.csv"
        genes_file2 = Path(temp_csv_directory) / "GSEs.test_DS2_genes.csv"
        sample_genes_df.to_csv(genes_file1)
        sample_genes_df.to_csv(genes_file2)

        with pytest.raises(ValueError, match="No complete datasets found"):
            load_gse_data_from_csv_directory(temp_csv_directory)


# ==============================================================================
# Test Cases for _write_output_to_csv
# ==============================================================================


class TestWriteOutputToCsv:
    """Tests for the _write_output_to_csv function."""

    def test_write_valid_output(self, temp_csv_directory, sample_genes_df, sample_pheno_df):
        """Test writing valid output to CSV files."""
        combined_result = {
            "genes": sample_genes_df,
            "pheno": sample_pheno_df,
            "class_cntl0_dis1": np.array([0] * 10 + [1] * 10),
        }

        genes_file = Path(temp_csv_directory) / "test_genes_samples.csv"
        samples_genes_file = Path(temp_csv_directory) / "test_genes.csv"
        pheno_file = Path(temp_csv_directory) / "test_pheno.csv"
        class_file = Path(temp_csv_directory) / "test_class.csv"

        _write_output_to_csv(
            combined_result,
            str(genes_file),
            str(samples_genes_file),
            str(pheno_file),
            str(class_file),
        )

        # Verify files were created
        assert genes_file.exists()
        assert samples_genes_file.exists()
        assert pheno_file.exists()
        assert class_file.exists()

        # Verify content
        genes_loaded = pd.read_csv(genes_file, index_col=0)
        assert genes_loaded.shape == sample_genes_df.shape

        class_loaded = pd.read_csv(class_file)
        assert "sample_id" in class_loaded.columns
        assert "class_label" in class_loaded.columns
        assert len(class_loaded) == 20

    def test_write_transposed_genes(self, temp_csv_directory, sample_genes_df, sample_pheno_df):
        """Test that samples × genes file is correctly transposed."""
        combined_result = {
            "genes": sample_genes_df,
            "pheno": sample_pheno_df,
            "class_cntl0_dis1": np.zeros(20),
        }

        genes_file = Path(temp_csv_directory) / "genes_s.csv"
        samples_genes_file = Path(temp_csv_directory) / "genes_t.csv"
        pheno_file = Path(temp_csv_directory) / "pheno.csv"
        class_file = Path(temp_csv_directory) / "class.csv"

        _write_output_to_csv(
            combined_result,
            str(genes_file),
            str(samples_genes_file),
            str(pheno_file),
            str(class_file),
        )

        # Check dimensions
        genes_t = pd.read_csv(samples_genes_file, index_col=0)
        assert genes_t.shape == (20, 50)  # Transposed: samples × genes

    def test_write_class_labels_as_integers(
        self, temp_csv_directory, sample_genes_df, sample_pheno_df, capsys
    ):
        """Test that class labels are written as integers, not floats."""
        combined_result = {
            "genes": sample_genes_df,
            "pheno": sample_pheno_df,
            "class_cntl0_dis1": np.array([0.0, 1.0, 0.0, 1.0] * 5),
        }

        genes_file = Path(temp_csv_directory) / "genes_s.csv"
        samples_genes_file = Path(temp_csv_directory) / "genes_t.csv"
        pheno_file = Path(temp_csv_directory) / "pheno.csv"
        class_file = Path(temp_csv_directory) / "class.csv"

        _write_output_to_csv(
            combined_result,
            str(genes_file),
            str(samples_genes_file),
            str(pheno_file),
            str(class_file),
        )

        class_loaded = pd.read_csv(class_file)
        # Check that values are integers
        assert class_loaded["class_label"].dtype in [np.int64, np.int32]

        # Check distribution message
        captured = capsys.readouterr()
        assert "Class distribution:" in captured.out


# ==============================================================================
# Test Cases for load_and_run_coconut_example
# ==============================================================================


class TestLoadAndRunCoconutExample:
    """Tests for the load_and_run_coconut_example function."""

    @patch("coconut.coconut_runner.load_gse_data_from_csv_directory")
    @patch("coconut.coconut_runner.coconut")
    def test_run_example_success(self, mock_coconut, mock_load_data, gse_data_list):
        """Test successful execution of COCONUT example."""
        mock_load_data.return_value = gse_data_list
        mock_coconut.return_value = {
            "COCONUTList": gse_data_list,
            "rawDiseaseList": gse_data_list,
            "controlList": {"bayes_params": {}, "get_platforms": []},
        }

        result = load_and_run_coconut_example("/fake/path")

        assert result is not None
        assert "COCONUTList" in result
        mock_load_data.assert_called_once()
        mock_coconut.assert_called_once()

        # Default column names should be forwarded to coconut() unchanged.
        _, kwargs = mock_coconut.call_args
        assert kwargs["control_0_col"] == "Healthy0.Sepsis1"
        assert kwargs["disease_col"] is None
        assert kwargs["by_platform"] is False
        assert kwargs["platform_col"] == "Platform_id"
        assert kwargs["par_prior"] is True
        assert kwargs["it_conv"] == 1e-04

    @patch("coconut.coconut_runner.load_gse_data_from_csv_directory")
    @patch("coconut.coconut_runner.coconut")
    def test_run_example_failure(self, mock_coconut, mock_load_data):
        """Test handling of COCONUT failure."""
        mock_load_data.return_value = []
        mock_coconut.side_effect = Exception("COCONUT failed")

        with pytest.raises(Exception, match="COCONUT failed"):
            load_and_run_coconut_example("/fake/path")

    @patch("coconut.coconut_runner.load_gse_data_from_csv_directory")
    @patch("coconut.coconut_runner.coconut")
    def test_run_example_custom_columns(self, mock_coconut, mock_load_data, gse_data_list):
        """Test that custom column names are forwarded to coconut() instead of the
        project's own defaults: this is the fix for the hardcoded-columns limitation."""
        mock_load_data.return_value = gse_data_list
        mock_coconut.return_value = {
            "COCONUTList": [],
            "rawDiseaseList": [],
            "controlList": {},
        }

        load_and_run_coconut_example(
            "/fake/path",
            control_0_col="Group",
            disease_col="Case",
            by_platform=True,
            platform_col="Batch",
            par_prior=False,
            it_conv=1e-03,
        )

        _, kwargs = mock_coconut.call_args
        assert kwargs["control_0_col"] == "Group"
        assert kwargs["disease_col"] == "Case"
        assert kwargs["by_platform"] is True
        assert kwargs["platform_col"] == "Batch"
        assert kwargs["par_prior"] is False
        assert kwargs["it_conv"] == 1e-03


# ==============================================================================
# Test Cases for runner
# ==============================================================================


class TestRunner:
    """Tests for the main runner function."""

    @patch("coconut.coconut_runner.load_gse_data_from_csv_directory")
    @patch("coconut.coconut_runner.load_and_run_coconut_example")
    @patch("coconut.coconut_runner.combine_coco_output")
    @patch("coconut.coconut_runner._write_output_to_csv")
    def test_runner_with_output_dir(
        self, mock_write, mock_combine, mock_run, mock_load, temp_csv_directory
    ):
        """Test runner with output directory specified."""
        mock_load.return_value = []
        mock_run.return_value = {
            "COCONUTList": [],
            "rawDiseaseList": [],
            "controlList": {},
        }
        mock_combine.return_value = {
            "genes": pd.DataFrame(),
            "pheno": pd.DataFrame(),
            "class_cntl0_dis1": np.array([]),
        }

        output_dir = os.path.join(temp_csv_directory, "output")
        runner("/fake/csv/dir", output_dir)

        # Verify output directory was created
        assert os.path.exists(output_dir)
        mock_write.assert_called_once()

        # runner() should forward its default column-name args to
        # load_and_run_coconut_example() unchanged.
        _, kwargs = mock_run.call_args
        assert kwargs["control_0_col"] == "Healthy0.Sepsis1"
        assert kwargs["platform_col"] == "Platform_id"

    @patch("coconut.coconut_runner.load_gse_data_from_csv_directory")
    @patch("coconut.coconut_runner.load_and_run_coconut_example")
    @patch("coconut.coconut_runner.combine_coco_output")
    def test_runner_without_output_dir(self, mock_combine, mock_run, mock_load):
        """Test runner without output directory (no file writing)."""
        mock_load.return_value = []
        mock_run.return_value = {
            "COCONUTList": [],
            "rawDiseaseList": [],
            "controlList": {},
        }
        mock_combine.return_value = {
            "genes": pd.DataFrame(),
            "pheno": pd.DataFrame(),
            "class_cntl0_dis1": np.array([]),
        }

        # Should not raise error without output_dir
        runner("/fake/csv/dir", None)

    @patch("coconut.coconut_runner.load_gse_data_from_csv_directory")
    def test_runner_handles_exception(self, mock_load, capsys):
        """Test that runner handles exceptions gracefully."""
        mock_load.side_effect = Exception("Load failed")

        runner("/fake/csv/dir", None)

        captured = capsys.readouterr()
        assert "Error:" in captured.out


# ==============================================================================
# Test Cases for main
# ==============================================================================


class TestMain:
    """Tests for the main entry point function."""

    @patch("coconut.coconut_runner.runner")
    def test_main_with_valid_args(self, mock_runner):
        """Test main function with valid arguments. Column-name keys are absent from
        `args`, so main() should fill in the same defaults load_and_run_coconut_example()
        uses."""
        args = {
            "csv_dir": "/fake/path",
            "output_dir": "/fake/output",
            "debug": False,
        }

        main(args)
        mock_runner.assert_called_once_with(
            csv_dir="/fake/path",
            output_dir="/fake/output",
            control_0_col="Healthy0.Sepsis1",
            disease_col=None,
            by_platform=False,
            platform_col="Platform_id",
            par_prior=True,
            it_conv=1e-04,
        )

    @patch("coconut.coconut_runner.runner")
    def test_main_with_custom_columns(self, mock_runner):
        """Test that column-name keys present in `args` (e.g. from --control-col or a
        config file) are passed through to runner() instead of the defaults."""
        args = {
            "csv_dir": "/fake/path",
            "output_dir": None,
            "debug": False,
            "control_0_col": "Group",
            "disease_col": "Case",
            "by_platform": True,
            "platform_col": "Batch",
            "par_prior": False,
            "it_conv": 1e-03,
        }

        main(args)
        mock_runner.assert_called_once_with(
            csv_dir="/fake/path",
            output_dir=None,
            control_0_col="Group",
            disease_col="Case",
            by_platform=True,
            platform_col="Batch",
            par_prior=False,
            it_conv=1e-03,
        )

    @patch("coconut.coconut_runner.runner")
    def test_main_with_debug_mode(self, mock_runner):
        """Test main function with debug mode enabled."""
        args = {
            "csv_dir": "/fake/path",
            "output_dir": None,
            "debug": True,
        }

        main(args)
        mock_runner.assert_called_once()

    @patch("coconut.coconut_runner.runner")
    @patch("sys.exit")
    def test_main_handles_exception(self, mock_exit, mock_runner, capsys):
        """Test that main handles exceptions and exits with error code."""
        mock_runner.side_effect = Exception("Runner failed")

        args = {"csv_dir": "/fake/path", "output_dir": None, "debug": False}
        main(args)

        mock_exit.assert_called_once_with(1)
        captured = capsys.readouterr()
        assert "Error:" in captured.err


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_workflow_with_real_files(
        self, temp_csv_directory, sample_genes_df, sample_pheno_df
    ):
        """Test the complete workflow with actual CSV files."""
        # Create test files
        for i in range(2):
            genes_file = Path(temp_csv_directory) / f"GSEs.test_DS{i}_genes.csv"
            pheno_file = Path(temp_csv_directory) / f"GSEs.test_DS{i}_pheno.csv"
            sample_genes_df.to_csv(genes_file)
            sample_pheno_df.to_csv(pheno_file)

        # Load data
        gse_data = load_gse_data_from_csv_directory(temp_csv_directory)

        # Verify results
        assert len(gse_data) == 2
        assert all(isinstance(d["genes"], pd.DataFrame) for d in gse_data)
        assert all(isinstance(d["pheno"], pd.DataFrame) for d in gse_data)
        assert all(d["genes"].shape[0] == 50 for d in gse_data)
        assert all(d["genes"].shape[1] == 20 for d in gse_data)


# ==============================================================================
# Edge Cases and Error Handling Tests
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_load_dataset_with_empty_dataframes(self, temp_csv_directory):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()

        genes_file = Path(temp_csv_directory) / "empty_genes.csv"
        pheno_file = Path(temp_csv_directory) / "empty_pheno.csv"
        empty_df.to_csv(genes_file)
        empty_df.to_csv(pheno_file)

        file_info = {"genes": genes_file, "pheno": pheno_file}

        # Should handle empty dataframes gracefully
        try:
            result = _load_single_dataset("EmptyDataset", file_info)
            # If it doesn't raise an error, check the result is meaningful
            assert result is not None
        except (ValueError, IndexError, KeyError) as e:
            # Expected behavior: raise an error for empty data
            assert True

    def test_load_directory_with_special_characters(
        self, temp_csv_directory, sample_genes_df, sample_pheno_df
    ):
        """Test handling of filenames with special characters."""
        # Create files with special characters
        genes_file = Path(temp_csv_directory) / "GSEs.test_DS-1_v2.0_genes.csv"
        pheno_file = Path(temp_csv_directory) / "GSEs.test_DS-1_v2.0_pheno.csv"
        sample_genes_df.to_csv(genes_file)
        sample_pheno_df.to_csv(pheno_file)

        result = load_gse_data_from_csv_directory(temp_csv_directory)
        assert len(result) == 1

    def test_dataset_with_numeric_sample_names(self, temp_csv_directory):
        """Test handling of purely numeric sample names."""
        genes_df = pd.DataFrame(
            np.random.randn(10, 5),
            index=[f"Gene_{i}" for i in range(10)],
            columns=[1, 2, 3, 4, 5],  # Numeric column names
        )
        pheno_df = pd.DataFrame({"attr": [0, 1, 0, 1, 0]}, index=[1, 2, 3, 4, 5])  # Numeric index

        genes_file = Path(temp_csv_directory) / "numeric_genes.csv"
        pheno_file = Path(temp_csv_directory) / "numeric_pheno.csv"
        genes_df.to_csv(genes_file)
        pheno_df.to_csv(pheno_file)

        file_info = {"genes": genes_file, "pheno": pheno_file}
        result = _load_single_dataset("NumericDataset", file_info)

        assert result["genes"].shape[1] == result["pheno"].shape[0]


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "--cov=coconut.coconut_runner",
            "--cov-report=term-missing",
            "--cov-report=html",
        ]
    )
