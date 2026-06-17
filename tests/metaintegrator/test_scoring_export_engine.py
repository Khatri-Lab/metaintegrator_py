"""
Comprehensive unit tests for scoring_export_engine.py module.
Tests cover all classes, methods, and functionality with edge cases.
"""

import json
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# Import the module under test and dependencies
from metaintegrator.scoring_export_engine import (
    ExportConfig,
    ExportEngine,
    ReportExporter,
    PlotExporter,
)

from metaintegrator.scoring_core import (
    ExportFormat,
    StatisticalSummary,
)

# ==================================================================
# TEST FIXTURES
# ==================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "gene_id": ["GENE1", "GENE2", "GENE3", "GENE4"],
            "effect_size": [1.5, -0.8, 2.1, 0.3],
            "p_value": [0.001, 0.05, 0.0001, 0.1],
            "fdr": [0.01, 0.1, 0.001, 0.2],
            "score": [0.85, 0.62, 0.95, 0.45],
        }
    )


@pytest.fixture
def sample_statistical_summary():
    """Create a sample StatisticalSummary for testing."""
    return StatisticalSummary(
        basic_stats={
            "effect_size": {
                "count": 100,
                "mean": 0.5,
                "std": 1.2,
                "min": -2.0,
                "max": 3.0,
                "median": 0.4,
                "iqr": 1.5,
                "missing_count": 5,
                "missing_percent": 5.0,
            },
            "p_value": {
                "count": 100,
                "mean": 0.05,
                "std": 0.02,
                "min": 0.0001,
                "max": 0.1,
                "median": 0.045,
                "iqr": 0.03,
                "missing_count": 2,
                "missing_percent": 2.0,
            },
        },
        distribution_stats={
            "effect_size": {
                "shapiro_p": 0.12,
                "skewness": 0.3,
                "kurtosis": 2.8,
                "outlier_count": 3,
                "outlier_percent": 3.0,
            }
        },
        correlation_stats={
            "pearson_matrix": {
                "effect_size": {"effect_size": 1.0, "p_value": -0.6},
                "p_value": {"effect_size": -0.6, "p_value": 1.0},
            },
            "correlation_summary": {
                "mean_abs_correlation": 0.3,
                "max_abs_correlation": 0.6,
                "strong_correlations_count": 1,
            },
            "high_correlations": [
                {
                    "variable1": "effect_size",
                    "variable2": "p_value",
                    "correlation": -0.6,
                }
            ],
        },
        threshold_counts={"significance": {"0.05": 45, "0.01": 20, "0.001": 5}},
        metadata={"analysis_date": "2024-01-15", "version": "1.0"},
    )


@pytest.fixture
def sample_filter_results():
    """Create a sample filter results object for testing."""

    class MockFilterResults:
        def __init__(self):
            self.pos_gene_names = ["GENE1", "GENE3", "GENE5"]
            self.neg_gene_names = ["GENE2", "GENE4"]
            self.filter_type = "effect_size"
            self.threshold = 0.5

    return MockFilterResults()


# ==================================================================
# TEST EXPORT CONFIG
# ==================================================================


class TestExportConfig:
    """Test ExportConfig class."""

    def test_default_export_config(self, temp_dir):
        """Test ExportConfig with default values."""
        config = ExportConfig(base_output_dir=temp_dir)

        assert config.base_output_dir == temp_dir
        assert config.include_metadata == True
        assert config.include_plots == True
        assert config.include_summary == True
        assert config.compress_outputs == False
        assert config.validate_exports == True
        assert config.timestamp_files == True
        assert config.custom_prefix is None
        assert temp_dir.exists()

    def test_custom_export_config(self, temp_dir):
        """Test ExportConfig with custom values."""
        config = ExportConfig(
            base_output_dir=temp_dir,
            include_metadata=False,
            include_plots=False,
            include_summary=False,
            compress_outputs=True,
            validate_exports=False,
            timestamp_files=False,
            custom_prefix="test_prefix",
        )

        assert config.base_output_dir == temp_dir
        assert config.include_metadata == False
        assert config.include_plots == False
        assert config.include_summary == False
        assert config.compress_outputs == True
        assert config.validate_exports == False
        assert config.timestamp_files == False
        assert config.custom_prefix == "test_prefix"

    def test_export_config_creates_directory(self, temp_dir):
        """Test that ExportConfig creates the output directory."""
        new_dir = temp_dir / "new_export_dir"
        assert not new_dir.exists()

        config = ExportConfig(base_output_dir=new_dir)
        assert new_dir.exists()

    def test_export_config_string_path(self, temp_dir):
        """Test ExportConfig accepts string paths."""
        config = ExportConfig(base_output_dir=str(temp_dir))
        assert config.base_output_dir == temp_dir


# ==================================================================
# TEST EXPORT ENGINE
# ==================================================================


class TestExportEngine:
    """Test ExportEngine class."""

    def test_export_engine_initialization(self, temp_dir):
        """Test ExportEngine initialization."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        assert engine.config == config
        assert engine.export_registry == {}
        assert hasattr(engine, "logger")

    def test_export_engine_default_config(self, temp_dir):
        """Test ExportEngine with default config."""
        with patch("metaintegrator.scoring_export_engine.ExportConfig") as mock_config:
            mock_config.return_value = ExportConfig(base_output_dir=temp_dir)
            engine = ExportEngine()
            assert engine.config is not None

    # Test dataframe export methods

    def test_export_dataframe_csv(self, temp_dir, sample_dataframe):
        """Test DataFrame export to CSV format."""
        config = ExportConfig(base_output_dir=temp_dir, timestamp_files=False)
        engine = ExportEngine(config=config)

        result_path = engine._export_dataframe(
            sample_dataframe, "test_data", ExportFormat.CSV
        )

        assert result_path.exists()
        assert result_path.suffix == ".csv"

        # Verify content
        reloaded = pd.read_csv(result_path)
        pd.testing.assert_frame_equal(reloaded, sample_dataframe)

    def test_export_dataframe_excel(self, temp_dir, sample_dataframe):
        """Test DataFrame export to Excel format."""
        config = ExportConfig(base_output_dir=temp_dir, timestamp_files=False)
        engine = ExportEngine(config=config)

        result_path = engine._export_dataframe(
            sample_dataframe, "test_data", ExportFormat.EXCEL
        )

        assert result_path.exists()
        assert result_path.suffix == ".xlsx"

        # Verify content can be read back
        reloaded = pd.read_excel(result_path, sheet_name="Data")
        pd.testing.assert_frame_equal(reloaded, sample_dataframe)

    def test_export_dataframe_tsv(self, temp_dir, sample_dataframe):
        """Test DataFrame export to TSV format."""
        config = ExportConfig(base_output_dir=temp_dir, timestamp_files=False)
        engine = ExportEngine(config=config)

        result_path = engine._export_dataframe(
            sample_dataframe, "test_data", ExportFormat.TSV
        )

        assert result_path.exists()
        assert result_path.suffix == ".tsv"

    def test_export_dataframe_parquet(self, temp_dir, sample_dataframe):
        """Test DataFrame export to Parquet format."""
        config = ExportConfig(base_output_dir=temp_dir, timestamp_files=False)
        engine = ExportEngine(config=config)

        result_path = engine._export_dataframe(
            sample_dataframe, "test_data", ExportFormat.PARQUET
        )

        assert result_path.exists()
        assert result_path.suffix == ".parquet"

    def test_export_dataframe_pickle(self, temp_dir, sample_dataframe):
        """Test DataFrame export to Pickle format."""
        config = ExportConfig(base_output_dir=temp_dir, timestamp_files=False)
        engine = ExportEngine(config=config)

        result_path = engine._export_dataframe(
            sample_dataframe, "test_data", ExportFormat.PICKLE
        )

        assert result_path.exists()
        assert result_path.suffix == ".pkl"

    def test_export_dataframe_json_format(self, temp_dir, sample_dataframe):
        """Test DataFrame export to JSON format."""
        config = ExportConfig(base_output_dir=temp_dir, timestamp_files=False)
        engine = ExportEngine(config=config)

        result_path = engine._export_dataframe(
            sample_dataframe, "test_data", ExportFormat.JSON
        )

        assert result_path.exists()
        assert result_path.suffix == ".json"

    def test_export_dataframe_invalid_format(self, temp_dir, sample_dataframe):
        """Test DataFrame export with invalid format."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        # Create a mock invalid format
        class InvalidFormat:
            value = "invalid_format"

        with pytest.raises(ValueError, match="Unsupported format"):
            engine._export_dataframe(sample_dataframe, "test_data", InvalidFormat())

    def test_export_json(self, temp_dir):
        """Test JSON data export."""
        config = ExportConfig(base_output_dir=temp_dir, timestamp_files=False)
        engine = ExportEngine(config=config)

        test_data = {
            "analysis": "test",
            "results": [1, 2, 3],
            "metadata": {"version": "1.0"},
        }

        result_path = engine._export_json(test_data, "test_json")

        assert result_path.exists()
        assert result_path.suffix == ".json"

        # Verify content
        with open(result_path, "r") as f:
            reloaded = json.load(f)
        assert reloaded == test_data

    def test_export_json_with_numpy_types(self, temp_dir):
        """Test JSON export with numpy data types."""
        config = ExportConfig(base_output_dir=temp_dir, timestamp_files=False)
        engine = ExportEngine(config=config)

        test_data = {
            "numpy_int": np.int64(42),
            "numpy_float": np.float64(3.14),
            "numpy_array": np.array([1, 2, 3]),
            "pandas_na": pd.NA,
        }

        result_path = engine._export_json(test_data, "test_numpy_json")

        assert result_path.exists()

        # Verify content loads without error
        with open(result_path, "r") as f:
            reloaded = json.load(f)
        assert reloaded["numpy_int"] == 42
        assert reloaded["numpy_float"] == 3.14
        assert reloaded["numpy_array"] == [1, 2, 3]

    # Test analysis results export

    def test_export_analysis_results(self, temp_dir, sample_dataframe):
        """Test complete analysis results export."""
        config = ExportConfig(base_output_dir=temp_dir, include_metadata=True)
        engine = ExportEngine(config=config)

        exported_files = engine.export_analysis_results(
            results=sample_dataframe,
            analysis_name="test_analysis",
            formats=[ExportFormat.CSV, ExportFormat.EXCEL],
        )

        assert "csv" in exported_files
        assert "xlsx" in exported_files
        assert "metadata" in exported_files

        # Verify files exist
        for file_path in exported_files.values():
            assert file_path.exists()

        # Check registry
        assert "test_analysis" in engine.export_registry
        assert "timestamp" in engine.export_registry["test_analysis"]
        assert "files" in engine.export_registry["test_analysis"]

    def test_export_analysis_results_with_additional_data(
        self, temp_dir, sample_dataframe
    ):
        """Test analysis results export with additional data."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        additional_df = pd.DataFrame({"extra_col": [1, 2, 3]})
        additional_data = {
            "extra_dataframe": additional_df,
            "extra_dict": {"key": "value"},
        }

        exported_files = engine.export_analysis_results(
            results=sample_dataframe,
            analysis_name="test_with_extra",
            additional_data=additional_data,
        )

        assert "extra_dataframe_csv" in exported_files
        assert "extra_dict_json" in exported_files

    def test_export_analysis_results_failure(self, temp_dir, sample_dataframe):
        """Test analysis results export with failures."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        # Mock _export_dataframe to raise exception
        with patch.object(
            engine, "_export_dataframe", side_effect=Exception("Export failed")
        ):
            exported_files = engine.export_analysis_results(
                results=sample_dataframe, analysis_name="test_failure"
            )

            # Should handle errors gracefully
            assert isinstance(exported_files, dict)

    # Test statistical summary export

    def test_export_statistical_summary(self, temp_dir, sample_statistical_summary):
        """Test statistical summary export."""
        config = ExportConfig(base_output_dir=temp_dir, include_plots=False)
        engine = ExportEngine(config=config)

        exported_files = engine.export_statistical_summary(
            summary=sample_statistical_summary,
            analysis_name="test_stats",
            generate_report=True,
        )

        assert "summary_json" in exported_files
        assert "basic_stats_csv" in exported_files
        assert "thresholds_csv" in exported_files
        assert "report" in exported_files

        # Verify files exist
        for file_path in exported_files.values():
            assert file_path.exists()

    def test_export_statistical_summary_with_plots(
        self, temp_dir, sample_statistical_summary
    ):
        """Test statistical summary export with plots enabled."""
        config = ExportConfig(base_output_dir=temp_dir, include_plots=True)
        engine = ExportEngine(config=config)

        with patch.object(
            engine,
            "_generate_statistical_plots",
            return_value={"plot1": Path("plot.png")},
        ):
            exported_files = engine.export_statistical_summary(
                summary=sample_statistical_summary,
                analysis_name="test_stats_plots",
                generate_report=False,
            )

            assert "plot1" in exported_files

    # Test filter results export

    def test_export_filter_results(self, temp_dir, sample_filter_results):
        """Test filter results export."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        exported_files = engine.export_filter_results(
            filter_results=sample_filter_results,
            filter_name="test_filter",
            include_genes=True,
        )

        assert "filter_json" in exported_files
        assert "genes_json" in exported_files
        assert "genes_csv" in exported_files

        # Verify files exist
        for file_path in exported_files.values():
            assert file_path.exists()

    def test_export_filter_results_no_genes(self, temp_dir, sample_filter_results):
        """Test filter results export without gene lists."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        exported_files = engine.export_filter_results(
            filter_results=sample_filter_results,
            filter_name="test_filter",
            include_genes=False,
        )

        assert "filter_json" in exported_files
        assert "genes_json" not in exported_files
        assert "genes_csv" not in exported_files

    def test_export_filter_results_dict(self, temp_dir):
        """Test filter results export with dictionary input."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        filter_dict = {
            "pos_gene_names": ["GENE1", "GENE2"],
            "neg_gene_names": ["GENE3"],
            "threshold": 0.05,
        }

        exported_files = engine.export_filter_results(
            filter_results=filter_dict, filter_name="test_dict_filter"
        )

        assert "filter_json" in exported_files

    # Test metadata export

    def test_export_metadata(self, temp_dir, sample_dataframe):
        """Test metadata export."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        exported_files = {"csv": Path("test.csv")}

        metadata_path = engine._export_metadata(
            df=sample_dataframe,
            analysis_name="test_analysis",
            exported_files=exported_files,
        )

        assert metadata_path.exists()
        assert metadata_path.suffix == ".json"

        # Verify metadata content
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["analysis_name"] == "test_analysis"
        assert "export_timestamp" in metadata
        assert metadata["data_shape"] == [4, 5]  # sample_dataframe shape
        assert "column_names" in metadata
        assert "exported_files" in metadata

    # Test plot generation

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.figure")
    @patch("seaborn.heatmap")
    def test_generate_statistical_plots(
        self,
        mock_heatmap,
        mock_figure,
        mock_savefig,
        temp_dir,
        sample_statistical_summary,
    ):
        """Test statistical plot generation."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        plot_paths = engine._generate_statistical_plots(
            summary=sample_statistical_summary, analysis_name="test_plots"
        )

        assert "overview" in plot_paths
        assert "correlation" in plot_paths

    def test_generate_statistical_plots_empty_stats(self, temp_dir):
        """Test statistical plot generation with empty stats."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        empty_summary = StatisticalSummary(basic_stats={})

        plot_paths = engine._generate_statistical_plots(
            summary=empty_summary, analysis_name="empty_plots"
        )

        assert plot_paths == {}

    # Test report generation

    def test_generate_statistical_report(self, temp_dir, sample_statistical_summary):
        """Test statistical report generation."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        report_path = engine._generate_statistical_report(
            summary=sample_statistical_summary, analysis_name="test_report"
        )

        assert report_path.exists()
        assert report_path.suffix == ".md"

        # Verify report content
        with open(report_path, "r") as f:
            content = f.read()

        assert "test_report" in content
        assert "Statistical Analysis Report" in content
        assert "Basic Statistics" in content

    # Test filename generation

    def test_generate_filename_default(self, temp_dir):
        """Test filename generation with default settings."""
        config = ExportConfig(base_output_dir=temp_dir, timestamp_files=True)
        engine = ExportEngine(config=config)

        filename = engine._generate_filename("test_file", "csv")

        assert filename.startswith("test_file_")
        assert filename.endswith(".csv")
        # Should contain timestamp
        assert len(filename.split("_")) >= 3

    def test_generate_filename_no_timestamp(self, temp_dir):
        """Test filename generation without timestamp."""
        config = ExportConfig(base_output_dir=temp_dir, timestamp_files=False)
        engine = ExportEngine(config=config)

        filename = engine._generate_filename("test_file", "csv")

        assert filename == "test_file.csv"

    def test_generate_filename_custom_prefix(self, temp_dir):
        """Test filename generation with custom prefix."""
        config = ExportConfig(
            base_output_dir=temp_dir, timestamp_files=False, custom_prefix="custom"
        )
        engine = ExportEngine(config=config)

        filename = engine._generate_filename("test_file", "csv")

        assert filename == "custom_test_file.csv"

    def test_generate_filename_sanitization(self, temp_dir):
        """Test filename sanitization."""
        config = ExportConfig(base_output_dir=temp_dir, timestamp_files=False)
        engine = ExportEngine(config=config)

        filename = engine._generate_filename("test file/with\\spaces", "csv")

        assert filename == "test_file_with_spaces.csv"

    # Test validation

    def test_validate_export_csv(self, temp_dir, sample_dataframe):
        """Test export validation for CSV format."""
        config = ExportConfig(base_output_dir=temp_dir, validate_exports=True)
        engine = ExportEngine(config=config)

        # Create a CSV file
        csv_path = temp_dir / "test.csv"
        sample_dataframe.to_csv(csv_path, index=False)

        is_valid = engine._validate_export(csv_path, sample_dataframe, ExportFormat.CSV)

        assert is_valid == True

    def test_validate_export_excel(self, temp_dir, sample_dataframe):
        """Test export validation for Excel format."""
        config = ExportConfig(base_output_dir=temp_dir, validate_exports=True)
        engine = ExportEngine(config=config)

        # Create an Excel file
        excel_path = temp_dir / "test.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            sample_dataframe.to_excel(writer, sheet_name="Data", index=False)

        is_valid = engine._validate_export(
            excel_path, sample_dataframe, ExportFormat.EXCEL
        )

        assert is_valid == True

    def test_validate_export_shape_mismatch(self, temp_dir, sample_dataframe):
        """Test export validation with shape mismatch."""
        config = ExportConfig(base_output_dir=temp_dir, validate_exports=True)
        engine = ExportEngine(config=config)

        # Create a CSV with different shape
        different_df = sample_dataframe.iloc[:-1]  # Remove last row
        csv_path = temp_dir / "test.csv"
        different_df.to_csv(csv_path, index=False)

        is_valid = engine._validate_export(csv_path, sample_dataframe, ExportFormat.CSV)

        assert is_valid == False

    def test_validate_export_unsupported_format(self, temp_dir, sample_dataframe):
        """Test export validation for unsupported format."""
        config = ExportConfig(base_output_dir=temp_dir, validate_exports=True)
        engine = ExportEngine(config=config)

        # Create a mock unsupported format
        class UnsupportedFormat:
            value = "unsupported"

        fake_path = temp_dir / "test.unsupported"

        is_valid = engine._validate_export(
            fake_path, sample_dataframe, UnsupportedFormat()
        )

        assert is_valid == True  # Should skip validation for unsupported formats

    # Test JSON conversion

    def test_convert_for_json_dict(self, temp_dir):
        """Test JSON conversion for dictionary."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        data = {
            "int": 42,
            "float": 3.14,
            "numpy_int": np.int64(10),
            "numpy_float": np.float64(2.5),
            "numpy_array": np.array([1, 2, 3]),
            "pandas_na": pd.NA,
            "nested": {"key": np.int32(5)},
        }

        converted = engine._convert_for_json(data)

        assert converted["int"] == 42
        assert converted["float"] == 3.14
        assert converted["numpy_int"] == 10
        assert converted["numpy_float"] == 2.5
        assert converted["numpy_array"] == [1, 2, 3]
        assert converted["pandas_na"] is None
        assert converted["nested"]["key"] == 5

    def test_convert_for_json_list(self, temp_dir):
        """Test JSON conversion for list."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        data = [1, np.int64(2), np.array([3, 4]), pd.NA]

        converted = engine._convert_for_json(data)

        assert converted == [1, 2, [3, 4], None]

    # Test archive creation

    def test_create_archive(self, temp_dir):
        """Test ZIP archive creation."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        # Create some test files
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.csv"
        file1.write_text("test content 1")
        file2.write_text("test content 2")

        files = [file1, file2]
        archive_path = engine._create_archive(files, "test_archive")

        assert archive_path.exists()
        assert archive_path.suffix == ".zip"

        # Verify archive contents
        with zipfile.ZipFile(archive_path, "r") as zipf:
            names = zipf.namelist()
            assert "file1.txt" in names
            assert "file2.csv" in names

    def test_create_archive_missing_file(self, temp_dir):
        """Test archive creation with missing files."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        # Include non-existent file
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "missing.txt"
        file1.write_text("test content")

        files = [file1, file2]
        archive_path = engine._create_archive(files, "test_archive")

        assert archive_path.exists()

        # Should only include existing file
        with zipfile.ZipFile(archive_path, "r") as zipf:
            names = zipf.namelist()
            assert "file1.txt" in names
            assert "missing.txt" not in names

    # Test utility methods

    def test_get_export_summary(self, temp_dir, sample_dataframe):
        """Test export summary generation."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        # Perform some exports first
        engine.export_analysis_results(sample_dataframe, "analysis1")
        engine.export_analysis_results(sample_dataframe, "analysis2")

        summary = engine.get_export_summary()

        assert summary["total_analyses"] == 2
        assert "analysis1" in summary["analyses"]
        assert "analysis2" in summary["analyses"]
        assert "engine_config" in summary

    def test_cleanup_exports(self, temp_dir, sample_dataframe):
        """Test cleanup of old export files."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        # Export some data
        exported_files = engine.export_analysis_results(
            sample_dataframe, "old_analysis"
        )

        # Manually set old timestamp
        old_time = datetime.now() - timedelta(days=31)
        engine.export_registry["old_analysis"]["timestamp"] = old_time

        # Create the actual files
        for file_path in exported_files.values():
            file_path.touch()

        removed_count = engine.cleanup_exports(older_than_days=30)

        assert removed_count > 0
        assert "old_analysis" not in engine.export_registry

    def test_cleanup_exports_no_old_files(self, temp_dir, sample_dataframe):
        """Test cleanup when no old files exist."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        # Export recent data
        engine.export_analysis_results(sample_dataframe, "recent_analysis")

        removed_count = engine.cleanup_exports(older_than_days=30)

        assert removed_count == 0
        assert "recent_analysis" in engine.export_registry


# ==================================================================
# TEST SPECIALIZED EXPORTERS
# ==================================================================


class TestReportExporter:
    """Test ReportExporter class."""

    def test_report_exporter_initialization(self):
        """Test ReportExporter initialization."""
        exporter = ReportExporter()
        assert hasattr(exporter, "logger")

    def test_generate_comprehensive_report(self, temp_dir):
        """Test comprehensive report generation."""
        exporter = ReportExporter()

        # Since the method is not implemented, test that it exists
        assert hasattr(exporter, "generate_comprehensive_report")


class TestPlotExporter:
    """Test PlotExporter class."""

    def test_plot_exporter_initialization(self):
        """Test PlotExporter initialization."""
        exporter = PlotExporter()
        assert hasattr(exporter, "logger")

    def test_export_analysis_plots(self, temp_dir):
        """Test analysis plots export."""
        exporter = PlotExporter()

        # Since the method is not implemented, test that it exists
        assert hasattr(exporter, "export_analysis_plots")


# ==================================================================
# TEST ERROR HANDLING
# ==================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_export_dataframe_write_permission_error(self, temp_dir, sample_dataframe):
        """Test DataFrame export with write permission error."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        with patch(
            "pandas.DataFrame.to_csv", side_effect=PermissionError("Access denied")
        ):
            with pytest.raises(PermissionError):
                engine._export_dataframe(sample_dataframe, "test", ExportFormat.CSV)

    def test_export_json_serialization_error(self, temp_dir):
        """Test JSON export with serialization error."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        # Create data that can't be serialized
        class UnserializableClass:
            pass

        data = {"obj": UnserializableClass()}

        # Should handle the error by converting with default=str
        result_path = engine._export_json(data, "test_error")
        assert result_path.exists()

    def test_export_validation_file_not_found(self, temp_dir, sample_dataframe):
        """Test export validation with missing file."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        missing_file = temp_dir / "missing.csv"

        is_valid = engine._validate_export(
            missing_file, sample_dataframe, ExportFormat.CSV
        )

        assert is_valid == False

    def test_plot_generation_error(self, temp_dir, sample_statistical_summary):
        """Test plot generation with matplotlib error."""
        config = ExportConfig(base_output_dir=temp_dir)
        engine = ExportEngine(config=config)

        with patch("matplotlib.pyplot.subplots", side_effect=Exception("Plot error")):
            plot_paths = engine._generate_statistical_plots(
                summary=sample_statistical_summary, analysis_name="error_plots"
            )

            # Should handle errors gracefully
            assert isinstance(plot_paths, dict)


# ==================================================================
# INTEGRATION TESTS
# ==================================================================


class TestIntegration:
    """Integration tests for multiple components."""

    def test_complete_analysis_workflow(
        self, temp_dir, sample_dataframe, sample_statistical_summary
    ):
        """Test complete analysis export workflow."""
        config = ExportConfig(
            base_output_dir=temp_dir,
            include_metadata=True,
            include_plots=True,
            include_summary=True,
        )
        engine = ExportEngine(config=config)

        # Export analysis results
        analysis_files = engine.export_analysis_results(
            results=sample_dataframe,
            analysis_name="complete_analysis",
            formats=[ExportFormat.CSV, ExportFormat.EXCEL, ExportFormat.JSON],
        )

        # Export statistical summary
        with patch.object(engine, "_generate_statistical_plots", return_value={}):
            stats_files = engine.export_statistical_summary(
                summary=sample_statistical_summary, analysis_name="complete_analysis"
            )

        # Verify all exports
        all_files = {**analysis_files, **stats_files}
        for file_path in all_files.values():
            assert file_path.exists()

        # Check export summary
        summary = engine.get_export_summary()
        assert summary["total_analyses"] >= 1

    def test_batch_export_multiple_analyses(self, temp_dir, sample_dataframe):
        """Test batch export of multiple analyses."""
        config = ExportConfig(base_output_dir=temp_dir, include_metadata=False)
        engine = ExportEngine(config=config)

        analyses = {
            "analysis1": sample_dataframe,
            "analysis2": sample_dataframe.copy(),
            "analysis3": sample_dataframe.copy(),
        }

        all_exports = {}
        for name, data in analyses.items():
            exports = engine.export_analysis_results(
                results=data, analysis_name=name, formats=[ExportFormat.CSV]
            )
            all_exports[name] = exports

        # Verify all analyses exported
        assert len(all_exports) == 3
        for exports in all_exports.values():
            assert "csv" in exports
            assert exports["csv"].exists()

        # Check registry
        assert len(engine.export_registry) == 3

    def test_export_with_compression_and_validation(self, temp_dir, sample_dataframe):
        """Test export with compression and validation enabled."""
        config = ExportConfig(
            base_output_dir=temp_dir,
            compress_outputs=True,
            validate_exports=True,
            timestamp_files=True,
        )
        engine = ExportEngine(config=config)

        exported_files = engine.export_analysis_results(
            results=sample_dataframe,
            analysis_name="compressed_validated",
            formats=[ExportFormat.CSV, ExportFormat.PARQUET],
        )

        # Files should exist and be validated
        for file_path in exported_files.values():
            if file_path.suffix not in [".json"]:  # Skip metadata
                assert file_path.exists()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
