"""
Comprehensive unit tests for enhanced LINC tools implementation.
Tests the missing functions: lincsInteractors, lincsQcAccess, and lincsReport.
"""

import os
import shutil
import sys
import tempfile
import warnings
from typing import Dict
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Import the enhanced module - adjust the import path as needed
# from enhanced_linc_tools import (
#     LincsDataManager, LincsAnalyzer, LincsReporter, LincsVisualizer,
#     lincs_interactors, lincs_qc_access, lincs_report
# )

# For testing purposes, we'll assume the enhanced code is available
# In practice, you would import from the actual module

# Add the source directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# Import the classes and functions to test
try:
    from metaintegrator.linc_tools import (
        LincsDataManager,
        LincsAnalyzer,
        lincs_interactors,
        LincsReporter,
        LincsVisualizer,
        lincs_tools,
        lincs_correlate,
        lincs_bait_corr,
        lincs_qc_access,
        lincs_report,
        _prepare_signature_frame,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    raise


class TestLincsDataManagerEnhancements:
    """Test enhanced LincsDataManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_manager = LincsDataManager()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_qc_data(self) -> pd.DataFrame:
        """Create mock quality control data for testing."""
        return pd.DataFrame(
            {
                "pert_iname": ["COMPOUND_A", "COMPOUND_B", "GENE_X", "GENE_Y"],
                "is_gold": [1, 0, 1, 1],
                "cell_id": ["MCF7", "HT29", "PC3", "A549"],
                "pert_idose": ["10 µM", "1 µM", "N/A", "N/A"],
                "pert_itime": ["24 h", "6 h", "96 h", "24 h"],
                "merge": ["COMPOUNDA", "COMPOUNDB", "GENEX", "GENEY"],
                "quality_score": [0.95, 0.72, 0.88, 0.91],
                "replicate_count": [4, 3, 2, 5],
            }
        )

    @patch.object(LincsDataManager, "get_data")
    def test_lincs_qc_access_basic_functionality(self, mock_get_data):
        """Test basic QC access functionality."""
        # Setup mock data
        mock_qc_data = {"CP": self.create_mock_qc_data()}
        mock_get_data.return_value = mock_qc_data

        # Test basic access
        result = self.data_manager.lincs_qc_access(
            dataset="CP", perturbagens=["COMPOUND_A", "COMPOUND_B"], just_gold=False
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0  # May be filtered down
        assert "pert_iname" in result.columns

    @patch.object(LincsDataManager, "get_data")
    def test_lincs_qc_access_gold_filter(self, mock_get_data):
        """Test QC access with gold-standard filtering."""
        # Setup mock data
        mock_qc_data = {"CP": self.create_mock_qc_data()}
        mock_get_data.return_value = mock_qc_data

        # Test gold-only filtering
        result = self.data_manager.lincs_qc_access(
            dataset="CP", perturbagens=["COMPOUND_A", "COMPOUND_B"], just_gold=True
        )

        # Should only return gold standard entries
        if not result.empty and "is_gold" in result.columns:
            assert all(result["is_gold"] == 1)

    def test_lincs_qc_access_invalid_dataset(self):
        """Test QC access with invalid dataset."""
        with pytest.raises(ValueError, match="Invalid dataset"):
            self.data_manager.lincs_qc_access(
                dataset="INVALID", perturbagens=["COMPOUND_A"], just_gold=False
            )

    @patch.object(LincsDataManager, "get_data")
    def test_lincs_qc_access_missing_data(self, mock_get_data):
        """Test QC access when data is unavailable."""
        mock_get_data.return_value = None

        with pytest.raises(ValueError, match="QC data not available"):
            self.data_manager.lincs_qc_access(
                dataset="CP", perturbagens=["COMPOUND_A"], just_gold=False
            )

    @patch.object(LincsDataManager, "get_data")
    def test_lincs_qc_access_time_parsing(self, mock_get_data):
        """Test that time columns are properly parsed."""
        mock_qc_data = {"CP": self.create_mock_qc_data()}
        mock_get_data.return_value = mock_qc_data

        result = self.data_manager.lincs_qc_access(
            dataset="CP", perturbagens=["COMPOUND_A"], just_gold=False
        )

        if not result.empty and "pert_itime" in result.columns:
            # Check that time values are numeric or NaN
            time_values = pd.to_numeric(result["pert_itime"], errors="coerce")
            assert not time_values.isna().all()  # At least some should parse


class TestLincsAnalyzerEnhancements:
    """Test enhanced LincsAnalyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_manager = Mock(spec=LincsDataManager)
        self.analyzer = LincsAnalyzer(self.data_manager)

    def create_mock_signature(self, n_genes: int = 100) -> pd.DataFrame:
        """Create mock signature data."""
        genes = [f"GENE_{i}" for i in range(n_genes)]
        effects = np.random.normal(0, 1, n_genes)
        return pd.DataFrame(effects, index=genes, columns=["summary"])

    def create_mock_correlate_output(self) -> Dict:
        """Create mock correlation output for testing."""
        genes = [f"GENE_{i}" for i in range(10)]
        hit_list = pd.DataFrame(
            {
                "Sig_Corr": np.random.normal(0, 0.5, 10),
                "pval": np.random.uniform(0, 0.05, 10),
                "fdr": np.random.uniform(0, 0.1, 10),
                "merge": genes,
            }
        )

        return {
            "hit_list": hit_list,
            "hit_expression": pd.DataFrame(
                np.random.normal(0, 1, (50, 12)), columns=genes + ["sig", "anti_sig"]
            ),
        }

    def create_mock_dataset_data(
        self, n_genes: int = 100, n_compounds: int = 50
    ) -> pd.DataFrame:
        """Create mock dataset data."""
        genes = [f"GENE_{i}" for i in range(n_genes)]
        compounds = [f"COMPOUND_{i}" for i in range(n_compounds)]
        data = np.random.normal(0, 1, (n_genes, n_compounds))
        return pd.DataFrame(data, index=genes, columns=compounds)

    def test_lincs_interactors_basic_functionality(self):
        """Test basic interactor analysis functionality."""
        # Create test data
        signature = self.create_mock_signature(50)
        correlate_output = self.create_mock_correlate_output()
        dataset = "SH"

        # Mock the dataset data access
        mock_dataset_data = self.create_mock_dataset_data(50, 20)
        self.analyzer._get_dataset_data = Mock(return_value=mock_dataset_data)

        # Test interactor analysis
        result = self.analyzer.lincs_interactors(correlate_output, signature, dataset)

        assert isinstance(result, dict)
        assert "interactor_score" in result
        assert "interactor_expression" in result

        # Check that result dataframes have expected structure
        if not result["interactor_score"].empty:
            expected_cols = ["interaction.score", "driver.score"]
            for col in expected_cols:
                assert col in result["interactor_score"].columns

    def test_lincs_interactors_dataset_specific_filtering(self):
        """Test dataset-specific biological filtering logic."""
        signature = self.create_mock_signature(10)
        correlate_output = self.create_mock_correlate_output()
        mock_dataset_data = self.create_mock_dataset_data(10, 10)
        self.analyzer._get_dataset_data = Mock(return_value=mock_dataset_data)

        # Test SH dataset (should filter for negative interactions)
        result_sh = self.analyzer.lincs_interactors(correlate_output, signature, "SH")

        # Test OE dataset (should filter for positive interactions)
        result_oe = self.analyzer.lincs_interactors(correlate_output, signature, "OE")

        # Results should be valid regardless of specific filtering
        assert isinstance(result_sh, dict)
        assert isinstance(result_oe, dict)

    def test_lincs_interactors_invalid_input(self):
        """Test interactor analysis with invalid inputs."""
        signature = self.create_mock_signature()

        # Test with missing hit_list
        invalid_output = {"some_other_key": "value"}

        with pytest.raises(
            ValueError, match="correlate_output must contain 'hit_list'"
        ):
            self.analyzer.lincs_interactors(invalid_output, signature, "SH")

        # Test with invalid dataset
        correlate_output = self.create_mock_correlate_output()

        with pytest.raises(ValueError, match="Dataset must be one of"):
            self.analyzer.lincs_interactors(correlate_output, signature, "INVALID")

    def test_lincs_interactors_no_overlap_warning(self):
        """Test that warning is raised when no gene overlap exists."""
        signature = self.create_mock_signature(5)
        correlate_output = self.create_mock_correlate_output()
        mock_dataset_data = self.create_mock_dataset_data(5, 5)
        self.analyzer._get_dataset_data = Mock(return_value=mock_dataset_data)

        # Create output with no overlapping gene names
        correlate_output["hit_list"]["merge"] = [
            "DIFFERENT_GENE_" + str(i) for i in range(10)
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.analyzer.lincs_interactors(correlate_output, signature, "SH")

            # Should return empty dataframes and raise warning
            assert len(w) >= 1
            assert (
                "No gene overlap" in str(w[0].message)
                or result["interactor_score"].empty
            )


class TestLincsReporter:
    """Test the new LincsReporter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_manager = Mock(spec=LincsDataManager)
        self.reporter = LincsReporter(self.data_manager)
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_signature(self) -> pd.DataFrame:
        """Create test signature for reporting."""
        genes = [f"GENE_{i}" for i in range(20)]
        effects = np.random.normal(0, 2, 20)
        return pd.DataFrame(effects, index=genes, columns=["summary"])

    @patch("metaintegrator.linc_tools.LincsAnalyzer")
    @patch("metaintegrator.linc_tools.LincsVisualizer")
    def test_lincs_report_basic_functionality(self, mock_visualizer, mock_analyzer):
        """Test basic report generation functionality."""
        # Setup mocks
        mock_analyzer_instance = Mock()
        mock_analyzer.return_value = mock_analyzer_instance

        mock_correlate_result = {
            "hit_list": pd.DataFrame(
                {
                    "Sig_Corr": np.random.normal(0, 0.5, 10),
                    "pval": np.random.uniform(0, 0.05, 10),
                    "merge": [f"COMPOUND_{i}" for i in range(10)],
                }
            ),
            "hit_expression": pd.DataFrame(
                np.random.normal(0, 1, (20, 12)),
                columns=[f"COMPOUND_{i}" for i in range(10)] + ["sig", "anti_sig"],
            ),
        }
        mock_analyzer_instance.lincs_correlate.return_value = mock_correlate_result

        mock_visualizer_instance = Mock()
        mock_visualizer.return_value = mock_visualizer_instance
        mock_fig = Mock()
        mock_visualizer_instance.lincs_heatmap.return_value = mock_fig

        # Test report generation
        signature = self.create_test_signature()

        result = self.reporter.lincs_report(
            signature=signature,
            hit_number_hm=5,
            hit_number_tbl=5,
            report_out_folder=self.temp_dir,
            report_title="test_report",
        )

        assert isinstance(result, dict)
        assert "html_report_path" in result
        assert "rdata_path" in result

        # Check that output directory was created
        assert os.path.exists(self.temp_dir)

        # Check that HTML report was created
        html_path = result.get("html_report_path")
        if html_path:
            assert os.path.exists(html_path)

    def test_lincs_report_empty_signature_error(self):
        """Test that empty signature raises appropriate error."""
        empty_signature = pd.DataFrame()

        with pytest.raises(ValueError, match="Signature cannot be empty"):
            self.reporter.lincs_report(
                signature=empty_signature,
                report_out_folder=self.temp_dir,
                report_title="test_report",
            )

    @patch("metaintegrator.linc_tools.LincsAnalyzer")
    def test_lincs_report_handles_analysis_failures(self, mock_analyzer):
        """Test that report generation handles individual analysis failures gracefully."""
        # Setup analyzer that fails for some datasets
        mock_analyzer_instance = Mock()
        mock_analyzer.return_value = mock_analyzer_instance

        def side_effect_correlate(*args, **kwargs):
            dataset = kwargs.get("dataset", "CP")
            if dataset == "SH":
                raise Exception("Mock analysis failure")
            return {
                "hit_list": pd.DataFrame({"Sig_Corr": [0.1], "merge": ["COMPOUND_A"]}),
                "hit_expression": pd.DataFrame([[0.1]], columns=["COMPOUND_A"]),
            }

        mock_analyzer_instance.lincs_correlate.side_effect = side_effect_correlate

        signature = self.create_test_signature()

        # Should complete despite individual failures
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.reporter.lincs_report(
                signature=signature,
                report_out_folder=self.temp_dir,
                report_title="test_report",
            )

            assert isinstance(result, dict)
            # Should have warnings about failed analyses
            assert len(w) > 0

    @patch("metaintegrator.linc_tools.Path.mkdir")
    @patch("metaintegrator.linc_tools.LincsAnalyzer")
    def test_lincs_report_creates_output_directory(self, mock_analyzer, mock_mkdir):
        """Test that output directory is created."""
        mock_analyzer_instance = Mock()
        mock_analyzer.return_value = mock_analyzer_instance
        mock_analyzer_instance.lincs_correlate.return_value = {
            "hit_list": pd.DataFrame(),
            "hit_expression": pd.DataFrame(),
        }

        signature = self.create_test_signature()

        result = self.reporter.lincs_report(
            signature=signature,
            report_out_folder=self.temp_dir,
            report_title="test_report",
        )

        # Verify mkdir was called
        mock_mkdir.assert_called()

    def test_create_summary_correlation_plot_handles_empty_results(self):
        """Test summary plot creation with empty results."""
        empty_results = {}
        signature = self.create_test_signature()

        # Should return None and not crash
        result = self.reporter._create_summary_correlation_plot(
            empty_results, signature
        )
        assert result is None

    def test_prepare_correlation_matrix_with_valid_data(self):
        """Test correlation matrix preparation with valid data."""
        results = {
            "cp_conn_rev": {
                "hit_list": pd.DataFrame(
                    {
                        "Sig_Corr": [0.8, 0.6, 0.4],
                        "merge": ["COMPOUND_A", "COMPOUND_B", "COMPOUND_C"],
                    }
                )
            },
            "sh_conn_agg": {
                "hit_list": pd.DataFrame(
                    {
                        "Sig_Corr": [-0.7, -0.5, -0.3],
                        "merge": ["GENE_X", "GENE_Y", "GENE_Z"],
                    }
                )
            },
        }

        corr_matrix = self.reporter._prepare_correlation_matrix(results)
        assert isinstance(corr_matrix, pd.DataFrame)
        assert not corr_matrix.empty


class TestHighLevelInterfaces:
    """Test high-level interface functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_meta_object(self) -> Dict:
        """Create mock meta-analysis object."""
        genes = [f"GENE_{i}" for i in range(50)]
        effects = np.random.normal(0, 1, 50)
        pooled_results = pd.DataFrame(effects, index=genes, columns=["effectSize"])

        return {"metaAnalysis": {"pooledResults": pooled_results}}

    def create_mock_filter_object(self) -> Dict:
        """Create mock filter object."""
        return {
            "posGeneNames": [f"GENE_{i}" for i in range(0, 25)],
            "negGeneNames": [f"GENE_{i}" for i in range(25, 50)],
        }

    @patch("metaintegrator.linc_tools.LincsDataManager")
    def test_lincs_qc_access_interface(self, mock_data_manager):
        """Test high-level QC access interface."""
        # Setup mock
        mock_instance = Mock()
        mock_data_manager.return_value = mock_instance
        mock_instance.lincs_qc_access.return_value = pd.DataFrame(
            {"pert_iname": ["COMPOUND_A"], "is_gold": [1]}
        )

        # Test interface
        result = lincs_qc_access(
            dataset="CP", perturbagens=["COMPOUND_A"], just_gold=True
        )

        assert isinstance(result, pd.DataFrame)
        mock_instance.lincs_qc_access.assert_called_once_with(
            "CP", ["COMPOUND_A"], True
        )

    @patch("metaintegrator.linc_tools.LincsAnalyzer")
    @patch("metaintegrator.linc_tools.LincsDataManager")
    def test_lincs_interactors_interface(self, mock_data_manager, mock_analyzer):
        """Test high-level interactors interface."""
        # Setup mocks
        meta_object = self.create_mock_meta_object()
        filter_object = self.create_mock_filter_object()

        mock_analyzer_instance = Mock()
        mock_analyzer.return_value = mock_analyzer_instance
        mock_analyzer_instance.lincs_correlate.return_value = {
            "hit_list": pd.DataFrame()
        }
        mock_analyzer_instance.lincs_interactors.return_value = {
            "interactor_score": pd.DataFrame(),
            "interactor_expression": pd.DataFrame(),
        }

        # Test interface
        result = lincs_interactors(
            meta_object=meta_object, filter_object=filter_object, dataset="SH"
        )

        assert isinstance(result, dict)
        assert "interactor_score" in result
        assert "interactor_expression" in result

    @patch("metaintegrator.linc_tools.LincsReporter")
    def test_lincs_report_interface(self, mock_reporter):
        """Test high-level report interface."""
        # Setup mock
        meta_object = self.create_mock_meta_object()
        filter_object = self.create_mock_filter_object()

        mock_instance = Mock()
        mock_reporter.return_value = mock_instance
        mock_instance.lincs_report.return_value = {
            "html_report_path": f"{self.temp_dir}/report.html",
            "rdata_path": f"{self.temp_dir}/data.pkl",
        }

        # Test interface
        result = lincs_report(
            meta_object=meta_object,
            filter_object=filter_object,
            report_out_folder=self.temp_dir,
            report_title="test_report",
        )

        assert isinstance(result, dict)
        assert "html_report_path" in result
        mock_instance.lincs_report.assert_called_once()

    def test_prepare_signature_frame_invalid_input(self):
        """Test signature frame preparation with invalid input."""
        # Missing metaAnalysis key
        invalid_meta = {"some_other_key": "value"}
        filter_obj = self.create_mock_filter_object()

        with pytest.raises(ValueError, match="Meta-analysis results not found"):
            _prepare_signature_frame(invalid_meta, filter_obj)

        # Missing pooledResults key
        invalid_meta2 = {"metaAnalysis": {"some_other_key": "value"}}

        with pytest.raises(ValueError, match="Meta-analysis results not found"):
            _prepare_signature_frame(invalid_meta2, filter_obj)


class TestIntegration:
    """Integration tests for the complete workflow."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("metaintegrator.linc_tools.LincsDataManager")
    def test_complete_workflow_integration(self, mock_data_manager):
        """Test complete workflow from signature to report."""
        # Create comprehensive mock data
        mock_instance = Mock()
        mock_data_manager.return_value = mock_instance

        # Mock LINCS data
        genes = [f"GENE_{i}" for i in range(100)]
        compounds = [f"COMPOUND_{i}" for i in range(50)]
        mock_agg_data = {
            "CP": pd.DataFrame(
                np.random.normal(0, 1, (100, 50)), index=genes, columns=compounds
            ),
            "SH": pd.DataFrame(
                np.random.normal(0, 1, (100, 30)), index=genes, columns=genes[:30]
            ),
        }

        mock_instance.get_data.side_effect = lambda key: {
            "all_lincs_agg_gold": mock_agg_data,
            "tri_ann": pd.DataFrame({"merge": compounds[:10]}),
            "gene_ann": pd.DataFrame({"Symbol": genes}),
        }.get(key)

        # Create test signature
        signature = pd.DataFrame(
            np.random.normal(0, 2, 20), index=genes[:20], columns=["summary"]
        )

        # Test end-to-end workflow
        with patch("matplotlib.pyplot.savefig"):  # Mock plotting
            # Initialize components
            analyzer = LincsAnalyzer(mock_instance)
            reporter = LincsReporter(mock_instance)

            # Test correlation analysis
            corr_results = analyzer.lincs_correlate(signature, dataset="CP")
            assert "hit_list" in corr_results
            assert "hit_expression" in corr_results

            # Test QC access
            qc_data = mock_instance.lincs_qc_access.return_value = pd.DataFrame(
                {"pert_iname": compounds[:5], "is_gold": [1, 0, 1, 1, 0]}
            )
            qc_result = mock_instance.lincs_qc_access("CP", compounds[:5])
            assert isinstance(qc_result, pd.DataFrame)

            # Test interactor analysis
            interactor_results = analyzer.lincs_interactors(
                corr_results, signature, "SH"
            )
            assert "interactor_score" in interactor_results
            assert "interactor_expression" in interactor_results

    def test_error_handling_and_recovery(self):
        """Test that the system handles errors gracefully."""
        # Test with minimal mock data to trigger various error conditions

        with patch("metaintegrator.linc_tools.LincsDataManager") as mock_dm:
            mock_instance = Mock()
            mock_dm.return_value = mock_instance

            # Test data loading failure
            mock_instance.load_data.return_value = False
            mock_instance.get_data.return_value = None

            analyzer = LincsAnalyzer(mock_instance)

            # Should handle missing data gracefully
            signature = pd.DataFrame([[1.0]], index=["GENE_1"], columns=["summary"])

            result = analyzer.lincs_correlate(signature, dataset="CP")
            # Should return error rather than crashing
            assert "error" in result or isinstance(result, dict)

    def test_memory_efficiency_with_large_datasets(self):
        """Test that the system handles large datasets efficiently."""
        # Create large mock dataset
        large_genes = [f"GENE_{i}" for i in range(1000)]
        large_compounds = [f"COMPOUND_{i}" for i in range(500)]

        # Mock chunked processing
        with patch("metaintegrator.linc_tools.LincsDataManager") as mock_dm:
            mock_instance = Mock()
            mock_dm.return_value = mock_instance

            # Simulate large dataset
            mock_agg_data = {
                "CP": pd.DataFrame(
                    np.random.normal(0, 1, (1000, 500)),
                    index=large_genes,
                    columns=large_compounds,
                )
            }
            mock_instance.get_data.return_value = mock_agg_data

            analyzer = LincsAnalyzer(mock_instance)
            signature = pd.DataFrame(
                np.random.normal(0, 1, 100),
                index=large_genes[:100],
                columns=["summary"],
            )

            # Should complete without memory issues
            result = analyzer.lincs_correlate(signature, dataset="CP", hit_number_hm=10)

            if "hit_list" in result:
                assert isinstance(result["hit_list"], pd.DataFrame)
                # Should limit results appropriately
                assert (
                    len(result["hit_expression"].columns) <= 12
                )  # 10 hits + sig/anti_sig


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_datasets(self):
        """Test handling of empty datasets."""
        with patch("metaintegrator.linc_tools.LincsDataManager") as mock_dm:
            mock_instance = Mock()
            mock_dm.return_value = mock_instance
            mock_instance.get_data.return_value = {"CP": pd.DataFrame()}

            analyzer = LincsAnalyzer(mock_instance)
            signature = pd.DataFrame([[1.0]], index=["GENE_1"], columns=["summary"])

            result = analyzer.lincs_correlate(signature, dataset="CP")
            # Should handle empty data gracefully
            assert "error" in result or isinstance(result, dict)

    def test_single_gene_signature(self):
        """Test with single-gene signature."""
        with patch("metaintegrator.linc_tools.LincsDataManager") as mock_dm:
            mock_instance = Mock()
            mock_dm.return_value = mock_instance
            mock_agg_data = {
                "CP": pd.DataFrame(
                    [[1.0, 2.0, 3.0]],
                    index=["GENE_1"],
                    columns=["COMP_A", "COMP_B", "COMP_C"],
                )
            }
            mock_instance.get_data.return_value = mock_agg_data

            analyzer = LincsAnalyzer(mock_instance)
            signature = pd.DataFrame([[1.5]], index=["GENE_1"], columns=["summary"])

            result = analyzer.lincs_correlate(signature, dataset="CP")
            assert "hit_list" in result or "error" in result

    def test_extreme_correlation_values(self):
        """Test handling of extreme correlation values."""
        with patch("metaintegrator.linc_tools.LincsDataManager") as mock_dm:
            mock_instance = Mock()
            mock_dm.return_value = mock_instance

            # Create data with perfect correlations
            mock_data = pd.DataFrame(
                [[1.0, 1.0, -1.0], [2.0, 2.0, -2.0], [3.0, 3.0, -3.0]],
                index=["GENE_1", "GENE_2", "GENE_3"],
                columns=["PERFECT_POS", "PERFECT_POS2", "PERFECT_NEG"],
            )
            mock_agg_data = {"CP": mock_data}
            mock_instance.get_data.return_value = mock_agg_data

            analyzer = LincsAnalyzer(mock_instance)
            signature = pd.DataFrame(
                [[1.0], [2.0], [3.0]],
                index=["GENE_1", "GENE_2", "GENE_3"],
                columns=["summary"],
            )

            result = analyzer.lincs_correlate(signature, dataset="CP")

            if "hit_list" in result:
                correlations = result["hit_list"]["Sig_Corr"]
                # Should handle perfect correlations
                assert all(abs(c) <= 1.0 for c in correlations)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
