"""
Unit tests for scoring_meta_integrator_orchestrator.py

This test suite provides comprehensive coverage of the meta-integrator orchestrator
with proper mocking of dependencies and testing of all major workflows.
"""

import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Use Mock for missing imports
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# Import exceptions that might be raised
from metaintegrator.scoring_core import (
    ConfigurationError,
    FilteringError,
    ScoringCalculationError,
    ValidationError,
)
from metaintegrator.scoring_meta_integrator_orchestrator import (
    EnhancedMetaIntegratorOrchestrator,
    MetaIntegratorOrchestrator,
    OrchestrationResults,
    OrchestratorConfig,
)

# Create a compatibility alias for any missing classes
try:
    from metaintegrator.scoring_core import StatisticalSummary
except ImportError:
    StatisticalSummary = Mock

try:
    from metaintegrator.scoring_meta_integrator_filter import EnhancedFilterResults
except ImportError:
    EnhancedFilterResults = Mock


# Helper function to create proper context manager mocks
def create_context_manager_mock():
    """Create a mock that properly supports context manager protocol."""
    return MagicMock()


# ==================================================================
# FIXTURES
# ==================================================================


@pytest.fixture
def sample_meta_object():
    """Create a sample meta-analysis object for testing."""
    # Create sample data
    genes = [f"GENE_{i}" for i in range(100)]
    data = pd.DataFrame(
        {
            "effect_size": np.random.normal(0, 1, 100),
            "effect_fdr": np.random.uniform(0, 1, 100),
            "p_value": np.random.uniform(0, 1, 100),
            "fold_change": np.random.lognormal(0, 1, 100),
        },
        index=genes,
    )

    return {
        "original_result": data,
        "datasetNames": ["Dataset1", "Dataset2", "Dataset3"],
        "class_labels": ["Control", "Treatment"],
        "study_count": 3,
    }


@pytest.fixture
def default_config():
    """Create a default orchestrator configuration."""
    return OrchestratorConfig(
        log_level="INFO",
        validate_inputs=True,
        validate_outputs=True,
        auto_export_results=False,  # Disable for testing
        auto_generate_reports=False,
        auto_cleanup=False,
        enable_parallel_processing=False,  # Disable for testing
        fail_fast=False,
    )


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_filter_results():
    """Create mock filter results."""
    # Create a mock object that behaves like EnhancedFilterResults
    mock_results = Mock()

    # Set basic attributes
    mock_results.pos_gene_names = ["GENE_1", "GENE_2", "GENE_3"]
    mock_results.neg_gene_names = ["GENE_4", "GENE_5"]
    mock_results.total_genes = 5
    mock_results.summary_statistics = {
        "total_significant": 5,
        "pos_count": 3,
        "neg_count": 2,
    }
    mock_results.filter_metadata = {"strategy": "standard", "threshold": 0.05}

    # Enhanced attributes
    mock_results.filter_performance = {}
    mock_results.gene_scores = {
        "GENE_1": 2.5,
        "GENE_2": 2.1,
        "GENE_3": 1.8,
        "GENE_4": -1.9,
        "GENE_5": -2.2,
    }
    mock_results.cross_study_consistency = {}
    mock_results.adaptive_thresholds_used = {}
    mock_results.filtering_steps = []

    return mock_results


@pytest.fixture
def mock_gene_scores():
    """Create mock gene scores."""
    genes = ["GENE_1", "GENE_2", "GENE_3", "GENE_4", "GENE_5"]
    scores = [2.5, 2.1, 1.8, -1.9, -2.2]
    return pd.Series(scores, index=genes, name="gene_score")


@pytest.fixture
def mock_statistical_summary():
    """Create mock statistical summary."""
    # Create a mock object that behaves like StatisticalSummary
    mock_summary = Mock()

    # Set the expected attributes
    mock_summary.gene_count = 100
    mock_summary.significant_genes = 25
    mock_summary.mean_effect_size = 0.15
    mock_summary.median_effect_size = 0.12
    mock_summary.effect_size_std = 0.8
    mock_summary.mean_p_value = 0.3
    mock_summary.significant_threshold = 0.05
    mock_summary.summary_metadata = {"analysis_type": "meta_analysis"}

    # For compatibility with dict-like access
    mock_summary.basic_stats = {
        "gene_count": 100,
        "significant_genes": 25,
        "mean_effect_size": 0.15,
    }
    mock_summary.distribution_stats = None
    mock_summary.correlation_stats = None
    mock_summary.threshold_counts = None
    mock_summary.metadata = {"analysis_type": "meta_analysis"}

    return mock_summary


# ==================================================================
# TEST ORCHESTRATOR CONFIG
# ==================================================================


class TestOrchestratorConfig:
    """Test suite for OrchestratorConfig."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = OrchestratorConfig()

        assert config.log_level == "INFO"
        assert config.enable_warnings is True
        assert config.enable_progress_tracking is True
        assert config.validate_inputs is True
        assert config.validate_outputs is True
        assert config.auto_export_results is True
        assert config.fail_fast is False
        assert config.max_workers == 4

    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = OrchestratorConfig(
            log_level="DEBUG", enable_warnings=False, max_workers=8, fail_fast=True
        )

        assert config.log_level == "DEBUG"
        assert config.enable_warnings is False
        assert config.max_workers == 8
        assert config.fail_fast is True

    def test_invalid_log_level_raises_error(self):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            OrchestratorConfig(log_level="INVALID")

    def test_invalid_max_workers_raises_error(self):
        """Test that invalid max_workers raises ValueError."""
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            OrchestratorConfig(max_workers=0)

    def test_path_string_conversion(self, temp_output_dir):
        """Test that string paths are converted to Path objects."""
        config = OrchestratorConfig(
            default_output_dir=str(temp_output_dir),
            cache_directory=str(temp_output_dir / "cache"),
        )

        assert isinstance(config.default_output_dir, Path)
        assert isinstance(config.cache_directory, Path)
        assert config.default_output_dir == temp_output_dir


# ==================================================================
# TEST ORCHESTRATION RESULTS
# ==================================================================


class TestOrchestrationResults:
    """Test suite for OrchestrationResults."""

    def test_basic_results_creation(self):
        """Test creating basic orchestration results."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=30)

        results = OrchestrationResults(
            analysis_name="test_analysis",
            success=True,
            start_time=start_time,
            end_time=end_time,
            total_duration=30.0,
        )

        assert results.analysis_name == "test_analysis"
        assert results.success is True
        assert results.total_duration == 30.0
        assert results.errors == []
        assert results.warnings == []

    def test_success_rate_calculation_all_successful(
        self, mock_filter_results, mock_gene_scores, mock_statistical_summary
    ):
        """Test success rate calculation when all components successful."""
        results = OrchestrationResults(
            analysis_name="test",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=10.0,
            filter_results=mock_filter_results,
            gene_scores=mock_gene_scores,
            statistical_summary=mock_statistical_summary,
            export_paths={"test": "path"},
        )

        assert results.success_rate == 1.0

    def test_success_rate_calculation_partial_success(self, mock_filter_results):
        """Test success rate calculation with partial success."""
        results = OrchestrationResults(
            analysis_name="test",
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=10.0,
            filter_results=mock_filter_results,
            errors=[{"component": "test", "error": "test error"}],
        )

        # 1/4 components successful (filter_results=1, others=0)
        assert results.success_rate == 0.25

    def test_get_summary(self, mock_filter_results, mock_gene_scores):
        """Test getting summary of orchestration results."""
        results = OrchestrationResults(
            analysis_name="test_analysis",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=25.5,
            filter_results=mock_filter_results,
            gene_scores=mock_gene_scores,
        )

        summary = results.get_summary()

        assert summary["analysis_name"] == "test_analysis"
        assert summary["success"] is True
        assert summary["duration_seconds"] == 25.5
        assert summary["components_completed"]["filtering"] is True
        assert summary["components_completed"]["gene_scoring"] is True
        assert summary["total_genes_filtered"] == 5
        assert summary["total_genes_scored"] == 5


# ==================================================================
# TEST ENHANCED META INTEGRATOR ORCHESTRATOR
# ==================================================================


class TestEnhancedMetaIntegratorOrchestrator:
    """Test suite for EnhancedMetaIntegratorOrchestrator."""

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_initialization(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        default_config,
    ):
        """Test orchestrator initialization."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(default_config)

        assert orchestrator.config == default_config
        assert hasattr(orchestrator, "meta_filter")
        assert hasattr(orchestrator, "gene_scorer")
        assert hasattr(orchestrator, "result_summarizer")
        assert hasattr(orchestrator, "export_engine")

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_initialization_with_default_config(
        self, mock_stat_analyzer, mock_export, mock_summarizer, mock_scorer, mock_filter
    ):
        """Test orchestrator initialization with default config."""
        orchestrator = EnhancedMetaIntegratorOrchestrator()

        assert isinstance(orchestrator.config, OrchestratorConfig)
        assert orchestrator.config.log_level == "INFO"

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_component_initialization_failure(
        self, mock_stat_analyzer, mock_export, mock_summarizer, mock_scorer, mock_filter
    ):
        """Test handling of component initialization failure."""
        # Make one component fail to initialize
        mock_filter.side_effect = Exception("Filter initialization failed")

        with pytest.raises(ConfigurationError, match="Failed to initialize components"):
            EnhancedMetaIntegratorOrchestrator()

    @patch(
        "metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorOrchestrator._run_standard_workflow"
    )
    @patch(
        "metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorOrchestrator._validate_and_preprocess_inputs"
    )
    @patch(
        "metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorOrchestrator._assess_input_quality"
    )
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_run_complete_analysis_standard_workflow(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        mock_quality,
        mock_validate,
        mock_workflow,
        sample_meta_object,
        default_config,
    ):
        """Test running complete analysis with standard workflow."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(default_config)

        results = orchestrator.run_complete_analysis(
            sample_meta_object, "test_analysis", "standard"
        )

        assert isinstance(results, OrchestrationResults)
        assert results.analysis_name == "test_analysis"
        mock_validate.assert_called_once()
        mock_quality.assert_called_once()
        mock_workflow.assert_called_once()

    @patch(
        "metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorOrchestrator._run_advanced_workflow"
    )
    @patch(
        "metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorOrchestrator._validate_and_preprocess_inputs"
    )
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_run_complete_analysis_advanced_workflow(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        mock_validate,
        mock_workflow,
        sample_meta_object,
        default_config,
    ):
        """Test running complete analysis with advanced workflow."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(default_config)

        results = orchestrator.run_complete_analysis(
            sample_meta_object, "test_analysis", "advanced"
        )

        assert isinstance(results, OrchestrationResults)
        mock_workflow.assert_called_once()

    @patch(
        "metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorOrchestrator._run_minimal_workflow"
    )
    @patch(
        "metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorOrchestrator._validate_and_preprocess_inputs"
    )
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_run_complete_analysis_minimal_workflow(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        mock_validate,
        mock_workflow,
        sample_meta_object,
        default_config,
    ):
        """Test running complete analysis with minimal workflow."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(default_config)

        results = orchestrator.run_complete_analysis(sample_meta_object, "test_analysis", "minimal")

        assert isinstance(results, OrchestrationResults)
        mock_workflow.assert_called_once()

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_run_complete_analysis_invalid_workflow(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        sample_meta_object,
    ):
        """Test running analysis with invalid workflow type."""
        # Disable validation so we reach the workflow type check
        config = OrchestratorConfig(
            validate_inputs=False,
            validate_outputs=False,
            enable_quality_assessment=False,
        )
        orchestrator = EnhancedMetaIntegratorOrchestrator(config)

        results = orchestrator.run_complete_analysis(
            sample_meta_object, "test_analysis", "invalid_workflow"
        )

        assert results.success is False
        assert len(results.errors) > 0
        assert "Unknown workflow type" in results.errors[0]["error_message"]

    @patch(
        "metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorOrchestrator._generate_batch_summary"
    )
    @patch(
        "metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorOrchestrator.run_complete_analysis"
    )
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_run_batch_analysis_sequential(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        mock_run_complete,
        mock_batch_summary,
        sample_meta_object,
    ):
        """Test batch analysis in sequential mode."""
        # Setup config for sequential processing
        config = OrchestratorConfig(enable_parallel_processing=False)
        orchestrator = EnhancedMetaIntegratorOrchestrator(config)

        # Setup mock return value with proper numeric attributes
        mock_result = Mock()
        mock_result.success = True
        mock_result.total_duration = 10.5  # Numeric value instead of Mock
        mock_result.get_summary.return_value = {"test": "summary"}
        mock_run_complete.return_value = mock_result

        meta_objects = {
            "analysis1": sample_meta_object,
            "analysis2": sample_meta_object,
        }

        results = orchestrator.run_batch_analysis(meta_objects, "test_batch")

        assert len(results) == 2
        assert "analysis1" in results
        assert "analysis2" in results
        assert mock_run_complete.call_count == 2
        mock_batch_summary.assert_called_once()

    @patch(
        "metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorOrchestrator._generate_batch_summary"
    )
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ThreadPoolExecutor")
    @patch(
        "metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorOrchestrator.run_complete_analysis"
    )
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_run_batch_analysis_parallel(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        mock_run_complete,
        mock_executor,
        mock_batch_summary,
        sample_meta_object,
    ):
        """Test batch analysis in parallel mode."""
        # Setup config for parallel processing
        config = OrchestratorConfig(enable_parallel_processing=True, max_workers=2)
        orchestrator = EnhancedMetaIntegratorOrchestrator(config)

        # Create a proper result mock with all needed attributes
        result_mock = Mock()
        result_mock.success = True
        result_mock.total_duration = 5.0
        result_mock.get_summary.return_value = {"test": "summary"}

        # Create unique future objects for each submit call
        def create_mock_future():
            future = Mock()
            future.result.return_value = result_mock
            return future

        # Setup mock executor with proper context manager support
        mock_executor_instance = MagicMock()
        # Make submit return a new future each time (important for dict keys)
        mock_executor_instance.submit.side_effect = lambda *args, **kwargs: create_mock_future()
        mock_executor_instance.__enter__ = Mock(return_value=mock_executor_instance)
        mock_executor_instance.__exit__ = Mock(return_value=False)
        mock_executor.return_value = mock_executor_instance

        # Need at least 2 meta_objects to trigger parallel execution (len > 1)
        meta_objects = {
            "analysis1": sample_meta_object,
            "analysis2": sample_meta_object,
        }

        results = orchestrator.run_batch_analysis(meta_objects, "test_batch")

        assert len(results) == 2
        assert "analysis1" in results
        assert "analysis2" in results
        # Check that the result is the same mock object returned by future.result()
        assert results["analysis1"].success == True
        assert results["analysis1"].total_duration == 5.0
        assert results["analysis2"].success == True
        mock_batch_summary.assert_called_once()


# ==================================================================
# TEST WORKFLOW METHODS
# ==================================================================


class TestWorkflowMethods:
    """Test suite for individual workflow methods."""

    def setup_method(self):
        """Setup for each test method."""
        self.config = OrchestratorConfig(fail_fast=False)

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_run_standard_workflow_success(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        sample_meta_object,
        mock_filter_results,
        mock_gene_scores,
        mock_statistical_summary,
    ):
        """Test successful standard workflow execution."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        # Setup mocks
        orchestrator.meta_filter.filter_meta_analysis.return_value = mock_filter_results
        orchestrator.gene_scorer.score_genes.return_value = mock_gene_scores
        orchestrator.result_summarizer.summarize_results.return_value = {
            "statistical_summary": mock_statistical_summary
        }

        results = OrchestrationResults(
            analysis_name="test",
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
        )

        orchestrator._run_standard_workflow(sample_meta_object, results)

        assert results.filter_results is not None
        assert results.gene_scores is not None
        assert results.statistical_summary is not None
        assert "filtering_duration" in results.performance_metrics
        assert "scoring_duration" in results.performance_metrics
        assert "summarization_duration" in results.performance_metrics

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_run_standard_workflow_filtering_failure(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        sample_meta_object,
    ):
        """Test standard workflow with filtering failure."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        # Make filtering fail
        orchestrator.meta_filter.filter_meta_analysis.side_effect = FilteringError(
            "Filtering failed"
        )

        results = OrchestrationResults(
            analysis_name="test",
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
        )

        orchestrator._run_standard_workflow(sample_meta_object, results)

        assert results.filter_results is None
        assert len(results.errors) > 0
        assert results.errors[0]["component"] == "filtering"

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_run_advanced_workflow_with_fallback(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        sample_meta_object,
        mock_filter_results,
        mock_gene_scores,
    ):
        """Test advanced workflow with adaptive filtering fallback."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        # Make adaptive filtering fail, standard succeed
        def side_effect(meta_obj, strategy):
            if strategy == "adaptive":
                raise Exception("Adaptive filtering failed")
            return mock_filter_results

        orchestrator.meta_filter.filter_meta_analysis.side_effect = side_effect
        orchestrator.gene_scorer.score_genes.return_value = mock_gene_scores

        results = OrchestrationResults(
            analysis_name="test",
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
        )

        orchestrator._run_advanced_workflow(sample_meta_object, results)

        assert results.filter_results is not None
        assert orchestrator.meta_filter.filter_meta_analysis.call_count == 2

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_run_minimal_workflow(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        sample_meta_object,
        mock_filter_results,
        mock_gene_scores,
    ):
        """Test minimal workflow execution."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        # Setup mocks
        orchestrator.meta_filter.filter_meta_analysis.return_value = mock_filter_results
        orchestrator.gene_scorer.score_genes.return_value = mock_gene_scores

        results = OrchestrationResults(
            analysis_name="test",
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
        )

        orchestrator._run_minimal_workflow(sample_meta_object, results)

        assert results.filter_results is not None
        assert results.gene_scores is not None
        # Minimal workflow shouldn't create comprehensive summary
        assert results.comprehensive_summary is None


# ==================================================================
# TEST VALIDATION METHODS
# ==================================================================


class TestValidationMethods:
    """Test suite for validation methods."""

    def setup_method(self):
        """Setup for each test method."""
        self.config = OrchestratorConfig()

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_validate_and_preprocess_inputs_valid(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        sample_meta_object,
    ):
        """Test input validation with valid meta object."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        results = OrchestrationResults(
            analysis_name="test",
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
        )

        orchestrator._validate_and_preprocess_inputs(sample_meta_object, results)

        assert results.validation_results["input_validation"] == "passed"
        assert len(results.errors) == 0

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_validate_and_preprocess_inputs_invalid_type(
        self, mock_stat_analyzer, mock_export, mock_summarizer, mock_scorer, mock_filter
    ):
        """Test input validation with invalid meta object type."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        results = OrchestrationResults(
            analysis_name="test",
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
        )

        # Pass a list instead of dict
        invalid_meta_object = [1, 2, 3]

        orchestrator._validate_and_preprocess_inputs(invalid_meta_object, results)

        assert results.validation_results["input_validation"] == "failed"
        assert len(results.errors) > 0
        assert "input_validation" in results.errors[0]["component"]

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_validate_and_preprocess_inputs_missing_data(
        self, mock_stat_analyzer, mock_export, mock_summarizer, mock_scorer, mock_filter
    ):
        """Test input validation with missing required data."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        results = OrchestrationResults(
            analysis_name="test",
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
        )

        # Meta object without required data
        invalid_meta_object = {"some_other_key": "value"}

        orchestrator._validate_and_preprocess_inputs(invalid_meta_object, results)

        assert results.validation_results["input_validation"] == "failed"
        assert len(results.errors) > 0

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_validate_and_preprocess_inputs_auto_fix(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        sample_meta_object,
    ):
        """Test input validation with auto-fix enabled."""
        config = OrchestratorConfig(auto_fix_common_issues=True)
        orchestrator = EnhancedMetaIntegratorOrchestrator(config)

        results = OrchestrationResults(
            analysis_name="test",
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
        )

        # Remove a column that can be auto-fixed
        data = sample_meta_object["original_result"].copy()
        data["fdr"] = data["effect_fdr"]
        del data["effect_fdr"]
        meta_object_with_missing_col = {"original_result": data}

        orchestrator._validate_and_preprocess_inputs(meta_object_with_missing_col, results)

        assert results.validation_results["input_validation"] == "passed"
        assert "effect_fdr" in meta_object_with_missing_col["original_result"].columns

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_assess_input_quality(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        sample_meta_object,
    ):
        """Test input quality assessment."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        results = OrchestrationResults(
            analysis_name="test",
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
        )

        orchestrator._assess_input_quality(sample_meta_object, results)

        assert "total_genes" in results.quality_assessment
        assert "missing_data_percentage" in results.quality_assessment
        assert "overall_quality_score" in results.quality_assessment
        assert isinstance(results.quality_assessment["overall_quality_score"], float)

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_validate_outputs(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        mock_filter_results,
        mock_gene_scores,
        mock_statistical_summary,
    ):
        """Test output validation."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        results = OrchestrationResults(
            analysis_name="test",
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
            filter_results=mock_filter_results,
            gene_scores=mock_gene_scores,
            statistical_summary=mock_statistical_summary,
        )

        orchestrator._validate_outputs(results)

        assert "filter_results" in results.validation_results
        assert "gene_scores" in results.validation_results
        assert "consistency" in results.validation_results
        assert results.validation_results["filter_results"] == "passed"
        assert results.validation_results["gene_scores"] == "passed"


# ==================================================================
# TEST ERROR HANDLING
# ==================================================================


class TestErrorHandling:
    """Test suite for error handling methods."""

    def setup_method(self):
        """Setup for each test method."""
        self.config = OrchestratorConfig()

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_handle_component_error(
        self, mock_stat_analyzer, mock_export, mock_summarizer, mock_scorer, mock_filter
    ):
        """Test component error handling."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        results = OrchestrationResults(
            analysis_name="test",
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
        )

        test_error = ValueError("Test error message")
        orchestrator._handle_component_error("test_component", test_error, results)

        assert len(results.errors) == 1
        assert results.errors[0]["component"] == "test_component"
        assert results.errors[0]["error_type"] == "ValueError"
        assert results.errors[0]["error_message"] == "Test error message"
        assert "timestamp" in results.errors[0]
        assert "traceback" in results.errors[0]

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_create_error_result(
        self, mock_stat_analyzer, mock_export, mock_summarizer, mock_scorer, mock_filter
    ):
        """Test creating error result for failed analyses."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        test_error = RuntimeError("Test runtime error")
        error_result = orchestrator._create_error_result("failed_analysis", test_error)

        assert error_result.analysis_name == "failed_analysis"
        assert error_result.success is False
        assert len(error_result.errors) == 1
        assert error_result.errors[0]["error_type"] == "RuntimeError"
        assert error_result.errors[0]["error_message"] == "Test runtime error"


# ==================================================================
# TEST UTILITY METHODS
# ==================================================================


class TestUtilityMethods:
    """Test suite for utility methods."""

    def setup_method(self):
        """Setup for each test method."""
        self.config = OrchestratorConfig()

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_get_component_status(
        self, mock_stat_analyzer, mock_export, mock_summarizer, mock_scorer, mock_filter
    ):
        """Test getting component status."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        status = orchestrator.get_component_status()

        assert "meta_filter" in status
        assert "gene_scorer" in status
        assert "result_summarizer" in status
        assert "export_engine" in status
        assert all(s == "initialized" for s in status.values())

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_clear_all_caches(
        self, mock_stat_analyzer, mock_export, mock_summarizer, mock_scorer, mock_filter
    ):
        """Test clearing all caches."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        # Setup mock clear methods
        orchestrator.meta_filter.clear_cache = Mock()
        orchestrator.gene_scorer.clear_cache = Mock()
        orchestrator.result_summarizer.clear_analysis_cache = Mock()

        orchestrator.clear_all_caches()

        orchestrator.meta_filter.clear_cache.assert_called_once()
        orchestrator.gene_scorer.clear_cache.assert_called_once()
        orchestrator.result_summarizer.clear_analysis_cache.assert_called_once()

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_get_component_versions(
        self, mock_stat_analyzer, mock_export, mock_summarizer, mock_scorer, mock_filter
    ):
        """Test getting component versions."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        versions = orchestrator._get_component_versions()

        assert "orchestrator" in versions
        assert "filter" in versions
        assert "scorer" in versions
        assert "summarizer" in versions
        assert "export_engine" in versions
        assert all(isinstance(v, str) for v in versions.values())


# ==================================================================
# TEST LEGACY COMPATIBILITY
# ==================================================================


class TestLegacyCompatibility:
    """Test suite for legacy compatibility class."""

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_legacy_orchestrator_initialization(
        self, mock_stat_analyzer, mock_export, mock_summarizer, mock_scorer, mock_filter
    ):
        """Test legacy orchestrator initialization."""
        legacy_config = {
            "log_level": "DEBUG",
            "enable_warnings": False,
            "cache_components": True,
            "validate_inputs": True,
            "default_output_dir": ".",
            "auto_cleanup": False,
        }

        orchestrator = MetaIntegratorOrchestrator(legacy_config)

        assert isinstance(orchestrator, EnhancedMetaIntegratorOrchestrator)
        assert orchestrator.config.log_level == "DEBUG"
        assert orchestrator.config.enable_warnings is False
        assert orchestrator.config.cache_components is True

    @patch(
        "metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorOrchestrator.run_complete_analysis"
    )
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_legacy_run_analysis_method(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        mock_run_complete,
        sample_meta_object,
        mock_filter_results,
        mock_gene_scores,
        mock_statistical_summary,
    ):
        """Test legacy run_analysis method."""
        # Setup mock return value
        mock_results = OrchestrationResults(
            analysis_name="test",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=10.0,
            filter_results=mock_filter_results,
            gene_scores=mock_gene_scores,
            statistical_summary=mock_statistical_summary,
            export_paths={"test": "path"},
        )
        mock_run_complete.return_value = mock_results

        orchestrator = MetaIntegratorOrchestrator()

        result = orchestrator.run_analysis(sample_meta_object, "test_analysis")

        # Verify legacy format
        assert "filter_results" in result
        assert "gene_scores" in result
        assert "statistical_summary" in result
        assert "success" in result
        assert "errors" in result
        assert "export_paths" in result

        # Verify method was called correctly
        mock_run_complete.assert_called_once_with(sample_meta_object, "test_analysis", "standard")


# ==================================================================
# TEST EXPORT AND REPORTING
# ==================================================================


class TestExportAndReporting:
    """Test suite for export and reporting functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.config = OrchestratorConfig(auto_export_results=False, auto_generate_reports=False)

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_auto_export_results(
        self,
        mock_stat_analyzer,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        mock_export_class,
        sample_meta_object,
        mock_filter_results,
        mock_gene_scores,
        mock_statistical_summary,
    ):
        """Test automatic export of results."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        # Setup export engine mock
        mock_export_engine = Mock()
        mock_export_engine.export_filter_results.return_value = {
            "csv": "path1",
            "json": "path2",
        }
        mock_export_engine.export_analysis_results.return_value = {"csv": "path3"}
        mock_export_engine.export_statistical_summary.return_value = {"json": "path4"}
        mock_export_engine._export_json.return_value = "path5"
        orchestrator.export_engine = mock_export_engine

        results = OrchestrationResults(
            analysis_name="test",
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
            filter_results=mock_filter_results,
            gene_scores=mock_gene_scores,
            statistical_summary=mock_statistical_summary,
            comprehensive_summary={"test": "data"},
        )

        orchestrator._auto_export_results(sample_meta_object, results)

        assert "filter" in results.export_paths
        assert "gene_scores" in results.export_paths
        assert "statistical_summary" in results.export_paths
        assert "comprehensive_summary" in results.export_paths

        mock_export_engine.export_filter_results.assert_called_once()
        mock_export_engine.export_analysis_results.assert_called_once()
        mock_export_engine.export_statistical_summary.assert_called_once()
        mock_export_engine._export_json.assert_called_once()

    @patch(
        "metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorOrchestrator._generate_comprehensive_reports"
    )
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_generate_comprehensive_reports(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        mock_generate_reports,
        temp_output_dir,
    ):
        """Test comprehensive report generation."""
        config = OrchestratorConfig(default_output_dir=temp_output_dir)
        orchestrator = EnhancedMetaIntegratorOrchestrator(config)

        results = OrchestrationResults(
            analysis_name="test_analysis",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=25.5,
            performance_metrics={"test": "metric"},
            quality_assessment={"test": "quality"},
            validation_results={"test": "validation"},
        )

        orchestrator._generate_comprehensive_reports(results)

        # Check that the method was called
        mock_generate_reports.assert_called_once_with(results)


# ==================================================================
# INTEGRATION TESTS
# ==================================================================


class TestIntegration:
    """Integration test suite with minimal mocking."""

    def setup_method(self):
        """Setup for each test method."""
        self.config = OrchestratorConfig(
            validate_inputs=True,
            validate_outputs=True,
            auto_export_results=False,
            auto_generate_reports=False,
            auto_cleanup=False,
            fail_fast=False,
        )

    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedMetaIntegratorFilter")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedGeneScorer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.EnhancedResultSummarizer")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.ExportEngine")
    @patch("metaintegrator.scoring_meta_integrator_orchestrator.StatisticalAnalyzer")
    def test_end_to_end_standard_workflow(
        self,
        mock_stat_analyzer,
        mock_export,
        mock_summarizer,
        mock_scorer,
        mock_filter,
        sample_meta_object,
        mock_filter_results,
        mock_gene_scores,
        mock_statistical_summary,
    ):
        """Test complete end-to-end standard workflow."""
        orchestrator = EnhancedMetaIntegratorOrchestrator(self.config)

        # Setup component mocks for full workflow
        orchestrator.meta_filter.filter_meta_analysis.return_value = mock_filter_results
        orchestrator.gene_scorer.score_genes.return_value = mock_gene_scores
        orchestrator.result_summarizer.summarize_results.return_value = {
            "statistical_summary": mock_statistical_summary,
            "comprehensive_data": {"test": "data"},
        }

        results = orchestrator.run_complete_analysis(
            sample_meta_object, "integration_test", "standard"
        )

        # Verify successful execution
        assert isinstance(results, OrchestrationResults)
        assert results.analysis_name == "integration_test"
        assert results.filter_results is not None
        assert results.gene_scores is not None
        assert results.statistical_summary is not None
        assert len(results.errors) == 0
        assert results.total_duration > 0


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
