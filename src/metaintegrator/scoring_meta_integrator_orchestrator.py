"""
Meta Integrator Orchestrator - Unified interface for meta-integrator workflows.
This module provides comprehensive orchestration of all meta-analysis components.

Highlights:
- Uses all modules with consolidated infrastructure
- Enhanced workflow management with configurable pipelines
- Comprehensive error handling and recovery mechanisms
- Advanced batch processing and parallel execution capabilities
- Integrated export and reporting through scoring_export_engine
- Better monitoring and progress tracking
- Flexible component configuration and swapping
- Multi-level orchestration (simple, advanced, batch, custom)
- Pipeline validation and optimization
- Automated quality assessment and recommendations
- Performance monitoring and profiling
- Result validation and consistency checking
- Advanced caching and result persistence
- Workflow templates and presets
"""

import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Import all refactored scoring modules (relative imports within package)
from .scoring_core import (  # Core infrastructure; Exceptions; Data structures
    ConfigurationError,
    LoggerMixin,
    StatisticalSummary,
    ValidationError,
)
from .scoring_export_engine import ExportConfig, ExportEngine

# Import refactored modules
from .scoring_gene_scorer import (
    EnhancedGeneScorer,
    EnhancedScoringConfig,
    ScoringMethod,
)
from .scoring_meta_integrator_filter import (
    EnhancedFilterConfig,
    EnhancedFilterResults,
    EnhancedMetaIntegratorFilter,
)
from .scoring_results_summarizer import (
    EnhancedResultSummarizer,
    EnhancedSummaryConfig,
)
from .scoring_statistical_analysis import StatisticalAnalyzer

# ==================================================================
# ORCHESTRATOR CONFIGURATION
# ==================================================================


@dataclass
class OrchestratorConfig:
    """
    Enhanced configuration for the MetaIntegratorOrchestrator.

    This dataclass defines the settings that control the behavior of the orchestrator. It
    includes options for logging, caching, validation, and workflow control.

    Attributes:
        log_level (str): Logging level. Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL.
        enable_warnings (bool): If true, show warning messages.
        enable_progress_tracking (bool): Show progress bars while components run.
        log_to_file (bool): Write log output to a file.
        log_file_path (Path, optional): Path of the log file. Ignored when log_to_file is false.
        cache_components (bool): Cache results of individual components to avoid recomputation.
        cache_directory (Path, optional): Directory to store cached data. Created automatically if missing.
        enable_parallel_processing (bool): Run components in parallel threads.
        max_workers (int): Maximum number of worker threads used when parallel processing is enabled.
        chunk_size (int): Number of samples processed per batch when a component is split into chunks.
        validate_inputs (bool): Validate input data before processing.
        validate_outputs (bool): Validate output data after processing.
        enable_quality_assessment (bool): Run quality assessment steps on the outputs.
        auto_fix_common_issues (bool): Attempt to correct common data issues automatically.
        default_output_dir (Path): Base directory for all output files.
        auto_export_results (bool): Export results automatically to the default_output_dir.
        auto_generate_reports (bool): Generate summary reports automatically.
        auto_cleanup (bool): Remove intermediate files after successful completion.
        fail_fast (bool): If true, stop immediately when an error occurs.
        retry_failed_components (bool): Retry components that fail during execution.
        max_retries (int): Maximum number of retry attempts for a failed component.
        timeout_seconds (int, optional): Global timeout for component execution. None means no timeout.
    """

    # Logging and monitoring
    log_level: str = "INFO"
    enable_warnings: bool = True
    enable_progress_tracking: bool = True
    log_to_file: bool = False
    log_file_path: Optional[Path] = None

    # Performance and caching
    cache_components: bool = True
    cache_directory: Optional[Path] = None
    enable_parallel_processing: bool = False
    max_workers: int = 4
    chunk_size: int = 1000

    # Validation and quality control
    validate_inputs: bool = True
    validate_outputs: bool = True
    enable_quality_assessment: bool = True
    auto_fix_common_issues: bool = True

    # Export and output
    default_output_dir: Path = field(default_factory=lambda: Path("."))
    auto_export_results: bool = True
    auto_generate_reports: bool = True
    auto_cleanup: bool = False

    # Workflow control
    fail_fast: bool = False  # Stop on first error vs. continue with warnings
    retry_failed_components: bool = True
    max_retries: int = 3
    timeout_seconds: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log level: {self.log_level}")

        if isinstance(self.default_output_dir, str):
            self.default_output_dir = Path(self.default_output_dir)

        if self.cache_directory and isinstance(self.cache_directory, str):
            self.cache_directory = Path(self.cache_directory)

        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")


# ==================================================================
# WORKFLOW RESULT CONTAINER
# ==================================================================


@dataclass
class OrchestrationResults:
    """
    Container for comprehensive orchestration results.

    This dataclass aggregates the results from the MetaIntegratorOrchestrator and
    provides utility properties and methods to summarize the outcomes. It holds core
    metadata about the run, results from individual components, quality and performance
    metrics, export paths, error and warning tracking, and configuration snapshots.

    Attributes:
        analysis_name (str): Identifier of the analysis run.
        success (bool): Overall success flag for the entire orchestration.
        start_time (datetime): Timestamp when the orchestration started.
        end_time (datetime): Timestamp when the orchestration finished.
        total_duration (float): Total runtime in seconds.
        filter_results (EnhancedFilterResults, optional): Results from the filtering step.
        gene_scores (pd.Series, optional): Gene‑level scores produced by the scoring component.
        pathway_scores (Dict[str, float], optional): Pathway‑level scores from the pathway
            scoring component.
        statistical_summary (StatisticalSummary, optional): Summary of statistical tests
            performed during the analysis.
        comprehensive_summary (Dict[str, Any], optional): A full summary of the analysis
            including raw and processed data.
        quality_assessment (Dict[str, Any]): Quality metrics computed during the run.
        performance_metrics (Dict[str, Any]): Runtime and resource usage statistics.
        validation_results (Dict[str, Any]): Results of validation checks on inputs and
            outputs.
        export_paths (Dict[str, Any]): File system paths where results were exported.
        report_paths (List[Path]): Paths to generated report documents.
        errors (List[Dict[str, Any]]): List of error records encountered during execution.
        warnings (List[Dict[str, Any]]): List of warning records.
        config_snapshot (Dict[str, Any]): Snapshot of the configuration used for the run.
        component_versions (Dict[str, str]): Versions of the individual components that were
            executed.

    """

    # Core results
    analysis_name: str
    success: bool
    start_time: datetime
    end_time: datetime
    total_duration: float

    # Component results
    filter_results: Optional[EnhancedFilterResults] = None
    gene_scores: Optional[pd.Series] = None
    pathway_scores: Optional[Dict[str, float]] = None
    statistical_summary: Optional[StatisticalSummary] = None
    comprehensive_summary: Optional[Dict[str, Any]] = None

    # Quality and performance metrics
    quality_assessment: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)

    # Export information
    export_paths: Dict[str, Any] = field(default_factory=dict)
    report_paths: List[Path] = field(default_factory=list)

    # Error and warning tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    component_versions: Dict[str, str] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate based on completed components."""
        total_components = 4  # filter, score, summary, export
        successful_components = sum(
            [
                self.filter_results is not None,
                self.gene_scores is not None,
                self.statistical_summary is not None,
                bool(self.export_paths),
            ]
        )
        return successful_components / total_components

    def get_summary(self) -> Dict[str, Any]:
        """Get a concise summary of the orchestration results."""
        return {
            "analysis_name": self.analysis_name,
            "success": self.success,
            "duration_seconds": self.total_duration,
            "success_rate": self.success_rate,
            "components_completed": {
                "filtering": self.filter_results is not None,
                "gene_scoring": self.gene_scores is not None,
                "pathway_scoring": self.pathway_scores is not None,
                "statistical_analysis": self.statistical_summary is not None,
                "export": bool(self.export_paths),
            },
            "total_genes_filtered": (
                len(self.filter_results.pos_gene_names + self.filter_results.neg_gene_names)
                if self.filter_results
                else 0
            ),
            "total_genes_scored": (len(self.gene_scores) if self.gene_scores is not None else 0),
            "total_pathways_scored": (len(self.pathway_scores) if self.pathway_scores else 0),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


# ==================================================================
# ENHANCED META INTEGRATOR ORCHESTRATOR
# ==================================================================


class EnhancedMetaIntegratorOrchestrator(LoggerMixin):
    """
    Enhanced orchestrator for comprehensive meta‑integrator workflows.

    This class provides a high‑level interface that coordinates all meta‑integrator
    components with advanced features including pipeline management, quality control,
    performance monitoring, and automated reporting.

    Attributes:
        config (OrchestratorConfig, optional): Configuration object controlling
            logging, caching, validation, and workflow behavior. When omitted,
            a default OrchestratorConfig instance is created.
        **kwargs: Additional keyword arguments passed to the LoggerMixin base class.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None, **kwargs) -> None:
        """
        Initialize the enhanced meta‑integrator orchestrator.

        Args:
            config (OrchestratorConfig, optional): Optional orchestrator configuration.
            **kwargs: Additional keyword arguments for LoggerMixin.
        """
        super().__init__(**kwargs)
        self.config = config or OrchestratorConfig()

        # Setup enhanced logging
        self._setup_logging()

        # Initialize component configurations
        self._initialize_component_configs()

        # Initialize components
        self._initialize_components()

        # Performance tracking
        self.performance_tracker = {}
        self.component_cache = {} if self.config.cache_components else None

        self.logger.info("Enhanced MetaIntegratorOrchestrator initialized")

    def run_complete_analysis(
        self,
        meta_object: Dict[str, Any],
        analysis_name: str,
        workflow_type: str = "standard",
    ) -> OrchestrationResults:
        """
        Run complete meta‑analysis workflow with specified type.

        Args:
            meta_object (Dict[str, Any]): Meta‑analysis object containing data.
            analysis_name (str): Name for this analysis.
            workflow_type (str, optional): Type of workflow ("standard",
                "advanced", or "minimal"). Defaults to "standard".

        Returns:
            OrchestrationResults: Comprehensive orchestration results.
        """
        start_time = datetime.now()
        self.logger.info(f"Starting complete analysis: {analysis_name} ({workflow_type})")

        # Initialize result container
        results = OrchestrationResults(
            analysis_name=analysis_name,
            success=False,
            start_time=start_time,
            end_time=start_time,  # Will be updated at the end
            total_duration=0.0,
            config_snapshot=self.config.__dict__.copy(),
            component_versions=self._get_component_versions(),
        )

        try:
            # Step 1: Input validation and preprocessing
            if self.config.validate_inputs:
                self._validate_and_preprocess_inputs(meta_object, results)

            # Step 2: Quality assessment (if enabled)
            if self.config.enable_quality_assessment:
                self._assess_input_quality(meta_object, results)

            # Step 3: Execute workflow based on type
            if workflow_type == "standard":
                self._run_standard_workflow(meta_object, results)
            elif workflow_type == "advanced":
                self._run_advanced_workflow(meta_object, results)
            elif workflow_type == "minimal":
                self._run_minimal_workflow(meta_object, results)
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")

            # Step 4: Output validation (if enabled)
            if self.config.validate_outputs:
                self._validate_outputs(results)

            # Step 5: Auto‑export and reporting (if enabled)
            if self.config.auto_export_results:
                self._auto_export_results(meta_object, results)

            if self.config.auto_generate_reports:
                self._generate_comprehensive_reports(results)

            # Step 6: Cleanup (if enabled)
            if self.config.auto_cleanup:
                self._cleanup_temporary_files(results)

            # Mark as successful if we made it this far
            results.success = len(results.errors) == 0

        except Exception as e:
            self._handle_orchestration_error(e, results)

        finally:
            # Update timing
            end_time = datetime.now()
            results.end_time = end_time
            results.total_duration = (end_time - start_time).total_seconds()

            # Log final status
            if results.success:
                self.logger.info(
                    f"Analysis completed successfully in {results.total_duration:.2f}s"
                )
            else:
                self.logger.error(f"Analysis failed after {results.total_duration:.2f}s")

        return results

    def run_batch_analysis(
        self,
        meta_objects: Dict[str, Dict[str, Any]],
        batch_name: str,
        workflow_type: str = "standard",
    ) -> Dict[str, OrchestrationResults]:
        """
        Run batch analysis on multiple meta‑objects.

        Args:
            meta_objects (Dict[str, Dict[str, Any]]): Mapping of analysis names to their meta‑objects.
            batch_name (str): Identifier for the entire batch.
            workflow_type (str): Workflow type to apply to all analyses. Defaults to "standard".

        Returns:
            Dict[str, OrchestrationResults]: Mapping of analysis names to their orchestration results.
        """
        self.logger.info(f"Starting batch analysis: {batch_name} ({len(meta_objects)} analyses)")

        batch_results = {}

        if self.config.enable_parallel_processing and len(meta_objects) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self.run_complete_analysis, meta_obj, name, workflow_type): name
                    for name, meta_obj in meta_objects.items()
                }

                for future in futures:
                    analysis_name = futures[future]
                    try:
                        batch_results[analysis_name] = future.result()
                    except Exception as e:
                        self.logger.error(f"Batch analysis failed for {analysis_name}: {e}")
                        # Create error result
                        batch_results[analysis_name] = self._create_error_result(analysis_name, e)
        else:
            # Sequential execution
            for analysis_name, meta_object in meta_objects.items():
                try:
                    batch_results[analysis_name] = self.run_complete_analysis(
                        meta_object, analysis_name, workflow_type
                    )
                except Exception as e:
                    self.logger.error(f"Batch analysis failed for {analysis_name}: {e}")
                    batch_results[analysis_name] = self._create_error_result(analysis_name, e)

        # Generate batch summary
        self._generate_batch_summary(batch_results, batch_name)

        return batch_results

    def _run_standard_workflow(
        self, meta_object: Dict[str, Any], results: OrchestrationResults
    ) -> None:
        """
        Run standard workflow: filter → score → summarize → export.

        Args:
            meta_object (Dict[str, Any]): Meta‑analysis data for this run.
            results (OrchestrationResults): Container to store results and metrics.

        This method executes the four main stages of the standard workflow. Each stage
        is wrapped in a try/except block so that errors are recorded in the results
        object and optionally propagate if fail_fast is enabled. Performance
        metrics for each stage are stored in results.performance_metrics.
        """
        # Step 1: Filtering
        self.logger.info("Step 1: Filtering meta-analysis results")
        try:
            start_time = time.time()
            filter_results = self.meta_filter.filter_meta_analysis(meta_object, strategy="standard")
            results.filter_results = filter_results
            results.performance_metrics["filtering_duration"] = time.time() - start_time
            self.logger.info(f"Filtered to {filter_results.total_genes} significant genes")
        except Exception as e:
            self._handle_component_error("filtering", e, results)
            if self.config.fail_fast:
                raise

        # Step 2: Gene scoring
        if results.filter_results and (
            results.filter_results.total_genes > 0 or not self.config.fail_fast
        ):
            self.logger.info("Step 2: Scoring genes")
            try:
                start_time = time.time()
                gene_scores = self.gene_scorer.score_genes(meta_object, ScoringMethod.COMBINED)
                results.gene_scores = gene_scores
                results.performance_metrics["scoring_duration"] = time.time() - start_time
                self.logger.info(f"Scored {len(gene_scores)} genes")
            except Exception as e:
                self._handle_component_error("gene_scoring", e, results)
                if self.config.fail_fast:
                    raise

        # Step 3: Statistical analysis and summarization
        if results.gene_scores is not None or not self.config.fail_fast:
            self.logger.info("Step 3: Statistical analysis and summarization")
            try:
                start_time = time.time()
                if "original_result" in meta_object:
                    data = meta_object["original_result"]
                elif "filterRes" in meta_object:
                    data = meta_object["filterRes"]
                else:
                    raise ValueError("No data found for summarization")

                comprehensive_summary = self.result_summarizer.summarize_results(
                    data, results.analysis_name
                )
                results.comprehensive_summary = comprehensive_summary
                results.statistical_summary = comprehensive_summary.get("statistical_summary")
                results.performance_metrics["summarization_duration"] = time.time() - start_time
                self.logger.info("Statistical analysis completed")
            except Exception as e:
                self._handle_component_error("summarization", e, results)
                if self.config.fail_fast:
                    raise

    def _run_advanced_workflow(
        self, meta_object: Dict[str, Any], results: OrchestrationResults
    ) -> None:
        """
        Run advanced workflow with multiple filtering strategies and pathway analysis.

        Args:
            meta_object (Dict[str, Any]): Meta‑analysis data for this run.
            results (OrchestrationResults): Container to store results and metrics.

        This method executes the advanced pipeline consisting of three main stages. Each
        stage records its performance time in results.performance_metrics and captures
        errors in the results object. If the configuration flag fail_fast is set, any
        exception propagates after being logged.

        Step 1: Advanced multi‑strategy filtering
            Attempts adaptive filtering first; if it fails, falls back to standard
            filtering. The chosen filter results are stored in results.filter_results
            and the duration is recorded.

        Step 2: Enhanced gene scoring with bootstrap confidence intervals
            Uses an EnhancedGeneScorer configured with bootstrap, cross‑validation
            and adaptive normalization. Scoring results are stored in
            results.gene_scores. If confidence intervals are produced, their count
            is added to the performance metrics.

        Step 3: Comprehensive statistical analysis
            Applies an EnhancedResultSummarizer that includes trend analysis,
            quality assessment, and visualization generation. The summarized
            output, statistical summary and quality assessment are stored in
            results.comprehensive_summary, results.statistical_summary and
            results.quality_assessment respectively. The duration of the analysis
            is recorded in the performance metrics.
        """

        # Step 1: Advanced filtering with multiple strategies
        self.logger.info("Step 1: Advanced multi-strategy filtering")
        try:
            start_time = time.time()
            # Try adaptive filtering first, fall back to standard if it fails
            try:
                filter_results = self.meta_filter.filter_meta_analysis(
                    meta_object, strategy="adaptive"
                )
            except:
                self.logger.warning("Adaptive filtering failed, falling back to standard")
                filter_results = self.meta_filter.filter_meta_analysis(
                    meta_object, strategy="standard"
                )

            results.filter_results = filter_results
            results.performance_metrics["filtering_duration"] = time.time() - start_time
            self.logger.info(f"Advanced filtering completed: {filter_results.total_genes} genes")
        except Exception as e:
            self._handle_component_error("advanced_filtering", e, results)
            if self.config.fail_fast:
                raise

        # Step 2: Enhanced gene scoring with bootstrap confidence intervals
        if results.filter_results and (
            results.filter_results.total_genes > 0 or not self.config.fail_fast
        ):
            self.logger.info("Step 2: Enhanced gene scoring")
            try:
                start_time = time.time()
                # Use enhanced scoring config with bootstrap
                enhanced_config = EnhancedScoringConfig(
                    enable_bootstrap=True,
                    enable_cross_validation=True,
                    adaptive_normalization=True,
                )
                enhanced_scorer = EnhancedGeneScorer(enhanced_config)
                gene_scores = enhanced_scorer.score_genes(meta_object, ScoringMethod.COMBINED)
                results.gene_scores = gene_scores
                results.performance_metrics["scoring_duration"] = time.time() - start_time

                # Get confidence intervals if available
                confidence_intervals = enhanced_scorer.get_last_confidence_intervals()
                if confidence_intervals:
                    results.performance_metrics["confidence_intervals_count"] = len(
                        confidence_intervals
                    )

                self.logger.info(f"Enhanced scoring completed: {len(gene_scores)} genes")
            except Exception as e:
                self._handle_component_error("enhanced_scoring", e, results)
                if self.config.fail_fast:
                    raise

        # Step 3: Comprehensive analysis with all features
        if results.gene_scores is not None or not self.config.fail_fast:
            self.logger.info("Step 3: Comprehensive statistical analysis")
            try:
                start_time = time.time()
                enhanced_config = EnhancedSummaryConfig(
                    include_trend_analysis=True,
                    include_quality_assessment=True,
                    generate_visualizations=True,
                    auto_export=False,  # We'll handle export separately
                )
                enhanced_summarizer = EnhancedResultSummarizer(enhanced_config)

                if "original_result" in meta_object:
                    data = meta_object["original_result"]
                elif "filterRes" in meta_object:
                    data = meta_object["filterRes"]
                else:
                    raise ValueError("No data found for comprehensive analysis")

                comprehensive_summary = enhanced_summarizer.summarize_results(
                    data, results.analysis_name
                )
                results.comprehensive_summary = comprehensive_summary
                results.statistical_summary = comprehensive_summary.get("statistical_summary")
                results.quality_assessment = comprehensive_summary.get("quality_assessment", {})
                results.performance_metrics["comprehensive_analysis_duration"] = (
                    time.time() - start_time
                )
                self.logger.info("Comprehensive analysis completed")
            except Exception as e:
                self._handle_component_error("comprehensive_analysis", e, results)
                if self.config.fail_fast:
                    raise

    def _run_minimal_workflow(
        self, meta_object: Dict[str, Any], results: OrchestrationResults
    ) -> None:
        """
        Run minimal workflow: basic filtering and scoring only.

        This lightweight pipeline performs only two steps:
        1. Standard filtering
        2. Effect‑size scoring

        Each step records its runtime in results.performance_metrics and handles
        errors via _handle_component_error. If the orchestrator configuration has
        fail_fast set, any exception is re‑raised after logging.
        """
        # Step 1: Basic filtering
        self.logger.info("Step 1: Basic filtering")
        try:
            start_time = time.time()
            filter_results = self.meta_filter.filter_meta_analysis(meta_object, strategy="standard")
            results.filter_results = filter_results
            results.performance_metrics["filtering_duration"] = time.time() - start_time
        except Exception as e:
            self._handle_component_error("basic_filtering", e, results)
            if self.config.fail_fast:
                raise

        # Step 2: Basic scoring
        if results.filter_results and (
            results.filter_results.total_genes > 0 or not self.config.fail_fast
        ):
            self.logger.info("Step 2: Basic scoring")
            try:
                start_time = time.time()
                gene_scores = self.gene_scorer.score_genes(meta_object, ScoringMethod.EFFECT_SIZE)
                results.gene_scores = gene_scores
                results.performance_metrics["scoring_duration"] = time.time() - start_time
            except Exception as e:
                self._handle_component_error("basic_scoring", e, results)
                if self.config.fail_fast:
                    raise

    def _setup_logging(self) -> None:
        """
        Setup enhanced logging configuration.

        The method pulls the desired log level from the orchestrator’s config,
        configures the root logger with a console handler, and optionally
        attaches a file handler if log_to_file is True.  Log format is
        %(asctime)s - %(name)s - %(levelname)s - %(message)s.
        """
        log_level = getattr(logging, self.config.log_level)

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Setup file logging if requested
        if self.config.log_to_file and self.config.log_file_path:
            file_handler = logging.FileHandler(self.config.log_file_path)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logging.getLogger().addHandler(file_handler)

    def _initialize_component_configs(self) -> None:
        """
        Initialize configurations for all components.

        The configurations are created once per orchestrator instance and can be
        overridden by specific workflow methods (e.g., bootstrap is enabled only
        in the advanced workflow).  Each configuration honours the cache and
        validation flags from the orchestrator’s top‑level config.
        """
        # Filter configuration
        self.filter_config = EnhancedFilterConfig(
            adaptive_thresholds=True,
            prioritize_by_strength=True,
            cache_intermediate_results=self.config.cache_components,
        )

        # Scoring configuration
        self.scoring_config = EnhancedScoringConfig(
            enable_bootstrap=False,  # Enabled only in advanced workflow
            adaptive_normalization=True,
            cache_intermediate=self.config.cache_components,
        )

        # Summary configuration
        self.summary_config = EnhancedSummaryConfig(
            include_trend_analysis=False,  # Enabled only in advanced workflow
            include_quality_assessment=self.config.enable_quality_assessment,
            auto_export=False,  # Export handled separately
        )

        # Export configuration
        self.export_config = ExportConfig(
            base_output_dir=self.config.default_output_dir,
            include_metadata=True,
            include_plots=True,
            validate_exports=self.config.validate_outputs,
        )

    ### START

    def _initialize_components(self) -> None:
        """
        Instantiate every core component of the orchestrator.

        Attrbitues:
        - EnhancedMetaIntegratorFilter – filters meta‑analysis results.
        - EnhancedGeneScorer – assigns gene‑level scores.
        - EnhancedResultSummarizer – produces the final summary and exports.
        - ExportEngine – handles file‑system export of results.
        - StatisticalAnalyzer – standalone statistical utilities.

        The method catches any exception during instantiation, logs it, and
        re‑raises a ConfigurationError so that the caller knows that the
        orchestrator cannot run until all components are ready.
        """
        try:
            self.meta_filter = EnhancedMetaIntegratorFilter(self.filter_config)
            self.gene_scorer = EnhancedGeneScorer(self.scoring_config)
            self.result_summarizer = EnhancedResultSummarizer(
                self.summary_config, self.export_config
            )
            self.export_engine = ExportEngine(self.export_config)

            # Statistical analyzer for use outside the main pipelines
            self.statistical_analyzer = StatisticalAnalyzer(self.summary_config)

            self.logger.info("All components initialized successfully")
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise ConfigurationError(f"Failed to initialize components: {e}")

    def _validate_and_preprocess_inputs(
        self, meta_object: Dict[str, Any], results: OrchestrationResults
    ) -> None:
        """
        Validate the user‑supplied meta_object and perform light‑weight
        preprocessing before any workflow is run.

        Validation steps:
        1. meta_object must be a dict containing either
           'original_result' or 'filterRes'.
        2. The contained data must be a non‑empty pandas.DataFrame.
        3. Required columns ('effect_size', 'effect_fdr') must be present.
           If columns are missing and auto_fix_common_issues is true,
           the method attempts a very small repair (e.g., mapping a legacy
           'fdr' column to 'effect_fdr').

        Successful validation is recorded in
        results.validation_results['input_validation'] = 'passed'.
        Failure triggers _handle_component_error and, if fail_fast
        is set, propagates the exception.

        Args:
        meta_object : dict
            The raw meta‑analysis payload supplied by the user.
        results : OrchestrationResults
            Container used to store status flags and error information.
        """
        try:
            # 1️⃣ Basic type & key checks
            if not isinstance(meta_object, dict):
                raise ValidationError("meta_object must be a dictionary")

            if not ("original_result" in meta_object or "filterRes" in meta_object):
                raise ValidationError("meta_object must contain 'original_result' or 'filterRes'")

            # 2️⃣ Retrieve the data frame
            data = meta_object.get("original_result")
            if data is None:
                data = meta_object.get("filterRes")

            if not isinstance(data, pd.DataFrame):
                raise ValidationError("Data must be a pandas DataFrame")

            if data.empty:
                raise ValidationError("Data cannot be empty")

            # 3️⃣ Ensure mandatory columns exist
            required_cols = ["effect_size", "effect_fdr"]
            missing_cols = [c for c in required_cols if c not in data.columns]
            if missing_cols:
                if self.config.auto_fix_common_issues:
                    self.logger.warning(f"Missing columns {missing_cols}, attempting to fix")
                    # Very simple, future‑proofing: map legacy names
                    if "effect_fdr" not in data.columns and "fdr" in data.columns:
                        data["effect_fdr"] = data["fdr"]
                        meta_object["original_result"] = data
                else:
                    raise ValidationError(f"Missing required columns: {missing_cols}")

            # 4️⃣ Record success
            results.validation_results["input_validation"] = "passed"

        except Exception as exc:
            results.validation_results["input_validation"] = "failed"
            self._handle_component_error("input_validation", exc, results)
            if self.config.fail_fast:
                raise

    def _assess_input_quality(
        self, meta_object: Dict[str, Any], results: OrchestrationResults
    ) -> None:
        """
        Evaluate the quality of the input data and store the assessment in the results object.

        Args:
            meta_object: Dictionary containing the raw or filtered result data. Either the key
                original_result or filterRes is expected to hold a pandas DataFrame.
            results: OrchestrationResults instance that will receive the quality metrics.

        This method calculates basic metrics such as the number of genes, the percentage of
        missing data, the number of duplicated genes, and the count of infinite numeric
        values. A quality score is derived from these metrics and added to the dictionary.
        If the score falls below a threshold a warning is emitted. All exceptions are
        forwarded to the internal component‑error handler.
        """
        try:
            if "original_result" in meta_object:
                data = meta_object["original_result"]
            else:
                data = meta_object["filterRes"]

            # Basic quality metrics
            quality_metrics = {
                "total_genes": len(data),
                "missing_data_percentage": data.isnull().sum().sum()
                / (len(data) * len(data.columns))
                * 100,
                "duplicate_genes": data.index.duplicated().sum(),
                "infinite_values": np.isinf(data.select_dtypes(include=[np.number])).sum().sum(),
            }

            # Quality score calculation
            quality_score = 100.0
            if quality_metrics["missing_data_percentage"] > 20:
                quality_score -= 20
            if quality_metrics["duplicate_genes"] > 0:
                quality_score -= 15
            if quality_metrics["infinite_values"] > 0:
                quality_score -= 10

            quality_metrics["overall_quality_score"] = quality_score

            results.quality_assessment = quality_metrics

            if quality_score < 50:
                self.logger.warning(f"Low data quality detected (score: {quality_score:.1f})")

        except Exception as e:
            self._handle_component_error("quality_assessment", e, results)

    def _auto_export_results(
        self, meta_object: Dict[str, Any], results: OrchestrationResults
    ) -> None:
        """
        Export the analysis results into several file formats and record the paths.

        Args:
            meta_object: Dictionary containing intermediate metadata (not used directly in
                the export logic but kept for consistency with the method signature).
            results: OrchestrationResults instance that holds the data to be exported.

        The function checks that at least one of the result components is present; if
        none are available a warning is logged. It then exports filter results, gene
        scores, statistical summary and a comprehensive summary via the export engine.
        The file paths returned by each export routine are stored in
        results.export_paths. Successful completion is logged.
        """
        try:
            if not any(
                [
                    results.filter_results,
                    results.gene_scores,
                    results.statistical_summary,
                ]
            ):
                self.logger.warning("No results available for export")
                return

            export_paths = {}

            # Export filter results
            if results.filter_results:
                filter_exports = self.export_engine.export_filter_results(
                    results.filter_results, f"{results.analysis_name}_filter"
                )
                export_paths["filter"] = filter_exports

            # Export gene scores
            if results.gene_scores is not None:
                scores_df = pd.DataFrame({"gene_score": results.gene_scores})
                score_exports = self.export_engine.export_analysis_results(
                    scores_df, f"{results.analysis_name}_gene_scores"
                )
                export_paths["gene_scores"] = score_exports

            # Export statistical summary
            if results.statistical_summary:
                summary_exports = self.export_engine.export_statistical_summary(
                    results.statistical_summary, f"{results.analysis_name}_statistics"
                )
                export_paths["statistical_summary"] = summary_exports

            # Export comprehensive summary
            if results.comprehensive_summary:
                comprehensive_path = self.export_engine._export_json(
                    results.comprehensive_summary,
                    f"{results.analysis_name}_comprehensive",
                )
                export_paths["comprehensive_summary"] = comprehensive_path

            results.export_paths = export_paths
            self.logger.info(f"Results exported: {len(export_paths)} components")

        except Exception as e:
            self._handle_component_error("export", e, results)

    def _generate_comprehensive_reports(self, results: OrchestrationResults) -> None:
        """
        Produce a JSON report that aggregates summary, performance, quality and validation
        data from the analysis.

        Args:
            results: OrchestrationResults instance containing all components of the
                analysis.

        The report is written to the default output directory specified in the
        configuration and its path is appended to results.report_paths. Any
        exceptions are captured by the component error handler.
        """
        try:
            # This would generate detailed reports based on all results
            # For now, create a simple summary report

            report_data = {
                "analysis_summary": results.get_summary(),
                "performance_metrics": results.performance_metrics,
                "quality_assessment": results.quality_assessment,
                "validation_results": results.validation_results,
            }

            report_path = (
                self.config.default_output_dir
                / f"{results.analysis_name}_comprehensive_report.json"
            )
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

            results.report_paths.append(report_path)
            self.logger.info(f"Comprehensive report generated: {report_path}")

        except Exception as e:
            self._handle_component_error("report_generation", e, results)

    def _handle_component_error(
        self, component: str, error: Exception, results: OrchestrationResults
    ) -> None:
        """
        Log and record errors that occur in individual components of the pipeline.

        Args:
            component: Name of the component that failed.
            error: Exception instance that was raised.
            results: OrchestrationResults instance to which the error information will be
                appended.

        The error details include component name, exception type, message, timestamp and
        the traceback. These are stored in results.errors and an error log entry is
        written.
        """
        error_info = {
            "component": component,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now(),
            "traceback": traceback.format_exc(),
        }

        results.errors.append(error_info)
        self.logger.error(f"Component {component} failed: {error}")

    def _handle_orchestration_error(self, error: Exception, results: OrchestrationResults) -> None:
        """
        Record a top‑level orchestration failure.

        Args:
            error: Exception instance that caused the orchestration to stop.
            results: OrchestrationResults instance where error information will be appended.

        The method stores the component name, error type, message, timestamp and traceback
        in the results.errors list and logs the failure.
        """
        error_info = {
            "component": "orchestrator",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now(),
            "traceback": traceback.format_exc(),
        }

        results.errors.append(error_info)
        self.logger.error(f"Orchestration failed: {error}")

    def _validate_outputs(self, results: OrchestrationResults) -> None:
        """
        Verify that the produced outputs meet quality and consistency expectations.

        Args:
            results: OrchestrationResults instance containing the outputs to be validated.

        Validation checks include the presence of genes after filtering, the existence of
        non‑null gene scores, and the consistency of gene lists between filtering and
        scoring. Validation messages are stored in results.validation_results. Any
        exceptions are forwarded to the component‑error handler.
        """
        try:
            validation_results = {}

            # Validate filter results
            if results.filter_results:
                if results.filter_results.total_genes == 0:
                    validation_results["filter_results"] = "warning: no genes passed filtering"
                else:
                    validation_results["filter_results"] = "passed"

            # Validate gene scores
            if results.gene_scores is not None:
                if len(results.gene_scores) == 0:
                    validation_results["gene_scores"] = "warning: no genes scored"
                elif results.gene_scores.isnull().all():
                    validation_results["gene_scores"] = "error: all scores are null"
                else:
                    validation_results["gene_scores"] = "passed"

            # Check consistency between components
            if results.filter_results and results.gene_scores is not None:
                filtered_genes = set(
                    results.filter_results.pos_gene_names + results.filter_results.neg_gene_names
                )
                scored_genes = set(results.gene_scores.index)

                if not filtered_genes.issubset(scored_genes):
                    validation_results["consistency"] = "warning: some filtered genes not scored"
                else:
                    validation_results["consistency"] = "passed"

            results.validation_results.update(validation_results)

        except Exception as e:
            self._handle_component_error("output_validation", e, results)

    def _cleanup_temporary_files(self, results: OrchestrationResults) -> None:
        """
        Remove temporary files created during the orchestration when auto‑cleanup is enabled.

        Args:
            results: OrchestrationResults instance (unused in the current implementation).

        The cleanup logic is expected to delete temporary artifacts. After completion
        a log entry confirms the cleanup.
        """
        # Implementation for cleaning up temporary files
        self.logger.info("Cleanup completed")

    def _generate_batch_summary(
        self, batch_results: Dict[str, OrchestrationResults], batch_name: str
    ) -> None:
        """
        Create a JSON report summarizing a batch of analyses.

        Args:
            batch_results: Mapping from analysis names to OrchestrationResults instances.
            batch_name: Identifier for the batch.

        The summary includes the batch name, the number of analyses, how many
        completed successfully, the total and average duration of all analyses,
        and a dictionary of individual summaries. The report is written to the
        default output directory and a log entry records the location of the
        generated file.
        """
        summary = {
            "batch_name": batch_name,
            "total_analyses": len(batch_results),
            "successful_analyses": sum(1 for r in batch_results.values() if r.success),
            "total_duration": sum(r.total_duration for r in batch_results.values()),
            "average_duration": sum(r.total_duration for r in batch_results.values())
            / len(batch_results),
            "individual_summaries": {name: r.get_summary() for name, r in batch_results.items()},
        }

        # Export batch summary
        summary_path = self.config.default_output_dir / f"{batch_name}_batch_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Batch summary generated: {summary_path}")

    def _create_error_result(self, analysis_name: str, error: Exception) -> OrchestrationResults:
        """
        Build an OrchestrationResults object that represents a failed analysis.

        Args:
            analysis_name: Name of the analysis that failed.
            error: Exception instance that caused the failure.

        The returned object records the analysis name, marks success as False,
        sets the start and end times to the current moment, and records a
        single error entry with component name, error type, message, timestamp
        and traceback. The total duration is zero.
        """
        now = datetime.now()
        return OrchestrationResults(
            analysis_name=analysis_name,
            success=False,
            start_time=now,
            end_time=now,
            total_duration=0.0,
            errors=[
                {
                    "component": "orchestrator",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "timestamp": now,
                    "traceback": traceback.format_exc(),
                }
            ],
        )

    def _get_component_versions(self) -> Dict[str, str]:
        """
        Retrieve the current software version for each orchestrator component.

        Returns:
            A dictionary mapping the component name to its version string.
        """
        return {
            "orchestrator": "2.0.0",
            "filter": "2.0.0",
            "scorer": "2.0.0",
            "summarizer": "2.0.0",
            "export_engine": "2.0.0",
        }

    def get_component_status(self) -> Dict[str, str]:
        """
        Determine whether key sub‑components have been instantiated.

        Returns:
            A dictionary with component names as keys and status strings
            ('initialized' or 'not_initialized') as values.
        """
        return {
            "meta_filter": ("initialized" if hasattr(self, "meta_filter") else "not_initialized"),
            "gene_scorer": ("initialized" if hasattr(self, "gene_scorer") else "not_initialized"),
            "result_summarizer": (
                "initialized" if hasattr(self, "result_summarizer") else "not_initialized"
            ),
            "export_engine": (
                "initialized" if hasattr(self, "export_engine") else "not_initialized"
            ),
        }

    def clear_all_caches(self) -> None:
        """
        Remove all cached data from sub‑components and the orchestrator’s
        internal component cache.

        This routine calls each component’s cache‑clearing method when the
        component is present, clears the orchestrator’s component cache if
        it exists, and logs the action.
        """
        if hasattr(self, "meta_filter"):
            self.meta_filter.clear_cache()
        if hasattr(self, "gene_scorer"):
            self.gene_scorer.clear_cache()
        if hasattr(self, "result_summarizer"):
            self.result_summarizer.clear_analysis_cache()

        if self.component_cache:
            self.component_cache.clear()

        self.logger.info("All caches cleared")


# ==================================================================
# LEGACY COMPATIBILITY CLASS
# ==================================================================


class MetaIntegratorOrchestrator(EnhancedMetaIntegratorOrchestrator):
    """Legacy compatibility class for backward compatibility."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        """
        Legacy initialization supporting legacy config format.

        Args:
            config: Optional dictionary containing legacy configuration values.
            **kwargs: Additional keyword arguments passed to the parent constructor.
        """
        if config:
            # Convert legacy config dict to OrchestratorConfig
            enhanced_config = OrchestratorConfig(
                log_level=config.get("log_level", "INFO"),
                enable_warnings=config.get("enable_warnings", True),
                cache_components=config.get("cache_components", True),
                validate_inputs=config.get("validate_inputs", True),
                default_output_dir=Path(config.get("default_output_dir", ".")),
                auto_cleanup=config.get("auto_cleanup", False),
            )
        else:
            enhanced_config = None

        super().__init__(enhanced_config, **kwargs)

    def run_analysis(
        self, meta_object: Dict[str, Any], analysis_name: str = "analysis"
    ) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.

        Args:
            meta_object: Dictionary containing meta data to analyze.
            analysis_name: Name of the analysis to run.

        Returns:
            Dictionary containing legacy formatted results, including filter_results,
            gene_scores, statistical_summary, success, errors, and export_paths.
        """
        results = self.run_complete_analysis(meta_object, analysis_name, "standard")

        # Convert to legacy format
        return {
            "filter_results": results.filter_results,
            "gene_scores": results.gene_scores,
            "statistical_summary": results.statistical_summary,
            "success": results.success,
            "errors": results.errors,
            "export_paths": results.export_paths,
        }
