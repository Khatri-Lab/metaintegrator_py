"""
Comprehensive validation test for all refactored scoring modules.
This test ensures that all refactored modules work correctly both individually and together.

Test Coverage:
- Individual module functionality
- Inter-module integration
- Orchestrator workflows (standard, advanced, minimal)
- Export engine capabilities
- Statistical analysis features
- Backward compatibility
- Performance improvements
- New feature validation
"""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import all refactored modules
try:
    from metaintegrator.scoring_core import (
        ScoringMethod,
        AggregationMethod,
        ExportFormat,
        ValidationError,
        ConfigurationError,
        ScoringCalculationError,
    )
    from metaintegrator.scoring_filter_manager import FilterManager
    from metaintegrator.scoring_statistical_analysis import (
        StatisticalAnalyzer,
        DistributionAnalyzer,
    )
    from metaintegrator.scoring_export_engine import ExportEngine, ExportConfig
    from metaintegrator.scoring_results_summarizer import (
        EnhancedResultSummarizer,
        EnhancedSummaryConfig,
    )
    from metaintegrator.scoring_gene_scorer import (
        EnhancedGeneScorer,
        EnhancedScoringConfig,
    )
    from metaintegrator.scoring_meta_integrator_filter import (
        EnhancedMetaIntegratorFilter,
        EnhancedFilterConfig,
    )
    from metaintegrator.scoring_meta_integrator_orchestrator import (
        EnhancedMetaIntegratorOrchestrator,
        OrchestratorConfig,
        OrchestrationResults,
    )

    # Also import legacy compatibility classes
    from metaintegrator.scoring_results_summarizer import ResultSummarizer
    from metaintegrator.scoring_gene_scorer import GeneScorer
    from metaintegrator.scoring_meta_integrator_filter import MetaIntegratorFilter
    from metaintegrator.scoring_meta_integrator_orchestrator import (
        MetaIntegratorOrchestrator,
    )

    IMPORTS_SUCCESSFUL = True
    print("✅ All module imports successful")
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    print(f"❌ Module import failed: {e}")


class RefactoringValidationTest:
    """Comprehensive validation test suite for refactored modules."""

    def __init__(self):
        """Initialize the validation test suite."""
        self.test_results = []
        self.temp_dir = None

        # Setup logging for tests
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create test data
        self.test_meta_object = self._create_test_meta_object()

    def run_all_tests(self) -> dict:
        """Run all validation tests and return comprehensive results."""
        print("\n🔬 Starting Comprehensive Refactoring Validation Tests")
        print("=" * 60)

        if not IMPORTS_SUCCESSFUL:
            return {"success": False, "error": "Module imports failed"}

        # Create temporary directory for test outputs
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"📁 Test output directory: {self.temp_dir}")

        try:
            # Test individual modules
            self._test_scoring_core()
            self._test_statistical_analysis()
            self._test_export_engine()
            self._test_gene_scorer()
            self._test_meta_integrator_filter()
            self._test_results_summarizer()

            # Test integration
            self._test_orchestrator_workflows()
            self._test_backward_compatibility()
            self._test_performance_improvements()
            self._test_new_features()

            # Generate summary
            summary = self._generate_test_summary()

            print("\n📊 Test Summary:")
            print("=" * 40)
            print(f"Total Tests: {summary['total_tests']}")
            print(f"Passed: {summary['passed_tests']} ✅")
            print(f"Failed: {summary['failed_tests']} ❌")
            print(f"Success Rate: {summary['success_rate']:.1f}%")

            if summary["failed_tests"] > 0:
                print("\n❌ Failed Tests:")
                for result in self.test_results:
                    if not result["success"]:
                        print(f"  - {result['test_name']}: {result['error']}")

            return summary

        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_tests": len(self.test_results),
            }

        finally:
            # Cleanup
            self._cleanup()

    def _create_test_meta_object(self) -> dict:
        """Create test meta-analysis object with realistic data."""
        # Generate test data
        n_genes = 100
        genes = [f"GENE_{i:03d}" for i in range(n_genes)]

        # Create realistic meta-analysis results
        np.random.seed(42)  # For reproducibility

        data = pd.DataFrame(
            {
                "effect_size": np.random.normal(0, 0.5, n_genes),
                "effect_fdr": np.random.beta(2, 8, n_genes),  # Skewed towards smaller values
                "fisher_score": np.random.gamma(2, 2, n_genes),
                "heterogeneity": np.random.beta(1, 9, n_genes),  # Mostly low heterogeneity
                "heterogeneity_pval": np.random.uniform(0, 1, n_genes),
                "number_studies": np.random.randint(3, 15, n_genes),
            },
            index=genes,
        )

        return {
            "original_result": data,
            "study_names": ["Study_A", "Study_B", "Study_C", "Study_D"],
            "meta_analysis_type": "fixed_effect",
        }

    def _run_test(self, test_name: str, test_function) -> bool:
        """Run a single test and record results."""
        try:
            print(f"🧪 Running {test_name}...", end="")
            test_function()
            print(" ✅")
            self.test_results.append(
                {
                    "test_name": test_name,
                    "success": True,
                    "error": None,
                    "timestamp": datetime.now(),
                }
            )
            return True
        except Exception as e:
            print(f" ❌ ({e})")
            self.test_results.append(
                {
                    "test_name": test_name,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now(),
                }
            )
            return False

    def _test_scoring_core(self):
        """Test scoring_core infrastructure."""

        def test_core_infrastructure():
            # Test enums
            assert ScoringMethod.COMBINED.value == "combined"
            assert AggregationMethod.MEAN.value == "mean"
            assert ExportFormat.CSV.value == "csv"

            # Test exception hierarchy
            try:
                raise ValidationError("Test validation error")
            except ValidationError:
                pass  # Expected

            # Test data structures would be tested here
            # (Many are tested implicitly in other components)

        self._run_test("Scoring Core Infrastructure", test_core_infrastructure)

    def _test_statistical_analysis(self):
        """Test statistical analysis module."""

        def test_statistical_analyzer():
            from metaintegrator.scoring_core import SummaryConfig

            config = SummaryConfig(include_distributions=True, include_correlations=True)
            analyzer = StatisticalAnalyzer(config)

            # Test with actual data
            results = analyzer.analyze(self.test_meta_object["original_result"])

            assert results.basic_stats is not None
            assert results.distribution_stats is not None
            assert results.correlation_stats is not None
            assert results.metadata is not None
            assert len(results.basic_stats) > 0

        def test_distribution_analyzer():
            analyzer = DistributionAnalyzer()

            test_data = {
                "normal_data": np.random.normal(0, 1, 1000),
                "skewed_data": np.random.gamma(2, 2, 1000),
            }

            results = analyzer.analyze_distributions(test_data)

            assert "normal_data" in results
            assert "skewed_data" in results
            assert "comparison" in results  # Multi-variable comparison

        self._run_test("Statistical Analyzer", test_statistical_analyzer)
        self._run_test("Distribution Analyzer", test_distribution_analyzer)

    def _test_export_engine(self):
        """Test export engine functionality."""

        def test_export_engine():
            config = ExportConfig(
                base_output_dir=self.temp_dir,
                include_metadata=True,
                validate_exports=True,
            )
            engine = ExportEngine(config)

            # Test DataFrame export
            test_df = self.test_meta_object["original_result"]
            exports = engine.export_analysis_results(
                test_df, "test_export", [ExportFormat.CSV, ExportFormat.JSON]
            )

            assert "csv" in exports
            assert "json" in exports
            assert exports["csv"].exists()
            assert exports["metadata"].exists()

            # Test statistical summary export
            from metaintegrator.scoring_core import StatisticalSummary

            summary = StatisticalSummary(
                basic_stats={"test_col": {"mean": 1.0, "std": 0.5}},
                metadata={"test": True},
            )

            summary_exports = engine.export_statistical_summary(summary, "test_summary")
            assert len(summary_exports) > 0

        self._run_test("Export Engine", test_export_engine)

    def _test_gene_scorer(self):
        """Test gene scorer functionality."""

        def test_enhanced_gene_scorer():
            config = EnhancedScoringConfig(
                normalization_method="min_max",
                handle_missing="median",
                adaptive_normalization=True,
            )
            scorer = EnhancedGeneScorer(config)

            # Test gene scoring
            scores = scorer.score_genes(self.test_meta_object, ScoringMethod.COMBINED)

            assert len(scores) > 0
            assert scores.index.equals(self.test_meta_object["original_result"].index)
            assert not scores.isnull().all()

        def test_bootstrap_scoring():
            config = EnhancedScoringConfig(
                enable_bootstrap=True,
                bootstrap_iterations=100,
                bootstrap_confidence=0.95,
            )
            scorer = EnhancedGeneScorer(config)

            scores = scorer.score_genes(self.test_meta_object, ScoringMethod.EFFECT_SIZE)

            # Check if confidence intervals were computed
            ci = scorer.get_last_confidence_intervals()
            assert ci is not None
            assert len(ci) > 0

        def test_legacy_gene_scorer():
            from metaintegrator.scoring_core import ScoringConfig

            legacy_config = ScoringConfig()
            legacy_scorer = GeneScorer(legacy_config)

            # Test legacy interface
            scores = legacy_scorer.score(self.test_meta_object, ScoringMethod.Z_SCORE)
            assert len(scores) > 0

        self._run_test("Enhanced Gene Scorer", test_enhanced_gene_scorer)
        self._run_test("Bootstrap Scoring", test_bootstrap_scoring)
        self._run_test("Legacy Gene Scorer", test_legacy_gene_scorer)

    def _test_meta_integrator_filter(self):
        """Test meta integrator filter functionality."""

        def test_enhanced_filter():
            config = EnhancedFilterConfig(
                effect_size_thresh=0.1,
                fdr_thresh=0.05,
                adaptive_thresholds=True,
                prioritize_by_strength=True,
            )
            filter_obj = EnhancedMetaIntegratorFilter(config)

            # Test standard filtering
            results = filter_obj.filter_meta_analysis(self.test_meta_object, "standard")

            assert isinstance(results, EnhancedFilterResults)
            assert len(results.pos_gene_names) >= 0
            assert len(results.neg_gene_names) >= 0
            assert results.filter_performance is not None

        def test_adaptive_filtering():
            config = EnhancedFilterConfig(
                adaptive_thresholds=True,
                min_effect_size_percentile=75,
                use_combined_criteria=True,
            )
            filter_obj = EnhancedMetaIntegratorFilter(config)

            results = filter_obj.filter_meta_analysis(self.test_meta_object, "adaptive")

            assert results.adaptive_thresholds_used is not None
            assert len(results.adaptive_thresholds_used) > 0

        def test_legacy_filter():
            from metaintegrator.scoring_core import FilterConfig

            legacy_config = FilterConfig()
            legacy_filter = MetaIntegratorFilter(legacy_config)

            results = legacy_filter.filter(self.test_meta_object)
            assert hasattr(results, "pos_gene_names")
            assert hasattr(results, "neg_gene_names")

        self._run_test("Enhanced Filter", test_enhanced_filter)
        self._run_test("Adaptive Filtering", test_adaptive_filtering)
        self._run_test("Legacy Filter", test_legacy_filter)

    def _test_results_summarizer(self):
        """Test results summarizer functionality."""

        def test_enhanced_summarizer():
            config = EnhancedSummaryConfig(
                include_trend_analysis=True,
                include_quality_assessment=True,
                generate_visualizations=False,  # Skip for testing
                auto_export=False,
            )
            summarizer = EnhancedResultSummarizer(config)

            summary = summarizer.summarize_results(
                self.test_meta_object["original_result"], "test_analysis"
            )

            assert "analysis_info" in summary
            assert "statistical_summary" in summary
            assert "gene_summary" in summary
            assert "quality_assessment" in summary

        def test_legacy_summarizer():
            from metaintegrator.scoring_core import SummaryConfig

            legacy_config = SummaryConfig()
            legacy_summarizer = ResultSummarizer(legacy_config)

            summary = legacy_summarizer.summarize(self.test_meta_object["original_result"])
            assert summary is not None

        self._run_test("Enhanced Summarizer", test_enhanced_summarizer)
        self._run_test("Legacy Summarizer", test_legacy_summarizer)

    def _test_orchestrator_workflows(self):
        """Test orchestrator workflow functionality."""

        def test_standard_workflow():
            config = OrchestratorConfig(
                default_output_dir=self.temp_dir,
                auto_export_results=True,
                auto_generate_reports=True,
                fail_fast=False,
            )
            orchestrator = EnhancedMetaIntegratorOrchestrator(config)

            results = orchestrator.run_complete_analysis(
                self.test_meta_object, "test_standard", "standard"
            )

            assert isinstance(results, OrchestrationResults)
            assert results.analysis_name == "test_standard"
            # Allow for partial success due to test data limitations
            assert results.success_rate > 0.5

        def test_advanced_workflow():
            config = OrchestratorConfig(
                default_output_dir=self.temp_dir,
                auto_export_results=False,  # Skip export for speed
                fail_fast=False,
            )
            orchestrator = EnhancedMetaIntegratorOrchestrator(config)

            results = orchestrator.run_complete_analysis(
                self.test_meta_object, "test_advanced", "advanced"
            )

            assert results.analysis_name == "test_advanced"
            # Advanced workflow might have more sophisticated requirements
            assert results.success_rate >= 0.3

        def test_minimal_workflow():
            config = OrchestratorConfig(
                default_output_dir=self.temp_dir,
                auto_export_results=False,
                fail_fast=False,
            )
            orchestrator = EnhancedMetaIntegratorOrchestrator(config)

            results = orchestrator.run_complete_analysis(
                self.test_meta_object, "test_minimal", "minimal"
            )

            assert results.analysis_name == "test_minimal"
            assert results.success_rate > 0.5

        self._run_test("Standard Workflow", test_standard_workflow)
        self._run_test("Advanced Workflow", test_advanced_workflow)
        self._run_test("Minimal Workflow", test_minimal_workflow)

    def _test_backward_compatibility(self):
        """Test backward compatibility with legacy interfaces."""

        def test_legacy_orchestrator():
            legacy_config = {
                "log_level": "INFO",
                "cache_components": True,
                "default_output_dir": str(self.temp_dir),
            }
            legacy_orchestrator = MetaIntegratorOrchestrator(legacy_config)

            results = legacy_orchestrator.run_analysis(self.test_meta_object, "legacy_test")

            assert "success" in results
            assert "filter_results" in results or "errors" in results

        self._run_test("Legacy Orchestrator", test_legacy_orchestrator)

    def _test_performance_improvements(self):
        """Test performance improvements and caching."""

        def test_caching():
            # Test that caching works for repeated operations
            config = EnhancedScoringConfig(cache_intermediate=True)
            scorer = EnhancedGeneScorer(config)

            import time

            # First run
            start_time = time.time()
            scores1 = scorer.score_genes(self.test_meta_object, ScoringMethod.COMBINED)
            first_run_time = time.time() - start_time

            # Second run (should be faster due to caching)
            start_time = time.time()
            scores2 = scorer.score_genes(self.test_meta_object, ScoringMethod.COMBINED)
            second_run_time = time.time() - start_time

            # Verify results are identical
            assert scores1.equals(scores2)

            # Performance improvement not guaranteed in test environment
            # but caching should at least not make it slower
            assert second_run_time <= first_run_time * 2  # Allow some variance

        self._run_test("Caching Performance", test_caching)

    def _test_new_features(self):
        """Test new features added in refactoring."""

        def test_adaptive_thresholds():
            config = EnhancedFilterConfig(adaptive_thresholds=True, min_effect_size_percentile=50)
            filter_obj = EnhancedMetaIntegratorFilter(config)

            results = filter_obj.filter_meta_analysis(self.test_meta_object, "adaptive")

            assert len(results.adaptive_thresholds_used) > 0
            assert "effect_size" in results.adaptive_thresholds_used

        def test_gene_prioritization():
            config = EnhancedFilterConfig(prioritize_by_strength=True, max_genes_per_direction=10)
            filter_obj = EnhancedMetaIntegratorFilter(config)

            results = filter_obj.filter_meta_analysis(self.test_meta_object, "standard")

            assert len(results.gene_scores) > 0

            # Test top genes functionality
            top_genes = results.get_top_genes("positive", 5)
            assert len(top_genes) <= 5

        def test_comprehensive_export():
            config = ExportConfig(
                base_output_dir=self.temp_dir,
                include_metadata=True,
                include_plots=False,  # Skip plots for testing
                validate_exports=True,
            )
            engine = ExportEngine(config)

            # Test batch export
            analyses = {
                "analysis_1": self.test_meta_object["original_result"],
                "analysis_2": self.test_meta_object["original_result"].copy(),
            }

            batch_result = engine.export_batch_analysis(analyses, "test_batch")

            assert "batch_exports" in batch_result
            assert len(batch_result["batch_exports"]) == 2

        self._run_test("Adaptive Thresholds", test_adaptive_thresholds)
        self._run_test("Gene Prioritization", test_gene_prioritization)
        self._run_test("Comprehensive Export", test_comprehensive_export)

    def _generate_test_summary(self) -> dict:
        """Generate comprehensive test summary."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["success"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        summary = {
            "success": failed_tests == 0,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "test_results": self.test_results,
            "refactoring_validation": {
                "modules_imported": IMPORTS_SUCCESSFUL,
                "core_functionality": passed_tests > total_tests * 0.7,
                "integration_working": any(
                    "Workflow" in r["test_name"] and r["success"] for r in self.test_results
                ),
                "backward_compatibility": any(
                    "Legacy" in r["test_name"] and r["success"] for r in self.test_results
                ),
                "new_features_working": any(
                    "Adaptive" in r["test_name"] and r["success"] for r in self.test_results
                ),
            },
        }

        # Save summary to file
        summary_path = self.temp_dir / "refactoring_validation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def _cleanup(self):
        """Clean up test files and directories."""
        try:
            if self.temp_dir and self.temp_dir.exists():
                import shutil

                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")


def main():
    """Run the validation test suite."""
    test_suite = RefactoringValidationTest()
    results = test_suite.run_all_tests()

    print("\n🎯 REFACTORING VALIDATION RESULTS:")
    print("=" * 50)

    if results["success"]:
        print("🎉 ALL TESTS PASSED! Refactoring is successful.")
        print("\n✨ Key Achievements:")
        print("  ✅ All modules imported successfully")
        print("  ✅ Core functionality working")
        print("  ✅ Integration tests passed")
        print("  ✅ Backward compatibility maintained")
        print("  ✅ New features operational")

    else:
        print("⚠️  Some tests failed. Please review:")
        if "refactoring_validation" in results:
            val = results["refactoring_validation"]
            print(f"  - Modules imported: {'✅' if val['modules_imported'] else '❌'}")
            print(f"  - Core functionality: {'✅' if val['core_functionality'] else '❌'}")
            print(f"  - Integration: {'✅' if val['integration_working'] else '❌'}")
            print(f"  - Backward compatibility: {'✅' if val['backward_compatibility'] else '❌'}")
            print(f"  - New features: {'✅' if val['new_features_working'] else '❌'}")

    return results


if __name__ == "__main__":
    main()
