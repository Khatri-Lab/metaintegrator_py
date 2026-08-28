"""
Comprehensive unit tests for power_calculation.py module.
Tests cover all classes, methods, and edge cases for meta-analysis power calculations.
"""

import os
import sys
import warnings
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.power_calculation import (
    MetaAnalysisPowerCalculator,
    PowerAnalysisPlotter,
    PowerAnalysisResult,
    PowerAnalysisSummary,
    calc_meta_power,
    print_power_summary,
)


class TestPowerAnalysisResult:
    """Test the PowerAnalysisResult dataclass."""

    def test_power_analysis_result_creation(self):
        """Test basic creation and attribute access of PowerAnalysisResult."""
        result = PowerAnalysisResult(
            power=0.8,
            effect_size=0.5,
            avg_sample_size=20.0,
            num_studies=5,
            heterogeneity=1.0,
            tail=2,
            variance_within=0.1,
            variance_between=0.05,
            total_variance=0.15,
            lambda_stat=2.5,
        )

        assert result.power == 0.8
        assert result.effect_size == 0.5
        assert result.avg_sample_size == 20.0
        assert result.num_studies == 5
        assert result.heterogeneity == 1.0
        assert result.tail == 2
        assert result.variance_within == 0.1
        assert result.variance_between == 0.05
        assert result.total_variance == 0.15
        assert result.lambda_stat == 2.5


class TestMetaAnalysisPowerCalculator:
    """Test the MetaAnalysisPowerCalculator class."""

    def setup_method(self):
        """Set up test instance before each test."""
        self.calculator = MetaAnalysisPowerCalculator()

    def test_initialization(self):
        """Test calculator initialization."""
        assert hasattr(self.calculator, "HETEROGENEITY_LEVELS")
        assert self.calculator.HETEROGENEITY_LEVELS["small"] == 0.33
        assert self.calculator.HETEROGENEITY_LEVELS["moderate"] == 1.0
        assert self.calculator.HETEROGENEITY_LEVELS["large"] == 3.0

    def test_calc_meta_power_basic(self):
        """Test basic power calculation functionality."""
        result = self.calculator.calc_meta_power(
            effect_size=0.5,
            avg_sample_size=20,
            num_studies=10,
            heterogeneity="moderate",
        )

        assert isinstance(result, PowerAnalysisResult)
        assert 0 <= result.power <= 1
        assert result.effect_size == 0.5
        assert result.avg_sample_size == 20
        assert result.num_studies == 10
        assert result.heterogeneity == 1.0  # moderate = 1.0
        assert result.tail == 2  # default two-tailed

    def test_calc_meta_power_with_parameters(self):
        """Test power calculation with various parameters."""
        result = self.calculator.calc_meta_power(
            effect_size=0.8,
            avg_sample_size=15,
            num_studies=5,
            heterogeneity=0.5,
            tail=1,
            alpha=0.01,
        )

        assert result.effect_size == 0.8
        assert result.avg_sample_size == 15
        assert result.num_studies == 5
        assert result.heterogeneity == 0.5
        assert result.tail == 1

    def test_calc_meta_power_mathematical_consistency(self):
        """Test that power increases with better conditions."""
        # Higher effect size should give higher power
        result1 = self.calculator.calc_meta_power(0.3, 20, 10, "moderate")
        result2 = self.calculator.calc_meta_power(0.8, 20, 10, "moderate")
        assert result2.power > result1.power

        # More studies should give higher power
        result3 = self.calculator.calc_meta_power(0.5, 20, 5, "moderate")
        result4 = self.calculator.calc_meta_power(0.5, 20, 20, "moderate")
        assert result4.power > result3.power

        # Larger sample size should give higher power
        result5 = self.calculator.calc_meta_power(0.5, 10, 10, "moderate")
        result6 = self.calculator.calc_meta_power(0.5, 50, 10, "moderate")
        assert result6.power > result5.power

    def test_resolve_heterogeneity_string(self):
        """Test heterogeneity resolution with string inputs."""
        assert self.calculator._resolve_heterogeneity("small") == 0.33
        assert self.calculator._resolve_heterogeneity("moderate") == 1.0
        assert self.calculator._resolve_heterogeneity("large") == 3.0
        assert self.calculator._resolve_heterogeneity("SMALL") == 0.33  # case insensitive

    def test_resolve_heterogeneity_numeric(self):
        """Test heterogeneity resolution with numeric inputs."""
        assert self.calculator._resolve_heterogeneity(0.5) == 0.5
        assert self.calculator._resolve_heterogeneity(1.5) == 1.5
        assert self.calculator._resolve_heterogeneity(0.0) == 0.0

    def test_resolve_heterogeneity_invalid_string(self):
        """Test heterogeneity resolution with invalid string."""
        with pytest.raises(ValueError, match="Invalid level"):
            self.calculator._resolve_heterogeneity("invalid")

    def test_get_critical_value(self):
        """Test critical value calculation."""
        # Two-tailed test, alpha=0.05
        critical = self.calculator._get_critical_value(2, 0.05)
        assert abs(critical - 1.96) < 0.01  # approximately 1.96

        # One-tailed test, alpha=0.05
        critical = self.calculator._get_critical_value(1, 0.05)
        assert abs(critical - 1.645) < 0.01  # approximately 1.645

        # Two-tailed test, alpha=0.01
        critical = self.calculator._get_critical_value(2, 0.01)
        assert abs(critical - 2.576) < 0.01  # approximately 2.576

    def test_check_positive(self):
        """Test positive value checking."""
        # Valid positive values should not raise
        self.calculator._check_positive("test", 1.0)
        self.calculator._check_positive("test", 0.001)

        # Zero and negative values should raise
        with pytest.raises(ValueError, match="test must be positive"):
            self.calculator._check_positive("test", 0.0)

        with pytest.raises(ValueError, match="test must be positive"):
            self.calculator._check_positive("test", -1.0)

    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        # Should not raise with valid inputs
        self.calculator._validate_inputs(0.5, 20, 10, "moderate", 2, 0.05)
        self.calculator._validate_inputs(1.0, 15.5, 5, 0.8, 1, 0.01)

    def test_validate_inputs_invalid_effect_size(self):
        """Test input validation with invalid effect size."""
        with pytest.raises(ValueError, match="Effect size must be positive"):
            self.calculator._validate_inputs(0, 20, 10, "moderate", 2, 0.05)

        with pytest.raises(ValueError, match="Effect size must be positive"):
            self.calculator._validate_inputs(-0.5, 20, 10, "moderate", 2, 0.05)

    def test_validate_inputs_invalid_sample_size(self):
        """Test input validation with invalid sample size."""
        with pytest.raises(ValueError, match="Average sample size must be positive"):
            self.calculator._validate_inputs(0.5, 0, 10, "moderate", 2, 0.05)

        with pytest.raises(ValueError, match="Average sample size must be positive"):
            self.calculator._validate_inputs(0.5, -20, 10, "moderate", 2, 0.05)

    def test_validate_inputs_invalid_num_studies(self):
        """Test input validation with invalid number of studies."""
        with pytest.raises(ValueError, match="Number of studies must be positive"):
            self.calculator._validate_inputs(0.5, 20, 0, "moderate", 2, 0.05)

        with pytest.raises(ValueError, match="Number of studies must be positive"):
            self.calculator._validate_inputs(0.5, 20, -5, "moderate", 2, 0.05)

    def test_validate_inputs_invalid_heterogeneity(self):
        """Test input validation with invalid heterogeneity."""
        with pytest.raises(ValueError, match="Heterogeneity must be non-negative"):
            self.calculator._validate_inputs(0.5, 20, 10, -1.0, 2, 0.05)

    def test_validate_inputs_invalid_tail(self):
        """Test input validation with invalid tail parameter."""
        with pytest.raises(
            ValueError, match="Tail must be 1 \\(one-tailed\\) or 2 \\(two-tailed\\)"
        ):
            self.calculator._validate_inputs(0.5, 20, 10, "moderate", 3, 0.05)

        with pytest.raises(
            ValueError, match="Tail must be 1 \\(one-tailed\\) or 2 \\(two-tailed\\)"
        ):
            self.calculator._validate_inputs(0.5, 20, 10, "moderate", 0, 0.05)

    def test_validate_inputs_invalid_alpha(self):
        """Test input validation with invalid alpha."""
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1 \\(exclusive\\)"):
            self.calculator._validate_inputs(0.5, 20, 10, "moderate", 2, 0.0)

        with pytest.raises(ValueError, match="Alpha must be between 0 and 1 \\(exclusive\\)"):
            self.calculator._validate_inputs(0.5, 20, 10, "moderate", 2, 1.0)

        with pytest.raises(ValueError, match="Alpha must be between 0 and 1 \\(exclusive\\)"):
            self.calculator._validate_inputs(0.5, 20, 10, "moderate", 2, 1.5)

    def test_power_analysis_grid(self):
        """Test power analysis grid generation."""
        effect_sizes = [0.2, 0.5, 0.8]
        sample_sizes = [10, 20, 30]
        num_studies_list = [5, 10]

        df = self.calculator.power_analysis_grid(effect_sizes, sample_sizes, num_studies_list)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(effect_sizes) * len(sample_sizes) * len(num_studies_list)
        assert all(
            col in df.columns
            for col in [
                "effect_size",
                "avg_sample_size",
                "num_studies",
                "heterogeneity",
                "power",
                "lambda_stat",
            ]
        )

        # Check that all combinations are present
        assert set(df["effect_size"].unique()) == set(effect_sizes)
        assert set(df["avg_sample_size"].unique()) == set(sample_sizes)
        assert set(df["num_studies"].unique()) == set(num_studies_list)

    def test_power_analysis_grid_with_heterogeneity(self):
        """Test power analysis grid with different heterogeneity settings."""
        df = self.calculator.power_analysis_grid([0.5], [20], [10], heterogeneity="large")

        assert df.iloc[0]["heterogeneity"] == 3.0

    def test_power_analysis_grid_error_handling(self):
        """Test power analysis grid handles errors gracefully."""
        with patch("builtins.print") as mock_print:
            # Use invalid parameters that would cause an error
            df = self.calculator.power_analysis_grid(
                [-0.5], [20], [10]  # negative effect size should cause error
            )

            # Should print error but continue
            mock_print.assert_called()
            assert len(df) == 0  # No valid results

    def test_find_required_studies(self):
        """Test finding required number of studies."""
        result = self.calculator.find_required_studies(
            effect_size=0.5,
            avg_sample_size=20,
            target_power=0.8,
            heterogeneity="moderate",
        )

        assert "required_studies" in result
        assert "target_power" in result
        assert "achieved_power" in result
        assert "study_counts" in result
        assert "power_curve" in result

        assert result["target_power"] == 0.8
        assert isinstance(result["study_counts"], list)
        assert isinstance(result["power_curve"], list)
        assert len(result["study_counts"]) == len(result["power_curve"])

    def test_find_required_studies_target_not_achieved(self):
        """Test finding required studies when target not achievable."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = self.calculator.find_required_studies(
                effect_size=0.01,  # Very small effect size
                avg_sample_size=5,  # Small sample size
                target_power=0.9,  # High target power
                max_studies=10,  # Low max studies
            )

            assert len(w) == 1
            assert "Target power" in str(w[0].message)
            assert result["required_studies"] is None

    def test_find_required_sample_size(self):
        """Test finding required sample size."""
        result = self.calculator.find_required_sample_size(
            effect_size=0.5, num_studies=10, target_power=0.8
        )

        assert "required_sample_size" in result
        assert "target_power" in result
        assert "achieved_power" in result
        assert "sample_sizes" in result
        assert "power_curve" in result

        assert result["target_power"] == 0.8

    def test_find_required_sample_size_target_not_achieved(self):
        """Test finding required sample size when target not achievable."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = self.calculator.find_required_sample_size(
                effect_size=0.01,  # Very small effect size
                num_studies=2,  # Few studies
                target_power=0.9,  # High target power
                max_sample_size=10,  # Small max sample size
            )

            assert len(w) == 1
            assert "Target power" in str(w[0].message)
            assert result["required_sample_size"] is None

    def test_find_detectable_effect_size(self):
        """Test finding minimum detectable effect size."""
        result = self.calculator.find_detectable_effect_size(
            avg_sample_size=20, num_studies=10, target_power=0.8
        )

        assert "min_detectable_effect_size" in result
        assert "target_power" in result
        assert "achieved_power" in result
        assert "effect_sizes" in result
        assert "power_curve" in result

        assert result["target_power"] == 0.8

        # Effect size should be positive if found
        if result["min_detectable_effect_size"] is not None:
            assert result["min_detectable_effect_size"] > 0

    def test_find_detectable_effect_size_target_not_achieved(self):
        """Test finding detectable effect size when target not achievable."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = self.calculator.find_detectable_effect_size(
                avg_sample_size=2,  # Very small sample size
                num_studies=2,  # Few studies
                target_power=0.99,  # Very high target power
                max_effect_size=0.1,  # Very small max effect size
            )

            assert len(w) == 1
            assert "Target power" in str(w[0].message)
            assert result["min_detectable_effect_size"] is None


class TestPowerAnalysisPlotter:
    """Test the PowerAnalysisPlotter class."""

    def setup_method(self):
        """Set up test instance before each test."""
        self.plotter = PowerAnalysisPlotter()

    def test_initialization(self):
        """Test plotter initialization."""
        assert self.plotter.figsize == (10, 6)

        # Test custom figsize
        plotter2 = PowerAnalysisPlotter(figsize=(12, 8))
        assert plotter2.figsize == (12, 8)

    def test_format_label(self):
        """Test label formatting."""
        assert self.plotter._format_label("effect_size") == "Effect Size"
        assert self.plotter._format_label("avg_sample_size") == "Avg Sample Size"
        assert self.plotter._format_label("num_studies") == "Num Studies"
        assert self.plotter._format_label("simple") == "Simple"

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_power_curve_studies(self, mock_savefig, mock_tight_layout, mock_subplots):
        """Test plotting power curve for studies."""
        # Mock matplotlib objects
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        power_data = {
            "study_counts": [1, 2, 3, 4, 5],
            "power_curve": [0.2, 0.4, 0.6, 0.8, 0.9],
            "target_power": 0.8,
            "required_studies": 4,
        }

        fig = self.plotter.plot_power_curve(power_data, "studies")

        # Verify matplotlib calls
        mock_subplots.assert_called_once()
        mock_ax.plot.assert_called()
        mock_ax.axhline.assert_called()
        mock_ax.axvline.assert_called()
        mock_ax.set_xlabel.assert_called_with("Number of Studies")
        mock_ax.set_ylabel.assert_called_with("Statistical Power")
        mock_tight_layout.assert_called_once()

        assert fig == mock_fig

    @patch("matplotlib.pyplot.subplots")
    def test_plot_power_curve_sample_size(self, mock_subplots):
        """Test plotting power curve for sample size."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        power_data = {
            "sample_sizes": [10, 20, 30, 40, 50],
            "power_curve": [0.3, 0.5, 0.7, 0.85, 0.95],
            "target_power": 0.8,
            "required_sample_size": 40,
        }

        self.plotter.plot_power_curve(power_data, "sample_size")

        mock_ax.set_xlabel.assert_called_with("Sample Size per Group")

    @patch("matplotlib.pyplot.subplots")
    def test_plot_power_curve_effect_size(self, mock_subplots):
        """Test plotting power curve for effect size."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        power_data = {
            "effect_sizes": [0.1, 0.3, 0.5, 0.7, 0.9],
            "power_curve": [0.2, 0.4, 0.6, 0.8, 0.95],
            "target_power": 0.8,
            "min_detectable_effect_size": 0.7,
        }

        self.plotter.plot_power_curve(power_data, "effect_size")

        mock_ax.set_xlabel.assert_called_with("Effect Size")

    def test_plot_power_curve_invalid_x_var(self):
        """Test plotting with invalid x_var."""
        power_data = {"power_curve": [0.5, 0.8]}

        with pytest.raises(ValueError, match="x_var must be one of"):
            self.plotter.plot_power_curve(power_data, "invalid")

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_power_curve_with_save(self, mock_savefig, mock_subplots):
        """Test plotting with save functionality."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        power_data = {
            "study_counts": [1, 2, 3],
            "power_curve": [0.3, 0.6, 0.8],
            "target_power": 0.8,
            "required_studies": 3,
        }

        self.plotter.plot_power_curve(power_data, "studies", save_path="/tmp/test.png")

        mock_savefig.assert_called_once_with("/tmp/test.png", dpi=300, bbox_inches="tight")

    @patch("seaborn.heatmap")
    @patch("matplotlib.pyplot.subplots")
    def test_plot_power_grid(self, mock_subplots, mock_heatmap):
        """Test plotting power grid."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Create test dataframe
        power_grid = pd.DataFrame(
            {
                "effect_size": [0.2, 0.2, 0.5, 0.5],
                "num_studies": [5, 10, 5, 10],
                "power": [0.3, 0.5, 0.6, 0.8],
            }
        )

        fig = self.plotter.plot_power_grid(power_grid, "effect_size", "num_studies")

        # Verify seaborn heatmap was called
        mock_heatmap.assert_called_once()
        assert fig == mock_fig


class TestPowerAnalysisSummary:
    """Test the PowerAnalysisSummary class."""

    def setup_method(self):
        """Set up test instance before each test."""
        self.summary = PowerAnalysisSummary()

    def test_initialization(self):
        """Test summary initialization."""
        assert isinstance(self.summary.calculator, MetaAnalysisPowerCalculator)
        assert isinstance(self.summary.plotter, PowerAnalysisPlotter)

    @patch("os.makedirs")
    def test_comprehensive_power_analysis_basic(self, mock_makedirs):
        """Test comprehensive power analysis without plots."""
        results = self.summary.comprehensive_power_analysis(
            effect_size=0.5, avg_sample_size=20, num_studies=10
        )

        assert "main_analysis" in results
        assert "studies_analysis" in results
        assert "sample_size_analysis" in results
        assert "effect_size_analysis" in results

        assert isinstance(results["main_analysis"], PowerAnalysisResult)
        assert isinstance(results["studies_analysis"], dict)
        assert isinstance(results["sample_size_analysis"], dict)
        assert isinstance(results["effect_size_analysis"], dict)

    @patch("os.makedirs")
    @patch.object(PowerAnalysisPlotter, "plot_power_curve")
    def test_comprehensive_power_analysis_with_plots(self, mock_plot, mock_makedirs):
        """Test comprehensive power analysis with plot generation."""
        mock_plot.return_value = MagicMock()

        results = self.summary.comprehensive_power_analysis(
            effect_size=0.5, avg_sample_size=20, num_studies=10, output_dir="/tmp/test"
        )

        assert "plots" in results
        assert len(results["plots"]) == 3

        # Verify directory creation
        mock_makedirs.assert_called_once_with("/tmp/test", exist_ok=True)

        # Verify plot calls
        assert mock_plot.call_count == 3


class TestStandaloneFunctions:
    """Test standalone convenience functions."""

    def test_calc_meta_power_function(self):
        """Test standalone calc_meta_power function."""
        power = calc_meta_power(
            effect_size=0.5,
            avg_sample_size=20,
            num_studies=10,
            heterogeneity="moderate",
        )

        assert isinstance(power, float)
        assert 0 <= power <= 1

    def test_calc_meta_power_function_with_tails(self):
        """Test calc_meta_power function with tails parameter."""
        power = calc_meta_power(
            effect_size=0.5,
            avg_sample_size=20,
            num_studies=10,
            heterogeneity="moderate",
            tails=1,  # Note: function uses 'tails' not 'tail'
        )

        assert isinstance(power, float)
        assert 0 <= power <= 1

    @patch("builtins.print")
    def test_print_power_summary_basic(self, mock_print):
        """Test basic power summary printing."""
        calculator = MetaAnalysisPowerCalculator()
        main_analysis = calculator.calc_meta_power(0.5, 20, 10, "moderate")

        results = {"main_analysis": main_analysis}

        print_power_summary(results)

        # Verify print was called
        assert mock_print.call_count > 0

    @patch("builtins.print")
    def test_print_power_summary_with_additional_analyses(self, mock_print):
        """Test power summary printing with additional analyses."""
        calculator = MetaAnalysisPowerCalculator()
        main_analysis = calculator.calc_meta_power(0.5, 20, 10, "moderate")
        studies_analysis = calculator.find_required_studies(0.5, 20)

        results = {"main_analysis": main_analysis, "studies_analysis": studies_analysis}

        print_power_summary(results)

        # Verify print was called multiple times for comprehensive summary
        assert mock_print.call_count > 5


class TestIntegration:
    """Integration tests across components."""

    def test_full_workflow(self):
        """Test a complete workflow from calculation to plotting."""
        calculator = MetaAnalysisPowerCalculator()
        plotter = PowerAnalysisPlotter()

        # Perform power calculation
        result = calculator.calc_meta_power(0.5, 20, 10, "moderate")
        assert isinstance(result, PowerAnalysisResult)

        # Find required studies
        studies_result = calculator.find_required_studies(0.5, 20)
        assert isinstance(studies_result, dict)
        assert "power_curve" in studies_result

        # Test that plotting doesn't crash (though we won't actually display)
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            fig = plotter.plot_power_curve(studies_result, "studies")
            assert fig == mock_fig

    def test_power_analysis_grid_integration(self):
        """Test integration between grid analysis and plotting."""
        calculator = MetaAnalysisPowerCalculator()
        plotter = PowerAnalysisPlotter()

        # Create power grid
        grid_df = calculator.power_analysis_grid(
            effect_sizes=[0.3, 0.5], sample_sizes=[15, 25], num_studies_list=[5, 10]
        )

        assert len(grid_df) == 8  # 2 * 2 * 2 combinations

        # Test plotting grid
        with (
            patch("seaborn.heatmap") as mock_heatmap,
            patch("matplotlib.pyplot.subplots") as mock_subplots,
        ):
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            fig = plotter.plot_power_grid(grid_df, "effect_size", "num_studies")
            mock_heatmap.assert_called_once()

    def test_edge_case_very_small_effect(self):
        """Test behavior with very small effect sizes."""
        calculator = MetaAnalysisPowerCalculator()

        result = calculator.calc_meta_power(0.01, 1000, 100, "small")
        assert 0 <= result.power <= 1
        assert result.effect_size == 0.01

    def test_edge_case_very_large_effect(self):
        """Test behavior with very large effect sizes."""
        calculator = MetaAnalysisPowerCalculator()

        result = calculator.calc_meta_power(3.0, 10, 5, "large")
        assert 0 <= result.power <= 1
        assert result.effect_size == 3.0

    def test_edge_case_single_study(self):
        """Test behavior with single study."""
        calculator = MetaAnalysisPowerCalculator()

        result = calculator.calc_meta_power(0.5, 20, 1, "moderate")
        assert 0 <= result.power <= 1
        assert result.num_studies == 1

    def test_consistency_across_methods(self):
        """Test that different methods give consistent results."""
        calculator = MetaAnalysisPowerCalculator()

        # Direct calculation
        direct_result = calculator.calc_meta_power(0.5, 20, 10, "moderate")

        # Using convenience function
        convenience_result = calc_meta_power(0.5, 20, 10, "moderate")

        # Should be very close (within floating point precision)
        assert abs(direct_result.power - convenience_result) < 1e-10


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
