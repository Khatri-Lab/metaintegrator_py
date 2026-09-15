"""
Enhanced unit tests for enhanced_plotting.py module.
These tests provide comprehensive coverage including integration tests,
file I/O operations, performance tests, and visual validation.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import os
import sys
import tempfile
import warnings
from pathlib import Path

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from visualization.enhanced_plotting import (
    DEFAULT_HEATMAP_MAX_GENES,
    EnhancedMetaAnalysisPlotter,
    forest_plot,
    gg_forest_plot,
    heatmap_plot,
    manhattan_plot,
    multiple_prc_plot,
    multiple_roc_plot,
    prc_plot,
    predictive_value_plot,
    roc_plot,
)


class TestApplyEqualUnitAxes:
    """Unit tests for ``EnhancedMetaAnalysisPlotter._apply_equal_unit_axes``.

    Isolated from the full ``roc_plot``/``prc_plot``/``multiple_roc_plot``/
    ``multiple_prc_plot`` tests below (which assert ``get_aspect() == 1.0``
    as one check among many) so the helper's own contract -- equal aspect
    plus evenly-spaced ticks at a configurable step -- has direct,
    dedicated coverage.
    """

    def _axes(self):
        """Build a blank figure/axes pair with unit [0, 1] limits on both axes."""
        fig, ax = plt.subplots()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        return fig, ax

    def test_sets_equal_aspect(self):
        """The helper sets a 1:1 aspect ratio on the axes."""
        fig, ax = self._axes()
        EnhancedMetaAnalysisPlotter._apply_equal_unit_axes(ax)
        assert ax.get_aspect() == 1.0
        plt.close(fig)

    def test_default_step_produces_eleven_ticks(self):
        """With the default 0.1 step, both axes get ticks at 0.0, 0.1, ..., 1.0."""
        fig, ax = self._axes()
        EnhancedMetaAnalysisPlotter._apply_equal_unit_axes(ax)
        assert list(ax.get_xticks()) == [round(0.1 * i, 1) for i in range(11)]
        assert list(ax.get_yticks()) == list(ax.get_xticks())
        plt.close(fig)

    def test_custom_step_is_respected(self):
        """A custom step argument produces ticks spaced accordingly."""
        fig, ax = self._axes()
        EnhancedMetaAnalysisPlotter._apply_equal_unit_axes(ax, step=0.25)
        assert list(ax.get_xticks()) == [0.0, 0.25, 0.5, 0.75, 1.0]
        plt.close(fig)

    def test_does_not_change_axis_limits(self):
        """Setting the aspect ratio and ticks leaves the axis limits untouched."""
        fig, ax = self._axes()
        EnhancedMetaAnalysisPlotter._apply_equal_unit_axes(ax)
        assert ax.get_xlim() == (0, 1)
        assert ax.get_ylim() == (0, 1)
        plt.close(fig)

    def test_callable_as_a_static_method_without_an_instance(self):
        """The four call sites (roc_plot, prc_plot, multiple_roc_plot,
        multiple_prc_plot) invoke this via `self._apply_equal_unit_axes(ax)`;
        confirm it works fine as an instance call regardless of instance state."""
        fig, ax = self._axes()
        plotter = EnhancedMetaAnalysisPlotter()
        plotter._apply_equal_unit_axes(ax)
        assert ax.get_aspect() == 1.0
        plt.close(fig)


class TestEnhancedMetaAnalysisPlotter:
    """Test cases for the EnhancedMetaAnalysisPlotter class."""

    @pytest.fixture
    def plotter(self):
        """Create a EnhancedMetaAnalysisPlotter instance for testing."""
        return EnhancedMetaAnalysisPlotter()

    @pytest.fixture
    def custom_plotter(self):
        """Create a custom EnhancedMetaAnalysisPlotter instance for testing."""
        # Use seaborn-v0_8-darkgrid instead of seaborn (deprecated in matplotlib 3.6+)
        return EnhancedMetaAnalysisPlotter(style="seaborn-v0_8-darkgrid", figsize=(12, 8))

    @pytest.fixture
    def sample_meta_object(self):
        """Create a sample meta object for testing."""
        return {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {
                        "study1": [1.2, 0.8, -0.5, 2.1, -1.3],
                        "study2": [1.0, 0.6, -0.3, 1.8, -1.1],
                        "study3": [1.1, 0.7, -0.4, 2.0, -1.2],
                        "study4": [0.9, 0.5, -0.2, 1.7, -1.0],
                    },
                    index=["gene1", "gene2", "gene3", "gene4", "gene5"],
                ),
                "datasetEffectSizeStandardErrors": pd.DataFrame(
                    {
                        "study1": [0.2, 0.1, 0.15, 0.25, 0.18],
                        "study2": [0.18, 0.12, 0.13, 0.22, 0.16],
                        "study3": [0.19, 0.11, 0.14, 0.24, 0.17],
                        "study4": [0.17, 0.13, 0.12, 0.21, 0.15],
                    },
                    index=["gene1", "gene2", "gene3", "gene4", "gene5"],
                ),
                "pooledResults": pd.DataFrame(
                    {
                        "effectSize": [1.1, 0.7, -0.4, 1.9, -1.15],
                        "effectSizeStandardError": [0.1, 0.08, 0.09, 0.12, 0.11],
                        "effectSizeFDR": [0.001, 0.03, 0.02, 0.0001, 0.005],
                    },
                    index=["gene1", "gene2", "gene3", "gene4", "gene5"],
                ),
            },
            "originalData": {
                "study1": {"formattedName": "Study 1 - Cancer Dataset"},
                "study2": {"formattedName": "Study 2 - Control Dataset"},
                "study3": {"formattedName": "Study 3 - Validation Set"},
                "study4": {"formattedName": "Study 4 - External Cohort"},
            },
        }

    @pytest.fixture
    def sample_filter_object(self):
        """Create a sample filter object for testing."""
        return {"posGeneNames": ["gene1", "gene4"], "negGeneNames": ["gene3", "gene5"]}

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        return {
            "keys": ["gene1", "gene2", "gene3", "gene4", "gene5"],
            "expr": np.array(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],  # gene1
                    [0.5, 1.5, 2.5, 3.5, 4.5],  # gene2
                    [-0.5, -1.0, -1.5, -2.0, -2.5],  # gene3
                    [2.0, 3.0, 4.0, 5.0, 6.0],  # gene4
                    [-1.5, -2.5, -3.5, -4.5, -5.5],  # gene5
                ]
            ),
            "class": [1, 1, 0, 1, 0],
            "formattedName": "Test Dataset",
        }

    @pytest.fixture
    def large_dataset(self):
        """Create a large dataset for performance testing."""
        n_genes, n_samples = 1000, 100
        return {
            "keys": [f"gene{i}" for i in range(n_genes)],
            "expr": np.random.randn(n_genes, n_samples),
            "class": np.random.randint(0, 2, n_samples),
            "formattedName": "Large Test Dataset",
        }

    def test_init_default(self):
        """Test EnhancedMetaAnalysisPlotter initialization with default parameters."""
        plotter = EnhancedMetaAnalysisPlotter()
        assert plotter.style == "default"
        assert plotter.figsize == (10, 6)

    def test_init_custom_params(self, custom_plotter):
        """Test EnhancedMetaAnalysisPlotter initialization with custom parameters."""
        assert custom_plotter.style == "seaborn-v0_8-darkgrid"
        assert custom_plotter.figsize == (12, 8)

    def test_ci_95_multiplier_constant(self, plotter):
        """Test that CI_95_MULTIPLIER is properly set."""
        assert plotter.CI_95_MULTIPLIER == 1.96

    def test_get_meta_data_success(self, plotter, sample_meta_object):
        """Test successful extraction of meta-analysis data."""
        effect_sizes, std_errors, pooled_results = plotter._get_meta_data(sample_meta_object)

        assert isinstance(effect_sizes, pd.DataFrame)
        assert isinstance(std_errors, pd.DataFrame)
        assert isinstance(pooled_results, pd.DataFrame)
        assert effect_sizes.shape == (5, 4)  # 5 genes, 4 studies
        assert list(effect_sizes.index) == ["gene1", "gene2", "gene3", "gene4", "gene5"]
        assert list(effect_sizes.columns) == ["study1", "study2", "study3", "study4"]

    def test_get_meta_data_missing_key(self, plotter):
        """Test _get_meta_data with missing metaAnalysis key."""
        meta_object = {"someOtherKey": "value"}

        with pytest.raises(ValueError, match="Meta-analysis results not found"):
            plotter._get_meta_data(meta_object)

    def test_get_meta_data_empty_meta_analysis(self, plotter):
        """Test _get_meta_data with empty metaAnalysis."""
        meta_object = {"metaAnalysis": {}}

        with pytest.raises(KeyError):
            plotter._get_meta_data(meta_object)

    def test_get_study_names(self, plotter, sample_meta_object):
        """Test extraction of study names."""
        studies = ["study1", "study2", "study3", "study4"]
        names = plotter._get_study_names(sample_meta_object, studies)

        expected = [
            "Study 1 - Cancer Dataset",
            "Study 2 - Control Dataset",
            "Study 3 - Validation Set",
            "Study 4 - External Cohort",
        ]
        assert names == expected

    def test_get_study_names_missing_formatted_name(self, plotter):
        """Test _get_study_names when formatted names are missing."""
        meta_object = {"originalData": {"study1": {}}}
        studies = ["study1", "study2"]
        names = plotter._get_study_names(meta_object, studies)

        assert names == ["study1", "study2"]

    def test_get_study_names_empty_original_data(self, plotter):
        """Test _get_study_names with empty originalData."""
        meta_object = {"originalData": {}}
        studies = ["study1", "study2"]
        names = plotter._get_study_names(meta_object, studies)

        assert names == ["study1", "study2"]

    def test_draw_pooled_diamond(self, plotter):
        """Test drawing pooled diamond shape with real matplotlib objects."""
        fig, ax = plt.subplots(figsize=(8, 6))
        initial_patch_count = len(ax.patches)

        plotter._draw_pooled_diamond(ax, 1.0, 0.2, 5.0, "red")

        # Check that a patch was added
        assert len(ax.patches) == initial_patch_count + 1

        # Check diamond properties
        diamond = ax.patches[-1]
        assert diamond.get_facecolor()[0:3] == (1.0, 0.0, 0.0)  # Red color (RGB)
        plt.close(fig)

    def test_draw_pooled_diamond_custom_ci(self, plotter):
        """Test drawing diamond with custom confidence interval."""
        fig, ax = plt.subplots(figsize=(8, 6))

        plotter._draw_pooled_diamond(ax, 2.0, 0.3, 3.0, "blue", ci_multiplier=2.58)

        diamond = ax.patches[-1]
        vertices = diamond.get_path().vertices

        # Diamond should have 4 vertices
        assert len(vertices) == 5  # matplotlib closes the polygon
        plt.close(fig)

    def test_forest_plot_success(self, plotter, sample_meta_object):
        """Test successful forest plot generation with real plotting."""
        fig = plotter.forest_plot(sample_meta_object, "gene1")

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Check basic plot elements
        assert ax.get_xlabel() == "Standardized Mean Difference (log2 scale)"
        assert ax.get_title() == "gene1"

        # Check that we have the expected number of y-tick labels (studies + pooled)
        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        expected_studies = 4  # Number of studies
        assert len(y_labels) == expected_studies + 1  # +1 for pooled

        plt.close(fig)

    def test_forest_plot_with_custom_colors(self, plotter, sample_meta_object):
        """Test forest plot with custom colors."""
        fig = plotter.forest_plot(
            sample_meta_object,
            "gene2",
            box_color="green",
            whisker_color="purple",
            summary_color="yellow",
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_forest_plot_gene_not_found(self, plotter, sample_meta_object):
        """Test forest plot with non-existent gene."""
        with pytest.raises(ValueError, match="No gene named 'nonexistent' in this metaObject"):
            plotter.forest_plot(sample_meta_object, "nonexistent")

    def test_forest_plot_with_save_path(self, plotter, sample_meta_object):
        """Test forest plot saving functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_forest_plot.png"

            fig = plotter.forest_plot(sample_meta_object, "gene1", save_path=str(save_path))

            assert save_path.exists()
            assert save_path.stat().st_size > 0  # File is not empty
            plt.close(fig)

    def test_gg_forest_plot_single_gene(self, plotter, sample_meta_object):
        """Test gg forest plot with single gene."""
        fig = plotter.gg_forest_plot(sample_meta_object, ["gene1"])

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_gg_forest_plot_multiple_genes(self, plotter, sample_meta_object):
        """Test gg forest plot with multiple genes."""
        genes = ["gene1", "gene2", "gene3"]
        fig = plotter.gg_forest_plot(sample_meta_object, genes, facet_cols=2)

        assert isinstance(fig, plt.Figure)
        # Should have enough subplots for the genes
        assert len(fig.axes) >= len(genes)
        plt.close(fig)

    def test_gg_forest_plot_custom_confidence_level(self, plotter, sample_meta_object):
        """Test gg forest plot with custom confidence level."""
        fig = plotter.gg_forest_plot(sample_meta_object, ["gene1", "gene2"], conf_level=0.99)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_prepare_heatmap_data(self, plotter, sample_meta_object, sample_filter_object):
        """Test heatmap data preparation."""
        result = plotter._prepare_heatmap_data(
            sample_meta_object,
            sample_filter_object,
            display_pooled=True,
            gene_order=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert "Pooled" in result.columns
        expected_genes = set(
            sample_filter_object["posGeneNames"] + sample_filter_object["negGeneNames"]
        )
        assert set(result.index) == expected_genes

    def test_prepare_heatmap_data_no_pooled(
        self, plotter, sample_meta_object, sample_filter_object
    ):
        """Test heatmap data preparation without pooled results."""
        result = plotter._prepare_heatmap_data(
            sample_meta_object,
            sample_filter_object,
            display_pooled=False,
            gene_order=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert "Pooled" not in result.columns

    def test_prepare_heatmap_data_custom_gene_order(
        self, plotter, sample_meta_object, sample_filter_object
    ):
        """Test heatmap data preparation with custom gene order."""
        custom_order = ["gene4", "gene1", "gene3", "gene5"]
        result = plotter._prepare_heatmap_data(
            sample_meta_object,
            sample_filter_object,
            display_pooled=True,
            gene_order=custom_order,
        )

        assert list(result.index) == custom_order

    def test_prepare_heatmap_data_no_genes(self, plotter, sample_meta_object):
        """Test heatmap data preparation with no valid genes."""
        empty_filter = {"posGeneNames": [], "negGeneNames": []}
        result = plotter._prepare_heatmap_data(
            sample_meta_object, empty_filter, display_pooled=True, gene_order=False
        )

        assert result.empty

    def test_prepare_heatmap_data_single_gene(self, plotter, sample_meta_object):
        """Test heatmap data preparation with single gene."""
        single_gene_filter = {"posGeneNames": ["gene1"], "negGeneNames": []}
        result = plotter._prepare_heatmap_data(
            sample_meta_object,
            single_gene_filter,
            display_pooled=True,
            gene_order=False,
        )

        assert result.empty  # Should return empty for < 2 genes

    def test_calculate_heatmap_figsize(self, plotter):
        """Test heatmap figure size calculation."""
        # Test small dataset
        small_data = pd.DataFrame(np.random.rand(3, 2))
        width, height = plotter._calculate_heatmap_figsize(small_data)
        assert width >= 8  # Minimum width
        assert height >= 6  # Minimum height

        # Test large dataset
        large_data = pd.DataFrame(np.random.rand(50, 20))
        width_large, height_large = plotter._calculate_heatmap_figsize(large_data)
        assert width_large > width  # Should be larger
        assert height_large > height

    def test_heatmap_plot_success(self, plotter, sample_meta_object, sample_filter_object):
        """Test successful heatmap plot generation."""
        fig = plotter.heatmap_plot(sample_meta_object, sample_filter_object)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Check that colorbar exists
        assert len(fig.axes) == 2  # Main plot + colorbar

        # Check title
        assert ax.get_title() == "Effect Size Heatmap"

        plt.close(fig)

    def test_heatmap_plot_custom_color_range(
        self, plotter, sample_meta_object, sample_filter_object
    ):
        """Test heatmap plot with custom color range."""
        fig = plotter.heatmap_plot(sample_meta_object, sample_filter_object, color_range=(-2, 2))

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_plot_no_pooled(self, plotter, sample_meta_object, sample_filter_object):
        """Test heatmap plot without pooled results."""
        fig = plotter.heatmap_plot(sample_meta_object, sample_filter_object, display_pooled=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_plot_empty_data(self, plotter, sample_meta_object):
        """Test heatmap plot with empty data."""
        empty_filter = {"posGeneNames": [], "negGeneNames": []}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = plotter.heatmap_plot(sample_meta_object, empty_filter)
            assert len(w) == 1
            assert "Must have at least 2 genes for heatmap" in str(w[0].message)
            assert result is None

    def test_heatmap_plot_large_gene_set(self, plotter):
        """Test heatmap plot with large gene set (>40 genes)."""
        # Create large meta object
        large_genes = [f"gene{i}" for i in range(50)]
        large_meta = {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    np.random.randn(50, 3),
                    index=large_genes,
                    columns=["study1", "study2", "study3"],
                ),
                "datasetEffectSizeStandardErrors": pd.DataFrame(
                    np.abs(np.random.randn(50, 3) * 0.1),
                    index=large_genes,
                    columns=["study1", "study2", "study3"],
                ),
                "pooledResults": pd.DataFrame(
                    {
                        "effectSize": np.random.randn(50),
                        "effectSizeStandardError": np.abs(np.random.randn(50) * 0.1),
                        "effectSizeFDR": np.random.uniform(0, 1, 50),
                    },
                    index=large_genes,
                ),
            },
            "originalData": {},
        }

        large_filter = {
            "posGeneNames": large_genes[:25],
            "negGeneNames": large_genes[25:],
        }

        fig = plotter.heatmap_plot(large_meta, large_filter)
        assert isinstance(fig, plt.Figure)

        # Check that y-labels are hidden for large datasets
        ax = fig.axes[0]
        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        assert len([label for label in y_labels if label]) == 0  # Should be empty

        plt.close(fig)

    @pytest.fixture
    def huge_meta_and_filter(self):
        """A gene set well beyond DEFAULT_HEATMAP_MAX_GENES (100) -- the
        raw/un-optimized scale that used to OOM-kill the process at
        savefig() time (unbounded figure height ~= n_genes * 0.3 inches;
        ~5,000 genes -> ~1,500 inches -> SIGKILL, exit code 137)."""
        n_genes = 5000
        genes = [f"gene{i}" for i in range(n_genes)]
        rng = np.random.RandomState(7)
        meta = {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {"study1": rng.uniform(-2, 2, n_genes), "study2": rng.uniform(-2, 2, n_genes)},
                    index=genes,
                ),
                "pooledResults": pd.DataFrame(
                    {"effectSize": rng.uniform(-2, 2, n_genes)}, index=genes
                ),
            },
            "originalData": {},
        }
        filter_object = {
            "posGeneNames": genes[: n_genes // 2],
            "negGeneNames": genes[n_genes // 2 :],
        }
        return meta, filter_object

    def test_prepare_heatmap_data_caps_to_default_max_genes(self, plotter, huge_meta_and_filter):
        """Regression test for the heatmap OOM crash: an un-capped gene set
        of several thousand genes must be reduced to at most
        DEFAULT_HEATMAP_MAX_GENES rows."""
        meta, filter_object = huge_meta_and_filter

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = plotter._prepare_heatmap_data(meta, filter_object, True, False)
            assert len(result) == DEFAULT_HEATMAP_MAX_GENES
            assert any("dropped" in str(warning.message) for warning in w)

    def test_prepare_heatmap_data_caps_to_custom_max_genes(self, plotter, huge_meta_and_filter):
        """A caller-supplied max_genes value caps the result to that size."""
        meta, filter_object = huge_meta_and_filter
        result = plotter._prepare_heatmap_data(meta, filter_object, True, False, max_genes=25)
        assert len(result) == 25

    def test_prepare_heatmap_data_cap_keeps_highest_effect_size_genes(
        self, plotter, huge_meta_and_filter
    ):
        """The cap should keep the most informative genes (by |effect
        size|), not an arbitrary subset."""
        meta, filter_object = huge_meta_and_filter
        pooled = meta["metaAnalysis"]["pooledResults"]

        result = plotter._prepare_heatmap_data(meta, filter_object, True, False, max_genes=10)

        top_10_by_effect_size = pooled["effectSize"].abs().sort_values(ascending=False).index[:10]
        assert set(result.index) == set(top_10_by_effect_size)

    def test_prepare_heatmap_data_max_genes_none_disables_cap(self, plotter, huge_meta_and_filter):
        """Passing max_genes=None disables the cap entirely, keeping every gene."""
        meta, filter_object = huge_meta_and_filter
        result = plotter._prepare_heatmap_data(meta, filter_object, True, False, max_genes=None)
        assert len(result) == 5000

    def test_prepare_heatmap_data_explicit_gene_order_bypasses_cap(
        self, plotter, huge_meta_and_filter
    ):
        """A caller-supplied gene_order list is an explicit request for
        those exact genes and must not be silently trimmed."""
        meta, filter_object = huge_meta_and_filter
        explicit_order = [f"gene{i}" for i in range(500)]

        result = plotter._prepare_heatmap_data(meta, filter_object, True, explicit_order)

        assert len(result) == 500
        assert list(result.index) == explicit_order

    def test_prepare_heatmap_data_small_gene_set_unaffected_by_cap(
        self, plotter, sample_meta_object, sample_filter_object
    ):
        """A gene set already well under the cap must behave exactly as
        before (no genes dropped, no cap warning)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = plotter._prepare_heatmap_data(
                sample_meta_object, sample_filter_object, True, False
            )
            expected_genes = set(
                sample_filter_object["posGeneNames"] + sample_filter_object["negGeneNames"]
            )
            assert set(result.index) == expected_genes
            assert not any("dropped" in str(warning.message) for warning in w)

    def test_calculate_heatmap_figsize_backstop_clamps_extreme_input(self, plotter):
        """Defense-in-depth: even if _prepare_heatmap_data's gene cap is
        bypassed (max_genes=None or an explicit huge gene_order),
        _calculate_heatmap_figsize must never itself return a figure size
        large enough to reproduce the OOM crash at savefig() time."""
        huge_data = pd.DataFrame(np.random.RandomState(3).randn(20000, 4))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            width, height = plotter._calculate_heatmap_figsize(huge_data)
            assert width <= plotter._MAX_HEATMAP_DIMENSION_INCHES
            assert height <= plotter._MAX_HEATMAP_DIMENSION_INCHES
            assert any("exceeds" in str(warning.message) for warning in w) or any(
                "clamped" in str(getattr(rec, "message", "")) for rec in w
            )

    def test_manhattan_plot(self, plotter, sample_meta_object):
        """Test Manhattan plot generation."""
        fig = plotter.manhattan_plot(sample_meta_object)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        assert ax.get_xlabel() == "Gene Index"
        assert ax.get_ylabel() == "-log₁₀(FDR)"
        assert ax.get_title() == "Manhattan Plot: Effect Size FDR"

        # Check that horizontal line for significance threshold exists
        horizontal_lines = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(horizontal_lines) > 0

        plt.close(fig)

    def test_manhattan_plot_custom_figsize(self, plotter, sample_meta_object):
        """Test Manhattan plot with custom figure size."""
        fig = plotter.manhattan_plot(sample_meta_object, figsize=(16, 8))

        assert fig.get_size_inches()[0] == 16
        assert fig.get_size_inches()[1] == 8
        plt.close(fig)

    def test_get_predictive_plot_title(self, plotter):
        """Test predictive plot title generation."""
        title = plotter._get_predictive_plot_title(0.857, 0.923, 2)
        expected = "Positive and Negative Predictive Value\n85.7% Sensitivity and 92.3% Specificity"
        assert title == expected

    def test_get_predictive_plot_title_different_rounding(self, plotter):
        """Test predictive plot title with different rounding."""
        title = plotter._get_predictive_plot_title(0.857, 0.923, 1)
        expected = "Positive and Negative Predictive Value\n85.7% Sensitivity and 92.3% Specificity"
        assert title == expected

    def test_calculate_predictive_values(self, plotter):
        """Test predictive values calculation."""
        prevalence, ppv, npv = plotter._calculate_predictive_values(0.8, 0.9, 100)

        assert len(prevalence) == 100
        assert len(ppv) == 100
        assert len(npv) == 100
        assert all(0 <= p <= 1 for p in ppv)
        assert all(0 <= n <= 1 for n in npv)

        # Test boundary conditions
        assert prevalence[0] == 0.001
        assert prevalence[-1] == 0.999

        # PPV should increase with prevalence
        assert ppv[-1] > ppv[0]

        # NPV should decrease with prevalence
        assert npv[0] > npv[-1]

    def test_calculate_predictive_values_extreme_cases(self, plotter):
        """Test predictive values with extreme sensitivity/specificity."""
        # Perfect test
        prevalence, ppv, npv = plotter._calculate_predictive_values(1.0, 1.0, 10)
        assert all(p == 1.0 for p in ppv)
        assert all(n == 1.0 for n in npv)

        # Useless test
        prevalence, ppv, npv = plotter._calculate_predictive_values(0.0, 0.0, 10)
        # PPV should approach prevalence when sens=0, spec=0
        # NPV should approach (1-prevalence)

    def test_predictive_value_plot(self, plotter):
        """Test predictive value plot generation."""
        fig = plotter.predictive_value_plot(0.85, 0.90)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        assert ax.get_xlabel() == "Prevalence (%)"
        assert ax.get_ylabel() == "Predictive Value (%)"
        assert "Positive and Negative Predictive Value" in ax.get_title()

        # Should have 2 lines (PPV and NPV)
        assert len(ax.lines) == 2

        # Check axis limits
        assert ax.get_xlim() == (0, 100)
        assert ax.get_ylim() == (0, 100)

        plt.close(fig)

    def test_predictive_value_plot_custom_title(self, plotter):
        """Test predictive value plot with custom title."""
        custom_title = "Custom Predictive Value Analysis"
        fig = plotter.predictive_value_plot(0.85, 0.90, title=custom_title)

        ax = fig.axes[0]
        assert ax.get_title() == custom_title
        plt.close(fig)

    def test_predictive_value_plot_missing_params(self, plotter):
        """Test predictive value plot with missing parameters."""
        with pytest.raises(ValueError, match="Sensitivity and specificity must be provided"):
            plotter.predictive_value_plot(None, 0.9)

        with pytest.raises(ValueError, match="Sensitivity and specificity must be provided"):
            plotter.predictive_value_plot(0.8, None)

    def test_calculate_auc_ci(self, plotter):
        """Test AUC confidence interval calculation."""
        y_true = np.array([0, 0, 1, 1])
        ci_lower, ci_upper = plotter._calculate_auc_ci(y_true, 0.75)

        assert 0 <= ci_lower <= 1
        assert 0 <= ci_upper <= 1
        assert ci_lower <= ci_upper
        assert ci_lower <= 0.75 <= ci_upper

    def test_calculate_auc_ci_edge_cases(self, plotter):
        """Test AUC CI calculation with edge cases."""
        # No positive samples
        y_true_no_pos = np.array([0, 0, 0, 0])
        ci_lower, ci_upper = plotter._calculate_auc_ci(y_true_no_pos, 0.5)
        assert ci_lower == 0.0
        assert ci_upper == 0.0

        # No negative samples
        y_true_no_neg = np.array([1, 1, 1, 1])
        ci_lower, ci_upper = plotter._calculate_auc_ci(y_true_no_neg, 0.5)
        assert ci_lower == 0.0
        assert ci_upper == 0.0

        # Single sample of each class
        y_true_single = np.array([0, 1])
        ci_lower, ci_upper = plotter._calculate_auc_ci(y_true_single, 0.5)
        assert 0 <= ci_lower <= 1
        assert 0 <= ci_upper <= 1

    def test_calculate_roc_data(self, plotter):
        """Test ROC statistics calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])

        fpr, tpr, roc_auc = plotter._calculate_roc_data(y_true, y_scores)

        assert len(fpr) > 0
        assert len(tpr) > 0
        assert len(fpr) == len(tpr)
        assert 0 <= roc_auc <= 1

        # ROC curve should start at (0,0) and end at (1,1)
        assert fpr[0] == 0.0
        assert tpr[0] == 0.0
        assert fpr[-1] == 1.0
        assert tpr[-1] == 1.0

    def test_roc_plot(self, plotter):
        """Test ROC plot generation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.8, 0.7, 0.3])

        fig = plotter.roc_plot(y_true, y_scores)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        assert ax.get_xlabel() == "False Positive Rate (1-Specificity)"
        assert ax.get_ylabel() == "True Positive Rate (Sensitivity)"
        assert ax.get_title() == "ROC Curve"

        # Should have at least 2 lines (ROC curve + diagonal)
        assert len(ax.lines) >= 2

        # Check axis limits
        assert ax.get_xlim() == (0, 1)
        assert ax.get_ylim() == (0, 1)

        # Equal-scaled axes so the plot renders as a true square,
        # not stretched by the figure's own aspect ratio.
        assert ax.get_aspect() == 1.0

        plt.close(fig)

    def test_roc_plot_custom_params(self, plotter):
        """Test ROC plot with custom parameters."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])

        fig = plotter.roc_plot(
            y_true, y_scores, title="Custom ROC", label="Custom Model", color="red"
        )

        ax = fig.axes[0]
        assert ax.get_title() == "Custom ROC"

        # Check that custom label is in legend
        legend_texts = [text.get_text() for text in ax.legend_.texts]
        assert any("Custom Model" in text for text in legend_texts)

        plt.close(fig)

    def test_calculate_auprc_confidence_interval(self, plotter):
        """Test AUPRC confidence interval calculation."""
        y_true = np.array([0, 0, 1, 1])
        ci_lower, ci_upper = plotter._calculate_auprc_confidence_interval(y_true, 0.75)

        assert 0 <= ci_lower <= 1
        assert 0 <= ci_upper <= 1
        assert ci_lower <= ci_upper

    def test_calculate_auprc_confidence_interval_no_positives(self, plotter):
        """Test AUPRC CI with no positive samples."""
        y_true = np.array([0, 0, 0, 0])
        ci_lower, ci_upper = plotter._calculate_auprc_confidence_interval(y_true, 0.5)

        # Should handle gracefully (no division by zero)
        # When there are no positives, return wide confidence interval
        assert ci_lower == 0.0
        assert ci_upper == 1.0

    def test_get_prc_baseline(self, plotter):
        """Test PRC baseline calculation."""
        y_true = np.array([0, 0, 1, 1])
        baseline = plotter._get_prc_baseline(y_true)
        assert baseline == 0.5

        # Test imbalanced dataset
        y_true_imbalanced = np.array([0, 0, 0, 1])
        baseline_imbalanced = plotter._get_prc_baseline(y_true_imbalanced)
        assert baseline_imbalanced == 0.25

    def test_get_prc_baseline_empty(self, plotter):
        """Test PRC baseline with empty array."""
        y_true = np.array([])
        baseline = plotter._get_prc_baseline(y_true)
        assert baseline == 0.0

    def test_calculate_prc_stats(self, plotter):
        """Test PRC statistics calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])

        precision, recall, auprc = plotter._calculate_prc_stats(y_true, y_scores)

        assert len(precision) > 0
        assert len(recall) > 0
        assert 0 <= auprc <= 1

        # Validate that all values are in valid range
        assert all(0 <= p <= 1 for p in precision)
        assert all(0 <= r <= 1 for r in recall)

        # Note: sklearn's precision_recall_curve adds a final point (1.0, 0.0)
        # so recall[-1] is always 0.0 by design

    def test_prc_plot(self, plotter):
        """Test PRC plot generation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.8, 0.7, 0.3])

        fig = plotter.prc_plot(y_true, y_scores)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        assert ax.get_xlabel() == "Recall"
        assert ax.get_ylabel() == "Precision"
        assert ax.get_title() == "Precision-Recall Curve"

        # Should have at least 2 lines (PRC curve + baseline)
        assert len(ax.lines) >= 2

        # Equal-scaled axes, same convention as roc_plot above.
        assert ax.get_aspect() == 1.0

        plt.close(fig)

    def test_prc_plot_custom_params(self, plotter):
        """Test PRC plot with custom parameters."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])

        fig = plotter.prc_plot(
            y_true, y_scores, title="Custom PRC", label="Custom Model", color="green"
        )

        ax = fig.axes[0]
        assert ax.get_title() == "Custom PRC"

        plt.close(fig)

    def test_get_gene_indices(self, plotter):
        """Test gene indices extraction."""
        dataset_keys = ["gene1", "gene2", "gene3", "gene4"]
        target_genes = ["gene1", "gene3"]

        indices = plotter._get_gene_indices(dataset_keys, target_genes)

        assert indices == [0, 2]

    def test_get_gene_indices_no_matches(self, plotter):
        """Test gene indices with no matches."""
        dataset_keys = ["gene1", "gene2"]
        target_genes = ["gene3", "gene4"]

        indices = plotter._get_gene_indices(dataset_keys, target_genes)

        assert indices == []

    def test_get_gene_indices_empty_targets(self, plotter):
        """Test gene indices with empty target genes."""
        dataset_keys = ["gene1", "gene2"]
        target_genes = []

        indices = plotter._get_gene_indices(dataset_keys, target_genes)

        assert indices == []

    def test_get_mean_expression_dataframe(self, plotter):
        """Test mean expression calculation with DataFrame."""
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        indices = [0, 2]

        result = plotter._get_mean_expression(df, indices)

        expected = pd.Series([4.0, 5.0, 6.0])  # Mean of rows 0 and 2
        pd.testing.assert_series_equal(result, expected)

    def test_get_mean_expression_array(self, plotter):
        """Test mean expression calculation with numpy array."""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        indices = [0, 2]

        result = plotter._get_mean_expression(arr, indices)

        expected = np.array([4.0, 5.0, 6.0])  # Mean of rows 0 and 2
        np.testing.assert_array_equal(result, expected)

    def test_get_mean_expression_single_index(self, plotter):
        """Test mean expression with single index."""
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        result = plotter._get_mean_expression(df, [1])
        expected = pd.Series([4.0, 5.0, 6.0])
        pd.testing.assert_series_equal(result, expected)

    def test_get_mean_expression_empty_indices(self, plotter):
        """Test mean expression with empty indices."""
        df = pd.DataFrame([[1, 2, 3]])
        result = plotter._get_mean_expression(df, [])
        assert result == 0.0

    def test_calculate_scores(self, plotter, sample_filter_object, sample_dataset):
        """Test score calculation."""
        scores = plotter._calculate_scores(sample_filter_object, sample_dataset)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 5  # Should match number of samples (columns in expr matrix)

    def test_calculate_scores_no_genes(self, plotter, sample_dataset):
        """Test score calculation with no genes."""
        empty_filter = {"posGeneNames": [], "negGeneNames": []}
        scores = plotter._calculate_scores(empty_filter, sample_dataset)

        # Should return array of zeros with correct shape
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 5
        assert np.allclose(scores, 0.0)

    def test_calculate_scores_delegates_to_score_calculator(
        self, plotter, sample_filter_object, sample_dataset
    ):
        """_calculate_scores() was consolidated onto
        metaintegrator.score_calculation.ScoreCalculator -- the R-parity,
        geometric-mean-based "gold standard" scorer already used by the CLI
        and by violin_plot.py -- instead of independently
        recomputing a simple mean-difference score.

        This is a direct regression test for that consolidation: the two
        must produce identical output for the same inputs, and the result must
        differ from the old mean-difference formula (proving real delegation
        happened, not just a docstring update)."""
        from metaintegrator.score_calculation import ScoreCalculator

        scores = plotter._calculate_scores(sample_filter_object, sample_dataset)

        dataset_as_df = dict(sample_dataset)
        dataset_as_df["expr"] = pd.DataFrame(sample_dataset["expr"])
        expected = ScoreCalculator.calculate_score(
            sample_filter_object, dataset_as_df, suppress_messages=True
        )
        assert np.allclose(scores, expected, equal_nan=True)

        pos_idx = [0, 3]  # gene1, gene4 in sample_dataset's expr matrix
        neg_idx = [2, 4]  # gene3, gene5
        old_mean_difference_scores = sample_dataset["expr"][pos_idx, :].mean(
            axis=0
        ) - sample_dataset["expr"][neg_idx, :].mean(axis=0)
        assert not np.allclose(scores, old_mean_difference_scores), (
            "Scores match the old simple mean-difference formula -- "
            "_calculate_scores() does not appear to be delegating to "
            "ScoreCalculator.calculate_score() as intended."
        )

    def test_calculate_scores_reuses_precomputed_score(self, plotter, sample_dataset):
        """If dataset['score'] is already populated (as it is after CLI
        _compute_scores_and_roc), _calculate_scores() must reuse
        it rather than recomputing -- the same short-circuit already used
        by violin_plot.py's _get_or_compute_scores(). This is what keeps
        ROC/PRC AUCs consistent with the exported scores_<dataset>.csv for
        a given run."""
        cached_scores = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
        dataset_with_score = dict(sample_dataset)
        dataset_with_score["score"] = cached_scores

        result = plotter._calculate_scores(
            {"posGeneNames": [], "negGeneNames": []}, dataset_with_score
        )

        assert np.array_equal(result, cached_scores)

    def test_calculate_scores_accepts_dataframe_expr_directly(
        self, plotter, sample_filter_object, sample_dataset
    ):
        """A dataset whose 'expr' is already a DataFrame (the shape real
        CLI/pipeline datasets use) must work without the ndarray->DataFrame
        conversion path, and must produce the exact same scores as the
        ndarray form of the same data."""
        dataset_df_expr = dict(sample_dataset)
        dataset_df_expr["expr"] = pd.DataFrame(sample_dataset["expr"])

        scores_from_df = plotter._calculate_scores(sample_filter_object, dataset_df_expr)
        scores_from_ndarray = plotter._calculate_scores(sample_filter_object, sample_dataset)

        assert np.allclose(scores_from_df, scores_from_ndarray, equal_nan=True)

    def test_multiple_roc_plot(self, plotter, sample_filter_object):
        """Test multiple ROC plot generation."""
        # Create multiple datasets
        datasets = []
        for i in range(3):
            dataset = {
                "keys": ["gene1", "gene2", "gene3"],
                "expr": np.random.randn(3, 10),
                "class": np.random.randint(0, 2, 10),
                "formattedName": f"Dataset {i+1}",
            }
            datasets.append(dataset)

        fig = plotter.multiple_roc_plot(datasets, sample_filter_object)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        assert ax.get_xlabel() == "False Positive Rate (1-Specificity)"
        assert ax.get_ylabel() == "True Positive Rate (Sensitivity)"

        # Should have multiple ROC curves + diagonal
        assert len(ax.lines) >= len(datasets) + 1

        # Equal-scaled axes, same convention as the single-dataset
        # roc_plot above.
        assert ax.get_aspect() == 1.0

        plt.close(fig)

    def test_multiple_prc_plot(self, plotter, sample_filter_object):
        """Test multiple PRC plot generation."""
        # Create multiple datasets
        datasets = []
        for i in range(3):
            dataset = {
                "keys": ["gene1", "gene2", "gene3"],
                "expr": np.random.randn(3, 10),
                "class": np.random.randint(0, 2, 10),
                "formattedName": f"Dataset {i+1}",
            }
            datasets.append(dataset)

        fig = plotter.multiple_prc_plot(datasets, sample_filter_object)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        assert ax.get_xlabel() == "Recall"
        assert ax.get_ylabel() == "Precision"

        # Should have multiple PRC curves
        assert len(ax.lines) >= len(datasets)

        # Equal-scaled axes, same convention as the single-dataset
        # prc_plot above.
        assert ax.get_aspect() == 1.0

        plt.close(fig)

    def test_get_dataset_roc_info(self, plotter, sample_dataset, sample_filter_object):
        """Test dataset ROC information extraction."""
        fpr, tpr, auc_value, label = plotter._get_dataset_roc_info(
            0, sample_dataset, sample_filter_object
        )

        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)
        assert isinstance(auc_value, float)
        assert isinstance(label, str)
        assert 0 <= auc_value <= 1
        assert "Test Dataset" in label
        assert "AUC=" in label

    def test_get_dataset_prc_info(self, plotter, sample_dataset, sample_filter_object):
        """Test dataset PRC information extraction."""
        precision, recall, auprc, label = plotter._get_dataset_prc_info(
            0, sample_dataset, sample_filter_object
        )

        assert isinstance(precision, np.ndarray)
        assert isinstance(recall, np.ndarray)
        assert isinstance(auprc, float)
        assert isinstance(label, str)
        assert 0 <= auprc <= 1
        assert "Test Dataset" in label
        assert "AUPRC=" in label


class TestConvenienceFunctions:
    """Test cases for the convenience wrapper functions."""

    @pytest.fixture
    def sample_meta_object(self):
        """Create a sample meta object for testing."""
        return {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {"study1": [1.2, 0.8], "study2": [1.0, 0.6]},
                    index=["gene1", "gene2"],
                ),
                "datasetEffectSizeStandardErrors": pd.DataFrame(
                    {"study1": [0.2, 0.1], "study2": [0.18, 0.12]},
                    index=["gene1", "gene2"],
                ),
                "pooledResults": pd.DataFrame(
                    {
                        "effectSize": [1.1, 0.7],
                        "effectSizeStandardError": [0.1, 0.08],
                        "effectSizeFDR": [0.01, 0.03],
                    },
                    index=["gene1", "gene2"],
                ),
            },
            "originalData": {
                "study1": {"formattedName": "Study 1"},
                "study2": {"formattedName": "Study 2"},
            },
        }

    @pytest.fixture
    def sample_filter_object(self):
        """Create a sample filter object for testing."""
        return {"posGeneNames": ["gene1"], "negGeneNames": ["gene2"]}

    def test_forest_plot_function(self, sample_meta_object):
        """Test forest_plot convenience function."""
        # Don't pass title as it's not a parameter of forest_plot method
        fig = forest_plot(sample_meta_object, "gene1")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_gg_forest_plot_function(self, sample_meta_object):
        """Test gg_forest_plot convenience function."""
        fig = gg_forest_plot(sample_meta_object, ["gene1", "gene2"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_plot_function(self, sample_meta_object, sample_filter_object):
        """Test heatmap_plot convenience function."""
        fig = heatmap_plot(sample_meta_object, sample_filter_object)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_manhattan_plot_function(self, sample_meta_object):
        """Test manhattan_plot convenience function."""
        fig = manhattan_plot(sample_meta_object)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_predictive_value_plot_function(self):
        """Test predictive_value_plot convenience function."""
        fig = predictive_value_plot(0.8, 0.9)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiple_roc_plot_function(self, sample_filter_object):
        """Test multiple_roc_plot convenience function."""
        datasets = [
            {
                "keys": ["gene1", "gene2"],
                "expr": np.random.randn(2, 5),
                "class": [0, 1, 0, 1, 1],
                "formattedName": "Test Dataset",
            }
        ]
        fig = multiple_roc_plot(datasets, sample_filter_object)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiple_prc_plot_function(self, sample_filter_object):
        """Test multiple_prc_plot convenience function."""
        datasets = [
            {
                "keys": ["gene1", "gene2"],
                "expr": np.random.randn(2, 5),
                "class": [0, 1, 0, 1, 1],
                "formattedName": "Test Dataset",
            }
        ]
        fig = multiple_prc_plot(datasets, sample_filter_object)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_roc_plot_function(self):
        """Test roc_plot convenience function."""
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.1, 0.9, 0.2, 0.8])
        fig = roc_plot(y_true, y_scores)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_prc_plot_function(self):
        """Test prc_plot convenience function."""
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.1, 0.9, 0.2, 0.8])
        fig = prc_plot(y_true, y_scores)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    @pytest.fixture
    def plotter(self):
        """Create a EnhancedMetaAnalysisPlotter instance for testing."""
        return EnhancedMetaAnalysisPlotter()

    def test_empty_arrays_roc_stats(self):
        """Test ROC stats calculation with empty arrays."""
        plotter = EnhancedMetaAnalysisPlotter()
        y_true = np.array([])
        y_scores = np.array([])

        with pytest.raises((ValueError, IndexError)):
            plotter._calculate_roc_data(y_true, y_scores)

    def test_single_class_roc_stats(self):
        """Test ROC stats with single class data."""
        plotter = EnhancedMetaAnalysisPlotter()
        y_true = np.array([1, 1, 1])
        y_scores = np.array([0.8, 0.9, 0.7])

        # This should handle gracefully or raise appropriate error
        try:
            fpr, tpr, roc_auc = plotter._calculate_roc_data(y_true, y_scores)
            # If it succeeds, check that arrays are valid
            assert len(fpr) > 0
            assert len(tpr) > 0
        except (ValueError, Warning):
            # It's acceptable for this to raise an error
            pass

    def test_mismatched_array_lengths(self):
        """Test with mismatched array lengths."""
        plotter = EnhancedMetaAnalysisPlotter()
        y_true = np.array([0, 1])
        y_scores = np.array([0.1, 0.9, 0.5])  # Different length

        with pytest.raises((ValueError, IndexError)):
            plotter._calculate_roc_data(y_true, y_scores)

    def test_invalid_probability_scores(self):
        """Test with invalid probability scores."""
        plotter = EnhancedMetaAnalysisPlotter()
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([-0.5, 1.5, 0.5, 0.8])  # Outside [0,1] range

        # sklearn ROC is tolerant of scores outside [0,1]
        try:
            fpr, tpr, roc_auc = plotter._calculate_roc_data(y_true, y_scores)
            assert isinstance(roc_auc, float)
            assert 0 <= roc_auc <= 1
        except ValueError:
            pass

    def test_nan_values_in_data(self):
        """Test handling of NaN values in prediction data."""
        plotter = EnhancedMetaAnalysisPlotter()
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.1, np.nan, 0.2, 0.8])  # NaN score

        with pytest.raises((ValueError, TypeError)):
            plotter._calculate_roc_data(y_true, y_scores)

    def test_inf_values_in_data(self):
        """Test handling of infinite values in prediction data."""
        plotter = EnhancedMetaAnalysisPlotter()
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.1, np.inf, 0.2, 0.8])  # Infinite score

        with pytest.raises((ValueError, TypeError)):
            plotter._calculate_roc_data(y_true, y_scores)

    def test_nan_values_in_meta_data(self):
        """Test handling of NaN values in meta-analysis data."""
        plotter = EnhancedMetaAnalysisPlotter()
        meta_object = {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {"study1": [1.2, np.nan], "study2": [np.nan, 0.6]},
                    index=["gene1", "gene2"],
                ),
                "datasetEffectSizeStandardErrors": pd.DataFrame(
                    {"study1": [0.2, 0.1], "study2": [0.18, 0.12]},
                    index=["gene1", "gene2"],
                ),
                "pooledResults": pd.DataFrame(
                    {
                        "effectSize": [1.1, 0.7],
                        "effectSizeStandardError": [0.1, 0.08],
                        "effectSizeFDR": [0.01, 0.03],
                    },
                    index=["gene1", "gene2"],
                ),
            }
        }

        # This should handle NaN values gracefully
        effect_sizes, std_errors, pooled_results = plotter._get_meta_data(meta_object)
        assert effect_sizes.isna().any().any()

    def test_zero_standard_error(self):
        """Test handling of zero standard error values."""
        plotter = EnhancedMetaAnalysisPlotter()
        # Test with valid data but edge case SE
        y_true = np.array([0, 0, 1, 1])
        ci_lower, ci_upper = plotter._calculate_auc_ci(y_true, 0.5)

        assert ci_lower <= ci_upper
        assert 0 <= ci_lower <= 1
        assert 0 <= ci_upper <= 1

    def test_extremely_small_standard_error(self):
        """Test with extremely small standard error."""
        plotter = EnhancedMetaAnalysisPlotter()
        y_true = np.array([0] * 1000 + [1] * 1000)  # Large balanced dataset
        ci_lower, ci_upper = plotter._calculate_auc_ci(y_true, 0.75)

        # With large sample, CI should be narrow
        assert ci_upper - ci_lower < 0.1

    def test_plot_with_invalid_save_path(self, plotter):
        """Test plotting with invalid save path."""
        meta_object = {
            "metaAnalysis": {
                "pooledResults": pd.DataFrame(
                    {"effectSizeFDR": [0.01, 0.03]}, index=["gene1", "gene2"]
                )
            }
        }

        # Try to save to invalid directory
        invalid_path = "/nonexistent/directory/plot.png"

        with pytest.raises(FileNotFoundError):
            fig = plotter.manhattan_plot(meta_object, save_path=invalid_path)


if __name__ == "__main__":
    # Run with specific markers or all tests
    pytest.main([__file__, "-v"])
