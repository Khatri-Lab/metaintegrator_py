"""
Comprehensive unit tests for plotting.py module.
This test suite covers the MetaAnalysisPlotter class and all its methods.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Set matplotlib to non-GUI backend for testing
matplotlib.use("Agg")

import os, sys

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# Import the module under test
from visualization.plotting import MetaAnalysisPlotter


class TestMetaAnalysisPlotter:
    """Test class for MetaAnalysisPlotter."""

    def test_init_default_parameters(self):
        """Test MetaAnalysisPlotter initialization with default parameters."""
        plotter = MetaAnalysisPlotter()
        assert plotter.style == "default"
        assert plotter.figsize == (10, 6)

    def test_init_custom_parameters(self):
        """Test MetaAnalysisPlotter initialization with custom parameters."""
        custom_style = "seaborn"
        custom_figsize = (12, 8)

        with patch("matplotlib.pyplot.style.use") as mock_style:
            plotter = MetaAnalysisPlotter(style=custom_style, figsize=custom_figsize)

        assert plotter.style == custom_style
        assert plotter.figsize == custom_figsize
        mock_style.assert_called_once_with(custom_style)

    def test_init_default_style_no_style_call(self):
        """Test that plt.style.use is not called for default style."""
        with patch("matplotlib.pyplot.style.use") as mock_style:
            plotter = MetaAnalysisPlotter(style="default")

        mock_style.assert_not_called()


class TestPlotEffectSizeDistribution:
    """Test class for plot_effect_size_distribution method."""

    @pytest.fixture
    def plotter(self):
        """Create plotter instance for tests."""
        return MetaAnalysisPlotter()

    @pytest.fixture
    def meta_object_with_effect_sizes(self):
        """Create meta object with effect sizes for testing."""
        return {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {
                        "study1": [1.2, 0.8, -0.5, 2.1, -1.3],
                        "study2": [1.0, 0.6, -0.3, 1.8, -1.1],
                        "study3": [1.1, 0.7, -0.4, 2.0, -1.2],
                    },
                    index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
                )
            }
        }

    def test_plot_effect_size_distribution_density(
        self, plotter, meta_object_with_effect_sizes
    ):
        """Test effect size distribution plot with density curves."""
        fig = plotter.plot_effect_size_distribution(
            meta_object_with_effect_sizes, overlay_density=True
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Effect Size"
        assert ax.get_ylabel() == "Density"
        assert ax.get_title() == "Effect Size Distribution by Dataset"
        assert ax.legend_ is not None

        # Clean up
        plt.close(fig)

    def test_plot_effect_size_distribution_histograms(
        self, plotter, meta_object_with_effect_sizes
    ):
        """Test effect size distribution plot with histograms."""
        fig = plotter.plot_effect_size_distribution(
            meta_object_with_effect_sizes, bins=30, overlay_density=False
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Effect Size"
        assert ax.get_ylabel() == "Density"
        assert ax.get_title() == "Effect Size Distribution by Dataset"

        # Check that we have histogram patches
        assert len(ax.patches) > 0

        # Clean up
        plt.close(fig)

    def test_plot_effect_size_distribution_save(
        self, plotter, meta_object_with_effect_sizes
    ):
        """Test saving effect size distribution plot."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_plot.png"

            fig = plotter.plot_effect_size_distribution(
                meta_object_with_effect_sizes, save_path=str(save_path)
            )

            assert save_path.exists()
            assert save_path.stat().st_size > 0

            # Clean up
            plt.close(fig)

    def test_plot_effect_size_distribution_missing_meta_analysis(self, plotter):
        """Test error handling when metaAnalysis key is missing."""
        invalid_meta_object = {"someOtherKey": "value"}

        with pytest.raises(ValueError, match="Meta-analysis results not found"):
            plotter.plot_effect_size_distribution(invalid_meta_object)

    def test_plot_effect_size_distribution_with_nans(self, plotter):
        """Test handling of NaN values in effect sizes."""
        meta_object_with_nans = {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {
                        "study1": [1.2, np.nan, -0.5],
                        "study2": [np.nan, 0.6, -0.3],
                        "study3": [1.1, 0.7, np.nan],
                    },
                    index=["GENE1", "GENE2", "GENE3"],
                )
            }
        }

        fig = plotter.plot_effect_size_distribution(meta_object_with_nans)

        # Should not raise an error
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)


class TestVolcanoPlot:
    """Test class for volcano plot methods."""

    @pytest.fixture
    def plotter(self):
        """Create plotter instance for tests."""
        return MetaAnalysisPlotter()

    @pytest.fixture
    def meta_object_for_volcano(self):
        """Create meta object for volcano plot testing."""
        return {
            "metaAnalysis": {
                "pooledResults": pd.DataFrame(
                    {
                        "effectSize": [2.1, 0.3, -1.8, 1.2, -0.4],
                        "effectSizeFDR": [0.001, 0.1, 0.005, 0.08, 0.6],
                    },
                    index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
                )
            }
        }

    def test_plot_volcano_basic(self, plotter, meta_object_for_volcano):
        """Test basic volcano plot creation."""
        fig = plotter.plot_volcano(meta_object_for_volcano)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Effect Size"
        assert ax.get_ylabel() == "-log₁₀(FDR)"
        assert ax.get_title() == "Volcano Plot: Effect Size vs. Significance"
        assert ax.legend_ is not None

        # Clean up
        plt.close(fig)

    def test_plot_volcano_custom_thresholds(self, plotter, meta_object_for_volcano):
        """Test volcano plot with custom thresholds."""
        fig = plotter.plot_volcano(
            meta_object_for_volcano, es_threshold=1.0, fdr_threshold=0.01
        )

        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_plot_volcano_no_gene_labels(self, plotter, meta_object_for_volcano):
        """Test volcano plot without gene labels."""
        fig = plotter.plot_volcano(meta_object_for_volcano, top_genes=0)

        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_plot_volcano_save(self, plotter, meta_object_for_volcano):
        """Test saving volcano plot."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "volcano_test.png"

            fig = plotter.plot_volcano(
                meta_object_for_volcano, save_path=str(save_path)
            )

            assert save_path.exists()
            assert save_path.stat().st_size > 0

            # Clean up
            plt.close(fig)

    def test_plot_volcano_with_zero_fdr(self, plotter):
        """Test volcano plot handling of zero FDR values."""
        meta_object_zero_fdr = {
            "metaAnalysis": {
                "pooledResults": pd.DataFrame(
                    {
                        "effectSize": [2.1, 0.3],
                        "effectSizeFDR": [0.0, 0.1],  # Zero FDR value
                    },
                    index=["GENE1", "GENE2"],
                )
            }
        }

        fig = plotter.plot_volcano(meta_object_zero_fdr)

        # Should handle zero FDR values by replacing with small value
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)


class TestForestPlot:
    """Test class for forest plot methods."""

    @pytest.fixture
    def plotter(self):
        """Create plotter instance for tests."""
        return MetaAnalysisPlotter()

    @pytest.fixture
    def meta_object_for_forest(self):
        """Create meta object for forest plot testing."""
        genes = ["BRCA1", "TP53", "MYC", "EGFR", "KRAS"]
        studies = ["study1", "study2", "study3"]

        return {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {
                        "study1": [1.2, 0.8, -0.5, 2.1, -1.3],
                        "study2": [1.0, 0.6, -0.3, 1.8, -1.1],
                        "study3": [1.1, 0.7, -0.4, 2.0, -1.2],
                    },
                    index=genes,
                ),
                "datasetEffectSizeStandardErrors": pd.DataFrame(
                    {
                        "study1": [0.2, 0.1, 0.15, 0.25, 0.18],
                        "study2": [0.18, 0.12, 0.13, 0.22, 0.16],
                        "study3": [0.19, 0.11, 0.14, 0.24, 0.17],
                    },
                    index=genes,
                ),
                "pooledResults": pd.DataFrame(
                    {
                        "effectSize": [1.1, 0.7, -0.4, 1.9, -1.15],
                        "effectSizeStandardError": [0.1, 0.08, 0.09, 0.12, 0.11],
                    },
                    index=genes,
                ),
            }
        }

    def test_plot_forest_plot_basic(self, plotter, meta_object_for_forest):
        """Test basic forest plot creation."""
        genes = ["BRCA1", "TP53", "MYC"]

        fig = plotter.plot_forest_plot(meta_object_for_forest, genes)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Effect Size"
        assert ax.get_title() == "Forest Plot: Effect Sizes Across Studies"
        assert ax.legend_ is not None

        # Clean up
        plt.close(fig)

    def test_plot_forest_plot_max_genes_limit(self, plotter, meta_object_for_forest):
        """Test forest plot with max genes limit."""
        genes = ["BRCA1", "TP53", "MYC", "EGFR", "KRAS"]

        fig = plotter.plot_forest_plot(meta_object_for_forest, genes, max_genes=3)

        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_plot_forest_plot_no_matching_genes(self, plotter, meta_object_for_forest):
        """Test forest plot with no matching genes."""
        genes = ["NONEXISTENT1", "NONEXISTENT2"]

        with pytest.raises(ValueError, match="No matching genes found in results"):
            plotter.plot_forest_plot(meta_object_for_forest, genes)

    def test_plot_forest_plot_save(self, plotter, meta_object_for_forest):
        """Test saving forest plot."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "forest_test.png"
            genes = ["BRCA1", "TP53"]

            fig = plotter.plot_forest_plot(
                meta_object_for_forest, genes, save_path=str(save_path)
            )

            assert save_path.exists()
            assert save_path.stat().st_size > 0

            # Clean up
            plt.close(fig)

    def test_plot_forest_plot_with_missing_data(self, plotter):
        """Test forest plot with missing data (NaN values)."""
        genes = ["GENE1", "GENE2"]
        meta_object_with_nans = {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {
                        "study1": [1.2, np.nan],
                        "study2": [np.nan, 0.6],
                    },
                    index=genes,
                ),
                "datasetEffectSizeStandardErrors": pd.DataFrame(
                    {
                        "study1": [0.2, np.nan],
                        "study2": [np.nan, 0.12],
                    },
                    index=genes,
                ),
                "pooledResults": pd.DataFrame(
                    {
                        "effectSize": [1.1, 0.7],
                        "effectSizeStandardError": [0.1, 0.08],
                    },
                    index=genes,
                ),
            }
        }

        fig = plotter.plot_forest_plot(meta_object_with_nans, genes)

        # Should handle NaN values gracefully
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)


class TestHeatmapPlot:
    """Test class for heatmap plot methods."""

    @pytest.fixture
    def plotter(self):
        """Create plotter instance for tests."""
        return MetaAnalysisPlotter()

    @pytest.fixture
    def meta_object_for_heatmap(self):
        """Create meta object for heatmap testing."""
        genes = ["BRCA1", "TP53", "MYC", "EGFR", "KRAS"]

        return {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {
                        "study1": [1.2, 0.8, -0.5, 2.1, -1.3],
                        "study2": [1.0, 0.6, -0.3, 1.8, -1.1],
                        "study3": [1.1, 0.7, -0.4, 2.0, -1.2],
                        "study4": [0.9, 0.5, -0.2, 1.7, -1.0],
                    },
                    index=genes,
                ),
                "pooledResults": pd.DataFrame(
                    {
                        "effectSizeFDR": [0.001, 0.03, 0.02, 0.0001, 0.005],
                    },
                    index=genes,
                ),
            }
        }

    def test_plot_heatmap_with_specific_genes(self, plotter, meta_object_for_heatmap):
        """Test heatmap with specific genes."""
        genes = ["BRCA1", "TP53", "MYC"]

        fig = plotter.plot_heatmap(meta_object_for_heatmap, genes=genes)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_title() == "Effect Size Heatmap Across Studies"
        assert ax.get_xlabel() == "Studies"
        assert ax.get_ylabel() == "Genes"

        # Clean up
        plt.close(fig)

    def test_plot_heatmap_top_genes(self, plotter, meta_object_for_heatmap):
        """Test heatmap with top genes selection."""
        fig = plotter.plot_heatmap(meta_object_for_heatmap, n_top_genes=3)

        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_plot_heatmap_no_clustering(self, plotter, meta_object_for_heatmap):
        """Test heatmap without clustering."""
        fig = plotter.plot_heatmap(meta_object_for_heatmap, cluster=False)

        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_plot_heatmap_save(self, plotter, meta_object_for_heatmap):
        """Test saving heatmap."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "heatmap_test.png"

            fig = plotter.plot_heatmap(
                meta_object_for_heatmap, save_path=str(save_path)
            )

            assert save_path.exists()
            assert save_path.stat().st_size > 0

            # Clean up
            plt.close(fig)

    def test_plot_heatmap_no_valid_genes(self, plotter):
        """Test heatmap with no valid genes."""
        meta_object_empty = {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(),
                "pooledResults": pd.DataFrame(columns=["effectSizeFDR"]),
            }
        }

        with pytest.raises(ValueError, match="No valid genes found for heatmap"):
            plotter.plot_heatmap(meta_object_empty)

    def test_plot_heatmap_all_nan_genes(self, plotter):
        """Test heatmap with genes that have all NaN values."""
        meta_object_all_nan = {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {
                        "study1": [np.nan, np.nan],
                        "study2": [np.nan, np.nan],
                    },
                    index=["GENE1", "GENE2"],
                ),
                "pooledResults": pd.DataFrame(
                    {
                        "effectSizeFDR": [0.001, 0.02],
                    },
                    index=["GENE1", "GENE2"],
                ),
            }
        }

        # This should raise an error when trying to plot heatmap with all NaN data
        with pytest.raises(ValueError, match="No valid data available for heatmap"):
            plotter.plot_heatmap(meta_object_all_nan, genes=["GENE1", "GENE2"])


class TestStudyOverview:
    """Test class for study overview methods."""

    @pytest.fixture
    def plotter(self):
        """Create plotter instance for tests."""
        return MetaAnalysisPlotter()

    @pytest.fixture
    def meta_object_for_overview(self):
        """Create meta object for study overview testing."""
        return {
            "originalData": {
                "study1": {
                    "class": [0, 0, 1, 1, 1],
                    "keys": ["GENE1", "GENE2", "GENE3", "GENE4"],
                    "formattedName": "Cancer Study 1",
                },
                "study2": {
                    "class": [0, 1, 1, 1, 1, 1],
                    "keys": ["GENE1", "GENE2", "GENE3"],
                    "formattedName": "Validation Study",
                },
                "study3": {
                    "class": [0, 0, 0, 1, 1],
                    "keys": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
                },
            }
        }

    def test_plot_study_overview_basic(self, plotter, meta_object_for_overview):
        """Test basic study overview plot."""
        fig = plotter.plot_study_overview(meta_object_for_overview)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplot grid

        # Check titles of subplots
        titles = [ax.get_title() for ax in fig.axes]
        expected_titles = [
            "Sample Distribution by Study",
            "Total Sample Size by Study",
            "Gene Count by Study",
            "Case/Control Balance by Study",
        ]

        for expected in expected_titles:
            assert expected in titles

        # Clean up
        plt.close(fig)

    def test_plot_study_overview_save(self, plotter, meta_object_for_overview):
        """Test saving study overview plot."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "overview_test.png"

            fig = plotter.plot_study_overview(
                meta_object_for_overview, save_path=str(save_path)
            )

            assert save_path.exists()
            assert save_path.stat().st_size > 0

            # Clean up
            plt.close(fig)

    def test_extract_study_stats(self, plotter, meta_object_for_overview):
        """Test _extract_study_stats method."""
        stats_df = plotter._extract_study_stats(
            meta_object_for_overview["originalData"]
        )

        assert isinstance(stats_df, pd.DataFrame)
        expected_columns = ["Study", "Cases", "Controls", "Total_Samples", "Genes"]
        for col in expected_columns:
            assert col in stats_df.columns

        # Check specific values
        assert len(stats_df) == 3  # Three studies
        assert all(
            stats_df["Total_Samples"] == stats_df["Cases"] + stats_df["Controls"]
        )


class TestMASummary:
    """Test class for meta-analysis summary methods."""

    @pytest.fixture
    def plotter(self):
        """Create plotter instance for tests."""
        return MetaAnalysisPlotter()

    @pytest.fixture
    def meta_object_for_summary(self):
        """Create meta object for summary testing."""
        return {
            "metaAnalysis": {
                "pooledResults": pd.DataFrame(
                    {
                        "effectSize": [1.2, 0.8, -0.5, 2.1, -1.3],
                        "effectSizePval": [0.01, 0.05, 0.02, 0.001, 0.03],
                        "numStudies": [5, 4, 6, 3, 5],
                        "heterogeneityPval": [0.1, 0.05, 0.3, 0.02, 0.8],
                    },
                    index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
                )
            }
        }

    def test_plot_ma_summary_basic(self, plotter, meta_object_for_summary):
        """Test basic meta-analysis summary plot."""
        fig = plotter.plot_ma_summary(meta_object_for_summary)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplot grid

        # Clean up
        plt.close(fig)

    def test_plot_ma_summary_save(self, plotter, meta_object_for_summary):
        """Test saving meta-analysis summary plot."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "summary_test.png"

            fig = plotter.plot_ma_summary(
                meta_object_for_summary, save_path=str(save_path)
            )

            assert save_path.exists()
            assert save_path.stat().st_size > 0

            # Clean up
            plt.close(fig)

    def test_plot_ma_summary_with_missing_columns(self, plotter):
        """Test summary plot with missing columns."""
        incomplete_meta_object = {
            "metaAnalysis": {
                "pooledResults": pd.DataFrame(
                    {
                        "effectSize": [1.2, 0.8],
                        # Missing other required columns
                    },
                    index=["GENE1", "GENE2"],
                )
            }
        }

        # Should raise an error due to missing columns
        with pytest.raises(KeyError):
            plotter.plot_ma_summary(incomplete_meta_object)


class TestHelperMethods:
    """Test class for helper methods."""

    @pytest.fixture
    def plotter(self):
        """Create plotter instance for tests."""
        return MetaAnalysisPlotter()

    def test_set_standard_labels(self, plotter):
        """Test _set_standard_labels helper method."""
        fig, ax = plt.subplots()

        plotter._set_standard_labels(ax, "X Label", "Y Label", "Test Title")

        assert ax.get_xlabel() == "X Label"
        assert ax.get_ylabel() == "Y Label"
        assert ax.get_title() == "Test Title"

        # Clean up
        plt.close(fig)

    def test_format_bar_ax(self, plotter):
        """Test _format_bar_ax helper method."""
        fig, ax = plt.subplots()
        labels = ["Study1", "Study2", "Study3"]

        plotter._format_bar_ax(ax, labels, "Y Label", "Test Title", legend=True)

        assert ax.get_ylabel() == "Y Label"
        assert ax.get_title() == "Test Title"
        assert len(ax.get_xticklabels()) == len(labels)

        # Clean up
        plt.close(fig)

    def test_plot_bar_metric(self, plotter):
        """Test _plot_bar_metric helper method."""
        fig, ax = plt.subplots()
        x = [0, 1, 2]
        values = [10, 20, 15]
        labels = ["A", "B", "C"]

        plotter._plot_bar_metric(ax, x, values, labels, "Y Label", "Test Title", "blue")

        assert len(ax.patches) == len(values)  # Should have bars
        assert ax.get_ylabel() == "Y Label"
        assert ax.get_title() == "Test Title"

        # Clean up
        plt.close(fig)

    def test_select_genes_for_heatmap_with_genes(self, plotter):
        """Test _select_genes_for_heatmap with provided genes."""
        genes = ["GENE1", "GENE3", "GENE5"]
        es_df = pd.DataFrame(index=["GENE1", "GENE2", "GENE3", "GENE4"])
        pooled_df = pd.DataFrame()

        result = plotter._select_genes_for_heatmap(genes, es_df, pooled_df, 10)

        # Should only return genes that exist in es_df
        expected = ["GENE1", "GENE3"]
        assert result == expected

    def test_select_genes_for_heatmap_top_genes(self, plotter):
        """Test _select_genes_for_heatmap with top gene selection."""
        es_df = pd.DataFrame(index=["GENE1", "GENE2", "GENE3"])
        pooled_df = pd.DataFrame(
            {"effectSizeFDR": [0.02, 0.001, 0.05]}, index=["GENE1", "GENE2", "GENE3"]
        )

        result = plotter._select_genes_for_heatmap(None, es_df, pooled_df, 2)

        # Should return top 2 genes by FDR (lowest first)
        expected = ["GENE2", "GENE1"]
        assert result == expected


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def plotter(self):
        """Create plotter instance for tests."""
        return MetaAnalysisPlotter()

    def test_empty_dataframes(self, plotter):
        """Test handling of empty dataframes."""
        empty_meta_object = {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(),
            }
        }

        # Should handle empty data gracefully
        fig = plotter.plot_effect_size_distribution(empty_meta_object)
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_single_study_data(self, plotter):
        """Test handling of single study data."""
        single_study_meta = {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {"study1": [1.2, 0.8, -0.5]}, index=["GENE1", "GENE2", "GENE3"]
                )
            }
        }

        fig = plotter.plot_effect_size_distribution(single_study_meta)
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_very_large_effect_sizes(self, plotter):
        """Test handling of very large effect sizes."""
        large_es_meta = {
            "metaAnalysis": {
                "pooledResults": pd.DataFrame(
                    {
                        "effectSize": [100, -50, 200],
                        "effectSizeFDR": [0.001, 0.1, 0.005],
                    },
                    index=["GENE1", "GENE2", "GENE3"],
                )
            }
        }

        fig = plotter.plot_volcano(large_es_meta)
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_all_identical_values(self, plotter):
        """Test handling of all identical values."""
        identical_meta = {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {
                        "study1": [1.0, 1.0, 1.0],
                        "study2": [1.0, 1.0, 1.0],
                    },
                    index=["GENE1", "GENE2", "GENE3"],
                )
            }
        }

        # Test with histograms instead of density curves to avoid KDE issues
        fig = plotter.plot_effect_size_distribution(
            identical_meta, overlay_density=False
        )
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

        # Also test density curves - should now handle identical values gracefully
        fig2 = plotter.plot_effect_size_distribution(
            identical_meta, overlay_density=True
        )
        assert isinstance(fig2, plt.Figure)

        # Clean up
        plt.close(fig2)


class TestIntegration:
    """Integration tests for the plotting module."""

    @pytest.fixture
    def plotter(self):
        """Create plotter instance for tests."""
        return MetaAnalysisPlotter()

    @pytest.fixture
    def complete_meta_object(self):
        """Create complete meta object for integration tests."""
        genes = ["BRCA1", "TP53", "MYC", "EGFR", "KRAS"]
        studies = ["TCGA", "GEO", "ArrayExpress"]

        return {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {study: np.random.normal(0, 1, len(genes)) for study in studies},
                    index=genes,
                ),
                "datasetEffectSizeStandardErrors": pd.DataFrame(
                    {
                        study: np.random.uniform(0.1, 0.3, len(genes))
                        for study in studies
                    },
                    index=genes,
                ),
                "pooledResults": pd.DataFrame(
                    {
                        "effectSize": np.random.normal(0, 1, len(genes)),
                        "effectSizeStandardError": np.random.uniform(
                            0.1, 0.2, len(genes)
                        ),
                        "effectSizeFDR": np.random.uniform(0.001, 0.1, len(genes)),
                        "effectSizePval": np.random.uniform(0.001, 0.1, len(genes)),
                        "numStudies": np.random.randint(2, 5, len(genes)),
                        "heterogeneityPval": np.random.uniform(0.01, 0.5, len(genes)),
                    },
                    index=genes,
                ),
            },
            "originalData": {
                study: {
                    "class": np.random.choice([0, 1], 50),
                    "keys": genes,
                    "formattedName": f"{study} Dataset",
                }
                for study in studies
            },
        }

    @pytest.mark.integration
    def test_full_plotting_workflow(self, plotter, complete_meta_object):
        """Test complete plotting workflow with all plot types."""
        # Test all major plotting functions
        plots = []

        # Effect size distribution
        fig1 = plotter.plot_effect_size_distribution(complete_meta_object)
        plots.append(fig1)

        # Volcano plot
        fig2 = plotter.plot_volcano(complete_meta_object)
        plots.append(fig2)

        # Forest plot
        fig3 = plotter.plot_forest_plot(complete_meta_object, ["BRCA1", "TP53"])
        plots.append(fig3)

        # Heatmap
        fig4 = plotter.plot_heatmap(complete_meta_object, n_top_genes=3)
        plots.append(fig4)

        # Study overview
        fig5 = plotter.plot_study_overview(complete_meta_object)
        plots.append(fig5)

        # MA summary
        fig6 = plotter.plot_ma_summary(complete_meta_object)
        plots.append(fig6)

        # All should be valid figures
        for fig in plots:
            assert isinstance(fig, plt.Figure)

        # Clean up
        for fig in plots:
            plt.close(fig)

    @pytest.mark.integration
    def test_save_all_plots_workflow(self, plotter, complete_meta_object):
        """Test saving all plot types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)

            # Generate all plots with save paths
            paths_and_figs = [
                (
                    base_path / "effect_dist.png",
                    plotter.plot_effect_size_distribution(
                        complete_meta_object,
                        save_path=str(base_path / "effect_dist.png"),
                    ),
                ),
                (
                    base_path / "volcano.png",
                    plotter.plot_volcano(
                        complete_meta_object, save_path=str(base_path / "volcano.png")
                    ),
                ),
                (
                    base_path / "forest.png",
                    plotter.plot_forest_plot(
                        complete_meta_object,
                        ["BRCA1", "TP53"],
                        save_path=str(base_path / "forest.png"),
                    ),
                ),
                (
                    base_path / "heatmap.png",
                    plotter.plot_heatmap(
                        complete_meta_object, save_path=str(base_path / "heatmap.png")
                    ),
                ),
                (
                    base_path / "overview.png",
                    plotter.plot_study_overview(
                        complete_meta_object, save_path=str(base_path / "overview.png")
                    ),
                ),
                (
                    base_path / "summary.png",
                    plotter.plot_ma_summary(
                        complete_meta_object, save_path=str(base_path / "summary.png")
                    ),
                ),
            ]

            # Check all files were created
            for path, fig in paths_and_figs:
                assert path.exists()
                assert path.stat().st_size > 0
                plt.close(fig)


@pytest.mark.performance
class TestPerformance:
    """Performance tests for plotting functions."""

    @pytest.fixture
    def plotter(self):
        """Create plotter instance for tests."""
        return MetaAnalysisPlotter()

    def test_large_dataset_performance(self, plotter):
        """Test performance with large datasets."""
        import time

        # Create large dataset
        n_genes, n_studies = 1000, 10
        genes = [f"GENE_{i:04d}" for i in range(n_genes)]
        studies = [f"study_{i}" for i in range(n_studies)]

        large_meta_object = {
            "metaAnalysis": {
                "datasetEffectSizes": pd.DataFrame(
                    {study: np.random.normal(0, 1, n_genes) for study in studies},
                    index=genes,
                ),
                "pooledResults": pd.DataFrame(
                    {
                        "effectSize": np.random.normal(0, 1, n_genes),
                        "effectSizeFDR": np.random.uniform(0.001, 0.1, n_genes),
                    },
                    index=genes,
                ),
            }
        }

        # Test effect size distribution (should be fast)
        start_time = time.time()
        fig = plotter.plot_effect_size_distribution(large_meta_object)
        elapsed = time.time() - start_time

        assert elapsed < 10.0  # Should complete in less than 10 seconds
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
