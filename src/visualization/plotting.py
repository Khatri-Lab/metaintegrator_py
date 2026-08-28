"""
Plotting functions for MetaIntegrator Python port.
This module handles visualization of meta-analysis results.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import stats


# ------------------------------------------------------------------
# MetaAnalysisPlotter Class
# ------------------------------------------------------------------
class MetaAnalysisPlotter:
    """Class for creating meta-analysis visualizations."""

    def __init__(self, style: str = "default", figsize: Tuple[int, int] = (10, 6)) -> None:
        """Initialize the plotter with a matplotlib style and default figure size.

        Args:
            style: Matplotlib style name to apply. Defaults to "default" (no change).
            figsize: Default figure size (width, height) in inches.
        """
        self.style = style
        self.figsize = figsize
        if style != "default":
            plt.style.use(style)

    def plot_effect_size_distribution(
        self,
        meta_object: Dict,
        bins: int = 50,
        overlay_density: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot the effect size distribution for a meta-analysis dataset.

        Args:
            meta_object: Meta-analysis results, with effect sizes under "metaAnalysis".
            bins: Number of histogram bins, used only when overlay_density is False.
            overlay_density: Overlay smoothed density curves instead of histograms.
            save_path: If given, save the figure to this path.

        Returns:
            The generated matplotlib Figure.

        Raises:
            ValueError: If "metaAnalysis" is missing from meta_object.
        """
        if "metaAnalysis" not in meta_object:
            raise ValueError("Meta-analysis results not found in meta_object")

        es_df = meta_object["metaAnalysis"]["datasetEffectSizes"]
        fig, ax = plt.subplots(figsize=self.figsize)

        # Convert wide format to long format efficiently
        long_df = es_df.melt(var_name="Study", value_name="value").dropna()

        if overlay_density:
            self._draw_density_curves(ax, long_df)
        else:
            self._draw_histograms(ax, long_df, bins)

        ax.set_xlabel("Effect Size")
        ax.set_ylabel("Density")
        ax.set_title("Effect Size Distribution by Dataset")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _draw_density_curves(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Draw a Gaussian KDE curve per study group (falls back to a vertical line
        when a group has too little variance to fit a KDE).

        Args:
            ax: Axes to draw on.
            df: Long-format data with "Study" and "value" columns.
        """
        for study, group in df.groupby("Study"):
            data = group["value"]

            # Skip if no valid data
            if len(data) == 0:
                continue

            # Check if all values are identical (would cause KDE to fail)
            if data.nunique() <= 1:
                # For identical values, draw a vertical line instead
                y_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
                ax.axvline(data.iloc[0], label=study, linewidth=2)
                continue

            try:
                kde = stats.gaussian_kde(data)
                x = np.linspace(data.min(), data.max(), 100)
                ax.plot(x, kde(x), label=study, linewidth=2)
            except np.linalg.LinAlgError:
                # Fallback: draw a vertical line at the data point
                y_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
                ax.axvline(data.iloc[0], label=study, linewidth=2)

    def _draw_histograms(self, ax: plt.Axes, df: pd.DataFrame, bins: int) -> None:
        """Draw a semi-transparent histogram of "value" for each "Study" group.

        Args:
            ax: Axes to draw on.
            df: Long-format data with "Study" and "value" columns.
            bins: Number of histogram bins.
        """
        for study, group in df.groupby("Study"):
            ax.hist(
                group["value"],
                bins=bins,
                alpha=0.6,
                label=study,
                density=True,
                edgecolor="black",
                linewidth=0.5,
            )

    def plot_volcano(
        self,
        meta_object: Dict,
        es_threshold: float = 0.5,
        fdr_threshold: float = 0.05,
        top_genes: int = 20,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot effect size vs. -log10(FDR) for pooled meta-analysis results.

        Args:
            meta_object: Meta-analysis results, with pooled results under
                meta_object["metaAnalysis"]["pooledResults"].
            es_threshold: Effect-size threshold marking significance.
            fdr_threshold: FDR threshold marking significance.
            top_genes: Number of top genes to label by significance; 0 skips labeling.
            save_path: If given, save the figure to this path.

        Returns:
            The generated matplotlib Figure.
        """
        pooled_df = meta_object["metaAnalysis"]["pooledResults"]

        # 1. Prepare Data
        x = pooled_df["effectSize"]
        y = -np.log10(pooled_df["effectSizeFDR"].replace(0, 1e-300))

        fig, ax = plt.subplots(figsize=(10, 8))

        # 2. Plotting Steps
        self._draw_volcano_points(ax, x, y, pooled_df["effectSizeFDR"], es_threshold, fdr_threshold)
        self._add_volcano_thresholds(ax, es_threshold, fdr_threshold)
        if top_genes > 0:
            self._label_top_genes(ax, x, y, top_genes)

        # 3. Finalize Plot
        ax.set_xlabel("Effect Size")
        ax.set_ylabel("-log₁₀(FDR)")
        ax.set_title("Volcano Plot: Effect Size vs. Significance")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _draw_volcano_points(
        self,
        ax: plt.Axes,
        x: pd.Series,
        y: np.ndarray,
        fdr: pd.Series,
        es_thr: float,
        fdr_thr: float,
    ) -> None:
        """Scatter all points in gray, then overlay significant ones colored by direction.

        Args:
            ax: Axes to draw on.
            x: Effect sizes (x-axis).
            y: -log10(FDR) values (y-axis).
            fdr: Raw FDR values used to test significance.
            es_thr: Effect-size threshold for significance.
            fdr_thr: FDR threshold for significance.
        """
        # Draw all points in background
        ax.scatter(x, y, alpha=0.4, s=30, c="gray", edgecolors="none", label="Not Significant")

        # Color significant genes by direction
        sig_mask = (fdr <= fdr_thr) & (np.abs(x) >= es_thr)
        if sig_mask.any():
            up = sig_mask & (x > 0)
            down = sig_mask & (x < 0)
            ax.scatter(x[up], y[up], c="red", s=40, alpha=0.8, label="Up-regulated")
            ax.scatter(x[down], y[down], c="blue", s=40, alpha=0.8, label="Down-regulated")

    def _add_volcano_thresholds(self, ax: plt.Axes, es_thr: float, fdr_thr: float) -> None:
        """Draw dashed threshold lines at +/-es_thr (vertical) and -log10(fdr_thr) (horizontal).

        Args:
            ax: Axes to draw on.
            es_thr: Effect-size threshold.
            fdr_thr: FDR threshold.
        """
        ax.axhline(
            -np.log10(fdr_thr),
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"FDR = {fdr_thr}",
        )
        ax.axvline(es_thr, color="gray", linestyle="--", alpha=0.7)
        ax.axvline(-es_thr, color="gray", linestyle="--", alpha=0.7)

    def _label_top_genes(self, ax: plt.Axes, x: pd.Series, y: pd.Series, count: int) -> None:
        """Annotate the top `count` genes (by y value) with their gene names.

        Args:
            ax: Axes to draw on.
            x: X-coordinates, indexed by gene name.
            y: Y-coordinates, indexed by gene name; used to rank the top genes.
            count: Number of top genes to label.
        """
        top_indices = y.nlargest(count).index
        for gene in top_indices:
            ax.annotate(
                gene,
                (x[gene], y[gene]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

    def plot_forest_plot(
        self,
        meta_object: Dict,
        genes: List[str],
        max_genes: int = 20,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot per-study effect sizes and the pooled estimate for each gene.

        Args:
            meta_object: Meta-analysis data with dataset and pooled effect sizes.
            genes: Genes to include in the plot.
            max_genes: Maximum number of genes to display.
            save_path: If given, save the figure to this path.

        Returns:
            The generated matplotlib Figure.

        Raises:
            ValueError: If none of `genes` are found in the results.
        """
        es_df = meta_object["metaAnalysis"]["datasetEffectSizes"]
        se_df = meta_object["metaAnalysis"]["datasetEffectSizeStandardErrors"]
        pooled_df = meta_object["metaAnalysis"]["pooledResults"]

        # Filter and limit genes
        plot_genes = [g for g in genes[:max_genes] if g in es_df.index]
        if not plot_genes:
            raise ValueError("No matching genes found in results")

        n_genes = len(plot_genes)
        fig, ax = plt.subplots(figsize=(12, max(6, n_genes * 0.5)))

        # Setup aesthetic elements
        colors = plt.cm.Set3(np.linspace(0, 1, es_df.shape[1]))

        # Plot from top to bottom
        for y_pos, gene in enumerate(reversed(plot_genes)):
            self._plot_study_effects(ax, gene, y_pos, es_df, se_df, colors)
            self._plot_pooled_effect(ax, gene, y_pos, pooled_df)

        # Finalize formatting
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_yticks(range(n_genes))
        ax.set_yticklabels(plot_genes)
        ax.set_xlabel("Effect Size")
        ax.set_title("Forest Plot: Effect Sizes Across Studies")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _plot_study_effects(
        self,
        ax: plt.Axes,
        gene: str,
        y_pos: int,
        es_df: pd.DataFrame,
        se_df: pd.DataFrame,
        colors: List,
    ) -> None:
        """Plot one point per study for `gene`, with a 95% CI line from its standard error.

        Args:
            ax: Axes to draw on.
            gene: Gene name (row label into es_df/se_df).
            y_pos: Y position for this gene's row in the forest plot.
            es_df: Effect sizes, genes x studies.
            se_df: Standard errors, genes x studies (same shape as es_df).
            colors: Per-study colors, in the same order as es_df's columns.
        """
        for j, study in enumerate(es_df.columns):
            es, se = es_df.loc[gene, study], se_df.loc[gene, study]
            if pd.notna(es) and pd.notna(se):
                ax.scatter(
                    es,
                    y_pos,
                    color=colors[j],
                    s=60,
                    alpha=0.7,
                    zorder=3,
                    label=study if y_pos == 0 else "",
                )
                ax.hlines(
                    y_pos,
                    es - 1.96 * se,
                    es + 1.96 * se,
                    color=colors[j],
                    alpha=0.7,
                    linewidth=2,
                )

    def _plot_pooled_effect(
        self, ax: plt.Axes, gene: str, y_pos: int, pooled_df: pd.DataFrame
    ) -> None:
        """Plot the pooled effect size and its 95% CI for `gene`, marked in black.

        Args:
            ax: Axes to draw on.
            gene: Gene name (row label into pooled_df).
            y_pos: Y position for this gene's row in the forest plot.
            pooled_df: Pooled results with "effectSize" and "effectSizeStandardError" columns.
        """
        if gene in pooled_df.index:
            es, se = (
                pooled_df.loc[gene, "effectSize"],
                pooled_df.loc[gene, "effectSizeStandardError"],
            )
            if pd.notna(es) and pd.notna(se):
                ax.scatter(
                    es,
                    y_pos,
                    color="black",
                    s=100,
                    marker="D",
                    zorder=4,
                    label="Pooled" if y_pos == 0 else "",
                )
                ax.hlines(
                    y_pos,
                    es - 1.96 * se,
                    es + 1.96 * se,
                    color="black",
                    linewidth=3,
                    alpha=0.8,
                )

    def plot_heatmap(
        self,
        meta_object: Dict,
        genes: Optional[List[str]] = None,
        n_top_genes: int = 50,
        cluster: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a heatmap of effect sizes (genes x studies) for the selected genes.

        Args:
            meta_object: Meta-analysis data with dataset effect sizes and pooled results.
            genes: Genes to include. If None, the top `n_top_genes` by FDR are used.
            n_top_genes: Number of top genes to select when `genes` is None.
            cluster: Currently unused; reserved for hierarchical clustering of rows.
            save_path: If given, save the figure to this path.

        Returns:
            The generated matplotlib Figure.

        Raises:
            ValueError: If no valid genes remain to plot.
        """
        es_df = meta_object["metaAnalysis"]["datasetEffectSizes"]
        pooled_df = meta_object["metaAnalysis"]["pooledResults"]

        # 1. Select and Validate Genes
        plot_genes = self._select_genes_for_heatmap(genes, es_df, pooled_df, n_top_genes)
        if not plot_genes:
            raise ValueError("No valid genes found for heatmap")

        # 2. Prepare Data (drop genes with all NaNs)
        heat_df = es_df.loc[plot_genes].dropna(how="all")

        # Check if we have any data left after dropping all-NaN rows
        if heat_df.empty or heat_df.isna().all().all():
            raise ValueError("No valid data available for heatmap after removing NaN values")

        # 3. Setup Figure
        width = max(8, len(heat_df.columns) * 0.8)
        height = max(6, len(heat_df) * 0.25)
        fig, ax = plt.subplots(figsize=(width, height))

        # 4. Draw Heatmap
        sns.heatmap(
            heat_df,
            center=0,
            cmap="RdBu_r",
            annot=False,
            cbar_kws={"label": "Effect Size"},
            ax=ax,
            xticklabels=True,
            yticklabels=len(heat_df) <= 100,
        )

        ax.set_title("Effect Size Heatmap Across Studies")
        ax.set_xlabel("Studies")
        ax.set_ylabel("Genes")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _select_genes_for_heatmap(
        self,
        genes: Optional[List[str]],
        es_df: pd.DataFrame,
        pooled_df: pd.DataFrame,
        n_top: int,
    ) -> List[str]:
        """Return `genes` filtered to those present in es_df, or the top `n_top` by FDR.

        Args:
            genes: Requested gene list, or None to select by FDR ranking instead.
            es_df: Effect sizes, indexed by gene name.
            pooled_df: Pooled results with an "effectSizeFDR" column, used for ranking.
            n_top: Number of top-ranked genes to select when `genes` is None.

        Returns:
            The selected gene names.
        """
        if genes is not None:
            return [g for g in genes if g in es_df.index]

        # Check if required column exists for sorting
        if "effectSizeFDR" not in pooled_df.columns or pooled_df.empty:
            return []

        # Sort by FDR and take top N
        return pooled_df.sort_values("effectSizeFDR").head(n_top).index.tolist()

    def plot_study_overview(self, meta_object: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """Plot a 2x2 grid of per-study statistics: sample distribution, total samples,
        gene count, and case/control ratio.

        Args:
            meta_object: Meta-analysis data with an "originalData" dict of per-study data.
            save_path: If given, save the figure to this path.

        Returns:
            The generated matplotlib Figure.
        """
        # 1. Data Preparation
        stats_df = self._extract_study_stats(meta_object["originalData"])
        x_indices = range(len(stats_df))
        study_names = stats_df["Study"]

        # 2. Setup Figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Subplot 1: Stacked Distribution (Cases/Controls)
        ax1.bar(x_indices, stats_df["Cases"], alpha=0.7, label="Cases", color="red")
        ax1.bar(
            x_indices,
            stats_df["Controls"],
            bottom=stats_df["Cases"],
            alpha=0.7,
            label="Controls",
            color="blue",
        )
        self._format_bar_ax(
            ax1,
            study_names,
            "Number of Samples",
            "Sample Distribution by Study",
            legend=True,
        )

        # Subplot 2: Total Samples
        self._plot_bar_metric(
            ax2,
            x_indices,
            stats_df["Total_Samples"],
            study_names,
            "Total Samples",
            "Total Sample Size by Study",
            "green",
        )

        # Subplot 3: Gene Count
        self._plot_bar_metric(
            ax3,
            x_indices,
            stats_df["Genes"],
            study_names,
            "Number of Genes",
            "Gene Count by Study",
            "orange",
        )

        # Subplot 4: Case Ratio
        stats_df["Ratio"] = stats_df["Cases"] / stats_df["Total_Samples"]
        self._plot_bar_metric(
            ax4,
            x_indices,
            stats_df["Ratio"],
            study_names,
            "Case Ratio",
            "Case/Control Balance by Study",
            "purple",
        )
        ax4.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Balanced")
        ax4.set_ylim(0, 1)
        ax4.legend()

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _extract_study_stats(self, original_data: Dict) -> pd.DataFrame:
        """Compute case/control counts, sample totals, and gene counts per study.

        Args:
            original_data: Per-study data keyed by study name; each value must have
                "class" and "keys", and may have "formattedName".

        Returns:
            One row per study with Study, Cases, Controls, Total_Samples, and Genes.
        """
        rows = []
        for name, study in original_data.items():
            classes = np.array(study["class"])
            n_cases = np.sum(classes == 1)
            n_controls = np.sum(classes == 0)
            rows.append(
                {
                    "Study": study.get("formattedName", name),
                    "Cases": n_cases,
                    "Controls": n_controls,
                    "Total_Samples": n_cases + n_controls,
                    "Genes": len(study["keys"]),
                }
            )
        return pd.DataFrame(rows)

    def _plot_bar_metric(
        self,
        ax: plt.Axes,
        x: Union[List[int], np.ndarray],
        values: Union[List[float], np.ndarray, pd.Series],
        labels: Union[List[str], np.ndarray, pd.Series],
        ylabel: str,
        title: str,
        color: Union[str, List[str]],
    ) -> None:
        """Draw a bar chart and apply standard axis formatting.

        Args:
            ax: Axes to draw on.
            x: Bar x-coordinates.
            values: Bar heights.
            labels: X-tick labels, one per bar.
            ylabel: Y-axis label.
            title: Chart title.
            color: Bar color, or one color per bar.
        """
        ax.bar(x, values, alpha=0.7, color=color)
        self._format_bar_ax(ax, labels, ylabel, title)

    def _format_bar_ax(
        self,
        ax: plt.Axes,
        labels: List[str],
        ylabel: str,
        title: str,
        legend: bool = False,
    ) -> None:
        """Apply standard x-tick labels, y-label, title, and grid to a bar chart axis.

        Args:
            ax: Axes to format.
            labels: X-tick labels.
            ylabel: Y-axis label.
            title: Chart title.
            legend: Whether to show the legend.
        """
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if legend:
            ax.legend()

    def plot_ma_summary(
        self, meta_object: Dict[str, Any], save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot a 2x2 summary grid: effect-size distribution, volcano-lite, studies-per-gene,
        and heterogeneity p-value distribution.

        Args:
            meta_object: Meta-analysis results, with pooled results under
                meta_object["metaAnalysis"]["pooledResults"].
            save_path: If given, save the figure to this path.

        Returns:
            The generated matplotlib Figure.
        """
        pooled_df: pd.DataFrame = meta_object["metaAnalysis"]["pooledResults"]
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        (ax1, ax2), (ax3, ax4) = axes

        # 1. Effect Size Distribution
        self._plot_es_dist(ax1, pooled_df["effectSize"])

        # 2. Volcano-lite (ES vs P-value)
        self._plot_volcano_lite(ax2, pooled_df["effectSize"], pooled_df["effectSizePval"])

        # 3. Studies per Gene
        self._plot_study_count_dist(ax3, pooled_df["numStudies"])

        # 4. Heterogeneity
        self._plot_heterogeneity(ax4, pooled_df["heterogeneityPval"])

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _plot_es_dist(self, ax: plt.Axes, es_series: pd.Series) -> None:
        """Plot a histogram of effect sizes with a reference line at zero.

        Args:
            ax: Axes to draw on.
            es_series: Effect sizes to plot.
        """
        ax.hist(es_series.dropna(), bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", alpha=0.7)
        self._set_standard_labels(ax, "Effect Size", "Frequency", "Distribution of Effect Sizes")

    def _plot_volcano_lite(self, ax: plt.Axes, es: pd.Series, pvals: pd.Series) -> None:
        """Scatter effect size against -log10(p-value), with no significance styling.

        Args:
            ax: Axes to draw on.
            es: Effect sizes (x-axis).
            pvals: P-values (transformed to -log10 for the y-axis).
        """
        ax.scatter(es, -np.log10(pvals.replace(0, 1e-300)), alpha=0.6, s=20)
        self._set_standard_labels(
            ax, "Effect Size", "-log₁₀(P-value)", "Effect Size vs. Significance"
        )

    def _plot_study_count_dist(self, ax: plt.Axes, num_studies: pd.Series) -> None:
        """Plot a histogram of how many studies each gene appears in.

        Args:
            ax: Axes to draw on.
            num_studies: Study count per gene (NaN values are dropped).
        """
        valid_counts = num_studies.dropna()
        bins = np.arange(1, valid_counts.max() + 2) if not valid_counts.empty else 10
        ax.hist(
            valid_counts,
            bins=bins,
            alpha=0.7,
            color="lightgreen",
            edgecolor="black",
            align="left",
        )
        self._set_standard_labels(
            ax, "Number of Studies", "Number of Genes", "Studies per Gene Distribution"
        )

    def _plot_heterogeneity(self, ax: plt.Axes, pvals: pd.Series) -> None:
        """Plot a histogram of heterogeneity p-values with a p=0.05 reference line.

        Falls back to a "No heterogeneity data" message when `pvals` has no valid values.

        Args:
            ax: Axes to draw on.
            pvals: Heterogeneity p-values, which may include NaN.
        """
        valid_pvals = pvals.dropna()
        if not valid_pvals.empty:
            ax.hist(valid_pvals, bins=50, alpha=0.7, color="salmon", edgecolor="black")
            ax.axvline(0.05, color="red", linestyle="--", alpha=0.7, label="p=0.05")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No heterogeneity data", transform=ax.transAxes, ha="center")
        self._set_standard_labels(
            ax, "Heterogeneity P-value", "Frequency", "Heterogeneity Test Results"
        )

    def _set_standard_labels(self, ax: plt.Axes, x: str, y: str, title: str) -> None:
        """Set x-label, y-label, title, and a light grid on an axis.

        Args:
            ax: Axes to format.
            x: X-axis label.
            y: Y-axis label.
            title: Plot title.
        """
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
