"""
Plotting functions for MetaIntegrator Python port.
This module handles visualization of meta-analysis results.
"""

from typing import Dict, List, Optional, Union, Tuple, Any

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

    def __init__(self, style: str = "default", figsize: Tuple[int, int] = (10, 6)):
        """
        Initializes the object with specified plot style and figure size.

        Args:
            style (str): The style of the plot. Defaults to "default".
            figsize (Tuple[int, int]): The size of the figure in inches as a tuple
                representing (width, height). Defaults to (10, 6).

        Args:
            style (str): The initial style to apply to the plot. Defaults to "default".
            figsize (Tuple[int, int]): A tuple specifying the figure size.
                Defaults to (10, 6).
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
        """
        Plot the effect size distribution for a meta-analysis dataset.

        This method visualizes the distribution of effect sizes from a meta-analysis
        dataset contained within the provided meta_object dictionary. The distribution
        can be displayed as histograms or overlaid with density curves.
        The generated plot includes labeled axes, a legend, a title, and a grid.
        Optionally, the plot can be saved to a specified file path.

        Args:
            meta_object: Dict
                A dictionary containing meta-analysis results, including effect sizes
                stored under the "metaAnalysis" key as a dataframe.
            bins: int, default=50
                The number of bins to use for plotting histograms if overlay_density
                is set to False.
            overlay_density: bool, default=True
                If True, overlay density curves on the plot. Otherwise, plot histograms
                for effect size distribution.
            save_path: Optional[str], default=None
                The file path to save the generated plot. If None, the plot is not saved.

        Returns:
            plt.Figure
                The matplotlib Figure object for the generated plot.

        Raises:
            ValueError
                If the "metaAnalysis" key is not found in the provided meta_object.
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

    def _draw_density_curves(self, ax: plt.Axes, df: pd.DataFrame):
        """
        Draw density curves for each study group in a given DataFrame.

        This method takes a Matplotlib Axes object and a pandas DataFrame as inputs.
        It groups the DataFrame by the "Study" column, computes a Gaussian kernel density
        estimate (KDE) for values in each group, and plots the resulting density curves
        on the provided Axes object.

        Args:
            ax : plt.Axes
                Matplotlib Axes object where the density curves will be plotted.

            df : pd.DataFrame
                Input DataFrame containing the data to be grouped and analyzed. The DataFrame
                must include a "Study" column to group the data and a "value" column for
                numerical analysis.
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

    def _draw_histograms(self, ax: plt.Axes, df: pd.DataFrame, bins: int):
        """
        Draws histograms for the given data grouped by 'Study'.

        This method iterates through grouped data based on the 'Study' column,
        and plots histograms of the 'value' column for each group. The histograms
        are styled with partial transparency, a defined bin width, and outlined
        borders.

        Args:
            ax (plt.Axes): The matplotlib Axes object where the histograms will
                be plotted.
            df (pd.DataFrame): The DataFrame containing the data to be grouped
                and plotted. It must include a 'Study' column for grouping and
                a 'value' column for histogram plotting.
            bins (int): The number of bins to use for the histogram.

        Raises:
            None
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
        """
        Generates a volcano plot visualizing the relationship between effect size and
        significance (FDR values) for pooled meta-analysis results.

        This function creates a scatter plot where each point represents a gene or
        another analyzed element. The x-axis represents the effect size, while the
        y-axis shows the -log10 transformed FDR values. The plot also includes thresholds
        for effect size and FDR significance, highlights top genes with effect size and
        significance, and optionally saves the figure.

        Args:
            meta_object: Dict
                Dictionary containing meta-analysis results. Expected to contain a key
                metaAnalysis with a sub-key pooledResults for access to the pooled
                data.
            es_threshold: float, optional
                The effect size threshold used to annotate significant elements. Defaults
                to 0.5.
            fdr_threshold: float, optional
                The FDR threshold used to annotate significant elements. Defaults to 0.05.
            top_genes: int, optional
                Number of top genes to label based on effect size and significance. Set this
                to 0 to skip labeling. Defaults to 20.
            save_path: Optional[str], optional
                File path for saving the generated plot. If None, the plot will not be
                saved. Defaults to None.

        Returns:
            matplotlib.figure.Figure
                The generated volcano plot as a Matplotlib Figure instance.

        Raises:
            Exception
                Any error encountered during data processing or plotting will be raised.
        """
        pooled_df = meta_object["metaAnalysis"]["pooledResults"]

        # 1. Prepare Data
        x = pooled_df["effectSize"]
        y = -np.log10(pooled_df["effectSizeFDR"].replace(0, 1e-300))

        fig, ax = plt.subplots(figsize=(10, 8))

        # 2. Plotting Steps
        self._draw_volcano_points(
            ax, x, y, pooled_df["effectSizeFDR"], es_threshold, fdr_threshold
        )
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

    def _draw_volcano_points(self, ax, x, y, fdr, es_thr, fdr_thr):
        """
        Draws volcano plot points on the provided matplotlib axis. The method visualizes
        a scatter plot where different significance levels are emphasized based on FDR
        values and effect size thresholds. Points not meeting the thresholds are plotted
        in gray as background, while significant points are colored to indicate direction
        (up-regulated or down-regulated).

        Args:
            - ax: matplotlib.axes.Axes
              The matplotlib Axes object where the points will be plotted.
            - x: numpy.ndarray
              The x-axis values (e.g., effect size) for each data point.
            - y: numpy.ndarray
              The y-axis values (e.g., -log10(p-value)) for each data point.
            - fdr: numpy.ndarray
              The False Discovery Rate (adjusted p-values) for each data point.
            - es_thr: float
              The effect size threshold above which data points are considered significant.
            - fdr_thr: float
              The FDR threshold below which data points are considered significant.
        """
        # Draw all points in background
        ax.scatter(
            x, y, alpha=0.4, s=30, c="gray", edgecolors="none", label="Not Significant"
        )

        # Color significant genes by direction
        sig_mask = (fdr <= fdr_thr) & (np.abs(x) >= es_thr)
        if sig_mask.any():
            up = sig_mask & (x > 0)
            down = sig_mask & (x < 0)
            ax.scatter(x[up], y[up], c="red", s=40, alpha=0.8, label="Up-regulated")
            ax.scatter(
                x[down], y[down], c="blue", s=40, alpha=0.8, label="Down-regulated"
            )

    def _add_volcano_thresholds(self, ax, es_thr, fdr_thr):
        """
        Adds vertical and horizontal threshold lines to a volcano plot.

        This method modifies the given matplotlib axis ('ax') by adding horizontal and
        vertical dashed lines that indicate threshold values for effect size ('es_thr')
        and false discovery rate ('fdr_thr'). These thresholds help in visually distinguishing
        significant data points in a volcano plot.

        Args:
            ax : matplotlib.axes.Axes
                The matplotlib axis object on which the thresholds will be drawn.
            es_thr : float
                The threshold for effect size. Vertical lines will be drawn at +es_thr and -es_thr.
            fdr_thr : float
                The False Discovery Rate (FDR) threshold. A horizontal line will be drawn
                at -log10(fdr_thr).

        Returns:
            None
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

    def _label_top_genes(self, ax, x, y, count):
        """
        Annotates the top genes on a plot based on their expression values.

        This method selects the top count genes with the highest expression values
        from the provided Series y and annotates them on the plot represented by
        ax. Each gene is labeled at its corresponding position in the x and y
        coordinates, with a small offset for better visibility.

        Args:
            x: pandas.Series
                A Series containing the x-coordinates for the genes.
            y: pandas.Series
                A Series containing the gene expression values where the gene names are
                the index.
            count: int
                The number of top genes to annotate on the plot.
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
        """
        Generates a forest plot to display effect sizes and pooled results for a set of genes.

        Summary:
        This function creates a forest plot to visualize the effect sizes and pooled
        results for a specified set of genes based on meta-analysis data. The function
        filters for the given genes, limits them to a specified maximum, and plots the
        results with aesthetic adjustments to enhance readability. If a save path is
        provided, the plot is saved to a file.

        Args:
            meta_object (Dict): Meta-analysis data containing effect sizes and pooled results.
            genes (List[str]): List of genes to include in the forest plot.
            max_genes (int, optional): Maximum number of genes to display in the plot.
                Default is 20.
            save_path (Optional[str], optional): File path to save the plot. If not provided,
                the plot is not saved.

        Returns:
            matplotlib.figure.Figure: A matplotlib Figure object containing the forest plot.

        Raises:
            ValueError: If no matching genes are found in the provided data or gene list.
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

    def _plot_study_effects(self, ax, gene, y_pos, es_df, se_df, colors):
        """
        Plots the effect sizes of a gene across multiple studies.

        This function creates a visualization in which the effect sizes and their
        corresponding standard errors for a specific gene across studies are displayed.
        Each effect size is represented as a point on the plot, with horizontal lines
        indicating the confidence intervals determined by the standard error.

        Args:
            ax : matplotlib.axes.Axes
                The matplotlib axis object where the plot will be drawn.
            gene : str
                The name of the gene whose effects are being plotted.
            y_pos : int
                The vertical position at which the gene's effect sizes are plotted.
            es_df : pandas.DataFrame
                A DataFrame containing the effect sizes. Each column corresponds to a
                study, and each row corresponds to a gene.
            se_df : pandas.DataFrame
                A DataFrame containing the standard errors. Each column corresponds to a
                study, and each row corresponds to a gene.
            colors : list
                A list of color values used to represent each study in the plot.

        Raises:
            ValueError
                If the dimensions of es_df and se_df are inconsistent or if any
                required data are missing or malformed.
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

    def _plot_pooled_effect(self, ax, gene, y_pos, pooled_df):
        """
        Plots the pooled effect of a specific gene on the given axis.

        This method visualizes the pooled effect size and its confidence interval for a
        specified gene on a matplotlib axis. It utilizes data from the pooled_df
        DataFrame to extract the effect size and its standard error if they exist and
        are not missing. The plotted effect is represented with a scatter marker, and
        the confidence interval is denoted with horizontal lines.

        Args:
            ax: The matplotlib axis on which to plot the pooled effect.
            gene: The gene for which the pooled effect is to be plotted.
            y_pos: The y-axis position where the pooled effect will be drawn.
            pooled_df: The DataFrame containing 'effectSize' and 'effectSizeStandardError' for
                each gene.
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
        """
        Generates a heatmap visualizing effect sizes across studies for selected genes.

        This method processes the meta-analysis data provided as input and plots a heatmap
        to illustrate the effect sizes of selected genes across various studies. It provides
        the option to cluster the data and export the resulting heatmap image to a specified
        file path.

        Args:
            meta_object (Dict): The meta-analysis object containing effect size data and
                pooled results required for generating the heatmap.
            genes (Optional[List[str]]): A list of genes to include in the heatmap. If None,
                genes are selected based on the top effect sizes.
            n_top_genes (int): The maximum number of top-ranked genes to include if 'genes'
                is None. Defaults to 50.
            cluster (bool): Whether to cluster the data before plotting. Defaults to True.
            save_path (Optional[str]): Full path to save the heatmap figure. If None, the
                figure is not saved locally.

        Returns:
            plt.Figure: The matplotlib Figure object containing the heatmap.

        Raises:
            ValueError: If no valid genes are found for the heatmap.
        """
        es_df = meta_object["metaAnalysis"]["datasetEffectSizes"]
        pooled_df = meta_object["metaAnalysis"]["pooledResults"]

        # 1. Select and Validate Genes
        plot_genes = self._select_genes_for_heatmap(
            genes, es_df, pooled_df, n_top_genes
        )
        if not plot_genes:
            raise ValueError("No valid genes found for heatmap")

        # 2. Prepare Data (drop genes with all NaNs)
        heat_df = es_df.loc[plot_genes].dropna(how="all")

        # Check if we have any data left after dropping all-NaN rows
        if heat_df.empty or heat_df.isna().all().all():
            raise ValueError(
                "No valid data available for heatmap after removing NaN values"
            )

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
        """
        Selects a subset of genes for creating a heatmap based on given parameters.

        The method either filters the provided list of genes to retain only those present
        in a reference DataFrame or selects the top-ranked genes based on a specified
        statistical metric from another DataFrame.

        Args:
            genes: List of gene names. Optional; if provided, only genes present in
                the reference index will be retained. If None, top-ranked genes will
                be selected instead.
            es_df: DataFrame containing reference gene information, where the index
                corresponds to gene names.
            pooled_df: DataFrame containing statistical metrics for genes, including
                a column "effectSizeFDR" used for ranking.
            n_top: Integer indicating the number of top-ranked genes to select when
                'genes' is not provided.

        Returns:
            List of selected gene names.
        """
        if genes is not None:
            return [g for g in genes if g in es_df.index]

        # Check if required column exists for sorting
        if "effectSizeFDR" not in pooled_df.columns or pooled_df.empty:
            return []

        # Sort by FDR and take top N
        return pooled_df.sort_values("effectSizeFDR").head(n_top).index.tolist()

    def plot_study_overview(
        self, meta_object: Dict, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plots an overview of studies, visualizing various statistics such as sample distribution,
        total sample size, gene count, and case/control balance in a grid of subplots.

        Args:
            meta_object (Dict): A dictionary containing metadata needed for generating the overview
                plot. The metadata is expected to include "originalData" as a key for extracting
                study statistics.
            save_path (Optional[str]): Path where the figure will be saved. If None, the figure is
                not saved.

        Returns:
            matplotlib.pyplot.Figure: The generated figure object containing the statistical
            visualizations.
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
        """
        Extracts statistical data from the provided study information.

        This method processes the input dictionary containing study data to compute
        various statistical metrics for each study. It calculates the number of
        cases, controls, total samples, and the number of genes, organizing this
        information into a structured pandas DataFrame.

        Args:
            original_data (Dict): A dictionary where keys are study identifiers
            and values are dictionaries containing study details. Sub-dictionaries
            must include 'class', 'keys', and optionally 'formattedName'.

        Returns:
            pd.DataFrame: A DataFrame containing summarized statistical information
            for each study, including the number of cases, controls, total samples,
            and genes.
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
        """
        Plots a bar chart on the provided axes using the given data and formats the axes.

        This function creates a bar plot on the passed in matplotlib axes object, using the provided
        x-coordinates, values, and labels. The bar chart is styled with the specified color and opacity.
        The method also applies formatting to the axes, including setting titles, labels, and other
        customizations.

        Args:
            ax: matplotlib.axes.Axes
                The matplotlib axes object where the bar chart will be plotted.
            x: Sequence
                The x-coordinates of the bars.
            values: Sequence
                The height or value of each bar in the bar chart.
            labels: Sequence
                The sequence of labels to be applied to the x-axis ticks corresponding to the bars.
            ylabel: str
                The label for the y-axis of the bar chart.
            title: str
                The title of the bar chart.
            color: str or list
                The color or list of colors used to fill the bars in the bar chart.
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
    ):
        """
        Formats the appearance of a bar chart axis with given labels, title, and other
        styling options like rotation and grid visibility.

        Args:
            ax : matplotlib.axes.Axes
                The axis object to format.
            labels : list[str]
                A list of labels for the x-axis ticks.
            ylabel : str
                The label for the y-axis.
            title : str
                The title of the chart.
            legend : bool, optional
                A flag indicating whether to display the legend on the axis. Defaults to False.
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
        """
        Generates a summary plot for meta-analysis results.

        This method visualizes meta-analysis outcome data by creating a 2x2 subplot figure
        composed of four types of visualizations:
        1. Effect size distribution
        2. A volcano-lite plot comparing effect sizes to P-values
        3. Distribution of studies per gene
        4. P-value distribution for heterogeneity.

        The generated figure is optionally saved to a specified file path.

        Args:
            meta_object (Dict[str, Any]): A dictionary containing meta-analysis results.
                Must include a 'metaAnalysis' key with a 'pooledResults' DataFrame, where
                effect sizes, P-values, study counts, and heterogeneity metrics should
                be accessible.
            save_path (Optional[str]): A file path to save the generated figure.
                If None, the figure will only be returned and not saved.

        Returns:
            plt.Figure: A matplotlib Figure object containing the generated 2x2 subplot.
        """
        pooled_df: pd.DataFrame = meta_object["metaAnalysis"]["pooledResults"]
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        (ax1, ax2), (ax3, ax4) = axes

        # 1. Effect Size Distribution
        self._plot_es_dist(ax1, pooled_df["effectSize"])

        # 2. Volcano-lite (ES vs P-value)
        self._plot_volcano_lite(
            ax2, pooled_df["effectSize"], pooled_df["effectSizePval"]
        )

        # 3. Studies per Gene
        self._plot_study_count_dist(ax3, pooled_df["numStudies"])

        # 4. Heterogeneity
        self._plot_heterogeneity(ax4, pooled_df["heterogeneityPval"])

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _plot_es_dist(self, ax: plt.Axes, es_series: pd.Series) -> None:
        """
        Plots the distribution of effect sizes on a given Axes object.

        This function takes a pandas Series containing effect sizes and plots
        a histogram representing the frequency of the effect sizes. It also
        includes a vertical red dashed line at zero for reference.

        Args:
            ax: plt.Axes
                The Matplotlib Axes object where the histogram will be plotted.
            es_series: pd.Series
                A pandas Series containing the effect sizes to be visualized.

        Returns:
            None
        """
        ax.hist(
            es_series.dropna(), bins=50, alpha=0.7, color="skyblue", edgecolor="black"
        )
        ax.axvline(0, color="red", linestyle="--", alpha=0.7)
        self._set_standard_labels(
            ax, "Effect Size", "Frequency", "Distribution of Effect Sizes"
        )

    def _plot_volcano_lite(self, ax: plt.Axes, es: pd.Series, pvals: pd.Series) -> None:
        """
        Plots a lightweight volcano plot on the provided Axes without any additional customizations.

        The method creates a scatter plot to visualize the relationship between effect sizes
        and the significance level (-log₁₀ of p-values). This minimal implementation allows
        customization and further modification by other methods or end-users.

        Args:
            ax (plt.Axes): The matplotlib Axes object where the volcano plot will be drawn.
            es (pd.Series): Pandas Series containing the effect sizes for the plot, typically from a
                statistical or computational experiment.
            pvals (pd.Series): Pandas Series containing the p-values corresponding to each effect size.

        Returns:
            None
        """
        ax.scatter(es, -np.log10(pvals.replace(0, 1e-300)), alpha=0.6, s=20)
        self._set_standard_labels(
            ax, "Effect Size", "-log₁₀(P-value)", "Effect Size vs. Significance"
        )

    def _plot_study_count_dist(self, ax: plt.Axes, num_studies: pd.Series) -> None:
        """
        Plots the distribution of the number of studies per gene.

        This method creates a histogram representing the distribution of gene study
        counts. Only non-NaN values from num_studies are considered. The plot is
        formatted using predefined aesthetic settings, including colors and labels.

        Args:
            ax (plt.Axes): The matplotlib Axes object on which the histogram will be plotted.
            num_studies (pd.Series): A pandas Series containing the number of studies per gene.

        Returns:
            None
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
        """
        Plots a histogram of heterogeneity p-values and highlights the significance threshold.

        This function visualizes the distribution of valid heterogeneity p-values within a histogram
        and marks a reference line at p=0.05 to indicate the common significance threshold. If there
        are no valid p-values in the input, a message "No heterogeneity data" is displayed in the plot.
        This plot is designed to provide an overview of the results of heterogeneity tests.

        Args:
            ax (plt.Axes): The Matplotlib Axes object where the histogram will be plotted.
            pvals (pd.Series): Series of heterogeneity p-values, which may include NaN values.

        Returns:
            None
        """
        valid_pvals = pvals.dropna()
        if not valid_pvals.empty:
            ax.hist(valid_pvals, bins=50, alpha=0.7, color="salmon", edgecolor="black")
            ax.axvline(0.05, color="red", linestyle="--", alpha=0.7, label="p=0.05")
            ax.legend()
        else:
            ax.text(
                0.5, 0.5, "No heterogeneity data", transform=ax.transAxes, ha="center"
            )
        self._set_standard_labels(
            ax, "Heterogeneity P-value", "Frequency", "Heterogeneity Test Results"
        )

    def _set_standard_labels(self, ax: plt.Axes, x: str, y: str, title: str) -> None:
        """
        Sets standard labels for an axis, including x-axis, y-axis, title, and grid visibility.

        Args:
            ax (plt.Axes): The Matplotlib axis to modify.
            x (str): The label for the x-axis.
            y (str): The label for the y-axis.
            title (str): The title of the plot.

        Returns:
            None
        """
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
