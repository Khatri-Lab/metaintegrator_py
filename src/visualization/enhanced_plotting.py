"""
Enhanced plotting functions for MetaIntegrator Python port.
This module provides comprehensive visualization capabilities including
forest plots, heatmaps, Manhattan plots, and all plotting functions from the R version.
"""

import warnings
from typing import Dict, List, Optional, Union, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Polygon
from scipy import stats
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


# ------------------------------------------------------------------
# EnhancedMetaAnalysisPlotter Class
# ------------------------------------------------------------------
class EnhancedMetaAnalysisPlotter:
    """Enhanced plotting class for meta-analysis visualizations."""

    CI_95_MULTIPLIER = 1.96

    def __init__(self, style: str = "default", figsize: Tuple[int, int] = (10, 6)):
        """
        Initializes the visualization configuration with a specified style and figure size.

        This class sets up the style and dimensions for a visual plot representation using
        a given style configuration. The style applies the corresponding visual appearance
        to the plot if it is not set to "default".

        Args:
            style: str
                The style configuration for the plot, such as a specific visualization theme
                (default is "default").
            figsize: Tuple[int, int]
                Dimensions of the figure in inches, given as a tuple (width, height)
                (default is (10, 6)).
        """
        self.style = style
        self.figsize = figsize
        if style != "default":
            plt.style.use(style)

    def _get_meta_data(self, meta_object: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract meta-analysis data from the provided meta-analysis object.

        This method retrieves specific meta-analysis results from the input dictionary.
        If the required meta-analysis data is not present, it raises a ValueError.

        Args:
            meta_object: Dict
                A dictionary containing meta-analysis data.

        Raises:
            ValueError:
                Raised if the "metaAnalysis" key is not found in the input dictionary.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
                A tuple containing the dataset effect sizes, dataset effect size
                standard errors, and pooled results in the form of pandas DataFrames.
        """
        if "metaAnalysis" not in meta_object:
            raise ValueError("Meta-analysis results not found")

        analysis = meta_object["metaAnalysis"]
        return (
            analysis["datasetEffectSizes"],
            analysis["datasetEffectSizeStandardErrors"],
            analysis["pooledResults"],
        )

    def _get_study_names(self, meta_object: Dict, studies: List[str]) -> List[str]:
        """
        Retrieves study names from a given metadata object.

        This method extracts study names from the provided metadata object. It uses
        the originalData key to look up formatted names for each study in the
        input list. If a formatted name is unavailable for a study, the study's
        original name from the input list will be returned as a fallback.

        Args:
            meta_object (Dict): A dictionary containing metadata information,
                                with an "originalData" key holding the relevant data.
            studies (List[str]): A list of study identifiers for which formatted
                                 names need to be retrieved.

        Returns:
            List[str]: A list of formatted study names corresponding to the
                       input study identifiers.
        """
        original_data = meta_object.get("originalData", {})
        return [original_data.get(s, {}).get("formattedName", s) for s in studies]

    def _draw_pooled_diamond(
        self,
        ax,
        effect: float,
        se: float,
        y_pos: float,
        color: str,
        ci_multiplier: float = 1.96,
    ):
        """
        Draws a diamond-shaped graphical representation of an effect with confidence intervals on a specified axis.

        The method visualizes the effect size by plotting a diamond shape, where the horizontal width
        of the diamond represents the confidence interval range, and the vertical height of the diamond
        is a fixed value. The color and other parameters of the diamond are customizable.

        Args:
            ax (matplotlib.axes.Axes): The Axes object where the diamond will be drawn.
            effect (float): The central value of the effect size.
            se (float): The standard error of the effect size.
            y_pos (float): The vertical position of the diamond on the axis.
            color (str): The fill color of the diamond.
            ci_multiplier (float): A multiplier for the confidence interval, with a default value of 1.96.

        Raises:
            None

        Returns:
            None
        """
        ci_half_width = ci_multiplier * se
        diamond_height = 0.2
        diamond = patches.Polygon(
            [
                (effect, y_pos - diamond_height),
                (effect + ci_half_width, y_pos),
                (effect, y_pos + diamond_height),
                (effect - ci_half_width, y_pos),
            ],
            facecolor=color,
            edgecolor="black",
        )
        ax.add_patch(diamond)

    def _map_es_col_to_se_col(self, es_col: str) -> str:
        """Map an effect-size column name to its standard-error counterpart.

        Internal column naming convention:
            effect sizes:     study_N_g
            standard errors:  study_N_se_g
        """
        # study_0_g -> study_0_se_g
        if es_col.endswith("_g"):
            prefix = es_col[: -len("_g")]
            return f"{prefix}_se_g"
        return es_col  # fallback – return as-is

    def _map_study_col_to_dataset_name(self, meta_object: Dict, study_col: str) -> str:
        """Map an internal study column name (e.g. 'study_0_g') to the
        corresponding dataset key in originalData."""
        import re

        m = re.match(r"study_(\d+)_", study_col)
        if m:
            idx = int(m.group(1))
            keys = list(meta_object.get("originalData", {}).keys())
            if idx < len(keys):
                return keys[idx]
        return study_col  # fallback

    def forest_plot(
        self,
        meta_object: Dict,
        gene_name: str,
        box_color: str = "blue",
        whisker_color: str = "lightblue",
        zero_line_color: str = "black",
        summary_color: str = "orange",
        text_color: str = "red",
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generates a forest plot to visualize effect sizes and confidence intervals for a specific gene across
        multiple studies.

        The forest plot includes individual studies and their respective confidence intervals, represented as
        box-and-whisker plots. The pooled result is visualized as a diamond, summarizing the combined effect size
        and confidence interval across studies. This function is designed for meta-analysis visualization.

        Attributes:
            None

        Args:
            meta_object (Dict): A dictionary containing meta-analysis data including effect sizes and
                standard errors for multiple studies.
            gene_name (str): Name of the gene for which the forest plot is generated.
            box_color (str, optional): Color of the boxes representing individual studies' effect sizes.
                Defaults to "blue".
            whisker_color (str, optional): Color of the whiskers (confidence intervals) for individual studies.
                Defaults to "lightblue".
            zero_line_color (str, optional): Color of the vertical line at zero marking no effect. Defaults
                to "black".
            summary_color (str, optional): Color of the diamond representing the pooled effect. Defaults
                to "orange".
            text_color (str, optional): Color of the text labels for study names. Defaults to "red".
            figsize (Optional[Tuple], optional): Tuple specifying the figure size. If None, the default
                figure size is used. Defaults to None.
            save_path (Optional[str], optional): File path to save the generated plot. If None, the plot
                is not saved. Defaults to None.

        Returns:
            plt.Figure: A Matplotlib Figure object containing the forest plot.

        Raises:
            ValueError: If the specified gene_name is not found in the metaObject dataset.
        """
        figsize = figsize or self.figsize
        effect_sizes, std_errors, pooled_results = self._get_meta_data(meta_object)

        if gene_name not in effect_sizes.index:
            raise ValueError(f"No gene named '{gene_name}' in this metaObject")

        # Prepare individual study data
        # Effect-size columns are like study_0_g; SE columns are study_0_se_g
        gene_effects = effect_sizes.loc[gene_name].dropna()
        se_cols = [self._map_es_col_to_se_col(c) for c in gene_effects.index]
        gene_ses = std_errors.loc[gene_name, se_cols]
        gene_ses.index = gene_effects.index  # align indices

        # Map internal study column names to actual dataset names
        study_names = [
            meta_object.get("originalData", {})
            .get(self._map_study_col_to_dataset_name(meta_object, col), {})
            .get("formattedName", self._map_study_col_to_dataset_name(meta_object, col))
            for col in gene_effects.index.tolist()
        ]

        # Sort alphabetically
        sort_idx = np.argsort(study_names)
        gene_effects, gene_ses = gene_effects.iloc[sort_idx], gene_ses.iloc[sort_idx]
        study_names = [study_names[i] for i in sort_idx]

        fig, ax = plt.subplots(figsize=figsize)
        n_studies = len(gene_effects)

        # Plot studies in reverse order (top to bottom)
        for i, (effect, se) in enumerate(zip(gene_effects, gene_ses)):
            y_pos = n_studies - i - 1
            # CI Line
            ax.hlines(
                y_pos,
                effect - 1.96 * se,
                effect + 1.96 * se,
                colors=whisker_color,
                linewidth=2,
            )
            # Inverse variance box
            weight = 1 / (se**2) if se > 0 else 1
            box_size = min(max(weight / 10, 0.1), 0.4)
            ax.add_patch(
                patches.Rectangle(
                    (effect - box_size / 2, y_pos - box_size / 2),
                    box_size,
                    box_size,
                    facecolor=box_color,
                    edgecolor="black",
                )
            )

        # Plot pooled diamond
        pooled_effect = pooled_results.loc[gene_name, "effectSize"]
        pooled_se = pooled_results.loc[gene_name, "effectSizeStandardError"]
        self._draw_pooled_diamond(ax, pooled_effect, pooled_se, n_studies, summary_color)

        # Formatting
        ax.axvline(0, color=zero_line_color, linestyle="--", alpha=0.7)
        ax.set_yticks(np.arange(n_studies + 1))
        ax.set_yticklabels(study_names + ["Pooled"], color=text_color)
        ax.set_title(gene_name)
        ax.set_xlabel("Standardized Mean Difference (log2 scale)")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def gg_forest_plot(
        self,
        meta_object: Dict,
        genes: List[str],
        conf_level: float = 0.95,
        facet_cols: Optional[int] = None,
        box_scales: Tuple = (6, 16),
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generates a forest plot for meta-analytic results of specified genes.

        This method creates a forest plot for the provided list of genes based
        on meta-analytic data. The forest plot will include individual dataset
        effects with confidence intervals as well as pooled results summary.
        The generated figure can either be displayed or saved to a file.

        Args:
            meta_object: Dict
                A dictionary containing meta-analytic data. The specific format of
                the dictionary is expected to include necessary effect sizes and other
                meta-analysis related data.

            genes: List[str]
                A list of gene names for which forest plots are to be generated.

            conf_level: float, default 0.95
                The confidence level to determine the width of the confidence intervals.

            facet_cols: Optional[int], optional
                The number of columns in the facet grid layout for the visualizations.
                Defaults to a value based on the number of genes.

            box_scales: Tuple
                A tuple specifying scaling factors for the boxes and text formatting.

            figsize: Optional[Tuple], optional
                Tuple representing the width and height of the figure in inches. If not
                provided, it defaults to a value based on the number of genes.

            save_path: Optional[str], optional
                Path to save the generated forest plot image. If not provided, the plot
                will only be displayed and not saved.

        Returns:
            matplotlib.figure.Figure
                A matplotlib Figure object representing the forest plot.
        """
        figsize = figsize or ((15, 8) if len(genes) > 1 else self.figsize)
        facet_cols = facet_cols or (1 if len(genes) == 1 else min(4, len(genes)))
        ci_multiplier = stats.norm.ppf((1 + conf_level) / 2)

        effect_sizes, std_errors, pooled_results = self._get_meta_data(meta_object)
        n_rows = int(np.ceil(len(genes) / facet_cols))
        fig, axes = plt.subplots(n_rows, facet_cols, figsize=figsize, squeeze=False, sharey=True)
        axes_flat = axes.flatten()

        for i, gene in enumerate(genes):
            if i >= len(axes_flat) or gene not in effect_sizes.index:
                continue
            ax = axes_flat[i]

            # Combine individual and pooled data
            gene_effects = effect_sizes.loc[gene].dropna()
            all_effects = list(gene_effects) + [pooled_results.loc[gene, "effectSize"]]
            all_ses = list(std_errors.loc[gene, gene_effects.index]) + [
                pooled_results.loc[gene, "effectSizeStandardError"]
            ]
            all_names = list(gene_effects.index) + ["Summary"]

            n_items = len(all_effects)
            for j, (effect, se, name) in enumerate(zip(all_effects, all_ses, all_names)):
                y_pos = n_items - j - 1
                if name == "Summary":
                    self._draw_pooled_diamond(ax, effect, se, y_pos, "coral", ci_multiplier)
                else:
                    # Draw study row (simplified logic for example)
                    ax.hlines(
                        y_pos,
                        effect - ci_multiplier * se,
                        effect + ci_multiplier * se,
                        colors="black",
                    )

                    # Specifying marker and color separately
                    ax.plot(effect, y_pos, marker="s", color="blue")

            ax.set_title(gene)
            ax.axvline(0, color="black", linestyle="--")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        return fig

    def _prepare_heatmap_data(
        self,
        meta_data: Dict,
        filter_object: Dict,
        display_pooled: bool,
        gene_order: Union[bool, List],
    ) -> pd.DataFrame:
        """
        Processes and prepares heatmap data based on metadata, a filter object, and display options.

        The function filters gene effect sizes from the provided metadata using the genes specified in
        the filter object. If necessary, it adds pooled results to the dataset and applies an ordering
        to the genes based on the display and sorting options provided.

        Args:
            meta_data: Dict
                Metadata containing gene-related information, effect sizes, and pooled results.
            filter_object: Dict
                Object containing genes to filter by ("posGeneNames" and "negGeneNames").
            display_pooled: bool
                Flag indicating whether to include pooled results in the heatmap data.
            gene_order: Union[bool, List]
                List specifying the order of genes, or a boolean indicating if default ordering
                should be applied.

        Returns:
            pd.DataFrame
                DataFrame containing the filtered and sorted heatmap data. Returns an
                empty DataFrame if no valid genes are present.
        """
        gene_names = filter_object.get("posGeneNames", []) + filter_object.get("negGeneNames", [])
        effect_sizes = meta_data["metaAnalysis"]["datasetEffectSizes"]
        pooled_results = meta_data["metaAnalysis"]["pooledResults"]

        # Filter to available genes
        gene_names = [g for g in gene_names if g in effect_sizes.index]
        if len(gene_names) <= 1:
            return pd.DataFrame()

        heat_data = effect_sizes.loc[gene_names].copy()

        if display_pooled:
            pooled_col = pooled_results.loc[gene_names, "effectSize"]
            pooled_col.name = "Pooled"
            heat_data = pd.concat([pooled_col, heat_data], axis=1)

        # Handle ordering
        if not isinstance(gene_order, list):
            if display_pooled and "Pooled" in heat_data.columns:
                gene_order = heat_data.sort_values("Pooled").index.tolist()
            else:
                gene_order = heat_data.index.tolist()

        return heat_data.loc[gene_order]

    def _calculate_heatmap_figsize(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculates the figure size for a heatmap based on the dimensions of the provided
        data. The function computes dimensions ensuring a minimum size while adjusting
        proportions dynamically according to the number of genes and datasets.

        Args:
            data (pd.DataFrame): A pandas DataFrame where the rows correspond to genes
            and the columns to datasets.

        Returns:
            Tuple[float, float]: A tuple containing the width and height of the heatmap
            figure in inches.
        """
        n_genes, n_datasets = data.shape
        return (max(8, n_datasets * 0.6), max(6, n_genes * 0.3))

    def heatmap_plot(
        self,
        meta_data: Dict,
        filter_object: Dict,
        color_range: Tuple[float, float] = (-1, 1),
        gene_order: Union[bool, List] = False,
        dataset_order: Union[bool, List] = False,
        display_pooled: bool = True,
        use_formatted_names: bool = True,
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        Plots a heatmap based on the provided meta data and filtering criteria. The heatmap
        visualizes the effect size values for genes across datasets and can be customized in terms
        of color range, figure size, gene/dataset order, and more. If the heatmap data is empty
        or has less than two genes, the method will return None.

        Args:
            meta_data (Dict): Contains metadata relevant for generating the heatmap.
            filter_object (Dict): Filters to determine which data to include in the heatmap.
            color_range (Tuple[float, float], optional): Specifies the minimum and maximum values
                for the color range in the heatmap. Defaults to (-1, 1).
            gene_order (Union[bool, List], optional): If True, the genes are ordered automatically.
                If a list is provided, it specifies the order of genes. Defaults to False for no ordering.
            dataset_order (Union[bool, List], optional): If True, datasets are ordered automatically.
                If a list is provided, it specifies the order of datasets. Defaults to False for no ordering.
            display_pooled (bool, optional): Whether to pool data for visualization. Defaults to True.
            use_formatted_names (bool, optional): If True, formatted names will be used for labels. Defaults to True.
            figsize (Optional[Tuple], optional): Tuple specifying the size of the heatmap figure.
                Automatically calculated if not provided. Defaults to None.
            save_path (Optional[str], optional): File path to save the heatmap plot. If None, the plot is not saved.
                Defaults to None.

        Returns:
            Optional[plt.Figure]: The resulting heatmap figure object if successfully created, or None if the
                data does not meet the criteria for generating a heatmap.

        Raises:
            None
        """
        heat_data = self._prepare_heatmap_data(meta_data, filter_object, display_pooled, gene_order)

        if heat_data.empty:
            warnings.warn("Must have at least 2 genes for heatmap")
            return None

        # Clip values to color range
        heat_data_clipped = heat_data.clip(lower=color_range[0], upper=color_range[1])
        figsize = figsize or self._calculate_heatmap_figsize(heat_data)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(
            heat_data_clipped.values,
            cmap="RdBu_r",
            aspect="auto",
            vmin=color_range[0],
            vmax=color_range[1],
        )

        # Labels and formatting
        ax.set_xticks(range(len(heat_data.columns)))
        ax.set_yticks(range(len(heat_data.index)))
        ax.set_xticklabels(heat_data.columns)

        if len(heat_data.index) > 40:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(heat_data.index)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Effect Size", rotation=270, labelpad=15)
        ax.set_title("Effect Size Heatmap")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def manhattan_plot(
        self,
        meta_object: Dict,
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generates a Manhattan plot to visualize -log10(FDR) values for
        pooled results from a meta-analysis. The x-axis represents gene
        indices, and the y-axis denotes -log10(FDR) values. Points are
        colored by significance, with a horizontal line indicating the
        threshold for FDR significance at 0.05.

        Args:
            meta_object (Dict): The input dictionary containing meta-analysis
                results, specifically the "metaAnalysis" and "pooledResults" keys.
            figsize (Optional[Tuple]): Optional size of the figure as a tuple of
                (width, height). Defaults to (14, 6) if not provided.
            save_path (Optional[str]): Optional path to save the generated plot as
                an image file. If not provided, the plot is not saved.

        Returns:
            plt.Figure: The generated Manhattan plot figure.

        Raises:
            KeyError: If expected keys are missing in the meta_object dictionary,
                such as "metaAnalysis" or "pooledResults".
            ValueError: If the data structure of pooled results is malformed
                or contains invalid entries.
        """
        if figsize is None:
            figsize = (14, 6)

        pooled_results = meta_object["metaAnalysis"]["pooledResults"]

        # Create a simple scatter plot of -log10(FDR) vs gene index
        # (Full implementation would need gene location data)
        fig, ax = plt.subplots(figsize=figsize)

        fdr_values = pooled_results["effectSizeFDR"]
        neg_log_fdr = -np.log10(fdr_values.replace(0, 1e-300))

        # Create x positions (gene indices)
        x_positions = np.arange(len(neg_log_fdr))

        # Color points by significance
        colors = ["red" if p < 0.05 else "blue" for p in fdr_values]

        ax.scatter(x_positions, neg_log_fdr, c=colors, alpha=0.6, s=20)
        ax.axhline(-np.log10(0.05), color="red", linestyle="--", alpha=0.7, label="FDR = 0.05")

        ax.set_xlabel("Gene Index")
        ax.set_ylabel("-log₁₀(FDR)")
        ax.set_title("Manhattan Plot: Effect Size FDR")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _get_predictive_plot_title(
        self, sensitivity: float, specificity: float, rounding: int
    ) -> str:
        """
        Generates a title for a predictive plot based on sensitivity and specificity values.

        This function computes sensitivity and specificity percentages and combines them
        to create a descriptive title for a predictive value plot.

        Args:
            sensitivity: The sensitivity value as a float, typically between 0 and 1.
            specificity: The specificity value as a float, typically between 0 and 1.
            rounding: The number of decimal places to round the sensitivity and specificity
                      percentages.

        Returns:
            A string representing the predictive plot's title, including formatted sensitivity
            and specificity percentages.
        """
        sens_pct = round(sensitivity * 100, rounding)
        spec_pct = round(specificity * 100, rounding)
        return (
            f"Positive and Negative Predictive Value\n"
            f"{sens_pct}% Sensitivity and {spec_pct}% Specificity"
        )

    def _calculate_predictive_values(
        self, sensitivity: float, specificity: float, n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates predictive values such as Positive Predictive Value (PPV) and Negative Predictive Value (NPV)
        based on given sensitivity, specificity, and steps for prevalence.

        Args:
            sensitivity: float
                Test sensitivity, representing the true positive rate.
            specificity: float
                Test specificity, representing the true negative rate.
            n_steps: int
                Number of steps to generate prevalence values in the range of 0.001 to 0.999.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]
                A tuple containing three numpy arrays:
                - prevalence: The prevalence values calculated using the given number of steps.
                - ppv: The positive predictive values corresponding to the prevalence values.
                - npv: The negative predictive values corresponding to the prevalence values.
        """
        prevalence = np.linspace(0.001, 0.999, n_steps)

        # PPV = (Prev * Sens) / (Prev * Sens + (1 - Spec) * (1 - Prev))
        ppv = (prevalence * sensitivity) / (
            prevalence * sensitivity + (1 - specificity) * (1 - prevalence)
        )

        # NPV = (Spec * (1 - Prev)) / ((1 - Sens) * Prev + Spec * (1 - Prev))
        npv = (specificity * (1 - prevalence)) / (
            (1 - sensitivity) * prevalence + specificity * (1 - prevalence)
        )

        return prevalence, ppv, npv

    def predictive_value_plot(
        self,
        sensitivity: float,
        specificity: float,
        n_steps: int = 1000,
        title: Optional[str] = None,
        rounding: int = 2,
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generates and returns a predictive value plot based on sensitivity, specificity, and prevalence.

        This method calculates predictive values (Positive Predictive Value and Negative Predictive Value)
        for a range of prevalence values and generates a plot. The plot displays the relationship between
        prevalence and predictive values as percentages. The graph can be customized through various optional parameters,
        such as the figure size or the addition of a title. The plot can also be saved to a specified path.

        Args:
            sensitivity (float): Sensitivity value of the test.
            specificity (float): Specificity value of the test.
            n_steps (int, optional): The number of steps to compute predictive values across prevalence.
                                     Defaults to 1000.
            title (str, optional): Title to display on the plot. Defaults to None.
            rounding (int, optional): Specifies the number of decimal places for rounding in the title.
                                      Defaults to 2.
            figsize (Tuple, optional): Tuple specifying the figure size for the plot. Defaults to None.
            save_path (str, optional): File path to save the plot. If provided, the plot is saved to this location.

        Raises:
            ValueError: If sensitivity or specificity is not provided.

        Returns:
            matplotlib.figure.Figure: A matplotlib Figure object containing the predictive value plot.
        """
        if sensitivity is None or specificity is None:
            raise ValueError("Sensitivity and specificity must be provided")

        figsize = figsize or self.figsize
        prevalence, ppv, npv = self._calculate_predictive_values(sensitivity, specificity, n_steps)

        # Convert to percentages for plotting
        prev_pct, ppv_pct, npv_pct = prevalence * 100, ppv * 100, npv * 100

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            prev_pct,
            ppv_pct,
            "b-",
            linewidth=2,
            label="Positive Predictive Value (PPV)",
            alpha=0.8,
        )
        ax.plot(
            prev_pct,
            npv_pct,
            "r-",
            linewidth=2,
            label="Negative Predictive Value (NPV)",
            alpha=0.8,
        )

        ax.set_xlabel("Prevalence (%)")
        ax.set_ylabel("Predictive Value (%)")
        ax.set_title(title or self._get_predictive_plot_title(sensitivity, specificity, rounding))

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _calculate_auc_confidence_interval(
        self, y_true: np.ndarray, roc_auc: float
    ) -> Tuple[float, float]:
        """
        Calculates the confidence interval for the Area Under the Curve (AUC) value. This method estimates the standard error
        of the AUC, computes the confidence interval boundaries, and ensures that the lower and upper bounds are within
        the valid range of [0, 1].

        Args:
            y_true: np.ndarray
                Array of true binary labels, where 1 represents the positive class and 0 represents the negative class.
            roc_auc: float
                The calculated AUC value for which the confidence interval will be determined.

        Returns:
            Tuple[float, float]
                The lower and upper bounds of the 95% confidence interval for the given AUC value.
        """
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.0, 0.0

        se_auc = np.sqrt((roc_auc * (1 - roc_auc)) / (n_pos * n_neg))
        ci_lower = max(0, roc_auc - self.CI_95_MULTIPLIER * se_auc)
        ci_upper = min(1, roc_auc + self.CI_95_MULTIPLIER * se_auc)
        return ci_lower, ci_upper

    def _calculate_roc_stats(
        self, y_true: np.ndarray, y_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculates Receiver Operating Characteristic (ROC) statistics including false positive rate (FPR),
        true positive rate (TPR), and the area under the ROC curve (AUC).

        This function is useful for evaluating the performance of a binary classification model by
        assessing its ability to distinguish between two classes. It computes the false positive rate and
        true positive rate across different thresholds and calculates the area under the ROC curve.

        Args:
            y_true: np.ndarray
                Ground truth binary labels where each entry corresponds to the true class of the sample.
            y_scores: np.ndarray
                Predicted scores or probabilities for the positive class of the classification model.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]
                A tuple containing:
                - fpr: False positive rates at various thresholds
                - tpr: True positive rates at various thresholds
                - roc_auc: The area under the ROC curve
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def _calculate_auc_ci(self, y_true: np.ndarray, auc_value: float) -> Tuple[float, float]:
        """
        Calculates the confidence interval for the Area Under the Curve (AUC) score.

        This method estimates the confidence interval for a given AUC score using the
        standard error formula derived from the AUC distribution. It requires the
        ground truth labels to determine the number of positive and negative samples.
        If either positive or negative samples are absent, the method returns a confidence
        interval of (0.0, 0.0).

        Args:
            y_true: np.ndarray
                The ground truth binary labels, where positive samples are denoted
                by 1 and negative samples by 0.
            auc_value: float
                The Area Under the Curve score for the binary classification.

        Returns:
            Tuple[float, float]
                A tuple containing the lower and upper bounds of the confidence interval
                for the AUC score.
        """
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.0, 0.0

        se_auc = np.sqrt((auc_value * (1 - auc_value)) / (n_pos * n_neg))
        ci_lower = max(0, auc_value - 1.96 * se_auc)
        ci_upper = min(1, auc_value + 1.96 * se_auc)
        return ci_lower, ci_upper

    def _calculate_roc_data(
        self, y_true: np.ndarray, y_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculates the Receiver Operating Characteristic (ROC) data including False Positive Rate (FPR),
        True Positive Rate (TPR), and Area Under the Curve (AUC) for given true labels and predicted scores.

        Args:
            y_true: np.ndarray
                True binary labels where 1 represents the positive class and 0 represents the negative class.

            y_scores: np.ndarray
                Target scores or probabilities for the positive class. These values are expected
                to reflect the likelihood of the positive class.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]
                A tuple containing:
                - False Positive Rates (FPR) as a NumPy array
                - True Positive Rates (TPR) as a NumPy array
                - AUC value as a float
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_value = auc(fpr, tpr)
        return fpr, tpr, auc_value

    def roc_plot(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "ROC Curve",
        label: Optional[str] = None,
        color: str = "blue",
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generates a Receiver Operating Characteristic (ROC) curve plot with optional AUC confidence
        intervals and customization options.

        Args:
            y_true (np.ndarray): Array of true binary labels. Expected to contain 0s and 1s.
            y_scores (np.ndarray): Array of predicted scores or probabilities corresponding to y_true.
            title (str): Title of the plot. Defaults to 'ROC Curve'.
            label (Optional[str]): Custom label for the AUC plot line. If not provided, a label
                with AUC and confidence interval will be generated automatically.
            color (str): Line color for the ROC curve. Defaults to 'blue'.
            figsize (Optional[Tuple]): Tuple specifying the figure size in inches. If not provided,
                the default size stored in self.figsize will be used.
            save_path (Optional[str]): File path to save the plot as an image. If not provided,
                the plot will not be saved.

        Returns:
            matplotlib.figure.Figure: The generated matplotlib figure object for the ROC curve.

        Raises:
            TypeError: If input types of parameters do not match expected types.
            ValueError: If there are inconsistencies in the input arrays, such as mismatched shapes.

        """
        figsize = figsize or self.figsize
        fpr, tpr, auc_val = self._calculate_roc_data(y_true, y_scores)
        ci_low, ci_high = self._calculate_auc_ci(y_true, auc_val)

        if label is None:
            label = f"AUC = {auc_val:.3f} (95% CI: {ci_low:.3f}-{ci_high:.3f})"

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random")

        ax.set_xlabel("False Positive Rate (1-Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _calculate_auprc_confidence_interval(
        self, y_true: np.ndarray, auprc: float
    ) -> Tuple[float, float]:
        """
        Calculates the 95% confidence interval for Area Under the Precision-Recall Curve (AUPRC).

        This function computes the confidence interval for a given AUPRC value based on the
        standard error derived from the number of positive samples within the true labels.

        Args:
            y_true (np.ndarray): Array of true labels. 1 represents positive samples.
            auprc (float): The Area Under the Precision-Recall Curve value.

        Returns:
            Tuple[float, float]: A tuple containing the lower and upper bounds of the
                                 95% confidence interval for the AUPRC.
        """
        n_pos = np.sum(y_true == 1)

        # When there are no positive samples, return wide confidence interval to indicate uncertainty
        if n_pos == 0:
            return 0.0, 1.0

        se_auprc = np.sqrt((auprc * (1 - auprc)) / n_pos)
        ci_lower = max(0, auprc - self.CI_95_MULTIPLIER * se_auprc)
        ci_upper = min(1, auprc + self.CI_95_MULTIPLIER * se_auprc)
        return ci_lower, ci_upper

    def _get_prc_baseline(self, y_true: np.ndarray) -> float:
        """
        Calculates the precision-recall curve baseline.

        This method computes the ratio of the sum of true labels to the total
        number of labels. If the input array y_true is empty, it returns 0.0.

        Args:
            y_true: np.ndarray
                The array of true labels, typically binary values.

        Returns:
            float
                The calculated precision-recall baseline value.
        """
        return np.sum(y_true) / len(y_true) if len(y_true) > 0 else 0.0

    def _calculate_prc_stats(
        self, y_true: np.ndarray, y_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculates precision-recall statistics and Average Precision (AP) score.

        This utility function computes the precision-recall curve, including
        precision and recall values for different thresholds, and calculates
        the average precision score (AUPRC) metric.

        Args:
            y_true: np.ndarray
                Ground truth binary labels. Each value should be either 0 or 1.
            y_scores: np.ndarray
                Predicted scores or probabilities, typically between 0 and 1, that
                correspond to each label in y_true.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]
                A tuple containing:
                - precision: ndarray of precision values for each threshold
                - recall: ndarray of recall values for each threshold
                - auprc: float representing the average precision score
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
        return precision, recall, auprc

    def prc_plot(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "Precision-Recall Curve",
        label: Optional[str] = None,
        color: str = "red",
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plots the Precision-Recall Curve (PRC) along with relevant metrics and confidence
        intervals to evaluate the performance of classification models. The function supports
        customization of the plot's appearance and the ability to save the resulting figure.

        Args:
            y_true : np.ndarray
                Ground truth binary labels.
            y_scores : np.ndarray
                Model-predicted probability scores or decision function values corresponding to
                the positive class.
            title : Optional[str]
                Title of the precision-recall plot (default is "Precision-Recall Curve").
            label : Optional[str], optional
                Custom label for the precision-recall curve. If not provided, a default label
                including AUPRC and its confidence interval values will be used.
            color : str, optional
                Color of the precision-recall line (default is "red").
            figsize : Optional[Tuple], optional
                Dimensions of the figure in inches. If not specified, default dimensions defined
                in the class attribute will be used.
            save_path : Optional[str], optional
                File path to save the generated figure as an image file. If not provided, the
                figure will not be saved.

        Returns:
            plt.Figure
                Matplotlib figure object representing the precision-recall curve.

        Raises:
            ValueError
                If the inputs are invalid, such as mismatched shapes of y_true and y_scores.

        Note:
        The function includes display of the area under the precision-recall curve (AUPRC)
        and its associated 95% confidence interval, along with the performance of a random
        baseline classifier for comparison.

        Example:
        This function generates a visual assessment but does not print or write to console
        about performance metrics. For saving output plots, the save_path parameter must
        be provided.
        """
        figsize = figsize or self.figsize
        precision, recall, auprc = self._calculate_prc_stats(y_true, y_scores)
        ci_lower, ci_upper = self._calculate_auprc_confidence_interval(y_true, auprc)
        baseline = self._get_prc_baseline(y_true)

        if label is None:
            label = f"AUPRC = {auprc:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})"

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, color=color, linewidth=2, label=label)

        ax.axhline(
            y=baseline,
            color="k",
            linestyle="--",
            alpha=0.6,
            label=f"Random (AUPRC={baseline:.3f})",
        )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _get_dataset_roc_info(
        self, i: int, dataset: Dict, filter_object: Dict
    ) -> Tuple[np.ndarray, np.ndarray, float, str]:
        """
        Calculates Receiver Operating Characteristic (ROC) information for a given dataset.

        This method processes a dataset to generate information relevant for plotting the ROC curve.
        It incorporates calculated scores and true class labels from the dataset, derives the
        false positive rate (FPR), true positive rate (TPR), and the area under the curve (AUC) value
        using ROC evaluation, and formats a label for the dataset.

        It is intended for internal use within the context of generating ROC-related data.

        Args:
            i: int
                The index of the dataset being processed, used for formatting fallback names.
            dataset: Dict
                A dataset containing data points and their classes.
            filter_object: Dict
                A filter configuration object used to calculate relevant dataset scores.

        Returns:
            Tuple[np.ndarray, np.ndarray, float, str]
                A tuple containing the following elements:
                    - fpr: False positive rates as a NumPy array.
                    - tpr: True positive rates as a NumPy array.
                    - auc_value: The calculated AUC value as a float.
                    - label: A formatted label string for the dataset.
        """
        scores = self._calculate_scores(filter_object, dataset)
        y_true = np.array(dataset["class"])

        fpr, tpr, auc_value = self._calculate_roc_data(y_true, scores)

        name = dataset.get("formattedName", f"Dataset {i + 1}")
        label = f"{name}: AUC={auc_value:.3f}"

        return fpr, tpr, auc_value, label

    def multiple_roc_plot(
        self,
        datasets: List[Dict],
        filter_object: Dict,
        title: str = "Multiple ROC Curves",
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generates a plot with multiple ROC curves for given datasets.

        This method creates a ROC (Receiver Operating Characteristic) plot for multiple datasets
        using their false positive rates and true positive rates. It allows customization of figure
        size, title, and an optional save path for the generated figure.

        Args:
            datasets (List[Dict]): A list of dataset dictionaries containing relevant data for
                generating ROC curves.
            filter_object (Dict): A dictionary used to filter or process the datasets.
            title (str): The title of the plot. Defaults to "Multiple ROC Curves".
            figsize (Optional[Tuple]): The size of the figure in inches as width and height.
                Defaults to (10, 8).
            save_path (Optional[str]): The file path to save the resulting plot image. If not
                provided, the plot is not saved.

        Returns:
            plt.Figure: The generated matplotlib figure containing the ROC plot.
        """
        figsize = figsize or (10, 8)
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.Set1(np.linspace(0, 1, len(datasets)))

        for i, dataset in enumerate(datasets):
            fpr, tpr, _, label = self._get_dataset_roc_info(i, dataset, filter_object)
            ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=label)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
        ax.set_xlabel("False Positive Rate (1-Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _get_dataset_prc_info(
        self, i: int, dataset: Dict, filter_object: Dict
    ) -> Tuple[np.ndarray, np.ndarray, float, str]:
        """
        Calculates precision-recall data and AUPRC (Area Under Precision-Recall Curve) for a specific dataset.

        This method takes a dataset, computes the scores using a provided filter object, and then generates
        precision, recall, and AUPRC statistics. It assigns a formatted label for the dataset to be used in
        visualizations or reporting.

        Args:
            i : int
                The index of the dataset in a collection of datasets.
            dataset : Dict
                A dictionary representing the dataset, which includes class labels under the key "class".
            filter_object : Dict
                A dictionary containing the filtering criteria or model parameters used to calculate the scores.

        Returns:
            Tuple[np.ndarray, np.ndarray, float, str]
                A tuple containing:
                - Precision values (numpy array)
                - Recall values (numpy array)
                - AUPRC (float)
                - Label for the dataset (string)
        """
        scores = self._calculate_scores(filter_object, dataset)
        y_true = np.array(dataset["class"])

        precision, recall, auprc = self._calculate_prc_stats(y_true, scores)

        name = dataset.get("formattedName", f"Dataset {i + 1}")
        label = f"{name}: AUPRC={auprc:.3f}"

        return precision, recall, auprc, label

    def multiple_prc_plot(
        self,
        datasets: List[Dict],
        filter_object: Dict,
        title: str = "Multiple PRC Curves",
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plots multiple Precision-Recall Curves (PRC) on the same graph for a list of datasets using
        custom filters and visualization parameters. Each dataset's PRC is computed, styled, and
        overlaid with distinct colors for clarity. The final figure can be optionally saved to a specified
        file path.

        Args:
            datasets: List[Dict]
                A list of datasets containing precision-recall data to plot. Each dataset should be formatted
                as a dictionary with keys and values necessary for processing PRC computations.
            filter_object: Dict
                A dictionary used for filtering or customizing the datasets during the generation of the
                Precision-Recall curves.
            title: str, optional
                The title of the plot. Defaults to "Multiple PRC Curves".
            figsize: Optional[Tuple], optional
                The size of the figure in the format (width, height). Defaults to (10, 8) if not provided.
            save_path: Optional[str], optional
                The file path where the generated figure will be saved. If None, the plot is not saved
                to a file. Defaults to None.

        Returns:
            matplotlib.figure.Figure
                The generated matplotlib figure containing the multiple PRC plot.
        """
        figsize = figsize or (10, 8)
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.Set1(np.linspace(0, 1, len(datasets)))

        for i, dataset in enumerate(datasets):
            precision, recall, _, label = self._get_dataset_prc_info(i, dataset, filter_object)
            ax.plot(recall, precision, color=colors[i], linewidth=2, label=label)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _get_gene_indices(self, dataset_keys: List[str], target_genes: List[str]) -> List[int]:
        """
        Gets the indices of target genes from a list of dataset keys.

        This function filters the indices of genes in 'dataset_keys' that match genes
        in the 'target_genes'. The indices of the matching genes are returned as a list.

        Args:
            dataset_keys: List[str]
                A list representing the keys of datasets.
            target_genes: List[str]
                A list of target genes to be searched in the dataset keys.

        Returns:
            List[int]
                A list containing the indices of the target genes in the dataset keys.
        """
        target_set = set(target_genes)
        return [i for i, key in enumerate(dataset_keys) if key in target_set]

    def _get_mean_expression(
        self, expr_matrix: Union[pd.DataFrame, np.ndarray], indices: List[int]
    ) -> Union[pd.Series, np.ndarray, float]:
        """
        Calculates the mean expression values for selected indices in an expression matrix.

        This method computes the mean of expression values based on provided indices.
        It supports two different input types for the expression matrix: pandas DataFrame or NumPy ndarray.
        If the input is a DataFrame, the method calculates the mean using the provided row indices.
        If the input is an ndarray, it calculates the mean across the specified rows of the array.

        Args:
            expr_matrix: Union[pd.DataFrame, np.ndarray]
                The expression matrix containing the data for computation. Can be a pandas DataFrame or a
                NumPy ndarray.
            indices: List[int]
                A list of integer indices to use for extracting rows from the expression matrix.

        Returns:
            Union[pd.Series, np.ndarray, float]
                The mean expression values calculated from the selected rows. If no indices are provided
                (empty list), a float value of 0.0 is returned. If the input is a DataFrame, the result
                is a pandas Series. If the input is an ndarray, the result is a NumPy ndarray.
        """
        if not indices:
            return 0.0

        if isinstance(expr_matrix, pd.DataFrame):
            return expr_matrix.iloc[indices].mean(axis=0)

        return np.mean(expr_matrix[indices, :], axis=0)

    def _calculate_scores(self, filter_object: Dict, dataset: Dict) -> np.ndarray:
        """
        Calculates scores based on positive and negative gene expressions.

        This method computes scores for a given dataset by calculating the mean gene
        expression for both positive and negative genes. The scores are derived as
        the difference between the mean expressions of positive and negative genes.

        Args:
            filter_object: A dictionary containing filter criteria including
                'posGeneNames' and 'negGeneNames' keys with lists of gene names as values.
            dataset: A dictionary containing the keys and expression matrix of genes.
                'keys' is the list of gene names and 'expr' is the expression matrix.

        Returns:
            np.ndarray: An array of scores calculated as the difference between the
            mean positive and mean negative gene expressions.
        """
        gene_keys = dataset["keys"]
        expr_matrix = dataset["expr"]

        # Simple implementation: mean of positive genes - mean of negative genes
        pos_indices = self._get_gene_indices(gene_keys, filter_object.get("posGeneNames", []))
        neg_indices = self._get_gene_indices(gene_keys, filter_object.get("negGeneNames", []))

        mean_pos_expr = self._get_mean_expression(expr_matrix, pos_indices)
        mean_neg_expr = self._get_mean_expression(expr_matrix, neg_indices)

        # Ensure we return an array even when both means are scalars
        scores = mean_pos_expr - mean_neg_expr
        if isinstance(scores, (int, float)):
            # If both indices are empty, return array of zeros with correct shape
            n_samples = (
                expr_matrix.shape[1] if hasattr(expr_matrix, "shape") else len(expr_matrix.columns)
            )
            return np.zeros(n_samples)

        return np.array(scores)

    def violin_plot(
        self,
        filter_object: Dict,
        dataset_object: Dict,
        label_column: str = "label",
        comparison_method: str = "wilcoxon",
        pairwise_comparisons: bool = True,
        auto_line_break: bool = True,
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compare groups within a single dataset using a violin plot.

        Given a filterObject and a datasetObject, this function uses the selected genes
        to calculate and compare the z-scores of the groups (e.g. cases vs. controls)
        by generating a violin plot. A violin plot is similar to a box plot, except
        the width of each violin is proportional to the density of points.

        Args:
            filter_object (Dict): MetaFilter object containing signature genes for z-score calculation.
                Must have 'posGeneNames' and/or 'negGeneNames' keys.
            dataset_object (Dict): Dataset object for group comparison. Must contain:
                - 'pheno': DataFrame with phenotype data
                - 'expr': DataFrame with expression data
                - 'keys': List/array of gene keys
                - 'formattedName': (optional) Display name for the dataset
            label_column (str): Column in pheno that specifies groups to compare (default: "label")
            comparison_method (str): Statistical test to use. Options: "wilcoxon", "t.test" (default: "wilcoxon")
            pairwise_comparisons (bool): If True, perform pairwise comparisons against first group.
                If False, perform global comparison (default: True)
            auto_line_break (bool): If True, insert line breaks into long labels (default: True)
            figsize (Optional[Tuple]): Figure size (width, height) in inches
            save_path (Optional[str]): Path to save the figure

        Returns:
            plt.Figure: Matplotlib figure object

        Raises:
            ValueError: If required columns are missing or data is invalid

        Examples:
            >>> plotter = EnhancedMetaAnalysisPlotter()
            >>> filter_obj = {'posGeneNames': ['GENE1', 'GENE2'],
            ...               'negGeneNames': ['GENE3']}
            >>> dataset = {'expr': expr_df, 'keys': keys, 'pheno': pheno_df,
            ...           'formattedName': 'Study 1'}
            >>> fig = plotter.violin_plot(filter_obj, dataset, label_column="group")
        """
        # 20260611dev
        from metaintegrator.score_calculation import ScoreCalculator

        figsize = figsize or self.figsize

        # Validate inputs
        if "pheno" not in dataset_object:
            raise ValueError("dataset_object must contain 'pheno' DataFrame")

        dataset_pheno = dataset_object["pheno"].copy()

        # Check label column exists
        if label_column not in dataset_pheno.columns:
            raise ValueError(f"Column '{label_column}' not found in dataset_object['pheno']")

        # Calculate scores
        scores = ScoreCalculator.calculate_score(
            filter_object, dataset_object, suppress_messages=True
        )
        dataset_pheno["score"] = scores
        dataset_pheno["group"] = dataset_pheno[label_column]

        # Create the plot
        formatted_name = dataset_object.get("formattedName", "")
        return self._create_violin_plot(
            dataset_pheno,
            comparison_column="group",
            score_column="score",
            formatted_name=formatted_name,
            comparison_method=comparison_method,
            pairwise_comparisons=pairwise_comparisons,
            auto_line_break=auto_line_break,
            figsize=figsize,
            save_path=save_path,
        )

    def _create_violin_plot(
        self,
        dataset_pheno: pd.DataFrame,
        comparison_column: str = "group",
        score_column: str = "score",
        formatted_name: str = "",
        comparison_method: str = "wilcoxon",
        pairwise_comparisons: bool = True,
        auto_line_break: bool = True,
        figsize: Tuple = (10, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Internal method to create violin plot.

        Args:
            dataset_pheno (pd.DataFrame): DataFrame with scores and group labels
            comparison_column (str): Column name for group labels
            score_column (str): Column name for scores
            formatted_name (str): Title for the plot
            comparison_method (str): "wilcoxon" or "t.test"
            pairwise_comparisons (bool): Pairwise vs global comparisons
            auto_line_break (bool): Break long labels
            figsize (Tuple): Figure size
            save_path (Optional[str]): Path to save figure

        Returns:
            plt.Figure: Matplotlib figure
        """
        # Visualization parameters
        jitter_width = 0.025
        segment_width = 0.05

        # Prepare data
        plot_data = pd.DataFrame(
            {"x": dataset_pheno[comparison_column], "y": dataset_pheno[score_column]}
        )

        # Process labels
        plot_data["x"] = plot_data["x"].astype(str).str.capitalize()
        if auto_line_break:
            plot_data["x"] = plot_data["x"].apply(self._add_line_breaks)

        # Get unique groups and sort
        groups = sorted(plot_data["x"].unique())
        plot_data["x"] = pd.Categorical(plot_data["x"], categories=groups, ordered=True)

        # Calculate standard errors for each group
        se_data = plot_data.groupby("x", observed=False)["y"].agg(["mean", "sem"]).reset_index()
        se_data = se_data.rename(columns={"mean": "y_mean", "sem": "se"})

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create violin plot
        parts = ax.violinplot(
            [plot_data[plot_data["x"] == g]["y"].values for g in groups],
            positions=range(len(groups)),
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        # Style violins
        for pc in parts["bodies"]:
            pc.set_facecolor("grey")
            pc.set_alpha(0.3)

        # Add jittered points
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        for i, group in enumerate(groups):
            y_data = plot_data[plot_data["x"] == group]["y"].values
            x_data = np.random.normal(i, jitter_width, size=len(y_data))
            ax.scatter(x_data, y_data, alpha=0.6, s=50, color=colors[i], zorder=3)

        # Add error bars (standard error)
        for i, row in se_data.iterrows():
            x_pos = i
            y_mean = row["y_mean"]
            se = row["se"]

            # Mean point
            ax.plot(x_pos, y_mean, "ko", markersize=8, zorder=4)

            # Error bars
            ax.plot(
                [x_pos - segment_width, x_pos + segment_width],
                [y_mean + se, y_mean + se],
                "k-",
                linewidth=2,
            )
            ax.plot(
                [x_pos - segment_width, x_pos + segment_width],
                [y_mean - se, y_mean - se],
                "k-",
                linewidth=2,
            )
            ax.plot([x_pos, x_pos], [y_mean - se, y_mean + se], "k-", linewidth=2)

        # Statistical comparisons
        if len(groups) > 1:
            self._add_statistical_annotations(
                ax, plot_data, groups, comparison_method, pairwise_comparisons
            )

        # Formatting
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, fontsize=12)
        ax.set_ylabel("Meta Score", fontsize=14)
        ax.set_xlabel(comparison_column.capitalize(), fontsize=14)
        ax.set_title(formatted_name, fontsize=16)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _add_line_breaks(self, text: str, max_length: int = 10) -> str:
        """Add line breaks to long labels."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) > max_length and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    def _add_statistical_annotations(
        self,
        ax,
        plot_data: pd.DataFrame,
        groups: List[str],
        comparison_method: str,
        pairwise: bool,
    ):
        """Add statistical comparison annotations to violin plot."""
        from scipy.stats import mannwhitneyu, ttest_ind

        # Choose statistical test
        if comparison_method.lower() in ["wilcoxon", "wilcox.test"]:
            test_func = mannwhitneyu
        elif comparison_method.lower() in ["t.test", "t_test", "ttest"]:
            test_func = lambda x, y: ttest_ind(x, y, equal_var=True)
        else:
            warnings.warn(f"Unknown comparison method: {comparison_method}, using Wilcoxon")
            test_func = mannwhitneyu

        # Perform comparisons
        y_max = plot_data["y"].max()
        y_range = plot_data["y"].max() - plot_data["y"].min()
        h = y_range * 0.05  # Height for annotation lines

        if pairwise and len(groups) > 1:
            # Pairwise comparisons against first group
            baseline_group = groups[0]
            baseline_data = plot_data[plot_data["x"] == baseline_group]["y"].values

            for i, group in enumerate(groups[1:], start=1):
                group_data = plot_data[plot_data["x"] == group]["y"].values

                try:
                    if test_func == mannwhitneyu:
                        stat, p_value = test_func(baseline_data, group_data)
                    else:
                        stat, p_value = test_func(baseline_data, group_data)

                    # Add annotation
                    y_pos = y_max + h * (i + 1)
                    self._draw_significance_bracket(ax, 0, i, y_pos, p_value)
                except Exception as e:
                    warnings.warn(f"Could not perform comparison: {e}")

    def _draw_significance_bracket(self, ax, x1: float, x2: float, y: float, p_value: float):
        """Draw significance bracket with p-value annotation."""
        # Determine significance stars
        if p_value < 0.001:
            sig_text = "***"
        elif p_value < 0.01:
            sig_text = "**"
        elif p_value < 0.05:
            sig_text = "*"
        else:
            sig_text = "ns"

        # Draw bracket
        ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], "k-", linewidth=1)

        # Add text
        ax.text((x1 + x2) / 2, y + 0.03, sig_text, ha="center", va="bottom", fontsize=12)

    def regression_plot(
        self,
        filter_object: Dict,
        dataset_object: Dict,
        continuous_variable_column: str = "continuous",
        formatted_variable_name: str = "Continuous Variable",
        cor_method: str = "pearson",
        correlation_corner: str = "bottomRight",
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate a regression plot comparing Meta Score with a continuous variable.

        This function creates a scatter plot with a regression line showing the relationship
        between the meta score (calculated from the gene signature) and a continuous phenotype
        variable (e.g., age, disease severity, biomarker level).

        Args:
            filter_object (Dict): MetaFilter object containing signature genes
            dataset_object (Dict): Dataset object containing:
                - 'pheno': DataFrame with phenotype data
                - 'expr': Expression data
                - 'keys': Gene keys
                - 'formattedName': (optional) Display name
            continuous_variable_column (str): Column name in pheno for continuous variable (default: "continuous")
            formatted_variable_name (str): Label for x-axis (default: "Continuous Variable")
            cor_method (str): Correlation method - "pearson" or "spearman" (default: "pearson")
            correlation_corner (str): Position for correlation text: "topLeft", "topRight",
                "bottomLeft", "bottomRight" (default: "bottomRight")
            figsize (Optional[Tuple]): Figure size (width, height) in inches
            save_path (Optional[str]): Path to save the figure

        Returns:
            plt.Figure: Matplotlib figure object

        Raises:
            ValueError: If required columns are missing

        Examples:
            >>> plotter = EnhancedMetaAnalysisPlotter()
            >>> filter_obj = {'posGeneNames': ['GENE1'], 'negGeneNames': []}
            >>> dataset = {'pheno': pheno_df, 'expr': expr_df, 'keys': keys}
            >>> fig = plotter.regression_plot(filter_obj, dataset,
            ...                              continuous_variable_column="age",
            ...                              formatted_variable_name="Age (years)")
        """
        from metaintegrator.score_calculation import ScoreCalculator

        figsize = figsize or self.figsize

        # Validate inputs
        if "pheno" not in dataset_object:
            raise ValueError("dataset_object must contain 'pheno' DataFrame")

        dataset_pheno = dataset_object["pheno"].copy()

        # Check continuous variable column exists
        if continuous_variable_column not in dataset_pheno.columns:
            raise ValueError(
                f"Column '{continuous_variable_column}' not found in dataset_object['pheno']"
            )

        # Calculate scores
        scores = ScoreCalculator.calculate_score(
            filter_object, dataset_object, suppress_messages=True
        )
        dataset_pheno["score"] = scores

        # ------------------------------------------------------------------
        # Convert the continuous variable column to a numeric vector.
        # The caller sometimes passes a *mixed* column that contains a
        # textual description together with a numeric value (e.g. "age: 61").
        # We try to extract a number from the string and, if that fails,
        # keep the original column unchanged.
        # ------------------------------------------------------------------
        continuous_col = dataset_pheno[continuous_variable_column]
        # If the column is already numeric, keep it as is

        if pd.api.types.is_numeric_dtype(continuous_col):
            dataset_pheno["continuous"] = continuous_col
        else:
            # Try to pull a numeric value from the string
            # Example pattern: 'age: 61' → 61
            extracted = continuous_col.astype(str).str.extract(r"age:\s*([0-9]+(?:\.[0-9]+)?)")[0]
            # If extraction succeeded, use the extracted value, otherwise fall back
            # to the original column (coerce to numeric will create NaNs).
            dataset_pheno["continuous"] = pd.to_numeric(extracted, errors="coerce")

            # If the extraction produced only NaNs (no 'age:' pattern found)
            # fall back to the original column.
            if dataset_pheno["continuous"].isna().all():
                dataset_pheno["continuous"] = pd.to_numeric(continuous_col, errors="coerce")

        # Ensure the score column is numeric (fails silently otherwise)
        dataset_pheno["score"] = pd.to_numeric(dataset_pheno["score"], errors="coerce")

        # Create the plot
        formatted_name = dataset_object.get("formattedName", "")
        return self._create_regression_plot(
            dataset_pheno,
            comparison_column="continuous",
            score_column="score",
            formatted_name=formatted_name,
            formatted_variable_name=formatted_variable_name,
            cor_method=cor_method,
            correlation_corner=correlation_corner,
            figsize=figsize,
            save_path=save_path,
        )

    def _create_regression_plot(
        self,
        dataset_pheno: pd.DataFrame,
        comparison_column: str = "continuous",
        score_column: str = "score",
        formatted_name: str = "",
        formatted_variable_name: str = "Continuous Variable",
        cor_method: str = "pearson",
        correlation_corner: str = "bottomRight",
        figsize: Tuple = (10, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Internal method to create regression plot.

        Args:
            dataset_pheno (pd.DataFrame): DataFrame with scores and continuous variable
            comparison_column (str): Column name for continuous variable
            score_column (str): Column name for scores
            formatted_name (str): Title for the plot
            formatted_variable_name (str): X-axis label
            cor_method (str): "pearson" or "spearman"
            correlation_corner (str): Position for correlation text
            figsize (Tuple): Figure size
            save_path (Optional[str]): Path to save figure

        Returns:
            plt.Figure: Matplotlib figure
        """

        # Prepare data
        plot_data = dataset_pheno[[comparison_column, score_column]].copy()

        # Convert both columns to numeric (drop rows that cannot be converted)
        plot_data[comparison_column] = pd.to_numeric(plot_data[comparison_column], errors="coerce")
        plot_data[score_column] = pd.to_numeric(plot_data[score_column], errors="coerce")
        plot_data = plot_data.dropna(subset=[comparison_column, score_column])

        x = plot_data[comparison_column].values
        y = plot_data[score_column].values

        # Calculate correlation
        if cor_method.lower() == "pearson":
            from scipy.stats import pearsonr

            corr, p_value = pearsonr(x, y)
        elif cor_method.lower() == "spearman":
            from scipy.stats import spearmanr

            corr, p_value = spearmanr(x, y)
        else:
            warnings.warn(f"Unknown correlation method: {cor_method}, using Pearson")
            from scipy.stats import pearsonr

            corr, p_value = pearsonr(x, y)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=50, color="steelblue", edgecolors="black", linewidth=0.5)

        # Regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=2, label="Regression line")

        # Confidence interval for regression line (optional enhancement)
        from scipy import stats as sp_stats

        predict_y = p(x)
        residuals = y - predict_y
        std_err = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
        margin = 1.96 * std_err  # 95% confidence interval
        ax.fill_between(
            x_line,
            p(x_line) - margin,
            p(x_line) + margin,
            alpha=0.2,
            color="red",
            label="95% CI",
        )

        # Add correlation text
        corr_text = f"R = {corr:.3f}\np = {p_value:.3e}"

        # Determine text position
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        if correlation_corner.lower() == "bottomright":
            x_pos, y_pos, ha, va = x_lim[1], y_lim[0], "right", "bottom"
        elif correlation_corner.lower() == "bottomleft":
            x_pos, y_pos, ha, va = x_lim[0], y_lim[0], "left", "bottom"
        elif correlation_corner.lower() == "topleft":
            x_pos, y_pos, ha, va = x_lim[0], y_lim[1], "left", "top"
        else:  # topRight
            x_pos, y_pos, ha, va = x_lim[1], y_lim[1], "right", "top"

        ax.text(
            x_pos,
            y_pos,
            corr_text,
            transform=ax.transData,
            fontsize=12,
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Formatting
        ax.set_xlabel(formatted_variable_name, fontsize=14)
        ax.set_ylabel("Meta Score", fontsize=14)
        ax.set_title(formatted_name, fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left" if "bottom" in correlation_corner.lower() else "lower left")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


# ------------------------------------------------------------------
# Convenience functions for backward compatibility
# ------------------------------------------------------------------
def forest_plot(meta_object: Dict, gene_name: str, **kwargs) -> plt.Figure:
    """
    Generates a forest plot based on the provided meta-analysis data and gene name.

    This function utilizes an EnhancedMetaAnalysisPlotter to create a visual representation
    of meta-analysis results, typically used in the context of genetic studies. The
    forest plot provides a graphical representation of individual studies and their
    combined effect size, helping researchers understand the variability and overall
    trend.

    Args:
        meta_object (Dict): A dictionary containing meta-analysis results,
            including effect sizes, confidence intervals, and other statistical
            information.
        gene_name (str): The name of the gene for which the meta-analysis is
            conducted.
        **kwargs: Additional keyword arguments passed to the plotting method,
            allowing customization of the generated forest plot.

    Returns:
        plt.Figure: A matplotlib figure object containing the forest plot.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.forest_plot(meta_object, gene_name, **kwargs)


def gg_forest_plot(meta_object: Dict, genes: List[str], **kwargs) -> plt.Figure:
    """
    Generate a forest plot using meta-analysis data and specified genes.

    This function creates a forest plot for visualizing results from a
    meta-analysis using the provided meta-object and a set of genes.
    Additional customizations for the plot can be provided through keyword
    arguments.

    Args:
        meta_object (Dict): The meta-analysis data structure containing the
            relevant data for the forest plot.
        genes (List[str]): A list of gene identifiers for which the forest
            plot will be generated.
        **kwargs: Arbitrary keyword arguments for customizing the plot.

    Returns:
        plt.Figure: A matplotlib Figure object containing the generated
            forest plot.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.gg_forest_plot(meta_object, genes, **kwargs)


def heatmap_plot(meta_object: Dict, filter_object: Dict, **kwargs) -> plt.Figure:
    """
    Generate a heatmap plot based on the provided meta-analysis data and filter criteria.

    Args:
        meta_object : Dict
            The metadata object containing data required for plotting.
        filter_object : Dict
            The filter criteria to refine data used in the heatmap plotting.
        **kwargs
            Additional keyword arguments passed to the underlying plotting function.

    Returns:
        plt.Figure
            A matplotlib Figure object representing the generated heatmap plot.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.heatmap_plot(meta_object, filter_object, **kwargs)


def manhattan_plot(meta_object: Dict, **kwargs) -> plt.Figure:
    """
    Generate a Manhattan plot based on a meta-analysis object and additional keyword arguments.

    This function utilizes the EnhancedMetaAnalysisPlotter class to produce a Manhattan plot
    for visualizing genetic association study results. The input meta_object contains
    relevant meta-analysis data, while additional configurations can be provided through
    keyword arguments.

    Args:
        meta_object (Dict): A dictionary containing meta-analysis data. The structure and
            contents of this dictionary should be compatible with the requirements of the
            EnhancedMetaAnalysisPlotter class.
        kwargs: Additional keyword arguments passed directly to the plotting function of
            the EnhancedMetaAnalysisPlotter.

    Returns:
        plt.Figure: A Matplotlib Figure object containing the generated Manhattan plot.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.manhattan_plot(meta_object, **kwargs)


def predictive_value_plot(sensitivity: float, specificity: float, **kwargs) -> plt.Figure:
    """
    Generates a predictive value plot using given sensitivity and specificity values.

    This function utilizes an EnhancedMetaAnalysisPlotter object to create a plot that
    visualizes the predictive value based on the provided sensitivity and specificity
    values. Additional keyword arguments can be passed for further customization
    of the plot.

    Args:
        sensitivity (float): The sensitivity value to be used for the plot.
        specificity (float): The specificity value to be used for the plot.
        **kwargs: Additional keyword arguments for plot customization.

    Returns:
        plt.Figure: The matplotlib Figure object containing the predictive value plot.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.predictive_value_plot(sensitivity, specificity, **kwargs)


def multiple_roc_plot(datasets: List[Dict], filter_object: Dict, **kwargs) -> plt.Figure:
    """
    Generates a single ROC curve plot for multiple datasets using a EnhancedMetaAnalysisPlotter.

    This function uses an EnhancedMetaAnalysisPlotter to create a ROC curve plot for the
    provided datasets applying a specified filter. Additional customization can
    be passed via keyword arguments.

    Args:
        datasets: List[Dict]
            A list containing dictionaries, each representing a dataset to be
            included in the ROC plot.
        filter_object: Dict
            A dictionary specifying the filter criteria to be applied to the
            datasets.
        **kwargs:
            Additional keyword arguments passed to the plotting function for
            customization.

    Returns:
        plt.Figure:
            The resulting matplotlib Figure containing the ROC curve plot.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.multiple_roc_plot(datasets, filter_object, **kwargs)


def multiple_prc_plot(datasets: List[Dict], filter_object: Dict, **kwargs) -> plt.Figure:
    """
    Generates a precision-recall curve (PRC) plot for multiple datasets.

    This function facilitates the creation of a precision-recall curve
    (PRC) plot using data from multiple datasets. It utilizes the
    EnhancedMetaAnalysisPlotter class to handle the actual plotting. Additional
    filtering or configuration can be applied through the provided filter
    object and keyword arguments.

    Args:
        datasets: List of dictionaries representing the datasets to be
            visualized on the PRC curve.
        filter_object: Dictionary containing filtering criteria for
            processing the datasets.
        **kwargs: Arbitrary keyword arguments passed to the underlying
            plotting function for customization.

    Returns:
        plt.Figure: A matplotlib figure containing the PRC plot.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.multiple_prc_plot(datasets, filter_object, **kwargs)


def roc_plot(y_true: np.ndarray, y_scores: np.ndarray, **kwargs) -> plt.Figure:
    """
    Plots the Receiver Operating Characteristic (ROC) curve using the given true binary
    labels and predicted probability scores. This helps visualize the trade-off between
    the true positive rate and false positive rate at various classification thresholds.

    Args:
        y_true (np.ndarray): Array of true binary labels (0/1 or False/True) for the dataset.
        y_scores (np.ndarray): Array of predicted probability scores corresponding to each
            instance in y_true.
        **kwargs: Additional keyword arguments to customize the ROC plot.

    Returns:
        plt.Figure: A matplotlib Figure object representing the generated ROC curve plot.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.roc_plot(y_true, y_scores, **kwargs)


def prc_plot(y_true: np.ndarray, y_scores: np.ndarray, **kwargs) -> plt.Figure:
    """
    Generate a Precision-Recall Curve (PRC) plot for the provided true values and
    predicted scores.

    This function creates a precision-recall curve using the given true binary
    labels and the predicted scores. It delegates the plotting to an internal
    utility capable of handling precision-recall visualizations, allowing for
    additional customization through keyword arguments.

    Args:
        y_true (np.ndarray): A 1-dimensional NumPy array containing the true binary
            labels (0 or 1) of the dataset.
        y_scores (np.ndarray): A 1-dimensional NumPy array containing the predicted
            scores or probabilities corresponding to each instance in the dataset.
        **kwargs: Additional keyword arguments that can be passed to customize the
            underlying plotting mechanism.

    Returns:
        plt.Figure: A Matplotlib figure object representing the generated precision-
            recall curve.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.prc_plot(y_true, y_scores, **kwargs)


# ------------------------------------------------------------------
# Backward-compatible deprecated alias
# ------------------------------------------------------------------
# This class used to be named ``MetaAnalysisPlotter`` — the same name as
# the unrelated, differently-scoped class in :mod:`visualization.plotting`
# (volcano/forest/heatmap/effect-size-distribution/study-overview/
# MA-summary plots). Having two independent classes share one name across
# sibling modules was a real footgun: `from visualization.enhanced_plotting
# import MetaAnalysisPlotter` and `from visualization.plotting import
# MetaAnalysisPlotter` silently returned two different classes with no
# overlapping methods. It is now ``EnhancedMetaAnalysisPlotter``.
#
# `__getattr__` (PEP 562) keeps `from visualization.enhanced_plotting
# import MetaAnalysisPlotter` working for existing callers, with a
# DeprecationWarning nudging them to the new name. New code should import
# `EnhancedMetaAnalysisPlotter` directly.
def __getattr__(name: str):
    if name == "MetaAnalysisPlotter":
        warnings.warn(
            "visualization.enhanced_plotting.MetaAnalysisPlotter has been "
            "renamed to EnhancedMetaAnalysisPlotter to avoid colliding "
            "with the unrelated visualization.plotting.MetaAnalysisPlotter "
            "class. Update your import: "
            "`from visualization.enhanced_plotting import "
            "EnhancedMetaAnalysisPlotter`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return EnhancedMetaAnalysisPlotter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
