"""
Enhanced plotting functions for MetaIntegrator Python port.
This module provides comprehensive visualization capabilities including
forest plots, heatmaps, Manhattan plots, and all plotting functions from the R version.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Polygon
from scipy import stats
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Default cap on the number of genes rendered as heatmap rows. With no
# upper bound, `_calculate_heatmap_figsize` scales figure height linearly
# with gene count (`n_genes * 0.3` inches); on an unfiltered/raw gene set
# (thousands of genes, as opposed to a gene-search-optimized panel of a few
# dozen) that produces a figure well over 1,000 inches tall. Rendering that
# at `savefig(dpi=300)` requires a multi-hundred-thousand-pixel-tall canvas,
# which reliably OOM-kills the process (SIGKILL, exit code 137: not a
# catchable Python exception) partway through `heatmap_plot()`, before the
# violin/ROC/PRC plots that are supposed to run afterward ever get a chance
# to. Capping to the top genes by |effect size| keeps the heatmap both
# renderable and actually readable (a 5,000-row heatmap is illegible either
# way); see `_prepare_heatmap_data`'s `max_genes` parameter.
DEFAULT_HEATMAP_MAX_GENES = 100


# ------------------------------------------------------------------
# EnhancedMetaAnalysisPlotter Class
# ------------------------------------------------------------------
class EnhancedMetaAnalysisPlotter:
    """Enhanced plotting class for meta-analysis visualizations."""

    CI_95_MULTIPLIER = 1.96

    # Hard upper bound (inches) on any heatmap figure dimension. Backstop
    # for `_calculate_heatmap_figsize`: see that method's docstring.
    _MAX_HEATMAP_DIMENSION_INCHES = 100.0

    @staticmethod
    def _apply_equal_unit_axes(ax: plt.Axes, step: float = 0.1) -> None:
        """Force equal-scaled, evenly-gridded 0-1 axes on a 0-1 ROC/PRC plot.

        A ROC or precision-recall curve plotted with independently
        scaled x/y axes can visually mislead an "at-a-glance" reader --
        the same underlying curve can look steeper or shallower purely
        because Matplotlib chose different pixel-per-unit scales for
        each axis, not because of any real difference in performance.
        ``ax.set_aspect("equal", adjustable="box")`` makes one data unit
        on x equal one data unit on y in on-screen size, so a 0-1 axis
        range renders as a true square regardless of the figure's
        overall width/height; ``set_xticks``/``set_yticks`` at a fixed
        ``step`` additionally keeps the two axes' gridlines at the same
        spacing, so equal-sized visual steps always represent equal
        changes in rate.

        Callers must set ``ax.set_xlim([0, 1])``/``ax.set_ylim([0, 1])``
        *before* calling this -- ``adjustable="box"`` reshapes the axes
        box to fit those limits equally rather than changing the data
        range itself.

        Args:
            ax: A Matplotlib ``Axes`` already limited to ``[0, 1]`` on
                both axes.
            step: Gridline/tick spacing on both axes (default ``0.1``,
                i.e. 11 evenly-spaced ticks from 0.0 to 1.0).
        """
        ax.set_aspect("equal", adjustable="box")
        # `linspace` + a rounding pass (not a bare `arange`/`linspace`)
        # deliberately: raw floating-point arithmetic on a 0-1 range
        # lands on values like 0.30000000000000004 instead of 0.3, which
        # would otherwise leak into tick labels. Rounding to 10 decimal
        # places is far finer than any `step` this helper is realistically
        # called with, so it only ever cleans up binary-float noise, never
        # the intended spacing itself.
        n_steps = int(round(1.0 / step))
        ticks = np.round(np.linspace(0.0, 1.0, n_steps + 1), decimals=10)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

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

    def _get_meta_data(self, meta_object: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Extract (effect sizes, standard errors, pooled results) from a meta-analysis object.

        Args:
            meta_object: Meta-analysis data with a "metaAnalysis" key.

        Raises:
            ValueError: If "metaAnalysis" is missing from meta_object.

        Returns:
            Dataset effect sizes, dataset effect size standard errors, and pooled results.
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
        """Look up each study's formatted display name, falling back to its raw name.

        Args:
            meta_object: Meta-analysis data with an "originalData" dict.
            studies: Study identifiers to resolve.

        Returns:
            The formatted (or raw, if unavailable) name for each study.
        """
        original_data = meta_object.get("originalData", {})
        return [original_data.get(s, {}).get("formattedName", s) for s in studies]

    def _draw_pooled_diamond(
        self,
        ax: plt.Axes,
        effect: float,
        se: float,
        y_pos: float,
        color: str,
        ci_multiplier: float = 1.96,
    ) -> None:
        """Draw a diamond marker for a pooled effect, its width spanning the confidence interval.

        Args:
            ax: Axes to draw on.
            effect: Pooled effect size (diamond center).
            se: Standard error of the pooled effect.
            y_pos: Vertical position of the diamond.
            color: Diamond fill color.
            ci_multiplier: Z-multiplier for the confidence interval width (default 1.96, ~95%).
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
        """Plot per-study effect sizes (as boxes, sized by inverse variance) and the pooled
        effect (as a diamond) for one gene.

        Args:
            meta_object: Meta-analysis data with effect sizes and standard errors per study.
            gene_name: Gene to plot.
            box_color: Color of individual-study effect boxes.
            whisker_color: Color of individual-study confidence-interval whiskers.
            zero_line_color: Color of the vertical zero-effect reference line.
            summary_color: Color of the pooled-effect diamond.
            text_color: Color of study-name labels.
            figsize: Figure size in inches. Defaults to self.figsize.
            save_path: If given, save the figure to this path.

        Returns:
            The generated matplotlib Figure.

        Raises:
            ValueError: If gene_name is not found in meta_object.
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
        """Plot a grid of forest plots (one facet per gene), each showing per-dataset
        effects and the pooled summary.

        Args:
            meta_object: Meta-analytic data with effect sizes and pooled results.
            genes: Genes to plot, one facet each.
            conf_level: Confidence level for the interval widths.
            facet_cols: Columns in the facet grid. Defaults based on gene count.
            box_scales: Scaling factors for box/text sizing (currently unused directly
                by this method's own drawing code; reserved for style tuning).
            figsize: Figure size in inches. Defaults based on gene count.
            save_path: If given, save the figure to this path.

        Returns:
            The generated matplotlib Figure.
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
        max_genes: Optional[int] = DEFAULT_HEATMAP_MAX_GENES,
    ) -> pd.DataFrame:
        """Filter effect sizes to the filter object's genes, optionally prepend pooled results,
        and order rows for the heatmap.

        Args:
            meta_data: Dict
                Metadata containing gene-related information, effect sizes, and pooled results.
            filter_object: Dict
                Object containing genes to filter by ("posGeneNames" and "negGeneNames").
            display_pooled: bool
                Flag indicating whether to include pooled results in the heatmap data.
            gene_order: Union[bool, List]
                List specifying the order of genes, or a boolean indicating if default ordering
                should be applied. Ignored for genes dropped by ``max_genes``.
            max_genes: Optional[int]
                Maximum number of genes to include, ranked by descending
                |effect size| (using the pooled effect size when available,
                otherwise the mean |effect size| across datasets). ``None``
                disables the cap. Defaults to
                :data:`DEFAULT_HEATMAP_MAX_GENES`: an unbounded gene count
                makes ``_calculate_heatmap_figsize``'s figure height grow
                linearly without limit, which can OOM-kill the process at
                ``savefig()`` for a raw (un-filtered-to-a-panel) gene set of
                several thousand genes. An explicit gene list passed via
                ``gene_order`` is *not* capped, since the caller asked for
                those specific genes.

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

        # Cap to the top genes by |effect size| *before* building heat_data,
        # so the (potentially very large) effect_sizes/pooled_results slices
        # below never exceed max_genes rows. An explicit gene_order list is
        # honored as-is: the caller chose those genes deliberately.
        if (
            max_genes is not None
            and not isinstance(gene_order, list)
            and len(gene_names) > max_genes
        ):
            if "effectSize" in pooled_results.columns:
                ranking = pooled_results.loc[gene_names, "effectSize"].abs()
            else:
                ranking = effect_sizes.loc[gene_names].abs().mean(axis=1)
            n_dropped = len(gene_names) - max_genes
            gene_names = ranking.sort_values(ascending=False).index[:max_genes].tolist()
            msg = (
                f"Heatmap: {n_dropped} gene(s) beyond the top {max_genes} "
                f"(ranked by |effect size|) were dropped to keep the figure "
                f"renderable and legible. Pass max_genes=None (or a larger "
                f"value) to heatmap_plot() to override."
            )
            logger.warning(msg)
            warnings.warn(msg)

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
        """Scale heatmap figure size to the gene/dataset count, clamped to a safe maximum.

        Args:
            data: Heatmap data, genes x datasets.

        Returns:
            (width, height) in inches, each clamped to `_MAX_HEATMAP_DIMENSION_INCHES`.
        """
        n_genes, n_datasets = data.shape
        width = max(8, n_datasets * 0.6)
        height = max(6, n_genes * 0.3)

        # Defense-in-depth backstop: `_prepare_heatmap_data`'s `max_genes`
        # cap is the primary fix for the unbounded-figure-size OOM crash
        # (SIGKILL at savefig() time for a ~1,500-inch-tall figure on a raw,
        # multi-thousand-gene set), but it can be bypassed by an explicit
        # `gene_order` list or `max_genes=None`. Clamp here too so this
        # method can never itself hand back a figure size large enough to
        # OOM the process, regardless of how `data` was produced.
        if (
            width > self._MAX_HEATMAP_DIMENSION_INCHES
            or height > self._MAX_HEATMAP_DIMENSION_INCHES
        ):
            clamp_msg = (
                f"Heatmap figure size ({width:.1f} x {height:.1f} in) exceeds the "
                f"{self._MAX_HEATMAP_DIMENSION_INCHES:.0f}-inch safety cap and was "
                f"clamped; genes/datasets may render at reduced legibility. Pass an "
                f"explicit figsize to override."
            )
            logger.warning(clamp_msg)
            warnings.warn(clamp_msg)
            width = min(width, self._MAX_HEATMAP_DIMENSION_INCHES)
            height = min(height, self._MAX_HEATMAP_DIMENSION_INCHES)

        return (width, height)

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
        max_genes: Optional[int] = DEFAULT_HEATMAP_MAX_GENES,
    ) -> Optional[plt.Figure]:
        """Plot a heatmap of effect sizes (genes x datasets), or return None if fewer
        than 2 genes are available to plot.

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
            max_genes (Optional[int], optional): Maximum number of genes to render, ranked by
                descending |effect size|. Protects against unbounded figure sizes (and the OOM
                crash they cause at save time) for large, un-optimized gene sets. Defaults to
                :data:`DEFAULT_HEATMAP_MAX_GENES` (100). Pass ``None`` to disable the cap, or
                pass an explicit gene list via ``gene_order`` to bypass it for those genes.

        Returns:
            Optional[plt.Figure]: The resulting heatmap figure object if successfully created, or None if the
                data does not meet the criteria for generating a heatmap.

        Raises:
            None
        """
        heat_data = self._prepare_heatmap_data(
            meta_data, filter_object, display_pooled, gene_order, max_genes=max_genes
        )

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
        """Plot -log10(FDR) per gene index, colored by significance at FDR < 0.05.

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
        """Build a title showing sensitivity/specificity as rounded percentages.

        Args:
            sensitivity: Sensitivity, typically between 0 and 1.
            specificity: Specificity, typically between 0 and 1.
            rounding: Decimal places to round the displayed percentages to.

        Returns:
            The formatted title string.
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
        """Compute PPV and NPV across a range of prevalence values, given sensitivity/specificity.

        Args:
            sensitivity: Test sensitivity (true positive rate).
            specificity: Test specificity (true negative rate).
            n_steps: Number of prevalence values to sample over [0.001, 0.999].

        Returns:
            (prevalence, ppv, npv) arrays, one value per step.
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
        """Plot PPV and NPV as a function of disease prevalence.

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

    def _calculate_auc_ci(self, y_true: np.ndarray, auc_value: float) -> Tuple[float, float]:
        """Estimate the 95% CI for an AUC score via its standard-error formula.

        Returns (0.0, 0.0) if either class has zero samples.

        Args:
            y_true: Binary ground-truth labels.
            auc_value: The AUC score to build a CI around.

        Returns:
            (lower, upper) bounds of the confidence interval.
        """
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.0, 0.0

        se_auc = np.sqrt((auc_value * (1 - auc_value)) / (n_pos * n_neg))
        ci_lower = max(0, auc_value - self.CI_95_MULTIPLIER * se_auc)
        ci_upper = min(1, auc_value + self.CI_95_MULTIPLIER * se_auc)
        return ci_lower, ci_upper

    def _calculate_roc_data(
        self, y_true: np.ndarray, y_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute the ROC curve (FPR, TPR) and AUC for binary labels and scores.

        Args:
            y_true: Binary ground-truth labels.
            y_scores: Predicted scores or probabilities for the positive class.

        Returns:
            (fpr, tpr, auc_value).
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
        """Plot the ROC curve with an AUC-and-95%-CI label.

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
        self._apply_equal_unit_axes(ax)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _calculate_auprc_confidence_interval(
        self, y_true: np.ndarray, auprc: float
    ) -> Tuple[float, float]:
        """Estimate the 95% CI for an AUPRC score via its standard-error formula.

        Returns a wide (0.0, 1.0) interval when there are no positive samples, since the
        standard-error formula is undefined in that case.

        Args:
            y_true: Binary ground-truth labels. 1 represents positive samples.
            auprc: The AUPRC value to build a CI around.

        Returns:
            (lower, upper) bounds of the confidence interval.
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
        """Return the no-skill PRC baseline: the positive-class fraction of y_true.

        Args:
            y_true: Binary ground-truth labels.

        Returns:
            The baseline precision (0.0 if y_true is empty).
        """
        return np.sum(y_true) / len(y_true) if len(y_true) > 0 else 0.0

    def _calculate_prc_stats(
        self, y_true: np.ndarray, y_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute the precision-recall curve and AUPRC for binary labels and scores.

        Args:
            y_true: Binary ground-truth labels.
            y_scores: Predicted scores or probabilities for the positive class.

        Returns:
            (precision, recall, auprc).
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
        """Plot the precision-recall curve with an AUPRC-and-95%-CI label, alongside the
        no-skill baseline.

        Args:
            y_true: Binary ground-truth labels.
            y_scores: Predicted scores or probabilities for the positive class.
            title: Plot title.
            label: Custom curve label. Defaults to "AUPRC = ... (95% CI: ...)".
            color: Curve line color.
            figsize: Figure size in inches. Defaults to self.figsize.
            save_path: If given, save the figure to this path.

        Returns:
            The generated matplotlib Figure.
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
        self._apply_equal_unit_axes(ax)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _get_dataset_roc_info(
        self, i: int, dataset: Dict, filter_object: Dict
    ) -> Tuple[np.ndarray, np.ndarray, float, str]:
        """Compute one dataset's ROC curve, AUC, and a "name: AUC=..." label for the legend.

        Args:
            i: Dataset index, used to build a fallback name ("Dataset {i+1}").
            dataset: Dataset dict with scores (or inputs to compute them) and class labels.
            filter_object: Filter used to compute signature scores.

        Returns:
            (fpr, tpr, auc_value, label).
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
        """Plot one ROC curve per dataset, overlaid on shared axes.

        Args:
            datasets: Datasets to plot, one curve each.
            filter_object: Filter used to compute each dataset's signature scores.
            title: Plot title.
            figsize: Figure size in inches. Defaults to (10, 8).
            save_path: If given, save the figure to this path.

        Returns:
            The generated matplotlib Figure.
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
        self._apply_equal_unit_axes(ax)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _get_dataset_prc_info(
        self, i: int, dataset: Dict, filter_object: Dict
    ) -> Tuple[np.ndarray, np.ndarray, float, str]:
        """Compute one dataset's precision-recall curve, AUPRC, and a legend label.

        Args:
            i: Dataset index, used to build a fallback name ("Dataset {i+1}").
            dataset: Dataset dict with a "class" key of binary labels.
            filter_object: Filter used to compute signature scores.

        Returns:
            (precision, recall, auprc, label).
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
        """Plot one precision-recall curve per dataset, overlaid on shared axes.

        Args:
            datasets: Datasets to plot, one curve each.
            filter_object: Filter used to compute each dataset's signature scores.
            title: Plot title.
            figsize: Figure size in inches. Defaults to (10, 8).
            save_path: If given, save the figure to this path.

        Returns:
            The generated matplotlib Figure.
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
        self._apply_equal_unit_axes(ax)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _get_gene_indices(self, dataset_keys: List[str], target_genes: List[str]) -> List[int]:
        """Return the positions in dataset_keys that match target_genes.

        Args:
            dataset_keys: Gene keys for a dataset, in expression-matrix row order.
            target_genes: Genes to find.

        Returns:
            Matching indices, in dataset_keys order.
        """
        target_set = set(target_genes)
        return [i for i, key in enumerate(dataset_keys) if key in target_set]

    def _get_mean_expression(
        self, expr_matrix: Union[pd.DataFrame, np.ndarray], indices: List[int]
    ) -> Union[pd.Series, np.ndarray, float]:
        """Average expression across the given row indices (genes), per sample.

        Args:
            expr_matrix: Expression data, genes x samples (DataFrame or ndarray).
            indices: Row indices to average across.

        Returns:
            Per-sample mean (a Series for a DataFrame input, an ndarray for an ndarray
            input), or 0.0 if `indices` is empty.
        """
        if not indices:
            return 0.0

        if isinstance(expr_matrix, pd.DataFrame):
            return expr_matrix.iloc[indices].mean(axis=0)

        return np.mean(expr_matrix[indices, :], axis=0)

    def _calculate_scores(self, filter_object: Dict, dataset: Dict) -> np.ndarray:
        """
         Calculates per-sample signature scores for a dataset.

         consolidated onto the single "gold standard" scoring
         implementation. This method used to compute its own simple
         mean-difference score (mean of positive genes minus mean of negative
         genes, no normalization), independently of
         :func:`metaintegrator.score_calculation.ScoreCalculator.calculate_score`
        : the R-parity, geometric-mean-based scorer used by the CLI (stage 7,
         ``_compute_scores_and_roc``) and by ``violin_plot.py``. Having two
         divergent implementations meant the ROC/PRC AUCs shown on
         ``roc_multiple.png``/``prc_multiple.png`` could disagree with the AUCs
         derivable from ``dataset["score"]`` / the exported ``scores_*.csv``
         files for the *same* run. This method now delegates to
         ``ScoreCalculator.calculate_score()`` so every code path: CLI,
         programmatic/Jupyter use, and this plotting module: produces
         identical scores for the same filter and dataset.

         If ``dataset`` already carries a precomputed ``"score"`` (as it does
         after CLI stage 7), that cached value is reused instead of
         recomputing: the same short-circuit already used by
         ``violin_plot.py``'s ``_get_or_compute_scores()``.

         Args:
             filter_object: A dictionary containing filter criteria including
                 'posGeneNames' and 'negGeneNames' keys with lists of gene names as values.
             dataset: A dictionary containing the keys and expression matrix of genes.
                 'keys' is the list of gene names and 'expr' is the expression matrix.
                 If it already has a 'score' key with a non-``None`` value, that
                 value is returned (converted to an array) unchanged.

         Returns:
             np.ndarray: Per-sample signature scores (R-parity Z-scores when
             freshly computed; whatever ``dataset["score"]`` holds when reused).
        """
        if dataset.get("score") is not None:
            return np.asarray(dataset["score"], dtype=float)

        from metaintegrator.score_calculation import ScoreCalculator

        expr_matrix = dataset.get("expr")
        if isinstance(expr_matrix, np.ndarray):
            # ScoreCalculator requires dataset['expr'] to be a pandas
            # DataFrame. Gene lookup inside ScoreCalculator is driven
            # entirely by the separate 'keys' list, matched positionally
            # row-by-row against 'expr': not by the DataFrame's own index —
            # so wrapping a raw ndarray with a plain default-index DataFrame
            # is sufficient and preserves the original gene/sample order.
            dataset = dict(dataset)  # shallow copy: don't mutate the caller's dataset
            dataset["expr"] = pd.DataFrame(expr_matrix)

        return np.asarray(
            ScoreCalculator.calculate_score(filter_object, dataset, suppress_messages=True),
            dtype=float,
        )

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
        ax: plt.Axes,
        plot_data: pd.DataFrame,
        groups: List[str],
        comparison_method: str,
        pairwise: bool,
    ) -> None:
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

    def _draw_significance_bracket(
        self, ax: plt.Axes, x1: float, x2: float, y: float, p_value: float
    ) -> None:
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
    """Module-level convenience wrapper around EnhancedMetaAnalysisPlotter.gg_forest_plot.

    Args:
        meta_object: Meta-analytic data with effect sizes and pooled results.
        genes: Genes to plot, one facet each.
        **kwargs: Passed through to EnhancedMetaAnalysisPlotter.gg_forest_plot.

    Returns:
        The generated matplotlib Figure.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.gg_forest_plot(meta_object, genes, **kwargs)


def heatmap_plot(meta_object: Dict, filter_object: Dict, **kwargs) -> plt.Figure:
    """Module-level convenience wrapper around EnhancedMetaAnalysisPlotter.heatmap_plot.

    Args:
        meta_object: Meta-analysis data with effect sizes and pooled results.
        filter_object: Filter defining which genes to include.
        **kwargs: Passed through to EnhancedMetaAnalysisPlotter.heatmap_plot.

    Returns:
        The generated matplotlib Figure.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.heatmap_plot(meta_object, filter_object, **kwargs)


def manhattan_plot(meta_object: Dict, **kwargs) -> plt.Figure:
    """Module-level convenience wrapper around EnhancedMetaAnalysisPlotter.manhattan_plot.

    Args:
        meta_object: Meta-analysis data with pooled results.
        **kwargs: Passed through to EnhancedMetaAnalysisPlotter.manhattan_plot.

    Returns:
        The generated matplotlib Figure.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.manhattan_plot(meta_object, **kwargs)


def predictive_value_plot(sensitivity: float, specificity: float, **kwargs) -> plt.Figure:
    """Module-level convenience wrapper around EnhancedMetaAnalysisPlotter.predictive_value_plot.

    Args:
        sensitivity: Test sensitivity.
        specificity: Test specificity.
        **kwargs: Passed through to EnhancedMetaAnalysisPlotter.predictive_value_plot.

    Returns:
        The generated matplotlib Figure.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.predictive_value_plot(sensitivity, specificity, **kwargs)


def multiple_roc_plot(datasets: List[Dict], filter_object: Dict, **kwargs) -> plt.Figure:
    """Module-level convenience wrapper around EnhancedMetaAnalysisPlotter.multiple_roc_plot.

    Args:
        datasets: Datasets to plot, one ROC curve each.
        filter_object: Filter used to compute each dataset's signature scores.
        **kwargs: Passed through to EnhancedMetaAnalysisPlotter.multiple_roc_plot.

    Returns:
        The generated matplotlib Figure.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.multiple_roc_plot(datasets, filter_object, **kwargs)


def multiple_prc_plot(datasets: List[Dict], filter_object: Dict, **kwargs) -> plt.Figure:
    """Module-level convenience wrapper around EnhancedMetaAnalysisPlotter.multiple_prc_plot.

    Args:
        datasets: Datasets to plot, one precision-recall curve each.
        filter_object: Filter used to compute each dataset's signature scores.
        **kwargs: Passed through to EnhancedMetaAnalysisPlotter.multiple_prc_plot.

    Returns:
        The generated matplotlib Figure.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.multiple_prc_plot(datasets, filter_object, **kwargs)


def roc_plot(y_true: np.ndarray, y_scores: np.ndarray, **kwargs) -> plt.Figure:
    """Module-level convenience wrapper around EnhancedMetaAnalysisPlotter.roc_plot.

    Args:
        y_true: Binary ground-truth labels.
        y_scores: Predicted scores or probabilities for the positive class.
        **kwargs: Passed through to EnhancedMetaAnalysisPlotter.roc_plot.

    Returns:
        The generated matplotlib Figure.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.roc_plot(y_true, y_scores, **kwargs)


def prc_plot(y_true: np.ndarray, y_scores: np.ndarray, **kwargs) -> plt.Figure:
    """Module-level convenience wrapper around EnhancedMetaAnalysisPlotter.prc_plot.

    Args:
        y_true: Binary ground-truth labels.
        y_scores: Predicted scores or probabilities for the positive class.
        **kwargs: Passed through to EnhancedMetaAnalysisPlotter.prc_plot.

    Returns:
        The generated matplotlib Figure.
    """
    plotter = EnhancedMetaAnalysisPlotter()
    return plotter.prc_plot(y_true, y_scores, **kwargs)


# ------------------------------------------------------------------
# Backward-compatible deprecated alias
# ------------------------------------------------------------------
# This class used to be named ``MetaAnalysisPlotter``: the same name as
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
def __getattr__(name: str) -> type:
    """Resolve the deprecated ``MetaAnalysisPlotter`` alias (see comment above) with a warning."""
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
