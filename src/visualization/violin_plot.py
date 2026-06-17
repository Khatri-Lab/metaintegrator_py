"""
R-parity violin plot for MetaIntegrator Python.

Replicates R MetaIntegrator's violinPlot() which uses ggpubr + ggplot2:
  - Individual (non-split) violins per phenotype group
  - Boxplot overlay showing IQR and median
  - Jittered individual data points
  - Wilcoxon rank-sum test with p-value annotations
  - Clean, publication-ready ggplot2 aesthetic
  - Supports labelColumn for phenotype subgroups (e.g. "label", "group")

Usage in notebook
-----------------
    from visualization.violin_plot import violin_plot, multi_violin_plot

    # Single dataset (matches R's violinPlot exactly)
    fig = violin_plot(filter_object, sle_data['originalData']['GSE11909_GPL96'],
                      label_column='label', title='GSE11909_GPL96')

    # All datasets in a figure grid
    fig = multi_violin_plot(filter_object, sle_data, title='SLE Meta-Score')
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ── ggplot2-inspired palette ────────────────────────────────────────
# Soft, high-contrast colors that mirror R's default ggplot2 / ggpubr
_GGPLOT2_COLORS = [
    "#F8766D",  # coral-red  (group 1 / disease)
    "#00BA38",  # green      (group 2 / control),  # "#bcfce1" # softer green
    "#619CFF",  # cornflower (group 3)
    "#F564E3",  # orchid     (group 4)
    "#FFA500",  # orange     (group 5)
    "#00BFC4",  # teal       (group 6)
]

# Matching muted fill (alpha ~0.35 equivalent via hex)
_GGPLOT2_FILLS = [
    "#F8766D90",
    "#00BA3890",  # "#bcfce190"
    "#619CFF90",
    "#F564E390",
    "#FFA50090",
    "#00BFC490",
]


# ── Single-dataset violin ──────────────────────────────────────────
def violin_plot(
    filter_object: Dict,
    dataset_object: Dict,
    label_column: str = "label",
    comparison_method: str = "wilcox",
    show_points: bool = True,
    show_box: bool = True,
    show_pvalue: bool = True,
    figsize: Tuple[float, float] = (5, 5.5),
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    fills: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    R-parity violin plot for a single dataset.

    Matches R MetaIntegrator's ``violinPlot(filterObject, datasetObject,
    labelColumn)``: one violin per phenotype group, boxplot overlay,
    jittered points, Wilcoxon p-value bracket.

    Parameters
    ----------
    filter_object : dict
        MetaFilter with ``posGeneNames`` / ``negGeneNames``.
    dataset_object : dict
        Dataset with ``expr``, ``keys``, ``class``, ``pheno``, and
        optionally ``score``.
    label_column : str
        Column in ``pheno`` that defines the groups. Falls back to
        binary class labels (0→Control, 1→Disease) when the column
        is missing or has ≤1 unique value.
    comparison_method : str
        ``'wilcox'`` (Mann–Whitney U) or ``'t'`` (Welch's t-test).
    show_points : bool
        Overlay jittered data points.
    show_box : bool
        Overlay a narrow boxplot inside each violin.
    show_pvalue : bool
        Show pairwise p-value brackets (ggpubr style).
    figsize : tuple
        Figure size in inches.
    title : str, optional
        Plot title. Defaults to dataset's ``geo_id`` if available.
    colors, fills : list[str], optional
        Per-group edge / fill colors. Defaults to ggplot2 palette.
    ax : Axes, optional
        Draw on an existing Axes (for multi-panel figures).
    save_path : str, optional
        Save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── 1. Compute scores if needed ─────────────────────────────────
    scores = _get_or_compute_scores(filter_object, dataset_object)
    classes = np.array(dataset_object["class"], dtype=float)

    # ── 2. Resolve group labels ─────────────────────────────────────
    groups = _resolve_groups(dataset_object, label_column)
    unique_groups = list(dict.fromkeys(groups))  # order-preserving unique

    colors = colors or _GGPLOT2_COLORS
    fills = fills or _GGPLOT2_FILLS

    # ── 3. Build per-group data ─────────────────────────────────────
    group_data = []
    group_labels = []
    group_classes = []  # track binary class per group for pairwise test
    for g in unique_groups:
        mask = np.array([gl == g for gl in groups])
        vals = scores[mask]
        valid = ~np.isnan(vals) & ~np.isinf(vals)
        if valid.sum() > 0:
            group_data.append(vals[valid])
            group_labels.append(g)
            # majority class for this group
            cls = classes[mask][valid]
            group_classes.append(int(np.median(cls)) if len(cls) > 0 else -1)

    if not group_data:
        raise ValueError("No valid score data found for any group.")

    # ------------------------------------------------------------------
    # 3.2. Re‑order the groups so that Disease is always first,
    # Control is always second (if present).  This guarantees that
    # the color list (red, green, …) is applied consistently.
    # ------------------------------------------------------------------
    # Desired order: Disease → Control → everything else (in original order)
    desired_order = []

    for g in ["Disease", "Control"]:
        if g in unique_groups:
            desired_order.append(g)
    # Append the remaining groups in the order they first appeared
    desired_order.extend([g for g in unique_groups if g not in desired_order])

    # Build a mapping from the desired order to the current indices
    idx_map = [unique_groups.index(g) for g in desired_order]

    # Re‑order the per‑group lists
    group_data = [group_data[i] for i in idx_map]
    group_labels = [group_labels[i] for i in idx_map]
    group_classes = [group_classes[i] for i in idx_map]

    # ── 4. Set up axes ──────────────────────────────────────────────
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _apply_ggplot2_theme(ax)

    # ── 5. Draw violins ────────────────────────────────────────────
    positions = np.arange(1, len(group_data) + 1)
    vp = ax.violinplot(
        group_data,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.72,
    )
    for i, body in enumerate(vp["bodies"]):
        ci = i % len(fills)
        body.set_facecolor(fills[ci])
        body.set_edgecolor(colors[ci])
        body.set_linewidth(1.2)
        body.set_alpha(0.85)

    # ── 6. Boxplot overlay ──────────────────────────────────────────
    if show_box:
        bp = ax.boxplot(
            group_data,
            positions=positions,
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            zorder=3,
        )
        for i, (box, median) in enumerate(zip(bp["boxes"], bp["medians"])):
            ci = i % len(colors)
            box.set_facecolor("white")
            box.set_edgecolor(colors[ci])
            box.set_linewidth(1.0)
            median.set_color(colors[ci])
            median.set_linewidth(2.0)
        for element in ["whiskers", "caps"]:
            for i, line in enumerate(bp[element]):
                ci = (i // 2) % len(colors)
                line.set_color(colors[ci])
                line.set_linewidth(1.0)

    # ── 7. Jittered points ──────────────────────────────────────────
    if show_points:
        rng = np.random.default_rng(42)
        for i, (data, pos) in enumerate(zip(group_data, positions)):
            ci = i % len(colors)
            jitter = rng.uniform(-0.12, 0.12, size=len(data))
            ax.scatter(
                pos + jitter,
                data,
                c=colors[ci],
                s=14,
                alpha=0.55,
                edgecolors="white",
                linewidths=0.3,
                zorder=4,
            )

    # ── 8. P-value bracket (ggpubr style) ───────────────────────────
    if show_pvalue and len(group_data) >= 2:
        _draw_pvalue_brackets(ax, group_data, positions, comparison_method, colors)

    # ── 9. Labels & title ───────────────────────────────────────────
    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, fontsize=11, fontweight="medium")
    ax.set_ylabel("Meta-Score (Z-score)", fontsize=12, fontweight="medium")
    ax.set_xlabel("")

    if title is None:
        title = dataset_object.get("geo_id", "")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # ── 10. Zero reference line ─────────────────────────────────────
    ax.axhline(0, color="#888888", linewidth=0.6, linestyle="--", alpha=0.5)

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ── Multi-dataset violin grid ──────────────────────────────────────
def multi_violin_plot(
    filter_object: Dict,
    meta_object: Dict,
    label_column: str = "label",
    ncols: int = 2,
    figsize_per_panel: Tuple[float, float] = (5, 5),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Violin plot grid for all labelled datasets in a meta-object.

    Parameters
    ----------
    filter_object : dict
        MetaFilter with gene signature.
    meta_object : dict
        Full meta-analysis object with ``originalData``.
    label_column : str
        Phenotype column for group labels.
    ncols : int
        Columns in the subplot grid.
    figsize_per_panel : tuple
        Size of each individual panel.
    title : str, optional
        Super-title for the whole figure.
    save_path : str, optional
        Save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    datasets = []
    ds_names = []
    for name, ds in meta_object["originalData"].items():
        if ds.get("class") is not None:
            datasets.append(ds)
            ds_names.append(name)

    n = len(datasets)
    if n == 0:
        raise ValueError("No labelled datasets found in meta_object.")

    nrows = int(np.ceil(n / ncols))
    fw = figsize_per_panel[0] * ncols
    fh = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (ds, name) in enumerate(zip(datasets, ds_names)):
        violin_plot(
            filter_object,
            ds,
            label_column=label_column,
            ax=axes[i],
            title=name,
            **kwargs,
        )

    # Hide unused panels
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════


def _get_or_compute_scores(filter_object, dataset_object):
    """Return scores from dataset, computing on the fly if needed."""
    if "score" in dataset_object and dataset_object["score"] is not None:
        return np.array(dataset_object["score"], dtype=float)

    # Import here to avoid circular dependency
    from metaintegrator.score_calculation import ScoreCalculator

    return ScoreCalculator.calculate_score(
        filter_object, dataset_object, suppress_messages=True
    )


def _resolve_groups(dataset_object, label_column):
    """Resolve per-sample group labels from pheno or binary class."""
    pheno = dataset_object.get("pheno", None)
    classes = np.array(dataset_object["class"], dtype=float)
    n = len(classes)

    # Try label_column in pheno
    if pheno is not None and isinstance(pheno, pd.DataFrame):
        if label_column in pheno.columns:
            labels = pheno[label_column].values
            unique = pd.Series(labels).dropna().unique()
            if len(unique) > 1:
                return [str(l) for l in labels]

    # Fall back to binary class
    class_map = {0: "Control", 1: "Disease"}
    return [class_map.get(int(c), f"Class_{int(c)}") for c in classes]


def _apply_ggplot2_theme(ax):
    """Apply a clean ggplot2-inspired theme to an axes."""
    ax.set_facecolor("#FAFAFA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(
        axis="both",
        which="both",
        length=3,
        width=0.6,
        color="#CCCCCC",
        labelsize=10,
        labelcolor="#333333",
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.5, zorder=0)


def _draw_pvalue_brackets(ax, group_data, positions, method, colors):
    """Draw ggpubr-style p-value brackets between consecutive groups."""
    pairs = []
    # All pairwise (for 2 groups this is just one pair)
    for i in range(len(group_data)):
        for j in range(i + 1, len(group_data)):
            pairs.append((i, j))

    # Limit to first 3 pairs to avoid clutter
    pairs = pairs[:3]

    y_max = max(d.max() for d in group_data)
    y_range = y_max - min(d.min() for d in group_data)
    bracket_offset = y_range * 0.08

    for k, (i, j) in enumerate(pairs):
        # Statistical test
        if method == "t":
            stat, pval = stats.ttest_ind(group_data[i], group_data[j], equal_var=False)
        else:
            stat, pval = stats.mannwhitneyu(
                group_data[i], group_data[j], alternative="two-sided"
            )

        # Format p-value (ggpubr style)
        if pval < 0.0001:
            p_text = f"p = {pval:.1e}"
        elif pval < 0.001:
            p_text = f"p = {pval:.4f}"
        elif pval < 0.05:
            p_text = f"p = {pval:.3f}"
        else:
            p_text = f"p = {pval:.2f}"

        # Stars
        if pval < 0.001:
            stars = "***"
        elif pval < 0.01:
            stars = "**"
        elif pval < 0.05:
            stars = "*"
        else:
            stars = "ns"

        # Bracket geometry
        y_bar = y_max + bracket_offset * (k + 1) + y_range * 0.04
        x1, x2 = positions[i], positions[j]
        tip = y_range * 0.015

        ax.plot(
            [x1, x1, x2, x2],
            [y_bar - tip, y_bar, y_bar, y_bar - tip],
            color="#333333",
            linewidth=0.8,
        )
        ax.text(
            (x1 + x2) / 2,
            y_bar + y_range * 0.01,
            f"{p_text}  ({stars})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )

    # Extend y-axis to accommodate brackets
    y_top = y_max + bracket_offset * (len(pairs) + 1) + y_range * 0.12
    ax.set_ylim(top=y_top)
