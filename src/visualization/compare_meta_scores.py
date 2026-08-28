"""
Compare meta-scores produced by the Python and R MetaIntegrator
pipelines.

The module ingests the per-sample CSV files (one from each pipeline),
joins them on the shared ``Sample`` column, and generates a
multi-panel diagnostic figure that makes parity easy to assess at a
glance.

Panels
------
1. **Scatter** – Python MetaScore vs R MetaScore with the identity
   line.  If the two pipelines agree perfectly every point sits on
   the diagonal.
2. **Bland–Altman** – mean of the two scores (x) vs their difference
   (y).  Systematic bias shows up as a non-zero mean line; heteroscedasticity
   shows up as a funnel shape.
3. **Residual histogram** – distribution of (Python − R) differences.
   A tight, symmetric bell centred on zero is ideal.
4. **Summary statistics** – Pearson *r*, *R²*, RMSE, MAE,
   max |Δ|, and the number of matched samples.

Usage
-----
::

    from visualization.compare_meta_scores import compare_meta_scores

    fig = compare_meta_scores(
        python_csv="output_sle_tutorial_analysis/plot_data/violin_GSE50635_GPL6244.csv",
        r_csv="output_sle_tutorial_r_analysis/plot_data/violin_GSE50635_subject_type.csv",
        title="GSE50635 – Python vs R Meta-Score Comparison",
        save_path="output_sle_tutorial_analysis/plot_data/meta_score_comparison_GSE50635.png",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── visual constants ────────────────────────────────────────────────
_FIG_BG = "#FAFAFA"
_SCATTER_C = "#4C72B0"
_HIST_C = "#55A868"
_LINE_C = "#C44E52"
_TEXT_BG = "#F0F0F0"


def _load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read a violin-data CSV and normalize column names."""
    df = pd.read_csv(path, index_col=0)
    # make column names lower-case for safety
    df.columns = [c.strip() for c in df.columns]
    return df


def _merge(py_df: pd.DataFrame, r_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the two frames on *Sample* and return tidy result."""
    merged = pd.merge(
        py_df[["Sample", "MetaScore"]].rename(columns={"MetaScore": "Python"}),
        r_df[["Sample", "MetaScore"]].rename(columns={"MetaScore": "R"}),
        on="Sample",
        how="inner",
    )
    merged["Diff"] = merged["Python"] - merged["R"]
    merged["Mean"] = (merged["Python"] + merged["R"]) / 2.0
    return merged


def _apply_theme(ax: plt.Axes) -> None:
    """Apply the module's minimal light theme (clean spines, light grid) to an axes."""
    ax.set_facecolor(_FIG_BG)
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
        labelsize=9,
        labelcolor="#333333",
    )
    ax.grid(axis="both", color="#EEEEEE", linewidth=0.5, zorder=0)


# ── public API ──────────────────────────────────────────────────────


def compare_meta_scores(
    python_csv: Union[str, Path],
    r_csv: Union[str, Path],
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create a four-panel parity diagnostic (see module docstring for the panels).

    Args:
        python_csv: CSV emitted by the Python pipeline (``extract_violin_data``).
        r_csv: Matching CSV from the R pipeline.
        title: Super-title for the whole figure.
        figsize: Figure size in inches.
        save_path: If given, save the figure to this location.

    Returns:
        The generated matplotlib Figure.
    """
    py_df = _load_csv(python_csv)
    r_df = _load_csv(r_csv)
    merged = _merge(py_df, r_df)

    n = len(merged)
    if n == 0:
        raise ValueError("No matching samples found between the two files.")

    py_vals = merged["Python"].values
    r_vals = merged["R"].values
    diff = merged["Diff"].values
    mean = merged["Mean"].values

    # ── statistics ──────────────────────────────────────────────────
    pearson_r, p_val = sp_stats.pearsonr(py_vals, r_vals)
    r_sq = pearson_r**2
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    # ── figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax in axes.flat:
        _apply_theme(ax)

    # ── 1. Scatter ──────────────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.scatter(
        r_vals,
        py_vals,
        c=_SCATTER_C,
        s=40,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    lo = min(r_vals.min(), py_vals.min()) - 0.2
    hi = max(r_vals.max(), py_vals.max()) + 0.2
    ax1.plot(
        [lo, hi],
        [lo, hi],
        color=_LINE_C,
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
        label="Identity",
        zorder=2,
    )
    ax1.set_xlabel("R MetaScore", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Python MetaScore", fontsize=11, fontweight="medium")
    ax1.set_title("Scatter – Python vs R", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(fontsize=9)
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect("equal", adjustable="box")

    # annotate r
    ax1.text(
        0.05,
        0.92,
        f"Pearson r = {pearson_r:.6f}\nR² = {r_sq:.6f}",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", fc=_TEXT_BG, ec="#CCCCCC", lw=0.5),
    )

    # ── 2. Bland–Altman ─────────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.scatter(
        mean,
        diff,
        c=_SCATTER_C,
        s=40,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    ax2.axhline(
        mean_diff,
        color=_LINE_C,
        linewidth=1.0,
        linestyle="-",
        label=f"Mean diff = {mean_diff:.2e}",
        zorder=2,
    )
    if n > 1:
        ax2.axhline(
            mean_diff + 1.96 * std_diff,
            color="#888888",
            linewidth=0.8,
            linestyle=":",
            zorder=2,
            label=f"+1.96 SD = {mean_diff + 1.96*std_diff:.2e}",
        )
        ax2.axhline(
            mean_diff - 1.96 * std_diff,
            color="#888888",
            linewidth=0.8,
            linestyle=":",
            zorder=2,
            label=f"−1.96 SD = {mean_diff - 1.96*std_diff:.2e}",
        )
    ax2.set_xlabel("Mean of Python & R", fontsize=11, fontweight="medium")
    ax2.set_ylabel("Difference (Python − R)", fontsize=11, fontweight="medium")
    ax2.set_title("Bland–Altman Plot", fontsize=12, fontweight="bold", pad=10)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))

    # ── 3. Residual histogram ───────────────────────────────────────
    ax3 = axes[1, 0]
    bins = max(10, n // 3)
    ax3.hist(
        diff,
        bins=bins,
        color=_HIST_C,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
        zorder=3,
    )
    ax3.axvline(0, color=_LINE_C, linewidth=1.0, linestyle="--", alpha=0.8, zorder=2)
    ax3.set_xlabel("Difference (Python − R)", fontsize=11, fontweight="medium")
    ax3.set_ylabel("Count", fontsize=11, fontweight="medium")
    ax3.set_title("Distribution of Differences", fontsize=12, fontweight="bold", pad=10)

    # ── 4. Summary stats ────────────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.axis("off")

    stats_text = (
        f"{'Metric':<22}{'Value':>16}\n"
        f"{'─' * 38}\n"
        f"{'Matched samples':<22}{n:>16d}\n"
        f"{'Pearson r':<22}{pearson_r:>16.10f}\n"
        f"{'R²':<22}{r_sq:>16.10f}\n"
        f"{'RMSE':<22}{rmse:>16.2e}\n"
        f"{'MAE':<22}{mae:>16.2e}\n"
        f"{'Max |Δ|':<22}{max_abs:>16.2e}\n"
        f"{'Mean diff':<22}{mean_diff:>16.2e}\n"
        f"{'SD of diff':<22}{std_diff:>16.2e}\n"
        f"{'p-value (Pearson)':<22}{p_val:>16.2e}\n"
    )

    ax4.text(
        0.08,
        0.92,
        stats_text,
        transform=ax4.transAxes,
        fontsize=11,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc=_TEXT_BG, ec="#CCCCCC", lw=0.8),
    )
    ax4.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=10)

    # ── finish ──────────────────────────────────────────────────────
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
