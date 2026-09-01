# Visualization API

This module provides tools for creating publication-quality visualizations of meta-analysis results.

## Modules

| Module | Focus |
|--------|-------|
| `plotting.py` | `MetaAnalysisPlotter`: volcano, forest, heatmap, effect-size distribution, study overview, MA summary |
| `enhanced_plotting.py` | `EnhancedMetaAnalysisPlotter` plus module-level functions: forest/heatmap/Manhattan/predictive-value/single- and multi-dataset ROC/PRC/violin/regression plots |
| `violin_plot.py` | R-parity violin plots with ggplot2/ggpubr styling |

> **Naming history.** `enhanced_plotting`'s plotter class used to also be
> named `MetaAnalysisPlotter`: an unrelated class in a different module
> that happened to share the same name as `plotting.MetaAnalysisPlotter`,
> with a completely disjoint method set. It has been renamed to
> `EnhancedMetaAnalysisPlotter` to remove the collision. A deprecated,
> warning-emitting alias (`visualization.enhanced_plotting.MetaAnalysisPlotter`)
> is kept for old imports, but new code should use `EnhancedMetaAnalysisPlotter`
> directly: see `visualization/__init__.py`'s module docstring for the
> full migration note.
>
> Every method documented below (`plot_volcano`, `plot_forest_plot`,
> `plot_heatmap`, `plot_effect_size_distribution`, `plot_study_overview`,
> `plot_ma_summary`) lives on `plotting.MetaAnalysisPlotter`.
> `EnhancedMetaAnalysisPlotter`'s methods are named `forest_plot`,
> `heatmap_plot`, `manhattan_plot`, `roc_plot`, `prc_plot`,
> `multiple_roc_plot`, `multiple_prc_plot`, `violin_plot`, `regression_plot`,
> etc.: not otherwise documented on this page. Import the one you mean
> explicitly:
> `from visualization.plotting import MetaAnalysisPlotter` or
> `from visualization.enhanced_plotting import EnhancedMetaAnalysisPlotter`.

---

## violin_plot (R-Parity)

Replicates R MetaIntegrator's `violinPlot()` using ggpubr + ggplot2 aesthetics:
individual violins per phenotype group, boxplot overlay, jittered data points,
Wilcoxon p-value brackets.

### `violin_plot()`

```python
from visualization.violin_plot import violin_plot

fig = violin_plot(
    filter_object,
    dataset_object,
    label_column='label',       # pheno column for groups
    comparison_method='wilcox',  # or 't'
    show_points=True,
    show_box=True,
    show_pvalue=True,
    title='GSE11909_GPL96',
    save_path='violin.png',
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_object` | dict |: | MetaFilter with `posGeneNames` / `negGeneNames` |
| `dataset_object` | dict |: | Dataset with `expr`, `keys`, `class`, `pheno` |
| `label_column` | str | `'label'` | Column in `pheno` defining groups. Falls back to binary `class` labels (0→Control, 1→Disease) when the column is missing or has ≤1 unique value. |
| `comparison_method` | str | `'wilcox'` | `'wilcox'` (Mann–Whitney) or `'t'` (Welch) |
| `show_points` | bool | `True` | Jittered data points |
| `show_box` | bool | `True` | Boxplot overlay (IQR + median) |
| `show_pvalue` | bool | `True` | P-value bracket annotations |
| `figsize` | tuple | `(5, 5.5)` | Figure size in inches |
| `title` | str | auto | Plot title |
| `colors` / `fills` | list | ggplot2 palette | Per-group edge/fill colors |
| `ax` | Axes | `None` | Draw on existing Axes |
| `save_path` | str | `None` | Save to file |

### `multi_violin_plot()`

Violin plot grid for all labelled datasets in a meta-object.

```python
from visualization.violin_plot import multi_violin_plot

fig = multi_violin_plot(
    filter_object,
    sle_meta_analysis,
    label_column='label',
    ncols=2,
    title='SLE Meta-Score Distribution',
    save_path='violin_grid.png',
)
```

---

## MetaAnalysisPlotter

Create comprehensive visualizations of meta-analysis results.

```python
class MetaAnalysisPlotter(
    style: str = "default",
    figsize: Tuple[int, int] = (10, 6)
)
```

### Constructor

| Name | Type | Description |
|------|------|-------------|
| style | str | Matplotlib style ('default', 'seaborn', etc.) |
| figsize | Tuple[int, int] | Default figure size in inches |

**Example**

```python
from visualization.plotting import MetaAnalysisPlotter

# Create plotter with default settings
plotter = MetaAnalysisPlotter()

# Use custom style
plotter = MetaAnalysisPlotter(style="seaborn", figsize=(12, 8))
```

---

### `plot_volcano`

```python
def plot_volcano(
    self,
    meta_object: Dict,
    es_threshold: float = 0.5,
    fdr_threshold: float = 0.05,
    top_genes: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure
```

Create a volcano plot showing effect size vs. significance.

| Name | Type | Description |
|------|------|-------------|
| meta_object | Dict | Meta-analysis results |
| es_threshold | float | Effect size threshold for highlighting (default: 0.5) |
| fdr_threshold | float | FDR threshold (default: 0.05) |
| top_genes | int | Number of top genes to label (default: 20, 0=none) |
| save_path | str, optional | Path to save figure |

**Returns**

| Type | Description |
|------|-------------|
| plt.Figure | Matplotlib figure object |

**Example**

```python
# Basic volcano plot
fig = plotter.plot_volcano(results)
plt.show()

# Custom thresholds with labels
fig = plotter.plot_volcano(
    results,
    es_threshold=0.8,
    fdr_threshold=0.01,
    top_genes=30,
    save_path="volcano.png"
)
```

**Visual Elements**
- Gray points: Non-significant genes
- Red points: Significantly up-regulated
- Blue points: Significantly down-regulated
- Dashed lines: Effect size and FDR thresholds
- Labels: Top genes by significance

---

### `plot_forest_plot`

```python
def plot_forest_plot(
    self,
    meta_object: Dict,
    genes: List[str],
    max_genes: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure
```

Create a forest plot showing effect sizes across studies.

| Name | Type | Description |
|------|------|-------------|
| meta_object | Dict | Meta-analysis results |
| genes | List[str] | Gene names to plot |
| max_genes | int | Maximum number of genes (default: 20) |
| save_path | str, optional | Path to save figure |

**Returns**

| Type | Description |
|------|-------------|
| plt.Figure | Matplotlib figure object |

**Example**

```python
# Get top genes
pooled = results["metaAnalysis"]["pooledResults"]
top_genes = pooled.nsmallest(10, "effectSizeFDR").index.tolist()

# Create forest plot
fig = plotter.plot_forest_plot(
    results,
    genes=top_genes,
    save_path="forest.png"
)
plt.show()
```

**Visual Elements**
- Colored points: Study-specific effect sizes
- Error bars: 95% confidence intervals
- Black diamonds: Pooled effect estimates
- Vertical line: No effect (effect size = 0)

---

### `plot_heatmap`

```python
def plot_heatmap(
    self,
    meta_object: Dict,
    genes: Optional[List[str]] = None,
    n_top_genes: int = 50,
    cluster: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure
```

Create a heatmap of effect sizes across studies.

| Name | Type | Description |
|------|------|-------------|
| meta_object | Dict | Meta-analysis results |
| genes | List[str], optional | Specific genes to plot (None = auto-select) |
| n_top_genes | int | Number of top genes if genes=None (default: 50) |
| cluster | bool | Accepted but currently has no effect -- reserved for future hierarchical clustering of rows (default: True) |
| save_path | str, optional | Path to save figure |

**Returns**

| Type | Description |
|------|-------------|
| plt.Figure | Matplotlib figure object |

**Example**

```python
# Auto-select top 50 genes
fig = plotter.plot_heatmap(results)

# Custom gene list
my_genes = ["GAPDH", "TP53", "ACTB", "MYC"]
fig = plotter.plot_heatmap(
    results,
    genes=my_genes,
    save_path="heatmap.png"
)
```

**Color Scheme**
- Red: Positive effect sizes (up-regulated)
- Blue: Negative effect sizes (down-regulated)
- White: No effect (effect size ≈ 0)

---

### `plot_effect_size_distribution`

```python
def plot_effect_size_distribution(
    self,
    meta_object: Dict,
    bins: int = 50,
    overlay_density: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure
```

Plot distribution of effect sizes by dataset.

| Name | Type | Description |
|------|------|-------------|
| meta_object | Dict | Meta-analysis results |
| bins | int | Number of histogram bins (default: 50) |
| overlay_density | bool | Show density curves vs. histograms (default: True) |
| save_path | str, optional | Path to save figure |

**Returns**

| Type | Description |
|------|-------------|
| plt.Figure | Matplotlib figure object |

**Example**

```python
# Density curves
fig = plotter.plot_effect_size_distribution(results)

# Histograms instead
fig = plotter.plot_effect_size_distribution(
    results,
    overlay_density=False,
    bins=30
)
```

---

### `plot_study_overview`

```python
def plot_study_overview(
    self,
    meta_object: Dict,
    save_path: Optional[str] = None
) -> plt.Figure
```

Create comprehensive overview of study characteristics.

| Name | Type | Description |
|------|------|-------------|
| meta_object | Dict | Meta-analysis object |
| save_path | str, optional | Path to save figure |

**Returns**

| Type | Description |
|------|-------------|
| plt.Figure | 2×2 subplot figure |

**Subplots**
1. Sample distribution (stacked cases/controls)
2. Total sample size by study
3. Gene count by study
4. Case/control balance ratio

**Example**

```python
fig = plotter.plot_study_overview(
    results,
    save_path="overview.png"
)
plt.show()
```

---

### `plot_ma_summary`

```python
def plot_ma_summary(
    self,
    meta_object: Dict,
    save_path: Optional[str] = None
) -> plt.Figure
```

Create summary panel with key meta-analysis diagnostics.

| Name | Type | Description |
|------|------|-------------|
| meta_object | Dict | Meta-analysis results |
| save_path | str, optional | Path to save figure |

**Returns**

| Type | Description |
|------|-------------|
| plt.Figure | 2×2 subplot figure |

**Subplots**
1. Effect size distribution
2. Volcano-lite (effect size vs. p-value)
3. Studies per gene distribution
4. Heterogeneity p-value distribution

**Example**

```python
fig = plotter.plot_ma_summary(results)
plt.show()
```

---

## Related

- [Analysis](analysis.md) - Generate results to visualize
- [User Guide](../user_guide/best_practices.md) - Interpretation guide
- [Tutorial](../tutorials/visualization.md) - Detailed examples

---

## Complete Visualization Workflow

```python
from visualization.plotting import MetaAnalysisPlotter
import matplotlib.pyplot as plt

# Initialize plotter
plotter = MetaAnalysisPlotter(figsize=(12, 8))

# 1. Overview of studies
fig = plotter.plot_study_overview(results)
plt.tight_layout()
plt.savefig("01_study_overview.png", dpi=300)
plt.show()

# 2. Meta-analysis summary
fig = plotter.plot_ma_summary(results)
plt.savefig("02_ma_summary.png", dpi=300)
plt.show()

# 3. Volcano plot
fig = plotter.plot_volcano(
    results,
    es_threshold=0.5,
    fdr_threshold=0.05,
    top_genes=20
)
plt.savefig("03_volcano.png", dpi=300)
plt.show()

# 4. Get significant genes for further plots
pooled = results["metaAnalysis"]["pooledResults"]
sig_genes = pooled[
    (pooled["effectSizeFDR"] < 0.05) &
    (pooled["effectSize"].abs() > 0.5)
].index.tolist()

# 5. Forest plot for top 10 genes
top_genes = sig_genes[:10]
fig = plotter.plot_forest_plot(results, genes=top_genes)
plt.savefig("04_forest.png", dpi=300)
plt.show()

# 6. Heatmap of significant genes
fig = plotter.plot_heatmap(
    results,
    genes=sig_genes,
)
plt.savefig("05_heatmap.png", dpi=300)
plt.show()

# 7. Effect size distributions by study
fig = plotter.plot_effect_size_distribution(results)
plt.savefig("06_distributions.png", dpi=300)
plt.show()
```

---

## Customizing Plots

### Adjusting Plot Style

```python
import matplotlib.pyplot as plt

# Set global style
plt.style.use('seaborn-v0_8-darkgrid')

# Or use plotter style
plotter = MetaAnalysisPlotter(style='bmh')

# Custom figure size per plot
fig = plotter.plot_volcano(results)
fig.set_size_inches(15, 10)
```

### Custom Color Schemes

```python
import seaborn as sns

# Set custom color palette
sns.set_palette("husl")

# Create plots
fig = plotter.plot_heatmap(results)

# Reset to default
sns.reset_defaults()
```

### Adding Annotations

```python
# Create base plot
fig = plotter.plot_volcano(results, top_genes=0)  # No auto-labels
ax = fig.axes[0]

# Add custom annotations
pooled = results["metaAnalysis"]["pooledResults"]
my_genes = ["GAPDH", "TP53", "MYC"]

for gene in my_genes:
    if gene in pooled.index:
        es = pooled.loc[gene, "effectSize"]
        fdr = pooled.loc[gene, "effectSizeFDR"]
        y = -np.log10(fdr)
        
        ax.annotate(
            gene,
            xy=(es, y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->")
        )

plt.show()
```

---

## Publication-Quality Figures

```python
import matplotlib.pyplot as plt
from visualization.plotting import MetaAnalysisPlotter

# Configure for publication
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42  # TrueType fonts
})

plotter = MetaAnalysisPlotter(figsize=(10, 7))

# Create volcano plot
fig = plotter.plot_volcano(
    results,
    es_threshold=0.8,
    fdr_threshold=0.01,
    top_genes=15
)

# Adjust layout
plt.tight_layout()

# Save in multiple formats
plt.savefig("volcano.pdf")    # Vector
plt.savefig("volcano.png")    # Raster
plt.savefig("volcano.svg")    # Web/vector
```

---

## Multi-Panel Figures

```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create custom layout
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig)

# Panel A: Study overview
ax1 = fig.add_subplot(gs[0, :])
study_fig = plotter.plot_study_overview(results)
# Transfer to main figure...

# Panel B: Volcano
ax2 = fig.add_subplot(gs[1, 0])
# Create volcano manually or transfer

# Panel C: Forest plot
ax3 = fig.add_subplot(gs[1, 1])
# Create forest manually

# Panel D: Heatmap
ax4 = fig.add_subplot(gs[2, :])
# Create heatmap

# Add panel labels
for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    ax.text(
        -0.1, 1.05,
        chr(65 + i),  # A, B, C, D
        transform=ax.transAxes,
        fontsize=18,
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig("figure_1.pdf")
```

---

## Interactive Visualizations

```python
import plotly.express as px
import plotly.graph_objects as go

# Get results
pooled = results["metaAnalysis"]["pooledResults"]

# Create interactive volcano plot
pooled["neg_log_fdr"] = -np.log10(pooled["effectSizeFDR"])

fig = px.scatter(
    pooled.reset_index(),
    x="effectSize",
    y="neg_log_fdr",
    hover_name="index",  # Gene names
    hover_data=["effectSizePval", "numStudies"],
    title="Interactive Volcano Plot"
)

# Add threshold lines
fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red")
fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
fig.add_vline(x=-0.5, line_dash="dash", line_color="gray")

# Save as HTML
fig.write_html("volcano_interactive.html")

# Display in notebook
fig.show()
```

---

## Troubleshooting

**Problem: Overlapping labels in volcano plot**

```python
# Reduce number of labels
fig = plotter.plot_volcano(results, top_genes=10)

# Or manually adjust
from adjustText import adjust_text  # pip install adjustText
# Use adjust_text to prevent overlaps
```

**Problem: Heatmap too large**

```python
# Limit to top genes
fig = plotter.plot_heatmap(results, n_top_genes=30)

# Or create multiple panels
top_50 = pooled.nsmallest(50, "effectSizeFDR").index
chunks = [top_50[i:i+25] for i in range(0, 50, 25)]

for i, chunk in enumerate(chunks):
    fig = plotter.plot_heatmap(results, genes=chunk.tolist())
    plt.savefig(f"heatmap_panel_{i+1}.png")
```

**Problem: Forest plot too crowded**

```python
# Show fewer genes
top_5 = pooled.nsmallest(5, "effectSizeFDR").index
fig = plotter.plot_forest_plot(results, genes=top_5.tolist())

# Increase figure height
plotter.figsize = (12, 10)
fig = plotter.plot_forest_plot(results, genes=top_genes)
```
