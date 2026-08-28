# Visualization and Plotting

Learn how to create publication-quality visualizations of meta-analysis results including volcano plots, forest plots, heatmaps, and effect size distributions. Effective visualization is crucial for interpreting results and communicating findings.

## Goal

Generate publication-ready figures that clearly communicate meta-analysis results, including statistical significance, effect sizes, and cross-study consistency.

## Prerequisites

- Python 3.10+
- metaIntegrator-py git repository cloned
- Completed meta-analysis (see [Meta-Analysis Tutorial](meta_analysis.md))
- Matplotlib and seaborn installed (included with metaIntegrator-py)

### Why Visualization Matters

Good visualizations:
- ✅ Reveal patterns humans might miss in tables
- ✅ Communicate results to non-statistical audiences
- ✅ Identify outliers and quality issues
- ✅ Meet journal publication standards

## Steps

### Step 1: Import Visualization Tools

```python
from visualization.plotting import MetaAnalysisPlotter
from visualization.enhanced_plotting import (
    forest_plot,
    heatmap_plot,
)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Configure matplotlib for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

print("✅ Visualization tools imported")
```

### Step 2: Load Meta-Analysis Results

```python
# Load results from previous tutorial
import pickle

# Option 1: Load saved meta-object
with open('meta_analysis_results/meta_object_with_results.pkl', 'rb') as f:
    meta_object = pickle.load(f)

# Option 2: Or read CSV results
# pooled = pd.read_csv('meta_analysis_results/all_genes_meta_analysis.csv', index_col=0)

# Extract results
meta_results = meta_object['metaAnalysis']
pooled = meta_results['pooledResults']

print(f"📊 Loaded results:")
print(f"   Total genes: {len(pooled):,}")
print(f"   Significant (FDR < 0.05): {(pooled['effectSizeFDR'] < 0.05).sum():,}")
```

**Expected output:**
```
📊 Loaded results:
   Total genes: 52,341
   Significant (FDR < 0.05): 1,245
```

### Step 3: Create a Volcano Plot

Volcano plots show effect size vs. statistical significance:

```python
print("\n📊 Creating volcano plot...")

# Create output directory
output_dir = Path("./figures")
output_dir.mkdir(exist_ok=True)

# Extract data
effect_col = 'effectSize'
effect_sizes = pooled[effect_col]
p_values = pooled['effectSizePval']

# Calculate -log10(p-value)
neg_log_p = -np.log10(p_values.replace(0, 1e-300))

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot - colored by significance
# Significant genes (FDR < 0.05) in red
is_significant = pooled['effectSizeFDR'] < 0.05

ax.scatter(
    effect_sizes[~is_significant], 
    neg_log_p[~is_significant],
    c='lightgray', 
    alpha=0.5, 
    s=20, 
    label='Not significant'
)

ax.scatter(
    effect_sizes[is_significant], 
    neg_log_p[is_significant],
    c='red', 
    alpha=0.7, 
    s=30, 
    label='FDR < 0.05'
)

# Add threshold lines
ax.axhline(y=-np.log10(0.05), color='blue', linestyle='--', 
           linewidth=1.5, alpha=0.7, label='p = 0.05')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

# Annotate top genes
top_genes = pooled.nsmallest(10, 'effectSizeFDR')
for gene_name, row in top_genes.iterrows():
    es = row[effect_col]
    pval = row['effectSizePval']
    
    # Only annotate if not too crowded
    if abs(es) > 0.5 or -np.log10(pval) > 10:
        ax.annotate(
            gene_name,
            xy=(es, -np.log10(pval)),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )

# Labels and styling
ax.set_xlabel('Effect Size (Hedges\' g)', fontsize=12, fontweight='bold')
ax.set_ylabel('-log₁₀(p-value)', fontsize=12, fontweight='bold')
ax.set_title('Volcano Plot: Differentially Expressed Genes', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(output_dir / 'volcano_plot.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_dir}/volcano_plot.png")

plt.show()
```

**What to look for in a volcano plot:**
- **Top corners**: Most significant genes with large effects
- **Symmetric distribution**: Balanced up/down regulation
- **Threshold lines**: Genes above/right of thresholds are significant

### Step 4: Create Effect Size Distribution Plot

```python
print("\n📊 Creating effect size distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Histogram
ax1 = axes[0]
ax1.hist(effect_sizes, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero effect')
ax1.axvline(x=0.5, color='orange', linestyle=':', linewidth=1.5, label='Medium effect')
ax1.axvline(x=-0.5, color='orange', linestyle=':', linewidth=1.5)

ax1.set_xlabel('Effect Size (Hedges\' g)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Effect Size Distribution', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right panel: Box plot by significance
ax2 = axes[1]
sig_effects = effect_sizes[is_significant]
nonsig_effects = effect_sizes[~is_significant]

bp = ax2.boxplot(
    [sig_effects, nonsig_effects],
    labels=['Significant\n(FDR < 0.05)', 'Not Significant'],
    patch_artist=True,
    widths=0.6
)

# Color boxes
bp['boxes'][0].set_facecolor('red')
bp['boxes'][0].set_alpha(0.5)
bp['boxes'][1].set_facecolor('lightgray')
bp['boxes'][1].set_alpha(0.5)

ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax2.set_ylabel('Effect Size (Hedges\' g)', fontsize=12, fontweight='bold')
ax2.set_title('Effect Size by Significance', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(output_dir / 'effect_size_distribution.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_dir}/effect_size_distribution.png")

plt.show()
```

### Step 5: Create Top Genes Heatmap

Show expression patterns of top genes across studies:

```python
print("\n📊 Creating top genes heatmap...")

# Get top 30 genes by FDR
n_top = 30
top_genes = pooled.nsmallest(n_top, 'effectSizeFDR')

# Collect expression data for these genes
heatmap_data_list = []
dataset_labels = []

for dataset_name, dataset in meta_object['originalData'].items():
    expr = dataset['expr']
    keys = dataset['keys']
    class_vector = dataset['class']
    
    # Find matching genes
    gene_mask = [k in top_genes.index for k in keys]
    
    if any(gene_mask):
        # Extract expression for top genes
        top_expr = expr.iloc[gene_mask]
        
        # Calculate mean expression per class
        controls = top_expr.iloc[:, class_vector == 0].mean(axis=1)
        cases = top_expr.iloc[:, class_vector == 1].mean(axis=1)
        
        # Calculate log fold change
        log_fc = cases - controls
        
        heatmap_data_list.append(log_fc)
        dataset_labels.append(dataset_name)

# Create heatmap DataFrame
if heatmap_data_list:
    heatmap_df = pd.concat(heatmap_data_list, axis=1)
    heatmap_df.columns = dataset_labels
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Plot heatmap
    sns.heatmap(
        heatmap_df,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Log₂ Fold Change\n(Case vs. Control)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray',
        vmin=-2,
        vmax=2
    )
    
    ax.set_title(f'Top {n_top} Genes: Expression Changes Across Studies', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Study', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gene', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'top_genes_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_dir}/top_genes_heatmap.png")
    
    plt.show()
else:
    print("   ⚠️ Could not create heatmap - no matching genes found")
```

**Heatmap interpretation:**
- **Red**: Up-regulated in cases
- **Blue**: Down-regulated in cases
- **Consistent colors across columns**: Gene shows same pattern in all studies

### Step 6: Create P-value Distribution Plot

QC check for proper p-value distribution:

```python
print("\n📊 Creating p-value distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Raw p-value histogram
ax1 = axes[0]
ax1.hist(p_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.axhline(
    y=len(p_values)/50, 
    color='red', 
    linestyle='--', 
    label='Uniform distribution'
)

ax1.set_xlabel('P-value', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Raw P-value Distribution', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right panel: QQ plot
ax2 = axes[1]
from scipy import stats

# Theoretical quantiles (uniform distribution)
n = len(p_values)
theoretical = np.linspace(0, 1, n)
observed = np.sort(p_values)

ax2.scatter(theoretical, observed, alpha=0.5, s=20, color='steelblue')
ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Expected (uniform)')

ax2.set_xlabel('Expected P-value', fontsize=12, fontweight='bold')
ax2.set_ylabel('Observed P-value', fontsize=12, fontweight='bold')
ax2.set_title('P-value Q-Q Plot', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(output_dir / 'pvalue_distribution.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_dir}/pvalue_distribution.png")

plt.show()
```

**What to look for:**
- **Uniform distribution**: Most p-values should be uniformly distributed
- **Excess at p < 0.05**: Indicates true signal (good!)
- **Deviation from diagonal in QQ plot**: Shows significant findings

### Step 7: Create Multi-Panel Summary Figure

Combine multiple plots into one publication figure:

```python
print("\n📊 Creating comprehensive summary figure...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Panel A: Volcano plot
ax1 = fig.add_subplot(gs[0, :])
ax1.scatter(effect_sizes[~is_significant], neg_log_p[~is_significant],
            c='lightgray', alpha=0.5, s=20)
ax1.scatter(effect_sizes[is_significant], neg_log_p[is_significant],
            c='red', alpha=0.7, s=30)
ax1.axhline(y=-np.log10(0.05), color='blue', linestyle='--', alpha=0.7)
ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax1.set_xlabel('Effect Size (Hedges\' g)', fontweight='bold')
ax1.set_ylabel('-log₁₀(p-value)', fontweight='bold')
ax1.set_title('A. Volcano Plot', fontweight='bold', loc='left')
ax1.grid(True, alpha=0.3)

# Panel B: Effect size distribution
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(effect_sizes, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Effect Size', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('B. Effect Size Distribution', fontweight='bold', loc='left')
ax2.grid(True, alpha=0.3, axis='y')

# Panel C: Significance counts
ax3 = fig.add_subplot(gs[1, 1])
sig_counts = {
    'FDR < 0.01': (pooled['effectSizeFDR'] < 0.01).sum(),
    'FDR < 0.05': (pooled['effectSizeFDR'] < 0.05).sum(),
    'p < 0.05': (pooled['effectSizePval'] < 0.05).sum(),
    'Total': len(pooled)
}
colors = ['darkred', 'red', 'orange', 'lightgray']
ax3.barh(list(sig_counts.keys()), list(sig_counts.values()), color=colors, alpha=0.7)
ax3.set_xlabel('Number of Genes', fontweight='bold')
ax3.set_title('C. Significance Summary', fontweight='bold', loc='left')
ax3.grid(True, alpha=0.3, axis='x')

# Panel D: Up vs Down regulation
ax4 = fig.add_subplot(gs[2, :])
sig_genes = pooled[pooled['effectSizeFDR'] < 0.05]
up_genes = sig_genes[sig_genes[effect_col] > 0].nlargest(15, effect_col)
down_genes = sig_genes[sig_genes[effect_col] < 0].nsmallest(15, effect_col)

y_pos = np.arange(len(up_genes))
ax4.barh(y_pos, up_genes[effect_col], color='red', alpha=0.7, label='Up-regulated')
ax4.barh(y_pos + len(up_genes) + 1, down_genes[effect_col], 
         color='blue', alpha=0.7, label='Down-regulated')

ax4.set_yticks(list(y_pos) + [len(up_genes)] + list(y_pos + len(up_genes) + 1))
ax4.set_yticklabels(
    list(up_genes.index) + [''] + list(down_genes.index),
    fontsize=9
)
ax4.set_xlabel('Effect Size (Hedges\' g)', fontweight='bold')
ax4.set_title('D. Top Differentially Expressed Genes', fontweight='bold', loc='left')
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig.savefig(output_dir / 'summary_figure.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_dir}/summary_figure.png")

plt.show()
```

### Step 8: Export High-Resolution Figures for Publication

```python
print("\n💾 Exporting publication-ready figures...")

# Export key plots in multiple formats
plots_to_export = [
    'volcano_plot',
    'effect_size_distribution',
    'top_genes_heatmap',
    'summary_figure'
]

for plot_name in plots_to_export:
    base_path = output_dir / plot_name
    
    # Check if plot exists
    if (output_dir / f'{plot_name}.png').exists():
        # Already saved as PNG (300 dpi)
        
        # Also save as PDF (vector format for journals)
        # Note: Would need to re-create plots with PDF backend
        print(f"   ✅ {plot_name}.png (300 dpi)")
    else:
        print(f"   ⚠️ {plot_name} not found")

print(f"\n✅ All figures saved to: {output_dir}/")
print("\nFigure formats:")
print("   • PNG (300 dpi): For presentations, manuscripts")
print("   • High resolution: Suitable for publication")
```

## Tips & Gotchas

### Tip 1: Use Consistent Color Schemes

```python
# Define publication color palette
COLORS = {
    'significant': '#E63946',      # Red
    'not_significant': '#A8DADC',  # Light blue
    'neutral': '#457B9D',          # Dark blue
    'positive': '#E63946',         # Red
    'negative': '#1D3557'          # Navy
}

# Use in all plots
ax.scatter(x, y, c=COLORS['significant'])
```

### Tip 2: Always Include Sample Sizes in Labels

```python
# Good: Include sample size
ax.set_xlabel(f'Effect Size (n = {len(effect_sizes):,} genes)')

# Better: Include multiple stats
n_total = len(pooled)
n_sig = (pooled['effectSizeFDR'] < 0.05).sum()
ax.set_title(f'Meta-Analysis Results (n = {n_total:,} genes, {n_sig} significant)')
```

### Tip 3: Save Plot Data for Reproducibility

```python
# Save data used to create each plot
plot_data = {
    'effect_sizes': effect_sizes.to_dict(),
    'p_values': p_values.to_dict(),
    'significance': is_significant.to_dict()
}

import json
with open(output_dir / 'plot_data.json', 'w') as f:
    json.dump(plot_data, f, indent=2)
```

### Gotcha 1: Matplotlib Style Changes

Different matplotlib versions have different styles:

```python
# Set explicit style for consistency
plt.style.use('seaborn-v0_8-darkgrid')

# Or use custom RC params
custom_params = {
    'figure.figsize': (10, 8),
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
}
plt.rcParams.update(custom_params)
```

### Gotcha 2: Log10(0) is Undefined

```python
# ❌ WRONG - will produce -inf or error
neg_log_p = -np.log10(p_values)

# ✅ CORRECT - replace 0 with very small number
neg_log_p = -np.log10(p_values.replace(0, 1e-300))
```

### Gotcha 3: Overlapping Labels in Volcano Plot

```python
# Use adjustText library to prevent overlap
try:
    from adjustText import adjust_text
    
    texts = []
    for gene_name, row in top_genes.iterrows():
        text = ax.text(row[effect_col], -np.log10(row['effectSizePval']), gene_name)
        texts.append(text)
    
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))
except ImportError:
    print("Install adjustText for better label placement:")
    print("pip install adjustText")
```

## Troubleshooting

### Q: Figures look pixelated

**A**: Increase DPI:

```python
# For screen display
plt.rcParams['figure.dpi'] = 150

# For saving
fig.savefig('plot.png', dpi=600)  # Very high quality
```

### Q: Text too small in saved figures

**A**: Scale font sizes:

```python
# Increase all text
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
```

### Q: Colors don't show in printed manuscript

**A**: Use printer-friendly colors:

```python
# Color-blind friendly palette
CB_COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'purple': '#949494'
}
```

### Q: PDF export fails

**A**: Use PDF backend:

```python
import matplotlib
matplotlib.use('PDF')

# Then create and save plots
fig.savefig('plot.pdf', format='pdf')
```

## Next Steps

After creating visualizations:

1. **Pathway Enrichment** – Analyze biological functions
2. **Signature Validation** – Test in independent cohorts
3. **Manuscript Writing** – Use figures in publications

### Advanced Visualization

- **Interactive plots** with Plotly
- **Forest plots** showing per-study effects
- **Network diagrams** for gene interactions
- **Pathway heatmaps** with gene annotations

### Key Takeaways

✅ Create volcano plots for significance vs. effect size  
✅ Use heatmaps to show cross-study consistency  
✅ Include sample sizes in all plot titles/labels  
✅ Save figures at 300+ DPI for publication  
✅ Use color-blind friendly palettes  
✅ Export in multiple formats (PNG, PDF)  
✅ Save plot data for reproducibility  

---

### Step 9: Violin Plots (R-Parity)

Violin plots are the standard way to validate a gene signature in an independent
dataset, showing the meta-score distribution for disease vs. control groups.
The Python implementation matches R MetaIntegrator's `violinPlot()` style.

```python
from visualization.violin_plot import violin_plot, multi_violin_plot

# Single dataset — matches R's violinPlot(filterObject, datasetObject, labelColumn)
fig = violin_plot(
    filter_object,
    sle_meta_analysis['originalData']['GSE11909_GPL96'],
    label_column='label',
    title='GSE11909_GPL96',
    save_path='figures/violin_GSE11909_GPL96.png',
)
plt.show()

# All datasets in a 2×2 grid
fig = multi_violin_plot(
    filter_object,
    sle_meta_analysis,
    label_column='label',
    title='SLE Meta-Score Distribution',
    save_path='figures/violin_all_datasets.png',
)
plt.show()
```

**Features matching R:**

- Individual violins per group (not split)
- Boxplot overlay with IQR and median
- Jittered individual data points
- Wilcoxon rank-sum test p-value brackets
- ggplot2-inspired color palette and theme

**Customization:**

```python
# Custom colors
fig = violin_plot(
    filter_object, dataset,
    colors=['#E41A1C', '#377EB8'],  # red for disease, blue for control
    fills=['#E41A1C80', '#377EB880'],
    show_pvalue=True,
    comparison_method='t',  # Welch's t-test instead of Wilcoxon
)
```

---

**Questions?** See [Best Practices](../user_guide/best_practices.md).
