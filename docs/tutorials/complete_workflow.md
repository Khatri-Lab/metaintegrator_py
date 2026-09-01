# Complete Meta-Analysis Workflow

This tutorial walks you through a complete meta-analysis of gene expression data, from downloading datasets to generating publication-ready figures. You'll learn the entire metaIntegrator-py workflow using real data from NCBI GEO.

## Goal

Perform an end-to-end meta-analysis to identify genes differentially expressed across multiple independent studies, with statistical validation and visualization.

## What You'll Accomplish

By the end of this tutorial, you will:

- ✅ Download 2 real datasets from GEO
- ✅ Validate data quality and structure
- ✅ Annotate genes with standardized symbols
- ✅ Run statistical meta-analysis
- ✅ Identify significantly differentially expressed genes
- ✅ Create publication-quality visualizations

**Time required**: 30-45 minutes  
**Difficulty**: Beginner-friendly

## Prerequisites

### Software Requirements

- Python 3.10 or higher
- metaIntegrator-py git repository cloned:
  ```bash
git clone https://github.com/Khatri-Lab/metaintegrator_py.git
  ```

### Data Requirements

- Internet connection (for first download only)
- ~50 MB disk space for cached data

### Knowledge Requirements

- Basic Python programming
- Familiarity with pandas DataFrames (helpful but not required)
- Basic understanding of gene expression analysis (helpful but not required)

## Step-by-Step Workflow

### Step 1: Import Required Modules

First, let's import all the modules we'll need:

```python
import warnings
warnings.filterwarnings('ignore')  # Suppress pandas future warnings

# Preprocessing: Factory-based data loading and gene annotation
from preprocessing.loader_factory import DataLoaderFactory
from preprocessing.gene_annotator import GeneAnnotator
from preprocessing.gene_validation import EnsemblGeneValidator

# Core meta-analysis
from metaintegrator.analysis import MetaAnalyzer, filter_genes
from metaintegrator.validation import DatasetValidator
from metaintegrator.core_analysis import MetaAnalysisCore
from metaintegrator.score_calculation import ScoreCalculator

# Visualization: the "basic" plotter (volcano, forest, heatmap,
# effect-size distribution). Don't confuse this with the differently-named,
# differently-shaped enhanced_plotting.EnhancedMetaAnalysisPlotter used for
# ROC/PRC/violin/regression plots: see the Visualization API reference.
from visualization.plotting import MetaAnalysisPlotter
from visualization.violin_plot import violin_plot

# Standard data science libraries
import numpy as np
import pandas as pd
from pathlib import Path

print("✅ All modules imported successfully!")
```

**Expected output:**
```
✅ All modules imported successfully!
```


!!! tip "Optional: Batch Normalization for Multi-Platform Data"
    If you're combining datasets from very different platforms, consider
    running COCONUT batch normalization first:
    ```python
    from coconut.coconut_runner import runner
    runner(csv_dir='./data', output_dir='./normalized')
    ```
    See [Complete COCONUT Workflow](../coconut/workflow.md) for details.

### Step 2: Download Datasets from GEO

Let's download two real datasets studying disease vs. healthy samples:

```python
# Define datasets to analyze
gse_ids = ["GSE5847", "GSE28750"]

print(f"📥 Downloading {len(gse_ids)} datasets from GEO...")
print("⏱️  First download: 2-5 minutes (cached for future use)")

# Create a loader using the Factory pattern
# strategy='r_parity' matches R MetaIntegrator output exactly
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')

# Download and process data
datasets = loader.load_multiple(gse_ids)

print(f"\n✅ Downloaded {len(datasets)} datasets")

# Display dataset information
for ds in datasets:
    print(f"   • {ds.geo_id}: {ds.expr.shape[0]:,} probes × {ds.expr.shape[1]} samples, "
          f"{len(set(ds.keys))} unique genes")
```

**Expected output:**
```
📥 Downloading 2 datasets from GEO...
⏱️  First download: 2-5 minutes (cached for future use)

✅ Downloaded 2 datasets
   • GSE5847_GPL96: 22,283 probes × 50 samples, 13,042 unique genes
   • GSE28750_GPL570: 54,675 probes × 30 samples, 20,436 unique genes
```

**What happened?**  
- `DataLoaderFactory` selected the appropriate Source and Handler for each dataset
- Expression matrices were extracted from GEO Series Matrix files
- Gene symbols were annotated using GPL platform tables and validated against Ensembl
- Files were cached locally for instant re-use on subsequent runs

### Step 3: Build the Meta-Object

Convert the loaded datasets into the meta-object dictionary format used by the analysis layer:

```python
# Convert ProcessedDataset objects to meta-object format.
# ProcessedDataset has no to_dict(): build the dict by hand. 'class' is
# left as None here; Step 5 assigns it.
meta_object = {"originalData": {}}
for ds in datasets:
    meta_object["originalData"][ds.geo_id] = {
        "expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
        "formattedName": ds.geo_id, "class": None,
    }

print(f"✅ Meta-object created with {len(meta_object['originalData'])} datasets")
for name in meta_object["originalData"]:
    print(f"   • {name}")
```

### Step 4: Validate Data Structure

Before analysis, verify that data is properly formatted:

```python
print("🔍 Validating data structure...")

for name, dataset in meta_object['originalData'].items():
    print(f"\nValidating: {name}")
    
    try:
        # Check required fields (expr, class, keys)
        MetaAnalysisCore.check_metagem_study(dataset)
        print(f"   ✅ Data structure valid")
        
        # Check for missing values
        expr = dataset['expr']
        missing_pct = (expr.isna().sum().sum() / expr.size) * 100
        print(f"   ✅ Missing values: {missing_pct:.2f}%")
        
        # Check expression dimensions
        n_genes, n_samples = expr.shape
        print(f"   ✅ Dimensions: {n_genes:,} genes × {n_samples} samples")
        
    except ValueError as e:
        print(f"   ❌ Validation failed: {e}")
```

**Expected output:**
```
🔍 Validating data structure...

Validating: GSE5847_GPL96
   ✅ Data structure valid
   ✅ Missing values: 0.03%
   ✅ Dimensions: 22,283 genes × 50 samples

Validating: GSE28750_GPL570
   ✅ Data structure valid
   ✅ Missing values: 0.01%
   ✅ Dimensions: 54,675 genes × 30 samples
```

### Step 5: Assign Class Labels

Define which samples are cases vs. controls:

```python
print("\n🎯 Assigning class labels (Case vs. Control)...")

# In a real analysis, you would extract this from phenotype data.
# For this tutorial, we'll create balanced groups.

for name, dataset in meta_object['originalData'].items():
    n_samples = dataset['expr'].shape[1]
    
    # Class 0 = Control, Class 1 = Case
    n_cases = n_samples // 2
    n_controls = n_samples - n_cases
    
    class_vector = np.concatenate([
        np.zeros(n_controls, dtype=int),  # Controls
        np.ones(n_cases, dtype=int),      # Cases
    ])
    
    # Shuffle to randomize
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(class_vector)
    
    dataset['class'] = class_vector
    
    print(f"   {name}: {n_controls} controls, {n_cases} cases")

print("✅ Class labels assigned")
```

**Expected output:**
```
🎯 Assigning class labels (Case vs. Control)...
   GSE5847_GPL96: 25 controls, 25 cases
   GSE28750_GPL570: 15 controls, 15 cases
✅ Class labels assigned
```

!!! warning "Use Real Labels in Production"
    In your own analysis, extract class labels from the phenotype data
    (`dataset['pheno']`) based on your experimental design. The shuffled
    labels above are for tutorial demonstration only.

### Step 6: Validate the Complete Meta-Object

Run the full validation suite before meta-analysis:

```python
print("🔍 Running full meta-object validation...")

is_valid, errors = DatasetValidator.validate_meta_object(
    meta_object,
    check_type="Meta",
    stage="Pre-Analysis"
)

if is_valid:
    print("✅ Meta-object is valid and ready for analysis")
else:
    print("❌ Validation failed:")
    for error in errors:
        print(f"   • {error}")
```

### Step 7: Run Meta-Analysis

Perform statistical meta-analysis across all datasets:

```python
print("\n📊 Running statistical meta-analysis...")
print("   Computing effect sizes and combining p-values...")

# Initialize analyzer
analyzer = MetaAnalyzer(max_cores=4)

# Run meta-analysis with leave-one-out validation
results = analyzer.run_meta_analysis(
    meta_object,
    run_leave_one_out=True
)

print("✅ Meta-analysis completed!\n")

# Display top results
pooled = results["metaAnalysis"]["pooledResults"]

# Count significant genes
sig_fdr_05 = (pooled['effectSizeFDR'] < 0.05).sum()
sig_fdr_01 = (pooled['effectSizeFDR'] < 0.01).sum()

print(f"📈 Results Summary:")
print(f"   • Total genes tested: {len(pooled):,}")
print(f"   • Significant (FDR < 0.05): {sig_fdr_05:,} genes")
print(f"   • Highly significant (FDR < 0.01): {sig_fdr_01:,} genes")

# Show top 10 genes
print(f"\n   🏆 Top 10 Most Significant Genes:")
print("   " + "-" * 70)

top_10 = pooled.nsmallest(10, "effectSizeFDR")
for idx, (gene_name, row) in enumerate(top_10.iterrows(), 1):
    es = row["effectSize"]
    fdr = row["effectSizeFDR"]
    print(f"   {idx:2}. {gene_name:<15} ES={es:<+8.3f} FDR={fdr:.2e}")
```

### Step 8: Filter Genes

Apply significance thresholds to identify the gene signature:

```python
print("\n🔬 Filtering significant genes...")

results = analyzer.filter_genes(
    results,
    effect_size_thresh=0.5,
    fdr_thresh=0.05,
    number_studies_thresh=2,
)

# Access the FilterResult object
fr = results["filterResults"]

print(f"✅ Gene signature identified:")
print(f"   • Up-regulated genes:   {fr.n_pos}")
print(f"   • Down-regulated genes: {fr.n_neg}")
print(f"   • Total signature size: {fr.total_genes}")

if fr.n_pos > 0:
    print(f"\n   Top 5 up-regulated: {fr.pos_gene_names[:5]}")
if fr.n_neg > 0:
    print(f"   Top 5 down-regulated: {fr.neg_gene_names[:5]}")
```

### Step 9: Create Visualizations

Generate publication-quality plots:

```python
print("\n📊 Creating visualizations...")

# Create output directory
output_dir = Path("./meta_analysis_results")
output_dir.mkdir(exist_ok=True)

import matplotlib.pyplot as plt

# Initialize plotter
plotter = MetaAnalysisPlotter()

# --- 1. Volcano Plot ---
print("   Creating volcano plot...")
fig = plotter.plot_volcano(
    results,
    es_threshold=0.5,
    fdr_threshold=0.05,
    top_genes=20,
    save_path=str(output_dir / "volcano_plot.png"),
)
print(f"      ✅ Saved: {output_dir}/volcano_plot.png")

# --- 2. Forest Plot (top genes) ---
print("   Creating forest plot...")
top_genes = pooled.nsmallest(10, "effectSizeFDR").index.tolist()
fig = plotter.plot_forest_plot(
    results,
    genes=top_genes,
    save_path=str(output_dir / "forest_plot.png"),
)
print(f"      ✅ Saved: {output_dir}/forest_plot.png")

# --- 3. Heatmap ---
print("   Creating heatmap...")
fig = plotter.plot_heatmap(
    results,
    n_top_genes=50,
    save_path=str(output_dir / "heatmap.png"),
)
print(f"      ✅ Saved: {output_dir}/heatmap.png")

# --- 4. Effect Size Distribution ---
print("   Creating effect size distribution...")
fig = plotter.plot_effect_size_distribution(
    results,
    save_path=str(output_dir / "effect_size_dist.png"),
)
print(f"      ✅ Saved: {output_dir}/effect_size_dist.png")

# --- 5. Save results table ---
print("   Saving results table...")
pooled.to_csv(output_dir / "pooled_results.csv")
print(f"      ✅ Saved: {output_dir}/pooled_results.csv")

print(f"\n✅ All visualizations saved to: {output_dir}/")
```

### Step 10: Save Your Work

Save the meta-object for later analysis:

```python
import pickle

# Save complete meta-object
with open('meta_object.pkl', 'wb') as f:
    pickle.dump(results, f)

print("✅ Meta-object saved to: meta_object.pkl")
print("\n💡 Load later with:")
print("   with open('meta_object.pkl', 'rb') as f:")
print("       results = pickle.load(f)")
```

## Complete Working Script

Here's the complete script with all steps combined:

```python
import warnings
warnings.filterwarnings('ignore')

from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.analysis import MetaAnalyzer
from metaintegrator.validation import DatasetValidator
from visualization.plotting import MetaAnalysisPlotter
import numpy as np
from pathlib import Path

# Step 1: Download data
print("📥 Downloading datasets...")
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(["GSE5847", "GSE28750"])

# Step 2: Build meta-object (no to_dict() on ProcessedDataset: build by hand)
meta_object = {"originalData": {
    ds.geo_id: {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
                "formattedName": ds.geo_id, "class": None}
    for ds in datasets
}}

# Step 3: Assign class labels (use real labels in production!)
print("🎯 Assigning class labels...")
for dataset in meta_object['originalData'].values():
    n = dataset['expr'].shape[1]
    dataset['class'] = np.concatenate([np.zeros(n // 2, dtype=int),
                                       np.ones(n - n // 2, dtype=int)])

# Step 4: Validate
is_valid, errors = DatasetValidator.validate_meta_object(
    meta_object, stage="Pre-Analysis"
)
assert is_valid, f"Validation failed: {errors}"

# Step 5: Run meta-analysis
print("📊 Running meta-analysis...")
analyzer = MetaAnalyzer(max_cores=4)
results = analyzer.run_meta_analysis(meta_object, run_leave_one_out=True)

# Step 6: Filter genes
results = analyzer.filter_genes(
    results, effect_size_thresh=0.5, fdr_thresh=0.05
)
fr = results["filterResults"]
print(f"✅ Signature: {fr.n_pos} up, {fr.n_neg} down")

# Step 7: Visualize
print("📊 Creating visualizations...")
output_dir = Path("./results")
output_dir.mkdir(exist_ok=True)

plotter = MetaAnalysisPlotter()

plotter.plot_volcano(results, save_path=str(output_dir / "volcano.png"))
top_genes = results["metaAnalysis"]["pooledResults"].nsmallest(
    10, "effectSizeFDR"
).index.tolist()
plotter.plot_forest_plot(results, genes=top_genes,
                         save_path=str(output_dir / "forest.png"))
plotter.plot_heatmap(results, save_path=str(output_dir / "heatmap.png"))

print(f"✅ Complete! Results in {output_dir}/")
```

## Tips & Gotchas

### Tip 1: Use Cached Data for Faster Development
After the first download, data is cached locally. The factory loader handles caching automatically:

```python
# First run: ~3 minutes download
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(["GSE5847"])

# Subsequent runs: < 5 seconds (loaded from cache)
```

### Tip 2: Check Data Quality Early
Always validate data structure before meta-analysis:

```python
from metaintegrator.core_analysis import MetaAnalysisCore

for dataset in meta_object['originalData'].values():
    try:
        MetaAnalysisCore.check_metagem_study(dataset)
        print("✅ Valid")
    except ValueError as e:
        print(f"❌ Invalid: {e}")
```

### Tip 3: Use the Factory for Different Strategies

```python
# R-parity mode (matches R output exactly): recommended
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')

# Full SOFT download (all metadata)
loader = DataLoaderFactory.create_geo_loader(strategy='full')

# Fast exploration mode
loader = DataLoaderFactory.create_geo_loader(strategy='fast')
```

### Gotcha 1: Class Labels Must Be 0/1
Meta-analysis requires binary class vectors (0 = control, 1 = case):

```python
# ✅ Correct
dataset['class'] = np.array([0, 0, 0, 1, 1, 1])

# ❌ Wrong - will fail
dataset['class'] = np.array([1, 1, 1, 2, 2, 2])
dataset['class'] = np.array(['control', 'control', 'case'])
```

### Gotcha 2: Filter Results Are a Dataclass, Not a Dict
`filter_genes()` returns a `FilterResult` dataclass that supports both attribute and dict-style access:

```python
fr = results["filterResults"]

# Attribute access (preferred)
fr.pos_gene_names
fr.neg_gene_names
fr.total_genes

# R-style dict access (backward-compatible)
fr["posGeneNames"]
fr["negGeneNames"]
```

### Gotcha 3: Handle Multiple Platforms Carefully
Some GSE IDs contain multiple platforms. The loader handles this automatically, but results will have multiple entries:

```python
# One GSE might produce multiple datasets
datasets = loader.load_multiple(["GSE12345"])
# Result might include: GSE12345_GPL570 and GSE12345_GPL96
for ds in datasets:
    print(ds.geo_id)  # e.g. "GSE12345_GPL570"
```

## Troubleshooting

### Q: Download fails with "Connection timeout"

**A**: NCBI GEO servers may be temporarily unavailable. Try:
1. Check your internet connection
2. Wait 5-10 minutes and retry
3. Try a different dataset to verify connectivity

### Q: "No common genes across datasets"

**A**: This happens when datasets use incompatible platforms. Solutions:
1. Verify all datasets use similar platforms (e.g., all Affymetrix)
2. Check gene annotation quality
3. Use datasets from the same organism (e.g., all human)

### Q: Memory error with large datasets

**A**: Large datasets (> 500 samples, > 50k genes) may require more RAM:
1. Use smaller datasets for testing
2. Filter genes before meta-analysis
3. Process one dataset at a time
4. Increase system swap space

### Q: Volcano plot shows no significant genes

**A**: This can happen with small sample sizes or incompatible studies:
1. Verify class labels are correctly assigned
2. Check that studies measure similar conditions
3. Ensure adequate sample size (> 10 samples per group)
4. Review data quality metrics

## Next Steps

Congratulations! You've completed a full meta-analysis workflow. Here's what to explore next:

1. **[Data Acquisition Tutorial](data_acquisition.md)** – Learn advanced download options
2. **[Gene Annotation Tutorial](gene_annotation.md)** – Deep dive into annotation strategies
3. **[Quality Control Tutorial](quality_control.md)** – Comprehensive QC procedures
4. **[Visualization Tutorial](visualization.md)** – Advanced plotting techniques

### Apply to Your Data

Ready to analyze your own datasets? Modify this workflow:

```python
# Replace with your GSE IDs
gse_ids = ["GSE12345", "GSE67890", "GSE11111"]

loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(gse_ids)

meta_object = {"originalData": {
    ds.geo_id: {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
                "formattedName": ds.geo_id, "class": None}
    for ds in datasets
}}

# Extract real class labels from phenotype data
for name, dataset in meta_object['originalData'].items():
    pheno = dataset['pheno']
    dataset['class'] = (pheno['disease_status'] == 'case').astype(int).values
```

## Summary

You've learned how to:

✅ Download real gene expression data from GEO using `DataLoaderFactory`  
✅ Validate and quality-check datasets with `DatasetValidator`  
✅ Build meta-objects from `ProcessedDataset` results  
✅ Run statistical meta-analysis with `MetaAnalyzer`  
✅ Filter significant genes and access `FilterResult` dataclass  
✅ Create publication-ready visualizations  

**Time invested**: 30-45 minutes  
**Skills gained**: Complete meta-analysis workflow  
**Next milestone**: Apply to your research data

---

**Questions or issues?** Open an issue on GitHub or check the [Best Practices Guide](../user_guide/best_practices.md).
