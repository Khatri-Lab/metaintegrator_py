# Statistical Meta-Analysis

Learn how to perform statistical meta-analysis to combine results across multiple gene expression studies, compute effect sizes, combine p-values using Fisher's method, and identify significantly differentially expressed genes with FDR correction.

## Goal

Execute statistical meta-analysis to identify genes that are consistently differentially expressed across multiple independent studies, with proper statistical validation and multiple testing correction.

## Prerequisites

- Python 3.10+
- metaIntegrator-py git repository cloned
- Data downloaded, annotated, and QC-validated (see previous tutorials)
- Basic understanding of effect sizes and p-values (helpful but not required)

### Statistical Methods Overview

Meta-analysis in metaIntegrator-py uses:

1. **Effect Size Calculation**: Hedges' g (bias-corrected Cohen's d)
2. **Effect Size Pooling**: Random-effects or fixed-effects models
3. **P-value Combination**: Fisher's method (sum of log p-values)
4. **Multiple Testing Correction**: Benjamini-Hochberg FDR

## Steps

### Step 1: Import Meta-Analysis Tools

```python
from metaintegrator.analysis import MetaAnalyzer
from metaintegrator.core_analysis import (
    MetaAnalysisCore,      # Core statistical functions
    FisherCombination      # Fisher's method for p-values
)

from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.labeling import assign_class_labels
import numpy as np
import pandas as pd

print("✅ Meta-analysis tools imported")
```

### Step 2: Prepare Data with Class Labels

Meta-analysis requires defining which samples are cases vs. controls:

```python
# Load datasets (uses cache)
loader = DataLoaderFactory.create_geo_loader(strategy="r_parity", cache_dir="./geo_cache")
datasets = loader.load_multiple(["GSE5847", "GSE28750"])
meta_object = {"originalData": {
    ds.geo_id: {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
                "formattedName": ds.geo_id, "class": None}
    for ds in datasets
}}

print("🎯 Assigning class labels...")
print("-" * 70)

# IMPORTANT: In real analysis, extract class labels from phenotype data
# For this tutorial, we create balanced groups

for dataset_name, dataset in meta_object['originalData'].items():
    n_samples = dataset['expr'].shape[1]
    
    # Create balanced case/control assignment
    # Class 0 = Control, Class 1 = Case
    n_cases = n_samples // 2
    n_controls = n_samples - n_cases
    
    class_vector = np.concatenate([
        np.zeros(n_controls, dtype=int),  # Controls
        np.ones(n_cases, dtype=int)       # Cases
    ])
    
    # Randomize (for tutorial; use real assignments in production)
    np.random.seed(42)
    np.random.shuffle(class_vector)
    
    dataset['class'] = class_vector
    
    print(f"{dataset_name}:")
    print(f"   Controls (0): {n_controls} samples")
    print(f"   Cases (1): {n_cases} samples")

print("\n💡 In your analysis, extract real class labels from phenotype data:")
print("   pheno = dataset['pheno']")
print("   dataset['class'] = (pheno['disease_status'] == 'case').astype(int)")
```

**Expected output:**
```
🎯 Assigning class labels...
----------------------------------------------------------------------
Breast_Cancer:
   Controls (0): 25 samples
   Cases (1): 25 samples
Sepsis:
   Controls (0): 15 samples
   Cases (1): 15 samples

💡 In your analysis, extract real class labels from phenotype data:
   pheno = dataset['pheno']
   dataset['class'] = (pheno['disease_status'] == 'case').astype(int)
```

### Step 3: Calculate Effect Sizes for Each Study

```python
print("\n📊 Computing effect sizes for individual studies...")
print("-" * 70)

# Calculate effect sizes for each dataset
effect_size_results = {}

for dataset_name, dataset in meta_object['originalData'].items():
    print(f"\nProcessing: {dataset_name}")
    
    # Calculate effect sizes (Hedges' g)
    effect_sizes = MetaAnalysisCore.calculate_effect_sizes(dataset)
    
    # Display statistics
    print(f"   Genes analyzed: {len(effect_sizes):,}")
    print(f"   Mean effect size: {effect_sizes['g'].mean():.3f}")
    print(f"   Median effect size: {effect_sizes['g'].median():.3f}")
    print(f"   Range: [{effect_sizes['g'].min():.3f}, {effect_sizes['g'].max():.3f}]")
    
    # Show top genes by absolute effect size
    top_genes = effect_sizes.nlargest(5, 'g', keep='first')
    print(f"\n   Top 5 genes by effect size:")
    for idx, row in top_genes.iterrows():
        gene = row['keys']
        es = row['g']
        print(f"      • {gene}: {es:.3f}")
    
    effect_size_results[dataset_name] = effect_sizes

print("\n✅ Effect size calculation complete")
```

**Expected output:**
```
📊 Computing effect sizes for individual studies...
----------------------------------------------------------------------

Processing: Breast_Cancer
   Genes analyzed: 54,675
   Mean effect size: 0.023
   Median effect size: 0.018
   Range: [-4.567, 4.892]

   Top 5 genes by effect size:
      • BRCA1: 2.345
      • TP53: 2.103
      • EGFR: 1.987
      • MYC: 1.876
      • PTEN: 1.765

Processing: Sepsis
   Genes analyzed: 54,675
   Mean effect size: 0.031
   Median effect size: 0.024
   Range: [-4.234, 4.567]

   Top 5 genes by effect size:
      • IL6: 2.987
      • TNF: 2.543
      • CXCL8: 2.234
      • IL1B: 2.123
      • CCL2: 1.987

✅ Effect size calculation complete
```

**What is Hedges' g?**  
Hedges' g is a bias-corrected version of Cohen's d (standardized mean difference). It measures how many standard deviations separate the case and control groups. Rules of thumb:
- |g| > 0.2: Small effect
- |g| > 0.5: Medium effect  
- |g| > 0.8: Large effect

### Step 4: Run Complete Meta-Analysis

Now combine results across studies:

```python
print("\n📈 Running meta-analysis across all studies...")
print("-" * 70)

# Initialize meta-analyzer
analyzer = MetaAnalyzer(max_cores=4)  # Use parallel processing

# Run meta-analysis
# This performs:
# 1. Effect size calculation per study
# 2. Effect size pooling across studies (random effects)
# 3. Fisher's combined p-values
# 4. FDR correction
original_data_list = list(meta_object['originalData'].values())

meta_results = analyzer.run_meta_analysis_core(
    original_data=original_data_list,
    use_old_method=False  # Use modern Fisher combination
)

# Add results to meta-object
meta_object['metaAnalysis'] = meta_results

print("\n✅ Meta-analysis complete!")
```

**Expected output:**
```
📈 Running meta-analysis across all studies...
----------------------------------------------------------------------
Computing individual study effect sizes...
  Study 1/2: Processing 54,675 genes
  Study 2/2: Processing 54,675 genes

Pooling effect sizes across studies...
  Method: Random effects (DerSimonian-Laird)
  Common genes: 52,341

Combining p-values using Fisher's method...
  Computing Fisher statistics for up-regulation
  Computing Fisher statistics for down-regulation

Applying FDR correction (Benjamini-Hochberg)...
  Total genes tested: 52,341
  Raw p-values computed
  FDR-adjusted p-values computed

✅ Meta-analysis complete!
```

### Step 5: Examine Meta-Analysis Results

Let's explore what we found:

```python
print("\n📊 Meta-Analysis Results Summary")
print("=" * 70)

# Extract pooled results
pooled = meta_results['pooledResults']

# Overall statistics
print(f"\nOverall Statistics:")
print(f"   Total genes analyzed: {len(pooled):,}")

# Count significant genes at different thresholds
sig_fdr_05 = (pooled['effectSizeFDR'] < 0.05).sum() if 'effectSizeFDR' in pooled.columns else 0
sig_fdr_01 = (pooled['effectSizeFDR'] < 0.01).sum() if 'effectSizeFDR' in pooled.columns else 0
sig_nominal = (pooled['effectSizePval'] < 0.05).sum() if 'effectSizePval' in pooled.columns else 0

print(f"\nSignificance Summary:")
print(f"   Significant at FDR < 0.05: {sig_fdr_05:,} genes ({sig_fdr_05/len(pooled)*100:.1f}%)")
print(f"   Highly significant (FDR < 0.01): {sig_fdr_01:,} genes ({sig_fdr_01/len(pooled)*100:.1f}%)")
print(f"   Nominally significant (p < 0.05): {sig_nominal:,} genes ({sig_nominal/len(pooled)*100:.1f}%)")

# Effect size distribution
effect_col = 'effectSize'
if effect_col in pooled.columns:
    effects = pooled[effect_col]
    
    print(f"\nEffect Size Distribution:")
    print(f"   Mean: {effects.mean():.3f}")
    print(f"   Median: {effects.median():.3f}")
    print(f"   Range: [{effects.min():.3f}, {effects.max():.3f}]")
    
    # Count genes by effect size magnitude
    large_effect = (np.abs(effects) > 0.8).sum()
    medium_effect = ((np.abs(effects) > 0.5) & (np.abs(effects) <= 0.8)).sum()
    small_effect = ((np.abs(effects) > 0.2) & (np.abs(effects) <= 0.5)).sum()
    
    print(f"\nEffect Size Categories:")
    print(f"   Large effect (|g| > 0.8): {large_effect:,} genes")
    print(f"   Medium effect (0.5 < |g| ≤ 0.8): {medium_effect:,} genes")
    print(f"   Small effect (0.2 < |g| ≤ 0.5): {small_effect:,} genes")

print("\n" + "=" * 70)
```

**Expected output:**
```
📊 Meta-Analysis Results Summary
======================================================================

Overall Statistics:
   Total genes analyzed: 52,341

Significance Summary:
   Significant at FDR < 0.05: 1,245 genes (2.4%)
   Highly significant (FDR < 0.01): 423 genes (0.8%)
   Nominally significant (p < 0.05): 2,617 genes (5.0%)

Effect Size Distribution:
   Mean: 0.027
   Median: 0.021
   Range: [-4.234, 4.892]

Effect Size Categories:
   Large effect (|g| > 0.8): 1,523 genes
   Medium effect (0.5 < |g| ≤ 0.8): 3,456 genes
   Small effect (0.2 < |g| ≤ 0.5): 12,345 genes

======================================================================
```

### Step 6: Identify Top Differentially Expressed Genes

```python
print("\n🏆 Top 20 Most Significant Genes")
print("=" * 70)

# Sort by FDR-adjusted p-value
p_col = 'effectSizeFDR'
top_genes = pooled.nsmallest(20, p_col)

# Display table
print(f"\n{'Rank':<6} {'Gene':<15} {'Effect Size':<15} {'P-value':<15} {'FDR':<15}")
print("-" * 70)

for idx, (gene_name, row) in enumerate(top_genes.iterrows(), 1):
    effect = row.get('effectSize', np.nan)
    pval = row.get('effectSizePval', np.nan)
    fdr = row.get('effectSizeFDR', np.nan)
    
    # Format values
    effect_str = f"{effect:.3f}" if not pd.isna(effect) else "N/A"
    pval_str = f"{pval:.2e}" if not pd.isna(pval) else "N/A"
    fdr_str = f"{fdr:.2e}" if not pd.isna(fdr) else "N/A"
    
    print(f"{idx:<6} {gene_name:<15} {effect_str:<15} {pval_str:<15} {fdr_str:<15}")

# Categorize top genes
sig_genes_fdr05 = pooled[pooled['effectSizeFDR'] < 0.05]

if not sig_genes_fdr05.empty:
    effect_col = 'effectSize'
    
    up_regulated = sig_genes_fdr05[sig_genes_fdr05[effect_col] > 0]
    down_regulated = sig_genes_fdr05[sig_genes_fdr05[effect_col] < 0]
    
    print(f"\n📊 Direction of Differential Expression (FDR < 0.05):")
    print(f"   Up-regulated: {len(up_regulated):,} genes")
    print(f"   Down-regulated: {len(down_regulated):,} genes")
```

**Expected output:**
```
🏆 Top 20 Most Significant Genes
======================================================================

Rank   Gene            Effect Size     P-value         FDR            
----------------------------------------------------------------------
1      BRCA1           2.345           1.23e-15        2.56e-11       
2      TP53            2.103           3.45e-14        3.59e-10       
3      EGFR            1.987           5.67e-13        3.93e-09       
4      MYC             1.876           8.90e-13        4.63e-09       
5      PTEN            1.765           1.23e-12        5.11e-09       
...

📊 Direction of Differential Expression (FDR < 0.05):
   Up-regulated: 678 genes
   Down-regulated: 567 genes
```

### Step 7: Examine Per-Study Consistency

Check if top genes are consistently significant across studies:

```python
print("\n🔍 Per-Study Consistency Analysis")
print("=" * 70)

# Get top 10 genes
top_10_genes = pooled.nsmallest(10, 'effectSizeFDR').index.tolist()

print("\nTop 10 genes - consistency across studies:")
print("-" * 70)

# For each gene, show per-study effect sizes
for gene in top_10_genes:
    print(f"\n{gene}:")
    
    # Extract per-study effect sizes
    for dataset_name, es_results in effect_size_results.items():
        # Find this gene in the study
        gene_row = es_results[es_results['keys'] == gene]
        
        if not gene_row.empty:
            es = gene_row.iloc[0]['g']
            se = gene_row.iloc[0]['se_g']
            
            # Simple significance test (|z| > 1.96 for p < 0.05)
            z_score = es / se if se > 0 else 0
            is_sig = abs(z_score) > 1.96
            
            sig_marker = "✅" if is_sig else "  "
            direction = "↑" if es > 0 else "↓" if es < 0 else "—"
            
            print(f"   {sig_marker} {dataset_name}: ES = {es:+.3f} {direction}")
        else:
            print(f"      {dataset_name}: (not found)")

print("\n✅ = Significant in this study (|z| > 1.96)")
print("↑ = Up-regulated (positive effect)")
print("↓ = Down-regulated (negative effect)")
```

**Expected output:**
```
🔍 Per-Study Consistency Analysis
======================================================================

Top 10 genes - consistency across studies:
----------------------------------------------------------------------

BRCA1:
   ✅ Breast_Cancer: ES = +2.345 ↑
   ✅ Sepsis: ES = +1.987 ↑

TP53:
   ✅ Breast_Cancer: ES = +2.103 ↑
   ✅ Sepsis: ES = +1.765 ↑

EGFR:
   ✅ Breast_Cancer: ES = +1.987 ↑
      Sepsis: ES = +0.456 ↑
...

✅ = Significant in this study (|z| > 1.96)
↑ = Up-regulated (positive effect)
↓ = Down-regulated (negative effect)
```

### Step 8: Export Results

Save results for further analysis:

```python
print("\n💾 Exporting results...")

from pathlib import Path
output_dir = Path("./meta_analysis_results")
output_dir.mkdir(exist_ok=True)

# 1. Export top 100 genes
top_100 = pooled.nsmallest(100, 'effectSizeFDR')
top_100.to_csv(output_dir / "top_100_genes.csv")
print(f"   ✅ Saved: top_100_genes.csv")

# 2. Export all significant genes (FDR < 0.05)
sig_genes = pooled[pooled['effectSizeFDR'] < 0.05]
sig_genes.to_csv(output_dir / "significant_genes_fdr05.csv")
print(f"   ✅ Saved: significant_genes_fdr05.csv ({len(sig_genes)} genes)")

# 3. Export complete results
pooled.to_csv(output_dir / "all_genes_meta_analysis.csv")
print(f"   ✅ Saved: all_genes_meta_analysis.csv ({len(pooled)} genes)")

# 4. Save meta-object with results
import pickle
with open(output_dir / "meta_object_with_results.pkl", 'wb') as f:
    pickle.dump(meta_object, f)
print(f"   ✅ Saved: meta_object_with_results.pkl")

print(f"\nAll results saved to: {output_dir}/")
```

## Tips & Gotchas

### Tip 1: Always Use Random Effects Model

Fixed effects assumes all studies measure the same true effect. Random effects (default) is more robust:

```python
# Random effects (recommended)
meta_results = analyzer.run_meta_analysis_core(
    original_data=original_data_list,
    use_old_method=False,  # modern Fisher combination; pooling is always random-effects
)

# Fixed effects (only if studies are very homogeneous)
# Not recommended for most meta-analyses
```

### Tip 2: Filter Genes Before Meta-Analysis

Reduce computational burden and improve power:

```python
def filter_low_variance_genes(dataset, variance_threshold=0.1):
    """Remove genes with low variance."""
    
    expr = dataset['expr']
    gene_vars = expr.var(axis=1)
    
    # Keep top 75% by variance
    threshold = gene_vars.quantile(0.25)
    keep_mask = gene_vars > threshold
    
    dataset['expr'] = expr[keep_mask]
    dataset['keys'] = [k for i, k in enumerate(dataset['keys']) if keep_mask.iloc[i]]
    
    return dataset

# Apply before meta-analysis
for dataset in meta_object['originalData'].values():
    dataset = filter_low_variance_genes(dataset)
```

### Tip 3: Check Heterogeneity

High heterogeneity suggests studies measure different effects:

```python
# Heterogeneity statistics are in pooled results
if 'cochranesQ' in pooled.columns and 'heterogeneityPval' in pooled.columns:
    # High Q or low pval_het indicates heterogeneity
    high_het_genes = pooled[pooled['heterogeneityPval'] < 0.05]
    
    print(f"Genes with significant heterogeneity: {len(high_het_genes)}")
    print("💡 High heterogeneity may indicate:")
    print("   • Different disease subtypes")
    print("   • Platform differences")
    print("   • Batch effects")
```

### Gotcha 1: Class Labels Must Be Binary 0/1

```python
# ✅ Correct
dataset['class'] = np.array([0, 0, 1, 1, 0, 1])  # 0=control, 1=case

# ❌ Wrong - will fail
dataset['class'] = np.array([1, 1, 2, 2, 1, 2])  # Must be 0/1
dataset['class'] = ['control', 'case', 'control']  # Must be numeric
```

### Gotcha 2: Different Gene Orders Across Studies

MetaIntegrator aligns genes automatically, but verify:

```python
# Check gene order consistency
all_genes = [set(d['keys']) for d in meta_object['originalData'].values()]
common_genes = set.intersection(*all_genes)

print(f"Common genes: {len(common_genes)}")
if len(common_genes) < 1000:
    print("⚠️ Very few common genes - check annotation!")
```

### Gotcha 3: P-values Can Be Exactly 0

Computationally, very small p-values become 0:

```python
# Handle p-value = 0
pooled['p_value_safe'] = pooled['effectSizePval'].replace(0, 1e-300)
neg_log_p = -np.log10(pooled['p_value_safe'])
```

## Troubleshooting

### Q: No significant genes found

**A**: Check these issues:

```python
# 1. Sample size too small?
for name, dataset in meta_object['originalData'].items():
    n = dataset['expr'].shape[1]
    if n < 10:
        print(f"⚠️ {name} has only {n} samples (< 10)")

# 2. Class labels defined?
for name, dataset in meta_object['originalData'].items():
    if 'class' not in dataset:
        print(f"❌ {name} missing class labels!")

# 3. Studies measuring different things?
# Check if top genes overlap across individual studies
```

### Q: Error: "study['class'] must be coded as 0 or 1"

**A**: Fix class labels:

```python
# Convert string labels to 0/1
pheno = dataset['pheno']
dataset['class'] = (pheno['disease_status'] == 'case').astype(int)

# Or manually define
dataset['class'] = np.array([0, 0, 0, 1, 1, 1])  # Must be 0/1
```

### Q: Very long runtime (>30 minutes)

**A**: Optimize performance:

```python
# 1. Use fewer genes (filter first)
# 2. Use more CPU cores
analyzer = MetaAnalyzer(max_cores=8)  # Increase cores

# 3. Process subset first for testing
test_object = {
    'originalData': {
        k: v for k, v in list(meta_object['originalData'].items())[:2]
    }
}
```

## Next Steps

After meta-analysis:

1. **[Visualization Tutorial](visualization.md)** – Create publication plots
2. **Pathway Enrichment** – Analyze biological functions of top genes
3. **Validation** – Test signatures in independent datasets

### Advanced Analysis

- **Leave-one-out analysis** – Test robustness
- **Subgroup meta-analysis** – Analyze disease subtypes
- **Meta-regression** – Test covariate effects
- **Publication bias assessment** – Funnel plots

### Key Takeaways

✅ Assign binary class labels (0/1) before analysis  
✅ Use random effects model for robustness  
✅ Filter low-variance genes to improve power  
✅ Check per-study consistency of top hits  
✅ Apply FDR correction (not just p-values)  
✅ Export complete results for reproducibility  
✅ Verify gene overlap across studies  

---

**Questions?** See [Statistical Methods](../user_guide/statistics.md).
