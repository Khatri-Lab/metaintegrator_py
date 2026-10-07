# Gene Filtering Strategies

This guide explains how to filter genes before meta-analysis to improve statistical power, reduce computational burden, and focus on biologically relevant features. Proper filtering is crucial for robust meta-analysis results.

## Contents

- [Introduction](#introduction)
- [Why Filter Genes?](#why-filter-genes)
- [metaintegrator_py's Built-In Gene Filtering](#metaintegrator_pys-built-in-gene-filtering)
- [Generic Pre-Analysis QC Techniques](#generic-pre-analysis-qc-techniques)
- [Advanced Filtering Strategies](#advanced-filtering-strategies)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Appendix](#appendix)

## Introduction

Gene filtering removes uninformative or low-quality features before statistical analysis. While microarray platforms measure 20,000-50,000+ probes, many are not relevant for a given analysis. Strategic filtering improves results by:

- Reducing multiple testing burden (fewer genes = less stringent FDR)
- Removing low-quality/noisy measurements
- Focusing on biologically interpretable genes
- Decreasing computation time and memory usage

**Important:** Apply filters **consistently across all studies** to avoid bias.

!!! note "What's actually built into metaintegrator_py?"
    Only two things on this page are metaintegrator_py functions: the threshold
    filters in [`filter_genes()`](#metaintegrator_pys-built-in-gene-filtering) (effect
    size, FDR, number of studies, heterogeneity, leave-one-out robustness -- the
    Python equivalent of R MetaIntegrator's `filterGenes()`), and the
    [`forward_search()` / `backward_search()`](#metaintegrator_pys-built-in-gene-filtering)
    signature-refinement functions. Everything under "Generic Pre-Analysis QC
    Techniques" and "Advanced Filtering Strategies" below is standard,
    library-agnostic data science practice for cleaning an expression matrix
    (variance filtering, detection-rate filtering, and so on) -- useful, but not
    something metaintegrator_py calls for you. You're welcome to apply it to your
    raw data before building a `meta_object`, but it isn't part of the package API.

## Why Filter Genes?

### Multiple Testing Problem

With 20,000+ genes, ~1,000 will appear significant by chance at p < 0.05:

```python
# Without filtering: 20,000 genes tested
# Expected false positives at p < 0.05: 20,000 × 0.05 = 1,000

# With filtering: 5,000 genes tested  
# Expected false positives: 5,000 × 0.05 = 250

# Reduces false discoveries by 75%!
```

### Signal-to-Noise Ratio

Low-variance genes contribute noise without useful information:

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulated gene with low variance (uninformative)
low_var_gene = np.random.normal(7.0, 0.1, 100)  # Mean=7, SD=0.1

# Simulated gene with high variance (potentially informative)
high_var_gene = np.random.normal(7.0, 1.5, 100)  # Mean=7, SD=1.5

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(low_var_gene, bins=20, alpha=0.7)
axes[0].set_title('Low Variance Gene (uninformative)')
axes[1].hist(high_var_gene, bins=20, alpha=0.7)
axes[1].set_title('High Variance Gene (informative)')
plt.show()
```

### Computational Efficiency

Filtering reduces runtime and memory:

| Genes | Runtime | Memory |
|-------|---------|--------|
| 50,000 | ~30 min | ~4 GB |
| 20,000 | ~10 min | ~1.5 GB |
| 10,000 | ~5 min | ~750 MB |
| 5,000 | ~2 min | ~400 MB |

## metaintegrator_py's Built-In Gene Filtering

Before reaching for any of the generic techniques below, know what the package
already gives you. These are the Python equivalents of R MetaIntegrator's own
filtering functions, and they operate on meta-analysis *results* (effect sizes,
FDRs, study counts) rather than on raw expression matrices.

### Threshold-Based Filtering: `filter_genes()`

Filters the pooled meta-analysis results down to a significant gene signature:

```python
from metaintegrator.analysis import MetaAnalyzer

analyzer = MetaAnalyzer()
results = analyzer.run_meta_analysis(meta_object)

results = analyzer.filter_genes(
    results,
    effect_size_thresh=0.5,      # minimum |effect size|
    fdr_thresh=0.05,             # maximum FDR (effect-size and Fisher)
    number_studies_thresh=2,     # minimum contributing studies
    heterogeneity_pval_thresh=0, # 0 disables this filter
    is_leave_one_out=True,       # gene must pass in every leave-one-out run
)

print(results['filterResults'].pos_gene_names)
```

**How leave-one-out filtering works.**
It follows R's `filterGenes(isLeaveOneOut = TRUE)` exactly.
The filter does not look at the main meta-analysis at all.
Instead, it runs every threshold on each leave-one-out result, where one study was left out.
Because one study is missing, the studies threshold drops by one in these runs.
A gene is kept only if it passes in **every** leave-one-out run.
So a gene that depends on a single study does not make the signature.

Leave-one-out filtering needs leave-one-out results and more than two datasets.
Otherwise the filter runs once on the main result,
and `filterResults.is_leave_one_out` is `False`.

`number_studies_thresh` is the built-in equivalent of "keep genes present in at
least N of my studies" -- if you've been thinking of that as a percentage, set
it to the number of studies that percentage corresponds to.

### Signature Refinement: `forward_search()` / `backward_search()`

Once you have a candidate gene set from `filter_genes()`, these iteratively
add (forward) or remove (backward) genes based on their contribution to the
signature's overall weighted AUC:

```python
from metaintegrator.gene_search_optimizer import GeneSearchOptimizer

optimizer = GeneSearchOptimizer()
filter_object = results['filterResults'].to_dict()  # posGeneNames / negGeneNames

refined = optimizer.forward_search(meta_object, filter_object, forward_thresh=0.0)
refined = optimizer.backward_search(meta_object, refined, back_thresh=0.0)
```

## Generic Pre-Analysis QC Techniques

The techniques below are common data-science practice for cleaning an
expression matrix -- they are not metaintegrator_py functions, and none of the
code on this page calls the package. Apply them (if at all) to your raw data
**before** building your `meta_object`, using whatever tooling you're
comfortable with (pandas/numpy, as shown, or otherwise).

## Core Filtering Methods

### 1. Variance Filtering

Remove genes with low variance (little signal):

```python
import pandas as pd
import numpy as np

def filter_by_variance(dataset, percentile=25):
    """
    Remove genes with low variance.
    
    Args:
        dataset: Dataset dictionary
        percentile: Keep genes above this variance percentile (default: 25 = top 75%)
    
    Returns:
        Filtered dataset
    """
    expr = dataset['expr']
    
    # Calculate variance for each gene
    gene_vars = expr.var(axis=1)
    
    # Determine threshold
    threshold = gene_vars.quantile(percentile / 100)
    
    # Filter
    keep_mask = gene_vars > threshold
    
    print(f"Variance filtering:")
    print(f"  Original genes: {len(gene_vars):,}")
    print(f"  Variance threshold: {threshold:.4f}")
    print(f"  Genes retained: {keep_mask.sum():,} ({keep_mask.sum()/len(gene_vars)*100:.1f}%)")
    
    # Apply filter
    dataset['expr'] = expr[keep_mask]
    dataset['keys'] = [k for i, k in enumerate(dataset['keys']) if keep_mask.iloc[i]]
    
    return dataset

# Example usage
filtered_dataset = filter_by_variance(dataset, percentile=25)
```

**Recommended threshold:** Keep top 50-75% by variance

### 2. Detection Rate Filtering

Remove genes not detected in sufficient samples:

```python
def filter_by_detection(dataset, min_detection_rate=0.5, detection_threshold=0):
    """
    Remove genes not detected in minimum fraction of samples.
    
    Args:
        dataset: Dataset dictionary
        min_detection_rate: Minimum fraction of samples with detection (0-1)
        detection_threshold: Expression value above which gene is "detected"
    
    Returns:
        Filtered dataset
    """
    expr = dataset['expr']
    
    # Calculate detection rate per gene
    detected = (expr > detection_threshold).sum(axis=1) / expr.shape[1]
    
    # Filter
    keep_mask = detected >= min_detection_rate
    
    print(f"Detection filtering:")
    print(f"  Original genes: {len(detected):,}")
    print(f"  Detection threshold: > {detection_threshold}")
    print(f"  Min detection rate: {min_detection_rate:.0%}")
    print(f"  Genes retained: {keep_mask.sum():,} ({keep_mask.sum()/len(detected)*100:.1f}%)")
    
    # Apply filter
    dataset['expr'] = expr[keep_mask]
    dataset['keys'] = [k for i, k in enumerate(dataset['keys']) if keep_mask.iloc[i]]
    
    return dataset

# Example usage
filtered_dataset = filter_by_detection(dataset, min_detection_rate=0.5)
```

**Recommended threshold:** 50% detection rate (gene present in >half of samples)

### 3. Absolute Expression Filtering

Remove genes with consistently low expression:

```python
def filter_by_expression(dataset, min_expression=5.0):
    """
    Remove genes with mean expression below threshold.
    
    Args:
        dataset: Dataset dictionary
        min_expression: Minimum mean log2 expression
    
    Returns:
        Filtered dataset
    """
    expr = dataset['expr']
    
    # Calculate mean expression per gene
    gene_means = expr.mean(axis=1)
    
    # Filter
    keep_mask = gene_means >= min_expression
    
    print(f"Expression level filtering:")
    print(f"  Original genes: {len(gene_means):,}")
    print(f"  Min expression: {min_expression}")
    print(f"  Genes retained: {keep_mask.sum():,} ({keep_mask.sum()/len(gene_means)*100:.1f}%)")
    
    # Apply filter
    dataset['expr'] = expr[keep_mask]
    dataset['keys'] = [k for i, k in enumerate(dataset['keys']) if keep_mask.iloc[i]]
    
    return dataset

# Example usage (for log2 data)
filtered_dataset = filter_by_expression(dataset, min_expression=5.0)
```

**Recommended threshold:** 5.0 for log₂ scale (≈32 in linear scale)

### 4. Cross-Study Gene Presence

Keep only genes present in minimum number of studies:

```python
def filter_by_study_presence(meta_object, min_studies=2):
    """
    Remove genes not present in minimum number of studies.
    
    Args:
        meta_object: Meta-object with multiple datasets
        min_studies: Minimum number of studies gene must appear in
    
    Returns:
        Filtered meta-object
    """
    # Collect all gene sets
    all_gene_sets = []
    for dataset in meta_object['originalData'].values():
        all_gene_sets.append(set(dataset['keys']))
    
    # Count studies each gene appears in
    from collections import Counter
    gene_counts = Counter()
    for gene_set in all_gene_sets:
        for gene in gene_set:
            gene_counts[gene] += 1
    
    # Identify genes to keep
    genes_to_keep = {gene for gene, count in gene_counts.items() if count >= min_studies}
    
    print(f"Cross-study filtering:")
    print(f"  Total unique genes: {len(gene_counts):,}")
    print(f"  Min studies required: {min_studies}/{len(all_gene_sets)}")
    print(f"  Genes retained: {len(genes_to_keep):,}")
    
    # Filter each dataset
    for name, dataset in meta_object['originalData'].items():
        keep_mask = [gene in genes_to_keep for gene in dataset['keys']]
        
        dataset['expr'] = dataset['expr'].iloc[keep_mask]
        dataset['keys'] = [k for k, keep in zip(dataset['keys'], keep_mask) if keep]
        
        print(f"  {name}: {sum(keep_mask):,} genes retained")
    
    return meta_object

# Example usage
filtered_meta = filter_by_study_presence(meta_object, min_studies=2)
```

**Recommended threshold:** All studies (ensures common gene set)

### 5. Control Probe Removal

Remove platform control probes:

```python
def filter_control_probes(dataset):
    """
    Remove platform control probes (AFFX, ILMN, etc.).
    
    Args:
        dataset: Dataset dictionary
    
    Returns:
        Filtered dataset
    """
    # Identify control probes
    control_patterns = ['AFFX-', 'ILMN_', 'DarkCorner', 'Negative']
    
    is_control = [
        any(pattern in str(key) for pattern in control_patterns)
        for key in dataset['keys']
    ]
    
    # Filter (keep non-controls)
    keep_mask = [not ctrl for ctrl in is_control]
    
    n_removed = sum(is_control)
    print(f"Control probe filtering:")
    print(f"  Control probes removed: {n_removed:,}")
    print(f"  Genes retained: {sum(keep_mask):,}")
    
    # Apply filter
    dataset['expr'] = dataset['expr'].iloc[keep_mask]
    dataset['keys'] = [k for k, keep in zip(dataset['keys'], keep_mask) if keep]
    
    return dataset

# Example usage
filtered_dataset = filter_control_probes(dataset)
```

### 6. Combined Filtering Pipeline

Apply multiple filters sequentially:

```python
def apply_standard_filters(dataset, 
                          variance_percentile=25,
                          detection_rate=0.5,
                          min_expression=5.0,
                          remove_controls=True):
    """
    Apply standard filtering pipeline.
    
    Args:
        dataset: Dataset dictionary
        variance_percentile: Variance filter percentile
        detection_rate: Minimum detection rate
        min_expression: Minimum mean expression
        remove_controls: Whether to remove control probes
    
    Returns:
        Filtered dataset
    """
    original_n = len(dataset['keys'])
    
    print("=" * 60)
    print("APPLYING STANDARD FILTER PIPELINE")
    print("=" * 60)
    
    # 1. Remove control probes
    if remove_controls:
        print("\nStep 1: Control probe removal")
        dataset = filter_control_probes(dataset)
    
    # 2. Expression level filter
    print("\nStep 2: Expression level filter")
    dataset = filter_by_expression(dataset, min_expression)
    
    # 3. Detection rate filter
    print("\nStep 3: Detection rate filter")
    dataset = filter_by_detection(dataset, detection_rate)
    
    # 4. Variance filter
    print("\nStep 4: Variance filter")
    dataset = filter_by_variance(dataset, variance_percentile)
    
    final_n = len(dataset['keys'])
    reduction = (1 - final_n / original_n) * 100
    
    print("\n" + "=" * 60)
    print(f"FILTERING SUMMARY")
    print(f"  Original genes: {original_n:,}")
    print(f"  Final genes: {final_n:,}")
    print(f"  Reduction: {reduction:.1f}%")
    print("=" * 60)
    
    return dataset

# Example usage
filtered_dataset = apply_standard_filters(dataset)
```

## Advanced Filtering Strategies

### Fold-Change Pre-filtering

Filter based on minimum fold change in individual studies:

```python
def filter_by_fold_change(dataset, min_fc=1.5):
    """
    Keep genes with fold change ≥ threshold in any direction.
    
    Args:
        dataset: Dataset with class labels
        min_fc: Minimum fold change (linear scale)
    
    Returns:
        Filtered dataset
    """
    expr = dataset['expr']
    class_vector = dataset['class']
    
    # Calculate fold changes
    case_mean = expr.loc[:, class_vector == 1].mean(axis=1)
    control_mean = expr.loc[:, class_vector == 0].mean(axis=1)
    
    # For log2 data: log2(FC) = case_mean - control_mean
    # Convert to linear fold change: 2^(log2_fc)
    log2_fc = case_mean - control_mean
    fc = 2 ** np.abs(log2_fc)
    
    # Filter
    keep_mask = fc >= min_fc
    
    print(f"Fold-change filtering:")
    print(f"  Minimum FC: {min_fc}x")
    print(f"  Genes passing: {keep_mask.sum():,} ({keep_mask.sum()/len(fc)*100:.1f}%)")
    
    # Apply filter
    dataset['expr'] = expr[keep_mask]
    dataset['keys'] = [k for i, k in enumerate(dataset['keys']) if keep_mask.iloc[i]]
    
    return dataset
```

### Coefficient of Variation (CV) Filtering

Filter based on relative variability:

```python
def filter_by_cv(dataset, max_cv=2.0):
    """
    Remove genes with excessive coefficient of variation.
    
    CV = StdDev / Mean
    High CV indicates unstable measurements.
    
    Args:
        dataset: Dataset dictionary
        max_cv: Maximum allowed CV
    
    Returns:
        Filtered dataset
    """
    expr = dataset['expr']
    
    # Calculate CV
    gene_means = expr.mean(axis=1)
    gene_stds = expr.std(axis=1)
    cv = gene_stds / (gene_means + 0.01)  # Add small constant to avoid division by zero
    
    # Filter
    keep_mask = cv <= max_cv
    
    print(f"CV filtering:")
    print(f"  Max CV: {max_cv}")
    print(f"  Median CV: {cv.median():.3f}")
    print(f"  Genes retained: {keep_mask.sum():,} ({keep_mask.sum()/len(cv)*100:.1f}%)")
    
    # Apply filter
    dataset['expr'] = expr[keep_mask]
    dataset['keys'] = [k for i, k in enumerate(dataset['keys']) if keep_mask.iloc[i]]
    
    return dataset
```

### Biological Annotation Filtering

Filter based on gene function/type:

```python
def filter_by_annotation(dataset, 
                         exclude_noncoding=True,
                         exclude_pseudogenes=True,
                         exclude_deprecated=True):
    """
    Filter genes based on biological annotation.
    
    Args:
        dataset: Dataset with annotated genes
        exclude_noncoding: Remove non-coding RNAs
        exclude_pseudogenes: Remove pseudogenes
        exclude_deprecated: Remove deprecated symbols
    
    Returns:
        Filtered dataset
    """
    # Patterns to exclude
    exclude_patterns = []
    
    if exclude_noncoding:
        exclude_patterns.extend(['LINC', 'MIR', 'SNORD', 'SNORA'])
    
    if exclude_pseudogenes:
        exclude_patterns.extend(['LOC', 'P'])
    
    if exclude_deprecated:
        exclude_patterns.extend(['C\\d+orf', 'FLJ'])
    
    # Filter
    keep_mask = [
        not any(pattern in str(gene) for pattern in exclude_patterns)
        for gene in dataset['keys']
    ]
    
    n_removed = len(keep_mask) - sum(keep_mask)
    
    print(f"Annotation filtering:")
    print(f"  Genes removed: {n_removed:,}")
    print(f"  Genes retained: {sum(keep_mask):,}")
    
    # Apply filter
    dataset['expr'] = dataset['expr'].iloc[keep_mask]
    dataset['keys'] = [k for k, keep in zip(dataset['keys'], keep_mask) if keep]
    
    return dataset
```

### Custom Gene List Filtering

Filter to specific gene set (e.g., pathway genes):

```python
def filter_to_gene_list(dataset, gene_list):
    """
    Keep only genes in provided list.
    
    Args:
        dataset: Dataset dictionary
        gene_list: List/set of genes to keep
    
    Returns:
        Filtered dataset
    """
    gene_set = set(gene_list)
    
    # Filter
    keep_mask = [gene in gene_set for gene in dataset['keys']]
    
    n_found = sum(keep_mask)
    n_requested = len(gene_set)
    
    print(f"Gene list filtering:")
    print(f"  Requested genes: {n_requested:,}")
    print(f"  Found in dataset: {n_found:,} ({n_found/n_requested*100:.1f}%)")
    
    # Apply filter
    dataset['expr'] = dataset['expr'].iloc[keep_mask]
    dataset['keys'] = [k for k, keep in zip(dataset['keys'], keep_mask) if keep]
    
    return dataset

# Example: Filter to immune genes
immune_genes = ['CD4', 'CD8A', 'CD8B', 'IL2', 'IL6', 'TNF', 'IFNG']
filtered_dataset = filter_to_gene_list(dataset, immune_genes)
```

## FAQ

### Q: When should I filter genes?

**A:** Filter **after** data acquisition and QC, but **before** meta-analysis:

```
Download → QC → Annotation → **FILTER** → Meta-Analysis
```

### Q: How aggressive should filtering be?

**A:** Depends on goals:

- **Exploratory analysis**: Keep 50-75% of genes (moderate filtering)
- **Targeted analysis**: Keep 25-50% (aggressive filtering)
- **Pathway-specific**: Keep only genes in pathway (<10%)

### Q: Should I filter differently for each dataset?

**A:** **No!** Apply **same filters** to all datasets to avoid bias:

```python
# ❌ WRONG - different filters per dataset
for name, dataset in meta_object['originalData'].items():
    # Different thresholds per dataset
    filter_by_variance(dataset, percentile=random.randint(10, 50))

# ✅ CORRECT - consistent filtering
for name, dataset in meta_object['originalData'].items():
    # Same thresholds for all
    dataset = apply_standard_filters(dataset)
```

### Q: Can filtering introduce bias?

**A:** Yes, if done incorrectly:

**Biased filtering:**
- Using outcome information (case/control labels)
- Different filters per study
- Filtering after meta-analysis

**Unbiased filtering:**
- Based on technical quality only
- Same criteria for all studies
- Before statistical testing

### Q: How many genes should remain after filtering?

**A:** Typical ranges:

- Microarray meta-analysis: 5,000-15,000 genes
- RNA-seq meta-analysis: 10,000-20,000 genes
- Targeted analysis: 100-1,000 genes

## Troubleshooting

### Issue: Too few genes remaining

**Cause:** Overly aggressive filtering

**Solution:**
```python
# Relax thresholds
filtered_dataset = apply_standard_filters(
    dataset,
    variance_percentile=10,    # Keep top 90% (instead of 75%)
    detection_rate=0.3,        # Require 30% detection (instead of 50%)
    min_expression=4.0         # Lower expression threshold
)
```

### Issue: Genes still too noisy after filtering

**Cause:** Insufficient filtering

**Solution:**
```python
# More stringent filters
filtered_dataset = apply_standard_filters(
    dataset,
    variance_percentile=50,    # Keep only top 50%
    detection_rate=0.7,        # Require 70% detection
    min_expression=6.0         # Higher expression threshold
)
```

### Issue: Different gene counts across studies

**Cause:** Studies have different numbers of genes

**Solution:**
```python
# Standardize to common genes
filtered_meta = filter_by_study_presence(
    meta_object,
    min_studies=len(meta_object['originalData'])  # All studies
)

# Verify common genes
all_genes = [set(d['keys']) for d in filtered_meta['originalData'].values()]
common = set.intersection(*all_genes)
print(f"Common genes: {len(common):,}")
```

## Appendix

### Recommended Filtering Pipeline

```python
def recommended_filtering(meta_object):
    """
    Apply recommended filtering for typical meta-analysis.
    """
    print("Applying recommended filtering pipeline...")
    
    for name, dataset in meta_object['originalData'].items():
        print(f"\nDataset: {name}")
        
        # Standard filters
        dataset = filter_control_probes(dataset)
        dataset = filter_by_expression(dataset, min_expression=5.0)
        dataset = filter_by_detection(dataset, min_detection_rate=0.5)
        dataset = filter_by_variance(dataset, percentile=25)
        
        meta_object['originalData'][name] = dataset
    
    # Ensure common genes
    meta_object = filter_by_study_presence(
        meta_object,
        min_studies=len(meta_object['originalData'])
    )
    
    return meta_object
```

### Filter Performance Comparison

| Filter Type | Computation Time | Memory Usage | Reduction |
|-------------|------------------|--------------|-----------|
| Control probes | < 1 sec | Minimal | ~1-5% |
| Expression | < 5 sec | Minimal | ~20-40% |
| Detection | ~10 sec | Minimal | ~10-30% |
| Variance | ~10 sec | Minimal | ~25-50% |
| Combined | ~30 sec | Minimal | ~50-75% |

### Related Documentation

- [Statistical Methods](statistics.md) - Impact on testing
- [Best Practices](best_practices.md) - Recommendations
- [Tutorials](../tutorials/meta_analysis.md) - Practical examples

---

**Last updated:** 2026-04-21  
**Version:** 1.0  
**Maintainer:** MetaIntegrator-py Team
