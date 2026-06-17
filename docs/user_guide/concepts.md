# Core Concepts

This guide explains the fundamental concepts, terminology, and theoretical foundations of meta-analysis for gene expression data. Understanding these concepts will help you design better studies, interpret results correctly, and avoid common pitfalls.

## Contents

- [Introduction](#introduction)
- [What is Meta-Analysis?](#what-is-meta-analysis)
- [Key Terminology](#key-terminology)
- [Meta-Analysis Workflow](#meta-analysis-workflow)
- [Effect Sizes](#effect-sizes)
- [Statistical Methods](#statistical-methods)
- [Study Design Considerations](#study-design-considerations)
- [Advanced Topics](#advanced-topics)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Appendix](#appendix)

## Introduction

Meta-analysis combines results from multiple independent studies to identify consistent patterns that may not be evident in individual studies. In gene expression research, meta-analysis helps identify robust disease signatures by integrating data across different cohorts, platforms, and laboratories.

**Why meta-analysis matters:**
- Increases statistical power by combining sample sizes
- Identifies genes that are consistently differentially expressed
- Distinguishes true biological signals from study-specific artifacts
- Validates findings across heterogeneous populations and platforms

## What is Meta-Analysis?

### Definition

**Meta-analysis** is a statistical technique that combines quantitative results from multiple independent studies addressing the same research question. Unlike traditional analysis of a single dataset, meta-analysis synthesizes evidence across studies to produce a pooled estimate with greater precision.

### Types of Meta-Analysis

#### 1. Fixed-Effects Meta-Analysis
Assumes all studies estimate the same true effect size. Variability between studies is attributed solely to sampling error.

```python
from metaintegrator.core_analysis import MetaAnalysisCore

# Fixed-effects pooling
pooled_results = MetaAnalysisCore.pool_inverse_variance(
    g_matrix=effect_sizes,
    se_matrix=standard_errors,
    method='fixed'
)
```

**When to use:**
- Studies are very homogeneous (same population, platform, protocols)
- Low between-study heterogeneity (I² < 25%)

#### 2. Random-Effects Meta-Analysis
Assumes true effect sizes vary across studies due to real differences in populations or methods. Accounts for both within-study and between-study variability.

```python
# Random-effects pooling (recommended)
pooled_results = MetaAnalysisCore.pool_inverse_variance(
    g_matrix=effect_sizes,
    se_matrix=standard_errors,
    method='random'  # Default
)
```

**When to use:**
- Studies differ in populations, platforms, or designs (typical in gene expression)
- Moderate to high heterogeneity expected
- **Recommended for most meta-analyses**

### Gene Expression Meta-Analysis Specifics

Gene expression meta-analysis differs from traditional clinical meta-analysis:

1. **Massive multiple testing**: Thousands of genes tested simultaneously
2. **Platform heterogeneity**: Different microarray platforms, probe designs
3. **Batch effects**: Technical variation between laboratories
4. **Gene-level analysis**: Each gene analyzed separately, then results combined

## Key Terminology

### Study and Dataset Terms

| Term | Definition | Example |
|------|------------|---------|
| **Study** | Independent research investigation | "Breast cancer cohort from UCSF" |
| **Dataset** | Gene expression data from one study | GSE5847 from GEO |
| **Meta-object** | Data structure containing multiple datasets | `{'originalData': {...}}` |
| **Sample** | Individual measurement (patient, cell line) | One patient's tumor biopsy |
| **Gene/Probe** | Measured feature | BRCA1 gene or probe 1007_s_at |

### Statistical Terms

| Term | Definition | Formula/Notes |
|------|------------|---------------|
| **Effect Size** | Standardized measure of difference between groups | Hedges' g, Cohen's d |
| **Hedges' g** | Bias-corrected effect size | Small-sample correction to Cohen's d |
| **Standard Error (SE)** | Uncertainty in effect size estimate | SE = √(variance) |
| **P-value** | Probability of observing effect by chance | Lower = more significant |
| **FDR** | False Discovery Rate | Controls for multiple testing |
| **Heterogeneity** | Variability in effect sizes across studies | Measured by I², Q statistic |

### Meta-Analysis Specific Terms

| Term | Definition | Implementation |
|------|------------|----------------|
| **Pooling** | Combining effect sizes across studies | Inverse-variance weighting |
| **Fisher's Method** | Combining p-values across studies | Sum of log(p-values) |
| **I² Statistic** | Percentage of variability due to heterogeneity | 0-25% low, 25-75% moderate, >75% high |
| **Cochran's Q** | Test statistic for heterogeneity | Chi-squared distributed |
| **Tau² (τ²)** | Between-study variance | Random-effects parameter |

## Meta-Analysis Workflow

### Standard Pipeline

```
┌─────────────────────┐
│  1. Data Acquisition │  ← Download from GEO
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  2. Preprocessing   │  ← Normalize, log-transform
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  3. Quality Control │  ← Check data quality
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  4. Gene Annotation │  ← Map to standard symbols
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  5. Gene Filtering  │  ← Remove low-quality genes
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  6. Effect Size     │  ← Calculate per-study effects
│     Calculation     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  7. Meta-Analysis   │  ← Pool across studies
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  8. Multiple Testing│  ← FDR correction
│     Correction      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  9. Visualization   │  ← Plots and figures
└─────────────────────┘
```

### Implementation

```python
from preprocessing.geo_data_loader import get_geo_data
from preprocessing.geo_annotation_engine import annotate_geo_dataset, GeneAnnotationEngine
from metaintegrator.analysis import MetaAnalyzer
import numpy as np

# 1. Data Acquisition
meta_object = get_geo_data(
    gse_vector=["GSE5847", "GSE28750"],
    q_norm=True
)

# 2-3. QC happens during acquisition

# 4. Gene Annotation
engine = GeneAnnotationEngine()
for name, dataset in meta_object['originalData'].items():
    meta_object['originalData'][name] = annotate_geo_dataset(dataset, engine)

# 5. Gene Filtering (optional, shown in filtering guide)

# 6. Assign class labels
for dataset in meta_object['originalData'].values():
    # Extract from phenotype data in real analysis
    n = dataset['expr'].shape[1]
    dataset['class'] = np.concatenate([np.zeros(n//2), np.ones(n-n//2)])

# 7-8. Meta-Analysis with FDR correction
analyzer = MetaAnalyzer()
results = analyzer.run_meta_analysis_core(
    list(meta_object['originalData'].values())
)

# 9. Visualization (see visualization tutorial)
```

## Effect Sizes

### What is an Effect Size?

An **effect size** quantifies the magnitude of difference between two groups (cases vs. controls). Unlike p-values, which only indicate whether an effect exists, effect sizes tell you **how large** the effect is.

### Hedges' g

MetaIntegrator uses **Hedges' g**, a bias-corrected version of Cohen's d:

```
g = J × d

where:
  d = (mean_cases - mean_controls) / pooled_SD
  J = correction factor for small samples
  pooled_SD = √[((n₁-1)×SD₁² + (n₂-1)×SD₂²) / (n₁+n₂-2)]
```

### Interpreting Effect Sizes

| Hedges' g | Interpretation | Example |
|-----------|----------------|---------|
| \|g\| < 0.2 | Negligible effect | Minimal biological relevance |
| 0.2 ≤ \|g\| < 0.5 | Small effect | Subtle gene expression change |
| 0.5 ≤ \|g\| < 0.8 | Medium effect | Moderate dysregulation |
| \|g\| ≥ 0.8 | Large effect | Dramatic change in expression |

**Positive g**: Gene up-regulated in cases  
**Negative g**: Gene down-regulated in cases

### Computing Effect Sizes

```python
from metaintegrator.core_analysis import MetaAnalysisCore

# For a single gene
effect_stats = MetaAnalysisCore.get_effect_size(
    expression=gene_expr_vector,  # Expression values
    groups=class_vector,           # 0=control, 1=case
    method='hedges_g'
)

print(f"Effect size: {effect_stats['g']:.3f}")
print(f"Standard error: {effect_stats['se_g']:.3f}")
print(f"Sample sizes: {effect_stats['n1']} cases, {effect_stats['n2']} controls")

# For all genes in a study
all_effects = MetaAnalysisCore.calculate_effect_sizes(dataset)
```

## Statistical Methods

### Effect Size Pooling

**Inverse-variance weighting**: Studies with more precise estimates (smaller SE) receive more weight:

```
Weight_i = 1 / SE_i²

Pooled effect = Σ(Weight_i × Effect_i) / Σ(Weight_i)
```

### P-value Combination: Fisher's Method

Combines p-values across studies using the sum of log-transformed p-values:

```
χ² = -2 × Σ log(p_i)

df = 2k  (k = number of studies)
```

```python
from metaintegrator.core_analysis import FisherCombination

# Calculate p-values for each study
p_value_list = []
for dataset in meta_object['originalData'].values():
    p_vals = FisherCombination.calculate_study_pvalues(dataset)
    p_value_list.append(p_vals)

# Combine p-values
combined = FisherCombination.sum_of_logs(p_value_list)
```

### Multiple Testing Correction

With 20,000+ genes, many will appear significant by chance. **False Discovery Rate (FDR)** control using Benjamini-Hochberg:

```
For genes ranked by p-value (p₁ ≤ p₂ ≤ ... ≤ pₘ):
  FDR_i = p_i × m / i
```

**Recommended threshold**: FDR < 0.05 (5% false discoveries expected)

```python
# FDR correction applied automatically
pooled = results['pooled_estimates']

# Filter significant genes
sig_genes = pooled[pooled['p_fdr'] < 0.05]
```

## Study Design Considerations

### Number of Studies

**Minimum**: 2 studies (but power is limited)  
**Recommended**: 3-5 studies for robust results  
**Optimal**: 5-10 studies balances power and feasibility

### Sample Size Requirements

**Per study minimum**: 10 samples per group (20 total)  
**Recommended**: 20+ samples per group  
**Optimal**: 30+ samples per group

**Total samples** (across all studies):
- Small meta-analysis: 100-200 samples
- Medium: 200-500 samples
- Large: 500+ samples

### Platform Compatibility

**Best**: All studies use same platform (e.g., all Affymetrix GPL570)  
**Good**: Compatible platforms (e.g., all Affymetrix or all Illumina)  
**Problematic**: Mix of very different technologies

```python
# Check platform compatibility
platforms = []
for name, dataset in meta_object['originalData'].items():
    platform = dataset.get('platform', 'Unknown')
    platforms.append((name, platform))
    print(f"{name}: {platform}")

# Look for inconsistencies
unique_platforms = set([p[1] for p in platforms])
if len(unique_platforms) > 1:
    print(f"⚠️ Multiple platforms detected: {unique_platforms}")
```

## Advanced Topics

### Handling Heterogeneity

High heterogeneity (I² > 75%) suggests studies measure different effects:

**Options:**
1. **Use random-effects model** (default, accounts for heterogeneity)
2. **Subgroup analysis** (e.g., separate by disease subtype)
3. **Meta-regression** (model heterogeneity sources)
4. **Exclude outlier studies** (if justified)

```python
# Check heterogeneity statistics
pooled = results['pooled_estimates']

if 'pval_het' in pooled.columns:
    high_het = pooled[pooled['pval_het'] < 0.05]
    print(f"Genes with significant heterogeneity: {len(high_het)}")
```

### Leave-One-Out Analysis

Test robustness by excluding each study sequentially:

```python
# Leave-one-out sensitivity analysis
for i, dataset_name in enumerate(meta_object['originalData'].keys()):
    # Create subset excluding one study
    datasets_loo = [
        d for j, d in enumerate(meta_object['originalData'].values()) 
        if j != i
    ]
    
    # Run meta-analysis on subset
    results_loo = analyzer.run_meta_analysis_core(datasets_loo)
    
    print(f"Excluding {dataset_name}:")
    print(f"  Significant genes: {(results_loo['pooled_estimates']['p_fdr'] < 0.05).sum()}")
```

### Publication Bias

Funnel plots can detect publication bias (small negative studies missing):

```python
import matplotlib.pyplot as plt

# Extract effect sizes and standard errors
effects = pooled['effect_size']
se = pooled['se_summary']

# Create funnel plot
fig, ax = plt.subplots()
ax.scatter(effects, 1/se, alpha=0.5)
ax.axvline(x=0, color='red', linestyle='--')
ax.set_xlabel('Effect Size')
ax.set_ylabel('Precision (1/SE)')
ax.set_title('Funnel Plot')
plt.show()

# Symmetric distribution suggests no bias
# Asymmetry suggests publication bias
```

## FAQ

### Q: How many studies do I need for meta-analysis?

**A:** Minimum 2, but 3-5 is recommended for robust results. With only 2 studies, heterogeneity cannot be reliably estimated. More studies increase power but also complexity.

### Q: Can I combine data from different platforms?

**A:** Yes, but with caution. After gene annotation to common symbols and appropriate normalization, different platforms can be combined. Ensure:
- Good gene overlap (>80%)
- Similar data quality
- Both are log-transformed
- Consider batch correction if needed

### Q: What's the difference between fixed and random effects?

**A:** 
- **Fixed-effects**: Assumes all studies estimate the same true effect. Use when studies are very similar.
- **Random-effects** (recommended): Assumes effect sizes vary across studies. More conservative and robust to heterogeneity.

### Q: Why use Hedges' g instead of fold change?

**A:** 
- **Fold change** doesn't account for sample size or variability
- **Hedges' g** is standardized (comparable across studies)
- **g** has known statistical properties for meta-analysis
- **g** incorporates uncertainty (standard error)

### Q: How do I handle genes with multiple probes?

**A:** MetaIntegrator automatically handles this by:
1. Calculating effect sizes for each probe
2. Summarizing multiple probes per gene (using inverse-variance weighting or selecting most significant)
3. Final meta-analysis uses one effect size per gene

```python
# Automatic probe summarization
summarized = MetaAnalysisCore.summarize_effects_within_study(
    effects=probe_level_effects,
    method='fixed_iv'  # or 'extreme' for max |z-score|
)
```

## Troubleshooting

### Issue: "Very few significant genes found"

**Solutions:**
1. Check sample sizes (need adequate power)
2. Verify class labels are correct
3. Ensure data is in log scale
4. Check gene filtering (may be too aggressive)
5. Verify studies measure similar phenotypes

```python
# Diagnostic checks
for name, dataset in meta_object['originalData'].items():
    print(f"\n{name}:")
    print(f"  Samples: {dataset['expr'].shape[1]}")
    print(f"  Genes: {len(dataset['keys'])}")
    print(f"  Class distribution: {np.bincount(dataset['class'])}")
```

### Issue: "Heterogeneity is very high (I² > 90%)"

**Solutions:**
1. Use random-effects model (default)
2. Investigate sources of heterogeneity
3. Consider subgroup analysis
4. Check for outlier studies
5. Verify all studies measure same phenotype

### Issue: "No common genes across studies"

**Solutions:**
1. Check gene annotation was applied
2. Verify all datasets are from same organism
3. Check platform compatibility
4. Re-run annotation with `enable_remote=True`

```python
# Check gene overlap
all_genes = [set(d['keys']) for d in meta_object['originalData'].values()]
common = set.intersection(*all_genes)
print(f"Common genes: {len(common)}")

if len(common) < 1000:
    print("⚠️ Poor gene overlap - check annotation")
```

## Appendix

### Glossary

**Meta-analysis**: Statistical synthesis of multiple studies  
**Effect size**: Standardized measure of group difference  
**Hedges' g**: Bias-corrected standardized mean difference  
**Heterogeneity**: Variability in effects across studies  
**I² statistic**: Proportion of variance due to heterogeneity  
**FDR**: False Discovery Rate (multiple testing correction)  
**Fisher's method**: P-value combination technique  
**Pooling**: Combining estimates across studies  

### Related Documentation

- [Data Structures Guide](data_structures.md) - Meta-object format
- [Statistical Methods](statistics.md) - Detailed formulas
- [GEO Integration](geo_integration.md) - Data acquisition
- [Best Practices](best_practices.md) - Recommendations
- [API Reference](../api/overview.md) - Function documentation

### Further Reading

**Papers:**
- Tseng GC, et al. (2012) "Comprehensive literature review and statistical considerations for microarray meta-analysis" *Nucleic Acids Research*
- Ramasamy A, et al. (2008) "Key issues in conducting a meta-analysis of gene expression microarray datasets" *PLoS Medicine*

**Books:**
- Borenstein M, et al. (2009) "Introduction to Meta-Analysis" Wiley
- Cooper H, et al. (2019) "The Handbook of Research Synthesis and Meta-Analysis" Russell Sage Foundation

### Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| g | Hedges' g (effect size) |
| SE | Standard error |
| τ² | Between-study variance (tau-squared) |
| I² | Heterogeneity percentage |
| Q | Cochran's Q statistic |
| df | Degrees of freedom |
| p | P-value |
| FDR | False discovery rate |
| n₁, n₂ | Sample sizes for groups 1 and 2 |

---

**Last updated:** 2026-04-21  
**Version:** 1.0  
**Maintainer:** MetaIntegrator-py Team

