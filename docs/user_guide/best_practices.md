# Best Practices

This guide provides recommendations, tips, and best practices for conducting robust and reproducible meta-analysis of gene expression data using metaIntegrator-py.

## Contents

- [Introduction](#introduction)
- [Study Design](#study-design)
- [Data Acquisition](#data-acquisition)
- [Data Quality](#data-quality)
- [Gene Filtering](#gene-filtering)
- [Statistical Analysis](#statistical-analysis)
- [Result Interpretation](#result-interpretation)
- [Reproducibility](#reproducibility)
- [Common Pitfalls](#common-pitfalls)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Appendix](#appendix)

## Introduction

Meta-analysis is powerful but requires careful execution. Following best practices ensures your results are robust, reproducible, and publishable. This guide distills lessons learned from hundreds of meta-analyses into actionable recommendations.

**Core principles:**
1. **Design carefully** - Plan before analyzing
2. **Document everything** - Enable reproducibility
3. **Validate rigorously** - Trust but verify
4. **Communicate clearly** - Share methods and code

## Study Design

### Define Research Question First

**Before downloading data**, clearly define:

```python
# Document your research question
research_plan = {
    'question': "Which genes are consistently differentially expressed in breast cancer?",
    'population': "Human breast cancer patients vs. healthy controls",
    'outcome': "Differentially expressed genes (FDR < 0.05)",
    'platforms': "Affymetrix microarrays (GPL570, GPL96)",
    'min_studies': 3,
    'min_samples_per_study': 20,
    'exclusion_criteria': [
        "Studies with < 10 samples per group",
        "Studies without clear case/control labels",
        "Cell line studies (not primary tissue)"
    ]
}

# Save for documentation
import json
with open('research_plan.json', 'w') as f:
    json.dump(research_plan, f, indent=2)
```

**✅ Good research questions:**
- "Which genes differentiate TB from bacterial pneumonia?"
- "What is the immune signature of sepsis?"

**❌ Poor research questions:**
- "Find interesting genes" (too vague)
- "Compare all diseases" (too broad)

### Sample Size Planning

**Recommended minimum:**

| Analysis Type | Min Studies | Min Total Samples | Min Per Group |
|---------------|-------------|-------------------|---------------|
| Exploratory | 2-3 | 100 | 10 |
| Standard | 3-5 | 200 | 20 |
| Robust | 5-10 | 500 | 30 |
| Large-scale | 10+ | 1000+ | 50 |

```python
def assess_power(n_studies, samples_per_study, effect_size=0.5):
    """
    Rough power calculation for meta-analysis.
    
    Args:
        n_studies: Number of studies
        samples_per_study: Average samples per study
        effect_size: Expected Hedges' g
    
    Returns:
        Estimated power
    """
    total_n = n_studies * samples_per_study
    
    # Simplified power calculation
    if total_n < 100:
        power = "Low (~60%)"
    elif total_n < 300:
        power = "Medium (~80%)"
    else:
        power = "High (>90%)"
    
    print(f"Meta-analysis power estimate:")
    print(f"  Studies: {n_studies}")
    print(f"  Total samples: {total_n}")
    print(f"  Expected effect: {effect_size}")
    print(f"  Power: {power}")
    
    return power

# Example
assess_power(n_studies=5, samples_per_study=40, effect_size=0.5)
```

### Platform Compatibility

**✅ Best: Homogeneous platforms**
```python
# All same platform - ideal
gse_ids = ["GSE5847", "GSE28750", "GSE40012"]
platforms = ["GPL570", "GPL570", "GPL570"]  # All Affymetrix HG-U133 Plus 2.0
```

**⚠️ Acceptable: Compatible platforms**
```python
# Similar technology - acceptable with caution
platforms = ["GPL570", "GPL571", "GPL96"]  # All Affymetrix, different chips
```

**❌ Problematic: Very different platforms**
```python
# Mixed technologies - high risk of batch effects
platforms = ["GPL570", "GPL6244", "GPL10558"]  # Affymetrix + Illumina
```

## Data Acquisition

### Use Systematic Search Strategy

```python
def systematic_geo_search(disease, organism='Homo sapiens'):
    """
    Document systematic search for GEO datasets.
    """
    search_log = {
        'date': '2026-04-21',
        'disease': disease,
        'organism': organism,
        'search_terms': [
            f"{disease} AND {organism}",
            f"{disease} AND expression profiling by array"
        ],
        'databases': ['GEO', 'ArrayExpress'],
        'filters': {
            'study_type': 'Expression profiling by array',
            'organism': organism,
            'sample_count': '> 20'
        }
    }
    
    # Save search record
    with open('search_log.json', 'w') as f:
        json.dump(search_log, f, indent=2)
    
    return search_log

# Document search
search_log = systematic_geo_search('breast cancer')
```

### Validate All Downloads

```python
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.labeling import assign_class_labels

def download_with_validation(gse_ids):
    """
    Download GEO data with comprehensive validation. There's no built-in
    check_geo_log_scale() — this uses a simple max-value heuristic instead
    (log2 microarray data is rarely above ~25-30).
    """
    loader = DataLoaderFactory.create_geo_loader(strategy="r_parity", cache_dir="./geo_cache")
    raw_datasets = loader.load_multiple(gse_ids, stop_on_error=False)

    meta_object = {"originalData": {}}
    validation_log = []

    for ds in raw_datasets:
        is_log_scale = ds.expr.max().max() < 30
        checks = {
            'dataset': ds.geo_id,
            'downloaded': True,
            'n_genes': len(ds.keys),
            'n_samples': ds.expr.shape[1],
            'log_scale': is_log_scale,
            'missing_pct': (ds.expr.isna().sum().sum() / ds.expr.size) * 100,
            'platform': ds.metadata.get('platform', 'Unknown'),
        }

        # Flag issues
        if checks['n_samples'] < 20:
            checks['warning'] = 'Low sample size'
        if checks['missing_pct'] > 5:
            checks['warning'] = 'High missing data'
        if not checks['log_scale']:
            checks['warning'] = 'Not in log scale'

        validation_log.append(checks)

        dataset = {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
                   "formattedName": ds.geo_id, "class": None}
        meta_object["originalData"][ds.geo_id] = assign_class_labels(
            dataset, column_name="title", disease_terms=["disease"]
        )

    # Save validation log
    pd.DataFrame(validation_log).to_csv('download_validation.csv', index=False)

    return meta_object

# Use it
meta_object = download_with_validation(["GSE5847", "GSE28750"])
```

## Data Quality

### Comprehensive QC Checklist

```python
def comprehensive_qc(meta_object):
    """
    Perform comprehensive quality control.
    """
    qc_report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'datasets': {}
    }
    
    for name, dataset in meta_object['originalData'].items():
        checks = {}
        
        # 1. Data structure
        checks['has_expr'] = 'expr' in dataset
        checks['has_class'] = 'class' in dataset
        checks['has_keys'] = 'keys' in dataset
        
        # 2. Dimensions match
        checks['dims_match'] = len(dataset['keys']) == dataset['expr'].shape[0]
        checks['class_match'] = len(dataset['class']) == dataset['expr'].shape[1]
        
        # 3. Data quality
        expr = dataset['expr']
        checks['missing_pct'] = (expr.isna().sum().sum() / expr.size) * 100
        checks['log_scale'] = expr.max().max() < 30  # heuristic; no built-in check
        
        # 4. Sample size
        checks['n_samples'] = expr.shape[1]
        checks['adequate_size'] = checks['n_samples'] >= 20
        
        # 5. Class balance
        if 'class' in dataset:
            unique, counts = np.unique(dataset['class'], return_counts=True)
            balance = min(counts) / max(counts) if len(counts) == 2 else 0
            checks['class_balance'] = balance
            checks['balanced'] = balance >= 0.3
        
        # 6. Gene annotation
        has_symbols = not any('_at' in str(k) for k in dataset['keys'][:10])
        checks['annotated'] = has_symbols
        
        qc_report['datasets'][name] = checks
    
    # Save report
    with open('qc_report.json', 'w') as f:
        json.dump(qc_report, f, indent=2)
    
    return qc_report

# Run QC
qc_report = comprehensive_qc(meta_object)
```

### QC Thresholds

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Missing data | < 1% | 1-5% | > 5% |
| Class balance | ≥ 0.8 | 0.3-0.8 | < 0.3 |
| Gene overlap | > 80% | 50-80% | < 50% |
| Sample size/group | ≥ 30 | 10-30 | < 10 |
| Annotation rate | > 90% | 80-90% | < 80% |

## Gene Filtering

### Standard Filtering Pipeline

```python
def standard_filtering_pipeline(meta_object):
    """
    Apply recommended filtering consistently.
    """
    print("STANDARD FILTERING PIPELINE")
    print("=" * 60)
    
    # Filter each dataset identically
    for name, dataset in meta_object['originalData'].items():
        print(f"\nDataset: {name}")
        original_n = len(dataset['keys'])
        
        # 1. Remove control probes
        dataset = filter_control_probes(dataset)
        
        # 2. Expression level filter (log2 > 5)
        dataset = filter_by_expression(dataset, min_expression=5.0)
        
        # 3. Detection filter (present in 50% of samples)
        dataset = filter_by_detection(dataset, min_detection_rate=0.5)
        
        # 4. Variance filter (top 75%)
        dataset = filter_by_variance(dataset, percentile=25)
        
        final_n = len(dataset['keys'])
        print(f"  Reduced from {original_n:,} to {final_n:,} genes ({final_n/original_n*100:.1f}%)")
        
        meta_object['originalData'][name] = dataset
    
    # 5. Cross-study filter (genes in all studies)
    print("\nCross-study filtering...")
    meta_object = filter_by_study_presence(
        meta_object,
        min_studies=len(meta_object['originalData'])
    )
    
    return meta_object
```

**✅ DO:**
- Apply same filters to all studies
- Filter before meta-analysis
- Document filter parameters
- Check gene overlap after filtering

**❌ DON'T:**
- Use different filters per study
- Filter after meta-analysis
- Use outcome information for filtering
- Remove genes based on individual study p-values

## Statistical Analysis

### Pre-Register Analysis Plan

```python
# Document analysis plan before running
analysis_plan = {
    'meta_analysis_method': 'random_effects',
    'effect_size_metric': 'hedges_g',
    'p_value_method': 'fisher',
    'multiple_testing_correction': 'benjamini_hochberg',
    'fdr_threshold': 0.05,
    'heterogeneity_assessment': 'I2_and_Q',
    'sensitivity_analyses': [
        'leave_one_out',
        'subgroup_by_platform'
    ]
}

# Save before analysis
with open('analysis_plan.json', 'w') as f:
    json.dump(analysis_plan, f, indent=2)
```

### Use Random Effects by Default

```python
from metaintegrator.analysis import MetaAnalyzer

# ✅ RECOMMENDED: run_meta_analysis() — always uses random-effects pooling
# and also runs leave-one-out automatically (3+ datasets)
analyzer = MetaAnalyzer()
results = analyzer.run_meta_analysis(meta_object)

# There is no "fixed-effects" toggle on run_meta_analysis() itself — the
# lower-level MetaAnalysisCore.pool_inverse_variance(method=...) does
# support 'fixed', but the recommended, R-parity entry point is random-effects.
```

### Check Assumptions

```python
def check_meta_analysis_assumptions(results):
    """
    Verify meta-analysis assumptions using the primary pooledResults output.
    """
    pooled = results['metaAnalysis']['pooledResults']

    print("META-ANALYSIS ASSUMPTION CHECKS")
    print("=" * 60)

    # 1. Check heterogeneity (I² is derived, not a stored column:
    #    I² = max(0, 100% * (Q - df) / Q); df = numStudies - 1)
    het_pval_median = pooled['heterogeneityPval'].median()
    print(f"\n1. Heterogeneity:")
    print(f"   Median heterogeneity p-value: {het_pval_median:.3f}")
    if het_pval_median >= 0.05:
        print("   ✅ Low-moderate heterogeneity (random-effects appropriate)")
    else:
        print("   ⚠️ Significant heterogeneity for many genes (investigate sources)")

    # 2. Check sample sizes
    n_studies = pooled['numStudies'].median()
    print(f"\n2. Study participation:")
    print(f"   Median studies per gene: {n_studies:.0f}")
    if n_studies >= 3:
        print("   ✅ Adequate study numbers")
    else:
        print("   ⚠️ Low study numbers (results may be unstable)")

    # 3. Check effect size distribution
    effects = pooled['effectSize']
    print(f"\n3. Effect size distribution:")
    print(f"   Mean: {effects.mean():.3f}")
    print(f"   Median: {effects.median():.3f}")
    print(f"   Range: [{effects.min():.3f}, {effects.max():.3f}]")

    # 4. Check p-value distribution
    p_values = pooled['effectSizePval']
    uniform_expected = len(p_values) * 0.05
    observed_p05 = (p_values < 0.05).sum()
    print(f"\n4. P-value distribution:")
    print(f"   Expected < 0.05 (uniform): {uniform_expected:.0f}")
    print(f"   Observed < 0.05: {observed_p05}")
    if observed_p05 > uniform_expected * 2:
        print("   ✅ Enrichment suggests true signal")
    else:
        print("   ⚠️ No enrichment (may lack power)")

# Check assumptions
check_meta_analysis_assumptions(results)
```

## Result Interpretation

### Multiple Comparisons Context

```python
def interpret_significance(results, fdr_threshold=0.05):
    """
    Interpret results in context of multiple testing.
    """
    pooled = results['metaAnalysis']['pooledResults']

    # Count significance levels
    sig_fdr = (pooled['effectSizeFDR'] < fdr_threshold).sum()
    sig_nominal = (pooled['effectSizePval'] < 0.05).sum()
    total = len(pooled)
    
    print("SIGNIFICANCE INTERPRETATION")
    print("=" * 60)
    print(f"Total genes tested: {total:,}")
    print(f"\nSignificant genes (FDR < {fdr_threshold}):")
    print(f"  Count: {sig_fdr:,} ({sig_fdr/total*100:.2f}%)")
    print(f"  Expected by chance: ~{total*fdr_threshold:.0f}")
    print(f"  Enrichment: {sig_fdr/(total*fdr_threshold):.1f}x")
    
    print(f"\nNominal significance (p < 0.05):")
    print(f"  Count: {sig_nominal:,} ({sig_nominal/total*100:.2f}%)")
    print(f"  Expected by chance: ~{total*0.05:.0f}")
    print(f"  Note: {sig_nominal - sig_fdr:,} do not survive FDR correction")
    
    # Effect size context
    sig_genes = pooled[pooled['effectSizeFDR'] < fdr_threshold]
    if len(sig_genes) > 0:
        large_effect = (np.abs(sig_genes['effectSize']) > 0.8).sum()
        medium_effect = ((np.abs(sig_genes['effectSize']) > 0.5) &
                        (np.abs(sig_genes['effectSize']) <= 0.8)).sum()
        
        print(f"\nEffect sizes of significant genes:")
        print(f"  Large (|g| > 0.8): {large_effect} ({large_effect/len(sig_genes)*100:.1f}%)")
        print(f"  Medium (0.5 < |g| ≤ 0.8): {medium_effect} ({medium_effect/len(sig_genes)*100:.1f}%)")

interpret_significance(results)
```

### Validate Top Genes

```python
def validate_top_genes(meta_object, results, n_genes=20):
    """
    Check consistency of top genes across studies.
    """
    pooled = results['metaAnalysis']['pooledResults']
    top_genes = pooled.nsmallest(n_genes, 'effectSizeFDR')

    print(f"TOP {n_genes} GENE VALIDATION")
    print("=" * 60)

    for gene in top_genes.index:
        print(f"\n{gene}:")
        print(f"  Pooled ES: {top_genes.loc[gene, 'effectSize']:.3f}")
        print(f"  FDR: {top_genes.loc[gene, 'effectSizeFDR']:.2e}")
        
        # Check per-study consistency
        directions = []
        for name, dataset in meta_object['originalData'].items():
            if gene in dataset['keys']:
                gene_idx = dataset['keys'].index(gene)
                expr = dataset['expr'].iloc[gene_idx]
                class_vec = dataset['class']
                
                case_mean = expr[class_vec == 1].mean()
                ctrl_mean = expr[class_vec == 0].mean()
                direction = "↑" if case_mean > ctrl_mean else "↓"
                directions.append(direction)
                
                print(f"    {name}: {direction} (Δ={case_mean-ctrl_mean:.2f})")
        
        # Check consistency
        if len(set(directions)) == 1:
            print(f"  ✅ Consistent direction across all studies")
        else:
            print(f"  ⚠️ Inconsistent direction (heterogeneity!)")

validate_top_genes(meta_object, results, n_genes=10)
```

## Reproducibility

### Version Control Everything

```python
# Save package versions
def save_environment():
    """Document computational environment."""
    import sys
    import pandas as pd
    import numpy as np
    import scipy
    
    env_info = {
        'date': pd.Timestamp.now().isoformat(),
        'python_version': sys.version,
        'packages': {
            'metaintegrator-py': '1.0.0',  # Your package version
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'scipy': scipy.__version__
        },
        'system': {
            'platform': sys.platform,
            'machine': platform.machine()
        }
    }
    
    with open('environment.json', 'w') as f:
        json.dump(env_info, f, indent=2)
    
    return env_info

# Document environment
save_environment()
```

### Save Complete Workflow

```python
def save_complete_workflow(meta_object, results):
    """
    Save everything for reproducibility.
    """
    from pathlib import Path
    import pickle
    
    # Create output directory
    output_dir = Path('./reproducibility_package')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Save meta-object
    with open(output_dir / 'meta_object.pkl', 'wb') as f:
        pickle.dump(meta_object, f)
    
    # 2. Save results
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # 3. Save significant genes
    pooled = results['metaAnalysis']['pooledResults']
    sig_genes = pooled[pooled['effectSizeFDR'] < 0.05]
    sig_genes.to_csv(output_dir / 'significant_genes.csv')
    
    # 4. Save session info
    save_environment()
    
    # 5. Create README
    readme = """
# Meta-Analysis Reproducibility Package

## Contents
- meta_object.pkl: Complete dataset collection
- results.pkl: Meta-analysis results
- significant_genes.csv: Genes passing FDR < 0.05
- environment.json: Computational environment

## Reproduce
```python
import pickle
with open('meta_object.pkl', 'rb') as f:
    meta_object = pickle.load(f)
with open('results.pkl', 'rb') as f:
    results = pickle.load(f)
```
"""
    
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme)
    
    print(f"✅ Reproducibility package saved to: {output_dir}")

# Save everything
save_complete_workflow(meta_object, results)
```

## Publication

### Reporting Checklist

```markdown
# Meta-Analysis Reporting Checklist

## Methods Section Must Include:

### Data Sources
- [ ] Search strategy described
- [ ] Inclusion/exclusion criteria defined
- [ ] Number of studies screened
- [ ] Number of studies included
- [ ] Reasons for exclusions documented

### Data Extraction
- [ ] Platform information reported
- [ ] Sample sizes reported
- [ ] Quality control steps described
- [ ] Gene filtering methods explained

### Statistical Analysis
- [ ] Effect size metric specified (Hedges' g)
- [ ] Meta-analysis model stated (random/fixed)
- [ ] P-value combination method (Fisher's)
- [ ] Multiple testing correction (FDR)
- [ ] Heterogeneity assessment (I², Q)
- [ ] Software and version reported

### Sensitivity Analyses
- [ ] Leave-one-out analysis
- [ ] Subgroup analyses (if applicable)
- [ ] Publication bias assessment

## Results Section Must Include:

- [ ] PRISMA flow diagram
- [ ] Study characteristics table
- [ ] Forest plot for top genes
- [ ] Heterogeneity statistics
- [ ] Number of significant genes
- [ ] Effect size distributions

## Supplementary Materials

- [ ] Complete gene list with statistics
- [ ] Per-study results
- [ ] Analysis code (GitHub/Zenodo)
- [ ] Raw data availability statement
```

### Create Figure Panel

```python
def create_publication_figure(results):
    """
    Create multi-panel figure for publication.
    """
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    pooled = results['metaAnalysis']['pooledResults']

    # Panel A: Volcano plot
    ax1 = fig.add_subplot(gs[0, :])
    # ... volcano plot code ...
    ax1.set_title('A. Volcano Plot', loc='left', fontweight='bold')
    
    # Panel B: Effect size distribution
    ax2 = fig.add_subplot(gs[1, 0])
    # ... distribution code ...
    ax2.set_title('B. Effect Size Distribution', loc='left', fontweight='bold')
    
    # Panel C: Heterogeneity
    ax3 = fig.add_subplot(gs[1, 1])
    # ... heterogeneity plot ...
    ax3.set_title('C. Heterogeneity', loc='left', fontweight='bold')
    
    # Panel D: Top genes
    ax4 = fig.add_subplot(gs[2, :])
    # ... top genes forest plot ...
    ax4.set_title('D. Top Genes', loc='left', fontweight='bold')
    
    plt.savefig('figure_1.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_1.png', dpi=300, bbox_inches='tight')
    
    print("✅ Publication figure saved")
```

## Common Pitfalls

### 1. P-Hacking

**❌ DON'T:**
```python
# Trying multiple thresholds until results look good
pooled = results['metaAnalysis']['pooledResults']
for threshold in [0.05, 0.10, 0.15, 0.20]:
    sig = pooled[pooled['effectSizeFDR'] < threshold]
    if len(sig) > 100:
        print(f"Using FDR < {threshold}")  # Cherry-picking!
        break
```

**✅ DO:**
```python
# Pre-specify threshold
FDR_THRESHOLD = 0.05  # Documented in analysis plan
pooled = results['metaAnalysis']['pooledResults']
sig_genes = pooled[pooled['effectSizeFDR'] < FDR_THRESHOLD]
```

### 2. HARKing (Hypothesizing After Results Known)

**❌ DON'T:**
```python
# Looking at results, then claiming you predicted them
results = analyzer.run_meta_analysis(meta_object)
# "We hypothesized gene X would be significant" (but you didn't!)
```

**✅ DO:**
```python
# Document hypothesis before analysis
hypothesis = "Gene X will be up-regulated based on prior literature"
# Save to analysis_plan.json BEFORE running analysis
```

### 3. Ignoring Heterogeneity

**❌ DON'T:**
```python
# run_meta_analysis() always uses random-effects pooling; deliberately
# reaching for the lower-level fixed-effects pooling despite known high
# heterogeneity, and not checking it, is the mistake here
from metaintegrator.core_analysis import MetaAnalysisCore
pooled_fixed = MetaAnalysisCore.pool_inverse_variance(g_matrix, se_matrix, method='fixed')  # Ignores heterogeneity!
```

**✅ DO:**
```python
# Use the recommended entry point (random-effects) and check heterogeneity
results = analyzer.run_meta_analysis(meta_object)
pooled = results['metaAnalysis']['pooledResults']
high_het = pooled[pooled['heterogeneityPval'] < 0.05]
print(f"{len(high_het)} genes with significant heterogeneity — investigate these")
```

### 4. Selective Reporting

**❌ DON'T:**
```python
# Only reporting "interesting" results
interesting_genes = ['BRCA1', 'TP53', 'EGFR']
report_only(pooled[pooled.index.isin(interesting_genes)])
```

**✅ DO:**
```python
# Report all significant genes + complete results in supplement
significant = pooled[pooled['effectSizeFDR'] < 0.05]
report_all(significant)
save_supplement(pooled)  # Full results
```

## FAQ

### Q: How many studies is enough?

**A:** Depends on goal:
- **Minimum**: 2 (but limited power)
- **Recommended**: 3-5
- **Optimal**: 5-10
- **Large-scale**: 10+

More studies = more power, but diminishing returns after ~10.

### Q: Should I exclude studies with small sample sizes?

**A:** Consider excluding if:
- n < 10 per group (very unreliable)
- Heavily influences results (leave-one-out test)
- Pre-specified exclusion criterion

**Don't** exclude post-hoc just because results change.

### Q: What if studies have different platforms?

**A:** Acceptable if:
- Same organism
- Similar technology (all microarray OR all RNA-seq)
- Good gene overlap (>80%) after annotation
- Account for heterogeneity (random-effects)

### Q: How do I handle missing data?

**A:** Options:
1. **Remove genes** with >20% missing (safest)
2. **Impute** using KNN or mean (if <5% missing)
3. **Report** missingness in results

**Don't** delete samples - loses power.

## Troubleshooting

### No Significant Results

**Checklist:**
- [ ] Sufficient sample size? (>200 total)
- [ ] Sufficient studies? (≥3)
- [ ] Studies actually comparable?
- [ ] Quality control passed?
- [ ] Filtering too aggressive?
- [ ] True null hypothesis?

### Very High Heterogeneity

**Actions:**
1. Check data quality
2. Verify studies measure same phenotype
3. Look for outlier studies
4. Consider subgroup analysis
5. Use random-effects (default)

### Unexpected Results

**Verify:**
1. Class labels correct?
2. Data in log scale?
3. Gene annotation worked?
4. No technical artifacts?
5. Results consistent across studies?

## Appendix

### Recommended Reading

**Meta-Analysis Methods:**
- Borenstein et al. (2009) "Introduction to Meta-Analysis"
- Higgins & Green (2011) "Cochrane Handbook for Systematic Reviews"

**Gene Expression Meta-Analysis:**
- Tseng et al. (2012) "Microarray meta-analysis" *Nucleic Acids Res*
- Ramasamy et al. (2008) "Key issues in meta-analysis" *PLoS Med*

### Software Alternatives

| Software | Language | Strengths | Limitations |
|----------|----------|-----------|-------------|
| MetaIntegrator-py | Python | Fast, modern, integrated | New, fewer features |
| MetaIntegrator (R) | R | Mature, validated | Slower, R-only |
| METAL | C++ | Very fast | Command-line only |
| metafor | R | Most flexible | Steep learning curve |

### Useful Resources

- [GEO Database](https://www.ncbi.nlm.nih.gov/geo/)
- [MetaIntegrator-py Documentation](https://github.com/Khatri-Lab/metaintegrator_py/tree/main/docs)

---

**Last updated:** 2026-04-21  
**Version:** 1.0  
**Maintainer:** MetaIntegrator-py Team
