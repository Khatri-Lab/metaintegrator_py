# Quality Control and Data Assessment

Learn how to assess data quality, verify study compatibility, and ensure your datasets are ready for meta-analysis. Comprehensive quality control prevents garbage-in-garbage-out problems and ensures reliable results.

## Goal

Perform systematic quality checks on gene expression datasets to identify issues before meta-analysis, ensuring data integrity, study compatibility, and appropriate transformations.

## Prerequisites

- Python 3.10+
- metaIntegrator-py git repository cloned
- Datasets downloaded and annotated (see previous tutorials)
- Understanding of log-scale transformation (helpful but not required)

### Why Quality Control Matters

Meta-analysis combines data from multiple studies. **Poor quality data from even one study can corrupt the entire analysis**. QC catches:

- ❌ Missing data that would bias results
- ❌ Studies in wrong scale (linear vs. log)
- ❌ Incompatible platforms
- ❌ Class imbalance (too few cases/controls)
- ❌ Poor gene overlap between studies
- ❌ Outlier samples or batch effects

## Steps

### Step 1: Import Quality Control Tools

```python
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.labeling import assign_class_labels
from metaintegrator.core_analysis import MetaAnalysisCore
from metaintegrator.validation import check_data_object, DatasetValidator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ QC tools imported successfully")
```

!!! note "No built-in one-call QC report"
    There's no `assess_data_quality()` function shipped with the package —
    the checks below (missing data, log-scale heuristic, gene overlap,
    class balance) are exactly what such a function would need to do. This
    tutorial defines `assess_data_quality()` as a small local helper so the
    rest of the walkthrough reads naturally; `check_data_object()` and
    `DatasetValidator` (imported above) are the real, built-in structural
    validators.

```python
def assess_data_quality(meta_object, generate_report=True):
    """Local QC helper: combines the checks from this tutorial into one report."""
    report = {}
    for name, dataset in meta_object['originalData'].items():
        expr = dataset['expr']
        missing_pct = (expr.isna().sum().sum() / expr.size) * 100
        is_log_scale = expr.max().max() < 30
        report[name] = {
            'n_samples': expr.shape[1],
            'n_genes': expr.shape[0],
            'missing_pct': missing_pct,
            'log_scale': is_log_scale,
            'range': (float(expr.min().min()), float(expr.max().max())),
        }
        if generate_report:
            print(f"\nDataset: {name}")
            print(f"   Samples: {report[name]['n_samples']}, Genes: {report[name]['n_genes']:,}")
            print(f"   Missing values: {missing_pct:.2f}%")
            print(f"   Log scale: {'✅ Yes' if is_log_scale else '❌ No'}")
    return report
```

### Step 2: Load Sample Data

```python
# Load previously downloaded and annotated datasets
loader = DataLoaderFactory.create_geo_loader(strategy="r_parity", cache_dir="./geo_cache")
datasets = loader.load_multiple(["GSE5847", "GSE28750"])
meta_object = {"originalData": {
    ds.geo_id: {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
                "formattedName": ds.geo_id, "class": None}
    for ds in datasets
}}

# Quick summary
print(f"📊 Loaded {len(meta_object['originalData'])} datasets")
for name, dataset in meta_object['originalData'].items():
    n_genes, n_samples = dataset['expr'].shape
    print(f"   • {name}: {n_genes:,} genes × {n_samples} samples")
```

**Expected output:**
```
📊 Loaded 2 datasets
   • Breast_Cancer: 54,675 genes × 50 samples
   • Sepsis: 54,675 genes × 30 samples
```

### Step 3: Generate Comprehensive Quality Report

```python
print("\n🔍 Generating comprehensive quality report...")
print("=" * 70)

# Generate detailed QC report
quality_report = assess_data_quality(
    meta_object, 
    generate_report=True  # Print detailed report
)

print("=" * 70)
print("✅ Quality assessment complete")
```

**Expected output:**
```
🔍 Generating comprehensive quality report...
======================================================================

Dataset: GSE5847_GPL96
   Samples: 50, Genes: 22,283
   Missing values: 0.03%
   Log scale: ✅ Yes

Dataset: GSE28750_GPL570
   Samples: 30, Genes: 54,675
   Missing values: 0.01%
   Log scale: ✅ Yes
======================================================================
✅ Quality assessment complete
```

### Step 4: Check Log-Scale Transformation

**Critical:** Meta-analysis requires data in log scale.

```python
print("\n🔬 Verifying log-scale transformation...")
print("-" * 70)

all_log_scale = True

for dataset_name, dataset in meta_object['originalData'].items():
    is_log_scale = dataset['expr'].max().max() < 30
    
    status = "✅" if is_log_scale else "❌"
    scale_type = "log scale" if is_log_scale else "linear scale"
    
    print(f"{status} {dataset_name}: {scale_type}")
    
    if not is_log_scale:
        all_log_scale = False
        print(f"   ⚠️ WARNING: Data must be log-transformed for meta-analysis!")
        
        # Show transformation recommendation
        print(f"   💡 Transformation needed:")
        print(f"      dataset['expr'] = np.log2(dataset['expr'] + 1)")

if all_log_scale:
    print("\n✅ All datasets are in log scale - ready for meta-analysis")
else:
    print("\n❌ Some datasets need log transformation")
```

**Expected output:**
```
🔬 Verifying log-scale transformation...
----------------------------------------------------------------------
✅ Breast_Cancer: log scale
✅ Sepsis: log scale

✅ All datasets are in log scale - ready for meta-analysis
```

**How the log-scale heuristic works:**
There's no built-in log-scale detector, so this tutorial uses a simple range
check: log₂ microarray intensities are rarely above ~20-25, while
linear-scale data commonly reaches into the thousands. `max().max() < 30` is
a reasonable cutoff in practice, but always sanity-check unfamiliar
platforms by eye.

### Step 5: Check for Missing Data

```python
print("\n📊 Missing Data Analysis...")
print("-" * 70)

def analyze_missing_data(dataset, dataset_name):
    """Analyze missing data patterns."""
    
    expr = dataset['expr']
    
    # Overall statistics
    total_values = expr.size
    missing_values = expr.isna().sum().sum()
    missing_pct = (missing_values / total_values) * 100
    
    # Per-gene missing
    missing_per_gene = expr.isna().sum(axis=1)
    genes_with_missing = (missing_per_gene > 0).sum()
    
    # Per-sample missing
    missing_per_sample = expr.isna().sum(axis=0)
    samples_with_missing = (missing_per_sample > 0).sum()
    
    print(f"\n{dataset_name}:")
    print(f"   Total values: {total_values:,}")
    print(f"   Missing values: {missing_values:,} ({missing_pct:.2f}%)")
    print(f"   Genes with missing data: {genes_with_missing:,}")
    print(f"   Samples with missing data: {samples_with_missing}")
    
    # Assess severity
    if missing_pct < 1:
        print(f"   ✅ Low missing data rate (< 1%)")
    elif missing_pct < 5:
        print(f"   ⚠️ Moderate missing data ({missing_pct:.2f}%)")
        print(f"      Consider imputation or filtering")
    else:
        print(f"   ❌ High missing data ({missing_pct:.2f}%)")
        print(f"      Data quality issue - investigate source")
    
    # Check for systematic missingness
    max_missing_sample = missing_per_sample.max()
    if max_missing_sample > total_values * 0.1:
        print(f"   ⚠️ One sample has {max_missing_sample} missing values")
        print(f"      Consider removing this sample")
    
    return {
        'total_pct': missing_pct,
        'genes_affected': genes_with_missing,
        'samples_affected': samples_with_missing
    }

# Analyze all datasets
missing_stats = {}
for name, dataset in meta_object['originalData'].items():
    stats = analyze_missing_data(dataset, name)
    missing_stats[name] = stats
```

**Expected output:**
```
📊 Missing Data Analysis...
----------------------------------------------------------------------

Breast_Cancer:
   Total values: 2,733,750
   Missing values: 820 (0.03%)
   Genes with missing data: 412
   Samples with missing data: 8
   ✅ Low missing data rate (< 1%)

Sepsis:
   Total values: 1,640,250
   Missing values: 164 (0.01%)
   Genes with missing data: 98
   Samples with missing data: 4
   ✅ Low missing data rate (< 1%)
```

### Step 6: Verify Class Balance

Meta-analysis requires adequate samples in each group:

```python
print("\n👥 Sample Distribution and Class Balance...")
print("-" * 70)

def check_class_balance(dataset, dataset_name, min_per_group=5):
    """Check if classes are adequately balanced."""
    
    n_samples = dataset['expr'].shape[1]
    
    print(f"\n{dataset_name}:")
    print(f"   Total samples: {n_samples}")
    
    if 'class' not in dataset:
        print("   ⚠️ No class labels defined")
        print("   💡 Define class labels before meta-analysis:")
        print("      dataset['class'] = np.array([0, 0, ..., 1, 1])  # 0=control, 1=case")
        return False
    
    class_vector = np.array(dataset['class'])
    unique_classes, counts = np.unique(class_vector, return_counts=True)
    
    # Check for binary classification
    if len(unique_classes) != 2:
        print(f"   ❌ Expected 2 classes, found {len(unique_classes)}")
        print(f"      Classes: {unique_classes}")
        return False
    
    # Check for 0/1 encoding
    if not set(unique_classes).issubset({0, 1}):
        print(f"   ❌ Classes must be 0 (control) and 1 (case)")
        print(f"      Found: {unique_classes}")
        return False
    
    # Display class distribution
    for cls, count in zip(unique_classes, counts):
        pct = (count / n_samples) * 100
        class_label = "Controls" if cls == 0 else "Cases"
        print(f"   • {class_label}: {count} samples ({pct:.1f}%)")
    
    # Check minimum samples per group
    min_count = min(counts)
    if min_count < min_per_group:
        print(f"   ⚠️ Small group size: {min_count} samples")
        print(f"      Recommended minimum: {min_per_group} samples per group")
        return False
    
    # Check balance ratio
    balance_ratio = min(counts) / max(counts)
    
    if balance_ratio >= 0.5:
        print(f"   ✅ Good balance (ratio: {balance_ratio:.2f})")
    elif balance_ratio >= 0.3:
        print(f"   ⚠️ Moderate imbalance (ratio: {balance_ratio:.2f})")
        print(f"      Consider stratified sampling or weighted analysis")
    else:
        print(f"   ❌ Severe imbalance (ratio: {balance_ratio:.2f})")
        print(f"      Results may be biased toward majority class")
    
    return True

# Check all datasets
for name, dataset in meta_object['originalData'].items():
    # Create example class labels for tutorial
    n = dataset['expr'].shape[1]
    dataset['class'] = np.concatenate([
        np.zeros(n//2, dtype=int), 
        np.ones(n - n//2, dtype=int)
    ])
    
    check_class_balance(dataset, name)
```

**Expected output:**
```
👥 Sample Distribution and Class Balance...
----------------------------------------------------------------------

Breast_Cancer:
   Total samples: 50
   • Controls: 25 samples (50.0%)
   • Cases: 25 samples (50.0%)
   ✅ Good balance (ratio: 1.00)

Sepsis:
   Total samples: 30
   • Controls: 15 samples (50.0%)
   • Cases: 15 samples (50.0%)
   ✅ Good balance (ratio: 1.00)
```

### Step 7: Check Gene Overlap Across Studies

**Critical for meta-analysis:**

```python
print("\n🧬 Gene Overlap Analysis...")
print("-" * 70)

# Collect all gene sets
all_gene_sets = {}
for name, dataset in meta_object['originalData'].items():
    genes = set(dataset['keys'])
    all_gene_sets[name] = genes
    print(f"\n{name}: {len(genes):,} genes")

# Calculate intersections
gene_sets_list = list(all_gene_sets.values())
common_genes = set.intersection(*gene_sets_list)
all_unique_genes = set.union(*gene_sets_list)

print(f"\n📊 Overlap Statistics:")
print(f"   Common genes (all datasets): {len(common_genes):,}")
print(f"   Total unique genes: {len(all_unique_genes):,}")
print(f"   Coverage: {len(common_genes)/len(all_unique_genes):.1%}")

# Pairwise overlaps (for >2 datasets)
if len(all_gene_sets) > 2:
    print(f"\n   Pairwise overlaps:")
    dataset_names = list(all_gene_sets.keys())
    for i, name1 in enumerate(dataset_names):
        for name2 in dataset_names[i+1:]:
            overlap = len(all_gene_sets[name1] & all_gene_sets[name2])
            pct = overlap / min(len(all_gene_sets[name1]), len(all_gene_sets[name2])) * 100
            print(f"      {name1} ∩ {name2}: {overlap:,} genes ({pct:.1f}%)")

# Assessment
if len(common_genes) < 1000:
    print(f"\n   ❌ Very few common genes (< 1,000)")
    print(f"      Possible causes:")
    print(f"      • Different organisms (human vs mouse)")
    print(f"      • Incompatible platforms")
    print(f"      • Poor gene annotation")
elif len(common_genes) < 5000:
    print(f"\n   ⚠️ Moderate gene overlap (1,000 - 5,000)")
    print(f"      Meta-analysis possible but limited")
else:
    coverage_pct = len(common_genes) / len(all_unique_genes) * 100
    if coverage_pct > 80:
        print(f"\n   ✅ Excellent gene overlap (> 80% coverage)")
        print(f"      Datasets are highly compatible")
    else:
        print(f"\n   ✅ Good gene overlap (> 5,000 common genes)")
        print(f"      Datasets are suitable for meta-analysis")
```

**Expected output:**
```
🧬 Gene Overlap Analysis...
----------------------------------------------------------------------

Breast_Cancer: 54,675 genes
Sepsis: 54,675 genes

📊 Overlap Statistics:
   Common genes (all datasets): 52,341
   Total unique genes: 54,675
   Coverage: 95.7%

   ✅ Excellent gene overlap (> 80% coverage)
      Datasets are highly compatible
```

### Step 8: Detect Outlier Samples

Outliers can distort meta-analysis results:

```python
print("\n🎯 Outlier Detection...")
print("-" * 70)

def detect_sample_outliers(dataset, dataset_name):
    """Detect outlier samples using median absolute deviation."""
    
    expr = dataset['expr']
    
    # Calculate per-sample statistics
    sample_means = expr.mean(axis=0)
    sample_medians = expr.median(axis=0)
    sample_stds = expr.std(axis=0)
    
    # Median absolute deviation (robust to outliers)
    median_of_means = np.median(sample_means)
    mad = np.median(np.abs(sample_means - median_of_means))
    
    # Define outliers as samples > 3 MAD from median
    threshold = 3
    outlier_mask = np.abs(sample_means - median_of_means) > threshold * mad
    
    n_outliers = outlier_mask.sum()
    
    print(f"\n{dataset_name}:")
    print(f"   Samples analyzed: {len(sample_means)}")
    print(f"   Median expression: {median_of_means:.2f}")
    print(f"   MAD: {mad:.2f}")
    
    if n_outliers > 0:
        print(f"   ⚠️ Detected {n_outliers} potential outlier(s)")
        outlier_indices = np.where(outlier_mask)[0]
        for idx in outlier_indices:
            print(f"      Sample {idx}: mean = {sample_means.iloc[idx]:.2f}")
        print(f"   💡 Review these samples for quality issues")
    else:
        print(f"   ✅ No outliers detected (threshold: {threshold} MAD)")
    
    return outlier_mask

# Detect outliers in all datasets
for name, dataset in meta_object['originalData'].items():
    outliers = detect_sample_outliers(dataset, name)
```

**Expected output:**
```
🎯 Outlier Detection...
----------------------------------------------------------------------

Breast_Cancer:
   Samples analyzed: 50
   Median expression: 7.82
   MAD: 0.15
   ✅ No outliers detected (threshold: 3 MAD)

Sepsis:
   Samples analyzed: 30
   Median expression: 7.91
   MAD: 0.12
   ✅ No outliers detected (threshold: 3 MAD)
```

### Step 9: Visualize Data Quality

Create quality control plots:

```python
print("\n📊 Creating QC visualizations...")

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Quality Control Summary', fontsize=16, fontweight='bold')

# Plot 1: Expression distributions
ax1 = axes[0, 0]
for name, dataset in meta_object['originalData'].items():
    expr = dataset['expr']
    # Sample first gene across all samples
    ax1.hist(expr.iloc[0], bins=30, alpha=0.5, label=name)

ax1.set_xlabel('Expression Value (log2)')
ax1.set_ylabel('Frequency')
ax1.set_title('Expression Distribution (Sample Gene)')
ax1.legend()

# Plot 2: Sample-wise mean expression
ax2 = axes[0, 1]
for name, dataset in meta_object['originalData'].items():
    sample_means = dataset['expr'].mean(axis=0)
    ax2.scatter(range(len(sample_means)), sample_means, alpha=0.6, label=name)

ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Mean Expression')
ax2.set_title('Sample-wise Mean Expression')
ax2.legend()

# Plot 3: Gene coverage Venn diagram (for 2 datasets)
ax3 = axes[1, 0]
if len(all_gene_sets) == 2:
    names = list(all_gene_sets.keys())
    set1, set2 = all_gene_sets[names[0]], all_gene_sets[names[1]]
    
    only1 = len(set1 - set2)
    only2 = len(set2 - set1)
    common = len(set1 & set2)
    
    categories = ['Only\n' + names[0], 'Common', 'Only\n' + names[1]]
    values = [only1, common, only2]
    colors = ['steelblue', 'green', 'coral']
    
    ax3.bar(categories, values, color=colors, alpha=0.7)
    ax3.set_ylabel('Number of Genes')
    ax3.set_title('Gene Overlap')
    
    # Add value labels
    for i, v in enumerate(values):
        ax3.text(i, v, f'{v:,}', ha='center', va='bottom')

# Plot 4: Missing data heatmap (sample)
ax4 = axes[1, 1]
dataset = list(meta_object['originalData'].values())[0]
expr_sample = dataset['expr'].iloc[:50, :20]  # First 50 genes, 20 samples
missing_mask = expr_sample.isna()

sns.heatmap(missing_mask, cmap='RdYlGn_r', cbar=True, 
            ax=ax4, xticklabels=False, yticklabels=False)
ax4.set_title('Missing Data Pattern (Sample)')
ax4.set_xlabel('Samples')
ax4.set_ylabel('Genes')

plt.tight_layout()
plt.savefig('qc_summary.png', dpi=300, bbox_inches='tight')
print("✅ QC plots saved to: qc_summary.png")

plt.show()
```

## Tips & Gotchas

### Tip 1: Run QC Before AND After Preprocessing

```python
# QC check 1: Raw data
print("QC Check 1: Raw downloaded data")
qc1 = assess_data_quality(meta_object)

# Apply filtering/normalization
# ... preprocessing steps ...

# QC check 2: After preprocessing
print("\nQC Check 2: After preprocessing")
qc2 = assess_data_quality(meta_object)
```

### Tip 2: Save QC Reports

```python
import json
from datetime import datetime

# Create QC report
qc_report = {
    'timestamp': datetime.now().isoformat(),
    'datasets': {},
    'overall_status': 'PASS'
}

for name, dataset in meta_object['originalData'].items():
    qc_report['datasets'][name] = {
        'n_genes': dataset['expr'].shape[0],
        'n_samples': dataset['expr'].shape[1],
        'missing_pct': (dataset['expr'].isna().sum().sum() / dataset['expr'].size) * 100,
        'log_scale': dataset['expr'].max().max() < 30
    }

# Save report
with open('qc_report.json', 'w') as f:
    json.dump(qc_report, f, indent=2)

print("✅ QC report saved to: qc_report.json")
```

### Tip 3: Automate QC with Thresholds

```python
def qc_pass_fail(dataset, name):
    """Automated pass/fail QC check."""
    
    checks = []
    
    # Check 1: Missing data
    missing_pct = (dataset['expr'].isna().sum().sum() / dataset['expr'].size) * 100
    checks.append(('Missing data', missing_pct < 5, f'{missing_pct:.2f}%'))
    
    # Check 2: Sample size
    n_samples = dataset['expr'].shape[1]
    checks.append(('Sample size', n_samples >= 10, f'{n_samples} samples'))
    
    # Check 3: Log scale
    is_log = dataset['expr'].max().max() < 30
    checks.append(('Log scale', is_log, 'Yes' if is_log else 'No'))
    
    # Check 4: Class balance (if defined)
    if 'class' in dataset:
        unique, counts = np.unique(dataset['class'], return_counts=True)
        balance = min(counts) / max(counts) if len(counts) == 2 else 0
        checks.append(('Class balance', balance >= 0.3, f'{balance:.2f}'))
    
    # Print results
    print(f"\n{name}:")
    all_pass = True
    for check_name, passed, value in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}: {value}")
        if not passed:
            all_pass = False
    
    return all_pass

# Run automated QC
all_passed = all(
    qc_pass_fail(dataset, name) 
    for name, dataset in meta_object['originalData'].items()
)

if all_passed:
    print("\n✅ All datasets passed QC")
else:
    print("\n❌ Some datasets failed QC - review issues above")
```

### Gotcha 1: QC After Every Major Step

Don't assume data quality persists through processing:

```python
# ❌ WRONG - QC once at start, then forget
qc_report = assess_data_quality(meta_object)
# ... many processing steps ...
# No QC verification - data might be corrupted!

# ✅ CORRECT - QC at checkpoints
print("QC: Initial data")
assess_data_quality(meta_object)

# Normalization
apply_normalization(meta_object)
print("\nQC: After normalization")
assess_data_quality(meta_object)

# Gene filtering
filter_low_variance_genes(meta_object)
print("\nQC: After filtering")
assess_data_quality(meta_object)
```

### Gotcha 2: Different Platforms May Have Different Ranges

Log-scale data should have similar ranges, but platform differences exist:

```python
# Check value ranges across datasets
for name, dataset in meta_object['originalData'].items():
    expr = dataset['expr']
    print(f"\n{name}:")
    print(f"   Min: {expr.min().min():.2f}")
    print(f"   Max: {expr.max().max():.2f}")
    print(f"   Mean: {expr.mean().mean():.2f}")
    
# If ranges differ significantly (e.g., [0, 20] vs [5, 15]):
# ⚠️ May indicate different preprocessing or platforms
# Consider additional normalization or batch correction
```

### Gotcha 3: Control Samples May Skew QC Metrics

If datasets include experimental controls:

```python
# Identify and handle control samples
if 'sample_type' in dataset['pheno'].columns:
    is_control = dataset['pheno']['sample_type'] == 'control'
    
    if is_control.any():
        print(f"   Found {is_control.sum()} control samples")
        print(f"   💡 Consider excluding from QC metrics")
        
        # QC on experimental samples only
        expr_experimental = dataset['expr'].loc[:, ~is_control]
```

## Troubleshooting

### Q: QC shows "FAIL" but data looks OK

**A**: Check threshold settings:

```python
# QC may have conservative thresholds
# Review specific failed checks
print(quality_report)

# Adjust thresholds if appropriate for your domain
# Example: Allow more missing data in older platforms
```

### Q: Gene overlap is very low (<50%)

**A**: Investigate causes:

```python
# 1. Check if annotation was applied
sample_keys = [list(d['keys'][:5]) for d in meta_object['originalData'].values()]
print(sample_keys)
# Should show gene symbols, not probe IDs

# 2. Check organisms match
for name, dataset in meta_object['originalData'].items():
    pheno = dataset.get('pheno', pd.DataFrame())
    if 'organism_ch1' in pheno.columns:
        org = pheno['organism_ch1'].iloc[0]
        print(f"{name}: {org}")

# 3. Check platform types
platforms = [d.get('platform') for d in meta_object['originalData'].values()]
print(f"Platforms: {platforms}")
```

### Q: All samples flagged as outliers

**A**: MAD threshold may be too strict:

```python
# Relax threshold
threshold = 5  # Instead of 3
outlier_mask = np.abs(sample_means - median_of_means) > threshold * mad

# Or use different outlier method
from scipy import stats
z_scores = stats.zscore(sample_means)
outliers = np.abs(z_scores) > 3
```

## Next Steps

After comprehensive QC:

1. **[Meta-Analysis Tutorial](meta_analysis.md)** – Proceed with statistical analysis
2. **[Visualization Tutorial](visualization.md)** – Create publication plots
3. **[Complete Workflow Tutorial](complete_workflow.md)** – See QC in full context

### Advanced QC Topics

- **Batch effect detection** – PCA, hierarchical clustering
- **Technical replicate analysis** – Assess reproducibility
- **Cross-platform normalization** – ComBat, quantile normalization
- **Sample quality metrics** – RNA integrity, library size

### Key Takeaways

✅ Run comprehensive QC with a helper like the `assess_data_quality()` defined in Step 1
✅ Verify log-scale transformation with a simple range heuristic (no built-in check exists)
✅ Ensure >80% gene overlap for reliable meta-analysis  
✅ Check class balance (avoid severe imbalance)  
✅ Monitor missing data (<5% acceptable)  
✅ Detect and remove outlier samples  
✅ Save QC reports for reproducibility  
✅ Re-run QC after major processing steps  

---

**Questions?** See [Best Practices](../user_guide/best_practices.md).
