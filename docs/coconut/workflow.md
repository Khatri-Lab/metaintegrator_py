# COCONUT Workflow - Step-by-Step Guide

Complete tutorial for batch effect correction with COCONUT, from data preparation to quality assessment.

---

## 📋 Overview

This guide walks you through the complete COCONUT workflow:

1. **Prepare Your Data** - Organize and validate files
2. **Load Data** - Import with automatic alignment
3. **Run COCONUT** - Execute normalization
4. **Combine Results** - Merge across studies
5. **Assess Quality** - Visualize and validate
6. **Export Results** - Save for downstream analysis

**Time required:** 15-20 minutes for first run

**What you'll learn:**
- How to prepare multi-study gene expression data
- How to run COCONUT with appropriate parameters
- How to validate normalization results
- How to export data for downstream tools

---

## Prerequisites

###  Required Software

```bash
git clone https://github.com/Khatri-Lab/metaintegrator_py.git
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

### Required Data

You need:
- **2+ gene expression studies** (CSV format)
- **Control samples in each study** (at least 3-5 recommended)
- **Common genes** across studies (at least 100 recommended)
- **Matching sample IDs** between genes and pheno files

### Data Format Check

Your files should look like this:

**genes.csv:**
```
         Sample1  Sample2  Sample3
Gene1    5.23     6.45     4.87
Gene2    7.12     5.98     8.34
```

**pheno.csv:**
```
         control  Age  Platform_id
Sample1  0        45   GPL570
Sample2  1        32   GPL6947
```

---

## Step 1: Prepare Your Data

### 1.1 Organize Files

Create a directory with your CSV files:

```bash
my_studies/
├── GSE28750_genes.csv
├── GSE28750_pheno.csv
├── GSE40012_genes.csv
├── GSE40012_pheno.csv
└── GSE66099_genes.csv
└── GSE66099_pheno.csv
```

### 1.2 Validate File Naming

Files must follow a consistent pattern:

```python
from pathlib import Path

csv_dir = Path('./my_studies')
genes_files = list(csv_dir.glob('*_genes.csv'))
pheno_files = list(csv_dir.glob('*_pheno.csv'))

print(f"Genes files: {len(genes_files)}")
print(f"Pheno files: {len(pheno_files)}")

# Should be equal!
if len(genes_files) != len(pheno_files):
    print("⚠️ Warning: Unmatched files!")
```

### 1.3 Check Sample Alignment

Verify sample names match between files:

```python
import pandas as pd

# Example for one dataset
genes = pd.read_csv('my_studies/GSE28750_genes.csv', index_col=0)
pheno = pd.read_csv('my_studies/GSE28750_pheno.csv', index_col=0)

genes_samples = set(genes.columns)
pheno_samples = set(pheno.index)

print(f"Genes samples: {len(genes_samples)}")
print(f"Pheno samples: {len(pheno_samples)}")
print(f"Common: {len(genes_samples & pheno_samples)}")

if genes_samples != pheno_samples:
    print("⚠️ Sample names don't match exactly")
    print(f"Only in genes: {genes_samples - pheno_samples}")
    print(f"Only in pheno: {pheno_samples - genes_samples}")
```

**Note:** `coconut_runner` automatically handles minor mismatches (whitespace, etc.)

### 1.4 Verify Control Column

Ensure your pheno files have a control column:

```python
pheno = pd.read_csv('my_studies/GSE28750_pheno.csv', index_col=0)

# Check for control column
print("Available columns:", list(pheno.columns))

# Verify it has 0 and 1 values
if 'control' in pheno.columns:
    print(pheno['control'].value_counts())
    # Should show: 0 (controls), 1 (disease)
```

---

## Step 2: Load Data

### 2.1 Simple Loading (Recommended)

Use `coconut_runner` for automatic alignment:

```python
from coconut.coconut_runner import load_gse_data_from_csv_directory

# Load all datasets
gse_data = load_gse_data_from_csv_directory('./my_studies')

print(f"Loaded {len(gse_data)} datasets")
```

**What happens:**
- ✅ Finds matching gene/pheno file pairs
- ✅ Aligns sample names automatically
- ✅ Reports any mismatches
- ✅ Validates data structure

### 2.2 Custom File Patterns

If your files don't follow `*_genes.csv` / `*_pheno.csv`:

```python
gse_data = load_gse_data_from_csv_directory(
    csv_directory='./my_studies',
    genes_pattern='.expression.csv',   # Your pattern
    pheno_pattern='.metadata.csv',     # Your pattern
    object_prefix='Study_'             # Your prefix
)
```

### 2.3 Verify Loading

Check what was loaded:

```python
for i, gse in enumerate(gse_data):
    n_genes = len(gse['genes'])
    n_samples = len(gse['genes'].columns)
    n_controls = (gse['pheno']['control'] == 0).sum()
    n_disease = (gse['pheno']['control'] == 1).sum()
    
    print(f"\nDataset {i+1}:")
    print(f"  Genes: {n_genes}")
    print(f"  Samples: {n_samples} ({n_controls} controls, {n_disease} disease)")
```

### 2.4 Check Gene Overlap

```python
from coconut.coconut import _get_common_genes

common_genes = _get_common_genes(gse_data)
print(f"\nCommon genes across all studies: {len(common_genes)}")

if len(common_genes) < 100:
    print("⚠️ Warning: Very few common genes!")
    print("   Consider checking gene naming consistency")
```

**Troubleshooting:** If no common genes:
```python
# Standardize gene names
for gse in gse_data:
    gse['genes'].index = gse['genes'].index.str.upper()
    
# Check again
common_genes = _get_common_genes(gse_data)
print(f"After standardization: {len(common_genes)} common genes")
```

---

## Step 3: Run COCONUT

### 3.1 Basic Usage

```python
from coconut.coconut import coconut

result = coconut(
    gses=gse_data,
    control_0_col='control'  # Column with 0=control, 1=disease
)

print("✅ COCONUT completed!")
```

### 3.2 Choose Parameters

**Do you have different platforms?**

```python
# Check platforms
for i, gse in enumerate(gse_data):
    if 'Platform_id' in gse['pheno'].columns:
        platforms = gse['pheno']['Platform_id'].unique()
        print(f"Dataset {i+1} platforms: {platforms}")

# If different platforms exist
result = coconut(
    gses=gse_data,
    control_0_col='control',
    by_platform=True,           # Enable platform-specific
    platform_col='Platform_id'  # Platform column name
)
```

**Choose parametric vs non-parametric:**

```python
# Parametric (faster, assumes normal distributions)
result = coconut(
    gses=gse_data,
    control_0_col='control',
    par_prior=True  # Default
)

# Non-parametric (slower, more robust)
result = coconut(
    gses=gse_data,
    control_0_col='control',
    par_prior=False  # For small datasets or outliers
)
```

### 3.3 Understand the Output

```python
print("Result structure:")
print(f"  COCONUTList: {len(result['COCONUTList'])} normalized datasets")
print(f"  rawDiseaseList: {len(result['rawDiseaseList'])} original datasets")
print(f"  controlList: Contains control data and batch parameters")

# Access batch correction parameters
gamma_star = result['controlList']['bayes_params']['gamma_star']
delta_star = result['controlList']['bayes_params']['delta_star']

print(f"\nBatch correction parameters:")
print(f"  Gamma (mean shifts): {gamma_star.shape}")
print(f"  Delta (variance scaling): {delta_star.shape}")
```

---

## Step 4: Combine Results

### 4.1 Merge Across Studies

```python
from coconut.coconut import combine_coco_output

combined = combine_coco_output(result)

print("\nCombined data:")
print(f"  Genes: {combined['genes'].shape}")  # genes × samples
print(f"  Pheno: {combined['pheno'].shape}")  # samples × attributes
print(f"  Labels: {combined['class_cntl0_dis1'].shape}")  # binary labels
```

### 4.2 Access Components

```python
# Normalized gene expression
genes = combined['genes']  # pd.DataFrame (genes × samples)

# Phenotype metadata (only common columns)
pheno = combined['pheno']  # pd.DataFrame (samples × attributes)

# Class labels (0=control, 1=disease)
labels = combined['class_cntl0_dis1']  # np.ndarray

print("\nData overview:")
print(f"Total samples: {len(genes.columns)}")
print(f"Total genes: {len(genes)}")
print(f"Controls: {(labels == 0).sum()}")
print(f"Disease: {(labels == 1).sum()}")
```

### 4.3 Verify No Data Loss

```python
# Check sample counts
input_total_samples = sum(
    len(gse['genes'].columns)
    for gse in gse_data
)

output_samples = len(combined['genes'].columns)

print(f"\nInput total samples: {input_total_samples}")
print(f"Output samples: {output_samples}")

# These should match -- combine_coco_output() keeps both the normalized
# controls and the normalized disease samples, nothing is dropped by design.
if input_total_samples != output_samples:
    print("⚠️ Sample count mismatch - investigate!")
```

---

## Step 5: Assess Quality

### 5.1 PCA Visualization

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before COCONUT (raw disease data)
raw_genes = pd.concat(
    [d['genes'] for d in result['rawDiseaseList']], 
    axis=1
)

pca = PCA(n_components=2)
coords_raw = pca.fit_transform(raw_genes.T)

ax1.scatter(coords_raw[:, 0], coords_raw[:, 1], 
            c=labels, cmap='RdYlBu_r', alpha=0.6)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax1.set_title('Before COCONUT')

# After COCONUT
coords_norm = pca.fit_transform(combined['genes'].T)

ax2.scatter(coords_norm[:, 0], coords_norm[:, 1], 
            c=labels, cmap='RdYlBu_r', alpha=0.6)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax2.set_title('After COCONUT')

plt.tight_layout()
plt.savefig('pca_comparison.png', dpi=300)
plt.show()

print("✅ PCA plot saved")
```

**Expected result:**
- Before: Samples cluster by study/batch
- After: Samples cluster by biology (control vs disease)

### 5.2 Expression Distributions

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before
for i, d in enumerate(result['rawDiseaseList']):
    values = d['genes'].values.flatten()
    ax1.hist(values, bins=50, alpha=0.4, label=f'Study {i+1}')

ax1.set_xlabel('Expression Value')
ax1.set_ylabel('Frequency')
ax1.set_title('Before COCONUT')
ax1.legend()

# After
values = combined['genes'].values.flatten()
ax2.hist(values, bins=50, alpha=0.7)
ax2.set_xlabel('Expression Value')
ax2.set_ylabel('Frequency')
ax2.set_title('After COCONUT')

plt.tight_layout()
plt.savefig('distributions.png', dpi=300)
plt.show()

print("✅ Distribution plot saved")
```

**Expected result:**
- Before: Different distributions per study
- After: Similar/overlapping distribution

### 5.3 Sample Correlations

```python
import seaborn as sns

# Compute sample correlations
corr = np.corrcoef(combined['genes'].T)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', center=0, square=True,
            cbar_kws={'label': 'Correlation'})
plt.title('Sample Correlation After COCONUT')
plt.tight_layout()
plt.savefig('correlation.png', dpi=300)
plt.show()

print("✅ Correlation heatmap saved")
```

**Expected result:**
- Correlation based on biological similarity, not batch

### 5.4 Quality Metrics

```python
# Expression range check
print("\nExpression statistics:")
print(f"Min: {combined['genes'].min().min():.2f}")
print(f"Max: {combined['genes'].max().max():.2f}")
print(f"Mean: {combined['genes'].mean().mean():.2f}")
print(f"Std: {combined['genes'].std().mean():.2f}")

# Missing values check
n_missing = combined['genes'].isna().sum().sum()
print(f"\nMissing values: {n_missing}")

if n_missing > 0:
    print("⚠️ Warning: Missing values detected!")

# Class balance
print(f"\nClass distribution:")
print(f"Controls: {(labels == 0).sum()} ({(labels == 0).mean():.1%})")
print(f"Disease: {(labels == 1).sum()} ({(labels == 1).mean():.1%})")
```

---

## Step 6: Export Results

### 6.1 Save to CSV

```python
import pandas as pd
from pathlib import Path

# Create output directory
output_dir = Path('./results')
output_dir.mkdir(exist_ok=True)

# Save normalized genes
combined['genes'].to_csv(output_dir / 'genes_normalized.csv')
print(f"✅ Saved: {output_dir / 'genes_normalized.csv'}")

# Save phenotype data
combined['pheno'].to_csv(output_dir / 'pheno.csv')
print(f"✅ Saved: {output_dir / 'pheno.csv'}")

# Save class labels
pd.DataFrame({
    'sample_id': combined['genes'].columns,
    'class_label': labels.astype(int)
}).to_csv(output_dir / 'class_labels.csv', index=False)
print(f"✅ Saved: {output_dir / 'class_labels.csv'}")

# Save quality control plots
print(f"✅ Saved: pca_comparison.png")
print(f"✅ Saved: distributions.png")
print(f"✅ Saved: correlation.png")
```

### 6.2 Export for R

```python
# R-compatible format (with rownames)
combined['genes'].to_csv(
    output_dir / 'genes_for_R.csv',
    index=True  # Keep gene names as rownames
)

combined['pheno'].to_csv(
    output_dir / 'pheno_for_R.csv',
    index=True  # Keep sample IDs as rownames
)

print("✅ R-compatible files saved")
```

**Load in R:**
```r
genes <- read.csv("results/genes_for_R.csv", row.names=1, check.names=FALSE)
pheno <- read.csv("results/pheno_for_R.csv", row.names=1, check.names=FALSE)
```

### 6.3 Create Analysis Report

```python
report = f"""
COCONUT Batch Correction Report
================================

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Input Data
----------
Number of studies: {len(gse_data)}
Total samples: {sum(len(g['genes'].columns) for g in gse_data)}
Common genes: {len(common_genes)}

Parameters Used
--------------
control_0_col: control
par_prior: True

Output Data
-----------
Normalized samples: {len(combined['genes'].columns)}
Genes: {len(combined['genes'])}
Controls: {(labels == 0).sum()}
Disease: {(labels == 1).sum()}

Files Generated
--------------
- genes_normalized.csv
- pheno.csv
- class_labels.csv
- pca_comparison.png
- distributions.png
- correlation.png

Quality Assessment
-----------------
Expression range: [{combined['genes'].min().min():.2f}, {combined['genes'].max().max():.2f}]
Missing values: {n_missing}
Class balance: {(labels == 0).mean():.1%} controls, {(labels == 1).mean():.1%} disease

Status: ✅ COMPLETE
"""

with open(output_dir / 'analysis_report.txt', 'w') as f:
    f.write(report)

print("✅ Analysis report saved")
print(report)
```

---

## Common Workflows

### Workflow A: Standard Multi-Study

```python
from coconut.coconut_runner import load_gse_data_from_csv_directory
from coconut.coconut import coconut, combine_coco_output

# Load
gse_data = load_gse_data_from_csv_directory('./studies')

# Normalize
result = coconut(gses=gse_data, control_0_col='control')

# Combine
combined = combine_coco_output(result)

# Export
combined['genes'].to_csv('genes.csv')
combined['pheno'].to_csv('pheno.csv')
```

---

### Workflow B: Platform-Specific

```python
# Load
gse_data = load_gse_data_from_csv_directory('./studies')

# Check platforms
for i, gse in enumerate(gse_data):
    print(f"Dataset {i}: {gse['pheno']['Platform_id'].unique()}")

# Normalize by platform
result = coconut(
    gses=gse_data,
    control_0_col='control',
    by_platform=True,
    platform_col='Platform_id'
)

# Combine and export
combined = combine_coco_output(result)
combined['genes'].to_csv('genes_platform_specific.csv')
```

---

### Workflow C: With Meta-Analysis

```python
from coconut.coconut import coconut, combine_coco_output
from metaintegrator.analysis import MetaAnalyzer

# COCONUT normalization
result = coconut(gses=gse_data, control_0_col='control')
combined = combine_coco_output(result)

# Prepare for meta-analysis (expr/class/keys are required)
meta_input = {
    'expr': combined['genes'],
    'class': combined['class_cntl0_dis1'],
    'keys': combined['genes'].index.tolist(),
}

# Run meta-analysis
analyzer = MetaAnalyzer()
meta_results = analyzer.run_meta_analysis_core([meta_input])
```

---

## Troubleshooting

### Issue: No common genes found

```python
# Check gene naming
for i, gse in enumerate(gse_data):
    print(f"Dataset {i} genes: {list(gse['genes'].index[:5])}")

# Standardize
for gse in gse_data:
    gse['genes'].index = gse['genes'].index.str.upper()
    
# Try again
common = _get_common_genes(gse_data)
```

**Full guide:** [Troubleshooting - No Common Genes](troubleshooting.md#error-no-common-genes-found)

---

### Issue: Sample mismatch warnings

```python
# Investigate
gse = gse_data[0]
genes_samples = set(gse['genes'].columns)
pheno_samples = set(gse['pheno'].index)

print("Only in genes:", genes_samples - pheno_samples)
print("Only in pheno:", pheno_samples - genes_samples)

# Fix if needed (coconut_runner auto-handles minor issues)
```

**Full guide:** [Troubleshooting - Sample Mismatch](troubleshooting.md#warning-sample-mismatch-in-dataset)

---

### Issue: Batch effects still visible

```python
# Try platform-specific
result = coconut(
    gses=gse_data,
    control_0_col='control',
    by_platform=True,
    platform_col='Platform_id'
)

# Or stricter convergence
result = coconut(
    gses=gse_data,
    control_0_col='control',
    it_conv=1e-06  # Stricter
)
```

**Full guide:** [Troubleshooting - Batch Effects Remain](troubleshooting.md#issue-batch-effects-still-visible-after-coconut)

---

## Next Steps

### Downstream Analysis

After COCONUT normalization:

1. **Differential Expression**
   ```python
   from scipy import stats
   
   controls = combined['genes'].T[labels == 0]
   disease = combined['genes'].T[labels == 1]
   
   p_values = []
   for gene in combined['genes'].index:
        _, p = stats.ttest_ind(controls[gene], disease[gene])
        p_values.append(p)
   ```

2. **Machine Learning**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   
   X = combined['genes'].T.values
   y = labels
   
   clf = RandomForestClassifier()
   clf.fit(X, y)
   ```

3. **Pathway Analysis**
   ```python
   # Use your favorite pathway enrichment tool
   # with the normalized gene expression matrix
   ```

### Learn More

- [Best Practices](best_practices.md) - Production recommendations
- [API Reference](api.md) - Complete function documentation
- [R to Python Guide](r_to_python.md) - For R users
- [Troubleshooting](troubleshooting.md) - Common issues

---

*Last updated: 2026-04-24*  
*Package: metaintegrator_py v0.1.0*
