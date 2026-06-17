# COCONUT Best Practices Guide

Practical recommendations for production-quality COCONUT workflows based on real-world usage patterns.

---

## 🎯 Core Principles

### 1. Use `coconut_runner` for CSV Workflows

**✅ DO:**
```python
from coconut.coconut_runner import runner

runner(csv_dir='./data', output_dir='./results')
```

**❌ DON'T:**
```python
# Manual CSV loading is error-prone
genes1 = pd.read_csv('study1_genes.csv')
pheno1 = pd.read_csv('study1_pheno.csv')
# ... lots of repetitive code
```

**Why:** `coconut_runner` automatically handles sample alignment, validation, and error reporting.

---

### 2. Always Validate Data Before Processing

**✅ DO:**
```python
# Quick validation run (no files saved)
runner(csv_dir='./data')  # Check for issues

# If looks good, run full pipeline
runner(csv_dir='./data', output_dir='./results')
```

**❌ DON'T:**
```python
# Don't skip validation
runner(csv_dir='./data', output_dir='./results')  # Hope it works!
```

---

### 3. Check Gene Overlap Before Running

**✅ DO:**
```python
from coconut.coconut import _get_common_genes

# Check overlap
common = _get_common_genes(gse_data)
print(f"Common genes: {len(common)}")

# Minimum threshold
if len(common) < 100:
    print("⚠️ Warning: Very few common genes")
```

**❌ DON'T:**
```python
# Don't assume gene names align
result = coconut(gses, ...)  # May fail or give poor results
```

---

## 📊 Data Preparation

### File Naming Conventions

**✅ GOOD:**
```
studies/
├── GSE28750_genes.csv
├── GSE28750_pheno.csv
├── GSE40012_genes.csv
└── GSE40012_pheno.csv
```

**❌ BAD:**
```
studies/
├── study1_expression.csv      # Inconsistent naming
├── study1_metadata.csv
├── S2_genes.csv                # Different prefix
└── s2_pheno.csv                # Case inconsistency
```

**Best Practice:**
- Use consistent prefixes (e.g., `GSE`, `Study`)
- Use consistent suffixes (e.g., `_genes.csv`, `_pheno.csv`)
- Avoid special characters and spaces
- Keep naming simple and predictable

---

### Sample Name Alignment

COCONUT requires exact sample name matching between genes and pheno files.

**✅ DO:**
```python
# Both files have identical sample IDs
# genes.csv columns: GSM1, GSM2, GSM3
# pheno.csv index:   GSM1, GSM2, GSM3
```

**❌ AVOID:**
```python
# Mismatched names will cause issues
# genes.csv columns: GSM1, GSM2, GSM3
# pheno.csv index:   Sample1, Sample2, Sample3
```

**Fix Mismatches:**
```python
# Standardize sample names before COCONUT
for gse in gse_data:
    # Example: Convert to uppercase
    gse['genes'].columns = gse['genes'].columns.str.upper()
    gse['pheno'].index = gse['pheno'].index.str.upper()
```

**Note:** `coconut_runner.load_gse_data_from_csv_directory()` automatically handles minor mismatches (whitespace, etc.), but standardizing names upfront is still recommended.

---

### Gene Identifier Consistency

**✅ DO:**
```python
# All datasets use same gene ID system
# Study 1: GAPDH, ACTB, TP53
# Study 2: GAPDH, ACTB, TP53
# Study 3: GAPDH, ACTB, TP53
```

**❌ AVOID:**
```python
# Mixed gene ID systems
# Study 1: GAPDH, ACTB, TP53          (gene symbols)
# Study 2: ENSG00000111640, ...       (Ensembl IDs)
# Study 3: 2597, 60, 7157             (Entrez IDs)
```

**Convert to Common Format:**
```python
# Before COCONUT, standardize gene IDs
from biomart import BiomartServer  # Example

def convert_to_symbols(genes_df, id_type='ensembl_id'):
    # Convert IDs to gene symbols
    # Implementation depends on your ID conversion method
    pass

for gse in gse_data:
    if needs_conversion(gse['genes']):
        gse['genes'] = convert_to_symbols(gse['genes'])
```

---

## ⚙️ Parameter Selection

### `by_platform` Decision Tree

```
Are you combining data from different platforms?
│
├─ YES → Different sequencing platforms (Affymetrix, Illumina, RNA-seq)
│         by_platform=True, platform_col='Platform_id'
│
└─ NO → Same platform across all studies
          by_platform=False
```

**Example:**
```python
# Different platforms
result = coconut(
    gses=gse_data,
    control_0_col='control',
    by_platform=True,           # ← Enable platform-specific
    platform_col='Platform_id'  # ← Specify platform column
)

# Same platform
result = coconut(
    gses=gse_data,
    control_0_col='control',
    by_platform=False  # ← Disable platform-specific
)
```

---

### `par_prior` Selection

Choose based on data characteristics:

| Situation | `par_prior` | Reasoning |
|-----------|-------------|-----------|
| Large datasets (>50 samples/batch) | `True` | Faster, sufficient data for parametric assumptions |
| Small datasets (<20 samples/batch) | `False` | More robust, fewer assumptions |
| Suspected outliers | `False` | Non-parametric is more robust |
| Normal-looking distributions | `True` | Faster without sacrificing accuracy |
| Unknown/mixed distributions | `False` | Safer choice when uncertain |

**Example:**
```python
# Large, well-behaved data
result = coconut(gses, control_0_col='control', par_prior=True)

# Small or questionable data
result = coconut(gses, control_0_col='control', par_prior=False)
```

---

### Convergence Threshold (`it_conv`)

```python
# Standard (recommended)
result = coconut(gses, control_0_col='control', it_conv=1e-04)

# Stricter (slower, more precise)
result = coconut(gses, control_0_col='control', it_conv=1e-06)

# Looser (faster, less precise)
result = coconut(gses, control_0_col='control', it_conv=1e-03)
```

**Best Practice:** Use default `1e-04` unless you have a specific reason to change it.

---

## 🔍 Quality Assessment

### Always Visualize Results

**Minimum QC Checklist:**

1. **PCA Plot** - Check batch effect removal
2. **Expression Distributions** - Verify similar distributions
3. **Sample Correlations** - Confirm biological clustering

**Complete QC Script:**
```python
from coconut.coconut import coconut, combine_coco_output
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Run COCONUT
result = coconut(gses=gse_data, control_0_col='control')
combined = combine_coco_output(result)

# 1. PCA Comparison (Before/After)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before (raw data)
raw_genes = pd.concat([d['genes'] for d in result['rawDiseaseList']], axis=1)
pca = PCA(n_components=2)
pca_raw = pca.fit_transform(raw_genes.T)
ax1.scatter(pca_raw[:, 0], pca_raw[:, 1], c=combined['class_cntl0_dis1'])
ax1.set_title('Before COCONUT')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')

# After
pca_norm = pca.fit_transform(combined['genes'].T)
ax2.scatter(pca_norm[:, 0], pca_norm[:, 1], c=combined['class_cntl0_dis1'])
ax2.set_title('After COCONUT')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')

plt.tight_layout()
plt.savefig('pca_comparison.png', dpi=300)

# 2. Expression Distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before
for i, d in enumerate(result['rawDiseaseList']):
    ax1.hist(d['genes'].values.flatten(), bins=50, alpha=0.3, label=f'Study {i+1}')
ax1.set_title('Before COCONUT')
ax1.set_xlabel('Expression Value')
ax1.legend()

# After
gene_values = combined['genes'].values.flatten()
ax2.hist(gene_values, bins=50)
ax2.set_title('After COCONUT')
ax2.set_xlabel('Expression Value')

plt.tight_layout()
plt.savefig('distributions.png', dpi=300)

# 3. Correlation Heatmap
corr = np.corrcoef(combined['genes'].T)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', center=0, square=True)
plt.title('Sample Correlation After COCONUT')
plt.savefig('correlation.png', dpi=300)

print("✅ QC plots saved")
```

---

### Expected Results

**Good COCONUT Normalization:**

✅ **PCA Plot:**
- Before: Samples cluster by study/batch
- After: Samples cluster by biology (control vs disease)

✅ **Distributions:**
- Before: Different distributions per study
- After: Similar/overlapping distributions

✅ **Correlations:**
- Before: High correlation within studies
- After: Correlation based on biological similarity

**Warning Signs:**

⚠️ **Over-Correction:**
- Controls and disease samples completely mixed in PCA
- Loss of biological signal

⚠️ **Under-Correction:**
- Still see batch clustering in PCA
- Different distributions persist

**If Over-Corrected:**
```python
# Try non-parametric
result = coconut(gses, control_0_col='control', par_prior=False)

# Or adjust convergence
result = coconut(gses, control_0_col='control', it_conv=1e-03)
```

**If Under-Corrected:**
```python
# Try stricter convergence
result = coconut(gses, control_0_col='control', it_conv=1e-06)

# Or enable platform-specific
result = coconut(gses, control_0_col='control', 
                 by_platform=True, platform_col='Platform_id')
```

---

## 💾 Data Management

### Save Intermediate Results

**✅ DO:**
```python
from coconut.coconut import coconut, combine_coco_output
import pickle

# Run COCONUT
result = coconut(gses=gse_data, control_0_col='control')

# Save raw result
with open('coconut_result.pkl', 'wb') as f:
    pickle.dump(result, f)

# Combine and save
combined = combine_coco_output(result)
combined['genes'].to_csv('genes_normalized.csv')
combined['pheno'].to_csv('pheno.csv')

# Can reload later
with open('coconut_result.pkl', 'rb') as f:
    result = pickle.load(f)
```

**❌ DON'T:**
```python
# Don't lose intermediate results
result = coconut(gses, control_0_col='control')
combined = combine_coco_output(result)
# Result is lost if you need to reprocess
```

---

### Version Control Data Processing

**Create Processing Log:**
```python
import datetime
import json

processing_log = {
    'date': datetime.datetime.now().isoformat(),
    'coconut_version': coconut.__version__,
    'parameters': {
        'control_0_col': 'Healthy0.Sepsis1',
        'by_platform': False,
        'par_prior': True,
        'it_conv': 1e-04
    },
    'input_files': list(Path('./data').glob('*.csv')),
    'n_datasets': len(gse_data),
    'n_genes': len(combined['genes']),
    'n_samples': len(combined['genes'].columns)
}

with open('processing_log.json', 'w') as f:
    json.dump(processing_log, f, indent=2, default=str)
```

---

## 🚀 Production Workflows

### Pattern 1: Batch Processing Multiple Projects

```python
from pathlib import Path
from coconut.coconut_runner import runner

projects = [
    'project_sepsis',
    'project_covid',
    'project_cancer'
]

for project in projects:
    print(f"\n🔄 Processing {project}...")
    
    try:
        runner(
            csv_dir=f'./data/{project}',
            output_dir=f'./results/{project}'
        )
        print(f"✅ {project} completed")
        
    except Exception as e:
        print(f"❌ {project} failed: {e}")
        # Log error but continue
```

---

### Pattern 2: Integration with Meta-Analysis

```python
from coconut.coconut import coconut, combine_coco_output
from metaintegrator.analysis import MetaAnalyzer

# Step 1: Normalize with COCONUT
result = coconut(gses=gse_data, control_0_col='control')
combined = combine_coco_output(result)

# Step 2: Prepare for meta-analysis
meta_input = {
    'genes': combined['genes'],
    'pheno': combined['pheno'],
    'class': combined['class_cntl0_dis1']
}

# Step 3: Run meta-analysis
analyzer = MetaAnalyzer()
meta_results = analyzer.run_meta_analysis_core([meta_input])
```

---

### Pattern 3: Machine Learning Pipeline

```python
from coconut.coconut import coconut, combine_coco_output
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Normalize
result = coconut(gses=gse_data, control_0_col='control')
combined = combine_coco_output(result)

# Prepare ML data
X = combined['genes'].T.values  # Samples × genes
y = combined['class_cntl0_dis1']

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)

print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 🎓 Common Patterns

### Loading Data from Multiple Sources

```python
from coconut.coconut_runner import load_gse_data_from_csv_directory

# Load from different directories
batch1 = load_gse_data_from_csv_directory('./batch1')
batch2 = load_gse_data_from_csv_directory('./batch2')

# Combine
all_data = batch1 + batch2

# Run COCONUT on combined data
result = coconut(gses=all_data, control_0_col='control')
```

---

### Handling Missing Values

```python
# Check for missing values BEFORE COCONUT
for i, gse in enumerate(gse_data):
    n_missing = gse['genes'].isna().sum().sum()
    if n_missing > 0:
        print(f"⚠️ Dataset {i} has {n_missing} missing values")
        
        # Option 1: Remove genes with missing values
        gse['genes'] = gse['genes'].dropna(axis=0)
        
        # Option 2: Impute missing values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        gse['genes'] = pd.DataFrame(
            imputer.fit_transform(gse['genes'].T).T,
            index=gse['genes'].index,
            columns=gse['genes'].columns
        )
```

---

### Filtering Low-Variance Genes

```python
# Remove genes with very low variance (likely uninformative)
def filter_low_variance(gse, percentile=10):
    """Keep only genes above specified variance percentile."""
    variances = gse['genes'].var(axis=1)
    threshold = np.percentile(variances, percentile)
    return gse['genes'][variances > threshold]

# Apply before COCONUT
for gse in gse_data:
    original_genes = len(gse['genes'])
    gse['genes'] = filter_low_variance(gse)
    print(f"Filtered: {original_genes} → {len(gse['genes'])} genes")
```

---

## 📝 Documentation Best Practices

### Document Your Workflow

**Create a README for Each Project:**

```markdown
# Sepsis Meta-Analysis

## Data Sources
- GSE28750: Sepsis patients (n=20)
- GSE40012: Sepsis patients (n=30)
- GSE66099: Sepsis patients (n=25)

## COCONUT Parameters
- control_0_col: 'Healthy0.Sepsis1'
- by_platform: False
- par_prior: True
- it_conv: 1e-04

## Processing Steps
1. Loaded data from CSV: 2026-04-23
2. Filtered genes: 12000 → 8500 (low variance removed)
3. Ran COCONUT: 2026-04-23
4. QC passed: PCA shows good batch correction
5. Exported results: ./results/

## Results
- Common genes: 8500
- Total samples: 75 (35 controls, 40 disease)
- Output files: genes_normalized.csv, pheno.csv, labels.csv
```

---

## 🐛 Error Prevention

### Common Mistakes to Avoid

**1. Forgetting to Align Sample Names**
```python
# ❌ BAD: Assume names match
result = coconut(gses, control_0_col='control')

# ✅ GOOD: Verify alignment
for gse in gse_data:
    assert set(gse['genes'].columns) == set(gse['pheno'].index)
```

**2. Using Wrong Control Column**
```python
# ❌ BAD: Guess the column name
result = coconut(gses, control_0_col='Status')  # Might be wrong

# ✅ GOOD: Verify column exists and has correct values
for gse in gse_data:
    print(gse['pheno']['Healthy0.Sepsis1'].value_counts())
    # Should show: 0 (controls), 1 (disease)
```

**3. Ignoring Warnings**
```python
# ❌ BAD: Ignore sample mismatch warnings
⚠️ Sample mismatch in GSE28750: Using 18/20 samples
# ← This needs investigation!

# ✅ GOOD: Investigate and fix
print("Genes samples:", gse['genes'].columns.tolist())
print("Pheno samples:", gse['pheno'].index.tolist())
# Find and fix the mismatch
```

---

## 🎯 Summary Checklist

Before running COCONUT in production:

- [ ] Data files follow consistent naming convention
- [ ] Sample names align between genes and pheno files
- [ ] Gene identifiers are consistent across studies
- [ ] Common genes identified and verified (>100 recommended)
- [ ] Control column correctly identified (0=control, 1=disease)
- [ ] Platform column specified if using `by_platform=True`
- [ ] Parameters documented in processing log
- [ ] QC visualizations prepared and reviewed
- [ ] Output files saved with version information
- [ ] Processing log created for reproducibility

---

## 📚 Further Reading

- [API Reference](api.md) - Complete function documentation
- [Troubleshooting Guide](troubleshooting.md) - Common issues
- [Workflow Tutorial](workflow.md) - Step-by-step examples
- [R to Python Guide](r_to_python.md) - For R users

---

*Last updated: 2026-04-24*  
*Package: metaintegrator_py*
