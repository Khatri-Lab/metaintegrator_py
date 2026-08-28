# COCONUT Troubleshooting Guide

Step-by-step solutions for common errors and issues when using COCONUT.

---

## 🔍 Quick Diagnosis

### Error Categories

1. [Data Loading Errors](#data-loading-errors) - CSV files, paths, file format
2. [Data Structure Errors](#data-structure-errors) - Gene/sample alignment
3. [Parameter Errors](#parameter-errors) - Invalid or missing parameters
4. [Normalization Errors](#normalization-errors) - Algorithm failures
5. [Output Errors](#output-errors) - Saving and exporting issues
6. [Quality Issues](#quality-issues) - Poor normalization results

---

## Data Loading Errors

### Error: `FileNotFoundError: Directory not found`

**Full Error:**
```
FileNotFoundError: Directory not found: ./my_data
```

**Cause:** Directory path doesn't exist or is incorrect.

**Solutions:**

1. **Verify the path exists:**
```python
from pathlib import Path

path = Path('./my_data')
print(f"Path exists: {path.exists()}")
print(f"Absolute path: {path.absolute()}")
```

2. **Use absolute paths:**
```python
from pathlib import Path

csv_dir = Path('/full/path/to/data').absolute()
runner(csv_dir=str(csv_dir), output_dir='./results')
```

3. **Check current working directory:**
```python
import os

print(f"Current directory: {os.getcwd()}")
print(f"Files here: {os.listdir('.')}")
```

---

### Error: `No CSV files found in directory`

**Full Error:**
```
FileNotFoundError: No CSV files found in ./data
```

**Cause:** Directory is empty or files don't match expected patterns.

**Solutions:**

1. **List files in directory:**
```python
from pathlib import Path

csv_dir = Path('./data')
print(f"All files: {list(csv_dir.glob('*'))}")
print(f"CSV files: {list(csv_dir.glob('*.csv'))}")
```

2. **Check file extensions:**
```python
# Files might be .CSV (uppercase)
for file in Path('./data').iterdir():
    print(f"{file.name} - {file.suffix}")
```

3. **Verify file patterns:**
```python
# Maybe files don't end with _genes.csv
# Adjust patterns to match your files
gse_data = load_gse_data_from_csv_directory(
    csv_directory='./data',
    genes_pattern='.expression.csv',  # Custom pattern
    pheno_pattern='.metadata.csv'      # Custom pattern
)
```

---

### Error: `No complete datasets found`

**Full Error:**
```
ValueError: No complete datasets found (each needs both genes and pheno files)
```

**Cause:** Missing genes or pheno file for some datasets.

**Solutions:**

1. **Check file pairing:**
```python
from pathlib import Path

csv_dir = Path('./data')
genes_files = sorted(csv_dir.glob('*_genes.csv'))
pheno_files = sorted(csv_dir.glob('*_pheno.csv'))

print("Genes files:", [f.name for f in genes_files])
print("Pheno files:", [f.name for f in pheno_files])

# Identify missing pairs
genes_names = {f.stem.replace('_genes', '') for f in genes_files}
pheno_names = {f.stem.replace('_pheno', '') for f in pheno_files}

print("Only genes:", genes_names - pheno_names)
print("Only pheno:", pheno_names - genes_names)
```

2. **Fix naming:**
```bash
# Rename files to match pattern
mv GSE28750.genes.csv GSE28750_genes.csv
mv GSE28750.pheno.csv GSE28750_pheno.csv
```

3. **Add missing files:**
```python
# Ensure each study has both files
# Study1_genes.csv needs Study1_pheno.csv
```

---

### Warning: `Sample mismatch in dataset`

**Full Warning:**
```
⚠️ Sample mismatch in GSE28750:
    Genes has: 20 samples
    Pheno has: 18 samples
    Only in genes: ['Sample19', 'Sample20']
    Using 18 common samples
```

**Cause:** Sample names differ between genes and pheno files.

**Impact:** Non-critical. COCONUT uses only common samples.

**Solutions:**

1. **Investigate the mismatch:**
```python
gse = gse_data[0]  # First dataset

genes_samples = set(gse['genes'].columns)
pheno_samples = set(gse['pheno'].index)

print("Only in genes:", genes_samples - pheno_samples)
print("Only in pheno:", pheno_samples - genes_samples)
```

2. **Fix source files if needed:**
```python
# If the mismatch is a data error, fix the CSV files
# Otherwise, using common samples is fine
```

3. **Standardize sample names:**
```python
# Before COCONUT, standardize names
for gse in gse_data:
    # Remove whitespace
    gse['genes'].columns = gse['genes'].columns.str.strip()
    gse['pheno'].index = gse['pheno'].index.str.strip()
    
    # Convert to uppercase
    gse['genes'].columns = gse['genes'].columns.str.upper()
    gse['pheno'].index = gse['pheno'].index.str.upper()
```

---

## Data Structure Errors

### Error: `No common genes found`

**Full Error:**
```
ValueError: No common genes found across datasets
```

**Cause:** Gene identifiers don't match across datasets.

**Solutions:**

1. **Check gene naming:**
```python
for i, gse in enumerate(gse_data):
    print(f"Dataset {i} genes: {list(gse['genes'].index[:5])}")
```

Example output showing the problem:
```
Dataset 0 genes: ['GAPDH', 'ACTB', 'TP53', 'BRCA1', 'MYC']
Dataset 1 genes: ['ENSG00000111640', 'ENSG00000075624', ...]  # Different IDs!
```

2. **Standardize gene names:**
```python
# Convert all to uppercase
for gse in gse_data:
    gse['genes'].index = gse['genes'].index.str.upper()
```

3. **Remove version numbers:**
```python
# If genes have version numbers (e.g., 'TP53.1', 'TP53.2')
for gse in gse_data:
    gse['genes'].index = gse['genes'].index.str.replace(r'\.\d+$', '', regex=True)
```

4. **Convert gene IDs:**
```python
# If using different ID systems, convert to common format
# This requires biomart or mygene libraries

import mygene

mg = mygene.MyGeneInfo()

def convert_to_symbols(gene_list, from_type='ensembl.gene'):
    results = mg.querymany(gene_list, scopes=from_type, fields='symbol')
    mapping = {r['query']: r.get('symbol', r['query']) for r in results}
    return mapping

for gse in gse_data:
    if gse['genes'].index[0].startswith('ENSG'):
        mapping = convert_to_symbols(gse['genes'].index.tolist())
        gse['genes'].index = [mapping.get(g, g) for g in gse['genes'].index]
```

---

### Error: `Dataset has < 1 control`

**Full Error:**
```
ValueError: Dataset at index 0 has < 1 control
```

**Cause:** No samples marked as controls (value 0) in the specified column.

**Solutions:**

1. **Check control column values:**
```python
for i, gse in enumerate(gse_data):
    col = 'Healthy0.Sepsis1'  # Your control column
    if col in gse['pheno'].columns:
        print(f"Dataset {i}:")
        print(gse['pheno'][col].value_counts())
    else:
        print(f"Dataset {i}: Column '{col}' not found!")
```

Expected output:
```
Dataset 0:
0    10  ← controls (must have at least 1)
1    10  ← disease
```

Problem output:
```
Dataset 0:
1    20  ← only disease, no controls!
```

2. **Verify column name:**
```python
# Maybe the column has a different name
for i, gse in enumerate(gse_data):
    print(f"Dataset {i} columns:")
    for col in gse['pheno'].columns:
        print(f"  {col}: {gse['pheno'][col].unique()}")
```

3. **Fix encoding:**
```python
# If controls are labeled differently (e.g., 'Control' vs 0)
for gse in gse_data:
    # Convert 'Control'/'Disease' to 0/1
    gse['pheno']['Healthy0.Sepsis1'] = gse['pheno']['Status'].map({
        'Control': 0,
        'Disease': 1
    })
```

---

### Error: `Column not found in phenotype data`

**Full Error:**
```
ValueError: Filter column 'Healthy0.Sepsis1' not found in phenotype data
```

**Cause:** Specified column doesn't exist in pheno DataFrame.

**Solutions:**

1. **List available columns:**
```python
for i, gse in enumerate(gse_data):
    print(f"Dataset {i} columns: {list(gse['pheno'].columns)}")
```

2. **Use correct column name:**
```python
# Check for similar names
import difflib

target = 'Healthy0.Sepsis1'
for gse in gse_data:
    close_matches = difflib.get_close_matches(
        target, 
        gse['pheno'].columns,
        n=3, 
        cutoff=0.6
    )
    print(f"Similar columns: {close_matches}")
```

3. **Standardize column names:**
```python
# Remove spaces, standardize
for gse in gse_data:
    gse['pheno'].columns = gse['pheno'].columns.str.strip()
    gse['pheno'].columns = gse['pheno'].columns.str.replace(' ', '_')
```

---

## Parameter Errors

### Error: `platform_col must be provided when by_platform=True`

**Full Error:**
```
ValueError: platform_col must be provided if by_platform=True
```

**Cause:** Enabled platform-specific normalization without specifying platform column.

**Solutions:**

1. **Specify platform column:**
```python
result = coconut(
    gses=gse_data,
    control_0_col='control',
    by_platform=True,
    platform_col='Platform_id'  # ← Add this
)
```

2. **Or disable platform-specific:**
```python
result = coconut(
    gses=gse_data,
    control_0_col='control',
    by_platform=False  # ← Set to False
)
```

3. **Check if platform column exists:**
```python
for i, gse in enumerate(gse_data):
    if 'Platform_id' in gse['pheno'].columns:
        print(f"Dataset {i}: Has platform column")
        print(gse['pheno']['Platform_id'].unique())
    else:
        print(f"Dataset {i}: No platform column!")
```

---

### Error: `At least 2 batches are required`

**Full Error:**
```
ValueError: At least 2 batches are required for ComBat normalization. 1 batch(es) found.
```

**Cause:** Only one dataset provided. COCONUT requires multiple studies.

**Solutions:**

1. **Check number of datasets:**
```python
print(f"Number of datasets: {len(gse_data)}")
# Need at least 2
```

2. **Add more datasets:**
```python
# Load additional studies
gse_data = load_gse_data_from_csv_directory('./more_studies')
```

3. **For single dataset:**
```python
# Single dataset doesn't need COCONUT
# Use standard normalization instead
from sklearn.preprocessing import StandardScaler

genes = gse_data[0]['genes']
scaler = StandardScaler()
genes_normalized = pd.DataFrame(
    scaler.fit_transform(genes.T).T,
    index=genes.index,
    columns=genes.columns
)
```

---

## Normalization Errors

### Error: `Convergence failure`

**Full Error:**
```
RuntimeError: EB algorithm failed to converge after max iterations
```

**Cause:** Algorithm couldn't converge within iteration limit.

**Solutions:**

1. **Increase convergence threshold:**
```python
# Looser threshold (easier to converge)
result = coconut(
    gses=gse_data,
    control_0_col='control',
    it_conv=1e-03  # Instead of default 1e-04
)
```

2. **Try non-parametric:**
```python
result = coconut(
    gses=gse_data,
    control_0_col='control',
    par_prior=False  # More robust
)
```

3. **Check for outliers:**
```python
# Extreme values can prevent convergence
for gse in gse_data:
    print(f"Expression range: {gse['genes'].min().min()} to {gse['genes'].max().max()}")
    
    # Clip extreme values if needed
    gse['genes'] = gse['genes'].clip(lower=-10, upper=20)
```

---

### Issue: Numerical instability (inf or NaN values)

**Symptom:** COCONUT doesn't explicitly check for `inf`/`NaN` values, so bad data here usually shows up as a NumPy `RuntimeWarning` (e.g. "invalid value encountered in ...") or NaN-filled results, not a single clean error message.

**Cause:** Invalid numerical values (infinity or NaN) in data.

**Solutions:**

1. **Check for missing values:**
```python
for i, gse in enumerate(gse_data):
    n_missing = gse['genes'].isna().sum().sum()
    print(f"Dataset {i}: {n_missing} missing values")
```

2. **Remove or impute missing values:**
```python
# Option 1: Remove genes with missing values
for gse in gse_data:
    gse['genes'] = gse['genes'].dropna(axis=0)

# Option 2: Impute
from sklearn.impute import SimpleImputer

for gse in gse_data:
    if gse['genes'].isna().any().any():
        imputer = SimpleImputer(strategy='mean')
        gse['genes'] = pd.DataFrame(
            imputer.fit_transform(gse['genes'].T).T,
            index=gse['genes'].index,
            columns=gse['genes'].columns
        )
```

3. **Check for inf values:**
```python
import numpy as np

for i, gse in enumerate(gse_data):
    has_inf = np.isinf(gse['genes']).any().any()
    if has_inf:
        print(f"Dataset {i} has infinite values!")
        # Replace inf with large number
        gse['genes'] = gse['genes'].replace([np.inf, -np.inf], np.nan)
        gse['genes'] = gse['genes'].fillna(gse['genes'].median())
```

---

### Issue: Very few samples in a batch

**Symptom:** No explicit warning is raised for this -- but batch-effect estimates get less reliable when a batch has very few samples (typically <5), so it's worth checking batch sizes yourself.

**Cause:** One batch has very few samples (typically <5).

**Impact:** Batch effect estimates less reliable for small batches.

**Solutions:**

1. **Check batch sizes:**
```python
for i, gse in enumerate(gse_data):
    n_samples = len(gse['genes'].columns)
    print(f"Dataset {i}: {n_samples} samples")
```

2. **Combine small batches (if appropriate):**
```python
# If studies are very similar, can combine
combined_small = {
    'genes': pd.concat([gse1['genes'], gse2['genes']], axis=1),
    'pheno': pd.concat([gse1['pheno'], gse2['pheno']], axis=0)
}
gse_data = [combined_small, gse3, gse4]
```

3. **Accept limitation:**
```python
# Small batches are corrected, but with more uncertainty
# Proceed if batch effects are not severe
result = coconut(gses=gse_data, control_0_col='control')
```

---

## Output Errors

### Error: `Permission denied when saving`

**Full Error:**
```
PermissionError: [Errno 13] Permission denied: './results/genes.csv'
```

**Cause:** Don't have write permission for output directory.

**Solutions:**

1. **Check directory permissions:**
```bash
ls -ld ./results
# Should show write permission (w)
```

2. **Create directory with proper permissions:**
```python
from pathlib import Path

output_dir = Path('./results')
output_dir.mkdir(exist_ok=True, mode=0o755)
```

3. **Use different output directory:**
```python
# Try user home directory
from pathlib import Path

output_dir = Path.home() / 'coconut_results'
runner(csv_dir='./data', output_dir=str(output_dir))
```

---

### Error: `Disk space full`

**Full Error:**
```
OSError: [Errno 28] No space left on device
```

**Cause:** Not enough disk space to save output files.

**Solutions:**

1. **Check available space:**
```bash
df -h .
```

2. **Clean up space:**
```bash
# Remove unnecessary files
rm -rf large_file.csv
```

3. **Save to different location:**
```python
# Use external drive or different partition
runner(csv_dir='./data', output_dir='/mnt/external/results')
```

---

## Quality Issues

### Issue: Batch effects still visible after COCONUT

**Symptom:** PCA plot still shows clustering by batch/study.

**Diagnosis:**
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Check PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(combined['genes'].T)

# Color by batch
batch_labels = combined['pheno']['batch_id']  # If you have batch info
plt.scatter(coords[:, 0], coords[:, 1], c=batch_labels)
plt.title('After COCONUT - still batch clustering?')
plt.show()
```

**Solutions:**

1. **Try platform-specific normalization:**
```python
result = coconut(
    gses=gse_data,
    control_0_col='control',
    by_platform=True,
    platform_col='Platform_id'
)
```

2. **Use stricter convergence:**
```python
result = coconut(
    gses=gse_data,
    control_0_col='control',
    it_conv=1e-06  # Stricter
)
```

3. **Check if platform effects are confounded with biology:**
```python
# If one platform only has controls and another only disease,
# COCONUT can't separate platform from biology
for gse in gse_data:
    print(f"Platform: {gse['pheno']['Platform_id'].iloc[0]}")
    print(gse['pheno']['control_col'].value_counts())
```

---

### Issue: Biological signal lost (over-correction)

**Symptom:** Controls and disease samples completely mixed in PCA.

**Diagnosis:**
```python
# Check separation between controls and disease
pca = PCA(n_components=2)
coords = pca.fit_transform(combined['genes'].T)

plt.scatter(coords[:, 0], coords[:, 1], c=combined['class_cntl0_dis1'])
plt.title('Biological separation preserved?')
plt.show()
```

**Solutions:**

1. **Use parametric priors (less aggressive):**
```python
result = coconut(
    gses=gse_data,
    control_0_col='control',
    par_prior=True  # Instead of False
)
```

2. **Loosen convergence:**
```python
result = coconut(
    gses=gse_data,
    control_0_col='control',
    it_conv=1e-03  # Looser
)
```

3. **Verify control column is correct:**
```python
# Make sure you're using the right phenotype column
for gse in gse_data:
    print(gse['pheno']['control_col'].value_counts())
```

---

### Issue: Different sample counts in output vs input

**Symptom:** Combined output has fewer samples than expected.

**Diagnosis:**
```python
# Count samples
input_samples = sum(len(gse['genes'].columns) for gse in gse_data)
output_samples = len(combined['genes'].columns)

print(f"Input: {input_samples} samples")
print(f"Output: {output_samples} samples")
print(f"Lost: {input_samples - output_samples} samples")
```

**Causes & Solutions:**

1. **Common genes filtering:**
```python
# COCONUT only uses genes common to all datasets
# Some samples might be dropped if they have no data for common genes

# Check common genes
from coconut.coconut import _get_common_genes
common = _get_common_genes(gse_data)
print(f"Common genes: {len(common)}")
```

2. **Controls are included in the combined output:**
```python
# combine_coco_output() keeps both the normalized controls and the
# normalized disease samples -- output_samples should equal the TOTAL
# input sample count, not just the disease count. If it doesn't, the
# real cause is usually sample-alignment during loading (see "Sample
# mismatch" above), not controls being dropped here.

n_controls = sum(
    (gse['pheno']['control_col'] == 0).sum() 
    for gse in gse_data
)
print(f"Controls (included in output): {n_controls}")
```

3. **Phenotype column filtering:**
```python
# combined['pheno'] only includes columns present in ALL datasets
print("Original pheno columns:")
for i, gse in enumerate(gse_data):
    print(f"  Dataset {i}: {list(gse['pheno'].columns)}")
    
print(f"\nCombined pheno columns: {list(combined['pheno'].columns)}")
```

---

## Debug Mode

Enable detailed output for troubleshooting:

```python
from coconut import coconut_runner

# Turn on debug mode
coconut_runner.debug_mode = True

# Now run with verbose output
from coconut.coconut_runner import runner
runner(csv_dir='./data', output_dir='./results')
```

**Debug Output:**
```
🔍 Scanning directory: ./data
📁 Found 6 CSV files
📊 Loading dataset: GSE28750
   📈 Genes: 12000 genes × 20 samples
   📈 Sample names in genes: ['GSM1', 'GSM2', 'GSM3']...
   📋 Pheno: 20 samples × 5 attributes
   📋 Sample names in pheno: ['GSM1', 'GSM2', 'GSM3']...
   🔍 Debug - Genes samples (cleaned): ['GSM1', 'GSM2']...
   🔍 Debug - Pheno samples (cleaned): ['GSM1', 'GSM2']...
   ✅ Final alignment: 20 samples
✅ Successfully loaded 3 datasets
```

---

## Getting Help

### Create Minimal Reproducible Example

If you need to report a bug:

```python
# 1. Use synthetic data
import numpy as np
import pandas as pd

# Create simple test case
gse1 = {
    'genes': pd.DataFrame(
        np.random.randn(100, 10),
        index=[f'Gene{i}' for i in range(100)],
        columns=[f'Sample{i}' for i in range(10)]
    ),
    'pheno': pd.DataFrame({
        'control': [0]*5 + [1]*5
    }, index=[f'Sample{i}' for i in range(10)])
}

gse2 = {
    'genes': pd.DataFrame(
        np.random.randn(100, 10),
        index=[f'Gene{i}' for i in range(100)],
        columns=[f'Sample{i}' for i in range(10, 20)]
    ),
    'pheno': pd.DataFrame({
        'control': [0]*5 + [1]*5
    }, index=[f'Sample{i}' for i in range(10, 20)])
}

# 2. Test with minimal data
from coconut.coconut import coconut
result = coconut([gse1, gse2], control_0_col='control')
```

### Include Relevant Information

When asking for help, provide:

1. **Error message** (full traceback)
2. **Code that reproduces error**
3. **Python version:** `python --version`
4. **Package version:** `import metaintegrator; print(metaintegrator.__version__)`
5. **Data characteristics:**
   - Number of datasets
   - Samples per dataset
   - Number of genes
   - Control/disease counts

---

## Quick Reference

| Error Type | First Thing to Check |
|------------|---------------------|
| `FileNotFoundError` | Path exists? Files have correct extension? |
| `ValueError: No common genes` | Gene names standardized across datasets? |
| `Dataset has < 1 control` | Control column has 0 values? |
| `Column not found` | Column name spelled correctly? Exists in all datasets? |
| `Sample mismatch` | Sample names match between genes/pheno? |
| `Convergence failure` | Try `par_prior=False` or loosen `it_conv` |
| `Numerical instability` | Check for NaN or inf values in data |
| Batch effects remain | Try `by_platform=True` |
| Signal lost | Try `par_prior=True` or loosen `it_conv` |

---

## Additional Resources

- [API Reference](api.md) - Complete function documentation
- [Best Practices](best_practices.md) - Production recommendations
- [Quick Start](quickstart.md) - Fast 5-minute guide
- [Workflow Guide](workflow.md) - Detailed tutorial

---

*Last updated: 2026-04-24*  
*Package: metaintegrator_py*
