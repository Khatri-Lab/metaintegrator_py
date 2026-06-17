# Data Acquisition from GEO

Learn how to download and prepare gene expression datasets from NCBI's Gene Expression Omnibus (GEO) using metaIntegrator-py's production-grade data loader with real NCBI GEO fetching, caching, and preprocessing capabilities.

## Goal

Download gene expression datasets from GEO, apply preprocessing (quantile normalization, log transformation), and organize data into MetaIntegrator-compatible format ready for meta-analysis.

## Prerequisites

- Python 3.10+
- metaIntegrator-py git repository cloned: `git clone https://github.com/Khatri-Lab/metaintegrator_py.git`
- Internet connection (first download only)
- ~100 MB disk space for caching

### Required Knowledge

- Basic understanding of GEO database structure
- Familiarity with gene expression matrices (genes × samples)
- Basic Python (dictionaries, DataFrames)

## Steps

### Step 1: Import the Data Loader

```python
from preprocessing.geo_data_loader import (
    get_geo_data,           # Main function for downloading
    check_geo_log_scale,    # Verify log-scale transformation
    GEODataLoader          # Advanced loader class
)

import pandas as pd
import numpy as np

print("✅ Data loader imported successfully")
```

**What is GEO?**  
The Gene Expression Omnibus (GEO) is NCBI's public repository of gene expression data. Each dataset has a unique GSE (GEO Series) identifier like `GSE5847`.

### Step 2: Download a Single Dataset

Let's start with downloading one small dataset:

```python
# Download one dataset
print("📥 Downloading dataset from GEO...")

meta_object = get_geo_data(
    gse_vector="GSE5847",           # GSE ID (can also be a list)
    formatted_names=["My_Study"],   # Optional: custom name
    q_norm=True,                    # Apply quantile normalization
    cache_dir="./geo_cache"         # Cache location
)

print("✅ Download complete!")

# Inspect the structure
print(f"\nDatasets downloaded: {list(meta_object['originalData'].keys())}")
```

**Expected output:**
```
📥 Downloading dataset from GEO...
Processing GSE5847 (1/1)...
✅ Download complete!

Datasets downloaded: ['GSE5847']
```

**What happened?**
1. Contacted NCBI GEO servers
2. Downloaded series matrix file
3. Extracted expression data, sample metadata, and gene annotations
4. Applied quantile normalization
5. Verified log-scale transformation
6. Cached files locally at `./geo_cache/`

### Step 3: Download Multiple Datasets

Meta-analysis requires multiple studies. Let's download several:

```python
# Download multiple datasets
gse_ids = ["GSE5847", "GSE28750", "GSE40012"]
study_names = ["Breast_Cancer", "Sepsis_Blood", "TB_Infection"]

print(f"📥 Downloading {len(gse_ids)} datasets...")

meta_object = get_geo_data(
    gse_vector=gse_ids,
    formatted_names=study_names,
    q_norm=True,
    cache_dir="./geo_cache"
)

# Display information
print(f"\n✅ Downloaded {len(meta_object['originalData'])} datasets:")
for name, dataset in meta_object['originalData'].items():
    n_genes, n_samples = dataset['expr'].shape
    print(f"   • {name}: {n_genes:,} genes × {n_samples} samples")
```

**Expected output:**
```
📥 Downloading 3 datasets...
Processing GSE5847 (1/3)...
Processing GSE28750 (2/3)...
Processing GSE40012 (3/3)...

✅ Downloaded 3 datasets:
   • Breast_Cancer: 54,675 genes × 50 samples
   • Sepsis_Blood: 54,675 genes × 30 samples
   • TB_Infection: 47,323 genes × 85 samples
```

**Performance Note:**  
First download: ~2-3 minutes per dataset (network dependent)  
Subsequent runs: < 5 seconds (uses cached data)

### Step 4: Inspect Dataset Structure

Let's examine what we downloaded:

```python
# Pick one dataset to inspect
dataset_name = list(meta_object['originalData'].keys())[0]
dataset = meta_object['originalData'][dataset_name]

print(f"📊 Inspecting dataset: {dataset_name}\n")

# Check available fields
print("Available fields:")
for key in dataset.keys():
    print(f"   • {key}")

# Examine expression matrix
print(f"\n📈 Expression Matrix:")
print(f"   Shape: {dataset['expr'].shape}")
print(f"   Type: {type(dataset['expr'])}")
print(f"\n   First 5 genes × 3 samples:")
print(dataset['expr'].iloc[:5, :3])

# Check gene identifiers
print(f"\n🧬 Gene Identifiers (first 10):")
print(dataset['keys'][:10])

# Examine sample metadata
print(f"\n👥 Sample Metadata:")
print(f"   Available: {dataset['pheno'].shape[0]} samples")
print(f"   Columns: {list(dataset['pheno'].columns[:5])}...")
```

**Expected output:**
```
📊 Inspecting dataset: Breast_Cancer

Available fields:
   • name
   • expr
   • keys
   • pheno
   • platform
   • class
   • formattedName
   • exp_comment
   • load_comment

📈 Expression Matrix:
   Shape: (54675, 50)
   Type: <class 'pandas.core.frame.DataFrame'>

   First 5 genes × 3 samples:
                GSM123456  GSM123457  GSM123458
AFFX_001          7.234      7.189      7.301
AFFX_002          8.456      8.423      8.489
1007_s_at         9.123      9.087      9.156
...

🧬 Gene Identifiers (first 10):
['AFFX-BioB-5_at', 'AFFX-BioB-M_at', 'AFFX-BioB-3_at', ...]

👥 Sample Metadata:
   Available: 50 samples
   Columns: ['title', 'geo_accession', 'source_name_ch1', ...]
```

### Step 5: Verify Data Quality

Check critical data quality metrics:

```python
def check_dataset_quality(dataset, dataset_name):
    """Quick quality check for downloaded dataset."""
    
    print(f"🔍 Quality Check: {dataset_name}")
    print("-" * 60)
    
    expr = dataset['expr']
    
    # Check 1: Missing values
    total_values = expr.size
    missing_values = expr.isna().sum().sum()
    missing_pct = (missing_values / total_values) * 100
    
    status_missing = "✅" if missing_pct < 5 else "⚠️"
    print(f"{status_missing} Missing values: {missing_pct:.2f}%")
    
    # Check 2: Log-scale verification
    is_log_scale = check_geo_log_scale(dataset)
    status_log = "✅" if is_log_scale else "❌"
    scale_type = "log scale" if is_log_scale else "linear scale"
    print(f"{status_log} Data transformation: {scale_type}")
    
    # Check 3: Value range
    min_val = expr.min().min()
    max_val = expr.max().max()
    mean_val = expr.mean().mean()
    
    print(f"   Range: [{min_val:.2f}, {max_val:.2f}]")
    print(f"   Mean: {mean_val:.2f}")
    
    # Check 4: Gene identifier type
    sample_keys = dataset['keys'][:5]
    has_gene_symbols = all(
        isinstance(k, str) and not k.startswith('AFFX') 
        for k in sample_keys
    )
    
    status_genes = "✅" if has_gene_symbols else "⚠️"
    key_type = "Gene symbols" if has_gene_symbols else "Probe IDs"
    print(f"{status_genes} Gene IDs: {key_type}")
    print(f"   Examples: {', '.join(map(str, sample_keys))}")
    
    print()

# Check all datasets
for name, dataset in meta_object['originalData'].items():
    check_dataset_quality(dataset, name)
```

**Expected output:**
```
🔍 Quality Check: Breast_Cancer
------------------------------------------------------------
✅ Missing values: 0.03%
✅ Data transformation: log scale
   Range: [2.34, 14.56]
   Mean: 7.82
⚠️ Gene IDs: Probe IDs
   Examples: AFFX-BioB-5_at, AFFX-BioB-M_at, ...

🔍 Quality Check: Sepsis_Blood
------------------------------------------------------------
✅ Missing values: 0.01%
✅ Data transformation: log scale
   Range: [2.45, 14.23]
   Mean: 7.91
⚠️ Gene IDs: Probe IDs
   Examples: AFFX-BioB-5_at, ...
```

**Note:** Probe IDs need to be converted to gene symbols (covered in [Gene Annotation Tutorial](gene_annotation.md))

### Step 6: Apply Quantile Normalization

Quantile normalization makes datasets comparable across different platforms:

```python
# Download WITHOUT quantile normalization
print("📥 Downloading without normalization...")
meta_raw = get_geo_data(
    gse_vector="GSE5847",
    q_norm=False,  # No normalization
    cache_dir="./geo_cache"
)

# Download WITH quantile normalization
print("📥 Downloading with normalization...")
meta_norm = get_geo_data(
    gse_vector="GSE5847",
    q_norm=True,   # Apply normalization
    cache_dir="./geo_cache"
)

# Compare distributions
expr_raw = meta_raw['originalData']['GSE5847']['expr']
expr_norm = meta_norm['originalData']['GSE5847']['expr']

print("\n📊 Comparing distributions:")
print(f"\nRaw data (no normalization):")
print(f"   Mean per sample: {expr_raw.mean(axis=0).mean():.3f}")
print(f"   Std per sample: {expr_raw.std(axis=0).mean():.3f}")

print(f"\nNormalized data:")
print(f"   Mean per sample: {expr_norm.mean(axis=0).mean():.3f}")
print(f"   Std per sample: {expr_norm.std(axis=0).mean():.3f}")
```

**Expected output:**
```
📥 Downloading without normalization...
📥 Downloading with normalization...

📊 Comparing distributions:

Raw data (no normalization):
   Mean per sample: 7.823
   Std per sample: 2.456

Normalized data:
   Mean per sample: 7.823
   Std per sample: 2.123
```

**When to use normalization:**
- ✅ **Use `q_norm=True`** when combining datasets from different platforms
- ❌ **Use `q_norm=False`** when datasets are already normalized or from the same batch

### Step 7: Handle Multi-Platform Datasets

Some GSE IDs contain data from multiple microarray platforms:

```python
# Example: Dataset with multiple platforms
meta_multi = get_geo_data(
    gse_vector="GSE12345",  # Hypothetical multi-platform dataset
    cache_dir="./geo_cache"
)

# Check how many platforms were detected
n_datasets = len(meta_multi['originalData'])
print(f"📊 Dataset contains {n_datasets} platform(s):")

for name in meta_multi['originalData'].keys():
    print(f"   • {name}")

# Each platform is stored separately
# Name format: GSE12345_GPL570, GSE12345_GPL96, etc.
```

**Expected output:**
```
📊 Dataset contains 2 platform(s):
   • GSE12345_GPL570
   • GSE12345_GPL96
```

**Handling strategy:**
- Analyze each platform separately, OR
- Merge after converting to common gene symbols (requires annotation)

### Step 8: Advanced Download Options with Loader Class

For more control, use the `GEODataLoader` class directly:

```python
from preprocessing.geo_data_loader import GEODataLoader

# Initialize loader with custom settings
loader = GEODataLoader(cache_dir="./custom_cache")

# Download with advanced options
print("📥 Using advanced loader...")

meta_object = loader.get_geo_data(
    gse_vector=["GSE5847"],
    formatted_names=["Study_A"],
    q_norm=True,
    dest_dir="./downloads"  # Optional: save raw files here
)

# Access internal methods for debugging
dataset = meta_object['originalData']['Study_A']

# Check if data is in log scale
is_log = loader._gem_log_check(dataset, dataset['expr'])
print(f"Log scale check: {is_log}")

# Examine annotation quality
gene_keys = dataset['keys'][:100]
print(f"Sample gene keys: {gene_keys[:5]}")
```

## Tips & Gotchas

### Tip 1: Cache Everything

**Always specify `cache_dir`** to avoid re-downloading:

```python
# ✅ Good: Uses cache
meta_object = get_geo_data(
    gse_vector=["GSE5847"],
    cache_dir="./geo_cache"  # Cache location
)

# ❌ Avoid: Downloads every time
meta_object = get_geo_data(
    gse_vector=["GSE5847"]
    # No cache_dir specified
)
```

### Tip 2: Check Platform Compatibility

Before meta-analysis, verify datasets use compatible platforms:

```python
# Check platforms
platforms = []
for name, dataset in meta_object['originalData'].items():
    platform = dataset.get('platform', 'Unknown')
    platforms.append((name, platform))
    print(f"{name}: {platform}")

# All should be same organism and similar technology
# Good: All GPL570 (Affymetrix Human Genome U133 Plus 2.0)
# Bad: Mix of GPL570 (Affymetrix) and GPL96 (Illumina)
```

### Tip 3: Handle Download Failures Gracefully

Some datasets may fail to download:

```python
import warnings

# Download multiple with error handling
gse_list = ["GSE5847", "GSE99999", "GSE28750"]  # GSE99999 doesn't exist

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    meta_object = get_geo_data(
        gse_vector=gse_list,
        cache_dir="./geo_cache"
    )
    
    # Check for warnings
    if w:
        print(f"⚠️ {len(w)} dataset(s) failed to download")
        for warning in w:
            print(f"   • {warning.message}")

# Continue with successfully downloaded datasets
print(f"✅ Successfully downloaded: {len(meta_object['originalData'])} datasets")
```

### Gotcha 1: Case-Sensitive GSE IDs

GEO IDs must be uppercase:

```python
# ✅ Correct
get_geo_data(gse_vector="GSE5847")
get_geo_data(gse_vector=["GSE5847", "GSE28750"])

# ❌ Wrong (will be auto-corrected but shows warning)
get_geo_data(gse_vector="gse5847")  # lowercase
```

### Gotcha 2: First Download is Slow

**Be patient!** The first download takes time:

```python
import time

# First download
start = time.time()
get_geo_data(gse_vector="GSE5847", cache_dir="./cache")
first_time = time.time() - start
print(f"First download: {first_time:.1f} seconds")

# Cached download
start = time.time()
get_geo_data(gse_vector="GSE5847", cache_dir="./cache")
cached_time = time.time() - start
print(f"Cached download: {cached_time:.1f} seconds")

# Typical results:
# First download: 180.2 seconds (~3 minutes)
# Cached download: 3.4 seconds
```

### Gotcha 3: Expression Data May Not Be in Log Scale

Always verify and transform if needed:

```python
# Check scale
dataset = meta_object['originalData']['Study_A']
is_log_scale = check_geo_log_scale(dataset)

if not is_log_scale:
    print("⚠️ Data is in linear scale - transforming...")
    # Transform to log2 scale
    dataset['expr'] = np.log2(dataset['expr'] + 1)  # +1 to avoid log(0)
    dataset['exp_comment'] = "Converted to log2 scale"
    print("✅ Transformation complete")
else:
    print("✅ Data already in log scale")
```

## Troubleshooting

### Q: Download fails with "Connection timeout"

**A**: NCBI servers may be temporarily overloaded. Solutions:

1. Wait 5-10 minutes and retry
2. Check [NCBI GEO status](https://www.ncbi.nlm.nih.gov/geo/)
3. Try a different dataset to verify connectivity
4. Use a VPN if institutional firewall blocks NCBI

```python
# Retry with timeout
import time

max_retries = 3
for attempt in range(max_retries):
    try:
        meta_object = get_geo_data(gse_vector="GSE5847")
        break
    except Exception as e:
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 30  # 30, 60, 90 seconds
            print(f"Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        else:
            raise
```

### Q: "Unable to correctly download [dataset]. Dataset will be excluded"

**A**: The GSE ID may be invalid or data format unsupported:

1. Verify GSE ID exists on [GEO website](https://www.ncbi.nlm.nih.gov/geo/)
2. Check if dataset has "Series Matrix File(s)" available
3. Some very old or unusual formats may not be supported

```python
# Verify GSE exists before downloading
def verify_gse_id(gse_id):
    """Check if GSE ID exists on GEO."""
    import requests
    
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        print(f"✅ {gse_id} exists on GEO")
        return True
    else:
        print(f"❌ {gse_id} not found on GEO")
        return False

verify_gse_id("GSE5847")  # Should print ✅
verify_gse_id("GSE99999")  # Should print ❌
```

### Q: Downloaded data has many missing values

**A**: Some platforms/datasets inherently have missing data:

```python
# Check and report missing data
expr = dataset['expr']
missing_pct = (expr.isna().sum().sum() / expr.size) * 100

print(f"Missing data: {missing_pct:.2f}%")

if missing_pct > 10:
    print("⚠️ High missing data rate!")
    print("Options:")
    print("  1. Impute missing values (e.g., KNN imputation)")
    print("  2. Filter genes with >20% missing")
    print("  3. Exclude this dataset from meta-analysis")
```

### Q: Gene keys are all probe IDs, not gene symbols

**A**: This is normal! GEO data uses probe IDs. Convert to gene symbols:

```python
# Probe IDs are the raw identifiers
print(f"Raw keys: {dataset['keys'][:5]}")
# Output: ['AFFX-BioB-5_at', 'AFFX-BioB-M_at', ...]

# Use gene annotation (next tutorial) to convert to symbols
# See: tutorials/gene_annotation.md
```

## Next Steps

Now that you can download GEO data, proceed to:

1. **[Gene Annotation Tutorial](gene_annotation.md)** – Convert probe IDs to gene symbols
2. **[Quality Control Tutorial](quality_control.md)** – Comprehensive data quality checks
3. **[Complete Workflow Tutorial](complete_workflow.md)** – See data acquisition in context

### Advanced Topics

- **Custom preprocessing pipelines** – Add your own normalization steps
- **Batch effect correction** – Handle technical variation
- **Multi-organism support** – Download data for mouse, rat, etc.
- **Private GEO data** – Access restricted datasets with credentials

### Key Takeaways

✅ Use `get_geo_data()` for simple downloads  
✅ Always specify `cache_dir` to avoid re-downloading  
✅ Apply `q_norm=True` for cross-platform compatibility  
✅ Verify data is in log scale with `check_geo_log_scale()`  
✅ Check platform compatibility before meta-analysis  
✅ Handle download failures gracefully with error handling  

---

**Questions?** Check the [API Reference](../api/data_loading.md) or open an issue on GitHub.
