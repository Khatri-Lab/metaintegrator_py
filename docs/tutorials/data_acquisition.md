# Data Acquisition from GEO

Learn how to download and prepare gene expression datasets from NCBI's Gene Expression Omnibus (GEO) using metaIntegrator-py's `DataLoaderFactory` — with real NCBI GEO fetching, local caching, and gene annotation built in.

## Goal

Download gene expression datasets from GEO and organize them into the
MetaIntegrator-compatible dict format used by `MetaAnalyzer`.

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
from preprocessing.loader_factory import DataLoaderFactory

import pandas as pd
import numpy as np

print("✅ Data loader imported successfully")
```

**What is GEO?**
The Gene Expression Omnibus (GEO) is NCBI's public repository of gene expression data. Each dataset has a unique GSE (GEO Series) identifier like `GSE5847`.

### Step 2: Download a Single Dataset

Let's start with downloading one small dataset:

```python
# Create a loader — strategy='r_parity' (an alias for 'hybrid') is the
# recommended, production strategy: it matches R MetaIntegrator's output
print("📥 Downloading dataset from GEO...")

loader = DataLoaderFactory.create_geo_loader(
    strategy="r_parity",
    cache_dir="./geo_cache",
)
datasets = loader.load_multiple(["GSE5847"])

print("✅ Download complete!")

# Inspect the structure — load_multiple() always returns a list of
# ProcessedDataset objects, one per platform
print(f"\nDatasets downloaded: {[ds.geo_id for ds in datasets]}")
```

**Expected output:**
```
📥 Downloading dataset from GEO...
✅ Download complete!

Datasets downloaded: ['GSE5847_GPL96']
```

**What happened?**
1. Contacted NCBI GEO servers
2. Downloaded the Series Matrix file
3. Extracted expression data and sample metadata
4. Mapped probe IDs to gene symbols (via `GeneAnnotator`, with Ensembl/GenBank filtering) — this happens automatically, as part of loading
5. Cached files locally at `./geo_cache/`

### Step 3: Download Multiple Datasets

Meta-analysis requires multiple studies. Let's download several:

```python
# Download multiple datasets
gse_ids = ["GSE5847", "GSE28750", "GSE40012"]

print(f"📥 Downloading {len(gse_ids)} datasets...")

loader = DataLoaderFactory.create_geo_loader(strategy="r_parity", cache_dir="./geo_cache")
datasets = loader.load_multiple(gse_ids, stop_on_error=False)

# Display information
print(f"\n✅ Downloaded {len(datasets)} datasets:")
for ds in datasets:
    n_probes, n_samples = ds.expr.shape
    print(f"   • {ds.geo_id}: {n_probes:,} probes × {n_samples} samples")
```

**Expected output:**
```
📥 Downloading 3 datasets...

✅ Downloaded 3 datasets:
   • GSE5847_GPL96: 22,283 probes × 50 samples
   • GSE28750_GPL570: 54,675 probes × 30 samples
   • GSE40012_GPL570: 54,675 probes × 85 samples
```

**Performance Note:**
First download: ~2-3 minutes per dataset (network dependent)
Subsequent runs against the same `cache_dir`: < 5 seconds (uses cached data)

!!! tip "stop_on_error"
    By default (`stop_on_error=False`), `load_multiple()` already skips
    datasets that fail (bad GSE ID, network issue, etc.) and returns the
    ones that succeeded. Pass `stop_on_error=True` if you want it to raise
    on the first failure instead.

### Step 4: Inspect Dataset Structure

Let's examine what we downloaded:

```python
# Pick one dataset to inspect
ds = datasets[0]

print(f"📊 Inspecting dataset: {ds.geo_id}\n")

# ProcessedDataset has exactly five fields: geo_id, expr, pheno, keys, metadata
print(f"📈 Expression Matrix:")
print(f"   Shape: {ds.expr.shape}")
print(f"   Type: {type(ds.expr)}")
print(f"\n   First 5 probes × 3 samples:")
print(ds.expr.iloc[:5, :3])

# Check gene identifiers — these are gene symbols already (annotation
# happened during loading), 'NA' for probes that couldn't be mapped
print(f"\n🧬 Gene Identifiers (first 10):")
print(ds.keys[:10])

# Examine sample metadata
print(f"\n👥 Sample Metadata:")
print(f"   Available: {ds.pheno.shape[0]} samples")
print(f"   Columns: {list(ds.pheno.columns[:5])}...")

# Platform / source info
print(f"\nℹ️  Metadata: {ds.metadata}")
```

**Expected output:**
```
📊 Inspecting dataset: GSE5847_GPL96

📈 Expression Matrix:
   Shape: (22283, 50)
   Type: <class 'pandas.core.frame.DataFrame'>

   First 5 probes × 3 samples:
                GSM123456  GSM123457  GSM123458
AFFX-BioB-5_at      7.234      7.189      7.301
AFFX-BioB-M_at      8.456      8.423      8.489
1007_s_at           9.123      9.087      9.156
...

🧬 Gene Identifiers (first 10):
['NA', 'NA', 'DDR1', 'RFC2', ...]

👥 Sample Metadata:
   Available: 50 samples
   Columns: ['title', 'geo_accession', 'source_name_ch1', ...]

ℹ️  Metadata: {'source': 'geo', 'platform': 'GPL96', 'handler': 'HybridMicroarrayHandler'}
```

**Note:** `keys` is already gene symbols, not probe IDs — control probes
(`AFFX-*`) and unmappable probes come through as `'NA'`. See the
[Gene Annotation Tutorial](gene_annotation.md) for how this mapping works
under the hood and how to run it directly if you need to.

### Step 5: Verify Data Quality

Check critical data quality metrics:

```python
def check_dataset_quality(ds):
    """Quick quality check for a downloaded ProcessedDataset."""

    print(f"🔍 Quality Check: {ds.geo_id}")
    print("-" * 60)

    expr = ds.expr

    # Check 1: Missing values
    total_values = expr.size
    missing_values = expr.isna().sum().sum()
    missing_pct = (missing_values / total_values) * 100

    status_missing = "✅" if missing_pct < 5 else "⚠️"
    print(f"{status_missing} Missing values: {missing_pct:.2f}%")

    # Check 2: Log-scale (heuristic — there's no built-in check).
    # Log2 microarray data is rarely above ~20-25; linear-scale data
    # commonly reaches into the thousands.
    is_log_scale = expr.max().max() < 30
    status_log = "✅" if is_log_scale else "❌"
    scale_type = "log scale" if is_log_scale else "linear scale"
    print(f"{status_log} Data transformation: {scale_type}")

    # Check 3: Value range
    min_val = expr.min().min()
    max_val = expr.max().max()
    mean_val = expr.mean().mean()

    print(f"   Range: [{min_val:.2f}, {max_val:.2f}]")
    print(f"   Mean: {mean_val:.2f}")

    # Check 4: Annotation rate
    n_mapped = sum(1 for k in ds.keys if k != 'NA')
    print(f"   Genes mapped: {n_mapped:,}/{len(ds.keys):,} ({n_mapped / len(ds.keys):.1%})")

    print()

# Check all datasets
for ds in datasets:
    check_dataset_quality(ds)
```

**Expected output:**
```
🔍 Quality Check: GSE5847_GPL96
------------------------------------------------------------
✅ Missing values: 0.03%
✅ Data transformation: log scale
   Range: [2.34, 14.56]
   Mean: 7.82
   Genes mapped: 13,042/22,283 (58.5%)
```

### Step 6: Handle Multi-Platform Datasets

Some GSE IDs contain data from multiple microarray platforms. `load_multiple()`
returns one `ProcessedDataset` per platform automatically:

```python
# Example: a GSE that spans multiple platforms
multi_datasets = loader.load_multiple(["GSE12345"])

print(f"📊 Dataset contains {len(multi_datasets)} platform(s):")
for ds in multi_datasets:
    print(f"   • {ds.geo_id}")

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
- Merge after checking gene-symbol overlap (they're already annotated to symbols, so this is usually just an intersection of `keys`)

### Step 7: Choosing a Loading Strategy

`create_geo_loader()` supports three strategies — pick based on what you need:

```python
# R-parity mode (matches R MetaIntegrator output exactly) — recommended default
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')

# Full SOFT download (all metadata, GEOparse-based) — supports optional
# log_transform / normalize toggles
loader = DataLoaderFactory.create_geo_loader(
    strategy='full', log_transform=True, normalize=True,
)

# Fast exploration mode (subset via Series Matrix — may miss some samples)
loader = DataLoaderFactory.create_geo_loader(strategy='fast')
```

`log_transform` and `normalize` only apply to `'full'`/`'fast'` — the
recommended `'r_parity'`/`'hybrid'` strategy handles this internally and
doesn't expose those toggles.

## Building the Meta-Object

Once you have `ProcessedDataset` objects, convert them into the dict format
`MetaAnalyzer` expects, and assign class labels (see
[Concepts Guide](../user_guide/concepts.md) for the full explanation of why
this is always a separate step):

```python
from metaintegrator.labeling import assign_class_labels

meta_object = {"originalData": {}}
for ds in datasets:
    dataset = {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
               "formattedName": ds.geo_id, "class": None}
    meta_object["originalData"][ds.geo_id] = assign_class_labels(
        dataset, column_name="title", disease_terms=["disease"]
    )
```

## Tips & Gotchas

### Tip 1: Cache Everything

**Always specify `cache_dir`** to avoid re-downloading:

```python
# ✅ Good: Uses cache
loader = DataLoaderFactory.create_geo_loader(strategy="r_parity", cache_dir="./geo_cache")

# ⚠️ Still works, but defaults to ~/.geo_cache — fine, just be intentional about it
loader = DataLoaderFactory.create_geo_loader(strategy="r_parity")
```

### Tip 2: Check Platform Compatibility

Before meta-analysis, verify datasets use compatible platforms:

```python
# Check platforms
for ds in datasets:
    print(f"{ds.geo_id}: {ds.metadata.get('platform', 'Unknown')}")

# All should be same organism and similar technology
# Good: All GPL570 (Affymetrix Human Genome U133 Plus 2.0)
# Bad: Mix of very different array technologies
```

### Tip 3: Handle Download Failures Gracefully

Some datasets may fail to download — use `stop_on_error=False` and diff
the requested vs. returned IDs:

```python
gse_list = ["GSE5847", "GSE99999", "GSE28750"]  # GSE99999 doesn't exist

loader = DataLoaderFactory.create_geo_loader(strategy="r_parity", cache_dir="./geo_cache")
datasets = loader.load_multiple(gse_list, stop_on_error=False)

succeeded = {ds.geo_id.split('_')[0] for ds in datasets}
failed = [gse for gse in gse_list if gse not in succeeded]

print(f"✅ Successfully downloaded: {len(datasets)} dataset(s)")
if failed:
    print(f"⚠️ Failed: {failed}")
```

### Gotcha 1: Case-Sensitive-Looking GSE IDs

Always write GSE IDs in their canonical uppercase form (`GSE5847`, not
`gse5847`) to avoid surprises with caching and NCBI lookups.

### Gotcha 2: First Download is Slow

**Be patient!** The first download takes time:

```python
import time

loader = DataLoaderFactory.create_geo_loader(strategy="r_parity", cache_dir="./cache")

# First download
start = time.time()
loader.load_multiple(["GSE5847"])
print(f"First download: {time.time() - start:.1f} seconds")

# Cached download
start = time.time()
loader.load_multiple(["GSE5847"])
print(f"Cached download: {time.time() - start:.1f} seconds")

# Typical results:
# First download: ~180 seconds (~3 minutes)
# Cached download: ~3 seconds
```

### Gotcha 3: Expression Data May Not Be in Log Scale

Always verify and transform if needed (there's no built-in check — use a
range heuristic):

```python
ds = datasets[0]
is_log_scale = ds.expr.max().max() < 30

if not is_log_scale:
    print("⚠️ Data is in linear scale - transforming...")
    log_expr = np.log2(ds.expr + 1)  # +1 to avoid log(0)
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
import time

max_retries = 3
loader = DataLoaderFactory.create_geo_loader(strategy="r_parity")
for attempt in range(max_retries):
    try:
        datasets = loader.load_multiple(["GSE5847"])
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

**A**: The GSE ID may be invalid or the data format unsupported:

1. Verify the GSE ID exists on the [GEO website](https://www.ncbi.nlm.nih.gov/geo/)
2. Check whether the dataset has "Series Matrix File(s)" available
3. Some very old or unusual formats may not be supported

### Q: Downloaded data has many missing values

**A**: Some platforms/datasets inherently have missing data:

```python
expr = ds.expr
missing_pct = (expr.isna().sum().sum() / expr.size) * 100
print(f"Missing data: {missing_pct:.2f}%")

if missing_pct > 10:
    print("⚠️ High missing data rate!")
    print("Options:")
    print("  1. Impute missing values (e.g., KNN imputation)")
    print("  2. Filter genes with >20% missing")
    print("  3. Exclude this dataset from meta-analysis")
```

### Q: Many gene keys are 'NA'

**A**: This is expected — control probes and probes without a mappable
symbol are set to `'NA'` (matching R MetaIntegrator's behavior). See
[Gene Annotation Tutorial](gene_annotation.md) for details on what gets
filtered and why.

## Next Steps

Now that you can download GEO data, proceed to:

1. **[Gene Annotation Tutorial](gene_annotation.md)** – How probe-to-symbol mapping actually works
2. **[Quality Control Tutorial](quality_control.md)** – Comprehensive data quality checks
3. **[Complete Workflow Tutorial](complete_workflow.md)** – See data acquisition in context

### Key Takeaways

✅ Use `DataLoaderFactory.create_geo_loader().load_multiple()` for downloads
✅ Gene annotation happens automatically as part of loading — no separate call needed
✅ Always specify `cache_dir` to avoid re-downloading
✅ `'r_parity'`/`'hybrid'` is the recommended, production strategy
✅ There's no built-in log-scale check — use a simple range heuristic
✅ Use `stop_on_error=False` to skip datasets that fail to download

---

**Questions?** Check the [API Reference](../api/preprocessing.md) or open an issue on GitHub.
