# GEO Integration

This guide explains how to download, process, and integrate gene expression data from NCBI's Gene Expression Omnibus (GEO) using metaIntegrator-py's production-grade GEO data loader.

## Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Configuration](#configuration)
- [Core Usage](#core-usage)
- [Advanced Topics](#advanced-topics)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Appendix](#appendix)

## Introduction

!!! tip "Updated for OO Redesign"
    GEO data loading now uses the `DataLoaderFactory` with pluggable
    Source/Handler strategies. See the [Architecture Overview](../architecture.md)
    and [Preprocessing API](../api/preprocessing.md) for the full design.

    **Quick start:**
    ```python
    from preprocessing.loader_factory import DataLoaderFactory
    loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
    datasets = loader.load_multiple(['GSE11909', 'GSE50635'])
    ```

The Gene Expression Omnibus (GEO) is NCBI's public repository containing over 100,000 gene expression datasets. MetaIntegrator-py provides seamless integration with GEO, enabling automated download, preprocessing, and organization of microarray data for meta-analysis.

**Key features:**
- Direct download from NCBI GEO servers
- Automatic platform detection and annotation parsing
- Built-in quality control and log-scale verification
- Intelligent caching for offline work
- R-compatible quantile normalization
- Support for multi-platform datasets

## Installation

### Standard Installation

```bash
# Install metaIntegrator-py with GEO support (included by default)
git clone https://github.com/Khatri-Lab/metaintegrator_py.git

# Verify installation
python -c "from preprocessing.loader_factory import DataLoaderFactory; print('✅ GEO integration ready')"
```

### Dependencies

GEO integration requires:

```bash
# Core dependencies (installed automatically via requirements.txt)
pip install pandas>=1.5.0
pip install numpy>=1.21.0
pip install requests>=2.25.0
pip install scipy>=1.9.0
pip install GEOparse>=2.0.3
```

## Configuration

There are no environment variables for configuring the GEO loader: all
configuration is passed explicitly to `DataLoaderFactory.create_geo_loader()`.

### Python Configuration

```python
from preprocessing.loader_factory import DataLoaderFactory

# Configure loader with custom settings
loader = DataLoaderFactory.create_geo_loader(
    strategy="r_parity",           # 'hybrid'/'r_parity' (recommended), 'full', or 'fast'
    cache_dir="./my_geo_cache",    # Custom cache location
)
```

### Cache Management

```python
from pathlib import Path
import shutil

# Check cache size
cache_dir = Path.home() / ".geo_cache"
if cache_dir.exists():
    size_mb = sum(f.stat().st_size for f in cache_dir.rglob('*')) / 1024 / 1024
    print(f"Cache size: {size_mb:.1f} MB")

# Clear cache
def clear_geo_cache(cache_dir):
    """Remove all cached GEO data."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
        print("✅ Cache cleared")

# clear_geo_cache(cache_dir)  # Uncomment to clear
```

## Core Usage

### Basic Data Download

#### Single Dataset

```python
from preprocessing.loader_factory import DataLoaderFactory

# Download one dataset
loader = DataLoaderFactory.create_geo_loader(strategy="r_parity", cache_dir="./geo_cache")
datasets = loader.load_multiple(["GSE5847"])

# Access dataset (load_multiple always returns a list)
ds = datasets[0]
print(f"Downloaded: {ds.expr.shape[0]} genes × {ds.expr.shape[1]} samples")
```

#### Multiple Datasets

```python
# Download multiple datasets for meta-analysis
gse_ids = ["GSE5847", "GSE28750", "GSE40012"]

loader = DataLoaderFactory.create_geo_loader(strategy="r_parity", cache_dir="./geo_cache")
datasets = loader.load_multiple(gse_ids, stop_on_error=False)

print(f"Downloaded {len(datasets)} datasets:")
for ds in datasets:
    print(f"  • {ds.geo_id}")
```

### Data Processing Pipeline

Gene annotation (probe → gene symbol) happens automatically inside the handler
during `load_multiple()`: there's no separate annotation step to call for the
standard workflow. Class labels are the one step you always do by hand:

```python
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.labeling import assign_class_labels

# 1. Download data (annotation happens inside the handler)
print("📥 Downloading GEO data...")
loader = DataLoaderFactory.create_geo_loader(strategy="r_parity", cache_dir="./geo_cache")
datasets = loader.load_multiple(["GSE5847", "GSE28750"])

# 2. Build the meta-object and assign class labels
print("\n🏷️  Assigning class labels...")
meta_object = {"originalData": {}}
for ds in datasets:
    dataset = {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
               "formattedName": ds.geo_id, "class": None}
    meta_object["originalData"][ds.geo_id] = assign_class_labels(
        dataset, column_name="title", disease_terms=["disease"]
    )
    n_mapped = sum(1 for k in ds.keys if k != "NA")
    print(f"  {ds.geo_id}: {n_mapped:,}/{len(ds.keys):,} probes mapped to a gene symbol")

print("\n✅ Data ready for meta-analysis")
```

!!! tip "Why the same `column_name` value works across every GEO dataset"
    `ds.pheno` always has the same five standardized columns —
    `title`, `source_name`, `organism`, `characteristics`, and
    `description`: no matter what the original GEO submitter called
    them (e.g. a raw `characteristics_ch1` field). Each loading
    strategy's handler builds `pheno` from GEOparse's per-sample
    metadata and normalizes the field names during loading, so
    `column_name` in `assign_class_labels()` (and `possible_columns` in
    a CLI config file) should reference one of these five standardized
    names rather than a dataset's raw GEO field name. That's what lets
    the same `column_name="title"` call, or a CLI config that lists
    `characteristics` as a candidate column, work unchanged across
    different GEO datasets.

### Platform-Specific Handling

Platform lives on `ProcessedDataset.metadata`, before it's converted into the
analysis dict (which doesn't carry platform info):

```python
# Check platforms
platforms = {ds.geo_id: ds.metadata.get('platform', 'Unknown') for ds in datasets}
for name, platform in platforms.items():
    print(f"{name}: {platform}")

# Filter by platform
affymetrix_datasets = [ds for ds in datasets if 'GPL570' in ds.metadata.get('platform', '')]

print(f"\nAffymetrix GPL570 datasets: {len(affymetrix_datasets)}")
```

### Working with Series Matrix Files

GEO datasets come as "Series Matrix" files. After loading, each `ProcessedDataset`
exposes the parsed result:

```python
# Examine what gets downloaded
ds = datasets[0]

print(f"geo_id: {ds.geo_id}")
print(f"  • expr: {ds.expr.shape} DataFrame")
print(f"  • pheno: {ds.pheno.shape} DataFrame")
print(f"  • keys: {len(ds.keys)} identifiers")
print(f"  • metadata: {ds.metadata}")

# Phenotype information
print(f"\nPhenotype columns:")
print(ds.pheno.columns.tolist()[:10])  # First 10 columns
```

## Advanced Topics

### Handling Multi-Platform Datasets

Some GSE IDs contain data from multiple platforms. `load_multiple()` returns
one `ProcessedDataset` per platform, with `geo_id` disambiguated accordingly:

```python
# Example: GSE that uses multiple platforms
datasets = loader.load_multiple(["GSE12345"])

print(f"Number of platforms: {len(datasets)}")
for ds in datasets:
    print(f"  • {ds.geo_id}")
    # Output might be:
    #   • GSE12345_GPL570  (Affymetrix)
    #   • GSE12345_GPL96   (Affymetrix)

# Access a specific platform
gpl570_data = next(ds for ds in datasets if ds.geo_id == "GSE12345_GPL570")
```

### Custom Normalization

The `'hybrid'`/`'r_parity'` strategy (recommended) handles scaling internally
and doesn't expose a normalization toggle. If you need a different
normalization, apply it yourself after loading:

```python
from sklearn.preprocessing import quantile_transform
import pandas as pd

ds = datasets[0]
expr = ds.expr.values

# Custom quantile normalization
normalized = quantile_transform(
    expr.T,  # Transpose for sample-wise normalization
    output_distribution='normal',
    n_quantiles=1000
).T

# Build the analysis dict with the normalized matrix
dataset = {
    "expr": pd.DataFrame(normalized, index=ds.expr.index, columns=ds.expr.columns),
    "pheno": ds.pheno, "keys": ds.keys, "formattedName": ds.geo_id,
    "class": None, "exp_comment": "Custom quantile normalization applied",
}
```

### Batch Download with Error Handling

`load_multiple(..., stop_on_error=False)` already skips datasets that fail to
download rather than aborting the whole batch: wrap it if you also want a
structured success/failure report:

```python
def download_geo_batch(gse_list, cache_dir="./geo_cache"):
    """Download multiple GSE IDs, reporting which ones failed."""
    loader = DataLoaderFactory.create_geo_loader(strategy="r_parity", cache_dir=cache_dir)
    datasets = loader.load_multiple(gse_list, stop_on_error=False)

    succeeded_ids = {ds.geo_id.split('_')[0] for ds in datasets}
    failed = [gse for gse in gse_list if gse not in succeeded_ids]

    print(f"\n📊 Results:")
    print(f"  Successful: {len(datasets)} dataset(s) from {len(succeeded_ids)}/{len(gse_list)} GSE IDs")
    if failed:
        print(f"  Failed GSE IDs: {failed}")

    return datasets, failed

# Use it
gse_list = ["GSE5847", "GSE28750", "GSE99999", "GSE40012"]
datasets, failed = download_geo_batch(gse_list)
```

### Programmatic Search for Datasets

```python
import requests

def search_geo(query, max_results=10):
    """Search GEO for datasets matching query."""
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    params = {
        'db': 'gds',
        'term': query,
        'retmax': max_results,
        'retmode': 'json'
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if 'esearchresult' in data and 'idlist' in data['esearchresult']:
        ids = data['esearchresult']['idlist']
        print(f"Found {len(ids)} datasets:")
        
        # Get details for each ID
        for geo_id in ids:
            # Would need additional esummary call for details
            print(f"  • GDS{geo_id}")
        
        return ids
    else:
        print("No results found")
        return []

# Example: Search for breast cancer datasets
# search_geo("breast cancer human")
```

### Direct Series Matrix Download

```python
import requests
import gzip
from pathlib import Path

def download_series_matrix(gse_id, output_dir="./downloads"):
    """Download Series Matrix file directly from GEO."""
    
    # Construct URL
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:-3]}nnn/{gse_id}/matrix/"
    
    # Try to get file listing
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse directory listing for .txt.gz file
        # (Simplified - real implementation would parse HTML)
        filename = f"{gse_id}_series_matrix.txt.gz"
        file_url = url + filename
        
        print(f"Downloading {filename}...")
        file_response = requests.get(file_url)
        
        if file_response.status_code == 200:
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(file_response.content)
            
            print(f"✅ Downloaded to {output_path}")
            return output_path
        else:
            print(f"❌ File not found: {file_url}")
            return None
    else:
        print(f"❌ Cannot access GEO FTP for {gse_id}")
        return None

# Example usage (commented out - direct FTP access)
# download_series_matrix("GSE5847")
```

## FAQ

### Q: How long does the first download take?

**A:** Download time depends on dataset size and network speed:
- Small dataset (< 50 samples): 30 seconds - 2 minutes
- Medium dataset (50-200 samples): 2-5 minutes
- Large dataset (> 200 samples): 5-15 minutes

Subsequent runs use cached data (< 5 seconds).

### Q: Can I download data from GEO without internet?

**A:** No, first download requires internet. However:
1. First run downloads and caches data
2. Subsequent runs use cache (offline capable)
3. Save meta-objects with `pickle` for permanent offline storage

```python
import pickle

# After downloading (online)
with open('meta_object.pkl', 'wb') as f:
    pickle.dump(meta_object, f)

# Later use (offline): no re-download needed at all
with open('meta_object.pkl', 'rb') as f:
    meta_object = pickle.load(f)
```

Even without pickling, re-running `load_multiple()` against the same `cache_dir`
is itself offline-capable: cached files are read from disk, not re-downloaded.

### Q: What GEO accession types are supported?

**A:** Currently supported:
- ✅ **GSE**: GEO Series (multiple samples from one study)
- ❌ **GSM**: Individual samples (not supported directly)
- ❌ **GPL**: Platform records (used internally, not for download)
- ❌ **GDS**: Curated datasets (different format, not supported)

Use GSE accessions for meta-analysis.

### Q: How do I find relevant GSE IDs?

**A:** Several options:
1. **GEO website**: https://www.ncbi.nlm.nih.gov/geo/
2. **Literature**: Check methods sections of papers
3. **Programmatic search**: Use NCBI E-utilities (see Advanced Topics)
4. **ArrayExpress**: https://www.ebi.ac.uk/arrayexpress/

### Q: What if a GSE has updated since I cached it?

**A:** Clear cache to force re-download:
```python
import shutil
from pathlib import Path

cache_dir = Path("./geo_cache")
gse_cache = cache_dir / "GSE5847"

if gse_cache.exists():
    shutil.rmtree(gse_cache)
    print(f"Cleared cache for GSE5847")

# Next load_multiple() call against this cache_dir will re-download
loader = DataLoaderFactory.create_geo_loader(strategy="r_parity", cache_dir=cache_dir)
datasets = loader.load_multiple(["GSE5847"])
```

## Troubleshooting

### Issue: "Connection timeout" or "Network error"

**Cause:** NCBI servers temporarily unavailable or network issues

**Solutions:**
1. Check internet connection
2. Wait 5-10 minutes and retry
3. Check [NCBI status](https://www.ncbi.nlm.nih.gov/)
4. Try different dataset to verify connectivity

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
            wait_time = (attempt + 1) * 30
            print(f"Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        else:
            raise
```

### Issue: "Unable to correctly download [GSE]. Dataset will be excluded"

**Cause:** GSE might not exist, be private, or have incompatible format

**Solutions:**
1. Verify GSE exists on GEO website
2. Check if dataset has "Series Matrix File(s)"
3. Ensure GSE is public (not private/controlled access)

```python
import requests

def verify_gse_exists(gse_id):
    """Check if GSE exists on GEO."""
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        print(f"✅ {gse_id} exists on GEO")
        return True
    else:
        print(f"❌ {gse_id} not found")
        return False

verify_gse_exists("GSE5847")  # Should return True
verify_gse_exists("GSE99999")  # Should return False
```

### Issue: "Expression data not in log scale"

**Cause:** Data needs log transformation for meta-analysis. There's no built-in
`check_geo_log_scale()` helper: a simple range heuristic works well in practice
(log₂ microarray data is typically in the 0–20 range; linear-scale data often
reaches into the thousands):

**Solution:**
```python
import numpy as np

ds = datasets[0]
expr = ds.expr

is_log_scale = expr.max().max() < 30  # heuristic: log2 data rarely exceeds ~20-25

if not is_log_scale:
    print("⚠️ Converting to log scale...")

    # Add small constant to avoid log(0)
    min_val = expr.min().min()
    if min_val < 0:
        expr = expr - min_val + 1

    log_expr = np.log2(expr + 1)
    print("✅ Transformation complete")
```

### Issue: "Out of memory error" with large datasets

**Cause:** Insufficient RAM for large expression matrices

**Solutions:**
1. Process one dataset at a time
2. Filter genes before analysis
3. Use smaller datasets for testing
4. Increase system RAM or swap space

```python
# Process datasets sequentially
gse_list = ["GSE5847", "GSE28750", "GSE40012"]
loader = DataLoaderFactory.create_geo_loader(strategy="r_parity")

for gse_id in gse_list:
    # Download one at a time
    ds = loader.load_multiple([gse_id])[0]

    # Process immediately
    # ... analysis code ...

    # Save results
    import pickle
    with open(f'{gse_id}_processed.pkl', 'wb') as f:
        pickle.dump(ds, f)

    # Clear from memory
    del ds
```

## Appendix

### Supported Platforms

Common microarray platforms that work well:

| Platform | Manufacturer | Genes | Notes |
|----------|--------------|-------|-------|
| GPL570 | Affymetrix | ~54k | Human Genome U133 Plus 2.0 |
| GPL96 | Affymetrix | ~22k | Human Genome U133A |
| GPL571 | Affymetrix | ~54k | Human Genome U133A 2.0 |
| GPL6244 | Affymetrix | ~33k | Affymetrix Human Gene 1.0 ST |
| GPL6947 | Illumina | ~48k | Human HT-12 V3.0 |
| GPL10558 | Illumina | ~48k | Human HT-12 V4.0 |

### GEO File Structure

```
GEO Series Matrix File (compressed):
GSE5847_series_matrix.txt.gz

Uncompressed content:
!Series_title: "Study of breast cancer"
!Series_summary: "We analyzed..."
!Sample_geo_accession: "GSM123456" "GSM123457" ...
!Sample_title: "Control 1" "Case 1" ...
^PLATFORM = GPL570
#ID_REF = probe IDs
GSM123456 GSM123457 GSM123458 ...  <- Sample IDs
1007_s_at 7.234 8.456 6.789 ...    <- Expression values
1053_at   9.123 8.765 9.234 ...
...
```

### Related Documentation

- [Data Acquisition Tutorial](../tutorials/data_acquisition.md) - Step-by-step guide
- [Data Structures](data_structures.md) - Meta-object format
- [Gene Annotation](../tutorials/gene_annotation.md) - Symbol mapping
- [API Reference](../api/data_loading.md) - Function signatures

### External Resources

- [GEO Homepage](https://www.ncbi.nlm.nih.gov/geo/)
- [GEO Help](https://www.ncbi.nlm.nih.gov/geo/info/)
- [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [GEOquery Bioconductor](https://bioconductor.org/packages/release/bioc/html/GEOquery.html) (R)

---

**Last updated:** 2026-04-21  
**Version:** 1.0  
**Maintainer:** MetaIntegrator-py Team
