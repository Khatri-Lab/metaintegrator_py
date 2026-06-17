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
# Core dependencies (installed automatically)
pip install pandas>=1.5.0
pip install numpy>=1.23.0
pip install requests>=2.28.0
pip install scipy>=1.9.0
```

### Optional Dependencies

```bash
# For advanced features
pip install bioservices  # Enhanced annotation
pip install GEOparse      # Alternative GEO parser
```

## Configuration

### Environment Variables

Configure GEO data loader behavior:

```bash
# Set cache directory (default: ~/.geo_cache)
export GEO_CACHE_DIR="/path/to/cache"

# Set download timeout (default: 300 seconds)
export GEO_DOWNLOAD_TIMEOUT=600

# Enable debug logging
export GEO_DEBUG=1
```

### Python Configuration

```python
from preprocessing.geo_data_loader import GEODataLoader

# Configure loader with custom settings
loader = GEODataLoader(
    cache_dir="./my_geo_cache",  # Custom cache location
)

# Access configuration
print(f"Cache directory: {loader.cache_dir}")
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
from preprocessing.geo_data_loader import get_geo_data

# Download one dataset
meta_object = get_geo_data(
    gse_vector="GSE5847",           # GSE accession
    formatted_names=["My_Study"],   # Custom name
    q_norm=True,                    # Apply quantile normalization
    cache_dir="./geo_cache"         # Cache location
)

# Access dataset
dataset = meta_object['originalData']['GSE5847']
print(f"Downloaded: {dataset['expr'].shape[0]} genes × {dataset['expr'].shape[1]} samples")
```

#### Multiple Datasets

```python
# Download multiple datasets for meta-analysis
gse_ids = ["GSE5847", "GSE28750", "GSE40012"]
study_names = ["Breast_Cancer", "Sepsis", "TB_Infection"]

meta_object = get_geo_data(
    gse_vector=gse_ids,
    formatted_names=study_names,
    q_norm=True,
    cache_dir="./geo_cache"
)

print(f"Downloaded {len(meta_object['originalData'])} datasets:")
for name in meta_object['originalData'].keys():
    print(f"  • {name}")
```

### Data Processing Pipeline

```python
from preprocessing.geo_data_loader import get_geo_data, check_geo_log_scale
from preprocessing.geo_annotation_engine import annotate_geo_dataset, GeneAnnotationEngine

# 1. Download data
print("📥 Downloading GEO data...")
meta_object = get_geo_data(
    gse_vector=["GSE5847", "GSE28750"],
    q_norm=True,
    cache_dir="./geo_cache"
)

# 2. Verify log scale
print("\n🔬 Verifying log-scale transformation...")
for name, dataset in meta_object['originalData'].items():
    is_log = check_geo_log_scale(dataset)
    print(f"  {name}: {'✅ log scale' if is_log else '❌ linear scale'}")

# 3. Annotate genes
print("\n🧬 Annotating genes...")
engine = GeneAnnotationEngine()
for name, dataset in meta_object['originalData'].items():
    meta_object['originalData'][name] = annotate_geo_dataset(dataset, engine)
    report = dataset.get('annotation_report', {})
    if report:
        print(f"  {name}: {report['mapped_identifiers']:,} genes mapped ({report['mapping_rate']:.1%})")

print("\n✅ Data ready for meta-analysis")
```

### Platform-Specific Handling

```python
# Check platforms
platforms = {}
for name, dataset in meta_object['originalData'].items():
    platform = dataset.get('platform', 'Unknown')
    platforms[name] = platform
    print(f"{name}: {platform}")

# Filter by platform
affymetrix_datasets = {
    name: dataset 
    for name, dataset in meta_object['originalData'].items()
    if 'GPL570' in dataset.get('platform', '')
}

print(f"\nAffymetrix GPL570 datasets: {len(affymetrix_datasets)}")
```

### Working with Series Matrix Files

GEO datasets come as "Series Matrix" files containing:

```python
# Examine what gets downloaded
dataset = meta_object['originalData']['GSE5847']

# Available data
print("Data fields:")
for key in dataset.keys():
    if key == 'expr':
        print(f"  • expr: {dataset['expr'].shape} DataFrame")
    elif key == 'pheno':
        print(f"  • pheno: {dataset['pheno'].shape} DataFrame")
    elif key == 'keys':
        print(f"  • keys: {len(dataset['keys'])} identifiers")
    else:
        print(f"  • {key}: {type(dataset[key]).__name__}")

# Phenotype information
if 'pheno' in dataset:
    print(f"\nPhenotype columns:")
    print(dataset['pheno'].columns.tolist()[:10])  # First 10 columns
```

## Advanced Topics

### Custom Download Location

```python
# Download to specific directory
import os

download_dir = "./geo_downloads"
os.makedirs(download_dir, exist_ok=True)

meta_object = get_geo_data(
    gse_vector="GSE5847",
    cache_dir=download_dir,
    dest_dir=download_dir  # Save raw files here
)
```

### Handling Multi-Platform Datasets

Some GSE IDs contain data from multiple platforms:

```python
# Example: GSE that uses multiple platforms
meta_object = get_geo_data(gse_vector="GSE12345")

# Check number of datasets returned
n_datasets = len(meta_object['originalData'])
print(f"Number of platforms: {n_datasets}")

# Each platform gets its own entry
for name in meta_object['originalData'].keys():
    print(f"  • {name}")
    # Output might be:
    #   • GSE12345_GPL570  (Affymetrix)
    #   • GSE12345_GPL96   (Affymetrix)

# Access specific platform
gpl570_data = meta_object['originalData']['GSE12345_GPL570']
```

### Custom Normalization

```python
from sklearn.preprocessing import quantile_transform

# Download without built-in normalization
meta_object = get_geo_data(
    gse_vector="GSE5847",
    q_norm=False  # Skip quantile normalization
)

# Apply custom normalization
dataset = meta_object['originalData']['GSE5847']
expr = dataset['expr'].values

# Custom quantile normalization
normalized = quantile_transform(
    expr.T,  # Transpose for sample-wise normalization
    output_distribution='normal',
    n_quantiles=1000
).T

# Update dataset
import pandas as pd
dataset['expr'] = pd.DataFrame(
    normalized,
    index=dataset['expr'].index,
    columns=dataset['expr'].columns
)
dataset['exp_comment'] = 'Custom quantile normalization applied'
```

### Batch Download with Error Handling

```python
import warnings

def download_geo_batch(gse_list, cache_dir="./geo_cache"):
    """Download multiple GSE IDs with error handling."""
    
    successful = {}
    failed = []
    
    for gse_id in gse_list:
        try:
            print(f"Downloading {gse_id}...")
            
            # Attempt download
            result = get_geo_data(
                gse_vector=gse_id,
                cache_dir=cache_dir
            )
            
            # Store if successful
            successful[gse_id] = result
            print(f"  ✅ Success")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            failed.append((gse_id, str(e)))
    
    # Summary
    print(f"\n📊 Results:")
    print(f"  Successful: {len(successful)}/{len(gse_list)}")
    print(f"  Failed: {len(failed)}/{len(gse_list)}")
    
    if failed:
        print(f"\nFailed downloads:")
        for gse_id, error in failed:
            print(f"  • {gse_id}: {error}")
    
    return successful, failed

# Use it
gse_list = ["GSE5847", "GSE28750", "GSE99999", "GSE40012"]
successful, failed = download_geo_batch(gse_list)
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

# Later use (offline)
with open('meta_object.pkl', 'rb') as f:
    meta_object = pickle.load(f)
```

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

# Next download will fetch latest version
meta_object = get_geo_data("GSE5847", cache_dir=cache_dir)
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
for attempt in range(max_retries):
    try:
        meta_object = get_geo_data("GSE5847")
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

**Cause:** Data needs log transformation for meta-analysis

**Solution:**
```python
from preprocessing.geo_data_loader import check_geo_log_scale
import numpy as np

dataset = meta_object['originalData']['GSE5847']

if not check_geo_log_scale(dataset):
    print("⚠️ Converting to log scale...")
    
    # Add small constant to avoid log(0)
    expr = dataset['expr']
    min_val = expr.min().min()
    
    if min_val < 0:
        expr = expr - min_val + 1
    
    dataset['expr'] = np.log2(expr + 1)
    dataset['exp_comment'] = 'Converted to log2 scale'
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

for gse_id in gse_list:
    # Download one at a time
    meta_obj = get_geo_data(gse_id)
    
    # Process immediately
    # ... analysis code ...
    
    # Save results
    import pickle
    with open(f'{gse_id}_processed.pkl', 'wb') as f:
        pickle.dump(meta_obj, f)
    
    # Clear from memory
    del meta_obj
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
