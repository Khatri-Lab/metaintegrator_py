# Data Structures

This guide documents the data structures used throughout metaIntegrator-py, including the meta-object format, dataset organization, and result structures. Understanding these structures is essential for working with the package programmatically and troubleshooting issues.

## Contents

- [Introduction](#introduction)
- [Meta-Object Structure](#meta-object-structure)
- [Dataset Structure](#dataset-structure)
- [Analysis Results](#analysis-results)
- [Annotation Structures](#annotation-structures)
- [Working with Data Structures](#working-with-data-structures)
- [Advanced Topics](#advanced-topics)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Appendix](#appendix)

## Introduction

MetaIntegrator-py organizes gene expression data and results in hierarchical structures
compatible with both R MetaIntegrator and Python data science workflows.

Since the OO redesign, the preprocessing layer uses **dataclass models**
(`ProcessedDataset`, `RawDataset`) while the analysis layer uses **dictionary
structures** matching R's list-based meta-objects.
See the [Architecture Overview](../architecture.md) for the full data flow.

## ProcessedDataset (Preprocessing Output)

The `ProcessedDataset` dataclass is the standard output of all preprocessing handlers:

```python
from preprocessing.base import ProcessedDataset

@dataclass
class ProcessedDataset:
    geo_id: str                         # "GSE11909_GPL96"
    expr: pd.DataFrame                  # probes × samples (float64)
    keys: List[str]                     # gene symbol per probe
    pheno: pd.DataFrame                 # sample metadata (GSM IDs as index)
    class_vector: Optional[np.ndarray]  # 0=control, 1=disease
    platform_id: Optional[str]          # "GPL96"
    annotation_method: Optional[str]    # how genes were mapped

    def to_dict(self) -> Dict:
        """Convert to R-compatible dict for MetaAnalyzer."""
```

**Converting to dict for analysis:**

```python
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(['GSE11909'])

# to_dict() produces the format expected by MetaAnalyzer
meta_object = {
    "originalData": {ds.geo_id: ds.to_dict() for ds in datasets}
}
```

## filter_object (Gene Filtering Output)

Produced by `MetaIntegratorFilter.filter_genes()`, consumed by `ScoreCalculator`
and visualization functions:

```python
filter_object = {
    "posGeneNames":       List[str],    # up-regulated signature genes
    "negGeneNames":       List[str],    # down-regulated signature genes
    "FDRThresh":          float,        # FDR threshold used (e.g. 0.05)
    "effectSizeThresh":   float,        # effect-size threshold (e.g. 0)
    "numberStudiesThresh": int,         # min contributing studies
    "isLeaveOneOut":      bool,         # LOOA applied?
}
```

## Meta-Object Structure

### Overview

The **meta-object** is the top-level container for all datasets and results:

```python
meta_object = {
    'originalData': {
        'Study_A': {...},      # Individual dataset
        'Study_B': {...},      # Individual dataset
        'Study_C': {...}       # Individual dataset
    },
    'metaAnalysis': {...}      # Meta-analysis results (added later)
}
```

### Complete Example

```python
from preprocessing.geo_data_loader import get_geo_data

# Download creates a meta-object
meta_object = get_geo_data(
    gse_vector=["GSE5847", "GSE28750"],
    formatted_names=["Breast_Cancer", "Sepsis"]
)

# Structure
print(meta_object.keys())
# Output: dict_keys(['originalData'])

print(list(meta_object['originalData'].keys()))
# Output: ['Breast_Cancer', 'Sepsis']
```

### Accessing Data

```python
# Get all datasets
datasets = meta_object['originalData']

# Get specific dataset
breast_cancer = meta_object['originalData']['Breast_Cancer']

# Get expression matrix
expr = breast_cancer['expr']

# Get gene identifiers
genes = breast_cancer['keys']

# Get sample metadata
pheno = breast_cancer['pheno']
```

## Dataset Structure

Each dataset within `originalData` has a standardized structure:

### Required Fields

```python
dataset = {
    # Expression data (REQUIRED)
    'expr': pd.DataFrame,      # Genes × Samples expression matrix
    'keys': list,              # Gene identifiers (symbols or IDs)
    'class': np.ndarray,       # Sample labels (0=control, 1=case)
    
    # Metadata (REQUIRED)
    'pheno': pd.DataFrame,     # Sample phenotype/metadata
    
    # Optional fields
    'name': str,               # Dataset identifier
    'formattedName': str,      # Human-readable name
    'platform': str,           # Platform ID (e.g., GPL570)
    
    # Processing comments
    'exp_comment': str,        # Expression processing notes
    'key_comment': str,        # Gene annotation notes
    'load_comment': str        # Data loading notes
}
```

### Expression Matrix (`expr`)

**Type**: `pandas.DataFrame`  
**Shape**: (n_genes, n_samples)  
**Index**: Gene/probe identifiers  
**Columns**: Sample identifiers  
**Values**: Log₂-transformed expression values

```python
# Example
dataset['expr']
#                GSM123456  GSM123457  GSM123458  ...
# 1007_s_at          7.234      7.189      7.301  ...
# 1053_at            8.456      8.423      8.489  ...
# 117_at             9.123      9.087      9.156  ...

# Properties
print(f"Shape: {dataset['expr'].shape}")
# Output: Shape: (54675, 50)  # 54,675 genes × 50 samples

# Data range (should be log scale)
print(f"Min: {dataset['expr'].min().min():.2f}")
print(f"Max: {dataset['expr'].max().max():.2f}")
# Output: Min: 2.34, Max: 14.56 (typical for log₂ scale)
```

### Gene Identifiers (`keys`)

**Type**: `list` or `np.ndarray`  
**Length**: Must match `expr.shape[0]`  
**Values**: Gene symbols (BRCA1, TP53) or probe IDs (1007_s_at)

```python
# Example
dataset['keys']
# ['BRCA1', 'TP53', 'EGFR', 'MYC', ...]

# Check length matches expression matrix
assert len(dataset['keys']) == dataset['expr'].shape[0]

# Check for duplicates
from collections import Counter
gene_counts = Counter(dataset['keys'])
duplicates = {gene: count for gene, count in gene_counts.items() if count > 1}
print(f"Genes with multiple probes: {len(duplicates)}")
```

### Class Labels (`class`)

**Type**: `np.ndarray` (integer)  
**Length**: Must match `expr.shape[1]`  
**Values**: Binary 0/1 encoding
- 0 = Control/reference group
- 1 = Case/treatment group

```python
# Example
dataset['class']
# array([0, 0, 0, 1, 1, 1])  # 3 controls, 3 cases

# Check encoding
assert set(dataset['class']).issubset({0, 1}), "Class must be 0 or 1"

# Check length
assert len(dataset['class']) == dataset['expr'].shape[1]

# Count samples per group
import numpy as np
unique, counts = np.unique(dataset['class'], return_counts=True)
print(f"Controls: {counts[0]}, Cases: {counts[1]}")
```

### Phenotype Data (`pheno`)

**Type**: `pandas.DataFrame`  
**Shape**: (n_samples, n_features)  
**Index**: Sample identifiers (matching `expr` columns)  
**Columns**: Clinical/technical variables

```python
# Example
dataset['pheno']
#              geo_accession  age  gender disease_status
# GSM123456      GSM123456   45    male        control
# GSM123457      GSM123457   52  female           case
# GSM123458      GSM123458   38    male        control

# Extract class labels from phenotype
class_vector = (dataset['pheno']['disease_status'] == 'case').astype(int)
dataset['class'] = class_vector.values
```

### Complete Dataset Example

```python
{
    'expr': pd.DataFrame(shape=(54675, 50)),
    'keys': ['BRCA1', 'TP53', ...],  # 54,675 genes
    'class': np.array([0,0,0,...,1,1,1]),  # 50 samples
    'pheno': pd.DataFrame(shape=(50, 15)),
    
    'name': 'GSE5847',
    'formattedName': 'Breast_Cancer',
    'platform': 'GPL570',
    
    'exp_comment': 'Data in log scale (quantile normalized)',
    'key_comment': 'Mapped to gene symbols using UCSC annotation',
    'load_comment': 'Loaded with GEOquery Python port'
}
```

## Analysis Results

After running meta-analysis, results are stored in `meta_object['metaAnalysis']`:

### Meta-Analysis Results Structure

```python
meta_results = {
    # Pooled estimates across studies (PRIMARY OUTPUT)
    'pooled_estimates': pd.DataFrame,
    
    # Per-study effect sizes
    'study_effects': dict,
    
    # Fisher combined p-values
    'fisher_results': pd.DataFrame,
    
    # Analysis metadata
    'analysis_info': dict
}
```

### Pooled Estimates DataFrame

**Type**: `pandas.DataFrame`  
**Index**: Gene symbols  
**Columns**: Statistical results

```python
pooled = meta_results['pooled_estimates']

# Columns
pooled.columns
# ['n_studies', 'summary', 'se_summary', 'tau2', 'p_value', 
#  'Q', 'df', 'pval_het', 'p_fdr', 'effect_size']

# Example rows
pooled.head()
#        n_studies  summary  se_summary   tau2    p_value     Q  df  pval_het    p_fdr  effect_size
# BRCA1          2    2.345       0.234  0.012  1.23e-15  0.82   1      0.36  2.56e-11        2.345
# TP53           2    2.103       0.198  0.008  3.45e-14  0.45   1      0.50  3.59e-10        2.103
# EGFR           2    1.987       0.187  0.015  5.67e-13  1.23   1      0.27  3.93e-09        1.987
```

### Column Descriptions

| Column | Description | Type | Range |
|--------|-------------|------|-------|
| `n_studies` | Number of studies gene appears in | int | 1 to n_datasets |
| `summary` | Pooled effect size | float | -∞ to +∞ |
| `se_summary` | Standard error of pooled effect | float | > 0 |
| `tau2` | Between-study variance | float | ≥ 0 |
| `p_value` | Meta-analysis p-value | float | 0 to 1 |
| `Q` | Cochran's Q statistic (heterogeneity) | float | ≥ 0 |
| `df` | Degrees of freedom | int | n_studies - 1 |
| `pval_het` | P-value for heterogeneity test | float | 0 to 1 |
| `p_fdr` | FDR-corrected p-value | float | 0 to 1 |
| `effect_size` | Same as `summary` | float | -∞ to +∞ |

### Per-Study Effects

```python
study_effects = meta_results['study_effects']

# Structure: dict of DataFrames
study_effects.keys()
# dict_keys(['Breast_Cancer', 'Sepsis'])

# Each study's effects
breast_effects = study_effects['Breast_Cancer']
#        n1   m1   sd1  n2   m2   sd2   diff  pooled_sd      g   se_g    keys
# 0      25  7.2  1.5  25  9.1  1.6    1.9       1.55  1.23   0.29  BRCA1
# 1      25  6.8  1.3  25  8.9  1.4    2.1       1.35  1.56   0.31   TP53
```

## Annotation Structures

### Annotation Report

After gene annotation, each dataset gets an annotation report:

```python
dataset['annotation_report'] = {
    'source_type': 'Affymetrix Probe ID',
    'total_identifiers': 54675,
    'mapped_identifiers': 20453,
    'mapping_rate': 0.923,
    'unmapped_identifiers': set(['AFFX-BioB-5_at', ...]),
    'annotation_method': 'UCSC RefSeq',
    'timestamp': '2026-04-21T10:30:00'
}
```

### Feature Data (`fData`)

Platform-specific annotation (optional):

```python
# May be available for some datasets
if 'fData' in dataset:
    fdata = dataset['fData']
    
    # Typical columns
    fdata.columns
    # ['ID', 'GB_ACC', 'SPOT_ID', 'Species Scientific Name',
    #  'Annotation Date', 'Sequence Type', 'Sequence Source',
    #  'Target Description', 'Representative Public ID',
    #  'Gene Title', 'Gene Symbol', 'ENTREZ_GENE_ID', ...]
```

## Working with Data Structures

### Creating a Meta-Object from Scratch

```python
import numpy as np
import pandas as pd

# Create expression matrix
expr = pd.DataFrame(
    np.random.randn(100, 20),  # 100 genes × 20 samples
    index=[f'Gene_{i}' for i in range(100)],
    columns=[f'Sample_{i}' for i in range(20)]
)

# Create dataset
dataset = {
    'expr': expr,
    'keys': [f'Gene_{i}' for i in range(100)],
    'class': np.array([0]*10 + [1]*10),  # 10 controls, 10 cases
    'pheno': pd.DataFrame({
        'sample_id': [f'Sample_{i}' for i in range(20)],
        'age': np.random.randint(30, 70, 20),
        'gender': np.random.choice(['M', 'F'], 20)
    }),
    'name': 'My_Study',
    'formattedName': 'Custom Dataset',
    'platform': 'Custom',
    'exp_comment': 'Simulated data',
    'key_comment': 'Gene symbols',
    'load_comment': 'Created manually'
}

# Create meta-object
meta_object = {
    'originalData': {
        'My_Study': dataset
    }
}
```

### Validating Data Structure

```python
from metaintegrator.core_analysis import MetaAnalysisCore

def validate_dataset(dataset):
    """Validate dataset structure."""
    
    # Check required fields
    required = ['expr', 'keys', 'class', 'pheno']
    for field in required:
        assert field in dataset, f"Missing required field: {field}"
    
    # Check dimensions
    n_genes = dataset['expr'].shape[0]
    n_samples = dataset['expr'].shape[1]
    
    assert len(dataset['keys']) == n_genes, "Keys length doesn't match expr rows"
    assert len(dataset['class']) == n_samples, "Class length doesn't match expr columns"
    
    # Check class encoding
    assert set(dataset['class']).issubset({0, 1}), "Class must be 0 or 1"
    
    # Use built-in validator
    MetaAnalysisCore.check_metagem_study(dataset)
    
    print("✅ Dataset structure valid")

# Validate
validate_dataset(dataset)
```

### Modifying Structures

```python
# Add new field
dataset['batch_id'] = 'Batch_001'

# Subset genes
keep_genes = dataset['expr'].var(axis=1) > 0.5
dataset['expr'] = dataset['expr'][keep_genes]
dataset['keys'] = [k for i, k in enumerate(dataset['keys']) if keep_genes.iloc[i]]

# Subset samples
keep_samples = dataset['pheno']['age'] > 40
dataset['expr'] = dataset['expr'].loc[:, keep_samples]
dataset['class'] = dataset['class'][keep_samples]
dataset['pheno'] = dataset['pheno'][keep_samples]

# Re-validate
validate_dataset(dataset)
```

### Saving and Loading

```python
import pickle

# Save meta-object
with open('meta_object.pkl', 'wb') as f:
    pickle.dump(meta_object, f)

# Load meta-object
with open('meta_object.pkl', 'rb') as f:
    meta_object = pickle.load(f)

# Save as JSON (smaller, but loses numpy/pandas structure)
import json

# Convert to JSON-serializable format
meta_json = {
    'originalData': {
        name: {
            'expr': dataset['expr'].to_dict(),
            'keys': dataset['keys'],
            'class': dataset['class'].tolist(),
            'name': dataset['name']
        }
        for name, dataset in meta_object['originalData'].items()
    }
}

with open('meta_object.json', 'w') as f:
    json.dump(meta_json, f, indent=2)
```

## Advanced Topics

### Memory-Efficient Storage

For large meta-objects (>1 GB):

```python
# Use HDF5 for large datasets
import h5py

def save_meta_hdf5(meta_object, filename):
    """Save meta-object to HDF5."""
    
    with h5py.File(filename, 'w') as f:
        for dataset_name, dataset in meta_object['originalData'].items():
            group = f.create_group(dataset_name)
            
            # Store expression matrix
            group.create_dataset('expr', data=dataset['expr'].values)
            group.create_dataset('expr_index', data=dataset['expr'].index.tolist())
            group.create_dataset('expr_columns', data=dataset['expr'].columns.tolist())
            
            # Store other fields
            group.create_dataset('keys', data=dataset['keys'])
            group.create_dataset('class', data=dataset['class'])
            
            # Store metadata as attributes
            group.attrs['name'] = dataset.get('name', '')
            group.attrs['platform'] = dataset.get('platform', '')

save_meta_hdf5(meta_object, 'meta_object.h5')
```

### Merging Meta-Objects

```python
def merge_meta_objects(obj1, obj2):
    """Merge two meta-objects."""
    
    merged = {
        'originalData': {
            **obj1['originalData'],
            **obj2['originalData']
        }
    }
    
    # Merge analysis results if present
    if 'metaAnalysis' in obj1 and 'metaAnalysis' in obj2:
        print("⚠️ Both objects have meta-analysis results")
        print("   Re-run meta-analysis on merged object")
    
    return merged

# Merge
combined = merge_meta_objects(meta_obj_A, meta_obj_B)
```

### Data Structure Conversion

```python
def to_r_format(meta_object):
    """Convert to R MetaIntegrator format (for rpy2)."""
    
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    r_meta = {}
    for name, dataset in meta_object['originalData'].items():
        r_dataset = {
            'expr': pandas2ri.py2rpy(dataset['expr']),
            'keys': ro.StrVector(dataset['keys']),
            'class': ro.IntVector(dataset['class']),
            'pheno': pandas2ri.py2rpy(dataset['pheno'])
        }
        r_meta[name] = r_dataset
    
    return r_meta
```

## FAQ

### Q: What's the difference between `keys` and `expr.index`?

**A:** 
- `keys`: Gene symbols (BRCA1, TP53) - used for meta-analysis
- `expr.index`: Probe IDs (1007_s_at) - platform-specific identifiers

After annotation, `keys` contains gene symbols while `expr.index` may retain probe IDs. Meta-analysis uses `keys` for matching genes across studies.

### Q: Can class labels be strings instead of 0/1?

**A:** No. Meta-analysis requires binary numeric encoding:
```python
# ❌ Wrong
dataset['class'] = ['control', 'case', 'control']

# ✅ Correct
dataset['class'] = np.array([0, 1, 0])  # 0=control, 1=case
```

### Q: How do I add custom fields to a dataset?

**A:** Simply assign new keys:
```python
dataset['custom_field'] = my_data
dataset['batch_info'] = batch_labels
dataset['qc_metrics'] = qc_results

# Fields are preserved through analysis
```

### Q: What if `pheno` is missing?

**A:** Phenotype data is technically optional but highly recommended:
```python
# Minimal pheno (if unavailable)
dataset['pheno'] = pd.DataFrame({
    'sample_id': dataset['expr'].columns
})
```

## Troubleshooting

### Issue: "Keys length doesn't match expression matrix"

**Cause:** Gene identifiers don't align with expression rows

**Solution:**
```python
# Check lengths
print(f"Expression rows: {dataset['expr'].shape[0]}")
print(f"Keys length: {len(dataset['keys'])}")

# Fix: Ensure they match
if len(dataset['keys']) != dataset['expr'].shape[0]:
    # Option 1: Use expression index as keys
    dataset['keys'] = dataset['expr'].index.tolist()
    
    # Option 2: Filter to matching length
    n = min(len(dataset['keys']), dataset['expr'].shape[0])
    dataset['expr'] = dataset['expr'].iloc[:n]
    dataset['keys'] = dataset['keys'][:n]
```

### Issue: "Class vector has wrong length"

**Cause:** Sample labels don't match expression columns

**Solution:**
```python
# Check lengths
print(f"Expression columns: {dataset['expr'].shape[1]}")
print(f"Class length: {len(dataset['class'])}")

# Fix: Ensure they match
if len(dataset['class']) != dataset['expr'].shape[1]:
    print("⚠️ Class labels don't match samples")
    print("   Recreate class vector or subset expression matrix")
```

### Issue: "Expression matrix has object dtype instead of numeric"

**Cause:** Non-numeric values in expression data

**Solution:**
```python
# Check dtype
print(dataset['expr'].dtypes)

# Convert to numeric (coerces invalid to NaN)
dataset['expr'] = dataset['expr'].apply(pd.to_numeric, errors='coerce')

# Check for NaNs introduced
n_missing = dataset['expr'].isna().sum().sum()
if n_missing > 0:
    print(f"⚠️ {n_missing} values couldn't be converted to numeric")
```

## Appendix

### Data Structure Diagram

```
meta_object
├── originalData (dict)
│   ├── Study_A (dict)
│   │   ├── expr (DataFrame): n_genes × n_samples
│   │   ├── keys (list): length n_genes
│   │   ├── class (ndarray): length n_samples
│   │   ├── pheno (DataFrame): n_samples × n_features
│   │   ├── name (str)
│   │   ├── formattedName (str)
│   │   ├── platform (str)
│   │   ├── exp_comment (str)
│   │   ├── key_comment (str)
│   │   └── annotation_report (dict)
│   ├── Study_B (dict)
│   │   └── ... (same structure)
│   └── Study_C (dict)
│       └── ... (same structure)
└── metaAnalysis (dict) - added after analysis
    ├── pooled_estimates (DataFrame)
    ├── study_effects (dict)
    ├── fisher_results (DataFrame)
    └── analysis_info (dict)
```

### Quick Reference

**Create empty meta-object:**
```python
meta_object = {'originalData': {}}
```

**Add dataset:**
```python
meta_object['originalData']['Study_Name'] = dataset
```

**Access dataset:**
```python
dataset = meta_object['originalData']['Study_Name']
```

**Get expression values:**
```python
expr = dataset['expr']
genes = dataset['keys']
labels = dataset['class']
```

**Get results:**
```python
results = meta_object['metaAnalysis']['pooled_estimates']
```

### Type Specifications

```python
from typing import Dict, List
import numpy as np
import pandas as pd

# Type hints for data structures
Dataset = Dict[str, Union[pd.DataFrame, List, np.ndarray, str]]
MetaObject = Dict[str, Dict[str, Dataset]]

def process_dataset(dataset: Dataset) -> None:
    expr: pd.DataFrame = dataset['expr']
    keys: List[str] = dataset['keys']
    class_labels: np.ndarray = dataset['class']
    # ... process data
```

### Related Documentation

- [Concepts Guide](concepts.md) - Meta-analysis fundamentals
- [GEO Integration](geo_integration.md) - Data acquisition
- [API Reference](../api/data_loading.md) - Function signatures
- [Tutorials](../tutorials/complete_workflow.md) - Hands-on examples

---

**Last updated:** 2026-04-21  
**Version:** 1.0  
**Maintainer:** MetaIntegrator-py Team
