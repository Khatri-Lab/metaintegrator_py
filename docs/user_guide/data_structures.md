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

The `ProcessedDataset` dataclass is the standard output of all preprocessing handlers.
It has **no** `class`/label field of its own: case/control labels are always assigned
as a separate step (see [Class Labels](#class-labels-class) below): and it has no
`to_dict()` method:

```python
from preprocessing.base import ProcessedDataset

@dataclass
class ProcessedDataset:
    geo_id: str                # "GSE11909_GPL96"
    expr: pd.DataFrame         # probes × samples (float64)
    pheno: pd.DataFrame        # sample metadata (GSM IDs as index)
    keys: List[str]            # gene symbol per probe (comma-separated for multi-gene probes)
    metadata: Dict[str, Any]   # source, platform, handler, etc.
```

**Converting to the analysis dict format:**

There is no built-in conversion method: build the dict by hand and assign labels
with `assign_class_labels()`:

```python
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.labeling import assign_class_labels

loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(['GSE11909'])

meta_object = {"originalData": {}}
for ds in datasets:
    dataset = {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
               "formattedName": ds.geo_id, "class": None}
    meta_object["originalData"][ds.geo_id] = assign_class_labels(
        dataset, column_name="title", disease_terms=["disease"]
    )
```

## filter_object (Gene Filtering Output)

`MetaAnalyzer.filter_genes()` (in `analysis.py`) stores a `FilterResult` dataclass
under `meta_object["filterResults"]`. It supports both attribute access and
R-style dict access:

```python
filter_result = meta_object["filterResults"]

filter_result.pos_gene_names      # List[str]: up-regulated signature genes
filter_result.neg_gene_names      # List[str]: down-regulated signature genes
filter_result.n_pos               # int
filter_result.n_neg               # int
filter_result.effect_size_thresh  # float
filter_result.fdr_thresh          # float

filter_object = filter_result.to_dict()  # R-style keys, e.g. for ScoreCalculator
filter_object["posGeneNames"]     # aliased to pos_gene_names
filter_object["negGeneNames"]     # aliased to neg_gene_names
filter_object["FDRThresh"]        # aliased to fdr_thresh
filter_object["effectSizeThresh"] # aliased to effect_size_thresh
```

!!! note "Two independent filtering systems"
    A separate, strategy-based pipeline: `scoring_meta_integrator_filter.MetaIntegratorFilter`
    (`.filter()`, not `.filter_genes()`): produces a differently-shaped `FilterResults`
    object. The two are not interchangeable; see
    [API Reference](../api/overview.md#package-metaintegrator).

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
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.labeling import assign_class_labels

loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(["GSE5847", "GSE28750"])

meta_object = {"originalData": {}}
for ds in datasets:
    dataset = {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
               "formattedName": ds.geo_id, "class": None}
    meta_object["originalData"][ds.geo_id] = assign_class_labels(
        dataset, column_name="title", disease_terms=["disease"]
    )

# Structure
print(meta_object.keys())
# Output: dict_keys(['originalData'])

print(list(meta_object['originalData'].keys()))
# Output: ['GSE5847_GPL570', 'GSE28750_GPL570']  (keyed by geo_id, not formattedName)
```

### Accessing Data

```python
# Get all datasets
datasets = meta_object['originalData']

# Get specific dataset (keyed by geo_id, e.g. "GSE5847_GPL570" -- see above)
breast_cancer = meta_object['originalData']['GSE5847_GPL570']

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

For data loaded from GEO specifically, `pheno` always exposes a standardized
set of columns (`title`, `source_name`, `organism`, `characteristics`,
`description`) regardless of the original GEO field names -- see
[GEO Integration](geo_integration.md#data-processing-pipeline) for details.
The generic example below applies to any `pheno` table, GEO-derived or not.

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

After `MetaAnalyzer.run_meta_analysis()`, results are stored in `meta_object['metaAnalysis']`:

### Meta-Analysis Results Structure

```python
meta_analysis = {
    # Primary output: R-parity, FDR- and Fisher-combined per-gene results
    'pooledResults': pd.DataFrame,

    # Raw inverse-variance pooling output, *before* the Fisher combination
    # is merged in: lower-level, snake_case columns, mainly of interest
    # if you need the pooling statistics on their own
    'pooled_estimates': pd.DataFrame,

    # Per-dataset effect sizes / standard errors (genes × datasets)
    'datasetEffectSizes': pd.DataFrame,
    'datasetEffectSizeStandardErrors': pd.DataFrame,

    'analysisDescription': str,
}
```

### pooledResults DataFrame (primary output)

**Type**: `pandas.DataFrame`
**Index**: Gene symbols
**Columns**: Statistical results (R-parity naming, matches the `filterGenes()` input)

```python
pooled = meta_object['metaAnalysis']['pooledResults']

pooled.columns
# ['effectSize', 'effectSizeStandardError', 'effectSizePval', 'effectSizeFDR',
#  'tauSquared', 'numStudies', 'cochranesQ', 'heterogeneityPval',
#  'fisherStatUp', 'fisherPvalUp', 'fisherFDRUp',
#  'fisherStatDown', 'fisherPvalDown', 'fisherFDRDown']

pooled.nsmallest(10, "effectSizeFDR")[["effectSize", "effectSizeFDR"]]
```

### Column Descriptions

| Column | Description | Type | Range |
|--------|-------------|------|-------|
| `effectSize` | Pooled Hedges' g effect size | float | -∞ to +∞ |
| `effectSizeStandardError` | Standard error of pooled effect | float | > 0 |
| `effectSizePval` | Meta-analysis p-value | float | 0 to 1 |
| `effectSizeFDR` | FDR-corrected p-value (the threshold `filterGenes()` uses) | float | 0 to 1 |
| `tauSquared` | Between-study variance | float | ≥ 0 |
| `numStudies` | Number of studies gene appears in | int | 1 to n_datasets |
| `cochranesQ` | Cochran's Q statistic (heterogeneity) | float | ≥ 0 |
| `heterogeneityPval` | P-value for the heterogeneity test | float | 0 to 1 |
| `fisherStatUp` / `fisherStatDown` | Fisher combined statistic, directional | float | ≥ 0 |
| `fisherPvalUp` / `fisherPvalDown` | Fisher combined p-value, directional | float | 0 to 1 |
| `fisherFDRUp` / `fisherFDRDown` | FDR-corrected Fisher p-value, directional | float | 0 to 1 |

### Per-Dataset Effect Sizes

`datasetEffectSizes` and `datasetEffectSizeStandardErrors` are gene × dataset
matrices (Hedges' g and its standard error per gene, per contributing study) —
not a dict keyed by study name:

```python
es_df = meta_object['metaAnalysis']['datasetEffectSizes']
#          GSE5847_GPL570  GSE28750_GPL570
# BRCA1              1.23              1.56
# TP53                NaN              1.02   # NaN = gene absent from that study
```

## Annotation Structures

There is currently no separate annotation-report or feature-data (`fData`) structure
attached to a dataset. Annotation results are reflected directly in `keys`:
`GeneAnnotator.annotate()` maps each probe to a gene symbol (comma-separated for
multi-gene probes), and unmappable probes are left as `'NA'`. To compute your own
mapping-rate summary:

```python
keys = dataset['keys']
n_total = len(keys)
n_mapped = sum(1 for k in keys if k != 'NA')
print(f"Mapped {n_mapped}/{n_total} probes ({n_mapped / n_total:.1%})")
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
│   │   ├── formattedName (str)
│   │   ├── exp_comment (str, optional)
│   │   └── key_comment (str, optional)
│   ├── Study_B (dict)
│   │   └── ... (same structure)
│   └── Study_C (dict)
│       └── ... (same structure)
├── metaAnalysis (dict) - added by run_meta_analysis()
│   ├── pooledResults (DataFrame)             # primary output
│   ├── pooled_estimates (DataFrame)          # pre-Fisher pooling only
│   ├── datasetEffectSizes (DataFrame)
│   ├── datasetEffectSizeStandardErrors (DataFrame)
│   └── analysisDescription (str)
├── leaveOneOutAnalysis (dict or None)        # added by run_meta_analysis()
└── filterResults (FilterResult)              # added by filter_genes()
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
pooled = meta_object['metaAnalysis']['pooledResults']
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
