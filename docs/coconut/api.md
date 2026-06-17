# COCONUT API Reference

Complete API documentation for all public functions and classes in the COCONUT Python package.

---

## 📚 Module Overview

### `coconut.coconut` - Core Algorithm
Low-level API for COCONUT normalization with maximum control.

### `coconut.coconut_runner` - High-Level Workflow
User-friendly interface for common workflows, especially CSV-based data.

---

## coconut.coconut_runner Module

### `runner()` ⭐ MOST POPULAR

Run the complete COCONUT pipeline with a single command.

```python
def runner(csv_dir: str, output_dir: str | None = None) -> None
```

**Parameters:**

- **csv_dir** *(str, required)*  
  Path to directory containing CSV files with matching gene/pheno pairs

- **output_dir** *(str, optional)*  
  Directory to save output CSV files. If `None`, results are only printed

**Output Files** (when `output_dir` provided):

- `mcoco_combined_genes.csv` - Gene expression (genes × samples)
- `mcoco_combined_genes_samples.csv` - Transposed (samples × genes)  
- `mcoco_combined_pheno.csv` - Sample metadata
- `mcoco_combined_class_labels.csv` - Binary labels (0=control, 1=disease)

**Example:**

```python
from coconut.coconut_runner import runner

# Process and save
runner(csv_dir='./studies', output_dir='./results')

# Or just process (no files saved)
runner(csv_dir='./studies')
```

**Raises:**

- `FileNotFoundError` - If directory doesn't exist or no CSV files found
- `ValueError` - If genes/pheno files don't align properly

---

### `load_gse_data_from_csv_directory()`

Load and validate multi-study gene expression data from CSV files with automatic sample alignment.

```python
def load_gse_data_from_csv_directory(
    csv_directory: str,
    genes_pattern: str = "_genes.csv",
    pheno_pattern: str = "_pheno.csv",
    object_prefix: str = "GSEs.test_"
) -> List[Dict[str, pd.DataFrame]]
```

**Parameters:**

- **csv_directory** *(str, required)*  
  Path to directory containing CSV files

- **genes_pattern** *(str, default: "_genes.csv")*  
  File pattern to identify gene expression files

- **pheno_pattern** *(str, default: "_pheno.csv")*  
  File pattern to identify phenotype files

- **object_prefix** *(str, default: "GSEs.test_")*  
  Prefix to remove from dataset names

**Returns:**

```python
[
    {
        'genes': pd.DataFrame,  # genes (rows) × samples (columns)
        'pheno': pd.DataFrame   # samples (rows) × attributes (columns)
    },
    # ... more datasets
]
```

**Example:**

```python
from coconut.coconut_runner import load_gse_data_from_csv_directory

# Standard usage
gse_data = load_gse_data_from_csv_directory('./my_studies')

# Custom patterns
gse_data = load_gse_data_from_csv_directory(
    csv_directory='./data',
    genes_pattern='.expression.csv',
    pheno_pattern='.metadata.csv',
    object_prefix='Study_'
)
```

**Expected File Structure:**

```
my_studies/
├── GSEs.test_GSE28750_genes.csv
├── GSEs.test_GSE28750_pheno.csv
├── GSEs.test_GSE40012_genes.csv
└── GSEs.test_GSE40012_pheno.csv
```

**Raises:**

- `FileNotFoundError` - Directory doesn't exist or no matching files
- `ValueError` - Incomplete datasets (missing genes or pheno file)

---

### `load_and_run_coconut_example()`

Interactive workflow that helps users understand their data and COCONUT parameters.

```python
def load_and_run_coconut_example(csv_directory: str) -> Dict
```

**Parameters:**

- **csv_directory** *(str, required)*  
  Path to directory with CSV files

**Returns:**

Dictionary containing COCONUT results (same structure as `coconut()`)

**What it does:**

1. Loads data from CSV directory
2. Shows available phenotype columns for each dataset
3. Helps identify the control column
4. Runs COCONUT with suggested parameters
5. Returns results for further analysis

**Example:**

```python
from coconut.coconut_runner import load_and_run_coconut_example

# Interactive exploration
result = load_and_run_coconut_example('./my_data')

# Example output shown:
# 🔍 Inspecting phenotype columns...
# Dataset 1 columns:
#   Healthy0.Sepsis1: [0, 1]
#   Platform_id: ['GPL570']
```

---

## coconut.coconut Module

### `coconut()`

Core COCONUT algorithm for batch effect normalization.

```python
def coconut(
    gses: List[Dict[str, pd.DataFrame]],
    control_0_col: str,
    disease_col: Optional[str] = None,
    by_platform: bool = False,
    platform_col: Optional[str] = None,
    par_prior: bool = True,
    it_conv: float = 1e-04,
    parallel: bool = False,
    mc_cores: int = 1
) -> Dict
```

**Parameters:**

- **gses** *(list[dict], required)*  
  List of GSE dictionaries, each containing `'genes'` and `'pheno'` DataFrames

- **control_0_col** *(str, required)*  
  Column name in pheno indicating controls (0) vs disease (1)

- **disease_col** *(str, optional)*  
  Specific disease column if multiple disease types present

- **by_platform** *(bool, default: False)*  
  Whether to normalize within platforms separately

- **platform_col** *(str, optional)*  
  Column name containing platform information (required if `by_platform=True`)

- **par_prior** *(bool, default: True)*  
  Use parametric priors (True) or non-parametric (False)

- **it_conv** *(float, default: 1e-04)*  
  Convergence threshold for EB algorithm

- **parallel** *(bool, default: False)*  
  Enable parallel processing (not yet implemented)

- **mc_cores** *(int, default: 1)*  
  Number of cores for parallel processing

**Returns:**

```python
{
    'COCONUTList': [          # Normalized disease samples
        {
            'genes': pd.DataFrame,
            'pheno': pd.DataFrame,
            'class_labels': np.ndarray
        },
        # ... more datasets
    ],
    'rawDiseaseList': [       # Original disease samples
        # ... same structure
    ],
    'controlList': {          # Control data + parameters
        'GSEs': [...],
        'bayes_params': {
            'gamma_star': np.ndarray,   # Mean shifts
            'delta_star': np.ndarray    # Variance scaling
        },
        'get_platforms': np.ndarray
    }
}
```

**Example:**

```python
from coconut.coconut import coconut

# Basic usage
result = coconut(
    gses=gse_data,
    control_0_col='Healthy0.Sepsis1'
)

# Platform-specific normalization
result = coconut(
    gses=gse_data,
    control_0_col='control',
    by_platform=True,
    platform_col='Platform_id'
)

# Non-parametric (more robust)
result = coconut(
    gses=gse_data,
    control_0_col='control',
    par_prior=False,
    it_conv=1e-06
)
```

**Raises:**

- `ValueError` - Missing required columns, insufficient data, or invalid parameters
- `RuntimeError` - Convergence failure or numerical issues

---

### `combine_coco_output()`

Combine COCONUT results into a single unified dataset.

```python
def combine_coco_output(
    coconut_result: Dict
) -> Dict[str, Union[pd.DataFrame, np.ndarray]]
```

**Parameters:**

- **coconut_result** *(dict, required)*  
  Output from `coconut()` function

**Returns:**

```python
{
    'genes': pd.DataFrame,              # All samples, normalized (genes × samples)
    'pheno': pd.DataFrame,              # Common columns only (samples × attributes)
    'class_cntl0_dis1': np.ndarray     # Binary labels (0=control, 1=disease)
}
```

**Example:**

```python
from coconut.coconut import coconut, combine_coco_output

# Run COCONUT
result = coconut(gses=gse_data, control_0_col='control')

# Combine results
combined = combine_coco_output(result)

# Access normalized data
genes = combined['genes']              # Expression matrix
pheno = combined['pheno']              # Metadata
labels = combined['class_cntl0_dis1'] # Class labels

# Export
genes.to_csv('normalized_genes.csv')
pheno.to_csv('phenotype_data.csv')
```

---

## Internal/Helper Functions

These functions are part of the internal API. Users typically don't call them directly, but they're documented here for advanced use cases.

### `_get_common_genes()`

Identify genes present in all datasets.

```python
def _get_common_genes(
    datasets: List[Dict[str, pd.DataFrame]]
) -> List[str]
```

**Parameters:**

- **datasets** *(list[dict], required)*  
  List of GSE dictionaries

**Returns:**

Sorted list of gene identifiers common to all datasets

**Example:**

```python
from coconut.coconut import _get_common_genes

common = _get_common_genes(gse_data)
print(f"Common genes: {len(common)}")
```

---

### `_get_data_subset()`

Extract control or disease subset from datasets.

```python
def _get_data_subset(
    datasets: List[Dict[str, pd.DataFrame]],
    gene_list: List[str],
    filter_col: str,
    filter_val: Optional[str] = None,
    is_control: bool = True
) -> List[Dict[str, Any]]
```

**Parameters:**

- **datasets** *(list[dict], required)*  
  List of GSE dictionaries

- **gene_list** *(list[str], required)*  
  Genes to include in subset

- **filter_col** *(str, required)*  
  Column name for filtering (e.g., 'control_column')

- **filter_val** *(str, optional)*  
  Specific disease value to filter

- **is_control** *(bool, default: True)*  
  Whether to extract controls (True) or disease samples (False)

**Returns:**

List of dictionaries with filtered data

**Example:**

```python
from coconut.coconut import _get_common_genes, _get_data_subset

# Get common genes
common = _get_common_genes(gse_data)

# Extract controls
controls = _get_data_subset(
    datasets=gse_data,
    gene_list=common,
    filter_col='Healthy0.Sepsis1',
    is_control=True
)

# Extract disease samples
disease = _get_data_subset(
    datasets=gse_data,
    gene_list=common,
    filter_col='Healthy0.Sepsis1',
    is_control=False
)
```

---

## Type Definitions

COCONUT uses TypedDict for type hints. These are defined in `coconut.coconut`:

### `GeneExpressionData`

```python
{
    'genes': pd.DataFrame,           # genes × samples
    'pheno': pd.DataFrame,           # samples × attributes  
    'class_labels': np.ndarray      # binary labels
}
```

### `BayesParams`

```python
{
    'B_hat': np.ndarray,            # batch-specific shifts
    'stand_mean': np.ndarray,       # standardized means
    'var_pooled': np.ndarray,       # pooled variances
    'gamma_star': np.ndarray,       # empirical Bayes mean shifts
    'delta_star': np.ndarray        # empirical Bayes variance scaling
}
```

### `CoconutOutput`

```python
{
    'COCONUTList': List[GeneExpressionData],      # normalized disease
    'rawDiseaseList': List[GeneExpressionData],   # original disease
    'controlList': {
        'GSEs': List[GeneExpressionData],
        'bayes_params': BayesParams,
        'get_platforms': np.ndarray
    }
}
```

### `CombinedCoconutOutput`

```python
{
    'genes': pd.DataFrame,              # combined expression
    'pheno': pd.DataFrame,              # combined metadata
    'class_cntl0_dis1': np.ndarray     # binary labels
}
```

---

## Debug Mode

Enable detailed progress messages:

```python
from coconut import coconut_runner

# Enable debug mode
coconut_runner.debug_mode = True

# Now all runner functions show detailed output
from coconut.coconut_runner import runner
runner(csv_dir='./data', output_dir='./results')
```

**Debug Output Example:**

```
🔍 Scanning directory: ./data
📁 Found 6 CSV files
📊 Loading dataset: GSE28750
   📈 Genes: 12000 genes × 20 samples
   📋 Pheno: 20 samples × 5 attributes
   ✅ Final alignment: 20 samples
```

---

## Constants

### `CONTROL_LABEL`

Default numeric value representing control samples.

```python
from coconut.coconut import CONTROL_LABEL

print(CONTROL_LABEL)  # 0
```

In pheno DataFrames, control samples should have value `0` in the control column, and disease samples should have value `1`.

---

## Complete Workflow Example

```python
from coconut.coconut_runner import load_gse_data_from_csv_directory
from coconut.coconut import coconut, combine_coco_output
import pandas as pd

# 1. Load data
gse_data = load_gse_data_from_csv_directory('./studies')

# 2. Run COCONUT
result = coconut(
    gses=gse_data,
    control_0_col='Healthy0.Sepsis1',
    by_platform=False,
    par_prior=True
)

# 3. Combine results
combined = combine_coco_output(result)

# 4. Access data
genes = combined['genes']              # Expression matrix
pheno = combined['pheno']              # Metadata
labels = combined['class_cntl0_dis1'] # Binary labels

# 5. Export
genes.to_csv('normalized_genes.csv')
pheno.to_csv('pheno_data.csv')
pd.DataFrame({
    'sample_id': genes.columns,
    'class': labels.astype(int)
}).to_csv('labels.csv', index=False)

# 6. Quality check
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
coords = pca.fit_transform(genes.T)

plt.scatter(coords[:, 0], coords[:, 1], c=labels)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA after COCONUT')
plt.show()
```

---

## Performance Tips

### Memory Efficiency

```python
# For large datasets, process in batches
import gc

for batch in dataset_batches:
    result = coconut(batch, control_0_col='control')
    # Process result
    gc.collect()  # Free memory
```

### Speed Optimization

```python
# Use parametric priors (faster)
result = coconut(gses, control_0_col='control', par_prior=True)

# Non-parametric is slower but more robust
result = coconut(gses, control_0_col='control', par_prior=False)
```

---

## Version Information

```python
import coconut

# Check version
print(coconut.__version__)
```

---

## See Also

- [Quick Start Guide](quickstart.md) - Get started in 5 minutes
- [Workflow Guide](workflow.md) - Step-by-step tutorial
- [Runner Guide](runner_guide.md) - Detailed `coconut_runner` documentation
- [Best Practices](best_practices.md) - Tips for production use
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

---

*Last updated: 2026-04-24*  
*Package: metaintegrator_py*
