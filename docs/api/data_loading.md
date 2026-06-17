# Data Validation & Processing API

This page documents the validation, data processing, and sample operations
modules that prepare and verify data for meta-analysis.

!!! note "Module Layout"
    After refactoring, these responsibilities are split across several focused
    modules rather than a single monolithic `data_loading` module:

    - **`metaintegrator.validation`** — Dataset and meta-object validation
    - **`metaintegrator.data_processing`** — Combined annotation tables
    - **`metaintegrator.sample_ops`** — Sample subsetting
    - **`metaintegrator.labeling`** — Class label assignment

---

## DatasetValidator

Validates dataset and meta-object structures. This is the Python equivalent
of R's `checkDataObject()`.

**Module:** `metaintegrator.validation`

### `check_class_vector`

```python
@staticmethod
def check_class_vector(dataset: Dict) -> bool
```

Check if dataset has a properly formatted class vector with sufficient cases and controls.

| Name | Type | Description |
|------|------|-------------|
| dataset | Dict | Dictionary containing `'expr'`, `'class'`, and `'keys'` |

**Returns:** `True` if class vector is valid with ≥2 cases and ≥2 controls.

**Example**

```python
from metaintegrator.validation import DatasetValidator

dataset = {
    "expr": expression_matrix,  # Genes × Samples
    "class": np.array([0, 0, 1, 1, 1, 0]),
    "keys": gene_list
}

is_valid = DatasetValidator.check_class_vector(dataset)
print(f"Valid class vector: {is_valid}")  # True (3 cases, 3 controls)
```

> **Note:** Class vector must contain only 0 (controls) and 1 (cases).

---

### `make_valid_name`

```python
@staticmethod
def make_valid_name(name: str) -> str
```

Convert a string to a valid Python identifier (equivalent of R's `make.names()`).

| Name | Type | Description |
|------|------|-------------|
| name | str | Original name string |

**Returns:** Valid Python identifier string.

**Example**

```python
invalid_name = "GSE-1234 (batch 1)"
valid_name = DatasetValidator.make_valid_name(invalid_name)
print(valid_name)  # "GSE_1234__batch_1_"
```

---

### `validate_study_structure`

```python
@staticmethod
def validate_study_structure(study: Dict) -> Tuple[bool, List[str]]
```

Comprehensive validation of a single study data structure.

| Name | Type | Description |
|------|------|-------------|
| study | Dict | Study dictionary to validate |

**Returns:** `Tuple[bool, List[str]]` — (is_valid, list_of_error_messages)

**Validation Checks:**

- Presence of required keys: `'expr'`, `'class'`, `'keys'`
- Expression matrix is 2D DataFrame or array
- Class vector contains only 0/1
- Dimensions are consistent (genes × samples)

**Example**

```python
study = {
    "expr": pd.DataFrame(...),
    "class": np.array([0, 1, 0, 1]),
    "keys": ["GENE1", "GENE2", "GENE3"]
}

is_valid, errors = DatasetValidator.validate_study_structure(study)
if not is_valid:
    for error in errors:
        print(f"Validation error: {error}")
```

---

### `validate_meta_object`

```python
@staticmethod
def validate_meta_object(
    meta_object: Dict,
    check_type: str = "Meta",
    stage: str = "Pre-Analysis"
) -> Tuple[bool, List[str]]
```

Validate complete meta-analysis object structure. This is the Python equivalent
of R's `checkDataObject(metaObj, "Meta", "Pre-Analysis")`.

| Name | Type | Description |
|------|------|-------------|
| meta_object | Dict | Meta-analysis object to validate |
| check_type | str | Type of check (default: `"Meta"`) |
| stage | str | Analysis stage: `"Pre-Analysis"` or `"Post-Analysis"` |

**Returns:** `Tuple[bool, List[str]]` — (is_valid, list_of_error_messages)

**Example**

```python
from metaintegrator.validation import DatasetValidator

meta_object = {
    "originalData": {
        "Study1": study1_dict,
        "Study2": study2_dict
    }
}

# Validate before analysis
is_valid, errors = DatasetValidator.validate_meta_object(
    meta_object,
    stage="Pre-Analysis"
)

if not is_valid:
    raise ValueError("Invalid meta-object:\n" + "\n".join(errors))
```

---

### `convert_original_data_names`

```python
@staticmethod
def convert_original_data_names(meta_object: Dict) -> Dict
```

Convert original data names to valid identifiers and preserve original names
in the `formattedName` field. Called automatically by `MetaAnalyzer.run_meta_analysis()`.

---

## Convenience Functions

### `check_data_object`

```python
from metaintegrator.validation import check_data_object

is_valid = check_data_object(obj, "Meta", "Pre-Analysis")
```

Backward-compatible function matching R's `checkDataObject()`. Returns `bool`.

### `validate_and_report`

```python
from metaintegrator.validation import validate_and_report

result = validate_and_report(obj, "Meta", "Pre-Analysis")
# result is a ValidationResult with .is_valid, .get_errors(), .get_warnings()
```

Returns a `ValidationResult` object with detailed issue tracking.

---

## DataProcessor

Creates combined annotation tables for meta-analysis.

**Module:** `metaintegrator.data_processing`

### `create_combined_annotation_table`

```python
@staticmethod
def create_combined_annotation_table(original_data: List[Dict]) -> pd.DataFrame
```

Create combined annotation table from multiple datasets. Merges probe-to-gene
mappings across all studies, handling overlapping probe IDs.

| Name | Type | Description |
|------|------|-------------|
| original_data | List[Dict] | List of dataset dictionaries |

**Returns:** Combined annotation table (`pd.DataFrame`) indexed by probe ID.

> **Note:** Called internally by `MetaAnalyzer.run_meta_analysis_core()`.

---

## Sample Operations

**Module:** `metaintegrator.sample_ops`

### `subset_original_data`

```python
def subset_original_data(
    dataset_object: Dict,
    keep_samples: Union[List[str], List[int], np.ndarray]
) -> Dict
```

Subset samples from a dataset object (Python port of R's `subsetOriginalData()`).
Subsets all relevant components (expr, class, pheno) to include only the desired samples.

| Name | Type | Description |
|------|------|-------------|
| dataset_object | Dict | Dataset containing `'expr'` and `'class'` |
| keep_samples | List[str], List[int], or ndarray | Samples to keep (names, indices, or boolean mask) |

**Returns:** Subsetted dataset `Dict` with same structure as input.

**Example**

```python
from metaintegrator.sample_ops import subset_original_data

# Subset by sample names
keep = ['Sample_1', 'Sample_3', 'Sample_5']
subsetted = subset_original_data(dataset, keep)

# Subset by indices (0-based)
keep = [0, 2, 4]
subsetted = subset_original_data(dataset, keep)

# Subset by boolean mask
keep = np.array([True, False, True, False, True])
subsetted = subset_original_data(dataset, keep)
```

> **Warning:** Missing sample names will trigger a warning but not fail.

### `subsetOriginalData`

R-compatible convenience function:

```python
from metaintegrator.sample_ops import subsetOriginalData

subsetted = subsetOriginalData(
    dataset,
    keepMe=['Sample 1', 'Sample 13', 'Sample 43']
)
```

---

## Class Label Assignment

**Module:** `metaintegrator.labeling`

### `assign_class_labels`

```python
from metaintegrator.labeling import assign_class_labels
```

Utility for assigning binary class labels from phenotype data.

---

## Related

- [Preprocessing API](preprocessing.md) — GEO data fetching and gene annotation
- [Analysis API](analysis.md) — Meta-analysis functions
- [User Guide: Data Structures](../user_guide/data_structures.md) — Data structure details

---

## Common Workflows

### Loading from GEO and Validating

```python
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.validation import DatasetValidator

# 1. Fetch from GEO using the factory
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(["GSE1234"])

# 2. Build meta-object
meta_object = {"originalData": {
    ds.geo_id: ds.to_dict() for ds in datasets
}}

# 3. Set class labels from phenotype data
for name, dataset in meta_object["originalData"].items():
    pheno = dataset["pheno"]
    dataset["class"] = (pheno["disease_status"] == "case").astype(int).values

# 4. Validate
is_valid, errors = DatasetValidator.validate_meta_object(
    meta_object, stage="Pre-Analysis"
)
if not is_valid:
    print("Errors:", errors)
```

### Quality Control Pipeline

```python
import pandas as pd
import numpy as np
from metaintegrator.validation import DatasetValidator

# 1. Load custom data
expr = pd.read_csv("expression.csv", index_col=0)
pheno = pd.read_csv("phenotypes.csv")

# 2. Build study dict manually
study = {
    "expr": expr,
    "class": (pheno["condition"] == "disease").astype(int).values,
    "keys": expr.index.tolist(),
    "formattedName": "My Study"
}

# 3. Validate
is_valid, errors = DatasetValidator.validate_study_structure(study)
if not is_valid:
    print("Errors:", errors)

# 4. Check class balance
has_balance = DatasetValidator.check_class_vector(study)
print(f"Sufficient cases and controls: {has_balance}")
```
