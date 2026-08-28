# Migration Guide: R to Python

Complete guide for migrating from **MetaIntegrator (R)** to **metaIntegrator-py (Python)**, including syntax mapping, workflow conversion, and common pitfalls.

## Contents

- [Overview](#overview)
- [Quick Start for R Users](#quick-start-for-r-users)
- [Syntax Translation](#syntax-translation)
- [Function Mapping](#function-mapping)
- [Data Structure Conversion](#data-structure-conversion)
- [Common Workflows](#common-workflows)
- [Pitfalls and Solutions](#pitfalls-and-solutions)
- [Best Practices](#best-practices)

---

## Overview

### Why Migrate?

Migrating from R to Python offers several advantages:

- ✅ **Performance**: Automatic GEO data caching avoids redundant downloads
- ✅ **Caching**: Automatic GEO data caching
- ✅ **Production**: Better containerization and cloud deployment
- ✅ **Integration**: Seamless ML/AI pipeline integration
- ✅ **Type Safety**: Type hints for better IDE support

### What to Expect

| Aspect | Migration Effort | Notes |
|--------|------------------|-------|
| **Core Analysis** | 🟢 Easy | Direct function mapping |
| **Data Loading** | 🟢 Easy | Similar workflow, better caching |
| **Visualization** | 🟡 Moderate | Different plotting library |
| **Advanced Features** | 🟡 Moderate | Some R-specific functions need adaptation |
| **Custom Code** | 🔴 Difficult | Requires Python rewrite |

---

## Quick Start for R Users

### Minimal Example Comparison

#### R Version
```r
library(MetaIntegrator)

# Get data
metaObj <- getGEOData(c("GSE1234", "GSE5678"))

# Run analysis
results <- runMetaAnalysis(metaObj, maxCores = 4)

# View results
head(results$metaAnalysis$pooledResults)
```

#### Python Equivalent

```python
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.labeling import assign_class_labels
from metaintegrator.analysis import MetaAnalyzer

# Get data (with automatic caching!)
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(["GSE1234", "GSE5678"])

# Build meta-object from ProcessedDataset objects and assign class labels
meta_object = {"originalData": {}}
for ds in datasets:
    dataset = {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
               "formattedName": ds.geo_id, "class": None}
    meta_object["originalData"][ds.geo_id] = assign_class_labels(
        dataset, column_name="title", disease_terms=["disease"]
    )

# Run analysis
analyzer = MetaAnalyzer(max_cores=4)
results = analyzer.run_meta_analysis(meta_object)

# View results
print(results["metaAnalysis"]["pooledResults"].head())
```

---

## Syntax Translation

### Basic Syntax Differences

| Concept | R | Python |
|---------|---|--------|
| **Assignment** | `<-` or `=` | `=` |
| **Indexing** | `data[1]` (1-based) | `data[0]` (0-based) |
| **Lists** | `list(a=1, b=2)` | `{"a": 1, "b": 2}` |
| **Vectors** | `c(1, 2, 3)` | `[1, 2, 3]` or `np.array([1,2,3])` |
| **Data Frames** | `data.frame()` | `pd.DataFrame()` |
| **Column Access** | `df$column` | `df["column"]` or `df.column` |
| **Row Subset** | `df[df$value > 5, ]` | `df[df["value"] > 5]` |
| **Function Def** | `myFunc <- function(x) { ... }` | `def my_func(x): ...` |
| **Comments** | `# comment` | `# comment` |
| **Print** | `print(x)` or just `x` | `print(x)` |
| **Help** | `?function` | `help(function)` |

### Indexing Examples

#### R (1-based)
```r
# First element
vec <- c(10, 20, 30, 40)
vec[1]  # 10

# Slice
vec[2:4]  # c(20, 30, 40)

# Data frame
df[1, ]     # First row
df[, 1]     # First column
df[1, 2]    # First row, second column
```

#### Python (0-based)
```python
import numpy as np
import pandas as pd

# First element
vec = np.array([10, 20, 30, 40])
vec[0]  # 10

# Slice
vec[1:4]  # array([20, 30, 40])

# Data frame
df.iloc[0, :]   # First row
df.iloc[:, 0]   # First column
df.iloc[0, 1]   # First row, second column

# Or by name
df.loc[row_name, column_name]
```

---

## Function Mapping

### Complete Function Translation Table

| R Function | Python Equivalent | Import From | Notes |
|------------|-------------------|-------------|-------|
| **Data Loading** | | | |
| `getGEOData()` | `DataLoaderFactory.create_geo_loader()` → `loader.load_multiple()` | `preprocessing.loader_factory` | Returns `List[ProcessedDataset]`; build `meta_object["originalData"]` from `.expr`/`.pheno`/`.keys`, then call `assign_class_labels()` |
| `checkDataObject()` | `DatasetValidator.validate_meta_object()` | `metaintegrator.validation` | Returns `Tuple[bool, List[str]]`. Also: `check_data_object()` convenience function |
| `subsetOriginalData()` | `subset_original_data()` / `subsetOriginalData()` | `metaintegrator.sample_ops` | Accepts names, indices, or boolean mask |
| **Meta-Analysis** | | | |
| `runMetaAnalysis()` | `MetaAnalyzer.run_meta_analysis()` | `metaintegrator.analysis` | Also: `run_meta_analysis()` convenience function |
| `.effect.sizes()` | `MetaAnalysisCore.calculate_effect_sizes()` | `metaintegrator.core_analysis` | Internal; called automatically by the engine |
| `.pool.inverseVar()` | `MetaAnalysisCore.pool_inverse_variance()` | `metaintegrator.core_analysis` | Supports `method="random"` or `"fixed"` |
| `.sum.of.logs()` | `FisherCombination.sum_of_logs()` | `metaintegrator.core_analysis` | Class-based |
| **Filtering** | | | |
| `filterGenes()` | `MetaAnalyzer.filter_genes()` / `filter_genes()` | `metaintegrator.analysis` | Returns `FilterResult` dataclass (not a plain dict) |
| `getMostRecentFilter()` | `meta_object["filterResults"]` | — | Returns `FilterResult` dataclass; access via `.pos_gene_names` or `["posGeneNames"]` |
| **Visualization** | | | |
| `forestPlot()` | `forest_plot()` | `visualization.enhanced_plotting` | Standalone function; also `gg_forest_plot()` |
| `violinPlot()` | `violin_plot()` / `multi_violin_plot()` | `visualization.violin_plot` | R-parity ggpubr styling |
| `volcanoPlot()` | `MetaAnalysisPlotter.plot_volcano()` | `visualization.enhanced_plotting` | — |
| `heatmapPlot()` | `heatmap_plot()` | `visualization.enhanced_plotting` | Standalone function |
| `multipleROCPlot()` | `multiple_roc_plot()` | `visualization.enhanced_plotting` | — |
| `multiplePRCPlot()` | `multiple_prc_plot()` | `visualization.enhanced_plotting` | — |
| **ROC/PRC** | | | |
| `calculateROC()` | `ROCAnalysisEngine.calculate_roc()` | `metaintegrator.roc_analysis_engine` | — |
| `calculatePRC()` | `PRCAnalyzer.calculate_prc()` | `metaintegrator.prc_analysis` | — |
| **Scoring** | | | |
| `calculateScore()` | `ScoreCalculator.calculate_score()` | `metaintegrator.score_calculation` | Also: `calculateScore()` convenience function |
| **Power Analysis** | | | |
| `calculatePower()` | `PowerCalculator` | `metaintegrator.power_calculation` | — |
| **Utilities** | | | |
| `make.names()` | `DatasetValidator.make_valid_name()` | `metaintegrator.validation` | Python identifier rules |
| `sapply()` | List comprehension or `map()` | Built-in | More Pythonic |
| `lapply()` | List comprehension or `map()` | Built-in | More Pythonic |

---

## Data Structure Conversion

### R Meta Object → Python Meta Object

#### R Structure
```r
metaObject <- list(
  originalData = list(
    Study1 = list(
      expr = matrix(...),
      class = c(0, 0, 1, 1, 1),
      keys = c("GENE1", "GENE2", "GENE3"),
      formattedName = "Study 1"
    ),
    Study2 = list(...)
  ),
  metaAnalysis = list(
    pooledResults = data.frame(...),
    datasetEffectSizes = matrix(...),
    ...
  )
)
```

#### Python Equivalent
```python
meta_object = {
    "originalData": {
        "Study1": {
            "expr": pd.DataFrame(...),  # Not matrix!
            "class": np.array([0, 0, 1, 1, 1]),
            "keys": ["GENE1", "GENE2", "GENE3"],  # List, not character vector
            "formattedName": "Study 1"
        },
        "Study2": {...}
    },
    "metaAnalysis": {
        "pooledResults": pd.DataFrame(...),
        "datasetEffectSizes": pd.DataFrame(...),  # Not matrix!
        ...
    }
}
```

### Key Differences

| Component | R Type | Python Type |
|-----------|--------|-------------|
| **Meta object** | `list` | `dict` |
| **Study collection** | Named `list` | `dict` with string keys |
| **Expression matrix** | `matrix` or `data.frame` | `pd.DataFrame` |
| **Class vector** | `numeric` vector | `np.ndarray` |
| **Gene keys** | `character` vector | `list` of strings |
| **Results** | `data.frame` | `pd.DataFrame` |

---

## Common Workflows

### Workflow 1: Complete Meta-Analysis

#### R Code
```r
library(MetaIntegrator)

# 1. Get data
metaObj <- getGEOData(c("GSE1234", "GSE5678", "GSE9012"))

# 2. Check data
checkDataObject(metaObj, "Meta", "Pre-Analysis")

# 3. Run analysis
results <- runMetaAnalysis(metaObj, 
                           runLeaveOneOutAnalysis = TRUE,
                           maxCores = 4)

# 4. Filter genes
filtered <- filterGenes(results,
                       effectSizeThresh = 0.5,
                       FDRThresh = 0.05,
                       numberStudiesThresh = 2)

# 5. Plot
forestPlot(results, filtered$posGeneNames[1:10])
```

#### Python Translation

```python
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.labeling import assign_class_labels
from metaintegrator.validation import DatasetValidator
from metaintegrator.analysis import MetaAnalyzer
from visualization.enhanced_plotting import forest_plot

# 1. Get data
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(["GSE1234", "GSE5678", "GSE9012"])

meta_object = {"originalData": {}}
for ds in datasets:
    dataset = {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
               "formattedName": ds.geo_id, "class": None}
    meta_object["originalData"][ds.geo_id] = assign_class_labels(
        dataset, column_name="title", disease_terms=["disease"]
    )

# 2. Check data
is_valid, errors = DatasetValidator.validate_meta_object(
    meta_object,
    stage="Pre-Analysis"
)
if not is_valid:
    print("Errors:", errors)

# 3. Run analysis
analyzer = MetaAnalyzer(max_cores=4)
results = analyzer.run_meta_analysis(
    meta_object,
    run_leave_one_out=True
)

# 4. Filter genes (equivalent of R filterGenes)
results = analyzer.filter_genes(
    results,
    effect_size_thresh=0.5,
    fdr_thresh=0.05,
    number_studies_thresh=2
)

# Get up-regulated genes (FilterResult dataclass)
up_genes = results["filterResults"].pos_gene_names

# 5. Plot (forest_plot takes one gene at a time)
for gene in up_genes[:10]:
    forest_plot(results, gene_name=gene)
```

### Workflow 2: Custom Data Analysis

#### R Code
```r
# Load custom data
expr1 <- read.csv("expr1.csv", row.names=1)
pheno1 <- read.csv("pheno1.csv")

# Create study object
study1 <- list(
  expr = as.matrix(expr1),
  class = ifelse(pheno1$condition == "disease", 1, 0),
  keys = rownames(expr1),
  formattedName = "Study 1"
)

# Create meta object
metaObj <- list(originalData = list(Study1 = study1))

# Run analysis
results <- runMetaAnalysis(metaObj)
```

#### Python Translation
```python
import pandas as pd
import numpy as np

# Load custom data
expr1 = pd.read_csv("expr1.csv", index_col=0)
pheno1 = pd.read_csv("pheno1.csv")

# Create study object
study1 = {
    "expr": expr1,
    "class": (pheno1["condition"] == "disease").astype(int).values,
    "keys": expr1.index.tolist(),
    "formattedName": "Study 1"
}

# Create meta object
meta_object = {"originalData": {"Study1": study1}}

# Run analysis
analyzer = MetaAnalyzer()
results = analyzer.run_meta_analysis(meta_object)
```

---

## Pitfalls and Solutions

### Pitfall 1: Indexing Confusion

**Problem**: R uses 1-based indexing, Python uses 0-based.

❌ **Wrong (R thinking in Python)**:
```python
# Get first gene
first_gene = pooled.iloc[1]  # This gets the SECOND gene!
```

✅ **Correct**:
```python
# Get first gene
first_gene = pooled.iloc[0]  # First gene
```

**Solution**: Always remember Python is 0-indexed.

---

### Pitfall 2: Data Frame Column Access

**Problem**: R's `$` operator doesn't exist in Python.

❌ **Wrong**:
```python
# R-style column access
pooled$effectSize  # SyntaxError!
```

✅ **Correct**:
```python
# Python-style column access
pooled["effectSize"]
# Or
pooled.effectSize  # If column name is valid Python identifier
```

---

### Pitfall 3: Matrix vs DataFrame

**Problem**: R uses matrices, Python uses DataFrames.

❌ **Wrong**:
```python
# Treating DataFrame like matrix
expr_matrix = np.array(geo_data["expr"])  # Loses row/column names!
```

✅ **Correct**:
```python
# Keep as DataFrame
expr_df = geo_data["expr"]  # pd.DataFrame with names
```

---

### Pitfall 4: Boolean Vectors

**Problem**: R and Python handle boolean filtering differently.

#### R
```r
# Filter rows
significant <- pooled[pooled$effectSizeFDR < 0.05, ]
```

#### Python
```python
# Don't forget the comma in R becomes just brackets in Python
significant = pooled[pooled["effectSizeFDR"] < 0.05]  # No comma!
```

---

### Pitfall 5: Function Return Values

**Problem**: Python requires explicit `return`, R doesn't.

#### R
```r
myFunc <- function(x) {
  x * 2  # Implicit return
}
```

#### Python
```python
def my_func(x):
    return x * 2  # Explicit return required
```

---

### Pitfall 6: Missing Value Handling

**Problem**: R's `NA` is not the same as Python's `np.nan`.

#### R
```r
# Check for missing
is.na(x)
# Remove missing
x[!is.na(x)]
```

#### Python
```python
# Check for missing
pd.isna(x)  # or np.isnan(x)
# Remove missing
x[~pd.isna(x)]  # Note the ~, not !
```

---

## Best Practices

### 1. Use Type Hints

```python
from typing import Dict, List
import pandas as pd
import numpy as np

def prepare_study(
    expr: pd.DataFrame,
    phenotype: np.ndarray,
    genes: List[str]
) -> Dict:
    """
    Prepare study object with type safety.
    """
    return {
        "expr": expr,
        "class": phenotype,
        "keys": genes
    }
```

### 2. Leverage Pandas Methods

Instead of loops:
```python
# ❌ Inefficient
results = []
for gene in gene_list:
    value = pooled[pooled.index == gene]["effectSize"].values[0]
    results.append(value)

# ✅ Efficient
results = pooled.loc[gene_list, "effectSize"].tolist()
```

### 3. Use Context Managers

```python
# ✅ Automatic resource cleanup
import matplotlib.pyplot as plt

with plt.style.context('seaborn-v0_8'):
    fig = plotter.plot_volcano(results)
    plt.savefig("volcano.png")
# Style automatically reset after block
```

### 4. Validate Early

```python
# ✅ Catch errors early
from metaintegrator.validation import DatasetValidator

is_valid, errors = DatasetValidator.validate_meta_object(meta_object)
if not is_valid:
    raise ValueError(f"Invalid meta-object:\\n" + "\\n".join(errors))
```

### 5. Use List Comprehensions

Instead of `sapply`/`lapply`:
```python
# R: sapply(datasets, function(d) nrow(d$expr))
# Python:
gene_counts = [dataset["expr"].shape[0] for dataset in datasets.values()]
```

---

## Next Steps

### Learning Resources

1. **Python for R Users**: [pandas comparison with R](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_r.html)
2. **Pandas for R Users**: [R vs Python data wrangling](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_r.html)
3. **NumPy for R Users**: [NumPy for absolute beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)

### Migration Checklist

- [ ] Install Python and required packages
- [ ] Convert R data structures to Python equivalents
- [ ] Update function calls using mapping table
- [ ] Fix indexing (1-based → 0-based)
- [ ] Test with small datasets first
- [ ] Validate results match R outputs
- [ ] Update visualization code
- [ ] Document any custom adaptations

### Getting Help

- **Documentation**: [API Reference](../api/overview.md)
- **Examples**: [Tutorials](../tutorials/complete_workflow.md)
- **Comparison**: [R vs Python](r_vs_python.md)
- **Community**: GitHub Issues

---

## Summary

Migrating from R to Python requires adapting to:
- **0-based indexing** instead of 1-based
- **Explicit returns** in functions
- **Different syntax** for data access
- **Class-based API** instead of functional

But you gain:
- **Automatic caching** for GEO data (avoids redundant downloads)
- **Type safety** with type hints
- **Modern tooling** for production deployment
- **Equivalent statistical results** — same algorithms, same outputs

The migration effort is **moderate** for standard workflows and **worthwhile** for production deployments or ML integration.
