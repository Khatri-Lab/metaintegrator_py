# Quick Start

## Preprocessing (Optional)

Before meta-analysis, consider batch normalization if combining
data from multiple platforms. See the 
[COCONUT Quick Start](../coconut/quickstart.md).

## 5-Minute Tutorial

### 1. Load Data

```python
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.labeling import assign_class_labels

# Download datasets from GEO (with automatic caching)
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(["GSE5847", "GSE28750"])

# Convert to meta-object format and assign case (1) / control (0)
# labels from a phenotype column — ProcessedDataset itself has no
# "class" field, so this is always a separate, manual step
meta_object = {"originalData": {}}
for ds in datasets:
    dataset = {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
               "formattedName": ds.geo_id, "class": None}
    meta_object["originalData"][ds.geo_id] = assign_class_labels(
        dataset, column_name="title", disease_terms=["disease"]
    )
```

!!! tip "Automatic Caching"
    The first download takes 2–5 minutes per dataset. Subsequent runs
    load from the local cache in seconds.

### 2. Run Meta-Analysis

```python
from metaintegrator.analysis import MetaAnalyzer

# Initialize analyzer
analyzer = MetaAnalyzer()

# Run analysis (validates input, computes effect sizes, runs leave-one-out)
results = analyzer.run_meta_analysis(meta_object)
```

### 3. View Results

```python
# Top significant genes
pooled = results["metaAnalysis"]["pooledResults"]
print(pooled.nsmallest(10, "effectSizeFDR")[["effectSize", "effectSizeFDR"]])
```

### 4. Filter Genes (Optional)

```python
# Apply gene filtering thresholds
results = analyzer.filter_genes(
    results,
    effect_size_thresh=0.5,
    fdr_thresh=0.05,
)

# Access filtered gene lists
sig = results["filterResults"]
print(f"Up-regulated: {sig.n_pos}, Down-regulated: {sig.n_neg}")
```

## Next Steps

- [Complete Tutorial](../tutorials/complete_workflow.md)
- [User Guide](../user_guide/concepts.md)
- [API Reference](../api/overview.md)
