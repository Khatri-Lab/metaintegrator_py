# Metaintegrator Python

!!! success "Welcome to Metaintegrator Python Documentation"
    Modern meta-analysis of gene expression data with optimized Python implementation 
    and the same trusted statistical methods as R MetaIntegrator.

## 🚀 Quick Start

```python
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.labeling import assign_class_labels
from metaintegrator.analysis import MetaAnalyzer

# Download datasets (cached after first run)
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(["GSE5847", "GSE28750"])

# Build the meta-object and assign case (1) / control (0) labels from
# a phenotype column — ProcessedDataset itself has no "class" field
meta_object = {"originalData": {}}
for ds in datasets:
    dataset = {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
               "formattedName": ds.geo_id, "class": None}
    meta_object["originalData"][ds.geo_id] = assign_class_labels(
        dataset, column_name="title", disease_terms=["disease"]
    )

# Run meta-analysis
analyzer = MetaAnalyzer()
results = analyzer.run_meta_analysis(meta_object)

# View significant genes
pooled = results["metaAnalysis"]["pooledResults"]
print(pooled.nsmallest(10, "effectSizeFDR")[["effectSize", "effectSizeFDR"]])
```

## Key Features

- **🥥 COCONUT Batch Normalization**: Remove technical batch effects
- **Meta-Analysis**: Combine results across studies
- **GEO Integration**: Direct data download


## ✨ Features

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Optimized Performance__

    ---

    NumPy/pandas vectorization for efficient computation on large datasets

-   :material-check-all:{ .lg .middle } __Complete Workflow__

    ---

    From GEO download to visualization in **7 simple steps**

-   :material-database:{ .lg .middle } __GEO Integration__

    ---

    Direct download and processing of datasets from **Gene Expression Omnibus**

-   :material-chart-line:{ .lg .middle } __Matplotlib Visualizations__

    ---

    Forest plots, volcano plots, and heatmaps with customizable styling

</div>

## 📚 Documentation Sections

- **[Getting Started](getting_started/installation.md)** - Installation and quick start
- **[Command-Line Interface](getting_started/cli.md)** - Run COCONUT and meta-analysis from the command line, no script required
- **[Tutorials](tutorials/complete_workflow.md)** - Step-by-step guides
- **[User Guide](user_guide/concepts.md)** - Core concepts and best practices
- **[API Reference](api/overview.md)** - Detailed API documentation
- **[R vs Python](comparison/r_vs_python.md)** - Migration guide for R users

## 🎯 Why MetaIntegrator Python?

| Feature | Python | R |
|---------|--------|---|
| **Performance** | NumPy/pandas optimized | R vectorized operations |
| **ML Integration** | ✅ Native (scikit-learn, TensorFlow) | Via reticulate |
| **Deployment** | ✅ Docker, cloud-ready | Shiny apps |
| **Visualization** | matplotlib/seaborn | ✅ ggplot2 (publication-ready) |
| **Ecosystem** | General data science | ✅ Bioconductor specialized |
| **Syntax** | Pythonic, type hints | R functional |

!!! note "Choosing Between Python and R"
    Both implementations provide robust meta-analysis capabilities. Choose based on:
    
    - **Python**: ML pipelines, cloud deployment, Python-first workflows
    - **R**: Bioconductor integration, ggplot2 graphics, R-first workflows

## 📖 Example Workflow

```python
# 1. Data Acquisition — download & annotate (gene annotation happens
#    inside the handler, as part of loading)
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(gse_ids)

# 2. Label Samples — case (1) vs. control (0), per dataset
labeled_dataset = assign_class_labels(dataset, column_name=col, disease_terms=[...])

# 3. Meta-Analysis
results = MetaAnalyzer().run_meta_analysis(meta_object)

# 4. Filter Genes — build the signature
results = MetaAnalyzer().filter_genes(results, effect_size_thresh=1, fdr_thresh=0.05)

# 5. Visualization
EnhancedMetaAnalysisPlotter().forest_plot(gene_name="GENE1", meta_object=results)
```

## 🔗 Quick Links

- [COCONUT Quick Start (5 min)](coconut/quickstart.md)
- [MetaIntegrator Quick Start](getting_started/quickstart.md)
- [Command-Line Interface](getting_started/cli.md)
- [Installation Guide](getting_started/installation.md)
- [Complete Tutorial](tutorials/complete_workflow.md)
- [API Reference](api/overview.md)
- [GitHub Repository](https://github.com/Khatri-Lab/metaintegrator_py)
- [Purvesh Khatri's Lab Repository ](https://khatrilab.stanford.edu/tools/metaintegrator_py)

---

!!! tip "For R Users"
    Check out our [Migration Guide](comparison/migration_guide.md) for side-by-side
    R and Python examples. Both implementations use the same statistical methods
    and produce equivalent analytical results.
