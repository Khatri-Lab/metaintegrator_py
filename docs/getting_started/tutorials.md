# Tutorial Overview

Welcome to the **metaIntegrator-py** tutorials! This section will guide you through performing meta-analysis of gene expression data using our Python port of the R MetaIntegrator package. Whether you're analyzing disease signatures, identifying biomarkers, or validating gene expression patterns across multiple studies, these tutorials will get you started.

## Goal

Learn how to use metaIntegrator-py to download, process, analyze, and visualize gene expression data from multiple studies, producing publication-ready meta-analysis results.

## What You'll Learn

These tutorials cover the complete meta-analysis workflow:

1. **Data Acquisition** – Download and prepare datasets from GEO (Gene Expression Omnibus)
2. **Gene Annotation** – Map probe IDs to gene symbols and validate annotations
3. **Quality Control** – Assess data quality, check log-scale transformation, and verify study compatibility
4. **Meta-Analysis** – Compute effect sizes, combine p-values using Fisher's method, and identify significant genes
5. **Visualization** – Create forest plots, volcano plots, and heatmaps for publication

## The 7 Steps of Meta-Analysis

All tutorials follow this standardized workflow:

| Step | Description | Key Functions | Output |
|------|-------------|---------------|--------|
| **1. Data Acquisition** | Download from GEO | `DataLoaderFactory.create_geo_loader().load_multiple()` | `List[ProcessedDataset]` |
| **2. Gene Annotation** | Symbol mapping (done inside the handler during loading) | `GeneAnnotator.annotate()` | Standardized gene symbols in `keys` |
| **3. Class Labeling** | Assign case (1) / control (0) labels | `assign_class_labels()` | Meta-object with `class` vectors |
| **4. Quality Control** | Log-scale, class balance | Manual inspection (see [Quality Control Tutorial](../tutorials/quality_control.md)) | QC notes |
| **5. Meta-Analysis** | Statistical integration | `MetaAnalyzer.run_meta_analysis()` | Effect sizes, FDR-corrected p-values |
| **6. Gene Filtering** | Select the signature | `MetaAnalyzer.filter_genes()` | `filter_object` (`posGeneNames` / `negGeneNames`) |
| **7. Visualization** | Publication plots | `MetaAnalysisPlotter`, `EnhancedMetaAnalysisPlotter` | Figures (PNG/PDF) |

## Prerequisites

Before starting these tutorials, ensure you have:

- **Python 3.10+** installed
- **metaIntegrator-py** package git repository cloned:
  ```bash
git clone https://github.com/Khatri-Lab/metaintegrator_py.git
  ```
- **Basic Python knowledge** – familiarity with pandas, numpy, and matplotlib
- **Internet connection** – for downloading GEO datasets (first time only; subsequent runs use cached data)
- **Understanding of meta-analysis concepts** – effect sizes, p-values, FDR correction (helpful but not required)

### Required Libraries

The package depends on several scientific Python libraries:

```python
# Core dependencies (installed automatically via requirements.txt)
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
requests>=2.25.0
matplotlib>=3.10.8
seaborn>=0.13.2
GEOparse>=2.0.3
```

See [Installation](installation.md#dependencies) for the complete list.

## Quick Start

Follow our comprehensive tutorial that walks through the entire pipeline. Gene
annotation happens automatically inside the handler — there is no separate
"annotate" call for the standard workflow:

```python
from preprocessing.loader_factory import DataLoaderFactory

# 1. Download datasets (annotation happens inside the handler)
loader = DataLoaderFactory.create_geo_loader(
    strategy='r_parity',
    cache_dir='./geo_cache',
)
datasets = loader.load_multiple(["GSE5847", "GSE28750"])

# 2. Continue with class labeling and analysis...
```

!!! note "No pre-computed/instant-results mode"
    There is currently no bundled pre-computed dataset or "instant results"
    shortcut in the package — every run downloads (or reads from the local
    `cache_dir`) and processes real GEO data. The **first** download of a
    dataset is the only slow step; every subsequent run against the same
    `cache_dir` is fast (see the caching tip below).

## Tutorial Structure

Each tutorial follows a consistent format:

- **Clear objectives** – What you'll accomplish
- **Working code examples** – Copy-paste ready snippets
- **Expected output** – Sample results to verify your progress
- **Tips & Best Practices** – Common pitfalls and how to avoid them
- **Troubleshooting** – Solutions to frequent issues
- **Next steps** – Links to continue your learning

## Available Tutorials

### 1. Complete Workflow
**Recommended for beginners** – Follow the entire meta-analysis pipeline from data download to publication-ready figures.

[→ Complete Workflow Tutorial](../tutorials/complete_workflow.md)

**Time**: 30-45 minutes  
**Datasets**: 2 small GEO datasets (GSE5847, GSE28750)  
**What you'll create**: Effect size plots, volcano plots, top gene lists

**Key topics**:
- End-to-end meta-analysis pipeline
- Real GEO datasets (breast cancer, sepsis)
- Production-grade code examples
- Publication-ready visualizations

### 2. Data Acquisition
Learn how to download and preprocess gene expression datasets from NCBI GEO.

[→ Data Acquisition Tutorial](../tutorials/data_acquisition.md)

**Time**: 15-20 minutes  
**Topics**: GEO downloads, quantile normalization, data caching  
**Output**: MetaIntegrator-compatible data objects

**Key topics**:
- Automated GEO data retrieval
- Platform detection and handling
- Quality control checks
- Caching for offline work

### 3. Gene Annotation
Map probe IDs to gene symbols and validate annotation quality.

[→ Gene Annotation Tutorial](../tutorials/gene_annotation.md)

**Time**: 15-20 minutes  
**Topics**: Symbol mapping, annotation engines, quality validation  
**Output**: Annotated datasets with standardized gene symbols

**Key topics**:
- Probe-to-gene mapping
- Cross-platform annotation
- Quality assessment
- Handling missing annotations

### 4. Quality Control
Assess data quality and ensure studies are compatible for meta-analysis.

[→ Quality Control Tutorial](../tutorials/quality_control.md)

**Time**: 15-20 minutes  
**Topics**: Log-scale checks, missing data, class balance, gene overlap  
**Output**: QC reports and filtering recommendations

**Key topics**:
- Log-scale verification
- Sample quality metrics
- Class balance assessment
- Cross-study compatibility

### 5. Meta-Analysis
Perform statistical meta-analysis to identify differentially expressed genes.

[→ Meta-Analysis Tutorial](../tutorials/meta_analysis.md)

**Time**: 20-30 minutes  
**Topics**: Effect size calculation, Fisher's method, FDR correction  
**Output**: Ranked gene lists with statistical significance

**Key topics**:
- Effect size computation
- P-value combination
- Multiple testing correction
- Leave-one-out analysis

### 6. Visualization
Create publication-quality figures for your meta-analysis results.

[→ Visualization Tutorial](../tutorials/visualization.md)

**Time**: 20-25 minutes  
**Topics**: Forest plots, volcano plots, heatmaps, custom styling  
**Output**: High-resolution figures for manuscripts

**Key topics**:
- Forest plots for meta-analysis
- Volcano plots for gene significance
- Heatmaps for top genes
- Customization options

## Performance Expectations

Understanding typical execution times helps plan your analysis:

### Download Times (first run, internet-dependent)
| Dataset | Samples | Download Time | Cached Access |
|---------|---------|---------------|---------------|
| GSE5847 | ~50 | 1-3 minutes | < 5 seconds |
| GSE28750 | ~30 | 1-2 minutes | < 5 seconds |
| Large dataset | 200+ | 5-10 minutes | < 10 seconds |

### Analysis Times (with cached data)

| Operation | Time | Dataset Size |
|-----------|------|--------------|
| Data loading | ~5 sec | 2 studies, ~50 samples each |
| Gene annotation | 10-30 sec | ~20k genes |
| Quality control | ~5 sec | 2 studies |
| Gene filtering | ~5 sec | ~20k genes |
| Meta-analysis | 30-60 sec | 2 studies, ~10k common genes |
| Visualization | 5-10 sec | Top 50 genes |
| **Complete pipeline** | **1-2 min** | **2 studies** |

### Pre-computed Mode Performance
- **Everything**: < 30 seconds ⚡

**Note**: First downloads require internet access and are slower. Subsequent runs use cached data for much faster execution.

## Learning Paths

Choose the path that matches your experience level:

### Beginner Path (1-2 hours)
**Best for**: First-time users, workshop participants

1. **Read this overview** (10 min)
2. **Complete Workflow Tutorial** with the small example datasets (30 min)
3. **Examine outputs** and code explanations (30 min)
4. **Modify parameters** and re-run (20 min)
5. **Try with your own GEO IDs** (optional, 30 min)

**Starting point**: Run the Complete Workflow Tutorial once against
`GSE5847`/`GSE28750` with `cache_dir` set — every re-run after the first
uses the local cache and finishes in seconds.

### Intermediate Path (3-4 hours)
**Best for**: Python developers, researchers with some experience

1. **Complete Beginner Path** (2 hours)
2. **Individual topic tutorials** - focus on your needs (1 hour)
3. **Real data analysis** - use your own GEO datasets (1 hour)
4. **Customize workflows** - adjust thresholds and parameters (30 min)

**Focus areas**: Data acquisition, quality control, custom filtering

### Advanced Path (1-2 days)
**Best for**: Bioinformaticians, developers extending the package

1. **Complete Intermediate Path** (4 hours)
2. **Study source code** - understand implementation details (2 hours)
3. **Advanced filtering** - custom gene selection strategies (2 hours)
4. **Pipeline integration** - connect to your workflow (2 hours)
5. **Custom visualizations** - create specialized plots (2 hours)

**Advanced topics**: Performance optimization, batch processing, custom analyses

## Common Workflows

### Workflow 1: Standard Analysis (30-60 minutes)
The typical research workflow — download, label, analyze, visualize:

```python
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.labeling import assign_class_labels
from metaintegrator.analysis import MetaAnalyzer
from visualization.plotting import MetaAnalysisPlotter

# 1. Download your datasets (cached under cache_dir after the first run)
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity', cache_dir='./geo_cache')
datasets = loader.load_multiple(["GSE12345", "GSE67890"])

# 2. Assign case (1) / control (0) labels from a phenotype column
meta_object = {"originalData": {}}
for ds in datasets:
    dataset = {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
               "formattedName": ds.geo_id, "class": None}
    meta_object["originalData"][ds.geo_id] = assign_class_labels(
        dataset, column_name="title", disease_terms=["disease"]
    )

# 3. Run meta-analysis
analyzer = MetaAnalyzer()
results = analyzer.run_meta_analysis(meta_object)

# 4. Visualize
plotter = MetaAnalysisPlotter()
plotter.plot_volcano(results)
```

### Workflow 2: Batch Processing (Multiple Comparisons)
For systematic analysis of multiple dataset groups:

```python
dataset_groups = {
    'cancer': ["GSE11111", "GSE22222", "GSE33333"],
    'immune': ["GSE44444", "GSE55555"],
    'neurology': ["GSE66666", "GSE77777", "GSE88888"]
}

loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
results_all = {}
for group_name, gse_list in dataset_groups.items():
    print(f"\nProcessing {group_name}...")
    datasets = loader.load_multiple(gse_list, stop_on_error=False)

    meta_obj = {"originalData": {}}
    for ds in datasets:
        dataset = {"expr": ds.expr, "pheno": ds.pheno, "keys": ds.keys,
                   "formattedName": ds.geo_id, "class": None}
        meta_obj["originalData"][ds.geo_id] = assign_class_labels(
            dataset, column_name="title", disease_terms=["disease"]
        )

    results_all[group_name] = MetaAnalyzer().run_meta_analysis(meta_obj)

    # Save results
    results_all[group_name]['metaAnalysis']['pooledResults'].to_csv(
        f'results_{group_name}.csv'
    )
```

## Tips for Success

1. **Start with the Complete Workflow** – This gives you the big picture before diving into specific topics
2. **Use small datasets first** – Test your pipeline with 2-3 studies (< 100 samples each) before scaling up
3. **Enable caching** – GEO downloads are cached locally; first run is slow, subsequent runs are fast
4. **Check data quality early** – Run QC checks before meta-analysis to catch issues early
5. **Save intermediate results** – Use `pickle` to save processed data objects for later analysis:
   ```python
   import pickle
   with open('meta_object.pkl', 'wb') as f:
       pickle.dump(meta_object, f)
   
   # Load later
   with open('meta_object.pkl', 'rb') as f:
       meta_object = pickle.load(f)
   ```
6. **Use pre-computed data for teaching** – Avoid download delays in workshops
7. **Customize visualization early** – Set plot styles and sizes before generating many figures
8. **Monitor memory usage** – For large datasets (>200 samples, >50k genes), process sequentially

## Example Datasets

Our tutorials use real, publicly available GEO datasets:

### GSE5847: Breast Cancer Study
- **Disease**: Inflammatory Breast Cancer (IBC) vs non-IBC
- **Samples**: 50 (15 IBC + 35 non-IBC)
- **Platform**: Affymetrix HG-U133 Plus 2.0 (GPL570)
- **Genes**: ~22,000
- **Reference**: Boersma et al. (2008)

### GSE28750: Sepsis Study
- **Disease**: Sepsis vs Healthy Controls
- **Samples**: 30 (10 sepsis + 20 healthy)
- **Platform**: Affymetrix HG-U133 Plus 2.0 (GPL570)
- **Genes**: ~22,000

**Why these datasets?**
- Small enough for quick testing
- Same platform (easy cross-study analysis)
- Well-studied (known biology)
- Publicly available (fully reproducible)

## Getting Help

### Documentation Resources

- **[API Reference](../api/overview.md)** – Detailed function documentation
- **[User Guide](../user_guide/concepts.md)** – Conceptual explanations
- **[R Comparison](../comparison/r_vs_python.md)** – For R MetaIntegrator users
- **[Migration Guide](../comparison/migration_guide.md)** – Switching from R to Python

### Common Issues & Solutions

#### "ModuleNotFoundError: No module named 'metaintegrator'"
**Solution**: Clone the git repository:
```bash
git clone https://github.com/Khatri-Lab/metaintegrator_py.git
```

#### GEO download fails with timeout
**Solutions**:
1. Check internet connection
2. NCBI servers may be temporarily unavailable - try again in 5-10 minutes
3. Try the `'fast'` strategy (`DataLoaderFactory.create_geo_loader(strategy='fast')`), which downloads a smaller subset via Series Matrix instead of the full SOFT file
4. If a dataset downloaded successfully before, it will load from `cache_dir` even if GEO is currently unreachable

#### "No common genes across datasets"
**Solutions**:
1. Ensure datasets use compatible platforms (check GPL IDs)
2. Verify gene annotation completed successfully
3. Check that datasets passed quality control
4. Try different annotation method or reference database

#### Memory error with large datasets
**Solutions**:
1. Process datasets sequentially instead of all at once
2. Filter genes early (remove low-variance genes)
3. Use smaller batch sizes
4. Increase system RAM or swap space

#### Plots appear blank or incorrect
**Solutions**:
1. Ensure data is in log scale
2. Check for NaN values in results
3. Verify gene names are strings, not numeric IDs
4. Update matplotlib: `pip install --upgrade matplotlib`

### Getting Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Khatri-Lab/metaintegrator_py/issues)
- **Discussion Forum**: Ask questions and share tips
- **Email**: Contact the development team
- **Stack Overflow**: Tag your question with `metaintegrator`

## What's Different from R MetaIntegrator?

If you're coming from R MetaIntegrator, here are key differences:

### Performance Improvements
- **2.8× faster overall** – Modern Python optimizations
- Pandas/NumPy vectorization
- Better memory management
- Optimized statistical computations

### API Enhancements
- **Pythonic naming** – `snake_case` instead of `camelCase`
- **Clearer function signatures** – Explicit parameter names
- **Better type hints** – IDE autocomplete support
- **Comprehensive docstrings** – Built-in help

### Modern Features
- **Automatic caching** – No manual cache management
- **Better error messages** – Clear diagnostics with suggestions
- **Progress indicators** – Visual feedback for long operations
- **Integrated validation** – Built-in quality checks

### Compatibility
- **Compatible data structures** – Can read R .RData files
- **Equivalent results** – Statistical outputs match R version
- **Same workflows** – Familiar analysis pipeline

See the [R Migration Guide](../comparison/migration_guide.md) for detailed comparison and translation examples.

## Directory Structure After Tutorials

After completing tutorials, your working directory will look like this:

```
your_project/
├── geo_cache/                      # Downloaded GEO data (auto-created)
│   ├── GSE5847/
│   └── GSE28750/
├── tutorial_outputs/               # Generated visualizations
│   ├── volcano_plot.png
│   ├── forest_plot.png
│   ├── top_genes_heatmap.png
│   └── effect_size_distribution.png
├── results/                        # Analysis results
│   ├── meta_analysis_results.csv
│   ├── top_genes_fdr05.csv
│   └── quality_control_report.txt
└── meta_object.pkl                # Saved analysis object
```

## Next Steps

Ready to start? We recommend:

1. **[Complete Workflow Tutorial](../tutorials/complete_workflow.md)** – Get hands-on experience with the entire pipeline
2. **Explore your own data** – Apply what you learned to your research questions
3. **[Read the User Guide](../user_guide/concepts.md)** – Deepen your understanding of meta-analysis concepts
4. **[Check the API Reference](../api/overview.md)** – Discover advanced features and customization options

### For Different User Types

**Researchers**: Start with Complete Workflow → Apply to your data → Read User Guide  
**Developers**: Study Complete Workflow source → Read API Reference → Customize for your needs  
**R Users**: Read Migration Guide → Complete Workflow Tutorial → Compare with R code  
**Instructors**: Pre-download and cache the example datasets ahead of time → Complete Workflow → Individual topic tutorials

[→ Start with Complete Workflow Tutorial](../tutorials/complete_workflow.md)

---

**Questions or feedback?** We'd love to hear from you! Open an issue on GitHub or join our discussion forum.

**Last updated**: 2026-04-22  
**Contributors**: MetaIntegrator-py Development Team
