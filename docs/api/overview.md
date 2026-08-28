# API Reference

MetaIntegrator-py is organized into four packages under `src/`. Each follows
object-oriented design with clear separation of concerns.

For the OO architecture, factory patterns, and data-flow diagrams see the
[Architecture Overview](../architecture.md).

## Quick Navigation

| Package | Focus | Key entry points |
|---------|-------|-----------------|
| [Data Validation & Processing](data_loading.md) | Validation, subsetting, labeling | `DatasetValidator`, `subset_original_data` |
| [Preprocessing](preprocessing.md) | Data acquisition & gene annotation | `DataLoaderFactory`, `GeneAnnotator` |
| [Analysis](analysis.md) | Meta-analysis & gene filtering | `MetaAnalyzer`, `MetaAnalysisFactory`, `ScoreCalculator` |
| [Visualization](visualization.md) | Publication-ready plots | `MetaAnalysisPlotter`, `violin_plot` |
| [COCONUT](../coconut/api.md) | Batch normalization | `coconut()`, `runner()` |

---

## Package: `preprocessing`

Data acquisition layer using a **Factory → Source → Handler** pattern.

| Module | Class / Function | Purpose |
|--------|-----------------|---------|
| `loader_factory` | `DataLoaderFactory` | Single entry point — creates loaders for GEO strategies |
| `base` | `DataSource`, `DataHandler`, `DataLoader`, `ProcessedDataset` | Abstract base classes |
| `hybrid_handler` | `HybridMicroarrayHandler` | **Production handler** — Series Matrix + GEOparse annotation |
| `hybrid_source` | `HybridGeoSource` | Fetches expr from Series Matrix, annotation from GEOparse |
| `gene_annotator` | `GeneAnnotator` | Probe → gene symbol with GenBank accession filter |
| `gene_validation` | `EnsemblGeneValidator` | Filters non-HGNC symbols via bundled Ensembl gene list |
| `gene_converter` | `GeneConverter` | Any gene ID (Ensembl/RefSeq/GenBank/Entrez) → HUGO symbol |
| `ncbi_network` | `NCBINetwork` | Low-level FTP/HTTPS transport with retry & disk caching — the injectable seam used by every source |
| `ncbi_fetcher` | `NCBIFetcher` | Series Matrix downloader behind the `'fast'` strategy (size guard, pickle caching) |
| `microarray_handler` | `MicroarrayHandler` | Full SOFT-based handler (GEOparse), backs the `'full'`/`'fast'` strategies |
| `r_parity_handler`, `r_parity_source` | `RParityMicroarrayHandler`, `RParityGeoSource` | **Deprecated aliases** — re-export `HybridMicroarrayHandler`/`HybridGeoSource` under their old names, not separate implementations |

```python
from preprocessing.loader_factory import DataLoaderFactory

loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(['GSE11909', 'GSE50635'])
```

---

## Package: `metaintegrator`

Core analysis, scoring, and gene filtering.

| Module | Class / Function | Purpose |
|--------|-----------------|---------|
| `analysis` | `MetaAnalyzer`, `MetaAnalysisFactory`, `FilterResult`, `filter_genes()` | Top-level orchestrator: effect sizes → pooling → LOOA → simple gene filtering |
| `core_analysis` | `MetaAnalysisCore`, `FisherCombination` | Statistical engine (Hedge's g, inverse-variance, Fisher) |
| `score_calculation` | `ScoreCalculator` | Geometric-mean z-score per sample |
| `sample_level_gene_data` | `SampleLevelGeneDataProcessor` | Probe-to-gene matrix for scoring |
| `scoring_meta_integrator_filter` | `MetaIntegratorFilter` | FDR + effect-size + LOOA gene filtering (the fuller, strategy-based filter — see note below) |
| `scoring_core` | `FilterConfig`, `FilterResults` (+ shared configs/validators) | Config and result dataclasses used by `MetaIntegratorFilter` |
| `scoring_meta_integrator_orchestrator` | `MetaIntegratorOrchestrator` | End-to-end scoring pipeline |
| `scoring_filter_manager` | `FilterManager` | Filter parameter management |
| `scoring_gene_scorer` | `GeneScorer` | Per-gene scoring logic |
| `scoring_results_summarizer` | `ResultSummarizer` | Summary statistics for filtered results |
| `scoring_statistical_analysis` | `StatisticalAnalyzer`, `DistributionAnalyzer` | Additional statistical tests |
| `scoring_export_engine` | `ExportEngine` | Export results to CSV/Excel |
| `roc_analysis_engine` | `ROCAnalysisEngine` | ROC curves, AUC, multi-dataset ROC |
| `prc_analysis` | `PRCAnalyzer` | Precision-recall curves |
| `power_calculation` | `MetaAnalysisPowerCalculator` | Sample-size & power estimation |
| `phenotype_processing` | `PhenotypeProcessor` | Extract class labels from pheno data |
| `validation` | `DatasetValidator` (+ `MetaAnalysisValidator`, `MetaFilterValidator`, `MetaObjectValidator`) | Validate dataset / meta-analysis / filter structures |
| `labeling` | `assign_class_labels()` | Build a class vector from a phenotype column (module-level function, not a class) |
| `biological_annotation` | `BiologicalAnnotator` | Gene set enrichment & pathway annotation |
| `clinical_metrics_analyzer` | `ClinicalParametersAnalyzer`, `YoudenJAnalyzer` | Sensitivity, specificity, PPV/NPV, optimal cutpoint |
| `meta_analysis_config` | `validate_config()`, `generate_template()`, `load_config_file()`, `DEFAULTS` | Config schema, validation, and template generation for the `meta_analysis` CLI command |
| `meta_analysis_runner` | `run_pipeline()`, `main()` | 8-stage pipeline orchestrator behind the `meta_analysis` CLI command — see the [Command-Line Interface guide](../getting_started/cli.md) |

> **Two independent gene-filtering paths exist** — `analysis.filter_genes()` /
> `analysis.FilterResult` is the lightweight filter that mirrors R's
> `filterGenes()` field-for-field (see [Object Structures](#filter_object-output-of-gene-filtering)
> below); `scoring_meta_integrator_filter.MetaIntegratorFilter` /
> `scoring_core.FilterResults` is a separate, strategy-based filtering
> pipeline (standard / adaptive / combined-criteria) built on top of the same
> statistics. They are not interchangeable — check which one a given
> workflow or tutorial is actually importing before assuming `filterResults`
> means the same shape in both.

```python
from metaintegrator.analysis import MetaAnalyzer
from metaintegrator.score_calculation import ScoreCalculator

analyzer = MetaAnalyzer()
results = analyzer.run_meta_analysis(meta_object)
scores = ScoreCalculator.calculate_score(filter_object, dataset)
```

---

## Package: `visualization`

Publication-quality plots matching R's MetaIntegrator output.

| Module | Class / Function | Purpose |
|--------|-----------------|---------|
| `plotting` | `MetaAnalysisPlotter` | Volcano, forest, heatmap, effect-size distribution |
| `enhanced_plotting` | `EnhancedMetaAnalysisPlotter`, plus module-level `forest_plot()`, `gg_forest_plot()`, `heatmap_plot()`, `manhattan_plot()`, `predictive_value_plot()`, `multiple_roc_plot()`, `multiple_prc_plot()`, `roc_plot()`, `prc_plot()` | Multi-dataset ROC/PRC, summary panels, and all R-parity convenience plot functions |
| `violin_plot` | `violin_plot()`, `multi_violin_plot()` | R-parity violin plots (ggplot2/ggpubr style) |

`enhanced_plotting`'s plotter class was previously also named
`MetaAnalysisPlotter` — an unrelated class in a different module with a
completely disjoint method set from `plotting.MetaAnalysisPlotter`, and a
real, easy mistake to import the wrong one. It has been renamed to
`EnhancedMetaAnalysisPlotter`; a deprecated, warning-emitting alias is kept
for old `from visualization.enhanced_plotting import MetaAnalysisPlotter`
imports, but new code should import `EnhancedMetaAnalysisPlotter` by name
directly.

```python
from visualization.violin_plot import violin_plot, multi_violin_plot
from visualization.enhanced_plotting import multiple_roc_plot

fig = violin_plot(filter_object, dataset, label_column='label')
fig = multiple_roc_plot(datasets, filter_object)
```

---

## Package: `coconut`

COCONUT (COmbat CO-Normalization Using conTrols) batch normalization.
Unlike the other three packages, this one is **function-based, not
class-based** — there is no `COCONUT` or `COCONUTRunner` class.

| Module | Function | Purpose |
|--------|----------|---------|
| `coconut` | `coconut()` | Core ComBat batch-normalization algorithm |
| `coconut_runner` | `runner()`, `main()`, `load_and_run_coconut_example()` | CLI runner and example driver for COCONUT workflows |

See [COCONUT API](../coconut/api.md) for the full function reference.

---

## Object Structures

### ProcessedDataset (analysis-ready dict format)

This is **not** the literal return type of `preprocessing` — the loaders
return the `ProcessedDataset` *dataclass* (`geo_id`, `expr`, `pheno`, `keys`,
`metadata`; see [Preprocessing API](preprocessing.md#processeddataset)),
which has no `class` field of its own. This dict is the shape callers build
by hand — adding study-specific class labels — before handing a dataset to
`MetaAnalyzer`:

```python
{
    "expr":   pd.DataFrame,   # probes × samples (float64)
    "keys":   List[str],      # gene symbol per probe (comma-sep for multi-gene)
    "class":  np.ndarray,     # 0=control, 1=disease — assigned separately, not part of ProcessedDataset
    "pheno":  pd.DataFrame,   # sample metadata (GSM IDs as index)
    "geo_id": str,            # e.g. "GSE11909_GPL96"
}
```

### Meta-analysis object (after `run_meta_analysis`)

```python
{
    "originalData": {
        "GSE11909_GPL96": { ... },  # dataset dicts, shape above
        "GSE50635_GPL6244": { ... },
    },
    "metaAnalysis": {
        "pooledResults": pd.DataFrame,  # genes × stats
        "datasetEffectSizes": pd.DataFrame,
        "datasetEffectSizeStandardErrors": pd.DataFrame,
    },
    "leaveOneOutAnalysis": { ... },     # optional
}
```

### filter_object (output of gene filtering)

`MetaAnalyzer.filter_genes()` stores a `FilterResult` **dataclass** under
`meta_object["filterResults"]`.  It supports both attribute access and
R-style dict access for backward compatibility. (This is the `analysis`
module's lightweight filter — see the note under [Package:
`metaintegrator`](#package-metaintegrator) above about the separate
`MetaIntegratorFilter`/`FilterResults` pipeline, which is not this class.)

```python
fr = meta_object["filterResults"]

# Attribute access (preferred)
fr.pos_gene_names        # List[str] — up-regulated signature genes
fr.neg_gene_names        # List[str] — down-regulated signature genes
fr.n_pos                 # int — count of positive genes
fr.n_neg                 # int — count of negative genes
fr.effect_size_thresh    # float — effect-size threshold used
fr.fdr_thresh            # float — FDR threshold used
fr.total_genes           # int (property: n_pos + n_neg)

# R-style dict access (backward-compatible)
fr["posGeneNames"]       # aliased to pos_gene_names
fr["negGeneNames"]       # aliased to neg_gene_names
fr["FDRThresh"]          # aliased to fdr_thresh
fr["effectSizeThresh"]   # aliased to effect_size_thresh

# Export to plain dict
fr.to_dict()             # returns R-style keys for serialization
```
