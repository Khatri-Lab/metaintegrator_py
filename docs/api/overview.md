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
| [COCONUT](../coconut/api.md) | Batch normalization | `COCONUT`, `COCONUTRunner` |

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
| `gene_validation` | `EnsemblValidator` | Filters non-HGNC symbols via Ensembl gene list |
| `gene_converter` | `GeneConverter` | Ensembl ↔ Entrez ↔ RefSeq ↔ GenBank translation |
| `ncbi_fetcher` | `NCBIGEOFetcher` | Low-level GEO download with caching & rate-limiting |
| `ncbi_network` | `NCBINetworkClient` | FTP/HTTPS transport with automatic fallback |
| `microarray_handler` | `MicroarrayHandler` | Full SOFT-based handler (GEOparse) |
| `r_parity_handler` | `RParityHandler` | Handler tuned for exact R output matching |
| `precomputed_results` | `PrecomputedResultsLoader` | Load pre-saved analysis results |

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
| `analysis` | `MetaAnalyzer` | Top-level orchestrator: effect sizes → pooling → LOOA |
| `core_analysis` | `MetaAnalysisCore`, `FisherCombination` | Statistical engine (Hedge's g, inverse-variance, Fisher) |
| `score_calculation` | `ScoreCalculator` | Geometric-mean z-score per sample |
| `sample_level_gene_data` | `SampleLevelGeneDataProcessor` | Probe-to-gene matrix for scoring |
| `scoring_meta_integrator_filter` | `MetaIntegratorFilter` | FDR + effect-size + LOOA gene filtering |
| `scoring_meta_integrator_orchestrator` | `ScoringOrchestrator` | End-to-end scoring pipeline |
| `scoring_filter_manager` | `FilterManager` | Filter parameter management |
| `scoring_gene_scorer` | `GeneScorer` | Per-gene scoring logic |
| `scoring_results_summarizer` | `ResultsSummarizer` | Summary statistics for filtered results |
| `scoring_statistical_analysis` | `StatisticalAnalysis` | Additional statistical tests |
| `scoring_export_engine` | `ExportEngine` | Export results to CSV/Excel |
| `roc_analysis_engine` | `ROCAnalysisEngine` | ROC curves, AUC, multi-dataset ROC |
| `prc_analysis` | `PRCAnalysis` | Precision-recall curves |
| `power_calculation` | `PowerCalculation` | Sample-size & power estimation |
| `phenotype_processing` | `PhenotypeProcessor` | Extract class labels from pheno data |
| `validation` | `DataValidator` | Validate dataset structures |
| `labeling` | `Labeling` | Sample labeling utilities |
| `biological_annotation` | `BiologicalAnnotation` | Gene set enrichment & pathway annotation |
| `clinical_metrics_analyzer` | `ClinicalMetricsAnalyzer` | Sensitivity, specificity, PPV/NPV |

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
| `enhanced_plotting` | `MetaAnalysisPlotter` (extended) | Multi-dataset ROC/PRC, summary panels |
| `violin_plot` | `violin_plot()`, `multi_violin_plot()` | R-parity violin plots (ggplot2/ggpubr style) |

```python
from visualization.violin_plot import violin_plot, multi_violin_plot
from visualization.enhanced_plotting import multiple_roc_plot

fig = violin_plot(filter_object, dataset, label_column='label')
fig = multiple_roc_plot(datasets, filter_object)
```

---

## Package: `coconut`

COCONUT (COmbat CO-Normalization Using conTrols) batch normalization.

| Module | Class / Function | Purpose |
|--------|-----------------|---------|
| `coconut` | `COCONUT` | Core batch normalization algorithm |
| `coconut_runner` | `COCONUTRunner` | CLI runner for COCONUT workflows |

---

## Object Structures

### ProcessedDataset (output of preprocessing)

```python
{
    "expr":   pd.DataFrame,   # probes × samples (float64)
    "keys":   List[str],      # gene symbol per probe (comma-sep for multi-gene)
    "class":  np.ndarray,     # 0=control, 1=disease
    "pheno":  pd.DataFrame,   # sample metadata (GSM IDs as index)
    "geo_id": str,            # e.g. "GSE11909_GPL96"
}
```

### Meta-analysis object (after `run_meta_analysis`)

```python
{
    "originalData": {
        "GSE11909_GPL96": { ... },  # ProcessedDataset dicts
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
R-style dict access for backward compatibility.

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
