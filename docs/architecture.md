# Architecture Overview

MetaIntegrator-py follows a layered, object-oriented architecture designed around the
**Open/Closed Principle**: new data sources, handlers, and analyses can be added without
modifying existing code.

## High-Level System Architecture

```mermaid
graph TB
    subgraph "Data Acquisition Layer"
        Factory["DataLoaderFactory"]
        Sources["DataSource<br/><i>GEO, Hybrid, R-parity</i>"]
        Handlers["DataHandler<br/><i>Microarray, Hybrid</i>"]
    end

    subgraph "Preprocessing Layer"
        Annotator["GeneAnnotator<br/><i>probe → gene symbol</i>"]
        Validator["EnsemblGeneValidator<br/><i>Ensembl filtering</i>"]
        Converter["GeneConverter<br/><i>ID translation</i>"]
    end

    subgraph "Batch Correction (optional)"
        Coconut["coconut()<br/><i>COCONUT batch normalization</i>"]
    end

    subgraph "Analysis Layer"
        Meta["MetaAnalyzer<br/><i>run_meta_analysis</i>"]
        Core["MetaAnalysisCore<br/><i>effect sizes, pooling</i>"]
        Fisher["FisherCombination<br/><i>directional p-values</i>"]
        Filter["MetaIntegratorFilter<br/><i>gene filtering</i>"]
        GeneSearch["GeneSearchOptimizer<br/><i>forward/backward search</i>"]
    end

    subgraph "Scoring Layer"
        Score["ScoreCalculator<br/><i>geometric mean z-scores</i>"]
        SampleGene["SampleLevelGeneDataProcessor<br/><i>probe → gene matrix</i>"]
        ROC["ROC / PRC Engines"]
    end

    subgraph "Visualization Layer"
        Plotting["plotting.MetaAnalysisPlotter<br/><i>volcano, forest, heatmap</i>"]
        Enhanced["EnhancedMetaAnalysisPlotter<br/><i>forest, heatmap, ROC/PRC</i>"]
        Violin["violin_plot.py<br/><i>R-parity ggplot2 style</i>"]
    end

    Factory --> Sources
    Factory --> Handlers
    Sources --> Handlers
    Handlers --> Annotator
    Annotator --> Validator
    Annotator --> Converter
    Handlers -->|ProcessedDataset| Meta
    Handlers -.->|optional, per-study CSVs| Coconut
    Coconut -.->|combine_coco_output → meta_input| Meta
    Meta --> Core
    Meta --> Fisher
    Core -->|pooledResults| Filter
    Filter -->|optional refinement| GeneSearch
    Filter --> Score
    GeneSearch -.-> Score
    Score --> SampleGene
    Score --> ROC
    ROC --> Enhanced
    Filter --> Violin

    style Coconut fill:#e0f7fa
    style GeneSearch fill:#f3e5f5
```

## Module Map

| Package | Module | Responsibility |
|---------|--------|---------------|
| **preprocessing** | `base.py` | Abstract base classes: `DataSource`, `DataHandler`, `DataLoader`, `ProcessedDataset` |
| | `loader_factory.py` | `DataLoaderFactory` — single entry point for creating loaders |
| | `geo_sources.py` | `NCBIGeoSource` — Series Matrix fetching; `GEOparseSource` — full SOFT-based fetching (the `'full'` strategy) |
| | `hybrid_source.py` | `HybridGeoSource` — Series Matrix expr + GEOparse annotation |
| | `hybrid_handler.py` | `HybridMicroarrayHandler` — orchestrates annotation, log2, pheno |
| | `microarray_handler.py` | `MicroarrayHandler` — full SOFT-based processing |
| | `gene_annotator.py` | `GeneAnnotator` — probe-to-gene mapping with GenBank/Ensembl filters |
| | `gene_validation.py` | `EnsemblGeneValidator` — HGNC symbol validation |
| | `gene_converter.py` | `GeneConverter` — Ensembl ↔ Entrez ↔ RefSeq ↔ GenBank translation |
| | `ncbi_fetcher.py` | `NCBIFetcher` — Series Matrix downloader behind the `'fast'` strategy, with size guard and pickle caching |
| | `ncbi_network.py` | `NCBINetwork` — low-level FTP/HTTPS transport with retry & disk caching, the injectable seam used by every source |
| | `r_parity_handler.py`, `r_parity_source.py` | **Deprecated aliases** — re-export `HybridMicroarrayHandler`/`HybridGeoSource` under their old names; not separate implementations |
| | `r_parity_loader.py` | `load_geo_datasets_r_parity()` — backward-compatible function wrapper around `DataLoaderFactory(strategy='r_parity')`, kept for older notebooks/tests |
| **metaintegrator** | `analysis.py` | `MetaAnalyzer` — top-level orchestrator for meta-analysis |
| | `core_analysis.py` | `MetaAnalysisCore`, `FisherCombination` — statistical engine |
| | `score_calculation.py` | `ScoreCalculator` — geometric-mean z-score computation |
| | `sample_level_gene_data.py` | `SampleLevelGeneDataProcessor` — probe-level to gene-level extraction |
| | `scoring_meta_integrator_filter.py` | `MetaIntegratorFilter` — gene filtering by FDR, effect size, leave-one-out |
| | `gene_search_optimizer.py` | `GeneSearchOptimizer` — AUC-driven `forward_search()` / `backward_search()` signature refinement |
| | `scoring_meta_integrator_orchestrator.py` | End-to-end scoring orchestration |
| | `roc_analysis_engine.py` | ROC curve computation and multi-dataset ROC plots |
| | `prc_analysis.py` | Precision-recall curve analysis |
| | `power_calculation.py` | Statistical power estimation |
| | `phenotype_processing.py` | Phenotype/class label extraction |
| | `validation.py` | Dataset structure validation |
| **visualization** | `plotting.py` | `MetaAnalysisPlotter` — volcano, forest, heatmap, effect-size distribution |
| | `enhanced_plotting.py` | `EnhancedMetaAnalysisPlotter` — forest/heatmap/Manhattan/ROC/PRC/violin/regression, plus module-level convenience functions. Previously also named `MetaAnalysisPlotter`; renamed to stop colliding with `plotting.MetaAnalysisPlotter` (an unrelated class). A deprecated, warning-emitting alias remains for old imports — see [Visualization API](api/visualization.md). |
| | `violin_plot.py` | R-parity violin plots (ggplot2 style) |
| **coconut** | `coconut.py` | COCONUT batch normalization |
| | `coconut_runner.py` | CLI runner for COCONUT workflows |

---

## Data Acquisition: Factory Pattern

The preprocessing layer uses a **Factory + Strategy** pattern. Users never instantiate
sources or handlers directly — `DataLoaderFactory` wires everything together.

```mermaid
classDiagram
    class DataLoaderFactory {
        +create_geo_loader(strategy) DataLoader
    }

    class DataSource {
        <<abstract>>
        +fetch(dataset_id) List~RawDataset~
    }

    class DataHandler {
        <<abstract>>
        +parse(raw_dataset) ProcessedDataset
    }

    class DataLoader {
        -source: DataSource
        -handler: DataHandler
        +load(dataset_id) List~ProcessedDataset~
        +load_multiple(dataset_ids, stop_on_error) List~ProcessedDataset~
    }

    class HybridGeoSource {
        +fetch(dataset_id) List~RawDataset~
    }

    class HybridMicroarrayHandler {
        -annotator: GeneAnnotator
        +parse(raw_dataset) ProcessedDataset
    }

    DataLoaderFactory ..> DataLoader : creates
    DataLoader o-- DataSource
    DataLoader o-- DataHandler
    DataSource <|-- HybridGeoSource
    DataHandler <|-- HybridMicroarrayHandler
```

**Strategies available:**

| Strategy | Source | Handler | Use case |
|----------|--------|---------|----------|
| `hybrid` / `r_parity` | `HybridGeoSource` | `HybridMicroarrayHandler` | Production — matches R output |
| `full` | `GEOparseSource` | `MicroarrayHandler` | Full SOFT download |
| `fast` | `NCBIGeoSource` | `MicroarrayHandler` | Quick exploration |

```python
from preprocessing.loader_factory import DataLoaderFactory

# Recommended: R-parity mode
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(['GSE11909', 'GSE50635'])
```

---

## Gene Annotation Pipeline

The annotation pipeline converts probe IDs to HGNC gene symbols with two levels
of filtering for R-parity:

```mermaid
flowchart LR
    Probes["Probe IDs<br/><i>~54,000</i>"] --> Annotator["GeneAnnotator<br/>.annotate()"]

    Annotator --> Strategy{"Annotation<br/>Strategy"}
    Strategy -->|"Gene Symbol col"| Direct["Direct mapping"]
    Strategy -->|"// format"| Parse["Parse delimited"]
    Strategy -->|"RefSeq/GenBank"| Convert["GeneConverter"]

    Direct --> Validate{"EnsemblGeneValidator<br/>(Ensembl)"}
    Parse --> Validate
    Convert --> Validate

    Validate -->|"Pass"| GenBank{"GenBank<br/>Accession<br/>Filter"}
    Validate -->|"Fail → NA"| Dropped1["Filtered out"]

    GenBank -->|"Pass"| Keys["Gene Keys<br/><i>~13,000</i>"]
    GenBank -->|"Accession → NA"| Dropped2["Filtered out"]

    style Validate fill:#e8f5e9
    style GenBank fill:#fff3e0
    style Dropped1 fill:#ffebee
    style Dropped2 fill:#ffebee
```

**Key design decisions for R-parity:**

1. **Ensembl validation** (`EnsemblGeneValidator`) — rejects non-HGNC symbols like LOC*, ORFs,
   and mitochondrial gene aliases. Controlled via `validator` parameter in `HybridMicroarrayHandler`.
2. **GenBank accession filter** (`GeneAnnotator.GENBANK_ACCESSION_RE`) — rejects accession IDs
   (AF*, AK*, BC*, etc.) that some platforms include alongside gene symbols.
3. Both filters together produce gene counts matching R within ±5 per platform.

---

## Scoring Pipeline

Score calculation converts a gene signature (from `filterGenes`) into per-sample
meta-scores for classification:

```mermaid
flowchart TB
    Filter["filterObject<br/><i>67 pos + 28 neg genes</i>"]
    Dataset["datasetObject<br/><i>expr, keys, class</i>"]

    Filter --> Score["ScoreCalculator.calculate_score()"]
    Dataset --> Score

    Score --> Shift["Shift expr to positive<br/><i>expr + |min| + 1</i>"]
    Shift --> PosGenes["get_sample_level_gene_data<br/><i>ALL 67 pos genes</i>"]
    Shift --> NegGenes["get_sample_level_gene_data<br/><i>ALL 28 neg genes</i>"]

    PosGenes --> Expand["Expand comma-separated keys"]
    NegGenes --> Expand
    Expand --> Group["Group by gene, mean across probes"]
    Group --> Fill["Fill missing genes with 1<br/><i>log(1) = 0</i>"]

    Fill --> GeoMean["Geometric mean per sample<br/><i>exp(mean(log(values)))</i>"]
    GeoMean --> ZScore["Z-score standardization<br/><i>(score - mean) / std</i>"]

    ZScore --> AUC["ROC AUC evaluation"]

    style Fill fill:#fff3e0
```

**Critical R-parity detail:** All gene names are passed to `get_sample_level_gene_data`,
including those not present in the dataset. Missing genes are filled with value `1`
(so `log(1) = 0` enters the geometric mean), matching R's `calculateScore` behavior.
This was previously a pre-filter bug that caused AUC regression.

---

## Quick Start for New Developers

### Running the SLE tutorial end-to-end

```bash
cd tutorials/
jupyter notebook jupyter_metaintegrator_sle_tutorial.ipynb
```

### Understanding the call chain

1. **Load data**: `DataLoaderFactory.create_geo_loader('r_parity')` → `HybridMicroarrayHandler.parse()` → `GeneAnnotator.annotate()`
2. **Meta-analysis**: `MetaAnalyzer.run_meta_analysis()` → `MetaAnalysisCore.calculate_effect_sizes()` → `pool_inverse_variance()`
3. **Filter genes**: `MetaAnalyzer.filter_genes()` (in `analysis.py`) → returns `filter_object` with `posGeneNames` / `negGeneNames`.
   A separate, strategy-based pipeline — `scoring_meta_integrator_filter.MetaIntegratorFilter` — also exists (standard / adaptive / combined-criteria); the two are not interchangeable, see [API Reference](api/overview.md#package-metaintegrator).
4. **Refine signature (optional)**: `GeneSearchOptimizer.forward_search()` / `.backward_search()` (in `gene_search_optimizer.py`) iteratively add/remove genes from `filter_object` based on their contribution to weighted AUC. See [User Guide: Gene Filtering](user_guide/filtering.md#metaintegrator_pys-built-in-gene-filtering).
5. **Score & evaluate**: `ScoreCalculator.calculate_score()` → `SampleLevelGeneDataProcessor` → ROC/AUC

### Key data structures

```python
# ProcessedDataset (from preprocessing)
dataset = {
    "expr":  pd.DataFrame,   # probes × samples
    "keys":  List[str],      # gene symbol per probe (comma-separated for multi-gene)
    "class": np.ndarray,     # 0=control, 1=disease
    "pheno": pd.DataFrame,   # sample metadata
    "geo_id": str,           # e.g. "GSE11909_GPL96"
}

# filter_object (from filterGenes)
filter_object = {
    "posGeneNames": List[str],  # up-regulated signature genes
    "negGeneNames": List[str],  # down-regulated signature genes
    "FDRThresh": float,
    "effectSizeThresh": float,
}
```
