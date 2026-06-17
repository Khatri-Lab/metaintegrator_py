# Preprocessing API

The preprocessing package uses a **Factory → Source → Handler** architecture
(see [Architecture](../architecture.md)) to load and annotate GEO microarray
data. Users interact through `DataLoaderFactory`; internal wiring is automatic.

## Quick Start

```python
from preprocessing.loader_factory import DataLoaderFactory

# One line to get R-parity datasets
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(['GSE11909', 'GSE50635'])

for ds in datasets:
    print(f"{ds.geo_id}: {ds.expr.shape[0]} probes, "
          f"{ds.expr.shape[1]} samples, {len(set(ds.keys))} genes")
```

---

## DataLoaderFactory

Single entry point for creating pre-configured loaders.

```python
class DataLoaderFactory:
    @staticmethod
    def create_geo_loader(
        strategy: str = 'full',
        cache_dir: str | Path = '~/.geo_cache',
    ) -> DataLoader
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | str | `'full'` | Loading strategy (see table below) |
| `cache_dir` | str / Path | `~/.geo_cache` | Cache directory for raw downloads |

**Strategies**

| Strategy | Source | Handler | Best for |
|----------|--------|---------|----------|
| `hybrid` / `r_parity` | `HybridGeoSource` | `HybridMicroarrayHandler` | **Production** — matches R output |
| `full` | `GEOparseSource` | `MicroarrayHandler` | Full SOFT download (all metadata) |
| `fast` | `NCBIGeoSource` | `MicroarrayHandler` | Quick exploration |

---

## DataLoader

Orchestration layer returned by the factory. Combines a `DataSource` with a `DataHandler`.

```python
class DataLoader:
    def load(self, geo_id: str) -> ProcessedDataset
    def load_multiple(self, geo_ids: List[str]) -> List[ProcessedDataset]
```

| Method | Returns | Description |
|--------|---------|-------------|
| `load(geo_id)` | `ProcessedDataset` | Load and process a single GEO series |
| `load_multiple(geo_ids)` | `list[ProcessedDataset]` | Load multiple series (handles multi-platform) |

---

## ProcessedDataset

Dataclass output of all handlers. This is the standard format consumed by the
analysis layer.

```python
@dataclass
class ProcessedDataset:
    geo_id: str                         # e.g. "GSE11909_GPL96"
    expr: pd.DataFrame                  # probes × samples (float64)
    keys: List[str]                     # gene symbol per probe
    pheno: pd.DataFrame                 # sample metadata
    class_vector: Optional[np.ndarray]  # 0=control, 1=disease
    platform_id: Optional[str]          # e.g. "GPL96"
    annotation_method: Optional[str]    # how genes were mapped
```

Convert to the dict format used by `MetaAnalyzer`:

```python
ds_dict = dataset.to_dict()  # → {"expr": ..., "keys": ..., "class": ..., ...}
```

---

## HybridMicroarrayHandler

The **production handler** for R-parity. Orchestrates:

1. Expression extraction from Series Matrix
2. Phenotype parsing
3. Gene annotation via `GeneAnnotator` + `EnsemblValidator`
4. Automatic log2 transformation
5. Column name cleaning

```python
class HybridMicroarrayHandler(DataHandler):
    def parse(self, raw: RawDataset) -> ProcessedDataset
```

Internally calls `GeneAnnotator.annotate()` with `validator=validator`
(Ensembl validation ON) for R-parity.

---

## GeneAnnotator

Maps probe IDs to gene symbols using GPL annotation tables.

```python
class GeneAnnotator:
    def annotate(
        self,
        gpl_table: pd.DataFrame,
        probe_ids: pd.Index,
        validator: Optional[GeneValidator] = None,
    ) -> List[str]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `gpl_table` | DataFrame | GPL platform annotation table |
| `probe_ids` | Index | Probe IDs to annotate |
| `validator` | GeneValidator | Optional Ensembl validator (recommended for R-parity) |

**Returns:** List of gene symbols (one per probe). `'NA'` for unmappable probes.

**Annotation strategies** (tried in priority order):

1. **Direct symbol columns** — `Gene Symbol`, `Symbol`, `HUGO_Gene_Nomenclature`
2. **Assignment columns** — `gene_assignment` (parsed from `accession // symbol // desc` format)
3. **ORF columns** — for microbial platforms

**R-parity filters** (applied after symbol extraction):

| Filter | What it catches | Impact |
|--------|----------------|--------|
| **Ensembl validation** | LOC*, mitochondrial aliases, non-HGNC symbols | ~500–3000 per platform |
| **GenBank accession** | AF*, AK*, BC*, etc. accession IDs | ~3400 entries on GPL570 |

```python
from preprocessing.gene_annotator import GeneAnnotator
from preprocessing.gene_validation import EnsemblGeneValidator

annotator = GeneAnnotator()
validator = EnsemblGeneValidator()

keys = annotator.annotate(gpl_table, probe_ids, validator=validator)
print(f"Mapped {sum(k != 'NA' for k in keys)} of {len(keys)} probes")
```

---

## EnsemblGeneValidator

Validates gene symbols against a curated Ensembl reference set (~20,000 HGNC symbols).

```python
class EnsemblGeneValidator:
    def __init__(self, gene_list_path: Optional[str] = None)
    def is_valid(self, symbol: str) -> bool
    def valid_genes(self) -> Set[str]
```

Data file: `preprocessing/data/ensembl_gene_symbols.txt`

---

## GeneConverter

Translates between gene ID systems.

```python
class GeneConverter:
    def ensembl_to_symbol(self, ensembl_ids: List[str]) -> Dict[str, str]
    def refseq_to_symbol(self, refseq_ids: List[str]) -> Dict[str, str]
    def genbank_to_symbol(self, genbank_ids: List[str]) -> Dict[str, str]
    def entrez_to_symbol(self, entrez_ids: List[str]) -> Dict[str, str]
```

Reference data in `preprocessing/data/`:

| File | Content |
|------|---------|
| `ensembl_genes.csv.gz` | ENSG → symbol |
| `refseq_genes.csv.gz` | NM_ → symbol |
| `genbank_genes.csv.gz` | Accession → symbol |
| `entrez_genes.csv.gz` | Entrez ID → symbol |

---

## NCBIGEOFetcher

Low-level GEO download with caching, rate-limiting, and multi-protocol fallback.

```python
class NCBIGEOFetcher:
    def __init__(self, cache_dir: Optional[str] = None)
    def fetch_geo_series_matrix(self, gse_id: str) -> Dict[str, Dict]
```

**Download fallback order:** FTP → HTTPS → SOFT file

**Rate limiting:** Automatic 3 requests/sec to comply with NCBI policies.

**Caching:** Downloaded files cached to `cache_dir` for instant re-use.

---

## Complete Workflow Example

```python
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.analysis import MetaAnalyzer

# 1. Load datasets (R-parity mode)
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(['GSE11909', 'GSE50635', 'GSE39088'])

# 2. Convert to meta-object format
meta_object = {"originalData": {}}
for ds in datasets:
    meta_object["originalData"][ds.geo_id] = ds.to_dict()

# 3. Set class labels (study-specific)
for name, ds in meta_object["originalData"].items():
    # ... set ds["class"] based on phenotype ...
    pass

# 4. Run meta-analysis
analyzer = MetaAnalyzer()
results = analyzer.run_meta_analysis(meta_object)
```

---

## Related

- [Architecture Overview](../architecture.md) — Factory/Handler/Source diagrams
- [Gene Annotation Tutorial](../tutorials/gene_annotation.md) — Step-by-step walkthrough
- [Analysis API](analysis.md) — What happens after preprocessing
