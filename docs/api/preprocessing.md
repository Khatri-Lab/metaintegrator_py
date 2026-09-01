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

Single entry point for creating pre-configured loaders. This is the single
canonical place for wiring sources to handlers: all convenience functions
and thin wrappers delegate here.

```python
class DataLoaderFactory:
    @staticmethod
    def create_geo_loader(
        strategy: str = 'full',
        cache_dir: str | Path = '~/.geo_cache',
        log_transform: bool = False,
        normalize: bool = False,
        **kwargs,
    ) -> DataLoader
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | str | `'full'` | Loading strategy (see table below) |
| `cache_dir` | str / Path | `~/.geo_cache` | Cache directory for raw downloads |
| `log_transform` | bool | `False` | Apply log2 transformation. **`full` / `fast` only**: `hybrid`/`r_parity` always auto-detects instead and ignores this flag. |
| `normalize` | bool | `False` | Apply quantile normalization. **`full` / `fast` only.** |
| `**kwargs` | | | Passed through to the underlying source constructor (`full` / `fast` only) |

**Strategies**

| Strategy | Source | Handler | Best for |
|----------|--------|---------|----------|
| `hybrid` / `r_parity` | `HybridGeoSource` | `HybridMicroarrayHandler` | **Production**: matches R output |
| `full` | `GEOparseSource` | `MicroarrayHandler` | Full SOFT download (all samples) |
| `fast` | `NCBIGeoSource` | `MicroarrayHandler` | Quick exploration (Series Matrix only, may miss samples) |

``'r_parity'`` is accepted as an alias for ``'hybrid'``.

**Other constructors on `DataLoaderFactory`**

| Method | Status |
|--------|--------|
| `create_microarray_loader(source_type='geo_full', cache_dir=..., **kwargs)` | General wrapper around `create_geo_loader`. `source_type` aliases: `geo_hybrid`/`geo_r_parity` → `hybrid`, `geo_full`/`geo` → `full`, `geo_fast` → `fast`. |
| `create_rnaseq_loader(...)` | Placeholder: raises `NotImplementedError` |
| `create_single_cell_loader(...)` | Placeholder: raises `NotImplementedError` |

---

## DataLoader

Orchestration layer returned by the factory. Combines a `DataSource` with a `DataHandler`.

```python
class DataLoader:
    def load(self, dataset_id: str, **kwargs) -> List[ProcessedDataset]
    def load_multiple(
        self,
        dataset_ids: List[str],
        stop_on_error: bool = False,
        **kwargs,
    ) -> List[ProcessedDataset]
```

| Method | Returns | Description |
|--------|---------|-------------|
| `load(dataset_id, **kwargs)` | `list[ProcessedDataset]` | Load and process a single GEO series: one entry per platform found in the series |
| `load_multiple(dataset_ids, stop_on_error=False, **kwargs)` | `list[ProcessedDataset]` | Load multiple series. Failures are logged and skipped unless `stop_on_error=True` |

Both methods return a **list**, even for a single-platform series: `load()` does not collapse to a bare `ProcessedDataset`.

---

## ProcessedDataset

Dataclass output of all handlers. This is the standard format consumed by the
analysis layer.

```python
@dataclass
class ProcessedDataset:
    geo_id: str              # e.g. "GSE11909_GPL96"
    expr: pd.DataFrame       # probes × samples (float64)
    pheno: pd.DataFrame      # sample metadata, index aligned to expr.columns
    keys: List[str]          # gene symbol per probe, aligned to expr.index
    metadata: Dict[str, Any] # source, platform, handler, cache_dir, ...
```

`__post_init__` validates the object on construction and raises immediately
if either invariant is violated:

| Check | Raises |
|-------|--------|
| `expr.columns` equals `pheno.index` | `ValueError` |
| `keys` is a `list` and `len(keys) == len(expr)` | `TypeError` / `ValueError` |

`keys` follows R MetaIntegrator's convention: multi-gene probes are
comma-separated (e.g. `'CD6,IL2RG'`), unmappable probes are `'NA'`.

> **`ProcessedDataset` has no `.to_dict()` method.** To get the dict format
> `MetaAnalyzer` expects, build it from the fields directly (see
> [Complete Workflow Example](#complete-workflow-example)) or call the
> deprecated `load_geo_datasets()` convenience function, which already
> returns `list[dict]` in that shape.
>
> Class labels (case/control) are **not** part of `ProcessedDataset` —
> they're study-specific and must be assigned separately after loading,
> based on each dataset's `pheno` table.

---

## HybridMicroarrayHandler

The **production handler** for R-parity. `parse()` orchestrates:

1. Build the phenotype DataFrame from GEOparse per-sample metadata
2. Annotate probes → gene symbols via `GeneAnnotator`, validated against
   `EnsemblGeneValidator` (loaded lazily on first use, cached thereafter)
3. Auto-detect and apply log2 transformation (median expression > 20 ⇒ raw
   intensity)
4. Diagnostic check against known probe counts for common platforms
   (`GPL96`, `GPL97`, `GPL6244`, `GPL570`): a mismatch logs a warning, not
   an error
5. If annotation from the family SOFT file yields zero genes, fall back to
   a standalone GPL download and re-annotate

```python
class HybridMicroarrayHandler(DataHandler):
    def __init__(
        self,
        annotator: GeneAnnotator | None = None,
        validator: EnsemblGeneValidator | None = None,
    )
    def parse(self, raw: RawDataset) -> ProcessedDataset
    def normalize(self, dataset: ProcessedDataset) -> ProcessedDataset
```

Internally calls `GeneAnnotator.annotate()` with `validator=validator`
(Ensembl validation ON) for R-parity.

`normalize()` is a no-op: Series Matrix data from NCBI is already
normalized; the log2 auto-detection in `parse()` handles the one exception.

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

**Annotation strategies** (columns tried in priority order: first match wins):

| Priority | Column name(s) | Format |
|----------|-----------------|--------|
| 1 | `Gene Symbol`, `GENE_SYMBOL`, `Gene_Symbol`, `gene_symbol`, `genesymbol`, `Symbol`, `SYMBOL`, `symbol`, `HUGO_Gene_Nomenclature` | Direct: value already is the gene symbol |
| 2 | `gene_assignment` | Assignment: `accession // symbol // description`; symbol is extracted from field 2 of the first `///`-delimited entry only |
| 3 | `Gene`, `GENE`, `gene` | Direct |
| 4 | `ORF` | Direct: microbial platforms |

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
    def __init__(self, data_dir: Optional[Path] = None)
    def is_valid(self, gene_symbol: str) -> bool
    def validate_list(self, gene_symbols: list) -> Tuple[list, list]
    def get_stats(self) -> Dict[str, int]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_dir` | Path, optional | Directory containing reference data. Defaults to the package's bundled `preprocessing/data/`. |

`valid_genes` (`Set[str]`) is a public attribute populated at construction
time: it is **not** a method.

Data file, tried in order (first found wins): `ensembl_genes.csv.gz` →
`ensembl_genes.csv` → `ensembl_gene_symbols.txt`. Raises `FileNotFoundError`
if none of the three exist.

```python
from preprocessing.gene_validation import EnsemblGeneValidator

validator = EnsemblGeneValidator()
validator.is_valid('TP53')                          # True
validator.is_valid('ENST00000396160')                # False (transcript ID)
valid, invalid = validator.validate_list(['TP53', 'FAKE_GENE'])
```

**Convenience functions** (operate on the dict-based `originalData` format, not `ProcessedDataset`):

| Function | Purpose |
|----------|---------|
| `apply_ensembl_validation(dataset_dict, validator)` | Validate one dataset's `keys` in place: R-parity: only the first gene of each comma-separated key is checked; if it fails, the entire key is set to `'NA'` |
| `validate_all_datasets(data_dict, validator=None)` | Apply validation across every dataset in `meta_object['originalData']` |
| `create_validator()` | Shortcut for `EnsemblGeneValidator()` |

---

## GeneConverter

Translates between gene ID systems.

```python
class GeneConverter:
    def __init__(self, data_dir: Optional[Path] = None)
    def to_gene_symbol(self, identifier: str) -> Optional[str]
    def is_valid_symbol(self, symbol: str) -> bool
    def convert_list(self, identifiers: list) -> list
    def get_stats(self) -> Dict[str, int]
```

`to_gene_symbol()` is the single entry point; it tries, in order: already a
valid HGNC symbol → `ENSG*` via the Ensembl table → `NM_`/`NR_`/`XM_`/`XR_`
(version suffix stripped) via the RefSeq table → GenBank accession table →
numeric or string Entrez ID via the Entrez table. Returns `None` if nothing
matches.

Reference data in `preprocessing/data/`:

| File | Content |
|------|---------|
| `ensembl_genes.csv.gz` | ENSG → symbol |
| `refseq_genes.csv.gz` | NM_ → symbol |
| `genbank_genes.csv.gz` | Accession → symbol |
| `entrez_genes.csv.gz` | Entrez ID → symbol |

```python
from preprocessing.gene_converter import GeneConverter

converter = GeneConverter()
converter.to_gene_symbol('ENSG00000141510')            # 'TP53'
converter.to_gene_symbol('NM_000546')                   # 'TP53'
converter.is_valid_symbol('TP53')                        # True
converter.convert_list(['ENSG00000141510', 'FAKE'])       # ['TP53', 'NA']
```

---

## NCBINetwork

Low-level GEO network I/O with retry, timeout, and disk caching. This is
the single seam injected into `HybridGeoSource`, `GEOparseSource`, and
`NCBIFetcher`: mock it for offline testing.

```python
class NCBINetwork:
    def __init__(
        self,
        cache_dir: str | Path,
        ftp_host: str = 'ftp.ncbi.nlm.nih.gov',
        ftp_timeout: int = 30,
        http_timeout: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    )
    def list_ftp_directory(self, ftp_path: str, suffix: Optional[str] = None) -> List[str]
    def download_file(self, ftp_path: str, filename: str) -> Path
    def download_bytes(self, ftp_path: str, filename: str) -> bytes

    @staticmethod
    def geo_series_ftp_path(gse_id: str, subpath: str = 'matrix') -> str
```

**Download strategy:** `download_file()` is HTTPS-only, with the local
cache checked first (no network call on a hit). `download_bytes()` tries
HTTPS first and falls back to FTP on failure, validating gzip integrity on
either path.

**Retry policy:** up to `max_retries` attempts (default 3) with exponential
backoff (`retry_backoff ** attempt` seconds between attempts).

```python
from preprocessing.ncbi_network import NCBINetwork

net = NCBINetwork(cache_dir='~/.geo_cache')
ftp_path = NCBINetwork.geo_series_ftp_path('GSE11909', 'matrix')
files = net.list_ftp_directory(ftp_path, suffix='_series_matrix.txt.gz')
local_path = net.download_file(ftp_path, files[0])
```

---

## NCBIFetcher

Higher-level Series Matrix downloader behind the `'fast'` strategy
(`NCBIGeoSource`). Lists matrix files (with sizes, via raw `ftplib`) so it
can skip anything over 100 MB, downloads through `NCBINetwork`, parses into
per-platform DataFrames, and caches the result to disk with `pickle`.

```python
class NCBIFetcher:
    def __init__(self, cache_dir: str | Path = '~/.geo_cache')
    def fetch_series_matrix(self, gse_id: str) -> Dict[str, pd.DataFrame]
    def fetch_minimal_metadata(self, gse_id: str) -> dict
```

**Returns:** `fetch_series_matrix` returns `{platform_id: DataFrame}`.
`fetch_minimal_metadata` returns `{'title', 'summary', 'platforms'}` parsed
from the series' MINiML XML file.

---

## Complete Workflow Example

```python
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.analysis import MetaAnalyzer

# 1. Load datasets (R-parity mode)
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(['GSE11909', 'GSE50635', 'GSE39088'])

# 2. Convert to meta-object format.
# ProcessedDataset has no .to_dict(): build the dict from its fields.
meta_object = {"originalData": {}}
for ds in datasets:
    meta_object["originalData"][ds.geo_id] = {
        "geo_id": ds.geo_id,
        "expr": ds.expr,
        "pheno": ds.pheno,
        "keys": ds.keys,
        "metadata": ds.metadata,
    }

# 3. Set class labels (study-specific, not part of ProcessedDataset)
for name, ds in meta_object["originalData"].items():
    # ... set ds["class"] based on phenotype ...
    pass

# 4. Run meta-analysis
analyzer = MetaAnalyzer()
results = analyzer.run_meta_analysis(meta_object)
```

---

## Related

- [Architecture Overview](../architecture.md): Factory/Handler/Source diagrams
- [Gene Annotation Tutorial](../tutorials/gene_annotation.md): Step-by-step walkthrough
- [Analysis API](analysis.md): What happens after preprocessing
