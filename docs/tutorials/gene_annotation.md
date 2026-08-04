# Gene Annotation and Symbol Mapping

Learn how probe IDs get converted to standardized gene symbols, how to validate annotation quality, and how to run annotation manually for debugging. Gene annotation is crucial because meta-analysis requires matching genes across different microarray platforms.

## Goal

Understand how platform-specific probe IDs become standardized HGNC gene
symbols, check annotation quality, and ensure datasets use consistent gene
identifiers for meta-analysis.

## Prerequisites

- Python 3.10+
- metaIntegrator-py git repository cloned
- Completed [Data Acquisition Tutorial](data_acquisition.md) or have downloaded GEO datasets
- Basic understanding of gene nomenclature (helpful but not required)

### Why Gene Annotation Matters

Different microarray platforms use different probe identifiers:
- **Affymetrix**: `1007_s_at`, `1053_at`, etc.
- **Illumina**: `ILMN_1343291`, `ILMN_1651838`, etc.
- **Agilent**: Probe IDs or gene symbols

**Meta-analysis requires** matching the same genes across studies, which means converting all identifiers to a common vocabulary: **gene symbols** (e.g., BRCA1, TP53, EGFR).

## Steps

### Step 1: Import Annotation Tools

```python
from preprocessing.gene_annotator import GeneAnnotator
from preprocessing.gene_validation import EnsemblGeneValidator
from preprocessing.gene_converter import GeneConverter

print("✅ Annotation tools imported successfully")
```

!!! tip "Automatic Annotation"
    In production, gene annotation is called automatically by
    `HybridMicroarrayHandler` during `DataLoaderFactory.create_geo_loader('r_parity').load_multiple(...)`.
    By the time you have a `ProcessedDataset`, `keys` is already gene
    symbols — not probe IDs. This tutorial shows the manual steps for
    learning and debugging.

### Step 2: Annotate Probes Directly

The `GeneAnnotator` maps probe IDs to gene symbols using GPL annotation tables:

```python
annotator = GeneAnnotator()
validator = EnsemblGeneValidator()

# gpl_table: DataFrame from GEOparse's GPL object (platform annotation table)
# probe_ids: Index of expression matrix rows
keys = annotator.annotate(gpl_table, probe_ids, validator=validator)

print(f"Mapped {sum(k != 'NA' for k in keys)} of {len(keys)} probes")
```

**What happens internally:**

1. `annotate()` locates the gene-symbol column in the GPL table
   (tries `Gene Symbol`, `Symbol`, `gene_assignment`, etc.)
2. For direct columns, it replaces `///` and `;` with commas (multi-gene probes become comma-separated)
3. For assignment columns, it parses `accession // symbol // description` format
4. **Ensembl validation** (`EnsemblGeneValidator`) rejects non-HGNC symbols (LOC*, ORFs, mitochondrial aliases) — probes get `'NA'` instead
5. **GenBank filter** (`GeneAnnotator.GENBANK_ACCESSION_RE`) rejects accession IDs (AF*, AK*, BC*, etc.) that some platforms include alongside gene symbols

Both filters together produce gene counts matching R within ±5 per platform.
See the [Architecture Overview](../architecture.md#gene-annotation-pipeline) for
the full annotation pipeline diagram.

### Step 3: Inspect an Already-Annotated Dataset

Since annotation runs automatically inside the loader, the normal way to
"see" it is to inspect `keys` after `load_multiple()`:

```python
from preprocessing.loader_factory import DataLoaderFactory

loader = DataLoaderFactory.create_geo_loader(strategy="r_parity", cache_dir="./geo_cache")
datasets = loader.load_multiple(["GSE5847"])
ds = datasets[0]

print("📊 Current Gene Identifiers:")
print(f"   Total probes: {len(ds.keys):,}")
print(f"   First 10 keys: {ds.keys[:10]}")

n_mapped = sum(1 for k in ds.keys if k != 'NA')
print(f"\n📈 Mapping rate: {n_mapped:,}/{len(ds.keys):,} ({n_mapped / len(ds.keys):.1%})")
```

**Expected output:**
```
📊 Current Gene Identifiers:
   Total probes: 22,283
   First 10 keys: ['NA', 'NA', 'DDR1', 'RFC2', 'HSPA6', 'PAX8', 'GUCA1A', ...]

📈 Mapping rate: 13,042/22,283 (58.5%)
```

**Note:** There is no separate `annotation_report` dict attached to a
dataset — `'NA'` entries in `keys` *are* the report. Compute your own
summary statistics from `keys` as shown above.

### Step 4: Handle Multiple Probe-to-Gene Mappings

Some probes map to the same gene — this is normal and handled automatically
downstream (`SampleLevelGeneDataProcessor` groups by gene and averages):

```python
from collections import Counter

gene_counts = Counter(k for k in ds.keys if k != 'NA')
duplicated_genes = {gene: count for gene, count in gene_counts.items() if count > 1}

if duplicated_genes:
    print(f"Found {len(duplicated_genes)} genes with multiple probes:")
    top_duplicated = sorted(duplicated_genes.items(), key=lambda x: x[1], reverse=True)[:5]
    for gene, count in top_duplicated:
        print(f"   • {gene}: {count} probes")
    print("\n💡 This is normal — meta-analysis summarizes probe-level data per gene.")
else:
    print("✅ No duplicate gene mappings")
```

**Expected output:**
```
Found 234 genes with multiple probes:
   • BRCA1: 8 probes
   • TP53: 6 probes
   • EGFR: 5 probes
   • MYC: 4 probes
   • PTEN: 4 probes

💡 This is normal — meta-analysis summarizes probe-level data per gene.
```

### Step 5: Convert Non-Symbol Gene IDs with GeneConverter

If you're working with a dataset where the identifiers are Ensembl, RefSeq,
GenBank, or Entrez IDs rather than raw probe IDs, `GeneConverter` maps any of
those to a HUGO gene symbol:

```python
converter = GeneConverter()

ensembl_ids = ["ENSG00000141510", "ENSG00000012048"]  # TP53, BRCA1
symbols = [converter.convert(gid) for gid in ensembl_ids]
print(symbols)  # ['TP53', 'BRCA1']
```

See the [Preprocessing API Reference](../api/preprocessing.md) for the full
`GeneConverter` signature and supported ID types.

### Step 6: Check Gene Overlap Across Datasets

For meta-analysis, you need adequate gene-symbol overlap between studies:

```python
datasets = loader.load_multiple(["GSE5847", "GSE28750"])

all_gene_sets = [set(k for k in ds.keys if k != 'NA') for ds in datasets]
common_genes = set.intersection(*all_gene_sets)

print(f"📊 Cross-Dataset Gene Overlap:")
for ds, genes in zip(datasets, all_gene_sets):
    print(f"   {ds.geo_id}: {len(genes):,} mapped genes")
print(f"   Common genes: {len(common_genes):,}")
print(f"   Overlap: {len(common_genes) / min(len(g) for g in all_gene_sets):.1%}")
```

**Expected output:**
```
📊 Cross-Dataset Gene Overlap:
   GSE5847_GPL96: 13,042 mapped genes
   GSE28750_GPL570: 20,436 mapped genes
   Common genes: 11,820
   Overlap: 90.6%
```

**Good overlap** (>80%) indicates datasets are suitable for meta-analysis.
**Poor overlap** (<50%) suggests platform incompatibility or an annotation problem.

## Tips & Gotchas

### Tip 1: Monitor Annotation Quality

Always check the mapping rate before meta-analysis:

```python
def check_annotation_quality(ds, min_rate=0.5):
    """Check if a ProcessedDataset's mapping rate is acceptable."""
    n_mapped = sum(1 for k in ds.keys if k != 'NA')
    rate = n_mapped / len(ds.keys)

    if rate >= min_rate:
        print(f"✅ {ds.geo_id}: mapping rate {rate:.1%}")
        return True
    else:
        print(f"❌ {ds.geo_id}: mapping rate {rate:.1%} (< {min_rate:.0%})")
        return False

for ds in datasets:
    check_annotation_quality(ds)
```

A mapping rate well below 50% is unusually low for a standard Affymetrix/
Illumina platform — a good sign the GPL table wasn't matched correctly.

### Tip 2: Control Probes Never Map

Control probes (`AFFX-*`) are expected to come through as `'NA'` — they
aren't genes, so this isn't a bug:

```python
control_probes = [k for k, key in zip(ds.expr.index, ds.keys)
                   if key == 'NA' and str(k).startswith('AFFX-')]
print(f"Control probes (expected 'NA'): {len(control_probes)}")
```

### Gotcha: Gene Symbol Case Matters

Gene symbols should be uppercase:

```python
# ✅ Correct
genes = ['BRCA1', 'TP53', 'EGFR']

# ❌ Wrong - will not match across datasets
genes = ['brca1', 'Tp53', 'egfr']

# Fix case
genes_fixed = [g.upper() for g in genes]
```

## Troubleshooting

### Q: Low mapping rate

**A**: Several possible causes:

1. **Old or custom platform**: the GPL table's gene-symbol column wasn't
   recognized by `find_symbol_column()` — inspect `gpl_table.columns` directly.
2. **Non-standard probe IDs**: inspect `ds.keys[:20]` and `ds.expr.index[:20]`
   to see what's actually there.
3. **Organism mismatch**: verify the organism in phenotype data —
   ```python
   if 'organism_ch1' in ds.pheno.columns:
       print(ds.pheno['organism_ch1'].unique())
   ```

### Q: Different studies have no overlapping genes

**A**: Check these common issues:

```python
# 1. Confirm annotation actually ran (keys shouldn't look like probe IDs)
for ds in datasets:
    sample_keys = ds.keys[:5]
    has_probes = any('_at' in str(k) or 'ILMN_' in str(k) for k in sample_keys)
    print(f"{'❌' if has_probes else '✅'} {ds.geo_id}: {sample_keys}")

# 2. Check platform compatibility
for ds in datasets:
    print(f"{ds.geo_id}: {ds.metadata.get('platform', 'Unknown')}")

# 3. Check gene overlap
all_genes = [set(k for k in ds.keys if k != 'NA') for ds in datasets]
common = set.intersection(*all_genes)
print(f"Common genes: {len(common)}")
if len(common) < 1000:
    print("⚠️ Very few common genes - check annotation quality")
```

## Next Steps

Now that you understand how gene symbols get standardized:

1. **[Quality Control Tutorial](quality_control.md)** – Verify gene overlap and data quality
2. **[Meta-Analysis Tutorial](meta_analysis.md)** – Perform statistical analysis
3. **[Complete Workflow Tutorial](complete_workflow.md)** – See annotation in full context

### Advanced Annotation Topics

- **Custom annotation databases** – Extend `GeneAnnotator` for your own probe-to-gene mappings
- **Cross-species annotation** – Handle orthologs for comparative studies
- **Ensembl ID mapping** – Use `GeneConverter` for datasets keyed by Ensembl/RefSeq/GenBank/Entrez IDs instead of symbols

### Key Takeaways

✅ Annotation happens automatically inside `HybridMicroarrayHandler` during `load_multiple()` — no separate call needed for the standard workflow
✅ Unmapped probes come through as `'NA'`, not a separate report object
✅ `GeneAnnotator.annotate()` is the manual entry point if you need direct control
✅ `EnsemblGeneValidator` filters non-HGNC symbols; the GenBank-accession regex filters accession IDs
✅ `GeneConverter` handles non-probe identifier types (Ensembl/RefSeq/GenBank/Entrez → symbol)
✅ Multiple probes per gene is normal and handled automatically downstream
✅ Verify gene overlap across datasets before meta-analysis

---

**Questions?** Check the [API Reference](../api/preprocessing.md) or see [Best Practices](../user_guide/best_practices.md).
