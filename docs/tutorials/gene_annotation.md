# Gene Annotation and Symbol Mapping

Learn how to convert probe IDs to standardized gene symbols, validate annotation quality, and prepare datasets for meta-analysis. Gene annotation is crucial because meta-analysis requires matching genes across different microarray platforms.

## Goal

Convert platform-specific probe IDs to standardized HGNC gene symbols, validate annotation quality, and ensure datasets use consistent gene identifiers for meta-analysis.

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
```

!!! tip "Automatic Annotation"
    In production, gene annotation is called automatically by the
    `HybridMicroarrayHandler` during `DataLoaderFactory.create_geo_loader('r_parity')`.
    This tutorial shows the manual steps for learning and debugging.

### Step 2: Annotate Probes

The `GeneAnnotator` maps probe IDs to gene symbols using GPL annotation tables:

```python
annotator = GeneAnnotator()
validator = EnsemblGeneValidator()

# gpl_table: DataFrame from GEOparse GPL object
# probe_ids: Index of expression matrix rows
keys = annotator.annotate(gpl_table, probe_ids, validator=validator)

print(f"Mapped {sum(k != 'NA' for k in keys)} of {len(keys)} probes")
```

**What happens internally:**

1. The annotator locates the gene-symbol column in the GPL table
   (tries `Gene Symbol`, `Symbol`, `gene_assignment`, etc.)
2. For direct columns, it replaces `///` and `;` with commas
3. For assignment columns, it parses `accession // symbol // description` format
4. **Ensembl validation** rejects non-HGNC symbols (LOC*, ORFs, mitochondrial aliases)
5. **GenBank filter** rejects accession IDs (AF*, AK*, BC*, etc.)

Both filters together produce gene counts matching R within ±5 per platform.
See the [Architecture Overview](../architecture.md#gene-annotation-pipeline) for
the full annotation pipeline diagram.
    GeneAnnotationEngine,         # Core annotation engine
    annotate_geo_dataset,         # Annotate a single dataset
    validate_geo_gene_symbols,    # Validate gene symbol quality
    GEOPlatformAnnotator         # Platform-specific annotation
)

from preprocessing.geo_data_loader import get_geo_data
import pandas as pd

print("✅ Annotation tools imported successfully")
```

### Step 2: Load Data and Inspect Gene Identifiers

First, let's see what identifiers we're working with:

```python
# Download a dataset (uses cache if available)
meta_object = get_geo_data(
    gse_vector="GSE5847",
    cache_dir="./geo_cache"
)

dataset = meta_object['originalData']['GSE5847']

# Examine current gene identifiers
print("📊 Current Gene Identifiers:")
print(f"   Total genes: {len(dataset['keys']):,}")
print(f"   First 10 keys: {dataset['keys'][:10]}")
print(f"   Key type comment: {dataset.get('key_comment', 'Not available')}")

# Check if these are probe IDs or gene symbols
sample_keys = dataset['keys'][:5]
are_probe_ids = any(
    '_at' in str(k) or 'ILMN_' in str(k) or 'AFFX' in str(k) 
    for k in sample_keys
)

if are_probe_ids:
    print("\n⚠️ Dataset uses probe IDs - annotation needed!")
else:
    print("\n✅ Dataset appears to use gene symbols")
```

**Expected output:**
```
📊 Current Gene Identifiers:
   Total genes: 54,675
   First 10 keys: ['1007_s_at', '1053_at', '117_at', '121_at', ...]
   Key type comment: Annotation absent

⚠️ Dataset uses probe IDs - annotation needed!
```

### Step 3: Initialize the Annotation Engine

The annotation engine handles mapping between different identifier types:

```python
print("🔧 Initializing Gene Annotation Engine...")

# Initialize with local annotation databases
annotation_engine = GeneAnnotationEngine(
    enable_remote=False  # Use local files (faster, offline-capable)
)

print("✅ Annotation engine ready")
print(f"   Available annotation databases:")
print(f"   • UCSC RefSeq table")
print(f"   • UCSC GenBank table")
print(f"   • Ensembl mapping tables")
```

**Note:** Setting `enable_remote=True` connects to online databases for more comprehensive annotation but is slower.

### Step 4: Annotate a Dataset

Now let's convert probe IDs to gene symbols:

```python
print("\n🧬 Annotating dataset...")

# Annotate the dataset
annotated_dataset = annotate_geo_dataset(dataset, annotation_engine)

# Compare before and after
print("\n📊 Annotation Results:")
print(f"   Original keys (first 5): {dataset['keys'][:5]}")
print(f"   Annotated keys (first 5): {annotated_dataset['keys'][:5]}")

# Check annotation report
if 'annotation_report' in annotated_dataset:
    report = annotated_dataset['annotation_report']
    
    print(f"\n📈 Annotation Statistics:")
    print(f"   Source ID type: {report['source_type']}")
    print(f"   Total identifiers: {report['total_identifiers']:,}")
    print(f"   Successfully mapped: {report['mapped_identifiers']:,}")
    print(f"   Mapping rate: {report['mapping_rate']:.1%}")
    print(f"   Unmapped: {len(report['unmapped_identifiers']):,}")
    
    # Show examples of unmapped IDs
    if report['unmapped_identifiers']:
        unmapped_sample = list(report['unmapped_identifiers'])[:5]
        print(f"   Unmapped examples: {', '.join(unmapped_sample)}")
```

**Expected output:**
```
🧬 Annotating dataset...

📊 Annotation Results:
   Original keys (first 5): ['1007_s_at', '1053_at', '117_at', '121_at', '1255_g_at']
   Annotated keys (first 5): ['DDR1', 'RFC2', 'HSPA6', 'PAX8', 'GUCA1A']

📈 Annotation Statistics:
   Source ID type: Affymetrix Probe ID
   Total identifiers: 54,675
   Successfully mapped: 20,453
   Mapping rate: 92.3%
   Unmapped: 4,222
   Unmapped examples: AFFX-BioB-5_at, AFFX-BioB-M_at, ...
```

**What happened?**
1. Detected probe ID format (Affymetrix)
2. Queried local annotation database
3. Mapped probes → gene symbols
4. Generated annotation quality report
5. Kept unmapped probes as-is (with warning)

### Step 5: Handle Multiple Probe-to-Gene Mappings

Some probes map to multiple genes (or vice versa):

```python
# Examine keys that might map to multiple genes
print("🔍 Checking for multi-gene mappings...")

# Count occurrences of each gene symbol
from collections import Counter

gene_counts = Counter(annotated_dataset['keys'])

# Find genes appearing multiple times
duplicated_genes = {gene: count for gene, count in gene_counts.items() if count > 1}

if duplicated_genes:
    print(f"\n   Found {len(duplicated_genes)} genes with multiple probes:")
    # Show top 5 most duplicated
    top_duplicated = sorted(duplicated_genes.items(), key=lambda x: x[1], reverse=True)[:5]
    for gene, count in top_duplicated:
        print(f"      • {gene}: {count} probes")
    
    print(f"\n   💡 This is normal! Multiple probes can target the same gene.")
    print(f"      Meta-analysis will handle this by summarizing probe-level data.")
else:
    print("   ✅ No duplicate gene mappings")
```

**Expected output:**
```
🔍 Checking for multi-gene mappings...

   Found 234 genes with multiple probes:
      • BRCA1: 8 probes
      • TP53: 6 probes
      • EGFR: 5 probes
      • MYC: 4 probes
      • PTEN: 4 probes

   💡 This is normal! Multiple probes can target the same gene.
      Meta-analysis will handle this by summarizing probe-level data.
```

### Step 6: Validate Gene Symbol Quality

Not all gene symbols are created equal. Let's validate quality:

```python
print("\n🔬 Validating gene symbol quality...")

# Take a sample for validation (full validation can be slow)
gene_sample = annotated_dataset['keys'][:500]

validation_results = validate_geo_gene_symbols(
    gene_sample, 
    annotation_engine
)

# Display validation metrics
metrics = validation_results['metrics']

print("\n📊 Validation Results:")
print(f"   Total symbols checked: {metrics['total_symbols']:,}")
print(f"   Valid symbols: {metrics['valid_symbols']:,} ({metrics['valid_rate']:.1%})")
print(f"   Invalid symbols: {metrics['invalid_symbols']:,} ({metrics['invalid_rate']:.1%})")
print(f"   Dubious symbols: {metrics['dubious_symbols']:,} ({metrics['dubious_rate']:.1%})")

# Show examples of problematic symbols
if validation_results['invalid_symbols']:
    invalid_sample = list(validation_results['invalid_symbols'])[:5]
    print(f"\n   ❌ Invalid examples: {', '.join(invalid_sample)}")
    
if validation_results['dubious_symbols']:
    dubious_sample = list(validation_results['dubious_symbols'])[:5]
    print(f"   ⚠️  Dubious examples: {', '.join(dubious_sample)}")
```

**Expected output:**
```
🔬 Validating gene symbol quality...

📊 Validation Results:
   Total symbols checked: 500
   Valid symbols: 463 (92.6%)
   Invalid symbols: 12 (2.4%)
   Dubious symbols: 25 (5.0%)

   ❌ Invalid examples: LOC100001, C9orf72-AS1, LINC00001, ...
   ⚠️  Dubious examples: RP11-123A4.1, CTD-2547L24.4, ...
```

**Symbol categories:**
- **Valid**: Standard HGNC gene symbols (BRCA1, TP53)
- **Invalid**: Non-standard or deprecated symbols
- **Dubious**: Provisional symbols, long non-coding RNAs (lncRNAs), read-through transcripts

### Step 7: Annotate Multiple Datasets

In meta-analysis, we need to annotate all datasets:

```python
# Download multiple datasets
print("📥 Downloading multiple datasets...")
meta_object = get_geo_data(
    gse_vector=["GSE5847", "GSE28750"],
    formatted_names=["Study_A", "Study_B"],
    cache_dir="./geo_cache"
)

print("\n🧬 Annotating all datasets...")

# Annotate each dataset
annotated_datasets = {}
for name, dataset in meta_object['originalData'].items():
    print(f"\n   Processing: {name}")
    
    annotated = annotate_geo_dataset(dataset, annotation_engine)
    annotated_datasets[name] = annotated
    
    # Show quick stats
    if 'annotation_report' in annotated:
        report = annotated['annotation_report']
        print(f"      ✅ Mapped {report['mapped_identifiers']:,} genes ({report['mapping_rate']:.1%})")

# Update meta-object
meta_object['originalData'] = annotated_datasets

# Check gene overlap across datasets
all_genes = [set(d['keys']) for d in annotated_datasets.values()]
common_genes = set.intersection(*all_genes)

print(f"\n📊 Cross-Dataset Gene Overlap:")
print(f"   Study_A genes: {len(all_genes[0]):,}")
print(f"   Study_B genes: {len(all_genes[1]):,}")
print(f"   Common genes: {len(common_genes):,}")
print(f"   Overlap: {len(common_genes)/min(len(g) for g in all_genes):.1%}")
```

**Expected output:**
```
📥 Downloading multiple datasets...

🧬 Annotating all datasets...

   Processing: Study_A
      ✅ Mapped 20,453 genes (92.3%)

   Processing: Study_B
      ✅ Mapped 20,451 genes (92.3%)

📊 Cross-Dataset Gene Overlap:
   Study_A genes: 20,453
   Study_B genes: 20,451
   Common genes: 18,234
   Overlap: 89.3%
```

**Good overlap** (>80%) indicates datasets are suitable for meta-analysis.  
**Poor overlap** (<50%) suggests platform incompatibility or quality issues.

### Step 8: Handle Platform-Specific Annotation

For platform-specific control, use the `GEOPlatformAnnotator`:

```python
from preprocessing.geo_annotation_engine import GEOPlatformAnnotator

print("🔧 Using platform-specific annotator...")

# Initialize platform annotator
platform_annotator = GEOPlatformAnnotator(annotation_engine)

# Get platform from dataset
dataset = meta_object['originalData']['Study_A']
platform_id = dataset.get('platform', 'Unknown')

print(f"   Platform: {platform_id}")

# Annotate with platform-specific logic
annotated_fdata = platform_annotator.annotate_platform_fdata(
    fdata=dataset.get('fData', pd.DataFrame()),
    platform_id=platform_id
)

print(f"   ✅ Platform-specific annotation complete")
```

## Tips & Gotchas

### Tip 1: Cache Annotation Results

Annotation can be slow for large datasets. Save results:

```python
import pickle

# Save annotated datasets
with open('annotated_datasets.pkl', 'wb') as f:
    pickle.dump(meta_object, f)

print("✅ Annotated datasets saved")

# Load later
with open('annotated_datasets.pkl', 'rb') as f:
    meta_object = pickle.load(f)
```

### Tip 2: Monitor Annotation Quality

Always check annotation rate before meta-analysis:

```python
def check_annotation_quality(dataset, min_rate=0.80):
    """Check if annotation rate is acceptable."""
    
    if 'annotation_report' not in dataset:
        print("⚠️ No annotation report available")
        return False
    
    report = dataset['annotation_report']
    mapping_rate = report['mapping_rate']
    
    if mapping_rate >= min_rate:
        print(f"✅ Good annotation rate: {mapping_rate:.1%}")
        return True
    else:
        print(f"❌ Poor annotation rate: {mapping_rate:.1%} (< {min_rate:.0%})")
        print(f"   Consider excluding this dataset or improving annotation")
        return False

# Check all datasets
for name, dataset in meta_object['originalData'].items():
    print(f"\n{name}:")
    check_annotation_quality(dataset)
```

### Tip 3: Handle Control Probes

Control probes (AFFX, ILMN controls) won't map to genes:

```python
# Identify and filter control probes
def filter_control_probes(dataset):
    """Remove control probes from dataset."""
    
    keys = dataset['keys']
    expr = dataset['expr']
    
    # Identify control probes
    is_control = [
        k.startswith('AFFX-') or k.startswith('ILMN_') 
        for k in keys
    ]
    
    n_controls = sum(is_control)
    print(f"   Found {n_controls:,} control probes")
    
    # Filter out controls
    keep_mask = [not c for c in is_control]
    dataset['keys'] = [k for k, keep in zip(keys, keep_mask) if keep]
    dataset['expr'] = expr.iloc[keep_mask]
    
    print(f"   Removed {n_controls:,} probes")
    print(f"   Remaining: {len(dataset['keys']):,} genes")
    
    return dataset

# Apply to a dataset
dataset = filter_control_probes(dataset)
```

### Gotcha 1: Annotation May Change Gene Order

After annotation, expression matrix and keys must stay aligned:

```python
# ❌ WRONG - keys changed but expr didn't
annotated = annotate_geo_dataset(dataset, engine)
dataset['keys'] = annotated['keys']  # Keys updated
# dataset['expr'] unchanged - NOW MISALIGNED!

# ✅ CORRECT - use the complete annotated dataset
annotated = annotate_geo_dataset(dataset, engine)
dataset = annotated  # Replace entire dataset
```

### Gotcha 2: Some Genes Have No Standard Symbol

Unmapped genes are kept with original IDs:

```python
# Check for unmapped genes in final dataset
original_ids = [
    k for k in annotated_dataset['keys'] 
    if '_at' in k or 'ILMN_' in k or 'AFFX' in k
]

if original_ids:
    print(f"⚠️ {len(original_ids)} genes still use original probe IDs")
    print(f"   These won't match across platforms in meta-analysis")
    print(f"   Examples: {original_ids[:5]}")
```

### Gotcha 3: Gene Symbol Case Matters

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

### Q: Low annotation rate (<80%)

**A**: Several possible causes:

1. **Old or custom platform**: Not in annotation database
   ```python
   # Check platform
   platform = dataset.get('platform', 'Unknown')
   print(f"Platform: {platform}")
   
   # Try enabling remote annotation
   engine = GeneAnnotationEngine(enable_remote=True)
   ```

2. **Non-standard probe IDs**: 
   ```python
   # Manually inspect probe format
   print(dataset['keys'][:20])
   # If format is unusual, may need custom mapping
   ```

3. **Organism mismatch**: Annotation is for wrong species
   ```python
   # Verify organism in phenotype data
   pheno = dataset.get('pheno', pd.DataFrame())
   if 'organism_ch1' in pheno.columns:
       print(pheno['organism_ch1'].unique())
   ```

### Q: Different studies have no overlapping genes

**A**: Check these common issues:

```python
# 1. Check if annotation was actually applied
for name, dataset in meta_object['originalData'].items():
    sample_keys = dataset['keys'][:5]
    has_probes = any('_at' in str(k) for k in sample_keys)
    
    if has_probes:
        print(f"❌ {name} still has probe IDs - annotation didn't work")
    else:
        print(f"✅ {name} has gene symbols")

# 2. Check platform compatibility
platforms = [d.get('platform') for d in meta_object['originalData'].values()]
print(f"Platforms: {platforms}")
# All should be for same organism

# 3. Check gene symbol format
all_genes = [set(d['keys']) for d in meta_object['originalData'].values()]
common = set.intersection(*all_genes)
print(f"Common genes: {len(common)}")

if len(common) < 1000:
    print("⚠️ Very few common genes - check annotation quality")
```

### Q: Memory error with large datasets

**A**: Annotate datasets one at a time:

```python
# Process datasets sequentially to save memory
for name in meta_object['originalData'].keys():
    dataset = meta_object['originalData'][name]
    
    # Annotate
    annotated = annotate_geo_dataset(dataset, annotation_engine)
    
    # Save immediately
    with open(f'{name}_annotated.pkl', 'wb') as f:
        pickle.dump(annotated, f)
    
    # Clear from memory
    del annotated
    print(f"✅ {name} annotated and saved")
```

### Q: Annotation is very slow

**A**: Speed up annotation:

1. **Use local annotation** (disable remote):
   ```python
   engine = GeneAnnotationEngine(enable_remote=False)  # Faster
   ```

2. **Cache results**:
   ```python
   # Save annotation engine state
   engine.save_cache('annotation_cache.pkl')
   
   # Load cached state
   engine = GeneAnnotationEngine.load_cache('annotation_cache.pkl')
   ```

3. **Process in batches** for very large datasets

## Next Steps

Now that your datasets have standardized gene symbols:

1. **[Quality Control Tutorial](quality_control.md)** – Verify gene overlap and data quality
2. **[Meta-Analysis Tutorial](meta_analysis.md)** – Perform statistical analysis
3. **[Complete Workflow Tutorial](complete_workflow.md)** – See annotation in full context

### Advanced Annotation Topics

- **Custom annotation databases** – Add your own probe-to-gene mappings
- **Cross-species annotation** – Handle orthologs for comparative studies
- **Ensembl ID mapping** – Work with Ensembl gene IDs instead of symbols
- **Isoform-level annotation** – Map to specific transcript isoforms

### Key Takeaways

✅ Use `annotate_geo_dataset()` to convert probe IDs to gene symbols  
✅ Check annotation quality with `validate_geo_gene_symbols()`  
✅ Aim for >80% mapping rate for reliable meta-analysis  
✅ Verify gene overlap across datasets before proceeding  
✅ Multiple probes per gene is normal and handled automatically  
✅ Filter control probes (AFFX, ILMN) before analysis  
✅ Save annotated datasets to avoid re-processing  

---

**Questions?** Check the [API Reference](../api/preprocessing.md) or see [Best Practices](../user_guide/best_practices.md).
