# Probe-to-Gene Mapping and Multi-Probe Genes

Microarrays measure probes, not genes.
Meta-analysis compares genes across studies.
This page explains how metaintegrator_py bridges that gap.

It answers two questions:

1. How does a probe get its gene symbol?
2. What happens when several probes measure the same gene?

The short answer:

- At **load time**, every probe gets a gene key. The expression matrix stays at the probe level.
- At **analysis time**, probes are collapsed to genes. The collapse rule depends on the step.

Both stages follow R MetaIntegrator closely.
Known differences are listed near the end of this page.

---

## The Big Picture

```text
GEO Series Matrix (probe x sample)      GPL annotation table
            |                                    |
            |                                    v
            |                         GeneAnnotator.annotate()
            |                           1. find symbol column
            |                           2. clean each value
            |                           3. Ensembl + accession filters
            v                                    v
   ProcessedDataset.expr  <-- aligned 1:1 -->  ProcessedDataset.keys
   (still probe-level)                        ("TP53", "CD6,IL2RG", "NA", ...)
            |
            |  analysis time
            v
  effect sizes -> drop NaN probes -> expand multi-gene keys -> drop "NA"
              -> pool probes per gene (fixed-effect inverse variance)
  p-values     -> expand -> keep the smallest p-value per gene
  scores       -> expand -> average probes per gene, sample by sample
```

---

## Part 1: Where the Mapping Lives

A loaded dataset is a `ProcessedDataset`.
It is defined in `src/preprocessing/base.py`.

| Field  | Shape                  | Meaning                                    |
|--------|------------------------|--------------------------------------------|
| `expr` | probes x samples       | Expression values. Index = probe IDs.      |
| `keys` | list, one per probe    | Gene key for each row of `expr`.           |
| `pheno`| samples x columns      | Sample metadata.                           |

`keys` mirrors the R `$keys` vector.
The constructor checks two rules:

- `keys` must be a `list`.
- `len(keys)` must equal the number of rows in `expr`.

A key can take three forms:

| Key            | Meaning                                  |
|----------------|------------------------------------------|
| `"TP53"`       | The probe maps to one gene.              |
| `"CD6,IL2RG"`  | The probe maps to several genes.         |
| `"NA"`         | The probe could not be mapped.           |

Note that `"NA"` is a string.
It is not `None` or `NaN`.

The loader never merges probes.
That is deliberate.
Each later step decides for itself how to combine probes.

---

## Part 2: How a Probe Gets Its Gene Symbol

The production loader is the `hybrid` strategy.
It pairs `HybridGeoSource` with `HybridMicroarrayHandler`.
Expression comes from the GEO Series Matrix file.
Annotation comes from the GPL platform table.

The handler calls `GeneAnnotator.annotate()`.
It lives in `src/preprocessing/gene_annotator.py`.
The logic ports R's `.GEO_fData_key_parser()` from `getGEOData.R`.

### Step 1: Find the gene-symbol column

The annotator checks the GPL table for these column names, in order.
The first match wins.

```text
Gene Symbol, GENE_SYMBOL, Gene_Symbol, gene_symbol, genesymbol,
Symbol, SYMBOL, symbol, HUGO_Gene_Nomenclature,
gene_assignment, Gene, GENE, gene, ORF
```

`Gene Symbol` covers most Affymetrix 3' arrays, such as GPL96 and GPL570.
`gene_assignment` covers Affymetrix Gene ST arrays, such as GPL6244.

### Step 2: Clean each annotation value

How a value is cleaned depends on the column type.

**Direct symbol columns** (`Gene Symbol`, `Symbol`, and similar).
The value already is the symbol.
Separators `///` and `;` become commas.
All genes are kept.
This matches R's `gsub(";| /// ", ",", keys)`.

| Raw GPL value              | Key            |
|----------------------------|----------------|
| `TP53`                     | `TP53`         |
| `CD6 /// IL2RG`            | `CD6,IL2RG`    |
| `HLA-DRB1;HLA-DRB4`        | `HLA-DRB1,HLA-DRB4` |
| `---`, empty, or missing   | `NA`           |

**Assignment columns** (`gene_assignment`, `mrna_assignment`).
Each entry looks like `accession // symbol // description // ...`.
The annotator takes the symbol from the second field.
It reads only the **first** `///` entry.
So these probes always get one gene.
This matches R's "Case 2" parsing.

| Raw GPL value                                                    | Key        |
|------------------------------------------------------------------|------------|
| `NM_002046 // GAPDH // glyceraldehyde... /// NR_... // GAPDHP1 // ...` | `GAPDH` |

### Step 3: Filter out keys that are not real genes

Two filters run after cleaning.
Both look only at the **first** gene in a key.
If the first gene fails, the whole key becomes `"NA"`.
If the first gene passes, the whole key is kept, even if a later gene is unknown.
This is exactly what R does.

1. **Ensembl check.**
   The first gene must appear in the bundled Ensembl symbol list.
   This is `EnsemblGeneValidator`.
   R uses `ens_ensgID_table$display_label` for the same check.
2. **Accession check.**
   GenBank or EMBL accessions such as `AK123456` are rejected.
   Without this filter, Python kept about 3,400 extra pseudo-genes on some platforms.

### Step 4: When no symbol column exists

Some GPL tables have none of the known columns.
In that case the annotator uses the probe IDs as keys.
These keys then go through the same filters as above.

So with the Ensembl validator, a probe ID like `1007_s_at` becomes `"NA"`.
A probe ID that is itself a gene symbol, such as `TP53`, is kept.
This matches R's "Case 0a".

This matters for the next step.

### Step 5: Standalone GPL fallback

Sometimes the GPL table inside the GSE "family" file yields zero genes.
When that happens, `HybridMicroarrayHandler` downloads the standalone GPL record and annotates again.

---

## Part 3: What Happens With Multiple Probes per Gene

There are two different situations.

- **Many probes, one gene.** For example, GPL570 has several probes for `TP53`.
- **One probe, many genes.** The key looks like `"CD6,IL2RG"`.

Both are resolved at analysis time.
The first step is always the same: **expand** multi-gene keys.
A probe keyed `"CD6,IL2RG"` becomes two rows, one for `CD6` and one for `IL2RG`.
Both rows carry the same probe values.
So that probe counts once for each of its genes.

After expansion, the rule for "many probes, one gene" depends on the step.

| Step                          | Python function                                         | R function                     | Rule                                  |
|-------------------------------|---------------------------------------------------------|--------------------------------|---------------------------------------|
| Effect sizes (meta-analysis)  | `MetaAnalysisCore.collapse_probes_to_genes()`           | `.summ.eff.within()`           | Fixed-effect inverse-variance pooling |
| Fisher p-values               | `FisherCombination.summarize_significance_within_study()` | `.summ.sigs.within()`        | Smallest p-value per gene             |
| Sample-level data and scores  | `SampleLevelGeneDataProcessor._extract_data_from_gem()` | `.extractDataFromGEM()`        | Mean of probes, per sample            |

### 3a. Effect sizes: inverse-variance pooling

This is the path that drives the main meta-analysis results.
It lives in `src/metaintegrator/core_analysis.py` and is called from `InverseVarianceEngine.combine_effect_sizes()`.

For each study, in order:

1. Compute Hedges' g and its standard error for every probe.
2. Drop probes whose g could not be computed.
   This happens when a group has fewer than two usable samples or when the probe has zero variance.
3. Expand multi-gene keys.
4. Drop `"NA"` keys.
5. For each gene:
   - one probe: keep its g and SE as is;
   - several probes: pool them with a fixed-effect model.

The fixed-effect formulas are:

```text
w_i      = 1 / SE_i^2
g_gene   = sum(w_i * g_i) / sum(w_i)
SE_gene  = sqrt(1 / sum(w_i))
```

R does the same with `rmeta::meta.summaries(method = "fixed")`.

**Worked example.**
Two probes measure TP53 in one study.

| Probe | g   | SE  | Weight (1/SE^2) |
|-------|-----|-----|-----------------|
| p1    | 1.0 | 1.0 | 1.00            |
| p2    | 3.0 | 2.0 | 0.25            |

```text
g_TP53  = (1.0 * 1.00 + 3.0 * 0.25) / 1.25 = 1.40
SE_TP53 = sqrt(1 / 1.25)                    = 0.894
```

The more precise probe gets more weight.
The pooled SE is smaller than either probe's SE.

The result is one row per gene per study.
Studies are then merged by gene symbol.
Finally, genes are pooled across studies.
That last step uses a random-effects model by default.

**Option: `within_method="extreme"`.**
Instead of pooling, keep the probe with the largest `|g / SE|`.
R offers the same option.
The default is `"fixed_iv"`.

### 3b. Fisher p-values: smallest p-value wins

For the Fisher's method branch, each probe gets a one-sided p-value for "up" and for "down".
Probes with a missing p-value or an `"NA"` key are dropped.
After expansion, each gene keeps the **smallest** `P.up` and the smallest `P.down` across its probes.
The two minimums are taken separately, so they can come from different probes.

### 3c. Sample-level data and scores: mean of probes

`getSampleLevelGeneData()` and the signature score use this path.
For each requested gene, the probe values are averaged sample by sample.
Values at or below zero, and missing values, are then replaced with 1.
This matches R, which applies the same replacements before the geometric mean.

---

## Part 4: Known Differences From R

These are the gaps we know about today.
None of them change results for the common Affymetrix and Illumina platforms we test against.

| Area | R MetaIntegrator | metaintegrator_py |
|------|------------------|-------------------|
| Finding the symbol column | Searches cell **contents** for `GAPDH` or `ICAM1`. | Matches known column **names**. |
| Non-symbol annotations | Converts ENSG, RefSeq, GenBank, and Entrez columns to symbols (R Cases 3 to 6). | `GeneAnnotator` does not do this yet. `GeneConverter` has the lookup tables but is not wired into the loader. |
| Accession inside a direct column | Keeps `NM_...` as the first gene, so the Ensembl check turns the key into NA. | Skips `NM_...` and `ENST...` entries, so a later symbol can become the first gene. |
| Missing value in one probe (sample-level) | `colMeans()` gives NA for that sample, which then becomes 1. | `mean()` skips the missing probe and averages the rest. |

The legacy `MicroarrayHandler` (strategies `full` and `fast`) has its own copy of the cleaning logic.
The `hybrid` strategy is the one to use for R parity.

---

## Part 5: Quick Checks on Your Own Data

```python
from collections import Counter

ds = datasets[0]  # a ProcessedDataset

n_probes = len(ds.keys)
n_na = sum(k == "NA" for k in ds.keys)
n_multi_gene = sum("," in k for k in ds.keys if k != "NA")

# Expand keys the same way the analysis does
gene_probe_counts = Counter(
    gene for k in ds.keys if k != "NA" for gene in k.split(",")
)
n_multi_probe = sum(1 for c in gene_probe_counts.values() if c > 1)

print(f"Probes:                 {n_probes}")
print(f"Unmapped (NA):          {n_na}")
print(f"Multi-gene probes:      {n_multi_gene}")
print(f"Genes:                  {len(gene_probe_counts)}")
print(f"Genes with >1 probe:    {n_multi_probe}")
```

Count genes after splitting on commas.
Counting raw keys treats `"CD6,IL2RG"` as one gene and undercounts.

---

## References

**Source code (this project)**

- `src/preprocessing/base.py`: `ProcessedDataset`
- `src/preprocessing/hybrid_handler.py`: `HybridMicroarrayHandler`
- `src/preprocessing/gene_annotator.py`: `GeneAnnotator`
- `src/preprocessing/gene_validation.py`: `EnsemblGeneValidator`
- `src/metaintegrator/core_analysis.py`: `collapse_probes_to_genes()`, `summarize_effects_within_study()`, `summarize_significance_within_study()`
- `src/metaintegrator/analysis.py`: `InverseVarianceEngine.combine_effect_sizes()`
- `src/metaintegrator/sample_level_gene_data.py`: `SampleLevelGeneDataProcessor`

**Original R implementation (MetaIntegrator)**

- `R/getGEOData.R`: `.GEO_fData_key_parser()`
- `R/metaAnalysisFoundation.R`: `.expand.df()`, `.combine.effect.sizes()`, `.summ.eff.within()`, `.combine.significances()`, `.summ.sigs.within()`
- `R/getSampleLevelGeneData.R`: `.extractDataFromGEM()`, `.summarizeSplit()`