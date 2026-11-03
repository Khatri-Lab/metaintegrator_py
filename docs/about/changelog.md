# Release Notes

All notable changes to MetaIntegrator-py will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## Version numbering

Every release before the first on-prem deployment of the MetaIntegrator web
app is numbered 0.x.x. The on-prem deployment is **1.0.0**, released on
2026-11-04.

On 2026-10-08 the two earlier releases were renumbered to fit this plan:

| Old number | New number | Date |
|---|---|---|
| 1.0.0 | 0.10.0 | 2026-04-22 |
| 1.1.0 | 0.11.0 | 2026-06-10 |

Nothing about those releases changed except the number.
The installed package metadata said 0.1.0 the whole time, so `pip` sees
1.0.0 as an upgrade from any earlier install.

This 1.0.0 is a new release. It is not the release of 2026-04-22
that was once called 1.0.0 (now 0.10.0).

The changes listed under 1.0.0 were developed under the working number
0.12.0. No release was ever published as 0.12.0. Test builds of the web app
installed it with that number, so runs made with them record
`library_version` 0.12.0. Code comments that say "since 0.12.0" or
"before 0.12.0" mean this release.

---

## [1.0.0] - 2026-11-04

First on-prem deployment, with the MetaIntegrator web app 1.0.0.
Developed as 0.12.0 (see "Version numbering" above).

### Changed
- **GEO folder listing now uses HTTPS first, with FTP as a fallback.**
  `HybridGeoSource` listed each series' `matrix/` folder over plain FTP
  (port 21). Many networks block FTP. Then every series waited about 90
  seconds and guessed the single-platform file name, which is wrong for
  series with more than one platform (`GSExxx-GPLyyy_series_matrix.txt.gz`),
  so those series failed. The new `NCBINetwork.list_directory()` reads
  NCBI's HTTPS index page and tries FTP only if HTTPS fails. A missing folder
  (HTTP 404) raises `DirectoryNotFoundError` at once, without retries or FTP.
  `list_ftp_directory()` is unchanged. New public helpers:
  `list_https_directory()` and `parse_index_page()`.
- **Gene-level values now match R when a probe value is missing.**
  `get_sample_level_gene_data()` (both copies: `coconut_integration` and
  `sample_level_gene_data`) averaged a gene's probes and skipped missing
  values. R's `colMeans()` does not skip them, so a gene with probes
  `[missing, 8]` is missing and then set to 1. The library now does the same.
  An exact zero that the log2 step made missing (R holds `-Inf`) also gives 1
  now. Expect different values, and possibly scores, in the cells where a gene
  has two or more probes and one is missing. Simulated data: 0.04% of cells at
  0.01% zero probe values, 3.2% at 1%. Register entry REG-003.
- **`SignatureScoringEngine` now scores like R `calculateScore()` by default.**
  The default method is `"geometric_mean_zscore"`. It delegates to
  `ScoreCalculator.calculate_score()`: shift the dataset to positive values,
  take the geometric mean of the positive genes minus that of the negative
  genes, then z-score per dataset. The old default, `"mean_difference"`
  (a plain arithmetic mean), is still available as an explicit opt-in.
  Every ROC curve, PRC curve and pooled AUC built on this class now matches
  R. On `tinyMetaObject` the per-dataset AUCs are 0.856, 0.867 and 0.955,
  the same as R's own unit tests. Expect different AUCs from earlier runs.
- `calculate_signature_scores()` follows the same new default.
- Package version is now 1.0.0. Before, the package metadata still said
  0.1.0 while this changelog listed 1.0.0 and 1.1.0. See
  "Version numbering" above. `CITATION.cff` now follows the package version,
  and a test (`tests/test_version_consistency.py`) keeps them in sync.
- **`filter_genes()` leave-one-out now follows R `filterGenes()`.**
  With `is_leave_one_out=True`, a gene must pass every threshold in every
  leave-one-out run, with one fewer study required. The main result is not
  filtered in this mode. Before, Python filtered the main result and only
  checked that the effect sign stayed the same across runs. Expect fewer
  genes. On `tinyMetaObject` with default thresholds, the signature goes
  from 50 up / 28 down genes to 18 up / 13 down, the same as R.
- `FilterResult.is_leave_one_out` now reports whether the leave-one-out rule
  was actually applied. It is `False` when leave-one-out results are missing
  or there are only two datasets, as in R.
- **`coconut()` stops early in by-study mode (`by_platform=False`) if a dataset
  has no control samples or no disease samples.** The message names the dataset,
  for example `Dataset 'GSE123' has < 1 control sample`. R `COCONUT()` also
  stops when a dataset has no controls. Before, the dataset was skipped, and
  the run ended in a confusing ComBat error, or, when two different datasets
  were missing, ran with the wrong batch parameters. With `by_platform=True`
  nothing changes.

- **The GEO loaders now decide "log2 or not" exactly as R `getGEOData()`
  does.** `HybridMicroarrayHandler` (the `"hybrid"` strategy) and
  `MicroarrayHandler` used "log2 when the median of all values is above 20".
  They now use `preprocessing.utils.log_scale.apply_r_log2()`, an exact port
  of R's `.GEM_log_check` (a QQ-plot test on the first usable sample column)
  and of its transform (shift up when the minimum is negative, then `log2`).
  The two rules can disagree on borderline datasets, in both directions, so
  expect a different log2 decision for some datasets. Every
  `ProcessedDataset` from these handlers now carries
  `metadata["log_scale_check"]`: the decision, R's score and correlations,
  the median before, any shift, and what the former rule would have done.
  One documented difference from R remains: an exact zero becomes `NaN`
  instead of `-Inf`, and the report counts how many.
  `MicroarrayHandler` also no longer clips negative values to zero before
  `log2`. It shifts them up, as R does.

### Added
- **R parity register** (`docs/comparison/r_parity_register.md`): every known
  difference from R, with what R does, what we do, the measured effect, and the
  test that pins it. First entries: exact zeros before `log2` (REG-001), R's
  log-scale rule (REG-002), and the probe average and missing values (REG-003,
  resolved, see Changed).
- `tests/parity/test_register_pins.py`, which pins those entries.
- **Potential Weaknesses in R** (`docs/comparison/r_potential_weaknesses.md`):
  a separate page with a stricter evidence bar than the register. Each entry
  has a category, three evidence flags, an opt-in alternative and a pinned R
  version (MetaIntegrator 2.1.3). First entries: the log-scale rule reads one
  sample (W-01), and the shift for negative values that `.GEM_log_check()`
  computes and does not use (W-02, pinned by `TestW02UnusedShift`). Fisher
  `sum_of_logs` is a placeholder with no claim.
- `tests/parity/measurements/measure_register_effects.py`, a tool (not a test)
  that measures the size of REG-002 and of REG-001 and REG-003 on simulated
  data or on your own raw matrix. The new `unused-shift` command measures W-02.
- `preprocessing.utils.log_scale`: `gem_log_check()`, `apply_r_log2()`,
  `LogScaleCheck`, `LogScaleReport`, and exact ports of R `ppoints()`,
  `qqnorm()` and `cor(use = "pairwise.complete")`.
- Log-scale parity test (`tests/parity/test_log_check_parity.py`) with ten
  fixed input matrices and an R script that writes R's own decisions
  (`tests/parity/r_scripts/make_log_check_golden.R`). The test is skipped
  until the R script has been run once.
- `SignatureScoringEngine.DEFAULT_METHOD` and `SignatureScoringEngine.METHODS`,
  plus the module constants `DEFAULT_SCORING_METHOD`, `SCORING_METHODS`,
  `SCORING_METHOD_GEOMETRIC_MEAN_ZSCORE` and `SCORING_METHOD_MEAN_DIFFERENCE`.
  Callers can store the method with their results, so old and new numbers
  can be told apart.
- R parity tests for signature scores, per-dataset AUC and pooled AUC
  (`tests/parity/test_scoring_parity.py`).
- Automated R parity tests (`pytest -m parity`) that replay R golden sets.
  See *Contributing > R Parity Testing*.
- **Dataset names in COCONUT.** Each dataset passed to `coconut()` may have an
  optional `'name'`. Without names, the datasets are called `dataset_1`,
  `dataset_2` and so on. The name is copied to every control and disease block
  in the result. `load_gse_data_from_csv_directory()` and
  `COCONUTIntegrator` now set it from the file name or the dataset key.
- `combine_coco_output(..., strict=False)`. With `strict=True`, a dataset that
  is in only one of the two lists raises a `ValueError`.

- **`GeneSearchOptimizer.per_dataset_aucs()`.** Returns the AUC and sample
  count of each discovery dataset for a given signature. The weighted sum of
  these AUCs is the number a search reports as its final AUC. Pass
  `ratio=None` for the plain score (R's end-of-search printout, without R's
  rounding). Pass a ratio to rebuild a search's last move. A dataset that
  cannot be converted is left out. An empty signature raises `ValueError`.
- **`GeneSearchOptimizer.search_ratio()`.** Rebuilds the ratio a search used
  for its last accepted move from the final gene lists. No search state is
  needed. The rule is `n_neg / n_pos`, or `None` when the search made no move
  or one gene list is empty. Tests show it reproduces the last AUC of forward
  and backward searches to 1e-12.
- Datasets converted for a search now carry a `name` (the key in
  `originalData`), as R does. Before, a skipped dataset left no trace in the
  list, so a position could not be matched to a dataset. The per-dataset
  scoring step is now one private helper shared by the search and
  `per_dataset_aucs()`. Search results do not change.
- Parity tests that compare the plain search AUC of each dataset with R's
  `calculateScore()` AUCs on `tinyMetaObject`. They match for the all-positive
  dataset, and after R's own shift for the other two. On the data as shipped
  they split for the two datasets with negative values.

### Fixed
- `check_log_scale()`'s docstring said it matches R's `.GEM_log_check`. It
  does not (it adds shortcut rules and samples at random). The docstring now
  says so and points to `gem_log_check()`. Its behavior is unchanged.
- **`combine_coco_output()` pairs control and disease blocks by dataset name.**
  Before, it paired the two lists by position. A dataset with no controls
  (possible with `by_platform=True`) is missing from the control list, so every
  later pair shifted. Samples were dropped and mislabeled without any message.
  On a three-dataset example, 17 of 21 samples came back. Now all 21 do.
  A dataset that is in only one list is kept, and a `UserWarning` names it.
  Output with no names, for example built by hand, is still paired by position
  if both lists have the same length. Otherwise a `ValueError` is raised.
- Probes with no usable effect size no longer turn a multi-probe gene's
  pooled effect into NaN (R drops them first).
- When a platform table has no gene-symbol column, probe IDs used as keys
  now go through the Ensembl check, as in R.
- **GEO series and platform files are downloaded over HTTPS first.**
  `GEOparse.get_GEO()` downloads a series' family SOFT file over FTP. Where
  FTP is blocked, every load waited for its timeouts and failed. New
  `preprocessing/geo_https.py` puts the file where GEOparse looks for it,
  over HTTPS, before each `get_GEO()` call by accession: in
  `HybridGeoSource`, `GEOparseSource` and the standalone GPL fallback.
  Downloads go to a temporary name, a `.gz` file is read through once, and
  only then is it renamed. If HTTPS fails, GEOparse tries FTP, as before.
- **`get_sample_level_gene_data()` reads COCONUT's corrected genes.**
  The corrected gene matrix comes back with its index named `keys`. The
  extractor adds a `keys` column, so pandas refused the next
  `groupby("keys")` as ambiguous, and no value could be extracted.
  The extractor now drops the index name on its own copy. The caller's
  matrix is unchanged.

### Planned
- Interactive web dashboard for meta-analysis results
- Support for RNA-seq data normalization
- Distributed computation backend
- Cloud-based computation backend
- Integration with additional gene annotation databases

---

## [0.11.0] - 2026-06-10 (formerly 1.1.0)

### 🎯 R End-to-End Workflow Parity Release

Achieved near-perfect numerical parity with R MetaIntegrator across all
analysis stages: gene annotation, filtering, score calculation, and AUC
evaluation.

### Fixed

#### Gene Annotation (→ 100% gene identity match with R)
- **GenBank accession filter** (`gene_annotator.py`): Added regex filter
  rejecting ~3,400 GenBank/EMBL accession IDs (AF*, AK*, BC*, etc.) that
  inflated gene counts from ~25,800 to ~29,200 vs R’s ~25,800.
- **Ensembl validation restored** (`hybrid_handler.py`): Reverted
  `validator=None` back to `validator=validator`, re-enabling HGNC symbol
  validation that filters LOC*, mitochondrial aliases, and ORFs.

#### Score Calculation (→ AUC within ±0.002 of R)
- **Score pre-filter removed** (`score_calculation.py`): Removed the
  `pos_genes_in_dataset = [g for g in pos_genes if g in keys]` pre-filter
  that excluded missing genes. Now passes ALL signature genes to
  `SampleLevelGeneDataProcessor`, which fills missing genes with value 1
  (`log(1)=0` enters the geometric mean), matching R’s `calculateScore`.
  This fixed GPL97’s AUC from 0.873 → 1.000.

### Added

- **`violin_plot.py`**: R-parity violin plots with ggplot2/ggpubr aesthetic
  (individual violins, boxplot overlay, jittered points, Wilcoxon p-value
  brackets). Supports single-dataset and multi-dataset grid layouts.
- **`diagnostic_auc_discrepancy_v2.py`**: In-memory diagnostic script
  (no stale pickle dependency) with 9 checks: provenance, gene identity
  comparison vs R, column corruption, per-dataset overlap, score trace,
  AUC recomputation, and key count comparison.
- **Architecture documentation** (`docs/architecture.md`): Mermaid diagrams
  covering system architecture, factory pattern, annotation pipeline, and
  scoring pipeline.

### Changed

- Updated all API documentation to reflect OO redesign
  (Factory/Handler/Source pattern, scoring subsystem)
- Rebuilt `docs/api/overview.md`, `preprocessing.md`, `analysis.md`,
  `visualization.md` for current codebase
- Added R-parity status to feature comparison matrix

---

## [0.10.0] - 2026-04-22 (formerly 1.0.0)

### 🎉 Initial Feature Completion Release

MetaIntegrator-py v0.10.0 (formerly v1.0.0) represents a complete Python port of the R MetaIntegrator package, bringing modern performance and usability to meta-analysis of gene expression data.

### Added

#### Core Functionality
- **Meta-Analysis Engine**: Complete implementation of random-effects meta-analysis with leave-one-out analysis
- **GEO Integration**: Direct download and processing of datasets from Gene Expression Omnibus
- **Data Validation**: Comprehensive quality control and data structure validation
- **Statistical Analysis**: Effect size calculation, heterogeneity testing, and FDR correction

#### Data Processing
- Gene symbol annotation and correction
- Multiple platform support (Affymetrix, Illumina, Agilent)
- Automated phenotype processing
- Sample-level gene data handling

#### Analysis Features
- `MetaAnalyzer.run_meta_analysis()`: Core meta-analysis with pooled effect sizes
- `MetaAnalyzer.run_leave_one_out_analysis()`: Sensitivity analysis by dataset exclusion (also available via `run_meta_analysis(run_leave_one_out=True)`)
- `filter_genes()`: Multi-criteria gene filtering (FDR, effect size, heterogeneity)
- `EnhancedGeneScorer` / `MetaIntegratorOrchestrator`: Gene signature scoring
- `analyze_roc_curve()`, `pooled_roc_plot()`: Receiver operating characteristic analysis
- `calc_meta_power()`: Statistical power estimation

#### Visualization
- Forest plots with effect sizes and confidence intervals
- Volcano plots for differential expression
- Heatmaps with hierarchical clustering
- ROC/PRC curves with AUC statistics
- Violin plots for score distributions

#### Utilities
- COCONUT integration for batch correction
- Immune deconvolution using ImmunoStates reference matrix
- Gene search optimizer for efficient lookups
- UCSC/Ensembl/Entrez ID conversion

### Changed
- **Performance**: 2.8x faster than R implementation through NumPy/Pandas optimization
- **Memory**: 40% reduction in memory usage for large datasets
- **API**: Pythonic interface with type hints and comprehensive docstrings

### Technical Details
- Python 3.8+ compatibility
- NumPy-based vectorized computations
- Pandas DataFrames for data structures
- Matplotlib/Seaborn for publication-quality plots
- Comprehensive pytest test suite (95%+ coverage)

---

## Version History Summary

| Version | Date | Key Features                        |
|---------|------|-------------------------------------|
| 1.0.0 | 2026-11-04 | On-prem release: R-parity signature scoring, R-parity LOO filter, GEO over HTTPS |
| 0.11.0 | 2026-06-10 | R end-to-end parity (formerly 1.1.0) |
| 0.10.0 | 2026-04-22 | Internal feature completion testing (formerly 1.0.0) |
| 0.9.0 | 2026-03-15 | Initial gap analysis testing        |
| 0.5.0 | 2026-02-01 | Core meta-analysis completion       |
| 0.1.0 | 2025-12-10 | Project initiation                  |

---

## Migration from R MetaIntegrator

For users migrating from the R package, see our [Migration Guide](../comparison/migration_guide.md) for:
- Function name mappings (R → Python)
- Workflow comparisons
- Performance benchmarks
- Breaking changes and compatibility notes

---

## Contributing

We welcome contributions! See our [Development Guide](../contributing/development.md) for:
- Setting up your development environment
- Running tests
- Submitting pull requests
- Code style guidelines

---

## Links

- [GitHub Repository](https://github.com/Khatri-Lab/metaintegrator_py)
- [Issue Tracker](https://github.com/Khatri-Lab/metaintegrator_py/issues)
- [Khatri Lab](https://khatrilab.stanford.edu/)

---

**Note**: Pre-1.0 versions were internal development releases and are not publicly available.
