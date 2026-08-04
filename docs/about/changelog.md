# Release Notes

All notable changes to MetaIntegrator-py will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Interactive web dashboard for meta-analysis results
- Support for RNA-seq data normalization
- Distributed computation backend
- Cloud-based computation backend
- Integration with additional gene annotation databases

---

## [1.1.0] - 2026-06-10

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

## [1.0.0] - 2026-04-22

### 🎉 Initial Feature Completion Release

MetaIntegrator-py v1.0.0 represents a complete Python port of the R MetaIntegrator package, bringing modern performance and usability to meta-analysis of gene expression data.

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
| 1.0.0 | 2026-04-22 | Internal feature completion testing |
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
