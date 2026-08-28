# MetaIntegrator-py

A Python implementation of [MetaIntegrator](https://cran.r-project.org/package=MetaIntegrator) for multi-cohort meta-analysis of gene expression data. Provides the same trusted statistical methods as the original R package — effect size combination, leave-one-out analysis, gene filtering — with a modern Python interface built on NumPy, pandas, and scikit-learn.

Developed by the [Khatri Lab](https://khatrilab.stanford.edu/) at Stanford University.

## Key Capabilities

- **Meta-analysis** — Hedges' g effect sizes, inverse-variance pooling, Fisher's method, FDR correction, and leave-one-out validation across multiple GEO datasets
- **COCONUT batch normalization** — COmbat CO-Normalization Using conTrols for removing technical batch effects across studies
- **GEO integration** — Direct download and processing of microarray datasets from NCBI Gene Expression Omnibus
- **Gene annotation** — Automated probe-to-gene-symbol mapping via platform annotations (GPL570, GPL96, GPL97, GPL6244)
- **ImmunoStates deconvolution** — Cell-type proportion estimation from bulk expression data
- **Visualization** — Forest plots, violin plots, heatmaps, ROC/PRC curves with matplotlib/seaborn
- **R parity** — Numerical validation against the original R MetaIntegrator to ensure equivalent analytical results

## Quick Start

```python
from preprocessing.loader_factory import DataLoaderFactory
from metaintegrator.analysis import MetaAnalyzer

# Download datasets from GEO (cached after first run)
loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
datasets = loader.load_multiple(["GSE5847", "GSE28750"])

# Build meta-object and run meta-analysis
meta_object = {"originalData": {ds.geo_id: ds.to_dict() for ds in datasets}}
analyzer = MetaAnalyzer()
results = analyzer.run_meta_analysis(meta_object)

# View top significant genes
pooled = results["metaAnalysis"]["pooledResults"]
print(pooled.nsmallest(10, "effectSizeFDR")[["effectSize", "effectSizeFDR"]])
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Khatri-Lab/metaintegrator_py.git
cd metaintegrator_py

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Python 3.10+** is required. Core dependencies: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, statsmodels, GEOparse.

## Project Structure

```
metaintegrator_py/
├── src/
│   ├── metaintegrator/    # Core analysis, scoring, filtering, ROC/PRC
│   ├── preprocessing/     # GEO loading, gene annotation, normalization
│   ├── coconut/           # COCONUT batch normalization
│   ├── visualization/     # Plotting (forest, violin, heatmap, ROC)
│   └── utils/             # Configuration, export, diagnostics
├── tests/                 # Unit and integration tests (pytest)
├── data/                  # Reference data, annotations, test fixtures
├── tutorials/             # Jupyter notebooks with SLE worked examples
├── docs/                  # MkDocs documentation source
├── mkdocs.yml             # Documentation site configuration
├── pyproject.toml         # Build and pytest configuration
└── requirements.txt       # Python dependencies
```

## Running Tests

```bash
# Run the full test suite
pytest

# Run a specific module
pytest tests/metaintegrator/test_core_analysis.py -v

# Skip slow integration tests
pytest -m "not slow"
```

## Documentation

Full documentation is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). To serve it locally:

```bash
pip install mkdocs-material mkdocs-minify-plugin
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser. Documentation covers installation, tutorials, API reference, COCONUT workflow, and an R-to-Python migration guide.

## Tutorials

The `tutorials/` directory contains Jupyter notebooks with end-to-end worked examples using real Systemic Lupus Erythematosus (SLE) datasets from GEO. These demonstrate the full pipeline from data acquisition through meta-analysis to publication-quality figures.

## Citation

If you use this software, please cite the original methodology paper:

> Haynes WA, Vallania F, Liu C, Rinchai D, Uhlig J, Banchereau J, Pascual V, Palucka K, Khatri P. **Empowering multi-cohort gene expression analysis to increase reproducibility.** *Bioinformatics.* 2017;33(4):596–598. doi:[10.1093/bioinformatics/btw694](https://doi.org/10.1093/bioinformatics/btw694)

If using COCONUT batch correction, also cite:

> Sweeney TE, Haynes WA, Vallania F, Ioannidis JPA, Khatri P. **Methods to increase reproducibility in differential gene expression via meta-analysis.** *Nucleic Acids Research.* 2017;45(1):e1. doi:[10.1093/nar/gkw797](https://doi.org/10.1093/nar/gkw797)

See [CITATION.cff](CITATION.cff) for the machine-readable citation file.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
