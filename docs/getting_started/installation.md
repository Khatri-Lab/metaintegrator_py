# Installation

## Requirements

- Python 3.10 or higher
- pip (Python package installer)

## Install from PyPI (Coming Soon)

```bash
pip install metaintegrator_py
```

## Install from Source (Current)

```bash
# Clone repository
git clone https://github.com/Khatri-Lab/metaintegrator_py.git
cd metaintegrator_py

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Verify Installation

```python
import metaintegrator
print(metaintegrator.__version__)
```

## Dependencies

Core packages (see `requirements.txt` for the authoritative list):

- scientific stack: pandas >= 1.5.0, numpy >= 1.21.0, scipy >= 1.9.0, scikit-learn >= 0.24.0
- additional statistical analysis: statsmodels >= 0.14.2
- plotting: matplotlib >= 3.10.8, seaborn >= 0.13.2, plotly >= 5.0.0
- GEO data access: GEOparse >= 2.0.3, requests >= 2.25.0, lxml >= 4.6.0
- Excel file support: openpyxl >= 3.0.0, xlrd >= 2.0.0
- Configuration and progress bars: PyYAML >= 6.0.3, tqdm >= 4.60.0

## Optional Dependencies

For Jupyter notebooks:
```bash
pip install jupyter notebook
```

For R comparison:
```bash
pip install rpy2
```
