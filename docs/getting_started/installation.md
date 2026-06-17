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

Core packages:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0

## Optional Dependencies

For Jupyter notebooks:
```bash
pip install jupyter notebook
```

For R comparison:
```bash
pip install rpy2
```
