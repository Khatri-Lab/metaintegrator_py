# Development Guide

Welcome to the **metaIntegrator-py** development guide! This document will help you set up your development environment, understand our development workflow, and contribute effectively to the project.

## Contents

- [Why Contribute?](#why-contribute)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Quality](#code-quality)
- [Debugging Tips](#debugging-tips)
- [Release Process](#release-process)

---

## Why Contribute?

Contributing to metaIntegrator-py helps:

- 🧬 **Advance Bioinformatics** - Improve meta-analysis tools for the research community
- 🎓 **Learn & Grow** - Gain experience with modern Python development practices
- 🤝 **Collaborate** - Work with bioinformatics and data science professionals
- 📈 **Build Portfolio** - Contribute to a production-quality scientific package
- 🌟 **Make Impact** - Your code will be used by researchers worldwide

We welcome contributors of all experience levels! Whether you're fixing a typo, adding a feature, or improving documentation, every contribution matters.

---

## Getting Started

### Prerequisites

- **Python**: 3.10 or later
- **Git**: For version control
- **GitHub Account**: To fork and submit pull requests
- **IDE**: VS Code, PyCharm, or your preferred editor

### Quick Start

```bash
# 1. Fork the repository on GitHub (click "Fork" button)

# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/metaintegrator_py.git
cd metaintegrator_py

# 3. Add upstream remote
git remote add upstream https://github.com/Khatri-Lab/metaintegrator_py.git

# 4. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 5. Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"

# 6. Install pre-commit hooks
pre-commit install

# 7. Verify installation
pytest tests/ -v
```

---

## Development Environment

### Recommended Setup

#### Virtual Environment

```bash
# Create isolated environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Development Dependencies

```bash
# Install package in editable mode with all extras
pip install -e ".[dev,test,docs]"

# Core dependencies
pip install numpy pandas scipy matplotlib seaborn

# Development tools
pip install black isort flake8 mypy pylint

# Testing tools
pip install pytest pytest-cov pytest-benchmark

# Documentation tools
pip install mkdocs mkdocs-material mkdocstrings[python]
```

#### Pre-commit Hooks

We use pre-commit to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually (optional)
pre-commit run --all-files
```

Example `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 26.1.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/pycqa/isort
    rev: 9.0.1
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.3.0
    hooks:
      - id: flake8
        args: ["--max-line-length=100", "--extend-ignore=E203,W503"]
```

---

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

### Key Directories

- **`src/`** - All source code
- **`tests/`** - Mirror structure of `src/` with test files
- **`docs/`** - MkDocs documentation

---

## Development Workflow

### 1. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feat/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description

# Or for documentation
git checkout -b docs/what-you-are-documenting
```

**Branch Naming Convention:**
- `feat/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/topic` - Documentation updates
- `refactor/component` - Code refactoring
- `test/test-description` - Test additions/improvements
- `perf/optimization-target` - Performance improvements

### 2. Make Changes

```bash
# Edit files in your favorite editor
code src/preprocessing/loader_factory.py

# Run tests frequently
pytest tests/preprocessing/test_base_and_factory.py -v

# Check code style
black src/
isort src/
flake8 src/
```

### 3. Write Tests

**Every new feature or bug fix should include tests!**

```python
# tests/preprocessing/test_my_feature.py
import pytest
import numpy as np
from preprocessing.my_feature import MyNewFeature


class TestMyNewFeature:
    """Test suite for MyNewFeature."""
    
    def test_basic_functionality(self):
        """Test basic functionality works as expected."""
        result = MyNewFeature.do_something([1, 2, 3])
        assert result == expected_value
    
    def test_edge_case_empty_input(self):
        """Test handling of empty input."""
        with pytest.raises(ValueError, match="Input cannot be empty"):
            MyNewFeature.do_something([])
    
    def test_invalid_input_type(self):
        """Test type validation."""
        with pytest.raises(TypeError):
            MyNewFeature.do_something("invalid")
```

### 4. Update Documentation

```python
def my_new_function(data: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Process data based on threshold.
    
    This function filters the input DataFrame based on the specified
    threshold value, keeping only rows that meet the criteria.
    
    Args:
        data: Input DataFrame to process
        threshold: Threshold value for filtering (default: 0.05)
    
    Returns:
        Filtered DataFrame
    
    Raises:
        ValueError: If threshold is not in range [0, 1]
    
    Examples:
        >>> data = pd.DataFrame({'value': [0.1, 0.01, 0.5]})
        >>> result = my_new_function(data, threshold=0.05)
        >>> len(result)
        2
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    
    return data[data['value'] <= threshold]
```

### 5. Commit Changes

```bash
# Stage changes
git add src/preprocessing/my_feature.py
git add tests/preprocessing/test_my_feature.py

# Commit with descriptive message
git commit -m "feat: add new data loading function

- Implement MyNewFeature class
- Add comprehensive tests
- Update documentation
- Fixes #123"
```

**Commit Message Format:**
```
<type>: <short summary>

<optional body>

<optional footer>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `style:` - Code style changes (formatting)
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### 6. Push and Create Pull Request

```bash
# Push to your fork
git push origin feat/your-feature-name

# Go to GitHub and create Pull Request
# Fill out the PR template with:
# - Description of changes
# - Related issue number
# - Testing performed
# - Screenshots (if applicable)
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/metaintegrator/test_validation.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/metaintegrator/test_validation.py::TestDatasetValidator::test_validate_class_vector_invalid_values

# Run tests matching pattern
pytest -k "test_effect_size"

# Run only failed tests from last run
pytest --lf

# Stop at first failure
pytest -x
```

### Test Organization

```python
# tests/metaintegrator/test_validation.py
import pytest
import numpy as np
import pandas as pd
from metaintegrator.validation import DatasetValidator


class TestDatasetValidator:
    """Test suite for DatasetValidator class."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        return {
            "expr": pd.DataFrame(np.random.randn(100, 10)),
            "class": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            "keys": [f"GENE_{i}" for i in range(100)]
        }
    
    def test_valid_dataset(self, sample_dataset):
        """Test validation of valid dataset."""
        assert DatasetValidator.check_class_vector(sample_dataset)
    
    def test_insufficient_cases(self):
        """Test detection of insufficient cases."""
        dataset = {
            "expr": pd.DataFrame(np.random.randn(100, 3)),
            "class": np.array([0, 0, 1]),  # Only 1 case
            "keys": [f"GENE_{i}" for i in range(100)]
        }
        assert not DatasetValidator.check_class_vector(dataset)
```

### Coverage Requirements

- **Minimum**: 80% overall coverage
- **Target**: 90% overall coverage
- **Critical modules**: 95%+ coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## Documentation

### Docstring Format

We use **Google-style docstrings**:

```python
def calculate_effect_size(
    expression: np.ndarray,
    groups: np.ndarray,
    method: str = "hedges_g"
) -> Dict[str, float]:
    """
    Calculate effect size for gene expression data.
    
    Computes Hedge's g or Cohen's d effect size for comparing two groups
    based on gene expression values. Includes small sample size correction
    for Hedge's g.
    
    Args:
        expression: Gene expression values (1D array)
        groups: Binary group labels (0=control, 1=case)
        method: Effect size method, either "hedges_g" or "cohen_d"
            (default: "hedges_g")
    
    Returns:
        Dictionary containing:
            - g: Effect size value
            - se_g: Standard error of effect size
            - n1: Sample size of cases
            - n2: Sample size of controls
            - m1: Mean of cases
            - m2: Mean of controls
    
    Raises:
        ValueError: If expression and groups have different lengths
        ValueError: If method is not "hedges_g" or "cohen_d"
        ValueError: If groups contains values other than 0 and 1
    
    Examples:
        >>> expression = np.array([1.2, 1.5, 1.8, 2.1, 2.4])
        >>> groups = np.array([0, 0, 1, 1, 1])
        >>> result = calculate_effect_size(expression, groups)
        >>> result['g']
        1.234
    
    Notes:
        - Hedge's g includes small sample correction
        - Minimum 2 samples per group required
        - NaN values are automatically removed
    
    References:
        Hedges, L. V. (1981). Distribution theory for Glass's estimator
        of effect size and related estimators. Journal of Educational
        Statistics, 6(2), 107-128.
    """
    # Implementation
    pass
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Serve documentation locally
mkdocs serve

# Open browser to http://127.0.0.1:8000

# Build static site
mkdocs build

# Deploy to GitHub Pages (maintainers only)
mkdocs gh-deploy
```

### Adding New Documentation

1. Create Markdown file in `docs/` directory
2. Add to `mkdocs.yml` navigation
3. Use relative links for cross-references
4. Include code examples
5. Add to appropriate section

```yaml
# mkdocs.yml
nav:
  - Home: index.md
  - API Reference:
      - Overview: api/overview.md
      - Your New Page: api/your_page.md  # Add here
```

---

## Code Quality

### Code Style

We follow **PEP 8** with some modifications:

```python
# Configuration in pyproject.toml
[tool.black]
line-length = 100
target-version = ['py310', 'py311', 'py312']

[tool.isort]
profile = "black"
line_length = 100
```

flake8 doesn't read its configuration from `pyproject.toml` without extra
tooling this project doesn't use. There's currently no `setup.cfg`, `.flake8`,
or `tox.ini` in the repo either, so flake8 runs with its defaults today; if
you want project-wide flake8 settings, add one of those files.

### Running Linters

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/

# All at once (pre-commit)
pre-commit run --all-files
```

### Type Hints

Always use type hints for function signatures:

```python
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd


def process_data(
    data: pd.DataFrame,
    threshold: float,
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Process data with type hints."""
    # Implementation
    pass
```

---

## Debugging Tips

### Using pdb

```python
# Insert breakpoint
import pdb; pdb.set_trace()

# Or in Python 3.7+
breakpoint()
```

### Using pytest with pdb

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger on first failure
pytest -x --pdb
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def my_function():
    logger.debug("Starting function")
    logger.info("Processing data")
    logger.warning("Potential issue detected")
    logger.error("Error occurred")
```

### Print Debugging

```python
# For quick debugging (remove before committing!)
print(f"DEBUG: variable = {variable}")
print(f"DEBUG: type = {type(variable)}, shape = {variable.shape}")
```

---

## Release Process

### Version Numbering

We follow **Semantic Versioning** (SemVer):

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Creating a Release (Maintainers Only)

```bash
# 1. Update version in pyproject.toml
vim pyproject.toml  # Change version = "1.2.3"

# 2. Update CHANGELOG.md
vim CHANGELOG.md

# 3. Commit changes
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 1.2.3"

# 4. Create tag
git tag -a v1.2.3 -m "Release version 1.2.3"

# 5. Push tag
git push origin v1.2.3

# 6. Build package
python -m build

# 7. Upload to PyPI
twine upload dist/*
```

---

## Getting Help

### Resources

- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions or discuss ideas
- **Documentation**: Check our comprehensive docs
- **Code of Conduct**: Review our community guidelines

### Need Help?

Don't hesitate to ask! We're here to help:

1. 💬 **Start a Discussion** on GitHub
2. 🐛 **Open an Issue** if you found a bug
3. 📧 **Contact Maintainers** for sensitive matters

### Community

- Be respectful and inclusive
- Help others learn
- Share your knowledge
- Celebrate contributions

---

## Quick Reference

### Common Commands

```bash
# Development
pip install -e ".[dev,test,docs]"
pytest -v
black src/ tests/
mkdocs serve

# Git workflow
git checkout -b feat/my-feature
git add .
git commit -m "feat: add new feature"
git push origin feat/my-feature

# Quality checks
pre-commit run --all-files
pytest --cov=src
flake8 src/
mypy src/
```

### File Templates

**New Module:**
```python
"""
Brief module description.

Detailed module description with usage examples.
"""

from typing import Dict, List
import numpy as np


class MyClass:
    """Class description."""
    
    def __init__(self):
        """Initialize MyClass."""
        pass
```

**New Test File:**
```python
"""Tests for my_module."""

import pytest
from my_module import MyClass


class TestMyClass:
    """Test suite for MyClass."""
    
    def test_basic(self):
        """Test basic functionality."""
        pass
```

---

## Next Steps

1. 📖 Read the [Contributing Guidelines](guidelines.md)
2. 🐛 Find a [good first issue](https://github.com/Khatri-Lab/metaintegrator_py/labels/good%20first%20issue)
3. 💬 Join the discussion on GitHub
4. 🚀 Start contributing!

**Thank you for contributing to metaIntegrator-py! 🎉**
