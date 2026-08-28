# Contributing Guidelines

Thank you for your interest in contributing to **metaIntegrator-py**! This document provides guidelines to ensure a smooth and productive contribution process for everyone.

## Contents

- [Getting Started](#getting-started)
- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Coding Style](#coding-style)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Review Process](#review-process)
- [Recognition](#recognition)

---

## Getting Started

### First Time Contributors

Welcome! We're excited to have you. Here's how to get started:

1. ⭐ **Star the repository** to show your support
2. 🍴 **Fork the repository** to your GitHub account
3. 📖 **Read the documentation** to understand the project
4. 🔍 **Browse open issues** labeled [`good first issue`](https://github.com/Khatri-Lab/metaintegrator_py/labels/good%20first%20issue)
5. 💬 **Ask questions** in GitHub Discussions
6. 🚀 **Make your first contribution**

### What Can You Contribute?

We welcome all types of contributions:

- 🐛 **Bug Fixes** - Fix issues or improve stability
- ✨ **New Features** - Add new functionality
- 📚 **Documentation** - Improve or expand docs
- 🧪 **Tests** - Increase test coverage
- 🎨 **Examples** - Add usage examples
- 🔧 **Tools** - Improve development tools
- 🌐 **Translations** - Translate documentation
- 💡 **Ideas** - Suggest improvements

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for everyone, regardless of:

- Age, body size, disability, ethnicity, gender identity and expression
- Level of experience, education, socio-economic status
- Nationality, personal appearance, race, religion
- Sexual identity and orientation

### Our Standards

**Positive behaviors:**
- ✅ Using welcoming and inclusive language
- ✅ Being respectful of differing viewpoints
- ✅ Gracefully accepting constructive criticism
- ✅ Focusing on what's best for the community
- ✅ Showing empathy towards others

**Unacceptable behaviors:**
- ❌ Harassment, trolling, or insulting comments
- ❌ Public or private harassment
- ❌ Publishing others' private information
- ❌ Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team. All complaints will be reviewed and investigated promptly and fairly.

**Consequences** for violations:
1. **Warning** - First-time minor violations
2. **Temporary Ban** - Repeated or serious violations
3. **Permanent Ban** - Severe or persistent violations

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org/), version 2.1.

---

## How to Contribute

### Types of Contributions

#### 🐛 Bug Reports

Found a bug? Help us fix it:

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** using the bug report template
3. **Provide details**:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Code snippet to reproduce (if applicable)
   - Error messages or screenshots

#### ✨ Feature Requests

Have an idea? Share it:

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** using the feature request template
3. **Provide details**:
   - Clear description of the feature
   - Use case and motivation
   - Proposed implementation (if any)
   - Alternatives considered

#### 📚 Documentation Improvements

Documentation is crucial:

- Fix typos or unclear wording
- Add missing documentation
- Improve examples
- Translate documentation
- Update outdated content

#### 🧪 Code Contributions

Ready to code:

1. **Find or create an issue** to discuss the change
2. **Fork and clone** the repository
3. **Create a branch** for your changes
4. **Write code** following our standards
5. **Add tests** for your changes
6. **Update documentation** as needed
7. **Submit a pull request**

---

## Coding Style

### Python Style Guide

We follow **PEP 8** with the following specifications:

#### Line Length

```python
# Maximum line length: 100 characters (not 79)
def long_function_name(
    parameter_one: str,
    parameter_two: int,
    parameter_three: Optional[Dict[str, Any]] = None
) -> Tuple[str, int]:
    """Function with proper line wrapping."""
    pass
```

#### Imports

```python
# Standard library
import os
import sys
from typing import Dict, List, Optional

# Third-party
import numpy as np
import pandas as pd
from scipy import stats

# Local application
from metaintegrator.validation import DatasetValidator
from metaintegrator.analysis import MetaAnalyzer
```

**Import ordering:**
1. Standard library
2. Third-party packages
3. Local application imports

**Use isort to automate:**
```bash
isort src/ tests/
```

#### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| **Modules** | `lowercase_with_underscores` | `data_loading.py` |
| **Classes** | `CapitalizedWords` | `DatasetValidator` |
| **Functions** | `lowercase_with_underscores` | `calculate_effect_size()` |
| **Variables** | `lowercase_with_underscores` | `effect_size` |
| **Constants** | `UPPER_CASE_WITH_UNDERSCORES` | `MAX_ITERATIONS` |
| **Private** | `_leading_underscore` | `_internal_function()` |

#### Code Formatting

Use **Black** for consistent formatting:

```bash
# Format all code
black src/ tests/

# Check without modifying
black --check src/
```

**Configuration** (in `pyproject.toml`):
```toml
[tool.black]
line-length = 100
target-version = ['py310', 'py311', 'py312']
exclude = '''
/(
    \.git
  | \.venv
  | build
  | dist
)/
'''
```

#### Type Hints

**Always use type hints** for function signatures:

```python
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def calculate_pooled_effect(
    effect_sizes: pd.DataFrame,
    standard_errors: pd.DataFrame,
    method: str = "random"
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calculate pooled effect sizes.
    
    Args:
        effect_sizes: Effect size matrix (genes × studies)
        standard_errors: Standard error matrix (genes × studies)
        method: Pooling method ("random" or "fixed")
    
    Returns:
        Tuple containing:
            - Pooled results DataFrame
            - Summary statistics dictionary
    """
    pass
```

#### Docstrings

**All public functions, classes, and modules** must have docstrings:

```python
def validate_meta_object(
    meta_object: Dict,
    stage: str = "Pre-Analysis"
) -> Tuple[bool, List[str]]:
    """
    Validate meta-analysis object structure.
    
    Comprehensive validation of meta-object including data structure
    checks, dimension consistency, and required field verification.
    
    Args:
        meta_object: Dictionary containing originalData and optionally
            metaAnalysis results
        stage: Analysis stage for context-specific validation.
            Options: "Pre-Analysis", "Post-Analysis"
    
    Returns:
        Tuple containing:
            - is_valid: True if all checks pass
            - errors: List of error messages (empty if valid)
    
    Raises:
        TypeError: If meta_object is not a dictionary
    
    Examples:
        >>> meta_obj = {"originalData": {...}}
        >>> is_valid, errors = validate_meta_object(meta_obj)
        >>> if not is_valid:
        ...     print("Errors:", errors)
    
    See Also:
        - DatasetValidator.validate_study_structure
        - checkDataObject (R equivalent)
    """
    pass
```

---

## Testing Requirements

### Test Coverage

- **Minimum**: 80% overall coverage
- **New code**: 100% coverage required
- **Critical modules**: 95%+ coverage

### Writing Tests

#### Test Structure

```python
# tests/metaintegrator/test_validation.py
import pytest
import numpy as np
import pandas as pd
from metaintegrator.validation import DatasetValidator


class TestDatasetValidator:
    """Test suite for DatasetValidator class."""
    
    @pytest.fixture
    def valid_dataset(self):
        """Create valid dataset for testing."""
        return {
            "expr": pd.DataFrame(np.random.randn(100, 10)),
            "class": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            "keys": [f"GENE_{i}" for i in range(100)]
        }
    
    def test_valid_dataset_passes(self, valid_dataset):
        """Test that valid dataset passes validation."""
        is_valid = DatasetValidator.check_class_vector(valid_dataset)
        assert is_valid is True
    
    def test_insufficient_cases_fails(self):
        """Test that dataset with insufficient cases fails."""
        dataset = {
            "expr": pd.DataFrame(np.random.randn(100, 3)),
            "class": np.array([0, 0, 1]),  # Only 1 case
            "keys": [f"GENE_{i}" for i in range(100)]
        }
        is_valid = DatasetValidator.check_class_vector(dataset)
        assert is_valid is False
    
    def test_missing_class_vector_fails(self):
        """Test that dataset without class vector fails."""
        dataset = {
            "expr": pd.DataFrame(np.random.randn(100, 10)),
            "keys": [f"GENE_{i}" for i in range(100)]
            # Missing "class"
        }
        is_valid = DatasetValidator.check_class_vector(dataset)
        assert is_valid is False
    
    @pytest.mark.parametrize("n_cases,n_controls,expected", [
        (0, 5, False),  # No cases
        (1, 5, False),  # Only 1 case
        (2, 1, False),  # Only 1 control
        (2, 2, True),   # Minimum valid
        (10, 10, True), # Normal case
    ])
    def test_case_control_requirements(self, n_cases, n_controls, expected):
        """Test case/control count requirements."""
        dataset = {
            "expr": pd.DataFrame(np.random.randn(100, n_cases + n_controls)),
            "class": np.array([1] * n_cases + [0] * n_controls),
            "keys": [f"GENE_{i}" for i in range(100)]
        }
        assert DatasetValidator.check_class_vector(dataset) == expected
```

#### Test Naming

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<specific_behavior>`

**Examples:**
- ✅ `test_valid_input_succeeds()`
- ✅ `test_empty_input_raises_error()`
- ✅ `test_invalid_type_raises_typeerror()`
- ❌ `test_function()` (too vague)

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/metaintegrator/test_validation.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run and show print statements
pytest -s

# Run and stop at first failure
pytest -x
```

---

## Documentation Standards

### Docstring Format

Use **Google-style docstrings**:

#### Sections

1. **Summary** - One-line description
2. **Extended Description** - Optional detailed explanation
3. **Args** - Parameter descriptions
4. **Returns** - Return value description
5. **Raises** - Exceptions that may be raised
6. **Examples** - Usage examples
7. **Notes** - Additional information
8. **See Also** - Related functions
9. **References** - Citations

#### Example

```python
def calculate_meta_analysis(
    datasets: List[Dict],
    method: str = "random",
    min_studies: int = 2
) -> Dict[str, pd.DataFrame]:
    """
    Perform meta-analysis on multiple datasets.
    
    Calculates effect sizes for each dataset, pools results using
    random or fixed effects models, and computes heterogeneity
    statistics.
    
    Args:
        datasets: List of dataset dictionaries, each containing:
            - expr: Expression matrix (genes × samples)
            - class: Binary phenotype vector
            - keys: Gene identifiers
        method: Meta-analysis method:
            - "random": Random effects (DerSimonian-Laird)
            - "fixed": Fixed effects (inverse variance)
        min_studies: Minimum number of studies required for pooling
    
    Returns:
        Dictionary containing:
            - pooledResults: Main results DataFrame with columns:
                - effectSize: Pooled effect size
                - effectSizePval: P-value
                - effectSizeFDR: FDR-adjusted p-value
            - datasetEffectSizes: Individual study effect sizes
            - datasetEffectSizeStandardErrors: Standard errors
    
    Raises:
        ValueError: If datasets list is empty
        ValueError: If method is not "random" or "fixed"
        ValueError: If fewer than min_studies datasets provided
    
    Examples:
        >>> datasets = [study1, study2, study3]
        >>> results = calculate_meta_analysis(datasets, method="random")
        >>> top_genes = results["pooledResults"].head(10)
        >>> print(top_genes)
    
    Notes:
        - Genes must overlap across studies
        - Missing values are handled automatically
        - Heterogeneity is assessed using Cochran's Q
    
    See Also:
        - MetaAnalysisCore.calculate_effect_sizes
        - MetaAnalysisCore.pool_inverse_variance
    
    References:
        DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical
        trials. Controlled Clinical Trials, 7(3), 177-188.
    """
    pass
```

### Markdown Documentation

#### File Structure

```markdown
# Page Title

Brief introduction paragraph.

## Contents

- [Section 1](#section-1)
- [Section 2](#section-2)

## Section 1

Content for section 1.

### Subsection 1.1

More specific content.

## Section 2

Content for section 2.
```

#### Code Blocks

````markdown
```python
# Always specify language
import numpy as np

result = np.array([1, 2, 3])
```
````

#### Links

```markdown
<!-- Internal links -->
See [API Reference](../api/overview.md) for details.

<!-- External links -->
Learn more at [NumPy Documentation](https://numpy.org/doc/).

<!-- Section links -->
See [Installation](#installation) above.
```

---

## Pull Request Process

### Before Submitting

**Checklist:**
- [ ] Code follows our style guide
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Branch is up-to-date with main

### Creating a Pull Request

1. **Push your branch** to your fork:
   ```bash
   git push origin feat/your-feature
   ```

2. **Create PR** on GitHub:
   - Click "New Pull Request"
   - Select your branch
   - Fill out the template

3. **PR Title Format**:
   ```
   <type>: <short description>
   ```
   
   Examples:
   - `feat: add GEO data caching`
   - `fix: correct effect size calculation`
   - `docs: update installation guide`

4. **PR Description** should include:
   - **What**: What changes were made
   - **Why**: Motivation for changes
   - **How**: How the changes work
   - **Testing**: How you tested the changes
   - **Issues**: Related issue numbers (e.g., "Fixes #123")

### PR Template

```markdown
## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?

Describe the tests you ran and how to reproduce.

- [ ] Test A
- [ ] Test B

## Checklist

- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where needed
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing unit tests pass locally
- [ ] Any dependent changes have been merged

## Related Issues

Fixes #(issue number)
```

### Review Process

1. **Automated Checks**: CI/CD runs automatically
   - Linting (black, flake8, isort)
   - Type checking (mypy)
   - Tests (pytest)
   - Coverage check

2. **Code Review**: Maintainer reviews code
   - At least 1 approval required
   - May request changes
   - Discussion in PR comments

3. **Revisions**: Address feedback
   ```bash
   # Make changes
   git add .
   git commit -m "fix: address review comments"
   git push origin feat/your-feature
   ```

4. **Merge**: Once approved
   - Squash and merge (preferred)
   - Regular merge (for multi-commit PRs)

### After Merge

- 🎉 Celebrate your contribution!
- 🔄 Sync your fork
- 🗑️ Delete your branch
- 📊 Check the CHANGELOG

---

## Issue Reporting

### Bug Reports

Use the bug report template and include:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Load dataset '...'
2. Run function '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Code snippet**
```python
# Minimal reproducible example
import metaintegrator_py

# Your code here
```

**Error message**
```
Full error traceback
```

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.12]
- Package version: [e.g., 0.1.0]
- NumPy version: [e.g., 1.24.3]
```

### Feature Requests

Include:

1. **Problem**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: What alternatives have you considered?
4. **Additional Context**: Any other relevant information

---

## Review Process

### What We Look For

#### Code Quality
- ✅ Follows style guide
- ✅ Well-documented
- ✅ Properly tested
- ✅ Type hints included
- ✅ No unnecessary complexity

#### Testing
- ✅ Tests pass
- ✅ Coverage maintained/improved
- ✅ Edge cases covered
- ✅ Integration tests if needed

#### Documentation
- ✅ Docstrings complete
- ✅ User docs updated
- ✅ Examples provided
- ✅ Comments for complex logic

### Feedback

Feedback may include:
- ✏️ **Suggestions**: Improvements to consider
- ❗ **Requests**: Changes required for approval
- 💡 **Questions**: Clarifications needed
- ✅ **Approval**: Ready to merge

---

## Recognition

### Contributors

All contributors are recognized in:

- **README.md** - Contributors section
- **CHANGELOG.md** - Release notes
- **GitHub Contributors** - Automatic tracking

### Levels of Contribution

- 🌱 **First-time Contributor**: Your first PR merged
- 🌿 **Regular Contributor**: 5+ PRs merged
- 🌳 **Core Contributor**: 20+ PRs or major features
- 👑 **Maintainer**: Invited based on sustained contributions

---

## Questions?

### Resources

- 📖 **Documentation**: [docs/ in the repository](https://github.com/Khatri-Lab/metaintegrator_py/tree/main/docs)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Khatri-Lab/metaintegrator_py/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Khatri-Lab/metaintegrator_py/issues)
- 📧 **Email**: khatrilab-support@stanford.edu

### Getting Help

1. **Search existing issues/discussions** - Your question may already be answered
2. **Ask in Discussions** - For general questions
3. **Open an Issue** - For specific problems
4. **Contact maintainers** - For sensitive matters

---

## Summary

**Key Points:**
- ✅ Follow the Code of Conduct
- ✅ Use our coding style and conventions
- ✅ Write tests for all changes
- ✅ Document your code thoroughly
- ✅ Submit clear, focused pull requests
- ✅ Be patient and respectful during review

**We appreciate your contributions! 🙏**

Every contribution, no matter how small, makes metaIntegrator-py better for everyone. Thank you for being part of our community!

---

## Related Documents

- [Development Guide](development.md) - Technical setup and workflow
- [License](../about/license.md) - Project license
- [Changelog](../about/changelog.md) - Version history
