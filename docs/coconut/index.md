# COCONUT Documentation

**Batch Effect Normalization for Multi-Study Gene Expression Analysis**

---

## 🥥 Welcome to COCONUT Python

COCONUT (COmbat CONormalization Using conTrols) is a specialized batch normalization algorithm that removes technical batch effects while preserving biological signals in multi-study gene expression data.

### What Makes COCONUT Different?

Unlike standard ComBat, COCONUT uses **only control samples** to estimate batch effects, preventing over-correction that can mask real biological differences in disease samples.

---

## 🚀 Quick Start

Get started in 5 minutes with the one-command workflow:

```python
from coconut.coconut_runner import runner

runner(csv_dir='./my_studies', output_dir='./results')
```

**That's it!** Creates 4 CSV files ready for downstream analysis.

👉 [**Full Quick Start Guide**](quickstart.md)

Prefer the command line over a Python script? The same pipeline is available
as `python -m src coconut_batch_effect_removal`: see the
[Command-Line Interface guide](../getting_started/cli.md).

---

## 📚 Documentation Structure

### For New Users

| Document | Purpose | Time |
|----------|---------|------|
| [**Quick Start**](quickstart.md) | Get running in 5 minutes | ⏱️ 5 min |
| [**Workflow Guide**](workflow.md) | Step-by-step tutorial | 📖 15 min |
| [**Best Practices**](best_practices.md) | Production tips | 💡 10 min |

### For R Users

| Document | Purpose | Time |
|----------|---------|------|
| [**R to Python Guide**](r_to_python.md) | Side-by-side comparisons | 🔄 10 min |
| [**Runner Guide**](runner_guide.md) | Python-exclusive features | ⭐ 15 min |

### Reference Materials

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [**API Reference**](api.md) | Complete function docs | Looking up parameters |
| [**Troubleshooting**](troubleshooting.md) | Error solutions | When stuck |

---

## 🎯 Common Use Cases

### Use Case 1: Load CSV Data and Normalize

```python
from coconut.coconut_runner import runner

# One command does everything
runner(csv_dir='./studies', output_dir='./results')
```

**What it does:**
- ✅ Loads all CSV files
- ✅ Validates data structure
- ✅ Runs COCONUT normalization
- ✅ Combines results
- ✅ Exports 4 CSV files

**Learn more:** [Runner Guide](runner_guide.md)

---

### Use Case 2: Multi-Platform Data

```python
from coconut.coconut import coconut, combine_coco_output

result = coconut(
    gses=gse_data,
    control_0_col='control',
    by_platform=True,           # Platform-specific normalization
    platform_col='Platform_id'
)

combined = combine_coco_output(result)
```

**When to use:** Combining Affymetrix, Illumina, RNA-seq, etc.

**Learn more:** [Workflow Guide](workflow.md#workflow-b-platform-specific)

---

### Use Case 3: Machine Learning Pipeline

```python
from coconut.coconut import coconut, combine_coco_output
from sklearn.ensemble import RandomForestClassifier

# Normalize
result = coconut(gses=gse_data, control_0_col='control')
combined = combine_coco_output(result)

# Train classifier
X = combined['genes'].T.values  # Samples × genes
y = combined['class_cntl0_dis1']

clf = RandomForestClassifier()
clf.fit(X, y)
```

**Learn more:** [Best Practices - ML Integration](best_practices.md#pattern-3-machine-learning-pipeline)

---

## 🔧 Python Advantages Over R

COCONUT Python offers **significant improvements** for CSV-based workflows:

| Feature | R Version | Python Version |
|---------|-----------|----------------|
| **CSV Loading** | Manual, 20+ lines | Automatic, 3 lines |
| **Sample Alignment** | Manual checking | Auto-aligned |
| **One-Command Pipeline** | ❌ Not available | ✅ `runner()` function |
| **Debug Mode** | ❌ Not available | ✅ Detailed progress |
| **Error Messages** | Generic | Specific, actionable |

**See comparison:** [R to Python Guide](r_to_python.md)

---

## 🎓 Core Concepts

### Two-Phase Normalization

**Phase 1 - GET (Controls Only):**
```
Controls from Study 1 ──┐
Controls from Study 2 ──┼─→ Estimate batch effects
Controls from Study 3 ──┘
```

**Phase 2 - HAVE (Apply to Disease):**
```
Disease samples ──→ Apply corrections ──→ Normalized data
   (preserve biology)
```

**Why this matters:** Disease signals are preserved because batch correction is learned from controls only.

**Learn more:** [Workflow Guide - Understanding COCONUT](workflow.md#step-3-run-coconut)

---

### Data Structure

COCONUT expects this format:

```python
gse_data = [
    {
        'genes': pd.DataFrame,  # genes (rows) × samples (columns)
        'pheno': pd.DataFrame   # samples (rows) × attributes (columns)
    },
    # ... more studies
]
```

**Control column requirement:** Must have 0 for controls, 1 for disease

**Learn more:** [API Reference - Data Structures](api.md#type-definitions)

---

## 📊 Expected Results

### Quality Indicators

After successful COCONUT normalization:

✅ **PCA Plot:**
- Before: Samples cluster by study/batch
- After: Samples cluster by biology (control vs disease)

✅ **Expression Distributions:**
- Before: Different distributions per study
- After: Similar/overlapping distributions

✅ **Sample Correlations:**
- Before: High correlation within batches
- After: Correlation based on biological similarity

**Learn more:** [Best Practices - Quality Assessment](best_practices.md#quality-assessment)

---

## 🛠️ Installation

```bash
# install from source
git clone https://github.com/Khatri-Lab/metaintegrator_py.git
cd metaintegrator_py
pip install -e .
```

**Verify installation:**
```python
from coconut.coconut import coconut
from coconut.coconut_runner import runner

print("✅ COCONUT ready!")
```

---

## 💡 Common Questions

### Q: When should I use COCONUT vs regular ComBat?

**Use COCONUT when:**
- ✅ Combining studies with different disease states
- ✅ You want to preserve disease signals
- ✅ You have control samples in each study

**Use regular ComBat when:**
- ✅ All samples similar biology
- ✅ Just removing technical batch effects
- ✅ No disease/control distinction needed

---

### Q: How many studies do I need?

**Minimum:** 2 studies (batches)

**Recommended:** 3+ studies for robust batch correction

**Each study should have:** At least 3-5 control samples

---

### Q: Can I use COCONUT with different platforms?

**Yes!** Enable platform-specific normalization:

```python
result = coconut(
    gses=gse_data,
    control_0_col='control',
    by_platform=True,
    platform_col='Platform_id'
)
```

**Learn more:** [API Reference - coconut()](api.md#coconut)

---

### Q: What if my gene names don't match across studies?

**Solution:** Standardize gene identifiers before COCONUT

```python
# Example: Convert to uppercase
for gse in gse_data:
    gse['genes'].index = gse['genes'].index.str.upper()
```

**Learn more:** [Troubleshooting - No Common Genes](troubleshooting.md#error-no-common-genes-found)

---

## 🔗 Related Tools

COCONUT works seamlessly with other metaintegrator_py modules:

### Meta-Analysis
```python
from metaintegrator.analysis import MetaAnalyzer

# After COCONUT normalization, wrap the combined output as a
# study dict (expr/class/keys are required by run_meta_analysis_core)
meta_input = {
    'expr': combined['genes'],
    'class': combined['class_cntl0_dis1'],
    'keys': combined['genes'].index.tolist(),
}
analyzer = MetaAnalyzer()
meta_results = analyzer.run_meta_analysis_core([meta_input])
```

### Visualization
PCA plotting isn't part of the package's public API: build one directly with
scikit-learn on the COCONUT-normalized matrix:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
coords = pca.fit_transform(combined['genes'].T)  # samples x components

plt.scatter(coords[:, 0], coords[:, 1], c=combined['class_cntl0_dis1'])
plt.xlabel("PC1")
plt.ylabel("PC2")
```

---

## 📞 Getting Help

### Documentation
- 📖 [Quick Start](quickstart.md) - Fast track to results
- 📖 [Workflow Guide](workflow.md) - Detailed tutorial
- 📖 [API Reference](api.md) - Complete function docs
- 📖 [Troubleshooting](troubleshooting.md) - Error solutions
- 📖 [Best Practices](best_practices.md) - Production tips

### Community
- 🐛 [Report bugs on GitHub](https://github.com/Khatri-Lab/metaintegrator_py/issues)
- 💬 [Discussions](https://github.com/Khatri-Lab/metaintegrator_py/discussions)
- 📧 Email: khatrilab-support@stanford.edu

### Debug Mode
```python
from coconut import coconut_runner
coconut_runner.debug_mode = True

# Now see detailed progress
runner(csv_dir='./data', output_dir='./results')
```

---

## 📝 Citation

If you use COCONUT in your research, please cite:

**COCONUT Algorithm:**
> Sweeney TE, et al. (2016). Robust classification of bacterial and viral infections via integrated host gene expression diagnostics. Science Translational Medicine. 8(346):346ra91.

**MetaIntegrator Python Package:**
> [Citation for this package - update when published]

---

## 📄 License

This package is licensed under the MIT License. See [LICENSE](../about/license.md) for details.

---

**Ready to get started?** 🚀

```python
from coconut.coconut_runner import runner

runner(csv_dir='./your_data', output_dir='./results')
```

**Next:** [Quick Start Guide →](quickstart.md)
