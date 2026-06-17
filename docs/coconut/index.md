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

## 📖 Learning Paths

### Path 1: Complete Beginners

1. **Read** [Quick Start](quickstart.md) (5 min)
2. **Run** the example code
3. **Read** [Workflow Guide](workflow.md) (15 min)
4. **Apply** to your own data
5. **Reference** [Best Practices](best_practices.md) as needed

**Total time:** ~30 minutes to first results

---

### Path 2: R Users Migrating

1. **Read** [R to Python Guide](r_to_python.md) (10 min)
   - See syntax differences
   - Discover Python advantages
2. **Read** [Runner Guide](runner_guide.md) (15 min)
   - Learn the one-command workflow
   - See CSV loading simplifications
3. **Run** example code
4. **Reference** [API Reference](api.md) for specifics

**Total time:** ~25 minutes to migration

---

### Path 3: Experienced Python Users

1. **Skim** [Quick Start](quickstart.md) (3 min)
2. **Run** example code
3. **Reference** [API Reference](api.md) as needed
4. **Consult** [Best Practices](best_practices.md) for production

**Total time:** ~10 minutes to first results

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

# After COCONUT normalization
analyzer = MetaAnalyzer()
meta_results = analyzer.run_meta_analysis_core([combined])
```

### Visualization
```python
from metaintegrator.visualization import plot_pca

# Visualize COCONUT results
plot_pca(combined['genes'], combined['class_cntl0_dis1'])
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
- 🐛 [Report bugs on GitHub](https://github.com/yourusername/metaintegrator_py/issues)
- 💬 [Discussions](https://github.com/yourusername/metaintegrator_py/discussions)
- 📧 Email: support@metaintegrator.org

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

## 🗺️ Site Map

### Getting Started
- [Quick Start](quickstart.md) ⚡
- [Workflow Guide](workflow.md) 📖
- [R to Python Guide](r_to_python.md) 🔄

### Reference
- [API Reference](api.md) 📚
- [Runner Guide](runner_guide.md) 🏃
- [Best Practices](best_practices.md) 💡
- [Troubleshooting](troubleshooting.md) 🔧

### Examples
- One-command workflow → [Quick Start](quickstart.md)
- Step-by-step tutorial → [Workflow Guide](workflow.md)
- CSV loading → [Runner Guide](runner_guide.md)
- Production patterns → [Best Practices](best_practices.md)

---

**Ready to get started?** 🚀

```python
from coconut.coconut_runner import runner

runner(csv_dir='./your_data', output_dir='./results')
```

**Next:** [Quick Start Guide →](quickstart.md)

---

*Last updated: 2026-04-24*  
*Package: metaintegrator_py v0.1.0*
