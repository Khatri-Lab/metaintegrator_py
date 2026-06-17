# COCONUT 5-Minute Quick Start

Get running with COCONUT in 5 minutes using the simplest possible workflow.

---

## ⚡ The One-Command Workflow

### Installation

```bash
git clone https://github.com/Khatri-Lab/metaintegrator_py.git
```

### Usage

```python
from coconut.coconut_runner import runner

runner(csv_dir='./my_studies', output_dir='./results')
```

**Done!** You now have normalized gene expression data ready for analysis.

---

## 📁 What You Need

### File Structure

Your CSV files should follow this pattern:

```
my_studies/
├── Study1_genes.csv       # Gene expression (genes × samples)
├── Study1_pheno.csv       # Sample metadata
├── Study2_genes.csv
├── Study2_pheno.csv
└── ...
```

### File Format Requirements

**genes.csv:**
```
         Sample1  Sample2  Sample3  ...
Gene1    5.23     6.45     4.87    ...
Gene2    7.12     5.98     8.34    ...
...
```
- Rows = genes
- Columns = samples
- First column = gene names

**pheno.csv:**
```
         control  Age  Sex  ...
Sample1  0        45   M    ...
Sample2  1        32   F    ...
Sample3  0        56   M    ...
...
```
- Rows = samples (matching genes.csv columns)
- Columns = attributes
- **Must have** a control column (0=control, 1=disease)
- First column = sample IDs

---

## 🎯 The Simplest Workflow

### Option 1: One Command (Recommended)

```python
from coconut.coconut_runner import runner

# Process and save results
runner(csv_dir='./studies', output_dir='./results')
```

**Output files created:**
- `mcoco_combined_genes.csv` - Normalized expression (genes × samples)
- `mcoco_combined_genes_samples.csv` - Transposed (samples × genes)
- `mcoco_combined_pheno.csv` - Sample metadata
- `mcoco_combined_class_labels.csv` - Binary labels (0=control, 1=disease)

---

### Option 2: Step-by-Step (More Control)

```python
from coconut.coconut_runner import load_gse_data_from_csv_directory
from coconut.coconut import coconut, combine_coco_output

# 1. Load data
gse_data = load_gse_data_from_csv_directory('./studies')

# 2. Run COCONUT
result = coconut(gses=gse_data, control_0_col='control')

# 3. Combine results
combined = combine_coco_output(result)

# 4. Access results
genes = combined['genes']              # Expression matrix
pheno = combined['pheno']              # Metadata
labels = combined['class_cntl0_dis1'] # Binary labels

# 5. Export
genes.to_csv('genes_normalized.csv')
pheno.to_csv('pheno.csv')
```

---

## 🔧 Common Customizations

### Custom File Patterns

If your files don't follow `*_genes.csv` / `*_pheno.csv` pattern:

```python
from coconut.coconut_runner import load_gse_data_from_csv_directory

gse_data = load_gse_data_from_csv_directory(
    csv_directory='./studies',
    genes_pattern='.expression.csv',   # Your pattern
    pheno_pattern='.metadata.csv'      # Your pattern
)

# Then run COCONUT
from coconut.coconut import coconut, combine_coco_output

result = coconut(gses=gse_data, control_0_col='your_control_column')
combined = combine_coco_output(result)
```

---

### Different Control Column

If your control column has a different name:

```python
# First, check available columns
gse_data = load_gse_data_from_csv_directory('./studies')

for i, gse in enumerate(gse_data):
    print(f"Dataset {i} columns: {list(gse['pheno'].columns)}")

# Then use the correct column name
result = coconut(gses=gse_data, control_0_col='Healthy0.Disease1')
```

---

### Multi-Platform Data

If combining different sequencing platforms (Affymetrix, Illumina, etc.):

```python
result = coconut(
    gses=gse_data,
    control_0_col='control',
    by_platform=True,           # Enable platform-specific
    platform_col='Platform_id'  # Platform column name
)
```

---

## ✅ Verify Your Results

### Quick PCA Check

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Run PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(combined['genes'].T)

# Plot
plt.scatter(coords[:, 0], coords[:, 1], c=combined['class_cntl0_dis1'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA after COCONUT')
plt.colorbar(label='Class (0=control, 1=disease)')
plt.show()
```

**Expected:** Controls and disease samples separated, not clustered by study.

---

## 🐛 Quick Troubleshooting

### Error: "No CSV files found"

**Solution:** Check your directory path
```python
from pathlib import Path

# Verify path exists
print(Path('./studies').exists())

# List CSV files
print(list(Path('./studies').glob('*.csv')))
```

---

### Error: "No common genes found"

**Solution:** Standardize gene names
```python
# Before COCONUT
for gse in gse_data:
    gse['genes'].index = gse['genes'].index.str.upper()
```

---

### Error: "Dataset has < 1 control"

**Solution:** Verify control column values
```python
for i, gse in enumerate(gse_data):
    print(f"Dataset {i}:")
    print(gse['pheno']['control'].value_counts())
    # Should show: 0 (controls), 1 (disease)
```

---

## 📊 Quick Reference

### Essential Parameters

| Parameter | Required? | Default | Description |
|-----------|-----------|---------|-------------|
| `csv_dir` | ✅ Yes | - | Directory with CSV files |
| `output_dir` | No | `None` | Where to save results |
| `control_0_col` | ✅ Yes | - | Control column name |
| `by_platform` | No | `False` | Platform-specific normalization |
| `platform_col` | Conditional | `None` | Required if `by_platform=True` |

### Quick Commands

```python
# Load data
from coconut.coconut_runner import load_gse_data_from_csv_directory
gse_data = load_gse_data_from_csv_directory('./studies')

# Run COCONUT
from coconut.coconut import coconut, combine_coco_output
result = coconut(gses=gse_data, control_0_col='control')

# Combine results
combined = combine_coco_output(result)

# Export
combined['genes'].to_csv('genes.csv')
combined['pheno'].to_csv('pheno.csv')
```

---

## 🚀 Next Steps

### Intermediate Users

Read the **[Workflow Guide](workflow.md)** for:
- Detailed step-by-step tutorial
- Quality assessment methods
- Visualization examples
- Advanced parameters

### R Users

Read the **[R to Python Guide](r_to_python.md)** for:
- Side-by-side syntax comparisons
- Python advantages for CSV workflows
- Data exchange between R and Python

### Production Use

Read the **[Best Practices Guide](best_practices.md)** for:
- Data validation techniques
- Parameter selection guidelines
- Quality assessment workflows
- Production patterns

### Having Issues?

Read the **[Troubleshooting Guide](troubleshooting.md)** for:
- Complete error solutions
- Diagnostic procedures
- Debug mode usage
- Common pitfalls

---

## 💡 Pro Tips

### Enable Debug Mode

See detailed progress messages:

```python
from coconut import coconut_runner

coconut_runner.debug_mode = True

# Now run with verbose output
from coconut.coconut_runner import runner
runner(csv_dir='./studies', output_dir='./results')
```

### Validate First, Save Later

```python
# Quick validation (no files saved)
runner(csv_dir='./studies')  # Check for issues

# If looks good, save results
runner(csv_dir='./studies', output_dir='./results')
```

### Check Gene Overlap

```python
from coconut.coconut import _get_common_genes

common = _get_common_genes(gse_data)
print(f"Common genes: {len(common)}")
# Minimum 100 recommended
```

---

## 📖 Complete Example

```python
from coconut.coconut_runner import runner

# 1. Run COCONUT (one command)
runner(csv_dir='./sepsis_studies', output_dir='./results')

# 2. Load and visualize results
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

genes = pd.read_csv('./results/mcoco_combined_genes.csv', index_col=0)
labels = pd.read_csv('./results/mcoco_combined_class_labels.csv')

# 3. PCA visualization
pca = PCA(n_components=2)
coords = pca.fit_transform(genes.T)

plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], c=labels['class_label'], 
            cmap='RdYlBu_r', s=50, alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Batch-Corrected Gene Expression')
plt.colorbar(label='Class (0=control, 1=disease)')
plt.tight_layout()
plt.savefig('pca_result.png', dpi=300)
plt.show()

print("✅ Analysis complete!")
```

---

## 📋 Checklist

Before running COCONUT:

- [ ] CSV files in correct directory
- [ ] Files follow naming pattern (*_genes.csv, *_pheno.csv)
- [ ] Sample names match between genes and pheno
- [ ] Control column exists (0=control, 1=disease)
- [ ] At least 2 studies/datasets
- [ ] Each study has control samples

Ready? Run this:

```python
from coconut.coconut_runner import runner
runner(csv_dir='./your_data', output_dir='./results')
```

---

**That's it!** You now have batch-corrected gene expression data ready for:
- Meta-analysis
- Machine learning
- Differential expression
- Biomarker discovery

**Next:** [Workflow Guide](workflow.md) for detailed tutorial

---

*Last updated: 2026-04-24*  
*Package: metaintegrator_py v0.1.0*
