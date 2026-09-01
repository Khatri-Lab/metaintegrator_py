# COCONUT Runner - Complete User Guide

## 🎯 Purpose

`coconut_runner` is a high-level, user-friendly interface for the COCONUT batch normalization workflow. It provides convenient functions that automate common tasks, making COCONUT accessible to users who want to focus on their analysis rather than data wrangling.

---

## 🌟 Why Use coconut_runner?

### The Problem with Manual Workflows

Traditional COCONUT workflows require:
- Manually loading each CSV file
- Ensuring sample names align between gene and pheno files
- Handling missing values and data types
- Combining results after normalization
- Exporting in the correct format

This can be 20+ lines of repetitive code prone to errors.

### The coconut_runner Solution

```python
from coconut.coconut_runner import runner

runner(csv_dir='./my_data', output_dir='./results')
```

**That's it!** Three lines replace 20+ lines of manual code.

!!! tip "Command-line equivalent"
    Prefer not to write a script at all? The same pipeline is available from
    the shell as `python -m src coconut_batch_effect_removal`: see the
    [Command-Line Interface guide](../getting_started/cli.md).

---

## 📦 Available Functions

### 1. `runner()` - Complete End-to-End Pipeline ⭐ MOST POPULAR

**Purpose**: Run the entire COCONUT workflow with a single function call.

**Signature**:
```python
def runner(
    csv_dir: str,
    output_dir: str | None = None,
    control_0_col: str = "Healthy0.Sepsis1",
    disease_col: str | None = None,
    by_platform: bool = False,
    platform_col: str | None = "Platform_id",
    par_prior: bool = True,
    it_conv: float = 1e-04,
) -> None
```

**Parameters**:
- `csv_dir` (str, required): Path to directory containing CSV files
- `output_dir` (str, optional): Where to save output files. If None, only prints results.
- `control_0_col` (str, optional): Phenotype column indicating control samples (0 = control). Defaults to `"Healthy0.Sepsis1"`: the column name used by this project's own sample data. **Pass your own column name if your phenotype data uses a different one.**
- `disease_col` (str, optional): Phenotype column indicating disease samples. If `None`, all non-control samples are treated as disease.
- `by_platform` (bool, optional): Whether to normalize per platform. Requires `platform_col` when `True`.
- `platform_col` (str, optional): Phenotype column with platform info, used when `by_platform=True`. Defaults to `"Platform_id"`.
- `par_prior` (bool, optional): Whether to use parametric priors in the Bayesian batch-effect adjustment.
- `it_conv` (float, optional): Convergence threshold for the iterative solution.

**What it does**:
1. Loads all CSV files from directory
2. Validates data structure
3. Runs COCONUT normalization
4. Combines results across studies
5. Exports 4 CSV files (if output_dir specified)

**Output Files** (when output_dir is specified):
- `mcoco_combined_genes_samples.csv` - Gene expression (genes × samples)
- `mcoco_combined_genes.csv` - Transposed version (samples × genes)
- `mcoco_combined_pheno.csv` - Sample metadata
- `mcoco_combined_class_labels.csv` - Binary class labels (0=control, 1=disease)

**Example**:
```python
from coconut.coconut_runner import runner

# Process data and save results
runner(csv_dir='./sepsis_studies', output_dir='./coconut_results')

# Or just process without saving (for exploration)
runner(csv_dir='./sepsis_studies')
```

**Use when**:
- You have standard CSV files from R or other sources
- You want the quickest path from data to results
- You're running a production pipeline
- You're new to COCONUT and want simplicity

---

### 2. `load_gse_data_from_csv_directory()` - Smart CSV Loading

**Purpose**: Load and validate multi-study gene expression data from CSV files.

**Signature**:
```python
def load_gse_data_from_csv_directory(
    csv_directory: str,
    genes_pattern: str = "_genes.csv",
    pheno_pattern: str = "_pheno.csv",
    object_prefix: str = "GSEs.test_"
) -> List[Dict[str, pd.DataFrame]]
```

**Parameters**:
- `csv_directory` (str): Path to directory with CSV files
- `genes_pattern` (str): Pattern to identify gene expression files (default: "_genes.csv")
- `pheno_pattern` (str): Pattern to identify phenotype files (default: "_pheno.csv")
- `object_prefix` (str): Prefix to remove from dataset names (default: "GSEs.test_")

**What it does**:
1. Scans directory for matching CSV files
2. Groups files by dataset name
3. Loads gene expression and phenotype data
4. **Automatically aligns sample names** between genes and pheno
5. Validates that each dataset has both genes and pheno files
6. Reports any mismatches or missing samples

**Returns**:
```python
[
    {
        'genes': pd.DataFrame,  # genes (rows) × samples (columns)
        'pheno': pd.DataFrame   # samples (rows) × attributes (columns)
    },
    # ... more datasets
]
```

**Example**:
```python
from coconut.coconut_runner import load_gse_data_from_csv_directory

# Standard usage
gse_data = load_gse_data_from_csv_directory('./my_studies')

# Custom file patterns
gse_data = load_gse_data_from_csv_directory(
    csv_directory='./custom_data',
    genes_pattern='.expression.csv',
    pheno_pattern='.metadata.csv',
    object_prefix='Study_'
)
```

**File Structure Expected**:
```
my_studies/
├── GSEs.test_GSE28750_genes.csv
├── GSEs.test_GSE28750_pheno.csv
├── GSEs.test_GSE40012_genes.csv
├── GSEs.test_GSE40012_pheno.csv
└── ...
```

**Key Features**:
- ✅ Handles sample name mismatches (e.g., "Sample1" vs "Sample1 ")
- ✅ Reports detailed statistics per dataset
- ✅ Validates data integrity before processing
- ✅ Provides helpful error messages
- ✅ Works with any file naming convention (via pattern parameters)

**Use when**:
- You want to load CSV data for custom workflows
- You need more control than `runner()` provides
- You want to inspect data before normalization
- Your CSV files don't follow standard naming

---

### 3. `load_and_run_coconut_example()` - Interactive Learning Tool

**Purpose**: Provide an interactive workflow that helps users understand their data and COCONUT parameters.

**Signature**:
```python
def load_and_run_coconut_example(
    csv_directory: str,
    control_0_col: str = "Healthy0.Sepsis1",
    disease_col: str | None = None,
    by_platform: bool = False,
    platform_col: str | None = "Platform_id",
    par_prior: bool = True,
    it_conv: float = 1e-04,
) -> Dict
```

**Parameters**:
- `csv_directory` (str): Path to directory containing CSV files
- `control_0_col`, `disease_col`, `by_platform`, `platform_col`, `par_prior`, `it_conv`: Same meaning and defaults as in [`runner()`](#1-runner-complete-end-to-end-pipeline-most-popular) above: these are forwarded straight to `coconut()`.

**What it does**:
1. Loads data from CSV directory
2. **Shows available phenotype columns** for each dataset: useful for confirming your `control_0_col` / `disease_col` / `platform_col` values are actually present in your data
3. Runs COCONUT using the column names you passed (or the defaults, if you didn't)
4. Returns results for further analysis

**Example Output**:
```
🥥 COCONUT Pipeline Starting...

[loading prints a "📊 DATASET SUMMARY" block here -- genes/samples
 counts and key phenotype columns per dataset]

🔍 Inspecting phenotype columns...

Dataset 1 columns:
  Healthy0.Sepsis1: [0, 1]
  Platform_id: ['GPL570', 'GPL6947']
  Age: [25, 34, 42, 56, 63]...

Dataset 2 columns:
  Healthy0.Sepsis1: [0, 1]
  Platform_id: ['GPL10558']
  Age: [31, 28, 45, 52]...

🥥 Running COCONUT algorithm...
✅ COCONUT completed successfully!
```

**Example Usage**:
```python
from coconut.coconut_runner import load_and_run_coconut_example

# Interactive exploration
result = load_and_run_coconut_example('./my_data')

# Access results
coconut_list = result['COCONUTList']
control_list = result['controlList']
```

**Use when**:
- You're new to COCONUT and learning the workflow
- You're not sure which column indicates controls
- You want to understand your data structure
- You're teaching COCONUT to others

---

## 🎓 Complete Workflow Examples

### Example 1: Simplest Workflow (One Command)

```python
from coconut.coconut_runner import runner

# That's all you need!
runner(csv_dir='./studies', output_dir='./results')
```

**Output**:
- 4 CSV files in `./results/` directory
- Ready for downstream analysis

---

### Example 2: Step-by-Step with More Control

```python
from coconut.coconut_runner import load_gse_data_from_csv_directory
from coconut.coconut import coconut, combine_coco_output

# Step 1: Load data (with automatic sample alignment)
gse_data = load_gse_data_from_csv_directory('./studies')

# Step 2: Run COCONUT
result = coconut(
    gses=gse_data,
    control_0_col='Healthy0.Disease1',
    by_platform=False,
    par_prior=True
)

# Step 3: Combine results
combined = combine_coco_output(result)

# Step 4: Access results
genes = combined['genes']
pheno = combined['pheno']
labels = combined['class_cntl0_dis1']

# Step 5: Export
genes.to_csv('normalized_genes.csv')
pheno.to_csv('phenotype_data.csv')
```

**When to use**: You need to inspect or modify data between steps

---

### Example 3: Custom File Patterns

```python
from coconut.coconut_runner import runner

# Your files: Study_A.expr.csv, Study_A.meta.csv, etc.
runner(
    csv_dir='./custom_format',
    output_dir='./results'
)

# Adjust patterns if needed:
from coconut.coconut_runner import load_gse_data_from_csv_directory

gse_data = load_gse_data_from_csv_directory(
    csv_directory='./custom_format',
    genes_pattern='.expr.csv',
    pheno_pattern='.meta.csv',
    object_prefix='Study_'
)
```

---

### Example 4: With Debug Mode

```python
from coconut import coconut_runner
from coconut.coconut_runner import runner

# Enable detailed progress messages
coconut_runner.debug_mode = True

# Now run with verbose output
runner(csv_dir='./studies', output_dir='./results')
```

**Debug Output**:
```
🔍 Scanning directory: ./studies
📁 Found 6 CSV files
📊 Loading dataset: GSE28750
   📈 Genes: 12000 genes × 20 samples
   📈 Sample names in genes: ['GSM1', 'GSM2', 'GSM3']...
   📋 Pheno: 20 samples × 5 attributes
   📋 Sample names in pheno: ['GSM1', 'GSM2', 'GSM3']...
   ✅ Final alignment: 20 samples
✅ Successfully loaded 3 datasets
```

**Use when**: Troubleshooting data loading issues

---

## 🔧 Advanced Examples

### Example 5: Batch Processing Multiple Projects

Process multiple independent COCONUT projects in sequence:

```python
from pathlib import Path
from coconut.coconut_runner import runner
import time

projects = [
    'sepsis_studies',
    'covid_studies',
    'cancer_studies'
]

results_summary = []

for project in projects:
    print(f"\n{'='*60}")
    print(f"🔄 Processing {project}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        runner(
            csv_dir=f'./data/{project}',
            output_dir=f'./results/{project}'
        )
        
        elapsed = time.time() - start_time
        results_summary.append({
            'project': project,
            'status': 'success',
            'time': f'{elapsed:.1f}s'
        })
        
        print(f"✅ {project} completed in {elapsed:.1f}s")
        
    except Exception as e:
        print(f"❌ {project} failed: {e}")
        results_summary.append({
            'project': project,
            'status': 'failed',
            'error': str(e)
        })

# Print summary
print(f"\n{'='*60}")
print("📊 BATCH PROCESSING SUMMARY")
print(f"{'='*60}")
for result in results_summary:
    status_icon = '✅' if result['status'] == 'success' else '❌'
    print(f"{status_icon} {result['project']}: {result.get('time', result.get('error'))}")
```

**Use when**: Processing multiple independent studies or projects

---

### Example 6: Custom Validation Workflow

Add validation checks before running COCONUT:

```python
from coconut.coconut_runner import load_gse_data_from_csv_directory
from coconut.coconut import coconut, combine_coco_output, _get_common_genes

def validate_and_run(csv_dir, output_dir, min_genes=100, min_controls=3):
    """
    Run COCONUT with validation checks.
    
    Args:
        csv_dir: Input directory
        output_dir: Output directory
        min_genes: Minimum required common genes
        min_controls: Minimum controls per dataset
    """
    print("🔍 Step 1: Loading data...")
    gse_data = load_gse_data_from_csv_directory(csv_dir)
    
    print("\n🔍 Step 2: Validating datasets...")
    
    # Check number of datasets
    if len(gse_data) < 2:
        raise ValueError(f"Need at least 2 datasets, found {len(gse_data)}")
    print(f"✅ Found {len(gse_data)} datasets")
    
    # Check common genes
    common_genes = _get_common_genes(gse_data)
    if len(common_genes) < min_genes:
        raise ValueError(f"Only {len(common_genes)} common genes (minimum: {min_genes})")
    print(f"✅ Found {len(common_genes)} common genes")
    
    # Check controls per dataset
    for i, gse in enumerate(gse_data):
        n_controls = (gse['pheno']['Healthy0.Sepsis1'] == 0).sum()
        if n_controls < min_controls:
            raise ValueError(f"Dataset {i} has only {n_controls} controls (minimum: {min_controls})")
        print(f"✅ Dataset {i}: {n_controls} controls")
    
    print("\n🚀 Step 3: Running COCONUT...")
    result = coconut(gses=gse_data, control_0_col='Healthy0.Sepsis1')
    
    print("\n🔗 Step 4: Combining results...")
    combined = combine_coco_output(result)
    
    print("\n💾 Step 5: Exporting...")
    Path(output_dir).mkdir(exist_ok=True)
    combined['genes'].to_csv(f'{output_dir}/genes.csv')
    combined['pheno'].to_csv(f'{output_dir}/pheno.csv')
    
    print("\n✅ COMPLETE!")
    return combined

# Use the validated workflow
result = validate_and_run('./studies', './results', min_genes=150, min_controls=5)
```

**Use when**: Need strict quality control before processing

---

### Example 7: Handling Edge Cases

Handle common edge cases gracefully:

```python
from coconut.coconut_runner import load_gse_data_from_csv_directory
from coconut.coconut import coconut, combine_coco_output
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def robust_coconut_pipeline(csv_dir, output_dir):
    """
    COCONUT pipeline with edge case handling.
    """
    # Load data
    gse_data = load_gse_data_from_csv_directory(csv_dir)
    
    # Handle edge case 1: Very small batches
    for i, gse in enumerate(gse_data):
        n_samples = len(gse['genes'].columns)
        if n_samples < 5:
            print(f"⚠️ Warning: Dataset {i} has only {n_samples} samples")
            print(f"   Results may be less reliable for this dataset")
    
    # Handle edge case 2: Missing data
    for i, gse in enumerate(gse_data):
        n_missing = gse['genes'].isna().sum().sum()
        if n_missing > 0:
            print(f"⚠️ Dataset {i}: Imputing {n_missing} missing values")
            imputer = SimpleImputer(strategy='mean')
            gse['genes'] = pd.DataFrame(
                imputer.fit_transform(gse['genes'].T).T,
                index=gse['genes'].index,
                columns=gse['genes'].columns
            )
    
    # Handle edge case 3: Extreme outliers
    for i, gse in enumerate(gse_data):
        min_val = gse['genes'].min().min()
        max_val = gse['genes'].max().max()
        
        if min_val < -20 or max_val > 20:
            print(f"⚠️ Dataset {i}: Extreme values detected ({min_val:.1f} to {max_val:.1f})")
            print(f"   Clipping to [-15, 15] range")
            gse['genes'] = gse['genes'].clip(lower=-15, upper=15)
    
    # Handle edge case 4: Check gene name consistency
    all_gene_types = []
    for gse in gse_data:
        first_genes = list(gse['genes'].index[:5])
        all_gene_types.append(first_genes)
    
    print("\n🔍 Gene naming check:")
    for i, genes in enumerate(all_gene_types):
        print(f"   Dataset {i}: {genes}")
    
    # Run COCONUT
    print("\n🚀 Running COCONUT...")
    result = coconut(gses=gse_data, control_0_col='Healthy0.Sepsis1')
    combined = combine_coco_output(result)
    
    # Export
    Path(output_dir).mkdir(exist_ok=True)
    combined['genes'].to_csv(f'{output_dir}/genes.csv')
    combined['pheno'].to_csv(f'{output_dir}/pheno.csv')
    
    return combined

# Run robust pipeline
result = robust_coconut_pipeline('./studies', './results')
```

**Use when**: Working with imperfect real-world data

---

### Example 8: Integration with Existing Pipeline

Integrate COCONUT into a larger analysis pipeline:

```python
from coconut.coconut_runner import runner
from coconut.coconut import combine_coco_output
import pandas as pd
from pathlib import Path

def complete_analysis_pipeline(
    input_dir,
    output_dir,
    run_qc=True,
    run_deseq=True
):
    """
    Complete pipeline: COCONUT → QC → Downstream analysis.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("STEP 1: COCONUT Normalization")
    print("=" * 60)
    
    # Run COCONUT
    runner(csv_dir=input_dir, output_dir=str(output_path / 'coconut'))
    
    # Load results
    genes = pd.read_csv(output_path / 'coconut' / 'mcoco_combined_genes_samples.csv', index_col=0)
    pheno = pd.read_csv(output_path / 'coconut' / 'mcoco_combined_pheno.csv', index_col=0)
    labels = pd.read_csv(output_path / 'coconut' / 'mcoco_combined_class_labels.csv')
    
    if run_qc:
        print("\n" + "=" * 60)
        print("STEP 2: Quality Control")
        print("=" * 60)
        
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        
        # PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(genes.T)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], c=labels['class_label'], cmap='RdYlBu_r')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA after COCONUT')
        plt.colorbar(label='Class')
        plt.savefig(output_path / 'pca.png', dpi=300)
        print("✅ QC plots saved")
    
    if run_deseq:
        print("\n" + "=" * 60)
        print("STEP 3: Differential Expression")
        print("=" * 60)
        
        # Prepare for DESeq2 or other tools
        genes.to_csv(output_path / 'genes_for_deseq.csv')
        pheno.to_csv(output_path / 'pheno_for_deseq.csv')
        print("✅ Files prepared for downstream analysis")
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    
    return {
        'genes': genes,
        'pheno': pheno,
        'labels': labels
    }

# Run complete pipeline
results = complete_analysis_pipeline(
    input_dir='./raw_data',
    output_dir='./analysis_results',
    run_qc=True,
    run_deseq=True
)
```

**Use when**: COCONUT is part of a larger workflow

---

### Example 9: Memory-Efficient Large Dataset Processing

For very large datasets, process in chunks to manage memory:

```python
from coconut.coconut_runner import load_gse_data_from_csv_directory
from coconut.coconut import coconut, combine_coco_output
import gc
import numpy as np

def memory_efficient_coconut(csv_dir, output_dir, max_genes_per_chunk=5000):
    """
    Process large datasets in gene chunks to reduce memory usage.
    """
    print("🔍 Loading data...")
    gse_data = load_gse_data_from_csv_directory(csv_dir)
    
    # Get all genes
    all_genes = sorted(set.intersection(*[set(gse['genes'].index) for gse in gse_data]))
    n_genes = len(all_genes)
    
    print(f"📊 Total genes: {n_genes}")
    print(f"📦 Processing in chunks of {max_genes_per_chunk} genes")
    
    # Process in chunks
    results = []
    
    for chunk_start in range(0, n_genes, max_genes_per_chunk):
        chunk_end = min(chunk_start + max_genes_per_chunk, n_genes)
        chunk_genes = all_genes[chunk_start:chunk_end]
        
        print(f"\n🔄 Processing genes {chunk_start+1}-{chunk_end} ({len(chunk_genes)} genes)...")
        
        # Subset data to chunk
        chunk_data = []
        for gse in gse_data:
            chunk_data.append({
                'genes': gse['genes'].loc[chunk_genes],
                'pheno': gse['pheno']
            })
        
        # Run COCONUT on chunk
        result = coconut(gses=chunk_data, control_0_col='Healthy0.Sepsis1')
        combined = combine_coco_output(result)
        
        results.append(combined['genes'])
        
        # Free memory
        del chunk_data, result, combined
        gc.collect()
        
        print(f"✅ Chunk complete")
    
    # Combine all chunks
    print("\n🔗 Combining all chunks...")
    final_genes = pd.concat(results, axis=0)
    
    # Save
    Path(output_dir).mkdir(exist_ok=True)
    final_genes.to_csv(f'{output_dir}/genes_normalized.csv')
    
    print(f"\n✅ Complete! Processed {n_genes} genes")
    return final_genes

# Use for large datasets
result = memory_efficient_coconut('./large_dataset', './results', max_genes_per_chunk=3000)
```

**Use when**: Working with >50K genes or limited memory

---

## 🔧 Troubleshooting

### Error: "No CSV files found in directory"

**Cause**: Directory is empty or path is wrong

**Solution**:
```python
from pathlib import Path

# Verify directory exists
print(Path('./my_data').exists())

# List CSV files
print(list(Path('./my_data').glob('*.csv')))
```

---

### Error: "No complete datasets found"

**Cause**: Missing genes or pheno file for some datasets

**Solution**: Ensure each dataset has BOTH files:
- `Dataset_A_genes.csv` AND `Dataset_A_pheno.csv`
- Check file naming patterns match `genes_pattern` and `pheno_pattern`

---

### Warning: "Sample mismatch"

**Cause**: Sample names differ between genes and pheno files

**Example**:
```
⚠️  Sample mismatch in GSE28750:
      Genes has: 20 samples
      Pheno has: 18 samples
      Only in genes: ['Sample19', 'Sample20']
      Using 18 common samples
```

**What happens**: `load_gse_data_from_csv_directory()` automatically uses only common samples

**If problematic**: Check your source files for data entry errors

---

### Error: "Dataset has < 1 control"

**Cause**: Wrong control column or missing control samples

**Solution**: Verify control column has 0 values (controls):
```python
for i, gse in enumerate(gse_data):
    print(f"Study {i}: {gse['pheno']['control_col'].value_counts()}")
```

Expected output:
```
Study 0: 
0    10  ← controls
1    10  ← disease
```

---

## 🆚 coconut_runner vs coconut (core API)

| Feature | coconut_runner | coconut (core) |
|---------|---------------|----------------|
| **Ease of use** | ⭐⭐⭐⭐⭐ Very easy | ⭐⭐⭐ Moderate |
| **Code needed** | 3 lines | 10-20 lines |
| **CSV loading** | ✅ Automatic | ❌ Manual |
| **Sample alignment** | ✅ Automatic | ❌ Manual |
| **Validation** | ✅ Built-in | ❌ Manual |
| **Error messages** | ✅ Helpful | ⚠️ Technical |
| **Flexibility** | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Maximum |
| **Custom workflows** | ⚠️ Limited | ✅ Full control |

**Recommendation**:
- **Start with `coconut_runner`** for most use cases
- **Use core API** only when you need custom workflows

---

## 💡 Best Practices

### 1. Always Use coconut_runner for CSV Data

```python
# ✅ GOOD: Use coconut_runner
from coconut.coconut_runner import runner
runner(csv_dir='./data', output_dir='./results')

# ❌ BAD: Manual CSV loading
genes1 = pd.read_csv('study1_genes.csv')
pheno1 = pd.read_csv('study1_pheno.csv')
# ... lots of repetitive code ...
```

### 2. Enable Debug Mode for First Run

```python
from coconut import coconut_runner
coconut_runner.debug_mode = True

# See detailed progress and catch issues early
runner(csv_dir='./data', output_dir='./results')
```

### 3. Validate Data Before Large Runs

```python
# Quick check without saving
runner(csv_dir='./data')  # No output_dir = only prints

# If looks good, run full pipeline
runner(csv_dir='./data', output_dir='./results')
```

### 4. Use Consistent File Naming

**Recommended naming convention**:
```
Study1_genes.csv
Study1_pheno.csv
Study2_genes.csv
Study2_pheno.csv
```

**Avoid**:
- Inconsistent prefixes (`Study1_genes.csv` vs `S2_genes.csv`)
- Missing patterns (`genes.csv` vs `expression.csv`)
- Special characters in dataset names

---

## 📊 Output File Guide

### File 1: `mcoco_combined_genes_samples.csv`

**Format**: Standard gene expression matrix
```
         Sample1  Sample2  Sample3  ...
Gene1    5.23     6.45     4.87     ...
Gene2    7.12     5.98     8.34     ...
...
```

**Use for**: Standard downstream analysis, visualization

---

### File 2: `mcoco_combined_genes.csv`

**Format**: Transposed (samples as rows)
```
         Gene1  Gene2  Gene3  ...
Sample1  5.23   7.12   9.45   ...
Sample2  6.45   5.98   8.23   ...
...
```

**Use for**: Machine learning (features = genes, samples = instances)

---

### File 3: `mcoco_combined_pheno.csv`

**Format**: Sample metadata
```
         Age  Sex  Platform_id
Sample1  45   M    GPL570
Sample2  32   F    GPL6947
...
```

**Use for**: Covariates in statistical models, sample annotation

---

### File 4: `mcoco_combined_class_labels.csv`

**Format**: Binary class labels
```
sample_id,class_label
Sample1,0
Sample2,1
Sample3,0
...
```

**Use for**: Classification tasks, statistical testing

---

## 🎯 Summary

**coconut_runner** makes COCONUT:
- ✅ **Easier**: 3 lines instead of 20+
- ✅ **Safer**: Automatic validation and alignment
- ✅ **Faster**: Production-ready workflows
- ✅ **Better**: Helpful error messages and debug mode

**Start here** for all CSV-based COCONUT workflows. Graduate to the core API only when you need maximum customization.

---

## 📞 Getting Help

- **Command-Line Interface**: [cli.md](../getting_started/cli.md)
- **Quick Start**: [quickstart.md](quickstart.md)
- **Workflow Guide**: [workflow.md](workflow.md)
- **API Reference**: [api.md](api.md)
- **Best Practices**: [best_practices.md](best_practices.md)
- **Troubleshooting**: [troubleshooting.md](troubleshooting.md)
- **R Comparison**: [r_to_python.md](r_to_python.md)
