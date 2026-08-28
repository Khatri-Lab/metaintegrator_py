# Reference Data Directory

This directory contains reference matrices used for immune cell deconvolution and other analyses.

## ImmunoStates Reference Matrix

### What is it?

The ImmunoStates reference matrix is a curated gene expression signature matrix for 24 human immune cell types. It's used for deconvolving bulk RNA-seq or microarray data to estimate cell type proportions.

**Citation:**
> Vallania, F., Tam, A., Lofgren, S., Schaffert, S., Azad, T. D., Bongen, E., ... & Khatri, P. (2018).
> Leveraging heterogeneity across multiple datasets increases cell-mixture deconvolution accuracy and reduces biological and technical biases.
> *Nature Communications*, 9(1), 4735.

### How to Obtain the Matrix

#### Option 1: Download from R Package (Recommended)

The original immunoStates matrix is distributed as part of the R `immunoStates` package:

```bash
# In R console:
install.packages("BiocManager")
BiocManager::install("immunoStates")

# Load the package
library(immunoStates)

# The matrix is available as immunoStatesMatrix
data(immunoStatesMatrix)

# Export to CSV
write.csv(immunoStatesMatrix, "immunoStates_reference_matrix.csv")
```

Then place the CSV file in this directory.

#### Option 2: Download from the Khatri Lab

The reference matrix is also available directly from the Khatri Lab:

```
http://khatrilab.stanford.edu/immunoStates
```

After downloading the `.rda` file, convert it using the provided script:

```bash
python convert_immunostates_rda_to_csv.py
```

#### Option 3: Use Provided Conversion Script

We provide a helper script to convert the R .rda format to CSV:

```bash
cd <project folder>/metaintegrator_py/src/metaintegrator/data/

# Download and convert
python scripts/download_and_convert_immunostates.py
```

### Expected Format

The CSV file should have:
- **Rows:** Gene symbols (HUGO nomenclature)
- **Columns:** Cell type names
- **Values:** Gene expression levels (typically log2 or normalized)
- **Dimensions:** ~317 genes × 24 cell types

**Example structure:**
```csv
,CD56bright_natural_killer_cell,CD56dim_natural_killer_cell,CD14_positive_monocyte,...
CD3D,1.234,0.567,0.123,...
CD4,2.345,1.234,0.234,...
CD8A,1.456,2.345,0.345,...
...
```

### File Location

Place the converted CSV file here:
```
<project folder>/metaintegrator_py/src/metaintegrator/data/immunoStates_reference_matrix.csv
```

### Validation

After placing the file, test that it loads correctly:

```python
from metaintegrator.reference_data_manager import load_immunostates_reference

# Load the matrix
matrix = load_immunostates_reference()

# Should show actual data, not mock data
print(f"Loaded {matrix.shape[0]} genes × {matrix.shape[1]} cell types")

# Verify it's real data (should NOT see mock data warning)
```

### Mock Data (Development Only)

⚠️ If the reference matrix is not found, the system will use **MOCK DATA** for development/testing.

**IMPORTANT:** Mock data is randomly generated and should NEVER be used for:
- Actual biological analyses
- Publications
- Clinical applications
- Production systems

You'll see a loud warning if mock data is being used.

### Cell Types Included

The full immunoStates matrix includes 24 cell types:

**NK Cells:**
- CD56bright natural killer cell
- CD56dim natural killer cell

**Monocytes:**
- CD14-positive monocyte
- CD16-positive monocyte

**B Cells:**
- Naive B cell
- Memory B cell
- Plasma cell

**T Cells:**
- CD8-positive alpha-beta T cell
- CD4-positive alpha-beta T cell
- Gamma-delta T cell
- CD4 Tcm (central memory)
- CD4 Tem (effector memory)
- CD8 Tcm
- CD8 Tem
- Regulatory T cell (Treg)
- Th1
- Th2
- Th17
- Tfh (follicular helper)

**Granulocytes:**
- Neutrophil
- Eosinophil
- Basophil

**Dendritic Cells:**
- Myeloid dendritic cell
- Plasmacytoid dendritic cell

### Troubleshooting

**"Matrix not found" error:**
- Verify the CSV file is in this directory
- Check the filename matches: `immunoStates_reference_matrix.csv`
- Ensure it's a valid CSV with proper structure

**"Matrix validation failed":**
- Check that the CSV has gene names as the first column (index)
- Verify all values are numeric
- Ensure proper formatting (no extra headers, footers, etc.)

**Still using mock data:**
- The reference data manager looks in multiple locations
- Check the console output to see where it's searching
- Verify file permissions allow reading

### Support

For issues with obtaining or converting the matrix:
1. Check the R immunoStates package documentation
2. See the conversion script in `scripts/`
3. Review the reference data manager code for supported formats
4. Contact the package maintainers

---

*Last updated: 2026-03-11*
