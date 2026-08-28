# 🧬 Running R MetaIntegrator Jupyter Notebook 

## 🎯 What Was Created

### ✅ R Jupyter Notebook
**Location:** `<project folder>/metaintegrator_py/tutorials/jupyter_metaintegrator_sle_r_tutorial.ipynb`

---

## 🚀 Quick Start

### Prerequisites

1. **Install Jupyter with R kernel:**
   ```bash
   # Install R kernel for Jupyter
   R -e "install.packages('IRkernel')"
   R -e "IRkernel::installspec(user = FALSE)"
   ```

2. **Install MetaIntegrator package:**
   ```r
   # In R console
   install.packages("BiocManager")
   BiocManager::install("MetaIntegrator")
   ```

3. **Verify installation:**
   ```bash
   jupyter kernelspec list
   # Should show 'ir' kernel
   ```

---

## 📝 Key Differences from Original Vignette

### Original Rmd Vignette
- **Format:** R Markdown (.Rmd)
- **Execution:** Most code chunks have `eval=FALSE`
- **Purpose:** Documentation only
- **Output:** Rendered HTML
- **Interactivity:** None

### New R Notebook
- **Format:** Jupyter Notebook (.ipynb)
- **Execution:** All code chunks executable
- **Purpose:** Interactive analysis + parity testing
- **Output:** Live in-notebook results
- **Interactivity:** Full

### Why Jupyter Instead of Rmd?

1. **Same format as Python** - Direct comparison
2. **Interactive execution** - Test changes immediately
3. **Cell-by-cell output** - Debug issues easily
4. **Version control** - Track changes better
5. **Developer friendly** - Familiar interface
6. **Parity testing** - Side-by-side comparison

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. R Kernel Not Found
```bash
# Error: "No kernel named 'ir' found"

# Solution:
R -e "install.packages('IRkernel')"
R -e "IRkernel::installspec(user = FALSE)"
jupyter kernelspec list
```

#### 2. MetaIntegrator Not Installed
```r
# Error: "there is no package called 'MetaIntegrator'"

# Solution:
install.packages("BiocManager")
BiocManager::install("MetaIntegrator")
```

#### 3. GEO Download Fails
```r
# Error: Connection timeout or "Cannot connect to GEO"

# Solutions:
# 1. Check internet connection
# 2. Try again later (GEO may be down)
# 3. Use cached data if available
```

#### 4. Memory Issues
```r
# Error: "cannot allocate vector of size..."

# Solutions:
# 1. Increase R memory limit
options(java.parameters = "-Xmx8g")
# 2. Close other applications
# 3. Run on machine with more RAM
```

#### 5. Plot Not Displaying
```r
# No plot shown in notebook

# Solution:
# Set plot size explicitly
options(repr.plot.width = 10, repr.plot.height = 8)
```

---

## 📚 Advanced Usage

#### 1. Enable Advanced Analyses
```r
# Uncomment in Section 6:
# Forward search
forwardRes <- forwardSearch(
    metaObject = myMetaAnalysis,
    filterObject = myMetaAnalysis$filterResults[[1]]
)

# LINCS correlation
lincsHits <- lincsCorrelate(
    metaObject = myMetaAnalysis,
    filterObject = myMetaAnalysis$filterResults[[1]],
    dataset = "CP",
    direction = "reverse"
)
```

### Exporting Results

```r
# Export gene signature
write.csv(
    data.frame(
        Gene = c(filterResults$posGeneNames, filterResults$negGeneNames),
        Direction = c(
            rep("Up", length(filterResults$posGeneNames)),
            rep("Down", length(filterResults$negGeneNames))
        )
    ),
    "sle_gene_signature.csv",
    row.names = FALSE
)

# Export meta-analysis results
write.csv(
    sleMetaAnalysis$metaAnalysis$datasetEffectSizes,
    "sle_meta_analysis_results.csv",
    row.names = TRUE
)

# Save workspace
save.image("sle_analysis_workspace.RData")
```

---

## 📖 Documentation References

### R MetaIntegrator
- **Main vignette:** `browseVignettes("MetaIntegrator")`
- **GitHub:** https://github.com/Khatri-Lab/MetaIntegrator
- **Paper:** Haynes et al. (2017) PSB 22:144-153

### Python metaintegrator-py
- **Documentation:** See `<project folder>/metaintegrator_py/docs`
- **Tutorial:** `jupyter_metaintegrator_sle_tutorial.ipynb`
- **Implementation:** See  `<project folder>/metaintegrator_py/src` modules

### Jupyter & R Kernel
- **IRkernel docs:** https://irkernel.github.io/
- **Jupyter docs:** https://jupyter.org/documentation

---
 
**Date:** 2026-06-12  
**Version:** 1.0
