# Performance Analysis

Algorithm complexity analysis and performance considerations for **metaIntegrator-py (Python)** and **MetaIntegrator (R)**.

!!! warning "Important: Benchmark Status"
    **The performance data in this document represents theoretical estimates based on algorithm complexity analysis and limited informal testing. Comprehensive head-to-head benchmarks with statistical validation have NOT been completed.**
    
    **What this means:**
    - The timing values below are illustrative estimates, not measured results
    - Actual performance depends heavily on dataset characteristics and hardware
    - We have NOT validated these numbers through rigorous benchmarking
    - Real-world performance may differ significantly
    
    **We recommend:**
    - Benchmark with your own data and use cases
    - Both implementations are performant for typical meta-analysis workflows
    - Choose based on ecosystem fit, not performance claims
    
    **Contributing:** We welcome contributions to develop a comprehensive, reproducible benchmark suite!

## Contents

- [Test Environment](#test-environment)
- [Algorithm Complexity](#algorithm-complexity)
- [Theoretical Performance](#theoretical-performance)
- [Memory Considerations](#memory-considerations)
- [GEO Data Caching](#geo-data-caching)
- [Optimization Tips](#optimization-tips)
- [Summary](#summary)

---

## Test Environment

### Reference Specifications

The theoretical estimates below assume the following environment:

| Component | Specification |
|-----------|---------------|
| **CPU** | Modern multi-core processor (8 cores) |
| **RAM** | 32 GB DDR4 |
| **Storage** | SSD |
| **OS** | Linux/macOS/Windows |

### Software Versions

| Software | Python | R |
|----------|--------|---|
| **Language** | Python 3.9+ | R 4.2+ |
| **Package** | metaIntegrator-py | MetaIntegrator (Bioconductor) |
| **Key Dependencies** | NumPy, pandas, scipy | Bioconductor packages |

### Test Dataset Sizes

| Dataset | Genes | Samples | Studies | Source |
|---------|-------|---------|---------|--------|
| **Small** | 5,000 | 50 | 3 | Typical GEO subset |
| **Medium** | 20,000 | 200 | 5 | Typical meta-analysis |
| **Large** | 50,000 | 500 | 10 | Comprehensive study |

---

## Algorithm Complexity

Both implementations use the same statistical algorithms with equivalent computational complexity:

### Core Operations

| Operation | Complexity | Notes |
|-----------|-----------|--------|
| **Effect Size Calculation** | O(n × g × s) | n=samples, g=genes, s=studies |
| **Random Effects Pooling** | O(g × s) | Per-gene across studies |
| **Fisher's Method** | O(g × s) | P-value combination |
| **FDR Correction** | O(g log g) | Benjamini-Hochberg |
| **Leave-One-Out** | O(g × s²) | Quadratic in studies |

**Key Insight**: Computational complexity is **identical** between Python and R implementations. Any performance differences come from:
- Vectorization efficiency (NumPy vs R's C backend)
- Memory management
- Parallelization overhead
- I/O operations

---

## Theoretical Performance

!!! info "Estimates Only"
    The following estimates are based on algorithm complexity and informal testing.
    **They have NOT been validated through rigorous benchmarking.**

### Core Operations (Estimated)

#### Effect Size Calculation

Theoretical performance for calculating effect sizes across all genes:

| Dataset | Estimated Time Range |
|---------|---------------------|
| Small (5K genes, 50 samples) | 1-3 seconds |
| Medium (20K genes, 200 samples) | 5-10 seconds |
| Large (50K genes, 500 samples) | 25-40 seconds |

**Both implementations** should perform comparably due to vectorized operations.

---

#### Random Effects Pooling

Theoretical performance for pooling effect sizes:

| Dataset | Estimated Time Range |
|---------|---------------------|
| Small | < 1 second |
| Medium | 2-4 seconds |
| Large | 8-15 seconds |

**Expected advantage**: NumPy's matrix operations may offer marginal improvements.

---

#### Complete Meta-Analysis

Full pipeline estimate (loading → validation → analysis):

| Dataset | Estimated Time Range |
|---------|---------------------|
| Small | 3-8 seconds |
| Medium | 15-30 seconds |
| Large | 60-120 seconds |

**Practical note**: For typical meta-analysis workflows, both implementations complete in seconds to minutes. The bottleneck is usually GEO data download, not computation.

---

## Memory Considerations

### Theoretical Memory Usage

Both implementations store expression matrices in memory:

| Dataset | Estimated Memory |
|---------|-----------------|
| Small (5K × 50) | 100-300 MB |
| Medium (20K × 200) | 500-1500 MB |
| Large (50K × 500) | 2-5 GB |

**Memory optimization strategies** (both platforms):
- Process studies sequentially instead of all at once
- Filter genes early (remove low-variance genes)
- Use sparse matrices when appropriate
- Leverage memory-mapped files for very large datasets

**Python specifics**:
- pandas DataFrames have overhead (~2-3× raw array size)
- NumPy arrays are memory-efficient
- Garbage collection is automatic

**R specifics**:
- data.frames have similar overhead to pandas
- Memory management is also automatic
- May use more memory for intermediate objects

---

## GEO Data Caching

### Python Advantage: Automatic Caching ✅

**metaIntegrator-py** includes automatic caching:

```python
# First download - network dependent (may take minutes)
meta_object = get_geo_data(
    gse_vector=["GSE5847"],
    cache_dir="./geo_cache"
)

# Subsequent runs - loads from cache (< 5 seconds)
meta_object = get_geo_data(
    gse_vector=["GSE5847"],
    cache_dir="./geo_cache"
)
```

**Benefit**: Repeated analyses are much faster without re-downloading data.

### R: Manual Caching Required

**MetaIntegrator (R)** requires manual caching:

```r
# Must manually save and load data
metaObj <- getGEOData(c("GSE5847"))
save(metaObj, file="cached_data.RData")

# Later sessions - manual load
load("cached_data.RData")
```

**Impact**: Python's automatic caching improves workflow efficiency for iterative analysis.

---

## Optimization Tips

### Python Optimizations

1. **Use NumPy Operations**
   ```python
   # Slow - Python loop
   results = [compute(x) for x in data]
   
   # Fast - NumPy vectorized
   results = np.vectorize(compute)(data)
   ```

2. **Pre-allocate Arrays**
   ```python
   # Slow - growing lists
   results = []
   for i in range(n):
       results.append(compute(i))
   
   # Fast - pre-allocated
   results = np.empty(n)
   for i in range(n):
       results[i] = compute(i)
   ```

3. **Leverage Pandas Methods**
   ```python
   # Use built-in methods instead of apply
   df['zscore'] = (df['value'] - df['value'].mean()) / df['value'].std()
   ```

---

### R Optimizations

1. **Vectorize Operations**
   ```r
   # Slow - loop
   for(i in 1:length(x)) {
     results[i] <- x[i] * 2
   }
   
   # Fast - vectorized
   results <- x * 2
   ```

2. **Use data.table**
   ```r
   # Faster than base R data.frame
   library(data.table)
   dt <- as.data.table(df)
   dt[value > threshold]
   ```

3. **Pre-allocate Vectors**
   ```r
   # Slow - growing vector
   results <- c()
   for(i in 1:n) {
     results <- c(results, compute(i))
   }
   
   # Fast - pre-allocated
   results <- numeric(n)
   for(i in 1:n) {
     results[i] <- compute(i)
   }
   ```

---

## Parallelization

### Python: ProcessPoolExecutor

```python
from concurrent.futures import ProcessPoolExecutor

def analyze_study(dataset):
    # Analysis code
    return results

# Parallel processing
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(analyze_study, datasets))
```

**Pros**:
- Cross-platform (Windows, macOS, Linux)
- Clean API
- Automatic load balancing

**Cons**:
- Process spawn overhead
- Data serialization overhead

### R: mclapply

```r
library(parallel)

# Parallel processing (Unix/macOS only)
results <- mclapply(datasets, analyze_study, mc.cores=4)

# Windows alternative
cl <- makeCluster(4)
results <- parLapply(cl, datasets, analyze_study)
stopCluster(cl)
```

**Pros**:
- Low overhead on Unix/macOS
- Integrated with R ecosystem

**Cons**:
- Platform-dependent (mclapply Unix/macOS only)
- Different APIs for different platforms

---

## Real-World Performance Factors

### What Actually Matters

In practice, performance is dominated by:

1. **GEO Download Speed** (network-dependent)
   - First download: minutes to hours
   - Cached data: seconds
   - **Python advantage**: Automatic caching

2. **Gene Filtering Strategy**
   - Removing 75% of genes → 4× faster analysis
   - More important than language choice

3. **Number of Studies**
   - Leave-one-out scales quadratically
   - 10 studies: manageable
   - 50+ studies: consider batching

4. **Hardware**
   - More RAM → can process larger datasets
   - More cores → faster parallelization
   - SSD → faster I/O

**Bottom line**: For typical meta-analysis workflows (3-10 studies, 10K-30K genes), both Python and R complete in seconds to minutes. The language choice has minimal impact on user experience.

---

## Summary

### Key Findings

1. **Computational Complexity**: Identical between implementations
2. **Practical Performance**: Both are fast enough for typical use cases
3. **Caching**: Python's automatic caching is a workflow advantage
4. **Bottlenecks**: Usually data download, not computation
5. **Optimization**: Good coding practices matter more than language

### Honest Comparison

| Aspect | Python | R |
|--------|--------|---|
| **Algorithm complexity** | Same | Same |
| **Typical workflow speed** | Seconds to minutes | Seconds to minutes |
| **GEO data caching** | ✅ Automatic | ⚠️ Manual |
| **Memory efficiency** | Good | Good |
| **Parallelization** | Cross-platform | Platform-dependent |
| **Real bottleneck** | Network download | Network download |

### Recommendation

**Choose based on ecosystem and workflow, not performance claims.**

Both implementations are performant for meta-analysis. The real differences are:
- **Python**: Better ML integration, cloud deployment, automatic caching
- **R**: Better Bioconductor integration, publication-ready graphics

---

## Contributing Benchmarks

We welcome contributions to develop a rigorous benchmark suite! A proper benchmark should include:

✅ **Reproducible test scripts**
✅ **Multiple dataset sizes**
✅ **Statistical analysis** (mean, SD, confidence intervals)
✅ **Hardware specifications**
✅ **Multiple runs** (n ≥ 10)
✅ **Same datasets** for both languages
✅ **Clear methodology documentation**

**To contribute**: Please open an issue or PR on GitHub with your benchmark methodology and results.

---

## Next Steps

- [R vs Python Comparison](r_vs_python.md) - Feature comparison
- [Migration Guide](migration_guide.md) - Migrate from R
- [API Reference](../api/overview.md) - Python API documentation
- [Tutorials](../tutorials/complete_workflow.md) - Hands-on examples
