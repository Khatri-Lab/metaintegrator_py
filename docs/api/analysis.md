# Analysis API

This module provides the full analysis pipeline: meta-analysis, gene filtering,
score calculation, and ROC/PRC evaluation.

## Module Overview

| Module | Class | Purpose |
|--------|-------|--------|
| `analysis` | `MetaAnalyzer` | Top-level orchestrator (effect sizes → pooling → LOOA) |
| `core_analysis` | `MetaAnalysisCore`, `FisherCombination` | Statistical engine |
| `scoring_meta_integrator_filter` | `MetaIntegratorFilter` | Gene filtering by FDR + ES + LOOA |
| `score_calculation` | `ScoreCalculator` | Geometric-mean z-score per sample |
| `sample_level_gene_data` | `SampleLevelGeneDataProcessor` | Probe → gene-level matrix |
| `roc_analysis_engine` | `ROCAnalysisEngine` | ROC curves, AUC, multi-dataset ROC |
| `prc_analysis` | `PRCAnalysis` | Precision-recall curves |
| `power_calculation` | `PowerCalculation` | Sample-size & power estimation |
| `scoring_core` | `FilterConfig`, `FilterResults`, validators | Shared infrastructure |
| `scoring_export_engine` | `ExportEngine` | Export to CSV / Excel |
| `scoring_meta_integrator_orchestrator` | `ScoringOrchestrator` | End-to-end pipeline |

---

## MetaAnalyzer

Main orchestrator for conducting meta-analysis with parallel processing support.
Delegates statistical computation to a pluggable `MetaAnalysisEngine`.

```python
class MetaAnalyzer(
    max_cores: Optional[int] = None,
    core=None,
    fisher=None,
    validator=None,
    processor=None,
    engine: Optional[MetaAnalysisEngine] = None,
)
```

### Constructor

| Name | Type | Description |
|------|------|-------------|
| max_cores | int, optional | Maximum CPU cores (default: `min(10, available)`) |
| engine | MetaAnalysisEngine, optional | Pluggable analysis strategy. Defaults to `RandomEffectsEngine`. |
| core | class, optional | Effect-size backend (shortcut: auto-wrapped into `RandomEffectsEngine` when `engine` is None) |
| fisher | class, optional | Fisher combination backend (same shortcut) |
| validator | class, optional | Override `DatasetValidator` |
| processor | class, optional | Override `DataProcessor` |

**Example**

```python
from metaintegrator.analysis import MetaAnalyzer, MetaAnalysisFactory

# Default (random effects)
analyzer = MetaAnalyzer()

# Limit to 4 cores
analyzer = MetaAnalyzer(max_cores=4)

# Via factory (recommended for switching strategies)
analyzer = MetaAnalysisFactory.create_analyzer("fixed_effects")
```

---

### `run_meta_analysis`

```python
def run_meta_analysis(
    self,
    meta_object: Dict,
    run_leave_one_out: bool = True,
    use_old_method: bool = False,
    max_cores: Optional[int] = None
) -> Dict
```

Execute complete meta-analysis with validation, effect size calculation, and optional leave-one-out analysis.

| Name | Type | Description |
|------|------|-------------|
| meta_object | Dict | Meta-analysis object with 'originalData' |
| run_leave_one_out | bool | Perform leave-one-out analysis (default: True) |
| use_old_method | bool | Use alternative Fisher method (default: False) |
| max_cores | int, optional | Override constructor core setting |

**Returns**

| Type | Description |
|------|-------------|
| Dict | Updated meta_object with 'metaAnalysis' and optionally 'leaveOneOutAnalysis' |

**Example**

```python
# Basic usage
meta_object = {
    "originalData": {
        "Study1": study1_dict,
        "Study2": study2_dict,
        "Study3": study3_dict
    }
}

analyzer = MetaAnalyzer()
results = analyzer.run_meta_analysis(meta_object)

# Access pooled results
pooled = results["metaAnalysis"]["pooledResults"]
top_genes = pooled.nsmallest(20, "effectSizeFDR")
print(top_genes[["effectSize", "effectSizeFDR"]])
```

**Output Structure**

```python
{
    "originalData": {...},  # Input data
    "metaAnalysis": {
        "pooledResults": pd.DataFrame,
        "datasetEffectSizes": pd.DataFrame,
        "datasetEffectSizeStandardErrors": pd.DataFrame,
        "analysisDescription": str
    },
    "leaveOneOutAnalysis": {  # If run_leave_one_out=True
        "removed_Study1": {...},
        "removed_Study2": {...},
        "removed_Study3": {...}
    }
}
```

---

### `run_leave_one_out_analysis`

```python
def run_leave_one_out_analysis(
    self,
    original_data: List[Dict],
    use_old_method: bool = False
) -> List[Dict]
```

Perform leave-one-out sensitivity analysis.

| Name | Type | Description |
|------|------|-------------|
| original_data | List[Dict] | List of study dictionaries |
| use_old_method | bool | Use alternative method (default: False) |

**Returns**

| Type | Description |
|------|-------------|
| List[Dict] | Results for each leave-one-out iteration |

**Example**

```python
studies = list(meta_object["originalData"].values())

# Requires at least 3 studies
if len(studies) >= 3:
    loo_results = analyzer.run_leave_one_out_analysis(studies)
    
    # Compare results across iterations
    for i, result in enumerate(loo_results):
        pooled = result["pooledResults"]
        print(f"\\nExcluding study {i+1}:")
        print(f"  Median effect size: {pooled['effectSize'].median():.3f}")
```

> **Note:** Requires minimum 3 studies. Automatically parallelized based on `max_cores`.

---

## MetaAnalysisCore

Core statistical functions for meta-analysis.

### `calculate_effect_sizes`

```python
@staticmethod
def calculate_effect_sizes(study: Dict) -> pd.DataFrame
```

Calculate effect sizes (Hedge's g) for all genes in a study.

| Name | Type | Description |
|------|------|-------------|
| study | Dict | Study object with 'expr', 'class', 'keys' |

**Returns**

| Type | Description |
|------|-------------|
| pd.DataFrame | Effect size statistics for each gene |

**Output Columns**
- `n1`, `n2` - Sample sizes (cases, controls)
- `m1`, `m2` - Means
- `sd1`, `sd2` - Standard deviations
- `diff` - Mean difference
- `pooled_sd` - Pooled standard deviation
- `g` - Hedge's g (effect size)
- `se_g` - Standard error
- `keys` - Gene identifier

**Example**

```python
from metaintegrator.core_analysis import MetaAnalysisCore

# Calculate for single study
effect_sizes = MetaAnalysisCore.calculate_effect_sizes(study1)

# Filter significant genes
significant = effect_sizes[effect_sizes["g"].abs() > 0.5]
print(f"Genes with |g| > 0.5: {len(significant)}")
```

---

### `pool_inverse_variance`

```python
@staticmethod
def pool_inverse_variance(
    g_matrix: pd.DataFrame,
    se_matrix: pd.DataFrame,
    method: str = "random"
) -> pd.DataFrame
```

Pool effect sizes across studies using inverse variance weighting.

| Name | Type | Description |
|------|------|-------------|
| g_matrix | pd.DataFrame | Effect sizes (genes × studies) |
| se_matrix | pd.DataFrame | Standard errors (genes × studies) |
| method | str | "random" or "fixed" effects model (default: "random") |

**Returns**

| Type | Description |
|------|-------------|
| pd.DataFrame | Pooled statistics with heterogeneity metrics |

**Output Columns**
- `n_studies` - Number of contributing studies
- `summary` - Pooled effect size
- `se_summary` - Standard error of pooled effect
- `tau2` - Between-study variance (random effects only)
- `p_value` - Two-tailed p-value
- `Q` - Cochran's Q statistic
- `df` - Degrees of freedom
- `pval_het` - P-value for heterogeneity test

**Example**

```python
# Prepare effect size matrices
studies = [study1, study2, study3]
effect_sizes = [MetaAnalysisCore.calculate_effect_sizes(s) for s in studies]

# Extract g and se_g matrices (assuming common genes)
g_matrix = pd.concat([es["g"] for es in effect_sizes], axis=1)
se_matrix = pd.concat([es["se_g"] for es in effect_sizes], axis=1)

# Pool with random effects
pooled = MetaAnalysisCore.pool_inverse_variance(
    g_matrix,
    se_matrix,
    method="random"
)

# Check for heterogeneity
heterogeneous = pooled[pooled["pval_het"] < 0.05]
print(f"Genes with significant heterogeneity: {len(heterogeneous)}")
```

**Fixed vs Random Effects**

- **Fixed effects** (`method="fixed"`)
  - Assumes all studies estimate same true effect
  - `tau2 = 0`
  - More powerful when heterogeneity is low

- **Random effects** (`method="random"`)
  - Allows true effects to vary across studies
  - Estimates `tau2` using DerSimonian-Laird method
  - More conservative when heterogeneity is present

---

### `get_effect_size`

```python
@staticmethod
def get_effect_size(
    expression: np.ndarray,
    groups: np.ndarray,
    method: str = "hedges_g"
) -> Dict
```

Calculate effect size for a single gene.

| Name | Type | Description |
|------|------|-------------|
| expression | np.ndarray | Gene expression values |
| groups | np.ndarray | Binary group labels (0=control, 1=case) |
| method | str | "hedges_g" or "cohen_d" (default: "hedges_g") |

**Returns**

| Type | Description |
|------|-------------|
| Dict | Dictionary with effect size statistics |

**Example**

```python
# Single gene analysis
gene_expr = study["expr"].iloc[0, :]  # First gene
groups = study["class"]

stats = MetaAnalysisCore.get_effect_size(gene_expr.values, groups)

print(f"Effect size (Hedge's g): {stats['g']:.3f}")
print(f"Standard error: {stats['se_g']:.3f}")
print(f"Cases: n={stats['n1']}, mean={stats['m1']:.2f}")
print(f"Controls: n={stats['n2']}, mean={stats['m2']:.2f}")
```

---

## FisherCombination

Fisher's method for combining p-values across studies.

### `calculate_study_pvalues`

```python
@staticmethod
def calculate_study_pvalues(study: Dict) -> pd.DataFrame
```

Calculate directional p-values for a study using t-tests.

| Name | Type | Description |
|------|------|-------------|
| study | Dict | Study object |

**Returns**

| Type | Description |
|------|-------------|
| pd.DataFrame | P-values with columns: P.up, P.down, keys |

**Example**

```python
from metaintegrator.core_analysis import FisherCombination

# Calculate p-values for one study
pvals = FisherCombination.calculate_study_pvalues(study1)

# Filter up-regulated genes
up_genes = pvals[pvals["P.up"] < 0.05]
print(f"Up-regulated genes (p < 0.05): {len(up_genes)}")
```

---

### `sum_of_logs`

```python
@staticmethod
def sum_of_logs(
    list_of_significance_matrices: List[pd.DataFrame]
) -> pd.DataFrame
```

Combine p-values across studies using Fisher's sum-of-logs method.

| Name | Type | Description |
|------|------|-------------|
| list_of_significance_matrices | List[pd.DataFrame] | P-value DataFrames from each study |

**Returns**

| Type | Description |
|------|-------------|
| pd.DataFrame | Combined Fisher statistics |

**Output Columns**
- `F.stat.up` - Fisher statistic for up-regulation
- `F.pval.up` - Combined p-value for up-regulation
- `F.stat.down` - Fisher statistic for down-regulation
- `F.pval.down` - Combined p-value for down-regulation

**Example**

```python
# Calculate p-values for each study
pval_list = [
    FisherCombination.calculate_study_pvalues(s)
    for s in [study1, study2, study3]
]

# Combine using Fisher's method
fisher_results = FisherCombination.sum_of_logs(pval_list)

# Find directionally consistent genes
up_regulated = fisher_results[fisher_results["F.pval.up"] < 0.05]
down_regulated = fisher_results[fisher_results["F.pval.down"] < 0.05]

print(f"Consistently up-regulated: {len(up_regulated)}")
print(f"Consistently down-regulated: {len(down_regulated)}")
```

---

## run_meta_analysis (Function)

Convenience function for running meta-analysis with minimal setup.

```python
def run_meta_analysis(
    meta_object: Dict,
    run_leave_one_out_analysis: bool = True,
    max_cores: int = None,
    use_old_method: bool = False
) -> Dict
```

| Name | Type | Description |
|------|------|-------------|
| meta_object | Dict | Meta-analysis object |
| run_leave_one_out_analysis | bool | Perform leave-one-out (default: True) |
| max_cores | int, optional | CPU cores to use |
| use_old_method | bool | Use alternative method (default: False) |

**Returns**

| Type | Description |
|------|-------------|
| Dict | Results dictionary |

**Example**

```python
from metaintegrator import run_meta_analysis

# Quick analysis
results = run_meta_analysis(meta_object, max_cores=4)

# Access results
pooled = results["metaAnalysis"]["pooledResults"]
```

---

## MetaAnalysisFactory

Factory for creating pre-configured `MetaAnalyzer` instances with different
statistical strategies.

```python
from metaintegrator.analysis import MetaAnalysisFactory
```

### `create_analyzer`

```python
@staticmethod
def create_analyzer(
    strategy: str = "random_effects",
    max_cores: Optional[int] = None,
    **kwargs,
) -> MetaAnalyzer
```

| Name | Type | Description |
|------|------|-------------|
| strategy | str | `"random_effects"` (default) or `"fixed_effects"` |
| max_cores | int, optional | Core limit passed to `MetaAnalyzer` |
| **kwargs | | Passed to the engine constructor |

**Example**

```python
# Random effects (default — matches R MetaIntegrator)
analyzer = MetaAnalysisFactory.create_analyzer("random_effects")

# Fixed effects
analyzer = MetaAnalysisFactory.create_analyzer("fixed_effects", max_cores=4)

# List available strategies
print(MetaAnalysisFactory.available_strategies())
# ['random_effects', 'fixed_effects']
```

---

## Analysis Engines

Engines implement the `MetaAnalysisEngine` ABC and define the statistical
algorithm for combining results across studies.

### RandomEffectsEngine (default)

Standard inverse-variance random-effects meta-analysis using DerSimonian–Laird
pooling. This is the default engine and replicates the R `MetaIntegrator`
statistical pipeline.

```python
from metaintegrator.analysis import RandomEffectsEngine
engine = RandomEffectsEngine()
analyzer = MetaAnalyzer(engine=engine)
```

### FixedEffectsEngine

Inverse-variance fixed-effects model. Between-study variance (τ²) is assumed
to be zero.

```python
from metaintegrator.analysis import FixedEffectsEngine
engine = FixedEffectsEngine()
analyzer = MetaAnalyzer(engine=engine)
```

### EffectSizeResult

Typed dataclass returned by `engine.compute_effect_sizes()`. Supports
dict-like access for backward compatibility.

```python
from metaintegrator.analysis import EffectSizeResult

# Fields:
#   g:                pd.DataFrame  — per-study effect-size matrix (genes × studies)
#   se_g:             pd.DataFrame  — per-study standard-error matrix
#   pooled_estimates: pd.DataFrame  — pooled statistics
```

### FilterResult

Typed dataclass returned by `MetaAnalyzer.filter_genes()`. Supports both
attribute access and R-style dict access.

```python
from metaintegrator.analysis import FilterResult

# Attribute access (preferred):
result.pos_gene_names       # List[str]
result.neg_gene_names       # List[str]
result.n_pos                # int
result.n_neg                # int
result.total_genes          # int (property)
result.effect_size_thresh   # float
result.fdr_thresh           # float

# R-style dict access:
result["posGeneNames"]
result["negGeneNames"]
result["FDRThresh"]

# Serialize:
result.to_dict()            # plain dict with R-style keys
```

---

## Related

- [Data Loading](data_loading.md) - Prepare meta-objects
- [Visualization](visualization.md) - Plot results
- [User Guide](../user_guide/statistics.md) - Statistical methods

---

## Complete Workflow Example

```python
from metaintegrator.data_processing import DataProcessor
from metaintegrator.analysis import MetaAnalyzer
from metaintegrator.core_analysis import (
    MetaAnalysisCore,
    FisherCombination
)

# 1. Prepare studies
studies = {
    "GSE1234": DataProcessor.prepare_study_object(...),
    "GSE5678": DataProcessor.prepare_study_object(...),
    "GSE9012": DataProcessor.prepare_study_object(...)
}

meta_object = {"originalData": studies}

# 2. Run meta-analysis
analyzer = MetaAnalyzer(max_cores=4)
results = analyzer.run_meta_analysis(
    meta_object,
    run_leave_one_out=True
)

# 3. Extract results
pooled = results["metaAnalysis"]["pooledResults"]

# 4. Filter significant genes
significant = pooled[
    (pooled["effectSizeFDR"] < 0.05) &
    (pooled["effectSize"].abs() > 0.5)
]

print(f"Significant genes: {len(significant)}")

# 5. Check heterogeneity
heterogeneous = significant[significant["heterogeneityPval"] < 0.05]
print(f"Genes with heterogeneity: {len(heterogeneous)}")

# 6. Examine leave-one-out results
if "leaveOneOutAnalysis" in results:
    loo = results["leaveOneOutAnalysis"]
    
    # Check stability across iterations
    for removed_study, loo_result in loo.items():
        loo_pooled = loo_result["pooledResults"]
        stable_genes = loo_pooled[
            (loo_pooled["effectSizeFDR"] < 0.05) &
            (loo_pooled.index.isin(significant.index))
        ]
        print(f"{removed_study}: {len(stable_genes)} stable genes")
```

---

## Advanced Usage

### Custom Effect Size Calculation

```python
# Calculate effect sizes with custom parameters
for study_name, study in meta_object["originalData"].items():
    # Get effect sizes
    es = MetaAnalysisCore.calculate_effect_sizes(study)
    
    # Filter by sample size
    reliable = es[(es["n1"] >= 10) & (es["n2"] >= 10)]
    
    # Filter by effect size
    large_effects = reliable[reliable["g"].abs() > 0.8]
    
    print(f"{study_name}: {len(large_effects)} large effects")
```

### Comparing Fixed and Random Effects

```python
# Pool with both methods
fixed_results = MetaAnalysisCore.pool_inverse_variance(
    g_matrix, se_matrix, method="fixed"
)

random_results = MetaAnalysisCore.pool_inverse_variance(
    g_matrix, se_matrix, method="random"
)

# Compare
comparison = pd.DataFrame({
    "fixed_es": fixed_results["summary"],
    "random_es": random_results["summary"],
    "difference": (random_results["summary"] - fixed_results["summary"]).abs(),
    "heterogeneity": random_results["pval_het"]
})

# Where do they differ most?
high_diff = comparison.nlargest(10, "difference")
print(high_diff)
```

### Directional Analysis

```python
# Analyze directional consistency
pvals = [FisherCombination.calculate_study_pvalues(s) for s in studies]
fisher = FisherCombination.sum_of_logs(pvals)

# Up-regulated across all studies
up_genes = fisher[
    (fisher["F.pval.up"] < 0.05) &
    (fisher["F.pval.down"] > 0.05)  # Not also down
]

# Down-regulated across all studies  
down_genes = fisher[
    (fisher["F.pval.down"] < 0.05) &
    (fisher["F.pval.up"] > 0.05)  # Not also up
]

# Inconsistent direction
inconsistent = fisher[
    (fisher["F.pval.up"] < 0.05) &
    (fisher["F.pval.down"] < 0.05)
]

print(f"Consistently up: {len(up_genes)}")
print(f"Consistently down: {len(down_genes)}")
print(f"Inconsistent: {len(inconsistent)}")
```

---

## ScoreCalculator

Computes per-sample meta-scores from a gene signature, matching R's `calculateScore`.

```python
from metaintegrator.score_calculation import ScoreCalculator

scores = ScoreCalculator.calculate_score(
    filter_object,     # dict with posGeneNames / negGeneNames
    dataset_object,    # dict with expr, keys, class
    suppress_messages=False,
)
```

**Algorithm:**

1. Shift expression values to positive: `expr + |min(expr)| + 1`
2. Pass **all** signature genes to `SampleLevelGeneDataProcessor` (missing genes filled with 1)
3. Compute geometric mean per sample: `exp(mean(log(values)))`
4. Score = `pos_geometric_mean - neg_geometric_mean`
5. Standardize to z-scores: `(score - mean) / std`

!!! note "R-Parity: All genes passed"
    All gene names are passed regardless of whether they appear in the dataset's keys.
    Missing genes are filled with value 1 (so `log(1) = 0` enters the geometric mean),
    matching R's `calculateScore` behavior. This was a critical fix for AUC parity.

**Returns:** `np.ndarray` of z-scores (one per sample).

```python
# Score and evaluate
scores = ScoreCalculator.calculate_score(filter_object, dataset)

# AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(dataset['class'], scores)
print(f"AUC = {auc:.3f}")
```

---

## SampleLevelGeneDataProcessor

Extracts gene-level expression matrices from probe-level data, handling
multi-probe genes and missing genes.

```python
from metaintegrator.sample_level_gene_data import SampleLevelGeneDataProcessor

gene_matrix = SampleLevelGeneDataProcessor.get_sample_level_gene_data(
    dataset_object,   # dict with expr, keys
    gene_names,       # list of gene symbols to extract
)
# Returns: pd.DataFrame (genes × samples)
```

**Processing steps:**

1. Expand comma-separated keys (`"GENE1,GENE2"` → two rows)
2. Filter to requested genes
3. Group by gene, average across probes
4. Fill missing genes with NaN
5. Replace NaN and zeros with 1
6. Apply `_make_valid_names()` to column names (R compatibility)

---

## MetaIntegratorFilter

Gene filtering by FDR, effect size, and leave-one-out analysis.

```python
from metaintegrator.scoring_meta_integrator_filter import MetaIntegratorFilter

filter_result = MetaIntegratorFilter.filter_genes(
    meta_object,
    fdr_threshold=0.05,
    effect_size_threshold=0,
    number_of_studies=1,
    is_leave_one_out=True,
    heterogeneity_pval_threshold=0,
)

# filter_result contains:
#   posGeneNames: list of up-regulated genes
#   negGeneNames: list of down-regulated genes
```

---

## ROCAnalysisEngine

ROC curve computation and multi-dataset evaluation.

```python
from metaintegrator.roc_analysis_engine import ROCAnalysisEngine

# Single dataset
roc_result = ROCAnalysisEngine.compute_roc(
    y_true=dataset['class'],
    y_scores=scores,
)
print(f"AUC = {roc_result['auc']:.3f}")
```

---

## ScoringOrchestrator

End-to-end pipeline: meta-analysis → filtering → scoring → ROC → export.

```python
from metaintegrator.scoring_meta_integrator_orchestrator import ScoringOrchestrator

orchestrator = ScoringOrchestrator()
results = orchestrator.run_full_pipeline(meta_object)
```

---

## Related

- [Architecture Overview](../architecture.md) — Scoring pipeline diagram
- [Preprocessing API](preprocessing.md) — Data loading before analysis
- [Visualization API](visualization.md) — Plotting results
