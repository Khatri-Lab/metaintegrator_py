# Statistical Methods

This guide provides detailed explanations of the statistical methods used in metaIntegrator-py, including mathematical formulas, implementation details, and interpretation guidelines.

## Contents

- [Introduction](#introduction)
- [Effect Size Calculation](#effect-size-calculation)
- [Meta-Analysis Methods](#meta-analysis-methods)
- [P-value Combination](#p-value-combination)
- [Multiple Testing Correction](#multiple-testing-correction)
- [Heterogeneity Assessment](#heterogeneity-assessment)
- [Advanced Topics](#advanced-topics)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Appendix](#appendix)

## Introduction

MetaIntegrator-py implements rigorous statistical methods for combining gene expression data across multiple studies. Understanding these methods helps you interpret results correctly and make informed analysis decisions.

**Key statistical components:**
1. **Effect sizes**: Quantify magnitude of gene expression changes
2. **Meta-analysis**: Combine effect sizes across studies
3. **P-value combination**: Integrate statistical significance
4. **FDR correction**: Control false discovery rate
5. **Heterogeneity**: Assess variability across studies

## Effect Size Calculation

### Hedges' g Statistic

**Definition**: Bias-corrected standardized mean difference

**Formula**:
```
Hedges' g = J × Cohen's d

where:
  d = (M₁ - M₂) / SD_pooled
  
  SD_pooled = √[((n₁-1)×SD₁² + (n₂-1)×SD₂²) / (n₁+n₂-2)]
  
  J = 1 - 3/(4(n₁+n₂) - 9)  [correction factor]
```

**Implementation**:

```python
from metaintegrator.core_analysis import MetaAnalysisCore
import numpy as np

def calculate_hedges_g_manual(cases, controls):
    """
    Calculate Hedges' g manually (for illustration).
    """
    n1, n2 = len(cases), len(controls)
    m1, m2 = np.mean(cases), np.mean(controls)
    sd1, sd2 = np.std(cases, ddof=1), np.std(controls, ddof=1)
    
    # Pooled standard deviation
    pooled_sd = np.sqrt(
        ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
    )
    
    # Cohen's d
    cohens_d = (m1 - m2) / pooled_sd
    
    # Correction factor
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    
    # Hedges' g
    hedges_g = J * cohens_d
    
    # Standard error
    se_g = np.sqrt(
        (n1 + n2) / (n1 * n2) + 
        (hedges_g**2) / (2 * (n1 + n2 - 3.94))
    )
    
    return {
        'g': hedges_g,
        'se_g': se_g,
        'n1': n1,
        'n2': n2,
        'm1': m1,
        'm2': m2,
        'pooled_sd': pooled_sd
    }

# Example
cases = np.random.normal(8.0, 1.5, 25)      # 25 cases, mean=8
controls = np.random.normal(7.0, 1.5, 25)   # 25 controls, mean=7

result = calculate_hedges_g_manual(cases, controls)
print(f"Hedges' g = {result['g']:.3f} ± {result['se_g']:.3f}")
```

### Interpreting Effect Sizes

| Hedges' g | Interpretation | Clinical Relevance |
|-----------|----------------|-------------------|
| \|g\| < 0.2 | Negligible | Unlikely to be clinically meaningful |
| 0.2 ≤ \|g\| < 0.5 | Small | May be clinically relevant |
| 0.5 ≤ \|g\| < 0.8 | Medium | Likely clinically relevant |
| \|g\| ≥ 0.8 | Large | Strong clinical relevance |

**Direction:**
- **Positive g**: Gene up-regulated in cases
- **Negative g**: Gene down-regulated in cases

### Confidence Intervals

95% confidence interval for Hedges' g:

```
CI = g ± 1.96 × SE_g
```

```python
def calculate_ci(g, se_g, confidence=0.95):
    """Calculate confidence interval for effect size."""
    from scipy import stats
    
    # Critical value (1.96 for 95% CI)
    z_critical = stats.norm.ppf((1 + confidence) / 2)
    
    ci_lower = g - z_critical * se_g
    ci_upper = g + z_critical * se_g
    
    return (ci_lower, ci_upper)

# Example
g, se_g = 0.65, 0.18
ci = calculate_ci(g, se_g)
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
# If CI doesn't include 0, effect is significant at p < 0.05
```

## Meta-Analysis Methods

### Random-Effects Model (Recommended)

**Model assumptions:**
- True effect sizes vary across studies
- Studies sampled from population of studies
- Accounts for between-study heterogeneity

**Formula**:
```
Pooled estimate = Σ(w_i* × g_i) / Σ(w_i*)

where:
  w_i* = 1 / (SE_i² + τ²)  [random-effects weight]
  τ² = between-study variance (DerSimonian-Laird estimator)
```

**Implementation**:

```python
from metaintegrator.core_analysis import MetaAnalysisCore
import pandas as pd

# Effect sizes from 3 studies
g_matrix = pd.DataFrame({
    'Study_A': [1.2, 0.8, -0.5],
    'Study_B': [1.1, 0.9, -0.4],
    'Study_C': [1.3, 0.7, -0.6]
}, index=['Gene1', 'Gene2', 'Gene3'])

se_matrix = pd.DataFrame({
    'Study_A': [0.3, 0.25, 0.20],
    'Study_B': [0.28, 0.26, 0.22],
    'Study_C': [0.32, 0.24, 0.21]
}, index=['Gene1', 'Gene2', 'Gene3'])

# Pool using random-effects
pooled = MetaAnalysisCore.pool_inverse_variance(
    g_matrix=g_matrix,
    se_matrix=se_matrix,
    method='random'  # Random-effects (default)
)

print(pooled)
# Output columns:
#   n_studies, summary, se_summary, tau2, p_value, Q, df, pval_het
```

### Fixed-Effects Model

**Model assumptions:**
- All studies estimate same true effect
- Between-study variability = 0
- Only within-study variance

**Formula**:
```
Pooled estimate = Σ(w_i × g_i) / Σ(w_i)

where:
  w_i = 1 / SE_i²  [fixed-effects weight]
```

**When to use:**
- Studies very homogeneous (same population, platform, protocol)
- Low heterogeneity (I² < 25%)
- Generally **NOT recommended** for gene expression meta-analysis

### DerSimonian-Laird Tau² Estimator

Between-study variance (τ²):

```
τ² = max(0, (Q - df) / C)

where:
  Q = Cochran's Q statistic
  df = k - 1 (k = number of studies)
  C = Σw_i - (Σw_i²) / Σw_i
```

```python
def estimate_tau2(effects, variances):
    """
    Estimate between-study variance (tau²).
    """
    # Fixed-effects weights
    weights = 1 / variances
    
    # Fixed-effects estimate
    fixed_estimate = np.sum(weights * effects) / np.sum(weights)
    
    # Cochran's Q
    Q = np.sum(weights * (effects - fixed_estimate)**2)
    
    # Degrees of freedom
    df = len(effects) - 1
    
    # C constant
    C = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
    
    # Tau²
    tau2 = max(0, (Q - df) / C) if C > 0 else 0
    
    return tau2, Q, df

# Example
effects = np.array([1.2, 1.1, 1.3])
variances = np.array([0.09, 0.078, 0.102])  # SE² 

tau2, Q, df = estimate_tau2(effects, variances)
print(f"τ² = {tau2:.4f}, Q = {Q:.2f}, df = {df}")
```

## P-value Combination

### Fisher's Method

Combines p-values across studies using log transformation:

**Formula**:
```
χ² = -2 × Σ log(p_i)

df = 2k  (k = number of studies)

P-value = P(χ² > observed value | df)
```

**Implementation**:

```python
from metaintegrator.core_analysis import FisherCombination
import numpy as np
from scipy import stats

def fisher_combine_pvalues(p_values):
    """
    Combine p-values using Fisher's method.
    """
    # Remove NaN values
    p_clean = p_values[~np.isnan(p_values)]
    
    # Fisher statistic
    chi2_stat = -2 * np.sum(np.log(p_clean))
    
    # Degrees of freedom
    df = 2 * len(p_clean)
    
    # Combined p-value
    combined_p = stats.chi2.sf(chi2_stat, df)
    
    return {
        'F_stat': chi2_stat,
        'df': df,
        'p_value': combined_p
    }

# Example
p_values = np.array([0.01, 0.03, 0.02])  # 3 studies
result = fisher_combine_pvalues(p_values)
print(f"Combined p-value: {result['p_value']:.2e}")
```

**Properties:**
- More powerful than simple p-value averaging
- Sensitive to very small p-values
- Assumes p-values are independent
- Works well even with unequal sample sizes

### Stouffer's Z-Score Method (Alternative)

Alternative p-value combination:

```
Z = Σ(Z_i × √w_i) / √Σw_i

where:
  Z_i = Φ⁻¹(1 - p_i)  [inverse normal transform]
  w_i = weight (typically sample size)
```

```python
def stouffer_combine(p_values, weights=None):
    """
    Combine p-values using Stouffer's Z-score method.
    """
    from scipy import stats
    
    # Convert p-values to Z-scores
    z_scores = stats.norm.ppf(1 - p_values)
    
    # Equal weights if not provided
    if weights is None:
        weights = np.ones(len(p_values))
    
    # Combined Z-score
    z_combined = np.sum(z_scores * np.sqrt(weights)) / np.sqrt(np.sum(weights))
    
    # Combined p-value
    p_combined = 1 - stats.norm.cdf(z_combined)
    
    return p_combined
```

## Multiple Testing Correction

### Benjamini-Hochberg FDR

Controls False Discovery Rate (proportion of false positives):

**Procedure:**
1. Rank p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
2. Find largest i where: p_i ≤ (i/m) × α
3. Reject hypotheses 1 through i

**Formula for adjusted p-values:**
```
FDR_i = min(1, p_i × m / i)
```

**Implementation**:

```python
def benjamini_hochberg_fdr(p_values, alpha=0.05):
    """
    Calculate FDR-adjusted p-values (Benjamini-Hochberg).
    """
    import pandas as pd
    
    # Number of tests
    m = len(p_values)
    
    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # Calculate FDR
    ranks = np.arange(1, m + 1)
    fdr = np.minimum(1, sorted_p * m / ranks)
    
    # Ensure monotonicity
    for i in range(m - 2, -1, -1):
        fdr[i] = min(fdr[i], fdr[i + 1])
    
    # Unsort to match original order
    fdr_original_order = np.empty(m)
    fdr_original_order[sorted_idx] = fdr
    
    # Identify significant genes
    is_significant = fdr_original_order <= alpha
    
    return {
        'fdr': fdr_original_order,
        'significant': is_significant,
        'n_significant': is_significant.sum()
    }

# Example
p_values = np.array([0.001, 0.01, 0.02, 0.05, 0.1, 0.5])
result = benjamini_hochberg_fdr(p_values, alpha=0.05)

print("P-value  FDR      Significant")
for p, fdr, sig in zip(p_values, result['fdr'], result['significant']):
    print(f"{p:.3f}    {fdr:.3f}    {'✅' if sig else '  '}")
```

### Bonferroni Correction (Alternative)

More conservative than FDR:

```
Adjusted p-value = p × m
Significance threshold = α / m
```

**When to use:**
- Very strict control of false positives needed
- Small number of tests (< 100)
- **Not recommended** for genome-wide analysis (too conservative)

## Heterogeneity Assessment

### I² Statistic

Percentage of variability due to heterogeneity (not chance):

**Formula**:
```
I² = max(0, 100% × (Q - df) / Q)

where:
  Q = Cochran's Q statistic
  df = k - 1 (k = number of studies)
```

**Interpretation**:

| I² | Heterogeneity Level | Interpretation |
|----|---------------------|----------------|
| 0-25% | Low | Studies are homogeneous |
| 25-50% | Moderate | Some heterogeneity present |
| 50-75% | Substantial | Considerable heterogeneity |
| 75-100% | Considerable | Very high heterogeneity |

**Implementation**:

```python
def calculate_i_squared(Q, df):
    """
    Calculate I² statistic for heterogeneity.
    """
    i_squared = max(0, 100 * (Q - df) / Q) if Q > 0 else 0
    
    # Interpretation
    if i_squared < 25:
        level = "Low"
    elif i_squared < 50:
        level = "Moderate"
    elif i_squared < 75:
        level = "Substantial"
    else:
        level = "Considerable"
    
    return {
        'I2': i_squared,
        'level': level
    }

# Example (from pooled results)
Q, df = 12.5, 2  # From Cochran's Q test
result = calculate_i_squared(Q, df)
print(f"I² = {result['I2']:.1f}% ({result['level']} heterogeneity)")
```

### Cochran's Q Test

Tests null hypothesis: all studies have same true effect

**Formula**:
```
Q = Σ w_i(g_i - ḡ)²

where:
  w_i = 1 / SE_i²
  ḡ = pooled estimate (fixed-effects)
```

**Test**:
```
Q ~ χ²(df = k-1)
P-value = P(χ² > Q | df = k-1)
```

**Interpretation:**
- **p < 0.05**: Significant heterogeneity (use random-effects)
- **p ≥ 0.05**: No significant heterogeneity (could use fixed-effects)

## Advanced Topics

### Leave-One-Out Analysis

Assess robustness by sequentially excluding each study:

```python
def leave_one_out_analysis(g_matrix, se_matrix):
    """
    Perform leave-one-out sensitivity analysis.
    """
    results = []
    
    for i, excluded_study in enumerate(g_matrix.columns):
        # Exclude one study
        g_loo = g_matrix.drop(columns=[excluded_study])
        se_loo = se_matrix.drop(columns=[excluded_study])
        
        # Pool remaining studies
        pooled = MetaAnalysisCore.pool_inverse_variance(
            g_loo, se_loo, method='random'
        )
        
        # Store results
        for gene in pooled.index:
            results.append({
                'gene': gene,
                'excluded': excluded_study,
                'pooled_es': pooled.loc[gene, 'summary'],
                'p_value': pooled.loc[gene, 'p_value']
            })
    
    return pd.DataFrame(results)
```

### Publication Bias Assessment

Funnel plot asymmetry test (Egger's test):

```python
def egger_test(effects, se):
    """
    Test for funnel plot asymmetry (publication bias).
    """
    from scipy import stats
    
    # Precision (1/SE)
    precision = 1 / se
    
    # Regression: effect ~ precision
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        precision, effects
    )
    
    # Significant intercept suggests asymmetry
    return {
        'intercept': intercept,
        'p_value': p_value,
        'significant_bias': p_value < 0.05
    }
```

### Meta-Regression

Model heterogeneity sources as covariates:

```python
# Pseudo-code (would require specialized implementation)
def meta_regression(effects, se, covariate):
    """
    Meta-regression with study-level covariate.
    
    Example covariate: sample size, mean age, publication year
    """
    # Weight by inverse variance
    weights = 1 / (se ** 2)
    
    # Weighted regression
    # effect_i = β₀ + β₁×covariate_i + ε_i
    
    # (Full implementation would use statsmodels or similar)
    pass
```

## FAQ

### Q: Why use Hedges' g instead of Cohen's d?

**A:** Hedges' g corrects for small-sample bias. With n < 20 per group, Cohen's d overestimates effect size. Hedges' g provides unbiased estimate regardless of sample size.

### Q: What's the difference between fixed and random effects?

**A:**

| Aspect | Fixed-Effects | Random-Effects |
|--------|---------------|----------------|
| Assumption | One true effect | Distribution of effects |
| Weights | 1/SE² | 1/(SE² + τ²) |
| CI | Narrower | Wider (more conservative) |
| Use when | Homogeneous studies | Heterogeneous studies |

**Recommendation**: Use random-effects for gene expression meta-analysis.

### Q: When is Fisher's method more powerful than meta-analysis of effect sizes?

**A:** Fisher's method is particularly powerful when:
- Studies have very different designs/platforms
- Effect sizes difficult to compare
- Some studies don't report effect sizes
- Interested only in significance, not magnitude

### Q: What FDR threshold should I use?

**A:** Common thresholds:
- **FDR < 0.05**: Standard (5% false discoveries expected)
- **FDR < 0.01**: Stringent (1% false discoveries)
- **FDR < 0.10**: Exploratory (10% false discoveries)

**Recommendation**: FDR < 0.05 for most analyses.

## Troubleshooting

### Issue: Very high heterogeneity (I² > 90%)

**Possible causes:**
- Different disease subtypes across studies
- Platform/batch effects
- Different sample populations

**Solutions:**
- Use random-effects (default)
- Investigate heterogeneity sources
- Consider subgroup analysis
- Check data quality

### Issue: No genes significant after FDR correction

**Causes:**
- Insufficient power (small studies, few studies)
- True null (no real differences)
- Excessive heterogeneity

**Solutions:**
- Combine more studies
- Use less stringent threshold (FDR < 0.10)
- Check nominal significance (p < 0.05)
- Review study quality

## Appendix

### Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| g | Hedges' g (effect size) |
| d | Cohen's d |
| SE | Standard error |
| τ² | Between-study variance (tau-squared) |
| I² | Heterogeneity percentage |
| Q | Cochran's Q statistic |
| df | Degrees of freedom |
| p | P-value |
| α | Significance level (typically 0.05) |
| FDR | False discovery rate |
| m | Number of tests/genes |
| k | Number of studies |
| n₁, n₂ | Sample sizes for groups 1 and 2 |
| M₁, M₂ | Means for groups 1 and 2 |
| SD | Standard deviation |
| w | Weight |

### Statistical Software Comparison

| Feature | MetaIntegrator-py | R MetaIntegrator | METAL | metafor (R) |
|---------|-------------------|------------------|-------|-------------|
| Effect size meta-analysis | ✅ | ✅ | ✅ | ✅ |
| Fisher's method | ✅ | ✅ | ✅ | ❌ |
| Random-effects | ✅ | ✅ | ✅ | ✅ |
| FDR correction | ✅ | ✅ | ✅ | ✅ |
| Heterogeneity stats | ✅ | ✅ | ✅ | ✅ |
| Meta-regression | ❌ | ❌ | ❌ | ✅ |

### References

**Statistical Methods:**
- DerSimonian R, Laird N (1986). "Meta-analysis in clinical trials." *Controlled Clinical Trials*
- Hedges LV (1981). "Distribution theory for Glass's estimator of effect size." *J Educational Statistics*
- Benjamini Y, Hochberg Y (1995). "Controlling the false discovery rate." *J Royal Statistical Society B*

**Meta-Analysis in Genomics:**
- Tseng GC, et al. (2012). "Comprehensive literature review and statistical considerations for microarray meta-analysis." *Nucleic Acids Research*
- Ramasamy A, et al. (2008). "Key issues in conducting a meta-analysis of gene expression microarray datasets." *PLoS Medicine*

---

**Last updated:** 2026-04-21  
**Version:** 1.0  
**Maintainer:** MetaIntegrator-py Team
