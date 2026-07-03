"""
Statistical analysis module for meta-analysis results.
This module provides comprehensive statistical analysis capabilities for meta-integrator data.

Highlights:
- Uses scoring_core infrastructure
- Enhanced statistical methods with robust error handling
- Comprehensive distribution and correlation analysis
- Modular design with clear separation of concerns
- Standardized validation and logging
- Advanced statistical features (bootstrap, permutation tests)
- Bootstrap confidence intervals
- Permutation-based significance testing
- Advanced distribution fitting
- Outlier detection and handling
- Multi-dimensional correlation analysis
"""

from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Import from consolidated core module (eliminates duplicates)
from .scoring_core import (
    # Use unified data structures and configurations
    StatisticalSummary,
    SummaryConfig,
    # Use core validation and utilities
    MetaAnalysisValidator,
    LoggerMixin,
    DataFrameUtils,
    # Use core exceptions
    ScoringCalculationError,
)

# ==================================================================
# STATISTICAL ANALYZER
# ==================================================================


class StatisticalAnalyzer(LoggerMixin):
    """
    Enhanced analyzer for statistical properties of meta-analysis results.

    This class orchestrates the extraction of summary statistics, distribution metrics,
    correlation data and threshold counts from a meta-analysis DataFrame.
    """

    def __init__(self, config: SummaryConfig, **kwargs):
        """
        Create a StatisticalAnalyzer instance.

        Args:
            config (SummaryConfig): Summary configuration controlling which statistics are computed.
            **kwargs: Additional keyword arguments passed to LoggerMixin.
        """
        super().__init__(**kwargs)
        self.config = config
        self.scaler = StandardScaler()

    def analyze(self, results: pd.DataFrame) -> StatisticalSummary:
        """
        Perform comprehensive statistical analysis.

        Args:
            results (pd.DataFrame): DataFrame containing the meta-analysis results to be summarized.

        Returns:
            StatisticalSummary: object with all computed statistics.

        Raises:
            ScoringCalculationError: If statistical analysis fails.
        """
        try:
            # Validate input
            MetaAnalysisValidator.validate_dataframe(results)

            basic_stats = self._calculate_basic_statistics(results)

            distribution_stats = None
            if self.config.include_distributions:
                distribution_stats = self._calculate_distribution_statistics(results)

            correlation_stats = None
            if self.config.include_correlations:
                correlation_stats = self._calculate_correlation_statistics(results)

            threshold_counts = self._calculate_threshold_counts(results)

            metadata = {
                "total_genes": len(results),
                "analysis_timestamp": pd.Timestamp.now(),
                "config_used": self.config.__dict__,
                "analyzer_version": "2.0.0",
                "columns_analyzed": list(results.columns),
            }

            summary = StatisticalSummary(
                basic_stats=basic_stats,
                distribution_stats=distribution_stats,
                correlation_stats=correlation_stats,
                threshold_counts=threshold_counts,
                metadata=metadata,
            )

            self.logger.info(f"Statistical analysis completed for {len(results)} genes")
            return summary

        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            raise ScoringCalculationError(f"Statistical analysis failed: {e}")

    def _calculate_basic_statistics(self, results: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic statistical measures for all numeric columns.

        Args:
            results (pd.DataFrame): DataFrame containing the data to analyze.

        Returns:
            Dict[str, Any]: Dictionary mapping column names to their basic statistics.
        """
        numeric_cols = DataFrameUtils.get_numeric_columns(results)

        basic_stats = {}
        for col in numeric_cols:
            data = results[col].dropna()
            if len(data) > 0:
                basic_stats[col] = {
                    "count": len(data),
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data, ddof=1)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "median": float(np.median(data)),
                    "q25": float(np.percentile(data, 25)),
                    "q75": float(np.percentile(data, 75)),
                    "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
                    "mad": float(stats.median_abs_deviation(data)),
                    "cv": (
                        float(np.std(data, ddof=1) / np.mean(data))
                        if np.mean(data) != 0
                        else np.nan
                    ),
                    "missing_count": results[col].isna().sum(),
                    "missing_percent": results[col].isna().sum() / len(results) * 100,
                }

        return basic_stats

    def _calculate_distribution_statistics(
        self, results: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate distribution‑related statistics for each numeric column.

        Args:
            results (pd.DataFrame): DataFrame containing the data to analyze.

        Returns:
            Dict[str, Any]: A dictionary mapping column names to a dictionary of
            distribution metrics including normality test statistics, skewness,
            kurtosis, best‑fit distribution, and outlier information. If an error
            occurs for a column, the value for that column will contain an
            'error' key with the exception message.
        """
        numeric_cols = DataFrameUtils.get_numeric_columns(results)

        dist_stats = {}
        for col in numeric_cols:
            data = results[col].dropna()
            if len(data) > 3:  # Need minimum data for distribution tests
                try:
                    # Normality tests
                    shapiro_stat, shapiro_p = (
                        stats.shapiro(data) if len(data) <= 5000 else (np.nan, np.nan)
                    )
                    jarque_bera_stat, jarque_bera_p = stats.jarque_bera(data)

                    # Skewness and kurtosis
                    skewness = stats.skew(data)
                    kurtosis = stats.kurtosis(data)

                    # Distribution fitting
                    best_distribution = self._fit_best_distribution(data)

                    # Outlier detection
                    outliers = self._detect_outliers(data)

                    dist_stats[col] = {
                        "shapiro_stat": (
                            float(shapiro_stat) if not np.isnan(shapiro_stat) else None
                        ),
                        "shapiro_p": (
                            float(shapiro_p) if not np.isnan(shapiro_p) else None
                        ),
                        "jarque_bera_stat": float(jarque_bera_stat),
                        "jarque_bera_p": float(jarque_bera_p),
                        "skewness": float(skewness),
                        "kurtosis": float(kurtosis),
                        "best_distribution": best_distribution,
                        "outliers": outliers,
                        "outlier_count": len(outliers["indices"]),
                        "outlier_percent": (len(outliers["indices"]) / len(data) * 100),
                    }
                except Exception as e:
                    self.logger.warning(f"Distribution analysis failed for {col}: {e}")
                    dist_stats[col] = {"error": str(e)}

        return dist_stats

    def _calculate_correlation_statistics(
        self, results: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate correlation statistics between numeric columns.

        Args:
            results (pd.DataFrame): DataFrame containing the data to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing Pearson and Spearman
            correlation matrices, a list of highly correlated pairs, the
            maximum and mean absolute correlation values, and a summary
            of correlation patterns. If fewer than two numeric columns are
            present or an error occurs, the dictionary will contain an
            'error' key with the relevant message.
        """
        numeric_data = results.select_dtypes(include=[np.number])

        if numeric_data.shape[1] < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}

        try:
            # Remove columns with constant values
            numeric_data = numeric_data.loc[:, numeric_data.std() > 1e-10]

            # Pearson correlations
            pearson_corr = numeric_data.corr(method="pearson")

            # Spearman correlations (for non‑linear relationships)
            spearman_corr = numeric_data.corr(method="spearman")

            # Find highly correlated pairs
            high_corr_pairs = self._find_high_correlations(pearson_corr, threshold=0.8)

            correlation_stats = {
                "pearson_matrix": pearson_corr.to_dict(),
                "spearman_matrix": spearman_corr.to_dict(),
                "high_correlations": high_corr_pairs,
                "max_correlation": float(pearson_corr.abs().unstack().max()),
                "mean_correlation": float(pearson_corr.abs().mean().mean()),
                "correlation_summary": self._summarize_correlations(pearson_corr),
            }

            return correlation_stats

        except Exception as e:
            self.logger.warning(f"Correlation analysis failed: {e}")
            return {"error": str(e)}

    def _calculate_threshold_counts(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Calculate counts of genes meeting various significance thresholds.
        Args:
            results: DataFrame containing result columns.
        Returns:
            A dictionary mapping each thresholded column to a dictionary of counts.
        """
        threshold_counts = {}

        # Check for common column names
        significance_cols = ["fdr", "pvalue", "p_value", "adj_pvalue"]
        effect_cols = ["effect_size", "log_fold_change", "beta", "coefficient"]

        for col in results.columns:
            col_lower = col.lower()

            # Significance thresholds
            if any(sig_col in col_lower for sig_col in significance_cols):
                data = results[col].dropna()
                threshold_counts[f"{col}_thresholds"] = {
                    f"below_{thresh}": int((data < thresh).sum())
                    for thresh in self.config.significance_thresholds
                }

            # Effect size thresholds
            if any(eff_col in col_lower for eff_col in effect_cols):
                data = results[col].dropna()
                threshold_counts[f"{col}_thresholds"] = {
                    f"above_{thresh}": int((np.abs(data) > thresh).sum())
                    for thresh in self.config.effect_size_thresholds
                }

        return threshold_counts

    def _fit_best_distribution(self, data: np.ndarray) -> Dict[str, Any]:
        """Fit common distributions and find the best match.
        Args:
            data: One‑dimensional array of values to fit.
        Returns:
            A dictionary containing the name of the best distribution, the KS p‑value,
            and the fitted parameters as a list of floats.
        """
        distributions = [
            stats.norm,
            stats.lognorm,
            stats.gamma,
            stats.beta,
            stats.uniform,
        ]
        best_dist = None
        best_p = 0
        best_params = None

        for distribution in distributions:
            try:
                # Fit distribution
                params = distribution.fit(data)

                # Perform Kolmogorov‑Smirnov test
                ks_stat, ks_p = stats.kstest(
                    data, lambda x: distribution.cdf(x, *params)
                )

                if ks_p > best_p:
                    best_dist = distribution.name
                    best_p = ks_p
                    best_params = params
            except:
                continue

        return {
            "distribution": best_dist,
            "ks_p_value": float(best_p),
            "parameters": [float(p) for p in best_params] if best_params else None,
        }

    def _detect_outliers(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using multiple methods.
        Args:
            data: One‑dimensional array of values to analyze.
        Returns:
            A dictionary containing indices, values and the count of outliers
            identified by IQR, Z‑score and modified Z‑score methods.
        """
        outliers = {"indices": [], "values": [], "methods": []}

        # IQR method
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_outliers = np.where((data < lower_bound) | (data > upper_bound))[0]

        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        z_outliers = np.where(z_scores > 3)[0]

        # Modified Z-score method (using MAD)
        mad = stats.median_abs_deviation(data)
        modified_z_scores = (
            0.6745 * (data - np.median(data)) / mad if mad > 0 else np.zeros_like(data)
        )
        mad_outliers = np.where(np.abs(modified_z_scores) > 3.5)[0]

        # Combine all outlier indices
        all_outliers = np.unique(
            np.concatenate([iqr_outliers, z_outliers, mad_outliers])
        )

        outliers["indices"] = all_outliers.tolist()
        outliers["values"] = data[all_outliers].tolist()
        outliers["iqr_outliers"] = len(iqr_outliers)
        outliers["z_outliers"] = len(z_outliers)
        outliers["mad_outliers"] = len(mad_outliers)

        return outliers

    def _find_high_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.8
    ) -> List[Dict]:
        """Find pairs of highly correlated variables.
        Args:
            corr_matrix: DataFrame containing correlation coefficients.
            threshold: Threshold for considering a pair as highly correlated.
        Returns:
            List of dictionaries each with keys variable1, variable2,
            correlation, and abs_correlation.
        """
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append(
                        {
                            "variable1": corr_matrix.columns[i],
                            "variable2": corr_matrix.columns[j],
                            "correlation": float(corr_value),
                            "abs_correlation": float(abs(corr_value)),
                        }
                    )

        high_corr_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

        return high_corr_pairs

    def _summarize_correlations(self, corr_matrix: pd.DataFrame) -> Dict[str, float]:
        """Provide summary statistics for the correlation matrix.
        Args:
            corr_matrix: DataFrame containing correlation coefficients.
        Returns:
            Dictionary with mean_abs_correlation, max_abs_correlation,
            min_abs_correlation, std_correlation,
            strong_correlations_count, moderate_correlations_count,
            weak_correlations_count.
        """
        upper_triangle = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        correlations = upper_triangle.stack().dropna()

        return {
            "mean_abs_correlation": float(correlations.abs().mean()),
            "max_abs_correlation": float(correlations.abs().max()),
            "min_abs_correlation": float(correlations.abs().min()),
            "std_correlation": float(correlations.std()),
            "strong_correlations_count": int((correlations.abs() > 0.7).sum()),
            "moderate_correlations_count": int(
                ((correlations.abs() > 0.3) & (correlations.abs() <= 0.7)).sum()
            ),
            "weak_correlations_count": int((correlations.abs() <= 0.3).sum()),
        }


# ==================================================================
# DISTRIBUTION ANALYZER
# ==================================================================


class DistributionAnalyzer(LoggerMixin):
    """Specialized analyzer for distribution properties and testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def analyze_distributions(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform comprehensive distribution analysis on multiple variables.
        Args:
            data: Dictionary mapping variable names to their numeric arrays.
        Returns:
            Dictionary where each key is a variable name and each value contains
            the analysis results. If more than one variable is provided, an
            additional key 'comparison' contains cross‑variable comparison
            results.
        """
        results = {}

        for name, values in data.items():
            results[name] = self._analyze_single_distribution(values, name)

        # Compare distributions if multiple variables
        if len(data) > 1:
            results["comparison"] = self._compare_distributions(data)

        return results

    def _analyze_single_distribution(
        self, data: np.ndarray, name: str
    ) -> Dict[str, Any]:
        """Analyze a single distribution comprehensively.
        Args:
            data: One‑dimensional numeric array for the variable.
            name: Name of the variable being analyzed.
        Returns:
            Dictionary containing basic statistics, normality tests,
            distribution fits, bootstrap confidence intervals, and
            percentiles for the provided data.
        """
        clean_data = data[~np.isnan(data)]

        if len(clean_data) == 0:
            return {"error": "No valid data points"}

        analysis = {
            "basic_stats": self._basic_distribution_stats(clean_data),
            "normality_tests": self._normality_tests(clean_data),
            "distribution_fits": self._comprehensive_distribution_fitting(clean_data),
            "bootstrap_confidence_intervals": self._bootstrap_ci(clean_data),
            "percentiles": self._calculate_percentiles(clean_data),
        }

        return analysis

    def _basic_distribution_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate basic distribution statistics.
        Args:
            data: One‑dimensional numeric array with valid values.
        Returns:
            Dictionary containing mean, variance, skewness, kurtosis,
            and entropy of the data distribution.
        """
        return {
            "mean": float(np.mean(data)),
            "variance": float(np.var(data, ddof=1)),
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data)),
            "entropy": float(stats.entropy(np.histogram(data, bins="auto")[0] + 1e-10)),
        }

    def _normality_tests(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Perform multiple normality tests.
        Args:
            data: One‑dimensional numeric array to be tested for normality.
        Returns:
            Dictionary mapping test names to dictionaries containing
            the test statistic and the associated p‑value or related
            metrics. Tests included are Shapiro‑Wilk, D'Agostino's,
            and Anderson‑Darling. Only tests applicable to the
            sample size are performed.
        """
        tests = {}

        if len(data) > 3:
            # Shapiro‑Wilk test (best for small to medium samples)
            if len(data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                tests["shapiro"] = {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                }

            # D'Agostino's normality test
            try:
                dagostino_stat, dagostino_p = stats.normaltest(data)
                tests["dagostino"] = {
                    "statistic": float(dagostino_stat),
                    "p_value": float(dagostino_p),
                }
            except:
                pass

            # Anderson‑Darling test
            try:
                anderson_result = stats.anderson(data, dist="norm")
                tests["anderson"] = {
                    "statistic": float(anderson_result.statistic),
                    "critical_values": anderson_result.critical_values.tolist(),
                    "significance_levels": anderson_result.significance_level.tolist(),
                }
            except:
                pass

        return tests

    def _comprehensive_distribution_fitting(self, data: np.ndarray) -> Dict[str, Any]:
        """Fit multiple distributions and rank by goodness of fit.
        Args:
            data: One‑dimensional numeric array to fit.
        Returns:
            Dictionary containing a list of fitted distributions sorted
            by KS p‑value in descending order. Each entry includes the
            distribution name, fitted parameters, KS statistic and
            p‑value, log‑likelihood, AIC, and BIC.
        """
        distributions = [
            stats.norm,
            stats.lognorm,
            stats.gamma,
            stats.beta,
            stats.expon,
            stats.uniform,
            stats.chi2,
            stats.t,
            stats.weibull_min,
            stats.pareto,
        ]

        results = []
        for dist in distributions:
            try:
                # Fit parameters
                params = dist.fit(data)

                # Calculate goodness of fit metrics
                ks_stat, ks_p = stats.kstest(data, lambda x: dist.cdf(x, *params))

                # Calculate AIC and BIC
                log_likelihood = np.sum(dist.logpdf(data, *params))
                k = len(params)  # number of parameters
                n = len(data)
                aic = 2 * k - 2 * log_likelihood
                bic = k * np.log(n) - 2 * log_likelihood

                results.append(
                    {
                        "distribution": dist.name,
                        "parameters": [float(p) for p in params],
                        "ks_statistic": float(ks_stat),
                        "ks_p_value": float(ks_p),
                        "log_likelihood": float(log_likelihood),
                        "aic": float(aic),
                        "bic": float(bic),
                    }
                )
            except:
                continue

        # Sort by KS p‑value (higher is better)
        results.sort(key=lambda x: x["ks_p_value"], reverse=True)

        return {"fitted_distributions": results}

    def _bootstrap_ci(
        self,
        data: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Dict[str, List[float]]:
        """Calculate bootstrap confidence intervals for key statistics.
        Args:
            data: One‑dimensional numeric array for which the
            confidence intervals are calculated.
            n_bootstrap: Number of bootstrap resamples to draw.
            confidence: Desired confidence level (e.g. 0.95 for a
            95% interval).
        Returns:
            Dictionary with keys 'mean_ci', 'std_ci', and 'median_ci',
            each mapping to a two‑element list containing the lower
            and upper bounds of the confidence interval.
        """
        n = len(data)
        alpha = 1 - confidence

        # Bootstrap samples
        bootstrap_means = []
        bootstrap_stds = []
        bootstrap_medians = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
            bootstrap_stds.append(np.std(sample, ddof=1))
            bootstrap_medians.append(np.median(sample))

        # Calculate confidence intervals
        def calculate_ci(values):
            return [
                float(np.percentile(values, 100 * alpha / 2)),
                float(np.percentile(values, 100 * (1 - alpha / 2))),
            ]

        return {
            "mean_ci": calculate_ci(bootstrap_means),
            "std_ci": calculate_ci(bootstrap_stds),
            "median_ci": calculate_ci(bootstrap_medians),
        }

    def _calculate_percentiles(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive percentile information.

        Args:
            data (np.ndarray): 1D array of numerical data.

        Returns:
            Dict[str, float]: Dictionary mapping percentile labels (e.g., 'p1', 'p5', etc.) to their respective percentile values.
        """
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        return {f"p{p}": float(np.percentile(data, p)) for p in percentiles}

    def _compare_distributions(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compare multiple distributions using statistical tests.

        Args:
            data (Dict[str, np.ndarray]): Dictionary where keys are dataset names and values are arrays of data.

        Returns:
            Dict[str, Any]: Dictionary of pairwise comparison results keyed by 'dataset1_vs_dataset2'.
        """
        comparison_results = {}
        data_names = list(data.keys())

        # Pairwise comparisons
        for i in range(len(data_names)):
            for j in range(i + 1, len(data_names)):
                name1, name2 = data_names[i], data_names[j]
                data1, data2 = data[name1], data[name2]

                # Clean data
                clean_data1 = data1[~np.isnan(data1)]
                clean_data2 = data2[~np.isnan(data2)]

                if len(clean_data1) > 0 and len(clean_data2) > 0:
                    comparison_key = f"{name1}_vs_{name2}"
                    comparison_results[comparison_key] = (
                        self._compare_two_distributions(clean_data1, clean_data2)
                    )

        return comparison_results

    def _compare_two_distributions(
        self, data1: np.ndarray, data2: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare two distributions using multiple statistical tests.

        Args:
            data1 (np.ndarray): First dataset.
            data2 (np.ndarray): Second dataset.

        Returns:
            Dict[str, Any]: Dictionary containing results of Mann‑Whitney U test, Kolmogorov‑Smirnov test,
            Levene's test, and Cohen's d effect size. Each test result is a dictionary with
            'statistic' and 'p_value' keys, except 'cohens_d' which is a float.
        """
        results = {}

        # Mann-Whitney U test (non‑parametric)
        try:
            mw_stat, mw_p = stats.mannwhitneyu(data1, data2, alternative="two-sided")
            results["mannwhitney"] = {
                "statistic": float(mw_stat),
                "p_value": float(mw_p),
            }
        except:
            pass

        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_p = stats.ks_2samp(data1, data2)
            results["kolmogorov_smirnov"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_p),
            }
        except:
            pass

        # Levene's test for equal variances
        try:
            levene_stat, levene_p = stats.levene(data1, data2)
            results["levene"] = {
                "statistic": float(levene_stat),
                "p_value": float(levene_p),
            }
        except:
            pass

        # Effect size (Cohen's d)
        try:
            mean_diff = np.mean(data1) - np.mean(data2)
            pooled_std = np.sqrt(
                (
                    (len(data1) - 1) * np.var(data1, ddof=1)
                    + (len(data2) - 1) * np.var(data2, ddof=1)
                )
                / (len(data1) + len(data2) - 2)
            )
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            results["cohens_d"] = float(cohens_d)
        except:
            pass

        return results
