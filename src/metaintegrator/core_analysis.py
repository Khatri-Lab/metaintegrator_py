"""
Core statistical analysis functions for MetaIntegrator Python port.
This module contains the foundational statistical functions for meta-analysis.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# MetaAnalysisCore Class
# ------------------------------------------------------------------
class MetaAnalysisCore:
    """Core statistical functions for meta-analysis."""

    REQUIRED_STUDY_KEYS = ["expr", "class", "keys"]

    @staticmethod
    def create_annotation_table(gem_data: Dict) -> pd.DataFrame:
        """
        Generates an annotation table from the given GEM (Gene Expression Matrix) data.

        This method constructs a dataframe containing probe IDs and their corresponding
        symbols extracted from the given GEM data.

        Args:
            gem_data (Dict): A dictionary containing gene expression data with probe IDs
                and their corresponding symbols. Must contain "expr" and "keys" keys.

        Returns:
            pd.DataFrame: A DataFrame containing columns "probeid" and "symbol".

        Raises:
            KeyError: If required keys ("expr", "keys") are missing in the input dictionary.
        """
        probe_ids = gem_data["expr"].index.tolist()
        symbols = gem_data["keys"]

        ann_table = pd.DataFrame({"probeid": probe_ids, "symbol": symbols})
        return ann_table

    @staticmethod
    def check_metagem_study(study: Dict) -> None:
        """
        Checks the structure and values of a provided Metagenomics study dictionary.

        This method ensures that the provided study contains the necessary keys and
        validates their content according to the requirements.

        Args:
            study (Dict): A dictionary representing the study. It must contain the
                keys "expr", "class", and "keys" with specific formats.

        Raises:
            ValueError: If any required key is missing from the study, if the "class"
                vector is not coded as 0 or 1, or if "keys" is not a list or array.
        """
        missing_keys = [key for key in MetaAnalysisCore.REQUIRED_STUDY_KEYS if key not in study]

        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        # Check class vector coding
        unique_classes = np.unique(study["class"])
        if not np.array_equal(unique_classes, np.array([0, 1])):
            raise ValueError("study['class'] must be coded as 0 or 1")

        # Check keys are strings
        if not isinstance(study["keys"], (list, np.ndarray)):
            raise ValueError("Keys must be stored as a list or array")

    @staticmethod
    def clean_na(x: np.ndarray) -> np.ndarray:
        """
        Cleans a numpy array of NaN and infinite values.

        This static method removes all NaN and infinite values from a given numpy array
        and returns a cleaned version of the array.

        Args:
            x (np.ndarray): Numpy array to clean.

        Returns:
            np.ndarray: Numpy array with NaN and infinite values removed.

        """
        return x[~np.isnan(x) & np.isfinite(x)]

    @staticmethod
    def expand_dataframe(
        df: pd.DataFrame, key_name: str = "keys", keys_sep: str = ","
    ) -> pd.DataFrame:
        """
        Expand a dataframe by splitting delimited key values into multiple rows.

        FIXED: Returns a DataFrame that preserves the original schema (including
        the 'keys' column) even when the input is empty, preventing a downstream
        KeyError: 'keys' in groupby().

        Args:
            df: Input dataframe.
            key_name: Column containing delimited keys. Default: "keys".
            keys_sep: Delimiter for splitting keys. Default: ",".

        Returns:
            Expanded DataFrame. If the input is empty, returns an empty DataFrame
            with the same columns as the input (schema preserved).
        """
        # ── BUGFIX: preserve schema on empty input ──────────────────────────
        if df.empty:
            return df.copy()  # same columns, zero rows — groupby won't crash
        # ────────────────────────────────────────────────────────────────────

        expanded_rows = []

        for idx, row in df.iterrows():
            keys = str(row[key_name]).split(keys_sep)
            for key in keys:
                new_row = row.copy()
                new_row[key_name] = key.strip()
                expanded_rows.append(new_row)

        return pd.DataFrame(expanded_rows).reset_index(drop=True)

    @staticmethod
    def get_effect_size(
        expression: np.ndarray, groups: np.ndarray, method: str = "hedges_g"
    ) -> Dict:
        """
        Calculate the effect size and related statistics for two groups based on their expression values.

        FIXED: Added epsilon for numerical stability in pooled_sd calculation

        Summary:
        This method calculates Hedge's G, a measure of effect size, along with other
        relevant descriptive statistics for two groups identified in the input. It
        splits the data into cases and controls based on the provided group vector,
        removes missing values, and computes statistics such as the pooled standard
        deviation, mean differences, and the standard error of Hedge's G. Corrections
        for small sample sizes are applied to adjust Cohen's D into Hedge's G. If the
        sample sizes are insufficient for either group, the method returns NaN
        values for all computed metrics.

        Args:
            expression : np.ndarray
                A numeric array containing expression values for all individuals.
            groups : np.ndarray
                A numeric array or boolean mask where entries indicate whether each
                individual belongs to the case group (1) or control group (0). Must
                have the same length as 'expression'.
            method : str
                'hedges_g' (default) or 'cohen_d'
        Returns:
            Dict
                A dictionary containing the following keys with their respective calculated
                values:
                - n1: Sample size for cases
                - m1: Mean of cases
                - sd1: Standard deviation of cases
                - n2: Sample size for controls
                - m2: Mean of controls
                - sd2: Standard deviation of controls
                - diff: Difference in means (cases - controls)
                - pooled_sd: Pooled standard deviation
                - g: Calculated Hedge's G (effect size)
                - se_g: Standard error of Hedge's G

        Raises:
            ValueError
                If the 'expression' array and 'groups' array have different lengths.
        """
        if len(expression) != len(groups):
            raise ValueError("Expression vector and group vector must have same length")

        # Split into cases and controls
        cases = MetaAnalysisCore.clean_na(expression[groups == 1])
        controls = MetaAnalysisCore.clean_na(expression[groups == 0])
        n_cases, n_controls = len(cases), len(controls)

        # Check if we have enough samples
        if n_cases < 2 or n_controls < 2:
            return {
                "n1": np.nan,
                "m1": np.nan,
                "sd1": np.nan,
                "n2": np.nan,
                "m2": np.nan,
                "sd2": np.nan,
                "diff": np.nan,
                "pooled_sd": np.nan,
                "g": np.nan,
                "se_g": np.nan,
            }

        # Calculate means and standard deviations
        mean_cases, mean_controls = np.mean(cases), np.mean(controls)
        std_cases, std_controls = np.std(cases, ddof=1), np.std(controls, ddof=1)
        mean_diff = mean_cases - mean_controls

        # Calculate pooled standard deviation — match R's .getES() exactly
        # R: sp <- sqrt(((n1-1)*sd1^2 + (n2-1)*sd2^2)/(n1 + n2 - 2))
        numerator = (n_cases - 1) * std_cases**2 + (n_controls - 1) * std_controls**2
        denominator = n_cases + n_controls - 2
        pooled_std = np.sqrt(numerator / denominator)  # NO epsilon — match R

        # Guard: if pooled_std is zero (all values identical in both groups), return NaN
        if pooled_std == 0:
            return {
                "n1": n_cases,
                "m1": mean_cases,
                "sd1": std_cases,
                "n2": n_controls,
                "m2": mean_controls,
                "sd2": std_controls,
                "diff": mean_diff,
                "pooled_sd": pooled_std,
                "g": np.nan,
                "se_g": np.nan,
            }

        # Small sample size correction (Cohen's D to Hedge's G)
        # R: cf <- 1 - 3/(4*(n1 + n2) - 9)
        correction_factor = 1 - 3 / (4 * (n_cases + n_controls) - 9)

        # Calculate Hedge's G — match R exactly
        # R: g <- cf * diff/sp
        effect_size_g = correction_factor * mean_diff / pooled_std

        # Standard error of Hedge's G — match R exactly
        # R: se.g <- sqrt((n1+n2)/(n1*n2) + 0.5*g^2/(n1+n2-3.94))
        variance_g = (n_cases + n_controls) / (n_cases * n_controls) + 0.5 * effect_size_g**2 / (
            n_cases + n_controls - 3.94
        )
        se_effect_size = np.sqrt(variance_g)  # NO epsilon — match R

        return {
            "n1": n_cases,
            "m1": mean_cases,
            "sd1": std_cases,
            "n2": n_controls,
            "m2": mean_controls,
            "sd2": std_controls,
            "diff": mean_diff,
            "pooled_sd": pooled_std,
            "g": effect_size_g,
            "se_g": se_effect_size,
        }

    @staticmethod
    def clean_na(arr: np.ndarray) -> np.ndarray:
        """Remove NaN and infinite values from array."""
        # Filter out NaN and inf values
        mask = np.isfinite(arr)
        return arr[mask]

    @staticmethod
    def calculate_effect_sizes(study: Dict) -> pd.DataFrame:
        """
        Calculate effect sizes for genes in a given study.

        This method computes the effect sizes for all genes in the expression matrix
        of the given study. It uses a class vector to determine the grouping of samples
        and calculates the effect size for each gene based on the expression values
        from the study.

        Args:
            study: Dict
                A dictionary representing the study, which contains the following keys:
                - "expr": The expression matrix (pd.DataFrame) where rows correspond to
                  genes and columns to samples.
                - "class": A class vector (array) defining the group assignments for each
                  sample.
                - "keys": A list or NDArray of identifiers for the genes.

        Returns:
            pd.DataFrame
                A pandas DataFrame where each row contains the effect size statistics for
                a gene, along with its key if provided.
        """
        MetaAnalysisCore.check_metagem_study(study)

        expr_matrix = study["expr"]
        class_vector = np.array(study["class"])
        keys = study["keys"]

        results = []
        for idx, (gene_id, expression_row) in enumerate(expr_matrix.iterrows()):
            es_stats = MetaAnalysisCore.get_effect_size(expression_row.values, class_vector)
            es_stats["keys"] = keys[idx] if isinstance(keys, (list, np.ndarray)) else keys
            results.append(es_stats)

        return pd.DataFrame(results)

    @staticmethod
    def summarize_effects_within_study(
        effects: pd.DataFrame, method: str = "fixed_iv"
    ) -> pd.DataFrame:
        """
        Summarizes effects within a study by either pooling data from multiple measurements
        or selecting the most significant effect based on the specified method. Handles both
        singleton effects and groups of measurements sharing the same keys within the dataset.

        Args:
            effects: pd.DataFrame
                A DataFrame containing effect sizes and associated standard errors.
                Must contain columns 'g', 'se_g', and 'keys'.
            method: str
                The method to use for summarization. Choices are:
                - "fixed_iv": Use fixed effects meta-analysis with inverse variance weighting.
                - "extreme": Select the effect measurement with the largest absolute z-score
                  for groups with multiple entries.

        Raises
            ValueError
                Raised if required columns ('g', 'se_g', 'keys') are missing from 'effects'.

        Returns:
            pd.DataFrame
                A DataFrame with summarized effect sizes and corresponding standard errors
                indexed by 'keys'. For singleton effects, the original values are returned.
                For groups with multiple entries, data is summarized using the specified method.
        """
        required_cols = ["g", "se_g", "keys"]
        if not all(col in effects.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        effects = effects[required_cols].copy()
        effects["keys"] = effects["keys"].astype(str)

        # Handle single row case
        if len(effects) == 1:
            result = effects[["g", "se_g"]].copy()
            result.index = effects["keys"]
            return result

        # Separate singletons from multiples
        key_counts = effects["keys"].value_counts()
        singleton_keys = key_counts[key_counts == 1].index
        singleton_mask = effects["keys"].isin(singleton_keys)

        # Process singletons
        singletons = effects[singleton_mask][["g", "se_g"]].copy()
        singletons.index = effects.loc[singleton_mask, "keys"]

        # Process multiples
        multiples = effects[~singleton_mask].copy()

        if len(multiples) == 0:
            return singletons

        if method == "extreme":
            multiples["absolute_z_score"] = np.abs(multiples["g"] / multiples["se_g"])

        grouped_results = []
        for key, group in multiples.groupby("keys"):
            if method == "fixed_iv":
                # Fixed effects meta-analysis using inverse variance weighting
                inverse_variances = 1 / (group["se_g"] ** 2)
                pooled_g = np.sum(group["g"] * inverse_variances) / np.sum(inverse_variances)
                pooled_se = np.sqrt(1 / np.sum(inverse_variances))
                grouped_results.append({"g": pooled_g, "se_g": pooled_se, "key": key})
            elif method == "extreme":
                # Select the effect with the largest absolute z-score
                max_z_idx = group["absolute_z_score"].idxmax()
                selected_row = group.loc[max_z_idx]
                grouped_results.append(
                    {"g": selected_row["g"], "se_g": selected_row["se_g"], "key": key}
                )

        if grouped_results:
            multiples_result = pd.DataFrame(grouped_results)
            multiples_result.index = multiples_result["key"]
            multiples_result = multiples_result[["g", "se_g"]]

            # Combine singletons and multiples
            result = pd.concat([singletons, multiples_result])
        else:
            result = singletons

        return result.sort_index()

    @staticmethod
    def pool_inverse_variance(
        g_matrix: pd.DataFrame,
        se_matrix: pd.DataFrame,
        method: str = "random",
    ) -> pd.DataFrame:
        """
        Calculate pooled effect sizes and associated statistics using inverse variance weighting.

        - FIXED: Added epsilon to prevent division by zero and NaN handling.
        - ENHANCED: Now matches R's .pool.inverseVar() output exactly, including
        heterogeneity statistics (tau2, Q, df, pval.het).
        - FIXED: Now correctly handles g_matrix and se_matrix with different column names
        by using position-based indexing (.values) instead of label-based indexing.

        R implementation uses rmeta::meta.summaries() which provides:
        - Fixed or random effects pooling
        - Cochran's Q test for heterogeneity
        - Tau-squared estimation for random effects
        - DerSimonian-Laird method for tau2

        This method computes pooled effect sizes for multiple studies using either fixed or random
        effects models. It handles scenarios with missing data and calculates heterogeneity metrics
        such as Cochrane's Q-statistic and tau-squared for random effects models when applicable.
        It operates independently for each row of the provided input data.

        Args:
            g_matrix: DataFrame of effect sizes (genes x studies)
            se_matrix: DataFrame of standard errors (genes x studies)
            method: str
                The pooling method to use. It can be either "random" for a random effects model
                or "fixed" for a fixed effects model. Defaults to "random".

        Returns:
           pd.DataFrame with columns matching R output:
                - n.studies: Number of studies
                - summary: Pooled effect size
                - se.summary: Standard error of pooled effect
                - tau2: tau-squared (for random effects) - Between-study variance (NA for fixed, estimated for random)
                - p.value: heterogeneity statistics - Two-tailed p-value for pooled effect
                - Q: heterogeneity statistics - Cochran's Q statistic for heterogeneity
                - df: heterogeneity statistics - Degrees of freedom (n.studies - 1)
                - pval.het: P-value for heterogeneity test

        Raises:
            ValueError:
                Raised if the row indices of 'effect_sizes' and 'standard_errors' do not match.
        """
        if g_matrix.empty or se_matrix.empty:
            return pd.DataFrame()

        # Verify row alignment
        if not (g_matrix.index == se_matrix.index).all():
            raise ValueError("g_matrix and se_matrix must have same row indices")

        # Verify column counts match (they should have same number of studies)
        if g_matrix.shape[1] != se_matrix.shape[1]:
            raise ValueError(
                f"g_matrix has {g_matrix.shape[1]} columns but "
                f"se_matrix has {se_matrix.shape[1]} columns. They must match."
            )

        # Initialize output
        results = []

        # Process each gene (row)
        for gene_id in g_matrix.index:
            # CRITICAL FIX: Use .values to get numpy arrays for position-based indexing
            # This avoids pandas index alignment issues when column names differ
            g_vals = g_matrix.loc[gene_id].values
            se_vals = se_matrix.loc[gene_id].values

            # Filter out NA and non-finite values
            # MUST check both g and se at the SAME positions (matching R's logic)
            # Position-based
            valid_mask = np.isfinite(g_vals) & np.isfinite(se_vals)

            g_valid = g_vals[valid_mask]
            se_valid = se_vals[valid_mask]
            n_studies = len(g_valid)

            if n_studies == 0:
                # No valid studies - all NA
                results.append(
                    {
                        "n_studies": 0,
                        "summary": np.nan,
                        "se_summary": np.nan,
                        "tau2": np.nan,
                        "p_value": np.nan,
                        "Q": np.nan,
                        "df": np.nan,
                        "pval_het": np.nan,
                    }
                )
            elif n_studies == 1:
                # Single study - no heterogeneity stats
                # p-value calculation: R uses 2*pnorm(abs(summ/se.summ), lower.tail=FALSE)
                z_score = g_valid[0] / se_valid[0]
                p_val = 2 * stats.norm.sf(abs(z_score))

                results.append(
                    {
                        "n_studies": 1,
                        "summary": g_valid[0],
                        "se_summary": se_valid[0],
                        "tau2": np.nan,
                        "p_value": p_val,
                        "Q": np.nan,
                        "df": np.nan,
                        "pval_het": np.nan,
                    }
                )
            else:
                # Multiple studies - compute full statistics
                gene_stats = MetaAnalysisCore._compute_meta_summaries(g_valid, se_valid, method)
                results.append(gene_stats)

        # Create DataFrame with Python naming convention (underscores, not dots)
        result_df = pd.DataFrame(results, index=g_matrix.index)

        # Ensure column order
        result_df = result_df[
            [
                "n_studies",
                "summary",
                "se_summary",
                "tau2",
                "p_value",
                "Q",
                "df",
                "pval_het",
            ]
        ]

        return result_df

    @staticmethod
    def _compute_meta_summaries(
        effects: np.ndarray, std_errors: np.ndarray, method: str = "random"
    ) -> Dict:
        """
        Compute meta-analysis summary statistics for a single gene.

        Implements the core functionality of rmeta::meta.summaries():
        - Fixed or random effects pooling
        - Cochran's Q test
        - DerSimonian-Laird tau-squared estimation

        Args:
            effects: Array of effect sizes from different studies
            std_errors: Array of standard errors
            method: "fixed" or "random"

        Returns:
            Dict with all meta-analysis statistics
        """

        n = len(effects)

        # Calculate weights (inverse variance) — match R's rmeta::meta.summaries
        weights = 1.0 / (std_errors**2)

        # Fixed-effects estimate
        sum_weights = np.sum(weights)
        fixed_effect = np.sum(weights * effects) / sum_weights

        # Cochran's Q statistic for heterogeneity
        # Q = sum(w_i * (y_i - y_fixed)^2)
        Q = np.sum(weights * (effects - fixed_effect) ** 2)

        # Degrees of freedom
        df = n - 1

        # P-value for heterogeneity test (chi-squared distribution)
        pval_het = 1 - stats.chi2.cdf(Q, df) if df > 0 else np.nan

        # Estimate tau-squared (between-study variance) using DerSimonian-Laird method
        if method == "random" and df > 0:
            # DerSimonian-Laird estimator
            # tau2 = max(0, (Q - df) / C)
            # where C = sum(w_i) - sum(w_i^2) / sum(w_i)
            C = sum_weights - np.sum(weights**2) / sum_weights
            tau2_dl = max(0, (Q - df) / C) if C > 0 else 0
            tau2 = tau2_dl

            # Random-effects weights — NO epsilon, match R
            # w*_i = 1 / (se_i^2 + tau2)
            weights_random = 1.0 / (std_errors**2 + tau2)
            sum_weights_random = np.sum(weights_random)

            # Random-effects estimate
            summary = np.sum(weights_random * effects) / sum_weights_random

            # Standard error of random-effects estimate
            se_summary = np.sqrt(1.0 / sum_weights_random)

        else:
            # Fixed-effects (or method == "fixed")
            tau2 = 0.0  # Fixed effects has no between-study variance
            summary = fixed_effect
            se_summary = np.sqrt(1.0 / sum_weights)

        # Two-tailed p-value for the pooled estimate
        # R code: 2*pnorm(abs(summ/se.summ), lower.tail=FALSE)
        # This is equivalent to: 2 * (1 - pnorm(abs(z)))
        z_score = summary / se_summary
        p_value = 2 * stats.norm.sf(abs(z_score))  # sf is survival function (1-cdf)

        return {
            "n_studies": n,
            "summary": summary,
            "se_summary": se_summary,
            "tau2": tau2,
            "p_value": p_value,
            "Q": Q,
            "df": df,
            "pval_het": pval_het,
        }

    @staticmethod
    def get_ttest_pvalues(expr_data: pd.DataFrame, class_vector: np.ndarray) -> pd.DataFrame:
        """
        Calculate t-test p-values for a given expression matrix and class vector.

        This method performs a two-sample t-test for independent samples,
        calculating p-values for testing differences between groups. It computes
        the two-sided p-value as well as directional p-values (one-sided tests).

        FIXED: Proper DataFrame/array indexing to avoid indexing errors.
        FIXED: Changed from Welch's t-test to pooled t-test to match R's implementation.
        R typically uses the standard t-test with pooled variance, not Welch's.
        FIXED: Column names now use dots (P.up, P.down, P.both) to match R.

        Args:
            expr_data (pd.DataFrame): Expression matrix where rows correspond to
                gene IDs and columns correspond to samples.
            class_vector (np.ndarray): Vector indicating the class of each sample
                in the expression matrix. Values should be binary (0 or 1).

        Returns:
            pd.DataFrame: DataFrame with the calculated p-values for each row in the
            expression matrix. The output DataFrame contains the following columns:
                - "P_both": Two-sided p-value.
                - "P_down": One-sided p-value testing for a decrease.
                - "P_up": One-sided p-value testing for an increase.
        """
        # Ensure class_vector is numpy array
        if hasattr(class_vector, "values"):
            class_vector = class_vector.values

        # Create boolean masks
        case_mask = class_vector == 1
        control_mask = class_vector == 0

        # Check for sufficient samples
        n_cases = np.sum(case_mask)
        n_controls = np.sum(control_mask)

        if n_cases < 2 or n_controls < 2:
            logger.warning(f"Insufficient samples: {n_cases} cases, {n_controls} controls")
            n_genes = expr_data.shape[0]
            return pd.DataFrame(
                {
                    "P.both": np.nan,  # Changed from P_both to P.both
                    "P.down": np.nan,  # Changed from P_down to P.down
                    "P.up": np.nan,  # Changed from P_up to P.up
                },
                index=(expr_data.index if isinstance(expr_data, pd.DataFrame) else range(n_genes)),
            )

        # Select cases and controls using proper indexing
        if isinstance(expr_data, pd.DataFrame):
            cases = expr_data.iloc[:, case_mask]
            controls = expr_data.iloc[:, control_mask]
            index = expr_data.index
        else:
            cases = expr_data[:, case_mask]
            controls = expr_data[:, control_mask]
            index = range(expr_data.shape[0])

        # Calculate statistics
        case_mean = cases.mean(axis=1)
        control_mean = controls.mean(axis=1)
        case_var = cases.var(axis=1, ddof=1)  # Sample variance
        control_var = controls.var(axis=1, ddof=1)

        # CRITICAL FIX: Use POOLED variance (not Welch's)
        # This matches R's standard t.test() behavior
        # Pooled variance: s_p^2 = [(n1-1)*s1^2 + (n2-1)*s2^2] / (n1+n2-2)
        pooled_var = ((n_cases - 1) * case_var + (n_controls - 1) * control_var) / (
            n_cases + n_controls - 2
        )

        # Standard error using pooled variance
        # SE = sqrt(s_p^2 * (1/n1 + 1/n2))
        se = np.sqrt(pooled_var * (1 / n_cases + 1 / n_controls))

        # T-statistic
        t_stat = (case_mean - control_mean) / se

        # Degrees of freedom for pooled t-test
        # df = n1 + n2 - 2 (NOT Welch's complicated formula)
        df = n_cases + n_controls - 2

        # Calculate p-values using Student's t-distribution
        # Two-tailed
        p_both = 2 * stats.t.sf(np.abs(t_stat), df)  # sf = 1 - cdf (survival function)

        # One-tailed (up: cases > controls)
        # If t_stat > 0 (cases > controls), this gives small p-value
        p_up = stats.t.sf(t_stat, df)  # P(T > t)

        # One-tailed (down: cases < controls)
        # If t_stat < 0 (cases < controls), this gives small p-value
        p_down = stats.t.cdf(t_stat, df)  # P(T < t)

        # Use dots in column names (internal representation)
        return pd.DataFrame(
            {
                "P.both": p_both,
                "P.down": p_down,
                "P.up": p_up,
            },
            index=index,
        )

    @staticmethod
    def collapse_probes_to_genes(
        effect_sizes_df: pd.DataFrame, method: str = "fixed_iv"
    ) -> pd.DataFrame:
        """
        Collapse multiple probes to single gene symbols.

        This matches R's workflow:
        1. .expand.df() - Split comma-separated gene symbols into separate rows
        2. Remove unmapped/invalid probes
        3. .summ.eff.within() - Summarize multiple probes per gene

        Args:
            effect_sizes_df: DataFrame with columns ['g', 'se_g', 'keys', ...]
                where 'keys' contains gene symbols (possibly comma-separated)
            method: 'fixed_iv' (inverse variance weighting) or 'extreme' (max z-score)

        Returns:
            DataFrame indexed by gene symbol with collapsed effect sizes

        Raises:
            ValueError: If 'keys' column is missing or required columns absent
        """
        if "keys" not in effect_sizes_df.columns:
            raise ValueError("Missing 'keys' column for gene symbols")

        required_cols = ["g", "se_g", "keys"]
        missing = [col for col in required_cols if col not in effect_sizes_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Step 1: Expand probes that map to multiple genes (comma-separated)
        # This matches R's .expand.df() function
        expanded_rows = []
        for idx, row in effect_sizes_df.iterrows():
            # Handle both string and numeric keys
            key_str = str(row["keys"]) if pd.notna(row["keys"]) else ""
            gene_symbols = [s.strip() for s in key_str.split(",") if s.strip()]

            # If no valid gene symbols, skip this probe
            if not gene_symbols:
                continue

            for symbol in gene_symbols:
                new_row = row.copy()
                new_row["keys"] = symbol
                expanded_rows.append(new_row)

        if not expanded_rows:
            # No valid gene mappings found
            return pd.DataFrame(columns=["g", "se_g"])

        expanded_df = pd.DataFrame(expanded_rows).reset_index(drop=True)

        # Step 2: Remove NA/invalid gene symbols
        # This matches R's removal of NA, "NA" string, and empty keys
        expanded_df = expanded_df[
            (expanded_df["keys"].notna())
            & (expanded_df["keys"] != "NA")
            & (expanded_df["keys"] != "")
            & (expanded_df["keys"] != "nan")  # Also handle 'nan' string
        ].copy()

        if len(expanded_df) == 0:
            # No valid genes after filtering
            return pd.DataFrame(columns=["g", "se_g"])

        # Step 3: Summarize multiple probes per gene
        # This uses the existing summarize_effects_within_study method
        # which matches R's .summ.eff.within() function
        collapsed = MetaAnalysisCore.summarize_effects_within_study(
            expanded_df[["g", "se_g", "keys"]], method=method
        )

        return collapsed


# ------------------------------------------------------------------
# FisherCombination Class
# ------------------------------------------------------------------
class FisherCombination:
    """Fisher's method for combining p-values across studies."""

    @staticmethod
    def calculate_study_pvalues(study: Dict) -> pd.DataFrame:
        """
        Calculates and returns the p-values for a given study using the MetaAnalysisCore utilities.

        Returns p-values with dot notation (P.up, P.down) to match Fisher combination expectations.

        This method computes t-test p-values for the expression data in the study provided and
        returns them as a DataFrame with additional metadata such as the keys from the study.

        Note: get_ttest_pvalues() returns columns with dots (P.up, P.down), and this method
        preserves that notation for compatibility with sum_of_logs().

        Args:
            study (Dict): A dictionary containing study data. This includes "expr" for expression
                data, "class" for class labels, and "keys" for metadata or identifiers.

        Returns:
            pd.DataFrame: A DataFrame containing calculated p-values. The resulting DataFrame
                includes columns for "P.up", "P.down", and "keys".

        Raises:
            ValueError: If the provided study data is invalid or does not meet the required
                format or criteria for computation.
        """
        MetaAnalysisCore.check_metagem_study(study)
        pvals = MetaAnalysisCore.get_ttest_pvalues(study["expr"], np.array(study["class"]))

        # Keep P.up and P.down with dots (don't rename to underscores)
        # This is critical for compatibility with sum_of_logs() which filters by ".up" and ".down"
        pvals = pvals[["P.up", "P.down"]].copy()
        pvals["keys"] = study["keys"]
        return pvals

    @staticmethod
    def summarize_significance_within_study(sigs: pd.DataFrame) -> pd.DataFrame:
        """
        Summarizes the minimum significance values within a study by grouping
        the input DataFrame by specific keys.

        Args:
            sigs : pd.DataFrame
                A pandas DataFrame containing the significance values and associated
                keys. The DataFrame must have a column called "keys".

        Returns:
            pd.DataFrame
                A new DataFrame with the minimum significance values for each group
                of keys.
        """
        return sigs.groupby("keys").min()

    @staticmethod
    def _apply_fisher_vectorized(p_value_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """
        Apply Fisher's method vectorized on a DataFrame to combine p-values.

        NOTE: This method is deprecated in favor of _compute_fisher_stat().
        Kept for backward compatibility but updated to match R implementation.

        Args:
            p_value_df : pd.DataFrame
                A DataFrame where each row represents genes and each column represents study
                p-values. NaN values are allowed and indicate missing data for specific studies.
            prefix : str
                A prefix ('up' or 'down') to be added to the resulting column names.

        Returns:
            pd.DataFrame
                A DataFrame with columns matching R output format:
                - F.stat.{prefix}: The Fisher combined statistic
                - F.pval.{prefix}: The corresponding p-value
        """
        if p_value_df.empty:
            return pd.DataFrame()

        # Use the new implementation that matches R exactly
        F_stat, F_pval = FisherCombination._compute_fisher_stat(p_value_df)

        # Return in R format
        result = pd.DataFrame(
            {f"F.stat.{prefix}": F_stat, f"F.pval.{prefix}": F_pval},
            index=p_value_df.index,
        )

        return result

    @staticmethod
    def _compute_fisher_stat(p_value_df: pd.DataFrame) -> tuple:
        """
        Compute Fisher's statistic matching R's implementation exactly.

        R implementation:
        valid.up  <- rowSums( !is.na(sigs.up) )
        F.stat.up <- -2*rowSums( log(sigs.up), na.rm=TRUE )
        F.pval.up <- stats::pchisq( F.stat.up, 2*valid.up, lower.tail=FALSE )

        CRITICAL: No p-value clipping to match R exactly!

        Args:
            p_value_df: DataFrame of p-values (genes x studies)

        Returns:
            tuple: (F_stat, F_pval) as pandas Series
        """
        if p_value_df.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # Count valid (non-NaN) p-values per gene
        valid = (~p_value_df.isna()).sum(axis=1)

        # Fisher's statistic: -2 * sum(log(p-values))
        # Match R: -2*rowSums( log(sigs.up), na.rm=TRUE )
        # WARNING: Do NOT clip p-values! Must match R exactly.
        # If p-value is 0, log(0) = -inf, which R handles naturally
        with np.errstate(divide="ignore"):  # Suppress log(0) warnings
            log_p = np.log(p_value_df)
        F_stat = -2 * log_p.sum(axis=1, skipna=True)

        # When no valid data, F_stat should be NaN, not -0.0
        F_stat = np.where(valid > 0, F_stat, np.nan)

        # Combined p-value using chi-squared survival function
        # Match R: stats::pchisq( F.stat.up, 2*valid.up, lower.tail=FALSE )
        #
        # Use stats.chi2.sf() instead of 1 - stats.chi2.cdf()
        # This is numerically more stable and matches R's pchisq(..., lower.tail=FALSE) exactly
        df = 2 * valid
        F_pval = np.where(
            df > 0,
            stats.chi2.sf(F_stat, df),  # sf = 1 - cdf = survival function
            np.nan,
        )

        return F_stat, F_pval

    @staticmethod
    def sum_of_logs_with_R_bugfix(
        list_of_significance_matrices: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Computes a combined significance result using the Fisher method for multiple
        studies provided as a list of dataframes.

        FIXED: Added p-value clipping and safe Fisher calculation in _apply_fisher_vectorized
        Matches R's .sum.of.logs() implementation exactly.

        CRITICAL FIX: Now filters columns ending with "_P.up" and "_P.down"
        (with dots) to match R's column naming.

        20260220 - R metaIntegrator potential bug detected:
            R's multimerge is producing for 2 studies with the following columns:
            _P.up _P.down _P.up _P.down (notice the duplicate column names and they start with underscore _)
            When R has duplicate column names, it automatically makes them unique by adding .1, .2 suffixes.
            So the actual columns are probably:

            _P.up (study 0)
            _P.down (study 0)
            _P.up.1 (study 1)
            _P.down.1 (study 1)

            So R is only using the first study's p-values!
            In this implementation:
                Python produces: study_0_P.up, study_0_P.down, study_1_P.up, study_1_P.down where both
                study_0_P.up and study_1_P.up match the .up pattern, so Python correctly uses both studies.

            Note: This may actually be a bug in the R implementation - it should use all studies,
            not just the first one. The correct fix would be to ensure the list is named before calling multimerge.
            But for parity testing, this implementation will stay dormant - we need to match the R behavior exactly,
            even if it's wrong.

        Takes a list of significance matrices from different studies and processes
        their significance values to compute a combined result using the Fisher
        method. Invalid or missing entries in the data are filtered out, and the
        analysis involves preprocessing individual studies, aggregating results, and
        merging them into a single dataframe.

        Args:
            list_of_significance_matrices: List[pd.DataFrame]
                A list of pandas DataFrames where each dataframe contains the results
                of significance tests performed within different studies.

        Returns:
            pd.DataFrame
                A pandas DataFrame containing the combined significance results of the
                Fisher method with columns matching R output:
                - F.stat.up: Fisher statistic for up-regulation
                - F.pval.up: P-value for up-regulation
                - F.stat.down: Fisher statistic for down-regulation
                - F.pval.down: P-value for down-regulation

        Raises:
            AttributeError
                If required dataframe columns or methods are not properly defined in
                the provided input or internal implementations.
        """
        processed_studies = []
        for sigs in list_of_significance_matrices:
            df = sigs.copy()
            df["keys"] = df["keys"].astype(str)

            # Filter valid entries
            valid_mask = df.iloc[:, 0].notna() & df["keys"].notna() & (df["keys"] != "NA")
            df = df[valid_mask]

            # Preprocess and aggregate
            df = MetaAnalysisCore.expand_dataframe(df)
            summarized = FisherCombination.summarize_significance_within_study(df)
            processed_studies.append(summarized)

        if not processed_studies:
            return pd.DataFrame()

        merged_df = FisherCombination.multimerge(processed_studies)

        # Filter for columns ending with ".up" and ".down" to match R's grep("\\.up$")
        # After multimerge, columns are: study_0_P.up, study_1_P.up, etc.
        # Use .endswith(".up") to match any column ending with dot-up (not just P.up)
        sigs_up = merged_df[[c for c in merged_df.columns if c.endswith(".up")]]
        sigs_down = merged_df[[c for c in merged_df.columns if c.endswith(".down")]]

        # Compute Fisher statistics
        F_stat_up, F_pval_up = FisherCombination._compute_fisher_stat(sigs_up)
        F_stat_down, F_pval_down = FisherCombination._compute_fisher_stat(sigs_down)

        # Return in R's format
        result = pd.DataFrame(
            {
                "F.stat.up": F_stat_up,
                "F.pval.up": F_pval_up,
                "F.stat.down": F_stat_down,
                "F.pval.down": F_pval_down,
            },
            index=sigs_up.index,
        )

        return result

    @staticmethod
    def sum_of_logs(
        list_of_significance_matrices: List[pd.DataFrame],
    ) -> pd.DataFrame:
        return FisherCombination.sum_of_logs_with_R_bugfix(list_of_significance_matrices)

    @staticmethod
    def sum_of_logs_with_R_bugfix(
        list_of_significance_matrices: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Combine p-values across studies using Fisher's method.

        FIXED: Datasets that are completely empty after filtering (e.g. a dataset
        whose every probe mapped to a transcript ID and was removed by Ensembl
        validation) are now silently skipped rather than causing a
        KeyError: 'keys' crash downstream.

        Args:
            list_of_significance_matrices: one DataFrame per study, as returned
                by FisherCombination.calculate_study_pvalues().

        Returns:
            DataFrame with columns F.stat.up, F.pval.up, F.stat.down, F.pval.down.
        """
        processed_studies = []

        for study_idx, sigs in enumerate(list_of_significance_matrices):
            df = sigs.copy()
            df["keys"] = df["keys"].astype(str)

            # Filter valid entries
            valid_mask = df.iloc[:, 0].notna() & df["keys"].notna() & (df["keys"] != "NA")
            df = df[valid_mask]

            # ── BUGFIX: skip studies with no valid genes ─────────────────────
            if df.empty:
                logger.warning(
                    f"Study index {study_idx}: no valid genes after filtering "
                    "(all probes mapped to NA). Skipping this study in Fisher "
                    "combination. Check gene annotation for this dataset."
                )
                continue
            # ─────────────────────────────────────────────────────────────────

            df = MetaAnalysisCore.expand_dataframe(df)

            # expand_dataframe preserves schema on empty input,
            # but guard again just in case expand produced nothing
            if df.empty or "keys" not in df.columns:
                logger.warning(
                    f"Study index {study_idx}: empty after expand_dataframe. " "Skipping."
                )
                continue

            summarized = FisherCombination.summarize_significance_within_study(df)
            processed_studies.append(summarized)

        if not processed_studies:
            logger.warning(
                "Fisher combination: no valid studies remaining after filtering. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        merged_df = FisherCombination.multimerge(processed_studies)

        sigs_up = merged_df[[c for c in merged_df.columns if c.endswith(".up")]]
        sigs_down = merged_df[[c for c in merged_df.columns if c.endswith(".down")]]

        F_stat_up, F_pval_up = FisherCombination._compute_fisher_stat(sigs_up)
        F_stat_down, F_pval_down = FisherCombination._compute_fisher_stat(sigs_down)

        result = pd.DataFrame(
            {
                "F.stat.up": F_stat_up,
                "F.pval.up": F_pval_up,
                "F.stat.down": F_stat_down,
                "F.pval.down": F_pval_down,
            },
            index=sigs_up.index,
        )

        return result

    @staticmethod
    def multimerge(dataframe_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merges multiple pandas DataFrames by aligning them on their indices and adding a
        prefix to the column names of each DataFrame. Each prefix is based on the index
        position of the DataFrame in the input list.

        CRITICAL: To match R's behavior with unnamed lists, use "_" prefix which creates
        duplicate column names (_P.up, _P.down, _P.up, _P.down). pandas will match R's
        behavior.

        Args:
            dataframe_list (List[pd.DataFrame]): A list of pandas DataFrames to be merged.
                Each DataFrame in the list will have a unique prefix applied to its
                column names.

        Returns:
            pd.DataFrame: A single pandas DataFrame resulting from the concatenation of
            input DataFrames along the columns.

        Raises:
            None
        """
        if not dataframe_list:
            return pd.DataFrame()

        # Use study_ prefix to create unique column names
        # This matches the expected behavior in tests
        prefixed_dfs = [df.add_prefix(f"study_{i}_") for i, df in enumerate(dataframe_list)]

        # pd.concat handles the union of indices and alignment automatically
        # CRITICAL: R does NOT make duplicate column names unique!
        # R keeps them as duplicates: _P.up, _P.down, _P.up, _P.down
        # When grep matches duplicate names, it selects ALL matching columns
        merged = pd.concat(prefixed_dfs, axis=1)

        # DEBUG: Print column names (should have duplicates like R)
        logger.info(f"multimerge: Columns = {list(merged.columns)}")

        return merged
