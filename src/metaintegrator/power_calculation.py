"""
Statistical power calculation functions for MetaIntegrator Python port.
This module contains functions to calculate power for random effects meta-analysis.

Based on the paper by Valentine et al.:
JC Valentine, TD Pigott, and HR Rothstein.
How Many Studies Do You Need? A Primer on Statistical Power for Meta-Analysis
J of Educational and Behavioral Statistics
April 2010 Vol 35, No 2, pp 215-247

The code is adapted from the R implementation in MetaIntegrator and
Dan Quintana's blog post on Towards Data Science, July 2017.
"""

import warnings
from dataclasses import dataclass
from typing import Union, List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------
# PowerAnalysisResult Class
# ------------------------------------------------------------------
@dataclass
class PowerAnalysisResult:
    """Results from power analysis calculation."""

    power: float
    effect_size: float
    avg_sample_size: float
    num_studies: int
    heterogeneity: float
    tail: int
    variance_within: float
    variance_between: float
    total_variance: float
    lambda_stat: float


class MetaAnalysisPowerCalculator:
    """Class for calculating statistical power of random effects meta-analysis."""

    # Predefined heterogeneity levels
    HETEROGENEITY_LEVELS = {"small": 0.33, "moderate": 1.0, "large": 3.0}

    def __init__(self):
        pass

    def calc_meta_power(
        self,
        effect_size: float,
        avg_sample_size: float,
        num_studies: int,
        heterogeneity: Union[float, str],
        tail: int = 2,
        alpha: float = 0.05,
    ) -> PowerAnalysisResult:
        """
        Calculate the statistical power of a random effects meta-analysis.

        Performs the statistical power calculation for a random-effects meta-analysis taking
        into account effect size, average sample size per study, number of studies, and
        heterogeneity among studies. The method supports computation for one-tailed or
        two-tailed hypothesis tests and allows the specification of an alpha level.

        Args:
            effect_size (float): The estimated average effect size for the meta-analysis.
            avg_sample_size (float): The average sample size per study.
            num_studies (int): The total number of studies in the meta-analysis.
            heterogeneity (Union[float, str]): Either a numeric heterogeneity value (tau-squared)
                or a string label corresponding to predefined heterogeneity levels.
            tail (int, optional): Indicates whether the test is one-tailed (1) or two-tailed (2).
                Default is 2 for a two-tailed test.
            alpha (float, optional): The significance level for the hypothesis test. Default is 0.05.

        Returns:
            PowerAnalysisResult: Includes fields for computed statistical power, effect size,
            average sample size, heterogeneity, variance components, and lambda statistic.

        Note:
        avg_sample_size is the average number of people in each group in each study.
        For example, if you have 4 studies, and each study compared 10 cases and
        10 controls, then avg_sample_size = 10.

        This code does not take multiple hypothesis testing into account.

        Examples:
            >>> calc = MetaAnalysisPowerCalculator()
            >>> result = calc.calc_meta_power(effect_size=0.7, avg_sample_size=10,
            ...                              num_studies=5, heterogeneity='small')
            >>> print(f"Power: {result.power:.3f}")
        """
        # 1. Validation and Parameter Resolution
        self._validate_inputs(effect_size, avg_sample_size, num_studies, heterogeneity, tail, alpha)
        het_val = self._resolve_heterogeneity(heterogeneity)
        n = avg_sample_size

        # 2. Variance Calculations (Valentine et al. Equations)
        # Sampling variance for each study (within-study)
        var_within = (2 / n) + (effect_size**2 / (4 * n))

        # Tau-squared (between-study variance)
        var_between = het_val * var_within

        # Typical sampling variance of random effects estimate (v*)
        var_total = var_between + var_within

        # Variance associated with weighted mean effect size (v*_circle)
        var_weighted_mean = var_total / num_studies

        # 3. Power Estimation
        lambda_stat = effect_size / np.sqrt(var_weighted_mean)
        critical_val = self._get_critical_value(tail, alpha)

        power = 1 - stats.norm.cdf(critical_val - lambda_stat)

        return PowerAnalysisResult(
            power=power,
            effect_size=effect_size,
            avg_sample_size=avg_sample_size,
            num_studies=num_studies,
            heterogeneity=het_val,
            tail=tail,
            variance_within=var_within,
            variance_between=var_between,
            total_variance=var_total,
            lambda_stat=lambda_stat,
        )

    def _resolve_heterogeneity(self, heterogeneity: Union[float, str]) -> float:
        """
        Resolves the heterogeneity parameter into a numeric value.

        If a string is provided, it is mapped to a predefined numeric level
        (e.g., 'small', 'moderate', 'large'). If a float is provided, it is
        returned directly.

        Args:
            heterogeneity (Union[float, str]): A numeric heterogeneity value or a
                descriptive string label.

        Returns:
            float: The numeric representation of the heterogeneity.

        Raises:
            ValueError: If a string is provided that does not match one of the
                predefined HETEROGENEITY_LEVELS keys.
        """
        if isinstance(heterogeneity, str):
            label = heterogeneity.lower()
            if label not in self.HETEROGENEITY_LEVELS:
                raise ValueError(
                    f"Invalid level. Must be one of {list(self.HETEROGENEITY_LEVELS.keys())}"
                )
            return self.HETEROGENEITY_LEVELS[label]
        return heterogeneity

    def _get_critical_value(self, tail: int, alpha: float) -> float:
        """
        Calculates the critical value from the standard normal distribution.

        This value corresponds to the point on the distribution beyond which the
        null hypothesis is rejected, adjusted for the number of tails and
        significance level.

        Note: For alpha=0.05, uses hardcoded values matching R's implementation
        for exact parity (R uses 1.96 and 1.64 instead of calculated values).

        Args:
            tail (int): The number of tails for the statistical test (1 or 2).
            alpha (float): The significance level (type I error rate).

        Returns:
            float: The z-score critical value.
        """
        # Use hardcoded values for alpha=0.05 to match R's calculatePower.R's calcMetaPower exactly
        # R uses: 1.96 for two-tailed, 1.64 for one-tailed
        if alpha == 0.05:
            if tail == 2:
                return 1.96  # Matches R's hardcoded value
            elif tail == 1:
                return 1.64  # Matches R's hardcoded value

        # For other alpha values, calculate exactly
        return stats.norm.ppf(1 - alpha / tail)

    def _validate_inputs(
        self,
        effect_size: float,
        avg_sample_size: float,
        num_studies: int,
        heterogeneity: Union[float, str],
        tail: int,
        alpha: float,
    ) -> None:
        """
        Validates the input parameters for correctness and consistency.

        This function ensures that all supplied parameters meet the required validation
        criteria, including positive values, range-bound constraints, and categorical
        limitations. It is primarily used to enforce input consistency before further
        processing.

        Args:
            effect_size (float): The expected magnitude of the effect being measured.
                Must be strictly positive.
            avg_sample_size (float): The average sample size per study. Must be strictly
                positive.
            num_studies (int): The number of studies included in the analysis. Must be
                strictly positive.
            heterogeneity (Union[float, str]): Describes the variability in effect sizes
                among studies. If given as a number, it must be a non-negative value.
            tail (int): Specifies whether the test is one-tailed (1) or two-tailed (2).
            alpha (float): The significance level for hypothesis testing. Must be within
                the exclusive range (0, 1).
        """
        # Validate strictly positive parameters
        params_to_check = {
            "Effect size": effect_size,
            "Average sample size": avg_sample_size,
            "Number of studies": num_studies,
        }
        for name, val in params_to_check.items():
            self._check_positive(name, val)

        # Validate heterogeneity
        if isinstance(heterogeneity, (int, float)) and heterogeneity < 0:
            raise ValueError("Heterogeneity must be non-negative")

        # Validate categorical and range-bound parameters
        if tail not in {1, 2}:
            raise ValueError("Tail must be 1 (one-tailed) or 2 (two-tailed)")

        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1 (exclusive)")

    def _check_positive(self, name: str, value: float) -> None:
        """
        Checks if a value is positive and raises an error if not.

        This helper method validates that the provided value is greater than zero
        and raises a ValueError otherwise. It is particularly useful for enforcing
        constraints on numerical values.

        Args:
            name: str
                A string representing the name of the value being checked; typically
                the parameter or attribute name. Used in the error message.
            value: float
                The numerical value to be validated as positive.

        Raises:
            ValueError
                If the provided value is less than or equal to zero.
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    def power_analysis_grid(
        self,
        effect_sizes: List[float],
        sample_sizes: List[float],
        num_studies_list: List[int],
        heterogeneity: Union[float, str] = "moderate",
        tail: int = 2,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Performs power analysis over a grid of parameters for meta-analysis studies.

        This function computes statistical power using combinations of input parameters for
        meta-analysis studies. It iterates over the provided combinations of effect sizes,
        sample sizes, and the number of studies, and calculates the power and other related
        statistics for each set of parameters.

        Args:
            effect_sizes: List[float]
                A list of effect size values to include in the power analysis.
            sample_sizes: List[float]
                A list of average sample sizes to include in the power analysis.
            num_studies_list: List[int]
                A list of the number of studies to include in the power analysis.
            heterogeneity: Union[float, str]
                The level of heterogeneity to assume in the calculations. Can be specified as a
                numeric value or as "low", "moderate", or "high". Default is "moderate".
            tail: int
                Specifies whether to use a one-tailed or two-tailed test. Acceptable values are
                1 (one-tailed) or 2 (two-tailed). Default is 2.
            alpha: float
                The significance level to be used in the power analysis. Default is 0.05.

        Returns:
            pd.DataFrame
                A DataFrame containing the results of the power analysis. The DataFrame includes
                the following columns:
                  - effect_size: The effect size corresponding to a given result.
                  - avg_sample_size: The average sample size corresponding to a given result.
                  - num_studies: The number of studies considered in the computation.
                  - heterogeneity: The heterogeneity value or label used in the calculation.
                  - power: The statistical power result for the given parameter combination.
                  - lambda_stat: The lambda statistic associated with the power calculation.

        Notes:
            This function wraps over the calc_meta_power method to perform the actual power
            calculations for each parameter combination. If an error occurs during calculation for
            any set of parameters, the error is logged to the console, but the process continues
            for the remaining parameter combinations.
        """
        import itertools

        results = []

        parameter_grid = itertools.product(effect_sizes, sample_sizes, num_studies_list)

        for es, n, ns in parameter_grid:
            try:
                result = self.calc_meta_power(es, n, ns, heterogeneity, tail, alpha)
                results.append(
                    {
                        "effect_size": es,
                        "avg_sample_size": n,
                        "num_studies": ns,
                        "heterogeneity": result.heterogeneity,
                        "power": result.power,
                        "lambda_stat": result.lambda_stat,
                    }
                )
            except Exception as e:
                print(f"Error calculating power for es={es}, n={n}, ns={ns}: {str(e)}")

        return pd.DataFrame(results)

    def find_required_studies(
        self,
        effect_size: float,
        avg_sample_size: float,
        target_power: float = 0.8,
        heterogeneity: Union[float, str] = "moderate",
        tail: int = 2,
        alpha: float = 0.05,
        max_studies: int = 100,
    ) -> Dict:
        """
        Calculate the minimum number of studies required to achieve the target statistical power for a meta-analysis.

        This method iteratively calculates the statistical power for an increasing number
        of studies until the target power is achieved or the maximum allowed number of studies
        is reached. If the target power is not achieved within the maximum limit, a warning is
        issued. The method also returns additional details, such as power curve and study
        counts, for analysis.

        Args:
            effect_size: A numeric value representing the expected true effect size.
            avg_sample_size: A numeric value specifying the average sample size per study.
            target_power: Desired statistical power of the meta-analysis, default is 0.8.
            heterogeneity: A numeric or string value denoting heterogeneity in the studies.
                Default is "moderate".
            tail: An integer indicating whether to perform a one-tailed (1) or two-tailed (2)
                analysis. Default is 2 (two-tailed).
            alpha: A float representing the significance level for the analysis. Default is 0.05.
            max_studies: Maximum number of studies to examine for achieving the target power.
                Default is 100.

        Returns:
            A dictionary containing:
                - required_studies (int or None): The minimum number of studies required to
                  achieve the target power, or None if not achieved within max_studies.
                - target_power (float): The desired target power.
                - achieved_power (float or None): The highest power obtained, or None if no
                  calculations were performed.
                - study_counts (list[int]): List of study counts for which power was calculated.
                - power_curve (list[float]): List of power values corresponding to study_counts.

        Raises:
            None
        """
        study_counts = list(range(1, max_studies + 1))
        powers = []
        required_studies = None

        for num_studies in study_counts:
            result = self.calc_meta_power(
                effect_size, avg_sample_size, num_studies, heterogeneity, tail, alpha
            )
            powers.append(result.power)

            if result.power >= target_power:
                required_studies = num_studies
                break

        if required_studies is None:
            warnings.warn(f"Target power {target_power} not achieved within {max_studies} studies.")

        return {
            "required_studies": required_studies,
            "target_power": target_power,
            "achieved_power": powers[-1] if powers else None,
            "study_counts": study_counts[: len(powers)],
            "power_curve": powers,
        }

    def find_required_sample_size(
        self,
        effect_size: float,
        num_studies: int,
        target_power: float = 0.8,
        heterogeneity: Union[float, str] = "moderate",
        tail: int = 2,
        alpha: float = 0.05,
        max_sample_size: int = 1000,
    ) -> Dict:
        """
        Finds the required sample size to achieve a specified statistical power for
        meta-analysis studies.

        This function iteratively calculates the statistical power for increasing
        sample sizes until the desired target power is reached or the maximum
        sample size is exceeded. The method relies on a calculation of meta-analytic
        power for various levels of heterogeneity and other given parameters.

        Args:
            effect_size : float
                The expected true effect size for the meta-analysis.
            num_studies : int
                The total number of independent studies included in the meta-analysis.
            target_power : float, optional
                The desired statistical power to achieve (default is 0.8).
            heterogeneity : Union[float, str], optional
                The level of between-study heterogeneity. Can be a numerical value
                or a string such as "low", "moderate", or "high" (default is "moderate").
            tail : int, optional
                Specifies whether to use a one-tailed (1) or two-tailed (2) test (default is 2).
            alpha : float, optional
                The significance level for the hypothesis test (default is 0.05).
            max_sample_size : int, optional
                The maximum sample size to test for achieving the target power
                (default is 1000).

        Returns:
            Dict
                A dictionary containing the following keys:
                - required_sample_size: The sample size needed to achieve the target power,
                  or None if the target power could not be achieved.
                - target_power: The specified target power.
                - achieved_power: The power achieved at the maximum sample size tested,
                  or None if no power calculations were performed.
                - sample_sizes: A list of sample sizes tested.
                - power_curve: A list of calculated power values corresponding to the tested
                  sample sizes.

        Raises:
            The function does not explicitly raise exceptions, but a warning
            is issued if the target power is not achieved within the specified
            maximum sample size.
        """
        sample_sizes = list(range(2, max_sample_size + 1))
        powers = []
        required_sample_size = None

        for sample_size in sample_sizes:
            result = self.calc_meta_power(
                effect_size, sample_size, num_studies, heterogeneity, tail, alpha
            )
            powers.append(result.power)

            if result.power >= target_power:
                required_sample_size = sample_size
                break

        if required_sample_size is None:
            warnings.warn(
                f"Target power {target_power} not achieved within {max_sample_size} samples per group."
            )

        return {
            "required_sample_size": required_sample_size,
            "target_power": target_power,
            "achieved_power": powers[-1] if powers else None,
            "sample_sizes": sample_sizes[: len(powers)],
            "power_curve": powers,
        }

    def find_detectable_effect_size(
        self,
        avg_sample_size: float,
        num_studies: int,
        target_power: float = 0.8,
        heterogeneity: Union[float, str] = "moderate",
        tail: int = 2,
        alpha: float = 0.05,
        max_effect_size: float = 5.0,
    ) -> Dict:
        """
        Calculates the minimum detectable effect size that achieves the target power while considering
        meta-analysis parameters like average sample size, number of studies, and heterogeneity.

        Args:
            avg_sample_size: float
                Average sample size per study in the meta-analysis.
            num_studies: int
                Total number of studies included in the meta-analysis.
            target_power: float, optional
                The desired statistical power level (default is 0.8).
            heterogeneity: Union[float, str], optional
                Measure of heterogeneity for the meta-analysis. Acceptable values include a float to directly
                specify heterogeneity or a predefined string level of heterogeneity (default is "moderate").
            tail: int, optional
                Specifies the number of tails for hypothesis testing (default is 2).
            alpha: float, optional
                Significance level for hypothesis testing (default is 0.05).
            max_effect_size: float, optional
                Upper limit of the effect size range to search for the minimum detectable effect size
                (default is 5.0).

        Returns:
            Dict
                A dictionary containing:
                    - "min_detectable_effect_size": Minimum detected effect size achieving the target power.
                    - "target_power": The target power value.
                    - "achieved_power": The achieved power corresponding to the calculated effect size.
                    - "effect_sizes": Array of effect sizes evaluated.
                    - "power_curve": Array of statistical power values corresponding to the effect sizes.

        Raises:
            None

        Warns:
            RuntimeWarning
                If the target power is not achieved within the given maximum effect size range.
        """
        effect_sizes = np.linspace(0.01, max_effect_size, 500)
        powers = []
        min_detectable_es = None

        for effect_size in effect_sizes:
            result = self.calc_meta_power(
                effect_size, avg_sample_size, num_studies, heterogeneity, tail, alpha
            )
            powers.append(result.power)

            if result.power >= target_power:
                min_detectable_es = effect_size
                break

        if min_detectable_es is None:
            warnings.warn(
                f"Target power {target_power} not achieved with effect size up to {max_effect_size}"
            )

        return {
            "min_detectable_effect_size": min_detectable_es,
            "target_power": target_power,
            "achieved_power": powers[-1] if powers else None,
            "effect_sizes": effect_sizes[: len(powers)],
            "power_curve": powers,
        }


# ------------------------------------------------------------------
# PowerAnalysisPlotter Class
# ------------------------------------------------------------------
class PowerAnalysisPlotter:
    """Class for plotting power analysis results."""

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Represents the initialization of an object with a customizable figure size.

        Args:
            figsize (Tuple[int, int]): A tuple representing the dimensions of the figure
                in inches. Defaults to (10, 6).
        """
        self.figsize = figsize

    def plot_power_curve(
        self,
        power_data: Dict,
        x_var: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plots a power analysis curve based on the provided data and configuration.

        This function generates a plot illustrating the statistical power curve
        for given data on a user-defined x-axis variable. It also incorporates
        target lines for power and required values if provided. The plot
        can be saved to a specified path if desired.

        Args:
            power_data (Dict): A dictionary containing the data required for plotting the power curve,
                               which includes values such as statistical power for different conditions,
                               target power levels, and additional context-specific data.
            x_var (str): A key identifying the x-axis variable. It should match one of
                         the predefined categories ('studies', 'sample_size', or 'effect_size') for configuration.
            title (Optional[str]): Custom title for the plot. If not specified, a default title
                                   based on the x-axis variable will be used.
            save_path (Optional[str]): File path where the plot will be saved. The file is
                                       saved as an image in high resolution if specified.

        Returns:
            plt.Figure: A matplotlib figure object representing the generated power analysis plot.

        Raises:
            ValueError: If the specified x_var does not match any predefined configuration keys.

        """
        # Configuration mapping for different x-axis variables
        configs = {
            "studies": {
                "data_key": "study_counts",
                "label": "Number of Studies",
                "target_key": "required_studies",
                "target_label": "Required Studies",
            },
            "sample_size": {
                "data_key": "sample_sizes",
                "label": "Sample Size per Group",
                "target_key": "required_sample_size",
                "target_label": "Required Sample Size",
            },
            "effect_size": {
                "data_key": "effect_sizes",
                "label": "Effect Size",
                "target_key": "min_detectable_effect_size",
                "target_label": "Min Detectable Effect Size",
            },
        }

        if x_var not in configs:
            raise ValueError(f"x_var must be one of {list(configs.keys())}")

        cfg = configs[x_var]
        x_data = power_data[cfg["data_key"]]
        y_data = power_data["power_curve"]
        target_val = power_data.get(cfg["target_key"])
        target_power = power_data.get("target_power", 0.8)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot core power curve and target lines
        ax.plot(x_data, y_data, "b-", linewidth=2, label="Statistical Power")
        ax.axhline(
            target_power,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Target Power ({target_power})",
        )

        # Add vertical line for required value
        if target_val is not None:
            ax.axvline(
                target_val,
                color="green",
                linestyle=":",
                alpha=0.7,
                label=f"{cfg['target_label']}: {target_val}",
            )

        # Formatting
        ax.set_xlabel(cfg["label"])
        ax.set_ylabel("Statistical Power")
        ax.set_title(title or f"Power Analysis: {cfg['label']}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_power_grid(
        self,
        power_grid: pd.DataFrame,
        x_var: str,
        y_var: str,
        z_var: str = "power",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generates a heatmap plot for the provided power grid data.

        This method creates a heatmap to visually represent the power grid data using seaborn.
        It pivots the input data to form a matrix for the heatmap, allowing for a clear visual
        depiction of the relationship between the specified variables.

        If the power grid has more than two varying parameters, the method will aggregate
        (using mean) across the additional dimensions before pivoting.

        Args:
            power_grid: pd.DataFrame
                The dataset containing the power grid information. It must include columns for
                at least x_var, y_var, and z_var.
            x_var: str
                The column name in power_grid representing the x-axis variable.
            y_var: str
                The column name in power_grid representing the y-axis variable.
            z_var: str, default="power"
                The column name in power_grid representing the values for the heatmap. Default is "power".
            title: Optional[str]
                The title of the heatmap. If not provided, a default title based on z_var is generated.
            save_path: Optional[str]
                The path to save the generated heatmap figure. If not provided, the figure is not saved.

        Returns:
            plt.Figure
                A matplotlib Figure object containing the generated heatmap.

        Raises:
            None
        """
        import seaborn as sns  # Preferred at top of file, kept here for context if needed

        # Aggregate data by x_var and y_var to handle cases where there are
        # multiple varying parameters (e.g., effect_size, sample_size, num_studies)
        # This prevents "duplicate entries" error when pivoting
        aggregated_data = power_grid.groupby([x_var, y_var], as_index=False)[z_var].mean()

        # Pivot and format labels
        heatmap_data = aggregated_data.pivot(index=y_var, columns=x_var, values=z_var)
        z_label = self._format_label(z_var)
        x_label = self._format_label(x_var)
        y_label = self._format_label(y_var)

        fig, ax = plt.subplots(figsize=self.figsize)

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            ax=ax,
            cbar_kws={"label": z_label},
        )

        ax.set_title(title or f"Power Analysis Heatmap: {z_label}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _format_label(self, text: str) -> str:
        """
        Formats the given label by replacing underscores with spaces and capitalizing each word.

        Args:
            text: str
                The label text to be formatted.

        Returns:
            str
                The formatted label with spaces instead of underscores and each word capitalized.
        """
        return text.replace("_", " ").title()


# ------------------------------------------------------------------
# Convenience function for backward compatibility
# ------------------------------------------------------------------
def calc_meta_power(
    effect_size: float,
    avg_sample_size: float,
    num_studies: int,
    heterogeneity: Union[float, str],
    tails: int = 2,
    alpha: float = 0.05,
) -> float:
    """
    Calculates the statistical power of a meta-analysis given the effect size, average
    sample size, number of studies, heterogeneity, number of tails, and significance level.

    This function utilizes a MetaAnalysisPowerCalculator object to compute the power of a
    meta-analysis based on the provided parameters. The power represents the probability
    of rejecting the null hypothesis when it is indeed false, reflecting the strength of
    the study design.

    Args:
        effect_size: float
            The expected standardized effect size of the meta-analysis.
        avg_sample_size: float
            The average sample size across the included studies.
        num_studies: int
            The number of studies included in the meta-analysis.
        heterogeneity: Union[float, str]
            The heterogeneity of the studies' true effect sizes. Can be a numeric value or a
            string representing a common heterogeneity estimate.
        tails: int, optional
            The number of tails for the test (default is 2, two-tailed test).
        alpha: float, optional
            The significance level for the test (default is 0.05, or 5% significance level).

    Returns:
        float
            The computed statistical power of the meta-analysis.
    """
    result = MetaAnalysisPowerCalculator().calc_meta_power(
        effect_size, avg_sample_size, num_studies, heterogeneity, tails, alpha
    )
    return result.power


# ------------------------------------------------------------------
# PowerAnalysisSummary Class
# ------------------------------------------------------------------
class PowerAnalysisSummary:
    """Class for creating comprehensive power analysis summaries."""

    def __init__(self):
        """
        Initializes  the summary generator.

        Attributes:
            calculator: MetaAnalysisPowerCalculator
                Handles calculations for meta-analysis power estimation.
            plotter: PowerAnalysisPlotter
                Handles plotting and visualization of power analysis.
        """
        self.calculator = MetaAnalysisPowerCalculator()
        self.plotter = PowerAnalysisPlotter()

    def comprehensive_power_analysis(
        self,
        effect_size: float,
        avg_sample_size: float,
        num_studies: int,
        heterogeneity: Union[float, str] = "moderate",
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Performs a comprehensive power analysis for meta-analysis research.

        The method calculates various power metrics such as power for the main analysis, required number of studies,
        required sample size, and detectable effect size. Additionally, it has the option to generate visualizations
        of power analysis results and save them to an output directory.

        Args:
            effect_size: float
                The effect size assumed for the meta-analysis.
            avg_sample_size: float
                The average sample size per study in the meta-analysis.
            num_studies: int
                The total number of studies included in the meta-analysis.
            heterogeneity: Union[float, str]
                The heterogeneity level in the studies. Can be specified as a numeric value or a string
                (e.g., "low", "moderate", "high"). Default is "moderate".
            output_dir: Optional[str]
                The directory path where visualizations of the power analysis results will be saved, if provided.

        Returns:
            Dict
                A dictionary containing the results from the power analysis, which includes:
                - main_analysis: Calculated power for the main analysis.
                - studies_analysis: An analysis of the required number of studies for power.
                - sample_size_analysis: An analysis of the required sample size for power.
                - effect_size_analysis: An analysis of the detectable effect size for power.
                If output_dir is provided, the dictionary also contains plots, which are the file paths or figures
                of the generated visualizations.

        Raises:
            ValueError
                If the input parameters are invalid or incompatible with the power analysis calculations.
            RuntimeError
                If an unexpected error occurs during analysis or plotting operations.
        """
        results = {
            # Main power calculation
            "main_analysis": self.calculator.calc_meta_power(
                effect_size, avg_sample_size, num_studies, heterogeneity
            ),
            # Power vs number of studies
            "studies_analysis": self.calculator.find_required_studies(
                effect_size, avg_sample_size, heterogeneity=heterogeneity
            ),
            # Power vs sample size
            "sample_size_analysis": self.calculator.find_required_sample_size(
                effect_size, num_studies, heterogeneity=heterogeneity
            ),
            # Minimum detectable effect size
            "effect_size_analysis": self.calculator.find_detectable_effect_size(
                avg_sample_size, num_studies, heterogeneity=heterogeneity
            ),
        }

        # Generate plots if output directory specified
        if output_dir:
            import os

            os.makedirs(output_dir, exist_ok=True)

            scenarios = [
                # Plot power vs studies
                (
                    "studies_analysis",
                    "studies",
                    "Power vs Number of Studies",
                    "power_vs_studies.png",
                ),
                # Plot power vs sample size
                (
                    "sample_size_analysis",
                    "sample_size",
                    "Power vs Sample Size",
                    "power_vs_sample_size.png",
                ),
                # Plot power vs effect size
                (
                    "effect_size_analysis",
                    "effect_size",
                    "Power vs Effect Size",
                    "power_vs_effect_size.png",
                ),
            ]

            results["plots"] = []
            for result_key, x_var, plot_title, filename in scenarios:
                fig = self.plotter.plot_power_curve(
                    results[result_key],
                    x_var=x_var,
                    title=plot_title,
                    save_path=os.path.join(output_dir, filename),
                )
                results["plots"].append(fig)

        return results


# ------------------------------------------------------------------
# Convenience function for backward compatibility
# ------------------------------------------------------------------
def print_power_summary(results: Dict) -> None:
    """
    Prints a detailed summary of a statistical power analysis based on the results
    provided. The summary includes key metrics such as effect size, sample size,
    number of studies, heterogeneity, test type, and statistical power. Additionally,
    it provides analysis for studies required, sample size needed, or minimum detectable
    effect size for achieving a statistical power of 80%.

    Args:
        results (Dict): A dictionary containing the main analysis results and optional
        additional analyses. The dictionary should include:
            - "main_analysis": A dictionary with keys like effect_size,
              avg_sample_size, num_studies, heterogeneity, tail, and power.
            - Optional keys such as "studies_analysis", "sample_size_analysis", and
              "effect_size_analysis", each containing a dictionary with specific keys
              (e.g., required_studies, required_sample_size, or
              min_detectable_effect_size) for additional analysis.

    Returns:
        None: The function prints the summary directly to the output with no return value.
    """
    main = results["main_analysis"]

    print("=" * 60)
    print("STATISTICAL POWER ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Effect Size: {main.effect_size:.3f}")
    print(f"Average Sample Size per Group: {main.avg_sample_size:.0f}")
    print(f"Number of Studies: {main.num_studies}")
    print(f"Heterogeneity: {main.heterogeneity:.3f}")
    print(f"Test Type: {'Two-tailed' if main.tail == 2 else 'One-tailed'}")
    print("-" * 60)
    print(f"STATISTICAL POWER: {main.power:.3f} ({main.power * 100:.1f}%)")
    print("-" * 60)

    # Mapping of analysis keys to their display configuration
    analysis_configs = [
        (
            "studies_analysis",
            "required_studies",
            "Studies needed for 80% power",
            ">100 (not feasible)",
            "{:.0f}",
        ),
        (
            "sample_size_analysis",
            "required_sample_size",
            "Sample size per group for 80% power",
            ">1000 (not feasible)",
            "{:.0f}",
        ),
        (
            "effect_size_analysis",
            "min_detectable_effect_size",
            "Minimum detectable effect size (80% power)",
            ">5.0 (very large)",
            "{:.3f}",
        ),
    ]

    for analysis_key, result_key, label, fallback, fmt in analysis_configs:
        if analysis_key in results:
            val = results[analysis_key].get(result_key)
            display_val = fmt.format(val) if val is not None else fallback
            print(f"{label}: {display_val}")

    print("=" * 60)
