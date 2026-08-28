"""
ROC and PRC analysis functions for MetaIntegrator Python port.
This module handles ROC/PRC curve analysis, AUC calculations,
gene search optimization, and related statistical computations.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils import resample

from .roc_analysis_engine import (
    ROCAnalysisEngine,
    UnifiedROCAnalyzer,
)

# Configure logging
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# PRCAnalyzer Class
# ------------------------------------------------------------------
class PRCAnalyzer:
    """Class for Precision-Recall curve analysis."""

    def __init__(self) -> None:
        """Initialize PRC analyzer."""
        pass

    def calculate_prc(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
        """
        Calculates the Precision-Recall Curve (PRC) metrics, Area Under the Precision-Recall
        Curve (AUPRC), baseline precision, and confidence intervals for the AUPRC.

        The method computes the key components required for PRC-based evaluation, including
        precision, recall, thresholds, and AUPRC value. It ensures that the PRC endpoints are
        prepared appropriately for visualization. Additionally, it calculates the confidence
        intervals for the AUPRC to provide statistical insights.

        Args:
            y_true (np.ndarray): Ground truth binary labels (0 or 1), with 1 denoting the positive class.
            y_scores (np.ndarray): Predicted scores or probabilities for the positive class.

        Returns:
            Dict: A dictionary containing the following keys:
                - "precision" (np.ndarray): Precision values for the PRC with all necessary endpoints.
                - "recall" (np.ndarray): Recall values for the PRC with all necessary endpoints.
                - "thresholds" (np.ndarray): Thresholds corresponding to the precision and recall values.
                - "auprc" (float): Calculated Area Under the Precision-Recall Curve (AUPRC).
                - "auprc_ci" (tuple(float, float)): Confidence interval tuple for the AUPRC.
                - "baseline" (float): Baseline precision, equal to the prevalence of the positive class.
                - "prc_curve" (pd.DataFrame): DataFrame containing the precision-recall curve values
                  for easy visualization or analysis.

        Raises:
            ValueError: If y_true and y_scores have mismatched dimensions or invalid inputs.
        """
        # 1. Calculate core PRC metrics
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        auprc_value = average_precision_score(y_true, y_scores)

        # 2. Ensure curve endpoints for visualization (Recall 0 to 1)
        # Precision is 1 at recall 0; Precision is baseline at recall 1
        baseline_precision = np.sum(y_true == 1) / len(y_true)
        precision_plot = np.concatenate([[1.0], precision, [baseline_precision]])
        recall_plot = np.concatenate([[0.0], recall, [1.0]])

        # 3. Calculate confidence intervals
        auprc_ci = self._calculate_auprc_ci(y_true, y_scores, auprc_value)

        return {
            "precision": precision_plot,
            "recall": recall_plot,
            "thresholds": thresholds,
            "auprc": auprc_value,
            "auprc_ci": auprc_ci,
            "baseline": baseline_precision,
            "prc_curve": pd.DataFrame({"precision": precision_plot, "recall": recall_plot}),
        }

    def _calculate_auprc_ci(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        auprc_value: float,
        confidence: float = 0.95,
        method: str = "binomial",
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for the area under the precision–recall curve.

        Args:
            y_true (np.ndarray): Ground‑truth binary labels (0 or 1).
            y_scores (np.ndarray): Predicted scores or probabilities for the positive class.
            auprc_value (float): Empirical area under the precision–recall curve.
            confidence (float, optional): Confidence level for the interval. Default is 0.95.
            method (str, optional): Method for interval estimation. Currently only the
                'binomial' method is implemented; other values return the point estimate twice.

        Returns:
            Tuple[float, float]: Lower and upper bounds of the confidence interval. If there
                are no positive samples or if an unsupported method is requested, the
                function returns (auprc_value, auprc_value).

        Notes:
            The binomial method uses the normal approximation with z‑critical value
                derived from the specified confidence level. The standard error is
                calculated as sqrt(auprc_value * (1 - auprc_value) / n_pos). The interval
                bounds are capped at 0.0 and 1.0 to respect the valid range of the AUPRC.
        """
        n_pos = np.sum(y_true == 1)
        if n_pos == 0:
            return (auprc_value, auprc_value)

        z_critical = stats.norm.ppf((1 + confidence) / 2)

        if method == "binomial":
            standard_error = np.sqrt((auprc_value * (1 - auprc_value)) / n_pos)
            lower = max(0.0, auprc_value - z_critical * standard_error)
            upper = min(1.0, auprc_value + z_critical * standard_error)
            return lower, upper

        return (auprc_value, auprc_value)

    def plot_prc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "Precision-Recall Curve",
        label: Optional[str] = None,
        color: str = "red",
        figsize: Tuple = (8, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plots a Precision-Recall curve with optional customization such as title, labels, and
        colors, and returns the generated Matplotlib figure. The curve includes precision-recall
        values computed from provided true labels and scores, as well as a baseline line
        representing random performance. The plot can also be saved to a file path.

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            y_scores (np.ndarray): Predicted scores associated with each data point.
            title (str, optional): Title of the plot. Default is "Precision-Recall Curve".
            label (str, optional): Custom label for the precision-recall curve. Default is None.
            color (str, optional): Color of the precision-recall curve. Default is "red".
            figsize (Tuple, optional): Size of the plot figure in inches (width, height).
            save_path (str, optional): Path to save the generated plot as an image.
                If None, the plot won't be saved.

        Returns:
            plt.Figure: Matplotlib Figure object containing the rendered plot.
        """
        prc_stats = self.calculate_prc(y_true, y_scores)
        baseline = np.sum(y_true == 1) / len(y_true)

        fig, ax = plt.subplots(figsize=figsize)

        # 1. Determine Display Labels
        curve_label = label or self._generate_prc_label(prc_stats)
        baseline_label = f"Random (AUPRC={baseline:.3f})"

        # 2. Draw Main Plot Components
        ax.plot(
            prc_stats["recall"],
            prc_stats["precision"],
            color=color,
            linewidth=2,
            label=curve_label,
        )
        ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.6, label=baseline_label)

        # 3. Finalize Plot Aesthetics
        self._configure_prc_axes(ax, title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _generate_prc_label(self, stats: Dict) -> str:
        """
        Generates a precision-recall curve label based on provided statistics.

        This method takes statistical data including the area under the precision-recall
        curve (AUPRC) value and its confidence interval. It returns a formatted string
        containing those values for display or reporting purposes.

        Args:
            stats : Dict
                A dictionary containing 'auprc' key with the AUPRC value (float), and
                'auprc_ci' key with a tuple of lower and upper bounds of the confidence
                interval (float, float).

        Returns:
            str
                A formatted string representing the AUPRC value and its confidence interval.
        """
        auprc_val = stats["auprc"]
        lower, upper = stats["auprc_ci"]
        return f"AUPRC = {auprc_val:.3f} (95% CI: {lower:.3f}-{upper:.3f})"

    def _configure_prc_axes(self, ax: plt.Axes, title: str) -> None:
        """
        Configures the axes object for a Precision-Recall Curve (PRC) plot.

        This method applies essential settings to the provided Axes object,
        such as axis labels, grid, and limits, to ensure a consistent and
        clear visualization of a Precision-Recall Curve.

        Args:
            ax (plt.Axes): The Matplotlib Axes object to configure.
            title (str): The title text for the axes.

        Returns:
            None
        """
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])


# ------------------------------------------------------------------
# Convenience functions for backward compatibility
# ------------------------------------------------------------------


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, **kwargs) -> plt.Figure:
    """
    Plot a single ROC curve.

    This function creates an instance of ROCAnalyzer and delegates the plotting
    task to it, passing the true labels, predicted scores and any additional
    keyword arguments.

    Args:
        y_true (np.ndarray): Array of true binary labels.
        y_scores (np.ndarray): Array of predicted scores or probabilities.
        **kwargs: Additional keyword arguments forwarded to the analyzer's
            plot_roc_curve method.

    Returns:
        plt.Figure: Figure object containing the plotted ROC curve.
    """
    analyzer = UnifiedROCAnalyzer()
    return analyzer.plot_roc_curve(y_true, y_scores, **kwargs)


def plot_prc_curve(y_true: np.ndarray, y_scores: np.ndarray, **kwargs) -> plt.Figure:
    """
    Plot a single PRC curve.

    This function creates an instance of PRCAnalyzer and delegates the plotting
    task to it, passing the true labels, predicted scores and any additional
    keyword arguments.

    Args:
        y_true (np.ndarray): Array of true binary labels.
        y_scores (np.ndarray): Array of predicted scores or probabilities.
        **kwargs: Additional keyword arguments forwarded to the analyzer's
            plot_prc_curve method.

    Returns:
        plt.Figure: Figure object containing the plotted PRC curve.
    """
    analyzer = PRCAnalyzer()
    return analyzer.plot_prc_curve(y_true, y_scores, **kwargs)


def multiple_roc_plot(datasets: List[Dict], filter_object: Dict, **kwargs) -> plt.Figure:
    """Plot multiple ROC curves on a single figure.

    Args:
        datasets (list[dict]): List of dataset dictionaries, each containing the
            necessary keys for ROC analysis.
        filter_object (dict): Dictionary specifying filtering criteria for the
            datasets.
        **kwargs: Optional keyword arguments forwarded to ROCAnalyzer.multiple_roc_plot.

    Returns:
        matplotlib.figure.Figure: Figure object that contains the plotted ROC curves.
    """
    analyzer = ROCAnalysisEngine()
    return analyzer.multiple_roc_plot(datasets, filter_object, **kwargs)


def multiple_prc_plot(datasets: List[Dict], filter_object: Dict, **kwargs) -> plt.Figure:
    """Plot multiple PRC curves on a single figure.

    Args:
        datasets (list[dict]): List of dataset dictionaries, each containing the
            necessary keys for PRC analysis.
        filter_object (dict): Dictionary specifying filtering criteria for the
            datasets.
        **kwargs: Optional keyword arguments forwarded to PRCAnalyzer.multiple_prc_plot.

    Returns:
        matplotlib.figure.Figure: Figure object that contains the plotted PRC curves.
    """
    analyzer = PRCAnalyzer()
    return analyzer.multiple_prc_plot(datasets, filter_object, **kwargs)


# ------------------------------------------------------------------
# Statistical utility functions
# ------------------------------------------------------------------
def calculate_auc_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate the confidence interval for the AUC.

    This function creates an instance of ROCAnalyzer and uses it to compute the
    confidence interval for the area under the receiver operating characteristic
    curve. The AUC value is first computed using roc_auc_score from scikit-learn.

    Args:
        y_true (np.ndarray): True binary labels.
        y_scores (np.ndarray): Predicted scores or probabilities.
        confidence (float): Confidence level for the interval; defaults to 0.95.

    Returns:
        Tuple[float, float]: Lower and upper bounds of the AUC confidence interval.
    """
    analyzer = ROCAnalysisEngine()
    auc_value = roc_auc_score(y_true, y_scores)

    return analyzer._calculate_auc_ci(y_true, auc_value, confidence)


def calculate_auprc_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate the confidence interval for the AUPRC.

    This function creates an instance of PRCAnalyzer and uses it to compute the
    confidence interval for the area under the precision‑recall curve. The AUPRC
    value is first computed using average_precision_score from scikit-learn.

    Args:
        y_true (np.ndarray): True binary labels.
        y_scores (np.ndarray): Predicted scores or probabilities.
        confidence (float): Confidence level for the interval; defaults to 0.95.

    Returns:
        Tuple[float, float]: Lower and upper bounds of the AUPRC confidence interval.
    """
    analyzer = PRCAnalyzer()
    auprc_value = average_precision_score(y_true, y_scores)
    return analyzer._calculate_auprc_ci(y_true, y_scores, auprc_value, confidence)
