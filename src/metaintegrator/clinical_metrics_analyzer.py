"""
Clinical Metrics Analyzer for MetaIntegrator

"""

import logging
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix

logger = logging.getLogger(__name__)


# ============================================================================
# Youden's J Statistic Analyzer
# ============================================================================


class YoudenJAnalyzer:
    """
    Analyzer for calculating Youden's J statistic and optimal cutoff points.

    Implements the functionality from R's .youdenJ() function with enhanced
    features and better error handling.
    """

    def __init__(self):
        """Initialize Youden's J analyzer."""
        pass

    def calculate_optimal_cutoff(
        self,
        roc_data: Union[pd.DataFrame, Dict, Tuple[np.ndarray, np.ndarray]],
        sensitivity_target: float = 0.95,
        return_details: bool = False,
    ) -> Union[Dict, float]:
        """
        Calculate optimal cutoff using Youden's J statistic.

        Implements R's .youdenJ() functionality with enhancements:
        - Finds optimal cutoff that maximizes (Sensitivity + Specificity - 1)
        - Reports performance at optimal cutoff
        - Finds specificity at target sensitivity level

        Args:
            roc_data: ROC curve data in one of these formats:
                     - DataFrame with 'fpr' and 'tpr' columns
                     - Dict with 'fpr' and 'tpr' keys
                     - Tuple of (fpr_array, tpr_array)
            sensitivity_target: Target sensitivity level for specificity calculation (default: 0.95)
            return_details: If True, return detailed analysis; if False, return only J statistic

        Returns:
            Dict with optimal cutoff analysis or float J statistic value

        Raises:
            ValueError: If input data is invalid or insufficient
            TypeError: If roc_data format is not supported

        Examples:
            >>> analyzer = YoudenJAnalyzer()
            >>> # From ROC curve data
            >>> result = analyzer.calculate_optimal_cutoff(roc_dataframe)
            >>> print(f"Optimal sensitivity: {result['optimal_sensitivity']:.1%}")
            >>> print(f"Optimal specificity: {result['optimal_specificity']:.1%}")
        """
        # Parse and validate input data
        fpr, tpr = self._parse_roc_data(roc_data)
        self._validate_roc_arrays(fpr, tpr)

        # Calculate specificity from FPR
        specificity = 1 - fpr
        sensitivity = tpr

        # Calculate Youden's J statistic (Sensitivity + Specificity - 1)
        youden_j = sensitivity + specificity - 1

        # Find optimal cutoff (maximum J)
        optimal_idx = np.argmax(youden_j)
        optimal_j = youden_j[optimal_idx]
        optimal_sensitivity = sensitivity[optimal_idx]
        optimal_specificity = specificity[optimal_idx]

        # Find specificity at target sensitivity
        specificity_at_target = self._find_specificity_at_sensitivity(
            sensitivity, specificity, sensitivity_target
        )

        if return_details:
            return {
                "youden_j": float(optimal_j),
                "optimal_sensitivity": float(optimal_sensitivity),
                "optimal_specificity": float(optimal_specificity),
                "optimal_sensitivity_percent": float(optimal_sensitivity * 100),
                "optimal_specificity_percent": float(optimal_specificity * 100),
                f"specificity_at_{sensitivity_target * 100:.0f}pct_sensitivity": float(
                    specificity_at_target
                ),
                f"specificity_at_{sensitivity_target * 100:.0f}pct_sensitivity_percent": float(
                    specificity_at_target * 100
                ),
                "optimal_index": int(optimal_idx),
                "n_points": len(fpr),
            }
        else:
            return float(optimal_j)

    def print_optimal_cutoff_summary(
        self,
        roc_data: Union[pd.DataFrame, Dict, Tuple[np.ndarray, np.ndarray]],
        sensitivity_target: float = 0.95,
        precision: int = 1,
    ) -> None:
        """
        Print summary of optimal cutoff analysis (matches R output format).

        Args:
            roc_data: ROC curve data
            sensitivity_target: Target sensitivity for specificity calculation
            precision: Number of decimal places for percentage display
        """
        try:
            result = self.calculate_optimal_cutoff(
                roc_data, sensitivity_target, return_details=True
            )

            # Format similar to R output
            print(
                f"At optimal cutoff, summary sensitivity = "
                f"{result['optimal_sensitivity_percent']:.{precision}f}%, "
                f"summary specificity = {result['optimal_specificity_percent']:.{precision}f}%"
            )

            target_pct = sensitivity_target * 100
            print(
                f"At summary sensitivity = {target_pct:.0f}%, "
                f"summary specificity = {result[f'specificity_at_{target_pct:.0f}pct_sensitivity_percent']:.{precision}f}%"
            )

        except Exception as e:
            logger.error(f"Failed to print optimal cutoff summary: {str(e)}")
            raise

    def _parse_roc_data(
        self, roc_data: Union[pd.DataFrame, Dict, Tuple]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse ROC data from various input formats."""
        if isinstance(roc_data, pd.DataFrame):
            if "fpr" in roc_data.columns and "tpr" in roc_data.columns:
                return roc_data["fpr"].values, roc_data["tpr"].values
            elif "FPR" in roc_data.columns and "TPR" in roc_data.columns:
                return roc_data["FPR"].values, roc_data["TPR"].values
            else:
                # Assume first column is FPR, second is TPR (like R)
                if roc_data.shape[1] >= 2:
                    return roc_data.iloc[:, 0].values, roc_data.iloc[:, 1].values
                else:
                    raise ValueError("DataFrame must have at least 2 columns for FPR and TPR")

        elif isinstance(roc_data, dict):
            if "fpr" in roc_data and "tpr" in roc_data:
                return np.array(roc_data["fpr"]), np.array(roc_data["tpr"])
            elif "FPR" in roc_data and "TPR" in roc_data:
                return np.array(roc_data["FPR"]), np.array(roc_data["TPR"])
            else:
                raise ValueError("Dict must contain 'fpr' and 'tpr' (or 'FPR' and 'TPR') keys")

        elif isinstance(roc_data, (tuple, list)) and len(roc_data) == 2:
            return np.array(roc_data[0]), np.array(roc_data[1])

        else:
            raise TypeError(
                "roc_data must be DataFrame, dict with fpr/tpr keys, or tuple of (fpr, tpr)"
            )

    def _validate_roc_arrays(self, fpr: np.ndarray, tpr: np.ndarray) -> None:
        """Validate FPR and TPR arrays."""
        if len(fpr) == 0 or len(tpr) == 0:
            raise ValueError("FPR and TPR arrays cannot be empty")

        if len(fpr) != len(tpr):
            raise ValueError(f"FPR and TPR must have same length: {len(fpr)} vs {len(tpr)}")

        if len(fpr) < 2:
            raise ValueError("Need at least 2 points for Youden's J calculation")

        if not (0 <= fpr.min() and fpr.max() <= 1):
            warnings.warn("FPR values should be in range [0, 1]")

        if not (0 <= tpr.min() and tpr.max() <= 1):
            warnings.warn("TPR values should be in range [0, 1]")

    def _find_specificity_at_sensitivity(
        self, sensitivity: np.ndarray, specificity: np.ndarray, target_sens: float
    ) -> float:
        """Find specificity at target sensitivity level."""
        # Find index closest to target sensitivity
        sens_diff = np.abs(sensitivity - target_sens)
        closest_idx = np.argmin(sens_diff)

        return specificity[closest_idx]


# ============================================================================
# Custom Array Wrapper for R-like behavior
# ============================================================================


class NamedArray:
    """Wrapper for numpy array with row/column names (R-like behavior)."""

    def __init__(self, data, row_names=None, col_names=None):
        self.data = np.array(data)
        self.row_names = row_names
        self.col_names = col_names

    def __getattr__(self, name):
        # Delegate to numpy array for all other attributes
        return getattr(self.data, name)

    def __getitem__(self, key):
        return self.data[key]

    def __repr__(self):
        return f"NamedArray(shape={self.data.shape}, row_names={self.row_names}, col_names={self.col_names})"


# ============================================================================
# Clinical Parameters Analyzer
# ============================================================================


class ClinicalParametersAnalyzer:
    """
    Analyzer for calculating clinical performance parameters across multiple studies.

    Implements the functionality from R's .clinParamMultiRoc() function with
    enhanced error handling and additional metrics.
    """

    def __init__(self):
        """Initialize clinical parameters analyzer."""
        pass

    def calculate_multi_study_parameters(
        self,
        study_data_list: List[Dict],
        threshold_method: str = "weighted_means",
        return_format: str = "dataframe",
        precision: int = 3,
    ) -> Union[pd.DataFrame, NamedArray]:
        """
        Calculate clinical performance parameters across multiple studies.

        Implements R's .clinParamMultiRoc() functionality with enhancements:
        - Multiple threshold calculation methods
        - Comprehensive clinical metrics
        - Flexible output formats
        - Graceful handling of individual study failures

        Args:
            study_data_list: List of study dictionaries, each containing:
                           - 'score': array of prediction scores
                           - 'group' or 'class': array of binary labels (0/1)
                           - Optional 'name': study identifier
            threshold_method: Method for threshold calculation:
                            - "weighted_means": R's original method (default)
                            - "youden": Use Youden's J optimal cutoff
                            - "closest_to_topleft": Closest to (0,1) corner
            return_format: "dataframe" or "array" (matches R's matrix output)
            precision: Number of significant digits for rounding

        Returns:
            DataFrame or NamedArray with clinical parameters:
            - sens: Sensitivity (True Positive Rate)
            - spec: Specificity (True Negative Rate)
            - PPV: Positive Predictive Value (Precision)
            - NPV: Negative Predictive Value
            - acc: Accuracy

        Raises:
            ValueError: If input data is invalid or insufficient
            TypeError: If study data format is incorrect

        Examples:
            >>> analyzer = ClinicalParametersAnalyzer()
            >>> studies = [
            ...     {'score': scores1, 'group': labels1, 'name': 'Study1'},
            ...     {'score': scores2, 'group': labels2, 'name': 'Study2'}
            ... ]
            >>> results = analyzer.calculate_multi_study_parameters(studies)
            >>> print(results)
        """
        # Basic validation - only check list format and non-empty
        self._basic_validate_study_data_list(study_data_list)

        results = []
        study_names = []
        successful_studies = 0

        for i, study_data in enumerate(study_data_list):
            try:
                # Extract data and standardize format
                scores, groups, study_name = self._extract_study_data(study_data, i)
                study_names.append(study_name)

                # Calculate threshold based on chosen method
                threshold = self._calculate_threshold(scores, groups, threshold_method)

                # Calculate confusion matrix elements
                tp, tn, fp, fn = self._calculate_confusion_matrix_elements(
                    scores, groups, threshold
                )

                # Calculate clinical parameters
                clinical_params = self._calculate_clinical_metrics(tp, tn, fp, fn)

                results.append(clinical_params)
                successful_studies += 1

                logger.debug(
                    f"Processed {study_name}: threshold={threshold:.4f}, "
                    f"sens={clinical_params[0]:.3f}, spec={clinical_params[1]:.3f}"
                )

            except Exception as e:
                warning_msg = f"Failed to process study {i + 1}: {str(e)}"
                logger.warning(warning_msg)
                warnings.warn(warning_msg)
                # Add NaN row for failed studies
                results.append([np.nan] * 5)
                study_names.append(f"Study_{i + 1}_FAILED")
                continue

        # Only raise error if no results at all (shouldn't happen with valid input)
        if not results:
            raise ValueError("No studies could be processed successfully")

        # Convert to numpy array and apply precision rounding (like R's signif)
        results_array = np.array(results)
        results_rounded = self._apply_significant_digits(results_array, precision)

        if return_format == "dataframe":
            return pd.DataFrame(
                results_rounded,
                columns=["sens", "spec", "PPV", "NPV", "acc"],
                index=study_names,
            )
        elif return_format == "array":
            # Return NamedArray with row/column names like R
            return NamedArray(
                results_rounded,
                row_names=study_names,
                col_names=["sens", "spec", "PPV", "NPV", "acc"],
            )
        else:
            raise ValueError("return_format must be 'dataframe' or 'array'")

    def _basic_validate_study_data_list(self, study_data_list: List[Dict]) -> None:
        """Basic validation of the input study data list."""
        if not isinstance(study_data_list, list):
            raise TypeError("study_data_list must be a list")

        if len(study_data_list) == 0:
            raise ValueError("study_data_list cannot be empty")

    def _extract_study_data(
        self, study_data: Dict, study_index: int
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Extract and validate data from a single study."""
        # Check if study_data is a dict
        if not isinstance(study_data, dict):
            raise TypeError(f"Study {study_index + 1} must be a dictionary")

        # Check for required keys
        has_score = "score" in study_data
        has_group = "group" in study_data or "class" in study_data

        if not has_score:
            raise ValueError(f"Study {study_index + 1} must contain 'score' key")
        if not has_group:
            raise ValueError(f"Study {study_index + 1} must contain 'group' or 'class' key")

        # Extract scores
        scores = np.array(study_data["score"])

        # Extract groups (try 'group' first, then 'class')
        if "group" in study_data:
            groups = np.array(study_data["group"])
        else:
            groups = np.array(study_data["class"])

        # Get study name
        study_name = study_data.get("name", f"Study_{study_index + 1}")

        # Validate data
        if len(scores) != len(groups):
            raise ValueError(
                f"Score and group arrays must have same length: {len(scores)} vs {len(groups)}"
            )

        if len(scores) < 4:  # Need at least 4 samples for meaningful clinical parameters
            warnings.warn(
                f"Study {study_name} has only {len(scores)} samples, may produce unreliable results"
            )
            if len(scores) < 2:
                raise ValueError(f"Study needs at least 2 samples, got {len(scores)}")

        # Check for binary groups
        unique_groups = np.unique(groups)
        if len(unique_groups) != 2:
            raise ValueError(
                f"Groups must be binary, found {len(unique_groups)} unique values: {unique_groups}"
            )

        # Ensure groups are 0/1 format
        if not np.array_equal(sorted(unique_groups), [0, 1]):
            # Remap to 0/1 if needed
            min_group = min(unique_groups)
            groups = (groups != min_group).astype(int)
            logger.debug(f"Remapped groups from {unique_groups} to [0, 1]")

        return scores, groups, study_name

    def _calculate_threshold(self, scores: np.ndarray, groups: np.ndarray, method: str) -> float:
        """Calculate decision threshold using specified method."""
        if method == "weighted_means":
            # R's original method: weighted combination of group means
            return self._weighted_means_threshold(scores, groups)
        elif method == "youden":
            # Youden's J optimal cutoff
            return self._youden_optimal_threshold(scores, groups)
        elif method == "closest_to_topleft":
            # Point closest to (0,1) in ROC space
            return self._closest_to_topleft_threshold(scores, groups)
        else:
            raise ValueError(f"Unknown threshold method: {method}")

    def _weighted_means_threshold(self, scores: np.ndarray, groups: np.ndarray) -> float:
        """
        Calculate threshold using R's weighted means method.

        This implements the exact formula from R's .clinParamMultiRoc:
        thresh <- (mean(score.1)/sd(score.1) + mean(score.2)/sd(score.2)) / (1/sd(score.1) + 1/sd(score.2))
        """
        # Separate scores by group
        score_0 = scores[groups == 0]  # Negative class
        score_1 = scores[groups == 1]  # Positive class

        if len(score_0) == 0 or len(score_1) == 0:
            raise ValueError("Both groups must have at least one sample")

        # Calculate means and standard deviations
        mean_0, std_0 = np.mean(score_0), np.std(score_0, ddof=1)
        mean_1, std_1 = np.mean(score_1), np.std(score_1, ddof=1)

        # Handle zero standard deviation
        if std_0 == 0 or std_1 == 0:
            warnings.warn("Zero standard deviation detected, using simple mean threshold")
            return (mean_0 + mean_1) / 2

        # R's weighted means formula
        numerator = (mean_0 / std_0) + (mean_1 / std_1)
        denominator = (1 / std_0) + (1 / std_1)

        return numerator / denominator

    def _youden_optimal_threshold(self, scores: np.ndarray, groups: np.ndarray) -> float:
        """Calculate threshold that maximizes Youden's J statistic."""
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(groups, scores)

        # Calculate Youden's J for each threshold
        j_scores = tpr - fpr  # Equivalent to tpr + (1-fpr) - 1
        optimal_idx = np.argmax(j_scores)

        return thresholds[optimal_idx]

    def _closest_to_topleft_threshold(self, scores: np.ndarray, groups: np.ndarray) -> float:
        """Calculate threshold for point closest to (0,1) in ROC space."""
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(groups, scores)

        # Distance to top-left corner (0,1)
        distances = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)
        optimal_idx = np.argmin(distances)

        return thresholds[optimal_idx]

    def _calculate_confusion_matrix_elements(
        self, scores: np.ndarray, groups: np.ndarray, threshold: float
    ) -> Tuple[int, int, int, int]:
        """Calculate TP, TN, FP, FN from scores, groups, and threshold."""
        # Make predictions based on threshold
        predictions = (scores >= threshold).astype(int)

        # Calculate confusion matrix elements
        tp = np.sum((predictions == 1) & (groups == 1))
        tn = np.sum((predictions == 0) & (groups == 0))
        fp = np.sum((predictions == 1) & (groups == 0))
        fn = np.sum((predictions == 0) & (groups == 1))

        return tp, tn, fp, fn

    def _calculate_clinical_metrics(self, tp: int, tn: int, fp: int, fn: int) -> List[float]:
        """Calculate clinical performance metrics from confusion matrix elements."""
        # Calculate metrics with safe division
        total_pos = tp + fn
        total_neg = tn + fp
        total_pred_pos = tp + fp
        total_pred_neg = tn + fn
        total_all = tp + tn + fp + fn

        # Sensitivity (TPR, Recall)
        sens = tp / total_pos if total_pos > 0 else 0.0

        # Specificity (TNR)
        spec = tn / total_neg if total_neg > 0 else 0.0

        # Positive Predictive Value (Precision)
        ppv = tp / total_pred_pos if total_pred_pos > 0 else 0.0

        # Negative Predictive Value
        npv = tn / total_pred_neg if total_pred_neg > 0 else 0.0

        # Accuracy
        acc = (tp + tn) / total_all if total_all > 0 else 0.0

        return [sens, spec, ppv, npv, acc]

    def _apply_significant_digits(self, array: np.ndarray, digits: int) -> np.ndarray:
        """Apply significant digits rounding (equivalent to R's signif function)."""

        def signif_scalar(x, digits):
            if np.isnan(x) or x == 0:
                return x
            return np.round(x, digits - int(np.floor(np.log10(np.abs(x)))) - 1)

        # Apply to each element
        vectorized_signif = np.vectorize(signif_scalar)
        return vectorized_signif(array, digits)


# ============================================================================
# Integration with Existing ROC Analysis System
# ============================================================================


def calculate_optimal_cutoff(roc_data, sensitivity_target=0.95, return_details=False):
    """
    Convenience function for calculating optimal cutoff using Youden's J statistic.

    Args:
        roc_data: ROC curve data (DataFrame, dict, or tuple)
        sensitivity_target: Target sensitivity for specificity calculation (default: 0.95)
        return_details: Return detailed analysis if True, J statistic only if False

    Returns:
        Optimal cutoff analysis or J statistic value
    """
    analyzer = YoudenJAnalyzer()
    return analyzer.calculate_optimal_cutoff(roc_data, sensitivity_target, return_details)


def print_optimal_cutoff_summary(roc_data, sensitivity_target=0.95, precision=1):
    """
    Print optimal cutoff summary (matches R's .youdenJ output format).

    Args:
        roc_data: ROC curve data
        sensitivity_target: Target sensitivity for specificity calculation
        precision: Decimal places for percentage display
    """
    analyzer = YoudenJAnalyzer()
    analyzer.print_optimal_cutoff_summary(roc_data, sensitivity_target, precision)


def calculate_multi_study_clinical_parameters(
    study_data_list, threshold_method="weighted_means", **kwargs
):
    """
    Calculate clinical parameters across multiple studies.

    Args:
        study_data_list: List of study dictionaries with 'score' and 'group'/'class'
        threshold_method: "weighted_means" (R's method), "youden", or "closest_to_topleft"
        **kwargs: Additional arguments for ClinicalParametersAnalyzer

    Returns:
        DataFrame with clinical performance metrics
    """
    analyzer = ClinicalParametersAnalyzer()
    return analyzer.calculate_multi_study_parameters(study_data_list, threshold_method, **kwargs)


# ============================================================================
# Export for integration with existing modules
# ============================================================================

__all__ = [
    "YoudenJAnalyzer",
    "ClinicalParametersAnalyzer",
    "calculate_optimal_cutoff",
    "print_optimal_cutoff_summary",
    "calculate_multi_study_clinical_parameters",
    "NamedArray",
]
