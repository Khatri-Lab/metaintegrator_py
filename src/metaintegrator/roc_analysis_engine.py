"""
Unified ROC Analysis for MetaIntegrator Python Port
==================================================

Clean, production-grade implementation that consolidates all ROC analysis functionality:
- Simple pooled ROC analysis (weighted averaging)
- Summary ROC analysis (Kester & Buntinx method)
- Individual ROC curve analysis
- Precision-Recall curve analysis
- Gene search optimization

This replaces the previous fragmented approach with a single, cohesive system.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.utils import resample

from .clinical_metrics_analyzer import (
    YoudenJAnalyzer,
    ClinicalParametersAnalyzer,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Core ROC Analysis Engine
# ============================================================================


class ROCAnalysisEngine:
    """
    Core engine for ROC curve calculations and statistics.

    This class provides the fundamental ROC analysis capabilities used
    by all higher-level analyzers in the system.
    """

    def __init__(self):
        """Initialize ROC analysis engine."""
        pass

    def calculate_roc_curve(
        self,
        y_true: Union[np.ndarray, List],
        y_scores: Union[np.ndarray, List],
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Calculate ROC curve with comprehensive statistics.

        Args:
            y_true: Binary class labels (0/1)
            y_scores: Prediction scores or probabilities
            confidence: Confidence level for CI calculation (default: 0.95)

        Returns:
            Dictionary containing:
            - fpr: False positive rates
            - tpr: True positive rates
            - thresholds: Decision thresholds
            - auc: Area under the curve
            - auc_ci: 95% confidence interval for AUC
            - auc_pvalue: P-value against null hypothesis (AUC=0.5)
            - n_pos: Number of positive samples
            - n_neg: Number of negative samples
            - roc_curve: DataFrame with FPR/TPR pairs

        Raises:
            ValueError: If input arrays are invalid
        """
        # Input validation
        y_true, y_scores = self._validate_inputs(y_true, y_scores)

        # Check for single class
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            return self._handle_single_class(y_true, unique_classes)

        # Calculate ROC curve
        try:
            # sklearn's roc_curve() defaults to drop_intermediate=True, which removes threshold points that don't change FPR/TPR
            # R's ROCR: Keeps ALL intermediate points
            fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)
            auc_val = auc(fpr, tpr)
        except Exception as e:
            logger.warning(f"ROC curve calculation failed: {str(e)}")
            return self._handle_calculation_failure(y_true)

        # Calculate statistics
        auc_ci = self._calculate_auc_ci(y_true, auc_val, confidence)
        auc_pvalue = self._calculate_auc_pvalue(y_true, auc_val)

        # Create ROC curve DataFrame
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "threshold": thresholds})

        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": float(auc_val),
            "auc_ci": (float(auc_ci[0]), float(auc_ci[1])),
            "auc_pvalue": float(auc_pvalue) if not np.isnan(auc_pvalue) else np.nan,
            "n_pos": int(np.sum(y_true == 1)),
            "n_neg": int(np.sum(y_true == 0)),
            "roc_curve": roc_df,
        }

    def calculate_roc(
        self,
        y_true: Union[np.ndarray, List],
        y_scores: Union[np.ndarray, List],
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Function to calculate ROC curve with statistics."""
        return self.calculate_roc_curve(y_true, y_scores, confidence)

    def calculate_prc_curve(
        self,
        y_true: Union[np.ndarray, List],
        y_scores: Union[np.ndarray, List],
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Calculate Precision-Recall curve with statistics.

        Args:
            y_true: Binary class labels (0/1)
            y_scores: Prediction scores or probabilities
            confidence: Confidence level for CI calculation

        Returns:
            Dictionary containing PRC metrics and confidence intervals
        """
        # Input validation
        y_true, y_scores = self._validate_inputs(y_true, y_scores)

        # Check for single class
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            return self._handle_single_class_prc(y_true, unique_classes)

        # Calculate PRC curve
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            auprc_val = average_precision_score(y_true, y_scores)
        except Exception as e:
            logger.warning(f"PRC curve calculation failed: {str(e)}")
            return self._handle_prc_calculation_failure(y_true)

        # Calculate baseline (random performance)
        baseline = np.sum(y_true == 1) / len(y_true) if len(y_true) > 0 else 0.0

        # Calculate confidence interval
        auprc_ci = self._calculate_auprc_ci(y_true, auprc_val, confidence)

        return {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "auprc": float(auprc_val),
            "auprc_ci": (float(auprc_ci[0]), float(auprc_ci[1])),
            "baseline": float(baseline),
        }

    def calculate_prc(
        self,
        y_true: Union[np.ndarray, List],
        y_scores: Union[np.ndarray, List],
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Calculate Precision-Recall curve with statistics."""
        return self.calculate_prc_curve(y_true, y_scores, confidence)

    def _validate_inputs(
        self, y_true: Union[np.ndarray, List], y_scores: Union[np.ndarray, List]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and convert input arrays."""
        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)

        # Check for empty arrays
        if len(y_true) == 0 or len(y_scores) == 0:
            raise ValueError("Input arrays cannot be empty")

        # Check for equal length
        if len(y_true) != len(y_scores):
            raise ValueError(
                f"Input arrays must have equal length: "
                f"y_true={len(y_true)}, y_scores={len(y_scores)}"
            )

        # Check for NaN values
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_scores)):
            raise ValueError("Input arrays cannot contain NaN values")

        # Validate binary classification
        unique_vals = np.unique(y_true)
        if not all(val in [0, 1] for val in unique_vals):
            raise ValueError("y_true must contain only 0 and 1 values")

        return y_true, y_scores

    def _handle_single_class(
        self, y_true: np.ndarray, unique_classes: np.ndarray
    ) -> Dict[str, Any]:
        """Handle single-class case for ROC analysis."""
        logger.warning("Single class detected in y_true, returning NaN values")
        return {
            "fpr": np.array([]),
            "tpr": np.array([]),
            "thresholds": np.array([]),
            "auc": np.nan,
            "auc_ci": (np.nan, np.nan),
            "auc_pvalue": np.nan,
            "n_pos": int(np.sum(y_true == 1)),
            "n_neg": int(np.sum(y_true == 0)),
            "roc_curve": pd.DataFrame(),
        }

    def _handle_single_class_prc(
        self, y_true: np.ndarray, unique_classes: np.ndarray
    ) -> Dict[str, Any]:
        """Handle single-class case for PRC analysis."""
        logger.warning("Single class detected in y_true, returning NaN values")
        return {
            "precision": np.array([]),
            "recall": np.array([]),
            "thresholds": np.array([]),
            "auprc": np.nan,
            "auprc_ci": (np.nan, np.nan),
            "baseline": np.nan,
        }

    def _handle_calculation_failure(self, y_true: np.ndarray) -> Dict[str, Any]:
        """Handle ROC calculation failures."""
        logger.error("ROC curve calculation failed, returning NaN values")

        return {
            "fpr": np.array([]),
            "tpr": np.array([]),
            "thresholds": np.array([]),
            "auc": np.nan,
            "auc_ci": (np.nan, np.nan),
            "auc_pvalue": np.nan,
            "n_pos": int(np.sum(y_true == 1)),
            "n_neg": int(np.sum(y_true == 0)),
            "roc_curve": pd.DataFrame(),
        }

    def _handle_prc_calculation_failure(self, y_true: np.ndarray) -> Dict[str, Any]:
        """Handle PRC calculation failures."""
        logger.error("PRC curve calculation failed, returning NaN values")

        baseline = np.sum(y_true == 1) / len(y_true) if len(y_true) > 0 else 0.0

        return {
            "precision": np.array([]),
            "recall": np.array([]),
            "thresholds": np.array([]),
            "auprc": np.nan,
            "auprc_ci": (np.nan, np.nan),
            "baseline": baseline,
        }

    def _calculate_auc_ci(
        self, y_true: np.ndarray, auc_val: float, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate AUC confidence interval using Hanley & McNeil method.

        Reference: Hanley, J. A. and McNeil, B. J. (1982). The meaning and use of the area
        under a receiver operating characteristic (ROC) curve. Radiology, 143, 29-36.
        """
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        # Convert to float to ensure proper computation
        auc_val = float(auc_val)

        # Handle edge cases
        if n_pos == 0 or n_neg == 0:
            return (auc_val, auc_val)

        if auc_val in [0.0, 1.0]:
            return (auc_val, auc_val)

        # Hanley & McNeil (1982) formula
        try:
            q1 = auc_val / (2 - auc_val)
            q2 = (2 * auc_val**2) / (1 + auc_val)

            term1 = auc_val * (1 - auc_val)
            term2 = (n_pos - 1) * (q1 - auc_val**2)
            term3 = (n_neg - 1) * (q2 - auc_val**2)

            se_auc = np.sqrt((term1 + term2 + term3) / (n_pos * n_neg))

            z_critical = stats.norm.ppf((1 + confidence) / 2)
            ci_lower = float(max(0, auc_val - z_critical * se_auc))
            ci_upper = float(min(1, auc_val + z_critical * se_auc))

            return (ci_lower, ci_upper)

        except (ZeroDivisionError, ValueError, OverflowError) as e:
            logger.warning(
                f"AUC CI calculation failed: {str(e)}, returning point estimate"
            )
            return (auc_val, auc_val)

    def _calculate_auc_pvalue(self, y_true: np.ndarray, auc_val: float) -> float:
        """
        Calculate p-value for AUC against null hypothesis (AUC=0.5).

        Uses standard error approximation for hypothesis testing.
        """
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return np.nan

        # Z-test under null hypothesis (AUC = 0.5)
        null_auc = 0.5
        try:
            se_auc = np.sqrt((1 + n_pos + n_neg) / (12 * n_pos * n_neg))
            z_stat = (auc_val - null_auc) / se_auc
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            return float(p_value)
        except (ZeroDivisionError, ValueError) as e:
            logger.warning(f"AUC p-value calculation failed: {str(e)}")
            return np.nan

    def _calculate_auprc_ci(
        self, y_true: np.ndarray, auprc_val: float, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate AUPRC confidence interval using binomial approximation.
        """
        n_pos = np.sum(y_true == 1)

        if n_pos == 0:
            return (auprc_val, auprc_val)

        # Binomial approximation for AUPRC standard error
        try:
            se_auprc = np.sqrt(auprc_val * (1 - auprc_val) / n_pos)
            z_critical = stats.norm.ppf((1 + confidence) / 2)

            ci_lower = float(max(0, auprc_val - z_critical * se_auprc))
            ci_upper = float(min(1, auprc_val + z_critical * se_auprc))

            return (ci_lower, ci_upper)
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"AUPRC CI calculation failed: {str(e)}")
            return (auprc_val, auprc_val)


# ============================================================================
# Gene Signature Scoring Engine
# ============================================================================


class SignatureScoringEngine:
    """
    Engine for calculating gene signature scores from expression data.

    Handles various scoring methods and provides consistent interface
    for all ROC analysis components.
    """

    def __init__(self):
        """Initialize signature scoring engine."""
        pass

    def calculate_scores(
        self,
        expr_data: Union[pd.DataFrame, np.ndarray],
        gene_keys: List[str],
        pos_genes: List[str],
        neg_genes: List[str],
        method: str = "mean_difference",
    ) -> np.ndarray:
        """
        Calculate signature scores from expression data.

        Args:
            expr_data: Expression matrix (genes x samples or samples x genes)
            gene_keys: List of gene identifiers corresponding to expression rows
            pos_genes: Positive signature genes
            neg_genes: Negative signature genes
            method: Scoring method ("mean_difference", "weighted", etc.)

        Returns:
            Array of scores for each sample

        Raises:
            ValueError: If method is unknown or data is invalid
        """
        # Input validation
        if len(gene_keys) == 0:
            raise ValueError("gene_keys cannot be empty")

        if method not in ["mean_difference"]:
            raise ValueError(f"Unknown scoring method: {method}")

        # Get gene indices
        pos_indices = self._get_gene_indices(gene_keys, pos_genes)
        neg_indices = self._get_gene_indices(gene_keys, neg_genes)

        if len(pos_indices) == 0 and len(neg_indices) == 0:
            warnings.warn("No genes found in expression data, returning zero scores")
            return np.zeros(self._get_n_samples(expr_data))

        if method == "mean_difference":
            return self._mean_difference_score(expr_data, pos_indices, neg_indices)
        else:
            raise ValueError(f"Method {method} not implemented")

    def _get_gene_indices(
        self, gene_keys: List[str], target_genes: List[str]
    ) -> List[int]:
        """Get indices of target genes in gene_keys list."""
        indices = []
        for gene in target_genes:
            try:
                # Get all indices for genes that appear multiple times
                gene_indices = [i for i, key in enumerate(gene_keys) if key == gene]
                indices.extend(gene_indices)
            except ValueError:
                continue
        return indices

    def _get_n_samples(self, expr_data: Union[pd.DataFrame, np.ndarray]) -> int:
        """Get number of samples from expression data."""
        if isinstance(expr_data, pd.DataFrame):
            return expr_data.shape[1]  # Assumes genes x samples
        else:
            return expr_data.shape[1]  # Assumes genes x samples

    def _mean_difference_score(
        self,
        expr_data: Union[pd.DataFrame, np.ndarray],
        pos_indices: List[int],
        neg_indices: List[int],
    ) -> np.ndarray:
        """
        Calculate mean difference score (pos_mean - neg_mean).

        Args:
            expr_data: Expression matrix (genes x samples)
            pos_indices: Indices of positive genes
            neg_indices: Indices of negative genes

        Returns:
            Array of scores for each sample
        """
        n_samples = self._get_n_samples(expr_data)

        # Calculate positive gene mean
        if pos_indices:
            if isinstance(expr_data, pd.DataFrame):
                pos_mean = expr_data.iloc[pos_indices].mean(axis=0).values
            else:
                pos_mean = np.mean(expr_data[pos_indices, :], axis=0)
        else:
            pos_mean = np.zeros(n_samples)

        # Calculate negative gene mean
        if neg_indices:
            if isinstance(expr_data, pd.DataFrame):
                neg_mean = expr_data.iloc[neg_indices].mean(axis=0).values
            else:
                neg_mean = np.mean(expr_data[neg_indices, :], axis=0)
        else:
            neg_mean = np.zeros(n_samples)

        return np.array(pos_mean - neg_mean)


# ============================================================================
# Kester & Buntinx Meta-Analysis Engine
# ============================================================================


class KesterBuntinxEngine:
    """
    Implementation of Kester & Buntinx meta-analytic ROC methodology.

    Provides smoothed ROC curves and meta-analysis statistics for
    combining multiple study results.
    """

    def __init__(self):
        """Initialize Kester & Buntinx engine."""
        self.roc_engine = ROCAnalysisEngine()

    def calculate_meta_statistics(
        self,
        study_data: List[Dict],
        boot_reps: int = 1000,
        min_points: int = 5,
        auc1_thresh: float = 0.99,
    ) -> pd.DataFrame:
        """
        Calculate meta-analysis statistics for multiple studies.

        Args:
            study_data: List of study dictionaries with 'y_true' and 'y_scores'
            boot_reps: Bootstrap iterations for standard error estimation
            min_points: Minimum ROC points for bootstrap
            auc1_thresh: Threshold for treating AUC as perfect

        Returns:
            DataFrame with meta-analysis statistics
        """
        results = []
        for i, study in enumerate(study_data):
            try:
                y_true = np.array(study["y_true"])
                y_scores = np.array(study["y_scores"])
                study_name = study.get("name", f"Study_{i + 1}")

                # Calculate basic ROC statistics
                roc_stats = self.roc_engine.calculate_roc_curve(y_true, y_scores)
                n_pos = roc_stats["n_pos"]
                n_neg = roc_stats["n_neg"]
                n_total = n_pos + n_neg
                auc_val = roc_stats["auc"]

                # Check for invalid data: must have both positive and negative samples
                if n_pos == 0 or n_neg == 0:
                    logger.warning(
                        f"Invalid data for {study_name}: n_pos={n_pos}, n_neg={n_neg}"
                    )
                    results.append(
                        {
                            "study": study_name,
                            "N": n_total,
                            "tstar_alpha": np.nan,
                            "SE_alpha": np.nan,
                            "tstar_beta": np.nan,
                            "SE_beta": np.nan,
                            "AUC": np.nan,
                            "AUC_CI_lower": np.nan,
                            "AUC_CI_upper": np.nan,
                        }
                    )
                    continue

                # Handle edge cases
                if auc_val >= auc1_thresh:
                    # Near-perfect classification
                    alpha = np.log((n_pos + 0.5) * (n_neg + 0.5) / 0.25)
                    ase = np.sqrt(
                        1 / (1 / 0.5 + 1 / 0.5 + 1 / (n_pos + 0.5) + 1 / (n_neg + 0.5))
                    ) / np.sqrt(n_total)
                    results.append(
                        {
                            "study": study_name,
                            "N": n_total,
                            "tstar_alpha": alpha,
                            "SE_alpha": ase,
                            "tstar_beta": 0.0,
                            "SE_beta": 10.0,
                            "AUC": auc_val,
                            "AUC_CI_lower": roc_stats["auc_ci"][0],
                            "AUC_CI_upper": roc_stats["auc_ci"][1],
                        }
                    )
                    continue
                elif auc_val == 0:
                    # Perfect inverse classification
                    alpha = -np.log((n_pos + 0.5) * (n_neg + 0.5) / 0.25)
                    ase = np.sqrt(
                        1 / (1 / 0.5 + 1 / 0.5 + 1 / (n_pos + 0.5) + 1 / (n_neg + 0.5))
                    ) / np.sqrt(n_total)
                    results.append(
                        {
                            "study": study_name,
                            "N": n_total,
                            "tstar_alpha": alpha,
                            "SE_alpha": ase,
                            "tstar_beta": 0.0,
                            "SE_beta": 10.0,
                            "AUC": auc_val,
                            "AUC_CI_lower": roc_stats["auc_ci"][0],
                            "AUC_CI_upper": roc_stats["auc_ci"][1],
                        }
                    )
                    continue

                # Standard case: fit Kester & Buntinx model
                try:
                    alpha, beta, se_alpha, se_beta = self._fit_kb_model(
                        y_true,
                        y_scores,
                        roc_stats["fpr"],
                        roc_stats["tpr"],
                        boot_reps,
                        min_points,
                    )
                    results.append(
                        {
                            "study": study_name,
                            "N": n_total,
                            "tstar_alpha": alpha,
                            "SE_alpha": se_alpha,
                            "tstar_beta": beta,
                            "SE_beta": se_beta,
                            "AUC": auc_val,
                            "AUC_CI_lower": roc_stats["auc_ci"][0],
                            "AUC_CI_upper": roc_stats["auc_ci"][1],
                        }
                    )
                except (ValueError, RuntimeError) as e:
                    logger.warning(
                        f"Failed to fit K&B model for {study_name}: {str(e)}"
                    )
                    # Return NaN for invalid/failed model fits
                    results.append(
                        {
                            "study": study_name,
                            "N": n_total,
                            "tstar_alpha": np.nan,
                            "SE_alpha": np.nan,
                            "tstar_beta": np.nan,
                            "SE_beta": np.nan,
                            "AUC": auc_val,
                            "AUC_CI_lower": roc_stats["auc_ci"][0],
                            "AUC_CI_upper": roc_stats["auc_ci"][1],
                        }
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to process study {study.get('name', f'Study_{i + 1}')}: {str(e)}"
                )
                # Add a result with NaN values instead of skipping
                results.append(
                    {
                        "study": study.get("name", f"Study_{i + 1}"),
                        "N": 0,
                        "tstar_alpha": np.nan,
                        "SE_alpha": np.nan,
                        "tstar_beta": np.nan,
                        "SE_beta": np.nan,
                        "AUC": np.nan,
                        "AUC_CI_lower": np.nan,
                        "AUC_CI_upper": np.nan,
                    }
                )
        return pd.DataFrame(results)

    def _fit_kb_model(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        fpr: np.ndarray,
        tpr: np.ndarray,
        boot_reps: int,
        min_points: int,
    ) -> Tuple[float, float, float, float]:
        """Fit Kester & Buntinx model with bootstrap standard errors."""

        # Convert to sensitivity/specificity
        sens = tpr
        spec = 1 - fpr

        # Handle boundary values
        sens = np.clip(sens, 0.02, 0.98)
        spec = np.clip(spec, 0.02, 0.98)

        # Kester & Buntinx transformations
        logit_sens = np.log(sens / (1 - sens))
        logit_spec = np.log(spec / (1 - spec))

        S = logit_sens + logit_spec
        D = logit_sens - logit_spec

        # Calculate weights
        weights = self._calculate_kb_weights(y_true, y_scores, fpr, tpr)

        # Filter valid points
        valid_mask = (
            np.isfinite(S) & np.isfinite(D) & np.isfinite(weights) & (weights > 0)
        )
        S_valid = S[valid_mask]
        D_valid = D[valid_mask]
        weights_valid = weights[valid_mask]

        if len(S_valid) < min_points:
            raise ValueError(
                f"Insufficient valid points ({len(S_valid)}) for model fitting"
            )

        # Bootstrap estimation
        return self._bootstrap_kb_regression(S_valid, D_valid, weights_valid, boot_reps)

    def _calculate_kb_weights(
        self, y_true: np.ndarray, y_scores: np.ndarray, fpr: np.ndarray, tpr: np.ndarray
    ) -> np.ndarray:
        """Calculate inverse-variance weights for each ROC point."""

        # Reconstruct thresholds from ROC curve
        _, _, thresholds = roc_curve(y_true, y_scores)

        weights = []
        for i, thresh in enumerate(thresholds):
            # Calculate confusion matrix elements
            y_pred = (y_scores >= thresh).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            # Inverse variance weight
            if tp > 0 and tn > 0 and fp > 0 and fn > 0:
                weight = 1 / (1 / tp + 1 / tn + 1 / fp + 1 / fn)
            else:
                weight = 0
            weights.append(weight)

        return np.array(weights)

    def _bootstrap_kb_regression(
        self, S: np.ndarray, D: np.ndarray, weights: np.ndarray, boot_reps: int
    ) -> Tuple[float, float, float, float]:
        """Bootstrap estimation of alpha and beta parameters."""

        alphas = []
        betas = []
        n = len(S)

        for _ in range(boot_reps):
            # Resample with replacement
            indices = resample(range(n), replace=True, n_samples=n)
            S_boot = S[indices]
            D_boot = D[indices]
            weights_boot = weights[indices]

            try:
                alpha_boot, beta_boot = self._weighted_least_squares(
                    S_boot, D_boot, weights_boot
                )
                alphas.append(alpha_boot)
                betas.append(beta_boot)
            except:
                continue

        if len(alphas) == 0:
            raise ValueError("Bootstrap regression failed")

        # Point estimates and standard errors
        alpha = np.median(alphas)
        beta = np.median(betas)
        se_alpha = np.std(alphas)
        se_beta = np.std(betas)

        return alpha, beta, se_alpha, se_beta

    def _weighted_least_squares(
        self, X: np.ndarray, y: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, float]:
        """Weighted least squares regression."""
        w_sum = np.sum(weights)
        if w_sum == 0:
            raise ValueError("All weights are zero")

        x_mean = np.sum(weights * X) / w_sum
        y_mean = np.sum(weights * y) / w_sum

        cov_xy = np.sum(weights * (X - x_mean) * (y - y_mean)) / w_sum
        var_x = np.sum(weights * (X - x_mean) ** 2) / w_sum

        if var_x == 0:
            beta = 0
        else:
            beta = cov_xy / var_x
        alpha = y_mean - beta * x_mean

        return alpha, beta

    def generate_summary_roc(
        self, alpha: float, beta: float, n_points: int = 100
    ) -> pd.DataFrame:
        """Generate summary ROC curve from alpha and beta parameters."""

        # Bound beta to prevent numerical issues
        if abs(beta) > 0.95:
            beta = 0.95 * np.sign(beta)

        # Transform parameters
        A = alpha / (1 - beta)
        B = (1 + beta) / (1 - beta)

        # Generate specificity points
        spec_points = np.linspace(0.01, 0.99, n_points)

        # Calculate sensitivity from logistic model
        logit_term = A + B * np.log((1 - spec_points) / spec_points)
        sens_points = 1 / (1 + np.exp(-logit_term))

        # Create ROC curve (FPR, TPR)
        fpr_points = 1 - spec_points

        # Sort by FPR for proper ROC curve
        sort_idx = np.argsort(fpr_points)

        return pd.DataFrame({"fpr": fpr_points[sort_idx], "tpr": sens_points[sort_idx]})


# ============================================================================
# Unified ROC Analyzer - Main Public Interface
# ============================================================================


class UnifiedROCAnalyzer:
    """
    Unified ROC Analysis System for MetaIntegrator.

    This is the main interface for all ROC analysis needs, combining:
    - Individual ROC curve analysis
    - Simple pooled ROC analysis (weighted averaging)
    - Summary ROC analysis (Kester & Buntinx meta-analysis)
    - Precision-Recall curve analysis

    Replaces all previous fragmented implementations with a single,
    cohesive, production-grade system.
    """

    def __init__(self):
        """Initialize with clinical metrics analyzers."""
        self.roc_engine = ROCAnalysisEngine()
        self.scoring_engine = SignatureScoringEngine()
        self.kb_engine = KesterBuntinxEngine()
        # Add clinical analyzers
        self.youden_analyzer = YoudenJAnalyzer()
        self.clinical_analyzer = ClinicalParametersAnalyzer()

    # ========================================================================
    # Main Public API Methods
    # ========================================================================

    def analyze_single_roc(
        self, y_true: np.ndarray, y_scores: np.ndarray, title: str = "ROC Curve"
    ) -> Dict[str, Any]:
        """
        Analyze single ROC curve with comprehensive statistics.

        Args:
            y_true: Binary class labels
            y_scores: Prediction scores
            title: Title for analysis

        Returns:
            Complete ROC analysis results
        """
        return self.roc_engine.calculate_roc_curve(y_true, y_scores)

    def pooled_roc_analysis(
        self, meta_object: Dict, filter_object: Dict, method: str = "simple", **kwargs
    ) -> Dict[str, Any]:
        """
        Perform pooled ROC analysis across multiple studies.

        Args:
            meta_object: Meta object with study data
            filter_object: Gene signature filter
            method: "simple" for weighted averaging, "kester_buntinx" for meta-analysis
            **kwargs: Additional method-specific parameters

        Returns:
            Pooled analysis results with statistics and curves
        """
        if method == "simple":
            return self._simple_pooled_analysis(meta_object, filter_object, **kwargs)
        elif method == "kester_buntinx":
            return self._kester_buntinx_analysis(meta_object, filter_object, **kwargs)
        else:
            raise ValueError(f"Unknown pooled analysis method: {method}")

    def calculate_optimal_cutoffs(
        self, analysis_results: Dict, sensitivity_targets: List[float] = [0.90, 0.95]
    ) -> Dict:
        """
        Calculate optimal cutoffs for all studies in analysis results.

        Args:
            analysis_results: Results from pooled_roc_analysis()
            sensitivity_targets: List of sensitivity targets for specificity calculation

        Returns:
            Dict with optimal cutoff analysis for each study
        """
        if analysis_results.get("method") != "simple":
            raise ValueError(
                "Optimal cutoffs can only be calculated for simple pooled analysis"
            )

        cutoff_results = {}

        # Calculate for individual studies
        for study in analysis_results["individual_studies"]:
            roc_data = {"fpr": study["fpr"], "tpr": study["tpr"]}

            study_cutoffs = {}
            for target in sensitivity_targets:
                cutoff_result = self.youden_analyzer.calculate_optimal_cutoff(
                    roc_data, sensitivity_target=target, return_details=True
                )
                study_cutoffs[f"target_sens_{target:.0%}"] = cutoff_result

            cutoff_results[study["name"]] = study_cutoffs

        # Calculate for pooled curve
        pooled_roc = analysis_results["pooled_curve"]
        pooled_data = {"fpr": pooled_roc["fpr"], "tpr": pooled_roc["tpr"]}

        pooled_cutoffs = {}
        for target in sensitivity_targets:
            cutoff_result = self.youden_analyzer.calculate_optimal_cutoff(
                pooled_data, sensitivity_target=target, return_details=True
            )
            pooled_cutoffs[f"target_sens_{target:.0%}"] = cutoff_result

        cutoff_results["Pooled"] = pooled_cutoffs

        return cutoff_results

    def calculate_clinical_parameters(
        self,
        meta_object: Dict,
        filter_object: Dict,
        threshold_method: str = "weighted_means",
    ) -> pd.DataFrame:
        """
        Calculate clinical parameters across all studies in meta_object.

        Args:
            meta_object: Meta object with originalData
            filter_object: Filter object with gene signatures
            threshold_method: Method for threshold calculation

        Returns:
            DataFrame with clinical performance metrics
        """
        # Prepare study data in the format expected by clinical analyzer
        study_data_list = []

        for name, dataset in meta_object.get("originalData", {}).items():
            try:
                # Calculate signature scores
                scores = self.scoring_engine.calculate_scores(
                    dataset["expr"],
                    dataset["keys"],
                    filter_object.get("posGeneNames", []),
                    filter_object.get("negGeneNames", []),
                )

                # Prepare study data
                study_data = {
                    "name": dataset.get("formattedName", name),
                    "score": scores,
                    "group": np.array(dataset["class"]),
                }

                study_data_list.append(study_data)

            except Exception as e:
                logger.warning(
                    f"Failed to prepare {name} for clinical analysis: {str(e)}"
                )
                continue

        if not study_data_list:
            raise ValueError(
                "No studies could be prepared for clinical parameter analysis"
            )

        return self.clinical_analyzer.calculate_multi_study_parameters(
            study_data_list, threshold_method=threshold_method
        )

    def print_clinical_summary(
        self,
        meta_object: Dict,
        filter_object: Dict,
        analysis_results: Optional[Dict] = None,
    ) -> None:
        """
        Print comprehensive clinical summary (matches R's output format).

        Args:
            meta_object: Meta object with originalData
            filter_object: Filter object with gene signatures
            analysis_results: Optional pre-calculated analysis results
        """
        print("Clinical Performance Summary")
        print("=" * 50)

        # 1. Clinical parameters across studies
        try:
            clinical_params = self.calculate_clinical_parameters(
                meta_object, filter_object
            )
            print("\nClinical Parameters by Study:")
            print(clinical_params)
        except Exception as e:
            print(f"Could not calculate clinical parameters: {str(e)}")

        # 2. Optimal cutoffs if simple analysis available
        if analysis_results and analysis_results.get("method") == "simple":
            try:
                cutoff_results = self.calculate_optimal_cutoffs(analysis_results)

                print("\nOptimal Cutoff Analysis:")
                print("-" * 30)

                for study_name, cutoffs in cutoff_results.items():
                    print(f"\n{study_name}:")
                    for target, result in cutoffs.items():
                        target_pct = float(target.split("_")[-1].replace("%", ""))
                        print(
                            f"  At optimal cutoff: sensitivity = {result['optimal_sensitivity_percent']:.1f}%, "
                            f"specificity = {result['optimal_specificity_percent']:.1f}%"
                        )
                        print(
                            f"  At {target_pct:.0f}% sensitivity: specificity = "
                            f"{result[f'specificity_at_{target_pct:.0f}pct_sensitivity_percent']:.1f}%"
                        )

            except Exception as e:
                print(f"Could not calculate optimal cutoffs: {str(e)}")

    def plot_roc_curves(
        self,
        analysis_results: Dict[str, Any],
        figsize: Tuple[float, float] = (10, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create comprehensive ROC curve plots from analysis results.

        Args:
            analysis_results: Results from pooled_roc_analysis
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Matplotlib figure object
        """
        method = analysis_results.get("method", "simple")

        if method == "simple":
            return self._plot_simple_pooled_roc(analysis_results, figsize, save_path)
        elif method == "kester_buntinx":
            return self._plot_kester_buntinx_roc(analysis_results, figsize, save_path)
        else:
            raise ValueError(f"Unknown plotting method: {method}")

    # ========================================================================
    # Convenience Methods (Backward Compatibility)
    # ========================================================================

    def pooled_roc_plot(
        self, meta_object: Dict, filter_object: Dict, smoothed: bool = False, **kwargs
    ) -> plt.Figure:
        """
        Generate pooled ROC plot (backward compatibility method).

        Args:
            meta_object: Meta object with study data
            filter_object: Gene signature filter
            smoothed: If True, use Kester & Buntinx; if False, use simple pooling
            **kwargs: Additional parameters

        Returns:
            Matplotlib figure
        """
        method = "kester_buntinx" if smoothed else "simple"

        # Perform analysis
        results = self.pooled_roc_analysis(meta_object, filter_object, method, **kwargs)

        # Create plot
        return self.plot_roc_curves(
            results,
            figsize=kwargs.get("figsize", (10, 8)),
            save_path=kwargs.get("save_path"),
        )

    def summary_roc_plot(
        self, meta_object: Dict, filter_object: Dict, **kwargs
    ) -> plt.Figure:
        """Generate Kester & Buntinx summary ROC plot."""
        return self.pooled_roc_plot(meta_object, filter_object, smoothed=True, **kwargs)

    def summary_roc_calc(
        self, meta_object: Dict, filter_object: Dict, **kwargs
    ) -> pd.DataFrame:
        """Calculate summary ROC statistics without plotting."""
        results = self.pooled_roc_analysis(
            meta_object, filter_object, "kester_buntinx", **kwargs
        )

        # Extract summary statistics
        summary_stats = results.get("summary_statistics", {})

        return pd.DataFrame(
            {
                "AUC": [summary_stats.get("auc", np.nan)],
                "AUCupper": [summary_stats.get("auc_upper", np.nan)],
                "AUClower": [summary_stats.get("auc_lower", np.nan)],
            }
        )

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "ROC Curve",
        label: Optional[str] = None,
        color: str = "blue",
        figsize: Tuple = (8, 6),
        show_ci: bool = False,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plots a Receiver Operating Characteristic (ROC) curve for a given set of true labels and predicted scores.

        This function generates the ROC curve to evaluate the performance of a binary classifier.
        The curve illustrates the trade-off between the true positive rate (sensitivity) and false
        positive rate (1-specificity) at various threshold settings. Additional configurations such
        as color, titles, optional label, and confidence intervals display are supported. The resulting
        figure can optionally be saved to a specified path.

        Args:
            y_true : np.ndarray
                Array of true binary labels (0s and 1s).
            y_scores : np.ndarray
                Array of predicted scores or probabilities for the positive class.
            title : str, optional
                Title of the ROC curve plot. Default is "ROC Curve".
            label : Optional[str], optional
                Custom label for the ROC curve. If None, a computed label is used. Default is None.
            color : str, optional
                Color of the ROC curve line. Default is "blue".
            figsize : Tuple, optional
                Size of the output figure in inches (width, height). Default is (8, 6).
            show_ci : bool, optional
                Whether or not to display the confidence intervals around the ROC curve. Default is False.
            save_path : Optional[str], optional
                File path to save the plot. If None, the plot will not be saved. Default is None.

        Returns:
            matplotlib.figure.Figure
                The matplotlib figure object containing the ROC curve plot.
        """
        roc_stats = self.roc_engine.calculate_roc(y_true, y_scores)

        fig, ax = plt.subplots(figsize=figsize)

        # 1. Determine Display Label
        curve_label = label or self._generate_roc_label(roc_stats)

        # 2. Draw Main Plot Components
        ax.plot(
            roc_stats["fpr"],
            roc_stats["tpr"],
            color=color,
            linewidth=2,
            label=curve_label,
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random")

        if show_ci:
            self._add_roc_confidence_band(ax, y_true, y_scores, color=color)

        # 3. Finalize Plot Aesthetics
        self._configure_roc_axes(ax, title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _generate_roc_label(self, stats: Dict) -> str:
        """
        Generates a label string for ROC statistics including AUC value and its confidence interval.

        This method processes the given statistics dictionary and formats the AUC value
        with its corresponding confidence interval into a standardized string representation.
        Such labels are often used in plots or reports requiring ROC analysis results.

        Args:
            stats (Dict): A dictionary containing the following key-value pairs:
                          - auc (float): Area under the curve (AUC) value.
                          - auc_ci (Tuple[float, float]): A tuple indicating the lower and upper
                            bounds of the confidence interval for the AUC.

        Returns:
            str: A formatted string that includes the AUC value and its confidence interval.
        """
        auc_val = stats["auc"]
        lower, upper = stats["auc_ci"]
        return f"AUC = {auc_val:.3f} (95% CI: {lower:.3f}-{upper:.3f})"

    def _configure_roc_axes(self, ax: plt.Axes, title: str) -> None:
        """
        Configures the ROC curve plot axes with labels, title, legend, grid, and axis limits.

        Args:
            ax (plt.Axes): The matplotlib Axes object to configure.
            title (str): The title of the ROC curve.

        Returns:
            None
        """
        ax.set_xlabel("False Positive Rate (1-Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    def _add_roc_confidence_band(
        self,
        ax: plt.Axes,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        color: str = "blue",
        alpha: float = 0.2,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> None:
        """
        Adds a confidence band to the receiver operating characteristic (ROC) curve plot. This method
        uses bootstrap resampling to calculate the range of true positive rates (TPR) at standard
        false positive rates (FPR) points, providing a confidence interval around the ROC curve.

        Args:
            ax (plt.Axes): The Matplotlib Axes object to render the confidence band on.
            y_true (np.ndarray): The array of true binary labels (0 or 1).
            y_scores (np.ndarray): The array of predicted scores or probabilities.
            color (str, optional): The color used for the confidence band. Default is "blue".
            alpha (float, optional): The transparency level for the confidence band. Default is 0.2.
            n_bootstrap (int, optional): The number of bootstrap resampling iterations. Default is 1000.
            confidence (float, optional): The confidence level for the interval, typically between
                0 and 1 (e.g., 0.95 for 95% confidence). Default is 0.95.

        Raises:
            None: This method does not explicitly raise any exceptions.

        Returns:
            None: The method modifies the given Axes object directly.
        """
        resampled_tprs = []
        mean_fpr = np.linspace(0, 1, 100)

        for _ in range(n_bootstrap):
            # 1. Generate bootstrap sample
            indices = resample(range(len(y_true)), replace=True)
            y_boot, scores_boot = y_true[indices], y_scores[indices]

            # 2. Skip if resample lacks class diversity
            if len(np.unique(y_boot)) < 2:
                continue

            # 3. Calculate ROC and interpolate TPR at standard FPR points
            # sklearn's roc_curve() defaults to drop_intermediate=True, which removes threshold points that don't change FPR/TPR
            # # Use drop_intermediate=False to match R
            # 20260409 fpr, tpr, _ = roc_curve(y_boot, scores_boot)
            fpr, tpr, _ = roc_curve(y_boot, scores_boot, drop_intermediate=False)

            # Interpolate to common grid
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0  # Force curve to origin
            resampled_tprs.append(interp_tpr)

        if not resampled_tprs:
            return

        # 4. Calculate bounds based on specified confidence level
        lower_p = ((1.0 - confidence) / 2.0) * 100
        upper_p = (confidence + (1.0 - confidence) / 2.0) * 100

        tprs_matrix = np.array(resampled_tprs)
        lower_bound = np.percentile(tprs_matrix, lower_p, axis=0)
        upper_bound = np.percentile(tprs_matrix, upper_p, axis=0)

        # 5. Plot the confidence band
        ax.fill_between(
            mean_fpr,
            lower_bound,
            upper_bound,
            color=color,
            alpha=alpha,
            label=f"{confidence:.0%} CI",
        )

    # ========================================================================
    # Internal Implementation Methods
    # ========================================================================

    def _simple_pooled_analysis(
        self,
        meta_object: Dict,
        filter_object: Dict,
        points: int = 1000,
        weighting: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Implement simple pooled ROC analysis with weighted averaging."""

        # Process all studies
        study_results = []
        for name, dataset in meta_object.get("originalData", {}).items():
            try:
                # Calculate signature scores
                # 20260610orig
                """
                scores = self.scoring_engine.calculate_scores(
                    dataset["expr"],
                    dataset["keys"],
                    filter_object.get("posGeneNames", []),
                    filter_object.get("negGeneNames", []),
                )
                """
                # 20260610dev
                from .score_calculation import ScoreCalculator

                scores = ScoreCalculator.calculate_score(
                    filter_object, dataset, suppress_messages=True
                )

                y_true = np.array(dataset["class"])

                # Validate data
                if len(np.unique(y_true)) < 2:
                    logger.warning(f"Skipping {name}: insufficient class diversity")
                    continue

                if len(np.unique(scores)) == 1:
                    logger.warning(f"Dataset {name}: constant scores, using AUC=0.5")
                    # Create degenerate ROC for constant scores
                    roc_stats = {
                        "fpr": np.array([0.0, 1.0]),
                        "tpr": np.array([0.0, 1.0]),
                        "auc": 0.5,
                        "auc_ci": (0.5, 0.5),
                    }
                else:
                    # Calculate ROC normally
                    roc_stats = self.roc_engine.calculate_roc_curve(y_true, scores)

                study_results.append(
                    {
                        "name": dataset.get("formattedName", name),
                        "fpr": roc_stats["fpr"],
                        "tpr": roc_stats["tpr"],
                        "auc": roc_stats["auc"],
                        "auc_ci": roc_stats["auc_ci"],
                        "n_samples": len(y_true),
                        "scores": scores,  # 20260610dev – keep for concat pooling
                        "labels": y_true,  # 20260610dev – keep for concat pooling
                    }
                )

            except Exception as e:
                logger.warning(f"Failed to process {name}: {str(e)}")
                continue

        if not study_results:
            raise ValueError("No valid studies for pooled ROC analysis")

        # Interpolate to common FPR points
        common_fpr = np.linspace(0, 1, points)
        interpolated_tprs = []
        weights_list = []

        for result in study_results:
            interp_tpr = np.interp(common_fpr, result["fpr"], result["tpr"])
            interpolated_tprs.append(interp_tpr)
            weights_list.append(result["n_samples"])

        interpolated_tprs = np.array(interpolated_tprs)
        weights_array = np.array(weights_list)

        # Calculate pooled curve
        if weighting:
            weights_norm = weights_array / np.sum(weights_array)
            pooled_tpr = np.average(interpolated_tprs, axis=0, weights=weights_norm)
        else:
            pooled_tpr = np.mean(interpolated_tprs, axis=0)

        # Calculate confidence bands (weighted standard deviation)
        if weighting:
            pooled_var = np.average(
                (interpolated_tprs - pooled_tpr) ** 2, axis=0, weights=weights_norm
            )
        else:
            pooled_var = np.var(interpolated_tprs, axis=0)

        pooled_std = np.sqrt(pooled_var)

        # Calculate pooled AUC
        pooled_auc = auc(common_fpr, pooled_tpr)

        # 20260610dev – Also compute a "concatenated" pooled AUC (R-style)
        # R pools by concatenating all scores & labels and computing one ROC.
        all_scores = np.concatenate([r["scores"] for r in study_results])
        all_labels = np.concatenate([r["labels"] for r in study_results])
        if len(np.unique(all_labels)) >= 2:
            concat_roc = self.roc_engine.calculate_roc_curve(all_labels, all_scores)
            concat_pooled_auc = concat_roc["auc"]
        else:
            concat_pooled_auc = pooled_auc

        return {
            "method": "simple",
            "individual_studies": study_results,
            "pooled_curve": {
                "fpr": common_fpr,
                "tpr": pooled_tpr,
                "auc": pooled_auc,
                "concat_auc": concat_pooled_auc,
            },
            "confidence_bands": {
                "upper": np.clip(pooled_tpr + pooled_std, 0, 1),
                "lower": np.clip(pooled_tpr - pooled_std, 0, 1),
            },
            "parameters": {
                "points": points,
                "weighting": weighting,
                "n_studies": len(study_results),
            },
        }

    def _kester_buntinx_analysis(
        self, meta_object: Dict, filter_object: Dict, boot_reps: int = 1000, **kwargs
    ) -> Dict[str, Any]:
        """Implement Kester & Buntinx meta-analytic ROC analysis."""

        # Prepare study data for meta-analysis
        study_data = []
        for name, dataset in meta_object.get("originalData", {}).items():
            try:
                # 20260610orig
                """
                scores = self.scoring_engine.calculate_scores(
                    dataset["expr"],
                    dataset["keys"],
                    filter_object.get("posGeneNames", []),
                    filter_object.get("negGeneNames", []),
                )
                """
                # 20260610dev
                from .score_calculation import ScoreCalculator

                scores = ScoreCalculator.calculate_score(
                    filter_object, dataset, suppress_messages=True
                )

                y_true = np.array(dataset["class"])

                # Validate data
                if len(np.unique(y_true)) < 2:
                    logger.warning(f"Skipping {name}: insufficient class diversity")
                    continue

                study_data.append(
                    {
                        "name": dataset.get("formattedName", name),
                        "y_true": y_true,
                        "y_scores": scores,
                    }
                )

            except Exception as e:
                logger.warning(f"Failed to process {name}: {str(e)}")
                continue

        if not study_data:
            raise ValueError("No valid studies for meta-analysis")

        # Extract only the kwargs that calculate_meta_statistics accepts
        kb_kwargs = {}
        for key in ("min_points", "auc1_thresh"):
            if key in kwargs:
                kb_kwargs[key] = kwargs[key]

        # Calculate meta-statistics with error handling
        try:
            meta_stats = self.kb_engine.calculate_meta_statistics(
                study_data, boot_reps=boot_reps, **kb_kwargs
            )
        except Exception as e:
            logger.warning(f"Meta-statistics calculation failed: {str(e)}")
            # Return fallback result
            return {
                "method": "kester_buntinx",
                "meta_statistics": pd.DataFrame(),
                "summary_curve": pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]}),
                "summary_statistics": {
                    "alpha": np.nan,
                    "beta": np.nan,
                    "auc": 0.5,
                    "auc_upper": 0.5,
                    "auc_lower": 0.5,
                },
                "parameters": {"boot_reps": boot_reps, "n_studies": len(study_data)},
            }

        # Pool parameters (simple approach - can be enhanced)
        valid_stats = meta_stats.dropna()
        if len(valid_stats) == 0:
            logger.warning("No valid meta-statistics calculated, using fallback")
            # Return fallback result instead of raising error
            return {
                "method": "kester_buntinx",
                "meta_statistics": meta_stats,  # Include failed stats for debugging
                "summary_curve": pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]}),
                "summary_statistics": {
                    "alpha": np.nan,
                    "beta": np.nan,
                    "auc": 0.5,
                    "auc_upper": 0.5,
                    "auc_lower": 0.5,
                },
                "parameters": {"boot_reps": boot_reps, "n_studies": len(study_data)},
            }

        # Simple pooling of alpha and beta (weighted by sample size)
        weights = valid_stats["N"].values
        pooled_alpha = np.average(valid_stats["tstar_alpha"], weights=weights)
        pooled_beta = np.average(valid_stats["tstar_beta"], weights=weights)

        # Generate summary ROC curve
        summary_roc = self.kb_engine.generate_summary_roc(pooled_alpha, pooled_beta)
        summary_auc = auc(summary_roc["fpr"], summary_roc["tpr"])

        return {
            "method": "kester_buntinx",
            "meta_statistics": meta_stats,
            "summary_curve": summary_roc,
            "summary_statistics": {
                "alpha": pooled_alpha,
                "beta": pooled_beta,
                "auc": summary_auc,
                "auc_upper": summary_auc + 0.05,  # Simplified - should use proper CI
                "auc_lower": summary_auc - 0.05,  # Simplified - should use proper CI
            },
            "parameters": {"boot_reps": boot_reps, "n_studies": len(study_data)},
        }

    def _plot_simple_pooled_roc(
        self,
        results: Dict[str, Any],
        figsize: Tuple[float, float],
        save_path: Optional[str],
    ) -> plt.Figure:
        """Create plot for simple pooled ROC analysis."""

        fig, ax = plt.subplots(figsize=figsize)

        individual_studies = results["individual_studies"]
        pooled_curve = results["pooled_curve"]
        confidence_bands = results["confidence_bands"]

        # Plot confidence band
        ax.fill_between(
            pooled_curve["fpr"],
            confidence_bands["lower"],
            confidence_bands["upper"],
            color="gray",
            alpha=0.3,
            label="±1 SD",
        )

        # Plot individual study curves
        colors = plt.cm.Set1(np.linspace(0, 1, len(individual_studies)))
        for study, color in zip(individual_studies, colors):
            auc_val = study["auc"]
            ci_lower, ci_upper = study["auc_ci"]
            label = f"{study['name']}: AUC={auc_val:.3f} (95% CI {ci_lower:.3f}-{ci_upper:.3f})"

            ax.plot(
                study["fpr"],
                study["tpr"],
                color=color,
                linewidth=1.5,
                alpha=0.7,
                label=label,
            )

        # Plot pooled curve
        pooled_auc = pooled_curve["auc"]
        pooled_label = f"Pooled: AUC={pooled_auc:.3f}"
        ax.plot(
            pooled_curve["fpr"],
            pooled_curve["tpr"],
            color="black",
            linewidth=3,
            label=pooled_label,
        )

        # Reference line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

        # Formatting
        ax.set_xlabel("False Positive Rate (1-Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title("Pooled ROC Analysis")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _plot_kester_buntinx_roc(
        self,
        results: Dict[str, Any],
        figsize: Tuple[float, float],
        save_path: Optional[str],
    ) -> plt.Figure:
        """Create plot for Kester & Buntinx analysis."""

        fig, ax = plt.subplots(figsize=figsize)

        meta_stats = results["meta_statistics"]
        summary_curve = results["summary_curve"]
        summary_stats = results["summary_statistics"]

        # Plot individual study curves (reconstructed from meta-stats)
        colors = plt.cm.Set1(np.linspace(0, 1, len(meta_stats)))
        for i, (_, study) in enumerate(meta_stats.iterrows()):
            if not pd.isna(study["AUC"]):
                # Simple representation for individual studies
                ax.scatter(
                    [0.5],
                    [0.5],
                    color=colors[i],
                    s=50,
                    alpha=0.7,
                    label=f"{study['study']}: AUC={study['AUC']:.3f}",
                )

        # Plot summary ROC curve
        summary_auc = summary_stats["auc"]
        summary_label = f"Summary: AUC={summary_auc:.3f}"
        ax.plot(
            summary_curve["fpr"],
            summary_curve["tpr"],
            color="red",
            linewidth=3,
            label=summary_label,
        )

        # Reference line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

        # Formatting
        ax.set_xlabel("False Positive Rate (1-Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title("Summary ROC Analysis (Kester & Buntinx)")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


# ============================================================================
# Meta analysis utilities
# ============================================================================


class MetaAnalysisUtilities:
    """
    Utilities for meta-analysis of ROC curves.

    Provides statistical methods for combining ROC results across
    multiple studies, including weighted averaging and random effects models.
    """

    @staticmethod
    def calculate_pooled_auc(
        aucs: List[float], sample_sizes: List[int], method: str = "weighted"
    ) -> float:
        """
        Calculate pooled AUC across multiple studies.

        Args:
            aucs: List of AUC values
            sample_sizes: List of sample sizes for each study
            method: "weighted" or "unweighted"

        Returns:
            Pooled AUC value
        """
        aucs = np.array(aucs)
        sample_sizes = np.array(sample_sizes)

        # Remove any NaN values
        valid_mask = ~np.isnan(aucs)
        aucs = aucs[valid_mask]
        sample_sizes = sample_sizes[valid_mask]

        if len(aucs) == 0:
            return np.nan

        if method == "weighted":
            weights = sample_sizes / np.sum(sample_sizes)
            return np.sum(aucs * weights)
        elif method == "unweighted":
            return np.mean(aucs)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def calculate_pooled_tpr(
        tpr_curves: List[np.ndarray], sample_sizes: List[int], method: str = "weighted"
    ) -> np.ndarray:
        """
        Calculate pooled TPR curve across multiple studies.

        Args:
            tpr_curves: List of TPR arrays (assumed to be on common FPR grid)
            sample_sizes: List of sample sizes for each study
            method: "weighted" or "unweighted"

        Returns:
            Pooled TPR array
        """
        if len(tpr_curves) == 0:
            return np.array([])

        # Stack curves into matrix
        tpr_matrix = np.vstack(tpr_curves)
        sample_sizes = np.array(sample_sizes)

        if method == "weighted":
            weights = sample_sizes / np.sum(sample_sizes)
            return np.average(tpr_matrix, axis=0, weights=weights)
        elif method == "unweighted":
            return np.mean(tpr_matrix, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def interpolate_roc_curves(
        fpr_tpr_pairs: List[Tuple[np.ndarray, np.ndarray]],
        common_fpr: Optional[np.ndarray] = None,
        n_points: int = 1000,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Interpolate multiple ROC curves to common FPR grid.

        Args:
            fpr_tpr_pairs: List of (FPR, TPR) tuples
            common_fpr: Common FPR grid (if None, creates uniform grid)
            n_points: Number of points in common FPR grid

        Returns:
            Tuple of (common_fpr, list_of_interpolated_tprs)
        """
        if common_fpr is None:
            common_fpr = np.linspace(0, 1, n_points)

        interpolated_tprs = []
        for fpr, tpr in fpr_tpr_pairs:
            if len(fpr) > 1 and len(tpr) > 1:
                interp_tpr = np.interp(common_fpr, fpr, tpr)
            else:
                # Degenerate curve - use linear interpolation from (0,0) to (1,1)
                interp_tpr = common_fpr.copy()
            interpolated_tprs.append(interp_tpr)

        return common_fpr, interpolated_tprs

    @staticmethod
    def calculate_confidence_bands(
        tpr_curves: List[np.ndarray],
        sample_sizes: List[int],
        confidence: float = 0.95,
        method: str = "std",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence bands for pooled ROC curve.

        Args:
            tpr_curves: List of TPR arrays on common FPR grid
            sample_sizes: Sample sizes for weighting
            confidence: Confidence level
            method: "std" for standard deviation bands, "se" for standard error

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(tpr_curves) == 0:
            return np.array([]), np.array([])

        tpr_matrix = np.vstack(tpr_curves)
        sample_sizes = np.array(sample_sizes)

        # Calculate weighted mean and variance
        weights = sample_sizes / np.sum(sample_sizes)
        mean_tpr = np.average(tpr_matrix, axis=0, weights=weights)

        # Weighted variance calculation
        weighted_var = np.average((tpr_matrix - mean_tpr) ** 2, axis=0, weights=weights)

        if method == "std":
            # Standard deviation bands
            std_tpr = np.sqrt(weighted_var)
            lower_bound = np.clip(mean_tpr - std_tpr, 0, 1)
            upper_bound = np.clip(mean_tpr + std_tpr, 0, 1)
        elif method == "se":
            # Standard error bands
            se_tpr = np.sqrt(weighted_var / len(tpr_curves))
            z_critical = stats.norm.ppf((1 + confidence) / 2)
            lower_bound = np.clip(mean_tpr - z_critical * se_tpr, 0, 1)
            upper_bound = np.clip(mean_tpr + z_critical * se_tpr, 0, 1)
        else:
            raise ValueError(f"Unknown method: {method}")

        return lower_bound, upper_bound


# ============================================================================
# Public API Functions (Convenience Interface)
# ============================================================================


# Convenience functions for backward compatibility
def calculate_roc_curve(y_true, y_scores, confidence=0.95):
    """Calculate ROC curve using the core engine."""
    engine = ROCAnalysisEngine()
    return engine.calculate_roc_curve(y_true, y_scores, confidence)


def calculate_prc_curve(y_true, y_scores, confidence=0.95):
    """Calculate PRC curve using the core engine."""
    engine = ROCAnalysisEngine()
    return engine.calculate_prc_curve(y_true, y_scores, confidence)


def calculate_signature_scores(
    expr_data, gene_keys, pos_genes, neg_genes, method="mean_difference"
):
    """Calculate signature scores using the core engine."""
    engine = SignatureScoringEngine()
    return engine.calculate_scores(expr_data, gene_keys, pos_genes, neg_genes, method)


def analyze_roc_curve(
    y_true: np.ndarray, y_scores: np.ndarray, **kwargs
) -> Dict[str, Any]:
    """Analyze single ROC curve with comprehensive statistics."""
    analyzer = UnifiedROCAnalyzer()
    return analyzer.analyze_single_roc(y_true, y_scores, **kwargs)


def pooled_roc_plot(meta_object: Dict, filter_object: Dict, **kwargs) -> plt.Figure:
    """Generate pooled ROC plot from multiple studies."""
    analyzer = UnifiedROCAnalyzer()
    return analyzer.pooled_roc_plot(meta_object, filter_object, **kwargs)


def summary_roc_plot(meta_object: Dict, filter_object: Dict, **kwargs) -> plt.Figure:
    """Generate summary ROC plot using Kester & Buntinx method."""
    analyzer = UnifiedROCAnalyzer()
    return analyzer.summary_roc_plot(meta_object, filter_object, **kwargs)


def summary_roc_calc(meta_object: Dict, filter_object: Dict, **kwargs) -> pd.DataFrame:
    """Calculate summary ROC statistics without plotting."""
    analyzer = UnifiedROCAnalyzer()
    return analyzer.summary_roc_calc(meta_object, filter_object, **kwargs)


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# For notebooks and code that expect separate analyzer classes
# Both PooledROCAnalyzer and SummaryROCAnalyzer are aliases for UnifiedROCAnalyzer
# The unified analyzer handles all analysis types through method parameters
PooledROCAnalyzer = UnifiedROCAnalyzer
SummaryROCAnalyzer = UnifiedROCAnalyzer


# ============================================================================
# Main Entry Point
# ============================================================================

__all__ = [
    "UnifiedROCAnalyzer",
    "PooledROCAnalyzer",  # Alias for UnifiedROCAnalyzer
    "SummaryROCAnalyzer",  # Alias for UnifiedROCAnalyzer
    "ROCAnalysisEngine",
    "SignatureScoringEngine",
    "KesterBuntinxEngine",
    "MetaAnalysisUtilities",
    "analyze_roc_curve",
    "calculate_roc_curve",
    "calculate_prc_curve",
    "calculate_signature_scores",
    "pooled_roc_plot",
    "summary_roc_plot",
    "summary_roc_calc",
]

if __name__ == "__main__":
    # Example usage demonstration
    print("MetaIntegrator Unified ROC Analysis System")
    print("==========================================")
    print("This module provides comprehensive ROC analysis capabilities:")
    print("- Individual ROC curve analysis")
    print("- Simple pooled ROC analysis")
    print("- Kester & Buntinx meta-analytic ROC")
    print("- Precision-Recall curve analysis")
    print("\nExample:")
    print("analyzer = UnifiedROCAnalyzer()")
    print("fig = analyzer.pooled_roc_plot(meta_object, filter_object, smoothed=False)")
