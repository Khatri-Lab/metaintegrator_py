"""
Score calculation functions for MetaIntegrator Python port.
This module implements calculateScore from the R package.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from .sample_level_gene_data import SampleLevelGeneDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoreCalculator:
    """
    Calculator for gene signature Z-scores.

    This class implements the calculateScore functionality from the R MetaIntegrator package,
    which summarizes gene expression using geometric means and standardizes to Z-scores.
    """

    @staticmethod
    def calculate_score(
        filter_object: Dict, dataset_object: Dict, suppress_messages: bool = False
    ) -> np.ndarray:
        """
        Calculate a signature Z-score for a set of genes in a single dataset.

        Given a gene set of interest, this method calculates the geometric mean of the
        expression level of all positive genes, minus the geometric mean of the expression
        level of all negative genes. The resulting scores are then standardized within the
        given dataset, such that the output Z-score has mean=0 and std. dev=1.

        Args:
            filter_object (Dict): A MetaFilter object containing signature genes.
                Must have 'posGeneNames' and 'negGeneNames' keys.
            dataset_object (Dict): A Dataset object for which the signature score will
                be calculated. Must have 'expr' (DataFrame) and 'keys' (list/array).
            suppress_messages (bool): Whether to display verbose output. Default: False.

        Returns:
            np.ndarray: Array of Z-scores, length ncol(dataset_object['expr'])

        Details:
            The Z-score is based off of the geometric mean of expression. As such, negative
            expression values are not allowed. A dataset is thus always scaled by its minimum
            value + 1, such that the lowest value = 1. Any individual NANs or NAs are also
            set to 1. If a dataset does not have any information on a given gene, the entire
            gene is simply left out of the score.

        Examples:
            >>> filter_obj = {'posGeneNames': ['GENE1', 'GENE2'],
            ...               'negGeneNames': ['GENE3']}
            >>> dataset = {'expr': expr_df, 'keys': gene_keys}
            >>> scores = ScoreCalculator.calculate_score(filter_obj, dataset)
        """
        # Validate inputs
        ScoreCalculator._validate_filter_object(filter_object)
        ScoreCalculator._validate_dataset_object(dataset_object)

        # Get the positive and negative genes, remove empty gene names
        pos_genes = [
            g for g in filter_object.get("posGeneNames", []) if g not in ["", None] and pd.notna(g)
        ]
        neg_genes = [
            g for g in filter_object.get("negGeneNames", []) if g not in ["", None] and pd.notna(g)
        ]

        # Adjust expression values so that they are all positive
        expr_data = dataset_object["expr"].copy()
        min_val = expr_data.min().min()
        if min_val < 0:
            expr_data = expr_data + abs(min_val) + 1

        # Update dataset with positive-only expression
        dataset_temp = dataset_object.copy()
        dataset_temp["expr"] = expr_data

        # Calculate the score of the positive genes from our signature
        # R-parity: pass ALL gene names, not just those found in keys.
        # getSampleLevelGeneData fills missing genes with NaN → 1, so
        # log(1)=0 enters the geometric mean, matching R's behavior.
        pos_score = np.zeros(expr_data.shape[1])
        pos_genes_matrix = None
        if len(pos_genes) > 0:
            pos_genes_matrix = SampleLevelGeneDataProcessor.get_sample_level_gene_data(
                dataset_temp, pos_genes
            )
            pos_score = pos_genes_matrix.apply(ScoreCalculator._geometric_mean, axis=0).values

        # Calculate the score of the negative genes from our signature
        neg_score = np.zeros(expr_data.shape[1])
        neg_genes_matrix = None
        if len(neg_genes) > 0:
            neg_genes_matrix = SampleLevelGeneDataProcessor.get_sample_level_gene_data(
                dataset_temp, neg_genes
            )
            neg_score = neg_genes_matrix.apply(ScoreCalculator._geometric_mean, axis=0).values

        # Handle cases based on what data we have
        if pos_genes_matrix is None and neg_genes_matrix is None:
            if not suppress_messages:
                logger.warning(
                    f"Used 0 of {len(pos_genes)} pos genes, and 0 of {len(neg_genes)} neg genes"
                )
            return np.zeros(expr_data.shape[1])

        if neg_genes_matrix is None:
            total_score = ScoreCalculator._standardize(pos_score)
        elif pos_genes_matrix is None:
            total_score = ScoreCalculator._standardize(-neg_score)
        else:
            total_score = ScoreCalculator._standardize(pos_score - neg_score)

        # Log gene usage
        if not suppress_messages:
            n_pos_used = 0 if pos_genes_matrix is None else len(pos_genes_matrix)
            n_neg_used = 0 if neg_genes_matrix is None else len(neg_genes_matrix)
            logger.info(
                f"Used {n_pos_used} of {len(pos_genes)} pos genes, and "
                f"{n_neg_used} of {len(neg_genes)} neg genes"
            )

        return total_score

    @staticmethod
    def _geometric_mean(x: pd.Series) -> float:
        """
        Calculate geometric mean from a series.

        Args:
            x (pd.Series): Numeric series

        Returns:
            float: Geometric mean

        Raises:
            ValueError: If x contains negative values
        """
        # Remove NaN values
        x_clean = x.dropna()

        if len(x_clean) == 0:
            return 1.0  # Default value for empty series

        # Convert to numpy array
        values = x_clean.values

        # Check for negative values
        if np.any(values < 0):
            raise ValueError("'x' contains negative value(s)")

        # Handle zeros by replacing with very small positive number
        values = np.where(values == 0, 1e-10, values)

        # Calculate geometric mean: exp(mean(log(x)))
        return np.exp(np.mean(np.log(values)))

    @staticmethod
    def _standardize(x: np.ndarray) -> np.ndarray:
        """
        Standardize array to mean=0, std=1 (equivalent to R's scale function).

        Args:
            x (np.ndarray): Input array

        Returns:
            np.ndarray: Standardized array
        """
        mean_x = np.mean(x)
        std_x = np.std(x, ddof=1)  # Use sample std dev (N-1)

        if std_x == 0:
            # If std is 0, all values are the same - return zeros
            return np.zeros_like(x)

        return (x - mean_x) / std_x

    @staticmethod
    def _validate_filter_object(filter_object: Dict) -> None:
        """Validate filter object structure."""
        if not isinstance(filter_object, dict):
            raise TypeError("filter_object must be a dictionary")

        if "posGeneNames" not in filter_object and "negGeneNames" not in filter_object:
            raise ValueError("filter_object must contain at least 'posGeneNames' or 'negGeneNames'")

    @staticmethod
    def _validate_dataset_object(dataset_object: Dict) -> None:
        """Validate dataset object structure."""
        if not isinstance(dataset_object, dict):
            raise TypeError("dataset_object must be a dictionary")

        required_keys = ["expr", "keys"]
        missing_keys = [key for key in required_keys if key not in dataset_object]
        if missing_keys:
            raise ValueError(f"dataset_object missing required keys: {missing_keys}")

        if not isinstance(dataset_object["expr"], pd.DataFrame):
            raise TypeError("dataset_object['expr'] must be a pandas DataFrame")


# Convenience function for backward compatibility
def calculateScore(
    filter_object: Dict, dataset_object: Dict, suppress_messages: bool = False
) -> np.ndarray:
    """
    Convenience function matching R interface.

    Calculate a signature Z-score for a set of genes in a single dataset.

    Args:
        filter_object (Dict): MetaFilter object with gene signatures
        dataset_object (Dict): Dataset object with expression data
        suppress_messages (bool): Whether to suppress informational messages

    Returns:
        np.ndarray: Array of Z-scores
    """
    return ScoreCalculator.calculate_score(filter_object, dataset_object, suppress_messages)
