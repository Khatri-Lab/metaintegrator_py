"""
ROC and PRC analysis functions for MetaIntegrator Python port.
This module handles ROC/PRC curve analysis, AUC calculations,
gene search optimization, and related statistical computations.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

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

# Import getSampleLevelGeneData for compatibility
try:
    from .sample_level_gene_data import getSampleLevelGeneData
except ImportError:
    # Fallback if module not found
    def getSampleLevelGeneData(dataset: Dict, genes: List[str]) -> Any:
        """Placeholder used only if :mod:`.sample_level_gene_data` fails to import.

        Always raises, since there is no real implementation to fall
        back to.
        """
        raise ImportError("getSampleLevelGeneData function not available")


from .roc_analysis_engine import (
    ROCAnalysisEngine,
)

# Configure logging
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# GeneSearchOptimizer Class
# ------------------------------------------------------------------
class GeneSearchOptimizer:
    """Class for gene selection optimization using forward and backward search algorithms."""

    def __init__(self):
        """Initialize gene search optimizer."""
        self.roc_analyzer = ROCAnalysisEngine()

    def backward_search(
        self, meta_object: Dict, filter_object: Dict, back_thresh: float = 0.0
    ) -> Dict:
        """
        Backward search for gene selection optimization.

        Iteratively removes genes that contribute least to the overall weighted AUC.

        Args:
            meta_object (Dict): The meta object containing discovery datasets
            filter_object (Dict): Filter object containing gene signatures
            back_thresh (float): Stopping threshold for backward search (default: 0)

        Returns:
            Dict: Updated filter object with optimized gene sets
        """
        # Validate inputs
        self._validate_meta_object(meta_object)
        self._validate_filter_object(filter_object)

        # Extract gene lists
        pos_genes = filter_object.get("posGeneNames", [])
        neg_genes = filter_object.get("negGeneNames", [])

        logger.info(
            f"Starting backward search with {len(pos_genes)} positive and {len(neg_genes)} negative genes"
        )

        # Convert discovery datasets to gene-level data
        discovery_genes = self._convert_discovery_list_to_genes(meta_object, pos_genes, neg_genes)

        # Perform backward search
        optimized_genes = self._backward_search_weighted(
            discovery_genes, pos_genes, neg_genes, back_thresh
        )

        # Update filter object
        updated_filter = filter_object.copy()
        updated_filter["posGeneNames"] = optimized_genes[0]
        updated_filter["negGeneNames"] = optimized_genes[1]

        # Update description
        original_desc = filter_object.get("filterDescription", "")
        updated_filter["filterDescription"] = (
            f"{original_desc}. Backward search with backThresh={back_thresh}"
        )

        logger.info(
            f"Backward search completed. Remaining: {len(optimized_genes[0])} positive, {len(optimized_genes[1])} negative genes"
        )

        return updated_filter

    def forward_search(
        self,
        meta_object: Dict,
        filter_object: Dict,
        yes_pos: Optional[List[str]] = None,
        yes_neg: Optional[List[str]] = None,
        forward_thresh: float = 0.0,
    ) -> Dict:
        """
        Forward search for gene selection optimization.

        Iteratively adds genes that contribute most to the overall weighted AUC.

        Args:
            meta_object (Dict): The meta object containing discovery datasets
            filter_object (Dict): Filter object containing gene signatures
            yes_pos (Optional[List[str]]): Starting positive genes (optional)
            yes_neg (Optional[List[str]]): Starting negative genes (optional)
            forward_thresh (float): Stopping threshold for forward search (default: 0)

        Returns:
            Dict: Updated filter object with optimized gene sets
        """
        # Validate inputs
        self._validate_meta_object(meta_object)
        self._validate_filter_object(filter_object)

        # Extract gene lists
        pos_genes = filter_object.get("posGeneNames", [])
        neg_genes = filter_object.get("negGeneNames", [])

        logger.info(
            f"Starting forward search with {len(pos_genes)} candidate positive and {len(neg_genes)} candidate negative genes"
        )

        # Convert discovery datasets to gene-level data
        discovery_genes = self._convert_discovery_list_to_genes(meta_object, pos_genes, neg_genes)

        # Perform forward search
        optimized_genes = self._forward_search_weighted(
            discovery_genes, pos_genes, neg_genes, yes_pos, yes_neg, forward_thresh
        )

        # Update filter object
        updated_filter = filter_object.copy()
        updated_filter["posGeneNames"] = optimized_genes[0]
        updated_filter["negGeneNames"] = optimized_genes[1]

        # Update description
        original_desc = filter_object.get("filterDescription", "")
        updated_filter["filterDescription"] = (
            f"{original_desc}. Forward search with forwardThresh={forward_thresh}"
        )

        logger.info(
            f"Forward search completed. Selected: {len(optimized_genes[0])} positive, {len(optimized_genes[1])} negative genes"
        )

        return updated_filter

    def _validate_meta_object(self, meta_object: Dict) -> None:
        """Validate meta object structure.

        Raises ValueError if meta_object is not a dict, missing 'originalData',
        'originalData' is not a dict, or empty.

        Args:
            meta_object (Dict): The meta object to validate.
        """
        if not isinstance(meta_object, dict):
            raise ValueError("meta_object must be a dictionary")

        if "originalData" not in meta_object:
            raise ValueError("meta_object must contain 'originalData' key")

        if not isinstance(meta_object["originalData"], dict):
            raise ValueError("meta_object['originalData'] must be a dictionary")

        if len(meta_object["originalData"]) == 0:
            raise ValueError("meta_object must contain at least one dataset")

    def _validate_filter_object(self, filter_object: Dict) -> None:
        """Validate filter object structure.

        Raises ValueError if filter_object is not a dictionary or if it
        lacks one of the required keys 'posGeneNames' or 'negGeneNames'.

        Args:
            filter_object (Dict): The filter object to validate.
        """
        if not isinstance(filter_object, dict):
            raise ValueError("filter_object must be a dictionary")

        required_keys = ["posGeneNames", "negGeneNames"]
        for key in required_keys:
            if key not in filter_object:
                raise ValueError(f"filter_object must contain '{key}' key")

    def _convert_discovery_list_to_genes(
        self, meta_object: Dict, pos_genes: List[str], neg_genes: List[str]
    ) -> List[Dict]:
        """Convert discovery datasets from probe-level to gene-level data.

        This method is a Python implementation of the R function
        .convertDiscoveryListToGenes. It first imports the helper
        getSampleLevelGeneData; if the import fails, an ImportError
        is raised. The helper converts each discovery dataset to
        gene-level data using the combined list of positive and
        negative genes. If any gene values are negative, an offset
        is added to ensure all values are positive. The processed
        datasets are stored as dictionaries containing a
        DataFrame named 'genes' and a NumPy array named 'class'.
        Any dataset that cannot be processed is skipped and a
        warning is logged. Finally, the method returns a list
        of the successfully processed dataset dictionaries.

        Args:
            meta_object (Dict): A dictionary containing discovery
            datasets under the key 'originalData'.
            pos_genes (List[str]): A list of positive gene names.
            neg_genes (List[str]): A list of negative gene names.

        Returns:
            List[Dict]: A list where each element is a dictionary
            with keys 'genes' (DataFrame) and 'class' (NumPy array).
        """
        # Use the already imported getSampleLevelGeneData function
        if getSampleLevelGeneData is None:
            raise ImportError("getSampleLevelGeneData function not available")

        all_genes = list(set(pos_genes + neg_genes))
        discovery_genes = []

        logger.info("Converting probe-level data to gene-level data for all discovery datasets")

        for name, dataset in meta_object["originalData"].items():
            try:
                # Get gene-level data
                gene_data = getSampleLevelGeneData(dataset, all_genes)

                # Ensure positive values (equivalent to R logic: add offset if min < 0)
                gene_min = gene_data.values.min()
                if gene_min < 0:
                    gene_data = gene_data + abs(gene_min) + 1

                # Store as dictionary with required structure
                dataset_dict = {
                    "genes": gene_data,  # DataFrame with genes as rows, samples as columns
                    "class": np.array(dataset["class"]),  # Class labels
                }

                # Verify data consistency
                expected_samples = gene_data.shape[1]  # Columns should be samples
                actual_samples = len(dataset["class"])

                if expected_samples != actual_samples:
                    logger.warning(
                        f"Dataset {dataset.get('formattedName', name)}: "
                        f"Gene matrix has {expected_samples} samples but class has {actual_samples} labels. "
                        f"This may cause dimension mismatch issues."
                    )

                discovery_genes.append(dataset_dict)
                logger.info(f"Processed dataset: {dataset.get('formattedName', name)}")

            except Exception as e:
                logger.warning(f"Failed to process dataset {name}: {str(e)}")
                continue

        logger.info(f"Successfully converted {len(discovery_genes)} datasets")
        return discovery_genes

    def _backward_search_weighted(
        self,
        discovery_genes: List[Dict],
        pos_genes: List[str],
        neg_genes: List[str],
        back_thresh: float,
    ) -> Tuple[List[str], List[str]]:
        """Perform weighted backward search algorithm.

        This method implements the core logic from R's .backSearchWeighted function. It calculates
        weights based on sample sizes of each discovery dataset and iteratively removes genes
        that lead to the smallest decrease (or largest increase) in weighted AUC. The algorithm
        continues until the improvement threshold back_thresh is not met or no genes remain to evaluate.
        The final positive and negative gene lists are returned.

        Args:
            discovery_genes (List[Dict]): List of discovery dataset dictionaries containing gene data and class labels.
            pos_genes (List[str]): List of genes initially selected as positive.
            neg_genes (List[str]): List of genes initially selected as negative.
            back_thresh (float): Threshold for stopping the backward search when the worst AUC decrease is greater than this value.

        Returns:
            Tuple[List[str], List[str]]: Updated lists of positive and negative genes after backward search.
        """
        # Calculate weights based on sample sizes
        weights = np.array([len(dataset["class"]) for dataset in discovery_genes])

        # Validate all genes are present
        self._match_all_genes(discovery_genes, pos_genes, neg_genes, None, None)

        # Get initial weighted AUC
        current_auc = self._get_weighted_aucs_genes_list(
            discovery_genes, pos_genes, neg_genes, weights
        )

        # Make copies to modify
        current_pos = pos_genes.copy()
        current_neg = neg_genes.copy()

        logger.info(f"Initial weighted AUC: {current_auc:.4f}")

        while True:
            # Find gene that when removed causes smallest decrease (or largest increase) in AUC
            gene_search = self._backward_search_inner(
                discovery_genes, current_pos, current_neg, current_auc, weights
            )

            if len(gene_search) == 0:
                break

            # Find worst gene (minimum difference means removing it helps least or hurts most)
            worst_improvement = gene_search["diff"].min()

            logger.info(f"Worst AUC decrease: {worst_improvement:.4f}")

            if worst_improvement <= back_thresh:
                # Remove the worst gene
                worst_idx = gene_search["diff"].idxmin()
                worst_gene = gene_search.loc[worst_idx, "gene"]
                new_auc = gene_search.loc[worst_idx, "search"]

                logger.info(f"Removing gene: {worst_gene}, new weighted AUC: {new_auc:.4f}")

                if worst_gene in current_pos:
                    current_pos.remove(worst_gene)
                elif worst_gene in current_neg:
                    current_neg.remove(worst_gene)

                current_auc = new_auc
            else:
                break

        # Final verification
        final_auc = self._get_weighted_aucs_genes_list(
            discovery_genes, current_pos, current_neg, weights
        )
        logger.info(f"Final weighted AUC: {final_auc:.4f}")

        return current_pos, current_neg

    def _forward_search_weighted(
        self,
        discovery_genes: List[Dict],
        pos_genes: List[str],
        neg_genes: List[str],
        yes_pos: Optional[List[str]],
        yes_neg: Optional[List[str]],
        forward_thresh: float,
    ) -> Tuple[List[str], List[str]]:
        """Perform weighted forward search algorithm.

        This method implements the core logic from R's .forwardSearchWeighted
        function. It first calculates weights based on the sample sizes of
        each discovery dataset and initializes the starting gene lists. It
        then removes the starting genes from the candidate pools and
        validates that all required genes are present. The algorithm
        iteratively searches for the best gene to add that maximizes the
        weighted AUC improvement. If the best improvement is at least
        forward_thresh, the gene is added to the selected set and the
        process continues. The search stops when no candidate gene meets
        the threshold. The final weighted AUC and the updated positive
        and negative gene lists are logged and returned.

        Args:
            discovery_genes (List[Dict]): List of discovery dataset dictionaries containing gene data and class labels.
            pos_genes (List[str]): List of genes initially considered positive.
            neg_genes (List[str]): List of genes initially considered negative.
            yes_pos (Optional[List[str]]): Optional list of genes that must be included in the positive set.
            yes_neg (Optional[List[str]]): Optional list of genes that must be included in the negative set.
            forward_thresh (float): Minimum improvement threshold required to add a new gene.

        Returns:
            Tuple[List[str], List[str]]: Updated lists of selected positive and negative genes after forward search.
        """
        # Calculate weights based on sample sizes
        weights = np.array([len(dataset["class"]) for dataset in discovery_genes])

        # Initialize starting genes
        selected_pos = yes_pos.copy() if yes_pos else []
        selected_neg = yes_neg.copy() if yes_neg else []

        # Remove starting genes from candidates
        candidate_pos = [g for g in pos_genes if g not in selected_pos]
        candidate_neg = [g for g in neg_genes if g not in selected_neg]

        # Validate genes
        self._match_all_genes(discovery_genes, pos_genes, neg_genes, selected_pos, selected_neg)

        # Get initial AUC
        if not selected_pos and not selected_neg:
            current_auc = 0.0
        else:
            current_auc = self._get_weighted_aucs_genes_list(
                discovery_genes,
                selected_pos,
                selected_neg,
                weights,
                print_results=False,
            )

        logger.info(f"Starting weighted AUC: {current_auc:.4f}")

        while True:
            # Find best gene to add
            gene_search = self._forward_search_inner(
                discovery_genes,
                candidate_pos,
                candidate_neg,
                selected_pos,
                selected_neg,
                current_auc,
                weights,
            )

            if len(gene_search) == 0:
                break

            # Find best gene (maximum improvement)
            best_improvement = gene_search["diff"].max()

            logger.info(f"Best AUC improvement: {best_improvement:.4f}")

            if best_improvement >= forward_thresh:
                # Add the best gene
                best_idx = gene_search["diff"].idxmax()
                best_gene = gene_search.loc[best_idx, "gene"]
                new_auc = gene_search.loc[best_idx, "search"]

                logger.info(f"Adding gene: {best_gene}, new weighted AUC: {new_auc:.4f}")

                if best_gene in candidate_pos:
                    candidate_pos.remove(best_gene)
                    selected_pos.append(best_gene)
                elif best_gene in candidate_neg:
                    candidate_neg.remove(best_gene)
                    selected_neg.append(best_gene)

                current_auc = new_auc
            else:
                break

        # Final verification
        final_auc = self._get_weighted_aucs_genes_list(
            discovery_genes, selected_pos, selected_neg, weights, print_results=True
        )
        logger.info(f"Final weighted AUC: {final_auc:.4f}")

        return selected_pos, selected_neg

    def _backward_search_inner(
        self,
        discovery_genes: List[Dict],
        pos_genes: List[str],
        neg_genes: List[str],
        current_auc: float,
        weights: np.ndarray,
    ) -> pd.DataFrame:
        """Inner loop for backward search - evaluate removing each gene.

        For every gene currently in the positive or negative set a copy of the
        gene list with that gene removed is used to compute a new weighted AUC.
        The difference between the original AUC and the new AUC is stored; a
        positive difference indicates that removing the gene hurts performance.
        The results for all tested genes are returned as a DataFrame with
        columns 'gene', 'original', 'search', and 'diff'.

        This matches R's .backsearchInner which calls .backsearchPos and
        .backsearchNeg separately, each using a ratio to balance positive and
        negative gene contributions when computing the score.

        Args:
            discovery_genes (List[Dict]): List of discovery dataset dictionaries.
            pos_genes (List[str]): Current list of positive genes.
            neg_genes (List[str]): Current list of negative genes.
            current_auc (float): Weighted AUC of the current gene set.
            weights (np.ndarray): Array of weights derived from sample sizes.

        Returns:
            pd.DataFrame: DataFrame containing test results for each gene.
        """
        results = []

        # Test removing each positive gene
        # Matches R's .backsearchPos:
        #   ratio <- length(neg.genes)/(length(pos.genes)-1)
        #   totalScore <- scale(posScore - ratio*negScore)
        #   Edge case: if length(pos.genes) <= 1, uses scale(-negScore)
        if pos_genes:
            if len(pos_genes) <= 1:
                # R edge case: only one pos gene left, removing it means
                # score = scale(-negScore), i.e. no pos contribution at all.
                for gene in pos_genes:
                    test_auc = self._get_weighted_aucs_genes_list(
                        discovery_genes,
                        [],
                        neg_genes,
                        weights,
                        print_results=False,
                        ratio=1.0,
                        pos_only_neg_sign=True,
                    )
                    results.append(
                        {
                            "gene": gene,
                            "original": current_auc,
                            "search": test_auc,
                            "diff": current_auc - test_auc,
                        }
                    )
            else:
                ratio = len(neg_genes) / (len(pos_genes) - 1)
                for gene in pos_genes:
                    test_pos = [g for g in pos_genes if g != gene]
                    test_auc = self._get_weighted_aucs_genes_list(
                        discovery_genes,
                        test_pos,
                        neg_genes,
                        weights,
                        print_results=False,
                        ratio=ratio,
                    )
                    results.append(
                        {
                            "gene": gene,
                            "original": current_auc,
                            "search": test_auc,
                            "diff": current_auc - test_auc,
                        }
                    )

        # Test removing each negative gene
        # Matches R's .backsearchNeg:
        #   ratio <- (length(neg.genes)-1)/length(pos.genes)
        #   totalScore <- scale(posScore - ratio*negScore)
        #   Edge case: if length(neg.genes) <= 1, uses scale(posScore)
        if neg_genes:
            if len(neg_genes) <= 1:
                # R edge case: only one neg gene left, removing it means
                # score = scale(posScore), i.e. no neg contribution.
                for gene in neg_genes:
                    test_auc = self._get_weighted_aucs_genes_list(
                        discovery_genes, pos_genes, [], weights, print_results=False, ratio=0.0
                    )
                    results.append(
                        {
                            "gene": gene,
                            "original": current_auc,
                            "search": test_auc,
                            "diff": current_auc - test_auc,
                        }
                    )
            else:
                ratio = (len(neg_genes) - 1) / len(pos_genes) if len(pos_genes) > 0 else 1.0
                for gene in neg_genes:
                    test_neg = [g for g in neg_genes if g != gene]
                    test_auc = self._get_weighted_aucs_genes_list(
                        discovery_genes,
                        pos_genes,
                        test_neg,
                        weights,
                        print_results=False,
                        ratio=ratio,
                    )
                    results.append(
                        {
                            "gene": gene,
                            "original": current_auc,
                            "search": test_auc,
                            "diff": current_auc - test_auc,
                        }
                    )

        return pd.DataFrame(results)

    def _forward_search_inner(
        self,
        discovery_genes: List[Dict],
        candidate_pos: List[str],
        candidate_neg: List[str],
        selected_pos: List[str],
        selected_neg: List[str],
        current_auc: float,
        weights: np.ndarray,
    ) -> pd.DataFrame:
        """Inner loop for forward search - evaluate adding each candidate gene.

        For every candidate gene in the positive and negative pools the
        function creates a temporary set that includes the candidate gene
        and calculates the new weighted AUC.  The difference between the
        new AUC and the current AUC is stored; a positive difference
        indicates that adding the gene improves performance.  All
        results are returned as a DataFrame with columns 'gene',
        'original', 'search', and 'diff'.

        This matches R's .forwardSearchInner which calls .forwardSearchPos
        and .forwardSearchNeg, each using a ratio to balance positive and
        negative gene contributions:
          .forwardSearchPos: ratio = length(yes.neg)/(length(yes.pos)+1)
          .forwardSearchNeg: ratio = (length(yes.neg)+1)/length(yes.pos)
                             (or ratio = 1 if yes.pos is NULL)

        Args:
            discovery_genes (List[Dict]): List of discovery dataset dictionaries.
            candidate_pos (List[str]): Genes available to add to the positive set.
            candidate_neg (List[str]): Genes available to add to the negative set.
            selected_pos (List[str]): Currently selected positive genes.
            selected_neg (List[str]): Currently selected negative genes.
            current_auc (float): Weighted AUC of the current gene set.
            weights (np.ndarray): Array of weights derived from sample sizes.

        Returns:
            pd.DataFrame: DataFrame containing test results for each candidate gene.
        """
        results = []

        # Test adding each candidate positive gene
        # Matches R's .forwardSearchPos:
        #   ratio <- length(yes.neg)/(length(yes.pos)+1)
        #   totalScore <- scale(posScore - ratio*negScore)
        if candidate_pos:
            ratio_pos = (
                len(selected_neg) / (len(selected_pos) + 1) if (len(selected_pos) + 1) > 0 else 0.0
            )
            for gene in candidate_pos:
                test_pos = selected_pos + [gene]
                test_auc = self._get_weighted_aucs_genes_list(
                    discovery_genes,
                    test_pos,
                    selected_neg,
                    weights,
                    print_results=False,
                    ratio=ratio_pos,
                )
                results.append(
                    {
                        "gene": gene,
                        "original": current_auc,
                        "search": test_auc,
                        "diff": test_auc - current_auc,
                    }
                )

        # Test adding each candidate negative gene
        # Matches R's .forwardSearchNeg:
        #   if(is.null(yes.pos)) ratio <- 1
        #   else ratio <- (length(yes.neg)+1)/length(yes.pos)
        #   totalScore <- scale(posScore - ratio*negScore)
        if candidate_neg:
            if not selected_pos:
                ratio_neg = 1.0
            else:
                ratio_neg = (len(selected_neg) + 1) / len(selected_pos)
            for gene in candidate_neg:
                test_neg = selected_neg + [gene]
                test_auc = self._get_weighted_aucs_genes_list(
                    discovery_genes,
                    selected_pos,
                    test_neg,
                    weights,
                    print_results=False,
                    ratio=ratio_neg,
                )
                results.append(
                    {
                        "gene": gene,
                        "original": current_auc,
                        "search": test_auc,
                        "diff": test_auc - current_auc,
                    }
                )

        return pd.DataFrame(results)

    def _get_weighted_aucs_genes_list(
        self,
        genes_list: List[Dict],
        pos_genes: List[str],
        neg_genes: List[str],
        weights: np.ndarray,
        print_results: bool = False,
        ratio: Optional[float] = None,
        pos_only_neg_sign: bool = False,
    ) -> float:
        """Calculate weighted AUC across all datasets for given gene signature.

        Uses the optimized _score_genes_matrix for scoring and
        sklearn's roc_auc_score for fast AUC computation.

        Args:
            genes_list: List of discovery dataset dictionaries.
            pos_genes: List of positive gene names.
            neg_genes: List of negative gene names.
            weights: Array of weights (sample sizes per dataset).
            print_results: If True, log individual dataset AUCs.
            ratio: Optional ratio for inner search scoring. When None,
                uses standard scoring (posScore - negScore) matching R's
                .scoreGenesMtx / .getWeightedAUCsGenesList.  When provided,
                uses posScore - ratio*negScore matching R's inner search
                functions.
            pos_only_neg_sign: If True, computes score as scale(-negScore)
                only (used when all positive genes removed in backward
                search, matching R's edge case in .backsearchPos).
        """
        if not pos_genes and not neg_genes:
            return 0.5 * np.sum(weights)

        aucs = []
        for i, dataset in enumerate(genes_list):
            if pos_only_neg_sign:
                # R edge case: score = scale(-negScore) when no pos genes remain
                score = self._score_genes_matrix(dataset["genes"], [], neg_genes, ratio=None)
                # _score_genes_matrix with empty pos returns -negScore (standardized)
                # which is correct since 0 - negScore = -negScore
            else:
                # use the corrected scoring method (geometric mean + ddof=1)
                score = self._score_genes_matrix(
                    dataset["genes"], pos_genes, neg_genes, ratio=ratio
                )
            y_true = dataset["class"]

            # Validate that dimensions match
            if len(score) != len(y_true):
                min_length = min(len(score), len(y_true))
                if min_length == 0:
                    auc_val = 0.5
                else:
                    score = score[:min_length]
                    y_true = y_true[:min_length]

                    if len(np.unique(y_true)) < 2:
                        auc_val = 0.5
                    else:
                        # use roc_auc_score for speed
                        auc_val = roc_auc_score(y_true, score)

                warnings.warn(
                    f"Dataset {i + 1}: Length mismatch between scores ({len(dataset['genes'].columns)}) "
                    f"and class labels ({len(dataset['class'])}). Using {min_length} samples."
                )
            else:
                if len(np.unique(y_true)) < 2:
                    auc_val = 0.5
                else:
                    # use roc_auc_score for speed
                    auc_val = roc_auc_score(y_true, score)

            aucs.append(auc_val)

            if print_results:
                dataset_name = f"Dataset_{i + 1}"
                logger.info(f"{dataset_name}: AUC = {auc_val:.3f}")

        weighted_auc = np.sum(np.array(aucs) * weights)

        if print_results:
            mean_auc = np.mean(aucs)
            logger.info(f"Mean AUC: {mean_auc:.3f}")
            logger.info(f"Weighted AUC sum: {weighted_auc:.3f}")

        return weighted_auc

    def _precompute_gene_indices(
        self, gene_matrix: pd.DataFrame, all_genes: List[str]
    ) -> Dict[str, int]:
        """Precompute a gene -> row-index mapping to cache gene index lookups."""
        index_map = {}
        for gene in all_genes:
            if gene in gene_matrix.index:
                index_map[gene] = gene_matrix.index.get_loc(gene)
        return index_map

    def _score_genes_matrix_fast(
        self,
        gene_matrix_values: np.ndarray,
        pos_indices: List[int],
        neg_indices: List[int],
        n_samples: int,
    ) -> np.ndarray:
        """Vectorized scoring using precomputed gene row indices.

        Note:
            Not currently called anywhere: prepared as a faster
            alternative to :meth:`_score_genes_matrix` for use inside
            :meth:`_backward_search_weighted` / :meth:`_forward_search_weighted`,
            where a dataset's gene expression DataFrame would be converted
            to a raw NumPy array once and passed here together with
            precomputed indices from :meth:`_precompute_gene_indices`,
            avoiding repeated pandas label lookups inside the search loop.

        Args:
            gene_matrix_values: Gene expression matrix as a raw NumPy
                array (genes as rows, samples as columns).
            pos_indices: Row indices of the positive-signature genes.
            neg_indices: Row indices of the negative-signature genes.
            n_samples: Number of samples (columns).

        Returns:
            Standardized score vector, one value per sample.
        """
        epsilon = 1e-10

        pos_score = np.zeros(n_samples)
        if pos_indices:
            pos_data = np.maximum(gene_matrix_values[pos_indices], epsilon)
            pos_score = np.exp(np.mean(np.log(pos_data), axis=0))

        neg_score = np.zeros(n_samples)
        if neg_indices:
            neg_data = np.maximum(gene_matrix_values[neg_indices], epsilon)
            neg_score = np.exp(np.mean(np.log(neg_data), axis=0))

        total = pos_score - neg_score
        std = np.std(total, ddof=1)
        if std > 0:
            total = (total - np.mean(total)) / std
        return total

    def _score_genes_matrix(
        self,
        gene_matrix: pd.DataFrame,
        pos_genes: List[str],
        neg_genes: List[str],
        ratio: Optional[float] = None,
    ) -> np.ndarray:
        """Calculate scores from gene expression matrix using positive and negative signatures.

        This implements the logic from R's .scoreGenesMtx function. The score for each
        sample is the geometric mean of the expression values of the positive genes
        minus the geometric mean of the negative genes.  If either signature is empty
        or contains no genes present in the matrix a zero score is used for that
        component.  The resulting vector is then standardized to mean zero and unit
        variance.

        When ratio is provided (used in inner search loops), the negative score is
        scaled: score = posScore - ratio * negScore.  This matches R's backsearch
        and forward search inner functions which balance the contribution of positive
        and negative genes based on their relative counts.

        Args:
            gene_matrix (pd.DataFrame): Gene expression matrix with genes as rows and samples as columns.
            pos_genes (List[str]): List of genes in the positive signature.
            neg_genes (List[str]): List of genes in the negative signature.
            ratio (Optional[float]): Scaling factor for negative gene score.
                When None (default), score = posScore - negScore (used for overall
                AUC calculation, matching R's .scoreGenesMtx).
                When provided, score = posScore - ratio * negScore (used in inner
                search loops, matching R's .backsearchPos/.backsearchNeg/
                .forwardSearchPos/.forwardSearchNeg).

        Returns:
            np.ndarray: Standardized score vector for each sample.
        """
        # Validate input
        if gene_matrix.empty:
            return np.array([])

        # Get positive gene score (geometric mean)
        pos_score = 0.0
        if pos_genes:
            pos_present = [g for g in pos_genes if g in gene_matrix.index]
            if pos_present:
                pos_data = gene_matrix.loc[pos_present]
                pos_score = self._geometric_mean(pos_data.values, axis=0)

        # Get negative gene score (geometric mean)
        neg_score = 0.0
        if neg_genes:
            neg_present = [g for g in neg_genes if g in gene_matrix.index]
            if neg_present:
                neg_data = gene_matrix.loc[neg_present]
                neg_score = self._geometric_mean(neg_data.values, axis=0)

        # Apply ratio scaling if provided (inner search loop balancing)
        if ratio is not None:
            neg_score_scaled = (
                np.asarray(neg_score) * ratio
                if not isinstance(neg_score, (int, float)) or neg_score != 0.0
                else 0.0
            )
        else:
            neg_score_scaled = neg_score

        # Calculate final score (positive - [ratio *] negative)
        if isinstance(pos_score, (int, float)) and isinstance(neg_score_scaled, (int, float)):
            # Handle scalar case
            if pos_score == 0 and neg_score_scaled == 0:
                total_score = np.zeros(gene_matrix.shape[1])
            else:
                total_score = np.full(gene_matrix.shape[1], pos_score - neg_score_scaled)
        else:
            # Handle array case
            total_score = np.asarray(pos_score) - np.asarray(neg_score_scaled)

        # Standardize scores (equivalent to R's scale function)
        if len(total_score) > 0 and np.std(total_score) > 0:

            # R's scale() function uses ddof=1 (sample standard deviation).
            total_score = (total_score - np.mean(total_score)) / np.std(total_score, ddof=1)

        return total_score

    def _geometric_mean(self, data: np.ndarray, axis: int = 0) -> Union[float, np.ndarray]:
        """
        Calculate geometric mean, handling zeros and negative values.

        This implements R's geomMean logic with proper handling of edge cases.

        Args:
            data (np.ndarray): Input array of values.
            axis (int): Axis along which to compute the mean. Default is 0.

        Returns:
            Union[float, np.ndarray]: Geometric mean of the input data.
        """
        # Handle zeros and negatives by adding small epsilon (equivalent to R logic)
        epsilon = 1e-10
        data_positive = np.maximum(data, epsilon)

        # Calculate geometric mean using log transformation
        log_data = np.log(data_positive)
        log_mean = np.mean(log_data, axis=axis)
        geom_mean = np.exp(log_mean)

        return geom_mean

    def _match_all_genes(
        self,
        discovery_genes: List[Dict],
        pos_genes: List[str],
        neg_genes: List[str],
        yes_pos: Optional[List[str]],
        yes_neg: Optional[List[str]],
    ) -> None:
        """
        Validate that all specified genes are present in discovery datasets.

        This implements the logic from R's .matchAll function.

        Args:
            discovery_genes (List[Dict]): List of discovery dataset dictionaries each containing a 'genes' DataFrame or similar structure with an index of gene identifiers.
            pos_genes (List[str]): List of positive gene identifiers to be checked.
            neg_genes (List[str]): List of negative gene identifiers to be checked.
            yes_pos (Optional[List[str]]): Optional list of genes that must be in pos_genes. May be None.
            yes_neg (Optional[List[str]]): Optional list of genes that must be in neg_genes. May be None.

        Returns:
            None
        """
        # Collect all genes present in discovery datasets
        all_discovery_genes = set()
        for dataset in discovery_genes:
            all_discovery_genes.update(dataset["genes"].index)

        # Check if all specified genes are present
        all_specified = set(pos_genes + neg_genes)
        missing_genes = all_specified - all_discovery_genes

        if missing_genes:
            warnings.warn(f"Some specified genes not found in discovery datasets: {missing_genes}")

        # Validate yes_pos and yes_neg are subsets
        if yes_pos:
            invalid_pos = set(yes_pos) - set(pos_genes)
            if invalid_pos:
                raise ValueError(f"yes_pos contains genes not in pos_genes: {invalid_pos}")

        if yes_neg:
            invalid_neg = set(yes_neg) - set(neg_genes)
            if invalid_neg:
                raise ValueError(f"yes_neg contains genes not in neg_genes: {invalid_neg}")


# ------------------------------------------------------------------
# Convenience functions for backward compatibility
# ------------------------------------------------------------------
def backward_search(meta_object: Dict, filter_object: Dict, back_thresh: float = 0.0) -> Dict:
    """
    Perform backward search for gene selection optimization.

    This function creates an instance of GeneSearchOptimizer and delegates the
    backward search operation to it, passing along the meta data, filter settings
    and a threshold parameter.

    Args:
        meta_object (Dict): Meta information used by the optimizer.
        filter_object (Dict): Filtering criteria for gene selection.
        back_thresh (float): Threshold value for the backward search; defaults to 0.0.

    Returns:
        Dict: Resulting dictionary from the optimizer's backward search.
    """
    optimizer = GeneSearchOptimizer()
    return optimizer.backward_search(meta_object, filter_object, back_thresh)


def forward_search(
    meta_object: Dict,
    filter_object: Dict,
    yes_pos: Optional[List[str]] = None,
    yes_neg: Optional[List[str]] = None,
    forward_thresh: float = 0.0,
) -> Dict:
    """
    Perform forward search for gene selection optimization.

    This function creates an instance of GeneSearchOptimizer and delegates the
    forward search operation to it, passing along the meta data, filter
    settings, any fixed positive or negative gene lists and a threshold
    parameter.

    Args:
        meta_object (Dict): Meta information used by the optimizer.
        filter_object (Dict): Filtering criteria for gene selection.
        yes_pos (Optional[List[str]]): Optional list of genes that must be
            considered positive. Defaults to None.
        yes_neg (Optional[List[str]]): Optional list of genes that must be
            considered negative. Defaults to None.
        forward_thresh (float): Threshold value for the forward search;
            defaults to 0.0.

    Returns:
        Dict: Resulting dictionary from the optimizer's forward search.
    """
    optimizer = GeneSearchOptimizer()
    return optimizer.forward_search(meta_object, filter_object, yes_pos, yes_neg, forward_thresh)
