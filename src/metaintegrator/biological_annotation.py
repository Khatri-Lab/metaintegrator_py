"""
Biological annotation functions for MetaIntegrator Python port.
This module handles biological metadata imputation and annotation.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ------------------------------------------------------------------
# BiologicalAnnotator Class
# ------------------------------------------------------------------
class BiologicalAnnotator:
    """Class for biological annotation and imputation tasks."""

    # Default sex-associated genes from iSEXS (immune Sex Expression Signature)
    DEFAULT_FEMALE_GENES = [
        "XIST",
        "RPS4X",
        "CD40LG",
        "ZRSR2",
        "EFHC2",
        "CA5B",
        "ZFX",
        "EIF1AX",
        "CA5BP1",
        "UBA1",
        "SYAP1",
        "DDX3X",
        "FUNDC1",
        "USP9X",
        "SMC1A",
        "NUP62CL",
        "NAA10",
    ]
    DEFAULT_MALE_GENES = [
        "KDM5D",
        "RPS4Y1",
        "EIF1AY",
        "USP9Y",
        "DDX3Y",
        "UTY",
        "PRKY",
        "ZFY",
        "TMSB4Y",
    ]

    def __init__(self):
        """Initialize the biological annotator."""
        self.default_female_genes = self.DEFAULT_FEMALE_GENES.copy()
        self.default_male_genes = self.DEFAULT_MALE_GENES.copy()

    def infer_sex(
        self,
        dataset: Dict,
        fem_genes: Optional[List[str]] = None,
        mal_genes: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> Optional[np.ndarray]:
        """
        Infers the sex of individuals in a dataset based on gene expression data. Uses known
        sex-associated genes to classify samples into male or female groups. If provided,
        specific gene sets for male and female can be used; otherwise, default genes from
        the instance are utilized. The function employs K-means clustering to group samples
        and then assigns a sex to each cluster based on the gene expression.

        Raises warnings if no sex-associated genes are found in the dataset or if the clustering
        results are ambiguous.

        Args:
            dataset (Dict): A dictionary containing gene expression data and dataset keys.
            fem_genes (Optional[List[str]]): A list of female-associated genes. Default is None;
            falls back to instance's default female genes.
            mal_genes (Optional[List[str]]): A list of male-associated genes. Default is None;
            falls back to instance's default male genes.
            random_state (int): Random state used for the K-means algorithm to ensure reproducibility.

        Returns:
            Optional[np.ndarray]: An array of inferred sexes ('female' or 'male') for each sample in
            the dataset. Returns None if sex determination is not possible.
        """

        # Validate input dataset
        if not isinstance(dataset, dict):
            warnings.warn("Dataset must be a dictionary")
            return None

        if "keys" not in dataset:
            warnings.warn("Dataset must contain 'keys' field")
            return None

        if "expr" not in dataset:
            warnings.warn("Dataset must contain 'expr' field")
            return None

        # Use default genes if not provided
        fem_genes = fem_genes or self.default_female_genes
        mal_genes = mal_genes or self.default_male_genes

        # Get available genes in dataset
        available_keys = set(dataset.get("keys", []))

        # Filter to genes present in dataset
        fem_avail = [g for g in fem_genes if g in available_keys]
        mal_avail = [g for g in mal_genes if g in available_keys]

        # Check if we have any sex-associated genes
        all_sex_genes = fem_avail + mal_avail

        if not all_sex_genes:
            warnings.warn("No sex classification genes found in dataset keys.")
            return None

        # Extract expression data for sex genes
        sex_expr = self._get_sample_level_gene_data(dataset, all_sex_genes)

        # CRITICAL FIX: Check None first to avoid AttributeError
        if sex_expr is None:
            warnings.warn("Could not extract expression data for sex genes.")
            return None

        if sex_expr.empty:
            warnings.warn("Could not extract expression data for sex genes.")
            return None

        # Check if we have enough samples for clustering
        n_samples = sex_expr.shape[1]  # Columns are samples
        if n_samples < 2:
            warnings.warn("Insufficient samples for clustering (need at least 2 samples)")
            return None

        # Run K-means clustering (k=2)
        # Cluster samples (columns)
        try:
            kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(sex_expr.T)  # Transpose for sample clustering
        except Exception as e:
            warnings.warn(f"K-means clustering failed: {e}")
            return None

        # Determine which cluster represents females
        female_idx = self._map_clusters_to_sex(
            kmeans.cluster_centers_, all_sex_genes, fem_avail, mal_avail
        )

        if female_idx is None:
            warnings.warn("Ambiguous cluster assignment. Try different gene sets.")
            return None

        # Assign sex based on cluster membership
        return np.where(cluster_labels == female_idx, "female", "male")

    def _get_sample_level_gene_data(
        self, dataset: Dict, genes: List[str]
    ) -> Optional[pd.DataFrame]:
        """
        Extracts gene expression data at the sample level from the given dataset and filters
        it based on a provided list of genes. The method attempts to identify and return
        only the rows of the data corresponding to the given genes.

        Args:
            dataset: Dict
                A dictionary containing at least two required entries:
                - "expr": The gene expression data, which can be in the form of a pandas DataFrame
                  or a numpy-like array.
                - "keys": A list of gene names corresponding to each row in the expression data.
            genes: List[str]
                A list of gene names for which expression data should be filtered.

        Returns:
            Optional[pd.DataFrame]
                A DataFrame containing the expression data for the provided genes if the genes are
                present in the dataset. Returns None if no matching genes are found or if an error
                occurs during the operation.
        """
        try:
            expr = dataset["expr"]
            keys = list(dataset["keys"])
            gene_set = set(genes)

            # Check for length mismatch between expr and keys
            if hasattr(expr, "shape") and len(expr.shape) >= 1 and len(keys) != expr.shape[0]:
                warnings.warn(
                    f"Mismatch between expression matrix rows ({expr.shape[0]}) and keys ({len(keys)})"
                )
                return None

            indices = [i for i, k in enumerate(keys) if k in gene_set]
            if not indices:
                return None

            if isinstance(expr, pd.DataFrame):
                return expr.iloc[indices]

            # Fallback for numpy-like arrays
            selected_expr = expr[indices, :]
            selected_keys = [keys[i] for i in indices]
            return pd.DataFrame(selected_expr, index=selected_keys)
        except Exception as e:
            warnings.warn(f"Error extracting gene expression data: {e}")
            return None

    def _map_clusters_to_sex(
        self,
        centroids: np.ndarray,
        all_genes: List[str],
        fem_genes: List[str],
        mal_genes: List[str],
    ) -> Optional[int]:
        """
        Maps clusters to their corresponding gender sex based on gene expression data.

        This method leverages provided centroids and gene lists to identify which cluster
        is most representative of female-like or male-like gene expression. It calculates
        means of expression for genes associated with each sex and determines the cluster
        alignment accordingly.

        Args:
            centroids: A 2D numpy array where each row corresponds to a cluster, and
                columns represent gene expression levels.
            all_genes: A list of all genes corresponding to the columns in the centroids
                matrix.
            fem_genes: A list of genes known to be associated with female characteristics.
            mal_genes: A list of genes known to be associated with male characteristics.

        Returns:
            An integer representing the index of the cluster most aligned to female-like
            gene expression or None if no clear mapping can be determined.
        """
        # Create mapping from genes to centroid positions
        gene_map = {gene: i for i, gene in enumerate(all_genes)}

        # Get female gene positions and calculate centroid means
        fem_pos = [gene_map[g] for g in fem_genes if g in gene_map]
        mal_pos = [gene_map[g] for g in mal_genes if g in gene_map]

        fem_means = centroids[:, fem_pos].mean(axis=1) if fem_pos else None
        mal_means = centroids[:, mal_pos].mean(axis=1) if mal_pos else None

        # Case 1: Both male and female genes available
        if fem_means is not None and mal_means is not None:
            # Female centroid should have high female gene expression
            # and low male gene expression
            high_fem, low_mal = np.argmax(fem_means), np.argmin(mal_means)
            return high_fem if high_fem == low_mal else None

        # Case 2: Only female genes available
        if fem_means is not None:
            # Female centroid should have high female gene expression
            return np.argmax(fem_means)
        # Case 3: Only male genes available
        if mal_means is not None:
            # Female centroid should have low male gene expression
            return np.argmin(mal_means)

        return None

    def validate_sex_predictions(
        self, predictions: np.ndarray, known_sex: Optional[List[str]] = None
    ) -> Dict:
        """
        Validates a set of sex predictions against optional known labels and computes various
        statistics about the predictions such as counts, ratios, and optionally accuracy.

        Args:
            predictions: np.ndarray
                Array of predicted sex labels. Expected values are strings 'female' or 'male'.
            known_sex: Optional[List[str]]
                List of true sex labels to evaluate the accuracy of predictions against. Should
                have the same length as predictions if provided.

        Returns:
            Dict
                A dictionary containing:
                    - n_samples: Total number of predictions.
                    - n_female: Number of predictions labeled as 'female'.
                    - n_male: Number of predictions labeled as 'male'.
                    - female_ratio: Proportion of predictions labeled as 'female'.
                    - accuracy: (Optional) Accuracy of predictions if known_sex is provided and has
                      matching length with predictions.
                    - confusion_matrix: (Optional) Confusion matrix as a pandas DataFrame if
                      known_sex is provided and has matching length with predictions.
        """
        results = {
            "n_samples": len(predictions),
            "n_female": np.sum(predictions == "female"),
            "n_male": np.sum(predictions == "male"),
            "female_ratio": np.mean(predictions == "female"),
        }

        if known_sex is not None:
            known_sex = np.array(known_sex)
            if len(known_sex) == len(predictions):
                # Calculate accuracy
                accuracy = np.mean(predictions == known_sex)
                results["accuracy"] = accuracy
                results["confusion_matrix"] = pd.crosstab(
                    pd.Series(known_sex, name="Known"),
                    pd.Series(predictions, name="Predicted"),
                    margins=True,
                )
            else:
                warnings.warn(
                    f"Length mismatch between predictions ({len(predictions)}) and known sex labels ({len(known_sex)})"
                )

        return results

    def add_sex_to_dataset(
        self,
        dataset: Dict,
        fem_genes: Optional[List[str]] = None,
        mal_genes: Optional[List[str]] = None,
    ) -> Dict:
        """
        Adds a "sex" field to the given dataset with inferred sex based on the
        provided female and male gene lists. The inference is done using the
        infer_sex method. If no predictions can be made, the dataset remains
        unchanged.

        Args:
            dataset: Dict
                A dictionary containing data that will be updated with the predicted
                "sex" field under the "pheno" sub-dictionary.
            fem_genes: Optional[List[str]]
                A list of genes considered to correspond to female-associated markers.
            mal_genes: Optional[List[str]]
                A list of genes considered to correspond to male-associated markers.

        Returns:
            Dict
                The updated dataset including the predicted "sex" field under the "pheno"
                sub-dictionary, if predictions could be made.
        """
        sex_predictions = self.infer_sex(dataset, fem_genes, mal_genes)
        if sex_predictions is not None:
            dataset.setdefault("pheno", {})["sex"] = sex_predictions
        return dataset

    def batch_impute_sex(
        self,
        meta_object: Dict,
        fem_genes: Optional[List[str]] = None,
        mal_genes: Optional[List[str]] = None,
    ) -> Dict:
        """
        Perform batch imputation of sex attributes for datasets within the given meta-object.

        The method iterates over each dataset contained in the 'originalData' key of the given
        meta-object, performing sex imputation using specified lists of female and male genes.
        The updated datasets are then stored in a copy of the original meta-object, which is returned.

        Args:
            meta_object (Dict): A dictionary containing the 'originalData' key with datasets
                to undergo sex imputation. Must contain the 'originalData' entry.
            fem_genes (Optional[List[str]]): A list of genes associated with female sex, used
                to assist in the imputation process. If not provided, a default set might be used.
            mal_genes (Optional[List[str]]): A list of genes associated with male sex, used
                to assist in the imputation process. If not provided, a default set might be used.

        Returns:
            Dict: A dictionary containing the updated 'originalData' with imputed sex attributes.

        Raises:
            ValueError: If the provided meta_object does not contain the 'originalData' key.
        """
        if "originalData" not in meta_object:
            raise ValueError("Meta object must contain originalData")

        updated_meta = meta_object.copy()

        for dataset_name, dataset in meta_object["originalData"].items():
            print(f"Imputing sex for dataset: {dataset_name}")
            updated_dataset = self.add_sex_to_dataset(dataset, fem_genes, mal_genes)
            updated_meta["originalData"][dataset_name] = updated_dataset

        return updated_meta


# ------------------------------------------------------------------
# Convenience function for backward compatibility
# ------------------------------------------------------------------
def impute_sex(
    dataset: Dict,
    fem_genes: Optional[List[str]] = None,
    mal_genes: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """
    Imputes the biological sex of samples in a given dataset based on the expression
    levels of specified female and male genes. This function acts as a wrapper around
    the BiologicalAnnotator's infer_sex method.

    Args:
        dataset: Dict
            A dictionary containing data for which the sex needs to be imputed. The
            dataset structure and required fields must be compatible with the
            BiologicalAnnotator class's requirements.
        fem_genes: Optional[List[str]]
            A list of gene identifiers considered indicative of female sex. Defaults
            to None, which may trigger the use of a default or predefined list within
            the BiologicalAnnotator.
        mal_genes: Optional[List[str]]
            A list of gene identifiers considered indicative of male sex. Defaults
            to None, which may trigger the use of a default or predefined list within
            the BiologicalAnnotator.

    Returns:
        Optional[np.ndarray]: An array containing the computed biological sex for
        each sample in the dataset. The format and exact interpretation of the results
        depend on the implementation within the BiologicalAnnotator. Returns None
        if the imputation process fails or if the implementation specifies no output.
    """
    annotator = BiologicalAnnotator()
    return annotator.infer_sex(
        dataset, fem_genes, mal_genes
    )  # FIXED: was calling impute_sex instead
