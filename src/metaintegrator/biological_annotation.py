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

    def __init__(self) -> None:
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
        """Infer each sample's sex from expression of known sex-linked genes.

        Filters the female/male gene panels (defaults from iSEXS, or the
        panels supplied by the caller) down to genes present in the
        dataset, k-means clusters samples into two groups, and labels
        each cluster by whether its centroid looks more female- or
        male-like. Emits a warning (rather than raising) on any failure
        path, such as missing fields, no sex-linked genes found, too few
        samples, or an ambiguous cluster split.

        Args:
            dataset: Dict with ``'expr'`` (expression matrix) and
                ``'keys'`` (gene names) entries.
            fem_genes: Female-associated genes to use. Falls back to this
                instance's default female gene panel when not given.
            mal_genes: Male-associated genes to use. Falls back to this
                instance's default male gene panel when not given.
            random_state: Random seed for the k-means clustering, for
                reproducibility.

        Returns:
            An array of ``'female'``/``'male'`` labels, one per sample,
            or None if sex could not be determined.
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
        """Return the expression rows for a set of genes, if present in the dataset.

        Args:
            dataset: Dict with an ``"expr"`` entry (expression data, as a
                DataFrame or numpy-like array) and a ``"keys"`` entry
                (gene names aligned to its rows).
            genes: Gene names to filter down to.

        Returns:
            A DataFrame of the matching rows, or None if no requested
            genes are present or an error occurs.
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
        """Decide which k-means cluster centroid looks female vs. male.

        Compares each centroid's mean expression across the female- and
        male-associated genes: the female cluster should show high
        expression of female genes and low expression of male genes.

        Args:
            centroids: Cluster centroids, shape ``(n_clusters, n_genes)``.
            all_genes: Gene names corresponding to the centroid columns.
            fem_genes: Genes associated with female expression.
            mal_genes: Genes associated with male expression.

        Returns:
            Index of the female-like cluster, or None if the two gene
            panels disagree on which cluster is female (only possible
            when both panels are available).
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
        """Summarize sex predictions, optionally scored against known labels.

        Args:
            predictions: Predicted labels (``'female'``/``'male'``), one
                per sample.
            known_sex: Ground-truth labels to score against, if
                available. Must be the same length as ``predictions``.

        Returns:
            A dict with ``n_samples``, ``n_female``, ``n_male``, and
            ``female_ratio``; plus ``accuracy`` and ``confusion_matrix``
            when ``known_sex`` is given and matches ``predictions`` in
            length.
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
        """Infer sex and add it to the dataset's phenotype data.

        Calls :meth:`infer_sex` and, if it succeeds, stores the result as
        ``dataset['pheno']['sex']``. Leaves the dataset unchanged if
        inference fails.

        Args:
            dataset: Dataset dict to update.
            fem_genes: Female-associated genes to use (see
                :meth:`infer_sex`).
            mal_genes: Male-associated genes to use (see
                :meth:`infer_sex`).

        Returns:
            The dataset, with ``pheno.sex`` set if inference succeeded.
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
    """Infer biological sex from expression data (convenience wrapper).

    A thin wrapper around :meth:`BiologicalAnnotator.infer_sex`.

    Args:
        dataset: Dataset dict compatible with :class:`BiologicalAnnotator`
            (needs ``'expr'`` and ``'keys'`` entries).
        fem_genes: Female-associated genes to use. Falls back to
            BiologicalAnnotator's defaults when not given.
        mal_genes: Male-associated genes to use. Falls back to
            BiologicalAnnotator's defaults when not given.

    Returns:
        An array of inferred sex labels, or None if inference failed.
    """
    annotator = BiologicalAnnotator()
    return annotator.infer_sex(dataset, fem_genes, mal_genes)
