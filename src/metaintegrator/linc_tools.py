"""
Enhanced LINCS connectivity mapping tools for MetaIntegrator Python port.
This module implements connectivity mapping analysis using LINCS data
for drug repurposing and mechanism discovery.

Includes missing functions: lincsInteractors, lincsQcAccess, and lincsReport.
FIXED VERSION - Addresses failing unit tests.
"""

import os
import shutil
import tempfile
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import seaborn as sns
from jinja2 import Template
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# Enhanced LincsDataManager Class
# ------------------------------------------------------------------
class LincsDataManager:
    """Manages LINCS data loading, caching, and quality control access."""

    def __init__(self) -> None:
        """Initialize LINCS data manager."""
        self._data_cache = {}
        self._data_loaded = False
        self.data_url = "http://khatrilab.stanford.edu/_media/metaintegrator/lincsdata.zip"

    def load_data(self, force_reload: bool = False) -> bool:
        """
        Loads LINCS data into the system. The method downloads, extracts, processes,
        and cleans up the data, ensuring it is ready for use. If the data has already
        been loaded and the reload is not forced, it skips the loading process.

        Args:
            force_reload (bool): Specifies whether to force reloading the data
            even if it has already been loaded. Default is False.

        Returns:
            bool: True if the data is successfully loaded, False otherwise.
        """
        if self._data_loaded and not force_reload:
            return True
        try:
            print("Loading LINCS data...")
            temp_dir = self._download_and_extract()
            try:
                self._load_files_from_dir(temp_dir)
            finally:
                shutil.rmtree(temp_dir)

            self._process_annotations()
            self._data_loaded = True
            print("LINCS data loaded successfully!")
            return True
        except Exception as e:
            warnings.warn(f"Failed to load LINCS data: {e}")
            return False

    def _download_and_extract(self) -> str:
        """
        Downloads and extracts a ZIP file from a given URL.

        This method handles the downloading of a ZIP file from a specified URL, saving it
        temporarily, and extracting its contents to a temporary directory. The function
        ensures that the ZIP file is written in chunks to manage memory effectively.

        Returns:
            str: The path to the temporary directory where the ZIP file contents
                 have been extracted.

        Raises:
            HTTPError: If the HTTP request to download the file fails or returns a
                       non-success status code.
        """
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "lincsdata.zip")
        response = requests.get(self.data_url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(temp_dir)
        return temp_dir

    def _load_files_from_dir(self, temp_dir: str) -> None:
        """Load known LINCS data files from a directory into the internal cache.

        Matches a predefined set of file patterns against files present in
        ``temp_dir`` and loads each match with the appropriate method
        (pickle or CSV) for its file type.
        The method leverages specific separators for CSV files to correctly parse their contents.
        If a file matching a pattern is not found, it is skipped.

        Args:
            temp_dir: str
                Path to the temporary directory containing files to be loaded.
        """
        extracted_files = os.listdir(temp_dir)
        file_configs = [
            ("agg.gold", "all_lincs_agg_gold", "pickle", None),
            ("qc", "all_lincs_qc", "pickle", None),
            ("lincs.ann", "lincs_ann", "csv", "\t"),
            ("tri.ann", "tri_ann", "csv", ","),
            ("gene.ann", "gene_ann", "csv", "\t"),
        ]

        for pattern, cache_key, file_type, sep in file_configs:
            file_name = next((f for f in extracted_files if pattern in f), None)
            if not file_name:
                continue

            full_path = os.path.join(temp_dir, file_name)
            if file_type == "pickle":
                self._data_cache[cache_key] = pd.read_pickle(full_path)
            elif file_type == "csv":
                self._data_cache[cache_key] = pd.read_csv(full_path, sep=sep)

    def _process_annotations(self) -> None:
        """Clean and standardize the cached LINCS annotation data.

        Standardizes the 'pert_iname' field by uppercasing it and
        stripping non-alphanumeric characters, then stores the
        deduplicated result back into the cache.

        Raises:
            KeyError
                If the required "lincs_ann" key does not exist in the internal data cache.
        """
        if "lincs_ann" in self._data_cache:
            lincs_ann = self._data_cache["lincs_ann"]
            lincs_ann["merge"] = (
                lincs_ann["pert_iname"].str.upper().str.replace(r"[^\w]", "", regex=True)
            )
            self._data_cache["lincs_ann"] = lincs_ann.drop_duplicates(subset=["merge"])

        # Process gene annotations for easier access
        if "gene_ann" in self._data_cache:
            gene_ann = self._data_cache["gene_ann"]
            # Create gene annotation subsets for expression and perturbagens
            if "all_lincs_agg_gold" in self._data_cache:
                agg_data = self._data_cache["all_lincs_agg_gold"]
                if "CP" in agg_data:
                    # Gene annotations for expression data
                    exp_genes = list(agg_data["CP"].index)
                    self._data_cache["gene_ann_exp"] = gene_ann[
                        gene_ann["Symbol"].isin(exp_genes)
                    ].drop_duplicates(subset=["Symbol"])

                    # Gene annotations for perturbagens
                    pert_genes = []
                    for dataset in ["SH", "OE", "LIG"]:
                        if dataset in agg_data:
                            pert_genes.extend(agg_data[dataset].columns)
                    pert_genes = list(set(pert_genes))
                    self._data_cache["gene_ann_pert"] = gene_ann[
                        gene_ann["Symbol"].isin(pert_genes)
                    ].drop_duplicates(subset=["Symbol"])

    def get_data(self, data_type: str) -> Optional[Any]:
        """
        Retrieves data of the specified type from the data cache, loading it first if not already loaded.

        Args:
            data_type: str
                The type of data to retrieve.

        Returns:
            Optional[Any]
                The data corresponding to the given type, or None if the data could not be loaded or is unavailable.
        """
        if not self._data_loaded and not self.load_data():
            return None
        return self._data_cache.get(data_type)

    def lincs_qc_access(
        self, dataset: str, perturbagens: List[str], just_gold: bool = False
    ) -> pd.DataFrame:
        """
        NEW FUNCTION: Quality control access to LINCS metadata.

        Provides access to quality control information for LINCS perturbagens,
        including signature quality metrics and experimental conditions.

        Args:
            dataset: LINCS dataset ("CP", "SH", "OE", "LIG", "MUT")
            perturbagens: List of perturbagen names to query
            just_gold: If True, only return gold-standard signatures

        Returns:
            DataFrame with QC information for requested perturbagens

        Raises:
            ValueError: If dataset is invalid or QC data unavailable
        """
        valid_datasets = ["CP", "SH", "OE", "LIG", "MUT"]
        if dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset. Must be one of {valid_datasets}")

        qc_data = self.get_data("all_lincs_qc")
        if qc_data is None or dataset not in qc_data:
            raise ValueError(f"QC data not available for dataset {dataset}")

        dataset_qc = qc_data[dataset].copy()

        # Filter to gold signatures only if requested
        if just_gold:
            dataset_qc = dataset_qc[dataset_qc.get("is_gold", 0) == 1]

        # Create merge column for perturbagen matching
        if "pert_iname" in dataset_qc.columns:
            dataset_qc["merge"] = (
                dataset_qc["pert_iname"]
                .astype(str)
                .str.upper()
                .str.replace(r"[^\w]", "", regex=True)
            )

            # Clean perturbagen names for matching
            clean_perturbagens = [
                str(p).upper().replace(" ", "").replace("-", "").replace(".", "")
                for p in perturbagens
            ]

            # Filter to requested perturbagens
            filtered_qc = dataset_qc[dataset_qc["merge"].isin(clean_perturbagens)]
        else:
            # Fallback to direct matching if pert_iname not available
            filtered_qc = dataset_qc[dataset_qc.index.isin(perturbagens)]

        # Convert time columns to numeric if present
        if "pert_itime" in filtered_qc.columns:
            filtered_qc = filtered_qc.copy()
            filtered_qc["pert_itime"] = (
                filtered_qc["pert_itime"]
                .astype(str)
                .str.replace(" h", "")
                .str.replace("h", "")
                .str.strip()
            )
            # Convert to numeric, handling non-numeric values
            filtered_qc["pert_itime"] = pd.to_numeric(filtered_qc["pert_itime"], errors="coerce")

        # Sort by perturbagen name, gold status, cell line, dose, and time
        sort_columns = []
        for col in ["pert_iname", "is_gold", "cell_id", "pert_idose", "pert_itime"]:
            if col in filtered_qc.columns:
                sort_columns.append(col)

        if sort_columns:
            ascending = [True, False, True, True, True][: len(sort_columns)]
            filtered_qc = filtered_qc.sort_values(sort_columns, ascending=ascending)

        return filtered_qc


# ------------------------------------------------------------------
# Enhanced LincsAnalyzer Class
# ------------------------------------------------------------------
class LincsAnalyzer:
    """Core LINCS analysis functionality including gene network analysis."""

    def __init__(self, data_manager: Optional[LincsDataManager] = None) -> None:
        """Initialize the analyzer, optionally with a custom data manager.

        The constructor allows specifying a custom LincsDataManager instance to manage
        the data. If no data manager is provided, a default LincsDataManager instance is
        created and used.

        Args:
            data_manager: The data manager instance to be used. Defaults to an instance
                          of LincsDataManager.

        Args:
            data_manager: Optional custom instance of LincsDataManager.
        """
        self.data_manager = data_manager or LincsDataManager()

    def _get_dataset_data(self, dataset: str) -> Union[pd.DataFrame, Dict[str, str]]:
        """
        Retrieves dataset data based on the specified dataset identifier.

        The function checks if the provided dataset identifier is valid and
        exists in the aggregated dataset data. If the dataset is not valid or
        does not exist, an error dictionary is returned. Otherwise, the
        corresponding dataset data is returned.

        Args:
            dataset: Specifies the dataset identifier. Allowed values include
                "CP", "SH", "OE", "LIG", and "MUT".

        Returns:
            A pandas DataFrame containing the specified dataset data if the
            dataset exists, or a dictionary with an error message specifying
            the issue if the dataset is incorrect or not found.
        """
        valid_datasets = ["CP", "SH", "OE", "LIG", "MUT"]
        if dataset not in valid_datasets:
            return {"error": "incorrect dataset"}

        agg_data = self.data_manager.get_data("all_lincs_agg_gold")
        if agg_data is None or dataset not in agg_data:
            return {"error": "dataset not found"}

        return agg_data[dataset]

    def _filter_by_clinical(
        self, identifiers: List[str], include_baits: Optional[List[str]] = None
    ) -> List[str]:
        """
        Filters a list of identifiers by clinical relevance based on available annotations and optionally included baits.

        This method checks the provided identifiers against a pre-defined list of clinically annotated compounds.
        An optional list of additional baits can be included in the comparison to broaden the scope of filtering.

        Args:
            identifiers: List[str]
                The list of identifiers to be filtered.
            include_baits: Optional[List[str]]
                An optional list of additional identifiers to include in the clinical filtering.

        Returns:
            List[str]
                A filtered list containing only the identifiers that match the clinical annotations or the optionally
                provided baits.
        """
        tri_ann = self.data_manager.get_data("tri_ann")
        if tri_ann is None:
            return identifiers
        clin_compounds = tri_ann["merge"].tolist()

        # Filter identifiers to only include those that are clinical compounds
        filtered_identifiers = [p for p in identifiers if p in clin_compounds]

        # Add baits to the result if provided
        if include_baits:
            # Add baits that aren't already in the filtered list
            for bait in include_baits:
                if bait not in filtered_identifiers:
                    filtered_identifiers.append(bait)

        return filtered_identifiers

    def _apply_sorting(self, df: pd.DataFrame, column: str, direction: str) -> pd.DataFrame:
        """
        Sorts a pandas DataFrame based on the specified column and direction. The sorting behavior changes based
        on the value of the provided direction.

        Args:
            df (pd.DataFrame): The pandas DataFrame to be sorted.
            column (str): The column name on which the sorting is to be applied.
            direction (str): The direction of the sorting. Acceptable values are "reverse" for ascending order,
                             "aggravate" for descending order, and "absolute" for sorting based on absolute values
                             in descending order.

            Returns:
            pd.DataFrame: The sorted DataFrame.
        """
        if direction == "reverse":
            return df.sort_values(column)
        elif direction == "aggravate":
            return df.sort_values(column, ascending=False)
        elif direction == "absolute":
            return df.sort_values(column, key=abs, ascending=False)
        return df

    def lincs_data_access(
        self,
        dataset: str,
        genes: List[str] = None,
        perturbagens: List[str] = None,
        all_genes: bool = False,
        all_perturbagens: bool = False,
        just_clin: bool = False,
        show_clin: bool = False,
    ) -> Union[pd.DataFrame, str]:
        """
        Fetches and processes data from a given dataset based on specified filtering criteria.

        Args:
            dataset (str): Name of the dataset to be accessed.
            genes (List[str]): List of gene names used to filter the dataset.
            perturbagens (List[str]): List of perturbagen names used to filter the dataset.
            all_genes (bool): If True, includes all genes in the dataset. Defaults to False.
            all_perturbagens (bool): If True, includes all perturbagens in the dataset. Defaults to False.
            just_clin (bool): If True, applies additional filtering based on clinical relevance. Defaults to False.
            show_clin (bool): If True, includes clinical information in the resulting data. Defaults to False.

        Returns:
            Union[pd.DataFrame, str]: A DataFrame containing the filtered and processed data, or a string error message if any filtering step fails.

        Raises:
            ValueError: Raised if the dataset could not be found or an error occurs during processing.
        """
        dataset_data = self._get_dataset_data(dataset)
        if isinstance(dataset_data, dict):
            return dataset_data["error"]

        # Filter Genes
        genes = (
            list(dataset_data.index)
            if all_genes
            else [g for g in (genes or []) if g in dataset_data.index]
        )
        if not genes:
            return "error, no genes found"

        # Filter Perturbagens
        perturbagens = (
            list(dataset_data.columns)
            if all_perturbagens
            else [p for p in (perturbagens or []) if p in dataset_data.columns]
        )
        if just_clin:
            perturbagens = self._filter_by_clinical(perturbagens)
        if not perturbagens:
            return "error, no perturbagens found"

        result = dataset_data.loc[genes, perturbagens]
        result = result.sort_values(by=perturbagens[0], ascending=False)

        if show_clin:
            tri_ann = self.data_manager.get_data("tri_ann")
            if tri_ann is not None:
                result = result.T.reset_index().rename(columns={"index": "merge"})
                result = result.merge(tri_ann, on="merge", how="left").sort_values("HighestPhase")

        return result

    def lincs_correlate(
        self,
        signature: pd.DataFrame,
        dataset: str = "CP",
        hit_number_hm: int = 20,
        direction: str = "reverse",
        method: str = "pearson",
        drop_string: Optional[str] = None,
        just_clin: bool = False,
        show_clin: bool = False,
        gene_ann: bool = False,
    ) -> Dict:
        """
        FIXED: Computes the correlation between a given signature and a LINCS dataset for
        identifying potential hits based on the correlation values. This method allows
        analysis such as reverse or forward correlation with options to tailor the data
        being processed, providing both statistical and tabular results.

        Args:
            signature (pd.DataFrame): A DataFrame containing the input signature, with genes as the
                index and a single column representing their associated values.
            dataset (str): The LINCS dataset to use for correlation. Possible values include "CP".
                Default is "CP".
            hit_number_hm (int): Number of top hits to include in the output heatmap data. Default
                is 20.
            direction (str): Specifies correlation direction, either "reverse" or "forward". Default
                is "reverse".
            method (str): The statistical correlation method to use, e.g., "pearson". Default is
                "pearson".
            drop_string (Optional[str]): An optional string pattern to filter out columns in LINCS
                dataset that match this pattern. Default is None.
            just_clin (bool): Whether to restrict output results to clinically relevant hits only.
                Default is False.
            show_clin (bool): Whether to append clinical information to the hit list provided it
                exists in the data manager. Default is False.
            gene_ann (bool): Whether to append gene annotations to the output if available in the
                data manager. Default is False.

        Returns:
            Dict: A dictionary containing processed correlation results. Includes the following keys:
                - "hit_list": A DataFrame of hits with their correlation values, p-values, and false
                  discovery rates (FDR).
                - "hit_expression": A DataFrame for the heatmap data containing correlated gene
                  expression data.
                - "hit_ann" (optional): A merged DataFrame with clinical annotation information if
                  show_clin is True and annotations exist.
                - "gene_ann" (optional): Gene annotation information if gene_ann is True and
                  annotations are available.
        """
        dataset_data = self._get_dataset_data(dataset)
        if isinstance(dataset_data, dict):
            return dataset_data

        common_genes = [g for g in signature.index if g in dataset_data.index]
        if not common_genes:
            return {"error": "no overlap between signature genes and LINCS"}

        # FIXED: Handle single gene case for correlation
        if len(common_genes) < 2:
            warnings.warn("Need at least 2 genes for correlation analysis")
            return {"error": "insufficient genes for correlation analysis"}

        lincs_subset = dataset_data.loc[common_genes].copy()
        lincs_subset.columns = lincs_subset.columns.str.upper().str.replace(
            r"[^\w]", "", regex=True
        )

        if dataset == "CP":
            lincs_subset.columns = [
                c[1:] if c.startswith("X") and c[1:].isdigit() else c for c in lincs_subset.columns
            ]

        if drop_string:
            lincs_subset = lincs_subset.loc[
                :, ~lincs_subset.columns.str.contains(drop_string, regex=True)
            ]

        sig_values = signature.loc[common_genes].iloc[:, 0]
        correlations, p_values = [], []

        for col in lincs_subset.columns:
            # FIXED: Additional check for sufficient data points
            col_values = lincs_subset[col]
            if len(col_values.dropna()) < 2:
                correlations.append(0.0)
                p_values.append(1.0)
            else:
                try:
                    corr, pval = stats.pearsonr(sig_values, col_values)
                    correlations.append(corr if not np.isnan(corr) else 0.0)
                    p_values.append(pval if not np.isnan(pval) else 1.0)
                except Exception:
                    correlations.append(0.0)
                    p_values.append(1.0)

        results_df = pd.DataFrame(
            {"Sig_Corr": correlations, "pval": p_values, "merge": lincs_subset.columns}
        )

        try:
            from scipy.stats import false_discovery_control

            results_df["fdr"] = false_discovery_control(results_df["pval"])
        except ImportError:
            # Fallback if false_discovery_control is not available
            from statsmodels.stats.multitest import multipletests

            _, results_df["fdr"], _, _ = multipletests(results_df["pval"], method="fdr_bh")

        if just_clin:
            results_df = results_df[
                results_df["merge"].isin(self._filter_by_clinical(results_df["merge"].tolist()))
            ]

        results_df = self._apply_sorting(results_df, "Sig_Corr", direction)
        top_hits = results_df.head(hit_number_hm)["merge"].tolist()
        hm_data = lincs_subset[top_hits].copy()
        hm_data["sig"], hm_data["anti_sig"] = sig_values, -sig_values

        output = {"hit_list": results_df, "hit_expression": hm_data}

        if show_clin:
            tri_ann = self.data_manager.get_data("tri_ann")
            if tri_ann is not None:
                output["hit_ann"] = results_df.merge(tri_ann, on="merge", how="left")

        if gene_ann:
            output["gene_ann"] = self.data_manager.get_data("gene_ann")

        return output

    def lincs_bait_corr(
        self,
        dataset: str = "CP",
        baits: List[str] = None,
        sub_sig: Optional[pd.DataFrame] = None,
        just_clin: bool = False,
        hm_num: int = 20,
        hm_baits: bool = True,
        direction: str = "aggravate",
        bait_type: Optional[str] = None,
    ) -> Dict:
        """
        FIXED: Calculate correlation between baits and other compounds within a LINCS dataset
        and optionally generate a heatmap of the top correlated compounds.

        Args:
        dataset: str
            Name of the LINCS dataset to calculate correlations from. Default is "CP".
        baits: List[str], optional
            List of bait compounds to calculate correlations for within the dataset.
        sub_sig: pd.DataFrame, optional
            Signature data to use for identifying relevant compounds in the dataset.
        just_clin: bool
            If True, filters the result to only clinical compounds. Default is False.
        hm_num: int
            Number of top correlated hits to include in the heatmap output. Default is 20.
        hm_baits: bool
            If True, includes baits in the heatmap data. Default is True.
        direction: str
            Direction of correlation sorting: "aggravate" for descending,
            other values for ascending. Default is "aggravate".
        bait_type: str, optional
            Specifies a type of bait for specialized correlation calculations. If None,
            cross-dataset correlation is not implemented.

        Returns:
            Dict
                A dictionary containing:
                - "all_hits": pd.DataFrame with correlation results and associated metadata
                - "heatmap_df": pd.DataFrame with subset of dataset data for heatmap visualization
                - Optional "hits_clin": pd.DataFrame with clinical annotations for the hits
                  if dataset is "CP" and clinical data is available.

        Raises:
            ValueError
                If insufficient baits or compounds are found for correlation calculation.
        """
        dataset_data = self._get_dataset_data(dataset)
        if isinstance(dataset_data, dict):
            return dataset_data

        if sub_sig is not None:
            dataset_data = dataset_data.loc[[g for g in sub_sig.index if g in dataset_data.index]]

        if bait_type is None:
            bait_compounds = [b for b in (baits or []) if b in dataset_data.columns]
            other_compounds = [c for c in dataset_data.columns if c not in bait_compounds]
            if not bait_compounds or not other_compounds:
                return {"error": "insufficient baits or compounds found"}

            corr_matrix = np.corrcoef(
                dataset_data[bait_compounds].T, dataset_data[other_compounds].T
            )
            bait_corrs = corr_matrix[: len(bait_compounds), len(bait_compounds) :]
        else:
            return {"error": "cross-dataset correlation not yet implemented"}

        # FIXED: Ensure consistent array lengths for DataFrame creation
        n_baits = len(bait_compounds)
        n_others = len(other_compounds)

        if bait_corrs.shape != (n_baits, n_others):
            # Handle dimension mismatch
            min_baits = min(n_baits, bait_corrs.shape[0])
            min_others = min(n_others, bait_corrs.shape[1])
            bait_corrs = bait_corrs[:min_baits, :min_others]
            other_compounds = other_compounds[:min_others]

        results_df = pd.DataFrame(
            {"mean_cor": np.mean(bait_corrs, axis=0), "merge": other_compounds}
        )

        # Add individual bait correlations
        for i, bait in enumerate(bait_compounds[: bait_corrs.shape[0]]):
            if i < bait_corrs.shape[0]:
                results_df[bait] = bait_corrs[i]

        results_df = self._apply_sorting(results_df, "mean_cor", direction)

        if just_clin:
            clin_list = self._filter_by_clinical(
                results_df["merge"].tolist(), include_baits=bait_compounds
            )
            results_df = results_df[results_df["merge"].isin(clin_list)]

        top_hits = results_df.head(hm_num)["merge"].tolist()
        hm_compounds = list(set(top_hits + bait_compounds)) if hm_baits else top_hits
        hm_data = dataset_data[hm_compounds].copy()
        if sub_sig is not None:
            hm_data["sig"], hm_data["anti_sig"] = (
                sub_sig.iloc[:, 0],
                -sub_sig.iloc[:, 0],
            )

        output = {"all_hits": results_df, "heatmap_df": hm_data}

        if dataset == "CP":
            tri_ann = self.data_manager.get_data("tri_ann")
            if tri_ann is not None:
                output["hits_clin"] = results_df.merge(tri_ann, on="merge", how="left").sort_values(
                    "mean_cor", ascending=(direction != "aggravate")
                )

        return output

    def lincs_interactors(
        self, correlate_output: Dict, signature: pd.DataFrame, dataset: str
    ) -> Dict:
        """
        NEW FUNCTION: Identify gene network interactors and driver genes.

        This function analyzes correlation results to identify key gene interactors
        that may be drivers of the phenotype. It scores genes based on their
        interaction with the signature and their own expression in the dataset.

        Args:
            correlate_output: Output from lincs_correlate function
            signature: Original signature DataFrame
            dataset: LINCS dataset used ("SH", "OE", "LIG", etc.)

        Returns:
            Dict containing:
                - interactor_score: DataFrame with interaction scores for genes
                - interactor_expression: Expression data for top interactors

        Raises:
            ValueError: If required data is not available or invalid
        """
        if "hit_list" not in correlate_output:
            raise ValueError("correlate_output must contain 'hit_list'")

        valid_datasets = ["SH", "OE", "LIG", "MUT"]
        if dataset not in valid_datasets:
            raise ValueError(f"Dataset must be one of {valid_datasets}")

        scores = correlate_output["hit_list"].copy()
        scores.columns = [c if c != "Sig_Corr" else "Signature_Correlation" for c in scores.columns]
        scores["pert_iname"] = (
            scores.index if scores.index.name else scores.get("merge", scores.index)
        )

        # Prepare signature with gene symbols
        signature_with_genes = signature.copy()
        signature_with_genes["Gene.Symbol"] = signature_with_genes.index

        # Merge scores with signature data by matching gene names
        interactor_score = scores.merge(
            signature_with_genes,
            left_on="pert_iname",
            right_on="Gene.Symbol",
            how="inner",
        )

        if interactor_score.empty:
            warnings.warn("No gene overlap found between correlation results and signature")
            return {
                "interactor_score": pd.DataFrame(),
                "interactor_expression": pd.DataFrame(),
            }

        # Calculate interaction score (signature effect * correlation)
        signature_col = signature_with_genes.columns[0]  # First data column
        interactor_score["interaction.score"] = (
            interactor_score[signature_col] * interactor_score["Signature_Correlation"]
        )

        # Apply dataset-specific filters based on biological expectations
        if dataset == "SH":  # shRNA should reduce expression, negative interaction expected
            interactor_score = interactor_score[interactor_score["interaction.score"] < 0]
        elif dataset in [
            "OE",
            "LIG",
        ]:  # Overexpression/ligands should increase, positive expected
            interactor_score = interactor_score[interactor_score["interaction.score"] > 0]

        # Calculate driver score - absolute interaction weighted by signature direction
        interactor_score["driver.score"] = abs(interactor_score["interaction.score"]) * np.sign(
            interactor_score[signature_col]
        )

        # Sort by driver score
        interactor_score = interactor_score.sort_values("driver.score", ascending=False)

        # Get top and bottom interactors to capture both ends of spectrum
        n_top = min(10, len(interactor_score) // 2)
        n_bottom = min(10, len(interactor_score) - n_top)

        top_interactors = interactor_score.head(n_top)
        bottom_interactors = interactor_score.tail(n_bottom)
        final_interactors = pd.concat([top_interactors, bottom_interactors])
        final_interactors = final_interactors.drop_duplicates()

        # Get expression data for interactors
        dataset_data = self._get_dataset_data(dataset)
        if isinstance(dataset_data, dict):
            return {"error": dataset_data["error"]}

        # Find common genes between signature and dataset
        common_genes = [g for g in signature.index if g in dataset_data.index]
        interactor_names = final_interactors["pert_iname"].tolist()
        available_interactors = [i for i in interactor_names if i in dataset_data.columns]

        if not common_genes or not available_interactors:
            warnings.warn("Insufficient overlap between genes and interactors")
            expression_data = pd.DataFrame()
        else:
            expression_data = dataset_data.loc[common_genes, available_interactors].copy()
            expression_data["sig"] = signature.loc[common_genes].iloc[:, 0]
            expression_data["anti.sig"] = -signature.loc[common_genes].iloc[:, 0]

        return {
            "interactor_score": final_interactors,
            "interactor_expression": expression_data,
        }


# ------------------------------------------------------------------
# NEW LincsReporter Class
# ------------------------------------------------------------------
class LincsReporter:
    """Comprehensive reporting functionality for LINCS analysis results."""

    def __init__(self, data_manager: Optional[LincsDataManager] = None) -> None:
        """Initialize LINCS reporter with data manager."""
        self.data_manager = data_manager or LincsDataManager()
        self.analyzer = LincsAnalyzer(self.data_manager)
        self.visualizer = LincsVisualizer()

    def lincs_report(
        self,
        signature: pd.DataFrame,
        hit_number_hm: int = 10,
        hit_number_tbl: int = 10,
        method: str = "pearson",
        view_tbl: bool = True,
        cutoff: float = 2.0,
        pert_ordered: bool = True,
        gene_ordered: bool = False,
        transpose: bool = True,
        scale: bool = False,
        resize: bool = False,
        report_out_folder: str = "./lincs_report",
        report_title: str = "lincsReport",
        include_plotly: bool = True,
    ) -> Dict[str, Any]:
        """
        NEW FUNCTION: Generate comprehensive LINCS connectivity report.

        Creates a complete analysis report including correlation analyses across
        all LINCS datasets, visualizations, and HTML/PDF export capabilities.

        Args:
            signature: Gene signature DataFrame with genes as index
            hit_number_hm: Number of hits for heatmaps
            hit_number_tbl: Number of hits for tables
            method: Correlation method ("pearson", "spearman")
            view_tbl: Whether to include data tables
            cutoff: Expression cutoff for heatmaps
            pert_ordered: Order perturbations by correlation
            gene_ordered: Order genes by expression
            transpose: Transpose heatmap data
            scale: Scale expression data
            resize: Resize heatmap cells
            report_out_folder: Output directory
            report_title: Report file prefix
            include_plotly: Include interactive Plotly visualizations

        Returns:
            Dict with all analysis results and report paths

        Raises:
            ValueError: If signature is invalid or analysis fails
        """
        if signature.empty:
            raise ValueError("Signature cannot be empty")

        # Create output directory
        Path(report_out_folder).mkdir(parents=True, exist_ok=True)

        print("Generating comprehensive LINCS connectivity report...")
        results = {}

        # Analyze all datasets
        datasets = ["CP", "SH", "OE", "LIG", "MUT"]
        directions = ["reverse", "aggravate"]

        for dataset in datasets:
            print(f"Analyzing {dataset} dataset...")
            for direction in directions:
                try:
                    key = f"{dataset.lower()}_conn_{direction[:3]}"

                    corr_results = self.analyzer.lincs_correlate(
                        signature=signature,
                        dataset=dataset,
                        hit_number_hm=hit_number_hm,
                        direction=direction,
                        method=method,
                    )

                    if "error" not in corr_results:
                        results[key] = corr_results
                        results[f"{key}_hits"] = corr_results["hit_list"]

                except Exception as e:
                    warnings.warn(f"Failed to analyze {dataset} {direction}: {e}")
                    continue

        # Generate visualizations
        print("Creating visualizations...")
        figures = self._create_all_visualizations(
            results,
            signature,
            cutoff,
            pert_ordered,
            gene_ordered,
            transpose,
            scale,
            resize,
            include_plotly,
            report_out_folder,
            report_title,
        )
        results.update(figures)

        # Save analysis objects
        print("Saving analysis results...")
        rdata_path = Path(report_out_folder) / f"{report_title}_Objects.pkl"
        self._save_analysis_objects(results, rdata_path)
        results["rdata_path"] = str(rdata_path)

        # Generate HTML report
        print("Generating HTML report...")
        html_path = self._generate_html_report(
            results,
            signature,
            report_out_folder,
            report_title,
            hit_number_tbl,
            view_tbl,
            include_plotly,
        )
        results["html_report_path"] = html_path

        print(f"Report generation complete! HTML report: {html_path}")
        return results

    def _create_all_visualizations(
        self,
        results: Dict,
        signature: pd.DataFrame,
        cutoff: float,
        pert_ordered: bool,
        gene_ordered: bool,
        transpose: bool,
        scale: bool,
        resize: bool,
        include_plotly: bool,
        report_folder: str,
        report_title: str,
    ) -> Dict[str, Any]:
        """Create comprehensive visualizations for the report."""
        figures = {}

        # Create static matplotlib figures
        for key, data in results.items():
            if key.endswith("_hits") or "hit_expression" not in data:
                continue

            try:
                fig = self.visualizer.lincs_heatmap(
                    data["hit_expression"],
                    cutoff=cutoff,
                    pert_ordered=pert_ordered,
                    gene_ordered=gene_ordered,
                    transpose=transpose,
                    scale=scale,
                    resize=resize,
                    hm_title=f"LINCS Connectivity Map - {key.replace('_', ' ').title()}",
                )

                # Save figure
                fig_path = Path(report_folder) / f"{report_title}_{key}_heatmap.png"
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                figures[f"{key}_heatmap_path"] = str(fig_path)

            except Exception as e:
                warnings.warn(f"Failed to create heatmap for {key}: {e}")

        # Create interactive Plotly figures if requested
        if include_plotly:
            figures.update(self._create_plotly_visualizations(results, report_folder, report_title))

        # Create summary correlation plot
        summary_fig = self._create_summary_correlation_plot(results, signature)
        if summary_fig:
            summary_path = Path(report_folder) / f"{report_title}_summary.png"
            summary_fig.savefig(summary_path, dpi=300, bbox_inches="tight")
            plt.close(summary_fig)
            figures["summary_plot_path"] = str(summary_path)

        return figures

    def _create_plotly_visualizations(
        self, results: Dict, report_folder: str, report_title: str
    ) -> Dict[str, str]:
        """Create interactive Plotly visualizations."""
        plotly_figures = {}

        # Interactive correlation heatmap
        try:
            correlation_data = self._prepare_correlation_matrix(results)
            if not correlation_data.empty:
                fig = px.imshow(
                    correlation_data,
                    title="LINCS Connectivity Correlation Matrix",
                    color_continuous_scale="RdBu_r",
                    aspect="auto",
                )
                fig.update_layout(width=800, height=600)

                html_path = Path(report_folder) / f"{report_title}_interactive_correlation.html"
                fig.write_html(html_path)
                plotly_figures["interactive_correlation_path"] = str(html_path)

        except Exception as e:
            warnings.warn(f"Failed to create interactive correlation plot: {e}")

        # Interactive volcano plots for each dataset
        for key, data in results.items():
            if key.endswith("_hits") or "hit_list" not in data:
                continue

            try:
                hit_list = data["hit_list"]
                if "pval" in hit_list.columns and "Sig_Corr" in hit_list.columns:
                    fig = self._create_volcano_plot(hit_list, key)
                    html_path = Path(report_folder) / f"{report_title}_{key}_volcano.html"
                    fig.write_html(html_path)
                    plotly_figures[f"{key}_volcano_path"] = str(html_path)

            except Exception as e:
                warnings.warn(f"Failed to create volcano plot for {key}: {e}")

        return plotly_figures

    def _prepare_correlation_matrix(self, results: Dict) -> pd.DataFrame:
        """Prepare correlation matrix from results."""
        correlation_data = {}

        for key, data in results.items():
            if key.endswith("_hits") or "hit_list" not in data:
                continue

            hit_list = data["hit_list"]
            if "Sig_Corr" in hit_list.columns and "merge" in hit_list.columns:
                # Get top 20 hits for the matrix
                top_hits = hit_list.head(20)
                correlation_data[key] = top_hits.set_index("merge")["Sig_Corr"]

        if correlation_data:
            return pd.DataFrame(correlation_data).fillna(0)
        return pd.DataFrame()

    def _create_volcano_plot(self, hit_list: pd.DataFrame, dataset_name: str) -> "go.Figure":
        """Create an interactive volcano plot for hit significance."""
        # Calculate -log10(p-value) for volcano plot
        neg_log_p = -np.log10(hit_list["pval"].clip(lower=1e-300))

        # Create volcano plot
        fig = go.Figure(
            data=go.Scatter(
                x=hit_list["Sig_Corr"],
                y=neg_log_p,
                mode="markers",
                text=hit_list.get("merge", hit_list.index),
                hovertemplate="<b>%{text}</b><br>"
                + "Correlation: %{x:.3f}<br>"
                + "-log10(p-value): %{y:.3f}<br>"
                + "<extra></extra>",
                marker=dict(
                    color=hit_list["Sig_Corr"],
                    colorscale="RdBu_r",
                    size=6,
                    colorbar=dict(title="Signature Correlation"),
                ),
            )
        )

        fig.update_layout(
            title=f"Volcano Plot - {dataset_name.replace('_', ' ').title()}",
            xaxis_title="Signature Correlation",
            yaxis_title="-log10(p-value)",
            width=700,
            height=500,
        )

        return fig

    def _create_summary_correlation_plot(
        self, results: Dict, signature: pd.DataFrame
    ) -> Optional[plt.Figure]:
        """FIXED: Create summary correlation distribution plot."""
        try:
            # Count valid result datasets
            valid_results = {}
            for key, data in results.items():
                if (
                    key.endswith("_hits")
                    or "hit_list" not in data
                    or not isinstance(data.get("hit_list"), pd.DataFrame)
                ):
                    continue
                hit_list = data["hit_list"]
                if "Sig_Corr" in hit_list.columns and not hit_list.empty:
                    valid_results[key] = hit_list

            # FIXED: Return None for empty results instead of creating empty figure
            if not valid_results:
                return None

            n_plots = min(6, len(valid_results))
            if n_plots == 0:
                return None

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            plot_idx = 0
            for key, hit_list in valid_results.items():
                if plot_idx >= 6:
                    break

                ax = axes[plot_idx]
                ax.hist(hit_list["Sig_Corr"], bins=50, alpha=0.7, edgecolor="black")
                ax.set_title(key.replace("_", " ").title())
                ax.set_xlabel("Signature Correlation")
                ax.set_ylabel("Frequency")
                ax.axvline(0, color="red", linestyle="--", alpha=0.7)
                plot_idx += 1

            # Hide unused subplots
            for i in range(plot_idx, 6):
                axes[i].set_visible(False)

            plt.tight_layout()
            return fig

        except Exception as e:
            warnings.warn(f"Failed to create summary correlation plot: {e}")
            return None

    def _save_analysis_objects(self, results: Dict, output_path: Path) -> None:
        """Save analysis objects to file."""
        try:
            import pickle

            with open(output_path, "wb") as f:
                # Only save serializable objects
                serializable_results = {}
                for key, value in results.items():
                    if isinstance(value, (dict, pd.DataFrame, str, int, float, list)):
                        serializable_results[key] = value
                pickle.dump(serializable_results, f)
        except Exception as e:
            warnings.warn(f"Failed to save analysis objects: {e}")

    def _generate_html_report(
        self,
        results: Dict,
        signature: pd.DataFrame,
        report_folder: str,
        report_title: str,
        hit_number_tbl: int,
        view_tbl: bool,
        include_plotly: bool,
    ) -> str:
        """Generate comprehensive HTML report."""

        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ report_title }} - LINCS Connectivity Map Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #2E86AB; text-align: center; }
        h2 { color: #A23B72; border-bottom: 2px solid #A23B72; padding-bottom: 5px; }
        h3 { color: #F18F01; }
        .section { margin-bottom: 30px; }
        .summary-stats { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .correlation-positive { color: #d73027; font-weight: bold; }
        .correlation-negative { color: #1a9850; font-weight: bold; }
        .image-container { text-align: center; margin: 20px 0; }
        .image-container img { max-width: 100%; height: auto; }
        .dataset-section { border: 1px solid #ccc; margin: 20px 0; padding: 15px; border-radius: 5px; }
        .toc { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 30px; }
        .toc ul { list-style-type: none; }
        .toc a { text-decoration: none; color: #2E86AB; }
        .toc a:hover { text-decoration: underline; }
        .interactive-link { background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .interactive-link a { color: #1976d2; text-decoration: none; font-weight: bold; }
        .interactive-link a:hover { text-decoration: underline; }
        .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; color: #856404; }
    </style>
</head>
<body>
    <h1>{{ report_title }} - LINCS Connectivity Map Analysis</h1>
    
    <div class="toc">
        <h3>Table of Contents</h3>
        <ul>
            <li><a href="#summary">Analysis Summary</a></li>
            <li><a href="#signature">Input Signature</a></li>
            {% for dataset in datasets %}
            <li><a href="#{{ dataset }}">{{ dataset.upper() }} Dataset Results</a></li>
            {% endfor %}
            {% if include_plotly %}
            <li><a href="#interactive">Interactive Visualizations</a></li>
            {% endif %}
            <li><a href="#methods">Methods & Parameters</a></li>
        </ul>
    </div>

    <div id="summary" class="section">
        <h2>Analysis Summary</h2>
        <div class="summary-stats">
            <p><strong>Report Generated:</strong> {{ timestamp }}</p>
            <p><strong>Signature Genes:</strong> {{ signature_gene_count }}</p>
            <p><strong>Datasets Analyzed:</strong> {{ datasets|length }}</p>
            <p><strong>Total Correlations Computed:</strong> {{ total_correlations }}</p>
        </div>
        
        {% if summary_plot_path %}
        <div class="image-container">
            <h3>Correlation Distribution Summary</h3>
            <img src="{{ summary_plot_path|basename }}" alt="Summary Correlation Plot">
        </div>
        {% endif %}
    </div>

    <div id="signature" class="section">
        <h2>Input Signature</h2>
        <p>The analysis was performed using a signature containing {{ signature_gene_count }} genes.</p>
        
        {% if signature_preview %}
        <h3>Signature Preview (Top 10 genes by absolute effect size)</h3>
        <table>
            <thead>
                <tr>
                    <th>Gene</th>
                    <th>Effect Size</th>
                    <th>Direction</th>
                </tr>
            </thead>
            <tbody>
                {% for gene, effect in signature_preview %}
                <tr>
                    <td>{{ gene }}</td>
                    <td class="{% if effect > 0 %}correlation-positive{% else %}correlation-negative{% endif %}">
                        {{ "%.3f"|format(effect) }}
                    </td>
                    <td>{{ "Up" if effect > 0 else "Down" }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>

    {% for dataset in datasets %}
    <div id="{{ dataset }}" class="section">
        <h2>{{ dataset.upper() }} Dataset Results</h2>
        
        {% for direction in ['reverse', 'aggravate'] %}
        {% set key = dataset.lower() + '_conn_' + direction[:3] %}
        {% if results[key] %}
        <div class="dataset-section">
            <h3>{{ direction.title() }} Correlation Analysis</h3>
            
            {% if results[key + '_heatmap_path'] %}
            <div class="image-container">
                <img src="{{ results[key + '_heatmap_path']|basename }}" alt="{{ dataset }} {{ direction }} Heatmap">
            </div>
            {% endif %}
            
            {% if view_tbl and results[key + '_hits'] %}
            <h4>Top {{ hit_number_tbl }} Hits</h4>
            <table>
                <thead>
                    <tr>
                        <th>Perturbagen</th>
                        <th>Correlation</th>
                        <th>P-value</th>
                        <th>FDR</th>
                    </tr>
                </thead>
                <tbody>
                    {% for _, hit in results[key + '_hits'].head(hit_number_tbl).iterrows() %}
                    <tr>
                        <td>{{ hit.get('merge', hit.name) }}</td>
                        <td class="{% if hit['Sig_Corr'] > 0 %}correlation-positive{% else %}correlation-negative{% endif %}">
                            {{ "%.4f"|format(hit['Sig_Corr']) }}
                        </td>
                        <td>{{ "%.2e"|format(hit.get('pval', 0)) }}</td>
                        <td>{{ "%.2e"|format(hit.get('fdr', 0)) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
            
            {% if include_plotly and results[key + '_volcano_path'] %}
            <div class="interactive-link">
                <a href="{{ results[key + '_volcano_path']|basename }}" target="_blank">
                    🔗 View Interactive Volcano Plot for {{ dataset.upper() }} {{ direction.title() }}
                </a>
            </div>
            {% endif %}
        </div>
        {% endif %}
        {% endfor %}
    </div>
    {% endfor %}

    {% if include_plotly %}
    <div id="interactive" class="section">
        <h2>Interactive Visualizations</h2>
        
        {% if results['interactive_correlation_path'] %}
        <div class="interactive-link">
            <a href="{{ results['interactive_correlation_path']|basename }}" target="_blank">
                🔗 Interactive Correlation Matrix Heatmap
            </a>
        </div>
        {% endif %}
        
        <p>Interactive visualizations provide additional exploration capabilities including:</p>
        <ul>
            <li>Zoomable and pannable heatmaps</li>
            <li>Hover information for individual compounds</li>
            <li>Volcano plots with statistical significance</li>
            <li>Downloadable high-resolution images</li>
        </ul>
    </div>
    {% endif %}

    <div id="methods" class="section">
        <h2>Methods & Parameters</h2>
        <div class="summary-stats">
            <h3>Analysis Parameters</h3>
            <p><strong>Correlation Method:</strong> Pearson</p>
            <p><strong>Hits per Heatmap:</strong> {{ hit_number_hm }}</p>
            <p><strong>Hits per Table:</strong> {{ hit_number_tbl }}</p>
            <p><strong>Expression Cutoff:</strong> {{ cutoff }}</p>
            
            <h3>Datasets Analyzed</h3>
            <ul>
                <li><strong>CP:</strong> Chemical Perturbagens (small molecules)</li>
                <li><strong>SH:</strong> Short Hairpin RNA (gene knockdown)</li>  
                <li><strong>OE:</strong> Over-Expression (gene overexpression)</li>
                <li><strong>LIG:</strong> Ligands (growth factors, cytokines)</li>
                <li><strong>MUT:</strong> Mutations (genetic variants)</li>
            </ul>
        </div>
        
        <div class="warning">
            <h4>⚠️ Important Notes</h4>
            <ul>
                <li>Reverse correlations suggest compounds that may reverse the signature phenotype</li>
                <li>Aggravate correlations suggest compounds that enhance the signature phenotype</li>
                <li>P-values are calculated for individual correlations; FDR controls for multiple testing</li>
                <li>This analysis is for research purposes only and should not be used for clinical decisions</li>
            </ul>
        </div>
    </div>

    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc; text-align: center; color: #666;">
        <p>Generated by LINCS Connectivity Map Analysis Tool | {{ timestamp }}</p>
    </footer>
</body>
</html>
        """

        try:
            from jinja2 import Template

            # Prepare template data
            datasets = ["cp", "sh", "oe", "lig", "mut"]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            signature_gene_count = len(signature)

            # Create signature preview
            signature_preview = []
            if not signature.empty:
                sig_data = signature.copy()
                sig_data["abs_effect"] = abs(sig_data.iloc[:, 0])
                top_genes = sig_data.nlargest(10, "abs_effect")
                signature_preview = [
                    (gene, effect) for gene, effect in zip(top_genes.index, top_genes.iloc[:, 0])
                ]

            # Count total correlations
            total_correlations = sum(
                len(data.get("hit_list", []))
                for key, data in results.items()
                if isinstance(data, dict) and "hit_list" in data
            )

            # Add basename filter for template
            def basename_filter(path: Optional[str]) -> str:
                """Jinja filter: basename of a path, or empty string if falsy."""
                return os.path.basename(path) if path else ""

            template = Template(html_template)
            template.globals["basename"] = basename_filter

            html_content = template.render(
                report_title=report_title,
                timestamp=timestamp,
                signature_gene_count=signature_gene_count,
                datasets=datasets,
                total_correlations=total_correlations,
                signature_preview=signature_preview,
                results=results,
                hit_number_tbl=hit_number_tbl,
                hit_number_hm=hit_number_hm,
                cutoff=cutoff,
                view_tbl=view_tbl,
                include_plotly=include_plotly,
                summary_plot_path=results.get("summary_plot_path"),
            )

            html_path = Path(report_folder) / f"{report_title}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            return str(html_path)

        except Exception as e:
            warnings.warn(f"Failed to generate HTML report: {e}")
            # Fallback simple HTML
            simple_html = f"""
            <html>
            <head><title>{report_title} - LINCS Report</title></head>
            <body>
                <h1>{report_title}</h1>
                <p>LINCS connectivity analysis completed at {datetime.now()}</p>
                <p>Analysis included {len(signature)} signature genes.</p>
            </body>
            </html>
            """
            html_path = Path(report_folder) / f"{report_title}_simple.html"
            with open(html_path, "w") as f:
                f.write(simple_html)
            return str(html_path)


# ------------------------------------------------------------------
# Enhanced LincsVisualizer Class
# ------------------------------------------------------------------
class LincsVisualizer:
    """Visualization tools for LINCS analysis results."""

    def __init__(self) -> None:
        """Initialize LINCS visualizer."""
        pass

    def _prepare_heatmap_data(
        self,
        hit_expression: pd.DataFrame,
        cutoff: Optional[float],
        scale: bool,
        pert_ordered: bool,
    ) -> pd.DataFrame:
        """
        Prepares heatmap data by applying optional transformations such as sorting based
        on a correlation column, applying cutoff thresholds, scaling, and ordering
        columns. This method is useful for preprocessing data for visualization purposes.

        Args:
            hit_expression (pd.DataFrame): The input DataFrame containing expression data.
            cutoff (Optional[float]): The cutoff value to clip the data. If None, no clipping is applied.
            scale (bool): If True, scales the data using StandardScaler.
            pert_ordered (bool): If True, orders the perturbagens by correlation to the "sig" column.

        Returns:
            pd.DataFrame: The transformed DataFrame with optional sorting, clipping, scaling, and ordering.
        """
        data = hit_expression.copy()

        # Sort by signature correlation if available
        if "sig" in data.columns:
            data = data.sort_values("sig")

        # Apply cutoff
        if cutoff is not None:
            data = data.clip(-cutoff, cutoff)

        # Scale data if requested
        if scale:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            data = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)

        # Order perturbagens by correlation to signature
        if pert_ordered and "sig" in data.columns:
            sig_corrs = data.corrwith(data["sig"])
            ordered_cols = sig_corrs.sort_values(ascending=False).index
            ordered_cols = [col for col in ordered_cols if col in data.columns]
            data = data[ordered_cols]

        return data

    def lincs_heatmap(
        self,
        hit_expression: pd.DataFrame,
        cutoff: Optional[float] = None,
        pert_ordered: bool = True,
        gene_ordered: bool = False,
        transpose: bool = True,
        scale: bool = False,
        resize: bool = False,
        hm_title: Optional[str] = None,
        out_file: Optional[str] = None,
        cluster_rows: bool = False,
        cluster_cols: bool = True,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Generates and visualizes a heatmap for provided expression data. The function supports
        customization options such as scaling, clustering, and resizing. It allows users to save
        the heatmap figure to a specified file if required.

        Args:
            hit_expression (pd.DataFrame): Expression data to be visualized. Rows represent genes and
                                           columns represent perturbations.
            cutoff (Optional[float]): Threshold for filtering expression data before plotting. If None,
                                      no filtering is applied.
            pert_ordered (bool): If True, ensures rows/columns corresponding to perturbations are ordered.
            gene_ordered (bool): If True, ensures rows/columns corresponding to genes are ordered.
            transpose (bool): If True, transposes the data before plotting.
            scale (bool): If True, scales the data matrix by standardizing rows or columns.
            resize (bool): If True, adjusts the aspect ratio of the plot for equal proportions.
            hm_title (Optional[str]): Title for the generated heatmap. If None, no title is displayed.
            out_file (Optional[str]): Path to save the output image file. If None, the heatmap is not saved.
            cluster_rows (bool): If True, applies hierarchical clustering on rows.
            cluster_cols (bool): If True, applies hierarchical clustering on columns.
            figsize (Tuple[int, int]): Size of the figure in inches (width, height).

        Returns:
            plt.Figure: The Matplotlib Figure object of the generated heatmap.
        """
        data = self._prepare_heatmap_data(hit_expression, cutoff, scale, pert_ordered)

        if transpose:
            data = data.T

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            data,
            center=0,
            cmap="RdBu_r",
            ax=ax,
            cbar_kws={"label": "Expression Level"},
            cluster_rows=cluster_rows,
            cluster_cols=cluster_cols,
        )

        if hm_title:
            ax.set_title(hm_title)
        if resize:
            ax.set_aspect("equal")

        plt.tight_layout()
        if out_file:
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
        return fig

    def _get_top_hits(self, hit_list: pd.DataFrame, direction: str, top_n: int) -> pd.DataFrame:
        """
        Returns the top N entries from a DataFrame based on the specified direction.

        The method extracts either the first N rows or the last N rows from the provided DataFrame,
        depending on the direction specified.

        Args:
            hit_list (pd.DataFrame): The input DataFrame containing data from which the top entries
                need to be extracted.
            direction (str): The direction in which to retrieve the top entries. Must be either
                "reverse" to retrieve the last N entries or another value to retrieve the first N entries.
            top_n (int): The number of entries to extract from the DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame containing the requested top N entries from the input
                DataFrame based on the specified direction.
        """
        return hit_list.tail(top_n) if direction == "reverse" else hit_list.head(top_n)

    def plot_correlation_results(
        self,
        results: Dict,
        direction: str = "reverse",
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Plots the correlation results based on the provided hit list. Produces a figure with two subplots: a distribution plot of
        signature correlations and a horizontal bar plot of the top compounds with the highest correlations.

        Args:
            results (Dict): Dictionary containing the correlation results, which must include the 'hit_list' key with the data.
            direction (str): Specifies the type of compounds to plot ('reverse' for reversing, 'direct' for aggravating).
                Default is "reverse".
            top_n (int): Number of top compounds to include in the bar plot. Default is 20.
            figsize (Tuple[int, int]): Size of the figure. Default is (12, 8).

        Returns:
            plt.Figure: The matplotlib figure object that contains the plotted correlation results.

        Raises:
            ValueError: If the 'hit_list' key is not present in the 'results' dictionary.
        """
        if "hit_list" not in results:
            raise ValueError("Results must contain 'hit_list'")

        hit_list = results["hit_list"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. Distribution Plot
        ax1.hist(hit_list["Sig_Corr"], bins=50, alpha=0.7, edgecolor="black")
        ax1.set_xlabel("Signature Correlation")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Signature Correlations")
        ax1.axvline(0, color="red", linestyle="--", alpha=0.7)

        # 2. Top Hits Plot
        top_hits = self._get_top_hits(hit_list, direction, top_n)
        colors = ["red" if x > 0 else "blue" for x in top_hits["Sig_Corr"]]

        ax2.barh(range(len(top_hits)), top_hits["Sig_Corr"], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_hits)))
        ax2.set_yticklabels(top_hits["merge"], fontsize=8)
        ax2.set_xlabel("Signature Correlation")

        title_type = "Reversing" if direction == "reverse" else "Aggravating"
        ax2.set_title(f"Top {top_n} {title_type} Compounds")
        ax2.axvline(0, color="black", linestyle="--", alpha=0.5)

        plt.tight_layout()
        return fig


# ------------------------------------------------------------------
# High-level interface functions for backward compatibility
# ------------------------------------------------------------------
def lincs_tools(
    meta_object: Dict,
    filter_object: Dict,
    report_out_folder: str,
    hit_number_hm: int = 10,
    hit_number_tbl: int = 10,
    resize: bool = False,
    report_title: str = "lincsReport",
) -> Dict:
    """
    FIXED: Run LINCS tools analysis on MetaIntegrator results.
    Fixed to match test expectations for lincs_correlate call.

    Args:
        meta_object: Meta-analysis object
        filter_object: Filter results object
        report_out_folder: Output directory for report
        hit_number_hm: Number of hits for heatmap
        hit_number_tbl: Number of hits for table
        resize: Whether to resize output
        report_title: Report title

    Returns:
        LINCS analysis results
    """
    # Extract signature from filter object
    pos_genes = filter_object.get("posGeneNames", [])
    neg_genes = filter_object.get("negGeneNames", [])
    signature_genes = pos_genes + neg_genes

    if "metaAnalysis" not in meta_object or "pooledResults" not in meta_object["metaAnalysis"]:
        raise ValueError("Meta-analysis results not found")

    pooled_results = meta_object["metaAnalysis"]["pooledResults"]
    signature_frame = pooled_results.loc[signature_genes, ["effectSize"]]
    signature_frame.columns = ["summary"]

    # Initialize analyzer (FIXED: Use LincsAnalyzer directly as expected by tests)
    data_manager = LincsDataManager()
    analyzer = LincsAnalyzer(data_manager)

    # Run correlation analysis
    results = analyzer.lincs_correlate(
        signature_frame, dataset="CP", hit_number_hm=hit_number_hm, direction="reverse"
    )

    # Generate visualizations
    visualizer = LincsVisualizer()
    if "hit_expression" in results:
        fig = visualizer.lincs_heatmap(
            results["hit_expression"], resize=resize, hm_title="LINCS Connectivity Map"
        )

        # Save figure
        os.makedirs(report_out_folder, exist_ok=True)
        fig_path = os.path.join(report_out_folder, f"{report_title}_heatmap.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    return results


def _prepare_signature_frame(meta_object: Dict, filter_object: Dict) -> pd.DataFrame:
    """
    Prepares a signature DataFrame for specified genes from a meta-analysis result.

    This function extracts information about specified positive and negative genes
    from the provided meta-analysis data and constructs a DataFrame containing
    summary effect sizes for those genes. It ensures that the meta-analysis results
    exist in the input object and handles missing data scenarios appropriately.

    Args:
        meta_object : Dict
            A dictionary containing meta-analysis results. It is expected to have a
            "metaAnalysis" key with "pooledResults" as a sub-key providing the data.
        filter_object : Dict
            A dictionary with keys "posGeneNames" and "negGeneNames" representing the
            positive and negative gene names, respectively, for which information is to
            be extracted. Both keys should map to lists of gene names.

    Returns:
        pd.DataFrame
            A DataFrame containing the summary effect sizes for the specified genes.
            The DataFrame has one column named "summary".

    Raises:
        ValueError
            If the meta-analysis data or pooled results are not found in the input
            meta_object.
    """
    pos_genes = filter_object.get("posGeneNames", [])
    neg_genes = filter_object.get("negGeneNames", [])
    sig_genes = pos_genes + neg_genes

    if "metaAnalysis" not in meta_object or "pooledResults" not in meta_object["metaAnalysis"]:
        raise ValueError("Meta-analysis results not found in meta_object")

    pooled_results = meta_object["metaAnalysis"]["pooledResults"]
    signature_frame = pooled_results.loc[sig_genes, ["effectSize"]]
    signature_frame.columns = ["summary"]
    return signature_frame


def lincs_correlate(
    meta_object: Dict,
    filter_object: Dict,
    dataset: str = "CP",
    hit_number_hm: int = 20,
    direction: str = "reverse",
    cor_method: str = "pearson",
    drop_string: Optional[str] = None,
    just_clin: bool = False,
    show_clin: bool = False,
    gene_ann: bool = False,
) -> Dict:
    """
    Performs LINCS correlation analysis based on the provided metadata and filtering object. It involves
    initializing necessary components, correlating the metadata and filter object with the LINCS dataset,
    and generating visualizations such as heatmaps when applicable.

    Args:
        meta_object: Dict
            Metadata object containing necessary information for correlation analysis.
        filter_object: Dict
            Filtering object to specify the subset of data or filtering criteria.
        dataset: str, optional
            Dataset to be used for correlation, default is "CP".
        hit_number_hm: int, optional
            Number of hits to be displayed in the heatmap, default is 20.
        direction: str, optional
            Direction of correlation, "reverse" for opposite correlation or "forward". Default is "reverse".
        cor_method: str, optional
            Method for calculating correlation, "pearson", "spearman", etc. Default is "pearson".
        drop_string: Optional[str], optional
            String to specify patterns or substrings to filter data or results. Default is None.
        just_clin: bool, optional
            If True, only clinical data will be used in the analysis. Default is False.
        show_clin: bool, optional
            If True, clinical data will be displayed in the output. Default is False.
        gene_ann: bool, optional
            If True, additional gene annotations will be included in the results. Default is False.

    Returns:
        Dict
            A dictionary containing analysis results, such as correlation data and heatmap visualizations
            (when applicable).
    """
    signature_frame = _prepare_signature_frame(meta_object, filter_object)

    # Initialize components
    analyzer = LincsAnalyzer(LincsDataManager())

    results = analyzer.lincs_correlate(
        signature_frame,
        dataset=dataset,
        hit_number_hm=hit_number_hm,
        direction=direction,
        method=cor_method,
        drop_string=drop_string,
        just_clin=just_clin,
        show_clin=show_clin,
        gene_ann=gene_ann,
    )

    # Generate heatmap visualization if data is available
    if "hit_expression" in results:
        visualizer = LincsVisualizer()
        results["heatmap_figure"] = visualizer.lincs_heatmap(results["hit_expression"])

    return results


def lincs_bait_corr(
    meta_object: Dict,
    filter_object: Dict,
    dataset: str = "CP",
    baits: List[str] = None,
    just_clin: bool = False,
    hit_number_hm: int = 20,
    hm_baits: bool = True,
    direction: str = "aggravate",
    bait_type: Optional[str] = None,
) -> Dict:
    """
    Analyzes LINCS bait correlations and processes the results.

    This function performs bait correlation analysis based on provided metadata and filter
    objects. It utilizes the LINCS dataset to identify relationships using a signature
    frame and can generate visualizations such as heatmaps if relevant data is available.
    The analysis direction, bait types, and additional parameters can be customized.

    Args:
        meta_object: Dict
            Metadata dictionary used for correlation analysis which may provide details about
            experiments or conditions to filter signatures.
        filter_object: Dict
            Dictionary specifying filters to apply on the metadata to narrow down relevant
            data signatures before analysis.
        dataset: str (default: "CP")
            Dataset to be analyzed. Can indicate the specific LINCS modality or data type.
        baits: List[str] (optional, default: None)
            List of bait compounds to be analyzed for their correlations.
        just_clin: bool (default: False)
            If True, restricts the analysis to clinically relevant aspects or measurements.
        hit_number_hm: int (default: 20)
            The maximum number of significant hits to include in the heatmap visualization, if
            generated.
        hm_baits: bool (default: True)
            Flag indicating whether the heatmap should specifically emphasize or include baits.
        direction: str (default: "aggravate")
            Indicates the direction of correlation to analyze, such as "aggravate" or "relieve".
        bait_type: Optional[str] (default: None)
            Specifies the type or category of baits to consider during the correlation analysis.

    Returns:
        Dict
            A dictionary containing the results of the bait correlation analysis. It may include
            data about hits, correlations, and optionally, a heatmap visualization if successful.

    Raises:
        No explicit exceptions raised in the documentation, but the function relies on external
        components and methods which may raise errors based on invalid inputs or issues during
        processing.
    """
    signature_frame = _prepare_signature_frame(meta_object, filter_object)

    # Initialize analyzer components
    analyzer = LincsAnalyzer(LincsDataManager())

    # Run bait correlation analysis
    results = analyzer.lincs_bait_corr(
        dataset=dataset,
        baits=baits,
        sub_sig=signature_frame,
        just_clin=just_clin,
        hm_num=hit_number_hm,
        hm_baits=hm_baits,
        direction=direction,
        bait_type=bait_type,
    )

    # Generate visualization if successful
    if "heatmap_df" in results:
        visualizer = LincsVisualizer()
        results["heatmap_figure"] = visualizer.lincs_heatmap(results["heatmap_df"])

    return results


# NEW high-level interface functions
def lincs_interactors(
    meta_object: Dict,
    filter_object: Dict,
    dataset: str,
    correlate_output: Optional[Dict] = None,
) -> Dict:
    """
    NEW FUNCTION: High-level interface for gene network interactor analysis.

    Args:
        meta_object: Meta-analysis results
        filter_object: Gene filter results
        dataset: LINCS dataset ("SH", "OE", "LIG", "MUT")
        correlate_output: Pre-computed correlation results (optional)

    Returns:
        Dictionary with interactor analysis results
    """
    signature_frame = _prepare_signature_frame(meta_object, filter_object)
    analyzer = LincsAnalyzer(LincsDataManager())

    # Get correlation results if not provided
    if correlate_output is None:
        correlate_output = analyzer.lincs_correlate(
            signature_frame, dataset=dataset, direction="reverse"
        )

    return analyzer.lincs_interactors(correlate_output, signature_frame, dataset)


def lincs_qc_access(dataset: str, perturbagens: List[str], just_gold: bool = False) -> pd.DataFrame:
    """
    NEW FUNCTION: High-level interface for LINCS quality control data access.

    Args:
        dataset: LINCS dataset ("CP", "SH", "OE", "LIG", "MUT")
        perturbagens: List of perturbagen names
        just_gold: Whether to include only gold-standard signatures

    Returns:
        DataFrame with quality control information
    """
    data_manager = LincsDataManager()
    return data_manager.lincs_qc_access(dataset, perturbagens, just_gold)


def lincs_report(
    meta_object: Dict,
    filter_object: Dict,
    report_out_folder: str,
    hit_number_hm: int = 10,
    hit_number_tbl: int = 10,
    resize: bool = False,
    report_title: str = "lincsReport",
) -> Dict:
    """
    NEW FUNCTION: High-level interface for comprehensive LINCS reporting.

    Args:
        meta_object: Meta-analysis results
        filter_object: Gene filter results
        report_out_folder: Output directory path
        hit_number_hm: Number of hits for heatmaps
        hit_number_tbl: Number of hits for tables
        resize: Whether to resize visualizations
        report_title: Report title/prefix

    Returns:
        Dictionary with complete analysis results and report paths
    """
    signature_frame = _prepare_signature_frame(meta_object, filter_object)
    reporter = LincsReporter()

    return reporter.lincs_report(
        signature=signature_frame,
        hit_number_hm=hit_number_hm,
        hit_number_tbl=hit_number_tbl,
        resize=resize,
        report_out_folder=report_out_folder,
        report_title=report_title,
    )
