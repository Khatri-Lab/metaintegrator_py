import warnings
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


class BayesParams(TypedDict):
    """Type definition for Bayesian parameters used in ComBat normalization."""

    B_hat: NDArray[float]
    stand_mean: NDArray[float]
    var_pooled: NDArray[float]
    gamma_star: NDArray[float]
    delta_star: NDArray[float]


# Use functional TypedDict syntax to handle reserved keyword 'class'
GeneExpressionData = TypedDict(
    "GeneExpressionData",
    {"genes": pd.DataFrame, "pheno": pd.DataFrame, "class_labels": NDArray[float]},
)


class CombatOutput(TypedDict):
    """Type definition for ComBat normalization output."""

    bayesdata: pd.DataFrame
    bayes_params: BayesParams


class CombatControlOutput(TypedDict):
    """Type definition for ComBat control phase output."""

    GSEs: List[GeneExpressionData]
    bayes_params: BayesParams
    get_platforms: NDArray[np.str_]


class CoconutOutput(TypedDict):
    """Type definition for COCONUT algorithm output."""

    COCONUTList: List[GeneExpressionData]
    rawDiseaseList: List[GeneExpressionData]
    controlList: CombatControlOutput


class CombinedCoconutOutput(TypedDict):
    """Type definition for combined COCONUT output."""

    genes: pd.DataFrame
    pheno: pd.DataFrame
    class_cntl0_dis1: NDArray[float]


CONTROL_LABEL = 0


def _get_common_genes(datasets: List[Dict[str, pd.DataFrame]]) -> List[str]:
    """
    Identify the set of gene identifiers that are present in every dataset provided.

    Each dataset is expected to be a mapping that contains a "genes" key. The gene identifiers are taken
    from the frame’s index. Only non‑empty, non‑null, and non‑blank identifiers are considered.

    Args:
        datasets : list[dict[str, pd.DataFrame]]
            A list of datasets, each represented as a dictionary that may contain a "genes" key.
            Index of the associated DataFrame value holds the gene identifiers. Datasets that do not contain a
            "genes" entry or whose gene frame is empty are ignored.

    Returns:
        list[str]
            A sorted list of gene identifiers that appear in all of the supplied datasets.
            If the input list is empty or if no genes are shared, an empty list is returned.
    """
    if not datasets:
        return []

    gene_sets = []
    for ds in datasets:
        if "genes" in ds and not ds["genes"].empty:
            genes = ds["genes"].index
            valid_genes = genes[pd.notnull(genes) & (genes.astype(str).str.strip() != "")]
            gene_sets.append(set(valid_genes))

    if not gene_sets:
        return []

    # 20251224orig
    """
    # Find intersection of all gene lists
    common = set(gene_indices[0])
    for genes in gene_indices[1:]:
        common = common.intersection(set(genes))
    """

    # Find intersection of all gene lists
    common = reduce(lambda x, y: x.intersection(y), gene_sets)
    return sorted(list(common))


def _get_data_subset(
    datasets: List[Dict[str, pd.DataFrame]],
    gene_list: List[str],
    filter_col: str,
    filter_val: Optional[str] = None,
    is_control: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract subset of data based on control/disease classification from a list of GSE‑style dictionaries.
        - "genes" – rows are genes, columns are samples.
        - "pheno" – one row per sample with phenotype information.

    Args:
        datasets : list[dict[str, pandas.DataFrame]]
            A collection of data dictionaries, each with keys "genes" and "pheno". The function operates on
            all non‑empty entries.
        gene_list : list[str]
            List of gene identifiers to keep in the returned subset.
        filter_col : str
            Name of the column in pheno that contains the sample classification. Commonly this is "cell_type" or "group".
        filter_val : str, optional
            When is_control is False and filter_val is provided, this column is expected to contain the
            disease‑specific label (e.g. "disease"). If omitted, all non‑control samples are used.
        is_control : bool, default True
            If True the function keeps samples where pheno[filter_col] == CONTROL_LABEL.  If False it keeps either all
            non‑control samples or, if filter_val is supplied, samples where pheno[filter_val] is not null.

    Returns:
        list[dict[str, Any]]
            One dictionary per input dataset containing:
            - "genes" : pandas.DataFrame – sub‑matrix of genes (rows) × selected samples (columns).
            - "pheno" : pandas.DataFrame – phenotype data for the selected samples.
            - "class_labels" : numpy.ndarray – numeric class labels for each sample. For controls
                the label is 0; for disease samples the label comes from the disease column (if present) or is inferred
                as 1 by default.

    Raises:
        ValueError
            - If a dataset does not contain the required "genes" or "pheno" keys.
            - If filter_col is missing from the phenotype data.
            - If filter_val is supplied but the corresponding column does not exist.
    """
    subsets = []
    for gse in datasets:
        # Validate input data structure
        if "genes" not in gse or "pheno" not in gse:
            raise ValueError("Each dataset must contain 'genes' and 'pheno' DataFrames")

        genes_df, pheno_df = gse["genes"], gse["pheno"]
        if genes_df.empty or pheno_df.empty:
            continue

        # Check if filter column exists
        if filter_col not in pheno_df.columns:
            raise ValueError(f"Filter column '{filter_col}' not found in phenotype data")

        # Get sample indices based on control/disease status
        if is_control:
            idx = pheno_df[pheno_df[filter_col] == CONTROL_LABEL].index
            class_vals = np.zeros(len(idx))
        else:
            if filter_val is None:
                idx = pheno_df[pheno_df[filter_col] != CONTROL_LABEL].index
                class_vals = pheno_df.loc[idx, filter_col].values
            else:
                if filter_val not in pheno_df.columns:
                    raise ValueError(f"Disease column '{filter_val}' not found")
                idx = pheno_df[pheno_df[filter_val].notnull()].index
                class_vals = pheno_df.loc[idx, filter_val].values

        # Filter genes to only include common genes and available samples
        available_samples = [s for s in idx if s in genes_df.columns]
        if not available_samples:
            continue

        # Extract subset of data
        subset_genes = genes_df.loc[gene_list, available_samples].copy()
        subsets.append(
            {
                "pheno": pheno_df.loc[available_samples].copy(),
                "genes": subset_genes,
                "class_labels": class_vals[: len(available_samples)].astype(float),
            }
        )
    return subsets


def _validate_config(
    full_gses: List[Dict[str, pd.DataFrame]],
    control_gses: List[Dict[str, Any]],
    by_platform: bool,
    platform_col: Optional[str],
) -> None:
    """
     Validate that the supplied GSE configurations are internally consistent.

     The function checks two aspects of the configuration:
     1. Platform consistency – When by_platform is True, it ensures that every platform represented in the full GSEs
        also appears in the control GSEs, and that the user supplied a column name that can be used to identify the
        platform in the phenotype DataFrame.
     2. Control sample presence – When by_platform is False, it ensures that each control GSE contains at least one
        sample column. An empty control set is treated as an error because it would lead to undefined posterior
        estimates later in the workflow.

    Args:
         full_gses : list[dict[str, pandas.DataFrame]]
             List of GSE dictionaries that constitute the full dataset. Each dictionary must contain a "genes" and
             a "pheno" DataFrame.
         control_gses : list[dict[str, Any]]
             List of GSE dictionaries that provide the control samples. The same structure as full_gses is expected.
         by_platform : bool
             Flag indicating whether platform‑specific checks should be performed. If True, every platform found in
             full_gses must also be present in control_gses. If False, the function simply verifies that each control
             GSE contains at least one sample column.
         platform_col : str | None
             The column (or a case‑insensitive substring of a column) that stores the platform identifier in the
             phenotype DataFrame. Required when by_platform=True.

     Raises:
         ValueError
             - If by_platform is True but platform_col is None.
             - If the set of platform identifiers extracted from full_gses differs from the set extracted from
               control_gses. This typically indicates that some platforms lack corresponding control data.
             - If any control GSE contains zero sample columns (i.e., an empty "genes" DataFrame).

     Notes:
         - Platform extraction is performed in a case‑insensitive manner; the function looks for the substring
           platform_col.lower() in each phenotype column name.
         - The function returns None and performs its work solely through side‑effects (raising exceptions on failure).

    """

    if by_platform:
        if platform_col is None:
            raise ValueError("platform_col must be provided if by_platform=True")

        def get_plat_list(gse_list: List[Dict[str, Any]]) -> List[str]:
            """Return the sorted, deduplicated platform identifiers found in gse_list."""
            plats = set()
            for ds in gse_list:
                cols = [c for c in ds["pheno"].columns if platform_col.lower() in c.lower()]
                if cols:
                    plats.add(str(ds["pheno"][cols[0]].iloc[0]))
            return sorted(list(plats))

        if get_plat_list(full_gses) != get_plat_list(control_gses):
            raise ValueError("by_platform=True but some platforms lack controls")
    else:
        for i, ds in enumerate(control_gses):
            if ds["genes"].shape[1] < 1:
                raise ValueError(f"Dataset at index {i} has < 1 control")


def coconut(
    gses: List[Dict[str, pd.DataFrame]],
    control_0_col: str,
    disease_col: Optional[str] = None,
    by_platform: bool = False,
    platform_col: Optional[str] = None,
    par_prior: bool = True,
    it_conv: float = 1e-04,
    parallel: bool = False,
    mc_cores: int = 1,
) -> Dict[str, Any]:
    """
    COmbat CONormalization Using conTrols (COCONUT)

    A method used in bioinformatics to remove "batch effects" (technical noise) when combining gene expression
    datasets from different sources or platforms.

    The core philosophy of COCONUT is that it calculates normalization parameters only using control samples and then
    applies those same parameters to the diseased/experimental samples. This prevents the normalization process from
    accidentally removing the biological signal you are trying to study.

    Args:
        gses: List of gene expression datasets, each containing 'genes' and 'pheno' DataFrames
        control_0_col: Column name indicating control samples (value 0 = control)
        disease_col: Column name for disease samples. If None, uses non-control samples
        by_platform: Whether to treat platforms as separate batches
        platform_col: Column name containing platform information (required if by_platform=True)
        par_prior: Whether to use parametric priors in Bayesian adjustment
        it_conv: Convergence threshold for iterative solution
        parallel: Whether to use parallel processing
        mc_cores: Number of cores for parallel processing

    Returns:
        Dictionary containing:
            - COCONUTList: Normalized disease samples
            - rawDiseaseList: Original unnormalized disease samples
            - controlList: Normalized control samples and parameters

    Raises:
        ValueError: If platform_col is required but not provided, or if datasets
                   have insufficient control samples
    """

    # Main COCONUT algorithm steps
    # 1. Gene alignment
    common_genes = _get_common_genes(gses)
    if not common_genes:
        raise ValueError("No common genes found across datasets")

    # 2. Isolate control samples
    control_datasets = _get_data_subset(gses, common_genes, control_0_col, is_control=True)

    # 3. Validation
    _validate_config(gses, control_datasets, by_platform, platform_col)

    # 4. Calculate normalization parameters (the "Get" phase)
    combat_control = _combat_custom(
        control_datasets,
        common_genes,
        params="get",
        by_platform=by_platform,
        platform_col=platform_col,
        par_prior=par_prior,
        it_conv=it_conv,
        parallel=parallel,
        mc_cores=mc_cores,
    )

    # 5. Isolate disease samples
    disease_datasets = _get_data_subset(
        gses, common_genes, control_0_col, filter_val=disease_col, is_control=False
    )

    # 6. Apply normalization (the "Have" phase)
    disease_datasets_combat = _combat_custom(
        disease_datasets,
        common_genes,
        params="have",
        by_platform=by_platform,
        platform_col=platform_col,
        bayes_params=combat_control["bayes_params"],
        get_platforms=combat_control["get_platforms"],
    )

    return {
        "COCONUTList": disease_datasets_combat,
        "rawDiseaseList": disease_datasets,
        "controlList": combat_control,
    }


def combine_coco_output(coconut_out: CoconutOutput) -> CombinedCoconutOutput:
    """
    Merge COCONUT-normalized data into a single consolidated object.
    This function takes the output from the COCONUT algorithm and combines all
    the individual datasets into unified matrices for downstream analysis.
    Args:
        coconut_out: Output dictionary from the coconut() function
    Returns:
        Dictionary containing:
            - genes: Combined gene expression matrix (all studies)
            - pheno: Combined phenotype metadata (common columns only)
            - class_cntl0_dis1: Combined classification vector (0=control, 1=disease)
    """
    # 1. Data extraction
    disease_list = coconut_out["COCONUTList"]
    control_list = coconut_out["controlList"]["GSEs"]

    # 2. Intra-dataset merging
    processed_datasets = []
    for control_data, disease_data in zip(control_list, disease_list):
        processed_datasets.append(_process_dataset_pair(control_data, disease_data))

    # 3. Global aggregation
    genes_mat = pd.concat([ds["genes"] for ds in processed_datasets], axis=1)

    # Identify pheno columns common to all combined entries
    pheno_common_cols = reduce(
        pd.Index.intersection, [ds["pheno"].columns for ds in processed_datasets]
    )
    pheno_mat = pd.concat([ds["pheno"][pheno_common_cols] for ds in processed_datasets], axis=0)

    combined_labels = np.concatenate([ds["class_labels"] for ds in processed_datasets])

    return {
        "genes": genes_mat,
        "pheno": pheno_mat,
        "class_cntl0_dis1": combined_labels.astype(float),
    }


def _process_dataset_pair(control_data: dict, disease_data: dict) -> dict:
    """
    Internal helper to merge control and disease components of a single dataset.

    The function expects each input dictionary to contain at least the keys "genes" (a pandas.DataFrame with
    genes in rows and samples in columns) and "pheno" (a pandas.DataFrame whose index corresponds to the sample
    identifiers of the associated gene matrix).  It concatenates the two gene matrices column‑wise, aligns
    the phenotypic information row‑wise by retaining only columns that are common to both  phenotypic tables, and
    builds a binary class label vector indicating whether each sample is a control (0) or a disease (1).

    Args:
        control_data : dict
            Dictionary containing the control‑group data.  It must provide control_data["genes"] (genes × samples)
            and control_data["pheno"] (sample × phenotype attributes).
        disease_data : dict
            Dictionary containing the disease‑group data, with the same structure as control_data. The two
            dictionaries may have different numbers of samples and different phenotype columns; the latter are
            intersected to keep only the shared attributes.

    Returns:
        dict
            A dictionary with three entries:
            "genes"   – the concatenated gene expression matrix
                            (shape: n_genes × (n_control + n_disease)),
            "pheno"   – the aligned phenotype table
                            (shape: (n_control + n_disease) × n_common_cols),
            "class_labels" – a numpy.ndarray of floats
                            (dtype float) of length
                            n_control + n_disease containing 0.0 for control
                            samples and 1.0 for disease samples.

    """

    # Combine genes (column-wise concatenation)
    # Gene identifiers (row indices) are assumed to be identical and aligned between the two input matrices.
    genes = pd.concat([control_data["genes"], disease_data["genes"]], axis=1)

    # Combine pheno (row-wise concatenation) - only common columns are kept
    common_cols = control_data["pheno"].columns.intersection(disease_data["pheno"].columns)
    pheno = pd.concat(
        [control_data["pheno"][common_cols], disease_data["pheno"][common_cols]],
        axis=0,
    )

    # Generate group labels (0 for control, 1 for disease) based on sample counts
    control_count = control_data["genes"].shape[1]
    disease_count = disease_data["genes"].shape[1]

    # The "class_labels" vector is explicitly cast to float to facilitate usage in libraries that expect continuous
    # class vectors (e.g., scikit‑learn, statsmodels).
    group_labels = np.concatenate([np.zeros(control_count), np.ones(disease_count)]).astype(float)

    return {"genes": genes, "pheno": pheno, "class_labels": group_labels}


def _prepare_batch_labels(
    gse_list: List[GeneExpressionData],
    combined_data: pd.DataFrame,
    by_platform: bool,
    platform_col: Optional[str],
) -> NDArray[np.str_]:
    """
    Generate batch labels and perform quantile normalization if by_platform is True.

    Args:
        gse_list: List of gene expression datasets
        combined_data: Combined gene expression matrix (modified in-place if by_platform=True)
        by_platform: Whether to treat platforms as separate batches
        platform_col: Column name containing platform information

    Returns:
        Array of batch labels for each sample
    """
    batch_labels = []

    if by_platform:
        if platform_col is None:
            raise ValueError("platform_col must be provided when by_platform=True")

        print("\nTreating platforms as batches...")
        for gse_data in gse_list:
            platform_val = _extract_platform_value(gse_data, platform_col)
            print(f"{platform_val}, ", end="")
            batch_labels.extend([platform_val] * gse_data["genes"].shape[1])

        batch_labels_array = np.array(batch_labels, dtype=str)
        _apply_quantile_normalization(combined_data, batch_labels_array)
    else:
        print("\nTreating datasets as batches: ")
        for i, gse_data in enumerate(gse_list):
            batch_labels.extend([str(i)] * gse_data["genes"].shape[1])
        batch_labels_array = np.array(batch_labels, dtype=str)

    return batch_labels_array


def _extract_platform_value(gse_data: GeneExpressionData, platform_col: str) -> str:
    """
    Return the platform identifier for a GSE from its phenotype data.

    The function searches the phenotype (pheno) DataFrame for a column whose name contains the supplied
    platform_col substring (case‑insensitive). It then extracts the value in the first row of that platform column
    and returns it as a plain Python string.

    Args:
        gse_data : GeneExpressionData
            A mapping containing at least a "pheno" DataFrame, typically obtained from a GEO download. The "pheno"
            DataFrame should have one row per sample and include a column that records the sequencing or microarray
            platform.
        platform_col : str
            Substring to match against the column names in the phenotype DataFrame.  The match is performed
            case‑insensitively; any column containing this substring is considered a candidate.

    Returns:
        The str value found in the first row of the matched platform column.

    Raises:
        ValueError
            - If no column name contains platform_col (ignoring case).
            - If the matched column exists but the first row is missing (e.g., the DataFrame is empty).
    """
    cols = [c for c in gse_data["pheno"].columns if platform_col.lower() in c.lower()]
    if not cols:
        raise ValueError(f"Platform column '{platform_col}' not found in dataset")
    # Only the first matching column is used – if several columns contain the substring, the one that appears
    # first in the DataFrame is returned.
    return str(gse_data["pheno"][cols[0]].iloc[0])


def _apply_quantile_normalization(
    combined_data: pd.DataFrame, batch_labels: NDArray[np.str_]
) -> None:
    """
    Perform quantile normalization within each identified platform batch.

    Quantile normalization forces the distribution of expression values to  be identical across all samples in the
    same batch. This implementation follows the classical approach: the values are ranked per sample, the mean of
    each rank is computed across the batch, and every value is replaced by the mean of its rank. The operation is
    performed in place on combined_data.

    Args:
        combined_data : pandas.DataFrame
            Gene‑expression matrix with genes as rows and samples as columns. The DataFrame is modified in place; no
            new object is returned.
        batch_labels : numpy.ndarray (dtype=np.str_)
            1‑D array whose length equals the number of columns in combined_data. Each element is a string identifying
            the platform (or batch) that the corresponding column belongs to. The function uses this array to group
            columns before normalizing.

    Returns:
        None; the input DataFrame is altered in place.
    """
    unique_platforms = np.unique(batch_labels)
    for plat in unique_platforms:
        idx = batch_labels == plat
        plat_data = combined_data.iloc[:, idx]

        # Calculate rank means across samples in the batch
        # Ranking uses method="first" so that ties receive the same rank (the first occurrence).
        rank_mean = (
            plat_data.stack().groupby(plat_data.rank(method="first").stack().astype(int)).mean()
        )

        # Map original ranks to the calculated means
        qnormed = plat_data.rank(method="first").stack().astype(int).map(rank_mean).unstack()
        combined_data.iloc[:, idx] = qnormed


def _split_combat_data(
    gse_list: List[GeneExpressionData], combat_data_matrix: pd.DataFrame
) -> List[GeneExpressionData]:
    """
    Split the combined normalized matrix back into the original GSE list format.

    Args:
        gse_list: Original list of gene expression datasets
        combat_data_matrix: Combined normalized gene expression matrix

    Returns:
        List of datasets with normalized gene expression data
    """

    def _update_gse_genes(gse: GeneExpressionData) -> GeneExpressionData:
        """Return a copy of gse with its "genes" columns replaced from combat_data_matrix."""
        new_gse = gse.copy()
        # Extract the relevant columns from the combined matrix based on the original structure
        new_gse["genes"] = combat_data_matrix[gse["genes"].columns].copy()
        return new_gse

    return [_update_gse_genes(gse) for gse in gse_list]


def _combat_custom(
    gse_list: List[GeneExpressionData],
    common: List[str],
    params: str,
    by_platform: bool,
    platform_col: Optional[str],
    bayes_params: Optional[BayesParams] = None,
    get_platforms: Optional[NDArray[np.str_]] = None,
    par_prior: bool = True,
    it_conv: float = 1e-04,
    parallel: bool = False,
    mc_cores: int = 1,
) -> Union[CombatControlOutput, List[GeneExpressionData]]:
    """
    Custom ComBat implementation with control-based philosophy.

    The "Control-Based" philosophy: You "get" the noise signature from healthy samples
    and then assume the "have" samples (disease) share that same technical noise signature,
    allowing you to subtract it without losing the disease-specific biological signal.

    Args:
        gse_list: List of gene expression datasets
        common: List of common genes across datasets
        params: Either "get" (calculate parameters) or "have" (apply parameters)
        by_platform: Whether to treat platforms as separate batches
        platform_col: Column name containing platform information
        bayes_params: Pre-calculated Bayesian parameters (required for "have" mode)
        get_platforms: Platform names from "get" phase (required for "have" mode)
        par_prior: Whether to use parametric priors
        it_conv: Convergence threshold
        parallel: Whether to use parallel processing
        mc_cores: Number of cores for parallel processing

    Returns:
        For "get" mode: Dictionary with normalized data and calculated parameters
        For "have" mode: List of normalized datasets

    Raises:
        ValueError: If params is not "get" or "have", or if required parameters are missing
    """
    if params not in ["get", "have"]:
        raise ValueError("params must be 'get' or 'have'")

    combined_data = pd.concat([x["genes"] for x in gse_list], axis=1)
    batches = _prepare_batch_labels(gse_list, combined_data, by_platform, platform_col)

    if params == "get":
        return _handle_combat_get_phase(
            gse_list, combined_data, batches, par_prior, it_conv, parallel, mc_cores
        )

    return _handle_combat_have_phase(gse_list, combined_data, batches, bayes_params, get_platforms)


def _handle_combat_get_phase(
    gse_list: List[GeneExpressionData],
    combined_data: pd.DataFrame,
    batches: NDArray[np.str_],
    par_prior: bool,
    it_conv: float,
    parallel: bool,
    mc_cores: int,
) -> CombatControlOutput:
    """
    Helper to process the 'get' phase of ComBat custom.

    Args:
        gse_list : list[GeneExpressionData]
            The original list of GSE dictionaries that were merged into combined_data. Each element must contain at
            least a "genes" and a "pheno" DataFrame.
        combined_data : pandas.DataFrame
            Gene‑expression matrix (genes × samples) that is fed directly to ComBat. The DataFrame is not modified by
            this function.
        batches : numpy.ndarray (dtype=np.str_)
            1‑D array of length equal to combined_data.shape[1] that indicates the batch (usually a platform) belonging
            to each sample. Values are case‑insensitive strings.
        par_prior : bool
            Whether to use a parameter‑wise prior for the batch variance estimates (True) or a global prior (False).
        it_conv : float
            The convergence tolerance for the iterative EM algorithm.
        parallel : bool
            A boolean flag indicating whether to execute the internal EM updates in parallel using multiple CPU cores.
        mc_cores : int
            The number of CPU cores to allocate for parallel execution of the EM updates.

    Returns:
        CombatControlOutput
            A mapping with three keys:
            - "GSEs" – the original GSE list with their "bayesdata" replaced by the batch‑parameter‑corrected data
              returned by _split_combat_data.
            - "bayes_params" – the dictionary of estimated bayes parameters returned by _combat_get_params_no_cov
              (e.g., mu, sigma and batch‐specific estimates).
            - "get_platforms" – a sorted numpy.ndarray of the unique batch identifiers encountered.
    """
    combat_out = _combat_get_params_no_cov(
        combined_data,
        batch=batches,
        par_prior=par_prior,
        it_conv=it_conv,
        parallel=parallel,
        mc_cores=mc_cores,
    )

    return {
        "GSEs": _split_combat_data(gse_list, combat_out["bayesdata"]),
        "bayes_params": combat_out["bayes_params"],
        "get_platforms": np.unique(batches),
    }


def _handle_combat_have_phase(
    gse_list: List[GeneExpressionData],
    combined_data: pd.DataFrame,
    batches: NDArray[np.str_],
    bayes_params: Optional[BayesParams],
    get_platforms: Optional[NDArray[np.str_]],
) -> List[GeneExpressionData]:
    """
    Apply previously estimated ComBat parameters to new data (the have phase).

    In a two‑stage ComBat workflow the get phase estimates the batch‑specific parameters from a reference set of
    samples (often controls). The have phase then uses those parameters to correct a second set of samples.
    This helper verifies that the batch ordering in the new data matches the reference ordering, applies
    the parameters, and splits the corrected matrix back into the original GSE objects.

    Args:
        gse_list : List[GeneExpressionData]
            The original list of GSE dictionaries that were merged into combined_data. Each element must contain at
            least a "genes" and a "pheno" DataFrame.
        combined_data : pandas.DataFrame
            Gene‑expression matrix (genes × samples) for the new set of samples to be corrected. Columns represent
            samples, rows represent genes.
        batches : NDArray[np.str_]
            1‑D array whose length matches the number of columns in combined_data. Each entry is a string identifying
            the batch (e.g., platform) to which the corresponding column belongs.
        bayes_params : Optional[BayesParams]
            The dictionary of Bayesian parameters that was produced by the get phase. Must contain the keys required by
            _combat_apply_params_no_cov (e.g., mu, sigma, and batch‑specific estimates). If None a ValueError
            is raised.
        get_platforms : Optional[NDArray[np.str_]]
            The array of platform identifiers that was returned by the get phase. It is used to check that the batch
            ordering in the new data matches the reference ordering. If None a ValueError is raised.

    Returns:
        List[GeneExpressionData]
            The same GSE list that was passed in, but with each dictionary's "bayesdata" attribute replaced by the
            ComBat‑corrected data produced by _combat_apply_params_no_cov.  The function calls _split_combat_data
            to map the corrected matrix back to the individual GSE objects.

    Raises:
        ValueError
            - If either bayes_params or get_platforms is None.
            - If the batch order in batches differs from that in get_platforms (i.e., the reference and new datasets
              were not processed in the same order).
    """
    if bayes_params is None or get_platforms is None:
        raise ValueError("bayes_params and get_platforms are required for 'have' mode")

    current_platforms = np.unique(batches)
    if not np.array_equal(current_platforms, get_platforms):
        raise ValueError("Batches not in identical order between controls and cases.")

    print("\nBatches identical between have-params and get-params...")
    combat_data = _combat_apply_params_no_cov(
        combined_data, batch=batches, bayes_params=bayes_params
    )

    return _split_combat_data(gse_list, combat_data)


def _calculate_stats(d_hat: NDArray[float]) -> Tuple[float, float]:
    """
    Compute the mean and variance for prior hyperparameter estimation.

    The function expects a one‑dimensional NumPy array of float values and returns the mean and the sample variance
    (i.e., variance with ddof=1).

    Args:
        d_hat : numpy.ndarray
            1‑D array of hyper‑parameter estimates. The array must contain at least one element; an empty array
            raises a ValueError.

    Returns:
        tuple[float, float]
            (mean, variance) where mean is the arithmetic mean of d_hat and variance is the unbiased sample variance.

    Raises
        ValueError
            If d_hat is empty.
    """
    if len(d_hat) == 0:
        raise ValueError("d_hat cannot be empty")
    mean_val = np.mean(d_hat)
    variance_val = np.var(d_hat, ddof=1)
    return float(mean_val), float(variance_val)


def get_a_prior(d_hat: NDArray[float]) -> float:
    """
    Compute the 'a' hyper‑parameter of the inverse gamma prior on delta, the batch variance.

    Args
        d_hat : numpy.ndarray
            1‑D array of empirical variance estimates for the batches. Must contain at least one element; an empty
            array triggers a ValueError.

    Returns:
        float
            The estimated 'a' hyperparameter value. If all variances in d_hat are identical (i.e., variance of the
            sample is zero) the function returns float("inf") which signals that the prior variance should be treated
            as infinite (effectively no shrinkage).

    Raises:
        ValueError
            If d_hat is empty.

    """
    m, s2 = _calculate_stats(d_hat)
    if s2 == 0:
        return float("inf")  # Handle zero variance case
    return (2 * s2 + m**2) / s2


def get_b_prior(d_hat: NDArray[float]) -> float:
    """
    Compute the 'b' hyperparameter for the inverse gamma prior on delta.

    Args:
        d_hat : numpy.ndarray
            1‑D array of estimated variances

    Returns:
        float
            The estimated scale parameter b.  If all elements of d_hat are identical (i.e. sample variance is zero),
            the function returns float("inf") which indicates that the prior variance should be treated as infinite
            (effectively no shrinkage).

    Raises:
        ValueError
            If d_hat is empty.
    """
    m, s2 = _calculate_stats(d_hat)
    if s2 == 0:
        return float("inf")  # Handle zero variance case
    return (m * s2 + m**3) / s2


def _combat_get_params_no_cov(
    dat: pd.DataFrame,
    batch: NDArray[np.str_],
    par_prior: bool = True,
    it_conv: float = 1e-04,
    parallel: bool = False,
    mc_cores: int = 1,
) -> CombatOutput:
    """
    Calculate ComBat normalization parameters without covariates.

    This function implements the "get" phase of ComBat, which estimates the batch effect parameters from the input data.

    Args:
        dat: Gene expression data matrix (genes x samples)
        batch: Array of batch labels for each sample
        par_prior: Whether to use parametric priors in Bayesian adjustment
        it_conv: Convergence threshold for iterative solution
        parallel: Whether to use parallel processing
        mc_cores: Number of cores for parallel processing

    Returns:
        Dictionary containing normalized data and calculated parameters
    """
    # 1. Initialization and Validation
    data_values = dat.values
    batch_series = pd.Series(batch)
    classes = batch_series.unique()
    n_batches_count = len(classes)

    if n_batches_count < 2:
        raise ValueError(
            f"At least 2 batches are required for ComBat normalization. {n_batches_count} batch(es) found."
        )

    design = pd.get_dummies(batch_series).values.astype(float)
    batches_idx = [np.where(batch_series == c)[0] for c in classes]
    n_samples_per_batch = np.array([len(idx) for idx in batches_idx])

    if any(n == 0 for n in n_samples_per_batch):
        raise ValueError("Some batches are empty")

    # 2. Standardization
    print("Standardizing Data across genes")
    b_hat, grand_mean, var_pooled, s_data = _standardize_data(
        data_values, design, n_samples_per_batch, n_batches_count
    )

    # 3. Estimating batch effects (L/S Model)
    print("Fitting L/S model and finding priors")
    gamma_hat, delta_hat = _fit_ls_model(s_data, design[:, :n_batches_count], batches_idx)

    # 4. Empirical Bayes adjustments
    print(f"Finding {'parametric' if par_prior else 'nonparametric'} adjustments")
    gamma_star, delta_star = _estimate_bayesian_params(
        s_data, batches_idx, gamma_hat, delta_hat, par_prior, it_conv
    )

    # 5. Adjusting the data
    print("Adjusting the Data")
    stand_mean = np.outer(grand_mean, np.ones(sum(n_samples_per_batch)))
    bayesdata = _adjust_data_final(
        s_data, batches_idx, gamma_star, delta_star, var_pooled, stand_mean
    )

    return {
        "bayesdata": pd.DataFrame(bayesdata, index=dat.index, columns=dat.columns),
        "bayes_params": {
            "B_hat": b_hat,
            "stand_mean": stand_mean,
            "var_pooled": var_pooled,
            "gamma_star": gamma_star,
            "delta_star": delta_star,
        },
    }


def _standardize_data(
    data: np.ndarray, design: np.ndarray, n_batches: np.ndarray, n_batch_count: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize expression data by removing batch effects.

    The function follows the two‑stage standardization procedure used in the ComBat algorithm:

    Args:
        data : ndarray, shape (n_genes, n_samples)
            The raw gene‑expression matrix to be standardized. Rows correspond to genes, columns to samples.
        design : ndarray, shape (n_samples, n_design)
            Design matrix describing the experimental factors (e.g. batch indicators, covariates). It must have
            full column rank.
        n_batches : ndarray, shape (n_batches,)
            Vector of sample counts per batch.  Length must match the number of batch columns in design.
        n_batch_count : int
            Number of batch columns in design.  This value is used to slice the coefficient matrix b_hat when
            computing the grand mean.

    Returns:
        b_hat : ndarray, shape (n_design, n_genes)
            Estimated coefficients for each design factor (batch effects first, followed by any covariate effects).
        grand_mean : ndarray, shape (n_batches, 1)
            Weighted grand mean of the batch effects; one value per batch.
        var_pooled : ndarray, shape (n_genes, 1)
            Pooled variance estimate for each gene (lower bounded by 1e‑8).
        s_data : ndarray, shape (n_genes, n_samples)
            Standardized expression matrix where each gene has mean zero (after batch correction) and unit variance.

    Raises:
        ValueError
            If the design matrix is singular and the OLS solve fails.

    """

    # 1. Batch‑effect coefficient (b_hat) estimation
    #    An ordinary least squares (OLS) fit is performed to estimate the batch‑specific effects (b_hat).
    #    If the design matrix is "singular" (non-invertible), ValueError will be raised.
    try:
        b_hat = np.linalg.solve(design.T @ design, design.T @ data.T)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Singular matrix encountered when solving for batch effects. Verify batch assignments."
        )

    # 2. Calculate grand mean
    # n_array is the total number of samples.
    n_array = sum(n_batches)
    # grand_mean calculates a weighted average of the batch effects. It extracts the relevant coefficients from b_hat
    # (up to n_batch_count) and weights them by the size of each batch to find a global baseline.
    grand_mean = (n_batches / n_array) @ b_hat[:n_batch_count, :]

    # 3. Estimate pooled variance
    #    The residual variance after removing the batch effects is calculated; a lower bound of 1e‑8 is a safety
    #    measure (clamping) to ensure that the variance is never zero and
    var_pooled = np.maximum(np.mean((data - (design @ b_hat).T) ** 2, axis=1, keepdims=True), 1e-8)

    # 4. Standardize data
    #    stand_mean broadcasts the grand mean across all samples so it matches the shape of the data
    stand_mean = np.outer(grand_mean, np.ones(n_array))
    # The data are centered by the grand mean and scaled by the pooled standard deviation, yielding the standardized
    #        matrix s_data.
    s_data = (data - stand_mean) / np.sqrt(var_pooled)
    return b_hat, grand_mean, var_pooled, s_data


def _fit_ls_model(
    s_data: np.ndarray, batch_design: np.ndarray, batches_idx: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a simple linear‐model to estimate batch effects on standardized data.

    The function implements the first stage of the ComBat algorithm where an ordinary least squares (OLS) regression
    is run on the standardized expression matrix s_data. The regression predicts the data from the batch indicator
    design matrix batch_design and returns the batch‐specific coefficients (gamma_hat) and the empirical
    within‑batch variances (delta_hat).

    Args:
        s_data : ndarray, shape (n_genes, n_samples)
            Standardized expression matrix produced by _standardize_data. Each column corresponds to a sample
            and each row to a gene.
        batch_design : ndarray, shape (n_samples, n_batches)
            Design matrix that contains a column for each batch. It should be a binary indicator matrix (0/1) and
            must have full rank.
        batches_idx : list[ndarray]
            A list where each element contains the sample indices belonging to one batch. The order of the indices
            must match the order of the batch columns in batch_design.

    Returns:
        gamma_hat : ndarray, shape (n_batches, n_genes)
            Batch‑specific coefficient estimates. The first dimension indexes batches, the second dimension indexes
            genes.
        delta_hat : ndarray, shape (n_batches, n_genes)
            Empirical within‑batch variance estimate for each gene. The variance is computed with ddof=1 and a lower
            bound of 1e‑8 is applied for numerical safety.

    Raises:
        ValueError
            If batch_design is singular, the regression cannot be solved.
    """

    # Calculate the Ordinary Least Squares (OLS) estimates for the batch effects
    # If batch_design.T @ batch_design is singular (non-invertible), it means the system cannot be solved.
    # This usually happens if there are empty batches or if the batch variables are perfectly collinear.
    try:
        # np.linalg.solve is more numerically stable than explicitly calculating the inverse of the matrix
        gamma_hat = np.linalg.solve(batch_design.T @ batch_design, batch_design.T @ s_data.T)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Singular matrix encountered during L/S model fitting. Check for empty batches."
        )

    # Calculate the variance of the data within each batch.
    delta_hat = np.array(
        [np.maximum(np.var(s_data[:, idx], axis=1, ddof=1), 1e-8) for idx in batches_idx]
    )
    return gamma_hat, delta_hat


def _estimate_bayesian_params(
    s_data: np.ndarray,
    batches_idx: List[np.ndarray],
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    par_prior: bool,
    it_conv: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the Bayesian batch‐effect parameters for each batch.

    In the second stage of ComBat the raw OLS estimates gamma_hat and within‑batch variances delta_hat are shrunk
    toward a global prior using either a parameter‑wise prior (when par_prior is True) or an empirical prior
    (when par_prior is False). The resulting shrinked estimates are returned as gamma_star and delta_star.

    Args:
        s_data : ndarray, shape (n_genes, n_samples)
            Standardized expression matrix produced by _standardize_data.  Rows are genes, columns are samples.
        batches_idx : list[ndarray]
            List of integer arrays.  batches_idx[i] contains the column indices that belong to batch i.
        gamma_hat : ndarray, shape (n_batches, n_genes)
            Batch‑specific OLS coefficient estimates. The first dimension indexes batches, the second dimension indexes
            genes.
        delta_hat : ndarray, shape (n_batches, n_genes)
            Gene‑wise within‑batch variance estimates obtained from _fit_ls_model.
        par_prior : bool
            If True a parameter‑wise inverse‑Gamma prior is used (parameters a_prior and b_prior are estimated from
            delta_hat).  If False the empirical prior that uses only the batch‑specific data is employed.
        it_conv : float
            Convergence threshold for the iterative solver used by _it_sol.  The iteration stops when the change in
            parameters is smaller than this value.

    Returns:
        gamma_star : ndarray, shape (n_batches, n_genes)
            Shrinked batch‑effect coefficient estimates for each gene.
        delta_star : ndarray, shape (n_batches, n_genes)
            Shrinked within‑batch variance estimates for each gene.

    """
    # gamma_bar and t2 are the global mean and variance of the OLS batch effects and are used to build the
    # parameter‑wise prior.
    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.maximum(np.var(gamma_hat, axis=1, ddof=1), 1e-8)

    # a_prior and b_prior are computed from the empirical within‑batch variances delta_hat using get_a_prior
    # and get_b_prior.
    a_prior = np.apply_along_axis(get_a_prior, 1, delta_hat)
    b_prior = np.apply_along_axis(get_b_prior, 1, delta_hat)

    gamma_star, delta_star = [], []
    for i, idx in enumerate(batches_idx):
        batch_data = s_data[:, idx]
        if par_prior:
            # _it_sol performs an iterative Bayesian update until convergence
            res = _it_sol(
                batch_data,
                gamma_hat[i, :],
                delta_hat[i, :],
                gamma_bar[i],
                t2[i],
                a_prior[i],
                b_prior[i],
                convergence_threshold=it_conv,
            )
        else:
            # when par_prior is False, the function int_eprior performs a simple empirical‑prior shrinkage
            res = _int_eprior(batch_data, gamma_hat[i, :], delta_hat[i, :])
        gamma_star.append(res[0])
        delta_star.append(res[1])

    #  the returned arrays have shape (n_batches, n_genes) – one row per batch and one column per gene.
    return np.array(gamma_star), np.array(delta_star)


def _adjust_data_final(
    s_data: np.ndarray,
    batches_idx: List[np.ndarray],
    gamma_star: np.ndarray,
    delta_star: np.ndarray,
    var_pooled: np.ndarray,
    stand_mean: np.ndarray,
) -> np.ndarray:
    """
    Apply the final ComBat data adjustment and revert to the original scale.

    Args:
        s_data : ndarray, shape (n_genes, n_samples)
            Standardized expression matrix from the first stage of ComBat (centered by the grand mean and scaled by
            the pooled variance). Rows are genes, columns are samples.
        batches_idx : list[ndarray]
            List where batches_idx[i] contains the column indices that belong to batch i. The order corresponds to the
            rows of gamma_star and delta_star.
        gamma_star : ndarray, shape (n_batches, n_genes)
            Shrunk batch‑effect coefficient estimates from _estimate_bayesian_params.
        delta_star : ndarray, shape (n_batches, n_genes)
            Shrunk within‑batch variance estimates (variance, not standard deviation) from _estimate_bayesian_params.
        var_pooled : ndarray, shape (n_genes, 1)
            Pooled variance estimate computed during the first stage (used to restore the original gene‑wise variance
            after correction). A lower bound of 1e‑8 is applied for numerical safety.
        stand_mean : ndarray, shape (n_genes, 1)
            The grand‑mean value that was subtracted during the standardization step (stand_mean in _standardize_data).
            will be added back to restore the original gene‑wise mean after batch correction.

    Returns:
        bayesdata : ndarray, shape (n_genes, n_samples)
            The ComBat‑adjusted expression matrix in the original scale of the data. Each batch has been corrected
            for its estimated mean shift (gamma_star) and variance scaling (delta_star), and then multiplied by the
            pooled standard deviation and shifted by the original mean.

    """
    bayesdata = s_data.copy()
    # The loop iterates over each batch; for each gene a gene‑specific shift and scale are applied to all samples in
    # the  batch. The shift g_star and scale d_star are broadcasted across genes using [:, None].
    for j, idx in enumerate(batches_idx):
        g_star = gamma_star[j, :][:, None]
        # A small constant 1e‑8 protects against division by zero or taking the square root of negative values.
        d_star = np.sqrt(np.maximum(delta_star[j, :], 1e-8))[:, None]
        bayesdata[:, idx] = (bayesdata[:, idx] - g_star) / d_star

    # The final rescaling uses var_pooled and stand_mean to convert the data back to the scale of the original
    # (non‑standardized) expression matrix.
    return (bayesdata * np.sqrt(var_pooled)) + stand_mean


def _combat_apply_params_no_cov(
    dat: pd.DataFrame, batch: NDArray[np.str_], bayes_params: BayesParams
) -> pd.DataFrame:
    """
    Apply pre-calculated batch correction parameters to new data.

    This function takes pre-calculated batch correction parameters (usually derived
    from a control group) and applies them to a new dataset (usually the
    disease/experimental group).

    Args:
        dat: Gene expression data matrix to be corrected
        batch: Array of batch labels for each sample
        bayes_params: Pre-calculated Bayesian parameters from control samples

    Returns:
        Batch-corrected gene expression data matrix
    """
    # 1. Initialization and parameters
    data_values = dat.values
    batch_series = pd.Series(batch)
    unique_batches = batch_series.unique()
    batch_indices = [np.where(batch_series == c)[0] for c in unique_batches]

    gamma_star = bayes_params["gamma_star"]
    delta_star = bayes_params["delta_star"]
    stand_mean_ref = bayes_params["stand_mean"][:, 0][:, None]  # Shape: (genes, 1)
    var_pooled = bayes_params["var_pooled"]
    std_dev = np.sqrt(var_pooled)  # Shape: (genes, 1)

    # 2. Standardization
    # (Data - Reference Mean) / Pooled StdDev
    s_data = (data_values - stand_mean_ref) / std_dev

    # 3. Applying empirical bayes adjustments
    print("Adjusting the Data Based on input gamma.star and delta.star")
    bayesdata = s_data.copy()

    for j, idx in enumerate(batch_indices):
        if j >= len(gamma_star) or j >= len(delta_star):
            raise ValueError(f"Batch index {j} exceeds available parameters")

        # Extract parameters for specific batch and reshape for broadcasting (genes, 1)
        g_star_batch = gamma_star[j, :][:, None]
        d_star_batch = np.sqrt(np.maximum(delta_star[j, :], 1e-8))[:, None]

        # Apply the shift and scale: (Standardized - Gamma) / Delta
        bayesdata[:, idx] = (bayesdata[:, idx] - g_star_batch) / d_star_batch

    # 4. Rescaling and return
    # Rescale back to original space: (Adjusted * StdDev) + Mean
    rescaled_data = (bayesdata * std_dev) + stand_mean_ref

    return pd.DataFrame(rescaled_data, index=dat.index, columns=dat.columns)


# ------------------------------------------------------------------
# Constants for the Parametric Empirical Bayes (PEB) solver
# ------------------------------------------------------------------
PARAMETRIC_BAYES_MAX_ITERATIONS = 1000
"""Maximum number of iterations allowed in the iterative PEB solver."""

PARAMETRIC_BAYES_EPSILON = 1e-10
"""Tiny value used to avoid division by zero in the posterior‐gamma update."""

PARAMETRIC_BAYES_MIN_VARIANCE = 1e-8
"""Minimum variance value to be used as a lower bound for the scale parameter (delta)."""


# ------------------------------------------------------------------
# Iterative solver for the parametric Empirical Bayes adjustment
# ------------------------------------------------------------------
def _it_sol(
    standardized_data: NDArray[float],
    initial_gamma: NDArray[float],
    initial_delta: NDArray[float],
    prior_gamma_mean: float,
    prior_gamma_var: float,
    prior_delta_shape: float,
    prior_delta_scale: float,
    convergence_threshold: float = 1e-04,
) -> Tuple[NDArray[float], NDArray[float]]:
    """
    Iterative solver for parametric empirical Bayes estimation.

    This function implements an iterative algorithm to estimate the posterior distributions of batch effect parameters
    (gamma and delta) using empirical Bayes methods with parametric priors.

    The algorithm alternates between updating the location parameters (gamma) and scale parameters (delta) until
    convergence is achieved.

    Args:
        standardized_data : ndarray, shape (n_genes, n_samples)
            The standardised gene‑expression matrix from the first ComBat stage (genes × samples). Each gene has been
            centred by the global grand‑mean and scaled by the pooled variance.
        initial_gamma : ndarray, shape (n_genes,)
            Initial OLS estimates of the batch‐effect location parameters. Typically these are the batch means for
            the current batch.
        initial_delta : ndarray, shape (n_genes,)
            Initial OLS estimates of the batch‐effect scale parameters.  These are the gene‑wise within‑batch
            variances for the current batch.
        prior_gamma_mean : float
            Prior mean for the gamma parameters – the global mean of the batch effects across all genes.
        prior_gamma_var : float
            Prior variance for the gamma parameters – the global variance of the batch effects across all genes.
        prior_delta_shape : float
            Shape parameter of the inverse‑Gamma prior on the delta parameters. It is derived from the empirical
            deltâ values via get_a_prior.
        prior_delta_scale : float
            Scale parameter of the inverse‑Gamma prior on the delta parameters. It is derived from the empirical
            delta values via get_b_prior̂
        convergence_threshold : float, default=1e-4
            Convergence criterion: the algorithm stops when the relative change of both gamma and delta across
            iterations falls below this threshold.

    Returns:
        Tuple[ndarray, ndarray]
            (gamma, delta) – the posterior (shrunk) estimates for the batch‐effect location and scale parameters,
            each array having shape (n_genes,). If the solver reaches the maximum number of iterations without
            meeting the convergence threshold, a RuntimeWarning is emitted and the last updated values are returned.

    Raises:
        RuntimeWarning
            Emitted when the algorithm does not converge after PARAMETRIC_BAYES_MAX_ITERATIONS iterations. The warning
            contains the final relative change.
    """
    # Input validation
    _validate_inputs(
        standardized_data,
        initial_gamma,
        initial_delta,
        prior_gamma_var,
        prior_delta_shape,
        prior_delta_scale,
        convergence_threshold,
    )

    num_genes, num_samples = standardized_data.shape

    # Handle edge cases
    if num_samples == 0:
        return initial_gamma.copy(), initial_delta.copy()

    gamma_old, delta_old = initial_gamma.copy(), initial_delta.copy()
    change = 1.0
    iteration = 0

    while change > convergence_threshold and iteration < PARAMETRIC_BAYES_MAX_ITERATIONS:
        iteration += 1

        # Posterior mean for gamma (location parameter)
        # Combines prior (g_bar) and data (g_hat) weighted by their precisions
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = prior_gamma_var * num_samples + delta_old
            gamma_new = np.where(
                denominator > PARAMETRIC_BAYES_EPSILON,
                (prior_gamma_var * num_samples * initial_gamma + delta_old * prior_gamma_mean)
                / denominator,
                initial_gamma,
            )

        # Posterior estimate for delta (scale parameter)
        # Sum of squared residuals from updated location estimates
        residuals = standardized_data - gamma_new.reshape(-1, 1)
        sum_sq_residuals = np.sum(residuals**2, axis=1)

        denominator_delta = num_samples / 2.0 + prior_delta_shape - 1.0
        if denominator_delta <= 0:
            delta_new = delta_old.copy()
        else:
            delta_new = np.maximum(
                (0.5 * sum_sq_residuals + prior_delta_scale) / denominator_delta,
                PARAMETRIC_BAYES_MIN_VARIANCE,
            )

        # _calculate_convergence returns the maximum relative change among all gamma and delta components that is
        # used as the stopping criterion.
        change = _calculate_convergence(
            gamma_old, gamma_new, delta_old, delta_new, convergence_threshold
        )

        gamma_old, delta_old = gamma_new.copy(), delta_new.copy()

    if iteration >= PARAMETRIC_BAYES_MAX_ITERATIONS:
        warnings.warn(
            f"Iterative solver did not converge after {PARAMETRIC_BAYES_MAX_ITERATIONS} iterations. Final change: {change:.6f}",
            RuntimeWarning,
        )
    return gamma_old, delta_old


def _validate_inputs(
    sdat: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
    t2: float,
    a: float,
    b: float,
    conv: float,
) -> None:
    """
    Perform sanity checks on the inputs to the parametric EB solver.

    Args:
        sdat : ndarray
            Standardized gene‑expression matrix supplied to the solver. Must be two‑dimensional (genes × samples).
        g_hat : array‑like
            Initial estimate of the batch‐effect location parameters (gamma_hat).  Should have one entry per gene.
        d_hat : array‑like
            Initial estimate of the batch‐effect scale parameters (delta hat).  Should have one entry per gene.
        t2 : float
            Prior variance for gamma (t²).  Must be strictly positive.
        a : float
            Shape parameter of the inverse‑Gamma prior on delta (αlpha). Must be strictly positive.
        b : float
            Scale parameter of the inverse‑Gamma prior on delta (beta).  Must be strictly positive.
        conv : float
            Convergence threshold supplied to the iterative solver. Must be greater than zero.

    Raises:
        ValueError
            - If sdat is not a 2‑D array.
            - If the lengths of g_hat or d_hat do not equal the number of genes (rows of sdat).
            - If any of the scalar hyper‑parameters t2, a, or b are non‑positive.
            - If the convergence threshold conv is not a positive value.
    """

    if sdat.ndim != 2:
        raise ValueError("sdat must be a 2D array")
    if len(g_hat) != sdat.shape[0] or len(d_hat) != sdat.shape[0]:
        raise ValueError("g_hat and d_hat must match number of genes in sdat")
    if conv <= 0:
        raise ValueError("Convergence threshold must be positive")
    if t2 <= 0 or a <= 0 or b <= 0:
        raise ValueError("Prior parameters (t2, a, b) must be positive")


def _calculate_convergence(
    g_old: np.ndarray, g_new: np.ndarray, d_old: np.ndarray, d_new: np.ndarray, threshold: float
) -> float:
    """
    Compute a single scalar convergence metric for the iterative PEB solver.

    The parametric Empirical Bayes solver iteratively updates the batch‑effect parameters gamma and delta. At each
    iteration we need to decide whether the updates have become sufficiently small.This helper returns the
    maximum relative change observed across all genes for both gamma and delta.

    Args:
        g_old : ndarray, shape (n_genes,)
            The gamma values from the previous iteration.
        g_new : ndarray, shape (n_genes,)
            The gamma values from the current iteration.
        d_old : ndarray, shape (n_genes,)
            The delta values from the previous iteration.
        d_new : ndarray, shape (n_genes,)
            The delta values from the current iteration.
        threshold : float
            The value to return if the computed change is not finite (e.g., due to division by zero). In practice
            this will be the same value used as the convergence threshold in the solver.

    Returns
        float
            The largest relative change observed among all genes for either gamma or delta. If the computed value
            is not finite, threshold is returned instead.

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # Relative changes are calculated element‑wise
        g_diff = np.abs(g_new - g_old)
        d_diff = np.abs(d_new - d_old)

        g_rel_change = np.where(
            np.abs(g_old) > PARAMETRIC_BAYES_EPSILON, g_diff / np.abs(g_old), g_diff
        )
        d_rel_change = np.where(
            np.abs(d_old) > PARAMETRIC_BAYES_EPSILON, d_diff / np.abs(d_old), d_diff
        )

        total_change = max(np.max(g_rel_change), np.max(d_rel_change))
    return total_change if np.isfinite(total_change) else threshold


def _int_eprior(
    sdat: NDArray[float], g_hat: NDArray[float], d_hat: NDArray[float]
) -> Tuple[NDArray[float], NDArray[float]]:
    """
     Compute non-parametric empirical Bayes estimation for a single batch.

     This function implements a non-parametric approach to empirical Bayes estimation where the prior distribution is
     estimated from the data itself. For each gene, it computes a leave‑one‑out weighted average of the other genes'
     OLS estimates (g_hat, d_hat). The resulting shrunk location and scale parameters are returned as g_star and d_star.

     Args:
         sdat : ndarray, shape (n_genes, n_samples)
             Standardized gene‑expression matrix for the current batch.
         g_hat : ndarray, shape (n_genes,)
             Initial estimates of location parameters (batch means).  One entry per gene.
         d_hat : ndarray, shape (n_genes,)
             Initial estimates of scale parameters (batch variances). One entry per gene.

     Returns:
         Tuple[ndarray, ndarray]
            - g_star: Updated location parameters based on empirical priors
            - d_star: Updated scale parameters based on empirical priors

    Raises:
        ValueError: If input arrays have incompatible shapes
    """
    # Input validation
    if sdat.ndim != 2:
        print(f"\nInput array has incompatible shape: {sdat.shape}")
        raise ValueError("Input gene expression data matrix, sdata, must be a 2D array")

    num_genes, num_samples = sdat.shape
    if len(g_hat) != num_genes or len(d_hat) != num_genes:
        raise ValueError(
            "Input batch means, g_hat, and batch variance, d_hat, must match number of genes in sdat"
        )

    # Handle edge cases
    # If the data contain only one gene or no samples, the function simply returns the input estimates unchanged.
    if num_genes <= 1 or num_samples == 0:
        return g_hat.copy(), d_hat.copy()

    g_star = np.zeros(num_genes, dtype=float)
    d_star = np.zeros(num_genes, dtype=float)

    # For each gene i, compute weighted estimates based on all other genes
    for i in range(num_genes):
        # Get parameter estimates from all other genes (leave-one-out)
        other_indices = np.arange(num_genes) != i
        # _compute_gene_weighted_estimates implements the leave‑one‑out weighting scheme.
        g_star[i], d_star[i] = _compute_gene_weighted_estimates(
            sdat[i, :],
            g_hat[other_indices],
            d_hat[other_indices],
            num_samples,
            g_hat[i],
            d_hat[i],
        )

    # Final validation and clipping
    # value for variable nan will be used to replace each nan
    g_star = np.nan_to_num(g_star, nan=0.0, posinf=1e6, neginf=-1e6)
    d_star = np.nan_to_num(d_star, nan=1e-8, posinf=1e6, neginf=1e-8)
    return g_star, np.maximum(d_star, 1e-8)


def _compute_gene_weighted_estimates(
    x_i: NDArray[float],
    g_others: NDArray[float],
    d_others: NDArray[float],
    n: int,
    g_fallback: float,
    d_fallback: float,
) -> Tuple[float, float]:
    """
    Compute weighted location and scale estimates for a single gene using a leave‑one‑out empirical prior.

    For the i‑th gene, it treats the other genes in the batch as a reference population. Each reference gene j
    supplies a location estimate g_others[j] and a variance estimate d_others[j]. The posterior probability
    that gene i was generated from gene j's distribution is proportional to the likelihood of the observed
    expression values under that distribution. These likelihoods are used as weights to compute a weighted mean
    of the reference parameters. The returned values represent the shrunk estimates that borrow strength from the
    remaining genes.

    Args:
        x_i : ndarray, shape (n_samples,)
            The expression values of the target gene across all samples in the current batch.
        g_others : ndarray, shape (n_genes - 1,)
            Location (mean) estimates of all other genes in the batch.
        d_others : ndarray, shape (n_genes - 1,)
            Scale (variance) estimates of all other genes in the batch.
        n : int
            Number of samples in the current batch (len(x_i)).
        g_fallback : float
            Value to return for the location estimate if no reliable weighted estimate can be computed.
        d_fallback : float
            Value to return for the scale estimate if no reliable weighted estimate can be computed.

    Returns
        Tuple[float, float]
            (g_out, d_out) – the weighted location and variance estimates for gene i.  d_out is guaranteed to be at
            least 1e-8 to avoid downstream numerical problems.
    """
    min_variance = 1e-10
    min_likelihood_sum = 1e-15

    # Single‑gene case*
    if len(g_others) == 0:
        return g_fallback, d_fallback

    # Small‑variance guard to avoid division by zero
    d_others_safe = np.maximum(d_others, min_variance)

    # Calculate log-likelihood for numerical stability
    # L = (2π*d)^(-n/2) * exp(-Σ(x-g)²/(2*d))
    residuals_squared_sum = np.sum((x_i.reshape(1, -1) - g_others.reshape(-1, 1)) ** 2, axis=1)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_likelihood = (-n / 2.0 * np.log(2 * np.pi * d_others_safe)) - (
            residuals_squared_sum / (2.0 * d_others_safe)
        )

        # Shift log-likelihoods for numerical stability before exponentiating
        max_log = (
            np.max(log_likelihood[np.isfinite(log_likelihood)])
            if np.any(np.isfinite(log_likelihood))
            else 0.0
        )
        likelihood = np.exp(log_likelihood - max_log)
        likelihood = np.nan_to_num(likelihood, nan=0.0, posinf=0.0, neginf=0.0)

    sum_likelihood = np.sum(likelihood)
    # Likelihood‑sum guard
    if sum_likelihood <= min_likelihood_sum:
        return g_fallback, d_fallback

    weights = likelihood / sum_likelihood
    g_out = np.sum(g_others * weights)
    d_out = np.sum(d_others * weights)

    return g_out, max(d_out, 1e-8)
