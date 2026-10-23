"""
Sample-level gene data extraction functions for MetaIntegrator Python port.
This module implements the R functions from getSampleLevelGeneData.R for extracting
and processing sample-level gene expression data from dataset objects.
"""

import logging
import warnings
from typing import Dict, List, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleLevelGeneDataProcessor:
    """
    A class for processing sample-level gene expression data.

    This class provides methods to extract, clean, and summarize gene expression data
    at the sample level from dataset objects, mirroring the functionality of the R
    getSampleLevelGeneData functions.
    """

    @staticmethod
    def get_sample_level_gene_data(dataset_object: Dict, gene_names: List[str]) -> pd.DataFrame:
        """
        Extract gene-level data from a given dataset object.

        This is the main exported function that extracts sample-level data for specified genes.
        It processes probe-level data and summarizes it to gene-level data using mean aggregation.

        Args:
            dataset_object (Dict): A dataset object containing at least:
                - 'expr': DataFrame of probe-level expression data (probes x samples)
                - 'keys': List/Series of probe-to-gene mappings
            gene_names (List[str]): List of gene names to extract data for

        Returns:
            pd.DataFrame: DataFrame with gene-level expression data (genes x samples)

        Raises:
            ValueError: If required keys are missing from dataset_object
            TypeError: If inputs are not of expected types

        Examples:
            >>> dataset = {
            ...     'expr': pd.DataFrame(...),  # probe expression data
            ...     'keys': ['GENE1', 'GENE2,GENE3', ...]  # probe-gene mappings
            ... }
            >>> genes = ['GENE1', 'GENE2', 'GENE3']
            >>> result = SampleLevelGeneDataProcessor.get_sample_level_gene_data(dataset, genes)
        """
        # Input validation
        SampleLevelGeneDataProcessor._validate_dataset_object(dataset_object)
        SampleLevelGeneDataProcessor._validate_gene_names(gene_names)

        logger.info(
            f"Processing {len(gene_names)} genes from dataset with {dataset_object['expr'].shape[0]} probes"
        )

        # Extract data from Gene Expression Matrix
        genes_matrix = SampleLevelGeneDataProcessor._extract_data_from_gem(
            dataset_object, gene_names
        )

        # Clean the data using the replacement functions
        genes_matrix = SampleLevelGeneDataProcessor._replace_values(
            genes_matrix, threshold_value=0, replace_value=1
        )
        genes_matrix = SampleLevelGeneDataProcessor._replace_nans(genes_matrix, replace_value=1)
        genes_matrix = SampleLevelGeneDataProcessor._replace_nas(genes_matrix, replace_value=1)

        logger.info(f"Successfully processed gene data. Result shape: {genes_matrix.shape}")
        return genes_matrix

    @staticmethod
    def _validate_dataset_object(dataset_object: Dict) -> None:
        """Validate that dataset_object has required structure."""
        if not isinstance(dataset_object, dict):
            raise TypeError("dataset_object must be a dictionary")

        required_keys = ["expr", "keys"]
        missing_keys = [key for key in required_keys if key not in dataset_object]
        if missing_keys:
            raise ValueError(f"dataset_object missing required keys: {missing_keys}")

        if not isinstance(dataset_object["expr"], pd.DataFrame):
            raise TypeError("dataset_object['expr'] must be a pandas DataFrame")

        # Check if expression matrix is empty
        if dataset_object["expr"].empty:
            raise ValueError("Expression matrix cannot be empty")

        if len(dataset_object["keys"]) != len(dataset_object["expr"]):
            raise ValueError("Length of keys must match number of rows in expression matrix")

    @staticmethod
    def _validate_gene_names(gene_names: List[str]) -> None:
        """Validate gene_names input."""
        if not isinstance(gene_names, (list, np.ndarray, pd.Series)):
            raise TypeError("gene_names must be a list, numpy array, or pandas Series")

        if len(gene_names) == 0:
            raise ValueError("gene_names cannot be empty")

    @staticmethod
    def _extract_data_from_gem(
        dataset_object: Dict, gene_names: List[str], keys_sep: str = ","
    ) -> pd.DataFrame:
        """
        Core data extraction from Gene Expression Matrix (GEM).

        This function expands expression data for multi-gene probe mappings,
        groups data by gene keys, and summarizes multiple probes per gene using means.

        Args:
            dataset_object (Dict): Dataset containing 'expr' and 'keys'
            gene_names (List[str]): Target gene names to extract
            keys_sep (str): Separator for multiple gene mappings (default: ",")

        Returns:
            pd.DataFrame: Gene-level expression data with genes as rows and samples as columns
        """
        # Create expanded DataFrame with probe-gene mappings
        expanded_df = SampleLevelGeneDataProcessor._expand_df_sample_level(
            dataset_object["expr"], dataset_object["keys"], keys_sep=keys_sep
        )

        # Filter to only include target genes
        unique_gene_names = list(set(gene_names))  # Remove duplicates
        expanded_df = expanded_df[expanded_df["keys"].isin(unique_gene_names)]

        if len(expanded_df) == 0:
            warnings.warn(f"No matching genes found for {gene_names}")
            # Return empty DataFrame with correct structure. dtype=float is
            # required here (matching the "some genes missing" fill-in a
            # few lines below): without it, an index/columns-only
            # DataFrame with no data defaults to dtype=object, which then
            # survives the fillna(1) calls in get_sample_level_gene_data
            # as Python `int` objects rather than floats. That silently
            # produces an object-dtype gene matrix downstream, and the
            # first thing that calls np.log() on it (e.g. a geometric-mean
            # score) fails with "loop of ufunc does not support argument 0
            # of type int which has no callable log method" instead of
            # this simply evaluating to log(1)=0 as intended.
            sample_cols = dataset_object["expr"].columns
            return pd.DataFrame(index=unique_gene_names, columns=sample_cols, dtype=float)

        # Group by gene and summarize
        gene_groups = expanded_df.groupby("keys")
        summarized_results = {}

        for gene_name, group_data in gene_groups:
            summarized_gene = SampleLevelGeneDataProcessor._summarize_split(group_data)
            summarized_results[gene_name] = summarized_gene

        # Convert to matrix format
        gene_expression_matrix = pd.DataFrame.from_dict(summarized_results, orient="index")

        # Ensure all requested genes are included (fill missing with NaN)
        missing_genes = set(unique_gene_names) - set(gene_expression_matrix.index)
        if missing_genes:
            missing_df = pd.DataFrame(
                index=list(missing_genes),
                columns=gene_expression_matrix.columns,
                dtype=float,
            )
            gene_expression_matrix = pd.concat([gene_expression_matrix, missing_df])

        # Reorder to match input gene order (maintaining duplicates if present)
        ordered_genes = []
        for gene in gene_names:
            if gene in gene_expression_matrix.index:
                ordered_genes.append(gene)

        if ordered_genes:
            gene_expression_matrix = gene_expression_matrix.loc[ordered_genes]

        # Clean column names (equivalent to make.names in R)
        gene_expression_matrix.columns = SampleLevelGeneDataProcessor._make_valid_names(
            gene_expression_matrix.columns
        )

        return gene_expression_matrix

    @staticmethod
    def _expand_df_sample_level(
        df: pd.DataFrame, keys: Union[List, pd.Series, np.ndarray], keys_sep: str = ","
    ) -> pd.DataFrame:
        """
        Expands DataFrame rows for multiple gene mappings.

        Every row in df may have multiple gene identities defined in keys,
        with individual keys separated by keys_sep. This function expands
        every row that has 1:n mapping to n x 1:1 rows.

        Args:
            df (pd.DataFrame): Expression data DataFrame
            keys (Union[List, pd.Series, np.ndarray]): Gene keys for each row
            keys_sep (str): Separator for multiple gene mappings

        Returns:
            pd.DataFrame: Expanded DataFrame with 'keys' column added
        """
        if len(keys) != len(df):
            raise ValueError("Length of keys must match number of rows in DataFrame")

        # Convert keys to strings and split by separator
        key_strings = [str(k) for k in keys]
        split_keys = [k.split(keys_sep) for k in key_strings]

        # Calculate how many times to repeat each row
        repeat_counts = [len(split_key_list) for split_key_list in split_keys]

        # Repeat rows according to number of gene mappings
        expanded_indices = np.repeat(df.index, repeat_counts)
        expanded_df = df.loc[expanded_indices].copy()

        # Flatten the split keys and assign to new 'keys' column
        flattened_keys = []
        for key_list in split_keys:
            flattened_keys.extend([k.strip() for k in key_list])

        expanded_df = expanded_df.reset_index(drop=True)
        expanded_df["keys"] = flattened_keys

        return expanded_df

    @staticmethod
    def _summarize_split(group_data: pd.DataFrame) -> pd.Series:
        """
        Summarizes grouped gene data by taking column means.

        This function handles the summarization of multiple probes mapping to
        the same gene by calculating the mean expression across all probes.

        Args:
            group_data (pd.DataFrame): DataFrame containing expression data for a single gene
                                     Must have 'keys' column

        Returns:
            pd.Series: Mean expression values across samples
        """
        if len(group_data) == 0:
            return pd.Series(dtype=float)

        # Exclude the 'keys' column for calculation
        expression_cols = [col for col in group_data.columns if col != "keys"]

        if len(expression_cols) == 0:
            return pd.Series(dtype=float)

        # If only one row, return that row's values as a Series with correct index
        if len(group_data) == 1:
            values = group_data[expression_cols].iloc[0]
            # Ensure we return a Series with proper index
            return pd.Series(values.values, index=expression_cols, dtype=float)

        # Calculate column means for multiple rows. R's colMeans() has no
        # na.rm, so one missing probe value makes the gene missing for that
        # sample (R parity register, REG-003). skipna=False does the same.
        return group_data[expression_cols].mean(axis=0, skipna=False)

    @staticmethod
    def _replace_values(
        x: pd.DataFrame, threshold_value: float = 0, replace_value: float = 1
    ) -> pd.DataFrame:
        """
        Replace values in DataFrame that are below or equal to threshold.

        Args:
            x (pd.DataFrame): Input DataFrame
            threshold_value (float): Values <= this will be replaced
            replace_value (float): Replacement value

        Returns:
            pd.DataFrame: DataFrame with values replaced
        """
        result = x.copy()
        mask = result <= threshold_value
        result[mask] = replace_value
        return result

    @staticmethod
    def _replace_nans(x: pd.DataFrame, replace_value: float = 1) -> pd.DataFrame:
        """
        Replace NaN values in DataFrame.

        Args:
            x (pd.DataFrame): Input DataFrame
            replace_value (float): Replacement value for NaNs

        Returns:
            pd.DataFrame: DataFrame with NaNs replaced
        """
        return x.fillna(replace_value)

    @staticmethod
    def _replace_nas(x: pd.DataFrame, replace_value: float = 1) -> pd.DataFrame:
        """
        Replace NA values in DataFrame (equivalent to R's is.na).

        In pandas, this is the same as replacing NaNs, but we keep this
        separate to match the R interface.

        Args:
            x (pd.DataFrame): Input DataFrame
            replace_value (float): Replacement value for NAs

        Returns:
            pd.DataFrame: DataFrame with NAs replaced
        """
        return x.fillna(replace_value)

    @staticmethod
    def _make_valid_names(names: Union[List[str], pd.Index]) -> List[str]:
        """
        Convert names to valid Python identifiers (equivalent to R's make.names).

        Args:
            names (Union[List[str], pd.Index]): Column names to convert

        Returns:
            List[str]: Valid Python identifiers
        """
        import re

        valid_names = []
        for name in names:
            # Convert to string and replace invalid characters
            str_name = str(name)
            # Replace dots and other invalid chars with underscores
            valid_name = re.sub(r"[^a-zA-Z0-9_]", "_", str_name)

            # Ensure doesn't start with number
            if valid_name and valid_name[0].isdigit():
                valid_name = "X" + valid_name

            # Ensure not empty
            if not valid_name:
                valid_name = "X"

            valid_names.append(valid_name)

        return valid_names


# Convenience functions for backward compatibility and easier access
def getSampleLevelGeneData(dataset_object: Dict, gene_names: List[str]) -> pd.DataFrame:
    """
    Convenience function matching R interface.

    Extract gene-level data from a given dataset object.

    Args:
        dataset_object (Dict): Dataset containing 'expr' and 'keys'
        gene_names (List[str]): Target gene names

    Returns:
        pd.DataFrame: Gene-level expression data
    """
    return SampleLevelGeneDataProcessor.get_sample_level_gene_data(dataset_object, gene_names)


def extractDataFromGEM(
    dataset_object: Dict, gene_names: List[str], keys_sep: str = ","
) -> pd.DataFrame:
    """
    Convenience function for core data extraction.

    Args:
        dataset_object (Dict): Dataset containing 'expr' and 'keys'
        gene_names (List[str]): Target gene names
        keys_sep (str): Separator for gene mappings

    Returns:
        pd.DataFrame: Raw extracted gene data (before cleaning)
    """
    return SampleLevelGeneDataProcessor._extract_data_from_gem(dataset_object, gene_names, keys_sep)


def replaceValues(
    x: pd.DataFrame, threshold_value: float = 0, replace_value: float = 1
) -> pd.DataFrame:
    """
    Convenience function for value replacement.

    Args:
        x (pd.DataFrame): Input data
        threshold_value (float): Threshold for replacement
        replace_value (float): Replacement value

    Returns:
        pd.DataFrame: Data with values replaced
    """
    return SampleLevelGeneDataProcessor._replace_values(x, threshold_value, replace_value)


def replaceNaNs(x: pd.DataFrame, replace_value: float = 1) -> pd.DataFrame:
    """
    Convenience function for NaN replacement.

    Args:
        x (pd.DataFrame): Input data
        replace_value (float): Replacement value

    Returns:
        pd.DataFrame: Data with NaNs replaced
    """
    return SampleLevelGeneDataProcessor._replace_nans(x, replace_value)


def replaceNAs(x: pd.DataFrame, replace_value: float = 1) -> pd.DataFrame:
    """
    Convenience function for NA replacement.

    Args:
        x (pd.DataFrame): Input data
        replace_value (float): Replacement value

    Returns:
        pd.DataFrame: Data with NAs replaced
    """
    return SampleLevelGeneDataProcessor._replace_nas(x, replace_value)
