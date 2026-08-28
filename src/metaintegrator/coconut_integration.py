"""
COCONUT integration functions for MetaIntegrator Python port.
This module handles conversion of MetaIntegrator objects to COCONUT format
and extraction of sample-level gene data.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# COCONUTIntegrator Class
# ------------------------------------------------------------------
class COCONUTIntegrator:
    """Class for integrating MetaIntegrator objects with COCONUT analysis."""

    def __init__(self) -> None:
        """Initialize the COCONUT integrator."""
        pass

    def coconut_meta_integrator(self, meta_object: Dict, **coconut_kwargs) -> Any:
        """
        Integrates meta-object data using the local COCONUT package.

        This method converts MetaIntegrator objects to the format expected by COCONUT
        and runs the COCONUT normalization algorithm.

        Args:
            meta_object : Dict
                A dictionary containing the meta-data to be processed.
            coconut_kwargs : Any
                Additional keyword arguments to customize the behavior of the COCONUT
                integration (e.g., by_platform, platform_col, disease_col).

        Returns:
        Any
            The COCONUT analysis results containing:
            - COCONUTList: Normalized disease samples
            - rawDiseaseList: Original unnormalized disease samples
            - controlList: Normalized control samples and parameters
        """
        from coconut.coconut import coconut as coconut_analysis  # cross-package; resolved via src/ on sys.path

        # Convert to gene-level meta format (dict of dicts)
        gene_level_data_dict = self.gene_level_meta(meta_object)

        # Convert from dict-of-dicts to list-of-dicts format expected by coconut
        # coconut expects: List[Dict[str, pd.DataFrame]] with "genes" and "pheno" keys
        gses_list = [
            {"genes": dataset["genes"], "pheno": dataset["pheno"]}
            for dataset_name, dataset in gene_level_data_dict.items()
        ]

        # Call COCONUT with the converted data
        # Note: parameter name is 'gses' (lowercase), not 'GSEs'
        return coconut_analysis(gses=gses_list, control_0_col="class", **coconut_kwargs)

    def gene_level_meta(self, meta_object: Dict) -> Dict:
        """
        Processes and transforms a metadata object by aggregating data at the gene level
        for each dataset present in the metadata's "originalData" field. The function
        validates each dataset structure, creates phenotype dataframes, extracts unique
        gene information, and generates corresponding gene-level datasets for further
        processing.

        Args:
            meta_object : Dict
                The metadata object containing information about datasets. It must include
                an 'originalData' field with dataset information.

        Returns:
            Dict
                A dictionary containing gene-level metadata for each dataset. The gene-level
                metadata includes phenotype dataframes and gene-specific data aggregated
                from the original metadata object.

        Raises:
            ValueError
                If the provided metadata object does not contain the field 'originalData'.
        """
        if "originalData" not in meta_object:
            raise ValueError("Meta object must contain 'originalData'")

        gene_level_meta = {}

        for dataset_name, dataset_object in meta_object["originalData"].items():
            # Validate dataset structure
            self._validate_dataset_object(dataset_object)

            # Create phenotype dataframe
            dataset_pheno = self._create_pheno_dataframe(dataset_object)

            # Get unique gene names (equivalent to unique(datasetObject$keys) in R)
            unique_genes = self._get_unique_genes(dataset_object["keys"])

            # Extract gene-level data
            gene_level_data = get_sample_level_gene_data(dataset_object, unique_genes)

            # Create dataset structure for COCONUT
            gene_level_dataset = {"pheno": dataset_pheno, "genes": gene_level_data}

            gene_level_meta[dataset_name] = gene_level_dataset

        return gene_level_meta

    def _validate_dataset_object(self, dataset_object: Dict) -> None:
        """
        Validates the given dataset object to ensure it contains all required keys.

        Args:
            dataset_object : Dict
                The dataset object to validate, which must include the required keys.

        Raises:
            ValueError
                If the dataset object is missing any of the required keys: "expr", "class",
                or "keys".
        """
        required_keys = ["expr", "class", "keys"]
        for key in required_keys:
            if key not in dataset_object:
                raise ValueError(f"Dataset object missing required key: {key}")

    def _create_pheno_dataframe(self, dataset_object: Dict) -> pd.DataFrame:
        """
        Creates a phenotype dataframe based on the provided dataset object. The dataframe is
        constructed with sample names from the expression matrix and includes class information
        from the dataset. If phenotype data is provided in the dataset, it validates and uses it;
        otherwise, it generates a minimal phenotype dataframe.

        Args:
            dataset_object : Dict
                A dictionary containing the dataset with keys "expr", "pheno", and "class".

        Returns:
            pd.DataFrame
                A DataFrame containing the phenotype data, including class information.

        Raises:
            Warning
                If the provided phenotype data length doesn't match the number of samples
                in the expression data.
        """
        # Get sample names from expression matrix
        expr_data = dataset_object["expr"]
        if isinstance(expr_data, pd.DataFrame):
            sample_names = expr_data.columns.tolist()
        else:
            sample_names = [f"Sample_{i}" for i in range(expr_data.shape[1])]

        # Create basic phenotype dataframe
        if "pheno" in dataset_object and dataset_object["pheno"] is not None:
            # Use existing phenotype data - this mimics the R behavior
            pheno_df = dataset_object["pheno"].copy()

            # Ensure valid index names (equivalent to make.names() in R)
            pheno_df.index = self._make_valid_names(pheno_df.index.tolist())

            # Verify that pheno data matches expression data dimensions
            if len(pheno_df) != len(sample_names):
                warnings.warn(
                    f"Phenotype data length ({len(pheno_df)}) doesn't match "
                    f"expression data samples ({len(sample_names)}). "
                    "Creating new phenotype dataframe."
                )
                pheno_df = pd.DataFrame(index=self._make_valid_names(sample_names))
        else:
            # Create minimal phenotype dataframe
            pheno_df = pd.DataFrame(index=self._make_valid_names(sample_names))

        # Add class information (this is required for COCONUT)
        pheno_df["class"] = dataset_object["class"]

        return pheno_df

    def _make_valid_names(self, names: List[str]) -> List[str]:
        """
        Converts a list of names into valid names following specific rules:
        1. Replace any invalid characters with underscores (dots and periods allowed).
        2. Prefix names starting with a digit with 'X'.
        3. Ensure names are not empty by substituting 'X' for empty values.

        Args:
            names (List[str]): A list of names to be processed.

        Returns:
            List[str]: A list of modified names adhering to the specified rules.
        """
        import re

        valid_names = []

        for name in names:
            # Convert to string if not already
            name = str(name)

            # Replace invalid characters with underscores (keep dots and periods)
            valid_name = re.sub(r"[^a-zA-Z0-9_.]", "_", name)

            # Ensure it doesn't start with a number
            if valid_name and valid_name[0].isdigit():
                valid_name = f"X{valid_name}"

            # Ensure it's not empty
            if not valid_name:
                valid_name = "X"

            valid_names.append(valid_name)

        return valid_names

    def _get_unique_genes(self, keys: List[str]) -> List[str]:
        """
        Get unique gene names from keys list.

        Args:
            keys: List of gene keys (may contain comma-separated multiple genes)

        Returns:
            List of unique gene names
        """
        unique_genes = set()

        for key in keys:
            if key and str(key) != "nan" and str(key) != "NA":
                # Split on comma and add each gene
                gene_names = str(key).split(",")
                for gene in gene_names:
                    gene = gene.strip()
                    if gene:
                        unique_genes.add(gene)

        return sorted(list(unique_genes))


# ------------------------------------------------------------------
# SampleLevelGeneExtractor Class
# ------------------------------------------------------------------
class SampleLevelGeneExtractor:
    """Class for extracting sample-level gene data from probe-level data."""

    DEFAULT_REPLACEMENT_VALUE = 1.0
    DEFAULT_THRESHOLD = 0.0

    def __init__(self) -> None:
        """Initialize the extractor."""
        pass

    def get_sample_level_gene_data(
        self, dataset_object: Dict, gene_names: List[str]
    ) -> pd.DataFrame:
        """
        Extracts and processes gene expression data at the sample level for specified genes.

        This method retrieves expression data for a given list of gene names
        from the provided gene expression matrix (GEM). After extracting the data,
        it ensures that problematic values, such as NaNs or values above a certain
        threshold, are replaced with predefined replacement values.

        Args:
            dataset_object (Dict): A dictionary representing the GEM data,
                containing required information to extract gene expression values.
            gene_names (List[str]): A list of gene names for which the sample-level
                data is to be extracted.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the processed gene expression
                data for the specified genes, with problematic values replaced.
        """
        # Extract data from GEM (Gene Expression Matrix)
        genes_matrix = self._extract_data_from_gem(dataset_object, gene_names)

        # Replace problematic values, NaNs, and NAs
        genes_matrix = self._replace_values(
            genes_matrix,
            threshold=self.DEFAULT_THRESHOLD,
            replace_value=self.DEFAULT_REPLACEMENT_VALUE,
        )
        genes_matrix = genes_matrix.fillna(self.DEFAULT_REPLACEMENT_VALUE)

        return genes_matrix

    def _prepare_expression_data(self, expr_data: Any) -> pd.DataFrame:
        """
        Processes and validates the input expression data, converting it into a pandas DataFrame if possible.

        Args:
            expr_data (Any): The input expression data. Must be either a pandas DataFrame or a
                2D array-like structure with a shape attribute.

        Returns:
            pd.DataFrame: The validated and converted expression data in pandas DataFrame format.

        Raises:
            ValueError: If the input expression data is not a pandas DataFrame or cannot be
                converted from a 2D array-like structure.
        """
        if isinstance(expr_data, pd.DataFrame):
            return expr_data
        if hasattr(expr_data, "shape") and len(expr_data.shape) == 2:
            return pd.DataFrame(expr_data)
        raise ValueError("Expression data must be a 2D array or DataFrame")

    def _extract_data_from_gem(
        self, dataset_object: Dict, gene_names: List[str], keys_sep: str = ","
    ) -> pd.DataFrame:
        """
        Extracts expression data for specific target genes from a given dataset object.

        This function processes a dataset object containing expression data and probe-to-gene
        mappings, and extracts expression data corresponding to a list of target genes. The
        data is aggregated at the gene level, with any splits summarized by taking the mean.
        Additionally, the resulting data is reindexed to ensure all target genes are present
        and the order is consistent with the input gene list.

        Args:
            dataset_object (Dict): A dictionary containing the expression data and
                probe-to-gene mappings. The dictionary is expected to include the keys:
                - "expr": A DataFrame with the expression data.
                - "keys": A mapping dict for probes to genes.
            gene_names (List[str]): A list of target gene names for which expression data is to
                be extracted.
            keys_sep (str, optional): A string used to separate values in the probe-to-gene
                mappings. Defaults to ",".

        Returns:
            pd.DataFrame: A DataFrame containing the aggregated expression data for the
            specified target genes. The rows correspond to genes in the input list, and the
            columns represent sample-level data.
        """
        expr_data = self._prepare_expression_data(dataset_object["expr"])
        keys = dataset_object["keys"]

        # Vectorized expansion of probe-to-gene mappings
        expanded_df = self._expand_df_sample_level(expr_data, keys, keys_sep)

        # Filter to target genes
        target_genes = set(gene_names)
        filtered_df = expanded_df[expanded_df["keys"].isin(target_genes)]

        # Aggregate data: Group by gene and take mean (summarize split)
        if not filtered_df.empty:
            result_df = filtered_df.groupby("keys").mean()
        else:
            result_df = pd.DataFrame(index=pd.Index([], name="keys"))

        # Ensure all target genes are present and order is preserved
        sample_names = self._make_valid_column_names(expr_data.columns.tolist())
        result_df = result_df.reindex(gene_names)

        # Handle column assignment based on DataFrame state
        # If result_df has 0 columns (e.g., no matching genes found),
        # create a proper DataFrame with the right structure
        if result_df.shape[1] == 0:
            # Create DataFrame with proper rows (genes) and columns (samples)
            # All values will be NaN, which will be filled with replacement value later
            result_df = pd.DataFrame(index=gene_names, columns=sample_names, dtype=float)
        else:
            # For non-empty DataFrames, assign columns normally
            result_df.columns = sample_names

        return result_df

    def _expand_df_sample_level(
        self, df: pd.DataFrame, keys: List[str], keys_sep: str = ","
    ) -> pd.DataFrame:
        """
        Expands a DataFrame at the sample level by adding and processing a "keys" column
        based on the provided list of keys. This involves filtering out invalid keys,
        splitting keys with a specified separator, and exploding the DataFrame to create
        separate rows for split keys.

        Args:
            df : pd.DataFrame
                The input DataFrame that will be expanded with the "keys" column.
            keys : List[str]
                A list of keys corresponding to the rows of the DataFrame. Its length must
                match the number of rows in the DataFrame.
            keys_sep : str, optional
                The separator used to split multiple keys in a single entry, by default ",".

        Raises:
            ValueError
                If the length of the keys list does not match the number of rows in the
                input DataFrame.

        Returns:
            pd.DataFrame
                A new DataFrame with the added "keys" column, filtered and expanded based
                on the specified rules. Rows may be replicated or removed depending on the
                validity and splitting of keys.
        """
        if len(keys) != len(df):
            raise ValueError("Keys length must match dataframe length")

        # Create a copy to avoid modifying original and add keys
        expanded_df = df.copy()
        expanded_df["keys"] = [str(k) if k is not None else "" for k in keys]

        # Filter out invalid keys
        invalid_keys = {"", "nan", "NA", "None"}
        expanded_df = expanded_df[~expanded_df["keys"].isin(invalid_keys)]

        if expanded_df.empty:
            return expanded_df

        # Split multiple gene mappings and explode into multiple rows
        expanded_df["keys"] = expanded_df["keys"].str.split(keys_sep)
        expanded_df = expanded_df.explode("keys")
        expanded_df["keys"] = expanded_df["keys"].str.strip()

        # Remove empty strings created by split/strip
        expanded_df = expanded_df[expanded_df["keys"] != ""]

        return expanded_df

    def _replace_values(
        self, df: pd.DataFrame, threshold: float, replace_value: float
    ) -> pd.DataFrame:
        """
        Replaces values in a DataFrame based on a threshold.

        This method processes the input DataFrame and replaces values less than or equal
        to a specified threshold with the provided replacement value. The original
        DataFrame structure and other data remain unchanged.

        Args:
            df: pd.DataFrame
                The input DataFrame in which values will be replaced.

            threshold: float
                The threshold value for comparison. All values in the DataFrame less than
                or equal to this threshold will be replaced.

            replace_value: float
                The replacement value that will substitute values in the DataFrame that
                meet the threshold condition.

        Returns:
            pd.DataFrame
                A DataFrame where all occurrences of values less than or equal to the
                threshold are replaced with the specified replacement value.
        """
        return df.mask(df <= threshold, replace_value)

    def _make_valid_column_names(self, names: List[str]) -> List[str]:
        """
        Replace invalid characters in column names and ensure valid formatting.

        This method processes a list of column names to make them valid by replacing
        invalid characters, ensuring they don't start with numbers or dots followed by
        numbers, and guaranteeing they're not empty.

        Args:
            names: A list of column names.

        Returns:
            A list of valid column names.
        """
        import re

        valid_names = []

        for name in names:
            name = str(name)
            # Replace invalid characters
            valid_name = re.sub(r"[^a-zA-Z0-9_.]", ".", name)

            # Ensure doesn't start with number or dot followed by number
            if valid_name and (
                valid_name[0].isdigit()
                or (valid_name.startswith(".") and len(valid_name) > 1 and valid_name[1].isdigit())
            ):
                valid_name = f"X{valid_name}"

            # Ensure not empty
            if not valid_name or valid_name == ".":
                valid_name = "X"

            valid_names.append(valid_name)

        return valid_names

    def _replace_values(
        self, df: pd.DataFrame, threshold: float = 0, replace_value: float = 1
    ) -> pd.DataFrame:
        """
        Replace values below threshold with replacement value.

        Args:
            df: Input dataframe
            threshold: Threshold value
            replace_value: Replacement value

        Returns:
            Modified dataframe
        """
        result = df.copy()
        mask = result <= threshold
        result[mask] = replace_value
        return result

    def _replace_nans(self, df: pd.DataFrame, replace_value: float = 1) -> pd.DataFrame:
        """
        Replaces NaN (Not a Number) values in the provided DataFrame with a specified value.

        Replaces all NaN values in the DataFrame df with the replace_value. This is
        useful for cleaning data and preparing it for further processing.

        Args:
            df : pd.DataFrame
                The DataFrame in which NaN values are to be replaced.
            replace_value : float, optional
                The value to replace NaN values with. Defaults to 1.

        Returns:
            pd.DataFrame
                A new DataFrame with NaN values replaced by the specified value.
        """
        return df.fillna(replace_value)

    def _replace_nas(self, df: pd.DataFrame, replace_value: float = 1) -> pd.DataFrame:
        """
        Replaces NaN values in a DataFrame with a specified value.

        This method takes a DataFrame and substitutes any NaN (Not a Number) values
        with the provided replace_value. This can be used to ensure that the DataFrame
        has no missing values by replacing them with a user-defined numerical value.

        Args:
            df (pd.DataFrame): The DataFrame in which NaN values are to be replaced.
            replace_value (float, optional): The value used to replace NaN values in
                                              the DataFrame. Defaults to 1.

        Returns:
            pd.DataFrame: The modified DataFrame with NaN values replaced.
        """
        return df.fillna(replace_value)


# ------------------------------------------------------------------
# Convenience functions for backward compatibility and easy access
# ------------------------------------------------------------------
def coconut_meta_integrator(meta_object: Dict, **coconut_kwargs) -> Any:
    """
    Wrapper function to run COCONUT on MetaIntegrator objects.

    Args:
        meta_object: MetaIntegrator formatted meta object
        **coconut_kwargs: Additional arguments to pass to COCONUT

    Returns:
        Results from COCONUT analysis
    """
    integrator = COCONUTIntegrator()
    return integrator.coconut_meta_integrator(meta_object, **coconut_kwargs)


def gene_level_meta(meta_object: Dict) -> Dict:
    """
    Convert MetaIntegrator object to gene-level meta format for COCONUT.

    Args:
        meta_object: MetaIntegrator meta object

    Returns:
        Dictionary in COCONUT format with gene-level data
    """
    integrator = COCONUTIntegrator()
    return integrator.gene_level_meta(meta_object)


def get_sample_level_gene_data(dataset_object: Dict, gene_names: List[str]) -> pd.DataFrame:
    """
    Extract gene-level data from a dataset object.

    Args:
        dataset_object: Dataset object with expression data and gene keys
        gene_names: List of target gene names

    Returns:
        DataFrame with gene-level expression data
    """
    extractor = SampleLevelGeneExtractor()
    return extractor.get_sample_level_gene_data(dataset_object, gene_names)


# ------------------------------------------------------------------
# COCONUTDataProcessor: Additional utilities for COCONUT data processing
# ------------------------------------------------------------------
class COCONUTDataProcessor:
    """Additional utilities for COCONUT data processing."""

    @staticmethod
    def prepare_coconut_input(meta_object: Dict, filter_results: Optional[Dict] = None) -> Dict:
        """
        Prepares input data for COCONUT analysis by optionally filtering genes
        from a given filter results dictionary or by using all available genes
        otherwise. The returned data formats are adapted according to the
        COCONUT analysis requirements.

        Args:
            meta_object (Dict): Dictionary containing meta-analysis data, specifically originalData.
            filter_results (Optional[Dict]): Optional dictionary containing gene
                filtration results with keys like 'posGeneNames' and/or
                'negGeneNames' to specify targeted genes.

        Returns:
            Dict: A dictionary containing the prepared COCONUT input data. The
                data will include filtered gene-level data and corresponding
                phenotype data for each dataset present in meta_object, or
                all genes' data if filtering is not applied.

        Raises:
            None
        """
        integrator = COCONUTIntegrator()

        # Check if filter_results is not None (not just truthy, since {} is falsy)
        if filter_results is not None:
            # Use genes from filter results
            target_genes = []
            if "posGeneNames" in filter_results:
                target_genes.extend(filter_results["posGeneNames"])
            if "negGeneNames" in filter_results:
                target_genes.extend(filter_results["negGeneNames"])

            if not target_genes:
                warnings.warn("No genes found in filter results, using all available genes")
                return integrator.gene_level_meta(meta_object)

            # Process each dataset with filtered genes
            coconut_data = {}
            for dataset_name, dataset_object in meta_object["originalData"].items():
                # Create phenotype dataframe
                dataset_pheno = integrator._create_pheno_dataframe(dataset_object)

                # Extract gene-level data for target genes only
                extractor = SampleLevelGeneExtractor()
                gene_level_data = extractor.get_sample_level_gene_data(dataset_object, target_genes)

                coconut_data[dataset_name] = {
                    "pheno": dataset_pheno,
                    "genes": gene_level_data,
                }

            return coconut_data
        else:
            return integrator.gene_level_meta(meta_object)

    @staticmethod
    def validate_coconut_format(coconut_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validates the format of the provided COCONUT data to ensure it adheres to the expected
        structure. This function checks that the input is a dictionary and that each dataset
        contains the required keys and correctly formatted data. The validation process also
        verifies that specific keys such as 'pheno' and 'genes' exist and checks for their
        respective data types or required attributes.

        Args:
            coconut_data (Dict): The COCONUT data to validate, expected to be structured as a dictionary
            with datasets as keys and corresponding details as values.

        Returns:
            Tuple[bool, List[str]]: A tuple where the first element is a boolean indicating validation
            success or failure, and the second element is a list of strings describing any errors
            encountered during validation.
        """
        errors = []

        if not isinstance(coconut_data, dict):
            errors.append("COCONUT data must be a dictionary")
            return False, errors

        for dataset_name, dataset in coconut_data.items():
            if not isinstance(dataset, dict):
                errors.append(f"Dataset '{dataset_name}' must be a dictionary")
                continue

            # Check required keys
            if "pheno" not in dataset:
                errors.append(f"Dataset '{dataset_name}' missing 'pheno' key")
            if "genes" not in dataset:
                errors.append(f"Dataset '{dataset_name}' missing 'genes' key")

            # Validate phenotype data
            if "pheno" in dataset:
                pheno = dataset["pheno"]
                if not isinstance(pheno, pd.DataFrame):
                    errors.append(f"Dataset '{dataset_name}' pheno must be a DataFrame")
                elif "class" not in pheno.columns:
                    errors.append(f"Dataset '{dataset_name}' pheno missing 'class' column")

            # Validate gene data
            if "genes" in dataset:
                genes = dataset["genes"]
                if not isinstance(genes, pd.DataFrame):
                    errors.append(f"Dataset '{dataset_name}' genes must be a DataFrame")

        return len(errors) == 0, errors

    @staticmethod
    def export_for_coconut(coconut_data: Dict, output_dir: str) -> None:
        """
        Exports COCONUT data to the specified output directory. The function saves both
        phenotype and gene expression data for each dataset in the provided dictionary
        to separate CSV files. Each dataset's files are named using the dataset name.

        Args:
            coconut_data: A dictionary where keys are dataset names and values are
                          another dictionary with keys 'pheno' and 'genes', each
                          containing data that can be saved to CSV.
            output_dir: Target directory where the data will be exported. If the
                        directory does not exist, it will be created.

        Returns:
            None
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        for dataset_name, dataset in coconut_data.items():
            # Save phenotype data
            pheno_file = os.path.join(output_dir, f"{dataset_name}_pheno.csv")
            dataset["pheno"].to_csv(pheno_file)

            # Save gene expression data
            genes_file = os.path.join(output_dir, f"{dataset_name}_genes.csv")
            dataset["genes"].to_csv(genes_file)

        print(f"COCONUT data exported to {output_dir}")

    @staticmethod
    def check_required_packages() -> Tuple[bool, List[str]]:
        """
        Checks the availability of required packages.

        Since coconut is a local package (sibling to metaintegrator), it should
        always be available. This function primarily checks that the standard
        dependencies (numpy, pandas) are present.

        Returns:
            Tuple[bool, List[str]]: A tuple where the first element is a boolean
            indicating whether all required packages are available,
            and the second element is a list of missing packages.
        """
        missing_packages = []

        # Check if we can import the local coconut package
        try:
            from coconut.coconut import coconut
        except ImportError as e:
            missing_packages.append(f"coconut (local package) - {str(e)}")

        return len(missing_packages) == 0, missing_packages


# ------------------------------------------------------------------
# MetaIntegratorToCOCONUT: Class for converting MetaIntegrator data to COCONUT format
# ------------------------------------------------------------------
class MetaIntegratorToCOCONUT:
    """
    Handles the conversion of meta-object data into the COCONUT format and provides
    functionalities for validation and integration with the COCONUT pipeline.

    This class offers tools to transform input data structures into a format compatible
    with the COCONUT framework while optionally validating the input and output data.
    Additionally, this class integrates full COCONUT analysis for enhanced functionality
    after transformation. It ensures data integrity through validation mechanisms and
    provides capabilities for handling gene-level metadata.

    Attributes:
        integrator: Instance of COCONUTIntegrator for performing data integration.
        extractor: Instance of SampleLevelGeneExtractor for extracting sample-level
                   gene data.
    """

    def __init__(self) -> None:
        """Initialize the converter."""
        self.integrator = COCONUTIntegrator()
        self.extractor = SampleLevelGeneExtractor()

    def convert_full_pipeline(
        self,
        meta_object: Dict,
        validate_input: bool = True,
        validate_output: bool = True,
    ) -> Dict:
        """
        Converts a given meta_object into the COCONUT format and optionally validates both
        the input and the output.

        This function transforms meta-object data into the COCONUT-specific
        format using a defined integrator. It also allows for validation of the input data
        before transformation and validation of the output data after the transformation.
        The validation ensures the correctness of data format and consistency.

        Args:
            meta_object (Dict): The meta-object data to be converted into COCONUT format.
            validate_input (bool): A flag to determine if the input meta_object should be
                validated prior to processing.
            validate_output (bool): A flag to determine if the converted COCONUT format
                data should be validated after processing.

        Returns:
            Dict: A dictionary containing the converted data in COCONUT format.

        Raises:
            ValueError: If the output validation fails due to any inconsistencies or
                errors in the converted COCONUT data.
        """
        if validate_input:
            self._validate_meta_object(meta_object)

        # Convert to COCONUT format
        coconut_data = self.integrator.gene_level_meta(meta_object)

        if validate_output:
            is_valid, errors = COCONUTDataProcessor.validate_coconut_format(coconut_data)
            if not is_valid:
                raise ValueError(f"Output validation failed: {'; '.join(errors)}")

        return coconut_data

    def _validate_meta_object(self, meta_object: Dict) -> None:
        """
        Validates the meta object to ensure it adheres to expected structure
        and contains all required keys. This function checks that the provided
        meta object includes the 'originalData' key, that it isn't empty, and that
        each dataset within 'originalData' has the keys 'expr', 'class', and 'keys'.

        Args:
            meta_object (Dict): The meta object to validate. Must contain data in
                                the expected format.

        Raises:
            ValueError: If 'originalData' is missing from the meta object.
            ValueError: If 'originalData' is empty.
            ValueError: If any dataset within 'originalData' is missing one or more
                        required keys such as 'expr', 'class', or 'keys'.
        """
        if "originalData" not in meta_object:
            raise ValueError("Meta object must contain 'originalData'")

        if not meta_object["originalData"]:
            raise ValueError("originalData cannot be empty")

        for dataset_name, dataset in meta_object["originalData"].items():
            required_keys = ["expr", "class", "keys"]
            for key in required_keys:
                if key not in dataset:
                    raise ValueError(f"Dataset '{dataset_name}' missing required key: {key}")

    def run_with_coconut(self, meta_object: Dict, **coconut_kwargs) -> Any:
        """
        Executes the coconut integration process with the provided meta_object and additional keyword arguments.

        The function utilizes the integrator's coconut_meta_integrator method to process the given meta_object
        along with any supplementary keyword arguments, enabling flexible and dynamic integration of metadata.

        Args:
            meta_object (Dict): The metadata object to be processed by the coconut integration.
            **coconut_kwargs: Additional keyword arguments to be passed to the integrator.

        Returns:
            Any: The result of the coconut meta integration process, as determined by the integrator's method.
        """
        return self.integrator.coconut_meta_integrator(meta_object, **coconut_kwargs)


# Main convenience function that directly mirrors the R function
def run_coconut_analysis(meta_object: Dict, **coconut_kwargs) -> Any:
    """
    Direct equivalent to the R coconutMetaIntegrator function.

    Args:
        meta_object: MetaIntegrator formatted meta object
        **coconut_kwargs: Arguments to pass to COCONUT (e.g., control_0_col="class")

    Returns:
        Results from COCONUT analysis on the MetaIntegrator object

    Example:
        >>> results = run_coconut_analysis(meta_object, control_0_col="class")
    """
    converter = MetaIntegratorToCOCONUT()
    return converter.run_with_coconut(meta_object, **coconut_kwargs)
