"""
Phenotype processing functions for MetaIntegrator Python port.
This module handles cleaning and preprocessing of phenotype data,
including class vector assignment and characteristics column processing.
"""

import warnings
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# PhenotypeProcessor Class
# ------------------------------------------------------------------
class PhenotypeProcessor:
    """Class for processing and cleaning phenotype data."""

    UNWANTED_COLUMNS = {
        "status",
        "submission_date",
        "last_update_date",
        "type",
        "channel_count",
        "molecule_ch1",
        "extract_protocol_ch1",
        "label_ch1",
        "label_protocol_ch1",
        "taxid_ch1",
        "hyb_protocol",
        "scan_protocol",
        "data_processing",
        "contact_name",
        "contact_email",
        "contact_institute",
        "contact_address",
        "contact_city",
        "contact_zip/postal_code",
        "contact_country",
        "supplementary_file",
        "data_row_count",
        "contact_state",
        "contact_fax",
        "contact_phone",
        "contact_web_link",
        "contact_department",
        "molecule_ch2",
        "extract_protocol_ch2",
        "label_ch2",
        "label_protocol_ch2",
        "taxid_ch2",
        "contact_laboratory",
    }

    DEFAULT_DISEASE_KEYWORDS = [
        "disease",
        "patient",
        "case",
        "affected",
        "positive",
        "tumor",
        "cancer",
        "infection",
        "infected",
        "sick",
        "pathology",
        "syndrome",
    ]

    DEFAULT_CONTROL_KEYWORDS = [
        "control",
        "healthy",
        "normal",
        "negative",
        "unaffected",
    ]

    def __init__(self):
        """Initialize phenotype processor."""
        pass

    def clean_up_pheno(self, dataset: Dict) -> Dict:
        """
        Cleans and processes phenotype data in the provided dataset.

        This method checks for the presence of phenotype data in the input dataset
        and performs cleanup operations. The cleaned-up phenotype data replaces
        the original data in the dataset dictionary. A copy of the original
        phenotype data is stored in the "rawPheno" key. If no phenotype data
        is found, a warning is issued, and the dataset remains unchanged.

        Args:
            dataset (Dict): A dictionary containing a "pheno" key with phenotype
                data to be processed.

        Returns:
            Dict: A dictionary with updated phenotype data after cleanup.
        """
        if "pheno" not in dataset:
            warnings.warn("No phenotype data found in dataset")
            return dataset

        dataset = dataset.copy()
        dataset["rawPheno"] = dataset["pheno"].copy()
        print("Cleaning up phenotype data...")

        pheno_clean = self._remove_useless_columns(dataset["pheno"])
        pheno_clean = self._clean_up_characteristics_col(pheno_clean)

        dataset["pheno"] = pheno_clean
        print(f"Phenotype cleanup completed. Shape: {pheno_clean.shape}")
        return dataset

    def _remove_useless_columns(self, pheno: pd.DataFrame) -> pd.DataFrame:
        """
        Removes unwanted columns from a provided DataFrame based on predefined unwanted
        columns and returns the filtered DataFrame. This method eliminates columns that
        are in a predefined set of unwanted column names and outputs the remaining
        DataFrame. A summary of removed columns is printed if any unused columns are
        detected.

        Special handling for 'status' column: Only removes it if it contains typical
        GEO status values (Public, Private, etc.). If it contains other values, it's
        kept as it likely contains meaningful phenotype information.

        Args:
            pheno (pd.DataFrame): The input DataFrame containing data to be filtered.

        Returns:
            pd.DataFrame: A filtered DataFrame with only the necessary columns retained.
        """
        cols_to_remove = set()

        for col in pheno.columns:
            if col in self.UNWANTED_COLUMNS:
                # Special handling for 'status' column
                if col == "status":
                    # Only remove if it contains GEO-type status values
                    unique_vals = pheno[col].dropna().astype(str).str.lower().unique()
                    geo_status_vals = {"public", "private", "submitted"}
                    if unique_vals.size > 0 and all(
                        val in geo_status_vals for val in unique_vals
                    ):
                        cols_to_remove.add(col)
                    # Otherwise keep it - it contains meaningful phenotype data
                else:
                    cols_to_remove.add(col)

        cols_to_keep = [col for col in pheno.columns if col not in cols_to_remove]

        if cols_to_remove:
            print(f"Removed {len(cols_to_remove)} useless columns: {cols_to_remove}")
        return pheno[cols_to_keep]

    def _clean_up_characteristics_col(self, pheno: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans up and processes characteristics columns in a phenotype DataFrame.

        This method identifies columns in the given phenotype DataFrame whose values
        contain characteristics in a "key: value" format. It processes these columns
        to extract relevant information, builds a new DataFrame with the extracted
        characteristics, and combines it with the remaining non-characteristic and
        non-channel-specific data in the original phenotype DataFrame.

        Raises a warning if the resulting DataFrame contains duplicate column names.

        Args:
            pheno (pd.DataFrame): A phenotype DataFrame where columns may contain
                characteristics in the "key: value" format.

        Returns:
            pd.DataFrame: A processed DataFrame where characteristics have been extracted
                and reorganized, and all remaining data is included.
        """
        is_characteristic_col = {
            col: pheno[col].astype(str).str.contains(": ").any()
            for col in pheno.columns
        }
        characteristics_cols = [
            col for col, is_char in is_characteristic_col.items() if is_char
        ]

        if not characteristics_cols:
            return pheno

        print(f"Found {len(characteristics_cols)} characteristics columns")
        self._warn_if_scrambled(pheno, characteristics_cols)

        all_sample_values = [
            self._extract_row_metadata(pheno, idx) for idx in pheno.index
        ]

        new_pheno = self._build_new_pheno_dataframe(all_sample_values, pheno.index)

        # Filter and combine with non-characteristic data
        ch1_cols = [col for col in pheno.columns if ":ch1" in col]
        non_char_cols = [
            col
            for col in pheno.columns
            if not is_characteristic_col[col] and col not in ch1_cols
        ]

        combined_pheno = (
            pd.concat([new_pheno, pheno[non_char_cols]], axis=1)
            if non_char_cols
            else new_pheno
        )

        if len(combined_pheno.columns) != len(set(combined_pheno.columns)):
            warnings.warn("Pheno contains duplicate column names.")

        return combined_pheno

    def _check_for_scrambling(
        self, pheno: pd.DataFrame, characteristics_cols: List[str]
    ) -> List[str]:
        """
        Check for potentially scrambled characteristics columns.

        This method checks if characteristics columns are scrambled by analyzing
        whether different samples have different "keys" (the part before ': ') in
        the same column. In a properly formatted column, all samples should have
        the same key structure.

        Args:
            pheno (pd.DataFrame): The phenotype DataFrame to check for scrambling.
            characteristics_cols (List[str]): List of characteristics column names to analyze.

        Returns:
            List[str]: A list of column names that appear to be scrambled.
        """
        scrambled_cols = []

        for col in characteristics_cols:
            # Extract keys (part before ': ') from each sample in this column
            keys_per_sample = []

            for idx in pheno.index:
                row_data = pheno.loc[idx, col]
                if (
                    pd.notna(row_data)
                    and isinstance(row_data, str)
                    and ": " in str(row_data)
                ):
                    # Extract the key (part before ': ')
                    key = str(row_data).split(": ", 1)[0].strip()
                    if key:
                        keys_per_sample.append(key)

            # Check if there are multiple unique keys in this column
            # If so, the column is scrambled
            unique_keys = set(keys_per_sample)
            if len(unique_keys) > 1:
                scrambled_cols.append(col)

        return scrambled_cols

    def _warn_if_scrambled(self, pheno: pd.DataFrame, characteristics_cols: List[str]):
        """
        Warns the user if characteristics columns in the provided phenotype DataFrame
        may be scrambled.

        This method checks for potential scrambling in the specified characteristics
        columns within the given phenotype DataFrame. If any columns are identified
        as scrambled, a warning is issued to alert the user. The warning
        indicates that the tool will attempt to unscramble the columns, but it is
        recommended to verify the raw phenotype data for any irregularities.

        Args:
            pheno: pd.DataFrame
                The phenotype DataFrame to check for potential scrambling.
            characteristics_cols: List[str]
                A list of column names in the phenotype DataFrame that represent
                characteristics to be checked.

        Raises:
            UserWarning
                If any characteristics columns are found to be potentially scrambled.
        """
        scrambled_cols = self._check_for_scrambling(pheno, characteristics_cols)
        if scrambled_cols:
            warnings.warn(
                f"{len(scrambled_cols)} characteristics columns may be scrambled. "
                "The tool will unscramble them, but check rawPheno for irregularities."
            )

    def _build_new_pheno_dataframe(
        self, sample_values: List[Dict], index: pd.Index
    ) -> pd.DataFrame:
        """
        Constructs a new phenotype DataFrame using the provided sample values and index.

        This method creates a DataFrame by combining multiple dictionaries, where each
        dictionary represents sample data with categorical keys and their respective
        values. Missing category values are filled with NaN, and an attempt is made to
        convert the data to numeric format where possible.

        Args:
            sample_values: List[Dict]
                A list of dictionaries containing sample data, where keys represent
                categories and values represent the associated values for each
                category.
            index: pd.Index
                The index to assign to the resulting DataFrame.

        Returns:
            pd.DataFrame
                A DataFrame constructed from the provided sample values and index, with
                numeric conversion applied where possible.
        """
        all_categories = sorted(
            {cat for chars in sample_values for cat in chars.keys()}
        )

        new_data = {
            cat: [chars.get(cat, np.nan) for chars in sample_values]
            for cat in all_categories
        }
        df = pd.DataFrame(new_data, index=index)
        return df.apply(self._convert_to_numeric_if_possible)

    def _extract_row_metadata(self, df: pd.DataFrame, index: Any) -> Dict[str, str]:
        """
        Extracts metadata from a specific row in the given DataFrame.

        This method processes non-null values in a specified row of the provided
        DataFrame, identifying key-value pairs formatted as "key: value". The processed
        metadata is returned as a dictionary, ensuring that both keys and values are
        clean and meaningful. Handles semicolon-separated key-value pairs like
        "tissue: liver; disease: healthy".

        Args:
            df (pd.DataFrame): A pandas DataFrame containing the row to extract metadata
                from.
            index (Any): The index of the row to process.

        Returns:
            Dict[str, str]: A dictionary where keys and values correspond to extracted
            metadata pairs from the row.
        """
        metadata = {}
        # Iterate over non-null values in the row to avoid unnecessary 'nan' string checks
        for raw_val in df.loc[index].dropna():
            val_str = str(raw_val)
            if ": " in val_str:
                # Split by semicolons first to handle multiple key-value pairs
                segments = val_str.split(";")
                for segment in segments:
                    segment = segment.strip()
                    if ": " in segment:
                        key, separator, value = segment.partition(": ")
                        if separator:
                            clean_key, clean_val = key.strip(), value.strip()
                            # Only add if both the key and value are meaningful
                            if clean_key and clean_val:
                                metadata[clean_key] = clean_val
        return metadata

    def _convert_to_numeric_if_possible(self, series: pd.Series) -> pd.Series:
        """
        Attempts to convert a pandas Series to a numeric data type. If the conversion
        does not result in data loss (other than the coercion of non-numeric strings
        to NaN), the converted Series is returned. Otherwise, the original Series is
        returned unchanged.

        Args:
            series (pd.Series): The pandas Series object to attempt numeric conversion on.

        Returns:
            pd.Series: A pandas Series object, which is either the successfully converted
            numeric Series or the original Series if conversion would result in data loss.
        """
        if series.dropna().empty:
            return series

        # Attempt conversion; 'coerce' turns non-numeric strings into NaN
        numeric_series = pd.to_numeric(series, errors="coerce")

        # If we didn't lose more data than we had (ignoring original NaNs), keep it
        if numeric_series.notnull().sum() == series.notnull().sum():
            return numeric_series

        return series

    def class_function(
        self, dataset: Dict, column: str, disease_terms: Union[str, List[str]]
    ) -> Dict:
        """
        Creates a binary class vector for a given dataset based on specific disease terms.

        The function processes the provided dataset to create a binary vector determining
        whether each sample corresponds to a disease term or a control. It checks if the
        required keys and column exist in the dataset and generates an updated dataset
        with the new class vector.

        Args:
            dataset: Dict
                A dictionary containing the dataset. The dataset must include a 'pheno' key
                with phenotype data as a DataFrame.
            column: str
                The name of the column in the phenotype data to be used for disease term
                filtering.
            disease_terms: Union[str, List[str]]
                A disease term (string) or list of disease terms used to determine the
                binary class vector. Accepts both single strings and lists for compatibility
                with JSON conversion from R.

        Raises:
            ValueError
                If the dataset does not contain the required 'pheno' key or if the specified
                column is not found in the phenotype data.

        Returns:
            Dict
                A dictionary containing the updated dataset with the added binary class vector
                indicating disease or control samples.
        """
        if "pheno" not in dataset:
            raise ValueError("Dataset must contain 'pheno' key")

        phenotype_df = dataset["pheno"]
        if column not in phenotype_df.columns:
            raise ValueError(f"Column '{column}' not found in phenotype data")

        # FIX: Ensure disease_terms is always a list
        # This handles the case where JSON converts single-element R vectors to strings
        if isinstance(disease_terms, str):
            disease_terms = [disease_terms]
        elif not isinstance(disease_terms, list):
            # Convert other iterables to list
            disease_terms = list(disease_terms)

        # Create mask based on disease terms
        labels = phenotype_df[column].astype(str).str.strip()
        disease_mask = labels.isin(disease_terms)

        class_vec = disease_mask.astype(int).values

        print(
            f"Created class vector: {class_vec.sum()} disease samples, "
            f"{len(class_vec) - class_vec.sum()} control samples"
        )
        print(f"Disease terms found: {labels[disease_mask].unique()}")

        dataset = dataset.copy()
        dataset["class"] = class_vec
        return dataset

    def auto_detect_disease_terms(
        self,
        df: pd.DataFrame,
        column: str,
        disease_keywords: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Detect and classify potential disease terms from a specified column in a DataFrame.

        This function analyzes the unique values within a given column in the provided
        DataFrame and partitions them into two categories: control terms and potential
        disease terms. The partitioning is based on the presence of predefined control
        keywords and user-specified or default disease keywords.

        Args:
            df : pd.DataFrame
                The DataFrame containing phenotype data.
            column : str
                The name of the column in the DataFrame to analyze for disease terms.
            disease_keywords : Optional[List[str]]
                A list of disease-related keywords used to identify potential disease
                terms. If not provided, default disease keywords are used.

        Returns:
            List[str]
                List of terms classified as potential disease terms.

        Raises:
            ValueError
                If the specified column is not found in the DataFrame.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in phenotype data")

        disease_kw = disease_keywords or self.DEFAULT_DISEASE_KEYWORDS
        control_kw = self.DEFAULT_CONTROL_KEYWORDS

        # Get cleaned unique values
        labels = df[column].dropna().astype(str).str.strip().unique()

        # Partition labels into control and potential disease
        control_terms = [
            term for term in labels if any(kw in term.lower() for kw in control_kw)
        ]

        # Any term that is not a control is a potential disease term
        potential_disease_terms = [term for term in labels if term not in control_terms]

        print(
            f"Found {len(potential_disease_terms)} potential disease terms: {potential_disease_terms}"
        )
        print(f"Found {len(control_terms)} potential control terms: {control_terms}")

        return potential_disease_terms

    def validate_class_assignment(
        self, dataset: Dict, min_cases: int = 3, min_controls: int = 3
    ) -> bool:
        """
        Validates the class distribution in a given dataset to ensure there are sufficient
        samples in both case and control groups.

        The function checks the 'class' vector in the given dataset (which is expected to
        differentiate between cases and controls) and performs validation based on the
        specified minimum number of cases and controls. Issues are logged as warnings
        if the dataset does not meet the specified requirements.

        Args:
            dataset: Dict
                The input dataset containing a "class" vector to evaluate.
            min_cases: int, default 3
                The minimum number of case samples required.
            min_controls: int, default 3
                The minimum number of control samples required.

        Returns:
            bool
                True if the 'class' vector meets the minimum requirements for cases and
                controls; False otherwise.

        Raises:
            UserWarning
                If the 'class' vector is missing or the dataset does not meet the
                minimum requirements.
        """
        class_vec = dataset.get("class")
        if class_vec is None:
            warnings.warn("No class vector found in dataset")
            return False

        # Convert to numpy array to ensure element-wise comparison works
        # (Python lists don't support element-wise comparison with ==)
        class_vec = np.asarray(class_vec)

        case_count = np.sum(class_vec == 1)
        control_count = np.sum(class_vec == 0)

        print(f"Class distribution: {case_count} cases, {control_count} controls")

        issues = []
        if case_count < min_cases:
            issues.append(f"{case_count} case samples (min {min_cases})")
        if control_count < min_controls:
            issues.append(f"{control_count} control samples (min {min_controls})")

        if issues:
            warnings.warn(f"Insufficient samples: {', '.join(issues)} recommended.")
            return False

        return True

    def create_class_summary(self, dataset: Dict, grouping_column: str) -> pd.DataFrame:
        """
        Generates a grouped class summary based on a provided dataset and grouping column.

        This method processes the input dataset to create a summary table, grouped by a specified
        phenotype column, containing counts of controls, cases, total samples, and their case ratio.
        The dataset must include both class information and phenotype data.

        Args:
            dataset: Dict
                A dictionary containing the dataset with keys "class" and "pheno". The "class" key should
                map to a vector with binary values (0 or 1), representing controls and cases, respectively.
                The "pheno" key should map to a DataFrame containing the phenotype data, including the
                grouping column.

            grouping_column: str
                A column name within the "pheno" DataFrame to be used for grouping the analysis. The column
                must exist in the provided phenotype data.

        Returns:
            pd.DataFrame
                A DataFrame summarizing the grouped class results. The result contains the grouping column
                and additional columns:
                    - n_controls: Count of controls (class = 0)
                    - n_cases: Count of cases (class = 1)
                    - total: Total number of samples
                    - case_ratio: Ratio of cases among the total samples

        Raises:
            ValueError
                If the dataset does not include the required keys ("class" and "pheno") or if the grouping
                column does not exist in the phenotype data.
        """
        if "class" not in dataset or "pheno" not in dataset:
            raise ValueError("Dataset must contain 'class' and 'pheno' keys")

        df = dataset["pheno"]
        if grouping_column not in df.columns:
            raise ValueError(f"Column '{grouping_column}' not found in phenotype data")

        # Temporarily combine class vector with phenotype for grouped analysis
        analysis_df = df[[grouping_column]].assign(class_val=dataset["class"])

        summary = (
            analysis_df.groupby(grouping_column)["class_val"]
            .agg(
                n_controls=lambda x: (x == 0).sum(),
                n_cases=lambda x: (x == 1).sum(),
                total="count",
                case_ratio="mean",
            )
            .reset_index()
        )

        return summary


# ------------------------------------------------------------------
# Convenience functions for backward compatibility
# ------------------------------------------------------------------
def clean_up_pheno(dataset: Dict) -> Dict:
    """
    Cleans up a given phenotype dataset.

    The function processes a given phenotype dataset using the PhenotypeProcessor
    class. It applies the clean_up_pheno method of the processor on the dataset,
    returning a processed and cleaned phenotype dataset.

    Args:
        dataset (Dict): The input dataset containing phenotype information.

    Returns:
        Dict: A cleaned and processed phenotype dataset.
    """
    processor = PhenotypeProcessor()
    return processor.clean_up_pheno(dataset)


def class_function(
    dataset: Dict, column: str, disease_terms: Union[str, List[str]]
) -> Dict:
    """
    Executes a classification function on the provided dataset using phenotype processor.

    The function leverages the PhenotypeProcessor to process a dataset for mapping or
    classifying observed data based on specified disease-related terms.

    Args:
        dataset: Dict
            The input dataset that contains the data to be classified.
        column: str
            The name of the column in the dataset that includes the data being analyzed.
        disease_terms: Union[str, List[str]]
            A disease term (string) or list of terms that represent diseases to be used
            for classification.

    Returns:
        Dict
            The processed and classified dataset as returned by the PhenotypeProcessor.
    """
    processor = PhenotypeProcessor()
    return processor.class_function(dataset, column, disease_terms)


def auto_detect_disease_terms(pheno: pd.DataFrame, column: str) -> List[str]:
    """
    Automatically detects disease terms related to phenotypic data within a given DataFrame column.

    The function utilizes a PhenotypeProcessor object to process phenotypic DataFrame data
    and identify relevant disease terms. This is particularly useful for analyzing and extracting
    key disease-related terms from large datasets in biomedical research or similar fields.

    Args:
        pheno (pd.DataFrame): A pandas DataFrame containing phenotypic data.
        column (str): The column name within the DataFrame that contains relevant data to be analyzed.

    Returns:
        List[str]: A list of detected disease terms extracted from the specified column of the
        DataFrame.
    """
    processor = PhenotypeProcessor()
    return processor.auto_detect_disease_terms(pheno, column)


def batch_process_phenotypes(
    datasets: Dict[str, Dict], disease_assignments: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    Processes multiple datasets of phenotype data by cleaning, classifying, and validating
    it according to provided disease assignments.

    This function utilizes a PhenotypeProcessor for handling cleaning, classification,
    and validation of phenotype data. It iterates over all datasets and applies specific
    class configuration and validation if applicable. The processed data is returned in a
    dictionary corresponding to the input datasets.

    Args:
        datasets (Dict[str, Dict]): A dictionary where each key represents the name of a
            dataset and its value is a dictionary containing phenotype data.
        disease_assignments (Dict[str, Dict]): A dictionary specifying class configuration
            per dataset. Each key corresponds to a dataset name, and its value should
            include configuration details like the column to classify and the
            disease terms.

    Returns:
        Dict[str, Dict]: A dictionary where each key corresponds to the name of a dataset
        and its value is a dictionary of processed phenotype data.
    """
    processor = PhenotypeProcessor()

    def _process_single_dataset(name: str, data: Dict) -> Dict:
        print(f"\nProcessing dataset: {name}")

        # 1. Clean up phenotype data
        processed = processor.clean_up_pheno(data)

        # 2. Assign and validate classes if configuration is provided
        if name in disease_assignments:
            config = disease_assignments[name]
            processed = processor.class_function(
                processed, config["column"], config["disease_terms"]
            )
            processor.validate_class_assignment(processed)

        return processed

    return {name: _process_single_dataset(name, ds) for name, ds in datasets.items()}


def analyze_phenotype_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes phenotype columns in a DataFrame to classify them into different categories
    and collect metadata about their characteristics.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        Dict[str, Any]: A dictionary containing the analysis results:
            - 'n_samples' (int): The number of rows in the DataFrame.
            - 'n_columns' (int): The total number of columns in the DataFrame.
            - 'column_types' (Dict[str, str]): A mapping of column names to their identified types,
              which can be 'empty', 'characteristics', 'numeric', or 'categorical'.
            - 'characteristics_columns' (List[str]): A list of columns identified as
              'characteristics' based on key-value-like content.
            - 'numeric_columns' (List[str]): A list of columns identified as numeric.
            - 'potential_class_columns' (List[Dict[str, Any]]): A list of potential class columns
              that may be suitable for binary or categorical classification. Each entry provides:
                 - 'column' (str): The column name.
                 - 'unique_values' (List[Any]): Unique values in the column.
                 - 'n_unique' (int): The number of unique values.
    """
    analysis = {
        "n_samples": len(df),
        "n_columns": len(df.columns),
        "column_types": {},
        "characteristics_columns": [],
        "numeric_columns": [],
        "potential_class_columns": [],
    }

    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            analysis["column_types"][col] = "empty"
            continue

        # 1. Identify Characteristics (Key: Value pairs)
        if series.astype(str).str.contains(": ").any():
            analysis["characteristics_columns"].append(col)
            analysis["column_types"][col] = "characteristics"

        # 2. Identify Numeric
        elif pd.api.types.is_numeric_dtype(series):
            analysis["numeric_columns"].append(col)
            analysis["column_types"][col] = "numeric"

        # 3. Categorical & Potential Class Columns
        else:
            unique_vals = series.unique()
            # If 2-10 unique values, it's a good candidate for binary/categorical classification
            if 2 <= len(unique_vals) <= 10:
                analysis["potential_class_columns"].append(
                    {
                        "column": col,
                        "unique_values": list(unique_vals),
                        "n_unique": len(unique_vals),
                    }
                )
            analysis["column_types"][col] = "categorical"

    return analysis
