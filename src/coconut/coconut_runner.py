import os
import pprint
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from utils.convert_floats_to_ints import run_convert_floats_to_int64
from .coconut import coconut, combine_coco_output, CombatControlOutput

# ------------------------------------------------------------------
# Global debug flag & helper
# ------------------------------------------------------------------
debug_mode = False


def debug_msg(msg: str):
    """Print a debug message only when `debug_mode` is True."""
    if debug_mode:
        sys.stderr.write(f"{msg}\n")
        # sys.stderr.write(f"[DEBUG] {msg}\n")


def load_gse_data_from_csv_directory(
    csv_directory: str,
    genes_pattern: str = "_genes.csv",
    pheno_pattern: str = "_pheno.csv",
    object_prefix: str = "GSEs.test_",
) -> List[Dict[str, pd.DataFrame]]:
    """
    Load GSE datasets from CSV files in a directory for use with COCONUT algorithm.

    This function reconstructs the nested GSE data structure from CSV files that were exported from R
    using the convert_nested_rdata.R script.

    Args:
        csv_directory: Path to directory containing CSV files
        genes_pattern: Pattern to identify gene expression files (default: "_genes.csv")
        pheno_pattern: Pattern to identify phenotype files (default: "_pheno.csv")
        object_prefix: Prefix to remove from filenames to get dataset names (default: "GSEs.test_")

    Returns:
        List of dictionaries, each containing:
            - 'genes': DataFrame with genes as rows, samples as columns
            - 'pheno': DataFrame with sample metadata

    Example:
        >>> gse_data = load_gse_data_from_csv_directory("data_txt/")
        >>> coconut_result = coconut(gse_data, control_0_col="Healthy0.Sepsis1")

    Raises:
        FileNotFoundError: If directory doesn't exist or no matching files found
        ValueError: If genes and pheno files don't match up properly
    """

    csv_dir = Path(csv_directory)
    if not csv_dir.exists():
        raise FileNotFoundError(f"Directory not found: {csv_directory}")

    debug_msg(f"🔍 Scanning directory: {csv_dir}")

    # Find all CSV files
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_directory}")

    debug_msg(f"📁 Found {len(csv_files)} CSV files")

    # Group files by dataset
    datasets = _group_files_by_dataset(
        csv_files, genes_pattern, pheno_pattern, object_prefix
    )

    # Load each dataset
    gse_list = []
    for dataset_name, file_info in datasets.items():
        debug_msg(f"\n📊 Loading dataset: {dataset_name}")

        try:
            gse_dict = _load_single_dataset(dataset_name, file_info)
            gse_list.append(gse_dict)

        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {e}")
            raise

    debug_msg(f"\n✅ Successfully loaded {len(gse_list)} datasets")
    _print_dataset_summary(gse_list)

    return gse_list


def _group_files_by_dataset(
    csv_files: List[Path],
    genes_pattern: str,
    pheno_pattern: str,
    object_prefix: str,
) -> Dict[str, Dict[str, Path]]:
    """
    Groups CSV files into datasets based on specified patterns for genes and phenotype file
    identifiers. Valid datasets must include both "genes" and "pheno" files, identified by
    matching their respective patterns in the filenames. An error is raised if no complete
    datasets are found.

    Args:
        csv_files: List[Path]
            A list of file paths to CSV files that should be grouped into datasets.
        genes_pattern: str
            The substring pattern used to identify gene files in the given filenames.
        pheno_pattern: str
            The substring pattern used to identify phenotype files in the given filenames.
        object_prefix: str
            The prefix to help extract the dataset name from the filenames.

    Returns:
        Dict[str, Dict[str, Path]]
            A dictionary where the keys are dataset names, and values are dictionaries containing
            categorized file paths for "genes" and "pheno". Only datasets with both files are
            included.

    Raises:
        ValueError
            If no datasets containing both "genes" and "pheno" files are found.
    """

    datasets = {}

    for file_path in csv_files:
        filename = file_path.name

        # Skip files that don't match our patterns
        if not (genes_pattern in filename or pheno_pattern in filename):
            debug_msg(f"⏭️  Skipping unrecognized file: {filename}")
            continue

        # Extract dataset name
        dataset_name = _extract_dataset_name(
            filename, object_prefix, genes_pattern, pheno_pattern
        )

        if dataset_name not in datasets:
            datasets[dataset_name] = {}

        # Categorize file type
        if genes_pattern in filename:
            datasets[dataset_name]["genes"] = file_path
        elif pheno_pattern in filename:
            datasets[dataset_name]["pheno"] = file_path

    # Validate that each dataset has both files
    valid_datasets = {}
    for name, files in datasets.items():
        if "genes" in files and "pheno" in files:
            valid_datasets[name] = files
        else:
            missing = [ftype for ftype in ["genes", "pheno"] if ftype not in files]
            print(f"⚠️  Incomplete dataset {name}, missing: {missing}")

    if not valid_datasets:
        raise ValueError(
            "No complete datasets found (each needs both genes and pheno files)"
        )

    return valid_datasets


def _extract_dataset_name(
    filename: str, prefix: str, genes_pattern: str, pheno_pattern: str
) -> str:
    """
    Extracts the dataset name from a given filename by removing specific patterns
    and prefixes. This function is useful in identifying the base name of a dataset
    file after stripping away extensions, prefixes, and other patterns that might
    be present to indicate file type or usage context.

    Args:
        filename: str
            The full name of the file (including extension) from which the dataset
            name is to be extracted.
        prefix: str
            The prefix, if present in the filename, to be removed before further
            processing.
        genes_pattern: str
            A specific pattern representing gene-related files that might be present
            in the filename and needs to be stripped.
        pheno_pattern: str
            A specific pattern representing phenotype-related files that might be
            present in the filename and needs to be stripped.

    Returns:
        str
            The extracted dataset name after the removal of the file extension,
            prefix, and specific patterns.

    """

    # Remove file extension
    name_without_ext = filename.replace(".csv", "")

    # Remove object prefix if present
    if prefix and name_without_ext.startswith(prefix):
        name_without_ext = name_without_ext[len(prefix) :]

    # Remove file type suffix
    for pattern in [
        genes_pattern.replace(".csv", ""),
        pheno_pattern.replace(".csv", ""),
    ]:
        if name_without_ext.endswith(pattern):
            name_without_ext = name_without_ext[: -len(pattern)]
            break

    return name_without_ext


def _load_single_dataset(
    dataset_name: str, file_info: Dict[str, Path]
) -> Dict[str, pd.DataFrame]:
    """
    Loads a single dataset consisting of genes and phenotype information, validates their alignment,
    and performs corrections to ensure consistency.

    This function reads in two files: one representing gene expression data and the other describing
    phenotype attributes for samples. It ensures that sample names in both data sources align by
    cleaning, comparing, and reconciling any mismatches. The output is a dictionary containing
    the cleaned and aligned data as pandas DataFrames.

    Args:
        dataset_name: str
            The name of the dataset being processed. Used for debug outputs.
        file_info: Dict[str, Path]
            A dictionary containing file paths for 'genes' and 'pheno'. Keys in the dictionary must be:
            'genes': Path to the gene expression data file (CSV).
            'pheno': Path to the phenotype data file (CSV).

    Returns:
        Dict[str, pd.DataFrame]
            A dictionary containing the processed and aligned dataset. Keys in the dictionary are:
            'genes': Aligned gene expression DataFrame.
            'pheno': Aligned phenotype information DataFrame.

    Raises:
        ValueError
            If there are no common samples between the gene and phenotype data, an error is raised
            as processing cannot continue.
    """

    # Load genes file
    genes_df = pd.read_csv(file_info["genes"], index_col=0)

    debug_msg(f"   📈 Genes: {genes_df.shape[0]} genes × {genes_df.shape[1]} samples")
    debug_msg(f"   📈 Sample names in genes: {list(genes_df.columns[:3])}...")

    # Load pheno file
    pheno_df = pd.read_csv(file_info["pheno"], index_col=0)

    debug_msg(
        f"   📋 Pheno: {pheno_df.shape[0]} samples × {pheno_df.shape[1]} attributes"
    )
    debug_msg(f"   📋 Sample names in pheno: {list(pheno_df.index[:3])}...")

    # Convert sample names to strings and strip whitespace for comparison
    genes_samples = set(str(col).strip() for col in genes_df.columns)
    pheno_samples = set(str(idx).strip() for idx in pheno_df.index)

    debug_msg(
        f"   🔍 Debug - Genes samples (cleaned): {sorted(list(genes_samples))[:3]}..."
    )
    debug_msg(
        f"   🔍 Debug - Pheno samples (cleaned): {sorted(list(pheno_samples))[:3]}..."
    )

    # Find common samples
    common_samples = genes_samples.intersection(pheno_samples)

    if not common_samples:
        raise ValueError(
            f"No common samples found between genes and pheno in {dataset_name}"
        )

    # Check if there's a mismatch and report it
    if genes_samples != pheno_samples:
        print(f"⚠️  Sample mismatch in {dataset_name}:")
        print(f"      Genes has: {len(genes_samples)} samples")
        print(f"      Pheno has: {len(pheno_samples)} samples")

        # Show specific mismatched samples
        genes_only = genes_samples - pheno_samples
        pheno_only = pheno_samples - genes_samples

        if genes_only:
            print(f"      Only in genes: {sorted(list(genes_only))[:5]}")
        if pheno_only:
            print(f"      Only in pheno: {sorted(list(pheno_only))[:5]}")

        print(f"      Using {len(common_samples)} common samples")

    # ALWAYS standardize sample names (not just when there's a mismatch)
    # This is the fix for whitespace and numeric type issues

    # Create standardized list of sample names
    common_samples_list = sorted(common_samples)

    # For genes DataFrame, find original column names that match cleaned versions
    genes_cols_to_keep = []
    for clean_name in common_samples_list:
        for orig_col in genes_df.columns:
            if str(orig_col).strip() == clean_name:
                genes_cols_to_keep.append(orig_col)
                break

    # For pheno DataFrame, find original index names that match cleaned versions
    pheno_rows_to_keep = []
    for clean_name in common_samples_list:
        for orig_idx in pheno_df.index:
            if str(orig_idx).strip() == clean_name:
                pheno_rows_to_keep.append(orig_idx)
                break

    # Filter to common samples
    genes_df = genes_df[genes_cols_to_keep]
    pheno_df = pheno_df.loc[pheno_rows_to_keep]

    # Standardize the sample names (ensure both use the same cleaned names)
    genes_df.columns = common_samples_list
    pheno_df.index = common_samples_list

    debug_msg(f"   ✅ Final alignment: {genes_df.shape[1]} samples")

    return {"genes": genes_df, "pheno": pheno_df}


def _print_dataset_summary(gse_list: List[Dict[str, pd.DataFrame]]) -> None:
    """
    Prints a summary of a given list of dataset information.

    This function processes a list of dataset dictionaries containing gene expression data
    and phenotype data for different datasets. It prints detailed information such as
    the count of genes and samples in each dataset, key columns present in phenotype
    data, and overall statistics for unique genes and sample counts.

    Args:
        gse_list: List of dictionaries where each dictionary represents a dataset
                  and contains:
                    - "genes" (pd.DataFrame): Contains the gene expression data where
                      rows represent genes and columns represent samples.
                    - "pheno" (pd.DataFrame): Contains phenotype data where rows
                      represent samples and columns represent attributes.

    Returns:
        None
    """

    print(f"\n{'=' * 60}")
    print("📊 DATASET SUMMARY")
    print(f"{'=' * 60}")

    total_genes = set()
    total_samples = 0

    for i, gse_data in enumerate(gse_list, 1):
        genes_df = gse_data["genes"]
        pheno_df = gse_data["pheno"]

        print(f"\n[{i}] Dataset {i}")
        print(
            f"    Genes: {genes_df.shape[0]} (rows) × {genes_df.shape[1]} samples (cols)"
        )
        print(
            f"    Pheno: {pheno_df.shape[0]} samples × {pheno_df.shape[1]} attributes"
        )

        # Show some column names from pheno
        key_columns = [
            col
            for col in pheno_df.columns
            if any(
                keyword in col.lower()
                for keyword in ["healthy", "sepsis", "diagnosis", "platform", "class"]
            )
        ]
        if key_columns:
            print(f"    Key columns: {', '.join(key_columns[:5])}")

        total_genes.update(genes_df.index)
        total_samples += genes_df.shape[1]

    print(f"\n📈 Overall:")
    print(f"    Total unique genes: {len(total_genes)}")
    print(f"    Total samples: {total_samples}")
    print(f"    Datasets: {len(gse_list)}")


def load_and_run_coconut_example(csv_directory: str) -> Dict:
    """
    Executes the COCONUT pipeline on data loaded from a directory containing CSV files. The function processes
    phenotype data, identifies control columns, and runs the COCONUT algorithm to normalize datasets.

    Args:
        csv_directory: str
            The path to the directory containing the CSV files for loading GSE data.

    Returns:
        Dict
            The results of the COCONUT algorithm containing normalized data and related information.

    Raises:
        Exception
            If the COCONUT processing pipeline fails, the exception is caught and re-raised after printing
            an error message and suggesting corrections to the user.
    """

    print("🥥 COCONUT Pipeline Starting...")

    # Step 1: Load data
    gse_data = load_gse_data_from_csv_directory(csv_directory)

    # Step 2: Inspect available columns to determine the right control column
    print(f"\n🔍 Inspecting phenotype columns...")
    for i, gse in enumerate(gse_data):
        print(f"\nDataset {i + 1} columns:")
        for col in gse["pheno"].columns:
            unique_vals = gse["pheno"][col].unique()
            print(f"  {col}: {unique_vals[:5]}{'...' if len(unique_vals) > 5 else ''}")

    # Step 3: Run COCONUT (you'll need to specify the correct column names)
    print(f"\n🥥 Running COCONUT algorithm...")

    # Example - adjust these column names based on your data structure
    try:
        coconut_result = coconut(
            gses=gse_data,
            control_0_col="Healthy0.Sepsis1",  # Adjust this based on your data
            disease_col=None,  # Adjust if needed
            by_platform=False,  # True,  # Set based on whether you want platform-specific normalization
            platform_col="Platform_id",  # Adjust based on your data
            par_prior=True,
            it_conv=1e-04,
        )

        print(f"✅ COCONUT completed successfully!")
        return coconut_result

    except Exception as e:
        print(f"❌ COCONUT failed: {e}")
        print(f"💡 Please check the column names in your phenotype data")
        raise


def _write_output_to_csv(
    combined_result: CombatControlOutput,
    genes_file: str,
    samples_genes_file: str,
    pheno_file: str,
    class_file: str,
):
    """
    Writes the processed output data to CSV files.

    This function saves gene data, phenotype data, and class labels to separate CSV files
    in a format that matches specific requirements, e.g., transposing matrices or converting
    labels to integers. It ensures compatibility with downstream processes such as R-language
    data consumption. Each file is saved with consistent formatting, and the function provides
    informative logs about the output data.

    Args:
        combined_result (CombatControlOutput): Dictionary containing processed data matrices -
                'genes' (pandas.DataFrame), 'pheno' (pandas.DataFrame), and class labels
                'class_cntl0_dis1' (numpy.ndarray or pandas.Series).
        genes_file (str): Path to save the gene data matrix CSV file.
        samples_genes_file (str): Path to save the transposed gene data matrix CSV file (samples × genes).
        pheno_file (str): Path to save the phenotype data matrix CSV file.
        class_file (str): Path to save the class labels CSV file.

    Raises:
        ValueError: If any data format issues or unsupported data transformations arise during CSV writing.

    """
    genes_df = combined_result["genes"]
    pheno_df = combined_result["pheno"]
    class_labels = combined_result["class_cntl0_dis1"]

    # Get sample IDs from the genes matrix (column names) or pheno matrix (index)
    sample_ids = genes_df.columns  # or pheno_df.index

    # Create properly formatted class labels DataFrame (matching R format)
    class_labels_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "class_label": class_labels.astype(
                int
            ),  # Convert to int to match R (0, 1 not 0.0, 1.0)
        }
    )

    # Save all three files with consistent formatting
    print("💾 Saving final combined data to CSV files...")

    # Save genes (ensure genes are rows, samples are columns)
    genes_df.to_csv(genes_file, index=True)  # Keep gene names as row index
    print(f"✅ Genes saved: {genes_df.shape[0]} genes × {genes_df.shape[1]} samples")

    # Save genes as SAMPLES × GENES (transpose the matrix)
    genes_transposed = (
        genes_df.T
    )  # Transpose: samples become rows, genes become columns
    genes_transposed.to_csv(samples_genes_file, index=True)  # Sample IDs as row index
    print(
        f"✅ Genes saved: {genes_transposed.shape[0]} samples × {genes_transposed.shape[1]} genes"
    )

    # Save pheno (samples as rows, attributes as columns)
    pheno_df.to_csv(pheno_file, index=True)  # Keep sample IDs as row index
    print(
        f"✅ Pheno saved: {pheno_df.shape[0]} samples × {pheno_df.shape[1]} attributes"
    )

    run_convert_floats_to_int64(pheno_file, pheno_file, debug_mode)

    # Save class labels
    class_labels_df.to_csv(class_file, index=False)  # No additional index
    print(f"✅ Class labels saved: {len(class_labels_df)} samples")

    # Print class distribution for verification
    class_counts = class_labels_df["class_label"].value_counts().sort_index()
    print(f"📊 Class distribution: {dict(class_counts)}")


# -------------------------------------------------------------
# Main routine to run COCONUT and combine the results
# -------------------------------------------------------------
def runner(csv_dir: str, output_dir: str | None = None):
    """
    Run the full COCONUT pipeline and optionally write the resulting matrices to files.

    Args:
        csv_dir : str
            Path to the directory that contains the CSV files (the same structure that
            load_gse_data_from_csv_directory expects).

        output_dir : str | None
            If provided, three files will be written:
                - genes.csv
                - pheno.csv
                - class_labels.csv
            The directory will be created automatically if it does not exist. If omitted, the result is only
            printed to stdout.
    """
    try:
        # Load the data from your CSV directory
        print(f"\n🔎 Loading CSV data from '{csv_dir}' …")
        gse_data = load_gse_data_from_csv_directory(csv_dir)

        # Run COCONUT
        print("\n🚀 Running COCONUT …")
        coconut_result = load_and_run_coconut_example(csv_dir)

        # Combine results
        print("\n🔗 Combining the COCONUT output …")
        combined_result = combine_coco_output(coconut_result)

        print(f"\n🎉 Final combined data:")
        print(f"   Genes matrix: {combined_result['genes'].shape}")
        print(f"   Pheno matrix: {combined_result['pheno'].shape}")
        print(f"   Class labels: {combined_result['class_cntl0_dis1'].shape}")

        # ------------------------------------------------------------------
        # Optional file output
        # ------------------------------------------------------------------
        if output_dir:
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            genes_file = os.path.join(output_dir, "mcoco_combined_genes_samples.csv")
            samples_genes_file = os.path.join(output_dir, "mcoco_combined_genes.csv")
            pheno_file = os.path.join(output_dir, "mcoco_combined_pheno.csv")
            class_file = os.path.join(output_dir, "mcoco_combined_class_labels.csv")

            _write_output_to_csv(
                combined_result, genes_file, samples_genes_file, pheno_file, class_file
            )

            print(f"✅ Files written:")
            print(f"   • {genes_file}")
            print(f"   • {samples_genes_file}")
            print(f"   • {pheno_file}")
            print(f"   • {class_file}")

        print("\n🏁 Done!\n")
    except Exception as e:
        print(f"Error: {e}")


# -------------------------------------------------------------
# entrypoint function
# -------------------------------------------------------------
def main(args: dict):
    print(f"Running coconut_runner.py with args:")
    pprint.pprint(args)

    # Set the global debug flag so that `debug_msg` knows
    debug_mode = args.get("debug", False)

    try:
        runner(csv_dir=args.get("csv_dir"), output_dir=args.get("output_dir"))
    except Exception as exc:
        sys.stderr.write(f"❌ Error: {exc}\n")
        sys.exit(1)
