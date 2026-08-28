#!/usr/bin/env python3
"""
compare_csv_folders.py

A small helper that walks two directories containing CSV files, finds files that match user‑supplied partial names,
loads them into pandas dataframes and reports whether the two files are identical (or within a numerical tolerance).

The two directories usually contain files with different naming
conventions, e.g.

    Folder 1   :  GSEs.COCO.combined_pheno.csv
    Folder 2   :  mcoco_combined_pheno.csv

The script accepts a list of partial strings – e.g. "combined_pheno", "combined_genes" – and automatically pairs
the corresponding files from each directory for comparison.

Typical usage
-------------
    python compare_csv_folders.py \
        --folder1 /path/to/COCO_filenames \
        --folder2 /path/to/mcoco_filenames \
        --partial combined_pheno combined_genes \
        --debug

"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1.  Helpers
# --------------------------------------------------------------------------- #
def debug_print(msg: str, debug: bool = False) -> None:
    """Print debug message to stderr if debugging is enabled."""
    if debug:
        sys.stderr.write(f"[DEBUG] {msg}\n")


def find_matching_file(folder: Path, partial: str, debug: bool = False) -> Optional[Path]:
    """
    Return the first file in *folder* whose name contains partial. If no match is found, return None.

    Args:
        folder : Path
            Directory to search.
        partial : str
            Sub‑string that must appear in the file name.
        debug : bool
            Enable verbose output.

    Returns:
        Path or None
    """
    debug_print(f"Searching for '*{partial}*' in {folder}", debug)
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if partial in p.name:
            debug_print(f"Found match: {p.name}", debug)
            return p
    debug_print(f"No match for partial '{partial}' in {folder}", debug)
    return None


def _find_sample_id_column(df: pd.DataFrame, filename: Path, debug: bool) -> str:
    """
    Find the sample ID column in the given DataFrame based on predefined possible names
    or certain conditions. If no specific conditions are met, the method will fall back
    to the first column or raise an error when no columns exist.

    Args
        df (pd.DataFrame): The DataFrame to search for the sample ID column.
        filename (Path): The filename associated with the DataFrame, included in debug messages or error raising.
        debug (bool): Flag to enable debugging messages or not.

    Returns:
        str: The name of the identified sample ID column.

    Raises:
        ValueError: If no suitable sample ID column is found in the provided DataFrame.
    """
    # common naming patterns
    possible_names = ["sampleId", "sample_id", "Sample_ID", "SampleID", "ID", "id"]

    # Check for exact matches first
    for name in possible_names:
        if name in df.columns:
            return name

    # Check for columns starting with 'Unnamed:' (often the index column)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        return unnamed_cols[0]  # Use the first unnamed column

    # If no clear sample ID column found, use the first column
    if len(df.columns) > 0:
        debug_print(
            f"Warning: No standard sample ID column found in {filename}. Using first column: '{df.columns[0]}'",
            debug,
        )
        return df.columns[0]

    raise ValueError(f"No sample ID column found in {filename}")


def compare_csvs(
    f1: Path,
    f2: Path,
    rtol: float = 1e-8,
    atol: float = 0.0,
    debug: bool = False,
) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    Compares two CSV files for structural and data equivalency within a given tolerance.

    The function compares two CSV files, checking for structural differences such as shape, column order,
    and sample ID alignment. It further analyzes numeric data differences considering relative
    and absolute tolerances, reporting any discrepancies.

    Args:
        f1: Path
            Path to the first CSV file to compare.
        f2: Path
            Path to the second CSV file to compare.
        rtol: float, optional
            Relative tolerance for numeric value comparison (default is 1e-8).
        atol: float, optional
            Absolute tolerance for numeric value comparison (default is 0.0).
        debug: bool, optional
            If True, provides detailed debug output for the comparison process (default is False).

    Returns:
        Tuple[bool, Optional[pd.DataFrame]]
            A tuple where the first element is a boolean indicating if the files are equivalent,
            and the second element is a DataFrame representing the differences (if any, or None otherwise).

    Raises:
        Exception
            If an error occurs during file reading or comparison, the exception will be caught and
            printed as a message.
    """
    try:
        # ------------------------------------------------------------------- #
        # Load files as strings initially for robust handling
        # ------------------------------------------------------------------- #
        debug_print(f">>> Loading {f1} …", debug)
        df1 = pd.read_csv(f1, header=0, quoting=0, dtype=str)
        debug_print(f">>> Loading {f2} …", debug)
        df2 = pd.read_csv(f2, header=0, quoting=0, dtype=str)

        debug_print(f"DF1 shape: {df1.shape}", debug)
        debug_print(f"DF2 shape: {df2.shape}", debug)

        if df1.shape != df2.shape:
            print(f"Shape mismatch: {df1.shape} vs {df2.shape}")
            return False, None

        # Column handling
        cols1, cols2 = list(df1.columns), list(df2.columns)
        if cols1 != cols2:
            s1, s2 = set(cols1), set(cols2)
            if s1 == s2:
                print("Re-ordering columns to match.")
                df1 = df1[cols2]
            else:
                if s1 - s2:
                    print(f"Only in file 1: {sorted(s1 - s2)}")
                if s2 - s1:
                    print(f"Only in file 2: {sorted(s2 - s1)}")
                return False, None

        # Index Alignment
        id_col1, id_col2 = _find_sample_id_column(df1, f1, debug), _find_sample_id_column(
            df2, f2, debug
        )

        debug_print(f"Using ID column '{id_col1}' for file 1", debug)
        debug_print(f"Using ID column '{id_col2}' for file 2", debug)

        # ------------------------------------------------------------------- #
        # Shape validation
        # ------------------------------------------------------------------- #
        if df1.shape != df2.shape:
            print(f"Shape mismatch: {df1.shape} vs {df2.shape}")
            return False, None

        # ------------------------------------------------------------------- #
        # Column validation and reordering
        # ------------------------------------------------------------------- #
        cols1, cols2 = list(df1.columns), list(df2.columns)
        if cols1 != cols2:
            set1, set2 = set(cols1), set(cols2)
            only_in_1 = set1 - set2
            only_in_2 = set2 - set1

            # Report genuinely missing columns
            if only_in_1:
                print(f"Columns only in file 1: {sorted(only_in_1)}")
            if only_in_2:
                print(f"Columns only in file 2: {sorted(only_in_2)}")

            # If the sets are equal but ordering differs, reorder df1 to match df2
            if set1 == set2:
                print(
                    "All columns found but column ordering differs. Re-ordering file 1 to match file 2."
                )
                df1 = df1[cols2]  # reorder df1 columns to match df2
            else:
                # Different column sets - cannot proceed
                return False, None

        # ------------------------------------------------------------------- #
        # Set sample ID as index and sort by sample ID
        # ------------------------------------------------------------------- #
        df1_indexed = df1.set_index(id_col1).sort_index()
        df2_indexed = df2.set_index(id_col2).sort_index()

        # Rename the index to a standard name for consistency
        df1_indexed.index.name = "sampleId"
        df2_indexed.index.name = "sampleId"

        # Check if sample IDs match
        idx1_set = set(df1_indexed.index)
        idx2_set = set(df2_indexed.index)

        if idx1_set != idx2_set:
            missing_in_df2 = idx1_set - idx2_set
            missing_in_df1 = idx2_set - idx1_set
            print("Sample ID mismatch:")
            if missing_in_df2:
                print(f"  Sample IDs in file1 but not file2: {sorted(missing_in_df2)}")
            if missing_in_df1:
                print(f"  Sample IDs in file2 but not file1: {sorted(missing_in_df1)}")
            return False, None

        # ------------------------------------------------------------------- #
        # Handle NA vs blank values consistently
        # ------------------------------------------------------------------- #
        # Replace both "NA" strings and empty strings with pandas NA
        df1_cleaned = df1_indexed.replace({"NA": np.nan, "": np.nan})
        df2_cleaned = df2_indexed.replace({"NA": np.nan, "": np.nan})

        debug_print(
            f"Cleaned shapes - DF1: {df1_cleaned.shape}, DF2: {df2_cleaned.shape}",
            debug,
        )

        # ------------------------------------------------------------------- #
        # Identify numeric vs non-numeric columns
        # ------------------------------------------------------------------- #
        numeric_columns = []
        non_numeric_columns = []

        for col in df1_cleaned.columns:
            try:
                # Try to convert to numeric, allowing for NaN values
                df1_col_numeric = pd.to_numeric(df1_cleaned[col], errors="coerce")
                df2_col_numeric = pd.to_numeric(df2_cleaned[col], errors="coerce")

                # Check if conversion was successful (original non-NaN values should remain non-NaN)
                df1_orig_notna = df1_cleaned[col].notna()
                df2_orig_notna = df2_cleaned[col].notna()
                df1_converted_notna = df1_col_numeric.notna()
                df2_converted_notna = df2_col_numeric.notna()

                # If conversion preserved the non-NaN pattern, it's numeric
                if (df1_orig_notna == df1_converted_notna).all() and (
                    df2_orig_notna == df2_converted_notna
                ).all():
                    df1_cleaned[col] = df1_col_numeric
                    df2_cleaned[col] = df2_col_numeric
                    numeric_columns.append(col)
                else:
                    non_numeric_columns.append(col)
            except:
                non_numeric_columns.append(col)

        debug_print(f"Numeric columns: {len(numeric_columns)}", debug)
        debug_print(f"Non-numeric columns: {len(non_numeric_columns)}", debug)

        # ------------------------------------------------------------------- #
        # Compare numeric columns with tolerance
        # ------------------------------------------------------------------- #
        numeric_identical = True
        differences = []
        different_rows = set()
        diff_df = None

        if numeric_columns:
            df1_num = df1_cleaned[numeric_columns]
            df2_num = df2_cleaned[numeric_columns]

            # Element-wise comparison
            for col in numeric_columns:
                col1_vals = df1_num[col].values
                col2_vals = df2_num[col].values

                # Use np.isclose for element-wise comparison with NaN handling
                is_close = np.isclose(col1_vals, col2_vals, rtol=rtol, atol=atol, equal_nan=True)

                if not is_close.all():
                    numeric_identical = False
                    diff_indices = np.where(~is_close)[0]
                    for idx in diff_indices:
                        sample_id = df1_num.index[idx]
                        different_rows.add(sample_id)
                        val1, val2 = col1_vals[idx], col2_vals[idx]
                        differences.append(
                            {
                                "sampleId": sample_id,
                                "column": col,
                                "file1_value": val1,
                                "file2_value": val2,
                                "abs_difference": (
                                    abs(val1 - val2)
                                    if not (np.isnan(val1) or np.isnan(val2))
                                    else np.nan
                                ),
                                "rel_difference": (
                                    abs((val1 - val2) / val1)
                                    if val1 != 0 and not (np.isnan(val1) or np.isnan(val2))
                                    else np.nan
                                ),
                                "type": "numeric",
                            }
                        )

            # Create overall numeric difference matrix for backward compatibility
            if not numeric_identical:
                arr1 = df1_num.values.astype(float)
                arr2 = df2_num.values.astype(float)
                diff = np.abs(arr1 - arr2)
                diff_df = pd.DataFrame(diff, columns=numeric_columns, index=df1_num.index)
                debug_print("Numeric values differ beyond tolerance.", debug)
            else:
                debug_print("Numeric columns match within tolerance.", debug)

        # ------------------------------------------------------------------- #
        # Compare non-numeric columns
        # ------------------------------------------------------------------- #
        non_numeric_identical = True
        if non_numeric_columns:
            df1_non = df1_cleaned[non_numeric_columns]
            df2_non = df2_cleaned[non_numeric_columns]

            # Fill NaN with sentinel for comparison
            df1_non_filled = df1_non.fillna("__NA__")
            df2_non_filled = df2_non.fillna("__NA__")

            # Element-wise comparison for non-numeric columns
            for col in non_numeric_columns:
                col1_vals = df1_non_filled[col].values
                col2_vals = df2_non_filled[col].values

                is_equal = col1_vals == col2_vals

                if not is_equal.all():
                    non_numeric_identical = False
                    diff_indices = np.where(~is_equal)[0]
                    for idx in diff_indices:
                        sample_id = df1_non.index[idx]
                        different_rows.add(sample_id)
                        # Convert back from sentinel for display
                        val1 = df1_non.iloc[idx][col]
                        val2 = df2_non.iloc[idx][col]
                        differences.append(
                            {
                                "sampleId": sample_id,
                                "column": col,
                                "file1_value": val1,
                                "file2_value": val2,
                                "abs_difference": np.nan,
                                "rel_difference": np.nan,
                                "type": "string",
                            }
                        )

            if not non_numeric_identical:
                debug_print("Non-numeric columns differ.", debug)
            else:
                debug_print("Non-numeric columns match.", debug)

        # ------------------------------------------------------------------- #
        # Final result and reporting
        # ------------------------------------------------------------------- #
        overall_identical = numeric_identical and non_numeric_identical

        if overall_identical:
            debug_print("Files are identical within tolerance.", debug)
            return True, None

        # Create comprehensive differences DataFrame
        if differences:
            comprehensive_diff_df = pd.DataFrame(differences)

            # Print summary
            print(f"Files are NOT identical!")
            print(f"Number of different rows: {len(different_rows)}")
            print(f"Total number of cell differences: {len(differences)}")

            # Show first different row details
            if different_rows:
                first_different_sample = sorted(different_rows)[0]
                print(f"\nFirst different row (sampleId: {first_different_sample}):")

                # Get all differences for this sample
                sample_diffs = comprehensive_diff_df[
                    comprehensive_diff_df["sampleId"] == first_different_sample
                ]

                print("File 1 values:")
                for _, diff in sample_diffs.iterrows():
                    print(f"  {diff['column']}: {diff['file1_value']}")

                print("File 2 values:")
                for _, diff in sample_diffs.iterrows():
                    print(f"  {diff['column']}: {diff['file2_value']}")

                print("Differences:")
                for _, diff in sample_diffs.iterrows():
                    if diff["type"] == "numeric" and not np.isnan(diff["abs_difference"]):
                        print(
                            f"  {diff['column']}: abs_diff = {diff['abs_difference']:.2e}, rel_diff = {diff['rel_difference']:.2e}"
                        )
                    else:
                        print(f"  {diff['column']}: string mismatch")

            # Return the numeric diff_df for backward compatibility, or comprehensive diff if no numeric
            return False, diff_df if diff_df is not None else comprehensive_diff_df

        return False, None

    except Exception as e:
        print(f"Error comparing CSV files: {str(e)}")
        return False, None


# --------------------------------------------------------------------------- #
# 2.  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """CLI entry point: parse --folder1/--folder2 and run the CSV comparison."""
    parser = argparse.ArgumentParser(
        description="Compare CSV files in two directories "
        "using user‑supplied partial file names."
    )
    parser.add_argument(
        "--folder1",
        type=Path,
        required=True,
        help="Path to first directory (e.g. GSEs.COCO folder).",
    )
    parser.add_argument(
        "--folder2",
        type=Path,
        required=True,
        help="Path to second directory (e.g. mcoco folder).",
    )
    parser.add_argument(
        "--partial",
        type=str,
        nargs="+",
        required=True,
        help="One or more partial strings that should appear in " "the file names to be compared.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance for numeric comparison.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for numeric comparison.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information to stderr.",
    )

    args = parser.parse_args()

    # Sanity checks
    if not args.folder1.is_dir():
        sys.exit(f"ERROR: {args.folder1} is not a directory.")
    if not args.folder2.is_dir():
        sys.exit(f"ERROR: {args.folder2} is not a directory.")

    # Loop over each partial string
    for partial in args.partial:
        f1 = find_matching_file(args.folder1, partial, args.debug)
        f2 = find_matching_file(args.folder2, partial, args.debug)

        if f1 is None or f2 is None:
            print(f"[{partial}] ->  ❌  Missing file in one of the directories.")
            continue

        identical, diff_df = compare_csvs(f1, f2, rtol=args.rtol, atol=args.atol, debug=args.debug)

        if identical:
            print(f"[{partial}] ->  ✅  Files are identical.")
        else:
            print(f"[{partial}] ->  ⚠️  Files differ.")
            # If we got a diff dataframe, show some stats
            if diff_df is not None:
                num_diff = (diff_df != 0).any(axis=1).sum()
                print(f"    Rows with any difference: {num_diff} / {diff_df.shape[0]}")
                print(f"    Example difference:")
                print(diff_df.head(3))

    print("\nComparison complete.")


# ------------------------------------------------------------------- #
# Test CSV file comparison
# ------------------------------------------------------------------- #
def test_compare_csvs() -> None:
    """Test function to confirm functionality with different CSV types"""

    import tempfile

    # Test data for different CSV types you mentioned

    # Type 1: sample_id,class_label format
    type1_data = {
        "sample_id": ["GSM102961", "GSM102962", "GSM102963", "GSM102964"],
        "class_label": [0, 0, 1, 1],
        "feature1": [1.5, 2.3, 3.1, 2.8],
        "feature2": [0.8, 1.2, 0.9, 1.1],
    }

    type1_data_modified = {
        "sample_id": ["GSM102961", "GSM102962", "GSM102963", "GSM102964"],
        "class_label": [0, 0, 1, 1],
        "feature1": [1.5, 2.3000001, 3.1, 2.8],  # Small difference
        "feature2": [0.8, 1.2, 0.9, 1.1],
    }

    # Type 3: Unnamed: 0 format (simulating pandas index column)
    type3_data = {
        "Unnamed: 0": ["GSM102961", "GSM102962", "GSM102963", "GSM102964"],
        "Title": ["Sample1", "Sample2", "Sample3", "Sample4"],
        "DiagnosisPrimary": ["Healthy", "Healthy", "Disease", "Disease"],
        "Platform_id": ["GPL570", "GPL570", "GPL570", "GPL570"],
        "AnyInfect.1yes.0no": [0, 0, 1, 0],
    }

    type3_data_modified = {
        "Unnamed: 0": ["GSM102961", "GSM102962", "GSM102963", "GSM102964"],
        "Title": ["Sample1", "Sample2", "Sample3", "Sample4"],
        "DiagnosisPrimary": ["Healthy", "Healthy", "Disease", "Disease"],
        "Platform_id": ["GPL570", "GPL570", "GPL570", "GPL570"],
        "AnyInfect.1yes.0no": [0, 0, 1, 1],  # Different value
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test Type 1 files (sample_id column)
        file1_type1 = Path(tmpdir) / "type1_file1.csv"
        file2_type1 = Path(tmpdir) / "type1_file2.csv"

        pd.DataFrame(type1_data).to_csv(file1_type1, index=False)
        pd.DataFrame(type1_data_modified).to_csv(file2_type1, index=False)

        print("=== Test 1: Type 1 files (sample_id column) ===")
        result, diff_df = compare_csvs(file1_type1, file2_type1, rtol=1e-6, debug=True)
        print(f"Result: {result}\n")

        # Test Type 3 files (Unnamed: 0 column)
        file1_type3 = Path(tmpdir) / "type3_file1.csv"
        file2_type3 = Path(tmpdir) / "type3_file2.csv"

        pd.DataFrame(type3_data).to_csv(file1_type3, index=False)
        pd.DataFrame(type3_data_modified).to_csv(file2_type3, index=False)

        print("=== Test 2: Type 3 files (Unnamed: 0 column) ===")
        result, diff_df = compare_csvs(file1_type3, file2_type3, rtol=1e-6, debug=True)
        print(f"Result: {result}\n")

        # Test gene expression data (Type 2) - simulated
        # Create a CSV similar to your gene expression format
        gene_data1 = pd.DataFrame(
            {
                "Unnamed: 0": ["GSM102961", "GSM102962"],
                "A1BG": [3.0411, 2.2604],
                "A2M": [2.6134, 2.1067],
                "AACS": [4.9823, 4.4646],
            }
        )

        gene_data2 = pd.DataFrame(
            {
                "Unnamed: 0": ["GSM102961", "GSM102962"],
                "A1BG": [3.0411, 2.2604],
                "A2M": [2.6134, 2.1067],
                "AACS": [4.9823, 4.4647],  # Small difference
            }
        )

        file1_genes = Path(tmpdir) / "genes_file1.csv"
        file2_genes = Path(tmpdir) / "genes_file2.csv"

        gene_data1.to_csv(file1_genes, index=False)
        gene_data2.to_csv(file2_genes, index=False)

        print("=== Test 3: Gene expression files (Type 2 simulation) ===")
        result, diff_df = compare_csvs(file1_genes, file2_genes, rtol=1e-6, debug=True)
        print(f"Result: {result}")
        if diff_df is not None:
            print("Differences found:")
            print(diff_df.head())


if __name__ == "__main__":
    # Test CSV file comparison
    # test_compare_csvs()
    main()
