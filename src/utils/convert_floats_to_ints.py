#!/usr/bin/env python3
"""
convert_floats_to_ints.py

A tiny utility that reads a CSV file (with all fields quoted if needed),
identifies columns that contain only whole‑number floats (plus missing values),
and casts those columns to the nullable‑integer dtype (Int64).

Usage
-----
    python convert_floats_to_ints.py \
        --input  path/to/file.csv \
        --output path/to/out.csv   # optional; if omitted, prints to stdout

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _convert_column_to_int64(series: pd.Series) -> pd.Series:
    """
    Convert a Series that contains only whole‑number floats (plus NaNs) to the nullable‑integer dtype Int64.

    Args:
        series : pd.Series
            Series that is assumed to be numeric (floats) or NaN.

    Returns:
        pd.Series
            Same data but with dtype Int64.
    """
    # Cast to the nullable‑integer type.
    # NaNs become pandas.NA automatically.
    return series.astype("Int64")


def _maybe_convert_series(series: pd.Series) -> pd.Series:
    """
    Inspect a Series.  If it is *numeric* and every non‑null value is a whole number, cast it to Int64.
    Otherwise, return it unchanged.

    The function also normalises missing values:
      * the literal string "NA" (case‑insensitive) → np.nan
      * empty strings → np.nan
    """
    # Normalize the missing values
    # (string → np.nan) – works for all string columns
    # The replace is cheap; only runs if needed.

    # 1. Normalize missing values (strings → np.nan)
    if series.dtype == object:
        series = series.replace(
            {"NA": np.nan, "": np.nan, "na": np.nan, "Na": np.nan, "nA": np.nan}
        )

    # 2. Try to coerce to numeric; non‑numeric → NaN
    numeric_series = pd.to_numeric(series, errors="coerce")

    # 3. If there are any numeric values…
    if numeric_series.notna().any():
        # …check if *all* of them are whole numbers.
        valid = numeric_series.notna()

        """
        Why round()?
        If the column really contains only values like 1.0, 0.0, 2.0, rounding them produces the same numbers 
        but ensures that they are exactly representable as integers. 
        The comparison (np.isclose) is then performed on the valid (non‑NA) part only.
        """
        whole = np.isclose(numeric_series[valid], numeric_series[valid].round())
        if whole.all():
            # Convert to nullable‑int (keeps the NaNs)
            return numeric_series.round().astype("Int64")

    # 4. In every other case return the original (possibly altered) Series
    return series


def convert_floats_to_int64(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the per‑column conversion logic to an entire DataFrame.

    Args;
        df : pd.DataFrame
            Input DataFrame read from a CSV.

    Returns:
        pd.DataFrame
            A new DataFrame with columns that were whole‑number floats cast to Int64; all other columns unchanged.
    """
    return df.apply(_maybe_convert_series, axis=0)


def run_convert_floats_to_int64(
    input_path: str, output_path: str | None = None, debug: bool = False
):
    """
    Reads a CSV file, transforms float columns that are whole numbers into int64,
    and writes the resulting modified data to either an output file or standard output.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str | None): The path to save the transformed CSV file. If None,
            the resulting data is written to standard output.
        debug (bool): If True, logs messages about processing details to stderr.

    Raises:
        FileNotFoundError: If the input file does not exist or cannot be accessed.
        ValueError: If an error occurs while reading or writing the CSV file.
    """
    # 1. Read the file – start with everything as string to preserve precision/format
    df = pd.read_csv(
        input_path,
        header=0,
        dtype=str,
    )

    # 2. Convert the whole‑number‑float columns
    df = convert_floats_to_int64(df)

    # 3. Write the result
    if output_path:
        df.to_csv(output_path, index=False)
        if debug:
            sys.stderr.write(f"Wrote transformed CSV to {output_path}\n")
    else:
        # Print to stdout if no output path is provided
        print(f"Transformed CSV: {df.shape[0]} rows, {df.shape[1]} columns; {df.head(3)}")


# --------------------------------------------------------------------------- #
#  Main routine
# --------------------------------------------------------------------------- #
def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Read a CSV and cast columns that are whole‑number floats to Int64."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the transformed CSV.  If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information to stderr.",
    )

    args = parser.parse_args(argv)

    if not args.input.is_file():
        sys.exit(f"ERROR: {args.input} is not a file.")

    run_convert_floats_to_int64(args.input, args.output, args.debug)


if __name__ == "__main__":
    main()
