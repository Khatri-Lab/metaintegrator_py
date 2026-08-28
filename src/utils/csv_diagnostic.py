#!/usr/bin/env python3

import sys
from pathlib import Path

import pandas as pd


def diagnose_csv_files(csv_directory: str) -> None:
    """
    Diagnose CSV files to understand sample name mismatch issues.
    """
    csv_dir = Path(csv_directory)

    print(f"🔍 Diagnosing CSV files in: {csv_dir}")
    print("=" * 60)

    # Find all CSV files
    csv_files = list(csv_dir.glob("*.csv"))

    # Group by dataset
    datasets = {}
    for file_path in csv_files:
        filename = file_path.name

        if "_genes.csv" in filename:
            dataset = filename.replace("GSEs.test_", "").replace("_genes.csv", "")
            if dataset not in datasets:
                datasets[dataset] = {}
            datasets[dataset]["genes"] = file_path

        elif "_pheno.csv" in filename:
            dataset = filename.replace("GSEs.test_", "").replace("_pheno.csv", "")
            if dataset not in datasets:
                datasets[dataset] = {}
            datasets[dataset]["pheno"] = file_path

    # Analyze each dataset
    for dataset_name, files in datasets.items():
        print(f"\n📊 Dataset: {dataset_name}")
        print("-" * 40)

        if "genes" not in files or "pheno" not in files:
            print("❌ Incomplete dataset (missing genes or pheno file)")
            continue

        # Load and analyze genes file
        print(f"\n🧬 Genes file: {files['genes'].name}")
        try:
            # Read first few lines to see structure
            with open(files["genes"], "r") as f:
                lines = f.readlines()[:5]
                print("First 5 lines:")
                for i, line in enumerate(lines):
                    print(f"  {i + 1}: {line.strip()[:100]}...")

            # Load with pandas
            genes_df = pd.read_csv(files["genes"], index_col=0)
            print(f"Shape: {genes_df.shape}")
            print(f"Index (genes): {list(genes_df.index[:3])}...")
            print(f"Columns (samples): {list(genes_df.columns)}")
            print(f"Column types: {[type(col).__name__ for col in genes_df.columns[:3]]}")

        except Exception as e:
            print(f"❌ Error reading genes file: {e}")
            continue

        # Load and analyze pheno file
        print(f"\n📋 Pheno file: {files['pheno'].name}")
        try:
            # Read first few lines to see structure
            with open(files["pheno"], "r") as f:
                lines = f.readlines()[:5]
                print("First 5 lines:")
                for i, line in enumerate(lines):
                    print(f"  {i + 1}: {line.strip()}")

            # Load with pandas
            pheno_df = pd.read_csv(files["pheno"], index_col=0)
            print(f"Shape: {pheno_df.shape}")
            print(f"Index (samples): {list(pheno_df.index)}")
            print(f"Index types: {[type(idx).__name__ for idx in pheno_df.index[:3]]}")
            print(f"Columns: {list(pheno_df.columns)}")

        except Exception as e:
            print(f"❌ Error reading pheno file: {e}")
            continue

        # Compare sample names
        print(f"\n🔍 Sample Name Comparison:")
        genes_samples = set(str(col) for col in genes_df.columns)
        pheno_samples = set(str(idx) for idx in pheno_df.index)

        print(f"Genes samples: {sorted(genes_samples)}")
        print(f"Pheno samples: {sorted(pheno_samples)}")

        common = genes_samples.intersection(pheno_samples)
        genes_only = genes_samples - pheno_samples
        pheno_only = pheno_samples - genes_samples

        print(f"\nCommon samples ({len(common)}): {sorted(common)}")
        if genes_only:
            print(f"Only in genes ({len(genes_only)}): {sorted(genes_only)}")
        if pheno_only:
            print(f"Only in pheno ({len(pheno_only)}): {sorted(pheno_only)}")

        # Check for whitespace issues
        print(f"\n🔍 Checking for whitespace issues:")
        genes_samples_repr = [repr(str(col)) for col in genes_df.columns]
        pheno_samples_repr = [repr(str(idx)) for idx in pheno_df.index]

        print(f"Genes samples (with quotes): {genes_samples_repr}")
        print(f"Pheno samples (with quotes): {pheno_samples_repr}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 csv_diagnostic.py <csv_directory>")
        sys.exit(1)

    diagnose_csv_files(sys.argv[1])
