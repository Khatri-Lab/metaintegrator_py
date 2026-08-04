"""
Ensembl Gene Symbol Validation - Updated to use bundled reference data

This module loads gene reference tables bundled with the package,
matching R MetaIntegrator's approach for robust offline validation.

Author: Based on R MetaIntegrator by Vallania et al.
"""

import logging
import gzip
from pathlib import Path
from typing import Set, Optional, Dict, Tuple
import pandas as pd

log = logging.getLogger(__name__)


class EnsemblGeneValidator:
    """
    Validates gene symbols against bundled Ensembl reference database.

    This matches R MetaIntegrator's validation using ens_ensgID_table.
    Data is bundled with the package for offline use and reproducibility.

    Examples
    --------
    >>> validator = EnsemblGeneValidator()
    >>> validator.is_valid('TP53')
    True
    >>> validator.is_valid('ENST00000396160')  # Transcript ID
    False
    >>> validator.is_valid('BC042590')  # GenBank ID
    False
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize validator with bundled Ensembl gene reference.

        Parameters
        ----------
        data_dir : Path, optional
            Directory containing reference data files.
            If None, uses bundled data in package.
        """
        self.valid_genes: Set[str] = set()
        self.data_dir = data_dir or Path(__file__).parent / "data"
        self._load_bundled_data()

    def _load_bundled_data(self) -> None:
        """Load bundled Ensembl gene reference from package data."""

        # Try compressed CSV first (preferred)
        csv_gz_path = self.data_dir / "ensembl_genes.csv.gz"
        csv_path = self.data_dir / "ensembl_genes.csv"
        txt_path = self.data_dir / "ensembl_gene_symbols.txt"

        # Option 1: Load from compressed CSV
        if csv_gz_path.exists():
            try:
                log.info(f"Loading bundled Ensembl genes from {csv_gz_path}")
                with gzip.open(csv_gz_path, "rt") as f:
                    df = pd.read_csv(f)

                # Extract unique gene symbols from display_label column
                self.valid_genes = set(df["display_label"].dropna().unique())
                log.info(f"✓ Loaded {len(self.valid_genes)} valid gene symbols from bundled data")
                return

            except Exception as e:
                log.error(f"Failed to load {csv_gz_path}: {e}")

        # Option 2: Load from uncompressed CSV
        elif csv_path.exists():
            try:
                log.info(f"Loading bundled Ensembl genes from {csv_path}")
                df = pd.read_csv(csv_path)
                self.valid_genes = set(df["display_label"].dropna().unique())
                log.info(f"✓ Loaded {len(self.valid_genes)} valid gene symbols from bundled data")
                return

            except Exception as e:
                log.error(f"Failed to load {csv_path}: {e}")

        # Option 3: Load from simple text file
        elif txt_path.exists():
            try:
                log.info(f"Loading bundled Ensembl genes from {txt_path}")
                with open(txt_path, "r") as f:
                    self.valid_genes = {line.strip() for line in f if line.strip()}
                log.info(f"✓ Loaded {len(self.valid_genes)} valid gene symbols from text file")
                return

            except Exception as e:
                log.error(f"Failed to load {txt_path}: {e}")

        # If no bundled data found, raise error
        log.error(f"No bundled gene data found in {self.data_dir}")
        log.error(
            "Expected files: ensembl_genes.csv.gz, ensembl_genes.csv, or ensembl_gene_symbols.txt"
        )
        raise FileNotFoundError(
            f"Bundled gene reference data not found in {self.data_dir}. "
            "Please ensure reference data is properly installed."
        )

    def is_valid(self, gene_symbol: str) -> bool:
        """
        Check if gene symbol is valid according to Ensembl.

        Parameters
        ----------
        gene_symbol : str
            Gene symbol to validate

        Returns
        -------
        bool
            True if gene is in Ensembl database, False otherwise
        """
        if not gene_symbol or pd.isna(gene_symbol):
            return False

        # Check exact match (case-sensitive, HUGO symbols are uppercase)
        return str(gene_symbol).strip() in self.valid_genes

    def validate_list(self, gene_symbols: list) -> Tuple[list, list]:
        """
        Validate a list of gene symbols.

        Parameters
        ----------
        gene_symbols : list
            List of gene symbols to validate

        Returns
        -------
        valid_genes : list
            Gene symbols that passed validation
        invalid_genes : list
            Gene symbols that failed validation
        """
        valid = []
        invalid = []

        for gene in gene_symbols:
            if self.is_valid(gene):
                valid.append(gene)
            else:
                invalid.append(gene)

        return valid, invalid

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about loaded gene reference."""
        return {"total_genes": len(self.valid_genes), "data_source": "bundled_ensembl"}


# ============================================================================
# Integration Functions for Notebook/Pipeline Use
# ============================================================================


def apply_ensembl_validation(
    dataset_dict: dict, validator: EnsemblGeneValidator, verbose: bool = True
) -> dict:
    """
    Apply Ensembl validation to a dataset's gene symbols.

    This matches R's behavior in .GEO_fData_key_parser() where genes
    not in ens_ensgID_table$display_label are set to NA.

    Parameters
    ----------
    dataset_dict : dict
        Dataset from sle_data['originalData']
    validator : EnsemblGeneValidator
        Initialized validator
    verbose : bool
        Print statistics

    Returns
    -------
    dict
        Updated dataset with only Ensembl-validated genes
    """
    if "keys" not in dataset_dict:
        return dataset_dict

    keys = dataset_dict["keys"]

    # ================================================================
    # R PARITY: Validate only the FIRST gene in comma-separated keys
    # ================================================================
    new_keys = []
    n_before = 0
    n_removed = 0
    removed_genes = []

    for k in keys:
        if k is None or k == "NA" or pd.isna(k):
            new_keys.append("NA")
            continue

        n_before += 1
        k_str = str(k).strip()

        # Extract the FIRST gene from comma-separated key
        first_gene = k_str.split(",")[0].strip()

        # Validate only the first gene (R parity)
        if validator.is_valid(first_gene):
            new_keys.append(k_str)
        else:
            new_keys.append("NA")
            n_removed += 1
            if len(removed_genes) < 15:
                removed_genes.append(k_str)

    dataset_dict["keys"] = new_keys

    # Statistics
    n_after = n_before - n_removed

    if verbose:
        log.info(
            f"Ensembl validation (R-parity, first-gene-only): "
            f"{n_before} genes → {n_after} genes "
            f"({n_removed} invalid removed)"
        )

        if removed_genes:
            examples = removed_genes[:10]
            log.info(f"  Examples of removed genes: {examples}")

            # Categorize removed genes for diagnostics
            transcript_ids = [g for g in removed_genes if g.startswith("ENST")]
            refseq_ids = [g for g in removed_genes if g.startswith("NM_") or g.startswith("NR_")]
            genbank_ids = [
                g for g in removed_genes if len(g) > 2 and g[:2].isalpha() and g[2:].isdigit()
            ]

            if transcript_ids:
                log.info(f"    Transcript IDs (ENST*): {len(transcript_ids)} removed")
            if refseq_ids:
                log.info(f"    RefSeq IDs (NM_*, NR_*): {len(refseq_ids)} removed")
            if genbank_ids:
                log.info(f"    GenBank IDs: {len(genbank_ids)} removed")

    return dataset_dict


def validate_all_datasets(
    data_dict: dict,
    validator: Optional[EnsemblGeneValidator] = None,
    verbose: bool = True,
) -> dict:
    """
    Apply Ensembl validation to all datasets in a MetaIntegrator object.

    Parameters
    ----------
    data_dict : dict
        MetaIntegrator data structure with 'originalData' key
    validator : EnsemblGeneValidator, optional
        Validator instance. If None, creates new one.
    verbose : bool
        Print statistics

    Returns
    -------
    dict
        Updated data structure with validated genes
    """
    if validator is None:
        validator = EnsemblGeneValidator()

    if verbose:
        stats = validator.get_stats()
        log.info(f"Using Ensembl reference with {stats['total_genes']} genes")
        log.info("=" * 80)

    total_before = 0
    total_after = 0
    total_removed = 0

    for dataset_name, dataset in data_dict.get("originalData", {}).items():
        if verbose:
            log.info(f"\n{dataset_name}:")

        # Count before
        before_count = sum(1 for k in dataset.get("keys", []) if k != "NA")
        total_before += before_count

        # Apply validation
        dataset = apply_ensembl_validation(dataset, validator, verbose=verbose)

        # Update in dict
        data_dict["originalData"][dataset_name] = dataset

        # Count after
        after_count = sum(1 for k in dataset.get("keys", []) if k != "NA")
        total_after += after_count
        removed = before_count - after_count
        total_removed += removed

        if verbose:
            log.info(f"  Before: {before_count} genes")
            log.info(f"  After:  {after_count} genes")
            log.info(f"  Removed: {removed} genes")

    if verbose:
        log.info("\n" + "=" * 80)
        log.info("VALIDATION SUMMARY")
        log.info("=" * 80)
        log.info(f"Total genes before: {total_before}")
        log.info(f"Total genes after:  {total_after}")
        log.info(f"Total removed:      {total_removed}")
        log.info(f"Removal rate:       {100*total_removed/total_before:.1f}%")
        log.info("=" * 80)

    return data_dict


# ============================================================================
# Convenience function for notebook use
# ============================================================================


def create_validator() -> EnsemblGeneValidator:
    """
    Create and return an Ensembl gene validator.

    This is a convenience function for notebook use.

    Returns
    -------
    EnsemblGeneValidator
        Initialized validator with bundled gene reference

    Examples
    --------
    >>> validator = create_validator()
    >>> validator.is_valid('TP53')
    True
    """
    return EnsemblGeneValidator()


if __name__ == "__main__":
    # Test the validator
    print("Testing EnsemblGeneValidator with bundled data...")
    print("=" * 80)

    try:
        validator = EnsemblGeneValidator()
        stats = validator.get_stats()
        print(f"\n✓ Validator initialized successfully")
        print(f"  Total genes loaded: {stats['total_genes']}")
        print(f"  Data source: {stats['data_source']}")

        # Test some symbols
        print("\nTesting gene symbols:")
        test_genes = [
            ("TP53", True),
            ("BRCA1", True),
            ("GAPDH", True),
            ("CD6", True),
            ("FCAR", True),
            ("ENST00000396160", False),  # Transcript ID
            ("NM_002046", False),  # RefSeq ID
            ("BC042590", False),  # GenBank ID
            ("FAKE_GENE", False),  # Not a gene
        ]

        for gene, expected in test_genes:
            result = validator.is_valid(gene)
            status = "✓" if result == expected else "✗"
            print(f"  {status} {gene:20s} : {result} (expected {expected})")

        print("\n" + "=" * 80)
        print("✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure reference data files are in src/preprocessing/data/")
        print("Expected files:")
        print("  - ensembl_genes.csv.gz (preferred)")
        print("  - ensembl_genes.csv (alternative)")
        print("  - ensembl_gene_symbols.txt (fallback)")
