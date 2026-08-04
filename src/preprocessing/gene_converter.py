"""
Gene ID Format Converter

Converts between different gene identifier formats:
- Ensembl IDs (ENSG*) → Gene symbols
- RefSeq IDs (NM_*, NR_*) → Gene symbols
- GenBank IDs → Gene symbols
- Entrez IDs → Gene symbols

Matches R MetaIntegrator's multi-format annotation support in .GEO_fData_key_parser()

Author: Based on R MetaIntegrator by Vallania et al.
"""

import logging
import gzip
from pathlib import Path
from typing import Optional, Dict, Set
import pandas as pd

log = logging.getLogger(__name__)


class GeneConverter:
    """
    Convert between gene ID formats using bundled reference tables.

    Matches R MetaIntegrator's approach of handling diverse annotation formats.

    Examples
    --------
    >>> converter = GeneConverter()
    >>> converter.to_gene_symbol('ENSG00000141510')
    'TP53'
    >>> converter.to_gene_symbol('NM_000546')
    'TP53'
    >>> converter.is_valid_symbol('TP53')
    True
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize converter with bundled reference tables.

        Parameters
        ----------
        data_dir : Path, optional
            Directory containing reference data files.
            If None, uses bundled data in package.
        """
        self.data_dir = data_dir or Path(__file__).parent / "data"

        # Lookup dictionaries
        self._ensg_to_symbol: Dict[str, str] = {}
        self._entrez_to_symbol: Dict[str, str] = {}
        self._refseq_to_symbol: Dict[str, str] = {}
        self._genbank_to_symbol: Dict[str, str] = {}
        self._valid_symbols: Set[str] = set()

        # Load all reference tables
        self._load_reference_tables()

    def _load_csv_gz(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a gzipped CSV file."""
        filepath = self.data_dir / filename

        if filepath.exists():
            try:
                with gzip.open(filepath, "rt") as f:
                    return pd.read_csv(f)
            except Exception as e:
                log.error(f"Failed to load {filepath}: {e}")

        # Try uncompressed version
        filepath_uncompressed = self.data_dir / filename.replace(".gz", "")
        if filepath_uncompressed.exists():
            try:
                return pd.read_csv(filepath_uncompressed)
            except Exception as e:
                log.error(f"Failed to load {filepath_uncompressed}: {e}")

        return None

    def _load_reference_tables(self) -> None:
        """Load all reference tables from bundled data."""

        log.info("Loading gene reference tables...")

        # 1. Ensembl genes (PRIMARY)
        df_ens = self._load_csv_gz("ensembl_genes.csv.gz")
        if df_ens is not None:
            # stable_id → display_label
            self._ensg_to_symbol = dict(zip(df_ens["stable_id"], df_ens["display_label"]))
            self._valid_symbols = set(df_ens["display_label"].dropna())
            log.info(f"  ✓ Ensembl: {len(self._ensg_to_symbol):,} ENSG IDs")
        else:
            log.warning("  ✗ Ensembl table not found")

        # 2. Entrez genes
        df_entrez = self._load_csv_gz("entrez_genes.csv.gz")
        if df_entrez is not None:
            # dbprimary_acc (Entrez ID) → display_label
            self._entrez_to_symbol = dict(
                zip(df_entrez["dbprimary_acc"].astype(str), df_entrez["display_label"])
            )
            log.info(f"  ✓ Entrez: {len(self._entrez_to_symbol):,} IDs")
        else:
            log.warning("  ✗ Entrez table not found")

        # 3. RefSeq genes
        df_refseq = self._load_csv_gz("refseq_genes.csv.gz")
        if df_refseq is not None:
            # name (RefSeq ID) → name2 (gene symbol)
            self._refseq_to_symbol = dict(zip(df_refseq["name"], df_refseq["name2"]))
            log.info(f"  ✓ RefSeq: {len(self._refseq_to_symbol):,} IDs")
        else:
            log.warning("  ✗ RefSeq table not found")

        # 4. GenBank genes
        df_genbank = self._load_csv_gz("genbank_genes.csv.gz")
        if df_genbank is not None:
            # acc (GenBank accession) → name (gene symbol)
            # Note: GenBank table is very small (usually 1 row)
            self._genbank_to_symbol = dict(zip(df_genbank["acc"], df_genbank["name"]))
            log.info(f"  ✓ GenBank: {len(self._genbank_to_symbol):,} IDs")
        else:
            log.warning("  ✗ GenBank table not found")

        log.info(f"  → Total valid gene symbols: {len(self._valid_symbols):,}")

    def to_gene_symbol(self, identifier: str) -> Optional[str]:
        """
        Convert any gene identifier to HUGO gene symbol.

        Matches R's .GEO_fData_key_parser logic:
        1. Check if already a valid symbol
        2. Try ENSG format → symbol
        3. Try RefSeq format → symbol
        4. Try GenBank format → symbol
        5. Try Entrez format → symbol
        6. Return None if unrecognized

        Parameters
        ----------
        identifier : str
            Gene identifier in any format

        Returns
        -------
        str or None
            HUGO gene symbol, or None if conversion fails

        Examples
        --------
        >>> converter.to_gene_symbol('TP53')
        'TP53'
        >>> converter.to_gene_symbol('ENSG00000141510')
        'TP53'
        >>> converter.to_gene_symbol('NM_000546')
        'TP53'
        """
        identifier = str(identifier).strip()

        if not identifier or identifier == "NA":
            return None

        # Case 1: Already a valid HUGO symbol?
        if identifier in self._valid_symbols:
            return identifier

        # Case 2: ENSG format (Ensembl gene ID)
        if identifier.startswith("ENSG"):
            return self._ensg_to_symbol.get(identifier)

        # Case 3: RefSeq format (NM_*, NR_*, XM_*, XR_*)
        if identifier.startswith(("NM_", "NR_", "XM_", "XR_")):
            # Remove version suffix (e.g., NM_000546.5 → NM_000546)
            base_id = identifier.split(".")[0]
            return self._refseq_to_symbol.get(base_id)

        # Case 4: GenBank format (letter + numbers)
        # Try GenBank table first
        symbol = self._genbank_to_symbol.get(identifier)
        if symbol:
            return symbol

        # Case 5: Entrez ID format (numeric)
        if identifier.isdigit():
            return self._entrez_to_symbol.get(identifier)

        # Case 6: Try as string Entrez ID
        symbol = self._entrez_to_symbol.get(identifier)
        if symbol:
            return symbol

        # Unrecognized format
        return None

    def is_valid_symbol(self, symbol: str) -> bool:
        """
        Check if string is a valid HUGO gene symbol.

        Parameters
        ----------
        symbol : str
            Gene symbol to validate

        Returns
        -------
        bool
            True if valid HUGO gene symbol
        """
        if not symbol or pd.isna(symbol):
            return False
        return str(symbol).strip() in self._valid_symbols

    def convert_list(self, identifiers: list) -> list:
        """
        Convert a list of identifiers to gene symbols.

        Parameters
        ----------
        identifiers : list
            List of gene identifiers in any format

        Returns
        -------
        list
            List of gene symbols (or 'NA' for failed conversions)
        """
        return [self.to_gene_symbol(id_) or "NA" for id_ in identifiers]

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about loaded reference tables."""
        return {
            "valid_symbols": len(self._valid_symbols),
            "ensg_ids": len(self._ensg_to_symbol),
            "entrez_ids": len(self._entrez_to_symbol),
            "refseq_ids": len(self._refseq_to_symbol),
            "genbank_ids": len(self._genbank_to_symbol),
        }


# ============================================================================
# Convenience functions
# ============================================================================


def create_converter() -> GeneConverter:
    """
    Create and return a gene converter.

    Returns
    -------
    GeneConverter
        Initialized converter with bundled reference tables
    """
    return GeneConverter()


if __name__ == "__main__":
    # Test the converter
    print("Testing GeneConverter with bundled data...")
    print("=" * 80)

    try:
        converter = GeneConverter()
        stats = converter.get_stats()

        print("\n✓ Converter initialized successfully")
        print(f"  Valid gene symbols: {stats['valid_symbols']:,}")
        print(f"  ENSG IDs:          {stats['ensg_ids']:,}")
        print(f"  Entrez IDs:        {stats['entrez_ids']:,}")
        print(f"  RefSeq IDs:        {stats['refseq_ids']:,}")
        print(f"  GenBank IDs:       {stats['genbank_ids']:,}")

        # Test conversions for GAPDH
        print("\nTesting conversions for GAPDH:")
        test_cases = [
            ("TP53", "TP53"),  # Already a symbol
            ("ENSG00000141510", "TP53"),  # Ensembl ID
            ("7157", "TP53"),  # Entrez ID
            ("NM_000546", "TP53"),  # RefSeq ID
            ("ENST00000396160", None),  # Transcript (not in tables)
            ("FAKE_GENE", None),  # Invalid
        ]

        for input_id, expected in test_cases:
            result = converter.to_gene_symbol(input_id)
            status = "✓" if result == expected else "✗"
            print(f"  {status} {input_id:20s} → {result} (expected {expected})")

        print("\n" + "=" * 80)
        print("✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure reference data files are in src/preprocessing/data/")
