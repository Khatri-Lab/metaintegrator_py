"""
Gene symbol extraction from GPL platform annotation tables.

Replicates R MetaIntegrator's .GEO_fData_key_parser() logic for mapping
microarray probe IDs to HGNC gene symbols with optional Ensembl validation.

This module is platform-agnostic: it operates on a pandas DataFrame (the
GPL annotation table) and a probe ID index, with no dependency on GEOparse,
network I/O, or file formats.  That makes it straightforward to unit-test
with small, in-memory DataFrames.

Usage
-----
    from preprocessing.gene_annotator import GeneAnnotator

    annotator = GeneAnnotator()
    keys = annotator.annotate(gpl_table, probe_ids)

    # With Ensembl validation
    from preprocessing.gene_validation import EnsemblGeneValidator
    validator = EnsemblGeneValidator()
    keys = annotator.annotate(gpl_table, probe_ids, validator=validator)
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Set

import pandas as pd

log = logging.getLogger(__name__)


class GeneAnnotator:
    """
    Map microarray probe IDs to gene symbols using GPL annotation tables.

    Implements R MetaIntegrator's .GEO_fData_key_parser() logic:

    1. Locate the gene-symbol column in the GPL table by trying a
       prioritised list of known column names.
    2. For *direct* symbol columns (``Gene Symbol``, ``Symbol``, ...),
       replace ``///`` and ``;`` separators with commas, keeping the
       full multi-gene string -- matching R's ``gsub``.
    3. For *assignment* columns (``gene_assignment``, ``mrna_assignment``),
       extract the gene symbol from ``field[1]`` of the first ``//``-
       delimited entry -- matching R's Case-2 extraction.
    4. Optionally validate the first gene in each symbol against an
       Ensembl reference set; set to ``'NA'`` if not found.
    5. Reject GenBank/EMBL accession IDs that are not gene symbols,
       matching R's behavior of only keeping proper HGNC symbols.
    """

    # Ordered candidate column names -- first match wins.
    SYMBOL_COLUMNS: List[str] = [
        "Gene Symbol",
        "GENE_SYMBOL",
        "Gene_Symbol",
        "gene_symbol",
        "genesymbol",
        "Symbol",
        "SYMBOL",
        "symbol",
        "HUGO_Gene_Nomenclature",
        "gene_assignment",
        "Gene",
        "GENE",
        "gene",
        "ORF",
    ]

    # Columns that use the ``accession // symbol // description`` layout.
    ASSIGNMENT_COLUMNS: frozenset = frozenset(
        {
            "gene_assignment",
            "mrna_assignment",
        }
    )

    # Accession-ID pattern -- entries that match are skipped (not gene symbols).
    ACCESSION_RE: re.Pattern = re.compile(r"^(ENST|NM_)[0-9]+$")

    # GenBank/EMBL accession patterns -- NOT gene symbols, must be rejected.
    # R's annotator does not include these; without this filter Python produces
    # ~3,400 extra entries (AF*, AK*, AC*, etc.) that inflate the gene count
    # from ~25,800 to ~29,200, causing FDR divergence vs R.
    GENBANK_ACCESSION_RE: re.Pattern = re.compile(
        r"^(AF|AK|AB|AC|AJ|AX|AL|AM|BC|BX|BT|CR|CT|DQ|EU|FJ|FN|HQ|JN|JQ|"
        r"KC|KF|KJ|KP|KT|KU|KX|KY|LC|LN|LT|MF|MG|MH|MK|MN|MT|MW|MZ|"
        r"OK|OM|ON|OP|OQ|OR)\d{5,}",
        re.IGNORECASE,
    )

    # Values treated as missing / empty.
    NA_SENTINELS: frozenset = frozenset({"", "---", "NA", "na", "N/A"})

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def annotate(
        self,
        gpl_table: pd.DataFrame,
        probe_ids: pd.Index,
        validator: object | None = None,
    ) -> List[str]:
        """
        Map *probe_ids* to gene symbols using *gpl_table*.

        Parameters
        ----------
        gpl_table : pd.DataFrame
            GPL annotation table.
        probe_ids : pd.Index
            Probe identifiers from the expression matrix.
        validator : object with ``valid_genes: Set[str]``, optional.
            When supplied, symbols whose first gene is absent from the
            set are replaced with ``'NA'``.

        Returns
        -------
        list[str]
            Gene symbols aligned 1-to-1 with *probe_ids*.
        """
        gene_col = self.find_symbol_column(gpl_table)
        if gene_col is None:
            log.warning("No gene-symbol column found in GPL table")
            return [str(p) for p in probe_ids]

        log.info(f"    Using column '{gene_col}' for gene symbols")

        is_assignment = gene_col in self.ASSIGNMENT_COLUMNS
        mapping = self._build_probe_mapping(gpl_table, gene_col)

        valid_genes: Optional[Set[str]] = None
        if validator is not None and hasattr(validator, "valid_genes"):
            valid_genes = validator.valid_genes

        keys: List[str] = []
        n_ensembl_filtered = 0
        n_accession_filtered = 0

        for probe in probe_ids:
            raw = mapping.get(str(probe))
            symbol = self.clean_symbol(raw, is_assignment)

            # R-parity: Ensembl validation on *first* gene only
            if valid_genes is not None and symbol != "NA":
                first_gene = symbol.split(",")[0].strip()
                if first_gene not in valid_genes:
                    symbol = "NA"
                    n_ensembl_filtered += 1

            # R-parity: reject GenBank/EMBL accession IDs that are NOT gene
            # symbols.  R's annotator does not include these; without this
            # filter Python produces ~3,400 extra entries that inflate the
            # gene count and FDR values.
            if symbol != "NA":
                first_gene = symbol.split(",")[0].strip()
                if self.GENBANK_ACCESSION_RE.match(first_gene):
                    symbol = "NA"
                    n_accession_filtered += 1

            keys.append(symbol)

        if n_ensembl_filtered > 0:
            log.info(f"    Ensembl-filtered: {n_ensembl_filtered} probes")

        if n_accession_filtered > 0:
            log.info(
                f"    Accession-filtered: {n_accession_filtered} probes "
                f"(GenBank/EMBL IDs rejected)"
            )

        return keys

    # ------------------------------------------------------------------ #
    #  Column detection                                                   #
    # ------------------------------------------------------------------ #

    def find_symbol_column(self, table: pd.DataFrame) -> Optional[str]:
        """Return the first recognised gene-symbol column in *table*."""
        for col in self.SYMBOL_COLUMNS:
            if col in table.columns:
                return col
        return None

    # ------------------------------------------------------------------ #
    #  Symbol cleaning                                                    #
    # ------------------------------------------------------------------ #

    def clean_symbol(self, raw: object, is_assignment: bool) -> str:
        """
        Clean a single raw annotation value into a gene-symbol string.

        Returns comma-separated symbols for multi-gene probes, or
        ``'NA'`` if the value is missing / unrecognised.
        """
        if pd.isna(raw) or str(raw).strip() in self.NA_SENTINELS:
            return "NA"

        raw_str = str(raw).strip()
        if not raw_str:
            return "NA"

        # Split on ``///`` or ``;``  (R's gsub(";| /// ", ",", keys))
        entries = re.split(r"\s*///\s*|;", raw_str)
        entries = [e.strip() for e in entries if e.strip()]

        collected: List[str] = []

        for entry in entries:
            fields = [f.strip() for f in entry.split("//") if f.strip()]
            if not fields:
                continue

            if is_assignment:
                # Assignment column: field[0]=accession, field[1]=symbol
                if len(fields) >= 2:
                    candidate = fields[1].strip()
                    candidate = candidate.split()[0] if " " in candidate else candidate
                    if candidate and candidate not in self.NA_SENTINELS:
                        collected.append(candidate)
                else:
                    first = fields[0].strip()
                    if (
                        first
                        and first not in self.NA_SENTINELS
                        and not self.ACCESSION_RE.match(first)
                    ):
                        collected.append(first)
                # R Case 2: only extract from the *first* ``///`` entry
                break
            else:
                # Direct symbol column: keep full string (R-parity)
                first = fields[0].strip()
                if self.ACCESSION_RE.match(first):
                    continue
                if first and first not in self.NA_SENTINELS:
                    collected.append(first)

        if not collected:
            return "NA"
        return ",".join(collected)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_probe_mapping(table: pd.DataFrame, gene_col: str) -> dict:
        """Build a ``{str(probe_id): raw_value}`` lookup from a GPL table."""
        if "ID" in table.columns:
            return {str(k): v for k, v in table.set_index("ID")[gene_col].to_dict().items()}
        return {str(k): v for k, v in zip(table.index, table[gene_col])}
