"""
Microarray data handler.

Processes microarray data from various sources into standardized format.
Handles both GEOparse (SOFT) and Series Matrix formats.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import DataHandler, ProcessedDataset, RawDataset
from .gene_validation import EnsemblGeneValidator

log = logging.getLogger(__name__)


class MicroarrayHandler(DataHandler):
    """
    Handler for microarray expression data.

    Processes raw microarray data from various sources and converts
    to standardized metaIntegrator format with:
    - Expression matrix (features × samples)
    - Phenotype data (samples × metadata)
    - Gene symbol annotations

    Supports:
    - GEOparse SOFT format (from GEOparseSource)
    - Series Matrix format (from NCBIGeoSource)

    Examples:
        >>> handler = MicroarrayHandler()
        >>> processed = handler.process(raw_dataset)
        >>> print(processed.expr.shape)
        (22645, 118)  # features × samples
    """

    def __init__(self, log_transform: bool = False, normalize: bool = False):
        """Initialize the handler.

        Args:
            log_transform: Apply log2 transformation if the data isn't
                already log-scale.
            normalize: Apply quantile normalization.
        """
        self.log_transform = log_transform
        self.normalize_flag = normalize

    @property
    def data_type(self) -> str:
        """Human-readable name of the data type this handler processes."""
        return "Microarray"

    def parse(self, raw_dataset: RawDataset) -> ProcessedDataset:
        """Parse raw microarray data into the standard format.

        Detects the format (SOFT vs. Series Matrix) automatically and
        applies the matching parsing logic.

        Args:
            raw_dataset: Raw data from a source.

        Returns:
            The parsed dataset in standard format.

        Raises:
            ValueError: If the raw data format is not recognized.
        """
        # Detect format based on metadata
        data_format = raw_dataset.metadata.get("format", "Unknown")

        log.info(
            f"Parsing {raw_dataset.source_id}_{raw_dataset.platform_id} "
            f"(format: {data_format})..."
        )

        if data_format == "SOFT":
            return self._parse_geoparse(raw_dataset)
        elif data_format == "SeriesMatrix":
            return self._parse_series_matrix(raw_dataset)
        else:
            raise ValueError(
                f"Unknown data format: {data_format}. " "Expected 'SOFT' or 'SeriesMatrix'"
            )

    def _parse_geoparse(self, raw_dataset: RawDataset) -> ProcessedDataset:
        """Parse GEOparse SOFT format data."""

        raw_data = raw_dataset.raw_data
        gse = raw_data["gse"]
        gpl = raw_data["gpl"]
        platform_samples = raw_data["samples"]

        sample_ids = [gsm.name for gsm in platform_samples]

        # Build expression matrix (may skip some samples)
        expr_df = self._build_expression_matrix(platform_samples, sample_ids)

        # expr_df may have fewer samples than platform_samples if some
        # were skipped above; keep only the samples that succeeded.
        actual_sample_ids = expr_df.columns.tolist()
        filtered_samples = [gsm for gsm in platform_samples if gsm.name in actual_sample_ids]

        # Build phenotype DataFrame (only for successfully processed samples)
        pheno_df = self._build_phenotype_dataframe(filtered_samples)

        # Align pheno_df to expr_df's sample order.
        pheno_df = pheno_df.loc[actual_sample_ids]

        # Get gene annotations
        gene_symbols_list = self._extract_gene_symbols(gpl, expr_df.index)

        # Ensure gene_symbols_list is a list of the expected length,
        # padding or truncating if the annotation step returned otherwise.
        if not isinstance(gene_symbols_list, list):
            log.warning(f"gene_symbols is {type(gene_symbols_list)}, converting to list")
            gene_symbols_list = list(gene_symbols_list)

        if len(gene_symbols_list) != len(expr_df):
            log.error(
                f"Gene symbols length mismatch: {len(gene_symbols_list)} symbols "
                f"vs {len(expr_df)} probes"
            )
            # Pad or truncate
            if len(gene_symbols_list) < len(expr_df):
                gene_symbols_list.extend([str(p) for p in expr_df.index[len(gene_symbols_list) :]])
            else:
                gene_symbols_list = gene_symbols_list[: len(expr_df)]

        # Create dataset identifier
        dataset_name = f"{raw_dataset.source_id}_{raw_dataset.platform_id}"

        # Create processed dataset
        processed = ProcessedDataset(
            geo_id=dataset_name,
            expr=expr_df,
            pheno=pheno_df,
            keys=gene_symbols_list,  # ← Now a list
            metadata={
                **raw_dataset.metadata,
                "n_features": len(expr_df),
                "n_samples": len(expr_df.columns),
                "n_samples_requested": len(platform_samples),
                "n_samples_skipped": len(platform_samples) - len(expr_df.columns),
                "handler": self.data_type,
            },
        )

        log.info(
            f"  ✓ Parsed {dataset_name}: "
            f"{len(expr_df)} features × {len(expr_df.columns)} samples"
        )

        if len(platform_samples) > len(expr_df.columns):
            log.info(
                f"    Note: Skipped {len(platform_samples) - len(expr_df.columns)} "
                f"samples due to data quality issues"
            )

        return processed

    def verify_gene_symbols(dataset):
        """Check whether a dataset's gene symbols look correctly formatted.

        Note:
            This function takes ``dataset`` as its only argument and has no
            ``self`` parameter, so it cannot be called as a bound instance
            method (``handler.verify_gene_symbols(x)`` would pass ``handler``
            as ``dataset``). It is not currently called anywhere in this
            codebase. Left as-is since fixing it would change the method
            signature.

        Args:
            dataset: A dataset dict with a ``"keys"`` entry (gene symbols).

        Returns:
            An ``(is_valid, issues)`` tuple: whether the symbols look valid,
            and a list of human-readable descriptions of any problems found.
        """
        keys = dataset.get("keys", [])

        issues = []

        # Check 1: Must be a list
        if not isinstance(keys, list):
            issues.append(f"keys is {type(keys)}, not list")

        # Check 2: Check for description-like symbols
        sample = keys[:100] if len(keys) > 100 else keys

        has_spaces = sum(1 for k in sample if isinstance(k, str) and " " in k)
        if has_spaces > 5:
            issues.append(f"{has_spaces}/100 symbols contain spaces (likely descriptions)")

        is_long = sum(1 for k in sample if isinstance(k, str) and len(k) > 20)
        if is_long > 5:
            issues.append(f"{is_long}/100 symbols are >20 chars (likely descriptions)")

        single_char = sum(1 for k in sample if isinstance(k, str) and len(k) <= 2)
        if single_char > 20:
            issues.append(f"{single_char}/100 symbols are 1-2 chars (likely fragments)")

        # Check 3: Reasonable number of unique genes
        unique = len(set(keys))
        ratio = unique / len(keys) if len(keys) > 0 else 0

        if ratio < 0.3:
            issues.append(f"Low uniqueness: {unique}/{len(keys)} = {ratio:.1%}")

        # Check 4: Should look like gene symbols
        # Real gene symbols: uppercase, 3-10 chars, letters/numbers
        import re

        valid_pattern = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")

        valid_format = sum(1 for k in sample if isinstance(k, str) and valid_pattern.match(k))

        if valid_format < 50:  # Less than 50% look like gene symbols
            issues.append(
                f"Only {valid_format}/100 symbols match gene symbol format "
                "(uppercase, 2-10 chars)"
            )

        is_valid = len(issues) == 0

        return is_valid, issues

    def _parse_series_matrix(self, raw_dataset: RawDataset) -> ProcessedDataset:
        """Parse Series Matrix format data."""

        matrix = raw_dataset.raw_data["matrix"]

        # Matrix is already in expr format (features × samples)
        expr_df = matrix.astype("float64")

        # Create minimal phenotype dataframe
        pheno_df = pd.DataFrame({"sample_id": expr_df.columns}, index=expr_df.columns)

        # For Series Matrix, we need to fetch annotations separately
        # (This would be done by the calling code via platform annotation fetch)
        # Use probe IDs as placeholder gene symbols (list, not dict)
        gene_symbols = [str(probe) for probe in expr_df.index]

        # Create dataset identifier
        dataset_name = f"{raw_dataset.source_id}_{raw_dataset.platform_id}"

        # Create processed dataset
        processed = ProcessedDataset(
            geo_id=dataset_name,
            expr=expr_df,
            pheno=pheno_df,
            keys=gene_symbols,
            metadata={
                **raw_dataset.metadata,
                "n_features": len(expr_df),
                "handler": self.data_type,
            },
        )

        log.info(
            f"  ✓ Parsed {dataset_name}: "
            f"{len(expr_df)} features × {len(expr_df.columns)} samples"
        )

        return processed

    def _build_expression_matrix(
        self, platform_samples: List, sample_ids: List[str]
    ) -> pd.DataFrame:
        """Build an expression matrix from GSM samples.

        When a sample has fewer probes than the reference sample, values
        are aligned by probe ID rather than padded naively at the end.
        """
        import numpy as np
        import pandas as pd

        expr_data = []
        probe_ids = None
        failed_samples = []

        for gsm in platform_samples:
            try:
                # Get probe IDs (only once, from first sample)
                if probe_ids is None:
                    if "ID_REF" in gsm.table.columns:
                        probe_ids = gsm.table["ID_REF"].values
                    elif "ID" in gsm.table.columns:
                        probe_ids = gsm.table["ID"].values
                    else:
                        probe_ids = gsm.table.index.values

                # Get expression values
                expr_values = None
                for col_name in ["VALUE", "value", "Signal", "signal", "Expression"]:
                    if col_name in gsm.table.columns:
                        expr_values = pd.to_numeric(gsm.table[col_name], errors="coerce").values
                        break

                if expr_values is None:
                    # Find first numeric column
                    numeric_cols = gsm.table.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        expr_values = gsm.table[numeric_cols[0]].values
                    else:
                        log.warning(f"Cannot find expression values for {gsm.name}, skipping")
                        failed_samples.append(gsm.name)
                        continue

                # Align values by probe ID rather than position — a sample
                # can have fewer probes than the reference sample.
                if len(expr_values) != len(probe_ids):
                    log.warning(
                        f"Shape mismatch in {gsm.name}: {len(expr_values)} values "
                        f"vs {len(probe_ids)} probes. Aligning by probe ID."
                    )

                    # Get THIS sample's probe IDs (not the reference)
                    if "ID_REF" in gsm.table.columns:
                        sample_probe_ids = gsm.table["ID_REF"].values
                    elif "ID" in gsm.table.columns:
                        sample_probe_ids = gsm.table["ID"].values
                    else:
                        sample_probe_ids = gsm.table.index.values

                    # Build a Series indexed by this sample's probe IDs
                    sample_series = pd.Series(
                        expr_values[: len(sample_probe_ids)], index=sample_probe_ids
                    )

                    # Reindex to the reference probe list — this correctly
                    # places NaN only where THIS sample truly has no data
                    aligned_series = sample_series.reindex(probe_ids)
                    expr_values = aligned_series.values

                    n_nan = pd.isna(expr_values).sum()
                    log.info(
                        f"  Aligned {gsm.name}: {n_nan} probes have NaN "
                        f"(truly missing from this sample)"
                    )

                # Check for too many NaN values
                n_nan = pd.isna(expr_values).sum()
                if n_nan > len(expr_values) * 0.5:
                    log.warning(
                        f"Sample {gsm.name} has {n_nan} NaN values "
                        f"({n_nan / len(expr_values) * 100:.1f}%), skipping"
                    )
                    failed_samples.append(gsm.name)
                    continue

                expr_data.append(expr_values)

            except Exception as e:
                log.warning(f"Failed to process {gsm.name}: {e}")
                failed_samples.append(gsm.name)
                continue

        # Report any samples skipped above.
        if failed_samples:
            log.warning(
                f"Skipped {len(failed_samples)} samples due to data issues: "
                f"{failed_samples[:5]}{'...' if len(failed_samples) > 5 else ''}"
            )

        if not expr_data:
            raise ValueError(
                f"No valid samples could be processed. "
                f"All {len(platform_samples)} samples failed."
            )

        # Keep only the sample IDs that were actually processed.
        processed_sample_ids = [
            sid for sid, gsm in zip(sample_ids, platform_samples) if gsm.name not in failed_samples
        ]

        # Create DataFrame
        expr_df = pd.DataFrame(np.array(expr_data).T, index=probe_ids, columns=processed_sample_ids)

        # Convert to float64
        expr_df = expr_df.astype("float64")

        # Auto-detect and apply log2 transformation (R parity): R's
        # getGEOData() returns Series Matrix data that's typically already
        # log2-transformed, while GEOparse SOFT format returns raw
        # intensities for Affymetrix arrays. Heuristic: a median expression
        # above 20 indicates raw intensity that still needs log2.
        finite_values = expr_df.values[np.isfinite(expr_df.values)]
        if len(finite_values) > 0:
            median_expr = np.median(finite_values)
            max_expr = np.max(finite_values)
            if median_expr > 20:
                log.info(
                    f"  Expression data appears to be raw intensity "
                    f"(median={median_expr:.1f}, max={max_expr:.1f}). "
                    f"Applying log2 transformation for R parity."
                )

                # Replace non-positive values with NaN before log2
                # (R handles this similarly with NA)
                expr_df = expr_df.clip(lower=0)
                expr_df = expr_df.replace(0, np.nan)
                expr_df = np.log2(expr_df)
                finite_post = expr_df.values[np.isfinite(expr_df.values)]
                log.info(
                    f"  After log2: range=[{np.min(finite_post):.2f}, "
                    f"{np.max(finite_post):.2f}], "
                    f"median={np.median(finite_post):.2f}"
                )
            else:
                log.info(
                    f"  Expression data appears log-transformed "
                    f"(median={median_expr:.2f}, max={max_expr:.2f})"
                )

        log.info(
            f"Built expression matrix: {expr_df.shape[0]} features × "
            f"{expr_df.shape[1]} samples "
            f"(skipped {len(failed_samples)} problematic samples)"
        )

        return expr_df

    def _build_phenotype_dataframe(self, platform_samples: List) -> pd.DataFrame:
        """Build phenotype DataFrame from GSM samples."""

        pheno_dict = {}

        for gsm in platform_samples:
            pheno_dict[gsm.name] = {
                "title": gsm.metadata.get("title", [""])[0],
                "source_name": gsm.metadata.get("source_name_ch1", [""])[0],
                "organism": gsm.metadata.get("organism_ch1", [""])[0],
                "characteristics": "; ".join(gsm.metadata.get("characteristics_ch1", [])),
                "description": gsm.metadata.get("description", [""])[0],
            }

        pheno_df = pd.DataFrame(pheno_dict).T
        return pheno_df

    def _extract_gene_symbols(self, gpl, probe_ids: pd.Index) -> list:
        """Extract gene-symbol annotations from a GPL object.

        Handles two annotation layouts:

        - ``gene_assignment`` format (Affymetrix Gene ST arrays like
          GPL6244), where the gene symbol is the *second* ``//``-separated
          field, not the first (which is a transcript/accession ID)::

              "ENST00000328113 // MIR181A2HG // description // location // entrez"
               ^ field 1: accession ID      ^ field 2: gene symbol (used)

        - The standard format, where entries are ``///``-separated and each
          entry is itself ``//``-separated::

              "NM_020937 // GENE1 // desc /// NM_000546 // GENE2 // desc"
        """

        if not hasattr(gpl, "table") or gpl.table is None:
            log.warning("GPL table not available, using probe IDs as keys")
            return [str(probe) for probe in probe_ids]

        log.info(f"   Extracting gene symbols from GPL table...")
        log.info(f"   Available columns: {gpl.table.columns.tolist()}")

        possible_symbol_columns = [
            # Unambiguous gene symbol columns (take value directly)
            "Gene Symbol",
            "GENE_SYMBOL",
            "Gene_Symbol",
            "gene_symbol",
            "genesymbol",
            "Symbol",
            "SYMBOL",
            "symbol",
            "HUGO_Gene_Nomenclature",
            # Assignment columns (need field extraction)
            "gene_assignment",
            # Other gene columns
            "Gene",
            "GENE",
            "gene",
            "ORF",
        ]

        # Columns where the value IS already the gene symbol (no field extraction)
        DIRECT_SYMBOL_COLUMNS = {
            "Gene Symbol",
            "GENE_SYMBOL",
            "Gene_Symbol",
            "gene_symbol",
            "genesymbol",
            "Symbol",
            "SYMBOL",
            "symbol",
            "HUGO_Gene_Nomenclature",
        }

        # Columns where the value is an accession//symbol//description record
        ASSIGNMENT_COLUMNS = {"gene_assignment", "mrna_assignment"}

        gene_col = None
        for col_name in possible_symbol_columns:
            if col_name in gpl.table.columns:
                gene_col = col_name
                log.info(f"   ✓ Using column '{gene_col}' for gene symbols")
                break

        if gene_col is None:
            log.warning(f"   ⚠️  No gene symbol column found!")
            log.warning(f"   Available columns: {gpl.table.columns.tolist()[:20]}")
            log.warning(f"   Using probe IDs as fallback")
            return [str(probe) for probe in probe_ids]

        # ------------------------------------------------------------------
        # Patterns that indicate an accession ID, NOT a gene symbol.
        # When seen as the first field in a // record, skip to the 2nd field.
        # ------------------------------------------------------------------
        import re

        # Regular expression to recognize accession‑style IDs (ENST*, NM_*, etc.)
        # Adjust the pattern if you have additional accession types.
        ACCESSION_PATTERN = re.compile(r"^(ENST|NM_)[0-9]+$")

        def clean_gene_symbol(symbol_str, col_is_assignment: bool = False) -> str:
            """Convert a probe annotation string into a comma-separated gene-symbol list.

            Emulates the R helper ``gsub(";| /// ", ",", keys)``.

            Args:
                symbol_str: Raw annotation string from the probe file,
                    e.g. ``"NM_000123 // TP53 // desc /// NM_000456 // ABC1 // something"``.
                col_is_assignment: Whether to treat ``symbol_str`` as an
                    *assignment*-format value, where the gene symbol is
                    always the second field of each block.

            Returns:
                A comma-separated string of every gene symbol found for the
                probe, or ``"NA"`` if none can be found.
            """
            if pd.isna(symbol_str) or str(symbol_str).strip() in (
                "",
                "---",
                "NA",
                "na",
                "N/A",
            ):
                return "NA"

            symbol_str = str(symbol_str).strip()

            if not symbol_str:
                return "NA"

            # 1. Split on '///' or ';' (multiple genes per probe)
            #    Matches R: gsub("; | /// ", ",", keys)
            multi_entries = re.split(r"\s*///\s*|;", symbol_str)
            multi_entries = [e.strip() for e in multi_entries if e.strip()]

            collected_genes = []

            for entry in multi_entries:
                # 2. Inside each entry, '//' separates fields
                fields = [f.strip() for f in entry.split("//") if f.strip()]
                if not fields:
                    continue

                if col_is_assignment:
                    # gene_assignment: field[0]=accession, field[1]=symbol
                    #
                    # R PARITY (Case 2): R splits the ENTIRE raw string
                    # on " /+ " (flat split, treating /// same as //)
                    # and takes the element at a fixed position (where
                    # GAPDH appears). This means R gets exactly ONE gene
                    # per probe, from the FIRST /// entry only.
                    #
                    # We match this: take field[1] from the FIRST entry,
                    # then STOP. Do not process further /// entries.
                    if len(fields) >= 2:
                        candidate = fields[1].strip()
                        candidate = candidate.split()[0] if " " in candidate else candidate
                        if candidate and candidate not in ("", "---", "NA"):
                            collected_genes.append(candidate)
                    else:
                        first = fields[0].strip()
                        if " " in first:
                            first = first.split()[0]
                        if (
                            first
                            and first not in ("", "---", "NA")
                            and not ACCESSION_PATTERN.match(first)
                        ):
                            collected_genes.append(first)
                    # R Case 2 only extracts ONE gene per probe
                    break
                else:
                    # Direct symbol column: value IS the gene symbol
                    # R Case 1 takes the RAW value, then gsub converts
                    # /// to commas. ALL genes are kept.
                    #
                    # R-PARITY FIX: Do NOT split on spaces for direct
                    # symbol columns. R's gsub(";| /// ",",",keys)
                    # only replaces /// and ; with commas — it never
                    # truncates multi-word values. Keeping the full
                    # string ensures Ensembl validation produces the
                    # same result as R (reject or accept identically).
                    first = fields[0].strip()
                    if ACCESSION_PATTERN.match(first):
                        continue
                    if first and first not in ("", "---", "NA"):
                        collected_genes.append(first)

            # 3. Return ALL genes comma-separated (R parity)
            if not collected_genes:
                return "NA"
            return ",".join(collected_genes)

        # ------------------------------------------------------------------
        # Build the probe → symbol list
        # ------------------------------------------------------------------
        col_is_assignment = gene_col in ASSIGNMENT_COLUMNS

        # ------------------------------------------------------------------
        # Load Ensembl gene validator for R-parity validation
        # R's .GEO_fData_key_parser() validates genes against
        # ens_ensgID_table$display_label INSIDE the function.
        # We match this by validating here, not as a separate step.
        # ------------------------------------------------------------------
        try:
            validator = EnsemblGeneValidator()
            ensembl_genes = validator.valid_genes
            log.info(f"   ✓ Loaded Ensembl validator: {len(ensembl_genes):,} valid genes")
        except Exception as e:
            log.warning(f"   ⚠️  Could not load Ensembl validator: {e}")
            log.warning(f"   Skipping Ensembl validation (results may differ from R)")
            ensembl_genes = None

        try:
            if "ID" in gpl.table.columns:
                mapping_series = gpl.table.set_index("ID")[gene_col]
            else:
                mapping_series = gpl.table[gene_col]
                mapping_series.index = gpl.table.index

            probe_to_raw = mapping_series.to_dict()

            gene_symbols_list = []
            n_validated_out = 0
            for probe in probe_ids:
                raw = probe_to_raw.get(probe, None)
                symbol = clean_gene_symbol(raw, col_is_assignment)

                # -------------------------------------------------------
                # R-parity: Ensembl validation (FIRST gene only)
                # R's .GEO_fData_key_parser() validates ONLY the first
                # gene in each comma-separated key:
                #   fake_genes <- which(!(sapply(keys,
                #       function(i) strsplit(i,",")[[1]][1])
                #       %in% genes$display_label))
                #   keys[fake_genes] <- NA
                #
                # If first gene is valid -> keep ENTIRE key (all genes)
                # If first gene is invalid -> set ENTIRE key to NA
                # Non-first genes survive even if not in Ensembl.
                # -------------------------------------------------------
                if ensembl_genes is not None and symbol and symbol != "NA":
                    first_gene = symbol.split(",")[0].strip()
                    if first_gene not in ensembl_genes:
                        symbol = "NA"
                        n_validated_out += 1

                gene_symbols_list.append(symbol)

            unique_genes = len(set(gene_symbols_list) - {"NA"})
            na_count = sum(1 for g in gene_symbols_list if g == "NA")

            log.info(f"   ✓ Extracted {len(gene_symbols_list)} gene symbols")
            log.info(f"     - Valid unique genes : {unique_genes}")
            log.info(f"     - NA / unmapped      : {na_count}")
            if ensembl_genes is not None:
                log.info(f"     - Ensembl-filtered   : {n_validated_out} probes had no valid genes")
            log.info(f"   Sample symbols: {gene_symbols_list[:10]}")

            return gene_symbols_list

        except Exception as e:
            log.error(f"   ❌ Failed to extract gene symbols: {e}")
            return [str(probe) for probe in probe_ids]

    def normalize(self, dataset: ProcessedDataset) -> ProcessedDataset:
        """Apply microarray-specific normalization.

        Currently supports log transformation and quantile normalization,
        each applied only if enabled on this handler. Other methods (RMA,
        MAS5, GCRMA, etc.) could be added here in the future.

        Args:
            dataset: A parsed dataset.

        Returns:
            The normalized dataset.
        """
        expr = dataset.expr.copy()

        # Log transformation
        if self.log_transform:
            log.info("Applying log2 transformation...")

            # Check if data is already log-scale
            # Heuristic: if all values < 50, likely already log-scale
            if expr.max().max() < 50:
                log.info("  Data appears to be log-scale already, skipping")
            else:
                # Add small constant to avoid log(0)
                expr = np.log2(expr + 1)

        # Quantile normalization
        if self.normalize_flag:
            log.info("Applying quantile normalization...")
            expr = self._quantile_normalize(expr)

        # Create new dataset with normalized expression
        normalized = ProcessedDataset(
            geo_id=dataset.geo_id,
            expr=expr,
            pheno=dataset.pheno,
            keys=dataset.keys,
            metadata={
                **dataset.metadata,
                "log_transformed": self.log_transform,
                "quantile_normalized": self.normalize_flag,
            },
        )

        return normalized

    def _quantile_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quantile normalization to an expression matrix.

        Args:
            df: Expression matrix (features × samples).

        Returns:
            The quantile-normalized expression matrix.
        """
        # Get ranks
        ranks = df.stack().groupby(df.rank(method="first").stack().astype(int)).mean()

        # Replace values with mean of rank
        normalized = df.rank(method="min").stack().astype(int).map(ranks).unstack()

        return normalized

    def is_valid_gene_symbol(symbol_str):
        """Check whether a string looks like a valid gene symbol.

        Note:
            This function takes ``symbol_str`` as its only argument and has
            no ``self`` parameter, so it cannot be called as a bound
            instance method. It is not currently called anywhere in this
            codebase. Left as-is since fixing it would change the method
            signature.

        Args:
            symbol_str: Candidate gene symbol string.

        Returns:
            True if ``symbol_str`` matches the expected gene-symbol pattern
            (this is pattern matching only — it does not check against a
            gene database).
        """
        symbol_str = str(symbol_str)

        # Invalid: NA
        if symbol_str == "NA":
            return False

        # Invalid: Probe IDs
        if re.match(r"^\d+_[a-z_]+_at$", symbol_str, re.IGNORECASE):
            return False

        # Pattern matching only - NO database validation!
        if re.match(r"^[A-Z0-9][A-Z0-9-]{0,19}$", symbol_str):
            return True

        return False
