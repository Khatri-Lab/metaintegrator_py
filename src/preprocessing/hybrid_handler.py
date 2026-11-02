"""
Hybrid Microarray Data Handler.

Processes raw datasets produced by :class:`HybridGeoSource` into
:class:`ProcessedDataset` objects:

- Phenotype extraction from GEOparse per-sample metadata
- Gene annotation via :class:`GeneAnnotator` + Ensembl validation
- Log2 detection and transformation, as R's getGEOData() does it
  (see ``preprocessing.utils.log_scale``)
- Standalone GPL fallback when family SOFT annotation yields 0 genes

This handler expects ``RawDataset.raw_data`` to contain:

- ``'expr_df'``: ``pd.DataFrame`` (probes × samples) from Series Matrix
- ``'gpl'``: GEOparse GPL object
- ``'gse'``: GEOparse GSE object
- ``'platform_id'``: GPL identifier string

Usage
-----
    from preprocessing.hybrid_source import HybridGeoSource
    from preprocessing.hybrid_handler import HybridMicroarrayHandler
    from preprocessing.base import DataLoader

    source = HybridGeoSource()
    handler = HybridMicroarrayHandler()
    loader = DataLoader(source, handler)
    datasets = loader.load_multiple(['GSE11909'])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import GEOparse
except ImportError:
    GEOparse = None

from .base import DataHandler, ProcessedDataset, RawDataset
from .gene_annotator import GeneAnnotator
from .gene_validation import EnsemblGeneValidator
from .geo_https import prefetch_geo_file_or_warn
from .utils.log_scale import apply_r_log2

log = logging.getLogger(__name__)

# Known expected probe counts for diagnostic verification.
# These are informational only: mismatches produce a warning, not an error.
EXPECTED_PROBES = {
    "GPL96": 22283,
    "GPL97": 22645,
    "GPL6244": 33297,
    "GPL570": 54675,
}


class HybridMicroarrayHandler(DataHandler):
    """
    Handler for hybrid-source microarray datasets.

    Works with :class:`HybridGeoSource` to produce datasets using the
    Series Matrix + GEOparse dual-source strategy.

    The processing pipeline (implemented in :meth:`parse`):

    1. Extract expression DataFrame from ``raw_data['expr_df']``
    2. Build phenotype DataFrame from GEOparse per-sample metadata
    3. Annotate probes → gene symbols via :class:`GeneAnnotator`
    4. Validate symbols against Ensembl reference
    5. Apply log2 when R's ``.GEM_log_check`` says the data is not on a
       log scale, and record the decision in ``metadata['log_scale_check']``
    6. Fall back to standalone GPL download if family SOFT yields 0 genes

    """

    def __init__(
        self,
        annotator: GeneAnnotator | None = None,
        validator: EnsemblGeneValidator | None = None,
    ):
        """Initialize the handler.

        Args:
            annotator: Custom annotator instance. Uses a default
                :class:`GeneAnnotator` when not given.
            validator: Custom Ensembl validator. When not given, one is
                created automatically on first use (with graceful fallback
                if that fails).
        """
        self._annotator = annotator or GeneAnnotator()
        self._validator = validator  # None means "load lazily on first use"
        self._validator_loaded = validator is not None

    @property
    def data_type(self) -> str:
        """Human-readable name of the data type this handler processes."""
        return "Microarray (Hybrid)"

    # ------------------------------------------------------------------ #
    #  DataHandler interface                                              #
    # ------------------------------------------------------------------ #

    def parse(self, raw_dataset: RawDataset) -> ProcessedDataset:
        """Parse a raw hybrid dataset into a :class:`ProcessedDataset`.

        Args:
            raw_dataset: Raw dataset whose ``raw_data`` must contain the
                keys ``'expr_df'``, ``'gpl'``, ``'gse'``, and
                ``'platform_id'``.

        Returns:
            The processed dataset.
        """
        rd = raw_dataset.raw_data
        expr_df: pd.DataFrame = rd["expr_df"]
        gpl = rd["gpl"]
        gse = rd["gse"]
        platform_id: str = rd["platform_id"]
        gse_id: str = raw_dataset.source_id
        dataset_name = f"{gse_id}_{platform_id}"

        log.info(f"  Processing {dataset_name}...")

        # Ensure validator is loaded
        validator = self._get_validator()

        # 1. Build pheno from GEOparse per-sample metadata
        pheno_df = self._build_pheno(gse, platform_id, expr_df.columns)

        # 2. Annotate probes → gene symbols
        keys = self._annotator.annotate(
            gpl.table,
            expr_df.index,
            validator=validator,  # R-parity: Ensembl + GenBank filtering
        )

        # 3. Apply log2 if needed, exactly as R's getGEOData() decides
        expr_df, log_scale_check = self._auto_log2_with_report(expr_df, dataset_name)

        # 4. Create ProcessedDataset
        dataset = ProcessedDataset(
            geo_id=dataset_name,
            expr=expr_df,
            pheno=pheno_df,
            keys=keys,
            metadata={
                "gse_id": gse_id,
                "platform": platform_id,
                "n_features": len(expr_df),
                "n_samples": len(expr_df.columns),
                "source": "Hybrid (SeriesMatrix + GEOparse)",
                "format": "SeriesMatrix",
                "handler": self.data_type,
                "cache_dir": raw_dataset.metadata.get("cache_dir", "~/.geo_cache"),
                # What the R-parity log-scale check decided and did. See
                # `preprocessing.utils.log_scale.LogScaleReport`.
                "log_scale_check": log_scale_check,
            },
        )

        n_valid = len(set(keys) - {"NA"})
        n_na = sum(1 for k in keys if k == "NA")
        log.info(
            f"    ✓ {dataset_name}: {len(expr_df)} probes × "
            f"{len(expr_df.columns)} samples, {n_valid} genes, {n_na} NA"
        )

        # 5. Diagnostic: verify probe count
        if platform_id in EXPECTED_PROBES:
            exp = EXPECTED_PROBES[platform_id]
            if len(expr_df) != exp:
                log.warning(
                    f"    ⚠️  Probe count mismatch for {platform_id}: "
                    f"got {len(expr_df)}, expected {exp}"
                )

        # 6. Fallback: standalone GPL if family SOFT yielded 0 genes
        if n_valid == 0:
            dataset = self._try_standalone_gpl_fallback(
                dataset,
                gse_id,
                platform_id,
                expr_df,
                validator,
            )

        return dataset

    def normalize(self, dataset: ProcessedDataset) -> ProcessedDataset:
        """No additional normalization needed for hybrid datasets.

        Series Matrix data from NCBI is typically already normalized and
        log-transformed; the ``_auto_log2`` step in :meth:`parse` handles
        the rare exception.
        """
        return dataset

    # ------------------------------------------------------------------ #
    #  Phenotype extraction                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_pheno(gse: "GEOparse.GSE", platform_id: str, sample_ids: "pd.Index") -> pd.DataFrame:
        """Build phenotype DataFrame from GEOparse per-sample metadata."""
        pheno_dict = {}

        for sample_id in sample_ids:
            sid = str(sample_id)
            if sid in gse.gsms:
                gsm = gse.gsms[sid]
                pheno_dict[sid] = {
                    "title": gsm.metadata.get("title", [""])[0],
                    "source_name": gsm.metadata.get("source_name_ch1", [""])[0],
                    "organism": gsm.metadata.get("organism_ch1", [""])[0],
                    "characteristics": "; ".join(
                        gsm.metadata.get("characteristics_ch1", []),
                    ),
                    "description": gsm.metadata.get("description", [""])[0],
                }
            else:
                log.warning(f"    Sample {sid} not found in GEOparse, using defaults")
                pheno_dict[sid] = {
                    "title": "",
                    "source_name": "",
                    "organism": "",
                    "characteristics": "",
                    "description": "",
                }

        pheno_df = pd.DataFrame(pheno_dict).T
        pheno_df = pheno_df.loc[[str(s) for s in sample_ids]]
        return pheno_df

    # ------------------------------------------------------------------ #
    #  Log2 detection                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _auto_log2_with_report(expr_df: pd.DataFrame, label: str = "") -> tuple[pd.DataFrame, dict]:
        """Log2-transform the matrix when R's rule says it is not log scale.

        Delegates to :func:`preprocessing.utils.log_scale.apply_r_log2`,
        an exact port of R MetaIntegrator's ``getGEOData()`` check and
        transform. Before 0.12.0 this used "median above 20", which can
        disagree with R on borderline datasets.

        Args:
            expr_df: Expression matrix, probes x samples.
            label: Dataset name, for the log.

        Returns:
            ``(matrix, report)``. ``report`` is the JSON-safe
            :meth:`LogScaleReport.to_metadata` dict.
        """
        result, report = apply_r_log2(expr_df)
        if report.log2_applied:
            finite = result.values[np.isfinite(result.values)]
            log.info(
                f"    {label}: not on a log scale (R log check, "
                f"{report.decided_by}, median={report.median_before:.1f}), "
                f"applying log2 -> median {np.median(finite):.2f}"
            )
        else:
            log.info(f"    {label}: already on a log scale (R log check, " f"{report.decided_by})")
        if report.zero_values_set_missing:
            log.warning(
                f"    {label}: {report.zero_values_set_missing} zero value(s) "
                "set to missing before log2 (R would give -Inf)"
            )
        if report.differs_from_legacy:
            log.warning(
                f"    {label}: R's log check and the former median rule disagree "
                f"(log2 applied: {report.log2_applied})"
            )
        return result, report.to_metadata()

    @staticmethod
    def _auto_log2(expr_df: pd.DataFrame, label: str = "") -> pd.DataFrame:
        """Log2-transform the matrix when R's rule says it is not log scale.

        Kept for callers that only need the matrix. See
        :meth:`_auto_log2_with_report`.
        """
        result, _ = HybridMicroarrayHandler._auto_log2_with_report(expr_df, label)
        return result

    # ------------------------------------------------------------------ #
    #  Standalone GPL fallback                                            #
    # ------------------------------------------------------------------ #

    def _try_standalone_gpl_fallback(
        self,
        dataset: ProcessedDataset,
        gse_id: str,
        platform_id: str,
        expr_df: pd.DataFrame,
        validator: Optional[EnsemblGeneValidator],
    ) -> ProcessedDataset:
        """Re-annotate using standalone GPL when family SOFT yields 0 genes."""
        dataset_name = f"{gse_id}_{platform_id}"
        log.warning(
            f"    0 genes for {dataset_name} from family SOFT: falling back to standalone GPL"
        )

        try:
            # Expand "~" here, so the HTTPS download and GEOparse use the
            # same folder. GEOparse would fetch the table over plain HTTP.
            destdir = str(Path(dataset.metadata.get("cache_dir", "~/.geo_cache")).expanduser())
            prefetch_geo_file_or_warn(platform_id, destdir)
            standalone_gpl = GEOparse.get_GEO(
                geo=platform_id,
                destdir=destdir,
                silent=True,
            )
            keys = self._annotator.annotate(
                standalone_gpl.table,
                expr_df.index,
                validator=validator,
            )
            n_valid = len(set(keys) - {"NA"})
            n_na = sum(1 for k in keys if k == "NA")
            dataset.keys = keys
            log.info(f"    ✓ Standalone GPL: {n_valid} genes, {n_na} NA")
        except Exception as e:
            log.error(f"    Standalone GPL fallback failed: {e}")

        return dataset

    # ------------------------------------------------------------------ #
    #  Validator lazy-loading                                             #
    # ------------------------------------------------------------------ #

    def _get_validator(self) -> Optional[EnsemblGeneValidator]:
        """Load Ensembl validator on first use (cached thereafter)."""
        if not self._validator_loaded:
            try:
                self._validator = EnsemblGeneValidator()
                log.info(f"  Ensembl validator: {len(self._validator.valid_genes):,} valid genes")
            except Exception as e:
                log.warning(f"  Could not load Ensembl validator: {e}")
                self._validator = None
            self._validator_loaded = True
        return self._validator
