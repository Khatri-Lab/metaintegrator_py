"""
Hybrid GEO Data Source.

Implements a hybrid fetch strategy combining two NCBI data sources:
  - Expression data from Series Matrix files (pre-aligned, pre-processed)
  - Gene annotation + phenotype data from GEOparse SOFT files

This approach avoids the probe-value misalignment that occurs when
building expression matrices from per-sample SOFT tables (where some
samples may have missing probes).  It matches the strategy used by
R's ``GEOquery::getGEO()``.

Network I/O is delegated to :class:`NCBINetwork`, which can be injected
for testing (mock) or customized (timeouts, retries).

Usage
-----
    from preprocessing.hybrid_source import HybridGeoSource
    from preprocessing.hybrid_handler import HybridMicroarrayHandler
    from preprocessing.base import DataLoader

    source = HybridGeoSource(cache_dir='~/.geo_cache')
    handler = HybridMicroarrayHandler()
    loader = DataLoader(source, handler)
    datasets = loader.load_multiple(['GSE11909', 'GSE50635'])
"""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import GEOparse
except ImportError:
    GEOparse = None

from .base import DataSource, RawDataset
from .ncbi_network import NCBINetwork

log = logging.getLogger(__name__)


class HybridGeoSource(DataSource):
    """Fetch GEO datasets using a hybrid strategy.

    Combines two data retrieval methods in a single fetch:

    1. **Series Matrix files** (via :class:`NCBINetwork`) for expression
       data — pre-aligned, probe × sample matrices.
    2. **GEOparse SOFT files** for GPL annotation tables and per-sample
       phenotype metadata.

    Each platform within a GSE produces one :class:`RawDataset` whose
    ``raw_data`` dict contains:

    - ``'expr_df'``: ``pd.DataFrame`` (probes × samples) from Series Matrix
    - ``'gpl'``: GEOparse GPL object (annotation table)
    - ``'gse'``: GEOparse GSE object (for phenotype metadata)
    - ``'platform_id'``: GPL identifier string

    Examples:
        >>> source = HybridGeoSource()
        >>> raw_datasets = source.fetch('GSE11909')
        >>> len(raw_datasets)
        2  # GPL96 + GPL97

        With a custom network (e.g. for testing):

        >>> mock_net = MockNCBINetwork(...)
        >>> source = HybridGeoSource(network=mock_net)
    """

    SERIES_MATRIX_SUFFIX = "_series_matrix.txt.gz"

    def __init__(
        self,
        cache_dir: str | Path = "~/.geo_cache",
        network: Optional[NCBINetwork] = None,
    ):
        """Initialize the hybrid GEO source.

        Args:
            cache_dir: Directory for caching downloaded files.
            network: Injectable network layer. When not given, a standard
                ``NCBINetwork(cache_dir)`` is created automatically; pass a
                mock or custom instance for testing or non-standard setups.
        """
        if GEOparse is None:
            raise ImportError("GEOparse is required: pip install GEOparse")
        super().__init__(cache_dir)
        self._network = network or NCBINetwork(cache_dir=self.cache_dir)

    @property
    def source_name(self) -> str:
        """Human-readable name of this data source."""
        return "NCBI GEO (Hybrid: SeriesMatrix + GEOparse)"

    # ------------------------------------------------------------------ #
    #  DataSource interface                                               #
    # ------------------------------------------------------------------ #

    def fetch(self, dataset_id: str, **kwargs) -> List[RawDataset]:
        """Fetch a single GSE using the hybrid strategy.

        Args:
            dataset_id: GEO Series ID, e.g. ``'GSE11909'``.

        Returns:
            One ``RawDataset`` per platform found in the GSE.

        Raises:
            ValueError: If no Series Matrix files are found, or no
                platforms match between Series Matrix and GEOparse.
            RuntimeError: If the download or GEOparse loading fails.
        """
        gse_id = dataset_id.strip().upper()
        log.info(f"Loading {gse_id} (hybrid mode: SeriesMatrix + GEOparse)...")

        # Step 1: Download & parse Series Matrix files → expression per platform
        sm_data = self._download_and_parse_series_matrices(gse_id)
        log.info(f"  Series Matrix: {len(sm_data)} platform(s): {list(sm_data.keys())}")

        # Step 2: Download family SOFT via GEOparse → pheno + GPL annotation
        gse = self._load_geoparse(gse_id)
        log.info(
            f"  GEOparse: {len(gse.gsms)} samples, "
            f"{len(gse.gpls)} platform(s): {list(gse.gpls.keys())}"
        )

        # Step 3: Build one RawDataset per platform
        raw_datasets: List[RawDataset] = []

        for platform_id, expr_df in sm_data.items():
            if platform_id not in gse.gpls:
                log.warning(f"    GPL {platform_id} not found in GEOparse data, skipping")
                continue

            gpl = gse.gpls[platform_id]

            # Diagnostic: verify GPL table is populated
            if hasattr(gpl, "table") and gpl.table is not None:
                log.info(
                    f"    GPL table: {len(gpl.table)} rows, "
                    f"columns: {gpl.table.columns.tolist()[:6]}..."
                )
            else:
                log.error(f"    GPL table is EMPTY for {platform_id} — annotation will fail!")

            raw = RawDataset(
                source_id=gse_id,
                platform_id=platform_id,
                raw_data={
                    "expr_df": expr_df,
                    "gpl": gpl,
                    "gse": gse,
                    "platform_id": platform_id,
                },
                metadata={
                    "gse_id": gse_id,
                    "platform": platform_id,
                    "n_features": len(expr_df),
                    "n_samples": len(expr_df.columns),
                    "source": self.source_name,
                    "format": "Hybrid (SeriesMatrix + GEOparse)",
                    "cache_dir": str(self.cache_dir),
                },
            )
            raw_datasets.append(raw)

        if not raw_datasets:
            raise ValueError(
                f"No platforms could be matched between Series Matrix " f"and GEOparse for {gse_id}"
            )

        return raw_datasets

    def fetch_multiple(
        self,
        dataset_ids: List[str],
        stop_on_error: bool = False,
        **kwargs,
    ) -> List[RawDataset]:
        """Fetch multiple GSE IDs.

        Args:
            dataset_ids: GEO Series IDs.
            stop_on_error: Raise on the first failure if True; skip and
                continue otherwise.

        Returns:
            All raw datasets across every GSE ID.
        """
        all_raw: List[RawDataset] = []

        for gse_id in dataset_ids:
            gse_id = gse_id.strip().upper()
            try:
                raw_datasets = self.fetch(gse_id, **kwargs)
                all_raw.extend(raw_datasets)
            except Exception as e:
                log.error(f"Failed to load {gse_id}: {e}")
                if stop_on_error:
                    raise
                continue

        log.info(f"✓ Fetched {len(all_raw)} raw datasets " f"from {len(dataset_ids)} GSE IDs")
        return all_raw

    # ------------------------------------------------------------------ #
    #  Series Matrix: Download & Parse                                    #
    # ------------------------------------------------------------------ #

    def _download_and_parse_series_matrices(
        self,
        gse_id: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download and parse Series Matrix files for a GSE.

        Uses :class:`NCBINetwork` for FTP listing and HTTPS download.
        """
        ftp_path = NCBINetwork.geo_series_ftp_path(gse_id, "matrix")

        # List matrix files via FTP (with fallback to default filename)
        try:
            matrix_files = self._network.list_ftp_directory(
                ftp_path,
                suffix=self.SERIES_MATRIX_SUFFIX,
            )
        except ConnectionError:
            log.warning("FTP listing failed, trying default filename")
            matrix_files = [f"{gse_id}{self.SERIES_MATRIX_SUFFIX}"]

        if not matrix_files:
            raise ValueError(f"No Series Matrix files found for {gse_id}")

        log.info(f"    Found {len(matrix_files)} matrix file(s): {matrix_files}")

        result: Dict[str, pd.DataFrame] = {}
        for mfile in matrix_files:
            local_path = self._network.download_file(ftp_path, mfile)
            platform_id, expr_df = self._parse_series_matrix(local_path)

            # Override platform from filename if available
            fn_match = re.search(r"-(GPL\d+)_series_matrix", mfile)
            if fn_match:
                platform_id = fn_match.group(1)

            result[platform_id] = expr_df
            log.info(
                f"    Parsed {mfile}: {platform_id}, "
                f"{expr_df.shape[0]} probes × {expr_df.shape[1]} samples"
            )

        return result

    # ------------------------------------------------------------------ #
    #  Series Matrix parsing (pure — no network)                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_series_matrix(filepath: Path) -> tuple[str, pd.DataFrame]:
        """Parse a Series Matrix file into ``(platform_id, expression_df)``.

        The Series Matrix is a pre-aligned, tab-delimited expression matrix
        covering all probes × all samples.
        """
        with gzip.open(str(filepath), "rt") as f:
            txt = f.read()

        # Extract platform from metadata
        platform_id = "unknown"
        for line in txt.splitlines():
            if line.startswith("!Sample_platform_id"):
                m = re.search(r"GPL\d+", line)
                if m:
                    platform_id = m.group(0)
                    break

        # Parse data table (skip all ! metadata lines)
        rows = [r for r in txt.splitlines() if r and not r.startswith("!")]
        if not rows:
            raise ValueError(f"Empty Series Matrix: {filepath}")

        header = [c.strip('"') for c in rows[0].split("\t")]
        data = [[c.strip('"') for c in row.split("\t")] for row in rows[1:]]
        df = pd.DataFrame(data, columns=header)

        # Convert expression columns to float64
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        # Set probe ID as index
        df.set_index(df.columns[0], inplace=True)
        df.index = df.index.astype(str)

        return platform_id, df

    # ------------------------------------------------------------------ #
    #  GEOparse: Load family SOFT                                         #
    # ------------------------------------------------------------------ #

    def _load_geoparse(self, gse_id: str) -> "GEOparse.GSE":
        """Load a GSE via GEOparse (family SOFT file) for pheno + annotation.

        Must use ``how='full'`` and ``include_data=True`` to ensure
        GEOparse fully parses the GPL annotation tables.
        """
        gse = GEOparse.get_GEO(
            geo=gse_id,
            destdir=str(self.cache_dir),
            silent=True,
            include_data=True,
            how="full",
        )
        return gse
