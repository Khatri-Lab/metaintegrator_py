"""
GEO (Gene Expression Omnibus) data sources.

Provides two strategies for fetching data from NCBI GEO:

- NCBIGeoSource: Series Matrix files (fast, subset of samples).
- GEOparseSource: SOFT files (slower, all samples: R parity).

Both sources validate input up front (only GSE series IDs are accepted;
GSM and GPL IDs are rejected with a clear error), and GEOparseSource
falls back to a direct HTTPS download when GEOparse's own FTP download
fails.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional

try:
    import GEOparse
except ImportError:
    GEOparse = None

from .base import DataSource, RawDataset
from .geo_https import prefetch_geo_file_or_warn
from .ncbi_network import NCBINetwork

log = logging.getLogger(__name__)


# ============================================================================
# Input Validation
# ============================================================================


def validate_geo_id(geo_id: str) -> tuple[str, str]:
    """Validate and classify a GEO identifier.

    Args:
        geo_id: GEO identifier, e.g. ``'GSE11909'``, ``'GSM11805'``, or
            ``'GPL96'``.

    Returns:
        A ``(geo_type, geo_id)`` tuple, where ``geo_type`` is ``'GSE'``,
        ``'GSM'``, or ``'GPL'``.

    Raises:
        ValueError: If ``geo_id`` is not a recognized format.
    """
    geo_id = geo_id.strip().upper()

    # Match pattern: GSE/GSM/GPL followed by numbers
    match = re.match(r"^(GSE|GSM|GPL)(\d+)$", geo_id)

    if not match:
        raise ValueError(
            f"Invalid GEO ID format: '{geo_id}'. "
            "Expected format: GSE##### (series), GSM##### (sample), "
            "or GPL##### (platform)"
        )

    geo_type = match.group(1)
    return geo_type, geo_id


def validate_series_id(geo_id: str) -> str:
    """Validate that a GEO ID is a series (GSE) ID.

    Args:
        geo_id: GEO identifier.

    Returns:
        The validated GSE ID.

    Raises:
        ValueError: If ``geo_id`` is not a GSE ID.
    """
    geo_type, validated_id = validate_geo_id(geo_id)

    if geo_type != "GSE":
        raise ValueError(
            f"Expected GSE (series) ID, got {geo_type} ID: '{geo_id}'. "
            f"This loader only accepts series (GSE) identifiers. "
            f"Single samples (GSM) and platforms (GPL) are not supported."
        )

    return validated_id


# ============================================================================
# GEOparse Workarounds
# ============================================================================


def download_soft_file_http(gse_id: str, cache_dir: Path) -> Path:
    """Download a SOFT file via HTTPS (used when GEOparse's FTP download fails).

    Delegates to :class:`NCBINetwork` for a reliable HTTPS download with
    retry logic.

    Args:
        gse_id: GEO Series ID, e.g. ``'GSE11909'``.
        cache_dir: Cache directory.

    Returns:
        Path to the downloaded SOFT file.

    Raises:
        RuntimeError: If the download fails.
    """
    network = NCBINetwork(cache_dir=cache_dir)
    ftp_path = NCBINetwork.geo_series_ftp_path(gse_id, "soft")
    filename = f"{gse_id}_family.soft.gz"
    return network.download_file(ftp_path, filename)


# ============================================================================
# GEOparse-based Source (Full Dataset Access)
# ============================================================================


class GEOparseSource(DataSource):
    """
    Fetch FULL GEO datasets using GEOparse (Python's GEOquery equivalent).

    This source downloads SOFT files which contain ALL samples,
    achieving parity with R's GEOquery without requiring R.

    Key Differences from NCBIGeoSource:
    - NCBIGeoSource: Series Matrix → SUBSET of samples (fast)
    - GEOparseSource: SOFT files → ALL samples (slower but complete)

    It validates input up front (rejecting GSM and accepting only GSE),
    works around a GEOparse FTP download bug, and gives clearer error
    messages than raw GEOparse.

    Requires the ``GEOparse`` package (``pip install GEOparse``).

    Examples:
        >>> source = GEOparseSource()
        >>> raw_datasets = source.fetch('GSE11909', use_soft=True)
        >>> # Returns 2 raw datasets: GPL96 (118 samples), GPL97 (60 samples)
    """

    def __init__(
        self,
        cache_dir: str | Path = "~/.geo_cache",
        silent: bool = True,
        use_http_fallback: bool = True,
    ):
        """Initialize the source.

        Args:
            cache_dir: Directory for caching downloaded GEO files.
            silent: Suppress GEOparse's download progress output.
            use_http_fallback: Fall back to a direct HTTPS download if
                GEOparse's FTP download fails.
        """
        super().__init__(cache_dir)

        if GEOparse is None:
            raise ImportError(
                "GEOparse is required for full GEO dataset access. "
                "Install it with: pip install GEOparse"
            )

        self.silent = silent
        self.use_http_fallback = use_http_fallback
        log.info(f"GEOparse cache directory: {self.cache_dir}")

    @property
    def source_name(self) -> str:
        """Human-readable name of this data source."""
        return "NCBI GEO (GEOparse/SOFT)"

    def fetch(self, dataset_id: str, use_soft: bool = True, **kwargs) -> List[RawDataset]:
        """Fetch a GEO dataset with all samples.

        Args:
            dataset_id: GEO Series ID, e.g. ``'GSE11909'``. Must be a
                series (GSE) ID: samples (GSM) and platforms (GPL) are
                rejected.
            use_soft: If True, download the SOFT file (the full dataset,
                with all samples). If False, use Series Matrix instead
                (a subset: not recommended).
            **kwargs: Additional arguments passed to ``GEOparse.get_GEO()``.

        Returns:
            One ``RawDataset`` per platform in the series.

        Raises:
            ValueError: If ``dataset_id`` is not a valid GSE ID.
            RuntimeError: If the download fails.

        Examples:
            >>> source = GEOparseSource()
            >>> raw_datasets = source.fetch('GSE11909', use_soft=True)
            >>> len(raw_datasets)
            2  # GSE11909 has 2 platforms: GPL96, GPL97
        """
        # VALIDATION: Only accept GSE IDs
        validated_id = validate_series_id(dataset_id)

        log.info(f"Fetching {validated_id} (SOFT format: {use_soft})...")

        # Try GEOparse first
        gse = None
        try:
            gse = self._fetch_with_geoparse(validated_id, use_soft, **kwargs)
        except Exception as e:
            log.warning(f"GEOparse download failed: {e}")

            # Try HTTP fallback for SOFT files
            if use_soft and self.use_http_fallback:
                log.info("Trying HTTP fallback download...")
                try:
                    gse = self._fetch_with_http_fallback(validated_id)
                except Exception as e2:
                    log.error(f"HTTP fallback also failed: {e2}")
                    raise RuntimeError(
                        f"Failed to download {validated_id} using both GEOparse and HTTP fallback. "
                        f"GEOparse error: {e}. HTTP error: {e2}"
                    ) from e
            else:
                raise RuntimeError(f"GEOparse download failed for {validated_id}: {e}") from e

        if gse is None:
            raise RuntimeError(f"Failed to download {validated_id}")

        # Process GSE object
        return self._process_gse(gse, validated_id, use_soft)

    def _fetch_with_geoparse(self, gse_id: str, use_soft: bool, **kwargs):
        """Fetch using standard GEOparse, with the file downloaded over HTTPS.

        GEOparse reads a series from its family SOFT file whatever ``how``
        is, and would download it over FTP (see ``geo_https.py``).
        """
        prefetch_geo_file_or_warn(gse_id, str(self.cache_dir))
        gse = GEOparse.get_GEO(
            geo=gse_id,
            destdir=str(self.cache_dir),
            silent=self.silent,
            include_data=True,
            how="full" if use_soft else "brief",
            **kwargs,
        )
        return gse

    def _fetch_with_http_fallback(self, gse_id: str):
        """Fetch SOFT file via HTTP and parse with GEOparse."""
        # Download SOFT file via HTTP
        soft_file = download_soft_file_http(gse_id, self.cache_dir)

        # Parse with GEOparse
        log.info(f"Parsing {soft_file} with GEOparse...")
        gse = GEOparse.get_GEO(filepath=str(soft_file), silent=self.silent)
        return gse

    def _process_gse(self, gse, gse_id: str, use_soft: bool) -> List[RawDataset]:
        """Process GSE object into RawDatasets."""

        log.info(
            f"Downloaded {gse_id}: {len(gse.gsms)} total samples, " f"{len(gse.gpls)} platforms"
        )

        # Create one RawDataset per platform
        raw_datasets = []

        for platform_id, gpl in gse.gpls.items():
            # Get samples for this platform
            platform_samples = [
                gsm
                for gsm in gse.gsms.values()
                if gsm.metadata.get("platform_id", [None])[0] == platform_id
            ]

            if not platform_samples:
                log.warning(f"No samples found for {platform_id}")
                continue

            log.info(f"  Platform {platform_id}: {len(platform_samples)} samples")

            # Create raw dataset
            raw = RawDataset(
                source_id=gse_id,
                platform_id=platform_id,
                raw_data={"gse": gse, "gpl": gpl, "samples": platform_samples},
                metadata={
                    "gse_id": gse_id,
                    "platform": platform_id,
                    "n_samples": len(platform_samples),
                    "source": self.source_name,
                    "format": "SOFT" if use_soft else "SeriesMatrix",
                },
            )

            raw_datasets.append(raw)

        if not raw_datasets:
            raise ValueError(f"No platforms with samples found in {gse_id}")

        log.info(f"✓ Fetched {gse_id}: {len(raw_datasets)} platforms")

        return raw_datasets

    def fetch_multiple(
        self,
        dataset_ids: List[str],
        stop_on_error: bool = False,
        use_soft: bool = True,
        **kwargs,
    ) -> List[RawDataset]:
        """Fetch multiple GEO datasets.

        Args:
            dataset_ids: GEO Series IDs (GSE only: GSM and GPL are not
                supported).
            stop_on_error: If True, raise on the first failure; if False,
                skip it and continue with the remaining datasets.
            use_soft: Use the SOFT format (full dataset).
            **kwargs: Additional arguments passed to :meth:`fetch`.

        Returns:
            All raw datasets across every GSE ID.

        Examples:
            >>> source = GEOparseSource()
            >>> raw_datasets = source.fetch_multiple(
            ...     ['GSE11909', 'GSE50635'],
            ...     use_soft=True
            ... )
            >>> len(raw_datasets)
            3  # GSE11909: 2 platforms, GSE50635: 1 platform
        """
        all_raw_datasets = []
        failed = []
        invalid = []

        for dataset_id in dataset_ids:
            # Validate first
            try:
                validate_series_id(dataset_id)
            except ValueError as e:
                log.error(f"Invalid dataset ID '{dataset_id}': {e}")
                invalid.append(dataset_id)
                if stop_on_error:
                    raise
                continue

            # Fetch
            try:
                raw_datasets = self.fetch(dataset_id, use_soft=use_soft, **kwargs)
                all_raw_datasets.extend(raw_datasets)

            except Exception as e:
                log.error(f"Failed to fetch {dataset_id}: {e}")
                failed.append(dataset_id)

                if stop_on_error:
                    raise

        if invalid:
            log.warning(f"Skipped {len(invalid)} invalid IDs: {invalid}")

        if failed:
            log.warning(f"Failed to fetch {len(failed)} datasets: {failed}")

        log.info(f"✓ Successfully fetched {len(all_raw_datasets)} total datasets")

        return all_raw_datasets


# ============================================================================
# Series Matrix-based Source (Fast but Incomplete) - UNCHANGED
# ============================================================================


class NCBIGeoSource(DataSource):
    """
    Fetch GEO datasets using Series Matrix files (SUBSET of samples).

    This is faster than GEOparseSource but may not include all samples.
    It wraps the existing :class:`NCBIFetcher` implementation; prefer
    GEOparseSource for complete datasets and R parity.

    Examples:
        >>> source = NCBIGeoSource()
        >>> raw_datasets = source.fetch('GSE11909')
        >>> # Returns 1 raw dataset: GPL96 (60 samples only - missing 58!)
    """

    def __init__(self, cache_dir: str | Path = "~/.geo_cache"):
        """Initialize the source.

        Args:
            cache_dir: Directory for caching downloaded files.
        """
        super().__init__(cache_dir)

        # Import here to avoid circular dependency
        from .ncbi_fetcher import NCBIFetcher

        self.fetcher = NCBIFetcher(cache_dir=str(cache_dir))

    @property
    def source_name(self) -> str:
        """Human-readable name of this data source."""
        return "NCBI GEO (Series Matrix)"

    def fetch(self, dataset_id: str, **kwargs) -> List[RawDataset]:
        """Fetch a GEO dataset via Series Matrix (may be incomplete).

        Warning:
            Series Matrix files may contain only a subset of samples. Use
            GEOparseSource for complete datasets.

        Args:
            dataset_id: GEO Series ID, e.g. ``'GSE11909'``.
            **kwargs: Additional arguments (currently unused).

        Returns:
            One ``RawDataset`` per platform (may have fewer samples than
            the SOFT-based sources).
        """
        # Validate
        validated_id = validate_series_id(dataset_id)

        log.info(f"Fetching {validated_id} via Series Matrix...")

        try:
            # Fetch using existing NCBIFetcher
            matrices = self.fetcher.fetch_series_matrix(validated_id)

            raw_datasets = []

            for platform_id, matrix in matrices.items():
                raw = RawDataset(
                    source_id=validated_id,
                    platform_id=platform_id,
                    raw_data={"matrix": matrix},
                    metadata={
                        "gse_id": validated_id,
                        "platform": platform_id,
                        "n_samples": len(matrix.columns),
                        "source": self.source_name,
                        "format": "SeriesMatrix",
                    },
                )
                raw_datasets.append(raw)

            log.info(f"✓ Fetched {validated_id}: {len(raw_datasets)} platforms")

            return raw_datasets

        except Exception as e:
            log.error(f"Failed to fetch {validated_id}: {e}")
            raise RuntimeError(f"Series Matrix fetch failed for {validated_id}: {e}") from e

    def fetch_multiple(
        self, dataset_ids: List[str], stop_on_error: bool = False, **kwargs
    ) -> List[RawDataset]:
        """Fetch multiple GEO datasets via Series Matrix.

        Args:
            dataset_ids: GEO Series IDs.
            stop_on_error: If True, raise on the first failure.
            **kwargs: Additional arguments (currently unused).

        Returns:
            All raw datasets across every GSE ID.
        """
        all_raw_datasets = []
        failed = []
        invalid = []

        for dataset_id in dataset_ids:
            # Validate first
            try:
                validate_series_id(dataset_id)
            except ValueError as e:
                log.error(f"Invalid dataset ID '{dataset_id}': {e}")
                invalid.append(dataset_id)
                if stop_on_error:
                    raise
                continue

            try:
                raw_datasets = self.fetch(dataset_id, **kwargs)
                all_raw_datasets.extend(raw_datasets)

            except Exception as e:
                log.error(f"Failed to fetch {dataset_id}: {e}")
                failed.append(dataset_id)

                if stop_on_error:
                    raise

        if invalid:
            log.warning(f"Skipped {len(invalid)} invalid IDs: {invalid}")

        if failed:
            log.warning(f"Failed to fetch {len(failed)} datasets: {failed}")

        log.info(f"✓ Successfully fetched {len(all_raw_datasets)} total datasets")

        return all_raw_datasets


# ============================================================================
# Convenience Functions
# ============================================================================


def create_geo_source(
    strategy: str = "full", cache_dir: str | Path = "~/.geo_cache", **kwargs
) -> DataSource:
    """Create the appropriate GEO data source for a strategy.

    Args:
        strategy: ``'full'`` uses GEOparseSource (SOFT files, all
            samples); ``'fast'`` uses NCBIGeoSource (Series Matrix,
            subset).
        cache_dir: Cache directory.
        **kwargs: Additional arguments passed to the source constructor.

    Returns:
        A configured data source.

    Examples:
        >>> source = create_geo_source('full')  # full dataset (R parity)
        >>> source = create_geo_source('fast')  # fast but incomplete
    """
    if strategy == "full":
        return GEOparseSource(cache_dir=cache_dir, **kwargs)
    elif strategy == "fast":
        return NCBIGeoSource(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'full' or 'fast'")
