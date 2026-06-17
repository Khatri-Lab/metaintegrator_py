# preprocessing/ncbi_fetcher.py
"""
Concrete NCBIFetcher for the 'fast' strategy.

Downloads GEO Series Matrix files, parses them into per-platform
DataFrames, and caches results on disk via pickle.

Network I/O for HTTPS/FTP downloads is delegated to
:class:`NCBINetwork`.  The FTP directory listing (with file sizes)
still uses raw ``ftplib`` because :class:`NCBINetwork` does not yet
expose a ``list_with_sizes()`` method.

Used internally by :class:`NCBIGeoSource` (the "fast" strategy).
"""

from __future__ import annotations

import csv
import ftplib
import gzip
import io
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .ncbi_network import NCBINetwork

log = logging.getLogger(__name__)


# ============================================================================
# Simple pickle-based disk cache (replaces removed CacheManager)
# ============================================================================


class _SimpleCache:
    """Minimal pickle-based disk cache keyed by string."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe_key = re.sub(r"[^\w\-]", "_", key)
        return self.cache_dir / f"{safe_key}.pkl"

    def get(self, key: str) -> Optional[object]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            log.warning(f"Cache read failed for {key}, removing {p}")
            p.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: object) -> None:
        p = self._path(key)
        try:
            with open(p, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            log.warning(f"Cache write failed for {key}: {e}")


# ============================================================================
# NCBIFetcher
# ============================================================================


class NCBIFetcher:
    """
    Lightweight downloader for GEO series matrices and minimal metadata.

    Uses:
      * ftplib for matrix file listing (with sizes)
      * NCBINetwork for HTTPS/FTP downloads
      * _SimpleCache for pickle-based disk caching
    """

    FTP_HOST = "ftp.ncbi.nlm.nih.gov"
    FTP_BASE = "geo/series"
    MINIML_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series"

    # File size guard
    MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

    # Simple retry policy
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2  # seconds

    def __init__(self, cache_dir: str | Path = "~/.geo_cache"):
        """
        Parameters
        ----------
        cache_dir : str or Path
            Directory for caching downloaded files and result pickles.
        """
        cache_path = Path(cache_dir).expanduser()
        self.cache = _SimpleCache(cache_path)

        # Shared network layer for HTTPS/FTP downloads
        self._network = NCBINetwork(cache_dir=cache_path)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def fetch_series_matrix(self, gse_id: str) -> Dict[str, pd.DataFrame]:
        """
        Return mapping {platform_id: DataFrame} for the GSE.

        The matrix files are cached on disk; if a matrix is already present
        the cached version is returned.

        FIXED: FTP connection is closed immediately after listing files,
        before HTTPS downloads begin. This prevents "421 Idle timeout" errors.
        """
        log.info(f"Fetching series matrix for {gse_id}")

        key = f"matrix_{gse_id}"
        cached = self.cache.get(key)
        if cached is not None:
            log.info(f"Using cached data for {gse_id}")
            return cached

        # 1. locate the FTP path
        gse_num = int(gse_id[3:])
        series_dir = f"GSE{gse_num // 1000}nnn"
        ftp_path = f"{self.FTP_BASE}/{series_dir}/{gse_id}/matrix"

        # 2. Get file list and sizes, then CLOSE FTP connection before downloading
        matrix_files = []
        file_sizes = {}

        with ftplib.FTP(self.FTP_HOST) as ftp:
            ftp.login()
            ftp.cwd(ftp_path)
            all_files = ftp.nlst()

            # Filter for series matrix files
            matrix_files = [f for f in all_files if f.endswith("_series_matrix.txt.gz")]

            if not matrix_files:
                raise ValueError(f"No matrix files found for {gse_id}")

            log.info(f"Found {len(matrix_files)} matrix files: {matrix_files}")

            # Get file sizes while FTP connection is still open
            for mfile in matrix_files:
                try:
                    size = ftp.size(mfile)
                except Exception:
                    size = self._size_from_listing(ftp, mfile)
                file_sizes[mfile] = size
                log.info(f"  {mfile}: {size} bytes")

        # FTP connection now closed - no idle timeout possible!
        log.info("FTP connection closed, proceeding with HTTPS downloads...")

        # 3. Download files using HTTPS (via NCBINetwork)
        result: Dict[str, pd.DataFrame] = {}

        for mfile in matrix_files:
            # Check size
            size = file_sizes.get(mfile, 0)
            if size and size > self.MAX_DOWNLOAD_SIZE:
                log.warning(
                    "Skipping %s: %s bytes > %s",
                    mfile,
                    size,
                    self.MAX_DOWNLOAD_SIZE,
                )
                continue

            # Download file (HTTPS with FTP fallback, via NCBINetwork)
            gz_bytes = self._network.download_bytes(ftp_path, mfile)

            # Parse into DataFrame and extract platform IDs from file content
            df, platform_ids = self._parse_series_matrix(gz_bytes)

            # Use platform IDs from file content (most reliable)
            # Fallback to filename extraction if none found
            if not platform_ids:
                log.warning(
                    f"No platform IDs in {mfile} metadata, trying filename extraction"
                )
                platform_ids = [self._probe_platform_from_filename(mfile)]

            # Handle multiple platforms in one file
            for platform_id in platform_ids:
                if platform_id == "unknown":
                    log.error(
                        f"Could not determine platform ID for {mfile}. "
                        f"File may be malformed or missing metadata."
                    )
                result[platform_id] = df
                log.info(f"Loaded {mfile} -> platform {platform_id}")

        if not result:
            raise ValueError(f"All matrix files for {gse_id} were too large")

        # Cache the result
        self.cache.set(key, result)

        return result

    def fetch_platform_annotation(self, platform_id: str) -> pd.DataFrame:
        """
        Fallback – returns an empty DataFrame.
        The real mapping comes from GeneAnnotator; kept for API parity.
        """
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _size_from_listing(self, ftp: ftplib.FTP, filename: str) -> int:
        """
        Return the size of *filename* by parsing the FTP LIST
        (or MLSD) output.  If the size cannot be determined, return 0.
        """
        # Try MLSD first
        try:
            for name, facts in ftp.mlsd():
                if name == filename and "size" in facts:
                    return int(facts["size"])
        except Exception:
            pass

        # Fallback: parse the output of LIST
        size = 0

        def callback(line: str):
            nonlocal size
            parts = line.split()
            if len(parts) >= 5 and parts[4].isdigit() and parts[-1] == filename:
                size = int(parts[4])

        try:
            ftp.retrlines("LIST", callback)
        except Exception:
            pass

        return size

    @staticmethod
    def _parse_series_matrix(gz_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
        """Parse a gzipped GEO series matrix into a tidy DataFrame and extract platform IDs.

        Returns:
            Tuple of (DataFrame, list of platform IDs)
            Platform IDs are extracted from !Series_platform_id lines in the file.
        """
        with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes)) as gz:
            txt = gz.read().decode("utf-8")

        # Extract platform IDs from metadata lines
        platform_ids = []
        for line in txt.splitlines():
            if line.startswith("!Series_platform_id"):
                match = re.search(r"GPL\d+", line)
                if match:
                    platform_ids.append(match.group(0))

        if not platform_ids:
            log.warning("No platform IDs found in series matrix metadata")

        # Parse data rows (skip metadata lines starting with !)
        rows = [row for row in txt.splitlines() if row and not row.startswith("!")]
        if not rows:
            raise ValueError("Series matrix contains no data lines")

        # Strip quotes from ALL fields
        header = [c.strip('"') for c in rows[0].split("\t")]
        data = [[c.strip('"') for c in row.split("\t")] for row in rows[1:]]
        df = pd.DataFrame(data, columns=header)

        # Convert numeric columns to float64 explicitly
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        # Verify conversion
        non_numeric_cols = [col for col in df.columns[1:] if df[col].dtype != "float64"]
        if non_numeric_cols:
            log.warning(f"Could not convert columns to float64: {non_numeric_cols}")

        # Keep probe ID as index
        df.set_index(df.columns[0], inplace=True)

        return df, platform_ids

    @staticmethod
    def _probe_platform_from_filename(filename: str) -> str:
        """
        Extract GPL code from matrix file name.
        Example: GSE11909-GPL96_series_matrix.txt.gz → GPL96
        """
        match = re.search(r"(GPL\d+)", filename)
        return match.group(1) if match else "unknown"

    # ------------------------------------------------------------------ #
    # Minimal metadata fetcher
    # ------------------------------------------------------------------ #
    def fetch_minimal_metadata(self, gse_id: str) -> dict:
        """
        Return minimal metadata: title, summary, platform list.
        Uses the MINiML XML file (small, ~1–10 MB).
        """
        import requests

        key = f"metadata_{gse_id}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        gse_num = int(gse_id[3:])
        series_dir = f"GSE{gse_num // 1000}nnn"
        url = f"{self.MINIML_URL}/{series_dir}/{gse_id}/MINiML.txt"

        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(f"Failed to download MINiML for {gse_id}") from e

        metadata = {
            "title": self._extract_tag(r.text, "title"),
            "summary": self._extract_tag(r.text, "summary"),
            "platforms": self._extract_platforms(r.text),
        }

        self.cache.set(key, metadata)
        return metadata

    @staticmethod
    def _extract_tag(xml: str, tag: str) -> str:
        """Very naive XML extraction – enough for MINiML tags."""
        pattern = rf"<{tag}>(.*?)</{tag}>"
        m = re.search(pattern, xml, flags=re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_platforms(xml: str) -> list[str]:
        """Return list of platform IDs (GPL numbers) from MINiML."""
        return re.findall(r"<platformId>(GPL\d+)</platformId>", xml)
