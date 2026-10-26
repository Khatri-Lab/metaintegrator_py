"""
Low-level NCBI network operations with retry, timeout, and disk caching.

Encapsulates directory listing (HTTPS first, FTP fallback), HTTPS file
download (with FTP fallback), and GEO-specific path construction.

HTTPS comes first because many networks block plain FTP (port 21). NCBI
serves the same tree over HTTPS, with an HTML index page per folder.  Designed for
dependency injection so that callers (e.g. :class:`RParityGeoSource`)
can be unit-tested with a mock network layer: no live connections
required.

Usage
-----
    from preprocessing.ncbi_network import NCBINetwork

    net = NCBINetwork(cache_dir='~/.geo_cache')

    # List Series Matrix files for a GSE (HTTPS first, FTP fallback)
    ftp_path = NCBINetwork.geo_series_ftp_path('GSE11909', 'matrix')
    files = net.list_directory(ftp_path, suffix='_series_matrix.txt.gz')

    # Download with disk caching
    local_path = net.download_file(ftp_path, files[0])

    # Download as bytes (HTTPS primary, FTP fallback, with retry)
    raw_bytes = net.download_bytes(ftp_path, files[0])
"""

from __future__ import annotations

import ftplib
import gzip
import io
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

#: ``href`` values in an NCBI (Apache-style) folder index page.
_HREF_PATTERN = re.compile(r'href\s*=\s*"([^"]+)"', re.IGNORECASE)


class DirectoryNotFoundError(ConnectionError):
    """The folder does not exist on the server (HTTP 404).

    A subclass of :class:`ConnectionError`, so callers that already catch
    listing failures keep working. :meth:`NCBINetwork.list_directory` does
    not retry or fall back to FTP for it, because FTP would not find the
    folder either.
    """


class NCBINetwork:
    """
    Low-level NCBI network operations with retry and caching.

    All network I/O goes through this class, making it the single seam to
    mock for offline testing.

    Examples:
        >>> net = NCBINetwork(cache_dir='~/.geo_cache')
        >>> ftp_path = NCBINetwork.geo_series_ftp_path('GSE11909', 'matrix')
        >>> files = net.list_directory(ftp_path, suffix='_series_matrix.txt.gz')
        >>> local = net.download_file(ftp_path, files[0])
    """

    DEFAULT_FTP_HOST = "ftp.ncbi.nlm.nih.gov"

    def __init__(
        self,
        cache_dir: str | Path,
        ftp_host: str = DEFAULT_FTP_HOST,
        ftp_timeout: int = 30,
        http_timeout: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        """Initialize the network layer.

        Args:
            cache_dir: Directory for caching downloaded files on disk.
            ftp_host: NCBI FTP hostname (also used for HTTPS mirror URLs).
            ftp_timeout: Timeout in seconds for FTP connections.
            http_timeout: Timeout in seconds for HTTPS downloads.
            max_retries: Maximum retry attempts for failed downloads.
            retry_backoff: Exponential backoff base, in seconds, between
                retries.
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ftp_host = ftp_host
        self.ftp_timeout = ftp_timeout
        self.http_timeout = http_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    # ------------------------------------------------------------------ #
    #  Directory listing: HTTPS first, FTP fallback                       #
    # ------------------------------------------------------------------ #

    def list_directory(
        self,
        ftp_path: str,
        suffix: Optional[str] = None,
    ) -> List[str]:
        """List files in an NCBI folder, over HTTPS first and FTP second.

        Use this instead of :meth:`list_ftp_directory`. Plain FTP is often
        blocked by firewalls, while HTTPS to NCBI usually works.

        Args:
            ftp_path: Path relative to the server root, e.g.
                ``'geo/series/GSE11nnn/GSE11909/matrix'``.
            suffix: If given, only return filenames ending with this
                suffix, e.g. ``'_series_matrix.txt.gz'``.

        Returns:
            Filenames (not full paths) in the folder.

        Raises:
            DirectoryNotFoundError: If HTTPS reports that the folder does
                not exist. FTP is not tried in that case.
            ConnectionError: If both HTTPS and FTP fail. The message
                names both errors.
        """
        try:
            return self.list_https_directory(ftp_path, suffix=suffix)
        except DirectoryNotFoundError:
            raise
        except ConnectionError as https_error:
            log.warning(f"HTTPS listing failed for {ftp_path}, trying FTP: {https_error}")
            try:
                return self.list_ftp_directory(ftp_path, suffix=suffix)
            except ConnectionError as ftp_error:
                raise ConnectionError(
                    f"Listing failed for {ftp_path}. " f"HTTPS: {https_error}. FTP: {ftp_error}"
                ) from ftp_error

    def list_https_directory(
        self,
        ftp_path: str,
        suffix: Optional[str] = None,
    ) -> List[str]:
        """List files in an NCBI folder by reading its HTTPS index page.

        Args:
            ftp_path: Path relative to the server root, e.g.
                ``'geo/series/GSE11nnn/GSE11909/matrix'``.
            suffix: If given, only return filenames ending with this
                suffix.

        Returns:
            Filenames (not full paths) in the folder, in page order.
            Subfolders and navigation links are left out.

        Raises:
            DirectoryNotFoundError: On HTTP 404. Not retried.
            ConnectionError: If every attempt fails for another reason.
        """
        url = f"https://{self.ftp_host}/{ftp_path.strip('/')}/"
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(url, timeout=self.http_timeout) as resp:
                    html = resp.read().decode("utf-8", errors="replace")
                files = self.parse_index_page(html)
                if suffix:
                    files = [f for f in files if f.endswith(suffix)]
                log.info(
                    f"HTTPS listing {ftp_path}: {len(files)} file(s)"
                    + (f" matching '*{suffix}'" if suffix else "")
                )
                return files
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise DirectoryNotFoundError(f"Folder not found: {url}") from e
                last_error = e
            except Exception as e:  # network errors, timeouts, bad pages
                last_error = e
            log.warning(
                f"HTTPS listing attempt {attempt + 1}/{self.max_retries} "
                f"failed for {ftp_path}: {last_error}"
            )
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_backoff**attempt)

        raise ConnectionError(
            f"HTTPS listing failed for {ftp_path} after "
            f"{self.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def parse_index_page(html: str) -> List[str]:
        """Extract filenames from an Apache-style folder index page.

        Keeps plain relative links only. Leaves out subfolders (ending in
        ``/``), the parent link, absolute paths, full URLs, sort links
        (``?C=N;O=D``) and anchors.

        Args:
            html: The page's HTML text.

        Returns:
            Unique filenames, in page order, with URL escapes decoded.

        Examples:
            >>> NCBINetwork.parse_index_page(
            ...     '<a href="../">Parent</a> <a href="?C=M;O=A">Sort</a>'
            ...     '<a href="GSE1-GPL96_series_matrix.txt.gz">x</a>'
            ...     '<a href="sub/">sub</a>'
            ... )
            ['GSE1-GPL96_series_matrix.txt.gz']
        """
        names: List[str] = []
        for href in _HREF_PATTERN.findall(html):
            if href.startswith(("?", "#", "/")) or "://" in href or "/" in href:
                continue
            name = urllib.parse.unquote(href)
            if name and name not in names:
                names.append(name)
        return names

    # ------------------------------------------------------------------ #
    #  FTP directory listing                                              #
    # ------------------------------------------------------------------ #

    def list_ftp_directory(
        self,
        ftp_path: str,
        suffix: Optional[str] = None,
    ) -> List[str]:
        """List files in an NCBI FTP directory over plain FTP (port 21).

        Prefer :meth:`list_directory`, which tries HTTPS first. This method
        stays as the fallback and for callers that need FTP explicitly.

        Args:
            ftp_path: Path relative to the FTP root, e.g.
                ``'geo/series/GSE11nnn/GSE11909/matrix'``.
            suffix: If given, only return filenames ending with this
                suffix, e.g. ``'_series_matrix.txt.gz'``.

        Returns:
            Filenames (not full paths) in the directory.

        Raises:
            ConnectionError: If the FTP connection fails after retries.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                with ftplib.FTP(self.ftp_host, timeout=self.ftp_timeout) as ftp:
                    ftp.login()
                    ftp.cwd(ftp_path)
                    files = ftp.nlst()

                if suffix:
                    files = [f for f in files if f.endswith(suffix)]

                log.info(
                    f"FTP listing {ftp_path}: {len(files)} file(s)"
                    + (f" matching '*{suffix}'" if suffix else "")
                )
                return files

            except Exception as e:
                last_error = e
                log.warning(f"FTP listing attempt {attempt + 1}/{self.max_retries} " f"failed: {e}")
                if attempt < self.max_retries - 1:
                    wait = self.retry_backoff**attempt
                    time.sleep(wait)

        raise ConnectionError(
            f"FTP listing failed for {ftp_path} after " f"{self.max_retries} attempts: {last_error}"
        )

    # ------------------------------------------------------------------ #
    #  File download (HTTPS with disk caching)                            #
    # ------------------------------------------------------------------ #

    def download_file(
        self,
        ftp_path: str,
        filename: str,
    ) -> Path:
        """Download a file via HTTPS and cache it on disk.

        If the file already exists in :attr:`cache_dir`, the cached
        version is returned immediately with no network call.

        Args:
            ftp_path: FTP directory path (used to build the HTTPS mirror
                URL).
            filename: Name of the file to download.

        Returns:
            Local path to the (possibly cached) file.

        Raises:
            RuntimeError: If the download fails after retries.
        """
        local = self.cache_dir / filename
        if local.exists():
            log.info(f"    Using cached: {filename}")
            return local

        url = f"https://{self.ftp_host}/{ftp_path}/{filename}"
        log.info(f"    Downloading {filename}...")

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                urllib.request.urlretrieve(url, str(local))
                log.info(f"    ✓ Downloaded {filename}")
                return local
            except Exception as e:
                last_error = e
                log.warning(
                    f"    HTTPS download attempt {attempt + 1}/" f"{self.max_retries} failed: {e}"
                )
                # Remove partial file
                if local.exists():
                    local.unlink()
                if attempt < self.max_retries - 1:
                    wait = self.retry_backoff**attempt
                    time.sleep(wait)

        raise RuntimeError(
            f"Download failed for {filename} after " f"{self.max_retries} attempts: {last_error}"
        )

    # ------------------------------------------------------------------ #
    #  Bytes download (HTTPS primary, FTP fallback)                       #
    # ------------------------------------------------------------------ #

    def download_bytes(
        self,
        ftp_path: str,
        filename: str,
    ) -> bytes:
        """Download a file and return its contents as bytes.

        Uses HTTPS as the primary method, falling back to FTP with retry
        if HTTPS fails. Validates gzip integrity when the filename ends
        with ``.gz``.

        Args:
            ftp_path: FTP directory path.
            filename: Name of the file to download.

        Returns:
            The raw file contents.

        Raises:
            RuntimeError: If both HTTPS and FTP fail.
        """
        is_gz = filename.endswith(".gz")

        # --- HTTPS (primary) ---
        https_error: Optional[Exception] = None
        url = f"https://{self.ftp_host}/{ftp_path}/{filename}"
        try:
            log.info(f"Downloading {filename} via HTTPS...")
            with urllib.request.urlopen(url, timeout=self.http_timeout) as resp:
                data = resp.read()
            if is_gz:
                self._validate_gzip(data, "HTTPS")
            log.info(f"✓ HTTPS download: {len(data)} bytes")
            return data
        except Exception as e:
            https_error = e
            log.warning(f"HTTPS download failed: {e}")

        # --- FTP fallback ---
        log.info("Falling back to FTP...")
        last_ftp_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                buf = bytearray()
                with ftplib.FTP(self.ftp_host, timeout=self.ftp_timeout) as ftp:
                    ftp.login()
                    ftp.set_pasv(True)
                    ftp.cwd(ftp_path)
                    ftp.retrbinary(
                        f"RETR {filename}",
                        buf.extend,
                        blocksize=262144,
                    )
                data = bytes(buf)
                if is_gz:
                    self._validate_gzip(data, "FTP")
                log.info(f"✓ FTP download: {len(data)} bytes")
                return data
            except Exception as e:
                last_ftp_error = e
                log.warning(f"FTP attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    wait = self.retry_backoff**attempt
                    time.sleep(wait)

        raise RuntimeError(
            f"Download failed for {filename} via both HTTPS and FTP. "
            f"HTTPS error: {https_error}; FTP error: {last_ftp_error}"
        )

    # ------------------------------------------------------------------ #
    #  GEO path helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def geo_series_ftp_path(gse_id: str, subpath: str = "matrix") -> str:
        """Build the FTP directory path for a GEO series.

        Args:
            gse_id: GEO Series ID, e.g. ``'GSE11909'``.
            subpath: Subdirectory within the series (``'matrix'``,
                ``'soft'``, ``'suppl'``).

        Returns:
            The FTP path, e.g. ``'geo/series/GSE11nnn/GSE11909/matrix'``.

        Examples:
            >>> NCBINetwork.geo_series_ftp_path('GSE11909', 'matrix')
            'geo/series/GSE11nnn/GSE11909/matrix'
            >>> NCBINetwork.geo_series_ftp_path('GSE50635', 'soft')
            'geo/series/GSE50nnn/GSE50635/soft'
        """
        gse_num = gse_id.replace("GSE", "")
        subdir = f"GSE{gse_num[:-3]}nnn"
        return f"geo/series/{subdir}/{gse_id}/{subpath}"

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_gzip(data: bytes, source_label: str) -> None:
        """Verify that ``data`` is a valid, non-empty gzip stream.

        Raises:
            ValueError: If the data is not valid gzip, or is empty.
        """
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                test_read = gz.read(1024)
                if len(test_read) == 0:
                    raise ValueError("Gzip file is empty")
        except Exception as e:
            raise ValueError(f"Invalid gzip data from {source_label}: {e}") from e
