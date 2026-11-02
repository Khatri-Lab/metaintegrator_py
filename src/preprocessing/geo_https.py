"""Download GEO files over HTTPS before GEOparse looks for them.

``GEOparse.get_GEO(geo=..., destdir=...)`` downloads a series' family SOFT
file from ``ftp://ftp.ncbi.nlm.nih.gov/...``. Many networks block FTP.
There, every call waits for its timeout and then fails, so no series can
be loaded at all.

GEOparse skips the download when the file is already in ``destdir``. This
module puts it there first, over HTTPS, under the exact name GEOparse
looks for. GEOparse then only parses it. NCBI serves the same files over
HTTPS from the same host.

Use :func:`prefetch_geo_file_or_warn` right before ``GEOparse.get_GEO``.
When the HTTPS download fails, it logs a warning and returns ``None``, so
GEOparse still tries its own FTP download, as before.

Supported accessions:

- ``GSE...``: the family SOFT file, ``<GSE>_family.soft.gz``.
- ``GPL...``: the platform table GEOparse reads by default
  (``include_data=False``, ``annotate_gpl=False``, ``how="full"``),
  ``<GPL>.txt``.

Other accessions (``GSM``, ``GDS``) are left to GEOparse.
"""

from __future__ import annotations

import gzip
import logging
import os
import re
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import IO, Optional

log = logging.getLogger(__name__)

#: Host of NCBI's file tree, served over HTTPS as well as FTP.
NCBI_FILES_HOST = "ftp.ncbi.nlm.nih.gov"
#: GEO's accession viewer, which GEOparse uses for platform tables.
GEO_ACC_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"

#: Seconds to wait for a connection or for the next block of data.
DEFAULT_TIMEOUT = 120
_CHUNK_BYTES = 1 << 20
_SERIES = re.compile(r"^GSE\d+$")
_PLATFORM = re.compile(r"^GPL\d+$")

#: Opens a URL, like ``urllib.request.urlopen``. Tests pass a fake.
Opener = Callable[..., IO[bytes]]


def _range_folder(accession: str) -> str:
    """GEO's folder for a range of 1,000 accessions.

    Examples:
        >>> _range_folder("GSE11909")
        'GSE11nnn'
        >>> _range_folder("GSE123")
        'GSEnnn'
    """
    prefix, number = accession[:3], accession[3:]
    return f"{prefix}{number[:-3]}nnn"


def geoparse_file_name(accession: str) -> Optional[str]:
    """The file name GEOparse looks for in ``destdir``.

    Args:
        accession: A GEO accession, such as ``"GSE11909"``.

    Returns:
        The file name, or ``None`` for accessions this module does not
        download.

    Examples:
        >>> geoparse_file_name("GSE11909")
        'GSE11909_family.soft.gz'
        >>> geoparse_file_name("GPL96")
        'GPL96.txt'
        >>> geoparse_file_name("GSM1") is None
        True
    """
    if _SERIES.match(accession):
        return f"{accession}_family.soft.gz"
    if _PLATFORM.match(accession):
        return f"{accession}.txt"
    return None


def geo_https_url(accession: str) -> Optional[str]:
    """The HTTPS address of the file GEOparse would download.

    Args:
        accession: A GEO accession.

    Returns:
        The URL, or ``None`` for accessions this module does not download.

    Examples:
        >>> geo_https_url("GSE11909")
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE11nnn/GSE11909/soft/GSE11909_family.soft.gz'
        >>> geo_https_url("GPL96")
        'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?targ=self&acc=GPL96&form=text&view=full'
    """
    if _SERIES.match(accession):
        return (
            f"https://{NCBI_FILES_HOST}/geo/series/{_range_folder(accession)}/"
            f"{accession}/soft/{accession}_family.soft.gz"
        )
    if _PLATFORM.match(accession):
        return f"{GEO_ACC_URL}?targ=self&acc={accession}&form=text&view=full"
    return None


def prefetch_geo_file(
    accession: str,
    destdir: str | Path,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    opener: Opener = urllib.request.urlopen,
) -> Optional[Path]:
    """Download one GEO file over HTTPS into ``destdir``, unless it is there.

    The file is written to a temporary name in ``destdir`` and renamed
    when complete, so GEOparse never sees half a file. A ``.gz`` file is
    read through once before the rename, so a cut-off download is caught.

    Args:
        accession: A GEO accession, such as ``"GSE11909"``. Upper case.
        destdir: The ``destdir`` that will be passed to ``GEOparse.get_GEO``.
            Used as given (``~`` is not expanded), so both look in the same
            folder. Created if missing.
        timeout: Seconds to wait for a connection or the next block of data.
        opener: Opens the URL. Only tests change it.

    Returns:
        The file's path, or ``None`` when this module does not handle the
        accession.

    Raises:
        OSError: The download failed, including timeouts and HTTP errors
            (``urllib.error.URLError`` is an ``OSError``).
        ValueError: A ``.gz`` download was not valid gzip.
    """
    name = geoparse_file_name(accession)
    url = geo_https_url(accession)
    if name is None or url is None:
        return None

    folder = Path(destdir)
    target = folder / name
    if target.is_file() and target.stat().st_size > 0:
        return target

    folder.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".part", dir=folder)
    temp = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as out, opener(url, timeout=timeout) as response:
            while True:
                block = response.read(_CHUNK_BYTES)
                if not block:
                    break
                out.write(block)
        if name.endswith(".gz"):
            _check_gzip(temp)
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    log.info("Downloaded %s over HTTPS to %s", accession, target)
    return target


def prefetch_geo_file_or_warn(
    accession: str,
    destdir: str | Path,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    opener: Opener = urllib.request.urlopen,
) -> Optional[Path]:
    """Like :func:`prefetch_geo_file`, but never raises.

    For use right before ``GEOparse.get_GEO``: when HTTPS fails, GEOparse
    still tries its own FTP download.

    Args:
        accession: A GEO accession.
        destdir: The ``destdir`` passed to ``GEOparse.get_GEO``.
        timeout: Seconds to wait for a connection or the next block of data.
        opener: Opens the URL. Only tests change it.

    Returns:
        The file's path, or ``None`` when the accession is not handled or
        the download failed.
    """
    try:
        return prefetch_geo_file(accession, destdir, timeout=timeout, opener=opener)
    except Exception as exc:  # noqa: BLE001 - GEOparse's FTP download is the fallback
        log.warning(
            "HTTPS download of %s failed (%s: %s). GEOparse will try FTP.",
            accession,
            type(exc).__name__,
            exc,
        )
        return None


def _check_gzip(path: Path) -> None:
    """Raise ``ValueError`` unless ``path`` is a complete, non-empty gzip file.

    Reads the whole stream, so a cut-off download fails here (its end and
    checksum are missing) instead of later, inside GEOparse.
    """
    try:
        total = 0
        with gzip.open(path, "rb") as stream:
            while block := stream.read(_CHUNK_BYTES):
                total += len(block)
        if total == 0:
            raise ValueError("empty gzip file")
    except (OSError, EOFError) as exc:
        raise ValueError(f"not a valid gzip file: {exc}") from exc
