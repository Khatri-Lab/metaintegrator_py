"""Tests for ``preprocessing/geo_https.py``.

GEOparse downloads series over FTP. On a network that blocks FTP, every
series load waited for its timeouts and failed. These tests pin the HTTPS
download that now runs first, without any network access.
"""

from __future__ import annotations

import gzip
import io
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from preprocessing import geo_https
from preprocessing.geo_https import (
    geo_https_url,
    geoparse_file_name,
    prefetch_geo_file,
    prefetch_geo_file_or_warn,
)

SOFT_TEXT = b"^SERIES = GSE11909\n!Series_title = test\n"


class _FakeResponse(io.BytesIO):
    """A urlopen response: a byte stream that is also a context manager."""


def _opener(payload: bytes, calls: list[tuple[str, float]] | None = None):
    def _open(url: str, timeout: float) -> _FakeResponse:
        if calls is not None:
            calls.append((url, timeout))
        return _FakeResponse(payload)

    return _open


def _failing_opener(url: str, timeout: float):
    raise urllib.error.URLError("timed out")


class TestNamesAndUrls:
    def test_series(self) -> None:
        assert geoparse_file_name("GSE11909") == "GSE11909_family.soft.gz"
        assert geo_https_url("GSE11909") == (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE11nnn/GSE11909/soft/"
            "GSE11909_family.soft.gz"
        )

    def test_series_below_one_thousand(self) -> None:
        assert "/geo/series/GSEnnn/GSE123/soft/" in geo_https_url("GSE123")

    def test_platform(self) -> None:
        assert geoparse_file_name("GPL96") == "GPL96.txt"
        url = geo_https_url("GPL96")
        assert url.startswith("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?")
        assert "acc=GPL96" in url and "view=full" in url

    @pytest.mark.parametrize("accession", ["GSM1", "GDS5", "gse11909", "GSE", "x"])
    def test_other_accessions_are_not_handled(self, accession: str) -> None:
        assert geoparse_file_name(accession) is None
        assert geo_https_url(accession) is None


class TestPrefetch:
    def test_downloads_series_under_geoparse_name(self, tmp_path: Path) -> None:
        calls: list[tuple[str, float]] = []

        path = prefetch_geo_file(
            "GSE11909", tmp_path, opener=_opener(gzip.compress(SOFT_TEXT), calls)
        )

        assert path == tmp_path / "GSE11909_family.soft.gz"
        assert gzip.decompress(path.read_bytes()) == SOFT_TEXT
        assert calls[0][0].startswith("https://")
        # No temporary file is left behind.
        assert [p.name for p in tmp_path.iterdir()] == ["GSE11909_family.soft.gz"]

    def test_downloads_platform_table(self, tmp_path: Path) -> None:
        path = prefetch_geo_file("GPL96", tmp_path, opener=_opener(b"^PLATFORM = GPL96\n"))
        assert path == tmp_path / "GPL96.txt"
        assert path.read_bytes() == b"^PLATFORM = GPL96\n"

    def test_existing_file_is_used_without_a_download(self, tmp_path: Path) -> None:
        existing = tmp_path / "GSE11909_family.soft.gz"
        existing.write_bytes(gzip.compress(SOFT_TEXT))
        opener = MagicMock()

        assert prefetch_geo_file("GSE11909", tmp_path, opener=opener) == existing
        opener.assert_not_called()

    def test_empty_existing_file_is_downloaded_again(self, tmp_path: Path) -> None:
        (tmp_path / "GSE11909_family.soft.gz").write_bytes(b"")

        path = prefetch_geo_file("GSE11909", tmp_path, opener=_opener(gzip.compress(SOFT_TEXT)))

        assert gzip.decompress(path.read_bytes()) == SOFT_TEXT

    def test_cut_off_gzip_is_rejected_and_not_kept(self, tmp_path: Path) -> None:
        truncated = gzip.compress(SOFT_TEXT * 100)[:-12]

        with pytest.raises(ValueError, match="gzip"):
            prefetch_geo_file("GSE11909", tmp_path, opener=_opener(truncated))

        assert list(tmp_path.iterdir()) == []

    def test_network_error_leaves_no_file(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            prefetch_geo_file("GSE11909", tmp_path, opener=_failing_opener)
        assert list(tmp_path.iterdir()) == []

    def test_creates_the_folder(self, tmp_path: Path) -> None:
        folder = tmp_path / "new" / "cache"
        prefetch_geo_file("GSE11909", folder, opener=_opener(gzip.compress(SOFT_TEXT)))
        assert (folder / "GSE11909_family.soft.gz").is_file()

    def test_unhandled_accession_returns_none(self, tmp_path: Path) -> None:
        opener = MagicMock()
        assert prefetch_geo_file("GSM1", tmp_path, opener=opener) is None
        opener.assert_not_called()

    def test_timeout_is_passed_on(self, tmp_path: Path) -> None:
        calls: list[tuple[str, float]] = []
        prefetch_geo_file(
            "GSE1", tmp_path, timeout=7, opener=_opener(gzip.compress(SOFT_TEXT), calls)
        )
        assert calls[0][1] == 7


class TestPrefetchOrWarn:
    def test_failure_returns_none_and_warns(self, tmp_path: Path, caplog) -> None:
        with caplog.at_level("WARNING", logger=geo_https.__name__):
            assert prefetch_geo_file_or_warn("GSE11909", tmp_path, opener=_failing_opener) is None
        assert "GEOparse will try FTP" in caplog.text

    def test_success_returns_the_path(self, tmp_path: Path) -> None:
        path = prefetch_geo_file_or_warn(
            "GSE11909", tmp_path, opener=_opener(gzip.compress(SOFT_TEXT))
        )
        assert path == tmp_path / "GSE11909_family.soft.gz"


class TestCallersDownloadOverHttpsFirst:
    """Every GEOparse call by accession is preceded by the HTTPS download."""

    def test_hybrid_source_family_soft(self, tmp_path: Path, monkeypatch) -> None:
        from preprocessing import hybrid_source

        order: list[str] = []
        monkeypatch.setattr(
            hybrid_source,
            "prefetch_geo_file_or_warn",
            lambda acc, destdir: order.append(f"https {acc} {destdir}"),
        )
        fake_geoparse = MagicMock()
        fake_geoparse.get_GEO.side_effect = lambda **kw: order.append(f"geoparse {kw['geo']}")
        monkeypatch.setattr(hybrid_source, "GEOparse", fake_geoparse)

        source = hybrid_source.HybridGeoSource(cache_dir=tmp_path, network=MagicMock())
        source._load_geoparse("GSE11909")

        assert order == [f"https GSE11909 {tmp_path}", "geoparse GSE11909"]

    def test_geoparse_source(self, tmp_path: Path, monkeypatch) -> None:
        from preprocessing import geo_sources

        order: list[str] = []
        monkeypatch.setattr(
            geo_sources,
            "prefetch_geo_file_or_warn",
            lambda acc, destdir: order.append(f"https {acc}"),
        )
        fake_geoparse = MagicMock()
        fake_geoparse.get_GEO.side_effect = lambda **kw: order.append(f"geoparse {kw['geo']}")
        monkeypatch.setattr(geo_sources, "GEOparse", fake_geoparse)

        source = geo_sources.GEOparseSource.__new__(geo_sources.GEOparseSource)
        source.cache_dir = tmp_path
        source.silent = True
        source._fetch_with_geoparse("GSE11909", use_soft=True)

        assert order == ["https GSE11909", "geoparse GSE11909"]

    def test_standalone_platform_table(self, tmp_path: Path, monkeypatch) -> None:
        from preprocessing import hybrid_handler

        order: list[str] = []
        monkeypatch.setattr(
            hybrid_handler,
            "prefetch_geo_file_or_warn",
            lambda acc, destdir: order.append(f"https {acc} {destdir}"),
        )
        fake_geoparse = MagicMock()

        def _get_geo(**kw):
            order.append(f"geoparse {kw['geo']} {kw['destdir']}")
            return MagicMock(table=None)

        fake_geoparse.get_GEO.side_effect = _get_geo
        monkeypatch.setattr(hybrid_handler, "GEOparse", fake_geoparse)

        handler = hybrid_handler.HybridMicroarrayHandler.__new__(
            hybrid_handler.HybridMicroarrayHandler
        )
        handler._annotator = MagicMock()
        handler._annotator.annotate.return_value = ["A1", "NA"]
        dataset = MagicMock(metadata={"cache_dir": str(tmp_path)})

        handler._try_standalone_gpl_fallback(dataset, "GSE1", "GPL96", MagicMock(), None)

        assert order == [f"https GPL96 {tmp_path}", f"geoparse GPL96 {tmp_path}"]
