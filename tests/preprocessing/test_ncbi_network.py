"""
Unit tests for NCBINetwork.

These tests verify network operations without live connections by mocking
ftplib, urllib, and the filesystem.
"""

from __future__ import annotations

import gzip
import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# ============================================================================
# Helpers
# ============================================================================


def _make_gz_bytes(content: str = "hello world") -> bytes:
    """Create valid gzip bytes for testing."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(content.encode("utf-8"))
    return buf.getvalue()


# ============================================================================
# geo_series_ftp_path (pure, no mocks needed)
# ============================================================================


class TestGeoSeriesFtpPath:
    """Tests for the static path builder."""

    def test_standard_gse(self):
        """A 5-digit GSE ID buckets into its GSE11nnn directory."""
        from preprocessing.ncbi_network import NCBINetwork

        path = NCBINetwork.geo_series_ftp_path("GSE11909", "matrix")
        assert path == "geo/series/GSE11nnn/GSE11909/matrix"

    def test_five_digit_gse(self):
        """A different 5-digit GSE ID buckets correctly with a "soft" subpath."""
        from preprocessing.ncbi_network import NCBINetwork

        path = NCBINetwork.geo_series_ftp_path("GSE50635", "soft")
        assert path == "geo/series/GSE50nnn/GSE50635/soft"

    def test_six_digit_gse(self):
        """A 6-digit GSE ID buckets into its GSE123nnn directory."""
        from preprocessing.ncbi_network import NCBINetwork

        path = NCBINetwork.geo_series_ftp_path("GSE123456", "matrix")
        assert path == "geo/series/GSE123nnn/GSE123456/matrix"

    def test_default_subpath(self):
        """Omitting the subpath argument defaults to "matrix"."""
        from preprocessing.ncbi_network import NCBINetwork

        path = NCBINetwork.geo_series_ftp_path("GSE11909")
        assert path.endswith("/matrix")

    def test_suppl_subpath(self):
        """The "suppl" subpath is passed through unchanged."""
        from preprocessing.ncbi_network import NCBINetwork

        path = NCBINetwork.geo_series_ftp_path("GSE11909", "suppl")
        assert path.endswith("/suppl")


# ============================================================================
# list_ftp_directory
# ============================================================================


class TestListFtpDirectory:
    """Tests for FTP directory listing."""

    @patch("preprocessing.ncbi_network.ftplib.FTP")
    def test_returns_filtered_files(self, MockFTP):
        """Only filenames matching the given suffix are returned."""
        from preprocessing.ncbi_network import NCBINetwork

        mock_ftp = MagicMock()
        MockFTP.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        MockFTP.return_value.__exit__ = MagicMock(return_value=False)
        mock_ftp.nlst.return_value = [
            "GSE11909-GPL96_series_matrix.txt.gz",
            "GSE11909-GPL97_series_matrix.txt.gz",
            "readme.txt",
        ]

        net = NCBINetwork(cache_dir="/tmp/test_cache")
        files = net.list_ftp_directory(
            "geo/series/GSE11nnn/GSE11909/matrix",
            suffix="_series_matrix.txt.gz",
        )

        assert len(files) == 2
        assert "readme.txt" not in files

    @patch("preprocessing.ncbi_network.ftplib.FTP")
    def test_returns_all_files_without_suffix(self, MockFTP):
        """With no suffix filter, every listed file is returned."""
        from preprocessing.ncbi_network import NCBINetwork

        mock_ftp = MagicMock()
        MockFTP.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        MockFTP.return_value.__exit__ = MagicMock(return_value=False)
        mock_ftp.nlst.return_value = ["a.txt", "b.gz"]

        net = NCBINetwork(cache_dir="/tmp/test_cache")
        files = net.list_ftp_directory("some/path")
        assert len(files) == 2

    @patch("preprocessing.ncbi_network.ftplib.FTP")
    def test_retries_on_failure(self, MockFTP):
        """A transient FTP failure is retried and succeeds on the next attempt."""
        from preprocessing.ncbi_network import NCBINetwork

        mock_ftp = MagicMock()
        MockFTP.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        MockFTP.return_value.__exit__ = MagicMock(return_value=False)
        mock_ftp.nlst.side_effect = [
            Exception("timeout"),
            ["file.gz"],
        ]

        net = NCBINetwork(
            cache_dir="/tmp/test_cache",
            max_retries=2,
            retry_backoff=0.01,
        )
        files = net.list_ftp_directory("some/path")
        assert files == ["file.gz"]

    @patch("preprocessing.ncbi_network.ftplib.FTP")
    def test_raises_after_max_retries(self, MockFTP):
        """A persistent FTP failure raises ConnectionError once retries are exhausted."""
        from preprocessing.ncbi_network import NCBINetwork

        mock_ftp = MagicMock()
        MockFTP.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        MockFTP.return_value.__exit__ = MagicMock(return_value=False)
        mock_ftp.nlst.side_effect = Exception("permanent failure")

        net = NCBINetwork(
            cache_dir="/tmp/test_cache",
            max_retries=2,
            retry_backoff=0.01,
        )
        with pytest.raises(ConnectionError, match="permanent failure"):
            net.list_ftp_directory("some/path")


# ============================================================================
# download_file (HTTPS with disk caching)
# ============================================================================


class TestDownloadFile:
    """Tests for HTTPS file download with caching."""

    def test_returns_cached_file(self, tmp_path):
        """Cached file is returned immediately without network call."""
        from preprocessing.ncbi_network import NCBINetwork

        # Pre-create cached file
        cached = tmp_path / "test.txt.gz"
        cached.write_text("cached content")

        net = NCBINetwork(cache_dir=tmp_path)
        result = net.download_file("some/path", "test.txt.gz")

        assert result == cached

    @patch("preprocessing.ncbi_network.urllib.request.urlretrieve")
    def test_downloads_when_not_cached(self, mock_retrieve, tmp_path):
        """File is downloaded when not in cache."""
        from preprocessing.ncbi_network import NCBINetwork

        # urlretrieve creates the file
        def fake_retrieve(url, path):
            Path(path).write_text("downloaded")

        mock_retrieve.side_effect = fake_retrieve

        net = NCBINetwork(cache_dir=tmp_path)
        result = net.download_file("some/path", "new_file.txt.gz")

        assert result == tmp_path / "new_file.txt.gz"
        assert result.exists()
        mock_retrieve.assert_called_once()

    @patch("preprocessing.ncbi_network.urllib.request.urlretrieve")
    def test_retries_on_failure(self, mock_retrieve, tmp_path):
        """Retries download on transient failures."""
        from preprocessing.ncbi_network import NCBINetwork

        call_count = 0

        def side_effect_fn(url, path):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("transient")
            Path(path).write_text("ok")

        mock_retrieve.side_effect = side_effect_fn

        net = NCBINetwork(
            cache_dir=tmp_path,
            max_retries=2,
            retry_backoff=0.01,
        )
        result = net.download_file("p", "f.gz")
        assert result.exists()

    @patch("preprocessing.ncbi_network.urllib.request.urlretrieve")
    def test_raises_after_max_retries(self, mock_retrieve, tmp_path):
        """Raises RuntimeError after all retries exhausted."""
        from preprocessing.ncbi_network import NCBINetwork

        mock_retrieve.side_effect = Exception("permanent")

        net = NCBINetwork(
            cache_dir=tmp_path,
            max_retries=2,
            retry_backoff=0.01,
        )
        with pytest.raises(RuntimeError, match="permanent"):
            net.download_file("p", "f.gz")


# ============================================================================
# download_bytes (HTTPS + FTP fallback)
# ============================================================================


class TestDownloadBytes:
    """Tests for bytes download with HTTPS/FTP fallback."""

    @patch("preprocessing.ncbi_network.urllib.request.urlopen")
    def test_https_success(self, mock_urlopen, tmp_path):
        """Returns bytes from HTTPS on success."""
        from preprocessing.ncbi_network import NCBINetwork

        gz_data = _make_gz_bytes("test data")
        mock_resp = MagicMock()
        mock_resp.read.return_value = gz_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        net = NCBINetwork(cache_dir=tmp_path)
        result = net.download_bytes("p", "f.gz")

        assert result == gz_data

    @patch("preprocessing.ncbi_network.ftplib.FTP")
    @patch("preprocessing.ncbi_network.urllib.request.urlopen")
    def test_falls_back_to_ftp(self, mock_urlopen, MockFTP, tmp_path):
        """Falls back to FTP when HTTPS fails."""
        from preprocessing.ncbi_network import NCBINetwork

        # HTTPS fails
        mock_urlopen.side_effect = Exception("HTTPS down")

        # FTP succeeds
        gz_data = _make_gz_bytes("ftp data")
        mock_ftp = MagicMock()
        MockFTP.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        MockFTP.return_value.__exit__ = MagicMock(return_value=False)

        def fake_retrbinary(cmd, callback, blocksize=8192):
            callback(gz_data)

        mock_ftp.retrbinary.side_effect = fake_retrbinary

        net = NCBINetwork(cache_dir=tmp_path, max_retries=1, retry_backoff=0.01)
        result = net.download_bytes("p", "f.gz")

        assert result == gz_data

    @patch("preprocessing.ncbi_network.ftplib.FTP")
    @patch("preprocessing.ncbi_network.urllib.request.urlopen")
    def test_raises_when_both_fail(self, mock_urlopen, MockFTP, tmp_path):
        """Raises RuntimeError when both HTTPS and FTP fail."""
        from preprocessing.ncbi_network import NCBINetwork

        mock_urlopen.side_effect = Exception("HTTPS down")

        mock_ftp = MagicMock()
        MockFTP.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        MockFTP.return_value.__exit__ = MagicMock(return_value=False)
        mock_ftp.retrbinary.side_effect = Exception("FTP down")

        net = NCBINetwork(cache_dir=tmp_path, max_retries=1, retry_backoff=0.01)

        with pytest.raises(RuntimeError, match="both HTTPS and FTP"):
            net.download_bytes("p", "f.gz")

    @patch("preprocessing.ncbi_network.urllib.request.urlopen")
    def test_skips_gzip_validation_for_non_gz(self, mock_urlopen, tmp_path):
        """Non-gz files skip gzip validation."""
        from preprocessing.ncbi_network import NCBINetwork

        raw_data = b"plain text content"
        mock_resp = MagicMock()
        mock_resp.read.return_value = raw_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        net = NCBINetwork(cache_dir=tmp_path)
        result = net.download_bytes("p", "f.txt")  # not .gz

        assert result == raw_data


# ============================================================================
# _validate_gzip
# ============================================================================


class TestValidateGzip:
    """Tests for gzip validation helper."""

    def test_valid_gzip(self):
        """Well-formed gzip bytes pass validation without raising."""
        from preprocessing.ncbi_network import NCBINetwork

        gz = _make_gz_bytes("good data")
        # Should not raise
        NCBINetwork._validate_gzip(gz, "test")

    def test_invalid_gzip(self):
        """Non-gzip bytes raise ValueError."""
        from preprocessing.ncbi_network import NCBINetwork

        with pytest.raises(ValueError, match="Invalid gzip"):
            NCBINetwork._validate_gzip(b"not gzip data", "test")

    def test_empty_gzip(self):
        """A gzip stream wrapping zero bytes of content raises ValueError."""
        from preprocessing.ncbi_network import NCBINetwork

        # Create gzip with empty content
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            pass  # write nothing
        empty_gz = buf.getvalue()

        with pytest.raises(ValueError, match="empty"):
            NCBINetwork._validate_gzip(empty_gz, "test")


# ============================================================================
# Integration: RParityGeoSource uses NCBINetwork
# ============================================================================


class TestSourceNetworkInjection:
    """Verify HybridGeoSource delegates to injected NCBINetwork."""

    @patch("preprocessing.hybrid_source.GEOparse")
    def test_source_uses_injected_network(self, mock_geoparse, tmp_path):
        """Source delegates FTP listing and download to the network layer."""
        from preprocessing.hybrid_source import HybridGeoSource
        from preprocessing.ncbi_network import NCBINetwork

        # Create mock network
        mock_net = MagicMock(spec=NCBINetwork)
        mock_net.list_ftp_directory.return_value = [
            "GSE99999_series_matrix.txt.gz",
        ]

        # Create a minimal gzipped Series Matrix file for download
        lines = [
            '!Sample_platform_id\t"GPL96"',
            '"ID_REF"\t"GSM100"\t"GSM101"',
            '"1000_at"\t"8.0"\t"7.0"',
        ]
        gz_path = tmp_path / "GSE99999_series_matrix.txt.gz"
        with gzip.open(str(gz_path), "wt") as f:
            f.write("\n".join(lines))

        mock_net.download_file.return_value = gz_path

        # Create mock GEOparse GSE
        mock_gpl = MagicMock()
        mock_gpl.table = MagicMock()
        mock_gpl.table.columns = MagicMock()
        mock_gpl.table.columns.tolist.return_value = ["ID", "Gene Symbol"]

        mock_gse = MagicMock()
        mock_gse.gpls = {"GPL96": mock_gpl}
        mock_gse.gsms = {
            "GSM100": MagicMock(
                metadata={
                    "title": ["T1"],
                    "source_name_ch1": ["s"],
                    "organism_ch1": ["Hs"],
                    "characteristics_ch1": [],
                    "description": ["d"],
                    "platform_id": ["GPL96"],
                }
            ),
            "GSM101": MagicMock(
                metadata={
                    "title": ["T2"],
                    "source_name_ch1": ["s"],
                    "organism_ch1": ["Hs"],
                    "characteristics_ch1": [],
                    "description": ["d"],
                    "platform_id": ["GPL96"],
                }
            ),
        }

        source = HybridGeoSource.__new__(HybridGeoSource)
        source.cache_dir = tmp_path
        source._network = mock_net
        source._load_geoparse = MagicMock(return_value=mock_gse)

        raw_datasets = source.fetch("GSE99999")

        # Verify network was used
        mock_net.list_ftp_directory.assert_called_once()
        mock_net.download_file.assert_called_once()
        assert len(raw_datasets) == 1

    @patch("preprocessing.hybrid_source.GEOparse")
    def test_source_creates_default_network(self, mock_geoparse, tmp_path):
        """Source creates a default NCBINetwork when none is injected."""
        from preprocessing.hybrid_source import HybridGeoSource
        from preprocessing.ncbi_network import NCBINetwork

        source = HybridGeoSource(cache_dir=tmp_path)
        assert isinstance(source._network, NCBINetwork)
        assert source._network.cache_dir == tmp_path
