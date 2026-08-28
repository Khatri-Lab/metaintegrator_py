"""
Comprehensive tests for NCBIFetcher (the 'fast' strategy downloader).

Tests cover:
  - _SimpleCache (disk-based pickle cache)
  - NCBIFetcher construction
  - _parse_series_matrix (pure function, no I/O)
  - _probe_platform_from_filename
  - _size_from_listing
  - fetch_series_matrix (mocked FTP + NCBINetwork)
  - fetch_platform_annotation
  - fetch_minimal_metadata (mocked requests)
  - Edge cases: malformed data, empty inputs, large matrices
"""

from __future__ import annotations

import ftplib
import gzip
import io
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from preprocessing.ncbi_fetcher import NCBIFetcher, _SimpleCache

# ============================================================================
# Helpers
# ============================================================================


def _make_gz_matrix(lines: list[str]) -> bytes:
    """Create gzipped bytes from a list of text lines."""
    text = "\n".join(lines)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


MINIMAL_MATRIX_LINES = [
    '!Series_title\t"Test Series"',
    '!Series_platform_id\t"GPL570"',
    '!Sample_geo_accession\t"GSM100"\t"GSM101"\t"GSM102"',
    "!series_matrix_table_begin",
    '"ID_REF"\t"GSM100"\t"GSM101"\t"GSM102"',
    '"PROBE_001"\t"7.5"\t"7.3"\t"8.2"',
    '"PROBE_002"\t"6.1"\t"6.0"\t"5.8"',
    '"GAPDH"\t"10.5"\t"10.7"\t"10.3"',
    "!series_matrix_table_end",
]


# ============================================================================
# _SimpleCache
# ============================================================================


class TestSimpleCache:
    """Tests for the internal pickle cache."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Provide a fresh _SimpleCache backed by a temp directory."""
        return _SimpleCache(tmp_path)

    def test_get_missing_key_returns_none(self, cache):
        """Looking up a key that was never set returns None."""
        assert cache.get("nonexistent") is None

    def test_set_and_get_roundtrip(self, cache):
        """A value written with set() comes back unchanged from get()."""
        data = {"GPL570": pd.DataFrame({"A": [1, 2, 3]})}
        cache.set("my_key", data)
        loaded = cache.get("my_key")
        assert loaded is not None
        assert "GPL570" in loaded
        pd.testing.assert_frame_equal(loaded["GPL570"], data["GPL570"])

    def test_cache_dir_is_created(self, tmp_path):
        """The cache directory is created automatically, including nested parents."""
        new_dir = tmp_path / "nested" / "cache"
        c = _SimpleCache(new_dir)
        assert c.cache_dir.is_dir()

    def test_corrupted_file_returns_none_and_cleans_up(self, cache):
        """A cache file with unreadable (non-pickle) contents is treated as a miss and removed."""
        key = "corrupt"
        p = cache._path(key)
        p.write_text("this is not pickle data")
        assert cache.get(key) is None
        assert not p.exists()

    def test_special_chars_in_key(self, cache):
        """Cache keys containing hyphens round-trip correctly."""
        cache.set("matrix_GSE-2034", {"a": 1})
        assert cache.get("matrix_GSE-2034") == {"a": 1}

    def test_overwrite_existing_key(self, cache):
        """Setting a key a second time replaces the previous value."""
        cache.set("k", 1)
        cache.set("k", 2)
        assert cache.get("k") == 2


# ============================================================================
# NCBIFetcher construction
# ============================================================================


class TestNCBIFetcherInit:
    """Tests for NCBIFetcher construction."""

    def test_creates_with_string_path(self, tmp_path):
        """cache_dir accepts a plain string path."""
        f = NCBIFetcher(cache_dir=str(tmp_path / "cache"))
        assert f.cache.cache_dir.is_dir()

    def test_creates_with_path_object(self, tmp_path):
        """cache_dir also accepts a pathlib.Path."""
        f = NCBIFetcher(cache_dir=tmp_path / "cache")
        assert f.cache.cache_dir.is_dir()

    def test_has_network_layer(self, tmp_path):
        """Construction wires up an internal network client."""
        f = NCBIFetcher(cache_dir=tmp_path)
        assert f._network is not None

    def test_class_constants(self, tmp_path):
        """FTP host and max download size are set to their expected defaults."""
        f = NCBIFetcher(cache_dir=tmp_path)
        assert f.FTP_HOST == "ftp.ncbi.nlm.nih.gov"
        assert f.MAX_DOWNLOAD_SIZE == 100 * 1024 * 1024


# ============================================================================
# _parse_series_matrix (pure, no I/O)
# ============================================================================


class TestParseSeriesMatrix:
    """Tests for _parse_series_matrix(), the pure gzipped-matrix parser."""

    def test_basic_parse(self):
        """A well-formed matrix parses into the expected shape, index, and values."""
        gz = _make_gz_matrix(MINIMAL_MATRIX_LINES)
        df, platforms = NCBIFetcher._parse_series_matrix(gz)

        assert platforms == ["GPL570"]
        assert df.shape == (3, 3)
        assert list(df.columns) == ["GSM100", "GSM101", "GSM102"]
        assert list(df.index) == ["PROBE_001", "PROBE_002", "GAPDH"]
        assert df.loc["PROBE_001", "GSM100"] == pytest.approx(7.5)

    def test_numeric_conversion(self):
        """Every value column is converted to float64."""
        gz = _make_gz_matrix(MINIMAL_MATRIX_LINES)
        df, _ = NCBIFetcher._parse_series_matrix(gz)
        for col in df.columns:
            assert df[col].dtype == np.float64

    def test_na_and_null_handling(self):
        """The literal tokens "NA" and "null" both parse as missing values."""
        lines = [
            '!Series_platform_id\t"GPL96"',
            '"ID_REF"\t"S1"\t"S2"',
            '"P1"\t"7.5"\t"null"',
            '"P2"\t"NA"\t"6.0"',
        ]
        gz = _make_gz_matrix(lines)
        df, platforms = NCBIFetcher._parse_series_matrix(gz)

        assert platforms == ["GPL96"]
        assert pd.isna(df.loc["P1", "S2"])
        assert pd.isna(df.loc["P2", "S1"])
        assert df.loc["P2", "S2"] == pytest.approx(6.0)

    def test_scientific_notation(self):
        """Values in scientific notation (lower- and upper-case exponent) parse correctly."""
        lines = [
            '!Series_platform_id\t"GPL96"',
            '"ID_REF"\t"S1"\t"S2"',
            '"P1"\t"1.5e-5"\t"2.3e-4"',
            '"P2"\t"5.6E+3"\t"7.8E+2"',
        ]
        gz = _make_gz_matrix(lines)
        df, _ = NCBIFetcher._parse_series_matrix(gz)

        assert df.loc["P1", "S1"] == pytest.approx(1.5e-5)
        assert df.loc["P2", "S1"] == pytest.approx(5.6e3)

    def test_no_platform_id(self):
        """A matrix with no !Series_platform_id line returns an empty platforms list."""
        lines = [
            '!Series_title\t"No Platform"',
            '"ID_REF"\t"S1"',
            '"P1"\t"5.0"',
        ]
        gz = _make_gz_matrix(lines)
        df, platforms = NCBIFetcher._parse_series_matrix(gz)

        assert platforms == []
        assert df.shape == (1, 1)

    def test_multiple_platform_ids(self):
        """Multiple !Series_platform_id lines are all collected, in order."""
        lines = [
            '!Series_platform_id\t"GPL570"',
            '!Series_platform_id\t"GPL96"',
            '"ID_REF"\t"S1"',
            '"P1"\t"5.0"',
        ]
        gz = _make_gz_matrix(lines)
        _, platforms = NCBIFetcher._parse_series_matrix(gz)

        assert platforms == ["GPL570", "GPL96"]

    def test_empty_data_raises(self):
        """A matrix with no data lines raises ValueError."""
        lines = ['!Series_title\t"Empty"']
        gz = _make_gz_matrix(lines)
        with pytest.raises(ValueError, match="no data lines"):
            NCBIFetcher._parse_series_matrix(gz)

    def test_quotes_stripped(self):
        """Double-quote wrapping is stripped from both index and column labels."""
        lines = [
            '"ID_REF"\t"GSM100"',
            '"PROBE_1"\t"5.5"',
        ]
        gz = _make_gz_matrix(lines)
        df, _ = NCBIFetcher._parse_series_matrix(gz)

        assert df.index[0] == "PROBE_1"
        assert df.columns[0] == "GSM100"

    def test_affymetrix_probe_suffixes(self):
        """Affymetrix-style probe IDs with suffixes (e.g. _s_at) parse without truncation."""
        lines = [
            '!Series_platform_id\t"GPL96"',
            '"ID_REF"\t"S1"\t"S2"',
            '"1007_s_at"\t"5.0"\t"6.0"',
            '"1053_at"\t"7.0"\t"8.0"',
            '"121_at"\t"9.0"\t"10.0"',
        ]
        gz = _make_gz_matrix(lines)
        df, _ = NCBIFetcher._parse_series_matrix(gz)

        assert "1007_s_at" in df.index
        assert "1053_at" in df.index

    def test_illumina_probe_ids(self):
        """Illumina-style ILMN_ probe IDs parse correctly."""
        lines = [
            '!Series_platform_id\t"GPL6947"',
            '"ID_REF"\t"S1"',
            '"ILMN_1234567"\t"5.0"',
            '"ILMN_7654321"\t"7.0"',
        ]
        gz = _make_gz_matrix(lines)
        df, _ = NCBIFetcher._parse_series_matrix(gz)

        assert "ILMN_1234567" in df.index


# ============================================================================
# _probe_platform_from_filename
# ============================================================================


class TestProbePlatformFromFilename:
    """Tests for _probe_platform_from_filename()'s GPL-ID extraction."""

    def test_standard_format(self):
        """A standard GSE-GPL series matrix filename yields its platform ID."""
        assert (
            NCBIFetcher._probe_platform_from_filename("GSE11909-GPL96_series_matrix.txt.gz")
            == "GPL96"
        )

    def test_no_gpl(self):
        """A filename without a GPL segment returns "unknown"."""
        assert (
            NCBIFetcher._probe_platform_from_filename("GSE11909_series_matrix.txt.gz") == "unknown"
        )

    def test_multi_digit_gpl(self):
        """A multi-digit GPL ID is extracted in full."""
        assert (
            NCBIFetcher._probe_platform_from_filename("GSE50635-GPL6947_series_matrix.txt.gz")
            == "GPL6947"
        )


# ============================================================================
# _size_from_listing
# ============================================================================


class TestSizeFromListing:
    """Tests for _size_from_listing()'s FTP file-size lookup."""

    def test_mlsd_success(self, tmp_path):
        """The file size is read from the MLSD listing when the server supports it."""
        fetcher = NCBIFetcher(cache_dir=tmp_path)
        mock_ftp = MagicMock()
        mock_ftp.mlsd.return_value = [
            ("file.gz", {"size": "12345"}),
            ("other.gz", {"size": "99999"}),
        ]

        assert fetcher._size_from_listing(mock_ftp, "file.gz") == 12345

    def test_fallback_to_list(self, tmp_path):
        """When MLSD is unsupported, the size falls back to parsing a LIST-style listing."""
        fetcher = NCBIFetcher(cache_dir=tmp_path)
        mock_ftp = MagicMock()
        mock_ftp.mlsd.side_effect = Exception("MLSD not supported")

        listing = "-rw-r--r-- 1 owner group 12345 Jan 01 2021 file.gz"

        def fake_retrlines(cmd, callback):
            callback(listing)

        mock_ftp.retrlines.side_effect = fake_retrlines

        assert fetcher._size_from_listing(mock_ftp, "file.gz") == 12345

    def test_returns_zero_on_failure(self, tmp_path):
        """If both MLSD and LIST fail, the size lookup returns 0 rather than raising."""
        fetcher = NCBIFetcher(cache_dir=tmp_path)
        mock_ftp = MagicMock()
        mock_ftp.mlsd.side_effect = Exception("fail")
        mock_ftp.retrlines.side_effect = Exception("fail")

        assert fetcher._size_from_listing(mock_ftp, "file.gz") == 0


# ============================================================================
# fetch_series_matrix (mocked network)
# ============================================================================


class TestFetchSeriesMatrix:
    """Tests for fetch_series_matrix() against a mocked FTP/network layer."""

    @pytest.fixture
    def fetcher(self, tmp_path):
        """Provide an NCBIFetcher backed by a temp cache directory."""
        return NCBIFetcher(cache_dir=tmp_path)

    def test_returns_cached_result(self, fetcher):
        """A cached matrix is returned directly, without touching FTP."""
        cached_data = {"GPL570": pd.DataFrame({"S1": [1.0]})}
        fetcher.cache.set("matrix_GSE2034", cached_data)

        result = fetcher.fetch_series_matrix("GSE2034")
        assert "GPL570" in result

    @patch("preprocessing.ncbi_fetcher.ftplib.FTP")
    def test_full_fetch_workflow(self, MockFTP, fetcher):
        """Mock FTP listing + NCBINetwork download + parse."""
        mock_ftp = MagicMock()
        MockFTP.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        MockFTP.return_value.__exit__ = MagicMock(return_value=False)

        mock_ftp.nlst.return_value = ["GSE2034_series_matrix.txt.gz"]
        mock_ftp.size.return_value = 1000

        gz = _make_gz_matrix(MINIMAL_MATRIX_LINES)
        fetcher._network = MagicMock()
        fetcher._network.download_bytes.return_value = gz

        result = fetcher.fetch_series_matrix("GSE2034")

        assert "GPL570" in result
        assert result["GPL570"].shape == (3, 3)
        MockFTP.assert_called()

    @patch("preprocessing.ncbi_fetcher.ftplib.FTP")
    def test_no_matrix_files_raises(self, MockFTP, fetcher):
        """When the FTP directory has no series-matrix files, ValueError is raised."""
        mock_ftp = MagicMock()
        MockFTP.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        MockFTP.return_value.__exit__ = MagicMock(return_value=False)
        mock_ftp.nlst.return_value = ["readme.txt"]

        with pytest.raises(ValueError, match="No matrix files found"):
            fetcher.fetch_series_matrix("GSE2034")

    @patch("preprocessing.ncbi_fetcher.ftplib.FTP")
    def test_skips_oversized_files(self, MockFTP, fetcher):
        """A file larger than MAX_DOWNLOAD_SIZE is rejected rather than downloaded."""
        mock_ftp = MagicMock()
        MockFTP.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        MockFTP.return_value.__exit__ = MagicMock(return_value=False)

        mock_ftp.nlst.return_value = ["GSE2034_series_matrix.txt.gz"]
        mock_ftp.size.return_value = fetcher.MAX_DOWNLOAD_SIZE + 1

        with pytest.raises(ValueError, match="too large"):
            fetcher.fetch_series_matrix("GSE2034")

    @patch("preprocessing.ncbi_fetcher.ftplib.FTP")
    def test_platform_from_filename_fallback(self, MockFTP, fetcher):
        """When no platform in metadata, extract from filename."""
        mock_ftp = MagicMock()
        MockFTP.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        MockFTP.return_value.__exit__ = MagicMock(return_value=False)

        mock_ftp.nlst.return_value = ["GSE2034-GPL96_series_matrix.txt.gz"]
        mock_ftp.size.return_value = 500

        # Matrix with no platform metadata
        lines = ['"ID_REF"\t"S1"', '"P1"\t"5.0"']
        gz = _make_gz_matrix(lines)
        fetcher._network = MagicMock()
        fetcher._network.download_bytes.return_value = gz

        result = fetcher.fetch_series_matrix("GSE2034")
        assert "GPL96" in result

    @patch("preprocessing.ncbi_fetcher.ftplib.FTP")
    def test_result_is_cached(self, MockFTP, fetcher):
        """After a successful fetch, the result is written to the cache for next time."""
        mock_ftp = MagicMock()
        MockFTP.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        MockFTP.return_value.__exit__ = MagicMock(return_value=False)

        mock_ftp.nlst.return_value = ["GSE2034_series_matrix.txt.gz"]
        mock_ftp.size.return_value = 500

        gz = _make_gz_matrix(MINIMAL_MATRIX_LINES)
        fetcher._network = MagicMock()
        fetcher._network.download_bytes.return_value = gz

        fetcher.fetch_series_matrix("GSE2034")

        # Second call should use cache
        cached = fetcher.cache.get("matrix_GSE2034")
        assert cached is not None
        assert "GPL570" in cached


# ============================================================================
# fetch_platform_annotation
# ============================================================================


class TestFetchPlatformAnnotation:
    """Tests for fetch_platform_annotation()."""

    def test_returns_empty_dataframe(self, tmp_path):
        """Platform annotation fetching is not yet implemented; it returns an empty DataFrame."""
        f = NCBIFetcher(cache_dir=tmp_path)
        result = f.fetch_platform_annotation("GPL570")
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ============================================================================
# fetch_minimal_metadata (mocked requests)
# ============================================================================


class TestFetchMinimalMetadata:
    """Tests for fetch_minimal_metadata() against a mocked MINiML response."""

    @patch("requests.get")
    def test_success(self, mock_get, tmp_path):
        """Title, summary, and platform IDs are extracted from a MINiML XML response."""
        fetcher = NCBIFetcher(cache_dir=tmp_path)

        mock_response = MagicMock()
        mock_response.text = """
        <MINiML>
            <title>My GEO Series</title>
            <summary>A test dataset</summary>
            <platformId>GPL570</platformId>
            <platformId>GPL96</platformId>
        </MINiML>
        """
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetcher.fetch_minimal_metadata("GSE2034")

        assert result["title"] == "My GEO Series"
        assert result["summary"] == "A test dataset"
        assert result["platforms"] == ["GPL570", "GPL96"]

    def test_cached_result(self, tmp_path):
        """A cached metadata result is returned directly, without an HTTP request."""
        fetcher = NCBIFetcher(cache_dir=tmp_path)
        cached = {"title": "Cached", "summary": "", "platforms": ["GPL570"]}
        fetcher.cache.set("metadata_GSE2034", cached)

        result = fetcher.fetch_minimal_metadata("GSE2034")
        assert result["title"] == "Cached"


# ============================================================================
# _extract_tag / _extract_platforms
# ============================================================================


class TestXmlHelpers:
    """Tests for the internal _extract_tag/_extract_platforms XML helpers."""

    def test_extract_tag_found(self):
        """A present tag's text content is returned."""
        xml = "<title>My Title</title>"
        assert NCBIFetcher._extract_tag(xml, "title") == "My Title"

    def test_extract_tag_not_found(self):
        """A missing tag returns an empty string rather than raising."""
        assert NCBIFetcher._extract_tag("<root/>", "missing") == ""

    def test_extract_tag_multiline(self):
        """Tag content spanning multiple lines is captured in full."""
        xml = "<summary>\nLine 1\nLine 2\n</summary>"
        result = NCBIFetcher._extract_tag(xml, "summary")
        assert "Line 1" in result

    def test_extract_platforms(self):
        """All <platformId> tags are collected, in document order."""
        xml = "<platformId>GPL570</platformId><platformId>GPL96</platformId>"
        assert NCBIFetcher._extract_platforms(xml) == ["GPL570", "GPL96"]

    def test_extract_platforms_none(self):
        """XML with no <platformId> tags returns an empty list."""
        assert NCBIFetcher._extract_platforms("<root/>") == []


# ============================================================================
# Performance / large matrix
# ============================================================================


class TestPerformance:
    """Performance tests for _parse_series_matrix() on large matrices."""

    @pytest.mark.slow
    def test_large_matrix_parse(self):
        """50k probes × 100 samples should parse in < 30s."""
        import time

        n_probes = 50000
        n_samples = 100

        lines = [f'!Series_platform_id\t"GPL570"']
        header = '"ID_REF"\t' + "\t".join([f'"GSM{i}"' for i in range(n_samples)])
        lines.append(header)
        for i in range(n_probes):
            row = f'"PROBE_{i}"\t' + "\t".join(
                [f'"{np.random.randn():.6f}"' for _ in range(n_samples)]
            )
            lines.append(row)

        gz = _make_gz_matrix(lines)

        start = time.time()
        df, platforms = NCBIFetcher._parse_series_matrix(gz)
        elapsed = time.time() - start

        assert elapsed < 30.0
        assert df.shape == (n_probes, n_samples)
        assert platforms == ["GPL570"]


# ============================================================================
# Data integrity
# ============================================================================


class TestDataIntegrity:
    """Tests that _parse_series_matrix() preserves numeric precision and column order."""

    def test_numeric_precision(self):
        """High-precision decimal values survive parsing without rounding loss."""
        lines = [
            '"ID_REF"\t"S1"\t"S2"',
            '"P1"\t"1.23456789012345"\t"2.34567890123456"',
        ]
        gz = _make_gz_matrix(lines)
        df, _ = NCBIFetcher._parse_series_matrix(gz)

        assert abs(df.iloc[0, 0] - 1.23456789012345) < 1e-10

    def test_sample_order_preserved(self):
        """Column order in the parsed DataFrame matches the header's sample order."""
        sample_ids = [f"GSM{i}" for i in range(20)]
        lines = [
            '"ID_REF"\t' + "\t".join([f'"{s}"' for s in sample_ids]),
            '"P1"\t' + "\t".join(['"5.0"'] * 20),
        ]
        gz = _make_gz_matrix(lines)
        df, _ = NCBIFetcher._parse_series_matrix(gz)

        assert list(df.columns) == sample_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
