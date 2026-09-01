"""
Additional tests for NCBIFetcher: corner cases, edge conditions, and
integration scenarios with NCBIGeoSource.

Supplements test_ncbi_fetcher.py with:
  - Multi-platform matrix handling
  - _SimpleCache stress tests
  - NCBIGeoSource integration (the 'fast' strategy)
  - Encoding and unicode edge cases
  - Inf / NaN data handling in parsed matrices
"""

from __future__ import annotations

import gzip
import io
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from preprocessing.ncbi_fetcher import NCBIFetcher, _SimpleCache

# ============================================================================
# Helpers
# ============================================================================


def _make_gz(lines: list[str]) -> bytes:
    """Create gzipped bytes from a list of text lines."""
    text = "\n".join(lines)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


# ============================================================================
# _SimpleCache: additional edge cases
# ============================================================================


class TestSimpleCacheAdditional:
    """Additional stress and edge-case tests for _SimpleCache."""

    def test_key_collision_prevention(self, tmp_path):
        """Similar keys (e.g. "GSE123" vs "GSE1234") don't collide in the cache."""
        c = _SimpleCache(tmp_path)
        c.set("GSE123", {"shape": (10, 3)})
        c.set("GSE1234", {"shape": (20, 5)})

        assert c.get("GSE123")["shape"] != c.get("GSE1234")["shape"]

    def test_large_value_roundtrip(self, tmp_path):
        """A large DataFrame (50k x 200) round-trips through the cache intact."""
        c = _SimpleCache(tmp_path)
        big = pd.DataFrame(np.random.randn(50000, 200))
        c.set("big", big)
        loaded = c.get("big")
        assert loaded is not None
        assert loaded.shape == (50000, 200)

    def test_rapid_set_get_cycles(self, tmp_path):
        """Many rapid set/get cycles on distinct keys all succeed."""
        c = _SimpleCache(tmp_path)
        data = {"x": 1}
        for i in range(50):
            c.set(f"k_{i}", data)
            assert c.get(f"k_{i}") is not None

    def test_none_value(self, tmp_path):
        """Explicitly caching None as a value still returns None from get() (as it would for a miss)."""
        c = _SimpleCache(tmp_path)
        c.set("k", None)
        # pickle stores None fine, but get() should return None (which is the value)
        assert c.get("k") is None

    def test_list_value(self, tmp_path):
        """Non-DataFrame values (e.g. a plain list) round-trip through the cache too."""
        c = _SimpleCache(tmp_path)
        c.set("k", [1, 2, 3])
        assert c.get("k") == [1, 2, 3]


# ============================================================================
# _parse_series_matrix: corner cases
# ============================================================================


class TestParseSeriesMatrixCornerCases:
    """Corner cases for _parse_series_matrix(): inf, empty, and mixed-validity values."""

    def test_inf_values(self):
        """ "Inf" and "-Inf" tokens parse to IEEE infinity."""
        lines = [
            '"ID_REF"\t"S1"\t"S2"',
            '"P1"\t"Inf"\t"-Inf"',
            '"P2"\t"5.0"\t"6.0"',
        ]
        gz = _make_gz(lines)
        df, _ = NCBIFetcher._parse_series_matrix(gz)

        assert np.isinf(df.loc["P1", "S1"])
        assert np.isinf(df.loc["P1", "S2"])
        assert df.loc["P2", "S1"] == pytest.approx(5.0)

    def test_empty_string_values(self):
        """An empty-string value parses as NaN."""
        lines = [
            '"ID_REF"\t"S1"\t"S2"',
            '"P1"\t""\t"6.0"',
        ]
        gz = _make_gz(lines)
        df, _ = NCBIFetcher._parse_series_matrix(gz)

        assert pd.isna(df.loc["P1", "S1"])
        assert df.loc["P1", "S2"] == pytest.approx(6.0)

    def test_single_probe_single_sample(self):
        """A 1x1 matrix (one probe, one sample) parses without error."""
        lines = [
            '!Series_platform_id\t"GPL96"',
            '"ID_REF"\t"GSM1"',
            '"P1"\t"8.5"',
        ]
        gz = _make_gz(lines)
        df, platforms = NCBIFetcher._parse_series_matrix(gz)

        assert df.shape == (1, 1)
        assert platforms == ["GPL96"]

    def test_many_probes_few_samples(self):
        """A wide-and-shallow matrix (1000 probes, 2 samples) parses to the right shape."""
        lines = ['!Series_platform_id\t"GPL96"']
        header = '"ID_REF"\t"S1"\t"S2"'
        lines.append(header)
        for i in range(1000):
            lines.append(f'"P{i}"\t"{np.random.randn():.4f}"\t"{np.random.randn():.4f}"')

        gz = _make_gz(lines)
        df, _ = NCBIFetcher._parse_series_matrix(gz)

        assert df.shape == (1000, 2)

    def test_metadata_lines_skipped(self):
        """Only !-prefixed lines should be skipped, not data."""
        lines = [
            '!Series_title\t"Title"',
            '!Series_summary\t"Summary"',
            '!Series_platform_id\t"GPL96"',
            "!series_matrix_table_begin",
            '"ID_REF"\t"S1"',
            '"P1"\t"5.0"',
            "!series_matrix_table_end",
        ]
        gz = _make_gz(lines)
        df, _ = NCBIFetcher._parse_series_matrix(gz)

        assert df.shape == (1, 1)
        assert df.loc["P1", "S1"] == pytest.approx(5.0)

    def test_agilent_probe_ids(self):
        """Agilent-style A_##_P###### probe IDs parse correctly."""
        lines = [
            '!Series_platform_id\t"GPL4133"',
            '"ID_REF"\t"S1"',
            '"A_23_P12345"\t"5.0"',
            '"A_24_P67890"\t"7.0"',
        ]
        gz = _make_gz(lines)
        df, _ = NCBIFetcher._parse_series_matrix(gz)

        assert "A_23_P12345" in df.index

    def test_mixed_valid_and_invalid_values(self):
        """A row mixing a valid number, unparseable text, and "NA" resolves each independently."""
        lines = [
            '"ID_REF"\t"S1"\t"S2"\t"S3"',
            '"P1"\t"5.0"\t"abc"\t"NA"',
        ]
        gz = _make_gz(lines)
        df, _ = NCBIFetcher._parse_series_matrix(gz)

        assert df.loc["P1", "S1"] == pytest.approx(5.0)
        assert pd.isna(df.loc["P1", "S2"])  # "abc" → NaN
        assert pd.isna(df.loc["P1", "S3"])  # "NA" → NaN


# ============================================================================
# fetch_series_matrix: multi-platform
# ============================================================================


class TestFetchMultiPlatform:
    """Tests for fetch_series_matrix() when a series spans multiple platform files."""

    @patch("preprocessing.ncbi_fetcher.ftplib.FTP")
    def test_two_matrix_files(self, MockFTP, tmp_path):
        """Two GSE-GPL matrix files for one series are fetched and keyed by their own platform."""
        fetcher = NCBIFetcher(cache_dir=tmp_path)

        mock_ftp = MagicMock()
        MockFTP.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        MockFTP.return_value.__exit__ = MagicMock(return_value=False)

        mock_ftp.nlst.return_value = [
            "GSE11909-GPL96_series_matrix.txt.gz",
            "GSE11909-GPL97_series_matrix.txt.gz",
        ]
        mock_ftp.size.return_value = 500

        gz96 = _make_gz(
            [
                '!Series_platform_id\t"GPL96"',
                '"ID_REF"\t"S1"',
                '"P1"\t"5.0"',
            ]
        )
        gz97 = _make_gz(
            [
                '!Series_platform_id\t"GPL97"',
                '"ID_REF"\t"S1"',
                '"Q1"\t"9.0"',
            ]
        )

        fetcher._network = MagicMock()
        fetcher._network.download_bytes.side_effect = [gz96, gz97]

        result = fetcher.fetch_series_matrix("GSE11909")

        assert "GPL96" in result
        assert "GPL97" in result
        assert result["GPL96"].loc["P1", "S1"] == pytest.approx(5.0)
        assert result["GPL97"].loc["Q1", "S1"] == pytest.approx(9.0)


# ============================================================================
# NCBIGeoSource integration (uses NCBIFetcher internally)
# ============================================================================


class TestNCBIGeoSourceIntegration:
    """Tests that NCBIGeoSource (the 'fast' strategy) correctly delegates to NCBIFetcher."""

    @patch("preprocessing.ncbi_fetcher.ftplib.FTP")
    def test_ncbi_geo_source_uses_fetcher(self, MockFTP, tmp_path):
        """NCBIGeoSource('fast' strategy) delegates to NCBIFetcher."""
        from preprocessing.geo_sources import NCBIGeoSource

        mock_ftp = MagicMock()
        MockFTP.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        MockFTP.return_value.__exit__ = MagicMock(return_value=False)

        mock_ftp.nlst.return_value = ["GSE2034_series_matrix.txt.gz"]
        mock_ftp.size.return_value = 500

        gz = _make_gz(
            [
                '!Series_platform_id\t"GPL570"',
                '"ID_REF"\t"S1"\t"S2"',
                '"P1"\t"5.0"\t"6.0"',
            ]
        )

        source = NCBIGeoSource(cache_dir=tmp_path)
        source.fetcher._network = MagicMock()
        source.fetcher._network.download_bytes.return_value = gz

        raw_datasets = source.fetch("GSE2034")

        assert len(raw_datasets) == 1
        assert raw_datasets[0].platform_id == "GPL570"
        assert raw_datasets[0].source_id == "GSE2034"

    def test_ncbi_geo_source_rejects_gsm(self, tmp_path):
        """A GSM (sample) accession is rejected: NCBIGeoSource only fetches series."""
        from preprocessing.geo_sources import NCBIGeoSource

        source = NCBIGeoSource(cache_dir=tmp_path)
        with pytest.raises(ValueError, match="GSM"):
            source.fetch("GSM12345")

    def test_ncbi_geo_source_rejects_gpl(self, tmp_path):
        """A GPL (platform) accession is rejected: NCBIGeoSource only fetches series."""
        from preprocessing.geo_sources import NCBIGeoSource

        source = NCBIGeoSource(cache_dir=tmp_path)
        with pytest.raises(ValueError, match="GPL"):
            source.fetch("GPL570")


# ============================================================================
# Unicode edge cases
# ============================================================================


class TestUnicodeEdgeCases:
    """Tests that unicode text in metadata lines doesn't break data parsing."""

    def test_unicode_in_metadata_not_in_data(self):
        """Unicode in !-prefixed lines doesn't affect data parsing."""
        lines = [
            '!Sample_title\t"Müller cells"\t"β-cells"',
            '!Series_platform_id\t"GPL96"',
            '"ID_REF"\t"S1"\t"S2"',
            '"P1"\t"5.0"\t"6.0"',
        ]
        gz = _make_gz(lines)
        df, platforms = NCBIFetcher._parse_series_matrix(gz)

        assert platforms == ["GPL96"]
        assert df.shape == (1, 2)


# ============================================================================
# _probe_platform_from_filename: more patterns
# ============================================================================


class TestPlatformFromFilenameAdditional:
    """Additional filename patterns for _probe_platform_from_filename()."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("GSE50635-GPL6947_series_matrix.txt.gz", "GPL6947"),
            ("GSE123456-GPL1261_series_matrix.txt.gz", "GPL1261"),
            ("GSE999_series_matrix.txt.gz", "unknown"),
            ("GPL570_annotation.txt", "GPL570"),  # not a matrix, but still finds GPL
        ],
    )
    def test_various_patterns(self, filename, expected):
        """Each (filename, expected GPL ID) pair extracts as expected."""
        assert NCBIFetcher._probe_platform_from_filename(filename) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
