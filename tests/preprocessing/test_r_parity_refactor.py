"""
Unit tests for the hybrid architecture (renamed from r_parity).

Tests:
1. HybridGeoSource.fetch() produces correctly structured RawDatasets
2. HybridMicroarrayHandler.parse() produces ProcessedDatasets
3. Backward-compat aliases (RParityGeoSource, etc.) still work
4. Thin wrapper load_geo_datasets_r_parity() delegates to factory
5. DataLoaderFactory supports 'hybrid' and 'r_parity' strategies
6. load_geo_datasets_processed() defaults to 'hybrid'
"""

from __future__ import annotations

import gzip
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# ============================================================================
# Test fixtures
# ============================================================================


def _make_expr_df(n_probes: int = 10, n_samples: int = 3):
    """Build a small random expression matrix indexed by probe ID."""
    probes = [f"{int(1000 + i)}_at" for i in range(n_probes)]
    samples = [f"GSM{100 + i}" for i in range(n_samples)]
    rng = np.random.RandomState(42)
    data = rng.normal(loc=8.0, scale=1.5, size=(n_probes, n_samples))
    return pd.DataFrame(data, index=probes, columns=samples, dtype="float64")


def _make_gpl_table(probes: List[str], genes: Optional[List[str]] = None):
    """Build a minimal GPL annotation table mapping probes to gene symbols."""
    if genes is None:
        genes = [f"GENE{i}" for i in range(len(probes))]
    return pd.DataFrame({"ID": probes, "Gene Symbol": genes})


def _make_mock_gse(platform_ids, sample_ids):
    """Build a mock GEOparse GSE object with the given platforms and samples."""
    gse = MagicMock()
    gpls = {}
    for pid in platform_ids:
        gpl = MagicMock()
        gpl.table = _make_gpl_table([f"{int(1000 + i)}_at" for i in range(10)])
        gpls[pid] = gpl
    gse.gpls = gpls

    gsms = {}
    for sid in sample_ids:
        gsm = MagicMock()
        gsm.metadata = {
            "title": [f"Title {sid}"],
            "source_name_ch1": ["tissue"],
            "organism_ch1": ["Homo sapiens"],
            "characteristics_ch1": ["age: 45"],
            "description": [f"Desc {sid}"],
            "platform_id": [platform_ids[0]],
        }
        gsms[sid] = gsm
    gse.gsms = gsms
    return gse


# ============================================================================
# HybridGeoSource
# ============================================================================


class TestHybridGeoSource:
    """Tests for HybridGeoSource.fetch() producing RawDataset objects."""

    @patch("preprocessing.hybrid_source.GEOparse")
    def test_source_name(self, _):
        """source_name identifies this as the Hybrid strategy."""
        from preprocessing.hybrid_source import HybridGeoSource

        source = HybridGeoSource.__new__(HybridGeoSource)
        assert "Hybrid" in source.source_name

    @patch("preprocessing.hybrid_source.GEOparse")
    def test_fetch_returns_raw_datasets(self, _, tmp_path):
        """fetch() returns one RawDataset per platform, carrying expr/gpl/gse metadata."""
        from preprocessing.base import RawDataset
        from preprocessing.hybrid_source import HybridGeoSource

        expr_df = _make_expr_df()
        gse = _make_mock_gse(["GPL96"], list(expr_df.columns))

        source = HybridGeoSource.__new__(HybridGeoSource)
        source.cache_dir = tmp_path
        source._download_and_parse_series_matrices = MagicMock(return_value={"GPL96": expr_df})
        source._load_geoparse = MagicMock(return_value=gse)

        rds = source.fetch("GSE11909")
        assert len(rds) == 1
        assert isinstance(rds[0], RawDataset)
        assert rds[0].source_id == "GSE11909"
        assert set(rds[0].raw_data.keys()) == {"expr_df", "gpl", "gse", "platform_id"}

    @patch("preprocessing.hybrid_source.GEOparse")
    def test_fetch_multi_platform(self, _, tmp_path):
        """A series spanning two platforms yields one RawDataset per platform."""
        from preprocessing.hybrid_source import HybridGeoSource

        expr96 = _make_expr_df()
        expr97 = _make_expr_df(n_probes=8)
        gse = _make_mock_gse(["GPL96", "GPL97"], list(expr96.columns))

        source = HybridGeoSource.__new__(HybridGeoSource)
        source.cache_dir = tmp_path
        source._download_and_parse_series_matrices = MagicMock(
            return_value={"GPL96": expr96, "GPL97": expr97}
        )
        source._load_geoparse = MagicMock(return_value=gse)

        rds = source.fetch("GSE11909")
        assert len(rds) == 2
        assert {rd.platform_id for rd in rds} == {"GPL96", "GPL97"}

    @patch("preprocessing.hybrid_source.GEOparse")
    def test_fetch_skips_unmatched_platform(self, _, tmp_path):
        """If the downloaded matrix's platform has no matching GPL in the GSE, fetch() raises."""
        from preprocessing.hybrid_source import HybridGeoSource

        expr_df = _make_expr_df()
        gse = _make_mock_gse(["GPL97"], list(expr_df.columns))

        source = HybridGeoSource.__new__(HybridGeoSource)
        source.cache_dir = tmp_path
        source._download_and_parse_series_matrices = MagicMock(return_value={"GPL96": expr_df})
        source._load_geoparse = MagicMock(return_value=gse)

        with pytest.raises(ValueError, match="No platforms"):
            source.fetch("GSE11909")

    def test_parse_series_matrix(self, tmp_path):
        """A gzipped series matrix file parses into its platform ID and expression DataFrame."""
        from preprocessing.hybrid_source import HybridGeoSource

        lines = [
            '!Sample_platform_id\t"GPL96"\t"GPL96"',
            '"ID_REF"\t"GSM100"\t"GSM101"',
            '"1000_at"\t"8.12"\t"7.55"',
            '"1001_at"\t"6.03"\t"9.22"',
        ]
        gz_path = tmp_path / "test_matrix.txt.gz"
        with gzip.open(str(gz_path), "wt") as f:
            f.write("\n".join(lines))

        platform_id, df = HybridGeoSource._parse_series_matrix(gz_path)
        assert platform_id == "GPL96"
        assert df.shape == (2, 2)
        assert df.loc["1000_at", "GSM100"] == pytest.approx(8.12)


# ============================================================================
# HybridMicroarrayHandler
# ============================================================================


class TestHybridMicroarrayHandler:
    """Tests for HybridMicroarrayHandler.parse() producing ProcessedDataset objects."""

    def _make_raw(self, n_probes=10, genes=None):
        """Build a RawDataset with a random expression matrix and matching GPL/GSE mocks."""
        from preprocessing.base import RawDataset

        expr_df = _make_expr_df(n_probes=n_probes)
        probes = list(expr_df.index)
        gpl_table = _make_gpl_table(probes, genes)
        gse = _make_mock_gse(["GPL96"], list(expr_df.columns))

        return RawDataset(
            source_id="GSE99999",
            platform_id="GPL96",
            raw_data={
                "expr_df": expr_df,
                "gpl": MagicMock(table=gpl_table),
                "gse": gse,
                "platform_id": "GPL96",
            },
            metadata={
                "gse_id": "GSE99999",
                "platform": "GPL96",
                "n_features": n_probes,
                "n_samples": 3,
                "source": "test",
                "format": "Hybrid",
                "cache_dir": "/tmp/test_cache",
            },
        )

    def test_data_type(self):
        """data_type identifies this as the Hybrid handler."""
        from preprocessing.hybrid_handler import HybridMicroarrayHandler

        h = HybridMicroarrayHandler()
        assert "Hybrid" in h.data_type

    def test_parse_produces_processed_dataset(self):
        """parse() returns a ProcessedDataset with the expected shape, ID, and gene keys."""
        from preprocessing.base import ProcessedDataset
        from preprocessing.gene_annotator import GeneAnnotator
        from preprocessing.hybrid_handler import HybridMicroarrayHandler

        h = HybridMicroarrayHandler(annotator=GeneAnnotator(), validator=None)
        h._validator_loaded = True

        ds = h.parse(self._make_raw())
        assert isinstance(ds, ProcessedDataset)
        assert ds.geo_id == "GSE99999_GPL96"
        assert ds.expr.shape == (10, 3)
        assert len(ds.keys) == 10
        assert "Hybrid" in ds.metadata["handler"]

    def test_keys_aligned_with_expr(self):
        """The number of gene keys always matches the number of expression rows."""
        from preprocessing.gene_annotator import GeneAnnotator
        from preprocessing.hybrid_handler import HybridMicroarrayHandler

        h = HybridMicroarrayHandler(annotator=GeneAnnotator(), validator=None)
        h._validator_loaded = True
        ds = h.parse(self._make_raw(genes=[f"ABC{i}" for i in range(10)]))
        assert len(ds.keys) == len(ds.expr)

    def test_pheno_matches_expr_columns(self):
        """The pheno table's row index matches the expression matrix's sample columns."""
        from preprocessing.gene_annotator import GeneAnnotator
        from preprocessing.hybrid_handler import HybridMicroarrayHandler

        h = HybridMicroarrayHandler(annotator=GeneAnnotator(), validator=None)
        h._validator_loaded = True
        ds = h.parse(self._make_raw())
        assert list(ds.pheno.index) == list(ds.expr.columns)

    def test_normalize_is_noop(self):
        """normalize() returns the same dataset object unchanged (parse() already normalized)."""
        from preprocessing.gene_annotator import GeneAnnotator
        from preprocessing.hybrid_handler import HybridMicroarrayHandler

        h = HybridMicroarrayHandler(annotator=GeneAnnotator(), validator=None)
        h._validator_loaded = True
        ds = h.parse(self._make_raw())
        assert h.normalize(ds) is ds

    def test_auto_log2_skips_log_scale(self):
        """Data already on a log2-like scale is left unchanged.

        Since 0.12.0 the decision is R's QQ-plot check, which needs a
        realistic number of values, so this uses 500 probes and a seed.
        """
        from preprocessing.hybrid_handler import HybridMicroarrayHandler

        rng = np.random.RandomState(7)
        df = pd.DataFrame(rng.normal(8.0, 1.5, (500, 3)), columns=["A", "B", "C"])
        result = HybridMicroarrayHandler._auto_log2(df.copy(), "test")
        np.testing.assert_array_almost_equal(result.values, df.values)

    def test_auto_log2_transforms_raw(self):
        """Data on a raw (non-log) scale is log2-transformed."""
        from preprocessing.hybrid_handler import HybridMicroarrayHandler

        rng = np.random.RandomState(7)
        df = pd.DataFrame(rng.lognormal(7.0, 1.5, (500, 3)), columns=["A", "B", "C"])
        result = HybridMicroarrayHandler._auto_log2(df.copy(), "test")
        np.testing.assert_array_almost_equal(result.values, np.log2(df.values))

    def test_parse_records_the_log_scale_decision(self):
        """parse() stores R's decision in metadata['log_scale_check']."""
        from preprocessing.gene_annotator import GeneAnnotator
        from preprocessing.hybrid_handler import HybridMicroarrayHandler

        h = HybridMicroarrayHandler(annotator=GeneAnnotator(), validator=None)
        h._validator_loaded = True
        ds = h.parse(self._make_raw())

        report = ds.metadata["log_scale_check"]
        assert report["rule"].startswith("R MetaIntegrator getGEOData()")
        assert isinstance(report["log2_applied"], bool)
        assert "legacy_median_rule_log2" in report

    def test_validator_lazy_loading(self):
        """The Ensembl validator is constructed on first use only, then reused."""
        from preprocessing.hybrid_handler import HybridMicroarrayHandler

        h = HybridMicroarrayHandler(validator=None)
        assert h._validator_loaded is False

        with patch("preprocessing.hybrid_handler.EnsemblGeneValidator") as MockV:
            mock_inst = MagicMock()
            mock_inst.valid_genes = {"TP53", "BRCA1"}
            MockV.return_value = mock_inst

            result = h._get_validator()
            assert h._validator_loaded is True
            assert result is mock_inst
            MockV.assert_called_once()

            h._get_validator()
            MockV.assert_called_once()  # still just once


# ============================================================================
# Backward-compatibility aliases
# ============================================================================


class TestBackwardCompatAliases:
    """Tests that the old r_parity names are aliases for the renamed hybrid classes."""

    def test_r_parity_geo_source_is_hybrid(self):
        """RParityGeoSource is literally HybridGeoSource, not a separate copy."""
        from preprocessing.hybrid_source import HybridGeoSource
        from preprocessing.r_parity_source import RParityGeoSource

        assert RParityGeoSource is HybridGeoSource

    def test_r_parity_handler_is_hybrid(self):
        """RParityMicroarrayHandler is literally HybridMicroarrayHandler."""
        from preprocessing.hybrid_handler import HybridMicroarrayHandler
        from preprocessing.r_parity_handler import RParityMicroarrayHandler

        assert RParityMicroarrayHandler is HybridMicroarrayHandler

    def test_init_exports_both_names(self):
        """Both the hybrid and legacy r_parity names are importable from the package root."""
        from preprocessing import HybridGeoSource, RParityGeoSource

        assert HybridGeoSource is RParityGeoSource

        from preprocessing import HybridMicroarrayHandler, RParityMicroarrayHandler

        assert HybridMicroarrayHandler is RParityMicroarrayHandler


# ============================================================================
# Thin wrapper + factory
# ============================================================================


class TestThinWrapper:
    """Tests for the legacy load_geo_datasets_r_parity() wrapper around the factory."""

    @patch("preprocessing.loader_factory.DataLoaderFactory")
    def test_wrapper_delegates_to_factory(self, MockFactory):
        """The wrapper asks the factory for a "hybrid"-strategy loader."""
        from preprocessing.r_parity_loader import load_geo_datasets_r_parity

        mock_loader = MagicMock()
        mock_loader.load_multiple.return_value = []
        MockFactory.create_geo_loader.return_value = mock_loader

        load_geo_datasets_r_parity(["GSE11909"], cache_dir="/tmp/test")

        MockFactory.create_geo_loader.assert_called_once_with(
            strategy="hybrid",
            cache_dir="/tmp/test",
        )

    @patch("preprocessing.loader_factory.DataLoaderFactory")
    def test_wrapper_passes_stop_on_error(self, MockFactory):
        """The wrapper forwards stop_on_error through to the loader's load_multiple()."""
        from preprocessing.r_parity_loader import load_geo_datasets_r_parity

        mock_loader = MagicMock()
        mock_loader.load_multiple.return_value = []
        MockFactory.create_geo_loader.return_value = mock_loader

        load_geo_datasets_r_parity(["GSE11909"], cache_dir="/tmp/x", stop_on_error=True)
        mock_loader.load_multiple.assert_called_once_with(["GSE11909"], stop_on_error=True)


class TestFactoryStrategies:
    """Tests for DataLoaderFactory's "hybrid" strategy and its aliases."""

    @patch("preprocessing.loader_factory.HybridMicroarrayHandler")
    @patch("preprocessing.loader_factory.HybridGeoSource")
    def test_hybrid_strategy(self, MockSource, MockHandler):
        """strategy="hybrid" builds a DataLoader wired to HybridGeoSource."""
        from preprocessing.base import DataLoader
        from preprocessing.loader_factory import DataLoaderFactory

        loader = DataLoaderFactory.create_geo_loader(strategy="hybrid", cache_dir="/tmp/t")
        assert isinstance(loader, DataLoader)
        MockSource.assert_called_once_with(cache_dir="/tmp/t")

    @patch("preprocessing.loader_factory.HybridMicroarrayHandler")
    @patch("preprocessing.loader_factory.HybridGeoSource")
    def test_r_parity_alias(self, MockSource, MockHandler):
        """'r_parity' is accepted as alias for 'hybrid'."""
        from preprocessing.base import DataLoader
        from preprocessing.loader_factory import DataLoaderFactory

        loader = DataLoaderFactory.create_geo_loader(strategy="r_parity")
        assert isinstance(loader, DataLoader)
        MockSource.assert_called_once()

    @patch("preprocessing.loader_factory.HybridMicroarrayHandler")
    @patch("preprocessing.loader_factory.HybridGeoSource")
    def test_geo_hybrid_source_type(self, MockSource, MockHandler):
        """source_type="geo_hybrid" builds a working microarray loader."""
        from preprocessing.loader_factory import DataLoaderFactory

        loader = DataLoaderFactory.create_microarray_loader(source_type="geo_hybrid")
        assert loader is not None
        MockSource.assert_called_once()

    @patch("preprocessing.loader_factory.HybridMicroarrayHandler")
    @patch("preprocessing.loader_factory.HybridGeoSource")
    def test_geo_r_parity_alias(self, MockSource, MockHandler):
        """source_type="geo_r_parity" is accepted as an alias for "geo_hybrid"."""
        from preprocessing.loader_factory import DataLoaderFactory

        loader = DataLoaderFactory.create_microarray_loader(source_type="geo_r_parity")
        assert loader is not None
        MockSource.assert_called_once()


class TestLoadGeoDatasetsProcessed:
    """Tests for load_geo_datasets_processed()'s default strategy and return type."""

    @patch("preprocessing.loader_factory.DataLoaderFactory.create_geo_loader")
    def test_defaults_to_hybrid(self, mock_create):
        """With no strategy argument, the "hybrid" strategy is used."""
        from preprocessing.loader_factory import load_geo_datasets_processed

        mock_loader = MagicMock()
        mock_loader.load_multiple.return_value = []
        mock_create.return_value = mock_loader

        load_geo_datasets_processed(["GSE11909"])
        mock_create.assert_called_once_with(
            strategy="hybrid",
            cache_dir="~/.geo_cache",
        )

    @patch("preprocessing.loader_factory.DataLoaderFactory.create_geo_loader")
    def test_returns_processed_datasets(self, mock_create):
        """The function returns the loader's ProcessedDataset results unmodified."""
        from preprocessing.base import ProcessedDataset
        from preprocessing.loader_factory import load_geo_datasets_processed

        mock_ds = MagicMock(spec=ProcessedDataset)
        mock_loader = MagicMock()
        mock_loader.load_multiple.return_value = [mock_ds]
        mock_create.return_value = mock_loader

        result = load_geo_datasets_processed(["GSE11909"])
        assert len(result) == 1
        assert result[0] is mock_ds
