"""
Comprehensive tests for base.py: RawDataset, ProcessedDataset, DataLoader,
and the DataSource / DataHandler abstract interfaces.

Covers:
  - RawDataset construction and defaults
  - ProcessedDataset validation (expr/pheno alignment, keys type and length)
  - DataLoader orchestration (load, load_multiple, error handling)
  - DataSource / DataHandler contract enforcement
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from preprocessing.base import (
    DataHandler,
    DataLoader,
    DataSource,
    ProcessedDataset,
    RawDataset,
)

# ============================================================================
# Helpers
# ============================================================================


def _expr(n_probes=5, n_samples=3):
    probes = [f"P{i}" for i in range(n_probes)]
    samples = [f"S{i}" for i in range(n_samples)]
    data = np.random.RandomState(42).normal(8, 1, (n_probes, n_samples))
    return pd.DataFrame(data, index=probes, columns=samples, dtype="float64")


def _pheno(samples):
    return pd.DataFrame({"title": [f"Title {s}" for s in samples]}, index=samples)


def _keys(n):
    return [f"GENE{i}" for i in range(n)]


# ============================================================================
# RawDataset
# ============================================================================


class TestRawDataset:

    def test_defaults(self):
        rd = RawDataset(source_id="GSE123")
        assert rd.source_id == "GSE123"
        assert rd.platform_id is None
        assert rd.raw_data is None
        assert rd.metadata == {}

    def test_metadata_default_not_shared(self):
        rd1 = RawDataset(source_id="A")
        rd2 = RawDataset(source_id="B")
        rd1.metadata["x"] = 1
        assert "x" not in rd2.metadata

    def test_full_construction(self):
        rd = RawDataset(
            source_id="GSE11909",
            platform_id="GPL96",
            raw_data={"matrix": pd.DataFrame()},
            metadata={"source": "test"},
        )
        assert rd.platform_id == "GPL96"
        assert "matrix" in rd.raw_data
        assert rd.metadata["source"] == "test"


# ============================================================================
# ProcessedDataset — validation
# ============================================================================


class TestProcessedDataset:

    def test_valid_construction(self):
        expr = _expr()
        pheno = _pheno(list(expr.columns))
        keys = _keys(len(expr))

        ds = ProcessedDataset(
            geo_id="GSE123_GPL96",
            expr=expr,
            pheno=pheno,
            keys=keys,
            metadata={"source": "test"},
        )
        assert ds.geo_id == "GSE123_GPL96"
        assert ds.expr.shape == (5, 3)
        assert len(ds.keys) == 5

    def test_sample_mismatch_raises(self):
        expr = _expr(n_probes=3, n_samples=3)
        pheno = _pheno(["X", "Y"])  # 2 samples ≠ 3 expr columns

        with pytest.raises(ValueError, match="Sample IDs mismatch"):
            ProcessedDataset(
                geo_id="test",
                expr=expr,
                pheno=pheno,
                keys=_keys(3),
                metadata={},
            )

    def test_keys_wrong_type_raises(self):
        expr = _expr(n_probes=3, n_samples=2)
        pheno = _pheno(list(expr.columns))

        with pytest.raises(TypeError, match="keys must be list"):
            ProcessedDataset(
                geo_id="test",
                expr=expr,
                pheno=pheno,
                keys={"P0": "GENE0"},  # dict, not list
                metadata={},
            )

    def test_keys_wrong_length_raises(self):
        expr = _expr(n_probes=3, n_samples=2)
        pheno = _pheno(list(expr.columns))

        with pytest.raises(ValueError, match="keys length"):
            ProcessedDataset(
                geo_id="test",
                expr=expr,
                pheno=pheno,
                keys=["GENE0"],  # 1 ≠ 3 probes
                metadata={},
            )

    def test_keys_with_na(self):
        expr = _expr(n_probes=3, n_samples=2)
        pheno = _pheno(list(expr.columns))
        keys = ["TP53", "NA", "BRCA1"]

        ds = ProcessedDataset(
            geo_id="test",
            expr=expr,
            pheno=pheno,
            keys=keys,
            metadata={},
        )
        assert ds.keys[1] == "NA"

    def test_empty_dataset(self):
        expr = pd.DataFrame(dtype="float64")
        pheno = pd.DataFrame()

        ds = ProcessedDataset(
            geo_id="empty",
            expr=expr,
            pheno=pheno,
            keys=[],
            metadata={},
        )
        assert len(ds.keys) == 0
        assert ds.expr.empty


# ============================================================================
# DataLoader — orchestration
# ============================================================================


class _FakeSource(DataSource):
    """Minimal concrete source for testing."""

    def __init__(self, raw_datasets=None):
        super().__init__("/tmp/test")
        self._raw = raw_datasets or []

    @property
    def source_name(self):
        return "FakeSource"

    def fetch(self, dataset_id, **kwargs):
        return [rd for rd in self._raw if rd.source_id == dataset_id]

    def fetch_multiple(self, dataset_ids, stop_on_error=False, **kwargs):
        result = []
        for did in dataset_ids:
            result.extend(self.fetch(did))
        return result


class _FakeHandler(DataHandler):
    """Minimal concrete handler for testing."""

    @property
    def data_type(self):
        return "FakeHandler"

    def parse(self, raw_dataset):
        expr = _expr(n_probes=3, n_samples=2)
        pheno = _pheno(list(expr.columns))
        return ProcessedDataset(
            geo_id=f"{raw_dataset.source_id}_{raw_dataset.platform_id}",
            expr=expr,
            pheno=pheno,
            keys=_keys(3),
            metadata={},
        )

    def normalize(self, dataset):
        return dataset


class _FailingHandler(DataHandler):
    """Handler that always raises."""

    @property
    def data_type(self):
        return "FailingHandler"

    def parse(self, raw_dataset):
        raise ValueError("parse failed deliberately")

    def normalize(self, dataset):
        return dataset


class TestDataLoader:

    def _make_loader(self, raw_datasets=None, handler=None):
        source = _FakeSource(raw_datasets or [])
        handler = handler or _FakeHandler()
        return DataLoader(source, handler)

    def test_load_single(self):
        rd = RawDataset(source_id="GSE1", platform_id="GPL96")
        loader = self._make_loader([rd])

        datasets = loader.load("GSE1")
        assert len(datasets) == 1
        assert datasets[0].geo_id == "GSE1_GPL96"

    def test_load_no_results(self):
        loader = self._make_loader([])
        datasets = loader.load("GSE_MISSING")
        assert datasets == []

    def test_load_multiple(self):
        rds = [
            RawDataset(source_id="GSE1", platform_id="GPL96"),
            RawDataset(source_id="GSE2", platform_id="GPL570"),
        ]
        loader = self._make_loader(rds)

        datasets = loader.load_multiple(["GSE1", "GSE2"])
        assert len(datasets) == 2
        ids = {ds.geo_id for ds in datasets}
        assert ids == {"GSE1_GPL96", "GSE2_GPL570"}

    def test_load_multiple_skip_errors(self):
        rds = [
            RawDataset(source_id="GSE1", platform_id="GPL96"),
            RawDataset(source_id="GSE2", platform_id="GPL570"),
        ]
        loader = self._make_loader(rds, handler=_FailingHandler())

        datasets = loader.load_multiple(["GSE1", "GSE2"], stop_on_error=False)
        assert datasets == []  # all failed but didn't raise

    def test_load_multiple_stop_on_error(self):
        rds = [RawDataset(source_id="GSE1", platform_id="GPL96")]
        loader = self._make_loader(rds, handler=_FailingHandler())

        # load_multiple re-raises the original exception (not wrapped)
        with pytest.raises(ValueError, match="parse failed deliberately"):
            loader.load_multiple(["GSE1"], stop_on_error=True)

    def test_load_error_wraps_original(self):
        rd = RawDataset(source_id="GSE1", platform_id="GPL96")
        loader = self._make_loader([rd], handler=_FailingHandler())

        with pytest.raises(RuntimeError, match="parse failed deliberately"):
            loader.load("GSE1")

    def test_repr(self):
        loader = self._make_loader()
        r = repr(loader)
        assert "FakeSource" in r
        assert "FakeHandler" in r

    def test_process_calls_parse_then_normalize(self):
        handler = MagicMock(spec=DataHandler)
        handler.data_type = "MockHandler"

        expr = _expr(n_probes=3, n_samples=2)
        pheno = _pheno(list(expr.columns))
        mock_ds = ProcessedDataset(
            geo_id="test",
            expr=expr,
            pheno=pheno,
            keys=_keys(3),
            metadata={},
        )
        handler.process.return_value = mock_ds

        rd = RawDataset(source_id="GSE1", platform_id="GPL96")
        loader = DataLoader(_FakeSource([rd]), handler)
        loader.load("GSE1")

        handler.process.assert_called_once_with(rd)


# ============================================================================
# DataHandler.process template method
# ============================================================================


class TestHandlerProcessTemplate:

    def test_process_calls_parse_then_normalize(self):
        handler = _FakeHandler()
        rd = RawDataset(source_id="GSE1", platform_id="GPL96")

        # Spy on parse and normalize
        original_parse = handler.parse
        original_normalize = handler.normalize
        parse_called = []
        normalize_called = []

        def spy_parse(raw):
            parse_called.append(True)
            return original_parse(raw)

        def spy_normalize(ds):
            normalize_called.append(True)
            return original_normalize(ds)

        handler.parse = spy_parse
        handler.normalize = spy_normalize

        result = handler.process(rd)

        assert len(parse_called) == 1
        assert len(normalize_called) == 1
        assert isinstance(result, ProcessedDataset)


# ============================================================================
# geo_sources validation functions
# ============================================================================


class TestGeoValidation:

    def test_validate_geo_id_gse(self):
        from preprocessing.geo_sources import validate_geo_id

        t, i = validate_geo_id("gse11909")
        assert t == "GSE"
        assert i == "GSE11909"

    def test_validate_geo_id_gsm(self):
        from preprocessing.geo_sources import validate_geo_id

        t, i = validate_geo_id("GSM12345")
        assert t == "GSM"

    def test_validate_geo_id_gpl(self):
        from preprocessing.geo_sources import validate_geo_id

        t, i = validate_geo_id("gpl570")
        assert t == "GPL"
        assert i == "GPL570"

    def test_validate_geo_id_invalid(self):
        from preprocessing.geo_sources import validate_geo_id

        with pytest.raises(ValueError, match="Invalid GEO ID"):
            validate_geo_id("INVALID123")

    def test_validate_geo_id_empty(self):
        from preprocessing.geo_sources import validate_geo_id

        with pytest.raises(ValueError):
            validate_geo_id("")

    def test_validate_series_id_accepts_gse(self):
        from preprocessing.geo_sources import validate_series_id

        assert validate_series_id("GSE2034") == "GSE2034"

    def test_validate_series_id_rejects_gsm(self):
        from preprocessing.geo_sources import validate_series_id

        with pytest.raises(ValueError, match="Expected GSE"):
            validate_series_id("GSM12345")

    def test_validate_series_id_rejects_gpl(self):
        from preprocessing.geo_sources import validate_series_id

        with pytest.raises(ValueError, match="Expected GSE"):
            validate_series_id("GPL570")

    def test_validate_series_id_lowercased(self):
        from preprocessing.geo_sources import validate_series_id

        assert validate_series_id("gse2034") == "GSE2034"

    def test_validate_series_id_with_whitespace(self):
        from preprocessing.geo_sources import validate_series_id

        assert validate_series_id("  GSE2034  ") == "GSE2034"


# ============================================================================
# DataLoaderFactory corner cases
# ============================================================================


class TestFactoryCornerCases:

    def test_unknown_strategy_raises(self):
        from preprocessing.loader_factory import DataLoaderFactory

        with pytest.raises(ValueError, match="Unknown"):
            DataLoaderFactory.create_microarray_loader(source_type="nonexistent")

    def test_rnaseq_not_implemented(self):
        from preprocessing.loader_factory import DataLoaderFactory

        with pytest.raises(NotImplementedError):
            DataLoaderFactory.create_rnaseq_loader()

    def test_single_cell_not_implemented(self):
        from preprocessing.loader_factory import DataLoaderFactory

        with pytest.raises(NotImplementedError):
            DataLoaderFactory.create_single_cell_loader()

    @patch("preprocessing.loader_factory.HybridMicroarrayHandler")
    @patch("preprocessing.loader_factory.HybridGeoSource")
    def test_load_geo_datasets_legacy_format(self, MockSource, MockHandler):
        from preprocessing.loader_factory import load_geo_datasets

        expr = _expr(n_probes=3, n_samples=2)
        pheno = _pheno(list(expr.columns))
        mock_ds = ProcessedDataset(
            geo_id="GSE1_GPL96",
            expr=expr,
            pheno=pheno,
            keys=_keys(3),
            metadata={"source": "test"},
        )

        mock_loader = MagicMock()
        mock_loader.load_multiple.return_value = [mock_ds]

        with patch(
            "preprocessing.loader_factory.DataLoaderFactory.create_geo_loader",
            return_value=mock_loader,
        ):
            result = load_geo_datasets(["GSE1"], strategy="hybrid")

        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["geo_id"] == "GSE1_GPL96"
        assert "expr" in result[0]
        assert "pheno" in result[0]
        assert "keys" in result[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
