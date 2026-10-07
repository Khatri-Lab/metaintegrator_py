"""Unit tests for ``utils.r_golden``.

Every test writes a tiny, hand-made golden directory to ``tmp_path``.
No R installation and no network access are needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.r_golden import compare_numeric_frames, find_golden_prefix, load_r_golden

PREFIX = "myMeta"


def _vector(path: Path, values) -> None:
    """Write an R vector the way robust_rdata_converter.R does."""
    pd.DataFrame({"index": range(1, len(values) + 1), "value": values}).to_csv(path, index=False)


def _write_golden(directory: Path, gz: bool = False, with_loo: bool = True) -> Path:
    """Write a minimal two-dataset golden set and return its directory."""
    ext = ".csv.gz" if gz else ".csv"
    directory.mkdir(parents=True, exist_ok=True)
    for name, n in [("Study.A", 4), ("Study.B", 6)]:
        stem = directory / f"{PREFIX}_originalData_{name}"
        samples = [f"S{i}" for i in range(n)]
        expr = pd.DataFrame(
            np.arange(3 * n, dtype=float).reshape(3, n),
            index=["p1", "p2", "p3"],
            columns=samples,
        )
        expr.to_csv(f"{stem}_expr{ext}")
        _vector(Path(f"{stem}_keys.csv"), ["G1", "G2,G3", None])
        _vector(Path(f"{stem}_class.csv"), [0, 1] * (n // 2))
        pd.DataFrame({"group": ["c", "d"] * (n // 2)}, index=samples).to_csv(f"{stem}_pheno.csv")
        (directory / f"{PREFIX}_originalData_{name}_formattedName.txt").write_text(f"{name} pretty")

    pooled = pd.DataFrame(
        {"effectSize": [0.5, -0.2], "effectSizeFDR": [0.01, 0.2]}, index=["G1", "G2"]
    )
    pooled.to_csv(directory / f"{PREFIX}_metaAnalysis_pooledResults{ext}")
    es = pd.DataFrame({"Study.B": [0.4, 0.1], "Study.A": [0.6, np.nan]}, index=["G1", "G2"])
    es.to_csv(directory / f"{PREFIX}_metaAnalysis_datasetEffectSizes.csv")
    es.to_csv(directory / f"{PREFIX}_metaAnalysis_datasetEffectSizeStandardErrors.csv")
    if with_loo:
        pooled.to_csv(directory / f"{PREFIX}_leaveOneOutAnalysis_removed_Study.A_pooledResults.csv")

    flt = directory / f"{PREFIX}_filterResults_FDR0.05_es0_nStudies1_looaTRUE_hetero0"
    _vector(Path(f"{flt}_FDRThresh.csv"), [0.05])
    _vector(Path(f"{flt}_effectSizeThresh.csv"), [0])
    _vector(Path(f"{flt}_numberStudiesThresh.csv"), [1])
    _vector(Path(f"{flt}_heterogeneityPvalThresh.csv"), [0])
    _vector(Path(f"{flt}_isLeaveOneOut.csv"), ["TRUE"])
    _vector(Path(f"{flt}_posGeneNames.csv"), ["G1"])
    # No negGeneNames file: R's exporter skips empty vectors.
    return directory


class TestLoadRGolden:
    """Tests for load_r_golden() and find_golden_prefix()."""

    def test_finds_prefix(self, tmp_path):
        """The prefix comes from the pooledResults file name."""
        assert find_golden_prefix(_write_golden(tmp_path / "g")) == PREFIX

    def test_missing_pooled_results_raises(self, tmp_path):
        """A directory without pooledResults is not a golden set."""
        tmp_path.joinpath("x.csv").write_text("a\n1\n")
        with pytest.raises(FileNotFoundError):
            find_golden_prefix(tmp_path)

    def test_dataset_order_follows_r_columns(self, tmp_path):
        """Dataset order is R's datasetEffectSizes column order."""
        golden = load_r_golden(_write_golden(tmp_path / "g"))
        assert golden.dataset_names == ["Study.B", "Study.A"]
        assert list(golden.original_data) == ["Study.B", "Study.A"]

    def test_dataset_contents(self, tmp_path):
        """Expression, keys, class, pheno and formattedName are all loaded."""
        golden = load_r_golden(_write_golden(tmp_path / "g"))
        ds = golden.original_data["Study.A"]
        assert ds["expr"].shape == (3, 4)
        assert ds["keys"] == ["G1", "G2,G3", "NA"]  # R NA becomes the 'NA' string
        assert list(ds["class"]) == [0, 1, 0, 1]
        assert list(ds["pheno"].index) == list(ds["expr"].columns)
        assert ds["formattedName"] == "Study.A pretty"

    def test_reads_gzip_files(self, tmp_path):
        """Large golden sets can be stored as .csv.gz."""
        golden = load_r_golden(_write_golden(tmp_path / "g", gz=True))
        assert golden.original_data["Study.B"]["expr"].shape == (3, 6)
        assert list(golden.pooled_results.index) == ["G1", "G2"]

    def test_leave_one_out_is_optional(self, tmp_path):
        """Leave-one-out results load when present and are empty otherwise."""
        assert set(load_r_golden(_write_golden(tmp_path / "a")).leave_one_out) == {"Study.A"}
        assert load_r_golden(_write_golden(tmp_path / "b", with_loo=False)).leave_one_out == {}

    def test_filter_results(self, tmp_path):
        """Filter thresholds and gene lists are parsed. A missing list is empty."""
        (flt,) = load_r_golden(_write_golden(tmp_path / "g")).filter_results
        assert flt.name == "FDR0.05_es0_nStudies1_looaTRUE_hetero0"
        assert flt.pos_gene_names == ["G1"]
        assert flt.neg_gene_names == []
        assert flt.is_leave_one_out is True
        assert flt.fdr_thresh == pytest.approx(0.05)
        assert flt.number_studies_thresh == 1

    def test_keys_length_mismatch_raises(self, tmp_path):
        """Keys that do not line up with expression rows are rejected."""
        directory = _write_golden(tmp_path / "g")
        _vector(directory / f"{PREFIX}_originalData_Study.A_keys.csv", ["G1"])
        with pytest.raises(ValueError, match="align"):
            load_r_golden(directory)

    def test_to_meta_object_returns_independent_copies(self, tmp_path):
        """Changing one meta_object must not change the golden data."""
        golden = load_r_golden(_write_golden(tmp_path / "g"))
        meta = golden.to_meta_object()
        meta["originalData"]["Study.A"]["expr"].iloc[0, 0] = -999.0
        meta["originalData"]["Study.A"]["keys"][0] = "CHANGED"
        assert golden.original_data["Study.A"]["expr"].iloc[0, 0] != -999.0
        assert golden.original_data["Study.A"]["keys"][0] == "G1"


class TestCompareNumericFrames:
    """Tests for compare_numeric_frames()."""

    @staticmethod
    def _r():
        return pd.DataFrame(
            {"a": [1.0, 2.0, np.nan], "b": [1e-20, 5.0, 6.0]}, index=["x", "y", "z"]
        )

    def test_identical_tables_pass(self):
        """A table always matches itself, row order aside."""
        r = self._r()
        report = compare_numeric_frames(r.iloc[::-1], r)
        assert report.ok
        assert report.summary() == "OK"

    def test_tiny_relative_noise_passes(self):
        """Floating-point noise far below rtol is accepted, even for tiny values."""
        py = self._r() * (1 + 1e-12)
        assert compare_numeric_frames(py, self._r()).ok

    def test_value_difference_is_reported(self):
        """A real difference names the column and the worst row."""
        py = self._r()
        py.loc["y", "b"] = 5.1
        report = compare_numeric_frames(py, self._r())
        assert not report.ok
        (bad,) = [c for c in report.columns if not c.ok]
        assert (bad.column, bad.worst_row, bad.n_mismatch) == ("b", "y", 1)
        assert "b: 1 rows differ" in report.summary()

    def test_one_sided_missing_value_is_a_mismatch(self):
        """NaN on one side only is a mismatch. NaN on both sides is fine."""
        py = self._r()
        py.loc["x", "a"] = np.nan
        report = compare_numeric_frames(py, self._r())
        (bad,) = [c for c in report.columns if not c.ok]
        assert bad.n_missing_mismatch == 1

    def test_row_and_column_differences_are_reported(self):
        """Extra rows, missing rows and missing columns all fail the report."""
        py = self._r().drop(index="z").drop(columns="b")
        py.loc["w"] = [0.0]
        report = compare_numeric_frames(py, self._r())
        assert report.only_in_python == ["w"]
        assert report.only_in_r == ["z"]
        assert report.missing_columns == ["b"]
        assert not report.ok


class TestDiscoverGoldenDirs:
    """Tests for the golden-set discovery in parity_support."""

    def test_extra_dirs_from_env_are_added(self, tmp_path, monkeypatch):
        """Directories in MIPY_PARITY_GOLDEN_DIRS are added after the built-ins."""
        from .parity_support import BUILTIN_GOLDEN_DIRS, GOLDEN_DIRS_ENV, discover_golden_dirs

        extra = _write_golden(tmp_path / "sle")
        monkeypatch.setenv(GOLDEN_DIRS_ENV, str(extra))
        dirs = discover_golden_dirs()
        assert dirs[-1] == extra
        assert len(dirs) == len([d for d in BUILTIN_GOLDEN_DIRS if d.is_dir()]) + 1

    def test_missing_env_dir_raises(self, tmp_path, monkeypatch):
        """A typo in MIPY_PARITY_GOLDEN_DIRS fails loudly instead of skipping tests."""
        from .parity_support import GOLDEN_DIRS_ENV, discover_golden_dirs

        monkeypatch.setenv(GOLDEN_DIRS_ENV, str(tmp_path / "does_not_exist"))
        with pytest.raises(FileNotFoundError, match="missing"):
            discover_golden_dirs()
