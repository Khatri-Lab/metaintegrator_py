"""Load R MetaIntegrator "golden" outputs and compare them with Python results.

A golden directory holds one R ``metaObject`` exported to CSV.
The export is made by ``src/utils/robust_rdata_converter.R``.
That script flattens every element of the object into files named like this::

    <prefix>_originalData_<dataset>_expr.csv
    <prefix>_originalData_<dataset>_keys.csv
    <prefix>_originalData_<dataset>_class.csv
    <prefix>_originalData_<dataset>_pheno.csv
    <prefix>_metaAnalysis_pooledResults.csv
    <prefix>_metaAnalysis_datasetEffectSizes.csv
    <prefix>_metaAnalysis_datasetEffectSizeStandardErrors.csv
    <prefix>_leaveOneOutAnalysis_removed_<dataset>_pooledResults.csv   (optional)
    <prefix>_filterResults_<filterName>_FDRThresh.csv                  (optional)
    ...

``<prefix>`` is the R object name, for example ``tinyMetaObject``.
Each file may also be gzip-compressed (``.csv.gz``).
That keeps large golden sets, such as the SLE tutorial, small on disk.

This module has two jobs:

1. :func:`load_r_golden` reads a golden directory into a
   :class:`RGoldenMetaObject`. Its ``original_data`` is ready to pass to
   ``MetaAnalyzer.run_meta_analysis()``.
2. :func:`compare_numeric_frames` compares a Python table with the R table
   and returns a :class:`FrameParityReport` that explains any mismatch.

The module has no dependency on the analysis code.
That way the parity tests can never "agree with themselves" by accident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "RFilterResult",
    "RGoldenMetaObject",
    "ColumnParity",
    "FrameParityReport",
    "find_golden_prefix",
    "load_r_golden",
    "compare_numeric_frames",
]

# Suffixes a golden file may use. Plain CSV is tried first.
_CSV_SUFFIXES = (".csv", ".csv.gz")

# R writes missing values as NA. An empty key means the same thing.
_NA_KEY = "NA"


# ============================================================================
# Data models
# ============================================================================


@dataclass(frozen=True)
class RFilterResult:
    """One ``filterGenes()`` result stored in an R ``metaObject``.

    Attributes:
        name: Filter name used by R, for example
            ``'FDR0.05_es0_nStudies1_looaTRUE_hetero0'``.
        pos_gene_names: Up-regulated genes, in R's order.
        neg_gene_names: Down-regulated genes, in R's order.
        effect_size_thresh: Minimum absolute effect size.
        fdr_thresh: Maximum FDR for both the effect-size and Fisher tests.
        number_studies_thresh: Minimum number of studies.
        heterogeneity_pval_thresh: Minimum heterogeneity p-value
            (0 disables the filter).
        is_leave_one_out: Whether R used the leave-one-out rule.
    """

    name: str
    pos_gene_names: List[str]
    neg_gene_names: List[str]
    effect_size_thresh: float
    fdr_thresh: float
    number_studies_thresh: int
    heterogeneity_pval_thresh: float
    is_leave_one_out: bool


@dataclass
class RGoldenMetaObject:
    """An R ``metaObject`` loaded from a golden directory.

    Attributes:
        name: Golden set name (the directory name).
        prefix: R object name used in the file names.
        dataset_names: Dataset names in R's order. This is the column order
            of ``dataset_effect_sizes`` and the order used for leave-one-out.
        original_data: Input datasets keyed by name. Each value has the keys
            ``expr``, ``keys``, ``class``, ``pheno`` and ``formattedName``,
            which is the layout ``MetaAnalyzer.run_meta_analysis()`` expects.
        pooled_results: R's ``metaAnalysis$pooledResults``, indexed by gene.
        dataset_effect_sizes: R's per-study effect sizes (genes x datasets).
        dataset_standard_errors: R's per-study standard errors.
        leave_one_out: R's leave-one-out ``pooledResults``, keyed by the name
            of the removed dataset. Empty if R did not run leave-one-out.
        filter_results: R's ``filterGenes()`` results. May be empty.
    """

    name: str
    prefix: str
    dataset_names: List[str]
    original_data: Dict[str, Dict]
    pooled_results: pd.DataFrame
    dataset_effect_sizes: pd.DataFrame
    dataset_standard_errors: pd.DataFrame
    leave_one_out: Dict[str, pd.DataFrame] = field(default_factory=dict)
    filter_results: List[RFilterResult] = field(default_factory=list)

    def to_meta_object(self) -> Dict:
        """Return a fresh Python ``meta_object`` built from the R inputs.

        Each call returns new dicts and copies of the tables. A test can
        therefore change the result without affecting other tests.

        Returns:
            A dict with a single ``'originalData'`` key, in R's dataset order.
        """
        return {
            "originalData": {
                name: {
                    "expr": ds["expr"].copy(),
                    "keys": list(ds["keys"]),
                    "class": ds["class"].copy(),
                    "pheno": ds["pheno"].copy(),
                    "formattedName": ds["formattedName"],
                }
                for name, ds in self.original_data.items()
            }
        }


@dataclass(frozen=True)
class ColumnParity:
    """Parity result for one column.

    Attributes:
        column: Column name.
        max_abs_diff: Largest absolute difference over rows where both
            values are present.
        n_mismatch: Rows outside the tolerance, including rows where only
            one side is missing.
        n_missing_mismatch: Rows where exactly one side is missing.
        worst_row: Row label of the largest difference, or ``None``.
    """

    column: str
    max_abs_diff: float
    n_mismatch: int
    n_missing_mismatch: int
    worst_row: Optional[str]

    @property
    def ok(self) -> bool:
        """True when every row is within tolerance."""
        return self.n_mismatch == 0


@dataclass
class FrameParityReport:
    """Result of :func:`compare_numeric_frames`.

    Attributes:
        only_in_python: Row labels found only in the Python table.
        only_in_r: Row labels found only in the R table.
        missing_columns: R columns that the Python table lacks.
        columns: Per-column parity for the shared columns and rows.
        rtol: Relative tolerance used.
        atol: Absolute tolerance used.
    """

    only_in_python: List[str]
    only_in_r: List[str]
    missing_columns: List[str]
    columns: List[ColumnParity]
    rtol: float
    atol: float

    @property
    def ok(self) -> bool:
        """True when rows, columns and values all agree."""
        return (
            not self.only_in_python
            and not self.only_in_r
            and not self.missing_columns
            and all(c.ok for c in self.columns)
        )

    def summary(self, max_items: int = 10) -> str:
        """Return a short, human-readable explanation for a failing test.

        Args:
            max_items: Maximum number of row labels to list per problem.

        Returns:
            A multi-line string. It is ``'OK'`` when everything matches.
        """
        if self.ok:
            return "OK"
        lines = [f"Parity failed (rtol={self.rtol:g}, atol={self.atol:g}):"]
        if self.only_in_python:
            lines.append(
                f"  {len(self.only_in_python)} rows only in Python: "
                f"{self.only_in_python[:max_items]}"
            )
        if self.only_in_r:
            lines.append(f"  {len(self.only_in_r)} rows only in R: {self.only_in_r[:max_items]}")
        if self.missing_columns:
            lines.append(f"  Columns missing in Python: {self.missing_columns}")
        for col in self.columns:
            if not col.ok:
                lines.append(
                    f"  {col.column}: {col.n_mismatch} rows differ "
                    f"({col.n_missing_mismatch} NA mismatches), "
                    f"max |diff|={col.max_abs_diff:.3g} at {col.worst_row!r}"
                )
        return "\n".join(lines)


# ============================================================================
# Loading
# ============================================================================


def _find_file(directory: Path, stem: str) -> Optional[Path]:
    """Return ``<stem>.csv`` or ``<stem>.csv.gz`` in *directory*, if present."""
    for suffix in _CSV_SUFFIXES:
        path = directory / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _require_file(directory: Path, stem: str) -> Path:
    """Like :func:`_find_file`, but raise when the file is missing."""
    path = _find_file(directory, stem)
    if path is None:
        raise FileNotFoundError(f"Golden file not found: {directory / stem}.csv[.gz]")
    return path


def _strip_csv_suffix(filename: str) -> str:
    """Remove a ``.csv`` or ``.csv.gz`` suffix from a file name."""
    for suffix in sorted(_CSV_SUFFIXES, key=len, reverse=True):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def _read_vector(path: Path) -> pd.Series:
    """Read an R vector exported as an ``"index","value"`` CSV."""
    return pd.read_csv(path)["value"]


def _read_scalar(directory: Path, stem: str, default: object) -> object:
    """Read a one-value R vector, or return *default* if the file is absent."""
    path = _find_file(directory, stem)
    if path is None:
        return default
    values = _read_vector(path)
    return default if values.empty else values.iloc[0]


def _read_gene_list(directory: Path, stem: str) -> List[str]:
    """Read a gene-name vector. R may skip writing an empty vector."""
    path = _find_file(directory, stem)
    if path is None:
        return []
    return [str(g) for g in _read_vector(path).dropna()]


def _to_key(value: object) -> str:
    """Convert one exported R key to the Python convention (``'NA'``)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return _NA_KEY
    text = str(value).strip()
    return text if text else _NA_KEY


def _to_bool(value: object) -> bool:
    """Convert an exported R logical (TRUE/FALSE, or 1/0) to ``bool``."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().upper() in {"TRUE", "T", "1"}


def find_golden_prefix(directory: Path) -> str:
    """Find the R object name used in a golden directory's file names.

    The prefix is taken from the single ``*_metaAnalysis_pooledResults``
    file in the directory.

    Args:
        directory: Golden directory.

    Returns:
        The prefix, for example ``'tinyMetaObject'``.

    Raises:
        FileNotFoundError: If no pooled-results file exists.
        ValueError: If more than one R object is present.
    """
    marker = "_metaAnalysis_pooledResults"
    prefixes = sorted(
        {
            _strip_csv_suffix(p.name)[: -len(marker)]
            for p in Path(directory).iterdir()
            if _strip_csv_suffix(p.name).endswith(marker)
        }
    )
    if not prefixes:
        raise FileNotFoundError(f"No '*{marker}.csv[.gz]' file in {directory}")
    if len(prefixes) > 1:
        raise ValueError(f"More than one R object in {directory}: {prefixes}")
    return prefixes[0]


def _load_dataset(directory: Path, prefix: str, name: str) -> Dict:
    """Load one ``originalData`` entry."""
    stem = f"{prefix}_originalData_{name}"
    expr = pd.read_csv(_require_file(directory, f"{stem}_expr"), index_col=0)
    expr.index = expr.index.astype(str)
    expr.columns = expr.columns.astype(str)

    # The exporter drops the names of R's keys vector. R itself requires
    # names(keys) == rownames(expr), so row order is the link.
    keys = [_to_key(k) for k in _read_vector(_require_file(directory, f"{stem}_keys"))]
    if len(keys) != len(expr):
        raise ValueError(
            f"{name}: {len(keys)} keys but {len(expr)} expression rows. "
            "Keys must align 1:1 with expression rows."
        )

    class_vector = _read_vector(_require_file(directory, f"{stem}_class")).to_numpy()
    if len(class_vector) != expr.shape[1]:
        raise ValueError(f"{name}: {len(class_vector)} class labels but {expr.shape[1]} samples")

    pheno_path = _find_file(directory, f"{stem}_pheno")
    if pheno_path is not None:
        pheno = pd.read_csv(pheno_path, index_col=0)
        pheno.index = pheno.index.astype(str)
    else:
        pheno = pd.DataFrame(index=expr.columns)

    formatted_path = directory / f"{stem}_formattedName.txt"
    formatted = formatted_path.read_text().strip() if formatted_path.exists() else name

    return {
        "expr": expr,
        "keys": keys,
        "class": class_vector,
        "pheno": pheno,
        "formattedName": formatted,
    }


def _load_filter_results(directory: Path, prefix: str) -> List[RFilterResult]:
    """Load every ``filterResults`` entry, discovered by its FDR file."""
    marker_re = re.compile(rf"^{re.escape(prefix)}_filterResults_(.+)_FDRThresh$")
    results = []
    for path in sorted(directory.iterdir()):
        match = marker_re.match(_strip_csv_suffix(path.name))
        if not match:
            continue
        filter_name = match.group(1)
        stem = f"{prefix}_filterResults_{filter_name}"
        results.append(
            RFilterResult(
                name=filter_name,
                pos_gene_names=_read_gene_list(directory, f"{stem}_posGeneNames"),
                neg_gene_names=_read_gene_list(directory, f"{stem}_negGeneNames"),
                effect_size_thresh=float(_read_scalar(directory, f"{stem}_effectSizeThresh", 0)),
                fdr_thresh=float(_read_scalar(directory, f"{stem}_FDRThresh", 0.05)),
                number_studies_thresh=int(
                    _read_scalar(directory, f"{stem}_numberStudiesThresh", 1)
                ),
                heterogeneity_pval_thresh=float(
                    _read_scalar(directory, f"{stem}_heterogeneityPvalThresh", 0)
                ),
                is_leave_one_out=_to_bool(_read_scalar(directory, f"{stem}_isLeaveOneOut", False)),
            )
        )
    return results


def load_r_golden(directory: Path | str, prefix: Optional[str] = None) -> RGoldenMetaObject:
    """Load an exported R ``metaObject`` from a golden directory.

    Args:
        directory: Directory produced by ``robust_rdata_converter.R``.
        prefix: R object name in the file names. Found automatically
            when not given.

    Returns:
        The loaded golden object.

    Raises:
        FileNotFoundError: If a required file is missing.
        ValueError: If the files are inconsistent with each other.
    """
    directory = Path(directory)
    prefix = prefix or find_golden_prefix(directory)

    def table(stem: str) -> pd.DataFrame:
        frame = pd.read_csv(_require_file(directory, f"{prefix}_{stem}"), index_col=0)
        frame.index = frame.index.astype(str)
        return frame

    pooled = table("metaAnalysis_pooledResults")
    effect_sizes = table("metaAnalysis_datasetEffectSizes")
    standard_errors = table("metaAnalysis_datasetEffectSizeStandardErrors")

    # R's column order is the order of originalData, so it defines the
    # dataset order for the whole golden set.
    dataset_names = [str(c) for c in effect_sizes.columns]
    original_data = {name: _load_dataset(directory, prefix, name) for name in dataset_names}

    leave_one_out = {}
    for name in dataset_names:
        path = _find_file(directory, f"{prefix}_leaveOneOutAnalysis_removed_{name}_pooledResults")
        if path is not None:
            frame = pd.read_csv(path, index_col=0)
            frame.index = frame.index.astype(str)
            leave_one_out[name] = frame

    return RGoldenMetaObject(
        name=directory.name,
        prefix=prefix,
        dataset_names=dataset_names,
        original_data=original_data,
        pooled_results=pooled,
        dataset_effect_sizes=effect_sizes,
        dataset_standard_errors=standard_errors,
        leave_one_out=leave_one_out,
        filter_results=_load_filter_results(directory, prefix),
    )


# ============================================================================
# Comparison
# ============================================================================


def compare_numeric_frames(
    python: pd.DataFrame,
    r: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> FrameParityReport:
    """Compare two numeric tables row by row and column by column.

    Rows are matched by index label, so row order does not matter.
    Two missing values count as equal. One missing value counts as a
    mismatch. Present values must satisfy
    ``|py - r| <= atol + rtol * |r|``, the same rule as ``numpy.isclose``.

    Args:
        python: Table produced by Python.
        r: Golden table produced by R.
        columns: Columns to compare. Defaults to every column in *r*.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        A report. Use ``report.ok`` and ``report.summary()``.
    """
    columns = list(r.columns if columns is None else columns)
    py_index = python.index.astype(str)
    r_index = r.index.astype(str)
    python = python.set_axis(py_index, axis=0)
    r = r.set_axis(r_index, axis=0)

    shared_rows = r_index.intersection(py_index)
    missing_columns = [c for c in columns if c not in python.columns]

    results: List[ColumnParity] = []
    for column in columns:
        if column in missing_columns:
            continue
        py_vals = pd.to_numeric(python.loc[shared_rows, column], errors="coerce").to_numpy(float)
        r_vals = pd.to_numeric(r.loc[shared_rows, column], errors="coerce").to_numpy(float)

        py_na, r_na = np.isnan(py_vals), np.isnan(r_vals)
        missing_mismatch = py_na != r_na
        both = ~py_na & ~r_na
        diff = np.where(both, np.abs(py_vals - r_vals), 0.0)
        out_of_tol = both & ~np.isclose(py_vals, r_vals, rtol=rtol, atol=atol)

        worst = int(np.argmax(diff)) if diff.size else None
        results.append(
            ColumnParity(
                column=column,
                max_abs_diff=float(diff.max()) if diff.size else 0.0,
                n_mismatch=int(out_of_tol.sum() + missing_mismatch.sum()),
                n_missing_mismatch=int(missing_mismatch.sum()),
                worst_row=str(shared_rows[worst]) if worst is not None else None,
            )
        )

    return FrameParityReport(
        only_in_python=sorted(set(py_index) - set(r_index)),
        only_in_r=sorted(set(r_index) - set(py_index)),
        missing_columns=missing_columns,
        columns=results,
        rtol=rtol,
        atol=atol,
    )
