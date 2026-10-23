"""Measure how large the known differences from R are (R parity register).

The register in ``docs/comparison/r_parity_register.md`` lists each place
where this library knowingly differs from R MetaIntegrator. This script
puts numbers on three of them, so a decision about each one rests on
data and not on a guess. It is a tool, not a test: pytest does not
collect it, and it prints a report.

Commands (run from the repository root)::

    python tests/parity/measurements/measure_register_effects.py fragility
    python tests/parity/measurements/measure_register_effects.py unused-shift
    python tests/parity/measurements/measure_register_effects.py signature-cells
    python tests/parity/measurements/measure_register_effects.py signature-cells \\
        --expr raw_matrix.csv --keys probe_to_gene.csv --genes IFI27,CD3E

``unused-shift`` (weaknesses page, W-02) counts how often R's log-scale
decision would flip if the shift that ``.GEM_log_check()`` computes for
negative values were really used. R computes the shift and then ignores it.

``fragility`` (REG-002) draws raw-intensity matrices and counts how often
R's own log-scale rule wrongly says "already log scale". Raw intensities
that R leaves untransformed are the risk.

``signature-cells`` (REG-001 and REG-003) works on a raw (not yet
log2-transformed) matrix and needs no R. The R side is modeled from R's
source: R turns a zero into ``-Inf`` on ``log2``, averages a gene's probes
with ``colMeans()`` (no ``na.rm``), then sets values at or below zero, and
missing values, to 1. It reports two counts:

* ``cells_differ``: gene cells where this library differs from that model.
  Since 0.12.0 this should be 0. A number above 0 means parity was lost.
* ``cells_changed_by_matching_r``: gene cells whose value changed when the
  library began to match R, because the old code skipped missing probe
  values in the average. This is the size of the old difference, and so
  of the change for runs made before 0.12.0.

With ``--expr`` and ``--keys`` it reads your own data:

* ``--expr``: raw values, probes in rows, samples in columns, the first
  column holding the probe ids (as downloaded from GEO).
* ``--keys``: two columns, ``probe`` and ``gene``. A probe that maps to
  several genes lists them separated by commas.
* ``--genes``: comma-separated gene symbols. Default: every mapped gene.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Sequence

import numpy as np
import pandas as pd

_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
sys.path.insert(0, _SRC)

from preprocessing.utils.log_scale import (  # noqa: E402
    LogScaleReport,
    apply_r_log2,
    gem_log_check,
    legacy_median_rule_log2,
)


def log_check_fragility(
    meanlog: float, sdlog: float, n_values: int, draws: int, seed: int
) -> dict[str, float]:
    """Count how often R's rule misjudges raw lognormal intensities.

    Args:
        meanlog: Mean of the log of the simulated intensities.
        sdlog: Standard deviation of the log of the intensities.
        n_values: Probes per simulated sample.
        draws: Number of simulated matrices.
        seed: Seed of the random generator, so the run can be repeated.

    Returns:
        ``median`` (the typical intensity), ``r_keeps_raw`` (draws that
        R's rule wrongly calls "already on a log scale") and
        ``median_rule_transforms`` (draws the former "median above 20"
        rule would transform). Counts, out of ``draws``.
    """
    rng = np.random.default_rng(seed)
    kept = 0
    legacy = 0
    for _ in range(draws):
        matrix = pd.DataFrame(
            rng.lognormal(meanlog, sdlog, size=(n_values, 3)),
            columns=["S1", "S2", "S3"],
        )
        kept += int(gem_log_check(matrix).is_log_scale)
        legacy += int(legacy_median_rule_log2(matrix))
    return {
        "median": float(np.exp(meanlog)),
        "r_keeps_raw": kept,
        "median_rule_transforms": legacy,
    }


def shift_negative_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    """Shift every column with a negative minimum up so its minimum is 1.

    This is the shift that R's ``.GEM_log_check()`` computes for the
    reference column (``x - min + 1``) and then does not use.
    """
    shifted = matrix.copy()
    for column in shifted.columns:
        low = shifted[column].min()
        if low < 0:
            shifted[column] = shifted[column] - low + 1
    return shifted


def unused_shift_flips(make_matrix, draws: int, seed: int) -> dict[str, int]:
    """Count decisions that change when R's unused shift is applied.

    Args:
        make_matrix: Called with a numpy generator, returns a matrix.
        draws: Number of simulated matrices.
        seed: Seed of the random generator.

    Returns:
        ``negative`` (draws whose first column has a negative value) and
        ``flips`` (draws where the log-scale decision differs between the
        matrix as it is, which is what R uses, and the shifted matrix,
        which is what R's comment says it intends).
    """
    rng = np.random.default_rng(seed)
    negative = flips = 0
    for _ in range(draws):
        matrix = make_matrix(rng)
        negative += int(matrix.iloc[:, 0].min() < 0)
        as_is = gem_log_check(matrix).is_log_scale
        shifted = gem_log_check(shift_negative_columns(matrix)).is_log_scale
        flips += int(as_is != shifted)
    return {"negative": negative, "flips": flips}


def r_style_matrix(raw: pd.DataFrame, report: LogScaleReport) -> pd.DataFrame:
    """The matrix R holds after ``getGEOData()``: zeros become ``-Inf``.

    Args:
        raw: The matrix as downloaded.
        report: The library's report for ``raw``, so both sides make the
            same log2 decision.

    Returns:
        ``raw`` itself when no log2 was applied, otherwise R's result.
    """
    if not report.log2_applied:
        return raw
    values = raw.to_numpy(dtype=float)
    low = np.nanmin(values)
    if low < 0:
        values = values - low + 1
    with np.errstate(divide="ignore"):
        return pd.DataFrame(np.log2(values), index=raw.index, columns=raw.columns)


def r_style_gene_level(
    expr: pd.DataFrame, keys: Sequence[str], genes: Sequence[str]
) -> pd.DataFrame:
    """Model R's ``getSampleLevelGeneData()`` on a probe matrix.

    Probes are averaged per gene with no skipping of missing values
    (R's ``colMeans()`` default). Then values at or below zero, and
    missing values, become 1.

    Args:
        expr: Probes x samples.
        keys: One entry per probe. Several genes are comma-separated.
        genes: Genes to return, in this order.

    Returns:
        Genes x samples. A gene with no probe is a row of 1, as in R.
    """
    members: dict[str, list[int]] = {}
    for row, key in enumerate(keys):
        for gene in str(key).split(","):
            members.setdefault(gene.strip(), []).append(row)
    values = expr.to_numpy(dtype=float)
    out = np.ones((len(genes), expr.shape[1]))
    for i, gene in enumerate(genes):
        rows = members.get(gene)
        if rows:
            mean = values[rows].mean(axis=0)  # numpy keeps NaN and -Inf
            mean = np.where(mean <= 0, 1.0, mean)
            out[i] = np.where(np.isnan(mean), 1.0, mean)
    return pd.DataFrame(out, index=list(genes), columns=expr.columns)


def legacy_gene_level(
    expr: pd.DataFrame, keys: Sequence[str], genes: Sequence[str]
) -> pd.DataFrame:
    """The gene-level values the library gave before 0.12.0 (REG-003).

    Same as :func:`r_style_gene_level`, except that missing probe values
    are skipped in the average.
    """
    members: dict[str, list[int]] = {}
    for row, key in enumerate(keys):
        for gene in str(key).split(","):
            members.setdefault(gene.strip(), []).append(row)
    values = expr.to_numpy(dtype=float)
    out = np.ones((len(genes), expr.shape[1]))
    for i, gene in enumerate(genes):
        rows = members.get(gene)
        if rows:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-missing column
                mean = np.nanmean(values[rows], axis=0)
            mean = np.where(mean <= 0, 1.0, mean)
            out[i] = np.where(np.isnan(mean), 1.0, mean)
    return pd.DataFrame(out, index=list(genes), columns=expr.columns)


def signature_cells(raw: pd.DataFrame, keys: Sequence[str], genes: Sequence[str]) -> dict[str, int]:
    """Count gene-level cells that differ from R on a raw matrix.

    Args:
        raw: Raw probe values, probes x samples.
        keys: One entry per probe.
        genes: The signature genes to compare.

    Returns:
        ``cells`` (genes x samples), ``log2_applied`` (1 or 0),
        ``zero_values`` (probe values of exactly 0 that R turns into
        ``-Inf``), ``raw_missing_values``, ``cells_differ`` (library
        against the R model, should be 0) and
        ``cells_changed_by_matching_r`` (library against the pre-0.12.0
        rule).
    """
    from metaintegrator.coconut_integration import get_sample_level_gene_data

    ours_matrix, report = apply_r_log2(raw)
    theirs = r_style_gene_level(r_style_matrix(raw, report), keys, genes)
    ours = get_sample_level_gene_data({"expr": ours_matrix, "keys": list(keys)}, list(genes))
    legacy = legacy_gene_level(ours_matrix, keys, genes)
    present = [g for g in genes if g in ours.index]
    a = ours.loc[present].to_numpy(dtype=float)
    b = theirs.loc[present].to_numpy(dtype=float)
    c = legacy.loc[present].to_numpy(dtype=float)
    differ = ~np.isclose(a, b, rtol=1e-9, atol=0.0)
    changed = ~np.isclose(a, c, rtol=1e-9, atol=0.0)
    return {
        "cells": int(a.size),
        "log2_applied": int(report.log2_applied),
        "zero_values": int(report.zero_values_set_missing),
        "raw_missing_values": int(raw.isna().to_numpy().sum()),
        "cells_differ": int(differ.sum()),
        "cells_changed_by_matching_r": int(changed.sum()),
    }


def _synthetic_raw(
    n_genes: int, probes_per_gene: int, n_samples: int, zero_fraction: float, seed: int
) -> tuple[pd.DataFrame, list[str]]:
    """Raw lognormal intensities, some set to exactly 0."""
    rng = np.random.default_rng(seed)
    n_probes = n_genes * probes_per_gene
    values = rng.lognormal(8.0, 1.5, size=(n_probes, n_samples))
    values[rng.random(values.shape) < zero_fraction] = 0.0
    keys = [f"G{i}" for i in range(n_genes) for _ in range(probes_per_gene)]
    frame = pd.DataFrame(
        values,
        index=[f"p{i}" for i in range(n_probes)],
        columns=[f"S{j}" for j in range(n_samples)],
    )
    return frame, keys


def _cmd_fragility(args: argparse.Namespace) -> None:
    print(f"Raw intensities, {args.draws} draws per row, seed {args.seed}.")
    print("'R keeps raw' counts draws that R's rule wrongly calls log scale.\n")
    print(
        f"{'meanlog':>8} {'sdlog':>6} {'values':>7} {'median':>8} "
        f"{'R keeps raw':>12} {'median rule transforms':>23}"
    )
    for meanlog, sdlog, n_values in (
        (7.0, 1.0, 500),
        (8.0, 1.5, 500),
        (8.0, 1.5, 20000),
        (10.0, 1.2, 20000),
    ):
        r = log_check_fragility(meanlog, sdlog, n_values, args.draws, args.seed)
        print(
            f"{meanlog:>8} {sdlog:>6} {n_values:>7} {r['median']:>8.0f} "
            f"{r['r_keeps_raw']:>9}/{args.draws} "
            f"{r['median_rule_transforms']:>20}/{args.draws}"
        )


def _print_unused_shift_golden() -> None:
    """Compare the score as R computes it with the score if shifted.

    Only the golden inputs whose reference column has a negative value
    are listed. The first score column is what R reports (the port
    matches it exactly in ``test_log_check_parity.py``).
    """
    folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "log_check", "inputs"
    )
    print("Golden inputs with a negative value in the first column")
    print(f"{'case':<32} {'score as R':>11} {'if shifted':>11} {'same decision':>14}")
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".csv"):
            continue
        matrix = pd.read_csv(os.path.join(folder, name), index_col=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            as_is = gem_log_check(matrix)
            shifted = gem_log_check(shift_negative_columns(matrix))
        reference = as_is.reference_column
        if reference is None or matrix[reference].min() >= 0:
            continue
        same = as_is.is_log_scale == shifted.is_log_scale
        print(
            f"{name[:-4]:<32} {as_is.score:>11.4f} {shifted.score:>11.4f} "
            f"{'yes' if same else 'no':>14}"
        )
    print()


def _cmd_unused_shift(args: argparse.Namespace) -> None:
    cols = ["S1", "S2", "S3"]

    def lognormal_minus(meanlog: float, sdlog: float, offset: float):
        return lambda rng: pd.DataFrame(
            rng.lognormal(meanlog, sdlog, size=(500, 3)) - offset, columns=cols
        )

    scenarios = (
        ("raw lognormal(7, 1.2) minus 300", lognormal_minus(7.0, 1.2, 300)),
        ("raw lognormal(8, 1.5) minus 500", lognormal_minus(8.0, 1.5, 500)),
        ("raw lognormal(5, 1.5) minus 100", lognormal_minus(5.0, 1.5, 100)),
        (
            "log scale normal(8, 2) minus 6",
            lambda rng: pd.DataFrame(rng.normal(8, 2, (500, 3)) - 6, columns=cols),
        ),
        (
            "log ratios normal(0, 1)",
            lambda rng: pd.DataFrame(rng.normal(0, 1, (500, 3)), columns=cols),
        ),
    )
    _print_unused_shift_golden()
    print(f"{args.draws} draws per row, seed {args.seed}.\n")
    print(f"{'scenario':<36} {'negative':>9} {'decision flips':>15}")
    for name, make in scenarios:
        r = unused_shift_flips(make, args.draws, args.seed)
        print(f"{name:<36} {r['negative']:>9} {r['flips']:>15}")


def _cmd_signature_cells(args: argparse.Namespace) -> None:
    if args.expr:
        raw = pd.read_csv(args.expr, index_col=0)
        mapping = pd.read_csv(args.keys).set_index("probe")["gene"]
        keys = [str(mapping.get(probe, "NA")) for probe in raw.index]
        if args.genes:
            genes = [g.strip() for g in args.genes.split(",")]
        else:
            genes = sorted({g for k in keys for g in k.split(",") if g not in ("", "NA")})
        print(f"{raw.shape[0]} probes x {raw.shape[1]} samples, {len(genes)} genes.")
        _print_cells(signature_cells(raw, keys, genes))
        return
    print("Synthetic raw lognormal(8, 1.5) data, 3 probes per gene, 200 genes, " "100 samples.\n")
    print(
        f"{'zero fraction':>14} {'zero values':>12} {'cells':>7} {'differ from R':>14} "
        f"{'changed':>8} {'share':>7}"
    )
    for fraction in (0.0, 0.0001, 0.001, 0.01):
        raw, keys = _synthetic_raw(200, 3, 100, fraction, args.seed)
        genes = sorted(set(keys))
        r = signature_cells(raw, keys, genes)
        print(
            f"{fraction:>14} {r['zero_values']:>12} {r['cells']:>7} "
            f"{r['cells_differ']:>14} {r['cells_changed_by_matching_r']:>8} "
            f"{r['cells_changed_by_matching_r'] / r['cells']:>7.2%}"
        )


def _print_cells(result: dict[str, int]) -> None:
    for name, value in result.items():
        print(f"{name:>20}: {value}")
    share = result["cells_changed_by_matching_r"] / max(result["cells"], 1)
    print(f"{'share changed':>20}: {share:.2%}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run one measurement command."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    frag = sub.add_parser("fragility", help="REG-002")
    frag.add_argument("--draws", type=int, default=40)
    frag.add_argument("--seed", type=int, default=20261020)
    frag.set_defaults(run=_cmd_fragility)
    shift = sub.add_parser("unused-shift", help="W-02")
    shift.add_argument("--draws", type=int, default=200)
    shift.add_argument("--seed", type=int, default=20261022)
    shift.set_defaults(run=_cmd_unused_shift)
    cells = sub.add_parser("signature-cells", help="REG-001 and REG-003")
    cells.add_argument("--expr")
    cells.add_argument("--keys")
    cells.add_argument("--genes")
    cells.add_argument("--seed", type=int, default=20261020)
    cells.set_defaults(run=_cmd_signature_cells)
    args = parser.parse_args(argv)
    if args.command == "signature-cells" and bool(args.expr) != bool(args.keys):
        parser.error("--expr and --keys go together")
    args.run(args)


if __name__ == "__main__":
    main()
