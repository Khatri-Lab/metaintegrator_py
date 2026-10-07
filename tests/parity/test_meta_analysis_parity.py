"""R parity tests for the meta-analysis and gene filter.

Each test takes R's inputs from a golden set, runs metaintegrator_py on
them, and compares the result with R's frozen output.

These tests start after data loading. Every golden set gives both tools the
exact same expression matrices, gene keys and class labels. So any failure
here points at the statistics, not at GEO downloads or annotation.

Run only these tests with::

    pytest -m parity
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import pandas as pd
import pytest

from utils.r_golden import RFilterResult, compare_numeric_frames

from .parity_support import ATOL, BUILTIN_GOLDEN_DIRS, RTOL, ParityRun

pytestmark = pytest.mark.parity


# ============================================================================
# Reference ports of R's filter rules
# ============================================================================


def r_filter_meta_run(
    pooled: pd.DataFrame,
    effect_size_thresh: float,
    fdr_thresh: float,
    number_studies_thresh: int,
    heterogeneity_pval_thresh: float,
) -> Tuple[List[str], List[str]]:
    """Line-by-line port of R's ``.filterMetaRun()`` (``metaAnalysis.R``).

    Note that R's ``which()`` drops rows where the comparison is NA.
    Pandas comparisons with NaN are False, which has the same effect.

    Args:
        pooled: A ``pooledResults`` table.
        effect_size_thresh: Minimum absolute effect size.
        fdr_thresh: Maximum effect-size FDR and Fisher FDR.
        number_studies_thresh: Minimum number of studies.
        heterogeneity_pval_thresh: Minimum heterogeneity p-value (0 = off).

    Returns:
        ``(pos_gene_names, neg_gene_names)`` in table order.
    """
    res = pooled[pooled["effectSize"].abs() >= effect_size_thresh]
    res = res[res["effectSizeFDR"] <= fdr_thresh]
    res = res[res["numStudies"] >= number_studies_thresh]
    if heterogeneity_pval_thresh > 0:
        res = res[res["heterogeneityPval"] >= heterogeneity_pval_thresh]
    pos = res[(res["fisherFDRUp"] <= fdr_thresh) & (res["effectSize"] > 0)]
    neg = res[(res["fisherFDRDown"] <= fdr_thresh) & (res["effectSize"] < 0)]
    return pos.index.tolist(), neg.index.tolist()


def r_filter_leave_one_out(
    loo_pooled: List[pd.DataFrame], flt: RFilterResult
) -> Tuple[List[str], List[str]]:
    """Port of R's ``filterGenes(isLeaveOneOut = TRUE)`` (``filterGenes.R``).

    R does not filter the main meta-analysis in this mode. It runs
    ``.filterMetaRun()`` on every leave-one-out result, using
    ``numberStudiesThresh - 1``, and keeps the genes that pass in all of
    them. The order comes from the first leave-one-out result.

    Args:
        loo_pooled: Leave-one-out ``pooledResults`` tables, in dataset order.
        flt: The filter thresholds.

    Returns:
        ``(pos_gene_names, neg_gene_names)``.
    """
    per_loo = [
        r_filter_meta_run(
            pooled,
            flt.effect_size_thresh,
            flt.fdr_thresh,
            flt.number_studies_thresh - 1,
            flt.heterogeneity_pval_thresh,
        )
        for pooled in loo_pooled
    ]
    pos, neg = per_loo[0]
    for other_pos, other_neg in per_loo[1:]:
        pos = [g for g in pos if g in set(other_pos)]
        neg = [g for g in neg if g in set(other_neg)]
    return pos, neg


def _python_filter(analyzer, parity_run: ParityRun, flt: RFilterResult) -> Tuple[List, List]:
    """Run Python's ``filter_genes()`` with the thresholds of an R filter."""
    meta_object = copy.copy(parity_run.meta_object)  # filter_genes adds a key
    meta_object = analyzer.filter_genes(
        meta_object,
        effect_size_thresh=flt.effect_size_thresh,
        fdr_thresh=flt.fdr_thresh,
        number_studies_thresh=flt.number_studies_thresh,
        heterogeneity_pval_thresh=flt.heterogeneity_pval_thresh,
        is_leave_one_out=flt.is_leave_one_out,
    )
    result = meta_object["filterResults"]
    return list(result.pos_gene_names), list(result.neg_gene_names)


def _assert_same_genes(label: str, python: List[str], r: List[str]) -> None:
    """Assert two gene lists hold the same genes, with a readable message."""
    only_py, only_r = sorted(set(python) - set(r)), sorted(set(r) - set(python))
    assert not only_py and not only_r, (
        f"{label}: Python {len(python)} genes vs R {len(r)} genes. "
        f"Only in Python: {only_py[:15]}. Only in R: {only_r[:15]}."
    )


# ============================================================================
# Meta-analysis parity
# ============================================================================


def test_builtin_golden_sets_exist():
    """The built-in golden data must be present.

    Without it, every parity test would be skipped and the suite would
    still look green.
    """
    missing = [str(d) for d in BUILTIN_GOLDEN_DIRS if not d.is_dir()]
    assert not missing, f"Built-in golden data missing: {missing}"


class TestMetaAnalysisParity:
    """Python ``run_meta_analysis()`` must reproduce R ``runMetaAnalysis()``."""

    def test_gene_universe_matches(self, parity_run: ParityRun):
        """Both tools analyze exactly the same genes."""
        py_genes = set(parity_run.meta_analysis["pooledResults"].index.astype(str))
        r_genes = set(parity_run.golden.pooled_results.index)
        _assert_same_genes("pooledResults genes", sorted(py_genes), sorted(r_genes))

    def test_pooled_results_match(self, parity_run: ParityRun):
        """Every pooledResults column matches R within tolerance."""
        report = compare_numeric_frames(
            parity_run.meta_analysis["pooledResults"],
            parity_run.golden.pooled_results,
            rtol=RTOL,
            atol=ATOL,
        )
        assert report.ok, report.summary()

    @pytest.mark.parametrize(
        "py_key, golden_attr",
        [
            ("datasetEffectSizes", "dataset_effect_sizes"),
            ("datasetEffectSizeStandardErrors", "dataset_standard_errors"),
        ],
        ids=["effect_sizes", "standard_errors"],
    )
    def test_per_study_values_match(self, parity_run: ParityRun, py_key: str, golden_attr: str):
        """Per-study effect sizes and standard errors match R.

        Python names its columns ``study_<i>_...`` in dataset order, so the
        columns are renamed to R's dataset names before comparing.
        """
        r_table: pd.DataFrame = getattr(parity_run.golden, golden_attr)
        py_table: pd.DataFrame = parity_run.meta_analysis[py_key]
        assert (
            py_table.shape[1] == r_table.shape[1]
        ), f"{py_key}: Python has {py_table.shape[1]} studies, R has {r_table.shape[1]}"
        py_table = py_table.set_axis(list(r_table.columns), axis=1)
        report = compare_numeric_frames(py_table, r_table, rtol=RTOL, atol=ATOL)
        assert report.ok, report.summary()

    def test_leave_one_out_pooled_results_match(self, parity_run: ParityRun):
        """Each leave-one-out pooledResults matches R's."""
        golden = parity_run.golden
        if not golden.leave_one_out:
            pytest.skip(f"{golden.name}: R did not run leave-one-out")
        failures = []
        for removed, r_pooled in golden.leave_one_out.items():
            py_pooled = parity_run.python_loo_pooled(removed)
            if py_pooled is None:
                failures.append(f"removed_{removed}: no Python result")
                continue
            report = compare_numeric_frames(py_pooled, r_pooled, rtol=RTOL, atol=ATOL)
            if not report.ok:
                failures.append(f"removed_{removed}:\n{report.summary()}")
        assert not failures, "\n".join(failures)


# ============================================================================
# Gene filter parity
# ============================================================================


# Golden sets whose saved filterResults were made by an older R release.
#
# tinyMetaObject ships with R MetaIntegrator. Its filterResults carry a 2015
# timestamp. Its metaAnalysis and leaveOneOutAnalysis tables still match
# today's R math (and Python) to about 1e-12. Its saved gene lists do not
# match today's filterGenes() rule, though. They match only if every
# leave-one-out run also requires numStudies >= 2, while current R uses
# numberStudiesThresh - 1. So we test filters against a port of the current
# R rule (run on R's own tables) and skip the saved lists for these sets.
STALE_FILTER_GOLDENS = frozenset({"tinyMetaObject"})


def _golden_filters(parity_run: ParityRun, leave_one_out: bool) -> List[RFilterResult]:
    """Return trusted golden filters of one kind, or skip the test."""
    golden = parity_run.golden
    if golden.name in STALE_FILTER_GOLDENS:
        pytest.skip(f"{golden.name}: saved filterResults predate the current R rule")
    filters = [f for f in golden.filter_results if f.is_leave_one_out == leave_one_out]
    if not filters:
        kind = "leave-one-out" if leave_one_out else "standard"
        pytest.skip(f"{golden.name}: no {kind} filterResults in golden set")
    return filters


def _threshold_sets(parity_run: ParityRun, leave_one_out: bool) -> List[RFilterResult]:
    """Threshold combinations to replay: two fixed sets plus every saved R filter."""
    base = [
        RFilterResult("default", [], [], 0.0, 0.05, 1, 0.0, leave_one_out),
        RFilterResult("strict", [], [], 0.5, 0.01, 2, 0.05, leave_one_out),
    ]
    saved = [
        RFilterResult(
            f.name,
            [],
            [],
            f.effect_size_thresh,
            f.fdr_thresh,
            f.number_studies_thresh,
            f.heterogeneity_pval_thresh,
            leave_one_out,
        )
        for f in parity_run.golden.filter_results
    ]
    return base + saved


def _r_loo_tables(parity_run: ParityRun) -> List[pd.DataFrame]:
    """R's leave-one-out tables in dataset order, or skip if incomplete."""
    golden = parity_run.golden
    if set(golden.leave_one_out) != set(golden.dataset_names):
        pytest.skip(f"{golden.name}: R leave-one-out results missing or incomplete")
    return [golden.leave_one_out[name] for name in golden.dataset_names]


class TestFilterParity:
    """Python ``filter_genes()`` must reproduce R ``filterGenes()``."""

    def test_standard_filter_matches_r_rule(self, analyzer, parity_run: ParityRun):
        """Without leave-one-out, Python matches R's ``.filterMetaRun()`` rule.

        This runs on every golden set, even ones without a saved R filter.
        It applies a line-by-line port of the R rule to R's own pooled
        results. So it checks Python against R's logic and R's numbers.
        """
        for flt in _threshold_sets(parity_run, leave_one_out=False):
            r_pos, r_neg = r_filter_meta_run(
                parity_run.golden.pooled_results,
                flt.effect_size_thresh,
                flt.fdr_thresh,
                flt.number_studies_thresh,
                flt.heterogeneity_pval_thresh,
            )
            py_pos, py_neg = _python_filter(analyzer, parity_run, flt)
            _assert_same_genes(f"[{flt.name}] posGeneNames", py_pos, r_pos)
            _assert_same_genes(f"[{flt.name}] negGeneNames", py_neg, r_neg)

    def test_standard_filter_matches_r_golden(self, analyzer, parity_run: ParityRun):
        """Python matches every saved R filter that did not use leave-one-out."""
        for flt in _golden_filters(parity_run, leave_one_out=False):
            py_pos, py_neg = _python_filter(analyzer, parity_run, flt)
            _assert_same_genes(f"[{flt.name}] posGeneNames", py_pos, flt.pos_gene_names)
            _assert_same_genes(f"[{flt.name}] negGeneNames", py_neg, flt.neg_gene_names)

    def test_r_leave_one_out_rule_reproduces_r_golden(self, parity_run: ParityRun):
        """Self-check: our port of R's leave-one-out rule matches R's saved genes.

        This test runs no Python analysis code. It proves that
        :func:`r_filter_leave_one_out` describes R correctly, so the
        Python leave-one-out test below can rely on it.
        """
        loo = _r_loo_tables(parity_run)
        for flt in _golden_filters(parity_run, leave_one_out=True):
            pos, neg = r_filter_leave_one_out(loo, flt)
            assert pos == flt.pos_gene_names, f"[{flt.name}] pos: {pos} vs {flt.pos_gene_names}"
            assert neg == flt.neg_gene_names, f"[{flt.name}] neg: {neg} vs {flt.neg_gene_names}"

    def test_leave_one_out_filter_matches_r_rule(self, analyzer, parity_run: ParityRun):
        """With leave-one-out, Python matches R's ``filterGenes()`` rule.

        This was a known gap until ``filter_genes()`` adopted R's rule.
        Before that, Python kept 50 up and 28 down genes on tinyMetaObject
        where R keeps 18 and 13.
        """
        loo = _r_loo_tables(parity_run)
        for flt in _threshold_sets(parity_run, leave_one_out=True):
            r_pos, r_neg = r_filter_leave_one_out(loo, flt)
            py_pos, py_neg = _python_filter(analyzer, parity_run, flt)
            _assert_same_genes(f"[{flt.name}] posGeneNames", py_pos, r_pos)
            _assert_same_genes(f"[{flt.name}] negGeneNames", py_neg, r_neg)
