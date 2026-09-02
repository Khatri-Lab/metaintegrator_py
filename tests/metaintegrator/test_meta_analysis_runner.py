"""
Unit tests for the meta_analysis_runner.py CLI pipeline.

Covers:
- _filter_genes() normalizes MetaAnalyzer.filter_genes()'s FilterResult
  dataclass return into a plain dict before handing it to the rest of
  the pipeline. Regression test for the reported crash:
      AttributeError: 'FilterResult' object has no attribute 'get'
  raised from _report_filter_summary() when running
  `python -m src meta_analysis` end to end.
- _report_filter_summary() works against the dict shape _filter_genes()
  now guarantees.
- plotting helpers (_plot_heatmap, _plot_violin, _plot_roc,
  _plot_prc) call the real visualization functions with the argument
  shapes those functions actually expect. Regression tests for three
  more crashes reported from the same end-to-end CLI run:
      EnhancedMetaAnalysisPlotter.heatmap_plot() missing 1 required
      positional argument: 'filter_object'
      violin_plot() got an unexpected keyword argument 'scores'
      ROC/PRC plot failed: string indices must be integers
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup: allow running with `python -m pytest` from project root
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator import meta_analysis_runner
from metaintegrator.analysis import FilterResult, MetaAnalyzer


def _make_filter_result(
    pos=("GENE1", "GENE2"),
    neg=("GENE3",),
    description="filter_genes: |ES|>=0.0, FDR<=0.05, nStudies>=1, hetPval>=0.0, LOO=True",
):
    """Build a FilterResult the same way MetaAnalyzer.filter_genes() does."""
    return FilterResult(
        pos_gene_names=list(pos),
        neg_gene_names=list(neg),
        n_pos=len(pos),
        n_neg=len(neg),
        effect_size_thresh=0.0,
        fdr_thresh=0.05,
        number_studies_thresh=1,
        heterogeneity_pval_thresh=0.0,
        is_leave_one_out=True,
        filter_description=description,
    )


_FILTER_CONFIG = {
    "filter": {
        "effect_size_thresh": 0.0,
        "fdr_thresh": 0.05,
        "number_studies_thresh": 1,
        "heterogeneity_pval_thresh": 0.0,
        "is_leave_one_out": True,
        "presence_threshold_mode": "count",
        "presence_threshold_value": None,
    }
}


class TestFilterGenesNormalization:
    """_filter_genes() (pipeline stage 5) must store a plain dict."""

    def test_filter_results_is_plain_dict_after_stage(self, monkeypatch):
        """A FilterResult dataclass from filter_genes() is converted to a plain dict."""
        fr = _make_filter_result()

        def fake_filter_genes(self, meta_object, **kwargs):
            # Mirrors the real MetaAnalyzer.filter_genes(): stores a
            # FilterResult dataclass instance, not a dict.
            meta_object = dict(meta_object)
            meta_object["filterResults"] = fr
            return meta_object

        monkeypatch.setattr(MetaAnalyzer, "filter_genes", fake_filter_genes)

        meta_object = {"originalData": {"ds1": {}}}
        result = meta_analysis_runner._filter_genes(meta_object, _FILTER_CONFIG)

        assert isinstance(result["filterResults"], dict)
        assert not isinstance(result["filterResults"], FilterResult)
        assert result["filterResults"]["posGeneNames"] == ["GENE1", "GENE2"]
        assert result["filterResults"]["negGeneNames"] == ["GENE3"]

    def test_dict_methods_used_by_later_stages_work_on_the_result(self, monkeypatch):
        """Downstream stages call .get()/.copy() and check isinstance(x, dict)
        (ScoreCalculator/GeneSearchOptimizer validation) on filterResults --
        confirm the normalized value actually supports all of that, not just
        a duck-typed lookalike."""
        fr = _make_filter_result()

        def fake_filter_genes(self, meta_object, **kwargs):
            meta_object = dict(meta_object)
            meta_object["filterResults"] = fr
            return meta_object

        monkeypatch.setattr(MetaAnalyzer, "filter_genes", fake_filter_genes)

        meta_object = meta_analysis_runner._filter_genes(
            {"originalData": {"ds1": {}}}, _FILTER_CONFIG
        )
        filter_results = meta_object["filterResults"]

        assert isinstance(filter_results, dict)  # required by isinstance() gates
        assert filter_results.get("posGeneNames") == ["GENE1", "GENE2"]
        assert filter_results.get("missingKey", "default") == "default"
        copied = filter_results.copy()
        copied["posGeneNames"] = ["GENE1"]
        assert filter_results["posGeneNames"] == ["GENE1", "GENE2"]  # copy() didn't mutate original

    def test_already_a_dict_passes_through_unchanged(self, monkeypatch):
        """If filterResults is already a dict (e.g. a future MetaAnalyzer
        change, or a call path that pre-converts it), _filter_genes() must
        not touch it."""
        plain = {"posGeneNames": ["A"], "negGeneNames": []}

        def fake_filter_genes(self, meta_object, **kwargs):
            meta_object = dict(meta_object)
            meta_object["filterResults"] = plain
            return meta_object

        monkeypatch.setattr(MetaAnalyzer, "filter_genes", fake_filter_genes)

        result = meta_analysis_runner._filter_genes({"originalData": {"ds1": {}}}, _FILTER_CONFIG)

        assert result["filterResults"] is plain


class TestReportFilterSummary:
    """Regression coverage for the reported AttributeError."""

    def test_does_not_raise_on_dict_shaped_filter_results(self):
        """_report_filter_summary() works against a dict-shaped filterResults."""
        meta_object = {
            "filterResults": {
                "posGeneNames": ["GENE1", "GENE2"],
                "negGeneNames": ["GENE3"],
            }
        }
        # Previously raised: AttributeError: 'FilterResult' object has no
        # attribute 'get', because filterResults was left as a raw
        # FilterResult dataclass instance.
        meta_analysis_runner._report_filter_summary(meta_object)

    def test_handles_missing_filter_results(self):
        """A None filterResults value is handled without raising."""
        meta_analysis_runner._report_filter_summary({"filterResults": None})

    def test_end_to_end_stage_then_report_does_not_raise(self, monkeypatch):
        """Full regression: run the actual stage 5 -> stage 5 reporting
        sequence the CLI performs, exactly as run_pipeline() does."""
        fr = _make_filter_result()

        def fake_filter_genes(self, meta_object, **kwargs):
            meta_object = dict(meta_object)
            meta_object["filterResults"] = fr
            return meta_object

        monkeypatch.setattr(MetaAnalyzer, "filter_genes", fake_filter_genes)

        meta_object = meta_analysis_runner._filter_genes(
            {"originalData": {"ds1": {}}}, _FILTER_CONFIG
        )
        meta_analysis_runner._report_filter_summary(meta_object)


# ===========================================================================
# Stage 8: plotting helpers
# ===========================================================================


def _make_plot_meta_object(n_genes=6, n_samples=8):
    """Build a meta_object shaped like a real post-pipeline run: two
    datasets with expr/keys/class/pheno/score, a filterResults dict (the
    normalized shape _filter_genes() now guarantees), and a metaAnalysis
    block with the effect-size tables the heatmap needs."""
    rng = np.random.RandomState(0)
    genes = [f"Gene{i}" for i in range(n_genes)]
    pos_genes = genes[: n_genes // 2]
    neg_genes = genes[n_genes // 2 :]

    fr = {"posGeneNames": pos_genes, "negGeneNames": neg_genes}

    def make_dataset(name, seed):
        r = np.random.RandomState(seed)
        classes = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
        samples = [f"{name}_S{i}" for i in range(n_samples)]
        expr = pd.DataFrame(r.normal(size=(n_genes, n_samples)), index=genes, columns=samples)
        for g in pos_genes:
            expr.loc[g] += classes * 1.5
        for g in neg_genes:
            expr.loc[g] += (1 - classes) * 1.5
        pos_idx = [genes.index(g) for g in pos_genes]
        neg_idx = [genes.index(g) for g in neg_genes]
        score = expr.iloc[pos_idx].mean(axis=0).values - expr.iloc[neg_idx].mean(axis=0).values
        pheno = pd.DataFrame({"label": np.where(classes == 1, "Disease", "Control")}, index=samples)
        return {
            "expr": expr,
            "keys": genes,
            "class": classes,
            "pheno": pheno,
            "score": score,
            "formattedName": name,
        }

    original_data = {
        "GSE_A_GPLxxxx": make_dataset("GSE_A", seed=1),
        "GSE_B_GPLyyyy": make_dataset("GSE_B", seed=2),
    }
    effect_sizes = pd.DataFrame(
        {name: rng.uniform(-1, 1, n_genes) for name in original_data}, index=genes
    )
    pooled_results = pd.DataFrame({"effectSize": rng.uniform(-1, 1, n_genes)}, index=genes)

    return {
        "originalData": original_data,
        "filterResults": fr,
        "metaAnalysis": {"datasetEffectSizes": effect_sizes, "pooledResults": pooled_results},
    }


class TestPlotStages:
    """Regression tests for the stage-8 crashes reported from the CLI run."""

    def test_plot_heatmap_does_not_raise(self, tmp_path):
        """_plot_heatmap() runs end-to-end and writes the heatmap PNG."""
        meta_object = _make_plot_meta_object()
        # Previously raised: TypeError: heatmap_plot() missing 1 required
        # positional argument: 'filter_object'.
        meta_analysis_runner._plot_heatmap(meta_object, {"output": {}}, tmp_path, "png", 100)
        assert (tmp_path / "heatmap_effect_sizes.png").exists()

    def test_plot_heatmap_skips_gracefully_without_filter_results(self, tmp_path):
        """With filterResults=None, the heatmap step is skipped rather than raising."""
        meta_object = _make_plot_meta_object()
        meta_object["filterResults"] = None
        meta_analysis_runner._plot_heatmap(meta_object, {"output": {}}, tmp_path, "png", 100)
        assert not (tmp_path / "heatmap_effect_sizes.png").exists()

    def test_plot_violin_does_not_raise_and_saves_all_datasets(self, tmp_path):
        """_plot_violin() runs end-to-end and writes one PNG per dataset."""
        meta_object = _make_plot_meta_object()
        # Previously raised: TypeError: violin_plot() got an unexpected
        # keyword argument 'scores'.
        meta_analysis_runner._plot_violin(meta_object, {"output": {}}, tmp_path, "png", 100)
        saved = sorted(p.name for p in tmp_path.glob("violin_*.png"))
        assert saved == ["violin_GSE_A_GPLxxxx.png", "violin_GSE_B_GPLyyyy.png"]

    def test_plot_roc_does_not_raise(self, tmp_path, caplog):
        """_plot_roc() runs end-to-end and writes the combined ROC plot."""
        meta_object = _make_plot_meta_object()
        # Previously logged: "ROC plot failed: string indices must be
        # integers" (multiple_roc_plot() was given a {name: dataset} dict
        # instead of a list of dataset dicts).
        meta_analysis_runner._plot_roc(meta_object, {"output": {}}, tmp_path, "png", 100)
        assert "ROC plot failed" not in caplog.text
        assert (tmp_path / "roc_multiple.png").exists()

    def test_plot_prc_does_not_raise(self, tmp_path, caplog):
        """_plot_prc() runs end-to-end and writes the combined PRC plot."""
        meta_object = _make_plot_meta_object()
        meta_analysis_runner._plot_prc(meta_object, {"output": {}}, tmp_path, "png", 100)
        assert "PRC plot failed" not in caplog.text
        assert (tmp_path / "prc_multiple.png").exists()


# ===========================================================================
# Stage 7: _compute_scores_and_roc()
# ===========================================================================


class TestComputeScoresAndRoc:
    """Regression coverage for the reported crash (round-13 console log,
    4 occurrences, one per dataset):

        'ROCAnalysisEngine' object has no attribute 'calculate_roc_auc'

    `ROCAnalysisEngine` never had a `calculate_roc_auc` method -- it
    exposes `calculate_roc_curve`/`calculate_roc`, each returning a dict
    that includes an "auc" key, and each taking `(y_true, y_scores)` in
    that order. The old call site both used the wrong method name and
    passed the arguments in the wrong order (`score` then `class`, not
    `class` then `score`). Every dataset's AUC computation raised
    AttributeError, was swallowed by `_compute_scores_and_roc`'s own
    `except Exception`, and surfaced only as a "ROC failed for <dataset>"
    warning -- silently leaving `roc_auc_per_dataset.csv` unwritten.
    """

    def test_does_not_raise_and_computes_auc_for_every_dataset(self, caplog):
        """Every dataset gets a valid AUC in [0, 1] with no "ROC failed" warning."""
        meta_object = _make_plot_meta_object()
        config = {"output": {"plots": ["roc", "prc"], "roc_datasets": None}}

        result = meta_analysis_runner._compute_scores_and_roc(meta_object, config)

        assert "ROC failed" not in caplog.text
        roc_results = result["_roc_results"]
        assert set(roc_results.keys()) == set(meta_object["originalData"].keys())
        for auc_value in roc_results.values():
            assert isinstance(auc_value, float)
            assert 0.0 <= auc_value <= 1.0

    def test_scores_are_populated_on_every_dataset(self):
        """Every dataset gets a score array aligned with its class labels."""
        meta_object = _make_plot_meta_object()
        config = {"output": {"plots": ["roc"], "roc_datasets": None}}

        result = meta_analysis_runner._compute_scores_and_roc(meta_object, config)

        for ds_obj in result["originalData"].values():
            assert ds_obj.get("score") is not None
            assert len(ds_obj["score"]) == len(ds_obj["class"])

    def test_respects_roc_datasets_allowlist(self, caplog):
        """Only the datasets named in roc_datasets get ROC results."""
        meta_object = _make_plot_meta_object()
        only_dataset = next(iter(meta_object["originalData"]))
        config = {"output": {"plots": ["roc"], "roc_datasets": [only_dataset]}}

        result = meta_analysis_runner._compute_scores_and_roc(meta_object, config)

        assert "ROC failed" not in caplog.text
        assert set(result["_roc_results"].keys()) == {only_dataset}

    def test_skips_roc_computation_when_not_requested(self):
        """With neither "roc" nor "prc" requested, no _roc_results key is added."""
        meta_object = _make_plot_meta_object()
        config = {"output": {"plots": ["violin"]}}  # neither "roc" nor "prc"

        result = meta_analysis_runner._compute_scores_and_roc(meta_object, config)

        assert "_roc_results" not in result
        # Scores are still computed regardless of which plots are requested.
        for ds_obj in result["originalData"].values():
            assert ds_obj.get("score") is not None
