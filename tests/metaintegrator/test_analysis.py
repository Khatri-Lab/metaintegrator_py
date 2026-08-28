import importlib
import logging
import multiprocessing as mp
import os
import pickle
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# Import the module under test
from metaintegrator.analysis import (
    EffectSizeResult,
    FixedEffectsEngine,
    InverseVarianceEngine,
    MetaAnalysisEngine,
    MetaAnalysisFactory,
    MetaAnalyzer,
    RandomEffectsEngine,
    _run_loo_subset,
    run_meta_analysis,
)

# Module reference — used only by tests that legitimately need to
# patch module-level names (e.g. testing the convenience function).
meta_analysis = importlib.import_module("metaintegrator.analysis")


# ===================================================================
# Helper factories — data builders
# ===================================================================


def make_dummy_effect_df(name, genes, values):
    """
    Creates a tiny effect DataFrame that mimics the structure expected by
    MetaAnalysisCore methods. The index is the gene list; columns are
    ``g`` and ``se_g``.
    """
    return pd.DataFrame(
        {
            "g": values,
            "se_g": [0.1] * len(values),
            "keys": [name] * len(values),
        },
        index=genes,
    )


def make_pooled_estimates(p_vals):
    """Return a DataFrame that mimics the pooled_estimates output."""
    return pd.DataFrame(
        {
            "p_value": p_vals,
            "p_fdr": [v + 0.01 for v in p_vals],
            "summary": [1.0] * len(p_vals),
            "se_summary": [0.5] * len(p_vals),
            "tau2": [0.2] * len(p_vals),
            "n_studies": [3] * len(p_vals),
            "Q": [2.0] * len(p_vals),
            "pval_het": [0.3] * len(p_vals),
        },
        index=[f"gene{i}" for i in range(len(p_vals))],
    )


def make_fisher_output(up_pvals, down_pvals):
    """Return a DataFrame that mimics Fisher output."""
    return pd.DataFrame(
        {
            "F.pval.up": up_pvals,
            "F.pval.down": down_pvals,
            "F.Qval.up": [v + 0.01 for v in up_pvals],
            "F.Qval.down": [v + 0.01 for v in down_pvals],
            "F.stat.up": [10] * len(up_pvals),
            "F.stat.down": [20] * len(down_pvals),
        },
        index=[f"gene{i}" for i in range(len(up_pvals))],
    )


def make_es_result(pooled=None, g=None, se_g=None):
    """Build an EffectSizeResult for testing."""
    return EffectSizeResult(
        g=g if g is not None else pd.DataFrame(),
        se_g=se_g if se_g is not None else pd.DataFrame(),
        pooled_estimates=pooled if pooled is not None else pd.DataFrame(),
    )


# ===================================================================
# Helper factories — mock collaborators
# ===================================================================


def _make_mock_core():
    """Build the standard mock MetaAnalysisCore."""
    return SimpleNamespace(
        calculate_effect_sizes=lambda df: df,
        expand_dataframe=lambda df: df,
        summarize_effects_within_study=lambda df, method: df,
        collapse_probes_to_genes=lambda df, method="fixed_iv": (
            df.set_index("keys")[["g", "se_g"]] if "keys" in df.columns else df
        ),
        pool_inverse_variance=lambda g, se, method: pd.DataFrame(
            {
                "summary": [0.5],
                "se_summary": [0.2],
                "p_value": [0.05],
                "tau2": [0.1],
                "n_studies": [3],
                "Q": [2.3],
                "pval_het": [0.1],
            },
            index=["gene0"],
        ),
    )


def _make_mock_fisher():
    """Build the standard mock FisherCombination."""
    return SimpleNamespace(
        calculate_study_pvalues=lambda ds: pd.DataFrame({"p": [0.01] * len(ds)}),
        sum_of_logs=lambda dfs: pd.DataFrame(
            {
                "F_stat_up": [5],
                "F_pval_up": [0.02],
                "F_stat_down": [6],
                "F_pval_down": [0.03],
            }
        ),
    )


def _make_mock_validator():
    """Build the standard mock DatasetValidator."""
    return SimpleNamespace(
        convert_original_data_names=lambda obj: obj,
        validate_meta_object=lambda obj, _, __: (True, []),
        check_class_vector=lambda ds: True,
    )


def _make_mock_processor():
    """Build the standard mock DataProcessor."""
    return SimpleNamespace(create_combined_annotation_table=lambda ds_list: None)


# ===================================================================
# DI helpers — build fully-injected objects for tests
# ===================================================================
# These replace the old autouse monkeypatch fixture.  Each test now
# declares its dependencies explicitly via constructor injection.


def _make_test_engine(engine_cls=RandomEffectsEngine, **overrides):
    """Build an engine with mock core/fisher via constructor injection.

    Usage::

        engine = _make_test_engine()                          # defaults
        engine = _make_test_engine(FixedEffectsEngine)        # fixed effects
        engine = _make_test_engine(core=custom_core)          # override core
    """
    core = overrides.pop("core", _make_mock_core())
    fisher = overrides.pop("fisher", _make_mock_fisher())
    return engine_cls(core=core, fisher=fisher, **overrides)


def _make_test_analyzer(**overrides):
    """Build a MetaAnalyzer with all dependencies mocked via DI.

    Usage::

        analyzer = _make_test_analyzer()                      # all mocks
        analyzer = _make_test_analyzer(max_cores=1)           # override
        analyzer = _make_test_analyzer(engine=custom_engine)  # custom engine
        analyzer = _make_test_analyzer(validator=failing_val) # custom validator
    """
    engine = overrides.pop("engine", _make_test_engine())
    validator = overrides.pop("validator", _make_mock_validator())
    processor = overrides.pop("processor", _make_mock_processor())
    max_cores = overrides.pop("max_cores", None)
    return MetaAnalyzer(
        engine=engine,
        validator=validator,
        processor=processor,
        max_cores=max_cores,
    )


# ===================================================================
# TESTS — MetaAnalyzer.__init__
# ===================================================================


def test_init_default(monkeypatch):
    """When no max_cores is supplied, it defaults to min(10, cpu_count())."""
    cpu_count = 4
    monkeypatch.setattr(mp, "cpu_count", lambda: cpu_count)
    analyzer = MetaAnalyzer()
    assert analyzer.max_cores == cpu_count


def test_init_with_custom(monkeypatch):
    """Provided max_cores are capped at 10 and the available cores."""
    cpu_count = 3
    monkeypatch.setattr(mp, "cpu_count", lambda: cpu_count)

    analyzer = MetaAnalyzer(max_cores=20)
    assert analyzer.max_cores == 3

    analyzer2 = MetaAnalyzer(max_cores=5)
    assert analyzer2.max_cores == 3

    analyzer3 = MetaAnalyzer(max_cores=2)
    assert analyzer3.max_cores == 2


# ===================================================================
# TESTS — run_meta_analysis_core
# ===================================================================


def test_run_meta_analysis_core_empty():
    """Empty original_data raises ValueError."""
    analyzer = _make_test_analyzer()
    with pytest.raises(ValueError, match="Original data cannot be empty"):
        analyzer.run_meta_analysis_core([])


def test_run_meta_analysis_core_calls(monkeypatch):
    """run_meta_analysis_core calls the three engine pipeline methods."""
    analyzer = _make_test_analyzer()
    called = {"combined": False, "fisher": False, "format": False}

    def fake_combined(data):
        called["combined"] = True
        return make_es_result(pooled=pd.DataFrame())

    def fake_fisher(data, use_old):
        called["fisher"] = True
        return pd.DataFrame()

    def fake_format(combined, fisher):
        called["format"] = True
        return {"dummy": 1}

    monkeypatch.setattr(analyzer._engine, "compute_effect_sizes", fake_combined)
    monkeypatch.setattr(analyzer._engine, "compute_fisher_results", fake_fisher)
    monkeypatch.setattr(analyzer._engine, "format_results", fake_format)

    result = analyzer.run_meta_analysis_core([{"dummy": "data"}])
    assert result == {"dummy": 1}
    assert all(called.values())


# ===================================================================
# TESTS — engine.compute_effect_sizes
# ===================================================================


def test_compute_effect_sizes(monkeypatch):
    """Ensure that FDR p-values are computed and sorting works."""
    engine = _make_test_engine()

    ds1 = make_dummy_effect_df("A", ["gene0"], [1.0])
    ds2 = make_dummy_effect_df("B", ["gene0"], [2.0])

    def fake_combine(effects):
        return make_es_result(
            pooled=make_pooled_estimates([0.05, 0.01, 0.20]),
        )

    monkeypatch.setattr(engine, "combine_effect_sizes", fake_combine)

    result = engine.compute_effect_sizes([ds1, ds2])

    assert "p_fdr" in result.pooled_estimates.columns
    assert "effect_size" in result.pooled_estimates.columns
    assert all(result.pooled_estimates["effect_size"] == result.pooled_estimates["summary"])

    sorted_indices = result.pooled_estimates["p_fdr"].argsort()
    assert list(sorted_indices) == sorted([1, 0, 2])


# ===================================================================
# TESTS — engine.compute_fisher_results
# ===================================================================


def test_compute_fisher_results_new(monkeypatch):
    """When use_old_method=False, the Fisher pipeline runs."""
    engine = _make_test_engine()

    monkeypatch.setattr(
        engine,
        "_adjust_fisher",
        lambda df, method="fdr_bh": df,
    )

    result = engine.compute_fisher_results([{"a": 1}, {"b": 2}], use_old_method=False)
    assert "F_pval_up" in result.columns
    assert "F_pval_down" in result.columns


# ===================================================================
# TESTS — engine.format_results
# ===================================================================


def test_format_final_results():
    """Test that the final result contains all expected keys."""
    engine = _make_test_engine()

    combined = make_es_result(pooled=make_pooled_estimates([0.05]))
    fisher_df = make_fisher_output([0.02], [0.03])

    result = engine.format_results(combined, fisher_df)

    assert isinstance(result, dict)

    expected_keys = [
        "datasetEffectSizes",
        "datasetEffectSizeStandardErrors",
        "pooledResults",
        "pooled_estimates",
        "analysisDescription",
    ]
    for key in expected_keys:
        assert key in result

    assert isinstance(result["pooled_estimates"], pd.DataFrame)
    assert "p_value" in result["pooled_estimates"].columns
    assert "p_fdr" in result["pooled_estimates"].columns
    assert "summary" in result["pooled_estimates"].columns

    pooled_cols = result["pooledResults"].columns
    expected_pooled_cols = [
        "effectSize",
        "effectSizeStandardError",
        "effectSizePval",
        "effectSizeFDR",
        "tauSquared",
        "numStudies",
        "cochranesQ",
        "heterogeneityPval",
        "fisherStatUp",
        "fisherPvalUp",
        "fisherFDRUp",
        "fisherStatDown",
        "fisherPvalDown",
        "fisherFDRDown",
    ]
    for col in expected_pooled_cols:
        assert col in pooled_cols


# ===================================================================
# TESTS — combine_effect_sizes
# ===================================================================


def test_combine_effect_sizes_empty():
    """An empty input list raises a ValueError."""
    engine = _make_test_engine()
    with pytest.raises(ValueError):
        engine.combine_effect_sizes([])


def test_combine_effect_sizes_success():
    """Ensure the method returns an EffectSizeResult with expected fields."""
    engine = _make_test_engine()

    df1 = make_dummy_effect_df("A", ["gene0"], [1.2])
    df2 = make_dummy_effect_df("B", ["gene1"], [2.5])

    out = engine.combine_effect_sizes([df1, df2])

    assert isinstance(out, EffectSizeResult)
    assert isinstance(out.g, pd.DataFrame)
    assert isinstance(out.se_g, pd.DataFrame)
    assert isinstance(out.pooled_estimates, pd.DataFrame)
    assert set(out.keys()) == {"g", "se_g", "pooled_estimates"}
    assert out["pooled_estimates"].shape == (1, 7)


# ===================================================================
# TESTS — _multimerge_effect_sizes (static method — no DI needed)
# ===================================================================


def test_multimerge_effect_sizes():
    """Check that two dataframes with overlapping indices are merged correctly."""
    df1 = pd.DataFrame({"g": [1], "se_g": [0.1], "keys": ["A"]}, index=["gene0"])
    df2 = pd.DataFrame({"g": [2], "se_g": [0.1], "keys": ["B"]}, index=["gene1"])

    study_names = ["study_gene0", "study_gene1"]
    merged = RandomEffectsEngine._multimerge_effect_sizes([df1, df2], study_names)

    assert merged.shape == (2, 6)
    assert merged.loc["gene0", "study_gene0_g"] == 1
    assert merged.loc["gene1", "study_gene1_g"] == 2
    assert np.isnan(merged.loc["gene0", "study_gene1_g"])
    assert np.isnan(merged.loc["gene1", "study_gene0_g"])


# ===================================================================
# TESTS — _adjust_fisher (static method — no DI needed)
# ===================================================================


def test_adjust_fisher():
    """Verify that p-value adjustments are performed correctly."""
    df = pd.DataFrame(
        {"F_pval_up": [0.05, 0.01], "F_pval_down": [0.10, 0.02]},
        index=["gene0", "gene1"],
    )
    out = RandomEffectsEngine._adjust_fisher(df)

    assert "F_pval_up" in out.columns
    assert "F_pval_down" in out.columns


# ===================================================================
# TESTS — run_leave_one_out_analysis
# ===================================================================


def test_run_leave_one_out_error():
    """Less than three datasets triggers a ValueError."""
    analyzer = _make_test_analyzer()
    with pytest.raises(ValueError, match="Need at least 3 datasets"):
        analyzer.run_leave_one_out_analysis([{"x": 1}, {"x": 2}])


def test_run_leave_one_out_sequential(monkeypatch):
    """With max_cores=1 the method runs sequentially."""
    analyzer = _make_test_analyzer(max_cores=1)

    is_use_old_method = False
    monkeypatch.setattr(
        analyzer,
        "run_meta_analysis_core",
        lambda fake_core, use_old_method=is_use_old_method: fake_core,
    )

    datasets = [{"index": 0}, {"index": 1}, {"index": 2}]
    results = analyzer.run_leave_one_out_analysis(datasets, use_old_method=is_use_old_method)

    assert isinstance(results, list)
    assert len(results) == 3
    assert results[0] == [{"index": 1}, {"index": 2}]
    assert results[1] == [{"index": 0}, {"index": 2}]
    assert results[2] == [{"index": 0}, {"index": 1}]


# ===================================================================
# TESTS — run_leave_one_out_analysis: ProcessPoolExecutor pickling
# ===================================================================
# Regression coverage for: "Can't pickle local object
# 'MetaAnalyzer.run_leave_one_out_analysis.<locals>.analyze_subset'"
#
# Root cause: the per-subset callable was a closure nested inside
# run_leave_one_out_analysis(). ProcessPoolExecutor must pickle every
# callable it submits, and pickle can only resolve a function "by name"
# (its module path) — a nested function has none. Every subset raised
# the exception, the method silently returned four None results, and
# filter_genes()'s leave-one-out filter was silently skipped even though
# is_leave_one_out=True was requested. Fix: hoist the per-subset logic to
# a module-level function (_run_loo_subset), which pickle can resolve.
#
# NOTE: the SimpleNamespace mocks used elsewhere in this file
# (_make_mock_core, etc.) hold lambdas, which are also unpicklable —
# fine for the sequential (max_cores=1) test above, but would mask the
# real fix in a parallel test for an unrelated reason. The tests below
# use plain module-level classes instead, so they exercise a genuine
# ProcessPoolExecutor round trip like the original bug.


class _RecordingLooEngine(MetaAnalysisEngine):
    """Minimal MetaAnalysisEngine that records which dataset names it was
    asked to combine, so a caller can assert exactly which dataset was
    excluded from each leave-one-out subset -- without needing genuine
    GEO-shaped statistical input data. Defined at module level (not
    inside a test function) so it -- and any MetaAnalyzer holding it --
    can actually be pickled and sent to a ProcessPoolExecutor worker."""

    def compute_effect_sizes(self, original_data):
        names = tuple(sorted(d["name"] for d in original_data))
        pooled = pd.DataFrame({"_names": [names]})
        empty = pd.DataFrame()
        return EffectSizeResult(g=empty, se_g=empty, pooled_estimates=pooled)

    def compute_fisher_results(self, original_data, use_old_method):
        return pd.DataFrame()

    def format_results(self, combined_results, output_fisher):
        return {"remaining_datasets": combined_results.pooled_estimates["_names"].iloc[0]}

    @property
    def description(self):
        return "recording stub engine for LOO pickling tests"


class _NoOpLooProcessor:
    """Stand-in annotation processor that skips combined-annotation-table creation entirely."""

    @staticmethod
    def create_combined_annotation_table(original_data):
        return None


def _make_picklable_loo_analyzer(max_cores):
    """Like _make_test_analyzer(), but with plain picklable module-level
    collaborators instead of SimpleNamespace/lambda mocks -- required for
    the real-ProcessPoolExecutor tests below."""
    return MetaAnalyzer(
        max_cores=max_cores,
        engine=_RecordingLooEngine(),
        processor=_NoOpLooProcessor,
    )


def test_run_loo_subset_is_picklable():
    """Direct regression test for the root cause: pickle must be able to
    resolve the callable ProcessPoolExecutor submits, by qualified name."""
    dumped = pickle.dumps(_run_loo_subset)
    restored = pickle.loads(dumped)
    assert restored is _run_loo_subset


def test_picklable_analyzer_instance_pickles():
    """The worker process also pickles `self` (passed as _run_loo_subset's
    first argument) -- confirm an analyzer built from picklable
    collaborators doesn't itself block pickling."""
    analyzer = _make_picklable_loo_analyzer(max_cores=2)
    restored = pickle.loads(pickle.dumps(analyzer))
    assert isinstance(restored, MetaAnalyzer)
    assert isinstance(restored._engine, _RecordingLooEngine)


def test_run_leave_one_out_analysis_parallel_path_does_not_raise():
    """max_cores>1 exercises the real ProcessPoolExecutor path -- the
    exact scenario the reported bug occurred in."""
    analyzer = _make_picklable_loo_analyzer(max_cores=3)
    original_data = [{"name": "A"}, {"name": "B"}, {"name": "C"}]

    # Previously: every subset raised "Can't pickle local object
    # '...analyze_subset'", caught internally and logged as an error,
    # leaving every entry None.
    results = analyzer.run_leave_one_out_analysis(original_data, use_old_method=False)

    assert len(results) == 3
    assert all(r is not None for r in results), (
        "One or more leave-one-out subsets returned None -- this is "
        "exactly the failure mode the pickling fix addresses."
    )
    assert sorted(r["remaining_datasets"] for r in results) == sorted(
        [("B", "C"), ("A", "C"), ("A", "B")]
    )


def test_run_leave_one_out_analysis_parallel_and_sequential_agree():
    """Same input through both branches should produce the same logical
    result, confirming the fix changed only how per-subset work gets
    dispatched, not the leave-one-out semantics themselves."""
    original_data = [{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}]

    seq = _make_picklable_loo_analyzer(max_cores=1).run_leave_one_out_analysis(
        original_data, use_old_method=False
    )
    par = _make_picklable_loo_analyzer(max_cores=4).run_leave_one_out_analysis(
        original_data, use_old_method=False
    )

    seq_sets = sorted(r["remaining_datasets"] for r in seq)
    par_sets = sorted(r["remaining_datasets"] for r in par)
    assert seq_sets == par_sets


# ===================================================================
# TESTS — run_meta_analysis: leave-one-out failures are no longer silent
# ===================================================================
# filter_genes()'s leave-one-out consistency step only runs
# `if meta_object.get("leaveOneOutAnalysis")`, which is falsy for an
# *empty* dict -- so when every LOO subset fails (as happened for the
# pickling bug above), that step is silently skipped even with
# is_leave_one_out=True in the config, with nothing in the log to say so.
# run_meta_analysis() now logs a warning whenever one or more subsets
# fail, so that silent skip can never be a surprise.


def test_run_meta_analysis_warns_when_all_loo_subsets_fail(monkeypatch, caplog):
    """When every leave-one-out subset fails, a warning logs the full failure count."""
    analyzer = _make_test_analyzer()
    monkeypatch.setattr(analyzer, "run_meta_analysis_core", lambda data, use_old: {"meta": "core"})
    monkeypatch.setattr(
        analyzer,
        "run_leave_one_out_analysis",
        lambda data, use_old: [None, None, None],
    )
    meta_obj = {"originalData": {"ds1": {"dummy": 1}, "ds2": {"dummy": 2}, "ds3": {"dummy": 3}}}

    with caplog.at_level(logging.WARNING):
        result = analyzer.run_meta_analysis(meta_obj, run_leave_one_out=True)

    assert result["leaveOneOutAnalysis"] == {}
    assert "leave-one-out subset(s) failed" in caplog.text
    assert "3/3" in caplog.text


def test_run_meta_analysis_warns_with_correct_counts_on_partial_loo_failure(monkeypatch, caplog):
    """When only some leave-one-out subsets fail, the warning reports the correct fraction."""
    analyzer = _make_test_analyzer()
    monkeypatch.setattr(analyzer, "run_meta_analysis_core", lambda data, use_old: {"meta": "core"})
    monkeypatch.setattr(
        analyzer,
        "run_leave_one_out_analysis",
        lambda data, use_old: [{"a": 1}, None, {"c": 3}],
    )
    meta_obj = {"originalData": {"ds1": {"dummy": 1}, "ds2": {"dummy": 2}, "ds3": {"dummy": 3}}}

    with caplog.at_level(logging.WARNING):
        result = analyzer.run_meta_analysis(meta_obj, run_leave_one_out=True)

    assert set(result["leaveOneOutAnalysis"].keys()) == {"removed_ds1", "removed_ds3"}
    assert caplog.text.count("leave-one-out subset(s) failed") == 1
    assert "1/3" in caplog.text


def test_run_meta_analysis_no_warning_when_all_loo_subsets_succeed(monkeypatch, caplog):
    """Companion to test_run_meta_analysis_leave_one_out above: same happy
    path, plus confirming the new warning stays silent when it should."""
    analyzer = _make_test_analyzer()
    meta_obj = {"originalData": {"ds1": {"dummy": 1}, "ds2": {"dummy": 2}, "ds3": {"dummy": 3}}}
    monkeypatch.setattr(analyzer, "run_meta_analysis_core", lambda data, use_old: {"meta": "core"})
    monkeypatch.setattr(
        analyzer,
        "run_leave_one_out_analysis",
        lambda data, use_old: [{"a": 1}, {"b": 2}, {"c": 3}],
    )

    with caplog.at_level(logging.WARNING):
        result = analyzer.run_meta_analysis(meta_obj, run_leave_one_out=True)

    assert len(result["leaveOneOutAnalysis"]) == 3
    assert "leave-one-out subset(s) failed" not in caplog.text


# ===================================================================
# TESTS — module-level run_meta_analysis convenience function
# ===================================================================
# The convenience function creates a MetaAnalyzer internally, so we
# need module-level patching here — this is the one legitimate use.


def test_run_meta_analysis_wrapper(monkeypatch):
    """The convenience wrapper correctly delegates to MetaAnalyzer."""
    called = {"run_called": False}

    class DummyAnalyzer(MetaAnalyzer):
        def run_meta_analysis(self, meta_object, run_leave_one_out, use_old_method):
            called["run_called"] = True
            return {"metaAnalysis": "done"}

    monkeypatch.setattr(meta_analysis, "MetaAnalyzer", DummyAnalyzer)

    meta_obj = {"originalData": {"ds1": {"dummy": 1}, "ds2": {"dummy": 2}}}
    result = meta_analysis.run_meta_analysis(meta_obj, max_cores=2, use_old_method=True)
    assert called["run_called"]
    assert result["metaAnalysis"] == "done"


# ===================================================================
# TESTS — run_meta_analysis edge cases
# ===================================================================


def test_run_meta_analysis_validation_fail():
    """Validation failure raises ValueError with proper message."""
    failing_validator = SimpleNamespace(
        convert_original_data_names=lambda obj: obj,
        validate_meta_object=lambda obj, _, __: (False, ["err1", "err2"]),
        check_class_vector=lambda ds: True,
    )
    analyzer = _make_test_analyzer(validator=failing_validator)

    meta_obj = {"originalData": {"ds1": {"dummy": 1}}}
    with pytest.raises(ValueError, match="Error in input object"):
        analyzer.run_meta_analysis(meta_obj)


def test_run_meta_analysis_leave_one_out(monkeypatch):
    """When run_leave_one_out is True, LOO results are attached."""
    analyzer = _make_test_analyzer()
    meta_obj = {"originalData": {"ds1": {"dummy": 1}, "ds2": {"dummy": 2}, "ds3": {"dummy": 3}}}
    monkeypatch.setattr(analyzer, "run_meta_analysis_core", lambda data, use_old: {"meta": "core"})
    monkeypatch.setattr(
        analyzer,
        "run_leave_one_out_analysis",
        lambda data, use_old: [{"a": 1}, {"b": 2}, {"c": 3}],
    )

    result = analyzer.run_meta_analysis(meta_obj, run_leave_one_out=True)
    assert result["leaveOneOutAnalysis"]["removed_ds1"] == {"a": 1}
    assert result["leaveOneOutAnalysis"]["removed_ds2"] == {"b": 2}
    assert result["leaveOneOutAnalysis"]["removed_ds3"] == {"c": 3}


# ===================================================================
# TESTS — integration: complete pipeline
# ===================================================================


def test_meta_analysis_complete_output_structure():
    """Integration test: full pipeline returns correct structure."""
    mock_core = SimpleNamespace(
        calculate_effect_sizes=lambda dataset: pd.DataFrame(
            {"g": [1.0, 2.0], "se_g": [0.5, 0.5], "keys": ["gene0", "gene1"]}
        ),
        expand_dataframe=lambda df: df,
        summarize_effects_within_study=lambda df, method: df,
        collapse_probes_to_genes=lambda df, method="fixed_iv": (
            df.set_index("keys")[["g", "se_g"]] if "keys" in df.columns else df
        ),
        pool_inverse_variance=lambda g_matrix, se_matrix, method: pd.DataFrame(
            {
                "summary": [1.5, 2.5],
                "se_summary": [0.3, 0.3],
                "p_value": [0.01, 0.05],
                "tau2": [0.1, 0.1],
                "n_studies": [2, 2],
                "Q": [1.0, 1.0],
                "df": [1, 1],
                "pval_het": [0.3, 0.3],
            },
            index=["gene0", "gene1"],
        ),
    )
    mock_fisher = SimpleNamespace(
        calculate_study_pvalues=lambda dataset: pd.DataFrame(
            {"P.up": [0.01, 0.05], "P.down": [0.99, 0.95]}, index=["gene0", "gene1"]
        ),
        sum_of_logs=lambda pvalue_list: pd.DataFrame(
            {
                "F.stat.up": [10.0, 8.0],
                "F.pval.up": [0.01, 0.05],
                "F.stat.down": [5.0, 3.0],
                "F.pval.down": [0.5, 0.6],
            },
            index=["gene0", "gene1"],
        ),
    )

    analyzer = _make_test_analyzer(
        engine=RandomEffectsEngine(core=mock_core, fisher=mock_fisher),
    )

    mock_dataset_1 = {"expr": pd.DataFrame([[1, 2, 3]]), "class": np.array([0, 0, 1])}
    mock_dataset_2 = {"expr": pd.DataFrame([[4, 5, 6]]), "class": np.array([0, 1, 1])}

    result = analyzer.run_meta_analysis_core([mock_dataset_1, mock_dataset_2], use_old_method=False)

    assert "pooled_estimates" in result
    assert isinstance(result["pooled_estimates"], pd.DataFrame)
    assert "effect_size" in result["pooled_estimates"].columns
    assert all(result["pooled_estimates"]["effect_size"] == result["pooled_estimates"]["summary"])

    for col in ["p_value", "p_fdr", "summary", "se_summary"]:
        assert col in result["pooled_estimates"].columns

    for key in [
        "datasetEffectSizes",
        "datasetEffectSizeStandardErrors",
        "pooledResults",
        "analysisDescription",
    ]:
        assert key in result


# ===================================================================
# Phase 3 TESTS — ABC, Engine, Factory, Pipeline, Gene Count
# ===================================================================


class TestMetaAnalysisEngineABC:
    """Tests for the MetaAnalysisEngine abstract base class."""

    def test_cannot_instantiate_abc(self):
        """MetaAnalysisEngine cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract method"):
            MetaAnalysisEngine()

    def test_subclass_must_implement_all_abstract_methods(self):
        """A subclass missing any abstract method cannot be instantiated."""

        class IncompleteEngine(MetaAnalysisEngine):
            def compute_effect_sizes(self, original_data):
                pass

            # Missing: compute_fisher_results, format_results, description

        with pytest.raises(TypeError, match="abstract method"):
            IncompleteEngine()

    def test_complete_subclass_can_be_instantiated(self):
        """A subclass implementing all abstract methods works."""

        class StubEngine(MetaAnalysisEngine):
            def compute_effect_sizes(self, original_data):
                return make_es_result()

            def compute_fisher_results(self, original_data, use_old_method):
                return pd.DataFrame()

            def format_results(self, combined, fisher):
                return {"pooledResults": pd.DataFrame(), "analysisDescription": "stub"}

            @property
            def description(self):
                return "Stub Engine"

        engine = StubEngine()
        assert engine.description == "Stub Engine"

    def test_abc_combine_effect_sizes_raises_not_implemented(self):
        """The default combine_effect_sizes on the ABC raises NotImplementedError."""

        class MinimalEngine(MetaAnalysisEngine):
            def compute_effect_sizes(self, d):
                return make_es_result()

            def compute_fisher_results(self, d, u):
                return pd.DataFrame()

            def format_results(self, c, f):
                return {}

            @property
            def description(self):
                return "minimal"

        engine = MinimalEngine()
        with pytest.raises(NotImplementedError):
            engine.combine_effect_sizes([])


class TestRandomEffectsEngine:
    """Tests for the RandomEffectsEngine concrete implementation."""

    def test_default_construction(self):
        """Defaults to MetaAnalysisCore and FisherCombination."""
        engine = RandomEffectsEngine()
        assert engine._core is not None
        assert engine._fisher is not None

    def test_di_construction(self):
        """Custom core and fisher can be injected."""
        mock_core = SimpleNamespace(name="custom_core")
        mock_fisher = SimpleNamespace(name="custom_fisher")
        engine = RandomEffectsEngine(core=mock_core, fisher=mock_fisher)
        assert engine._core.name == "custom_core"
        assert engine._fisher.name == "custom_fisher"

    def test_description_property(self):
        """Description returns the expected string."""
        engine = RandomEffectsEngine()
        assert engine.description == "MetaAnalysis: Random Effects Model"

    def test_between_method_is_random(self):
        """_between_method is 'random'."""
        engine = RandomEffectsEngine()
        assert engine._between_method == "random"

    def test_combine_effect_sizes_empty_raises(self):
        """combine_effect_sizes rejects empty input."""
        engine = _make_test_engine()
        with pytest.raises(ValueError, match="List of effects cannot be empty"):
            engine.combine_effect_sizes([])

    def test_combine_effect_sizes_returns_effect_size_result(self):
        """combine_effect_sizes returns EffectSizeResult."""
        engine = _make_test_engine()
        df1 = make_dummy_effect_df("A", ["gene0"], [1.0])
        df2 = make_dummy_effect_df("B", ["gene1"], [2.0])
        result = engine.combine_effect_sizes([df1, df2])
        assert isinstance(result, EffectSizeResult)
        assert isinstance(result.g, pd.DataFrame)
        assert isinstance(result.se_g, pd.DataFrame)
        assert isinstance(result.pooled_estimates, pd.DataFrame)

    def test_separate_matrices(self):
        """_separate_matrices correctly splits by column suffix."""
        merged = pd.DataFrame(
            {
                "study_0_g": [1.0, 2.0],
                "study_0_se_g": [0.1, 0.2],
                "study_1_g": [3.0, 4.0],
                "study_1_se_g": [0.3, 0.4],
            },
            index=["geneA", "geneB"],
        )
        g, se = RandomEffectsEngine._separate_matrices(merged)
        assert list(g.columns) == ["study_0_g", "study_1_g"]
        assert list(se.columns) == ["study_0_se_g", "study_1_se_g"]
        assert g.shape == (2, 2)
        assert se.shape == (2, 2)

    def test_multimerge_vectorized(self):
        """_multimerge uses pd.concat and produces correct shape."""
        df1 = pd.DataFrame({"g": [1.0], "se_g": [0.1]}, index=["geneA"])
        df2 = pd.DataFrame({"g": [2.0], "se_g": [0.2]}, index=["geneB"])
        df3 = pd.DataFrame({"g": [3.0], "se_g": [0.3]}, index=["geneA"])

        merged = RandomEffectsEngine._multimerge_effect_sizes([df1, df2, df3], ["s0", "s1", "s2"])
        assert merged.shape == (2, 6)
        assert merged.loc["geneA", "s0_g"] == 1.0
        assert merged.loc["geneA", "s2_g"] == 3.0
        assert np.isnan(merged.loc["geneB", "s0_g"])

    def test_adjust_fisher_adds_qval_columns(self):
        """_adjust_fisher adds F.Qval.up and F.Qval.down columns."""
        df = pd.DataFrame(
            {
                "F.pval.up": [0.01, 0.05],
                "F.pval.down": [0.10, 0.02],
                "F.stat.up": [10, 8],
                "F.stat.down": [5, 3],
            },
            index=["g0", "g1"],
        )
        result = RandomEffectsEngine._adjust_fisher(df)
        assert "F.Qval.up" in result.columns
        assert "F.Qval.down" in result.columns


class TestMetaAnalyzerEngineIntegration:
    """Tests for MetaAnalyzer ↔ engine integration."""

    def test_default_engine_is_random_effects(self):
        """Without explicit engine, a RandomEffectsEngine is created."""
        analyzer = MetaAnalyzer()
        assert isinstance(analyzer._engine, RandomEffectsEngine)

    def test_explicit_engine_is_used(self):
        """An explicitly passed engine is used instead of creating one."""

        class DummyEngine(MetaAnalysisEngine):
            def compute_effect_sizes(self, d):
                return make_es_result()

            def compute_fisher_results(self, d, u):
                return pd.DataFrame()

            def format_results(self, c, f):
                return {"analysisDescription": "dummy"}

            @property
            def description(self):
                return "Dummy"

        engine = DummyEngine()
        analyzer = MetaAnalyzer(engine=engine)
        assert analyzer._engine is engine
        assert analyzer._engine.description == "Dummy"

    def test_core_fisher_di_creates_engine(self):
        """Phase-2 style DI (core=, fisher=) creates a RandomEffectsEngine."""
        mock_core = SimpleNamespace(name="injected_core")
        mock_fisher = SimpleNamespace(name="injected_fisher")
        analyzer = MetaAnalyzer(core=mock_core, fisher=mock_fisher)
        assert isinstance(analyzer._engine, RandomEffectsEngine)
        assert analyzer._engine._core.name == "injected_core"
        assert analyzer._engine._fisher.name == "injected_fisher"

    def test_engine_overrides_core_fisher(self):
        """When engine is provided, core/fisher params are ignored."""
        mock_core = SimpleNamespace(name="should_be_ignored")
        engine = RandomEffectsEngine(core=SimpleNamespace(name="engine_core"))
        analyzer = MetaAnalyzer(core=mock_core, engine=engine)
        assert analyzer._engine._core.name == "engine_core"

    def test_delegates_compute_effect_sizes(self, monkeypatch):
        """MetaAnalyzer.run_meta_analysis_core delegates to engine."""
        analyzer = _make_test_analyzer()
        called = {"hit": False}

        def fake_compute(original_data):
            called["hit"] = True
            return make_es_result(
                pooled=pd.DataFrame({"p_value": [0.05], "summary": [1.0]}, index=["g0"]),
            )

        monkeypatch.setattr(analyzer._engine, "compute_effect_sizes", fake_compute)
        monkeypatch.setattr(
            analyzer._engine,
            "compute_fisher_results",
            lambda data, use_old: pd.DataFrame(),
        )
        monkeypatch.setattr(
            analyzer._engine,
            "format_results",
            lambda combined, fisher: {"pooledResults": pd.DataFrame({"x": [1]})},
        )
        analyzer.run_meta_analysis_core([{"dummy": 1}])
        assert called["hit"]

    def test_delegates_combine_effect_sizes(self, monkeypatch):
        """MetaAnalyzer.combine_effect_sizes delegates to engine."""
        analyzer = _make_test_analyzer()
        called = {"hit": False}

        def fake_combine(*args, **kwargs):
            called["hit"] = True
            return make_es_result()

        monkeypatch.setattr(analyzer._engine, "combine_effect_sizes", fake_combine)
        analyzer.combine_effect_sizes([pd.DataFrame()])
        assert called["hit"]

    def test_analysis_description_comes_from_engine(self):
        """The analysisDescription field in results comes from engine.description."""
        engine = _make_test_engine()
        combined = make_es_result(pooled=make_pooled_estimates([0.05]))
        fisher_df = make_fisher_output([0.02], [0.03])

        result = engine.format_results(combined, fisher_df)
        assert result["analysisDescription"] == "MetaAnalysis: Random Effects Model"


class TestMetaAnalysisFactory:
    """Tests for MetaAnalysisFactory."""

    def test_create_random_effects(self):
        """create_analyzer('random_effects') returns a valid analyzer."""
        analyzer = MetaAnalysisFactory.create_analyzer("random_effects")
        assert isinstance(analyzer, MetaAnalyzer)
        assert isinstance(analyzer._engine, RandomEffectsEngine)

    def test_create_fixed_effects(self):
        """create_analyzer('fixed_effects') returns a valid analyzer."""
        analyzer = MetaAnalysisFactory.create_analyzer("fixed_effects")
        assert isinstance(analyzer, MetaAnalyzer)
        assert isinstance(analyzer._engine, FixedEffectsEngine)

    def test_default_strategy(self):
        """Default strategy is random_effects."""
        analyzer = MetaAnalysisFactory.create_analyzer()
        assert isinstance(analyzer._engine, RandomEffectsEngine)

    def test_create_with_max_cores(self, monkeypatch):
        """max_cores is forwarded to MetaAnalyzer."""
        monkeypatch.setattr(mp, "cpu_count", lambda: 8)
        analyzer = MetaAnalysisFactory.create_analyzer(max_cores=4)
        assert analyzer.max_cores == 4

    def test_unknown_strategy_raises(self):
        """Unknown strategy names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            MetaAnalysisFactory.create_analyzer("bayesian")

    def test_available_strategies(self):
        """available_strategies returns the list of implemented strategies."""
        strategies = MetaAnalysisFactory.available_strategies()
        assert "random_effects" in strategies
        assert "fixed_effects" in strategies
        assert isinstance(strategies, list)

    def test_kwargs_forwarded_to_engine(self):
        """Extra kwargs are passed to the engine constructor."""
        mock_core = SimpleNamespace(name="factory_core")
        analyzer = MetaAnalysisFactory.create_analyzer(core=mock_core)
        assert analyzer._engine._core.name == "factory_core"

    def test_hyphenated_strategy_name(self):
        """Hyphenated strategy names are normalized."""
        analyzer = MetaAnalysisFactory.create_analyzer("random-effects")
        assert isinstance(analyzer._engine, RandomEffectsEngine)


class TestTotalGenesLogging:
    """Tests for 'Total genes analyzed' log message."""

    def test_total_genes_logged_at_info_level(self, caplog):
        """run_meta_analysis_core logs 'Total genes analyzed: N' at INFO level."""
        mock_core = SimpleNamespace(
            calculate_effect_sizes=lambda dataset: pd.DataFrame(
                {
                    "g": [1.0, 2.0, 3.0],
                    "se_g": [0.1, 0.2, 0.3],
                    "keys": ["geneA", "geneB", "geneC"],
                }
            ),
            collapse_probes_to_genes=lambda df, method="fixed_iv": (
                df.set_index("keys")[["g", "se_g"]] if "keys" in df.columns else df
            ),
            pool_inverse_variance=lambda g, se, method: pd.DataFrame(
                {
                    "summary": [0.5, 1.0, 1.5],
                    "se_summary": [0.1, 0.2, 0.3],
                    "p_value": [0.01, 0.05, 0.10],
                    "tau2": [0.1, 0.1, 0.1],
                    "n_studies": [2, 2, 2],
                    "Q": [1.0, 1.0, 1.0],
                    "pval_het": [0.3, 0.3, 0.3],
                },
                index=["geneA", "geneB", "geneC"],
            ),
        )
        mock_fisher = SimpleNamespace(
            calculate_study_pvalues=lambda dataset: pd.DataFrame(
                {"P.up": [0.01, 0.05, 0.10], "P.down": [0.99, 0.95, 0.90]},
                index=["geneA", "geneB", "geneC"],
            ),
            sum_of_logs=lambda pvalue_list: pd.DataFrame(
                {
                    "F.stat.up": [10, 8, 6],
                    "F.pval.up": [0.01, 0.05, 0.10],
                    "F.stat.down": [5, 3, 2],
                    "F.pval.down": [0.5, 0.6, 0.7],
                },
                index=["geneA", "geneB", "geneC"],
            ),
        )

        analyzer = _make_test_analyzer(
            engine=RandomEffectsEngine(core=mock_core, fisher=mock_fisher),
        )

        with caplog.at_level(logging.INFO, logger="metaintegrator.analysis"):
            analyzer.run_meta_analysis_core([{"d": 1}, {"d": 2}])

        assert any(
            "Total genes analyzed: 3" in msg for msg in caplog.messages
        ), f"Expected 'Total genes analyzed: 3' in log messages. Got: {caplog.messages}"

    def test_total_genes_count_matches_pooled_results(self, monkeypatch, caplog):
        """The logged gene count matches len(pooledResults)."""
        analyzer = _make_test_analyzer()

        def fake_compute_es(data):
            return make_es_result(
                pooled=pd.DataFrame(
                    {
                        "summary": range(5),
                        "se_summary": range(5),
                        "p_value": [0.01] * 5,
                        "p_fdr": [0.02] * 5,
                        "tau2": [0.1] * 5,
                        "n_studies": [2] * 5,
                        "Q": [1.0] * 5,
                        "pval_het": [0.3] * 5,
                        "effect_size": range(5),
                    },
                    index=[f"g{i}" for i in range(5)],
                ),
            )

        def fake_fisher(data, use_old):
            return pd.DataFrame(
                {
                    "F.stat.up": range(5),
                    "F.pval.up": [0.01] * 5,
                    "F.Qval.up": [0.02] * 5,
                    "F.stat.down": range(5),
                    "F.pval.down": [0.05] * 5,
                    "F.Qval.down": [0.06] * 5,
                },
                index=[f"g{i}" for i in range(5)],
            )

        monkeypatch.setattr(analyzer._engine, "compute_effect_sizes", fake_compute_es)
        monkeypatch.setattr(analyzer._engine, "compute_fisher_results", fake_fisher)

        with caplog.at_level(logging.INFO, logger="metaintegrator.analysis"):
            result = analyzer.run_meta_analysis_core([{"d": 1}])

        assert len(result["pooledResults"]) == 5
        assert any("Total genes analyzed: 5" in msg for msg in caplog.messages)


class TestCustomEnginePluggability:
    """Tests demonstrating that a custom engine can replace the default."""

    def test_custom_engine_in_full_pipeline(self):
        """A custom engine's methods are called through run_meta_analysis_core."""
        call_log = []

        class TrackingEngine(MetaAnalysisEngine):
            def compute_effect_sizes(self, original_data):
                call_log.append("effect_sizes")
                return EffectSizeResult(
                    g=pd.DataFrame({"s0_g": [1.0]}, index=["g0"]),
                    se_g=pd.DataFrame({"s0_se_g": [0.1]}, index=["g0"]),
                    pooled_estimates=pd.DataFrame(
                        {
                            "summary": [1.0],
                            "se_summary": [0.1],
                            "p_value": [0.01],
                            "p_fdr": [0.02],
                            "tau2": [0.1],
                            "n_studies": [2],
                            "Q": [1.0],
                            "pval_het": [0.3],
                            "effect_size": [1.0],
                        },
                        index=["g0"],
                    ),
                )

            def compute_fisher_results(self, original_data, use_old_method):
                call_log.append("fisher")
                return pd.DataFrame(
                    {
                        "F.stat.up": [10],
                        "F.pval.up": [0.01],
                        "F.Qval.up": [0.02],
                        "F.stat.down": [5],
                        "F.pval.down": [0.5],
                        "F.Qval.down": [0.6],
                    },
                    index=["g0"],
                )

            def format_results(self, combined, fisher):
                call_log.append("format")
                pooled = combined.pooled_estimates
                pr = pooled[
                    [
                        "summary",
                        "se_summary",
                        "p_value",
                        "p_fdr",
                        "tau2",
                        "n_studies",
                        "Q",
                        "pval_het",
                    ]
                ].copy()
                pr.columns = [
                    "effectSize",
                    "effectSizeStandardError",
                    "effectSizePval",
                    "effectSizeFDR",
                    "tauSquared",
                    "numStudies",
                    "cochranesQ",
                    "heterogeneityPval",
                ]
                fr = fisher[
                    [
                        "F.stat.up",
                        "F.pval.up",
                        "F.Qval.up",
                        "F.stat.down",
                        "F.pval.down",
                        "F.Qval.down",
                    ]
                ].copy()
                fr.columns = [
                    "fisherStatUp",
                    "fisherPvalUp",
                    "fisherFDRUp",
                    "fisherStatDown",
                    "fisherPvalDown",
                    "fisherFDRDown",
                ]
                merged = pr.join(fr, how="outer")
                return {
                    "datasetEffectSizes": combined.g,
                    "datasetEffectSizeStandardErrors": combined.se_g,
                    "pooledResults": merged,
                    "pooled_estimates": pooled,
                    "analysisDescription": self.description,
                }

            @property
            def description(self):
                return "Custom Tracking Engine"

        analyzer = _make_test_analyzer(engine=TrackingEngine())
        result = analyzer.run_meta_analysis_core([{"d": 1}])

        assert call_log == ["effect_sizes", "fisher", "format"]
        assert result["analysisDescription"] == "Custom Tracking Engine"
        assert "pooledResults" in result
        assert len(result["pooledResults"]) == 1


class TestBackwardCompatibility:
    """Ensure all Phase 1/2 interfaces still work after refactoring."""

    def test_phase2_di_still_works(self):
        """MetaAnalyzer(core=..., fisher=...) still constructs correctly."""
        mc = _make_mock_core()
        mf = _make_mock_fisher()
        analyzer = MetaAnalyzer(core=mc, fisher=mf)
        assert analyzer._engine._core is mc
        assert analyzer._engine._fisher is mf

    def test_module_level_run_meta_analysis_exists(self):
        """The module-level run_meta_analysis function is importable."""
        assert callable(run_meta_analysis)

    def test_meta_analyzer_has_public_delegate(self):
        """MetaAnalyzer exposes combine_effect_sizes as a public delegate."""
        analyzer = MetaAnalyzer()
        assert callable(analyzer.combine_effect_sizes)

    def test_shims_removed(self):
        """Underscore-prefixed delegate shims no longer exist on MetaAnalyzer."""
        analyzer = MetaAnalyzer()
        assert not hasattr(analyzer, "_compute_effect_sizes")
        assert not hasattr(analyzer, "_compute_fisher_results")
        assert not hasattr(analyzer, "_format_final_results")
        assert not hasattr(analyzer, "_adjust_fisher")

    def test_engine_is_accessible(self):
        """The engine is accessible via analyzer._engine."""
        analyzer = MetaAnalyzer()
        assert hasattr(analyzer, "_engine")
        assert isinstance(analyzer._engine, MetaAnalysisEngine)


# ===================================================================
# Phase 4 TESTS — EffectSizeResult, InverseVarianceEngine, FixedEffects
# ===================================================================


class TestEffectSizeResult:
    """Tests for the EffectSizeResult data-model contract."""

    def test_construction(self):
        """EffectSizeResult can be constructed with three DataFrames."""
        g = pd.DataFrame({"s0_g": [1.0]}, index=["g0"])
        se = pd.DataFrame({"s0_se_g": [0.1]}, index=["g0"])
        pooled = pd.DataFrame({"summary": [0.5]}, index=["g0"])
        r = EffectSizeResult(g=g, se_g=se, pooled_estimates=pooled)
        pd.testing.assert_frame_equal(r.g, g)
        pd.testing.assert_frame_equal(r.se_g, se)
        pd.testing.assert_frame_equal(r.pooled_estimates, pooled)

    def test_attribute_access(self):
        """Fields are accessible as attributes."""
        r = make_es_result()
        _ = r.g
        _ = r.se_g
        _ = r.pooled_estimates

    def test_dict_getitem_compat(self):
        """Dict-style key access works for backward compat."""
        pooled = pd.DataFrame({"x": [1]})
        r = EffectSizeResult(g=pd.DataFrame(), se_g=pd.DataFrame(), pooled_estimates=pooled)
        pd.testing.assert_frame_equal(r["pooled_estimates"], pooled)

    def test_dict_getitem_bad_key_raises_key_error(self):
        """Accessing a non-existent key raises KeyError, not AttributeError."""
        r = make_es_result()
        with pytest.raises(KeyError):
            _ = r["nonexistent"]

    def test_dict_setitem_compat(self):
        """Dict-style assignment works for existing fields."""
        r = make_es_result()
        new_pooled = pd.DataFrame({"y": [2]})
        r["pooled_estimates"] = new_pooled
        pd.testing.assert_frame_equal(r.pooled_estimates, new_pooled)

    def test_dict_setitem_bad_key_raises(self):
        """Setting a non-existent key raises KeyError."""
        r = make_es_result()
        with pytest.raises(KeyError):
            r["bad_key"] = 42

    def test_contains(self):
        """'in' operator works for field names."""
        r = make_es_result()
        assert "g" in r
        assert "se_g" in r
        assert "pooled_estimates" in r
        assert "nonexistent" not in r

    def test_keys(self):
        """keys() returns the three field names."""
        r = make_es_result()
        assert r.keys() == {"g", "se_g", "pooled_estimates"}

    def test_missing_field_at_construction(self):
        """Omitting a required field raises TypeError."""
        with pytest.raises(TypeError):
            EffectSizeResult(g=pd.DataFrame(), se_g=pd.DataFrame())


class TestInverseVarianceEngine:
    """Tests for the shared InverseVarianceEngine base class."""

    def test_cannot_instantiate_directly(self):
        """InverseVarianceEngine is abstract (missing _between_method, description)."""
        with pytest.raises(TypeError, match="abstract method"):
            InverseVarianceEngine()

    def test_shared_methods_accessible_on_random(self):
        """Shared helpers are inherited by RandomEffectsEngine."""
        engine = RandomEffectsEngine()
        assert callable(engine._adjust_fisher)
        assert callable(engine._collapse_study)
        assert callable(engine._multimerge_effect_sizes)
        assert callable(engine._separate_matrices)

    def test_shared_methods_accessible_on_fixed(self):
        """Shared helpers are inherited by FixedEffectsEngine."""
        engine = FixedEffectsEngine()
        assert callable(engine._adjust_fisher)
        assert callable(engine._collapse_study)
        assert callable(engine._multimerge_effect_sizes)
        assert callable(engine._separate_matrices)

    def test_custom_iv_subclass(self):
        """A custom InverseVarianceEngine subclass can be created."""

        class WeightedEngine(InverseVarianceEngine):
            @property
            def _between_method(self):
                return "random"

            @property
            def description(self):
                return "Custom Weighted"

        engine = WeightedEngine()
        assert engine._between_method == "random"
        assert engine.description == "Custom Weighted"

    def test_adjust_fisher_shared_across_engines(self):
        """_adjust_fisher produces same results on both engines."""
        df = pd.DataFrame(
            {"F.pval.up": [0.01, 0.05], "F.pval.down": [0.10, 0.02]},
            index=["g0", "g1"],
        )
        r_result = RandomEffectsEngine._adjust_fisher(df.copy())
        f_result = FixedEffectsEngine._adjust_fisher(df.copy())
        pd.testing.assert_frame_equal(r_result, f_result)

    def test_separate_matrices_shared_across_engines(self):
        """_separate_matrices is the same static method on both engines."""
        assert RandomEffectsEngine._separate_matrices is FixedEffectsEngine._separate_matrices

    def test_multimerge_shared_across_engines(self):
        """_multimerge_effect_sizes is the same static method on both engines."""
        assert (
            RandomEffectsEngine._multimerge_effect_sizes
            is FixedEffectsEngine._multimerge_effect_sizes
        )


class TestFixedEffectsEngine:
    """Tests for the FixedEffectsEngine concrete implementation."""

    def test_between_method_is_fixed(self):
        """_between_method returns 'fixed'."""
        engine = FixedEffectsEngine()
        assert engine._between_method == "fixed"

    def test_description(self):
        """Description identifies as fixed effects."""
        engine = FixedEffectsEngine()
        assert "Fixed Effects" in engine.description

    def test_di_construction(self):
        """Custom core and fisher can be injected."""
        mock_core = SimpleNamespace(name="fixed_core")
        engine = FixedEffectsEngine(core=mock_core)
        assert engine._core.name == "fixed_core"

    def test_combine_uses_fixed_method(self):
        """combine_effect_sizes passes method='fixed' to pool_inverse_variance."""
        captured = {}

        def spy_pool(g, se, method):
            captured["method"] = method
            return pd.DataFrame(
                {
                    "summary": [0.5],
                    "se_summary": [0.2],
                    "p_value": [0.05],
                    "tau2": [0.0],
                    "n_studies": [2],
                    "Q": [1.0],
                    "pval_het": [0.5],
                },
                index=["g0"],
            )

        mock_core = _make_mock_core()
        mock_core.pool_inverse_variance = spy_pool
        engine = FixedEffectsEngine(core=mock_core)

        df = make_dummy_effect_df("A", ["gene0"], [1.0])
        engine.combine_effect_sizes([df])

        assert captured["method"] == "fixed"

    def test_factory_creates_fixed_effects(self):
        """MetaAnalysisFactory can create a fixed-effects analyzer."""
        analyzer = MetaAnalysisFactory.create_analyzer("fixed_effects")
        assert isinstance(analyzer._engine, FixedEffectsEngine)
        assert analyzer._engine._between_method == "fixed"

    def test_inherits_from_inverse_variance(self):
        """FixedEffectsEngine is a subclass of InverseVarianceEngine."""
        assert issubclass(FixedEffectsEngine, InverseVarianceEngine)

    def test_inherits_from_meta_analysis_engine(self):
        """FixedEffectsEngine is a subclass of MetaAnalysisEngine."""
        assert issubclass(FixedEffectsEngine, MetaAnalysisEngine)


class TestEngineHierarchy:
    """Tests for the class hierarchy relationships."""

    def test_random_extends_iv(self):
        """RandomEffectsEngine is a subclass of InverseVarianceEngine."""
        assert issubclass(RandomEffectsEngine, InverseVarianceEngine)

    def test_fixed_extends_iv(self):
        """FixedEffectsEngine is a subclass of InverseVarianceEngine."""
        assert issubclass(FixedEffectsEngine, InverseVarianceEngine)

    def test_iv_extends_abc(self):
        """InverseVarianceEngine is a subclass of the MetaAnalysisEngine ABC."""
        assert issubclass(InverseVarianceEngine, MetaAnalysisEngine)

    def test_random_and_fixed_share_format_results(self):
        """Both engines inherit the same format_results implementation."""
        assert RandomEffectsEngine.format_results is FixedEffectsEngine.format_results

    def test_random_and_fixed_share_compute_fisher(self):
        """Both engines inherit the same compute_fisher_results implementation."""
        assert (
            RandomEffectsEngine.compute_fisher_results is FixedEffectsEngine.compute_fisher_results
        )

    def test_random_and_fixed_differ_only_in_between_method(self):
        """The only functional difference is _between_method."""
        r = RandomEffectsEngine()
        f = FixedEffectsEngine()
        assert r._between_method != f._between_method
        assert r.description != f.description
        # All other methods are inherited from the same base
        assert type(r).combine_effect_sizes is type(f).combine_effect_sizes

    def test_combine_between_method_override(self):
        """Explicit between_method arg overrides the engine's default."""
        captured = {}

        def spy_pool(g, se, method):
            captured["method"] = method
            return pd.DataFrame(
                {
                    "summary": [0.5],
                    "se_summary": [0.2],
                    "p_value": [0.05],
                    "tau2": [0.1],
                    "n_studies": [3],
                    "Q": [2.3],
                    "pval_het": [0.1],
                },
                index=["gene0"],
            )

        mock_core = _make_mock_core()
        mock_core.pool_inverse_variance = spy_pool
        engine = RandomEffectsEngine(core=mock_core)

        df = make_dummy_effect_df("A", ["g0"], [1.0])
        engine.combine_effect_sizes([df], between_method="fixed")

        assert captured["method"] == "fixed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
