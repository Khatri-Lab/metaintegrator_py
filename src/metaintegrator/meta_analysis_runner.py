"""
Headless CLI runner for the MetaIntegrator meta-analysis pipeline.

This module mirrors the interactive workflow of the SLE tutorial notebook
but drives every step from a validated JSON/YAML config file, making it
suitable for unattended, reproducible, and scriptable execution.

Pipeline stages
---------------
1. **Download** — fetch GEO datasets via ``DataLoaderFactory``
2. **Reformat** — build the ``meta_data`` dict expected by downstream code
3. **Label** — assign binary case/control class vectors
4. **Analyse** — run meta-analysis (with optional leave-one-out)
5. **Filter** — extract significant genes
6. **Search** — backward and/or forward gene-set optimization
7. **Score & Evaluate** — compute signature scores, ROC, PRC
8. **Plot** — generate all requested figures
9. **Export** — write CSVs and plots to the output directory

Usage
-----
From the CLI (the recommended entry point)::

    python -m src meta_analysis --config my_config.yaml
    python -m src meta_analysis --config my_config.yaml --dry-run

Programmatically::

    from metaintegrator.meta_analysis_runner import run_pipeline
    run_pipeline("my_config.yaml")

See Also
--------
``meta_analysis_config.py`` — config validation and template generation.
``coconut/coconut_runner.py`` — the analogous runner for COCONUT.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ── project imports ──────────────────────────────────────────────────
from .meta_analysis_config import (
    load_config_file,
    validate_config,
    generate_template,
    _DEFAULT_POSSIBLE_COLUMNS,
)

logger = logging.getLogger(__name__)

__all__ = [
    "run_pipeline",
    "dry_run",
    "main",
]


# =====================================================================
# Pipeline orchestrator
# =====================================================================


def run_pipeline(
    config_path: str,
    *,
    override_output_dir: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute the full meta-analysis pipeline from a config file.

    This is the primary programmatic entry point.  It loads the config,
    validates it, and runs every stage in sequence.

    Args:
    config_path : str
        Path to a ``.json`` or ``.yaml`` config file.
    override_output_dir : str, optional
        If given, overrides ``output.output_dir`` from the config.
    verbose : bool
        If ``True``, set log level to ``DEBUG`` for richer progress output.

    Returns:
    dict
        The final ``meta_object`` with all results attached.

    Raises
    SystemExit
        If the config is invalid (errors are printed to stderr).
    """
    _configure_logging(verbose)

    # ── 0. Load & validate config ────────────────────────────────────
    _banner("LOADING CONFIGURATION")
    raw = load_config_file(config_path)
    config, errors = validate_config(raw)

    if errors:
        _print_errors(errors)
        sys.exit(1)

    if override_output_dir:
        config["output"]["output_dir"] = override_output_dir

    output_dir = Path(config["output"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir = output_dir / "plot_data"
    plot_data_dir.mkdir(exist_ok=True)

    # Save a copy of the resolved config alongside the results
    _save_resolved_config(config, output_dir)

    logger.info(f"Output directory: {output_dir}")
    logger.info(
        f"Datasets: {config['datasets']['gse_ids']}  " f"Strategy: {config['datasets']['strategy']}"
    )

    t0 = time.time()

    # ── 1. Download datasets ─────────────────────────────────────────
    _banner("STAGE 1 / 8 — DOWNLOADING GEO DATASETS")
    datasets = _download_datasets(config)
    logger.info(f"Downloaded {len(datasets)} dataset(s).")

    # ── 2. Build meta-analysis dict ──────────────────────────────────
    _banner("STAGE 2 / 8 — BUILDING META-ANALYSIS DATA STRUCTURE")
    meta_object = _build_meta_object(datasets)
    logger.info(
        f"Meta object contains {len(meta_object['originalData'])} dataset(s): "
        f"{list(meta_object['originalData'].keys())}"
    )

    # ── 3. Label samples ─────────────────────────────────────────────
    _banner("STAGE 3 / 8 — ASSIGNING CLASS LABELS")
    meta_object = _label_samples(meta_object, config)
    _report_class_summary(meta_object)

    # ── 4. Run meta-analysis ─────────────────────────────────────────
    _banner("STAGE 4 / 8 — RUNNING META-ANALYSIS")
    meta_object = _run_meta_analysis(meta_object, config)
    _report_meta_analysis_summary(meta_object)

    # ── 5. Filter genes ──────────────────────────────────────────────
    _banner("STAGE 5 / 8 — FILTERING GENES")
    meta_object = _filter_genes(meta_object, config)
    _report_filter_summary(meta_object)

    # ── 6. Gene search ───────────────────────────────────────────────
    _banner("STAGE 6 / 8 — GENE SEARCH OPTIMIZATION")
    meta_object = _run_gene_search(meta_object, config)

    # ── 7. Score & evaluate ──────────────────────────────────────────
    _banner("STAGE 7 / 8 — COMPUTING SCORES & EVALUATION METRICS")
    meta_object = _compute_scores_and_roc(meta_object, config)

    # ── 8. Export results & plots ────────────────────────────────────
    _banner("STAGE 8 / 8 — SAVING RESULTS & GENERATING PLOTS")
    _export_results(meta_object, config, output_dir, plot_data_dir)
    _generate_plots(meta_object, config, output_dir, plot_data_dir)

    elapsed = time.time() - t0
    _banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    logger.info(f"All outputs saved to: {output_dir}")

    return meta_object


# =====================================================================
# Dry-run mode
# =====================================================================


def dry_run(config_path: str, *, verbose: bool = False) -> bool:
    """Validate config and check GEO connectivity without running the pipeline.

    Args:
    config_path : str
        Path to the config file.
    verbose : bool
        Extra logging.

    Returns:
    bool
        ``True`` if everything looks good.
    """
    _configure_logging(verbose)
    _banner("DRY RUN — VALIDATING CONFIGURATION")

    raw = load_config_file(config_path)
    config, errors = validate_config(raw)

    if errors:
        _print_errors(errors)
        return False

    print("  ✅ Config schema is valid.")

    # Check that labeling covers every GSE
    gse_ids = config["datasets"]["gse_ids"]
    labeling = config["labeling"]
    for gse_id in gse_ids:
        matched = any(pattern in gse_id or gse_id.startswith(pattern) for pattern in labeling)
        if not matched:
            # Also check if a labeling key is a substring of a derived ID
            # (e.g. GSE11909 → GSE11909_GPL96)
            # We can't know platform suffixes yet, so just check prefix matching
            matched = any(gse_id.startswith(pattern) for pattern in labeling)
        if not matched:
            print(
                f"  ⚠️  No labeling entry matches GSE ID '{gse_id}'.  "
                f"Datasets derived from it may fail at the labeling stage."
            )

    print(f"  ✅ Labeling entries cover {len(labeling)} pattern(s): " f"{list(labeling.keys())}")

    # Quick connectivity check — try importing the loader and listing
    # the cache directory
    try:
        from preprocessing.loader_factory import DataLoaderFactory  # noqa: F401

        print("  ✅ Preprocessing modules import successfully.")
    except ImportError as exc:
        print(f"  ❌ Cannot import preprocessing modules: {exc}")
        return False  # pragma: no cover

    output_dir = Path(config["output"]["output_dir"]).expanduser()
    print(f"  ✅ Output directory: {output_dir}")
    print(f"  ✅ Plots requested: {config['output']['plots']}")

    ma = config["meta_analysis"]
    print(f"  ✅ Meta-analysis: strategy={ma['strategy']}, " f"LOO={ma['run_leave_one_out']}")

    gs = config["gene_search"]
    print(f"  ✅ Gene search: backward={gs['run_backward']}, " f"forward={gs['run_forward']}")

    _banner("DRY RUN PASSED")
    return True


# =====================================================================
# Stage implementations
# =====================================================================


def _download_datasets(config: Dict) -> list:
    """Stage 1: Download GEO datasets."""
    from preprocessing.loader_factory import DataLoaderFactory

    ds_cfg = config["datasets"]
    loader = DataLoaderFactory.create_geo_loader(
        strategy=ds_cfg["strategy"],
        cache_dir=ds_cfg["cache_dir"],
    )
    datasets = loader.load_multiple(
        ds_cfg["gse_ids"],
        stop_on_error=ds_cfg.get("stop_on_error", False),
    )
    return datasets


def _build_meta_object(datasets: list) -> Dict:
    """Stage 2: Convert downloaded datasets into the meta-analysis dict."""
    meta_object: Dict[str, Any] = {
        "originalData": {},
        "expr": {},
        "pheno": {},
        "class": {},
    }
    for ds in datasets:
        geo_id = ds.geo_id
        meta_object["originalData"][geo_id] = {
            "expr": ds.expr,
            "pheno": ds.pheno,
            "keys": ds.keys,
            "formattedName": geo_id,
            "class": None,
        }
        meta_object["expr"][geo_id] = ds.expr
        meta_object["pheno"][geo_id] = ds.pheno
        meta_object["class"][geo_id] = None
    return meta_object


def _label_samples(meta_object: Dict, config: Dict) -> Dict:
    """Stage 3: Assign class labels using labeling config."""
    from .labeling import assign_class_labels

    labeling_cfg = config["labeling"]

    for dataset_key in list(meta_object["originalData"].keys()):
        # Find the first labeling pattern that matches this dataset key
        matched_spec = None
        for pattern, spec in labeling_cfg.items():
            if pattern in dataset_key:
                matched_spec = spec
                break

        if matched_spec is None:
            logger.warning(f"No labeling config matches dataset '{dataset_key}' — skipping.")
            continue

        disease_terms = matched_spec["disease_terms"]
        possible_columns = matched_spec.get("possible_columns", list(_DEFAULT_POSSIBLE_COLUMNS))

        dataset = meta_object["originalData"][dataset_key]
        pheno = dataset["pheno"]

        # Find the first matching column
        col = None
        for candidate in possible_columns:
            if candidate in pheno.columns:
                col = candidate
                break

        if col is None:
            logger.warning(
                f"No suitable column found in {dataset_key} pheno "
                f"(tried {possible_columns}).  Skipping label assignment."
            )
            continue

        try:
            meta_object["originalData"][dataset_key] = assign_class_labels(
                dataset,
                column_name=col,
                disease_terms=disease_terms,
                dataset_name=dataset_key,
            )
        except Exception as exc:
            logger.error(f"Label assignment failed for {dataset_key} (column='{col}'): {exc}")

    return meta_object


def _run_meta_analysis(meta_object: Dict, config: Dict) -> Dict:
    """Stage 4: Run the core meta-analysis."""
    from .analysis import MetaAnalysisFactory

    ma_cfg = config["meta_analysis"]
    analyzer = MetaAnalysisFactory.create_analyzer(
        strategy=ma_cfg["strategy"],
        max_cores=ma_cfg.get("max_cores"),
    )
    meta_object = analyzer.run_meta_analysis(
        meta_object=meta_object,
        run_leave_one_out=ma_cfg["run_leave_one_out"],
        use_old_method=ma_cfg.get("use_old_method", False),
    )
    return meta_object


def _filter_genes(meta_object: Dict, config: Dict) -> Dict:
    """Stage 5: Filter significant genes."""
    from .analysis import MetaAnalyzer

    f_cfg = config["filter"]
    analyzer = MetaAnalyzer()
    meta_object = analyzer.filter_genes(
        meta_object=meta_object,
        effect_size_thresh=f_cfg["effect_size_thresh"],
        fdr_thresh=f_cfg["fdr_thresh"],
        number_studies_thresh=f_cfg["number_studies_thresh"],
        heterogeneity_pval_thresh=f_cfg["heterogeneity_pval_thresh"],
        is_leave_one_out=f_cfg["is_leave_one_out"],
    )
    return meta_object


def _run_gene_search(meta_object: Dict, config: Dict) -> Dict:
    """Stage 6: Run backward and/or forward gene search optimization."""
    gs_cfg = config["gene_search"]
    run_backward = gs_cfg.get("run_backward", True)
    run_forward = gs_cfg.get("run_forward", True)

    if not run_backward and not run_forward:
        logger.info("Gene search disabled — skipping.")
        return meta_object

    filter_results = meta_object.get("filterResults")
    if filter_results is None:
        logger.warning("No filterResults found — cannot run gene search.")
        return meta_object

    try:
        from .gene_search_optimizer import GeneSearchOptimizer

        optimizer = GeneSearchOptimizer()
    except ImportError as exc:
        logger.warning(f"GeneSearchOptimizer not available: {exc}")
        return meta_object

    if run_backward:
        logger.info("Running backward search …")
        try:
            filter_results = optimizer.backward_search(
                meta_object,
                filter_results,
                back_thresh=gs_cfg.get("back_thresh", 0.0),
            )
            meta_object["filterResults"] = filter_results
            bk_genes = filter_results.get("backwardSearchGenes")
            if bk_genes is not None:
                n = len(bk_genes) if hasattr(bk_genes, "__len__") else "N/A"
                logger.info(f"  Backward search complete — {n} gene(s) selected.")
        except Exception as exc:
            logger.error(f"Backward search failed: {exc}")

    if run_forward:
        logger.info("Running forward search …")
        try:
            filter_results = optimizer.forward_search(
                meta_object,
                filter_results,
                forward_thresh=gs_cfg.get("forward_thresh", 0.0),
            )
            meta_object["filterResults"] = filter_results
            fw_genes = filter_results.get("forwardSearchGenes")
            if fw_genes is not None:
                n = len(fw_genes) if hasattr(fw_genes, "__len__") else "N/A"
                logger.info(f"  Forward search complete — {n} gene(s) selected.")
        except Exception as exc:
            logger.error(f"Forward search failed: {exc}")

    return meta_object


def _compute_scores_and_roc(meta_object: Dict, config: Dict) -> Dict:
    """Stage 7: Calculate signature scores and ROC/PRC metrics."""
    filter_results = meta_object.get("filterResults")
    if filter_results is None:
        logger.warning("No filterResults — skipping score computation.")
        return meta_object

    # ── Scores ────────────────────────────────────────────────────────
    try:
        from .score_calculation import ScoreCalculator

        for ds_name, ds_obj in meta_object["originalData"].items():
            try:
                scores = ScoreCalculator.calculate_score(
                    filter_results, ds_obj, suppress_messages=True
                )
                ds_obj["score"] = scores
                logger.info(
                    f"  Scores for {ds_name}: "
                    f"mean={np.nanmean(scores):.3f}, "
                    f"std={np.nanstd(scores):.3f}"
                )
            except Exception as exc:
                logger.warning(f"  Score calculation failed for {ds_name}: {exc}")
    except ImportError:
        logger.warning("ScoreCalculator not available — skipping scores.")

    # ── ROC / PRC ─────────────────────────────────────────────────────
    requested_plots = set(config["output"].get("plots", []))
    if "roc" in requested_plots or "prc" in requested_plots:
        try:
            from .roc_analysis_engine import ROCAnalysisEngine

            roc_engine = ROCAnalysisEngine()
            roc_datasets = config["output"].get("roc_datasets")
            target_datasets = (
                {k: v for k, v in meta_object["originalData"].items() if k in roc_datasets}
                if roc_datasets
                else meta_object["originalData"]
            )

            roc_results = {}
            for ds_name, ds_obj in target_datasets.items():
                if ds_obj.get("score") is None or ds_obj.get("class") is None:
                    continue
                try:
                    auc_val = roc_engine.calculate_roc_auc(ds_obj["score"], ds_obj["class"])
                    roc_results[ds_name] = auc_val
                    logger.info(f"  AUC for {ds_name}: {auc_val:.4f}")
                except Exception as exc:
                    logger.warning(f"  ROC failed for {ds_name}: {exc}")

            meta_object["_roc_results"] = roc_results
        except ImportError:
            logger.warning("ROC modules not available — skipping ROC/PRC.")

    return meta_object


# =====================================================================
# Export helpers
# =====================================================================


def _export_results(
    meta_object: Dict,
    config: Dict,
    output_dir: Path,
    plot_data_dir: Path,
) -> None:
    """Write tabular results to CSV files."""
    save_plot_data = config["output"].get("save_plot_data", True)

    # ── Meta-analysis pooled results ──────────────────────────────────
    ma = meta_object.get("metaAnalysis")
    if ma and "pooledResults" in ma:
        pooled = ma["pooledResults"]
        if isinstance(pooled, pd.DataFrame):
            path = output_dir / "meta_analysis_pooled_results.csv"
            pooled.to_csv(path)
            logger.info(f"  Saved: {path}")

    # ── Effect sizes ──────────────────────────────────────────────────
    if ma and "datasetEffectSizes" in ma:
        es = ma["datasetEffectSizes"]
        if isinstance(es, pd.DataFrame):
            path = output_dir / "meta_analysis_effect_sizes.csv"
            es.to_csv(path)
            logger.info(f"  Saved: {path}")

    # ── Filter results ────────────────────────────────────────────────
    fr = meta_object.get("filterResults")
    if fr:

        fr_dict = fr.to_dict()
        for key in ("posGeneNames", "negGeneNames"):
            genes = fr_dict.get(key)
            if genes is not None and len(genes) > 0:
                path = plot_data_dir / f"filter_{key}.csv"
                pd.DataFrame({key: genes}).to_csv(path, index=False)
                logger.info(f"  Saved: {path}")

        for key in ("backwardSearchGenes", "forwardSearchGenes"):
            genes = fr_dict.get(key)
            if genes is not None:
                genes_list = list(genes) if not isinstance(genes, list) else genes
                if genes_list:
                    path = plot_data_dir / f"{key}.csv"
                    pd.DataFrame({"gene": genes_list}).to_csv(path, index=False)
                    logger.info(f"  Saved: {path}")

    # ── ROC AUC summary ──────────────────────────────────────────────
    roc_results = meta_object.get("_roc_results", {})
    if roc_results:
        path = plot_data_dir / "roc_auc_per_dataset.csv"
        pd.DataFrame([{"dataset": k, "auc": v} for k, v in roc_results.items()]).to_csv(
            path, index=False
        )
        logger.info(f"  Saved: {path}")

    # ── Score distributions per dataset ───────────────────────────────
    if save_plot_data:
        for ds_name, ds_obj in meta_object["originalData"].items():
            score = ds_obj.get("score")
            cls = ds_obj.get("class")
            if score is not None and cls is not None:
                cls_arr = cls.values if hasattr(cls, "values") else np.asarray(cls)
                path = plot_data_dir / f"scores_{ds_name}.csv"
                pd.DataFrame({"score": score, "class": cls_arr}).to_csv(path, index=False)
                logger.info(f"  Saved: {path}")


def _generate_plots(
    meta_object: Dict,
    config: Dict,
    output_dir: Path,
    plot_data_dir: Path,
) -> None:
    """Generate and save requested plots."""
    import matplotlib

    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt

    requested = set(config["output"].get("plots", []))
    fmt = config["output"].get("plot_format", "png")
    dpi = config["output"].get("plot_dpi", 300)

    if not requested:
        logger.info("No plots requested.")
        return

    # ── Forest plot ───────────────────────────────────────────────────
    if "forest" in requested:
        _safe_plot("_plot_forest", meta_object, config, output_dir, fmt, dpi)

    # ── Heatmap ───────────────────────────────────────────────────────
    if "heatmap" in requested:
        _safe_plot("_plot_heatmap", meta_object, config, output_dir, fmt, dpi)

    # ── Violin ────────────────────────────────────────────────────────
    if "violin" in requested:
        _safe_plot("_plot_violin", meta_object, config, output_dir, fmt, dpi)

    # ── ROC ───────────────────────────────────────────────────────────
    if "roc" in requested:
        _safe_plot("_plot_roc", meta_object, config, output_dir, fmt, dpi)

    # ── PRC ───────────────────────────────────────────────────────────
    if "prc" in requested:
        _safe_plot("_plot_prc", meta_object, config, output_dir, fmt, dpi)

    # ── Regression ────────────────────────────────────────────────────
    if "regression" in requested:
        _safe_plot("_plot_regression", meta_object, config, output_dir, fmt, dpi)

    plt.close("all")


# ── Individual plot helpers ───────────────────────────────────────────


def _safe_plot(fn_name: str, *args, **kwargs) -> None:
    """Call a plot function by name, catching and logging errors."""
    fn = globals().get(fn_name)
    if fn is None:
        logger.warning(f"Plot function {fn_name} not found.")
        return
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        logger.error(f"Plot '{fn_name}' failed: {exc}")


def _plot_forest(meta_object: Dict, config: Dict, output_dir: Path, fmt: str, dpi: int) -> None:
    from visualization.enhanced_plotting import MetaAnalysisPlotter

    plotter = MetaAnalysisPlotter()
    gene = config["output"].get("forest_gene")

    if gene is None:
        # Pick the top gene by absolute effect size from filterResults
        fr = meta_object.get("filterResults", {})

        fr_dict = fr.to_dict()
        pos = fr_dict.get("posGeneNames", [])
        neg = fr_dict.get("negGeneNames", [])
        candidates = list(pos) + list(neg)
        if candidates:
            gene = candidates[0]

    if gene is None:
        logger.warning("No gene specified or found for forest plot — skipping.")
        return

    fig = plotter.forest_plot(meta_object, gene_name=gene)
    if fig is not None:
        path = output_dir / f"forest_{gene}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info(f"  Saved: {path}")


def _plot_heatmap(meta_object: Dict, config: Dict, output_dir: Path, fmt: str, dpi: int) -> None:
    from visualization.enhanced_plotting import MetaAnalysisPlotter

    plotter = MetaAnalysisPlotter()
    fig = plotter.heatmap_plot(meta_object)
    if fig is not None:
        path = output_dir / f"heatmap_effect_sizes.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info(f"  Saved: {path}")


def _plot_violin(meta_object: Dict, config: Dict, output_dir: Path, fmt: str, dpi: int) -> None:
    from visualization.violin_plot import violin_plot

    fr = meta_object.get("filterResults")
    if fr is None:
        logger.warning("No filterResults for violin plot — skipping.")
        return

    target_datasets = config["output"].get("violin_datasets")
    datasets_to_plot = (
        {k: v for k, v in meta_object["originalData"].items() if k in target_datasets}
        if target_datasets
        else meta_object["originalData"]
    )

    for ds_name, ds_obj in datasets_to_plot.items():
        if ds_obj.get("score") is None or ds_obj.get("class") is None:
            continue
        try:
            fig = violin_plot(
                scores=ds_obj["score"],
                classes=ds_obj["class"],
                dataset_name=ds_name,
            )
            if fig is not None:
                safe_name = ds_name.replace("/", "_")
                path = output_dir / f"violin_{safe_name}.{fmt}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                logger.info(f"  Saved: {path}")
        except Exception as exc:
            logger.warning(f"  Violin plot for {ds_name} failed: {exc}")


def _plot_roc(meta_object: Dict, config: Dict, output_dir: Path, fmt: str, dpi: int) -> None:
    from visualization.enhanced_plotting import multiple_roc_plot

    fr = meta_object.get("filterResults")
    if fr is None:
        return

    datasets = meta_object["originalData"]
    scorable = {
        k: v
        for k, v in datasets.items()
        if v.get("score") is not None and v.get("class") is not None
    }

    if not scorable:
        logger.warning("No datasets with scores/classes for ROC — skipping.")
        return

    try:
        fig = multiple_roc_plot(scorable, filter_object=fr)
        if fig is not None:
            path = output_dir / f"roc_multiple.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info(f"  Saved: {path}")
    except Exception as exc:
        logger.warning(f"  ROC plot failed: {exc}")


def _plot_prc(meta_object: Dict, config: Dict, output_dir: Path, fmt: str, dpi: int) -> None:
    from visualization.enhanced_plotting import multiple_prc_plot

    fr = meta_object.get("filterResults")
    if fr is None:
        return

    datasets = meta_object["originalData"]
    scorable = {
        k: v
        for k, v in datasets.items()
        if v.get("score") is not None and v.get("class") is not None
    }

    if not scorable:
        logger.warning("No datasets with scores/classes for PRC — skipping.")
        return

    try:
        fig = multiple_prc_plot(scorable, filter_object=fr)
        if fig is not None:
            path = output_dir / f"prc_multiple.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info(f"  Saved: {path}")
    except Exception as exc:
        logger.warning(f"  PRC plot failed: {exc}")


def _plot_regression(meta_object: Dict, config: Dict, output_dir: Path, fmt: str, dpi: int) -> None:
    reg_dataset = config["output"].get("regression_dataset")
    reg_column = config["output"].get("regression_column")

    if not reg_dataset or not reg_column:
        logger.info(
            "Regression plot requires regression_dataset and "
            "regression_column in config — skipping."
        )
        return

    ds_obj = meta_object["originalData"].get(reg_dataset)
    if ds_obj is None or ds_obj.get("score") is None:
        logger.warning(f"Dataset '{reg_dataset}' not found or has no scores.")
        return

    from visualization.enhanced_plotting import MetaAnalysisPlotter

    plotter = MetaAnalysisPlotter()
    try:
        fig = plotter.regression_plot(ds_obj, covariate_column=reg_column, dataset_name=reg_dataset)
        if fig is not None:
            safe_name = reg_dataset.replace("/", "_")
            path = output_dir / f"regression_{safe_name}_{reg_column}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info(f"  Saved: {path}")
    except Exception as exc:
        logger.warning(f"  Regression plot failed: {exc}")


# =====================================================================
# Reporting helpers
# =====================================================================


def _report_class_summary(meta_object: Dict) -> None:
    for ds_name, ds_obj in meta_object["originalData"].items():
        cls = ds_obj.get("class")
        if cls is None:
            logger.warning(f"  {ds_name}: NO CLASS LABELS")
            continue
        arr = cls.values if hasattr(cls, "values") else np.asarray(cls)
        n_case = int(np.sum(arr == 1))
        n_ctrl = int(np.sum(arr == 0))
        logger.info(f"  {ds_name}: {n_case} cases, {n_ctrl} controls")


def _report_meta_analysis_summary(meta_object: Dict) -> None:
    ma = meta_object.get("metaAnalysis")
    if ma is None:
        logger.warning("  Meta-analysis produced no results.")
        return
    pooled = ma.get("pooledResults")
    if isinstance(pooled, pd.DataFrame):
        n_genes = len(pooled)
        n_sig = (
            int((pooled.get("FDR", pd.Series()) < 0.05).sum()) if "FDR" in pooled.columns else "?"
        )
        logger.info(f"  Pooled results: {n_genes} genes, {n_sig} with FDR < 0.05")
    loo = meta_object.get("leaveOneOutAnalysis")
    if loo:
        logger.info(f"  Leave-one-out: {len(loo)} iteration(s)")


def _report_filter_summary(meta_object: Dict) -> None:
    fr = meta_object.get("filterResults")
    if fr is None:
        logger.warning("  No filter results produced.")
        return

    fr_dict = fr.to_dict()
    pos = fr_dict.get("posGeneNames", [])
    neg = fr_dict.get("negGeneNames", [])
    logger.info(f"  Filtered genes: {len(pos)} up-regulated, {len(neg)} down-regulated")


# =====================================================================
# Utility helpers
# =====================================================================


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _banner(text: str) -> None:
    width = max(60, len(text) + 6)
    border = "=" * width
    logger.info("")
    logger.info(border)
    logger.info(f"   {text}")
    logger.info(border)


def _print_errors(errors: List[str]) -> None:
    sys.stderr.write("\n❌  Configuration errors:\n")
    for i, err in enumerate(errors, 1):
        sys.stderr.write(f"   {i}. {err}\n")
    sys.stderr.write("\n")


def _save_resolved_config(config: Dict, output_dir: Path) -> None:
    """Save the fully-resolved config to output_dir for reproducibility."""
    path = output_dir / "resolved_config.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, default=str)
    logger.info(f"  Resolved config saved: {path}")


# =====================================================================
# CLI entry point (called from __main__.py)
# =====================================================================


def main(args: Dict[str, Any]) -> None:
    """Entry point called by ``src/__main__.py`` after argparse dispatch.

    Args:
    args : dict
        Merged CLI + config-file arguments.  Expected keys:

        - ``config`` (str): path to the config file (required)
        - ``dry_run`` (bool): if ``True``, validate only
        - ``output_dir`` (str or None): override output directory
        - ``verbose`` (bool): enable debug logging
        - ``generate_config`` (str or None): format to generate template
    """
    # ── Template generation mode ──────────────────────────────────────
    gen_fmt = args.get("generate_config")
    if gen_fmt:
        template = generate_template(gen_fmt)
        ext = "yaml" if gen_fmt in ("yaml", "yml") else "json"
        filename = f"my_meta_analysis_config_template.{ext}"
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(template)
        print(f"✅ Template written to: {filename}")
        print(f"   Edit it, then run:  python -m src meta_analysis --config {filename}")
        return

    # ── Normal pipeline execution ─────────────────────────────────────
    config_path = args.get("config")
    if not config_path:
        sys.stderr.write(
            "❌  --config is required.  " "Use --generate-config yaml to create a template.\n"
        )
        sys.exit(1)

    verbose = args.get("verbose", False)

    if args.get("dry_run", False):
        ok = dry_run(config_path, verbose=verbose)
        sys.exit(0 if ok else 1)

    run_pipeline(
        config_path,
        override_output_dir=args.get("output_dir"),
        verbose=verbose,
    )
