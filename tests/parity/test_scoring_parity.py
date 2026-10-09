"""R parity tests for signature scores, per-dataset AUC and pooled AUC.

Reference values
----------------
The expected numbers come from R MetaIntegrator's own unit tests, run on
the ``tinyMetaObject`` that ships with the R package:

- ``inst/unitTests/test_calculateScore.R``: two ``calculateScore()`` values
  for the first dataset.
- ``inst/unitTests/test_calculateROC.R``: the ``calculateROC()`` AUC of each
  of the three datasets.

R checks them with a tolerance of 1e-3. We use the same tolerance.

R's ``pooledROCPlot()`` scores every dataset with ``calculateScore()``. It
then averages the per-dataset ROC curves, weighted by sample size. So a
pooled AUC built from R-parity scores must land very close to the
sample-size-weighted mean of R's per-dataset AUCs. Before version 0.12.0,
``SignatureScoringEngine`` used a different score and broke this.

A note on the gene search
-------------------------
R's gene search (``.convertDiscoveryListToGenes``) and R's
``calculateScore()`` treat values at or below zero differently. The search
calls ``getSampleLevelGeneData()`` first, which replaces every value <= 0
with 1. ``calculateScore()`` shifts the whole dataset to positive values
first. On data that is already positive, the two scores are identical. On
data with negative values, R itself reports different search and ROC AUCs.
The Python port keeps both R behaviors. The last two tests pin that.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import pytest

from metaintegrator.gene_search_optimizer import GeneSearchOptimizer
from metaintegrator.roc_analysis_engine import (
    MetaAnalysisUtilities,
    ROCAnalysisEngine,
    SignatureScoringEngine,
)
from utils.r_golden import RGoldenMetaObject, load_r_golden

from .parity_support import BUILTIN_GOLDEN_DIRS

pytestmark = pytest.mark.parity

# Tolerance used by R's own unit tests (checkEquals(..., tolerance=1e-3)).
R_TOL = 1e-3

# R: test_calculateScore.R. R indexes from 1, so [[1]] and [[45]].
R_SCORE_FIRST_DATASET = {0: -0.03687, 44: 2.12351}

# R: test_calculateROC.R, in R's originalData order.
R_AUC_BY_DATASET = {
    "Whole.Blood.Study.1": 0.85606,
    "Whole.Blood.Study.2": 0.86738,
    "PBMC.Study.1": 0.95469,
}

# Pooled AUC versus the weighted mean of per-dataset AUCs. These two are
# different statistics, so they never match exactly, but they stay close.
POOLED_VS_WEIGHTED_MEAN_TOL = 0.01

# Search AUC versus pooled AUC on positive data (the plan's acceptance bar).
SEARCH_VS_POOLED_TOL = 0.02


@pytest.fixture(scope="module")
def golden() -> RGoldenMetaObject:
    """R's built-in tinyMetaObject. The reference values above belong to it."""
    return load_r_golden(BUILTIN_GOLDEN_DIRS[0])


@pytest.fixture(scope="module")
def signature(golden: RGoldenMetaObject) -> Dict[str, List[str]]:
    """R's ``tinyMetaObject$filterResults[[1]]``."""
    flt = golden.filter_results[0]
    return {"posGeneNames": flt.pos_gene_names, "negGeneNames": flt.neg_gene_names}


def _score(dataset: Dict, signature: Dict[str, List[str]]) -> np.ndarray:
    """Score one dataset with the engine's default method."""
    return SignatureScoringEngine().calculate_scores(
        dataset["expr"],
        dataset["keys"],
        signature["posGeneNames"],
        signature["negGeneNames"],
    )


def _pooled_auc(datasets: Dict[str, Dict], signature: Dict[str, List[str]]) -> float:
    """Pooled AUC the way the web app builds it.

    Each dataset's ROC curve is interpolated onto a common FPR grid. The TPRs
    are averaged with sample-size weights, and the AUC is taken under the
    averaged curve.
    """
    roc = ROCAnalysisEngine()
    pairs, sizes = [], []
    for dataset in datasets.values():
        result = roc.calculate_roc_curve(np.asarray(dataset["class"]), _score(dataset, signature))
        pairs.append((np.asarray(result["fpr"]), np.asarray(result["tpr"])))
        sizes.append(len(dataset["class"]))
    fpr, tprs = MetaAnalysisUtilities.interpolate_roc_curves(pairs, n_points=1000)
    tpr = MetaAnalysisUtilities.calculate_pooled_tpr(tprs, sizes, method="weighted")
    return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))


def _search_auc(meta_object: Dict, signature: Dict[str, List[str]]) -> float:
    """The gene search's own AUC, normalized to a 0 to 1 weighted mean."""
    weighted_sum = GeneSearchOptimizer().score_signature(meta_object, signature)
    total = sum(len(d["class"]) for d in meta_object["originalData"].values())
    return weighted_sum / total


def _shift_positive(meta_object: Dict) -> Dict:
    """Return a copy with each dataset shifted so its minimum is 1."""
    shifted = copy.deepcopy(meta_object)
    for dataset in shifted["originalData"].values():
        expr = dataset["expr"]
        dataset["expr"] = expr + abs(expr.min().min()) + 1
    return shifted


def test_dataset_order_matches_r(golden: RGoldenMetaObject) -> None:
    """The reference values assume R's originalData order."""
    assert golden.dataset_names == list(R_AUC_BY_DATASET)


def test_scores_match_r_calculate_score(golden, signature) -> None:
    first = golden.original_data[golden.dataset_names[0]]
    scores = _score(first, signature)
    for index, expected in R_SCORE_FIRST_DATASET.items():
        assert scores[index] == pytest.approx(expected, abs=R_TOL)


@pytest.mark.parametrize("name", list(R_AUC_BY_DATASET))
def test_per_dataset_auc_matches_r(golden, signature, name: str) -> None:
    dataset = golden.original_data[name]
    result = ROCAnalysisEngine().calculate_roc_curve(
        np.asarray(dataset["class"]), _score(dataset, signature)
    )
    assert result["auc"] == pytest.approx(R_AUC_BY_DATASET[name], abs=R_TOL)


def test_pooled_auc_matches_weighted_mean_of_r_aucs(golden, signature) -> None:
    sizes = [len(golden.original_data[n]["class"]) for n in R_AUC_BY_DATASET]
    r_weighted_mean = float(np.average(list(R_AUC_BY_DATASET.values()), weights=sizes))
    pooled = _pooled_auc(golden.original_data, signature)
    assert pooled == pytest.approx(r_weighted_mean, abs=POOLED_VS_WEIGHTED_MEAN_TOL)


def test_search_auc_and_pooled_auc_agree_on_positive_data(golden, signature) -> None:
    meta_object = _shift_positive(golden.to_meta_object())
    search = _search_auc(meta_object, signature)
    pooled = _pooled_auc(meta_object["originalData"], signature)
    assert search == pytest.approx(pooled, abs=SEARCH_VS_POOLED_TOL)


def test_search_auc_differs_on_negative_data_as_in_r(golden, signature) -> None:
    """Pins an R behavior, not a Python bug. See the module docstring.

    Two of the three tinyMetaObject datasets contain negative values. R's
    search clamps them to 1 and loses information. ``calculateScore()``
    shifts them instead. If this test starts failing, someone changed one of
    the two R-ported scorers. Check both against R before accepting it.
    """
    meta_object = golden.to_meta_object()
    search = _search_auc(meta_object, signature)
    pooled = _pooled_auc(meta_object["originalData"], signature)
    assert pooled - search > 0.2
