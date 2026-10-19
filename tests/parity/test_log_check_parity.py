"""Parity test: the log-scale check and transform against R MetaIntegrator.

Inputs live in ``tests/parity/data/log_check/inputs/`` (written by
``make_inputs.py`` there). R's own results live in ``expected/``,
written by ``tests/parity/r_scripts/make_log_check_golden.R``. Until
someone with R runs that script, ``expected/`` does not exist and these
tests are skipped with a message that says how to create it.

What must match R:

* The decision (log2 or not) for every case: exactly.
* The branch of R's rule, the score and the three QQ-plot correlations:
  to 1e-9.
* The transformed matrix: to 1e-9, except that R's ``-Inf`` (from
  ``log2(0)``) is ``NaN`` here, the one documented difference.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(current_dir, "..", "..", "src")))

from preprocessing.utils.log_scale import apply_r_log2  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data" / "log_check"
INPUT_DIR = DATA_DIR / "inputs"
EXPECTED_DIR = DATA_DIR / "expected"
DECISIONS = EXPECTED_DIR / "decisions.csv"
TOL = 1e-9

pytestmark = pytest.mark.skipif(
    not DECISIONS.exists(),
    reason=(
        "R golden files are missing. Create them with: "
        "Rscript tests/parity/r_scripts/make_log_check_golden.R"
    ),
)

CASES = sorted(p.stem for p in INPUT_DIR.glob("*.csv"))


def _load_input(case: str) -> pd.DataFrame:
    return pd.read_csv(INPUT_DIR / f"{case}.csv", index_col=0)


def _expected_row(case: str) -> pd.Series:
    decisions = pd.read_csv(DECISIONS)
    return decisions.set_index("case").loc[case]


def _python_report(case: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return apply_r_log2(_load_input(case))


def _close(python_value, r_value) -> bool:
    """Compare a Python number with R's, treating NA and Inf like R."""
    if r_value is None or (isinstance(r_value, float) and math.isnan(r_value)):
        return python_value is None
    if python_value is None:
        return False
    return math.isclose(float(python_value), float(r_value), rel_tol=0, abs_tol=TOL) or (
        math.isinf(float(r_value)) and float(python_value) == float(r_value)
    )


def test_every_input_has_an_r_decision() -> None:
    decisions = pd.read_csv(DECISIONS)
    assert sorted(decisions["case"]) == CASES


@pytest.mark.parametrize("case", CASES)
def test_decision_matches_r(case: str) -> None:
    _, report = _python_report(case)
    expected = _expected_row(case)
    assert report.log2_applied is bool(expected["log2_applied"])
    assert report.decided_by == expected["decided_by"]
    assert report.reference_column == str(expected["reference_column"])


@pytest.mark.parametrize("case", CASES)
def test_score_and_correlations_match_r(case: str) -> None:
    _, report = _python_report(case)
    expected = _expected_row(case)
    for field in ("score", "cor_real", "cor_log", "cor_exp"):
        assert _close(getattr(report, field), expected[field]), field


@pytest.mark.parametrize("case", CASES)
def test_transformed_matrix_matches_r(case: str) -> None:
    result, report = _python_report(case)
    path = EXPECTED_DIR / f"{case}_transformed.csv"
    if not report.log2_applied:
        assert not path.exists()
        return

    r_matrix = pd.read_csv(path, index_col=0).to_numpy(dtype=float)
    python_matrix = result.to_numpy(dtype=float)

    # The one documented difference: R's -Inf from log2(0) is NaN here.
    r_matrix = np.where(np.isneginf(r_matrix), np.nan, r_matrix)
    np.testing.assert_allclose(python_matrix, r_matrix, rtol=0, atol=TOL, equal_nan=True)
