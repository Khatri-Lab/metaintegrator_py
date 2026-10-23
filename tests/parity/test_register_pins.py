"""Tests that pin the entries of the R parity register.

``docs/comparison/r_parity_register.md`` lists each place where this
library knowingly differs from R MetaIntegrator, or knowingly copies an
R weakness. Each entry names the test that pins it. Most pins live in
the parity tests next to the code. The ones here have no better home.

A pin does not say the behavior is right. It says the behavior is
known. If one of these tests fails, someone changed a registered
behavior. Update the register in the same change, and decide on purpose
whether the new behavior should match R.

REG-001  An exact zero becomes ``NaN`` before ``log2``. R gives ``-Inf``.
REG-002  R's log2 rule judges one sample, and misses low raw intensities.
REG-003  Probe averaging keeps missing values, like R's ``colMeans()``.
W-02     R computes a shift for negative values in its log2 rule and
         never uses it. See ``docs/comparison/r_potential_weaknesses.md``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(current_dir, "..", "..", "src")))

from scipy.stats import norm  # noqa: E402

from preprocessing.utils.log_scale import (  # noqa: E402
    apply_r_log2,
    gem_log_check,
    legacy_median_rule_log2,
    r_ppoints,
)

#: What R writes for a gene value that is at or below zero, or missing.
R_FILLER = 1.0


def _lognormal_quantiles(meanlog: float, sdlog: float, n: int) -> np.ndarray:
    """A lognormal sample made of evenly spaced quantiles.

    It has no random draws, so the result is the same on every machine
    and every numpy version.
    """
    return np.exp(meanlog + sdlog * norm.ppf(r_ppoints(n)))


class TestReg001ZeroBecomesMissing:
    """R: ``log2(0)`` is ``-Inf``. Here: ``NaN``, and the count is reported."""

    def test_zero_is_missing_where_r_holds_minus_inf(self) -> None:
        values = _lognormal_quantiles(8.0, 1.5, 200)
        raw = pd.DataFrame({"S1": values, "S2": values * 1.2})
        raw.iloc[7, 1] = 0.0

        transformed, report = apply_r_log2(raw)

        assert report.log2_applied
        assert report.zero_values_set_missing == 1
        assert np.isnan(transformed.iloc[7, 1])
        with np.errstate(divide="ignore"):
            assert np.log2(0.0) == -np.inf  # what R keeps in that cell

    def test_other_values_are_plain_log2(self) -> None:
        values = _lognormal_quantiles(8.0, 1.5, 200)
        raw = pd.DataFrame({"S1": values, "S2": values * 1.2})
        raw.iloc[7, 1] = 0.0

        transformed, _ = apply_r_log2(raw)

        kept = transformed.drop(index=transformed.index[7])
        expected = np.log2(raw.drop(index=raw.index[7]).to_numpy())
        np.testing.assert_allclose(kept.to_numpy(), expected)


class TestReg002RLog2Rule:
    """R's rule looks at one sample. Low raw intensities can pass as logs."""

    @pytest.mark.parametrize("n_values", [500, 2000])
    def test_r_keeps_low_intensity_raw_data(self, n_values: int) -> None:
        values = _lognormal_quantiles(7.0, 1.0, n_values)  # median about 1,100
        matrix = pd.DataFrame({"S1": values, "S2": values * 1.1})

        check = gem_log_check(matrix)

        assert check.is_log_scale  # R leaves raw intensities as they are
        assert legacy_median_rule_log2(matrix)  # the former median rule would not

    def test_same_shape_at_higher_intensity_is_transformed(self) -> None:
        values = _lognormal_quantiles(8.0, 1.5, 500)  # median about 3,000
        matrix = pd.DataFrame({"S1": values, "S2": values * 1.1})

        assert not gem_log_check(matrix).is_log_scale

    def test_only_the_first_usable_sample_decides(self) -> None:
        raw = _lognormal_quantiles(8.0, 1.5, 500)
        logged = np.log2(raw)
        # The first column looks logged, the second is raw. R reads one.
        matrix = pd.DataFrame({"S1": logged, "S2": raw})

        assert gem_log_check(matrix).is_log_scale


class TestW02UnusedShift:
    """R builds a shifted copy for negative values, then ignores it.

    In R's ``.GEM_log_check()`` the shifted copy is ``x - min + 1``.
    The three ``qqnorm`` calls read the unshifted column. This port
    does the same. The case below has a negative minimum, and the
    decision differs between the data as given (what R uses) and the
    shifted data (what the R comment says it wants).
    """

    @staticmethod
    def _shifted(values: np.ndarray) -> np.ndarray:
        return values - values.min() + 1 if values.min() < 0 else values

    def test_decision_uses_the_data_as_given(self) -> None:
        values = _lognormal_quantiles(5.0, 1.5, 200) - 530  # minimum about -528
        matrix = pd.DataFrame({"S1": values, "S2": values * 1.1})
        assert values.min() < 0

        assert gem_log_check(matrix).is_log_scale

    def test_a_shifted_copy_would_decide_differently(self) -> None:
        values = _lognormal_quantiles(5.0, 1.5, 200) - 530
        shifted = pd.DataFrame({"S1": self._shifted(values), "S2": self._shifted(values * 1.1)})

        assert not gem_log_check(shifted).is_log_scale


IMPLEMENTATIONS = ["app_extractor", "scoring_processor"]


def _sample_level(kind: str):
    """One of the two library implementations of ``getSampleLevelGeneData()``."""
    if kind == "app_extractor":
        module = pytest.importorskip("metaintegrator.coconut_integration")
        return module.get_sample_level_gene_data
    module = pytest.importorskip("metaintegrator.sample_level_gene_data")
    return module.SampleLevelGeneDataProcessor.get_sample_level_gene_data


class TestReg003ProbeMeanKeepsMissing:
    """R's ``colMeans()`` keeps ``NA``, so the gene becomes 1. We do the same.

    Before 0.12.0 the library skipped the missing probe. These tests make
    sure it does not go back to that.
    """

    @staticmethod
    def _dataset() -> dict:
        expr = pd.DataFrame(
            {"s1": [np.nan, 8.0, 5.0], "s2": [6.0, 8.0, 5.0]}, index=["p1", "p2", "p3"]
        )
        return {"expr": expr, "keys": ["A", "A", "B"]}

    @pytest.mark.parametrize("kind", IMPLEMENTATIONS)
    def test_a_missing_probe_makes_the_gene_one_like_r(self, kind: str) -> None:
        sample_level = _sample_level(kind)
        result = sample_level(self._dataset(), ["A", "B"])

        assert result.loc["A", "s1"] == R_FILLER  # R: mean(NA, 8) is NA, then 1

    @pytest.mark.parametrize("kind", IMPLEMENTATIONS)
    def test_complete_probes_are_averaged_like_r(self, kind: str) -> None:
        sample_level = _sample_level(kind)
        result = sample_level(self._dataset(), ["A", "B"])

        assert result.loc["A", "s2"] == pytest.approx(7.0)
        assert result.loc["B", "s2"] == pytest.approx(5.0)
        assert result.loc["B", "s1"] == pytest.approx(5.0)

    @pytest.mark.parametrize("kind", IMPLEMENTATIONS)
    def test_a_single_missing_probe_becomes_one_like_r(self, kind: str) -> None:
        sample_level = _sample_level(kind)
        dataset = {
            "expr": pd.DataFrame({"s1": [np.nan, 4.0]}, index=["p1", "p2"]),
            "keys": ["A", "B"],
        }

        assert sample_level(dataset, ["A", "B"]).loc["A", "s1"] == R_FILLER

    @pytest.mark.parametrize("kind", IMPLEMENTATIONS)
    def test_values_at_or_below_zero_become_one_like_r(self, kind: str) -> None:
        sample_level = _sample_level(kind)
        dataset = {
            "expr": pd.DataFrame({"s1": [0.0, -2.0, 3.0]}, index=["p1", "p2", "p3"]),
            "keys": ["A", "B", "C"],
        }

        result = sample_level(dataset, ["A", "B", "C"])

        assert result["s1"].tolist() == [R_FILLER, R_FILLER, 3.0]

    @pytest.mark.parametrize("kind", IMPLEMENTATIONS)
    def test_a_zero_that_log2_made_missing_gives_one_like_r_s_minus_inf(self, kind: str) -> None:
        sample_level = _sample_level(kind)
        """REG-001 and REG-003 together: R holds -Inf, we hold NaN. Both give 1."""
        raw = pd.DataFrame(
            {"s1": [0.0, 900.0, 5000.0], "s2": [700.0, 900.0, 5000.0]},
            index=["p1", "p2", "p3"],
        )
        transformed = raw.copy()
        transformed[:] = np.log2(raw.mask(raw == 0))  # zeros missing, as in apply_r_log2
        r_matrix = raw.copy()
        with np.errstate(divide="ignore"):
            r_matrix[:] = np.log2(raw)  # zeros -Inf, as R holds them
        keys = ["A", "A", "B"]

        ours = sample_level({"expr": transformed, "keys": keys}, ["A", "B"])
        theirs = sample_level({"expr": r_matrix, "keys": keys}, ["A", "B"])

        assert ours.loc["A", "s1"] == R_FILLER
        np.testing.assert_allclose(ours.to_numpy(), theirs.to_numpy())
