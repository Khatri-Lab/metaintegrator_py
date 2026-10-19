"""Tests for ``preprocessing.utils.log_scale``: the R-parity log-scale check.

The module ports R MetaIntegrator's ``getGEOData()`` log2 decision
(``.GEM_log_check``) and transform. These tests pin the R building
blocks (``ppoints``, ``qqnorm``, ``cor``) to values R gives, each branch
of R's decision, and the transform. Comparison against R's own output on
shared inputs lives in ``tests/parity/test_log_check_parity.py``.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(current_dir, "..", "..", "src")))

from preprocessing.utils import log_scale  # noqa: E402
from preprocessing.utils.log_scale import (  # noqa: E402
    DECIDED_BY_CHECK_FAILED,
    DECIDED_BY_FALLBACK,
    DECIDED_BY_NO_USABLE_COLUMN,
    DECIDED_BY_SCORE,
    R_RULE_NAME,
    apply_r_log2,
    gem_log_check,
    legacy_median_rule_log2,
    r_cor_pairwise,
    r_ppoints,
    r_qqnorm,
)


def _frame(values: np.ndarray) -> pd.DataFrame:
    """Wrap an array as a probes x samples frame with GSM-style columns."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return pd.DataFrame(
        values,
        index=[f"p{i}" for i in range(values.shape[0])],
        columns=[f"GSM{j}" for j in range(values.shape[1])],
    )


# ---------------------------------------------------------------------------
# R building blocks
# ---------------------------------------------------------------------------


class TestRPpoints:
    def test_small_n_uses_three_eighths(self) -> None:
        # R: ppoints(5) = (1:5 - 3/8) / 5.25
        np.testing.assert_allclose(
            r_ppoints(5), [0.1190476190, 0.3095238095, 0.5, 0.6904761905, 0.8809523810]
        )

    def test_large_n_uses_one_half(self) -> None:
        # R: ppoints(11) = (1:11 - 0.5) / 11
        np.testing.assert_allclose(r_ppoints(11), (np.arange(1, 12) - 0.5) / 11)

    def test_zero_gives_empty(self) -> None:
        assert r_ppoints(0).size == 0


class TestRQqnorm:
    def test_quantiles_follow_the_ranks(self) -> None:
        # R: qqnorm(c(3, 1, 2))$x is 0.8694238 -0.8694238 0.0000000
        x, y = r_qqnorm(np.array([3.0, 1.0, 2.0]))
        np.testing.assert_allclose(x, [0.8694238, -0.8694238, 0.0], atol=1e-7)
        np.testing.assert_array_equal(y, [3.0, 1.0, 2.0])

    def test_missing_values_stay_missing(self) -> None:
        x, y = r_qqnorm(np.array([2.0, np.nan, 1.0]))
        assert np.isnan(x[1]) and np.isnan(y[1])
        # Two values left, so ppoints(2) is used for the others.
        np.testing.assert_allclose(x[[0, 2]], [0.5894558, -0.5894558], atol=1e-7)

    def test_ties_keep_their_original_order(self) -> None:
        """order(order(y)) is stable: the first of two equal values ranks first."""
        x, _ = r_qqnorm(np.array([5.0, 5.0, 1.0]))
        assert x[0] < x[1]

    def test_infinite_values_are_ranked_last(self) -> None:
        x, _ = r_qqnorm(np.array([np.inf, 1.0, 2.0]))
        assert x[0] == max(x)

    def test_all_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="only NAs"):
            r_qqnorm(np.array([np.nan, np.nan]))


class TestRCorPairwise:
    def test_matches_numpy_on_complete_data(self) -> None:
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=50), rng.normal(size=50)
        assert r_cor_pairwise(a, b) == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-12)

    def test_drops_incomplete_pairs(self) -> None:
        a = np.array([1.0, 2.0, np.nan, 4.0])
        b = np.array([2.0, 4.0, 6.0, np.nan])
        assert r_cor_pairwise(a, b) == pytest.approx(1.0)

    def test_constant_side_gives_nan(self) -> None:
        assert np.isnan(r_cor_pairwise(np.array([1.0, 2.0, 3.0]), np.ones(3)))

    def test_huge_values_do_not_overflow(self) -> None:
        """exp() of microarray values can reach 1e300. R sums in extended
        precision, so its correlation stays finite; ours must too."""
        x = np.linspace(-2, 2, 100)
        y = np.exp(np.linspace(600, 700, 100))
        result = r_cor_pairwise(x, y)
        expected = r_cor_pairwise(x, y / 1e300)
        assert np.isfinite(result)
        assert result == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


class TestGemLogCheck:
    def test_raw_intensities_are_not_log_scale(self) -> None:
        rng = np.random.default_rng(1)
        check = gem_log_check(_frame(rng.lognormal(8, 1.5, (2000, 3))))
        assert check.is_log_scale is False
        assert check.decided_by == DECIDED_BY_SCORE
        assert check.score is not None and check.score > 0

    def test_log_scale_data_is_left_alone(self) -> None:
        rng = np.random.default_rng(1)
        check = gem_log_check(_frame(rng.normal(8, 2, (2000, 3))))
        assert check.is_log_scale is True
        assert check.score is not None and check.score <= 0

    def test_low_intensity_raw_data_is_caught(self) -> None:
        """A median near 12 fooled the former "median > 20" rule. R's
        QQ-plot rule still sees the skewed shape of raw intensities."""
        rng = np.random.default_rng(1)
        data = _frame(rng.lognormal(2.5, 1.5, (2000, 3)))
        assert gem_log_check(data).is_log_scale is False
        assert legacy_median_rule_log2(data) is False

    def test_r_can_keep_data_the_median_rule_would_transform(self) -> None:
        """The rules also disagree the other way: a flat spread of large
        values looks "normal enough" to R's QQ-plot check."""
        rng = np.random.default_rng(1)
        data = _frame(rng.uniform(100, 10000, (2000, 3)))
        assert gem_log_check(data).is_log_scale is True
        assert legacy_median_rule_log2(data) is True

    def test_only_the_first_usable_column_counts(self) -> None:
        """R looks at one column: the first that is not constant or missing."""
        rng = np.random.default_rng(2)
        values = np.column_stack(
            [
                np.full(500, 7.0),  # constant: skipped
                np.full(500, np.nan),  # all missing: skipped
                rng.normal(8, 2, 500),  # the reference column
                rng.lognormal(8, 1.5, 500),  # ignored
            ]
        )
        check = gem_log_check(_frame(values))
        assert check.reference_column == "GSM2"
        assert check.is_log_scale is True

    def test_negative_values_are_checked_unshifted(self) -> None:
        """R shifts a copy of the reference column, then never uses it.
        Shifting the input first must therefore change the result's
        correlations, proving this port also checks the raw values."""
        rng = np.random.default_rng(3)
        values = rng.normal(0, 2, 1000)
        raw = gem_log_check(_frame(values))
        shifted = gem_log_check(_frame(values - values.min() + 1))
        assert raw.cor_log != pytest.approx(shifted.cor_log)

    def test_exp_overflow_everywhere_uses_the_fallback(self) -> None:
        """Values above ~710 make exp() infinite. R then compares cor_real
        with cor_log only."""
        rng = np.random.default_rng(4)
        check = gem_log_check(_frame(rng.lognormal(9, 1.0, (1000, 2)) + 1000))
        assert check.decided_by == DECIDED_BY_FALLBACK
        assert check.cor_exp is None
        assert check.score is None
        assert check.is_log_scale is False

    def test_no_usable_column_warns_and_leaves_data_alone(self) -> None:
        with pytest.warns(UserWarning, match="constant or missing"):
            check = gem_log_check(_frame(np.full((10, 3), 5.0)))
        assert check.is_log_scale is True
        assert check.decided_by == DECIDED_BY_NO_USABLE_COLUMN

    def test_uncomputable_score_warns_like_r(self, monkeypatch) -> None:
        """0/0 gives NaN. R warns, and getGEOData() leaves the data alone."""
        monkeypatch.setattr(log_scale, "r_cor_pairwise", lambda x, y: 0.5)
        rng = np.random.default_rng(5)
        with pytest.warns(UserWarning, match="log check failed"):
            check = gem_log_check(_frame(rng.normal(8, 2, (100, 2))))
        assert check.is_log_scale is True
        assert check.decided_by == DECIDED_BY_CHECK_FAILED

    def test_accepts_a_plain_array(self) -> None:
        rng = np.random.default_rng(6)
        assert gem_log_check(rng.normal(8, 2, (300, 2))).is_log_scale is True

    def test_is_deterministic(self) -> None:
        """No random sampling, unlike the older check_log_scale()."""
        rng = np.random.default_rng(7)
        data = _frame(rng.lognormal(4, 1.2, (5000, 2)))
        assert gem_log_check(data) == gem_log_check(data)


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------


class TestApplyRLog2:
    def test_log_scale_data_is_returned_unchanged(self) -> None:
        rng = np.random.default_rng(8)
        data = _frame(rng.normal(8, 2, (500, 3)))

        result, report = apply_r_log2(data)

        assert result is data
        assert report.log2_applied is False
        assert report.shift_added is None
        assert report.zero_values_set_missing == 0

    def test_raw_data_gets_plain_log2(self) -> None:
        rng = np.random.default_rng(8)
        data = _frame(rng.lognormal(8, 1.5, (500, 3)))

        result, report = apply_r_log2(data)

        np.testing.assert_allclose(result.to_numpy(), np.log2(data.to_numpy()))
        assert report.log2_applied is True
        assert report.rule == R_RULE_NAME
        assert list(result.columns) == list(data.columns)

    def test_negative_values_shift_the_whole_matrix_like_r(self) -> None:
        """R: expr - min + 1, then log2. The smallest value becomes log2(1) = 0."""
        rng = np.random.default_rng(9)
        values = rng.lognormal(8, 1.5, (500, 3))
        values[0, 0] = -50.0
        data = _frame(values)

        result, report = apply_r_log2(data)

        assert report.log2_applied is True
        assert report.shift_added == pytest.approx(51.0)
        assert result.iloc[0, 0] == pytest.approx(0.0)
        np.testing.assert_allclose(result.to_numpy(), np.log2(values + 51.0))

    def test_zeros_become_missing_and_are_counted(self) -> None:
        """R gives -Inf for log2(0). This library uses NaN and says so."""
        rng = np.random.default_rng(10)
        values = rng.lognormal(8, 1.5, (500, 3))
        values[:4, 1] = 0.0
        result, report = apply_r_log2(_frame(values))

        assert report.zero_values_set_missing == 4
        assert result.iloc[:4, 1].isna().all()
        assert not np.isinf(result.to_numpy()).any()

    def test_existing_missing_values_stay_missing(self) -> None:
        rng = np.random.default_rng(11)
        values = rng.lognormal(8, 1.5, (500, 3))
        values[5, 2] = np.nan
        result, report = apply_r_log2(_frame(values))
        assert np.isnan(result.iloc[5, 2])
        assert report.zero_values_set_missing == 0

    def test_input_is_not_modified(self) -> None:
        rng = np.random.default_rng(12)
        data = _frame(rng.lognormal(8, 1.5, (500, 3)))
        before = data.copy()
        apply_r_log2(data)
        pd.testing.assert_frame_equal(data, before)

    def test_all_missing_matrix_is_left_alone(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, report = apply_r_log2(_frame(np.full((5, 2), np.nan)))
        assert report.log2_applied is False
        assert report.decided_by == DECIDED_BY_NO_USABLE_COLUMN
        assert report.median_before is None
        assert result.isna().all().all()


class TestReport:
    def test_metadata_is_strict_json(self) -> None:
        rng = np.random.default_rng(13)
        _, report = apply_r_log2(_frame(rng.lognormal(8, 1.5, (500, 3))))

        metadata = report.to_metadata()

        json.dumps(metadata, allow_nan=False)
        assert metadata["log2_applied"] is True
        assert metadata["reference_column"] == "GSM0"
        assert metadata["differs_from_legacy"] is False

    def test_infinite_score_is_kept_as_text(self) -> None:
        report = log_scale.LogScaleReport(
            rule=R_RULE_NAME,
            log2_applied=True,
            decided_by=DECIDED_BY_SCORE,
            score=float("inf"),
            cor_real=1.0,
            cor_log=0.5,
            cor_exp=1.0,
            reference_column="GSM0",
            median_before=100.0,
            shift_added=None,
            zero_values_set_missing=0,
            legacy_median_rule_log2=True,
        )
        assert report.to_metadata()["score"] == "inf"

    def test_disagreement_with_the_former_rule_is_flagged(self) -> None:
        rng = np.random.default_rng(1)
        _, report = apply_r_log2(_frame(rng.lognormal(2.5, 1.5, (2000, 3))))
        assert report.log2_applied is True
        assert report.legacy_median_rule_log2 is False
        assert report.differs_from_legacy is True
