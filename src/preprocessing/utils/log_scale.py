"""R-parity log-scale check and log2 transform for expression matrices.

R MetaIntegrator's ``getGEOData()`` decides whether a GEO expression
matrix is already on a log scale, and log2-transforms it when it is not.
This module is an exact port of that decision and that transform, so a
dataset is treated the same way in Python as in R.

The decision: ``.GEM_log_check``
--------------------------------

R looks at one sample column only: the first column that is not constant
and not all missing. It builds three normal QQ plots of that column:

* the values as they are (``real``),
* ``exp()`` of the values (``exp``),
* ``log(abs(value) + 0.00001)`` (``log``).

Data that is already on a log scale looks most "normal" as it is, so its
``real`` QQ plot is close to the ``log`` one and far from the ``exp``
one. R scores this as::

    score = log2(|r_real^2 - r_log^2| / |r_real^2 - r_exp^2|)

where ``r`` is the QQ-plot correlation. A score of zero or below means
"already on a log scale". If ``exp()`` overflows for every value, R falls
back to "log scale if ``r_real > r_log``". If the score cannot be
computed (``NaN``), R only warns. Its caller then treats the dataset as
already on a log scale and leaves it unchanged.

Two R details are kept on purpose, because parity matters more than
tidiness here:

* R shifts negative values of the reference column up before the check,
  but then builds the QQ plots from the *unshifted* column. The shift has
  no effect. This port does the same.
* R's ``qqnorm()`` ranks values with ``order(order(y))`` and uses
  ``ppoints(n)`` for the theoretical quantiles. Both are reproduced
  exactly (:func:`r_ppoints`, :func:`r_qqnorm`).

The transform
-------------

When the data is not on a log scale, R shifts the whole matrix up when
its smallest value is negative (``expr - min + 1``), then takes
``log2``. :func:`apply_r_log2` does the same, with one documented
difference: R turns a value of exactly zero into ``-Inf``. This library
turns it into a missing value (``NaN``) instead, as it always has,
because ``-Inf`` breaks the downstream statistics. The number of such
values is reported, so the difference is never silent.

Earlier versions of this library used a different rule in its loaders:
"log2 when the median of all values is above 20". That rule can disagree
with R on borderline datasets. :class:`LogScaleReport` still records
what the old rule would have decided (``legacy_median_rule_log2``), so a
change in behavior can be spotted and explained.

The older :func:`preprocessing.utils.normalization.check_log_scale` is
not an exact port (it adds shortcut rules and samples at random). Use
:func:`gem_log_check` wherever R parity matters.

References:
    R MetaIntegrator, ``R/getGEOData.R``: ``getGEOData()`` and
    ``.GEM_log_check()``. R ``stats::qqnorm`` and ``stats::ppoints``.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

log = logging.getLogger(__name__)

#: Name of the rule, stored with every report.
R_RULE_NAME = "R MetaIntegrator getGEOData() .GEM_log_check"

#: The median threshold of this library's former rule.
LEGACY_MEDIAN_THRESHOLD = 20.0

#: How the decision was reached. Stored in reports as plain strings.
DECIDED_BY_SCORE = "qq_score"
DECIDED_BY_FALLBACK = "qq_fallback_real_vs_log"
DECIDED_BY_CHECK_FAILED = "check_failed"
DECIDED_BY_NO_USABLE_COLUMN = "no_usable_column"


@dataclass(frozen=True)
class LogScaleCheck:
    """The outcome of :func:`gem_log_check`.

    Attributes:
        is_log_scale: ``True`` when the data should be left as it is.
        decided_by: Which branch of R's rule decided. One of
            ``qq_score``, ``qq_fallback_real_vs_log``, ``check_failed``
            (R warns and leaves the data unchanged) and
            ``no_usable_column`` (every column is constant or missing;
            R itself would fail here).
        score: R's ``log_check_score``, or ``None`` when not computed.
            Can be ``inf`` or ``-inf``.
        cor_real: QQ-plot correlation of the values as they are.
        cor_log: QQ-plot correlation of ``log(abs(value) + 1e-5)``.
        cor_exp: QQ-plot correlation of ``exp(value)``, or ``None`` when
            ``exp()`` overflowed for every value.
        reference_column: Label of the sample column R's rule looked at.
    """

    is_log_scale: bool
    decided_by: str
    score: Optional[float]
    cor_real: Optional[float]
    cor_log: Optional[float]
    cor_exp: Optional[float]
    reference_column: Optional[str]


@dataclass(frozen=True)
class LogScaleReport:
    """What :func:`apply_r_log2` decided and did, ready for metadata.

    Attributes:
        rule: Always :data:`R_RULE_NAME`.
        log2_applied: Whether the matrix was log2-transformed.
        decided_by: See :attr:`LogScaleCheck.decided_by`.
        score: See :attr:`LogScaleCheck.score`.
        cor_real: See :attr:`LogScaleCheck.cor_real`.
        cor_log: See :attr:`LogScaleCheck.cor_log`.
        cor_exp: See :attr:`LogScaleCheck.cor_exp`.
        reference_column: See :attr:`LogScaleCheck.reference_column`.
        median_before: Median of all finite values before any change.
        shift_added: The amount added to every value before ``log2``
            (``1 - min``) when the smallest value was negative, else
            ``None``.
        zero_values_set_missing: How many exact zeros became ``NaN``
            instead of R's ``-Inf``. Zero when no transform ran.
        legacy_median_rule_log2: What the former "median above 20" rule
            would have decided.
    """

    rule: str
    log2_applied: bool
    decided_by: str
    score: Optional[float]
    cor_real: Optional[float]
    cor_log: Optional[float]
    cor_exp: Optional[float]
    reference_column: Optional[str]
    median_before: Optional[float]
    shift_added: Optional[float]
    zero_values_set_missing: int
    legacy_median_rule_log2: bool

    @property
    def differs_from_legacy(self) -> bool:
        """Whether R's rule and the former median rule disagree here."""
        return self.log2_applied != self.legacy_median_rule_log2

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-safe dict (no ``NaN`` or infinity).

        Infinite scores become the strings ``"inf"`` and ``"-inf"``, so
        the value survives strict JSON and still reads clearly.
        """
        data = asdict(self)
        data["differs_from_legacy"] = self.differs_from_legacy
        for key in ("score", "cor_real", "cor_log", "cor_exp", "median_before"):
            data[key] = _json_number(data[key])
        return data


# ---------------------------------------------------------------------------
# R building blocks
# ---------------------------------------------------------------------------


def r_ppoints(n: int) -> np.ndarray:
    """Port of R ``stats::ppoints(n)``.

    Args:
        n: Number of points.

    Returns:
        ``(1:n - a) / (n + 1 - 2a)`` with ``a = 3/8`` when ``n <= 10``,
        else ``1/2``. Empty for ``n <= 0``.
    """
    if n <= 0:
        return np.array([], dtype=float)
    a = 3.0 / 8.0 if n <= 10 else 0.5
    return (np.arange(1, n + 1, dtype=float) - a) / (n + 1 - 2 * a)


def r_qqnorm(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Port of R ``stats::qqnorm(y, plot.it = FALSE)``.

    Missing values (``NaN``) are left out of the ranking and come back as
    ``NaN`` in both outputs, as in R. Infinite values are ranked like any
    other value, also as in R.

    Args:
        y: The sample values.

    Returns:
        ``(x, y)``: the theoretical normal quantiles and the values.

    Raises:
        ValueError: If ``y`` has no non-missing value (R stops with
            "y is empty or has only NAs").
    """
    y = np.asarray(y, dtype=float)
    missing = np.isnan(y)
    present = y[~missing]
    n = present.size
    if n == 0:
        raise ValueError("y is empty or has only NAs")
    # order(order(y)) is the stable rank of each value.
    ranks = np.argsort(np.argsort(present, kind="stable"), kind="stable")
    quantiles = norm.ppf(r_ppoints(n))[ranks]
    x = np.full(y.shape, np.nan)
    x[~missing] = quantiles
    return x, y.copy()


def r_cor_pairwise(x: np.ndarray, y: np.ndarray) -> float:
    """Port of R ``cor(x, y, use = "pairwise.complete")`` (Pearson).

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        The correlation over pairs where both values are present, or
        ``NaN`` when fewer than two pairs remain or either side has zero
        variance (R returns ``NA`` with a warning).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if x.size < 2:
        return float("nan")
    dx, dy = _centered_unit(x), _centered_unit(y)
    if dx is None or dy is None:
        return float("nan")
    denom = math.sqrt(float(np.dot(dx, dx)) * float(np.dot(dy, dy)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(dx, dy) / denom)


def _centered_unit(v: np.ndarray) -> Optional[np.ndarray]:
    """Center ``v`` and divide by its largest absolute deviation.

    Correlation does not change with scale, so this changes nothing in
    the result. It matters for ``exp()`` values near ``1e300``: their
    squares overflow a double, while R sums them in extended precision.
    Scaling first keeps every product finite, as in R.

    Returns:
        The scaled deviations, or ``None`` for a constant or non-finite
        vector.
    """
    if not np.all(np.isfinite(v)):
        return None
    mean = v.mean() if np.all(np.abs(v) < 1e300) else np.mean(v / 1e300) * 1e300
    deviations = v - mean
    peak = float(np.max(np.abs(deviations)))
    if peak == 0.0 or not math.isfinite(peak):
        return None
    return deviations / peak


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def _reference_column_index(values: np.ndarray) -> Optional[int]:
    """Index of the first column that is neither constant nor all missing.

    R keeps columns whose ``sd(na.rm = TRUE)`` is not zero. An ``sd`` of
    ``NA`` (fewer than two values) is dropped too, because ``which()``
    skips ``NA``. "Not constant" is tested as ``max != min``: R's ``sd``
    is exactly zero for a constant column, while a floating-point
    ``np.std`` can come out a hair above zero.
    """
    for j in range(values.shape[1]):
        column = values[:, j]
        present = column[~np.isnan(column)]
        if present.size < 2:
            continue
        if np.max(present) != np.min(present):
            return j
    return None


def gem_log_check(expr: pd.DataFrame | np.ndarray) -> LogScaleCheck:
    """Port of R MetaIntegrator's ``.GEM_log_check``.

    Args:
        expr: Expression matrix, features x samples.

    Returns:
        A :class:`LogScaleCheck`. ``is_log_scale`` is ``True`` when R
        would leave the data unchanged.
    """
    if isinstance(expr, pd.DataFrame):
        values = expr.to_numpy(dtype=float)
        labels = [str(c) for c in expr.columns]
    else:
        values = np.asarray(expr, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        labels = [str(j) for j in range(values.shape[1])]

    ref = _reference_column_index(values) if values.size else None
    if ref is None:
        warnings.warn(
            "Log-scale check: every sample column is constant or missing. "
            "Treating the data as already on a log scale.",
            UserWarning,
            stacklevel=2,
        )
        return LogScaleCheck(
            is_log_scale=True,
            decided_by=DECIDED_BY_NO_USABLE_COLUMN,
            score=None,
            cor_real=None,
            cor_log=None,
            cor_exp=None,
            reference_column=None,
        )

    column = values[:, ref]
    # R also shifts negative values of `inputC` up here, but never uses
    # `inputC` again: the QQ plots below are built from the raw column.

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        real_x, real_y = r_qqnorm(column)
        exp_x, exp_y = r_qqnorm(np.exp(column))
        log_x, log_y = r_qqnorm(np.log(np.abs(column) + 0.00001))

    # R: obj_exp$x[is.infinite(...)] <- NA, and the same for y.
    exp_x = np.where(np.isinf(exp_x), np.nan, exp_x)
    exp_y = np.where(np.isinf(exp_y), np.nan, exp_y)

    cor_real = r_cor_pairwise(real_x, real_y)
    cor_exp = r_cor_pairwise(exp_x, exp_y)
    cor_log = r_cor_pairwise(log_x, log_y)
    label = labels[ref]

    def _result(is_log: bool, decided_by: str, score: Optional[float]) -> LogScaleCheck:
        return LogScaleCheck(
            is_log_scale=is_log,
            decided_by=decided_by,
            score=score,
            cor_real=_none_if_nan(cor_real),
            cor_log=_none_if_nan(cor_log),
            cor_exp=_none_if_nan(cor_exp),
            reference_column=label,
        )

    if math.isnan(cor_exp):
        # R: `if (cor_real > cor_log) TRUE else FALSE`.
        return _result(bool(cor_real > cor_log), DECIDED_BY_FALLBACK, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = abs(cor_real**2 - cor_log**2)
        denominator = abs(cor_real**2 - cor_exp**2)
        ratio = np.float64(numerator) / np.float64(denominator)
        score = float(np.log2(ratio))

    if math.isnan(score):
        # `NaN` comes from 0/0. (An `NA` correlation would make R stop
        # with an error instead; it cannot happen for a non-constant
        # reference column, and is treated the same way here.)
        # R only warns here. Its caller compares the warning text with
        # FALSE, which is never equal, so the data is left unchanged.
        warnings.warn(
            "log check failed - check to make sure this dataset is log2 normalized",
            UserWarning,
            stacklevel=2,
        )
        return _result(True, DECIDED_BY_CHECK_FAILED, None)

    return _result(score <= 0, DECIDED_BY_SCORE, score)


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------


def legacy_median_rule_log2(expr: pd.DataFrame) -> bool:
    """What this library's former rule decided: log2 when median > 20.

    Args:
        expr: Expression matrix.

    Returns:
        ``True`` when the old rule would have log2-transformed the data.
    """
    finite = expr.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    return bool(finite.size and np.median(finite) > LEGACY_MEDIAN_THRESHOLD)


def apply_r_log2(expr: pd.DataFrame) -> Tuple[pd.DataFrame, LogScaleReport]:
    """Decide like R's ``getGEOData()`` and log2-transform when needed.

    R's steps, when :func:`gem_log_check` says the data is not on a log
    scale:

    1. If the smallest value is negative, add ``1 - min`` to every value.
    2. Take ``log2`` of every value.

    Exact zeros become ``NaN`` here, where R would give ``-Inf``. See
    the module docstring.

    Args:
        expr: Expression matrix, features x samples. Not modified.

    Returns:
        ``(matrix, report)``. ``matrix`` is ``expr`` itself when nothing
        changed, otherwise a new frame with the same index and columns.
    """
    values = expr.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    median_before = float(np.median(finite)) if finite.size else None
    legacy = legacy_median_rule_log2(expr)

    if finite.size == 0:
        check = LogScaleCheck(
            is_log_scale=True,
            decided_by=DECIDED_BY_NO_USABLE_COLUMN,
            score=None,
            cor_real=None,
            cor_log=None,
            cor_exp=None,
            reference_column=None,
        )
    else:
        check = gem_log_check(expr)

    shift: Optional[float] = None
    zeros = 0
    result = expr
    if not check.is_log_scale:
        min_value = float(np.nanmin(values))
        shifted = expr
        if min_value < 0:
            shift = 1.0 - min_value
            shifted = expr - min_value + 1
        zero_mask = shifted.to_numpy(dtype=float) == 0
        zeros = int(zero_mask.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.log2(shifted.mask(zero_mask))

    report = LogScaleReport(
        rule=R_RULE_NAME,
        log2_applied=not check.is_log_scale,
        decided_by=check.decided_by,
        score=check.score,
        cor_real=check.cor_real,
        cor_log=check.cor_log,
        cor_exp=check.cor_exp,
        reference_column=check.reference_column,
        median_before=median_before,
        shift_added=shift,
        zero_values_set_missing=zeros,
        legacy_median_rule_log2=legacy,
    )
    return result, report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _none_if_nan(value: float) -> Optional[float]:
    """Return ``None`` for ``NaN``, else the value."""
    return None if value is None or math.isnan(value) else float(value)


def _json_number(value: Optional[float]) -> Any:
    """Make a number strict-JSON safe: infinities become strings."""
    if value is None:
        return None
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return None
    return float(value)
