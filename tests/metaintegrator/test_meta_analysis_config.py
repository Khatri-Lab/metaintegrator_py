"""
Unit tests for ``meta_analysis_config.py``.

Covers:
- ``resolve_presence_threshold()``  count mode, percent mode,
  rounding rule (ceiling), clamping, and error handling.
- ``validate_config()``'s new ``filter.number_studies_thresh`` /
  ``filter.presence_threshold_mode`` / ``filter.presence_threshold_value``
  validation.
- Backward compatibility: a config written before ``presence_threshold_*``
  keys to support gene presence at datasets was implemented. Still validates
  and defaults exactly as it always has.
"""

import os
import sys

import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.meta_analysis_config import (
    DEFAULTS,
    resolve_presence_threshold,
    validate_config,
)

# ===================================================================
# resolve_presence_threshold: "count" mode
# ===================================================================


class TestResolvePresenceThresholdCountMode:
    """Tests for resolve_presence_threshold() in "count" mode."""

    def test_plain_count_passes_through(self):
        """An in-range integer count passes through unchanged."""
        assert resolve_presence_threshold(mode="count", value=3, n_datasets=4) == 3

    def test_count_is_clamped_to_n_datasets(self):
        """A count above n_datasets is clamped down to n_datasets."""
        # Can't require presence in more datasets than exist.
        assert resolve_presence_threshold(mode="count", value=99, n_datasets=4) == 4

    def test_fractional_count_rounds_up(self):
        """A fractional count value rounds up to the next integer."""
        assert resolve_presence_threshold(mode="count", value=2.1, n_datasets=4) == 3

    def test_count_below_one_raises(self):
        """A count value below 1 raises ValueError."""
        with pytest.raises(ValueError, match="value must be >= 1"):
            resolve_presence_threshold(mode="count", value=0, n_datasets=4)

    def test_count_of_exactly_one_datasets_total(self):
        """With exactly one dataset total, a count of 1 resolves to 1."""
        assert resolve_presence_threshold(mode="count", value=1, n_datasets=1) == 1


# ===================================================================
# resolve_presence_threshold: "percent" mode
# ===================================================================


class TestResolvePresenceThresholdPercentMode:
    """Tests for resolve_presence_threshold() in "percent" mode."""

    def test_exact_division_matches_req1_worked_example(self):
        """The exact worked example from REQ-1.1: 75% of 4 datasets is 3."""
        # REQ-1.1's own example: 4 datasets, 75% -> exactly 3.
        assert resolve_presence_threshold(mode="percent", value=75, n_datasets=4) == 3

    def test_non_exact_division_rounds_up(self):
        """A non-exact percentage rounds up (ceiling), not down or to nearest."""
        # 75% of 3 = 2.25 -> ceil -> 3 (the "round up / ceiling" decision).
        assert resolve_presence_threshold(mode="percent", value=75, n_datasets=3) == 3

    def test_small_percentage_still_requires_at_least_one(self):
        """A very small percentage still resolves to at least 1 dataset."""
        assert resolve_presence_threshold(mode="percent", value=1, n_datasets=10) == 1

    def test_hundred_percent_requires_every_dataset(self):
        """100% resolves to every dataset."""
        assert resolve_presence_threshold(mode="percent", value=100, n_datasets=5) == 5

    def test_zero_percent_raises(self):
        """0% is out of the valid (0, 100] range and raises ValueError."""
        with pytest.raises(ValueError, match=r"value must be in \(0, 100\]"):
            resolve_presence_threshold(mode="percent", value=0, n_datasets=4)

    def test_over_hundred_percent_raises(self):
        """A percentage above 100 raises ValueError."""
        with pytest.raises(ValueError, match=r"value must be in \(0, 100\]"):
            resolve_presence_threshold(mode="percent", value=101, n_datasets=4)


# ===================================================================
# resolve_presence_threshold: shared behavior
# ===================================================================


class TestResolvePresenceThresholdSharedBehaviour:
    """Tests for resolve_presence_threshold() behavior shared by both modes."""

    def test_unknown_mode_raises(self):
        """A mode other than "count" or "percent" raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            resolve_presence_threshold(mode="bogus", value=1, n_datasets=4)

    def test_zero_datasets_degenerates_to_one(self):
        """With n_datasets=0, both modes degenerate to a threshold of 1."""
        # Defensive: nothing to compute a percentage/count against.
        assert resolve_presence_threshold(mode="count", value=3, n_datasets=0) == 1
        assert resolve_presence_threshold(mode="percent", value=75, n_datasets=0) == 1

    def test_result_never_exceeds_n_datasets(self):
        """The resolved threshold never exceeds n_datasets, in either mode."""
        for mode, value in [("count", 50), ("percent", 100)]:
            resolved = resolve_presence_threshold(mode=mode, value=value, n_datasets=6)
            assert resolved <= 6

    def test_result_is_always_at_least_one(self):
        """The resolved threshold is never rounded down to 0."""
        resolved = resolve_presence_threshold(mode="percent", value=1, n_datasets=1000)
        assert resolved >= 1


# ===================================================================
# validate_config: presence-threshold fields
# ===================================================================


def _base_raw() -> dict:
    """Build the minimal valid raw config: dataset IDs plus disease-term labeling."""
    return {
        "datasets": {"gse_ids": ["GSE11909"]},
        "labeling": {"GSE11909": {"disease_terms": ["SLE"]}},
    }


class TestValidateConfigPresenceThreshold:
    """Tests for validate_config()'s presence-threshold field validation."""

    def test_defaults_are_backward_compatible(self):
        """A pre-REQ-1 config with no filter section validates with the legacy defaults."""
        # A config with no filter section at all -- exactly what every
        # pre-REQ-1 config looks like -- must still validate cleanly and
        # resolve to the unchanged legacy behavior.
        config, errors = validate_config(_base_raw())
        assert errors == []
        assert config["filter"]["number_studies_thresh"] == 1
        assert config["filter"]["presence_threshold_mode"] == "count"
        assert config["filter"]["presence_threshold_value"] is None

    def test_valid_percent_mode_passes(self):
        """A valid percent-mode filter section validates cleanly and round-trips."""
        raw = _base_raw()
        raw["filter"] = {
            "presence_threshold_mode": "percent",
            "presence_threshold_value": 75,
        }
        config, errors = validate_config(raw)
        assert errors == []
        assert config["filter"]["presence_threshold_mode"] == "percent"
        assert config["filter"]["presence_threshold_value"] == 75

    def test_invalid_mode_is_rejected(self):
        """An unrecognized presence_threshold_mode produces a validation error."""
        raw = _base_raw()
        raw["filter"] = {"presence_threshold_mode": "bogus"}
        _, errors = validate_config(raw)
        assert any("presence_threshold_mode" in e for e in errors)

    def test_percent_value_out_of_range_is_rejected(self):
        """A presence_threshold_value above 100 in percent mode produces a validation error."""
        raw = _base_raw()
        raw["filter"] = {
            "presence_threshold_mode": "percent",
            "presence_threshold_value": 150,
        }
        _, errors = validate_config(raw)
        assert any("presence_threshold_value" in e for e in errors)

    def test_count_value_below_one_is_rejected_when_mode_is_count(self):
        """A presence_threshold_value of 0 in count mode produces a validation error."""
        raw = _base_raw()
        raw["filter"] = {
            "presence_threshold_mode": "count",
            "presence_threshold_value": 0,
        }
        _, errors = validate_config(raw)
        assert any("presence_threshold_value" in e for e in errors)

    def test_number_studies_thresh_below_one_is_rejected(self):
        """A number_studies_thresh of 0 produces a validation error."""
        raw = _base_raw()
        raw["filter"] = {"number_studies_thresh": 0}
        _, errors = validate_config(raw)
        assert any("number_studies_thresh" in e for e in errors)

    def test_number_studies_thresh_non_integer_is_rejected(self):
        """A non-integer number_studies_thresh produces a validation error."""
        raw = _base_raw()
        raw["filter"] = {"number_studies_thresh": 2.5}
        _, errors = validate_config(raw)
        assert any("number_studies_thresh" in e for e in errors)


class TestDefaultsUnchangedShape(object):
    """Guards against accidentally renaming/removing the legacy key while
    adding the new ones -- DEFAULTS is read by every caller that doesn't
    override `filter`."""

    def test_legacy_key_still_present(self):
        """The pre-REQ-1 number_studies_thresh default is still present and unchanged."""
        assert DEFAULTS["filter"]["number_studies_thresh"] == 1

    def test_new_keys_present_with_inert_defaults(self):
        """The new presence_threshold_* keys default to inert (no-op) values."""
        assert DEFAULTS["filter"]["presence_threshold_mode"] == "count"
        assert DEFAULTS["filter"]["presence_threshold_value"] is None
