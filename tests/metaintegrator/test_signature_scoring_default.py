"""Unit tests for the default scoring method of ``SignatureScoringEngine``.

Since version 0.12.0 the engine delegates to ``ScoreCalculator`` (R's
``calculateScore()``) unless a caller asks for the legacy
``"mean_difference"`` method by name. These tests pin that contract.

The R-parity numbers themselves are checked in
``tests/parity/test_scoring_parity.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from metaintegrator.roc_analysis_engine import (
    DEFAULT_SCORING_METHOD,
    SCORING_METHOD_GEOMETRIC_MEAN_ZSCORE,
    SCORING_METHOD_MEAN_DIFFERENCE,
    SCORING_METHODS,
    SignatureScoringEngine,
    calculate_signature_scores,
)
from metaintegrator.score_calculation import ScoreCalculator

POS_GENES = ["G1", "G2"]
NEG_GENES = ["G3"]


@pytest.fixture
def dataset() -> dict:
    """A small dataset with negative values, like centered expression data.

    Negative values matter. They make the R-parity method shift the whole
    dataset before the geometric mean, which the legacy method never does.
    """
    rng = np.random.default_rng(7)
    expr = pd.DataFrame(
        rng.normal(loc=0.5, scale=2.0, size=(5, 12)),
        index=[f"probe{i}" for i in range(5)],
        columns=[f"S{i}" for i in range(12)],
    )
    return {"expr": expr, "keys": ["G1", "G2", "G3", "G4", "G1"]}


@pytest.fixture
def engine() -> SignatureScoringEngine:
    """A fresh engine."""
    return SignatureScoringEngine()


class TestDefaultMethod:
    """The default must be the R-parity method."""

    def test_constants(self) -> None:
        assert DEFAULT_SCORING_METHOD == SCORING_METHOD_GEOMETRIC_MEAN_ZSCORE
        assert set(SCORING_METHODS) == {
            SCORING_METHOD_GEOMETRIC_MEAN_ZSCORE,
            SCORING_METHOD_MEAN_DIFFERENCE,
        }

    def test_class_attributes_expose_the_default(self) -> None:
        # The web app reads DEFAULT_METHOD to tag stored ROC results.
        assert SignatureScoringEngine.DEFAULT_METHOD == DEFAULT_SCORING_METHOD
        assert SignatureScoringEngine.METHODS == SCORING_METHODS

    def test_default_equals_score_calculator(self, engine, dataset) -> None:
        scores = engine.calculate_scores(dataset["expr"], dataset["keys"], POS_GENES, NEG_GENES)
        expected = ScoreCalculator.calculate_score(
            {"posGeneNames": POS_GENES, "negGeneNames": NEG_GENES},
            dataset,
            suppress_messages=True,
        )
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)

    def test_explicit_method_equals_default(self, engine, dataset) -> None:
        default = engine.calculate_scores(dataset["expr"], dataset["keys"], POS_GENES, NEG_GENES)
        explicit = engine.calculate_scores(
            dataset["expr"],
            dataset["keys"],
            POS_GENES,
            NEG_GENES,
            method=SCORING_METHOD_GEOMETRIC_MEAN_ZSCORE,
        )
        np.testing.assert_array_equal(default, explicit)

    def test_default_is_z_scored(self, engine, dataset) -> None:
        scores = engine.calculate_scores(dataset["expr"], dataset["keys"], POS_GENES, NEG_GENES)
        assert scores.mean() == pytest.approx(0.0, abs=1e-12)
        assert scores.std(ddof=1) == pytest.approx(1.0, abs=1e-12)

    def test_default_differs_from_legacy(self, engine, dataset) -> None:
        """Guard against the default silently going back to the legacy method."""
        default = engine.calculate_scores(dataset["expr"], dataset["keys"], POS_GENES, NEG_GENES)
        legacy = engine.calculate_scores(
            dataset["expr"],
            dataset["keys"],
            POS_GENES,
            NEG_GENES,
            method=SCORING_METHOD_MEAN_DIFFERENCE,
        )
        assert not np.allclose(default, legacy)

    def test_convenience_function_uses_default(self, dataset) -> None:
        via_function = calculate_signature_scores(
            dataset["expr"], dataset["keys"], POS_GENES, NEG_GENES
        )
        via_engine = SignatureScoringEngine().calculate_scores(
            dataset["expr"], dataset["keys"], POS_GENES, NEG_GENES
        )
        np.testing.assert_array_equal(via_function, via_engine)


class TestInputs:
    """Input handling for the R-parity path."""

    def test_numpy_input_matches_dataframe(self, engine, dataset) -> None:
        from_df = engine.calculate_scores(dataset["expr"], dataset["keys"], POS_GENES, NEG_GENES)
        from_np = engine.calculate_scores(
            dataset["expr"].to_numpy(), dataset["keys"], POS_GENES, NEG_GENES
        )
        np.testing.assert_allclose(from_df, from_np, atol=1e-12)

    def test_keys_as_numpy_array(self, engine, dataset) -> None:
        scores = engine.calculate_scores(
            dataset["expr"], np.asarray(dataset["keys"]), POS_GENES, NEG_GENES
        )
        assert scores.shape == (12,)

    def test_keys_length_mismatch_raises(self, engine, dataset) -> None:
        with pytest.raises(ValueError, match="gene_keys has 4 entries"):
            engine.calculate_scores(dataset["expr"], dataset["keys"][:4], ["G1"], ["G3"])

    def test_unknown_method_raises(self, engine, dataset) -> None:
        with pytest.raises(ValueError, match="Unknown scoring method"):
            engine.calculate_scores(
                dataset["expr"], dataset["keys"], POS_GENES, NEG_GENES, method="weighted"
            )

    def test_no_signature_gene_found_returns_zeros(self, engine, dataset) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scores = engine.calculate_scores(dataset["expr"], dataset["keys"], ["NOPE"], [])
        assert any("No genes found" in str(w.message) for w in caught)
        np.testing.assert_array_equal(scores, np.zeros(12))

    def test_only_negative_genes_flip_sign(self, engine, dataset) -> None:
        """With no positive genes, R scores ``scale(-negScore)``."""
        neg_only = engine.calculate_scores(dataset["expr"], dataset["keys"], [], ["G3"])
        pos_only = engine.calculate_scores(dataset["expr"], dataset["keys"], ["G3"], [])
        np.testing.assert_allclose(neg_only, -pos_only, atol=1e-12)

    def test_input_is_not_modified(self, engine, dataset) -> None:
        before = dataset["expr"].copy()
        engine.calculate_scores(dataset["expr"], dataset["keys"], POS_GENES, NEG_GENES)
        pd.testing.assert_frame_equal(dataset["expr"], before)
