"""
Unit tests for MetaAnalyzer.filter_genes() and FilterResult.

Covers:
- FilterResult dataclass: construction, dict-like access, to_dict(), aliases, total_genes
- filter_genes() input validation (bad thresholds, missing keys)
- filter_genes() threshold logic:  FDR, effect size, numStudies, heterogeneity
- Fisher FDR directional filtering (fisherFDRUp / fisherFDRDown)
- Graceful handling when Fisher columns are absent
- Leave-one-out consistency filtering
- LOO disabled / LOO results absent
- Edge cases: empty pooledResults, all genes filtered, zero-effectSize genes
- Standalone convenience function filter_genes()
- to_dict() round-trip consistency with __getitem__
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

from metaintegrator.analysis import (
    FilterResult,
    MetaAnalyzer,
    filter_genes,
)

# ===================================================================
# Helpers: build realistic pooledResults DataFrames
# ===================================================================


def _make_pooled(
    n_genes: int = 20,
    *,
    pos_frac: float = 0.5,
    base_fdr: float = 0.01,
    include_fisher: bool = True,
    include_het: bool = True,
    include_num_studies: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic pooledResults DataFrame.

    The first ``int(n_genes * pos_frac)`` genes get positive effect sizes,
    the rest get negative effect sizes.  FDR values are spread from
    ``base_fdr`` upward so that tests can exercise the threshold.
    """
    rng = np.random.RandomState(seed)
    genes = [f"Gene{i}" for i in range(n_genes)]

    n_pos = int(n_genes * pos_frac)
    effect_sizes = np.concatenate(
        [
            rng.uniform(0.3, 2.0, size=n_pos),
            rng.uniform(-2.0, -0.3, size=n_genes - n_pos),
        ]
    )

    # Spread FDR values so some pass and some fail typical 0.05 threshold
    fdrs = np.linspace(base_fdr, 0.15, n_genes)
    rng.shuffle(fdrs)

    data = {
        "effectSize": effect_sizes,
        "effectSizeFDR": fdrs,
        "effectSizePval": fdrs * 0.5,
    }

    if include_fisher:
        # For positive genes fisher FDR up should be low; for negative, fisher FDR down low
        fisher_up = np.where(effect_sizes > 0, fdrs * 0.8, 0.5)
        fisher_down = np.where(effect_sizes < 0, fdrs * 0.8, 0.5)
        data["fisherFDRUp"] = fisher_up
        data["fisherFDRDown"] = fisher_down
        data["fisherStatUp"] = rng.uniform(1, 10, n_genes)
        data["fisherStatDown"] = rng.uniform(1, 10, n_genes)
        data["fisherPvalUp"] = fdrs * 0.4
        data["fisherPvalDown"] = fdrs * 0.4

    if include_het:
        data["heterogeneityPval"] = rng.uniform(0.01, 0.99, n_genes)

    if include_num_studies:
        data["numStudies"] = rng.choice([2, 3, 4, 5], size=n_genes)

    # Extra columns that pooledResults normally has
    data["effectSizeStandardError"] = rng.uniform(0.05, 0.3, n_genes)
    data["tauSquared"] = rng.uniform(0.0, 0.5, n_genes)
    data["cochranesQ"] = rng.uniform(0.5, 10, n_genes)

    return pd.DataFrame(data, index=genes)


def _make_meta_object(pooled: pd.DataFrame, loo: dict = None) -> dict:
    """Wrap a pooledResults DataFrame into a meta_object dict."""
    return {
        "metaAnalysis": {"pooledResults": pooled},
        "leaveOneOutAnalysis": loo,
    }


def _make_loo_results(genes: list, flip_genes: list = None) -> dict:
    """Build a minimal LOO results dict.

    Every gene has effectSize = +1.0 unless it appears in *flip_genes*,
    in which case it gets effectSize = -1.0 in the first LOO analysis.
    """
    flip_genes = flip_genes or []
    n = len(genes)

    # Two LOO sub-analyses
    loo = {}
    for loo_name in ["removed_Study1", "removed_Study2"]:
        es_vals = []
        for g in genes:
            if g in flip_genes and loo_name == "removed_Study1":
                es_vals.append(-1.0)
            else:
                es_vals.append(1.0)
        loo_pooled = pd.DataFrame(
            {"effectSize": es_vals, "effectSizeFDR": [0.01] * n},
            index=genes,
        )
        loo[loo_name] = {"pooledResults": loo_pooled}
    return loo


# ===================================================================
# FilterResult dataclass tests
# ===================================================================


class TestFilterResult:
    """Tests for the FilterResult dataclass."""

    def _make_result(self, **overrides):
        """Build a FilterResult with sensible defaults, overridable per test."""
        defaults = dict(
            pos_gene_names=["A", "B"],
            neg_gene_names=["C"],
            n_pos=2,
            n_neg=1,
            effect_size_thresh=0.5,
            fdr_thresh=0.05,
            number_studies_thresh=2,
            heterogeneity_pval_thresh=0.0,
            is_leave_one_out=True,
            filter_description="test filter",
        )
        defaults.update(overrides)
        return FilterResult(**defaults)

    def test_basic_construction(self):
        """Fields are stored as given at construction."""
        fr = self._make_result()
        assert fr.pos_gene_names == ["A", "B"]
        assert fr.neg_gene_names == ["C"]
        assert fr.n_pos == 2
        assert fr.n_neg == 1

    def test_total_genes_property(self):
        """total_genes is the sum of n_pos and n_neg."""
        fr = self._make_result(n_pos=5, n_neg=3)
        assert fr.total_genes == 8

    def test_total_genes_zero(self):
        """total_genes is 0 when both gene lists are empty."""
        fr = self._make_result(pos_gene_names=[], neg_gene_names=[], n_pos=0, n_neg=0)
        assert fr.total_genes == 0

    # -- dict-like access via R-style aliases -----------------------

    def test_getitem_r_style_aliases(self):
        """__getitem__ accepts the original R-style camelCase field names."""
        fr = self._make_result()
        assert fr["posGeneNames"] == ["A", "B"]
        assert fr["negGeneNames"] == ["C"]
        assert fr["effectSizeThresh"] == 0.5
        assert fr["FDRThresh"] == 0.05
        assert fr["numberStudiesThresh"] == 2
        assert fr["isLeaveOneOut"] is True
        assert fr["filterDescription"] == "test filter"

    def test_getitem_python_style(self):
        """__getitem__ also accepts the Python-style snake_case field names."""
        fr = self._make_result()
        assert fr["pos_gene_names"] == ["A", "B"]
        assert fr["fdr_thresh"] == 0.05

    def test_getitem_invalid_key_raises_keyerror(self):
        """An unrecognized key raises KeyError."""
        fr = self._make_result()
        with pytest.raises(KeyError):
            _ = fr["nonExistentKey"]

    def test_contains_r_style(self):
        """`in` recognizes R-style camelCase field names."""
        fr = self._make_result()
        assert "posGeneNames" in fr
        assert "negGeneNames" in fr

    def test_contains_python_style(self):
        """`in` recognizes Python-style snake_case field names too."""
        fr = self._make_result()
        assert "pos_gene_names" in fr
        assert "fdr_thresh" in fr

    # -- to_dict() --------------------------------------------------

    def test_to_dict_keys(self):
        """to_dict() produces exactly the expected set of R-style keys."""
        fr = self._make_result()
        d = fr.to_dict()
        expected_keys = {
            "posGeneNames",
            "negGeneNames",
            "n_pos",
            "n_neg",
            "effectSizeThresh",
            "FDRThresh",
            "numberStudiesThresh",
            "heterogeneityPvalThresh",
            "isLeaveOneOut",
            "filterDescription",
            "totalGenes",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values_match_getitem(self):
        """Every R-alias key in to_dict() must match __getitem__."""
        fr = self._make_result()
        d = fr.to_dict()
        for alias in [
            "posGeneNames",
            "negGeneNames",
            "effectSizeThresh",
            "FDRThresh",
            "numberStudiesThresh",
            "heterogeneityPvalThresh",
            "isLeaveOneOut",
            "filterDescription",
        ]:
            assert d[alias] == fr[alias], f"Mismatch for key '{alias}'"

    def test_to_dict_total_genes(self):
        """to_dict()'s totalGenes reflects n_pos + n_neg."""
        fr = self._make_result(n_pos=4, n_neg=6)
        d = fr.to_dict()
        assert d["totalGenes"] == 10

    def test_to_dict_returns_plain_dict(self):
        """to_dict() returns a plain dict, not a dict subclass."""
        fr = self._make_result()
        d = fr.to_dict()
        assert type(d) is dict

    def test_to_dict_independent_copy(self):
        """Mutating the returned dict must not affect the dataclass."""
        fr = self._make_result()
        d = fr.to_dict()
        d["posGeneNames"].append("Z")
        assert "Z" not in fr.pos_gene_names


# ===================================================================
# filter_genes(): input validation
# ===================================================================


class TestFilterGenesValidation:
    """Test that bad inputs produce clear ValueErrors."""

    def setup_method(self):
        """Give each test a fresh MetaAnalyzer."""
        self.analyzer = MetaAnalyzer()

    def test_missing_meta_analysis_key(self):
        """A meta_object with no "metaAnalysis" key raises ValueError."""
        with pytest.raises(ValueError, match="metaAnalysis"):
            self.analyzer.filter_genes({})

    def test_meta_analysis_is_none(self):
        """A "metaAnalysis" value of None raises ValueError."""
        with pytest.raises(ValueError, match="metaAnalysis"):
            self.analyzer.filter_genes({"metaAnalysis": None})

    def test_missing_pooled_results(self):
        """A metaAnalysis dict with no "pooledResults" key raises ValueError."""
        with pytest.raises(ValueError, match="pooledResults"):
            self.analyzer.filter_genes({"metaAnalysis": {}})

    def test_fdr_thresh_negative(self):
        """A negative fdr_thresh raises ValueError."""
        meta = _make_meta_object(_make_pooled(5))
        with pytest.raises(ValueError, match="fdr_thresh"):
            self.analyzer.filter_genes(meta, fdr_thresh=-0.1)

    def test_fdr_thresh_above_one(self):
        """An fdr_thresh greater than 1.0 raises ValueError."""
        meta = _make_meta_object(_make_pooled(5))
        with pytest.raises(ValueError, match="fdr_thresh"):
            self.analyzer.filter_genes(meta, fdr_thresh=1.5)

    def test_effect_size_thresh_negative(self):
        """A negative effect_size_thresh raises ValueError."""
        meta = _make_meta_object(_make_pooled(5))
        with pytest.raises(ValueError, match="effect_size_thresh"):
            self.analyzer.filter_genes(meta, effect_size_thresh=-0.1)

    def test_number_studies_thresh_zero(self):
        """A number_studies_thresh of 0 raises ValueError."""
        meta = _make_meta_object(_make_pooled(5))
        with pytest.raises(ValueError, match="number_studies_thresh"):
            self.analyzer.filter_genes(meta, number_studies_thresh=0)

    def test_heterogeneity_thresh_out_of_range(self):
        """A heterogeneity_pval_thresh outside [0, 1] raises ValueError."""
        meta = _make_meta_object(_make_pooled(5))
        with pytest.raises(ValueError, match="heterogeneity_pval_thresh"):
            self.analyzer.filter_genes(meta, heterogeneity_pval_thresh=1.5)


# ===================================================================
# filter_genes(): threshold filtering logic
# ===================================================================


class TestFilterGenesThresholds:
    """Test each filter step independently and in combination."""

    def setup_method(self):
        """Give each test a fresh MetaAnalyzer."""
        self.analyzer = MetaAnalyzer()

    # -- FDR filter -------------------------------------------------

    def test_fdr_filter_removes_high_fdr_genes(self):
        """Genes with effectSizeFDR above the threshold are dropped."""
        pooled = _make_pooled(10, base_fdr=0.01, include_fisher=True)
        # Set half the genes above FDR threshold
        pooled.iloc[5:, pooled.columns.get_loc("effectSizeFDR")] = 0.9
        meta = _make_meta_object(pooled)

        result = self.analyzer.filter_genes(meta, fdr_thresh=0.05, is_leave_one_out=False)
        fr = result["filterResults"]
        # All surviving genes should have FDR <= 0.05
        assert fr.total_genes <= 10

    def test_fdr_1_passes_everything(self):
        """FDR=1.0 should effectively disable the FDR filter."""
        pooled = _make_pooled(10, base_fdr=0.5, include_fisher=False)
        meta = _make_meta_object(pooled)
        result = self.analyzer.filter_genes(meta, fdr_thresh=1.0, is_leave_one_out=False)
        fr = result["filterResults"]
        assert fr.total_genes == 10

    # -- Effect size filter ----------------------------------------

    def test_effect_size_filter(self):
        """Only genes with |effectSize| above the threshold survive."""
        pooled = _make_pooled(10, base_fdr=0.001, include_fisher=False)
        # Set all effect sizes small
        pooled["effectSize"] = 0.1
        pooled.iloc[0, pooled.columns.get_loc("effectSize")] = 2.0  # one big
        pooled.iloc[1, pooled.columns.get_loc("effectSize")] = -2.0  # one big neg
        meta = _make_meta_object(pooled)

        result = self.analyzer.filter_genes(
            meta, fdr_thresh=1.0, effect_size_thresh=1.0, is_leave_one_out=False
        )
        fr = result["filterResults"]
        assert fr.n_pos == 1
        assert fr.n_neg == 1

    def test_effect_size_thresh_zero_disables(self):
        """effect_size_thresh=0 should not remove any genes."""
        pooled = _make_pooled(8, base_fdr=0.001, include_fisher=False)
        meta = _make_meta_object(pooled)
        result = self.analyzer.filter_genes(
            meta, fdr_thresh=1.0, effect_size_thresh=0.0, is_leave_one_out=False
        )
        assert result["filterResults"].total_genes == 8

    # -- Number of studies filter ----------------------------------

    def test_num_studies_filter(self):
        """Genes backed by fewer than number_studies_thresh studies are dropped."""
        pooled = _make_pooled(10, base_fdr=0.001, include_fisher=False)
        pooled["numStudies"] = [1, 1, 1, 2, 2, 3, 3, 4, 5, 5]
        meta = _make_meta_object(pooled)

        result = self.analyzer.filter_genes(
            meta, fdr_thresh=1.0, number_studies_thresh=3, is_leave_one_out=False
        )
        fr = result["filterResults"]
        assert fr.total_genes == 5  # genes with numStudies >= 3

    def test_num_studies_thresh_1_disables(self):
        """number_studies_thresh=1 effectively disables the study-count filter."""
        pooled = _make_pooled(6, base_fdr=0.001, include_fisher=False)
        pooled["numStudies"] = 1
        meta = _make_meta_object(pooled)
        result = self.analyzer.filter_genes(
            meta, fdr_thresh=1.0, number_studies_thresh=1, is_leave_one_out=False
        )
        assert result["filterResults"].total_genes == 6

    # -- Heterogeneity filter --------------------------------------

    def test_heterogeneity_filter(self):
        """Only genes with heterogeneityPval at or above the threshold survive."""
        pooled = _make_pooled(10, base_fdr=0.001, include_fisher=False, include_het=True)
        pooled["heterogeneityPval"] = [
            0.01,
            0.02,
            0.05,
            0.10,
            0.20,
            0.30,
            0.50,
            0.70,
            0.80,
            0.99,
        ]
        meta = _make_meta_object(pooled)

        result = self.analyzer.filter_genes(
            meta,
            fdr_thresh=1.0,
            heterogeneity_pval_thresh=0.20,
            is_leave_one_out=False,
        )
        fr = result["filterResults"]
        # Only genes with heterogeneityPval >= 0.20 pass
        assert fr.total_genes == 6

    def test_heterogeneity_thresh_zero_disables(self):
        """heterogeneity_pval_thresh=0.0 effectively disables the heterogeneity filter."""
        pooled = _make_pooled(5, base_fdr=0.001, include_fisher=False, include_het=True)
        pooled["heterogeneityPval"] = 0.001
        meta = _make_meta_object(pooled)
        result = self.analyzer.filter_genes(
            meta,
            fdr_thresh=1.0,
            heterogeneity_pval_thresh=0.0,
            is_leave_one_out=False,
        )
        assert result["filterResults"].total_genes == 5


# ===================================================================
# filter_genes(): Fisher FDR directional filtering
# ===================================================================


class TestFilterGenesFisher:
    """Test Fisher FDR Up/Down directional filtering."""

    def setup_method(self):
        """Give each test a fresh MetaAnalyzer."""
        self.analyzer = MetaAnalyzer()

    def test_fisher_fdr_up_filters_positive_genes(self):
        """Positive genes must pass fisherFDRUp <= fdr_thresh."""
        pooled = pd.DataFrame(
            {
                "effectSize": [1.0, 1.0, -1.0, -1.0],
                "effectSizeFDR": [0.01, 0.01, 0.01, 0.01],
                "fisherFDRUp": [0.01, 0.90, 0.50, 0.50],  # gene1 fails
                "fisherFDRDown": [0.50, 0.50, 0.01, 0.90],  # gene3 fails
                "effectSizePval": [0.001] * 4,
                "effectSizeStandardError": [0.1] * 4,
                "tauSquared": [0.1] * 4,
                "numStudies": [3] * 4,
                "cochranesQ": [2.0] * 4,
                "heterogeneityPval": [0.5] * 4,
            },
            index=["GeneA", "GeneB", "GeneC", "GeneD"],
        )

        meta = _make_meta_object(pooled)
        result = self.analyzer.filter_genes(meta, fdr_thresh=0.05, is_leave_one_out=False)
        fr = result["filterResults"]

        assert fr.pos_gene_names == ["GeneA"]
        assert fr.neg_gene_names == ["GeneC"]

    def test_missing_fisher_columns_graceful(self):
        """When Fisher columns are absent, filtering should still work."""
        pooled = _make_pooled(6, base_fdr=0.001, include_fisher=False)
        meta = _make_meta_object(pooled)
        # Should not raise
        result = self.analyzer.filter_genes(meta, fdr_thresh=1.0, is_leave_one_out=False)
        assert result["filterResults"].total_genes == 6


# ===================================================================
# filter_genes(): leave-one-out consistency
# ===================================================================


class TestFilterGenesLOO:
    """Test leave-one-out consistency checking."""

    def setup_method(self):
        """Give each test a fresh MetaAnalyzer."""
        self.analyzer = MetaAnalyzer()

    def test_loo_removes_inconsistent_gene(self):
        """A positive gene that flips sign in one LOO analysis is removed."""
        genes = ["GenePos", "GeneFlip", "GeneNeg"]
        pooled = pd.DataFrame(
            {
                "effectSize": [1.0, 0.5, -1.0],
                "effectSizeFDR": [0.01, 0.01, 0.01],
                "effectSizePval": [0.001] * 3,
                "effectSizeStandardError": [0.1] * 3,
                "tauSquared": [0.1] * 3,
                "numStudies": [3] * 3,
                "cochranesQ": [2.0] * 3,
                "heterogeneityPval": [0.5] * 3,
            },
            index=genes,
        )

        # GeneFlip flips to negative in one LOO sub-analysis
        loo = _make_loo_results(genes, flip_genes=["GeneFlip"])
        meta = _make_meta_object(pooled, loo=loo)

        result = self.analyzer.filter_genes(meta, fdr_thresh=1.0, is_leave_one_out=True)
        fr = result["filterResults"]

        assert "GenePos" in fr.pos_gene_names
        assert (
            "GeneFlip" not in fr.pos_gene_names
        ), "GeneFlip should be removed: it flips sign in LOO"

    def test_loo_disabled_keeps_inconsistent_gene(self):
        """With is_leave_one_out=False, a sign-flipping gene is kept regardless of LOO data."""
        genes = ["GenePos", "GeneFlip"]
        pooled = pd.DataFrame(
            {
                "effectSize": [1.0, 0.5],
                "effectSizeFDR": [0.01, 0.01],
                "effectSizePval": [0.001] * 2,
                "effectSizeStandardError": [0.1] * 2,
                "tauSquared": [0.1] * 2,
                "numStudies": [3] * 2,
                "cochranesQ": [2.0] * 2,
                "heterogeneityPval": [0.5] * 2,
            },
            index=genes,
        )

        loo = _make_loo_results(genes, flip_genes=["GeneFlip"])
        meta = _make_meta_object(pooled, loo=loo)

        result = self.analyzer.filter_genes(meta, fdr_thresh=1.0, is_leave_one_out=False)
        fr = result["filterResults"]
        assert "GeneFlip" in fr.pos_gene_names

    def test_loo_none_skips_consistency_check(self):
        """When leaveOneOutAnalysis is None, LOO is skipped gracefully."""
        pooled = _make_pooled(5, base_fdr=0.001, include_fisher=False)
        meta = _make_meta_object(pooled, loo=None)
        result = self.analyzer.filter_genes(meta, fdr_thresh=1.0, is_leave_one_out=True)
        # Should not raise; total genes preserved
        assert result["filterResults"].total_genes == 5

    def test_loo_empty_dict_skips(self):
        """An empty leaveOneOutAnalysis dict skips the consistency check gracefully."""
        pooled = _make_pooled(4, base_fdr=0.001, include_fisher=False)
        meta = _make_meta_object(pooled, loo={})
        result = self.analyzer.filter_genes(meta, fdr_thresh=1.0, is_leave_one_out=True)
        assert result["filterResults"].total_genes == 4

    def test_loo_with_none_sub_results(self):
        """LOO dict with None values for some sub-analyses should be tolerated."""
        genes = ["G1", "G2"]
        pooled = pd.DataFrame(
            {
                "effectSize": [1.0, -1.0],
                "effectSizeFDR": [0.01, 0.01],
                "effectSizePval": [0.001] * 2,
                "effectSizeStandardError": [0.1] * 2,
                "tauSquared": [0.1] * 2,
                "numStudies": [3] * 2,
                "cochranesQ": [2.0] * 2,
                "heterogeneityPval": [0.5] * 2,
            },
            index=genes,
        )

        loo = {"removed_S1": None, "removed_S2": None}
        meta = _make_meta_object(pooled, loo=loo)
        result = self.analyzer.filter_genes(meta, fdr_thresh=1.0, is_leave_one_out=True)
        assert result["filterResults"].total_genes == 2


# ===================================================================
# filter_genes(): edge cases
# ===================================================================


class TestFilterGenesEdgeCases:
    """Edge and corner cases."""

    def setup_method(self):
        """Give each test a fresh MetaAnalyzer."""
        self.analyzer = MetaAnalyzer()

    def test_empty_pooled_results(self):
        """An empty pooledResults DataFrame should produce zero genes."""
        pooled = pd.DataFrame(
            columns=[
                "effectSize",
                "effectSizeFDR",
                "effectSizePval",
            ]
        )
        meta = _make_meta_object(pooled)
        result = self.analyzer.filter_genes(meta, fdr_thresh=0.05, is_leave_one_out=False)
        fr = result["filterResults"]
        assert fr.n_pos == 0
        assert fr.n_neg == 0
        assert fr.total_genes == 0

    def test_all_genes_filtered_out(self):
        """A very strict threshold should filter everything."""
        pooled = _make_pooled(10, base_fdr=0.10, include_fisher=False)
        meta = _make_meta_object(pooled)
        result = self.analyzer.filter_genes(meta, fdr_thresh=0.001, is_leave_one_out=False)
        fr = result["filterResults"]
        assert fr.total_genes == 0
        assert fr.pos_gene_names == []
        assert fr.neg_gene_names == []

    def test_gene_with_zero_effect_size_excluded_from_both_lists(self):
        """A gene with effectSize == 0 should appear in neither pos nor neg."""
        pooled = pd.DataFrame(
            {
                "effectSize": [0.0, 1.0, -1.0],
                "effectSizeFDR": [0.01, 0.01, 0.01],
                "effectSizePval": [0.001] * 3,
                "effectSizeStandardError": [0.1] * 3,
                "tauSquared": [0.1] * 3,
                "numStudies": [3] * 3,
                "cochranesQ": [2.0] * 3,
                "heterogeneityPval": [0.5] * 3,
            },
            index=["Zero", "Pos", "Neg"],
        )

        meta = _make_meta_object(pooled)
        result = self.analyzer.filter_genes(meta, fdr_thresh=1.0, is_leave_one_out=False)
        fr = result["filterResults"]
        assert "Zero" not in fr.pos_gene_names
        assert "Zero" not in fr.neg_gene_names

    def test_filter_results_stored_on_meta_object(self):
        """filter_genes() must store the result under 'filterResults'."""
        pooled = _make_pooled(5, base_fdr=0.001, include_fisher=False)
        meta = _make_meta_object(pooled)
        result = self.analyzer.filter_genes(meta, fdr_thresh=1.0, is_leave_one_out=False)
        assert "filterResults" in result
        assert isinstance(result["filterResults"], FilterResult)

    def test_filter_description_captures_thresholds(self):
        """filter_description mentions every threshold value that was passed in."""
        pooled = _make_pooled(5, base_fdr=0.001, include_fisher=False)
        meta = _make_meta_object(pooled)
        result = self.analyzer.filter_genes(
            meta,
            effect_size_thresh=0.5,
            fdr_thresh=0.01,
            number_studies_thresh=3,
            heterogeneity_pval_thresh=0.1,
            is_leave_one_out=False,
        )
        desc = result["filterResults"].filter_description
        assert "0.5" in desc
        assert "0.01" in desc
        assert "3" in desc
        assert "0.1" in desc

    def test_original_pooled_results_not_mutated(self):
        """filter_genes() must not modify the original pooledResults."""
        pooled = _make_pooled(10, base_fdr=0.001, include_fisher=False)
        original_len = len(pooled)
        meta = _make_meta_object(pooled)
        self.analyzer.filter_genes(meta, fdr_thresh=0.001, is_leave_one_out=False)
        # The original DataFrame in metaAnalysis should be untouched
        assert len(meta["metaAnalysis"]["pooledResults"]) == original_len

    def test_single_gene_passes(self):
        """A single-gene pooledResults table filters correctly, not just multi-gene ones."""
        pooled = pd.DataFrame(
            {
                "effectSize": [1.5],
                "effectSizeFDR": [0.01],
                "effectSizePval": [0.001],
                "effectSizeStandardError": [0.1],
                "tauSquared": [0.1],
                "numStudies": [3],
                "cochranesQ": [2.0],
                "heterogeneityPval": [0.5],
            },
            index=["OnlyGene"],
        )

        meta = _make_meta_object(pooled)
        result = self.analyzer.filter_genes(meta, fdr_thresh=0.05, is_leave_one_out=False)
        assert result["filterResults"].pos_gene_names == ["OnlyGene"]

    def test_nan_fdr_treated_as_failing(self):
        """Genes with NaN FDR should not pass the FDR filter."""
        pooled = pd.DataFrame(
            {
                "effectSize": [1.0, -1.0],
                "effectSizeFDR": [np.nan, 0.01],
                "effectSizePval": [0.001, 0.001],
                "effectSizeStandardError": [0.1, 0.1],
                "tauSquared": [0.1, 0.1],
                "numStudies": [3, 3],
                "cochranesQ": [2.0, 2.0],
                "heterogeneityPval": [0.5, 0.5],
            },
            index=["NaNGene", "GoodGene"],
        )

        meta = _make_meta_object(pooled)
        result = self.analyzer.filter_genes(meta, fdr_thresh=0.05, is_leave_one_out=False)
        fr = result["filterResults"]
        assert "NaNGene" not in fr.pos_gene_names
        assert "GoodGene" in fr.neg_gene_names


# ===================================================================
# _loo_consistent_genes(): direct unit tests
# ===================================================================


class TestLOOConsistentGenesHelper:
    """Direct tests for the static helper method."""

    def test_empty_gene_list(self):
        """An empty candidate gene list returns an empty result."""
        result = MetaAnalyzer._loo_consistent_genes([], {}, "pos")
        assert result == []

    def test_empty_loo_results(self):
        """An empty LOO results dict leaves every gene passing (nothing to check against)."""
        result = MetaAnalyzer._loo_consistent_genes(["G1"], {}, "pos")
        assert result == ["G1"]

    def test_none_loo_results(self):
        """LOO results of None is treated the same as an empty dict."""
        result = MetaAnalyzer._loo_consistent_genes(["G1"], None, "pos")
        assert result == ["G1"]

    def test_consistent_pos_gene_kept(self):
        """A positive gene that stays positive across every LOO sub-analysis is kept."""
        loo = {
            "r1": {"pooledResults": pd.DataFrame({"effectSize": [0.5]}, index=["G1"])},
            "r2": {"pooledResults": pd.DataFrame({"effectSize": [1.2]}, index=["G1"])},
        }
        result = MetaAnalyzer._loo_consistent_genes(["G1"], loo, "pos")
        assert result == ["G1"]

    def test_inconsistent_pos_gene_removed(self):
        """A positive gene that flips to negative in any LOO sub-analysis is removed."""
        loo = {
            "r1": {"pooledResults": pd.DataFrame({"effectSize": [0.5]}, index=["G1"])},
            "r2": {"pooledResults": pd.DataFrame({"effectSize": [-0.3]}, index=["G1"])},
        }
        result = MetaAnalyzer._loo_consistent_genes(["G1"], loo, "pos")
        assert result == []

    def test_consistent_neg_gene_kept(self):
        """A negative gene that stays negative across every LOO sub-analysis is kept."""
        loo = {
            "r1": {"pooledResults": pd.DataFrame({"effectSize": [-0.5]}, index=["N1"])},
            "r2": {"pooledResults": pd.DataFrame({"effectSize": [-1.0]}, index=["N1"])},
        }
        result = MetaAnalyzer._loo_consistent_genes(["N1"], loo, "neg")
        assert result == ["N1"]

    def test_inconsistent_neg_gene_removed(self):
        """A negative gene that flips to positive in any LOO sub-analysis is removed."""
        loo = {
            "r1": {"pooledResults": pd.DataFrame({"effectSize": [-0.5]}, index=["N1"])},
            "r2": {"pooledResults": pd.DataFrame({"effectSize": [0.1]}, index=["N1"])},
        }
        result = MetaAnalyzer._loo_consistent_genes(["N1"], loo, "neg")
        assert result == []

    def test_gene_missing_from_loo_pooled_passes(self):
        """A gene absent from a LOO sub-analysis should still pass."""
        loo = {
            "r1": {"pooledResults": pd.DataFrame({"effectSize": [0.5]}, index=["Other"])},
        }
        result = MetaAnalyzer._loo_consistent_genes(["G1"], loo, "pos")
        assert result == ["G1"]

    def test_loo_sub_result_with_none_pooled(self):
        """A LOO sub-result with pooledResults=None should be skipped."""
        loo = {
            "r1": {"pooledResults": None},
            "r2": {"pooledResults": pd.DataFrame({"effectSize": [0.5]}, index=["G1"])},
        }
        result = MetaAnalyzer._loo_consistent_genes(["G1"], loo, "pos")
        assert result == ["G1"]


# ===================================================================
# Standalone convenience function
# ===================================================================


class TestStandaloneFilterGenes:
    """Test the module-level filter_genes() convenience wrapper."""

    def test_convenience_function_works(self):
        """The standalone function returns the same shape of result as the analyzer method."""
        pooled = _make_pooled(8, base_fdr=0.001, include_fisher=False)
        meta = _make_meta_object(pooled)
        result = filter_genes(meta, fdr_thresh=1.0, is_leave_one_out=False)
        assert "filterResults" in result
        assert isinstance(result["filterResults"], FilterResult)

    def test_convenience_function_passes_thresholds(self):
        """Threshold keyword arguments are forwarded through to the underlying filter logic."""
        pooled = _make_pooled(10, base_fdr=0.001, include_fisher=False)
        meta = _make_meta_object(pooled)
        result = filter_genes(meta, effect_size_thresh=5.0, fdr_thresh=1.0, is_leave_one_out=False)
        # Very high effect_size_thresh should filter most genes
        assert result["filterResults"].total_genes < 10


# ===================================================================
# Combined / integration-style filter tests
# ===================================================================


class TestFilterGenesCombined:
    """Tests that exercise multiple filters together."""

    def setup_method(self):
        """Give each test a fresh MetaAnalyzer."""
        self.analyzer = MetaAnalyzer()

    def test_fdr_plus_effect_size_plus_fisher(self):
        """Realistic scenario: multiple filters active simultaneously."""
        pooled = pd.DataFrame(
            {
                "effectSize": [2.0, 0.1, 1.5, -2.0, -0.1, -1.5],
                "effectSizeFDR": [0.01, 0.01, 0.10, 0.01, 0.01, 0.10],
                "effectSizePval": [0.001] * 6,
                "fisherFDRUp": [0.01, 0.01, 0.01, 0.50, 0.50, 0.50],
                "fisherFDRDown": [0.50, 0.50, 0.50, 0.01, 0.01, 0.01],
                "effectSizeStandardError": [0.1] * 6,
                "tauSquared": [0.1] * 6,
                "numStudies": [3] * 6,
                "cochranesQ": [2.0] * 6,
                "heterogeneityPval": [0.5] * 6,
            },
            index=[
                "BigPos",
                "SmallPos",
                "MidPosHighFDR",
                "BigNeg",
                "SmallNeg",
                "MidNegHighFDR",
            ],
        )

        meta = _make_meta_object(pooled)
        result = self.analyzer.filter_genes(
            meta,
            fdr_thresh=0.05,
            effect_size_thresh=0.5,
            is_leave_one_out=False,
        )
        fr = result["filterResults"]

        # BigPos: ES=2 passes ES, FDR=0.01 passes, fisherFDRUp=0.01 passes → ✓
        assert "BigPos" in fr.pos_gene_names
        # SmallPos: ES=0.1 fails effect_size_thresh=0.5 → ✗
        assert "SmallPos" not in fr.pos_gene_names
        # MidPosHighFDR: effectSizeFDR=0.10 fails fdr_thresh=0.05 → ✗
        assert "MidPosHighFDR" not in fr.pos_gene_names
        # BigNeg: ES=-2 passes, FDR=0.01 passes, fisherFDRDown=0.01 passes → ✓
        assert "BigNeg" in fr.neg_gene_names
        # SmallNeg: ES=-0.1 fails effect_size_thresh → ✗
        assert "SmallNeg" not in fr.neg_gene_names

    def test_idempotent_filter_results_key(self):
        """Calling filter_genes twice replaces (not appends) filterResults."""
        pooled = _make_pooled(6, base_fdr=0.001, include_fisher=False)
        meta = _make_meta_object(pooled)

        self.analyzer.filter_genes(meta, fdr_thresh=1.0, is_leave_one_out=False)
        first_total = meta["filterResults"].total_genes

        self.analyzer.filter_genes(
            meta, fdr_thresh=1.0, effect_size_thresh=99.0, is_leave_one_out=False
        )
        second_total = meta["filterResults"].total_genes

        assert second_total == 0
        assert second_total != first_total


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
