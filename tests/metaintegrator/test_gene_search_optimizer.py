"""
Unit tests for roc_prc_analysis module.
Tests ROC/PRC curve analysis, gene search optimization, and related functionality.
"""

import os
import sys
import warnings
from unittest import result
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from metaintegrator.gene_search_optimizer import (
    GeneSearchOptimizer,
    backward_search,
    forward_search,
)


class TestGeneSearchOptimizer:
    """Test suite for GeneSearchOptimizer class."""

    def setup_method(self):
        """Setup test data for gene search optimization."""
        np.random.seed(42)
        self.optimizer = GeneSearchOptimizer()

        # Create realistic test data with gene expression
        self.meta_object = {
            "originalData": {
                "discovery1": {
                    "formattedName": "Discovery Dataset 1",
                    "expr": pd.DataFrame(
                        np.random.randn(10, 20) + 2,  # 10 probes, 20 samples
                        index=[f"probe_{i}" for i in range(10)],
                        columns=[f"sample_{i}" for i in range(20)],
                    ),
                    "keys": [
                        "GENE1",
                        "GENE2",
                        "GENE3",
                        "GENE4",
                        "GENE5",
                        "GENE1",
                        "GENE2",
                        "GENE6",
                        "GENE7",
                        "GENE8",
                    ],  # Some genes have multiple probes
                    "class": np.array([0] * 10 + [1] * 10),  # Binary classification
                },
                "discovery2": {
                    "formattedName": "Discovery Dataset 2",
                    "expr": pd.DataFrame(
                        np.random.randn(10, 16) + 2,  # 10 probes, 16 samples
                        index=[f"probe_{i}" for i in range(10)],
                        columns=[f"sample_{i}" for i in range(16)],
                    ),
                    "keys": [
                        "GENE1",
                        "GENE2",
                        "GENE3",
                        "GENE4",
                        "GENE5",
                        "GENE6",
                        "GENE7",
                        "GENE8",
                        "GENE1",
                        "GENE3",
                    ],  # Different probe mapping
                    "class": np.array([0] * 8 + [1] * 8),
                },
            }
        }

        self.filter_object = {
            "posGeneNames": ["GENE1", "GENE2", "GENE3", "GENE4"],
            "negGeneNames": ["GENE5", "GENE6"],
            "filterDescription": "Test filter",
        }

    def test_validate_meta_object_valid(self):
        """Test validation of valid meta object."""
        # Should not raise any exception
        self.optimizer._validate_meta_object(self.meta_object)

    def test_validate_meta_object_invalid(self):
        """Test validation of invalid meta objects."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            self.optimizer._validate_meta_object("not_dict")

        with pytest.raises(ValueError, match="must contain 'originalData'"):
            self.optimizer._validate_meta_object({"wrong_key": {}})

        with pytest.raises(ValueError, match="must contain at least one dataset"):
            self.optimizer._validate_meta_object({"originalData": {}})

    def test_validate_filter_object_valid(self):
        """Test validation of valid filter object."""
        # Should not raise any exception
        self.optimizer._validate_filter_object(self.filter_object)

    def test_validate_filter_object_invalid(self):
        """Test validation of invalid filter objects."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            self.optimizer._validate_filter_object("not_dict")

        with pytest.raises(ValueError, match="must contain 'posGeneNames'"):
            self.optimizer._validate_filter_object({"negGeneNames": []})

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_convert_discovery_list_to_genes(self, mock_get_sample_level):
        """Test conversion of discovery datasets to gene-level data."""
        pos_genes = ["GENE1", "GENE2"]
        neg_genes = ["GENE3", "GENE4"]

        result = self.optimizer._convert_discovery_list_to_genes(
            self.meta_object, pos_genes, neg_genes
        )

        assert len(result) == 2  # Two discovery datasets
        assert all("genes" in dataset for dataset in result)
        assert all("class" in dataset for dataset in result)

    def test_geometric_mean(self):
        """Test geometric mean calculation."""
        # Test with positive values
        data = np.array([[1, 4], [2, 8]])
        result = self.optimizer._geometric_mean(data, axis=0)
        expected = np.array([np.sqrt(2), np.sqrt(32)])  # sqrt(1*2), sqrt(4*8)
        np.testing.assert_array_almost_equal(result, expected)

    def test_geometric_mean_with_zeros(self):
        """Test geometric mean with zero values."""
        data = np.array([[0, 4], [2, 0]])
        result = self.optimizer._geometric_mean(data, axis=0)

        # Should handle zeros by adding epsilon
        assert len(result) == 2
        assert all(result > 0)  # All values should be positive

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_score_genes_matrix(self, mock_get_sample_level):
        """Test gene scoring from expression matrix."""
        # Create test gene matrix
        gene_matrix = pd.DataFrame(
            {"sample1": [2.0, 4.0, 1.0, 8.0], "sample2": [3.0, 2.0, 2.0, 6.0]},
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )

        pos_genes = ["GENE1", "GENE2"]
        neg_genes = ["GENE3"]

        scores = self.optimizer._score_genes_matrix(gene_matrix, pos_genes, neg_genes)

        assert len(scores) == 2  # Two samples
        assert isinstance(scores, np.ndarray)
        # Scores should be standardized (mean=0, std=1)
        # Note: We use ddof=1 (sample std) to match R's scale() function.
        # With ddof=0 (population std), a 2-element standardized vector has
        # std ≈ 0.707 rather than 1.0. This is intentional R-parity behavior.
        assert abs(np.mean(scores)) < 1e-10
        assert abs(np.std(scores, ddof=1) - 1) < 1e-10

    def test_score_genes_matrix_no_genes(self):
        """Test gene scoring with no genes present."""
        gene_matrix = pd.DataFrame(
            {"sample1": [1.0, 2.0], "sample2": [1.5, 2.5]}, index=["GENE1", "GENE2"]
        )

        scores = self.optimizer._score_genes_matrix(gene_matrix, [], [])

        assert len(scores) == 2
        assert all(score == 0 for score in scores)

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_get_weighted_aucs_genes_list(self, mock_get_sample_level):
        """Test weighted AUC calculation across gene lists."""
        # Convert to discovery genes format
        discovery_genes = self.optimizer._convert_discovery_list_to_genes(
            self.meta_object, ["GENE1", "GENE2"], ["GENE3", "GENE4"]
        )

        weights = np.array([20, 16])  # Sample sizes

        weighted_auc = self.optimizer._get_weighted_aucs_genes_list(
            discovery_genes, ["GENE1"], ["GENE2"], weights
        )

        assert isinstance(weighted_auc, float)
        assert weighted_auc >= 0

    def test_get_weighted_aucs_genes_list_no_genes(self):
        """Test weighted AUC with no genes."""
        discovery_genes = [
            {"genes": pd.DataFrame(), "class": np.array([0, 1, 0, 1])},
            {"genes": pd.DataFrame(), "class": np.array([0, 0, 1, 1])},
        ]
        weights = np.array([4, 4])

        result = self.optimizer._get_weighted_aucs_genes_list(discovery_genes, [], [], weights)

        assert result == 0.5 * np.sum(weights)  # Random performance

    def test_match_all_genes_valid(self):
        """Test gene matching validation with valid inputs."""
        discovery_genes = [
            {
                "genes": pd.DataFrame(index=["GENE1", "GENE2", "GENE3"]),
                "class": np.array([0, 1]),
            }
        ]

        # Should not raise any exception
        self.optimizer._match_all_genes(
            discovery_genes, ["GENE1", "GENE2"], ["GENE3"], ["GENE1"], []
        )

    def test_match_all_genes_invalid_yes_pos(self):
        """Test gene matching with invalid yes_pos."""
        discovery_genes = [
            {"genes": pd.DataFrame(index=["GENE1", "GENE2"]), "class": np.array([0, 1])}
        ]

        with pytest.raises(ValueError, match="yes_pos contains genes not in pos_genes"):
            self.optimizer._match_all_genes(discovery_genes, ["GENE1"], [], ["GENE2"], [])

    def test_match_all_genes_missing_genes_warning(self):
        """Test gene matching with missing genes issues warning."""
        discovery_genes = [
            {
                "genes": pd.DataFrame(index=["GENE1"]),  # Missing GENE2
                "class": np.array([0, 1]),
            }
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.optimizer._match_all_genes(discovery_genes, ["GENE1", "GENE2"], [], [], [])
            assert len(w) == 1
            assert "not found in discovery datasets" in str(w[0].message)

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_backward_search_integration(self, mock_get_sample_level):
        """Test backward search integration (mocked)."""
        # Mock the gene data conversion
        mock_gene_data = pd.DataFrame(
            np.random.randn(6, 20) + 3,
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6"],
            columns=[f"sample_{i}" for i in range(20)],
        )
        mock_get_sample_level.return_value = mock_gene_data

        result = self.optimizer.backward_search(
            self.meta_object, self.filter_object, back_thresh=-0.1
        )

        assert isinstance(result, dict)
        assert "posGeneNames" in result
        assert "negGeneNames" in result
        assert "filterDescription" in result
        assert "Backward search" in result["filterDescription"]

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_forward_search_integration(self, mock_get_sample_level):
        """Test forward search integration (mocked)."""
        # Mock the gene data conversion
        mock_gene_data = pd.DataFrame(
            np.random.randn(6, 20) + 3,
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6"],
            columns=[f"sample_{i}" for i in range(20)],
        )
        mock_get_sample_level.return_value = mock_gene_data

        result = self.optimizer.forward_search(
            self.meta_object,
            self.filter_object,
            yes_pos=["GENE1"],
            yes_neg=[],
            forward_thresh=0.1,
        )

        assert isinstance(result, dict)
        assert "posGeneNames" in result
        assert "negGeneNames" in result
        assert "filterDescription" in result
        assert "Forward search" in result["filterDescription"]

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_backward_search_inner(self, mock_get_sample_level):
        """Test backward search inner loop."""
        discovery_genes = self.optimizer._convert_discovery_list_to_genes(
            self.meta_object, ["GENE1", "GENE2"], ["GENE3", "GENE4"]
        )

        weights = np.array([20, 16])
        current_auc = 15.0

        result = self.optimizer._backward_search_inner(
            discovery_genes, ["GENE1", "GENE2"], ["GENE3"], current_auc, weights
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 2 pos + 1 neg gene
        assert all(col in result.columns for col in ["gene", "original", "search", "diff"])

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_forward_search_inner(self, mock_get_sample_level):
        """Test forward search inner loop."""
        discovery_genes = self.optimizer._convert_discovery_list_to_genes(
            self.meta_object, ["GENE1", "GENE2", "GENE3", "GENE4"], []
        )

        weights = np.array([20, 16])
        current_auc = 10.0

        result = self.optimizer._forward_search_inner(
            discovery_genes, ["GENE3", "GENE4"], [], ["GENE1"], [], current_auc, weights
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 candidate genes
        assert all(col in result.columns for col in ["gene", "original", "search", "diff"])


class TestRatioScaling:
    """Tests for the ratio scaling fix in inner search loops.

    The R MetaIntegrator uses a ratio to balance positive and negative gene
    contributions when evaluating gene additions/removals in the inner search
    loops.  The Python port was missing this ratio, leading to different gene
    selections (Jaccard ~0.5-0.6 vs expected ~1.0).  These tests verify that
    the ratio is correctly computed and applied.

    R formulas:
      backsearchPos:    ratio = #neg / (#pos - 1)
      backsearchNeg:    ratio = (#neg - 1) / #pos
      forwardSearchPos: ratio = #selected_neg / (#selected_pos + 1)
      forwardSearchNeg: ratio = (#selected_neg + 1) / #selected_pos
    """

    def setup_method(self):
        """Build deterministic gene-level discovery data (no mocking needed)."""
        np.random.seed(42)
        self.optimizer = GeneSearchOptimizer()

        # Gene matrix: 6 genes x 20 samples, values > 0 (required by geom mean)
        self.gene_matrix = pd.DataFrame(
            np.random.randn(6, 20) * 0.5 + 5,
            index=["G1", "G2", "G3", "G4", "G5", "G6"],
            columns=[f"s{i}" for i in range(20)],
        )

        # Two discovery datasets with different sample sizes
        self.discovery_genes = [
            {
                "genes": self.gene_matrix.iloc[:, :12].copy(),
                "class": np.array([0] * 6 + [1] * 6),
            },
            {
                "genes": self.gene_matrix.iloc[:, :8].copy(),
                "class": np.array([0] * 4 + [1] * 4),
            },
        ]
        self.weights = np.array([12, 8])

    # ------------------------------------------------------------------
    # _score_genes_matrix ratio parameter
    # ------------------------------------------------------------------
    def test_score_genes_matrix_ratio_changes_scores(self):
        """Scores with ratio != 1 must differ from scores without ratio."""
        pos = ["G1", "G2"]
        neg = ["G3"]

        scores_no_ratio = self.optimizer._score_genes_matrix(self.gene_matrix, pos, neg, ratio=None)
        scores_with_ratio = self.optimizer._score_genes_matrix(
            self.gene_matrix, pos, neg, ratio=2.0
        )

        # Both should be standardized (mean≈0, sample std≈1)
        assert abs(np.mean(scores_no_ratio)) < 1e-10
        assert abs(np.mean(scores_with_ratio)) < 1e-10

        # But actual values must differ because negative contribution is scaled
        assert not np.allclose(
            scores_no_ratio, scores_with_ratio
        ), "Scores with ratio=2.0 should differ from scores without ratio"

    def test_score_genes_matrix_ratio_one_equals_no_ratio(self):
        """ratio=1.0 should produce the same result as ratio=None."""
        pos = ["G1", "G2"]
        neg = ["G3"]

        scores_none = self.optimizer._score_genes_matrix(self.gene_matrix, pos, neg, ratio=None)
        scores_one = self.optimizer._score_genes_matrix(self.gene_matrix, pos, neg, ratio=1.0)

        np.testing.assert_array_almost_equal(scores_none, scores_one)

    def test_score_genes_matrix_ratio_zero_ignores_neg(self):
        """ratio=0.0 should produce the same result as having no negative genes."""
        pos = ["G1", "G2"]
        neg = ["G3"]

        scores_ratio_zero = self.optimizer._score_genes_matrix(
            self.gene_matrix, pos, neg, ratio=0.0
        )
        scores_no_neg = self.optimizer._score_genes_matrix(self.gene_matrix, pos, [], ratio=None)

        np.testing.assert_array_almost_equal(scores_ratio_zero, scores_no_neg)

    # ------------------------------------------------------------------
    # _get_weighted_aucs_genes_list ratio threading
    # ------------------------------------------------------------------
    def test_weighted_auc_with_ratio_differs(self):
        """Weighted AUC with a non-unity ratio must differ from without."""
        pos = ["G1", "G2", "G3"]
        neg = ["G4", "G5"]

        auc_no_ratio = self.optimizer._get_weighted_aucs_genes_list(
            self.discovery_genes, pos, neg, self.weights, ratio=None
        )
        auc_with_ratio = self.optimizer._get_weighted_aucs_genes_list(
            self.discovery_genes, pos, neg, self.weights, ratio=3.0
        )

        # They *can* coincidentally match, but with ratio=3.0 on real data
        # the scoring is very different so the AUCs should diverge.
        assert auc_no_ratio != pytest.approx(
            auc_with_ratio, abs=1e-6
        ), "Weighted AUC with ratio=3.0 should differ from no-ratio"

    def test_weighted_auc_pos_only_neg_sign(self):
        """pos_only_neg_sign=True should score as scale(-negScore)."""
        neg = ["G4", "G5"]

        auc = self.optimizer._get_weighted_aucs_genes_list(
            self.discovery_genes, [], neg, self.weights, ratio=1.0, pos_only_neg_sign=True
        )

        assert isinstance(auc, float)
        assert auc > 0  # should be a real weighted AUC, not degenerate

    # ------------------------------------------------------------------
    # Backward search inner loop ratio values
    # ------------------------------------------------------------------
    def test_backward_inner_ratio_for_pos_removal(self):
        """Removing a pos gene: ratio should be #neg / (#pos - 1).

        We verify indirectly: the search AUC for removing a positive gene
        must differ between the ratio-aware implementation and a manual
        call without ratio.
        """
        pos = ["G1", "G2", "G3"]  # 3 pos
        neg = ["G4", "G5"]  # 2 neg
        # Expected ratio = 2 / (3-1) = 1.0
        current_auc = self.optimizer._get_weighted_aucs_genes_list(
            self.discovery_genes, pos, neg, self.weights
        )

        result = self.optimizer._backward_search_inner(
            self.discovery_genes, pos, neg, current_auc, self.weights
        )

        assert len(result) == 5  # 3 pos + 2 neg
        assert set(result["gene"].tolist()) == {"G1", "G2", "G3", "G4", "G5"}
        assert all(result["original"] == current_auc)

    def test_backward_inner_single_pos_gene_edge_case(self):
        """With only 1 pos gene, removing it should use scale(-negScore).

        R's .backsearchPos: if(length(pos.genes) <= 1) totalScore = scale(-negScore)
        """
        pos = ["G1"]  # single pos
        neg = ["G4", "G5"]  # 2 neg
        current_auc = self.optimizer._get_weighted_aucs_genes_list(
            self.discovery_genes, pos, neg, self.weights
        )

        result = self.optimizer._backward_search_inner(
            self.discovery_genes, pos, neg, current_auc, self.weights
        )

        # Should have 3 rows: 1 pos + 2 neg
        assert len(result) == 3
        pos_row = result[result["gene"] == "G1"]
        assert len(pos_row) == 1
        # The search AUC should be a real value (not NaN or 0)
        assert np.isfinite(pos_row["search"].iloc[0])

    def test_backward_inner_single_neg_gene_edge_case(self):
        """With only 1 neg gene, removing it should use scale(posScore).

        R's .backsearchNeg: if(length(neg.genes) <= 1) totalScore = scale(posScore)
        """
        pos = ["G1", "G2", "G3"]
        neg = ["G4"]  # single neg
        current_auc = self.optimizer._get_weighted_aucs_genes_list(
            self.discovery_genes, pos, neg, self.weights
        )

        result = self.optimizer._backward_search_inner(
            self.discovery_genes, pos, neg, current_auc, self.weights
        )

        assert len(result) == 4  # 3 pos + 1 neg

        neg_row = result[result["gene"] == "G4"]
        assert len(neg_row) == 1
        assert np.isfinite(neg_row["search"].iloc[0])

        # Verify the neg-removal AUC equals using pos-only scoring
        auc_pos_only = self.optimizer._get_weighted_aucs_genes_list(
            self.discovery_genes, pos, [], self.weights, ratio=0.0
        )
        assert neg_row["search"].iloc[0] == pytest.approx(auc_pos_only, abs=1e-10)

    # ------------------------------------------------------------------
    # Forward search inner loop ratio values
    # ------------------------------------------------------------------
    def test_forward_inner_ratio_for_pos_addition(self):
        """Adding a pos gene: ratio = #selected_neg / (#selected_pos + 1).

        With selected_pos=[G1], selected_neg=[G4], adding a pos candidate:
          ratio = 1 / (1 + 1) = 0.5
        """
        selected_pos = ["G1"]
        selected_neg = ["G4"]
        candidate_pos = ["G2", "G3"]
        candidate_neg = []
        current_auc = self.optimizer._get_weighted_aucs_genes_list(
            self.discovery_genes, selected_pos, selected_neg, self.weights
        )

        result = self.optimizer._forward_search_inner(
            self.discovery_genes,
            candidate_pos,
            candidate_neg,
            selected_pos,
            selected_neg,
            current_auc,
            self.weights,
        )

        assert len(result) == 2  # 2 candidates
        assert set(result["gene"].tolist()) == {"G2", "G3"}

        # Verify the AUC was computed with ratio=0.5
        expected_ratio = len(selected_neg) / (len(selected_pos) + 1)  # 0.5
        assert expected_ratio == pytest.approx(0.5)

        for _, row in result.iterrows():
            gene = row["gene"]
            test_pos = selected_pos + [gene]
            manual_auc = self.optimizer._get_weighted_aucs_genes_list(
                self.discovery_genes, test_pos, selected_neg, self.weights, ratio=expected_ratio
            )
            assert row["search"] == pytest.approx(
                manual_auc, abs=1e-10
            ), f"Forward inner AUC for {gene} should match manual ratio=0.5 calculation"

    def test_forward_inner_ratio_for_neg_addition(self):
        """Adding a neg gene: ratio = (#selected_neg + 1) / #selected_pos.

        With selected_pos=[G1, G2], selected_neg=[G4], adding a neg candidate:
          ratio = (1 + 1) / 2 = 1.0
        """
        selected_pos = ["G1", "G2"]
        selected_neg = ["G4"]
        candidate_pos = []
        candidate_neg = ["G5", "G6"]
        current_auc = self.optimizer._get_weighted_aucs_genes_list(
            self.discovery_genes, selected_pos, selected_neg, self.weights
        )

        result = self.optimizer._forward_search_inner(
            self.discovery_genes,
            candidate_pos,
            candidate_neg,
            selected_pos,
            selected_neg,
            current_auc,
            self.weights,
        )

        assert len(result) == 2
        expected_ratio = (len(selected_neg) + 1) / len(selected_pos)  # 1.0
        assert expected_ratio == pytest.approx(1.0)

        for _, row in result.iterrows():
            gene = row["gene"]
            test_neg = selected_neg + [gene]
            manual_auc = self.optimizer._get_weighted_aucs_genes_list(
                self.discovery_genes, selected_pos, test_neg, self.weights, ratio=expected_ratio
            )
            assert row["search"] == pytest.approx(
                manual_auc, abs=1e-10
            ), f"Forward inner AUC for {gene} should match manual ratio=1.0 calculation"

    def test_forward_inner_neg_addition_no_pos_selected(self):
        """When no pos genes selected, R uses ratio=1 for neg addition.

        R: if(is.null(yes.pos)) ratio <- 1
        """
        selected_pos = []
        selected_neg = []
        candidate_pos = []
        candidate_neg = ["G4", "G5"]

        result = self.optimizer._forward_search_inner(
            self.discovery_genes,
            candidate_pos,
            candidate_neg,
            selected_pos,
            selected_neg,
            0.0,
            self.weights,
        )

        assert len(result) == 2

        # Verify ratio=1.0 was used
        for _, row in result.iterrows():
            gene = row["gene"]
            test_neg = [gene]
            manual_auc = self.optimizer._get_weighted_aucs_genes_list(
                self.discovery_genes, [], test_neg, self.weights, ratio=1.0
            )
            assert row["search"] == pytest.approx(
                manual_auc, abs=1e-10
            ), f"With no pos selected, neg candidate {gene} should use ratio=1.0"

    # ------------------------------------------------------------------
    # Backward inner ratio correctness (cross-check with manual calc)
    # ------------------------------------------------------------------
    def test_backward_inner_pos_removal_ratio_matches_manual(self):
        """Cross-check: backward inner AUC for pos removal matches manual
        _get_weighted_aucs_genes_list call with the correct R-formula ratio.
        """
        pos = ["G1", "G2", "G3", "G4"]  # 4 pos
        neg = ["G5", "G6"]  # 2 neg
        # R ratio for pos removal = #neg / (#pos - 1) = 2 / 3
        expected_ratio = len(neg) / (len(pos) - 1)

        current_auc = self.optimizer._get_weighted_aucs_genes_list(
            self.discovery_genes, pos, neg, self.weights
        )

        result = self.optimizer._backward_search_inner(
            self.discovery_genes, pos, neg, current_auc, self.weights
        )

        # Check each pos gene's search AUC
        for gene in pos:
            row = result[result["gene"] == gene]
            test_pos = [g for g in pos if g != gene]
            manual_auc = self.optimizer._get_weighted_aucs_genes_list(
                self.discovery_genes, test_pos, neg, self.weights, ratio=expected_ratio
            )
            assert row["search"].iloc[0] == pytest.approx(
                manual_auc, abs=1e-10
            ), f"Backward inner AUC for removing {gene} should use ratio={expected_ratio:.4f}"

    def test_backward_inner_neg_removal_ratio_matches_manual(self):
        """Cross-check: backward inner AUC for neg removal matches manual
        _get_weighted_aucs_genes_list call with the correct R-formula ratio.
        """
        pos = ["G1", "G2", "G3"]  # 3 pos
        neg = ["G4", "G5", "G6"]  # 3 neg
        # R ratio for neg removal = (#neg - 1) / #pos = 2 / 3
        expected_ratio = (len(neg) - 1) / len(pos)

        current_auc = self.optimizer._get_weighted_aucs_genes_list(
            self.discovery_genes, pos, neg, self.weights
        )

        result = self.optimizer._backward_search_inner(
            self.discovery_genes, pos, neg, current_auc, self.weights
        )

        for gene in neg:
            row = result[result["gene"] == gene]
            test_neg = [g for g in neg if g != gene]
            manual_auc = self.optimizer._get_weighted_aucs_genes_list(
                self.discovery_genes, pos, test_neg, self.weights, ratio=expected_ratio
            )
            assert row["search"].iloc[0] == pytest.approx(
                manual_auc, abs=1e-10
            ), f"Backward inner AUC for removing {gene} should use ratio={expected_ratio:.4f}"


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.y_true = np.array([0, 0, 0, 1, 1, 1])
        self.y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_backward_search_convenience(self, mock_get_sample_level):
        """Test backward search convenience function."""
        # Mock data
        mock_gene_data = pd.DataFrame(np.random.randn(4, 10) + 3)
        mock_get_sample_level.return_value = mock_gene_data

        meta_object = {
            "originalData": {
                "study1": {
                    "expr": pd.DataFrame(np.random.randn(5, 10)),
                    "keys": ["G1", "G2", "G3", "G4", "G5"],
                    "class": np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
                }
            }
        }

        filter_object = {
            "posGeneNames": ["G1", "G2"],
            "negGeneNames": ["G3", "G4"],
            "filterDescription": "Test",
        }

        result = backward_search(meta_object, filter_object, back_thresh=0.0)

        assert isinstance(result, dict)
        assert "posGeneNames" in result
        assert "negGeneNames" in result

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_forward_search_convenience(self, mock_get_sample_level):
        """Test forward search convenience function."""
        # Mock data
        mock_gene_data = pd.DataFrame(np.random.randn(4, 10) + 3)
        mock_get_sample_level.return_value = mock_gene_data

        meta_object = {
            "originalData": {
                "study1": {
                    "expr": pd.DataFrame(np.random.randn(5, 10)),
                    "keys": ["G1", "G2", "G3", "G4", "G5"],
                    "class": np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
                }
            }
        }

        filter_object = {
            "posGeneNames": ["G1", "G2"],
            "negGeneNames": ["G3", "G4"],
            "filterDescription": "Test",
        }

        result = forward_search(
            meta_object, filter_object, yes_pos=["G1"], yes_neg=[], forward_thresh=0.0
        )

        assert isinstance(result, dict)
        assert "posGeneNames" in result
        assert "negGeneNames" in result


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def setup_method(self):
        """Setup test data."""
        pass

    def test_gene_search_with_no_discovery_datasets(self):
        """Test gene search with empty meta object."""
        optimizer = GeneSearchOptimizer()
        meta_object = {"originalData": {}}
        filter_object = {"posGeneNames": [], "negGeneNames": []}

        with pytest.raises(ValueError, match="must contain at least one dataset"):
            optimizer.backward_search(meta_object, filter_object)

    def test_gene_search_with_invalid_genes(self):
        """Test gene search with genes not in datasets."""
        optimizer = GeneSearchOptimizer()

        meta_object = {
            "originalData": {
                "study1": {
                    "expr": pd.DataFrame({"sample1": [1, 2]}, index=["probe1", "probe2"]),
                    "keys": ["GENE1", "GENE2"],
                    "class": np.array([0, 1]),
                }
            }
        }

        filter_object = {
            "posGeneNames": ["MISSING_GENE"],
            "negGeneNames": [],
            "filterDescription": "Test",
        }

        # Should handle missing genes gracefully with warning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # This might work but issue warnings, or raise exceptions
            # depending on implementation details
            try:
                result = optimizer.backward_search(meta_object, filter_object)
                assert isinstance(result, dict)
            except ImportError:
                # Expected if getSampleLevelGeneData not available
                pass


class TestPerformanceAndRobustness:
    """Test suite for performance and robustness."""

    def test_memory_efficiency_gene_search(self):
        """Test memory efficiency of gene search algorithms."""
        optimizer = GeneSearchOptimizer()

        # Create large mock gene matrix
        large_gene_matrix = pd.DataFrame(
            np.random.randn(1000, 100) + 3,  # 1000 genes, 100 samples
            index=[f"GENE_{i}" for i in range(1000)],
            columns=[f"sample_{i}" for i in range(100)],
        )

        pos_genes = [f"GENE_{i}" for i in range(50)]
        neg_genes = [f"GENE_{i}" for i in range(50, 100)]

        # This should not cause memory issues
        scores = optimizer._score_genes_matrix(large_gene_matrix, pos_genes, neg_genes)

        assert len(scores) == 100  # Number of samples
        assert isinstance(scores, np.ndarray)


class TestMaxIterationsAndProgressCallback:
    """Tests for the ``max_iterations`` / ``progress_callback`` parameters
    added to ``backward_search`` / ``forward_search`` (both the class
    methods and their private ``_weighted`` implementations, plus the
    module-level convenience functions).

    Uses the same deterministic, gene-level ``discovery_genes`` fixture as
    ``TestRatioScaling`` above (no ``getSampleLevelGeneData`` mocking
    needed, since ``_backward_search_weighted`` / ``_forward_search_weighted``
    take already-converted gene-level data directly), so these tests are
    fully deterministic: no random test-to-test flakiness from where a
    real search happens to stop.
    """

    def setup_method(self):
        """Build the same deterministic gene-level discovery data
        ``TestRatioScaling`` uses, so a natural (uncapped) search's stopping
        point is reproducible across the calls each test makes."""
        np.random.seed(42)
        self.optimizer = GeneSearchOptimizer()

        self.gene_matrix = pd.DataFrame(
            np.random.randn(6, 20) * 0.5 + 5,
            index=["G1", "G2", "G3", "G4", "G5", "G6"],
            columns=[f"s{i}" for i in range(20)],
        )
        self.discovery_genes = [
            {
                "genes": self.gene_matrix.iloc[:, :12].copy(),
                "class": np.array([0] * 6 + [1] * 6),
            },
            {
                "genes": self.gene_matrix.iloc[:, :8].copy(),
                "class": np.array([0] * 4 + [1] * 4),
            },
        ]
        self.weights = np.array([12, 8])

    # ------------------------------------------------------------------
    # max_iterations=None: behavior must be unchanged from before this
    # parameter existed (the same regression guarantee this whole
    # enhancement was built on top of).
    # ------------------------------------------------------------------
    def test_backward_default_none_converges_and_reports_it(self):
        pos, neg, converged, iterations_run = self.optimizer._backward_search_weighted(
            self.discovery_genes, ["G1", "G2", "G3"], ["G4", "G5", "G6"], back_thresh=-1.0
        )
        assert converged is True
        assert isinstance(iterations_run, int)
        assert iterations_run >= 0

    def test_forward_default_none_converges_and_reports_it(self):
        pos, neg, converged, iterations_run = self.optimizer._forward_search_weighted(
            self.discovery_genes,
            ["G1", "G2", "G3"],
            ["G4", "G5", "G6"],
            None,
            None,
            forward_thresh=-1.0,
        )
        assert converged is True
        assert isinstance(iterations_run, int)
        assert iterations_run >= 0

    # ------------------------------------------------------------------
    # max_iterations actually cuts the search off before natural
    # convergence, deterministically: run once uncapped to find the
    # natural stopping point N, then again capped at N-1 and confirm it
    # stops exactly there with converged=False.
    # ------------------------------------------------------------------
    def test_backward_max_iterations_cuts_off_before_convergence(self):
        pos_genes = ["G1", "G2", "G3"]
        neg_genes = ["G4", "G5", "G6"]

        _, _, natural_converged, natural_iterations = self.optimizer._backward_search_weighted(
            self.discovery_genes, pos_genes, neg_genes, back_thresh=-1.0
        )
        assert natural_converged is True
        assert natural_iterations >= 1, (
            "test fixture must produce at least 1 natural removal for this "
            "test to meaningfully exercise a cap; adjust back_thresh/seed "
            "if this fails."
        )

        capped_at = natural_iterations - 1
        _, _, capped_converged, capped_iterations = self.optimizer._backward_search_weighted(
            self.discovery_genes,
            pos_genes,
            neg_genes,
            back_thresh=-1.0,
            max_iterations=capped_at,
        )
        assert capped_iterations == capped_at
        assert capped_converged is False

    def test_forward_max_iterations_cuts_off_before_convergence(self):
        pos_genes = ["G1", "G2", "G3"]
        neg_genes = ["G4", "G5", "G6"]

        _, _, natural_converged, natural_iterations = self.optimizer._forward_search_weighted(
            self.discovery_genes, pos_genes, neg_genes, None, None, forward_thresh=-1.0
        )
        assert natural_converged is True
        assert natural_iterations >= 1, (
            "test fixture must produce at least 1 natural addition for this "
            "test to meaningfully exercise a cap; adjust forward_thresh/seed "
            "if this fails."
        )

        capped_at = natural_iterations - 1
        _, _, capped_converged, capped_iterations = self.optimizer._forward_search_weighted(
            self.discovery_genes,
            pos_genes,
            neg_genes,
            None,
            None,
            forward_thresh=-1.0,
            max_iterations=capped_at,
        )
        assert capped_iterations == capped_at
        assert capped_converged is False

    def test_max_iterations_zero_makes_no_moves(self):
        """An edge case worth its own test: max_iterations=0 should stop
        before evaluating anything, leaving the input signature untouched."""
        pos_genes = ["G1", "G2", "G3"]
        neg_genes = ["G4", "G5", "G6"]

        pos, neg, converged, iterations_run = self.optimizer._backward_search_weighted(
            self.discovery_genes, pos_genes, neg_genes, back_thresh=-1.0, max_iterations=0
        )
        assert iterations_run == 0
        assert converged is False
        assert pos == pos_genes
        assert neg == neg_genes

    # ------------------------------------------------------------------
    # progress_callback: invoked once per accepted move, with the
    # documented (iteration, gene, auc_before, auc_after,
    # candidates_remaining) signature.
    # ------------------------------------------------------------------
    def test_backward_progress_callback_invoked_once_per_removal(self):
        calls = []

        def _capture(iteration, gene, auc_before, auc_after, candidates_remaining):
            calls.append((iteration, gene, auc_before, auc_after, candidates_remaining))

        pos_genes = ["G1", "G2", "G3"]
        neg_genes = ["G4", "G5", "G6"]
        all_genes = set(pos_genes) | set(neg_genes)

        _, _, converged, iterations_run = self.optimizer._backward_search_weighted(
            self.discovery_genes,
            pos_genes,
            neg_genes,
            back_thresh=-1.0,
            progress_callback=_capture,
        )

        assert len(calls) == iterations_run
        # Iteration numbers are 1-indexed and sequential.
        assert [c[0] for c in calls] == list(range(1, iterations_run + 1))
        # Every reported gene actually came from the starting signature.
        assert all(c[1] in all_genes for c in calls)
        # candidates_remaining strictly decreases by exactly 1 each call
        # (one gene leaves the signature per accepted removal).
        remaining_counts = [c[4] for c in calls]
        assert remaining_counts == sorted(remaining_counts, reverse=True)
        if len(remaining_counts) > 1:
            deltas = [
                remaining_counts[i] - remaining_counts[i + 1]
                for i in range(len(remaining_counts) - 1)
            ]
            assert all(d == 1 for d in deltas)

    def test_forward_progress_callback_invoked_once_per_addition(self):
        calls = []

        def _capture(iteration, gene, auc_before, auc_after, candidates_remaining):
            calls.append((iteration, gene, auc_before, auc_after, candidates_remaining))

        pos_genes = ["G1", "G2", "G3"]
        neg_genes = ["G4", "G5", "G6"]
        all_genes = set(pos_genes) | set(neg_genes)

        _, _, converged, iterations_run = self.optimizer._forward_search_weighted(
            self.discovery_genes,
            pos_genes,
            neg_genes,
            None,
            None,
            forward_thresh=-1.0,
            progress_callback=_capture,
        )

        assert len(calls) == iterations_run
        assert [c[0] for c in calls] == list(range(1, iterations_run + 1))
        assert all(c[1] in all_genes for c in calls)

    def test_progress_callback_exception_propagates_and_aborts_search(self):
        """A callback that raises must abort the search immediately rather
        than being swallowed: silently continuing after a reporting
        failure could hide a real problem in whatever the callback was
        doing (e.g. persisting progress, refreshing a lock)."""

        class _Boom(Exception):
            pass

        def _raising_callback(iteration, gene, auc_before, auc_after, candidates_remaining):
            raise _Boom("callback failed")

        with pytest.raises(_Boom):
            self.optimizer._backward_search_weighted(
                self.discovery_genes,
                ["G1", "G2", "G3"],
                ["G4", "G5", "G6"],
                back_thresh=-1.0,
                progress_callback=_raising_callback,
            )

    # ------------------------------------------------------------------
    # Public methods and module-level convenience functions thread the
    # new parameters through correctly, end to end.
    # ------------------------------------------------------------------
    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_public_backward_search_returns_converged_and_iterations_run(
        self, mock_get_sample_level
    ):
        mock_get_sample_level.return_value = pd.DataFrame(
            np.random.randn(6, 20) + 3,
            index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6"],
            columns=[f"sample_{i}" for i in range(20)],
        )
        meta_object = {
            "originalData": {
                "discovery1": {
                    "formattedName": "Discovery Dataset 1",
                    "expr": pd.DataFrame(
                        np.random.randn(10, 20) + 2,
                        index=[f"probe_{i}" for i in range(10)],
                        columns=[f"sample_{i}" for i in range(20)],
                    ),
                    "keys": [
                        "GENE1",
                        "GENE2",
                        "GENE3",
                        "GENE4",
                        "GENE5",
                        "GENE1",
                        "GENE2",
                        "GENE6",
                        "GENE7",
                        "GENE8",
                    ],
                    "class": np.array([0] * 10 + [1] * 10),
                }
            }
        }
        filter_object = {
            "posGeneNames": ["GENE1", "GENE2", "GENE3", "GENE4"],
            "negGeneNames": ["GENE5", "GENE6"],
            "filterDescription": "Test filter",
        }

        result = self.optimizer.backward_search(
            meta_object, filter_object, back_thresh=-0.1, max_iterations=1
        )

        assert "converged" in result
        assert "iterationsRun" in result
        assert result["iterationsRun"] <= 1

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_convenience_backward_search_passes_through_new_params(self, mock_get_sample_level):
        mock_get_sample_level.return_value = pd.DataFrame(np.random.randn(4, 10) + 3)
        meta_object = {
            "originalData": {
                "study1": {
                    "expr": pd.DataFrame(np.random.randn(5, 10)),
                    "keys": ["G1", "G2", "G3", "G4", "G5"],
                    "class": np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
                }
            }
        }
        filter_object = {
            "posGeneNames": ["G1", "G2"],
            "negGeneNames": ["G3", "G4"],
            "filterDescription": "Test",
        }
        calls = []

        result = backward_search(
            meta_object,
            filter_object,
            back_thresh=0.0,
            max_iterations=0,
            progress_callback=lambda *args: calls.append(args),
        )

        assert result["iterationsRun"] == 0
        assert result["converged"] is False
        assert calls == []  # max_iterations=0 means the callback never fires

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_convenience_forward_search_passes_through_new_params(self, mock_get_sample_level):
        mock_get_sample_level.return_value = pd.DataFrame(np.random.randn(4, 10) + 3)
        meta_object = {
            "originalData": {
                "study1": {
                    "expr": pd.DataFrame(np.random.randn(5, 10)),
                    "keys": ["G1", "G2", "G3", "G4", "G5"],
                    "class": np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
                }
            }
        }
        filter_object = {
            "posGeneNames": ["G1", "G2"],
            "negGeneNames": ["G3", "G4"],
            "filterDescription": "Test",
        }

        result = forward_search(
            meta_object,
            filter_object,
            yes_pos=["G1"],
            yes_neg=[],
            forward_thresh=0.0,
            max_iterations=0,
        )

        assert result["iterationsRun"] == 0
        assert result["converged"] is False


class TestScoreSignature:
    """Tests for ``GeneSearchOptimizer.score_signature``, added so a
    caller can get a gene set's weighted AUC without running a search --
    most usefully, the starting AUC of a search that ends up making zero
    moves, which ``progress_callback`` never reports (it only fires once
    a move actually happens).

    Reuses the same realistic ``meta_object``/``filter_object`` fixture
    shape as ``TestGeneSearchOptimizer.setup_method`` (own copy, not
    shared via inheritance, matching this file's existing convention of
    ``TestRatioScaling``/``TestMaxIterationsAndProgressCallback`` each
    building their own fixtures rather than subclassing).
    """

    def setup_method(self):
        np.random.seed(42)
        self.optimizer = GeneSearchOptimizer()

        self.meta_object = {
            "originalData": {
                "discovery1": {
                    "formattedName": "Discovery Dataset 1",
                    "expr": pd.DataFrame(
                        np.random.randn(10, 20) + 2,
                        index=[f"probe_{i}" for i in range(10)],
                        columns=[f"sample_{i}" for i in range(20)],
                    ),
                    "keys": [
                        "GENE1",
                        "GENE2",
                        "GENE3",
                        "GENE4",
                        "GENE5",
                        "GENE1",
                        "GENE2",
                        "GENE6",
                        "GENE7",
                        "GENE8",
                    ],
                    "class": np.array([0] * 10 + [1] * 10),
                },
                "discovery2": {
                    "formattedName": "Discovery Dataset 2",
                    "expr": pd.DataFrame(
                        np.random.randn(10, 16) + 2,
                        index=[f"probe_{i}" for i in range(10)],
                        columns=[f"sample_{i}" for i in range(16)],
                    ),
                    "keys": [
                        "GENE1",
                        "GENE2",
                        "GENE3",
                        "GENE4",
                        "GENE5",
                        "GENE6",
                        "GENE7",
                        "GENE8",
                        "GENE1",
                        "GENE3",
                    ],
                    "class": np.array([0] * 8 + [1] * 8),
                },
            }
        }
        self.filter_object = {
            "posGeneNames": ["GENE1", "GENE2", "GENE3", "GENE4"],
            "negGeneNames": ["GENE5", "GENE6"],
            "filterDescription": "Test filter",
        }

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_returns_a_non_negative_float(self, mock_get_sample_level):
        result = self.optimizer.score_signature(self.meta_object, self.filter_object)
        assert isinstance(result, float)
        assert result >= 0

    def test_empty_signature_returns_zero(self):
        """An empty signature (no genes named) returns ``0.0``, matching
        ``_forward_search_weighted``'s own internal placeholder for the
        same case. Not a special case in ``score_signature`` itself:
        ``getSampleLevelGeneData`` requires a non-empty gene list, so
        every dataset fails to convert and is skipped (logged via
        ``logger.warning``, a log record rather than a Python
        ``warnings.warn`` -- nothing for ``pytest.warns`` to catch
        here), leaving an empty ``discovery_genes``/``weights`` pair.
        ``_get_weighted_aucs_genes_list``'s own ``0.5 * np.sum(weights)``
        branch then evaluates against that empty array, which is
        ``0.0``."""
        empty_filter = {
            "posGeneNames": [],
            "negGeneNames": [],
            "filterDescription": "empty",
        }
        result = self.optimizer.score_signature(self.meta_object, empty_filter)
        assert result == 0.0

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_matches_manual_get_weighted_aucs_genes_list_computation(self, mock_get_sample_level):
        """Confirms this method is a thin wrapper, not a reimplementation:
        the same genes, scored by hand through the lower-level pieces
        ``score_signature`` itself calls, must produce the identical
        value."""
        pos_genes = self.filter_object["posGeneNames"]
        neg_genes = self.filter_object["negGeneNames"]

        discovery_genes = self.optimizer._convert_discovery_list_to_genes(
            self.meta_object, pos_genes, neg_genes
        )
        weights = np.array([len(dataset["class"]) for dataset in discovery_genes])
        expected = self.optimizer._get_weighted_aucs_genes_list(
            discovery_genes, pos_genes, neg_genes, weights
        )

        result = self.optimizer.score_signature(self.meta_object, self.filter_object)

        assert result == pytest.approx(expected)

    def test_rejects_invalid_meta_object(self):
        with pytest.raises(ValueError, match="must be a dictionary"):
            self.optimizer.score_signature("not_a_dict", self.filter_object)

    def test_rejects_invalid_filter_object(self):
        with pytest.raises(ValueError, match="must contain 'posGeneNames'"):
            self.optimizer.score_signature(self.meta_object, {"negGeneNames": []})

    @patch("metaintegrator.sample_level_gene_data.getSampleLevelGeneData")
    def test_gene_missing_from_every_dataset_is_silently_nan_padded_not_warned(
        self, mock_get_sample_level, recwarn
    ):
        """Corrects a disproven assumption caught by the real test suite
        (see this method's own docstring): a gene named in
        ``filter_object`` but absent from every discovery dataset does
        NOT trigger
        ``_match_all_genes``'s "not found" warning here, because
        ``_extract_data_from_gem`` backfills any unmatched requested
        gene as a NaN-valued row, so it is always present in the
        returned gene-level matrix's index by the time
        ``_match_all_genes`` checks for missing ones. Verified by
        actually running this test (the original version, asserting
        the opposite, failed with "DID NOT WARN") rather than by
        tracing the code alone."""
        filter_with_unknown_gene = {
            "posGeneNames": ["GENE1", "NOT_A_REAL_GENE"],
            "negGeneNames": ["GENE5"],
            "filterDescription": "unknown gene",
        }
        result = self.optimizer.score_signature(self.meta_object, filter_with_unknown_gene)
        assert isinstance(result, float)
        assert len(recwarn) == 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
