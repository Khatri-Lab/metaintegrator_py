"""
Unit tests for GeneAnnotator.

Tests gene symbol extraction from GPL annotation tables without any
network access, GEOparse dependency, or file I/O.  Every test uses
small in-memory DataFrames that exercise specific parsing paths.
"""

import os
import sys

import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

from preprocessing.gene_annotator import GeneAnnotator

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def annotator():
    """Provide a fresh GeneAnnotator instance for each test."""
    return GeneAnnotator()


def _make_gpl_table(id_col, gene_col, gene_col_name="Gene Symbol"):
    """Helper: build a minimal GPL table DataFrame."""
    return pd.DataFrame({"ID": id_col, gene_col_name: gene_col})


class FakeValidator:
    """Minimal stand-in for EnsemblGeneValidator."""

    def __init__(self, valid: set):
        """Store the set of gene symbols this validator considers valid."""
        self.valid_genes = valid


# ============================================================================
# clean_symbol — direct symbol columns
# ============================================================================


class TestCleanSymbolDirect:
    """Test clean_symbol() for direct Gene Symbol columns."""

    def test_simple_gene(self, annotator):
        """A single gene symbol passes through unchanged."""
        assert annotator.clean_symbol("CD6", is_assignment=False) == "CD6"

    def test_triple_slash_separator(self, annotator):
        """Triple-slash (///) separated symbols become comma-joined."""
        assert annotator.clean_symbol("CD6 /// IL2RG", is_assignment=False) == "CD6,IL2RG"

    def test_semicolon_separator(self, annotator):
        """Semicolon-separated symbols become comma-joined."""
        assert annotator.clean_symbol("BRCA1;TP53", is_assignment=False) == "BRCA1,TP53"

    def test_mixed_separators(self, annotator):
        """Triple-slash and semicolon separators can mix in the same value."""
        result = annotator.clean_symbol("A /// B;C", is_assignment=False)
        assert result == "A,B,C"

    def test_accession_filtered(self, annotator):
        """Accession IDs (NM_, ENST) should be skipped in direct columns."""
        assert annotator.clean_symbol("NM_001234", is_assignment=False) == "NA"

    def test_accession_mixed_with_gene(self, annotator):
        """An accession entry mixed with a real gene symbol is dropped, keeping the gene."""
        result = annotator.clean_symbol("ENST00000396160 /// TP53", is_assignment=False)
        assert result == "TP53"

    def test_dash_sentinel(self, annotator):
        """The "---" sentinel value maps to NA."""
        assert annotator.clean_symbol("---", is_assignment=False) == "NA"

    def test_empty_string(self, annotator):
        """An empty string maps to NA."""
        assert annotator.clean_symbol("", is_assignment=False) == "NA"

    def test_none(self, annotator):
        """A None value maps to NA."""
        assert annotator.clean_symbol(None, is_assignment=False) == "NA"

    def test_nan(self, annotator):
        """A float NaN value maps to NA."""
        assert annotator.clean_symbol(float("nan"), is_assignment=False) == "NA"

    def test_na_string(self, annotator):
        """The literal string "NA" maps to itself."""
        assert annotator.clean_symbol("NA", is_assignment=False) == "NA"

    def test_whitespace_only(self, annotator):
        """A whitespace-only value maps to NA."""
        assert annotator.clean_symbol("   ", is_assignment=False) == "NA"

    def test_double_slash_within_entry(self, annotator):
        """Double-slash (//) splits fields within an entry."""
        result = annotator.clean_symbol("TP53 // tumor protein p53", is_assignment=False)
        assert result == "TP53"


# ============================================================================
# clean_symbol — assignment columns
# ============================================================================


class TestCleanSymbolAssignment:
    """Test clean_symbol() for gene_assignment / mrna_assignment columns."""

    def test_standard_assignment(self, annotator):
        """The gene symbol is the second field in a standard `//`-delimited assignment string."""
        raw = "NM_001004195 // OR4F4 // olfactory receptor, family 4, subfamily F, member 4 // 1p36.33 // 79501"
        assert annotator.clean_symbol(raw, is_assignment=True) == "OR4F4"

    def test_ensembl_transcript_assignment(self, annotator):
        """The same second-field extraction works when the first field is an Ensembl transcript ID."""
        raw = "ENST00000328113 // OR4G2P // olfactory receptor, family 4, subfamily G, member 2"
        assert annotator.clean_symbol(raw, is_assignment=True) == "OR4G2P"

    def test_multiple_entries_takes_first(self, annotator):
        """R Case 2: only extract from first /// entry."""
        raw = "NR_024437 // LOC728323 // uncharacterized LOC728323 /// XM_006711854 // OTHER // other desc"
        result = annotator.clean_symbol(raw, is_assignment=True)
        assert result == "LOC728323"

    def test_single_field_accession_skipped(self, annotator):
        """If only field is an accession, return NA."""
        raw = "NM_001234"
        assert annotator.clean_symbol(raw, is_assignment=True) == "NA"

    def test_single_field_gene_kept(self, annotator):
        """If only field is a gene symbol (not accession), keep it."""
        raw = "TP53"
        assert annotator.clean_symbol(raw, is_assignment=True) == "TP53"

    def test_assignment_dash_value(self, annotator):
        """The "---" sentinel maps to NA in assignment columns too."""
        assert annotator.clean_symbol("---", is_assignment=True) == "NA"


# ============================================================================
# find_symbol_column
# ============================================================================


class TestFindSymbolColumn:
    """Tests for find_symbol_column()'s column-name detection."""

    def test_gene_symbol_column(self, annotator):
        """A "Gene Symbol" column is recognized."""
        df = pd.DataFrame({"ID": [1], "Gene Symbol": ["TP53"], "Description": ["x"]})
        assert annotator.find_symbol_column(df) == "Gene Symbol"

    def test_gene_assignment_column(self, annotator):
        """A "gene_assignment" column is recognized when no direct symbol column exists."""
        df = pd.DataFrame({"ID": [1], "gene_assignment": ["NM_1 // TP53 // x"]})
        assert annotator.find_symbol_column(df) == "gene_assignment"

    def test_priority_order(self, annotator):
        """Gene Symbol should be preferred over gene_assignment."""
        df = pd.DataFrame(
            {
                "ID": [1],
                "Gene Symbol": ["TP53"],
                "gene_assignment": ["NM_1 // TP53 // x"],
            }
        )
        assert annotator.find_symbol_column(df) == "Gene Symbol"

    def test_no_match(self, annotator):
        """A table with no recognized symbol column returns None."""
        df = pd.DataFrame({"ID": [1], "probe_sequence": ["ATCG"]})
        assert annotator.find_symbol_column(df) is None


# ============================================================================
# _build_probe_mapping — the int/str coercion regression test
# ============================================================================


class TestBuildProbeMapping:
    """Regression tests for the GPL6244 probe-ID type mismatch bug."""

    def test_int_ids_coerced_to_str(self, annotator):
        """Integer probe IDs in the GPL table are coerced to strings."""
        table = pd.DataFrame(
            {
                "ID": [7892501, 7892502, 7892503],
                "Gene Symbol": ["TP53", "BRCA1", "---"],
            }
        )
        mapping = annotator._build_probe_mapping(table, "Gene Symbol")
        assert "7892501" in mapping
        assert mapping["7892501"] == "TP53"

    def test_str_ids_preserved(self, annotator):
        """String probe IDs pass through unchanged."""
        table = pd.DataFrame(
            {
                "ID": ["200000_s_at", "200001_at"],
                "Gene Symbol": ["CD6", "IL2RG"],
            }
        )
        mapping = annotator._build_probe_mapping(table, "Gene Symbol")
        assert mapping["200000_s_at"] == "CD6"

    def test_fallback_to_index(self, annotator):
        """When no 'ID' column, use the DataFrame index."""
        table = pd.DataFrame(
            {"Gene Symbol": ["TP53", "BRCA1"]},
            index=[100, 200],
        )
        mapping = annotator._build_probe_mapping(table, "Gene Symbol")
        assert "100" in mapping
        assert mapping["100"] == "TP53"


# ============================================================================
# annotate — end-to-end
# ============================================================================


class TestAnnotate:
    """Tests for annotate()'s end-to-end probe-to-gene mapping."""

    def test_basic_annotation(self, annotator):
        """Probes map to their gene symbols, with an unrecognized symbol becoming NA."""
        table = _make_gpl_table(
            id_col=["P1", "P2", "P3"],
            gene_col=["TP53", "BRCA1", "---"],
        )
        probe_ids = pd.Index(["P1", "P2", "P3"])
        keys = annotator.annotate(table, probe_ids)
        assert keys == ["TP53", "BRCA1", "NA"]

    def test_probe_not_in_table(self, annotator):
        """Probes absent from the GPL table should map to NA."""
        table = _make_gpl_table(
            id_col=["P1"],
            gene_col=["TP53"],
        )
        probe_ids = pd.Index(["P1", "P_MISSING"])
        keys = annotator.annotate(table, probe_ids)
        assert keys == ["TP53", "NA"]

    def test_numeric_probe_ids_match(self, annotator):
        """Regression: int IDs in GPL table must match str probe IDs."""
        table = pd.DataFrame(
            {
                "ID": [7892501, 7892502],
                "Gene Symbol": ["TP53", "BRCA1"],
            }
        )
        probe_ids = pd.Index(["7892501", "7892502"])
        keys = annotator.annotate(table, probe_ids)
        assert keys == ["TP53", "BRCA1"]

    def test_assignment_column(self, annotator):
        """annotate() extracts symbols from a gene_assignment column, not just a direct one."""
        table = pd.DataFrame(
            {
                "ID": ["P1", "P2"],
                "gene_assignment": [
                    "NM_001 // OR4F4 // olfactory receptor // 1p36",
                    "---",
                ],
            }
        )
        probe_ids = pd.Index(["P1", "P2"])
        keys = annotator.annotate(table, probe_ids)
        assert keys == ["OR4F4", "NA"]

    def test_with_ensembl_validator(self, annotator):
        """An invalid gene symbol is replaced with NA when a validator rejects it."""
        table = _make_gpl_table(
            id_col=["P1", "P2", "P3"],
            gene_col=["TP53", "FAKE_GENE", "BRCA1"],
        )
        validator = FakeValidator(valid={"TP53", "BRCA1"})
        probe_ids = pd.Index(["P1", "P2", "P3"])
        keys = annotator.annotate(table, probe_ids, validator=validator)
        assert keys == ["TP53", "NA", "BRCA1"]

    def test_validator_checks_first_gene_only(self, annotator):
        """Multi-gene probes: validate only the first gene (R-parity)."""
        table = _make_gpl_table(
            id_col=["P1"],
            gene_col=["TP53 /// FAKE"],
        )
        validator = FakeValidator(valid={"TP53"})
        keys = annotator.annotate(table, pd.Index(["P1"]), validator=validator)
        # First gene TP53 is valid → keep entire entry
        assert keys == ["TP53,FAKE"]

    def test_no_symbol_column_falls_back_to_probe_ids(self, annotator):
        """With no recognizable symbol column, the probe IDs themselves are used as keys."""
        table = pd.DataFrame({"ID": ["P1"], "probe_seq": ["ATCG"]})
        keys = annotator.annotate(table, pd.Index(["P1"]))
        assert keys == ["P1"]

    def test_empty_table(self, annotator):
        """An empty GPL table with an empty probe index returns an empty key list."""
        table = pd.DataFrame({"ID": [], "Gene Symbol": []})
        keys = annotator.annotate(table, pd.Index([]))
        assert keys == []

    def test_preserves_probe_order(self, annotator):
        """Output must be aligned 1:1 with probe_ids."""
        table = _make_gpl_table(
            id_col=["P3", "P1", "P2"],
            gene_col=["C", "A", "B"],
        )
        probe_ids = pd.Index(["P2", "P3", "P1"])
        keys = annotator.annotate(table, probe_ids)
        assert keys == ["B", "C", "A"]


# ============================================================================
# Edge cases from real GPL platforms
# ============================================================================


class TestRealWorldEdgeCases:
    """Patterns observed from actual GPL96, GPL97, GPL570, GPL6244 data."""

    def test_gpl96_multi_gene_probe(self, annotator):
        """GPL96 Gene Symbol: 'IGLC1 /// IGLC2 /// IGLC7'"""
        result = annotator.clean_symbol("IGLC1 /// IGLC2 /// IGLC7", is_assignment=False)
        assert result == "IGLC1,IGLC2,IGLC7"

    def test_gpl6244_assignment_with_description(self, annotator):
        """GPL6244 gene_assignment with full fields."""
        raw = (
            "NM_001005484 // OR4F5 // olfactory receptor, family 4, "
            "subfamily F, member 5 // 1p36.33 // 79501"
        )
        assert annotator.clean_symbol(raw, is_assignment=True) == "OR4F5"

    def test_gpl6244_mitochondrial(self, annotator):
        """GPL6244-style mitochondrial gene assignment (MT-TM) is extracted correctly."""
        raw = "ENST00000387377 // MT-TM // mitochondrially encoded tRNA methionine // --- // ---"
        assert annotator.clean_symbol(raw, is_assignment=True) == "MT-TM"

    def test_gpl570_na_value(self, annotator):
        """GPL570-style literal "NA" values map to NA."""
        assert annotator.clean_symbol("NA", is_assignment=False) == "NA"

    def test_gpl6244_multiple_assignments(self, annotator):
        """Only the first /// entry should be used for assignment columns."""
        raw = (
            "NR_024437 // LOC728323 // uncharacterized LOC728323 // 2q37.3 // 728323 "
            "/// XM_006711854 // LOC101928953 // other"
        )
        result = annotator.clean_symbol(raw, is_assignment=True)
        assert result == "LOC728323"
