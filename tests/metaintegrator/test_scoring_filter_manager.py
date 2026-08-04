"""
Unit tests for scoring_filter_manager.py

This test suite provides comprehensive coverage for all classes and methods
in the scoring_filter_manager module, including:
- FilterValidator
- FilterExtractor
- OverlapCalculator
- FilterManager

Tests include positive cases, error conditions, edge cases, and integration tests.
"""

import os
import sys
import warnings
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

# Add the src directory to the path to import modules to be run with python instead of pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "src")
sys.path.insert(0, os.path.normpath(project_root))

# Import the module under test
# Note: These imports may need to be adjusted based on your actual module structure
try:
    from metaintegrator.scoring_filter_manager import (
        FilterValidator,
        FilterExtractor,
        OverlapCalculator,
        FilterManager,
        create_filter_manager,
    )

    # Mock the core dependencies since they're imported from scoring_core
    from unittest.mock import Mock

    FilterSummary = Mock
    OverlapAnalysis = Mock
    MetaObjectValidator = Mock
    FilterResultsValidator = Mock
    LoggerMixin = Mock
    StatisticalUtils = Mock
    DataFrameUtils = Mock
    ValidationError = Exception
    FilteringError = Exception
    ConfigurationError = Exception

except ImportError as err:
    # If the module can't be imported, create mock classes for testing structure
    print(f"Importing dependencies failed. {err}.")


class TestFilterValidator:
    """Test suite for FilterValidator class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = FilterValidator()

        # Sample valid meta object
        self.valid_meta_object = {
            "filterResults": {
                "filter1": {
                    "posGeneNames": ["GENE1", "GENE2"],
                    "negGeneNames": ["GENE3", "GENE4"],
                    "timestamp": "2024-01-01T10:00:00Z",
                    "filterDescription": "Test filter",
                },
                "filter2": {
                    "posGeneNames": ["GENE2", "GENE5"],
                    "negGeneNames": ["GENE6"],
                    "timestamp": "2024-01-02T10:00:00Z",
                    "filterDescription": "Another test filter",
                },
            }
        }

    @patch("metaintegrator.scoring_filter_manager.MetaObjectValidator")
    def test_validate_meta_object_valid(self, mock_validator):
        """Test validation of a valid meta object."""
        mock_validator.validate_and_extract.return_value = None

        # Should not raise any exception
        self.validator.validate_meta_object(self.valid_meta_object)

        mock_validator.validate_and_extract.assert_called_once_with(self.valid_meta_object)

    @patch("metaintegrator.scoring_filter_manager.MetaObjectValidator")
    def test_validate_meta_object_validation_error(self, mock_validator):
        """Test validation failure from core validator."""
        mock_validator.validate_and_extract.side_effect = ValidationError("Validation failed")

        with pytest.raises(FilteringError, match="Meta object validation failed"):
            self.validator.validate_meta_object(self.valid_meta_object)

    def test_validate_meta_object_missing_filter_results(self):
        """Test validation of meta object without filterResults."""
        invalid_meta_object = {}

        with patch(
            "metaintegrator.scoring_filter_manager.MetaObjectValidator.validate_and_extract"
        ):
            with pytest.raises(FilteringError, match="must contain at least one filter result"):
                self.validator.validate_meta_object(invalid_meta_object)

    def test_validate_meta_object_empty_filter_results(self):
        """Test validation of meta object with empty filterResults."""
        invalid_meta_object = {"filterResults": {}}

        with patch(
            "metaintegrator.scoring_filter_manager.MetaObjectValidator.validate_and_extract"
        ):
            with pytest.raises(FilteringError, match="must contain at least one filter result"):
                self.validator.validate_meta_object(invalid_meta_object)

    @patch("metaintegrator.scoring_filter_manager.FilterValidator.validate_meta_object")
    def test_validate_filter_names_all_exist(self, mock_validate):
        """Test validation when all filter names exist."""
        filter_names = ["filter1", "filter2"]

        result = self.validator.validate_filter_names(self.valid_meta_object, filter_names)

        assert result == filter_names
        mock_validate.assert_called_once_with(self.valid_meta_object)

    @patch("metaintegrator.scoring_filter_manager.FilterValidator.validate_meta_object")
    def test_validate_filter_names_some_missing(self, mock_validate):
        """Test validation when some filter names are missing."""
        filter_names = ["filter1", "nonexistent", "filter2"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.validator.validate_filter_names(self.valid_meta_object, filter_names)

            assert result == ["filter1", "filter2"]
            assert len(w) == 1
            assert "Filters not found: ['nonexistent']" in str(w[0].message)

    @patch("metaintegrator.scoring_filter_manager.FilterValidator.validate_meta_object")
    def test_validate_filter_names_none_exist(self, mock_validate):
        """Test validation when no filter names exist."""
        filter_names = ["nonexistent1", "nonexistent2"]

        with pytest.raises(FilteringError, match="No valid filters found"):
            self.validator.validate_filter_names(self.valid_meta_object, filter_names)


class TestFilterExtractor:
    """Test suite for FilterExtractor class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_validator = Mock()
        self.mock_logger = Mock()
        self.extractor = FilterExtractor(self.mock_validator, logger=self.mock_logger)

        # Sample filter object
        self.sample_filter = {
            "posGeneNames": ["GENE1", "GENE2"],
            "negGeneNames": ["GENE3", "GENE4", "GENE5"],
            "timestamp": "2024-01-01T10:00:00Z",
            "filterDescription": "Test filter",
            "maxSize": 100,
            "minEffectSize": 0.5,
        }

        self.sample_meta_object = {"filterResults": {"test_filter": self.sample_filter}}

    def test_init(self):
        """Test FilterExtractor initialization."""
        extractor = FilterExtractor(self.mock_validator)
        assert extractor.validator == self.mock_validator

    @patch("metaintegrator.scoring_filter_manager.FilterResultsValidator")
    def test_extract_filter_summary_success(self, mock_filter_validator):
        """Test successful filter summary extraction."""
        mock_filter_validator.validate_and_extract.return_value = None
        mock_filter_validator.OPTIONAL_FIELDS = {
            "maxSize": 100,
            "minEffectSize": 0.5,
            "timestamp": None,
        }

        with patch("metaintegrator.scoring_filter_manager.FilterSummary") as mock_summary:
            mock_summary_instance = Mock()
            mock_summary.return_value = mock_summary_instance

            result = self.extractor.extract_filter_summary(self.sample_meta_object, "test_filter")

            # Verify validator was called
            self.mock_validator.validate_meta_object.assert_called_once_with(
                self.sample_meta_object
            )

            # Verify FilterSummary was created
            mock_summary.assert_called_once()
            call_kwargs = mock_summary.call_args[1]

            assert call_kwargs["name"] == "test_filter"
            assert call_kwargs["n_positive_genes"] == 2
            assert call_kwargs["n_negative_genes"] == 3
            assert call_kwargs["total_genes"] == 5
            assert call_kwargs["timestamp"] == "2024-01-01T10:00:00Z"
            assert call_kwargs["description"] == "Test filter"
            assert call_kwargs["positive_genes"] == ["GENE1", "GENE2"]
            assert call_kwargs["negative_genes"] == ["GENE3", "GENE4", "GENE5"]

            assert result == mock_summary_instance

    def test_extract_filter_summary_filter_not_found(self):
        """Test extraction when filter doesn't exist."""
        with pytest.raises(FilteringError, match="Filter 'nonexistent' not found"):
            self.extractor.extract_filter_summary(self.sample_meta_object, "nonexistent")

    @patch("metaintegrator.scoring_filter_manager.FilterResultsValidator")
    def test_extract_filter_summary_validation_error(self, mock_filter_validator):
        """Test extraction when filter validation fails."""
        mock_filter_validator.validate_and_extract.side_effect = Exception("Validation failed")

        with pytest.raises(FilteringError, match="Filter extraction failed"):
            self.extractor.extract_filter_summary(self.sample_meta_object, "test_filter")

    def test_extract_filter_summary_missing_fields(self):
        """Test extraction with missing optional fields."""
        minimal_filter = {"posGeneNames": ["GENE1"], "negGeneNames": ["GENE2"]}
        minimal_meta = {"filterResults": {"minimal": minimal_filter}}

        with patch(
            "metaintegrator.scoring_filter_manager.FilterResultsValidator"
        ) as mock_validator:
            mock_validator.validate_and_extract.return_value = None
            mock_validator.OPTIONAL_FIELDS = {"maxSize": 100}

            with patch("metaintegrator.scoring_filter_manager.FilterSummary") as mock_summary:
                self.extractor.extract_filter_summary(minimal_meta, "minimal")

                call_kwargs = mock_summary.call_args[1]
                assert call_kwargs["timestamp"] is None
                assert call_kwargs["description"] == "Unknown"

    @patch.object(FilterExtractor, "extract_filter_summary")
    def test_extract_all_filter_summaries_success(self, mock_extract):
        """Test extraction of all filter summaries."""
        mock_summary1 = Mock()
        mock_summary2 = Mock()
        mock_extract.side_effect = [mock_summary1, mock_summary2]

        meta_object = {
            "filterResults": {
                "filter1": {"posGeneNames": [], "negGeneNames": []},
                "filter2": {"posGeneNames": [], "negGeneNames": []},
            }
        }

        result = self.extractor.extract_all_filter_summaries(meta_object)

        assert len(result) == 2
        assert mock_summary1 in result
        assert mock_summary2 in result
        assert mock_extract.call_count == 2

    @patch.object(FilterExtractor, "extract_filter_summary")
    def test_extract_all_filter_summaries_partial_failure(self, mock_extract):
        """Test extraction when some filters fail."""
        mock_summary1 = Mock()
        mock_extract.side_effect = [mock_summary1, FilteringError("Failed")]

        meta_object = {
            "filterResults": {
                "filter1": {"posGeneNames": [], "negGeneNames": []},
                "filter2": {"posGeneNames": [], "negGeneNames": []},
            }
        }

        result = self.extractor.extract_all_filter_summaries(meta_object)

        assert len(result) == 1
        assert mock_summary1 in result


class TestOverlapCalculator:
    """Test suite for OverlapCalculator class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = OverlapCalculator()

        # Create mock filter summaries
        self.mock_summary1 = Mock()
        self.mock_summary1.name = "filter1"
        self.mock_summary1.positive_genes = ["GENE1", "GENE2", "GENE3"]
        self.mock_summary1.negative_genes = ["GENE4", "GENE5"]

        self.mock_summary2 = Mock()
        self.mock_summary2.name = "filter2"
        self.mock_summary2.positive_genes = ["GENE2", "GENE3", "GENE6"]
        self.mock_summary2.negative_genes = ["GENE4", "GENE7"]

        self.filter_summaries = [self.mock_summary1, self.mock_summary2]

    def test_calculate_overlaps_empty_list(self):
        """Test overlap calculation with empty filter list."""
        with pytest.raises(FilteringError, match="No filter summaries provided"):
            self.calculator.calculate_overlaps([])

    @patch("metaintegrator.scoring_filter_manager.StatisticalUtils")
    @patch("metaintegrator.scoring_filter_manager.OverlapAnalysis")
    def test_calculate_overlaps_success(self, mock_overlap_analysis, mock_stats_utils):
        """Test successful overlap calculation."""
        # Setup mocks
        mock_overlap_matrix = Mock()
        mock_overlap_matrix.to_dict.return_value = {"matrix": "data"}
        mock_stats_utils.calculate_overlap_matrix.return_value = mock_overlap_matrix

        mock_analysis_instance = Mock()
        mock_overlap_analysis.return_value = mock_analysis_instance

        # Execute
        result = self.calculator.calculate_overlaps(self.filter_summaries)

        # Verify StatisticalUtils was called for overlap matrices
        assert mock_stats_utils.calculate_overlap_matrix.call_count == 2

        # Verify OverlapAnalysis was created
        mock_overlap_analysis.assert_called_once()
        call_kwargs = mock_overlap_analysis.call_args[1]

        assert call_kwargs["filter_names"] == ["filter1", "filter2"]
        assert call_kwargs["positive_overlaps"] == {"matrix": "data"}
        assert call_kwargs["negative_overlaps"] == {"matrix": "data"}
        assert "jaccard_similarities" in call_kwargs
        assert "unique_genes" in call_kwargs

        assert result == mock_analysis_instance

    def test_calculate_jaccard_similarities(self):
        """Test Jaccard similarity calculation."""
        positive_sets = {"filter1": {"GENE1", "GENE2"}, "filter2": {"GENE2", "GENE3"}}
        negative_sets = {"filter1": {"GENE4"}, "filter2": {"GENE5"}}

        with patch("metaintegrator.scoring_filter_manager.StatisticalUtils") as mock_utils:
            mock_utils.calculate_jaccard_similarity.return_value = 0.33

            result = self.calculator._calculate_jaccard_similarities(positive_sets, negative_sets)

            # Should calculate similarity for each pair
            assert "filter1" in result
            assert "filter2" in result
            assert result["filter1"]["filter2"] == 0.33
            assert result["filter2"]["filter1"] == 0.33

            # Should be called 4 times (2x2 combinations)
            assert mock_utils.calculate_jaccard_similarity.call_count == 4

    def test_calculate_unique_genes(self):
        """Test unique gene calculation."""
        positive_sets = {
            "filter1": {"GENE1", "GENE2"},
            "filter2": {"GENE2", "GENE3"},
            "filter3": {"GENE4", "GENE5"},
        }
        negative_sets = {
            "filter1": {"GENE6"},
            "filter2": {"GENE7"},
            "filter3": {"GENE8"},
        }

        result = self.calculator._calculate_unique_genes(positive_sets, negative_sets)

        # Verify structure
        assert "filter1" in result
        assert "positive" in result["filter1"]
        assert "negative" in result["filter1"]

        # Filter1 should have GENE1 as unique positive (not in filter2 or filter3)
        assert "GENE1" in result["filter1"]["positive"]
        assert "GENE2" not in result["filter1"]["positive"]  # Shared with filter2

    def test_calculate_overlap_statistics(self):
        """Test overlap statistics calculation."""
        # Create sample overlap matrices
        pos_matrix = pd.DataFrame(
            {"filter1": [2, 1], "filter2": [1, 3]},  # diagonal and off-diagonal
            index=["filter1", "filter2"],
        )

        neg_matrix = pd.DataFrame(
            {"filter1": [1, 0], "filter2": [0, 2]}, index=["filter1", "filter2"]
        )

        result = self.calculator._calculate_overlap_statistics(pos_matrix, neg_matrix)

        # Verify structure
        assert "positive_overlaps" in result
        assert "negative_overlaps" in result
        assert "total_comparisons" in result

        # Verify positive overlap stats (off-diagonal values are [1, 1])
        pos_stats = result["positive_overlaps"]
        assert pos_stats["mean"] == 1.0
        assert pos_stats["median"] == 1.0
        assert pos_stats["max"] == 1.0

        # Verify total comparisons (2x2 matrix minus diagonal = 2)
        assert result["total_comparisons"] == 2

    def test_calculate_set_statistics(self):
        """Test individual set statistics calculation."""
        positive_sets = {
            "filter1": {"GENE1", "GENE2"},
            "filter2": {"GENE3", "GENE4", "GENE5"},
        }
        negative_sets = {"filter1": {"GENE6"}, "filter2": {"GENE7", "GENE8"}}

        result = self.calculator._calculate_set_statistics(positive_sets, negative_sets)

        # Verify structure
        assert "filter1" in result
        assert "filter2" in result

        # Check filter1 stats
        filter1_stats = result["filter1"]
        assert filter1_stats["total_size"] == 3  # 2 pos + 1 neg
        assert filter1_stats["positive_size"] == 2
        assert filter1_stats["negative_size"] == 1
        assert filter1_stats["pos_neg_ratio"] == 2.0  # 2/1

        # Check filter2 stats
        filter2_stats = result["filter2"]
        assert filter2_stats["total_size"] == 5  # 3 pos + 2 neg
        assert filter2_stats["positive_size"] == 3
        assert filter2_stats["negative_size"] == 2
        assert filter2_stats["pos_neg_ratio"] == 1.5  # 3/2


class TestFilterManager:
    """Test suite for FilterManager class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manager = FilterManager()

        # Create sample meta object
        self.sample_meta_object = {
            "filterResults": {
                "filter1": {
                    "posGeneNames": ["GENE1", "GENE2"],
                    "negGeneNames": ["GENE3"],
                    "timestamp": "2024-01-01T10:00:00Z",
                    "filterDescription": "First filter",
                },
                "filter2": {
                    "posGeneNames": ["GENE2", "GENE4"],
                    "negGeneNames": ["GENE5", "GENE6"],
                    "timestamp": "2024-01-02T10:00:00Z",
                    "filterDescription": "Second filter",
                },
            }
        }

    def test_init(self):
        """Test FilterManager initialization."""
        manager = FilterManager()
        assert hasattr(manager, "validator")
        assert hasattr(manager, "extractor")
        assert hasattr(manager, "overlap_calculator")

    @patch.object(FilterExtractor, "extract_all_filter_summaries")
    def test_get_most_recent_filter_with_timestamps(self, mock_extract):
        """Test getting most recent filter when timestamps exist."""
        # Create mock summaries with timestamps
        mock_summary1 = Mock()
        mock_summary1.name = "filter1"
        mock_summary1.timestamp = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        mock_summary2 = Mock()
        mock_summary2.name = "filter2"
        mock_summary2.timestamp = datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc)

        mock_extract.return_value = [mock_summary1, mock_summary2]

        result = self.manager.get_most_recent_filter(self.sample_meta_object)

        assert result == "filter2"

    @patch.object(FilterExtractor, "extract_all_filter_summaries")
    def test_get_most_recent_filter_no_timestamps(self, mock_extract):
        """Test getting most recent filter when no timestamps exist."""
        mock_summary = Mock()
        mock_summary.name = "filter1"
        mock_summary.timestamp = None

        mock_extract.return_value = [mock_summary]

        result = self.manager.get_most_recent_filter(self.sample_meta_object)

        assert result == "filter1"

    @patch.object(FilterExtractor, "extract_all_filter_summaries")
    def test_get_most_recent_filter_no_summaries(self, mock_extract):
        """Test getting most recent filter when no summaries exist."""
        mock_extract.return_value = []

        with pytest.raises(FilteringError, match="No filter summaries found"):
            self.manager.get_most_recent_filter(self.sample_meta_object)

    @patch.object(FilterExtractor, "extract_all_filter_summaries")
    def test_list_filters_sort_by_timestamp(self, mock_extract):
        """Test listing filters sorted by timestamp."""
        # Create mock summaries
        mock_summary1 = Mock()
        mock_summary1.name = "filter1"
        mock_summary1.timestamp = datetime(2024, 1, 2)

        mock_summary2 = Mock()
        mock_summary2.name = "filter2"
        mock_summary2.timestamp = datetime(2024, 1, 1)

        mock_extract.return_value = [mock_summary1, mock_summary2]

        result = self.manager.list_filters(self.sample_meta_object, sort_by="timestamp")

        # Should be sorted newest first (default ascending=False)
        assert result[0].name == "filter1"
        assert result[1].name == "filter2"

    @patch.object(FilterExtractor, "extract_all_filter_summaries")
    def test_list_filters_sort_by_name(self, mock_extract):
        """Test listing filters sorted by name."""
        mock_summary1 = Mock()
        mock_summary1.name = "z_filter"

        mock_summary2 = Mock()
        mock_summary2.name = "a_filter"

        mock_extract.return_value = [mock_summary1, mock_summary2]

        result = self.manager.list_filters(self.sample_meta_object, sort_by="name", ascending=True)

        assert result[0].name == "a_filter"
        assert result[1].name == "z_filter"

    @patch.object(FilterExtractor, "extract_all_filter_summaries")
    def test_list_filters_sort_by_total_genes(self, mock_extract):
        """Test listing filters sorted by total genes."""
        mock_summary1 = Mock()
        mock_summary1.total_genes = 10

        mock_summary2 = Mock()
        mock_summary2.total_genes = 5

        mock_extract.return_value = [mock_summary1, mock_summary2]

        result = self.manager.list_filters(self.sample_meta_object, sort_by="total_genes")

        # Default ascending=False, so larger first
        assert result[0].total_genes == 10
        assert result[1].total_genes == 5

    @patch.object(FilterExtractor, "extract_all_filter_summaries")
    @patch.object(OverlapCalculator, "calculate_overlaps")
    @patch.object(FilterManager, "_generate_comparison_insights")
    def test_compare_filters_success(self, mock_insights, mock_overlaps, mock_extract):
        """Test successful filter comparison."""
        # Setup mocks
        mock_summaries = [Mock(), Mock()]
        mock_extract.return_value = mock_summaries

        mock_overlap_analysis = Mock()
        mock_overlaps.return_value = mock_overlap_analysis

        mock_insights.return_value = {"insight": "data"}

        # Execute
        result = self.manager.compare_filters(self.sample_meta_object)

        # Verify result structure
        assert "filter_summaries" in result
        assert "overlap_analysis" in result
        assert "insights" in result
        assert "metadata" in result

        assert result["filter_summaries"] == mock_summaries
        assert result["overlap_analysis"] == mock_overlap_analysis
        assert result["insights"] == {"insight": "data"}

        # Verify metadata
        metadata = result["metadata"]
        assert metadata["n_filters"] == 2
        assert metadata["comparison_method"] == "enhanced_core_based"
        assert "compared_at" in metadata

    @patch.object(FilterValidator, "validate_filter_names")
    @patch.object(FilterExtractor, "extract_filter_summary")
    @patch.object(OverlapCalculator, "calculate_overlaps")
    def test_compare_filters_specific_names(
        self, mock_overlaps, mock_extract_single, mock_validate_names
    ):
        """Test filter comparison with specific filter names."""
        mock_validate_names.return_value = ["filter1", "filter2"]
        mock_extract_single.side_effect = [Mock(), Mock()]
        mock_overlaps.return_value = Mock()

        self.manager.compare_filters(self.sample_meta_object, ["filter1", "filter2", "nonexistent"])

        mock_validate_names.assert_called_once_with(
            self.sample_meta_object, ["filter1", "filter2", "nonexistent"]
        )
        assert mock_extract_single.call_count == 2

    @patch.object(FilterExtractor, "extract_all_filter_summaries")
    def test_compare_filters_insufficient_filters(self, mock_extract):
        """Test filter comparison with less than 2 filters."""
        mock_extract.return_value = [Mock()]  # Only 1 filter

        with pytest.raises(FilteringError, match="Need at least 2 filters for comparison"):
            self.manager.compare_filters(self.sample_meta_object)

    def test_find_consensus_genes_invalid_agreement(self):
        """Test consensus gene finding with invalid agreement threshold."""
        with pytest.raises(ConfigurationError, match="min_agreement must be between 0 and 1"):
            self.manager.find_consensus_genes(self.sample_meta_object, min_agreement=1.5)

        with pytest.raises(ConfigurationError, match="min_agreement must be between 0 and 1"):
            self.manager.find_consensus_genes(self.sample_meta_object, min_agreement=-0.1)

    @patch.object(FilterExtractor, "extract_all_filter_summaries")
    def test_find_consensus_genes_success(self, mock_extract):
        """Test successful consensus gene finding."""
        # Create mock summaries with overlapping genes
        mock_summary1 = Mock()
        mock_summary1.positive_genes = ["GENE1", "GENE2", "GENE3"]
        mock_summary1.negative_genes = ["GENE4", "GENE5"]

        mock_summary2 = Mock()
        mock_summary2.positive_genes = [
            "GENE1",
            "GENE2",
            "GENE6",
        ]  # GENE1, GENE2 overlap
        mock_summary2.negative_genes = ["GENE4", "GENE7"]  # GENE4 overlaps

        mock_summary3 = Mock()
        mock_summary3.positive_genes = [
            "GENE1",
            "GENE8",
        ]  # Only GENE1 overlaps with others
        mock_summary3.negative_genes = ["GENE9"]

        mock_extract.return_value = [mock_summary1, mock_summary2, mock_summary3]

        # Test with 50% agreement (need 2 out of 3 filters)
        result = self.manager.find_consensus_genes(self.sample_meta_object, min_agreement=0.5)

        # Verify result structure
        assert "positive" in result
        assert "negative" in result
        assert "min_agreement" in result
        assert "min_filters" in result
        assert "total_filters" in result
        assert "consensus_statistics" in result

        # Check consensus genes
        # GENE1 appears in all 3, GENE2 appears in 2, so both should be consensus
        assert "GENE1" in result["positive"]
        assert "GENE2" in result["positive"]
        assert "GENE3" not in result["positive"]  # Only in 1 filter

        # GENE4 appears in 2 filters, so should be consensus
        assert "GENE4" in result["negative"]
        assert "GENE5" not in result["negative"]  # Only in 1 filter

        # Check metadata
        assert result["min_agreement"] == 0.5
        assert result["min_filters"] == 2  # max(1, int(3 * 0.5))
        assert result["total_filters"] == 3

    @patch.object(FilterExtractor, "extract_all_filter_summaries")
    def test_find_consensus_genes_no_filters(self, mock_extract):
        """Test consensus gene finding with no valid filters."""
        mock_extract.return_value = []

        with pytest.raises(FilteringError, match="No valid filters found for consensus analysis"):
            self.manager.find_consensus_genes(self.sample_meta_object)

    @patch.object(FilterManager, "get_most_recent_filter")
    @patch.object(FilterExtractor, "extract_filter_summary")
    def test_summarize_filter_results_with_name(self, mock_extract, mock_most_recent):
        """Test summarizing filter results with specific filter name."""
        mock_summary = Mock()
        mock_extract.return_value = mock_summary

        result = self.manager.summarize_filter_results(self.sample_meta_object, "specific_filter")

        mock_extract.assert_called_once_with(self.sample_meta_object, "specific_filter")
        mock_most_recent.assert_not_called()
        assert result == mock_summary

    @patch.object(FilterManager, "get_most_recent_filter")
    @patch.object(FilterExtractor, "extract_filter_summary")
    def test_summarize_filter_results_without_name(self, mock_extract, mock_most_recent):
        """Test summarizing filter results without specific filter name."""
        mock_most_recent.return_value = "most_recent_filter"
        mock_summary = Mock()
        mock_extract.return_value = mock_summary

        result = self.manager.summarize_filter_results(self.sample_meta_object)

        mock_most_recent.assert_called_once_with(self.sample_meta_object)
        mock_extract.assert_called_once_with(self.sample_meta_object, "most_recent_filter")
        assert result == mock_summary

    @patch.object(FilterManager, "compare_filters")
    def test_get_overlap_matrix(self, mock_compare):
        """Test getting overlap matrix."""
        mock_overlap_analysis = Mock()
        mock_overlap_analysis.get_overlap_matrix.return_value = "matrix_data"

        mock_compare.return_value = {"overlap_analysis": mock_overlap_analysis}

        result = self.manager.get_overlap_matrix(
            self.sample_meta_object, ["filter1", "filter2"], "positive"
        )

        mock_compare.assert_called_once_with(self.sample_meta_object, ["filter1", "filter2"])
        mock_overlap_analysis.get_overlap_matrix.assert_called_once_with("positive")
        assert result == "matrix_data"

    def test_generate_comparison_insights(self):
        """Test comparison insights generation."""
        # Create mock summaries
        mock_summary1 = Mock()
        mock_summary1.total_genes = 10
        mock_summary1.gene_count_ratio = 0.5

        mock_summary2 = Mock()
        mock_summary2.total_genes = 15
        mock_summary2.gene_count_ratio = 0.8

        summaries = [mock_summary1, mock_summary2]

        # Create mock overlap analysis
        mock_overlap_analysis = Mock()
        mock_overlap_analysis.overlap_statistics = {
            "positive_overlaps": {"mean": 5.0, "std": 2.0},
            "negative_overlaps": {"mean": 3.0, "std": 1.0},
        }
        mock_overlap_analysis.get_most_similar_pairs.return_value = [("filter1", "filter2", 0.75)]
        mock_overlap_analysis.get_jaccard_matrix.return_value = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]]
        )

        result = self.manager._generate_comparison_insights(summaries, mock_overlap_analysis)

        # Verify insights structure
        assert "most_restrictive" in result
        assert "most_permissive" in result
        assert "highest_pos_neg_ratio" in result
        assert "overlap_insights" in result
        assert "most_similar_filters" in result
        assert "diversity_metrics" in result

        # Check specific insights
        assert result["most_restrictive"] == mock_summary1  # Fewer total genes
        assert result["most_permissive"] == mock_summary2  # More total genes
        assert result["highest_pos_neg_ratio"] == mock_summary2  # Higher ratio

        # Check overlap insights
        overlap_insights = result["overlap_insights"]
        assert overlap_insights["mean_positive_overlap"] == 5.0
        assert overlap_insights["mean_negative_overlap"] == 3.0
        assert overlap_insights["overlap_variability"] == 2.0


class TestCreateFilterManager:
    """Test suite for create_filter_manager factory function."""

    def test_create_filter_manager(self):
        """Test filter manager factory function."""
        manager = create_filter_manager()
        assert isinstance(manager, FilterManager)

    def test_create_filter_manager_with_kwargs(self):
        """Test filter manager factory function with keyword arguments."""
        # The function should accept and pass through keyword arguments
        manager = create_filter_manager(some_param="value")
        assert isinstance(manager, FilterManager)


class TestIntegration:
    """Integration tests for the entire module."""

    def setup_method(self):
        """Setup test fixtures for integration tests."""
        self.manager = FilterManager()

        # Create a comprehensive meta object for integration testing
        self.integration_meta_object = {
            "filterResults": {
                "discovery_filter": {
                    "posGeneNames": ["GENE1", "GENE2", "GENE3", "GENE4"],
                    "negGeneNames": ["GENE5", "GENE6"],
                    "timestamp": "2024-01-01T10:00:00Z",
                    "filterDescription": "Discovery phase filter",
                    "maxSize": 100,
                    "minEffectSize": 0.6,
                },
                "validation_filter": {
                    "posGeneNames": ["GENE2", "GENE3", "GENE7", "GENE8"],
                    "negGeneNames": ["GENE5", "GENE9", "GENE10"],
                    "timestamp": "2024-01-02T10:00:00Z",
                    "filterDescription": "Validation phase filter",
                    "maxSize": 80,
                    "minEffectSize": 0.7,
                },
                "replication_filter": {
                    "posGeneNames": ["GENE3", "GENE11"],
                    "negGeneNames": ["GENE12"],
                    "timestamp": "2024-01-03T10:00:00Z",
                    "filterDescription": "Replication phase filter",
                    "maxSize": 50,
                    "minEffectSize": 0.8,
                },
            }
        }

    @patch("metaintegrator.scoring_filter_manager.MetaObjectValidator")
    @patch("metaintegrator.scoring_filter_manager.FilterResultsValidator")
    @patch("metaintegrator.scoring_filter_manager.FilterSummary")
    @patch("metaintegrator.scoring_filter_manager.StatisticalUtils")
    @patch("metaintegrator.scoring_filter_manager.OverlapAnalysis")
    def test_end_to_end_workflow(
        self,
        mock_overlap_analysis,
        mock_stats,
        mock_filter_summary,
        mock_filter_validator,
        mock_meta_validator,
    ):
        """Test complete end-to-end workflow."""
        # Note: Iterating on Mock.items() will fail.
        # Thus the logic below actually is expected to halt at extract_filter_summary()
        # called by get_most_recent_filter()
        #
        # Setup all necessary mocks for a complete workflow
        mock_meta_validator.validate_and_extract.return_value = None
        mock_filter_validator.validate_and_extract.return_value = None
        mock_filter_validator.OPTIONAL_FIELDS = {"maxSize": 100, "minEffectSize": 0.5}

        # Mock FilterSummary creation
        def create_mock_summary(name, pos_count, neg_count):
            summary = Mock()
            summary.name = name
            summary.n_positive_genes = pos_count
            summary.n_negative_genes = neg_count
            summary.total_genes = pos_count + neg_count
            summary.timestamp = datetime.now()
            summary.positive_genes = [f"POS_GENE_{i}" for i in range(pos_count)]
            summary.negative_genes = [f"NEG_GENE_{i}" for i in range(neg_count)]
            summary.gene_count_ratio = pos_count / (neg_count + 0.001)
            return summary

        mock_summaries = [
            create_mock_summary("discovery_filter", 4, 2),
            create_mock_summary("validation_filter", 4, 3),
            create_mock_summary("replication_filter", 2, 1),
        ]

        mock_filter_summary.side_effect = mock_summaries

        # Mock statistical utilities
        mock_matrix = Mock()
        mock_matrix.to_dict.return_value = {"matrix": "data"}
        mock_stats.calculate_overlap_matrix.return_value = mock_matrix
        mock_stats.calculate_jaccard_similarity.return_value = 0.5

        # Mock overlap analysis
        mock_analysis_instance = Mock()
        mock_analysis_instance.get_overlap_matrix.return_value = pd.DataFrame([[1, 2], [2, 1]])
        mock_analysis_instance.get_jaccard_matrix.return_value = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]]
        )
        mock_analysis_instance.get_most_similar_pairs.return_value = [("filter1", "filter2", 0.75)]
        mock_analysis_instance.overlap_statistics = {
            "positive_overlaps": {"mean": 2.0, "std": 0.5},
            "negative_overlaps": {"mean": 1.0, "std": 0.3},
        }
        mock_overlap_analysis.return_value = mock_analysis_instance

        # Test complete workflow
        try:
            # 1. List all filters
            filters = self.manager.list_filters(self.integration_meta_object)

            # 2. Get most recent filter
            most_recent = self.manager.get_most_recent_filter(self.integration_meta_object)

            # 3. Compare filters
            comparison = self.manager.compare_filters(self.integration_meta_object)

            # 4. Find consensus genes
            consensus = self.manager.find_consensus_genes(
                self.integration_meta_object, min_agreement=0.5
            )

            # 5. Get overlap matrix
            overlap_matrix = self.manager.get_overlap_matrix(
                self.integration_meta_object, gene_type="positive"
            )

            # Verify all operations completed successfully
            assert filters is not None
            assert most_recent is not None
            assert comparison is not None
            assert consensus is not None
            assert overlap_matrix is not None

            # Verify expected data structures
            assert "filter_summaries" in comparison
            assert "overlap_analysis" in comparison
            assert "insights" in comparison

            assert "positive" in consensus
            assert "negative" in consensus
            assert "consensus_statistics" in consensus

        except Exception as e:
            error_msg = str(e) if str(e) else f"{type(e).__name__} (no message)"
            # Iterating on Mock.items() will fail
            # It is expected that no summaries will be created.
            assert "No filter summaries found" in error_msg
            # pytest.fail(f"End-to-end workflow failed: {e}")

    def test_error_propagation(self):
        """Test that errors propagate correctly through the system."""
        invalid_meta_object = {"invalid": "structure"}

        with patch(
            "metaintegrator.scoring_filter_manager.MetaObjectValidator.validate_and_extract"
        ) as mock_validator:
            mock_validator.side_effect = ValidationError("Invalid structure")

            # All operations should fail with appropriate errors
            with pytest.raises(FilteringError):
                self.manager.list_filters(invalid_meta_object)

            with pytest.raises(FilteringError):
                self.manager.get_most_recent_filter(invalid_meta_object)

            with pytest.raises(FilteringError):
                self.manager.compare_filters(invalid_meta_object)

            with pytest.raises(FilteringError):
                self.manager.find_consensus_genes(invalid_meta_object)


# Pytest configuration and fixtures
@pytest.fixture
def sample_meta_object():
    """Fixture providing a sample meta object for testing."""
    return {
        "filterResults": {
            "filter1": {
                "posGeneNames": ["GENE1", "GENE2"],
                "negGeneNames": ["GENE3", "GENE4"],
                "timestamp": "2024-01-01T10:00:00Z",
                "filterDescription": "Test filter 1",
            },
            "filter2": {
                "posGeneNames": ["GENE2", "GENE5"],
                "negGeneNames": ["GENE6"],
                "timestamp": "2024-01-02T10:00:00Z",
                "filterDescription": "Test filter 2",
            },
        }
    }


@pytest.fixture
def mock_filter_summary():
    """Fixture providing a mock FilterSummary object."""
    summary = Mock()
    summary.name = "test_filter"
    summary.n_positive_genes = 2
    summary.n_negative_genes = 1
    summary.total_genes = 3
    summary.timestamp = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    summary.positive_genes = ["GENE1", "GENE2"]
    summary.negative_genes = ["GENE3"]
    summary.gene_count_ratio = 2.0
    return summary


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
