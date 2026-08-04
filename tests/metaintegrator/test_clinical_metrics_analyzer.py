"""
Comprehensive Unit Tests for Clinical Metrics Analyzer - FIXED VERSION
====================================================

Tests for the missing R functions implementation:
- YoudenJAnalyzer tests
- ClinicalParametersAnalyzer tests
- Integration tests
- Edge case handling

Fixed issues:
- Updated error handling expectations to match graceful failure approach
- Fixed numpy array attribute handling
- Improved warning capture and validation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import warnings

# Import our implementations
from metaintegrator.clinical_metrics_analyzer import (
    YoudenJAnalyzer,
    ClinicalParametersAnalyzer,
    NamedArray,
    calculate_optimal_cutoff,
    print_optimal_cutoff_summary,
    calculate_multi_study_clinical_parameters,
)


class TestYoudenJAnalyzer:
    """Test suite for YoudenJAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = YoudenJAnalyzer()

        # Create sample ROC data
        self.sample_fpr = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.sample_tpr = np.array([0.0, 0.6, 0.8, 0.9, 0.95, 1.0])

        # Perfect classifier ROC
        self.perfect_fpr = np.array([0.0, 0.0, 1.0])
        self.perfect_tpr = np.array([0.0, 1.0, 1.0])

        # Random classifier ROC
        self.random_fpr = np.array([0.0, 0.5, 1.0])
        self.random_tpr = np.array([0.0, 0.5, 1.0])

    def test_calculate_optimal_cutoff_dataframe_input(self):
        """Test optimal cutoff calculation with DataFrame input."""
        # Test with standard column names
        roc_df = pd.DataFrame({"fpr": self.sample_fpr, "tpr": self.sample_tpr})
        result = self.analyzer.calculate_optimal_cutoff(roc_df, return_details=True)

        assert isinstance(result, dict)
        assert "youden_j" in result
        assert "optimal_sensitivity" in result
        assert "optimal_specificity" in result
        assert 0 <= result["youden_j"] <= 1
        assert 0 <= result["optimal_sensitivity"] <= 1
        assert 0 <= result["optimal_specificity"] <= 1

        # Test with uppercase column names
        roc_df_upper = pd.DataFrame({"FPR": self.sample_fpr, "TPR": self.sample_tpr})
        result_upper = self.analyzer.calculate_optimal_cutoff(roc_df_upper, return_details=True)
        assert result["youden_j"] == result_upper["youden_j"]

    def test_calculate_optimal_cutoff_dict_input(self):
        """Test optimal cutoff calculation with dictionary input."""
        roc_dict = {"fpr": self.sample_fpr, "tpr": self.sample_tpr}
        result = self.analyzer.calculate_optimal_cutoff(roc_dict, return_details=True)

        assert isinstance(result, dict)
        assert result["youden_j"] >= 0

        # Test with uppercase keys
        roc_dict_upper = {"FPR": self.sample_fpr, "TPR": self.sample_tpr}
        result_upper = self.analyzer.calculate_optimal_cutoff(roc_dict_upper, return_details=True)
        assert result["youden_j"] == result_upper["youden_j"]

    def test_calculate_optimal_cutoff_tuple_input(self):
        """Test optimal cutoff calculation with tuple input."""
        roc_tuple = (self.sample_fpr, self.sample_tpr)
        result = self.analyzer.calculate_optimal_cutoff(roc_tuple, return_details=True)

        assert isinstance(result, dict)
        assert result["youden_j"] >= 0

    def test_calculate_optimal_cutoff_simple_return(self):
        """Test optimal cutoff calculation with simple return (float)."""
        roc_data = pd.DataFrame({"fpr": self.sample_fpr, "tpr": self.sample_tpr})
        result = self.analyzer.calculate_optimal_cutoff(roc_data, return_details=False)

        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_youden_j_calculation_accuracy(self):
        """Test accuracy of Youden's J calculation."""
        # For known ROC curve, manually verify Youden's J
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.8, 1.0])
        roc_data = pd.DataFrame({"fpr": fpr, "tpr": tpr})

        result = self.analyzer.calculate_optimal_cutoff(roc_data, return_details=True)

        # At the middle point: sensitivity=0.8, specificity=0.5, J=0.3
        expected_j = 0.8 + 0.5 - 1  # = 0.3
        assert abs(result["youden_j"] - expected_j) < 1e-10

    def test_perfect_classifier(self):
        """Test with perfect classifier ROC curve."""
        roc_data = pd.DataFrame({"fpr": self.perfect_fpr, "tpr": self.perfect_tpr})
        result = self.analyzer.calculate_optimal_cutoff(roc_data, return_details=True)

        # Perfect classifier should have J=1 at optimal point
        assert abs(result["youden_j"] - 1.0) < 1e-10
        assert abs(result["optimal_sensitivity"] - 1.0) < 1e-10
        assert abs(result["optimal_specificity"] - 1.0) < 1e-10

    def test_random_classifier(self):
        """Test with random classifier ROC curve."""
        roc_data = pd.DataFrame({"fpr": self.random_fpr, "tpr": self.random_tpr})
        result = self.analyzer.calculate_optimal_cutoff(roc_data, return_details=True)

        # Random classifier should have J=0 at optimal point
        assert abs(result["youden_j"] - 0.0) < 1e-10

    def test_specificity_at_target_sensitivity(self):
        """Test specificity calculation at target sensitivity."""
        roc_data = pd.DataFrame({"fpr": self.sample_fpr, "tpr": self.sample_tpr})
        result = self.analyzer.calculate_optimal_cutoff(
            roc_data, sensitivity_target=0.8, return_details=True
        )

        assert "specificity_at_80pct_sensitivity" in result
        assert 0 <= result["specificity_at_80pct_sensitivity"] <= 1

        # Test different target
        result_95 = self.analyzer.calculate_optimal_cutoff(
            roc_data, sensitivity_target=0.95, return_details=True
        )
        assert "specificity_at_95pct_sensitivity" in result_95

    def test_invalid_input_formats(self):
        """Test handling of invalid input formats."""
        with pytest.raises(TypeError):
            self.analyzer.calculate_optimal_cutoff("invalid_string")

        with pytest.raises(ValueError):
            # Empty DataFrame
            self.analyzer.calculate_optimal_cutoff(pd.DataFrame())

        with pytest.raises(ValueError):
            # Dict without required keys
            self.analyzer.calculate_optimal_cutoff({"wrong_key": [1, 2, 3]})

    def test_invalid_array_dimensions(self):
        """Test handling of invalid array dimensions."""
        with pytest.raises(ValueError):
            # Mismatched array lengths
            roc_data = pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1, 2]})
            self.analyzer.calculate_optimal_cutoff(roc_data)

        with pytest.raises(ValueError):
            # Empty arrays
            roc_data = pd.DataFrame({"fpr": [], "tpr": []})
            self.analyzer.calculate_optimal_cutoff(roc_data)

        with pytest.raises(ValueError):
            # Single point
            roc_data = pd.DataFrame({"fpr": [0.5], "tpr": [0.5]})
            self.analyzer.calculate_optimal_cutoff(roc_data)

    def test_print_optimal_cutoff_summary(self):
        """Test the print summary function."""
        roc_data = pd.DataFrame({"fpr": self.sample_fpr, "tpr": self.sample_tpr})

        # Test that it doesn't raise an error
        with patch("builtins.print") as mock_print:
            self.analyzer.print_optimal_cutoff_summary(roc_data)
            assert mock_print.call_count == 2  # Should print 2 lines

    def test_edge_case_identical_points(self):
        """Test edge case with identical ROC points."""
        # All points the same
        fpr = np.array([0.5, 0.5, 0.5])
        tpr = np.array([0.5, 0.5, 0.5])
        roc_data = pd.DataFrame({"fpr": fpr, "tpr": tpr})

        result = self.analyzer.calculate_optimal_cutoff(roc_data, return_details=True)
        assert result["youden_j"] == 0.0  # J = 0.5 + 0.5 - 1 = 0

    def test_warning_for_out_of_range_values(self):
        """Test warnings for FPR/TPR values outside [0,1]."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # FPR values outside range
            roc_data = pd.DataFrame({"fpr": [-0.1, 0.5, 1.1], "tpr": [0, 0.5, 1]})
            self.analyzer.calculate_optimal_cutoff(roc_data)

            assert len(w) > 0
            assert "FPR values should be in range [0, 1]" in str(w[0].message)


class TestClinicalParametersAnalyzer:
    """Test suite for ClinicalParametersAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ClinicalParametersAnalyzer()

        # Create sample study data
        np.random.seed(42)  # For reproducible tests

        self.sample_studies = [
            {
                "name": "Study1",
                "score": np.array([0.1, 0.3, 0.6, 0.8, 0.2, 0.7, 0.9, 0.4]),
                "group": np.array([0, 0, 1, 1, 0, 1, 1, 0]),
            },
            {
                "name": "Study2",
                "score": np.array([0.2, 0.5, 0.7, 0.9, 0.1, 0.6, 0.8, 0.3]),
                "class": np.array([0, 1, 1, 1, 0, 1, 1, 0]),  # Test 'class' instead of 'group'
            },
        ]

        # Perfect separation study
        self.perfect_study = {
            "name": "Perfect",
            "score": np.array([0.1, 0.2, 0.8, 0.9]),
            "group": np.array([0, 0, 1, 1]),
        }

    def test_calculate_multi_study_parameters_basic(self):
        """Test basic multi-study clinical parameters calculation."""
        result = self.analyzer.calculate_multi_study_parameters(self.sample_studies)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 5)
        assert list(result.columns) == ["sens", "spec", "PPV", "NPV", "acc"]
        assert list(result.index) == ["Study1", "Study2"]

        # Check that all values are in valid range [0, 1]
        assert (result >= 0).all().all()
        assert (result <= 1).all().all()

    def test_calculate_multi_study_parameters_array_format(self):
        """Test multi-study parameters with array return format."""
        result = self.analyzer.calculate_multi_study_parameters(
            self.sample_studies, return_format="array"
        )

        assert isinstance(result, NamedArray)
        assert result.data.shape == (2, 5)
        assert result.row_names == ["Study1", "Study2"]
        assert result.col_names == ["sens", "spec", "PPV", "NPV", "acc"]

    def test_weighted_means_threshold_method(self):
        """Test the weighted means threshold calculation method."""
        result = self.analyzer.calculate_multi_study_parameters(
            self.sample_studies, threshold_method="weighted_means"
        )

        # Should not raise any errors and return valid results
        assert isinstance(result, pd.DataFrame)
        assert not result.isna().any().any()

    def test_youden_threshold_method(self):
        """Test Youden's J threshold calculation method."""
        result = self.analyzer.calculate_multi_study_parameters(
            self.sample_studies, threshold_method="youden"
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.isna().any().any()

    def test_closest_to_topleft_threshold_method(self):
        """Test closest to top-left threshold calculation method."""
        result = self.analyzer.calculate_multi_study_parameters(
            self.sample_studies, threshold_method="closest_to_topleft"
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.isna().any().any()

    def test_perfect_separation_study(self):
        """Test with perfect separation between classes."""
        result = self.analyzer.calculate_multi_study_parameters([self.perfect_study])

        # Perfect separation should yield perfect metrics
        assert result.loc["Perfect", "sens"] == 1.0
        assert result.loc["Perfect", "spec"] == 1.0
        assert result.loc["Perfect", "PPV"] == 1.0
        assert result.loc["Perfect", "NPV"] == 1.0
        assert result.loc["Perfect", "acc"] == 1.0

    def test_clinical_metrics_calculation(self):
        """Test individual clinical metrics calculation."""
        # Test with known confusion matrix: TP=2, TN=2, FP=1, FN=1
        metrics = self.analyzer._calculate_clinical_metrics(tp=2, tn=2, fp=1, fn=1)

        expected_sens = 2 / (2 + 1)  # TP / (TP + FN) = 2/3
        expected_spec = 2 / (2 + 1)  # TN / (TN + FP) = 2/3
        expected_ppv = 2 / (2 + 1)  # TP / (TP + FP) = 2/3
        expected_npv = 2 / (2 + 1)  # TN / (TN + FN) = 2/3
        expected_acc = (2 + 2) / 6  # (TP + TN) / Total = 4/6

        assert abs(metrics[0] - expected_sens) < 1e-10
        assert abs(metrics[1] - expected_spec) < 1e-10
        assert abs(metrics[2] - expected_ppv) < 1e-10
        assert abs(metrics[3] - expected_npv) < 1e-10
        assert abs(metrics[4] - expected_acc) < 1e-10

    def test_weighted_means_threshold_calculation(self):
        """Test the weighted means threshold calculation specifically."""
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        groups = np.array([0, 0, 1, 1])

        threshold = self.analyzer._weighted_means_threshold(scores, groups)

        # Manual calculation:
        # Group 0: mean=0.15, std=0.05
        # Group 1: mean=0.85, std=0.05
        # threshold = (0.15/0.05 + 0.85/0.05) / (1/0.05 + 1/0.05) = (3 + 17) / (20 + 20) = 0.5
        expected_threshold = 0.5
        assert abs(threshold - expected_threshold) < 1e-10

    def test_invalid_study_data_formats(self):
        """Test handling of invalid study data formats."""
        with pytest.raises(TypeError):
            # Not a list
            self.analyzer.calculate_multi_study_parameters("invalid")

        with pytest.raises(ValueError):
            # Empty list
            self.analyzer.calculate_multi_study_parameters([])

    def test_mismatched_array_lengths(self):
        """Test handling of mismatched score and group array lengths."""
        invalid_study = {
            "score": np.array([0.1, 0.2, 0.3]),
            "group": np.array([0, 1]),  # Different length
        }

        # Should handle gracefully with warning, not raise ValueError immediately
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.analyzer.calculate_multi_study_parameters([invalid_study])

            # Should warn about failed study processing
            assert len(w) > 0
            warning_messages = [str(warning.message) for warning in w]
            assert any(
                "Failed to process study" in msg or "must have same length" in msg
                for msg in warning_messages
            )

            # Result should contain NaN values for failed study
            assert isinstance(result, pd.DataFrame)
            assert result.shape == (1, 5)
            assert "Study_1_FAILED" in result.index[0]
            # Check that all values are NaN for the failed study
            assert result.isna().all().all()

    def test_insufficient_samples(self):
        """Test handling of studies with too few samples."""
        small_study = {
            "name": "SmallStudy",
            "score": np.array([0.1, 0.2]),
            "group": np.array([0, 1]),
        }

        # Should handle gracefully by adding warning and potentially working
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.analyzer.calculate_multi_study_parameters([small_study])

            # Should warn about small sample size
            assert len(w) > 0
            warning_messages = [str(warning.message) for warning in w]
            assert any("may produce unreliable results" in msg for msg in warning_messages)

            # Should still produce some result
            assert isinstance(result, pd.DataFrame)
            assert result.shape == (1, 5)

    def test_all_studies_fail(self):
        """Test handling when all studies fail processing."""
        # All studies have insufficient samples (less than 2)
        failing_studies = [
            {"score": np.array([0.1]), "group": np.array([0])},
            {"score": np.array([0.2]), "group": np.array([1])},
        ]

        # Should now return DataFrame with NaN values for all failed studies
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.analyzer.calculate_multi_study_parameters(failing_studies)

            # Should warn about failed studies
            assert len(w) > 0
            warning_messages = [str(warning.message) for warning in w]

            # Check for the warning messages (could be different specific messages)
            assert any(
                "Failed to process study" in msg or "Study needs at least" in msg
                for msg in warning_messages
            )

            # Should return DataFrame with all NaN values
            assert isinstance(result, pd.DataFrame)
            assert result.shape == (2, 5)  # Two studies, 5 metrics
            assert result.isna().all().all()  # All values should be NaN

            # Check study names indicate failure
            assert all("FAILED" in name for name in result.index)

    def test_non_binary_groups(self):
        """Test handling of non-binary group labels."""
        multi_class_study = {
            "score": np.array([0.1, 0.2, 0.3, 0.4]),
            "group": np.array([0, 1, 2, 1]),  # Three classes
        }

        # Should handle gracefully with warning, not raise ValueError immediately
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.analyzer.calculate_multi_study_parameters([multi_class_study])

            # Should warn about failed study processing
            assert len(w) > 0
            warning_messages = [str(warning.message) for warning in w]
            assert any(
                "Failed to process study" in msg or "Groups must be binary" in msg
                for msg in warning_messages
            )

            # Result should contain NaN values for failed study
            assert isinstance(result, pd.DataFrame)
            assert "Study_1_FAILED" in result.index[0]
            # Check that all values are NaN for the failed study
            assert result.isna().all().all()

    def test_group_remapping(self):
        """Test automatic remapping of non-0/1 group labels."""
        remapped_study = {
            "name": "Remapped",
            "score": np.array([0.1, 0.2, 0.8, 0.9]),
            "group": np.array([2, 2, 5, 5]),  # Will be remapped to 0, 0, 1, 1
        }

        # Should not raise error and should work correctly
        result = self.analyzer.calculate_multi_study_parameters([remapped_study])
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 5)

    def test_zero_standard_deviation_handling(self):
        """Test handling of zero standard deviation in weighted means method."""
        zero_std_study = {
            "name": "ZeroStd",
            "score": np.array([0.5, 0.5, 0.5, 0.5]),  # All same scores
            "group": np.array([0, 0, 1, 1]),
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.analyzer.calculate_multi_study_parameters([zero_std_study])

            # Should warn about zero standard deviation
            assert len(w) > 0
            warning_messages = [str(warning.message) for warning in w]
            assert any("Zero standard deviation" in msg for msg in warning_messages)

            # Should still produce valid results
            assert isinstance(result, pd.DataFrame)

    def test_precision_rounding(self):
        """Test significant digits precision rounding."""
        result = self.analyzer.calculate_multi_study_parameters(self.sample_studies, precision=2)

        # Check that values have appropriate precision
        # (Difficult to test exact rounding without knowing exact values)
        assert isinstance(result, pd.DataFrame)
        assert not result.isna().any().any()

    def test_failed_study_handling(self):
        """Test handling when some studies fail processing."""
        mixed_studies = [
            self.sample_studies[0],  # Valid study
            {"invalid": "data"},  # Invalid study - missing required keys
            self.sample_studies[1],  # Valid study
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.analyzer.calculate_multi_study_parameters(mixed_studies)

            # Should process valid studies and warn about failures
            assert isinstance(result, pd.DataFrame)
            assert result.shape[0] == 3  # Should include failed study with NaN

            # Check that valid studies processed correctly
            valid_rows = ~result.isna().all(axis=1)
            assert valid_rows.sum() == 2  # Two valid studies

            # Check that failed study has appropriate name and NaN values
            failed_rows = result.isna().all(axis=1)
            assert failed_rows.sum() == 1  # One failed study
            failed_study_name = result.index[failed_rows].values[0]
            assert "FAILED" in failed_study_name

    def test_significant_digits_function(self):
        """Test the significant digits rounding function."""
        test_array = np.array([[0.123456, 0.000123], [123.456, 0.0]])

        result = self.analyzer._apply_significant_digits(test_array, 3)

        # Test specific rounding cases
        assert abs(result[0, 0] - 0.123) < 1e-10  # 0.123456 -> 0.123
        assert abs(result[0, 1] - 0.000123) < 1e-10  # 0.000123 -> 0.000123 (already 3 sig figs)
        assert abs(result[1, 0] - 123) < 1e-10  # 123.456 -> 123
        assert result[1, 1] == 0.0  # 0.0 stays 0.0


class TestNamedArray:
    """Test suite for NamedArray class."""

    def test_named_array_creation(self):
        """Test NamedArray creation and basic functionality."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        row_names = ["Row1", "Row2"]
        col_names = ["Col1", "Col2", "Col3"]

        named_array = NamedArray(data, row_names, col_names)

        assert named_array.row_names == row_names
        assert named_array.col_names == col_names
        assert np.array_equal(named_array.data, data)
        assert named_array.shape == (2, 3)

    def test_named_array_indexing(self):
        """Test NamedArray indexing works like numpy array."""
        data = np.array([[1, 2], [3, 4]])
        named_array = NamedArray(data, ["R1", "R2"], ["C1", "C2"])

        assert named_array[0, 0] == 1
        assert named_array[1, 1] == 4
        assert np.array_equal(named_array[0, :], np.array([1, 2]))


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_roc_data = pd.DataFrame(
            {
                "fpr": np.array([0.0, 0.2, 0.4, 0.8, 1.0]),
                "tpr": np.array([0.0, 0.6, 0.8, 0.95, 1.0]),
            }
        )

        self.sample_study_data = [
            {
                "name": "Study1",
                "score": np.array([0.1, 0.3, 0.6, 0.8]),
                "group": np.array([0, 0, 1, 1]),
            }
        ]

    def test_calculate_optimal_cutoff_function(self):
        """Test the convenience function for optimal cutoff calculation."""
        result = calculate_optimal_cutoff(self.sample_roc_data, return_details=True)

        assert isinstance(result, dict)
        assert "youden_j" in result

    def test_print_optimal_cutoff_summary_function(self):
        """Test the convenience function for printing optimal cutoff summary."""
        with patch("builtins.print") as mock_print:
            print_optimal_cutoff_summary(self.sample_roc_data)
            assert mock_print.call_count == 2

    def test_calculate_multi_study_clinical_parameters_function(self):
        """Test the convenience function for multi-study clinical parameters."""
        result = calculate_multi_study_clinical_parameters(self.sample_study_data)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 5)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Simulate a complete analysis workflow

        # 1. Create realistic study data
        np.random.seed(123)
        studies = []

        for i in range(3):
            n_samples = 50 + i * 25
            # Create realistic scores with some separation
            scores_neg = np.random.beta(2, 5, n_samples // 2)  # Lower scores for negative class
            scores_pos = np.random.beta(5, 2, n_samples // 2)  # Higher scores for positive class

            scores = np.concatenate([scores_neg, scores_pos])
            groups = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

            studies.append({"name": f"Study_{i+1}", "score": scores, "group": groups})

        # 2. Calculate clinical parameters across studies
        clinical_analyzer = ClinicalParametersAnalyzer()
        clinical_results = clinical_analyzer.calculate_multi_study_parameters(
            studies, threshold_method="weighted_means"
        )

        # 3. For each study, also calculate optimal cutoff
        youden_analyzer = YoudenJAnalyzer()
        optimal_cutoffs = []

        for study in studies:
            # Generate ROC curve for this study
            from sklearn.metrics import roc_curve

            fpr, tpr, _ = roc_curve(study["group"], study["score"])
            roc_data = pd.DataFrame({"fpr": fpr, "tpr": tpr})

            cutoff_result = youden_analyzer.calculate_optimal_cutoff(roc_data, return_details=True)
            optimal_cutoffs.append(cutoff_result)

        # Verify results
        assert clinical_results.shape == (3, 5)
        assert len(optimal_cutoffs) == 3

        # All metrics should be reasonable (between 0 and 1)
        assert (clinical_results >= 0).all().all()
        assert (clinical_results <= 1).all().all()

        # Youden's J should be reasonable
        for cutoff_result in optimal_cutoffs:
            assert 0 <= cutoff_result["youden_j"] <= 1

    def test_comparison_with_r_expected_values(self):
        """Test with values that should match R's implementation."""
        # Create data that matches a known R calculation

        # Simple case where we can manually verify the calculations
        study_data = {
            "name": "TestStudy",
            "score": np.array([0.1, 0.2, 0.8, 0.9]),
            "group": np.array([0, 0, 1, 1]),
        }

        analyzer = ClinicalParametersAnalyzer()
        result = analyzer.calculate_multi_study_parameters([study_data])

        # With perfect separation, we should get perfect metrics
        # when using appropriate threshold
        assert result.loc["TestStudy", "sens"] == 1.0
        assert result.loc["TestStudy", "spec"] == 1.0
        assert result.loc["TestStudy", "acc"] == 1.0


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
