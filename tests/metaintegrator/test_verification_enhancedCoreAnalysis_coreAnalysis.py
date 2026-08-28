#!/usr/bin/env python3
"""
Comprehensive test verification script for all metaintegrator_py fixes.
Run this to verify all fixes are working correctly before running the full test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from metaintegrator.core_analysis import FisherCombination
from metaintegrator.enhanced_core_analysis import EnhancedMetaAnalysis


class Colors:
    """Terminal color codes for pretty output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def test_1_dot_separator_handling():
    """Test 1: Verify dot separator columns are recognized."""
    print_header("TEST 1: Dot Separator Column Handling")

    try:
        # Create test data with dot separators (R convention)
        study1 = pd.DataFrame(
            {
                "P.up": [0.01, 0.05, 0.001],
                "P.down": [0.9, 0.8, 0.99],
                "keys": ["GENE1", "GENE2", "GENE3"],
            }
        )

        print_info("Created test data with P.up and P.down columns")

        result = EnhancedMetaAnalysis.combine_significances([study1])

        # Verify columns were found
        assert not result["sigs_up"].empty, "sigs_up should not be empty"
        assert not result["sigs_down"].empty, "sigs_down should not be empty"
        assert result["sigs_up"].shape[1] > 0, "sigs_up should have columns"
        assert result["sigs_down"].shape[1] > 0, "sigs_down should have columns"

        print_success(
            f"Dot separators recognized: {result['sigs_up'].shape[1]} up columns, {result['sigs_down'].shape[1]} down columns"
        )
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        return False


def test_2_underscore_separator_handling():
    """Test 2: Verify underscore separator columns are recognized."""
    print_header("TEST 2: Underscore Separator Column Handling")

    try:
        # Create test data with underscore separators (Python convention)
        study1 = pd.DataFrame(
            {
                "Q_up": [0.015, 0.04, 0.001],
                "Q_down": [0.92, 0.82, 0.97],
                "keys": ["GENE1", "GENE2", "GENE3"],
            }
        )

        print_info("Created test data with Q_up and Q_down columns")

        result = EnhancedMetaAnalysis.combine_significances([study1])

        # Verify columns were found
        assert not result["sigs_up"].empty, "sigs_up should not be empty"
        assert not result["sigs_down"].empty, "sigs_down should not be empty"
        assert result["sigs_up"].shape[1] > 0, "sigs_up should have columns"
        assert result["sigs_down"].shape[1] > 0, "sigs_down should have columns"

        print_success(
            f"Underscore separators recognized: {result['sigs_up'].shape[1]} up columns, {result['sigs_down'].shape[1]} down columns"
        )
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        return False


def test_3_mixed_separator_handling():
    """Test 3: Verify mixed separators work together."""
    print_header("TEST 3: Mixed Separator Handling (Critical)")

    try:
        # Create test data with MIXED separators (like the failing test)
        study1 = pd.DataFrame(
            {
                "P.up": [0.01, 0.05, 0.001, 0.1],
                "P.down": [0.9, 0.8, 0.99, 0.7],
                "keys": ["GENE1", "GENE2", "GENE3", "GENE4"],
            }
        )

        study2 = pd.DataFrame(
            {
                "P.up": [0.02, 0.03, 0.002],
                "P.down": [0.95, 0.85, 0.98],
                "keys": ["GENE1", "GENE2", "GENE5"],
            }
        )

        study3 = pd.DataFrame(
            {
                "Q_up": [0.015, 0.04, 0.001],  # Underscore separator
                "Q_down": [0.92, 0.82, 0.97],  # Underscore separator
                "keys": ["GENE1", "GENE3", "GENE7"],
            }
        )

        print_info("Created 3 studies: 2 with dots (P.up), 1 with underscores (Q_up)")

        result = EnhancedMetaAnalysis.combine_significances([study1, study2, study3])

        # Verify all studies were included
        assert (
            result["sigs_up"].shape[1] == 3
        ), f"Should have 3 studies in sigs_up, got {result['sigs_up'].shape[1]}"
        assert (
            result["sigs_down"].shape[1] == 3
        ), f"Should have 3 studies in sigs_down, got {result['sigs_down'].shape[1]}"

        # Check GENE1 specifically (appears in all 3 studies)
        gene1_up = result["sigs_up"].loc["GENE1"]
        non_nan_count = gene1_up.notna().sum()

        print_info(f"GENE1 found in {non_nan_count} studies")
        assert non_nan_count == 3, f"GENE1 should appear in all 3 studies, found in {non_nan_count}"

        print_success(
            f"All 3 studies included: {result['sigs_up'].shape[1]} up columns, {result['sigs_down'].shape[1]} down columns"
        )
        print_success(f"GENE1 correctly found in all 3 studies ✓")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_4_fisher_combination():
    """Test 4: Verify Fisher combination works with both separators."""
    print_header("TEST 4: Fisher Combination with Mixed Separators")

    try:
        # Create significance matrices with different separators
        matrix1 = pd.DataFrame(
            {
                "P.up": [0.01, 0.05, 0.001],
                "P.down": [0.9, 0.8, 0.99],
                "keys": ["GENE1", "GENE2", "GENE3"],
            }
        )

        matrix2 = pd.DataFrame(
            {
                "P_up": [0.02, 0.03, 0.002],  # Underscore
                "P_down": [0.95, 0.85, 0.98],  # Underscore
                "keys": ["GENE1", "GENE2", "GENE4"],
            }
        )

        print_info("Created 2 matrices: 1 with dots, 1 with underscores")

        # Combine significances
        combined = EnhancedMetaAnalysis.combine_significances([matrix1, matrix2])

        # Apply Fisher's method
        fisher_result = FisherCombination._apply_fisher_vectorized(combined["sigs_up"], "up")

        print_info(f"Fisher result shape: {fisher_result.shape}")
        print_info(f"Fisher result columns: {list(fisher_result.columns)}")

        # Verify Fisher statistics exist
        assert "F.stat.up" in fisher_result.columns, "Missing F.stat.up column"
        assert "F.pval.up" in fisher_result.columns, "Missing F.pval.up column"

        # Verify we have valid statistics
        valid_stats = fisher_result["F.stat.up"].dropna()
        assert len(valid_stats) > 0, "No valid Fisher statistics"
        assert (valid_stats >= 0).all(), "Fisher statistics should be non-negative"

        valid_pvals = fisher_result["F.pval.up"].dropna()
        assert (valid_pvals >= 0).all() and (
            valid_pvals <= 1
        ).all(), "P-values should be between 0 and 1"

        print_success(f"Fisher combination successful: {len(valid_stats)} valid statistics")
        print_success(f"P-values valid: min={valid_pvals.min():.4f}, max={valid_pvals.max():.4f}")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_5_extreme_pvalues():
    """Test 5: Verify handling of extreme p-values."""
    print_header("TEST 5: Extreme P-value Handling")

    try:
        # Create matrix with very small p-values
        small_pval_matrix = pd.DataFrame(
            {
                "P.up": [1e-50, 1e-100, 1e-300, 0.05],
                "P.down": [0.9, 0.8, 0.7, 0.6],
                "keys": ["GENE1", "GENE2", "GENE3", "GENE4"],
            }
        )

        print_info("Created matrix with extreme p-values (1e-50 to 1e-300)")

        result = EnhancedMetaAnalysis.combine_significances([small_pval_matrix])

        # Values should remain finite
        sigs_up = result["sigs_up"]
        finite_count = np.isfinite(sigs_up.values).sum()

        assert finite_count > 0, "Should have finite values"

        print_success(f"Extreme p-values handled correctly: {finite_count} finite values")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print_header("Metaintegrator_py FIX VERIFICATION SUITE")
    print_info("Testing column naming fixes for Fisher combination\n")

    all_tests = [
        test_1_dot_separator_handling,
        test_2_underscore_separator_handling,
        test_3_mixed_separator_handling,
        test_4_fisher_combination,
        test_5_extreme_pvalues,
    ]

    results = []
    for i, test_func in enumerate(all_tests, 1):
        try:
            passed = test_func()
            results.append((test_func.__name__, passed))
        except Exception as e:
            print_error(f"Test {i} crashed: {e}")
            results.append((test_func.__name__, False))

    # Summary
    print_header("TEST SUMMARY")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        if passed:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")

    print()
    if passed_count == total_count:
        print_success(f"ALL {total_count} TESTS PASSED! ✓")
        print()
        print_info("You can now run the full test suite:")
        print_info("  pytest tests/metaintegrator/test_enhanced_core_analysis.py -v")
        print_info(
            "  pytest tests/metaintegrator/test_data_generators_with_enhanced_core_analysis.py -v"
        )
        return 0
    else:
        print_error(f"{passed_count}/{total_count} tests passed")
        print()
        print_info("Please review the errors above and check:")
        print_info("  1. All files have been updated correctly")
        print_info("  2. The enhanced filter in enhanced_core_analysis.py accepts both separators")
        print_info("  3. Test fixtures use consistent column naming")
        return 1


if __name__ == "__main__":
    sys.exit(main())
